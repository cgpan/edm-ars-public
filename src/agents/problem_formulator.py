"""ProblemFormulator agent: designs a prediction research question and retrieves literature."""
from __future__ import annotations

import json
import os
import random
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any

import requests
import yaml

from src.agents.base import BaseAgent, parse_llm_json
from src.citations import normalize_doi
from src.errors import ProviderError

# Backward-compatible re-export of HSLS:09 temporal ordering
from src.dataset_adapter import HSLS09_TEMPORAL_ORDER as TEMPORAL_ORDER  # noqa: F401


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


#: How the generation-mode task names each dataset. HSLS:09 keeps the
#: exact wording every earlier prompt used.
_DATASET_LABELS: dict[str, str] = {
    "hsls09_public": "HSLS:09",
    "els_2002": "ELS:2002",
    "assistments_0910": "ASSISTments 2009-10",
    "did_els_hsls_panel": "ELS:2002 x HSLS:09 cross-cohort panel",
}


def _dataset_label(registry: dict | None, dataset_name: str | None) -> str:
    """Short name of the run's dataset for the PF task line (C1).

    The generation branch always asked for "a prediction research question
    using the HSLS:09 dataset", whatever dataset the run had loaded.
    """
    reg = registry if isinstance(registry, dict) else {}
    fallback = dataset_name if isinstance(dataset_name, str) else ""
    key = str(reg.get("name") or fallback or "")
    return (
        _DATASET_LABELS.get(key)
        or str(reg.get("full_name") or "").strip()
        or key
        or "HSLS:09"
    )


def _build_registry_var_map(registry: dict) -> dict[str, dict]:
    """Return a flat {variable_name: metadata_dict} map from all registry sections."""
    var_map: dict[str, dict] = {}
    variables = registry.get("variables", {})

    # Outcomes section (flat list)
    for var in variables.get("outcomes", []):
        if isinstance(var, dict) and "name" in var:
            var_map[var["name"]] = var

    # Predictors section (nested dict of lists keyed by category)
    predictors = variables.get("predictors", {})
    if isinstance(predictors, dict):
        for _category, var_list in predictors.items():
            if isinstance(var_list, list):
                for var in var_list:
                    if isinstance(var, dict) and "name" in var:
                        var_map[var["name"]] = var

    return var_map


def _get_tier3_exact_matches(registry: dict) -> set[str]:
    """Return exact-match variable names excluded by Tier 3 rules."""
    tier3 = registry.get("tier3_exclusion_rules", {})
    return set(tier3.get("exact_matches", []))


def _spec_one_liner(spec: dict) -> str:
    """Return a compact one-liner describing a research spec for diversity injection."""
    outcome = spec.get("outcome_variable", "?")
    rq = spec.get("research_question", "")
    n_preds = len(spec.get("predictor_set", []))
    novelty = spec.get("novelty_score_self_assessment", "?")
    summary = rq[:80] if rq else f"outcome={outcome}"
    return f"{summary} | {n_preds} predictors | novelty={novelty}"


# ---------------------------------------------------------------------------
# Citation verification helpers (3-layer system, inspired by AutoResearchClaw)
# ---------------------------------------------------------------------------

_JACCARD_THRESHOLD = 0.80
_CROSSREF_BASE_URL = "https://api.crossref.org/works"
_CROSSREF_TIMEOUT_S = 5
_CROSSREF_PROJECT_URL = "https://github.com/cgpan/edm-ars-public"


def _crossref_mailto(config: dict | None) -> str | None:
    """Contact address for Crossref's polite pool, if the operator gave one.

    ``CROSSREF_MAILTO`` in the environment wins over
    ``semantic_scholar.crossref_mailto`` in config.yaml. Crossref asks
    clients to identify themselves with a mailto; requests that do are
    routed to a more reliable pool. None when neither is set -- the
    project never invents an address.
    """
    env_value = (os.environ.get("CROSSREF_MAILTO") or "").strip()
    if env_value:
        return env_value
    s2_cfg = (config or {}).get("semantic_scholar") or {}
    value = s2_cfg.get("crossref_mailto") if isinstance(s2_cfg, dict) else None
    value = str(value).strip() if value else ""
    return value or None


def _crossref_request_args(mailto: str | None) -> tuple[dict[str, str], dict[str, str]]:
    """(extra query params, headers) for a Crossref request."""
    if mailto:
        agent = f"EDM-ARS (+{_CROSSREF_PROJECT_URL}; mailto:{mailto})"
        return {"mailto": mailto}, {"User-Agent": agent}
    return {}, {"User-Agent": f"EDM-ARS (+{_CROSSREF_PROJECT_URL})"}


def _tokenize_title(title: str) -> set[str]:
    """Lowercase word tokenization for Jaccard title similarity."""
    return set(re.sub(r"[^\w\s]", "", title.lower()).split())


def _jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


_RANK_MISSING = 10 ** 6


def _int_cfg(cfg: dict, key: str, default: int) -> int:
    """Read an int config value, falling back to ``default`` on anything odd.

    A typo in ``config.yaml`` must never abort literature retrieval — the
    stage runs first and a crash here costs the whole pipeline run.
    """
    try:
        value = cfg.get(key, default)
        return default if value is None else int(value)
    except (TypeError, ValueError):
        return default


def _retrieval_rank_or_last(paper: dict) -> int:
    """S2 relevance rank of a record; last place when absent or unusable.

    Legacy pools (and arXiv records) carry no ``retrieval_rank`` at all, and
    a pool round-tripped through JSON can carry it as a string. Both degrade
    to "least relevant" rather than raising.
    """
    rank = paper.get("retrieval_rank")
    if rank is None:
        return _RANK_MISSING
    try:
        return int(rank)
    except (TypeError, ValueError):
        return _RANK_MISSING


# ---------------------------------------------------------------------------
# OpenAlex records
# ---------------------------------------------------------------------------

#: A cap on a rebuilt abstract. A few OpenAlex "abstracts" are whole
#: sections of a paper, and every record goes into the formulator's prompt.
_OPENALEX_ABSTRACT_MAX_CHARS = 3000
_HTML_TAG = re.compile(r"<[^>]+>")
_ARXIV_DOI = re.compile(r"^10\.48550/arxiv\.(.+)$", re.IGNORECASE)
_ARXIV_ABS_URL = re.compile(r"arxiv\.org/(?:abs|pdf)/([^?#\s]+?)(?:\.pdf)?/?$", re.IGNORECASE)


def _openalex_abstract(inverted: Any) -> str:
    """The abstract text rebuilt from OpenAlex's ``abstract_inverted_index``.

    OpenAlex ships each abstract as ``{word: [positions]}``, not as text.
    """
    if not isinstance(inverted, dict) or not inverted:
        return ""
    slots: dict[int, str] = {}
    for word, positions in inverted.items():
        if not isinstance(positions, list):
            continue
        for pos in positions:
            if isinstance(pos, int) and pos >= 0:
                slots[pos] = str(word)
    text = " ".join(slots[i] for i in sorted(slots))
    return text[:_OPENALEX_ABSTRACT_MAX_CHARS]


def _openalex_publication_types(item: dict, source: dict) -> list[str]:
    """OpenAlex's work and source types in Semantic Scholar's words.

    ``citations.classify_entry`` picks @article or @inproceedings from S2's
    ``publicationTypes``. An OpenAlex "article" is a journal article only
    when its source is a journal; anything unclear stays unlabelled, so the
    entry falls back to the venue wording and the DOI instead of a guess.
    """
    work_type = str(item.get("type") or "").strip().lower()
    source_type = str(source.get("type") or "").strip().lower()
    if work_type in ("conference-paper", "proceedings-article") or source_type == "conference":
        return ["Conference"]
    if work_type == "review":
        return ["Review"]
    if work_type == "article" and source_type == "journal":
        return ["JournalArticle"]
    if work_type == "book":
        return ["Book"]
    if work_type == "book-chapter":
        return ["BookSection"]
    return []


def _openalex_arxiv_id(doi: str, landing_page_url: str) -> str:
    """The arXiv identifier of a work whose main copy is an arXiv preprint."""
    match = _ARXIV_DOI.match(doi or "")
    if match:
        return match.group(1)
    match = _ARXIV_ABS_URL.search(landing_page_url or "")
    return match.group(1) if match else ""


def _openalex_paper(item: Any, query: str, rank: int) -> dict | None:
    """One OpenAlex work in the literature pool's record shape.

    The shape is the Semantic Scholar record's (see ``_run_single_s2_query``)
    so the prompt, the citation filter and the BibTeX builder treat both
    alike. ``paperId`` is ``openalex_W…``: stable, and already a valid
    BibTeX key. None for a work without an id or a title.
    """
    if not isinstance(item, dict):
        return None
    work_id = str(item.get("id") or "").rstrip("/").rsplit("/", 1)[-1]
    title = " ".join(_HTML_TAG.sub("", str(item.get("display_name") or item.get("title") or "")).split())
    if not re.fullmatch(r"W\d+", work_id) or not title:
        return None
    location = item.get("primary_location")
    location = location if isinstance(location, dict) else {}
    source = location.get("source")
    source = source if isinstance(source, dict) else {}
    doi = normalize_doi(item.get("doi"))
    landing = str(location.get("landing_page_url") or "")
    open_access = item.get("open_access")
    authors = [
        str((a.get("author") or {}).get("display_name") or "")
        for a in (item.get("authorships") or [])
        if isinstance(a, dict) and isinstance(a.get("author"), dict)
    ]
    year = item.get("publication_year")
    cited = item.get("cited_by_count")
    record: dict[str, Any] = {
        "paperId": f"openalex_{work_id}",
        "title": title,
        "authors": [a for a in authors if a],
        "year": year if isinstance(year, int) else None,
        "abstract": _openalex_abstract(item.get("abstract_inverted_index")),
        "venue": str(source.get("display_name") or location.get("raw_source_name") or ""),
        "doi": doi,
        "citationCount": cited if isinstance(cited, int) else None,
        # OpenAlex has no influential-citation or reference counts on a
        # search result; None says "no data", as for S2 records.
        "influentialCitationCount": None,
        "referenceCount": None,
        "publicationDate": str(item.get("publication_date") or ""),
        "fieldsOfStudy": [],
        "publicationTypes": _openalex_publication_types(item, source),
        "isOpenAccess": (open_access or {}).get("is_oa") if isinstance(open_access, dict) else None,
        "url": landing if landing.startswith("http") else "",
        "matched_query": query,
        "retrieval_rank": rank,
        "source": "openalex",
    }
    arxiv_id = _openalex_arxiv_id(doi, landing)
    if arxiv_id:
        record["arxiv_id"] = arxiv_id
    return record


def _verify_paper_three_layers(
    paper: dict,
    real_ids: set[str],
    real_title_tokens: list[tuple[set[str], dict]],
    crossref_mailto: str | None = None,
) -> str:
    """Return 'VERIFIED', 'SUSPICIOUS', or 'HALLUCINATED' for a single paper.

    Layer 1: exact S2 paper ID match.
    Layer 2: CrossRef title search with Jaccard similarity ≥ 0.80. The
        request identifies the client (User-Agent) and, when configured,
        carries ``mailto`` for Crossref's polite pool (E9).
    Layer 3: Jaccard against actual S2 result titles ≥ 0.80.
    """
    # Layer 1: exact S2 ID
    if paper.get("paperId") and paper["paperId"] in real_ids:
        return "VERIFIED"

    title = paper.get("title", "")
    if not title:
        return "HALLUCINATED"

    # Layer 2: CrossRef
    try:
        extra_params, headers = _crossref_request_args(crossref_mailto)
        resp = requests.get(
            _CROSSREF_BASE_URL,
            params={"query.title": title, "rows": 1, "select": "title", **extra_params},
            headers=headers,
            timeout=_CROSSREF_TIMEOUT_S,
        )
        if resp.status_code == 200:
            items = resp.json().get("message", {}).get("items") or []
            if items:
                cr_title_list = items[0].get("title") or []
                cr_title = cr_title_list[0] if cr_title_list else ""
                if cr_title and _jaccard_similarity(
                    _tokenize_title(title), _tokenize_title(cr_title)
                ) >= _JACCARD_THRESHOLD:
                    return "SUSPICIOUS"
    except Exception:
        pass  # CrossRef unavailable — proceed to Layer 3

    # Layer 3: Jaccard against actual S2 result titles
    paper_tokens = _tokenize_title(title)
    for real_tokens, _ in real_title_tokens:
        if _jaccard_similarity(paper_tokens, real_tokens) >= _JACCARD_THRESHOLD:
            return "SUSPICIOUS"

    return "HALLUCINATED"


class ProblemFormulator(BaseAgent):
    """Designs a prediction research question using HSLS:09 and Semantic Scholar literature."""

    # Per-search bookkeeping for retrieval_status, reset by _search_literature.
    _s2_query_outcomes: list[str] | None = None
    _last_s2_outcome: str | None = None
    _arxiv_query_outcomes: list[str] | None = None
    _arxiv_http_status: int | None = None
    _openalex_query_outcomes: list[str] | None = None
    _openalex_http_status: int | None = None
    _openalex_plan: str | None = None
    #: The search words of the literature search in progress, by prompt;
    #: None outside ``_search_literature``.
    _lit_query_memo: dict[str | None, list[str]] | None = None

    def run(
        self,
        user_prompt: str | None = None,
        revision_instructions: str | None = None,
        findings_memory_summary: str = "",
        n_candidate_specs: int = 1,
        studied_outcomes: list[str] | None = None,
        locked_research_spec: dict | None = None,
        **kwargs: Any,
    ) -> dict:
        """
        Args:
            user_prompt: Optional free-text research direction from the user.
            revision_instructions: Critic feedback from a prior review cycle.
            findings_memory_summary: Summary of prior runs from FindingsMemory.
            n_candidate_specs: Number of candidate specs to generate (1 = current behavior).
            studied_outcomes: Outcome variables already studied in prior runs.
            locked_research_spec: Phase 3b.5 / wiring unblock. When non-None,
                threaded into _build_user_message so the causal_soo PF prompt
                can refine the locked spec rather than complaining there's
                no spec to refine.

        Returns:
            dict with keys ``research_spec`` and ``literature_context``.
        """
        registry = self.load_registry()
        task_template_data = self.load_task_template()

        # Fetch literature BEFORE calling the LLM (S2 + arXiv merged)
        s2_context = self._search_literature(user_prompt)

        if n_candidate_specs > 1:
            return self._run_multi_branch(
                user_prompt=user_prompt,
                registry=registry,
                task_template_data=task_template_data,
                s2_context=s2_context,
                findings_memory_summary=findings_memory_summary,
                n_candidate_specs=n_candidate_specs,
                studied_outcomes=studied_outcomes or [],
                locked_research_spec=locked_research_spec,
            )

        return self._run_single(
            user_prompt=user_prompt,
            revision_instructions=revision_instructions,
            registry=registry,
            task_template_data=task_template_data,
            s2_context=s2_context,
            findings_memory_summary=findings_memory_summary,
            studied_outcomes=studied_outcomes or [],
            locked_research_spec=locked_research_spec,
        )

    def _run_single(
        self,
        user_prompt: str | None,
        revision_instructions: str | None,
        registry: dict,
        task_template_data: dict,
        s2_context: dict,
        findings_memory_summary: str = "",
        studied_outcomes: list[str] | None = None,
        locked_research_spec: dict | None = None,
    ) -> dict:
        """Single-branch generation — the original behavior."""
        user_message = self._build_user_message(
            registry=registry,
            task_template=task_template_data,
            s2_context=s2_context,
            user_prompt=user_prompt,
            revision_instructions=revision_instructions,
            findings_memory_summary=findings_memory_summary,
            prior_specs=[],
            studied_outcomes=studied_outcomes or [],
            locked_research_spec=locked_research_spec,
        )

        llm_response = self.call_llm(user_message)
        parsed = parse_llm_json(llm_response)

        research_spec = parsed.get("research_spec") or {}
        literature_context = parsed.get("literature_context") or s2_context
        literature_context = self._filter_hallucinated_papers(literature_context, s2_context)
        literature_context = self._with_retrieval_status(literature_context, s2_context)

        self._log_validation_warnings(research_spec, registry)

        return {
            "research_spec": research_spec,
            "literature_context": literature_context,
            # Arc P3: keep the FULL retrieved pool, not just the 8-12 the
            # model echoed. The Writer tops the reference list back up
            # from this to reach venue citation norms; discarding it was
            # the reason manuscripts carried 4-26 references.
            "retrieved_literature": s2_context,
        }

    def _run_multi_branch(
        self,
        user_prompt: str | None,
        registry: dict,
        task_template_data: dict,
        s2_context: dict,
        findings_memory_summary: str,
        n_candidate_specs: int,
        studied_outcomes: list[str],
        locked_research_spec: dict | None = None,
    ) -> dict:
        """N-branch generation: generate N candidate specs and select the best."""
        candidates: list[dict] = []
        literature_contexts: list[dict] = []
        prior_specs: list[str] = []

        for i in range(n_candidate_specs):
            user_message = self._build_user_message(
                registry=registry,
                task_template=task_template_data,
                s2_context=s2_context,
                user_prompt=user_prompt,
                revision_instructions=None,
                findings_memory_summary=findings_memory_summary,
                prior_specs=prior_specs,
                studied_outcomes=studied_outcomes,
                locked_research_spec=locked_research_spec,
            )
            # Increasing temperature for diversity: 0.7 → 0.85 → 1.0
            temp_override = min(0.7 + i * 0.15, 1.0)
            llm_response = self.call_llm(user_message, temperature_override=temp_override)

            try:
                parsed = parse_llm_json(llm_response)
            except (ValueError, json.JSONDecodeError):
                self.ctx.log.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "agent": self.agent_name,
                    "message": f"Multi-branch: failed to parse candidate {i + 1}; skipping.",
                })
                continue

            spec = parsed.get("research_spec") or {}
            lit = self._filter_hallucinated_papers(
                parsed.get("literature_context") or s2_context, s2_context
            )
            lit = self._with_retrieval_status(lit, s2_context)
            candidates.append(spec)
            literature_contexts.append(lit)
            prior_specs.append(_spec_one_liner(spec))

        if not candidates:
            # All branches failed — fall back to single-branch
            return self._run_single(
                user_prompt=user_prompt,
                revision_instructions=None,
                registry=registry,
                task_template_data=task_template_data,
                s2_context=s2_context,
                findings_memory_summary=findings_memory_summary,
                studied_outcomes=studied_outcomes,
            )

        best_idx = self._select_best_candidate(candidates, registry, studied_outcomes, user_prompt)
        best_spec = candidates[best_idx]
        best_lit = literature_contexts[best_idx]

        self._log_validation_warnings(best_spec, registry)
        self.ctx.log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "agent": self.agent_name,
            "message": (
                f"Multi-branch: generated {len(candidates)} candidates, "
                f"selected candidate {best_idx + 1} "
                f"(outcome={best_spec.get('outcome_variable', '?')})."
            ),
        })

        return {
            "research_spec": best_spec,
            "literature_context": best_lit,
            # Arc P3: full retrieved pool (see _run_single).
            "retrieved_literature": s2_context,
        }

    @staticmethod
    def _with_retrieval_status(literature_context: dict, s2_context: dict) -> dict:
        """Carry the search's ``retrieval_status`` onto the context the
        model returned, which is what the orchestrator stores."""
        status = (s2_context or {}).get("retrieval_status")
        if status is None or not isinstance(literature_context, dict):
            return literature_context
        return {**literature_context, "retrieval_status": dict(status)}

    def _select_best_candidate(
        self,
        candidates: list[dict],
        registry: dict,
        studied_outcomes: list[str],
        user_prompt: str | None = None,
    ) -> int:
        """Rule-based scoring; returns index of best candidate (no extra LLM call)."""
        # Extract any HSLS variable names explicitly mentioned in the user prompt
        explicit_outcomes: set[str] = set()
        if user_prompt:
            explicit_outcomes = set(re.findall(r"\b[A-Z][0-9][A-Z0-9_]+\b", user_prompt))

        scores: list[float] = []
        for spec in candidates:
            score = 0.0
            outcome = spec.get("outcome_variable", "")

            # User explicitly named this outcome → honour their intent over diversity incentive
            if outcome and outcome in explicit_outcomes:
                score += 10.0

            # Prefer unstudied outcomes (only applies when user hasn't specified explicitly)
            elif outcome and outcome not in studied_outcomes:
                score += 2.0

            # Reward higher novelty score
            novelty = spec.get("novelty_score_self_assessment", 3)
            if isinstance(novelty, (int, float)) and novelty > 3:
                score += float(novelty - 3)

            # Penalise temporal violations
            warnings = self.task_template.validate_research_spec(
                spec, registry, self.dataset_adapter
            )
            if any("TEMPORAL VIOLATION" in w for w in warnings):
                score -= 10.0

            scores.append(score)

        return scores.index(max(scores))

    # ------------------------------------------------------------------
    # Semantic Scholar API
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Semantic Scholar helpers
    # ------------------------------------------------------------------

    _DEFAULT_S2_QUERIES: list[str] = [
        "educational data mining prediction student achievement high school",
        "machine learning student outcome prediction longitudinal survey",
        "postsecondary access college enrollment academic predictors",
    ]

    def _generate_search_queries(self, user_prompt: str | None) -> list[str]:
        """Use a lightweight LLM call to convert a user prompt into 3 short S2 keyword queries.

        Each query must be 4–8 words — the sweet spot for the S2 full-text search endpoint.
        Falls back to ``_DEFAULT_S2_QUERIES`` on any failure.
        """
        if not user_prompt:
            return self._DEFAULT_S2_QUERIES

        try:
            instruction = (
                "You are a research librarian helping search Semantic Scholar.\n"
                "Given the research topic below, produce EXACTLY 3 short keyword search queries "
                "suitable for the Semantic Scholar API.\n\n"
                "Rules:\n"
                "- Each query must be 4–8 words (no more).\n"
                "- Use general educational/ML terms — do NOT include dataset identifiers "
                "(e.g. 'HSLS:09', 'NCES'), variable names (e.g. 'X4EVRATNDCLG'), "
                "or year ranges.\n"
                "- Queries should cover: (1) the prediction outcome, "
                "(2) the methodology, (3) the broader domain.\n"
                "- Return ONLY a JSON array of 3 strings, no other text.\n\n"
                f"Research topic: {user_prompt}"
            )
            response = self.call_llm(instruction, max_tokens=512)
            # Strip any markdown fences and parse
            cleaned = re.sub(r"^```(?:json)?\s*", "", response.strip(), flags=re.MULTILINE)
            cleaned = re.sub(r"\s*```\s*$", "", cleaned.strip(), flags=re.MULTILINE)
            queries = json.loads(cleaned)
            if isinstance(queries, list) and len(queries) >= 1:
                valid = [str(q).strip() for q in queries if str(q).strip()][:3]
                if valid:
                    self.ctx.log.append({
                        "timestamp": datetime.utcnow().isoformat(),
                        "agent": self.agent_name,
                        "message": f"S2 keyword queries generated: {valid}",
                    })
                    return valid
        except ProviderError:
            # A rejected key, an empty account or a provider that stayed
            # unreachable through every retry is not a reason to fall back
            # to default queries: the next call would fail the same way,
            # after the literature search and a second round of waits.
            raise
        except Exception as exc:  # noqa: BLE001
            self.ctx.log.append({
                "timestamp": datetime.utcnow().isoformat(),
                "agent": self.agent_name,
                "message": f"Query generation failed ({exc}); using defaults.",
            })
        return self._DEFAULT_S2_QUERIES

    def _literature_queries(self, user_prompt: str | None) -> list[str]:
        """The search words for every source of the current search.

        Semantic Scholar and arXiv each asked the model for their own
        words: two calls per run where one does, at the formulator's
        temperature (0.7), so the two sources could search for different
        things. Within ``_search_literature`` the words are asked for
        once; a direct call outside it asks as before.
        """
        memo = self._lit_query_memo
        if memo is None:
            return self._generate_search_queries(user_prompt)
        if user_prompt not in memo:
            memo[user_prompt] = list(self._generate_search_queries(user_prompt))
        return list(memo[user_prompt])

    # Arc P5 (F-P5-DEPTH-RECENCY-SKEW): the ranking signals are requested
    # here AND hand-mapped in the comprehension below. Adding a name to
    # this string without extending the mapping is a silent no-op.
    # All of these are free on /paper/search: no extra quota, no extra
    # request, only a larger response payload. `tldr` is deliberately
    # excluded (model-generated, sparse, duplicates `abstract`).
    _S2_FIELDS: str = (
        "paperId,title,authors,year,abstract,venue,externalIds,"
        "citationCount,influentialCitationCount,referenceCount,"
        "publicationDate,fieldsOfStudy,publicationTypes,isOpenAccess"
    )

    # Tag stamped on records retrieved by the unwindowed seminal-work
    # query so the trim (and any downstream ranker) can tell them apart
    # from the topical queries' records.
    _SEMINAL_QUERY_TAG: str = "__seminal__"

    def _run_single_s2_query(
        self,
        query: str,
        base_url: str,
        limit: int,
        headers: dict[str, str],
        max_retries: int,
        backoff_base: float,
        backoff_factor: float,
        use_jitter: bool,
        delay_s: float,
        year_range: str | None = None,
        min_citation_count: int | None = None,
    ) -> list[dict]:
        """Execute one S2 search query with retry/backoff. Returns list of paper dicts.

        Args:
            year_range: S2 ``year`` filter value, or ``None`` to send no
                year filter at all. S2 accepts ``"2019"``, ``"2016-2020"``
                (inclusive), ``"2010-"`` (open forward) and ``"-2015"``
                (open backward). ``None`` is what makes seminal work
                reachable — a rolling "last N years" window can never
                return it (F-P5-DEPTH-RECENCY-SKEW).
            min_citation_count: server-side ``minCitationCount`` filter,
                or ``None`` to omit. Used by the seminal-work query to
                force a high-influence slice into the pool independent of
                where relevance ranking happened to place it.
        """
        last_exc: Exception | None = None
        last_http: int | None = None
        retryable = True
        self._last_s2_outcome = "failed"

        for attempt in range(max_retries + 1):
            try:
                if attempt == 0:
                    time.sleep(delay_s)
                else:
                    delay = backoff_base * (backoff_factor ** (attempt - 1))
                    if use_jitter:
                        delay *= random.uniform(0.75, 1.25)
                    self.ctx.log.append({
                        "timestamp": datetime.utcnow().isoformat(),
                        "agent": self.agent_name,
                        "message": f"S2 retry {attempt}/{max_retries} for '{query[:50]}' after {delay:.1f}s",
                    })
                    time.sleep(delay)

                params: dict[str, Any] = {
                    "query": query,
                    # Arc P3: `venue` is REQUIRED for honest BibTeX.
                    # Without it every non-arXiv entry fell back to a
                    # fabricated EDM-proceedings booktitle.
                    "fields": self._S2_FIELDS,
                    "limit": limit,
                }
                if year_range:
                    params["year"] = year_range
                if min_citation_count:
                    params["minCitationCount"] = min_citation_count

                resp = requests.get(
                    f"{base_url}/paper/search",
                    params=params,
                    headers=headers,
                    timeout=15,
                )
                last_http = resp.status_code if isinstance(resp.status_code, int) else None

                if 400 <= resp.status_code < 500 and resp.status_code != 429:
                    retryable = False
                    if resp.status_code == 403:
                        raise requests.RequestException(
                            "S2 API HTTP 403 — set SEMANTIC_SCHOLAR_API_KEY in environment."
                        )
                    raise requests.RequestException(
                        f"S2 API HTTP {resp.status_code} (non-retryable)"
                    )

                if resp.status_code == 429 or resp.status_code >= 500:
                    raise requests.RequestException(
                        f"S2 API HTTP {resp.status_code} (retryable)"
                    )

                data = resp.json()
                self._last_s2_outcome = "ok"
                return [
                    {
                        "paperId": item.get("paperId", ""),
                        "title": item.get("title", ""),
                        "authors": [a.get("name", "") for a in item.get("authors", [])],
                        "year": item.get("year"),
                        "abstract": item.get("abstract") or "",
                        "venue": item.get("venue") or "",
                        "doi": (item.get("externalIds") or {}).get("DOI") or "",
                        # Arc P5 — ranking signals. None (not 0) when S2
                        # omits them, so a ranker can tell "no data" from
                        # "genuinely uncited"; a 0 default would rank every
                        # record S2 has no counts for dead last.
                        "citationCount": item.get("citationCount"),
                        "influentialCitationCount": item.get("influentialCitationCount"),
                        "referenceCount": item.get("referenceCount"),
                        "publicationDate": item.get("publicationDate") or "",
                        "fieldsOfStudy": item.get("fieldsOfStudy") or [],
                        # Consumed by citations.classify_entry() to pick
                        # @article vs @inproceedings from metadata instead
                        # of guessing from the venue string.
                        "publicationTypes": item.get("publicationTypes") or [],
                        "isOpenAccess": item.get("isOpenAccess"),
                        # Provenance: S2 returns RELEVANCE order and we must
                        # not lose it again (F-P5-DEPTH-RECENCY-SKEW).
                        "matched_query": query,
                        "retrieval_rank": rank,
                        "source": "s2",
                    }
                    for rank, item in enumerate(data.get("data", []))
                    if item.get("paperId")
                ]

            except (requests.ConnectionError, requests.Timeout, requests.RequestException) as exc:
                last_exc = exc
                if not retryable or attempt == max_retries:
                    break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                break

        self._last_s2_outcome = "rate_limited" if last_http == 429 else "failed"
        self.ctx.log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "agent": self.agent_name,
            "message": f"S2 query '{query[:50]}' failed after all retries: {last_exc}",
        })
        return []

    @staticmethod
    def _build_year_range(
        year_filter: Any,
        year_floor: Any,
        current_year: int,
    ) -> str | None:
        """Return the S2 ``year`` param value for the topical queries.

        ``year_filter`` is the legacy rolling window in years. A rolling
        window is exactly what made foundational work unreachable
        (F-P5-DEPTH-RECENCY-SKEW): with ``year_filter: 10`` the request
        carried ``year=2016-2026``, so no client-side re-ranking could
        ever surface a 1983 paper — the record was excluded by the
        request itself.

        Semantics:
            ``year_filter`` > 0        -> ``"<current-N>-<current>"`` (legacy)
            ``year_filter`` null/0     -> ``"<year_floor>-<current>"``
            ...and ``year_floor`` null -> ``None`` (no year param at all)
        """
        try:
            window = int(year_filter) if year_filter is not None else 0
        except (TypeError, ValueError):
            window = 0
        if window > 0:
            return f"{current_year - window}-{current_year}"
        if year_floor is None:
            return None
        try:
            floor = int(year_floor)
        except (TypeError, ValueError):
            return None
        return f"{floor}-{current_year}"

    @classmethod
    def _trim_pool_preserving_seminal(
        cls,
        papers: list[dict],
        max_results: int,
        reserve: int,
    ) -> list[dict]:
        """Trim to ``max_results`` without discarding the seminal records.

        The pool is trimmed off a year-descending sort, so the records the
        seminal query exists to retrieve — the oldest ones — are the first
        casualties. Retrieving them and then trimming them away would make
        the whole retrieval change a no-op. Up to ``reserve`` seminal
        records displace the *least relevant* non-seminal records (worst
        ``retrieval_rank`` first) rather than the newest ones.
        """
        if max_results <= 0:
            return []
        if len(papers) <= max_results or reserve <= 0:
            return papers[:max_results]

        kept = list(papers[:max_results])
        dropped_seminal = [
            p for p in papers[max_results:]
            if p.get("matched_query") == cls._SEMINAL_QUERY_TAG
        ]
        if not dropped_seminal:
            return kept

        n_kept_seminal = sum(
            1 for p in kept if p.get("matched_query") == cls._SEMINAL_QUERY_TAG
        )
        promote = dropped_seminal[: max(0, reserve - n_kept_seminal)]
        if not promote:
            return kept

        # Eviction order: worst retrieval_rank first, then latest position.
        evictable = [
            (i, p) for i, p in enumerate(kept)
            if p.get("matched_query") != cls._SEMINAL_QUERY_TAG
        ]
        evictable.sort(key=lambda ip: (-_retrieval_rank_or_last(ip[1]), -ip[0]))
        for paper in promote:
            if not evictable:
                break
            idx, _ = evictable.pop(0)
            kept[idx] = paper

        kept.sort(key=lambda p: p.get("year") or 0, reverse=True)
        return kept

    def _run_seminal_s2_query(
        self,
        query: str,
        sem_cfg: dict,
        base_url: str,
        headers: dict[str, str],
        max_retries: int,
        backoff_base: float,
        backoff_factor: float,
        use_jitter: bool,
        delay_s: float,
    ) -> list[dict]:
        """One extra, deliberately UNWINDOWED S2 request for seminal work.

        ``minCitationCount`` is a server-side filter, so this forces a
        high-influence slice into the pool independent of where relevance
        ranking happened to place it. Failure is non-fatal by contract:
        the caller keeps whatever the topical queries returned.

        Cost: +1 request and ~+1.5 s (1.0 s inter-query pause + the
        existing ``request_delay_s``) per run. No added quota.
        """
        try:
            time.sleep(1.0)
            papers = self._run_single_s2_query(
                query=query,
                base_url=base_url,
                limit=int(sem_cfg.get("limit", 20)),
                headers=headers,
                max_retries=max_retries,
                backoff_base=backoff_base,
                backoff_factor=backoff_factor,
                use_jitter=use_jitter,
                delay_s=delay_s,
                year_range=None,  # explicitly unwindowed — the whole point
                min_citation_count=int(sem_cfg.get("min_citations", 50)),
            )
        except Exception as exc:  # noqa: BLE001 — must never abort the search
            self.ctx.log.append({
                "timestamp": datetime.utcnow().isoformat(),
                "agent": self.agent_name,
                "message": f"S2 seminal query failed (non-fatal): {exc}",
            })
            return []
        for paper in papers:
            paper["matched_query"] = self._SEMINAL_QUERY_TAG
        return papers

    def _search_semantic_scholar(self, user_prompt: str | None) -> dict:
        """Query S2 with multiple short keyword queries and return merged results.

        Inspired by AutoResearchClaw's ``search_papers_multi_query`` pattern:
        run 2–3 focused keyword queries, merge by paperId, deduplicate, and sort
        by year descending.  This avoids the zero-result problem caused by passing
        long natural-language prompts or dataset-specific identifiers to the S2
        full-text search endpoint.

        Arc P5: the topical queries no longer carry a rolling recency floor,
        and one extra unwindowed high-citation query runs after them, so
        foundational work is reachable at all (F-P5-DEPTH-RECENCY-SKEW).
        Cost: 4 S2 requests per run instead of 3, ~+1.5 s, no added quota.
        """
        cfg = self.config.get("semantic_scholar", {})
        base_url = cfg.get("base_url", "https://api.semanticscholar.org/graph/v1")
        max_results = int(cfg.get("max_results", 10))
        delay_s = float(cfg.get("request_delay_s", 0.5))
        max_retries = int(cfg.get("max_retries", 3))
        backoff_base = float(cfg.get("backoff_base_s", 1.0))
        backoff_factor = float(cfg.get("backoff_factor", 2.0))
        use_jitter = bool(cfg.get("backoff_jitter", True))
        sem_cfg = cfg.get("seminal_query") or {}

        current_year = datetime.utcnow().year
        year_range = self._build_year_range(
            cfg.get("year_filter", 10), cfg.get("year_floor", 1900), current_year
        )

        s2_api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        headers: dict[str, str] = {}
        if s2_api_key:
            headers["X-API-KEY"] = s2_api_key

        # Generate short keyword queries from user_prompt via lightweight LLM call
        queries = self._literature_queries(user_prompt)
        per_query_limit = max(max_results, 10)  # fetch at least 10 per query before dedup

        # Run all queries and merge by paperId (dedup)
        seen_ids: set[str] = set()
        merged_papers: list[dict] = []
        # One outcome per topical query ("ok" | "rate_limited" | "failed"),
        # read back by _search_literature for retrieval_status.
        topical_outcomes: list[str] = []
        self._s2_query_outcomes = topical_outcomes

        for i, query in enumerate(queries):
            if i > 0:
                # Brief inter-query delay (mirrors AutoResearchClaw's 1.5s inter-query pause)
                time.sleep(1.0)
            papers = self._run_single_s2_query(
                query=query,
                base_url=base_url,
                limit=per_query_limit,
                headers=headers,
                max_retries=max_retries,
                backoff_base=backoff_base,
                backoff_factor=backoff_factor,
                use_jitter=use_jitter,
                delay_s=delay_s,
                year_range=year_range,
            )
            for paper in papers:
                pid = paper.get("paperId", "")
                if pid and pid not in seen_ids:
                    seen_ids.add(pid)
                    merged_papers.append(paper)
            outcome = self._take_s2_outcome(papers)
            topical_outcomes.append(outcome)
            self._lit_progress(
                "semantic_scholar", i + 1, len(queries), len(papers), outcome
            )
            self.ctx.log.append({
                "timestamp": datetime.utcnow().isoformat(),
                "agent": self.agent_name,
                "message": (
                    f"S2 query {i+1}/{len(queries)} '{query[:60]}': "
                    f"{len(papers)} results, {len(merged_papers)} unique total"
                ),
            })

        # Arc P5 (F-P5-DEPTH-RECENCY-SKEW): one extra UNWINDOWED request so
        # foundational work is reachable at all. Skipped when the topical
        # queries produced a degenerate pool — that means S2 is failing or
        # the queries are unusable, and spending another request would only
        # compound 429 exposure for a pool that cannot be used anyway.
        min_primary_pool = _int_cfg(sem_cfg, "min_primary_pool", 3)
        if (
            bool(sem_cfg.get("enabled", True))
            and queries
            and len(merged_papers) >= min_primary_pool
        ):
            seminal = self._run_seminal_s2_query(
                query=queries[0],
                sem_cfg=sem_cfg,
                base_url=base_url,
                headers=headers,
                max_retries=max_retries,
                backoff_base=backoff_base,
                backoff_factor=backoff_factor,
                use_jitter=use_jitter,
                delay_s=delay_s,
            )
            n_new = 0
            for paper in seminal:
                pid = paper.get("paperId", "")
                if pid and pid not in seen_ids:
                    seen_ids.add(pid)
                    merged_papers.append(paper)
                    n_new += 1
            self._lit_progress(
                "semantic_scholar_seminal", 1, 1, len(seminal),
                self._take_s2_outcome(seminal),
            )
            self.ctx.log.append({
                "timestamp": datetime.utcnow().isoformat(),
                "agent": self.agent_name,
                "message": (
                    f"S2 seminal query (no year window, minCitationCount="
                    f"{sem_cfg.get('min_citations', 50)}): {len(seminal)} results, "
                    f"{n_new} new, {len(merged_papers)} unique total"
                ),
            })

        # Sort by year descending, trim to max_results — but never trim away
        # the seminal records, which by construction sort last.
        merged_papers.sort(key=lambda p: p.get("year") or 0, reverse=True)
        final_papers = self._trim_pool_preserving_seminal(
            merged_papers,
            max_results,
            _int_cfg(sem_cfg, "reserved_pool_slots", _int_cfg(sem_cfg, "limit", 20)),
        )

        if not final_papers:
            return {
                "search_query": queries[0],
                "papers": [],
                "novelty_evidence": (
                    "Semantic Scholar returned no results for any search query. "
                    "Citations will be placeholders."
                ),
            }

        return {
            "search_query": queries[0],  # primary query for reference
            "papers": final_papers,
            "novelty_evidence": "",
        }

    # ------------------------------------------------------------------
    # arXiv search
    # ------------------------------------------------------------------

    # https directly: the http address answers 301, so every query cost
    # arXiv two requests.
    _ARXIV_API_URL = "https://export.arxiv.org/api/query"
    _ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}
    #: arXiv's API terms: "no more than one request every three seconds".
    _ARXIV_DELAY_S = 3.0
    _ARXIV_HEADERS = {
        "User-Agent": f"EDM-ARS (+{_CROSSREF_PROJECT_URL})",
        "Accept": "application/atom+xml, application/xml;q=0.9, */*;q=0.8",
    }
    #: Statuses with which arXiv turns a client away rather than failing.
    #: Its front end answers 406 with an empty body, without reaching the
    #: API behind it, to some clients whose request misses its cache; the
    #: headers do not change that (see _search_arxiv). Asking again in the
    #: same search only adds load, so the remaining queries are skipped.
    _ARXIV_REFUSAL_STATUSES = frozenset({403, 406, 429})

    def _search_arxiv(self, queries: list[str], max_results_per_query: int = 10) -> list[dict]:
        """Query arXiv API with multiple keyword queries and return merged, deduped results.

        Returns paper dicts compatible with the S2 paper schema (paperId uses
        the arXiv ID prefixed with ``arxiv:`` to avoid collision with S2 IDs).

        Each query's outcome ("ok", "failed", "refused" or "skipped") is
        kept in ``_arxiv_query_outcomes`` and the refusal's status (else
        the first non-200 one) in ``_arxiv_http_status``, for
        ``retrieval_status``. After a refusal
        (``_ARXIV_REFUSAL_STATUSES``) the remaining queries are not sent.
        A refusal is not fixed by headers: arXiv's front end turned away
        requests from Python's urllib that carried exactly the headers
        with which ``requests`` got 200, and on a Mac it turned away
        ``requests`` with curl's headers while curl got 200.
        """
        seen_ids: set[str] = set()
        papers: list[dict] = []
        outcomes: list[str] = []
        self._arxiv_query_outcomes = outcomes
        self._arxiv_http_status = None
        refused: int | None = None

        for i, query in enumerate(queries):
            if refused is not None:
                outcomes.append("skipped")
                self._lit_progress("arxiv", i + 1, len(queries), 0, "skipped")
                continue
            if i > 0:
                time.sleep(self._ARXIV_DELAY_S)
            try:
                resp = requests.get(
                    self._ARXIV_API_URL,
                    params={
                        "search_query": f"all:{query}",
                        "start": 0,
                        "max_results": max_results_per_query,
                        "sortBy": "relevance",
                        "sortOrder": "descending",
                    },
                    headers=dict(self._ARXIV_HEADERS),
                    timeout=15,
                )
                if resp.status_code != 200:
                    code = resp.status_code if isinstance(resp.status_code, int) else None
                    outcome = "refused" if code in self._ARXIV_REFUSAL_STATUSES else "failed"
                    if self._arxiv_http_status is None or outcome == "refused":
                        # A refusal ends the search, so its status is the
                        # one the warning names.
                        self._arxiv_http_status = code
                    outcomes.append(outcome)
                    self._lit_progress(
                        "arxiv", i + 1, len(queries), 0, outcome, http_status=code
                    )
                    rest = len(queries) - i - 1
                    if outcome == "refused":
                        refused = code
                        self._note(
                            f"arXiv refused query {i + 1}/{len(queries)} "
                            f"'{query[:50]}' with HTTP {code}"
                            + (f"; not sending the other {rest} arXiv "
                               f"quer{'y' if rest == 1 else 'ies'}" if rest else "")
                        )
                    else:
                        self._note(
                            f"arXiv query {i + 1}/{len(queries)} '{query[:50]}' "
                            f"failed with HTTP {code}"
                        )
                    continue

                root = ET.fromstring(resp.text)
                entries = root.findall("atom:entry", self._ARXIV_NS)
                count = 0
                for entry in entries:
                    arxiv_id_url = entry.findtext("atom:id", "", self._ARXIV_NS)
                    # arXiv ID is the last segment of the URL
                    arxiv_id = arxiv_id_url.rsplit("/", 1)[-1] if arxiv_id_url else ""
                    if not arxiv_id or arxiv_id in seen_ids:
                        continue
                    seen_ids.add(arxiv_id)

                    title = (entry.findtext("atom:title", "", self._ARXIV_NS)
                             .replace("\n", " ").strip())
                    summary = (entry.findtext("atom:summary", "", self._ARXIV_NS)
                               .replace("\n", " ").strip())
                    authors_els = entry.findall("atom:author", self._ARXIV_NS)
                    authors = [
                        a.findtext("atom:name", "", self._ARXIV_NS)
                        for a in authors_els
                    ]
                    published = entry.findtext("atom:published", "", self._ARXIV_NS)
                    year = int(published[:4]) if published and len(published) >= 4 else None

                    papers.append({
                        "paperId": f"arxiv:{arxiv_id}",
                        "title": title,
                        "authors": authors,
                        "year": year,
                        "abstract": summary[:500],
                        "source": "arxiv",
                    })
                    count += 1

                outcomes.append("ok")
                self._lit_progress("arxiv", i + 1, len(queries), count, "ok")
                self.ctx.log.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "agent": self.agent_name,
                    "message": (
                        f"arXiv query {i+1}/{len(queries)} '{query[:50]}': "
                        f"{count} results, {len(papers)} unique total"
                    ),
                })
            except Exception as exc:  # noqa: BLE001
                outcomes.append("failed")
                self._lit_progress("arxiv", i + 1, len(queries), 0, "failed")
                self.ctx.log.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "agent": self.agent_name,
                    "message": f"arXiv query '{query[:50]}' failed: {exc}",
                })

        return papers

    # ------------------------------------------------------------------
    # OpenAlex search (stands in for arXiv)
    # ------------------------------------------------------------------

    _OPENALEX_API_URL = "https://api.openalex.org/works"
    _OPENALEX_HEADERS = {
        "User-Agent": f"EDM-ARS (+{_CROSSREF_PROJECT_URL})",
        "Accept": "application/json",
    }
    #: Only what ``_openalex_paper`` reads; a work's full ``locations``
    #: list more than doubles the response.
    _OPENALEX_SELECT = ",".join((
        "id", "doi", "display_name", "publication_year", "publication_date",
        "type", "authorships", "primary_location", "abstract_inverted_index",
        "cited_by_count", "open_access",
    ))
    #: 429 is OpenAlex's answer both to more than 100 requests a second
    #: and to a spent daily budget (US$0.10 a day without a key); the rest
    #: of the search would be turned away the same way.
    _OPENALEX_LIMIT_STATUSES = frozenset({429})
    _OPENALEX_REFUSAL_STATUSES = frozenset({401, 403, 406})
    _OPENALEX_WHEN = ("arxiv_unavailable", "always")
    _OPENALEX_SEARCH_IN = ("title_and_abstract", "all")
    _OPENALEX_FILTER_UNSAFE = re.compile(r"[,:|()!\"*]")

    def _openalex_settings(self) -> dict[str, Any]:
        """The ``openalex`` block of config.yaml, with its defaults.

        A config without the block (an older ``config.yaml``) gets the
        defaults: on, and asked only when arXiv does not answer. An
        unknown value falls back to its default rather than stopping the
        literature step, which runs first.
        """
        raw = self.config.get("openalex")
        cfg = raw if isinstance(raw, dict) else {}
        when = str(cfg.get("when") or "").strip().lower()
        search_in = str(cfg.get("search_in") or "").strip().lower()
        try:
            delay = max(0.0, float(cfg.get("request_delay_s", 1.0)))
        except (TypeError, ValueError):
            delay = 1.0
        return {
            "enabled": bool(cfg.get("enabled", True)),
            "when": when if when in self._OPENALEX_WHEN else "arxiv_unavailable",
            "max_results_per_query": min(100, max(1, _int_cfg(cfg, "max_results_per_query", 10))),
            "request_delay_s": delay,
            "search_in": search_in if search_in in self._OPENALEX_SEARCH_IN else "title_and_abstract",
        }

    def _openalex_params(self, query: str, per_page: int, search_in: str) -> dict[str, Any]:
        """Query parameters for one OpenAlex search.

        ``title_and_abstract`` matches the words in titles and abstracts
        only. OpenAlex calls this filter form older than ``search=``, but
        ``search=`` also matches full texts: for "college enrollment
        prediction machine learning" it ranked papers on medical imaging
        and cardiovascular risk first, where the filter returned studies
        of enrollment and dropout. Both cost the same.
        """
        params: dict[str, Any] = {"per_page": per_page, "select": self._OPENALEX_SELECT}
        if search_in == "all":
            params["search"] = query
        else:
            # A comma separates filters and a colon or pipe is syntax.
            words = " ".join(self._OPENALEX_FILTER_UNSAFE.sub(" ", query).split())
            params["filter"] = f"title_and_abstract.search:{words}"
        return params

    def _search_openalex(
        self,
        queries: list[str],
        max_results_per_query: int = 10,
        delay_s: float = 1.0,
        search_in: str = "title_and_abstract",
    ) -> list[dict]:
        """Query OpenAlex with the search words arXiv was given.

        Returns records in the literature pool's shape (``_openalex_paper``),
        deduplicated by OpenAlex work id. Each query's outcome ("ok",
        "failed", "rate_limited", "refused" or "skipped") is kept in
        ``_openalex_query_outcomes`` and the stopping (else the first
        failing) HTTP status in ``_openalex_http_status``. After a 429 or
        a refusal the remaining queries are not sent.

        No contact address is sent: OpenAlex retired its mailto "polite
        pool" in February 2026. An ``OPENALEX_API_KEY`` in the environment
        raises the daily budget tenfold; it goes in the Authorization
        header, never in the URL, so no error message or log carries it.
        """
        seen_ids: set[str] = set()
        papers: list[dict] = []
        outcomes: list[str] = []
        self._openalex_query_outcomes = outcomes
        self._openalex_http_status = None
        stopped_by: str | None = None
        headers = dict(self._OPENALEX_HEADERS)
        api_key = (os.environ.get("OPENALEX_API_KEY") or "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        for i, query in enumerate(queries):
            if stopped_by is not None:
                outcomes.append("skipped")
                self._lit_progress(
                    "openalex", i + 1, len(queries), 0, "skipped", skipped_after=stopped_by
                )
                continue
            if i > 0 and delay_s > 0:
                time.sleep(delay_s)
            try:
                resp = requests.get(
                    self._OPENALEX_API_URL,
                    params=self._openalex_params(query, max_results_per_query, search_in),
                    headers=headers,
                    timeout=15,
                )
                if resp.status_code != 200:
                    code = resp.status_code if isinstance(resp.status_code, int) else None
                    if code in self._OPENALEX_LIMIT_STATUSES:
                        outcome = "rate_limited"
                    elif code in self._OPENALEX_REFUSAL_STATUSES:
                        outcome = "refused"
                    else:
                        outcome = "failed"
                    if self._openalex_http_status is None or outcome != "failed":
                        self._openalex_http_status = code
                    outcomes.append(outcome)
                    self._lit_progress(
                        "openalex", i + 1, len(queries), 0, outcome, http_status=code
                    )
                    rest = len(queries) - i - 1
                    if outcome == "failed":
                        self._note(
                            f"OpenAlex query {i + 1}/{len(queries)} '{query[:50]}' "
                            f"failed with HTTP {code}"
                        )
                        continue
                    stopped_by = outcome
                    self._note(
                        f"OpenAlex turned away query {i + 1}/{len(queries)} "
                        f"'{query[:50]}' with HTTP {code}"
                        + (f"; not sending the other {rest} OpenAlex "
                           f"quer{'y' if rest == 1 else 'ies'}" if rest else "")
                    )
                    continue

                results = resp.json().get("results") or []
                count = 0
                for rank, item in enumerate(results):
                    paper = _openalex_paper(item, query, rank)
                    if paper is None or paper["paperId"] in seen_ids:
                        continue
                    seen_ids.add(paper["paperId"])
                    papers.append(paper)
                    count += 1
                outcomes.append("ok")
                self._lit_progress("openalex", i + 1, len(queries), count, "ok")
                self._note(
                    f"OpenAlex query {i + 1}/{len(queries)} '{query[:50]}': "
                    f"{count} results, {len(papers)} unique total"
                )
            except Exception as exc:  # noqa: BLE001 -- a source must not stop the search
                outcomes.append("failed")
                self._lit_progress("openalex", i + 1, len(queries), 0, "failed")
                self._note(
                    f"OpenAlex query {i + 1}/{len(queries)} '{query[:50]}' failed: "
                    f"{type(exc).__name__}: {' '.join(str(exc).split())[:200]}"
                )

        return papers

    def _openalex_wanted(self, arxiv_enabled: bool, settings: dict[str, Any]) -> str:
        """Whether this search asks OpenAlex, as the state it reports when
        it does not: "disabled", "not_needed", or "ask".

        With ``when: arxiv_unavailable`` OpenAlex stands in for an arXiv
        that was asked and did not answer: every query refused or failed.
        An arXiv turned off in config.yaml was not asked, so OpenAlex is
        not either; ``when: always`` asks it in every search.
        """
        if not settings["enabled"]:
            return "disabled"
        if settings["when"] == "always":
            return "ask"
        outcomes = getattr(self, "_arxiv_query_outcomes", None)
        if not arxiv_enabled or not outcomes or "ok" in outcomes:
            return "not_needed"
        return "ask"

    @staticmethod
    def _new_to_pool(
        candidates: list[dict], pool: list[dict], within: bool = False
    ) -> list[dict]:
        """The candidates not already in ``pool``: same DOI, or a title
        with Jaccard similarity >= 0.80 to a title there.

        ``within`` also drops a candidate matching one kept before it (an
        OpenAlex preprint and its published version are separate works).
        """
        pool_tokens = [t for t in (_tokenize_title(p.get("title", "")) for p in pool) if t]
        pool_dois = {normalize_doi(p.get("doi")).lower() for p in pool} - {""}
        kept: list[dict] = []
        for paper in candidates:
            doi = normalize_doi(paper.get("doi")).lower()
            tokens = _tokenize_title(paper.get("title", ""))
            if doi and doi in pool_dois:
                continue
            if any(_jaccard_similarity(tokens, t) >= _JACCARD_THRESHOLD for t in pool_tokens):
                continue
            kept.append(paper)
            if within:
                if tokens:
                    pool_tokens.append(tokens)
                if doi:
                    pool_dois.add(doi)
        return kept

    # ------------------------------------------------------------------
    # Combined literature search (S2 + arXiv + OpenAlex)
    # ------------------------------------------------------------------

    def _search_literature(self, user_prompt: str | None) -> dict:
        """Search Semantic Scholar and arXiv, and OpenAlex when arXiv does
        not answer; merge, deduplicate by DOI and title.

        Returns the same dict format as ``_search_semantic_scholar()``,
        plus ``retrieval_status`` (CONTRACT section 6)::

            {"semantic_scholar": "ok|failed|rate_limited|skipped",
             "arxiv": "ok|failed|refused|disabled",
             "openalex": "ok|failed|rate_limited|refused|not_needed|disabled",
             "n_papers": int, "degraded": bool,
             "n_semantic_scholar": int, "n_arxiv": int, "n_openalex": int}

        plus ``"arxiv_http_status": int`` when arXiv returned nothing and
        answered with an HTTP error ("refused" is 403, 406 or 429), and
        ``"openalex_http_status": int`` likewise for OpenAlex.
        ``openalex`` is "not_needed" when arXiv answered (see
        ``_openalex_wanted``).

        ``degraded`` is true when S2 contributed no papers or the pool is
        empty. A run that went on with placeholders or arXiv alone used to
        finish COMPLETED with the failure recorded only in checkpoint.json
        (E9). It is now written to pipeline.log, emitted as a warning
        event, and carried into run_status.json by the orchestrator.
        """
        self._s2_query_outcomes = None
        self._arxiv_query_outcomes = None
        self._arxiv_http_status = None
        self._openalex_query_outcomes = None
        self._openalex_http_status = None
        self._openalex_plan = None
        arxiv_enabled = bool(self.config.get("arxiv", {}).get("enabled", True))
        self._lit_query_memo = {}
        try:
            result = self._search_literature_sources(user_prompt, arxiv_enabled)
        finally:
            self._lit_query_memo = None
        status = self._retrieval_status(result, arxiv_enabled)
        result["retrieval_status"] = status
        if status["degraded"]:
            hint = (
                " Set SEMANTIC_SCHOLAR_API_KEY for a dedicated rate limit."
                if not os.environ.get("SEMANTIC_SCHOLAR_API_KEY") else ""
            )
            message = (
                "Literature retrieval degraded: Semantic Scholar "
                f"{status['semantic_scholar']} ({status['n_semantic_scholar']} papers), "
                f"{self._arxiv_phrase(status)}, "
                f"{self._openalex_phrase(status)}"
                f"{status['n_papers']} papers in total. Related work and "
                "citations will be thin or placeholders." + hint
            )
            self._note(message)
            self._emit(
                "warning",
                plain="The literature search came back thin",
                code="LITERATURE_DEGRADED",
                message=message,
                retrieval_status=status,
            )
        return result

    def _take_s2_outcome(self, papers: list[dict]) -> str:
        """Outcome of the S2 request that just ran, then forget it."""
        outcome = getattr(self, "_last_s2_outcome", None)
        self._last_s2_outcome = None
        if outcome is None:
            # _run_single_s2_query was replaced (tests); judge by results.
            return "ok" if papers else "failed"
        return str(outcome)

    @staticmethod
    def _arxiv_phrase(status: dict) -> str:
        """arXiv's part of the degraded-literature warning."""
        state = status.get("arxiv")
        code = status.get("arxiv_http_status")
        n = status.get("n_arxiv", 0)
        if state == "refused" and code is not None:
            return f"arXiv refused the request (HTTP {code}, {n} papers)"
        if state == "failed" and code is not None:
            return f"arXiv failed (HTTP {code}, {n} papers)"
        return f"arXiv {state} ({n} papers)"

    @staticmethod
    def _openalex_phrase(status: dict) -> str:
        """OpenAlex's part of the degraded-literature warning, with its
        trailing separator; empty when OpenAlex was not asked."""
        state = status.get("openalex")
        if state in (None, "not_needed", "disabled"):
            return ""
        code = status.get("openalex_http_status")
        n = status.get("n_openalex", 0)
        if code is not None and state in ("failed", "refused", "rate_limited"):
            return f"OpenAlex {state} (HTTP {code}, {n} papers), "
        return f"OpenAlex {state} ({n} papers), "

    def _lit_progress(
        self, source: str, query_index: int, n_queries: int,
        papers_found: int, status: str, http_status: int | None = None,
        skipped_after: str = "refused",
    ) -> None:
        """One ``lit.progress`` event per literature request.

        ``http_status`` is the error status of a request that failed with
        one; a query skipped after a refusal (or, for OpenAlex, after a
        429: ``skipped_after="rate_limited"``) has status ``"skipped"``.
        """
        label = {
            "semantic_scholar": "Semantic Scholar",
            "semantic_scholar_seminal": "Semantic Scholar (seminal works)",
            "arxiv": "arXiv",
            "openalex": "OpenAlex",
        }.get(source, source)
        if status == "refused" and http_status is not None:
            plain = f"{label} refused the request (HTTP {http_status})"
        elif status == "rate_limited" and http_status is not None:
            plain = f"{label} turned the request away: too many requests (HTTP {http_status})"
        elif status == "skipped":
            why = ("turned away an earlier request as too many"
                   if skipped_after == "rate_limited" else "refused an earlier request")
            plain = f"Skipped {label} ({query_index}/{n_queries}): {label} {why}"
        else:
            plain = f"Searching {label} ({query_index}/{n_queries}): {papers_found} found"
        extra: dict[str, Any] = {} if http_status is None else {"http_status": http_status}
        self._emit(
            "lit.progress",
            plain=plain,
            source=source,
            query_index=query_index,
            n_queries=n_queries,
            papers_found=papers_found,
            status=status,
            **extra,
        )

    def _retrieval_status(self, result: dict, arxiv_enabled: bool) -> dict:
        """Summarise what each source returned (CONTRACT section 6)."""
        papers = result.get("papers") or []
        n_arxiv = sum(1 for p in papers if p.get("source") == "arxiv")
        n_openalex = sum(1 for p in papers if p.get("source") == "openalex")
        n_s2 = len(papers) - n_arxiv - n_openalex

        s2_outcomes = getattr(self, "_s2_query_outcomes", None)
        if s2_outcomes is None:
            # The S2 search did not run its query loop (it was replaced);
            # judge by what it returned.
            s2_state = "ok" if n_s2 else "failed"
        elif not s2_outcomes:
            s2_state = "skipped"
        elif "ok" in s2_outcomes:
            s2_state = "ok"
        elif "rate_limited" in s2_outcomes:
            s2_state = "rate_limited"
        else:
            s2_state = "failed"

        if not arxiv_enabled:
            arxiv_state = "disabled"
        else:
            arxiv_outcomes = getattr(self, "_arxiv_query_outcomes", None)
            if arxiv_outcomes is None:
                arxiv_state = "ok" if n_arxiv else "failed"
            elif "ok" in arxiv_outcomes:
                arxiv_state = "ok"
            elif "refused" in arxiv_outcomes:
                arxiv_state = "refused"
            else:
                arxiv_state = "failed"

        openalex_state = self._openalex_state(n_openalex)

        status: dict[str, Any] = {
            "semantic_scholar": s2_state,
            "arxiv": arxiv_state,
            "openalex": openalex_state,
            "n_papers": len(papers),
            # Unchanged by OpenAlex: a pool without Semantic Scholar also
            # lacks its seminal-works query.
            "degraded": n_s2 == 0 or len(papers) == 0,
            "n_semantic_scholar": n_s2,
            "n_arxiv": n_arxiv,
            "n_openalex": n_openalex,
        }
        http_status = getattr(self, "_arxiv_http_status", None)
        if arxiv_state in ("failed", "refused") and isinstance(http_status, int):
            status["arxiv_http_status"] = http_status
        oa_http = getattr(self, "_openalex_http_status", None)
        if openalex_state in ("failed", "refused", "rate_limited") and isinstance(oa_http, int):
            status["openalex_http_status"] = oa_http
        return status

    def _openalex_state(self, n_openalex: int) -> str:
        """OpenAlex's entry in ``retrieval_status``: the plan when it was
        not asked, else the same rule as the other sources."""
        plan = getattr(self, "_openalex_plan", None)
        if plan in ("disabled", "not_needed"):
            return str(plan)
        outcomes = getattr(self, "_openalex_query_outcomes", None)
        if outcomes is None:
            # Not asked through _search_openalex (replaced in tests, or the
            # sources search itself was replaced); judge by the pool.
            if plan is None and not n_openalex:
                return "not_needed"
            return "ok" if n_openalex else "failed"
        for state in ("ok", "rate_limited", "refused"):
            if state in outcomes:
                return state
        return "failed"

    def _search_literature_sources(
        self, user_prompt: str | None, arxiv_enabled: bool
    ) -> dict:
        """The S2 + arXiv (+ OpenAlex) search itself (see ``_search_literature``)."""
        # 1. Run S2 search (primary source)
        s2_context = self._search_semantic_scholar(user_prompt)
        s2_papers = s2_context.get("papers", [])

        # 2. Run arXiv search with same queries
        arxiv_cfg = self.config.get("arxiv", {})
        queries: list[str] | None = None
        arxiv_papers: list[dict] = []
        if arxiv_enabled:
            queries = self._literature_queries(user_prompt)
            arxiv_per_query = int(arxiv_cfg.get("max_results_per_query", 10))
            arxiv_papers = self._search_arxiv(queries, max_results_per_query=arxiv_per_query)

        # 3. OpenAlex, with the same queries, when arXiv did not answer:
        #    arXiv's front end refuses some Python clients outright.
        oa_settings = self._openalex_settings()
        plan = self._openalex_wanted(arxiv_enabled, oa_settings)
        self._openalex_plan = plan
        openalex_papers: list[dict] = []
        if plan == "ask":
            if queries is None:
                queries = self._literature_queries(user_prompt)
            if oa_settings["when"] == "arxiv_unavailable":
                how = "refused" if "refused" in (self._arxiv_query_outcomes or []) else "failed"
                n_q = len(queries)
                self._note(
                    f"arXiv {how}; asking OpenAlex with the same {n_q} "
                    f"quer{'y' if n_q == 1 else 'ies'}"
                )
            openalex_papers = self._search_openalex(
                queries,
                max_results_per_query=oa_settings["max_results_per_query"],
                delay_s=oa_settings["request_delay_s"],
                search_in=oa_settings["search_in"],
            )

        if not arxiv_papers and not openalex_papers:
            return dict(s2_context)

        # 4. Deduplicate against the pool by DOI and title Jaccard
        new_papers = self._new_to_pool(arxiv_papers, s2_papers)
        new_openalex = self._new_to_pool(
            openalex_papers, s2_papers + new_papers, within=True
        )

        merged = s2_papers + new_papers + new_openalex
        merged.sort(key=lambda p: p.get("year") or 0, reverse=True)

        # Trim to combined max. Second year-descending trim site: without
        # the seminal reservation the arXiv preprints (all recent) would
        # push the newly-retrieved old S2 records straight back out of the
        # pool, making the retrieval change a no-op (F-P5-DEPTH-RECENCY-SKEW).
        s2_cfg = self.config.get("semantic_scholar", {})
        sem_cfg = s2_cfg.get("seminal_query") or {}
        max_total = int(s2_cfg.get("max_results", 20))
        merged = self._trim_pool_preserving_seminal(
            merged,
            max_total,
            _int_cfg(sem_cfg, "reserved_pool_slots", _int_cfg(sem_cfg, "limit", 20)),
        )

        self.ctx.log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "agent": self.agent_name,
            "message": (
                f"Literature search merged: {len(s2_papers)} S2 + "
                f"{len(new_papers)} arXiv (deduped)"
                + (f" + {len(new_openalex)} OpenAlex (deduped)" if plan == "ask" else "")
                + f" = {len(merged)} total papers"
            ),
        })

        return {
            **s2_context,
            "papers": merged,
        }

    # ------------------------------------------------------------------
    # Message builders
    # ------------------------------------------------------------------

    def _build_user_message(
        self,
        registry: dict,
        task_template: dict,
        s2_context: dict,
        user_prompt: str | None,
        revision_instructions: str | None,
        findings_memory_summary: str = "",
        prior_specs: list[str] | None = None,
        studied_outcomes: list[str] | None = None,
        locked_research_spec: dict | None = None,
    ) -> str:
        parts = [
            "## Dataset Registry (YAML)",
            "```yaml",
            yaml.dump(registry, default_flow_style=False, allow_unicode=True),
            "```",
            "",
            "## Task Template",
            "```yaml",
            yaml.dump(task_template, default_flow_style=False, allow_unicode=True),
            "```",
            "",
            "## Retrieved Literature (Semantic Scholar + arXiv)",
            "```json",
            # retrieval_status is bookkeeping for run_status.json, not
            # literature; leaving it out keeps this block what it was.
            json.dumps(
                {k: v for k, v in s2_context.items() if k != "retrieval_status"}
                if isinstance(s2_context, dict) else s2_context,
                indent=2,
            ),
            "```",
        ]
        # V3.2 Arc D: deterministic design-feasibility report + gap
        # matrix. Computed here (not at call sites) so every PF path —
        # fresh generation, multi-candidate, and locked-spec refine —
        # receives them. Both are pure functions of inputs this method
        # already holds; failures are non-fatal (advisory context, and
        # PF must still run on registries predating design_feasibility).
        try:
            from src.design_selector import format_design_report, select_design

            task_type = getattr(self.ctx, "task_type", None)
            report = select_design(
                registry,
                question=user_prompt,
                intent=(
                    "targeting"
                    if task_type == "causal_itr"
                    else "causal"
                    if task_type == "causal_soo"
                    else None
                ),
            )
            parts += ["", format_design_report(report)]
        except Exception:
            pass
        try:
            from src.gap_miner import build_gap_matrix, format_gap_matrix

            parts += ["", format_gap_matrix(build_gap_matrix(s2_context))]
        except Exception:
            pass
        if findings_memory_summary:
            parts += [
                "",
                "## Findings Memory Summary",
                findings_memory_summary,
            ]
        if studied_outcomes:
            parts += [
                "",
                "## Studied Outcomes (already investigated in prior runs)",
                "\n".join(f"  - {o}" for o in studied_outcomes),
            ]
        if prior_specs:
            parts += [
                "",
                "## Prior Candidate Specs (already generated this session — generate something DIFFERENT)",
                "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(prior_specs)),
            ]
        if user_prompt:
            parts += [
                "",
                "## User Research Prompt",
                user_prompt,
            ]
        if revision_instructions:
            parts += [
                "",
                "## Revision Instructions from Critic",
                revision_instructions,
            ]
        # Phase 3b.5 (narrow exception #3): wire the locked research_spec
        # through to the user message so the causal_soo PF prompt's
        # "refine the locked spec" branch has the spec to refine.
        if locked_research_spec is not None:
            parts += [
                "",
                "## Locked Research Spec (refine, do not redesign)",
                "```json",
                json.dumps(locked_research_spec, indent=2),
                "```",
            ]
            parts += [
                "",
                "## Task",
                (
                    "Refine the locked research_spec above per the system "
                    "prompt's instructions. Apply the methodology skills (G1 "
                    "DAG, G2 estimand definition, hsls09-causal-conventions) "
                    "to the locked treatment, outcome, adjustment set, and "
                    "method battery. Flag methodological concerns inline "
                    "(ESC-07 median-split, IDF-02 post-treatment covariates, "
                    "etc.). DO NOT redesign the study; the user has chosen "
                    "the variables and methods deliberately. Select 8-12 of "
                    "the most relevant papers from the retrieved literature "
                    "to populate literature_context.papers (copy paperId, "
                    "title, authors, year, abstract exactly). Return ONLY a "
                    "JSON object with 'research_spec' and 'literature_context' "
                    "keys, wrapped in a ```json code block."
                ),
            ]
        else:
            parts += [
                "",
                "## Task",
                (
                    "Design a prediction research question using the "
                    f"{_dataset_label(registry, getattr(self.ctx, 'dataset_name', None))} dataset. "
                    "Select 8-12 of the most relevant papers from the retrieved literature "
                    "(copy their paperId, title, authors, year, abstract exactly) to populate "
                    "literature_context.papers. Ground the novelty claim using these papers. "
                    "Return ONLY a JSON object with 'research_spec' and 'literature_context' "
                    "keys, wrapped in a ```json code block."
                ),
            ]
        return "\n".join(parts)

    def _filter_hallucinated_papers(self, literature_context: dict, s2_context: dict) -> dict:
        """Three-layer citation verification (inspired by AutoResearchClaw).

        Layer 1: exact paper ID match (S2 or arXiv IDs from combined search).
        Layer 2: CrossRef title search with Jaccard similarity ≥ 0.80.
        Layer 3: Jaccard against actual search result titles ≥ 0.80.

        Returns ``literature_context`` with papers annotated with a
        ``verification_status`` field (``"VERIFIED"`` or ``"SUSPICIOUS"``);
        ``"HALLUCINATED"`` papers are silently dropped.  The return dict keys are
        identical to the original, so downstream agents are unaffected.
        """
        real_papers = s2_context.get("papers", [])
        real_ids = {p["paperId"] for p in real_papers if p.get("paperId")}

        if not real_ids:
            # S2 returned nothing — discard any LLM-fabricated papers entirely
            return {
                "search_query": literature_context.get("search_query", s2_context.get("search_query", "")),
                "papers": [],
                "novelty_evidence": literature_context.get("novelty_evidence", s2_context.get("novelty_evidence", "")),
            }

        # Precompute token sets for Layer 3 (avoid re-tokenizing on every paper)
        real_title_tokens: list[tuple[set[str], dict]] = [
            (_tokenize_title(p.get("title", "")), p)
            for p in real_papers
            if p.get("title")
        ]

        verified: list[dict] = []
        suspicious: list[dict] = []

        mailto = _crossref_mailto(self.config)
        for paper in literature_context.get("papers", []):
            status = _verify_paper_three_layers(
                paper, real_ids, real_title_tokens, crossref_mailto=mailto
            )
            if status == "VERIFIED":
                verified.append({**paper, "verification_status": "VERIFIED"})
            elif status == "SUSPICIOUS":
                suspicious.append({**paper, "verification_status": "SUSPICIOUS"})
                self.ctx.log.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "agent": self.agent_name,
                    "message": (
                        f"Citation '{paper.get('title', '?')[:60]}' not in S2 exact results "
                        "but title matches via CrossRef/Jaccard — marked SUSPICIOUS"
                    ),
                })
            # HALLUCINATED: silently dropped (same as before)

        all_papers = verified + suspicious
        return {**literature_context, "papers": all_papers}

    def _log_validation_warnings(self, research_spec: dict, registry: dict) -> None:
        warnings = self._validate_spec(research_spec, registry)
        for w in warnings:
            self.ctx.log.append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "agent": self.agent_name,
                    "message": f"Validation warning: {w}",
                }
            )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_spec(
        self,
        research_spec: dict,
        registry: dict,
    ) -> list[str]:
        """
        Validate the research_spec and return a list of warning strings.

        Delegates to the task template for task-specific validation logic.
        Warnings are non-fatal; the Critic enforces hard failures.
        """
        return self.task_template.validate_research_spec(
            research_spec, registry, self.dataset_adapter
        )
