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

# Backward-compatible re-export of HSLS:09 temporal ordering
from src.dataset_adapter import HSLS09_TEMPORAL_ORDER as TEMPORAL_ORDER  # noqa: F401


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


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


def _tokenize_title(title: str) -> set[str]:
    """Lowercase word tokenization for Jaccard title similarity."""
    return set(re.sub(r"[^\w\s]", "", title.lower()).split())


def _jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _verify_paper_three_layers(
    paper: dict,
    real_ids: set[str],
    real_title_tokens: list[tuple[set[str], dict]],
) -> str:
    """Return 'VERIFIED', 'SUSPICIOUS', or 'HALLUCINATED' for a single paper.

    Layer 1: exact S2 paper ID match.
    Layer 2: CrossRef title search with Jaccard similarity ≥ 0.80.
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
        resp = requests.get(
            _CROSSREF_BASE_URL,
            params={"query.title": title, "rows": 1, "select": "title"},
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

    def run(
        self,
        user_prompt: str | None = None,
        revision_instructions: str | None = None,
        findings_memory_summary: str = "",
        n_candidate_specs: int = 1,
        studied_outcomes: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        """
        Args:
            user_prompt: Optional free-text research direction from the user.
            revision_instructions: Critic feedback from a prior review cycle.
            findings_memory_summary: Summary of prior runs from FindingsMemory.
            n_candidate_specs: Number of candidate specs to generate (1 = current behavior).
            studied_outcomes: Outcome variables already studied in prior runs.

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
            )

        return self._run_single(
            user_prompt=user_prompt,
            revision_instructions=revision_instructions,
            registry=registry,
            task_template_data=task_template_data,
            s2_context=s2_context,
            findings_memory_summary=findings_memory_summary,
            studied_outcomes=studied_outcomes or [],
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
        )

        llm_response = self.call_llm(user_message)
        parsed = parse_llm_json(llm_response)

        research_spec = parsed.get("research_spec") or {}
        literature_context = parsed.get("literature_context") or s2_context
        literature_context = self._filter_hallucinated_papers(literature_context, s2_context)

        self._log_validation_warnings(research_spec, registry)

        return {
            "research_spec": research_spec,
            "literature_context": literature_context,
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
        }

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
        except Exception as exc:  # noqa: BLE001
            self.ctx.log.append({
                "timestamp": datetime.utcnow().isoformat(),
                "agent": self.agent_name,
                "message": f"Query generation failed ({exc}); using defaults.",
            })
        return self._DEFAULT_S2_QUERIES

    def _run_single_s2_query(
        self,
        query: str,
        base_url: str,
        limit: int,
        year_start: int,
        current_year: int,
        headers: dict[str, str],
        max_retries: int,
        backoff_base: float,
        backoff_factor: float,
        use_jitter: bool,
        delay_s: float,
    ) -> list[dict]:
        """Execute one S2 search query with retry/backoff. Returns list of paper dicts."""
        last_exc: Exception | None = None
        retryable = True

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

                resp = requests.get(
                    f"{base_url}/paper/search",
                    params={
                        "query": query,
                        "fields": "paperId,title,authors,year,abstract",
                        "limit": limit,
                        "year": f"{year_start}-{current_year}",
                    },
                    headers=headers,
                    timeout=15,
                )

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
                return [
                    {
                        "paperId": item.get("paperId", ""),
                        "title": item.get("title", ""),
                        "authors": [a.get("name", "") for a in item.get("authors", [])],
                        "year": item.get("year"),
                        "abstract": item.get("abstract") or "",
                    }
                    for item in data.get("data", [])
                    if item.get("paperId")
                ]

            except (requests.ConnectionError, requests.Timeout, requests.RequestException) as exc:
                last_exc = exc
                if not retryable or attempt == max_retries:
                    break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                break

        self.ctx.log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "agent": self.agent_name,
            "message": f"S2 query '{query[:50]}' failed after all retries: {last_exc}",
        })
        return []

    def _search_semantic_scholar(self, user_prompt: str | None) -> dict:
        """Query S2 with multiple short keyword queries and return merged results.

        Inspired by AutoResearchClaw's ``search_papers_multi_query`` pattern:
        run 2–3 focused keyword queries, merge by paperId, deduplicate, and sort
        by year descending.  This avoids the zero-result problem caused by passing
        long natural-language prompts or dataset-specific identifiers to the S2
        full-text search endpoint.
        """
        cfg = self.config.get("semantic_scholar", {})
        base_url = cfg.get("base_url", "https://api.semanticscholar.org/graph/v1")
        max_results = int(cfg.get("max_results", 10))
        year_filter = int(cfg.get("year_filter", 10))
        delay_s = float(cfg.get("request_delay_s", 0.5))
        max_retries = int(cfg.get("max_retries", 3))
        backoff_base = float(cfg.get("backoff_base_s", 1.0))
        backoff_factor = float(cfg.get("backoff_factor", 2.0))
        use_jitter = bool(cfg.get("backoff_jitter", True))

        current_year = datetime.utcnow().year
        year_start = current_year - year_filter

        s2_api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        headers: dict[str, str] = {}
        if s2_api_key:
            headers["X-API-KEY"] = s2_api_key

        # Generate short keyword queries from user_prompt via lightweight LLM call
        queries = self._generate_search_queries(user_prompt)
        per_query_limit = max(max_results, 10)  # fetch at least 10 per query before dedup

        # Run all queries and merge by paperId (dedup)
        seen_ids: set[str] = set()
        merged_papers: list[dict] = []

        for i, query in enumerate(queries):
            if i > 0:
                # Brief inter-query delay (mirrors AutoResearchClaw's 1.5s inter-query pause)
                time.sleep(1.0)
            papers = self._run_single_s2_query(
                query=query,
                base_url=base_url,
                limit=per_query_limit,
                year_start=year_start,
                current_year=current_year,
                headers=headers,
                max_retries=max_retries,
                backoff_base=backoff_base,
                backoff_factor=backoff_factor,
                use_jitter=use_jitter,
                delay_s=delay_s,
            )
            for paper in papers:
                pid = paper.get("paperId", "")
                if pid and pid not in seen_ids:
                    seen_ids.add(pid)
                    merged_papers.append(paper)
            self.ctx.log.append({
                "timestamp": datetime.utcnow().isoformat(),
                "agent": self.agent_name,
                "message": (
                    f"S2 query {i+1}/{len(queries)} '{query[:60]}': "
                    f"{len(papers)} results, {len(merged_papers)} unique total"
                ),
            })

        # Sort by year descending, trim to max_results
        merged_papers.sort(key=lambda p: p.get("year") or 0, reverse=True)
        final_papers = merged_papers[:max_results]

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

    _ARXIV_API_URL = "http://export.arxiv.org/api/query"
    _ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}

    def _search_arxiv(self, queries: list[str], max_results_per_query: int = 10) -> list[dict]:
        """Query arXiv API with multiple keyword queries and return merged, deduped results.

        Returns paper dicts compatible with the S2 paper schema (paperId uses
        the arXiv ID prefixed with ``arxiv:`` to avoid collision with S2 IDs).
        """
        seen_ids: set[str] = set()
        papers: list[dict] = []

        for i, query in enumerate(queries):
            if i > 0:
                time.sleep(1.0)  # rate-limit courtesy
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
                    timeout=15,
                )
                if resp.status_code != 200:
                    self.ctx.log.append({
                        "timestamp": datetime.utcnow().isoformat(),
                        "agent": self.agent_name,
                        "message": f"arXiv query '{query[:50]}' HTTP {resp.status_code}",
                    })
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

                self.ctx.log.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "agent": self.agent_name,
                    "message": (
                        f"arXiv query {i+1}/{len(queries)} '{query[:50]}': "
                        f"{count} results, {len(papers)} unique total"
                    ),
                })
            except Exception as exc:  # noqa: BLE001
                self.ctx.log.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "agent": self.agent_name,
                    "message": f"arXiv query '{query[:50]}' failed: {exc}",
                })

        return papers

    # ------------------------------------------------------------------
    # Combined literature search (S2 + arXiv)
    # ------------------------------------------------------------------

    def _search_literature(self, user_prompt: str | None) -> dict:
        """Search both Semantic Scholar and arXiv, merge, deduplicate by title.

        Returns the same dict format as ``_search_semantic_scholar()``.
        """
        # 1. Run S2 search (primary source)
        s2_context = self._search_semantic_scholar(user_prompt)
        s2_papers = s2_context.get("papers", [])

        # 2. Run arXiv search with same queries
        arxiv_cfg = self.config.get("arxiv", {})
        if not arxiv_cfg.get("enabled", True):
            return s2_context

        queries = self._generate_search_queries(user_prompt)
        arxiv_per_query = int(arxiv_cfg.get("max_results_per_query", 10))
        arxiv_papers = self._search_arxiv(queries, max_results_per_query=arxiv_per_query)

        if not arxiv_papers:
            return s2_context

        # 3. Deduplicate arXiv papers against S2 results by title Jaccard
        s2_title_tokens = [
            _tokenize_title(p.get("title", ""))
            for p in s2_papers
        ]
        new_papers: list[dict] = []
        for arxiv_paper in arxiv_papers:
            arxiv_tokens = _tokenize_title(arxiv_paper.get("title", ""))
            is_dup = any(
                _jaccard_similarity(arxiv_tokens, s2_t) >= 0.80
                for s2_t in s2_title_tokens
                if s2_t
            )
            if not is_dup:
                new_papers.append(arxiv_paper)

        merged = s2_papers + new_papers
        merged.sort(key=lambda p: p.get("year") or 0, reverse=True)

        # Trim to combined max
        max_total = int(self.config.get("semantic_scholar", {}).get("max_results", 20))
        merged = merged[:max_total]

        self.ctx.log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "agent": self.agent_name,
            "message": (
                f"Literature search merged: {len(s2_papers)} S2 + "
                f"{len(new_papers)} arXiv (deduped) = {len(merged)} total papers"
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
            json.dumps(s2_context, indent=2),
            "```",
        ]
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
        parts += [
            "",
            "## Task",
            (
                "Design a prediction research question using the HSLS:09 dataset. "
                "Select 8–12 of the most relevant papers from the retrieved literature "
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

        for paper in literature_context.get("papers", []):
            status = _verify_paper_three_layers(paper, real_ids, real_title_tokens)
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
