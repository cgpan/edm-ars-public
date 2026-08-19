"""Arc T / T1b - Stage 2: novelty as a VETO ONLY.

This module answers exactly one question about an idea card: *is there a
specific paper that already did this, and can I quote the sentence that
proves it?* It answers CLEAR, COLLISION or UNVERIFIABLE. It never
answers "how novel is this", because that question has no measurable
answer in this system.

Why the shape is a veto and not a score (C1)
--------------------------------------------
Two measurements, both ours, both bad:

* ``novelty_score_self_assessment`` correlates r = -0.35 with the LSAR
  Novelty score it was built to predict, and 8 of its 11 recorded values
  are literally the constant hard-coded in the prompt template.
* Published LLM novelty judgements correlate rho = -0.291 with realized
  impact.

A positive novelty claim is an absence-of-evidence claim ("no paper does
this"), and an absence-of-evidence claim cannot be checked. The negative
claim can: "this paper already did it, here is the sentence". So only
the negative claim is computed, and only the negative claim acts.
Corroborating this from our own anchor corpus: 2 of 8 bottom-band
Novelty reviews were punished specifically for an *unsupported*
first-claim, so an unsubstantiated first-claim scores worse than making
no claim at all. :func:`_sanitize_delta_sentence` enforces that on the
model's prose.

What is deterministic and what is judged
----------------------------------------
**The verdict is deterministic.** It is a lexical facet-overlap test
against a retrieved record, plus the requirement that a snippet can be
quoted verbatim out of that record. No model is consulted about whether
a collision exists.

The model is used for one thing only: writing the human-readable
*delta sentence* in the CLEAR case. That is prose, not a verdict, so
C4's order-swap / k-sample discipline does not bind here - there is no
judged verdict to stabilise. If a future revision lets a model
*escalate* to COLLISION, C4 binds immediately and this docstring is
wrong.

Consequence, stated plainly: the trigger is lexical, so it misses
collisions phrased in different words (risk R6). The guards are
(a) UNVERIFIABLE is a distinct third state, (b) the local anchor corpus
seeds retrieval, and (c) every COLLISION carries a citation and a
quoted snippet, so a wrong veto is visible to a human in one glance
rather than being an unaccountable opinion.

UNVERIFIABLE
------------
Retrieval failure must never silently kill an idea. :func:`is_veto`
returns True for COLLISION only, so callers treat UNVERIFIABLE exactly
like CLEAR when deciding whether a card survives. It is nevertheless a
distinct verdict string so :func:`verdict_counts` can make the rate
visible: a screen that "clears" everything because the network was down
is worthless, and the only way to notice is to count.

Offline by default
------------------
``collision_check`` performs NO network I/O unless the caller passes a
``retrieve`` callable. The default corpus is the local LSAR anchor
directory, which is plain file reads. Production wires the online path
explicitly via :func:`problem_formulator_retriever`.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from src.ideation import feasibility as _feas
from src.ideation.cards import IdeaCard

# --------------------------------------------------------------------------
# Verdicts
# --------------------------------------------------------------------------

CLEAR = "CLEAR"
COLLISION = "COLLISION"
UNVERIFIABLE = "UNVERIFIABLE"

VERDICTS: tuple[str, str, str] = (CLEAR, COLLISION, UNVERIFIABLE)

#: Agent key -> ``agent_prompts/idea_priorart.yaml``.
AGENT_KEY = "idea_priorart"

#: Fallback default for ``ideation.priorart.anchor_corpus``. The
#: ``ideation:`` block landed in config.yaml (H5, 2026-07-11); the config
#: value wins whenever present, this constant is the offline fallback.
DEFAULT_ANCHOR_CORPUS = os.environ.get(
    "LSAR_HOME", "../LSAR"
) + "/outputs"

# --------------------------------------------------------------------------
# Trigger thresholds
#
# NOT CALIBRATED. There is no labelled collision set in this repo, so
# these are conservative guesses, deliberately set so that the test
# FAILS to fire rather than fires wrongly: a false COLLISION destroys an
# idea permanently and invisibly (risk R3), whereas a missed collision
# is caught downstream by a reviewer. Every one is overridable from
# ``ideation.priorart`` in config.yaml.
# --------------------------------------------------------------------------

#: Fraction of the card's PURPOSE terms that must appear in the paper.
#: This is the arm that separates. Measured 2026-08-03, nearest-neighbour
#: purpose coverage against the 34 local anchors: 33 real archived specs
#: max 0.20 / median 0.08; a hand-written paraphrase of an anchor 0.89;
#: a card lifted from an anchor's own abstract 1.00. Population and
#: outcome are named entities, so their vocabulary is not free.
PURPOSE_COVERAGE_MIN = 0.60

#: Fraction of the card's MECHANISM terms that must appear in the paper.
#:
#: DISABLED BY DEFAULT (0.0) - the mechanism facet acts as a
#: CORROBORATION arm only (``MIN_SHARED_TERMS`` shared terms still
#: required), not as a second identity arm. This is a measurement, not a
#: preference. Same probe, same day: mechanism coverage against the
#: nearest anchor was median 0.57 across 33 UNRELATED archived specs but
#: 0.45 on the deliberate paraphrase - i.e. the arm scored the
#: near-duplicate LOWER than the unrelated specs, so a coverage
#: threshold on it separates in the wrong direction. Cause: "we audit
#: fairness metrics" and "we examine algorithmic fairness" are the same
#: act in different words, whereas "college enrollment" is "college
#: enrollment" in every paper that studies it.
#:
#: Set it to a positive value in ``ideation.priorart`` to re-enable the
#: stricter conjunction; nothing else changes.
MECHANISM_COVERAGE_MIN = 0.0
#: Absolute floor on shared terms per facet, so a 2-term facet cannot
#: reach 60% coverage on a single generic word.
MIN_SHARED_TERMS = 2
#: How many nearest records are reported (evidence, not a ranking of ideas).
N_NEAREST = 3
#: Hard cap on a quoted snippet.
SNIPPET_MAX_WORDS = 45
#: Hard cap on the model's delta sentence.
DELTA_MAX_WORDS = 45

# --------------------------------------------------------------------------
# Tokenisation
# --------------------------------------------------------------------------

_TOKEN = re.compile(r"[a-z0-9]+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
#: NCES-style variable codes (X4EVRATNDCLG, S1MTESTS, F2EVRATT) and this
#: repo's internal method identifiers (M8, M9, M10). Neither ever appears
#: in a paper's prose, so leaving them in a facet only inflates the
#: denominator. Variable codes are replaced by their registry label
#: words; method codes are simply dropped.
_VAR_CODE = re.compile(r"^[a-z]{1,2}\d[a-z0-9]{2,}$")
_METHOD_CODE = re.compile(r"^m\d{1,2}$")

_FUNCTION_WORDS = frozenset(
    """
    the and for with from that this than then they them their there these those
    are was were will would could should have has had been being does did doing
    not but its it's into onto over under about above after before between both
    each more most other some such only own same too very can just also how why
    what when where which who whom whose while our ours you your yours upon
    per via out off any all one two three
    """.split()
)

#: Terms so ubiquitous in an educational-data-mining corpus that sharing
#: them carries no evidence of a collision. Excluding them makes the
#: trigger strictly harder to fire, which is the direction we want.
DOMAIN_STOPWORDS = frozenset(
    """
    student students learner learners education educational learning teaching
    school schools data dataset datasets analysis analyses analytic analytics
    model models modeling modelling method methods methodology approach
    approaches study studies research paper result results finding findings
    using use used using based effect effects evidence work
    """.split()
)

_STOPWORDS = _FUNCTION_WORDS | DOMAIN_STOPWORDS

#: Phrases that turn a delta sentence into an unsupported first-claim.
#: Measured in the anchor corpus: reviewers punish these specifically.
FIRST_CLAIM_PHRASES: tuple[str, ...] = (
    "first ",
    "the first",
    "novel",
    "novelty",
    "unprecedented",
    "no prior work",
    "no previous work",
    "no existing work",
    "has never been",
    "never been studied",
    "for the first time",
    "unexplored",
    "untouched",
)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fold(token: str) -> str:
    """Crude, deterministic morphological fold. Not a stemmer, on purpose.

    Only three suffix families are folded, all chosen because they cost
    us real matches in an EDM corpus without merging distinct concepts:
    plurals, ``-ic(al)(ly)`` (algorithmically / algorithmic), and
    ``-ness`` (fairness / fair). A general ``-ly`` rule is deliberately
    NOT applied: it turns "apply" into "app".
    """
    if len(token) > 7 and token.endswith("ically"):
        token = token[:-6] + "ic"
    elif len(token) > 5 and token.endswith("ical"):
        token = token[:-4] + "ic"
    elif len(token) > 5 and token.endswith("ness"):
        token = token[:-4]
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("es") and not token.endswith("ses"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def terms(text: str) -> set[str]:
    """Content terms of a piece of text. Deterministic, no model."""
    out: set[str] = set()
    for raw in _TOKEN.findall((text or "").lower()):
        if len(raw) < 3 or raw in _STOPWORDS:
            continue
        if raw.isdigit():
            continue
        folded = _fold(raw)
        if len(folded) < 3 or folded in _STOPWORDS:
            continue
        out.add(folded)
    return out


def _is_var_code(token: str) -> bool:
    """True for a registry variable code or an internal method code."""
    low = token.lower()
    return bool(_VAR_CODE.match(low) or _METHOD_CODE.match(low))


# --------------------------------------------------------------------------
# Facet decomposition
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Facets:
    """The card, split three ways (spec sec. 3 Stage 2 step 1).

    * **purpose** - population and outcome: who is studied and what is
      being explained.
    * **mechanism** - design and estimator: how the claim would be
      identified.
    * **evaluation** - what would count as the result.

    The split matters because facet-matched reranking is the lever this
    stage rests on. (Reported 89.66% vs 13.79% against general-relevance
    reranking on a 58-idea set. That baseline is below chance and is
    probably degenerate, so the direction is the usable part, not the
    6.5x magnitude. We have not replicated either number.)
    """

    purpose: str = ""
    mechanism: str = ""
    evaluation: str = ""
    sources: dict[str, list[str]] = field(default_factory=dict)

    @property
    def purpose_terms(self) -> set[str]:
        return terms(self.purpose)

    @property
    def mechanism_terms(self) -> set[str]:
        return terms(self.mechanism)

    @property
    def evaluation_terms(self) -> set[str]:
        return terms(self.evaluation)

    def to_dict(self) -> dict:
        return {
            "purpose": self.purpose,
            "mechanism": self.mechanism,
            "evaluation": self.evaluation,
            "sources": {k: list(v) for k, v in sorted(self.sources.items())},
        }

    def queries(self) -> list[str]:
        """Short retrieval queries, most specific first."""
        return build_queries(self)


def _as_card(card: IdeaCard | dict) -> IdeaCard:
    if isinstance(card, IdeaCard):
        return card
    return IdeaCard.from_dict(card or {})


def _label_words(
    names: Iterable[str],
    var_map: dict[str, dict],
) -> list[str]:
    """Registry labels for variable codes; the code itself when unknown."""
    out: list[str] = []
    for name in names:
        if not name:
            continue
        meta = var_map.get(name) or {}
        label = str(meta.get("label") or "").strip()
        out.append(label if label else str(name))
    return out


def _spec_strings(value: Any) -> list[str]:
    """Flatten a spec_draft sub-block into plain strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (int, float)):
        return [str(value)]
    if isinstance(value, dict):
        out: list[str] = []
        for key in ("variable", "name", "column", "definition", "description"):
            if key in value:
                out.extend(_spec_strings(value[key]))
        return out
    if isinstance(value, (list, tuple, set)):
        out = []
        for item in value:
            out.extend(_spec_strings(item))
        return out
    return [str(value)]


def decompose(
    card: IdeaCard | dict,
    *,
    registry: dict | None = None,
    registry_dir: str | os.PathLike[str] | None = None,
) -> Facets:
    """Split a card into purpose / mechanism / evaluation facets.

    Deterministic and offline. Variable codes are swapped for their
    registry labels where the registry knows them, because ``X4EVRATNDCLG``
    matches no paper on earth while "ever attended college" matches the
    right ones.
    """
    obj = _as_card(card)
    draft = obj.spec_draft if isinstance(obj.spec_draft, dict) else {}

    var_map: dict[str, dict] = {}
    if registry is None and obj.dataset:
        registry, _path = _feas.load_registry(obj.dataset, registry_dir)
    if registry:
        try:
            var_map = _feas.build_var_map(registry)
        except Exception:  # noqa: BLE001 - a registry shape change must not
            var_map = {}  # break the veto; it degrades to code-free terms.

    sources: dict[str, list[str]] = {}

    # --- purpose: population + outcome ---------------------------------
    #
    # ``why_it_matters`` is deliberately EXCLUDED. It is motivation, not
    # identity: it belongs to the significance dimension the pairwise
    # judge scores, not to "who is studied and what is explained".
    # Measured 2026-08-03 on a hand-built near-duplicate of the
    # "Examining the Algorithmic Fairness in Predicting High School
    # Dropouts" anchor: including it grew the purpose facet from 9 to 18
    # terms and dropped purpose coverage from 0.78 to 0.39, i.e. the
    # motivation prose alone pushed a deliberate duplicate below the veto
    # threshold. A facet padded with rhetoric cannot veto anything.
    purpose_bits: list[str] = []
    purpose_src: list[str] = []
    if obj.research_question:
        purpose_bits.append(obj.research_question)
        purpose_src.append("card.research_question")
    population = draft.get("target_population") or draft.get("population")
    if population:
        purpose_bits.extend(_spec_strings(population))
        purpose_src.append("spec_draft.target_population")
    outcome_names = [
        n
        for n in _spec_strings(draft.get("outcome_variable"))
        + _spec_strings(draft.get("outcome"))
        + ([obj.resolved_target] if obj.resolved_target else [])
        if n
    ]
    if outcome_names:
        seen: list[str] = []
        for name in outcome_names:
            if name not in seen:
                seen.append(name)
        purpose_bits.extend(_label_words(seen, var_map))
        purpose_src.append("spec_draft.outcome/resolved_target (registry label)")
    if purpose_src:
        sources["purpose"] = purpose_src

    # --- mechanism: design + estimator ---------------------------------
    mech_bits: list[str] = []
    mech_src: list[str] = []
    if obj.what_we_would_do:
        mech_bits.append(obj.what_we_would_do)
        mech_src.append("card.what_we_would_do")
    if obj.task_type:
        mech_bits.append(obj.task_type.replace("_", " "))
        mech_src.append("cell.task_type")
    if obj.method_family:
        mech_bits.append(obj.method_family.replace("_", " "))
        mech_src.append("card.method_family")
    for key in ("primary_method", "methods", "secondary_methods",
                "target_estimand_hint", "estimator", "design"):
        values = _spec_strings(draft.get(key))
        if values:
            mech_bits.extend(values)
            mech_src.append(f"spec_draft.{key}")
    treatment = _spec_strings(draft.get("treatment"))
    if treatment:
        mech_bits.extend(_label_words(treatment, var_map))
        mech_src.append("spec_draft.treatment (registry label)")
    if mech_src:
        sources["mechanism"] = mech_src

    # --- evaluation: what would count as the result --------------------
    eval_bits: list[str] = []
    eval_src: list[str] = []
    if obj.what_counts_as_the_result:
        eval_bits.append(obj.what_counts_as_the_result)
        eval_src.append("card.what_counts_as_the_result")
    if obj.second_contribution:
        eval_bits.append(obj.second_contribution)
        eval_src.append("card.second_contribution")
    if eval_src:
        sources["evaluation"] = eval_src

    def _join(bits: list[str]) -> str:
        cleaned = [str(b).strip() for b in bits if str(b).strip()]
        return " ".join(cleaned)

    # Variable codes that survived (no registry label) are dropped: they
    # cannot match a paper and would only inflate the denominator.
    _PUNCT = ".,;:()[]{}?!\"'`"

    def _drop_codes(text: str) -> str:
        return " ".join(
            tok for tok in text.split() if not _is_var_code(tok.strip(_PUNCT))
        )

    return Facets(
        purpose=_drop_codes(_join(purpose_bits)),
        mechanism=_drop_codes(_join(mech_bits)),
        evaluation=_drop_codes(_join(eval_bits)),
        sources=sources,
    )


def build_queries(facets: Facets, *, max_words: int = 8) -> list[str]:
    """Short keyword queries derived from the facets, most specific first.

    Fed to the retriever. Deliberately short: the S2 full-text endpoint
    returns nothing for long natural-language strings, which is the
    lesson already encoded in ``ProblemFormulator._generate_search_queries``.
    """
    out: list[str] = []
    for name in ("purpose", "mechanism", "evaluation"):
        text = getattr(facets, name, "") or ""
        picked: list[str] = []
        for raw in _TOKEN.findall(text.lower()):
            if len(raw) < 4 or raw in _STOPWORDS or raw.isdigit():
                continue
            if _is_var_code(raw):
                continue
            if raw not in picked:
                picked.append(raw)
            if len(picked) >= max_words:
                break
        if picked:
            query = " ".join(picked)
            if query not in out:
                out.append(query)
    return out


# --------------------------------------------------------------------------
# The local anchor corpus
# --------------------------------------------------------------------------


def _clean_title(line: str) -> str:
    text = re.sub(r"[*#`_]+", " ", line or "")
    text = re.sub(r"\[\d+\]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _parse_paper_md(text: str) -> tuple[str, str]:
    """``(title, abstract)`` from an LSAR ``paper.md``.

    Keyed on the document text, never the directory stem: the EDM 2024
    anchor directories are rotated relative to their contents (the
    ``theory_building_dbr_*`` directory holds the DRL pedagogical-policy
    paper), and ``metadata.json``'s title is empty or a single character
    for a large share of them.
    """
    lines = [ln.rstrip() for ln in (text or "").splitlines()]
    title = ""
    idx = 0
    for i, line in enumerate(lines):
        candidate = _clean_title(line)
        if not candidate:
            continue
        low = candidate.lower()
        # Journal running heads / DOI lines are not titles.
        if low.startswith("volume") or "doi.org" in low or "http" in low:
            continue
        if len(candidate) < 8:
            continue
        title = candidate
        idx = i
        break

    abstract_lines: list[str] = []
    started = False
    for line in lines[idx + 1:]:
        stripped = _clean_title(line)
        low = stripped.lower()
        if not started:
            if low.startswith("abstract"):
                started = True
                remainder = stripped[len("abstract"):].strip(" :.-")
                if remainder:
                    abstract_lines.append(remainder)
            continue
        if not stripped:
            if abstract_lines:
                break
            continue
        if low.startswith(("1.", "introduction", "keywords", "ccs concepts")):
            break
        abstract_lines.append(stripped)
        if sum(len(x.split()) for x in abstract_lines) > 400:
            break
    return title, re.sub(r"\s+", " ", " ".join(abstract_lines)).strip()


def _metadata_abstract(directory: Path) -> str:
    """``metadata.json['abstract']`` if readable, else ``''``."""
    path = directory / "metadata.json"
    if not path.is_file():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace")) or {}
    except (OSError, ValueError):
        return ""
    return re.sub(r"\s+", " ", str(data.get("abstract") or "")).strip()


def _slug(text: str, *, max_len: int = 60) -> str:
    slug = "-".join(_TOKEN.findall((text or "").lower()))
    return slug[:max_len] or "untitled"


def load_anchor_corpus(
    path: str | os.PathLike[str] | None = None,
) -> list[dict]:
    """Load the local LSAR anchor papers as retrieval records.

    Returns ``[]`` (never raises) when the directory is absent - a
    machine without the LSAR checkout still runs the screen, it just
    retrieves less, which surfaces as UNVERIFIABLE rather than as a
    false CLEAR.

    Records are deduplicated by title: the directory holds repeat review
    runs of the same paper, and counting one paper twice would double
    its weight in the veto.

    ``paperId`` is derived from the TITLE, not the directory stem, for
    the rotation reason in :func:`_parse_paper_md`. ``year`` is ``None``:
    the anchor artifacts carry no publication year field, and inventing
    one from the directory timestamp would be a fabricated fact (C2).
    """
    root = Path(path or DEFAULT_ANCHOR_CORPUS)
    if not root.is_dir():
        return []
    records: list[dict] = []
    seen_titles: set[str] = set()
    for child in sorted(root.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        paper_md = child / "paper.md"
        if not paper_md.is_file():
            continue
        try:
            text = paper_md.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        title, abstract = _parse_paper_md(text)
        if not title:
            continue
        if not abstract:
            # Measured 2026-08-03: 10 of the 34 anchors (the JEDM/JLA
            # PDFs) carry no "Abstract" heading in paper.md - the
            # abstract simply follows the author block. metadata.json
            # holds a usable abstract for all 41 directories that have
            # one, so it is the fallback for the BODY only. The TITLE
            # still comes from paper.md: metadata's title is empty or a
            # single character for a large share of the corpus.
            abstract = _metadata_abstract(child)
        key = _slug(title)
        if key in seen_titles:
            continue
        seen_titles.add(key)
        venue = None
        vc_path = child / "venue_classification.json"
        if vc_path.is_file():
            try:
                venue = (
                    json.loads(vc_path.read_text(encoding="utf-8", errors="replace"))
                    or {}
                ).get("venue")
            except (OSError, ValueError):
                venue = None
        records.append(
            {
                "paperId": f"anchor:{key}",
                "title": title,
                "abstract": abstract,
                "year": None,
                "venue": venue or "local anchor corpus",
                "source": "anchor",
                "source_dir": child.name,
            }
        )
    return records


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Match:
    paper: dict
    shared: dict[str, list[str]]
    coverage: dict[str, float]  # internal only; never emitted (C1)
    snippet: str

    @property
    def order_key(self) -> tuple:
        combined = (
            self.coverage.get("purpose", 0.0)
            + self.coverage.get("mechanism", 0.0)
            + 0.5 * self.coverage.get("evaluation", 0.0)
        )
        return (
            -round(combined, 6),
            -len(self.shared.get("purpose", [])),
            str(self.paper.get("paperId") or ""),
        )


def _paper_text(paper: dict) -> str:
    return " ".join(
        str(paper.get(key) or "") for key in ("title", "abstract")
    ).strip()


def _pick_snippet(paper: dict, wanted: set[str], max_words: int) -> str:
    """Verbatim sentence from the record carrying the most shared terms.

    The returned string is always a substring of ``title + ' ' + abstract``
    after whitespace normalisation, so a quoted snippet can be checked
    against its source.
    """
    abstract = re.sub(r"\s+", " ", str(paper.get("abstract") or "")).strip()
    title = re.sub(r"\s+", " ", str(paper.get("title") or "")).strip()
    candidates = [s.strip() for s in _SENTENCE_SPLIT.split(abstract) if s.strip()]
    if title:
        candidates.append(title)
    best = ""
    best_hits = -1
    for sentence in candidates:
        hits = len(terms(sentence) & wanted)
        if hits > best_hits:
            best_hits = hits
            best = sentence
    if not best or best_hits <= 0:
        return ""
    words = best.split()
    if len(words) > max_words:
        best = " ".join(words[:max_words])
    return best.strip()


def _match(paper: dict, facets: Facets, snippet_max_words: int) -> _Match:
    text_terms = terms(_paper_text(paper))
    shared: dict[str, list[str]] = {}
    coverage: dict[str, float] = {}
    for name in ("purpose", "mechanism", "evaluation"):
        facet_terms = getattr(facets, f"{name}_terms")
        hit = sorted(facet_terms & text_terms)
        shared[name] = hit
        coverage[name] = (len(hit) / len(facet_terms)) if facet_terms else 0.0
    wanted = set(shared["purpose"]) | set(shared["mechanism"])
    snippet = _pick_snippet(paper, wanted, snippet_max_words)
    return _Match(paper=paper, shared=shared, coverage=coverage, snippet=snippet)


def _is_collision(match: _Match, cfg: dict) -> bool:
    """Deterministic veto trigger: "veto if no defensible delta".

    Coverage is measured over the CARD's facet terms, so it asks the
    right question - is everything this idea is about already inside
    that paper? The terms NOT found are the delta. A veto therefore
    fires only when the delta is nearly empty on both identity facets.

    All five must hold:
      1. the record carries a citable ``paperId``,
      2. a snippet can be quoted verbatim from it,
      3. purpose coverage at or above threshold (the identity arm),
      4. mechanism coverage at or above threshold (0.0 by default, i.e.
         corroboration only - see MECHANISM_COVERAGE_MIN),
      5. at least ``min_shared_terms`` shared terms in EACH of the two.

    (1) and (2) are not formalities. A COLLISION that cannot name a
    paper and quote a sentence out of it is an opinion, and this module
    does not ship opinions - such a record can only ever be reported as
    the nearest CLEAR neighbour.

    Threshold provenance, stated because it matters. There is no
    labelled collision set in this repo, so nothing here is calibrated
    in the sense the word deserves. What IS measured (2026-08-03, 34
    local anchors, purpose / mechanism coverage against the nearest
    anchor):

    ==================================== ======= =========
    probe                                purpose mechanism
    ==================================== ======= =========
    33 real archived specs, max          0.20    1.00
    33 real archived specs, median       0.08    0.57
    hand paraphrase of an anchor         0.89    0.45
    card lifted from an anchor abstract  1.00    1.00
    ==================================== ======= =========

    The purpose column separates by a factor of four; the mechanism
    column does not separate at all and ranks the paraphrase below
    unrelated work. Hence the arm split. The 0.60 purpose threshold is
    NOT fitted to any of these numbers - it sits three times above the
    worst real spec and well below both duplicates, which is the only
    property a threshold can honestly claim on this evidence.

    What is NOT measured: the false-veto rate against a live S2/arXiv
    pool of hundreds of records rather than 34 local anchors. Until that
    exists, run this stage in advisory mode (record the verdict, do not
    act on it) exactly as spec sec. 9 stages the rest of T1b.
    """
    if not str(match.paper.get("paperId") or "").strip():
        return False
    if not match.snippet:
        return False
    purpose_min = float(cfg.get("purpose_coverage_min", PURPOSE_COVERAGE_MIN))
    mech_min = float(cfg.get("mechanism_coverage_min", MECHANISM_COVERAGE_MIN))
    min_shared = int(cfg.get("min_shared_terms", MIN_SHARED_TERMS))
    if len(match.shared.get("purpose", [])) < min_shared:
        return False
    if len(match.shared.get("mechanism", [])) < min_shared:
        return False
    if match.coverage.get("purpose", 0.0) < purpose_min:
        return False
    if match.coverage.get("mechanism", 0.0) < mech_min:
        return False
    return True


# --------------------------------------------------------------------------
# Delta sentence
# --------------------------------------------------------------------------


def _fallback_delta(facets: Facets, match: _Match | None) -> str:
    """Deterministic delta sentence. Used when no model is wired, and
    whenever the model's sentence has to be discarded."""
    if match is None:
        return (
            "No prior record was retrieved to compare against, so no "
            "difference can be stated."
        )
    weakest = min(
        ("purpose", "mechanism", "evaluation"),
        key=lambda name: (match.coverage.get(name, 0.0), name),
    )
    facet_text = getattr(facets, weakest, "") or ""
    words = facet_text.split()
    if len(words) > 20:
        facet_text = " ".join(words[:20])
    title = str(match.paper.get("title") or "the nearest record")
    return (
        f"Differs from \"{title}\" on the {weakest} facet: {facet_text}."
        if facet_text
        else f"Differs from \"{title}\" on the {weakest} facet."
    )


def _sanitize_delta_sentence(text: str) -> tuple[str, str | None]:
    """``(sentence, rejection_reason)``.

    Rejects the sentence outright when the model smuggles in a
    first-claim. An unsupported first-claim measurably scores WORSE than
    no claim in our own anchor reviews, and a delta sentence is supposed
    to state a difference, not assert precedence.
    """
    cleaned = re.sub(r"\s+", " ", (text or "")).strip()
    cleaned = cleaned.strip("`").strip()
    if not cleaned:
        return "", "model returned nothing"
    low = " " + cleaned.lower()
    for phrase in FIRST_CLAIM_PHRASES:
        if phrase in low:
            return "", f"model emitted an unsupported first-claim ({phrase.strip()!r})"
    # Reject an emitted rating, not the word "score". "Test scores" and
    # "math score" are ordinary EDM outcome nouns and a delta sentence
    # about them is legitimate; a percentage, a decimal or an "out of 10"
    # is the model rating something, which it was not asked to do. Bare
    # integers stay legal so a sentence can cite a publication year.
    if re.search(r"\d\s*%|\b\d+\.\d+\b|\bnovelty\b|\bout of (?:5|10)\b", low):
        return "", "model emitted a rating, a decimal or a percentage"
    sentences = [s for s in _SENTENCE_SPLIT.split(cleaned) if s.strip()]
    cleaned = sentences[0].strip() if sentences else cleaned
    words = cleaned.split()
    if len(words) > DELTA_MAX_WORDS:
        cleaned = " ".join(words[:DELTA_MAX_WORDS])
    return cleaned, None


def build_delta_prompt(facets: Facets, match: _Match) -> str:
    """User message for the delta-sentence call. No verdict is requested."""
    paper = match.paper
    return "\n".join(
        [
            "## The idea, split into facets",
            f"purpose: {facets.purpose or '(not stated)'}",
            f"mechanism: {facets.mechanism or '(not stated)'}",
            f"evaluation: {facets.evaluation or '(not stated)'}",
            "",
            "## Nearest retrieved prior work",
            f"paperId: {paper.get('paperId')}",
            f"title: {paper.get('title')}",
            f"year: {paper.get('year')}",
            f"venue: {paper.get('venue')}",
            f"quoted snippet: \"{match.snippet}\"",
            "",
            "## Terms this idea shares with that record",
            f"purpose: {', '.join(match.shared.get('purpose', [])) or '(none)'}",
            f"mechanism: {', '.join(match.shared.get('mechanism', [])) or '(none)'}",
            f"evaluation: {', '.join(match.shared.get('evaluation', [])) or '(none)'}",
            "",
            "Write ONE sentence naming the concrete difference between the "
            "idea and that record. Name which facet differs and say how. "
            "Do not judge the idea. Do not claim anything is first, new, "
            "novel or unexplored. Do not output a number, score or "
            "percentage. Return the sentence and nothing else.",
        ]
    )


# --------------------------------------------------------------------------
# The check
# --------------------------------------------------------------------------

Retriever = Callable[[list[str]], Sequence[dict]]


def _priorart_cfg(config: dict | None) -> dict:
    ideation = (config or {}).get("ideation") or {}
    return ideation.get("priorart") or {}


def _dedupe_papers(papers: Iterable[dict]) -> list[dict]:
    """By paperId, then by title slug.

    Anchors win ties because they are locally checkable - we hold the
    full text, so a quoted snippet can be verified by opening a file.
    But the anchor artifacts carry no year and only a coarse venue, so
    fields MISSING on the winner are filled from the duplicate that lost.
    Otherwise deduplication would silently strip the publication year
    off a citation, which is the sort of quiet evidence loss C2 exists
    to prevent.
    """
    out: list[dict] = []
    by_id: dict[str, dict] = {}
    by_title: dict[str, dict] = {}
    ordered = list(papers)
    ordered.sort(key=lambda p: 0 if p.get("source") == "anchor" else 1)
    for paper in ordered:
        if not isinstance(paper, dict):
            continue
        pid = str(paper.get("paperId") or "")
        title_key = _slug(str(paper.get("title") or ""))
        kept = by_id.get(pid) if pid else None
        if kept is None and title_key != "untitled":
            kept = by_title.get(title_key)
        if kept is not None:
            for key, value in paper.items():
                if kept.get(key) in (None, "", []) and value not in (None, "", []):
                    kept[key] = value
            continue
        record = dict(paper)
        if pid:
            by_id[pid] = record
        if title_key != "untitled":
            by_title[title_key] = record
        out.append(record)
    return out


def _nearest_entry(match: _Match) -> dict:
    paper = match.paper
    return {
        "paperId": paper.get("paperId"),
        "title": paper.get("title"),
        "year": paper.get("year"),
        "venue": paper.get("venue"),
        "snippet": match.snippet,
    }


def collision_check(
    card: IdeaCard | dict,
    *,
    retrieve: Retriever | None = None,
    call_llm: Callable[[str], str] | None = None,
    anchors: Sequence[dict] | None = None,
    anchor_dir: str | os.PathLike[str] | None = None,
    registry: dict | None = None,
    registry_dir: str | os.PathLike[str] | None = None,
    config: dict | None = None,
) -> dict:
    """Stage 2 of the cascade: does a specific paper already do this?

    Returns::

        {"verdict": "CLEAR" | "COLLISION" | "UNVERIFIABLE",
         "nearest": [{"paperId","title","year","venue","snippet"}, ...],
         "delta_sentence": str | None,
         ...}

    plus ``facets``, ``evidence`` (C2), ``retrieval`` and ``card_id``.

    Nothing numeric describing the idea is returned. There is no score,
    no rank and no percentage anywhere in the payload - only counts of
    retrieved records and the publication years of cited papers.

    Args:
        card: an :class:`IdeaCard` or its dict form.
        retrieve: ``queries -> papers``. OMIT IT for a fully offline
            check against the local anchors; pass
            :func:`problem_formulator_retriever` in production. An
            exception raised here is caught and downgrades the verdict
            to UNVERIFIABLE - retrieval failure never kills an idea.
        call_llm: ``user_message -> text``, used ONLY to word the delta
            sentence. Never consulted about the verdict.
        anchors: pre-loaded anchor records (tests pass ``[]``).
        anchor_dir: override for ``ideation.priorart.anchor_corpus``.
    """
    obj = _as_card(card)
    cfg = _priorart_cfg(config)
    n_nearest = int(cfg.get("n_nearest", N_NEAREST))
    snippet_max = int(cfg.get("snippet_max_words", SNIPPET_MAX_WORDS))

    facets = decompose(obj, registry=registry, registry_dir=registry_dir)
    queries = build_queries(facets)

    pool: list[dict] = []
    sources: list[str] = []
    errors: list[str] = []

    if anchors is None:
        anchor_path = anchor_dir or cfg.get("anchor_corpus") or DEFAULT_ANCHOR_CORPUS
        try:
            anchors = load_anchor_corpus(anchor_path)
        except Exception as exc:  # noqa: BLE001
            anchors = []
            errors.append(f"anchor corpus unreadable: {exc}")
    if anchors:
        pool.extend(anchors)
        sources.append(f"anchors({len(anchors)})")

    if retrieve is not None:
        try:
            retrieved = list(retrieve(queries) or [])
        except Exception as exc:  # noqa: BLE001 - retrieval must never kill
            retrieved = []
            errors.append(f"retrieval failed: {type(exc).__name__}: {exc}")
        else:
            pool.extend(p for p in retrieved if isinstance(p, dict))
            sources.append(f"retrieved({len(retrieved)})")
    else:
        sources.append("retrieved(skipped: no retriever wired)")

    pool = _dedupe_papers(pool)
    usable = [p for p in pool if _paper_text(p)]

    retrieval = {
        "queries": queries,
        "sources": sources,
        "n_considered": len(usable),
        "errors": errors,
    }

    base = {
        "card_id": obj.candidate_id,
        "tournament_id": obj.tournament_id,
        "facets": facets.to_dict(),
        "retrieval": retrieval,
        "checked_at": _now(),
    }

    # --- UNVERIFIABLE: nothing to compare against ----------------------
    if not usable:
        why = "; ".join(errors) if errors else "no records with usable text"
        return {
            **base,
            "verdict": UNVERIFIABLE,
            "nearest": [],
            "delta_sentence": None,
            "evidence": (
                f"UNVERIFIABLE: retrieval returned 0 usable records "
                f"({why}). Sources tried: {', '.join(sources)}. "
                "Treated as CLEAR by callers (is_veto is False); recorded "
                "distinctly so the rate stays visible."
            ),
        }

    matches = sorted(
        (_match(paper, facets, snippet_max) for paper in usable),
        key=lambda m: m.order_key,
    )
    nearest = [_nearest_entry(m) for m in matches[:n_nearest]]
    top = matches[0]

    # --- COLLISION: a specific paper + a quoted snippet ----------------
    colliding = next((m for m in matches if _is_collision(m, cfg)), None)
    if colliding is not None:
        entry = _nearest_entry(colliding)
        ordered = [entry] + [
            e for e in nearest if e["paperId"] != entry["paperId"]
        ]
        return {
            **base,
            "verdict": COLLISION,
            "nearest": ordered[:n_nearest],
            "delta_sentence": None,
            "evidence": (
                f"COLLISION: paperId={colliding.paper.get('paperId')} "
                f"(\"{colliding.paper.get('title')}\") shares purpose terms "
                f"{colliding.shared.get('purpose')} and mechanism terms "
                f"{colliding.shared.get('mechanism')} with this card, and the "
                f"snippet \"{colliding.snippet}\" is quoted verbatim from its "
                "title/abstract. Deterministic trigger; no model was asked "
                "whether this is a collision."
            ),
        }

    # --- CLEAR: nearest record named, delta stated ---------------------
    delta = _fallback_delta(facets, top)
    delta_note = "deterministic template (no model wired)"
    if call_llm is not None:
        try:
            raw = call_llm(build_delta_prompt(facets, top))
        except Exception as exc:  # noqa: BLE001
            delta_note = f"model call failed ({type(exc).__name__}); template used"
        else:
            sentence, reason = _sanitize_delta_sentence(raw)
            if sentence:
                delta = sentence
                delta_note = "model-written, sanitized"
            else:
                delta_note = f"model sentence discarded ({reason}); template used"

    return {
        **base,
        "verdict": CLEAR,
        "nearest": nearest,
        "delta_sentence": delta,
        "evidence": (
            f"CLEAR: nearest is paperId={top.paper.get('paperId')} "
            f"(\"{top.paper.get('title')}\"), which shares purpose terms "
            f"{top.shared.get('purpose')} and mechanism terms "
            f"{top.shared.get('mechanism')} - below the conjunctive veto "
            f"trigger. Considered {len(usable)} record(s) from "
            f"{', '.join(sources)}. Delta sentence: {delta_note}."
        ),
    }


# --------------------------------------------------------------------------
# Caller-facing helpers
# --------------------------------------------------------------------------


def is_veto(result: dict | str | None) -> bool:
    """True only for COLLISION.

    UNVERIFIABLE is deliberately NOT a veto: a retrieval outage must
    never quietly destroy an idea. It is still recorded as its own
    verdict so :func:`verdict_counts` makes the rate visible.
    """
    if result is None:
        return False
    verdict = result if isinstance(result, str) else result.get("verdict")
    return verdict == COLLISION


def verdict_counts(results: Iterable[dict | str]) -> dict[str, int]:
    """Counts per verdict. Counts, not a rate - see C1 / requirement 5."""
    counts = {verdict: 0 for verdict in VERDICTS}
    for item in results:
        verdict = item if isinstance(item, str) else (item or {}).get("verdict")
        if verdict in counts:
            counts[verdict] += 1
    return counts


def save_report(
    result: dict,
    out_dir: str | os.PathLike[str],
    *,
    card_id: str | None = None,
) -> str:
    """Write ``<out_dir>/<card_id>.json`` (spec sec. 1.3). Returns the path."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    name = card_id or str(result.get("card_id") or "unknown")
    path = directory / f"{name}.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, sort_keys=True)
    return str(path)


# --------------------------------------------------------------------------
# Production wiring (kept out of the default path so tests stay offline)
# --------------------------------------------------------------------------


@dataclass
class _PriorArtContext:
    """The minimal surface BaseAgent reads off a pipeline context.

    Mirrors ``generate._GeneratorContext`` deliberately; both exist so a
    BaseAgent can be built without a live pipeline.
    """

    dataset_name: str
    task_type: str = "prediction"
    output_dir: str | None = None
    revision_cycle: int = 0
    log: list = field(default_factory=list)


class _NoExecutor:
    def run(self, *args: Any, **kwargs: Any) -> dict:
        raise RuntimeError(
            "The prior-art veto does not execute code. If this was reached, "
            "something routed a sandbox job to the wrong agent."
        )


def resolve_priorart_model(config: dict) -> str | None:
    """``ideation.models.priorart``, else ``ideation.models.judge``, else None.

    None leaves BaseAgent's per-stage resolution alone. No model ID is
    hardcoded here. The spec's cost table puts this stage on the cheap
    tier, which is why the judge model is the fallback.
    """
    ideation = (config or {}).get("ideation") or {}
    models = ideation.get("models") or {}
    model = models.get("priorart") or models.get("judge")
    return str(model) if model else None


class PriorArtAgent:
    """Thin BaseAgent wrapper for the delta sentence. Never asked for a
    verdict. Constructed lazily so importing this module touches no
    provider SDK and needs no API key."""

    AGENT_NAME = AGENT_KEY

    def __init__(
        self,
        config: dict,
        *,
        dataset: str,
        task_type: str = "prediction",
        output_dir: str | None = None,
        model: str | None = None,
    ) -> None:
        from src.agents.base import BaseAgent

        class _Agent(BaseAgent):  # BaseAgent.run is abstract
            def run(self, **kwargs: Any) -> dict:
                raise NotImplementedError(
                    "The prior-art veto is driven by "
                    "src.ideation.priorart.collision_check, not by a "
                    "pipeline stage runner."
                )

        context = _PriorArtContext(
            dataset_name=dataset, task_type=task_type, output_dir=output_dir
        )
        self.agent = _Agent(
            context, self.AGENT_NAME, config, executor=_NoExecutor()
        )
        if model:
            self.agent.model = model
        self.model = self.agent.model

    def __call__(self, user_message: str) -> str:
        return self.agent.call_llm(user_message, max_tokens=256)


def make_delta_writer(
    config: dict,
    *,
    dataset: str,
    task_type: str = "prediction",
    output_dir: str | None = None,
) -> tuple[Callable[[str], str], str]:
    """``(caller, model_id)`` routed through ``BaseAgent.call_llm``."""
    agent = PriorArtAgent(
        config,
        dataset=dataset,
        task_type=task_type,
        output_dir=output_dir,
        model=resolve_priorart_model(config),
    )
    return agent, agent.model


def problem_formulator_retriever(
    config: dict,
    *,
    dataset: str,
    task_type: str = "prediction",
    output_dir: str | None = None,
) -> Retriever:
    """Retrieval through the EXISTING literature path, not a new one.

    Wraps ``ProblemFormulator._search_literature`` - the shipped S2 +
    arXiv + title-Jaccard-dedup pipeline, including its retry/backoff,
    seminal-query reservation and year windowing. Nothing about
    retrieval is reimplemented here.

    Reusability note (hand-off): ``_search_literature`` is an *instance
    method* on a ``BaseAgent`` subclass and takes a single free-text
    prompt, so using it requires constructing a ProblemFormulator (which
    builds a provider client and therefore needs the provider's API key
    in the environment) and flattening the facet queries into one
    string. That works, and it is what this adapter does. A module-level
    extraction would be cleaner - see the hand-off report; it is NOT
    performed here because ``src/agents/problem_formulator.py`` belongs
    to another slice.

    The returned callable never raises: on any failure it returns ``[]``,
    which ``collision_check`` turns into UNVERIFIABLE.
    """

    def _retrieve(queries: list[str]) -> list[dict]:
        try:
            from src.agents.problem_formulator import ProblemFormulator

            context = _PriorArtContext(
                dataset_name=dataset,
                task_type=task_type,
                output_dir=output_dir,
            )
            agent = ProblemFormulator(
                context, "ProblemFormulator", config, executor=_NoExecutor()
            )
            seed = "; ".join(queries) if queries else ""
            result = agent._search_literature(seed or None) or {}
            papers = [p for p in (result.get("papers") or []) if isinstance(p, dict)]
            for paper in papers:
                paper.setdefault("source", "s2_arxiv")
            return papers
        except Exception:  # noqa: BLE001 - never kill an idea on an outage
            return []

    return _retrieve
