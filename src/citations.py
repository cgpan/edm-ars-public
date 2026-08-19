"""Arc P3 — deterministic citation reconciliation and literature-pool depth.

Three problems this module fixes, all found by the Arc P1 linter on real
shipped manuscripts:

1. **Bib drift** (F-E2A-SECTIONWISE-BIB-DRIFT). Journal manuscripts are
   written section by section; each section cites freely, while
   ``references.bib`` came from a separate LLM call that never saw the
   sections. Result: 22 cited keys with no bib entry, rendering as
   ``[?]`` in the PDF.
2. **Fabricated venues**. ``venue`` was never requested from Semantic
   Scholar, so every non-arXiv entry was stamped
   ``@inproceedings ... booktitle = {Proceedings of the Educational Data
   Mining Conference}`` — 29 such entries across shipped papers.
3. **Citation depth**. The ProblemFormulator retrieves ~100 papers and
   persists only the 8-12 the LLM echoes, so manuscripts carry 4-26
   references against venue norms of 15 (EDM) / 47 (JLA) / 54 (JEDM).

Everything here is deterministic: no LLM calls, no network. That is
deliberate — a reference list is a factual artifact, and the generative
path is exactly what produced the fabrications above.
"""
from __future__ import annotations

import math
import re
import statistics
from datetime import datetime
from typing import Any, Iterable, Optional

# Reuse the linter's extraction primitives so the producer and the
# checker can never disagree about what counts as a citation.
from src.manuscript_linter import _BIB_ENTRY, _CITE_CMD, cited_keys

# Written verbatim when there is no real literature to cite; pinned by
# tests/test_writer.py (S2-failure path must stay honest, not invent).
# Kept byte-identical to writer._S2_FAILURE_BIB_COMMENT.
S2_FAILURE_BIB_COMMENT = (
    "% Semantic Scholar API was unavailable; citations are placeholders only.\n"
)


# ---------------------------------------------------------------------------
# BibTeX construction from real retrieved metadata
# ---------------------------------------------------------------------------


def sanitize_key(raw_id: str) -> str:
    """BibTeX-safe citation key (``arxiv:2401.1`` -> ``arxiv_2401.1``)."""
    return (raw_id or "unknown").replace(":", "_")


# --- venue classification --------------------------------------------------
#
# F-P5-VENUE-TYPE-MISLABEL: the previous heuristic asked only whether the
# venue string contained one of five journal words, so every journal whose
# title lacks them was stamped ``@inproceedings ... booktitle = {PLoS ONE}``
# and rendered as "In: Proceedings of PLoS ONE." — 23 of 34 @inproceedings
# entries on a shipped manuscript. The cascade below prefers explicit
# metadata (Semantic Scholar ``publicationTypes``), then unambiguous venue
# wording, then the DOI registrant, and falls back to ``@misc`` rather than
# claiming a proceedings that may not exist.

_CONFERENCE_VENUE_RE = re.compile(
    r"(?:\bproceedings\b|\bproc\.|\bconference\b|\bconf\.|\bworkshop\b|"
    r"\bsymposium\b|\bannual meeting\b|\bcongress\b|\bcolloquium\b)",
    re.IGNORECASE,
)
_JOURNAL_VENUE_RE = re.compile(
    r"(?:\bjournal\b|\bjournals\b|(?:^|\s)j\.|\btransactions\b|\breviews?\b|"
    r"\bquarterly\b|\bannals\b|\bbulletin\b|\bletters\b|\bacta\b|"
    r"\barchives\b|\bmagazine\b)",
    re.IGNORECASE,
)

# S2 ``publicationTypes`` values that identify a serial article.
_JOURNAL_PUB_TYPES = frozenset({"journalarticle", "review", "metaanalysis"})
_CONFERENCE_PUB_TYPES = frozenset({"conference"})
_BOOK_PUB_TYPES = frozenset({"book", "booksection"})

# DOI registrants that publish journals only (or effectively only). ACM
# (10.1145), IEEE (10.1109) and ACL (10.18653) are deliberately ABSENT:
# they register both proceedings and journals, so the prefix proves
# nothing and guessing from it is what produced the defect.
_JOURNAL_DOI_PREFIXES = frozenset({
    "10.1001", "10.1002", "10.1016", "10.1017", "10.1037", "10.1057",
    "10.1073", "10.1080", "10.1093", "10.1097", "10.1098", "10.1108",
    "10.1111", "10.1126", "10.1136", "10.1155", "10.1158", "10.1177",
    "10.1186", "10.1207", "10.1371", "10.1590", "10.18608", "10.2307",
    "10.3102", "10.3389", "10.3390", "10.5951",
})
# Nature Portfolio: journals (Scientific Reports, npj *) share 10.1038.
_JOURNAL_DOI_PREFIXES = _JOURNAL_DOI_PREFIXES | {"10.1038"}
# Springer/Kluwer register journals as 10.1007/s… and 10.1023/A…; their
# conference proceedings (LNCS) are ISBN-derived, 10.1007/978-… .
_SPRINGER_JOURNAL_PREFIXES = ("10.1007", "10.1023")


def normalize_doi(raw: Any) -> str:
    """Bare DOI (``10.x/y``) from whatever form the record carries."""
    doi = str(raw or "").strip()
    if not doi:
        return ""
    doi = re.sub(r"^(?:https?://)?(?:dx\.)?doi\.org/", "", doi, flags=re.I)
    doi = re.sub(r"^doi:\s*", "", doi, flags=re.I)
    doi = doi.strip().rstrip(".")
    return doi if doi.lower().startswith("10.") else ""


def _doi_suggests_journal(doi: str) -> bool:
    if not doi:
        return False
    prefix = doi.split("/", 1)[0]
    if prefix in _JOURNAL_DOI_PREFIXES:
        return True
    if prefix in _SPRINGER_JOURNAL_PREFIXES:
        suffix = doi.split("/", 1)[1] if "/" in doi else ""
        # 10.1007/s11162-… journal article; 10.1007/978-3-… book/proceedings.
        return bool(re.match(r"^[sA-Z]\d", suffix))
    return False


def classify_entry(paper: dict) -> tuple[str, str, str]:
    """``(entry_type, venue_field, venue_value)`` for one record.

    Never guesses a proceedings name. When nothing on the record
    identifies the publication channel, the venue is reported verbatim on
    an ``@misc`` entry — naming where the work appeared without asserting
    what kind of thing it is.
    """
    raw_id = str(paper.get("paperId") or "unknown")
    venue = str(paper.get("venue") or "").strip()
    is_arxiv = (
        raw_id.startswith("arxiv:")
        or raw_id.startswith("arxiv_")
        or paper.get("source") == "arxiv"
    )
    if is_arxiv:
        return "misc", "note", "arXiv preprint"
    if not venue:
        # No venue metadata: say so rather than inventing one.
        return "misc", "note", "Venue metadata unavailable"

    types = {
        str(t).strip().lower().replace(" ", "")
        for t in (paper.get("publicationTypes") or [])
        if str(t).strip()
    }
    venue_says_conference = bool(_CONFERENCE_VENUE_RE.search(venue))

    # 1. Explicit metadata first.
    if types & _CONFERENCE_PUB_TYPES and types & _JOURNAL_PUB_TYPES:
        # S2 tags both (journal special issues of a conference). The venue
        # string is the tie-breaker; default to the journal reading.
        if venue_says_conference:
            return "inproceedings", "booktitle", venue
        return "article", "journal", venue
    if types & _CONFERENCE_PUB_TYPES:
        return "inproceedings", "booktitle", venue
    if types & _JOURNAL_PUB_TYPES:
        return "article", "journal", venue
    if types & _BOOK_PUB_TYPES:
        # A book or chapter is neither; name the venue, claim nothing.
        return "misc", "note", venue

    # 2. Unambiguous venue wording.
    if venue_says_conference:
        return "inproceedings", "booktitle", venue
    if _JOURNAL_VENUE_RE.search(venue):
        return "article", "journal", venue

    # 3. DOI registrant (journal-only registrants).
    if _doi_suggests_journal(normalize_doi(paper.get("doi"))):
        return "article", "journal", venue

    # 4. Unknown channel: @misc rather than a fabricated proceedings.
    return "misc", "note", venue


def build_bib_entry(paper: dict) -> str:
    """Render ONE BibTeX entry from a retrieved paper record.

    Honesty rule: when the venue is genuinely unknown we emit ``@misc``
    with an explicit note. We never guess a proceedings name — a
    fabricated venue is a citation-integrity defect that survives into
    the PDF and misleads readers and reviewers alike.

    The DOI is emitted whenever the record carries one (F-P5-BIB-NO-DOI:
    0 of 62 shipped entries had a DOI against 61 of 62 source records
    that did). A ``url`` is emitted only when the record supplies one and
    there is no DOI — never synthesised.
    """
    key = sanitize_key(str(paper.get("paperId") or "unknown"))
    title = paper.get("title") or "Unknown Title"
    year = paper.get("year") or ""
    authors = paper.get("authors") or []
    author_str = " and ".join(authors) if authors else "Unknown Author"
    entry_type, venue_key, venue_val = classify_entry(paper)

    lines = [
        f"@{entry_type}{{{key},",
        f"  author    = {{{author_str}}},",
        f"  title     = {{{title}}},",
        f"  year      = {{{year}}},",
        f"  {venue_key} = {{{venue_val}}},",
    ]
    doi = normalize_doi(paper.get("doi"))
    if doi:
        lines.append(f"  doi       = {{{doi}}},")
    else:
        url = str(paper.get("url") or "").strip()
        if url.startswith("http"):
            lines.append(f"  url       = {{{url}}},")
    lines.append("}")
    return "\n".join(lines)


def build_bibtex(papers: Optional[Iterable[dict]]) -> str:
    """Render a full ``references.bib`` from retrieved papers."""
    entries = [build_bib_entry(p) for p in (papers or [])]
    if not entries:
        return S2_FAILURE_BIB_COMMENT
    return "\n\n".join(entries) + "\n"


# ---------------------------------------------------------------------------
# Pool expansion (citation depth)
# ---------------------------------------------------------------------------


def _title_tokens(title: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", (title or "").lower()) if len(t) > 2}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# --- recency-aware composition (F-P5-DEPTH-RECENCY-SKEW) -------------------
#
# Arc P3 raised HOW MANY references reach the manuscript without
# constraining WHICH, and the retrieval pool was sorted year-descending,
# so a shipped bibliography had 62/62 references from 2024-2026. A
# reference list is a *composition* with a target age distribution, not a
# filter over individual papers. Bins and profiles are measured on 34 LSAR
# anchors / 1,769 dated references (docs/f_p5_citation_recency_spec.md §1).
#
# The bins live in ``manuscript_linter`` (docs §4) so the producer and the
# checker cannot disagree about what "old" means — the same rule that
# already governs ``_BIB_ENTRY``/``_CITE_CMD``. The fallback exists only so
# this module still imports against a linter that predates the recency
# block; the values are identical and a test pins that they agree.
try:
    from src.manuscript_linter import (  # noqa: F401
        AGE_BUCKETS,
        DEFAULT_REF_AGE_PROFILE,
        FILL_ORDER,
        bucket_of_age,
    )
except ImportError:  # pragma: no cover - linter recency block not present
    AGE_BUCKETS = ("le2", "3_5", "6_10", "11_20", "gt20")  # type: ignore[no-redef]
    # Scarcest bin first: a tight budget must protect the historical tail
    # rather than the abundant new work.
    FILL_ORDER = ("gt20", "11_20", "6_10", "3_5", "le2")  # type: ignore[no-redef]
    DEFAULT_REF_AGE_PROFILE = {  # type: ignore[no-redef]
        "le2": 0.246, "3_5": 0.237, "6_10": 0.241, "11_20": 0.170, "gt20": 0.106,
    }

    def bucket_of_age(age: int) -> str:  # type: ignore[no-redef]
        """Age (years) -> fixed bin id."""
        if age <= 2:
            return "le2"
        if age <= 5:
            return "3_5"
        if age <= 10:
            return "6_10"
        if age <= 20:
            return "11_20"
        return "gt20"


# Producer-side targets, measured on the same 34 anchors (docs §1.1). The
# linter's ``venue_age_profile`` is the CHECKER-side lookup: it returns
# (profile, tolerance, source) and, until venue_norms.yaml gains ``ref_age``
# blocks (P5.1), always answers with the pooled default. Both read the same
# YAML block once it exists, so they converge; until then every per-venue
# value below sits inside the linter's ±10-12 pp tolerance.
VENUE_REF_AGE_PROFILES: dict[str, dict[str, float]] = {
    "EDM": {"le2": 0.239, "3_5": 0.239, "6_10": 0.236, "11_20": 0.185, "gt20": 0.100},
    "JEDM": {"le2": 0.312, "3_5": 0.235, "6_10": 0.190, "11_20": 0.148, "gt20": 0.116},
    "JLA": {"le2": 0.169, "3_5": 0.238, "6_10": 0.314, "11_20": 0.180, "gt20": 0.099},
}

# Topicality signals. A pool assembled from broad keyword queries picks up
# clinical-medicine hits (six oncology/cardiology papers reached a shipped
# bibliography); those are rejected outright rather than merely ranked low.
ONTOPIC_VENUE_TERMS = frozenset({
    "educational data mining", "learning analytics",
    "artificial intelligence in education", "computers & education",
    "computers and education", "educational technology", "learning at scale",
    "user modeling", "intelligent tutoring", "educational measurement",
    "psychometrika", "applied psychological measurement",
    "educational psychology", "educational researcher",
    "review of educational research", "learning technologies", "aied",
    "lak", "edm", "jedm", "jla",
})
OFFTOPIC_VENUE_TERMS = frozenset({
    "oncology", "cancer", "cardiovascular", "cardiology", "clinical",
    "diabetes", "transplant", "surgery", "radiology", "nursing",
    "psychiatry", "npj", "lancet", "bmj",
})
ONTOPIC_FIELDS = frozenset({
    "education", "computer science", "psychology", "mathematics",
    "sociology", "linguistics", "economics",
})


def composition_age_profile(
    venue: Optional[str], norms: Optional[dict] = None
) -> dict[str, float]:
    """Target age distribution for ``venue``; pooled default when unknown.

    This is the **producer** side of the recency contract — the thing
    :func:`expand_literature_pool` aims at — deliberately named apart from
    the linter's checker-side ``venue_age_profile`` because that one
    returns ``(profile, tolerance, source)``.

    Prefers a mined ``ref_age.buckets`` block from ``venue_norms.yaml``
    when one exists (P5.1 writes it); otherwise falls back to the values
    pinned in docs §1.1.
    """
    block = ((norms or {}).get(venue or "") or {}).get("ref_age") or {}
    buckets = block.get("buckets") or {}
    if isinstance(buckets, dict) and all(b in buckets for b in AGE_BUCKETS):
        return {b: float(buckets[b]) for b in AGE_BUCKETS}
    return dict(VENUE_REF_AGE_PROFILES.get(venue or "", DEFAULT_REF_AGE_PROFILE))


def allocate_age_slots(target: int, profile: dict[str, float]) -> dict[str, int]:
    """Largest-remainder slot allocation; ties go to the OLDER bin.

    ``sum(result) == target`` for every non-negative ``target``.
    """
    if target <= 0:
        return {b: 0 for b in AGE_BUCKETS}
    exact = {b: target * float(profile.get(b, 0.0)) for b in AGE_BUCKETS}
    slots = {b: int(exact[b]) for b in AGE_BUCKETS}
    short = target - sum(slots.values())
    order = sorted(
        AGE_BUCKETS,
        key=lambda b: (-(exact[b] - int(exact[b])), -AGE_BUCKETS.index(b)),
    )
    for b in order[:max(0, short)]:
        slots[b] += 1
    return slots


def _year_of(paper: dict) -> Optional[int]:
    raw = paper.get("year")
    if raw is None or isinstance(raw, bool):
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        match = re.search(r"(?:19|20)\d{2}", str(raw))
        return int(match.group(0)) if match else None


def _topicality(paper: dict) -> float:
    """``-1.0`` rejects the record; otherwise a small on-topic bonus."""
    fields = [str(f).strip().lower() for f in (paper.get("fieldsOfStudy") or [])]
    if fields and not any(f in ONTOPIC_FIELDS for f in fields):
        return -1.0
    venue = (paper.get("venue") or "").lower()
    on = any(term in venue for term in ONTOPIC_VENUE_TERMS)
    off = any(term in venue for term in OFFTOPIC_VENUE_TERMS)
    if off and not on:
        return -1.0
    return 0.25 if on else 0.0


def _retrieval_rank(paper: dict) -> Optional[int]:
    rank = paper.get("retrieval_rank")
    if isinstance(rank, bool) or not isinstance(rank, (int, float)):
        return None
    return int(rank)


def rank_pool(
    pool: Optional[Iterable[dict]], now_year: int
) -> tuple[list[dict], list[str]]:
    """Score pool records for the quota fill. Returns ``(scored, degraded)``.

    Every signal degrades independently, because pools persisted before
    Arc P5 carry only seven keys:

    * ``retrieval_rank`` absent pool-wide -> constant relevance, the term
      drops out of the ordering;
    * citation counts absent for one record (arXiv) -> the pool MEDIAN,
      never 0, since 0 systematically ranks every preprint last; absent
      pool-wide -> constant, the term drops out;
    * ``fieldsOfStudy``/``venue`` absent -> neutral topicality, not
      rejection;
    * no ``year`` -> ineligible (an undated reference is not citable).

    Scored records are COPIES carrying ``_score``/``_bucket``/``_age`` and
    an ``_orig`` handle on the untouched pool record.
    """
    pool_list = list(pool or [])
    degraded: list[str] = []
    if not any(_retrieval_rank(p) is not None for p in pool_list):
        degraded.append("retrieval_rank")

    raw_influence: list[Optional[float]] = []
    for paper in pool_list:
        influence = paper.get("influentialCitationCount")
        if influence is None and paper.get("citationCount") is not None:
            try:
                influence = float(paper["citationCount"]) / 10.0
            except (TypeError, ValueError):
                influence = None
        try:
            raw_influence.append(
                math.log1p(float(influence)) if influence is not None else None
            )
        except (TypeError, ValueError):
            raw_influence.append(None)
    known = [v for v in raw_influence if v is not None]
    if known:
        median_influence = statistics.median(known)
        hi = max(known) or 1.0
    else:
        degraded.append("influentialCitationCount")
        median_influence, hi = 0.5, 1.0

    scored: list[dict] = []
    for index, paper in enumerate(pool_list):
        year = _year_of(paper)
        if not paper.get("title") or year is None:
            continue
        topic = _topicality(paper)
        if topic < 0:
            continue
        rank = _retrieval_rank(paper)
        relevance = 60.0 / (60 + rank) if rank is not None else 0.5
        if known:
            value = raw_influence[index]
            influence = (value if value is not None else median_influence) / hi
        else:
            influence = 0.5
        age = max(0, now_year - year)
        scored.append({
            **paper,
            "_orig": paper,
            "_age": age,
            "_bucket": bucket_of_age(age),
            "_rel": relevance,
            "_infl": influence,
            "_topic": topic,
            "_rank": rank,
            # Weights sum to 1.0. `+ topic` (not `* topic`) was a typo:
            # _topicality returns 0.25, so the "small on-topic bonus"
            # its own docstring promises was landing at 6.7x intent —
            # ~72% of the whole relevance range, enough for a mediocre
            # paper with an on-topic venue to outrank a highly relevant
            # one. As a weighted term it is the tiebreaker intended.
            "_score": 0.55 * relevance + 0.30 * influence + 0.15 * topic,
        })
    return scored, degraded


def _sort_key(paper: dict) -> tuple:
    """Total order -> no ties -> deterministic under input permutation."""
    rank = paper.get("_rank")
    return (
        -round(float(paper["_score"]), 9),
        -round(float(paper["_infl"]), 9),
        rank if rank is not None else 10 ** 6,
        -int(paper["_age"]),                      # prefer the OLDER paper
        str(paper.get("paperId") or ""),
        str(paper.get("title") or ""),
    )


def _place_selected(
    selected: Optional[Iterable[dict]], pool: list[dict]
) -> tuple[list[dict], set[str], list[set[str]]]:
    """Selected block, unchanged in order, with pool metadata substituted."""
    pool_by_key: dict[str, dict] = {}
    for paper in pool:
        pool_by_key.setdefault(sanitize_key(paper.get("paperId", "")), paper)

    out: list[dict] = []
    seen_keys: set[str] = set()
    seen_titles: list[set[str]] = []
    for paper in selected or []:
        key = sanitize_key(paper.get("paperId", ""))
        authoritative = pool_by_key.get(key)
        out.append(authoritative if authoritative is not None else paper)
        seen_keys.add(key)
        # Dedup against BOTH titles when a record is substituted: the echo
        # and the pool record describe the same work, so either spelling
        # must still block a near-duplicate later in the pool.
        seen_titles.append(_title_tokens(paper.get("title", "")))
        if authoritative is not None:
            seen_titles.append(_title_tokens(authoritative.get("title", "")))
    return out, seen_keys, seen_titles


def expand_literature_pool(
    selected: Optional[Iterable[dict]],
    pool: Optional[Iterable[dict]],
    target: int,
    dedup_threshold: float = 0.80,
    profile: Optional[dict[str, float]] = None,
    now_year: Optional[int] = None,
    stats: Optional[dict[str, Any]] = None,
) -> list[dict]:
    """Top the LLM-selected papers back up from the retrieved pool.

    ``selected`` (the 8-12 the ProblemFormulator kept, most relevant,
    order preserved) come first; pool papers are appended until
    ``target`` is reached. Dedup is by sanitized paperId, then by title
    Jaccard so the same work retrieved from both S2 and arXiv is not
    cited twice.

    **Metadata always comes from the pool, never from the echo.**
    ``selected`` is the model's *transcription* of records it was shown;
    ``pool`` is what Semantic Scholar actually returned. A selected paper
    whose id is in the pool is therefore emitted as the POOL record —
    only its *position* is taken from ``selected``, because that ordering
    is the LLM-judged relevance we want to keep. Measured on
    ``runs/arc_p_validation_20260711``: the 10 echoed papers carried 0/10
    DOIs against 97/100 in the pool, and any dropped ``venue`` silently
    downgrades a real reference to ``@misc / Venue metadata unavailable``.
    Papers absent from the pool (title-only verification matches) pass
    through untouched.

    With ``profile`` supplied (see :func:`composition_age_profile`) the append
    path becomes a **composition**: slots are allocated across the five
    age bins by largest remainder, the papers already selected consume
    their own bin's slots, and the remainder is filled oldest-bin-first
    from the ranked pool. Unfillable quota spills — again oldest-first —
    so a pool with no historical work still returns a full list instead of
    crashing or looping. ``profile=None`` keeps the exact legacy
    behaviour, so a caller can opt out and old snapshots reproduce.

    Pure set arithmetic over already-retrieved records — no network.
    """
    pool_list = list(pool or [])
    out, seen_keys, seen_titles = _place_selected(selected, pool_list)
    n_selected = len(out)

    def _finish(result: list[dict], degraded: list[str], rejected: int) -> list[dict]:
        if stats is not None:
            years = [y for y in (_year_of(p) for p in pool_list) if y is not None]
            achieved: dict[str, int] = {b: 0 for b in AGE_BUCKETS}
            if now_year_value is not None:
                for paper in result:
                    year = _year_of(paper)
                    if year is not None:
                        achieved[bucket_of_age(max(0, now_year_value - year))] += 1
            stats.update({
                "target": target,
                "n_selected": n_selected,
                "n_appended": len(result) - n_selected,
                "n_total": len(result),
                "degraded_signals": degraded,
                "n_offtopic_rejected": rejected,
                "pool_n": len(pool_list),
                "pool_year_min": min(years) if years else None,
                "pool_year_max": max(years) if years else None,
                "achieved_buckets": achieved if now_year_value is not None else None,
                "target_slots": target_slots,
                "now_year": now_year_value,
            })
        return result

    now_year_value: Optional[int] = None
    target_slots: Optional[dict[str, int]] = None

    if target <= len(out) or profile is None:
        if profile is not None:
            now_year_value = int(now_year) if now_year else datetime.utcnow().year
            target_slots = allocate_age_slots(target, profile)
        if target <= len(out):
            return _finish(out, [], 0)

        # Legacy path: relevance order, no age composition.
        for paper in pool_list:
            if len(out) >= target:
                break
            key = sanitize_key(paper.get("paperId", ""))
            if key in seen_keys:
                continue
            toks = _title_tokens(paper.get("title", ""))
            if any(_jaccard(toks, prev) >= dedup_threshold for prev in seen_titles):
                continue
            # A citable reference needs at minimum a title and a year.
            if not paper.get("title") or not paper.get("year"):
                continue
            out.append(paper)
            seen_keys.add(key)
            seen_titles.append(toks)
        return _finish(out, [], 0)

    # --- composition path --------------------------------------------------
    now_year_value = int(now_year) if now_year else datetime.utcnow().year
    slots = allocate_age_slots(target, profile)
    target_slots = dict(slots)
    for paper in out:
        year = _year_of(paper)
        if year is not None:
            bucket = bucket_of_age(max(0, now_year_value - year))
            slots[bucket] = max(0, slots[bucket] - 1)

    ranked, degraded = rank_pool(pool_list, now_year_value)
    rejected = sum(1 for p in pool_list if _topicality(p) < 0)

    by_bucket: dict[str, list[dict]] = {b: [] for b in AGE_BUCKETS}
    for paper in sorted(ranked, key=_sort_key):
        key = sanitize_key(paper.get("paperId", ""))
        if key in seen_keys:
            continue
        toks = _title_tokens(paper.get("title", ""))
        if any(_jaccard(toks, prev) >= dedup_threshold for prev in seen_titles):
            continue
        seen_keys.add(key)
        seen_titles.append(toks)
        by_bucket[paper["_bucket"]].append(paper)

    budget = target - len(out)
    appended: list[dict] = []
    # PASS 1 — honour quotas, OLDEST BIN FIRST so a tight budget protects
    # the scarce historical tail rather than the abundant new work.
    for bucket in FILL_ORDER:
        take = min(slots[bucket], budget - len(appended), len(by_bucket[bucket]))
        if take > 0:
            appended += by_bucket[bucket][:take]
            by_bucket[bucket] = by_bucket[bucket][take:]
    # PASS 2 — spill unfilled quota, again OLDEST FIRST. This is the rule
    # that stops a pool with no old work from silently re-skewing to le2.
    for bucket in FILL_ORDER:
        take = min(budget - len(appended), len(by_bucket[bucket]))
        if take > 0:
            appended += by_bucket[bucket][:take]
            by_bucket[bucket] = by_bucket[bucket][take:]

    # Emit the untouched pool records, not the scored copies.
    return _finish(out + [p["_orig"] for p in appended], degraded, rejected)


def venue_citation_target(
    venue: Optional[str],
    norms: Optional[dict] = None,
    statistic: str = "median",
) -> Optional[int]:
    """Reference-count target for a venue, from the mined anchor norms.

    Defaults to the anchor MEDIAN (not P25) so the linter's P25 floor has
    headroom — aiming exactly at the floor means half of runs land under it.
    """
    if not venue:
        return None
    if norms is None:
        from src.manuscript_linter import load_venue_norms

        norms = load_venue_norms()
    block = (norms or {}).get(venue) or {}
    refs = block.get("refs") or {}
    value = refs.get(statistic)
    return int(round(value)) if isinstance(value, (int, float)) else None


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------


def _rewrite_cite_commands(tex: str, drop: set[str]) -> tuple[str, int]:
    """Remove ``drop`` keys from every citation command in ``tex``.

    Multi-key commands keep their surviving keys; a command whose keys
    are all dropped is removed entirely, and a resulting space before
    punctuation is collapsed so the sentence still reads cleanly.
    """
    n_removed = 0

    def repl(match: re.Match) -> str:
        nonlocal n_removed
        whole = match.group(0)
        keys = [k.strip() for k in match.group(1).split(",") if k.strip()]
        kept = [k for k in keys if k not in drop]
        if len(kept) == len(keys):
            return whole
        n_removed += len(keys) - len(kept)
        if not kept:
            return ""
        return whole.replace(match.group(1), ", ".join(kept))

    out = _CITE_CMD.sub(repl, tex)
    if n_removed:
        out = re.sub(r"[ \t]+([,.;:)])", r"\1", out)
        out = re.sub(r"\(\s*\)", "", out)
    return out, n_removed


def reconcile_citations(
    paper_tex: str,
    bibtex: str,
    papers: Optional[Iterable[dict]] = None,
) -> tuple[str, str, dict[str, Any]]:
    """Make ``references.bib`` and the prose agree. Deterministic.

    Policy:

    * cited key present in the retrieved pool but missing from the bib
      -> **append** the real entry (never fabricate);
    * cited key in neither -> **strip** that key from the citation
      command (the model invented it);
    * ``placeholder*`` keys, when real papers exist -> strip;
    * no real papers at all -> return unchanged, preserving the honest
      S2-failure comment.

    Returns ``(tex, bib, stats)``.
    """
    papers = list(papers or [])
    stats: dict[str, Any] = {
        "cited": 0, "backfilled": 0, "stripped": 0,
        "invented_keys": [], "bib_entries": 0, "skipped": None,
    }
    if not paper_tex:
        stats["skipped"] = "no paper text"
        return paper_tex, bibtex, stats

    cited = cited_keys(paper_tex)
    stats["cited"] = len(cited)

    if not papers:
        # S2 failure path: nothing verified to reconcile against. Leave
        # the manuscript and its honest failure comment alone.
        stats["skipped"] = "no retrieved papers"
        stats["bib_entries"] = len(set(_BIB_ENTRY.findall(bibtex or "")))
        return paper_tex, bibtex, stats

    pool_by_key = {sanitize_key(p.get("paperId", "")): p for p in papers}
    bib_keys = set(_BIB_ENTRY.findall(bibtex or ""))

    missing = {k for k in cited if k not in bib_keys}
    backfill = sorted(k for k in missing if k in pool_by_key)
    invented = sorted(
        k for k in missing
        if k not in pool_by_key or k.startswith("placeholder")
    )
    invented += sorted(
        k for k in cited
        if k.startswith("placeholder") and k not in invented
    )
    invented = sorted(set(invented))

    # Guard: never strip every citation out of a manuscript that has real
    # literature available — an empty bibliography is a worse defect than
    # a dangling key, and it would silently gut the Related Work section.
    survivors = cited - set(invented)
    if not survivors and not backfill:
        stats["skipped"] = "would remove all citations"
        return paper_tex, bibtex, stats

    out_tex = paper_tex
    if invented:
        out_tex, n_removed = _rewrite_cite_commands(out_tex, set(invented))
        stats["stripped"] = n_removed
        stats["invented_keys"] = invented

    out_bib = bibtex or ""
    if backfill:
        new_entries = [build_bib_entry(pool_by_key[k]) for k in backfill]
        joiner = "" if out_bib.endswith("\n") else "\n"
        out_bib = out_bib + joiner + "\n" + "\n\n".join(new_entries) + "\n"
        stats["backfilled"] = len(new_entries)

    stats["bib_entries"] = len(set(_BIB_ENTRY.findall(out_bib)))
    return out_tex, out_bib, stats


def format_citation_key_block(
    papers: Optional[Iterable[dict]],
    target: Optional[int] = None,
    max_listed: int = 120,
) -> str:
    """Prompt block enumerating the ONLY citation keys the model may use.

    The sectionwise writer previously received the literature context as
    a JSON blob with no statement of which keys were legal, which is how
    invented keys entered the prose in the first place.
    """
    papers = list(papers or [])
    if not papers:
        return ""
    lines = [
        "## Available Citation Keys",
        "",
        "Cite ONLY these keys. Never invent a citation key: any key not "
        "listed here is deleted from the manuscript automatically, taking "
        "your sentence's support with it.",
    ]
    if target:
        lines.append(
            f"Target for this venue: cite at least {target} distinct keys "
            "across the manuscript, concentrated in Introduction, Related "
            "Work, and Discussion."
        )
    lines.append("")
    for p in papers[:max_listed]:
        key = sanitize_key(p.get("paperId", ""))
        year = p.get("year") or "n.d."
        title = (p.get("title") or "")[:95]
        lines.append(f"- {key} | {year} | {title}")
    if len(papers) > max_listed:
        lines.append(f"- ... and {len(papers) - max_listed} more in the JSON above")
    lines.append("")
    return "\n".join(lines)
