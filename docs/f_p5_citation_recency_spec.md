<!-- Generated 2026-07-11 by the citation-recency-profile workflow.
     Measured on 34 LSAR anchors / 1,769 dated references.
     Tracks backlog G1 (F-P5-DEPTH-RECENCY-SKEW). -->

# F-P5-DEPTH-RECENCY-SKEW — Implementation Specification

**Status:** ready to implement. No file in the worktree was modified while writing this (read-only analysis; written from a scratch directory).

**Defect restated in one line:** Arc P3 raised *how many* references reach the manuscript without constraining *which*, and the pool it draws from is sorted year-descending at `problem_formulator.py:578` and `:728`, so the count metric (62/62 available refs) went green while every reference in the shipped bibliography came from 2024–2026.

**Design principle for the whole fix:** the reference list is a *composition* with a target distribution, not a filter over individual papers. Every gate below is a bin-membership gate. Padding with arbitrary old papers is explicitly not the goal — the retrieval change exists so that *relevant* old work is reachable, and the ranking change exists so it is chosen on relevance+influence within its age bin.

---

## 1. TARGET PROFILE

Age is defined as `age = manuscript_year − reference_year`, clamped at 0. `manuscript_year` is the year the paper is written (`datetime.utcnow().year` at lint time), injectable as `now_year` for tests.

Five bins, fixed everywhere in the system (linter, citations module, miner, harness):

| bin id | age range |
|---|---|
| `le2` | 0–2 y |
| `3_5` | 3–5 y |
| `6_10` | 6–10 y |
| `11_20` | 11–20 y |
| `gt20` | > 20 y |

### 1.1 Per-venue targets

Measured on 34 LSAR anchors / 1,769 dated references, with anchor publication years corrected to EDM=2024, JEDM=2026, JLA=2026 (JEDM running footer "Volume 18, No 1, 2026"; JLA self-DOI `10.18608/jla.2026.<id>`).

| venue | `le2` | `3_5` | `6_10` | `11_20` | `gt20` | median age | tolerance | n anchors / n refs |
|---|---|---|---|---|---|---|---|---|
| **EDM** | 0.239 | 0.239 | 0.236 | 0.185 | 0.100 | 6 | ±12 pp | 15 / 589 |
| **JEDM** | 0.312 | 0.235 | 0.190 | 0.148 | 0.116 | 5 | ±10 pp | 10 / 664 |
| **JLA** | 0.169 | 0.238 | 0.314 | 0.180 | 0.099 | 7 | ±12 pp | 9 / 516 |
| **default** (pooled, unknown venue) | 0.246 | 0.237 | 0.241 | 0.170 | 0.106 | 6 | ±12 pp | 34 / 1,769 |

Venue-specific targets are justified: a single global target mis-calibrates JLA's newest bin by 14 pp (0.169 vs 0.246 pooled) and JEDM's by 7 pp in the other direction. That spread is larger than the noise in either venue's estimate.

### 1.2 Where the anchor data is too thin — state it, don't hide it

- **JEDM (n=10, 664 refs)** — soundest per-bin estimates in the corpus. Tolerance ±10 pp.
- **EDM (n=15, 589 refs)** — bins carry extra noise: four anchors have lossy PDF→Markdown conversions with non-contiguous `[n]` numbering (`chatgpt_cs1_codegen` 15/28, `llm_judgments_content` 36/51, `phone_use_programming` 32/47, `theory_building_dbr` 44/76), so their age distribution is computed from a subset of entries. Alphabetically-ordered lists losing contiguous slices should be roughly year-neutral, but widen tolerance to ±12 pp.
- **JLA (n=9, 516 refs)** — **thinnest venue.** Its distinctive signature (the 6–10 y mode at 31.4%, the lowest `le2` at 16.9%) rests on nine papers. Use it, at ±12 pp, and re-mine when the JLA corpus grows past ~15 anchors. Do not treat the JLA `le2`=0.169 figure as precise to better than ~5 pp.
- The JEDM/JLA numbers depend on the 2026 publication-year correction. Under the alternative 2025 assumption, JEDM `le2` moves 0.312 → 0.431 and JLA 0.169 → 0.244. The miner must **record the assumed publication year per venue in the output file** so this is auditable and re-derivable (§5.4).

### 1.3 Two hard floors (near-universal in the corpus; what a reviewer notices)

1. **≥ 1 reference older than 15 years.** True in 94% of anchors (32/34), 100% of both journals. Median anchor carries 7–8 such references (EDM 6, JEDM 10, JLA 9).
2. **≥ 25% of references older than 10 years** (`11_20` + `gt20`). Anchor values: EDM 28.5%, JEDM 26.4%, JLA 27.9%, pooled 27.6%. The linter warns below 15% (forgiving floor); the composition algorithm aims at the venue target.

A pre-2000 citation is present in 30/34 anchors but is **not** made a floor: for some topics (LLM-in-education, for instance) no pre-2000 paper is genuinely topical, and manufacturing one is exactly the padding failure we are trying to avoid. It is reported as a metric only.

### 1.4 Reference *count* stays a separate lever — do not conflate

The anchors are long **and** old-skewed; these are independent. Reference-count targets continue to come from `data_registry/venue_norms.yaml` `refs.median` via `citations.venue_citation_target()` (EDM 34, JEDM 62, JLA 65) — unchanged by this fix.

> **Open item to raise separately (do not bundle):** the miner's `count_refs()` heuristic gives EDM `refs.median` = 34 over 12 anchors, while the year-parse method used for the recency study gives 47 over 15 anchors. Raising EDM's count target moves the linter's P25 floor and is a deliberate, separately-reviewed change. File as **F-P5-EDM-REFCOUNT-METHOD-DRIFT**.

### 1.5 Concrete slot allocations (largest-remainder, deterministic; pinned by tests)

| venue | N (`refs.median`) | `le2` | `3_5` | `6_10` | `11_20` | `gt20` |
|---|---|---|---|---|---|---|
| EDM | 34 | 8 | 8 | 8 | 6 | 4 |
| JEDM | 62 | 19 | 15 | 12 | 9 | 7 |
| JLA | 65 | 11 | 16 | 20 | 12 | 6 |
| default | 55 | 14 | 13 | 13 | 9 | 6 |

Remainder tie-break: **prefer the older bin** (higher bin index).

---

## 2. IS RANKING ENOUGH?

**No. Decisively no. Retrieval must change, and the year window must be removed at the HTTP layer.**

Three quantitative statements, in increasing order of how badly they close the question:

1. **The persisted pool has no old work at all.** `runs/arc_p_validation_20260711/output/retrieved_literature.json` holds 100 records: 2024×4, 2025×79, 2026×17. `pool_min_year = 2024`. A *perfect* rerank of that pool yields **0** references older than 5 years and at most **4** older than 2 years, of which 3 were already selected by the LLM and already cited. The EDM target at N=34 needs 6 in `11_20` + 4 in `gt20` = **10 references ≥ 11 years old**, plus 8 in `6_10` → **18 of 34 must be older than 5 years**. The reranking ceiling is 0 of 18. Ranking closes 0% of the gap.

2. **The untrimmed candidate set cannot supply them either.** The year-desc trim discarded 198 of 298 S2 candidates, but every one of those is bounded below by the API filter `year=2016-2026` (`problem_formulator.py:460`, `year_filter: 10`). At manuscript year 2026 the maximum achievable reference age is **10 years**. The `gt20` bin (10–12% of every anchor venue) and most of `11_20` are **structurally unreachable at the HTTP layer**. No amount of client-side reranking, blending, or quota reservation can recover a record the request excluded. Tatsuoka 1983, Junker & Sijtsma 2001, de la Torre 2011 are not retrievable today.

3. **Ranking is nonetheless necessary, and there is direct evidence it works.** Within this same run, the 12 LLM-selected papers reached pool positions 96/97/98 (the three oldest records) and were cited at **83%** (10/12); the 50 year-desc mechanical top-ups were cited at **8%** (4/50), and 6 of them were clinical-oncology/cardiology papers that the LLM selected none of. 48 of the 62 bib entries were uncited. So relevance-aware ranking is worth roughly a 10× cite-rate difference — it is what makes the retrieved old work actually get used rather than merely sit in the `.bib`.

**Conclusion:** retrieval change (§3) is the *necessary* condition; ranking change (§4) is the *sufficient* condition. Shipping either alone fails — §3 without §4 lets the year-desc sorts re-skew the newly available old records back out of the pool; §4 without §3 is a no-op against an all-2025 pool.

---

## 3. RETRIEVAL CHANGES

**File:** `src/agents/problem_formulator.py`. **Functions:** `_run_single_s2_query`, `_search_semantic_scholar`, `_search_literature`.
**Config:** `config.yaml` → `semantic_scholar:` block.

### 3.1 `_run_single_s2_query` (currently lines ~410–509)

**(a) Fields.** Line 458, replace:

```
"fields": "paperId,title,authors,year,abstract,venue,externalIds"
```
with
```
"fields": ("paperId,title,authors,year,abstract,venue,externalIds,"
           "citationCount,influentialCitationCount,referenceCount,"
           "publicationDate,fieldsOfStudy,publicationTypes,isOpenAccess")
```

All of these are free on `/paper/search` — no extra quota, no special access, no extra request; only the response payload grows (~2–3× on a 100-row response). Do **not** add `tldr`: model-generated, sparse, duplicates `abstract`, inflates payload.

**(b) LOAD-BEARING — extend the record mapping.** The dict comprehension at lines 482–494 hand-maps exactly 7 keys. **Adding fields to the request without editing this comprehension is a complete no-op.** New mapping:

```python
return [
    {
        "paperId": item.get("paperId", ""),
        "title": item.get("title", ""),
        "authors": [a.get("name", "") for a in item.get("authors", [])],
        "year": item.get("year"),
        "abstract": item.get("abstract") or "",
        "venue": item.get("venue") or "",
        "doi": (item.get("externalIds") or {}).get("DOI") or "",
        # Arc P5 — ranking signals. None (not 0) when S2 omits them, so
        # the ranker can tell "no data" from "genuinely uncited".
        "citationCount": item.get("citationCount"),
        "influentialCitationCount": item.get("influentialCitationCount"),
        "referenceCount": item.get("referenceCount"),
        "publicationDate": item.get("publicationDate") or "",
        "fieldsOfStudy": item.get("fieldsOfStudy") or [],
        "publicationTypes": item.get("publicationTypes") or [],
        # Provenance: S2 returns RELEVANCE order and we must not lose it
        # again (F-P5-DEPTH-RECENCY-SKEW).
        "matched_query": query,
        "retrieval_rank": rank,
        "source": "s2",
    }
    for rank, item in enumerate(data.get("data", []))
    if item.get("paperId")
]
```

**(c) Optional year window.** Change the `year_start`/`current_year` parameters to a single `year_range: str | None`, and build params conditionally:

```python
params = {"query": query, "fields": FIELDS, "limit": limit}
if year_range:
    params["year"] = year_range
if min_citation_count:
    params["minCitationCount"] = min_citation_count
```

`year` semantics confirmed against the S2 OpenAPI spec: `"2019"`, `"2016-2020"` (inclusive), `"2010-"` (open forward), `"-2015"` (open backward).

### 3.2 `_search_semantic_scholar` (currently lines ~511–595)

**(a) Drop the recency window on the three primary queries.** Relevance — not a date cutoff — is what keeps the pool on-topic; S2's default order is already a trained relevance re-ranker. New config semantics:

```yaml
semantic_scholar:
  max_results: 100
  # Arc P5: null = no year window. A rolling "last N years" filter can
  # NEVER return seminal work (F-P5-DEPTH-RECENCY-SKEW); the 2016-2026
  # window made >20y references structurally unreachable.
  year_filter: null
  seminal_query:
    enabled: true
    min_citations: 50      # judgment call, NOT calibrated — see §7 note
    limit: 20
```

`year_filter: null` (or `0`) ⇒ pass `year_range=None`. Keep the key for backwards compatibility with older run snapshots.

**(b) DELETE the year-desc sort at 577–579.** Replace the merge/trim with a rank-aware interleave:

```python
# Round-robin across queries by S2 relevance rank, so each query
# contributes ~max_results/len(queries) instead of query recency deciding.
per_query: dict[str, list[dict]] = {}          # insertion-ordered
for paper in merged_papers:                     # merged_papers keeps arrival order
    per_query.setdefault(paper["matched_query"], []).append(paper)
for lst in per_query.values():
    lst.sort(key=lambda p: p.get("retrieval_rank", 10**6))

final_papers, i = [], 0
while len(final_papers) < max_results and any(i < len(v) for v in per_query.values()):
    for lst in per_query.values():              # deterministic: dict order
        if i < len(lst) and len(final_papers) < max_results:
            final_papers.append(lst[i])
    i += 1
```

Records from the seminal query (§3.3) participate in the interleave as their own "query", so they get a guaranteed ~1/4 share of the pool rather than competing on relevance rank against topical queries.

**(c) Add the seminal-work query (Option B — recommended).** One extra request, executed after the three primary queries:

```python
sem = cfg.get("seminal_query", {})
if sem.get("enabled", True) and queries:
    time.sleep(1.0)
    seminal = self._run_single_s2_query(
        query=queries[0],                    # highest-signal keyword query
        year_range=None,                     # explicitly unwindowed
        min_citation_count=int(sem.get("min_citations", 50)),
        limit=int(sem.get("limit", 20)),
        ...
    )
    for p in seminal:
        p["matched_query"] = "__seminal__"
        if p["paperId"] and p["paperId"] not in seen_ids:
            seen_ids.add(p["paperId"]); merged_papers.append(p)
```

`minCitationCount` is a server-side filter — it forces a high-influence slice into the pool independent of where relevance ranking happened to place it. Failure of this single query must be **non-fatal**: log and continue with the three primary queries (the composition algorithm's spill rule in §4 absorbs the deficit).

**Rejected — Option C:** mirroring all three queries unfiltered = 6 S2 requests, ~+4.5 s, roughly doubles 429 exposure on an endpoint that returned 429 on both unauthenticated probes today. Gate it behind `os.environ.get("SEMANTIC_SCHOLAR_API_KEY")` if ever wanted; do not enable by default.

### 3.3 `_search_literature` (currently lines ~690–746)

- **DELETE the second year-desc sort at 728–732.** Two sort sites must change, not one.
- Replace the trim with a reserved-slot merge: keep the top `max_results − arxiv_reserve` S2 records in relevance order and append up to `arxiv_reserve` deduped arXiv records. New config `arxiv.reserved_pool_slots: 10`. Persisted pool size stays 100. (Today 27 of 30 arXiv hits are discarded by the year sort.)
- **Fix the duplicated LLM call:** `_generate_search_queries` is invoked at line 539 *and* line 704 with the same input. Cache on `self._search_queries` and reuse. Two benefits: one fewer LLM call per run, and the arXiv queries can no longer diverge from the S2 queries (which would corrupt `matched_query` provenance).

### 3.4 Cost and rate-limit budget

| | today | after |
|---|---|---|
| S2 requests / run | 3 | **4** |
| arXiv requests / run | 3 | 3 |
| `_generate_search_queries` LLM calls | 2 | **1** |
| added latency | — | **~+1.5 s** (0.5 s `request_delay_s` + 1.0 s inter-query) |
| added quota | — | **0** (fields are free; `minCitationCount` is a filter, not a cost) |

Pacing is unchanged: `request_delay_s=0.5` before the first attempt of every query, `time.sleep(1.0)` between queries, exponential backoff `1.0 × 2^(n−1)` with ±25% jitter, `max_retries=3`. 4 requests × up to 4 attempts is comfortably inside a 1 RPS keyed budget.

**Prerequisite, user action, not code:** set `SEMANTIC_SCHOLAR_API_KEY`. Two of two unauthenticated probes to `/paper/search` returned 429 twelve seconds apart. Unauthenticated is a 1,000 req/s pool *shared across all users*, further throttled under load; a key gives a dedicated 1 RPS. The code already sends `X-API-KEY` when the env var exists (`problem_formulator.py:533-536`) — this is config, not a code change. **Do not schedule a live validation run of §3 until the key is set.**

---

## 4. RANKING FUNCTION

**File:** `src/citations.py`. New public functions `allocate_age_slots`, `rank_pool`, and a new `profile`/`now_year` parameter on `expand_literature_pool`.

Bucket helpers (`bucket_of_age`, `AGE_BUCKETS`, `bib_entry_years`) live in `src/manuscript_linter.py` and are **imported** by `citations.py` — the same producer/checker-cannot-disagree rule already used for `_BIB_ENTRY`, `_CITE_CMD`, `cited_keys` (`citations.py:30`).

### 4.1 Signals, and how each degrades

| signal | source | when missing |
|---|---|---|
| `relevance` | reciprocal-rank fusion over `retrieval_rank` per `matched_query` | **legacy pools have no `retrieval_rank`** → every record gets the same constant; the term drops out of the ordering entirely |
| `influence` | `log1p(influentialCitationCount)`, else `log1p(citationCount/10)` | record-level missing (arXiv) → **pool median of the non-missing values** (neutral prior, never 0 — a 0 systematically ranks every preprint last and re-introduces a different bias). Pool-wide missing (legacy) → all 0.5, term drops out |
| `topicality` | `fieldsOfStudy` first, `venue` keywords as fallback | neither present → 0.0 (neutral, not rejected) |
| `age` | `year` | no `year` ⇒ **ineligible** (existing contract, `citations.py:179`) |

Graceful degradation is therefore total: on the existing `runs/arc_p_validation_20260711` pool shape (7 keys, no counts, no ranks) the ranker reduces to *topicality + age-bucket quota + deterministic tie-breaks* and still runs, still returns `target` records, and still refuses the six clinical-oncology entries.

### 4.2 Pseudocode

```python
AGE_BUCKETS = ("le2", "3_5", "6_10", "11_20", "gt20")
FILL_ORDER  = ("gt20", "11_20", "6_10", "3_5", "le2")   # scarcest first

ONTOPIC = {"educational data mining", "learning analytics", "artificial intelligence
           in education", "computers & education", "computers and education",
           "educational technology", "learning at scale", "user modeling",
           "intelligent tutoring", "educational measurement", "psychometrika",
           "applied psychological measurement", "educational psychology",
           "educational researcher", "review of educational research",
           "learning technologies", "aied", "lak", "edm", "jedm", "jla"}
OFFTOPIC = {"oncology", "cancer", "cardiovascular", "cardiology", "clinical",
            "diabetes", "transplant", "surgery", "radiology", "nursing",
            "psychiatry", "npj", "lancet", "bmj"}
ONTOPIC_FIELDS = {"Education", "Computer Science", "Psychology", "Mathematics",
                  "Sociology", "Linguistics", "Economics"}

def bucket_of_age(age: int) -> str:
    if age <= 2:  return "le2"
    if age <= 5:  return "3_5"
    if age <= 10: return "6_10"
    if age <= 20: return "11_20"
    return "gt20"

def allocate_age_slots(target: int, profile: dict[str, float]) -> dict[str, int]:
    """Largest remainder; ties go to the OLDER bin. sum(result) == target."""
    if target <= 0: return {b: 0 for b in AGE_BUCKETS}
    exact  = {b: target * profile[b] for b in AGE_BUCKETS}
    slots  = {b: int(exact[b]) for b in AGE_BUCKETS}
    short  = target - sum(slots.values())
    order  = sorted(AGE_BUCKETS,
                    key=lambda b: (-(exact[b] - int(exact[b])),
                                   -AGE_BUCKETS.index(b)))     # older wins ties
    for b in order[:short]:
        slots[b] += 1
    return slots

def _topicality(p) -> float:
    fos = [f.lower() for f in (p.get("fieldsOfStudy") or [])]
    if fos and not any(f in {x.lower() for x in ONTOPIC_FIELDS} for f in fos):
        return -1.0                                   # HARD REJECT
    v = (p.get("venue") or "").lower()
    on  = any(t in v for t in ONTOPIC)
    off = any(t in v for t in OFFTOPIC)
    if off and not on:  return -1.0                   # HARD REJECT
    return 0.25 if on else 0.0

def rank_pool(pool, now_year) -> tuple[list[dict], list[str]]:
    """Returns (eligible records with _score attached, degraded_signals)."""
    degraded = []
    # --- relevance: RRF over per-query retrieval ranks --------------------
    have_rank = any(p.get("retrieval_rank") is not None for p in pool)
    if not have_rank: degraded.append("retrieval_rank")
    # --- influence: pool-median imputation, max-normalised ---------------
    raw = {}
    for p in pool:
        inf = p.get("influentialCitationCount")
        if inf is None and p.get("citationCount") is not None:
            inf = p["citationCount"] / 10.0
        raw[id(p)] = math.log1p(inf) if inf is not None else None
    known = sorted(v for v in raw.values() if v is not None)
    if not known:
        degraded.append("influentialCitationCount")
        med, hi = 0.5, 1.0
    else:
        med, hi = statistics.median(known), max(known) or 1.0

    out = []
    for p in pool:
        if not p.get("title") or p.get("year") is None:  continue
        topic = _topicality(p)
        if topic < 0:                                    continue   # off-topic
        rel = (1.0 / (60 + p["retrieval_rank"])) * 60 if have_rank and \
              p.get("retrieval_rank") is not None else 0.5
        infl = ((raw[id(p)] if raw[id(p)] is not None else med) / hi) \
               if known else 0.5
        p = {**p,
             "_age": max(0, now_year - int(p["year"])),
             "_rel": rel, "_infl": infl, "_topic": topic}
        p["_bucket"] = bucket_of_age(p["_age"])
        p["_score"]  = 0.55 * rel + 0.30 * infl + 0.15 + p["_topic"]
        out.append(p)
    return out, degraded

def _sort_key(p):
    # Total order -> no ties -> deterministic under input permutation.
    return (-round(p["_score"], 9),
            -round(p["_infl"],  9),
            p.get("retrieval_rank", 10**6),
            -p["_age"],                       # prefer the OLDER paper
            p.get("paperId", ""))

def expand_literature_pool(selected, pool, target, dedup_threshold=0.80,
                           profile=None, now_year=None):
    out, seen_keys, seen_titles = _place_selected(selected, pool)   # UNCHANGED
    if target <= len(out) or profile is None:
        return out                       # profile=None -> exact legacy behaviour

    now_year  = now_year or datetime.utcnow().year
    slots     = allocate_age_slots(target, profile)
    for p in out:                        # selected consume their own bin's slots
        if p.get("year") is not None:
            b = bucket_of_age(max(0, now_year - int(p["year"])))
            slots[b] = max(0, slots[b] - 1)

    ranked, degraded = rank_pool(pool, now_year)
    by_bucket = {b: [] for b in AGE_BUCKETS}
    for p in sorted(ranked, key=_sort_key):
        k = sanitize_key(p.get("paperId", ""))
        if k in seen_keys:                                    continue
        toks = _title_tokens(p.get("title", ""))
        if any(_jaccard(toks, t) >= dedup_threshold for t in seen_titles): continue
        by_bucket[p["_bucket"]].append(p)

    appended, budget = [], target - len(out)
    # PASS 1 — honour quotas, OLDEST BIN FIRST so a tight budget protects
    # the scarce historical tail rather than the abundant new work.
    for b in FILL_ORDER:
        take = min(slots[b], budget - len(appended), len(by_bucket[b]))
        appended += by_bucket[b][:take]; by_bucket[b] = by_bucket[b][take:]
    # PASS 2 — spill unfilled quota, again OLDEST FIRST. This is the rule
    # that stops a pool with no old work from silently re-skewing to le2.
    for b in FILL_ORDER:
        take = min(budget - len(appended), len(by_bucket[b]))
        appended += by_bucket[b][:take]; by_bucket[b] = by_bucket[b][take:]

    return out + appended        # selected block first, order untouched
```

### 4.3 Contracts this must preserve

- `selected` are never dropped and never reordered (`citations.py:154-164` unchanged), authoritative pool metadata still substituted by `paperId`.
- A record with no `title` or no `year` is never appended.
- Dedup by sanitized `paperId` then title-Jaccard ≥ 0.80, against both the echo title and the pool title.
- `profile=None` ⇒ byte-identical legacy behaviour (needed so old tests keep passing and so a caller can opt out).

### 4.4 Orchestrator wiring (`src/orchestrator.py::_expand_literature_for_depth`, lines 702–749)

- Pass `profile=venue_age_profile(venue)` and `now_year`.
- Write `citation_depth_report.json` beside `literature_context_expanded.json`: `{venue, target, n_selected, n_appended, achieved_buckets, achieved_fractions, target_fractions, deficits_by_bucket, degraded_signals, pool_year_min, pool_year_max, pool_n_older_than_10, n_offtopic_rejected}`. This is what the Arc I harness reads without re-deriving anything.
- **Log loudly when `selected == []`.** Today that silently degrades to `pool[:62]` — pure recency, zero relevance input, and it is the exact state the live run was in at REVISING cycle 1. Emit a WARNING-level pipeline.log line naming the degradation, and let the age quotas still apply (they do, since the quota path does not depend on `selected`).

---

## 5. LINTER / METRIC CHANGE

**File:** `src/manuscript_linter.py`.

### 5.1 New helpers — place immediately after `_BIB_ENTRY` (~line 74)

```python
_BIB_ENTRY_YEAR = re.compile(
    r"@\w+\s*\{\s*([^,\s]+)\s*,(.*?)(?=\n@|\Z)", re.DOTALL)
_YEAR_FIELD = re.compile(r"\byear\s*=\s*[{\"']?\s*((?:19|20)\d{2})")

def bib_entry_years(bib: str) -> dict[str, Optional[int]]:
    """{citation key -> publication year or None} from references.bib."""
```

### 5.2 New section "Reference recency" — between `_check_tex` and the compile-log section (~line 211)

Constants: `AGE_BUCKETS`, `FILL_ORDER`, `bucket_of_age()`, and

```python
DEFAULT_REF_AGE_PROFILE = {"le2": 0.246, "3_5": 0.237, "6_10": 0.241,
                           "11_20": 0.170, "gt20": 0.106}
DEFAULT_REF_AGE_TOLERANCE_PP = 12.0
MIN_REFS_FOR_DISTRIBUTION_CHECKS = 10
MIN_REFS_FOR_FOUNDATIONAL_FLOOR  = 15
```

New function `_check_reference_recency(report, cited, bib, venue, norms, now_year)`. The venue profile comes from `norms[venue]["ref_age"]` (§5.4) and falls back to `DEFAULT_REF_AGE_PROFILE` with `ref_age_profile_source = "default"`.

### 5.3 Metrics and checks

**Metrics (always emitted — this is the Arc I payload):**

`ref_year_parsed`, `ref_year_missing`, `ref_age_median`, `ref_age_mean`, `ref_age_max`, `ref_age_buckets` (counts, cited refs), `ref_age_fractions`, `ref_age_target_fractions`, `ref_age_profile_source`, `n_refs_older_than_10`, `n_refs_older_than_15`, `frac_refs_older_than_10`, `n_refs_pre_2000`, `bib_age_buckets` and `bib_age_fractions` (whole `.bib`, so "what was available" is separable from "what was cited").

**Checks:**

| code | severity | condition | rationale |
|---|---|---|---|
| `reference-recency-collapse` | **error** | `frac(le2) ≥ 0.90` and `ref_year_parsed ≥ 10` | the exact F-P5 signature (shipped run: 1.00). A paper with no prior-decade citations reads as unscholarly; the reviewer marks down Related Work and the review is burned. |
| `no-foundational-references` | **error** | `n_refs_older_than_15 == 0` and `ref_year_parsed ≥ 15` | 94% of anchors and 100% of both journals cite something older than 15 y; median anchor carries 7–8. |
| `reference-recency-skew` | warn | any bin \|achieved − target\| > venue tolerance (pp), with `ref_year_parsed ≥ 10` | steering signal; message names each offending bin and its signed delta so the Arc P4 revision prompt can act on it. |
| `thin-historical-tail` | warn | `frac_refs_older_than_10 < 0.15` (aspiration 0.25) | anchors: 26.4–28.5% per venue. |
| `bib-recency-collapse` | warn | same as row 1 but over the whole `.bib`, when `n_bib_entries > n_citations_distinct` | catches a skewed *pool* even when the model happened to cite the two old ones — i.e. it fires on the upstream cause, not just the symptom. |

All checks are hard-gated on the minimum-reference counts so an honestly short bibliography is never punished (EDM has genuine 13-reference anchors).

`error` is the right severity for the first two despite lint being advisory to the gate: `format_clean=False` does not fail the run (`review_gate.py:261-305`), but errors are surfaced first in the revision prompt and in `manuscript_lint.json`. That is precisely the loudness that was missing — the count check was green while this happened.

### 5.4 `lint_manuscript` wiring (lines 343–411)

Hoist norms loading above the tex branch so the recency check can run venue-agnostically:

```python
norms = load_venue_norms(norms_path)
...
    _check_front_matter(report, tex)
    _check_tex(report, tex, bib)
    _check_reference_recency(report, cited_keys(tex), bib, venue, norms, now_year)
...
if venue:
    _check_venue_norms(report, venue, norms)
```

Add `now_year: Optional[int] = None` to the `lint_manuscript` signature (defaults to `datetime.utcnow().year`) purely for test determinism.

### 5.5 `data_registry/venue_norms.yaml` + `scripts/mine_venue_norms.py`

Extend the miner to parse **years** out of the references block (not just count entries) and emit, per venue:

```yaml
venues:
  EDM:
    pub_year: 2024              # evidence: max cited year 2024, "EDM 2024" venue strings
    ref_age:
      buckets: {le2: 0.239, "3_5": 0.239, "6_10": 0.236, "11_20": 0.185, gt20: 0.100}
      median_age: 6
      mean_age: 9.04
      p90_age: 20
      tolerance_pp: 12
      frac_papers_with_ref_older_than_15: 0.867
      median_n_refs_older_than_15: 6
      n_refs_dated: 589
      n_anchors: 15
      notes: "4 anchors have lossy PDF conversions; bins carry extra noise"
defaults:
  ref_age:
    buckets: {le2: 0.246, "3_5": 0.237, "6_10": 0.241, "11_20": 0.170, gt20: 0.106}
    ...
```

Pin `ANCHOR_PUB_YEAR = {"EDM": 2024, "JEDM": 2026, "JLA": 2026}` in the miner **with the evidence string in a comment**, and emit `pub_year` into the file so the 2025-vs-2026 sensitivity stays auditable. Fix the two parser bugs already found: `^\s*\d+\.\s*REFERENCES` heading form, and reject a lowercase mid-body `reference` line as a bibliography start (require the heading form).

### 5.6 Arc I lesson — the benchmark battery must track distribution, not count

This defect is the canonical instance of a green count metric hiding a quality regression: "62 of 62 available references" was true and the scholarship got worse. Encode the lesson structurally:

1. **Rule for `docs/v5_roadmap.md` §Arc I / I1:** *every count-shaped target in the battery must ship with a paired distribution metric over the same population.* Reference count → age histogram. Word count → per-section word distribution. Model count → per-family performance spread.
2. **`scripts/evaluation_harness.py` ledger columns to add:** `ref_age_median`, `ref_age_frac_le2`, `ref_age_frac_gt10`, `n_refs_older_than_15`, `n_bib_uncited`, `pool_year_min`, `pool_year_span`, `pool_n_older_than_10`, `n_offtopic_rejected`, and the count of new linter defect codes fired. Source them from `manuscript_lint.json` + `citation_depth_report.json` — no re-derivation.
3. **Pool-level assertion in the battery** (fails the battery row, not the run): `pool_year_span ≥ 15` and `pool_n_older_than_10 ≥ 15`. This catches a silent retrieval regression *before* a paper is written, which is where this defect actually lived.

---

## 6. TEST PLAN

New file **`tests/test_arc_p5_recency.py`** (follows the existing `test_arc_p3_p4.py` arc-named convention); linter tests may also extend `tests/test_manuscript_linter.py`. All offline — `requests.get` monkeypatched, no network, no LLM.

**A. Slot allocation — `citations.allocate_age_slots`**
1. `test_slots_sum_to_target` — every venue × N ∈ {0, 1, 13, 34, 55, 62, 65, 100}: `sum(slots) == N`.
2. `test_slots_match_published_profile` — pins §1.5 exactly: EDM/34 → `[8,8,8,6,4]`, JEDM/62 → `[19,15,12,9,7]`, JLA/65 → `[11,16,20,12,6]`, default/55 → `[14,13,13,9,6]`.
3. `test_remainder_ties_favour_the_older_bucket`.
4. `test_unknown_venue_uses_pooled_default_profile`.

**B. Composition — `citations.expand_literature_pool`**
5. `test_selected_papers_are_never_dropped_or_reordered` (existing contract, re-pinned under the new code path).
6. `test_old_papers_are_preferred_when_buckets_demand_them` — pool = 50 records at `now_year` (top relevance ranks) + 10 at `now_year-18` (worst ranks); target 20, EDM profile ⇒ ≥ 3 of the 18-year-old records appear.
7. **`test_not_all_references_come_from_the_last_two_years`** — *the pinned regression for F-P5-DEPTH-RECENCY-SKEW; put the F-item id in the docstring.* Pool spans 2001–2026; assert on the expanded list: `frac(age ≤ 2) ≤ 0.55` **and** `n(age > 10) ≥ 2` **and** `n(age > 15) ≥ 1`.
8. `test_legacy_behaviour_is_preserved_when_profile_is_none` — `profile=None` reproduces the 100%-`le2` output byte-for-byte, documenting the defect and guarding against a future refactor silently changing the opt-out default.
9. `test_bucket_deficit_spills_to_next_oldest_not_newest` — pool with zero `gt20` records ⇒ the 4 `gt20` slots go to `11_20` (when available) before any reach `le2`.
10. `test_offtopic_venue_records_are_not_promoted` — pool containing `venue="Journal of Clinical Oncology"` and `fieldsOfStudy=["Medicine"]` records ⇒ none appear in the output. Pins the 6 clinical entries that shipped.
11. `test_arxiv_records_get_neutral_influence_not_zero` — arXiv records (no counts) mixed with high-count S2 records; a top-relevance arXiv record must still be appended.
12. **Degradation — `test_ranks_legacy_pool_without_citation_counts`** — feed the exact 7-key shape of `runs/arc_p_validation_20260711/output/retrieved_literature.json` (no `citationCount`, no `retrieval_rank`): assert (a) no exception, (b) `len(out) == target` when the pool is large enough, (c) two identical calls give identical id order, (d) the reported `degraded_signals` names both `retrieval_rank` and `influentialCitationCount`.
13. `test_records_without_year_are_never_appended`.
14. `test_deterministic_under_input_permutation` — `random.Random(42).shuffle(pool)`, assert identical output ids and order. Proves `_sort_key` is a total order.

**C. Retrieval — `problem_formulator`**
15. `test_s2_request_asks_for_influence_fields` — `fields` contains `citationCount`, `influentialCitationCount`, `publicationDate`, `fieldsOfStudy`.
16. `test_s2_request_omits_year_when_window_disabled` — `year_filter: null` ⇒ no `year` key in `params`.
17. **`test_new_fields_survive_the_record_mapping`** — fake response carries `citationCount=123`; the persisted record must carry it. This is the load-bearing detail: adding request fields without extending the dict comprehension at 482–494 is a silent no-op.
18. `test_retrieval_rank_and_matched_query_are_stamped`.
19. `test_seminal_query_is_one_extra_request` — exactly 4 GETs; the 4th carries `minCitationCount` and no `year`.
20. `test_seminal_query_failure_is_non_fatal` — 4th call 429s through all retries ⇒ pool still returned from the first three.
21. **`test_pool_is_not_sorted_by_year`** — fake responses with mixed years where the highest-relevance record is the oldest; assert the persisted pool is **not** monotone non-increasing in year and that the old high-rank record survives the trim. Directly pins the two deleted sorts.
22. `test_search_queries_generated_once_per_run` — pins the 539/704 duplicate-call fix.

**D. Linter — `manuscript_linter`**
23. `test_all_recent_bibliography_is_an_error` — 30 entries all at `now_year` ⇒ `reference-recency-collapse`, severity `error`.
24. `test_bibliography_matching_anchor_profile_is_clean` — synthetic bib built from EDM/34 slots ⇒ none of the five new codes fire.
25. `test_no_reference_older_than_15_years_is_an_error` — 20 refs, oldest 8 y ⇒ `no-foundational-references`.
26. `test_short_bibliography_is_exempt` — 8 recent refs ⇒ no error (guards the ≥10 / ≥15 gates).
27. `test_bucket_deviation_over_tolerance_warns` — 40% in `le2` vs EDM target 23.9% (Δ 16 pp > 12 pp) ⇒ `reference-recency-skew`, message names the bin and the signed delta.
28. `test_metrics_expose_the_distribution_not_just_the_count` — `ref_age_buckets`, `ref_age_fractions`, `ref_age_median`, `n_refs_older_than_15` all present in `report.metrics`.
29. `test_unparseable_years_do_not_raise` — entries with `year = {n.d.}` / no year field ⇒ `ref_year_missing` counted, no exception, linter never raises (existing invariant).
30. `test_venue_absent_falls_back_to_pooled_profile` — `venue=None` ⇒ `ref_age_profile_source == "default"`.
31. **`test_shipped_arc_p_validation_bib_would_have_been_caught`** — copy the shipped 62-entry `references.bib` to `tests/fixtures/arc_p_validation_references.bib`; assert `reference-recency-collapse` fires with `now_year=2026`. This is the "would have caught it" proof and it must be in the suite.

**E. Norms schema**
32. `test_venue_norms_yaml_has_ref_age_for_every_venue` — each venue block has `ref_age.buckets` with all five keys summing to 1.0 ± 0.01, plus `median_age`, `tolerance_pp`, `pub_year`, `n_refs_dated`. Schema pin so a regeneration that drops the block fails the suite.

---

## 7. SEQUENCING

### 7.1 Hard blocker

A live pipeline is executing from this worktree. **Nothing under `src/` may be touched until it finishes** — `citations.py` and `manuscript_linter.py` are imported at WRITING/REVIEWING, `problem_formulator.py` at FORMULATING (and re-imported on every revision cycle), `orchestrator.py` throughout. `config.yaml` is snapshotted per run. `data_registry/venue_norms.yaml` is **read live** by `load_venue_norms()` at REVIEWING, so it is blocked too even though it is not code.

### 7.2 Can be prepared now (not in the live import path)

- `scripts/mine_venue_norms.py` extension (§5.5) — `scripts/` is not imported by the pipeline.
- `tests/fixtures/arc_p_validation_references.bib` (copy of the shipped bib).
- `docs/v5_roadmap.md` Arc I rule (§5.6.1) and a backlog entry for **F-P5-EDM-REFCOUNT-METHOD-DRIFT** (§1.4).
- Setting `SEMANTIC_SCHOLAR_API_KEY` in the environment (user action; no repo write).

### 7.3 Landing order after the run completes

| step | change | why this order |
|---|---|---|
| **P5.1** | Regenerate `data_registry/venue_norms.yaml` with `ref_age` + `pub_year` blocks; add test 32 | pure data; no behaviour change until a consumer reads it |
| **P5.2** | `manuscript_linter.py` recency checks + tests 23–31 | **measurement first.** Advisory-only, cannot break a run, and it is the instrument that proves the rest works. Land and run it against the shipped artifacts before touching any producer. |
| **P5.3** | `citations.py` `allocate_age_slots` + `rank_pool` + `expand_literature_pool(profile=…)`; orchestrator `citation_depth_report.json` + loud `selected == []` logging; tests 1–14 | behaviour change confined to the append path; `profile=None` keeps the legacy path intact |
| **P5.4** | `problem_formulator.py` fields + record mapping + delete both year sorts + interleave + seminal query; `config.yaml` `year_filter: null`, `seminal_query`, `arxiv.reserved_pool_slots`; tests 15–22 | riskiest (live API, 429 exposure). Requires the API key from §7.2. Lands last. |
| **P5.5** | `scripts/evaluation_harness.py` ledger columns + pool assertion (§5.6.2–3) | consumes artifacts the earlier steps emit |

### 7.4 What must land before the next live gated run

**All of P5.1 → P5.4.** They are not independently shippable to a live run:

- P5.3 without P5.4 cannot reach the target — the pool contains zero papers older than 2024, so the quota loop spills everything into `le2` and the linter fires the same warnings. Harmless but useless.
- P5.4 without P5.3 makes old work reachable and then lets the append path take the head of the list — different pool, same skew.
- P5.2 alone is safe and worth landing immediately after the run regardless: it converts an invisible regression into a logged defect.

**If schedule forces a subset**, the minimum viable set is **P5.2 + P5.4 + the cheap first cut of P5.3**: require ≥ 25% of appended references older than 10 years and ≥ 2 older than 15 years, filling those two constraints greedily by score before taking the rest in relevance order. That captures most of the gain without the full bucket allocator, and the full allocator can follow.

### 7.5 Calibration debt to close, not to skip

`seminal_query.min_citations = 50` is a judgment call, not a measured threshold. Calibrate it against the anchor corpus: for the references in the anchors that fall in `11_20` and `gt20`, measure the citationCount distribution and set the floor at roughly its P25. Do this before the value is treated as pinned; until then, keep it in `config.yaml` with an explicit `# NOT CALIBRATED` comment.