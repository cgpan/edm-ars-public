# V5 Roadmap — Robustness, Taste, Self-Improvement

Status: **user-confirmed 2026-07-11** (arc order P → T → I; user serves as
taste anchor; Arc I diffs are PR-gated). SOTA investigation summarized in
the 2026-07-10 discussion (AI Scientist-v2, Darwin Gödel Machine,
AlphaEvolve/GEPA, Weng 2026 "Harness Engineering for Self-Improvement",
ideation studies: Si et al. ICLR-25, HindSight, Nova).

Guiding read: EDM-ARS already has the two prerequisites the field
identifies as hardest — a **calibrated anchored evaluator** (LSAR: EDM
6.3, JEDM 5.15, JLA 5.4) and a **regression harness** (full pytest suite
+ certified synthetic gates). V5 closes the remaining loops.

---

## Arc P — Polish (presentation quality) — FIRST

Root causes addressed: citation count bounded upstream (PF selects 8–12
papers; journals carry 40–80 refs); "PDF compiled" is the only
deterministic presentation check (latex/biber logs unmined); LSAR reviews
content, not typesetting.

- **P1. Manuscript linter** (deterministic verifier battery): parse
  pdflatex/biber logs + rendered PDF text for undefined citations, `??`
  cross-refs, unreferenced tables/figures, orphaned floats, per-section
  word counts, citation count + density. Runs post-compile, feeds the
  revision loop and (later) Arc I.
- **P2. Venue norms mined from the anchor corpus**: the 19 journal + 16
  conference anchor PDFs already ingested by LSAR are measured for
  reference counts, section lengths, table/figure counts → empirical
  per-venue norm profiles → linter thresholds (e.g. "JEDM P25 refs").
- **P3. Citation-depth stage**: per-section S2 retrieval during writing
  (not only at formulation); claim-level citation matching; targets the
  venue norm from P2.
- **P4. Revise–resubmit ×2**: review_gate max_cycles → 2 with
  section-targeted revision driven by LSAR's weakest dimension + P1
  linter defects. (Revision-path deepseek routing fixed 2026-07-11.)

**Exit criteria**: refs ≥ venue-norm floor; zero format-class defects on
a live run; gate-score delta reported for the revision cycle.

### Arc P progress

- [x] **P1 shipped 2026-07-11** — `src/manuscript_linter.py` (citation
  keys vs bib, placeholder cites, dangling crossrefs, unreferenced
  floats, latex/biber log mining, `??`-in-PDF, venue-norm floors);
  advisory hook in `review_gate.prepare_pdf` writes
  `manuscript_lint.json`; 15 tests.
- [x] **P2 shipped 2026-07-11** — `scripts/mine_venue_norms.py` →
  `data_registry/venue_norms.yaml` from 31 measured anchors:
  **EDM refs P25 15 (median 34); JEDM P25 54 (median 61.5); JLA P25 47
  (median 65); journal body-words P25 ≈ 8,360**. Confirms the citation
  gap quantitatively (our papers cite ~10–26).
- **Live finding (F-E2A-SECTIONWISE-BIB-DRIFT)**: linting the 7.2-scoring
  E2-validation JEDM manuscript found **22 cited keys absent from
  references.bib** (sectionwise sections cite freely; the single
  bibliography call only covers literature_context) + 26 distinct
  citations vs JEDM P25 54. The conference psychometrics paper: 4
  in-text citations, 1 missing key (samejima1969), 4,025 words vs EDM
  P25 6,746. LSAR scored past all of this — deterministic linting was
  the missing verifier. **P3 must fix bib-drift**: bibliography
  generation driven by the union of keys actually cited, with retrieval
  backfill.
- [x] **P3 shipped 2026-07-11** (commit 75cb478) — `src/citations.py`.
  Key discovery: **retrieval was never the bottleneck, the discard was.**
  The ProblemFormulator already fetches ~100 papers and persists only the
  8–12 the model echoes, so depth needs ZERO new network calls. Delivered:
  deterministic `reconcile_citations` (back-fill from the real pool /
  strip invented keys / never strip all / no-op on S2 failure);
  bibliography rebuilt deterministically from retrieved metadata instead
  of being LLM-authored; `expand_literature_pool` to the venue anchor
  median; `## Available Citation Keys` in the Writer prompts; skill v1.2.
  **Second integrity defect found and fixed**: `venue` was never requested
  from S2, so every non-arXiv entry was stamped a fabricated
  "Proceedings of the Educational Data Mining Conference" — 29 such
  entries across shipped papers. Missing venue now emits `@misc` +
  "Venue metadata unavailable".
- [x] **P4 shipped 2026-07-11** (commit 75cb478) — linter defects + venue
  citation target + section targeting from the weakest LSAR dimension
  feed the revision prompt; Analyst-owned dimensions route to Limitations
  (a prose reviser must not "fix" rigor by editing numbers). Guards:
  no-op detection, truncation rejection, float/graphics invariance
  (revision discarded, pre-revision manuscript kept), post-revision
  reconciliation. Latent gate-killing bug fixed (1-arg `_log_fn` call).
- **Harness leak closed (proven by socket probe)**: every e2e test was
  opening live HTTPS connections (OutlineAgent + LSAR metadata
  extractor), safe only because a fake key 401s — but `src/main.py` calls
  `load_dotenv()` at import, so once any test imported it the real key
  was live and those became billed calls. Now stubbed at the orchestrator
  boundary; zero outbound connections, e2e 86s → 39s.
- [ ] **P5 live validation** (running 2026-07-11): `runs/configs/arc_p_validation.yaml`
  on the ASSISTments CDM journal paper. Deliberate STRESS harness — JEDM's
  real gate is 5.15 and this manuscript last scored ~7.2, so it would pass
  on cycle 1 and never exercise P4; the config forces failure
  (`pass_threshold: 9.9`, anchor detached, `max_cycles: 3`) so both
  revisions run. Real verdict = final score vs 5.15.

## Arc T — Taste (topic quality) — SECOND

Field lessons: LLM ideas rate novel but infeasible (Si et al.);
self-assessed novelty anti-correlates with impact (HindSight); what works
is generate-many × evidence-grounded evaluation × selection (Nova).
Our edge: feasibility is deterministically checkable (registries, design
feasibility, certified estimators, data on disk).

- **T1. Idea tournament**: 15–25 candidate RQs per dataset from diverse
  personas (measurement, equity, causal, replication-gap,
  method-transfer). Scores: (a) novelty-with-evidence — S2 retrieval of
  nearest prior work + explicit delta; (b) deterministic feasibility —
  registry/data probes; (c) proposal review by an LSAR-derived AE
  screening persona; (d) venue-conversation fit mined from anchors.
  Select top 1–2 before any full run (~$0.2 idea stage vs $5–7 paper).
- **T2. Taste memory**: user ranks top-5 proposals occasionally (agreed
  2026-07-11); preferences persist and become few-shot exemplars for the
  generator (RLHF-lite, human as reward model).

**Exit criteria**: user blind-ranks tournament winners vs baseline
questions; gate scores of tournament-selected papers ≥ baseline.

## Arc I — Improvement loop (self-improvement) — THIRD

The manual F-item discipline (run → failure signature → skill/prompt
diff → regression test → re-run) IS the Darwin-Gödel loop, human-executed.
Arc I automates it **PR-gated** (user decision 2026-07-11).

- **I1. Frozen benchmark battery**: ~5 locked research specs re-run per
  pipeline version; scores tracked in the ledger. Turns "did that change
  help?" into measurement. Precondition for everything below.
- **I2. Retrospective agent**: post-run parse of pipeline.log + LSAR
  dimension scores + linter defects → proposed skill/prompt diffs, each
  with a pinned regression test → **human-approved PR**. Never
  auto-applied.
- **I3. GEPA-lite**: offline reflective evolution of ONE high-leverage
  prompt (Writer style or Analyst) against the battery.
- **I4. Experience library**: retrieve best-scoring prior sections/specs
  as few-shot exemplars (rejection-sampled experience, no training).

**Safety rails (hard)**: the retrospective agent may NEVER modify LSAR,
the anchors, or the linter (evaluator stays outside the loop); every
diff passes the full test suite + battery before merge; diversity guard
against template collapse.

**Exit criteria**: battery score trend across ≥2 pipeline versions;
≥1 retrospective-agent PR merged after human review.

---

## Cost notes

P is mostly deterministic code. T adds ~$0.5/run idea-stage tokens.
I1 battery ≈ 5 runs (~$30) per version bump — run at version boundaries
only. Model tiering (2026-07-11) offsets: outline_agent + LSAR stages
1–3 on deepseek-v4-flash; LSAR review+scoring calibration-pinned to
deepseek-v4-pro.
