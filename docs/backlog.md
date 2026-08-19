# EDM-ARS Backlog (logged 2026-07-08 at V3 staging)

Durable record of every deferred/open job so nothing is lost between
sessions. Mirror of the session task list; update BOTH when closing items.

## A. Dataset onboarding (user: "hold for now")

- [ ] **A1. ASSISTments 2009-10** — data on disk (83MB); tier-2 draft registry
  exists (`assistments_0910_tier2_draft.yaml`). Needs Tier-1 curation, adapter,
  clickstream conventions skill. Also the natural CDM dataset (skill tags →
  Q-matrix derivable) and first IRT item-response matrix.
- [ ] **A2. ECLS-K:2011** — data on disk (482MB); SPS fixed-width parser proven
  on its 26,060 columns (G1 kit). Needs curation + adapter + conventions skill.
- [ ] **A3. PISA 2022** — STU_QQQ SAV on disk (2.1GB). Blocked on a
  **plausible-values methodology skill** (all PVs + Rubin's rules) before any
  achievement outcome is defensible. Cognitive item files not yet downloaded.

## B. Causal-inference residuals (phase paused at V3, resumable)

- [ ] **B1. RD live enablement** — estimator certified (Arc Q); data-gated:
  needs a dataset with a documented cutoff running variable in
  `design_feasibility.running_variables`.
- [ ] **B2. IV live enablement** — estimator certified (Arc Q); **user-deferred**:
  requires human instrument curation with written exclusion-restriction
  justifications.
- [ ] **B3. F-S2-MCT-SKIPPED** — model_comparison_test skipped when LR wins
  (runner-up rule not followed). Fix if it recurs: compute deterministically in
  analyst.py post-processing from saved probabilities.
- [ ] **B4. F-S2-GROUPS-EMPTY-ON-RETRY** — top_feature_groups dropped by
  revision-cycle regenerated code. Same deterministic post-processing fix.
- [ ] **B4b. F-P3-LICENSE-FLAG**: psychometrics group_mean_comparison_permitted inconsistent with the invariance ladder (conservative under-claim) — add deterministic license==f(ladder) pre-critic check.
- [ ] **B5. Verify F-R3-M6/M7 closed-by-helpers** on the next causal_itr run.
- [ ] **B6. DiD extensions** (optional): more cohort pairs (ECLS-K/PISA once
  onboarded); richer panels.
- [ ] **B7. DE first-pass reliability ~50% under deepseek** — deterministic
  DE-recipe caching idea (Arc H residual).
- [ ] **B8. Executor warning-only-stderr false-failures** (Arc H residual).

## C. Evaluation & LSAR (user decision gates)

- [ ] **C1. G-E1**: ~7-paper evaluation batch (~$45) for the controlled matrix.
- [ ] **C2. G-E2**: human review arm — no reviewers currently available.
- [ ] **C3. LSAR calibration corpus expansion**: EDM conference beyond 16 anchors; JEDM+JLA CALIBRATED 2026-07-10 (P25 5.15 / 5.4); remaining journals (JEM/JEBS/MBR/Psychometrika/JREE) are paywalled - need user-provided PDFs or OA subsets to anchor.
- [ ] **C4. Journal-level expansion** (user, 2026-07-08): extend beyond
  conference venues — needs (a) a journal anchor corpus for LSAR calibration
  (JEDM, JLA, JEBS candidates), (b) journal-format writing skills (length,
  structure, review depth), (c) venue-selection logic. Treat as its own arc.

## D. Infra / cleanup

- [ ] **D1. README.md version stale** — says v1.2.0. (Fixed in the v3.0.0
  staging commit — verify.)
- [ ] **D2. MiniMax config blocks** retained in configs for rollback; remove
  when confident.
- [ ] **D3. pymupdf-layout ONNX int32/int64 upstream bug** — we fall back to
  classic mode; consider version pin/upgrade when fixed upstream.
- [ ] **D4. `tmp_orch_test/`, `tmp_orch_test2/`, stray `pipeline_stdout.txt`**
  untracked artifacts in repo root — delete or gitignore.

## E. V4 Psychometrics — WAVE 1 COMPLETE 2026-07-09 (both papers Accepted: 7.5 record + 7.3; see docs/v4_psychometrics_arc_report.md)

- [x] **E1. Scope doc** (task families, CTT/CDM decision, method IDs P1..Pn).
- [x] **E2. R-bridge executor** (Rscript subprocess + JSON contract; R 4.4.1
  with lavaan/mirt/psych/semTools/difR/CDM confirmed installed) + skill md for
  running R from Python + sandbox implications (R not in Docker image).
- [x] **E3. Certified deterministic helpers** with synthetic gates (simulate
  from known IRT/DIF/invariance structure → recover; null-honesty).
- [x] **E4. Task template(s) + skills + prompt variants** (causal_did playbook).
- [x] **E5. Item-bank curation**: HSLS 286 S1 items / ELS 447 BYS items —
  identify the scale item sets (math identity, self-efficacy, engagement...)
  in registries.
- [x] **E6. Gated live run(s)**: first paper candidate — measurement
  invariance / DIF of a math attitude scale across SES/race/sex.

## E2. Wave-2 follow-ups (2026-07-10)

- [x] **E2a. DONE 2026-07-10 (7,654 words vs 5.9k; advisory 7.2 vs 5.6) — Journal-length sectionwise writing**: manuscript came out
  ~5.9k words vs the 8k target; writer max_tokens is the binding
  constraint — generate journal papers section-by-section.
- [x] **E2b. DONE 2026-07-10 (compare mode + single-attribute degeneracy honesty, gate-certified) — G-DINA comparison arm**: JEDM reviewer dinged single-model
  DINA (Novelty 4); add gdina() comparison + model-selection reporting
  to the P7 recipe.
- [x] **E2c. DONE 2026-07-10 (JEDM anchored: 10 OA papers, P25 5.15, gate calibrated; other journals stay advisory until anchored) — Journal anchor calibration (C4, priority RAISED)**: journal
  review spread was 3.7 points across samples vs conference MAD 1.9 —
  advisory mode is honest but journal-grade gating needs anchors.

## G. V5 Arc P residuals (found by the 2026-07-11 validation run)

Full detail + numbers in `runs/arc_p_validation_20260711/FINDINGS.md`.
Every one of these was found by inspecting artifacts, not by a failing
test — the suite was green throughout.

- [x] **G1. F-P5-DEPTH-RECENCY-SKEW** — SHIPPED 0cf0f0c: retrieval (seminal query + citationCount) + bin-quota composition + linter recency check. Measured 0% -> 32% refs older than 10y. (highest value). Citation depth
  reached its target (62 refs vs JEDM median 62) while the reference
  list got *worse*: every reference was 2024-2026 because the pool is
  sorted by year DESC before trimming and expansion preserves that
  order. A cognitive-diagnosis paper citing no foundational work is
  padded, not well-referenced. Fix spec being derived from the anchor
  corpus (target recency distribution per venue) + S2 `citationCount`
  ranking + possibly a second un-year-filtered "seminal work" query.
  **Also add a distribution check to the linter** — the count metric
  cannot see this failure.
- [x] **G2. F-P5-CTT-SPARSE-NAN** — SHIPPED 0cf0f0c: psy_ctt does pairwise-present and returns an explicit not-estimable instead of NaN.. `psy_ctt()` computes Cronbach's alpha
  by complete-case deletion; on ASSISTments skill-builder logs (27.6%
  fill, no student answers all 47 items) `n_complete` is 0 and alpha is
  NaN. CTT assumes a near-complete persons x items matrix; tutor logs
  violate that structurally, while IRT/CDM handle it natively. Fix:
  pairwise deletion with per-covariance n, or an honest "not estimable
  on a structurally sparse matrix" return. The psychometrics protocol
  skill should state the constraint so P1 is never requested on log data.
- [x] **G3. F-P5-BATTERY-SCOPE-CREEP** — SHIPPED 0cf0f0c: post-Analyst assertion that measurement_results matches the locked method_battery. (intermittent — fired in 1 of 2
  runs, cost 24 min + a CFA timeout). The Analyst ran the full P1-P7
  battery though the locked spec said `["P1","P7"]`. Fix: deterministic
  post-Analyst assertion that `measurement_results` blocks match the
  locked `method_battery`, mirroring the post-DE pre-flight.
- [x] **G4. F-P5-PSY-SCHEMA-KEYS** — SHIPPED 0cf0f0c: required keys now sourced from the TaskTemplate.. `_REQUIRED_KEYS` in
  `src/agents/analyst.py` is a hardcoded *prediction* schema applied to
  every task type, so psychometrics/causal runs get a phantom
  "results.json is missing required keys" error injected into
  `results.json.errors` and handed to the Critic as a real analysis
  failure. Fix: source required keys from the TaskTemplate.
- [x] **G5. Uncited-bib-entry warning is noise by design.** — SHIPPED 0cf0f0c: uncited-bib warning scoped to PDF-visible entries. With the
  bibliography intentionally a superset the reviser can draw from,
  "many-uncited-bib-entries" fires on every run. BibTeX only prints
  cited entries, so it is not reader-visible. Either scope the check to
  entries that reach the PDF, or drop it.
- [x] **G6. Writer reconciliation logs to `ctx.log`, not `pipeline.log`** — SHIPPED 0cf0f0c.
  — the "Bib reconciliation:" line is invisible to log monitoring and
  only recoverable from checkpoint.json. Route agent-level Arc P
  telemetry to the pipeline log.


## H. Arc T residuals (2026-07-11)

Full evidence: `docs/v5_arc_t_v2_backtest_verdict.md` (v1 null) and
`docs/v5_arc_t_t1b_verdict.md` (v2 blind table also null).

- [x] **H1. The venue-fit ranker measures nothing, twice over.** v1 headline
  rho +0.79 was entirely circular (external-only rho **+0.002** on n=24). A
  blind re-derivation (v2) scored **-0.63 on the calibrated population**,
  **+0.147 (p=0.245)** on n=24, and **ties the pre-registered pair at -2.50** --
  it cannot tell a 3.7 Reject from a 7.0 Accept on two runs whose only
  difference is the idea. A pure run-recency baseline beats both. **Do not
  adopt either table as a ranker.** Both remain advisory; `tournament.py`
  refuses live selection in code.
- [x] **H2. THE REAL FINDING -- a capability gap, not a ranking bug.** 5 of — DONE 2026-07-11, docs/v5_arc_t_h2_capability_roadmap.md. Headline: the 5 shapes are really ONE capability (their union is 18/34 anchors) + two FREE framing fixes + two replications already finished on disk and never claimed.
  v2's 8 blind rules fire on **0 of 34** of our own specs while firing on
  21-44% of published anchors. They describe study shapes this pipeline
  **structurally cannot produce**. On our candidate space v2 collapses to a
  3-rule negative-only table, two of which fire on 68% and 82% of specs --
  which is *why* it is null. This is the most actionable output of Arc T so
  far: it says what to BUILD, not how to rank. Extract the 5 unproducible
  shapes into concrete capability items.
- [ ] **H3. The archive cannot validate a venue-fit ranker at all.** Our 34
  specs do not span the feature space the anchors do, so several rules are
  *untestable* here rather than refuted. Any future validation needs either a
  wider candidate space (H2) or an out-of-archive criterion.
- [ ] **H4. No live LLM call has been made anywhere in Arc T.** Position-bias
  rate, tie rate, false-veto rate, and real dedupe/diversity are all
  unmeasured. Stage 4 (absolute AE screen on the top 2) is unimplemented.
- [ ] **H5. `config.yaml` has no `ideation:` block.** Every tuned value across
  T0/T1a/T1b is a code default (judge_samples 3, temp 0.2,
  purpose_coverage_min 0.60, BT weights 0.30/0.20, prior_sd 1.0), and
  `priorart.DEFAULT_ANCHOR_CORPUS` is a hardcoded machine path. Violates the
  project rule that config values come from config.yaml.


- [x] **H6. VENUE DECISION (owner gate, blocks the expensive path).** Measured — **DECIDED 2026-07-11: Option C, dual-target with venue routing** (owner decision). Causal + survey work routes to the policy family (AERA Open first — OA, calibratable autonomously; JREE/EEPA later, paywalled); measurement, psychometrics and prediction-METHOD work stays EDM/JEDM/JLA. VF2-01/02 become routing signals, not idea penalties.
  on a 1,101-abstract counter-corpus (ERIC, 2022-2025) with the SHIPPED
  detectors: observational causal work runs 19.7% of EEPA abstracts, 13.9%
  JREE, and **30.0% of a random n=30 AERA Open FULL TEXTS** — against
  **0 of 34** in our anchor corpus. **3 of those 30 AERA Open papers use
  HSLS:09, our exact dataset.** So the pipeline is not doing work nobody
  publishes; it is doing work published constantly at venues absent from the
  anchor corpus. Options: (A) stay EDM/JEDM/JLA and acquire learner text;
  (B) retarget causal+survey work to the policy family; (C) dual-target with
  venue routing (recommended). `LSAR/venue_criteria/jree_2026.yaml` already
  declares family `causal-applied` with `calibrated: false` — **the gap is
  anchors, not criteria.** NOT settled by the data: nothing here measures
  acceptance, and EDM 2026's own CFP lists causal inference as in scope.
- [ ] **H7. VF2-04/05/08 are ONE capability, not three.** Their anchor union
  is 18 of 34 (53%); pairwise overlaps 7/9/5. "An LLM measures a construct
  from learner text, validated against human codes, inside a workflow with a
  real human step" fires all three (up to +3.0 at JEDM). Costing them
  separately triple-counts the work. Blocked on DATA: we hold **zero**
  learner text (HSLS max cell = 142 chars over 2,000 rows x 9,614 cols;
  ELS numeric codes; ASSISTments `answer_text` is tier-3 excluded).
  Candidate public corpora that SHIP human labels, removing the human-hours
  cost: ASAP-AES (2 raters/essay), PERSUADE (discourse annotations).
- [ ] **H8. Wire the v2 rule table (FREE, blocks measurement).** `venue_fit.py`
  loads v1, so **nothing is currently scored against VF2-06 or VF2-07 at
  all**. Needs a ~40-line clause evaluator (reference impl:
  `scripts/derive_venue_rules.py::evaluate_predicate`) + a de-hyphenation
  pass before any future derivation.

## I. Routed AERA_OPEN run audit F-items (2026-08-06)

Full evidence: `runs/aera_open_routed_did_20260711/REPORT.md`. The run
scored 7.5 vs the calibrated 7.3 gate (median of 3, all Accept) but the
pass is INVALIDATED: fabricated numbers + missing UNVERIFIED flag. Found
by the 3-auditor adversarial verification workflow, not by any pipeline
check — every fabricated value sailed through Critic, lint, and LSAR.

- [x] **I1. F-WRITER-NULL-FABRICATION (critical).** — SHIPPED 2026-08-06:
  (a) `_mark_null_values` renders nulls as NOT-AVAILABLE markers in both
  Writer prompt builders; (b) numeric-reconciliation lint
  (`unreconciled-table-numerals` / `unreconciled-ci-interval`) checks
  table numerals + prose CI intervals against results/checkpoint/
  data_report/research_spec + small summary CSVs, with sibling
  sums/diffs as legitimate derivations; error-severity findings BLOCK
  the review gate (HONESTY_BLOCKING_CODES). Backtested: flags the
  fabricated tab:m8_2x2 (6/10), tab:m10_contrasts (9/21), the invented
  [-14.19,+7.36] CI, PLUS caught a 4th invented CI the human audit
  missed; 0 false positives on 3 historical papers after adversarial-
  review hardening (digit-grouping {,}, column-spec dimensions,
  sci-notation, row-level CSV exclusion). Writer prompt carried
  null cell_means / null follow_wave / no pairwise CIs for race5+pared3;
  the Writer invented plausible-looking values for all of them (Table
  tab:m8_2x2 means, probe -0.84 [-2.61, +0.93], tab:m10_contrasts CIs,
  "PS clipping at 1st/99th percentiles" vs actual fixed 0.02/0.98).
  "Never fabricate" is prompt rule 5 — LLM-obedience only. Fix is
  deterministic: (a) render null artifact fields as explicit
  "NOT AVAILABLE — do not report a value" markers in the Writer prompt;
  (b) numeric-reconciliation lint: extract numerals from generated
  tables/abstract and cross-check against results/checkpoint JSON;
  unmatched numbers = ERROR that blocks the gate.
- [x] **I2. F-UNVERIFIED-FLAG-NOT-ENFORCED (critical).** — SHIPPED 2026-08-06: deterministic `_inject_unverified_flag` (block + Critic appendix) on every Writer path; linter ERROR `unverified-block-missing`; gate-blocking; `run_is_unverified` shared contract with explicit-flag-wins semantics (orchestrator now writes unverified=False on effective PASS so the evaluator-override path cannot stamp a passing paper). Orchestrator set
  review_report.unverified=True (REVISING crashed → WRITING (UNVERIFIED))
  but the SPEC §4.5 warning block is absent from the paper. Enforcement
  exists only as Writer prompt rule 6. Fix: deterministically inject the
  block during template reassembly when ctx.review_report.unverified is
  truthy + linter ERROR when the flag is set and the block is missing.
- [x] **I3. F-REVISING-NAN-CRASH.** — SHIPPED 2026-08-06: `_sanitize_nonfinite` (numbers.Real incl. numpy scalars + non-finite dict keys) with path-listing warning; string-first serialization so a raise can never truncate results.json; warnings-as-non-list coercion. Revised follow-wave probe = NaN →
  json dump crashed ("Out of range float values...") → results.json on
  disk is TRUNCATED/invalid JSON; checkpoint kept stale nulls even though
  the revision had correctly recomputed cell_means (arithmetic verifies).
  Fix: sanitize NaN/Inf → None (+ warning) before serialization in the
  Analyst results path so a partial revision lands instead of vanishing.
- [ ] **I4. F-LSAR-STAGE4-VENUE-TEMPLATE.** No AERA_OPEN entry in
  prompt_builder._get_template_name → reviews framed against the EDM
  conference CFP; system prompt also said "no anchored calibration exists
  for AERA_OPEN" (stale). MITIGATION: the 11 anchors were reviewed under
  the same fallback (23 EDM vs 3 AERA mentions in anchor reviews), so the
  calibration is internally consistent. Fix MUST bundle: aera_open
  template + fresh anchor batch re-review + new P25 (user-gated: ~12
  reviews of cost). Do not fix the template alone.
- [x] **I5. F-GATE-SUMMARY-PROVENANCE.** — SHIPPED 2026-08-06: gate_summary now carries threshold_used/threshold_source/advisory_mode/venue/dimension_floor + per-cycle median_sampling with gated_sample_dir; final_review_path points at the gated sample; stale-lint reset in prepare_pdf; futile-revision skip when only honesty lint blocks a reviewer pass. gate_summary.json: final_review
  _path points at the 7.2 sample; per_cycle_scores attributes the median
  7.5 to cycle 1; median_sampling annotation set after lsar_report.json
  is written so the [7.2, 7.5, 7.5] set survives only in pipeline.log.
  Fix: persist sample_scores + threshold_source + advisory flag into
  gate_summary.json.
- [x] **J1. LSAR scored EMPTY and TRUNCATED reviews as Accept** —
  SHIPPED 2026-08-07 (LSAR 6dd1958; full write-up
  `LSAR/docs/j1_truncated_review_defect.md`). Found by adversarially
  reviewing the I4 bundle, not by any test. LSAR's scorer builds its
  prompt from `review.strengths[:5]` + `review.weaknesses[:5]` and
  NOTHING else; every LLM client returned `content or ""` without
  checking finish_reason; `max_tokens` was 4096; and the output format
  puts Strengths (3) before Weaknesses (4) — so truncation deleted the
  CRITICAL half and the paper was graded on its praise. **21% of all 57
  historical runs are degenerate, scoring +0.83 higher.** Reached 5 of 9
  JLA anchors, 1 of 13 JEDM, 1 of 12 AERA_OPEN, and **cycle_102 of the
  audited routed run — the exact sample whose 7.5 the gate used as its
  median**. Published gates barely move when cleaned (JEDM -0.08, JLA
  -0.12, AERA_OPEN 0.00) since inflated scores sit above a lower-tail
  P25; the damage is per-decision. Fixed in five layers + an anchor-set
  health gate. **Implication for EDM-ARS: any gate verdict before this
  date could rest on a degenerate review — check
  `lsar_review/cycle_*/review.json` for non-empty strengths AND
  weaknesses before trusting a historical score.**
- [ ] **I4b. JEDM and JLA have the SAME template defect (found while
  fixing I4, 2026-08-07).** LSAR ships review templates for only AIED,
  EDM, L@S, LAK and (now) AERA_OPEN. **JEDM and JLA reviews are framed
  against the EDM conference CFP and scored against "SCORING ANCHORS FOR
  EDM"** — their calibrations (P25 5.15 / 5.4) are internally consistent
  because their anchors were reviewed the same way, exactly as AERA_OPEN
  was. Fixing either template REQUIRES re-reviewing that venue's 10
  anchors and recomputing its P25 (`scripts/calibration_reanchor_venue.py`).
  Cost: ~10 reviews per venue. USER-GATED — do not fix a template alone.
  `prompt_builder._get_template_name` now logs a WARNING on every
  fallback so this cannot hide again.
- [ ] **I6. F-LSAR-INGESTION-MOJIBAKE-TRUNCATION.** Ingested title is
  mojibake ("The** **SES** **..."), one 7.5 sample scored around a
  "truncated abstract". Also stale ADVISORY note under calibrated: true
  in venue_criteria/aera_open_2026.yaml; apa7 option typo floatsintex.
- [ ] **I7. Seminal-query calibration.** Pool held only 3 pre-2016 papers
  of 100 vs ~20 older slots wanted at 66 refs; composer placed all 3
  (proven working). min_citations 50 / limit 20 needs tuning + Writer
  uptake (cited 19 of 66 available refs) needs a citation-usage floor.

## F. Interactive CLI (user, 2026-07-09 — future phase)

- [ ] **F1. openclaw-style terminal interface**: turn EDM-ARS into an
  interactive CLI so users drive runs through a terminal conversation
  rather than `python -m src.main ...`. Scope sketch: a REPL/agentic
  shell wrapping the orchestrator (choose dataset/task/venue, launch +
  monitor runs, browse ledger/reports, answer decision gates
  interactively); config-free onboarding prompts; progress streaming
  from pipeline.log. Design questions to settle with the user: single
  long-lived process vs job-spawning; where LLM keys/config live;
  whether the CLI itself is LLM-driven (agentic) or menu-driven.
