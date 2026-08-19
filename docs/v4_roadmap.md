# EDM-ARS V4 Roadmap — From Single-Dataset causal_soo to a General Educational Research System

> **Status**: Planning document, agreed 2026-07-03. Each arc below is written to be
> launched with a one-line hand-off ("go with R1") — phases carry their own scope,
> acceptance criteria, and decision gates, so execution can be hands-off except where a
> gate explicitly needs the user.
>
> **Execution status (2026-07-03, autonomous run-all-arcs directive):**
> Arc H DONE (V2.1 5/5 slim; DE hardening; gated runs 6.4 + 7.0). Arc R DONE
> (causal_itr live: LSAR 6.7; synthetic gate certified; R3-followup shipped
> deterministic ITR helpers). Arc D DONE (selector + gap miner live; design
> memo in the R3 paper). Arc G: G0 report + G1 kit DONE; acquired ASSISTments
> (83MB) + ECLS-K:2011 (460MB, layout parsed 26,060 cols); PISA staged; ELS
> gated on the EDAT wizard (user step). Arc Q DONE at machinery level
> (replicated synthetic gates all pass; live registration deferred until a
> dataset's design_feasibility supports each design). Arc S DONE (archetypes +
> ban skills, mandatory, all task types). Arc L pilot IN FLIGHT (16-anchor
> EDM-2024 batch reviewing; calibration analyzer ready). Arc E harness DONE +
> pilot IN FLIGHT (ledger 3 papers; prediction pilot running; gates G-E1/G-E2
> documented).
>
> **User decisions 2026-07-03 (post-program):** (a) LSAR switched to
> deepseek-v4-pro — MiniMax retired everywhere; anchor corpus re-reviewing under
> deepseek before the calibrated numbers ship. (b) Calibrated P25 gate ADOPTED —
> wired via review_gate.calibration_path (overall P25 overrides; per-dimension
> advisory). (c) ELS:2002 EDAT export = user manual step (instructions
> delivered). (d) PISA 2022 download approved and running. (e) **IV instrument
> curation → FUTURE WORK** (too human-involved for now; Q2 stays synthetic-
> certified). (f) **Arc E human-review arm (G-E2) → FUTURE WORK** (no reviewer
> capacity); the LSAR-vs-human agreement analysis waits with it.
>
> **Decisions locked at authoring time** (user-confirmed): hardening before any new
> capability; optimal treatment regimes first among causal expansions (then RD → IV →
> DiD); dataset targets = ELS:2002, PISA 2022, ECLS-K:2011, ASSISTments **contingent on
> public-use availability — free-use data only**; LSAR calibration **deferred** until
> after the causal expansion (design preserved in Arc L).

---

## §0 Vision and baseline

**Vision** (user statement, 2026-07-03): an agentic automated research pipeline covering
prediction *and* causal inference in quantitative educational research — quasi-experimental
designs (IV, DiD, RD, PSM), treatment-effect estimation, and optimal treatment regimes for
personalized decision-making. It should generate research questions grounded in real
research gaps, choose its own identification strategy, generalize across common public-use
educational datasets, avoid formulaic paper styles, and be reviewed by a calibrated
auto-reviewer.

**Baseline (what V2.x/V3.0 already proved)**:

- Skill-based architecture works: 53 skills, runtime matching/composition, per-tier caps,
  mandatory bypass. 3 of 5 agents slim (OutlineAgent, ProblemFormulator, Critic).
- One causal task type (`causal_soo`, selection-on-observables) runs end-to-end on
  HSLS:09: G1–G5 identification skills, D1 encoding contract, M1–M5 estimator battery,
  deterministic Critic verdicts, LSAR gate. Validated across 7 gated live runs.
- The migration/hardening playbook is reusable: single-variable Δ, Pattern A (offline
  rendered-prompt verification) / Pattern B (live re-runs), adversarial
  content-preservation verification, harvest-then-retag, two-commit sub-waves.
- Known instabilities (all with open F-items): DE codegen flakiness (F-3b17, promotion
  trigger met), positivity nondeterminism under DE retry churn (F-3b23.5), M1 dtype
  errors (2-run pattern), refuters non-execution (reopened-adjacent), prediction-shaped
  pre-critic checks on causal runs (F-3b11 carryover).

**Arc order** (dependencies in §9): H → R → D → G → Q(RD→IV→DiD) → S → L → E.

---

## §1 Arc H — Hardening (IMMEDIATE; phase 3b.23.7, then V2.1 completion 3b.24–3b.25.5)

Everything downstream multiplies DE/Analyst codegen paths; stabilize first.

### H1 (3b.23.7): DE hardening + M1 fix + refuters stabilization

Single-issue sub-waves, one commit each:

1. **Orchestrator post-DE pre-flight** (F-3b17 promotion, trigger met in 3b.23.5): after
   ENGINEERING, deterministically assert the D1 causal-data contract on the *produced
   matrices* (treatment column present + binary; no `registry type 'unknown'`
   passthroughs for spec-listed covariates; propensity-relevant covariates encoded per
   registry type; extreme-tail-fraction sanity bound). Fail → targeted DE retry with the
   violation text injected, not a blind re-prompt. This also addresses
   F-3b23.5-POSITIVITY-NONDETERMINISM (same fix surface).
2. **M1 dtype discipline** (F-3b21.5-M1-DTYPE-ERROR, 2-run pattern): amend the M1
   regression-adjustment skill with explicit Series-vs-DataFrame / `.values` casting
   rules (same family as the historical SHAP+NumPy fix).
3. **Refuters stabilization** (F-3b13 family reopened-adjacent in 3b.23.5): G5 skill
   amendment making refuter execution unconditional-with-fallback and its status a
   pre-critic assertion.
4. **Pattern-B validation run** — doubles as the 3b.23.5b disambiguation: a post-hardening
   re-run on the same smoketest. Acceptance: COMPLETED; no DE 4-attempt failures;
   extreme-tail fraction back under 0.10; refuters `status=ran`; LSAR gate ≥ 5.5
   restores the arc's passing streak.

### H2 (3b.24): Writer slim — per migration spec §3.4, using the §3.3.8 precedent list
(formatter-cap survival is load-bearing at Writer: `latex-figure-discipline` and
`bibtex-from-literature-context` demonstrably DROP in live runs — audit severity before
applying; output contracts stay in the prompt body).

### H3 (3b.25): Analyst slim + **3b.25.5 re-run** — per migration spec §3.5. Closes V2.1
(5 of 5 agents slim). The re-run also validates H1 durability.

**Arc exit criteria**: V2.1 complete; two consecutive gated runs with no DE/Analyst
codegen F-items and LSAR ≥ 5.5.

---

## §2 Arc R — Optimal treatment regimes (V3.1, new task type `causal_itr`)

First causal expansion (user priority; also the most data-feasible: builds on the existing
M5 causal forest, and HSLS supports it today).

- **R0 — scope doc** (3b.20-style): estimands (policy value V(π), regret vs oracle;
  treat-all / treat-none / threshold-rule baselines); method battery: doubly-robust
  policy learning (econml policy trees / DR-learner + rule extraction), cross-fitted
  policy-value estimation with CIs; **fairness constraints on rules** (subgroup value
  parity — an education-specific contribution surface); optional stretch: 2-stage dynamic
  regimes over HSLS wave-sequenced treatments (e.g., 9th→11th course intensity).
- **R1 — skills + template**: task template `causal_itr`; new skills M6
  (policy-learning), M7 (policy-value evaluation + honesty rules); G-family and D1 reuse
  with an ITR extension (rule covariates must be *actionable/observable at decision
  time* — a new temporal-style constraint); `critic-checklist-causal-itr`
  (mandatory); Writer conventions for ITR papers (decision-relevant framing, value
  tables, rule cards). Offline Pattern-A verification throughout.
- **R2 — synthetic-DGP validation gate** (NEW discipline element, mandatory for every
  new estimator battery from here on): before any live run, the battery must recover
  known-truth on simulated DGPs grounded in real HSLS covariates (known optimal rule →
  measured regret; null-effect DGP → no false rule). ITR errors are silent without
  ground truth; this gate is cheap and reusable.
- **R3 — live arc**: locked-spec smoketest (e.g., "for whom does raising math
  self-efficacy change college attendance — learn and evaluate a targeting rule over
  X1SES × X1SEX × X1TXMTSCOR"), then the single-issue amendment loop (the 3b.11→3b.19
  pattern) until two clean gated runs.

**Arc exit criteria**: synthetic gate green; two clean gated live runs; paper passes LSAR
with the ITR-specific sections intact.

---

## §3 Arc D — Design-selection intelligence (V3.2)

The "methodologist brain": the system chooses its identification strategy, and question
generation becomes genuinely gap-driven.

- **D1 — registry schema extension**: per-dataset `design_feasibility` block — candidate
  running variables + cutoffs (RD), candidate instruments with written
  exclusion-restriction justifications (IV), policy/timing/cohort variation (DiD), panel
  linkage keys, treatment-sequence variables (DTR). Curated fields; Tier-2 auto-drafts
  flagged for human review.
- **D2 — deterministic feasibility predicates** (`src/design_selector.py`): data-structure
  checks per design (cutoff density near threshold; first-stage strength precheck;
  pre-period availability). No LLM in the predicate layer.
- **D3 — design-selection skill family + PF integration**: PF emits a **design memo** in
  `research_spec` (chosen design, feasibility evidence, rejected alternatives with
  reasons; falls back to SOO with an explicit statement of the stronger designs that were
  infeasible and why). Critic gets a design-appropriateness checklist (mandatory).
- **D4 — gap-mining upgrade** (serves "questions based on real research gaps"): extend the
  literature step from retrieve-and-cite to structured extraction — pull
  limitations/future-work claims from retrieved abstracts, cluster into an
  unstudied-cell matrix (population × treatment × outcome × design), and require the
  research question to name the cell it fills, with the supporting excerpts attached as
  `novelty_evidence`. Findings-memory becomes a cross-run *research program* memory
  (asked-and-answered cells excluded).

**Arc exit criteria**: on HSLS, the selector correctly routes ≥ 3 hand-authored probe
questions to {SOO, ITR, infeasible-explain}; gap memos cite real retrieved excerpts
(spot-checked); Critic design gate exercised in one live run.

---

## §4 Arc G — Dataset generalization (V3.3)

User constraint: **public-use / free data only.** G0 verifies licensing before anything
is built.

- **G0 — public-use feasibility investigation** (one report, no code): confirm free
  public-use availability and download paths for ELS:2002 (NCES EDAT public file),
  ECLS-K:2011 (NCES public-use), PISA 2022 (OECD public files), ASSISTments (public
  skill-builder releases). Record variable-suppression caveats per file (the HSLS lesson:
  suppression shapes what designs are feasible). Output: go/no-go per dataset + a
  `design_feasibility` sketch each.
- **G1 — onboarding kit**: codebook/CSV → auto-profile → Tier-2 registry draft →
  curation checklist → dataset skill layer (missing-code conventions, wave/prefix
  mapping, CSV quirks). Goal: onboarding a new NCES-family dataset ≤ 1 phase.
- **G2 — ELS:2002 first** (closest structural cousin to HSLS — cheapest true test that
  the dataset-skill layer generalizes; also the DiD partner cohort for Arc Q3).
  Acceptance: one prediction run + one causal_soo run complete on ELS with no
  HSLS-specific skill leakage (rendered-prompt checks).
- **G3 — ECLS-K:2011** (different age band/structure — a stronger generalization
  stressor).
- **G4 — PISA 2022** (plausible-values methodology sub-arc: PV-aware estimation +
  variance rules as skills; international framing in Writer conventions).
- **G5 — ASSISTments** (prediction-side: clickstream feature engineering skills;
  knowledge-tracing baselines).
- **G6 — cross-dataset regression suite**: the smoketest matrix (dataset × task type)
  runs offline-verifiable checks per cell; live runs sampled, not exhaustive.

**Arc exit criteria**: ≥ 2 new datasets fully onboarded with clean gated runs; the
onboarding kit documented well enough that a future dataset is one phase of work.

---

## §5 Arc Q — Quasi-experimental designs (V3.4–V3.6)

Sequenced by feasibility; **every battery passes the R2-style synthetic-DGP gate before
touching real data** (public-use suppression makes real-data validation weak for these
designs — synthetic-first protects correctness).

- **Q1 — Regression discontinuity (V3.4)**: local-polynomial estimation with robust
  bias-corrected CIs (rdrobust-family Python port), McCrary/density manipulation test,
  bandwidth sensitivity curves, placebo cutoffs, covariate smoothness checks. Runs
  against whichever onboarded dataset G0 found a defensible cutoff in; if none, the
  machinery still ships validated-on-synthetic and the design selector marks RD
  infeasible-with-reasons per dataset (that honesty is itself a feature).
- **Q2 — Instrumental variables (V3.5)**: 2SLS/LIML, first-stage F + weak-IV-robust
  inference (Anderson–Rubin CIs), overidentification tests where applicable. The crux is
  the **instrument registry**: per-dataset curated candidate instruments with written
  exclusion-restriction justifications (D1's schema), and a Critic gate that scores the
  *argument*, not just the statistics. No instrument in the registry → IV infeasible,
  say so.
- **Q3 — Difference-in-differences (V3.6)**: modern estimators (Callaway–Sant'Anna
  style), event-study plots, parallel-trends pretests, honest-DiD-style sensitivity.
  Data path: cross-cohort designs (HSLS × ELS:2002) or policy variation within a
  dataset; depends on G2. Library maturity in Python is the main risk — evaluate
  `pyfixest`/ports in the scope doc before committing.

**Arc exit criteria per design**: synthetic gate green (bias/coverage within tolerance);
one live gated run on a real dataset where feasible; design-selector integration
(routes + explains).

---

## §6 Arc S — Anti-formulaic paper style (V3.7)

Depends on H2 (Writer slim) so style knowledge lives in skills, not the prompt.

- Narrative **archetype skills** selected by results shape: null-result paper (we already
  produce good ones), heterogeneity/targeting paper (pairs with Arc R), methods-comparison
  paper, policy-implication paper. The OutlineAgent's emphasis-trigger mechanism is the
  existing hook — archetypes extend it.
- **Formulaic-construction ban list** as a mandatory writing skill (the "signs of
  AI writing" genre: inflated significance claims, rule-of-three padding, boilerplate
  limitation paragraphs) + venue style profiles.
- A cheap style-critic pass (LLM checklist over the draft) before LSAR; LSAR's Clarity
  dimension is the outcome measure.
- **Known tension** (recorded now for Arc L): style diversity fights anchored review
  calibrated on typical accepted papers. Resolution: anchors calibrate rigor/empirical
  dimensions only; novelty/clarity stay absolute.

---

## §7 Arc L — LSAR calibration (DEFERRED by decision; design preserved)

When activated (after Arc Q or alongside Arc E), work happens in the LSAR repo:

1. **Anchor corpus**: 30–60 open-access papers (EDM proceedings, JEDM; LAK where
   accessible), stratified full/short paper as quality strata.
2. **Reliability first**: test–retest LSAR on a fixed subset → per-dimension ICC; if
   noise dominates (plausible given the observed 5.0–7.0 swings), add multi-sample
   median scoring before any calibration.
3. **Calibration**: per-dimension accepted-paper score distributions → percentile gates
   (e.g., pass ≥ 25th percentile of accepted) — or, stronger, **pairwise placement**:
   compare the candidate against 2–3 anchors per dimension (LLM judges are more reliable
   pairwise than absolute).
4. **Validation**: calibrated LSAR must separate the known strata; then re-gate the
   pipeline (`review_gate` config swaps absolute 5.5 for calibrated thresholds).
5. Novelty-tension mitigation per §6.

Until then, the absolute 5.5 gate stays, with run-over-run comparison (as in the 3b arc)
as the practical guard against gate noise.

---

## §8 Arc E — Controlled evaluation (SPEC §10 Phase 5, extended)

The finale, after ≥ 2 task types are stable on ≥ 2 datasets: N ≈ 10 pipeline papers
across task types + matched human-authored baselines, blind review by EDM/LAK-familiar
researchers, LSAR-vs-human agreement analysis (which retro-validates Arc L's
calibration). This is the publishable evidence for the system itself.

---

## §9 Sequencing, dependencies, decision gates

```
H (now) ──► R (ITR) ──► D (selector) ──► Q1 (RD) ──► Q2 (IV) ──► Q3 (DiD)
                │                            ▲                       ▲
                └──► G0..G2 (ELS) ───────────┘        G2 is Q3's data prerequisite
                         └──► G3/G4/G5 (parallel with Q, as capacity allows)
S (style) after H2; L (LSAR) deferred → before/with E; E last.
```

**Decision gates needing the user** (everything else is hands-off):
1. Arc G0 results: which datasets to actually download/onboard given the public-use
   findings (licensing/registration steps may need the user's NCES/OECD account).
2. Arc Q2: approval of the curated instrument list per dataset (validity argumentation
   is a scientific judgment call, not just engineering).
3. Arc L activation timing; Arc E recruiting human reviewers.
4. Budget checkpoints: live gated runs cost ~$6 and ~1h each; a new task type has
   historically taken ~8–10 gated runs to stabilize (the causal_soo arc took 3b.5→3b.19).

**Standing disciplines carried forward**: single-variable Δ per amendment; Pattern A
before Pattern B; adversarial content-preservation verification for every prompt/skill
migration; synthetic-DGP gate for every new estimator battery (new, from Arc R on);
orchestrator-path rendering in offline tests; output contracts never in cap-droppable
skills; non-ASCII markers in rendered-prompt tests.

## §10 Risks

| Risk | Mitigation |
|---|---|
| DE/Analyst codegen nondeterminism (proximate cause of the 3b.23.5 gate fail) | Arc H's post-DE pre-flight; deterministic contract assertions grow with each new task type |
| Python library maturity for RD/DiD (rdrobust ports, CS-DiD) | Scope docs evaluate libraries before committing; sandbox image pinning; fallback implementations |
| Public-use suppression limits real-data IV/RD/DiD | Synthetic-first validation; design selector's infeasible-with-reasons honesty; G0 catalogs limits per dataset |
| LSAR noise gates good work / passes bad work until Arc L | Run-over-run comparison discipline; multi-sample median as a cheap interim if swings recur |
| Anchored review vs anti-formulaic style | Calibrate rigor dimensions only; novelty/clarity absolute (recorded in §6/§7) |
| Scope creep across arcs | One-line-launchable phases with exit criteria; arcs end, they don't taper |
