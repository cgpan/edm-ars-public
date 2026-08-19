# V3.0 Phase 3a — Causal Inference Skill Specification (Audit)

**Status:** Design only. No SKILL.md files, no code, no registry edits in this phase.
**Scope:** 12 skills across method (5), methodology (6), and dataset (1) layers, supporting causal inference under selection-on-observables for HSLS:09 observational analysis. (Phase 3b.12 added the 12th: `causal-data-engineer-contract`. See the Phase 3b.12 amendment near the end of this document.)
**Proposed `task_type` slug:** `causal_soo` (selection-on-observables; rationale in §4.5).
**Pre-audit confirmation:** existing skill format conventions (frontmatter schema, body structure) internalized from `skills/methodology/{school-aware-train-test-split, cluster-id-reconstruction-from-fingerprints, subgroup-fairness-analysis}/SKILL.md`. The V2.0.1 empty-match inventory shows zero empty matches for `task_type=prediction`; the `task_type=causal_soo` slot is currently unpopulated and is the gap V3.0 fills. School-cluster reconstruction methodology skill ID confirmed: `cluster-id-reconstruction-from-fingerprints`.

---

## Section 1 — Threat Enumeration (Pass 1)

26 failures across the four threat categories, prioritized by plausibility. Plausibility ratings: H = high (mainstream EDM/AIED/LAK reviewers would catch and EDM-ARS would realistically commit), M = medium (might pass review but evident under careful read), L = low (unlikely either way).

### 1.1 Identification Failures (IDF)

| ID | Description | HSLS Example | Agent | Plausibility |
|---|---|---|---|---|
| **IDF-01** | Unmeasured confounding ignored (no available proxy for an obviously needed confounder) | Studying effect of `X1MTHID` (math identity) on `X4EVRATNDCLG` (college attendance) without a measure of *prior teacher quality* — HSLS proxies (`X1TMEFF`, `X1TMEXP`) cover only 9th-grade math teacher and have ~36% missing | PF (omits proxy from predictor_set) + Analyst (proceeds without flagging) | **H** |
| **IDF-02** | Post-treatment conditioning (covariate set includes a variable measured AFTER the treatment) | Treatment = `X1MTHEFF` (9th-grade math self-efficacy), outcome = `X4EVRATNDCLG`, covariates include `X3TGPAMAT` (12th-grade math GPA — a downstream consequence of treatment) | PF (selects covariates) + Analyst (uses them in adjustment) | **H** |
| **IDF-03** | Collider-induced selection bias (analytic sample restricted by a variable that is a common effect of treatment + outcome) | Restricting analysis to college attendees only when studying treatment → STEM major (`X4RFDGMJSTEM`); college attendance is a collider on the path treatment → ?? → STEM major | PF (target_population) + DataEngineer (analytic sample construction) | **H** |
| **IDF-04** | Positivity violation in observed strata (some treatment × covariate strata have ~0% propensity) | Treatment = "took advanced math by 11th grade", computed propensity; lowest `X1SESQ5` × `X1POVERTY185` cell has 2/300 treated → propensity score < 0.01 → IPW weights explode | Analyst (estimates propensity, computes weights) | **H** |
| **IDF-05** | Selection bias from outcome-conditional missingness (postsecondary outcomes only observed for `X4SQSTAT` respondents — MNAR) | Analyzing `X5STEMCRED` (postsecondary STEM credential) with complete-case PSM — `X5STEMCRED` is only observed for the ~50% of cohort with transcript data; missingness is correlated with treatment | DataEngineer (analytic sample) + Analyst (no IPW-for-missingness adjustment) | **H** |
| **IDF-06** | Multilevel confounding ignored (between-school confounding that within-school variation would identify) | School-level treatment policies (e.g. % of students placed in honors math) confound within-school student-level effects but `X1SCHOOLCLI`/`X1CONTROL` are aggregated, not policy-specific | Analyst (no fixed-effects or multi-level estimator) | **M** |
| **IDF-07** | Treatment-version inconsistency (operationalized treatment is a coarsening of a continuous variable with no defined contrast) | Treatment defined as "binary high-vs-low math self-efficacy" by `X1MTHEFF >= median`, masking that "+1 SD shift" and "median split" are different estimands | PF (research_spec.treatment_definition) | **M** |

**Subtotal: 7 IDF failures (5 H, 2 M).**

### 1.2 Estimand Confusion (ESC)

| ID | Description | HSLS Example | Agent | Plausibility |
|---|---|---|---|---|
| **ESC-01** | PSM gives ATT, reported as "the average causal effect" without naming as ATT | Run PSM matching treated to controls; report `mean(Y_treated_matched) - mean(Y_control_matched) = 0.X` and call it "the ATE" — ATT vs ATE divergence material when treated and control populations differ on `X1SES` | Analyst + Writer | **H** |
| **ESC-02** | "Predicts well" → "estimates causal effect well" slippage in PF specification | PF research_spec says "this study estimates the *effect* of `X1MTHEFF` on college attendance using XGBoost feature importance" — predictive task framed as causal | PF (research_question + expected_contribution wording) | **H** |
| **ESC-03** | Causal forest CATE distribution averaged to a scalar reported as if from a designed ATE estimator | Report `np.mean(causal_forest.effect(X))` as "the ATE" with no acknowledgment of (a) target population, (b) variance from honest splitting, (c) regularization bias for averaging | Analyst + Writer | **H** |
| **ESC-04** | Sample (HSLS analytic) vs population (national 9th-grade cohort 2009) estimand conflation | Paper says "the effect of X on Y in U.S. 9th-graders" but estimator targets the HSLS analytic sample (post listwise deletion, no survey weights, no school clustering); marginal effect identifiable in sample is not population-marginal effect | Writer | **H** |
| **ESC-05** | Continuous treatment "effect" with no defined contrast | Treatment = `X1MTHEFF` (continuous attitudinal scale, range -2.92 to 1.62); "causal effect" reported with no contrast (vs what?) — implicitly assumes per-unit linear effect | PF + Analyst | **M** |
| **ESC-06** | Multi-valued treatment estimand poorly defined | Treatment = `X1STUEDEXPCT` (10 ordered categories); pairwise contrasts not specified; reports a single "effect" pooled over all comparisons | PF | **M** |

**Subtotal: 6 ESC failures (4 H, 2 M).**

### 1.3 Inferential Failures (INF)

| ID | Description | HSLS Example | Agent | Plausibility |
|---|---|---|---|---|
| **INF-01** | Clustered SEs ignored when school-cluster reconstruction is available | IPW estimator computed on n=20,594 students; SE assumes IID; school-cluster pseudo-IDs available via `cluster-id-reconstruction-from-fingerprints` but not used → SEs underestimated by 2–4× depending on ICC | Analyst | **H** |
| **INF-02** | Bootstrap on matched pairs uses naive student resampling instead of matched-pair resampling | After PSM, bootstrap 1000 iterations resampling 20,594 student rows independently — destroys the matched-pair structure; should resample matched pairs as units (or use Abadie-Imbens analytic SEs) | Analyst | **H** |
| **INF-03** | TMLE / AIPW point estimate computed with naive plug-in variance instead of influence-function variance | Implement TMLE following the targeting step but report SE = bootstrap-of-mean rather than the empirical variance of the influence function; understates precision and breaks asymptotic theory | Analyst | **H** |
| **INF-04** | Positivity diagnostics computed and reported but extreme-weight rows kept in the estimator | Diagnose: 47 rows with propensity < 0.02 → IPW weight > 50; report this in `data_report.warnings`; then proceed to fit IPW estimator with all 47 included → estimator dominated by ~50 rows | Analyst | **H** |
| **INF-05** | Missingness handled via complete-case under MAR (loses identification under selection-on-observables) | Drop 4,000 rows with any missing covariate before PSM → analytic sample is no longer representative of the target population because missingness is associated with treatment | DataEngineer + Analyst | **H** |
| **INF-06** | Multiple-comparisons correction omitted across CATE subgroups | Causal forest produces CATE for each (`X1SEX`, `X1RACE`, `X1SESQ5`) cell; report subgroup-level confidence intervals at α=0.05 across 40 cells; ~2 false positives expected and treated as findings | Analyst + Writer | **M** |
| **INF-07** | Honest splitting flag absent or off in causal-forest variance | `econml.CausalForestDML(honest=False)` (default in some configs); report variance from non-honest forest as if it had honesty's asymptotic guarantees | Analyst | **M** |

**Subtotal: 7 INF failures (5 H, 2 M).**

### 1.4 Interpretation Failures (INT)

| ID | Description | HSLS Example | Agent | Plausibility |
|---|---|---|---|---|
| **INT-01** | Causal language for results that explicitly came from selection-on-observables | "Increasing math self-efficacy *causes* a 12 percentage-point rise in college attendance" — language asserts intervention effect but identification depends on no-unmeasured-confounding | Writer | **H** |
| **INT-02** | Effect heterogeneity ignored — report ATE despite CATE varying dramatically across `X1SES` | CATE quartiles span [-0.05, +0.18] across `X1SES` × `X1RACE` cells; paper reports only ATE = 0.07 in abstract; no heterogeneity discussion | Writer | **H** |
| **INT-03** | SHAP/feature importance from outcome ML model treated as causal | Writer cites "X1MTHEFF is the third-most important feature in the XGBoost model predicting Y, with mean SHAP = 0.09" as evidence of treatment effect magnitude | Writer | **H** |
| **INT-04** | Subgroup fishing on CATE without honesty / multiple-comparisons / replication | Causal forest reports CATE = 0.31 for "Black female 1st-quintile SES" subgroup (n=27), no honest split, no replication, no MC correction; finding presented as policy-relevant | Analyst (computes) + Writer (presents) | **H** |
| **INT-05** | "Robust to one E-value" → "robust" overclaiming | Compute E-value = 1.8 for ATE estimate; report "the result is robust to unmeasured confounding"; ignore that E-value of 1.8 is consistent with confounders of moderate strength existing | Writer | **M** |
| **INT-06** | Generalizing beyond HSLS sampling frame | Discussion section asserts "national policy implications for U.S. 9th-graders" without acknowledging (a) HSLS:09 sampled in 2009 (cohort effect), (b) survey weights not applied, (c) listwise deletion changed sampling distribution | Writer | **M** |

**Subtotal: 6 INT failures (4 H, 2 M).**

**Total enumerated: 26 failures; 18 high plausibility, 8 medium.**

---

## Section 2 — Coverage Mapping (Pass 2)

Skills (column headers, kebab-case IDs):

| ID | Short |
|---|---|
| M1 | `causal-regression-adjustment` |
| M2 | `causal-propensity-score-matching` |
| M3 | `causal-inverse-probability-weighting` |
| M4 | `causal-aipw-tmle` |
| M5 | `causal-forest-cate` |
| G1 | `causal-dag-identification` |
| G2 | `causal-estimand-definition` |
| G3 | `causal-positivity-diagnostics` |
| G4 | `causal-balance-diagnostics` |
| G5 | `causal-sensitivity-unmeasured-confounding` |
| D1 | `hsls09-causal-conventions` |

(M = method, G = methodology, D = dataset.)

### 2.1 Coverage matrix

P = primary defense; S = secondary defense; · = no role.

| Failure | M1 | M2 | M3 | M4 | M5 | G1 | G2 | G3 | G4 | G5 | D1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| IDF-01 unmeasured confounding | S | · | · | S | · | **P** | · | · | · | S | · |
| IDF-02 post-treatment conditioning | · | · | · | · | · | **P** | · | · | · | · | S |
| IDF-03 collider-induced selection | · | · | · | · | · | **P** | · | · | · | · | S |
| IDF-04 positivity violation | · | · | · | · | · | · | · | **P** | S | · | · |
| IDF-05 outcome-conditional missingness | · | · | · | S | · | S | · | · | · | · | **P** |
| IDF-06 multilevel confounding | · | · | · | · | · | S | · | · | · | · | **P** |
| IDF-07 treatment-version inconsistency | · | · | · | · | · | S | **P** | · | · | · | · |
| ESC-01 PSM ATT-as-ATE | · | **P** | · | · | · | · | **P** | · | · | · | · |
| ESC-02 predicts→causal slippage in PF | · | · | · | · | · | S | **P** | · | · | · | · |
| ESC-03 causal-forest mean-CATE-as-ATE | · | · | · | · | **P** | · | **P** | · | · | · | · |
| ESC-04 sample vs population estimand | · | · | · | · | · | · | **P** | · | · | · | S |
| ESC-05 continuous treatment contrast | · | · | · | · | · | · | **P** | · | · | · | · |
| ESC-06 multi-valued treatment | · | · | · | · | · | · | **P** | · | · | · | · |
| INF-01 clustered SEs ignored (IPW) | S | · | **P** | **P** | · | · | · | · | · | · | **P** |
| INF-02 matched-pair bootstrap | · | **P** | · | · | · | · | · | · | · | · | · |
| INF-03 TMLE/AIPW influence-function variance | · | · | · | **P** | · | · | · | · | · | · | · |
| INF-04 positivity diagnostics ignored | · | S | S | S | · | · | · | **P** | · | · | · |
| INF-05 complete-case under MAR | S | S | S | **P** | · | · | · | · | · | · | S |
| INF-06 CATE multiple comparisons | · | · | · | · | **P** | · | · | · | · | · | · |
| INF-07 honest splitting variance | · | · | · | · | **P** | · | · | · | · | · | · |
| INT-01 causal language slippage | · | · | · | · | · | · | **P** | · | · | · | · |
| INT-02 effect heterogeneity ignored | · | · | · | · | **P** | · | S | · | · | · | · |
| INT-03 SHAP-as-causal | · | · | · | · | · | S | **P** | · | · | · | · |
| INT-04 CATE subgroup fishing | · | · | · | · | **P** | · | · | · | · | · | · |
| INT-05 sensitivity overclaiming | · | · | · | · | · | · | · | · | · | **P** | · |
| INT-06 over-generalization beyond HSLS | · | · | · | · | · | · | S | · | · | · | **P** |

### 2.2 Gap analysis

Every failure has at least one **P** cell. **No structural gaps.** The matrix surfaces three observations worth recording:

1. **G1 (`causal-dag-identification`) is heavily loaded** as primary defense for IDF-01/02/03/06 — all the "wrong-DAG-or-wrong-conditioning" failures concentrate here. The skill body must be substantial; it's not just "draw a DAG" but "draw a DAG, identify the estimand, document which paths are blocked."
2. **G2 (`causal-estimand-definition`) appears as primary for 7 failures** spanning ESC-01..06 + INT-01 + INT-03. This is correct: estimand discipline is the load-bearing methodology skill of V3.0. It must be tagged `rule_severity: mandatory`.
3. **D1 (`hsls09-causal-conventions`) is primary for IDF-05, IDF-06, INT-06**, and secondary for several others. The decision to merge the two candidate dataset skills into one is justified by this concentration: a single dataset skill addressing pre/post-treatment classification + selection-bias-from-attrition + cluster-extension keeps related rules together. The merge decision is recorded in §4.

### 2.3 Resolutions for borderline cases

Three borderline cases worth surfacing — none is a structural gap, just placement choices:

- **INF-05 (complete-case under MAR)** has the existing V2.0 `missingness-tiered-protocol` (mandatory) as a partial defense, but that skill's tiered decision tree is calibrated for prediction tasks (where MAR + IterativeImputer is acceptable). For causal tasks under selection-on-observables, complete-case loses identification entirely. **Resolution: M4 (`causal-aipw-tmle`) is primary** because DR methods can adjust for missingness via the propensity-of-being-observed; D1 (`hsls09-causal-conventions`) is secondary with HSLS-specific MAR diagnostics for postsecondary outcomes. The existing `missingness-tiered-protocol` is left untouched for prediction tasks.
- **INT-01 (causal language slippage)** could plausibly live in the V2.0 `paper-writing-style-rules` skill (which already has a "no causal language for correlational findings" rule). For V3.0 the slippage is the OPPOSITE — selection-on-observables results being described as causal when they should be described with explicit no-unmeasured-confounding caveats. **Resolution: G2 (`causal-estimand-definition`) is primary** (Writer-stage section "How to describe selection-on-observables results in prose"); existing `paper-writing-style-rules` is unchanged.
- **M1 ↔ G4 composition coherence:** as originally drafted, M1 (`causal-regression-adjustment`) composes G4 (`causal-balance-diagnostics`), but G4's required content was specified in propensity-context terms (pre/post-adjustment SMD where "post-adjustment" means after weighting or matching). M1 has no propensity score, so the literal G4 recipe does not apply, leaving M1's composition of G4 incoherent. **Resolution: Path A — extend G4 to cover regression-context diagnostics.** G4's required content is split into two branches: a propensity-context branch (default, used by M2/M3/M4) and a regression-context branch invoked only by M1 (covariate overlap, Cook's distance leverage, residuals-vs-treatment misspecification). M1's required content explicitly invokes the regression-context branch with the `<80%` overlap, `4/n` Cook's D, and `0.10 SD` residual-gap thresholds. Path A was chosen over Path B (decoupling M1 from G4 + adding a new `causal-regression-extrapolation-leverage` methodology skill) to keep the skill count at 11 — the regression-context diagnostics are conceptually balance-checks (residual confounding by imperfect adjustment) and belong with the propensity-context balance checks under one skill body. The added length to G4's body is bounded (≈40 lines for the regression branch); if G4 becomes hard to read coherently during 3b authoring, fall back to Path B and split.

---

## Section 3 — Per-Skill Specifications (Pass 3)

11 skills total in the original V3.0 audit; Phase 3b.12 added a 12th (`causal-data-engineer-contract`) — see the Phase 3b.12 amendment near the end of this document. Notation throughout: `composes` = this skill's `references_skills` list (V2.0 schema field); `composed by` = skills that list this one in their `references_skills`.

### 3.1 G1 — `causal-dag-identification`

**applies_to:**
- stage: `[ProblemFormulator, Analyst, Critic]`
- task_type: `[causal_soo]`
- dataset: `[]` (cross-dataset)

**Composes:** none (root methodology).
**Composed by:** M1, M2, M3, M4, M5, G2.

**Purpose (one sentence):** Force every causal study to begin with an explicit DAG that names the estimand's identifying assumptions and surfaces every observable confounder, mediator, and collider in the analytic sample.

**Required content elements:**
- Definition of "selection-on-observables" identification with the no-unmeasured-confounding assumption stated as a checkable claim, not boilerplate.
- DAG drawing instructions for the agent: nodes = treatment T, outcome Y, every covariate C in the candidate set; edges = directed; explicit instruction that the DAG must be in `research_spec.dag` as a NetworkX/DOT-serializable structure.
- Adjustment-set selection rule: the back-door criterion stated operationally — (a) close all back-door paths from T to Y, (b) include no descendants of T (to avoid post-treatment conditioning), (c) include no colliders unless the back-door path requires conditioning on one of their ancestors.
- Mandatory checklist before estimation: "for each candidate covariate, declare its temporal status relative to T (pre / post / contemporaneous) and its DAG role (confounder / mediator / collider / instrument / pre-treatment correlate)."
- Identification-failure escalation: if the DAG implies un-block-able back-door paths (i.e., an unmeasured confounder is required), the Analyst MUST stop and the Critic MUST issue REVISE; do NOT silently proceed with "best available" adjustment.
- Composition note: every method skill (M1–M5) must reference this skill via `references_skills` so the DAG check happens before any estimator runs.

**Failures prevented:** IDF-01, IDF-02, IDF-03, IDF-06, IDF-07 (S), ESC-02 (S), INT-03 (S).

**Python implementation guidance:**
- Primary library: **`dowhy`** (PyWhy). Use `dowhy.CausalModel(data, treatment, outcome, graph=DOT_string)` to construct, `model.identify_effect(method_name="default")` to derive identification, `model.refute_estimate(...)` for placebo / random common-cause / unobserved common-cause refuters.
- Key functions / classes: `dowhy.CausalModel`, `model.identify_effect`, `model.refute_estimate`, helpers in `dowhy.causal_identifier`.
- Function signatures the Analyst should produce:
  ```python
  def build_causal_model(
      df: pd.DataFrame,
      treatment: str,
      outcome: str,
      dag_dot: str,
  ) -> dowhy.CausalModel: ...

  def identify_estimand(model: dowhy.CausalModel) -> dowhy.causal_identifier.IdentifiedEstimand: ...

  def assert_no_post_treatment_in_adjustment(
      adjustment_set: list[str],
      treatment_var: str,
      var_temporal_table: dict[str, str],  # var -> 'pre' | 'post' | 'contemporaneous'
  ) -> None: ...
  ```
- Library pitfalls: `dowhy` has had API churn; pin `dowhy>=0.11`. The `graph` argument accepts DOT or NetworkX `nx.DiGraph`; prefer DOT for round-trip with `research_spec.json`. `identify_effect` will silently return `None` for unidentifiable estimands — the Analyst code must check and fail loudly. `refute_estimate` is best-effort, not proof of identification.

**Acceptance test:**
- The SKILL.md must contain: (1) the back-door criterion operationally stated, (2) the pre/post/contemporaneous temporal-status mandate for every covariate, (3) the un-block-able-back-door escalation rule, (4) the `dowhy.CausalModel` code skeleton, (5) the `references_skills: []` (no upstream deps) and a note that all method skills compose this one.
- A Writer using this skill must be able to produce a §Methods/Identification subsection that names the no-unmeasured-confounding assumption and lists every conditioning covariate with its DAG role.
- An Analyst code artifact using this skill must produce: (a) `research_spec.dag` as a DOT string, (b) `data_report.causal_identification` block with `{adjustment_set: [...], identified: bool, identification_method: str, unmeasured_confounders_named: [...]}`, (c) explicit `validation_passed: false` when `identified == False`.

---

### 3.2 G2 — `causal-estimand-definition`

**applies_to:**
- stage: `[ProblemFormulator, Analyst, Critic, Writer]`
- task_type: `[causal_soo]`
- dataset: `[]`

**Composes:** G1 (must come AFTER identification logically).
**Composed by:** M1, M2, M3, M4, M5; consulted by Writer for §Methods narrative.

**Purpose (one sentence):** Force every causal research_spec to declare its estimand explicitly — ATE / ATT / ATC / CATE — with target population (sample vs super-population) named, treatment contrast spelled out, and a binding rule that prose descriptions of results match the declared estimand.

**Required content elements:**
- Estimand taxonomy: ATE, ATT, ATC, ATU, CATE — defined formally and operationally with HSLS examples (e.g., "ATT for treatment X1MTHEFF≥0 vs <0 in the analytic sample = expected change in `X4EVRATNDCLG` if every above-median student had been below-median").
- Target-population taxonomy: (a) HSLS analytic sample (default for V3.0 — survey weights not applied), (b) HSLS sampling-frame population (would require `W4W1W2W3STU`), (c) super-population (out of scope for V3.0; see `hsls09-causal-conventions`).
- Treatment-contrast specification: every continuous treatment must declare its contrast in `research_spec.treatment_contrast` (e.g., `{type: "binary_split", threshold: "median"}`, `{type: "1sd_increase"}`, `{type: "categorical_pairwise", reference: "level_0"}`). No causal estimate may be reported without a defined contrast.
- Method → default-estimand mapping (mandatory rule):
  - Regression adjustment → ATE (with marginal standardization) OR conditional effect — must be declared
  - PSM → ATT by default (matched controls to treated); ATE only with bilateral matching
  - IPW (with stabilized weights) → ATE in the population; ATT if conditional weighting is used
  - AIPW / TMLE → ATE
  - Causal forest → CATE; ATE only via averaging over a declared population (not a default)
- Writer-stage rules ("How to describe selection-on-observables results"):
  - Required prose template: "Under the assumption that all confounders are observed and properly modeled (see DAG, §Identification), the estimated [ATE/ATT/CATE] of [treatment contrast] on [outcome] in [target population] is X.XX (95% CI [X.XX, X.XX])."
  - Forbidden phrases: "X causes Y", "the effect of X" (without estimand qualifier), "increasing X leads to Y".
  - Required hedges: "estimated [ATE/ATT]", "under the no-unmeasured-confounding assumption", "in the HSLS analytic sample".
- Mandatory tagging: `rule_severity: mandatory`. Estimand mismatch is silent corruption.

**Failures prevented:** ESC-01, ESC-02, ESC-03 (P), ESC-04, ESC-05, ESC-06, INT-01, INT-03 (P); INT-02 (S).

**Python implementation guidance:**
- Primary library: `pydantic` v2 (already in scientific-Python ecosystem) for schema validation of `research_spec.causal_estimand` block. No causal-specific library needed.
- Key data structures: dataclass `CausalEstimand(estimand_type: Literal[...], target_population: Literal[...], treatment_contrast: TreatmentContrast)`.
- Function signatures the Analyst should produce:
  ```python
  def declare_estimand(
      estimand_type: Literal["ATE", "ATT", "ATC", "ATU", "CATE"],
      target_population: Literal["sample", "frame", "super_population"],
      treatment_contrast: dict,
  ) -> dict: ...  # returns the validated estimand block

  def assert_method_estimand_compatible(
      method_name: str,
      declared_estimand: str,
  ) -> None: ...  # raises if PSM declared with ATE without bilateral matching
  ```
- Library pitfalls: none directly. Risk is that the LLM ignores the declaration and produces prose that contradicts it; the Critic checklist must catch this (see acceptance test).

**Acceptance test:**
- The SKILL.md must contain: (1) the formal definitions of ATE/ATT/ATC/CATE, (2) the method→default-estimand mapping table, (3) the Writer prose template + forbidden-phrase list, (4) `rule_severity: mandatory`.
- A Writer using this skill must be able to produce §Results / §Discussion prose that matches the template within ±5 words on the required hedges.
- An Analyst code artifact using this skill must produce: `research_spec.causal_estimand = {estimand_type, target_population, treatment_contrast}` and `results.causal_estimand_check = {declared, used_by_estimator, match: bool}` with `match=False` triggering a critical Critic issue.

---

### 3.3 G3 — `causal-positivity-diagnostics`

**applies_to:**
- stage: `[Analyst, Critic]`
- task_type: `[causal_soo]`
- dataset: `[]`

**Composes:** G1 (positivity is checked AFTER identification + adjustment-set selection).
**Composed by:** M2, M3, M4, M5.

**Purpose (one sentence):** Compute and act on common-support diagnostics for the propensity score (or treatment-assignment density for continuous T); never proceed with an estimator on a sample with documented positivity violations without explicit handling.

**Required content elements:**
- Positivity assumption stated formally: ∀ x in the support of X, 0 < P(T=1 | X) < 1.
- Diagnostics to compute:
  - Propensity-score histogram by treatment arm, overlaid (saved as `propensity_overlap.png`)
  - Trimmed common-support range (e.g., trim where min(propensity_treated, propensity_control) < 0.05)
  - Count of rows in the extreme tails (propensity < 0.05 OR > 0.95) and as fraction of n
  - Effective sample size after trimming
- Decision rule (mandatory):
  - If `extreme_tail_fraction < 0.02`: trim and proceed; document trimmed n in `data_report.warnings`
  - If `0.02 ≤ extreme_tail_fraction < 0.10`: trim, proceed, AND restrict the estimand to the overlap region (estimand becomes "ATE-on-overlap-population", named explicitly)
  - If `extreme_tail_fraction ≥ 0.10`: positivity violation — set `validation_passed: false`; Analyst MUST flag, Critic MUST issue REVISE
- Output schema:
  ```json
  "positivity_diagnostics": {
    "propensity_min": 0.0, "propensity_max": 1.0,
    "extreme_tail_count": 0, "extreme_tail_fraction": 0.0,
    "trimming_applied": true, "trimmed_n": 0,
    "decision": "proceed | proceed_with_restricted_estimand | abort"
  }
  ```
- Mandatory tagging: `rule_severity: mandatory`. Positivity ignored is silent corruption (estimator dominated by extreme weights).

**Failures prevented:** IDF-04 (P), INF-04 (P); IDF-01 (S via diagnostics), IDF-05 (handled in M4 secondarily).

**Python implementation guidance:**
- Primary library: `sklearn.linear_model.LogisticRegression` (or `sklearn.ensemble.GradientBoostingClassifier`) for propensity estimation; `matplotlib` for the overlap plot. No specialized library needed.
- Key functions / classes: `LogisticRegression`, `predict_proba`, `np.histogram`, `matplotlib.pyplot.hist`.
- Function signatures the Analyst should produce:
  ```python
  def estimate_propensity(
      df: pd.DataFrame,
      treatment_col: str,
      covariates: list[str],
      estimator: str = "logistic",  # or "gradient_boosting"
  ) -> np.ndarray: ...

  def positivity_diagnostics(
      propensity: np.ndarray,
      treatment: np.ndarray,
      tail_threshold: float = 0.05,
  ) -> dict: ...  # returns the schema above; saves overlap plot

  def apply_positivity_decision(
      df: pd.DataFrame,
      diagnostics: dict,
  ) -> tuple[pd.DataFrame, str]: ...  # returns trimmed df and decision str
  ```
- Library pitfalls: `LogisticRegression(max_iter=100)` defaults too low for n=20K HSLS; use `max_iter=1000`. `GradientBoostingClassifier` defaults are fine but check calibration before using as propensity.

**Acceptance test:**
- The SKILL.md must contain: (1) the positivity assumption formally, (2) the three-tier decision rule with thresholds, (3) the output schema, (4) `rule_severity: mandatory`.
- A Writer using this skill must be able to produce a §Methods/Positivity subsection naming the trimmed n, the extreme-tail fraction, and the resulting estimand label.
- An Analyst code artifact using this skill must produce: `propensity_overlap.png`, `results.positivity_diagnostics` populated per schema, and a `validation_passed: false` when the decision is `"abort"`.

---

### 3.4 G4 — `causal-balance-diagnostics`

**applies_to:**
- stage: `[Analyst, Critic, Writer]`
- task_type: `[causal_soo]`
- dataset: `[]`

**Composes:** G1, G3.
**Composed by:** M1 (regression-context mode), M2 (PSM, propensity-context), M3 (IPW, propensity-context), M4 (AIPW/TMLE for the propensity-side balance).

**Purpose (one sentence):** Verify that adjustment achieves a defensible match between treated and control populations — for propensity-based methods (M2/M3/M4) via standardized mean differences and Love plots, and for regression adjustment (M1) via covariate overlap, Cook's distance leverage, and residual-treatment misspecification — flagging residual imbalance, extrapolation risk, and outcome-model misfit before the headline ATE is interpreted.

**Required content elements:**

**Propensity-context diagnostics (default mode, when invoked from M2/M3/M4):**
- Standardized mean difference (SMD) defined: (mean_treated - mean_control) / pooled_SD; absolute value reported.
- Pre/post-adjustment SMD comparison required for every covariate in the adjustment set, where "post-adjustment" means after propensity weighting (M3, M4) or matching (M2).
- Threshold rules:
  - `|SMD| < 0.10` → balanced (acceptable)
  - `0.10 ≤ |SMD| < 0.25` → imbalance flag, document in warnings, consider re-specifying propensity model
  - `|SMD| ≥ 0.25` → severe imbalance, Critic issues REVISE
- Love plot (Cohen-style dot plot) showing pre vs post-adjustment SMD for every covariate; saved as `love_plot.png`.
- For categorical covariates with ≥3 levels: SMD computed per level, reported as max-across-levels.
- For interaction terms (e.g., `X1SES × X1RACE`): balance must be checked on the interaction, not just the main effects.

**Regression-context diagnostics (when invoked from M1):**

M1 (regression adjustment) does not estimate a propensity score, so the propensity-based pre/post-SMD recipe above does not apply. M1 instead requires three regression-flavored analogues of balance:

- **Covariate overlap diagnostic:** for each continuous covariate in the adjustment set, compute the overlap of the treated vs. control distributions (e.g., quantile-quantile range overlap, or the fraction of the treated covariate range that lies within the control covariate range). Flag covariates with < 80% overlap as extrapolation risks — the regression is interpolating across populations rather than comparing like with like.
- **Leverage diagnostic:** compute Cook's distance for the regression's ATE coefficient; flag any single observation with `Cook's D > 4/n` as high-leverage on the causal estimate. A handful of high-leverage rows can dominate the ATE.
- **Outcome-model misspecification diagnostic:** for linear outcome models, plot residuals vs. treatment indicator; if the residual mean differs across treatment arms by more than 0.10 SD of the residuals, flag misspecification (the outcome model is failing to absorb the treatment-confounder interaction). For logistic outcome models, use Pearson residuals stratified by treatment.

These three are M1's analogues of propensity-based balance and serve the same role: surfacing residual confounding-by-imperfect-adjustment that the headline ATE does not reveal.

**Output schema:**
```json
"balance_diagnostics": {
  "mode": "propensity | regression",
  // propensity mode (M2/M3/M4)
  "pre_adjustment_smd": {"<covariate>": 0.XX, ...},
  "post_adjustment_smd": {"<covariate>": 0.XX, ...},
  "max_residual_smd": 0.XX,
  "flagged_covariates": [...],
  "love_plot_path": "love_plot.png",
  // regression mode (M1)
  "covariate_overlap": {"<covariate>": 0.XX, ...},
  "low_overlap_covariates": [...],
  "high_leverage_rows": 0,
  "max_cook_d": 0.0,
  "residual_mean_gap_sd": 0.0,
  "misspecification_flag": false
}
```

**Failures prevented:** IDF-01 (S), IDF-04 (S — overlapping with positivity); regression-context: extrapolation, leverage, misspecification (composed by M1).

**Python implementation guidance:**
- Primary library (propensity mode): **`tableone`** (`pip install tableone`) for SMD computation across covariates with both continuous and categorical handling; or roll-your-own using `numpy` for SMD + `matplotlib` for Love plot.
- Primary library (regression mode): **`statsmodels.stats.outliers_influence.OLSInfluence`** for Cook's distance and leverage; `numpy` / `scipy.stats` for quantile-overlap and residual-gap computations.
- Note: `tableone` produces standardized differences by default but in a Table 1 format, not a balance-diagnostic format. Recommend a thin wrapper.
- Function signatures the Analyst should produce:
  ```python
  # Propensity-mode (M2/M3/M4)
  def compute_smd(
      df: pd.DataFrame,
      treatment_col: str,
      covariates: list[str],
      weights: np.ndarray | None = None,
  ) -> dict[str, float]: ...

  def love_plot(
      pre_smd: dict[str, float],
      post_smd: dict[str, float],
      output_path: str,
      threshold: float = 0.10,
  ) -> None: ...

  # Regression-mode (M1)
  def covariate_overlap(
      df: pd.DataFrame,
      treatment_col: str,
      covariates: list[str],
  ) -> dict[str, float]: ...  # fraction of treated range inside control range

  def regression_leverage_diagnostics(
      fitted_ols_results,  # statsmodels OLSResults
  ) -> dict: ...  # uses OLSInfluence.cooks_distance; returns {max_cook_d, high_leverage_rows}

  def residual_treatment_gap(
      fitted_ols_results,
      treatment: np.ndarray,
  ) -> float: ...  # |mean(resid|T=1) - mean(resid|T=0)| / sd(resid)
  ```
- Library pitfalls:
  - `tableone`'s SMD computation does not natively handle weighted samples (for IPW); use a custom weighted SMD: `(weighted_mean_treated - weighted_mean_control) / weighted_pooled_SD`.
  - `OLSInfluence.cooks_distance` returns a tuple `(d, p)`; the Cook's distance values are `d[0]`. Compute on the fitted regression that includes both treatment and adjustment covariates so leverage is measured on the ATE-bearing model, not a treatment-free outcome model.
  - For logistic outcome models in M1, `OLSInfluence` does not apply; use `GLMInfluence` from the same submodule.

**Acceptance test:**
- The SKILL.md must contain: (1) the SMD formula, (2) the three-tier threshold rule, (3) the love-plot specification, (4) the per-level SMD rule for categoricals, (5) the regression-context branch with covariate-overlap (`<80%`), Cook's distance (`>4/n`), and residual-treatment-gap (`>0.10 SD`) rules and the explicit note that M1 invokes G4 in regression-context mode.
- A Writer using this skill must be able to produce a §Methods/Balance subsection with the pre/post SMD comparison and a reference to the Love plot figure (propensity mode), or a §Methods/Diagnostics subsection naming low-overlap covariates, the count of high-leverage rows, and the residual-treatment gap (regression mode).
- An Analyst code artifact using this skill must produce: in propensity mode, `love_plot.png` and `results.balance_diagnostics` populated with `mode: "propensity"` per schema, `validation_passed: false` when `max_residual_smd >= 0.25`; in regression mode (when invoked by M1), `results.balance_diagnostics` populated with `mode: "regression"` per schema, with low-overlap / high-leverage / misspecification flags appended to `results.warnings`.

---

### 3.5 G5 — `causal-sensitivity-unmeasured-confounding`

**applies_to:**
- stage: `[Analyst, Critic, Writer]`
- task_type: `[causal_soo]`
- dataset: `[]`

**Composes:** G1, G2.
**Composed by:** M1, M2, M3, M4, M5.

**Purpose (one sentence):** Quantify how strong an unmeasured confounder would need to be to overturn the headline causal estimate — via E-value (mandatory) and Rosenbaum bounds (where applicable) — and report results with calibrated language that prevents "robust to one sensitivity analysis → robust" overclaiming.

**Required content elements:**
- E-value: defined formally (the minimum strength of association, on the risk-ratio scale, that an unmeasured confounder would need to have with both treatment and outcome to fully explain away the observed effect).
- E-value computation rule: every causal point estimate AND its CI lower bound must report an E-value.
- E-value interpretation table (mandatory in Writer's §Sensitivity subsection):
  - E-value < 1.5 → "result is fragile to unmeasured confounding"
  - 1.5 ≤ E-value < 2.5 → "result requires moderate-strength unmeasured confounder to overturn"
  - E-value ≥ 2.5 → "result requires strong unmeasured confounder to overturn (still possible)"
  - **Forbidden phrase: "result is robust to unmeasured confounding"** — any E-value < ∞ is consistent with some unmeasured confounder existing.
- Rosenbaum bounds: defined for matched designs (PSM); compute Γ such that conclusions become inconclusive; report alongside E-value when matching is the estimator.
- DoWhy refuters (mandatory): run at least two of {`random_common_cause`, `placebo_treatment`, `data_subset_refuter`, `add_unobserved_common_cause`} from `dowhy`; report whether estimate stays significant after each. **Phase 3b.14 amendment**: G5 now also contains a prescriptive "DoWhy refuter invocation (mandatory for causal_soo)" section with the four-step build → CausalModel → identify_effect → estimate_effect → refute_estimate sequence, the column-name-as-node-ID rule, the `dowhy_refuters` output schema, and exception-capture guidance. See the Phase 3b.14 amendment near the end of this document. **Phase 3b.16 refinement**: the graph format was switched from DOT-string to NetworkX-DiGraph after 3b.15 surfaced F-3b15-DOWHY-PYGRAPHVIZ-DEPENDENCY-GAP. See the Phase 3b.16 amendment near the end of this document.
- Mandatory tagging: `rule_severity: mandatory`. Sensitivity-analysis omission is structural incompleteness for any causal study.

**Failures prevented:** IDF-01 (S), INT-05 (P).

**Python implementation guidance:**
- Primary library for E-value: `EValue` does not have a clean Python implementation; recommend a 10-line custom function based on the VanderWeele & Ding (2017) formula. Alternatively use the R package `EValue` via `rpy2` (extra dependency, only if R is acceptable).
- Recommended primary: **custom Python** (10 lines for E-value; well-defined formula).
- Primary library for Rosenbaum bounds: no clean Python option; `pymatch` has been unmaintained since 2018. Recommend custom implementation following Rosenbaum (2002) for the Wilcoxon-signed-rank-based bound on matched pairs.
- Primary library for refutation tests: `dowhy.causal_refuters.{RandomCommonCause, PlaceboTreatmentRefuter, DataSubsetRefuter, AddUnobservedCommonCause}`. Active and maintained.
- Function signatures the Analyst should produce:
  ```python
  def evalue_for_estimate(
      point_estimate_rr: float,  # risk ratio scale
      ci_lower_rr: float,
  ) -> dict: ...  # {"evalue": float, "evalue_ci_lower": float}

  def rosenbaum_bounds_matched(
      matched_pairs_treated_outcome: np.ndarray,
      matched_pairs_control_outcome: np.ndarray,
      gamma_grid: list[float] = [1.0, 1.5, 2.0, 2.5, 3.0],
  ) -> dict: ...

  def run_dowhy_refuters(
      estimate: dowhy.causal_estimator.CausalEstimate,
      refuter_names: list[str],
  ) -> dict: ...  # {refuter_name: {p_value, conclusion}}
  ```
- Library pitfalls: VanderWeele E-value formula assumes risk-ratio-scale estimate; for continuous outcomes (e.g., GPA), apply transformation per VanderWeele & Ding (2017) Appendix. DoWhy refuters can be slow on large samples (sample down to n=10K for refuter calls).

**Acceptance test:**
- The SKILL.md must contain: (1) E-value formula, (2) the three-tier interpretation table, (3) the forbidden "robust" phrase rule, (4) the DoWhy refuter list with required count ≥2, (5) `rule_severity: mandatory`.
- A Writer using this skill must be able to produce a §Discussion/Sensitivity subsection with E-value, refuter results, and explicit calibrated language (no "robust").
- An Analyst code artifact using this skill must produce: `results.sensitivity_analysis = {evalue, evalue_ci_lower, rosenbaum_gamma_breakpoint, refuter_results: [...]}`.

---

### 3.6 D1 — `hsls09-causal-conventions`

**applies_to:**
- stage: `[ProblemFormulator, DataEngineer, Analyst, Critic, Writer]`
- task_type: `[causal_soo]`
- dataset: `[hsls09_public]`

**Composes:** `cluster-id-reconstruction-from-fingerprints` (existing V2.0 methodology skill).
**Composed by:** M1, M2, M3, M4, M5 (all method skills compose this when running on HSLS).

**Purpose (one sentence):** HSLS-specific causal conventions covering treatment-relevant variable patterns, pre/post-treatment temporal classification, MNAR diagnostics for postsecondary outcomes, and clustered-SE handling for causal estimators.

**Required content elements:**
- Treatment-relevant variable inventory:
  - Continuous attitudinal scales suitable as treatments (`X1MTHID`, `X1MTHEFF`, `X1SCHOOLBEL`, `X1SCIID`, etc.) — these are **psychometric scales** that require thresholding or binning into a defined contrast (no "per-unit" causal interpretation).
  - Course-taking / attainment indicators suitable as treatments (`X3TCREDMAT`, `X4HSCOMPSTAT`, `X4EVRATNDCLG`).
  - Variables NEVER suitable as treatments: weights (`W*`), sampling indicators (`*QSTAT`), demographic protected attributes used unalterably (`X1SEX`, `X1RACE` — use only as moderators).
- Pre/post-treatment temporal classification table:
  - Pre-treatment for any X1 treatment: only `X1` baseline variables AND demographics
  - Pre-treatment for any X2 treatment: X1 + X2_baseline (excluding X2 outcomes)
  - Post-treatment relative to a given treatment: any variable measured in a wave after the treatment wave (per `hsls09-temporal-ordering`)
  - Mandatory: every covariate in `research_spec.adjustment_set` must have its temporal status declared in `research_spec.covariate_temporal_table`.
- Selection-bias-from-attrition rules for postsecondary outcomes:
  - `X4*` outcomes have ~26% MAR/MNAR missingness; postsecondary `X5*` have 50%+
  - For causal targets on postsecondary outcomes, the analytic sample is structurally restricted to respondents — report both ITT (intent-to-treat-analogue, with X4SQSTAT-respondents only) and the as-treated equivalent.
  - Recommend running parallel IPW-for-missingness analyses to bound the missingness-induced bias.
- Clustered-SE handling for causal estimators (extends `cluster-id-reconstruction-from-fingerprints`):
  - For IPW: cluster-robust SEs at school level via `statsmodels` `cov_type='cluster'` with `cluster_groups=pseudo_school_id`.
  - For PSM: cluster bootstrap on matched pairs aggregated at school level (matched pair as unit, school as cluster).
  - For AIPW/TMLE: influence function variance with clustered correction (sum within clusters then variance across).
  - For causal forest: honest splits + cluster-robust forest variance via `econml`'s `inference="auto"` doesn't natively cluster — note this limitation; recommend cluster-bootstrap of the ATE estimate.
- Survey-weights handling:
  - V3.0 default: do NOT apply HSLS survey weights. Estimand is the analytic-sample marginal effect, not the population marginal effect.
  - Writer must state this explicitly in §Methods and §Limitations.
  - Composes `hsls09-survey-weights-limitations-paragraph` (existing V2.0 writing skill).
- Mandatory tagging: `rule_severity: mandatory`. Pre/post misclassification is silent corruption (introduces post-treatment bias).

- **Phase 3b.6 amendment**: added the Analyst-side "Encoded-column lookup" rule (`resolve_encoded_columns(varname, df_columns)` prefix-match) so the Analyst code never silently misses one-hot variants of adjustment-set categoricals. See the Phase 3b.6 amendment near the end of this document.

- **Phase 3b.18 amendment**: added the DataEngineer-side "Encoding-type discipline" rule — variables tagged `type=continuous` in the registry MUST NOT be one-hot encoded; type-aware dispatch produces deterministic encoding behavior. See the Phase 3b.18 amendment near the end of this document. (This complements the 3b.6 Analyst-side rule: 3b.6 reads whatever encoding the DE wrote; 3b.18 governs what the DE writes in the first place.)

**Failures prevented:** IDF-05 (P), IDF-06 (P), INT-06 (P); IDF-02 (S), IDF-03 (S), ESC-04 (S), INF-01 (P with clustered-SE rules), INF-05 (S); F-3b15-DE-CONTINUOUS-AS-CATEGORICAL (P; 3b.18 amendment).

**Python implementation guidance:**
- Primary library: `pandas` + `statsmodels` (`cov_type='cluster'`).
- Reference helper for cluster-bootstrap of causal estimates:
  ```python
  def cluster_bootstrap_ate(
      df: pd.DataFrame,
      cluster_col: str,
      ate_estimator_fn: Callable[[pd.DataFrame], float],
      n_boot: int = 1000,
      random_state: int = 42,
  ) -> tuple[float, float]: ...  # returns (ci_lower, ci_upper)

  def declare_covariate_temporal_status(
      covariates: list[str],
      treatment_wave: str,
  ) -> dict[str, Literal["pre", "post", "contemporaneous"]]: ...
  ```
- Library pitfalls: `statsmodels` `cov_type='cluster'` requires `groups` argument as 1-D array of cluster IDs; pseudo-school-IDs must be aligned by `iloc` to the regression's exogenous matrix.

**Acceptance test:**
- The SKILL.md must contain: (1) the treatment-relevant variable inventory with examples, (2) the pre/post-treatment temporal table, (3) the postsecondary-attrition rules, (4) the clustered-SE handling table per estimator, (5) the no-survey-weights default, (6) `rule_severity: mandatory`, (7) `references_skills: [cluster-id-reconstruction-from-fingerprints]`.
- A Writer using this skill must be able to produce a §Methods subsection naming the analytic-sample estimand, the no-survey-weights choice, and the school-clustering correction method used.
- An Analyst code artifact using this skill must produce: `data_report.causal_covariate_temporal_table`, `results.causal_estimate.cluster_se_method`, and `results.causal_estimate.cluster_se_ci`.

---

### 3.7 M1 — `causal-regression-adjustment`

**applies_to:**
- stage: `[Analyst]`
- task_type: `[causal_soo]`
- dataset: `[]`

**Composes:** G1, G2, G4, G5, D1.
**Composed by:** none (terminal method).

**Purpose (one sentence):** Estimate ATE/ATT via outcome regression on (treatment, adjustment-set covariates), with cluster-robust SEs and explicit marginal-vs-conditional reporting.

**Required content elements:**
- Estimator definition: outcome model `Y ~ T + adjustment_set` (linear / logistic depending on outcome), then for ATE: marginal-standardized prediction over the analytic sample with treatment fixed at 1 vs 0.
- Choice rule: linear for continuous Y, logistic for binary Y. Report on the difference scale (risk difference) by default; risk ratio if requested.
- Interaction-with-treatment specification: by default, no T × covariate interactions (effect homogeneity assumption); if heterogeneity is suspected, switch to M5 (causal forest).
- SE rule: cluster-robust at school level (pseudo-school-IDs from D1).
- Reporting: ATE (95% CI, cluster-robust), R² for the outcome model (sanity check, not effect interpretation), residual diagnostics if linear.
- Mandatory rule: this skill is the **baseline / comparator**. Every causal study should run M1 alongside its primary method. If M1's ATE diverges from the primary method by > 50%, flag in `results.warnings`.
- **G4 in regression-context mode (mandatory):** after fitting the outcome model, invoke G4's regression-context branch (covariate overlap, Cook's distance leverage, residuals-vs-treatment misspecification — see §3.4). Specifically: (a) compute per-covariate overlap of treated vs. control distributions and flag any covariate with < 80% overlap, (b) compute Cook's distance from the fitted regression (via `OLSInfluence` / `GLMInfluence`) and flag any observation with `Cook's D > 4/n`, (c) compute the residual-mean gap across treatment arms in SDs of the residuals and flag if > 0.10. Append all flags to `results.warnings`; populate `results.balance_diagnostics` with `mode: "regression"` and the regression-mode fields per G4's output schema. M1's composition of G4 is interpreted in this regression-context mode, not the propensity mode.

**Failures prevented:** IDF-01 (S), INF-01 (S), INF-05 (S), ESC-01 (S); regression-context extrapolation, leverage, and outcome-model misspecification (via G4 regression-context diagnostics).

**Python implementation guidance:**
- Primary library: **`statsmodels`** (`OLS` / `GLM` with `cov_type='cluster'`). Clean, mature, supports clustering and weights.
- Function signatures:
  ```python
  def regression_adjustment_ate(
      df: pd.DataFrame,
      treatment_col: str,
      outcome_col: str,
      covariates: list[str],
      cluster_col: str,
      family: Literal["gaussian", "binomial"] = "gaussian",
  ) -> dict: ...
      # returns {"ate": float, "ci_lower": float, "ci_upper": float,
      #          "model_summary": str, "outcome_model_r2": float}
  ```
- Library pitfalls: `statsmodels` GLM interprets the link function — for risk-difference reporting on binary Y, use `Binomial(link=identity)` rather than the default logit link.

**Acceptance test:**
- The SKILL.md must contain: (1) outcome model + marginal standardization recipe, (2) the no-T×covariate-interactions default, (3) the cluster-robust SE rule, (4) the comparator role, (5) the explicit G4-regression-context invocation (covariate overlap, Cook's D, residual-treatment gap) with the `<80%` / `4/n` / `0.10 SD` thresholds.
- An Analyst code artifact using this skill must produce: `results.estimates.regression_adjustment = {ate, ci_lower, ci_upper, cluster_se_method, model_diagnostics}` AND `results.balance_diagnostics` populated in regression mode.

---

### 3.8 M2 — `causal-propensity-score-matching`

**applies_to:**
- stage: `[Analyst]`
- task_type: `[causal_soo]`
- dataset: `[]`

**Composes:** G1, G2, G3, G4, G5, D1.
**Composed by:** none.

**Purpose (one sentence):** Estimate ATT via 1:k nearest-neighbor matching on the propensity score with caliper, cluster-aware bootstrap SEs (or Abadie-Imbens analytic SEs), and pre/post balance reporting.

**Required content elements:**
- Matching specification: 1:1 nearest-neighbor by default; 1:5 alternative if controls plentiful; with-replacement allowed when control pool is small (see Matching algorithm below).
- Caliper rule: ≤ 0.2 SD of the propensity score (Austin 2011). Unmatched treated units → drop and report.
- Estimand: **ATT by default** (matched controls to treated); explicit declaration required per G2.

**Matching algorithm (locked):**
- **Greedy 1:1 nearest-neighbor** matching on the propensity score (not optimal/Hungarian matching). Justification: greedy is the EDM/AIED publication standard; optimal matching's gains are marginal at HSLS scale (n≈20K) and not worth the extra dependency cost (`networkx` min-cost-flow or `scipy.optimize.linear_sum_assignment` + custom adapter).
- **Match order:** ascending propensity score among treated units (treated unit with lowest propensity matched first). Deterministic; reproducible.
- **Tie handling:** when two control units are equidistant from a treated unit on propensity, break ties by ascending row index (`df.index` order). Deterministic.
- **Replacement threshold:** matching is without replacement by default. Switch to with-replacement only when `n_control / n_treated < 5`. Document the switch in `data_report.warnings`.
- **Caliper enforcement:** caliper applied as post-hoc filter on returned `NearestNeighbors` distances; treated units whose nearest control exceeds the caliper are dropped and counted in `n_unmatched_treated`.

- SE rule:
  - Default: cluster-bootstrap on matched pairs (resample matched pairs as units, with school-level clustering). Per `INF-02`, do NOT bootstrap student rows independently.
  - Alternative: Abadie-Imbens analytic SE for matching estimators (no Python implementation; flag if requested and recommend custom).
- Balance check: G4 (`causal-balance-diagnostics`) must run after matching; max residual SMD ≥ 0.10 → re-specify propensity model and re-match (up to 2 iterations).
- Output schema:
  ```json
  "psm_results": {
    "n_treated": 0, "n_control_matched": 0, "n_unmatched_treated": 0,
    "caliper_used": 0.2, "match_ratio": "1:1",
    "att_estimate": 0.0, "att_ci_lower": 0.0, "att_ci_upper": 0.0,
    "se_method": "cluster_bootstrap_matched_pairs",
    "balance_max_residual_smd": 0.0
  }
  ```

**Failures prevented:** ESC-01 (P), INF-02 (P), INF-04 (S), INF-05 (S).

**Python implementation guidance:**
- Primary library: **custom implementation** using `sklearn.neighbors.NearestNeighbors` for matching + `sklearn.linear_model.LogisticRegression` or `GradientBoostingClassifier` for propensity. Justification: `psmpy` (the only PyPI option) is single-author, low test coverage, last commit 2023; `causalml.match.NearestNeighborMatch` exists but has limited caliper / matching-ratio control.
- Custom implementation is < 100 LOC and gives full control over caliper, replacement, and matched-pair tracking (needed for the cluster bootstrap).
- Function signatures:
  ```python
  def estimate_propensity_for_matching(
      df: pd.DataFrame,
      treatment_col: str,
      covariates: list[str],
  ) -> np.ndarray: ...

  def match_nearest_neighbor(
      propensity: np.ndarray,
      treatment: np.ndarray,
      caliper_sd: float = 0.2,
      ratio: int = 1,
      with_replacement: bool = False,  # auto-switch to True if n_control/n_treated < 5
  ) -> dict: ...
      # Implementation locks (see Matching algorithm in §3.8 required content):
      #   - greedy 1:1 NN, NOT optimal/Hungarian
      #   - match order: ascending propensity among treated units
      #   - tie-break: ascending row index (df.index)
      #   - caliper: post-hoc filter on returned NearestNeighbors distances;
      #     drop treated units whose nearest control exceeds caliper_sd * sd(propensity)
      #   - replacement: respect with_replacement flag; auto-flip to True if
      #     n_control/n_treated < 5 (record flip in data_report.warnings)
      # returns {"matched_pairs": [(treated_idx, control_idx), ...],
      #          "unmatched_treated": [idx, ...],
      #          "with_replacement_used": bool}

  def att_from_matched_pairs(
      df: pd.DataFrame,
      matched_pairs: list[tuple[int, int]],
      outcome_col: str,
  ) -> float: ...

  def cluster_bootstrap_att(
      matched_pairs: list[tuple[int, int]],
      pair_cluster_ids: list[int],  # each pair's school cluster
      df: pd.DataFrame,
      outcome_col: str,
      n_boot: int = 1000,
  ) -> tuple[float, float]: ...
  ```
- Library pitfalls: `sklearn.neighbors.NearestNeighbors` doesn't natively support caliper — implement as post-hoc filter on returned distances. `psmpy` v1.x has bugs in caliper handling; do NOT use as drop-in.

**Acceptance test:**
- The SKILL.md must contain: (1) matching spec + caliper rule, (2) ATT-by-default estimand declaration, (3) cluster-bootstrap-on-pairs SE rule, (4) the re-match-on-imbalance loop, (5) the output schema, (6) the locked-mechanics block: greedy 1:1, ascending-propensity match order, ascending-index tie-break, `n_control/n_treated < 5` replacement threshold, post-hoc caliper enforcement.
- An Analyst code artifact using this skill must produce: `results.estimates.psm` populated per schema; `validation_passed: false` if `balance_max_residual_smd >= 0.25` after 2 re-match iterations.

---

### 3.9 M3 — `causal-inverse-probability-weighting`

**applies_to:**
- stage: `[Analyst]`
- task_type: `[causal_soo]`
- dataset: `[]`

**Composes:** G1, G2, G3, G4, G5, D1.
**Composed by:** none.

**Purpose (one sentence):** Estimate ATE via stabilized IPW with weight trimming + cluster-robust SEs from a weighted regression on treatment.

**Required content elements:**
- Weights: stabilized IPW (`SW = P(T=1) / e(X)` for treated, `SW = P(T=0) / (1 - e(X))` for controls); reduces variance vs unstabilized.
- Weight trimming: per G3 positivity rule, trim rows where propensity < 0.05 OR > 0.95. Document trimmed n.
- Estimator: weighted regression of `Y ~ T` (no covariates beyond T; weights do the adjustment) with `weights=SW` and `cov_type='cluster'` for cluster-robust SE.
- Estimand: **ATE in the analytic sample** by default (stabilized weights target ATE); explicit declaration per G2.
- Diagnostic: report the effective sample size (ESS = (sum of weights)² / sum of weights²) and flag if ESS < 0.5 × n (weight degeneracy warning).
- **Weighted balance check (mandatory):** after computing stabilized weights, run G4 (`causal-balance-diagnostics`) with `weights=stabilized_weights` and report **weighted** SMDs for every covariate in the adjustment set. Apply the same three-tier threshold rule as G4 (`|SMD|<0.10` balanced, `0.10–0.25` flag, `≥0.25` REVISE). Save `love_plot_ipw.png` showing pre-weighting vs. post-weighting SMDs.
- Output schema:
  ```json
  "ipw_results": {
    "weight_max": 0.0, "weight_min": 0.0, "weight_ess": 0,
    "trimmed_n": 0, "trimming_threshold": 0.05,
    "ate_estimate": 0.0, "ate_ci_lower": 0.0, "ate_ci_upper": 0.0,
    "se_method": "cluster_robust",
    "stabilized_weights": true,
    "weighted_balance": {
      "max_post_weighted_smd": 0.0,
      "flagged_covariates": [],
      "love_plot_path": "love_plot_ipw.png"
    }
  }
  ```

**Failures prevented:** INF-01 (P), INF-04 (S), INF-05 (S), ESC-01 (S — explicit ATE rule); residual covariate imbalance under weighting (via G4 weighted balance check).

**Python implementation guidance:**
- Primary library: `statsmodels.regression.linear_model.WLS` or `statsmodels.GLM` with `freq_weights=stabilized_weights` and `cov_type='cluster'`. Mature, supports clustering.
- Alternative: `causalml.inference.meta.LRSRegressor` provides IPW directly but lacks cluster-robust SEs.
- Function signatures:
  ```python
  def compute_stabilized_weights(
      propensity: np.ndarray,
      treatment: np.ndarray,
  ) -> np.ndarray: ...

  def trim_extreme_weights(
      df: pd.DataFrame,
      propensity: np.ndarray,
      weights: np.ndarray,
      threshold: float = 0.05,
  ) -> tuple[pd.DataFrame, np.ndarray]: ...

  def ipw_ate(
      df: pd.DataFrame,
      treatment_col: str,
      outcome_col: str,
      weights: np.ndarray,
      cluster_col: str,
  ) -> dict: ...  # returns ipw_results schema above
  ```
- Library pitfalls: `statsmodels` GLM does not always honor `cov_type='cluster'` correctly with non-identity links; for binary outcomes, use linear probability + clustered SE (then the ATE is the risk difference) for safety.

**Acceptance test:**
- The SKILL.md must contain: (1) stabilized-weights formula, (2) trimming rule referencing G3, (3) ESS degeneracy check, (4) cluster-robust SE rule, (5) the output schema, (6) the explicit weighted-balance-check bullet invoking G4 with `weights=stabilized_weights` and the three-tier SMD threshold rule.
- An Analyst code artifact using this skill must produce: `results.estimates.ipw` per schema (including the `weighted_balance` block) and `love_plot_ipw.png`; `validation_passed: false` when `weight_ess < 0.5 * n` OR `max_post_weighted_smd >= 0.25`.

---

### 3.10 M4 — `causal-aipw-tmle`

**applies_to:**
- stage: `[Analyst]`
- task_type: `[causal_soo]`
- dataset: `[]`

**Composes:** G1, G2, G3, G4, G5, D1.
**Composed by:** none.

**Purpose (one sentence):** Estimate ATE via doubly-robust methods — AIPW (single-step DR) and TMLE (targeted update) — with influence-function variance, sample-splitting / cross-fitting for nuisance estimation, and missingness-as-IPW handling.

**Required content elements:**
- AIPW estimator: `AIPW = E[μ₁(X) - μ₀(X)] + E[T·(Y - μ₁(X))/e(X)] - E[(1-T)·(Y - μ₀(X))/(1-e(X))]`.
- TMLE estimator: estimate initial outcome model, compute "clever covariate" `H(T,X) = T/e(X) - (1-T)/(1-e(X))`, fluctuate via parametric submodel, target.
- Cross-fitting (mandatory for ML nuisance estimators): K=5 folds; outcome model and propensity model fit on K-1 folds, evaluated on held-out fold; estimands averaged over folds.
- Variance: empirical variance of the influence function, NOT bootstrap-of-mean. Per `INF-03`.
- Missingness handling: when outcome `Y` is MAR, augment AIPW/TMLE with a missingness propensity model `P(R=1|X)`; the estimator becomes IPCW-AIPW. This is the primary defense against `INF-05`.
- Output schema:
  ```json
  "aipw_tmle_results": {
    "method": "AIPW | TMLE",
    "ate_estimate": 0.0,
    "ate_ci_lower": 0.0, "ate_ci_upper": 0.0,
    "se_method": "influence_function",
    "cross_fitting_folds": 5,
    "outcome_model_cv_score": 0.0,
    "propensity_model_cv_score": 0.0,
    "missingness_adjusted": false
  }
  ```

**Failures prevented:** INF-03 (P), INF-05 (P); IDF-01 (S), INF-04 (S).

**Python implementation guidance:**
- AIPW primary library: **`econml.dr.DRLearner`**. Active, well-tested, supports arbitrary sklearn-compatible nuisance estimators and cross-fitting. Returns ATE + IF variance.
- TMLE primary library: **`zEpid`** (`pip install zEpid`, currently v0.9.x). The `zepid.causal.doublyrobust.tmle.TMLE` class implements the targeting step and IF variance. Active maintenance as of 2024 but smaller community than econml.
  - Caveat: `zEpid` API has been stable in recent versions but the maintainer is single-person. Pin `zEpid>=0.9.0`.
- **EconML does NOT implement TMLE** (verified in econml docs as of 0.15.x). The AIPW + TMLE skill must split the implementation: `econml.dr.DRLearner` for AIPW, `zEpid` for TMLE.
- Fallback path if `zEpid` is unavailable / unstable: custom TMLE following Targeted Learning textbook (Van der Laan & Rose 2011) — ~150 LOC for binary T + Y; tractable but adds maintenance burden. Recommend `zEpid` primary, custom secondary.
- Function signatures:
  ```python
  def aipw_ate(
      df: pd.DataFrame,
      treatment_col: str,
      outcome_col: str,
      covariates: list[str],
      cluster_col: str,
      outcome_model: BaseEstimator = GradientBoostingRegressor(),
      propensity_model: BaseEstimator = LogisticRegression(),
      cv_folds: int = 5,
  ) -> dict: ...

  def tmle_ate(
      df: pd.DataFrame,
      treatment_col: str,
      outcome_col: str,
      covariates: list[str],
      cluster_col: str,
      missingness_col: str | None = None,  # if MAR adjustment needed
  ) -> dict: ...
  ```
- Library pitfalls:
  - `econml.dr.DRLearner` requires explicit `model_propensity` and `model_regression` arguments; defaults are sklearn `LogisticRegression()` which underfits attendances on n=20K.
  - `zEpid` TMLE assumes binary T; for continuous T, use AIPW only.
  - Cross-fitting fold assignments must respect school clusters (use `GroupKFold` with school IDs as groups).

**Acceptance test:**
- The SKILL.md must contain: (1) AIPW + TMLE estimator formulas, (2) the cross-fitting K=5 mandate with cluster-respecting folds, (3) IF variance rule, (4) the IPCW-AIPW for MAR handling, (5) the explicit `econml` (AIPW) + `zEpid` (TMLE) library split, (6) the output schema.
- An Analyst code artifact using this skill must produce: `results.estimates.aipw` AND `results.estimates.tmle` (both, when binary T); compare ATE estimates and flag if divergence > 30% (DR property suggests they should agree).

---

### 3.11 M5 — `causal-forest-cate`

**applies_to:**
- stage: `[Analyst]`
- task_type: `[causal_soo]`
- dataset: `[]`

**Composes:** G1, G2, G3, G4, G5, D1.
**Composed by:** none.

**Purpose (one sentence):** Estimate Conditional Average Treatment Effects (CATE) via causal forest with honest splitting; surface effect heterogeneity; report ATE-via-averaging only with explicit target-population label and honest-CI variance.

**Required content elements:**
- Estimator: `econml.dml.CausalForestDML` with `honest=True` (mandatory), cross-fitting K=5.
- CATE outputs:
  - `cate_predictions[i]` for every test-set unit
  - CATE distribution: histogram, percentiles (10/25/50/75/90), saved as `cate_distribution.png`
  - CATE by subgroup: `(X1SEX, X1RACE, X1SESQ5)` cells, with cell n; flagged if any cell n < 100
- ATE-via-averaging: `mean(cate_predictions[overlap_region])` reported as **"ATE-on-overlap-population"**, NOT as "ATE" (per ESC-03).
  - Variance: from honest splitting, via `econml`'s `inference="auto"` with bootstrap of forest.
- Heterogeneity tests:
  - BLP (best linear projection of CATE on covariates) via `econml.cate_interpreter`
  - Variance ratio test: var(CATE) / var(ATE) > 2 → heterogeneity present, report
- Subgroup honesty rule (mandatory): subgroups defined a priori in `research_spec.subgroup_analyses`; CATE for unspecified subgroups is exploratory and must be labeled as such in the paper.
- Multiple-comparisons correction (mandatory): when reporting subgroup CATEs, apply Benjamini-Hochberg FDR correction at q=0.05; report adjusted p-values.
- Output schema:
  ```json
  "causal_forest_results": {
    "ate_on_overlap": 0.0, "ate_on_overlap_ci": [0.0, 0.0],
    "cate_percentiles": {"10": 0.0, "50": 0.0, "90": 0.0},
    "cate_variance_ratio": 0.0,
    "subgroup_cate": {"<subgroup>": {"cate": 0.0, "ci_adj": [0.0, 0.0], "n": 0}},
    "blp_coefficients": {...},
    "honest": true,
    "cv_folds": 5
  }
  ```

**Failures prevented:** ESC-03 (P), INT-02 (P), INT-04 (P), INF-06 (P), INF-07 (P).

**Python implementation guidance:**
- Primary library: **`econml.dml.CausalForestDML`**. Best-in-class for CATE; active maintenance; honest splitting first-class.
- Function signatures:
  ```python
  def fit_causal_forest(
      df: pd.DataFrame,
      treatment_col: str,
      outcome_col: str,
      covariates: list[str],
      n_estimators: int = 500,
      honest: bool = True,
      cv_folds: int = 5,
  ) -> CausalForestDML: ...

  def cate_distribution(
      forest: CausalForestDML,
      X: pd.DataFrame,
  ) -> dict: ...  # percentiles + histogram + plot

  def subgroup_cate(
      forest: CausalForestDML,
      X: pd.DataFrame,
      subgroup_attrs: list[str],
      fdr_q: float = 0.05,
  ) -> dict: ...  # subgroup CATEs with BH-corrected CIs

  def best_linear_projection(
      forest: CausalForestDML,
      X: pd.DataFrame,
  ) -> dict: ...  # BLP coefficients
  ```
- Library pitfalls:
  - `CausalForestDML(honest=False)` is the default in some `econml` versions; MUST set `honest=True` explicitly.
  - `econml`'s `inference` argument controls variance; use `inference="auto"` (not `"bootstrap"` for forest variance).
  - `n_estimators=100` is too few for n=20K HSLS; recommend `n_estimators >= 500`.
  - Cross-fitting folds must respect school clusters (use `cv=GroupKFold(5)` with school IDs).

**Acceptance test:**
- The SKILL.md must contain: (1) `honest=True` mandate, (2) ATE-via-averaging label rule (ATE-on-overlap-population), (3) BH FDR correction for subgroup CATEs, (4) min-cell-n flag (n<100), (5) cluster-respecting cross-fitting, (6) the output schema.
- An Analyst code artifact using this skill must produce: `cate_distribution.png`, `results.estimates.causal_forest` per schema; `results.warnings` populated with any subgroup with n<100.

---

## Section 4 — Cross-cutting Concerns (Pass 4)

### 4.1 Composition graph

```
[ Methodology layer ]
G1 (causal-dag-identification)         ← root
  ↑ composed by all M1-M5 + G2

G2 (causal-estimand-definition)
  composes G1
  ↑ composed by all M1-M5

G3 (causal-positivity-diagnostics)
  composes G1
  ↑ composed by M2, M3, M4, M5

G4 (causal-balance-diagnostics)
  composes G1, G3
  ↑ composed by M2, M3, M4

G5 (causal-sensitivity-unmeasured-confounding)
  composes G1, G2
  ↑ composed by all M1-M5

[ Dataset layer ]
D1 (hsls09-causal-conventions)
  composes cluster-id-reconstruction-from-fingerprints (existing V2.0)
  ↑ composed by all M1-M5

[ Method layer ]
M1 (causal-regression-adjustment)  composes G1, G2, G4, G5, D1
M2 (causal-propensity-score-matching)  composes G1, G2, G3, G4, G5, D1
M3 (causal-inverse-probability-weighting)  composes G1, G2, G3, G4, G5, D1
M4 (causal-aipw-tmle)  composes G1, G2, G3, G4, G5, D1
M5 (causal-forest-cate)  composes G1, G2, G3, G4, G5, D1
```

**Acyclic verification:** All edges flow from method → methodology / dataset → root (G1). No method composes another method. No methodology composes a method. G2 composes G1; G3 composes G1; G4 composes G1, G3; G5 composes G1, G2 — these form a strict partial order rooted at G1. Cycle-free by construction.

**Composition completeness check** (per spec requirement: every method composes the relevant methodology skills):
- G2 (estimand definition): composed by all 5 methods ✓
- G3 (positivity): composed by M2, M3, M4, M5 (NOT M1 — regression adjustment doesn't model the treatment, so propensity-based positivity is less directly applicable; M1's analogue of positivity is the regression-context covariate-overlap diagnostic in G4). Acceptable.
- G4 (balance): composed by M1, M2, M3, M4 (NOT M5 because causal forest is balance-by-randomization-via-honest-splits in spirit, though balance still useful diagnostically). M1 invokes G4 in **regression-context mode** (covariate overlap + Cook's D + residual-treatment gap); M2/M3/M4 invoke G4 in **propensity-context mode** (pre/post SMD + Love plot). Both branches live in the single G4 skill per the §2.3 Path A resolution.
- G1 (DAG): composed by all methods ✓
- G5 (sensitivity): composed by all methods ✓

### 4.2 Agent-stage attachment

Per skill, the receiving agents (after V2.0.1 stage-mapping):

| Skill | PF | DE | Analyst | Critic | Outline | Writer | tier (post-3b.12) |
|---|---|---|---|---|---|---|---|
| G1 dag-identification | ✓ | · | ✓ | ✓ | · | · | recommended |
| G2 estimand-definition | ✓ | · | ✓ | ✓ | · | ✓ | **mandatory** |
| G3 positivity-diagnostics | · | · | ✓ | ✓ | · | · | **mandatory** |
| G4 balance-diagnostics | · | · | ✓ | ✓ | · | ✓ | recommended |
| G5 sensitivity | · | · | ✓ | ✓ | · | ✓ | **mandatory** |
| D1 hsls09-causal-conventions | ✓ | ✓ | ✓ | ✓ | · | ✓ | **mandatory** |
| M1-M5 (5 method skills) | · | · | ✓ | · | · | · | **mandatory** (post-3b.8) |
| causal-data-engineer-contract (3b.12) | · | ✓ | ✓ | · | · | · | **mandatory** (post-3b.12) |

Stage strings used (matching V2.0.1 `STAGE_BY_AGENT`): `ProblemFormulator`, `DataEngineer`, `Analyst`, `Critic`, `Writer`. (`OutlineAgent` not needed — outline structure is task-type-agnostic; existing `paper-narrative-outline` skill suffices.)

**Mandatory inventory at Analyst stage for `task_type=causal_soo` (post-3b.12): 10 skills** — G2, G3, G5, D1 (the 3b.1 mandatory four), M1, M2, M3, M4, M5 (3b.8 promotions), and `causal-data-engineer-contract` (3b.12). G1 and G4 remain `recommended`. **Mandatory inventory at DataEngineer stage for `task_type=causal_soo` (post-3b.12): 1 skill** — `causal-data-engineer-contract`. See the Phase 3b.8 amendment and Phase 3b.12 amendment near the end of this document for rationale.

**No new orchestrator stages required.** All V3.0 skills attach to existing agent stages from V2.0.1.

### 4.3 HSLS dataset coupling

| Skill | Couples to HSLS via D1? |
|---|---|
| G1 dag-identification | NO (cross-dataset; HSLS specifics flow in via D1) |
| G2 estimand-definition | NO (cross-dataset) |
| G3 positivity-diagnostics | NO (cross-dataset) |
| G4 balance-diagnostics | NO (cross-dataset) |
| G5 sensitivity | NO (cross-dataset) |
| D1 hsls09-causal-conventions | YES (the dataset skill itself) |
| M1-M5 | YES — each composes D1 because clustered SEs and pre/post temporal table are HSLS-specific in V3.0 |

**Portability assessment for V4 (other datasets):** Methods M1-M5 currently couple to HSLS via D1. To port to e.g. NLSY or ECLS, V4 would author parallel dataset skills (`nlsy-causal-conventions`, `ecls-causal-conventions`) and modify M1-M5 to compose `<dataset>-causal-conventions` selected by the registry's dataset filter. Architecturally clean. No changes to G1-G5.

### 4.4 LSAR rubric alignment (out of scope for V3.0 implementation)

The current LSAR reviewer (configured via `review_gate.lsar_project_path`) is trained on prediction-task rubric. The Pass 1 failures it is **unlikely to catch** without rubric extensions:

| Failure | Why current LSAR misses it |
|---|---|
| IDF-01 unmeasured confounding | Rubric scores predictive performance; doesn't ask "could a confounder explain this?" |
| IDF-02 post-treatment conditioning | Predictive rubric encourages many features; doesn't penalize temporal violations |
| IDF-03 collider-induced selection | Rubric doesn't model the data-generating process |
| ESC-01 PSM ATT-as-ATE | Rubric doesn't distinguish estimands |
| ESC-03 causal-forest CATE-mean-as-ATE | Same |
| INF-03 TMLE/AIPW IF variance | Predictive rubric uses bootstrap CI; doesn't recognize IF variance |
| INT-01 causal language slippage | Rubric encourages strong claims |
| INT-04 CATE subgroup fishing | Rubric rewards "rich subgroup analysis" |
| INT-05 sensitivity overclaiming | Rubric has no sensitivity-analysis criterion |

**Out of scope for V3.0:** rubric extensions for the LSAR reviewer to catch the above. The Critic agent's checklist (via the V2.0 `prediction-critic-checklist` skill, which would be paralleled by a new `causal-critic-checklist` in a future Phase 3b sub-batch) is the V3.0 internal defense; LSAR rubric expansion is V3.x or V4.

**Tracked deferral:** create `audit/v3_lsar_rubric_extensions.md` in Phase 3b (or skip until LSAR shows real misses against V3.0 outputs).

### 4.5 Empty-match resolution

V2.0.1 confirmed 0 empty matches for `task_type=prediction`. There is currently no `task_type=causal_*` slot, so the V2.0.1 inventory does not need to list one — but as soon as V3.0 lands, the `task_type=causal_soo` slot (proposed slug below) becomes a new dimension that must be populated.

**Proposed `task_type` slug: `causal_soo`** (selection-on-observables) — single token, terse, explicit about the V3.0 identification strategy. Phase 3c will add quasi-experimental designs (IV, RD, DiD, ITS, synthetic control), all of which are also observational; a broader slug naming the data regime (e.g., one based on the word "observational") would semantically scope to "all observational" while in practice meaning "selection on observables only," which becomes a misnomer when 3c lands. `causal_soo` leaves clean slug space for `causal_iv`, `causal_did`, `causal_rd`, `causal_its`, `causal_synth` without renaming. Follows existing `prediction` style (single token, lowercase, underscored).

**Post-V3.0 expected match count per agent (post-3b.12; 12 new skills attached):**
- ProblemFormulator: G1, G2, D1 → 3 skills (matches PF's V2.0.1 count of 6 for prediction, similar order)
- DataEngineer: D1, `causal-data-engineer-contract` → 2 skills (post-3b.12; the second skill addresses the original concern flagged in the next-line "monitor and extend with a treatment-derivation skill if regression cycles surface gaps" bullet — see Phase 3b.12 amendment near the end of this document)
- Analyst: G1, G2, G3, G4, G5, D1, M1-M5, `causal-data-engineer-contract` → 12 skills (vs prediction's 18; reasonable)
- Critic: G1, G2, G3, G4, G5, D1 + a future `causal-critic-checklist` → 7 skills (vs prediction's 10)
- Writer: G2, G4, G5, D1 + existing writing skills (acm-template, bibtex, style, etc., which are dataset-agnostic) → 8-10 skills (vs prediction's 10; comparable)

**Empty-match risk for V3.0:** none, assuming all 12 skills are authored with the stage attachments in §4.2. The DataEngineer's two-skill match is the lowest count and remains monitored — if regression cycles on causal tasks surface further DE-specific failures (e.g., the F-3b11-DE-CONTINUOUS-AS-CATEGORICAL revision-cycle pattern), additional dataset-layer or methodology-layer skills may follow.

---

## Appendix — Open questions and deferred decisions

1. **`causal-treatment-derivation` skill (DataEngineer-side, deferred to Phase 3b):** the current V3.0 design relies on the PF declaring the treatment in `research_spec.treatment` and the DE simply selecting it. If regression cycles on causal tasks reveal failures around treatment derivation (e.g., binarizing continuous scales without a defined contrast, deriving treatment indicators from MAR variables), add a new dataset-layer skill `hsls09-causal-treatment-derivation` or methodology skill `causal-treatment-derivation`. Track in V3.0 retrospective.

2. **Causal-critic-checklist (Critic-side, deferred to Phase 3b):** the existing `prediction-critic-checklist` skill is task-type-coupled to prediction. V3.0 needs a parallel `causal-critic-checklist` enumerating per-section review items (DAG completeness, estimand declared, positivity satisfied, balance achieved, sensitivity reported, no causal language without hedges). Out of scope for this audit (audit's focus was method/methodology/dataset layers); plan to add in Phase 3b after method skills are authored and at least one causal regression run produces a Critic output to ground the checklist.

3. **Causal narrative outline emphasis triggers:** the V2.0 `paper-narrative-outline` skill has emphasis triggers for prediction outputs (e.g., "if subgroup_gap_large → promote subgroup section"). For causal papers, analogous triggers: "if cate_variance_ratio > 2 → expand heterogeneity section"; "if evalue < 1.5 → expand sensitivity section to a full subsection". Consider extending in Phase 3b once the first causal paper draft exists.

4. **Survey-weight handling beyond V3.0 default:** V3.0 explicitly does NOT apply HSLS survey weights; estimand is sample-marginal. If V3.x wants population-marginal estimands, add `hsls09-survey-weights-causal-application` skill with `WeightedTMLE` recipe (open research area; Targeted Learning literature has only partial guidance on combining survey weights with TMLE).

5. **Multi-treatment / multi-arm extensions:** V3.0 covers binary treatment by default. ESC-06 (multi-valued treatment estimand) is enumerated but the only defense is G2's "declare contrast" rule. Continuous-treatment causal forest (`econml.dml.LinearDML`) and dose-response curves are deferred to V3.x.

6. **Mediation analysis:** explicitly out of scope per the V3.0 spec. If users request "the effect of X on Y mediated by Z" research questions, the PF must reject and recommend a separate phase. Add a Critic check in Phase 3b to detect mediation-shaped questions and abort early.

7. **`zEpid` library risk:** single-maintainer dependency for TMLE. If the package becomes unmaintained, M4's TMLE branch falls back to custom implementation (~150 LOC). Track upstream activity quarterly; if the latest tag is > 12 months old at V4 planning, plan to vendor the relevant TMLE class.

8. **`dowhy` API churn:** has historically had unstable API. Pin `dowhy>=0.11,<0.13`. If `dowhy` becomes unmaintained, G1's `CausalModel` skeleton can be replaced with a thin custom wrapper around NetworkX + back-door criterion logic (~80 LOC).

9. **Cluster-bootstrap for causal forest:** §3.11 notes that `econml`'s native variance does not cluster. The proposed cluster-bootstrap is computationally expensive (1000 iterations × full forest fit). For n=20K HSLS with `n_estimators=500`, this is on the order of hours. Consider sample-down-and-bootstrap or sub-sampling strategies if regression times become unacceptable.

10. **Ordering of authoring in Phase 3b:** recommended dependency-respecting order:
    - First: G1 (dag-identification) — root, blocks every method
    - Next: G2, D1 — both depend only on G1; both heavily composed
    - Then: G3, G4, G5 — methodology supports
    - Last: M1-M5 — each method skill is straightforward once methodology + D1 exist
    - Finally: Phase 3c sub-batches: `causal-critic-checklist`, optional `causal-treatment-derivation`, narrative emphasis triggers

---

## Phase 3b.8 amendment (post-3b.7 formatter discovery)

**Change:** M1–M5 promoted from `rule_severity: recommended` (default per the
original 3b.2 spec) to `rule_severity: mandatory`.

**Rationale.** The original 3b.2 spec tagged M-skills `recommended` on the rationale
that method skills are method-specific while methodology skills (G2/G3/G5) and
dataset skills (D1) are universal. 3b.7's smoke test surfaced that the formatter's
`max_chars` cap silently dropped the recommended-tier M-skill bodies under budget
pressure (F-3b7-FORMATTER-TRUNCATES-METHOD-SKILLS): with 7 mandatory skills already
exceeding the 12,000-char budget, every recommended skill — including all five
M-skills — was truncated out before reaching the LLM. The 3b.6 amendments to D1, M1,
M2, M3, M4 (encoded-column lookup), to M4 (cluster-aware IF + sanity check), and to
the subgroup-fairness causal branch were silently neutralized for that reason.

Phase 3b.8 fixes this in two orthogonal layers:

1. **Per-tier cap in the formatter** (`composer.format_skills_for_prompt`,
   `mandatory_chars_unlimited=True` default): mandatory tier renders in full
   regardless of budget; recommended/reference tiers compete for `max_chars` of
   their own. Mandatory cost no longer counts against the recommended budget.
2. **Promote M1–M5 to mandatory.** With per-tier protection in place, the M-skill
   bodies always render at the Analyst stage when matched.

**Why mandatory was the right tier change** (rather than e.g. raising `max_chars`
unconditionally): the 3b.2 audit defined `mandatory` as "violation produces
methodologically-invalid output." M-skill bodies meet this criterion — picking the
wrong matching mechanic, the wrong IF variance estimator, or the wrong CATE
estimator produces an estimate that looks valid but is methodologically wrong.
The original "method-specific vs. universal" distinction is preserved in the
per-skill `applicable_task_types: [causal_soo]` field — M-skills only get matched
in causal contexts. The mandatory tag now means **"render in full when matched,"**
not "always relevant."

**G1 and G4 NOT promoted.** Their original 3b.1 rationale stands: G1 (DAG) is a
conceptual scaffold whose violation produces a worse paper but not silent
corruption; G4 (balance) is mode-discriminator-dependent (regression vs. propensity
context) and the violations it prevents surface in the rendered paper. Promoting
G1/G4 is a separate spec decision and is deliberately out of scope for 3b.8.

**Inventory after 3b.8.** Mandatory at Analyst stage for `task_type=causal_soo`:
G2, G3, G5, D1, M1, M2, M3, M4, M5 (9 skills). G1 and G4: recommended (subject to
the per-tier budget when not above the budget floor).

**Effect on prediction tasks.** None — M-skills declare
`applicable_task_types: [causal_soo]`, so the matcher does not return them for
prediction tasks. The mandatory promotion only affects causal_soo runs.

---

## Phase 3b.12 amendment (post-3b.11 DE causal-contract gap discovery)

**Change:** new methodology skill `causal-data-engineer-contract`
authored at `rule_severity: mandatory`, attached to the DataEngineer
and Analyst stages. Total V3.0 causal skills: **11 → 12**. Mandatory
inventory at Analyst stage for `task_type=causal_soo`: **9 → 10**.
New: 1 mandatory at DataEngineer stage (was 0).

**Rationale.** The 3b.11 LSAR review scored 5.0 (Borderline; below
the 5.5 gate) — a 0.9-point regression vs the 3b.5 baseline of 5.9.
Post-run analysis (`runs/v3_0_smoketest_mtheff_college_20260501_3b11/REPORT.md`)
attributed −6 of the −8 dimension points to a single new structural
defect: **F-3b11-DE-MISSING-TREATMENT-COLUMN**. The DataEngineer's
analytic-CSV carve-out contract was prediction-task-shaped — it
extracted only `adjustment_set + outcome` from the research_spec
into `train_X.csv`, dropping the treatment column. The PF correctly
omits treatment from `adjustment_set` per causal DAG identification
(treatment is the exposure, not a covariate to adjust on); the DE
interpreted that exclusion as "drop the column entirely." The
Analyst, applying D1's `resolve_encoded_columns` rule, correctly
diagnosed the missing column and substituted the closest available
proxy (`X1MTHID` for the locked spec's `X1MTHEFF`). All five method
estimates were computed against the proxy. LSAR cited this as
fatal: *"substitution of the wrong variable (math identity vs.
self-efficacy) invalidates the core estimand."*

The defect was independent of every 3b.6/3b.10/3b.10.5 change — it
had been latent since `causal_soo` was added in 3b.4 and was
masked by coincidental DE plumbing in 3b.5 that happened to retain
the treatment column. The cleaner 3b.10.5 prompt regressed off that
coincidence. **Section 4.5's own prediction** ("DataEngineer's
single-skill match is the lowest count and should be monitored —
if regression cycles surface DE-specific failures, add a skill in
a sub-batch") was the structural anticipation of exactly this gap.

**The fix lands in two coordinated layers** (per the
project's hardening pattern: positive guidance + runtime guardrail):

1. **Skill body — positive guidance for the DE LLM.** The new skill
   (`skills/methodology/causal-data-engineer-contract/SKILL.md`)
   codifies the carve-out rule: in `causal_soo`, the analytic CSV
   must contain `treatment + adjustment_set` (with operationalization
   handling for forms like `median_split_binary`); the outcome lives
   in `train_y.csv` per existing convention. The body includes a
   prescriptive Python recipe (`causal_soo_carve_out` function) that
   the DE LLM should follow when emitting code. `references_skills`
   composes with G2 (estimand declaration depends on correct
   treatment identification) and D1 (`resolve_encoded_columns` is
   downstream of the carve-out).

2. **Orchestrator guardrail — runtime fail-fast.**
   `src/causal_data_contract.py` exposes `CausalDataContractError`
   and `assert_causal_soo_data_contract(train_X_path, research_spec)`.
   The orchestrator's `_run_engineering` invokes the assertion
   between the DE stage's `validation_passed` check and the
   transition to `ANALYZING`. A contract violation halts the
   pipeline with a clear error message that names the expected
   treatment column and cites the new skill, before the Analyst is
   ever invoked. No-op for non-causal task types — prediction's
   DE contract is unchanged.

**Why both stages.** The DE produces the contract-compliant
artifact; the Analyst reads it. One-sided attachment would either
let the DE drop the column or let the Analyst silently substitute
again. Per `applicable_stages: [DataEngineer, Analyst]`, both
stages now receive the skill body in their rendered prompts.

**Why mandatory.** Per the original 3b.2 mandatory criterion
("violation produces methodologically-invalid output"). The whole
3b.11 LSAR collapse traces to this exact violation. Recommended-
tier rendering would let the formatter drop the body under budget
pressure (same bug class as F-3b7-FORMATTER-TRUNCATES-METHOD-SKILLS);
mandatory + the 3b.8 per-tier formatter protection ensure the body
always renders when matched.

**Why mandatory at DataEngineer specifically.** The original 3b.2
spec did not include a DE-specific mandatory skill because
`hsls09-causal-conventions` (D1) attaches at DE and was assumed
sufficient. 3b.11 surfaced that D1's body covers HSLS-specific
conventions but does NOT codify the **carve-out shape** —
treatment-column-presence is a cross-dataset rule, methodology-
layer in nature. The new skill is therefore methodology-layer (not
dataset-layer), distinguishing the DE carve-out contract from D1's
dataset-specific rules.

**The DataEngineer V1 prompt was already wired for skill injection.**
`agent_prompts/data_engineer.yaml` already carried the
`{{SKILLS}}` placeholder (added in V2.0.1's slim DE pass — V2.0
ships partial; DE was the one fully-slimmed agent). Phase 3b.12
exercises that wire-up for the first time under causal_soo by
attaching a DE-applicable skill. The hand-off's §12.3.3 anticipated
the placeholder might be missing; verification confirmed it was
present, so no DE prompt edit was required. The new test
`tests/test_rendered_prompt_contains_all_mskills.py::TestRenderedPromptIncludes3b12DEContract::test_de_prompt_has_skills_placeholder`
locks the placeholder in place against future regressions.

**Effect on prediction tasks.** None — the new skill declares
`applicable_task_types: [causal_soo]`, and the Orchestrator
guardrail no-ops for non-causal task types. Prediction-task DE runs
are byte-identical to the pre-3b.12 path.

**Inventory after 3b.12.** Mandatory at Analyst stage for
`task_type=causal_soo`: G2, G3, G5, D1, M1, M2, M3, M4, M5,
`causal-data-engineer-contract` (10 skills). Mandatory at
DataEngineer stage: `causal-data-engineer-contract` (1 skill). G1
and G4 remain `recommended`. Total V3.0 causal skills: 12.

**3b.13 readiness.** With 3b.12 landed, 3b.13 re-runs the smoke
test on the same locked spec / same provider configuration as 3b.11
(gpt-5.4 Analyst+Writer; DeepSeek elsewhere). The differential
question is narrow: did the DE-contract fix resolve
F-3b11-DE-MISSING-TREATMENT-COLUMN and lift the LSAR score back
above 5.5?

---

## Phase 3b.14 amendment (post-3b.13 DoWhy refuter-graph-format gap)

**Change:** existing G5 SKILL.md (`causal-sensitivity-unmeasured-
confounding`) extended with a new "DoWhy refuter invocation (mandatory
for causal_soo)" section. No new skill; no frontmatter change; no
attachment-table change. Total V3.0 causal skill count: **12
unchanged**. Mandatory inventory at Analyst (10), DataEngineer (1):
unchanged.

**Rationale.** Phase 3b.13 (LSAR 6.0; first passing causal_soo run)
named one specific gap by reference in the LSAR Methodological Rigor
justification: *"the unresolved M4 AIPW SE anomaly … and failed
sensitivity package are significant gaps."* The LSAR review's "Areas
for Improvement" section is more direct: *"DoWhy refuters failed:
Sensitivity analysis is incomplete because the DAG-format error
prevented the planned sensitivity package from running. This is a gap
that should be addressed in revision."* — and the §Specific
Recommendations: *"DoWhy refuters: Was the DAG-format error debugged?
Would re-running with a corrected DAG format complete the sensitivity
analysis? This seems like a tractable fix."*

3b.13's `output/results.json.sensitivity.refuter_results: []` (empty
array). The Analyst attempted refuters; the call sequence threw on
`identify_effect()` with the error *"Incorrect format: Please provide
graph as a networkx DiGraph, GCM model, or as a string in either GML
or DOT format."* The error message is misleading. The DOT format was
syntactically valid; the failure was semantic — the Analyst built a
DOT graph with node aliases `T` and `Y` (with labels containing the
variable names) and then called
`CausalModel(treatment="X1MTHEFF_binary", graph=dot_string)`. DoWhy
looked up node `X1MTHEFF_binary` in the graph, didn't find it
(because the node ID is `T`), and emitted a confusingly-worded
format-detection error.

**The fix is structural in G5 SKILL.md** (no orchestrator changes
required). The new section adds:

1. **Mandatory rule: DAG node names MUST match column names.** No
   aliasing. The exact wording of the 3b.13 failure (`T` and `Y`
   nodes vs `X1MTHEFF_binary` / `X4EVRATNDCLG` columns) is cited.
2. **Graph construction recipe (`build_dowhy_graph`).** DOT-format
   string with column-name node IDs and explicit confounder edges
   from each adjustment-set variable to both treatment and outcome.
3. **Four-step invocation sequence.** `build_dowhy_graph` → 
   `CausalModel(graph=dot_string)` → `identify_effect(proceed_when_
   unidentifiable=True)` → `estimate_effect(method_name="backdoor.
   linear_regression")` → per-refuter `refute_estimate`. Each step is
   named with the reason it must come before the next.
4. **Output schema.** `sensitivity.dowhy_refuters` is now a map
   keyed by refuter name (`random_common_cause` / `placebo_treatment_
   refuter`), each entry carrying `new_effect`, `p_value`, `status`
   (`"ran"` / `"failed"`), and `error` (exception string or null).
   This is a deliberate shape change from the pre-3b.14
   `refuter_results: [{...}]` array — the map form makes the per-
   refuter `status` visible to the Writer for honest §Limitations
   acknowledgment. The Validation Criteria section retains backward-
   compat tolerance for the array form so 3b.5 / 3b.7 / 3b.11
   artifacts still parse.
5. **Exception handling (mandatory).** Each refuter is wrapped in
   `try/except`. A single refuter failure must not block the other;
   failures are recorded with `status: "failed"` instead of being
   omitted. This converts silent degradation (3b.13's `refuter_
   results: []`) into explicit acknowledgment.
6. **Writer interpretation guidance.** §Sensitivity must report
   `status: "ran"` results; §Limitations must acknowledge `status:
   "failed"` cases. The "result is robust to unmeasured confounding"
   forbidden-phrase rule (from the original 3b.1 G5 audit) is
   unchanged.

**Why structural addition (not rewrite).** The existing G5 content
(E-value, Rosenbaum bounds, declarative refuter list, validation
criteria, source provenance) is correct and lands in the rendered
prompt as designed in 3b.6 / 3b.8. The 3b.13 evidence pinpointed a
specific subgap — the DoWhy refuter call sequence — that the
declarative rules under-specified. The amendment adds the prescriptive
form alongside the declarative form rather than replacing either.
This mirrors the 3b.12 pattern that produced clean attribution: a
narrowly-scoped addition with its own rendered-prompt marker tests.

**Why no orchestrator-side guardrail (unlike 3b.12).** F-3b11-DE-
MISSING-TREATMENT-COLUMN was a silent failure (Analyst substituted a
proxy variable downstream); a runtime assertion was the right safety
net. F-3b13-DOWHY-REFUTERS-GRAPH-FORMAT is a noisy failure — the
Analyst's wrapper already captures the exception into `warnings`, and
the resulting `refuter_results: []` is itself a clear "this didn't
run" signal in the artifact. The Critic in 3b.13 cycle 1 flagged it
as `[critical] sensitivity_analysis_incomplete`. No additional
runtime check is needed; positive guidance in the rendered prompt is
the proportionate fix.

**3b.15 readiness.** With 3b.14 landed, 3b.15 re-runs the same
locked-spec smoke test on the unchanged 3b.13 provider configuration.
The single variable changed since 3b.13 is the G5 amendment. The
differential question: does the prescriptive DoWhy invocation produce
non-empty `dowhy_refuters` output and lift LSAR's Methodological
Rigor / Empirical Support beyond 3b.13's 6 / 6?

---

## Phase 3b.16 amendment (post-3b.15 pygraphviz-dependency refinement)

**Change:** G5 SKILL.md's "DoWhy refuter invocation (mandatory for
causal_soo)" section — authored in 3b.14 — was refined to use a
`networkx.DiGraph` object as the `graph=` argument to
`CausalModel(...)`, replacing the 3b.14 prescription of a DOT-format
string. The `build_dowhy_graph(...)` recipe's return type changes from
`str` to `nx.DiGraph`; the function body becomes the NetworkX edge-
addition idiom (`g.add_edge(treatment, outcome)` plus confounder
edges); the four-step invocation sequence is otherwise unchanged. No
new skill; no frontmatter change; no attachment-table change. Total
V3.0 causal skill count: **12 unchanged**. Mandatory inventory at
Analyst (10), DataEngineer (1): unchanged.

**Rationale.** Phase 3b.15 (LSAR 7.0 / Accept; first non-Borderline run)
produced the
`v3_0_smoketest_mtheff_college_20260502_3b15` artifact set. The
Analyst followed the 3b.14 amendment verbatim — including the DOT-
string graph format — with node names correctly matching the
`treatment`/`outcome` arguments. Both DoWhy refuter calls
nevertheless raised:

> `ValueError: "Incorrect format: Please provide graph as a networkx
> DiGraph, GCM model, or as a string or text file in dot, gml
> format."`

The DOT string itself was syntactically valid. Root-cause analysis
identified that DoWhy 0.12's DOT-string parser requires `pygraphviz`
at runtime, which in turn requires the system-level `graphviz` C
library. The project's deployment venv does not ship pygraphviz
(empirically verified: `import pygraphviz` raises `ModuleNotFoundError`
in the 3b.15 venv). When pygraphviz is unavailable, DoWhy's graph-
detection routine emits the misleading "Incorrect format" message
even on otherwise-valid DOT input. The 3b.14 amendment was
prescriptively correct against DoWhy's public API documentation; it
collided with a runtime requirement the documentation does not
foreground.

The 3b.15 LSAR review reframed the resulting failure from "critical
tractable gap" (3b.13's framing) to **MINOR + transparency strength**
(3b.15's framing) — the 3b.14 amendment's `try/except` pattern
recorded both refuters as `{"status": "failed", "error": "..."}`
rather than silently dropping them, and LSAR credited that
transparency in Methodological Rigor / Empirical Support
justifications. The Q2.5 verdict for 3b.15 was RESOLVED-WITH-CAVEAT:
amendment structurally landed (recipe copied verbatim, schema
populated, exception handling fired, LSAR rewarded transparency)
without actually running refuters. The 3b.16 refinement closes the
remaining technical gap so refuters actually execute.

**The fix is well-localized:**

1. **`build_dowhy_graph` return type.** `str` (DOT) → `nx.DiGraph`.
   Function body changes from string concatenation to NetworkX
   edge-addition calls. Node-naming semantics are identical (column
   names as node names).
2. **`CausalModel(graph=...)` call site.** `graph=graph_dot` →
   `graph=graph_nx`; comment updated to cite the pygraphviz-
   dependency rationale.
3. **Failure-mode subsection.** Restructured to cover both
   F-3b13-DOWHY-REFUTERS-GRAPH-FORMAT (node-name mismatch — 3b.14's
   fix, preserved) and F-3b15-DOWHY-PYGRAPHVIZ-DEPENDENCY-GAP (DOT-
   parser dependency — 3b.16's fix). Both rules together close the
   refuter-execution failure surface.

**What's preserved unchanged from 3b.14:**
- The node-naming rule ("DAG node names MUST match the strings passed
  as `treatment`, `outcome`, and adjustment-set elements"). This
  addresses F-3b13, not F-3b15; both rules must hold.
- The full four-step invocation sequence (`build → CausalModel →
  identify_effect → estimate_effect → refute_estimate`).
- The `try/except` per-refuter exception-handling pattern. LSAR's
  3b.15 review credited the transparent failure-reporting as a
  methodological strength; the pattern stays as a safety net against
  future upstream-library quirks.
- The `sensitivity.dowhy_refuters` output schema (map keyed by
  refuter name with `new_effect` / `p_value` / `status` / `error`).
- The Writer interpretation guidance (`§Sensitivity` reports
  `status: "ran"`; `§Limitations` acknowledges `status: "failed"`).

**Empirical verification before authoring** (Sub-wave 0 of 3b.16):
the pygraphviz-dependency diagnosis was reproduced offline against
DoWhy 0.12 using a minimal causal-inference smoke test with binary
treatment and outcome plus two confounders. The DOT-string path
raises `ValueError` with the same error message LSAR cited; the
NetworkX-DiGraph path completes `identify_effect` →
`estimate_effect` → `refute_estimate` end-to-end and returns a
populated `refutation_result` with a `p_value`. Diagnosis correct;
fix direction correct.

**Why no new orchestrator-side guardrail (unlike 3b.12).** Same
rationale as 3b.14: DoWhy refuter failures are noisy (they raise
`ValueError`, the wrapper records `status: "failed"`, the Writer
acknowledges in `§Limitations`), not silent. The existing exception-
handling pattern is the right safety net.

**3b.17 readiness.** With 3b.16 landed, 3b.17 re-runs the same
locked-spec smoke test on the unchanged 3b.15 provider configuration.
The single variable changed since 3b.15 is the NetworkX-DiGraph
refinement. The differential question: does the refuter invocation
actually run (Q2.5.b: SUCCEEDED rather than 3b.15's FAILED-NON-FATAL)
and produce populated `dowhy_refuters` output that lifts LSAR's
Methodological Rigor from 3b.15's 6 to 7?

After 3b.17, the project closes the causal-skill cleanup arc
(3b.12 → 3b.13 → 3b.14 → 3b.15 → 3b.16 → 3b.17) and pivots to the
V2.1 slim migration scope doc as 3b.18 — the natural strategic
break-point given LSAR has already crossed the Accept threshold.

*(3b.17 outcome update: the G5/DoWhy arc closed empirically — refuters
ran cleanly, refuter values match M1 ATE to 4 decimals, LSAR reframed
DoWhy from critical-tractable to absent. But LSAR scored 6.0
Borderline (vs 3b.15's 7.0 Accept) due to an unrelated regression:
F-3b15-DE-CONTINUOUS-AS-CATEGORICAL recurred at cycle 0 and did NOT
recover at cycle 1 (3 mis-encoded continuous variables → 115-column
matrix → 21% extreme-tail-fraction positivity violation flagged by
LSAR as "fundamental identification failure"). The cleanup arc
extends one more pair: 3b.18 = D1 encoding amendment; 3b.19 =
verification re-run; 3b.20 = V2.1 pivot.)*

---

## Phase 3b.18 amendment (post-3b.17 DE continuous-as-categorical recurrence)

**Change:** D1 SKILL.md (`hsls09-causal-conventions`) extended with an
"Encoding-type discipline (mandatory for DataEngineer)" section. No
new skill; no frontmatter change; no attachment-table change. Total
V3.0 causal skill count: **12 unchanged**. Mandatory inventory at
Analyst (10), DataEngineer (1): unchanged.

**Rationale.** Phase 3b.17 (LSAR 6.0 / Borderline; G5/DoWhy arc
empirically closed) surfaced F-3b15-DE-CONTINUOUS-AS-CATEGORICAL as
the new rate-limiting failure mode. The 3b.6 D1 work introduced the
Analyst-side `resolve_encoded_columns` rule but did not codify a
DataEngineer-side rule for which variables to one-hot encode. Without
that rule, the DataEngineer defaults to `pd.get_dummies(...)` on every
adjustment-set variable that is not already binary — including
continuous psychometric scales whose finite factor-score grid the DE
mis-interprets as categorical.

Cross-run evidence chain (see `runs/.../02_data_engineer/data_report_cycle0.json`
for each run):

| Run | adjustment_set vars | Cycle 0 cols | Cycle 1 cols | Mis-encoded continuous scales | Outcome |
|---|---:|---:|---:|---|---|
| 3b.13 | 11 | 56 | 56 | `X1SCIID` (16 dummies) | Clean — fewer continuous vars in adjustment_set masked the issue |
| 3b.15 | 10 | 109 | 11 | `X1MTHID` + `X1MTHUTI` (~70+16 dummies) | Cycle-1 over-corrected; LSAR 7.0 Accept |
| 3b.17 | 13 | 116 | 115 | `X1MTHID` + `X1MTHUTI` + `X1STUEDEXPCT` (17+56+11 dummies) | Cycle-1 did NOT recover; LSAR 6.0 Borderline; 21% extreme-tail positivity violation |

**3b.13's apparent cleanness was stochastic luck on PF's adjustment_set
composition, NOT structural protection.** The DataEngineer behavior was
identical in 3b.13: type-blind one-hot of every non-binary adjustment-
set variable. The number of cycle-0 columns was lower simply because
fewer continuous variables happened to be in PF's chosen adjustment_set,
and the cycle-1 recovery was itself stochastic (recovered in 3b.13 and
3b.15, did not recover in 3b.17). The DE's encoding rule needs
deterministic structural protection rather than reliance on cycle-1
Critic feedback.

**Pre-amendment investigation (Sub-wave 0 of 3b.18) confirmed:**

1. The 3b.17 `train_X.csv` has 14 raw predictors expanding to 115
   encoded columns. The expansion is concentrated in three variables:
   `X1MTHUTI` (56 dummies), `X1MTHID` (17), `X1STUEDEXPCT` (11) — the
   first two are tagged `type=continuous` in the registry; the third
   is tagged `categorical` and is encoded correctly.
2. The variable registry has reliable `type` tagging. Spot-checked
   variables: `X1MTHEFF`, `X1MTHID`, `X1MTHUTI`, `X1MTHINT`,
   `X1SCHOOLBEL`, `X1SES`, `X1TXMTSCOR` all `type=continuous`;
   `X1RACE`, `X1PAREDU`, `X1STUEDEXPCT`, `X1LOCALE` all
   `type=categorical`; `X1SEX`, `X1CONTROL` both `type=binary`. The
   amendment's dispatch logic is well-founded.
3. The current D1 has the Analyst-side `resolve_encoded_columns`
   rule (3b.6) but no DE-side encoding rule. The gap is genuine.
4. The DataEngineer V1 prompt (`agent_prompts/data_engineer.yaml`)
   characterizes `train_X.csv` as "one-hot encoded" (line 25) and
   references the registry's `type` field only as informational
   (line 120). The prompt provides no type-aware dispatch guidance;
   the DE has no incentive to NOT one-hot encode continuous variables.

**The fix is well-localized to D1:**

1. **Registry-type dispatch rule.** `type=continuous` → pass through.
   `type=categorical` or `type=binary` → one-hot. `type=ordinal` →
   per the registry tag (the HSLS:09 registry uses `categorical` for
   ordinal-as-categorical variables like `X1PAREDU`, so the rule
   reduces to "follow the registry"). Unknown type → pass through
   with warning.
2. **Prescriptive Python recipe** (`encode_for_causal_soo`). Inline
   in the SKILL.md body so the DE LLM can copy it verbatim. Takes a
   `variable_registry` parameter so the rule is testable in isolation.
3. **Concrete examples table** listing continuous variables observed
   in prior adjustment-sets (`X1MTHEFF`, `X1MTHID`, `X1MTHINT`,
   `X1MTHUTI`, `X1SCIID`, `X1SCHOOLBEL`, `X1SES`, `X1TXMTSCOR`) so the
   LLM has anchor points for the rule.
4. **Failure-mode citation** with the 3b.13 / 3b.15 / 3b.17 column-
   count evidence and the mechanism (finite factor-score grid →
   categorical mis-interpretation → propensity overfit → positivity
   violation).
5. **Cross-reference to existing D1 rules** clarifying that 3b.6
   `resolve_encoded_columns` is downstream of this rule (Analyst-side
   read, not DE-side write) and that 3b.12 `causal-data-engineer-
   contract` is upstream (governs which columns to carve out, not how
   to encode them).

**What's preserved unchanged:**
- D1 frontmatter (layer, stages, severity, task_type, references).
- The 3b.6 Analyst-side `resolve_encoded_columns` rule. The two rules
  are complementary: 3b.6 reads whatever encoding the DE wrote; 3b.18
  governs what the DE writes.
- All other D1 sections (treatment-relevant variable inventory,
  temporal classification, attrition rules, clustered-SE handling,
  survey-weights handling, validation criteria, source provenance).

**Why no orchestrator-side guardrail (unlike 3b.12).** Same rationale
as 3b.14 and 3b.16: over-encoding is noisy (visible in cycle-0
data_report.json's `n_predictors_encoded`, in propensity diagnostics'
extreme-tail-fraction, and in Critic's positivity-violation flag), not
silent. The amendment + LLM compliance is the first line; a guardrail
becomes warranted only if 3b.19 shows the LLM still ignores the rule
under the prescriptive amendment.

**3b.19 readiness.** With 3b.18 landed, 3b.19 re-runs the same locked-
spec smoke test on the unchanged 3b.17 provider configuration. The
single variable changed since 3b.17 is the D1 encoding-type-discipline
amendment. The differential questions:

1. Does the DataEngineer cycle 0 produce ~14 raw → ~14 encoded columns
   (deterministic registry-driven dispatch) instead of 14 → 116?
2. Does the resulting propensity model fit cleanly with positivity
   diagnostics within bounds (extreme_tail_fraction < 0.10)?
3. Does LSAR's Methodological Rigor score recover from 3b.17's 4 to at
   least 6?

If 3b.19 returns Outcome 1 (clean closure), 3b.20 begins the V2.1 slim
migration scope doc per Path C.

---

**End of V3.0 Phase 3a audit (Phase 3b.8 + 3b.12 + 3b.14 + 3b.16 + 3b.18 amendments).**
