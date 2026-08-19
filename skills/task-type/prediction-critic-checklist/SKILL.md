---
name: prediction-critic-checklist
layer: task-type
description: Per-section Critic review checklist for prediction studies (Problem Formulation / Data Preparation / Analysis / Substantive Validity).
trigger_keywords:
  - critic
  - checklist
  - review
  - reviews
  - verdict
  - severity
applicable_task_types:
  - prediction
applicable_datasets: []
applicable_stages:
  - Critic
priority: 1
references_skills: []
resources: []
version: "1.1"
rule_severity: mandatory
---

# Prediction Critic Checklist

The Critic works through four review sections, each with checklist
items. Each item carries a severity (`critical | major | minor`) and a
target agent. Each failure becomes an issue entry in
`review_report.json` with the appropriate severity and a concrete
recommendation.

This skill is intentionally long; it is the authoritative checklist
the Critic agent walks through before issuing a verdict. Decision 2
from the Phase 0 audit kept it as one skill rather than splitting it
into per-section sub-skills; future phases may revisit.

## Verdict criteria (apply after walking the checklist)

| Verdict | Condition |
|---|---|
| ABORT | Any item with `severity == critical` is confirmed AND not fixable in the current pipeline run (e.g., unanswerable question, confirmed leakage, `analytic_n < 1000`). |
| REVISE | Any fixable critical issue OR `> 2` major issues OR `overall_quality_score < 7`, AND `revision_cycle < max_revision_cycles`. |
| PASS | No critical issues AND `major_issue_count ≤ 2` AND `overall_quality_score ≥ 7`. |

If max revision cycles have been exhausted and the verdict would be
REVISE, set the verdict to PASS — the orchestrator will mark the
paper UNVERIFIED. Never set ABORT merely because revision cycles are
exhausted.

Section scores are integers 1–10, weighted:
- Problem formulation: 0.25
- Data preparation: 0.25
- Analysis: 0.30
- Substantive validity: 0.20

`overall_quality_score` is the weighted mean rounded to the nearest
integer.

## Section 1 — Problem Formulation

Evaluated against `research_spec.json`, `literature_context.json`,
`data_report.json` (for feasibility).

| ID | Item | Severity | Check |
|---|---|---|---|
| `pf_01` | Research question is specific and answerable | critical | Names a specific outcome variable, target population, prediction approach. Reject vague "explore factors related to achievement". Must be answerable with supervised ML on the dataset. |
| `pf_02` | All predictors temporally precede the outcome | critical | For each predictor, predictor wave appears strictly before outcome wave in `temporal_order`. Same-wave or future-wave → critical. See `hsls09-temporal-ordering`. |
| `pf_03` | Predictor rationales are educationally grounded | major | Each `predictor_set[*].rationale` references a theoretical mechanism, prior empirical evidence, or established educational construct. "It is available in the dataset" → major. |
| `pf_04` | Novelty claim supported by `literature_context.novelty_evidence` | major | Specifically contrasts the study against ≥1 retrieved paper. Empty/null/generic → major. If S2 API failed (papers list empty), downgrade to minor. |
| `pf_05` | Target population is well-defined | minor | Non-empty string describing the analytic sample. Generic "students" → minor. |
| `pf_06` | Feasibility: `analytic_n ≥ 10,000` | critical | From `data_report.analytic_n`. < 10,000 → critical (unless structural-MNAR exception applies — see `hsls09-structural-mnar-outcomes`, where the floor is 1,000). |
| `pf_07` | `novelty_score_self_assessment ≥ 3` | major | < 3 → major. See `prediction-research-question-design`. |
| `pf_08` | (findings-memory only) Question novel relative to prior runs | minor | When findings memory is enabled and the outcome + similar predictor set was already studied, flag minor unless the framing differs. See `findings-memory-novelty-cross-run`. |
| `pf_09` | (findings-memory only) Contribution builds on prior runs | minor | Aspirational. Verifies `expected_contribution` references open questions or strong predictors from the memory. |

## Section 2 — Data Preparation

Evaluated against `data_report.json`.

| ID | Item | Severity | Check |
|---|---|---|---|
| `dp_01` | No data leakage (temporal or target) | critical | (a) No predictor wave at or after outcome wave. (b) Outcome variable does NOT appear in `train_X.csv` columns. (c) Scalers/imputers fit on training data only. |
| `dp_02` | Missing-data handling matches the protocol | major | For each variable in `data_report.missingness_summary`, `imputation_method` matches the tier in `missingness-tiered-protocol` for the recorded `pct_missing`. |
| `dp_03` | `analytic_n ≥ 10 × n_predictors_raw` (10p rule) | major | Use `n_predictors_raw`, not `n_predictors_encoded`. < 10 → major. |
| `dp_04` | Class balance reported; extreme imbalance addressed | major | Binary outcomes only. Majority/minority ratio > 9:1 with no mitigation in warnings → major. Continuous → skip. |
| `dp_05` | No constant (zero-variance) predictors remain | major | `data_report.validation_passed == true`. If false due to constants → critical. |
| `dp_06` | Train/test split stratified for classification | major | Binary/categorical outcomes: split is stratified (via `StratifiedGroupKFold` in the pilot). Documentation in `data_report.split_info` should confirm. |
| `dp_07` | Multilevel limitation acknowledged in `data_report.warnings` | major | Semantic match for the multilevel-limitation string; absent → major. See `school-aware-train-test-split` for the canonical text. |
| `dp_08` | Train/test split is school-aware (group-disjoint) | critical | No school may appear in both train and test. Verify `data_report.split_info.group_overlap == 0`. Missing `split_info` or `group_overlap > 0` → critical (school-level leakage). See `school-aware-train-test-split`. |
| `dp_09` | `is_imbalanced` flag consistent with reported class balance | major | Binary outcomes: if the minority class share < 20%, `data_report.is_imbalanced` must be `true`. Inconsistent flag → major (the ablation checks `an_11`–`an_13` key off this field, so a wrong flag silently disables them). |

## Section 3 — Analysis

Evaluated against `results.json`, `data_report.json`.

| ID | Item | Severity | Check |
|---|---|---|---|
| `an_01` | At least 5 individual model families + StackingEnsemble | major | Expect LR, RF, XGBoost, ElasticNet, MLP + Stacking. < 4 individual families → major. (When `mlp_enabled: false`, 4 individual + Stacking = 5 total is acceptable.) |
| `an_02` | Hyperparameters tuned via inner CV on training data only | critical | `results.warnings` / `results.errors` mention test-set leakage during tuning → critical. Missing tuning info for RF/XGBoost → major. See `inner-cv-tuning-discipline`. |
| `an_03` | All evaluation metrics from held-out test set | critical | Red flag: unusually high values (e.g., `R² ≈ 1.0` for a complex outcome) → suspect training-set evaluation. |
| `an_04` | Confidence intervals provided for primary metric | major | Classification: `auc_ci_lower` / `auc_ci_upper` non-null for every model. Regression: `rmse_ci_lower` / `rmse_ci_upper`. Missing → major. See `bootstrap-confidence-intervals`. |
| `an_05` | SHAP analysis present for best individual model with appropriate explainer | major | `results.top_features` non-empty AND `results.figures_generated` includes `shap_summary.png` and `shap_importance.png`. Explainer matches the model family per `shap-explainer-selection`. SHAP absent due to documented timeout → minor. |
| `an_06` | Subgroup analysis conducted for all protected attributes | major | Every attribute in `research_spec.subgroup_analyses` appears as a key in `results.subgroup_performance`. Missing any → major. See `subgroup-fairness-analysis`. |
| `an_07` | Subgroup gaps > 5% are flagged | major | For each attribute, `max(metric) - min(metric)` computed; > 0.05 with no warning in `results.warnings` → major. |
| `an_08` | StackingEnsemble in `model_comparison.csv` but NOT in SHAP outputs | major | The SHAP source model (the one in `top_features`) MUST NOT be StackingEnsemble. SHAP computed for stacking → major. |
| `an_09` | KernelExplainer used only for MLP with appropriate constraints | major | If KernelExplainer was used: model must be MLP, sample cap ≤ 1,000, `nsamples` ≤ 500. Wrong model → major. Constraints violated → minor. |
| `an_10` | MLP KernelExplainer timeout fallback documented | major | If MLP timed out: SHAP outputs must come from the next-best non-MLP individual model AND the fallback must be documented in `results.warnings` with the model name. Undocumented or missing → major. |
| `an_11` | Ablation present when data is imbalanced | major | If `data_report.is_imbalanced == true` AND classification task: `results.ablation` must be present (non-null). Missing → major. See `smote-imbalance-handling`. |
| `an_12` | Ablation test-set size matches primary results | critical | `n_test` must match between the ablation and the primary results. A mismatch means SMOTE was applied to the test set — critical bug (synthetic samples contaminate the evaluation). |
| `an_13` | F2 and balanced accuracy reported when SMOTE applied | major | If SMOTE was applied: `f2` and `balanced_accuracy` must be present for every model in `results.all_models`. Missing → major. |

## Section 4 — Substantive Validity

The Critic uses domain judgment here; these are not purely mechanical
checks.

| ID | Item | Severity | Check |
|---|---|---|---|
| `sv_01` | Top SHAP features make educational sense | major | For each of the top 3–5 features by `shap_mean_abs`, assess plausibility against the educational literature and the registry's rationale for that variable. School ID or a processing flag in top features → leakage (critical). |
| `sv_02` | Findings are not trivially obvious | minor | Predicting math achievement from prior math score alone (with high AUC) is uninteresting unless novel predictors, novel population, or novel methodology are added. Purely confirmatory → minor. |
| `sv_03` | `AUC > 0.95` flagged as suspicious | major | Always flag, regardless of other indicators. Investigate dominant SHAP features. Top feature is a strong proxy for the outcome → critical (confirmed leakage). Otherwise → major (high performance unverified). |
| `sv_04` | Limitations honestly and specifically acknowledged | minor | Verify (semantic match): (1) multilevel structure not modeled, (2) survey weights not applied, (3) missing-data mechanism assumed MAR. The Writer incorporates these via the writing limitations skills. |
| `sv_05` | Unexpected/counterintuitive findings flagged for the Writer | minor | If a top SHAP feature has an unexpected direction (e.g., engagement scale with negative SHAP for an achievement outcome), note in `revision_instructions.Writer` so the Writer interprets explicitly rather than ignoring. |

## Severity definitions

- **critical**: Invalidates the study or makes results uninterpretable.
  Triggers REVISE if fixable, ABORT if not.
- **major**: Significantly weakens validity, credibility, or
  reproducibility. Each counts toward the > 2 major issues threshold
  for REVISE.
- **minor**: Worth noting but does not threaten validity. Does not
  affect the PASS/REVISE threshold; noted for the Writer.

## Cross-references

The Critic checks individual items by composing other skills:

- Temporal ordering → `hsls09-temporal-ordering`
- Missing-data protocol → `missingness-tiered-protocol`
- SHAP explainer → `shap-explainer-selection`
- Subgroup analysis → `subgroup-fairness-analysis`
- Inner-CV discipline → `inner-cv-tuning-discipline`
- Bootstrap CIs → `bootstrap-confidence-intervals`
- Quality gate → `prediction-quality-gate`
- SMOTE handling → `smote-imbalance-handling`

## Source provenance

Canonical source: `data_registry/evaluation_rubrics/methodological_checklist.yaml`
(entire file, L24-L419). Verdict criteria and severity definitions
are also in `agent_prompts/critic.yaml`, but the methodological
checklist is the more complete catalog and is the canonical source
for this skill.

Rows `dp_08`–`dp_09` and `an_11`–`an_13` were harvested in V2.1 Phase
3b.23 from the V1 Critic prompt (`agent_prompts/critic.v1.yaml.bak`
L118–L120 and L142–L146): the carrier skills for those checks
(`school-aware-train-test-split`, `smote-imbalance-handling`) are
DataEngineer-/Analyst-stage only and never reach the Critic prompt, so
the checks had to live here. `rule_severity: mandatory` was added in
the same phase (per migration spec §3.3.4): dropping this checklist
under the per-tier formatter cap would make the review structurally
incomplete.
