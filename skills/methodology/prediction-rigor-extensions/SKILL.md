---
name: prediction-rigor-extensions
layer: methodology
description: Reviewer-grade rigor for prediction papers — moderation sub-questions must be COMPUTED via run_moderation_analysis, dummy SHAP grouped by parent variable, best-model claims paired-tested, calibration quantified.
trigger_keywords:
  - moderation
  - interaction
  - calibration
  - rigor
  - shap
  - auc
applicable_task_types:
  - prediction
applicable_datasets: []
applicable_stages:
  - Analyst
  - Critic
  - Writer
priority: 1
references_skills:
  - shap-explainer-selection
  - subgroup-fairness-analysis
  - clustered-bootstrap-ci-and-icc
resources: []
version: "1.0"
rule_severity: mandatory
---

# Prediction Rigor Extensions

Four reviewer-named gaps, each with a certified deterministic helper.
Generated code MUST call the helpers — reimplementation is a contract
violation.

## 1. Every moderation sub-question must be COMPUTED

If the research question or research_spec promises a moderation /
interaction analysis ("above and beyond", "does X moderate", "varies by
SES"), the Analyst MUST run it — never silently drop it:

```python
# Signature (use EXACTLY these parameter names):
#   run_moderation_analysis(X, y, focal_cols, moderator_col,
#                           n_boot=200, random_state=42)
X_all = pd.concat([train_X, test_X], ignore_index=True)
y_all = np.concatenate([train_y_arr, test_y_arr])
results["moderation_analysis"] = analysis_helpers.run_moderation_analysis(
    X=X_all, y=y_all,
    focal_cols=[c for c in X_all.columns if c.startswith("BYSTEXP")],
    moderator_col="BYSES1",
)
```

focal_cols are the ENCODED dummy columns of the focal construct (prefix
match on the encoded matrix), and the moderator is a continuous encoded
column.

The helper returns an interaction LRT (test + df + p) plus the focal
block's incremental AUC within moderator tertiles with a bootstrap CI on
the top-minus-bottom difference. If genuinely infeasible, results must
carry `moderation_analysis: {"status": "skipped", "reason": ...}` AND the
Writer must descope it explicitly in Limitations.

## 2. Dummy SHAP grouped by parent variable

Per-dummy SHAP values (`BYSTEXP_5.0`, `BYSTEXP_6.0`, ...) understate the
parent construct and invite reference-category misreadings. Always also
report the grouped view:

```python
results["top_feature_groups"] = analysis_helpers.group_shap_by_parent(
    feature_names, shap_mean_abs_values)
```

Writer rule: interpret any dummy-level SHAP direction RELATIVE TO THE
REFERENCE CATEGORY, and say so in one sentence the first time; feature
importance prose leads with the grouped table.

## 3. Best-model claims need a paired test

"Model A outperformed B" requires the paired bootstrap difference —
cluster-aware when school IDs exist. NEVER skip this field:

```python
# Signature: bootstrap_auc_difference(y_true, prob_a, prob_b,
#                                     school_ids=None, n_boot=1000,
#                                     random_state=42)
if best_model_name != "LogisticRegression":
    a, b = prob_best, prob_logistic_baseline     # best vs LR baseline
else:
    a, b = prob_best, prob_runner_up             # LR won: test vs runner-up
results["model_comparison_test"] = analysis_helpers.bootstrap_auc_difference(
    y_true=test_y_arr, prob_a=a, prob_b=b, school_ids=test_school_ids)
```

If the logistic baseline itself is the best model, the comparison is
baseline-vs-runner-up and the paper reports that the simplest model was
not beaten — that honesty is a rigor feature, not a weakness. If
`significant` is false either way, say the models are statistically
indistinguishable.

## 4. Calibration quantified

```python
# Signature: compute_calibration_metrics(y_true, y_prob, n_bins=10)
# Returns ALL FOUR fields (brier, ece, calibration_slope,
# calibration_intercept) from this ONE call — never compute any of them
# by hand or leave them null.
results["calibration"] = analysis_helpers.compute_calibration_metrics(
    y_true=test_y_arr, y_prob=prob_best)
```

Writer reports Brier score and calibration slope/intercept alongside AUC
(a discriminative model can still be badly calibrated; say which it is).

## Critic rows (walk every one)

| ID | Item | Severity | Check |
|---|---|---|---|
| `rig_01` | Moderation computed or descoped | critical | Any moderation phrasing in the RQ/spec → `results.moderation_analysis.status == "computed"`, else an explicit skipped-reason + Limitations descope. |
| `rig_02` | Grouped SHAP present | major | `results.top_feature_groups` non-empty when SHAP ran. |
| `rig_03` | Best-model claim tested | major | `results.model_comparison_test` present; prose claims match `significant`. |
| `rig_04` | Calibration reported | major | `results.calibration.brier` present; Writer reports it. |
| `rig_05` | Reference-category sentence | minor | Paper explains dummy SHAP signs relative to the reference category. |
