---
name: prediction-rigor-extensions
layer: methodology
description: Reviewer-grade rigor for prediction papers — moderation sub-questions must be COMPUTED via run_moderation_analysis, "above and beyond" claims via run_incremental_validity, dummy SHAP grouped by parent variable, best-model claims paired-tested, calibration quantified.
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

Five reviewer-named gaps, each with a certified deterministic helper.
Generated code MUST call the helpers — reimplementation is a contract
violation.

## 1. Every moderation sub-question must be COMPUTED

If the research question or research_spec promises a moderation /
interaction analysis ("does X moderate", "varies by SES"), the Analyst
MUST run it — never silently drop it. ("Above and beyond" is a
different question; see section 1b.)

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

## 1b. Every incremental-validity claim must be COMPUTED

"Does X predict Y above and beyond (over and above) A and B" asks whether
the focal block adds predictive power over a baseline block. That needs
two nested models compared on the held-out test set. A SHAP ranking
inside one model does not answer it, and neither does moderation.

```python
# Signature (use EXACTLY these parameter names):
#   run_incremental_validity(train_X, train_y, test_X, test_y,
#                            focal_cols, baseline_cols=None,
#                            school_ids=None, n_boot=1000, random_state=42,
#                            outcome_type=None)
baseline_cols = [c for c in train_X.columns
                 if c.startswith(("X1TXMTSCOR", "X1SES"))]   # the named A and B
focal_cols = [c for c in train_X.columns
              if c.startswith(("X1MTHID", "X1SCHOOLBEL"))]  # ONLY what the question credits
results["incremental_validity"] = analysis_helpers.run_incremental_validity(
    train_X, train_y_arr, test_X, test_y_arr,
    focal_cols=focal_cols, baseline_cols=baseline_cols,
    school_ids=test_school_ids,            # None when test_school_ids.csv is absent
    outcome_type="binary")                 # data_report.json outcome_type: "binary" or "continuous"
```

The baseline is everything the question names after "above and beyond",
controls included ("academic achievement, SES, and demographic controls"
is three things); map it to predictor_set variables and take their
ENCODED columns. focal_cols holds only the constructs the question
credits. Never default focal_cols to "every other column": a column in
focal_cols is credited to the focal constructs, while a column in neither
list is left out of both models.

The helper handles binary outcomes (baseline_auc, full_auc, delta_auc)
and continuous ones (baseline_r2, full_r2, delta_r2 and both RMSEs), each
with a bootstrap CI on the difference, and records the column lists it
used. It returns `{"status": "skipped" | "error", "reason": ...}` instead
of raising, so never wrap it in a try/except that writes null or a note.

There is no descope for this one, and it is never "not applicable to
regression": the claim is the paper's contribution. Only `"status":
"ok"` satisfies the pre-review check (pcc_07); null, an empty record or
a skipped/error record does not, and the study stops rather than publish
the untested claim. Writer: report delta_auc (or delta_r2) with its CI;
when the CI includes 0, say the focal block adds no detectable predictive
power over the baseline.

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
#                                     random_state=42,
#                                     model_a=None, model_b=None)
if best_model_name != "LogisticRegression":
    a, b = prob_best, prob_logistic_baseline     # best vs LR baseline
    name_a, name_b = best_model_name, "LogisticRegression"
else:
    a, b = prob_best, prob_runner_up             # LR won: test vs runner-up
    name_a, name_b = best_model_name, runner_up_name
results["model_comparison_test"] = analysis_helpers.bootstrap_auc_difference(
    y_true=test_y_arr, prob_a=a, prob_b=b, school_ids=test_school_ids,
    model_a=name_a, model_b=name_b)
```

**`model_a` and `model_b` are mandatory.** Omitting them leaves
results.json with an `auc_diff` and no record of what was compared, and
the helper marks the result `comparands_unnamed: true`. Two delivered
papers both named RandomForest as the runner-up they had tested; the
stored difference was bit-for-bit XGBoost minus LogisticRegression in
one and the reverse in the other. In both, that was the paper's only
inferential test. The Writer must quote `contrast` rather than reasoning
about which model "should" have been the comparator.

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
| `rig_07` | Incremental validity computed | critical | "Above and beyond" / "over and above" / "incremental validity" in the RQ → `results.incremental_validity.status == "ok"` with delta and CI; prose claims match the CI. No descope. |
| `rig_02` | Grouped SHAP present | major | `results.top_feature_groups` non-empty when SHAP ran. |
| `rig_03` | Best-model claim tested | major | `results.model_comparison_test` present; prose claims match `significant`. |
| `rig_06` | Comparands named | **critical** | `results.model_comparison_test.model_a` and `.model_b` are non-null and `comparands_unnamed` is absent. Any prose naming the compared models must match `contrast` exactly. An unnamed comparison cannot be written up. |
| `rig_04` | Calibration reported | major | `results.calibration.brier` present; Writer reports it. |
| `rig_05` | Reference-category sentence | minor | Paper explains dummy SHAP signs relative to the reference category. |
