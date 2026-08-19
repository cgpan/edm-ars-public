---
name: sensitivity-analysis-high-missingness
layer: methodology
description: Mandatory sensitivity rerun when any predictor has >20% missingness; report metric change and top-feature stability.
trigger_keywords:
  - sensitivity
  - missingness
  - robustness
  - ablation
applicable_task_types: []
applicable_datasets: []
applicable_stages:
  - Analyst
  - Writer
priority: 2
references_skills: []
resources: []
version: "1.0"
rule_severity: mandatory
---

# Sensitivity Analysis for High-Missingness Predictors

When any predictor has >20% missingness, the imputation choice can drive
the model's findings. A sensitivity analysis reruns the best-model class
on a reduced predictor set (high-missingness vars excluded) and reports
whether the conclusions are robust.

This is **not optional**. If high-missingness predictors exist and no
sensitivity analysis is reported, reviewers will (rightly) ask for one.

## When to run

After the main analysis (model comparison + SHAP + subgroup) is complete,
inspect `data_report.missingness_summary`:

```python
high_miss_vars = [
    var for var, info in data_report.get("missingness_summary", {}).items()
    if info.get("pct_missing", 0) > 20
]
```

If `high_miss_vars` is non-empty, run the sensitivity analysis. Otherwise,
set `results["sensitivity_analysis"] = None`.

## How to run

Use the project helper, which handles refit + reevaluation on the same
test set:

```python
import analysis_helpers

if high_miss_vars:
    sensitivity = analysis_helpers.run_sensitivity_analysis(
        best_model_class=type(best_individual_model),
        best_model_params=best_individual_model.get_params(),
        train_X=train_X,
        train_y=train_y_arr,
        test_X=test_X,
        test_y=test_y_arr,
        high_miss_vars=high_miss_vars,
        is_classification=is_classification,
    )
    results["sensitivity_analysis"] = sensitivity
else:
    results["sensitivity_analysis"] = None
```

Save the full sensitivity payload to `sensitivity_analysis.json` in the
output directory so the Writer can cite it.

If the sensitivity analysis itself fails (e.g., too few columns remain
after exclusion), log the failure in `results.warnings` and set
`results["sensitivity_analysis"] = {"error": "<description>"}` instead of
omitting the key.

## What to record

The helper returns a dict shaped like this:

```json
{
  "sensitivity_analysis": {
    "excluded_variables": ["X1PAREDU"],
    "full_model_metric": 0.855,
    "reduced_model_metric": 0.841,
    "metric_change_pct": -1.6,
    "significant_change": false,
    "top5_overlap": 4,
    "conclusion": "Results are robust to exclusion of high-missingness variables."
  }
}
```

Decision rule for `significant_change`: `true` if the absolute relative
change in the primary metric exceeds 5%.

## Writer's responsibility

The Writer MUST report sensitivity findings in §Limitations whenever
`results.sensitivity_analysis` is non-null. State which variables were
excluded, the metric change, the top-5 SHAP overlap, and the conclusion.
Reviewers specifically look for this paragraph.

## Source provenance

Canonical source: `agent_prompts/analyst.yaml` §"Sensitivity Analysis".

Merged content from: none — this section is single-sourced. The Writer
reporting requirement is also referenced in
`agent_prompts/writer.yaml` §"Sensitivity Analysis — Report Results".
