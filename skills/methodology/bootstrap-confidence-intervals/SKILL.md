---
name: bootstrap-confidence-intervals
layer: methodology
description: 1000-iteration bootstrap CI for the primary metric, computed via the project helper.
trigger_keywords:
  - bootstrap
  - confidence
  - ci
  - resampling
applicable_task_types: []
applicable_datasets: []
applicable_stages:
  - Analyst
priority: 1
references_skills: []
resources: []
version: "1.0"
---

# Bootstrap Confidence Intervals

Every model's primary metric (AUC for classification, RMSE for regression)
must be reported with a 95% bootstrap confidence interval computed on the
held-out test set.

## Procedure

- **Iterations**: 1,000.
- **Random seed**: `random_state=42` (reproducibility).
- **Resampling unit**: rows of the test set, with replacement.
- **CI definition**: 2.5th / 97.5th percentiles of the bootstrap
  distribution of the metric.

Always go through the project helper:

```python
import analysis_helpers

ci_lower, ci_upper = analysis_helpers.bootstrap_ci(
    y_true=test_y_arr,
    y_pred=best_model_preds_or_probs,  # probabilities for AUC, predictions for RMSE
    metric_fn=primary_metric_fn,        # roc_auc_score or sqrt(MSE)
    n_iterations=1000,
    random_state=42,
)
```

The helper handles the AUC-vs-RMSE distinction (probabilities vs point
predictions), edge cases (all-positive or all-negative bootstrap samples
for AUC), and percentile computation.

## What to record in `results.json`

Per-model CIs go inside `all_models`:

```json
{
  "all_models": {
    "XGBoost": {
      "auc": 0.823,
      "auc_ci_lower": 0.798,
      "auc_ci_upper": 0.847
    },
    "LinearRegression": {
      "rmse": 0.71,
      "rmse_ci_lower": 0.69,
      "rmse_ci_upper": 0.73
    }
  }
}
```

For classification use the `auc_ci_lower` / `auc_ci_upper` keys; for
regression use the `rmse_ci_lower` / `rmse_ci_upper` keys.

## When clustered data are present

For clustered data (e.g., students within schools), the row-level
bootstrap above underestimates uncertainty for cluster-level predictors.
Compose this skill with `clustered-bootstrap-ci-and-icc` and report
**both** standard and clustered CIs for the best model so reviewers can
see the difference.

## Verification rules (Critic)

The Critic verifies that for every model entry in `results.all_models`,
the `*_ci_lower` and `*_ci_upper` fields are present and non-null.
Missing CIs are a major issue.

## Source provenance

Canonical source: `agent_prompts/analyst.yaml` §"Evaluation Protocol"
(the `analysis_helpers.bootstrap_ci` invocation pattern).

Merged content from:
- `data_registry/task_templates/prediction.yaml` §`evaluation.metrics`
  (the `ci_iterations: 1000`, `ci_random_state: 42` configuration)
- `data_registry/evaluation_rubrics/methodological_checklist.yaml` item
  `an_04` (Critic verification language)
