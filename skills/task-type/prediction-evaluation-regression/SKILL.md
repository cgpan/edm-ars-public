---
name: prediction-evaluation-regression
layer: task-type
description: Regression evaluation protocol — RMSE primary, MAE/R² secondary, bootstrap CI, residual plot.
trigger_keywords:
  - regression
  - rmse
  - mae
  - r2
  - residual
  - residuals
applicable_task_types:
  - prediction
applicable_datasets: []
applicable_stages:
  - Analyst
priority: 1
references_skills: []
resources: []
version: "1.0"
---

# Prediction Evaluation — Regression

For continuous outcomes. All metrics computed on the held-out test set
only.

## Primary metric

**RMSE**, computed as `sqrt(mean_squared_error)`. Lower is better.

Confidence interval: 1000-iteration bootstrap on RMSE,
`random_state=42`, through `analysis_helpers.bootstrap_ci()`. See
`bootstrap-confidence-intervals`.

## Secondary metrics (all reported per model)

| Metric | sklearn fn |
|---|---|
| MAE | `mean_absolute_error` |
| R² | `r2_score` |

## Required figures

| File | Description |
|---|---|
| `residual_plot.png` | Predicted vs. actual scatter for the best model |

Plus the cross-task-type figures (best model only) from
`shap-explainer-selection`: `shap_summary.png`, `shap_importance.png`,
`pdp_*.png`.

Use `matplotlib.use('Agg')` to avoid display issues in subprocess
execution. Save at 150 dpi minimum.

## Best-model selection

`results.best_model` = the family with the **lowest** test-set RMSE.
Ties broken by simpler model first (LR > ElasticNet > RF > XGBoost
> MLP).

StackingEnsemble appears in `model_comparison.csv` but is excluded
from the SHAP/interpretability "best individual model" — see
`shap-explainer-selection`.

## What goes in `results.json`

```json
{
  "best_model": "XGBoost",
  "best_metric_value": 0.61,
  "primary_metric": "RMSE",
  "all_models": {
    "LinearRegression": {
      "rmse": 0.71, "mae": 0.55, "r2": 0.38,
      "rmse_ci_lower": 0.69, "rmse_ci_upper": 0.73
    },
    "RandomForest": {
      "rmse": 0.65, "mae": 0.50, "r2": 0.48,
      "rmse_ci_lower": 0.63, "rmse_ci_upper": 0.67
    },
    "XGBoost": {
      "rmse": 0.61, "mae": 0.47, "r2": 0.53,
      "rmse_ci_lower": 0.59, "rmse_ci_upper": 0.63
    }
  },
  "figures_generated": [
    "residual_plot.png",
    "shap_summary.png", "shap_importance.png", "pdp_X1TXMTSC.png", ...
  ]
}
```

## R² floor for SHAP

The quality gate (`prediction-quality-gate`) uses **R² ≥ 0.05** as the
minimum threshold for a regression model to be eligible for SHAP. A
model with R² < 0.05 does not have meaningful predictive structure;
its SHAP values would be noise.

## Source provenance

Canonical source: `data_registry/task_templates/prediction.yaml`
§`evaluation.metrics.regression` (L469-L484) + `agent_prompts/analyst.yaml`
§"Evaluation Protocol — Regression".

The bootstrap CI procedure is the cross-cutting methodology skill
`bootstrap-confidence-intervals`.
