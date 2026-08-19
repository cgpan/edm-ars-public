---
name: model-logistic-regression
layer: task-type
description: Logistic / linear regression baseline for the prediction battery; L2 by default; LinearExplainer for SHAP.
trigger_keywords:
  - logistic
  - logisticregression
  - linear
  - linearregression
  - baseline
  - regression
applicable_task_types:
  - prediction
applicable_datasets: []
applicable_stages:
  - Analyst
priority: 2
references_skills: []
resources: []
version: "1.0"
---

# Logistic / Linear Regression (Baseline)

Always present in the prediction battery as the baseline. No
hyperparameter grid in the pilot — `C=1.0` for classification, default
parameters for regression. Use the result as the floor against which the
ensembles must demonstrate improvement.

## Implementation

```python
from sklearn.linear_model import LogisticRegression, LinearRegression

if is_classification:
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
else:
    model = LinearRegression()

model.fit(train_X, train_y)
```

## Hyperparameters (pilot — fixed, not tuned)

| Parameter | Value |
|---|---|
| `C` (classification) | 1.0 |
| `max_iter` | 1000 (avoids ConvergenceWarning on most HSLS-scale problems) |
| `random_state` | 42 |
| `penalty` | `'l2'` (default) |

## SHAP explainer

`shap.LinearExplainer`. The model is linear, so SHAP values are exact
and fast — no sample cap needed.

## Typical failure modes

- **ConvergenceWarning at default `max_iter`**. Bump to `max_iter=2000`
  if it persists; do not set `random_state=None`. Log the warning in
  `results.warnings` but do not abort.
- **Class imbalance with default `class_weight=None`** produces a
  classifier that always predicts the majority class on extreme
  imbalance. SMOTE handling lives in the `smote-imbalance-handling`
  skill; LR is one of the most sensitive models in the battery to
  imbalance and benefits the most from SMOTE.
- **Multicollinearity** among predictors inflates standard errors but
  does not break the fit. SHAP values become unstable across runs
  when collinear features compete for attribution; document in
  `results.warnings` if `condition_number(train_X)` is suspicious.

## Source provenance

Canonical source: `agent_prompts/analyst.yaml` Pilot Model Battery
row 1 + `data_registry/task_templates/prediction.yaml` `model_lr`.
Per-family extraction is the Decision-9 expansion of the audit's
`prediction-model-battery` skill.
