---
name: model-random-forest
layer: task-type
description: Random Forest classifier/regressor; tune via 5-fold inner CV; TreeExplainer for SHAP.
trigger_keywords:
  - random
  - forest
  - randomforest
  - rf
  - tree
  - trees
  - ensemble
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

# Random Forest

A tree-bagging ensemble. Robust to outliers and unscaled features,
handles mixed-type inputs cleanly, and (with `TreeExplainer`) produces
fast, exact SHAP values. Strong baseline ensemble — the result the
gradient-boosted model has to beat.

## Implementation

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

cls = RandomForestClassifier if is_classification else RandomForestRegressor
estimator = cls(random_state=42)
```

## Hyperparameter grid (5-fold inner CV; group-aware when school IDs are present)

| Parameter | Grid |
|---|---|
| `n_estimators` | `[100, 300, 500]` |
| `max_depth` | `[5, 10, None]` |
| `min_samples_leaf` | `[1, 5, 10]` |

Tuning protocol: see the `inner-cv-tuning-discipline` methodology skill.
Use `GridSearchCV` with `scoring='roc_auc'` (classification) or
`'neg_root_mean_squared_error'` (regression).

```python
from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(
    estimator,
    param_grid={
        "n_estimators": [100, 300, 500],
        "max_depth": [5, 10, None],
        "min_samples_leaf": [1, 5, 10],
    },
    cv=cv_splits,            # group-aware folds when train_school_ids exists
    scoring=scoring,
    n_jobs=-1,
)
grid_search.fit(train_X, train_y)
best_rf = grid_search.best_estimator_
```

`random_state=42` for the underlying estimator. `n_jobs=-1` is fine for
both fitting and grid search; the per-step timeout is 300s.

## SHAP explainer

`shap.TreeExplainer` — exact and fast for forests; no sample cap
needed for typical HSLS-scale test sets.

## Typical failure modes

- **`max_depth=None` + huge `n_estimators` on small training sets**
  overfits aggressively. The grid above caps depth at `None` only
  alongside `min_samples_leaf >= 5` to limit single-leaf decisions.
- **TreeExplainer + categorical encoding mismatch** can produce SHAP
  values that don't match the model's predictions. Always pass the
  same column order/encoding used at fit time.

## Source provenance

Canonical source: `agent_prompts/analyst.yaml` Pilot Model Battery
row 2 + `data_registry/task_templates/prediction.yaml` `model_rf`.
Per-family extraction (Decision 9).
