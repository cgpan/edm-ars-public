---
name: model-xgboost
layer: task-type
description: XGBoost gradient-boosted trees; tune via 5-fold inner CV; TreeExplainer for SHAP. LightGBM is the documented alternative.
trigger_keywords:
  - xgboost
  - xgb
  - gradient
  - boosting
  - lightgbm
  - lgbm
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

# XGBoost (Gradient-Boosted Trees)

XGBoost is the primary gradient boosting library in the pilot;
LightGBM is acceptable as a substitute. Often the strongest individual
model on tabular EDM data — the model the linear baseline and the RF
must beat for the analysis to feel non-trivial.

## Implementation

```python
from xgboost import XGBClassifier, XGBRegressor

cls = XGBClassifier if is_classification else XGBRegressor
estimator = cls(
    random_state=42,
    eval_metric="logloss" if is_classification else "rmse",
    use_label_encoder=False,  # silences XGBoost 1.x warning; harmless on 2.x
)
```

## Hyperparameter grid (5-fold inner CV; group-aware when school IDs are present)

| Parameter | Grid |
|---|---|
| `learning_rate` | `[0.01, 0.05, 0.1]` |
| `n_estimators` | `[100, 300, 500]` |
| `max_depth` | `[3, 5, 7]` |

Tuning protocol: see `inner-cv-tuning-discipline`.

```python
from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(
    estimator,
    param_grid={
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [100, 300, 500],
        "max_depth": [3, 5, 7],
    },
    cv=cv_splits,
    scoring=scoring,
    n_jobs=-1,
)
grid_search.fit(train_X, train_y)
best_xgb = grid_search.best_estimator_
```

`random_state=42` for the underlying estimator. Per-step timeout is 300s
(this grid trains 27 × CV-folds models).

## SHAP explainer

`shap.TreeExplainer` — exact and fast for boosted trees. Use the
`output_margin=False` default for classification so SHAP values are in
log-odds space; the `analysis_helpers` plotters handle the
log-odds → probability narrative.

## LightGBM substitution

If `xgboost` is not installed, swap in `LGBMClassifier` /
`LGBMRegressor`. The hyperparameter grid translates 1:1; SHAP
explainer remains `TreeExplainer`.

## Typical failure modes

- **`use_label_encoder=False` on XGBoost 1.x** — required to silence a
  deprecation warning. On 2.x it has been removed; pass it conditional
  on `xgboost.__version__ < "2.0"` or wrap in `try/except TypeError`.
- **Imbalanced classification without `scale_pos_weight`**: XGBoost
  defaults assume balanced classes. SMOTE handling
  (`smote-imbalance-handling`) is the pilot's mitigation; do NOT also
  set `scale_pos_weight` when SMOTE is applied (double correction).

## Source provenance

Canonical source: `agent_prompts/analyst.yaml` Pilot Model Battery
row 3 + `data_registry/task_templates/prediction.yaml` `model_xgb`.
Per-family extraction (Decision 9).
