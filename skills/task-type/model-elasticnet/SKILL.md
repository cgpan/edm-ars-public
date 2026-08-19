---
name: model-elasticnet
layer: task-type
description: ElasticNet (regression) or SGDClassifier with elasticnet penalty (classification); tune via 5-fold inner CV; LinearExplainer.
trigger_keywords:
  - elasticnet
  - elastic
  - sgd
  - sgdclassifier
  - regularized
  - regularization
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

# ElasticNet (L1 + L2 Regularized Linear)

Linear baseline with mixed L1/L2 regularization. Useful as a sparse
counterpart to plain Logistic/Linear Regression — the L1 component
zeroes out predictors the model finds uninformative, which is a free
poor-man's feature selection.

For **regression**, use `sklearn.linear_model.ElasticNet`. For
**classification**, the pilot uses `SGDClassifier(loss='log_loss',
penalty='elasticnet')` — sklearn does not ship a stand-alone
`LogisticRegression` with elasticnet penalty in stable form.

## Implementation

```python
from sklearn.linear_model import ElasticNet, SGDClassifier

if is_classification:
    estimator = SGDClassifier(
        loss="log_loss",
        penalty="elasticnet",
        random_state=42,
        max_iter=1000,
    )
else:
    estimator = ElasticNet(random_state=42, max_iter=1000)
```

## Hyperparameter grid (5-fold inner CV; group-aware when school IDs are present)

| Parameter | Grid |
|---|---|
| `alpha` | `[0.001, 0.01, 0.1, 1.0]` |
| `l1_ratio` | `[0.1, 0.5, 0.7, 0.9]` |

`alpha` is the overall regularization strength; `l1_ratio` is the
L1/L2 mix (0 = pure L2 / Ridge, 1 = pure L1 / Lasso).

```python
from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(
    estimator,
    param_grid={
        "alpha": [0.001, 0.01, 0.1, 1.0],
        "l1_ratio": [0.1, 0.5, 0.7, 0.9],
    },
    cv=cv_splits,
    scoring=scoring,
    n_jobs=-1,
)
grid_search.fit(train_X, train_y)
best_enet = grid_search.best_estimator_
```

## SHAP explainer

`shap.LinearExplainer`. The model is linear in the features (after
the elasticnet penalty), so SHAP values are exact and fast.

For `SGDClassifier`, `LinearExplainer` reads the `coef_` attribute
the same way it does for `LogisticRegression`. No special handling.

## Typical failure modes

- **`alpha` too large** drives all coefficients to 0; the model
  predicts the prior probability for every test row. AUC ≈ 0.5,
  RMSE ≈ baseline. The grid above includes `alpha=0.001` to give the
  model a fighting chance.
- **`SGDClassifier` ignores `class_weight` by default** for the
  elasticnet penalty. SMOTE handling (`smote-imbalance-handling`) is
  the pilot's mitigation.
- **Convergence under `max_iter=1000`** can fail on very wide feature
  matrices. Bump to `max_iter=2000` if you see ConvergenceWarning;
  `tol=1e-3` is a reasonable second knob.

## Source provenance

Canonical source: `agent_prompts/analyst.yaml` Pilot Model Battery
row 4 + `data_registry/task_templates/prediction.yaml` `model_enet`.
Per-family extraction (Decision 9).
