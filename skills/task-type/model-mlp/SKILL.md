---
name: model-mlp
layer: task-type
description: MLPClassifier/MLPRegressor with early stopping; tune via 5-fold inner CV; KernelExplainer with bounded sample cap.
trigger_keywords:
  - mlp
  - neural
  - network
  - networks
  - mlpclassifier
  - mlpregressor
  - feedforward
applicable_task_types:
  - prediction
applicable_datasets: []
applicable_stages:
  - Analyst
priority: 3
references_skills: []
resources: []
version: "1.0"
---

# MLP (Multi-Layer Perceptron)

The non-tree, non-linear model in the battery. Often does not beat
XGBoost on tabular EDM data, but worth fitting because some outcomes
do show non-linear interactions among continuous predictors that the
gradient-boosted model under-represents.

This is the only model in the battery that uses `KernelExplainer` for
SHAP, which carries serious operational risk — see the constraints
section below.

## Configuration toggle

The Analyst checks `## Configuration` in the user message for
`mlp_enabled`:
- `mlp_enabled: false` → **skip MLP entirely**. Build StackingEnsemble
  from the four remaining base models. Total models = 5.
- `mlp_enabled: true` (or absent) → include MLP. Total models = 6.

When MLP is skipped, do NOT add a placeholder MLP entry to
`results.all_models`; simply omit it.

## Implementation

```python
from sklearn.neural_network import MLPClassifier, MLPRegressor

cls = MLPClassifier if is_classification else MLPRegressor
estimator = cls(
    random_state=42,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
)
```

`max_iter=500`, `early_stopping=True`, and `validation_fraction=0.1`
are **fixed** — not part of the tuning grid. Early stopping uses an
internal 10% validation split (taken from the training set, not the
held-out test set).

## Hyperparameter grid (5-fold inner CV; group-aware when school IDs are present)

| Parameter | Grid |
|---|---|
| `hidden_layer_sizes` | `[(64,), (128,), (64, 32)]` |
| `learning_rate_init` | `[0.001, 0.01]` |
| `alpha` (L2) | `[0.0001, 0.001]` |

```python
from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(
    estimator,
    param_grid={
        "hidden_layer_sizes": [(64,), (128,), (64, 32)],
        "learning_rate_init": [0.001, 0.01],
        "alpha": [0.0001, 0.001],
    },
    cv=cv_splits,
    scoring=scoring,
    n_jobs=-1,
)
grid_search.fit(train_X, train_y)
best_mlp = grid_search.best_estimator_
```

## SHAP explainer (CRITICAL constraints)

MLP is the only family in the pilot that uses `shap.KernelExplainer`.
KernelExplainer is model-agnostic but slow — without bounded sample
caps it will silently grow into a multi-hour computation that blows
the per-step timeout.

Hard constraints (also in the methodology skill `shap-explainer-selection`):

| Constraint | Value |
|---|---|
| Sample cap | 1,000 rows max from the test set (sample with `random_state=42` if `n_test > 1000`) |
| Background data | `shap.kmeans(train_X, 100)` |
| `nsamples` | 500 |
| Hard timeout | 600 seconds (signal-based or subprocess-wrapped) |

If KernelExplainer times out for MLP, the Analyst falls back to the
**next-best non-MLP individual model** for ALL interpretability outputs
(beeswarm, bar plot, PDPs, `feature_importance.csv`) and documents the
fallback in `results.warnings`.

## Typical failure modes

- **Convergence under `max_iter=500`** is common on hard problems.
  Early stopping mitigates: the trained network is the best
  validation-loss network seen during training, not necessarily the
  last iterate.
- **Feature scaling**: MLP requires features on a common scale.
  Continuous predictors should be `StandardScaler`-transformed (fit
  on train only) before MLP fit. Tree models in the battery do not
  require this; if your pipeline scales for MLP, scale only the MLP's
  copy of the features.
- **KernelExplainer timeout**: see above. The fallback rule is
  authoritative — do NOT silently emit empty interpretability outputs.

## Source provenance

Canonical source: `agent_prompts/analyst.yaml` Pilot Model Battery
row 5 + §"KernelExplainer constraints" + `data_registry/task_templates/prediction.yaml`
`model_mlp` + `model_mlp.shap_kernel_explainer_constraints`.

Per-family extraction (Decision 9). The KernelExplainer rules also
appear in `shap-explainer-selection` (cross-cutting methodology); the
two should stay in sync.
