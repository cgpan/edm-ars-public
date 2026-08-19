---
name: prediction-model-battery
layer: task-type
description: Six-family pilot battery for prediction; composes the per-family skills via references_skills.
trigger_keywords:
  - models
  - model
  - battery
  - ensembles
  - baseline
  - prediction
applicable_task_types:
  - prediction
applicable_datasets: []
applicable_stages:
  - Analyst
priority: 1
references_skills:
  - model-logistic-regression
  - model-random-forest
  - model-xgboost
  - model-elasticnet
  - model-mlp
  - model-stacking-ensemble
resources: []
version: "1.1"
rule_severity: mandatory
---

# Prediction Model Battery

The pilot trains and evaluates **six model families**:

| # | Family | Role | SHAP explainer |
|---|---|---|---|
| 1 | Logistic / Linear Regression | Baseline (no tuning) | `LinearExplainer` |
| 2 | Random Forest | Bagged trees (tuned) | `TreeExplainer` |
| 3 | XGBoost (or LightGBM) | Boosted trees (tuned) | `TreeExplainer` |
| 4 | ElasticNet / SGDClassifier(elasticnet) | Regularized linear (tuned) | `LinearExplainer` |
| 5 | MLP | Neural net (tuned, optional via `mlp_enabled`) | `KernelExplainer` (sample_cap=1000) |
| 6 | StackingEnsemble | Stacks the tuned base models (no tuning) | **SKIP** (no SHAP for stacking) |

**SVM is intentionally excluded.** The KernelExplainer cost without a
bounded sample cap is prohibitive, and the pilot already covers SVM's
typical role with the linear baseline + ElasticNet.

This meta-skill composes the six per-family skills via
`references_skills`. Matching `prediction-model-battery` automatically
pulls in `model-logistic-regression`, `model-random-forest`,
`model-xgboost`, `model-elasticnet`, `model-mlp`, and
`model-stacking-ensemble`.

## Rules that apply to every family

1. **`random_state=42`** for every stochastic operation that supports
   the parameter.
2. **5-fold inner cross-validation on the training set only** for
   tuning (RF, XGBoost, ElasticNet, MLP). Group-aware
   (`StratifiedGroupKFold` / `GroupKFold`) when `train_school_ids.csv`
   is present. See `inner-cv-tuning-discipline`.
3. **Same primary metric** across the battery: AUC for classification,
   RMSE for regression. See `prediction-evaluation-classification`
   and `prediction-evaluation-regression`.
4. **Same held-out test set** for all final metrics. Never tune on
   test; never train on test.
5. **Same SHAP explainer mapping** per family (table above). See
   `shap-explainer-selection` for the operational details and the
   StackingEnsemble exclusion.
6. **Per-step timeout: 300s** for any single model training step;
   600s for SHAP. Failed model → log in `results.errors`, continue
   with remaining models.

## Build order

```
1. Train baseline (Logistic / Linear Regression — no tuning, no CV)
2. Tune RF       (5-fold inner CV)
3. Tune XGBoost  (5-fold inner CV)
4. Tune ElasticNet (5-fold inner CV)
5. Tune MLP      (5-fold inner CV)  ← skip if mlp_enabled: false
6. Build StackingEnsemble from the tuned base models above
```

The stack is always built last because it consumes the tuned base
estimators.

## Output

`results.all_models` contains one entry per family that was trained:

```json
{
  "all_models": {
    "LogisticRegression": {...},
    "RandomForest":       {...},
    "XGBoost":            {...},
    "ElasticNet":         {...},
    "MLP":                {...},   // omitted if mlp_enabled: false
    "StackingEnsemble":   {...}
  }
}
```

`results.best_model` is the family with the best primary metric
(highest AUC / lowest RMSE) on the held-out test set. The
`prediction-quality-gate` skill applies a minimum-performance floor
before SHAP; the `shap_model` recorded in `results.json` may differ
from `best_model` if the best model failed the gate or if MLP timed
out during KernelExplainer.


## Hyperparameter grids (authoritative — SPEC §4.3; survives even when per-model skills are cap-dropped)

| Model | Grid |
|---|---|
| LogisticRegression / LinearRegression | baseline, `C=1.0` default — no tuning |
| RandomForest | `n_estimators` ∈ {100, 300, 500}; `max_depth` ∈ {5, 10, None}; `min_samples_leaf` ∈ {1, 5, 10} |
| XGBoost | `learning_rate` ∈ {0.01, 0.05, 0.1}; `n_estimators` ∈ {100, 300, 500}; `max_depth` ∈ {3, 5, 7} |
| ElasticNet / SGDClassifier | `alpha` ∈ {0.001, 0.01, 0.1, 1.0}; `l1_ratio` ∈ {0.1, 0.5, 0.7, 0.9} |
| MLP (when `mlp_enabled`) | `hidden_layer_sizes` ∈ {(64,), (128,), (64, 32)}; `learning_rate_init` ∈ {0.001, 0.01}; `alpha` ∈ {0.0001, 0.001}; `max_iter=500, early_stopping=True, validation_fraction=0.1` |
| StackingEnsemble | no grid — meta-learner self-tunes via `RidgeCV()` / `LogisticRegressionCV()`, `cv=5`, `passthrough=False` |

All tuning via 5-fold inner CV on the training split only (see
`inner-cv-tuning-discipline`).

## Source provenance

Canonical source: `data_registry/task_templates/prediction.yaml`
§`model_training.model_battery` (L269-L361) — meta content. The
per-family details live in the six referenced skills (Decision 9
expansion).
