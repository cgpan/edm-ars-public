---
name: model-stacking-ensemble
layer: task-type
description: StackingClassifier/StackingRegressor over the tuned base models; meta-learner self-tunes via RidgeCV/LogisticRegressionCV; NEVER compute SHAP for stacking.
trigger_keywords:
  - stacking
  - stack
  - ensemble
  - meta-learner
  - meta
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

# Stacking Ensemble

Built **after** all individual base models are tuned. Stacks the
tuned base estimators with a self-tuning meta-learner. Reported in
`model_comparison.csv` so reviewers see whether the stack improved
over the best individual model — but the stack is **never** used for
SHAP or any interpretability output.

## Implementation

```python
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegressionCV, RidgeCV

base_estimators = [
    ("lr", best_lr),
    ("rf", best_rf),
    ("xgb", best_xgb),
    ("enet", best_enet),
]
if mlp_enabled:
    base_estimators.append(("mlp", best_mlp))

if is_classification:
    estimator = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegressionCV(cv=5, random_state=42),
        cv=5,
        passthrough=False,
        n_jobs=-1,
    )
else:
    estimator = StackingRegressor(
        estimators=base_estimators,
        final_estimator=RidgeCV(),
        cv=5,
        passthrough=False,
        n_jobs=-1,
    )

estimator.fit(train_X, train_y)
```

## Hyperparameters

- **No grid.** The meta-learner self-tunes via cross-validation
  (`LogisticRegressionCV` for classification, `RidgeCV` for
  regression).
- `cv=5`, `passthrough=False`. `passthrough=True` would feed the
  original features into the meta-learner alongside base predictions;
  the pilot keeps `False` to make the stack interpretable as
  "weighted combination of base models."
- `random_state=42` only on the meta-learner where the parameter
  applies (`LogisticRegressionCV` accepts it; `RidgeCV` does not).

## SHAP — DO NOT compute

`shap_explainer: skip` per the SPEC. Reasons:

1. SHAP for a stacking model requires SHAP for the meta-learner, which
   only sees the base predictions (not the original features). The
   resulting SHAP values are for "contribution of LR's prediction" /
   "contribution of XGB's prediction", not for original features —
   not what readers want.
2. To get SHAP in original-feature space you would need to nest a
   model-agnostic explainer over the entire stack, which is
   prohibitively slow and doesn't add interpretive value over the
   best individual model's SHAP.

The Critic verifies this: if SHAP outputs were computed for
StackingClassifier or StackingRegressor, that's a major issue
(see methodology skill `shap-explainer-selection` for the rule and
the Critic checklist item `an_08`).

## Reporting

Include the stack in `results.all_models` and `model_comparison.csv`.
Do NOT include it in `top_features`, `feature_importance.csv`, or
SHAP figures. The "best individual model" used for SHAP excludes the
stack regardless of the stack's relative performance.

## Typical failure modes

- **Stack underperforms the best base model** when the base models
  are highly correlated (they all make the same errors). Report the
  result honestly — a stack that ties or loses is informative.
- **Long fit time** when MLP is in the base set. The 5-fold internal
  CV during stacking refits each base model 5 more times. Per-step
  timeout for the stacking fit is 300s; budget accordingly when
  `mlp_enabled: true`.

## Source provenance

Canonical source: `agent_prompts/analyst.yaml` Pilot Model Battery
row 6 + `data_registry/task_templates/prediction.yaml` `model_stacking`.
Per-family extraction (Decision 9). The "no SHAP for stacking" rule
also lives in `shap-explainer-selection`.
