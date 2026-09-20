---
name: prediction-evaluation-classification
layer: task-type
description: Classification evaluation protocol — AUC primary, secondary metrics, bootstrap CI, ROC overlay, calibration, confusion matrix.
trigger_keywords:
  - classification
  - auc
  - roc
  - calibration
  - confusion
  - precision
  - recall
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

# Prediction Evaluation — Classification

For binary or categorical outcomes. All metrics computed on the
held-out test set only.

## Primary metric

**AUC-ROC**, via `sklearn.metrics.roc_auc_score`. Higher is better.

Confidence interval: 1000-iteration bootstrap on AUC, `random_state=42`,
through `analysis_helpers.bootstrap_ci()`. See
`bootstrap-confidence-intervals`.

## Secondary metrics (all reported per model)

Compute them with ONE certified call, never by hand:

```python
results["all_models"][name].update(
    analysis_helpers.classification_metrics(test_y_arr, y_pred))
```

It returns every metric under a name that says which averaging produced
it:

| Key | sklearn equivalent |
|---|---|
| `accuracy` | `accuracy_score` |
| `balanced_accuracy` | `balanced_accuracy_score` |
| `precision_macro` / `recall_macro` / `f1_macro` | `average='macro'` |
| `precision_positive` / `recall_positive` / `f1_positive` | `average='binary'` |
| `confusion_matrix` | `{tn, fp, fn, tp}` |
| `n_positive_true`, `n_flagged` | counts the paper quotes |

**MANDATORY RULE — never store a bare `precision`, `recall` or `f1`.**
Those keys held the *macro* values, and every reader downstream took
them for positive-class values. One delivered paper reported precision
0.576 where the positive class was 0.212, called a model that catches
10 of 545 true cases (1.8%) a "precision-favouring operating point that
flags about half of true episodes", and the word "macro" appeared
nowhere in it. Six of that paper's twelve catalogued defects trace to
this one naming choice.

`recall_macro` IS balanced accuracy. If a results.json shows `recall`
equal to `balanced_accuracy` to the last floating-point digit, the
"recall" is macro under a positive-class name, and the deterministic
check `INV_MACRO_METRIC_MISLABEL` fails the run.

When the paper discusses early-warning behaviour it must quote the
`*_positive` keys and say so; a macro figure in that sentence is wrong
by roughly the class imbalance.

Imbalanced-classification additions (when SMOTE is applied — see
`smote-imbalance-handling`):

| Metric | sklearn fn | Why |
|---|---|---|
| F2 | `fbeta_score(y_true, y_pred, beta=2)` | Emphasizes recall — the right metric for early-warning systems |
| Balanced accuracy | `balanced_accuracy_score` | Fairer than plain accuracy under imbalance |
| Per-class precision/recall | `precision_score(average=None)` / `recall_score(average=None)` | Reports both classes explicitly |

Note: even with SMOTE, **accuracy alone is misleading** for imbalanced
data. Report it but emphasize AUC + F2 + balanced accuracy in the
discussion.

## Required figures

| File | Description |
|---|---|
| `roc_curves.png` | ROC curves for ALL models overlaid on one plot |
| `calibration_curve.png` | Calibration curve for the best model |
| `confusion_matrix.png` | Confusion matrix heatmap for the best model |

Use `matplotlib.use('Agg')` to avoid display issues in subprocess
execution. Save at 150 dpi minimum.

## Best-model selection

`results.best_model` = the family with the highest test-set AUC.
Ties are broken by simpler model first (LR > ElasticNet > RF > XGBoost
> MLP).

StackingEnsemble appears in `model_comparison.csv` but is excluded
from the SHAP/interpretability "best individual model" — see
`shap-explainer-selection` for that exclusion.

## Suspicious AUC flag

`AUC > 0.95` is automatically flagged in `results.warnings` as
suspicious for potential leakage. The Critic investigates which
features dominate SHAP values; if a top feature is a strong proxy
for the outcome (e.g. a transcript-based variable predicting a
transcript-based GPA), it escalates to a critical issue.

## What goes in `results.json`

```json
{
  "best_model": "XGBoost",
  "best_metric_value": 0.823,
  "primary_metric": "AUC",
  "all_models": {
    "LogisticRegression": {
      "auc": 0.78, "accuracy": 0.71, "precision": 0.69,
      "recall": 0.66, "f1": 0.67,
      "auc_ci_lower": 0.76, "auc_ci_upper": 0.80
    },
    "XGBoost": {
      "auc": 0.823, "accuracy": 0.76, "precision": 0.74,
      "recall": 0.71, "f1": 0.72,
      "auc_ci_lower": 0.798, "auc_ci_upper": 0.847
    }
  },
  "figures_generated": [
    "roc_curves.png", "calibration_curve.png", "confusion_matrix.png",
    "shap_summary.png", "shap_importance.png", "pdp_X1TXMTSC.png", ...
  ]
}
```

## Source provenance

Canonical source: `data_registry/task_templates/prediction.yaml`
§`evaluation.metrics.classification` (L441-L468) + `agent_prompts/analyst.yaml`
§"Evaluation Protocol — Classification".

The bootstrap CI procedure is the cross-cutting methodology skill
`bootstrap-confidence-intervals`. The SMOTE-conditional metrics are
documented operationally in `smote-imbalance-handling`.
