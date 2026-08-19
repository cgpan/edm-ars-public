---
name: smote-imbalance-handling
layer: task-type
description: SMOTE applied to training set only; optional ablation Phase A (no SMOTE) vs Phase B (SMOTE) for binary classification.
trigger_keywords:
  - smote
  - imbalance
  - imbalanced
  - oversampling
  - minority
  - ablation
applicable_task_types:
  - prediction
applicable_datasets: []
applicable_stages:
  - Analyst
priority: 2
references_skills: []
resources: []
version: "1.0"
rule_severity: mandatory
---

# SMOTE Imbalance Handling (Classification Only)

When the binary outcome's minority class proportion drops below the
configured threshold (default 20%), apply SMOTE to the training set.
Optionally run an ablation that compares pre-SMOTE vs post-SMOTE
results so the paper can quantify SMOTE's effect.

## When to apply

Read `data_report.is_imbalanced` (set by the DataEngineer when
`minority_pct < minority_threshold`), or compute it inline:

```python
minority_pct = min(np.bincount(y_train)) / len(y_train)
is_imbalanced = minority_pct < minority_threshold  # default 0.20
```

## Phase A — Ablation baseline (NO SMOTE)

When `ablation_enabled: true` AND imbalance is detected:

1. Train ALL models on the **original** training data (`train_X`,
   `train_y` as-is).
2. Evaluate ALL models on the **original test set** (`test_X`,
   `test_y` — never touched).
3. Record results in `ablation.all_models` (same schema as
   primary `all_models`).

## Phase B — SMOTE-augmented training (primary results)

```python
import analysis_helpers

X_train_smote, y_train_smote, smote_meta = analysis_helpers.apply_smote(
    train_X, train_y,
    minority_threshold=minority_threshold,  # config default 0.20
    random_state=42,
    k_neighbors=smote_k_neighbors,          # config default 5
)
```

1. Re-train ALL models on the SMOTE-augmented training data.
2. Evaluate ALL models on the **same original test set** (NEVER SMOTE
   the test set).
3. These are the primary `all_models` results in `results.json`.

## When ablation is disabled

When `ablation_enabled: false` and imbalance is detected: skip Phase A;
just apply SMOTE and train once.

## When NOT imbalanced

Train on the original data as usual. Set `results["ablation"] = None`
explicitly so downstream consumers know the absence is intentional.

## Critical rules

1. **NEVER apply SMOTE to the test set** — test set must remain the
   original distribution for the metrics to reflect deployment
   conditions.
2. **NEVER apply SMOTE before the train/test split** — the
   DataEngineer split happens before the Analyst sees the data.
3. **Use `random_state=42` for SMOTE** so the synthetic minority
   examples are reproducible.
4. **Use `analysis_helpers.apply_smote()`**, never `imblearn.SMOTE`
   directly. The helper handles the imbalance threshold check, the
   k-neighbor fallback when the minority class is too small for the
   default `k=5`, and the metadata return.
5. **Ablation and primary evaluations use the same original test
   set** — the only thing that changes between Phase A and Phase B is
   the training data.

## Imbalanced metrics (apply when SMOTE is used)

When SMOTE is applied, ADD the following fields to BOTH primary and
ablation `all_models` entries:

- `f2` (`fbeta_score(y_true, y_pred, beta=2)`) — emphasizes recall.
- `balanced_accuracy` (`balanced_accuracy_score(y_true, y_pred)`).

Accuracy alone is misleading under imbalance. Continue to report it,
but emphasize AUC + F2 + balanced accuracy in the discussion.

See `prediction-evaluation-classification` for the full metric set.

## What goes in `results.json`

```json
{
  "ablation": {
    "description": "Pre-SMOTE baseline for comparison",
    "all_models": {
      "LogisticRegression": {
        "auc": 0.72, "accuracy": 0.85, "precision": 0.40,
        "recall": 0.18, "f1": 0.25, "f2": 0.20,
        "balanced_accuracy": 0.55,
        "auc_ci_lower": 0.69, "auc_ci_upper": 0.74
      }
    },
    "smote_applied": true,
    "minority_class_pct_before": 0.12,
    "minority_class_pct_after": 0.50,
    "smote_n_train_before": 15000,
    "smote_n_train_after": 22000
  }
}
```

When no imbalance is detected: `"ablation": null`.

## Verification rules (Critic)

The Critic verifies:

1. If `data_report.is_imbalanced == true` AND classification AND
   `ablation_enabled: true` → `results.ablation` MUST be present
   (non-null). Missing → major.
2. Ablation `n_test` must match primary `n_test`. Mismatch means
   SMOTE was applied to the test set — **critical** issue.
3. When SMOTE was applied, F2 and balanced_accuracy must be reported
   for every model in primary and ablation results. Missing → major.

## Source provenance

Canonical source: `agent_prompts/analyst.yaml` §"Class Imbalance
Handling" (L194-L261).

The metric formulas and the "accuracy is misleading under imbalance"
narrative also live in `prediction-evaluation-classification`.
