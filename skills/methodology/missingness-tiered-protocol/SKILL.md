---
name: missingness-tiered-protocol
layer: methodology
description: Tiered decision tree for handling predictor missingness, with a hard floor on complete-case retention.
trigger_keywords:
  - missingness
  - imputation
  - missing
  - mice
  - median
  - iterativeimputer
applicable_task_types: []
applicable_datasets: []
applicable_stages:
  - DataEngineer
  - Critic
priority: 1
references_skills: []
resources: []
version: "1.0"
rule_severity: mandatory
---

# Missingness Tiered Protocol

Apply this decision tree to **each predictor variable independently** after
dropping rows with missing outcomes. Compute missingness percentages on the
analytic sample (post-outcome-drop), not on the full raw dataset.

| Missingness in variable | Method |
|---|---|
| < 5% | Median imputation (continuous) or mode imputation (categorical) |
| 5% – 20% | Multiple imputation via `IterativeImputer(max_iter=5, random_state=42)` |
| > 20% | Impute with `IterativeImputer` AND add a warning to `data_report.warnings` |
| Complete-case analysis < 60% of original n | ABORT (`validation_passed: false`), unless the outcome has structural/MNAR missingness — see dataset-specific extension |

## Hard rules

1. **NEVER impute the outcome variable.** Drop rows where the outcome is
   missing FIRST, before computing predictor missingness.
2. **NEVER fit imputers, scalers, or encoders on the full dataset.** Fit on
   the training set, then transform both train and test.
3. **Categorical predictors** must be label-encoded to integers before
   `IterativeImputer` (which only operates on numeric data); do not
   coerce text labels with `pd.to_numeric(errors='coerce')` — that
   silently nulls every categorical value.
4. The complete-case retention floor (60% of original n) is a hard abort
   trigger for typical datasets. Datasets with documented structural
   missingness in the outcome (e.g., postsecondary outcomes only defined
   for college attendees) override this floor — see the dataset-layer
   skill for that extension.

## Reference implementation sketch

```python
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, SimpleImputer

# Step 1: drop missing outcomes first.
df = df.dropna(subset=[outcome_variable])

# Step 2: per-column missingness on the analytic sample.
pct_missing = df[predictor_columns].isna().mean() * 100

# Step 3: pick imputer per column.
for col in predictor_columns:
    miss = pct_missing[col]
    if miss < 5:
        imputer = SimpleImputer(strategy="median")  # or "most_frequent"
    else:
        imputer = IterativeImputer(max_iter=5, random_state=42)
    # Fit on train only; transform both partitions.
    train_X[[col]] = imputer.fit_transform(train_X[[col]])
    test_X[[col]] = imputer.transform(test_X[[col]])
    if miss > 20:
        data_report["warnings"].append(
            f"High missingness: {col} has {miss:.1f}% missing values; "
            "imputed but flagged as a limitation."
        )
```

## What to record in `data_report.json`

```json
{
  "missingness_summary": {
    "X1TXMTSC": {"pct_missing": 1.2, "imputation_method": "median"},
    "X1MTHEFF": {"pct_missing": 14.7, "imputation_method": "IterativeImputer"}
  },
  "warnings": ["High missingness: X1MTHEFF has 27.3% ..."]
}
```

The Critic verifies the recorded `imputation_method` matches the protocol
for the recorded `pct_missing` and flags any mismatch as a major issue.

## Source provenance

Canonical source: `agent_prompts/data_engineer.yaml` §"Missing Data Protocol"
(most complete prose with thresholds and abort condition).

Merged content from:
- `data_registry/task_templates/prediction.yaml` substep `dp_06` (workflow embedding of the same rules)
- `data_registry/evaluation_rubrics/methodological_checklist.yaml` item `dp_02` (verification language used by the Critic)

Dataset-layer extension (structural MNAR exception for postsecondary
outcomes) lives in the HSLS dataset skill `hsls09-structural-mnar-outcomes`
to keep this skill task- and dataset-agnostic.
