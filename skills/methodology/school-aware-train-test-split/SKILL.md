---
name: school-aware-train-test-split
layer: methodology
description: Group-aware train/test split that prevents cluster leakage when students are nested in schools (or analogous clustered units).
trigger_keywords:
  - split
  - train-test
  - groupkfold
  - stratifiedgroupkfold
  - cluster
  - nested
applicable_task_types: []
applicable_datasets:
  - hsls09_public
  - els_2002
applicable_stages:
  - DataEngineer
priority: 1
references_skills: []
resources: []
version: "1.0"
rule_severity: mandatory
---

# School-Aware Train/Test Split

When students are nested in schools (or any analogous clustered design),
a naive `train_test_split` lets the same school appear in both partitions.
School-level features then leak from train to test and inflate every
metric. Use a group-aware split instead.

## Pick the splitter by task type

- **Classification** — `StratifiedGroupKFold(n_splits=5, shuffle=True,
  random_state=42)`. Take fold 0 as the test set. Stratification preserves
  the class ratio approximately while keeping every school inside one fold.
- **Regression** — `GroupShuffleSplit(n_splits=1, test_size=0.2,
  random_state=42)`.

Always wrap the call in `analysis_helpers.grouped_train_test_split()`
rather than calling sklearn splitters directly. The helper enforces the
group constraint, returns a `split_meta` dict for `data_report.json`, and
falls back gracefully when only one fold is feasible.

```python
import analysis_helpers

train_idx, test_idx, split_meta = analysis_helpers.grouped_train_test_split(
    df=analytic_df,
    y=analytic_df[outcome_variable],
    groups=analytic_df["pseudo_school_id"],
    test_size=0.2,
    stratify=(outcome_type in ["binary", "categorical"]),
    random_state=42,
)

train_df = analytic_df.iloc[train_idx]
test_df = analytic_df.iloc[test_idx]
```

## Persist group IDs for downstream stages

Save the cluster ID for each partition so the Analyst can use the same
groups for inner-CV and clustered bootstrap CIs:

```python
analytic_df.iloc[train_idx]["pseudo_school_id"].to_csv(
    "train_school_ids.csv", index=False, header=["pseudo_school_id"]
)
analytic_df.iloc[test_idx]["pseudo_school_id"].to_csv(
    "test_school_ids.csv", index=False, header=["pseudo_school_id"]
)
```

## What to record in `data_report.json`

Add `split_meta` under key `"split_info"`:

```json
{
  "split_info": {
    "split_method": "StratifiedGroupKFold",
    "group_aware": true,
    "n_groups_train": 712,
    "n_groups_test": 178,
    "group_overlap": 0
  }
}
```

`group_overlap` MUST be `0`. Any non-zero value means a school appears in
both partitions and the split is invalid.

Also append the partial-multilevel acknowledgement to
`data_report.warnings`:

> Multilevel structure (students nested in schools) is partially addressed:
> train/test split is school-aware (no school in both sets) and clustered
> bootstrap CIs account for within-school correlation. However, models do
> not include school-level random effects.

## Mandatory: pd.qcut requires `duplicates='drop'`

If you bin a continuous outcome to enable stratification (e.g., for
`StratifiedGroupKFold` on a regression target), `pd.qcut` raises
`ValueError: Bin edges must be unique` whenever the outcome
distribution has ties at quantile boundaries. This is common with
discrete or near-discrete outcomes (Likert scales, GPAs, integer
counts) and with small samples, and the original DataEngineer code
observed in `regression/slim_problem_formulator_retry/` crashed on
exactly this. Always pass `duplicates='drop'`:

```python
# WRONG -- crashes on tied bin edges:
strat_bins = pd.qcut(y, q=10)

# RIGHT -- drops duplicate edges, may produce fewer than q bins:
strat_bins = pd.qcut(y, q=10, duplicates='drop')

# After binning, check that you got enough bins for stratification:
if strat_bins.cat.categories.size < 5:
    # Too few bins to stratify usefully -- fall back to a non-stratified
    # group split (GroupShuffleSplit) rather than failing silently.
    train_idx, test_idx, split_meta = analysis_helpers.grouped_train_test_split(
        df=analytic_df,
        y=analytic_df[outcome_variable],
        groups=analytic_df["pseudo_school_id"],
        test_size=0.2,
        stratify=False,           # explicit fall-back
        random_state=42,
    )
    split_meta["stratification_fallback_reason"] = (
        f"qcut produced only {strat_bins.cat.categories.size} bins; "
        "stratification skipped"
    )
```

This rule is mandatory: a `pd.qcut` crash aborts the whole pipeline.

## Verification rules (Critic)

The Critic verifies `data_report.split_info.group_overlap == 0` and that
the multilevel-limitation warning is present. Failure on either is a
major issue.

## Source provenance

Canonical source: `agent_prompts/data_engineer.yaml` §"Step 7"
(complete code path with `analysis_helpers.grouped_train_test_split` and
the group-ID persistence).

Merged content from:
- `data_registry/task_templates/prediction.yaml` substep `dp_09` and
  check `check_dp_grouped_split` (workflow + verification language)

This skill is generic across nested-survey datasets; the cluster-ID
source (`pseudo_school_id`) is dataset-specific and is set up by a
dataset-layer skill (`hsls09-school-cluster-reconstruction` for HSLS).
