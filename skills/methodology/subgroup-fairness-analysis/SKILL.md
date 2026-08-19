---
name: subgroup-fairness-analysis
layer: methodology
description: Per-subgroup primary-metric breakdown for every protected attribute, with a 5% gap flag.
trigger_keywords:
  - subgroup
  - fairness
  - protected
  - disparity
  - bias
applicable_task_types: []
applicable_datasets: []
applicable_stages:
  - Analyst
  - Critic
priority: 1
references_skills: []
resources: []
version: "1.0"
rule_severity: mandatory
---

# Subgroup Fairness Analysis

## Mandatory: subgroup analysis is required when subgroup_analyses is non-empty

If `research_spec.subgroup_analyses` is non-empty, `results.subgroup_performance`
MUST be populated with metrics for each protected attribute. An empty
`subgroup_performance` array when subgroup_analyses was specified is structural
incompleteness — the paper's fairness section becomes meaningless and the
Critic short-circuits to ABORT via `pcc_05`. The Phase 2c R3.5 OpenAI run
hit exactly this: `subgroup_performance: []` despite
`subgroup_analyses: ["X1SEX", "X1RACE"]`.

**Required output schema for each subgroup attribute:**

```json
"subgroup_performance": {
  "X1SEX": {
    "Male":   {"auc": 0.78, "auc_ci_lower": 0.76, "auc_ci_upper": 0.80, "n": 7800},
    "Female": {"auc": 0.81, "auc_ci_lower": 0.79, "auc_ci_upper": 0.83, "n": 7592},
    "max_gap": 0.03
  }
}
```

For regression outcomes, swap `auc` keys for `rmse`/`rmse_ci_*`. If a
group has fewer than 100 observations, document the imbalance in
`results.warnings` but still report the metric.

For each protected attribute listed in `research_spec.subgroup_analyses`,
compute the primary metric (AUC for classification, RMSE for regression)
separately for each level of that attribute. Flag any group gap larger
than 5 percentage points (absolute) as a fairness concern.

## Use the project helper, not raw column slicing

The protected attribute labels live in `test_protected.csv` — the
DataEngineer captures pre-encoding text labels (e.g., "Male"/"Female")
before any imputation or one-hot encoding. Reconstructing labels from
one-hot encoded columns in `test_X.csv` is error-prone and loses rows
where the label was imputed. Always use the helper:

```python
import analysis_helpers

subgroup_results = analysis_helpers.run_subgroup_analysis(
    model=best_model,
    test_X=test_X,
    test_y=test_y_arr,
    test_protected_path="test_protected.csv",
    subgroup_attrs=research_spec.get("subgroup_analyses", []),
    is_classification=is_classification,
    warnings_list=warnings_list,  # gap > 5% warnings appended automatically
)
```

The helper:

1. Loads `test_protected.csv` and aligns indices to `test_X` rows.
2. Iterates each attribute in `subgroup_attrs`.
3. For each level, computes the primary metric and the group sample size.
4. Appends a warning to `warnings_list` for any attribute whose
   `max(metric) - min(metric) > 0.05`.

Save the full table to `subgroup_performance.csv`.

## What to record in `results.json`

```json
{
  "subgroup_performance": {
    "<sex_variable>": {
      "Male":   {"auc": 0.83, "n": 7800},
      "Female": {"auc": 0.79, "n": 7592}
    },
    "<race_variable>": {
      "Group A": {"auc": 0.84, "n": 8120},
      "Group B": {"auc": 0.76, "n": 1830},
      "Group C": {"auc": 0.78, "n": 2410}
    }
  }
}
```

Use the original protected-attribute variable name as the outer key and
the original label values as the inner dict keys. Numeric class codes
are not informative in the paper.

## CSV format

`subgroup_performance.csv` has one row per (attribute, group) pair:

| attribute | group | primary_metric_value | n |
|---|---|---|---|
| `<sex_var>` | Male | 0.83 | 7800 |
| `<sex_var>` | Female | 0.79 | 7592 |
| `<race_var>` | Group A | 0.84 | 8120 |
| ... | ... | ... | ... |

## Verification rules (Critic)

The Critic verifies:

1. Every attribute listed in `research_spec.subgroup_analyses` appears as
   a key in `results.subgroup_performance`. Missing any → major issue.
2. For each attribute, `max(metric) - min(metric) > 0.05` is flagged in
   `results.warnings`. Missing the flag when the gap is real → major
   issue.

## Why the helper exists

Two recurring failure modes motivate the helper:

- **Reconstructing labels from one-hot encoded columns** loses rows where
  the original label was imputed (the imputer fills numeric codes that
  no longer map cleanly back to a level), and silently mis-bins minority
  groups.
- **Index misalignment** between `test_X` and `test_protected.csv` after
  a downstream agent reorders rows produces nonsense subgroups.

The helper enforces the canonical join via the saved index and refuses
to silently drop rows on mismatch.

## Causal mode (when task_type == "causal_soo")

In causal mode, subgroup analysis estimates **per-subgroup ATEs** rather than per-subgroup
predictive performance. The protected-attribute groupings (X1SEX, X1RACE, X1PAREDU
quartiles, etc.) define subgroups; for each subgroup g, fit the same causal estimator
restricted to subgroup g and report:

```python
def causal_subgroup_analysis(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    adjustment_set: list[str],
    method_id: str,  # one of "M1", "M2", "M3", "M4"; M5 has its own native CATE pathway
    protected_attribute: str,
    cluster_col: str = "school_id",
) -> dict[str, dict]:
    """
    Returns: {subgroup_value: {ate, se, ci_lower, ci_upper, n_treated, n_control, ...}}

    Cells with n_treated < 50 OR n_control < 50 are reported but flagged as
    underpowered ("low_n: True"). Cells with n_treated < 10 OR n_control < 10
    are skipped entirely with explanation.

    BH FDR correction at q=0.05 across the subgroup ATE p-values is mandatory.
    Cluster-robust SEs use the same strategy as the main estimator (cluster_bootstrap
    for M1/M2, cluster_robust for M3, cluster-aware IF for M4).
    """
    ...
```

**M5 (causal-forest-cate) does not need this skill** — M5 has native CATE estimation and
its own subgroup-effect logic in the M5 skill body. The causal-mode subgroup analysis
described here is for M1–M4.

**Failure mode this prevents (F-SUBGROUP-NONETYPE):** in 3b.5, the Analyst attempted to
call `model.predict_proba()` on a non-existent fitted classifier in causal mode and got a
NoneType error. The subgroup_performance.json was empty. The causal branch above gives
the Analyst a method-appropriate path.

## Source provenance

Canonical source: `agent_prompts/analyst.yaml` §"Subgroup Analysis"
(complete code path with `analysis_helpers.run_subgroup_analysis`).

Merged content from:
- `data_registry/task_templates/prediction.yaml` substep `interp_05`
  (workflow embedding with the 5% gap rule)
- `data_registry/evaluation_rubrics/methodological_checklist.yaml` items
  `an_06`, `an_07` (Critic verification language)

Protected attribute lists are dataset-specific; this skill consumes
whatever is in `research_spec.subgroup_analyses` and is dataset-agnostic.
Concrete attribute names are supplied by the dataset-layer skill.
