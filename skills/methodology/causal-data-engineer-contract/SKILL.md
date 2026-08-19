---
name: causal-data-engineer-contract
layer: methodology
description: In causal_soo, the DataEngineer must carve treatment + adjustment_set + outcome into the analytic CSV. Dropping the treatment column silently substitutes a proxy variable downstream and invalidates the estimand (F-3b11-DE-MISSING-TREATMENT-COLUMN).
trigger_keywords:
  - causal
  - treatment
  - adjustment
  - carve-out
  - data engineer
  - data-engineering
  - analytic
  - contract
applicable_task_types:
  - causal_soo
  - causal_itr
applicable_datasets: []
applicable_stages:
  - DataEngineer
  - Analyst
priority: 1
references_skills:
  - causal-estimand-definition
  - hsls09-causal-conventions
resources: []
version: "1.0"
rule_severity: mandatory
---

# Causal data-engineer contract

In `task_type=causal_soo`, the DataEngineer constructs an analytic CSV
(`train_X.csv` plus matching test/y/cluster files) that downstream agents
(Analyst, Critic) read. The contract for what the analytic CSV must
contain in causal mode is **different** from the prediction-task contract.

## What the analytic CSV must contain (mandatory)

In causal mode, `train_X.csv` MUST contain — at minimum — three column
classes drawn from the research_spec:

1. **The treatment column.** The variable named in
   `research_spec.treatment.variable` (e.g., `X1MTHEFF`). This is NOT
   in `research_spec.adjustment_set` — PF correctly excludes treatment
   from the adjustment set per causal DAG identification (treatment is
   the exposure to be estimated, not a covariate to adjust on). The
   DataEngineer MUST NOT interpret this exclusion as "drop the column."
   The treatment column must be present in the analytic CSV under its
   original variable name (or, if pre-binarized per
   `research_spec.treatment.operationalization`, under the operationalized
   name).

   **NEVER one-hot-encode the treatment column** (3b.23.7 attempt-2
   failure shape: `X1MTHEFF_binary_0` + `X1MTHEFF_binary_1` dummy pair
   instead of the single column). The treatment is the exposure, not a
   categorical covariate: it must be exactly ONE column of 0/1 integer
   values named `<var>` or `<var>_binary`. Exclude the treatment (and
   the derived `<var>_binary`) from any `pd.get_dummies` /
   OneHotEncoder call — encode categoricals from the adjustment set
   ONLY. The orchestrator's pre-flight deterministically collapses a
   complementary dummy pair back to one column and logs a warning, but
   relying on that repair still costs a violation warning in
   `data_report.warnings`.

2. **The full adjustment set.** Every variable in
   `research_spec.adjustment_set`, in encoded form per D1's
   conventions (one-hot for categorical, raw for continuous; see D1
   for HSLS:09 specifics). The Analyst applies `resolve_encoded_columns`
   to look these up at analysis time.

3. **The outcome column.** The variable named in
   `research_spec.outcome.variable`. Goes in `train_y.csv` /
   `test_y.csv` per the existing convention, not in `train_X.csv`.

## Cluster ID columns (if applicable)

When the dataset declares cluster IDs (e.g., HSLS:09's reconstructed
school IDs from `cluster-id-reconstruction-from-fingerprints`), the
cluster ID column MUST be present in a sibling file (e.g.,
`train_school_ids.csv`) so the Analyst can pass `groups=` to GroupKFold
for cross-fitting and can compute cluster-aware standard errors.

## What the analytic CSV must NOT contain

- Predictor-task scaffolding columns (e.g., `model_fold`, `fold_id`)
  unless the spec explicitly requests them. Causal mode does not use
  these.
- Auxiliary variables NOT in `adjustment_set` and NOT the treatment.
  Causal estimation operates on a deliberately-curated covariate set;
  including extra columns invites the Analyst to either ignore them
  (wasting prompt budget) or accidentally include them in the propensity
  / outcome model (violating the curated identification strategy).

## Carve-out rule (the prescriptive form for the DataEngineer)

```python
def causal_soo_carve_out(research_spec, source_dataframe):
    """Return train_X, test_X, train_y, test_y, train_clusters, test_clusters."""
    treatment_col = research_spec["treatment"]["variable"]
    outcome_col = research_spec["outcome"]["variable"]
    adjustment_set = research_spec["adjustment_set"]
    cluster_col = research_spec.get("cluster_id_column")  # may be None

    # Build the analytic feature set: treatment + adjustment_set
    feature_cols = [treatment_col] + list(adjustment_set)

    # If treatment is operationalized (e.g., median_split_binary),
    # apply the operationalization here.
    operationalization = research_spec["treatment"].get("operationalization")
    if operationalization == "median_split_binary":
        median = source_dataframe[treatment_col].median()
        source_dataframe[treatment_col + "_binary"] = (
            source_dataframe[treatment_col] >= median
        ).astype(int)
        feature_cols = [treatment_col + "_binary"] + list(adjustment_set)

    # Standard train/test split (cluster-aware where applicable)
    # ... split logic ...

    train_X = train[feature_cols]
    test_X = test[feature_cols]
    train_y = train[outcome_col]
    test_y = test[outcome_col]
    train_clusters = train[cluster_col] if cluster_col else None
    test_clusters = test[cluster_col] if cluster_col else None

    return train_X, test_X, train_y, test_y, train_clusters, test_clusters
```

The DataEngineer MUST follow this carve-out shape. The treatment column
must appear in `train_X.csv` first (or at any consistent position) and
its name must match `research_spec.treatment.variable` (or the
operationalized form if applicable). Encoding categorical adjustment-set
variables is per D1.

## Failure mode this prevents (F-3b11-DE-MISSING-TREATMENT-COLUMN)

If the DataEngineer carves out only `adjustment_set + outcome`, the
treatment column is silently dropped. The Analyst cannot recover it via
`resolve_encoded_columns` (D1) because the column was never written to
`train_X.csv`. The Analyst — applying its own column-resolution
discipline — will substitute the closest available proxy variable.
The proxy is a different construct. All downstream causal estimates
are computed against the wrong exposure. LSAR will identify the
construct-validity failure as a fatal Methodological Rigor / Empirical
Support issue.

## Verification (Orchestrator-side guardrail)

The Orchestrator asserts treatment column presence in `train_X.csv`
after the DataEngineer stage and before the Analyst stage runs. A
violation raises `CausalDataContractError` with the message:

> "DataEngineer produced train_X.csv missing the treatment column
>  '<name>' declared in research_spec.treatment.variable. Causal mode
>  requires the treatment column to be carved out alongside the
>  adjustment_set; see causal-data-engineer-contract skill."

The pipeline halts at this point — the Analyst is not invoked.
This guardrail is mandatory under causal_soo.
