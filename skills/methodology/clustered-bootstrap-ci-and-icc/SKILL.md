---
name: clustered-bootstrap-ci-and-icc
layer: methodology
description: Compute the intraclass correlation for the outcome and report cluster-level bootstrap CIs alongside standard bootstrap CIs.
trigger_keywords:
  - icc
  - cluster
  - clustered
  - bootstrap
  - multilevel
  - intraclass
applicable_task_types: []
applicable_datasets:
  - hsls09_public
  - els_2002
applicable_stages:
  - Analyst
priority: 2
references_skills:
  - bootstrap-confidence-intervals
resources: []
version: "1.1"
rule_severity: mandatory
---

# Clustered Bootstrap CIs and ICC

When the data have a clustered structure (e.g., students within schools),
standard row-level bootstrap CIs underestimate uncertainty for predictors
that vary primarily at the cluster level. Report **both** standard and
clustered CIs so the paper can discuss the difference.

This skill assumes a `pseudo_school_id` (or analogous cluster ID) exists
for both train and test partitions. The DataEngineer-side skill
`school-aware-train-test-split` is the upstream dependency.

## a) ICC for the outcome variable

Compute the intraclass correlation on the test set using the cluster IDs:

```python
import analysis_helpers
import pandas as pd

test_school_ids = pd.read_csv("test_school_ids.csv")["pseudo_school_id"].values
icc_result = analysis_helpers.compute_icc(test_y_arr, test_school_ids)
results["icc"] = icc_result
```

Record the result as:

```json
{
  "icc": {
    "icc": 0.12,
    "interpretation": "small",
    "n_clusters": 890,
    "avg_cluster_size": 3.9
  }
}
```

Interpretation thresholds:

| ICC | Interpretation | Implication |
|---|---|---|
| < 0.05 | Negligible | Standard (unclustered) CIs are approximately valid; clustering can be ignored for the primary metric. |
| 0.05 – 0.10 | Small | Clustered CIs matter; report the difference. |
| 0.10 – 0.20 | Moderate | Clustered CIs strictly required. |
| > 0.20 | Large | Standard CIs likely understate uncertainty meaningfully. |

## b) Clustered bootstrap CI for the primary metric

Replace the standard `bootstrap_ci` call for the **best model** with the
clustered version:

```python
ci_lower, ci_upper = analysis_helpers.clustered_bootstrap_ci(
    y_true=test_y_arr,
    y_pred=best_model_preds,
    cluster_ids=test_school_ids,
    metric_fn=primary_metric_fn,
)
```

Cluster-level resampling samples whole clusters with replacement, then
collects every row in the sampled clusters. This preserves within-cluster
correlation and produces wider CIs than naive row-level bootstrap when
ICC > 0.

## What to record in `results.json`

Report **both** CIs so the comparison is auditable:

```json
{
  "best_model_ci_standard": [0.842, 0.870],
  "best_model_ci_clustered": [0.835, 0.878],
  "ci_widened_by_clustering": true
}
```

`ci_widened_by_clustering` is `true` if the clustered interval is wider
than the standard interval.

## c) Missing-cluster-ID fallback

If `train_school_ids.csv` or `test_school_ids.csv` is absent (e.g., the
DataEngineer step that produces them was skipped), fall back to standard
unclustered bootstrap CIs and append to `results.warnings`:

> School cluster IDs not available. Using standard (unclustered) bootstrap
> CIs. Standard errors may be underestimated for school-level predictors.

## Source provenance

Canonical source: `agent_prompts/analyst.yaml` §"Clustered Standard
Errors and ICC".

Merged content from: none — this section is single-sourced. The
upstream cluster-ID generation lives in `school-aware-train-test-split`
(methodology) and `hsls09-school-cluster-reconstruction` (dataset).
