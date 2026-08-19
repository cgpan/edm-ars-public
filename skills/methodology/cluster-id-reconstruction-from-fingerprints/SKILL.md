---
name: cluster-id-reconstruction-from-fingerprints
layer: methodology
description: When a dataset suppresses cluster IDs but keeps cluster-level scale variables, recover pseudo-cluster IDs by grouping rows with matching scale-variable profiles.
trigger_keywords:
  - cluster
  - clusters
  - reconstruction
  - reconstruct
  - fingerprint
  - fingerprints
  - pseudo-id
  - suppressed
applicable_task_types: []
applicable_datasets:
  - hsls09_public
applicable_stages:
  - DataEngineer
priority: 2
references_skills: []
resources: []
version: "1.0"
---

# Cluster ID Reconstruction from Fingerprints

When a public-use file suppresses the cluster identifier (school ID,
hospital ID, classroom ID, ...) but retains cluster-level scale
variables that are constant within each cluster, you can recover
**pseudo-cluster IDs** by grouping rows whose scale-variable profile
matches.

This is a generic technique. The dataset-layer skill is responsible for
naming the actual fingerprint variables; this skill describes the
recipe and the quality diagnostics.

## When the technique applies

All of the following must hold:

1. The true cluster ID is suppressed in the data file you have access
   to (it would normally be in a restricted-use release).
2. There are *K ≥ 5* scale or categorical variables that are
   **cluster-level aggregates** — same value for every row that belongs
   to the same cluster, by data-collection design.
3. You can tolerate approximate clusters: cluster boundaries are
   recovered up to ties (two genuinely distinct clusters with the same
   scale profile collapse into one pseudo-cluster).

If the third assumption fails — i.e., you need exact cluster identity —
this technique is unsuitable; use the restricted-use file or skip
cluster-aware analyses.

## Recipe

```python
import pandas as pd

# 1. Variance check. Each fingerprint variable must be ~constant
#    within whatever proxy grouping you can construct (e.g. sample row
#    of expected cluster size).
fingerprint_vars = ["<scale_1>", "<scale_2>", ...]
for v in fingerprint_vars:
    assert df[v].nunique() < df.shape[0] / expected_cluster_size, \
        f"{v} varies too much to be a cluster-level fingerprint"

# 2. Group by the joint profile.
df["pseudo_cluster_id"] = df.groupby(fingerprint_vars, dropna=False).ngroup()

# 3. Quality diagnostics.
n_clusters = df["pseudo_cluster_id"].nunique()
sizes = df.groupby("pseudo_cluster_id").size()
diagnostics = {
    "n_clusters": int(n_clusters),
    "expected_n_clusters": expected_cluster_count,
    "cluster_size_mean": float(sizes.mean()),
    "cluster_size_median": float(sizes.median()),
    "cluster_size_min": int(sizes.min()),
    "cluster_size_max": int(sizes.max()),
}
```

## Quality diagnostics — what to check

| Diagnostic | Pass criterion |
|---|---|
| `n_clusters` | Within ±10% of `expected_n_clusters`; large under-count means fingerprints have too few combinations and clusters are colliding. |
| `cluster_size_mean` | Within ±20% of (n_rows / expected_n_clusters). |
| `cluster_size_max` | No more than 3× the expected cluster size; outliers are likely collisions. |
| Singleton clusters (size 1) | Should be rare (< 5% of clusters); large singleton count means the fingerprint vars contain too much noise. |

If diagnostics fail, do NOT silently use the pseudo-IDs as if they were
correct. Either:

- Add another fingerprint variable and retry, or
- Log the failure in `data_report.warnings` and proceed without
  cluster-aware operations (no clustered CIs, no group-aware split).

## What pseudo-IDs are good for

- **Group-aware train/test split** (`school-aware-train-test-split`):
  ensures no cluster appears in both partitions.
- **Group-aware inner CV** (`inner-cv-tuning-discipline`): prevents
  within-cluster leakage during hyperparameter tuning.
- **Clustered bootstrap CIs** (`clustered-bootstrap-ci-and-icc`):
  cluster-level resampling for honest standard errors.

## What pseudo-IDs are NOT good for

- **Mixed-effects models with cluster random effects**: the
  approximation introduces measurement error in cluster identity that
  biases variance components. Use the restricted-use file instead.
- **Reporting cluster-level summaries** (e.g., school-mean GPA):
  collisions inflate the apparent variance of cluster-level statistics.

## Handling the variance-check failure mode

If a putative fingerprint variable shows excessive within-cluster
variation, it is not actually cluster-level. Drop it from the
fingerprint set and retry. Common culprits in survey panels:

- Wave-revised composite scales (the value depends on which wave's
  composite was used).
- Variables that look school-level but are actually student-level
  responses to a school question.

## Source provenance

Canonical source: `agent_prompts/data_engineer.yaml` §step 6 "School
Cluster Reconstruction" (the recipe; HSLS variable names elided here).

This skill is the generic methodology half of the Decision-6 split. The
HSLS-specific dataset skill `hsls09-school-fingerprints` supplies the
concrete fingerprint variable names and the expected cluster count
(944 schools).
