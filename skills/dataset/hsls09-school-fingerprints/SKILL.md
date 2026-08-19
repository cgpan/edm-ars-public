---
name: hsls09-school-fingerprints
layer: dataset
description: HSLS:09 SCH_ID is suppressed; recover pseudo-school-IDs by grouping students with matching school-level scale variables.
trigger_keywords:
  - hsls
  - hsls09
  - school
  - schools
  - sch_id
  - cluster
  - clusters
  - fingerprint
  - fingerprints
applicable_task_types: []
applicable_datasets:
  - hsls09_public
applicable_stages:
  - DataEngineer
priority: 1
references_skills:
  - cluster-id-reconstruction-from-fingerprints
resources: []
version: "1.0"
rule_severity: mandatory
---

# HSLS:09 School Fingerprints

The HSLS:09 public-use file suppresses school identifiers (`SCH_ID`,
`X1NCESID`, `X2NCESID`) but retains seven school-level scale variables
that are constant within each school by design. Group students by the
joint profile of those variables to recover **pseudo-school-IDs**, then
use them for school-aware splitting, group-aware inner CV, and
clustered bootstrap CIs.

The generic technique (variance check, joint groupby, quality
diagnostics, what pseudo-IDs are and are NOT good for) is in the
`cluster-id-reconstruction-from-fingerprints` methodology skill,
which this skill references.

## The seven HSLS school fingerprint variables

| Variable | Type | Role |
|---|---|---|
| `X1SCHOOLCLI` | continuous | Administrator's school-climate scale |
| `X1COUPERTEA` | continuous | Counselor's perception of teacher expectations |
| `X1COUPERCOU` | continuous | Counselor's perception of counselor expectations |
| `X1COUPERPRI` | continuous | Counselor's perception of principal expectations |
| `X1CONTROL` | binary | Public vs. Catholic/private |
| `X1LOCALE` | categorical | Urbanicity (City / Suburb / Town / Rural) |
| `X1REGION` | categorical | Geographic region (Northeast / Midwest / South / West) |

All seven are at `level: school` in the registry and are replicated
across student rows in the same school.

## Expected cluster count

HSLS:09 sampled approximately **944 schools** with ~25 students per
school (per `data_registry/datasets/hsls09_public.yaml` `levels`). Use
this as `expected_n_clusters` in the methodology skill's diagnostics.

## Operational use

```python
import analysis_helpers

# Reconstruction MUST happen BEFORE the train/test split (DataEngineer
# step 6, before step 7).
school_ids, school_meta = analysis_helpers.reconstruct_school_ids(
    df=analytic_df,  # must still contain X1SCHOOLCLI, X1COUPERTEA, etc.
    validate=True,
)
analytic_df["pseudo_school_id"] = school_ids

# Then drop the fingerprint vars from the predictor matrix.
fingerprint_cols = [
    "X1SCHOOLCLI", "X1COUPERTEA", "X1COUPERCOU", "X1COUPERPRI",
    "X1CONTROL", "X1LOCALE", "X1REGION",
]
analytic_df.drop(
    columns=[c for c in fingerprint_cols
             if c in analytic_df.columns and c not in predictor_set],
    inplace=True,
)
```

The `analysis_helpers.reconstruct_school_ids()` helper wraps the
generic recipe and returns the diagnostics dict described in the
methodology skill.

## Mandatory: Handle the zero-cluster degenerate case

The fingerprint groupby can return 0 clusters if (a) all 7 fingerprint
variables are missing for the entire analytic sample, (b) the
fingerprint values are all identical (no variance), or (c) a coding
error collapses all rows into a single nan-key bucket. The original
DataEngineer code observed in `regression/slim_problem_formulator_retry/`
produced exactly this — `n_clusters: 0` — then silently fell back to a
random split.

**Required checks immediately after groupby:**

1. **`n_clusters == 0`** → set `school_reconstruction.status =
   "failed_zero_clusters"`, set `school_reconstruction.fallback_used =
   "random_split"`. Continue with a random split, but the Critic must
   see the failure flag in the data report.
2. **`n_clusters < 50`** → set `status = "degenerate_few_clusters"`.
   School-cluster ICC and clustered bootstrap won't be meaningful with
   fewer than ~50 clusters; the result is still computable but the
   Writer must caveat it in §Limitations.
3. **`n_clusters >= 50`** → `status = "success"`. Proceed normally.

The `data_report.school_reconstruction` block is **mandatory in every
run**, even when reconstruction succeeded. Use this richer schema
(supersedes the older shape):

```json
{
  "school_reconstruction": {
    "status": "success" | "degenerate_few_clusters" | "failed_zero_clusters",
    "n_clusters": 890,
    "expected_n_schools": 944,
    "n_students_per_cluster": {
      "mean": 19.5,
      "median": 20,
      "min": 1,
      "max": 47
    },
    "fallback_used": null,
    "validation_passed": true,
    "validation_warnings": []
  }
}
```

When `fallback_used == "random_split"`, set `validation_passed: true`
(the pipeline can still produce valid metrics on a random split) and
add a warning to `data_report.warnings` describing the fallback.

The Writer's `hsls09-multilevel-limitations-paragraph` must mention
the reconstruction status in §Limitations whenever `status !=
"success"`.

## Critical rules (HSLS-specific)

1. **Run reconstruction BEFORE the train/test split.** The split
   (`school-aware-train-test-split`) requires `pseudo_school_id` to
   already exist. DataEngineer step 6 → step 7 ordering.
2. **Do NOT include `pseudo_school_id` as a predictor** in `train_X` /
   `test_X`. It is for clustered CIs and ICC computation only.
3. **Do NOT include the seven fingerprint variables as predictors**
   unless the research_spec explicitly requested one of them — they are
   structural metadata, not student-level features. Drop them from the
   predictor set after reconstruction.
4. **If validation fails** (`n_clusters` far from 944 or large
   collision rate), log a warning in `data_report.warnings` but
   continue. The pseudo-clusters are still usable for approximate
   clustering corrections even when imperfect — see the methodology
   skill's "What pseudo-IDs are NOT good for" section for what to
   avoid in that case.

## Reporting in §Limitations

The HSLS multilevel limitations paragraph
(`hsls09-multilevel-limitations-paragraph`, writing layer) handles the
required prose for the paper, including stating the number of
recovered clusters and acknowledging the mixed-effects gap.

## Source provenance

Canonical source: `agent_prompts/data_engineer.yaml` §step 6 "School
Cluster Reconstruction" (HSLS-specific variable list and the
DataEngineer step ordering).

The generic recipe (variance check, groupby, diagnostics) lives in
`cluster-id-reconstruction-from-fingerprints` per the Decision-6
split. This skill is the dataset half.
