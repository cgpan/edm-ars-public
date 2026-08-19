---
name: causal-did-cross-cohort
layer: methodology
description: Cross-cohort DiD (M8 raw + M9 composition-adjusted AIPW + M10 ML heterogeneity) — rank-based estimand honesty, bundled-treatment caveat, certified helper mandate, contrast-based heterogeneity inference, stability probe (NOT a pre-trend test).
trigger_keywords:
  - did
  - difference-in-differences
  - cohort
  - gap
  - panel
applicable_task_types:
  - causal_did
applicable_datasets: []
applicable_stages:
  - ProblemFormulator
  - DataEngineer
  - Analyst
  - Critic
  - Writer
priority: 1
references_skills:
  - bootstrap-confidence-intervals
resources: []
version: "1.1"
rule_severity: mandatory
---

# M8: Cross-Cohort Gap-in-Gaps DiD

## The estimand (and the two claims you may never make)

On a harmonized two-cohort panel where the outcome is the WITHIN-COHORT
percentile rank of a test score, the ONLY defensible DiD estimand is a
change in a GROUP GAP between cohorts:

```
DID_GAP_CHANGE = [mean(rank|g=1,post) - mean(rank|g=0,post)]
              - [mean(rank|g=1,pre)  - mean(rank|g=0,pre)]
```

Two claims are structurally forbidden, whatever the numbers say:

1. **Absolute achievement change** (CCD-01). The cohorts took different,
   non-equated tests; within-cohort ranks are mean-zero-gap-free by
   construction. Only the GAP between groups travels across cohorts.
2. **Single-policy attribution** (CCD-02). A multi-year cohort contrast
   bundles every policy, curricular, demographic, and compositional
   change in between. The paper names the bundle, not one policy.

## Certified implementation (helper mandate)

Analyst code MUST call the deterministic helpers — reimplementation is a
contract violation (the M1/M6 lesson: certified recipes drift when
regenerated):

```python
from analysis_helpers import did_gap_in_gaps, did_placebo_follow_wave
core = did_gap_in_gaps(df, "rank_base", "low_ses", "cohort")
probe = did_placebo_follow_wave(df, "rank_base", "rank_follow", "low_ses", "cohort")
```

`did_gap_in_gaps` computes the 2x2 estimate with a stratified bootstrap
(resampling within each group-x-cohort cell) — certified against the
synthetic DiD gate (bias 0.006). Report the full 2x2 cell means AND cell
sizes, never just the difference.

## The follow-wave probe is NOT a pre-trend test

With two cohorts there is no pre-period, so parallel trends is NOT
testable. `did_placebo_follow_wave` re-runs the gap-in-gaps on the
follow-wave ranks; divergence beyond 2 SEs flags wave-instability. Call
it a "stability probe" everywhere — Critic flags "parallel trends
verified" as critical.

## Recorded limitations (Writer must carry all)

- Non-equated tests → rank-gap-only estimand (CCD-01).
- No pre-period → assumption, not test (see probe above).
- Base-wave grade offset between the cohorts (e.g. g10 vs g9).
- Compositional change: who is low-SES (and who is tested) shifts
  between cohorts.
- School clustering not carried into the harmonized panel; the
  stratified bootstrap is student-level.

## M9: composition-adjusted gap change (v2 primary)

The raw 2x2 (M8) confounds the gap change with COMPOSITIONAL SHIFT
between cohorts (who is in the low-SES band changed). M9 standardizes
both cohorts to each group's pooled covariate distribution via AIPW with
a WITHIN-GROUP binary cohort propensity — certified: bias 0.0003, CI
coverage 1.0, and it removes exactly the built-in confounding in the
synthetic gate.

```python
m9 = analysis_helpers.did_dr_gap_change(
    df, outcome_col, group_col, post_col,
    covariate_cols=["race5", "pared3", "expect_ba", "female"],
    n_boot=200)
```

NEVER use a cross-group 4-cell propensity: SES-band membership is
near-deterministic given SES-component covariates (structural positivity
failure), and "holding fixed" a component of the group-defining
construct over-adjusts. The helper's within-group design is the
certified shape. Report `n_ps_clipped` and both `adjusted_change_*`
fields. M8 (raw) is ALWAYS reported alongside M9 — the M8-vs-M9 contrast
(composition-driven vs composition-fixed change) IS the paper's story.

## M10: ML heterogeneity of the gap change (descriptive)

```python
m10 = analysis_helpers.did_ml_heterogeneity(
    df, outcome_col, group_col, post_col,
    covariate_cols=["race5", "pared3", "expect_ba", "female"],
    subgroup_cols=["female", "expect_ba", "race5", "pared3"],
    n_boot=60)
```

Gradient-boosted per-cell outcome models; tau(x) summarized by
subgroup. **Inference is CONTRAST-based only** (`contrast_ci`,
`pairwise_difference`) — absolute per-level tau carries shared
regularization bias (found and fixed in the certification gate); the
Writer must never attach a CI claim to an absolute level. "Missing"
levels are excluded from reporting by design (nonresponse is not a
subgroup). No causal-forest asymptotics are claimed; call it
descriptive heterogeneity.

## Critic rows (walk every one)

| ID | Item | Severity | Check |
|---|---|---|---|
| `did_01` | Helpers used, not reimplemented | critical | Analyst code imports `did_gap_in_gaps` (+ `did_dr_gap_change` / `did_ml_heterogeneity` when in the method set); no hand-rolled DiD. |
| `did_02` | Full 2x2 visible | major | `cell_means` and `cell_ns` all four cells present in results. |
| `did_03` | Rank-gap-only language | critical | No absolute achievement-change claim anywhere (results, warnings, paper). |
| `did_04` | No single-policy attribution | critical | Cohort contrast described as bundled; no named-policy causal claim. |
| `did_05` | Probe labeled correctly | critical | Follow-wave probe present and called a stability probe, never a pre-trend/parallel-trends test. |
| `did_06` | CI on the headline | major | `ci_lower/upper` non-null, `se_method == "stratified_bootstrap"`. |
| `did_07` | Null honesty | critical | If the primary CI covers 0, headline leads with "no detectable change" — for M9-primary runs that means "no detectable composition-fixed change". |
| `did_08` | Subgroup heterogeneity present | major | M10 (or per-level gap-in-gaps) reported with contrast CIs. |
| `did_09` | M8 and M9 both reported | critical | v2 runs: raw AND composition-adjusted estimates present with the estimand difference stated; neither suppressed. |
| `did_10` | Positivity surface reported | major | `estimates.M9.n_ps_clipped` present; if > 5% of rows, flagged in warnings. |
| `did_11` | Heterogeneity claims are contrast-based | critical | Any M10 significance claim cites `contrast_ci`/`pairwise_difference`, never an absolute-level CI. |
