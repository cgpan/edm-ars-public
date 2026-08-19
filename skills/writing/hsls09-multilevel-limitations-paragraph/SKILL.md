---
name: hsls09-multilevel-limitations-paragraph
layer: writing
description: Required HSLS:09 multilevel-structure limitations paragraph covering reconstruction, ICC, and clustered CIs.
trigger_keywords:
  - multilevel
  - hsls
  - limitations
  - school
  - icc
  - cluster
applicable_task_types: []
applicable_datasets:
  - hsls09_public
applicable_stages:
  - Writer
priority: 1
references_skills: []
resources: []
version: "1.1"
rule_severity: mandatory
---

# HSLS:09 Multilevel Structure — Limitations Paragraph

The HSLS:09 public-use file has a known multilevel constraint that must
be acknowledged in §Limitations with precise, authoritative language.
Vague hedges ("future work could consider clustering") are not
acceptable — the reviewer needs to see the tradeoff is understood.

## Required content (write a paragraph that includes ALL of these)

1. HSLS:09 sampled approximately 25 students within each of 944 schools,
   creating a nested structure.
2. School identifiers (`SCH_ID`) are suppressed in the public-use file,
   but school-level variables (`X1SCHOOLCLI`, `X1COUPERTEA`,
   `X1CONTROL`, `X1LOCALE`, etc.) are school-level aggregates identical
   for all students within the same school.
3. We reconstructed pseudo-school clusters by grouping students with
   matching school-level variable profiles, recovering approximately N
   clusters (cite `data_report.school_reconstruction.n_clusters`).
4. We computed the ICC for the outcome variable (cite `results.icc`).
   State the ICC value and its interpretation
   (negligible / small / moderate / large).
5. We used cluster-level bootstrap resampling to compute confidence
   intervals that account for within-school correlation (cite
   `results.best_model_ci_clustered`).
6. **REMAINING LIMITATION**: this approach provides clustered CIs for
   the primary metric but does not estimate school-level random effects.
   A full mixed-effects model would require either the restricted-use
   file (with true `SCH_ID`) or adaptation of the scikit-learn pipeline
   to use `statsmodels` `MixedLM`, which is beyond the scope of this
   automated system's current capability.
7. **If ICC < 0.05**: state that the negligible ICC suggests ignoring the
   nested structure has minimal impact on the current results.
8. **If ICC ≥ 0.05**: state that the non-negligible ICC makes the
   clustered CIs important, and that the standard (unclustered) CIs
   reported for individual models may underestimate uncertainty.

## Fallback when reconstruction is missing

If `data_report.school_reconstruction` or `results.icc` is missing
(the DataEngineer step that produces school clusters did not run), fall
back to the generic multilevel limitation statement that
`data_report.warnings` already contains and note that cluster
reconstruction was not available for this run.

## Methods/Data subsection — short companion paragraph

In `%%PLACEHOLDER:METHODS_DATA%%`, include this short paragraph (only
when reconstruction succeeded):

> Because school identifiers are suppressed in the HSLS:09 public-use
> file, we reconstructed school clusters by grouping students with
> matching school-level variable profiles (school climate scale,
> counselor perception scales, school control, locale, and region).
> This yielded [N] pseudo-school clusters (expected: 944 based on the
> HSLS:09 sampling frame), with a mean cluster size of [X] students.
> The intraclass correlation for [outcome] was [ICC value]
> ([interpretation]), indicating [degree] of between-school variance.
> Confidence intervals for the primary metric were computed using
> cluster-level bootstrap resampling to account for within-school
> correlation.

## Source provenance

Canonical source: `agent_prompts/writer.yaml` §"Multilevel Structure —
What We Did and What Remains" + §"Methods/Data — School Cluster
Reconstruction Paragraph".

Merged content from: none — single-sourced.
