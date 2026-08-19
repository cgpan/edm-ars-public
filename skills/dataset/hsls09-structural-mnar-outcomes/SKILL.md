---
name: hsls09-structural-mnar-outcomes
layer: dataset
description: Postsecondary HSLS:09 outcomes have structural missingness; relax the 60% retention floor and 10,000 analytic-n threshold for them.
trigger_keywords:
  - hsls
  - hsls09
  - mnar
  - structural
  - postsecondary
  - retention
  - threshold
  - thresholds
applicable_task_types: []
applicable_datasets:
  - hsls09_public
applicable_stages:
  - DataEngineer
  - Critic
priority: 2
references_skills: []
resources: []
version: "1.0"
---

# HSLS:09 Structural MNAR Outcomes

Some HSLS:09 outcome variables are only applicable to a structural
subpopulation. For example, `X4RFDGMJSTEM` (whether a student's
reference degree's first major is STEM) is undefined for students who
did not attend college. The base-year sample is ~23,500 students, but
the analytic sample for `X4RFDGMJSTEM` is conceptually capped at the
~12,000 students who attended college and reported a major.

For these outcomes, low retention is **not** a data quality problem —
the analytic sample IS the target population. The standard 60%
complete-case retention floor and the 10,000 analytic-n threshold
must be relaxed.

## When this rule applies

ALL of the following must hold:

1. The outcome variable's wave is `first_follow_up`,
   `second_follow_up`, `update_panel`, or `postsecondary_records` (any
   wave after `base_year`).
2. The registry lists `pct_missing > 30%` for the outcome variable.
3. The analytic sample after listwise deletion has `analytic_n >= 1000`.

When all three hold, do NOT set `validation_passed: false` for either
the 60% retention threshold or the `analytic_n < 10,000` threshold.

## What still applies

The `analytic_n >= 10 × n_predictors` rule (the 10p rule) **always**
applies, regardless of structural missingness. Even with a relaxed
sample-size floor, you cannot fit a 50-predictor model on 800 rows.

## Required limitation warning

Append this exact warning to `data_report.warnings` whenever the
relaxed thresholds are applied:

> Outcome [variable] has structural missingness ([pct]% in full
> dataset); analytic sample represents students with a valid
> [outcome_label] record. This is a population restriction, not random
> dropout. Findings generalize to this subpopulation only.

The Writer's `hsls09-survey-weights-limitations-paragraph` skill can
reference this in §Limitations to make the population restriction
explicit to readers.

## Examples of structural-MNAR outcomes in HSLS:09

| Outcome | Wave | pct_missing | Subpopulation |
|---|---|---|---|
| `X4HSCOMPSTAT` | update_panel | 26.24% | Students with a 2016 panel response |
| `X4EVRATNDCLG` | update_panel | 26.24% | Students with a 2016 panel response |
| `X4RFDGMJ14Y` | update_panel | 50.81% | Students with a reference degree |
| `X4RFDGMJSTEM` | update_panel | 50.81% | Students with a reference degree |
| `X4ATPRLVLA` | update_panel | 45.04% | Students who attended any postsecondary institution |
| `X5STEMCRED` | postsecondary_records | 61.57% | Students with a transcript-confirmed degree |
| `X5STEM1GPA` | postsecondary_records | 52.44% | Students with STEM coursework on transcript |

## What about the outcome `dropout_derived`?

`dropout_derived` is derived from `X4HSCOMPSTAT` labels. It also
qualifies for the relaxed thresholds because the underlying source has
structural missingness from the 2016 panel response.

## Source provenance

Canonical source: `agent_prompts/data_engineer.yaml` §"Structural vs
random outcome missingness" (L80-L93).

Variable `pct_missing` and wave assignments come from
`data_registry/datasets/hsls09_public.yaml` `variables.outcomes`.
