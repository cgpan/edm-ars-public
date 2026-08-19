---
name: hsls09-temporal-ordering
layer: dataset
description: HSLS:09 wave prefix mapping; predictor wave must be strictly less than outcome wave to avoid temporal leakage.
trigger_keywords:
  - hsls
  - hsls09
  - wave
  - waves
  - temporal
  - leakage
  - prefix
  - prefixes
applicable_task_types: []
applicable_datasets:
  - hsls09_public
applicable_stages:
  - ProblemFormulator
  - Critic
priority: 1
references_skills: []
resources: []
version: "1.0"
---

# HSLS:09 Temporal Ordering

HSLS:09 collects data across five waves. Variable names encode the
wave via a prefix letter + wave number. Any predictor whose wave is
**at or after** the outcome wave is temporal leakage.

## Wave map

| Wave | Year | Grade | Variable prefixes | Label |
|---|---|---|---|---|
| `base_year` | 2009 | 9 | `X1`, `S1`, `P1`, `A1`, `C1`, `M1`, `N1`, `W1` | 9th grade |
| `first_follow_up` | 2012 | 11 | `X2`, `S2`, `P2`, `A2`, `C2`, `M2`, `N2`, `W2` | 11th grade |
| `second_follow_up` | 2013 | 12 | `X3`, `S3`, `P3`, `A3`, `C3`, `M3`, `N3`, `W3` | 12th grade / update |
| `update_panel` | 2016 | — | `X4`, `S4`, `P4`, `A4`, `C4`, `M4`, `N4`, `W4` | second follow-up panel |
| `postsecondary_records` | 2017 | — | `X5`, `S5`, `P5`, `A5`, `C5`, `M5`, `N5`, `W5` | postsecondary records / financial aid |

Canonical `temporal_order` (in `registry["temporal_order"]`):

```yaml
temporal_order:
  - base_year
  - first_follow_up
  - second_follow_up
  - update_panel
  - postsecondary_records
```

## The validation rule

For every entry in `research_spec.predictor_set`:

```
index_of(predictor.wave) < index_of(outcome.wave)   # STRICT inequality
```

Same-wave predictors are leakage (e.g., predicting `X3TGPAMAT` from
`X3TCREDMAT` is forbidden — both come from the same transcript wave).

The `RegistryLoader.validate_temporal_order(predictor_wave, outcome_wave)`
helper enforces this. The PredictionTemplate's `validate_research_spec`
flags violations as warnings; the Critic's checklist `pf_02` escalates
them to critical issues.

## Common errors

- **Including X2 (11th-grade) predictors when the outcome is X3 GPA**
  is OK (X2 < X3).
- **Including X3 transcript variables when the outcome is X3TGPAMAT**
  is leakage — same-wave proxy.
- **Including X4 (2016 panel) predictors when the outcome is
  X4HSCOMPSTAT** is also same-wave leakage.
- **Including any X5 variable as a predictor for an X4 outcome**
  reverses time — predictor wave is AFTER outcome wave.

## Variables flagged with `temporal_warning` in the registry

The registry attaches a `temporal_warning` note to many later-wave
variables (anything from `first_follow_up` onward) to signal that they
should NOT be used to predict earlier-wave outcomes:

> Later-wave predictor; do not use to predict earlier-wave outcomes.

Honor these warnings during predictor selection.

## Source provenance

Canonical source: `data_registry/datasets/hsls09_public.yaml` §`waves`
+ §`temporal_order` (L7-L40).

Merged content from: `agent_prompts/problem_formulator.yaml` Validation
Rule 1 (the strict-inequality formulation and example violations).
