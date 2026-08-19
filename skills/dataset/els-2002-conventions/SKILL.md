---
name: els-2002-conventions
layer: dataset
description: ELS:2002 conventions — numeric-coded CSV (NOT labeled like HSLS), negative sentinels with the continuous-composite exception, BY/F1/F2/F3 waves, tier-3 weight/ID exclusions.
trigger_keywords:
  - els
  - els2002
  - cohort
  - sentinel
applicable_task_types: []
applicable_datasets:
  - els_2002
applicable_stages:
  - ProblemFormulator
  - DataEngineer
  - Analyst
  - Critic
  - Writer
priority: 1
references_skills: []
resources: []
version: "1.0"
rule_severity: mandatory
---

# ELS:2002 Conventions

## The one thing to get right (opposite of HSLS)

This CSV stores **numeric codes**, not labeled strings. `pd.to_numeric`
is safe everywhere. But NEGATIVE INTEGERS ARE MISSING-DATA SENTINELS
and must be mapped to `NaN` BEFORE any imputation, encoding, or
statistics:

- Integer-coded variables: every negative value (−1 don't know, −2
  refused, −3 legitimate skip, −4 nonrespondent, −8 component missing,
  −9 missing) is a sentinel.
- **Continuous composites (BYSES1, BYMATHSE) have VALID negative scale
  values** (BYSES1 range −2.11..1.82). For these, ONLY exact codes
  ≤ −3 are sentinels. Blanket `v < 0 → NaN` silently deletes the lower
  half of the SES distribution (registry pitfall
  `continuous_composite_sentinels`, severity critical).

## Waves and temporal ordering

`BY` (2002, grade 10) → `F1` (2004, grade 12) → `F2` (2006) → `F3`
(2012). Prefix-based wave inference: a predictor's prefix must come
strictly before the outcome's prefix in that order. BY predictors for
F1/F2/F3 outcomes are safe; F1 predictors for F2/F3 outcomes are safe
with the usual later-wave caveats.

## Tier-3 exclusions

Never model: `STU_ID`, `SCH_ID`, `STRAT_ID`, `PSU`, `F1SCH_ID`,
`F*UNIV*` universe flags, any `^BYSTUWT`/`W[0-9]`/`BRR`/`*WT` weight,
and `*IM` imputation-flag twins (e.g. `BYSEXIM` next to `BYSEX`).

## Encoding

One-hot categorical CODES with meaningful names: map codes to the
registry's `codebook_codes` labels when constructing dummy names
(`BYRACE_White` not `BYRACE_7`) so downstream SHAP/subgroup outputs
are readable. Continuous variables pass through as single numeric
columns per D1.

## School clustering on ELS (F1SCH_ID — use directly, no reconstruction)

Base-year `SCH_ID` is 100% suppressed (−5), but **`F1SCH_ID` carries 752
real school IDs covering ~76% of students**. Use it as the cluster ID for
the school-aware split, clustered bootstrap CIs, and ICC — NEVER as a
predictor (tier-3). Do NOT run fingerprint reconstruction on ELS.

```python
sch = pd.to_numeric(df["F1SCH_ID"], errors="coerce")
school_ids = sch.where(sch >= 0)
# students with no F1 school (~24%): unique singleton clusters
school_ids = school_ids.fillna(-(pd.Series(range(len(df)), index=df.index) + 1))
```

Report in the paper: clustering is PARTIAL (F1 school, one wave after
baseline; ~24% singletons) — better than ignoring the design, weaker than
true base-year clustering. This replaces the old blanket "no multilevel
structure available" limitation.

## HSLS analogues (for cross-cohort work)

| ELS | HSLS | Construct |
|---|---|---|
| BYTXMSTD | X1TXMTSCOR | baseline std math score |
| BYSES1 | X1SES | SES composite |
| BYMATHSE | X1MTHEFF | math self-efficacy |
| BYSEX / BYRACE | X1SEX / X1RACE | protected attributes |
| F2EVRATT | X4EVRATNDCLG | ever attended postsecondary |

## Source provenance

Authored in Arc G Phase A (2026-07-03) from the EDAT export codebook +
sentinel-aware profiling of the delivered CSV.
