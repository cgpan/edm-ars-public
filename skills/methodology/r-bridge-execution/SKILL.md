---
name: r-bridge-execution
layer: methodology
description: Running R from Python — certified Rscript+JSON bridge for psychometrics (lavaan/mirt/difR); generated code calls psy_* wrappers, NEVER writes raw R.
trigger_keywords:
  - psychometric
  - irt
  - cfa
  - invariance
  - dif
  - reliability
  - lavaan
  - mirt
applicable_task_types:
  - psychometrics
applicable_datasets: []
applicable_stages:
  - DataEngineer
  - Analyst
  - Critic
priority: 1
references_skills: []
resources: []
version: "1.0"
rule_severity: mandatory
---

# Running R from Python (the certified bridge)

Psychometric estimation uses R's field-standard packages (lavaan, mirt,
MASS/difR) through a deterministic bridge — because Python's
psychometrics libraries did not meet the certification bar, and because
LLM-generated R is not certifiable.

## The one rule

**Generated code NEVER writes R code and NEVER calls `Rscript`,
`subprocess`, or `src.r_bridge` directly.** It calls the certified
Python wrappers in `analysis_helpers`:

```python
# Signatures (use EXACTLY these parameter names):
res_cfa = analysis_helpers.psy_cfa(items_df, model="F =~ it1 + it2 + it3")
res_inv = analysis_helpers.psy_invariance(items_df, group=group_series,
                                          model="F =~ it1 + it2 + it3")
res_grm = analysis_helpers.psy_grm(items_df)              # Likert GRM
res_dif = analysis_helpers.psy_dif(items_df, group=group_series)
res_ctt = analysis_helpers.psy_ctt(items_df)               # pure Python
res_om  = analysis_helpers.psy_omega(res_cfa)              # from CFA loadings
```

Each wrapper serializes the item matrix to JSON, runs the matching
certified script in `r_helpers/` via `Rscript --vanilla`, and returns a
plain dict. Hand-rolling any of these models (in R OR Python) is a
contract violation — the recipes are certification-locked
(`scripts/psychometric_gates.py`: CFA loading error .016, GRM a-error
.066, DIF hit rate 1.0 / false positives 0.0, invariance null honesty).

## Data conventions at the bridge

- Items go in as NUMERIC columns; NCES sentinels must already be NaN.
- **Items are never imputed.** lavaan uses FIML (`missing="fiml"`);
  mirt handles NA natively; the ordinal-DIF models use listwise deletion
  and report the analyzed n. lavaan drops all-missing ROWS — expect the
  returned `n` to be ≤ the input rows.
- Likert items must be positive-integer categories (1..k) for GRM/DIF;
  reverse-coded items must be recoded BEFORE the call (per the
  registry's item bank flags).

## Environment facts (Critic-checkable)

- R ≥ 4.4 resolved via config `r_bridge.rscript_path` →
  `EDM_ARS_RSCRIPT` env → common install dirs → PATH. Missing R fails
  loudly with remediation — never silently skip an analysis.
- R is NOT in the Docker sandbox image: psychometrics runs use the
  subprocess executor (`sandbox.enabled: false`).
- The bridge rejects any script path outside `r_helpers/`.
