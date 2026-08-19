---
name: causal-regression-adjustment
layer: methodology
description: Estimate ATE/ATT via outcome regression on (treatment, adjustment-set covariates), with cluster-robust SEs, marginal standardization, and explicit G4 regression-context diagnostics (covariate overlap, Cook's D leverage, residual-treatment misspecification).
trigger_keywords:
  - causal
  - regression
  - adjustment
  - ate
  - att
  - marginal-standardization
  - statsmodels
applicable_task_types:
  - causal_soo
  - causal_itr
applicable_datasets: []
applicable_stages:
  - Analyst
priority: 1
references_skills:
  - causal-dag-identification
  - causal-estimand-definition
  - causal-balance-diagnostics
  - causal-sensitivity-unmeasured-confounding
  - hsls09-causal-conventions
resources: []
version: "1.1"
rule_severity: mandatory
---

# Causal Regression Adjustment (M1)

The simplest causal estimator under selection-on-observables: fit an
outcome model on (treatment + adjustment set), then either read the
treatment coefficient (conditional effect) or marginally standardize
to recover the ATE on the analytic sample. Cheap to run, easy to
interpret, and the natural baseline against which the more expensive
DR / forest estimators justify their existence.

M1 does **not** estimate a propensity score, so it does not compose
G3 (`causal-positivity-diagnostics`) — that is intentional per the
spec §4.1 composition completeness check. Instead, M1 invokes G4 in
**regression-context mode** (see G4 in regression-context mode below).

**Adjustment-set resolution:** apply D1's `resolve_encoded_columns` rule when constructing
the design matrix for the outcome model. Do not look up original categorical names in
`df.columns` directly — categorical adjustment variables (X1RACE, X1PAREDU, X1SEX, etc.)
have been one-hot encoded by DataEngineer into `<varname>_<level>` columns. See
`skills/dataset/hsls09-causal-conventions/SKILL.md` § Encoded-column lookup.

## Estimator definition

Outcome model: `Y ~ T + adjustment_set` (linear or logistic depending
on outcome). For the **ATE**, marginally standardize the prediction
over the analytic sample with the treatment fixed at 1 vs 0:

```
ATE_hat = mean( y_hat(T=1, X_i) - y_hat(T=0, X_i) )  for i in analytic_sample
```

## Choice rule (outcome model family)

- **Linear** for continuous Y. Report on the difference scale (risk
  difference) by default.
- **Logistic** for binary Y. Risk difference is still the default
  reporting scale; risk ratio only if requested. For binary Y on the
  difference scale, prefer `Binomial(link=identity)` (linear
  probability model) over the default logit link to keep the ATE on
  the same scale as the linear case.

## Interaction-with-treatment specification

By default, **no T × covariate interactions** (effect homogeneity
assumption). If heterogeneity is suspected, switch to M5
(`causal-forest-cate`) — do not bolt T × X interactions onto M1 ad
hoc.

## SE rule

**Cluster-robust at school level** using the pseudo-school-IDs from
D1 (`hsls09-causal-conventions`). `statsmodels` `cov_type='cluster'`
with `groups=pseudo_school_id`.

## Reporting

- ATE (95% CI, cluster-robust)
- R² for the outcome model — sanity check, **not** an effect-size
  interpretation
- Residual diagnostics if linear

## Comparator role (mandatory)

This skill is the **baseline / comparator**. Every causal study
should run M1 alongside its primary method. If M1's ATE diverges
from the primary method's ATE by **> 50%**, flag in
`results.warnings` — the divergence is not necessarily wrong, but
it deserves an explanation in §Discussion.

## G4 in regression-context mode (mandatory)

After fitting the outcome model, invoke G4's regression-context
branch (covariate overlap, Cook's distance leverage,
residuals-vs-treatment misspecification — see G4 §"Regression-context
diagnostics"). Specifically:

- **(a) Per-covariate overlap:** for each continuous covariate in
  the adjustment set, compute the overlap of treated vs. control
  distributions. **Flag any covariate with `< 80%` overlap** as an
  extrapolation risk.
- **(b) Cook's distance leverage:** compute Cook's distance from the
  fitted regression (via `OLSInfluence` for linear models,
  `GLMInfluence` for logistic). **Flag any observation with
  `Cook's D > 4/n`** as high-leverage on the ATE.
- **(c) Residual-treatment gap:** compute the mean residual gap
  across treatment arms in SDs of the residuals. **Flag if `> 0.10`**
  — the outcome model is failing to absorb the treatment-confounder
  interaction.

Append all three flags to `results.warnings`; populate
`results.balance_diagnostics` with `mode: "regression"` and the
regression-mode fields per G4's output schema. M1's composition of
G4 is interpreted in this regression-context mode, **not** the
propensity mode.

## Output schema

```json
"estimates": {
  "regression_adjustment": {
    "ate": 0.0,
    "ci_lower": 0.0,
    "ci_upper": 0.0,
    "cluster_se_method": "cluster_robust",
    "model_diagnostics": {
      "outcome_model_r2": 0.0,
      "residual_normality_p": 0.0
    }
  }
}
```

Plus `results.balance_diagnostics` populated in regression mode (per
G4's output schema).

## Failures prevented

IDF-01 (S), INF-01 (S), INF-05 (S), ESC-01 (S); regression-context
extrapolation, leverage, and outcome-model misspecification (via G4
regression-context diagnostics).

## Python implementation guidance

**Primary library:** `statsmodels` (`OLS` / `GLM` with
`cov_type='cluster'`). Clean, mature, supports clustering and
weights. No causal-specific library needed for M1.

**Function signatures:**

```python
def regression_adjustment_ate(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    covariates: list[str],
    cluster_col: str,
    family: Literal["gaussian", "binomial"] = "gaussian",
) -> dict: ...
    # returns {"ate": float, "ci_lower": float, "ci_upper": float,
    #          "model_summary": str, "outcome_model_r2": float}
```

**Library pitfalls:**

- `statsmodels` GLM interprets the link function — for risk-difference
  reporting on binary Y, use `Binomial(link=identity)` rather than the
  default logit link. With logit you get a log-odds-ratio coefficient
  that has to be transformed (and the cluster SE doesn't cleanly
  carry through).
- For the regression-context G4 diagnostics, see G4's "Library
  pitfalls" — `OLSInfluence.cooks_distance` returns `(d, p)`; the
  Cook's distances are `d[0]`. Compute on the regression that
  includes treatment + adjustment covariates so leverage is measured
  on the ATE-bearing model.

## Dtype discipline (MANDATORY — prevents F-3b21.5-M1-DTYPE-ERROR)

Two recurring M1 crashes come from handing statsmodels the wrong
pandas shapes ("'DataFrame' object has no attribute 'dtype'";
"Pandas data cast to numpy dtype of object. Check input data with
np.asarray(data)"). Follow these rules exactly:

1. **endog is a 1-D float array, never a DataFrame.** Build it as
   `y = df[outcome_col].astype(float).to_numpy()`. Do NOT use
   `df[[outcome_col]]` (double brackets produce a DataFrame).
2. **exog is an all-numeric float matrix.** Build it as
   `X = df[[treatment_col] + covariates].apply(pd.to_numeric, errors="raise").astype(float)`
   then `sm.add_constant(X, has_constant="add")`. If `pd.to_numeric`
   raises, a covariate reached M1 unencoded — fail with a message
   naming the column; do not coerce with `errors="coerce"` (silent
   NaNs).
3. **Assert before fitting:**
   `bad = X.select_dtypes(exclude="number").columns.tolist()`
   `assert not bad, f"non-numeric columns reached M1: {bad}"`.
   CSV round-trips reintroduce object dtypes; never assume the
   in-memory frame is clean because the contract checked the file.
4. **Cluster groups are 1-D:** pass
   `groups=df.loc[X.index, cluster_col].to_numpy()` — a Series or
   DataFrame here is the other classic `np.asarray(...)-object` crash,
   and `.loc[X.index]` keeps groups aligned after any row drops.
5. **Row-align everything through one dropna:** subset
   `df[[outcome_col, treatment_col, cluster_col] + covariates].dropna()`
   FIRST, then derive y / X / groups from that single frame, so the
   three arrays can never disagree on length.

## Validation criteria

The SKILL contract requires that:

1. The outcome model + marginal standardization recipe is present.
2. The no-T×covariate-interactions default is named.
3. The cluster-robust SE rule references D1's pseudo-school-IDs.
4. The comparator role (run-alongside-primary, > 50% divergence
   flag) is named.
5. The explicit G4-regression-context invocation is present, with
   the `< 80%` overlap, `> 4/n` Cook's D, and `> 0.10 SD`
   residual-treatment gap thresholds.

An Analyst code artifact using this skill must produce:

- `results.estimates.regression_adjustment` per the output schema,
- `results.balance_diagnostics` populated in regression mode,
- `results.warnings` entries for each tripped G4 regression-context
  threshold.

## Source provenance

Canonical source: `docs/v3_0_causal_skill_specification.md` §3.7
(M1 per-skill specification, including the §3a.1 R2 Path A
extension that ties M1 to G4's regression-context branch).
