---
name: causal-inverse-probability-weighting
layer: methodology
description: Estimate ATE via stabilized IPW with weight trimming, cluster-robust SEs from a weighted regression on treatment, and an explicit weighted-balance check that invokes G4 with weights=stabilized_weights.
trigger_keywords:
  - causal
  - ipw
  - inverse-probability
  - weighting
  - stabilized
  - propensity
  - ess
applicable_task_types:
  - causal_soo
applicable_datasets: []
applicable_stages:
  - Analyst
priority: 1
references_skills:
  - causal-dag-identification
  - causal-estimand-definition
  - causal-positivity-diagnostics
  - causal-balance-diagnostics
  - causal-sensitivity-unmeasured-confounding
  - hsls09-causal-conventions
resources: []
version: "1.0"
rule_severity: mandatory
---

# Causal Inverse-Probability Weighting (M3)

Estimate the ATE by reweighting each row by the inverse of its
propensity to receive its observed treatment. Stabilized weights
shrink variance vs. unstabilized; trimming bounds the worst weights
per G3's positivity rule; cluster-robust SEs come from a weighted
regression on treatment with school-level clustering.

The **weighted balance check** in this skill is non-negotiable —
M3 composes G4 implicitly via `references_skills`, but the body
below names the check explicitly so the LLM cannot drop it.

**Adjustment-set resolution:** apply D1's `resolve_encoded_columns` rule when constructing
the design matrix for the propensity model. Do not look up original categorical names in
`df.columns` directly — categorical adjustment variables (X1RACE, X1PAREDU, X1SEX, etc.)
have been one-hot encoded by DataEngineer into `<varname>_<level>` columns. See
`skills/dataset/hsls09-causal-conventions/SKILL.md` § Encoded-column lookup.

## Stabilized weights

For each row `i`:

```
SW_i = P(T=1)        / e(X_i)        if T_i = 1
SW_i = P(T=0)        / (1 - e(X_i))  if T_i = 0
```

where `e(X)` is the estimated propensity and `P(T=1)` / `P(T=0)`
are the marginal treatment proportions. Stabilization shrinks the
weight variance relative to unstabilized
(`1/e(X)` and `1/(1-e(X))`).

## Weight trimming

Per G3 (`causal-positivity-diagnostics`)'s positivity rule, **trim
rows where propensity `< 0.05` OR `> 0.95`**. Document the trimmed
`n` in `data_report.warnings`. If the trim takes more than 10% of
the sample, escalate per G3's `extreme_tail_fraction ≥ 0.10` rule.

## Estimator

Weighted regression of `Y ~ T` (no covariates beyond T — the weights
do the adjustment) with `weights=SW` and `cov_type='cluster'` on the
school pseudo-IDs from D1.

## Estimand

**ATE in the analytic sample** by default (stabilized weights target
ATE). Explicit estimand declaration per G2
(`causal-estimand-definition`).

## ESS degeneracy diagnostic

Report the effective sample size:

```
ESS = (sum(SW))^2 / sum(SW^2)
```

**Flag if `ESS < 0.5 * n`** — weight degeneracy means a handful of
rows dominate the estimator, and the stated `n` overstates the
information content. Append to `results.warnings`; if combined with
positivity violation, the estimate should not be reported.

## Weighted balance check (mandatory)

After computing stabilized weights, run G4
(`causal-balance-diagnostics`) with `weights=stabilized_weights` and
report **weighted** SMDs for every covariate in the adjustment set.
Apply the same three-tier threshold rule as G4
(`|SMD| < 0.10` balanced, `0.10–0.25` flag, `≥ 0.25` REVISE). Save
`love_plot_ipw.png` showing pre-weighting vs. post-weighting SMDs.

This is the IPW-specific analogue of M2's matched-balance check.
M3 composes G4 implicitly via `references_skills`, but the
explicitness of this bullet is the enforcement.

## Output schema

```json
"ipw_results": {
  "weight_max": 0.0,
  "weight_min": 0.0,
  "weight_ess": 0,
  "trimmed_n": 0,
  "trimming_threshold": 0.05,
  "ate_estimate": 0.0,
  "ate_ci_lower": 0.0,
  "ate_ci_upper": 0.0,
  "se_method": "cluster_robust",
  "stabilized_weights": true,
  "weighted_balance": {
    "max_post_weighted_smd": 0.0,
    "flagged_covariates": [],
    "love_plot_path": "love_plot_ipw.png"
  }
}
```

## Failures prevented

INF-01 (P), INF-04 (S), INF-05 (S), ESC-01 (S — explicit ATE rule);
residual covariate imbalance under weighting (via G4 weighted
balance check).

## Python implementation guidance

**Primary library:** `statsmodels.regression.linear_model.WLS` or
`statsmodels.GLM` with `freq_weights=stabilized_weights` and
`cov_type='cluster'`. Mature, supports clustering.

**Alternative:** `causalml.inference.meta.LRSRegressor` provides IPW
directly but lacks cluster-robust SEs.

**Function signatures:**

```python
def compute_stabilized_weights(
    propensity: np.ndarray,
    treatment: np.ndarray,
) -> np.ndarray: ...

def trim_extreme_weights(
    df: pd.DataFrame,
    propensity: np.ndarray,
    weights: np.ndarray,
    threshold: float = 0.05,
) -> tuple[pd.DataFrame, np.ndarray]: ...

def ipw_ate(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    weights: np.ndarray,
    cluster_col: str,
) -> dict: ...  # returns ipw_results schema above
```

**Library pitfalls:**

- `statsmodels` GLM does not always honor `cov_type='cluster'`
  correctly with non-identity links; for binary outcomes, use a
  linear probability model (`OLS` / `WLS` with binary Y) plus
  clustered SE — then the ATE is the risk difference. Safer than
  fighting GLM's link-function plumbing.
- `freq_weights` vs `var_weights` in `statsmodels`: prefer
  `freq_weights` for IPW (treats the weights as case counts; matches
  the intent that the weighted sample stands in for the target
  population).

## Validation criteria

The SKILL contract requires that:

1. The stabilized-weights formula is present.
2. The trimming rule referencing G3 is present.
3. The ESS degeneracy check (`ESS < 0.5 * n`) is present.
4. The cluster-robust SE rule references D1's pseudo-school-IDs.
5. The output schema is present.
6. The explicit weighted-balance-check bullet invoking G4 with
   `weights=stabilized_weights` and the three-tier SMD threshold
   rule is present.

An Analyst code artifact using this skill must produce:

- `results.estimates.ipw` per schema (including the
  `weighted_balance` block),
- `love_plot_ipw.png`,
- `validation_passed: false` when `weight_ess < 0.5 * n` OR
  `max_post_weighted_smd >= 0.25`.

## Source provenance

Canonical source: the v3.0 causal-methods specification (internal) §3.9
(M3 per-skill specification, including the §3a.1 R4 explicit
weighted-balance-check bullet).
