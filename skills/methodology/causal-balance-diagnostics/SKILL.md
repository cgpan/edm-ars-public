---
name: causal-balance-diagnostics
layer: methodology
description: Verify adjustment achieves a defensible match between treated and control populations — propensity-context for M2/M3/M4 (SMD, Love plot) or regression-context for M1 (covariate overlap, Cook's D leverage, residuals-vs-treatment misspecification).
trigger_keywords:
  - causal
  - balance
  - smd
  - love-plot
  - overlap
  - leverage
  - cook
  - misspecification
applicable_task_types:
  - causal_soo
  - causal_itr
applicable_datasets: []
applicable_stages:
  - Analyst
  - Critic
  - Writer
priority: 1
references_skills:
  - causal-dag-identification
  - causal-positivity-diagnostics
resources: []
version: "1.0"
---

# Causal Balance Diagnostics (Dual-Mode)

The headline ATE is defensible only if the adjustment actually
matched the treated and control populations on the covariates the DAG
identified as confounders. Different methods need different balance
checks: propensity-based methods (M2, M3, M4) get standardized mean
differences and Love plots; regression adjustment (M1) — which has no
propensity score — gets the regression-context analogues (covariate
overlap, Cook's distance leverage, residual-treatment misspecification).
Both branches live in this single skill, distinguished by a `mode`
discriminator in the output schema.

## Propensity-context diagnostics (default mode, when invoked from M2/M3/M4)

- **Standardized mean difference (SMD)** defined:
  `SMD = (mean_treated - mean_control) / pooled_SD`. Absolute value is
  reported.
- **Pre/post-adjustment SMD comparison** is required for every
  covariate in the adjustment set, where "post-adjustment" means after
  propensity weighting (M3, M4) or matching (M2).
- **Threshold rules:**

  | `|SMD|` | Verdict |
  |---|---|
  | `< 0.10` | **Balanced** (acceptable). |
  | `0.10 ≤ ... < 0.25` | **Imbalance flag.** Document in warnings; consider re-specifying the propensity model. |
  | `≥ 0.25` | **Severe imbalance.** Critic issues REVISE. |

- **Love plot** (Cohen-style dot plot) showing pre vs. post-adjustment
  SMD for every covariate; saved as `love_plot.png`.
- For **categorical covariates with ≥ 3 levels**: SMD computed per
  level, reported as max-across-levels.
- For **interaction terms** (e.g., `X1SES × X1RACE`): balance must be
  checked on the interaction, not just the main effects.

## Regression-context diagnostics (when invoked from M1)

M1 (regression adjustment) does not estimate a propensity score, so
the propensity-based pre/post-SMD recipe above does not apply. M1
instead requires three regression-flavored analogues of balance:

- **Covariate overlap diagnostic:** for each continuous covariate in
  the adjustment set, compute the overlap of the treated vs. control
  distributions (e.g., quantile-quantile range overlap, or the
  fraction of the treated covariate range that lies within the
  control covariate range). **Flag covariates with < 80% overlap as
  extrapolation risks** — the regression is interpolating across
  populations rather than comparing like with like.
- **Leverage diagnostic:** compute Cook's distance for the
  regression's ATE coefficient; **flag any single observation with
  `Cook's D > 4/n`** as high-leverage on the causal estimate. A
  handful of high-leverage rows can dominate the ATE.
- **Outcome-model misspecification diagnostic:** for linear outcome
  models, plot residuals vs. treatment indicator; **if the residual
  mean differs across treatment arms by more than 0.10 SD of the
  residuals, flag misspecification** (the outcome model is failing
  to absorb the treatment-confounder interaction). For logistic
  outcome models, use Pearson residuals stratified by treatment.

These three are M1's analogues of propensity-based balance and serve
the same role: surfacing residual confounding-by-imperfect-adjustment
that the headline ATE does not reveal.

## Output schema

The single output schema covers both modes via the `mode`
discriminator:

```json
"balance_diagnostics": {
  "mode": "propensity | regression",
  // propensity mode (M2/M3/M4)
  "pre_adjustment_smd": {"<covariate>": 0.XX, ...},
  "post_adjustment_smd": {"<covariate>": 0.XX, ...},
  "max_residual_smd": 0.XX,
  "flagged_covariates": [...],
  "love_plot_path": "love_plot.png",
  // regression mode (M1)
  "covariate_overlap": {"<covariate>": 0.XX, ...},
  "low_overlap_covariates": [...],
  "high_leverage_rows": 0,
  "max_cook_d": 0.0,
  "residual_mean_gap_sd": 0.0,
  "misspecification_flag": false
}
```

## Failures prevented

IDF-01 (S), IDF-04 (S — overlapping with positivity); regression-context:
extrapolation, leverage, and outcome-model misspecification (composed
by M1).

## Python implementation guidance

**Primary library (propensity mode):** `tableone`
(`pip install tableone`) for SMD computation across covariates with
both continuous and categorical handling; or roll-your-own using
`numpy` for SMD + `matplotlib` for the Love plot.

**Primary library (regression mode):**
`statsmodels.stats.outliers_influence.OLSInfluence` for Cook's
distance and leverage; `numpy` / `scipy.stats` for quantile-overlap
and residual-gap computations.

**Note:** `tableone` produces standardized differences by default but
in a Table 1 format, not a balance-diagnostic format. Recommend a
thin wrapper.

**Function signatures the Analyst should produce:**

```python
# Propensity-mode (M2/M3/M4)
def compute_smd(
    df: pd.DataFrame,
    treatment_col: str,
    covariates: list[str],
    weights: np.ndarray | None = None,
) -> dict[str, float]: ...

def love_plot(
    pre_smd: dict[str, float],
    post_smd: dict[str, float],
    output_path: str,
    threshold: float = 0.10,
) -> None: ...

# Regression-mode (M1)
def covariate_overlap(
    df: pd.DataFrame,
    treatment_col: str,
    covariates: list[str],
) -> dict[str, float]: ...  # fraction of treated range inside control range

def regression_leverage_diagnostics(
    fitted_ols_results,  # statsmodels OLSResults
) -> dict: ...  # uses OLSInfluence.cooks_distance; returns {max_cook_d, high_leverage_rows}

def residual_treatment_gap(
    fitted_ols_results,
    treatment: np.ndarray,
) -> float: ...  # |mean(resid|T=1) - mean(resid|T=0)| / sd(resid)
```

**Library pitfalls:**

- `tableone`'s SMD computation does not natively handle weighted
  samples (for IPW); use a custom weighted SMD:
  `(weighted_mean_treated - weighted_mean_control) / weighted_pooled_SD`.
- `OLSInfluence.cooks_distance` returns a tuple `(d, p)`; the Cook's
  distance values are `d[0]`. Compute on the fitted regression that
  includes both treatment and adjustment covariates so leverage is
  measured on the ATE-bearing model, not a treatment-free outcome
  model.
- For logistic outcome models in M1, `OLSInfluence` does not apply;
  use `GLMInfluence` from the same submodule.

## Validation criteria

The SKILL contract requires that:

1. The SMD formula is present.
2. The three-tier threshold rule (`<0.10` / `0.10–0.25` / `≥0.25`) is
   present.
3. The Love-plot specification is present.
4. The per-level SMD rule for categoricals is present.
5. The regression-context branch is present, with covariate-overlap
   (`<80%`), Cook's distance (`>4/n`), and residual-treatment-gap
   (`>0.10 SD`) rules and the explicit note that M1 invokes G4 in
   regression-context mode.

A Writer using this skill must be able to produce either:

- a **§Methods/Balance** subsection with the pre/post SMD comparison
  and a reference to the Love plot figure (propensity mode), or
- a **§Methods/Diagnostics** subsection naming low-overlap covariates,
  the count of high-leverage rows, and the residual-treatment gap
  (regression mode).

An Analyst code artifact using this skill must produce:

- in **propensity mode**: `love_plot.png` and
  `results.balance_diagnostics` with `mode: "propensity"` per schema;
  `validation_passed: false` when `max_residual_smd >= 0.25`.
- in **regression mode** (when invoked by M1):
  `results.balance_diagnostics` with `mode: "regression"` per schema,
  with low-overlap / high-leverage / misspecification flags appended
  to `results.warnings`.

## Source provenance

Canonical source: the v3.0 causal-methods specification (internal) §3.4
(G4 per-skill specification, including the §3a.1 R2 Path A
extension that adds the regression-context branch invoked by M1).
