---
name: causal-aipw-tmle
layer: methodology
description: Estimate ATE via doubly-robust methods — AIPW (single-step DR via econml.dr.DRLearner) + TMLE (targeted update via zEpid) — with cross-fitting (K=5, cluster-respecting), influence-function variance, and IPCW augmentation for MAR outcomes.
trigger_keywords:
  - causal
  - aipw
  - tmle
  - doubly-robust
  - dr
  - drlearner
  - cross-fitting
  - influence-function
  - econml
  - zepid
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

# Causal AIPW + TMLE (M4)

Doubly-robust (DR) ATE estimation. Two estimators in one skill:
**AIPW** (single-step augmented IPW) and **TMLE** (targeted maximum
likelihood, an iterative DR estimator with the targeting step).
Both are unbiased if **either** the outcome model **or** the
propensity model is correctly specified — the property that makes
DR methods the workhorse of modern observational causal inference.

Cross-fitting (K=5, cluster-respecting via `GroupKFold`) handles
ML-based nuisance estimators without the bias from in-sample
prediction. Variance is computed from the empirical influence
function — **not** from bootstrap-of-mean (per **INF-03**).

**Adjustment-set resolution:** apply D1's `resolve_encoded_columns` rule when constructing
the design matrix for the propensity model and the outcome model. Do not look up original
categorical names in `df.columns` directly — categorical adjustment variables (X1RACE,
X1PAREDU, X1SEX, etc.) have been one-hot encoded by DataEngineer into `<varname>_<level>`
columns. See `skills/dataset/hsls09-causal-conventions/SKILL.md` § Encoded-column lookup.

## AIPW estimator

```
AIPW = E[μ₁(X) − μ₀(X)]
     + E[ T·(Y − μ₁(X)) / e(X) ]
     − E[ (1−T)·(Y − μ₀(X)) / (1 − e(X)) ]
```

where `μ_t(X) = E[Y | T=t, X]` is the outcome model and `e(X) = P(T=1 | X)`
is the propensity. The first term is the regression imputation; the
second + third are the IPW correction.

## TMLE estimator

1. Estimate the initial outcome model `μ_t(X)`.
2. Compute the "clever covariate":
   ```
   H(T, X) = T / e(X) − (1 − T) / (1 − e(X))
   ```
3. Fluctuate via a parametric submodel (logistic for binary Y,
   linear for continuous), updating the outcome prediction in the
   direction of the clever covariate.
4. Target — iterate until the score equation is satisfied.

The targeting step gives TMLE its sample-double-robustness and lets
the influence-function variance be honest about the actual estimator,
not just an asymptotic argument.

## Cross-fitting (mandatory for ML nuisance estimators)

- **K = 5 folds.**
- Outcome model and propensity model fit on K−1 folds, evaluated on
  the held-out fold.
- Estimands averaged over folds.
- Fold assignments **must respect school clusters** — use
  `GroupKFold(5)` with `groups=pseudo_school_id` from D1.

## Variance

**Empirical variance of the influence function**, NOT bootstrap-of-mean
(per **INF-03**). Both `econml.dr.DRLearner` (for AIPW) and `zEpid` (for
TMLE) report IF-based SEs natively.

## Cluster-aware influence-function variance (AIPW)

When the data has clustering (HSLS:09 has school clustering — see D1), the standard
influence-function variance estimator UNDERSTATES the true variance because it
assumes independent observations.

**Correct formula (mandatory for clustered data):**

```python
# Compute observation-level influence function: phi_i = AIPW score function evaluated at unit i
phi = compute_influence_function(...)  # shape: (n,)

# Cluster-aware variance: sum-of-squares of cluster-level mean influences
clusters = df["school_id"].values
cluster_means = pd.Series(phi).groupby(clusters).mean()
n_clusters = len(cluster_means)
var_aipw = (cluster_means ** 2).sum() / n_clusters  # NOT phi.var() / n
se_aipw = np.sqrt(var_aipw / n_clusters)

# 95% CI uses t-distribution with df = n_clusters - 1 (NOT n - 1)
import scipy.stats
t_crit = scipy.stats.t.ppf(0.975, df=n_clusters - 1)
ci_lower = ate_aipw - t_crit * se_aipw
ci_upper = ate_aipw + t_crit * se_aipw
```

**Sanity-check rule (mandatory):** after computing `se_aipw`, compare to comparator-method SEs.

```python
comparator_ses = [se_m1, se_m2, se_m3]  # cluster-bootstrap or cluster-robust SEs
median_comparator_se = np.median(comparator_ses)
if se_aipw < 0.5 * median_comparator_se:
    warnings.warn(
        f"AIPW SE ({se_aipw:.4f}) is < half the median comparator SE "
        f"({median_comparator_se:.4f}). This is implausible under shared clustering "
        f"and likely indicates the IF was computed without cluster aggregation. "
        f"Re-check the cluster-mean step."
    )
    results["warnings"].append("M4_aipw_se_implausibly_narrow")
```

**Failure mode this prevents (F-AIPW-NARROW-CI):** in 3b.5, AIPW CI was [-0.0113, -0.0073]
(width 0.004) versus comparator widths 0.031–0.071. LSAR flagged this as MAJOR. The
likely cause was IF variance computed at observation level rather than cluster-mean level.
The sanity check above would have fired.

## Missingness handling

When outcome `Y` is MAR, augment AIPW/TMLE with a missingness
propensity model `P(R=1 | X)` (where `R` is the response indicator).
The estimator becomes **IPCW-AIPW** (inverse probability of censoring
weighted). This is the **primary defense against INF-05** for causal
analyses on postsecondary outcomes (`X4*`, `X5*` outcomes per D1).

## Output schema

```json
"aipw_tmle_results": {
  "method": "AIPW | TMLE",
  "ate_estimate": 0.0,
  "ate_ci_lower": 0.0,
  "ate_ci_upper": 0.0,
  "se_method": "influence_function",
  "cross_fitting_folds": 5,
  "outcome_model_cv_score": 0.0,
  "propensity_model_cv_score": 0.0,
  "missingness_adjusted": false
}
```

## Failures prevented

INF-03 (P), INF-05 (P); IDF-01 (S), INF-04 (S).

## Python implementation guidance

### AIPW: `econml.dr.DRLearner` (primary)

`econml.dr.DRLearner` is **active, well-tested, and supports
arbitrary sklearn-compatible nuisance estimators** with built-in
cross-fitting. Returns ATE + IF variance.

### TMLE: `zEpid` (primary)

```
pip install zEpid    # currently v0.9.x
```

The `zepid.causal.doublyrobust.tmle.TMLE` class implements the
targeting step and IF variance. Active maintenance as of 2024 but
smaller community than econml.

**Pin: `zEpid>=0.9.0`** (per audit Open Question #7).

The `zEpid` API has been stable in recent versions but the
maintainer is single-person — pin the lower bound to lock in the
TMLE class signature, but allow upper-bound updates so we get
bug-fix releases.

### Why a library split

**EconML does NOT implement TMLE** (verified in econml docs as of
0.15.x). The AIPW + TMLE skill must split the implementation:
`econml.dr.DRLearner` for AIPW, `zEpid` for TMLE.

### Fallback: custom TMLE (~150 LOC)

If `zEpid` becomes unavailable or unstable, fall back to a custom
TMLE following the Targeted Learning textbook (Van der Laan & Rose
2011). For binary T + binary/continuous Y the recipe is ~150 LOC:

1. Initial outcome model (any sklearn regressor / classifier).
2. Propensity model (any sklearn classifier).
3. Clever covariate as defined above.
4. Logistic-regression fluctuation step on the clever covariate
   (with the initial outcome prediction as the offset).
5. Iterate until the score equation is satisfied.
6. Compute IF: `IF_i = H_i · (Y_i − μ_target(T_i, X_i)) + μ_target(1, X_i) − μ_target(0, X_i) − ATE`.
7. Variance: `Var(ATE) = sum(IF_i^2) / n^2`.

Tractable but adds maintenance burden. **Recommend `zEpid` primary,
custom secondary.**

### Function signatures

```python
def aipw_ate(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    covariates: list[str],
    cluster_col: str,
    outcome_model: BaseEstimator = GradientBoostingRegressor(),
    propensity_model: BaseEstimator = LogisticRegression(),
    cv_folds: int = 5,
) -> dict: ...

def tmle_ate(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    covariates: list[str],
    cluster_col: str,
    missingness_col: str | None = None,  # if MAR adjustment needed
) -> dict: ...
```

### Library pitfalls

- `econml.dr.DRLearner` requires explicit `model_propensity` and
  `model_regression` arguments; defaults are sklearn
  `LogisticRegression()` which underfits attendances on n=20K HSLS.
  Pass `GradientBoostingClassifier`/`GradientBoostingRegressor` or
  similar.
- `zEpid` TMLE assumes binary T; for continuous T, use AIPW only.
- Cross-fitting fold assignments must respect school clusters
  (use `GroupKFold` with school IDs as groups, **not** `KFold` —
  `KFold` will leak schools across folds and inflate the propensity
  model's apparent CV score).

## Validation criteria

The SKILL contract requires that:

1. The AIPW estimator formula is present.
2. The TMLE estimator definition (initial model → clever covariate →
   fluctuation → target) is present.
3. The cross-fitting K=5 mandate with cluster-respecting folds
   (`GroupKFold` with school IDs) is present.
4. The IF-variance rule (NOT bootstrap-of-mean) is present.
5. The IPCW-AIPW recipe for MAR outcome handling is present.
6. The explicit `econml` (AIPW) + `zEpid` (TMLE) library split is
   stated, with the `zEpid>=0.9.0` pin.
7. The custom-TMLE fallback path (~150 LOC) is documented.
8. The output schema is present.

An Analyst code artifact using this skill must produce:

- `results.estimates.aipw` AND `results.estimates.tmle` (both, when
  binary T),
- a comparison of the two ATE estimates, flagging if divergence
  > 30% (the DR property suggests they should agree).

## Source provenance

Canonical source: the v3.0 causal-methods specification (internal) §3.10
(M4 per-skill specification, including the dual library split,
the `zEpid>=0.9.0` pin from audit Open Question #7, and the custom
TMLE fallback path).
