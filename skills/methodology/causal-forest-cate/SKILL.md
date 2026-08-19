---
name: causal-forest-cate
layer: methodology
description: Estimate Conditional Average Treatment Effects (CATE) via econml.dml.CausalForestDML with honest=True, GroupKFold cross-fitting, BH FDR-corrected subgroup CATEs, and ATE-via-averaging labeled "ATE-on-overlap-population" — never "ATE".
trigger_keywords:
  - causal
  - forest
  - cate
  - heterogeneity
  - honest
  - econml
  - causalforestdml
  - subgroup
  - fdr
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
  - causal-positivity-diagnostics
  - causal-balance-diagnostics
  - causal-sensitivity-unmeasured-confounding
  - hsls09-causal-conventions
resources: []
version: "1.0"
rule_severity: mandatory
---

# Causal Forest CATE (M5)

Estimate the Conditional Average Treatment Effect — the treatment
effect as a function of covariates `τ(X) = E[Y(1) − Y(0) | X]` —
via `econml.dml.CausalForestDML` with honest splitting. Surfaces
effect heterogeneity that ATE-only methods (M1, M3, M4) average over.

CATE is **the** estimand of M5 — averaging CATE predictions to a
single number is allowed only when explicitly labeled
"ATE-on-overlap-population" (per **ESC-03**). Honest splitting plus
BH FDR correction on subgroup CATEs are non-negotiable
defenses against subgroup fishing (**INT-04**).

## Estimator

`econml.dml.CausalForestDML` with:

- **`honest=True`** (mandatory) — splits and leaf estimates use
  separate sub-samples, giving asymptotic normality of CATE
  estimates. The `honest=False` default in some `econml` versions
  destroys variance guarantees.
- **Cross-fitting K=5**, with folds respecting school clusters via
  `GroupKFold(5)` and `groups=pseudo_school_id` from D1.

## CATE outputs

- `cate_predictions[i]` for every test-set unit.
- **CATE distribution**: histogram + percentiles (10 / 25 / 50 / 75
  / 90), saved as `cate_distribution.png`.
- **CATE by subgroup**: `(X1SEX, X1RACE, X1SESQ5)` cells, with cell
  `n`. **Flagged if any cell `n < 100`** — small-cell CATEs are
  noise even with honest splitting.

## ATE-via-averaging (label discipline)

`mean(cate_predictions[overlap_region])` is reported as
**"ATE-on-overlap-population"**, NOT as "ATE" (per **ESC-03**). The
overlap region is set by G3's positivity diagnostics; reporting the
overlap-population number as the unconditional ATE is the failure
G2's estimand-name discipline catches.

Variance for the ATE-on-overlap: from honest splitting, via
`econml`'s `inference="auto"` (which uses bootstrap of forest
internally — **not** the same as a naive bootstrap on the data
matrix).

## Heterogeneity tests

- **BLP** (best linear projection of CATE on covariates) via
  `econml.cate_interpreter`. Coefficients identify the directions of
  largest CATE variation.
- **Variance ratio test:** `var(CATE) / var(ATE) > 2` → heterogeneity
  present, report explicitly. The threshold is from the `econml`
  documentation conventions; a ratio > 2 means individual-level
  effects vary substantially around the ATE and a single-number
  summary is misleading.

## Subgroup honesty rule (mandatory)

Subgroups must be **defined a priori** in
`research_spec.subgroup_analyses`. CATE estimates for unspecified
subgroups (post-hoc fishing) are exploratory and **must be labeled
as such** in the paper — they go in §Limitations or an appendix,
not §Results.

## Multiple-comparisons correction (mandatory)

When reporting subgroup CATEs, apply **Benjamini-Hochberg FDR
correction at q = 0.05**; report adjusted p-values alongside the
nominal ones. This is the primary defense against **INF-06** (CATE
multiple comparisons).

## Output schema

```json
"causal_forest_results": {
  "ate_on_overlap": 0.0,
  "ate_on_overlap_ci": [0.0, 0.0],
  "cate_percentiles": {"10": 0.0, "50": 0.0, "90": 0.0},
  "cate_variance_ratio": 0.0,
  "subgroup_cate": {"<subgroup>": {"cate": 0.0, "ci_adj": [0.0, 0.0], "n": 0}},
  "blp_coefficients": {...},
  "honest": true,
  "cv_folds": 5
}
```

## Failures prevented

ESC-03 (P), INT-02 (P), INT-04 (P), INF-06 (P), INF-07 (P).

## Python implementation guidance

**Primary library: `econml.dml.CausalForestDML`.** Best-in-class for
CATE; active maintenance; honest splitting first-class.

**Function signatures:**

```python
def fit_causal_forest(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    covariates: list[str],
    n_estimators: int = 500,
    honest: bool = True,
    cv_folds: int = 5,
) -> CausalForestDML: ...

def cate_distribution(
    forest: CausalForestDML,
    X: pd.DataFrame,
) -> dict: ...  # percentiles + histogram + plot

def subgroup_cate(
    forest: CausalForestDML,
    X: pd.DataFrame,
    subgroup_attrs: list[str],
    fdr_q: float = 0.05,
) -> dict: ...  # subgroup CATEs with BH-corrected CIs

def best_linear_projection(
    forest: CausalForestDML,
    X: pd.DataFrame,
) -> dict: ...  # BLP coefficients
```

**Library pitfalls:**

- `CausalForestDML(honest=False)` is the default in some `econml`
  versions; **MUST set `honest=True` explicitly** — relying on the
  default is fragile across `econml` releases.
- `econml`'s `inference` argument controls variance; use
  `inference="auto"` (**not** `inference="bootstrap"` for forest
  variance — `"auto"` selects the honest-split-aware variance
  estimator).
- `n_estimators=100` is too few for n=20K HSLS; recommend
  `n_estimators >= 500`.
- Cross-fitting folds must respect school clusters (use
  `cv=GroupKFold(5)` with school IDs); `KFold` will leak schools
  across folds.
- **Cluster-bootstrap variance for the ATE-on-overlap is
  computationally expensive** — 1000 iterations × full forest fit
  with `n_estimators=500` is on the order of hours at HSLS scale
  (per audit Open Question #9). The default `inference="auto"` does
  **not** cluster the variance; if a clustered SE on the
  ATE-on-overlap is required, sample-down-and-bootstrap or
  sub-sampling strategies should be considered. Document the
  `inference="auto"` choice (and the absence of cluster-bootstrap
  by default) in §Methods.

## Validation criteria

The SKILL contract requires that:

1. The `honest=True` mandate is named (and the
   `honest=False`-as-default warning).
2. The ATE-via-averaging label rule
   ("ATE-on-overlap-population", not "ATE") is named verbatim.
3. The BH FDR correction at q=0.05 for subgroup CATEs is named.
4. The min-cell-n flag (`n < 100`) is named.
5. The cluster-respecting cross-fitting (`GroupKFold(5)` with
   school IDs) is named.
6. `inference="auto"` is named (not `"bootstrap"`).
7. The cluster-bootstrap timing-risk note from audit Open
   Question #9 is present in "Library pitfalls".
8. The output schema is present.

An Analyst code artifact using this skill must produce:

- `cate_distribution.png`,
- `results.estimates.causal_forest` per the output schema,
- `results.warnings` populated for any subgroup with `n < 100`,
- BH-corrected p-values in `subgroup_cate[*].ci_adj`.

## Source provenance

Canonical source: `docs/v3_0_causal_skill_specification.md` §3.11
(M5 per-skill specification, including the `honest=True` mandate,
the ATE-on-overlap-population label, BH FDR correction, and the
cluster-bootstrap timing-risk note from audit Open Question #9).
