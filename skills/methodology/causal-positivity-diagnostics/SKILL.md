---
name: causal-positivity-diagnostics
layer: methodology
description: Compute and act on common-support diagnostics for the propensity score; never proceed with an estimator on a sample with documented positivity violations without explicit handling.
trigger_keywords:
  - causal
  - positivity
  - propensity
  - common-support
  - overlap
  - trimming
  - extreme-weights
applicable_task_types:
  - causal_soo
  - causal_itr
applicable_datasets: []
applicable_stages:
  - Analyst
  - Critic
priority: 1
references_skills:
  - causal-dag-identification
resources: []
version: "1.0"
rule_severity: mandatory
---

# Causal Positivity Diagnostics

The positivity assumption is the second pillar of identification under
selection-on-observables (the first being no-unmeasured-confounding,
G1). When positivity fails — i.e., some treatment × covariate strata
have ~0 propensity — the estimator is dominated by a handful of rows
with extreme weights, the influence function blows up, and the
resulting "ATE" describes a slice of the data the analyst did not
intend to study. Positivity ignored is silent corruption, so this
skill is `rule_severity: mandatory`.

## Positivity assumption (formal)

For all `x` in the support of covariate vector `X`,

```
0 < P(T=1 | X=x) < 1
```

That is, every covariate profile must have a non-trivial probability
of receiving each treatment level. Violation means there is no
counterfactual to compare against in some part of the covariate space.

## Diagnostics to compute

After identification (G1) and adjustment-set selection, fit a
propensity model `e(X) = P(T=1 | X)`. Then compute and report:

- **Propensity-score histogram by treatment arm**, overlaid → save as
  `propensity_overlap.png`.
- **Trimmed common-support range** — e.g., trim where
  `min(propensity_treated, propensity_control) < 0.05`.
- **Count of rows in the extreme tails** (propensity < 0.05 OR > 0.95)
  and as a fraction of `n`.
- **Effective sample size after trimming**.

## Decision rule (mandatory)

| `extreme_tail_fraction` | Action |
|---|---|
| `< 0.02` | Trim and proceed. Document trimmed `n` in `data_report.warnings`. |
| `0.02 ≤ ... < 0.10` | Trim, proceed, **AND** restrict the estimand to the overlap region. The estimand becomes "**ATE-on-overlap-population**", named explicitly in §Methods and §Results. |
| `≥ 0.10` | **Positivity violation.** Set `validation_passed: false`. Analyst MUST flag, Critic MUST issue REVISE. |

The estimand-renaming step at the middle tier is non-negotiable: an
"ATE" reported on a sample restricted to the overlap region is not the
ATE on the original target population, and the prose must say so.

## Output schema

```json
"positivity_diagnostics": {
  "propensity_min": 0.0, "propensity_max": 1.0,
  "extreme_tail_count": 0, "extreme_tail_fraction": 0.0,
  "trimming_applied": true, "trimmed_n": 0,
  "decision": "proceed | proceed_with_restricted_estimand | abort"
}
```

## Mandatory tagging

`rule_severity: mandatory`. Estimators run on positivity-violating
data are dominated by ~50 rows with weight > 50 — the headline ATE
becomes a property of those rows, not of the target population.

## Python implementation guidance

**Primary library:** `sklearn.linear_model.LogisticRegression` (or
`sklearn.ensemble.GradientBoostingClassifier`) for propensity
estimation; `matplotlib` for the overlap plot. No specialized causal
library is needed for the diagnostics themselves.

**Key functions / classes:** `LogisticRegression`, `predict_proba`,
`np.histogram`, `matplotlib.pyplot.hist`.

**Function signatures the Analyst should produce:**

```python
def estimate_propensity(
    df: pd.DataFrame,
    treatment_col: str,
    covariates: list[str],
    estimator: str = "logistic",  # or "gradient_boosting"
) -> np.ndarray: ...

def positivity_diagnostics(
    propensity: np.ndarray,
    treatment: np.ndarray,
    tail_threshold: float = 0.05,
) -> dict: ...  # returns the schema above; saves overlap plot

def apply_positivity_decision(
    df: pd.DataFrame,
    diagnostics: dict,
) -> tuple[pd.DataFrame, str]: ...  # returns trimmed df and decision str
```

**Library pitfalls:**

- `LogisticRegression(max_iter=100)` defaults are too low for n=20K
  HSLS; use `max_iter=1000`.
- `GradientBoostingClassifier` defaults are fine but check calibration
  (e.g., reliability diagram) before using it as a propensity score —
  uncalibrated probabilities make the trimming threshold misbehave.

## Validation criteria

The SKILL contract requires that:

1. The positivity assumption is stated formally.
2. The three-tier decision rule is present with the exact thresholds
   (`< 0.02`, `0.02–0.10`, `≥ 0.10`).
3. The output schema is present.
4. `rule_severity: mandatory` is set in frontmatter.

A Writer using this skill must be able to produce a §Methods /
Positivity subsection naming (a) the trimmed `n`, (b) the
extreme-tail fraction, and (c) the resulting estimand label
(unrestricted / overlap-population).

An Analyst code artifact using this skill must produce:

- `propensity_overlap.png`,
- `results.positivity_diagnostics` populated per schema,
- `validation_passed: false` when `decision == "abort"`.

## Source provenance

Canonical source: `docs/v3_0_causal_skill_specification.md` §3.3
(G3 per-skill specification, including the three-tier decision rule
with the explicit thresholds).
