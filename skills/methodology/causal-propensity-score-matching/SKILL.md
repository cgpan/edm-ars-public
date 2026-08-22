---
name: causal-propensity-score-matching
layer: methodology
description: Estimate ATT via greedy 1:k nearest-neighbor matching on the propensity score with caliper, cluster-aware bootstrap SEs (or Abadie-Imbens analytic SEs), and pre/post balance reporting; matching mechanics are locked (greedy, ascending-propensity match order, ascending-index tie-break).
trigger_keywords:
  - causal
  - psm
  - matching
  - propensity-score
  - nearest-neighbor
  - caliper
  - att
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

# Causal Propensity Score Matching (M2)

Estimate the ATT by matching every treated unit to its nearest
control on the estimated propensity score, then averaging the
treatment-control outcome differences over matched pairs.
Cluster-aware bootstrap SEs aggregate matched pairs at the school
level. Matching mechanics are explicitly locked below — the custom
implementation is < 100 LOC and gives full control over caliper,
replacement, and matched-pair tracking (needed for the cluster
bootstrap).

**Adjustment-set resolution:** apply D1's `resolve_encoded_columns` rule when
constructing the design matrix for the propensity model. Do not look up original
categorical names in `df.columns` directly — categorical adjustment variables (X1RACE,
X1PAREDU, X1SEX, etc.) have been one-hot encoded by DataEngineer into `<varname>_<level>`
columns. See `skills/dataset/hsls09-causal-conventions/SKILL.md` § Encoded-column lookup.

## Matching specification

- **1:1 nearest-neighbor by default.**
- **1:5 alternative** if controls are plentiful.
- **With-replacement allowed** when the control pool is small (see
  Matching algorithm below for the threshold).

## Caliper rule

**Caliper ≤ 0.2 SD of the propensity score** (Austin 2011).
Unmatched treated units → drop and report in `n_unmatched_treated`.

## Estimand

**ATT by default** (matched controls to treated). Explicit
declaration required per G2 (`causal-estimand-definition`). PSM
gives the ATT, **not** the ATE — reporting an unqualified "ATE" from
a 1:1 PSM run is the ESC-01 failure G2 is built to prevent.

## Matching algorithm (locked)

- **Greedy 1:1 nearest-neighbor** matching on the propensity score
  (**not** optimal/Hungarian matching). Justification: greedy is
  the EDM/AIED publication standard; optimal matching's gains are
  marginal at HSLS scale (n ≈ 20K) and not worth the extra
  dependency cost (`networkx` min-cost-flow or
  `scipy.optimize.linear_sum_assignment` + custom adapter).
- **Match order:** ascending propensity score among treated units
  (treated unit with the **lowest** propensity matched first).
  Deterministic; reproducible.
- **Tie handling:** when two control units are equidistant from a
  treated unit on propensity, break ties by **ascending row index**
  (`df.index` order). Deterministic.
- **Replacement threshold:** matching is **without replacement** by
  default. Switch to with-replacement only when
  `n_control / n_treated < 5`. Document the switch in
  `data_report.warnings`.
- **Caliper enforcement:** caliper applied as a **post-hoc filter on
  returned `NearestNeighbors` distances**; treated units whose
  nearest control exceeds the caliper are dropped and counted in
  `n_unmatched_treated`.

These five locks make the PSM result reproducible across runs and
across implementations — critical for the regression and audit
cycles that compare M2 results to M1/M3/M4.

## SE rule

- **Default:** cluster-bootstrap on matched pairs (resample matched
  pairs as units, with school-level clustering). Per **INF-02**, do
  **NOT** bootstrap student rows independently — that destroys the
  matched-pair structure.
- **Alternative:** Abadie-Imbens analytic SE for matching estimators
  (no clean Python implementation; flag if requested and recommend a
  custom port).

## Balance check

G4 (`causal-balance-diagnostics`) must run **after** matching (in
propensity-context mode). If `max_residual_smd ≥ 0.10`, re-specify
the propensity model and re-match. Allow up to **2 iterations**;
after 2 iterations with `≥ 0.25` residual SMD,
`validation_passed: false`.

## Output schema

```json
"psm_results": {
  "n_treated": 0,
  "n_control_matched": 0,
  "n_unmatched_treated": 0,
  "caliper_used": 0.2,
  "match_ratio": "1:1",
  "att_estimate": 0.0,
  "att_ci_lower": 0.0,
  "att_ci_upper": 0.0,
  "se_method": "cluster_bootstrap_matched_pairs",
  "balance_max_residual_smd": 0.0
}
```

## Failures prevented

ESC-01 (P), INF-02 (P), INF-04 (S), INF-05 (S).

## Python implementation guidance

**Primary library: custom implementation** using
`sklearn.neighbors.NearestNeighbors` for matching +
`sklearn.linear_model.LogisticRegression` or
`GradientBoostingClassifier` for propensity. Justification:
`psmpy` (the only PyPI option) is single-author, low test coverage,
last commit 2023; `causalml.match.NearestNeighborMatch` exists but
has limited caliper / matching-ratio control.

The custom implementation is **< 100 LOC** and gives full control
over caliper, replacement, and matched-pair tracking (needed for
the cluster bootstrap).

**Function signatures:**

```python
def estimate_propensity_for_matching(
    df: pd.DataFrame,
    treatment_col: str,
    covariates: list[str],
) -> np.ndarray: ...

def match_nearest_neighbor(
    propensity: np.ndarray,
    treatment: np.ndarray,
    caliper_sd: float = 0.2,
    ratio: int = 1,
    with_replacement: bool = False,  # auto-switch to True if n_control/n_treated < 5
) -> dict: ...
    # Implementation locks (see Matching algorithm in §3.8 required content):
    #   - greedy 1:1 NN, NOT optimal/Hungarian
    #   - match order: ascending propensity among treated units
    #   - tie-break: ascending row index (df.index)
    #   - caliper: post-hoc filter on returned NearestNeighbors distances;
    #     drop treated units whose nearest control exceeds caliper_sd * sd(propensity)
    #   - replacement: respect with_replacement flag; auto-flip to True if
    #     n_control/n_treated < 5 (record flip in data_report.warnings)
    # returns {"matched_pairs": [(treated_idx, control_idx), ...],
    #          "unmatched_treated": [idx, ...],
    #          "with_replacement_used": bool}

def att_from_matched_pairs(
    df: pd.DataFrame,
    matched_pairs: list[tuple[int, int]],
    outcome_col: str,
) -> float: ...

def cluster_bootstrap_att(
    matched_pairs: list[tuple[int, int]],
    pair_cluster_ids: list[int],  # each pair's school cluster
    df: pd.DataFrame,
    outcome_col: str,
    n_boot: int = 1000,
) -> tuple[float, float]: ...
```

**Library pitfalls:**

- `sklearn.neighbors.NearestNeighbors` doesn't natively support
  caliper — implement as a post-hoc filter on returned distances.
- `psmpy` v1.x has bugs in caliper handling; do **NOT** use as a
  drop-in replacement.
- `pair_cluster_ids` for the cluster bootstrap should resolve via
  the treated unit's school (matched controls inherit the treated
  unit's cluster for resampling purposes — this matches the design
  of the cluster bootstrap on pairs).

## Validation criteria

The SKILL contract requires that:

1. The matching specification + caliper rule are present.
2. The ATT-by-default estimand declaration is present.
3. The cluster-bootstrap-on-pairs SE rule is present (and the
   "do NOT bootstrap student rows independently" warning).
4. The re-match-on-imbalance loop (up to 2 iterations) is present.
5. The output schema is present.
6. The locked-mechanics block is present verbatim: greedy 1:1,
   ascending-propensity match order, ascending-index tie-break,
   `n_control/n_treated < 5` replacement threshold, post-hoc caliper
   enforcement.

An Analyst code artifact using this skill must produce:

- `results.estimates.psm` per the output schema,
- `validation_passed: false` if `balance_max_residual_smd >= 0.25`
  after 2 re-match iterations.

## Source provenance

Canonical source: the v3.0 causal-methods specification (internal) §3.8
(M2 per-skill specification, including the §3a.1 R3
locked-mechanics block).
