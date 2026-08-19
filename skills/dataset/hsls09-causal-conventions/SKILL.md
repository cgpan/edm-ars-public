---
name: hsls09-causal-conventions
layer: dataset
description: HSLS-specific causal conventions — treatment-relevant variable inventory, pre/post-treatment temporal classification, MNAR diagnostics for postsecondary outcomes, and clustered-SE handling for causal estimators.
trigger_keywords:
  - hsls
  - hsls09
  - causal
  - treatment
  - mnar
  - clustered
  - cluster-robust
  - pre-treatment
  - post-treatment
applicable_task_types:
  - causal_soo
  - causal_itr
applicable_datasets:
  - hsls09_public
applicable_stages:
  - ProblemFormulator
  - DataEngineer
  - Analyst
  - Critic
  - Writer
priority: 1
references_skills:
  - cluster-id-reconstruction-from-fingerprints
resources: []
version: "1.0"
rule_severity: mandatory
---

# HSLS:09 Causal Conventions

HSLS-specific extensions to the cross-dataset causal methodology
skills (G1–G5). Covers the four areas where causal analysis on HSLS
needs concrete, dataset-anchored rules: treatment-variable choice,
pre/post temporal classification, attrition-driven MNAR for
postsecondary outcomes, and clustered-SE handling at the school
level. Pre/post misclassification is silent corruption (introduces
post-treatment bias), so this skill is `rule_severity: mandatory`.

## Treatment-relevant variable inventory

### Continuous attitudinal scales (psychometric)

These are **psychometric scales** that require thresholding or binning
into a defined contrast. **No "per-unit" causal interpretation** is
admissible — the unit is a standardized factor score, not a behavior.

| Variable | Label |
|---|---|
| `X1MTHID` | Math identity scale |
| `X1MTHEFF` | Math self-efficacy scale |
| `X1SCHOOLBEL` | School belonging scale |
| `X1SCIID` | Science identity scale |
| ... | other `X1*` attitudinal composites in the registry |

A causal study that uses one of these as a treatment **must** declare
its `treatment_contrast` in `research_spec.treatment_contrast`
(median split, `>= threshold`, `+1 SD`, etc., per G2's contrast
schema).

### Course-taking / attainment indicators

These are observable behavioral indicators usable as treatments
without further binning:

| Variable | Label |
|---|---|
| `X3TCREDMAT` | Math credits earned |
| `X4HSCOMPSTAT` | High school credential status |
| `X4EVRATNDCLG` | Ever attended college |

### Variables NEVER suitable as treatments

| Pattern | Reason |
|---|---|
| `W*` (e.g., `W4W1W2W3STU`) | Sampling weights; not causally manipulable. |
| `*QSTAT` (e.g., `X1SQSTAT`) | Survey-status indicators; reflect missingness, not treatment. |
| `X1SEX`, `X1RACE` | Demographic protected attributes; **use only as moderators** in CATE / subgroup analyses, never as treatments under selection-on-observables. |

## Pre/post-treatment temporal classification

| Treatment wave | Pre-treatment covariates allowed |
|---|---|
| Any X1 treatment | `X1` baseline variables AND demographics. |
| Any X2 treatment | X1 + X2_baseline (excluding X2 outcomes). |
| Any X3 treatment | X1 + X2 + X3_baseline. |
| Any X4 treatment | X1 + X2 + X3 + X4_baseline. |

Anything from a wave **after** the treatment wave is **post-treatment**
and forbidden in the adjustment set per G1's back-door criterion (no
descendants of `T`).

**Mandatory rule:** every covariate in `research_spec.adjustment_set`
must have its temporal status declared in
`research_spec.covariate_temporal_table`. The wave-prefix mapping is
defined in `hsls09-temporal-ordering` (existing V2.0 dataset skill);
this skill extends it for the causal-adjustment-set use case.

## Selection-bias-from-attrition rules for postsecondary outcomes

| Outcome wave | Approximate missingness | MAR/MNAR risk |
|---|---|---|
| `X4*` (2016 update panel) | ~26% | MAR-leaning, with treatment-correlated patterns |
| `X5*` (2017 postsecondary records) | 50%+ | MAR-to-MNAR, structurally |

For causal targets on postsecondary outcomes:

- The analytic sample is **structurally restricted to respondents** —
  attrition is not random across treatment groups.
- Report **both** an ITT-analogue (X4SQSTAT-respondents only) and the
  as-treated equivalent. Divergence between the two is informative
  about selection bias.
- Recommend running parallel **IPW-for-missingness** analyses (see M4
  `causal-aipw-tmle`'s missingness-augmented branch) to bound the
  missingness-induced bias.

## Clustered-SE handling for causal estimators

This section extends `cluster-id-reconstruction-from-fingerprints`
(existing V2.0 methodology skill) for the causal context — the
pseudo-school-IDs reconstructed there flow into the causal-estimator
SE machinery here.

| Estimator | Clustered-SE recipe |
|---|---|
| **IPW (M3)** | `statsmodels` weighted regression with `cov_type='cluster'`, `cluster_groups=pseudo_school_id`. |
| **PSM (M2)** | Cluster-bootstrap on matched pairs aggregated at school level (matched pair as resampling unit, school as cluster). |
| **AIPW / TMLE (M4)** | Influence-function variance with clustered correction (sum within clusters, then variance across clusters). |
| **Causal forest (M5)** | `econml`'s `inference="auto"` does NOT natively cluster — note this limitation; recommend cluster-bootstrap of the ATE estimate (computationally expensive — see audit Open Question #9). |

## Survey-weights handling

**V3.0 default: do NOT apply HSLS survey weights.**

The estimand is the **analytic-sample marginal effect**, not the
population marginal effect. The Writer must state this explicitly in
both §Methods and §Limitations.

This skill composes `hsls09-survey-weights-limitations-paragraph`
(existing V2.0 writing skill) so the Writer's limitation prose stays
consistent with the prediction-task convention.

## Mandatory tagging

`rule_severity: mandatory`. Pre/post misclassification is silent
corruption — the resulting "causal" estimate absorbs post-treatment
mediation into the treatment effect, biasing the headline number in a
direction that is hard to recover from after the fact.

## Python implementation guidance

**Primary library:** `pandas` + `statsmodels` (`cov_type='cluster'`).

**Reference helpers:**

```python
def cluster_bootstrap_ate(
    df: pd.DataFrame,
    cluster_col: str,
    ate_estimator_fn: Callable[[pd.DataFrame], float],
    n_boot: int = 1000,
    random_state: int = 42,
) -> tuple[float, float]: ...  # returns (ci_lower, ci_upper)

def declare_covariate_temporal_status(
    covariates: list[str],
    treatment_wave: str,
) -> dict[str, Literal["pre", "post", "contemporaneous"]]: ...
```

**Library pitfalls:**

- `statsmodels` `cov_type='cluster'` requires `groups` argument as
  a 1-D array of cluster IDs; pseudo-school-IDs must be aligned by
  `iloc` to the regression's exogenous matrix.
- The pseudo-school-IDs come from
  `cluster-id-reconstruction-from-fingerprints` — that skill's
  diagnostics MUST pass before its IDs are used here, or the
  clustered SE inflates without basis.

## Validation criteria

The SKILL contract requires that:

1. The treatment-relevant variable inventory (with examples) is
   present.
2. The pre/post-treatment temporal table is present.
3. The postsecondary-attrition rules are present.
4. The clustered-SE handling table per estimator is present.
5. The no-survey-weights default is stated.
6. `rule_severity: mandatory` is set in frontmatter.
7. `references_skills` includes `cluster-id-reconstruction-from-fingerprints`.

A Writer using this skill must be able to produce a §Methods
subsection naming (a) the analytic-sample estimand, (b) the
no-survey-weights choice, and (c) the school-clustering correction
method actually used.

An Analyst code artifact using this skill must produce:

- `data_report.causal_covariate_temporal_table`,
- `results.causal_estimate.cluster_se_method`,
- `results.causal_estimate.cluster_se_ci`.

## Encoded-column lookup (mandatory for analysis steps)

Categorical variables in HSLS:09 (e.g., X1RACE, X1PAREDU, X1SEX, X1LOCALE, X1CONTROL,
X1STUEDEXPCT) are one-hot encoded by the DataEngineer stage. The encoded columns are
named `<original>_<level>` — for example, `X1RACE` becomes `X1RACE_1` ... `X1RACE_8`.

**Rule (mandatory):** when an analysis step references an adjustment-set variable that
was originally categorical, do NOT search train_X.csv for the original name. Instead,
discover the encoded columns by prefix match:

```python
def resolve_encoded_columns(varname: str, df_columns: list[str]) -> list[str]:
    """Return all encoded column names for a single original categorical variable."""
    direct = [c for c in df_columns if c == varname]
    if direct:
        return direct  # variable was not encoded (continuous or already binary)
    encoded = [c for c in df_columns if c.startswith(varname + "_")]
    if not encoded:
        raise ValueError(f"Adjustment-set variable {varname!r} not found in df.columns")
    return encoded
```

The full adjustment set passed to a regression / propensity model / weight estimator is the
union of `resolve_encoded_columns(v, df.columns)` over all v in the spec's `adjustment_set`.

**Failure mode this prevents (F-COVARIATE-SET-MISMATCH):** if the adjustment set declares
`X1RACE` and the Analyst code searches for `"X1RACE" in df.columns`, the lookup fails
silently and the propensity / outcome model is fit without race covariates — invalidating
the no-unmeasured-confounding assumption. LSAR will catch this as a FATAL issue.

## Encoding-type discipline (mandatory for DataEngineer)

The previous "Encoded-column lookup" section governs the **Analyst-side**
prefix-match lookup after encoding has already happened. The encoding step
itself — what the DataEngineer writes to `train_X.csv` — is what this
section governs. The two rules are complementary: the DataEngineer chooses
which variables to one-hot; the Analyst looks up whichever choice was made.

The HSLS:09 variable registry (`data_registry/datasets/hsls09_public.yaml`)
tags each variable with a `type` field. The DataEngineer MUST respect the
registry's `type` when constructing the analytic CSV. The 3b.6 D1 work
introduced the registry but did not codify a DataEngineer-side encoding
rule against it; that gap is what this section closes.

### Continuous variables — pass through, do NOT one-hot encode

Variables tagged `type=continuous` in the registry represent measurements
or scales with natural ordering and meaningful distances. Examples from
the registry that have appeared in adjustment-sets in prior runs:

| Variable | Registry `type` | What it is |
|---|---|---|
| `X1MTHEFF` | continuous | Math self-efficacy scale (Wave-1 psychometric factor score, range −2.92 to 1.62) |
| `X1MTHID` | continuous | Math identity scale (range −1.73 to 1.76) |
| `X1MTHINT` | continuous | Math course interest scale (range −2.46 to 2.08) |
| `X1MTHUTI` | continuous | Math utility scale (range −3.51 to 1.31) |
| `X1SCIID` | continuous | Science identity scale (range −1.57 to 2.15) |
| `X1SCHOOLBEL` | continuous | School belonging scale (range −4.35 to 1.59) |
| `X1SES` | continuous | Socioeconomic-status composite (range −1.93 to 2.88) |
| `X1TXMTSCOR` | continuous | Standardized math IRT score (range 24.0 to 82.2) |

These variables MUST be retained as scalar columns in `train_X.csv` and
`test_X.csv`. The DataEngineer MUST NOT call `pd.get_dummies(...)` on them,
MUST NOT bin them into categories, and MUST NOT one-hot encode their
distinct values — even when the underlying factor-score machinery
produces a finite discrete set of unique values that *looks* categorical.
A psychometric scale with 56 distinct factor scores is still a continuous
scale; encoding it as 56 binary dummies discards the scale information
and inflates the predictor matrix in proportions that overfit downstream
propensity models.

### Categorical / binary variables — one-hot encode

Variables tagged `type=categorical` or `type=binary` ARE one-hot encoded.
Concrete examples from the locked-spec adjustment-set space:

| Variable | Registry `type` | What it is |
|---|---|---|
| `X1RACE` | categorical | Race/ethnicity composite (8 unordered categories) |
| `X1PAREDU` | categorical | Parents' highest education (7 ordered codes, but registry-tagged categorical) |
| `X1STUEDEXPCT` | categorical | Student educational expectations (11 ordered codes, registry-tagged categorical) |
| `X1LOCALE` | categorical | School urbanicity (4 unordered categories) |
| `X1REGION` | categorical | Census region (4 unordered categories) |
| `X1CONTROL` | binary | Public vs Catholic/private |
| `X1SEX` | binary | Student sex |

Use `pd.get_dummies(df[col], prefix=col, drop_first=False)` (no reference-
category dropping; downstream models handle the collinearity, and the
3b.6 `resolve_encoded_columns` Analyst-side lookup expects all levels
present).

### Ordinal-but-registry-tagged-categorical — use the registry tag

When the registry tags a variable `categorical` even though it has natural
ordering (e.g., `X1PAREDU` 1–7 on an education ladder, `X1STUEDEXPCT`
1–11), respect the registry tag. The registry authors chose `categorical`
because the codebook codes carry semantically-meaningful labels (`1 =
Less than high school`, `2 = HS diploma/GED`, etc.) and the ordering
between adjacent codes is not numerically comparable. One-hot encoding
preserves the option for downstream models to fit non-linear category
effects.

### Operationalization is separate from encoding

If the research_spec specifies an operationalization for a continuous
treatment (e.g., `median_split_binary` for `X1MTHEFF`), apply the
operationalization per the `causal-data-engineer-contract` skill (3b.12).
The operationalization produces ONE additional column
(`X1MTHEFF_binary`); it does NOT trigger one-hot encoding of the
underlying continuous variable. The continuous variable may itself be
retained as a separate column, dropped, or kept-only-as-operationalized
per the spec's intent.

### Prescriptive recipe

```python
import warnings
import pandas as pd

def encode_for_causal_soo(
    df: pd.DataFrame,
    feature_cols: list[str],
    variable_registry: dict[str, dict],
) -> pd.DataFrame:
    """Encode adjustment-set + treatment columns respecting registry types.

    Args:
        df: source DataFrame containing all `feature_cols`.
        feature_cols: variable names from research_spec
            (treatment + adjustment_set, in any order).
        variable_registry: map from variable name to its registry entry,
            including the `type` field.

    Returns:
        DataFrame with continuous/ordinal-as-continuous columns passed
        through as scalars and categorical/binary columns one-hot
        encoded with explicit prefixes.
    """
    out = pd.DataFrame(index=df.index)
    for col in feature_cols:
        entry = variable_registry.get(col, {})
        var_type = entry.get("type", "unknown")

        if var_type == "continuous":
            # Pass through. Do NOT one-hot encode, even if df[col] has
            # a small number of unique values (factor scores often do).
            out[col] = df[col]

        elif var_type in ("categorical", "binary"):
            # One-hot encode. drop_first=False keeps all levels so the
            # Analyst's resolve_encoded_columns prefix-match (3b.6 rule)
            # finds every encoded variant.
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
            out = pd.concat([out, dummies], axis=1)

        else:
            # Conservative: pass through and warn. Registry-incomplete
            # tagging is a registry concern, not a DataEngineer concern;
            # do not silently re-classify.
            warnings.warn(
                f"Variable {col!r} has registry type {var_type!r}; "
                f"passing through as scalar. Update the registry if "
                f"this should be one-hot encoded."
            )
            out[col] = df[col]

    return out
```

The function takes a `variable_registry` parameter rather than reading the
YAML directly so the rule is testable in isolation. The DataEngineer's
code loads the YAML once and passes the appropriate slice (typically the
full `variables.predictors.*` flat dict) to this function.

### Failure mode this prevents (F-3b15-DE-CONTINUOUS-AS-CATEGORICAL)

The 3b.15 / 3b.17 evidence chain (recoverable in some runs by cycle-1
Critic feedback, not recoverable in others):

| Run | Cycle 0 cols | Cycle 1 cols | Mis-encoded continuous variables (cycle 0) | Outcome |
|---|---:|---:|---|---|
| 3b.13 | 56 | 56 | `X1SCIID` (16 dummies) | Clean — fewer continuous vars in PF's adjustment_set masked the issue |
| 3b.15 | 109 | 11 | `X1MTHID` (16), `X1MTHUTI` (~70) | Cycle-1 over-corrected; LSAR 7.0 Accept |
| 3b.17 | 116 | 115 | `X1MTHID` (17), `X1MTHUTI` (56), `X1STUEDEXPCT` (11) | Cycle-1 did NOT recover; LSAR 6.0 Borderline; 21% extreme-tail positivity violation flagged as "fundamental identification failure" (`F-3b15-DE-CONTINUOUS-AS-CATEGORICAL`) |

The mechanism is the same across runs: the DataEngineer defaults to
`pd.get_dummies(...)` on every adjustment-set variable that isn't already
binary, regardless of registry `type`. When a continuous variable has a
finite set of distinct factor-score values (psychometric scales typically
do — IRT estimators produce a discrete grid), the DE interprets the
finite-value set as categorical and one-hot encodes. The effect compounds
when multiple continuous scales appear in the adjustment_set: 3b.13's
single mis-encoded scale produced a manageable 56-column matrix; 3b.17's
three mis-encoded scales produced a 115-column matrix that overfit the
propensity model and broke positivity.

The Analyst-side `resolve_encoded_columns` rule (3b.6) is the wrong fix
for this — it correctly handles whatever encoding the DE chose, but it
cannot un-do an incorrect encoding choice. The DE-side rule above is the
fix: registry-type dispatch produces deterministic correct behavior
independent of which continuous variables happen to be in the
adjustment_set.

**Cross-reference to existing D1 rules:**
- The Analyst-side `resolve_encoded_columns` lookup (above) reads
  whatever encoding the DataEngineer wrote and is unaffected by this
  rule.
- The temporal-ordering rule in `hsls09-temporal-ordering` (V2.0
  dataset skill) is independent — temporal classification of a
  covariate happens before encoding decisions are made.
- The `causal-data-engineer-contract` rule (3b.12) governs which
  columns the DE writes (treatment + adjustment_set + outcome).
  This rule governs HOW each adjustment_set column is encoded.

## Source provenance

Canonical source: `docs/v3_0_causal_skill_specification.md` §3.6
(D1 per-skill specification). HSLS variable details cross-reference
`data_registry/datasets/hsls09_public.yaml` and the existing V2.0
skills `hsls09-temporal-ordering`, `hsls09-structural-mnar-outcomes`,
`hsls09-school-fingerprints`, and
`cluster-id-reconstruction-from-fingerprints`.
