---
name: causal-estimand-definition
layer: methodology
description: Every causal study must declare its estimand explicitly — ATE / ATT / ATC / CATE — with target population, treatment contrast, and prose-rule binding to keep results faithful to the declared estimand.
trigger_keywords:
  - causal
  - estimand
  - ate
  - att
  - atc
  - cate
  - contrast
  - population
applicable_task_types:
  - causal_soo
  - causal_itr
applicable_datasets: []
applicable_stages:
  - ProblemFormulator
  - Analyst
  - Critic
  - Writer
priority: 1
references_skills:
  - causal-dag-identification
resources: []
version: "1.0"
rule_severity: mandatory
---

# Causal Estimand Definition

Every causal `research_spec` must declare its **estimand** explicitly.
"What causal quantity are we estimating, in which target population,
under which treatment contrast?" answered up front. No causal estimate
may be reported without a declared estimand. Estimand mismatch
between the declared spec and the prose description is silent
corruption — `rule_severity: mandatory`.

## Estimand taxonomy

| Estimand | Definition |
|---|---|
| **ATE** (Average Treatment Effect) | `E[Y(1) - Y(0)]` — average potential-outcome contrast across the entire target population. |
| **ATT** (Average Treatment effect on the Treated) | `E[Y(1) - Y(0) | T=1]` — average effect among units who actually received treatment. |
| **ATC** (Average Treatment effect on the Controls) | `E[Y(1) - Y(0) | T=0]` — average effect among untreated units. |
| **ATU** (Average Treatment effect on the Untreated) | Synonym for ATC. |
| **CATE** (Conditional Average Treatment Effect) | `E[Y(1) - Y(0) | X=x]` — effect conditional on covariate profile `X=x`; produced by causal forest. |

**HSLS-grounded ATT example:** for treatment `T = 1{X1MTHEFF >= median}`
and outcome `Y = X4EVRATNDCLG`,

> ATT = expected change in `X4EVRATNDCLG` if every above-median student
> had instead been below-median (and every below-median had stayed
> below-median).

## Target-population taxonomy

| Target | Definition / V3.0 default |
|---|---|
| **HSLS analytic sample** | The rows surviving listwise deletion + analytic-sample filters. **V3.0 default** — no survey weights applied. |
| **HSLS sampling-frame population** | The U.S. 9th-grade cohort of 2009. Estimating this requires applying `W4W1W2W3STU` survey weights. **Out of scope for V3.0.** |
| **Super-population** | An idealized infinite population beyond HSLS. **Out of scope for V3.0** — see `hsls09-causal-conventions`. |

If V3.x or later wants population-marginal estimands, see audit
Open Question #4 (`hsls09-survey-weights-causal-application` skill,
deferred).

## Treatment-contrast specification

Every causal spec must populate `research_spec.treatment_contrast`.
Allowed schemas:

```json
{"type": "binary_split", "threshold": "median"}
{"type": "binary_split", "threshold": 0.0}
{"type": "1sd_increase"}
{"type": "categorical_pairwise", "reference": "level_0"}
{"type": "binary_indicator"}   // already binary in the data
```

**No causal estimate may be reported without a defined contrast.**
"The effect of `X1MTHEFF`" is not a sentence — `X1MTHEFF` is a
continuous attitudinal scale (range −2.92 to 1.62); the contrast must
spell out what 0/1 (or low/high, or +1 SD) means before a number can
be produced.

## Method → default-estimand mapping (mandatory rule)

| Method | Default estimand | Notes |
|---|---|---|
| Regression adjustment (M1) | ATE (with marginal standardization) OR conditional effect | The choice must be declared. |
| PSM (M2) | **ATT** (matched controls to treated) | ATE only with bilateral matching; otherwise PSM gives ATT. |
| IPW with stabilized weights (M3) | ATE in the population | ATT if conditional weighting is used. |
| AIPW / TMLE (M4) | ATE | DR estimator targets ATE. |
| Causal forest (M5) | CATE | "ATE" only via averaging over a declared population — never the default. |

The Critic must verify that the declared estimand in
`research_spec.causal_estimand` matches the estimand the chosen
estimator actually targets. Mismatch is a critical issue.

## Writer-stage rules — describing selection-on-observables results

### Required prose template

> Under the assumption that all confounders are observed and properly
> modeled (see DAG, §Identification), the estimated **[ATE/ATT/CATE]**
> of **[treatment contrast]** on **[outcome]** in **[target population]**
> is X.XX (95% CI [X.XX, X.XX]).

### Forbidden phrases

The following phrases are **forbidden** when describing
selection-on-observables results:

- "X causes Y"
- "the effect of X" (without an estimand qualifier)
- "increasing X leads to Y"

### Required hedges

Every prose description of a causal estimate must include all three:

- "**estimated [ATE/ATT]**" — name the estimand
- "**under the no-unmeasured-confounding assumption**" — name the
  identification assumption
- "**in the HSLS analytic sample**" — name the target population (or
  the equivalent if a different population is targeted)

## Mandatory tagging

`rule_severity: mandatory`. Estimand mismatch is silent corruption:
the declared "ATE" in the abstract while the estimator returns ATT
produces a paper that misleads readers about who the result applies
to. Any failure of the Critic's estimand-match check is a **critical**
issue.

## Python implementation guidance

**Primary library:** `pydantic` v2 (already in the scientific-Python
ecosystem) for schema validation of `research_spec.causal_estimand`.
No causal-specific library needed.

**Key data structures:**

```python
@dataclass
class CausalEstimand:
    estimand_type: Literal["ATE", "ATT", "ATC", "ATU", "CATE"]
    target_population: Literal["sample", "frame", "super_population"]
    treatment_contrast: TreatmentContrast
```

**Function signatures the Analyst should produce:**

```python
def declare_estimand(
    estimand_type: Literal["ATE", "ATT", "ATC", "ATU", "CATE"],
    target_population: Literal["sample", "frame", "super_population"],
    treatment_contrast: dict,
) -> dict: ...  # returns the validated estimand block

def assert_method_estimand_compatible(
    method_name: str,
    declared_estimand: str,
) -> None: ...  # raises if PSM declared with ATE without bilateral matching
```

**Library pitfalls:** none directly. The risk is that the LLM ignores
the declaration and produces prose that contradicts it; the Critic
checklist must catch this.

## Validation criteria

The SKILL contract requires that:

1. The formal definitions of ATE/ATT/ATC/CATE are present in the body.
2. The method → default-estimand mapping table is present and
   referenced by every method skill.
3. The Writer prose template + forbidden-phrase list + required-hedge
   list are present verbatim.
4. `rule_severity: mandatory` is set in frontmatter.

A Writer using this skill must be able to produce §Results /
§Discussion prose that matches the template within ±5 words on the
required hedges.

An Analyst code artifact using this skill must produce:

- `research_spec.causal_estimand = {estimand_type, target_population, treatment_contrast}`,
- `results.causal_estimand_check = {declared, used_by_estimator, match: bool}`,
- `match: false` triggers a critical Critic issue.

## Source provenance

Canonical source: the v3.0 causal-methods specification (internal) §3.2
(G2 per-skill specification, including the estimand taxonomy, the
method→estimand mapping table, the Writer prose template, the
forbidden phrases, and the required hedges).
