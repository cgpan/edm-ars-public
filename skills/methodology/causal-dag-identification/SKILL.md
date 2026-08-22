---
name: causal-dag-identification
layer: methodology
description: Every causal study under selection-on-observables must begin with an explicit DAG that names identifying assumptions and surfaces every observable confounder, mediator, and collider in the analytic sample.
trigger_keywords:
  - causal
  - dag
  - identification
  - confounder
  - confounding
  - mediator
  - collider
  - back-door
  - backdoor
  - adjustment-set
applicable_task_types:
  - causal_soo
  - causal_itr
applicable_datasets: []
applicable_stages:
  - ProblemFormulator
  - Analyst
  - Critic
priority: 1
references_skills: []
resources: []
version: "1.0"
---

# Causal DAG and Identification

Every causal study in `task_type=causal_soo` (selection-on-observables)
begins with an explicit DAG. The DAG is the contract between the
research question and the estimator — it names which variables play
which roles (treatment, outcome, confounder, mediator, collider,
instrument, pre-treatment correlate) and which back-door paths the
adjustment set must close. Any method skill (M1–M5) composes this
skill so the DAG check happens before any estimator runs.

## Selection-on-observables identification

The selection-on-observables strategy assumes:

> **No-unmeasured-confounding (NUC):** conditional on the adjustment set
> `C`, treatment assignment `T` is independent of the potential outcomes
> `(Y(0), Y(1))`. That is, every back-door path between `T` and `Y` is
> blocked by `C`, and `C` contains no descendants of `T`.

NUC is a **checkable claim about the DAG**, not a boilerplate
assumption. Every causal study must state the claim, name the
covariates that make it plausible, and acknowledge — by name — any
confounder it *cannot* observe.

## DAG drawing instructions

For every causal research question, construct a DAG with:

- Nodes = treatment `T`, outcome `Y`, and every covariate `C` in the
  candidate adjustment set.
- Edges = directed; an edge `A → B` asserts that `A` causally
  influences `B`.
- Serialization = the DAG MUST be saved in `research_spec.dag` as a
  DOT-string (round-trippable with NetworkX `nx.DiGraph`). DOT is
  preferred over pickled NetworkX so the DAG survives JSON
  round-trip with the rest of `research_spec.json`.

## Adjustment-set selection — the back-door criterion (operational)

For an adjustment set `C` to identify the causal effect of `T` on `Y`:

1. **Close every back-door path** from `T` to `Y`. A back-door path is
   any path beginning with an edge **into** `T` (i.e., `T ← ...`).
   Conditioning on `C` blocks the path iff `C` contains a non-collider
   node on the path.
2. **Include no descendants of `T`** — conditioning on a post-treatment
   variable bars identification by absorbing part of the causal effect
   (post-treatment / mediator-as-confounder bias).
3. **Include no colliders** on the back-door paths unless the path
   requires conditioning on one of the collider's ancestors (the
   ancestor-of-collider rule).

If the DAG implies an un-block-able back-door path — i.e., the
adjustment set requires a confounder that is **not observed** in the
data — identification fails under selection-on-observables.

## Mandatory pre-estimation checklist

Before any estimator runs, the Analyst must produce a covariate
temporal-and-role table. For each candidate covariate `C` in
`research_spec.adjustment_set`, declare:

| Field | Allowed values |
|---|---|
| `temporal_status` | `pre` / `post` / `contemporaneous` (relative to `T`) |
| `dag_role` | `confounder` / `mediator` / `collider` / `instrument` / `pre-treatment correlate` |

Save the table to `research_spec.covariate_temporal_table` and
`data_report.causal_identification.covariate_role_table`. Any covariate
with `temporal_status: post` is forbidden in the adjustment set per
rule 2 of the back-door criterion above.

## Identification-failure escalation

If `dowhy.CausalModel.identify_effect` returns `None`, OR if the DAG
contains an un-block-able back-door path because a required confounder
is unobserved:

- **Analyst** MUST stop. Do NOT silently proceed with "best available"
  adjustment.
- **Critic** MUST issue REVISE.
- The Writer must NOT report a causal estimate.

Write the failure into `data_report.causal_identification`:

```json
{
  "adjustment_set": ["..."],
  "identified": false,
  "identification_method": "backdoor.linear_regression",
  "unmeasured_confounders_named": ["prior teacher quality", "..."]
}
```

`validation_passed: false` follows automatically.

## Composition note

Every method skill (M1–M5) lists this skill in its `references_skills`
so the DAG check + identification step always runs before any
estimator. G2 (estimand definition) also composes this skill — the
estimand is meaningful only against a DAG that identifies it.

## Python implementation guidance

**Primary library:** `dowhy` (PyWhy). `dowhy.CausalModel` constructs
the DAG from a DOT string (or NetworkX `DiGraph`); `identify_effect`
derives the back-door / front-door / IV identification recipe;
`refute_estimate` runs placebo / random-common-cause /
unobserved-common-cause sensitivity checks.

**Pin:** `dowhy>=0.11,<0.13`. The `dowhy` API has historically had
churn between minor versions — pinning the upper bound prevents
silent breakage when an upstream rename lands.

**Fallback:** if `dowhy` becomes unmaintained, the `CausalModel`
skeleton can be replaced with a thin custom wrapper around NetworkX
+ a back-door criterion implementation (~80 LOC: enumerate paths
between `T` and `Y`, classify each edge, check whether the proposed
adjustment set blocks every back-door path).

**Key functions / classes:**

- `dowhy.CausalModel(data, treatment, outcome, graph=DOT_string)`
- `model.identify_effect(method_name="default")` → returns
  `dowhy.causal_identifier.IdentifiedEstimand` or `None`
- `model.refute_estimate(estimate, method_name="random_common_cause" | "placebo_treatment_refuter" | ...)`
- helpers in `dowhy.causal_identifier`

**Function signatures the Analyst should produce:**

```python
def build_causal_model(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    dag_dot: str,
) -> dowhy.CausalModel: ...

def identify_estimand(model: dowhy.CausalModel) -> dowhy.causal_identifier.IdentifiedEstimand: ...

def assert_no_post_treatment_in_adjustment(
    adjustment_set: list[str],
    treatment_var: str,
    var_temporal_table: dict[str, str],  # var -> 'pre' | 'post' | 'contemporaneous'
) -> None: ...
```

**Library pitfalls:**

- `dowhy` has had API churn; pin `dowhy>=0.11,<0.13`.
- The `graph` argument accepts DOT or NetworkX `nx.DiGraph`; prefer DOT
  for round-trip with `research_spec.json`.
- `identify_effect` will silently return `None` for unidentifiable
  estimands — the Analyst code must check and fail loudly.
- `refute_estimate` is best-effort, not proof of identification.

## Validation criteria

The SKILL contract requires that:

1. The back-door criterion is operationally stated (the three rules
   above).
2. The pre/post/contemporaneous temporal-status mandate is enforced
   for every covariate.
3. The un-block-able-back-door escalation rule is honored
   (`validation_passed: false`).
4. The `dowhy.CausalModel` code skeleton is followed (no ad-hoc
   adjustment-set picking).
5. `references_skills` is empty (this skill is the root) and every
   method skill (M1–M5) composes this one.

A Writer using this skill must be able to produce a §Methods /
Identification subsection that names the no-unmeasured-confounding
assumption and lists every conditioning covariate with its DAG role.

An Analyst code artifact using this skill must produce:

- `research_spec.dag` as a DOT string,
- `data_report.causal_identification` with
  `{adjustment_set, identified, identification_method, unmeasured_confounders_named}`,
- explicit `validation_passed: false` when `identified == False`.

## Source provenance

Canonical source: the v3.0 causal-methods specification (internal) §3.1
(G1 per-skill specification, including back-door criterion, DoWhy
implementation skeleton, dowhy version pin, and NetworkX fallback).
