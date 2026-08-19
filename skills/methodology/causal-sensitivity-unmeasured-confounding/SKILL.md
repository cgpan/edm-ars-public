---
name: causal-sensitivity-unmeasured-confounding
layer: methodology
description: Quantify how strong an unmeasured confounder would need to be to overturn the headline causal estimate via E-value (mandatory) + Rosenbaum bounds (matched designs) + DoWhy refuters; report with calibrated language that prevents "robust" overclaiming.
trigger_keywords:
  - causal
  - sensitivity
  - evalue
  - e-value
  - rosenbaum
  - refuter
  - refutation
  - unmeasured-confounding
  - unmeasured
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
  - causal-estimand-definition
resources: []
version: "1.1"
rule_severity: mandatory
---

# Causal Sensitivity to Unmeasured Confounding

Selection-on-observables identification depends on the assumption that
all confounders are observed. Sensitivity analysis quantifies how
strong an unobserved confounder would need to be to overturn the
headline causal estimate. Omitting sensitivity analysis is structural
incompleteness for any causal study (no way for the reader to gauge
how fragile the claim is), so this skill is `rule_severity: mandatory`.

## E-value (mandatory)

### Formal definition

The **E-value** is the minimum strength of association, on the
risk-ratio scale, that an unmeasured confounder would need to have
with **both** the treatment and the outcome to fully explain away the
observed effect (VanderWeele & Ding 2017).

### Computation rule

Every causal point estimate AND its CI lower bound must report an
E-value. Two values, not one.

### Interpretation table (mandatory in Writer's §Sensitivity subsection)

| E-value | Required wording |
|---|---|
| `< 1.5` | "result is **fragile** to unmeasured confounding" |
| `1.5 ≤ E-value < 2.5` | "result requires **moderate-strength** unmeasured confounder to overturn" |
| `≥ 2.5` | "result requires a **strong** unmeasured confounder to overturn (still possible)" |

### Forbidden phrase

> **"result is robust to unmeasured confounding"**

This phrase is **forbidden**. Any E-value `< ∞` is consistent with
some unmeasured confounder existing — claiming "robustness" against
unmeasured confounding misrepresents what the E-value tells us.
Calibrate the language to the table above.

## Rosenbaum bounds (matched designs)

For matched designs (PSM), compute the Rosenbaum sensitivity
parameter `Γ` such that conclusions become inconclusive (i.e., the
significance of the matched-pair test would no longer reject the
null at α=0.05 if a hidden confounder shifted treatment-assignment
odds by a factor of `Γ`). Report `Γ` alongside the E-value when
matching is the estimator. Standard practice: report at
`Γ ∈ {1.0, 1.5, 2.0, 2.5, 3.0}` and identify the breakpoint.

## DoWhy refuters (mandatory)

Every causal estimate must run **at least two** DoWhy refuters from
the following set:

- `random_common_cause`
- `placebo_treatment_refuter`
- `data_subset_refuter`
- `add_unobserved_common_cause`

For each refuter, report whether the estimate stays significant at
α=0.05 after the refuter perturbation. Refuters are best-effort, not
proof; chained with E-value + Rosenbaum bounds they collectively
discipline the "how confident" question. "Best-effort" refers to
INTERPRETATION, not invocation — invocation is unconditional (see the
status contract below).

## Refuter execution status contract (MANDATORY — pre-critic asserted)

Attempting the refuters is NOT optional and NOT skippable under time
pressure. The orchestrator's pre-critic guard (pcc_c01) deterministically
asserts this contract on `results.json`:

- `sensitivity.dowhy_refuters` MUST exist with **at least two** refuter
  entries (e.g. `random_common_cause`, `placebo_treatment_refuter`).
- Every entry MUST carry `"status": "ran"` or `"status": "failed"`.
  A failed attempt is acceptable ONLY when documented: set
  `status="failed"` and put the exception text in `"error"`.
- NEVER omit the key, emit an empty dict, or write prose like
  "refuters not executed" in warnings as a substitute — a missing or
  empty `dowhy_refuters` fails pcc_c01 (major, target Analyst) and
  forces a REVISE-weight issue regardless of everything else.

Healthy shape (from run 3b.19):

```json
"dowhy_refuters": {
  "random_common_cause":       {"new_effect": 0.0246, "p_value": 1.0, "status": "ran", "error": null},
  "placebo_treatment_refuter": {"new_effect": -1.55e-15, "p_value": 0.0, "status": "ran", "error": null}
}
```

## DoWhy refuter invocation (mandatory for causal_soo)

The declarative rules above name what to run; this section is the
prescriptive call sequence. Phase 3b.13 (LSAR review of run
`v3_0_smoketest_mtheff_college_20260502_3b13`) found
`sensitivity.refuter_results = []` — the Analyst attempted refuters
but `model.identify_effect()` threw before any refuter could fire.
LSAR cited this gap explicitly (`"the DAG-format error prevented the
planned sensitivity package from running"`) and named it a tractable
fix. This subsection closes it.

### Mandatory rule: DAG node names MUST match the column names

DoWhy's `CausalModel(treatment=..., outcome=..., graph=...)` validates
that the strings passed as `treatment` and `outcome` are **node names
in the graph** — NOT node labels, NOT aliases. The 3b.13 failure was
exactly this confusion: the Analyst built a DOT graph with node IDs
`T` and `Y` and human-readable labels (`label="X1MTHEFF\n(median-
split)"`), then called `CausalModel(treatment="X1MTHEFF_binary",
graph=dot_string)`. DoWhy looked up the node `X1MTHEFF_binary` in the
graph, didn't find it (because the node is named `T`), and threw a
`"Incorrect format: Please provide graph as a networkx DiGraph, GCM
model, or as a string in either GML or DOT format."` error. The
error message is misleading — the format was valid DOT; the name
mismatch is what broke identification.

**Rule:** every node in the DAG must be named with the exact column
name it represents in the analytic dataframe. No aliases. No `T`/`Y`
shorthand. Labels (the `label="..."` attribute) may be added for
readability, but the node IDs themselves must match the column names
passed as `treatment`, `outcome`, and adjustment-set elements.

### Graph construction (required first step)

Build the DAG from `research_spec.adjustment_set + treatment + outcome`,
using column names directly as node IDs:

```python
import networkx as nx


def build_dowhy_graph(
    treatment: str,
    outcome: str,
    adjustment_set: list[str],
) -> nx.DiGraph:
    """Return a NetworkX directed graph for dowhy.CausalModel.

    Node names MUST match the strings passed as ``treatment``,
    ``outcome``, and adjustment-set elements exactly — DoWhy looks them
    up by name. Treatment → outcome is the causal edge of interest;
    every adjustment-set variable is a confounder (edges to both
    treatment and outcome).
    """
    g = nx.DiGraph()
    g.add_edge(treatment, outcome)
    for var in adjustment_set:
        g.add_edge(var, treatment)
        g.add_edge(var, outcome)
    return g
```

The graph is a ``networkx.DiGraph`` object passed directly to
``CausalModel(graph=...)``. DoWhy 0.12 accepts NetworkX graphs natively
and extracts the causal structure without invoking the DOT-string
parser — which requires ``pygraphviz`` as a native dependency (in turn
requires the graphviz C library). Many deployment environments do not
ship pygraphviz, in which case DoWhy raises ``ValueError: "Incorrect
format: Please provide graph as a networkx DiGraph, GCM model, or as a
string or text file in dot, gml format."`` on what is otherwise
syntactically-valid DOT. Using NetworkX directly avoids the pygraphviz
path entirely; NetworkX is already a project dependency.

The 3b.14 amendment originally prescribed DOT-string format on the
basis of DoWhy's public API documentation; the 3b.15 smoke-test run
exposed the runtime pygraphviz requirement (cataloged as
F-3b15-DOWHY-PYGRAPHVIZ-DEPENDENCY-GAP). The 3b.16 refinement above
replaces the DOT path with the NetworkX path. The empirical proof
([3b.15 evidence](#failure-mode-this-prevents) + a direct DoWhy 0.12
reproduction): the DOT path raises ``ValueError`` on
``identify_effect()``; the NetworkX path completes
``identify_effect`` → ``estimate_effect`` → ``refute_estimate`` end-
to-end and returns a populated ``refutation_result`` dict with
``p_value``.

**The node-naming rule from 3b.14 is unchanged.** DAG node names MUST
match the strings passed as ``treatment``, ``outcome``, and
adjustment-set elements. Node-name mismatch is a separate failure
mode (F-3b13) from the graph-format issue (F-3b15); the refinement
addresses only the latter. Both rules must hold for refuters to run.

### Full invocation sequence

DoWhy refutation requires **four ordered steps**: build graph,
construct `CausalModel`, identify the estimand, estimate it, then
refute. Skipping or reordering steps causes silent failures. Use this
exact sequence:

```python
from dowhy import CausalModel
import warnings


def run_dowhy_refuters(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    adjustment_set: list[str],
) -> dict:
    """Run random_common_cause + placebo_treatment_refuter; return
    structured results. Refuters are non-fatal: if either raises,
    capture as a warning.

    Returns the `dowhy_refuters` sub-dict for `sensitivity.dowhy_refuters`.
    """
    # Step 1: build NetworkX DiGraph with column-name node IDs
    # (see build_dowhy_graph). Returning a NetworkX object — NOT a DOT
    # string — sidesteps DoWhy 0.12's pygraphviz dependency.
    graph_nx = build_dowhy_graph(treatment, outcome, adjustment_set)

    # Step 2: construct CausalModel — treatment/outcome args MUST appear
    # as node names in graph_nx (the 3b.14 node-naming rule is unchanged).
    model = CausalModel(
        data=df[[treatment, outcome] + list(adjustment_set)],
        treatment=treatment,
        outcome=outcome,
        graph=graph_nx,  # NetworkX DiGraph (avoids pygraphviz dependency)
    )

    # Step 3: identify the estimand. Must precede estimate_effect.
    identified = model.identify_effect(proceed_when_unidentifiable=True)

    # Step 4: estimate. method_name=backdoor.linear_regression is a
    # reasonable consistent-with-M1 choice; binary outcomes can use
    # backdoor.generalized_linear_model with glm_family='binomial'.
    estimate = model.estimate_effect(
        identified,
        method_name="backdoor.linear_regression",
    )

    # Step 5: refute. Each refuter is wrapped so one failure doesn't
    # block the other.
    results = {}
    for name in ("random_common_cause", "placebo_treatment_refuter"):
        try:
            ref = model.refute_estimate(identified, estimate, method_name=name)
            results[name] = {
                "new_effect": float(ref.new_effect),
                "p_value": (
                    float(ref.refutation_result["p_value"])
                    if isinstance(getattr(ref, "refutation_result", None), dict)
                    and "p_value" in ref.refutation_result
                    else None
                ),
                "status": "ran",
                "error": None,
            }
        except Exception as e:
            results[name] = {
                "new_effect": None,
                "p_value": None,
                "status": "failed",
                "error": str(e),
            }
            warnings.warn(
                f"DoWhy refuter {name} failed: {e}", RuntimeWarning,
            )
    return results
```

### Required output schema in `sensitivity` (or `sensitivity_analysis.json`)

The Analyst's sensitivity output MUST include a `dowhy_refuters` key
with the structure below. `status` is always present; failed refuters
are recorded with `status: "failed"` rather than omitted, so the
Writer's limitations section can acknowledge them honestly.

```json
{
  "e_value_point": 1.09,
  "e_value_ci": 1.01,
  "rosenbaum_gamma_critical": 1.0,
  "rosenbaum_applicable_methods": ["M2"],
  "dowhy_refuters": {
    "random_common_cause": {
      "new_effect": 0.012,
      "p_value": 0.74,
      "status": "ran",
      "error": null
    },
    "placebo_treatment_refuter": {
      "new_effect": 0.001,
      "p_value": 0.97,
      "status": "ran",
      "error": null
    }
  }
}
```

### Exception handling (mandatory)

DoWhy refuters CAN fail in production (graph format, identification
non-trivial, library version drift). The Analyst's wrapper MUST
capture exceptions per refuter and continue with the remaining
refuters; ONE refuter failure must not prevent the other from running.
Each failure is recorded with `status: "failed"` + the exception
string under `error`. This converts a silent degradation (the 3b.13
shape — `refuter_results: []`) into an explicit acknowledgment that
the Writer can reference.

### Interpretation in the paper (mandatory for Writer)

The Writer's §Sensitivity (or §Robustness) subsection MUST report
each refuter's result when `status == "ran"`:

- `random_common_cause` near the original effect with high p-value
  supports robustness to omitted confounders.
- `placebo_treatment_refuter` near zero with high p-value supports
  identification.
- If `status == "failed"` for either refuter, the paper MUST
  acknowledge the failure in §Limitations alongside the existing
  sensitivity-analysis caveats — do not silently omit. The honest
  framing: "The DoWhy {refuter_name} refuter could not be evaluated
  due to {error}; the corresponding sensitivity check is reported
  as incomplete."

### Failure modes this prevents

Two distinct DoWhy refuter failures led to the current prescriptive
form. Both must be addressed for refuters to run; the rules above
close each.

**F-3b13-DOWHY-REFUTERS-GRAPH-FORMAT — node-name mismatch.** In the
3b.13 run (`v3_0_smoketest_mtheff_college_20260501_3b11`), the
Analyst built a DOT graph with node IDs `T` and `Y` (with human-
readable `label=...` attributes) and then called
`CausalModel(treatment="X1MTHEFF_binary", outcome="X4EVRATNDCLG",
graph=dot_string)`. DoWhy could not reconcile the node ID `T` with
the treatment argument `X1MTHEFF_binary`, the graph's edges named
`T`/`Y` against the column names declared in `data`, and surfaced the
mismatch via a confusingly-worded ValueError. `sensitivity.
refuter_results` was an empty array — no refuter ran. LSAR
(Methodological Rigor justification) named this as a significant
tractable gap.

The fix: **column-name node IDs (no aliasing).** Every node in the
graph must be named exactly with the column name it represents in the
analytic dataframe. The 3b.14 amendment added this rule; 3b.16
preserves it unchanged.

**F-3b15-DOWHY-PYGRAPHVIZ-DEPENDENCY-GAP — DOT-parser pygraphviz
requirement.** In the 3b.15 run
(`v3_0_smoketest_mtheff_college_20260502_3b15`), the Analyst followed
the 3b.14 amendment verbatim — including the prescribed DOT-string
graph format — with node names correctly matching the
`treatment`/`outcome` arguments. The DOT itself was syntactically
valid. DoWhy 0.12 nevertheless raised:

> `ValueError: "Incorrect format: Please provide graph as a networkx
> DiGraph, GCM model, or as a string or text file in dot, gml
> format."`

The root cause: DoWhy 0.12's DOT-string parser requires `pygraphviz`
at runtime (which in turn requires the `graphviz` C library). When
`pygraphviz` is unavailable, DoWhy's graph-detection routine emits
the misleading "Incorrect format" message even on otherwise-valid
DOT input. The 3b.14 amendment's prescriptive form was correct in
isolation but not in the deployment environment.

The fix: **NetworkX-DiGraph instead of DOT-string.** DoWhy 0.12 parses
NetworkX graph objects natively (no pygraphviz path); NetworkX is
already a project dependency. The 3b.16 refinement above replaces
the DOT path with the NetworkX path.

**Both rules together** close the refuter-execution failure surface:
node names match the `CausalModel` arguments (F-3b13 fix), and the
graph format avoids the pygraphviz runtime dependency (F-3b15 fix).
The remaining prescriptions — (a) four-step invocation sequence
(`build → CausalModel → identify_effect → estimate_effect →
refute_estimate`), (b) `dowhy_refuters` output schema, (c) per-refuter
exception capture, (d) Writer transparency for `status: "failed"`
records — are unchanged from 3b.14 and remain as the safety net that
LSAR's 3b.15 review credited as methodological transparency.

## Mandatory tagging

`rule_severity: mandatory`. Sensitivity-analysis omission is
structural incompleteness for any causal study.

## Python implementation guidance

**E-value primary library:** there is no clean Python package for the
VanderWeele E-value formula (the R package `EValue` is canonical but
brings a heavy `rpy2` dependency). **Recommended primary: custom
Python (~10 lines)** based on the VanderWeele & Ding (2017) closed
form.

**Rosenbaum bounds primary library:** no clean Python option;
`pymatch` has been unmaintained since 2018. **Recommend a custom
implementation** following Rosenbaum (2002) — Wilcoxon-signed-rank
based bound on matched pairs.

**Refutation tests primary library:**
`dowhy.causal_refuters.{RandomCommonCause, PlaceboTreatmentRefuter, DataSubsetRefuter, AddUnobservedCommonCause}`.
Active and maintained.

**Function signatures the Analyst should produce:**

```python
def evalue_for_estimate(
    point_estimate_rr: float,  # risk ratio scale
    ci_lower_rr: float,
) -> dict: ...  # {"evalue": float, "evalue_ci_lower": float}

def rosenbaum_bounds_matched(
    matched_pairs_treated_outcome: np.ndarray,
    matched_pairs_control_outcome: np.ndarray,
    gamma_grid: list[float] = [1.0, 1.5, 2.0, 2.5, 3.0],
) -> dict: ...

def run_dowhy_refuters(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    adjustment_set: list[str],
) -> dict: ...  # {refuter_name: {new_effect, p_value, status, error}}
# See "DoWhy refuter invocation" section above for the prescriptive
# four-step build → CausalModel → identify → estimate → refute sequence.
```

**Library pitfalls:**

- The VanderWeele E-value formula assumes a risk-ratio-scale
  estimate; for continuous outcomes (e.g., GPA), apply the
  transformation per VanderWeele & Ding (2017) Appendix.
- DoWhy refuters can be slow on large samples — sample down to
  n=10K for refuter calls if the wall-clock cost is unacceptable.

## Validation criteria

The SKILL contract requires that:

1. The E-value formula is present.
2. The three-tier interpretation table (`<1.5` / `1.5–2.5` / `≥2.5`)
   is present verbatim.
3. The forbidden phrase ("result is robust to unmeasured confounding")
   is named verbatim.
4. The DoWhy refuter list with the **count ≥ 2** requirement is
   present.
5. The DoWhy refuter invocation sequence (build → CausalModel →
   identify_effect → estimate_effect → refute_estimate) is present
   verbatim, including the column-name-as-node-ID rule and the
   NetworkX-DiGraph graph format. (Added 3b.14 for
   F-3b13-DOWHY-REFUTERS-GRAPH-FORMAT; refined 3b.16 for
   F-3b15-DOWHY-PYGRAPHVIZ-DEPENDENCY-GAP — graph format moved from
   DOT string to NetworkX DiGraph.)
6. `rule_severity: mandatory` is set in frontmatter.

A Writer using this skill must be able to produce a §Discussion /
Sensitivity subsection with E-value, refuter results (including
honest acknowledgment of `status: "failed"` cases per the 3b.14
guidance), and explicit calibrated language (no "robust" claim).

An Analyst code artifact using this skill must produce:

```json
"sensitivity_analysis": {
  "evalue": 0.0,
  "evalue_ci_lower": 0.0,
  "rosenbaum_gamma_breakpoint": 0.0,
  "dowhy_refuters": {
    "random_common_cause": {
      "new_effect": 0.0,
      "p_value": 0.0,
      "status": "ran",
      "error": null
    },
    "placebo_treatment_refuter": {
      "new_effect": 0.0,
      "p_value": 0.0,
      "status": "ran",
      "error": null
    }
  }
}
```

(The pre-3b.14 `refuter_results: [...]` array-of-records shape is
deprecated; the `dowhy_refuters` map-keyed-by-method shape replaces
it. Both shapes are tolerated for backward compatibility with 3b.5 /
3b.7 / 3b.11 artifacts, but new runs MUST emit the map shape so the
`status` field is per-refuter.)

## Source provenance

Canonical source: `docs/v3_0_causal_skill_specification.md` §3.5
(G5 per-skill specification, including the E-value interpretation
table, the forbidden "robust" phrase, the Rosenbaum-bounds recipe
for matched designs, and the DoWhy refuter list with the count
requirement).
