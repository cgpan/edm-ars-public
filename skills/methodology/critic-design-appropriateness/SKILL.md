---
name: critic-design-appropriateness
layer: methodology
description: Critic gate — research_spec.design_memo present, consistent with the deterministic feasibility report, rejected alternatives honestly carried.
trigger_keywords:
  - design
  - memo
  - appropriateness
  - identification
  - feasibility
applicable_task_types: []
applicable_datasets: []
applicable_stages:
  - Critic
priority: 2
references_skills:
  - design-selection-memo
resources: []
version: "1.0"
rule_severity: mandatory
---

# Critic — Design Appropriateness (Arc D)

| ID | Item | Severity | Check |
|---|---|---|---|
| `da_01` | `design_memo` present in research_spec | major | Object with `chosen_design`, `feasibility_evidence`, `rejected_alternatives`. Absent → major, target ProblemFormulator. |
| `da_02` | Chosen design matches the run's task type | critical | `design_memo.chosen_design` == the configured task type. A memo claiming a design the pipeline did not execute misrepresents the study. |
| `da_03` | No overruled infeasibility | critical | The memo must not claim RD/IV/DiD feasibility on a dataset whose feasibility report marked them infeasible — argumentation cannot cure missing cutoffs/instruments/cohorts. |
| `da_04` | Rejected alternatives carried with reasons | minor | Each infeasible stronger design appears in `rejected_alternatives` with a substantive reason (the honesty-is-a-feature rule; the Writer needs it for the identification section). |

## Source provenance

Authored in V3.2 Arc D (D3).
