---
name: design-selection-memo
layer: methodology
description: Consume the deterministic Design Feasibility Report; emit a design_memo in research_spec; never overrule a deterministic infeasibility.
trigger_keywords:
  - design
  - feasibility
  - identification
  - memo
  - strategy
applicable_task_types: []
applicable_datasets: []
applicable_stages:
  - ProblemFormulator
priority: 1
references_skills: []
resources: []
version: "1.0"
rule_severity: mandatory
---

# Design-Selection Memo (Arc D)

The user message contains a **"## Design Feasibility Report
(deterministic)"** section produced by the orchestrator's design
selector. It is data, not opinion.

## Rules

1. Add a `design_memo` object to `research_spec`:

   ```json
   "design_memo": {
     "chosen_design": "causal_soo|causal_itr|prediction",
     "feasibility_evidence": "1-2 sentences citing the report",
     "rejected_alternatives": [
       {"design": "rd", "reason": "<the report's reason, restated>"}
     ]
   }
   ```

2. `chosen_design` MUST be consistent with the report's recommendation
   and the run's task type. You may argue nuance in
   `feasibility_evidence`; you may NOT declare a design feasible that
   the report marks infeasible — the predicates are data checks
   (missing cutoffs, no curated instruments, single cohort) that
   argumentation cannot cure.
3. `rejected_alternatives` must list every stronger quasi-experimental
   design the report marked infeasible, with its reason. Stating what
   the study could NOT do and why is part of the contribution — the
   Writer carries it into the paper's identification section.
4. When the report recommends a different task type than the run is
   configured for, do not switch designs mid-run: note the tension in
   `potential_limitations` and proceed with the configured task type.

## Source provenance

Authored in V3.2 Arc D (D3); consumes `src/design_selector.py` output.
