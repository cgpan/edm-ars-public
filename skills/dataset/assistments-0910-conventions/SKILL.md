---
name: assistments-0910-conventions
layer: dataset
description: ASSISTments 2009-10 conventions — interaction LOG not survey; original==1 filter, first-attempt dedupe by order_id, structural sparsity (never impute), skill tags → Q-matrix, template_id = CDM item unit.
trigger_keywords:
  - assistments
  - skill builder
  - log
  - cdm
  - q-matrix
applicable_task_types: []
applicable_datasets:
  - assistments_0910
applicable_stages:
  - ProblemFormulator
  - DataEngineer
  - Analyst
  - Critic
  - Writer
priority: 1
references_skills: []
resources: []
version: "1.0"
rule_severity: mandatory
---

# ASSISTments 2009-10 Conventions

## This is LOG data (the three filters, in order)

525,534 rows = problem ATTEMPTS, not students. Every analysis pipeline
starts with exactly these steps:

```python
df = df[df["original"] == 1]                # 1. main problems only
df = df.dropna(subset=["skill_id"])          # 2. skill-tagged only (~15% untagged)
df = df.sort_values("order_id").drop_duplicates(
    ["user_id", "template_id"], keep="first")  # 3. FIRST attempt per item
```

Violations are silent-corruption class: scaffolding rows deflate
difficulty; repeat attempts leak practice effects.

## Item/response conventions (CDM & IRT)

- **Item unit = `template_id`** (same template = same problem
  structure); `problem_id` instances of one template are interchangeable.
- **Response = `correct`** (binary first-attempt) from the deduped frame.
- **Q-matrix** from skill tags: template → its skill(s); with one skill
  per template this is a simple-structure Q-matrix (DINA-appropriate).
  Record Q-matrix provenance (tag-derived, not expert-validated) as a
  limitation.
- Scope floors: skills by user coverage; templates ≥ 300 responses;
  students ≥ 5 items answered. Report the resulting n's.
- The wide student × item matrix is mostly EMPTY BY DESIGN — never
  impute; CDM/mirt handle NA natively.

## Clustering and identifiers

Unit of analysis = student (`user_id`); attempts cluster within
students. `school_id`/`teacher_id`/`student_class_id` are tier-3
(administrative), never predictors. No demographics exist in this
public log — fairness analyses by demographic groups are NOT possible
and the paper must say so rather than improvise proxies.
