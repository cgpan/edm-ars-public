---
name: gap-driven-question-mining
layer: methodology
description: Use the deterministic Gap Matrix — pick a sparse (outcome × method) cell, name it in expected_contribution, scope the claim to the retrieved corpus.
trigger_keywords:
  - gap
  - matrix
  - novelty
  - contribution
  - unstudied
applicable_task_types: []
applicable_datasets: []
applicable_stages:
  - ProblemFormulator
priority: 2
references_skills: []
resources: []
version: "1.0"
rule_severity: mandatory
---

# Gap-Driven Question Mining (Arc D)

The user message contains a **"## Gap Matrix (deterministic)"**
section: retrieved-paper counts per (outcome family × method family)
cell, with sparse cells listed.

## Rules

1. Target a **sparse or thin cell** — a question in a cell already
   dense with retrieved papers needs an explicit differentiation
   argument against those specific papers.
2. `expected_contribution` MUST name the cell in plain words (e.g.
   "no retrieved paper applies targeting/ITR methods to college
   enrollment") and cite the nearest retrieved paper(s) it builds
   beyond.
3. **Scope honestly**: the matrix covers the RETRIEVED corpus only.
   Write "within the retrieved corpus" (or equivalent) — never claim
   a literature-wide first.
4. The matrix complements — never replaces — the gap-framing rules in
   the task-type question-design skills; a sparse cell still needs an
   educationally meaningful question.

## Source provenance

Authored in V3.2 Arc D (D4); consumes `src/gap_miner.py` output.
