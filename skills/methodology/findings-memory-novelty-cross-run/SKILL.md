---
name: findings-memory-novelty-cross-run
layer: methodology
description: When findings memory is available, build research questions on prior runs (don't replicate); Critic adds an optional novelty review.
trigger_keywords:
  - findings-memory
  - prior-runs
  - novelty
  - replication
applicable_task_types: []
applicable_datasets: []
applicable_stages:
  - ProblemFormulator
  - Critic
priority: 3
references_skills: []
resources: []
version: "1.0"
---

# Findings Memory and Cross-Run Novelty

When `## Findings Memory Summary` appears in the user message, the
ProblemFormulator and Critic both treat it as authoritative context about
what prior runs have already produced. The goal is to keep each run
contributing new knowledge instead of restating existing findings.

## ProblemFormulator obligations

When findings memory is present:

1. Treat the memory as a **catalogue of what has already been studied**.
   Do not replicate prior research questions verbatim; build on open
   questions or explore unstudied outcomes.
2. If prior runs surfaced strong predictors (e.g., "X1TXMTSCOR appeared
   frequently"), you may include them — but the **combination** of
   outcome + predictor set should be meaningfully novel.
3. Prefer outcomes listed under `## Studied Outcomes (already
   investigated in prior runs)` only if the user prompt specifically
   requests them or no viable alternative exists.

When `## Prior Candidate Specs (already generated this session)` is
present (multi-candidate generation mode):

1. Your spec MUST differ meaningfully from each prior candidate. Choose a
   different outcome variable, a different theoretical framing, or a
   substantially different predictor set.
2. Do not simply swap one predictor for a near-synonym. Aim for genuine
   conceptual diversity.

## Critic novelty_review section

When findings memory is present, the Critic adds an optional
`"novelty_review"` key to its JSON output (after `revision_instructions`):

```json
{
  "novelty_review": {
    "score": 7,
    "compared_to_prior_runs": "This study examines <outcome> using <predictors>. Prior runs focused on <other-outcome>, so the outcome is novel, but the predictor set overlaps substantially with run_<id>.",
    "contribution_builds_on_memory": true
  }
}
```

Field semantics:

- `score` (1–10): how novel this run is relative to the accumulated
  memory.
- `compared_to_prior_runs`: 1–3 sentences comparing to specific prior
  runs.
- `contribution_builds_on_memory`: `true` if the research question
  extends or responds to open questions or strong predictors from prior
  runs.

This section is **optional**. Omit it entirely if no `## Findings Memory
Summary` is in the user message. Do not include `"novelty_review": null`
— simply omit the key.

## Verification rules (Critic checklist `pf_08`, `pf_09`)

When findings memory is enabled (the orchestrator sets a flag when the
memory file is non-empty):

1. `pf_08` (minor): if the outcome variable and a substantially similar
   predictor set have already been studied in a prior run (per
   `studied_outcomes`), flag as minor unless the research question offers
   a meaningfully different framing or predictor combination. Use
   `novelty_review.score` as supporting evidence.
2. `pf_09` (minor): verify that the research question or
   `expected_contribution` references open questions or strong predictors
   from the findings memory when relevant. Aspirational — minor, not
   blocking.

## Source provenance

Canonical source: `agent_prompts/problem_formulator.yaml` §"Using
Findings Memory" + §"Generating Diverse Candidates".

Merged content from:
- `agent_prompts/critic.yaml` §"Optional: Novelty Review Against Prior
  Runs"
- `data_registry/evaluation_rubrics/methodological_checklist.yaml` items
  `pf_08`, `pf_09`
