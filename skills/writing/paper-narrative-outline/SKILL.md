---
name: paper-narrative-outline
layer: writing
description: Generate a data-driven paper outline before writing; emphasis triggers expand or compress sections based on actual results.
trigger_keywords:
  - outline
  - structure
  - sections
  - narrative
  - emphasis
applicable_task_types: []
applicable_datasets: []
applicable_stages:
  - OutlineAgent
  - Writer
priority: 2
references_skills: []
resources: []
version: "1.1"
---

# Paper Narrative Outline

The OutlineAgent runs before the Writer and produces a paper-specific
outline that adapts to what the data actually shows, rather than
following a fixed template. The Writer consumes the outline and
generates the body in one cohesive pass.

## Output schema

```json
{
  "narrative_hook": "One sentence describing the paper's most interesting finding or angle",
  "sections": [
    {
      "id": "introduction",
      "title": "Introduction",
      "emphasis": "standard",
      "word_target": 900,
      "guidance": "Open with [hook]. Frame around the gap: [gap statement]."
    },
    {
      "id": "related_work",
      "title": "Related Work",
      "subsections": [
        {"title": "Descriptive subsection title", "guidance": "Cover [specific topic]"}
      ],
      "emphasis": "standard",
      "word_target": 600
    },
    {
      "id": "methods",
      "title": "Methods",
      "subsections": [],
      "emphasis": "compressed|standard|expanded",
      "word_target": 1200
    },
    {
      "id": "results",
      "title": "Descriptive Results Title",
      "subsections": [
        {
          "title": "Descriptive subsection title reflecting actual finding",
          "emphasis": "expanded",
          "guidance": "This is the key result — give it extra depth"
        }
      ],
      "word_target": 1500
    },
    {
      "id": "discussion",
      "title": "Discussion",
      "subsections": [],
      "word_target": 1200
    }
  ]
}
```

## Outline design rules

1. **Sections always present**: Introduction, Related Work, Methods,
   Results, Discussion.
2. **Subsection titles should be descriptive, not generic.** Instead of
   "Feature Importance", write "SES and Prior Achievement Dominate
   Predictions, But School Belonging Surprises".
3. **Emphasis allocation** based on findings:
   - If all models perform within 2% of each other: compress Model
     Comparison, expand SHAP.
   - If one model dramatically outperforms: expand Model Comparison.
   - If a surprising feature appears in top-5 SHAP: promote to its own
     Results subsection.
   - If subgroup gaps > 5%: promote Subgroup Analysis to a prominent
     Results subsection.
   - If sensitivity analysis changed conclusions: add a dedicated
     Robustness section.
   - If ICC is non-negligible (≥ 0.05): add a "School-Level Variance"
     subsection in Results.
4. **Methods can be compressed.** "Data and Sample", "Predictors and
   Missing Data", "Models and Evaluation" is often enough.
5. **Discussion subsections should make arguments, not follow a
   checklist.**
6. **The narrative_hook drives the paper.** Identify the single most
   interesting finding.
7. **The narrative_hook is a substantive finding or a named decision,
   never a feature-importance ranking.** 0 of 1,135 measured abstracts
   (34 EDM/JEDM/JLA anchors + 1,101 policy/ed-psych venue abstracts)
   headline an importance ranking. Rankings are supporting evidence
   inside Results subsections — see the abstract content rules in
   `paper-section-content-prediction`.

## Emphasis triggers

The orchestrator pre-computes a small set of emphasis triggers from
`results.json` and `data_report.json` (e.g., "subgroup_gap_large":
true, "icc_nonneg": true) and includes them in the user message. Use
these to determine which findings deserve expanded treatment and which
can be compressed.

## Output format

Output ONLY valid JSON wrapped in a ```json code block. No prose before
or after.

## Source provenance

Canonical source: `agent_prompts/outline_agent.yaml` (entire file).

Merged content from: none — single-sourced. The OutlineAgent's role and
JSON I/O contract live in the agent prompt itself; this skill is the
substantive guidance.

v1.1 (2026-08-06, V5 Arc T H2): added outline rule 7 — the
narrative_hook is never a feature-importance ranking (VF2-03,
0/1,135 evidence in the v5 capability roadmap (internal) §1).
