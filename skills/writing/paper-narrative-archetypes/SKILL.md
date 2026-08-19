---
name: paper-narrative-archetypes
layer: writing
description: Select a narrative archetype from the RESULTS SHAPE (null-result, heterogeneity/targeting, methods-comparison, effect-with-implications) and structure the paper around it — papers must not all read the same.
trigger_keywords:
  - narrative
  - archetype
  - story
  - framing
  - structure
applicable_task_types: []
applicable_datasets: []
applicable_stages:
  - OutlineAgent
  - Writer
priority: 1
references_skills:
  - paper-writing-style-rules
resources: []
version: "1.0"
rule_severity: mandatory
---

# Narrative Archetypes (Arc S)

A paper's structure should follow its FINDINGS, not a fixed template.
Pick ONE archetype using the deterministic selection rule, name it in
the outline's `narrative_hook`, and let it drive section emphasis.

## Selection rule (apply in order, first match wins)

1. **Null-result paper** — the headline CI covers zero (prediction:
   best model ≈ baseline; causal: effect CI spans 0; ITR: value-gain
   CI spans 0). The null IS the story: lead with it, devote Discussion
   to what the well-estimated null rules out and for whom.
2. **Heterogeneity / targeting paper** — subgroup gaps > 5%, CATE
   spread is substantial, or an ITR rule with a positive gain. The
   average is the backdrop; the WHO-differs question is the spine.
3. **Methods-comparison paper** — estimators/models materially
   disagree (beyond 2 SE) or a robustness battery is the most
   informative output. The disagreement is the finding; explain WHY
   the methods diverge on this data.
4. **Effect-with-implications paper** — a clear, robust effect.
   Classic arc, but the Discussion must earn it: concrete decisions a
   school/policymaker would change.

## Per-archetype emphasis

| Archetype | Expand | Compress |
|---|---|---|
| Null-result | Discussion (what the null excludes), power/precision framing | Related-work tour |
| Heterogeneity/targeting | Subgroup/rule sections, fairness of targeting | Average-effect exposition |
| Methods-comparison | Methods + the divergence anatomy | Implications |
| Effect-with-implications | Implications, mechanisms | Methods minutiae |

## Anti-sameness rules

- Two papers with different archetypes must not share their opening
  paragraph structure. Do not open every abstract with "In this
  study, we...".
- The title should signal the archetype (a null-result paper's title
  says so — "No Detectable...", "Little Evidence That...").
- Never force archetype 4's triumphant arc onto shapes 1–3; that is
  the formulaic failure this skill exists to prevent.

## Source provenance

Authored in V3.7 Arc S per docs/v4_roadmap.md §6; extends the
OutlineAgent emphasis-trigger mechanism.
