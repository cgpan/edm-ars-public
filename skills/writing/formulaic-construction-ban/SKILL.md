---
name: formulaic-construction-ban
layer: writing
description: Ban list of AI-formulaic constructions (stock phrases, rule-of-three padding, boilerplate limitations, inflated significance) + concreteness requirements.
trigger_keywords:
  - style
  - formulaic
  - phrasing
  - boilerplate
  - prose
applicable_task_types: []
applicable_datasets: []
applicable_stages:
  - Writer
priority: 1
references_skills:
  - paper-writing-style-rules
resources: []
version: "1.0"
rule_severity: mandatory
---

# Formulaic-Construction Ban (Arc S)

Reviewers discount prose that reads machine-generated. These
constructions are BANNED in the paper body:

## Stock phrases (never use)

- "delve", "delves into"
- "plays a crucial/pivotal/vital role"
- "It is important to note that", "It is worth noting that"
- "In today's rapidly evolving educational landscape"
- "groundbreaking", "cutting-edge", "paradigm shift"
- "shed light on", "paves the way for"
- "In conclusion," as a paragraph opener (the section header already
  says it)
- "More research is needed" as a bare sentence — every future-work
  claim names WHAT research and WHY it would change the conclusion

## Structural tics (avoid)

- **Rule-of-three padding**: "X, Y, and Z" triplets in every other
  sentence. Lists earn their place or become prose.
- **Negative parallelism overuse**: "not only ... but also" at most
  once per paper.
- **Uniform paragraph openers**: consecutive paragraphs must not all
  begin with the same construction ("The results show... The results
  also indicate... The results further suggest...").
- **Hedge stacking**: one hedge per claim maximum ("may potentially
  suggest" → "may suggest").
- **Em-dash chains**: at most one em-dash clause per paragraph.

## Concreteness requirements (always)

- Every comparative claim carries its numbers inline: not "the rule
  substantially outperformed", but "the rule's value exceeded
  treat-none by 0.031 [0.016, 0.046]".
- Limitations are SPECIFIC to this study's data and design — a
  limitation paragraph that could be pasted into any EDM paper is
  boilerplate and must be rewritten (the HSLS limitation skills give
  the study-specific content).
- At least one sentence in the Discussion says what a named actor
  (teacher, counselor, district) would DO differently.

## Source provenance

Authored in V3.7 Arc S per the v4 roadmap (internal) §6 (the "signs of AI
writing" genre applied to EDM-ARS output).
