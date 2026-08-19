---
name: paper-writing-style-rules
layer: writing
description: Voice, number formatting, terminology, causal language, and LaTeX font-declaration rules for paper prose.
trigger_keywords:
  - style
  - voice
  - prose
  - causal
  - apa
applicable_task_types: []
applicable_datasets: []
applicable_stages:
  - Writer
priority: 2
references_skills: []
resources: []
version: "1.0"
rule_severity: mandatory
---

# Paper Writing Style Rules

These apply to every paper section — Introduction through Discussion.

## Voice and prose

1. **Active voice wherever possible.** "XGBoost outperformed Logistic
   Regression" — not "XGBoost was outperformed by". Active voice is
   shorter and easier to read; reviewers prefer it.
2. **"Students" not "subjects" or "observations".** The units of
   analysis are people, and the paper should refer to them as such.
3. **Hedge only when uncertain.** If a finding is robust and
   replicated, state it directly. Do not add unnecessary hedges to every
   sentence.
4. **Honest, specific limitations.** Incorporate the Critic's
   limitation feedback and the dataset-specific limitation skills
   verbatim. Vague boilerplate ("future work is needed") is not
   acceptable.

## Numbers and statistics

1. **Precise numbers**: `AUC = 0.82, 95\% CI [0.79, 0.85]` — always
   include the unit and the CI when reporting a primary metric.
2. **Connect every number to educational meaning.** Do not report a
   number without interpretation. If you write "AUC = 0.82" you also
   write what that means for identifying at-risk students.

## Causal language — DO NOT USE for correlational findings

Prediction is not causation. Never write "X causes Y", "X leads to Y",
or "X results in Y" for observational findings. Use:

- "associated with"
- "predicts"
- "is a significant predictor of"

This rule has no exceptions in EDM-ARS papers because every analysis
is observational.

## Citations

`\cite{}` produces APA 7 in-text format via the `ACM-Reference-Format`
bibliography style (set by the template). Do not manually format
author-year citations.

## LaTeX font-declaration rule

Declarations like `\small`, `\footnotesize`, `\large`, `\itshape` are
*declarations*, not commands that take arguments. The declaration must
be **inside** the group it scopes:

- CORRECT: `{\small text}` or `\noindent{\small text}`
- WRONG: `\small{text}` or `\noindent\small{text}` — leaks into
  surrounding text

The same rule applies to all size and shape declarations.

## Source provenance

Canonical source: `agent_prompts/writer.yaml` §"Writing Style Rules" +
§"Font/size declarations".

Merged content from: none — single-sourced.
