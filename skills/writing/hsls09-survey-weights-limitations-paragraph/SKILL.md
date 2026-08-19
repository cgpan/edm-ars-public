---
name: hsls09-survey-weights-limitations-paragraph
layer: writing
description: Required HSLS:09 survey-weights limitations paragraph; explains why weights were not applied and what it means for generalizability.
trigger_keywords:
  - weights
  - survey
  - hsls
  - limitations
  - sampling
  - design
applicable_task_types: []
applicable_datasets:
  - hsls09_public
applicable_stages:
  - Writer
priority: 1
references_skills: []
resources: []
version: "1.1"
rule_severity: mandatory
---

# HSLS:09 Survey Weights — Limitations Paragraph

HSLS:09 uses a complex stratified multi-stage probability sampling
design with analysis weights (`W1STUDENT`, `W2W1STU`, `W4W1W2W3STU`,
etc.). The pipeline does not apply these weights. The §Limitations
section must explain why and acknowledge the consequence for
generalizability.

## Required content (write a paragraph that includes ALL of these)

1. HSLS:09 uses a complex stratified multi-stage probability sampling
   design with analysis weights (`W1STUDENT`, `W2W1STU`, `W4W1W2W3STU`,
   etc.).
2. The machine learning models were trained and evaluated WITHOUT survey
   weights because scikit-learn's standard estimators do not support
   complex survey variance estimation (stratification + primary sampling
   unit clustering).
3. Some models (Logistic Regression, Random Forest, XGBoost) accept a
   `sample_weight` parameter, but using weights without proper variance
   estimation produces correctly weighted point estimates with incorrect
   standard errors — which would mislead readers about precision.
4. The reported metrics reflect unweighted sample performance and may
   not generalize exactly to the national population of 9th graders.
5. Future work should use survey-aware ML packages (e.g., weighted
   bootstrap procedures or the `survey` package in R) to produce
   properly weighted estimates.

## Tone

State the limitation directly. Do not hedge with "this might be a
limitation" — it IS a limitation, and acknowledging it precisely is
what distinguishes a credible paper from a sloppy one.

## Source provenance

Canonical source: `agent_prompts/writer.yaml` §"Survey Weights — Why
Not Applied and What It Means".

Merged content from: none — single-sourced.
