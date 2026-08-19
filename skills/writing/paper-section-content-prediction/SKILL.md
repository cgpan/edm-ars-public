---
name: paper-section-content-prediction
layer: writing
description: Per-section content rules for prediction papers — title patterns, word targets, abstract content rules (substantive finding + named decision, feature-importance de-headlined), section-by-section requirements (SMOTE sentence, ablation table, F2 reporting, within/cross-context contrast), limitations ordering, model-quality caveats, sensitivity reporting.
trigger_keywords:
  - paper
  - section
  - abstract
  - introduction
  - results
  - discussion
  - limitations
  - title
applicable_task_types:
  - prediction
applicable_datasets: []
applicable_stages:
  - Writer
priority: 1
references_skills:
  - paper-writing-style-rules
  - hsls09-multilevel-limitations-paragraph
  - hsls09-survey-weights-limitations-paragraph
resources: []
version: "1.1"
rule_severity: mandatory
---

# Prediction Paper — Per-Section Content Rules

What each section of a prediction paper must contain. Structure and
word budgets come first; the per-section requirements below are
checked by reviewers (and the Critic) item by item.

## Title format

Concise, descriptive, specific — accurately reflecting the research
question, methods, and context. No fixed structure required; useful
patterns include:

- "[Method] for Predicting [Outcome] Using [Key Predictors] in [Context]"
- "Predicting [Outcome] in [Population]: A [Method] Approach"
- "Early Identification of [Outcome]: Insights from [Key Predictors] and [Method]"
- "Who Is at Risk? Predicting [Outcome] with [Method] in [Context]"

Avoid titles that are generic or interchangeable with other EDM papers.

## Structure and word targets

| Section | Target words |
|---|---|
| Abstract | 200–300 |
| 1. Introduction | 1200–1800 |
| 2. Related Work | 800–1200 |
| 3. Methods (all subsections) | 1200–2000 |
| 4. Results (all subsections) | 1000–1600 |
| 5. Discussion (all subsections) | 1000–1600 |

## Abstract — required content

The word budget alone is not sufficient. Three content rules
(V5 Arc T H2; evidence base: 34 EDM/JEDM/JLA anchor papers, a
1,101-abstract policy/ed-psych venue counter-corpus, and 30 AERA Open
full texts — see `docs/v5_arc_t_h2_capability_roadmap.md` §1–2):

1. **The abstract states the substantive finding and its use.
   Feature-importance output (SHAP rankings, top-predictor lists) is
   supporting evidence, never the stated contribution.** Computing
   SHAP is fine; promoting the ranking to the abstract or title is
   the defect. This rule is load-bearing, not taste:
   0 of 1,135 measured abstracts (0/34 anchors, 0/1,101
   counter-corpus) and 0 of 30 AERA Open full texts headline a
   feature-importance ranking. No measured venue — computational EDM,
   education policy, or educational psychology — publishes an
   importance ranking as the takeaway. Convert the ranking into
   either:
   - a substantive claim ("models distinguish X from Y with up to
     88% accuracy, indicating detectable disparities"), or
   - a named decision ("a focus-of-attention tool for admissions
     committees when standardized scores are unavailable").

2. **The final abstract sentence names the specific practice,
   decision, or design the result feeds — in concrete terms.**
   Name an actor, a decision, and what changes: "these results
   suggest reweighting engagement indicators in ninth-grade advising
   contexts", not "these findings have implications for education".
   Anchor abstracts do this in 7/34; ours in 1/13 — and the material
   usually already exists in the Discussion, so promote it. The
   sentence NEVER overclaims beyond the evidence: no "this will
   transform education", no deployment claim the study did not test.

3. **When the split is school-aware (see the Results rule below),
   the abstract may claim cross-context evaluation** — the headline
   metric is then computed over students in schools
   never seen during training, which is a cross-context
   generalization estimate. Make this claim only when
   `data_report.json` records a successful school reconstruction and
   the school-aware split warning; never for a random split.

## Introduction

Motivate the research question educationally and practically. State
the gap in the literature. Introduce the dataset. State contributions
clearly. End with a roadmap sentence.

## Related Work

Cite ≥ 5 papers from `literature_context.papers`. For each, briefly
summarize its contribution (1–2 sentences) and position the present
study against it. Conclude with an explicit statement of how this
study extends the literature.

## Methods — Models

List all model families present in `results.json.all_models`. If MLP
is absent (disabled by configuration), describe 4 individual models +
StackingEnsemble (5 total) and do not mention MLP. If SMOTE was
applied, include this sentence (verbatim or near-verbatim): "To
address class imbalance, we applied Synthetic Minority Over-sampling
Technique (SMOTE) to the training set only, preserving the original
distribution of the test set for unbiased evaluation."

## Results — Model Comparison

- All models' primary metric in a booktabs table.
- Best model with 95% CI in the exact format
  `metric = X.XX, 95\% CI [X.XX, X.XX]`.
- For imbalanced classification: also report F2 and Balanced Accuracy
  (table or text) and note that Accuracy alone is misleading for
  imbalanced data.

## Results — within/cross-context contrast (school-aware splits)

When the train/test split is school-aware, the Results section states
that the headline test metric is computed over students in schools
not seen during training. When the results ALSO contain a
within-context (random-split) estimate of the same metric (e.g. a
split-contrast field in `results.json` or paired rows in
`model_comparison.csv`), report both numbers side by side with their
difference and its CI — the cross-context replication is
invisible unless the contrast is printed. Never invent a comparison
number the analysis did not compute; report only what the results
artifacts contain.

## Results — Ablation (only when `results.ablation` is non-null)

Omit the subsection entirely when `ablation` is null. When present:

- Comparison table: Model | AUC (No SMOTE) | AUC (SMOTE) | Delta.
- Interpretation paragraph: which models benefited most and why
  (linear vs tree-based sensitivity to imbalance).
- SMOTE metadata: minority share before/after, training-set size change.
- Whether SMOTE improved recall/F2 at the cost of precision.

## Discussion

- Connect every statistical finding to educational meaning.
- Address each limitation from `data_report.warnings` and the Critic's
  issues honestly.
- NO causal language: "associated with", "predicts" — never "causes"
  or "leads to" (see `paper-writing-style-rules`).
- Future work must be concrete, not generic.
- For imbalanced classification: discuss why Accuracy misleads for
  at-risk identification; emphasize Recall/F2 for early-warning use;
  note SMOTE calibration effects and deployment implications.

## Limitations — required content and ordering

Most consequential first. Must include, when applicable:

1. Multilevel structure paragraph (reconstruction, ICC, clustered CIs
   — see `hsls09-multilevel-limitations-paragraph`).
2. Survey-weights paragraph (see
   `hsls09-survey-weights-limitations-paragraph`).
3. Sensitivity-analysis results (see below).
4. Model-quality caveats (see below).
5. Critic feedback items.
6. Remaining `data_report.warnings`.
7. Automated-generation disclosure (if not already in Methods).

## Sensitivity analysis reporting (when `results.sensitivity_analysis` is non-null)

1. State which high-missingness variables were excluded.
2. Report whether the primary metric changed by more than 5%.
3. Report whether the top-5 SHAP features changed.
4. State the conclusion (robust / not robust).

This is NOT optional — reviewers specifically look for it.

## Model-quality caveats

If `results.shap_skipped` is true: state clearly that SHAP analysis
was not conducted because no model met the minimum performance
threshold, and that model-comparison results should be interpreted
with caution. If any models failed the quality gate
(`results.model_quality_gate`), name them and why.

## Source provenance

Harvested in V4 Arc H / Phase 3b.24 from
`agent_prompts/writer.v1.yaml.bak` §Title Format, §Paper Structure and
Word Counts, §per-section content rules (L222–L289), §Sensitivity
Analysis, §Model Quality Caveats, and the DISCUSSION_LIMITATIONS
ordering list. Prediction-task-specific by design — causal papers get
their section conventions from the causal writing path
(`writer_causal_soo.yaml` + causal methodology skills).

v1.1 (2026-08-06, V5 Arc T H2 framing fixes): added the
Abstract — required content section (VF2-03 feature-importance
de-headlining with the 0/1,135 evidence; VF2-07 tier 1
final-sentence decision naming, anchors 7/34 vs ours 1/13) and the
Results within/cross-context contrast rule for school-aware splits
(VF2-06). Evidence and reframing templates:
`docs/v5_arc_t_h2_capability_roadmap.md` §1–2. Enforced downstream by
the manuscript linter's abstract-content checks
(`src/manuscript_linter.py`).
