---
name: prediction-research-question-design
layer: task-type
description: Gap-driven framing rules; contrast framing; surprising-predictor emphasis; novelty score calibration.
trigger_keywords:
  - research
  - question
  - questions
  - novelty
  - gap
  - contrast
  - framing
applicable_task_types:
  - prediction
applicable_datasets: []
applicable_stages:
  - ProblemFormulator
priority: 1
references_skills: []
resources: []
version: "1.0"
rule_severity: mandatory
---

# Prediction Research Question Design

The ProblemFormulator's job is not just to pick an outcome and a
predictor list — it is to design a question that produces NEW
KNOWLEDGE. Every EDM prediction study can predict X; what matters is
what new thing this particular study tells the reader.

## Gap-driven framing (CRITICAL)

The research question must identify a SPECIFIC GAP in the existing
literature, not just ask "Can we predict X?".

**REJECTED patterns (too generic):**
- "Can ML models predict college attendance from 9th-grade factors?"
- "Can we predict dropout using HSLS:09 data?"
- "What predicts science GPA in high school?"

**ACCEPTED patterns (gap-driven):**
- "Do non-cognitive factors (math identity, school belonging) predict
  college attendance BEYOND what academic achievement and SES explain?"
  *(gap: prior work controls for achievement/SES but omits identity
  constructs)*
- "Is the relative importance of academic vs. motivational predictors
  for STEM enrollment stable between 9th and 11th grade, or does it
  shift?" *(gap: longitudinal shift in predictor importance is
  unstudied)*
- "Do models trained on the full HSLS:09 sample show differential
  predictive validity for first-generation college students vs.
  continuing-generation students?" *(gap: fairness evaluation
  specifically for first-generation status)*

The `expected_contribution` field must state the gap explicitly. "This
study adds to the literature by applying ML to HSLS:09" is **NOT** an
acceptable contribution statement.

## Contrast framing

The research question should imply a CONTRAST or COMPARISON that makes
the answer non-obvious:

- "What predicts X?" — obvious answer (prior achievement + SES) — bad.
- "Does school belonging predict X ABOVE AND BEYOND achievement and
  SES?" — answer not obvious — good.

If a reviewer can predict your answer from the question alone, the
question is uninteresting.

## Surprising-predictor emphasis

If the predictor set includes attitudinal / non-cognitive constructs
(math identity, science identity, school belonging, self-efficacy),
the research question should center them as the interesting
component, not bury them in a generic predictor list. These constructs
are what differentiate longitudinal-survey datasets like HSLS:09 from
administrative records — use them.

## Predictor-set rationale coherence

The predictors should form a theoretically coherent set, not a
kitchen-sink selection. If the question is about non-cognitive
predictors, the predictor set should include specific non-cognitive
constructs with achievement/SES as controls — not 20 variables of
every type.

## Novelty self-assessment calibration

`novelty_score_self_assessment` is a 1–5 integer, your honest
self-assessment.

| Score | Meaning |
|---|---|
| 1–2 | Replicates a known finding | (Auto-rejected — regenerate.) |
| **3** | Applies known methods to a known dataset with a slightly different variable set | **Minimum acceptable.** |
| 4 | Genuine gap or contrast | Aim here. |
| 5 | Question that would surprise a reviewer | Stretch goal. |

If you cannot justify a score of 4, redesign the question. Do NOT
inflate the score — the Critic checks alignment between the score and
the actual question, and inflated scores produce a credibility hit.

## When findings memory is available

Research questions MUST build on prior findings rather than replicate
them. If prior runs found that "SES is consistently the top
predictor," do NOT produce another study confirming this. Instead,
ask: "In what subpopulations does SES lose its predictive dominance?"
or "Does SES predict through different pathways for different
outcomes?" See `findings-memory-novelty-cross-run` for the operational
protocol.

## Mandatory: No redundant composites in predictor_set

Two variables are "redundant composites" when one is a transformation
of the other (binning, scaling, polynomial expansion, percentile
ranks). Including both adds no information and inflates feature count,
which can mislead SHAP attribution and tuning. It also stresses the
DataEngineer's downstream code — discrete + continuous-counterpart
pairs trip `pd.qcut` in stratification helpers.

**Disallowed pairings (HSLS:09 specific examples):**

- `X1SES` + `X1SES_U` — same construct, different scaling
- `X1SES` + `X1SESQ5` — continuous + its quintile binning
- `X1TXMTSC` + `X1TXMTSCOR` — raw vs IRT-scaled math score
- Any continuous variable + its `*Q5`, `*Q4`, or `*Q3` quintile /
  quartile / tertile counterpart
- Any standardized score + its raw counterpart with the same root name

**General rule:** if two variables have the same root prefix
(everything before a trailing `_U`, `Q5`, `Q4`, `Q3`, `R`, `_STD`,
`SCOR` modifier), include only one. Prefer the continuous form unless
the research question specifically requires the binned form.

**Validation checklist before emitting `research_spec.json`:**

1. Build the set of "root prefixes" by stripping the suffixes above
   from each variable name in `predictor_set`.
2. Check that every root prefix appears at most once.
3. If any pair shares a root prefix per the rule above, drop one and
   document the choice in that predictor's `rationale` (e.g.,
   "X1SESQ5 dropped in favor of continuous X1SES — see no-redundant-composites rule").

This rule applies even when both forms appear in `subgroup_analyses` —
the redundancy in the predictor matrix is the problem, not the
subgroup labels.

## Other validation rules (from the agent prompt)

These also apply but are mostly mechanical and less prone to LLM
under-application:

1. **Temporal ordering**: every predictor's wave must come strictly
   before the outcome's wave. See `hsls09-temporal-ordering`.
2. **Sample size**: estimated `analytic_n ≥ 10,000` after listwise
   deletion based on registry `pct_missing` values. (Relaxed for
   structural-MNAR outcomes — see `hsls09-structural-mnar-outcomes`.)
3. **Predictor rationale**: every predictor must have an educational
   rationale. "It is available in the dataset" is not a rationale.
4. **Outcome variable** must be from the registry `outcomes` section.
5. **Protected attributes** used as predictors MUST also appear in
   `subgroup_analyses`. See `subgroup-fairness-analysis`.
6. **No redundant composites**: do not include both `X1SES` and
   `X1SES_U` (one is a rescaling of the other).
7. **Feasibility**: question must be answerable with supervised ML on
   the dataset; do NOT propose causal or experimental designs.
8. **Tier-3 exclusions**: never include weights, flags, IDs. See
   `hsls09-tier3-exclusions`.

## Source provenance

Canonical source: `agent_prompts/problem_formulator.yaml` §Validation
Rules 8-13 + §Canonical Research Questions (L108-L163).

Findings-memory cross-run handling lives in
`findings-memory-novelty-cross-run`. Mechanical validation rules
(temporal ordering, tier-3 exclusions) live in their dataset skills.
