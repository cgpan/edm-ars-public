# V4 Wave-2 Report — Journal Format, ASSISTments, CDM, LSAR Journal Mode (2026-07-10)

All four user jobs delivered and validated by ONE integrated live run:
a **DINA cognitive-diagnosis paper on ASSISTments, written as a
JEDM-style APA-7 journal manuscript (biber-compiled), reviewed by LSAR
in journal associate-editor mode with an honest ADVISORY verdict**.

## 1. Journal writing (from the user's APA_7th_Template.zip)

`templates/paper_template_journal.tex` (placeholder-derived, preamble
protected, fixed authors) — biber compile-proven; `writer.venue_format:
journal` selects it; `journal-apa7-style` skill (mandatory) carries the
~8000-word IMRaD depth table + \parencite/\textcite rules; compile_latex
and the review-gate compiler are biblatex-aware.

## 2. ASSISTments 2009-10 onboarded

Tier-1 registry + adapter + log-data conventions skill (the three
filters: original==1 → skill-tagged → first attempt per user×template;
template_id = item unit; skill tags → Q-matrix; structural sparsity
never imputed). 525,534 attempts / 4,217 students / 123 skills.

## 3. CDM (P7) enabled and certified

`cdm_fit.R` (R CDM package DINA/G-DINA) + `psy_cdm` wrapper +
`cdm_gate` certification **PASSED** (guess bias .010, slip .012,
prevalence error .006). Protocol skill v1.1 rows psy_09 (Q-matrix
provenance) + psy_10 (no individual diagnoses).

## 4. LSAR journal reviewer (tag v0.3.0 + section fix)

Seven journal venue profiles (JEDM, JLA, JEM, JEBS, MBR, Psychometrika,
JREE — applied-EDM / measurement-methods / quant-methods families),
config-driven venue whitelist, journal AE persona, 60-page sanity cap,
APA-heading section extraction. **Calibration honesty**: journal venues
are explicitly uncalibrated; the EDM-ARS gate runs ADVISORY (full
median-sampled review + would-be verdict, never a fail against an
unanchored threshold).

## The integrated run

- **DINA converged**: 1,586 students × 47 template items × 8 skills;
  mastery prevalence .467 (Percent Of) – .728 (Probability of a Single
  Event); per-item guess/slip; AIC 23,318 / BIC 25,192.
- Manuscript: ~5,900 words (below the 8,000 target — first-shot gap),
  biber references rendered, honest concern carry-through (tag-derived
  Q-matrix, no demographics, structural sparsity).
- **JEDM advisory review: median 5.6 "Borderline"** (samples [3.1, 5.6,
  6.8]); advisory mode logged "would have failed 6.3" without failing
  the run. Median dims: Relevance 8, Clarity 7, Rigor 6; Novelty 4 and
  Ethics 4 drag (single-model DINA without G-DINA comparison; no
  demographic fairness possible on this dataset).

## Findings about the system itself

1. **Journal-mode single-review variance is LARGE** (spread 3.7 vs the
   conference test-retest MAD 1.9) — advisory-only gating was the right
   call, and journal anchor calibration (backlog C4) is now clearly the
   binding constraint on journal-grade evaluation.
2. Word budget under-shoot (~5.9k vs 8k) — writer max_tokens is the
   likely binding constraint for journal length; candidate fix:
   sectionwise generation for journal mode (logged below).
3. Four in-flight F-items found+fixed with regression tests: binary-item
   validation, PF-invented item_columns, gate biber pass, APA
   per-word-bold section extraction.

## Follow-ups logged

- C4 (journal anchors) upgraded in priority by the variance finding.
- NEW backlog: journal-length sectionwise writing; G-DINA comparison arm
  for CDM papers; F (openclaw CLI) recorded per user item 5.
