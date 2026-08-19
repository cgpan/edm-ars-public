# EDM-ARS V3 Changelog — The Causal-Inference Capability Phase

**Tag**: `v3.0.0` (2026-07-08). Previous: `v2.0.1` (skill architecture, DataEngineer slim).
**Span**: V2.1 slim migration → Arc H hardening → Arcs R/D/G/Q/S/L/E → three-phase
program (ELS cross-dataset, causal_did, median gate) → two-stream program
(prediction rigor, causal_did v2).

## Headline capabilities added since v2.0.1

1. **All five agent prompts slim + skill-injected** (V2.1 complete: OutlineAgent
   93→25, PF 207→95, Critic 235→140, Writer 485→77, Analyst 605→115 lines).
   65 skills; mandatory-severity system; per-layer caps with mandatory bypass.
2. **Three causal task types, all gate-passing**:
   - `causal_soo` (M1 regression adjustment, M2 PSM, M3 IPW, M4 AIPW/TMLE,
     M5 causal forest CATE) — best score 7.0.
   - `causal_itr` (M6 DR policy learning, M7 cross-fitted policy value) — 6.7.
   - `causal_did` (M8 raw gap-in-gaps, M9 composition-adjusted within-group
     AIPW, M10 GBM heterogeneity with contrast-based inference) — 7.0 Accept.
3. **Prediction rigor extensions** — moderation analysis (interaction LRT +
   tertile incremental AUC), grouped SHAP by parent variable, cluster-aware
   paired AUC tests, calibration metrics (Brier/ECE/Cox slope). First
   gate-passing prediction paper (6.6 vs 6.3; prior plateau 6.2).
4. **Cross-dataset capability**: ELS:2002 onboarded (registry + adapter +
   conventions skill + F1SCH_ID school clustering); harmonized ELS×HSLS
   cross-cohort panel (deterministic harmonizer, 16,862 students, harmonized
   race5/pared3/expect_ba/ses_std covariates).
5. **Design-selection intelligence** (Arc D): deterministic feasibility
   predicates (rd/iv/did/soo/itr) + design memo contract + gap miner; causal
   intent routes to the strongest feasible executable design.
6. **Synthetic certification discipline** (standing rule): every estimator
   battery passes a replicated synthetic-DGP gate before live use
   (`scripts/quasi_experimental_gates.py`); certified recipes ship as
   deterministic `analysis_helpers` functions that generated code must call.
   RD (local-quadratic, bias .008) and IV (2SLS + weak-flag honesty 100%)
   are certified and shelf-ready, awaiting suitable data.
7. **Calibrated review gate**: LSAR P25-of-accepted-EDM-papers threshold
   (6.3) + borderline-triggered median sampling (k=3, band ±1.5) —
   live-validated repeatedly; prevented at least one noise-pass and one
   noise-fail.
8. **Anti-formulaic writing** (Arc S): archetype + formulaic-ban skills;
   null-result archetype fired live twice.
9. **Evaluation ledger** (`evaluation/ledger.json`): 8 papers, every live
   task type with a gate-passing exemplar.

## Robustness/hardening highlights (selected F-items, all fixed)

- Post-DE pre-flight: matrix-level causal contract + split-sanity check
  (file-verified, not self-reported) + targeted-retry machinery.
- Dataset-blind skill hazard: school-cluster skills scoped per dataset
  (F-A1); every methodology skill encoding dataset structure must declare
  `applicable_datasets`.
- Copy-paste-safe skill examples: exact signatures with real kwarg names
  (F-S2-KWARG-COPY) — LLMs copy examples verbatim.
- Artifact filenames are contracts: consumers name files exactly
  (F-B1 panel filename).
- Critic malformed-JSON single re-prompt before SPEC §8 abort (F-B2).
- Outline None-safety (F-A3), v1-template fallback (F-A4), deterministic
  bibliography injection (F-A5), DE duplicate-keep-cols rule + stderr
  diagnosis hints (F-A2).
- UTF-8 encoding on all prompt/registry reads (mojibake fix, 15 sites).

## LSAR upgrades in this phase (separate repo, tag `v0.2.0`)

- Provider migration MiniMax → DeepSeek (OpenAI-compatible client;
  MiniMax-era outputs quarantined).
- Anchored calibration: 16 EDM-2024 papers batch-reviewed; accepted-full
  median 6.6, **P25 6.3 adopted as the live gate** (provider-stable);
  test-retest MAD 1.9 measured → motivated the median-sampling gate.
- Stage-1 ingestion robustness (F-LSAR-ONNX-LAYOUT-CRASH): pymupdf4llm
  layout→classic→fitz fallback ladder; plain-text-tolerant abstract/
  section/reference extraction; no more 1-element heading early-return.

## Provider stack

All six EDM-ARS stages + LSAR reviewer on deepseek-v4-pro (user directive
2026-07-03; supersedes gpt-5.4 Analyst/Writer routing and all MiniMax use).

## Score matrix at V3 close

| task type | dataset | LSAR | verdict |
|---|---|---|---|
| causal_soo | hsls09 | 7.0 / 6.4 | pass |
| causal_itr | hsls09 | 6.7 | pass |
| causal_did v2 | els×hsls panel | 7.0 Accept | pass |
| prediction | els_2002 | 6.6 | pass (first) |
| prediction (pre-rigor) | hsls09 / els | 6.2 ×3 | sub-gate history |
| causal_did v1 | els×hsls panel | 3.7 | honest venue-fit reject |
