# V4 Psychometrics Arc — Scope (locked 2026-07-08)

User decisions: R-bridge primary engine; CTT included now, CDM reserved
(P7, ASSISTments wave); two-paper live wave after verified build.

## Method IDs

| ID | Method | Engine | Certification gate |
|---|---|---|---|
| P1 | CTT item analysis (difficulty, item-total discrimination) + Cronbach's alpha | Python | closed-form checks vs known structure |
| P2 | Reliability: McDonald's omega (from CFA loadings) + IRT conditional reliability | R lavaan / mirt | recovery on simulated congeneric data |
| P3 | CFA: single-group fit (CFI/TLI/RMSEA/SRMR), loadings | R lavaan | loading recovery, fit-index sanity on true/misspecified models |
| P4 | IRT calibration: GRM (Likert), 2PL (binary) + item/test information | R mirt | parameter recovery over seeds |
| P5 | DIF: logistic-regression DIF (uniform + non-uniform, effect sizes) | R difR/mirt | known-DIF hit rate + null false-positive honesty |
| P6 | Measurement invariance ladder: configural→metric→scalar, ΔCFI/ΔRMSEA rules, partial invariance | R lavaan/semTools | violation detection + full-invariance null honesty |
| P7 | (reserved) CDM: DINA/G-DINA with Q-matrix | R CDM | wave 2, ASSISTments |

## Architecture rules (carried from V3)

1. **Certified before live**: `scripts/psychometric_gates.py`, replicated
   over seeds, run live in the test suite.
2. **Deterministic helpers only**: generated code calls
   `analysis_helpers.psy_*` wrappers; the wrappers call fixed, certified
   R scripts in `r_helpers/*.R` through `src/r_bridge.py`. Generated code
   NEVER writes raw R.
3. **R bridge**: `Rscript --vanilla` subprocess + JSON files in/out. No
   rpy2. R not in the Docker sandbox → psychometrics runs use the
   subprocess executor (current default). R location resolved: config
   `r_bridge.rscript_path` → common install dirs → PATH.
4. Items with NCES sentinels → NaN before modeling; **items are never
   imputed** — lavaan uses FIML (`missing="fiml"`), mirt handles NA
   natively; missingness reported per item.
5. Contrast/claim honesty analogues: invariance conclusions follow the
   ΔCFI ≤ .01 / ΔRMSEA ≤ .015 decision rules, stated with the rule; DIF
   flags require the effect-size threshold (pseudo-ΔR² ≥ .035 moderate),
   not p-values alone; group mean comparisons FORBIDDEN unless scalar
   (or justified partial-scalar) invariance holds.

## Task type

`psychometrics` — locked-spec entry (like causal): scale_name,
item_columns, response_scale, grouping_vars, method battery, invariance
target. DE contract: `items_analytic.csv` (items + grouping vars, NaN
allowed) + per-item missingness report; no split, no encoding, no
imputation. New skills: r-bridge-execution, psychometrics-measurement-
protocol (mandatory; critic rows psy_01..), writer measurement-paper
shape. Pre-critic: psychometrics branch (no refuters/split checks).

## Item banks (wave 1 data)

- HSLS S1 math attitude items (identity/self-efficacy/interest/utility;
  ~4 scales) + school belonging/engagement items; grouping: X1SEX,
  X1RACE, X1SESQ5.
- ELS BYS math/reading attitude analogues; grouping: BYSEX, BYRACE,
  BYSES1QU.
- Registry extension: `item_banks:` block per dataset (scale → items,
  response categories, reverse-coded flags).

## Two-paper live wave

1. **Invariance/DIF paper (HSLS)**: "Is the mathematics identity /
   self-efficacy scale measurement-invariant across sex, race, and SES?"
   — P3+P5+P6 headline, P1/P2 supporting.
2. **Calibration/reliability paper (ELS)**: GRM calibration + information
   + omega of the ELS math attitude scales, with CTT baseline — P4+P2+P1
   headline (+P5 secondary if items allow).

Gate: existing calibrated 6.3 median gate (journal venues are a separate
arc, backlog C4).
