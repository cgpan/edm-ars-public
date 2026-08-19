# V4 Psychometrics Arc — Wave-1 Report (2026-07-09)

**Outcome: BOTH live papers gate-PASSED with "Accept" on the first
bounded attempt each.** Paper 1 set a new all-time best (7.5; prior 7.0).

| Paper | Dataset | Battery | LSAR (median of 3) | Verdict |
|---|---|---|---|---|
| 1. Math self-efficacy + utility: invariance/DIF across sex & SES | HSLS:09 (7 S1 items) | P1 P2 P3 P5 P6 | **7.5** [7.1, 7.5, 7.7] | **Accept — new record** |
| 2. Math self-efficacy GRM calibration + reliability + sex DIF | ELS:2002 (5 BYS89 items) | P1 P2 P3 P4 P5 | **7.3** [6.1, 7.3, 7.4] | **Accept** |

## Findings (both real, both honest)

- **Paper 1**: 2-factor CFA CFI .991 / RMSEA .048; α .837 / ω .922;
  **zero DIF flags and SCALAR invariance across both sex and SES
  extremes** — measurement-equivalence evidence for two scales EDM
  studies routinely consume as features. Critic PASS @ 8 clean.
- **Paper 2**: GRM discriminations 3.4–4.4, marginal reliability .906,
  ω .933, no sex DIF; CFA fit honestly reported as mixed (RMSEA > .06
  with named candidate sources) — the honesty survived review at 7.3.

## Build (commits b5ca800..this; skills 65→67)

R bridge (`Rscript --vanilla` + JSON; certified `r_helpers/` only) —
lavaan CFA/invariance ladder, mirt GRM, ordinal-logistic DIF with
McFadden-rescaled bands; certification gate ALL PASS (CFA loading err
.016, GRM a-err .066, DIF hit 1.0/FP 0.0, invariance null-honest) and
running live in the test suite. `psychometrics` task type: split-less
items contract, four agent variants, measurement-protocol skill
(psy_01–08), item banks curated in both registries.

## F-items

- **F-P1** (fixed): the split-sanity pre-flight fired on the split-less
  task type — exemption widened + regression test.
- **F-P2** (fixed): R bridge unimportable from flat output-dir execution
  — bridge copied beside helpers, flat-import fallback, orchestrator
  exports EDM_ARS_R_HELPERS; verified by simulating the exact context.
- **F-P3** (open, minor): paper 1 set group_mean_comparison_permitted
  = False despite scalar invariance holding (conservative under-claim).
  Harvest: a deterministic consistency check license==f(ladder) in
  pre-critic.

## Wave 2 (reserved)

P7 CDM (R `CDM` package installed) awaits ASSISTments onboarding
(backlog A1) for its Q-matrix. Ledger now 10 papers; five task types
all gate-passing: psychometrics 7.5/7.3, causal_soo 7.0, causal_did
7.0, causal_itr 6.7, prediction 6.6.
