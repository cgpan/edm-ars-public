---
name: psychometrics-measurement-protocol
layer: methodology
description: Measurement-study protocol (P1-P6) — decision-rule honesty for invariance/DIF, no group comparisons without scalar invariance, items never imputed, legit-skip population framing; critic rows psy_01-psy_08.
trigger_keywords:
  - psychometric
  - measurement
  - invariance
  - dif
  - irt
  - reliability
  - cfa
  - scale
applicable_task_types:
  - psychometrics
applicable_datasets: []
applicable_stages:
  - ProblemFormulator
  - DataEngineer
  - Analyst
  - Critic
  - Writer
priority: 1
references_skills:
  - r-bridge-execution
resources: []
version: "1.2"
rule_severity: mandatory
---

# Measurement Study Protocol (P1–P6)

## The claims and their decision rules (state the rule with the claim)

| Claim | Rule (verbatim in results AND paper) |
|---|---|
| Reliability | report BOTH Cronbach's alpha (P1) and omega-total (P2); alpha alone is insufficient |
| Structure (P3) | CFA fit judged by CFI/TLI ≥ .95 (adequate ≥ .90), RMSEA ≤ .06, SRMR ≤ .08; MLR robust estimation for 4-point items (ordinality noted as a limitation) |
| Calibration (P4) | GRM for Likert; report discrimination + thresholds + test information + marginal reliability; convergence stated |
| DIF (P5) | flagged iff p_overall < .01 AND McFadden ΔR² ≥ .02 (.05 large — McFadden-scaled bands; certified hit rate 1.0 / false positives 0.0) |
| Invariance (P6) | ladder configural→metric→scalar; a step HOLDS iff ΔCFI ≥ −.01 AND ΔRMSEA ≤ .015 (Chen 2007); report the full ladder table |

| CDM (P7) | `analysis_helpers.psy_cdm(responses_df, q_matrix, attributes, model="compare")` fits DINA AND G-DINA (certified: guess/slip bias < .01): report the comparison block (AIC/BIC both models, LR test, selection). **Single-attribute Q-matrices make the comparison degenerate by construction** (the models coincide) — the helper flags this and retains DINA; the paper states it verbatim. Q-matrix provenance stated (tag-derived = named limitation); mastery claims are population-level prevalence, never individual diagnoses |

## The two structural NEVERs

1. **No group mean comparison without scalar invariance** (or an
   explicitly justified partial-scalar model). If the ladder stopped at
   metric or configural, the paper must say which comparisons are NOT
   licensed — that IS a finding.
2. **Items are never imputed.** FIML (CFA/invariance) and native-NA
   (GRM) handle missingness; DIF models use listwise and report the
   analyzed n. Only all-items-missing rows may be dropped.

## Population honesty (PSY-01)

Item missingness from "legitimate skip" codes means the student was not
enrolled in the subject — the analytic population is *enrolled
students*, and every measurement claim is about that population. The
question, results, and limitations all say so.

## Execution contract

All estimation through `analysis_helpers.psy_*` wrappers (see
r-bridge-execution). Reverse-coded items recoded BEFORE any call, per
the registry item bank. Grouping variables with >2 levels: DIF uses the
two largest levels (named); invariance uses all levels with n ≥ 500 per
group, else the two largest.

## Critic rows (walk every one)

| ID | Item | Severity | Check |
|---|---|---|---|
| `psy_01` | Wrappers used, no raw R / hand-rolled models | critical | Analyst code imports analysis_helpers and calls psy_*; no Rscript/subprocess/sklearn factor hacks. |
| `psy_02` | Decision rules stated verbatim | critical | Every DIF/invariance claim carries its rule (effect band; ΔCFI/ΔRMSEA); p-value-only claims are a violation. |
| `psy_03` | Group-comparison license respected | critical | `group_mean_comparison_permitted` consistent with the ladder result; no mean comparison in the paper without it. |
| `psy_04` | Items never imputed | critical | No imputer touched item columns; missingness reported per item. |
| `psy_05` | Population framing (PSY-01) | major | Legit-skip missingness → enrolled-population claims, stated in question + limitations. |
| `psy_06` | Alpha AND omega | major | Both present; omega from the CFA loadings. |
| `psy_07` | Ladder table complete | major | All fitted steps with deltas reported, not just the verdict. |
| `psy_08` | Ordinality limitation | minor | MLR-on-4-point-items noted; categorical estimation named as future work. |
| `psy_09` | CDM Q-matrix provenance + filters | critical | P7 runs: Q-matrix source stated; original==1 + first-attempt + tag filters applied (log datasets); scope floors reported. |
| `psy_10` | No individual diagnosis claims | critical | CDM mastery reported as prevalence/profiles, never "student X has mastered". |
