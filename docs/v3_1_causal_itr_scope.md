# V3.1 Arc R — Optimal Treatment Regimes (`causal_itr`) Scope & Decisions

> Status: R0–R2 implemented 2026-07-03 per `docs/v4_roadmap.md` Arc R.
> This doc records the decisions; the code/skills/tests are the source of truth.

## Estimands and headline

- Headline: **policy value V(π̂)** of a learned rule and its **gain over the best
  constant policy** (max of treat-all / treat-none), cross-fitted, with a
  school-cluster bootstrap CI. NOT a single ATE.
- Honesty contract: gain CI covering zero ⇒ the headline IS "no detectable
  targeting benefit". Regret-vs-oracle language is banned on real data (oracle
  unknown); regret lives only in the synthetic gate.

## Architecture decisions

1. **`CausalITRTemplate(CausalSOOTemplate)`** — ITR = SOO + policy learning. All
   identification machinery (G1–G5, D1 contract, post-DE pre-flight, dummied-
   treatment repair, refuter contract/pcc_c01) inherited by widening the
   task-type gates to the causal family. New spec fields: `rule_covariates`
   (must be ⊆ adjustment_set; observable/actionable at decision time — the ITR
   analogue of temporal ordering), `primary_method: M6`.
2. **No econml.** The sandbox ships sklearn only. M6 uses the standard
   weighted-classification reduction: cross-fitted DR pseudo-outcomes
   (GBM outcome models + logistic propensity, GroupKFold on schools) → shallow
   `DecisionTreeClassifier(max_depth≤2, min_samples_leaf=200)` on
   `sign(Γ)` weighted by `|Γ|`, features = rule_covariates ONLY.
3. **M7 evaluation is a separate mandatory skill** (Analyst + Critic stages):
   cross-fitted DR policy value, treat-all/treat-none baselines, cluster-
   bootstrap gain CI, **subgroup value parity** (fairness of the rule — the
   education-specific contribution surface).
4. **Variants over new prompts**: PF/Analyst/Writer get `_causal_itr.yaml`
   variants adapted from the causal_soo variants (refine mode, forbidden
   patterns, ITR output schema, rule-card + value-table + parity paper
   sections). Writer ITR conventions live in the VARIANT (output contracts
   never in cap-droppable skills — §3.3.8 rule); no separate writing skill.
5. **Skills**: 3 new (`causal-itr-policy-learning` M6, `causal-itr-policy-
   value-evaluation` M7, `critic-checklist-causal-itr`), 9 existing causal
   skills extended to `causal_itr`. Registry: 57 skills. Caps: causal_itr
   mirrors causal_soo with methodology 12→14.
6. **Deferred (stretch, recorded not implemented)**: 2-stage dynamic regimes
   over wave-sequenced treatments; fairness-CONSTRAINED rule learning (parity
   is reported, not yet enforced in the objective).

## R2 synthetic gate (standing discipline)

`scripts/itr_synthetic_gate.py` mirrors the M6/M7 recipes on two HSLS-scaled
DGPs. **Result at authoring: PASSED** — heterogeneous DGP (oracle rule
"treat iff SES<0"): 99.97% oracle agreement, regret 1.7e-5; null DGP: gain
−0.0007 (no false rule). Gate thresholds: agreement ≥ 0.80, regret ≤ 0.02,
|null gain| ≤ 0.02. Wired into pytest (`TestSyntheticGate`).

## R3 (live) plan

Fixture `runs/fixtures/spec_x1mtheff_itr.json` (same treatment/outcome as the
SOO smoketest for comparability; rule_covariates X1SES/X1TXMTSCOR/X1SEX),
config `runs/configs/smoketest_itr.yaml` (all-deepseek). The single live run
also exercises Arc D's design memo + Critic design gate (per the roadmap, the
D exit criterion may piggyback). Bounded at ≤ 3 attempts; amendments beyond
that go to a follow-up phase per the established single-issue loop.
