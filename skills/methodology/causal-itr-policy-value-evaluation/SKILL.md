---
name: causal-itr-policy-value-evaluation
layer: methodology
description: M7 — cross-fitted policy-value estimation with cluster-bootstrap CIs, treat-all/treat-none/best-constant baselines, subgroup value parity, and no-benefit honesty rules.
trigger_keywords:
  - policy
  - value
  - evaluation
  - baseline
  - parity
  - gain
applicable_task_types:
  - causal_itr
applicable_datasets: []
applicable_stages:
  - Analyst
  - Critic
priority: 1
references_skills:
  - subgroup-fairness-analysis
  - bootstrap-confidence-intervals
resources: []
version: "1.1"
rule_severity: mandatory
---

# Causal ITR Policy-Value Evaluation (M7)

Estimate the value of the learned rule and — the actual headline — its
GAIN over the best constant policy, honestly.

## Value estimator (cross-fitted, DR)

For any candidate policy π, using the out-of-fold DR pseudo-outcomes
from M6:

```
V̂(π)   = mean( π(x_i) · Γ_i ) + mean( μ̂₀(x_i) + (1−T_i)(Y_i − μ̂₀(x_i))/(1−ê(x_i)) )
       ≡ mean over i of the DR score of the outcome under π
V̂_all  = V̂(π ≡ 1);  V̂_none = V̂(π ≡ 0)
gain    = V̂(π̂) − max(V̂_all, V̂_none)
```

The rule π̂ evaluated on fold k must have been LEARNED on the other
folds (cross-fitted policy evaluation) — in-sample value is the ITR
analogue of evaluating on the training set and is forbidden.

## Deterministic implementation (MANDATORY - R3-followup)

Call `analysis_helpers.itr_crossfit_policy_value(df, T, Y,
adjustment_cols, rule_covariate_cols, groups=school_ids)` - it
computes value, baselines, gain, and the cluster-bootstrap CI on the
SAME gain statistic (F-R3-M7-CI-INCONSISTENT was a live point estimate
outside its own CI from a hand-rolled bootstrap).

## Uncertainty

Cluster bootstrap (school as the resampling unit, per D1), 1000
iterations, on the `gain` statistic → `gain_ci_lower/upper`.
`random_state=42`.

## Honesty rules (mandatory)

1. If the gain CI covers zero, the HEADLINE is "no detectable
   targeting benefit over the best constant policy" — the rule card is
   still reported, framed as exploratory.
2. Never report V̂(π̂) alone: always alongside `value_treat_all` and
   `value_treat_none`. A "high" policy value that equals treat-all's
   value is not a targeting finding.
3. "Regret vs oracle" language is BANNED on real data (the oracle is
   unknown); regret is a synthetic-gate concept only.

## Subgroup value parity (fairness of the rule)

For each protected attribute in `research_spec.subgroup_analyses`
(via the D1 label snapshot): report per-level `policy_value` and `n`,
plus the share of each level the rule would treat. A rule whose value
gain accrues to one group while another is systematically untreated
must be flagged in `warnings` and discussed by the Writer. See
`subgroup-fairness-analysis` for the label-preservation mechanics.

## Output schema

`estimates.M7` per the causal_itr Analyst prompt: `policy_value`,
`value_treat_all`, `value_treat_none`, `value_gain_vs_best_constant`,
`gain_ci_lower/upper`, `se_method: cluster_bootstrap`, `n_folds`,
`subgroup_value_parity`.

## Failures prevented

In-sample value inflation; headline-hides-null; missing baselines;
rules that look valuable only because treat-all is valuable;
inequitable rules shipped without a parity table.

## Source provenance

Authored in V3.1 Arc R (R1) per the causal-ITR scope note (internal).
