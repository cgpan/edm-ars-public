---
name: causal-itr-policy-learning
layer: methodology
description: M6 — doubly-robust policy learning; DR pseudo-outcomes + shallow sklearn policy tree restricted to rule_covariates; no econml dependency.
trigger_keywords:
  - policy
  - rule
  - targeting
  - regime
  - itr
  - treatment-rule
applicable_task_types:
  - causal_itr
applicable_datasets: []
applicable_stages:
  - Analyst
priority: 1
references_skills:
  - causal-itr-policy-value-evaluation
  - causal-forest-cate
  - causal-positivity-diagnostics
  - causal-data-engineer-contract
resources: []
version: "1.1"
rule_severity: mandatory
---

# Causal ITR Policy Learning (M6)

Learn an interpretable treatment rule π(x) ∈ {0, 1} that maximizes the
doubly-robust estimate of the population outcome, conditioning ONLY on
`research_spec.rule_covariates`.

## The recipe (sklearn only — econml is NOT installed in the sandbox)

1. **Cross-fitted nuisance models** (K = 5, `GroupKFold` on the
   school cluster IDs per D1; `random_state=42` everywhere):
   - Outcome models `μ̂₁(x)`, `μ̂₀(x)`: `GradientBoostingClassifier`
     (binary Y, use `predict_proba`) or `GradientBoostingRegressor`
     (continuous Y), fit separately on treated / control rows of the
     TRAINING folds, predicted on the held-out fold.
   - Propensity `ê(x)`: `LogisticRegression(max_iter=1000)` on the full
     adjustment set; clip to `[0.02, 0.98]`.
2. **DR pseudo-outcome (per unit, out-of-fold):**

   ```
   Γ_i = μ̂₁(x_i) − μ̂₀(x_i)
         + T_i · (Y_i − μ̂₁(x_i)) / ê(x_i)
         − (1 − T_i) · (Y_i − μ̂₀(x_i)) / (1 − ê(x_i))
   ```

3. **Policy tree via weighted classification**: fit
   `DecisionTreeClassifier(max_depth=2, min_samples_leaf=200,
   random_state=42)` on features = rule_covariates ONLY, labels =
   `(Γ_i > 0)`, `sample_weight = |Γ_i|`. This is the standard
   weighted-classification reduction of policy learning.
4. **Rule extraction**: walk the fitted tree and emit
   `policy_rule_text` as a human-readable conjunction per leaf (e.g.
   "treat if X1SES < -0.12 and X1TXMTSCOR < 51.3"). Depth > 3 is a
   contract violation — interpretability is the point.
5. Report `n_treated_by_rule` / `share_treated_by_rule` on the analytic
   sample. A rule that treats < 2% or > 98% of students is effectively
   constant — flag in warnings and let M7's best-constant comparison
   carry the story.

## Deterministic implementation (MANDATORY - R3-followup)

Do NOT re-implement the recipe: call the shipped helpers
(`import analysis_helpers`):

```python
gamma = analysis_helpers.itr_dr_pseudo_outcomes(df, T, Y, adjustment_cols, groups=school_ids)
tree, rule_text, share = analysis_helpers.itr_learn_policy_tree(df, gamma, rule_covariate_cols)
```

`itr_learn_policy_tree` raises ValueError on a degenerate rule (share
outside [0.02, 0.98]) - catch it and report "no meaningful targeting
rule" honestly (F-R3-M6-SCALE-DEGENERATE-RULE was a live
treat-everyone rule with thresholds outside the covariate ranges from
a hand-rolled re-implementation).

## Hard constraints

- Rule features = `rule_covariates` (resolved via D1's
  `resolve_encoded_columns`) and NOTHING else. Using the full
  adjustment set in the tree is a violation: rule covariates are the
  variables a school could actually observe and act on at decision
  time.
- The tree is fit on out-of-fold Γ̂ — never refit nuisances in-sample.
- Positivity first: apply G3's trimming/overlap handling BEFORE
  computing Γ (clipped ê is a floor, not a substitute for the G3
  diagnostics).

## Output schema

`estimates.M6` per the causal_itr Analyst prompt: `policy_rule_text`,
`tree_depth`, `rule_covariates_used`, `n_treated_by_rule`,
`share_treated_by_rule`, `notes`.

## Failures prevented

Rule-covariate leakage (conditioning on unactionable covariates);
in-sample policy evaluation; black-box uninterpretable regimes;
degenerate near-constant rules reported as "targeting".

## Source provenance

Authored in V3.1 Arc R (R1) per the causal-ITR scope note (internal). The
weighted-classification reduction follows the standard
policy-learning literature (offline evaluation via DR scoring); the
sklearn-only constraint comes from `requirements-sandbox.txt`.
