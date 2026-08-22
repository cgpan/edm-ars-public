---
name: critic-checklist-causal-itr
layer: task-type
description: Per-section Critic review checklist for ITR/policy-learning studies — rule actionability, cross-fitted value, baselines, gain CI, parity, no-benefit honesty.
trigger_keywords:
  - critic
  - checklist
  - review
  - policy
  - rule
  - verdict
applicable_task_types:
  - causal_itr
applicable_datasets: []
applicable_stages:
  - Critic
priority: 1
references_skills:
  - causal-itr-policy-learning
  - causal-itr-policy-value-evaluation
  - causal-positivity-diagnostics
  - causal-sensitivity-unmeasured-confounding
resources: []
version: "1.0"
rule_severity: mandatory
---

# Causal ITR Critic Checklist

Walk every row; each failure becomes an issue in `review_report.json`
with the stated severity. The generic causal rows (identification,
positivity, balance, sensitivity/refuters) are carried by the injected
G-family skills — this checklist adds the ITR-specific surface.

| ID | Item | Severity | Check |
|---|---|---|---|
| `itr_01` | Rule conditions only on `rule_covariates` | critical | `estimates.M6.rule_covariates_used` ⊆ `research_spec.rule_covariates`. Any other feature in the rule = actionability violation. |
| `itr_02` | Policy value is cross-fitted | critical | `estimates.M7.n_folds ≥ 2` and the Analyst's code learned the rule out-of-fold. In-sample value = leakage-equivalent. |
| `itr_03` | Baselines reported | major | `value_treat_all` AND `value_treat_none` present; gain computed vs the BEST constant. |
| `itr_04` | Gain CI present via cluster bootstrap | major | `gain_ci_lower/upper` non-null; `se_method == "cluster_bootstrap"`. |
| `itr_05` | No-benefit honesty | critical | If the gain CI covers 0, the stated headline (and later the paper's abstract) must lead with no-detectable-benefit. A buried null = critical. |
| `itr_06` | Rule interpretability | major | `tree_depth ≤ 3` and `policy_rule_text` is a readable conjunction. |
| `itr_07` | Degenerate rule flagged | major | `share_treated_by_rule` in [0.02, 0.98]; outside → must be flagged as effectively-constant in warnings. |
| `itr_08` | Subgroup value parity present | major | `estimates.M7.subgroup_value_parity` has every attribute from `research_spec.subgroup_analyses`, each with per-level value + n. |
| `itr_09` | Parity discussed, not just tabulated | minor | Warnings or notes engage with who the rule treats across protected groups. |
| `itr_10` | Individual-guarantee language absent | major | No "student i will benefit" claims anywhere in notes/summary — population-level targeting language only. |

## Verdict interaction

These rows feed the standard severity counts consumed by the
deterministic verdict evaluator; `itr_01`/`itr_02`/`itr_05` are the
ABORT-eligible class when unfixable.

## Source provenance

Authored in V3.1 Arc R (R1) per the causal-ITR scope note (internal).
