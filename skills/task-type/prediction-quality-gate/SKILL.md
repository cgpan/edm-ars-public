---
name: prediction-quality-gate
layer: task-type
description: Performance floor (AUC ≥ 0.60 / R² ≥ 0.05) before SHAP; if no model passes, skip SHAP entirely.
trigger_keywords:
  - quality
  - gate
  - floor
  - threshold
  - thresholds
  - shap-eligible
applicable_task_types:
  - prediction
applicable_datasets: []
applicable_stages:
  - Analyst
  - Critic
priority: 1
references_skills: []
resources: []
version: "1.0"
rule_severity: mandatory
---

# Prediction Quality Gate

Apply this gate **after** evaluating all models on the test set and
**before** running SHAP. It prevents the system from interpreting noise
as signal — feature attributions for a non-discriminative model are
worse than no attributions at all because they look authoritative.

## Floors

| Task type | Metric | Floor |
|---|---|---|
| Classification | AUC | `0.60` |
| Regression | R² | `0.05` |

A model passes the gate iff its primary-metric value on the test set
is **≥** the floor.

## Operational sequence

```python
import analysis_helpers

gate_results = analysis_helpers.model_quality_gate(
    all_models=results["all_models"],
    is_classification=is_classification,
    auc_floor=0.60,
    r2_floor=0.05,
)

# Find the best SHAP-eligible model.
shap_eligible = [
    (name, gate_results[name]["metric_value"])
    for name, g in gate_results.items()
    if g["shap_eligible"]
]

if shap_eligible:
    shap_model_name = max(shap_eligible, key=lambda x: x[1])[0]
    # ... run SHAP on shap_model_name ...
else:
    # NO model passed the gate — skip ALL SHAP analysis.
    warnings_list.append(
        "MODEL QUALITY WARNING: No model achieved the minimum performance "
        f"threshold ({'AUC >= 0.60' if is_classification else 'R² >= 0.05'}). "
        "SHAP feature importance analysis was skipped because interpreting "
        "feature contributions from a non-discriminative model would produce "
        "misleading results."
    )
    results["shap_skipped"] = True
    results["shap_skip_reason"] = "no_model_passed_quality_gate"
```

## Per-model warnings

For each model that FAILS the gate, append a warning:

```python
for name, g in gate_results.items():
    if not g["passed"]:
        warnings_list.append(
            f"Model {name} failed quality gate: {g['metric_name']} = "
            f"{g['metric_value']:.4f} (floor = {g['floor']}). This model "
            "is reported in the comparison table but excluded from SHAP "
            "interpretability analysis."
        )
```

## Why these floors

- **AUC = 0.60** is roughly the threshold below which test-set
  predictions are statistically distinguishable from chance only at
  large sample sizes. Below 0.60, SHAP values represent the model's
  best guess about features that don't actually predict the outcome
  — meaningless attribution.
- **R² = 0.05** is the floor below which the model explains less than
  5% of outcome variance. PDP and SHAP for such a model show structure
  in noise.

## What goes in `results.json`

```json
{
  "model_quality_gate": {
    "LogisticRegression": {
      "passed": true, "metric_name": "auc",
      "metric_value": 0.78, "floor": 0.60, "shap_eligible": true
    },
    "MLP": {
      "passed": false, "metric_name": "auc",
      "metric_value": 0.55, "floor": 0.60, "shap_eligible": false
    }
  },
  "shap_model": "XGBoost",
  "shap_skipped": false,
  "shap_skip_reason": null
}
```

When SHAP is skipped:

```json
{
  "shap_model": null,
  "shap_skipped": true,
  "shap_skip_reason": "no_model_passed_quality_gate"
}
```

## Interaction with the MLP timeout fallback

Two SHAP-source decisions can change the model used for SHAP:

1. **Quality gate** — best individual model fails the floor → fall
   back to next-best gate-passing individual model.
2. **MLP KernelExplainer timeout** — MLP is gate-eligible best but
   times out → fall back to next-best non-MLP individual model
   (per `shap-explainer-selection`).

When both apply, quality gate fires first; the MLP-timeout fallback
only runs if MLP passed the gate AND was selected for SHAP.

## Writer's responsibility

When `shap_skipped: true`, the Writer must state in §Limitations
that SHAP was not computed because no model met the performance
threshold, and that reported model comparison numbers should be
interpreted with caution. The `paper-writing-style-rules` and
`hsls09-multilevel-limitations-paragraph` skills describe how this
fits into the limitations narrative.

## Source provenance

Canonical source: `agent_prompts/analyst.yaml` §"Model Quality Gate"
(L380-L427).
