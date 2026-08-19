---
name: shap-explainer-selection
layer: methodology
description: Pick the right SHAP explainer per model family; constrain KernelExplainer; never SHAP a stacking ensemble.
trigger_keywords:
  - shap
  - explainer
  - importance
  - interpretability
  - kernelexplainer
  - treeexplainer
applicable_task_types: []
applicable_datasets: []
applicable_stages:
  - Analyst
  - Critic
priority: 1
references_skills: []
resources: []
version: "1.0"
---

# SHAP Explainer Selection

Compute SHAP values for the **best individual model only**.
StackingEnsemble is excluded from SHAP regardless of performance.

## Explainer mapping (complete)

| Model family | SHAP explainer |
|---|---|
| `LogisticRegression`, `LinearRegression` | `shap.LinearExplainer` |
| `ElasticNet`, `SGDClassifier` (elasticnet) | `shap.LinearExplainer` |
| `RandomForestClassifier`, `RandomForestRegressor` | `shap.TreeExplainer` |
| `XGBClassifier`, `XGBRegressor`, `LGBMClassifier`, `LGBMRegressor` | `shap.TreeExplainer` |
| `MLPClassifier`, `MLPRegressor` | `shap.KernelExplainer` (with the constraints below) |
| `StackingClassifier`, `StackingRegressor` | **SKIP** — never compute SHAP for stacking |

## KernelExplainer constraints (MLP only)

`KernelExplainer` is acceptable **only** for MLP models, and only with all
of the following constraints. Without these, the call will silently grow
into a multi-hour computation.

- **Sample cap**: 1,000 rows from the test set (`random_state=42` if
  `n_test > 1000`).
- **Background**: `shap.kmeans(train_X, 100)`.
- **`nsamples`**: 500.
- **Hard timeout**: 600 seconds. Wrap the call in a subprocess or
  signal-based timeout.
- **Fallback on timeout**: skip SHAP for MLP and fall back to the
  next-best non-MLP individual model for ALL interpretability outputs
  (beeswarm, bar plot, PDPs, `feature_importance.csv`). Document the
  fallback in `results.warnings` with the name of the model used.

## Use the project's helper, not raw `shap` calls

Always go through `analysis_helpers` so list-vs-array edge cases and plot
formatting are handled consistently:

```python
import analysis_helpers
import numpy as np

# 1. Compute SHAP values — handles list-vs-array, Explanation objects, etc.
shap_vals = analysis_helpers.safe_shap_values(explainer, X_shap)
# shap_vals is always a clean 2D numpy array (n_samples, n_features)

# 2. Derive feature importance.
mean_abs_shap = np.abs(shap_vals).mean(axis=0)
top_feat_names = list(X_shap.columns[np.argsort(mean_abs_shap)[::-1]])

# 3. Save SHAP plots — returns ["shap_summary.png", "shap_importance.png"].
shap_figs = analysis_helpers.save_shap_plots(shap_vals, X_shap, output_dir)

# 4. Save partial dependence plots — returns list of filenames.
pdp_figs = analysis_helpers.save_pdp_plots(
    best_model, train_X, top_feat_names, output_dir
)
```

Do **not** call `explainer.shap_values()` directly and do **not** call
`shap.summary_plot()` directly — both have list-vs-array gotchas the
helpers already smooth over.

## Interpretability output rule

All interpretability outputs come from the best individual model only.
StackingEnsemble is reported in `model_comparison.csv` but never used as
the SHAP source — even if it is the overall best model.

If the best individual model is MLP and KernelExplainer times out, the
fallback non-MLP individual model produces every interpretability output;
document the fallback in `results.warnings`.

## Verification rules (Critic)

The Critic must verify:

1. The model that produced the SHAP outputs has the correct explainer per
   the table above.
2. StackingEnsemble does NOT appear as the SHAP source.
3. If KernelExplainer was used, the model is MLP and the constraints
   (sample cap ≤ 1,000, `nsamples` ≤ 500) are satisfied.
4. If MLP timed out, the fallback model is documented in
   `results.warnings`.
5. If SHAP is absent because every model failed the quality gate, the
   warning is present and `shap_skipped: true` is set.

Each violation is a major issue.

## Source provenance

Canonical source: `agent_prompts/analyst.yaml` §"SHAP Interpretability
Protocol" (most complete prose with explainer table, KernelExplainer
constraints, helper-function guidance).

Merged content from:
- `data_registry/task_templates/prediction.yaml` §`shap_protocol`
  (machine-readable mapping including `XGBClassifier` vs `XGBRegressor`
  and `MLPClassifier` vs `MLPRegressor` variants)
- `data_registry/evaluation_rubrics/methodological_checklist.yaml` items
  `an_05`, `an_08`, `an_09`, `an_10` (Critic verification language and
  the StackingEnsemble exclusion + MLP fallback documentation rules)
