---
name: inner-cv-tuning-discipline
layer: methodology
description: Hyperparameter tuning uses 5-fold inner CV on the training set only; group-aware folds when cluster IDs are available.
trigger_keywords:
  - tuning
  - hyperparameter
  - cv
  - cross-validation
  - gridsearch
  - groupkfold
applicable_task_types: []
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

# Inner-CV Tuning Discipline

Hyperparameters are tuned via **5-fold inner cross-validation on the
training set only**. The test set is never touched during tuning — it
exists solely for final evaluation reporting.

When cluster IDs (e.g., `train_school_ids.csv`) are available, the inner
CV must be group-aware to prevent within-cluster leakage during tuning.

## Group-aware inner CV

```python
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.model_selection import GridSearchCV
import pandas as pd

train_school_ids = pd.read_csv("train_school_ids.csv")["pseudo_school_id"].values

if is_classification:
    inner_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    cv_splits = list(inner_cv.split(train_X, train_y, train_school_ids))
else:
    inner_cv = GroupKFold(n_splits=5)
    cv_splits = list(inner_cv.split(train_X, train_y, train_school_ids))

grid_search = GridSearchCV(
    estimator,
    param_grid,
    cv=cv_splits,
    scoring=scoring,
)
grid_search.fit(train_X, train_y)
```

If `train_school_ids.csv` does not exist, fall back to plain `cv=5`. The
Analyst should log a warning when this fallback is taken so reviewers
know clustering was not respected during tuning.

## Hard rules

1. **Never use the test set during tuning**, including for early-stopping
   validation. Use a portion of the training set (e.g.,
   `validation_fraction=0.1` inside MLP's built-in early stopping) when
   needed.
2. **Random seed `random_state=42`** for every stochastic operation in
   tuning (CV shuffle, classifier initialization, GridSearchCV's internal
   randomization where applicable).
3. **Refit on the full training set** with the best hyperparameters before
   computing test-set metrics.
4. **No tuning grids on the test partition.** The Critic will flag any
   evidence of test-set use during tuning as a critical issue.

## Verification rules (Critic)

The Critic verifies via `results.json` metadata or warnings that:

1. Tuning used training data only. If `results.warnings` or
   `results.errors` mention test-set leakage during tuning → critical.
2. Tuning information is present for the tunable models (RF, XGBoost,
   ElasticNet, MLP). If absent → major (tuning should be documented in
   results metadata).
3. `random_state=42` was used (minor if missing).

## Source provenance

Canonical source: `agent_prompts/analyst.yaml` §"Hyperparameter Tuning"
(complete code path with group-aware CV and the test-set-leakage
prohibition).

Merged content from:
- `data_registry/task_templates/prediction.yaml` substeps `mt_02`
  through `mt_06` and check `check_mt_train_only` (workflow + verification
  language)
- `data_registry/evaluation_rubrics/methodological_checklist.yaml` item
  `an_02` (Critic verification)
