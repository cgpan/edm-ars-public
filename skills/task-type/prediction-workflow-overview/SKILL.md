---
name: prediction-workflow-overview
layer: task-type
description: Four-stage end-to-end supervised prediction pipeline orchestrated across DataEngineer → Analyst.
trigger_keywords:
  - prediction
  - workflow
  - pipeline
  - stages
  - stage
applicable_task_types:
  - prediction
applicable_datasets: []
applicable_stages:
  - DataEngineer
  - Analyst
priority: 1
references_skills: []
resources: []
version: "1.0"
---

# Prediction Workflow Overview

End-to-end supervised prediction has four stages, executed in order
by the orchestrator. Each stage corresponds to a unit of agent work
and emits artifacts the next stage consumes.

## Stage map

| # | Stage | Agent | Inputs | Primary outputs |
|---|---|---|---|---|
| 1 | Data preparation | DataEngineer | `research_spec.json`, dataset registry, raw CSV path | `train_X.csv`, `train_y.csv`, `test_X.csv`, `test_y.csv`, `test_protected.csv`, `train_school_ids.csv`, `test_school_ids.csv`, `data_report.json` |
| 2 | Model training | Analyst | training partition + `data_report.json` + `research_spec.json` | tuned base models + StackingEnsemble (in-memory) |
| 3 | Evaluation | Analyst (continues) | trained models + held-out test set | `results.all_models` metrics + bootstrap CIs + `model_comparison.csv` + ROC/calibration/confusion or residual plot |
| 4 | Interpretation | Analyst (continues) | best individual model + test set + `test_protected.csv` | SHAP figures + `feature_importance.csv` + PDPs + `subgroup_performance.csv` + ICC + clustered CIs + sensitivity analysis |

The Critic reviews stage 1, 2, 3, 4 outputs together; the Writer
consumes everything plus the OutlineAgent's outline.

## Stage 1 — Data preparation (substep IDs)

Substeps from `data_registry/task_templates/prediction.yaml`:

| ID | Step | Notes |
|---|---|---|
| `dp_01` | `load_raw_data` | `pd.read_csv(raw_data_path)`. Path is provided in user message — see `core-data-engineer-role` for the exact convention. |
| `dp_02` | `select_variables` | Keep predictor cols + outcome + subgroup cols + dataset-specific extras (e.g. school fingerprint vars for HSLS). |
| `dp_03` | `recode_nces_missing` | Replace numeric sentinels and text labels with `NaN`. See dataset skills `hsls09-missing-codes`, `hsls09-csv-format-quirks`. |
| `dp_04` | `drop_missing_outcome` | Always drop missing outcomes; never impute the outcome. |
| `dp_05` | `assess_missingness` | Per-column `pct_missing` on the analytic sample (post outcome-drop). |
| `dp_06` | `apply_missing_data_protocol` | See `missingness-tiered-protocol`. Snapshot subgroup labels BEFORE imputation. |
| `dp_07` | `school_cluster_reconstruction` | When the dataset suppresses cluster IDs (HSLS:09), recover pseudo-IDs via `cluster-id-reconstruction-from-fingerprints` + `hsls09-school-fingerprints`. |
| `dp_08` | `grouped_train_test_split` | See `school-aware-train-test-split`. Run AFTER reconstruction, BEFORE encoding. |
| `dp_09` | `save_test_protected` | Save pre-encoding subgroup labels for the Analyst. |
| `dp_10` | `encode_categoricals` | `pd.get_dummies(drop_first=True)` on training set; `reindex(columns=...)` on test. |
| `dp_11` | `drop_constant_columns` | Drop zero-variance columns after encoding. |
| `dp_12` | `report_class_balance` | Binary outcomes: `is_imbalanced: true` if minority < 20%. |
| `dp_13` | `save_outputs` + `add_multilevel_warning` | Save train/test CSVs; append the multilevel limitation warning to `data_report.warnings`. |

## Stage 2 — Model training

See `prediction-model-battery` (which composes the six per-family
skills). Order: baseline → RF → XGBoost → ElasticNet → MLP →
StackingEnsemble.

## Stage 3 — Evaluation

See `prediction-evaluation-classification` /
`prediction-evaluation-regression`. All metrics on the held-out test
set; bootstrap CIs for the primary metric; figures saved at 150+ dpi.

## Stage 4 — Interpretation

- Quality gate (`prediction-quality-gate`) BEFORE SHAP.
- SHAP for the best individual model (`shap-explainer-selection`).
- PDPs for top 3 features.
- Subgroup analysis (`subgroup-fairness-analysis`).
- ICC + clustered CIs (`clustered-bootstrap-ci-and-icc`).
- Sensitivity analysis if any predictor has > 20% missingness
  (`sensitivity-analysis-high-missingness`).

## ABORT and REVISE rules

ABORT (orchestrator-side, before invoking Critic):
- `data_report.validation_passed == false`
- `analytic_n < 1000`
- `JSONDecodeError` from `parse_llm_json()` on agent output

ABORT (Critic-issued):
- Confirmed temporal or target leakage
- Unanswerable research question

REVISE (Critic-issued):
- Any critical issue
- More than 2 major issues
- `overall_quality_score < 7`
- Up to `max_revision_cycles` (config); when exhausted, proceed to
  WRITING with the UNVERIFIED flag set (see writing skill
  `unverified-flag-and-appendix`).

## Source provenance

Canonical source: `data_registry/task_templates/prediction.yaml`
§`standard_workflow` (L105-L635) + §`quality_gates`. Per-step details
are in the linked methodology, dataset, and per-family skills.
