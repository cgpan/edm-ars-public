# V2.0 Phase 2b — Extraction Plan

CSV counts verified against `audit/skill_candidates.csv`:
- `layer == "task-type"`: **8** ✓
- `layer == "dataset"`: **7** ✓

Plan below is committed before any SKILL.md extraction so the count
discrepancies (D6 + D9 expansions) are visible up front.

## Dataset skills (7 from CSV → 7 files; one bundles a YAML resource)

| skill_id | source_file | source_section |
|---|---|---|
| `hsls09-variable-registry` | `data_registry/datasets/hsls09_public.yaml` | `variables` (L20-L1700) including `protected_attribute` flags |
| `hsls09-csv-format-quirks` | `agent_prompts/data_engineer.yaml` | "HSLS:09 CSV Format Warning" (L304-L321) |
| `hsls09-missing-codes` | `data_registry/datasets/hsls09_public.yaml` + `agent_prompts/data_engineer.yaml` | `missingness.sentinel_codes` (L1739-L1762) + DataEngineer step 3 |
| `hsls09-temporal-ordering` | `data_registry/datasets/hsls09_public.yaml` + `agent_prompts/problem_formulator.yaml` | `waves` + `temporal_order` (L7-L40) + PF Validation Rule 1 |
| `hsls09-school-cluster-reconstruction` (→ renamed to **`hsls09-school-fingerprints`** per D6) | `agent_prompts/data_engineer.yaml` | "School Cluster Reconstruction step 6" (L156-L192) |
| `hsls09-tier3-exclusions` | `data_registry/datasets/hsls09_public.yaml` | `tier3_exclusion_rules` (L42-L75) |
| `hsls09-structural-mnar-outcomes` | `agent_prompts/data_engineer.yaml` | "Structural vs random outcome missingness" (L80-L93) |

## Task-type skills (8 from CSV → 14 files after D9)

| skill_id | source_file | source_section |
|---|---|---|
| `prediction-workflow-overview` | `data_registry/task_templates/prediction.yaml` | `standard_workflow` (L105-L635) |
| `prediction-model-battery` (**meta-skill, D9**) | `data_registry/task_templates/prediction.yaml` | `model_training.model_battery` (L269-L361) — meta only; per-family content moves to the six new skills below |
| `prediction-evaluation-classification` | `data_registry/task_templates/prediction.yaml` + `agent_prompts/analyst.yaml` | `evaluation.metrics.classification` (L441-L468) + analyst Evaluation Protocol |
| `prediction-evaluation-regression` | `data_registry/task_templates/prediction.yaml` + `agent_prompts/analyst.yaml` | `evaluation.metrics.regression` (L469-L484) + analyst Evaluation Protocol |
| `prediction-quality-gate` | `agent_prompts/analyst.yaml` | "Model Quality Gate" (L380-L427) |
| `prediction-research-question-design` | `agent_prompts/problem_formulator.yaml` | "Validation Rules 8-13 + Canonical Questions" (L108-L163) |
| `prediction-critic-checklist` | `data_registry/evaluation_rubrics/methodological_checklist.yaml` | full file (L24-L419) — ~180-line skill body, kept as one per Decision 2 |
| `smote-imbalance-handling` | `agent_prompts/analyst.yaml` | "Class Imbalance Handling" (L194-L261) |

## Decision 9 — per-family model battery expansion

`prediction-model-battery` becomes 1 meta-skill (same name, kept) + 6 per-family skills (new). Net addition = +6 task-type files.

| New skill | Source content | SHAP explainer |
|---|---|---|
| `model-logistic-regression` | `agent_prompts/analyst.yaml` Pilot Model Battery row 1 + `prediction.yaml` `model_lr` | `LinearExplainer` |
| `model-random-forest` | analyst row 2 + `prediction.yaml` `model_rf` | `TreeExplainer` |
| `model-xgboost` | analyst row 3 + `prediction.yaml` `model_xgb` | `TreeExplainer` |
| `model-elasticnet` | analyst row 4 + `prediction.yaml` `model_enet` | `LinearExplainer` |
| `model-mlp` | analyst row 5 + `prediction.yaml` `model_mlp` (incl. KernelExplainer constraints + early-stopping) | `KernelExplainer` (sample_cap=1000) |
| `model-stacking-ensemble` | analyst row 6 + `prediction.yaml` `model_stacking` | **SKIP** (no SHAP for stacking) |

The meta-skill `prediction-model-battery` declares
`references_skills: [model-logistic-regression, model-random-forest, model-xgboost, model-elasticnet, model-mlp, model-stacking-ensemble]`
so a single match pulls the whole battery.

## Decision 6 — school-cluster reconstruction split

The CSV row `hsls09-school-cluster-reconstruction` becomes two files:

| New skill | Layer | Notes |
|---|---|---|
| `cluster-id-reconstruction-from-fingerprints` | **methodology** | Generic recipe: variance check, multi-key groupby, quality diagnostics. No HSLS variable names in the body. |
| `hsls09-school-fingerprints` | **dataset** | Lists the HSLS-specific fingerprint vars (`X1SCHOOLCLI`, `X1COUPERTEA`, `X1COUPERCOU`, `X1COUPERPRI`, `X1CONTROL`, `X1LOCALE`, `X1REGION`); explains SCH_ID suppression; `references_skills: [cluster-id-reconstruction-from-fingerprints]`. |

Net change: dataset count stays at 7 (one renamed); methodology gains +1 (now 11 total).

## Resource bundling plan

| Skill | Bundled file | Source |
|---|---|---|
| `dataset/hsls09-variable-registry` | `variable_registry.yaml` (~1,834 lines) | byte-identical copy of `data_registry/datasets/hsls09_public.yaml` |

(Phase 2a already bundled `paper_template_v2.tex` under `acm-acmart-sigconf-template`.)

## Provisional total file count

- Dataset SKILL.md: 7
- Dataset bundled YAML: 1
- Task-type SKILL.md: 8 base + 6 D9 family additions = 14 *(spec text said "13 (8+5)" — that math doesn't reconcile; 8 base + 6 net-new families = 14, confirmed against the per-skill list above)*
- Methodology SKILL.md: 1 new (D6)
- **Phase 2b SKILL.md total: 22 + 1 YAML resource**

Combined registry after Phase 2b:
- Phase 2a: 19 (10 methodology + 9 writing)
- Phase 2b: +22 (7 dataset + 14 task-type + 1 methodology)
- **Expected `registry.count()` = 41**

By layer after Phase 2b:
- task-type: 14
- dataset: 7
- methodology: 11 (10 from 2a + 1 from D6)
- writing: 9 (unchanged from 2a)

## Naming notes

- The CSV row `hsls09-school-cluster-reconstruction` is renamed to `hsls09-school-fingerprints` per the D6 spec (the name reflects what stays in the dataset skill — the variable list — rather than the technique).
- All trigger keywords include plural forms (e.g. `[variable, variables]`) until Phase 2c adds matcher stemming.
