# EDM-ARS: Educational Data Mining Automated Research System
## Definitive Implementation Specification v1.0

> **Status**: Authoritative. This document supersedes `edm_auto_research_architecture.md` wherever
> the two conflict. All implementation decisions made during the design interview are recorded here.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Configuration](#2-configuration)
3. [Data Registry Schema](#3-data-registry-schema)
4. [Agent Specifications](#4-agent-specifications)
   - 4.1 ProblemFormulator
   - 4.2 DataEngineer
   - 4.3 Analyst
   - 4.4 Critic
   - 4.5 Writer
5. [Orchestrator Design](#5-orchestrator-design)
6. [Inter-Agent Message Formats](#6-inter-agent-message-formats)
7. [Output Directory Structure](#7-output-directory-structure)
8. [Error Handling](#8-error-handling)
9. [Dependencies](#9-dependencies)
10. [Pilot Scope & Roadmap](#10-pilot-scope--roadmap)

---

## 1. System Overview

EDM-ARS is a domain-specific multi-agent system that automates the end-to-end workflow of
prediction-focused educational data mining research. Given a dataset and an optional user prompt,
it produces a complete, reviewer-ready LaTeX research paper with real citations, validated
methodology, and interpretable results.

```
┌──────────────────────────────────────────────────────────────┐
│                      ORCHESTRATOR                            │
│            (State Machine + Message Router)                  │
│                                                              │
│  ┌──────────┐   ┌──────────┐   ┌─────────┐   ┌──────────┐  │
│  │  Problem  │──▶│   Data   │──▶│ Analyst │──▶│  Writer  │  │
│  │Formulator│   │ Engineer │   │         │   │          │  │
│  └──────────┘   └──────────┘   └─────────┘   └──────────┘  │
│       │              │              │              │         │
│       └──────────────┴──────────────┴──────────────┘        │
│                          ▲                                   │
│                     ┌────┴────┐                              │
│                     │ Critic  │  (reviews all agent outputs) │
│                     └─────────┘                              │
│                          ▲                                   │
│                  ┌───────┴────────┐                          │
│                  │  Data Registry │                          │
│                  └────────────────┘                          │
└──────────────────────────────────────────────────────────────┘
```

**Pilot scope constraints:**
- Dataset: HSLS:09 public-use file only
- Task type: Prediction
- Output format: LaTeX (ACM `acmart` template, `sigconf` proceedings style)
- Code sandbox: Docker (configurable; subprocess fallback available)

---

## 2. Configuration

A central `config.yaml` at the project root drives all model selection and pipeline parameters.
Never hardcode model IDs inside agent classes.

```yaml
# config.yaml
models:
  problem_formulator: claude-sonnet-4-6
  data_engineer: claude-sonnet-4-6
  analyst: claude-sonnet-4-6
  critic: claude-opus-4-6
  writer: claude-sonnet-4-6

pipeline:
  max_revision_cycles: 2
  random_state: 42
  cost_budget_usd: 5.0     # soft limit; log warning if exceeded, do not abort

sandbox:
  enabled: true
  image: "edm-ars-sandbox:latest"
  memory_limit: "4g"
  cpu_count: 2
  network_disabled: true   # LLM-generated code must not make outbound HTTP calls
  auto_build: true         # build image automatically if not found

semantic_scholar:
  base_url: https://api.semanticscholar.org/graph/v1
  max_results: 10
  year_filter: 10          # papers published within last N years
  request_delay_s: 0.5     # delay between S2 API calls to respect rate limit

paths:
  data_registry: data_registry/
  raw_data: data/raw/
  output_base: output/
  agent_prompts: agent_prompts/
```

**Usage in agents:**
```python
import yaml

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

config = load_config()
model = config["models"]["critic"]   # "claude-opus-4-6"
```

At the start of each pipeline run, copy `config.yaml` to `{output_dir}/config_snapshot.yaml` for
reproducibility.

---

## 3. Data Registry Schema

### 3.1 Registry File Structure

```
data_registry/
├── datasets/
│   └── hsls09_public.yaml      ← pilot: public-use file only
├── task_templates/
│   └── prediction.yaml
└── evaluation_rubrics/
    ├── methodological_checklist.yaml
    └── edm_review_criteria.yaml
```

ASSISTments and PISA 2022 registry entries are deferred to Phase 2/3 (see §10).

### 3.2 HSLS:09 Public-Use Registry (`hsls09_public.yaml`)
Find it at data_registry/datasets/hsls09_public.yaml

#### Validation Rules
- All predictors must precede outcome in `temporal_order`
- Combined missingness must not reduce sample below 10,000
- Every predictor must have an educational rationale
- `novelty_score_self_assessment` must be ≥ 3 (otherwise regenerate)

---

### 4.2 DataEngineer Agent

| Property | Value |
|---|---|
| Model | `config.models.data_engineer` |
| Temperature | 0.0 |
| Input | `research_spec.json`, dataset registry, raw data path |
| Output | `train_X.csv`, `train_y.csv`, `test_X.csv`, `test_y.csv`, `data_report.json` |

#### Missing Data Protocol

| Missingness | Method |
|---|---|
| < 5% | Median (continuous) or mode (categorical) imputation |
| 5–20% | Multiple imputation (`IterativeImputer`, 5 iterations) |
| > 20% | Flag in `data_report.warnings`, impute anyway, note as limitation |
| Complete-case < 60% of original n | **ABORT** |

#### Validation Checklist (all must pass before `validation_passed: true`)
- [ ] No NaN values remain in train or test sets
- [ ] Outcome variable has expected type and range
- [ ] No constant (zero-variance) predictors remain
- [ ] Train/test split is stratified (classification outcomes)
- [ ] Class balance reported (majority/minority ratio)
- [ ] Feature count after encoding reported
- [ ] Sample sizes `n_train`, `n_test` reported
- [ ] School clustering warning added to `data_report.warnings`:
  `"Multilevel structure (students nested in schools) is not modeled. This is a limitation."`

#### Critical Rules
- **NEVER** impute the outcome variable; drop rows with missing outcomes
- **NEVER** include the outcome in the predictor matrix
- **NEVER** fit scalers on the full dataset; fit on train set, transform test set
- Test set: minimum 20% of analytic sample
- Fixed seed: `random_state = config["pipeline"]["random_state"]` (default 42)

#### `data_report.json` Schema
```json
{
  "dataset": "hsls09_public",
  "original_n": 23503,
  "analytic_n": 0,
  "n_train": 0,
  "n_test": 0,
  "outcome_variable": "string",
  "outcome_type": "binary|continuous",
  "class_balance": {"class_0": 0.0, "class_1": 0.0},
  "n_predictors_raw": 0,
  "n_predictors_encoded": 0,
  "missingness_summary": {
    "variable_name": {"pct_missing": 0.0, "imputation_method": "string"}
  },
  "variables_flagged": [],
  "validation_passed": true,
  "warnings": []
}
```

---

### 4.3 Analyst Agent

| Property | Value |
|---|---|
| Model | `config.models.analyst` |
| Temperature | 0.0 |
| Input | Processed data files, `data_report.json`, `research_spec.json` |
| Output | `results.json`, `model_comparison.csv`, `feature_importance.csv`, figures |

#### Model Battery

**Pilot model battery:**

| # | Model | Notes |
|---|---|---|
| 1 | Logistic / Linear Regression (L2) | Baseline; `C=1.0` default |
| 2 | Random Forest | Tune via 5-fold inner CV |
| 3 | XGBoost or LightGBM | Tune via 5-fold inner CV |
| 4 | ElasticNet / SGDClassifier(elasticnet) | Tune via 5-fold inner CV |
| 5 | MLP (MLPClassifier / MLPRegressor) | Tune via 5-fold inner CV; early stopping |
| 6 | StackingEnsemble | Built on 5 tuned base models; no SHAP |

**Removed from pilot:** SVM. Rationale: `KernelExplainer` timeout risk without bounded sample cap.

**Hyperparameter grids:**

*Random Forest*: `n_estimators` ∈ {100, 300, 500}, `max_depth` ∈ {5, 10, None},
`min_samples_leaf` ∈ {1, 5, 10}

*XGBoost/LightGBM*: `learning_rate` ∈ {0.01, 0.05, 0.1}, `n_estimators` ∈ {100, 300, 500},
`max_depth` ∈ {3, 5, 7}

*ElasticNet / SGDClassifier*: `alpha` ∈ {0.001, 0.01, 0.1, 1.0}, `l1_ratio` ∈ {0.1, 0.5, 0.7, 0.9}

*MLP*: `hidden_layer_sizes` ∈ {(64,), (128,), (64, 32)}, `learning_rate_init` ∈ {0.001, 0.01},
`alpha` (L2) ∈ {0.0001, 0.001}; use `max_iter=500, early_stopping=True, validation_fraction=0.1`

*StackingEnsemble*: no hyperparameter grid; meta-learner self-tunes via `RidgeCV()` (regression)
or `LogisticRegressionCV()` (classification); `cv=5, passthrough=False`

#### SHAP Protocol

**Explainer mapping (complete):**

| Model | SHAP Explainer |
|---|---|
| LogisticRegression / LinearRegression | LinearExplainer |
| ElasticNet / SGDClassifier | LinearExplainer |
| RandomForest* | TreeExplainer |
| XGB* / LightGBM* | TreeExplainer |
| MLP* | KernelExplainer (with constraints below) |
| Stacking* | **SKIP** — do not compute SHAP |

**KernelExplainer constraints (MLP only):**
- Sample cap: 1,000 rows max (sample from test set if n_test > 1,000)
- Background: `shap.kmeans(train_X, 100)`
- Hard timeout: 600s; if exceeded, skip SHAP for MLP, fall back to best non-MLP individual model for all interpretability outputs, log in `results.warnings`
- `nsamples=500`

**Interpretability output rule:** All interpretability outputs (beeswarm, bar plot, PDPs,
`feature_importance.csv`) come from the **best individual model only** (StackingEnsemble excluded).
If the best individual model is MLP and KernelExplainer times out, fall back to the next-best
non-MLP individual model for SHAP outputs; document the fallback in `results.warnings`.

Subprocess timeout: **300s** for model training steps; **600s** for SHAP computation steps
(implemented as per-step timeouts, not a global limit).

#### Evaluation Protocol

*Classification:*
- Primary: AUC-ROC
- Secondary: Accuracy, Precision, Recall, F1
- CI: 1000-iteration bootstrap on AUC
- Outputs: confusion matrix, ROC curves (all models overlaid), calibration curve

*Regression:*
- Primary: RMSE
- Secondary: MAE, R²
- CI: 1000-iteration bootstrap on RMSE
- Outputs: residual plot (predicted vs. actual)

#### Interpretability Outputs (best-performing model only)
1. SHAP summary plot (beeswarm) → `shap_summary.png`
2. SHAP bar plot (mean |SHAP|) → `shap_importance.png`
3. Partial dependence plots for top 3 features → `pdp_{feature}.png`
4. Subgroup AUC/RMSE for each protected attribute level → `subgroup_performance.csv`

#### Critical Rules
- **NEVER** evaluate on training data
- **NEVER** use test data during hyperparameter tuning (inner CV on train only)
- `random_state = config["pipeline"]["random_state"]` everywhere
- Always generate CIs for the primary metric
- If a model fails to converge, log error in `results.json.errors` and continue

#### `results.json` Schema
```json
{
  "best_model": "string",
  "best_metric_value": 0.0,
  "primary_metric": "AUC|RMSE",
  "all_models": {
    "LogisticRegression": {
      "auc": 0.0, "accuracy": 0.0, "precision": 0.0,
      "recall": 0.0, "f1": 0.0,
      "auc_ci_lower": 0.0, "auc_ci_upper": 0.0
    }
  },
  "top_features": [
    {"feature": "string", "shap_mean_abs": 0.0, "direction": "positive|negative"}
  ],
  "subgroup_performance": {
    "X1SEX": {"Male": {"auc": 0.0, "n": 0}, "Female": {"auc": 0.0, "n": 0}}
  },
  "figures_generated": [],
  "tables_generated": [],
  "errors": [],
  "warnings": []
}
```

---

### 4.4 Critic Agent

| Property | Value |
|---|---|
| Model | `config.models.critic` (default: `claude-opus-4-6`) |
| Temperature | 0.0 |
| Input | `research_spec.json`, `data_report.json`, `results.json`, registry, task template |
| Output | `review_report.json` |

#### Review Checklist

*Problem Formulation:*
- [ ] Research question is specific and answerable
- [ ] All predictors temporally precede outcome (validate against registry wave metadata)
- [ ] Predictor rationales are educationally grounded
- [ ] Novelty claim is supported by `literature_context.novelty_evidence`
- [ ] Target population well-defined
- [ ] Feasibility verified (`analytic_n` ≥ 10,000)

*Data Preparation:*
- [ ] No data leakage (temporal or target)
- [ ] Missing data handling appropriate and documented
- [ ] `analytic_n` ≥ 10× number of predictors
- [ ] Class balance reasonable or addressed
- [ ] No constant predictors remain
- [ ] Train/test split properly stratified
- [ ] Multilevel limitation acknowledged in `data_report.warnings`

*Analysis:*
- [ ] ≥ 5 individual model families present (LR, RF, XGBoost, ElasticNet, MLP) + StackingEnsemble in model_comparison.csv; flag major if < 4 individual models present
- [ ] Hyperparameters tuned via inner CV only
- [ ] All metrics from held-out test set
- [ ] CIs provided for primary metric
- [ ] SHAP analysis present for best individual model; uses correct explainer (LinearExplainer for LR/ElasticNet, TreeExplainer for RF/XGBoost, KernelExplainer with sample cap ≤ 1,000 for MLP); no SHAP for StackingEnsemble
- [ ] StackingEnsemble in model_comparison.csv but NOT in SHAP outputs; if SHAP computed for Stacking, flag as major
- [ ] If best individual model is MLP and KernelExplainer timed out, SHAP outputs use next-best non-MLP model and fallback documented in results.warnings
- [ ] Subgroup analysis conducted for all protected attributes
- [ ] Performance differences > 5% across subgroups flagged

*Substantive Validity:*
- [ ] Top features make educational sense
- [ ] Findings not trivially obvious
- [ ] AUC suspiciously high (> 0.95) → flag as potential leakage
- [ ] Limitations honestly acknowledged
- [ ] Unexpected findings flagged for Writer interpretation

#### Verdict Criteria

| Verdict | Condition |
|---|---|
| PASS | No critical issues, ≤ 2 major issues, quality score ≥ 7 |
| REVISE | Any critical issue OR > 2 major issues; `revision_cycle < max_revision_cycles` |
| ABORT | Fundamental flaw (unanswerable question, `analytic_n` < 1,000, confirmed leakage) |

#### `review_report.json` Schema
```json
{
  "overall_verdict": "PASS|REVISE|ABORT",
  "overall_quality_score": 8,
  "problem_formulation_review": {
    "score": 8,
    "issues": [
      {
        "severity": "critical|major|minor",
        "category": "string",
        "description": "string",
        "recommendation": "string",
        "target_agent": "ProblemFormulator|DataEngineer|Analyst"
      }
    ]
  },
  "data_preparation_review": {"score": 0, "issues": []},
  "analysis_review": {"score": 0, "issues": []},
  "substantive_review": {
    "score": 0,
    "educational_meaningfulness": "string",
    "issues": []
  },
  "revision_instructions": {
    "ProblemFormulator": "string or null",
    "DataEngineer": "string or null",
    "Analyst": "string or null"
  }
}
```

---

### 4.5 Writer Agent

| Property | Value |
|---|---|
| Model | `config.models.writer` |
| Temperature | 0.3 |
| Input | `research_spec.json`, `data_report.json`, `results.json`, `review_report.json`, `literature_context.json`, all tables/figures |
| Output | `paper.tex`, `references.bib` |

#### Output Format

**LaTeX with ACM `acmart` template** (`sigconf` proceedings style):

```latex
\documentclass[sigconf]{acmart}
\usepackage{booktabs}
\usepackage{graphicx}
```

- Tables: `\begin{table}` / `tabular` environment with `\toprule`, `\midrule`, `\bottomrule`
- Figures: `\includegraphics[width=\columnwidth]{filename.png}` + `\caption`
- Citations: `\cite{paperId}` keyed to `literature_context.papers[*].paperId`
- References: generated as `references.bib` (BibTeX from S2 metadata, APA-style format)

**`references.bib` generation**: for each paper in `literature_context.papers`, produce:
```bibtex
@inproceedings{paperId,
  author    = {Last, First and Last2, First2},
  title     = {Paper Title},
  year      = {2023},
  booktitle = {Proceedings of the ...},   % use venue from S2 if available
}
```

**S2 failure fallback**: if `literature_context == null`, Writer uses placeholder citations in
`[Author, Year]` format and generates an empty `references.bib` with a comment explaining the
S2 API was unavailable.

#### UNVERIFIED Flag

If `review_report.overall_verdict != "PASS"` (i.e., max revision cycles exhausted without PASS),
the Writer **must** prepend this block at the top of the paper body:

```latex
\begin{quote}
\textbf{WARNING: This paper has unresolved methodological issues identified by automated
review. The issues listed in the appendix were not resolved within the allowed revision
cycles. Use with caution.}
\end{quote}
```

And append a new section:

```latex
\section*{Appendix: Automated Critic Review Report}
% Paste the full review_report.json here, formatted as a description list
```

#### Paper Structure

**Title format**: `[Method/Approach] for Predicting [Outcome] Using [Key Predictors] in [Context]`

| Section | Words |
|---|---|
| Abstract | 150–250 |
| 1. Introduction | 800–1200 |
| 2. Related Work | 500–800 |
| 3. Methods (Data, Models, Evaluation) | 800–1300 |
| 4. Results (Model Comparison, Feature Importance, Subgroups) | 600–1000 |
| 5. Discussion (Findings, Implications, Limitations, Future Work) | 600–1000 |
| References | — |

**Related Work**: cite ≥ 5 papers from `literature_context.papers`. Summarize each briefly.
Position the present study explicitly against the literature.

**Writing Style Rules:**
1. Active voice wherever possible
2. Numbers reported precisely: `AUC = 0.82, 95% CI [0.79, 0.85]`
3. Connect every statistical finding to educational meaning
4. Use "students" not "subjects" or "observations"
5. Include a sentence in §3 Methods acknowledging automated generation
6. No causal language for correlational findings (never "X causes Y")
7. Honest, specific limitations (incorporate Critic's feedback)
8. Do not hedge unless genuinely uncertain

---

## 5. Orchestrator Design

### 5.1 State Machine

```
INITIALIZED
    │
    ▼
FORMULATING ──fail──► ABORTED
    │
    ▼
ENGINEERING ──validation_failed──► ABORTED
    │
    ▼
ANALYZING
    │
    ▼
CRITIQUING
    ├── PASS ──────────────────────────────────────► WRITING ──► COMPLETED
    ├── REVISE + cycles < max ──► REVISING ──► CRITIQUING (loop)
    ├── REVISE + cycles = max ──────────────────────► WRITING (UNVERIFIED)
    └── ABORT ──────────────────────────────────────► ABORTED
```

### 5.2 Checkpointing

After each stage completes successfully, serialize the full `PipelineContext` to
`{output_dir}/checkpoint.json`.

On Orchestrator startup: if `checkpoint.json` exists in `output_dir`, load it and resume
from `current_state`. Skip all stages listed in `completed_stages`.

```json
{
  "schema_version": "1.0",
  "current_state": "ANALYZING",
  "completed_stages": ["FORMULATING", "ENGINEERING"],
  "revision_cycle": 0,
  "research_spec": {...},
  "literature_context": {...},
  "data_report": {...},
  "results_object": null,
  "review_report": null,
  "paper_text": null,
  "errors": [],
  "log": [
    {"timestamp": "ISO-8601", "agent": "string", "message": "string"}
  ],
  "timestamp": "ISO-8601"
}
```

### 5.3 Topological Revision Cascade

When the Critic issues a REVISE verdict, `_execute_revisions()` determines the **lowest-level
targeted agent** in the dependency graph and re-runs it plus all downstream agents:

```
Dependency graph: ProblemFormulator → DataEngineer → Analyst
```

| Critic targets | Agents re-run |
|---|---|
| ProblemFormulator only | ProblemFormulator → DataEngineer → Analyst |
| ProblemFormulator + DataEngineer | ProblemFormulator → DataEngineer → Analyst |
| DataEngineer only | DataEngineer → Analyst |
| DataEngineer + Analyst | DataEngineer → Analyst |
| Analyst only | Analyst |

**Implementation:**
```python
AGENT_ORDER = ["ProblemFormulator", "DataEngineer", "Analyst"]

def _execute_revisions(self):
    instructions = self.ctx.review_report["revision_instructions"]
    targeted = [a for a in AGENT_ORDER if instructions.get(a)]
    if not targeted:
        return  # no revisions needed (shouldn't happen on REVISE verdict)
    start_idx = AGENT_ORDER.index(targeted[0])  # lowest-level targeted agent
    for agent_name in AGENT_ORDER[start_idx:]:
        self._run_agent(agent_name, revision_instructions=instructions.get(agent_name))
```

### 5.4 JSON Recovery

All LLM responses that are expected to be JSON must be parsed with `parse_llm_json()`:

```python
import re, json

def parse_llm_json(text: str) -> dict:
    """Strip markdown code fences and parse JSON."""
    # Remove opening fence (```json or ```)
    text = re.sub(r'^```(?:json)?\s*\n?', '', text.strip(), flags=re.MULTILINE)
    # Remove closing fence
    text = re.sub(r'\n?```\s*$', '', text.strip(), flags=re.MULTILINE)
    return json.loads(text)  # JSONDecodeError propagates → pipeline logs and ABORTs
```

### 5.5 Base Agent Class

```python
from abc import ABC, abstractmethod
import anthropic, json, subprocess, yaml

class BaseAgent(ABC):
    def __init__(self, context, agent_name: str, config: dict):
        self.ctx = context
        self.agent_name = agent_name
        self.config = config
        self.model = config["models"][agent_name.lower().replace(" ", "_")]
        self.temperature = self._get_temperature()
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    def _get_temperature(self) -> float:
        temps = {
            "problem_formulator": 0.7,
            "data_engineer": 0.0,
            "analyst": 0.0,
            "critic": 0.0,
            "writer": 0.3,
        }
        return temps[self.agent_name.lower().replace(" ", "_")]

    def call_llm(self, user_message: str, max_tokens: int = 8192) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=self.temperature,
            system=self.system_prompt,
            messages=[{"role": "user", "content": user_message}]
        )
        self.ctx.log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "agent": self.agent_name,
            "tokens_used": response.usage.input_tokens + response.usage.output_tokens
        })
        return response.content[0].text

    def execute_code(self, code: str, timeout_s: int = 300) -> dict:
        """Execute generated Python code via configured executor (Docker sandbox or subprocess)."""
        return self._executor.run(
            code=code,
            output_dir=self.ctx.output_dir,
            raw_data_path=getattr(self.ctx, "raw_data_path", None),
            timeout_s=timeout_s,
        )

    def load_registry(self) -> dict:
        path = self.config["paths"]["data_registry"] + f"{self.ctx.dataset_name}.yaml"
        with open(path) as f:
            return yaml.safe_load(f)

    def load_task_template(self) -> dict:
        path = self.config["paths"]["data_registry"] + "task_templates/prediction.yaml"
        with open(path) as f:
            return yaml.safe_load(f)

    @abstractmethod
    def run(self, **kwargs) -> dict:
        pass
```

### 5.6 PipelineContext

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class PipelineContext:
    # Configuration
    dataset_name: str
    raw_data_path: str
    output_dir: str
    max_revision_cycles: int = 2

    # Agent outputs
    research_spec: Optional[dict] = None
    literature_context: Optional[dict] = None   # NEW
    data_report: Optional[dict] = None
    results_object: Optional[dict] = None
    review_report: Optional[dict] = None
    paper_text: Optional[str] = None

    # Pipeline metadata
    current_state: str = "INITIALIZED"
    completed_stages: list = field(default_factory=list)  # NEW
    revision_cycle: int = 0
    errors: list = field(default_factory=list)
    log: list = field(default_factory=list)
```

---

## 6. Inter-Agent Message Formats

### ProblemFormulator → (DataEngineer, Critic, Writer)
```json
{
  "research_spec": {
    "research_question": "string",
    "outcome_variable": "X3TGPAMAT",
    "outcome_type": "continuous",
    "predictor_set": [
      {"variable": "X1TXMTSC", "rationale": "Prior math achievement is the strongest predictor of subsequent GPA", "wave": "base_year"}
    ],
    "target_population": "full sample",
    "subgroup_analyses": ["X1SEX", "X1RACE"],
    "expected_contribution": "string",
    "potential_limitations": ["Multilevel structure not modeled"],
    "novelty_score_self_assessment": 4
  },
  "literature_context": {
    "search_query": "math GPA prediction educational data mining",
    "papers": [
      {"paperId": "abc123", "title": "...", "authors": ["..."], "year": 2022, "abstract": "..."}
    ],
    "novelty_evidence": "string"
  }
}
```

### DataEngineer → (Analyst, Critic)
```json
{
  "dataset": "hsls09_public",
  "original_n": 23503,
  "analytic_n": 19240,
  "n_train": 15392,
  "n_test": 3848,
  "outcome_variable": "X3TGPAMAT",
  "outcome_type": "continuous",
  "n_predictors_raw": 12,
  "n_predictors_encoded": 28,
  "class_balance": null,
  "missingness_summary": {
    "X1TXMTSC": {"pct_missing": 1.2, "imputation_method": "median"}
  },
  "variables_flagged": [],
  "validation_passed": true,
  "warnings": ["Multilevel structure (students nested in schools) is not modeled. This is a limitation."]
}
```

### Analyst → (Critic, Writer)
```json
{
  "best_model": "XGBoost",
  "best_metric_value": 0.614,
  "primary_metric": "RMSE",
  "all_models": {
    "LinearRegression": {"rmse": 0.71, "mae": 0.55, "r2": 0.38, "rmse_ci_lower": 0.69, "rmse_ci_upper": 0.73},
    "RandomForest":     {"rmse": 0.65, "mae": 0.50, "r2": 0.48, "rmse_ci_lower": 0.63, "rmse_ci_upper": 0.67},
    "XGBoost":          {"rmse": 0.61, "mae": 0.47, "r2": 0.53, "rmse_ci_lower": 0.59, "rmse_ci_upper": 0.63}
  },
  "top_features": [
    {"feature": "X1TXMTSC", "shap_mean_abs": 0.18, "direction": "positive"},
    {"feature": "X1MTHEFF", "shap_mean_abs": 0.09, "direction": "positive"}
  ],
  "subgroup_performance": {
    "X1SEX": {
      "Male":   {"rmse": 0.63, "n": 7800},
      "Female": {"rmse": 0.59, "n": 7592}
    }
  },
  "figures_generated": ["roc_curves.png", "shap_summary.png", "shap_importance.png", "pdp_X1TXMTSC.png"],
  "tables_generated": ["model_comparison.csv", "feature_importance.csv", "subgroup_performance.csv"],
  "errors": [],
  "warnings": []
}
```

---

## 7. Output Directory Structure

```
output/
└── run_{timestamp}/
    ├── checkpoint.json           ← stage-level checkpoint (resume support)
    ├── config_snapshot.yaml      ← copy of config.yaml at run time
    │
    ├── research_spec.json        ← ProblemFormulator output
    ├── literature_context.json   ← S2 papers + novelty evidence
    │
    ├── data_report.json          ← DataEngineer output
    ├── train_X.csv
    ├── train_y.csv
    ├── test_X.csv
    ├── test_y.csv
    │
    ├── results.json              ← Analyst output
    ├── model_comparison.csv
    ├── feature_importance.csv
    ├── subgroup_performance.csv
    ├── roc_curves.png            ← classification only
    ├── shap_summary.png
    ├── shap_importance.png
    ├── pdp_{feature1}.png
    ├── pdp_{feature2}.png
    ├── pdp_{feature3}.png
    │
    ├── review_report.json        ← Critic output
    │
    ├── paper.tex                 ← Writer output (ACM acmart LaTeX)
    ├── references.bib            ← BibTeX from S2 metadata
    │
    └── pipeline.log              ← timestamped event log
```

Run directory naming: `run_{YYYYMMDD_HHMMSS}` (e.g., `run_20260310_142300`).

---

## 8. Error Handling

| Condition | Action |
|---|---|
| `data_report.validation_passed == false` | ABORT; log reason; return context |
| `analytic_n < 1000` | ABORT; log reason; return context |
| `JSONDecodeError` from `parse_llm_json()` | Log error; set state to ABORTED; return context |
| Subprocess `returncode != 0` | Log stderr; skip failed model; continue with remaining models |
| SHAP computation timeout (600s) | Log timeout; skip SHAP for that model; note in `results.warnings` |
| KernelExplainer timeout (600s) for MLP | Log timeout; fall back to best non-MLP individual model for all SHAP/interpretability outputs; document fallback in `results.warnings` |
| S2 API error or non-200 response | Log warning; set `literature_context = null`; Writer uses placeholders |
| Max revision cycles reached without PASS | Set UNVERIFIED flag; proceed to WRITING |
| Critic verdict = ABORT | Set state to ABORTED; return context with full Critic report |
| Checkpoint found on startup | Load checkpoint; resume from `current_state`; skip completed stages |
| AUC > 0.95 | Critic automatically flags as suspicious (potential leakage) |
| Docker daemon not reachable (`sandbox.enabled: true`) | Emit RuntimeWarning; fall back to SubprocessExecutor; log warning |
| Docker image not found + `auto_build: true` | Build image from project-root Dockerfile; retry once; fall back to subprocess on build failure |
| Container OOM-killed (exit code 137) | Treated as non-zero returncode; Analyst logs in `results.errors` and continues with remaining models |

---

## 9. Dependencies

```
# requirements.txt
anthropic>=0.40.0
pandas>=2.0
numpy>=1.25
scikit-learn>=1.4
xgboost>=2.0          # primary gradient boosting library; lightgbm>=4.0 as alternative
shap>=0.45
matplotlib>=3.7
seaborn>=0.13
PyYAML>=6.0
requests>=2.31        # Semantic Scholar API calls
docker>=7.0           # Docker SDK for Python (sandbox mode); optional at runtime
```

**Note on Docker Engine**: Docker Engine ≥ 24.0 is required on the host when `sandbox.enabled: true`. The docker Python SDK is optional at import time; if absent or if the daemon is unreachable, the system falls back to SubprocessExecutor automatically.

**Note on SVM**: intentionally excluded from the pilot model battery due to `KernelExplainer` timeout risk without bounded sample cap.

---

## 10. Pilot Scope & Roadmap

### Pilot (v1.0)
- Dataset: HSLS:09 public-use file
- Task type: Prediction only
- Output: LaTeX (ACM acmart, sigconf)
- Model battery: LR + RF + XGBoost + ElasticNet + MLP + StackingEnsemble (6 models)
- SHAP: TreeExplainer + LinearExplainer + KernelExplainer (MLP only, sample cap 1,000)
- Citations: Semantic Scholar API (ProblemFormulator step)
- Sandbox: Docker container (`edm-ars-sandbox:latest`); subprocess fallback when Docker unavailable
- Evaluation: Automated checks; human review protocol aspirational

### Phase 2 — Hardening
- ASSISTments dataset registry (clickstream feature engineering)
- Causal inference task template
- SVM re-added with KernelExplainer (bounded sample cap)

### Phase 3 — Extended Datasets
- PISA 2022 (requires plausible values methodology)
- Fairness audit task template (equalized odds, demographic parity)

### Phase 4 — Human-in-the-Loop
- Interactive checkpoint after research specification
- Interactive checkpoint after data preparation
- Human edit of paper draft with tracked changes

### Phase 5 — Controlled Evaluation
- N = 10 EDM-ARS papers on HSLS prediction questions
- N = 3+ human-authored papers on matched questions
- Blind review by EDM/LAK-familiar researchers
- Comparison metrics: quality score, error rate, time-to-completion

---

*Document generated: 2026-03-10. Supersedes `edm_auto_research_architecture.md`.*
