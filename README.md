# EDM-ARS: Educational Data Mining Automated Research System

A multi-agent pipeline that automates end-to-end prediction-focused EDM research. Given the HSLS:09 dataset and an optional research prompt, it produces a complete, reviewer-ready LaTeX paper with real citations, validated methodology, and interpretable results. See website for a brief introduction: [edmars.ai](https://edmars.ai/).

> **Current version: v1.1.3** — See [What's New in v1.1](#whats-new-in-v11) for AutoResearchClaw-inspired improvements.

---

## Table of Contents

1. [How It Works](#how-it-works)
2. [What's New in v1.1](#whats-new-in-v11)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Data Setup](#data-setup)
6. [Configuration](#configuration)
7. [Running the Pipeline](#running-the-pipeline)
8. [Docker Sandbox](#docker-sandbox)
9. [Resuming an Interrupted Run](#resuming-an-interrupted-run)
10. [Output Structure](#output-structure)
11. [Development](#development)
12. [Troubleshooting](#troubleshooting)

---

## How It Works

Five agents run in sequence, coordinated by a state-machine orchestrator:

```
ProblemFormulator → DataEngineer → Analyst → Critic → Writer
```

| Agent | Role |
|---|---|
| **ProblemFormulator** | Selects a research question, predictor set, and fetches real citations from Semantic Scholar |
| **DataEngineer** | Generates and executes data-cleaning / feature-engineering code; produces train/test splits |
| **Analyst** | Trains 6 model families (LR, RF, XGBoost, ElasticNet, MLP, Stacking), runs SHAP, subgroup analysis |
| **Critic** | Reviews all outputs for methodological correctness; issues PASS / REVISE / ABORT verdict (uses Opus model) |
| **Writer** | Assembles the final ACM `acmart` LaTeX paper and BibTeX reference file |

The Critic can trigger revision cycles (up to `max_revision_cycles` in `config.yaml`). If the pipeline exceeds the cycle limit without a PASS verdict, the paper is marked UNVERIFIED but still produced.

---

## What's New in v1.1

Five targeted improvements inspired by [AutoResearchClaw](https://github.com/aiming-lab/AutoResearchClaw), all independently deployable with no architectural changes.

### 1. Deterministic Pre-Critic Guard (`src/pre_critic_checks.py`)

Before the expensive Opus Critic call, six deterministic checks run with zero LLM calls:

| Check | Severity | What it catches |
|---|---|---|
| `pcc_01` | critical | Outcome variable present in `train_X.csv` header (leakage) |
| `pcc_02` | major | Fewer than 4 individual models in `results.json` |
| `pcc_03` | major | `shap_summary.png` or `shap_importance.png` missing |
| `pcc_04` | major | `results.json.top_features` is empty |
| `pcc_05` | major | `results.json.subgroup_performance` is empty |
| `pcc_06` | critical | `data_report.validation_passed == False` |

On a **critical** failure the pipeline short-circuits — it synthesizes a REVISE/ABORT verdict without any Opus API call, saving cost. On **major** failures the confirmed issues are prepended to the Critic's prompt so Opus does not waste tokens re-deriving them.

### 2. LaTeX Crutch-Phrase Quality Gate (`src/latex_quality.py`)

After the Writer produces a draft, 12 regex patterns scan for incomplete content:

| Pattern | Severity | Catches |
|---|---|---|
| `lq_01` | error | `(not shown)` — suppressed figures |
| `lq_02` | error | `[Insert figure/table here]` — unfilled insert placeholders |
| `lq_03` | error | `TODO` markers |
| `lq_04` | error | `FIXME` markers |
| `lq_05` | error | `\ldots % fill/todo/add` — ellipsis with completion comment |
| `lq_06` | error | `[Author, Year]` / `[Citation needed]` — unfilled citation placeholders |
| `lq_07` | warning | `omitted for brevity` / `results not shown` |
| `lq_08` | warning | `will be discussed` / `to be determined` |
| `lq_09` | error | `%%PLACEHOLDER:…%%` — unfilled template markers |
| `lq_10` | warning | `see supplementary material` / `see appendix for details` |
| `lq_11` | error | `[NEEDS CITATION]` |
| `lq_12` | warning | `described in detail elsewhere` |

If errors are found, the Writer makes one targeted repair attempt, passing all issues back to the LLM with the pipeline data needed to fill them. A **template ratio** (matched characters / total characters) > 0.5% also triggers a repair. All issues are logged to `pipeline.log`.

### 3. Multi-Persona Critic Reasoning (`agent_prompts/critic.yaml`)

The Critic now reasons through three analytical lenses **before** writing its JSON verdict — all within the same single Opus call (zero extra cost):

- **LENS A — METHODOLOGIST**: Examines correctness and rigor (leakage, model battery, CIs, SHAP selection, stratification)
- **LENS B — SKEPTIC**: Challenges what the Methodologist found acceptable; probes for inflated metrics, boilerplate limitations
- **LENS C — SYNTHESIZER**: Weighs both perspectives, resolves disagreements, applies PASS/REVISE/ABORT rules

The scratchpad reasoning is saved to `critic_reasoning.txt` in the output directory for debugging. The JSON verdict always reflects the Synthesizer's conclusion, not the most lenient or harshest view.

### 4. Three-Layer Citation Verification (`src/agents/problem_formulator.py`)

The previous `_filter_hallucinated_papers()` discarded any paper not matching an exact Semantic Scholar ID. The new three-layer version recovers legitimate papers returned under different canonical IDs:

| Layer | Method | Result |
|---|---|---|
| 1 | Exact S2 paper ID match | `VERIFIED` |
| 2 | CrossRef title search (Jaccard ≥ 0.80) | `SUSPICIOUS` |
| 3 | Jaccard ≥ 0.80 against actual S2 result titles | `SUSPICIOUS` |

`HALLUCINATED` papers are silently dropped. `SUSPICIOUS` papers are included but logged. Each paper in `literature_context.papers` gains a `verification_status` field. No new dependencies — `requests` was already present.

### 5. Analyst Error Classification + Targeted Repair (`src/agents/analyst.py`)

Previously every code execution failure triggered the same generic repair prompt. Now `_classify_error(stderr)` maps stderr to one of eight categories, each with a specific fix instruction:

| Error type | Targeted hint |
|---|---|
| `ImportError` | Only use sandbox-available packages; do not import `lightgbm` |
| `MemoryError` | Reduce `n_estimators` ≤ 100; SHAP sample cap ≤ 500 |
| `ConvergenceWarning` | Increase MLP `max_iter` to 1000; LR `max_iter` to 2000 |
| `FileNotFoundError` | All paths must be absolute; use paths from the prompt exactly |
| `SHAPTimeout` | Skip SHAP for MLP; fall back to next-best non-MLP model |
| `ValueError` | Check y array shape, label encoder fit/transform order, NaN values |
| `TypeError` | Ensure all features are numeric; convert sparse matrices to dense |
| `RuntimeError` | Wrap each model block in try/except; log to `results.errors` |

The repair prompt uses the **last 3,000 characters of stderr** for maximum context. If `results.json` was partially written before the crash, the completed models are identified and the LLM is told to preserve them rather than re-train.

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11+ | |
| Docker Engine | 24.0+ | Required for sandboxed code execution (optional — see below) |
| Anthropic API key | — | Primary provider: `claude-sonnet-4-6` + `claude-opus-4-6` |
| *or* MiniMax API key | — | Alternative provider: `MiniMax-M2.5` (see [LLM Provider](#llm-provider)) |
| Semantic Scholar API key | — | — |
| HSLS:09 public-use CSV | — | Obtainable from NCES; see [Data Setup](#data-setup) |

---

## Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd edm-ars

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your API keys

# --- Option A: Anthropic (default) ---
# Windows (PowerShell)
$env:ANTHROPIC_API_KEY = "sk-ant-..."
# macOS / Linux
export ANTHROPIC_API_KEY="sk-ant-..."

# --- Option B: MiniMax (alternative) ---
# Windows (PowerShell)
$env:MINIMAX_API_KEY = "your-minimax-key"
# macOS / Linux
export MINIMAX_API_KEY="your-minimax-key"
# Then set llm_provider: minimax in config.yaml

# --- Semantic Scholar API Key ---
# Windows (PowerShell)
$env:SEMANTIC_SCHOLAR_API_KEY = "your-s2-api-key"
# macOS / Linux
export SEMANTIC_SCHOLAR_API_KEY="your-s2-api-key"
```

> **Never put your API key in code or config files.** The pipeline reads keys exclusively from environment variables (`ANTHROPIC_API_KEY` or `MINIMAX_API_KEY`).

---

## Data Setup

Download the HSLS:09 public-use labeled CSV from NCES and place it at:

```
data/raw/hsls_17_student_pets_sr_v1_0.csv
```

The `data/raw/` directory is gitignored. The file is expected to contain labeled text values (e.g., `"Male"`, `"Female"`) — not raw numeric codes.

---

## Configuration

All pipeline behaviour is controlled by `config.yaml` in the project root. Key settings:

```yaml
models:
  problem_formulator: claude-sonnet-4-6
  data_engineer:      claude-sonnet-4-6
  analyst:            claude-sonnet-4-6
  critic:             claude-opus-4-6    # always Opus; do not change
  writer:             claude-sonnet-4-6

pipeline:
  max_revision_cycles: 2       # Critic can trigger up to 2 revision loops
  random_state: 42
  cost_budget_usd: 5.0         # Soft limit — logs a warning if exceeded, does not abort

sandbox:
  enabled: true                # Set false to skip Docker and use subprocess
  image: "edm-ars-sandbox:latest"
  memory_limit: "4g"
  cpu_count: 2
  network_disabled: true
  auto_build: true             # Builds the image automatically if not found
```

> **Typical run cost is ~$7–8 USD** (above the default `$5` soft limit). Consider raising `cost_budget_usd` to avoid repeated log warnings.

### LLM Provider

The pipeline supports two LLM providers. Set `llm_provider` in `config.yaml`:

| Provider | Value | Models used | Notes |
|---|---|---|---|
| **Anthropic** (default) | `anthropic` | Sonnet for 4 agents, Opus for Critic | Higher quality; recommended for final runs |
| **MiniMax M2.5** | `minimax` | `MiniMax-M2.5` for all 5 agents | Lower cost alternative; uses Anthropic SDK-compatible endpoint |

To switch to MiniMax:

```yaml
# config.yaml
llm_provider: minimax
```

MiniMax M2.5 uses a single model for all agents (including Critic), whereas Anthropic uses the more capable Opus model specifically for the Critic agent. The MiniMax endpoint (`https://api.minimax.io/anthropic`) is Anthropic SDK-compatible, so no code changes are needed — only the config switch and the `MINIMAX_API_KEY` environment variable.

---

## Running the Pipeline

### Basic run

```bash
python -m src.main --dataset hsls09_public
```

This creates a timestamped output directory (`output/run_YYYYMMDD_HHMMSS/`) and runs all five agents.

### With a custom research prompt

Guide the ProblemFormulator toward a specific research question:

```bash
python -m src.main --dataset hsls09_public \
  --prompt "Predict high school dropout risk using baseline socioeconomic and attitudinal factors"
```

### Without Docker (subprocess fallback)

If Docker is unavailable or you want faster iteration without isolation:

1. Set `sandbox.enabled: false` in `config.yaml`, then:

```bash
python -m src.main --dataset hsls09_public
```

Alternatively, Docker is automatically skipped with a warning if the daemon is unreachable.

### All CLI options

```
python -m src.main [OPTIONS]

Options:
  --dataset     Dataset name (default: hsls09_public)
  --output-dir  Reuse an existing output directory (required when resuming)
  --config      Path to config.yaml (default: config.yaml)
  --resume      Resume from a checkpoint in the given --output-dir
  --prompt      Optional free-text research direction passed to ProblemFormulator
```

---

## Docker Sandbox

The Docker sandbox runs all LLM-generated data and analysis code in an isolated container with:
- No network access
- 4 GB memory cap, 2 CPU cap
- Non-root `sandbox` user

### Build the sandbox image

```bash
# Option A — direct docker build
docker build -t edm-ars-sandbox:latest .

# Option B — via Compose
docker compose build sandbox
```

With `auto_build: true` in `config.yaml`, the image is built automatically on first run if not found. If the build fails, the pipeline transparently falls back to subprocess execution.

### Verify the image

```bash
docker image inspect edm-ars-sandbox:latest
```

---

## Resuming an Interrupted Run

The orchestrator writes a `checkpoint.json` after each stage completes. To resume from where you left off:

```bash
python -m src.main \
  --dataset hsls09_public \
  --output-dir output/run_YYYYMMDD_HHMMSS \
  --resume
```

Without `--resume`, any existing checkpoint in the target directory is deleted and the run starts fresh.

> **Tip:** If a run was aborted and you need to re-run from a specific stage, open the `checkpoint.json` in the output directory, set `current_state` to the desired stage (e.g., `"ANALYZING"`), and remove that stage from `completed_stages` before resuming.

---

## Output Structure

Each run produces a self-contained directory:

```
output/run_YYYYMMDD_HHMMSS/
│
├── config_snapshot.yaml        # Copy of config.yaml at run time (reproducibility)
├── checkpoint.json             # Stage-level checkpoint (resume support)
├── pipeline.log                # Timestamped event log for all agents
│
├── research_spec.json          # ProblemFormulator: research question + predictor set
├── literature_context.json     # ProblemFormulator: Semantic Scholar citations
│
├── data_report.json            # DataEngineer: missingness summary, validation status
├── train_X.csv                 # DataEngineer: training features
├── train_y.csv                 # DataEngineer: training labels
├── test_X.csv                  # DataEngineer: test features (never seen during tuning)
├── test_y.csv                  # DataEngineer: test labels
│
├── results.json                # Analyst: model comparison, top features, subgroup results
├── model_comparison.csv        # Analyst: metrics for all 6 models
├── feature_importance.csv      # Analyst: SHAP mean |value| per feature
├── subgroup_performance.csv    # Analyst: AUC/RMSE broken down by protected attributes
├── roc_curves.png              # Analyst: overlaid ROC curves (classification tasks only)
├── shap_summary.png            # Analyst: SHAP beeswarm plot (best individual model)
├── shap_importance.png         # Analyst: SHAP bar chart
├── pdp_<feature>.png           # Analyst: partial dependence plots (top 3 features)
│
├── review_report.json          # Critic: PASS / REVISE / ABORT verdict + issue list
├── critic_reasoning.txt        # Critic: LENS A/B/C scratchpad (v1.1; omitted if no preamble)
│
├── paper.tex                   # Writer: full ACM acmart LaTeX paper
└── references.bib              # Writer: BibTeX entries from Semantic Scholar
```

### Compiling the paper

```bash
cd output/run_YYYYMMDD_HHMMSS
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

Requires a LaTeX distribution with the `acmart` package (e.g., TeX Live 2023+).

---

## Development

### Run tests

```bash
# Unit tests only (no API calls)
pytest tests/ -v -k "not integration"

# All tests including integration (requires ANTHROPIC_API_KEY)
pytest tests/ -v --run-integration
```

### Lint and type-check

```bash
ruff check src/ tests/
mypy src/
```

### Project layout

```
src/
├── main.py                  # CLI entry point
├── orchestrator.py          # State-machine coordinator
├── context.py               # PipelineContext dataclass + checkpoint serialization
├── config.py                # Config loader
├── sandbox.py               # DockerSandbox + SubprocessExecutor
├── registry.py              # Data registry loader
├── analysis_helpers.py      # Shared ML utilities
├── pre_critic_checks.py     # v1.1: Deterministic pre-Critic guard (6 checks, zero LLM)
├── latex_quality.py         # v1.1: LaTeX crutch-phrase quality gate (12 patterns)
└── agents/
    ├── base.py               # BaseAgent (call_llm, execute_code, load_registry)
    ├── problem_formulator.py # v1.1: 3-layer citation verification
    ├── data_engineer.py
    ├── analyst.py            # v1.1: error classification + targeted repair prompts
    ├── critic.py             # v1.1: multi-persona JSON extraction + pre-critic integration
    └── writer.py             # v1.1: LaTeX quality scan + repair loop

agent_prompts/               # YAML system prompts for each agent (never hardcoded in Python)
  critic.yaml                # v1.1: 3-lens structured reasoning protocol
data_registry/               # Variable registries, task templates, evaluation rubrics
templates/                   # ACM acmart LaTeX skeleton
tests/                       # pytest test suite (319 unit tests)
```

---

## Troubleshooting

### `FileNotFoundError` for the CSV

Ensure the HSLS:09 file is at the exact path `data/raw/hsls_17_student_pets_sr_v1_0.csv` relative to the project root. The pipeline uses absolute paths internally, so run from the project root directory.

### Docker daemon not reachable

The pipeline will emit a `RuntimeWarning` and automatically fall back to subprocess execution. No action required unless you specifically need the sandbox isolation.

### `ANTHROPIC_API_KEY` not set

```
anthropic.AuthenticationError: No API key provided.
```

Set the environment variable before running (see [Installation](#installation)).

### Run aborted by Critic

Check `output/<run>/review_report.json` for the `overall_verdict` and the `issues` lists. An `ABORT` verdict means a fundamental flaw was detected (e.g., analytic sample below 1,000 rows, or confirmed data leakage). Inspect `pipeline.log` for the full event trace.

### High API cost / budget warning

Raise `cost_budget_usd` in `config.yaml`. The budget is a soft limit — the pipeline logs a warning but does not abort. A full run with 2 revision cycles typically costs ~$7–8 USD.

### Resuming into wrong stage

Edit `checkpoint.json` in the output directory: set `current_state` to the desired stage name and remove it from `completed_stages`, then re-run with `--resume`.

---

## Notes

- The Critic agent always uses `claude-opus-4-6`; all other agents use `claude-sonnet-4-6`.
- Test set is always 20% of the analytic sample, stratified for classification tasks.
- The outcome variable is never imputed — rows with missing outcomes are dropped.
- All random operations use `random_state: 42`.
- The pre-Critic guard (`src/pre_critic_checks.py`) short-circuits on critical failures before any Opus call.
- LaTeX quality warnings appear in `pipeline.log` under the key `"LaTeX quality warning: …"`.
- `critic_reasoning.txt` contains the Critic's LENS A/B/C scratchpad when present.
- See [SPEC.md](SPEC.md) for the full system specification and inter-agent message schemas.

