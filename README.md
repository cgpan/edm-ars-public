# EDM-ARS — Educational Data Mining Automated Research System

A multi-agent pipeline that takes a curated dataset and a research question and
produces a complete, submission-ready LaTeX manuscript: literature retrieved
from Semantic Scholar and arXiv, an analytic sample built by generated code, a
certified estimator battery, an internal critique, and a written paper with real
citations.

Six LLM agents do the work. A deterministic state machine decides what happens
next — **no model decides control flow** — and a layer of checks verifies the
output before anything is called finished.

> **Version 5.** Five study types, ten certified estimators, four curated
> datasets, 70 composable skill units, ~2,100 automated tests. A complete gated
> paper takes 18–46 minutes and costs about **$0.15** in API spend at DeepSeek
> rates (measured, not estimated — see [Cost](#cost)).

---

## Contents

- [What it can study](#what-it-can-study)
- [How it works](#how-it-works)
- [Verification: why this is not just a prompt chain](#verification-why-this-is-not-just-a-prompt-chain)
- [Install](#install)
- [Data setup](#data-setup)
- [API keys](#api-keys)
- [Running the pipeline](#running-the-pipeline)
- [The review gate (optional)](#the-review-gate-optional)
- [Cost](#cost)
- [Repository layout](#repository-layout)
- [Tests](#tests)
- [Known limitations](#known-limitations)
- [Credits and citation](#credits-and-citation)

---

## What it can study

Each family is a workflow the pipeline knows end to end — how to frame the
question, prepare the sample, run the estimator, check its own assumptions, and
write it up.

**Prediction.** Logistic and linear regression, random forest, gradient boosting
(XGBoost), elastic net, neural network (multi-layer perceptron), stacking
ensemble, feature attribution (SHAP), class-imbalance correction (SMOTE), and a
subgroup fairness and calibration audit.

**Causal inference — observational.** Regression adjustment, propensity score
matching (PSM), inverse probability weighting (IPW), doubly robust estimation
(AIPW / TMLE), causal forest for heterogeneous effects (CATE), overlap and
balance diagnostics, and unmeasured-confounding sensitivity analysis.

**Causal inference — change over time.** Difference-in-differences (DiD,
cross-cohort), gap-in-gaps decomposition, composition-adjusted change, effect
heterogeneity via machine learning, and placebo/stability probes.

**Individualised treatment rules.** Optimal treatment regimes (ITR) — *for whom*
is a treatment worth doing, not merely whether it works on average — via
policy-tree learning, doubly robust pseudo-outcomes, and cross-fit policy value
estimation.

**Measurement and psychometrics.** Item response theory (IRT, graded response
model), cognitive diagnosis models (CDM — DINA and generalised DINA),
differential item functioning (DIF), measurement invariance (configural, metric,
scalar), confirmatory factor analysis (CFA), and classical test theory
reliability (Cronbach's alpha, McDonald's omega). Psychometric estimation runs
through R via a bridge — see [Install](#install).

**Certified but not yet runnable:** regression discontinuity (RD) and
instrumental variables (IV). Both recover the correct answer from simulated data
where the truth is known, but no curated dataset here supplies a running
variable with a documented cutoff or a defensible instrument.

---

## How it works

```
dataset + question
        |
        v
  1 ProblemFormulator   research question, predictors, literature (S2 + arXiv)
        v
  2 DataEngineer        generated pandas -> analytic sample, run in a sandbox
        v
  3 Analyst             estimator battery + bootstrap CIs, via certified helpers
        v
  4 Critic              internal review  --REVISE--+
        v                                          |  the cascade re-runs the
  5 OutlineAgent        data-driven outline        |  lowest targeted agent and
        v                                          |  everything downstream
  6 Writer              LaTeX + BibTeX  <----------+
        v
  review gate           calibrated venue threshold (optional)
        v
  paper.tex + references.bib + PDF
```

The orchestrator is a ten-state machine (`INITIALIZED` → `FORMULATING` →
`ENGINEERING` → `ANALYZING` → `CRITIQUING` → `[REVISING]` → `WRITING` →
`REVIEWING` → `COMPLETED` / `ABORTED`). It checkpoints after every stage, so
`--resume` picks up where an interrupted run stopped.

**Skills.** Methodology, dataset quirks, task workflows and writing conventions
live in 70 `SKILL.md` files under `skills/`, matched at runtime and injected
into each agent's system prompt through a `{{SKILLS}}` placeholder. Adding a
capability means adding a skill, not enlarging a prompt.

---

## Verification: why this is not just a prompt chain

The pipeline assumes its own agents will sometimes get things wrong, and checks
them. Every layer below is deterministic code, not an instruction to a model:

| Check | What it prevents |
|---|---|
| Pre-Critic assertions | Structural defects in the results object, caught with no LLM call at all |
| Verdict evaluator | The Critic's own accept/revise call is recomputed from issue counts; the model is overridden when they disagree |
| Numeric reconciliation | Every numeral in a table and every confidence interval must be derivable from the analysis artifacts — invented numbers block the gate |
| UNVERIFIED flag | Injected in code when a run is flagged, so a paper cannot silently present itself as clean |
| Review health | A truncated or empty review cannot produce a passing score |

This exists because it was needed. An earlier release produced a paper that
passed its quality gate while containing a fabricated results table — the
analysis stage had emitted nulls and the Writer filled them in. Instructing the
model not to fabricate did not stop it; the deterministic checker did. The
regression tests in `tests/test_honesty_guards.py` pin the exact historical
failures.

---

## Install

```bash
git clone https://github.com/cgpan/edm-ars-public.git
cd edm-ars-public
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.11 or newer.

**LaTeX** is required to compile papers — TeX Live or MiKTeX, with `pdflatex`
and `biber` on `PATH`.

**R** is required only for the psychometrics task type. Install R 4.4+ and:

```r
install.packages(c("lavaan", "mirt", "psych", "semTools", "difR", "CDM"))
```

**Docker** is optional. Generated analysis code runs in a sandboxed container
when `sandbox.enabled: true`; otherwise it falls back to a subprocess. The
container has no network access.

```bash
docker build -t edm-ars-sandbox:latest .    # or: docker compose build sandbox
```

---

## Data setup

**No data ships with this repository.** The datasets are public but must be
obtained from their sources directly, under those sources' terms of use.

| Dataset | Source | Place it at |
|---|---|---|
| HSLS:09 public-use | <https://nces.ed.gov/surveys/hsls09/> | `data/raw/hsls_17_student_pets_sr_v1_0.csv` |
| ELS:2002 public-use | <https://nces.ed.gov/surveys/els2002/> | `data/raw/els_2002/` |
| ASSISTments 2009–10 | <https://sites.google.com/site/assistmentsdata/> | `data/raw/assistments_0910/` |

The cross-cohort panel (`did_els_hsls_panel`) is built from HSLS:09 and ELS:2002
by `scripts/harmonize_els_hsls.py` once both are present.

Variable registries in `data_registry/datasets/` describe each dataset's
variables, waves, missingness conventions and known pitfalls. To add your own
dataset, start from `scripts/onboard_dataset.py`.

---

## API keys

Keys are read from the environment and are **never** stored in this repository.

```bash
export DEEPSEEK_API_KEY="..."     # default provider
export ANTHROPIC_API_KEY="..."    # if llm_provider: anthropic
export OPENAI_API_KEY="..."       # if llm_provider: openai
```

A local `.env` file also works and is gitignored. Choose the provider and the
per-agent models in `config.yaml` under `llm_provider` and the matching
`<provider>.models` block.

---

## Running the pipeline

```bash
# Validate configuration and exit before any API call is made
python -m src.main --dataset hsls09_public --dry-run

# A full prediction run
python -m src.main --dataset hsls09_public

# Steer the research question
python -m src.main --dataset hsls09_public \
  --prompt "Predict postsecondary enrolment from ninth-grade attitudes"

# Run a locked specification (reproducible; overrides the config task type)
python -m src.main --dataset did_els_hsls_panel \
  --research-spec my_spec.json --output-dir runs/my_run/output

# Resume an interrupted run
python -m src.main --dataset hsls09_public \
  --output-dir runs/my_run/output --resume
```

Set the study type in `config.yaml` (`pipeline.task_type`): `prediction`,
`causal_soo`, `causal_itr`, `causal_did`, or `psychometrics`.

Each run writes `paper.tex`, `references.bib`, the compiled PDF, every
intermediate artifact (`research_spec.json`, `data_report.json`, `results.json`,
`review_report.json`), a full `pipeline.log`, a resumable `checkpoint.json`, and
measured token usage (`token_usage.jsonl`, `run_cost.json`).

---

## The review gate (optional)

Manuscripts can be scored by **LSAR**, a separate automated reviewer that reads
only the compiled PDF and grades it on eight dimensions against thresholds
calibrated from real published papers at a target venue. A paper passes if it
scores at or above the 25th percentile of what that venue actually publishes.

LSAR lives in a **separate repository**. To enable the gate:

```bash
export LSAR_HOME=/path/to/LSAR
```

and set `review_gate.enabled: true` in `config.yaml`. Without it, leave the gate
disabled — the pipeline produces papers perfectly well without being graded.

---

## Cost

Measured from one fully instrumented run (84 API calls, DeepSeek, at the rates
in `config.yaml`):

| | Calls | Cost |
|---|---|---|
| Pipeline (six agents) | 21 | $0.091 |
| Review gate (six sampled reviews) | 63 | $0.055 |
| **One complete paper** | **84** | **$0.146** |

About 74% of input tokens were served from the provider's prompt cache, which is
the single largest lever on cost. Every run records prompt, completion and
cached-token counts separately in `token_usage.jsonl`, and `run_cost.json`
prices them using the rates in `config.yaml`. **A model with no configured rate
reports `null`, never `$0`** — and because raw counts are stored, changing a rate
re-prices historical runs without re-running them.

Verify the rates against your provider's current price list before quoting a
figure; they are operator input, not a measurement.

---

## Repository layout

```
src/                    pipeline source
  agents/               one module per agent, all inheriting BaseAgent
  skills/               runtime skill matching and prompt composition
  ideation/             research-idea screening and ranking (advisory only)
  orchestrator.py       the state machine
  analysis_helpers.py   certified estimators the Analyst calls
  manuscript_linter.py  deterministic post-compile checks
  review_gate.py        calibrated venue gate
  cost.py               token metering and cost accounting
agent_prompts/          system prompts (YAML) — never hardcoded in Python
skills/                 70 SKILL.md units across four layers
data_registry/          dataset registries, task templates, venue norms
templates/              LaTeX templates (ACM sigconf, APA 7 journal)
r_helpers/              certified R scripts for psychometrics
scripts/                onboarding, synthetic-DGP gates, diagnostics
docs/                   design specifications, roadmaps, changelogs
tests/                  ~2,100 tests
```

`SPEC.md` is the authoritative specification. `CLAUDE.md` records the working
rules the project holds itself to.

---

## Tests

```bash
pytest tests/ -q                       # full suite, offline, no API calls
pytest tests/ -q -k "not integration"  # skip integration-marked tests
ruff check src/ tests/                 # lint
mypy src/                              # type check
```

The suite never makes a live API call — provider clients are faked in
`tests/conftest.py`.

---

## Known limitations

Stated plainly, because a research tool that hides them is worth less:

- **The Writer can still fabricate.** Given an empty results field it will
  sometimes invent a plausible value. Detection is reliable; prevention is not
  solved. Do not publish output without reading the lint report.
- **Multilevel structure is not modelled.** Students are nested in schools, and
  the public-use files suppress the identifiers needed for proper multilevel or
  design-based variance estimation. Every paper must state this limitation.
- **The reviewer is noisy.** Test–retest mean absolute difference is about 1.9
  points on a 10-point scale, which is why borderline scores trigger three
  independent reviews and gate on the median.
- **Automated idea ranking does not work yet.** It has been built and measured
  twice, correlates with nothing, and therefore stays advisory and refuses live
  selection in code.
- **Survey weights** are not applied by default. Where design-based estimates
  matter, that is a limitation to state, not a result to report.

Open work is tracked in `docs/backlog.md`.

---

## Credits and citation

This project is developed **in collaboration with
[Claude Code](https://claude.com/claude-code)**, Anthropic's agentic coding
tool, which contributed to the architecture, implementation, verification layers
and documentation throughout.

For the methodology and system design, see the technical report:

> **EDM-ARS: An Automated Research System for Educational Data Mining.**
> <https://arxiv.org/pdf/2603.18273>

```bibtex
@article{pan2026edmars,
  title  = {EDM-ARS: An Automated Research System for Educational Data Mining},
  author = {Pan, Chenguang},
  year   = {2026},
  eprint = {2603.18273},
  url    = {https://arxiv.org/pdf/2603.18273}
}
```

Papers produced by this system carry a fixed author line (EDM-ARS, AI_Name,
Human_Author_Name) and a Methods sentence disclosing automated generation. Replace the
two placeholder names with your own in `templates/paper_template_v2.tex` and
`templates/paper_template_journal.tex`. Please keep the automated-generation
disclosure in anything you publish from it.

## License

MIT — see [LICENSE](LICENSE).
