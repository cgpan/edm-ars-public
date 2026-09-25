# EDM-ARS — Educational Data Mining Automated Research System

A multi-agent pipeline that takes a curated dataset and a research question and
produces a complete draft LaTeX manuscript: literature retrieved from Semantic
Scholar and arXiv, an analytic sample built by generated code, a certified
estimator battery, an internal critique, and a written paper with real
citations.

Six LLM agents do the work. A deterministic state machine decides what happens
next — **no model decides control flow** — and a layer of checks verifies the
output before anything is called finished.

> **Version 5.** Five study types, ten certified estimators, four curated
> datasets, 70 composable skill units, ~3,000 automated tests. A complete gated
> paper takes 18–46 minutes and costs about **$0.15** in API spend at DeepSeek
> rates (token counts measured on one instrumented run, priced at the rates in
> `config.yaml`, one of which is not yet verified — see [Cost](#cost)).

## Disclaimer and privacy

Everything EDM-ARS produces is an **AI-generated draft**. Numbers, citations and
claims can be wrong: check every one yourself before you share, submit, publish
or act on anything, and keep the automated-generation disclosure in the paper.

EDM-ARS runs AI-written Python code **on your computer, without an isolated
sandbox**, and sends your question, descriptions of the data, results computed
from it and error output (which can contain a few data values) to the AI service
you choose. It collects no telemetry. Read [DISCLAIMER.md](DISCLAIMER.md) and
[PRIVACY.md](PRIVACY.md) before your first run; running EDM-ARS means you accept
them.

---

## Contents

- [Disclaimer and privacy](#disclaimer-and-privacy)
- [What it can study](#what-it-can-study)
- [How it works](#how-it-works)
- [Verification: why this is not just a prompt chain](#verification-why-this-is-not-just-a-prompt-chain)
- [Easiest install (preview)](#easiest-install-preview)
- [Prerequisites](#prerequisites)
- [Install](#install)
- [Data setup](#data-setup)
- [API keys](#api-keys)
- [Running the pipeline](#running-the-pipeline)
- [When a run ends: exit codes, resuming, re-running](#when-a-run-ends-exit-codes-resuming-re-running)
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
through R via a bridge — see [Prerequisites](#prerequisites).

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
  2 DataEngineer        generated pandas -> analytic sample, run on your computer
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
  verification          deterministic checks over the finished manuscript
        v
  paper.tex + references.bib + PDF
```

The orchestrator is a twelve-state machine (`INITIALIZED` → `FORMULATING` →
`ENGINEERING` → `ANALYZING` → `CRITIQUING` → `[REVISING]` → `WRITING` →
`[REVIEWING]` → `VERIFYING` → `COMPLETED` / `INCOMPLETE` / `ABORTED`). It
checkpoints after every stage, so `--resume` picks up where an interrupted or
failed run stopped (see [When a run ends](#when-a-run-ends-exit-codes-resuming-re-running)).

**Verification.** `VERIFYING` holds the finished manuscript against the run's
own artifacts and writes `invariants.json`, `obligations.json` and
`run_status.json` beside the paper. The checks in `src/invariants.py` are
deterministic and use no model. The 38 of them recompute arithmetic the paper
states, try to bind every numeral in the prose to a number the run actually
produced, catch a macro-averaged score described as a positive-class one, catch
a limitation claiming a variable was unavailable when the run used it, check
that every figure on disk is referenced and every figure referenced exists, and
catch a `\begin{env}` the manuscript never closes. `verification.blocking`
ships `false` — a critical finding is recorded, not fatal — because a check has
no business stopping a run until somebody has measured its false-positive rate.

One check is promoted. `INV_LATEX_NO_PDF` — pdflatex said it produced no PDF,
none sits beside the log that proves one was attempted, or pdflatex never ran at
all (not installed or not on `PATH`, so there is a manuscript with neither a log
nor a PDF beside it) — is listed in `verification.blocking_codes`, so it ends
the run `INCOMPLETE` and exits 2. It earned that: zero false positives across
every archived manuscript and every template, and the finding it makes is the
least arguable one available, namely that the deliverable does not exist. (That
measurement predates the never-ran case; an archived folder that kept
`paper.tex` but lost both its log and its PDF would now be flagged.) Note that setting `blocking_codes` at all
overrides `blocking`: only the listed codes stop a run.

An optional LLM judge (`src/agents/verifier.py`, `verification.judge_enabled`)
reads the manuscript with vision and reports what the deterministic checks
cannot reach. It is off by default: it costs a call per run plus one per
figure, and its findings are opinions that a validator has to filter before
they are worth anything.

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

## Easiest install (preview)

A one-command installer and an `edmars` command (a setup wizard, a guided
"new study" flow, a live progress view and plain-language results) are in
development on the `feat/edmars-cli` branch. They are not on this branch yet;
until they land, install by hand as described below.

---

## Prerequisites

| You need | For | Notes |
|---|---|---|
| **Python 3.11 or 3.12** | everything | Not 3.13 or newer: the pinned `dowhy<0.13` cannot be installed there. |
| **An API key** for an AI provider | everything | DeepSeek is the default. See [API keys](#api-keys). |
| **LaTeX** with `pdflatex`, `bibtex` and `biber` on `PATH` | the PDF | Conference papers (the default, `writer.venue_format: conference`) use the `acmart` class with BibTeX; journal manuscripts (`writer.venue_format: journal`) use `apa7` with `biblatex-apa` and `biber`. See below. |
| **R 4.4 or newer** with `jsonlite`, `lavaan`, `mirt` and `CDM` | psychometric studies only | `MASS` ships with R. See below. |
| **About 16 GB of RAM** | HSLS:09 studies | The 2 GB HSLS:09 CSV takes about 6.6 GB of memory once loaded. |
| **About 3 GB of free disk**, plus LaTeX | everything | Python packages about 1.2 GB; HSLS:09 is a 0.3 GB download that unzips to 2 GB. |
| [LSAR](https://github.com/cgpan/LSAR-public) | the optional review gate | See [The review gate](#the-review-gate-optional). |
| Docker | nothing by default | An experimental sandbox; see [Install](#install). |

**LaTeX.** TeX Live (full scheme) or MiKTeX has everything; in MiKTeX
Console, set missing-package installation to "Always install missing packages
on-the-fly". With a minimal distribution
(BasicTeX, TinyTeX), install the classes and bibliography tools, then add
anything else `pdflatex` reports missing:

```bash
tlmgr install acmart apa7 biblatex biblatex-apa biber
```

Without LaTeX the analysis still runs, but no PDF is produced and the run ends
`INCOMPLETE`.

**R** is needed only for psychometric studies. Install R 4.4 or newer, then in R:

```r
install.packages(c("jsonlite", "lavaan", "mirt", "CDM"))
```

EDM-ARS looks for `Rscript` on `PATH` and in the usual install folders. The
Windows installer does not put R on `PATH`; if R is not found, set
`r_bridge.rscript_path` in `config.yaml` (or the `EDM_ARS_RSCRIPT` environment
variable) to the full path of `Rscript`, for example
`C:/Program Files/R/R-4.5.1/bin/Rscript.exe`.

---

## Install

macOS / Linux (bash or zsh):

```bash
git clone https://github.com/cgpan/edm-ars-public.git
cd edm-ars-public
python3.12 -m venv .venv            # or python3.11
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                # then put your key in .env (see API keys)
```

Windows (PowerShell):

```powershell
git clone https://github.com/cgpan/edm-ars-public.git
cd edm-ars-public
py -3.12 -m venv .venv              # or: py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env         # then put your key in .env (see API keys)
```

`py` is the Python launcher that comes with the python.org installer. If
PowerShell refuses to run `Activate.ps1`, skip that line and type
`.\.venv\Scripts\python.exe` wherever this README says `python` (and
`.\.venv\Scripts\python.exe -m pip` for `pip`).

**Docker (experimental, off by default).** With `sandbox.enabled: true` the
generated code runs in a container with no network access. No validated run
has used it, the image has no R (so psychometric studies cannot run in it), and
its 4 GB memory limit is below what a full HSLS:09 load needs. Leave it off
unless you are testing it. To build the image by hand:

```bash
docker build -t edm-ars-sandbox:latest .    # or: docker compose build sandbox
```

---

## Data setup

**No data ships with this repository.** The datasets are public but must be
obtained from their sources directly, under those sources' terms of use (for
NCES data: make no attempt to identify anyone, and cite the source). Each file
must sit at exactly the path below; a run checks this before it spends
anything.

| Dataset | Save it as | Format |
|---|---|---|
| HSLS:09 public-use student file (2017 release) | `data/raw/hsls_17_student_pets_sr_v1_0.csv` | CSV with **value labels** |
| ELS:2002 public-use base-year to third follow-up student file | `data/raw/els_2002/els_02_12_byf3pststu_v1_0.csv` | CSV with numeric codes |
| ASSISTments 2009–10 skill-builder data | `data/raw/assistments_0910/skill_builder_0910.csv` | CSV, 525,534 rows |
| ELS:2002 × HSLS:09 cross-cohort panel | `data/raw/did_els_hsls_panel/panel.csv` | built by a script, below |

**HSLS:09** comes from NCES as a direct download:
<https://nces.ed.gov/EDAT/Data/Zip/HSLS_2017_PETS_SR_v1_0_CSV_Datasets.zip>
(about 297 MB; the CSV inside is about 2 GB). Take
`hsls_17_student_pets_sr_v1_0.csv` out of the zip and put it in `data/raw/`.
EDM-ARS needs this **labelled CSV**, whose cells hold text such as `Male`,
`Yes` or `Unit non-response`. The SPSS, Stata and R versions of the download,
or a CSV of numeric codes, will not work: the variable registry and the
prompts assume the labels.

macOS / Linux:

```bash
mkdir -p data/raw
curl -L -o hsls.zip https://nces.ed.gov/EDAT/Data/Zip/HSLS_2017_PETS_SR_v1_0_CSV_Datasets.zip
unzip -j hsls.zip '*hsls_17_student_pets_sr_v1_0.csv' -d data/raw/
rm hsls.zip
```

Windows (PowerShell):

```powershell
New-Item -ItemType Directory -Force data\raw | Out-Null
$ProgressPreference = 'SilentlyContinue'    # makes the download much faster
Invoke-WebRequest https://nces.ed.gov/EDAT/Data/Zip/HSLS_2017_PETS_SR_v1_0_CSV_Datasets.zip -OutFile hsls.zip
Expand-Archive hsls.zip -DestinationPath hsls_zip
Get-ChildItem hsls_zip -Recurse -Filter hsls_17_student_pets_sr_v1_0.csv | Move-Item -Destination data\raw\
Remove-Item hsls_zip -Recurse; Remove-Item hsls.zip
```

**ELS:2002.** Export the public-use BY–F3 student file as CSV from NCES
(<https://nces.ed.gov/surveys/els2002/>, via the EDAT data tool) and save it
under the name above. This file stores numeric codes; that is expected.

**ASSISTments 2009–10.** Download the skill-builder data from
<https://sites.google.com/site/assistmentsdata/>, choosing the version with
525,534 rows (the one the registry was built on), and rename it to
`skill_builder_0910.csv`. Cite the dataset as the site asks.

**Cross-cohort panel** (`did_els_hsls_panel`, for difference-in-differences):
once HSLS:09 and ELS:2002 are both in place, build it with

```bash
python scripts/harmonize_els_hsls.py
```

Variable registries in `data_registry/datasets/` describe each dataset's
variables, waves, missingness conventions and known pitfalls. To add your own
dataset, start from `scripts/onboard_dataset.py`.

---

## API keys

Keys are read from environment variables, or from a `.env` file in the
repository folder. They are **never** stored in this repository: `.env` is
gitignored, and `.env.example` (which is tracked) holds only empty values.

| Variable | Needed | What for |
|---|---|---|
| `DEEPSEEK_API_KEY` | with `llm_provider: deepseek` (the default), and for the review gate | all agents |
| `OPENAI_API_KEY` | with `llm_provider: openai` | all agents |
| `OPENAI_BASE_URL` | optional | your own OpenAI-compatible server |
| `ANTHROPIC_API_KEY` | with `llm_provider: anthropic` | all agents |
| `SEMANTIC_SCHOLAR_API_KEY` | optional, recommended | literature search; without it requests share an anonymous pool that is often rate-limited ([request a key](https://www.semanticscholar.org/product/api#api-key-form)) |
| `CROSSREF_MAILTO` | optional | a contact email for Crossref citation checks |
| `LSAR_HOME` | for the review gate | folder of your LSAR checkout |
| `EDM_ARS_RSCRIPT` | if R is not found | full path to `Rscript` |

**Option 1 — a `.env` file.** Copy `.env.example` to `.env` (the install steps
above do this) and fill in the values you use, for example
`DEEPSEEK_API_KEY=sk-...`. Leave the rest empty.

**Option 2 — environment variables.** For the current terminal only:

```bash
export DEEPSEEK_API_KEY="..."            # macOS / Linux
```

```powershell
$env:DEEPSEEK_API_KEY = "..."            # Windows PowerShell
```

To keep a variable for future terminals, add the `export` line to `~/.zshrc` or
`~/.bashrc`, or on Windows run `setx DEEPSEEK_API_KEY "..."` once (it applies to
terminals opened afterwards). A variable set in the terminal wins over the same
name in `.env`. The AI-written analysis code runs with API keys removed from its
own environment, so they do not show up in what it prints. That is not a
barrier: code running as your user can still reach keys kept either way, so
use a separate key with a spending limit; see [PRIVACY.md](PRIVACY.md).

Choose the provider in `config.yaml` with `llm_provider`. Each provider has a
block naming the model for every agent (`deepseek.models`, `minimax.models`,
and `models` for Anthropic). For OpenAI, uncomment the `openai:` block in
`config.yaml` and name a model for each agent; there is no built-in default.

---

## Running the pipeline

These commands are the same in bash and PowerShell. Run them from the
repository folder with the virtual environment active.

```bash
# Check the setup: config, data file, key, LaTeX, R (psychometrics), LSAR (gate).
# Makes no API call and writes nothing.
python -m src.main --dry-run

# A prediction study; the ProblemFormulator picks the question
python -m src.main --dataset hsls09_public

# Steer the question
python -m src.main --dataset hsls09_public --prompt "Predict postsecondary enrolment from ninth-grade attitudes"

# A causal or psychometric study from a locked research spec
python -m src.main --research-spec runs/fixtures/spec_x1mtheff_x4college.json --output-dir output/my_causal_run

# Continue a run that stopped
python -m src.main --output-dir output/my_causal_run --resume
```

While a run works it prints one short progress line per stage, wait, retry
and warning (`--quiet` turns these off); `pipeline.log` in the run folder has
the full record. `--debug` shows Python tracebacks for errors that are
otherwise reported in one line.

**Study types.** Prediction studies start from a free-text question (or none).
The other four study types need a locked research spec, a JSON file that names
its study type, dataset, variables and methods; the spec overrides
`pipeline.task_type` in `config.yaml`. Example specs in `runs/fixtures/`:

| Spec | Study type | Dataset |
|---|---|---|
| `spec_x1mtheff_x4college.json` | `causal_soo` (observational causal effect) | HSLS:09 |
| `spec_x1mtheff_itr.json` | `causal_itr` (individualised treatment rule) | HSLS:09 |
| `spec_did_ses_gap_v2.json` | `causal_did` (difference-in-differences) | cross-cohort panel |
| `spec_psy_hsls_matheff_invariance.json` | `psychometrics` (invariance, DIF) | HSLS:09 |
| `spec_psy_els_mathse_calibration.json` | `psychometrics` (IRT calibration) | ELS:2002 |
| `spec_psy_assistments_cdm.json` | `psychometrics` (cognitive diagnosis) | ASSISTments |

Each run writes, into its folder (`output/run_<date>_<time>/` unless you pass
`--output-dir`): `paper.tex`, `references.bib`, the compiled PDF, every
intermediate artifact (`research_spec.json`, `data_report.json`,
`results.json`, `review_report.json`), a full `pipeline.log`, a resumable
`checkpoint.json`, `run_status.json`, a copy of every prompt and reply under
`prompts/`, and measured token usage (`token_usage.jsonl`, `run_cost.json`).

---

## When a run ends: exit codes, resuming, re-running

Every run writes `run_status.json`: its final `state`, whether it was
`released`, a machine-readable `reason_code`, and for a run that stopped an
`abort` block naming the stage, the cause (`code`), a message, and whether it
is `resumable`. `reason_code` is `CLEAN`, or names what kept the run from being
clean: `ADVISORY_FINDINGS`, `CRITIC_UNVERIFIED` (the paper carries the
UNVERIFIED warning), `GATE_FAILED`, `GATE_NOT_RUN`, `BLOCKING_FINDINGS`,
`VERIFICATION_NOT_RUN`, `ABORTED` or `INTERRUPTED`. The process exit code says
the same thing:

| Exit code | Meaning | What to do |
|---|---|---|
| `0` | Finished and released (`COMPLETED`). | Read the paper and check it. |
| `1` | Nothing ran: a usage or setup problem, such as a missing data file, key or config value. | Fix what the message names and start again. |
| `2` | Finished but not releasable (`INCOMPLETE`), for example no PDF was produced. | See `reason_code` and `invariants.json`. |
| `3` | Stopped by a failure (`ABORTED`). | See `abort` in `run_status.json`. If `resumable` is true (no credit, network error, rate limit, and similar), fix the cause and resume. |
| `4` | Interrupted (`INTERRUPTED`): Ctrl-C or the process was terminated. | Resume. The checkpoint was kept. |
| `5` | Crashed on an unexpected error. | The traceback is in `crash.log` in the run folder. Resume once fixed, or report it. |

**Resuming.** `--resume` always needs `--output-dir` pointing at the run
folder. The run continues after its last completed stage, or from the stage
that failed if it was aborted, and reads its dataset, study type and locked
spec from `checkpoint.json`; flags that disagree with the checkpoint are
reported and the checkpoint wins. A `COMPLETED` or `INCOMPLETE` run is finished
and is not run again.

**Re-running into the same folder.** Starting a new run with `--output-dir`
pointing at a folder that already holds a run is refused, so a finished or
half-finished run is never lost by accident. Add `--resume` to continue it, or
`--overwrite` to replace it (this removes only that run's own files).

---

## The review gate (optional)

Manuscripts can be scored by **LSAR**, a separate automated reviewer that reads
only the compiled PDF and grades it on eight dimensions against thresholds
calibrated from real published papers at a target venue. A paper passes if it
scores at or above the 25th percentile of what that venue actually publishes.

LSAR is public at <https://github.com/cgpan/LSAR-public>. It runs inside the
EDM-ARS process, so its Python packages must be installed into the same
environment. From the EDM-ARS folder:

macOS / Linux:

```bash
git clone https://github.com/cgpan/LSAR-public.git ../LSAR
pip install -r requirements-lsar.txt       # or: pip install -r ../LSAR/requirements.txt
export LSAR_HOME="$(cd ../LSAR && pwd)"    # or put LSAR_HOME=<that folder> in .env
```

Windows (PowerShell):

```powershell
git clone https://github.com/cgpan/LSAR-public.git ..\LSAR
pip install -r requirements-lsar.txt       # or: pip install -r ..\LSAR\requirements.txt
$env:LSAR_HOME = (Resolve-Path ..\LSAR).Path   # or put LSAR_HOME=<that folder> in .env
```

Then set `review_gate.enabled: true` in `config.yaml`. LSAR's review and
scoring models are pinned to DeepSeek, the models its thresholds were
calibrated with, so the gate needs `DEEPSEEK_API_KEY` even when the pipeline
itself uses another provider. A `TAVILY_API_KEY` is optional (LSAR's web
search). `--dry-run` reports whether LSAR can be loaded. A gate that could not
run is recorded as not run (`reason_code: GATE_NOT_RUN`), never as a score of
zero. Without LSAR, leave the gate disabled — the pipeline produces papers
perfectly well without being graded.

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
figure; they are operator input, not a measurement. One shipped rate,
`deepseek-flash` (the outline and verifier stages), is marked `verified: false`;
a run that uses a model whose rate is unverified never has its cost labelled
`measured` in `run_cost.json`: it says `estimated`, or `partial` when some
model has no rate at all. `pipeline.cost_budget_usd` only logs a warning; it
never stops a run, so set a spending limit with your provider.

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
runs/                   example research specs (fixtures/) and run configs (configs/)
scripts/                onboarding, synthetic-DGP gates, diagnostics
tests/                  ~3,000 tests
```

`SPEC.md` is the original design specification; where it and `config.yaml`
disagree (models, providers, the sandbox default), `config.yaml` and the code
are current. `CLAUDE.md` records the working rules the project holds itself to.

---

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q                       # full suite, offline, about 15 minutes
python -m pytest tests/ -q -k "not integration"  # skip integration-marked tests
ruff check src/ tests/                           # lint (reports known findings; not yet a gate)
mypy src/                                        # type check (not yet clean)
```

The suite never makes a live API call — provider clients are faked in
`tests/conftest.py`. Tests that need R and its packages are skipped when those
are not installed.

---

## Known limitations

Stated plainly, because a research tool that hides them is worth less:

- **The Writer can still fabricate.** Given an empty results field it will
  sometimes invent a plausible value. Detection is reliable; prevention is not
  solved. Do not publish output without reading the lint report.
- **Generated code is not sandboxed by default.** AI-written analysis code runs
  on your computer with your permissions; see [DISCLAIMER.md](DISCLAIMER.md).
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

Open work is tracked internally; this mirror ships the code and its tests rather than the planning documents behind them.

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
@techreport{pan2026edmars,
  title={{EDM-ARS}: A Domain-Specific Multi-Agent System for Automated Educational Data Mining Research},
  author={Pan, Chenguang and Zhang, Zhou and Xiao, Weixuan and Yao, Chengyuan},
  institution={arXiv},
  number={arXiv:2603.18273},
  type={Technical Report},
  year={2026},
  url={https://arxiv.org/abs/2603.18273}
}
```

Papers produced by this system name EDM-ARS as an author and carry a Methods
sentence disclosing automated generation. By default EDM-ARS is the only author.
To add people to a conference paper (the default format), uncomment the AI and
human author blocks in `templates/paper_template_v2.tex` and fill them in. A
journal paper (`writer.venue_format: journal`) takes its byline from
`paper: authors: [...]` in `config.yaml` (there is a commented example); leave
the byline in `templates/paper_template_journal.tex` as it is, because the
Writer fills it. Please keep EDM-ARS in the byline and the automated-generation
disclosure in anything you publish from it.

## License

MIT — see [LICENSE](LICENSE).
