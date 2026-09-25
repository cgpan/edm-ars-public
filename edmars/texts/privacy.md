# Privacy and data handling

This page lists what EDM-ARS sends off your computer, to whom, and what it
keeps locally. It is written so you can hand it to an ethics board or IT
department. It describes the software, not any promise by a third party.

The sections below describe the pipeline as you run it from a copy of the
repository (`python -m src.main`). If you installed EDM-ARS with the
installer and use the `edmars` command, the pipeline sends exactly the same
things; what differs (where keys and files are kept, what `edmars` itself
downloads or checks, and how to delete everything) is in
[If you use the `edmars` command](#if-you-use-the-edmars-command).

## EDM-ARS itself collects nothing

EDM-ARS has **no telemetry, analytics or crash reporting**. Its authors never
receive your questions, data, keys, papers or usage. There is no EDM-ARS
server.

## What is sent, and to whom

| Sent to | What | When |
|---|---|---|
| **The AI service you choose** (DeepSeek, OpenAI, Anthropic, MiniMax, or an OpenAI-compatible server you run) | Your research question or locked research spec; dataset variable names, labels and documentation; summary statistics and results computed from the data; the analysis code the AI writes; **error messages and printed output from that code, which can include a few individual data values** (up to about 3,500 characters per failed attempt); the folder paths of the dataset and of the run on your computer (these can include your user name); a summary of your earlier runs (see *Findings memory* below); the draft paper and, if the review gate is on, the reviewer's comments on it; titles and abstracts of related papers; images of the paper's figures, only if you turn on the optional judge (`verification.judge_enabled`) | Throughout every run |
| **Semantic Scholar, arXiv, Crossref** | Search words derived from your question; titles of papers, to check that citations exist; your Semantic Scholar key and Crossref contact email, if you set them | Literature search in every run |
| **The automated reviewer's AI service** (LSAR; DeepSeek in LSAR's shipped configuration) | The finished paper's text; search words for related work | After the paper is written, only if the review gate is on |
| **Semantic Scholar, arXiv, Crossref** (from LSAR) | Search words about the paper's topic; titles and DOIs of the papers it cites | Only if the review gate is on |
| **Tavily** (only if you set a `TAVILY_API_KEY` for LSAR) | Web search words about the paper's topic | LSAR related-work search |
| **Docker Hub and the Python Package Index** | Ordinary download requests (your IP address) | Only if you turn on the experimental Docker sandbox and its image has to be built |

**The dataset file itself is never uploaded.** The AI never receives the
whole dataset; it receives descriptions of it, results computed from it, and
occasionally small excerpts inside error messages or printed output.

Apart from the Docker case above, the pipeline downloads nothing by itself.
You download the datasets, LSAR, LaTeX and R packages yourself (or let the
`edmars` command do it after you approve each download; see below), and
those sites see ordinary download requests. (If MiKTeX is set to install missing packages
on the fly, compiling a paper can make MiKTeX download LaTeX packages.)

Each AI service handles what it receives under its own terms and privacy
policy, including where it is processed, how long it is kept, and whether it
may be used for training. **Read your provider's policy before you start.**
Some providers process data outside your country (for example, DeepSeek
states that it processes data in the People's Republic of China). Some
institutions restrict particular providers; check yours. If you run a model
on your own computer or server, nothing in the first row leaves your
machine.

## What stays on your computer

| Item | Where | Notes |
|---|---|---|
| API keys | Environment variables, or the `.env` file in the repository folder | `.env` is a plain-text file readable by your user account; git ignores it. Keys are not written into `config.yaml` or run folders. |
| Settings | `config.yaml` in the repository folder | No secrets. Each run folder keeps a copy (`config_snapshot.yaml`). |
| Datasets | `data/raw/` in the repository folder | Public-use files you downloaded |
| Run folders | `output/run_<date>_<time>/` by default, or the folder you give with `--output-dir` | Each contains the paper, figures, results, logs, **row-level data extracts made from the dataset** (for example `train_X.csv` and `test_X.csv`), the generated code, and a `prompts/` folder with a copy of every prompt the pipeline's agents sent to the AI service and every reply. The review gate's revision requests and LSAR's own requests are not copied there, but the paper and the reviews they carry are in the run folder. Treat run folders like the dataset itself: do not post them publicly without removing the data extracts and `prompts/`. |
| Findings memory | `findings_memory/memory.yaml` in the repository folder | The question, variables, headline result and open questions of each finished run. At the start of later runs this summary goes into the prompts sent to the AI service. Set `findings_memory.enabled: false` in `config.yaml` to turn it off. |

## API keys and the AI-written code

The AI-written analysis code runs on your computer, as your user account.
EDM-ARS removes API keys and tokens from that code's own environment, and
from the environment it compiles the paper's LaTeX in. This keeps the keys
out of what the code prints, which is sent back to the AI service when an
attempt fails and saved in the run folder's `prompts/`.

**This is not a security barrier.** Code running as you can still find your
keys elsewhere: in the environment of the EDM-ARS process that started it, in
the `.env` file, in your shell profile, or, for keys saved with `setx` on
Windows, in your user registry. Keeping the keys in environment variables
instead of `.env` does not change this. To limit what a leaked key could cost:

- use a separate key for EDM-ARS, with a spending limit or a small prepaid
  balance at the provider;
- revoke the key and make a new one if you think it was exposed, and when you
  stop using EDM-ARS;
- for isolation, run the generated code in the Docker sandbox
  (`sandbox.enabled: true`), which sees only the run folder and the dataset.
  It is experimental and cannot run every study type; see the README's
  Install section.

## Deleting everything

Delete the repository folder: that removes the settings, the datasets in
`data/raw/`, the run folders under `output/`, the findings memory and the
`.env` file. Run folders you created elsewhere with `--output-dir`, and your
LSAR checkout, are separate folders you delete yourself. Remove any keys you
saved as environment variables in your shell profile or with `setx`. To
delete data held by an AI provider, use that provider's account tools, and
revoke keys you no longer use from the provider's website.

## If you use the `edmars` command

The `edmars` command runs the same pipeline, so everything in *What is
sent, and to whom* above applies. In addition:

**What `edmars` itself sends, and when**

| Sent to | What | When |
|---|---|---|
| **The AI service you choose, and Semantic Scholar** | Your API key, to confirm it works: a request that lists the service's models (for DeepSeek, a second one reads your account balance), or one small Semantic Scholar search | When `edmars setup` checks a key, and in `edmars doctor --deep` |
| **NCES** (nces.ed.gov) | Ordinary download requests (your IP address) | Only when you accept a dataset's terms and download it (`edmars setup`, `edmars data install`) |
| **GitHub and the Python Package Index** | Ordinary download requests | Only when you approve installing the automated reviewer (LSAR) |
| **yihui.org, GitHub and CTAN mirrors** | Ordinary download requests | Only when you approve installing TinyTeX |
| **Posit Package Manager** (packagemanager.posit.co) | Ordinary download requests | Only when you approve installing R packages |
| **GitHub** (api.github.com) | An ordinary request for the latest release number | Only when you run `edmars update` |
| **astral.sh, GitHub and the Python Package Index** | Ordinary download requests | Only while the installer runs (it fetches uv, a private Python, EDM-ARS and its packages) |

`edmars` has no telemetry either, and makes no other network requests.

**What stays on your computer**

| Item | Where | Notes |
|---|---|---|
| API keys | Your operating system's credential store (Windows Credential Manager, macOS Keychain, or the Linux Secret Service), under the name `edm-ars` | Never written into settings or study folders. If no credential store works, `edmars` asks before using a file readable only by your user. A key you set as an environment variable takes priority over a stored one. |
| Settings | `settings.yaml` in your user configuration folder (`edmars doctor` prints the path) | No secrets. Your name and affiliation (for the paper's author line) and a Crossref contact email, if you give them. |
| Datasets, the automated reviewer (LSAR) and the findings memory | Your user data folder (on Windows `%LOCALAPPDATA%\edm-ars`) | The findings memory works as described above; each study's `run_config.yaml` shows where it is. |
| Studies | The studies folder you chose in setup (default `~/EDM-ARS/studies`), one folder per study | The same contents as a run folder above, plus `run_config.yaml` (the settings the study ran with; no secrets), `runner.json` (how it was started, including your question) and `console.log`. Treat study folders like the dataset itself. |
| Support file | Only if you run `edmars doctor --bundle` | A zip with the check results, your settings with your name, email and home folder removed, version numbers and, if you agree when asked, the last study's logs with keys removed. It never includes `prompts/` or data files. Read it before you share it. |

`edmars` hands your keys to the pipeline only through the environment of
the process it starts, and the pipeline removes them from the environment
of the AI-written code, as described in *Keys are kept away from generated
code*. A key kept in the fallback file (see the table) is a file your user
account can read, so that code could read it too.

**Deleting everything**

`edmars uninstall` removes your settings, the keys EDM-ARS stored in your
credential store, the automated reviewer, the findings memory and caches,
and asks separately whether to delete your downloaded datasets and your
study folders. It then lists the program files the installer created (the
program, its private Python, the `edmars` command and any PATH change) for
you to delete once the window is closed, because a running program cannot
delete itself. Keys you set as environment variables, and data held by an AI
provider, are removed the same way as described above.

## Questions

Open an issue at <https://github.com/cgpan/edm-ars-public/issues>. Do not
paste keys, data, run folders, `prompts/` or CSV files into an issue. If you
attach `pipeline.log` or `run_status.json`, read them first. With the
`edmars` command, `edmars doctor --bundle` makes a support file with keys
removed and asks before including any study's logs.
