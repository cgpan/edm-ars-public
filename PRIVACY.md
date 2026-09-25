# Privacy and data handling

This page lists what EDM-ARS sends off your computer, to whom, and what it
keeps locally. It is written so you can hand it to an ethics board or IT
department. It describes the software, not any promise by a third party.

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

Apart from the Docker case above, EDM-ARS downloads nothing by itself. You
download the datasets, LSAR, LaTeX and R packages yourself, and those sites
see ordinary download requests. (If MiKTeX is set to install missing packages
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

## Keys are kept away from generated code

The AI-written analysis code runs on your computer. EDM-ARS removes API keys
and tokens from the environment of that code, so it cannot read your keys
from its environment. It can still read files your user account can read,
including a `.env` file. If that matters to you, set the keys as environment
variables in the terminal you start EDM-ARS from instead of keeping them in
`.env`.

## Deleting everything

Delete the repository folder: that removes the settings, the datasets in
`data/raw/`, the run folders under `output/`, the findings memory and the
`.env` file. Run folders you created elsewhere with `--output-dir`, and your
LSAR checkout, are separate folders you delete yourself. Remove any keys you
saved as environment variables in your shell profile or with `setx`. To
delete data held by an AI provider, use that provider's account tools, and
revoke keys you no longer use from the provider's website.

## Questions

Open an issue at <https://github.com/cgpan/edm-ars-public/issues>. Do not
paste keys, data, run folders, `prompts/` or CSV files into an issue. If you
attach `pipeline.log` or `run_status.json`, read them first.
