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
| **The AI service you choose** (DeepSeek, OpenAI, Anthropic, or your own server) | Your research question; dataset variable names, labels and documentation; summary statistics; the analysis code the AI writes; **error messages and printed output from that code, which can include a few individual data values** (up to about 2,000 characters per failed attempt); the draft paper; titles and abstracts of related papers | Throughout every study |
| **Semantic Scholar, arXiv, Crossref** | Search words derived from your question; titles of papers to check that citations exist | Literature search in every study |
| **The automated reviewer's AI service** (DeepSeek, if you turn LSAR on) | The finished paper's text; search words for related work | After the paper is written, only if LSAR is on |
| **Tavily** (only if you add a Tavily key for LSAR) | Web search words about the paper's topic | LSAR related-work search |
| **NCES, GitHub, TinyTeX, CRAN / Posit mirrors, astral.sh** | Ordinary download requests (your IP address) | Only when you approve an install or download |
| **The key checks** (provider `/models` or balance endpoints, Semantic Scholar) | Your API key, to confirm it works | When you run `edmars setup` or `edmars doctor --deep` |

**The dataset file itself is never uploaded.** The AI never receives the
whole dataset; it receives descriptions of it, results computed from it, and
occasionally small excerpts inside error messages or printed output.

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
| API keys | Your operating system's credential store (Windows Credential Manager, macOS Keychain, or the Linux Secret Service) | Never written into study folders. If no credential store is available, EDM-ARS asks before using a file readable only by your user. |
| Settings | `settings.yaml` in your user configuration folder (`edmars doctor` prints the path) | No secrets |
| Datasets | Your user data folder | Public-use files you downloaded or imported |
| Studies | The studies folder you chose (default `~/EDM-ARS/studies`) | Each study folder contains the paper, figures, results, **row-level data extracts made from the dataset** (train/test files), the generated code, and a `prompts/` folder with a copy of **everything sent to the AI service**. Treat study folders like the dataset itself: do not post them publicly without removing the data extracts and `prompts/`. |

## Keys are kept away from generated code

The AI-written analysis code runs on your computer. EDM-ARS removes API keys
and tokens from the environment of that code, so it cannot read or send your
keys.

## Deleting everything

`edmars uninstall` removes the application, settings and stored keys, and
asks separately about datasets and studies. You can also delete the studies
folder yourself. To delete data held by an AI provider, use that provider's
account tools.

## Questions

Open an issue at <https://github.com/cgpan/edm-ars-public/issues>. Do not
paste keys, data or study folders into an issue; `edmars doctor --bundle`
makes a support file with secrets removed and asks before including anything
else.
