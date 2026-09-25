# EDM-ARS: Educational Data Mining Automated Research System

## What
Multi-agent pipeline that automates educational data mining research on
curated public-use datasets. Five study types: prediction (`prediction`),
observational causal inference (`causal_soo`), individualised treatment rules
(`causal_itr`), cross-cohort difference-in-differences (`causal_did`) and
psychometrics (`psychometrics`). Six agents (ProblemFormulator → DataEngineer →
Analyst → Critic → OutlineAgent → Writer) coordinated by a state-machine
orchestrator, plus an optional seventh (Verifier, `src/agents/verifier.py`) that
reads the finished manuscript. Given a dataset and a research prompt (or, for
every study type except prediction, a locked research spec passed with
`--research-spec`), it produces a complete draft LaTeX paper with real
citations. Everything it produces is an AI-generated draft; see
`DISCLAIMER.md` and `PRIVACY.md`.

**Verification.** After the Writer — and after the review gate, when that is
enabled — a `VERIFYING` stage holds the finished manuscript against the run's
own artifacts. The battery in `src/invariants.py` is deterministic and uses no
model; `src/obligations.py` tracks whether the Writer acted on the Critic's
instructions. The stage writes `invariants.json`, `obligations.json` and
`run_status.json`, and ends the run `COMPLETED` or `INCOMPLETE`. It is
advisory by default (`verification.blocking: false`): promote a check to
blocking only once its false-positive rate has been measured. One code is
promoted — `INV_LATEX_NO_PDF`, measured at zero false positives over 37
archived manuscripts and the four templates. It also fires when pdflatex never
ran (a manuscript with neither `paper.log` nor `paper.pdf`); that case was
added after the measurement and has not been re-measured on the archive.
Note that setting
`verification.blocking_codes` at all overrides `blocking`: only the listed
codes stop a run.

**Skill-based architecture.** Composable knowledge units (SKILL.md files) are
matched at runtime by `SkillRegistry` and injected into agent prompts via a
`{{SKILLS}}` placeholder. The six pipeline agents' prompts in
`agent_prompts/` (one per agent and study type) are slim and skill-injected;
the older monolithic prompts are kept beside them as
`agent_prompts/<agent>.v1.yaml.bak` for reference only. See "Skill-Based
Architecture" below.

## Status
Five study types, ten certified estimation methods (M1-M10) plus the
psychometric battery (P1-P7, estimated in R through `src/r_bridge.py`), four
curated datasets (hsls09_public, els_2002, did_els_hsls_panel,
assistments_0910), and an optional calibrated review gate (LSAR, a separate
public repository: https://github.com/cgpan/LSAR-public) with median sampling
for borderline scores. Regression discontinuity and instrumental variables are
certified on synthetic data but have no runnable dataset. The change log and
backlog are kept internally and are not part of this mirror.

## Authoritative Spec
@SPEC.md is the original design specification and the reference for schemas
and agent contracts. Its model IDs, provider and sandbox default are
historical: `config.yaml` and the code are current where they differ.

## Tech Stack
- Python 3.11 or 3.12 (not 3.13+: `dowhy<0.13` cannot be installed there)
- LLM providers behind `BaseAgent.call_llm()`: DeepSeek by default (OpenAI-compatible endpoint via the `openai` SDK), OpenAI or any OpenAI-compatible server, Anthropic, and MiniMax (legacy). Models per agent are set in `config.yaml`.
- pandas, scikit-learn, xgboost, shap, matplotlib, seaborn, dowhy
- PyYAML for registry parsing
- requests for Semantic Scholar, arXiv and Crossref
- LaTeX (pdflatex + bibtex for the ACM template; biber for the APA 7 journal template)
- R 4.4+ with jsonlite, lavaan, mirt, CDM (MASS ships with R) — psychometrics only
- Docker (optional, EXPERIMENTAL, off by default: `sandbox.enabled: false`); the default executor runs generated code as a local child process with API keys stripped from its environment
- No frameworks (custom orchestrator, no LangChain/LangGraph)

## Project Layout
- SPEC.md — original implementation spec (schemas, agent designs)
- DISCLAIMER.md, PRIVACY.md — user-facing disclaimer and data-handling notice; keep them true to the code
- config.yaml — central configuration (providers, model IDs, paths, pipeline params)
- .env.example — every environment variable the pipeline reads, with empty values
- requirements.txt (runtime), requirements-dev.txt (tests, lint, types), requirements-lsar.txt (review gate), requirements-sandbox.txt (Docker image)
- data/raw/ — dataset files (gitignored; see README "Data setup" for exact names)
- data_registry/datasets/ — YAML variable registries (Tier 1 curated, Tier 2 auto)
- data_registry/task_templates/ — task workflow definitions
- agent_prompts/ — YAML files with system prompts for each agent and study type
- templates/ — LaTeX templates: paper_template_v2.tex (ACM sigconf) and paper_template_journal.tex (APA 7)
- skills/ — skill library; one SKILL.md per skill; layers: task-type/, dataset/, methodology/, writing/
- src/ — all Python source code
- src/agents/ — one module per agent, all inherit from BaseAgent
- src/skills/ — skill registry infrastructure (schema, loader, matcher, composer, registry facade)
- r_helpers/ — certified R scripts the psychometric helpers run
- runs/fixtures/ — example locked research specs; runs/configs/ — configs of archived validation runs
- scripts/ — onboarding, synthetic-DGP gates, diagnostics (verify_skill_flow.py, audit_public_paths.py, ...)
- tests/ — pytest test suite
- output/ — pipeline run outputs (gitignored)

## Skill-Based Architecture

EDM-ARS uses a skill-based architecture for methodology, dataset-specific
knowledge, task-type workflows, and writing conventions. Skills live in
`skills/<layer>/<name>/SKILL.md` and are matched + composed at runtime by
`SkillRegistry` (`src/skills/`). Matched skill bodies are injected into the
agent's system prompt via a `{{SKILLS}}` placeholder.

### Layers
- **task-type/** — research procedure (prediction workflow, model batteries, evaluation, quality gate, critic checklist)
- **dataset/** — dataset-specific quirks (HSLS:09 NCES codes, variable registry, CSV format, school fingerprints)
- **methodology/** — crosscutting techniques (missingness protocol, SHAP, bootstrap CIs, subgroup analysis, inner-CV discipline)
- **writing/** — paper output (ACM template, style rules, BibTeX, limitations prose, UNVERIFIED flag)

### Severity tiers
- `mandatory` — violation produces invalid output (crash-risk, silent corruption, structural incompleteness, methodological invalidity). Renders with strong "MANDATORY RULE" banner; sorts first; bypasses per-layer cap.
- `recommended` (default) — violation produces worse output but output is structurally valid.
- `reference` — informational only.

See `skills/README.md` for the expanded mandatory criterion.

### Rules learned while slimming the prompts
Slimming the monolithic prompts exposed rules they had carried implicitly
(sample-size retention, qcut duplicates, the one-hot cardinality guard, and
others); each was harvested into a skill. When a prompt is slimmed, the
previous version is kept as `agent_prompts/<agent>.v1.yaml.bak` (tests check
that the backups exist). Output contracts must never live in
a skill that a per-layer cap can drop: tag such skills `mandatory`, and check
the rendered prompt through the orchestrator path (`match_and_compose` with
caps and context), never a bare `match()`.

### Adding a new skill
1. Create `skills/<layer>/<name>/SKILL.md` with required frontmatter (see `skills/README.md`).
2. Run `python scripts/verify_skill_flow.py` to confirm the skill reaches its declared stages.
3. If the skill's violation produces silent corruption / structural incompleteness, tag `rule_severity: mandatory`.
4. Run `pytest tests/`.

## Key Commands
- Install dev tools: `pip install -r requirements-dev.txt`
- Run tests: `python -m pytest tests/ -q` (offline; about 15 minutes)
- Lint: `ruff check src/ tests/` (known findings remain; not yet a gate)
- Type check: `mypy src/` (not yet clean)
- Public-mirror audit: `python scripts/audit_public_paths.py` (must report 0 findings)
- Pre-flight without spending: `python -m src.main --dry-run`
- Run pipeline: `python -m src.main --dataset hsls09_public`
- Locked spec: `python -m src.main --research-spec runs/fixtures/<spec>.json --output-dir output/<name>`
- Resume: `python -m src.main --output-dir output/<name> --resume`
- Build the experimental sandbox image: `docker build -t edm-ars-sandbox:latest .`

## Coding Rules
- Type hints on ALL functions and method signatures
- Agent system prompts live in agent_prompts/*.yaml, NEVER hardcoded in Python
- **Skill content lives in `skills/<layer>/<name>/SKILL.md`, not in agent prompts.** To add capabilities, add a skill — do not bloat agent prompts.
- All LLM calls go through BaseAgent.call_llm() — never call provider APIs directly
- All random operations use random_state=42
- Config values come from config.yaml via src/config.py — never hardcode model IDs
- A new config key is read with `.get()` and a default, and is added to config.yaml with a comment in the same change
- Each agent is a separate module in src/agents/. Do not merge agents.
- Log all pipeline events to output/{run_dir}/pipeline.log
- Follow the inter-agent message schemas defined in SPEC §6 exactly
- LLM-generated code must not make HTTP calls. Only the Docker sandbox enforces this (network_disabled: true); the default local executor does not, so it is a rule for prompts and skills, not a guarantee
- When requirements-sandbox.txt changes, rebuild the image: `docker build -t edm-ars-sandbox:latest .`
- Running generated code happens only in src/sandbox.py; subprocess calls stay in src/sandbox.py, apart from the existing R bridge (src/r_bridge.py) and LaTeX/review-gate paths — never add one in agent or base code
- The Writer fills templates/paper_template_v2.tex (conference) or templates/paper_template_journal.tex (journal, `writer.venue_format: journal`) — NEVER generates a LaTeX preamble from scratch
- Agents never modify paper authors. A conference paper prints the author block in templates/paper_template_v2.tex: EDM-ARS, plus commented-out AI and human author blocks a user may fill in. A journal paper's byline comes from config `paper.authors` (default EDM-ARS). The Writer only checks that EDM-ARS is still credited.
- When behaviour changes what leaves the user's computer or what runs on it, update PRIVACY.md / DISCLAIMER.md in the same change

## IMPORTANT
- NEVER put API keys in code, config.yaml, tests or docs. Keys come from environment variables or a gitignored .env (DEEPSEEK_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, MINIMAX_API_KEY, SEMANTIC_SCHOLAR_API_KEY).
- The Critic runs on the strongest model tier configured for the active provider (SPEC); with the default DeepSeek config every reasoning-heavy agent uses deepseek-v4-pro and only the outline and verifier stages use the cheap tier.
- Test set is ALWAYS 20% of analytic sample, stratified for classification.
- NEVER impute the outcome variable. Drop rows with missing outcomes.
- Public mirror: no absolute user paths, usernames or emails in any tracked file.

## Context Docs (read when relevant)
- @SPEC.md — full system spec with all schemas and agent designs
- @data_registry/datasets/hsls09_public.yaml — variable registry with domain knowledge
