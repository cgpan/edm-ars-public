# EDM-ARS: Educational Data Mining Automated Research System

## What
Multi-agent pipeline that automates prediction-focused EDM research.
Five agents (ProblemFormulator → DataEngineer → Analyst → Critic → Writer)
coordinated by a state-machine orchestrator. Given HSLS:09 data and a research
prompt, produces a complete LaTeX paper with real citations.

**V2.0 architecture** uses a skill-based system: composable knowledge units
(SKILL.md files) are matched at runtime by `SkillRegistry` and injected into
agent prompts via a `{{SKILLS}}` placeholder. See "V2.0 Skill-Based Architecture"
below. V2.0 ships partial: DataEngineer is slim and skill-injected (production
verified); the other four agents retain V1 monolithic prompts pending V2.1
work (see runbook below).

## V3 Status (tag v3.0.0, 2026-07-08)

Causal-inference capability phase COMPLETE: four task types (prediction,
causal_soo, causal_itr, causal_did), ten certified estimation methods
(M1-M10), three runnable datasets (hsls09_public, els_2002,
did_els_hsls_panel), calibrated LSAR gate with median sampling. All task
types have gate-passing papers. See `docs/v3_changelog.md` for the full
change log and `docs/backlog.md` for every deferred job. Next phase:
V4 psychometrics (plan in discussion).

## Authoritative Spec
@SPEC.md is the single source of truth. When in doubt, follow the SPEC.

## Tech Stack
- Python 3.11+
- Anthropic SDK (claude-sonnet-4-6 for most agents, claude-opus-4-6 for Critic)
- pandas, scikit-learn, xgboost, shap, matplotlib, seaborn
- PyYAML for registry parsing
- requests for Semantic Scholar API
- Docker (optional; sandboxed code execution for LLM-generated analysis code)
- docker Python SDK >= 7.0 (host-side; optional — falls back to subprocess if absent)
- No frameworks (custom orchestrator, no LangChain/LangGraph)

## Project Layout
- SPEC.md — definitive implementation spec (READ THIS FIRST)
- config.yaml — central configuration (model IDs, paths, pipeline params)
- data/raw/ — HSLS:09 CSV (gitignored, ~24K rows)
- data_registry/datasets/ — YAML variable registries (Tier 1 curated, Tier 2 auto)
- data_registry/task_templates/ — task workflow definitions
- agent_prompts/ — YAML files with system prompts for each agent
- templates/ — LaTeX paper template (V2 bundled in skills/writing/acm-acmart-sigconf-template/)
- skills/ — V2.0 skill library; one SKILL.md per skill; layers: task-type/, dataset/, methodology/, writing/
- src/ — all Python source code
- src/agents/ — one module per agent, all inherit from BaseAgent
- src/skills/ — skill registry infrastructure (schema, loader, matcher, composer, registry facade)
- tests/ — pytest test suite
- regression/ — Phase 2c regression artifacts (slim drafts in proposed_slim/, run captures by phase)
- audit/ — Phase 0 + Phase 2c audit documents
- scripts/ — diagnostics (verify_skill_flow.py, fingerprint_run.py, sanity checks)
- output/ — pipeline run outputs (gitignored)

## V2.0 Skill-Based Architecture

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

### Adoption status (V2.0 ships partial)
- **DataEngineer**: slim (130 lines, was 363) — production verified on OpenAI gpt-5.4 + MiniMax-M2.7
- **ProblemFormulator, Analyst, Critic, Writer, OutlineAgent**: V1 monolithic prompts retained pending V2.1 slim work
- **Reason**: V2.0 slim cascade exposed multiple latent rules that V1 monolithic prompts contained implicitly. Retention rule for sample-size, qcut duplicates, one-hot cardinality guard, etc. — all surfaced and fixed during Phase 2c. Additional rules remain to be discovered via continued slim attempts. Rather than block shipping, V2.0 ships with one slim verified and the rest deferred.

### Adding a new skill
1. Create `skills/<layer>/<name>/SKILL.md` with required frontmatter (see `skills/README.md`).
2. Run `python scripts/verify_skill_flow.py` to confirm the skill reaches its declared stages.
3. If the skill's violation produces silent corruption / structural incompleteness, tag `rule_severity: mandatory`.
4. Run `pytest tests/`.

### V2.1 slim runbook
Future work to slim the four remaining agent prompts:
1. Apply the slim draft from `regression/proposed_slim/<agent>.yaml` to `agent_prompts/<agent>.yaml` (back up the v1 to `agent_prompts/<agent>.v1.yaml.bak`).
2. Run regression on OpenAI gpt-5.4 (or stronger) — `regression/regression_config_openai.yaml`.
3. If pipeline fails on a rule violation: harvest the rule from the V1 monolithic prompt into the relevant skill (mandatory if silent corruption / structural).
4. Tag mandatory; re-verify with `scripts/verify_skill_flow.py`.
5. Re-run regression. Continue until clean.
6. Repeat for the next agent. Recommended order matches dependency: ProblemFormulator → Analyst → Critic → Writer + OutlineAgent.

## Key Commands
- Run tests: `pytest tests/ -v`
- Lint: `ruff check src/ tests/`
- Type check: `mypy src/`
- Run pipeline: `python -m src.main --dataset hsls09_public`
- Build sandbox image: `docker build -t edm-ars-sandbox:latest .`
- Build via Compose: `docker compose build sandbox`
- Run pipeline without sandbox (subprocess fallback): set `sandbox.enabled: false` in config.yaml

## Coding Rules
- Type hints on ALL functions and method signatures
- Agent system prompts live in agent_prompts/*.yaml, NEVER hardcoded in Python
- **Skill content lives in `skills/<layer>/<name>/SKILL.md`, not in agent prompts.** To add capabilities, add a skill — do not bloat agent prompts.
- All LLM calls go through BaseAgent.call_llm() — never call Anthropic / OpenAI APIs directly
- All random operations use random_state=42
- Config values come from config.yaml via src/config.py — never hardcode model IDs
- Each agent is a separate module in src/agents/. Do not merge agents.
- Log all pipeline events to output/{run_dir}/pipeline.log
- Follow the inter-agent message schemas defined in SPEC §6 exactly
- Sandbox has NO network access (network_disabled: true); LLM-generated code must not make HTTP calls
- When requirements-sandbox.txt changes, rebuild the image: `docker build -t edm-ars-sandbox:latest .`
- subprocess.run() must ONLY appear in src/sandbox.py (SubprocessExecutor); never in agent or base code
- Writer agent uses templates/paper_template.tex — NEVER generates LaTeX preamble from scratch
- Paper authors are fixed (EDM-ARS, Claude AI, Chenguang Pan) — never modified by agents

## IMPORTANT
- NEVER put API keys in code. Use ANTHROPIC_API_KEY env variable.
- Critic agent MUST use opus model. All others use sonnet.
- Test set is ALWAYS 20% of analytic sample, stratified for classification.
- NEVER impute the outcome variable. Drop rows with missing outcomes.

## Context Docs (read when relevant)
- @SPEC.md — full system spec with all schemas and agent designs
- @data_registry/datasets/hsls09_public.yaml — variable registry with domain knowledge