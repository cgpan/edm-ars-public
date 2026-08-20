# EDM-ARS Change Memo

A chronological record of major changes to EDM-ARS. Most recent entries at the top.
Format per entry: "## <Version/Phase>: <Title> — <YYYY-MM-DD>" followed by Goal and Key Changes.

---

## V2.0.1: Wire `{{SKILLS}}` into V1 Monolithic Prompts — 2026-04-26

**Goal:** Make matched skills actually reach the four V1 monolithic agents (PF, Analyst, Critic, Writer) without doing full slim-role migration. Patch the YAMLs only; reuse all V2.0 substitution infrastructure.

**Adapted scope (vs original V2.0.1 spec):** The substitution chokepoint (`BaseAgent.render_system_prompt`), per-stage match invocation (`Orchestrator._inject_skills`), and severity-aware composition (`format_skills_for_prompt`) all already exist from V2.0 Phase 2c recovery. The original spec proposed a parallel `render_agent_prompt` orchestrator function and topological composition order; both would have duplicated existing infra. After review, executed Option A: YAML edits only + 2 new tests, leveraging existing infra.

**Changes:**
- `agent_prompts/{problem_formulator,analyst,critic,writer}.yaml` — added a single `# Binding Rules` block containing `{{SKILLS}}`, positioned after each agent's role/persona, before `## Your Outputs` / `## Structured Reasoning Protocol`. Total: ~10 lines added per file.
- `tests/test_v2_0_1_skill_injection.py` — 6 tests: 4 per-V1-agent integration (real registry, real skills, asserts placeholder substituted + binding-rules section present + matched skills in output) + 1 stage-mapping regression + 1 lock-in test preventing future addition of a duplicate substitution path in the orchestrator.

**Adoption status:** All V2.0+V2.0.1 tests pass (48/48 across `test_skill_registry.py`, `test_orchestrator_skill_wiring.py`, `test_data_engineer_helpers.py`, `test_v2_0_1_skill_injection.py`).

**Per-agent skill match inventory (production registry, baseline context):**

| V1 Agent | Stage | Skills matched | Mandatory injections |
|---|---|---|---|
| problem_formulator | ProblemFormulator | 6 | 2 (`prediction-research-question-design`, `hsls09-tier3-exclusions`) |
| analyst | Analyst | 18 | 5 (`prediction-quality-gate`, `smote-imbalance-handling`, `subgroup-fairness-analysis`, `inner-cv-tuning-discipline`, `sensitivity-analysis-high-missingness`) |
| critic | Critic | 10 | 4 (`prediction-quality-gate`, `subgroup-fairness-analysis`, `inner-cv-tuning-discipline`, `missingness-tiered-protocol`) |
| writer | Writer | 10 | 4 (`sensitivity-analysis-high-missingness`, `latex-table-discipline`, `unverified-flag-and-appendix`, `paper-writing-style-rules`) |

**Empty-match inventory: 0 agents.** Every V1 agent matched ≥6 skills. Permanent inventory captured at `regression/v2_0_1_verify/skill_match_inventory.txt`.

**End-to-end regression evidence (`regression/v2_0_1_verify/`):**
- Pipeline reached COMPLETED, Critic verdict PASS, quality score 7
- 6 models trained (LR, RF, XGBoost, ElasticNet, MLP, StackingEnsemble); paper.pdf compiled cleanly
- **`results.subgroup_performance`: `["X1RACE"]`** — populated, in contrast to V2.0 final which had `[]`. Direct evidence that the now-mandatory `subgroup-fairness-analysis` skill is reaching the V1 monolithic Analyst and being honored.
- One Critic-issued REVISE → revision cycle → second Critic verdict PASS (cost ~$7.65 due to revision; expected for first run with new prompts under MiniMax)

**Confirmed working under V2.0.1:** mandatory rules now reach all 5 stages, not just DE. The DataEngineer slim from V2.0 + V1 monolithic prompts with `{{SKILLS}}` placeholder is a viable production configuration.

**Deferred (still V2.1):** full slim-role migration of the V1 YAMLs (~200-600 lines per agent → ~75-130 lines). The slim drafts in `regression/proposed_slim/` remain ready; V2.1 work resumes there when the rule-recovery loop finishes harvesting lost-from-V1 rules.

---

## V2.0 Phase 2c: Ships Partial (Option A) — 2026-04-25

**Goal:** Slim agent prompts and wire `SkillRegistry` into the orchestrator. Ships with substantial-but-partial completion.

**Shipped under V2.0:**
- Skill registry infrastructure: schema, loader, matcher (with stemming + mandatory cap-bypass), composer (with cycle detection), registry facade
- 41 skills extracted across 4 layers (task-type=14, dataset=7, methodology=11, writing=9)
- 14 skills tagged `mandatory` under expanded criterion (crash-risk + silent corruption + structural incompleteness + methodological invalidity)
- DataEngineer slim (363 → 130 lines, -64%), production verified on OpenAI gpt-5.4 + MiniMax-M2.7
- Severity-aware skill rendering (mandatory rules render with binding banner, sort first, bypass per-layer cap)
- `scripts/verify_skill_flow.py` permanent diagnostic — confirms 0 mandatory-skill misfires under shipped configuration
- `analysis_helpers` symmetry fix between Analyst and DataEngineer (pre-existing latent bug surfaced + fixed during refactor)
- `_HELPERS_SRC` path correctness in Analyst (pre-existing latent bug — pointed at `src/agents/analysis_helpers.py` for months — surfaced + fixed during refactor)
- Algorithmic guards: zero-cluster school reconstruction, qcut duplicates, one-hot cardinality, retention floor (existing)
- OpenAI provider support added to `BaseAgent` (alongside existing Anthropic + MiniMax providers)
- `regression/v2_0_final/` — final regression on shipped configuration: COMPLETED, Critic verdict PASS, paper.pdf compiled, RMSE = 6.85

**Deferred to V2.1:**
- Slim ProblemFormulator, Analyst, Critic, Writer, OutlineAgent
- Slim drafts retained in `regression/proposed_slim/` (5 files)
- Reason: 5 regression cycles each surfaced new latent rules from V1 monolithic prompts. Adversarial audit (22 gaps in 30 min) was 100x more efficient than regression but still missed the rule that blocked T1.3 (predictor-set retention enforcement). Rather than continue the rule-recovery loop indefinitely, ship infrastructure + verified DE slim now; complete remaining slims as part of Phase 3+ work where additional rules can be harvested incrementally.

**Recovery sub-history (latent defects surfaced and fixed during Phase 2c):**
1. Checkpoint 3: rendering severity gap → `rule_severity` system + mandatory tagging convention
2. Checkpoint 4b: `analysis_helpers` not copied to DE execution dir + `_HELPERS_SRC` orchestrator bugs (pre-existing, masked by LLM retry inlining)
3. R2.3: redundant composites + qcut duplicates + zero-cluster guard (3 algorithmic guards)
4. R3.5: one-hot cardinality + subgroup analysis enforcement (2 mandatory rule additions)
5. T1.3: predictor-retention sample-size rule (deferred to V2.1 slim work; existing skill text covers it but not as a `## Mandatory:` callout)

**Lesson for Phase 3+:** The slim architecture exposes latent defects that monolithic prompts hide. Prefer adversarial audit (no API spend) over regression-driven discovery ($5+/cycle) when looking for gaps. The skill-flow verification script (`scripts/verify_skill_flow.py`) should be run whenever new skills are added to confirm they reach their declared stages.

**V2.1 runbook:** documented in `CLAUDE.md` under "V2.0 Skill-Based Architecture / V2.1 slim runbook".

**Phase 3 readiness:** infrastructure is ready for the causal-inference task type. New skills can be added under `skills/task-type/causal-*` and `skills/methodology/<causal-*>` without further refactor. V2.1 slim work can proceed in parallel with Phase 3 as latent rules are harvested from regression cycles on causal-inference research questions.

---

## V2.0 Phase 2b: Task-Type + Dataset Skill Extraction — 2026-04-17

**Goal:** Extract the 15 CSV-listed task-type + dataset skills, plus Decision 6 + 9 expansions (22 total SKILL.md files in this phase). No agent, orchestrator, or source YAML changes.

**Scope:**
- 7 dataset skills → `skills/dataset/` (including `hsls09-variable-registry` with bundled 1,800-line YAML per D5)
- 14 task-type files → `skills/task-type/` (8 base entries from CSV; `prediction-model-battery` becomes 1 meta + 6 per-family skills per D9, net +6)
- 1 new methodology skill from D6 split → `skills/methodology/cluster-id-reconstruction-from-fingerprints/`

**Decision 5 applied:** `dataset/hsls09-variable-registry/` bundles `variable_registry.yaml` (62,342 bytes — byte-identical copy of `data_registry/datasets/hsls09_public.yaml`). The SKILL.md is a short (~120 line) index explaining how to load and what's in the YAML.

**Decision 6 applied:** School-cluster reconstruction split into:
- `methodology/cluster-id-reconstruction-from-fingerprints` — generic recipe (variance check, joint groupby, quality diagnostics; no HSLS variable names in the body)
- `dataset/hsls09-school-fingerprints` — HSLS-specific (the seven fingerprint vars `X1SCHOOLCLI`, `X1COUPER*`, `X1CONTROL`, `X1LOCALE`, `X1REGION`; expected 944 schools); `references_skills: [cluster-id-reconstruction-from-fingerprints]`

**Decision 9 applied:** `prediction-model-battery` is a meta-skill that composes six per-family skills via `references_skills`:
- `model-logistic-regression` (LinearExplainer, baseline, no tuning)
- `model-random-forest` (TreeExplainer, GridSearchCV)
- `model-xgboost` (TreeExplainer, GridSearchCV; LightGBM as alternative)
- `model-elasticnet` (LinearExplainer, GridSearchCV; SGDClassifier for classification)
- `model-mlp` (KernelExplainer with sample_cap=1000; configurable via `mlp_enabled`)
- `model-stacking-ensemble` (no SHAP per SPEC; meta-learner self-tunes)

**Sanity check:** `scripts/phase2b_sanity_check.py` confirms (all 6 assertions PASS):
1. 41 skills loaded; 14/7/11/9 by layer
2. `hsls09-variable-registry` resource resolves to a 62,342-byte file
3. Analyst + model-battery query pulls in `prediction-model-battery` and all six per-family skills via composition (no duplicates)
4. DataEngineer + HSLS query pulls in `cluster-id-reconstruction-from-fingerprints` via `references_skills` from `hsls09-school-fingerprints`
5. Causal-inference query correctly excludes prediction task-type skills (task_type filter works); methodology skills still flow through
6. Non-HSLS dataset query (`els2002`) correctly excludes all 9 HSLS-coupled skills (7 dataset + 2 HSLS writing); cross-cutting task-type and writing skills still flow through

**Ambiguities flagged during extraction:**
- The Phase 2b prompt's "Four commits" constraint conflicts with the five commit blocks in tasks 2b.1–2b.5. Took the task-block count as authoritative and pushed five commits (plan, dataset, task-type, sanity, memo). Confirm intent for next phase.
- `prediction-critic-checklist` body is ~250 lines — within the "if over 300 split" guidance but borderline. Per Decision 2 (Phase 0) it stays as one skill. Worth revisiting if it grows.
- The CSV row `hsls09-school-cluster-reconstruction` was renamed to `hsls09-school-fingerprints` per the D6 plan (the new name reflects what stays in the dataset skill — the variable list — rather than the technique). The CSV is unchanged but the new name is documented in `scripts/phase2b_extraction_plan.md`.
- `prediction-evaluation-classification` body briefly references `X1TXMTSC` in an example PDP filename. This is a generic example placeholder, not a hard dependency on the HSLS dataset. Left as-is for readability.

**Deferred to Phase 2c:**
- Slim agent YAMLs to core-role only (~380 lines total across 5 agents)
- Wire `SkillRegistry` into `Orchestrator.run()` to inject matched skills into each agent call
- Regression test: re-run existing HSLS:09 prediction pipeline; compare outputs to a Phase 1 baseline
- Delete duplicated content from source YAMLs (now in `## Source provenance` sections of extracted skills)
- Delete `templates/paper_template.tex` v1 (D1)
- Add plural-aware stemming to matcher `_tokenize` to fix the `table`/`tables` issue from Phase 2a

---

## V2.0 Phase 2a: Methodology + Writing Skill Extraction — 2026-04-17

**Goal:** Extract the least-coupled skills (methodology + writing) into real SKILL.md files under `skills/`. No agent prompts, orchestrator, or source YAMLs modified.

**Scope:**
- 10 methodology skills → `skills/methodology/`
- 9 writing skills → `skills/writing/` (CSV had 9 in this layer, not 8 as the Phase 2a prompt assumed; CSV is source of truth so all 9 extracted)
- ACM template moved to bundled resource at `skills/writing/acm-acmart-sigconf-template/paper_template_v2.tex`

**Triplication resolved:**
- `missingness-tiered-protocol` — canonical: `agent_prompts/data_engineer.yaml` §"Missing Data Protocol" (most complete prose; merged from `prediction.yaml` and `methodological_checklist.yaml`)
- `shap-explainer-selection` — canonical: `agent_prompts/analyst.yaml` §"SHAP Interpretability Protocol" (full helper-function guidance and KernelExplainer constraints; merged from `prediction.yaml` shap_protocol and `methodological_checklist.yaml` an_05/an_08/an_09/an_10)
- `inner-cv-tuning-discipline` — canonical: `agent_prompts/analyst.yaml` §"Hyperparameter Tuning" (group-aware CV code path; merged from `prediction.yaml` mt_02-mt_06 and `methodological_checklist.yaml` an_02)

Each SKILL.md has a `## Source provenance` section. Originals remain in place; Phase 2c will delete duplicated content.

**Sanity check:** `scripts/phase2a_sanity_check.py` output confirms:
- 19 skills loaded
- Analyst+SHAP+fairness query matches `subgroup-fairness-analysis`, `shap-explainer-selection`, `bootstrap-confidence-intervals`, `inner-cv-tuning-discipline`
- Writer+ACM query matches `acm-acmart-sigconf-template`, `latex-figure-discipline`, `bibtex-from-literature-context`, `hsls09-multilevel-limitations-paragraph`, `hsls09-survey-weights-limitations-paragraph`

**Deferred to Phase 2b:** 8 task-type skills, 7 dataset skills, including the HSLS registry as bundled resource and the school-cluster split.
**Deferred to Phase 2c:** agent YAML slimming, orchestrator wiring, regression test, deletion of duplicated content and v1 LaTeX template.

**Ambiguities flagged during extraction:**
- CSV has 9 writing skills; Phase 2a prompt expected 8. The extra is `paper-writing-style-rules` (extracted; `applicable_stages: [Writer]`).
- Two writing skills are HSLS-coupled (`hsls09-multilevel-limitations-paragraph`, `hsls09-survey-weights-limitations-paragraph`). They live in `skills/writing/` per the CSV but declare `applicable_datasets: [hsls09_public]`. Confirm this is the intended layer (vs. moving to `skills/dataset/` in Phase 2b).
- `subgroup-fairness-analysis` source contains HSLS variable name examples (X1SEX, X1RACE). I generalized them to placeholders in the SKILL.md body but kept the canonical source pointer; the dataset-layer skill will supply concrete attribute names.
- `latex-table-discipline` did not appear in the Writer sanity-check match because its keyword `table` (singular) does not tokenize to match "tables" (plural) in the query context. The matcher does no stemming. Worth tuning trigger_keywords in Phase 2c if this pattern recurs.

---

## V2.0 Phase 1: Skill Registry Infrastructure — 2026-04-17

**Goal:** Build registry infrastructure for the V2.0 skill-based architecture. No agent or orchestrator changes; no real skill content extracted.

**Decisions locked in (from Phase 0 review):**
- D5: HSLS registry uses resource-file pattern (SKILL.md + bundled YAML)
- D6: School-cluster reconstruction split into methodology + dataset skills with cross-reference
- D9: Model battery is per-family; composition via `references_skills`

**Key changes:**
- `src/skills/` — `Skill` dataclass, loader, matcher, composer, registry facade, package `__init__`
- Schema includes `references_skills` (Decisions 6, 9) and `resources` (Decision 5); `layer` is required
- Matcher supports stage + task-type + dataset hard filters with per-layer top-k caps (defaults: task-type=2, dataset=2, methodology=4, writing=3)
- Composer resolves `references_skills` transitively with cycle detection (logs warning, breaks cycle, never crashes); missing references log warning, do not crash
- `tests/fixtures/skills/` — 6 valid + 1 broken fixture skill (incl. one bundled YAML resource)
- `tests/test_skill_registry.py` — 27 tests, all passing
- `skills/` — empty directory skeleton with layer READMEs (real skills arrive in Phase 2)

**Deferred to Phase 2:** D1 (delete v1 LaTeX template), D2, D3, D4, D7, D8, D10 — see `audit/AUDIT_REPORT.md` §7.

---

## V2.0 Phase 0: Skill Extraction Audit — 2026-04-17

**Goal:** Inventory existing prompts to identify candidate skills for the V2.0 skill-based architecture.

**Key changes:**
- Added `files/` to `.gitignore`
- Created `audit/` directory (to be populated during this phase)

**Results:**
- Total candidate skills: **39**
- Breakdown by layer: task-type=8, dataset=7, methodology=10, writing=8, core-role=6
- Top 3 decisions flagged for the project lead:
  1. Canonical LaTeX template — keep `paper_template_v2.tex` (outline-first single body) and delete v1?
  2. Critic checklist — keep `prediction-critic-checklist` as one ~180-line skill, or split into 4 sub-skills (PF / DP / Analysis / Substantive)?
  3. SMOTE classification — `smote-imbalance-handling` stays as `task-type`, or promote to `methodology` for future fairness/causal reuse?
- Artifacts: `audit/skill_candidates.csv`, `audit/AUDIT_REPORT.md`

---
