# V2.1 Slim Migration Specification

> **Status:** Authoritative design document for V2.1 implementation phases (3b.21+). This document supersedes the V2.1 slim runbook note in `CLAUDE.md` wherever the two diverge — it incorporates the V3.0 causal cleanup arc's findings (Phases 3b.4 through 3b.19) and the discovery that V2.0.1 already authored slim drafts for every agent.

---

## §1. Executive Summary

### 1.1 The migration goal

Bring all six agents (DataEngineer, ProblemFormulator, Analyst, Critic, Writer, OutlineAgent) to the same slim shape DataEngineer reached in V2.0.1: a thin prompt that contains role + output contract + critical universal rules + a `{{SKILLS}}` placeholder, with all methodology, dataset-specific, and task-type-specific guidance delegated to skills selected by the matcher.

DataEngineer reached this shape in V2.0.1 (commit `3ba93a6`). The other five agents remain on V1 monolithic prompts — ProblemFormulator (207 lines), Analyst (605), Critic (235), Writer (485), OutlineAgent (93) — with methodology and task-type-specific rules baked into the prompt body. V2.1 applies the V2.0.1-staged slim drafts (in `regression/proposed_slim/`) to those five agents.

### 1.2 The shape of V2.1 — applying staged drafts, not designing them

The most important finding from Sub-wave 0 investigation: **V2.0.1 already authored complete slim drafts for every agent.** They live in `regression/proposed_slim/`:

| Agent | V1 current | V2.0.1 slim draft (staged) | Reduction |
|---|---:|---:|---:|
| DataEngineer | 130 | 130 (already shipped) | 0% (baseline) |
| ProblemFormulator | 207 | 95 | −54% |
| Analyst | 605 | 115 | −81% |
| Critic | 235 | 123 | −48% |
| Writer | 485 | 77 | −84% |
| OutlineAgent | 93 | 25 | −73% |

V2.0.1 did the slim-design work. V2.0.1's Phase 2c surfaced multiple rules that the V1 prompts contained implicitly (qcut duplicates, one-hot cardinality guard, sample-size retention, no-redundant-composites — see commits `092c7b3`, `5cda588`, `c52370c`). Those rules were harvested into skills then. The drafts in `regression/proposed_slim/` reflect that work and are ready to apply.

**V2.1 is not a design phase. V2.1 is an application + verification phase.** Each migration phase:
1. Applies the existing V2.0.1 draft to the live `agent_prompts/<agent>.yaml`.
2. Backs up V1 to `agent_prompts/<agent>.v1.yaml.bak`.
3. Verifies the slim agent renders correctly (rendered-prompt tests + content-presence).
4. If runtime exercises surface rules that were in V1 but not in any skill, harvests those rules into appropriate skills as separate single-issue phases.
5. Re-runs the locked-spec smoke test in interleaved verification phases (3b.X.5 cadence).

### 1.3 Why now

The V3.0 causal cleanup arc (Phases 3b.4 → 3b.19) closed in 3b.19 with Outcome B (CLOSURE-B). The four targeted amendments (3b.12 DE-contract, 3b.14 G5 DoWhy, 3b.16 NetworkX, 3b.18 D1 encoding) validated end-to-end. Remaining LSAR-named gaps (M5 inference, M4 SE, subgroup analysis, pre-critic prediction-field carryover) are tracked as the V2.1 implementation backlog, not blockers.

The architectural debt of V1 monolithic prompts is now the dominant remaining structural concern. Per the V3.0 audit (`docs/v3_0_causal_skill_specification.md` §10.5 and the various phase amendments), every V3.0 causal skill was authored against the assumption that V2.1 slim migration would eventually apply. The cleanup arc's pattern of "harvest rule from V1 prompt into mandatory skill, render-prompt-verify, smoke-test re-run" is the exact pattern V2.1 implementation phases will follow.

### 1.4 High-level migration sequence

| Phase | Action | Wall-clock | Verification |
|---|---|---|---|
| 3b.21 | OutlineAgent slim apply (smallest test of pattern) | Pattern A | Rendered-prompt + skill-loadability |
| 3b.21.5 | Smoke-test re-run (verify pipeline still completes) | Pattern B re-run | Q2.7.a–d differential vs 3b.19 |
| 3b.22 | ProblemFormulator slim apply | Pattern A | Same + JSON-schema regression |
| 3b.23 | Critic slim apply | Pattern A | Same + verdict-evaluator integration test |
| 3b.23.5 | Smoke-test re-run (mid-arc check) | Pattern B re-run | Cumulative Δ vs 3b.21.5 |
| 3b.24 | Writer slim apply | Pattern A | Same + LaTeX-compile regression |
| 3b.25 | Analyst slim apply (largest, last) | Pattern A | Same + 10-mandatory-skill budget check |
| 3b.25.5 | Final smoke-test re-run | Pattern B re-run | Cumulative Δ vs 3b.19; LSAR Accept target |

**Total estimated phase count: 5 migration + 3 re-runs = 8 phases.**

Each migration phase is single-issue Pattern A — offline verification, no LLM calls. Each re-run phase is a controlled single-variable Δ vs the prior re-run, following the 3b.13 / 3b.15 / 3b.17 / 3b.19 controlled-comparison pattern that produced clean attribution across four phases.

### 1.5 Top 3 takeaways

1. **The drafts exist; the design work was V2.0.1.** V2.1 is the deployment arc. This significantly de-risks each migration phase: the slim form has been thought through. The remaining risks are runtime-discovered rule gaps (handled by harvest-into-skill single-issue phases) and downstream-agent compatibility (handled by rendered-prompt + schema regression tests).

2. **Task-type branching is RETAINED, not eliminated.** The hand-off recommended elimination; Sub-wave 0 investigation found this infeasible — the `_causal_soo.yaml` variants for PF / Analyst / Writer have structurally-different output schemas from their base prompts (causal `estimand`/`estimates` vs prediction `all_models`/`top_features`). The variants are already slim (Analyst 97 lines, PF 113, Writer 83); the V2.1 migration brings the BASE prompts to slim shape; both shapes coexist. See §2.4 for the design rationale and §8 for the open question about future schema unification.

3. **OutlineAgent is the natural first target.** It is the smallest agent (93 lines V1 → 25 lines slim), has no `_causal_soo` variant (so no task-type branching to worry about), and is the only agent whose V1 prompt LACKS the `{{SKILLS}}` placeholder (3b.12 confirmed DE/PF/Analyst/Critic/Writer all have it). 3b.21's OutlineAgent migration is therefore the cleanest possible first test of the slim-application pattern.

---

## §2. Migration target shape

The V2.0.1 DataEngineer prompt (`agent_prompts/data_engineer.yaml`, 130 lines) is the reference shape. Every other agent's slim form follows the same structure with agent-specific content in the variable slots.

### 2.1 Required structural elements

Every slim agent prompt MUST contain these sections in this order:

1. **YAML metadata** (4 lines): `agent_name`, `model_config_key`, `temperature`, `max_tokens`.
2. **Role/persona block** (1–2 sentences): "You are the X agent for EDM-ARS." A short statement of identity. No methodology, no task-type-specific guidance.
3. **`# Binding Rules` section header + `{{SKILLS}}` placeholder.** The `Binding Rules` heading frames the skill block; the placeholder is where the matcher injects skills at runtime.
4. **`# Your Role and Contract` section.** A few sentences of what the agent produces.
5. **Output schema** (JSON or structured format expectations). For agents producing structured JSON, the schema is given verbatim.
6. **Execution Environment notes** (if applicable — DE, Analyst, Critic agents that execute code).
7. **Critical Universal Rules** (numbered list, ≤8 items). Cross-task constants that hold for every agent run (e.g., "Never impute the outcome variable", "Test set ≥ 20%", "Use random_state=42"). These are NOT methodology — they are universal contracts.
8. **Input You Will Receive** (paragraph or bullet list naming the user-message components).
9. **Output Format** (1–2 lines specifying delimiters, e.g., "Wrap your JSON in a ```json block. No prose before or after.").

The DataEngineer prompt has all 9 of these. Each migration phase verifies the same structure exists in the slim agent prompt.

### 2.2 What gets removed from V1 prompts

The V1 → slim diff is dominated by removing content that should be in skills:

- **Methodology-specific guidance.** Model batteries, hyperparameter grids, evaluation protocols, SHAP explainer dispatch, sensitivity-analysis recipes, balance-diagnostic tables. These are covered by the V3.0 causal G1–G5 + M1–M5 skills and the V2.0 prediction-task skills.
- **Dataset-specific conventions.** HSLS:09 variable codes, temporal-ordering rules, NCES missing-code handling, school-cluster fingerprints, registry-type encoding dispatch. Covered by D1 (`hsls09-causal-conventions`) and the V2.0 dataset skills (`hsls09-temporal-ordering`, `hsls09-tier3-exclusions`, etc.).
- **Task-type-specific procedures.** Prediction-task SMOTE + ablation, classification confusion matrices, causal estimand declaration, DoWhy refuter invocation. Covered by task-type-tagged skills (the V3.0 12 causal skills + V2.0 prediction skills).
- **Failure-mode handling.** "If KernelExplainer times out, fall back to next-best model" → covered by `shap-explainer-selection` skill. "If qcut produces duplicate bins, use rank-based binning" → covered by V2.0 skills. "DOT graph format requires pygraphviz" → 3b.16 G5 amendment.

### 2.3 What stays in V1 prompts

After methodology / dataset-specific / task-type-specific / failure-mode content moves to skills, the remaining V1 content forms the slim body:

- **Agent identity and role definition** (the 1–2 sentence role block).
- **Output JSON schema or format expectations.** Schemas are agent-specific contracts that downstream agents (and the orchestrator's verdict-evaluator) depend on; they must be in the prompt so the LLM knows the precise field layout. Skills can govern WHICH fields are populated but the schema declaration sits in the prompt.
- **Execution Environment details** for code-executing agents (DE, Analyst). Sandbox-vs-subprocess routing, `RAW_DATA_PATH` env var, relative-path conventions — these are infrastructure constants tied to the orchestrator, not methodology.
- **Critical Universal Rules** (≤8 numbered items). The DE has 6: outcome-not-imputed, outcome-not-in-X, scaler-fit-on-train, test_size≥0.2, random_state=42, split-before-encode. These are universal contracts that don't fit cleanly in any skill (they are cross-task and cross-dataset).
- **Input You Will Receive** (naming the user-message components — research_spec.json, data_report.json, registry YAML, etc.).
- **Output Format** (markdown-fence delimiters, prose-or-not rules).

### 2.4 Task-type branching post-migration — RETAINED, not eliminated

The hand-off recommended eliminating per-task-type prompt files in favor of a single slim prompt per agent + task-type-aware skill matching. Sub-wave 0 investigation found this **infeasible** for the current `_causal_soo` variants.

**Evidence:**

The `_causal_soo.yaml` variants for ProblemFormulator (113 lines), Analyst (97 lines), and Writer (83 lines) have output schemas that are **structurally different** from their base prompts:

| Agent | Prediction (base) output schema | Causal (`_causal_soo`) output schema |
|---|---|---|
| ProblemFormulator | `research_question`, `outcome_variable`, `predictor_set[]`, `novelty_score_self_assessment`, ... | `task_type=causal_soo`, `treatment{}`, `outcome{}`, `estimand`, `target_population`, `adjustment_set[]`, `primary_method`, `secondary_methods[]`, `methodological_concerns[]`, ... |
| Analyst | `best_model`, `best_metric_value`, `primary_metric=AUC\|RMSE`, `all_models{}`, `top_features[]`, `subgroup_performance{}`, ... | `estimand=ATE\|ATT\|...`, `estimates{<method>: {point_estimate, se, ci_lower, ci_upper, ...}}`, `balance_diagnostics{}`, `positivity_diagnostics{}`, `sensitivity{dowhy_refuters{}, e_value_point, ...}`, ... |
| Writer | Prediction paper structure (Results = model comparison, SHAP, subgroup) | Causal paper structure (Results = ATE point estimate, balance diagnostics, positivity, sensitivity) |

The output schemas are not field-renamings of a common base — they are different research products. The prediction Analyst produces a model-comparison report; the causal Analyst produces a causal-effect-estimation report. A single slim prompt would have to express both schemas as alternatives ("if task_type=causal_soo, use schema A; else use schema B"), which reintroduces task-type branching at the prompt level rather than eliminating it. The complexity moves rather than diminishes.

**Decision: V2.1 retains per-task-type prompt variants.** Each agent has up to two slim prompts: the base prompt (prediction default) and a `_causal_soo.yaml` variant when needed. The orchestrator's `load_prompt(agent, task_type=...)` already routes correctly. Skill matching layers on top of whichever prompt is selected to provide task-type-aware skill bodies.

**The `_causal_soo` variants are ALREADY SLIM** (Analyst 97 lines, PF 113, Writer 83) — they were authored in Phase 3b.4 as slim replacements for the corresponding sections of the V1 monolithic base prompts. V2.1's migration scope is the **base (prediction) prompts only**. The causal variants need only a verification pass.

**Schema unification is deferred to a future V2.2+ effort.** See §8 open question 1.

### 2.5 Line-count and budget targets

Each slim prompt's target size is taken directly from `regression/proposed_slim/`:

| Agent | V1 | Slim target | Slim `_causal_soo` (already exists) |
|---|---:|---:|---:|
| DataEngineer | 130 | 130 (baseline; no change) | n/a (DE is task-type-agnostic at prompt level) |
| ProblemFormulator | 207 | 95 | 113 |
| Analyst | 605 | 115 | 97 |
| Critic | 235 | 123 | (no variant) |
| Writer | 485 | 77 | 83 |
| OutlineAgent | 93 | 25 | (no variant) |

**Total V1 lines:** 1,755 across the five non-DE agents.
**Total slim lines:** 435 across the same five.
**Reduction:** −1,320 lines (−75%).

These targets are not arbitrary — they are the V2.0.1 draft sizes. Implementation phases verify the applied prompt matches the staged draft and that LSAR-relevant content has been harvested into skills before deletion.

Skill-budget impact: with V1 prompts cut by 75%, the rendered-prompt size at runtime drops correspondingly. The 3b.8 formatter's `max_chars=30000` cap was set for the V1-monolithic-plus-skills case; under slim prompts, recommended-tier skills will have substantially more headroom. No formatter change is required, but the budget cap could be revisited in a future tuning phase.

---

## §3. Per-agent migration specs

Five subsections, one per agent. Each specifies the migration's current state, slim target, content-to-skill mapping, new-skill requirements (if any), task-type branching handling, risks, and success criteria.

### §3.1 OutlineAgent (3b.21)

#### 3.1.1 Current state

- **V1 line count:** 93.
- **V1 path:** `agent_prompts/outline_agent.yaml`.
- **`{{SKILLS}}` placeholder:** **ABSENT** — the only agent without it. (3b.12 wire-up confirmed DE has the placeholder from V2.0.1; the V2.0.1 `{{SKILLS}}` wire-up commit `3ba93a6` excluded OutlineAgent.)
- **Task-type branching:** none — no `_causal_soo` variant.
- **Content sections:** Role, Input, Output, Outline Design Rules (6 numbered items), Emphasis Triggers, Output Format.
- **Output schema:** JSON outline with `narrative_hook`, `sections[]` (each with id, title, subsections, emphasis, word_target, guidance).
- **The 3b.13 F-OUTLINE-AGENT-TYPE-ERROR:** OutlineAgent's LLM response parsing has surfaced `"string indices must be integers, not 'str'"` errors in some runs (e.g., 3b.13 cycle 1) — likely an LLM response-shape mismatch. The current prompt does not constrain the JSON shape tightly. Slim migration is an opportune moment to address this incidentally if it surfaces during verification.

#### 3.1.2 Slim target

- **Slim line count:** 25 (from `regression/proposed_slim/outline_agent.yaml`).
- **Structural elements:** role (1 sentence), `# Binding Rules` + `{{SKILLS}}` placeholder (NEW — wire-up step), role-and-contract block (2 sentences), output format, input listing.
- The slim draft's `Output Format` says "Output ONLY valid JSON wrapped in a ```json code block. No prose before or after." — same as the V1 prompt's last section.

#### 3.1.3 Content-to-skill mapping

| V1 content block | Migration target |
|---|---|
| Outline Design Rules 1 (sections always present) | `paper-narrative-outline` skill (V2.0; methodology layer) |
| Outline Design Rules 2 (descriptive subsection titles) | Same skill |
| Outline Design Rules 3 (emphasis allocation based on findings — SHAP, model-comparison, subgroup, sensitivity, ICC) | Same skill OR new `outline-emphasis-rules` skill |
| Outline Design Rules 4 (Methods can be compressed) | Same skill |
| Outline Design Rules 5 (Discussion subsections make arguments) | Same skill |
| Outline Design Rules 6 (narrative_hook drives the paper) | Same skill |
| Emphasis Triggers (pre-computed orchestrator data) | Reference in skill body; data still flows through user message |

The `paper-narrative-outline` skill (or its successor name) is the natural absorber. Sub-wave 0 did not verify whether that specific skill exists; 3b.21's pre-amendment investigation must confirm.

#### 3.1.4 New skills required

**Likely zero new skills.** The Outline Design Rules content is paper-structure guidance that should fit in a single methodology skill. If `paper-narrative-outline` doesn't yet exist, 3b.21 authors it (or extends an analogous skill) as part of the migration. If `paper-narrative-outline` exists but doesn't cover the emphasis-allocation rules, 3b.21 amends it.

#### 3.1.5 Task-type branching handling

None required at the prompt level — OutlineAgent has no variant. If outline design rules differ between prediction and causal tasks (e.g., causal papers don't have "Model Comparison" sections), the difference is handled by skill matching on `applicable_task_types`. The skill body can have task-type-conditional content rendered via the matcher.

#### 3.1.6 Risks and open questions

- **Risk:** `paper-narrative-outline` skill content may need amendment to cover causal-paper-specific emphasis rules (no SHAP, no model_quality_gate, but YES positivity diagnostics, balance plots, sensitivity tables). If the skill is prediction-only, 3b.21 must extend it before deleting V1 content.
- **Risk:** F-3b13-OUTLINE-AGENT-TYPE-ERROR may recur during 3b.21.5 verification if the slim prompt doesn't constrain output shape as tightly as the V1 verbose schema does. The slim draft's "Output Format" is brief; consider adding a JSON schema literal as in the V1 prompt.
- **Open question:** does the LSAR-named-gap subgroup analysis need an outline subsection? Currently outline rules say "promote Subgroup Analysis to prominent Results subsection if gaps > 5%" — this stays in slim form.

#### 3.1.7 Success criteria

- **Rendered-prompt tests** (mandatory before commit):
  - `{{SKILLS}}` placeholder substitutes correctly at OutlineAgent stage.
  - The `paper-narrative-outline` skill body (or equivalent) reaches the OutlineAgent's rendered prompt.
  - V1 content that was removed does NOT appear in the rendered prompt (i.e., it was removed from the prompt AND not re-added by a too-eager skill body).
- **Regression checks** (no new failures):
  - `tests/test_v3_0_phase_3b14_g5_dowhy_amendment.py`, `test_v3_0_phase_3b18_d1_encoding_amendment.py`, `test_rendered_prompt_contains_all_mskills.py` all pass unchanged.
  - V2.0 / V2.0.1 OutlineAgent-related tests (if they exist) pass.
- **3b.21.5 re-run** verifies pipeline completes; OutlineAgent produces a valid `paper_outline.json`; Writer consumes it without falling back to v1 template.

#### 3.1.8 Phase 3b.21 implementation outcome (post-3b.21)

**Status: MIGRATED.** Two commits on `phase-3-causal-inference`:
- `07c46c8` — V2.1 Phase 3b.21 (sub-wave 1): apply OutlineAgent slim + wire up skill injection
- (sub-wave 2 commit hash — added when sub-wave 2 lands)

**Sub-wave 0 corrections to the original 3b.21 hand-off:**

1. **The staged slim draft already contains `{{SKILLS}}`** (line 10 of `regression/proposed_slim/outline_agent.yaml`). The 3b.20 finding — that OutlineAgent's V1 production prompt LACKED `{{SKILLS}}` — is correct, but the V2.0.1-authored slim draft includes it. Applying the draft IS the wire-up; no separate placeholder-addition step required.

2. **Matcher/composer wire-up was Case 1 (already complete).** Orchestrator line 467 calls `self._inject_skills(outline_agent, "OutlineAgent")` and `BaseAgent.render_system_prompt()` substitutes `{{SKILLS}}`. The infrastructure was set up but its output was being silently discarded by V1's missing placeholder. Zero code changes needed in 3b.21 — neither orchestrator nor matcher nor BaseAgent.

These two corrections together mean **3b.21 is the simplest possible Case 1 migration:** a single-file overwrite of `agent_prompts/outline_agent.yaml` with the staged slim draft (+ V1 preserved as `.v1.yaml.bak`).

**Content preservation verified:**

The `paper-narrative-outline` skill (`skills/writing/paper-narrative-outline/SKILL.md`) carries the FULL V1 JSON schema verbatim (lines 31–78 of the skill = lines 15–62 of V1) + all 6 outline design rules + emphasis triggers description. Its source provenance line cites: *"Canonical source: `agent_prompts/outline_agent.yaml` (entire file). Merged content from: none — single-sourced."* V2.0.1 designed this skill specifically as the OutlineAgent V1 content absorber.

After slim apply + skill injection, the rendered OutlineAgent prompt contains the same schema fields and design rules as V1 had. The 29 verification tests in `tests/test_v2_1_phase_3b21_outline_slim.py` confirm this byte-by-byte (all 11 V1 schema fields + all 6 V1 design rules present in the rendered prompt).

**Migration metrics:**

| Metric | V1 | Post-3b.21 slim |
|---|---:|---:|
| `agent_prompts/outline_agent.yaml` lines | 93 | 25 (−73%) |
| `system_prompt` body lines | ~88 | ~21 |
| Has `{{SKILLS}}` placeholder | NO | YES |
| Output JSON schema | inline in prompt (48 lines) | delegated to `paper-narrative-outline` skill (same content) |
| 6 outline design rules | inline in prompt | delegated to `paper-narrative-outline` skill (verbatim) |
| Rendered prompt size at runtime | ~3.5 K chars | ~4 K chars (slim base ~470 + injected skill ~3.5 K) — comparable to V1, with mandatory-rule banner framing |

**F-3b13-OUTLINE-AGENT-TYPE-ERROR status: DEFERRED.** Sub-wave 0.5 found the error occurred ONCE in 3b.13 (the original detection) and did NOT recur in 3b.15 / 3b.17 / 3b.19 (4 consecutive successes). It's non-deterministic LLM-response-shape stochasticity, not a deterministic bug. The slim form provides better schema guidance (via the injected skill containing the full schema + design rules) than V1 had standalone — opportunistic probability reduction, but not deterministic prevention. Per single-issue discipline: no F-3b13 regression test in 3b.21. Closure deferred until either (a) the error recurs and the slim form's effect can be measured, or (b) a future phase explicitly targets it.

**Test surface:** 29 new tests in `tests/test_v2_1_phase_3b21_outline_slim.py`:
- TestOutlineSlimApplied (3): line count, no V1 schema block residue, `{{SKILLS}}` present.
- TestOutlineSkillInjectionWorks (3): paper-narrative-outline matches at OutlineAgent + Writer stages; matches for both causal and prediction task types; placeholder substitutes (no leakage).
- TestOutlineSchemaPreserved (11 parametrized): all V1 schema fields appear in the rendered prompt.
- TestOutlineDesignRulesPreserved (6 parametrized): all 6 V1 design rules appear in the rendered prompt.
- TestOutlineRolePreserved (4): agent_name, model_config_key, temperature, max_tokens, role statement.
- TestV1BackupPreserved (2): `.v1.yaml.bak` file exists with V1 content.

All 29 tests pass.

**Wire-up precedent for 3b.22+:** the 3b.21 migration confirms the V2.0.1 infrastructure pattern fully works:
1. Apply the staged slim draft (which already has `{{SKILLS}}`).
2. Back up V1 as `.v1.yaml.bak`.
3. Orchestrator's existing `_inject_skills(<agent>, "<AgentName>")` call site provides matcher routing automatically.
4. `BaseAgent.render_system_prompt()` provides substitution automatically.

Subsequent migrations (3b.22 PF, 3b.23 Critic, 3b.24 Writer, 3b.25 Analyst) inherit this pattern. The only per-agent variability is which skills must exist (and possibly be authored) to absorb the V1 content being removed.

#### 3.1.9 Phase 3b.21.5 live validation (post-3b.21.5)

**Status: VALIDATED LIVE.** Run `v3_0_smoketest_mtheff_college_20260701_3b21_5` (commit `bf25f72`), single-variable Δ vs 3b.19 (only `agent_prompts/outline_agent.yaml` differs among pipeline-consumed files). COMPLETED in a single attempt, 73m3s. All four acceptance criteria PASS:

| Criterion | Verdict | Evidence |
|---|---|---|
| AC1 outline structural validity | PASS | `narrative_hook` + 5 canonical sections; emphasis pattern shape-identical to 3b.19's V1 output |
| AC2 emphasis triggers fire | PASS | Skill §Emphasis-triggers in live system prompt; orchestrator's pre-computed trigger JSON in user message; values legitimately falsy |
| AC3 F-3b13 no recurrence | PASS | Valid JSON, single call, 26s; 5th consecutive clean run, first on slim |
| AC4 no new injection failures | PASS | 0 literal `{{SKILLS}}`; skill-body markers present; no formatter drops at OutlineAgent |

**Measured slim effect:** OutlineAgent 7,109 tokens vs 8,185 in 3b.19 (−13.1%); 26s vs 80s. LSAR gate passed at 6.1 "Weak Accept" (+0.2 vs 3b.19, non-degradation confirmed).

**F-3b13-OUTLINE-AGENT-TYPE-ERROR: CLOSURE RECOMMENDED** (upgrades §3.1.8's DEFERRED). The 3b.21.5 run is the 5th consecutive non-recurrence and the first on the slim prompt — condition (a) of the deferral (recurrence) never materialized across the entire observation window, and the migration replaced the prompt wholesale. Closed as overtaken-by-migration; reopen only if the shape reappears.

Analyst-stage churn in the run (M1 dtype failure, cycle-0 exec timeout — see `runs/.../REPORT.md` §7) is upstream of the OutlineAgent in the pipeline DAG and NOT slim-attributable.

---

### §3.2 ProblemFormulator (3b.22)

#### 3.2.1 Current state

- **V1 line count:** 207 (prediction-task; base).
- **V1 path:** `agent_prompts/problem_formulator.yaml`.
- **`{{SKILLS}}` placeholder:** present (V2.0.1 wire-up, line 16).
- **Task-type branching:** **YES** — `agent_prompts/problem_formulator_causal_soo.yaml` (113 lines, already slim) is the causal variant.
- **Content sections (V1 base):**
  - Role declaration (4 lines)
  - Output schemas (research_spec + literature_context JSON, ~50 lines)
  - Literature Selection (CRITICAL — select 8–12 papers) (~20 lines)
  - **Validation Rules 1–14 (~73 lines)** — temporal ordering, sample size, novelty calibration, gap-driven framing, surprising-predictor emphasis, predictor coherence, Findings Memory integration
  - Canonical Research Questions (examples)
  - Input description
  - Using Findings Memory
  - Generating Diverse Candidates
  - Critical Constraints

#### 3.2.2 Slim target

- **Slim line count:** 95 (from `regression/proposed_slim/problem_formulator.yaml`).
- The slim draft retains: role + Binding Rules + research_spec/literature_context schemas + Output Format. It removes Validation Rules 1–14 (delegated to skills) and the verbose Findings-Memory / Diverse-Candidates / Canonical-Questions sections.

#### 3.2.3 Content-to-skill mapping

| V1 content block | Migration target |
|---|---|
| Validation Rule 1 (temporal ordering) | `hsls09-temporal-ordering` (V2.0 dataset skill, already exists) |
| Validation Rule 2 (sample size ≥ 10,000) | New `prediction-feasibility-floor` skill OR existing methodology skill (`missingness-tiered-protocol` covers retention logic) |
| Validation Rule 3 (novelty self-assessment ≥ 3) | New `research-question-design` skill (covers Rules 3, 9, 10, 11, 13) |
| Validation Rule 4 (predictor rationale) | Same `research-question-design` skill |
| Validation Rule 5 (outcome from registry outcomes section) | Same skill OR `hsls09-variable-registry` (V2.0) |
| Validation Rule 6 (protected attributes in subgroup_analyses) | `subgroup-fairness-analysis` (V2.0) — extend to include the predictor-set rule |
| Validation Rule 7 (no redundant composites) | Existing `no-redundant-composites` skill (authored in V2.0.1 Phase 2c per commit `092c7b3`) |
| Validation Rule 8 (no causal-task framing on prediction) | New `prediction-vs-causal-framing` skill OR existing task-type skill |
| Validation Rule 9 (gap-driven framing — REJECTED/ACCEPTED examples) | `research-question-design` skill |
| Validation Rule 10 (contrast framing) | Same skill |
| Validation Rule 11 (surprising predictor emphasis) | Same skill |
| Validation Rule 12 (predictor set coherence) | Same skill |
| Validation Rule 13 (novelty calibration) | Same skill |
| Validation Rule 14 (Findings Memory build-on) | Existing `findings-memory-novelty-cross-run` skill (V2.0; check coverage) |
| Canonical Research Questions list | Move to `data_registry/datasets/hsls09_public.yaml` `canonical_research_questions:` (already there in the registry per the V3.0 audit; just reference it) |
| Using Findings Memory section | `findings-memory-novelty-cross-run` skill |
| Generating Diverse Candidates section | `findings-memory-novelty-cross-run` skill OR new candidate-diversity skill |

#### 3.2.4 New skills required

**Up to 2 new skills.** Sub-wave 0 did not enumerate the existing skill inventory exhaustively; 3b.22's pre-amendment investigation must confirm coverage. If gaps exist:

1. **`research-question-design`** (methodology layer, `applicable_stages: [ProblemFormulator]`, `applicable_task_types: [prediction]` — or task-type-agnostic if causal-mode PF also benefits). Absorbs V1 Validation Rules 3, 4, 9, 10, 11, 12, 13. Mandatory severity (poor research question is a structural failure mode). Contracts:
   - Mandatory rule: research_question must identify a SPECIFIC GAP, not just ask "Can we predict X?"
   - Mandatory rule: contrast framing (the answer must be non-obvious).
   - Mandatory rule: novelty_score ≥ 3 calibration; what scores 3/4/5 mean.
   - Mandatory rule: predictor-set coherence (theoretically motivated, not kitchen-sink).
   - Mandatory rule: surprising-predictor emphasis when attitudinal/non-cognitive predictors are present.

2. **`prediction-feasibility-floor`** (methodology layer, `applicable_stages: [ProblemFormulator]`, `applicable_task_types: [prediction]`). Absorbs V1 Validation Rules 2 and 5. Mandatory severity. Contracts:
   - analytic_n ≥ 10,000 after listwise deletion (estimate from registry pct_missing).
   - Outcome must come from registry outcomes section, not predictors.

If `research-question-design` already exists (need to verify in 3b.22's investigation), the migration uses it. If not, 3b.22 authors it as part of the migration.

#### 3.2.5 Task-type branching handling

PF has two prompts: `problem_formulator.yaml` (base, prediction-task — 207 lines V1, 95 lines slim) and `problem_formulator_causal_soo.yaml` (113 lines, already slim). After 3b.22 migration: both prompts in slim form, both with `{{SKILLS}}` placeholder, each routed by `load_prompt('problem_formulator', task_type=...)`.

The skills attached to each prompt are governed by `applicable_task_types`. Prediction-task skills (`research-question-design`, `prediction-feasibility-floor`, etc.) attach to prediction prompts; causal-task skills (G1, G2, D1, `causal-data-engineer-contract`, etc.) attach to causal prompts. No prompt-level branching is changed.

#### 3.2.6 Risks and open questions

- **Risk:** The V1 prompt contains content (e.g., "REJECTED patterns" / "ACCEPTED patterns" in Validation Rule 9) that is concrete and prescriptive. If the new `research-question-design` skill doesn't preserve that prescriptive shape, the LLM may regress to generic research questions.
- **Risk:** Findings Memory integration is currently described verbosely in the V1 prompt. The existing `findings-memory-novelty-cross-run` skill may not cover the same surface. If 3b.22 surfaces a gap, the skill needs amendment before the migration commits.
- **Open question:** is the registry's `canonical_research_questions:` field actually consumed by the matcher, or does the V1 prompt's enumeration need to live somewhere the LLM can see? Sub-wave 0 didn't verify the consumption path. If the registry field is informational-only, the canonical-questions list must move to a skill body.

#### 3.2.7 Success criteria

- Rendered-prompt tests: all skill bodies (existing + new) reach the rendered PF prompt; V1-removed content does NOT appear.
- JSON schema regression: `research_spec.json` produced under slim PF matches V1's schema byte-identical (downstream agents — DE, Analyst, Writer — depend on field names + types).
- F-3b11-PRECRITIC-PREDICTION-CARRYOVER (a known open backlog item): post-3b.22 slim PF will still produce prediction-shaped research_spec for prediction tasks; the carryover surfaces in Critic's pre-critic checks regardless of PF slim. Not a 3b.22 blocker.
- 3b.22 has no associated re-run (re-runs are interleaved at 3b.21.5, 3b.23.5, 3b.25.5).

#### 3.2.8 Phase 3b.22 implementation outcome (post-3b.22)

**Status: MIGRATED.** Sub-wave 1 applied in commit `b97fd12`; verification suite + this doc update in sub-wave 2. Base prompt 207 → 95 lines (54% reduction); V1 backed up to `agent_prompts/problem_formulator.v1.yaml.bak`.

**Three corrections to this section's pre-migration expectations:**

1. **§3.2.4's "up to 2 new skills" resolved to ZERO.** The hypothesized `research-question-design` exists as `prediction-research-question-design` (task-type layer, `rule_severity: mandatory`, matches at ProblemFormulator/prediction). It carries Rules 3–4, 7, 9–13 in full prescriptive form — including the REJECTED/ACCEPTED example patterns verbatim and the novelty-calibration table — and Rules 1, 2, 5, 6, 8 as cross-referenced summaries under "Other validation rules". The hypothesized `prediction-feasibility-floor` is therefore unnecessary (Rules 2 + 5 covered).
2. **§3.2.6's open question RESOLVED — the canonical-questions list travels via the data path.** `ProblemFormulator._build_user_message()` yaml-dumps the FULL registry into the user message, and the registry carries `canonical_research_questions:`. No skill needs to carry the list. Bonus finding: the V1 inline list contained two variable names that do not exist in the registry (`X4ENTMJST`, `X3TGPAALL`) — dropping the inline list is a correction, not a loss.
3. **§3.2.3's Rule-7 mapping cell named a standalone `no-redundant-composites` skill — none exists.** The rule lives as a dedicated mandatory section inside `prediction-research-question-design` (with the HSLS-specific disallowed pairings and the pre-emit checklist). Mapping holds; location differs.

**Content preservation verified two ways:**
- 21-agent adversarial verification workflow (one refuter per V1 content block: 14 validation rules, literature selection, both schemas, findings-memory usage, diverse candidates, critical constraints, revision handling) against the rendered slim prompt: **21/21 PRESERVED, 0 weakened, 0 lost**.
- Formatter-cap check: all 6 skills matched at ProblemFormulator/prediction render IN (block 24,318 chars < 30K non-mandatory budget); all 7 at causal_soo render IN (mandatory bypass). Locked by a regression test so future skill growth can't silently drop V1-migrated content.

**Task-type branching (§3.2.5) confirmed as specified:** `problem_formulator_causal_soo.yaml` (113 lines, already slim, has `{{SKILLS}}`) untouched; `load_prompt` routing unchanged; `prediction-research-question-design` correctly does NOT match under causal_soo (its Pre-Emit reference in the slim base prompt is safe because the base file is only rendered for prediction).

**Migration metrics:**

| Metric | V1 (pre-3b.22) | Slim (post-3b.22) |
|---|---|---|
| Base prompt lines | 207 | 95 |
| `{{SKILLS}}` placeholder | present (V2.0.1) | present |
| Validation Rules 1–14 location | in-prompt | `prediction-research-question-design` + dataset skills |
| Literature Selection location | in-prompt | `literature-search-s2-arxiv` |
| Findings-Memory / Diverse-Candidates location | in-prompt | `findings-memory-novelty-cross-run` |
| Canonical questions location | in-prompt (2 hallucinated var names) | registry data path (authoritative) |
| Output schemas | in-prompt | in-prompt (unchanged — field names byte-identical) |
| New skills authored | — | 0 |
| Orchestrator/code changes | — | 0 (Case-1 wire-up) |

**Routing-test expectation repair (sub-wave 1):** two tests in `tests/test_v1_prompt_task_branching.py` asserted V1 body content ("Do not propose causal") in the raw base prompt file; post-migration the rule reaches the rendered prompt via the skill. Updated to assert routing invariants (base-file selection, legacy-path equivalence, no CAUSAL_SOO leakage); the content assertion moved to the 3b.22 verification suite (rule-08 marker). Module docstring amended to scope 3b.4's "V1 prompts unchanged" contract as predating V2.1.

**Test surface:** 59 tests in `tests/test_v2_1_phase_3b22_pf_slim.py` (10 classes): SlimApplied (4), SkillInjectionWorks (4, incl. the formatter-cap lock and the causal non-leak check), ValidationRulesPreserved (15 parametrized), LiteratureSelectionPreserved (4), FindingsMemoryPreserved (3), SchemaPreserved (16), UniversalConstraintsPreserved (5), RolePreserved (3), CausalVariantUntouched (3), V1BackupPreserved (2). All pass.

**Precedent notes for 3b.23 (Critic):** this migration adds two reusable checks to the 3b.21 four-step pattern — (5) verify every matched skill survives the per-tier formatter cap at the target stage (the 30K budget drops non-mandatory skills silently; 3b.19/3b.21.5 stdout shows real drops at Critic and Writer stages, so this check is LOAD-BEARING for 3b.23/3b.24); (6) sweep existing tests for stale V1-body-content assertions before committing the apply (grep tests/ for distinctive V1 prompt phrases).

---

### §3.3 Critic (3b.23)

#### 3.3.1 Current state

- **V1 line count:** 235.
- **V1 path:** `agent_prompts/critic.yaml`.
- **`{{SKILLS}}` placeholder:** present (V2.0.1, line 14).
- **Task-type branching:** **NO** — no `_causal_soo.yaml` variant. Critic is task-type-agnostic at prompt level; task-type behavior comes from skill matching + the orchestrator's deterministic verdict-evaluator (3b.10).
- **Content sections:**
  - Role declaration
  - Binding Rules + `{{SKILLS}}`
  - Structured Reasoning Protocol (Lens A Methodologist, Lens B Skeptic, Lens C Synthesizer)
  - Output (review_report.json schema)
  - Review Checklist (Problem Formulation, Data Preparation, Analysis, Substantive Validity) — these are PREDICTION-shaped (mention SHAP, AUC, SMOTE, class imbalance, ablation)
  - Verdict Criteria table (overridden by 3b.10 deterministic evaluator at orchestrator level)
  - Revision Instructions guidance
  - Severity Definitions
  - Optional: Novelty Review Against Prior Runs
  - Input description
  - Critical Rules

#### 3.3.2 Slim target

- **Slim line count:** 123 (from `regression/proposed_slim/critic.yaml`).
- The slim draft retains: role, Binding Rules, structured reasoning protocol, review_report.json schema, severity definitions. It removes: the prediction-shaped review checklist (covered by skills), Verdict Criteria table (orchestrator owns this since 3b.10), Optional Novelty Review (redundant with skill matching).

#### 3.3.3 Content-to-skill mapping

| V1 content block | Migration target |
|---|---|
| Review Checklist: Problem Formulation items | New `critic-checklist-problem-formulation-prediction` skill OR extend existing `pre_critic_checks` Python code to render the list |
| Review Checklist: Data Preparation items | New `critic-checklist-data-preparation` skill |
| Review Checklist: Analysis items (SHAP, AUC, SMOTE) | New `critic-checklist-analysis-prediction` skill (prediction-task-specific) |
| Review Checklist: Substantive Validity | New `critic-checklist-substantive-validity` skill |
| Verdict Criteria table | Already handled by orchestrator's `src/verdict_evaluator.py` (3b.10) — REMOVE from prompt |
| Optional: Novelty Review | Already handled by `findings-memory-novelty-cross-run` skill — REMOVE from prompt |

For CAUSAL tasks, the Critic needs a different checklist surface — review items about estimand declaration, positivity, balance, sensitivity. Two design options:

- **Option A:** Single Critic prompt + task-type-aware checklist skills. The slim Critic prompt has a generic `## Review Checklist` section that the matcher fills with task-type-matched skill bodies (`critic-checklist-prediction` for prediction tasks; `critic-checklist-causal-soo` for causal tasks).
- **Option B:** Author `critic_causal_soo.yaml` variant analogous to PF/Analyst/Writer variants. Each prompt has the appropriate review checklist baked in (or pulled from skills).

**Recommendation: Option A.** The Critic prompt is already task-type-agnostic structurally; only the checklist contents differ by task type. Skill matching is the right level of granularity. This also avoids creating a Critic variant for the first time — Critic has uniquely stayed without one since 3b.4.

#### 3.3.4 New skills required

**Likely 4 new skills.** All `applicable_stages: [Critic]`:

1. **`critic-checklist-problem-formulation-prediction`** (methodology, `applicable_task_types: [prediction]`, mandatory). Covers the V1 PF-review checklist items.
2. **`critic-checklist-data-preparation`** (methodology, task-type-agnostic, mandatory). The V1 data-preparation checklist items are mostly cross-task (no-leakage, n≥10p, school-aware-split, multilevel-acknowledged, missingness-protocol-followed).
3. **`critic-checklist-analysis-prediction`** (methodology, `applicable_task_types: [prediction]`, mandatory). Covers SHAP / AUC / SMOTE / class-imbalance / ablation review.
4. **`critic-checklist-substantive-validity`** (methodology, task-type-agnostic, mandatory). Covers Top-features-make-sense / findings-not-trivial / AUC>0.95-suspicious / limitations-honest / unexpected-flagged.

The CAUSAL counterpart of `critic-checklist-analysis-prediction` is implicitly covered by the existing causal skills (G1–G5, D1, M1–M5 — these are mostly Analyst-stage but several already attach to Critic too per the V3.0 audit §4.2). 3b.23's pre-amendment investigation confirms whether a `critic-checklist-analysis-causal-soo` skill is also needed.

#### 3.3.5 Task-type branching handling

**No new prompt variant.** Critic stays single-prompt; task-type-aware skill matching provides the checklist surface. This is Option A from §3.3.3.

#### 3.3.6 Risks and open questions

- **Risk (high):** The 3b.10 deterministic verdict-evaluator depends on `review_report.json` having specific fields (`overall_verdict`, `overall_quality_score`, `problem_formulation_review.score`, etc.). Slim Critic must preserve the schema byte-identically. The schema-regression test is critical here.
- **Risk (medium):** F-3b11-PRECRITIC-PREDICTION-CARRYOVER — `pre_critic_checks.py` (orchestrator-side Python) is hardcoded to prediction-task fields (`all_models`, `best_metric_value`, etc.). Even with slim Critic prompt, the pre-critic check fires prediction-task false-positives on causal runs. This is the V2.1 backlog item identified in 3b.17 / 3b.19; not a 3b.23 blocker but a known interaction.
- **Risk (low):** The "Structured Reasoning Protocol" (Lens A/B/C) in the V1 prompt is a prompting technique; slim form retains it. If runtime exercise shows it has minimal effect, it can be cut in a future amendment.
- **Open question:** is there a meaningful difference between "task-type-aware checklist skills" (Option A) and a hypothetical `critic_causal_soo.yaml` variant (Option B) at the rendered-prompt level? The matcher already provides task-type-conditional skill bodies; the variant approach is essentially "freeze the matcher's choice at design time." Option A is more flexible. 3b.23 confirms during implementation.

#### 3.3.7 Success criteria

- Rendered-prompt tests: existing skills + the 4 new checklist skills reach the rendered Critic prompt; V1 checklist content is GONE; Verdict Criteria table is GONE.
- Schema regression: `review_report.json` produced under slim Critic matches V1's schema byte-identical.
- 3b.10 deterministic verdict-evaluator integration: confirm `verdict_evaluator.py` still parses Critic's output correctly under slim Critic. Run the existing 3b.10 verdict-evaluator tests.
- 3b.23.5 re-run verifies pipeline completes cleanly; Critic produces cycle-0 + cycle-1 verdicts of expected shape.

#### 3.3.8 Phase 3b.23 implementation outcome (post-3b.23)

**Status: MIGRATED.** Sub-wave 1 applied in commit `a922b78`; verification suite + this doc update in sub-wave 2. Applied prompt 235 → 140 lines (40% reduction; staged draft 123 + two harvested blocks); V1 backed up to `agent_prompts/critic.v1.yaml.bak`.

**Corrections to this section's pre-migration expectations:**

1. **§3.3.4's "likely 4 new skills" resolved to ZERO.** The existing `prediction-critic-checklist` (task-type layer) already carries all four checklist sections with per-item IDs/severities/checks, plus verdict criteria and severity definitions. The hypothesized per-section skills were unnecessary.
2. **§3.3.2's claim that the staged draft removes the Verdict Criteria table is wrong** — the draft retains it, and retention is correct: the table guides the LLM's self-assessment while the 3b.10 evaluator owns the effective verdict. Bonus alignment: the checklist skill's REVISE row includes the `overall_quality_score < 7` floor (matching the evaluator), which V1's prompt table omitted.
3. **Semantic Δ, deliberate:** V1's prompt said `overall_quality_score` is the *plain mean* of section scores; the checklist skill (V2.0) and the slim prompt both use the *weighted mean* (PF 0.25 / DP 0.25 / Analysis 0.30 / Substantive 0.20). V1's prompt contradicted the skill; the slim resolves the contradiction in the skill's favor. The 3b.10 evaluator consumes the LLM-reported score either way.

**Harvest amendments (runbook step 3, driven by adversarial verification — initial pass: 10 PRESERVED / 2 WEAKENED / 1 LOST across 13 V1 content blocks):**

- `prediction-critic-checklist` v1.0 → v1.1: added `dp_08` (school-aware split verified via `data_report.split_info.group_overlap == 0`, critical), `dp_09` (`is_imbalanced` consistency, major), `an_11`–`an_13` (ablation-present / ablation-`n_test`-parity-critical / F2+balanced-accuracy). Root cause: their nominal carrier skills (`school-aware-train-test-split`, `smote-imbalance-handling`) are DataEngineer-/Analyst-stage-only and never reach the Critic prompt.
- `prediction-critic-checklist` tagged `rule_severity: mandatory` per §3.3.4's prescription. This is the LOAD-BEARING cap fix: live runs (3b.19, 3b.21.5) drop three non-mandatory skills at the Critic stage under the 30K per-tier budget; a cap-droppable checklist is a structurally incomplete review.
- "Revision Instructions" guidance harvested into the applied prompt (+9 lines) — no skill carried it.
- `novelty_review` output contract added to the applied prompt body (+9 lines). Initially LOST: its nominal carrier (`findings-memory-novelty-cross-run`, non-mandatory) is **cap-dropped at Critic under causal_soo in live runs** — verified in 3b.19/3b.21.5 stdout and reproduced offline. Design rule extracted: **output-format contracts must live in the prompt body, never in cap-droppable skills.**

Post-harvest adversarial recheck: 3/3 PRESERVED (final 13/13; verifiers noted the harvested rows are strictly stronger than V1 — explicit severities, unconditional `n_test` parity, and `dp_09` guards the flag that gates `an_11`–`an_13`).

**Production bug found and fixed (F-3B23-PROMPT-ENCODING-MOJIBAKE):** `load_prompt` / `load_registry` / `load_task_template` opened YAML without `encoding="utf-8"`; on Windows (cp1252) every em dash and typographic character in every agent prompt was silently mojibaked in every live run to date. Surfaced by the 3b.23 rendered-prompt tests' non-ASCII markers; four `open()` sites fixed in `src/agents/base.py`. Prompts are now byte-faithful; no behavioral regression observed (models parsed through the mojibake).

**Methodology lesson for offline verification:** bare `SkillRegistry.match()` under-represents the production skill set (no context keywords, different caps) — it produced one false LOST and several false-IN readings (skill-NAME mentions in other skills' cross-references vs actual body presence). Offline rendering must reproduce the orchestrator path: `match_and_compose(stage, task_type, dataset, context, top_k_per_layer=_resolve_skill_caps(task_type))`, and presence checks must use body-content markers, not skill names.

**Migration metrics:**

| Metric | V1 (pre-3b.23) | Slim (post-3b.23) |
|---|---|---|
| Prompt lines | 235 | 140 |
| `{{SKILLS}}` placeholder | present (V2.0.1) | present |
| Review checklist location | in-prompt (prediction-shaped, also noise on causal runs) | `prediction-critic-checklist` (mandatory; prediction-only — causal runs get the causal review skills instead) |
| Verdict criteria | in-prompt (plain-mean formula, no quality floor in REVISE row) | in-prompt (weighted-mean, aligned with skill + 3b.10 evaluator) |
| novelty_review contract | in-prompt | in-prompt (kept — cap-droppable skill is not a safe carrier) |
| Task-type variants | none | none (Option A confirmed; routing test locks it) |
| New skills authored | — | 0 (5 rows harvested into the existing checklist skill) |
| Orchestrator/code changes | — | encoding fix in `base.py` (bug fix, not wire-up) |

**Test surface:** 60 tests in `tests/test_v2_1_phase_3b23_critic_slim.py` (10 classes), including the formatter-cap survival lock, the verdict-evaluator integration (spec §3.3.7), the Option-A routing lock, and cross-task marker checks against orchestrator-path renderings for both task types. All pass; full suite 874 passed / 4 pre-existing failures.

**Precedent notes for 3b.24 (Writer):** three additions to the reusable check set — (7) offline rendering must use the orchestrator path (`match_and_compose` + caps + context), never bare `match()`; (8) output-format contracts stay in the prompt body, never in cap-droppable skills (Writer's BibTeX/figure-discipline skills are non-mandatory and DID drop at Writer in live runs — audit them before slimming); (9) keep non-ASCII markers in rendered-prompt tests — they catch encoding bugs the ASCII-only 3b.21/3b.22 suites missed.

---

### §3.4 Writer (3b.24)

#### 3.4.1 Current state

- **V1 line count:** 485 (prediction-task base).
- **V1 path:** `agent_prompts/writer.yaml`.
- **`{{SKILLS}}` placeholder:** present (V2.0.1, line 10).
- **Task-type branching:** **YES** — `agent_prompts/writer_causal_soo.yaml` (83 lines, already slim).
- **Content sections (V1 base):**
  - Role + Binding Rules
  - Output spec (paper.tex + references.bib)
  - LaTeX Template — Fill Placeholders, Do NOT Generate Structure (template-reassembly safeguard from V1.2)
  - Rules for placeholder filling
  - Placeholder Reference table
  - Automated Generation Disclosure
  - Tables (CRITICAL — column count must match, threeparttable for notes)
  - Figures (CRITICAL — full figure environment for every figure)
  - Title Format
  - Paper Structure and Word Counts
  - Per-section content rules: INTRODUCTION, RELATED_WORK, METHODS_MODELS, RESULTS_MODEL_COMPARISON, RESULTS_ABLATION, DISCUSSION_*
  - BibTeX Generation
  - Citation Rules (CRITICAL)
  - UNVERIFIED Flag (Critic-verdict-not-PASS handling)
  - Writing Style Rules
  - HSLS:09 Methodological Context: Multilevel Structure, Survey Weights, Sensitivity Analysis, Model Quality Caveats, School Cluster Reconstruction
  - Input
  - Critical Rules

#### 3.4.2 Slim target

- **Slim line count:** 77 (from `regression/proposed_slim/writer.yaml`).
- The slim draft retains: role + Binding Rules + Output spec + a brief Critical Rules section. All template-filling guidance, per-section content rules, citation rules, UNVERIFIED-flag handling, and HSLS:09 methodological context move to skills.

#### 3.4.3 Content-to-skill mapping

| V1 content block | Migration target |
|---|---|
| LaTeX Template rules (placeholder-fill, no-structure-generation) | Existing `acm-acmart-sigconf-template` skill (V2.0 writing layer) |
| Placeholder Reference table | Same skill |
| Tables rules (column count, threeparttable) | New `latex-table-discipline` skill (or extend existing writing-layer skill) |
| Figures rules (full environment, includegraphics) | New `latex-figure-discipline` skill — **MENTIONED in 3b.9 report as already-existing** |
| Title Format | `paper-title-format` skill (likely exists; verify in 3b.24 investigation) |
| Paper Structure + Word Counts | `paper-narrative-outline` skill (shared with OutlineAgent's source skill) |
| Per-section content rules | `paper-section-content-prediction` skill (per-task-type) |
| BibTeX Generation | Existing `bibtex-from-s2-metadata` skill (V2.0) |
| Citation Rules | Same skill |
| UNVERIFIED Flag handling | `unverified-paper-banner` skill (per V2.0 writing layer; verify) |
| Writing Style Rules | `paper-style-rules` skill (V2.0 writing layer; mentioned in 3b.9 report) |
| HSLS Multilevel Structure paragraph | `hsls09-multilevel-limitations-paragraph` skill (V2.0 writing layer; verify name) |
| HSLS Survey Weights paragraph | `hsls09-survey-weights-limitations-paragraph` skill (V2.0; confirmed in 3b.9) |
| Sensitivity Analysis reporting | G5 `causal-sensitivity-unmeasured-confounding` skill (V3.0; covers Writer-side interpretation per 3b.14 amendment) |
| Model Quality Caveats | `model-quality-gate` skill OR extend existing |
| School Cluster Reconstruction paragraph | D1 `hsls09-causal-conventions` (V3.0; cluster section already in body) OR a dedicated `hsls09-cluster-reconstruction-paragraph` writing skill |

Many skills referenced here are V2.0 writing-layer skills that the V3.0 audit cites but that Sub-wave 0 did not exhaustively enumerate. 3b.24's pre-amendment investigation confirms exact skill names.

#### 3.4.4 New skills required

**Up to 3 new skills.** Exact count depends on 3b.24's investigation:

1. **`latex-table-discipline`** if not already covered (threeparttable rule, column-count matching, wide-table resizing). High likelihood of existing under a different name; verify.
2. **`paper-section-content-prediction`** to absorb per-section content rules for prediction papers (INTRODUCTION, RELATED_WORK, METHODS_MODELS, RESULTS_MODEL_COMPARISON, RESULTS_ABLATION, DISCUSSION). The causal counterpart is the existing `paper-narrative-outline` plus method-skill Writer-interpretation guidance.
3. **`hsls09-cluster-reconstruction-paragraph`** as a Writer-side companion to D1's analysis-side cluster rules. May be folded into D1 or an existing writing skill.

#### 3.4.5 Task-type branching handling

Same as PF / Analyst: base prompt (`writer.yaml` — 485 V1 → 77 slim) handles prediction; `writer_causal_soo.yaml` (83 lines, already slim) handles causal. Both have `{{SKILLS}}` placeholder; per-task-type skill matching provides the section-content / methodology bodies.

#### 3.4.6 Risks and open questions

- **Risk (high):** LaTeX template-fill is fragile. V1 contains extensive "CRITICAL" prose around tables and figures because the Writer LLM has historically broken LaTeX in subtle ways (column-count mismatches, missing `\end{tablenotes}`, font declarations outside braces). Moving these rules to skills must preserve their CRITICAL framing. If the skill body's rendering downgrades them to suggestions, paper-compile regressions occur.
- **Risk (medium):** UNVERIFIED Flag handling depends on the Critic's verdict propagating correctly. The 3b.10 deterministic verdict-evaluator's PASS-UNVERIFIED path must continue to trigger the Writer's UNVERIFIED banner. Schema regression test for `review_report.json.overall_verdict` is the safety net.
- **Risk (low):** Writing Style Rules (active voice, "students" not "subjects", no causal language for correlational findings) are mostly cross-task. Some are task-type-specific (no causal language only applies in prediction mode; causal mode is allowed causal language with the no-unmeasured-confounding caveat). Slim form must route correctly.
- **Open question:** the V1 prompt's "Paper Structure and Word Counts" table specifies prediction-paper structure (Abstract / Intro / Related Work / Methods / Results / Discussion with specific word ranges). The causal variant's paper structure is similar but with different Results subsections. Whether one shared `paper-narrative-outline` skill can cover both or whether per-task-type subsections are needed is a design question for 3b.24 implementation.

#### 3.4.7 Success criteria

- Rendered-prompt tests: all skill bodies reach the Writer rendered prompt; V1 per-section / template / style content is GONE.
- LaTeX-compile regression: post-3b.24 Writer's paper.tex compiles cleanly via `pdflatex → bibtex → pdflatex → pdflatex`. The 4 pre-existing `test_writer.py::TestTemplateFile` failures persist (unrelated to migration; documented out of scope) but no NEW compile failures.
- Paper word-count regression: post-3b.24 Writer produces papers in the 4,000–6,500 word range (the 3b.13 / 3b.15 / 3b.17 / 3b.19 observed range). If word count drops below 2,000 (LSAR sanity-check floor), the slim Writer is under-specifying paper content.
- 3b.24 has no associated re-run (interleaved at 3b.25.5).

#### 3.4.8 Phase 3b.24 implementation outcome (post-3b.24, executed within V4 Arc H)

**Status: MIGRATED.** Writer 485 → 77 lines (84% reduction); V1 backed up. Corrections to this section's expectations: (1) of the “up to 3 new skills”, ONE was authored (`paper-section-content-prediction`, mandatory, prediction-only — absorbs title patterns, word budgets, per-section rules, limitations ordering, quality caveats, sensitivity reporting); `latex-table-discipline` existed; the cluster-reconstruction paragraph was already carried by the multilevel-limitations skill. (2) The V1 per-slot placeholder table is DEAD contract — the v1 template was deliberately deleted in V2.0 Phase 2c (9dbb987); TestTemplateFile was rewritten to the v2 (PAPER_BODY) contract, fixing the last long-standing suite failures. (3) LOAD-BEARING severity work: five writing skills retagged mandatory (acm template, figure discipline, bibtex, both HSLS limitation paragraphs) — live runs cap-drop recommended writing skills at Writer, and these carry compile-critical/structural content. fonts-in-braces harvested into latex-table-discipline v1.1. Marker lesson recorded: skill bodies line-wrap and capitalize — use wrap-safe, case-exact markers. Verification: 30 tests in test_v2_1_phase_3b24_writer_slim.py; live validation interleaved at 3b.25.5.

---

### §3.5 Analyst (3b.25)

#### 3.5.1 Current state

- **V1 line count:** 605 (prediction-task base) — **the largest agent prompt**.
- **V1 path:** `agent_prompts/analyst.yaml`.
- **`{{SKILLS}}` placeholder:** present (V2.0.1, line 11).
- **Task-type branching:** **YES** — `agent_prompts/analyst_causal_soo.yaml` (97 lines, already slim).
- **Content sections (V1 base):**
  - Role + Binding Rules
  - Output (results.json + CSVs + figures)
  - results.json Schema
  - Pilot Model Battery (LR, RF, XGBoost, ElasticNet, MLP, StackingEnsemble)
  - MLP Toggle
  - Hyperparameter Tuning (per-model grids)
  - Evaluation Protocol (classification + regression + CIs + figures)
  - Class Imbalance Handling (SMOTE + Ablation; 4 steps)
  - SHAP Interpretability Protocol (explainer dispatch, sample caps, fallback)
  - Figure Outputs
  - Subgroup Analysis
  - High-Missingness Warning
  - Model Quality Gate (CRITICAL — AUC floor 0.60, R² floor 0.05)
  - Sensitivity Analysis (MANDATORY for high-missingness)
  - Clustered Standard Errors and ICC (ICC computation, clustered bootstrap, missing-school-IDs handling)
  - Error Handling
  - Critical Rules — Never Violate
  - Data Loading — CRITICAL
  - Execution Environment
  - Input
  - Output Format

#### 3.5.2 Slim target

- **Slim line count:** 115 (from `regression/proposed_slim/analyst.yaml`).
- The slim draft retains: role, Binding Rules, results.json schema, Data Loading section, Execution Environment, Input, Output Format. Everything methodology-heavy moves to skills.

This is the largest reduction (−490 lines, −81%). Most of the harvest work was completed in V2.0.1 Phase 2c; remaining gaps are confirmed at 3b.25 implementation time.

#### 3.5.3 Content-to-skill mapping

| V1 content block | Migration target |
|---|---|
| Pilot Model Battery (LR, RF, XGBoost, ElasticNet, MLP, StackingEnsemble) | New `prediction-model-battery` skill |
| MLP Toggle | Same skill |
| Hyperparameter Tuning (per-model grids) | Same skill |
| Evaluation Protocol (AUC/RMSE primary, CIs, figures) | New `prediction-evaluation-protocol` skill OR extend the battery skill |
| Class Imbalance Handling (SMOTE + Ablation) | Existing `class-imbalance-smote-ablation` skill (likely exists in V2.0) |
| SHAP Interpretability Protocol | Existing `shap-explainer-selection` (V2.0; explicit V2.0.1 work) |
| Figure Outputs | `prediction-evaluation-protocol` skill (figures are part of evaluation) |
| Subgroup Analysis | Existing `subgroup-fairness-analysis` (V2.0) |
| High-Missingness Warning | Existing `missingness-tiered-protocol` (V2.0) — extend if needed |
| Model Quality Gate (AUC floor 0.60, R² floor 0.05) | New `prediction-model-quality-gate` skill OR extend an existing skill |
| Sensitivity Analysis | Existing `sensitivity-analysis-high-missingness` (V2.0) |
| Clustered SEs and ICC (ICC, clustered bootstrap, missing-school-IDs) | Existing `clustered-bootstrap-ci-and-icc` (V2.0) + `cluster-id-reconstruction-from-fingerprints` (V2.0) |
| Error Handling | Critical Universal Rules (stays in prompt) OR new minimal skill |
| Critical Rules — Never Violate | Critical Universal Rules section (stays in prompt) |

#### 3.5.4 New skills required

**Up to 3 new skills:**

1. **`prediction-model-battery`** (methodology, `applicable_stages: [Analyst]`, `applicable_task_types: [prediction]`, mandatory). The 6-model battery with hyperparameter grids, MLP toggle, stacking ensemble.
2. **`prediction-evaluation-protocol`** (methodology, `applicable_stages: [Analyst]`, `applicable_task_types: [prediction]`, mandatory). Primary/secondary metrics, CIs, figure outputs.
3. **`prediction-model-quality-gate`** (methodology, `applicable_stages: [Analyst]`, `applicable_task_types: [prediction]`, mandatory). AUC floor 0.60 + R² floor 0.05; SHAP-eligibility downstream effect.

Each absorbs a previously-prompt-bound block. The pattern matches 3b.18 (D1 encoding amendment): single-issue skill + content-to-skill mapping + rendered-prompt verification.

#### 3.5.5 Task-type branching handling

Base prompt (`analyst.yaml` — 605 V1 → 115 slim) handles prediction; `analyst_causal_soo.yaml` (97 lines, already slim) handles causal. Skill matching attaches the appropriate model/evaluation/sensitivity skills per task type.

The slim Analyst (115 lines) is larger than the causal variant (97 lines) because the base prompt has more universal scaffolding (e.g., the data-loading-CRITICAL section explaining how to read the outcome column header — applies in both task types but the causal variant doesn't need to handle classification metrics).

#### 3.5.6 Risks and open questions

- **Risk (highest):** **Skill-budget overflow.** With 10 mandatory + several recommended skills attached for causal_soo Analyst (per 3b.18 final inventory: G2, G3, G5, D1, M1, M2, M3, M4, M5, causal-data-engineer-contract), the prediction-task Analyst would add `prediction-model-battery`, `prediction-evaluation-protocol`, `prediction-model-quality-gate`, plus existing recommended skills (`shap-explainer-selection`, `class-imbalance-smote-ablation`, `subgroup-fairness-analysis`, `missingness-tiered-protocol`, `sensitivity-analysis-high-missingness`, `clustered-bootstrap-ci-and-icc`, `cluster-id-reconstruction-from-fingerprints`, `inner-cv-tuning-discipline`, `bootstrap-confidence-intervals`). The 3b.8 formatter cap (30K chars) must accommodate.
  - Mitigation: the slim Analyst prompt is 115 lines (~3K chars) vs V1's 605 lines (~16K chars), freeing ~13K chars of prompt budget. Skills will use that recovered budget.
  - Test: `tests/test_rendered_prompt_contains_all_mskills.py::TestBudgetSufficiencyForCausalSOO` already exists; ensure it still passes post-migration.
- **Risk (high):** SHAP-related fallback chains are subtle. The V1 prompt has a multi-step "if MLP best but KernelExplainer times out → fall back to next-best non-MLP" rule. The existing `shap-explainer-selection` skill needs verification that it preserves this fallback.
- **Risk (medium):** `Data Loading — CRITICAL` section in slim draft explicitly handles the `target_col = train_y_df.columns[0]` pattern (reading the actual outcome variable name from the CSV header). This is a defensive rule learned from prior failures. Slim form retains it; verify it stays.
- **Risk (medium):** Slim Analyst is the LAST migration phase. Cumulative risk from the prior four migrations may surface here. The 3b.25.5 re-run is the cumulative-validation gate.
- **Open question:** how does the F-3b19-M5-NO-INFERENCE backlog item interact with slim Analyst? The M5 inference path is governed by M5 SKILL.md (causal-forest-cate); slim Analyst doesn't change that. But if M5 SKILL.md amendment lands in a backlog phase before 3b.25, the interaction needs to be tested.

#### 3.5.7 Success criteria

- Rendered-prompt tests: all skill bodies reach the rendered Analyst prompt for both prediction and causal task types; V1 prediction-methodology content is GONE; budget-sufficiency test still passes.
- Output schema regression: `results.json` produced under slim Analyst matches V1 schema byte-identical for both task types.
- The 3b.25.5 re-run verifies cumulative migration end-to-end:
  - Pipeline completes single-attempt (no F-3b17 flakiness recurrence).
  - LSAR ≥ 5.5 (gate pass).
  - All preceding cleanup-arc closures hold: Q2.0 (DE contract), Q2.5 (DoWhy refuters), Q2.6 (encoding dispatch).
  - Target: LSAR Accept (≥ 6.5) or at minimum a known-pattern Outcome B.

#### 3.5.8 Phase 3b.25 implementation outcome (post-3b.25, executed within V4 Arc H)

**Status: MIGRATED.** Analyst 605 → 115 lines (81%, the largest). Corrections: ZERO new skills (the spec's three hypothesized skills all existed as `prediction-model-battery`, `prediction-evaluation-classification`/`-regression`, `prediction-quality-gate`). The real gap was CAP EXPOSURE, measured at Analyst/prediction: model-elasticnet, model-mlp, model-stacking-ensemble, and clustered-bootstrap-ci-and-icc bodies dropped under the 30K non-mandatory budget. Fixes: the authoritative hyperparameter grids (SPEC §4.3) harvested into `prediction-model-battery` (now mandatory — grids survive even when per-model elaboration drops); `clustered-bootstrap-ci-and-icc` retagged mandatory (ICC + clustered CIs are SPEC-mandated reporting). Data-Loading/target_col defensive rule retained in the slim body per §3.5.6. Budget-sufficiency suite green (36 tests); causal Analyst path untouched (variant + M-skills; no battery leak). Verification: 17 tests in test_v2_1_phase_3b25_analyst_slim.py; cumulative live validation at 3b.25.5.

---

## §4. Migration phase sequencing rationale

The recommended order is **OutlineAgent → ProblemFormulator → Critic → Writer → Analyst**. Rationale per agent:

### OutlineAgent first (3b.21)

- **Smallest content surface** (93 V1 → 25 slim). Minimal risk if something goes wrong; minimal payoff if successful (no LSAR-named gap depends on it). Perfect first test of the slim-application pattern.
- **No `_causal_soo` variant.** No task-type branching to think about; one prompt, one migration.
- **Missing `{{SKILLS}}` placeholder.** The wire-up step is part of the migration — testing that the placeholder substitution path works at runtime for an agent that has never had it.
- **Opportunistic backlog closure.** F-3b13-OUTLINE-AGENT-TYPE-ERROR may resolve naturally during the migration if the slim form constrains output shape better than V1.

### ProblemFormulator second (3b.22)

- **Already has a `_causal_soo` variant.** Task-type branching is a known-handled mechanism here; the migration is bounded to the base prompt.
- **3b.7 locked-spec invariants work** gives us pre-existing patterns for prompt-preserving PF behavior — useful precedent.
- **PF output is structured JSON.** Schema regression is straightforward to verify (compare `research_spec.json` field-by-field).
- **Downstream blast radius is bounded.** DataEngineer + Analyst + Critic + Writer all consume PF output; if PF output regresses, the regression is visible immediately at the next stage.

### Critic third (3b.23)

- **No `_causal_soo` variant.** Decision point: introduce one or rely on task-type-aware skill matching. Recommended Option A (skill matching) per §3.3.5.
- **Deterministic verdict-evaluator integration.** The 3b.10 evaluator is orchestrator-side, not prompt-side. Slim Critic prompt cannot break verdict logic.
- **F-3b7-CRITIC-V1-LEAKAGE-VISIBLE** is naturally addressed by slim migration — many V1-prompt-bound prediction-task rules (e.g., AUC > 0.95 flag) move to skills and become task-type-conditional.
- **Mid-arc re-run after 3b.23 (3b.23.5).** Three of five agents slimmed; cumulative verification before the larger Writer + Analyst migrations.

### Writer fourth (3b.24)

- **More complex output (full LaTeX paper).** Several edge cases (template fill, table/figure environment, citations, UNVERIFIED banner). Wait until the slim pattern is proven before tackling.
- **Multiple existing skill cross-references** (G2, G5 interpretation guidance landed in 3b.14 / 3b.16 amendments). Writer is the agent where existing skill content most heavily affects output already.
- **No re-run** until 3b.25.5 (the cumulative-end-of-arc gate). Writer migration's effect compounds with Analyst's; assess them together.

### Analyst last (3b.25)

- **Largest content surface** (605 V1 → 115 slim, −81%).
- **Most failure modes have been Analyst-domain** (F-3b9-ANALYST-CODEGEN-CRASH, F-3b11-DE-MISSING-TREATMENT-COLUMN downstream effects, F-3b13-M5-CLASSIFIER-CONTINUOUS-TARGET, F-3b15-DE-CONTINUOUS-AS-CATEGORICAL effects on propensity model, F-3b11-M4-SE-IMPLAUSIBLE).
- **10 mandatory skills attach** for causal_soo per 3b.18 final inventory. Skill-budget overflow is the highest-risk single concern.
- **The 3b.25.5 re-run is the cumulative-validation gate.** Five agents slimmed; the cumulative diff vs 3b.19 baseline is the broadest single-variable comparison the project has run.

### Sequencing exceptions

This recommended order can be revised by implementation-phase evidence:

- If 3b.21 OutlineAgent surfaces issues that change the slim-application pattern (e.g., `{{SKILLS}}` placeholder substitution has runtime quirks not seen in V2.0.1 DE), pause subsequent phases until the pattern is corrected.
- If 3b.22 ProblemFormulator surfaces F-3b11-PRECRITIC-PREDICTION-CARRYOVER interactions (Critic flagging prediction-task fields missing in PF research_spec), the carryover backlog item may need to be addressed before 3b.23 Critic to avoid confounded Critic verdicts at re-run time.
- If skill-budget overflow surfaces earlier than 3b.25 (e.g., 3b.24 Writer with many writing-layer skills), the cap may need adjustment, affecting 3b.25 timing.

---

## §5. Re-run interleaving strategy

Three re-runs across the 8-phase implementation arc:

### 3b.21.5 — Post-OutlineAgent re-run

- **Trigger:** after 3b.21 OutlineAgent slim is committed.
- **Differential vs:** 3b.19 (the cleanup-arc-close baseline).
- **Single variable:** OutlineAgent slim + `{{SKILLS}}` placeholder wire-up.
- **Primary question (Q2.7):** does the OutlineAgent produce a valid paper outline under slim form? Does the Writer consume it without falling back to v1 template?
- **Secondary questions:** does F-3b13-OUTLINE-AGENT-TYPE-ERROR recur, resolve, or change shape? Cumulative LSAR Δ vs 3b.19.

### 3b.23.5 — Mid-arc re-run

- **Trigger:** after 3b.23 Critic slim is committed.
- **Differential vs:** 3b.21.5 (the post-OutlineAgent baseline).
- **Single variable (jointly):** PF slim (3b.22) + Critic slim (3b.23). Two amendments; this re-run validates them together because PF slim doesn't have its own re-run.
- **Primary question:** cumulative effect of PF + Critic slim on cycle-0 / cycle-1 quality scores and revision dynamics. Verdict-evaluator integration with slim Critic must hold.
- **Secondary:** any LSAR-reported regressions attributable to the prompt simplifications (e.g., "research question lacks gap-driven framing" if `research-question-design` skill under-specifies).

### 3b.25.5 — Final post-Analyst re-run

- **Trigger:** after 3b.25 Analyst slim is committed.
- **Differential vs:** 3b.23.5 (the mid-arc baseline).
- **Single variable (jointly):** Writer slim (3b.24) + Analyst slim (3b.25).
- **Cumulative differential vs 3b.19:** the entire V2.1 migration validation.
- **Primary question:** does the fully-slim pipeline achieve LSAR ≥ 5.5 with no new failure modes that the V1 monolithic prompts wouldn't have caught? Target: LSAR Accept (≥ 6.5) or known-pattern Outcome B.

Each re-run follows the Pattern B → live LLM-call structure of 3b.13 / 3b.15 / 3b.17 / 3b.19. Same locked spec (`runs/fixtures/spec_x1mtheff_x4college.json`), same config (`runs/configs/smoketest_3b11.yaml`), same provider mix.

---

## §6. New skill backlog

Inventory of new skills surfaced by §3 per-agent specs:

| New skill | Layer | Agent stages | Task types | Severity | Surfaced in |
|---|---|---|---|---|---|
| `research-question-design` | methodology | PF | prediction (or task-agnostic) | mandatory | §3.2 (PF) |
| `prediction-feasibility-floor` | methodology | PF | prediction | mandatory | §3.2 (PF) |
| `critic-checklist-problem-formulation-prediction` | methodology | Critic | prediction | mandatory | §3.3 (Critic) |
| `critic-checklist-data-preparation` | methodology | Critic | task-agnostic | mandatory | §3.3 (Critic) |
| `critic-checklist-analysis-prediction` | methodology | Critic | prediction | mandatory | §3.3 (Critic) |
| `critic-checklist-substantive-validity` | methodology | Critic | task-agnostic | mandatory | §3.3 (Critic) |
| `latex-table-discipline` | writing | Writer | task-agnostic | mandatory | §3.4 (Writer) |
| `paper-section-content-prediction` | writing | Writer | prediction | mandatory | §3.4 (Writer) |
| `hsls09-cluster-reconstruction-paragraph` | writing | Writer | task-agnostic (hsls09) | recommended | §3.4 (Writer) |
| `prediction-model-battery` | methodology | Analyst | prediction | mandatory | §3.5 (Analyst) |
| `prediction-evaluation-protocol` | methodology | Analyst | prediction | mandatory | §3.5 (Analyst) |
| `prediction-model-quality-gate` | methodology | Analyst | prediction | mandatory | §3.5 (Analyst) |

**Up to 12 new skills.** Authoring strategy:

- **Single-issue Pattern A authoring per skill.** Same cadence as 3b.12 / 3b.14 / 3b.16 / 3b.18 amendments: pre-amendment investigation + draft + rendered-prompt verification.
- **Authored alongside migration phases.** Rather than authoring all 12 skills up-front, each migration phase authors the skills it needs as part of that phase (or in a single-issue sub-phase before it). The 3b.18 amendment is a precedent — D1 was extended in 3b.18 because the encoding rule was specifically needed for the migration's pipeline-stability story.
- **Investigation must verify existing skill coverage.** Several proposed new skills may already exist under slightly different names; 3b.21+ phase investigations clarify what's actually new vs. already covered.

If §6's count is high (12 new skills feels heavy), that's a real budget concern for the V2.1 implementation arc. Each new skill is a single-issue commit; 12 of them across 5 migration phases is ~2–3 skill commits per migration. The total commit count for V2.1 implementation grows from 5 (just migrations) to ~17 (migrations + new skills + re-runs).

---

## §7. Risk register

### R1. Skill coverage gaps (high probability, high impact)

**Description:** A V1 prompt content block has no existing skill coverage AND no proposed new skill. Migration cannot proceed without absorbing the rule somewhere.

**Mitigation:**
- Each migration phase's pre-amendment investigation enumerates V1 content → skill mapping rigorously (per §3 templates).
- New skills authored in single-issue sub-phases BEFORE the migration commit.
- If a gap is discovered mid-migration, the migration phase pauses and forks into "harvest the rule, then migrate."

**Surfaces in implementation:** §3 spec is investigation-grounded but Sub-wave 0 did not exhaustively read every V2.0 skill body. 3b.22's PF investigation is the first place where this risk concretely materializes.

### R2. Task-type branching elimination infeasibility (resolved)

**Description:** The hand-off recommended eliminating per-task-type variants; investigation found this requires unifying structurally-different output schemas, which is out of scope.

**Mitigation:** §2.4 documents the decision to RETAIN task-type branching at the prompt level. Open question 1 in §8 tracks future schema unification as a V2.2+ candidate.

**Surfaces in implementation:** No — resolved at design time.

### R3. Output schema preservation (medium probability, very high impact)

**Description:** Slim migration must NOT change agent output schemas. Downstream agents (orchestrator, Critic, verdict-evaluator) depend on field names + types. If a slim migration drops a schema field, downstream failures cascade.

**Mitigation:**
- Schema regression tests per agent: compare `research_spec.json` / `data_report.json` / `results.json` / `review_report.json` / `paper.tex` produced under slim form vs V1 form, field-by-field.
- The slim drafts in `regression/proposed_slim/` were designed against the V1 schemas; staged drafts should already preserve schemas. Implementation phases verify.

### R4. Skill-budget overflow (medium probability, high impact)

**Description:** The 3b.8 formatter `max_chars=30000` cap holds. Slim Analyst plus 10+ mandatory skills could overflow on prediction-task runs (more skills attach than on causal). The 3b.18 rendered-prompt budget-sufficiency test currently passes; post-migration it must continue to pass.

**Mitigation:**
- Slim prompts free 12–13K chars vs V1 (Analyst alone: 605 lines → 115 = ~12K char reduction). The recovered budget accommodates skill growth.
- Each migration phase verifies `TestBudgetSufficiencyForCausalSOO::test_no_drops_for_causal_soo_analyst_at_default_budget` still passes.
- If overflow surfaces, the cap can be bumped (single-issue phase) — but the slim Analyst's reduced prompt size should make this unnecessary.

### R5. LSAR-score regression post-migration (medium probability, medium impact)

**Description:** Slim migration is architectural; it should not change methodology. If LSAR scores drop post-migration, that signals content was lost in translation — the harvest-into-skill step left something behind.

**Mitigation:**
- Re-run phases (3b.21.5, 3b.23.5, 3b.25.5) use the same single-variable Δ pattern that successfully attributed +1.0 (3b.13) and −1.0 (3b.17) to single changes. LSAR-score changes attributable to migration are diagnosable.
- The 3b.19 backlog (subgroup, M5, M4 SE, pre-critic) is tracked SEPARATELY from migration effects — backlog items don't surface as "migration broke this."

### R6. 3b.19 backlog items surfacing during migration (high probability, low impact)

**Description:** F-3b11-SUBGROUP-NOT-IMPLEMENTED, F-3b19-M5-NO-INFERENCE, F-3b11-M4-SE-IMPLAUSIBLE may become MORE visible as the pipeline stabilizes. Each re-run's LSAR review may flag them differently as previously-masked-now-visible concerns.

**Mitigation:**
- Track separately from migration phase outcomes; don't bundle backlog fixes into migration phases (single-issue discipline).
- Re-run reports explicitly distinguish migration-attributable changes from backlog-attributable changes.
- If a backlog item becomes pipeline-blocking during V2.1, it gets a dedicated single-issue phase (analogous to 3b.18's relationship to 3b.17's surfaced F-3b15 concern).

### R7. Cascading prompt edits during the migration arc (low probability, high impact)

**Description:** Slim migration is "one agent at a time," but an agent's prompt edits may affect downstream agents (e.g., slim PF produces slightly differently-shaped `research_spec.json`, which propagates to DE/Analyst/Critic/Writer). Cascading effects are subtle.

**Mitigation:**
- Schema regression tests (R3 mitigation) catch downstream-blast.
- The 3b.21.5 / 3b.23.5 / 3b.25.5 re-run phases catch end-to-end cascade effects.
- Sequencing (OutlineAgent first, Analyst last) puts the lowest-cascade agents first.

### R8. F-3b17-DE-ENGINEERING-CYCLE0-FLAKINESS during re-runs (medium probability, low impact)

**Description:** The 3b.17 multi-attempt-needed pattern may recur on re-run phases. Each re-run's wall-clock is unpredictable.

**Mitigation:**
- Allow multiple attempts per re-run (3b.17 documented this is acceptable).
- Preserve aborted-attempt artifacts (`output_attempt1_*/`).
- If flakiness becomes blocking, an orchestrator-side guardrail (post-DE pre-flight check) is a separate single-issue phase.

---

## §8. Open questions

### Q1. Task-type variant schema unification (V2.2+ candidate)

The decision to RETAIN per-task-type variants (§2.4) is for V2.1. A future architectural effort could:
- Define a unified output schema that subsumes prediction + causal cases (e.g., `results.json` has a top-level `task_type` discriminator and per-case sub-schemas).
- Migrate all agents to single-prompt + skill-matching.
- Delete `_causal_soo.yaml` variants.

This is V2.2+ scope. The current design retains them.

### Q2. `dispatch_*` methods

Per the hand-off and CLAUDE.md, `dispatch_*` methods (deferred since 3b.4) may make sense in V2.1 (agents dispatch to method-specific code paths) or may not. Sub-wave 0 did not investigate this in depth. The scope-doc does not pre-commit; implementation phases can introduce dispatch_* if the slim migration surfaces a clear need, or defer further.

### Q3. `data_registry/task_templates/causal_soo.yaml`

A task-template file for causal_soo has been deferred since 3b.4. The current orchestrator infers task-type behavior from `task_type` in the research_spec + skill matching. Whether an explicit task-template YAML is needed is undecided. Implementation phases can author it if a clear gap emerges; otherwise defer.

### Q4. Variant file disposition

If at some point (V2.2+ or otherwise) task-type branching IS eliminated, what happens to the existing `_causal_soo.yaml` variants?
- **Delete:** lose history but minimize confusion.
- **Archive:** move to `agent_prompts/archive/` for traceability.
- **Repurpose as test fixtures:** the variant slim drafts are useful baselines for rendered-prompt tests in implementation phases.

The scope doc doesn't pre-commit; the decision belongs to whichever phase eliminates the variants.

### Q5. Branch strategy

Should V2.1 migration be branched off `phase-3-causal-inference` to a new `phase-4-v2.1-slim` branch, or continued on the same branch?
- **New branch:** cleaner history; V2.1 work is architecturally distinct from V3.0 work.
- **Same branch:** simpler git workflow; V2.1 is a continuation of the 3b series; no merge conflicts.

The 3b.20 commit lands the scope doc on `phase-3-causal-inference`. The 3b.21+ migration phases can branch later if convenient.

### Q6. Investigation memo vs. inline §3 evidence

§3 per-agent specs cite content-to-skill mappings; some of those skills may not exist under the proposed names. Should 3b.21 implementation phases author an "investigation memo" enumerating actual vs. expected skill names before each migration, or fold the investigation into the migration phase's pre-authoring step?

The hand-off prescribes the latter (sub-wave 0 investigation per phase). The scope doc trusts that.

### Q7. F-3b19-CYCLE1-SUBGROUP-NON-RECOVERY interaction with PF + Analyst slim migrations

The 3b.19 backlog item about cycle-1 subgroup non-recovery is rooted in `subgroup-fairness-analysis` skill being non-prescriptive. If slim PF or slim Analyst migration depends on `subgroup-fairness-analysis` being prescriptive, the dependency must be resolved first.

This is a sequencing question for 3b.22 / 3b.25 implementation. The scope doc surfaces it; the implementer resolves it.

---

## §9. Cross-references

- **V3.0 skill spec:** `docs/v3_0_causal_skill_specification.md`. Particularly §4.2 (attachment table), §10–13 (Phase 3b.8 / 3b.12 / 3b.14 / 3b.16 / 3b.18 amendments).
- **V2.0.1 DE slim reference:** commit `3ba93a6` (`V2.0.1: wire {{SKILLS}} placeholder into V1 monolithic prompts`), commit `092c7b3` (no-redundant-composites skill), commit `5cda588` (V2.0 Phase 2c BLOCKED at Checkpoint 3 — DE slim diagnosis), commit `c52370c` (V2.0 Phase 2c BLOCKED at Checkpoint 4b — DataEngineer helper-import defect). The history of V2.0.1 slim work is in commits dated 2026-04-22 through 2026-04-26.
- **Slim drafts staged:** `regression/proposed_slim/<agent>.yaml`.
- **CLAUDE.md V2.1 runbook:** the project memory note describing the V2.1 slim migration runbook (this scope doc supersedes it).
- **3b series amendments (in audit doc):**
  - 3b.6: D1 encoded-column lookup + M4 cluster-aware IF + subgroup causal mode
  - 3b.8: M1–M5 promoted to mandatory tier
  - 3b.12: `causal-data-engineer-contract` skill
  - 3b.14: G5 DoWhy refuter invocation
  - 3b.16: G5 NetworkX-DiGraph refinement
  - 3b.18: D1 encoding-type discipline
- **3b.19 closure report:** `runs/v3_0_smoketest_mtheff_college_20260502_3b19/REPORT.md`. Particularly the §14 forward-backlog section that motivates this scope doc.
- **F-IDs cataloged:** F-3b11-DE-MISSING-TREATMENT-COLUMN (closed in 3b.13), F-3b13-DOWHY-REFUTERS-GRAPH-FORMAT (closed in 3b.15), F-3b15-DOWHY-PYGRAPHVIZ-DEPENDENCY-GAP (closed in 3b.17), F-3b15-DE-CONTINUOUS-AS-CATEGORICAL (closed in 3b.19), F-3b11-PRECRITIC-PREDICTION-CARRYOVER (V2.1 backlog), F-3b19-M5-NO-INFERENCE (V2.1 backlog), F-3b11-M4-SE-IMPLAUSIBLE (V2.1 backlog), F-3b11-SUBGROUP-NOT-IMPLEMENTED (V2.1 backlog), F-3b13-OUTLINE-AGENT-TYPE-ERROR (may resolve naturally in 3b.21), F-3b17-DE-ENGINEERING-CYCLE0-FLAKINESS (watch list).

---

## §10. Acceptance and approval

This scope doc is a design artifact. The 3b.21 implementation phase MUST NOT begin until this doc is reviewed and approved by the project lead. Approval is signaled by:

1. **PR merge** of the 3b.20 commit landing this doc on `phase-3-causal-inference`, OR
2. **Explicit approval message** from the project lead in the conversation thread following the 3b.20 hand-off completion.

Pre-approval discussion is encouraged on:

- Task-type branching elimination (§2.4 decision RETAIN — the hand-off recommended eliminate; the investigation found infeasible).
- New skills count (§6 — 12 new skills is heavy; some may already exist under different names).
- Sequencing rationale (§4 — order is OutlineAgent → PF → Critic → Writer → Analyst; alternative orderings have been considered).
- Branch strategy (§8 Q5).

If approval is conditional on revisions, sub-wave 2 of 3b.20 applies them and re-submits.

After approval: 3b.21 hand-off can be authored. The hand-off should reference this scope doc's §3.1 (OutlineAgent spec) as the implementation contract.

---

*Document version 1.0, Phase 3b.20 deliverable. Authored 2026-05-15 against `phase-3-causal-inference` @ `e443c8a`. Supersedes the V2.1 slim runbook note in `CLAUDE.md`.*
