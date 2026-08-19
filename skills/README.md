# EDM-ARS Skill Library

This directory holds the V2.0 skill library — composable, reusable units of
knowledge that the multi-agent pipeline injects into agent system prompts at
runtime. Real skills are extracted from the existing prompts during Phase 2.
For now, the directory exists as a structural skeleton.

The audit that motivated this layout lives at
[`audit/AUDIT_REPORT.md`](../audit/AUDIT_REPORT.md), and the candidate
inventory is [`audit/skill_candidates.csv`](../audit/skill_candidates.csv).

## Layers

A skill belongs to exactly one layer. Layer drives matching semantics — the
matcher applies hard filters (stage / task type / dataset) and a per-layer
top-k cap. Layers also dictate where a SKILL.md file lives on disk.

| Layer | What it encodes | Coupled to |
|---|---|---|
| `task-type` | Research procedure: workflow, model battery, evaluation logic. | One task type (e.g. `prediction`). |
| `dataset` | Dataset-specific quirks: missing codes, wave ordering, weight variables, registry rules. | One dataset (e.g. `hsls09_public`). |
| `methodology` | Crosscutting technique reusable across task types and datasets (e.g. MICE, SHAP variant selection, cluster bootstrap CI). | None — applies broadly. |
| `writing` | Paper output shape: template structure, citation style, limitations prose, narrative archetype. | None — applies broadly. |

`core-role` content (agent role definition, JSON I/O contract, universal
constraints like "no network") stays in `agent_prompts/` and is **not** a
skill. Slimming the existing agent prompts to core-role only is also a
Phase 2 deliverable.

## Directory convention

```
skills/
├── task-type/<name>/SKILL.md      (+ optional bundled resources)
├── dataset/<name>/SKILL.md        (+ optional bundled resources)
├── methodology/<name>/SKILL.md    (+ optional bundled resources)
└── writing/<name>/SKILL.md        (+ optional bundled resources)
```

The loader (`src/skills/loader.py`) walks this tree and infers `layer` from
the parent directory name. If a SKILL.md frontmatter declares a different
`layer`, the loader logs a warning and prefers the frontmatter value.

## SKILL.md frontmatter schema

```yaml
---
name: kebab-case-name           # required, unique within a layer
layer: methodology              # required, one of {task-type, dataset, methodology, writing}
description: One sentence.      # required; used by the matcher
trigger_keywords:               # optional; keyword overlap boosts match score
  - shap
  - feature-importance
applicable_task_types: []       # optional; empty = all task types
applicable_datasets: []         # optional; empty = all datasets
applicable_stages: []           # optional; empty = all agents
priority: 5                     # optional; lower = higher rank (default 5)
references_skills: []           # optional; other skills composed alongside this one
resources: []                   # optional; bundled non-SKILL.md filenames in the same dir
version: "1.0"                  # optional; default "1.0"
rule_severity: recommended      # optional; one of {mandatory, recommended, reference}
---

# Skill body (markdown)

The body after the closing `---` is the actual prompt content that gets
concatenated into the agent's system prompt at match time.
```

### When to tag a skill `rule_severity: mandatory`

A skill is **mandatory** if violating it produces output that is structurally
invalid, regardless of whether the pipeline crashes. Specifically, mandatory iff
at least one of:

1. **Crash-risk** — violation causes a runtime exception (e.g. `pd.qcut` on
   non-unique edges; `import` of a module not present).
2. **Silent corruption** — violation produces output that looks valid but
   contains incorrect numbers, columns, or structure (e.g. one-hot encoding a
   continuous variable, producing 20K columns; coercing categorical labels to
   NaN via `pd.to_numeric(errors='coerce')`).
3. **Structural incompleteness** — violation produces output missing a required
   section that downstream consumers depend on (e.g. empty
   `subgroup_performance` when `subgroup_analyses` was specified; missing
   `data_report.school_reconstruction` block).
4. **Methodological invalidity** — violation produces results that look correct
   but are statistically wrong (e.g. fitting a scaler on the full dataset
   before train/test split → leakage; computing clustering ICC with <50
   clusters).

A skill is `recommended` (default) if violating it produces *worse* output but
the output is still structurally valid and methodologically defensible.

A skill is `reference` if it provides background context the agent may consult
but is not expected to follow line-by-line.

**Do not tag skills mandatory just because they are useful.** Reserve mandatory
for rules whose violation invalidates the output. Mandatory rules render with a
strong header + binding-rules banner and bypass the per-layer cap in the
matcher (so they always reach the agent), so over-tagging dilutes the signal.
The current registry has 6 mandatory tags; aim to keep the total under ~10.

## Composition via `references_skills`

A skill can name other skills in `references_skills`. At match time, the
composer walks references transitively, deduplicates, and breaks cycles
(logging a warning). Output order is referrer-first, then references in
declaration order.

This is how the per-family model battery composes (Decision 9 from the
Phase 0 audit): a `prediction-model-battery` meta-skill `references_skills`
each `model-*` skill so all six bodies come along together. It is also how
the HSLS school-cluster split (Decision 6) works: the dataset skill
`hsls09-school-fingerprint-vars` references the generic methodology skill
`cluster-id-reconstruction-from-fingerprints`.

## Resource files (Decision 5)

Large reference data (e.g. the 1,700-line HSLS variable registry) does not
belong in a SKILL.md body. Instead, list it under `resources:` and place the
file alongside SKILL.md in the same directory. `Skill.resource_paths` returns
the absolute paths so agents can load them on demand:

```python
skill = registry.get("hsls09-variable-registry")
for path in skill.resource_paths:
    data = yaml.safe_load(path.read_text())
```

## How agents will use this (Phase 2)

Phase 1 ships only the registry infrastructure. Phase 2 will wire it into
agent prompts using a single placeholder:

```python
from src.skills import SkillRegistry

registry = SkillRegistry(Path("skills"))
prompt_skills = registry.format_for_prompt(
    stage="Analyst",
    task_type="prediction",
    dataset="hsls09_public",
    context=research_question_text,
)
system_prompt = base_role_prompt.replace("{{SKILLS}}", prompt_skills)
```

The loader, matcher, composer, and registry facade all live in
[`src/skills/`](../src/skills/). See `tests/test_skill_registry.py` for
worked examples of every feature.

## Phase 2 status

Empty. Real skills are extracted from `agent_prompts/`, `data_registry/`,
and `templates/` during Phase 2. See `audit/skill_candidates.csv` for the
39-row inventory and the priority-1 skills slated to land first.
