---
name: hsls09-variable-registry
layer: dataset
description: Index to the bundled HSLS:09 variable registry YAML; explains schema, missing codes, and how to load it.
trigger_keywords:
  - hsls
  - hsls09
  - registry
  - variable
  - variables
  - outcome
  - outcomes
  - predictor
  - predictors
applicable_task_types: []
applicable_datasets:
  - hsls09_public
applicable_stages:
  - ProblemFormulator
  - DataEngineer
  - Analyst
  - Critic
  - Writer
priority: 1
references_skills: []
resources:
  - variable_registry.yaml
version: "1.0"
---

# HSLS:09 Variable Registry

The full HSLS:09 public-use variable registry is bundled alongside this
SKILL.md as `variable_registry.yaml` (~1,800 lines, byte-identical copy
of `data_registry/datasets/hsls09_public.yaml`). This skill is a short
index that explains how to load it and what it contains; the YAML itself
is the source of truth.

## Loading the resource

```python
import yaml

# `skill` here is the Skill returned by SkillRegistry.get(...).
registry_path = next(
    p for p in skill.resource_paths if p.name == "variable_registry.yaml"
)
registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
```

In agent code, the `RegistryLoader` class (`src/registry.py`) wraps the
same YAML with helpers like `get_variable(name)`, `get_outcomes()`,
`get_predictors(category)`, `validate_temporal_order(pred, out)`,
`is_protected_attribute(name)`, and `is_excluded(name, category)`.

## Top-level schema

The registry is a YAML mapping with these keys:

| Key | Contents |
|---|---|
| `name`, `full_name`, `source`, `documentation_url`, `access` | Dataset metadata. |
| `data_structure: multilevel` + `levels: {student: 23503, school: 944}` | Sample sizes; reminds you that SCH_ID is suppressed and clustering must be reconstructed. |
| `multilevel_note` | Prose noting school IDs are suppressed and design-based variance estimation requires the restricted-use file. |
| `waves` | Five waves: `base_year` (2009, X1), `first_follow_up` (2012, X2), `second_follow_up` (2013, X3), `update_panel` (2016, X4), `postsecondary_records` (2017, X5). Each wave has a `prefix`, `year`, `grade`, `label`. |
| `temporal_order` | Ordered list `[base_year, first_follow_up, second_follow_up, update_panel, postsecondary_records]`. Drives every leakage check. |
| `tier3_exclusion_rules` | Patterns that disqualify variables from being predictors (weights `^W[0-9]`, BRR/replicate, `_IM/_FLG/FLAG/_I` suffixes, IDs like `STU_ID`, `SCH_ID`). |
| `tier: 1` + `variables` | Curated outcomes and predictors with educational rationale. |
| `tier2_config` | Pointer to auto-generated Tier 2 file with looser usage policy. |
| `missingness` | Mechanism assumption + sentinel codes/labels + recommended methods + MNAR candidates. |
| `common_pitfalls` | Named pitfalls (`temporal_leakage`, `same_wave_target_leakage`, `public_use_suppression`, `school_level_misinterpretation`, `protected_attribute_misuse`) each with severity. |
| `canonical_research_questions` | Example prediction / fairness / policy questions to inspire research-spec generation. |

## `variables` section

```yaml
variables:
  outcomes:        # list of outcome variables, e.g. X3TGPAMAT, X4EVRATNDCLG, dropout_derived
    - name: ...
      label: ...
      type: continuous|binary|categorical
      wave: ...
      pct_missing: ...
      range: [...] or categories: [...]
      codebook_codes: {...}
  predictors:      # mapping of category → list of variables
    demographic: [...]
    family: [...]
    academic: [...]
    math_attitudes: [...]
    academic_followup: [...]
    course_taking: [...]
    school_level: [...]
    teacher: [...]
    behavioral: [...]
    postsecondary: [...]
    x5_records: [...]
```

Each predictor entry has at minimum `name`, `label`, `type`, `wave`,
`pct_missing`, and either `range` (continuous) or `categories` +
`codebook_codes` (categorical/binary). Some entries also have
`protected_attribute: true` (drives subgroup analysis), `note`,
`temporal_warning` (later-wave warning), or `derived: true`.

## NCES missing codes (treat all of these as missing)

Numeric sentinels: `-1` (legitimate skip), `-4` (nonrespondent),
`-5` (data suppressed), `-6` (component not applicable),
`-7` (item legitimate skip / NA), `-8` (unit non-response),
`-9` (missing).

Text labels (the public-use CSV stores labels, not numeric codes — see
`hsls09-csv-format-quirks`): `"Missing"`, `"Unit non-response"`,
`"Item legitimate skip/NA"`, `"Data suppressed"`,
`"Component not applicable"`, `"Don't know"`.

The full sentinel list is in `registry["missingness"]["sentinel_codes_or_labels"]`.

## Related skills

- `hsls09-csv-format-quirks` — public-use CSV stores text labels; do
  not call `pd.to_numeric(errors='coerce')` on categorical columns.
- `hsls09-temporal-ordering` — predictor wave must be strictly less
  than outcome wave.
- `hsls09-missing-codes` — operational use of the sentinel list.
- `hsls09-tier3-exclusions` — automatically excluded variable patterns.
- `hsls09-protected-attributes` (implicit in registry; see
  `protected_attribute` flags on `X1SEX`, `X1RACE`, `X1SES`,
  `X1SES_U`, `X1SESQ5`).
- `hsls09-structural-mnar-outcomes` — postsecondary outcome retention
  carve-out.
- `hsls09-school-fingerprints` (+ methodology
  `cluster-id-reconstruction-from-fingerprints`) — recover suppressed
  SCH_ID via fingerprint variables.

## Source provenance

Canonical source: `data_registry/datasets/hsls09_public.yaml`, bundled
here byte-identical as `variable_registry.yaml`. Phase 2c may delete
the original after agents are wired through this skill.
