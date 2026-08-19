# dataset skills

Dataset-specific knowledge: missing-code conventions, wave/temporal
ordering, weight variables, registry rules, format quirks. Each subdirectory
is one skill: `dataset/<skill-name>/SKILL.md`.

A dataset skill declares `applicable_datasets: [hsls09_public]` (or whatever
applies) so the matcher filters it out for other datasets. Large reference
tables (e.g. the HSLS variable registry YAML) live alongside SKILL.md as
bundled resources rather than inlined into the body — see
[`../README.md`](../README.md) section "Resource files".

Empty until Phase 2. Planned priority-1 skills: `hsls09-variable-registry`,
`hsls09-csv-format-quirks`, `hsls09-missing-codes`, `hsls09-temporal-ordering`.
See [`../../audit/skill_candidates.csv`](../../audit/skill_candidates.csv).
