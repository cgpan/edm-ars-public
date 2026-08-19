# methodology skills

Crosscutting techniques reusable across task types and datasets. Each
subdirectory is one skill: `methodology/<skill-name>/SKILL.md`.

A methodology skill typically declares neither `applicable_task_types` nor
`applicable_datasets` (so the matcher returns it for any caller), and uses
`trigger_keywords` to rank itself against a free-text context.

Empty until Phase 2. Planned priority-1 skills: `missingness-tiered-protocol`,
`shap-explainer-selection`, `school-aware-train-test-split`,
`bootstrap-confidence-intervals`, `subgroup-fairness-analysis`,
`inner-cv-tuning-discipline`. See
[`../../audit/skill_candidates.csv`](../../audit/skill_candidates.csv).
