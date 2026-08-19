# task-type skills

Research procedures coupled to a single task type. Each subdirectory is one
skill: `task-type/<skill-name>/SKILL.md`.

A task-type skill encodes workflow, model/estimator choice, and validation
logic specific to a research style — e.g. supervised prediction, causal
inference (TMLE), fairness audit. It declares
`applicable_task_types: [prediction]` (or whatever applies) so the matcher
filters it out for other task types.

Empty until Phase 2. Planned priority-1 skills: `prediction-workflow-overview`,
`prediction-model-battery`, `prediction-evaluation-classification`,
`prediction-evaluation-regression`, `prediction-quality-gate`,
`prediction-research-question-design`, `prediction-critic-checklist`. See
[`../../audit/skill_candidates.csv`](../../audit/skill_candidates.csv).
