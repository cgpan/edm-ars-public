---
name: prediction-fixture
layer: task-type
description: Fixture skill for the prediction task type; references shap-fixture.
applicable_task_types:
  - prediction
applicable_stages:
  - DataEngineer
  - Analyst
priority: 2
references_skills:
  - shap-fixture
trigger_keywords:
  - prediction
  - supervised
---

# Prediction Workflow (Fixture)

End-to-end supervised ML pipeline:

1. Load data
2. Train models
3. Evaluate on a held-out test set
4. Interpret with SHAP

This is a fixture; the real prediction skill lives elsewhere.
