---
name: hsls09-tier3-exclusions
layer: dataset
description: Variable patterns that are automatically excluded from the HSLS:09 candidate predictor pool (weights, flags, IDs).
trigger_keywords:
  - hsls
  - hsls09
  - tier3
  - exclusion
  - exclusions
  - weight
  - weights
  - flag
  - flags
applicable_task_types: []
applicable_datasets:
  - hsls09_public
applicable_stages:
  - ProblemFormulator
  - DataEngineer
priority: 2
references_skills: []
resources: []
version: "1.0"
rule_severity: mandatory
---

# HSLS:09 Tier-3 Exclusions

The registry's `tier3_exclusion_rules` define the variables that the
ProblemFormulator must NOT include as predictors and that the
DataEngineer should not load as features. These are administrative
variables, design weights, and processing flags that have no
substantive predictive meaning and would either leak information
(processing flags carry response history) or break models (weights are
not features).

## Pattern-based exclusions

### Prefix patterns (regex)

| Pattern | Matches |
|---|---|
| `^W[0-9]` | All survey/replicate weights: `W1STUDENT`, `W2W1STU`, `W4W1W2W3STU`, ... |
| `^BRR` | Balanced repeated replicate variance estimators |

### Suffix patterns (regex)

| Pattern | Matches |
|---|---|
| `_IM$` | Imputation flags |
| `_FLG$` | Processing flags (alt spelling) |
| `FLAG$` | Processing flags |
| `_I$` | Imputation indicators |

## Exact-match exclusions

```
STU_ID, SCH_ID, STRAT_ID, PSU, psu,
X1NCESID, X2NCESID,
X1SQSTAT, X1PQSTAT, X1TMQSTAT, X1TSQSTAT, X1AQSTAT, X1CQSTAT,
X2SQSTAT, X2PQSTAT, X2AQSTAT, X2CQSTAT,
X3SQSTAT, X4SQSTAT
```

These are administrative IDs and questionnaire-status fields
(`*QSTAT`). They encode response history (e.g., whether a student
completed the parent questionnaire), which can leak information about
the outcome.

## Category-label exclusions

The registry tags some variables with category labels that
auto-exclude them:

```
administrative, sampling, weight, imputation_flag, processing_flag,
interviewer_variable, logical_inference_flag
```

Variables in `tier3_exclusion_rules.category_labels` are dropped from
the candidate predictor pool regardless of their name.

## Operational use

Use `RegistryLoader.is_excluded(name, category_label=None)`:

```python
from src.registry import RegistryLoader

loader = RegistryLoader("data_registry/datasets/hsls09_public.yaml")
if loader.is_excluded("W1STUDENT"):
    ...  # skip; matches ^W[0-9]
```

The loader applies all four kinds of rules (exact, prefix, suffix,
category label) in turn and returns `True` if any matches.

## Why this matters

Without these exclusions:

- **Weight variables** (`W*`) can dominate tree-based feature importance
  because they correlate with sampling design quirks; including them
  produces confounded SHAP rankings.
- **Imputation/processing flags** (`*_IM`, `*_FLG`, `*FLAG`) carry
  response history and can leak the outcome (e.g., a flag for
  imputed-via-prior-wave directly encodes whether a follow-up
  response exists).
- **IDs and PSUs** are essentially unique per row and produce
  pathological overfitting.

## Source provenance

Canonical source: `data_registry/datasets/hsls09_public.yaml`
§`tier3_exclusion_rules` (L42-L75).

The operational helper that consumes these rules is
`src/registry.py::RegistryLoader.is_excluded()`. Phase 2c may
relocate the rules into this skill's body if the registry YAML is
slimmed down; for now they remain in the registry and this skill
documents them.
