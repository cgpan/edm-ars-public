---
name: hsls09-missing-codes
layer: dataset
description: NCES sentinel values and text labels that must be treated as missing in HSLS:09 before any imputation or modeling.
trigger_keywords:
  - hsls
  - hsls09
  - missing
  - missingness
  - nces
  - sentinel
  - code
  - codes
applicable_task_types: []
applicable_datasets:
  - hsls09_public
applicable_stages:
  - DataEngineer
priority: 1
references_skills: []
resources: []
version: "1.0"
rule_severity: mandatory
---

# HSLS:09 Missing Codes

HSLS:09 mixes two missing-data conventions that must both be treated as
NaN before any analytic computation:

1. **Numeric sentinels** for skip patterns and non-response.
2. **Text labels** stored in the public-use CSV in place of those
   numeric codes for many variables.

Failing to convert either category produces silent garbage downstream:
sentinels are interpreted as valid values (`-9` looks like a real
score), and text labels survive into the model as additional categories
with no observations.

## Numeric sentinels (numeric variables)

| Code | Meaning |
|---|---|
| `-1` | Item legitimate skip / NA |
| `-4` | Nonrespondent |
| `-5` | Data suppressed |
| `-6` | Component not applicable |
| `-7` | Item legitimate skip / NA (alt) |
| `-8` | Unit non-response |
| `-9` | Missing |

Treat ALL of these as `NaN`. The full list lives in the registry under
`missingness.sentinel_codes_or_labels`.

## Text labels (categorical variables in the public-use CSV)

| Label |
|---|
| `"Missing"` |
| `"Unit non-response"` |
| `"Item legitimate skip/NA"` |
| `"Data suppressed"` |
| `"Component not applicable"` |
| `"Don't know"` |

Treat ALL of these as `NaN` before label-encoding the remaining values.

## Operational sequence

```python
import pandas as pd

missing_codes = [-1, -4, -5, -6, -7, -8, -9]
missing_labels = [
    "Missing",
    "Unit non-response",
    "Item legitimate skip/NA",
    "Data suppressed",
    "Component not applicable",
    "Don't know",
]

df = df.replace(missing_codes + missing_labels, pd.NA)
```

Apply this **before** the outcome-drop step
(`df = df.dropna(subset=[outcome_variable])`) so the outcome's missing
values are properly recognized.

## Mechanism assumption

Per the registry: `mechanism_assumption: mixed_MAR_MNAR`. Some HSLS
variables (especially postsecondary outcomes) are missing
non-randomly because the question was conditional on a prior event
(e.g., college attendance). For those, see `hsls09-structural-mnar-outcomes`.

The recommended operational handling is `multiple_imputation_or_indicator_plus_sensitivity`
(see the methodology skill `missingness-tiered-protocol`).

## MNAR candidates flagged in the registry

| Variable | Reason |
|---|---|
| `dropout_derived` | Conditional on observation through high school |
| `X4EVRATNDCLG` | Conditional on response in the 2016 panel |
| `X4RFDGMJ14Y` | Conditional on attending college and selecting a major |
| `X5STEMCRED` | Conditional on transcript-confirmed degree |

## Source provenance

Canonical source: `data_registry/datasets/hsls09_public.yaml`
§`missingness` (L1739-L1762).

Merged content from: `agent_prompts/data_engineer.yaml` §step 3
(operational sequence and the order of operations relative to
outcome-drop).
