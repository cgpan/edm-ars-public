---
name: hsls09-csv-format-quirks
layer: dataset
description: HSLS:09 public-use CSV stores text labels for categorical variables, not numeric codes; never coerce all columns to numeric.
trigger_keywords:
  - hsls
  - hsls09
  - csv
  - categorical
  - label
  - labels
  - to_numeric
  - coerce
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

# HSLS:09 CSV Format Quirks

The HSLS:09 public-use CSV stores **text labels** for categorical
variables, NOT numeric codes. This is the single most error-prone
aspect of working with the file and has caused real pipeline failures
in the past.

## What the data actually looks like

| Variable | Sample values |
|---|---|
| `X1SEX` | `"Male"`, `"Female"` (not `1`, `2`) |
| `X1RACE` | `"White, non-Hispanic"`, `"Black/African-American, non-Hispanic"`, ... |
| `X1STUEDEXPCT` | `"Complete a Bachelor's degree"`, `"High school diploma or GED"`, ... |
| `X1PAREDU` | `"Bachelor's degree"`, `"High school diploma or GED"`, ... |
| `X4EVERDROP` | `"Yes"`, `"No"` (not `1`, `0`) |
| Missing values | `"Missing"`, `"Unit non-response"`, `"Item legitimate skip/NA"`, `"Data suppressed"`, `"Component not applicable"`, `"Don't know"` |

## NEVER do this

```python
# WRONG — silently nulls every categorical value, producing 100% missingness.
df = df.apply(lambda col: pd.to_numeric(col, errors="coerce"))
```

`pd.to_numeric(errors='coerce')` on a column of text labels turns every
non-numeric string into `NaN`. Apply it to all columns and you destroy
every categorical predictor in the dataset. The pipeline will continue
without raising, the imputer will fill the resulting NaNs, and the
final results will be junk.

## DO this instead

1. **First**, replace the documented missing labels (text) and NCES
   negative sentinels (numeric) with `NaN` (see `hsls09-missing-codes`):

   ```python
   missing_codes = [-1, -4, -5, -6, -7, -8, -9]
   missing_labels = [
       "Missing", "Unit non-response", "Item legitimate skip/NA",
       "Data suppressed", "Component not applicable", "Don't know",
   ]
   df = df.replace(missing_codes + missing_labels, pd.NA)
   ```

2. **For categorical predictors**, label-encode the remaining text
   values to integers before any numeric operation (e.g.,
   `IterativeImputer`):

   ```python
   from sklearn.preprocessing import LabelEncoder
   for col in categorical_columns:
       df[col] = LabelEncoder().fit_transform(df[col].astype(str))
   ```

3. **For binary outcomes with text labels** (e.g., `"Yes"`/`"No"`), map
   explicitly:

   ```python
   df[outcome] = df[outcome].map({"Yes": 1, "No": 0})
   ```

4. **Only call `pd.to_numeric()`** on columns you have already confirmed
   are continuous/numeric.

## Mandatory: Cardinality guard before one-hot encoding

`pd.get_dummies` and similar one-hot encoders produce **one column per
unique value**. When applied to a continuous variable with thousands of
unique decimal values (e.g. `X1TXMTSCOR` with values like 50.123,
50.456, 50.789), the result is a junk feature space with tens of
thousands of columns and no meaningful signal. The Phase 2c R3.5
OpenAI run hit exactly this — `train_X.csv` came back at 20,112
columns / 1.6 GB because the LLM-generated DE code applied
`pd.get_dummies` to `X1TXMTSCOR`.

**Required check before one-hot encoding any column:**

```python
# WRONG -- one-hots everything object-typed:
df_encoded = pd.get_dummies(df)

# WRONG -- one-hots high-cardinality columns:
df[col] = pd.get_dummies(df[col])  # if df[col].nunique() > 100, this is wrong

# RIGHT -- gate one-hot on dtype AND cardinality:
def safe_encode_column(df, col):
    if df[col].dtype not in ("object", "category"):
        # continuous; do not encode -- keep numeric column as-is
        return df[col]
    n_unique = df[col].nunique(dropna=False)
    if n_unique > 100:
        # high-cardinality; do not one-hot. Use label encoding instead.
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        return le.fit_transform(df[col].fillna("_MISSING_").astype(str))
    return pd.get_dummies(df[col], prefix=col)
```

**Validation after encoding:** verify
`train_X.shape[1] < 5 * len(predictor_set)` for typical HSLS predictor
sets (≤30 raw variables → ≤150 encoded columns). A column count above
1000 is structural corruption; abort with `validation_passed: false`
and add a warning naming the offending column(s).

## Snapshot original labels before encoding (for subgroup analysis)

The Analyst's subgroup analysis needs the original text labels (e.g.
`"Male"`/`"Female"`, race/ethnicity names) to produce readable group
keys in `subgroup_performance.csv`. Capture them on the analytic
DataFrame BEFORE any encoding:

```python
subgroup_cols = research_spec.get("subgroup_analyses", [])
available_subgroup_cols = [c for c in subgroup_cols if c in df.columns]
subgroup_snapshot = df[available_subgroup_cols].copy()
# ... (later) align subgroup_snapshot to test indices and save as test_protected.csv
```

## Source provenance

Canonical source: `agent_prompts/data_engineer.yaml` §"CRITICAL: HSLS:09
CSV Format Warning".

Merged content from: none — single-sourced.
