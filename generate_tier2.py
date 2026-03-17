"""
generate_tier2.py — Auto-generate Tier 2 variable entries from the HSLS:09 data file.

This script reads the actual HSLS:09 CSV/Stata/SPSS data file, extracts column metadata,
filters out Tier 1 (already curated) and Tier 3 (excluded) variables, and produces a
YAML file with basic metadata for the remaining variables.

Usage:
    python generate_tier2.py \
        --data data/raw/hsls_16_student_v1_0.csv \
        --tier1 data_registry/datasets/hsls09_public.yaml \
        --output data_registry/datasets/hsls09_tier2_auto.yaml

For Stata (.dta) files, variable labels are extracted directly.
For CSV files, labels are inferred from column names using HSLS naming conventions.
"""

import argparse
import re
import yaml
import sys
from pathlib import Path

import pandas as pd
import numpy as np


# ============================================================================
# HSLS:09 Variable Naming Convention Parser
# ============================================================================
# HSLS variable names follow systematic prefixes that encode wave and source:
#   X1 = base year composite/derived
#   X2 = first follow-up composite/derived
#   X3 = second follow-up / transcript composite/derived
#   X4 = 2016 update panel composite/derived
#   S1 = base year student questionnaire raw
#   S2 = first follow-up student questionnaire raw
#   S3 = 2013 update / transcript raw
#   S4 = second follow-up student questionnaire raw
#   P1 = base year parent questionnaire
#   P2 = first follow-up parent questionnaire
#   N1 = base year school counselor
#   A1 = base year school administrator
#   C1 = base year school characteristics (derived)
#   T1 = base year math teacher
#   T2 = base year science teacher
#   W* = weights
#   BRR* = BRR replicate weights

WAVE_PREFIX_MAP = {
    "X1": "base_year",
    "S1": "base_year",
    "P1": "base_year",
    "N1": "base_year",
    "A1": "base_year",
    "C1": "base_year",
    "T1": "base_year",
    "T2": "base_year",
    "X2": "first_follow_up",
    "S2": "first_follow_up",
    "P2": "first_follow_up",
    "X3": "second_follow_up",
    "S3": "second_follow_up",
    "X4": "update_panel",
    "S4": "update_panel",
}

SOURCE_PREFIX_MAP = {
    "X": "composite/derived",
    "S": "student_questionnaire",
    "P": "parent_questionnaire",
    "N": "counselor_questionnaire",
    "A": "administrator_questionnaire",
    "C": "school_characteristics",
    "T": "teacher_questionnaire",
}


def infer_wave(varname: str) -> str:
    """Infer the wave from the variable name prefix."""
    for prefix, wave in WAVE_PREFIX_MAP.items():
        if varname.upper().startswith(prefix):
            return wave
    return "unknown"


def infer_source(varname: str) -> str:
    """Infer the data source from the variable name prefix."""
    first_char = varname[0].upper() if varname else ""
    return SOURCE_PREFIX_MAP.get(first_char, "unknown")


def infer_type(series: pd.Series) -> str:
    """Infer variable type from pandas Series."""
    if series.dtype in ("object", "category"):
        return "categorical"
    n_unique = series.nunique()
    if n_unique == 2:
        return "binary"
    if n_unique <= 10 and series.dtype in ("int64", "int32", "float64"):
        # Likely categorical or ordinal
        return "categorical"
    return "continuous"


def compute_missingness(series: pd.Series, negative_as_missing: bool = True) -> float:
    """
    Compute percent missing.
    HSLS uses negative codes (-9, -8, -7, -6, -4, -1) for various types of
    missing data. By default, these are counted as missing.
    """
    n_total = len(series)
    n_missing = series.isna().sum()
    if negative_as_missing:
        n_missing += (series < 0).sum()
    return round(100 * n_missing / n_total, 1)


# ============================================================================
# Tier 3 Exclusion Logic
# ============================================================================

def load_exclusion_rules(tier1_path: str) -> dict:
    """Load Tier 3 exclusion rules from the Tier 1 registry."""
    with open(tier1_path) as f:
        registry = yaml.safe_load(f)
    return registry.get("tier3_exclusion_rules", {})


def is_excluded(varname: str, rules: dict) -> bool:
    """Check if a variable should be excluded (Tier 3)."""
    varname_upper = varname.upper()

    # Check exact matches
    if varname_upper in [v.upper() for v in rules.get("exact_matches", [])]:
        return True

    # Check prefix patterns
    for pattern in rules.get("prefix_patterns", []):
        if re.match(pattern, varname_upper):
            return True

    # Check suffix patterns
    for pattern in rules.get("suffix_patterns", []):
        if re.search(pattern, varname_upper):
            return True

    return False


def get_tier1_varnames(tier1_path: str) -> set:
    """Extract all variable names from Tier 1 registry."""
    with open(tier1_path) as f:
        registry = yaml.safe_load(f)

    tier1_vars = set()
    variables = registry.get("variables", {})

    # Outcomes
    for var in variables.get("outcomes", []):
        tier1_vars.add(var["name"].upper())

    # Predictors (nested by category)
    predictors = variables.get("predictors", {})
    for category_name, var_list in predictors.items():
        if isinstance(var_list, list):
            for var in var_list:
                tier1_vars.add(var["name"].upper())

    return tier1_vars


# ============================================================================
# Main Generation Logic
# ============================================================================

def generate_tier2(data_path: str, tier1_path: str) -> list:
    """
    Generate Tier 2 variable entries from the data file.

    Returns a list of dicts, each representing a Tier 2 variable with:
    - name, label, type, wave, source, pct_missing, n_unique, range (if continuous)
    """
    # Load data
    suffix = Path(data_path).suffix.lower()
    print(f"Loading data from {data_path} ({suffix} format)...")

    if suffix == ".csv":
        df = pd.read_csv(data_path, nrows=0)  # just headers for column names
        df_sample = pd.read_csv(data_path, nrows=5000)  # sample for type inference
        has_labels = False
    elif suffix == ".dta":
        df_full = pd.read_stata(data_path, iterator=True)
        chunk = df_full.read(5000)
        df = chunk
        df_sample = chunk
        # Stata files have variable labels
        import io
        reader = pd.io.stata.StataReader(data_path)
        var_labels = reader.variable_labels()
        has_labels = True
        reader.close()
    elif suffix in (".sav", ".zsav"):
        import pyreadstat
        df_sample, meta = pyreadstat.read_sav(data_path, row_limit=5000)
        var_labels = meta.column_labels_and_names
        has_labels = True
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    # Load exclusion rules and Tier 1 variable names
    exclusion_rules = load_exclusion_rules(tier1_path)
    tier1_vars = get_tier1_varnames(tier1_path)

    all_columns = list(df_sample.columns)
    print(f"Total columns in data file: {len(all_columns)}")

    tier2_entries = []
    n_excluded_tier3 = 0
    n_already_tier1 = 0

    for col in all_columns:
        col_upper = col.upper()

        # Skip if already in Tier 1
        if col_upper in tier1_vars:
            n_already_tier1 += 1
            continue

        # Skip if excluded by Tier 3 rules
        if is_excluded(col, exclusion_rules):
            n_excluded_tier3 += 1
            continue

        # Build Tier 2 entry
        series = df_sample[col]
        var_type = infer_type(series)
        pct_miss = compute_missingness(series)
        wave = infer_wave(col)
        source = infer_source(col)
        n_unique = int(series.nunique())

        entry = {
            "name": col,
            "type": var_type,
            "wave": wave,
            "source": source,
            "pct_missing": pct_miss,
            "n_unique": n_unique,
            "tier": 2,
        }

        # Add label if available from Stata/SPSS metadata
        if has_labels and suffix == ".dta":
            entry["label"] = var_labels.get(col, col)
        elif has_labels and suffix in (".sav", ".zsav"):
            idx = list(df_sample.columns).index(col)
            if idx < len(var_labels):
                entry["label"] = var_labels[idx][1] if var_labels[idx][1] else col
        else:
            entry["label"] = col  # placeholder; user should enrich later

        # Add range for continuous variables
        if var_type == "continuous":
            valid_values = series[series >= 0] if (series < 0).any() else series
            valid_values = valid_values.dropna()
            if len(valid_values) > 0:
                entry["range"] = [round(float(valid_values.min()), 2),
                                  round(float(valid_values.max()), 2)]

        # Add value codes for categorical variables with few levels
        if var_type in ("categorical", "binary") and n_unique <= 15:
            valid_values = series[(series >= 0) | series.isna()].dropna()
            if len(valid_values) > 0:
                entry["codes"] = {int(v): f"code_{int(v)}" for v in sorted(valid_values.unique())}

        tier2_entries.append(entry)

    print(f"Tier 1 (already curated): {n_already_tier1}")
    print(f"Tier 3 (excluded): {n_excluded_tier3}")
    print(f"Tier 2 (auto-generated): {len(tier2_entries)}")

    return tier2_entries


def write_tier2_yaml(entries: list, output_path: str):
    """Write Tier 2 entries to a YAML file."""
    output = {
        "tier": 2,
        "description": "Auto-generated variable entries from HSLS:09 data file. "
                       "These variables have basic metadata but lack hand-curated "
                       "substantive annotations. ProblemFormulator may use these "
                       "with stronger justification requirements.",
        "generation_note": "Generated by generate_tier2.py. Do not edit manually. "
                           "Re-run the script to regenerate after updating Tier 1 or Tier 3 rules.",
        "n_variables": len(entries),
        "variables": entries,
    }

    # Organize by wave for readability
    by_wave = {}
    for entry in entries:
        wave = entry.get("wave", "unknown")
        if wave not in by_wave:
            by_wave[wave] = []
        by_wave[wave].append(entry)

    output["variables_by_wave"] = by_wave
    output["wave_counts"] = {w: len(vs) for w, vs in by_wave.items()}

    with open(output_path, "w") as f:
        yaml.dump(output, f, default_flow_style=False, sort_keys=False, width=120)

    print(f"\nTier 2 registry written to {output_path}")
    print("Wave distribution:")
    for wave, count in output["wave_counts"].items():
        print(f"  {wave}: {count} variables")


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate Tier 2 variable registry from HSLS:09 data file"
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to HSLS:09 data file (.csv, .dta, or .sav)"
    )
    parser.add_argument(
        "--tier1", required=True,
        help="Path to Tier 1 registry YAML (hsls09_public.yaml)"
    )
    parser.add_argument(
        "--output", default="data_registry/datasets/hsls09_tier2_auto.yaml",
        help="Output path for Tier 2 YAML"
    )

    args = parser.parse_args()

    if not Path(args.data).exists():
        print(f"ERROR: Data file not found: {args.data}")
        sys.exit(1)
    if not Path(args.tier1).exists():
        print(f"ERROR: Tier 1 registry not found: {args.tier1}")
        sys.exit(1)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    entries = generate_tier2(args.data, args.tier1)
    write_tier2_yaml(entries, args.output)


if __name__ == "__main__":
    main()
