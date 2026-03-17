"""Integration tests for the DataEngineer agent.

Run with:  pytest tests/test_data_engineer.py -v --run-integration
These tests make real Anthropic API calls and require ANTHROPIC_API_KEY in the environment.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.data_engineer import DataEngineer
from src.config import load_config
from src.context import PipelineContext

CONFIG_PATH = str(Path(__file__).parent.parent / "config.yaml")

# ---------------------------------------------------------------------------
# Hardcoded research spec targeting X3TGPAMAT prediction
# ---------------------------------------------------------------------------
_RESEARCH_SPEC: dict = {
    "research_question": (
        "Can we predict students' 12th-grade math GPA using 9th-grade "
        "academic, attitudinal, and demographic factors?"
    ),
    "outcome_variable": "X3TGPAMAT",
    "outcome_type": "continuous",
    "predictor_set": [
        {
            "variable": "X1TXMTSC",
            "rationale": "Prior math achievement is the strongest predictor of subsequent GPA.",
            "wave": "base_year",
        },
        {
            "variable": "X1MTHEFF",
            "rationale": (
                "Math self-efficacy reflects students' beliefs about their "
                "ability to succeed in math, a key motivational predictor."
            ),
            "wave": "base_year",
        },
        {
            "variable": "X1SES_U",
            "rationale": (
                "Socioeconomic status is a well-established predictor of "
                "academic outcomes."
            ),
            "wave": "base_year",
        },
        {
            "variable": "X1SEX",
            "rationale": "Sex differences in math GPA are well-documented in the literature.",
            "wave": "base_year",
        },
        {
            "variable": "X1RACE",
            "rationale": (
                "Race/ethnicity captures structural inequities affecting "
                "educational outcomes."
            ),
            "wave": "base_year",
        },
    ],
    "target_population": "full sample",
    "subgroup_analyses": ["X1SEX", "X1RACE"],
    "expected_contribution": (
        "Demonstrates the relative importance of attitudes vs. prior achievement "
        "in predicting math GPA using a nationally representative sample."
    ),
    "potential_limitations": [
        "Multilevel structure not modeled",
        "Survey weights not applied",
    ],
    "novelty_score_self_assessment": 4,
}


def _make_synthetic_hsls(path: str, n: int = 200) -> None:
    """Create a small synthetic HSLS-like CSV mimicking the relevant columns.

    Includes realistic distributions and ~5% missing values on predictors
    and ~3% missing on the outcome.
    """
    rng = np.random.default_rng(42)

    df = pd.DataFrame(
        {
            "STU_ID": range(1, n + 1),
            "SCH_ID": rng.integers(1, 20, size=n),
            # Demographic predictors
            "X1SEX": rng.choice([1, 2], size=n),
            "X1RACE": rng.choice([1, 2, 3, 4, 5, 6, 7, 8], size=n),
            "X1SES_U": rng.normal(0.0, 1.0, size=n).round(2),
            # Academic / attitudinal predictors
            "X1TXMTSC": rng.normal(0.0, 1.0, size=n).round(2),
            "X1MTHEFF": rng.normal(0.0, 1.0, size=n).round(2),
            # Outcome
            "X3TGPAMAT": np.clip(rng.normal(2.5, 0.8, size=n), 0.0, 4.0).round(2),
            # Extra columns the LLM should ignore
            "W1STUDENT": rng.uniform(100, 500, size=n).round(3),
        }
    )

    # Introduce ~5% missing on each predictor
    for col in ["X1SES_U", "X1TXMTSC", "X1MTHEFF"]:
        missing_idx = rng.choice(n, size=max(1, int(0.05 * n)), replace=False)
        df.loc[missing_idx, col] = np.nan

    # Introduce ~3% missing on outcome (must be dropped, not imputed)
    missing_outcome_idx = rng.choice(n, size=max(1, int(0.03 * n)), replace=False)
    df.loc[missing_outcome_idx, "X3TGPAMAT"] = np.nan

    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_data_engineer_full_run(tmp_path: Path) -> None:
    """Full integration test: LLM generates code, code runs, outputs are validated.

    Requires:  ANTHROPIC_API_KEY in environment.
    Run with:  pytest tests/test_data_engineer.py -v --run-integration
    """
    csv_path = str(tmp_path / "synthetic_hsls.csv")
    _make_synthetic_hsls(csv_path)

    config = load_config(CONFIG_PATH)
    ctx = PipelineContext(
        dataset_name="hsls09_public",
        raw_data_path=csv_path,
        output_dir=str(tmp_path),
        max_revision_cycles=2,
    )

    agent = DataEngineer(ctx, "data_engineer", config)
    data_report = agent.run(research_spec=_RESEARCH_SPEC)

    # --- output files must exist ---
    for fname in ["train_X.csv", "train_y.csv", "test_X.csv", "test_y.csv", "data_report.json"]:
        assert (tmp_path / fname).exists(), f"Missing output file: {fname}"

    # --- data_report fields ---
    assert data_report["validation_passed"] is True, (
        f"validation_passed is False.\nWarnings: {data_report.get('warnings', [])}"
    )
    assert data_report["n_train"] > 0, "n_train must be positive"
    assert data_report["n_test"] > 0, "n_test must be positive"
    assert data_report["outcome_variable"] == "X3TGPAMAT"
    assert data_report["n_predictors_encoded"] > 0

    # n_test >= 20% of analytic_n (SPEC requirement)
    analytic_n = data_report["n_train"] + data_report["n_test"]
    assert data_report["n_test"] >= 0.19 * analytic_n, (
        f"Test set ({data_report['n_test']}) is less than 20% of analytic_n ({analytic_n})"
    )

    # --- no NaN in output CSVs ---
    train_X = pd.read_csv(tmp_path / "train_X.csv")
    train_y = pd.read_csv(tmp_path / "train_y.csv")
    test_X = pd.read_csv(tmp_path / "test_X.csv")
    test_y = pd.read_csv(tmp_path / "test_y.csv")

    assert train_X.isna().sum().sum() == 0, "NaN values found in train_X.csv"
    assert train_y.isna().sum().sum() == 0, "NaN values found in train_y.csv"
    assert test_X.isna().sum().sum() == 0, "NaN values found in test_X.csv"
    assert test_y.isna().sum().sum() == 0, "NaN values found in test_y.csv"

    # --- no zero-variance predictors ---
    zero_var = [col for col in train_X.columns if train_X[col].nunique() <= 1]
    assert not zero_var, f"Zero-variance predictors found: {zero_var}"

    # --- outcome column not leaked into predictor matrix ---
    assert "X3TGPAMAT" not in train_X.columns, "Outcome variable leaked into train_X"
    assert "X3TGPAMAT" not in test_X.columns, "Outcome variable leaked into test_X"

    # --- multilevel limitation warning present ---
    warnings = data_report.get("warnings", [])
    assert any("Multilevel" in w for w in warnings), (
        "Multilevel structure warning missing from data_report.warnings"
    )

    # --- data_report.json on disk matches returned dict ---
    with open(tmp_path / "data_report.json") as f:
        on_disk = json.load(f)
    assert on_disk["validation_passed"] == data_report["validation_passed"]
    assert on_disk["n_train"] == data_report["n_train"]
