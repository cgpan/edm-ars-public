"""A design matrix wide enough that nothing can train must stop the run.

A2, observed on a real JEDM run. `X1TXMTSCOR` -- a standardised maths
theta score with 20,741 distinct values -- was one-hot encoded into
16,705 dummy columns. `X1SCHOOLENG` added another 191. The design matrix
reached 18,806 x 16,945 and 1.8 GB, no model trained inside any timeout,
and three consecutive runs failed while being misdiagnosed as
"slowness".

The generated code contained a guard meant to prevent exactly this:

    onehot_cols = categorical_cols.copy()
    for col in onehot_cols:
        if train_X_encoded[col].nunique() > 100:
            onehot_cols.remove(col)      # mutates the list being iterated

Removing an element mid-iteration advances the iterator past the next
element, so roughly half the offenders survive. Prose guidance did not
hold, which is why the ceiling is enforced in the orchestrator where a
violation triggers a targeted retry.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest

from src.orchestrator import MAX_ENCODED_COLUMNS, check_design_matrix_width


def _write_matrix(tmp_path: Path, columns: list[str]) -> None:
    pd.DataFrame({c: [0, 1] for c in columns}).to_csv(
        tmp_path / "train_X.csv", index=False
    )


def test_the_observed_explosion_is_refused(tmp_path: Path) -> None:
    """16,705 dummies from one continuous column."""
    cols = [f"X1TXMTSCOR_{i}" for i in range(1200)] + ["X1SES", "X1SEX"]
    _write_matrix(tmp_path, cols)
    violation = check_design_matrix_width(str(tmp_path))
    assert violation is not None
    assert "above the ceiling" in violation
    assert "X1TXMTSCOR" in violation, "the violation must name the culprit"


def test_a_normal_matrix_passes(tmp_path: Path) -> None:
    """~50 columns is what a healthy prediction spec produces."""
    cols = ["X1SES", "X1TXMTSCOR", "X1MTHEFF"] + [
        f"X1RACE_{i}" for i in range(8)
    ] + [f"X1PAREDU_{i}" for i in range(7)]
    _write_matrix(tmp_path, cols)
    assert check_design_matrix_width(str(tmp_path)) is None


def test_exactly_at_the_ceiling_is_allowed(tmp_path: Path) -> None:
    _write_matrix(tmp_path, [f"c{i}" for i in range(MAX_ENCODED_COLUMNS)])
    assert check_design_matrix_width(str(tmp_path)) is None


def test_one_over_the_ceiling_is_refused(tmp_path: Path) -> None:
    _write_matrix(
        tmp_path, [f"c{i}" for i in range(MAX_ENCODED_COLUMNS + 1)]
    )
    assert check_design_matrix_width(str(tmp_path)) is not None


def test_absent_matrix_is_not_a_violation(tmp_path: Path) -> None:
    """The pre-flight must never be the thing that breaks a healthy run."""
    assert check_design_matrix_width(str(tmp_path)) is None


def test_unreadable_matrix_is_not_a_violation(tmp_path: Path) -> None:
    (tmp_path / "train_X.csv").write_bytes(b"\x00\x01\x02 not a csv")
    assert check_design_matrix_width(str(tmp_path)) is None


def test_the_message_explains_the_dtype_trap(tmp_path: Path) -> None:
    """The fix is non-obvious: the guard failed because sentinel
    replacement flips a numeric column to object dtype."""
    _write_matrix(tmp_path, [f"X1TXMTSCOR_{i}" for i in range(600)])
    violation = check_design_matrix_width(str(tmp_path))
    assert "registry" in violation.lower()
    assert "dtype" in violation.lower()
    assert "iterating" in violation.lower() or "while iterating" in violation.lower()


@pytest.mark.parametrize("n", [501, 1000, 16945])
def test_scales_to_the_observed_magnitude(tmp_path: Path, n: int) -> None:
    _write_matrix(tmp_path, [f"c{i}" for i in range(n)])
    violation = check_design_matrix_width(str(tmp_path))
    assert violation is not None
    assert str(n) in violation
