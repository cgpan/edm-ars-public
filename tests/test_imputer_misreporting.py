"""`IterativeImputer` on one column is mean imputation wearing another name.

A3, observed across five JEDM papers. The generated data-engineering code
did:

    imputer = IterativeImputer(max_iter=5, random_state=RANDOM_STATE)
    train_vals = train_X_raw[[col]].values          # ONE column
    train_imputed = imputer.fit_transform(train_vals)

`IterativeImputer` models each feature from the OTHER features. Given a
single column there are no others, so it degenerates to mean imputation.
The code then recorded
`missingness_summary[col]["imputation_method"] = "IterativeImputer"` and
the manuscript repeated it — so five papers named a multivariate imputer
and performed mean-fill, on variables missing 28–36% (`X1PAREDEXPCT`
36.1%, `X1PAREDU` 28.6%).

That is a reporting-accuracy defect, not a methods preference. The check
blocks rather than warns because the code runs cleanly and produces
plausible numbers: nothing downstream can notice, and the output is a
manuscript that misstates its own methods.

The skill's own reference implementation taught the pattern, which is why
a prose fix alone would not hold — the same shape as the `n_iterations`
defect.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.sandbox import SubprocessExecutor, check_silent_misbehaviour

SINGLE_COLUMN = """
from sklearn.impute import IterativeImputer
imputer = IterativeImputer(max_iter=5, random_state=RANDOM_STATE)
train_vals = train_X_raw[[col]].values
train_imputed = imputer.fit_transform(train_X_raw[[col]])
"""

FULL_BLOCK = """
from sklearn.impute import IterativeImputer
imputer = IterativeImputer(max_iter=5, random_state=42)
train_X[numeric_block] = imputer.fit_transform(train_X[numeric_block])
test_X[numeric_block] = imputer.transform(test_X[numeric_block])
"""

SIMPLE_PER_COLUMN = """
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
train_X[[col]] = imputer.fit_transform(train_X[[col]])
"""


def test_the_pattern_that_shipped_is_detected() -> None:
    findings = check_silent_misbehaviour(SINGLE_COLUMN)
    assert findings, "single-column IterativeImputer was not detected"
    assert "mean imputation" in findings[0].lower()


def test_fitting_on_the_full_block_is_allowed() -> None:
    assert check_silent_misbehaviour(FULL_BLOCK) == []


def test_simple_imputer_per_column_is_allowed() -> None:
    """A median IS a column-wise statistic; per-column SimpleImputer is
    correct and must not be swept up by the check."""
    assert check_silent_misbehaviour(SIMPLE_PER_COLUMN) == []


@pytest.mark.parametrize(
    "code",
    [
        "import pandas as pd\nX = pd.read_csv('train_X.csv')\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "df[['a', 'b']] = df[['a', 'b']].fillna(0)\n",
    ],
    ids=["read-csv", "sklearn-import", "double-bracket-no-imputer"],
)
def test_ordinary_code_is_not_flagged(code: str) -> None:
    """A check that fires on normal code gets switched off."""
    assert check_silent_misbehaviour(code) == []


def test_executor_blocks_before_running(tmp_path: Path) -> None:
    result = SubprocessExecutor().run(
        code=SINGLE_COLUMN + "\nprint('should not run')\n",
        output_dir=str(tmp_path),
        timeout_s=30,
    )
    assert result["returncode"] == 3
    assert "SILENT-MISBEHAVIOUR" in result["stderr"]
    assert result["stdout"] == ""
    assert not (tmp_path / "_generated_script.py").exists()


def test_executor_still_runs_the_correct_form(tmp_path: Path) -> None:
    """Guarding against a check that blocks the fix as well as the bug."""
    result = SubprocessExecutor().run(
        code="print('correct imputation form ran')\n",
        output_dir=str(tmp_path),
        timeout_s=60,
    )
    assert result["returncode"] == 0
    assert "correct imputation form ran" in result["stdout"]


def test_the_skill_no_longer_teaches_the_broken_pattern() -> None:
    """The reference implementation was the source of the defect."""
    skill = (
        Path(__file__).resolve().parents[1]
        / "skills" / "methodology" / "missingness-tiered-protocol" / "SKILL.md"
    )
    text = skill.read_text(encoding="utf-8")
    # The wrong form may appear ONLY inside an explicitly-labelled WRONG block.
    assert "MANDATORY RULE" in text
    assert "never fit `IterativeImputer` on a single column" in text
    assert "degenerates to" in text
    # And the corrected sketch must fit on a block, not a column.
    assert "imputer.fit_transform(train_X[numeric_block])" in text
