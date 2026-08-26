"""A continuous protected attribute is a measurement, not a set of groups.

run_subgroup_analysis grouped by raw value. X1SES is a continuous SES
composite, so a live run produced 3,126 groups of one or two students --
every one skipped for being too small, a subgroup table with nothing
usable in it, 3,126 of the run's 3,129 warnings, and a "performance gap
> 5%" fairness flag computed off whichever handful of bins happened to
clear the size threshold.

Fairness reporting is one of the claims these papers make, so a subgroup
analysis that silently degenerates is worse than one that fails.

Quantile bins are what the Critic asked for by name after seeing the same
failure in an earlier run: "using quartile or quintile bins, not raw
continuous values".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis_helpers import (
    MAX_SUBGROUP_LEVELS,
    SUBGROUP_QUANTILES,
    _grouping_series,
    run_subgroup_analysis,
)


# --- choosing whether to bin -------------------------------------------

def test_a_continuous_composite_is_binned() -> None:
    """The observed case: a float SES score with thousands of values."""
    rng = np.random.default_rng(42)
    series, binned = _grouping_series(pd.Series(rng.normal(size=3000)))
    assert binned
    assert series.nunique() == SUBGROUP_QUANTILES


def test_a_categorical_attribute_is_left_alone() -> None:
    series, binned = _grouping_series(pd.Series(["Male", "Female"] * 500))
    assert not binned
    assert series.nunique() == 2


def test_low_cardinality_numeric_codes_are_left_alone() -> None:
    """Quintile codes 1-5 are ALREADY groups; binning them would be wrong."""
    rng = np.random.default_rng(0)
    series, binned = _grouping_series(pd.Series(rng.integers(1, 6, size=1000)))
    assert not binned
    assert series.nunique() == 5


def test_the_threshold_is_the_boundary() -> None:
    values = pd.Series(list(range(MAX_SUBGROUP_LEVELS)) * 40)
    assert not _grouping_series(values)[1]
    wider = pd.Series(list(range(MAX_SUBGROUP_LEVELS + 1)) * 40)
    assert _grouping_series(wider)[1]


def test_a_constant_column_is_not_binned() -> None:
    """qcut on a constant yields one bin, which is not a grouping."""
    assert not _grouping_series(pd.Series([1.0] * 100))[1]


def test_a_heavily_tied_column_degrades_gracefully() -> None:
    """Shared bin edges must not lose the attribute entirely."""
    values = pd.Series([0.0] * 900 + list(np.linspace(1, 2, 100)))
    series, _ = _grouping_series(values)
    assert series is not None and len(series) == len(values)


# --- end to end ---------------------------------------------------------

class _Model:
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        rng = np.random.default_rng(7)
        p = rng.uniform(0.05, 0.95, size=len(X))
        return np.column_stack([1 - p, p])


@pytest.fixture()
def _fixture(tmp_path):
    rng = np.random.default_rng(3)
    n = 800
    test_X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    test_y = rng.integers(0, 2, size=n)
    protected = pd.DataFrame(
        {
            "X1SES": rng.normal(size=n),                       # continuous
            "X1SEX": rng.choice(["Male", "Female"], size=n),   # categorical
        }
    )
    path = tmp_path / "test_protected.csv"
    protected.to_csv(path)
    return test_X, test_y, str(path)


def test_a_continuous_attribute_now_yields_usable_bands(_fixture) -> None:
    test_X, test_y, path = _fixture
    warnings: list[str] = []
    out = run_subgroup_analysis(
        _Model(), test_X, test_y, path, ["X1SES"], True, warnings
    )
    assert out["X1SES"], "continuous attribute produced no reportable groups"
    assert len(out["X1SES"]) >= 2
    for band in out["X1SES"].values():
        assert band["n"] >= 10


def test_the_warning_flood_is_gone(_fixture) -> None:
    """3,126 near-identical warnings buried the three that mattered."""
    test_X, test_y, path = _fixture
    warnings: list[str] = []
    run_subgroup_analysis(_Model(), test_X, test_y, path, ["X1SES"], True, warnings)
    assert len(warnings) < 10, f"{len(warnings)} warnings for one attribute"


def test_the_binning_is_disclosed(_fixture) -> None:
    """A reader must know the bands are quantiles, not native categories."""
    test_X, test_y, path = _fixture
    warnings: list[str] = []
    run_subgroup_analysis(_Model(), test_X, test_y, path, ["X1SES"], True, warnings)
    assert any("quantile bins" in w for w in warnings)


def test_a_categorical_attribute_is_unaffected(_fixture) -> None:
    test_X, test_y, path = _fixture
    warnings: list[str] = []
    out = run_subgroup_analysis(
        _Model(), test_X, test_y, path, ["X1SEX"], True, warnings
    )
    assert set(out["X1SEX"]) == {"Male", "Female"}
    assert not any("quantile bins" in w for w in warnings)


def test_small_levels_are_summarised_not_enumerated(_fixture) -> None:
    """One line naming a few, with a count -- not one line each."""
    test_X, test_y, path = _fixture
    protected = pd.read_csv(path, index_col=0)
    protected["tiny"] = [f"g{i}" for i in range(len(protected))]  # all n=1
    protected.to_csv(path)
    warnings: list[str] = []
    run_subgroup_analysis(_Model(), test_X, test_y, path, ["tiny"], True, warnings)
    summary = [w for w in warnings if "fewer than 10 samples" in w]
    assert len(summary) == 1
    assert "more)" in summary[0]


def test_a_missing_attribute_is_still_reported(_fixture) -> None:
    """The gender analysis was silently skipped in a live run; keep it loud."""
    test_X, test_y, path = _fixture
    warnings: list[str] = []
    run_subgroup_analysis(
        _Model(), test_X, test_y, path, ["X1RACE"], True, warnings
    )
    assert any("not found in test_protected.csv" in w for w in warnings)


# --- missing values are not a group ------------------------------------

def test_missing_values_do_not_become_a_band() -> None:
    """A live run reported X1SES='nan' (n=440, AUC 0.651) as a sixth band.

    astype(str) turns NaN into the literal string "nan", which groupby
    then treats as a level -- so "missing" was reported alongside five
    real quintiles as though it were a socioeconomic group, and its AUC
    was eligible for the disparity range.
    """
    rng = np.random.default_rng(1)
    values = pd.Series(rng.normal(size=3000))
    values[:400] = np.nan
    grouping, binned = _grouping_series(values)

    assert binned
    assert grouping.nunique(dropna=True) == SUBGROUP_QUANTILES
    assert "nan" not in set(grouping.dropna().astype(str))
    assert int(grouping.isna().sum()) == 400


def test_excluded_rows_are_reported(tmp_path) -> None:
    """Silent exclusion would make the band n's fail to add up unexplained."""
    rng = np.random.default_rng(5)
    n = 600
    test_X = pd.DataFrame({"f1": rng.normal(size=n)})
    test_y = rng.integers(0, 2, size=n)
    ses = pd.Series(rng.normal(size=n))
    ses[:80] = np.nan
    path = tmp_path / "test_protected.csv"
    pd.DataFrame({"X1SES": ses}).to_csv(path)

    warnings: list[str] = []
    out = run_subgroup_analysis(
        _Model(), test_X, test_y, str(path), ["X1SES"], True, warnings
    )
    assert any("no value and are excluded" in w for w in warnings)
    assert "nan" not in out["X1SES"]


def test_a_complete_column_reports_no_exclusions(tmp_path) -> None:
    rng = np.random.default_rng(6)
    n = 600
    test_X = pd.DataFrame({"f1": rng.normal(size=n)})
    test_y = rng.integers(0, 2, size=n)
    path = tmp_path / "test_protected.csv"
    pd.DataFrame({"X1SES": rng.normal(size=n)}).to_csv(path)

    warnings: list[str] = []
    run_subgroup_analysis(
        _Model(), test_X, test_y, str(path), ["X1SES"], True, warnings
    )
    assert not any("excluded from all subgroup bands" in w for w in warnings)
