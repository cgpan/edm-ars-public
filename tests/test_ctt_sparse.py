"""G2 / F-P5-CTT-SPARSE-NAN — pairwise-present Cronbach's alpha.

`psy_ctt` used to call `items_df.dropna()`. On the ASSISTments
skill-builder matrix (1586 students x 47 items, fill rate 0.276) no
student answered all 47 items, so zero rows survived and alpha plus all
47 per-item statistics came back NaN — an unusable, unstateable result
that also breaks strict JSON.

These tests pin the replacement contract:
  * pairwise-present estimation by DEFAULT (no flag, no agent decision),
  * every number carries the n it was computed from,
  * a matrix too sparse for a defensible estimate returns an explicit
    "not estimable" result with the fill rate and the reason,
  * NEVER a bare NaN.

Offline: pure numpy/pandas, no LLM, no network, no R.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis_helpers import psy_ctt

ROOT = Path(__file__).resolve().parent.parent
REAL_ITEMS = ROOT / "runs" / "arc_p_validation_20260711" / "output" / \
    "items_analytic.csv"


# ---------------------------------------------------------------------------
# fixtures / builders
# ---------------------------------------------------------------------------

def _complete_likert(n: int = 400, k: int = 6, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    theta = rng.normal(size=n)
    return pd.DataFrame({
        f"i{j}": np.clip(np.round(2.5 + 0.8 * theta + rng.normal(0, 0.7, n)),
                         1, 4)
        for j in range(k)
    })


def _rotating_gaps(n: int = 400, k: int = 8, seed: int = 3) -> pd.DataFrame:
    """Sparse BY DESIGN: every person is missing exactly one item, so the
    listwise-complete set is EMPTY while every item pair still overlaps
    on hundreds of persons."""
    rng = np.random.default_rng(seed)
    theta = rng.normal(size=n)
    df = pd.DataFrame({
        f"i{j}": (theta + rng.normal(0, 0.8, n) > 0).astype(float)
        for j in range(k)
    })
    for row in range(n):
        df.iat[row, row % k] = np.nan
    return df


def _listwise_rows(df: pd.DataFrame) -> int:
    """What the old complete-case implementation had to work with."""
    return int(len(df.dropna()))


# ---------------------------------------------------------------------------


class TestCompleteMatrixUnchanged:
    """The sparse-safe path must reproduce classical CTT exactly when the
    matrix is complete — otherwise every existing psychometrics paper's
    numbers move for no reason."""

    def test_matches_hand_computed_classical_alpha(self) -> None:
        df = _complete_likert()
        k = df.shape[1]
        total = df.sum(axis=1)
        expected = (k / (k - 1)) * (
            1 - df.var(ddof=1).sum() / total.var(ddof=1))

        out = psy_ctt(df)

        assert out["estimable"] is True
        assert out["method"] == "pairwise_present"
        assert out["cronbach_alpha"] == pytest.approx(expected, abs=1e-12)
        assert out["alpha_listwise"] == pytest.approx(expected, abs=1e-12)
        assert out["n_complete"] == len(df)
        assert out["n_items"] == k and out["n_items_in_alpha"] == k
        assert out["matrix_fill_rate"] == pytest.approx(1.0)
        assert out["caveats"] == []

    def test_item_stats_match_classical_definitions(self) -> None:
        df = _complete_likert()
        total = df.sum(axis=1)
        out = psy_ctt(df)

        for row in out["items"]:
            col = df[row["item"]]
            rest = total - col
            assert row["mean"] == pytest.approx(float(col.mean()))
            assert row["sd"] == pytest.approx(float(col.std(ddof=1)))
            assert row["difficulty"] == pytest.approx(
                float(col.mean() / col.max()))
            assert row["item_total_r"] == pytest.approx(
                float(np.corrcoef(col, rest)[0, 1]), abs=1e-12)
            # the n behind every number is carried
            assert row["n_used"] == len(df)
            assert row["item_total_r_n"] == len(df)
            assert row["in_alpha"] is True


class TestZeroCompleteCases:
    """The defect itself: no complete case, yet the data are informative."""

    def test_listwise_would_have_produced_nothing(self) -> None:
        assert _listwise_rows(_rotating_gaps()) == 0

    def test_alpha_is_estimated_pairwise_not_nan(self) -> None:
        df = _rotating_gaps()
        out = psy_ctt(df)

        assert out["estimable"] is True
        assert out["method"] == "pairwise_present"
        assert out["n_complete"] == 0
        alpha = out["cronbach_alpha"]
        assert alpha is not None and np.isfinite(alpha)
        assert 0.0 < alpha < 1.0
        assert out["alpha_listwise"] is None
        assert out["n_items_in_alpha"] == df.shape[1]

    def test_every_number_carries_its_n(self) -> None:
        df = _rotating_gaps()
        out = psy_ctt(df)

        assert out["pair_n"]["threshold"] == 30
        assert out["pair_n"]["min"] >= 30
        assert out["pair_n"]["n_pairs"] == 8 * 7 // 2  # unordered pairs
        for row in out["items"]:
            assert 0 < row["n_used"] < len(df)
            assert row["n_used"] == int(df[row["item"]].notna().sum())
            assert 0 < row["item_total_r_n"] <= row["n_used"]
            assert row["n_pair_min"] >= 30
            assert row["mean"] is not None and row["item_total_r"] is not None

    def test_prose_can_state_the_result_truthfully(self) -> None:
        out = psy_ctt(_rotating_gaps())
        assert "pairwise-present" in out["summary"]
        assert "Listwise-complete estimation is impossible" in out["summary"]
        assert any("no person answered every item" in c
                   for c in out["caveats"])
        assert any("missing at random" in c for c in out["caveats"])


class TestNotEstimable:
    """Too sparse for a defensible estimate -> say so, with the numbers."""

    def _disjoint_persons(self) -> pd.DataFrame:
        # 8 items, each answered by a DIFFERENT block of 60 persons:
        # every item is individually fine, no item pair overlaps at all.
        rng = np.random.default_rng(11)
        n, k, block = 480, 8, 60
        df = pd.DataFrame(np.nan, index=range(n),
                          columns=[f"i{j}" for j in range(k)])
        for j in range(k):
            rows = slice(j * block, (j + 1) * block)
            df.iloc[rows, j] = rng.integers(0, 2, block).astype(float)
        return df

    def test_returns_explicit_not_estimable_never_nan(self) -> None:
        df = self._disjoint_persons()
        out = psy_ctt(df)

        assert out["estimable"] is False
        assert out["method"] == "not_estimable"
        assert out["cronbach_alpha"] is None
        assert out["alpha_listwise"] is None
        assert out["n_items_in_alpha"] == 0
        reason = out["not_estimable_reason"]
        assert "jointly-observed" in reason and "30" in reason
        assert out["matrix_fill_rate"] == pytest.approx(1.0 / 8, abs=1e-9)
        assert out["n_complete"] == 0
        assert "not estimable" in out["summary"]
        assert "0.125" in out["summary"] or "fill rate" in out["summary"]
        assert any("IRT/CDM" in c for c in out["caveats"])

    def test_item_level_statistics_are_still_reported(self) -> None:
        # means/difficulties ARE estimable per item even when alpha is not;
        # returning nothing would throw away real information.
        out = psy_ctt(self._disjoint_persons())
        assert len(out["items"]) == 8
        for row in out["items"]:
            assert row["n_used"] == 60
            assert row["mean"] is not None
            assert row["difficulty"] is not None
            assert row["in_alpha"] is False

    def test_single_item_does_not_crash(self) -> None:
        # the old implementation raised TypeError: float(None) here
        out = psy_ctt(pd.DataFrame({"i0": [1.0, 2.0, 3.0, 4.0] * 20}))
        assert out["estimable"] is False
        assert out["cronbach_alpha"] is None
        assert "fewer than 2 usable items" in out["not_estimable_reason"]

    def test_all_missing_matrix(self) -> None:
        df = pd.DataFrame({"i0": [np.nan] * 50, "i1": [np.nan] * 50})
        out = psy_ctt(df)
        assert out["estimable"] is False
        assert out["cronbach_alpha"] is None
        assert out["matrix_fill_rate"] == 0.0


class TestItemPruningIsAudited:
    def test_thin_item_dropped_and_recorded(self) -> None:
        df = _rotating_gaps(n=400, k=6)
        # a 7th item only 20 persons share with anything: enough responses
        # of its own (40) but no defensible covariance with the others
        thin = np.full(400, np.nan)
        thin[:40] = np.arange(40) % 2
        df = df.assign(thin_item=thin)
        df.loc[20:39, [f"i{j}" for j in range(6)]] = np.nan

        out = psy_ctt(df)

        assert out["estimable"] is True
        assert out["n_items"] == 7 and out["n_items_in_alpha"] == 6
        excluded = {e["item"]: e["reason"] for e in out["items_excluded"]}
        assert "thin_item" in excluded
        assert "jointly-observed" in excluded["thin_item"]
        by_item = {r["item"]: r for r in out["items"]}
        assert by_item["thin_item"]["in_alpha"] is False
        assert by_item["thin_item"]["excluded_reason"] is not None
        assert by_item["thin_item"]["n_used"] == 40  # still described
        assert any("6 of 7 items" in c for c in out["caveats"])
        assert "6 of 7 items" in out["summary"]

    def test_low_n_and_zero_variance_items_excluded_with_reasons(self) -> None:
        df = _complete_likert(n=200, k=5)
        df["constant"] = 3.0
        sparse = np.full(200, np.nan)
        sparse[:10] = 2.0
        sparse[10:20] = 4.0
        df["rare"] = sparse

        out = psy_ctt(df)

        excluded = {e["item"]: e["reason"] for e in out["items_excluded"]}
        assert "zero variance" in excluded["constant"]
        assert "10 observed" in excluded["rare"] or \
            "20 observed" in excluded["rare"]
        assert out["n_items_in_alpha"] == 5
        assert out["estimable"] is True

    def test_non_numeric_columns_are_excluded_not_coerced(self) -> None:
        df = _complete_likert(n=200, k=4)
        df.insert(0, "user_id", [f"u{i}" for i in range(200)])
        out = psy_ctt(df)
        excluded = {e["item"]: e["reason"] for e in out["items_excluded"]}
        assert "non-numeric" in excluded["user_id"]
        assert out["n_items"] == 4 and out["n_items_in_alpha"] == 4

    def test_min_pair_n_threshold_is_honoured(self) -> None:
        df = _rotating_gaps(n=400, k=8)
        strict = psy_ctt(df, min_pair_n=400)  # unreachable overlap
        assert strict["estimable"] is False
        assert "400" in strict["not_estimable_reason"]
        assert strict["pair_n"]["threshold"] == 400

    def test_deterministic(self) -> None:
        df = _rotating_gaps(n=300, k=7)
        a, b = psy_ctt(df), psy_ctt(df)
        assert json.dumps(a, allow_nan=False) == json.dumps(b, allow_nan=False)


class TestNoBareNaNEver:
    """results.json is written with json.dump; a NaN there is both invalid
    JSON and unstateable prose. allow_nan=False is the structural pin."""

    @pytest.mark.parametrize("name", ["complete", "rotating", "disjoint",
                                      "single", "empty"])
    def test_strict_json_serializable(self, name: str) -> None:
        if name == "complete":
            df = _complete_likert()
        elif name == "rotating":
            df = _rotating_gaps()
        elif name == "disjoint":
            df = pd.DataFrame({"a": [1.0] * 40 + [np.nan] * 40,
                               "b": [np.nan] * 40 + [0.0, 1.0] * 20})
        elif name == "single":
            df = pd.DataFrame({"i0": [1.0, 2.0] * 30})
        else:
            df = pd.DataFrame({"i0": [np.nan] * 40, "i1": [np.nan] * 40})

        out = psy_ctt(df)
        json.dumps(out, allow_nan=False)  # raises ValueError on any NaN
        assert isinstance(out["summary"], str) and out["summary"]
        assert isinstance(out["estimable"], bool)


@pytest.mark.skipif(not REAL_ITEMS.exists(),
                    reason="arc_p validation run artifacts not present")
class TestRealAssistmentsMatrix:
    """The matrix that produced the all-NaN P1 block in the Arc P run."""

    def _load(self) -> pd.DataFrame:
        df = pd.read_csv(REAL_ITEMS, encoding="utf-8")
        return df.drop(columns=["user_id"])

    def test_alpha_is_recovered_with_full_provenance(self) -> None:
        df = self._load()
        assert df.shape == (1586, 47)
        assert _listwise_rows(df) == 0  # why the old helper returned NaN

        out = psy_ctt(df)

        assert out["estimable"] is True
        assert out["n_complete"] == 0
        assert out["matrix_fill_rate"] == pytest.approx(0.276, abs=0.005)
        assert out["cronbach_alpha"] == pytest.approx(0.894, abs=0.01)
        assert out["n_items_in_alpha"] == 47
        assert out["pair_n"]["min"] >= 30
        assert out["pair_n"]["n_pairs"] == 47 * 46 // 2
        assert all(r["mean"] is not None and r["item_total_r"] is not None
                   for r in out["items"])
        json.dumps(out, allow_nan=False)

    def test_non_psd_pairwise_matrix_is_disclosed(self) -> None:
        # pairwise covariances come from different, tutor-assigned
        # subsamples, so the assembled matrix is indefinite here; the
        # paper must be able to say that instead of quoting .894 flatly.
        out = psy_ctt(self._load())
        assert out["covariance_psd"] is False
        assert out["min_eigenvalue"] < 0
        assert any("positive semi-definite" in c for c in out["caveats"])
        assert "approximate" in out["summary"]
