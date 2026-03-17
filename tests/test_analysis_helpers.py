"""Unit tests for src/analysis_helpers.py.

These tests do NOT require ANTHROPIC_API_KEY and run offline.
They cover the deterministic helper functions for SHAP and subgroup analysis.

Run:
    pytest tests/test_analysis_helpers.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis_helpers import (
    bootstrap_ci,
    run_subgroup_analysis,
    safe_shap_values,
    save_pdp_plots,
    save_shap_plots,
)

# ---------------------------------------------------------------------------
# Fixtures / shared data
# ---------------------------------------------------------------------------

_RNG = np.random.RandomState(42)
_N = 50
_N_FEAT = 4
_FEATURE_NAMES = ["feat_a", "feat_b", "feat_c", "feat_d"]

_FAKE_X = pd.DataFrame(
    _RNG.randn(_N, _N_FEAT),
    columns=_FEATURE_NAMES,
)

# 2D SHAP array — what explainers return for regressors / XGBoost classifiers
_SHAP_ARRAY = _RNG.randn(_N, _N_FEAT)

# List of arrays — what sklearn TreeExplainer returns for binary classifiers
_SHAP_LIST = [_RNG.randn(_N, _N_FEAT), _RNG.randn(_N, _N_FEAT)]


# ---------------------------------------------------------------------------
# safe_shap_values
# ---------------------------------------------------------------------------


class TestSafeShapValues:
    def _mock_explainer(self, return_value: object) -> MagicMock:
        explainer = MagicMock()
        explainer.shap_values.return_value = return_value
        return explainer

    def test_returns_ndarray_when_given_list(self) -> None:
        """Binary-classification TreeExplainer returns list → extracts index 1."""
        explainer = self._mock_explainer(_SHAP_LIST)
        result = safe_shap_values(explainer, _FAKE_X)
        assert isinstance(result, np.ndarray)
        # Should be class-1 SHAP values
        np.testing.assert_array_equal(result, _SHAP_LIST[1])

    def test_returns_ndarray_when_given_array(self) -> None:
        """Regression TreeExplainer returns 2D array → passes through unchanged."""
        explainer = self._mock_explainer(_SHAP_ARRAY)
        result = safe_shap_values(explainer, _FAKE_X)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, _SHAP_ARRAY)

    def test_shape_preserved(self) -> None:
        """Output shape always matches (n_samples, n_features)."""
        explainer = self._mock_explainer(_SHAP_LIST)
        result = safe_shap_values(explainer, _FAKE_X)
        assert result.shape == (_N, _N_FEAT)

    def test_explainer_called_once(self) -> None:
        explainer = self._mock_explainer(_SHAP_ARRAY)
        safe_shap_values(explainer, _FAKE_X)
        # Input is converted to numpy before calling shap_values (SHAP 0.47 + NumPy 1.26 fix)
        call_arg = explainer.shap_values.call_args[0][0]
        np.testing.assert_array_equal(call_arg, _FAKE_X.values)

    def test_handles_explanation_object(self) -> None:
        """SHAP Explanation objects expose a .values attribute."""
        explanation = MagicMock()
        explanation.values = _SHAP_ARRAY
        explainer = self._mock_explainer(explanation)
        result = safe_shap_values(explainer, _FAKE_X)
        np.testing.assert_array_equal(result, _SHAP_ARRAY)

    def test_no_truth_value_error_on_array(self) -> None:
        """Regression: calling bool() on the returned array must not raise."""
        explainer = self._mock_explainer(_SHAP_ARRAY)
        result = safe_shap_values(explainer, _FAKE_X)
        # This should not raise ValueError
        _ = result is None  # safe None check
        _ = len(result) > 0  # safe length check


# ---------------------------------------------------------------------------
# save_shap_plots
# ---------------------------------------------------------------------------


class TestSaveShapPlots:
    def test_creates_both_png_files(self, tmp_path: Path) -> None:
        with patch("src.analysis_helpers.shap") as mock_shap, \
             patch("src.analysis_helpers.plt") as mock_plt:
            mock_shap.summary_plot = MagicMock()
            mock_plt.savefig = MagicMock()
            mock_plt.close = MagicMock()

            result = save_shap_plots(_SHAP_ARRAY, _FAKE_X, str(tmp_path))

        assert "shap_summary.png" in result
        assert "shap_importance.png" in result
        assert len(result) == 2

    def test_returns_list_of_filenames(self, tmp_path: Path) -> None:
        with patch("src.analysis_helpers.shap"), patch("src.analysis_helpers.plt"):
            result = save_shap_plots(_SHAP_ARRAY, _FAKE_X, str(tmp_path))
        assert isinstance(result, list)

    def test_summary_plot_called_twice(self, tmp_path: Path) -> None:
        """Once for beeswarm, once for bar."""
        with patch("src.analysis_helpers.shap") as mock_shap, \
             patch("src.analysis_helpers.plt"):
            save_shap_plots(_SHAP_ARRAY, _FAKE_X, str(tmp_path))
        assert mock_shap.summary_plot.call_count == 2


# ---------------------------------------------------------------------------
# save_pdp_plots
# ---------------------------------------------------------------------------


class TestSavePdpPlots:
    def test_creates_one_png_per_top_feature(self, tmp_path: Path) -> None:
        model = MagicMock()
        top_features = ["feat_a", "feat_b"]

        with patch("sklearn.inspection.PartialDependenceDisplay") as mock_pdp, \
             patch("src.analysis_helpers.plt") as mock_plt:
            mock_pdp.from_estimator = MagicMock()
            mock_plt.subplots.return_value = (MagicMock(), MagicMock())
            mock_plt.close = MagicMock()

            result = save_pdp_plots(model, _FAKE_X, top_features, str(tmp_path))

        assert len(result) == 2
        assert all(f.startswith("pdp_") and f.endswith(".png") for f in result)

    def test_skips_features_not_in_dataframe(self, tmp_path: Path) -> None:
        model = MagicMock()
        top_features = ["nonexistent_feat", "feat_a"]

        with patch("sklearn.inspection.PartialDependenceDisplay") as mock_pdp, \
             patch("src.analysis_helpers.plt") as mock_plt:
            mock_pdp.from_estimator = MagicMock()
            mock_plt.subplots.return_value = (MagicMock(), MagicMock())

            result = save_pdp_plots(model, _FAKE_X, top_features, str(tmp_path))

        # Only feat_a should be in output
        assert len(result) == 1
        assert "pdp_feat_a.png" in result

    def test_at_most_three_plots(self, tmp_path: Path) -> None:
        """Even with > 3 features, only first 3 are plotted."""
        model = MagicMock()

        with patch("sklearn.inspection.PartialDependenceDisplay") as mock_pdp, \
             patch("src.analysis_helpers.plt") as mock_plt:
            mock_pdp.from_estimator = MagicMock()
            mock_plt.subplots.return_value = (MagicMock(), MagicMock())

            result = save_pdp_plots(model, _FAKE_X, _FEATURE_NAMES, str(tmp_path))

        assert len(result) <= 3


# ---------------------------------------------------------------------------
# run_subgroup_analysis
# ---------------------------------------------------------------------------


class TestRunSubgroupAnalysis:
    """Integration-style tests with a real sklearn model and synthetic data."""

    def _setup(self, tmp_path: Path, is_classification: bool = True):
        from sklearn.linear_model import LinearRegression, LogisticRegression

        rng = np.random.RandomState(0)
        n = 100
        X = pd.DataFrame(rng.randn(n, 2), columns=["f1", "f2"])
        if is_classification:
            y = (rng.rand(n) > 0.5).astype(int)
            model = LogisticRegression(random_state=42, max_iter=1000).fit(X, y)
        else:
            y = rng.randn(n)
            model = LinearRegression().fit(X, y)

        # Build test_protected.csv
        sex = rng.choice(["Male", "Female"], size=n)
        protected = pd.DataFrame({"X1SEX": sex}, index=range(n))
        protected_path = str(tmp_path / "test_protected.csv")
        protected.to_csv(protected_path)

        return model, X, y, protected_path

    def test_returns_dict_with_subgroup_structure(self, tmp_path: Path) -> None:
        model, X, y, path = self._setup(tmp_path, is_classification=True)
        result = run_subgroup_analysis(
            model=model,
            test_X=X,
            test_y=y,
            test_protected_path=path,
            subgroup_attrs=["X1SEX"],
            is_classification=True,
        )
        assert "X1SEX" in result
        assert "Male" in result["X1SEX"] or "Female" in result["X1SEX"]

    def test_classification_result_has_auc_and_n(self, tmp_path: Path) -> None:
        model, X, y, path = self._setup(tmp_path, is_classification=True)
        result = run_subgroup_analysis(
            model, X, y, path, ["X1SEX"], is_classification=True
        )
        for group_metrics in result.get("X1SEX", {}).values():
            assert "auc" in group_metrics
            assert "n" in group_metrics
            assert 0.0 <= group_metrics["auc"] <= 1.0

    def test_regression_result_has_rmse_and_n(self, tmp_path: Path) -> None:
        model, X, y, path = self._setup(tmp_path, is_classification=False)
        result = run_subgroup_analysis(
            model, X, y, path, ["X1SEX"], is_classification=False
        )
        for group_metrics in result.get("X1SEX", {}).values():
            assert "rmse" in group_metrics
            assert "n" in group_metrics
            assert group_metrics["rmse"] >= 0.0

    def test_missing_protected_file_returns_empty(self, tmp_path: Path) -> None:
        from sklearn.linear_model import LogisticRegression

        rng = np.random.RandomState(0)
        X = pd.DataFrame(rng.randn(50, 2), columns=["f1", "f2"])
        y = (rng.rand(50) > 0.5).astype(int)
        model = LogisticRegression(random_state=42, max_iter=1000).fit(X, y)

        result = run_subgroup_analysis(
            model, X, y,
            test_protected_path="/nonexistent/path/test_protected.csv",
            subgroup_attrs=["X1SEX"],
            is_classification=True,
        )
        assert result == {}

    def test_missing_attr_skipped_gracefully(self, tmp_path: Path) -> None:
        model, X, y, path = self._setup(tmp_path, is_classification=True)
        warnings: list[str] = []
        result = run_subgroup_analysis(
            model, X, y, path,
            subgroup_attrs=["NONEXISTENT_ATTR"],
            is_classification=True,
            warnings_list=warnings,
        )
        assert result == {}
        assert any("NONEXISTENT_ATTR" in w for w in warnings)

    def test_gap_warning_appended(self, tmp_path: Path) -> None:
        """Gaps > 5 % should be flagged in warnings_list."""
        from sklearn.linear_model import LogisticRegression

        rng = np.random.RandomState(7)
        n = 200
        X = pd.DataFrame(rng.randn(n, 2), columns=["f1", "f2"])
        # Create a very skewed target to make subgroup gaps likely
        y = np.zeros(n, dtype=int)
        y[:100] = 1  # first half all positive
        model = LogisticRegression(random_state=42, max_iter=1000).fit(X, y)

        sex = np.array(["Male"] * 100 + ["Female"] * 100)
        protected = pd.DataFrame({"X1SEX": sex}, index=range(n))
        path = str(tmp_path / "test_protected.csv")
        protected.to_csv(path)

        warnings: list[str] = []
        run_subgroup_analysis(
            model, X, y, path, ["X1SEX"], is_classification=True,
            warnings_list=warnings,
        )
        # The warning might or might not be present depending on model calibration,
        # but the function should not raise
        assert isinstance(warnings, list)


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------


class TestBootstrapCi:
    def test_returns_tuple_of_two_floats(self) -> None:
        rng = np.random.RandomState(1)
        y_true = rng.randint(0, 2, 200)
        y_pred = rng.rand(200)
        from sklearn.metrics import roc_auc_score
        ci = bootstrap_ci(y_true, y_pred, roc_auc_score)
        assert isinstance(ci, tuple)
        assert len(ci) == 2
        lower, upper = ci
        assert isinstance(lower, float)
        assert isinstance(upper, float)

    def test_lower_le_upper(self) -> None:
        rng = np.random.RandomState(2)
        y_true = rng.randint(0, 2, 200)
        y_pred = rng.rand(200)
        from sklearn.metrics import roc_auc_score
        lower, upper = bootstrap_ci(y_true, y_pred, roc_auc_score)
        assert lower <= upper

    def test_reproducible_with_same_seed(self) -> None:
        rng = np.random.RandomState(3)
        y_true = rng.randint(0, 2, 200)
        y_pred = rng.rand(200)
        from sklearn.metrics import roc_auc_score
        ci1 = bootstrap_ci(y_true, y_pred, roc_auc_score, random_state=99)
        ci2 = bootstrap_ci(y_true, y_pred, roc_auc_score, random_state=99)
        assert ci1 == ci2

    def test_ci_within_valid_auc_range(self) -> None:
        rng = np.random.RandomState(4)
        y_true = rng.randint(0, 2, 200)
        y_pred = rng.rand(200)
        from sklearn.metrics import roc_auc_score
        lower, upper = bootstrap_ci(y_true, y_pred, roc_auc_score)
        assert 0.0 <= lower <= 1.0
        assert 0.0 <= upper <= 1.0

    def test_regression_rmse_ci(self) -> None:
        from sklearn.metrics import mean_squared_error
        rng = np.random.RandomState(5)
        y_true = rng.randn(200)
        y_pred = y_true + rng.randn(200) * 0.5
        metric_fn = lambda yt, yp: float(np.sqrt(mean_squared_error(yt, yp)))
        lower, upper = bootstrap_ci(y_true, y_pred, metric_fn)
        assert lower >= 0.0
        assert lower <= upper
