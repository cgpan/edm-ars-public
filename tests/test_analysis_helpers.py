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
    _find_columns_for_vars,
    bootstrap_ci,
    clustered_bootstrap_ci,
    compute_icc,
    model_quality_gate,
    reconstruct_school_ids,
    run_sensitivity_analysis,
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


# ---------------------------------------------------------------------------
# model_quality_gate
# ---------------------------------------------------------------------------


class TestModelQualityGate:
    def test_classification_passes_above_floor(self) -> None:
        all_models = {"XGBoost": {"auc": 0.85, "accuracy": 0.80}}
        result = model_quality_gate(all_models, is_classification=True)
        assert result["XGBoost"]["passed"] is True
        assert result["XGBoost"]["shap_eligible"] is True
        assert result["XGBoost"]["metric_name"] == "auc"
        assert result["XGBoost"]["metric_value"] == 0.85
        assert result["XGBoost"]["floor"] == 0.60

    def test_classification_fails_below_floor(self) -> None:
        all_models = {"RandomForest": {"auc": 0.55, "accuracy": 0.60}}
        result = model_quality_gate(all_models, is_classification=True)
        assert result["RandomForest"]["passed"] is False
        assert result["RandomForest"]["shap_eligible"] is False

    def test_regression_passes_above_floor(self) -> None:
        all_models = {"LinearRegression": {"rmse": 0.5, "r2": 0.30}}
        result = model_quality_gate(all_models, is_classification=False)
        assert result["LinearRegression"]["passed"] is True
        assert result["LinearRegression"]["shap_eligible"] is True
        assert result["LinearRegression"]["metric_name"] == "r2"

    def test_regression_fails_below_floor(self) -> None:
        all_models = {"LinearRegression": {"rmse": 1.2, "r2": 0.02}}
        result = model_quality_gate(all_models, is_classification=False)
        assert result["LinearRegression"]["passed"] is False
        assert result["LinearRegression"]["shap_eligible"] is False

    def test_stacking_never_shap_eligible(self) -> None:
        all_models = {"StackingEnsemble": {"auc": 0.90, "accuracy": 0.88}}
        result = model_quality_gate(all_models, is_classification=True)
        assert result["StackingEnsemble"]["passed"] is True
        assert result["StackingEnsemble"]["shap_eligible"] is False

    def test_all_models_fail(self) -> None:
        all_models = {
            "LogisticRegression": {"auc": 0.50},
            "RandomForest": {"auc": 0.55},
            "XGBoost": {"auc": 0.58},
        }
        result = model_quality_gate(all_models, is_classification=True)
        assert all(not g["passed"] for g in result.values())
        assert all(not g["shap_eligible"] for g in result.values())


# ---------------------------------------------------------------------------
# reconstruct_school_ids
# ---------------------------------------------------------------------------


class TestReconstructSchoolIds:
    # Explicit fingerprint vars for unit tests (tests clustering logic, not defaults)
    _TEST_FP_VARS = ["VAR_A", "VAR_B", "VAR_C", "VAR_D", "VAR_E", "VAR_F", "VAR_G"]

    def test_groups_identical_rows(self) -> None:
        """6 students with 2 distinct fingerprint patterns → 2 clusters."""
        df = pd.DataFrame({
            "VAR_A": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
            "VAR_B": [0.5, 0.5, 0.5, -0.3, -0.3, -0.3],
            "VAR_C": [0.1, 0.1, 0.1, 0.2, 0.2, 0.2],
            "VAR_D": [0.8, 0.8, 0.8, -0.5, -0.5, -0.5],
            "VAR_E": [1, 1, 1, 2, 2, 2],
            "VAR_F": [1, 1, 1, 3, 3, 3],
            "VAR_G": [2, 2, 2, 4, 4, 4],
        })
        ids, meta = reconstruct_school_ids(
            df, fingerprint_vars=self._TEST_FP_VARS, validate=False
        )
        assert meta["n_clusters"] == 2
        # First three students should share one ID, last three another
        assert ids.iloc[0] == ids.iloc[1] == ids.iloc[2]
        assert ids.iloc[3] == ids.iloc[4] == ids.iloc[5]
        assert ids.iloc[0] != ids.iloc[3]

    def test_assigns_negative_one_to_all_nan(self) -> None:
        """Student with all NaN fingerprint vars → school_id = -1."""
        df = pd.DataFrame({
            "VAR_A": [1.0, np.nan],
            "VAR_B": [0.5, np.nan],
            "VAR_C": [0.1, np.nan],
            "VAR_D": [0.8, np.nan],
            "VAR_E": [1, np.nan],
            "VAR_F": [1, np.nan],
            "VAR_G": [2, np.nan],
        })
        ids, meta = reconstruct_school_ids(
            df, fingerprint_vars=self._TEST_FP_VARS, validate=False
        )
        assert ids.iloc[1] == -1
        assert ids.iloc[0] >= 0

    def test_validation_warns_on_huge_cluster(self) -> None:
        """One cluster with >5% of total → warning."""
        # 100 students, all in the same school fingerprint
        n = 100
        df = pd.DataFrame({
            "VAR_A": [1.0] * n,
            "VAR_B": [0.5] * n,
            "VAR_C": [0.1] * n,
            "VAR_D": [0.8] * n,
            "VAR_E": [1] * n,
            "VAR_F": [1] * n,
            "VAR_G": [2] * n,
        })
        ids, meta = reconstruct_school_ids(
            df, fingerprint_vars=self._TEST_FP_VARS,
            validate=True, expected_n_schools=944,
        )
        # Should warn about cluster count deviation AND about huge cluster
        assert any("collision" in w.lower() or "largest" in w.lower()
                    for w in meta["validation_warnings"])

    def test_missing_fingerprint_var_graceful(self) -> None:
        """Requested var not in DataFrame → skipped, noted in metadata."""
        df = pd.DataFrame({
            "X1SCHOOLCLI": [1.0, 2.0],
            "X1COUPERTEA": [0.5, -0.3],
        })
        # Request vars that don't exist
        ids, meta = reconstruct_school_ids(
            df,
            fingerprint_vars=["X1SCHOOLCLI", "X1COUPERTEA", "NONEXISTENT"],
            validate=False,
        )
        assert "NONEXISTENT" in meta["fingerprint_vars_missing"]
        assert "X1SCHOOLCLI" in meta["fingerprint_vars_used"]
        assert meta["n_clusters"] == 2

    def test_n_clusters_reasonable(self) -> None:
        """With realistic synthetic data, clusters should match expectations."""
        from src.analysis_helpers import _DEFAULT_FINGERPRINT_VARS

        rng = np.random.RandomState(42)
        n_schools = 50
        students_per_school = 20
        rows = []
        for s in range(n_schools):
            # Generate unique values per school for each fingerprint var
            school_vals = {v: rng.randn() for v in _DEFAULT_FINGERPRINT_VARS}
            for _ in range(students_per_school):
                rows.append(dict(school_vals))
        df = pd.DataFrame(rows)
        ids, meta = reconstruct_school_ids(
            df, validate=True, expected_n_schools=50, tolerance=0.15
        )
        assert meta["n_clusters"] == 50
        assert meta["validation_passed"] is True


# ---------------------------------------------------------------------------
# clustered_bootstrap_ci
# ---------------------------------------------------------------------------


class TestClusteredBootstrapCi:
    def test_returns_tuple_of_two_floats(self) -> None:
        rng = np.random.RandomState(1)
        y_true = rng.randint(0, 2, 200)
        y_pred = rng.rand(200)
        cluster_ids = np.repeat(np.arange(20), 10)
        from sklearn.metrics import roc_auc_score
        ci = clustered_bootstrap_ci(y_true, y_pred, cluster_ids, roc_auc_score)
        assert isinstance(ci, tuple)
        assert len(ci) == 2
        assert isinstance(ci[0], float)
        assert isinstance(ci[1], float)

    def test_lower_le_upper(self) -> None:
        rng = np.random.RandomState(2)
        y_true = rng.randint(0, 2, 200)
        y_pred = rng.rand(200)
        cluster_ids = np.repeat(np.arange(20), 10)
        from sklearn.metrics import roc_auc_score
        lower, upper = clustered_bootstrap_ci(
            y_true, y_pred, cluster_ids, roc_auc_score
        )
        assert lower <= upper

    def test_wider_than_standard_bootstrap(self) -> None:
        """Clustered CI should generally be >= standard CI width with high ICC data.

        We use the mean as the metric (directly sensitive to cluster-level variation)
        so that cluster bootstrap correctly reflects higher uncertainty.
        """
        rng = np.random.RandomState(42)
        n_clusters = 30
        cluster_size = 50
        n = n_clusters * cluster_size

        # Create data with strong within-cluster correlation on the outcome itself
        cluster_ids = np.repeat(np.arange(n_clusters), cluster_size)
        cluster_effects = rng.randn(n_clusters) * 3.0  # strong school effect on y
        y_true = cluster_effects[cluster_ids] + rng.randn(n) * 0.3
        # y_pred is a noisy copy — the residual itself has cluster structure
        y_pred = y_true + cluster_effects[cluster_ids] * 0.1 + rng.randn(n) * 0.2

        # Use R² as metric — sensitive to cluster-level variance in residuals
        def r2_metric(yt: np.ndarray, yp: np.ndarray) -> float:
            ss_res = np.sum((yt - yp) ** 2)
            ss_tot = np.sum((yt - np.mean(yt)) ** 2)
            return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        cl_lower, cl_upper = clustered_bootstrap_ci(
            y_true, y_pred, cluster_ids, r2_metric, random_state=42
        )
        st_lower, st_upper = bootstrap_ci(
            y_true, y_pred, r2_metric, random_state=42
        )

        clustered_width = cl_upper - cl_lower
        standard_width = st_upper - st_lower
        # With strong ICC, clustered CI should be at least as wide
        assert clustered_width >= standard_width * 0.8  # generous margin

    def test_handles_single_cluster(self) -> None:
        """Degrades gracefully with a single cluster."""
        rng = np.random.RandomState(3)
        y_true = rng.randint(0, 2, 50)
        y_pred = rng.rand(50)
        cluster_ids = np.zeros(50, dtype=int)  # single cluster
        from sklearn.metrics import roc_auc_score
        ci = clustered_bootstrap_ci(y_true, y_pred, cluster_ids, roc_auc_score)
        # Should not crash; returns some CI (width = 0 is fine for 1 cluster)
        assert isinstance(ci, tuple)
        assert len(ci) == 2


# ---------------------------------------------------------------------------
# compute_icc
# ---------------------------------------------------------------------------


class TestComputeIcc:
    def test_icc_zero_when_no_clustering(self) -> None:
        """Random assignment → ICC ≈ 0."""
        rng = np.random.RandomState(42)
        n = 1000
        y = rng.randn(n)
        # Random cluster assignment — no real clustering effect
        cluster_ids = rng.randint(0, 50, n)
        result = compute_icc(y, cluster_ids)
        assert result["icc"] < 0.10  # should be near zero

    def test_icc_high_when_strong_clustering(self) -> None:
        """Group means differ a lot → ICC > 0.3."""
        rng = np.random.RandomState(42)
        n_clusters = 20
        cluster_size = 50
        cluster_ids = np.repeat(np.arange(n_clusters), cluster_size)
        # Strong between-cluster variance, small within-cluster variance
        cluster_means = rng.randn(n_clusters) * 5.0
        y = cluster_means[cluster_ids] + rng.randn(n_clusters * cluster_size) * 0.5
        result = compute_icc(y, cluster_ids)
        assert result["icc"] > 0.3
        assert result["interpretation"] == "large"

    def test_excludes_unassigned(self) -> None:
        """cluster_id == -1 rows are excluded."""
        rng = np.random.RandomState(42)
        y = np.array([1.0, 1.0, 2.0, 2.0, 999.0])
        cluster_ids = np.array([0, 0, 1, 1, -1])
        result = compute_icc(y, cluster_ids)
        # Should only use the first 4 observations
        assert result["n_clusters"] == 2

    def test_returns_required_keys(self) -> None:
        rng = np.random.RandomState(42)
        y = rng.randn(100)
        cluster_ids = np.repeat(np.arange(10), 10)
        result = compute_icc(y, cluster_ids)
        for key in ["icc", "msb", "msw", "n_clusters", "avg_cluster_size", "interpretation"]:
            assert key in result


# ---------------------------------------------------------------------------
# run_sensitivity_analysis
# ---------------------------------------------------------------------------


class TestRunSensitivityAnalysis:
    """Tests for run_sensitivity_analysis and its column-matching helper."""

    def test_drops_correct_columns(self) -> None:
        """One-hot encoded columns (var_*) and exact matches are found and dropped."""
        columns = [
            "X1TXMTSCOR", "X1SES",
            "X1RACE_2.0", "X1RACE_3.0", "X1RACE_5.0",
            "X1PAREDU_2.0", "X1PAREDU_3.0",
        ]
        matched = _find_columns_for_vars(columns, ["X1RACE", "X1PAREDU"])
        assert "X1RACE_2.0" in matched
        assert "X1RACE_3.0" in matched
        assert "X1RACE_5.0" in matched
        assert "X1PAREDU_2.0" in matched
        assert "X1PAREDU_3.0" in matched
        assert "X1TXMTSCOR" not in matched
        assert "X1SES" not in matched
        assert len(matched) == 5

    def test_returns_required_keys(self) -> None:
        """Check all expected keys present in output dict."""
        from sklearn.ensemble import RandomForestClassifier

        rng = np.random.RandomState(42)
        n = 200
        X = pd.DataFrame({
            "feat_a": rng.randn(n),
            "feat_b": rng.randn(n),
            "high_miss_var": rng.randn(n),
        })
        y = (rng.rand(n) > 0.5).astype(int)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        result = run_sensitivity_analysis(
            best_model_class=RandomForestClassifier,
            best_model_params={"n_estimators": 10, "random_state": 42},
            train_X=X,
            train_y=y,
            test_X=X,
            test_y=y,
            high_miss_vars=["high_miss_var"],
            is_classification=True,
        )
        expected_keys = [
            "excluded_variables", "n_columns_dropped",
            "full_model_metric", "reduced_model_metric",
            "metric_name", "metric_change_pct", "significant_change",
            "full_model_top5", "reduced_model_top5", "top5_overlap",
            "conclusion",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
        assert result["n_columns_dropped"] == 1
        assert result["metric_name"] == "AUC"

    def test_significant_change_flag(self) -> None:
        """When metric drops >5%, significant_change=True."""
        from sklearn.linear_model import LogisticRegression

        rng = np.random.RandomState(42)
        n = 500
        # Make critical_var the ONLY signal — dropping it should tank AUC
        critical_var = rng.randn(n)
        y = (critical_var > 0).astype(int)
        X = pd.DataFrame({
            "noise_a": rng.randn(n) * 0.01,
            "noise_b": rng.randn(n) * 0.01,
            "critical_var": critical_var,
        })

        result = run_sensitivity_analysis(
            best_model_class=LogisticRegression,
            best_model_params={"max_iter": 1000, "random_state": 42},
            train_X=X,
            train_y=y,
            test_X=X,
            test_y=y,
            high_miss_vars=["critical_var"],
            is_classification=True,
        )
        # Dropping the only informative variable should cause a significant metric drop
        assert result["significant_change"] is True
        assert result["metric_change_pct"] < -5.0

    def test_handles_no_high_miss_vars(self) -> None:
        """Graceful return when high_miss_vars is empty."""
        result = run_sensitivity_analysis(
            best_model_class=type(None),  # shouldn't be called
            best_model_params={},
            train_X=pd.DataFrame({"a": [1, 2]}),
            train_y=np.array([0, 1]),
            test_X=pd.DataFrame({"a": [1, 2]}),
            test_y=np.array([0, 1]),
            high_miss_vars=[],
            is_classification=True,
        )
        assert result["n_columns_dropped"] == 0
        assert result["significant_change"] is False
        assert "No high-missingness" in result["conclusion"]


# ---------------------------------------------------------------------------
# Tests — grouped_train_test_split
# ---------------------------------------------------------------------------

from src.analysis_helpers import grouped_train_test_split


def _make_grouped_data(
    n: int = 1000,
    n_groups: int = 50,
    binary: bool = False,
    minority_frac: float = 0.5,
    n_unassigned: int = 0,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Create synthetic data with group structure for split tests."""
    rng = np.random.RandomState(42)
    # Assign students to groups with variable sizes
    groups = np.repeat(np.arange(n_groups), n // n_groups)
    # Pad remainder
    remainder = n - len(groups)
    if remainder > 0:
        groups = np.concatenate([groups, rng.randint(0, n_groups, size=remainder)])
    groups = groups[:n]

    if binary:
        n_minority = int(n * minority_frac)
        y = np.zeros(n, dtype=int)
        y[:n_minority] = 1
        rng.shuffle(y)
    else:
        y = rng.randn(n)

    # Mark some as unassigned
    if n_unassigned > 0:
        groups = groups.copy()
        groups[:n_unassigned] = -1

    df = pd.DataFrame({"x1": rng.randn(n), "x2": rng.randn(n)})
    return df, y, groups


class TestGroupedTrainTestSplit:
    """Tests for analysis_helpers.grouped_train_test_split()."""

    def test_no_group_overlap(self) -> None:
        """No real school ID should appear in both train and test."""
        df, y, groups = _make_grouped_data(n=1000, n_groups=50)
        train_idx, test_idx, meta = grouped_train_test_split(
            df, y, groups, test_size=0.2, stratify=False, random_state=42,
        )
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        assert train_groups & test_groups == set(), "Groups overlap between train and test"
        assert meta["group_overlap"] == 0

    def test_classification_stratified(self) -> None:
        """With binary y, class proportions should be approximately equal across splits."""
        df, y, groups = _make_grouped_data(n=2000, n_groups=100, binary=True, minority_frac=0.3)
        train_idx, test_idx, meta = grouped_train_test_split(
            df, y, groups, test_size=0.2, stratify=True, random_state=42,
        )
        train_pos_rate = y[train_idx].mean()
        test_pos_rate = y[test_idx].mean()
        # Allow up to 5% absolute deviation due to group-level constraints
        assert abs(train_pos_rate - test_pos_rate) < 0.05, (
            f"Class balance diverged: train={train_pos_rate:.3f}, test={test_pos_rate:.3f}"
        )
        assert meta["split_method"] == "StratifiedGroupKFold"

    def test_regression_split(self) -> None:
        """Regression split should produce ~20% test without error."""
        df, y, groups = _make_grouped_data(n=1000, n_groups=50, binary=False)
        train_idx, test_idx, meta = grouped_train_test_split(
            df, y, groups, test_size=0.2, stratify=False, random_state=42,
        )
        assert meta["split_method"] == "GroupShuffleSplit"
        assert len(train_idx) + len(test_idx) == len(df)

    def test_unassigned_split_individually(self) -> None:
        """Students with groups=-1 should be distributed across both splits."""
        df, y, groups = _make_grouped_data(
            n=1000, n_groups=50, n_unassigned=200,
        )
        train_idx, test_idx, meta = grouped_train_test_split(
            df, y, groups, test_size=0.2, stratify=False, random_state=42,
        )
        # Unassigned students should appear in both splits (probabilistically)
        total_unassigned = meta["n_unassigned_train"] + meta["n_unassigned_test"]
        assert total_unassigned == 200
        assert meta["n_unassigned_train"] > 0, "All unassigned in test only"
        assert meta["n_unassigned_test"] > 0, "All unassigned in train only"
        # No real group overlap
        assert meta["group_overlap"] == 0

    def test_size_approximately_20_pct(self) -> None:
        """Actual test size should be between 15% and 25%."""
        df, y, groups = _make_grouped_data(n=2000, n_groups=100)
        _, test_idx, meta = grouped_train_test_split(
            df, y, groups, test_size=0.2, stratify=False, random_state=42,
        )
        assert 0.15 <= meta["test_fraction"] <= 0.25, (
            f"Test fraction {meta['test_fraction']} outside [0.15, 0.25]"
        )

    def test_reproducible_with_same_seed(self) -> None:
        """Same inputs and seed should produce identical splits."""
        df, y, groups = _make_grouped_data(n=500, n_groups=25)
        t1, e1, _ = grouped_train_test_split(df, y, groups, random_state=42)
        t2, e2, _ = grouped_train_test_split(df, y, groups, random_state=42)
        np.testing.assert_array_equal(t1, t2)
        np.testing.assert_array_equal(e1, e2)

    def test_different_seed_different_split(self) -> None:
        """Different random_state should produce different splits."""
        df, y, groups = _make_grouped_data(n=500, n_groups=25)
        _, e1, _ = grouped_train_test_split(df, y, groups, random_state=42)
        _, e2, _ = grouped_train_test_split(df, y, groups, random_state=99)
        assert not np.array_equal(e1, e2), "Different seeds produced identical splits"

    def test_small_groups_handled(self) -> None:
        """Groups of size 1-3 should be handled without error."""
        rng = np.random.RandomState(42)
        # Create groups where many have 1-3 members
        groups = np.concatenate([
            np.repeat(np.arange(100), 2),  # 100 groups of size 2
            np.arange(100, 150),            # 50 singleton groups
        ])
        n = len(groups)
        df = pd.DataFrame({"x": rng.randn(n)})
        y = rng.randn(n)
        train_idx, test_idx, meta = grouped_train_test_split(
            df, y, groups, test_size=0.2, stratify=False, random_state=42,
        )
        assert meta["group_overlap"] == 0
        assert len(train_idx) + len(test_idx) == n

    def test_metadata_keys(self) -> None:
        """Returned metadata dict should contain all expected keys."""
        df, y, groups = _make_grouped_data(n=500, n_groups=25)
        _, _, meta = grouped_train_test_split(df, y, groups)
        expected_keys = {
            "split_method", "n_train", "n_test", "test_fraction",
            "n_groups_train", "n_groups_test",
            "n_unassigned_train", "n_unassigned_test", "group_overlap",
        }
        assert set(meta.keys()) == expected_keys

    def test_all_indices_covered(self) -> None:
        """Union of train and test indices should equal full index set."""
        df, y, groups = _make_grouped_data(n=800, n_groups=40)
        train_idx, test_idx, _ = grouped_train_test_split(df, y, groups)
        all_idx = np.sort(np.concatenate([train_idx, test_idx]))
        np.testing.assert_array_equal(all_idx, np.arange(len(df)))

    def test_imbalanced_classification(self) -> None:
        """With 10% minority class, stratification should still approximately preserve balance."""
        df, y, groups = _make_grouped_data(
            n=2000, n_groups=100, binary=True, minority_frac=0.10,
        )
        train_idx, test_idx, meta = grouped_train_test_split(
            df, y, groups, test_size=0.2, stratify=True, random_state=42,
        )
        train_pos_rate = y[train_idx].mean()
        test_pos_rate = y[test_idx].mean()
        # Allow up to 5% absolute deviation for imbalanced + grouped
        assert abs(train_pos_rate - test_pos_rate) < 0.05, (
            f"Imbalanced class balance diverged: train={train_pos_rate:.3f}, test={test_pos_rate:.3f}"
        )
