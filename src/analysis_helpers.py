"""
analysis_helpers.py — Deterministic helpers for SHAP, subgroup analysis, and SMOTE.

This module is copied into the Analyst's output directory before code execution,
so LLM-generated code can `import analysis_helpers` without needing src.* access.

IMPORTANT: No imports from src.* — this file must run in an isolated subprocess
or Docker container where only stdlib + scientific Python packages are available.
"""
from __future__ import annotations

import os
from typing import Callable

import matplotlib
matplotlib.use("Agg")  # headless backend; must be set before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


# ---------------------------------------------------------------------------
# Class imbalance / SMOTE
# ---------------------------------------------------------------------------


def apply_smote(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    minority_threshold: float = 0.20,
    random_state: int = 42,
    k_neighbors: int = 5,
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Conditionally apply SMOTE to the training set if minority class < threshold.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels (1D array, binary 0/1).
        minority_threshold: Apply SMOTE when minority fraction < this value.
        random_state: Random seed for SMOTE reproducibility.
        k_neighbors: Number of nearest neighbours for SMOTE.

    Returns:
        Tuple of ``(X_resampled, y_resampled, metadata)``.
        ``metadata`` contains: ``applied``, ``minority_pct_before``,
        ``minority_pct_after``, ``n_before``, ``n_after``.
    """
    from imblearn.over_sampling import SMOTE  # local import: sandbox has imblearn

    y_arr = np.asarray(y_train).ravel()
    unique, counts = np.unique(y_arr, return_counts=True)
    class_dist = dict(zip(unique, counts))
    n_total = len(y_arr)

    minority_count = min(counts)
    minority_pct = minority_count / n_total

    metadata: dict = {
        "applied": False,
        "minority_pct_before": round(float(minority_pct), 4),
        "minority_pct_after": round(float(minority_pct), 4),
        "n_before": n_total,
        "n_after": n_total,
        "class_dist_before": {int(k): int(v) for k, v in class_dist.items()},
    }

    if minority_pct >= minority_threshold:
        # Balanced enough — return unchanged
        return X_train, y_arr, metadata

    smote = SMOTE(
        random_state=random_state,
        k_neighbors=k_neighbors,
    )
    X_res, y_res = smote.fit_resample(X_train, y_arr)

    # Preserve DataFrame structure (column names)
    X_res = pd.DataFrame(X_res, columns=X_train.columns)

    unique_after, counts_after = np.unique(y_res, return_counts=True)
    class_dist_after = dict(zip(unique_after, counts_after))
    minority_count_after = min(counts_after)
    n_after = len(y_res)

    metadata["applied"] = True
    metadata["minority_pct_after"] = round(float(minority_count_after / n_after), 4)
    metadata["n_after"] = n_after
    metadata["class_dist_after"] = {int(k): int(v) for k, v in class_dist_after.items()}

    return X_res, y_res, metadata


# ---------------------------------------------------------------------------
# Model quality gate
# ---------------------------------------------------------------------------


def model_quality_gate(
    all_models: dict,
    is_classification: bool,
    auc_floor: float = 0.60,
    r2_floor: float = 0.05,
) -> dict:
    """Assess each model and return a dict of model_name -> gate_result.

    gate_result = {
        "passed": bool,          # True if model passes quality floor
        "metric_name": str,      # "auc" or "r2"
        "metric_value": float,
        "floor": float,
        "shap_eligible": bool,   # True if model passes AND is not StackingEnsemble
    }

    For classification: a model passes if auc >= auc_floor.
    For regression: a model passes if r2 >= r2_floor.
    StackingEnsemble is never SHAP-eligible regardless of metric.

    Models that fail the gate:
    - Should still appear in the model comparison table
    - Must NOT have SHAP analysis run on them
    - Must be flagged in results.json warnings
    """
    results: dict = {}
    for model_name, metrics in all_models.items():
        if is_classification:
            metric_name = "auc"
            metric_value = float(metrics.get("auc", 0.0))
            floor = auc_floor
            passed = metric_value >= floor
        else:
            metric_name = "r2"
            metric_value = float(metrics.get("r2", 0.0))
            floor = r2_floor
            passed = metric_value >= floor

        is_stacking = model_name.lower().replace("_", "").replace(" ", "") in (
            "stackingensemble", "stacking",
        )
        shap_eligible = passed and not is_stacking

        results[model_name] = {
            "passed": passed,
            "metric_name": metric_name,
            "metric_value": round(metric_value, 4),
            "floor": floor,
            "shap_eligible": shap_eligible,
        }

    return results


# ---------------------------------------------------------------------------
# SHAP helpers
# ---------------------------------------------------------------------------


def safe_shap_values(explainer: object, X: pd.DataFrame) -> np.ndarray:
    """Call explainer.shap_values(X) and normalise to a single 2D numpy array.

    TreeExplainer for sklearn binary classifiers (e.g. RandomForestClassifier)
    returns a *list* of two arrays ``[class0_vals, class1_vals]``.  Calling
    ``if shap_values:`` or ``np.abs(shap_values).mean(axis=0)`` directly on that
    list raises ``ValueError: The truth value of an array with more than one
    element is ambiguous``.

    This function always returns a single 2D numpy array of shape
    ``(n_samples, n_features)``.

    Args:
        explainer: A fitted SHAP explainer (TreeExplainer, LinearExplainer, etc.).
        X: The sample matrix to explain (same columns as training data).

    Returns:
        2D numpy array of SHAP values, shape ``(n_samples, n_features)``.
    """
    # Convert to numpy before calling shap_values to avoid SHAP 0.47 + NumPy 1.26
    # "Multi-dimensional indexing (obj[:, None]) is no longer supported" error
    X_input = X.values if hasattr(X, "values") else np.asarray(X)

    vals = explainer.shap_values(X_input)

    # TreeExplainer for sklearn binary classifiers returns list [class0, class1]
    if isinstance(vals, list):
        # Use positive-class (index 1) SHAP values for binary classification
        vals = vals[1]

    # shap.Explanation objects (newer SHAP API) expose .values
    if hasattr(vals, "values"):
        vals = vals.values

    arr = np.asarray(vals, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def save_shap_plots(
    shap_vals: np.ndarray,
    X_shap: pd.DataFrame,
    output_dir: str,
) -> list[str]:
    """Save SHAP summary (beeswarm) and bar plots to *output_dir*.

    Args:
        shap_vals: 2D numpy array from :func:`safe_shap_values`.
        X_shap: The sample matrix used to compute shap_vals (same index/columns).
        output_dir: Directory where PNGs are written.

    Returns:
        List of filenames saved (relative, e.g. ``["shap_summary.png", ...]``).
    """
    feature_names = list(X_shap.columns)
    saved: list[str] = []

    # Beeswarm / summary plot
    shap.summary_plot(shap_vals, X_shap, feature_names=feature_names, show=False)
    path = os.path.join(output_dir, "shap_summary.png")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    saved.append("shap_summary.png")

    # Bar plot (mean |SHAP|)
    shap.summary_plot(
        shap_vals, X_shap, feature_names=feature_names, plot_type="bar", show=False
    )
    path = os.path.join(output_dir, "shap_importance.png")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    saved.append("shap_importance.png")

    return saved


def save_pdp_plots(
    model: object,
    X_train: pd.DataFrame,
    top_features: list[str],
    output_dir: str,
) -> list[str]:
    """Save partial dependence plots for up to the first 3 features in *top_features*.

    Args:
        model: A fitted sklearn-compatible model.
        X_train: Training feature matrix (used as the background distribution).
        top_features: Feature names ordered by importance (most important first).
        output_dir: Directory where PNGs are written.

    Returns:
        List of filenames saved.
    """
    from sklearn.inspection import PartialDependenceDisplay  # local import: only here

    saved: list[str] = []
    cols = list(X_train.columns)

    for feat in top_features[:3]:
        if feat not in cols:
            continue
        feat_idx = cols.index(feat)
        fig, ax = plt.subplots(figsize=(6, 4))
        try:
            PartialDependenceDisplay.from_estimator(model, X_train, [feat_idx], ax=ax)
            ax.set_title(f"Partial Dependence: {feat}")
        except Exception as exc:  # noqa: BLE001
            ax.set_title(f"PDP unavailable for {feat}: {exc}")

        # Sanitise feature name for use as a filename component
        safe_name = (
            feat.replace("/", "_")
            .replace(" ", "_")
            .replace(",", "")
            .replace("(", "")
            .replace(")", "")
        )
        fname = f"pdp_{safe_name}.png"
        fig.savefig(os.path.join(output_dir, fname), bbox_inches="tight", dpi=150)
        plt.close(fig)
        saved.append(fname)

    return saved


# ---------------------------------------------------------------------------
# Subgroup analysis
# ---------------------------------------------------------------------------


def run_subgroup_analysis(
    model: object,
    test_X: pd.DataFrame,
    test_y: np.ndarray,
    test_protected_path: str,
    subgroup_attrs: list[str],
    is_classification: bool,
    warnings_list: list[str] | None = None,
) -> dict:
    """Compute the primary metric per group level for each protected attribute.

    Loads subgroup labels from *test_protected_path* (written by DataEngineer
    before one-hot encoding) rather than trying to reconstruct them from the
    one-hot encoded ``test_X``.

    Args:
        model: A fitted sklearn-compatible model.
        test_X: One-hot encoded test feature matrix.
        test_y: True labels / outcomes for the test set (1D numpy array).
        test_protected_path: Path to ``test_protected.csv`` (pre-encoding labels).
        subgroup_attrs: List of protected attribute column names (e.g. ["X1SEX"]).
        is_classification: True for AUC; False for RMSE.
        warnings_list: Optional list to append warning strings to.

    Returns:
        Nested dict: ``{attr: {group_label: {"auc"|"rmse": float, "n": int}}}``.
        If *test_protected_path* does not exist, returns ``{}``.
    """
    from sklearn.metrics import mean_squared_error, roc_auc_score  # local import

    if warnings_list is None:
        warnings_list = []

    results: dict = {}

    if not os.path.exists(test_protected_path):
        warnings_list.append(
            f"test_protected.csv not found at {test_protected_path}; "
            "subgroup analysis skipped."
        )
        return results

    protected = pd.read_csv(test_protected_path, index_col=0)

    # Both test_X and test_protected were produced by the same train/test split
    # and have matching positional order.  Reset both to 0-based integer index.
    protected = protected.reset_index(drop=True)
    test_X_pos = test_X.reset_index(drop=True)
    test_y_arr = np.asarray(test_y).ravel()

    # Compute predictions once (not per subgroup)
    try:
        if is_classification:
            y_pred_all = model.predict_proba(test_X_pos)[:, 1]
        else:
            y_pred_all = model.predict(test_X_pos).ravel()
    except Exception as exc:  # noqa: BLE001
        warnings_list.append(f"Could not generate predictions for subgroup analysis: {exc}")
        return results

    for attr in subgroup_attrs:
        if attr not in protected.columns:
            warnings_list.append(
                f"Subgroup attribute '{attr}' not found in test_protected.csv; skipping."
            )
            continue

        results[attr] = {}

        for group_val, group_idx_labels in protected.groupby(attr).groups.items():
            # group_idx_labels are index labels in protected (0-based after reset_index)
            pos_idx = np.asarray(group_idx_labels, dtype=int)
            n = len(pos_idx)
            if n < 10:
                warnings_list.append(
                    f"Subgroup {attr}={group_val}: only {n} samples, skipping."
                )
                continue

            gy_true = test_y_arr[pos_idx]
            gy_pred = y_pred_all[pos_idx]

            try:
                if is_classification:
                    metric_val = float(roc_auc_score(gy_true, gy_pred))
                    results[attr][str(group_val)] = {
                        "auc": round(metric_val, 4),
                        "n": n,
                    }
                else:
                    metric_val = float(np.sqrt(mean_squared_error(gy_true, gy_pred)))
                    results[attr][str(group_val)] = {
                        "rmse": round(metric_val, 4),
                        "n": n,
                    }
            except Exception as exc:  # noqa: BLE001
                warnings_list.append(
                    f"Subgroup {attr}={group_val}: metric computation failed — {exc}"
                )

        # Flag gaps > 5 %
        if results.get(attr):
            metric_key = "auc" if is_classification else "rmse"
            vals = [v[metric_key] for v in results[attr].values() if metric_key in v]
            if len(vals) >= 2:
                gap = max(vals) - min(vals)
                if gap > 0.05:
                    warnings_list.append(
                        f"Subgroup performance gap > 5% detected for {attr}: "
                        f"range = [{min(vals):.4f}, {max(vals):.4f}], gap = {gap:.4f}"
                    )

    return results


# ---------------------------------------------------------------------------
# Bootstrap confidence interval
# ---------------------------------------------------------------------------


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_iter: int = 1000,
    random_state: int = 42,
) -> tuple[float, float]:
    """Compute a 95% bootstrap confidence interval for a scalar metric.

    Args:
        y_true: True labels / outcomes (1D array).
        y_pred: Predicted scores or values (1D array, same length).
        metric_fn: Callable ``(y_true, y_pred) -> float``.
        n_iter: Number of bootstrap iterations (default 1000).
        random_state: Random seed for reproducibility.

    Returns:
        ``(lower, upper)`` — the 2.5th and 97.5th percentile of bootstrap scores.
    """
    rng = np.random.RandomState(random_state)
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    n = len(y_true)
    scores: list[float] = []
    for _ in range(n_iter):
        idx = rng.randint(0, n, n)
        try:
            scores.append(float(metric_fn(y_true[idx], y_pred[idx])))
        except Exception:  # noqa: BLE001
            pass  # skip invalid bootstrap samples (e.g. single-class AUC)
    if not scores:
        return (float("nan"), float("nan"))
    arr = np.array(scores)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


# ---------------------------------------------------------------------------
# Sensitivity analysis (high-missingness variables)
# ---------------------------------------------------------------------------


def _find_columns_for_vars(
    columns: list[str],
    raw_vars: list[str],
) -> list[str]:
    """Map raw variable names to one-hot encoded column names.

    A raw variable ``X1RACE`` may appear as ``X1RACE`` (unchanged) or as one-hot
    encoded columns ``X1RACE_2.0``, ``X1RACE_3.0``, etc.  This helper finds all
    columns that either exactly match a raw variable name or start with ``{var}_``.
    """
    matched: list[str] = []
    for var in raw_vars:
        for col in columns:
            if col == var or col.startswith(f"{var}_"):
                matched.append(col)
    return matched


def run_sensitivity_analysis(
    best_model_class: type,
    best_model_params: dict,
    train_X: pd.DataFrame,
    train_y: np.ndarray,
    test_X: pd.DataFrame,
    test_y: np.ndarray,
    high_miss_vars: list[str],
    is_classification: bool,
    random_state: int = 42,
) -> dict:
    """Re-train the best model excluding high-missingness variables and compare metrics.

    Steps:
        1. Identify which columns in train_X/test_X correspond to *high_miss_vars*
           (handles one-hot encoded columns: ``{var}_*``).
        2. Drop those columns from copies of train_X and test_X.
        3. Re-train ``best_model_class(**best_model_params)`` on the reduced training set.
        4. Evaluate on the reduced test set.
        5. Compare: did the primary metric change by > 5%?

    Args:
        best_model_class: The class of the best individual model (e.g. ``XGBClassifier``).
        best_model_params: ``model.get_params()`` dict from the best model.
        train_X: Full training feature matrix.
        train_y: Training labels / outcomes.
        test_X: Full test feature matrix.
        test_y: Test labels / outcomes.
        high_miss_vars: Raw variable names with > 20% missingness.
        is_classification: True for AUC, False for RMSE/R².
        random_state: Random seed (injected into model params if applicable).

    Returns:
        Dict with keys: ``excluded_variables``, ``n_columns_dropped``,
        ``full_model_metric``, ``reduced_model_metric``, ``metric_name``,
        ``metric_change_pct``, ``significant_change``, ``full_model_top5``,
        ``reduced_model_top5``, ``top5_overlap``, ``conclusion``.
    """
    from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score

    train_y = np.asarray(train_y).ravel()
    test_y = np.asarray(test_y).ravel()

    # Handle empty high_miss_vars
    if not high_miss_vars:
        return {
            "excluded_variables": [],
            "n_columns_dropped": 0,
            "full_model_metric": None,
            "reduced_model_metric": None,
            "metric_name": "AUC" if is_classification else "RMSE",
            "metric_change_pct": 0.0,
            "significant_change": False,
            "full_model_top5": [],
            "reduced_model_top5": [],
            "top5_overlap": None,
            "conclusion": "No high-missingness variables to exclude.",
        }

    all_cols = list(train_X.columns)
    cols_to_drop = _find_columns_for_vars(all_cols, high_miss_vars)

    if not cols_to_drop:
        return {
            "excluded_variables": high_miss_vars,
            "n_columns_dropped": 0,
            "full_model_metric": None,
            "reduced_model_metric": None,
            "metric_name": "AUC" if is_classification else "RMSE",
            "metric_change_pct": 0.0,
            "significant_change": False,
            "full_model_top5": [],
            "reduced_model_top5": [],
            "top5_overlap": None,
            "conclusion": (
                "High-missingness variables were not found in the encoded feature "
                "matrix (may have been excluded earlier)."
            ),
        }

    remaining_cols = [c for c in all_cols if c not in cols_to_drop]
    if len(remaining_cols) == 0:
        return {
            "excluded_variables": high_miss_vars,
            "n_columns_dropped": len(cols_to_drop),
            "full_model_metric": None,
            "reduced_model_metric": None,
            "metric_name": "AUC" if is_classification else "RMSE",
            "metric_change_pct": 0.0,
            "significant_change": False,
            "full_model_top5": [],
            "reduced_model_top5": [],
            "top5_overlap": None,
            "conclusion": "All columns would be dropped — sensitivity analysis not possible.",
        }

    train_X_reduced = train_X[remaining_cols].copy()
    test_X_reduced = test_X[remaining_cols].copy()

    # Evaluate the FULL model on test set
    params = dict(best_model_params)
    if "random_state" in params:
        params["random_state"] = random_state

    full_model = best_model_class(**params)
    full_model.fit(train_X, train_y)

    reduced_model = best_model_class(**params)
    reduced_model.fit(train_X_reduced, train_y)

    if is_classification:
        metric_name = "AUC"
        full_preds = full_model.predict_proba(test_X)[:, 1]
        reduced_preds = reduced_model.predict_proba(test_X_reduced)[:, 1]
        full_metric = float(roc_auc_score(test_y, full_preds))
        reduced_metric = float(roc_auc_score(test_y, reduced_preds))
    else:
        metric_name = "RMSE"
        full_preds = full_model.predict(test_X).ravel()
        reduced_preds = reduced_model.predict(test_X_reduced).ravel()
        full_metric = float(np.sqrt(mean_squared_error(test_y, full_preds)))
        reduced_metric = float(np.sqrt(mean_squared_error(test_y, reduced_preds)))

    if full_metric != 0:
        change_pct = round((reduced_metric - full_metric) / abs(full_metric) * 100, 2)
    else:
        change_pct = 0.0

    significant_change = abs(change_pct) > 5.0

    # SHAP on both models for top-5 comparison (best-effort)
    full_top5: list[str] = []
    reduced_top5: list[str] = []
    top5_overlap: int | None = None

    try:
        import shap

        # Full model SHAP
        if hasattr(full_model, "feature_importances_"):
            explainer_full = shap.TreeExplainer(full_model)
        else:
            explainer_full = shap.LinearExplainer(full_model, train_X)
        sv_full = safe_shap_values(explainer_full, test_X)
        mean_abs_full = np.abs(sv_full).mean(axis=0)
        full_top5 = [all_cols[i] for i in np.argsort(mean_abs_full)[::-1][:5]]

        # Reduced model SHAP
        if hasattr(reduced_model, "feature_importances_"):
            explainer_red = shap.TreeExplainer(reduced_model)
        else:
            explainer_red = shap.LinearExplainer(reduced_model, train_X_reduced)
        sv_red = safe_shap_values(explainer_red, test_X_reduced)
        mean_abs_red = np.abs(sv_red).mean(axis=0)
        reduced_top5 = [remaining_cols[i] for i in np.argsort(mean_abs_red)[::-1][:5]]

        top5_overlap = len(set(full_top5) & set(reduced_top5))
    except Exception:  # noqa: BLE001
        pass  # SHAP is best-effort

    # Conclusion
    if not significant_change:
        conclusion = (
            "Results are robust to exclusion of high-missingness variables. "
            f"Primary metric changed by {change_pct:+.1f}% (within 5% threshold)."
        )
    else:
        conclusion = (
            f"Excluding high-missingness variables changed the primary metric by "
            f"{change_pct:+.1f}% (exceeds 5% threshold). Findings may be sensitive "
            "to imputation of high-missingness predictors."
        )

    return {
        "excluded_variables": high_miss_vars,
        "n_columns_dropped": len(cols_to_drop),
        "full_model_metric": round(full_metric, 4),
        "reduced_model_metric": round(reduced_metric, 4),
        "metric_name": metric_name,
        "metric_change_pct": change_pct,
        "significant_change": significant_change,
        "full_model_top5": full_top5,
        "reduced_model_top5": reduced_top5,
        "top5_overlap": top5_overlap,
        "conclusion": conclusion,
    }


# ---------------------------------------------------------------------------
# School cluster reconstruction
# ---------------------------------------------------------------------------

_DEFAULT_FINGERPRINT_VARS: list[str] = [
    "X1SCHOOLCLI",
    "X1COUPERTEA",
    "X1COUPERCOU",
    "X1COUPERPRI",
    "X1CONTROL",
    "X1LOCALE",
    "X1REGION",
]


def reconstruct_school_ids(
    df: pd.DataFrame,
    fingerprint_vars: list[str] | None = None,
    validate: bool = True,
    expected_n_schools: int = 944,
    tolerance: float = 0.15,
) -> tuple[pd.Series, dict]:
    """Reconstruct pseudo-school-IDs by grouping students with identical school-level variables.

    School-level variables in HSLS:09 (X1SCHOOLCLI, X1COUPERTEA, etc.) are continuous scales
    that are identical for all students within the same school.  Grouping on these variables
    reconstructs the nested structure without needing the suppressed SCH_ID.

    Args:
        df: Student-level DataFrame containing school-level columns.
        fingerprint_vars: Columns to use as the school fingerprint.  If None, uses the
            default list of 7 HSLS:09 school-level variables.
        validate: If True, run validation checks on reconstructed clusters.
        expected_n_schools: Expected number of schools in HSLS:09 (944).
        tolerance: Acceptable deviation from expected_n_schools (default 15%).

    Returns:
        Tuple of (school_ids Series, metadata dict).
    """
    if fingerprint_vars is None:
        fingerprint_vars = list(_DEFAULT_FINGERPRINT_VARS)

    # Determine which vars are actually present
    vars_used: list[str] = [v for v in fingerprint_vars if v in df.columns]
    vars_missing: list[str] = [v for v in fingerprint_vars if v not in df.columns]

    warnings: list[str] = []
    if vars_missing:
        warnings.append(
            f"Fingerprint variables not found in DataFrame: {vars_missing}. "
            "Using remaining variables for reconstruction."
        )

    if not vars_used:
        # No fingerprint vars at all — assign everyone to -1
        school_ids = pd.Series(-1, index=df.index, dtype=int)
        return school_ids, {
            "n_clusters": 0,
            "expected_n_schools": expected_n_schools,
            "cluster_size_mean": 0.0,
            "cluster_size_median": 0.0,
            "cluster_size_min": 0,
            "cluster_size_max": 0,
            "fingerprint_vars_used": [],
            "fingerprint_vars_missing": vars_missing,
            "validation_passed": False,
            "validation_warnings": ["No fingerprint variables available for reconstruction."],
        }

    fp = df[vars_used]

    # Rows where ALL fingerprint vars are NaN → unassigned (-1)
    all_nan_mask = fp.isna().all(axis=1)

    # Build a tuple key for grouping (NaN-safe: convert to string repr)
    # We convert each row to a tuple of values; rows with identical tuples are same school.
    # For NaN handling: we use fillna with a sentinel so NaN == NaN within a column.
    fp_filled = fp.copy()
    for col in vars_used:
        fp_filled[col] = fp_filled[col].astype(str)

    group_keys = fp_filled.apply(tuple, axis=1)

    # Assign integer IDs via factorize
    codes, _uniques = pd.factorize(group_keys, sort=False)
    school_ids = pd.Series(codes, index=df.index, dtype=int)

    # Mark all-NaN rows as -1
    school_ids.loc[all_nan_mask] = -1

    # Compute cluster stats (excluding unassigned)
    assigned_mask = school_ids >= 0
    assigned_ids = school_ids[assigned_mask]

    if len(assigned_ids) == 0:
        n_clusters = 0
        sizes = pd.Series(dtype=int)
    else:
        sizes = assigned_ids.value_counts()
        n_clusters = len(sizes)

    meta: dict = {
        "n_clusters": n_clusters,
        "expected_n_schools": expected_n_schools,
        "cluster_size_mean": round(float(sizes.mean()), 1) if len(sizes) > 0 else 0.0,
        "cluster_size_median": round(float(sizes.median()), 1) if len(sizes) > 0 else 0.0,
        "cluster_size_min": int(sizes.min()) if len(sizes) > 0 else 0,
        "cluster_size_max": int(sizes.max()) if len(sizes) > 0 else 0,
        "fingerprint_vars_used": vars_used,
        "fingerprint_vars_missing": vars_missing,
        "validation_passed": True,
        "validation_warnings": list(warnings),
    }

    if validate and n_clusters > 0:
        # (a) n_clusters within tolerance of expected
        deviation = abs(n_clusters - expected_n_schools) / expected_n_schools
        if deviation >= tolerance:
            meta["validation_warnings"].append(
                f"Reconstructed {n_clusters} clusters vs {expected_n_schools} expected "
                f"(deviation={deviation:.1%}, tolerance={tolerance:.0%})."
            )
            meta["validation_passed"] = False

        # (b) No single cluster > 5% of total
        total_n = len(df)
        max_pct = int(sizes.max()) / total_n if total_n > 0 else 0.0
        if max_pct > 0.05:
            meta["validation_warnings"].append(
                f"Largest cluster contains {sizes.max()} students "
                f"({max_pct:.1%} of total) — possible school-fingerprint collision."
            )

        # (c) > 10% unassigned
        n_unassigned = int(all_nan_mask.sum())
        unassigned_pct = n_unassigned / total_n if total_n > 0 else 0.0
        if unassigned_pct > 0.10:
            meta["validation_warnings"].append(
                f"{n_unassigned} students ({unassigned_pct:.1%}) have all fingerprint "
                "variables missing and could not be assigned to a school cluster."
            )

    return school_ids, meta


# ---------------------------------------------------------------------------
# Group-aware (school-aware) train/test split
# ---------------------------------------------------------------------------


def grouped_train_test_split(
    df: pd.DataFrame,
    y: "pd.Series | np.ndarray",
    groups: "pd.Series | np.ndarray",
    test_size: float = 0.2,
    stratify: bool = False,
    random_state: int = 42,
) -> "tuple[np.ndarray, np.ndarray, dict]":
    """Split data into train/test sets respecting group (school) boundaries.

    Ensures that **no group (school) appears in both train and test sets**,
    preventing information leakage through shared school-level features.

    Parameters
    ----------
    df : pd.DataFrame
        The analytic DataFrame (used only for its length / index).
    y : array-like
        Outcome variable.  Used for stratification when *stratify=True*.
    groups : array-like
        Group labels (pseudo_school_id).  Students with ``groups == -1``
        (unassigned) are each given a unique synthetic ID so they split
        individually — they share no school information.
    test_size : float
        Target proportion of the dataset to include in the test split.
        Actual proportion may vary slightly due to group granularity.
    stratify : bool
        If *True* (classification tasks), use ``StratifiedGroupKFold`` to
        preserve approximate class balance.  If *False* (regression), use
        ``GroupShuffleSplit``.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    train_indices : np.ndarray
        Positional indices (iloc-style) for the training set.
    test_indices : np.ndarray
        Positional indices (iloc-style) for the test set.
    meta : dict
        Split metadata including method used, counts, and group overlap
        (should always be 0).
    """
    from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

    n = len(df)
    y_arr = np.asarray(y)
    groups_arr = np.asarray(groups, dtype=np.int64 if np.issubdtype(np.asarray(groups).dtype, np.integer) else np.float64)

    # --- handle unassigned students (group == -1) -------------------------
    # Give each a unique negative ID so they act as singleton groups.
    unassigned_mask = groups_arr == -1
    n_unassigned = int(unassigned_mask.sum())
    if n_unassigned > 0:
        # Synthetic IDs: -2, -3, -4, …
        synthetic_ids = np.arange(-2, -2 - n_unassigned, -1)
        groups_arr = groups_arr.copy()
        groups_arr[unassigned_mask] = synthetic_ids

    # Ensure integer type after potential float conversion
    groups_arr = groups_arr.astype(np.int64)

    # --- perform the split -------------------------------------------------
    if stratify:
        # StratifiedGroupKFold with n_splits ≈ 1/test_size ensures each fold
        # is ~test_size of the data.  We take the first fold as the test set.
        n_splits = max(2, round(1.0 / test_size))
        sgkf = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state,
        )
        # Take the first split (fold 0 = test)
        train_idx, test_idx = next(sgkf.split(df, y_arr, groups_arr))
        method = "StratifiedGroupKFold"
    else:
        gss = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state,
        )
        train_idx, test_idx = next(gss.split(df, y_arr, groups_arr))
        method = "GroupShuffleSplit"

    # --- validation --------------------------------------------------------
    train_groups = set(groups_arr[train_idx])
    test_groups = set(groups_arr[test_idx])
    # Only count overlap among real school IDs (>= 0)
    real_train = {g for g in train_groups if g >= 0}
    real_test = {g for g in test_groups if g >= 0}
    group_overlap = len(real_train & real_test)

    test_fraction = len(test_idx) / n if n > 0 else 0.0

    # Count unassigned in each split
    n_unassigned_train = int(np.sum(unassigned_mask[train_idx]))
    n_unassigned_test = int(np.sum(unassigned_mask[test_idx]))

    meta = {
        "split_method": method,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "test_fraction": round(test_fraction, 4),
        "n_groups_train": len(real_train),
        "n_groups_test": len(real_test),
        "n_unassigned_train": n_unassigned_train,
        "n_unassigned_test": n_unassigned_test,
        "group_overlap": group_overlap,
    }

    return np.asarray(train_idx), np.asarray(test_idx), meta


# ---------------------------------------------------------------------------
# Clustered bootstrap confidence interval
# ---------------------------------------------------------------------------


def clustered_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cluster_ids: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float]:
    """Compute confidence intervals using cluster-level bootstrap resampling.

    Instead of resampling individual observations (which ignores within-cluster correlation),
    this function resamples ENTIRE CLUSTERS with replacement, then computes the metric on
    all observations within the resampled clusters.

    Args:
        y_true: True labels/values.
        y_pred: Predicted probabilities or values.
        cluster_ids: Array of cluster (school) IDs aligned with y_true/y_pred.
        metric_fn: Metric function ``(y_true, y_pred) -> float``.
        n_bootstrap: Number of bootstrap iterations.
        ci_level: Confidence level (default 0.95).
        random_state: Random seed.

    Returns:
        Tuple of (ci_lower, ci_upper).
    """
    rng = np.random.RandomState(random_state)
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    cluster_ids = np.asarray(cluster_ids).ravel()

    # Exclude unassigned students (cluster_id == -1)
    valid_mask = cluster_ids >= 0
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    cluster_ids = cluster_ids[valid_mask]

    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    if n_clusters == 0:
        return (float("nan"), float("nan"))

    # Pre-compute index arrays per cluster for efficiency
    cluster_indices: dict[int, np.ndarray] = {}
    for cid in unique_clusters:
        cluster_indices[int(cid)] = np.where(cluster_ids == cid)[0]

    alpha = 1.0 - ci_level
    scores: list[float] = []

    for _ in range(n_bootstrap):
        # Resample cluster IDs with replacement
        sampled_cids = rng.choice(unique_clusters, size=n_clusters, replace=True)

        # Collect all observation indices for resampled clusters
        idx_parts: list[np.ndarray] = []
        for cid in sampled_cids:
            idx_parts.append(cluster_indices[int(cid)])
        boot_idx = np.concatenate(idx_parts)

        try:
            scores.append(float(metric_fn(y_true[boot_idx], y_pred[boot_idx])))
        except Exception:  # noqa: BLE001
            pass  # skip invalid bootstrap samples (e.g. single-class AUC)

    if not scores:
        return (float("nan"), float("nan"))

    arr = np.array(scores)
    return (
        float(np.percentile(arr, 100 * alpha / 2)),
        float(np.percentile(arr, 100 * (1 - alpha / 2))),
    )


# ---------------------------------------------------------------------------
# Intraclass correlation coefficient (ICC)
# ---------------------------------------------------------------------------


def compute_icc(
    y: np.ndarray,
    cluster_ids: np.ndarray,
) -> dict:
    """Compute the intraclass correlation coefficient (ICC) for a continuous or binary outcome.

    Uses one-way random effects ANOVA decomposition:
    - MSB = mean square between clusters
    - MSW = mean square within clusters
    - ICC = (MSB - MSW) / (MSB + (n0 - 1) * MSW)
      where n0 is the harmonic-mean cluster size for unbalanced designs.

    Args:
        y: Outcome values (continuous or binary 0/1).
        cluster_ids: Array of cluster IDs aligned with y.

    Returns:
        Dict with keys: icc, msb, msw, n_clusters, avg_cluster_size, interpretation.
    """
    y = np.asarray(y, dtype=float).ravel()
    cluster_ids = np.asarray(cluster_ids).ravel()

    # Exclude unassigned (cluster_id == -1)
    valid_mask = cluster_ids >= 0
    y = y[valid_mask]
    cluster_ids = cluster_ids[valid_mask]

    unique_clusters = np.unique(cluster_ids)
    k = len(unique_clusters)  # number of clusters

    if k <= 1:
        return {
            "icc": 0.0,
            "msb": 0.0,
            "msw": 0.0,
            "n_clusters": k,
            "avg_cluster_size": float(len(y)) if k == 1 else 0.0,
            "interpretation": "negligible",
        }

    n_total = len(y)
    grand_mean = np.mean(y)

    # Compute SS_between and SS_within
    ss_between = 0.0
    ss_within = 0.0
    cluster_sizes: list[int] = []

    for cid in unique_clusters:
        mask = cluster_ids == cid
        y_cluster = y[mask]
        n_j = len(y_cluster)
        cluster_sizes.append(n_j)
        cluster_mean = np.mean(y_cluster)
        ss_between += n_j * (cluster_mean - grand_mean) ** 2
        ss_within += np.sum((y_cluster - cluster_mean) ** 2)

    msb = ss_between / (k - 1)
    msw = ss_within / (n_total - k) if n_total > k else 0.0

    # Harmonic mean of cluster sizes (n0)
    sizes_arr = np.array(cluster_sizes, dtype=float)
    n0 = float(k / np.sum(1.0 / sizes_arr)) if np.all(sizes_arr > 0) else np.mean(sizes_arr)

    # ICC formula
    denom = msb + (n0 - 1) * msw
    if denom <= 0:
        icc_val = 0.0
    else:
        icc_val = (msb - msw) / denom

    # Clamp to [0, 1] — negative ICC values are possible but conventionally set to 0
    icc_val = max(0.0, min(1.0, icc_val))

    # Interpretation
    if icc_val < 0.05:
        interp = "negligible"
    elif icc_val < 0.15:
        interp = "small"
    elif icc_val < 0.30:
        interp = "moderate"
    else:
        interp = "large"

    return {
        "icc": round(icc_val, 4),
        "msb": round(msb, 4),
        "msw": round(msw, 4),
        "n_clusters": k,
        "avg_cluster_size": round(n0, 1),
        "interpretation": interp,
    }
