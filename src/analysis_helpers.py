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


# ---------------------------------------------------------------------------
# V3.1 Arc R (R3-followup) - deterministic ITR helpers (M6 + M7)
# ---------------------------------------------------------------------------
# The synthetic gate (scripts/itr_synthetic_gate.py) certified these
# recipes; R3's live run showed LLM re-implementations deviate
# (F-R3-M6-SCALE-DEGENERATE-RULE, F-R3-M7-CI-INCONSISTENT). Generated
# code must CALL these instead of re-implementing (per the M6/M7
# skills).

def itr_dr_pseudo_outcomes(df, treatment_col, outcome_col, adjustment_cols,
                           groups=None, n_folds=5, random_state=42):
    """Cross-fitted doubly-robust pseudo-outcomes (M6 recipe)."""
    import numpy as np
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, KFold

    X = df[adjustment_cols].astype(float).to_numpy()
    t = df[treatment_col].astype(float).to_numpy()
    y = df[outcome_col].astype(float).to_numpy()
    gamma = np.zeros(len(df))
    if groups is not None:
        splits = GroupKFold(n_splits=n_folds).split(X, t, groups=np.asarray(groups))
    else:
        splits = KFold(n_splits=n_folds, shuffle=True,
                       random_state=random_state).split(X)
    for tr, te in splits:
        ps = LogisticRegression(max_iter=1000).fit(X[tr], t[tr])
        e = np.clip(ps.predict_proba(X[te])[:, 1], 0.02, 0.98)
        m1 = GradientBoostingRegressor(random_state=random_state).fit(
            X[tr][t[tr] == 1], y[tr][t[tr] == 1])
        m0 = GradientBoostingRegressor(random_state=random_state).fit(
            X[tr][t[tr] == 0], y[tr][t[tr] == 0])
        mu1, mu0 = m1.predict(X[te]), m0.predict(X[te])
        tt, yy = t[te], y[te]
        gamma[te] = (mu1 - mu0 + tt * (yy - mu1) / e
                     - (1 - tt) * (yy - mu0) / (1 - e))
    return gamma


def itr_learn_policy_tree(df, gamma, rule_covariate_cols,
                          max_depth=2, min_samples_leaf=200, random_state=42):
    """M6: shallow policy tree on RULE covariates; returns
    (tree, rule_text, share_treated). Raises ValueError on a
    degenerate rule (share outside [0.02, 0.98]) so callers must
    report the no-meaningful-rule case honestly
    (F-R3-M6-SCALE-DEGENERATE-RULE guard)."""
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier, export_text

    feats = df[rule_covariate_cols].astype(float).to_numpy()
    tree = DecisionTreeClassifier(max_depth=max_depth,
                                  min_samples_leaf=min_samples_leaf,
                                  random_state=random_state)
    tree.fit(feats, (gamma > 0).astype(int), sample_weight=np.abs(gamma))
    share = float(tree.predict(feats).mean())
    rule_text = export_text(tree, feature_names=list(rule_covariate_cols))
    if not (0.02 <= share <= 0.98):
        raise ValueError(
            "degenerate policy rule: share_treated={:.3f}; report "
            "'no meaningful targeting rule' per the M6 skill instead of "
            "shipping a treat-all/none rule".format(share))
    return tree, rule_text, share


def itr_crossfit_policy_value(df, treatment_col, outcome_col, adjustment_cols,
                              rule_covariate_cols, groups=None, n_folds=5,
                              n_boot=1000, random_state=42):
    """M7: cross-fitted policy value + gain over best constant, with a
    cluster-bootstrap CI computed on the SAME gain statistic
    (F-R3-M7-CI-INCONSISTENT guard)."""
    import numpy as np
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, KFold
    from sklearn.tree import DecisionTreeClassifier

    X = df[adjustment_cols].astype(float).to_numpy()
    R = df[rule_covariate_cols].astype(float).to_numpy()
    t = df[treatment_col].astype(float).to_numpy()
    y = df[outcome_col].astype(float).to_numpy()
    n = len(df)
    gamma = itr_dr_pseudo_outcomes(df, treatment_col, outcome_col,
                                   adjustment_cols, groups, n_folds,
                                   random_state)
    mu0_hat = np.zeros(n)
    e_hat = np.zeros(n)
    pi_hat = np.zeros(n)
    if groups is not None:
        splits = GroupKFold(n_splits=n_folds).split(X, t, groups=np.asarray(groups))
    else:
        splits = KFold(n_splits=n_folds, shuffle=True,
                       random_state=random_state).split(X)
    for tr, te in splits:
        m0 = GradientBoostingRegressor(random_state=random_state).fit(
            X[tr][t[tr] == 0], y[tr][t[tr] == 0])
        mu0_hat[te] = m0.predict(X[te])
        ps = LogisticRegression(max_iter=1000).fit(X[tr], t[tr])
        e_hat[te] = np.clip(ps.predict_proba(X[te])[:, 1], 0.02, 0.98)
        fold_tree = DecisionTreeClassifier(max_depth=2, min_samples_leaf=200,
                                           random_state=random_state)
        fold_tree.fit(R[tr], (gamma[tr] > 0).astype(int),
                      sample_weight=np.abs(gamma[tr]))
        pi_hat[te] = fold_tree.predict(R[te])
    v0c = mu0_hat + (1 - t) * (y - mu0_hat) / (1 - e_hat)
    scores_rule = v0c + pi_hat * gamma
    scores_all = v0c + gamma
    scores_none = v0c
    v_rule = float(scores_rule.mean())
    v_all = float(scores_all.mean())
    v_none = float(scores_none.mean())
    gain = v_rule - max(v_all, v_none)
    rng = np.random.default_rng(random_state)
    cluster_ids = np.asarray(groups) if groups is not None else np.arange(n)
    uniq = np.unique(cluster_ids)
    boots = []
    for _ in range(n_boot):
        take = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(cluster_ids == c)[0] for c in take])
        b_rule = scores_rule[idx].mean()
        b_best = max(scores_all[idx].mean(), scores_none[idx].mean())
        boots.append(b_rule - b_best)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "policy_value": v_rule, "value_treat_all": v_all,
        "value_treat_none": v_none, "value_gain_vs_best_constant": gain,
        "gain_ci_lower": float(lo), "gain_ci_upper": float(hi),
        "se_method": "cluster_bootstrap", "n_folds": n_folds,
    }


# ---------------------------------------------------------------------------
# Phase B - deterministic cross-cohort DiD helpers (M8)
# ---------------------------------------------------------------------------
# Certified by scripts/quasi_experimental_gates.py (DiD bias 0.006,
# pretrend detection 1.0). Generated code must CALL these (the
# M1/M6/M7 lesson).

def did_gap_in_gaps(df, outcome_col, group_col, post_col, n_boot=1000,
                    random_state=42):
    """2x2 DiD on a cross-cohort student panel with a bootstrap CI on
    the SAME statistic: (gap in POST cohort) - (gap in PRE cohort)
    where gap = mean(outcome | group=1) - mean(outcome | group=0).
    Returns the estimates.M8 core dict."""
    import numpy as np

    y = df[outcome_col].astype(float).to_numpy()
    g = df[group_col].astype(float).to_numpy()
    p = df[post_col].astype(float).to_numpy()

    def _did(idx):
        yy, gg, pp = y[idx], g[idx], p[idx]
        def gap(post_val):
            m = pp == post_val
            return yy[m & (gg == 1)].mean() - yy[m & (gg == 0)].mean()
        return gap(1.0) - gap(0.0)

    import numpy as np
    all_idx = np.arange(len(df))
    point = float(_did(all_idx))
    rng = np.random.default_rng(random_state)
    boots = []
    # stratified bootstrap within the four cells to preserve the design
    cells = [(gv, pv) for gv in (0.0, 1.0) for pv in (0.0, 1.0)]
    cell_idx = {c: all_idx[(g == c[0]) & (p == c[1])] for c in cells}
    for _ in range(n_boot):
        take = np.concatenate([
            rng.choice(cell_idx[c], size=len(cell_idx[c]), replace=True)
            for c in cells
        ])
        boots.append(_did(take))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "point_estimate": point, "ci_lower": float(lo), "ci_upper": float(hi),
        "se": float(np.std(boots)), "se_method": "stratified_bootstrap",
        "n": int(len(df)),
    }


def did_placebo_follow_wave(df, base_col, follow_col, group_col, post_col,
                            n_boot=500, random_state=42):
    """Stability probe: re-run the gap-in-gaps on the FOLLOW-wave ranks.
    A materially different estimate flags wave-instability of the gap
    change (the cross-cohort analogue of a pre-trend check; with only
    two cohorts a true pre-period does not exist and the paper must
    say so)."""
    base = did_gap_in_gaps(df.dropna(subset=[base_col]), base_col,
                           group_col, post_col, n_boot, random_state)
    fol = did_gap_in_gaps(df.dropna(subset=[follow_col]), follow_col,
                          group_col, post_col, n_boot, random_state)
    diverges = abs(base["point_estimate"] - fol["point_estimate"]) > (
        2 * max(base["se"], fol["se"]))
    return {"base_wave": base, "follow_wave": fol,
            "wave_instability_flag": bool(diverges)}


# ---------------------------------------------------------------------------
# Prediction rigor extensions (V4 stream-2, 2026-07-04)
# ---------------------------------------------------------------------------
# Deterministic implementations for the reviewer-named gaps: moderation
# analyses must be COMPUTED (not promised), dummy SHAP must be grouped by
# parent variable, best-model claims need a paired test, and calibration
# must be quantified. Generated code must CALL these.

def run_moderation_analysis(X, y, focal_cols, moderator_col, n_boot=200,
                            random_state=42):
    """Test whether the focal block's association with y varies with a
    continuous moderator.

    (a) Likelihood-ratio test of the focal x moderator interaction block
        in an unpenalized logistic model (inference model, fit on the
        full analytic sample - separate from the prediction models).
    (b) Descriptive gradient: incremental AUC of the focal block within
        moderator tertiles, with a bootstrap CI on the top-minus-bottom
        difference of increments.
    """
    import numpy as np
    import pandas as pd
    from scipy import stats
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    X = X.reset_index(drop=True).copy()
    y = np.asarray(y).astype(float)
    focal_cols = [c for c in focal_cols if c in X.columns]
    if not focal_cols or moderator_col not in X.columns:
        return {"status": "skipped",
                "reason": "focal or moderator columns absent"}

    inter = pd.DataFrame(
        {f"{c}__x__{moderator_col}": X[c] * X[moderator_col]
         for c in focal_cols},
        index=X.index,
    )
    X_full = pd.concat([X, inter], axis=1)
    base_cols = list(X.columns)

    def _loglik(frame, cols):
        m = LogisticRegression(penalty=None, max_iter=2000)
        m.fit(frame[cols], y)
        p = np.clip(m.predict_proba(frame[cols])[:, 1], 1e-12, 1 - 1e-12)
        return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))

    ll_reduced = _loglik(X_full, base_cols)
    ll_full = _loglik(X_full, base_cols + list(inter.columns))
    lrt = max(0.0, 2.0 * (ll_full - ll_reduced))
    df = len(inter.columns)
    p_value = float(stats.chi2.sf(lrt, df))

    # (b) tertile incremental AUC of the focal block
    tert = pd.qcut(X[moderator_col], 3, labels=["low", "mid", "high"],
                   duplicates="drop")
    rng = np.random.default_rng(random_state)
    cols_wo = [c for c in X.columns if c not in focal_cols]

    def _inc_for(row_idx):
        Xi, yi = X.iloc[row_idx], y[row_idx]
        if yi.sum() < 10 or (1 - yi).sum() < 10:
            return None
        try:
            m_full = LogisticRegression(penalty=None, max_iter=2000).fit(Xi, yi)
            m_red = LogisticRegression(penalty=None, max_iter=2000).fit(
                Xi[cols_wo], yi)
            return (roc_auc_score(yi, m_full.predict_proba(Xi)[:, 1])
                    - roc_auc_score(yi, m_red.predict_proba(Xi[cols_wo])[:, 1]))
        except Exception:
            return None

    lev_idx = {lev: np.where((tert == lev).to_numpy())[0]
               for lev in ("low", "mid", "high")}
    tertile_inc = {lev: _inc_for(idx) for lev, idx in lev_idx.items()}

    diff = None
    ci = [None, None]
    if tertile_inc["low"] is not None and tertile_inc["high"] is not None:
        diff = tertile_inc["high"] - tertile_inc["low"]
        boots = []
        for _ in range(n_boot):
            bl = rng.choice(lev_idx["low"], len(lev_idx["low"]), replace=True)
            bh = rng.choice(lev_idx["high"], len(lev_idx["high"]), replace=True)
            il, ih = _inc_for(bl), _inc_for(bh)
            if il is not None and ih is not None:
                boots.append(ih - il)
        if len(boots) >= max(50, n_boot // 4):
            ci = [float(np.percentile(boots, 2.5)),
                  float(np.percentile(boots, 97.5))]

    return {
        "status": "computed",
        "lrt_stat": float(lrt), "lrt_df": int(df), "lrt_p": p_value,
        "tertile_incremental_auc": {
            k: (float(v) if v is not None else None)
            for k, v in tertile_inc.items()
        },
        "top_minus_bottom_diff": (float(diff) if diff is not None else None),
        "diff_ci": ci,
        "interpretation": (
            "interaction significant at alpha=0.05" if p_value < 0.05
            else "no detectable moderation (interaction LRT p >= 0.05)"
        ),
    }


def group_shap_by_parent(feature_names, shap_mean_abs, sep="_"):
    """Aggregate mean |SHAP| of one-hot dummy columns by parent variable.

    Parent = the token before the first ``sep``; columns without ``sep``
    are their own parent. Returns a list sorted by total mean |SHAP|,
    each entry {parent, total_shap_mean_abs, n_columns, columns}.
    """
    import numpy as np

    groups = {}
    for name, val in zip(feature_names, shap_mean_abs):
        s = str(name)
        parent = s.split(sep, 1)[0] if sep in s else s
        g = groups.setdefault(parent, {"parent": parent,
                                       "total_shap_mean_abs": 0.0,
                                       "n_columns": 0, "columns": []})
        g["total_shap_mean_abs"] += float(abs(val))
        g["n_columns"] += 1
        g["columns"].append(s)
    out = sorted(groups.values(), key=lambda d: -d["total_shap_mean_abs"])
    for d in out:
        d["total_shap_mean_abs"] = float(np.round(d["total_shap_mean_abs"], 6))
    return out


def bootstrap_auc_difference(y_true, prob_a, prob_b, school_ids=None,
                             n_boot=1000, random_state=42):
    """Paired bootstrap test of AUC(a) - AUC(b) on the same test rows.

    Cluster-aware when ``school_ids`` is given (resamples clusters).
    Guards "the best model outperforms the baseline" claims.
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y_true).astype(float)
    a = np.asarray(prob_a, dtype=float)
    b = np.asarray(prob_b, dtype=float)
    point = float(roc_auc_score(y, a) - roc_auc_score(y, b))
    rng = np.random.default_rng(random_state)
    boots = []
    if school_ids is not None:
        sid = np.asarray(school_ids)
        clusters = np.unique(sid)
        cluster_rows = {c: np.where(sid == c)[0] for c in clusters}
        for _ in range(n_boot):
            take = rng.choice(clusters, len(clusters), replace=True)
            idx = np.concatenate([cluster_rows[c] for c in take])
            if len(np.unique(y[idx])) < 2:
                continue
            boots.append(roc_auc_score(y[idx], a[idx])
                         - roc_auc_score(y[idx], b[idx]))
    else:
        n = len(y)
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            if len(np.unique(y[idx])) < 2:
                continue
            boots.append(roc_auc_score(y[idx], a[idx])
                         - roc_auc_score(y[idx], b[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "auc_diff": point, "ci_lower": float(lo), "ci_upper": float(hi),
        "significant": bool(lo > 0 or hi < 0),
        "se_method": ("cluster_bootstrap" if school_ids is not None
                      else "bootstrap"),
        "n_boot_effective": len(boots),
    }


def compute_calibration_metrics(y_true, y_prob, n_bins=10):
    """Brier score, expected calibration error, and logistic
    recalibration slope/intercept (Cox calibration)."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss

    y = np.asarray(y_true).astype(float)
    p = np.clip(np.asarray(y_prob, dtype=float), 1e-12, 1 - 1e-12)
    brier = float(brier_score_loss(y, p))
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    which = np.digitize(p, bins[1:-1])
    ece = 0.0
    for b in range(n_bins):
        m = which == b
        if m.sum() == 0:
            continue
        ece += (m.mean() * abs(y[m].mean() - p[m].mean()))
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    cal = LogisticRegression(penalty=None, max_iter=2000).fit(logit, y)
    return {
        "brier": brier,
        "ece": float(ece),
        "calibration_slope": float(cal.coef_[0][0]),
        "calibration_intercept": float(cal.intercept_[0]),
        "n_bins": int(n_bins),
    }


# ---------------------------------------------------------------------------
# Stream-1 (2026-07-04): M9 composition-adjusted DiD + M10 ML heterogeneity
# ---------------------------------------------------------------------------
# Certified by scripts/quasi_experimental_gates.py::did_dr_gate /
# did_het_gate (replicated over seeds; see gate thresholds there).
# Generated code must CALL these.

def _did_design_matrix(df, covariate_cols):
    """Shared design-matrix builder for M9/M10.

    Object/categorical columns are one-hot encoded WITH an explicit
    "Missing" level (keeps n; composition adjustment stays honest about
    nonresponse). Continuous columns: NaN -> 0 plus a missing-flag
    column.
    """
    import numpy as np
    import pandas as pd

    parts = []
    for c in covariate_cols:
        col = df[c]
        if col.dtype == object or str(col.dtype) == "category":
            filled = col.astype(object).where(col.notna(), "Missing")
            parts.append(pd.get_dummies(filled, prefix=c, dtype=float))
        else:
            v = pd.to_numeric(col, errors="coerce")
            flag = v.isna().astype(float)
            parts.append(pd.DataFrame({c: v.fillna(0.0),
                                       f"{c}__missing": flag},
                                      index=df.index))
    X = pd.concat(parts, axis=1)
    # drop constant columns (e.g. a Missing level that never occurs)
    keep = [c for c in X.columns if X[c].nunique() > 1]
    return X[keep]


def did_dr_gap_change(df, outcome_col, group_col, post_col, covariate_cols,
                      n_boot=200, random_state=42, ps_clip=(0.02, 0.98)):
    """M9: composition-adjusted (AIPW) cross-cohort gap-in-gaps.

    Adjusts for COHORT compositional shift WITHIN each SES group: for
    each group g, standardizes both cohorts to the group's pooled
    covariate distribution via AIPW with a binary cohort propensity
    e_g(x) = P(post=1 | x, group=g), then differences the adjusted
    within-group changes:

        Delta_g = E_x~g[ mu_{g,1}(x) - mu_{g,0}(x) ]   (AIPW)
        tau     = Delta_1 - Delta_0

    Deliberately NOT a 4-cell standardization across groups: SES-band
    membership is near-deterministic given covariates that are
    components of the SES composite (e.g. parent education), so a
    cross-group propensity has structural positivity failure - and
    "holding fixed" a component of the group-defining construct would
    over-adjust. Cohort overlap within group is where positivity
    actually holds; clip counts are reported so the Critic can check.
    """
    import numpy as np
    from sklearn.linear_model import LinearRegression, LogisticRegression

    work = df.dropna(subset=[outcome_col]).reset_index(drop=True)
    y = work[outcome_col].astype(float).to_numpy()
    g = work[group_col].astype(int).to_numpy()
    p = work[post_col].astype(int).to_numpy()
    X = _did_design_matrix(work, covariate_cols)
    Xv = X.to_numpy(dtype=float)

    def _delta_for_group(idx_g):
        Xi, yi, pi_ = Xv[idx_g], y[idx_g], p[idx_g]
        e_model = LogisticRegression(max_iter=2000)
        e_model.fit(Xi, pi_)
        e = np.clip(e_model.predict_proba(Xi)[:, 1],
                    ps_clip[0], ps_clip[1])
        n_clip = int(np.sum((e <= ps_clip[0]) | (e >= ps_clip[1])))
        psis = {}
        for post_val in (0, 1):
            rows = pi_ == post_val
            mu = LinearRegression().fit(Xi[rows], yi[rows]).predict(Xi)
            denom = e if post_val == 1 else (1.0 - e)
            corr = np.where(rows, (yi - mu) / denom, 0.0)
            psis[post_val] = float(np.mean(mu + corr))
        return psis[1] - psis[0], n_clip

    def _tau(idx):
        gl = g[idx]
        d1, c1 = _delta_for_group(idx[gl == 1])
        d0, c0 = _delta_for_group(idx[gl == 0])
        return d1 - d0, d0, d1, c0 + c1

    all_idx = np.arange(len(work))
    point, delta_g0, delta_g1, n_clipped = _tau(all_idx)

    rng = np.random.default_rng(random_state)
    cell = g * 2 + p
    cell_idx = {c: all_idx[cell == c] for c in (0, 1, 2, 3)}
    boots = []
    for _ in range(n_boot):
        take = np.concatenate([
            rng.choice(cell_idx[c], size=len(cell_idx[c]), replace=True)
            for c in (0, 1, 2, 3)
        ])
        try:
            boots.append(_tau(take)[0])
        except Exception:
            continue
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "point_estimate": float(point),
        "ci_lower": float(lo), "ci_upper": float(hi),
        "se": float(np.std(boots)),
        "se_method": "stratified_bootstrap",
        "adjusted_change_high_ses": float(delta_g0),
        "adjusted_change_low_ses": float(delta_g1),
        "covariates_used": list(covariate_cols),
        "n": int(len(work)),
        "n_ps_clipped": int(n_clipped),
        "estimator": "AIPW_within_group_cohort_standardization",
    }


def did_ml_heterogeneity(df, outcome_col, group_col, post_col,
                         covariate_cols, subgroup_cols,
                         n_boot=100, random_state=42):
    """M10: ML-based heterogeneity of the gap change.

    Fits a gradient-boosted outcome model per group x cohort cell and
    forms tau(x) = mu_11(x) - mu_01(x) - mu_10(x) + mu_00(x) on every
    student, then summarizes tau over interpretable subgroups with
    stratified-bootstrap CIs. Descriptive heterogeneity - no
    causal-forest asymptotics are claimed, and the paper must say so.
    """
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingRegressor

    work = df.dropna(subset=[outcome_col]).reset_index(drop=True)
    y = work[outcome_col].astype(float).to_numpy()
    g = work[group_col].astype(int).to_numpy()
    p = work[post_col].astype(int).to_numpy()
    X = _did_design_matrix(work, covariate_cols)
    Xv = X.to_numpy(dtype=float)
    cell = g * 2 + p

    def _tau_x(idx, predict_on):
        taus = np.zeros(len(predict_on))
        sign = {3: 1.0, 1: -1.0, 2: -1.0, 0: 1.0}
        for c in (0, 1, 2, 3):
            rows_c = idx[cell[idx] == c]
            m = HistGradientBoostingRegressor(
                max_iter=200, random_state=random_state)
            m.fit(Xv[rows_c], y[rows_c])
            taus += sign[c] * m.predict(Xv[predict_on])
        return taus

    all_idx = np.arange(len(work))
    tau = _tau_x(all_idx, all_idx)

    # subgroup summaries (levels of original columns; ses-style continuous
    # subgroup columns are tercile-cut)
    def _levels(colname):
        col = work[colname]
        if col.dtype == object or str(col.dtype) == "category":
            return col.astype(object).where(col.notna(), "Missing")
        v = pd.to_numeric(col, errors="coerce")
        if v.nunique() <= 6:  # binary / small-integer codes: use as-is
            return v.astype(object).where(v.notna(), "Missing")
        return pd.qcut(v, 3, labels=["low", "mid", "high"],
                       duplicates="drop").astype(object).where(
                           v.notna(), "Missing")

    rng = np.random.default_rng(random_state)
    cell_idx = {c: all_idx[cell == c] for c in (0, 1, 2, 3)}
    boot_taus = []
    for _ in range(n_boot):
        take = np.concatenate([
            rng.choice(cell_idx[c], size=len(cell_idx[c]), replace=True)
            for c in (0, 1, 2, 3)
        ])
        try:
            boot_taus.append(_tau_x(take, all_idx))
        except Exception:
            continue
    boot_taus = np.asarray(boot_taus) if boot_taus else None

    # Per-level ABSOLUTE tau means are descriptive only: boosted-model
    # regularization biases absolute levels in small cells (found in the
    # did_het_gate null runs). CONTRASTS - level minus overall, and
    # pairwise for two-level attributes - cancel the shared bias, so
    # inference (CIs) is reported on contrasts only.
    have_boot = boot_taus is not None and len(boot_taus) >= 25
    overall_dist = boot_taus.mean(axis=1) if have_boot else None
    subgroups = {}
    for sc in subgroup_cols:
        lv = _levels(sc)
        per_level = {}
        level_dists = {}
        for lev in pd.unique(lv):
            if str(lev) == "Missing":
                # nonresponse is not a substantive subgroup; estimation
                # keeps these rows, but reporting them invites artifact
                # readings (cohort-asymmetric missingness).
                continue
            m = (lv == lev).to_numpy()
            if m.sum() < 50:
                continue
            entry = {"tau_mean": float(np.mean(tau[m])), "n": int(m.sum())}
            if have_boot:
                dist = boot_taus[:, m].mean(axis=1)
                level_dists[str(lev)] = dist
                contrast = dist - overall_dist
                entry["contrast_vs_overall"] = float(np.mean(tau[m])
                                                     - np.mean(tau))
                entry["contrast_ci"] = [float(np.percentile(contrast, 2.5)),
                                        float(np.percentile(contrast, 97.5))]
            per_level[str(lev)] = entry
        block = {"levels": per_level}
        if have_boot and len(level_dists) == 2:
            (la, da), (lb, db) = sorted(level_dists.items())
            diff = db - da
            block["pairwise_difference"] = {
                "levels": [lb, la],
                "estimate": float(per_level[lb]["tau_mean"]
                                  - per_level[la]["tau_mean"]),
                "ci": [float(np.percentile(diff, 2.5)),
                       float(np.percentile(diff, 97.5))],
            }
        subgroups[sc] = block

    overall_ci = [None, None]
    if boot_taus is not None and len(boot_taus) >= 40:
        dist = boot_taus.mean(axis=1)
        overall_ci = [float(np.percentile(dist, 2.5)),
                      float(np.percentile(dist, 97.5))]

    return {
        "overall_tau_mean": float(np.mean(tau)),
        "overall_ci": overall_ci,
        "tau_sd_across_students": float(np.std(tau)),
        "subgroups": subgroups,
        "model": "HistGradientBoostingRegressor(max_iter=200) per cell",
        "se_method": "stratified_bootstrap_refit",
        "n": int(len(work)),
        "caveat": ("descriptive heterogeneity; no causal-forest "
                   "asymptotics claimed"),
    }


# ---------------------------------------------------------------------------
# V4 psychometrics wrappers (P1-P6) - 2026-07-08
# ---------------------------------------------------------------------------
# Certified by scripts/psychometric_gates.py. R-backed wrappers call the
# fixed r_helpers/ scripts through src.r_bridge; generated code calls
# THESE, never raw R and never the bridge directly.

def _items_payload(items_df):
    import numpy as np

    out = {}
    for c in items_df.columns:
        v = items_df[c]
        out[str(c)] = [None if (x is None or (isinstance(x, float) and
                                              np.isnan(x))) else float(x)
                       for x in v.tolist()]
    return out


def _num_or_none(x) -> float | None:
    """float(x) unless it is missing/non-finite - then None.

    CTT results are serialized into results.json; a bare NaN both breaks
    strict JSON and reads as a number that nobody can interpret. Every
    numeric field in psy_ctt goes through here.
    """
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def psy_ctt(
    items_df: pd.DataFrame,
    *,
    min_pair_n: int = 30,
    min_item_n: int = 30,
    min_rest_items: int = 3,
) -> dict:
    """P1: classical test theory item analysis + Cronbach's alpha.

    Pure Python. Items = numeric columns (Likert categories or 0/1).

    PAIRWISE-PRESENT BY DEFAULT. Complete-case (listwise) CTT assumes a
    near-complete persons x items matrix. Intelligent-tutor logs are
    sparse BY DESIGN - a student only sees the items the skill path
    assigns - so listwise deletion can leave ZERO usable rows (observed:
    ASSISTments skill-builder, 1586 x 47 at 27.6% fill, max 45 items per
    student => n_complete = 0 and every statistic NaN). Item means and
    covariances are still perfectly estimable from the responses that
    exist, so this helper estimates them pairwise and carries the n
    behind every number.

    What is returned (never a bare NaN - non-estimable numbers are None):
      * ``estimable`` / ``method`` / ``not_estimable_reason`` - whether
        alpha could be defended on this matrix and, if not, why.
      * ``cronbach_alpha`` from the pairwise-present covariance matrix,
        every entry of which uses >= ``min_pair_n`` jointly-observed
        responses (items that cannot meet that are dropped from alpha
        and listed in ``items_excluded``).
      * ``alpha_listwise`` for comparability when complete cases exist.
      * per item: ``n_used``, ``item_total_r`` + ``item_total_r_n``,
        ``n_pair_min``, ``in_alpha``.
      * ``matrix_fill_rate``, ``n_complete``, ``pair_n`` summary,
        ``covariance_psd`` / ``min_eigenvalue``, ``caveats`` and a
        ``summary`` sentence that states the result truthfully.

    On a complete matrix this reproduces the classical listwise numbers
    exactly (mean-of-rest is a positive linear transform of sum-of-rest,
    and the covariance-matrix total equals the total-score variance).

    Args:
        items_df: persons x items frame; NaN = not administered/answered.
        min_pair_n: minimum jointly-observed responses for a covariance
            to enter alpha.
        min_item_n: minimum observed responses for an item to be usable.
        min_rest_items: minimum other items a person must have answered
            to contribute to a corrected item-total correlation (capped
            at the size of the rest pool, so short scales still work).

    Returns:
        dict as described above.
    """
    frame = pd.DataFrame(items_df)
    caveats: list[str] = []
    excluded: dict[str, str] = {}

    numeric_cols = []
    for c in frame.columns:
        # bool is numeric here: 0/1 items are legitimate CTT items
        if pd.api.types.is_numeric_dtype(frame[c]):
            numeric_cols.append(c)
        else:
            excluded[str(c)] = "non-numeric column; not treated as an item"
    X = frame[numeric_cols].astype(float)
    obs = X.notna()

    n_persons = int(len(frame))
    n_items = int(X.shape[1])
    cell_total = n_persons * n_items
    fill_rate = (float(obs.to_numpy().sum()) / cell_total) if cell_total else 0.0
    n_complete = int(obs.all(axis=1).sum()) if n_items else 0
    n_used = {c: int(obs[c].sum()) for c in numeric_cols}

    # ---- per-item descriptives (estimable from each item's own column)
    base: dict[str, dict] = {}
    for c in numeric_cols:
        col = X[c].dropna()
        mx = float(col.max()) if len(col) else float("nan")
        base[str(c)] = {
            "item": str(c),
            "n_used": n_used[c],
            "mean": _num_or_none(col.mean()) if len(col) else None,
            "sd": _num_or_none(col.std(ddof=1)) if len(col) > 1 else None,
            "max_observed": _num_or_none(mx),
            "difficulty": (_num_or_none(col.mean() / mx)
                           if len(col) and np.isfinite(mx) and mx else None),
            "item_total_r": None,
            "item_total_r_n": 0,
            "n_pair_min": None,
            "in_alpha": False,
            "excluded_reason": None,
        }

    # ---- eligibility: enough responses AND some variance
    eligible = []
    for c in numeric_cols:
        sd = base[str(c)]["sd"]
        if n_used[c] < min_item_n:
            excluded[str(c)] = (f"only {n_used[c]} observed responses "
                                f"(< min_item_n={min_item_n})")
        elif sd is None or sd <= 0:
            excluded[str(c)] = "zero variance among observed responses"
        else:
            eligible.append(c)

    # ---- corrected item-total r against the mean of each person's OTHER
    #      observed eligible items (sparse-safe analogue of the rest score)
    rest_needed = max(1, min(min_rest_items, max(len(eligible) - 1, 1)))
    if len(eligible) >= 2:
        E = X[eligible]
        e_obs = obs[eligible]
        row_sum = E.sum(axis=1, skipna=True)
        row_cnt = e_obs.sum(axis=1)
        for c in eligible:
            rest_sum = row_sum - E[c].fillna(0.0)
            rest_cnt = row_cnt - e_obs[c].astype(int)
            keep = e_obs[c] & (rest_cnt >= rest_needed)
            if int(keep.sum()) >= min_item_n:
                x = E.loc[keep, c].to_numpy(dtype=float)
                rest = (rest_sum[keep] / rest_cnt[keep]).to_numpy(dtype=float)
                if x.std() > 0 and rest.std() > 0:
                    r = float(np.corrcoef(x, rest)[0, 1])
                    base[str(c)]["item_total_r"] = _num_or_none(r)
                base[str(c)]["item_total_r_n"] = int(keep.sum())

    def _result(estimable: bool, alpha: float | None, method: str,
                kept: list, reason: str | None, pair_summary: dict,
                psd: bool | None, min_eig: float | None,
                alpha_listwise: float | None, summary: str) -> dict:
        for c in numeric_cols:
            base[str(c)]["in_alpha"] = c in kept
            base[str(c)]["excluded_reason"] = excluded.get(str(c))
        return {
            "estimable": bool(estimable),
            "method": method,
            "not_estimable_reason": reason,
            "cronbach_alpha": _num_or_none(alpha) if alpha is not None else None,
            "alpha_listwise": (_num_or_none(alpha_listwise)
                               if alpha_listwise is not None else None),
            "items": [base[str(c)] for c in numeric_cols],
            "items_excluded": [{"item": k, "reason": v}
                               for k, v in excluded.items()],
            "n_items": n_items,
            "n_items_in_alpha": len(kept),
            "n_persons": n_persons,
            "n_complete": n_complete,
            "matrix_fill_rate": _num_or_none(fill_rate),
            "pair_n": pair_summary,
            "covariance_psd": psd,
            "min_eigenvalue": (_num_or_none(min_eig)
                               if min_eig is not None else None),
            "caveats": caveats,
            "summary": summary,
        }

    sparse_note = (f"matrix fill rate {fill_rate:.3f} "
                   f"({n_persons} persons x {n_items} items, "
                   f"{n_complete} complete cases)")
    empty_pairs = {"n_pairs": 0, "min": None, "median": None, "max": None,
                   "threshold": int(min_pair_n)}

    if len(eligible) < 2:
        reason = (f"fewer than 2 usable items ({len(eligible)} of {n_items} "
                  f"had >= {min_item_n} observed responses and non-zero "
                  f"variance); Cronbach's alpha is undefined")
        return _result(False, None, "not_estimable", [], reason, empty_pairs,
                       None, None, None,
                       f"Cronbach's alpha is not estimable: {reason}. "
                       f"Item-level statistics come from each item's "
                       f"available responses ({sparse_note}).")

    # ---- pairwise co-observation counts, then prune items whose overlap
    #      with the rest is too thin for a defensible covariance
    o = obs[eligible].astype(float)
    counts = pd.DataFrame(o.T.to_numpy() @ o.to_numpy(),
                          index=eligible, columns=eligible)
    kept = list(eligible)
    while True:
        sub = counts.loc[kept, kept].to_numpy()
        thin = (sub < min_pair_n) & ~np.eye(len(kept), dtype=bool)
        if not thin.any():
            break
        if len(kept) <= 2:
            kept = []
            break
        per_item = thin.sum(axis=1)
        drop_i = min(range(len(kept)),
                     key=lambda i: (-int(per_item[i]), n_used[kept[i]],
                                    str(kept[i])))
        excluded[str(kept[drop_i])] = (
            f"{int(per_item[drop_i])} of {len(kept) - 1} item pairs had "
            f"< {min_pair_n} jointly-observed responses")
        kept = kept[:drop_i] + kept[drop_i + 1:]

    # min overlap of every eligible item with the retained alpha set
    for c in eligible:
        others = [o_ for o_ in kept if o_ != c]
        if others:
            base[str(c)]["n_pair_min"] = int(counts.loc[c, others].min())

    if len(kept) < 2:
        reason = (f"no set of >= 2 items has at least {min_pair_n} "
                  f"jointly-observed responses for every item pair; the "
                  f"persons x items matrix is too sparse for a defensible "
                  f"internal-consistency estimate")
        caveats.append("CTT assumes a near-complete persons x items matrix; "
                       "this one is structurally sparse (adaptive item "
                       "assignment). Use IRT/CDM, which model sparsity "
                       "natively, for reliability-type claims.")
        return _result(False, None, "not_estimable", [], reason, empty_pairs,
                       None, None, None,
                       f"Cronbach's alpha is not estimable: {reason} "
                       f"({sparse_note}). Item-level statistics below are "
                       f"computed from each item's available responses.")

    k = len(kept)
    sub = counts.loc[kept, kept].to_numpy()
    off = sub[np.triu_indices(k, k=1)]  # unordered item pairs
    pair_summary = {
        "n_pairs": int(off.size),
        "min": int(off.min()) if off.size else None,
        "median": _num_or_none(np.median(off)) if off.size else None,
        "max": int(off.max()) if off.size else None,
        "threshold": int(min_pair_n),
    }

    S = X[kept].cov(min_periods=min_pair_n)
    S_arr = S.to_numpy(dtype=float)
    if not np.isfinite(S_arr).all():
        reason = ("the pairwise covariance matrix has non-finite entries "
                  "even after pruning thin item pairs")
        return _result(False, None, "not_estimable", [], reason, pair_summary,
                       None, None, None,
                       f"Cronbach's alpha is not estimable: {reason} "
                       f"({sparse_note}).")

    item_var_sum = float(np.trace(S_arr))
    total_var = float(S_arr.sum())
    if not np.isfinite(total_var) or total_var <= 0:
        reason = (f"the implied total-score variance is {total_var:.4g} "
                  f"(<= 0), so alpha has no defensible value; pairwise "
                  f"covariances estimated on different subsamples need not "
                  f"form a valid covariance matrix")
        return _result(False, None, "not_estimable", [], reason, pair_summary,
                       None, None, None,
                       f"Cronbach's alpha is not estimable: {reason} "
                       f"({sparse_note}).")

    alpha = (k / (k - 1)) * (1.0 - item_var_sum / total_var)

    # PSD diagnostic: pairwise covariances come from different subsamples,
    # so the assembled matrix can be indefinite - alpha is then only an
    # approximation and the paper must say so.
    psd: bool | None = None
    min_eig: float | None = None
    d_sd = np.sqrt(np.diag(S_arr))
    if np.all(d_sd > 0):
        R = S_arr / np.outer(d_sd, d_sd)
        min_eig = float(np.linalg.eigvalsh((R + R.T) / 2.0).min())
        psd = bool(min_eig >= -1e-8)

    # listwise comparison over all numeric items, when it exists at all
    alpha_listwise: float | None = None
    if n_complete >= min_item_n and n_items > 1:
        d = X.dropna()
        tot = d.sum(axis=1)
        tv = float(tot.var(ddof=1))
        if np.isfinite(tv) and tv > 0:
            alpha_listwise = float((n_items / (n_items - 1))
                                   * (1.0 - float(d.var(ddof=1).sum()) / tv))

    dropped_for_alpha = [c for c in numeric_cols if c not in kept]
    if dropped_for_alpha:
        caveats.append(
            f"alpha is computed on {k} of {n_items} items; "
            f"{len(dropped_for_alpha)} were excluded (see items_excluded).")
    if k < 0.75 * n_items:
        caveats.append(
            f"alpha refers to a {k}-item subset, not the full {n_items}-item "
            f"instrument; do not report it as the reliability of the whole "
            f"scale.")
    if n_complete == 0:
        caveats.append(
            "no person answered every item, so listwise-complete CTT is "
            "impossible on this matrix; alpha is pairwise-present.")
    if fill_rate < 0.9:
        caveats.append(
            "pairwise covariances are estimated on different subsamples of "
            "students (who saw which items is not random under adaptive "
            "assignment), so alpha is conditional on that mechanism and may "
            "be biased if data are not missing at random.")
    if psd is False:
        caveats.append(
            f"the pairwise covariance matrix is not positive semi-definite "
            f"(minimum eigenvalue of the implied correlation matrix "
            f"{min_eig:.3f}); no single covariance structure fits all "
            f"pairwise estimates, so alpha is an approximation.")

    scope = f"{k} items" if k == n_items else f"{k} of {n_items} items"
    summary = (
        f"Cronbach's alpha = {alpha:.3f} across {scope}, estimated by "
        f"pairwise-present covariance ({sparse_note}); every covariance "
        f"uses at least {pair_summary['min']} jointly-observed responses "
        f"(median {pair_summary['median']:.0f})."
    )
    if n_complete == 0:
        summary += (" Listwise-complete estimation is impossible here: no "
                    "student answered all items.")
    elif alpha_listwise is not None:
        summary += (f" Listwise-complete alpha on {n_complete} students is "
                    f"{alpha_listwise:.3f}.")
    if psd is False:
        summary += (" The assembled pairwise covariance matrix is not "
                    "positive semi-definite, so treat the value as "
                    "approximate.")

    return _result(True, alpha, "pairwise_present", kept, None, pair_summary,
                   psd, min_eig, alpha_listwise, summary)


def psy_omega(cfa_result):
    """P2: McDonald's omega-total from a psy_cfa result.

    omega = (sum lambda)^2 / ((sum lambda)^2 + sum theta) using
    standardized loadings (theta_i = 1 - lambda_i^2).
    """
    lams = [row["est_std"] for row in cfa_result["loadings"]]
    s = sum(lams)
    theta = sum(1.0 - l * l for l in lams)
    return {"omega_total": s * s / (s * s + theta),
            "n_items": len(lams),
            "from_loadings": lams}


def psy_cfa(items_df, model, estimator="MLR"):
    """P3: single-group CFA (lavaan, FIML). Returns fit + std loadings."""
    try:
        from src.r_bridge import run_r_script
    except ModuleNotFoundError:  # copied flat into a run output dir
        from r_bridge import run_r_script

    return run_r_script("cfa_fit.R", {
        "items": _items_payload(items_df),
        "model": model,
        "estimator": estimator,
    })


def psy_invariance(items_df, group, model):
    """P6: configural->metric->scalar ladder (lavaan, Chen 2007 rules)."""
    try:
        from src.r_bridge import run_r_script
    except ModuleNotFoundError:  # copied flat into a run output dir
        from r_bridge import run_r_script

    return run_r_script("invariance_ladder.R", {
        "items": _items_payload(items_df),
        "group": [None if g is None else str(g) for g in list(group)],
        "model": model,
    }, timeout_s=1200)


def psy_grm(items_df, itemtype="graded"):
    """P4: IRT calibration (mirt GRM for Likert; "2PL" for binary)."""
    import numpy as np

    try:
        from src.r_bridge import run_r_script
    except ModuleNotFoundError:  # copied flat into a run output dir
        from r_bridge import run_r_script

    payload = {}
    for c in items_df.columns:
        v = items_df[c]
        payload[str(c)] = [None if (isinstance(x, float) and np.isnan(x))
                           else int(x) for x in v.tolist()]
    return run_r_script("irt_grm.R", {"items": payload,
                                      "itemtype": itemtype},
                        timeout_s=1200)


def psy_dif(items_df, group):
    """P5: ordinal logistic DIF (McFadden-scaled effect bands .02/.05)."""
    import numpy as np

    try:
        from src.r_bridge import run_r_script
    except ModuleNotFoundError:  # copied flat into a run output dir
        from r_bridge import run_r_script

    payload = {}
    for c in items_df.columns:
        v = items_df[c]
        payload[str(c)] = [None if (isinstance(x, float) and np.isnan(x))
                           else int(x) for x in v.tolist()]
    return run_r_script("dif_ordinal.R", {
        "items": payload,
        "group": [None if g is None else str(g) for g in list(group)],
    }, timeout_s=1200)


def psy_cdm(responses_df, q_matrix, attributes, model="DINA"):
    """P7: cognitive diagnosis (DINA/GDINA via the R CDM package).

    responses_df: binary (0/1/NaN) student x item frame.
    q_matrix: {item_name: [1-based attribute indices]}.
    Structural sparsity (NaN) is handled natively - never impute.
    """
    import numpy as np

    try:
        from src.r_bridge import run_r_script
    except ModuleNotFoundError:  # copied flat into a run output dir
        from r_bridge import run_r_script

    payload = {}
    for c in responses_df.columns:
        v = responses_df[c]
        payload[str(c)] = [None if (isinstance(x, float) and np.isnan(x))
                           else int(x) for x in v.tolist()]
    return run_r_script("cdm_fit.R", {
        "responses": payload,
        "q_matrix": {str(k): list(v) for k, v in q_matrix.items()},
        "attributes": list(attributes),
        "model": model,
    }, timeout_s=1200)
