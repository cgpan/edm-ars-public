"""
analysis_helpers.py — Deterministic helpers for SHAP and subgroup analysis.

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
