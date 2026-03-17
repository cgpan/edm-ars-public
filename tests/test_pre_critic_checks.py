"""Tests for src/pre_critic_checks.py — deterministic pre-Critic guard."""
from __future__ import annotations

import csv
import json
import os
import types
from unittest.mock import MagicMock

import pytest

from src.pre_critic_checks import (
    CheckFailure,
    PreCriticResult,
    run_pre_critic_checks,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(
    outcome_variable: str = "X4RFDGMJSTEM",
    all_models: dict | None = None,
    top_features: list | None = None,
    subgroup_performance: dict | None = None,
    validation_passed: bool | None = True,
) -> MagicMock:
    ctx = MagicMock()
    ctx.research_spec = {"outcome_variable": outcome_variable}
    ctx.results_object = {
        "all_models": all_models if all_models is not None else {
            "LogisticRegression": {"auc": 0.65},
            "RandomForest": {"auc": 0.70},
            "XGBoost": {"auc": 0.72},
            "ElasticNet": {"auc": 0.63},
            "MLP": {"auc": 0.68},
            "StackingEnsemble": {"auc": 0.74},
        },
        "top_features": top_features if top_features is not None else [
            {"feature": "X1TXMTSCOR", "shap_mean_abs": 0.15}
        ],
        "subgroup_performance": subgroup_performance if subgroup_performance is not None else {
            "X1SEX": {"Male": {"auc": 0.71, "n": 3000}}
        },
    }
    ctx.data_report = {"validation_passed": validation_passed}
    return ctx


def _write_train_x(output_dir: str, headers: list[str]) -> None:
    path = os.path.join(output_dir, "train_X.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerow(["value"] * len(headers))


def _write_shap_figures(output_dir: str) -> None:
    for name in ("shap_summary.png", "shap_importance.png"):
        with open(os.path.join(output_dir, name), "wb") as f:
            f.write(b"\x89PNG")


# ---------------------------------------------------------------------------
# PreCriticResult unit tests
# ---------------------------------------------------------------------------


def test_pre_critic_result_empty_has_no_critical() -> None:
    result = PreCriticResult()
    assert not result.has_critical
    assert result.critical_count == 0
    assert result.major_count == 0


def test_pre_critic_result_has_critical_when_critical_present() -> None:
    result = PreCriticResult(
        failures=[CheckFailure("pcc_01", "critical", "leakage", "DataEngineer")]
    )
    assert result.has_critical
    assert result.critical_count == 1


def test_pre_critic_result_counts() -> None:
    result = PreCriticResult(
        failures=[
            CheckFailure("pcc_01", "critical", "msg", "DataEngineer"),
            CheckFailure("pcc_02", "major", "msg", "Analyst"),
            CheckFailure("pcc_03", "major", "msg", "Analyst"),
        ]
    )
    assert result.critical_count == 1
    assert result.major_count == 2


# ---------------------------------------------------------------------------
# pcc_01: outcome in train_X
# ---------------------------------------------------------------------------


def test_outcome_in_train_x_is_critical(tmp_path: "Path") -> None:
    ctx = _make_ctx(outcome_variable="X4RFDGMJSTEM")
    _write_train_x(str(tmp_path), ["X1TXMTSCOR", "X1SES", "X4RFDGMJSTEM"])
    _write_shap_figures(str(tmp_path))

    result = run_pre_critic_checks(ctx, str(tmp_path))
    ids = {f.check_id for f in result.failures}
    assert "pcc_01" in ids
    assert result.has_critical


def test_outcome_not_in_train_x_passes(tmp_path: "Path") -> None:
    ctx = _make_ctx(outcome_variable="X4RFDGMJSTEM")
    _write_train_x(str(tmp_path), ["X1TXMTSCOR", "X1SES", "X1SEX"])
    _write_shap_figures(str(tmp_path))

    result = run_pre_critic_checks(ctx, str(tmp_path))
    pcc01 = [f for f in result.failures if f.check_id == "pcc_01"]
    assert not pcc01


def test_missing_train_x_does_not_raise(tmp_path: "Path") -> None:
    # train_X.csv doesn't exist — pcc_01 should be skipped gracefully
    ctx = _make_ctx()
    _write_shap_figures(str(tmp_path))
    result = run_pre_critic_checks(ctx, str(tmp_path))
    pcc01 = [f for f in result.failures if f.check_id == "pcc_01"]
    assert not pcc01


def test_no_outcome_variable_skips_check(tmp_path: "Path") -> None:
    ctx = _make_ctx(outcome_variable="")
    _write_train_x(str(tmp_path), ["X1TXMTSCOR"])
    _write_shap_figures(str(tmp_path))
    result = run_pre_critic_checks(ctx, str(tmp_path))
    pcc01 = [f for f in result.failures if f.check_id == "pcc_01"]
    assert not pcc01


# ---------------------------------------------------------------------------
# pcc_02: model count
# ---------------------------------------------------------------------------


def test_too_few_individual_models_is_major(tmp_path: "Path") -> None:
    ctx = _make_ctx(
        all_models={
            "LogisticRegression": {"auc": 0.65},
            "RandomForest": {"auc": 0.70},
            "StackingEnsemble": {"auc": 0.74},
        }
    )
    _write_train_x(str(tmp_path), ["X1TXMTSCOR"])
    _write_shap_figures(str(tmp_path))
    result = run_pre_critic_checks(ctx, str(tmp_path))
    pcc02 = [f for f in result.failures if f.check_id == "pcc_02"]
    assert pcc02
    assert pcc02[0].severity == "major"


def test_four_individual_models_passes(tmp_path: "Path") -> None:
    ctx = _make_ctx(
        all_models={
            "LogisticRegression": {},
            "RandomForest": {},
            "XGBoost": {},
            "ElasticNet": {},
            "StackingEnsemble": {},
        }
    )
    _write_train_x(str(tmp_path), ["X1TXMTSCOR"])
    _write_shap_figures(str(tmp_path))
    result = run_pre_critic_checks(ctx, str(tmp_path))
    pcc02 = [f for f in result.failures if f.check_id == "pcc_02"]
    assert not pcc02


# ---------------------------------------------------------------------------
# pcc_03: required SHAP figures
# ---------------------------------------------------------------------------


def test_missing_shap_summary_is_major(tmp_path: "Path") -> None:
    ctx = _make_ctx()
    _write_train_x(str(tmp_path), ["X1TXMTSCOR"])
    # Only write shap_importance, not shap_summary
    with open(os.path.join(str(tmp_path), "shap_importance.png"), "wb") as f:
        f.write(b"\x89PNG")

    result = run_pre_critic_checks(ctx, str(tmp_path))
    pcc03 = [f for f in result.failures if f.check_id == "pcc_03"]
    assert pcc03
    assert pcc03[0].severity == "major"


def test_both_shap_figures_present_no_pcc03(tmp_path: "Path") -> None:
    ctx = _make_ctx()
    _write_train_x(str(tmp_path), ["X1TXMTSCOR"])
    _write_shap_figures(str(tmp_path))
    result = run_pre_critic_checks(ctx, str(tmp_path))
    pcc03 = [f for f in result.failures if f.check_id == "pcc_03"]
    assert not pcc03


# ---------------------------------------------------------------------------
# pcc_04/05: top_features and subgroup_performance
# ---------------------------------------------------------------------------


def test_empty_top_features_is_major(tmp_path: "Path") -> None:
    ctx = _make_ctx(top_features=[])
    _write_train_x(str(tmp_path), ["X1TXMTSCOR"])
    _write_shap_figures(str(tmp_path))
    result = run_pre_critic_checks(ctx, str(tmp_path))
    pcc04 = [f for f in result.failures if f.check_id == "pcc_04"]
    assert pcc04


def test_empty_subgroup_performance_is_major(tmp_path: "Path") -> None:
    ctx = _make_ctx(subgroup_performance={})
    _write_train_x(str(tmp_path), ["X1TXMTSCOR"])
    _write_shap_figures(str(tmp_path))
    result = run_pre_critic_checks(ctx, str(tmp_path))
    pcc05 = [f for f in result.failures if f.check_id == "pcc_05"]
    assert pcc05


# ---------------------------------------------------------------------------
# pcc_06: data_report.validation_passed
# ---------------------------------------------------------------------------


def test_validation_passed_false_is_critical(tmp_path: "Path") -> None:
    ctx = _make_ctx(validation_passed=False)
    ctx.data_report = {"validation_passed": False, "warnings": ["NaN values remain"]}
    _write_train_x(str(tmp_path), ["X1TXMTSCOR"])
    _write_shap_figures(str(tmp_path))
    result = run_pre_critic_checks(ctx, str(tmp_path))
    pcc06 = [f for f in result.failures if f.check_id == "pcc_06"]
    assert pcc06
    assert pcc06[0].severity == "critical"


def test_validation_passed_true_no_pcc06(tmp_path: "Path") -> None:
    ctx = _make_ctx(validation_passed=True)
    _write_train_x(str(tmp_path), ["X1TXMTSCOR"])
    _write_shap_figures(str(tmp_path))
    result = run_pre_critic_checks(ctx, str(tmp_path))
    pcc06 = [f for f in result.failures if f.check_id == "pcc_06"]
    assert not pcc06


# ---------------------------------------------------------------------------
# Clean outputs produce no failures
# ---------------------------------------------------------------------------


def test_clean_outputs_produce_no_failures(tmp_path: "Path") -> None:
    ctx = _make_ctx()
    _write_train_x(str(tmp_path), ["X1TXMTSCOR", "X1SES", "X1SEX"])
    _write_shap_figures(str(tmp_path))
    result = run_pre_critic_checks(ctx, str(tmp_path))
    assert not result.failures
    assert not result.has_critical
