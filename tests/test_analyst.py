"""Tests for the Analyst agent.

Unit tests run without any API calls.
Integration tests (marked @pytest.mark.integration) require ANTHROPIC_API_KEY
and pre-written train/test CSVs.

Run unit tests:
    pytest tests/test_analyst.py -v -k "not integration"

Run integration tests:
    pytest tests/test_analyst.py -v --run-integration
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.analyst import (
    Analyst,
    _AUC_SUSPICION_THRESHOLD,
    _REPAIR_HINTS,
    _REQUIRED_KEYS,
    _classify_error,
)
from src.config import load_config
from src.context import PipelineContext

CONFIG_PATH = str(Path(__file__).parent.parent / "config.yaml")

# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------

_RESEARCH_SPEC: dict = {
    "research_question": (
        "Can we predict students' 12th-grade math GPA using 9th-grade academic, "
        "attitudinal, and demographic factors?"
    ),
    "outcome_variable": "X3TGPAMAT",
    "outcome_type": "continuous",
    "predictor_set": [
        {"variable": "X1TXMTSC", "rationale": "Prior math achievement.", "wave": "base_year"},
        {"variable": "X1MTHEFF", "rationale": "Math self-efficacy.", "wave": "base_year"},
        {"variable": "X1SES_U", "rationale": "Socioeconomic status.", "wave": "base_year"},
        {"variable": "X1SEX", "rationale": "Sex differences in math GPA.", "wave": "base_year"},
    ],
    "target_population": "full sample",
    "subgroup_analyses": ["X1SEX"],
    "expected_contribution": "Demonstrates relative importance of attitudes vs. prior achievement.",
    "potential_limitations": ["Multilevel structure not modeled"],
    "novelty_score_self_assessment": 4,
}

_DATA_REPORT: dict = {
    "dataset": "hsls09_public",
    "original_n": 23503,
    "analytic_n": 19240,
    "n_train": 15392,
    "n_test": 3848,
    "outcome_variable": "X3TGPAMAT",
    "outcome_type": "continuous",
    "class_balance": None,
    "n_predictors_raw": 4,
    "n_predictors_encoded": 5,
    "missingness_summary": {
        "X1TXMTSC": {"pct_missing": 1.2, "imputation_method": "median"},
    },
    "variables_flagged": [],
    "validation_passed": True,
    "warnings": [
        "Multilevel structure (students nested in schools) is not modeled. This is a limitation."
    ],
}


def _make_ctx(tmp_path: Path) -> PipelineContext:
    return PipelineContext(
        dataset_name="hsls09_public",
        raw_data_path="data/raw/nonexistent.csv",
        output_dir=str(tmp_path),
        max_revision_cycles=2,
    )


def _make_config() -> dict:
    return load_config(CONFIG_PATH)


def _write_synthetic_splits(tmp_path: Path, n: int = 300, outcome_type: str = "continuous") -> None:
    """Write small synthetic train/test CSV splits for integration testing."""
    rng = np.random.default_rng(42)
    n_train = int(n * 0.8)
    n_test = n - n_train

    for split, size in [("train", n_train), ("test", n_test)]:
        X = pd.DataFrame(
            {
                "X1TXMTSC": rng.normal(0.0, 1.0, size=size).round(2),
                "X1MTHEFF": rng.normal(0.0, 1.0, size=size).round(2),
                "X1SES_U": rng.normal(0.0, 1.0, size=size).round(2),
                "X1SEX": rng.choice([1, 2], size=size).astype(float),
            }
        )
        if outcome_type == "continuous":
            y = pd.DataFrame(
                {"X3TGPAMAT": np.clip(rng.normal(2.5, 0.8, size=size), 0.0, 4.0).round(2)}
            )
        else:
            y = pd.DataFrame(
                {"dropout_derived": rng.choice([0, 1], size=size, p=[0.85, 0.15]).astype(float)}
            )
        X.to_csv(tmp_path / f"{split}_X.csv", index=False)
        y.to_csv(tmp_path / f"{split}_y.csv", index=False)


def _make_valid_results(outcome_type: str = "continuous") -> dict:
    """Return a minimal valid results dict matching the SPEC §6 schema."""
    if outcome_type == "continuous":
        return {
            "best_model": "XGBoost",
            "best_metric_value": 0.61,
            "primary_metric": "RMSE",
            "all_models": {
                "LinearRegression": {"rmse": 0.71, "mae": 0.55, "r2": 0.38,
                                     "rmse_ci_lower": 0.69, "rmse_ci_upper": 0.73},
                "RandomForest": {"rmse": 0.65, "mae": 0.50, "r2": 0.48,
                                 "rmse_ci_lower": 0.63, "rmse_ci_upper": 0.67},
                "XGBoost": {"rmse": 0.61, "mae": 0.47, "r2": 0.53,
                            "rmse_ci_lower": 0.59, "rmse_ci_upper": 0.63},
            },
            "top_features": [
                {"feature": "X1TXMTSC", "shap_mean_abs": 0.18, "direction": "positive"},
                {"feature": "X1MTHEFF", "shap_mean_abs": 0.09, "direction": "positive"},
            ],
            "subgroup_performance": {
                "X1SEX": {
                    "Male": {"rmse": 0.63, "n": 1920},
                    "Female": {"rmse": 0.59, "n": 1928},
                }
            },
            "figures_generated": ["shap_summary.png", "shap_importance.png", "residual_plot.png"],
            "tables_generated": ["model_comparison.csv", "feature_importance.csv"],
            "errors": [],
            "warnings": [],
        }
    else:
        return {
            "best_model": "XGBoost",
            "best_metric_value": 0.82,
            "primary_metric": "AUC",
            "all_models": {
                "LogisticRegression": {"auc": 0.75, "accuracy": 0.87, "precision": 0.60,
                                       "recall": 0.45, "f1": 0.51,
                                       "auc_ci_lower": 0.72, "auc_ci_upper": 0.78},
                "RandomForest": {"auc": 0.80, "accuracy": 0.88, "precision": 0.65,
                                 "recall": 0.50, "f1": 0.57,
                                 "auc_ci_lower": 0.77, "auc_ci_upper": 0.83},
                "XGBoost": {"auc": 0.82, "accuracy": 0.89, "precision": 0.68,
                            "recall": 0.52, "f1": 0.59,
                            "auc_ci_lower": 0.79, "auc_ci_upper": 0.85},
            },
            "top_features": [
                {"feature": "X1TXMTSC", "shap_mean_abs": 0.22, "direction": "negative"},
            ],
            "subgroup_performance": {
                "X1SEX": {
                    "Male": {"auc": 0.81, "n": 1920},
                    "Female": {"auc": 0.83, "n": 1928},
                }
            },
            "figures_generated": ["roc_curves.png", "shap_summary.png", "shap_importance.png"],
            "tables_generated": ["model_comparison.csv", "feature_importance.csv"],
            "errors": [],
            "warnings": [],
        }


# ---------------------------------------------------------------------------
# Unit tests — schema validation logic
# ---------------------------------------------------------------------------


class TestValidateResults:
    """Unit tests for Analyst._validate_results()."""

    def _make_agent(self, tmp_path: Path) -> Analyst:
        config = _make_config()
        ctx = _make_ctx(tmp_path)
        with patch("anthropic.Anthropic"):
            return Analyst(ctx, "analyst", config)

    def test_valid_regression_results_unchanged(self, tmp_path: Path) -> None:
        agent = self._make_agent(tmp_path)
        results = _make_valid_results("continuous")
        validated = agent._validate_results(results)
        assert validated["errors"] == []
        assert validated["warnings"] == []
        assert validated["best_model"] == "XGBoost"

    def test_valid_classification_results_unchanged(self, tmp_path: Path) -> None:
        agent = self._make_agent(tmp_path)
        results = _make_valid_results("classification")
        validated = agent._validate_results(results)
        assert validated["errors"] == []
        assert validated["warnings"] == []

    def test_missing_required_keys_reported(self, tmp_path: Path) -> None:
        agent = self._make_agent(tmp_path)
        results: dict[str, Any] = {
            "best_model": "XGBoost",
            # Deliberately missing all other required keys
        }
        validated = agent._validate_results(results)
        # At least one error for the missing keys
        assert validated["errors"], "Expected at least one error for missing keys"
        missing_error = next(
            (e for e in validated["errors"] if "missing required keys" in e), None
        )
        assert missing_error is not None, (
            f"Expected 'missing required keys' error, got: {validated['errors']}"
        )
        # "errors" and "warnings" are auto-initialized by setdefault before the
        # missing-key check runs, so they will NOT appear in the error message.
        actually_checked = _REQUIRED_KEYS - {"best_model", "errors", "warnings"}
        for key in actually_checked:
            assert key in missing_error, f"Expected missing key '{key}' in error message"

    def test_all_required_keys_present_no_error(self, tmp_path: Path) -> None:
        agent = self._make_agent(tmp_path)
        results = _make_valid_results("continuous")
        validated = agent._validate_results(results)
        assert not any("missing required keys" in e for e in validated["errors"])

    def test_auc_above_threshold_triggers_warning_on_best_metric(
        self, tmp_path: Path
    ) -> None:
        agent = self._make_agent(tmp_path)
        results = _make_valid_results("classification")
        results["best_metric_value"] = 0.97  # above threshold
        validated = agent._validate_results(results)
        assert any("Suspiciously high AUC" in w for w in validated["warnings"]), (
            f"Expected AUC suspicion warning, got warnings: {validated['warnings']}"
        )

    def test_auc_above_threshold_on_individual_model_triggers_warning(
        self, tmp_path: Path
    ) -> None:
        agent = self._make_agent(tmp_path)
        results = _make_valid_results("classification")
        results["all_models"]["XGBoost"]["auc"] = 0.99
        validated = agent._validate_results(results)
        assert any("XGBoost" in w for w in validated["warnings"]), (
            f"Expected per-model AUC warning, got: {validated['warnings']}"
        )

    def test_auc_exactly_at_threshold_not_flagged(self, tmp_path: Path) -> None:
        agent = self._make_agent(tmp_path)
        results = _make_valid_results("classification")
        results["best_metric_value"] = _AUC_SUSPICION_THRESHOLD  # exactly at boundary
        validated = agent._validate_results(results)
        # > threshold, not >=, so this should NOT trigger
        assert not any("Suspiciously high AUC" in w for w in validated["warnings"])

    def test_regression_auc_not_checked(self, tmp_path: Path) -> None:
        """For regression tasks (RMSE primary), AUC check must not run."""
        agent = self._make_agent(tmp_path)
        results = _make_valid_results("continuous")
        results["best_metric_value"] = 0.99  # would trigger if AUC logic ran
        validated = agent._validate_results(results)
        assert not any("Suspiciously high AUC" in w for w in validated["warnings"])

    def test_all_models_not_dict_adds_error(self, tmp_path: Path) -> None:
        agent = self._make_agent(tmp_path)
        results = _make_valid_results("continuous")
        results["all_models"] = "not a dict"  # type: ignore[assignment]
        validated = agent._validate_results(results)
        assert any("all_models" in e for e in validated["errors"])
        assert validated["all_models"] == {}

    def test_missing_top_features_defaults_to_list(self, tmp_path: Path) -> None:
        agent = self._make_agent(tmp_path)
        results = _make_valid_results("continuous")
        results["top_features"] = None  # type: ignore[assignment]
        validated = agent._validate_results(results)
        assert validated["top_features"] == []
        assert any("top_features" in w for w in validated["warnings"])

    def test_errors_and_warnings_initialised_if_absent(self, tmp_path: Path) -> None:
        agent = self._make_agent(tmp_path)
        results = _make_valid_results("continuous")
        del results["errors"]
        del results["warnings"]
        validated = agent._validate_results(results)
        assert isinstance(validated["errors"], list)
        assert isinstance(validated["warnings"], list)


class TestExtractCodeBlock:
    """Unit tests for Analyst._extract_code_block()."""

    def test_extracts_python_fenced_block(self) -> None:
        text = "Some prose.\n```python\nprint('hello')\n```\nMore prose."
        assert Analyst._extract_code_block(text) == "print('hello')"

    def test_falls_back_to_plain_fenced_block(self) -> None:
        text = "```\nx = 1\n```"
        assert Analyst._extract_code_block(text) == "x = 1"

    def test_raises_if_no_code_block(self) -> None:
        with pytest.raises(ValueError, match="No Python code block"):
            Analyst._extract_code_block("No fences here at all.")


class TestReadResults:
    """Unit tests for Analyst._read_results()."""

    def _make_agent(self, tmp_path: Path) -> Analyst:
        config = _make_config()
        ctx = _make_ctx(tmp_path)
        with patch("anthropic.Anthropic"):
            return Analyst(ctx, "analyst", config)

    def test_reads_results_json_from_disk(self, tmp_path: Path) -> None:
        agent = self._make_agent(tmp_path)
        expected = _make_valid_results("continuous")
        (tmp_path / "results.json").write_text(json.dumps(expected))
        result = agent._read_results("no json block here")
        assert result["best_model"] == expected["best_model"]

    def test_falls_back_to_json_block_in_llm_response(self, tmp_path: Path) -> None:
        agent = self._make_agent(tmp_path)
        expected = _make_valid_results("continuous")
        llm_response = f"Analysis done.\n```json\n{json.dumps(expected)}\n```\n"
        result = agent._read_results(llm_response)
        assert result["best_model"] == expected["best_model"]

    def test_returns_empty_results_on_total_failure(self, tmp_path: Path) -> None:
        agent = self._make_agent(tmp_path)
        result = agent._read_results("no results here at all")
        assert result["errors"]
        assert result["best_model"] == ""


class TestRunOrchestration:
    """Unit tests for Analyst.run() orchestration (no real API calls)."""

    def _make_agent(self, tmp_path: Path) -> Analyst:
        config = _make_config()
        ctx = _make_ctx(tmp_path)
        with patch("anthropic.Anthropic"):
            return Analyst(ctx, "analyst", config)

    def test_run_writes_results_json_to_disk(self, tmp_path: Path) -> None:
        agent = self._make_agent(tmp_path)
        expected = _make_valid_results("continuous")

        # Stub call_llm to return a code block (no-op) + a json block
        code_block = "```python\n# no-op\n```"
        json_block = f"```json\n{json.dumps(expected)}\n```"
        agent.call_llm = MagicMock(return_value=code_block + "\n" + json_block)

        # Stub execute_code to succeed immediately without running anything
        agent.execute_code = MagicMock(return_value={"returncode": 0, "stdout": "", "stderr": ""})

        # Pre-write results.json so _read_results picks it up from disk
        (tmp_path / "results.json").write_text(json.dumps(expected))

        result = agent.run(data_report=_DATA_REPORT, research_spec=_RESEARCH_SPEC)

        assert result["best_model"] == "XGBoost"
        assert (tmp_path / "results.json").exists()
        on_disk = json.loads((tmp_path / "results.json").read_text())
        assert on_disk["best_model"] == result["best_model"]

    def test_run_retries_on_code_failure(self, tmp_path: Path) -> None:
        agent = self._make_agent(tmp_path)
        expected = _make_valid_results("continuous")

        code_block = "```python\n# failing code\n```"
        fix_block = "```python\n# fixed code\n```"
        json_block = f"```json\n{json.dumps(expected)}\n```"

        # First call returns failing code; second call returns fixed code + json
        agent.call_llm = MagicMock(
            side_effect=[
                code_block,  # initial generation
                fix_block + "\n" + json_block,  # fix attempt 1
            ]
        )
        fail_result = {"returncode": 1, "stdout": "", "stderr": "SyntaxError: bad syntax"}
        success_result = {"returncode": 0, "stdout": "", "stderr": ""}
        agent.execute_code = MagicMock(side_effect=[fail_result, success_result])

        # Pre-write results.json so _read_results picks it up from disk
        (tmp_path / "results.json").write_text(json.dumps(expected))

        result = agent.run(data_report=_DATA_REPORT, research_spec=_RESEARCH_SPEC)
        assert result["best_model"] == "XGBoost"
        assert agent.call_llm.call_count == 2

    def test_run_raises_if_no_data_report(self, tmp_path: Path) -> None:
        agent = self._make_agent(tmp_path)
        with pytest.raises(ValueError, match="data_report is required"):
            agent.run(research_spec=_RESEARCH_SPEC)

    def test_run_raises_if_no_research_spec(self, tmp_path: Path) -> None:
        agent = self._make_agent(tmp_path)
        with pytest.raises(ValueError, match="research_spec is required"):
            agent.run(data_report=_DATA_REPORT)

    def test_run_uses_context_fallback_when_kwargs_absent(self, tmp_path: Path) -> None:
        """If data_report/research_spec are not in kwargs, they should come from ctx."""
        config = _make_config()
        ctx = _make_ctx(tmp_path)
        ctx.data_report = _DATA_REPORT
        ctx.research_spec = _RESEARCH_SPEC
        with patch("anthropic.Anthropic"):
            agent = Analyst(ctx, "analyst", config)

        expected = _make_valid_results("continuous")
        (tmp_path / "results.json").write_text(json.dumps(expected))
        agent.call_llm = MagicMock(return_value="```python\n# no-op\n```")
        agent.execute_code = MagicMock(return_value={"returncode": 0, "stdout": "", "stderr": ""})

        result = agent.run()  # no explicit kwargs
        assert result["best_model"] == "XGBoost"


# ---------------------------------------------------------------------------
# Integration test — full Analyst run with real LLM + code execution
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_analyst_full_run_regression(tmp_path: Path) -> None:
    """Full integration test: LLM generates analysis code, code executes, outputs validated.

    Requires:  ANTHROPIC_API_KEY in environment, pandas/sklearn/xgboost/shap installed.
    Run with:  pytest tests/test_analyst.py -v --run-integration
    """
    _write_synthetic_splits(tmp_path, n=500, outcome_type="continuous")

    config = _make_config()
    ctx = PipelineContext(
        dataset_name="hsls09_public",
        raw_data_path="data/raw/nonexistent.csv",
        output_dir=str(tmp_path),
        max_revision_cycles=2,
    )
    ctx.data_report = {**_DATA_REPORT, "n_train": 400, "n_test": 100}
    ctx.research_spec = _RESEARCH_SPEC

    agent = Analyst(ctx, "analyst", config)
    results = agent.run(data_report=ctx.data_report, research_spec=ctx.research_spec)

    # --- results.json written to disk ---
    assert (tmp_path / "results.json").exists(), "results.json was not written"

    # --- required schema keys present ---
    for key in _REQUIRED_KEYS:
        assert key in results, f"Required key missing from results: {key}"

    # --- primary metric is RMSE (regression task) ---
    assert results["primary_metric"] == "RMSE", (
        f"Expected RMSE as primary metric, got: {results['primary_metric']}"
    )

    # --- at least one model was trained ---
    assert results["all_models"], "No models were trained"

    # --- best_metric_value is a positive number ---
    assert isinstance(results["best_metric_value"], (int, float))
    assert results["best_metric_value"] > 0, "best_metric_value should be positive RMSE"

    # --- top_features present ---
    assert isinstance(results["top_features"], list)
    assert len(results["top_features"]) > 0, "top_features should not be empty"
    for feature in results["top_features"]:
        assert "feature" in feature
        assert "shap_mean_abs" in feature
        assert "direction" in feature

    # --- key figures generated ---
    assert results["figures_generated"], "No figures were generated"
    assert any(
        "shap" in f.lower() for f in results["figures_generated"]
    ), "SHAP figure missing from figures_generated"

    # --- key tables generated ---
    assert results["tables_generated"], "No tables were generated"
    assert any(
        "model_comparison" in t for t in results["tables_generated"]
    ), "model_comparison.csv missing from tables_generated"

    # --- no critical errors ---
    if results["errors"]:
        # Partial failures (one model) are acceptable; total failure is not
        assert results["all_models"], (
            f"All models failed. Errors: {results['errors']}"
        )


# ---------------------------------------------------------------------------
# Unit tests — _classify_error()
# ---------------------------------------------------------------------------


class TestClassifyError:
    def test_import_error_detected(self) -> None:
        assert _classify_error("ImportError: No module named 'lightgbm'") == "ImportError"

    def test_module_not_found_detected(self) -> None:
        assert _classify_error("ModuleNotFoundError: No module named 'lightgbm'") == "ImportError"

    def test_no_module_named_detected(self) -> None:
        assert _classify_error("no module named 'scipy'") == "ImportError"

    def test_memory_error_detected(self) -> None:
        assert _classify_error("MemoryError: unable to allocate 8.0 GiB") == "MemoryError"

    def test_oom_detected(self) -> None:
        assert _classify_error("out of memory error occurred") == "MemoryError"

    def test_convergence_warning_detected(self) -> None:
        assert _classify_error("ConvergenceWarning: lbfgs failed to converge") == "ConvergenceWarning"

    def test_did_not_converge_detected(self) -> None:
        assert _classify_error("solver did not converge after max_iter iterations") == "ConvergenceWarning"

    def test_file_not_found_detected(self) -> None:
        assert _classify_error("FileNotFoundError: [Errno 2] No such file or directory") == "FileNotFoundError"

    def test_no_such_file_detected(self) -> None:
        assert _classify_error("no such file or directory: '/data/train_X.csv'") == "FileNotFoundError"

    def test_shap_timeout_detected(self) -> None:
        assert _classify_error("shap computation timed out after 600s") == "SHAPTimeout"

    def test_shap_timeout_error_variant(self) -> None:
        assert _classify_error("shap: TimeoutError exceeded") == "SHAPTimeout"

    def test_value_error_detected(self) -> None:
        assert _classify_error("ValueError: Input contains NaN") == "ValueError"

    def test_type_error_detected(self) -> None:
        assert _classify_error("TypeError: unsupported operand type(s)") == "TypeError"

    def test_unknown_error_returns_runtime(self) -> None:
        assert _classify_error("something unexpected went wrong") == "RuntimeError"

    def test_empty_stderr_returns_runtime(self) -> None:
        assert _classify_error("") == "RuntimeError"


# ---------------------------------------------------------------------------
# Unit tests — _build_fix_message() and _read_partial_results()
# ---------------------------------------------------------------------------


class TestBuildFixMessage:
    def _make_agent(self, tmp_path: Path) -> Analyst:
        config = _make_config()
        ctx = _make_ctx(tmp_path)
        with patch("anthropic.Anthropic"):
            return Analyst(ctx, "analyst", config)

    def _exec_result(self, stderr: str, stdout: str = "") -> dict:
        return {"returncode": 1, "stderr": stderr, "stdout": stdout}

    def test_targeted_hint_appears_for_import_error(self, tmp_path: Path) -> None:
        agent = self._make_agent(tmp_path)
        msg = agent._build_fix_message(
            code="import lightgbm",
            exec_result=self._exec_result("ImportError: No module named 'lightgbm'"),
            attempt=1,
        )
        assert "ImportError" in msg
        assert _REPAIR_HINTS["ImportError"][:30] in msg

    def test_targeted_hint_appears_for_memory_error(self, tmp_path: Path) -> None:
        agent = self._make_agent(tmp_path)
        msg = agent._build_fix_message(
            code="model.fit(X)",
            exec_result=self._exec_result("MemoryError: unable to allocate"),
            attempt=1,
        )
        assert "MemoryError" in msg
        assert "n_estimators" in msg

    def test_stderr_truncated_to_3k(self, tmp_path: Path) -> None:
        agent = self._make_agent(tmp_path)
        long_stderr = "x" * 10000 + "\nValueError: final line"
        msg = agent._build_fix_message(
            code="x = 1",
            exec_result=self._exec_result(long_stderr),
            attempt=1,
        )
        # last 3000 chars of long_stderr should be in the message
        assert long_stderr[-3000:] in msg

    def test_partial_results_hint_included_when_models_present(self, tmp_path: Path) -> None:
        agent = self._make_agent(tmp_path)
        partial = {"all_models": {"LogisticRegression": {"rmse": 0.7}, "RandomForest": {"rmse": 0.65}}}
        msg = agent._build_fix_message(
            code="x = 1",
            exec_result=self._exec_result("RuntimeError: crash"),
            attempt=2,
            partial_results=partial,
        )
        assert "LogisticRegression" in msg
        assert "RandomForest" in msg
        assert "Preserve them" in msg

    def test_partial_results_not_included_when_empty(self, tmp_path: Path) -> None:
        agent = self._make_agent(tmp_path)
        msg = agent._build_fix_message(
            code="x = 1",
            exec_result=self._exec_result("RuntimeError: crash"),
            attempt=1,
            partial_results={"all_models": {}},
        )
        assert "Partial Results" not in msg

    def test_partial_results_none_no_section(self, tmp_path: Path) -> None:
        agent = self._make_agent(tmp_path)
        msg = agent._build_fix_message(
            code="x = 1",
            exec_result=self._exec_result("RuntimeError: crash"),
            attempt=1,
            partial_results=None,
        )
        assert "Partial Results" not in msg


class TestReadPartialResults:
    def _make_agent(self, tmp_path: Path) -> Analyst:
        config = _make_config()
        ctx = _make_ctx(tmp_path)
        with patch("anthropic.Anthropic"):
            return Analyst(ctx, "analyst", config)

    def test_returns_none_when_no_results_json(self, tmp_path: Path) -> None:
        agent = self._make_agent(tmp_path)
        assert agent._read_partial_results() is None

    def test_returns_dict_when_valid_json(self, tmp_path: Path) -> None:
        agent = self._make_agent(tmp_path)
        data = {"all_models": {"LogisticRegression": {"rmse": 0.7}}}
        (tmp_path / "results.json").write_text(json.dumps(data), encoding="utf-8")
        result = agent._read_partial_results()
        assert result is not None
        assert "all_models" in result

    def test_returns_none_on_invalid_json(self, tmp_path: Path) -> None:
        agent = self._make_agent(tmp_path)
        (tmp_path / "results.json").write_text("not json {{", encoding="utf-8")
        assert agent._read_partial_results() is None


@pytest.mark.integration
def test_analyst_full_run_classification(tmp_path: Path) -> None:
    """Full integration test with a binary classification outcome (dropout_derived)."""
    _write_synthetic_splits(tmp_path, n=500, outcome_type="binary")

    classification_spec = {
        **_RESEARCH_SPEC,
        "outcome_variable": "dropout_derived",
        "outcome_type": "binary",
        "research_question": "Can we predict high school dropout from 9th-grade factors?",
    }
    classification_report = {
        **_DATA_REPORT,
        "outcome_variable": "dropout_derived",
        "outcome_type": "binary",
        "class_balance": {"class_0": 0.85, "class_1": 0.15},
        "n_train": 400,
        "n_test": 100,
    }

    config = _make_config()
    ctx = PipelineContext(
        dataset_name="hsls09_public",
        raw_data_path="data/raw/nonexistent.csv",
        output_dir=str(tmp_path),
        max_revision_cycles=2,
    )

    agent = Analyst(ctx, "analyst", config)
    results = agent.run(
        data_report=classification_report,
        research_spec=classification_spec,
    )

    assert (tmp_path / "results.json").exists()
    assert results["primary_metric"] == "AUC", (
        f"Expected AUC for classification, got: {results['primary_metric']}"
    )
    assert results["all_models"]
    assert isinstance(results["best_metric_value"], (int, float))
    # AUC must be in [0, 1]
    assert 0.0 <= results["best_metric_value"] <= 1.0, (
        f"AUC out of range: {results['best_metric_value']}"
    )
    assert any(
        "roc" in f.lower() for f in results["figures_generated"]
    ), "ROC curve figure missing for classification task"
