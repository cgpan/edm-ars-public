"""Tests for the Critic agent.

Unit tests run without any API calls.
Integration tests (marked @pytest.mark.integration) require ANTHROPIC_API_KEY.

Run unit tests:
    pytest tests/test_critic.py -v -k "not integration"

Run integration tests:
    pytest tests/test_critic.py -v --run-integration
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.critic import (
    Critic,
    _REQUIRED_REVIEW_KEYS,
    _VALID_AGENTS,
    _VALID_VERDICTS,
)
from src.config import load_config
from src.context import PipelineContext

CONFIG_PATH = str(Path(__file__).parent.parent / "config.yaml")

# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------

_RESEARCH_SPEC: dict = {
    "research_question": (
        "Can we predict students' 12th-grade math GPA using 9th-grade academic and "
        "motivational factors?"
    ),
    "outcome_variable": "X3TGPAMAT",
    "outcome_type": "continuous",
    "predictor_set": [
        {"variable": "X1TXMTSC", "rationale": "Prior math achievement.", "wave": "base_year"},
        {"variable": "X1MTHEFF", "rationale": "Math self-efficacy.", "wave": "base_year"},
        {"variable": "X1SES", "rationale": "Socioeconomic status.", "wave": "base_year"},
        {"variable": "X1SEX", "rationale": "Sex differences in GPA.", "wave": "base_year"},
    ],
    "target_population": "Full HSLS:09 sample",
    "subgroup_analyses": ["X1SEX"],
    "expected_contribution": "Novel motivational combination.",
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
    "n_predictors_encoded": 6,
    "missingness_summary": {
        "X1TXMTSC": {"pct_missing": 5.4, "imputation_method": "IterativeImputer"},
    },
    "variables_flagged": [],
    "validation_passed": True,
    "warnings": [
        "Multilevel structure (students nested in schools) is not modeled. This is a limitation."
    ],
}

_RESULTS_OBJECT: dict = {
    "best_model": "XGBoost",
    "best_metric_value": 0.61,
    "primary_metric": "RMSE",
    "all_models": {
        "LinearRegression": {
            "rmse": 0.71, "mae": 0.55, "r2": 0.38,
            "rmse_ci_lower": 0.69, "rmse_ci_upper": 0.73,
        },
        "RandomForest": {
            "rmse": 0.65, "mae": 0.50, "r2": 0.48,
            "rmse_ci_lower": 0.63, "rmse_ci_upper": 0.67,
        },
        "XGBoost": {
            "rmse": 0.61, "mae": 0.47, "r2": 0.53,
            "rmse_ci_lower": 0.59, "rmse_ci_upper": 0.63,
        },
    },
    "top_features": [
        {"feature": "X1TXMTSC", "shap_mean_abs": 0.18, "direction": "positive"},
        {"feature": "X1MTHEFF", "shap_mean_abs": 0.09, "direction": "positive"},
    ],
    "subgroup_performance": {
        "X1SEX": {
            "Male": {"rmse": 0.63, "n": 7800},
            "Female": {"rmse": 0.59, "n": 7592},
        }
    },
    "figures_generated": ["shap_summary.png", "shap_importance.png"],
    "tables_generated": ["model_comparison.csv"],
    "errors": [],
    "warnings": [],
}

_PASS_REVIEW_REPORT: dict = {
    "overall_verdict": "PASS",
    "overall_quality_score": 8,
    "problem_formulation_review": {"score": 8, "issues": []},
    "data_preparation_review": {"score": 8, "issues": []},
    "analysis_review": {"score": 8, "issues": []},
    "substantive_review": {
        "score": 8,
        "educational_meaningfulness": "Strong and educationally relevant findings.",
        "issues": [],
    },
    "revision_instructions": {
        "ProblemFormulator": None,
        "DataEngineer": None,
        "Analyst": None,
    },
}


def _make_ctx(tmp_path: Path) -> PipelineContext:
    ctx = PipelineContext(
        dataset_name="hsls09_public",
        raw_data_path="data/raw/nonexistent.csv",
        output_dir=str(tmp_path),
        max_revision_cycles=2,
    )
    ctx.research_spec = _RESEARCH_SPEC
    ctx.data_report = _DATA_REPORT
    ctx.results_object = _RESULTS_OBJECT
    return ctx


def _make_agent(tmp_path: Path) -> Critic:
    config = load_config(CONFIG_PATH)
    ctx = _make_ctx(tmp_path)
    with patch("anthropic.Anthropic"):
        return Critic(ctx, "critic", config)


# ---------------------------------------------------------------------------
# Unit tests — _validate_review_report()
# ---------------------------------------------------------------------------


class TestValidateReviewReport:
    def _agent(self, tmp_path: Path) -> Critic:
        return _make_agent(tmp_path)

    def test_valid_pass_report_unchanged(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        validated = agent._validate_review_report(dict(_PASS_REVIEW_REPORT))
        assert validated["overall_verdict"] == "PASS"
        assert validated["overall_quality_score"] == 8
        # revision_instructions still present
        assert validated["revision_instructions"] is not None

    def test_valid_revise_report_unchanged(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        report = {
            **_PASS_REVIEW_REPORT,
            "overall_verdict": "REVISE",
            "revision_instructions": {
                "ProblemFormulator": None,
                "DataEngineer": "Relax missingness threshold.",
                "Analyst": None,
            },
        }
        validated = agent._validate_review_report(report)
        assert validated["overall_verdict"] == "REVISE"
        assert validated["revision_instructions"]["DataEngineer"] == "Relax missingness threshold."

    def test_valid_abort_report_unchanged(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        report = {**_PASS_REVIEW_REPORT, "overall_verdict": "ABORT"}
        validated = agent._validate_review_report(report)
        assert validated["overall_verdict"] == "ABORT"

    def test_invalid_verdict_replaced_with_abort(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        report = {**_PASS_REVIEW_REPORT, "overall_verdict": "UNKNOWN_VALUE"}
        validated = agent._validate_review_report(report)
        assert validated["overall_verdict"] == "ABORT"
        assert "_validation_errors" in validated
        assert any("Invalid verdict" in e for e in validated["_validation_errors"])

    def test_empty_verdict_replaced_with_abort(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        report = {**_PASS_REVIEW_REPORT, "overall_verdict": ""}
        validated = agent._validate_review_report(report)
        assert validated["overall_verdict"] == "ABORT"

    def test_missing_required_keys_defaulted_to_none(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        # Minimally valid — only overall_verdict present
        report: dict[str, Any] = {"overall_verdict": "PASS"}
        validated = agent._validate_review_report(report)
        for key in _REQUIRED_REVIEW_KEYS:
            assert key in validated, f"Required key '{key}' missing after validation"

    def test_unknown_agent_in_revision_instructions_removed(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        report = {
            **_PASS_REVIEW_REPORT,
            "overall_verdict": "REVISE",
            "revision_instructions": {
                "ProblemFormulator": None,
                "DataEngineer": "Fix something.",
                "Analyst": None,
                "UnknownAgent": "This should be dropped.",
            },
        }
        validated = agent._validate_review_report(report)
        assert "UnknownAgent" not in validated["revision_instructions"]
        assert "DataEngineer" in validated["revision_instructions"]
        assert "_validation_errors" in validated
        assert any("UnknownAgent" in e for e in validated["_validation_errors"])

    def test_all_valid_agents_present_in_revision_instructions(self, tmp_path: Path) -> None:
        """Even if only one agent is targeted, all three agents must be in revision_instructions."""
        agent = self._agent(tmp_path)
        report = {
            **_PASS_REVIEW_REPORT,
            "overall_verdict": "REVISE",
            "revision_instructions": {
                "DataEngineer": "Fix missing data handling.",
                # ProblemFormulator and Analyst intentionally missing
            },
        }
        validated = agent._validate_review_report(report)
        for valid_agent in _VALID_AGENTS:
            assert valid_agent in validated["revision_instructions"], (
                f"Agent '{valid_agent}' missing from revision_instructions"
            )
        assert validated["revision_instructions"]["ProblemFormulator"] is None
        assert validated["revision_instructions"]["Analyst"] is None

    def test_non_dict_revision_instructions_replaced(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        report = {**_PASS_REVIEW_REPORT, "revision_instructions": "not a dict"}
        validated = agent._validate_review_report(report)
        assert isinstance(validated["revision_instructions"], dict)
        for valid_agent in _VALID_AGENTS:
            assert valid_agent in validated["revision_instructions"]

    def test_none_revision_instructions_replaced(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        report = {**_PASS_REVIEW_REPORT, "revision_instructions": None}
        validated = agent._validate_review_report(report)
        assert isinstance(validated["revision_instructions"], dict)


# ---------------------------------------------------------------------------
# Unit tests — run() orchestration
# ---------------------------------------------------------------------------


class TestRunOrchestration:
    def _agent(self, tmp_path: Path) -> Critic:
        return _make_agent(tmp_path)

    def test_run_returns_dict_with_required_keys(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        llm_json = json.dumps(_PASS_REVIEW_REPORT)
        agent.call_llm = MagicMock(return_value=f"```json\n{llm_json}\n```")

        result = agent.run()

        for key in _REQUIRED_REVIEW_KEYS:
            assert key in result, f"Required key '{key}' missing from run() result"

    def test_run_writes_review_report_json(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        llm_json = json.dumps(_PASS_REVIEW_REPORT)
        agent.call_llm = MagicMock(return_value=f"```json\n{llm_json}\n```")

        agent.run()

        report_path = tmp_path / "review_report.json"
        assert report_path.exists(), "review_report.json was not written to output_dir"
        on_disk = json.loads(report_path.read_text())
        assert on_disk["overall_verdict"] == "PASS"

    def test_run_uses_ctx_inputs_when_no_kwargs(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        llm_json = json.dumps(_PASS_REVIEW_REPORT)
        agent.call_llm = MagicMock(return_value=f"```json\n{llm_json}\n```")

        # No explicit kwargs — should read from ctx
        result = agent.run()
        assert result["overall_verdict"] == "PASS"

    def test_run_uses_explicit_kwargs_over_ctx(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        llm_json = json.dumps(_PASS_REVIEW_REPORT)
        agent.call_llm = MagicMock(return_value=f"```json\n{llm_json}\n```")

        custom_report = {**_DATA_REPORT, "analytic_n": 5000}
        agent.run(data_report=custom_report)

        # The user message passed to the LLM should contain the custom analytic_n
        call_args = agent.call_llm.call_args
        user_message = call_args[0][0]
        assert "5000" in user_message

    def test_run_validates_verdict_from_llm(self, tmp_path: Path) -> None:
        """If the LLM returns an invalid verdict, it must be replaced with ABORT."""
        agent = self._agent(tmp_path)
        bad_report = {**_PASS_REVIEW_REPORT, "overall_verdict": "INVALID_VERDICT"}
        llm_json = json.dumps(bad_report)
        agent.call_llm = MagicMock(return_value=f"```json\n{llm_json}\n```")

        result = agent.run()
        assert result["overall_verdict"] == "ABORT"

    def test_run_includes_revision_cycle_in_message(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        agent.ctx.revision_cycle = 1
        llm_json = json.dumps(_PASS_REVIEW_REPORT)
        agent.call_llm = MagicMock(return_value=f"```json\n{llm_json}\n```")

        agent.run()
        user_message = agent.call_llm.call_args[0][0]
        assert "1" in user_message  # revision_cycle=1 should appear

    def test_run_notes_max_cycles_reached_in_message(self, tmp_path: Path) -> None:
        """When revision_cycle equals max_revision_cycles, a note must appear in the prompt."""
        agent = self._agent(tmp_path)
        agent.ctx.revision_cycle = 2  # equals max_revision_cycles=2
        llm_json = json.dumps(_PASS_REVIEW_REPORT)
        agent.call_llm = MagicMock(return_value=f"```json\n{llm_json}\n```")

        agent.run()
        user_message = agent.call_llm.call_args[0][0]
        assert "Max revision cycles" in user_message or "max" in user_message.lower()

    def test_run_calls_call_llm_exactly_once(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        llm_json = json.dumps(_PASS_REVIEW_REPORT)
        agent.call_llm = MagicMock(return_value=f"```json\n{llm_json}\n```")

        agent.run()
        assert agent.call_llm.call_count == 1

    def test_run_with_revise_verdict(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        revise_report = {
            **_PASS_REVIEW_REPORT,
            "overall_verdict": "REVISE",
            "revision_instructions": {
                "ProblemFormulator": None,
                "DataEngineer": "Increase analytic_n.",
                "Analyst": None,
            },
        }
        agent.call_llm = MagicMock(
            return_value=f"```json\n{json.dumps(revise_report)}\n```"
        )
        result = agent.run()
        assert result["overall_verdict"] == "REVISE"
        assert result["revision_instructions"]["DataEngineer"] == "Increase analytic_n."


# ---------------------------------------------------------------------------
# Unit tests — FindingsMemory integration
# ---------------------------------------------------------------------------


class TestFindingsMemoryIntegration:
    """Tests for the findings_memory_summary parameter and optional novelty_review."""

    def _agent(self, tmp_path: Path) -> Critic:
        from src.agents.critic import Critic

        config = load_config(CONFIG_PATH)
        ctx = PipelineContext(
            dataset_name="hsls09_public",
            raw_data_path="data/raw/nonexistent.csv",
            output_dir=str(tmp_path),
            max_revision_cycles=2,
        )
        ctx.research_spec = _RESEARCH_SPEC
        ctx.data_report = _DATA_REPORT
        ctx.results_object = _RESULTS_OBJECT
        ctx.revision_cycle = 0
        with patch("anthropic.Anthropic"):
            return Critic(ctx, "critic", config)

    def _minimal_review(self) -> dict:
        return {
            "overall_verdict": "PASS",
            "overall_quality_score": 8,
            "problem_formulation_review": {"score": 8, "issues": []},
            "data_preparation_review": {"score": 8, "issues": []},
            "analysis_review": {"score": 8, "issues": []},
            "substantive_review": {"score": 8, "educational_meaningfulness": "Good.", "issues": []},
            "revision_instructions": {
                "ProblemFormulator": None,
                "DataEngineer": None,
                "Analyst": None,
            },
        }

    def test_findings_memory_summary_appears_in_built_message(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        agent.call_llm = MagicMock(
            return_value=f"```json\n{json.dumps(self._minimal_review())}\n```"
        )
        summary = "Prior runs: outcome=X2TXMTSCOR, best_model=XGBoost"
        agent.run(findings_memory_summary=summary)

        user_message = agent.call_llm.call_args[0][0]
        assert "Prior runs: outcome=X2TXMTSCOR" in user_message
        assert "Findings Memory Summary" in user_message

    def test_no_findings_memory_summary_no_section(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        agent.call_llm = MagicMock(
            return_value=f"```json\n{json.dumps(self._minimal_review())}\n```"
        )
        agent.run(findings_memory_summary="")

        user_message = agent.call_llm.call_args[0][0]
        assert "Findings Memory Summary" not in user_message

    def test_novelty_review_optional_in_validate(self, tmp_path: Path) -> None:
        """review_report without novelty_review key must pass validation without error."""
        agent = self._agent(tmp_path)
        review = self._minimal_review()
        # Explicitly omit novelty_review
        assert "novelty_review" not in review
        validated = agent._validate_review_report(review)
        assert validated["overall_verdict"] == "PASS"
        assert "novelty_review" not in validated

    def test_novelty_review_preserved_when_present(self, tmp_path: Path) -> None:
        """If the LLM includes novelty_review, it should be preserved in the output."""
        agent = self._agent(tmp_path)
        review = {
            **self._minimal_review(),
            "novelty_review": {
                "score": 7,
                "compared_to_prior_runs": "Novel outcome studied.",
                "contribution_builds_on_memory": True,
            },
        }
        validated = agent._validate_review_report(review)
        assert "novelty_review" in validated
        assert validated["novelty_review"]["score"] == 7


# ---------------------------------------------------------------------------
# Multi-persona reasoning: _extract_last_json_block
# ---------------------------------------------------------------------------


class TestExtractLastJsonBlock:
    def _agent(self, tmp_path: Path) -> Critic:
        return _make_agent(tmp_path)

    def test_single_json_block_extracted(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        text = '```json\n{"overall_verdict": "PASS"}\n```'
        result = agent._extract_last_json_block(text)
        assert result == '{"overall_verdict": "PASS"}'

    def test_reasoning_preamble_before_json_is_handled(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        text = (
            "LENS A — METHODOLOGIST\nModel battery is complete.\n"
            "METHODOLOGIST ASSESSMENT: No issues.\n\n"
            "LENS B — SKEPTIC\nAUC seems reasonable.\n"
            "SKEPTIC ASSESSMENT: All good.\n\n"
            "SYNTHESIZER VERDICT: PASS — strong methodology.\n\n"
            '```json\n{"overall_verdict": "PASS", "overall_quality_score": 8}\n```'
        )
        result = agent._extract_last_json_block(text)
        assert result == '{"overall_verdict": "PASS", "overall_quality_score": 8}'

    def test_last_json_block_used_when_multiple_present(self, tmp_path: Path) -> None:
        """Prose may contain an example JSON block before the real one."""
        agent = self._agent(tmp_path)
        text = (
            'For example: ```json\n{"overall_verdict": "EXAMPLE"}\n```\n\n'
            'Real output: ```json\n{"overall_verdict": "PASS"}\n```'
        )
        result = agent._extract_last_json_block(text)
        assert result == '{"overall_verdict": "PASS"}'

    def test_no_json_block_returns_full_text(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        text = "plain text no fences"
        assert agent._extract_last_json_block(text) == text

    def test_run_with_reasoning_preamble_parses_correctly(self, tmp_path: Path) -> None:
        """When LLM response has LENS sections before the JSON, run() must return correct dict."""
        agent = _make_agent(tmp_path)
        minimal = {
            "overall_verdict": "PASS",
            "overall_quality_score": 8,
            "problem_formulation_review": {"score": 8, "issues": []},
            "data_preparation_review": {"score": 8, "issues": []},
            "analysis_review": {"score": 8, "issues": []},
            "substantive_review": {
                "score": 8,
                "educational_meaningfulness": "Good.",
                "issues": [],
            },
            "revision_instructions": {
                "ProblemFormulator": None,
                "DataEngineer": None,
                "Analyst": None,
            },
        }
        response = (
            "LENS A — METHODOLOGIST\nAll checks pass.\n"
            "METHODOLOGIST ASSESSMENT: No critical issues.\n\n"
            "LENS B — SKEPTIC\nFindings seem robust.\n"
            "SKEPTIC ASSESSMENT: Acceptable.\n\n"
            "SYNTHESIZER VERDICT: PASS — solid pipeline.\n\n"
            f"```json\n{json.dumps(minimal)}\n```"
        )
        agent.call_llm = MagicMock(return_value=response)
        result = agent.run()
        assert result["overall_verdict"] == "PASS"

    def test_reasoning_file_written_when_preamble_present(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        minimal = {
            "overall_verdict": "PASS",
            "overall_quality_score": 8,
            "problem_formulation_review": {"score": 8, "issues": []},
            "data_preparation_review": {"score": 8, "issues": []},
            "analysis_review": {"score": 8, "issues": []},
            "substantive_review": {
                "score": 8,
                "educational_meaningfulness": "Good.",
                "issues": [],
            },
            "revision_instructions": {
                "ProblemFormulator": None,
                "DataEngineer": None,
                "Analyst": None,
            },
        }
        preamble = "LENS A reasoning here.\nLENS B reasoning here.\n"
        agent.call_llm = MagicMock(
            return_value=f"{preamble}```json\n{json.dumps(minimal)}\n```"
        )
        agent.run()
        reasoning_path = tmp_path / "critic_reasoning.txt"
        assert reasoning_path.exists()
        assert "LENS A" in reasoning_path.read_text()


class TestPreCriticFailuresInMessage:
    def _agent(self, tmp_path: Path) -> Critic:
        return _make_agent(tmp_path)

    def test_pre_critic_failures_appear_in_message(self, tmp_path: Path) -> None:
        from src.pre_critic_checks import CheckFailure

        agent = self._agent(tmp_path)
        minimal_json = json.dumps({
            "overall_verdict": "REVISE",
            "overall_quality_score": 4,
            "problem_formulation_review": {"score": 4, "issues": []},
            "data_preparation_review": {"score": 4, "issues": []},
            "analysis_review": {"score": 4, "issues": []},
            "substantive_review": {
                "score": 4,
                "educational_meaningfulness": "Issues found.",
                "issues": [],
            },
            "revision_instructions": {
                "ProblemFormulator": None,
                "DataEngineer": "Fix NaN",
                "Analyst": None,
            },
        })
        agent.call_llm = MagicMock(return_value=f"```json\n{minimal_json}\n```")
        failures = [
            CheckFailure("pcc_03", "major", "shap_summary.png missing", "Analyst")
        ]
        agent.run(pre_critic_failures=failures)

        user_message = agent.call_llm.call_args[0][0]
        assert "Pre-Critic Automated Checks" in user_message
        assert "pcc_03" in user_message
        assert "shap_summary.png missing" in user_message

    def test_empty_pre_critic_failures_no_section(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        minimal_json = json.dumps({
            "overall_verdict": "PASS",
            "overall_quality_score": 8,
            "problem_formulation_review": {"score": 8, "issues": []},
            "data_preparation_review": {"score": 8, "issues": []},
            "analysis_review": {"score": 8, "issues": []},
            "substantive_review": {
                "score": 8,
                "educational_meaningfulness": "Good.",
                "issues": [],
            },
            "revision_instructions": {
                "ProblemFormulator": None,
                "DataEngineer": None,
                "Analyst": None,
            },
        })
        agent.call_llm = MagicMock(return_value=f"```json\n{minimal_json}\n```")
        agent.run(pre_critic_failures=[])

        user_message = agent.call_llm.call_args[0][0]
        assert "Pre-Critic Automated Checks" not in user_message


# ---------------------------------------------------------------------------
# Integration test — full Critic run with real LLM
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_critic_full_run(tmp_path: Path) -> None:
    """Full integration: real LLM (claude-opus-4-6) reviews pipeline outputs.

    Requires ANTHROPIC_API_KEY in environment.
    Run with:  pytest tests/test_critic.py -v --run-integration
    """
    config = load_config(CONFIG_PATH)
    ctx = PipelineContext(
        dataset_name="hsls09_public",
        raw_data_path="data/raw/nonexistent.csv",
        output_dir=str(tmp_path),
        max_revision_cycles=2,
    )
    ctx.research_spec = _RESEARCH_SPEC
    ctx.data_report = _DATA_REPORT
    ctx.results_object = _RESULTS_OBJECT

    agent = Critic(ctx, "critic", config)
    result = agent.run(
        research_spec=_RESEARCH_SPEC,
        data_report=_DATA_REPORT,
        results_object=_RESULTS_OBJECT,
    )

    # --- all required keys present ---
    for key in _REQUIRED_REVIEW_KEYS:
        assert key in result, f"Required key missing: {key}"

    # --- verdict is valid ---
    assert result["overall_verdict"] in _VALID_VERDICTS, (
        f"Invalid verdict: {result['overall_verdict']}"
    )

    # --- quality score in range ---
    score = result.get("overall_quality_score")
    assert isinstance(score, (int, float)), "overall_quality_score must be numeric"
    assert 1 <= score <= 10, f"overall_quality_score out of range: {score}"

    # --- review_report.json written ---
    assert (tmp_path / "review_report.json").exists(), "review_report.json not written"

    # --- revision_instructions has all agents ---
    ri = result.get("revision_instructions", {})
    for agent_name in _VALID_AGENTS:
        assert agent_name in ri, f"Agent '{agent_name}' missing from revision_instructions"
