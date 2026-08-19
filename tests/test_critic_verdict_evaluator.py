"""V3.0 Phase 3b.10 / §10.3 — Critic verdict-evaluator tests.

The evaluator (``src.agents.verdict_evaluator.evaluate_critic_verdict``)
recomputes the verdict deterministically from (quality_score,
n_critical, n_major) per the documented thresholds, regardless of the
Critic LLM's self-reported value. Tests:

  1. PASS path requires quality≥7 AND zero critical AND ≤2 major.
  2. REVISE when any critical is present, even at high quality.
  3. REVISE when quality<7 (no critical, no major).
  4. F-CRITIC-PASSED-WITH-LOW-SCORE recurrence reproduction (3b.9 case).
  5. LLM disagreement is logged at WARNING; evaluator wins.
  6. Clean reviews still PASS (regression).

Plus orchestrator integration tests confirming the verdict-evaluator
is wired in the CRITIQUING stage runner.
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from src.agents.verdict_evaluator import (
    CriticVerdictResult,
    evaluate_critic_verdict,
)


# ---------------------------------------------------------------------------
# Threshold semantics (flat-schema variants, mirroring hand-off § 10.3)
# ---------------------------------------------------------------------------


class TestThresholds:
    def test_pass_requires_quality_ge_7_and_zero_critical(self) -> None:
        review = {
            "quality_score": 7,
            "critical_issues": [],
            "major_issues": [],
        }
        result = evaluate_critic_verdict(review)
        assert result.verdict == "PASS"
        assert result.deterministic_verdict == "PASS"
        assert result.unverified is False

    def test_revise_when_critical_issue_present_even_with_high_quality(self) -> None:
        review = {
            "quality_score": 9,
            "critical_issues": [{"severity": "critical", "msg": "data leakage"}],
            "major_issues": [],
        }
        result = evaluate_critic_verdict(review)
        assert result.verdict == "REVISE"
        assert result.deterministic_verdict == "REVISE"
        assert result.n_critical == 1
        assert "n_critical=1" in result.rationale

    def test_revise_when_quality_below_7(self) -> None:
        review = {
            "quality_score": 6,
            "critical_issues": [],
            "major_issues": [],
        }
        result = evaluate_critic_verdict(review)
        assert result.verdict == "REVISE"
        assert "quality_score=6" in result.rationale

    def test_revise_when_more_than_2_major_issues(self) -> None:
        review = {
            "quality_score": 8,
            "critical_issues": [],
            "major_issues": [
                {"severity": "major"},
                {"severity": "major"},
                {"severity": "major"},
            ],
        }
        result = evaluate_critic_verdict(review)
        assert result.verdict == "REVISE"
        assert result.n_major == 3

    def test_pass_with_two_major_issues_allowed(self) -> None:
        review = {
            "quality_score": 7,
            "critical_issues": [],
            "major_issues": [
                {"severity": "major"},
                {"severity": "major"},
            ],
        }
        result = evaluate_critic_verdict(review)
        assert result.verdict == "PASS"

    def test_pass_path_unchanged_for_clean_review(self) -> None:
        """Regression: clean reviews still PASS."""
        review = {
            "quality_score": 8,
            "critical_issues": [],
            "major_issues": [],
        }
        result = evaluate_critic_verdict(review)
        assert result.verdict == "PASS"
        assert result.unverified is False


# ---------------------------------------------------------------------------
# F-CRITIC-PASSED-WITH-LOW-SCORE recurrence (3b.5 + 3b.9 case)
# ---------------------------------------------------------------------------


class TestF3b9Recurrence:
    def test_3b9_recurrence_case_pre_3b10_would_pass_post_3b10_revises(self) -> None:
        """Reproduces F-CRITIC-PASSED-WITH-LOW-SCORE: quality=6 + 1 critical
        + 1 major. Pre-3b.10 the orchestrator trusted llm_reported_verdict
        and advanced to Writer on PASS. Post-3b.10 the deterministic
        evaluator overrides to REVISE (cycles-not-exhausted) or
        PASS+unverified (cycles-exhausted)."""
        review = {
            "quality_score": 6,
            "critical_issues": [
                {"severity": "critical", "msg": "all_models is empty"},
            ],
            "major_issues": [
                {"severity": "major", "msg": "estimand=None"},
            ],
            "overall_verdict": "PASS",  # The LLM said PASS — evaluator must override.
        }
        # Cycles available — should REVISE.
        result = evaluate_critic_verdict(
            review, revision_cycle=0, max_revision_cycles=1
        )
        assert result.verdict == "REVISE"
        assert result.deterministic_verdict == "REVISE"
        assert result.llm_verdict == "PASS"
        assert result.llm_disagreement is True
        assert result.unverified is False
        # Both threshold violations should appear in the rationale.
        assert "n_critical=1" in result.rationale
        assert "quality_score=6" in result.rationale

    def test_3b9_recurrence_at_max_cycles_downgrades_to_pass_unverified(self) -> None:
        """When cycles are exhausted, REVISE downgrades to PASS+unverified
        per the documented "If max_revision_cycles have been exhausted ...
        set the verdict to PASS" rule. The flag tells the orchestrator to
        advance to WRITING with the UNVERIFIED template."""
        review = {
            "quality_score": 6,
            "critical_issues": [{"severity": "critical", "msg": "..."}],
            "major_issues": [{"severity": "major", "msg": "..."}],
            "overall_verdict": "PASS",
        }
        result = evaluate_critic_verdict(
            review, revision_cycle=1, max_revision_cycles=1
        )
        assert result.verdict == "PASS"
        assert result.deterministic_verdict == "REVISE"  # Strict computation still REVISE.
        assert result.unverified is True


class TestLLMDisagreementLogging:
    def test_llm_reported_verdict_disagreement_logged_at_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """If the LLM says PASS but the evaluator says REVISE, evaluator
        wins and a WARNING is logged."""
        review = {
            "quality_score": 6,
            "critical_issues": [{"severity": "critical", "msg": "..."}],
            "major_issues": [],
            "overall_verdict": "PASS",  # LLM wrong.
        }
        with caplog.at_level(logging.WARNING, logger="src.agents.verdict_evaluator"):
            result = evaluate_critic_verdict(
                review, revision_cycle=0, max_revision_cycles=2
            )
        assert result.verdict == "REVISE"
        assert any(
            "LLM reported 'PASS'" in rec.message
            and "REVISE" in rec.message
            for rec in caplog.records
        ), f"Expected disagreement WARNING; got: {[r.message for r in caplog.records]}"

    def test_llm_agreement_no_warning_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When LLM and evaluator agree, no WARNING is emitted."""
        review = {
            "quality_score": 8,
            "critical_issues": [],
            "major_issues": [],
            "overall_verdict": "PASS",
        }
        with caplog.at_level(logging.WARNING, logger="src.agents.verdict_evaluator"):
            evaluate_critic_verdict(review)
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert not warnings


# ---------------------------------------------------------------------------
# Production review_report.json schema (nested per-section issues)
# ---------------------------------------------------------------------------


class TestProductionReviewSchema:
    def test_nested_review_sections_counted_correctly(self) -> None:
        """The actual review_report.json has nested per-section issues
        (problem_formulation_review.issues, data_preparation_review.issues,
        analysis_review.issues, substantive_review.issues). The evaluator
        must walk all of them."""
        review = {
            "overall_verdict": "PASS",
            "overall_quality_score": 5,
            "problem_formulation_review": {
                "score": 6,
                "issues": [{"severity": "major", "msg": "x"}],
            },
            "data_preparation_review": {
                "score": 4,
                "issues": [
                    {"severity": "critical", "msg": "y"},
                    {"severity": "major", "msg": "z"},
                ],
            },
            "analysis_review": {"score": 3, "issues": []},
            "substantive_review": {"score": 5, "issues": []},
        }
        result = evaluate_critic_verdict(
            review, revision_cycle=0, max_revision_cycles=1
        )
        assert result.n_critical == 1
        assert result.n_major == 2
        assert result.quality_score == 5
        assert result.verdict == "REVISE"

    def test_overall_quality_score_preferred_over_quality_score_alias(self) -> None:
        review = {
            "overall_quality_score": 8,
            "quality_score": 3,  # alias should NOT win.
            "critical_issues": [],
            "major_issues": [],
        }
        result = evaluate_critic_verdict(review)
        assert result.quality_score == 8
        assert result.verdict == "PASS"


# ---------------------------------------------------------------------------
# Orchestrator integration — verdict-evaluator wired in CRITIQUING stage
# ---------------------------------------------------------------------------


class TestOrchestratorVerdictEvaluatorIntegration:
    """End-to-end: when the Critic returns the F-3b9 recurrence shape
    (verdict=PASS, quality=6, critical issue present), the orchestrator
    must treat it as REVISE (cycles available) — not advance to WRITING.
    """

    def test_orchestrator_overrides_llm_pass_to_revise_on_critical_issue(
        self, tmp_path: "Path", caplog: pytest.LogCaptureFixture
    ) -> None:
        from pathlib import Path as _Path
        from src.context import PipelineContext, PipelineState
        from src.orchestrator import Orchestrator

        ctx = PipelineContext(
            dataset_name="hsls09_public",
            raw_data_path=str(tmp_path / "raw.csv"),
            output_dir=str(tmp_path / "orch_out"),
            task_type="prediction",
            max_revision_cycles=2,  # cycles available, so REVISE not UNVERIFIED.
        )
        ctx.research_spec = {"outcome_variable": "X4VAR"}
        ctx.results_object = {"all_models": {"X": {}}}
        ctx.data_report = {"validation_passed": True}

        config: dict = {
            "llm_provider": "minimax",
            "models": {
                k: "x"
                for k in (
                    "problem_formulator",
                    "data_engineer",
                    "analyst",
                    "critic",
                    "writer",
                )
            },
            "minimax": {"base_url": "x", "models": {}},
            "pipeline": {"task_type": "prediction", "max_revision_cycles": 2},
            "findings_memory": {"enabled": False},
            "review_gate": {"enabled": False},
            "paths": {
                "data_registry": "data_registry/",
                "agent_prompts": "agent_prompts/",
            },
            "sandbox": {"enabled": False},
        }

        # Build orchestrator with stubbed agents so init doesn't try to
        # construct any real LLM clients.
        with patch(
            "src.agents.problem_formulator.ProblemFormulator.__init__",
            return_value=None,
        ), patch(
            "src.agents.data_engineer.DataEngineer.__init__", return_value=None,
        ), patch(
            "src.agents.analyst.Analyst.__init__", return_value=None,
        ), patch(
            "src.agents.critic.Critic.__init__", return_value=None,
        ), patch(
            "src.agents.writer.Writer.__init__", return_value=None,
        ), patch(
            "src.sandbox.create_executor", return_value=object(),
        ):
            orch = Orchestrator(ctx, config)

        # Stub the Critic.run() to return the F-3b9 recurrence shape.
        f_3b9_review = {
            "overall_verdict": "PASS",  # The LLM "wrong" verdict.
            "overall_quality_score": 6,
            "problem_formulation_review": {"score": 4, "issues": []},
            "data_preparation_review": {
                "score": 4,
                "issues": [{"severity": "critical", "msg": "F-3b9 critical"}],
            },
            "analysis_review": {
                "score": 4,
                "issues": [{"severity": "major", "msg": "F-3b9 major"}],
            },
            "substantive_review": {"score": 5, "issues": []},
            "revision_instructions": {
                "ProblemFormulator": None,
                "DataEngineer": None,
                "Analyst": "fix the critical issue",
            },
        }
        orch.critic.run = MagicMock(return_value=f_3b9_review)
        orch._inject_skills = MagicMock()

        ctx.current_state = PipelineState.CRITIQUING
        with caplog.at_level(logging.INFO):
            orch._run_critiquing()

        # Post-condition: orchestrator detected the threshold violation
        # and advanced to REVISING (NOT to WRITING per the LLM-reported
        # PASS).
        assert ctx.current_state == PipelineState.REVISING, (
            f"Expected REVISING (verdict-evaluator override of LLM PASS); "
            f"got {ctx.current_state}"
        )
        assert ctx.revision_cycle == 1


def test_pass_at_max_cycles_with_critical_yields_unverified_writing(
    tmp_path: "Path",
) -> None:
    """When cycles are exhausted, the orchestrator advances to WRITING
    with the UNVERIFIED flag set on the review_report — even though the
    LLM said PASS and the deterministic evaluator said REVISE."""
    from src.context import PipelineContext, PipelineState
    from src.orchestrator import Orchestrator

    ctx = PipelineContext(
        dataset_name="hsls09_public",
        raw_data_path=str(tmp_path / "raw.csv"),
        output_dir=str(tmp_path / "orch_out"),
        task_type="prediction",
        max_revision_cycles=1,
    )
    ctx.revision_cycle = 1  # cycles exhausted.
    ctx.research_spec = {"outcome_variable": "X4VAR"}
    ctx.results_object = {"all_models": {"X": {}}}
    ctx.data_report = {"validation_passed": True}

    config: dict = {
        "llm_provider": "minimax",
        "models": {
            k: "x" for k in (
                "problem_formulator", "data_engineer", "analyst",
                "critic", "writer",
            )
        },
        "minimax": {"base_url": "x", "models": {}},
        "pipeline": {"task_type": "prediction", "max_revision_cycles": 1},
        "findings_memory": {"enabled": False},
        "review_gate": {"enabled": False},
        "paths": {
            "data_registry": "data_registry/",
            "agent_prompts": "agent_prompts/",
        },
        "sandbox": {"enabled": False},
    }

    with patch(
        "src.agents.problem_formulator.ProblemFormulator.__init__",
        return_value=None,
    ), patch(
        "src.agents.data_engineer.DataEngineer.__init__", return_value=None,
    ), patch(
        "src.agents.analyst.Analyst.__init__", return_value=None,
    ), patch(
        "src.agents.critic.Critic.__init__", return_value=None,
    ), patch(
        "src.agents.writer.Writer.__init__", return_value=None,
    ), patch(
        "src.sandbox.create_executor", return_value=object(),
    ):
        orch = Orchestrator(ctx, config)

    f_3b9_review = {
        "overall_verdict": "PASS",
        "overall_quality_score": 6,
        "problem_formulation_review": {"score": 4, "issues": []},
        "data_preparation_review": {
            "score": 4,
            "issues": [{"severity": "critical", "msg": "..."}],
        },
        "analysis_review": {"score": 5, "issues": []},
        "substantive_review": {"score": 5, "issues": []},
        "revision_instructions": {
            "ProblemFormulator": None,
            "DataEngineer": None,
            "Analyst": None,
        },
    }
    orch.critic.run = MagicMock(return_value=f_3b9_review)
    orch._inject_skills = MagicMock()

    ctx.current_state = PipelineState.CRITIQUING
    orch._run_critiquing()

    assert ctx.current_state == PipelineState.WRITING
    assert ctx.review_report.get("unverified") is True
