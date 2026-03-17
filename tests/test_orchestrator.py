import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.context import PipelineContext
from src.orchestrator import Orchestrator

CONFIG_PATH = str(Path(__file__).parent.parent / "config.yaml")

# Stub return value for DataEngineer.run() — mirrors what the Phase-1 stub returned.
# Used by orchestrator-level tests that test state-machine logic, not data prep.
_STUB_DE_RESULT: dict = {
    "dataset": "hsls09_public",
    "original_n": 23503,
    "analytic_n": 19240,
    "n_train": 15392,
    "n_test": 3848,
    "outcome_variable": "X3TGPAMAT",
    "outcome_type": "continuous",
    "class_balance": None,
    "n_predictors_raw": 12,
    "n_predictors_encoded": 28,
    "missingness_summary": {
        "X1TXMTSC": {"pct_missing": 1.2, "imputation_method": "median"},
    },
    "variables_flagged": [],
    "validation_passed": True,
    "warnings": [
        "Multilevel structure (students nested in schools) is not modeled. "
        "This is a limitation."
    ],
}


def _stub_de_run(**kwargs: Any) -> dict:
    """Drop-in stub for DataEngineer.run() that avoids real API/file I/O."""
    return _STUB_DE_RESULT


# Stub return value for ProblemFormulator.run()
_STUB_PF_RESULT: dict = {
    "research_spec": {
        "research_question": (
            "Can we predict students' 12th-grade math GPA using 9th-grade "
            "academic, attitudinal, and demographic factors?"
        ),
        "outcome_variable": "X3TGPAMAT",
        "outcome_type": "continuous",
        "predictor_set": [
            {"variable": "X1TXMTSC", "rationale": "Prior math achievement.", "wave": "base_year"},
            {"variable": "X1MTHEFF", "rationale": "Math self-efficacy.", "wave": "base_year"},
        ],
        "target_population": "full sample",
        "subgroup_analyses": ["X1SEX", "X1RACE"],
        "expected_contribution": "Novel motivational combination.",
        "potential_limitations": ["Multilevel structure not modeled"],
        "novelty_score_self_assessment": 4,
    },
    "literature_context": {
        "search_query": "math GPA prediction educational data mining",
        "papers": [
            {
                "paperId": "abc123",
                "title": "Predicting Academic Achievement",
                "authors": ["Smith, J."],
                "year": 2022,
                "abstract": "ML methods for achievement prediction.",
            }
        ],
        "novelty_evidence": "Prior work uses test scores; this study uses GPA.",
    },
}


def _stub_pf_run(**kwargs: Any) -> dict:
    """Drop-in stub for ProblemFormulator.run() that avoids real API/S2 calls."""
    return _STUB_PF_RESULT


# Stub return value for Writer.run() — a minimal but structurally valid LaTeX string.
_STUB_WRITER_RESULT: str = (
    r"\documentclass[sigconf]{acmart}" + "\n"
    r"\begin{document}" + "\n"
    r"\maketitle" + "\n"
    r"\begin{abstract}Stub abstract.\end{abstract}" + "\n"
    r"\section{Introduction}Stub introduction." + "\n"
    r"\bibliographystyle{ACM-Reference-Format}" + "\n"
    r"\bibliography{references}" + "\n"
    r"\end{document}" + "\n"
)


def _stub_writer_run(**kwargs: Any) -> str:
    """Drop-in stub for Writer.run() that avoids real API calls and file I/O."""
    return _STUB_WRITER_RESULT


# Stub return value for Analyst.run() — mirrors SPEC §6 schema for a regression task.
_STUB_ANALYST_RESULT: dict = {
    "best_model": "XGBoost",
    "best_metric_value": 0.614,
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
        {"feature": "X1SES", "shap_mean_abs": 0.07, "direction": "positive"},
    ],
    "subgroup_performance": {
        "X1SEX": {"Male": {"rmse": 0.63, "n": 7800}, "Female": {"rmse": 0.59, "n": 7592}}
    },
    "figures_generated": ["shap_summary.png", "shap_importance.png", "pdp_X1TXMTSC.png"],
    "tables_generated": ["model_comparison.csv", "feature_importance.csv"],
    "errors": [],
    "warnings": [],
}


def _stub_analyst_run(**kwargs: Any) -> dict:
    """Drop-in stub for Analyst.run() that avoids real API/file I/O."""
    return _STUB_ANALYST_RESULT


def _make_ctx(tmp_path) -> PipelineContext:
    return PipelineContext(
        dataset_name="hsls09_public",
        raw_data_path="data/raw/nonexistent.csv",
        output_dir=str(tmp_path),
        max_revision_cycles=2,
    )


def test_full_pipeline_completes(tmp_path) -> None:
    config = load_config(CONFIG_PATH)
    ctx = _make_ctx(tmp_path)
    orchestrator = Orchestrator(ctx, config, config_path=CONFIG_PATH)
    orchestrator.problem_formulator.run = _stub_pf_run  # avoid real API/S2 calls
    orchestrator.data_engineer.run = _stub_de_run        # avoid real LLM/file I/O
    orchestrator.analyst.run = _stub_analyst_run         # avoid real LLM/file I/O
    orchestrator.critic.run = lambda **kw: {             # avoid real Opus API call
        "overall_verdict": "PASS", "overall_quality_score": 8,
        "problem_formulation_review": {"score": 8, "issues": []},
        "data_preparation_review": {"score": 8, "issues": []},
        "analysis_review": {"score": 8, "issues": []},
        "substantive_review": {"score": 8, "educational_meaningfulness": "stub", "issues": []},
        "revision_instructions": {"ProblemFormulator": None, "DataEngineer": None, "Analyst": None},
    }
    orchestrator.writer.run = _stub_writer_run           # avoid real LLM/file I/O
    result = orchestrator.run()
    assert result.current_state == "COMPLETED"
    assert "FORMULATING" in result.completed_stages
    assert "ENGINEERING" in result.completed_stages
    assert "ANALYZING" in result.completed_stages
    assert "WRITING" in result.completed_stages


def test_checkpoint_save_load(tmp_path) -> None:
    config = load_config(CONFIG_PATH)

    # First run: run to completion, saving checkpoint along the way
    ctx1 = _make_ctx(tmp_path)
    orch1 = Orchestrator(ctx1, config, config_path=CONFIG_PATH)
    orch1.problem_formulator.run = _stub_pf_run  # avoid real API/S2 calls
    orch1.data_engineer.run = _stub_de_run        # avoid real LLM/file I/O
    orch1.analyst.run = _stub_analyst_run         # avoid real LLM/file I/O
    orch1.critic.run = lambda **kw: {             # avoid real Opus API call
        "overall_verdict": "PASS", "overall_quality_score": 8,
        "problem_formulation_review": {"score": 8, "issues": []},
        "data_preparation_review": {"score": 8, "issues": []},
        "analysis_review": {"score": 8, "issues": []},
        "substantive_review": {"score": 8, "educational_meaningfulness": "stub", "issues": []},
        "revision_instructions": {"ProblemFormulator": None, "DataEngineer": None, "Analyst": None},
    }
    orch1.writer.run = _stub_writer_run           # avoid real LLM/file I/O
    orch1.run()

    # Second orchestrator pointing at same output_dir: should load checkpoint
    ctx2 = _make_ctx(tmp_path)
    orch2 = Orchestrator(ctx2, config, config_path=CONFIG_PATH)

    assert orch2.ctx.current_state == "COMPLETED"
    assert "FORMULATING" in orch2.ctx.completed_stages
    assert "WRITING" in orch2.ctx.completed_stages


def test_revision_cascade(tmp_path) -> None:
    config = load_config(CONFIG_PATH)
    ctx = _make_ctx(tmp_path)
    orch = Orchestrator(ctx, config, config_path=CONFIG_PATH)

    # Track which agents ran during the revision cascade.
    # DataEngineer is now a real agent that calls the LLM, so we stub it
    # to avoid API calls — the orchestrator state-machine behaviour is what
    # this test exercises, not the agent internals.
    revision_calls: list = []

    def tracking_de_run(**kwargs: Any) -> dict:
        revision_calls.append("DataEngineer")
        return _STUB_DE_RESULT

    def tracking_analyst_run(**kwargs: Any) -> dict:
        revision_calls.append("Analyst")
        return _STUB_ANALYST_RESULT

    orch.problem_formulator.run = _stub_pf_run  # avoid real API/S2 calls
    orch.data_engineer.run = tracking_de_run
    orch.analyst.run = tracking_analyst_run
    orch.writer.run = _stub_writer_run           # avoid real LLM/file I/O

    revise_report = {
        "overall_verdict": "REVISE",
        "overall_quality_score": 5,
        "problem_formulation_review": {"score": 7, "issues": []},
        "data_preparation_review": {"score": 5, "issues": []},
        "analysis_review": {"score": 7, "issues": []},
        "substantive_review": {"score": 6, "educational_meaningfulness": "stub", "issues": []},
        "revision_instructions": {
            "ProblemFormulator": None,
            "DataEngineer": "Increase sample size by relaxing missingness threshold.",
            "Analyst": None,
        },
    }
    pass_report = {
        "overall_verdict": "PASS",
        "overall_quality_score": 8,
        "problem_formulation_review": {"score": 8, "issues": []},
        "data_preparation_review": {"score": 8, "issues": []},
        "analysis_review": {"score": 8, "issues": []},
        "substantive_review": {"score": 8, "educational_meaningfulness": "stub", "issues": []},
        "revision_instructions": {
            "ProblemFormulator": None,
            "DataEngineer": None,
            "Analyst": None,
        },
    }

    call_count = [0]

    def mock_critic_run(**kwargs):
        call_count[0] += 1
        return revise_report if call_count[0] == 1 else pass_report

    orch.critic.run = mock_critic_run

    result = orch.run()

    # DataEngineer AND Analyst should both have been called during the revision cascade
    assert "DataEngineer" in revision_calls, "DataEngineer was not re-run during revision"
    assert "Analyst" in revision_calls, "Analyst was not re-run during revision"
    assert result.current_state == "COMPLETED"
    assert call_count[0] == 2  # Critic called twice: once REVISE, once PASS


# ---------------------------------------------------------------------------
# Tests — FindingsMemory lifecycle in Orchestrator
# ---------------------------------------------------------------------------


def test_findings_memory_disabled_by_default(tmp_path) -> None:
    """findings_memory.enabled=false (default) → self.findings_memory is None."""
    config = load_config(CONFIG_PATH)
    # Ensure enabled=false (already default in config; paranoia check)
    config["findings_memory"]["enabled"] = False
    ctx = _make_ctx(tmp_path)
    orch = Orchestrator(ctx, config, config_path=CONFIG_PATH)
    assert orch.findings_memory is None


def test_findings_memory_updated_on_complete(tmp_path) -> None:
    """When findings_memory is enabled, add_run + save should be called after COMPLETED."""
    import tempfile

    from unittest.mock import MagicMock, patch

    config = load_config(CONFIG_PATH)

    with tempfile.TemporaryDirectory() as mem_dir:
        mem_path = mem_dir + "/memory.yaml"
        config["findings_memory"]["enabled"] = True
        config["findings_memory"]["path"] = mem_path
        config["findings_memory"]["n_candidate_specs"] = 1

        ctx = _make_ctx(tmp_path)
        orch = Orchestrator(ctx, config, config_path=CONFIG_PATH)

        # Stub all agents to avoid real API calls
        orch.problem_formulator.run = _stub_pf_run
        orch.data_engineer.run = _stub_de_run
        orch.analyst.run = _stub_analyst_run
        orch.critic.run = lambda **kw: {
            "overall_verdict": "PASS",
            "overall_quality_score": 8,
            "problem_formulation_review": {"score": 8, "issues": []},
            "data_preparation_review": {"score": 8, "issues": []},
            "analysis_review": {"score": 8, "issues": []},
            "substantive_review": {
                "score": 8,
                "educational_meaningfulness": "stub",
                "issues": [],
            },
            "revision_instructions": {
                "ProblemFormulator": None,
                "DataEngineer": None,
                "Analyst": None,
            },
        }
        orch.writer.run = _stub_writer_run

        result = orch.run()
        assert result.current_state == "COMPLETED"

        # Memory file should exist with one run
        from src.findings_memory import FindingsMemory
        loaded = FindingsMemory.load(mem_path)
        assert len(loaded.runs) == 1
        assert loaded.runs[0].verdict == "PASS"


def test_findings_memory_not_updated_when_disabled(tmp_path) -> None:
    """When disabled, no memory file should be written."""
    import tempfile

    config = load_config(CONFIG_PATH)
    config["findings_memory"]["enabled"] = False

    ctx = _make_ctx(tmp_path)
    orch = Orchestrator(ctx, config, config_path=CONFIG_PATH)
    orch.problem_formulator.run = _stub_pf_run
    orch.data_engineer.run = _stub_de_run
    orch.analyst.run = _stub_analyst_run
    orch.critic.run = lambda **kw: {
        "overall_verdict": "PASS",
        "overall_quality_score": 8,
        "problem_formulation_review": {"score": 8, "issues": []},
        "data_preparation_review": {"score": 8, "issues": []},
        "analysis_review": {"score": 8, "issues": []},
        "substantive_review": {
            "score": 8,
            "educational_meaningfulness": "stub",
            "issues": [],
        },
        "revision_instructions": {
            "ProblemFormulator": None,
            "DataEngineer": None,
            "Analyst": None,
        },
    }
    orch.writer.run = _stub_writer_run
    orch.run()

    # Default memory path should NOT exist
    default_path = config["findings_memory"]["path"]
    assert not Path(default_path).exists(), (
        f"Memory file should not be created when disabled: {default_path}"
    )
