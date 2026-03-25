"""End-to-end mock pipeline tests.

All agent .run() methods are replaced with deterministic stubs that also write
the files each real agent would produce.  Tests verify:
  - Full pipeline INITIALIZED → COMPLETED with all output files present
  - Checkpoint is saved after every stage
  - REVISE loop: Critic returns REVISE once then PASS (DataEngineer + Analyst re-run)
  - ABORT: Critic returns ABORT, pipeline stops, Writer never runs
  - validation_passed=False and analytic_n < 1000 both trigger ABORT
  - Max revision cycles exhausted → Writer runs with UNVERIFIED flag
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from src.config import load_config
from src.context import PipelineContext, PipelineState
from src.orchestrator import Orchestrator

CONFIG_PATH = str(Path(__file__).parent.parent / "config.yaml")

# ── Canned pipeline payloads ───────────────────────────────────────────────

_RESEARCH_SPEC: dict = {
    "research_question": "Can we predict 12th-grade math GPA from 9th-grade factors?",
    "outcome_variable": "X3TGPAMAT",
    "outcome_type": "continuous",
    "predictor_set": [
        {"variable": "X1TXMTSC", "rationale": "Prior math achievement.", "wave": "base_year"},
        {"variable": "X1MTHEFF", "rationale": "Math self-efficacy.", "wave": "base_year"},
    ],
    "target_population": "full sample",
    "subgroup_analyses": ["X1SEX", "X1RACE"],
    "expected_contribution": "Novel combination of motivational constructs.",
    "potential_limitations": ["Multilevel structure not modeled"],
    "novelty_score_self_assessment": 4,
}

_LITERATURE_CONTEXT: dict = {
    "search_query": "math GPA prediction educational data mining",
    "papers": [
        {
            "paperId": "abc123",
            "title": "Predicting Academic Achievement with ML",
            "authors": ["Smith, J.", "Doe, A."],
            "year": 2022,
            "abstract": "We use ML methods to predict academic achievement.",
        }
    ],
    "novelty_evidence": "Prior work focuses on test scores; this study predicts GPA.",
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
    "n_predictors_raw": 12,
    "n_predictors_encoded": 28,
    "missingness_summary": {
        "X1TXMTSC": {"pct_missing": 1.2, "imputation_method": "median"},
    },
    "variables_flagged": [],
    "validation_passed": True,
    "warnings": [
        "Multilevel structure (students nested in schools) is not modeled. This is a limitation."
    ],
}

_RESULTS: dict = {
    "best_model": "XGBoost",
    "best_metric_value": 0.614,
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
    "figures_generated": ["shap_summary.png", "shap_importance.png", "pdp_X1TXMTSC.png"],
    "tables_generated": ["model_comparison.csv", "feature_importance.csv"],
    "errors": [],
    "warnings": [],
}

_PASS_REVIEW: dict = {
    "overall_verdict": "PASS",
    "overall_quality_score": 8,
    "problem_formulation_review": {"score": 8, "issues": []},
    "data_preparation_review": {"score": 8, "issues": []},
    "analysis_review": {"score": 8, "issues": []},
    "substantive_review": {
        "score": 8,
        "educational_meaningfulness": "Top features make educational sense.",
        "issues": [],
    },
    "revision_instructions": {
        "ProblemFormulator": None,
        "DataEngineer": None,
        "Analyst": None,
    },
}

_REVISE_REVIEW: dict = {
    "overall_verdict": "REVISE",
    "overall_quality_score": 5,
    "problem_formulation_review": {"score": 7, "issues": []},
    "data_preparation_review": {
        "score": 5,
        "issues": [
            {
                "severity": "major",
                "category": "imputation",
                "description": "Imputation threshold too strict.",
                "recommendation": "Use IterativeImputer with 5 iterations.",
                "target_agent": "DataEngineer",
            }
        ],
    },
    "analysis_review": {"score": 7, "issues": []},
    "substantive_review": {
        "score": 6,
        "educational_meaningfulness": "Acceptable.",
        "issues": [],
    },
    "revision_instructions": {
        "ProblemFormulator": None,
        "DataEngineer": "Relax missingness threshold and apply IterativeImputer.",
        "Analyst": None,
    },
}

_ABORT_REVIEW: dict = {
    "overall_verdict": "ABORT",
    "overall_quality_score": 2,
    "problem_formulation_review": {
        "score": 2,
        "issues": [
            {
                "severity": "critical",
                "category": "leakage",
                "description": "Outcome variable found in predictor set.",
                "recommendation": "Remove outcome from predictors.",
                "target_agent": "DataEngineer",
            }
        ],
    },
    "data_preparation_review": {"score": 2, "issues": []},
    "analysis_review": {"score": 2, "issues": []},
    "substantive_review": {
        "score": 2,
        "educational_meaningfulness": "Uninterpretable due to leakage.",
        "issues": [],
    },
    "revision_instructions": {
        "ProblemFormulator": None,
        "DataEngineer": None,
        "Analyst": None,
    },
}

_PAPER_TEX = (
    r"\documentclass[sigconf]{acmart}" + "\n"
    r"\usepackage{booktabs}" + "\n"
    r"\begin{document}" + "\n"
    r"\maketitle" + "\n"
    r"\begin{abstract}Mock abstract.\end{abstract}" + "\n"
    r"\section{Introduction}Mock introduction." + "\n"
    r"\end{document}" + "\n"
)

_BIBTEX = (
    "@inproceedings{abc123,\n"
    "  author    = {Smith, J. and Doe, A.},\n"
    "  title     = {Predicting Academic Achievement with ML},\n"
    "  year      = {2022},\n"
    "  booktitle = {Proceedings of the EDM Conference},\n"
    "}\n"
)

# ── Stub run functions ─────────────────────────────────────────────────────

def _pf_stub(output_dir: str, **_kw: Any) -> dict:
    return {"research_spec": _RESEARCH_SPEC, "literature_context": _LITERATURE_CONTEXT}


def _de_stub(output_dir: str, **_kw: Any) -> dict:
    with open(os.path.join(output_dir, "data_report.json"), "w") as f:
        json.dump(_DATA_REPORT, f)
    return _DATA_REPORT


def _analyst_stub(output_dir: str, **_kw: Any) -> dict:
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(_RESULTS, f)
    return _RESULTS


def _critic_stub(output_dir: str, review: dict, **_kw: Any) -> dict:
    with open(os.path.join(output_dir, "review_report.json"), "w") as f:
        json.dump(review, f)
    return review


def _writer_stub(output_dir: str, **_kw: Any) -> str:
    with open(os.path.join(output_dir, "paper.tex"), "w", encoding="utf-8") as f:
        f.write(_PAPER_TEX)
    with open(os.path.join(output_dir, "references.bib"), "w", encoding="utf-8") as f:
        f.write(_BIBTEX)
    return _PAPER_TEX


# ── Helper builders ────────────────────────────────────────────────────────

def _make_ctx(tmp_path: Path, max_revision_cycles: int = 2) -> PipelineContext:
    return PipelineContext(
        dataset_name="hsls09_public",
        raw_data_path="data/raw/nonexistent.csv",
        output_dir=str(tmp_path),
        max_revision_cycles=max_revision_cycles,
    )


def _make_orch(tmp_path: Path, max_revision_cycles: int = 2) -> Orchestrator:
    config = load_config(CONFIG_PATH)
    ctx = _make_ctx(tmp_path, max_revision_cycles=max_revision_cycles)
    return Orchestrator(ctx, config, config_path=CONFIG_PATH)


def _wire_stubs(orch: Orchestrator, critic_review: dict = _PASS_REVIEW) -> None:
    """Attach deterministic stub methods to all agents."""
    out = orch.ctx.output_dir
    orch.problem_formulator.run = lambda **kw: _pf_stub(out, **kw)
    orch.data_engineer.run = lambda **kw: _de_stub(out, **kw)
    orch.analyst.run = lambda **kw: _analyst_stub(out, **kw)
    orch.critic.run = lambda **kw: _critic_stub(out, critic_review, **kw)
    orch.writer.run = lambda **kw: _writer_stub(out, **kw)


# ── Tests ──────────────────────────────────────────────────────────────────

def test_happy_path_completes(tmp_path: Path) -> None:
    """Full pipeline INITIALIZED → COMPLETED with all SPEC §7 output files present."""
    orch = _make_orch(tmp_path)
    _wire_stubs(orch)

    result = orch.run()

    assert result.current_state == PipelineState.COMPLETED

    # All SPEC §7 output files must exist
    required = [
        "checkpoint.json",
        "config_snapshot.yaml",
        "pipeline.log",
        "research_spec.json",
        "literature_context.json",
        "data_report.json",
        "results.json",
        "review_report.json",
        "paper.tex",
        "references.bib",
    ]
    for fname in required:
        assert (tmp_path / fname).exists(), f"Missing required output file: {fname}"

    # Correct completed_stages sequence (REVIEWING appended when review_gate enabled)
    core_stages = ["FORMULATING", "ENGINEERING", "ANALYZING", "CRITIQUING", "WRITING"]
    assert result.completed_stages[:5] == core_stages
    if len(result.completed_stages) > 5:
        assert result.completed_stages[5] == "REVIEWING"

    # research_spec.json has correct content
    spec = json.loads((tmp_path / "research_spec.json").read_text())
    assert spec["outcome_variable"] == "X3TGPAMAT"

    # pipeline.log captures key events
    log = (tmp_path / "pipeline.log").read_text()
    assert "FORMULATING" in log
    assert "ENGINEERING" in log
    assert "ANALYZING" in log
    assert "WRITING" in log

    # config_snapshot.yaml was copied
    assert (tmp_path / "config_snapshot.yaml").stat().st_size > 0


def test_checkpoint_saved_after_each_stage(tmp_path: Path) -> None:
    """Checkpoint file is written after every stage; final checkpoint is COMPLETED."""
    orch = _make_orch(tmp_path)
    checkpoints: list[dict] = []

    original_save = orch._save_checkpoint

    def tracking_save() -> None:
        original_save()
        checkpoints.append(
            json.loads((tmp_path / "checkpoint.json").read_text())
        )

    orch._save_checkpoint = tracking_save  # type: ignore[method-assign]
    _wire_stubs(orch)
    orch.run()

    # At least 5 saves: FORMULATING, ENGINEERING, ANALYZING, CRITIQUING, WRITING
    assert len(checkpoints) >= 5

    # Each successive checkpoint accumulates stages
    stages_seen: set[str] = set()
    for cp in checkpoints:
        for stage in cp["completed_stages"]:
            stages_seen.add(stage)

    for stage in ("FORMULATING", "ENGINEERING", "ANALYZING", "WRITING"):
        assert stage in stages_seen, f"Stage {stage} never appeared in any checkpoint"

    # Final checkpoint is COMPLETED
    final = json.loads((tmp_path / "checkpoint.json").read_text())
    assert final["current_state"] == "COMPLETED"


def test_checkpoint_resume(tmp_path: Path) -> None:
    """Second Orchestrator pointing at same output_dir resumes from checkpoint."""
    config = load_config(CONFIG_PATH)

    # First run to completion
    ctx1 = _make_ctx(tmp_path)
    orch1 = Orchestrator(ctx1, config, config_path=CONFIG_PATH)
    _wire_stubs(orch1)
    orch1.run()

    # Second orchestrator loads checkpoint without running any agents
    ctx2 = _make_ctx(tmp_path)
    orch2 = Orchestrator(ctx2, config, config_path=CONFIG_PATH)

    assert orch2.ctx.current_state == PipelineState.COMPLETED
    assert "FORMULATING" in orch2.ctx.completed_stages
    assert "WRITING" in orch2.ctx.completed_stages
    assert orch2.ctx.research_spec is not None


def test_revise_then_pass(tmp_path: Path) -> None:
    """Critic returns REVISE once then PASS; DataEngineer and Analyst each run twice."""
    orch = _make_orch(tmp_path)
    out = orch.ctx.output_dir

    counts: dict[str, int] = {"critic": 0, "de": 0, "analyst": 0}

    def critic_run(**kw: Any) -> dict:
        counts["critic"] += 1
        review = _REVISE_REVIEW if counts["critic"] == 1 else _PASS_REVIEW
        return _critic_stub(out, review, **kw)

    def de_run(**kw: Any) -> dict:
        counts["de"] += 1
        return _de_stub(out, **kw)

    def analyst_run(**kw: Any) -> dict:
        counts["analyst"] += 1
        return _analyst_stub(out, **kw)

    orch.problem_formulator.run = lambda **kw: _pf_stub(out, **kw)
    orch.data_engineer.run = de_run
    orch.analyst.run = analyst_run
    orch.critic.run = critic_run
    orch.writer.run = lambda **kw: _writer_stub(out, **kw)

    result = orch.run()

    assert result.current_state == PipelineState.COMPLETED
    assert counts["critic"] == 2, "Critic should be called twice (REVISE + PASS)"
    # _REVISE_REVIEW targets DataEngineer → cascade re-runs DataEngineer + Analyst
    assert counts["de"] == 2, "DataEngineer should be re-run during revision"
    assert counts["analyst"] == 2, "Analyst should be re-run during revision"
    assert result.revision_cycle == 1
    # paper.tex and references.bib must still be written
    assert (tmp_path / "paper.tex").exists()
    assert (tmp_path / "references.bib").exists()


def test_abort_on_critic_abort(tmp_path: Path) -> None:
    """Critic issues ABORT → pipeline stops in ABORTED state; Writer never runs."""
    orch = _make_orch(tmp_path)
    out = orch.ctx.output_dir

    writer_called = [False]

    def writer_run(**kw: Any) -> str:
        writer_called[0] = True
        return _writer_stub(out, **kw)

    _wire_stubs(orch, critic_review=_ABORT_REVIEW)
    orch.writer.run = writer_run  # override writer to track calls

    result = orch.run()

    assert result.current_state == PipelineState.ABORTED
    assert not writer_called[0], "Writer must not run after ABORT verdict"
    assert "WRITING" not in result.completed_stages
    assert len(result.errors) > 0
    # Checkpoint must still be saved even on abort
    assert (tmp_path / "checkpoint.json").exists()
    cp = json.loads((tmp_path / "checkpoint.json").read_text())
    assert cp["current_state"] == "ABORTED"


def test_abort_on_validation_failed(tmp_path: Path) -> None:
    """DataEngineer returns validation_passed=False → ABORT before Analyst runs."""
    orch = _make_orch(tmp_path)
    out = orch.ctx.output_dir

    failed_report = {**_DATA_REPORT, "validation_passed": False, "analytic_n": 5000}
    analyst_called = [False]

    def analyst_run(**kw: Any) -> dict:
        analyst_called[0] = True
        return _analyst_stub(out, **kw)

    orch.problem_formulator.run = lambda **kw: _pf_stub(out, **kw)
    orch.data_engineer.run = lambda **kw: failed_report
    orch.analyst.run = analyst_run
    orch.critic.run = lambda **kw: _critic_stub(out, _PASS_REVIEW, **kw)
    orch.writer.run = lambda **kw: _writer_stub(out, **kw)

    result = orch.run()

    assert result.current_state == PipelineState.ABORTED
    assert not analyst_called[0], "Analyst must not run after validation failure"
    assert any("validation_passed" in e or "ENGINEERING" in e for e in result.errors)


def test_abort_on_low_analytic_n(tmp_path: Path) -> None:
    """DataEngineer returns analytic_n < 1000 → ABORT (SPEC §8)."""
    orch = _make_orch(tmp_path)
    out = orch.ctx.output_dir

    low_n_report = {**_DATA_REPORT, "validation_passed": True, "analytic_n": 800}
    orch.problem_formulator.run = lambda **kw: _pf_stub(out, **kw)
    orch.data_engineer.run = lambda **kw: low_n_report
    orch.analyst.run = lambda **kw: _analyst_stub(out, **kw)
    orch.critic.run = lambda **kw: _critic_stub(out, _PASS_REVIEW, **kw)
    orch.writer.run = lambda **kw: _writer_stub(out, **kw)

    result = orch.run()

    assert result.current_state == PipelineState.ABORTED
    assert any("analytic_n" in e for e in result.errors)


def test_max_revisions_unverified(tmp_path: Path) -> None:
    """After max_revision_cycles of REVISE, Writer runs and review marked UNVERIFIED."""
    orch = _make_orch(tmp_path, max_revision_cycles=1)
    out = orch.ctx.output_dir

    writer_called = [False]

    def writer_run(**kw: Any) -> str:
        writer_called[0] = True
        return _writer_stub(out, **kw)

    # Critic always returns REVISE — orchestrator should exhaust cycles and go to WRITING
    orch.problem_formulator.run = lambda **kw: _pf_stub(out, **kw)
    orch.data_engineer.run = lambda **kw: _de_stub(out, **kw)
    orch.analyst.run = lambda **kw: _analyst_stub(out, **kw)
    orch.critic.run = lambda **kw: _critic_stub(out, _REVISE_REVIEW, **kw)
    orch.writer.run = writer_run

    result = orch.run()

    assert result.current_state == PipelineState.COMPLETED
    assert writer_called[0], "Writer must run even when max cycles exhausted"
    assert result.review_report is not None
    assert result.review_report.get("unverified") is True
    assert "WRITING" in result.completed_stages


def test_pipeline_log_has_all_stages(tmp_path: Path) -> None:
    """pipeline.log should contain timestamped entries for every stage."""
    orch = _make_orch(tmp_path)
    _wire_stubs(orch)
    orch.run()

    log_text = (tmp_path / "pipeline.log").read_text()
    for keyword in ("FORMULATING", "ENGINEERING", "ANALYZING", "CRITIQUING", "WRITING"):
        assert keyword in log_text, f"pipeline.log missing entry for {keyword}"
