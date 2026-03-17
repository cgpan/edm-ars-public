"""Tests for FindingsMemory, RunEntry, and KnowledgeGraph."""
from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock

import pytest

from src.findings_memory import FindingsMemory, KnowledgeGraph, RunEntry


def _make_entry(**overrides) -> RunEntry:
    defaults = dict(
        run_id="run_20260315_100000",
        dataset="hsls09_public",
        task_type="prediction",
        outcome_variable="X2TXMTSCOR",
        predictor_set=["X1TXMTSCOR", "X1SES", "X1MTHEFF"],
        best_model="XGBoost",
        best_metric_value=0.82,
        primary_metric="AUC",
        verdict="PASS",
        quality_score=8,
        top_features=["X1TXMTSCOR", "X1SES", "X1MTHEFF"],
        open_questions=["Does SES moderate the effect of prior achievement?"],
        research_question="Predict 11th-grade math score from 9th-grade baseline predictors",
        timestamp="2026-03-15T10:00:00+00:00",
    )
    defaults.update(overrides)
    return RunEntry(**defaults)


class TestRunEntry:
    def test_round_trip(self) -> None:
        entry = _make_entry()
        assert RunEntry.from_dict(entry.to_dict()).run_id == entry.run_id
        assert RunEntry.from_dict(entry.to_dict()).best_metric_value == pytest.approx(0.82)

    def test_from_dict_defaults(self) -> None:
        entry = RunEntry.from_dict({})
        assert entry.run_id == ""
        assert entry.task_type == "prediction"
        assert entry.predictor_set == []
        assert entry.quality_score is None

    def test_from_pipeline_context(self) -> None:
        ctx = MagicMock()
        ctx.dataset_name = "hsls09_public"
        ctx.task_type = "prediction"
        ctx.run_start_time = ""
        ctx.research_spec = {
            "research_question": "Test RQ",
            "outcome_variable": "X2TXMTSCOR",
            "predictor_set": [
                {"variable": "X1TXMTSCOR"},
                {"variable": "X1SES"},
            ],
        }
        ctx.results_object = {
            "best_model": "XGBoost",
            "best_metric_value": 0.79,
            "primary_metric": "AUC",
            "top_features": [
                {"feature": "X1TXMTSCOR", "shap_mean_abs": 0.2},
                {"feature": "X1SES", "shap_mean_abs": 0.1},
            ],
        }
        ctx.review_report = {
            "overall_verdict": "PASS",
            "overall_quality_score": 8,
            "substantive_review": {"issues": []},
        }
        entry = RunEntry.from_pipeline_context(ctx, run_id="run_test")
        assert entry.run_id == "run_test"
        assert entry.outcome_variable == "X2TXMTSCOR"
        assert entry.best_model == "XGBoost"
        assert entry.verdict == "PASS"
        assert "X1TXMTSCOR" in entry.predictor_set
        assert "X1TXMTSCOR" in entry.top_features


class TestFindingsMemoryLoad:
    def test_load_empty_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "nonexistent.yaml")
            mem = FindingsMemory.load(path)
            assert mem.runs == []
            assert mem.knowledge_graph.studied_outcomes == {}

    def test_load_corrupt_file_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "memory.yaml")
            with open(path, "w") as f:
                f.write(":: not valid yaml ::")
            mem = FindingsMemory.load(path)
            assert mem.runs == []

    def test_load_valid_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "memory.yaml")
            mem = FindingsMemory(path)
            mem.add_run(_make_entry())
            mem.save()
            loaded = FindingsMemory.load(path)
            assert len(loaded.runs) == 1
            assert loaded.runs[0].run_id == "run_20260315_100000"


class TestFindingsMemoryAddAndSave:
    def test_add_run_and_save_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "memory.yaml")
            mem = FindingsMemory(path)
            entry = _make_entry()
            mem.add_run(entry)
            mem.save()

            loaded = FindingsMemory.load(path)
            assert len(loaded.runs) == 1
            r = loaded.runs[0]
            assert r.outcome_variable == "X2TXMTSCOR"
            assert r.best_metric_value == pytest.approx(0.82)
            assert r.verdict == "PASS"

    def test_add_multiple_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "memory.yaml")
            mem = FindingsMemory(path)
            mem.add_run(_make_entry(run_id="run_001", outcome_variable="X2TXMTSCOR"))
            mem.add_run(_make_entry(run_id="run_002", outcome_variable="X3TGPAMAT"))
            mem.save()

            loaded = FindingsMemory.load(path)
            assert len(loaded.runs) == 2
            outcomes = loaded.knowledge_graph.studied_outcomes
            assert "X2TXMTSCOR" in outcomes
            assert "X3TGPAMAT" in outcomes

    def test_atomic_save_no_corruption(self) -> None:
        """After save(), the file must be readable (tmp rename completed)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "memory.yaml")
            mem = FindingsMemory(path)
            mem.add_run(_make_entry())
            mem.save()
            # Verify no .tmp file left behind
            assert not os.path.exists(path + ".tmp")
            # Verify file is valid
            loaded = FindingsMemory.load(path)
            assert len(loaded.runs) == 1


class TestFindingsMemoryKnowledgeGraph:
    def test_studied_outcomes_updated(self) -> None:
        mem = FindingsMemory("dummy.yaml")
        mem.add_run(_make_entry(run_id="r1", outcome_variable="X2TXMTSCOR"))
        mem.add_run(_make_entry(run_id="r2", outcome_variable="X2TXMTSCOR"))
        mem.add_run(_make_entry(run_id="r3", outcome_variable="X3TGPAMAT"))
        kg = mem.knowledge_graph
        assert "X2TXMTSCOR" in kg.studied_outcomes
        assert len(kg.studied_outcomes["X2TXMTSCOR"]) == 2
        assert "r1" in kg.studied_outcomes["X2TXMTSCOR"]
        assert "r3" in kg.studied_outcomes.get("X3TGPAMAT", [])

    def test_strong_predictors_accumulated(self) -> None:
        mem = FindingsMemory("dummy.yaml")
        mem.add_run(_make_entry(top_features=["X1TXMTSCOR", "X1SES"]))
        mem.add_run(_make_entry(top_features=["X1TXMTSCOR", "X1MTHEFF"]))
        assert mem.knowledge_graph.strong_predictors["X1TXMTSCOR"] == 2
        assert mem.knowledge_graph.strong_predictors["X1SES"] == 1

    def test_open_questions_deduplicated(self) -> None:
        mem = FindingsMemory("dummy.yaml")
        q = "Does SES moderate the effect of prior achievement?"
        mem.add_run(_make_entry(open_questions=[q]))
        mem.add_run(_make_entry(open_questions=[q, "Another question?"]))
        questions = mem.knowledge_graph.open_questions
        # Same question should not appear twice
        assert questions.count(q) == 1


class TestGetStudiedOutcomes:
    def test_returns_list(self) -> None:
        mem = FindingsMemory("dummy.yaml")
        mem.add_run(_make_entry(outcome_variable="X2TXMTSCOR"))
        mem.add_run(_make_entry(outcome_variable="X3TGPAMAT"))
        outcomes = mem.get_studied_outcomes()
        assert "X2TXMTSCOR" in outcomes
        assert "X3TGPAMAT" in outcomes

    def test_empty_memory_returns_empty_list(self) -> None:
        mem = FindingsMemory("dummy.yaml")
        assert mem.get_studied_outcomes() == []


class TestToSummaryStr:
    def test_empty_memory_returns_empty_string(self) -> None:
        mem = FindingsMemory("dummy.yaml")
        assert mem.to_summary_str() == ""

    def test_non_empty_memory_produces_summary(self) -> None:
        mem = FindingsMemory("dummy.yaml")
        mem.add_run(_make_entry())
        summary = mem.to_summary_str()
        assert summary != ""
        assert "X2TXMTSCOR" in summary
        assert "Prior runs recorded" in summary

    def test_summary_contains_studied_outcomes(self) -> None:
        mem = FindingsMemory("dummy.yaml")
        mem.add_run(_make_entry(outcome_variable="X2TXMTSCOR"))
        mem.add_run(_make_entry(outcome_variable="X3TGPAMAT", run_id="run_002"))
        summary = mem.to_summary_str()
        assert "X2TXMTSCOR" in summary
        assert "X3TGPAMAT" in summary

    def test_summary_reasonable_length(self) -> None:
        """Summary should be concise — estimate < 3000 chars for typical use."""
        mem = FindingsMemory("dummy.yaml")
        for i in range(5):
            mem.add_run(_make_entry(run_id=f"run_{i:03d}"))
        summary = mem.to_summary_str()
        assert len(summary) < 5000  # generous upper bound
