"""The test suite must never touch the live findings memory.

tests/conftest.py redirects the repository's findings_memory/ directory
to a per-test temp dir. These tests pin that redirect: a run the suite
drives to a terminal state must not leave a fake entry behind for the
next real run's ProblemFormulator to read.
"""
from __future__ import annotations

from pathlib import Path

from src.findings_memory import FindingsMemory, RunEntry

REPO_ROOT = Path(__file__).resolve().parents[1]
LIVE = REPO_ROOT / "findings_memory" / "memory.yaml"


def _entry(run_id: str) -> RunEntry:
    return RunEntry(
        run_id=run_id,
        dataset="hsls09_public",
        task_type="prediction",
        outcome_variable="X3TGPAMAT",
        predictor_set=[],
        best_model="",
        best_metric_value=0.0,
        primary_metric="",
        verdict="PASS",
        quality_score=None,
        top_features=[],
        open_questions=[],
        research_question="",
        timestamp="2026-01-01T00:00:00",
    )


def test_live_path_is_redirected_for_save(tmp_path: Path) -> None:
    before = LIVE.read_bytes() if LIVE.exists() else None
    mem = FindingsMemory(str(LIVE))
    mem.add_run(_entry("isolation_probe"))
    mem.save()
    assert Path(mem.path).resolve() != LIVE.resolve()
    assert Path(mem.path).exists()
    after = LIVE.read_bytes() if LIVE.exists() else None
    assert after == before


def test_relative_live_path_is_redirected(monkeypatch) -> None:
    monkeypatch.chdir(REPO_ROOT)
    mem = FindingsMemory.load("findings_memory/memory.yaml")
    assert Path(mem.path).resolve() != LIVE.resolve()
    # Whatever a developer's real memory holds, a test sees an empty store.
    assert mem.runs == []


def test_other_paths_pass_through(tmp_path: Path) -> None:
    path = tmp_path / "memory.yaml"
    mem = FindingsMemory(str(path))
    mem.add_run(_entry("kept"))
    mem.save()
    assert mem.path == str(path)
    assert [r.run_id for r in FindingsMemory.load(str(path)).runs] == ["kept"]
