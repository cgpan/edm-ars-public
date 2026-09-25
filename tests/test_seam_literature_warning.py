"""A degraded literature search is announced once, not twice.

fix/wp-prov's ProblemFormulator writes ``retrieval_status`` and emits a
``warning`` event (code LITERATURE_DEGRADED) when the search came back
thin; fix/wp-orch's orchestrator emitted the same warning again after the
stage. The orchestrator still logs the line and records it in the run's
errors, but emits the event only for a context the ProblemFormulator did
not annotate (an older checkpoint, a stub).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src import events
from tests.test_end_to_end import _pf_stub
from tests.test_orchestrator_terminal import _config, _orch, _wire


def _degraded_warnings(out: Path) -> list[dict]:
    path = out / "events.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return [r for r in rows if r["type"] == "warning"
            and r["data"].get("code") == "LITERATURE_DEGRADED"]


def _no_invariants(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.invariants.run_invariants", lambda _d: [])


def test_a_warning_the_formulator_emitted_is_not_emitted_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_invariants(monkeypatch)
    orch = _orch(tmp_path, _config(tmp_path))
    _wire(orch)
    out = orch.ctx.output_dir

    def pf(**_kw: Any) -> dict:
        res = _pf_stub(out)
        lit = dict(res["literature_context"])
        lit["papers"] = []
        lit["retrieval_status"] = {"semantic_scholar": "failed", "arxiv": "ok",
                                   "n_papers": 0, "degraded": True}
        # What the real ProblemFormulator does when the pool is degraded.
        events.emit(orch.ctx, "warning", code="LITERATURE_DEGRADED",
                    message="Literature retrieval degraded")
        return {**res, "literature_context": lit}

    orch.problem_formulator.run = pf
    ctx = orch.run()

    assert len(_degraded_warnings(tmp_path)) == 1
    # Still in the run's own record.
    assert any("Literature retrieval degraded" in e for e in ctx.errors)
    assert "WARNING: Literature retrieval degraded" in (
        tmp_path / "pipeline.log").read_text(encoding="utf-8")


def test_a_context_without_retrieval_status_is_announced_here(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_invariants(monkeypatch)
    orch = _orch(tmp_path, _config(tmp_path))
    _wire(orch)
    out = orch.ctx.output_dir

    def pf(**_kw: Any) -> dict:
        res = _pf_stub(out)
        lit = {k: v for k, v in res["literature_context"].items()
               if k != "retrieval_status"}
        lit["papers"] = []
        return {**res, "literature_context": lit}

    orch.problem_formulator.run = pf
    orch.run()

    assert len(_degraded_warnings(tmp_path)) == 1
