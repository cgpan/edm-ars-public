"""What a resumed run reads back from its checkpoint.

Checkpoint atomicity and a readable error for a corrupt one (D1), and the
checkpoint -- not the command line -- deciding which run this is (D2).

Every test stubs the agents and the LaTeX compile, so nothing here calls a
provider or needs TeX installed.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import pytest

from src.config import load_config
from src.context import PipelineContext, PipelineState
from src.orchestrator import CheckpointCorruptError, Orchestrator
from tests.test_end_to_end import (
    _PASS_REVIEW,
    _analyst_stub,
    _critic_stub,
    _de_stub,
    _pf_stub,
    _writer_stub,
)

CONFIG_PATH = str(Path(__file__).parent.parent / "config.yaml")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _fake_compile_ok(output_dir: str, *_a: Any, **_k: Any) -> dict:
    Path(output_dir, "paper.pdf").write_bytes(b"%PDF-1.5 stub")
    Path(output_dir, "paper.log").write_text(
        "This is pdfTeX, Version 3.14\nOutput written on paper.pdf (1 page).\n",
        encoding="utf-8",
    )
    return {
        "success": True,
        "steps": [
            {"cmd": "pdflatex -interaction=nonstopmode paper.tex",
             "returncode": 0, "stdout": "", "stderr": ""}
        ],
    }


def _fake_compile_no_pdflatex(output_dir: str, *_a: Any, **_k: Any) -> dict:
    """What compile_latex returns when pdflatex is not installed: rc -1,
    and nothing at all written to the run directory."""
    return {
        "success": False,
        "steps": [
            {"cmd": "pdflatex -interaction=nonstopmode paper.tex",
             "returncode": -1, "stdout": "",
             "stderr": "'pdflatex' not found - is it installed and on PATH?"}
        ],
    }


@pytest.fixture(autouse=True)
def _compile_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default: a compile that writes a clean log and a PDF."""
    monkeypatch.setattr("src.orchestrator.compile_latex", _fake_compile_ok)


def _config(tmp_path: Path, **overrides: Any) -> dict:
    cfg = copy.deepcopy(load_config(CONFIG_PATH))
    # Keep the cross-run memory inside the test's own directory.
    cfg.setdefault("findings_memory", {})["path"] = str(
        tmp_path / "memory" / "memory.yaml"
    )
    cfg.setdefault("review_gate", {})["enabled"] = False
    for dotted, value in overrides.items():
        node = cfg
        keys = dotted.split("__")
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
    return cfg


def _ctx(out: Path, **kw: Any) -> PipelineContext:
    return PipelineContext(
        dataset_name=kw.pop("dataset_name", "hsls09_public"),
        raw_data_path=kw.pop("raw_data_path", str(out / "no_such_raw.csv")),
        output_dir=str(out),
        max_revision_cycles=kw.pop("max_revision_cycles", 2),
        **kw,
    )


def _orch(out: Path, cfg: dict, **ctx_kw: Any) -> Orchestrator:
    return Orchestrator(_ctx(out, **ctx_kw), cfg, config_path=CONFIG_PATH)


def _wire(orch: Orchestrator, review: dict = _PASS_REVIEW) -> dict[str, int]:
    out = orch.ctx.output_dir
    calls = {"pf": 0, "de": 0, "analyst": 0, "critic": 0, "writer": 0}

    def counted(name: str, fn: Any) -> Any:
        def run(**kw: Any) -> Any:
            calls[name] += 1
            return fn(**kw)
        return run

    orch.problem_formulator.run = counted("pf", lambda **kw: _pf_stub(out, **kw))
    orch.data_engineer.run = counted("de", lambda **kw: _de_stub(out, **kw))
    orch.analyst.run = counted("analyst", lambda **kw: _analyst_stub(out, **kw))
    orch.critic.run = counted("critic", lambda **kw: _critic_stub(out, review, **kw))
    orch.writer.run = counted("writer", lambda **kw: _writer_stub(out, **kw))
    return calls


def _status(out: Path) -> dict:
    return json.loads((out / "run_status.json").read_text(encoding="utf-8"))


def _events(out: Path) -> list[dict]:
    path = out / "events.jsonl"
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _no_invariants(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate the release decision from the stub paper's own findings."""
    monkeypatch.setattr("src.invariants.run_invariants", lambda _d: [])


# ---------------------------------------------------------------------------
# D1 -- atomic checkpoint, readable failure on a corrupt one
# ---------------------------------------------------------------------------


def test_checkpoint_survives_a_failed_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    orch = _orch(tmp_path, _config(tmp_path))
    orch.ctx.research_spec = {"q": "good"}
    orch._save_checkpoint()
    good = (tmp_path / "checkpoint.json").read_text(encoding="utf-8")

    # 1. An unserialisable value fails before the file is touched.
    orch.ctx.research_spec = {"q": object()}
    with pytest.raises(TypeError):
        orch._save_checkpoint()
    assert (tmp_path / "checkpoint.json").read_text(encoding="utf-8") == good

    # 2. A failure mid-write (here: at fsync) leaves the old file intact
    #    and no temp file behind.
    orch.ctx.research_spec = {"q": "newer"}

    def fsync_fails(_fd: int) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "fsync", fsync_fails)
    with pytest.raises(OSError):
        orch._save_checkpoint()
    assert (tmp_path / "checkpoint.json").read_text(encoding="utf-8") == good
    assert not [p for p in os.listdir(tmp_path) if p.endswith(".tmp")]


def test_corrupt_checkpoint_raises_a_named_error(tmp_path: Path) -> None:
    (tmp_path / "checkpoint.json").write_text('{"current_state": "ANA', encoding="utf-8")
    with pytest.raises(CheckpointCorruptError) as info:
        _orch(tmp_path, _config(tmp_path))
    assert "checkpoint.json" in str(info.value)


# ---------------------------------------------------------------------------
# D2 -- the checkpoint, not the command line, says what run this is
# ---------------------------------------------------------------------------


def test_resume_adopts_the_checkpoints_identity(tmp_path: Path) -> None:
    locked = {"task_type": "causal_did", "research_question": "gap change?"}
    cp = PipelineContext(
        dataset_name="did_els_hsls_panel",
        raw_data_path=str(tmp_path / "panel.csv"),
        output_dir=str(tmp_path),
        task_type="causal_did",
        locked_research_spec=locked,
    )
    cp.current_state = PipelineState.ANALYZING
    cp.completed_stages = ["FORMULATING", "ENGINEERING"]
    cp.paper_outline = {"sections": ["intro"]}
    cp.run_start_time = "2026-09-01T10:00:00+00:00"
    cp.abort_info = None
    (tmp_path / "checkpoint.json").write_text(json.dumps(cp.to_dict()), encoding="utf-8")

    with pytest.warns(RuntimeWarning):
        orch = _orch(tmp_path, _config(tmp_path))  # CLI said hsls09_public/prediction

    assert orch.ctx.dataset_name == "did_els_hsls_panel"
    assert orch.ctx.task_type == "causal_did"
    assert orch.ctx.raw_data_path == str(tmp_path / "panel.csv")
    assert orch.ctx.locked_research_spec == locked
    assert type(orch.task_template).__name__ == "CausalDIDTemplate"
    assert orch.ctx.paper_outline == {"sections": ["intro"]}
    assert orch.ctx.run_start_time == "2026-09-01T10:00:00+00:00"
    log = (tmp_path / "pipeline.log").read_text(encoding="utf-8")
    assert "disagrees with the checkpoint" in log


def test_checkpoint_round_trip_restores_every_field(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    orch = _orch(tmp_path, cfg)
    sentinels = {
        "research_spec": {"s": 1},
        "literature_context": {"papers": [{"paperId": "p"}]},
        "retrieved_literature": {"papers": []},
        "data_report": {"analytic_n": 5000},
        "results_object": {"best_model": "X"},
        "review_report": {"overall_verdict": "PASS"},
        "paper_text": "tex",
        "paper_outline": {"o": 1},
        "review_gate_result": {"ran": False},
        "completed_stages": ["FORMULATING"],
        "revision_cycle": 1,
        "errors": ["e1"],
        "abort_info": {"stage": "ENGINEERING", "code": "NETWORK"},
        "run_start_time": "2026-09-01T00:00:00+00:00",
    }
    for k, v in sentinels.items():
        setattr(orch.ctx, k, v)
    orch.ctx.current_state = PipelineState.ENGINEERING
    orch._save_checkpoint()

    again = _orch(tmp_path, cfg)
    for k, v in sentinels.items():
        assert getattr(again.ctx, k) == v, k
    assert again.ctx.current_state == PipelineState.ENGINEERING
