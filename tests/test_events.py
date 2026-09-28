"""src/events.py: the live side channel must be append-only, resumable and
must never raise into the pipeline."""
from __future__ import annotations

import json
import os

from src.context import PipelineContext, PipelineState
from src.events import EventLogList, EventSink, attach, emit


def _read(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_emit_appends_versioned_records(tmp_path) -> None:
    sink = EventSink(str(tmp_path), run_id="r1")
    sink.emit("stage.start", stage=PipelineState.ANALYZING, cycle=0, plain="Running the analysis")
    sink.emit("llm.end", agent="Analyst", cost_usd=0.01)
    rows = _read(os.path.join(tmp_path, "events.jsonl"))
    assert [r["seq"] for r in rows] == [1, 2]
    assert rows[0]["v"] == 1 and rows[0]["stage"] == "ANALYZING"
    assert rows[0]["ts"].endswith("Z")
    assert rows[1]["data"]["cost_usd"] == 0.01
    status = json.load(open(os.path.join(tmp_path, "live_status.json"), encoding="utf-8"))
    assert status["stage"] == "ANALYZING" and status["llm_calls"] == 1


def test_sequence_continues_after_resume(tmp_path) -> None:
    EventSink(str(tmp_path)).emit("run.start")
    sink2 = EventSink(str(tmp_path))
    sink2.emit("run.start", resumed=True)
    rows = _read(os.path.join(tmp_path, "events.jsonl"))
    assert [r["seq"] for r in rows] == [1, 2]


def test_emit_never_raises_on_unwritable_dir(tmp_path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x")
    sink = EventSink(str(blocker / "sub"))  # a path under a regular file
    sink.emit("stage.start", stage="FORMULATING")  # must not raise


def test_log_list_mirrors_notes_but_not_token_rows(tmp_path) -> None:
    sink = EventSink(str(tmp_path))
    log = EventLogList([], sink=sink)
    log.append({"agent": "ProblemFormulator", "message": "S2 query 1/3"})
    log.append({"agent": "Analyst", "tokens_used": 10})
    rows = _read(os.path.join(tmp_path, "events.jsonl"))
    assert len(rows) == 1 and rows[0]["type"] == "agent.note"
    assert rows[0]["data"]["message"] == "S2 query 1/3"
    assert len(log) == 2


def test_attach_rewraps_after_checkpoint_load(tmp_path) -> None:
    ctx = PipelineContext(dataset_name="d", raw_data_path="x", output_dir=str(tmp_path))
    attach(ctx, str(tmp_path))
    ctx.log = list(ctx.log)  # what a checkpoint load does
    attach(ctx, str(tmp_path))
    assert isinstance(ctx.log, EventLogList)
    ctx.log.append({"agent": "Orchestrator", "message": "hello"})
    emit(ctx, "stage.end", stage="FORMULATING")
    types = [r["type"] for r in _read(os.path.join(tmp_path, "events.jsonl"))]
    assert types == ["agent.note", "stage.end"]
    assert "event_sink" not in ctx.to_dict()


def test_emit_without_sink_is_a_noop() -> None:
    ctx = PipelineContext(dataset_name="d", raw_data_path="x", output_dir="y")
    emit(ctx, "stage.start", stage="FORMULATING")  # no sink attached: nothing happens
