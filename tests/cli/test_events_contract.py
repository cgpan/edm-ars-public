"""The reader against the real writer: src/events.py EventSink -> RunState.

The pipeline side of the event stream is built on another branch; this
test pins the seam. Anything the sink writes must fold into a sensible
state, including a run that is resumed after an abort.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.cli import _run_support  # noqa: F401 -- installs stand-ins

from src.context import PipelineState
from src.events import EventSink

from edmars.endstates import classify
from edmars.runstate import load_state


def _emit_prefix(sink: EventSink) -> None:
    sink.emit("run.start", plain="Starting the study", task_type="prediction",
              dataset="hsls09_public", provider="deepseek", resumed=False)
    sink.emit("stage.start", stage=PipelineState.FORMULATING, cycle=0, plain="Framing the question")
    sink.emit("lit.progress", stage="FORMULATING", source="semantic_scholar", query_index=1,
              n_queries=3, papers_found=41, status="ok")
    sink.emit("llm.start", agent="ProblemFormulator", model="deepseek-v4-pro", provider="deepseek")
    sink.emit("llm.end", agent="ProblemFormulator", model="deepseek-v4-pro", prompt_tokens=9000,
              completion_tokens=800, cached_tokens=4000, cost_usd=0.0021, duration_s=12.5)
    sink.emit("stage.end", stage=PipelineState.FORMULATING, outcome="ok", duration_s=70)
    sink.emit("stage.start", stage=PipelineState.ENGINEERING, cycle=0, plain="Preparing the data")
    sink.emit("attempt.start", stage="ENGINEERING", attempt=1, max_attempts=4, timeout_s=600)


def test_live_run_from_the_real_sink(tmp_path: Path) -> None:
    run = tmp_path / "run"
    sink = EventSink(str(run))
    _emit_prefix(sink)
    state = load_state(run)  # pid comes from live_status.json: this test process
    assert state.source == "events"
    assert not state.finished
    stages = {s.key: s.status for s in state.stages}
    assert stages["FORMULATING"] == "done" and stages["ENGINEERING"] == "running"
    assert state.metrics["papers_found"] == 41
    assert state.llm_calls == 1 and state.cost_usd == pytest.approx(0.0021)
    assert state.code_running and "attempt 1 of 4" in state.now_text
    assert classify(run).kind == "running"


def test_aborted_then_resumed_run_from_the_real_sink(tmp_path: Path) -> None:
    run = tmp_path / "run"
    sink = EventSink(str(run))
    _emit_prefix(sink)
    sink.emit("attempt.end", stage="ENGINEERING", attempt=1, returncode=0)
    sink.emit("error", stage="ENGINEERING", plain="The AI account is out of credit",
              code="NO_CREDIT", message="Insufficient Balance")
    sink.emit("stage.end", stage="ENGINEERING", outcome="failed")
    sink.emit("run.end", state="ABORTED", released=False, reason_code="ABORTED", exit_code=3,
              cost_usd=0.0021)
    state = load_state(run)
    assert state.finished and state.final_state == "ABORTED"
    out = classify(run)
    assert out.kind == "stopped" and out.code == "NO_CREDIT"

    # A resume appends to the same file; numbering continues.
    sink2 = EventSink(str(run))
    sink2.emit("run.start", plain="Resuming", resumed=True, task_type="prediction")
    sink2.emit("stage.start", stage="ENGINEERING", cycle=0)
    state = load_state(run)
    assert not state.finished and state.resumed == 1
    assert {s.key: s.status for s in state.stages}["ENGINEERING"] == "running"
    sink2.emit("stage.end", stage="ENGINEERING", outcome="ok")
    for stage in ("ANALYZING", "CRITIQUING", "WRITING", "VERIFYING"):
        sink2.emit("stage.start", stage=stage)
        if stage == "CRITIQUING":
            sink2.emit("verdict", stage=stage, critic_score=8, verdict="PASS", unverified=False, cycle=0)
        if stage == "WRITING":
            sink2.emit("compile.end", stage=stage, pdf_exists=True, missing_tool=None, failed_step=None)
        sink2.emit("stage.end", stage=stage, outcome="ok")
    sink2.emit("verify.end", released=True, reason_code="CLEAN", counts={"critical": 0, "major": 0, "minor": 0})
    sink2.emit("run.end", state="COMPLETED", released=True, reason_code="CLEAN", exit_code=0, cost_usd=0.05)
    (run / "paper.pdf").write_bytes(b"%PDF")
    state = load_state(run)
    assert state.finished and state.final_state == "COMPLETED"
    assert state.cost_usd == pytest.approx(0.05)
    assert {s.key: s.status for s in state.stages}["REVISING"] == "skipped"
    out = classify(run)
    assert out.kind in ("ready", "ready_with_issues")  # no run_status/invariants.json written here


def test_a_study_over_its_spending_warning_says_so_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Setup promises a warning when a study passes the budget. The pipeline
    used to write it only to pipeline.log (a "log" event, which neither the
    console nor `edmars status` shows)."""
    import json
    import types

    import src.cost as cost
    from src.cost import CostSummary
    from src.main import _progress_line
    from src.orchestrator import Orchestrator

    run = tmp_path / "run"
    sink = EventSink(str(run))
    _emit_prefix(sink)
    orch = Orchestrator.__new__(Orchestrator)  # only _check_cost and _log are exercised
    orch.ctx = types.SimpleNamespace(output_dir=str(run), event_sink=sink, current_state="ANALYZING", log=[])
    orch.config = {"pipeline": {"cost_budget_usd": 0.10}}
    monkeypatch.setattr(cost, "load_usage_best", lambda _run_dir: ["one priced call"])
    monkeypatch.setattr(cost, "load_pricing", lambda _config: {})
    monkeypatch.setattr(cost, "summarize", lambda _usages, _pricing: CostSummary(n_calls=1, cost_usd=0.25))

    orch._check_cost()
    orch._check_cost()  # runs after every stage; the warning must not repeat

    state = load_state(run)
    over = [w for w in state.warnings if "spending warning" in w]
    assert len(over) == 1 and "US$0.25" in over[0] and "US$0.10" in over[0]
    assert any("spending warning" in line for line in state.recent)
    records = [json.loads(line) for line in (run / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    shown = [_progress_line(r) for r in records if (r.get("data") or {}).get("code") == "COST_OVER_BUDGET"]
    assert len(shown) == 1 and shown[0] and "COST_OVER_BUDGET" in shown[0]
    assert "exceeds budget" in (run / "pipeline.log").read_text(encoding="utf-8")


def test_the_review_gate_seen_through_the_real_sink(tmp_path: Path) -> None:
    """The review gate's side of the stream as fix/r3-gate writes it: its
    log lines mirrored as "log" events (agent ReviewGate), an llm.end per
    metered call with component "review", and a gate.cost running total
    after each review. The CLI follows the reviews and counts their cost."""
    from edmars.runstate import cost_line

    run = tmp_path / "run"
    sink = EventSink(str(run))
    _emit_prefix(sink)
    sink.emit("stage.end", stage="ENGINEERING", outcome="ok")
    sink.emit("stage.start", stage=PipelineState.REVIEWING, cycle=0)

    def gate_log(message: str) -> None:
        sink.emit("log", stage=PipelineState.REVIEWING, agent="ReviewGate", message=message)

    gate_log("--- Review gate cycle 1/2 ---")
    sink.emit("gate.cycle", stage="REVIEWING", cycle=1, plain="Review gate: cycle 1 of 2", max_cycles=2)
    gate_log("Running LSAR review (cycle 1, venue=EDM)")
    state = load_state(run)
    assert "reviews not yet counted" in cost_line(state)
    assert {s.key: s.detail for s in state.stages}["REVIEWING"] == "review 1 of up to 6"
    for _ in range(7):
        sink.emit("llm.end", stage="REVIEWING", cycle=1, agent="LSAR", model="deepseek-v4-flash",
                  provider="deepseek", ok=True, prompt_tokens=20000, completion_tokens=1500,
                  cached_tokens=0, cost_usd=0.005, cost_estimated=False, component="review")
    sink.emit("gate.cost", stage="REVIEWING", cycle=1,
              plain="LSAR review cycle_1 used 7 AI calls; the review gate has cost US$0.04 so far "
                    "(7 AI calls)",
              cost_usd=0.035, n_calls=7, unpriced_calls=0, cost_estimated=False)
    gate_log("LSAR review complete (cycle 1): overall_score=3.1")
    state = load_state(run)
    assert state.llm_calls == 8 and state.review_calls == 7
    assert cost_line(state) == ("Cost so far: US$0.037 (8 AI calls, including US$0.035 for the "
                                "automated peer review)")
    assert "LSAR review 1 finished, score 3.1" in state.recent
    assert not any("cycle_1" in line for line in state.recent)
