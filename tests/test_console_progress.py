"""Progress on the console while a run works (D8).

The command line echoes a selection of the run's events to stderr -- a
stage starting or ending, a rate-limit wait, a failed generated-code
attempt, a warning -- so a 20-45 minute run no longer looks like a hang.
``--quiet`` turns it off; events.jsonl and pipeline.log are unaffected.
"""
from __future__ import annotations

from pathlib import Path

import pytest


def test_progress_lines_for_the_events_a_user_waits_on() -> None:
    from src.main import _progress_line

    assert _progress_line({"type": "stage.start", "stage": "ENGINEERING",
                           "plain": "Preparing the data", "cycle": 0}) == "Preparing the data"
    assert "revision cycle 1" in _progress_line(
        {"type": "stage.start", "stage": "ANALYZING", "plain": "Analysing", "cycle": 1})
    assert _progress_line({"type": "llm.wait", "plain": "Waiting 60 s for deepseek"}) \
        == "  Waiting 60 s for deepseek"
    failed = _progress_line({"type": "attempt.end", "data": {
        "attempt": 1, "max_attempts": 3, "returncode": 1, "error_class": "KeyError"}})
    assert failed == ("  generated code attempt 1 of 3 failed (KeyError); "
                      "asking the model to fix it")
    assert _progress_line({"type": "attempt.end", "data": {
        "attempt": 1, "max_attempts": 3, "returncode": 0}}) is None
    assert _progress_line({"type": "warning", "data": {
        "code": "NO_PDF", "message": "LaTeX produced no paper.pdf"}}) \
        == "  warning [NO_PDF]: LaTeX produced no paper.pdf"
    # The Critic's verdict has a score; the automatic checks' has none, even
    # in a record written with the old placeholder 1 (the Mac test's
    # console.log said "critic verdict: ABORT, score 1").
    assert _progress_line({"type": "verdict", "data": {"verdict": "PASS", "critic_score": 8}}) \
        == "  critic verdict: PASS, score 8"
    for score in (None, 1):
        assert _progress_line({"type": "verdict", "data": {
            "verdict": "ABORT", "critic_score": score, "source": "pre_critic"}}) \
            == "  automatic pre-review check: ABORT"
    # Chatty events stay in pipeline.log / events.jsonl only.
    for etype in ("log", "agent.note", "llm.start", "llm.end", "lit.progress", "metric"):
        assert _progress_line({"type": etype, "plain": "x", "data": {}}) is None


def test_console_progress_echoes_sink_events_and_restores(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import argparse

    from src import events
    from src.main import _console_progress

    sink = events.EventSink(str(tmp_path))
    with _console_progress(argparse.Namespace(quiet=False)):
        sink.emit("stage.start", stage="WRITING", plain="Writing the paper")
        sink.emit("log", message="not echoed")
    sink.emit("stage.start", stage="VERIFYING", plain="after the run")

    err = capsys.readouterr().err
    assert "Writing the paper" in err
    assert "not echoed" not in err
    assert "after the run" not in err
    assert events._echo is None
    # The file record is unaffected by the echo.
    lines = (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3


def test_quiet_turns_the_console_echo_off(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import argparse

    from src import events
    from src.main import _console_progress

    sink = events.EventSink(str(tmp_path))
    with _console_progress(argparse.Namespace(quiet=True)):
        sink.emit("stage.start", stage="WRITING", plain="Writing the paper")
    assert "Writing the paper" not in capsys.readouterr().err


def test_a_failing_echo_never_breaks_the_event_record(tmp_path: Path) -> None:
    from src import events

    def boom(_record: dict) -> None:
        raise RuntimeError("console gone")

    previous = events.set_echo(boom)
    try:
        sink = events.EventSink(str(tmp_path))
        sink.emit("stage.start", stage="WRITING", plain="x")
    finally:
        events.set_echo(previous)
    assert (tmp_path / "events.jsonl").read_text(encoding="utf-8").count("\n") == 1
