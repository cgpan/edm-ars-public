"""The live view: screen rendering at several widths, plain mode, watch()."""
from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path

import pytest
from rich.cells import cell_len
from rich.console import Console

from tests.cli._run_support import (  # also installs stand-ins
    FULL_LOG,
    PREDICTION_RESULTS,
    alive_pid,
    event,
    log_lines,
    make_run,
)

from edmars import view
from edmars.runstate import RunState, fold, load_state

NOW = datetime(2026, 9, 25, 13, 21, tzinfo=timezone.utc)

LONG_Q = ("Do ninth-grade mathematics self-efficacy and school belonging predict whether "
          "students from every socioeconomic background enrol in college by 2016, over and above prior achievement?")


def _states() -> list[RunState]:
    running = fold([
        event(1, "run.start", 0, task_type="causal_soo", dataset="hsls09_public", provider="deepseek"),
        event(2, "stage.start", 0, stage="FORMULATING", plain="Framing the question"),
        event(3, "stage.end", 1, stage="FORMULATING", outcome="ok"),
        event(4, "stage.start", 1, stage="ENGINEERING"),
        event(5, "stage.end", 4, stage="ENGINEERING", outcome="ok"),
        event(6, "stage.start", 4, stage="ANALYZING"),
        event(7, "attempt.start", 5, stage="ANALYZING", attempt=2, max_attempts=4, timeout_s=1200),
        event(8, "warning", 6, plain="The first analysis attempt failed; trying again with the error message"),
        event(9, "llm.end", 6, cost_usd=0.041),
    ])
    running.question = LONG_Q
    running.metrics.update({"analytic_n": 17335, "n_predictors": 42, "papers_found": 38})
    running.stages[0].detail = "38 papers found"
    running.stages[1].detail = "17,335 students · 42 predictors"
    running.now_text = "Running the analysis code (attempt 2 of 4) — 6m12s so far; this step can take up to 20 minutes"
    finished = fold([event(1, "stage.start", 0, stage="VERIFYING"),
                     event(2, "run.end", 3, state="COMPLETED", cost_usd=0.12)])
    finished.question = "短い質問 with CJK characters that are two cells wide each 質問質問質問質問"
    finished.lsar_enabled = True
    empty = RunState()
    return [running, finished, empty]


@pytest.mark.parametrize("width", [60, 80, 120])
@pytest.mark.parametrize("plain", [True, False])
def test_render_screen_fits_width(width: int, plain: bool) -> None:
    for state in _states():
        lines = view.render_screen(state, width=width, plain=plain, now=NOW, cpu_percent=37.0)
        assert lines
        for text, _style in lines:
            assert cell_len(text) <= width, (width, text)
        if plain:
            joined = "\n".join(t for t, _ in lines)
            for glyph in ("█", "░", "✓", "◐", "✗", "·", "—", "…"):
                assert glyph not in joined, glyph


def test_running_screen_content() -> None:
    running = _states()[0]
    text = view.screen_text(running, width=120, plain=True, now=NOW)
    assert 'EDM-ARS - "Do ninth-grade' in text
    assert "Step 3 of 7" in text
    assert "Usually done between" in text
    assert "[ok] 1 Framing the question & finding related studies" in text
    assert "[..] 3 Running the analysis" in text
    assert "attempt 2 of 4" in text
    assert "Cost so far: US$0.041 (1 AI call)" in text
    assert "Safe to close this window" in text
    assert "Ctrl+C: leave or stop" in text
    assert "trying again with the error message" in text
    assert "  Framing the question" not in text.splitlines()  # stage starts are rows, not news


def test_rich_render_does_not_raise() -> None:
    from rich.text import Text

    for width in (60, 80, 120):
        console = Console(file=io.StringIO(), width=width, force_terminal=True, color_system="truecolor")
        for state in _states():
            for text, style in view.render_screen(state, width=width, plain=False, now=NOW):
                console.print(Text(text, style=style, no_wrap=True, overflow="ellipsis"))
        assert console.file.getvalue()  # type: ignore[attr-defined]


def test_finish_range_is_rounded_to_five_minutes() -> None:
    t = datetime(2026, 9, 25, 14, 37, 20, tzinfo=timezone.utc)
    assert view._round5(t, up=False).minute == 35
    assert view._round5(t, up=True).minute == 40
    exact = datetime(2026, 9, 25, 14, 40, tzinfo=timezone.utc)
    assert view._round5(exact, up=True) == exact


def test_cost_line_variants() -> None:
    s = RunState()
    assert view.cost_line(s) == "Cost so far: US$0.00 (no AI calls yet)"
    s.llm_calls = 3
    assert "not priced" in view.cost_line(s)
    s.cost_usd, s.cost_unpriced = 0.01, True
    assert "at least US$0.010" in view.cost_line(s)
    s.finished = True
    assert view.cost_line(s).startswith("Cost:")
    assert view.cost_line(RunState(finished=True)) == "Cost: not recorded for this study"


def test_plain_printer_prints_only_new_things() -> None:
    printer = view.PlainPrinter(width=80, heartbeat_s=60)
    s1 = fold([event(1, "stage.start", 0, stage="FORMULATING")])
    first = printer.lines(s1, now=NOW, clock=0)
    assert any("EDM-ARS" in line for line in first)  # attaches with one full screen
    assert printer.lines(s1, now=NOW, clock=10) == []  # nothing new, no heartbeat yet
    s2 = fold([event(2, "stage.end", 1, stage="FORMULATING", outcome="ok"),
               event(3, "stage.start", 1, stage="ENGINEERING"),
               event(4, "warning", 2, plain="Semantic Scholar is busy; using arXiv only")], s1)
    new = printer.lines(s2, now=NOW, clock=20)
    assert any(line.startswith("[ok] Step 1 of 7 done") for line in new)
    assert any(line.startswith("[..] Step 2 of 7: Preparing the data") for line in new)
    assert any("Semantic Scholar is busy" in line for line in new)
    assert printer.lines(s2, now=NOW, clock=30) == []
    beat = printer.lines(s2, now=NOW, clock=200)
    assert beat and beat[0].startswith("... still working on step 2 of 7")
    s3 = fold([event(5, "run.end", 5, state="ABORTED")], s2)
    s3.now_text = "The study stopped before finishing."
    end = printer.lines(s3, now=NOW, clock=210)
    assert any(line.startswith("[x] Step 2 of 7 did not finish") for line in end)
    assert any(line.startswith("Finished.") for line in end)
    for line in first + new + beat + end:
        assert len(line) <= 80
        assert "✓" not in line and "·" not in line


def test_watch_returns_zero_for_a_finished_run(run_home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run = make_run(run_home, results=PREDICTION_RESULTS)
    assert view.watch(run, plain=True) == view.EXIT_ENDED
    out = capsys.readouterr().out
    assert "EDM-ARS" in out and "Step 7 of 7" in out


def test_watch_missing_folder(tmp_path: Path) -> None:
    assert view.watch(tmp_path / "nope", plain=True) == view.EXIT_NO_RUN


def _interrupting_sleep(*_a: object) -> None:
    raise KeyboardInterrupt


@pytest.mark.parametrize("choice, expected", [("leave", view.EXIT_LEFT_RUNNING), ("stop", view.EXIT_STOPPED)])
def test_ctrl_c_asks_leave_or_stop(run_home: Path, monkeypatch: pytest.MonkeyPatch,
                                   capsys: pytest.CaptureFixture[str], choice: str, expected: int) -> None:
    from edmars import runner, ui

    run = make_run(run_home, pid=alive_pid(), pdf=False,
                   log=log_lines((0, "Starting FORMULATING stage")))
    stopped: list[Path] = []
    monkeypatch.setattr(view.time, "sleep", _interrupting_sleep)
    monkeypatch.setattr(ui, "select", lambda message, choices, default=None: choice)
    monkeypatch.setattr(runner, "stop", lambda run_dir: stopped.append(Path(run_dir)))
    assert view.watch(run, plain=True) == expected
    assert stopped == ([run] if choice == "stop" else [])
    out = capsys.readouterr().out
    assert ("edmars resume" in out) if choice == "stop" else ("edmars status" in out)


def test_ctrl_c_without_a_terminal_leaves_it_running(run_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from edmars import ui

    run = make_run(run_home, pid=alive_pid(), pdf=False, log=log_lines((0, "Starting FORMULATING stage")))
    monkeypatch.setattr(view.time, "sleep", _interrupting_sleep)

    def refuse(*_a: object, **_k: object) -> str:
        raise ui.NonInteractiveError("no terminal")

    monkeypatch.setattr(ui, "select", refuse)
    assert view.watch(run, plain=True) == view.EXIT_LEFT_RUNNING


def test_live_mode_finishes(run_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from edmars import ui

    run = make_run(run_home, results=PREDICTION_RESULTS)
    console = Console(file=io.StringIO(), width=100, force_terminal=True)
    monkeypatch.setattr(ui, "console", console)
    monkeypatch.setattr(ui, "is_plain", lambda: False)
    assert view.watch(run) == view.EXIT_ENDED
    assert "Framing the question" in console.file.getvalue()  # type: ignore[attr-defined]


def test_full_log_state_renders(run_home: Path) -> None:
    run = make_run(run_home, log=FULL_LOG, results=PREDICTION_RESULTS)
    state = load_state(run)
    text = view.screen_text(state, width=80, plain=True, now=NOW)
    assert "Best: XGBoost, AUC 0.78 [0.76-0.80]" in text
    assert "Finished" in text
    assert "Ctrl+C" not in text
