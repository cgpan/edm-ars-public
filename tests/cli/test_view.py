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
    assert any(line.startswith("Stopped.") for line in end)
    assert not any(line.startswith("Finished.") for line in end)
    for line in first + new + beat + end:
        assert len(line) <= 80
        assert "✓" not in line and "·" not in line


@pytest.mark.parametrize("final", ["ABORTED", "INTERRUPTED"])
def test_a_study_that_stopped_early_is_not_shown_as_done(final: str) -> None:
    state = fold([
        event(1, "run.start", 0, task_type="prediction", dataset="hsls09_public", provider="deepseek"),
        event(2, "stage.start", 0, stage="FORMULATING"),
        event(3, "stage.end", 1, stage="FORMULATING", outcome="ok"),
        event(4, "stage.start", 1, stage="ENGINEERING"),
        event(5, "run.end", 3, state=final),
    ])
    text = view.screen_text(state, width=80, plain=True, now=NOW)
    overall = next(line for line in text.splitlines() if line.startswith("Overall"))
    assert "Stopped at step 2 of 7" in overall and "Stopped " in overall
    assert "Finished" not in text and "Step 7 of 7" not in text
    assert "#" * 20 not in overall  # the bar shows the one finished step, not all seven


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
    # The live view hands rich's Live the real Console (ui.get_console()).
    monkeypatch.setattr(ui, "get_console", lambda stderr=False: console)
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


def test_an_experimental_plan_is_labelled_on_the_live_view(run_home: Path) -> None:
    run = make_run(run_home, log=FULL_LOG, study={"experimental": True})
    text = view.screen_text(load_state(run), width=80, plain=True, now=NOW)
    assert "[EXPERIMENTAL] not a tested example study" in text
    tested = make_run(run_home, name="2026-09-25_1400_tested_ef01", log=FULL_LOG)
    assert "EXPERIMENTAL" not in view.screen_text(load_state(tested), width=80,
                                                  plain=True, now=NOW)



@pytest.mark.parametrize("plain", [True, False])
def test_an_error_in_the_recent_list_is_wrapped_not_cut(plain: bool) -> None:
    # The Mac study's live view cut pcc_07's sentence (why the study
    # stopped) at the screen's edge: "... which commits the pa...".
    message = ("pcc_07: The research question says 'above and beyond', which commits the paper to an "
               "incremental-validity / nested-model comparison, but no such analysis appears in results.json.")
    state = fold([
        event(1, "run.start", 0, task_type="prediction", dataset="hsls09_public", provider="deepseek"),
        event(2, "stage.start", 0, stage="CRITIQUING"),
        event(3, "warning", 1, plain="A warning that is long enough to be cut at the edge of an eighty column screen."),
        event(4, "error", 1, stage="CRITIQUING", code="PRE_CRITIC_ABORT", message=message),
        event(5, "run.end", 1, state="ABORTED"),
    ])
    lines = [text for text, _ in view.render_screen(state, width=80, plain=plain, now=NOW)]
    assert all(cell_len(text) <= 80 for text in lines)
    start = next(i for i, text in enumerate(lines) if text.startswith("  pcc_07:"))
    joined = " ".join(text.strip() for text in lines[start:start + 3])
    assert message in joined
    warning = next(text for text in lines if "A warning" in text)
    assert warning.endswith("..." if plain else "\u2026")  # other lines still fit on one line



def test_the_view_docstring_says_its_codes_are_not_exit_codes() -> None:
    doc = " ".join((view.__doc__ or "").split())
    assert "not exit codes" in doc and "edmars.model.EXIT_" in doc



def _ev(seq: int, etype: str, clock: str, *, stage: str | None = None, **data: object) -> dict:
    """An event at 13:<clock> (mm:ss), for timings finer than a minute."""
    return {"v": 1, "seq": seq, "ts": f"2026-09-25T13:{clock}.000Z", "run_id": "run", "type": etype,
            "stage": stage, "cycle": None, "agent": None, "plain": None, "data": data}


_BEFORE_STOP = [
    _ev(1, "run.start", "41:24", task_type="prediction", dataset="hsls09_public", provider="deepseek"),
    _ev(2, "stage.start", "41:24", stage="FORMULATING"),
    _ev(3, "stage.end", "42:47", stage="FORMULATING", outcome="ok"),
    _ev(4, "stage.start", "42:47", stage="ENGINEERING"),
    _ev(5, "stage.end", "47:29", stage="ENGINEERING", outcome="ok"),
    _ev(6, "stage.start", "47:29", stage="ANALYZING"),
]
_THE_STOP = [
    _ev(7, "stage.end", "48:37", stage="ANALYZING", outcome="interrupted"),
    _ev(8, "error", "48:37", stage="ANALYZING", code="INTERRUPTED",
        message="Interrupted (Ctrl-C or a termination signal)"),
    _ev(9, "run.end", "48:37", state="INTERRUPTED"),
]
_AFTER_RESUME = [
    _ev(10, "run.start", "49:09", resumed=True),
    _ev(11, "stage.start", "49:09", stage="ANALYZING"),
    _ev(12, "stage.end", "55:21", stage="ANALYZING", outcome="ok"),
]


def test_an_interrupted_step_is_shown_as_interrupted_not_done() -> None:
    # After `edmars stop`, the Mac's `status --plain` feed printed
    # "[ok] Step 3 of 8 done (1m08s): Running the analysis" for an
    # analysis that had not finished.
    printer = view.PlainPrinter(width=100)
    printer.lines(fold(_BEFORE_STOP), now=NOW, clock=0.0)
    stopped = fold(_BEFORE_STOP + _THE_STOP)
    out = printer.lines(stopped, now=NOW, clock=1.0)
    assert "[x] Step 3 of 7 interrupted after 1m08s: Running the analysis" in out
    assert not any("done" in line and "Running the analysis" in line for line in out)
    analysis = stopped.stage("ANALYZING")
    assert analysis.status == "failed" and analysis.interrupted
    screen = view.screen_text(stopped, width=100, now=NOW)
    assert "Stopped at step 3 of 7" in screen
    row = next(line for line in screen.splitlines() if "Running the analysis" in line)
    assert row.startswith(" [x] 3 ") and "interrupted" in row and "1m08s" in row


def test_after_a_resume_a_step_shows_this_attempts_time_and_the_total() -> None:
    # The Mac view showed 7m20s (1m08s + 6m12s) for an analysis that took
    # 6m12s after the resume.
    state = fold(_BEFORE_STOP + _THE_STOP + _AFTER_RESUME)
    analysis = state.stage("ANALYZING")
    assert analysis.status == "done" and not analysis.interrupted
    assert analysis.duration_s() == 372.0 and analysis.total_s() == 440.0
    assert analysis.rounds == 1
    assert view.stage_time(analysis) == "6m12s (7m20s incl. the interrupted attempt)"
    row = next(line for line in view.screen_text(state, width=100, now=NOW).splitlines()
               if "Running the analysis" in line)
    assert row.startswith(" [ok] 3 ") and "6m12s (7m20s incl. the interrupted attempt)" in row
    printer = view.PlainPrinter(width=120)
    printer.lines(fold(_BEFORE_STOP + _THE_STOP + _AFTER_RESUME[:2]), now=NOW, clock=0.0)
    out = printer.lines(state, now=NOW, clock=1.0)
    assert ("[ok] Step 3 of 7 done (6m12s; 7m20s incl. the interrupted attempt): "
            "Running the analysis") in out


def test_a_step_cut_off_without_a_stage_end_counts_as_an_earlier_attempt() -> None:
    # A killed process writes no stage.end: the round ends at the last
    # event it wrote, and the resumed round is timed on its own.
    killed = _BEFORE_STOP + [_ev(7, "llm.start", "48:29", stage="ANALYZING")]
    state = fold(killed + [_ev(8, "run.start", "50:00", resumed=True),
                           _ev(9, "stage.start", "50:00", stage="ANALYZING"),
                           _ev(10, "stage.end", "53:00", stage="ANALYZING", outcome="ok")])
    analysis = state.stage("ANALYZING")
    assert analysis.duration_s() == 180.0 and analysis.earlier_s == 60.0



def _stopped_mid_call() -> list[dict]:
    return [
        event(1, "run.start", 0, task_type="prediction", dataset="hsls09_public", provider="deepseek"),
        event(2, "stage.start", 0, stage="ANALYZING"),
        event(3, "llm.end", 1, ok=True, cost_usd=0.02),
        event(4, "llm.end", 2, ok=False, error_class="KeyboardInterrupt", cost_usd=None),
        event(5, "stage.end", 2, stage="ANALYZING", outcome="interrupted"),
        event(6, "run.end", 2, state="INTERRUPTED"),
    ]


def test_a_call_cut_off_by_the_stop_is_counted_and_explained() -> None:
    # The Mac view said "at least US$0.039 (9 AI calls)" at the stop while
    # pipeline.log counted 8 calls, and nothing said what the ninth was.
    state = fold(_stopped_mid_call())
    assert (state.llm_calls, state.calls_cut_off, state.calls_failed) == (2, 1, 0)
    assert view.cost_line(state) == "Cost: at least US$0.020 (2 AI calls, 1 cut off when the study was stopped)"
    screen = " ".join(view.screen_text(state, width=80, now=NOW).split())
    assert "may still be billed by the AI service" in screen


def test_a_call_that_failed_makes_the_cost_a_lower_bound_without_calling_it_cut_off() -> None:
    state = fold([event(1, "llm.end", 0, ok=True, cost_usd=0.01),
                  event(2, "llm.end", 1, ok=False, error_class="APIStatusError", cost_usd=None)])
    assert view.cost_line(state) == "Cost so far: at least US$0.010 (2 AI calls)"
    assert not state.cost_unpriced  # the model has a price; the call had no answer
    only_cut_off = fold([event(1, "llm.end", 0, ok=False, error_class="KeyboardInterrupt"),
                         event(2, "run.end", 0, state="INTERRUPTED")])
    assert view.cost_line(only_cut_off) == \
        "Cost: at least US$0.000 (1 AI call, 1 cut off when the study was stopped)"


def test_a_call_still_waiting_when_the_process_died_is_cut_off() -> None:
    state = fold([event(1, "llm.end", 0, ok=True, cost_usd=0.01),
                  event(2, "llm.start", 1),
                  event(3, "run.start", 5, resumed=True)])
    assert (state.llm_calls, state.calls_cut_off) == (2, 1)
    assert not state.waiting_ai



@pytest.mark.parametrize("name", ["SIGTERM", "SIGHUP"])
def test_killing_the_full_screen_view_shows_the_cursor_again(
    run_home: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    # On the Mac, `kill` (SIGTERM) on `edmars status` in a terminal left the
    # cursor hidden: Python's default handler ended the process before rich
    # could show it again. The signal now ends the view through Python.
    import signal

    from edmars import ui

    sig = getattr(signal, name, None)
    if sig is None:
        pytest.skip(f"this system has no {name}")
    run = make_run(run_home, pid=alive_pid(), pdf=False, log=log_lines((0, "Starting FORMULATING stage")))
    console = Console(file=io.StringIO(), width=100, force_terminal=True)
    monkeypatch.setattr(ui, "get_console", lambda stderr=False: console)
    monkeypatch.setattr(ui, "is_plain", lambda: False)

    class _NotHandled(Exception):
        pass

    def _unhandled(signum: int, frame: object) -> None:
        raise _NotHandled(f"the view left {name} to the previous handler")

    # A stand-in for the default handler, which would end pytest itself.
    previous = signal.signal(sig, _unhandled)
    try:
        monkeypatch.setattr(view.time, "sleep", lambda seconds: signal.raise_signal(sig))
        with pytest.raises(SystemExit) as ended:
            view.watch(run)
        restored = signal.getsignal(sig)
    finally:
        signal.signal(sig, previous)
    assert ended.value.code == 128 + sig
    out = console.file.getvalue()  # type: ignore[attr-defined]
    hidden, shown = out.rfind("\x1b[?25l"), out.rfind("\x1b[?25h")
    assert hidden >= 0 and shown > hidden, "the cursor was left hidden"
    assert restored is _unhandled  # the earlier handler is put back
