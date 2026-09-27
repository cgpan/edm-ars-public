"""The live view of a running study.

Two modes:

* the full screen (rich ``Live``, about four redraws a second): the
  question, a progress bar with a finish-time range, one line per step
  with its time and result, what is happening now, cost so far and the
  last few plain-language notes;
* plain mode (``--plain``, pipes, screen readers): no redraw at all. It
  prints the screen once, then one line per new thing that happened,
  plus a short "still working" line when a step stays quiet for a while.

Closing the window never stops the study; Ctrl+C asks whether to leave it
running (the default) or stop it.

``watch()`` returns 0 when the study ended, 10 when the person left it
running, 11 when they stopped it, and 1 when there is no such folder.
These are for the caller only, not exit codes: ``edmars status`` (and
``new``, ``run`` and ``resume`` while they watch) turns them into the
result screen's exit codes, ``edmars.model.EXIT_*``: 0 ready or still
running (left running), 2 finished but not ready, 3 stopped (also when
stopped from the view), 1 no matching study.
"""
from __future__ import annotations

import signal
import textwrap
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from rich.cells import cell_len

from edmars.runstate import (
    CUT_OFF_NOTE,
    EXPERIMENTAL_LINE,
    RunState,
    StageState,
    StateReader,
    fmt_duration,
    progress,
    stage_title,
    step_position,
    stopped_early,
)
from edmars.runstate import cost_line as _cost_line

EXIT_ENDED = 0
EXIT_LEFT_RUNNING = 10
EXIT_STOPPED = 11
EXIT_NO_RUN = 1

GLYPHS = {"done": "✓", "running": "◐", "pending": "·", "failed": "✗", "skipped": "–"}
PLAIN_GLYPHS = {"done": "[ok]", "running": "[..]", "pending": "[  ]", "failed": "[x]", "skipped": "[--]"}
STYLES = {"done": "green", "running": "bold yellow", "pending": "dim", "failed": "bold red", "skipped": "dim"}

_ASCII = {
    "·": "-", "–": "-", "—": "-", "…": "...", "✓": "[ok]", "✗": "[x]", "◐": "[..]",
    "█": "#", "░": "-", "’": "'", "‘": "'", "“": '"', "”": '"', "²": "2", "→": "->",
    "≥": ">=", "≤": "<=", "×": "x", " ": " ",
}

#: Print a "still working" line in plain mode after this much silence.
PLAIN_HEARTBEAT_S = 300.0

#: An error in the "Recent" list is wrapped, up to this many lines.
RECENT_ERROR_LINES = 8

Line = tuple[str, str]  # (text, rich style)


def to_ascii(text: str) -> str:
    """Replace box-drawing and typographic glyphs; letters are kept."""
    for src, dst in _ASCII.items():
        if src in text:
            text = text.replace(src, dst)
    return text


def _truncate(text: str, width: int, plain: bool) -> str:
    if width <= 0:
        return ""
    if cell_len(text) <= width:
        return text
    ellipsis = "..." if plain else "…"
    keep = max(width - cell_len(ellipsis), 0)
    out = ""
    for ch in text:
        if cell_len(out + ch) > keep:
            break
        out += ch
    return out + ellipsis if width > len(ellipsis) else out[:width]


def _two_col(left: str, right: str, width: int, plain: bool) -> list[str]:
    """``left .... right`` on one line when it fits, else two lines."""
    if not right:
        return [_truncate(left, width, plain)]
    if cell_len(left) + 2 + cell_len(right) <= width:
        return [left + " " * (width - cell_len(left) - cell_len(right)) + right]
    return [_truncate(left, width, plain), _truncate("    " + right, width, plain)]


def _wrap(text: str, width: int, indent: str = "") -> list[str]:
    if not text:
        return []
    return textwrap.wrap(text, width=max(width, 20), subsequent_indent=indent,
                         break_long_words=True, break_on_hyphens=False) or [""]


def _local(dt: datetime | None) -> str:
    if dt is None:
        return "--:--"
    return dt.astimezone().strftime("%H:%M")


def _round5(dt: datetime, *, up: bool) -> datetime:
    """Round to five minutes so the finish-time range does not flicker."""
    base = dt.replace(second=0, microsecond=0)
    extra = base.minute % 5
    if up and (extra or dt.second or dt.microsecond):
        return base + timedelta(minutes=5 - extra)
    return base - timedelta(minutes=extra)


def _labels() -> dict[str, Any]:
    try:
        from edmars.endstates import messages

        return messages()
    except Exception:  # noqa: BLE001
        return {}


def _type_label(task_type: str) -> str:
    table = _labels().get("task_types") or {}
    return str(table.get(task_type, task_type or "Study")) if isinstance(table, dict) else task_type


def _provider_label(provider: str) -> str:
    table = _labels().get("providers") or {}
    return str(table.get(provider, provider)) if isinstance(table, dict) else provider


def cost_line(state: RunState) -> str:
    """The cost line (edmars.runstate.cost_line, shared with the result
    screen and summary.html)."""
    return _cost_line(state)


def render_screen(
    state: RunState,
    *,
    width: int = 80,
    plain: bool = False,
    now: datetime | None = None,
    cpu_percent: float | None = None,
) -> list[Line]:
    """The full live screen as (text, style) lines, each at most ``width``
    cells wide."""
    width = max(int(width), 30)
    ref = now or datetime.now(timezone.utc)
    glyphs = PLAIN_GLYPHS if plain else GLYPHS
    out: list[Line] = []

    def add(text: str, style: str = "") -> None:
        if plain:
            text = to_ascii(text)
        out.append((_truncate(text, width, plain), style))

    question = state.question or "(the study plan is being written)"
    add(f'EDM-ARS · "{question}"', "bold")
    if state.experimental:
        add(EXPERIMENTAL_LINE, "bold yellow")
    left = " · ".join(p for p in (_type_label(state.task_type), state.dataset, _provider_label(state.provider)) if p)
    right = f"Started {_local(state.started)} · now {_local(ref)}"
    for text in _two_col(left, right, width, plain):
        add(text, "dim")

    fraction, low, high = progress(state, ref)
    step, total = step_position(state)
    bar_w = 20 if width >= 80 else 10
    filled = int(round(fraction * bar_w))
    bar = ("#" * filled + "-" * (bar_w - filled)) if plain else ("█" * filled + "░" * (bar_w - filled))
    early = stopped_early(state)
    left = f"Overall {bar}  {'Stopped at step' if early else 'Step'} {step} of {total}"
    if state.finished:
        right = f"{'Stopped' if early else 'Finished'} {_local(state.updated)}"
    elif low is not None and high is not None:
        right = f"Usually done between {_local(_round5(low, up=False))} and {_local(_round5(high, up=True))}"
    else:
        right = ""
    for text in _two_col(left, right, width, plain):
        add(text)

    for number, st in enumerate(state.visible_stages(), start=1):
        title = stage_title(state, st)
        stage_left = f" {glyphs.get(st.status, '·')} {number} {title}"
        dur = stage_time(st, ref) if st.status != "pending" else ""
        detail = st.detail
        if st.status == "running" and not detail:
            detail = _running_hint(state)
        if st.status == "failed" and st.interrupted:
            detail = "interrupted" + (f" · {detail}" if detail else "")
        stage_right = "  ".join(p for p in (dur, detail) if p)
        for text in _two_col(stage_left, stage_right, width, plain):
            add(text, STYLES.get(st.status, ""))

    now_text = state.now_text
    if not state.finished and cpu_percent is not None and not state.waiting_ai and \
            not state.attempt and running_for(state, ref) > 120:
        now_text = f"{now_text} (computer busy: {cpu_percent:.0f}% CPU)"
    for i, text in enumerate(_wrap(f"Now: {now_text}", width, indent="     ")):
        add(text, "bold" if i == 0 else "")
    tail = "" if state.finished else " · Safe to close this window — the study keeps running"
    if state.finished and state.calls_cut_off:
        tail = f". {CUT_OFF_NOTE}"
    for text in _wrap(cost_line(state) + tail, width):
        add(text, "dim")
    recent = state.recent[-3:]
    if recent:
        add("Recent:", "dim")
        errors = {" ".join(e.split()) for e in state.errors}
        notices = set(state.notices)
        for line in recent:
            if line not in errors and line not in notices:
                add(f"  {line}", "dim")
                continue
            # An error is the one line a person needs whole (why the study
            # stopped), and a notice says why a step is being redone and
            # which revision this is: wrap them instead of cutting them at
            # the screen's edge.
            wrapped = _wrap(to_ascii(line) if plain else line, width - 2, indent="  ")
            if len(wrapped) > RECENT_ERROR_LINES:
                wrapped = wrapped[:RECENT_ERROR_LINES]
                wrapped[-1] = _truncate(wrapped[-1] + " ...", width - 2, plain)
            for text in wrapped:
                add(f"  {text}", "red" if line in errors else "yellow")
    if not state.finished:
        add("Ctrl+C: leave or stop", "dim")
    return out


def stage_time(st: StageState, ref: datetime | None = None, *, bracketed: bool = False) -> str:
    """A step's time in this attempt; after a resume, also the total with
    the interrupted attempt (the Mac view showed only the sum, 7m20s, for
    an analysis that took 6m12s after the resume). ``bracketed`` is for
    text already inside brackets: "6m12s; 7m20s incl. ..." instead of
    "6m12s (7m20s incl. ...)"."""
    dur = fmt_duration(st.duration_s(ref))
    if st.earlier_s and dur:
        total = f"{fmt_duration(st.total_s(ref))} incl. the interrupted attempt"
        dur += f"; {total}" if bracketed else f" ({total})"
    return dur


def _running_hint(state: RunState) -> str:
    if state.waiting_ai:
        return "waiting for the AI's reply…"
    if state.code_running:
        return "running code…"
    return ""


def running_for(state: RunState, now: datetime | None = None) -> float:
    for st in state.stages:
        if st.status == "running":
            return st.duration_s(now) or 0.0
    return 0.0


def screen_text(state: RunState, *, width: int = 80, plain: bool = True,
                now: datetime | None = None) -> str:
    return "\n".join(text for text, _ in render_screen(state, width=width, plain=plain, now=now))


# ---------------------------------------------------------------------------
# Plain mode: one line per new event
# ---------------------------------------------------------------------------


class PlainPrinter:
    """Turns successive states into new lines only; never redraws."""

    def __init__(self, width: int = 80, heartbeat_s: float = PLAIN_HEARTBEAT_S) -> None:
        self.width = max(width, 40)
        self.heartbeat_s = heartbeat_s
        self._status: dict[str, str] = {}
        self._printed_recent: list[str] = []
        self._last_output: float | None = None
        self._finished_said = False
        self._started = False

    def _emit(self, text: str) -> list[str]:
        return _wrap(to_ascii(text), self.width, indent="    ")

    def lines(self, state: RunState, *, now: datetime | None = None,
              clock: float | None = None) -> list[str]:
        ref = now or datetime.now(timezone.utc)
        tick = time.monotonic() if clock is None else clock
        out: list[str] = []
        if not self._started:
            self._started = True
            out.extend(screen_text(state, width=self.width, plain=True, now=ref).splitlines())
            for st in state.stages:
                self._status[st.key] = st.status
            self._printed_recent = list(state.recent)
            self._last_output = tick
            if state.finished:
                self._finished_said = True
            return out

        visible = state.visible_stages()
        total = len(visible)
        for number, st in enumerate(visible, start=1):
            before = self._status.get(st.key)
            if st.status == before:
                continue
            self._status[st.key] = st.status
            title = stage_title(state, st)
            dur = stage_time(st, ref, bracketed=True)
            if st.status == "running":
                out += self._emit(f"[..] Step {number} of {total}: {title}")
            elif st.status == "done":
                text = f"[ok] Step {number} of {total} done"
                text += f" ({dur})" if dur else ""
                text += f": {title}"
                if st.detail:
                    text += f" - {st.detail}"
                out += self._emit(text)
            elif st.status == "failed" and st.interrupted:
                text = f"[x] Step {number} of {total} interrupted"
                text += f" after {dur}" if dur else ""
                out += self._emit(f"{text}: {title}")
            elif st.status == "failed":
                out += self._emit(f"[x] Step {number} of {total} did not finish: {title}")
            elif st.status == "skipped" and before is not None:
                out += self._emit(f"[--] Step {number} of {total} not needed: {title}")
        fresh = [line for line in state.recent if line not in self._printed_recent]
        for line in fresh:
            out += self._emit(f"     {line}")
        self._printed_recent = (self._printed_recent + fresh)[-200:]
        if state.finished and not self._finished_said:
            self._finished_said = True
            ended = "Stopped" if stopped_early(state) else "Finished"
            out += self._emit(f"{ended}. {state.now_text} {cost_line(state)}.")
            if state.calls_cut_off:
                out += self._emit(CUT_OFF_NOTE)
        if out:
            self._last_output = tick
        elif not state.finished and self._last_output is not None and \
                tick - self._last_output >= self.heartbeat_s:
            step, total = step_position(state)
            out += self._emit(
                f"... still working on step {step} of {total} "
                f"({fmt_duration(running_for(state, ref))} so far). {state.now_text}"
            )
            self._last_output = tick
        return out


# ---------------------------------------------------------------------------
# watch()
# ---------------------------------------------------------------------------


class _ProcTree:
    """CPU use of the pipeline process and its children (best effort)."""

    def __init__(self) -> None:
        self._pid: int | None = None
        self._procs: dict[int, Any] = {}

    def percent(self, pid: int | None) -> float | None:
        if not pid:
            return None
        try:
            import psutil

            if pid != self._pid:
                self._pid = pid
                self._procs = {}
            root = psutil.Process(pid)
            members = [root] + root.children(recursive=True)
            total = 0.0
            fresh: dict[int, Any] = {}
            for p in members:
                known = self._procs.get(p.pid)
                if known is None:
                    p.cpu_percent(None)  # first call primes the counter
                    fresh[p.pid] = p
                    continue
                total += known.cpu_percent(None)
                fresh[p.pid] = known
            self._procs = fresh
            count = psutil.cpu_count() or 1
            return total / count
        except Exception:  # noqa: BLE001
            return None


def _ask_leave_or_stop(run_dir: Path, say: Callable[[str], None]) -> int:
    from edmars import ui

    try:
        choice = ui.select(
            "The study is still running. What would you like to do?",
            [("leave", "Leave it running (default)"), ("stop", "Stop the study")],
            default="leave",
        )
    except (KeyboardInterrupt, EOFError):
        choice = "leave"
    except Exception:  # noqa: BLE001 -- NonInteractiveError or a broken terminal
        choice = "leave"
    from edmars.endstates import quote_path

    if choice == "stop":
        from edmars import runner

        say("Stopping the study (this can take up to 30 seconds)...")
        try:
            runner.stop(run_dir)
        except Exception as exc:  # noqa: BLE001
            say(f"Could not stop the study: {exc}")
            return EXIT_LEFT_RUNNING
        say("Stopped. Your finished steps are saved. Continue later with: "
            f"edmars resume {quote_path(run_dir)}")
        return EXIT_STOPPED
    say("The study keeps running in the background. Check on it any time with: "
        f"edmars status {quote_path(run_dir)}")
    return EXIT_LEFT_RUNNING


def _say_plain(text: str) -> None:
    print(to_ascii(text), flush=True)


def watch(run_dir: Path | str, *, plain: bool = False, poll_s: float = 1.0) -> int:
    """Follow a study until it ends or the person leaves (see module doc)."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        _say_plain(f"There is no study folder at {run_dir}.")
        return EXIT_NO_RUN
    try:
        from edmars import ui

        plain = plain or bool(ui.is_plain())
    except Exception:  # noqa: BLE001
        plain = True
    reader = StateReader(run_dir)
    if plain:
        return _watch_plain(reader, run_dir, poll_s=max(poll_s, 1.0))
    return _watch_live(reader, run_dir, poll_s=poll_s)


def _terminal_width(default: int = 80) -> int:
    try:
        import shutil

        return shutil.get_terminal_size((default, 24)).columns
    except Exception:  # noqa: BLE001
        return default


def _watch_plain(reader: StateReader, run_dir: Path, *, poll_s: float) -> int:
    printer = PlainPrinter(width=min(_terminal_width(), 100))
    try:
        while True:
            state = reader.refresh()
            for line in printer.lines(state):
                print(line, flush=True)
            if state.finished:
                return EXIT_ENDED
            time.sleep(max(poll_s, 2.0))
    except KeyboardInterrupt:
        return _ask_leave_or_stop(run_dir, _say_plain)


#: Signals that end the full-screen view the way Ctrl+C ends a command:
#: through Python, so rich's Live shows the cursor again on the way out.
_END_SIGNALS = ("SIGTERM", "SIGHUP")


@contextmanager
def _terminal_restored_on_signals(console: Any) -> Iterator[None]:
    """While the full-screen view runs, turn SIGTERM and SIGHUP into
    ``SystemExit(128 + signal)`` and show the cursor again at the end.

    rich hides the cursor while Live draws and only Live.stop shows it
    again. Python's default for SIGTERM ends the process at once, so
    `kill`-ing `edmars status` left the terminal without a cursor (it
    needed `tput cnorm`). Ctrl+C already arrives as KeyboardInterrupt.
    The previous handlers are put back afterwards; in a thread other than
    the main one, where handlers cannot be set, nothing changes.
    """
    previous: dict[int, Any] = {}

    def _end(signum: int, _frame: Any) -> None:
        raise SystemExit(128 + signum)

    for name in _END_SIGNALS:
        sig = getattr(signal, name, None)
        if sig is None:  # no SIGHUP on Windows
            continue
        try:
            previous[sig] = signal.signal(sig, _end)
        except (ValueError, OSError):
            continue
    try:
        yield
    finally:
        for sig, handler in previous.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError, TypeError):
                pass
        try:
            console.show_cursor(True)
        except Exception:  # noqa: BLE001 - a closed terminal has no cursor to show
            pass


def _watch_live(reader: StateReader, run_dir: Path, *, poll_s: float) -> int:
    from rich.console import Group
    from rich.live import Live
    from rich.text import Text

    from edmars import ui

    # rich's Live keeps a console of its own: hand it the real Console, not
    # ui.console (a forwarding proxy for plain print calls).
    get_console = getattr(ui, "get_console", None)
    console = get_console() if callable(get_console) else ui.console
    tree = _ProcTree()

    def renderable(state: RunState) -> Group:
        width = max(min(console.width, 120), 30)
        cpu = tree.percent(state.pid) if state.pid else None
        lines = render_screen(state, width=width, plain=False, cpu_percent=cpu)
        return Group(*[Text(text, style=style, no_wrap=True, overflow="ellipsis") for text, style in lines])

    state = reader.refresh()
    last_read = time.monotonic()
    try:
        with _terminal_restored_on_signals(console), \
                Live(renderable(state), console=console, refresh_per_second=4,
                     transient=False, auto_refresh=False) as live:
            while True:
                if time.monotonic() - last_read >= poll_s:
                    state = reader.refresh()
                    last_read = time.monotonic()
                live.update(renderable(state), refresh=True)
                if state.finished:
                    break
                time.sleep(0.25)
    except KeyboardInterrupt:
        return _ask_leave_or_stop(run_dir, lambda t: console.print(t, markup=False, highlight=False))
    console.print(state.now_text, markup=False, highlight=False)
    return EXIT_ENDED


__all__ = [
    "EXIT_ENDED",
    "EXIT_LEFT_RUNNING",
    "EXIT_NO_RUN",
    "EXIT_STOPPED",
    "PlainPrinter",
    "cost_line",
    "render_screen",
    "screen_text",
    "to_ascii",
    "watch",
]
