"""Terminal output and prompts for edmars.

Everything the application prints or asks goes through this module, so
the accessibility and robustness rules live in one place:

* **Plain mode** prints ASCII-safe text with no colour, no boxes and no
  live redraw. It is switched on by ``--plain`` and automatically when
  stdout is not a terminal, ``TERM=dumb``, ``NO_COLOR`` or
  ``EDMARS_PLAIN`` is set, or a Windows screen reader is running
  (:func:`plain_reason` says which).
* **Status is never colour alone.** Every message carries a glyph:
  ``✓ ! ✗ i`` normally, ``[ok] [!] [x] [i]`` in plain mode or when the
  terminal cannot encode the symbols.
* **Prompts** use questionary (arrow-key menus, masked key entry) in a
  real terminal and fall back to numbered ``input()`` prompts otherwise,
  including Git Bash's mintty, where questionary cannot run.
* **Non-interactive** runs (``--yes``, or no terminal to ask in) never
  hang on a prompt: a prompt with a default returns it, and one without
  raises :class:`NonInteractiveError` with a message saying what to pass
  instead.

Messages: :func:`ok` and :func:`info` go to stdout, :func:`warn` and
:func:`fail` to stderr, so ``--json`` output on stdout stays parseable.
:func:`set_machine_output` sends ``ok``/``info`` to stderr too while a
command writes JSON.
"""

from __future__ import annotations

import contextlib
import getpass
import os
import re
import sys
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from edmars.model import Check


class NonInteractiveError(RuntimeError):
    """A question needed an answer and there was no one to ask."""

    def __init__(self, question: str, hint: str | None = None) -> None:
        self.question = question.strip().rstrip(":?").strip()
        if _input_closed:
            why = "it reached the end of the input, so no one is there to answer"
        elif _non_interactive:
            why = "--yes was given"
        else:
            why = "there is no terminal to ask in"
        advice = hint or (
            "Run the command in a terminal window, or give the answer as an "
            "option (see --help)."
        )
        super().__init__(
            f'EDM-ARS needs an answer to "{self.question}" but cannot ask ({why}). {advice}'
        )


# --- Mode state -----------------------------------------------------------

_plain_forced = False
_non_interactive = False
#: Set once a prompt hit end of input: nobody is there to answer, so every
#: later prompt stops too, and is_interactive() says so.
_input_closed = False
_machine_output = False
_screen_reader: bool | None = None
_consoles: dict[tuple[bool, bool], Console] = {}

_TRUTHY_OFF = {"", "0", "false", "no", "off"}


def set_plain(flag: bool) -> None:
    """Force plain mode on (``--plain``). False returns to automatic."""
    global _plain_forced
    _plain_forced = bool(flag)


def set_non_interactive(flag: bool) -> None:
    """Never prompt (``--yes``): use defaults or raise NonInteractiveError."""
    global _non_interactive
    _non_interactive = bool(flag)


def is_non_interactive() -> bool:
    """True when ``--yes`` was given."""
    return _non_interactive


def set_machine_output(flag: bool) -> None:
    """Route every human message to stderr (a command is printing JSON)."""
    global _machine_output
    _machine_output = bool(flag)


def reset() -> None:
    """Return every mode to its default (for tests and repeated invocations)."""
    global _plain_forced, _non_interactive, _machine_output, _screen_reader, _input_closed
    _plain_forced = False
    _non_interactive = False
    _input_closed = False
    _machine_output = False
    _screen_reader = None
    _consoles.clear()


def _isatty(stream: Any) -> bool:
    try:
        return bool(stream is not None and stream.isatty())
    except (AttributeError, ValueError, OSError):
        return False


def _detect_screen_reader() -> bool:
    """Ask Windows whether a screen reader is running (SPI_GETSCREENREADER).

    NVDA, JAWS and Narrator set this flag. Best effort: any failure means
    "no", and other platforms are not probed.
    """
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        flag = ctypes.c_int(0)
        spi_getscreenreader = 0x0046
        done = ctypes.windll.user32.SystemParametersInfoW(  # type: ignore[attr-defined,unused-ignore]
            spi_getscreenreader, 0, ctypes.byref(flag), 0
        )
        return bool(done) and bool(flag.value)
    except Exception:
        return False


def screen_reader_active() -> bool:
    """True when a Windows screen reader is running (checked once)."""
    global _screen_reader
    if _screen_reader is None:
        _screen_reader = _detect_screen_reader()
    return _screen_reader


def plain_reason() -> str | None:
    """Why plain mode is on, in words, or None when it is off."""
    if _plain_forced:
        return "--plain was given"
    if os.environ.get("EDMARS_PLAIN", "").strip().lower() not in _TRUTHY_OFF:
        return "EDMARS_PLAIN is set"
    if os.environ.get("NO_COLOR", ""):
        return "NO_COLOR is set"
    if os.environ.get("TERM", "").strip().lower() == "dumb":
        return "TERM is dumb"
    if not _isatty(sys.stdout):
        return "output is not a terminal"
    if screen_reader_active():
        return "a screen reader is running"
    return None


def is_plain() -> bool:
    """True when output must be plain ASCII text without redraws."""
    return plain_reason() is not None


_MSYS_PTY = re.compile(r"(?:msys|cygwin)-[0-9a-f]+-pty\d+-(?:from|to)-master", re.IGNORECASE)


def _is_msys_pty(stream: Any) -> bool:
    """True when ``stream`` is the pipe mintty uses to fake a terminal.

    Native Windows programs under mintty see pipes, not a console, so
    ``isatty()`` is False even though a person is typing. The pipe's name
    gives it away.
    """
    try:
        import ctypes
        import msvcrt
        from ctypes import wintypes

        class _FileNameInfo(ctypes.Structure):
            _fields_ = [("FileNameLength", wintypes.DWORD), ("FileName", wintypes.WCHAR * 1024)]

        handle = msvcrt.get_osfhandle(stream.fileno())
        info = _FileNameInfo()
        func = ctypes.windll.kernel32.GetFileInformationByHandleEx  # type: ignore[attr-defined,unused-ignore]
        func.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        func.restype = wintypes.BOOL
        file_name_info = 2
        if not func(handle, file_name_info, ctypes.byref(info), ctypes.sizeof(info)):
            return False
        name = info.FileName[: info.FileNameLength // 2]
        return bool(_MSYS_PTY.search(name))
    except Exception:
        return False


def is_mintty() -> bool:
    """True inside Git Bash's mintty window (no Windows console attached)."""
    if sys.platform != "win32" or not os.environ.get("MSYSTEM"):
        return False
    if _isatty(sys.stdin):
        return False  # bash inside Windows Terminal or winpty: a real console
    return _is_msys_pty(sys.stdin)


def is_interactive() -> bool:
    """True when a person can answer prompts right now."""
    if _non_interactive or _input_closed:
        return False
    return _isatty(sys.stdin) or is_mintty()


def _use_questionary() -> bool:
    if not is_interactive() or is_plain() or is_mintty():
        return False
    if not (_isatty(sys.stdin) and _isatty(sys.stdout)):
        return False
    try:
        import questionary  # noqa: F401
    except Exception:
        return False
    return True


# --- Consoles -------------------------------------------------------------


def _console_for(stderr: bool) -> Console:
    plain = is_plain()
    key = (stderr, plain)
    found = _consoles.get(key)
    if found is None:
        found = Console(
            stderr=stderr,
            highlight=False,
            emoji=False,
            soft_wrap=plain,
            no_color=True if plain else None,
            color_system=None if plain else "auto",
        )
        _consoles[key] = found
    return found


class _ConsoleProxy:
    """Forwards to the rich Console that fits the current mode.

    Modules do ``from edmars.ui import console`` once at import time; the
    proxy lets ``--plain`` (parsed later) still change what they print to,
    and follows ``sys.stdout`` when a test harness replaces it.
    """

    def __init__(self, stderr: bool) -> None:
        self._stderr = stderr

    def __getattr__(self, name: str) -> Any:
        return getattr(_console_for(self._stderr), name)

    def __repr__(self) -> str:
        return f"<edmars console proxy stderr={self._stderr}>"


#: rich Console for normal output (stdout). A forwarding proxy: use it for
#: ``console.print(...)``; hand :func:`get_console` to rich objects that
#: keep a console of their own (``Live``, ``Progress``).
console: Console = cast(Console, _ConsoleProxy(stderr=False))
#: rich Console for warnings and errors (stderr).
err_console: Console = cast(Console, _ConsoleProxy(stderr=True))


def get_console(stderr: bool = False) -> Console:
    """The real rich Console for the current mode (for ``Live``/``Progress``)."""
    return _console_for(stderr or _machine_output)


# --- Glyphs ---------------------------------------------------------------

_UNICODE_GLYPHS: dict[str, str] = {
    "ok": "✓",
    "warn": "!",
    "fail": "✗",
    "info": "i",
    "running": "◐",
    "pending": "·",
    "arrow": "→",
    "bullet": "•",
    "dash": "—",
    "ellipsis": "…",
}

_ASCII_GLYPHS: dict[str, str] = {
    "ok": "[ok]",
    "warn": "[!]",
    "fail": "[x]",
    "info": "[i]",
    "running": "[..]",
    "pending": "[ ]",
    "arrow": "->",
    "bullet": "-",
    "dash": "--",
    "ellipsis": "...",
}


def _can_encode(stream: Any, text: str) -> bool:
    encoding = getattr(stream, "encoding", None) or "ascii"
    try:
        text.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return False
    return True


def use_ascii() -> bool:
    """True when symbols must be ASCII (plain mode or a limited encoding)."""
    return is_plain() or not _can_encode(sys.stdout, "".join(_UNICODE_GLYPHS.values()))


def glyph(name: str) -> str:
    """The symbol for ``name`` (ok, warn, fail, info, running, pending, ...)."""
    table = _ASCII_GLYPHS if use_ascii() else _UNICODE_GLYPHS
    return table.get(name, _ASCII_GLYPHS.get(name, name))


def glyphs() -> dict[str, str]:
    """All symbols for the current mode (for the live view)."""
    return dict(_ASCII_GLYPHS if use_ascii() else _UNICODE_GLYPHS)


# --- Messages -------------------------------------------------------------

_STYLES = {"ok": "bold green", "warn": "bold yellow", "fail": "bold red", "info": "bold cyan"}


def _say(kind: str, message: str) -> None:
    to_stderr = kind in ("warn", "fail") or _machine_output
    target = err_console if to_stderr else console
    symbol = glyph(kind)
    if is_plain():
        target.print(f"{symbol} {message}", markup=False, highlight=False, soft_wrap=True)
    else:
        target.print(Text.assemble((symbol, _STYLES[kind]), " ", message))


def ok(msg: str) -> None:
    """Something succeeded."""
    _say("ok", msg)


def info(msg: str) -> None:
    """A neutral fact or a next step."""
    _say("info", msg)


def warn(msg: str) -> None:
    """Something needs attention but does not stop anything."""
    _say("warn", msg)


def fail(msg: str) -> None:
    """Something failed; say what to do next in the same or next message."""
    _say("fail", msg)


def say(msg: str = "") -> None:
    """Print a plain line (no glyph, no markup interpretation)."""
    target = err_console if _machine_output else console
    target.print(msg, markup=False, highlight=False, soft_wrap=is_plain())


def show_checks(checks: Iterable[Check]) -> None:
    """Print a checklist: one line per check, then its fix when there is one."""
    printers = {"ok": ok, "warn": warn, "fail": fail, "info": info}
    for check in checks:
        printers.get(check.status, info)(f"{check.name}: {check.detail}")
        if check.fix and check.status in ("warn", "fail"):
            say(f"    {glyph('arrow')} {check.fix}")


def panel(title: str, body: str) -> None:
    """Show a titled block of text (a boxed panel, or a heading in plain mode)."""
    body = body.rstrip("\n")
    if is_plain():
        say(title)
        say("-" * max(3, min(len(title), 72)))
        say(body)
        say()
        return
    console.print(
        Panel(
            Text(body),
            title=Text(title, style="bold"),
            title_align="left",
            expand=False,
            padding=(0, 1),
        )
    )


def table(
    columns: Sequence[str], rows: Sequence[Sequence[Any]], title: str | None = None
) -> None:
    """Show rows under column headings (aligned ASCII columns in plain mode)."""
    cells = [[("" if c is None else str(c)) for c in row] for row in rows]
    if is_plain():
        if title:
            say(title)
        widths = [len(c) for c in columns]
        for row in cells:
            for i, cell in enumerate(row[: len(widths)]):
                widths[i] = max(widths[i], len(cell))

        def fmt(values: Sequence[str]) -> str:
            padded = [v.ljust(widths[i]) for i, v in enumerate(values[: len(widths)])]
            return "  ".join(padded).rstrip()

        say(fmt(list(columns)))
        say(fmt(["-" * w for w in widths]))
        for row in cells:
            say(fmt(row))
        return
    grid = Table(title=title, box=box.SIMPLE_HEAD, show_edge=False)
    for col in columns:
        grid.add_column(col)
    for row in cells:
        grid.add_row(*(Text(c) for c in row))
    console.print(grid)


@contextlib.contextmanager
def status(message: str) -> Iterator[None]:
    """Show ``message`` while a slow step runs (a spinner, or one line in plain mode)."""
    if is_plain() or _machine_output:
        info(message)
        yield
        return
    with console.status(Text(message)):
        yield


def open_path(path: Path | str) -> None:
    """Open a file or folder with the computer's default application."""
    from edmars import proc

    try:
        proc.open_with_default_app(Path(path))
    except Exception:
        info(f"Could not open it automatically. It is here: {path}")


# --- Prompts ----------------------------------------------------------------

Validator = Callable[[str], "bool | str | None"]


def _validation_error(validate: Validator | None, value: str) -> str | None:
    """None when ``value`` passes, else the message to show."""
    if validate is None:
        return None
    result = validate(value)
    if result is True or result is None:
        return None
    if result is False:
        return "That answer is not valid; please try again."
    return str(result)


def _read_line(prompt: str) -> str | None:
    """``input()`` that returns None at end of input instead of raising."""
    try:
        return input(prompt)
    except EOFError:
        return None


def _end_of_input(message: str) -> NonInteractiveError:
    """The error for a prompt that read end of input.

    A person who presses Enter sends an empty line; end of input means no
    one is typing (a pipe, a closed console, or Git Bash's pipes mistaken
    for mintty's). The default is NOT taken: a default of "Start the
    study" would spend money nobody agreed to.
    """
    global _input_closed
    _input_closed = True
    say()
    return NonInteractiveError(message)


def select(
    message: str,
    choices: list[tuple[str, str]],
    default: str | None = None,
    *,
    disabled: dict[str, str] | None = None,
) -> str:
    """Ask the user to pick one of ``choices`` and return its value.

    ``choices`` are ``(value, label)`` pairs. ``disabled`` maps a value to
    the reason it cannot be picked right now; such options are shown
    greyed out (or marked "not available") with that reason.
    """
    if not choices:
        raise ValueError("select() needs at least one choice")
    values = [value for value, _ in choices]
    unavailable = dict(disabled or {})
    if default is not None and default not in values:
        raise ValueError(f"default {default!r} is not one of the choices")
    if default in unavailable:
        default = None

    if not is_interactive():
        if default is not None:
            return default
        raise NonInteractiveError(message)

    if _use_questionary():
        import questionary

        options = [
            questionary.Choice(title=label, value=value, disabled=unavailable.get(value))
            for value, label in choices
        ]
        answer = questionary.select(message, choices=options, default=default).unsafe_ask()
        if answer is None:
            raise KeyboardInterrupt
        return str(answer)

    say(message)
    for number, (value, label) in enumerate(choices, start=1):
        line = f"  {number}) {label}"
        if value in unavailable:
            line += f"  (not available: {unavailable[value]})"
        elif value == default:
            line += "  (default)"
        say(line)
    default_number = values.index(default) + 1 if default is not None else None
    prompt = f"Type a number from 1 to {len(values)}"
    prompt += f" [{default_number}]: " if default_number else ": "
    while True:
        raw = _read_line(prompt)
        if raw is None:
            raise _end_of_input(message)
        raw = raw.strip()
        if not raw and default is not None:
            return default
        pick: str | None = None
        if raw.isdigit() and 1 <= int(raw) <= len(values):
            pick = values[int(raw) - 1]
        elif raw in values:
            pick = raw
        if pick is None:
            say("Please type one of the numbers shown.")
            continue
        if pick in unavailable:
            say(f"That option is not available: {unavailable[pick]}")
            continue
        return pick


def text(message: str, default: str | None = None, validate: Validator | None = None) -> str:
    """Ask for a line of text.

    ``validate`` returns True/None when the answer is fine, or an error
    message (or False) to ask again.
    """
    if not is_interactive():
        if default is not None:
            return default
        raise NonInteractiveError(message)

    if _use_questionary():
        import questionary

        def _q_validate(value: str) -> bool | str:
            error = _validation_error(validate, value.strip())
            return error or True

        answer = questionary.text(
            message, default=default or "", validate=_q_validate if validate else None
        ).unsafe_ask()
        if answer is None:
            raise KeyboardInterrupt
        answer = str(answer).strip()
        return answer if answer or default is None else default

    suffix = f" [{default}]" if default else ""
    while True:
        raw = _read_line(f"{message}{suffix}: ")
        if raw is None:
            raise _end_of_input(message)
        value = raw.strip()
        if not value and default is not None:
            value = default
        error = _validation_error(validate, value)
        if error:
            say(error)
            continue
        return value


def _without_hidden_claim(message: str) -> str:
    """``message`` without "it stays hidden", for a terminal that shows typing."""
    message = re.sub(r"\s*\(it stays hidden\)", "", message)
    return re.sub(r"it stays hidden[;,]\s*", "", message)


def _erase_rows(typed_chars: int) -> str:
    """ANSI codes that erase the ``typed_chars`` characters just echoed.

    After Enter the cursor is at the start of the next row. The prompt and
    the input filled ceil(typed_chars / width) rows above it; go up that
    many and clear to the end of the screen. The width is the terminal's
    when it can be read, else 80 columns. A wider window only means a
    line or two above the prompt is cleared as well.
    """
    import shutil

    width = max(1, shutil.get_terminal_size((80, 24)).columns)
    rows = max(1, -(-max(typed_chars, 1) // width))
    return f"\x1b[{rows}A\r\x1b[J"


def secret(message: str) -> str:
    """Ask for a secret (an API key) without showing it on screen.

    Git Bash's window (mintty) cannot hide typing: there the key is shown
    while it is pasted and erased from the screen after Enter.
    """
    if not is_interactive():
        raise NonInteractiveError(
            message,
            hint=(
                "Set the key in the environment variable the documentation "
                "names, or run `edmars setup` in a terminal window."
            ),
        )

    if _use_questionary():
        import questionary

        answer = questionary.password(message).unsafe_ask()
        if answer is None:
            raise KeyboardInterrupt
        return str(answer).strip()

    if is_mintty():
        warn(
            "Git Bash cannot hide what you type. The key will be erased from "
            "the screen after you press Enter. (PowerShell or Windows Terminal "
            "hide it as you type.)"
        )
        prompt = f"{_without_hidden_claim(message)}: "
        raw = _read_line(prompt)
        if raw is None:
            raise _end_of_input(message)
        # A long key wraps over several rows; erase every row the prompt and
        # the key took, not only the last one (mintty understands ANSI escapes).
        sys.stdout.write(_erase_rows(len(prompt) + len(raw)))
        sys.stdout.flush()
        return raw.strip()

    if _isatty(sys.stdin):
        return getpass.getpass(f"{message}: ").strip()

    raw = _read_line(f"{message}: ")
    if raw is None:
        raise _end_of_input(message)
    return raw.strip()


def confirm(message: str, default: bool = True) -> bool:
    """Ask a yes/no question.

    Without anyone to ask (``--yes`` or no terminal) the default is
    returned. A caller about to spend money or delete something should
    therefore check :func:`is_interactive` itself and require an explicit
    flag, rather than rely on a default of True.
    """
    if not is_interactive():
        return default

    if _use_questionary():
        import questionary

        answer = questionary.confirm(message, default=default).unsafe_ask()
        if answer is None:
            raise KeyboardInterrupt
        return bool(answer)

    hint = "[Y/n]" if default else "[y/N]"
    while True:
        raw = _read_line(f"{message} {hint} ")
        if raw is None:
            raise _end_of_input(message)
        answer_text = raw.strip().lower()
        if not answer_text:
            return default
        if answer_text in ("y", "yes"):
            return True
        if answer_text in ("n", "no"):
            return False
        say("Please answer y or n.")
