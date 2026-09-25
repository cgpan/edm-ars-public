"""Terminal output modes and prompts, without a real terminal."""

from __future__ import annotations

import io
import sys

import pytest

from edmars import ui
from edmars.model import Check


@pytest.fixture
def typed(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    """Pretend a person is at a plain terminal and will type the given lines."""

    def feed(*lines: str) -> io.StringIO:
        stream = io.StringIO("".join(line + "\n" for line in lines))
        monkeypatch.setattr(sys, "stdin", stream)
        monkeypatch.setattr(ui, "is_interactive", lambda: True)
        monkeypatch.setattr(ui, "_use_questionary", lambda: False)
        return stream

    return feed


@pytest.fixture
def fancy_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend stdout is a UTF-8 terminal with no screen reader."""
    monkeypatch.setattr(ui, "_isatty", lambda stream: True)
    monkeypatch.setattr(ui, "screen_reader_active", lambda: False)
    monkeypatch.setattr(ui, "_can_encode", lambda stream, text: True)
    monkeypatch.delenv("TERM", raising=False)


# --- modes -----------------------------------------------------------------------


def test_output_that_is_not_a_terminal_is_plain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ui, "_isatty", lambda stream: False)
    assert ui.is_plain()
    assert ui.plain_reason() == "output is not a terminal"


def test_plain_flag_and_environment(monkeypatch: pytest.MonkeyPatch, fancy_terminal: None) -> None:
    assert not ui.is_plain()
    ui.set_plain(True)
    assert ui.plain_reason() == "--plain was given"
    ui.set_plain(False)
    monkeypatch.setenv("EDMARS_PLAIN", "1")
    assert ui.plain_reason() == "EDMARS_PLAIN is set"
    monkeypatch.setenv("EDMARS_PLAIN", "0")
    assert ui.plain_reason() is None
    monkeypatch.setenv("NO_COLOR", "1")
    assert ui.plain_reason() == "NO_COLOR is set"
    monkeypatch.delenv("NO_COLOR")
    monkeypatch.setenv("TERM", "dumb")
    assert ui.plain_reason() == "TERM is dumb"


def test_screen_reader_makes_output_plain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ui, "_isatty", lambda stream: True)
    monkeypatch.setattr(ui, "screen_reader_active", lambda: True)
    assert ui.plain_reason() == "a screen reader is running"


def test_glyphs_are_ascii_in_plain_mode(fancy_terminal: None) -> None:
    assert ui.glyph("ok") == "✓"
    ui.set_plain(True)
    assert ui.glyph("ok") == "[ok]"
    assert ui.glyph("fail") == "[x]"
    assert all(ord(ch) < 128 for g in ui.glyphs().values() for ch in g)


def test_glyphs_fall_back_when_the_terminal_cannot_encode_them(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ui, "is_plain", lambda: False)
    monkeypatch.setattr(ui, "_can_encode", lambda stream, text: False)
    assert ui.glyph("warn") == "[!]"


def test_messages_carry_glyphs_and_go_to_the_right_stream(capsys: pytest.CaptureFixture[str]) -> None:
    ui.set_plain(True)
    ui.ok("all good")
    ui.info("a fact")
    ui.warn("careful")
    ui.fail("broken [bold]not markup[/bold]")
    out, err = capsys.readouterr()
    assert "[ok] all good" in out and "[i] a fact" in out
    assert "[!] careful" in err
    assert "[x] broken [bold]not markup[/bold]" in err


def test_machine_output_moves_everything_to_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    ui.set_plain(True)
    ui.set_machine_output(True)
    ui.ok("done")
    ui.say("plain line")
    out, err = capsys.readouterr()
    assert out == ""
    assert "done" in err and "plain line" in err


def test_show_checks_prints_fixes_for_problems(capsys: pytest.CaptureFixture[str]) -> None:
    ui.set_plain(True)
    ui.show_checks(
        [
            Check("Python", "ok", "3.11"),
            Check("LaTeX", "fail", "pdflatex was not found", fix="edmars setup pdf"),
            Check("Docker", "info", "not used"),
        ]
    )
    out, err = capsys.readouterr()
    assert "[ok] Python: 3.11" in out
    assert "[x] LaTeX: pdflatex was not found" in err
    assert "-> edmars setup pdf" in out


def test_panel_and_table_are_plain_ascii(capsys: pytest.CaptureFixture[str]) -> None:
    ui.set_plain(True)
    ui.panel("Ready to start", "Question: does X predict Y?\nType: prediction\n")
    ui.table(["Study", "State"], [["2026-09-25_1402_gpa_ab12", "Ready"], ["x", None]])
    out = capsys.readouterr().out
    assert "Ready to start" in out and "Type: prediction" in out
    assert "Study" in out and "2026-09-25_1402_gpa_ab12  Ready" in out
    assert all(ord(ch) < 128 for ch in out)


def test_status_in_plain_mode_prints_one_line(capsys: pytest.CaptureFixture[str]) -> None:
    ui.set_plain(True)
    with ui.status("Checking the data"):
        pass
    assert capsys.readouterr().out.strip() == "[i] Checking the data"


def test_console_proxy_follows_mode(fancy_terminal: None) -> None:
    fancy = ui._console_for(False)
    ui.set_plain(True)
    plain = ui._console_for(False)
    assert fancy is not plain
    assert plain.no_color


# --- non-interactive -------------------------------------------------------------


def test_yes_mode_uses_defaults_or_fails_clearly() -> None:
    ui.set_non_interactive(True)
    assert not ui.is_interactive()
    assert ui.select("Pick", [("a", "A"), ("b", "B")], default="b") == "b"
    assert ui.text("Name", default="Ada") == "Ada"
    assert ui.confirm("Go on?", default=False) is False
    with pytest.raises(ui.NonInteractiveError) as info:
        ui.select("Which AI service?", [("a", "A")])
    assert "Which AI service" in str(info.value)
    assert "--yes was given" in str(info.value)
    with pytest.raises(ui.NonInteractiveError):
        ui.text("Your name")
    with pytest.raises(ui.NonInteractiveError):
        ui.secret("Paste your key")


def test_no_terminal_is_also_non_interactive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    assert not ui.is_interactive()
    with pytest.raises(ui.NonInteractiveError) as info:
        ui.text("Your name")
    assert "no terminal to ask in" in str(info.value)


def test_mintty_needs_msystem(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MSYSTEM", raising=False)
    assert not ui.is_mintty()


def test_msys_pipe_name_pattern() -> None:
    assert ui._MSYS_PTY.search(r"\msys-1888ae32e00d56aa-pty0-from-master")
    assert ui._MSYS_PTY.search(r"\cygwin-e022582115c10879-pty3-to-master")
    assert not ui._MSYS_PTY.search(r"\Device\NamedPipe\something-else")


# --- plain prompts ---------------------------------------------------------------


def test_plain_select_by_number_value_and_default(typed, capsys: pytest.CaptureFixture[str]) -> None:  # type: ignore[no-untyped-def]
    choices = [("deepseek", "DeepSeek (recommended)"), ("openai", "OpenAI"), ("local", "Local")]
    typed("9", "2")
    assert ui.select("Which AI service?", choices, default="deepseek") == "openai"
    out = capsys.readouterr().out
    assert "1) DeepSeek (recommended)  (default)" in out
    assert "Please type one of the numbers shown." in out
    typed("")
    assert ui.select("Which AI service?", choices, default="deepseek") == "deepseek"
    typed("local")
    assert ui.select("Which AI service?", choices) == "local"


def test_plain_select_without_a_default_needs_a_pick(typed, capsys: pytest.CaptureFixture[str]) -> None:  # type: ignore[no-untyped-def]
    # Consent prompts (the setup notice, a dataset's terms) pass no default:
    # an empty line must ask again, never answer.
    typed("", "2")
    picked = ui.select("Do you accept?", [("read", "Read it first"), ("accept", "I accept")], default=None)
    assert picked == "accept"
    out = capsys.readouterr().out
    assert "(default)" not in out
    assert "Please type one of the numbers shown." in out


def test_plain_select_refuses_disabled_options(typed, capsys: pytest.CaptureFixture[str]) -> None:  # type: ignore[no-untyped-def]
    typed("2", "1")
    picked = ui.select(
        "Dataset",
        [("hsls09_public", "HSLS:09"), ("did_els_hsls_panel", "ELS + HSLS panel")],
        disabled={"did_els_hsls_panel": "needs both datasets"},
    )
    assert picked == "hsls09_public"
    out = capsys.readouterr().out
    assert "(not available: needs both datasets)" in out
    assert "That option is not available: needs both datasets" in out


def test_end_of_input_never_takes_the_default(typed) -> None:  # type: ignore[no-untyped-def]
    # End of input is not a person pressing Enter: under Git Bash, pipes that
    # look like mintty's made "Start this study? [1]" read EOF and start a
    # paid study on the default. Every prompt now stops instead.
    typed()
    with pytest.raises(ui.NonInteractiveError) as info:
        ui.select("Start this study?", [("start", "Start"), ("cancel", "Cancel")], default="start")
    assert "end of the input" in str(info.value)
    # ... and nothing after it tries to ask again (the fixture patches
    # is_interactive itself, so the flag behind it is checked).
    assert ui._input_closed
    with pytest.raises(ui.NonInteractiveError):
        ui.select("Pick", [("a", "A")])
    ui.reset()
    typed()
    with pytest.raises(ui.NonInteractiveError):
        ui.text("Venue", default="EDM")
    ui.reset()
    typed()
    with pytest.raises(ui.NonInteractiveError):
        ui.confirm("Go ahead?", default=True)


def test_select_rejects_a_default_that_is_not_a_choice() -> None:
    with pytest.raises(ValueError):
        ui.select("Pick", [("a", "A")], default="z")
    with pytest.raises(ValueError):
        ui.select("Pick", [])


def test_plain_text_validates_and_uses_default(typed, capsys: pytest.CaptureFixture[str]) -> None:  # type: ignore[no-untyped-def]
    typed("bad", "  good  ")
    answer = ui.text("Word", validate=lambda v: v == "good" or "Please type good.")
    assert answer == "good"
    assert "Please type good." in capsys.readouterr().out
    typed("")
    assert ui.text("Venue", default="EDM") == "EDM"


def test_plain_confirm(typed) -> None:  # type: ignore[no-untyped-def]
    typed("maybe", "y")
    assert ui.confirm("Continue?", default=False) is True
    typed("")
    assert ui.confirm("Continue?", default=False) is False
    typed("NO")
    assert ui.confirm("Continue?") is False


def test_plain_secret_reads_a_stripped_line(typed) -> None:  # type: ignore[no-untyped-def]
    typed("  sk-fake-pasted-0123456789  ")
    assert ui.secret("Paste your key") == "sk-fake-pasted-0123456789"


def test_git_bash_secret_erases_every_row_the_key_took(typed, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:  # type: ignore[no-untyped-def]
    # mintty echoes the pasted key. Erasing only the last row left 158 of a
    # 164-character OpenAI key on screen in an 80-column window, right after
    # the prompt had said the key "stays hidden".
    import os
    import shutil

    key = "sk-proj-" + "F" * 156  # fake, 164 characters like a real project key
    typed(key)
    monkeypatch.setattr(ui, "is_mintty", lambda: True)
    monkeypatch.setattr(shutil, "get_terminal_size", lambda fallback=(80, 24): os.terminal_size((80, 24)))
    message = "Paste your OpenAI key (it stays hidden; press Enter on an empty line to go back)"
    assert ui.secret(message) == key
    out = capsys.readouterr().out
    prompt = "Paste your OpenAI key (press Enter on an empty line to go back): "
    assert prompt in out and "stays hidden" not in out
    rows = -(-(len(prompt) + len(key)) // 80)
    assert rows >= 3
    assert out.endswith(f"\x1b[{rows}A\r\x1b[J")
    assert ui._without_hidden_claim("Paste the server's key (it stays hidden)") == "Paste the server's key"


def test_get_console_returns_a_real_console_for_rich_widgets() -> None:
    from rich.console import Console

    assert isinstance(ui.get_console(), Console)
    assert not ui.get_console().stderr
    ui.set_machine_output(True)
    assert ui.get_console().stderr


def test_is_interactive_is_false_after_end_of_input(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ui, "_isatty", lambda stream: True)
    assert ui.is_interactive()
    monkeypatch.setattr(ui, "_input_closed", True)
    assert not ui.is_interactive()
