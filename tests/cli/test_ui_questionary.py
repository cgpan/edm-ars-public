"""The arrow-key prompts, driven through prompt_toolkit's pipe input.

These use the REAL questionary and prompt_toolkit (only the terminal is
simulated), so a call that does not match questionary's API fails here
rather than in front of a user.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from edmars import ui

DOWN = "\x1b[B"
ENTER = "\r"


@pytest.fixture
def keys(monkeypatch: pytest.MonkeyPatch) -> Iterator:  # type: ignore[type-arg]
    monkeypatch.setattr(ui, "is_interactive", lambda: True)
    monkeypatch.setattr(ui, "_use_questionary", lambda: True)
    with create_pipe_input() as pipe:
        with create_app_session(input=pipe, output=DummyOutput()):
            yield pipe.send_text


CHOICES = [("deepseek", "DeepSeek (recommended)"), ("openai", "OpenAI"), ("local", "Local")]


def test_select_moves_and_returns_the_value(keys) -> None:  # type: ignore[no-untyped-def]
    keys(DOWN + ENTER)
    assert ui.select("Which AI service?", CHOICES) == "openai"


def test_select_starts_on_the_default(keys) -> None:  # type: ignore[no-untyped-def]
    keys(ENTER)
    assert ui.select("Which AI service?", CHOICES, default="local") == "local"


def test_select_skips_disabled_options(keys) -> None:  # type: ignore[no-untyped-def]
    keys(DOWN + ENTER)
    picked = ui.select("Which AI service?", CHOICES, disabled={"openai": "no key yet"})
    assert picked == "local"


def test_text_with_default_and_validation(keys) -> None:  # type: ignore[no-untyped-def]
    keys(ENTER)
    assert ui.text("Venue", default="EDM") == "EDM"


def test_text_validation_rejects_then_accepts(keys) -> None:  # type: ignore[no-untyped-def]
    keys("bad" + ENTER + "\x7f\x7f\x7f" + "good" + ENTER)
    assert ui.text("Word", validate=lambda v: v == "good" or "Please type good.") == "good"


def test_secret_is_stripped(keys) -> None:  # type: ignore[no-untyped-def]
    keys("  sk-fake-typed-0123456789  " + ENTER)
    assert ui.secret("Paste your key") == "sk-fake-typed-0123456789"


def test_confirm(keys) -> None:  # type: ignore[no-untyped-def]
    keys("n")
    assert ui.confirm("Continue?", default=True) is False
