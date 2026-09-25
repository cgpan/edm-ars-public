"""Fixtures shared by every test of the ``edmars`` application.

All of them are autouse, so no test in ``tests/cli/`` can reach the real
machine state by accident:

* ``EDMARS_HOME`` points at a temp folder, so settings, datasets, caches
  and the default studies folder all live under ``tmp_path``.
* The credential store is an in-memory fake, patched both where
  ``edmars.secrets`` looks it up and on the ``keyring`` module itself (in
  case another module calls keyring directly). The real Windows
  Credential Manager / macOS Keychain is never read or written.
* Provider key variables are removed from the environment. The parent
  conftest injects fake ones, and a developer's shell may hold REAL ones;
  either would make "no key yet" tests pass or fail for the wrong reason.
* ``requests`` refuses to open connections; a test that needs an HTTP
  answer patches ``requests.get``/``Session.request`` itself.
* The ui module's global modes are reset before and after each test.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

_CLEARED_ENV = (
    "DEEPSEEK_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "SEMANTIC_SCHOLAR_API_KEY",
    "TAVILY_API_KEY",
    "MINIMAX_API_KEY",
    "EDMARS_APP_ROOT",
    "EDMARS_DEBUG",
    "EDMARS_PLAIN",
    "NO_COLOR",
    "MSYSTEM",
)


class FakeKeyring:
    """A dict-backed stand-in for the ``keyring`` module and a backend."""

    priority = 1

    def __init__(self) -> None:
        self.store: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, name: str) -> str | None:
        return self.store.get((service, name))

    def set_password(self, service: str, name: str, value: str) -> None:
        self.store[(service, name)] = value

    def delete_password(self, service: str, name: str) -> None:
        import keyring.errors

        if (service, name) not in self.store:
            raise keyring.errors.PasswordDeleteError("not found")
        del self.store[(service, name)]

    def get_keyring(self) -> Any:
        return self


@pytest.fixture(autouse=True)
def edmars_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "edmars_home"
    monkeypatch.setenv("EDMARS_HOME", str(home))
    for name in _CLEARED_ENV:
        monkeypatch.delenv(name, raising=False)
    return home


@pytest.fixture(autouse=True)
def fake_keyring(monkeypatch: pytest.MonkeyPatch) -> FakeKeyring:
    fake = FakeKeyring()
    import keyring

    import edmars.secrets as secrets

    monkeypatch.setattr(secrets, "_keyring", lambda: fake)
    monkeypatch.setattr(secrets, "_SEEN", set())
    monkeypatch.setattr(secrets, "_WARNED", set())
    for attr in ("get_password", "set_password", "delete_password", "get_keyring"):
        monkeypatch.setattr(keyring, attr, getattr(fake, attr))
    return fake


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    import requests

    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("network access is disabled in the edmars tests")

    monkeypatch.setattr(requests.sessions.Session, "request", refuse)


@pytest.fixture(autouse=True)
def reset_ui() -> Iterator[None]:
    from edmars import ui

    ui.reset()
    yield
    ui.reset()


# The run/live-view/results fixtures (``run_home`` and the synthetic run
# folders) live in ``_run_support.py``; importing it here makes them
# available to every test module.
from tests.cli._run_support import run_home  # noqa: E402,F401
