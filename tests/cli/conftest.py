"""Fixtures shared by every test of the ``edmars`` application.

All of them are autouse, so no test in ``tests/cli/`` can reach the real
machine state by accident:

* ``EDMARS_HOME`` points at a temp folder, so settings, datasets, caches
  and the default studies folder all live under ``tmp_path``.
* Provider key variables are removed from the environment. The parent
  conftest injects fake ones, and a developer's shell may hold REAL ones;
  either would make "no key yet" tests pass or fail for the wrong reason.
* The ui module's global modes are reset before and after each test.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

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


@pytest.fixture(autouse=True)
def edmars_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "edmars_home"
    monkeypatch.setenv("EDMARS_HOME", str(home))
    for name in _CLEARED_ENV:
        monkeypatch.delenv(name, raising=False)
    return home


@pytest.fixture(autouse=True)
def reset_ui() -> Iterator[None]:
    from edmars import ui

    ui.reset()
    yield
    ui.reset()
