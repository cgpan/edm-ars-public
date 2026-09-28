"""The wizard and doctor tests run against fakes; the fakes must match the real modules.

``wizard_fakes.py`` was written while the modules it stands in for were
being built on other branches. A fake that accepts a call the real
function rejects (or offers a function the real module does not have)
lets a wizard test pass while the real ``edmars setup`` crashes. This
test compares every function each fake module exports with the real
module's function of the same name:

* the real module has it;
* every parameter the fake names, the real function accepts too;
* every parameter the real function requires, the fake names too.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Any

import pytest

from tests.cli.wizard_fakes import install_fakes

#: Fake modules whose real counterparts are compared. ``cli`` is the typer
#: app (called, not introspected) and ``model`` holds only the dataclass.
COMPARED = ("ui", "paths", "settings", "disclosure", "secrets", "proc", "providers", "datasets",
            "toolchain", "lsar", "runner")

REAL = {name: importlib.import_module(f"edmars.{name}") for name in COMPARED}

#: The real defaults, read from the shipped config.yaml before any fake
#: replaces edmars.paths (install_fakes points app_root at a stub app).
REAL_DEFAULT_MODELS = {p: REAL["providers"].default_models(p) for p in ("deepseek", "openai", "anthropic", "local")}


def _params(func: Any) -> tuple[set[str], set[str], bool]:
    """(named parameters, required parameters, accepts **kwargs)."""
    signature = inspect.signature(func)
    named: set[str] = set()
    required: set[str] = set()
    var_kw = False
    for param in signature.parameters.values():
        if param.name == "self":
            continue
        if param.kind is param.VAR_KEYWORD:
            var_kw = True
            continue
        if param.kind is param.VAR_POSITIONAL:
            continue
        named.add(param.name)
        if param.default is param.empty:
            required.add(param.name)
    return named, required, var_kw


def _fake_modules(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    fakes = install_fakes(monkeypatch, tmp_path)
    return {name: fakes.modules[name] for name in COMPARED}


@pytest.mark.parametrize("name", COMPARED)
def test_fake_functions_match_the_real_ones(name: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = _fake_modules(monkeypatch, tmp_path)[name]
    real = REAL[name]
    problems: list[str] = []
    for attr, value in vars(fake).items():
        if attr.startswith("_"):
            continue
        if not hasattr(real, attr):
            problems.append(f"edmars.{name} has no {attr!r}")
            continue
        real_value = getattr(real, attr)
        if not (callable(value) and inspect.isroutine(value) and inspect.isroutine(real_value)):
            continue
        fake_named, fake_required, fake_kw = _params(value)
        real_named, real_required, real_kw = _params(real_value)
        if fake_kw and not fake_named:
            continue  # a guard such as proc.run(*a, **k) that fails the test if called
        extra = fake_named - real_named
        if extra and not real_kw:
            problems.append(f"{name}.{attr}: the fake takes {sorted(extra)}, the real one does not")
        missing = real_required - fake_named
        if missing:
            problems.append(f"{name}.{attr}: the real one requires {sorted(missing)}, the fake does not take them")
    assert problems == []


def test_fake_default_models_are_empty_exactly_where_the_real_ones_are(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Setup asks for a model only when the shipped config names none; the
    fake once invented OpenAI defaults, which hid that the wizard never asked."""
    fake = _fake_modules(monkeypatch, tmp_path)["providers"]
    for provider in ("deepseek", "openai", "anthropic", "local"):
        assert bool(fake.default_models(provider)) == bool(REAL_DEFAULT_MODELS[provider]), provider
