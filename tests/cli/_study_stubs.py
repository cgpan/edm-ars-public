"""Minimal stand-ins for sibling ``edmars`` modules, for the study-flow tests.

The ``edmars`` package is built in parallel pieces. ``edmars/study.py``
imports ``edmars.model`` (StudyPlan, Check) and ``edmars.paths``
(app_root, cache_dir) at module level; those modules are owned by the
foundation branch. This file installs a stub for each one ONLY when the
real module cannot be imported, so the same tests run unchanged once
the branches are merged and the real modules exist.

The stubs follow CLI_SPEC section 18 exactly and carry no behaviour
the study tests depend on beyond it.
"""
from __future__ import annotations

import importlib
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _stub_model() -> types.ModuleType:
    module = types.ModuleType("edmars.model")

    @dataclass
    class StudyPlan:
        task_type: str
        dataset: str
        research_question: str
        prompt: str | None = None
        spec: dict | None = None
        example_id: str | None = None
        experimental: bool = False
        venue: str = "EDM"
        paper_format: str = "conference"
        review: bool = False

    @dataclass
    class Check:
        name: str
        status: Literal["ok", "warn", "fail", "info"]
        detail: str
        fix: str | None = None

    StudyPlan.__module__ = "edmars.model"
    Check.__module__ = "edmars.model"
    module.StudyPlan = StudyPlan  # type: ignore[attr-defined]
    module.Check = Check  # type: ignore[attr-defined]
    return module


def _stub_paths() -> types.ModuleType:
    module = types.ModuleType("edmars.paths")

    def home_override() -> Path | None:
        value = os.environ.get("EDMARS_HOME")
        return Path(value) if value else None

    def app_root() -> Path:
        value = os.environ.get("EDMARS_APP_ROOT")
        return Path(value) if value else _REPO_ROOT

    def _base() -> Path:
        home = home_override()
        return home if home is not None else Path.home() / ".edm-ars-test"

    def config_dir() -> Path:
        return _base()

    def data_dir() -> Path:
        return _base() / "data-home"

    def cache_dir() -> Path:
        return _base() / "cache"

    def settings_path() -> Path:
        return config_dir() / "settings.yaml"

    def default_studies_dir() -> Path:
        return Path.home() / "EDM-ARS" / "studies"

    def sync_provider(path: Path) -> str | None:
        return None

    for fn in (home_override, app_root, config_dir, data_dir, cache_dir,
               settings_path, default_studies_dir, sync_provider):
        setattr(module, fn.__name__, fn)
    return module


_STUBS = {
    "edmars.model": _stub_model,
    "edmars.paths": _stub_paths,
}


def install() -> list[str]:
    """Install a stub for every missing sibling module; return their names."""
    installed: list[str] = []
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    for name, factory in _STUBS.items():
        try:
            importlib.import_module(name)
            continue
        except ModuleNotFoundError as exc:
            if exc.name not in (name, "edmars"):
                raise  # the real module exists but is broken: surface it
        module = factory()
        sys.modules[name] = module
        parent = sys.modules.get("edmars")
        if parent is None:
            parent = importlib.import_module("edmars")
        setattr(parent, name.rsplit(".", 1)[1], module)
        installed.append(name)
    return installed
