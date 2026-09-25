"""Shared fixtures and fakes for the providers/datasets/toolchain/LSAR tests.

Other edmars modules (model, paths, settings, proc, secrets) are built on
a sibling branch. When one of them is missing here, a minimal stand-in
that follows the fixed API in CLI_SPEC section 18 is registered, so these
tests run on this branch alone. Once the branches are merged the real
modules exist and no stand-in is installed.

Fixtures are imported into each test module by name (pytest discovers
them there); nothing here is a conftest, so it cannot collide with the
package conftest another branch adds.
"""

from __future__ import annotations

import importlib
import os
import shutil
import socket
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Literal

import pytest
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Stand-ins for modules owned by other branches
# ---------------------------------------------------------------------------


def _build_model(mod: types.ModuleType) -> None:
    @dataclass
    class Check:
        name: str
        status: Literal["ok", "warn", "fail", "info"]
        detail: str
        fix: str | None = None

    Check.__module__ = mod.__name__
    mod.Check = Check  # type: ignore[attr-defined]


def _build_paths(mod: types.ModuleType) -> None:
    def home_override() -> Path | None:
        value = os.environ.get("EDMARS_HOME")
        return Path(value) if value else None

    def app_root() -> Path:
        value = os.environ.get("EDMARS_APP_ROOT")
        return Path(value) if value else REPO_ROOT

    def config_dir() -> Path:
        home = home_override()
        return home if home else Path.home() / ".edm-ars-test-config"

    def data_dir() -> Path:
        home = home_override()
        return (home / "data") if home else Path.home() / ".edm-ars-test-data"

    def cache_dir() -> Path:
        home = home_override()
        return (home / "cache") if home else Path.home() / ".edm-ars-test-cache"

    def settings_path() -> Path:
        return config_dir() / "settings.yaml"

    def sync_provider(path: Path) -> str | None:
        return None

    for fn in (home_override, app_root, config_dir, data_dir, cache_dir,
               settings_path, sync_provider):
        setattr(mod, fn.__name__, fn)


def _build_settings(mod: types.ModuleType) -> None:
    import copy

    import yaml

    defaults: dict[str, Any] = {
        "schema": 1, "datasets": {}, "latex": {"mode": None, "pdflatex": None},
        "r": {"rscript": None, "packages_ok": False},
        "lsar": {"enabled": False, "auto_review": False, "home": None, "ref": None},
    }

    def _path() -> Path:
        from edmars import paths

        return Path(paths.settings_path())

    def get(settings: dict[str, Any], dotted: str, default: Any = None) -> Any:
        node: Any = settings
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set_(settings: dict[str, Any], dotted: str, value: Any) -> None:
        node = settings
        parts = dotted.split(".")
        for part in parts[:-1]:
            if not isinstance(node.get(part), dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = value

    def load() -> dict[str, Any]:
        data = copy.deepcopy(defaults)
        if _path().is_file():
            data.update(yaml.safe_load(_path().read_text(encoding="utf-8")) or {})
        return data

    def save(settings: dict[str, Any]) -> None:
        _path().parent.mkdir(parents=True, exist_ok=True)
        tmp = _path().with_suffix(".tmp")
        tmp.write_text(yaml.safe_dump(settings), encoding="utf-8")
        os.replace(tmp, _path())

    mod.DEFAULTS = defaults  # type: ignore[attr-defined]
    for fn in (get, set_, load, save):
        setattr(mod, fn.__name__, fn)


def _build_proc(mod: types.ModuleType) -> None:
    def run(args: list[str], *, timeout: float | None = None,
            env: dict[str, str] | None = None,
            cwd: str | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(  # noqa: S603 - test stand-in for edmars.proc
            list(args), capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=timeout, env=env, cwd=cwd,
            stdin=subprocess.DEVNULL,
        )

    def which(name: str) -> str | None:
        return shutil.which(name)

    mod.run = run  # type: ignore[attr-defined]
    mod.which = which  # type: ignore[attr-defined]


def _build_secrets(mod: types.ModuleType) -> None:
    def get_secret(name: str) -> str | None:
        return os.environ.get(name) or None

    mod.get_secret = get_secret  # type: ignore[attr-defined]


def _install(name: str, build: Callable[[types.ModuleType], None]) -> None:
    full = f"edmars.{name}"
    try:
        importlib.import_module(full)
        return
    except ModuleNotFoundError as exc:
        if exc.name != full:
            raise
    import edmars

    mod = types.ModuleType(full)
    mod.__dict__["__edmars_test_stub__"] = True
    build(mod)
    sys.modules[full] = mod
    setattr(edmars, name, mod)


for _name, _builder in (("model", _build_model), ("paths", _build_paths),
                        ("settings", _build_settings), ("proc", _build_proc),
                        ("secrets", _build_secrets)):
    _install(_name, _builder)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def edmars_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway EDMARS_HOME and a keyring that stores nothing."""
    home = tmp_path / "edmars-home"
    home.mkdir()
    monkeypatch.setenv("EDMARS_HOME", str(home))
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.null.Keyring")
    monkeypatch.delenv("EDM_ARS_RSCRIPT", raising=False)
    return home


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any real HTTP request or outbound socket fails the test loudly."""

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(f"real network access attempted: {args[:2]!r}")

    monkeypatch.setattr(requests.Session, "request", refuse)
    real_connect = socket.socket.connect

    def guarded_connect(self: socket.socket, address: Any) -> Any:
        if self.family in (socket.AF_INET, socket.AF_INET6):
            host = address[0] if isinstance(address, tuple) else address
            if host not in ("127.0.0.1", "::1", "localhost"):
                raise AssertionError(f"real network access attempted: {address!r}")
        return real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


# ---------------------------------------------------------------------------
# HTTP fakes
# ---------------------------------------------------------------------------


class FakeResponse:
    """Enough of requests.Response for the code under test."""

    def __init__(
        self,
        status: int = 200,
        *,
        json_data: Any = None,
        body: bytes = b"",
        headers: dict[str, str] | None = None,
        fail_after: int | None = None,
        chunk: int = 7,
    ) -> None:
        self.status_code = status
        self._json = json_data
        self._body = body
        self.headers = dict(headers or {})
        self._fail_after = fail_after
        self._chunk = chunk

    @property
    def text(self) -> str:
        if self._json is not None:
            import json

            return json.dumps(self._json)
        return self._body.decode("utf-8", "replace")

    def json(self) -> Any:
        if self._json is None:
            raise ValueError("not json")
        return self._json

    def iter_content(self, chunk_size: int = 1) -> Iterator[bytes]:
        sent = 0
        for start in range(0, len(self._body), self._chunk):
            piece = self._body[start:start + self._chunk]
            if self._fail_after is not None and sent + len(piece) > self._fail_after:
                cut = self._fail_after - sent
                if cut > 0:
                    yield piece[:cut]
                raise requests.exceptions.ConnectionError("connection reset (fake)")
            sent += len(piece)
            yield piece

    def close(self) -> None:
        pass

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class FakeSession:
    """Records every GET and answers with ``handler(url, headers, params)``."""

    def __init__(self, handler: Callable[[str, dict[str, str], dict[str, Any]], Any]) -> None:
        self.handler = handler
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, headers: dict[str, str] | None = None,
            params: dict[str, Any] | None = None, stream: bool = False,
            timeout: Any = None) -> Any:
        self.calls.append({"url": url, "headers": dict(headers or {}),
                           "params": dict(params or {}), "timeout": timeout,
                           "stream": stream})
        result = self.handler(url, dict(headers or {}), dict(params or {}))
        if isinstance(result, BaseException):
            raise result
        return result


def serve_bytes(
    body: bytes, *, etag: str = '"v1"', fail_first_after: int | None = None,
    honour_range: bool = True,
) -> FakeSession:
    """A session serving ``body`` with Range support; optionally drops once."""
    state = {"dropped": False}

    def handler(url: str, headers: dict[str, str], params: dict[str, Any]) -> FakeResponse:
        rng = headers.get("Range")
        if rng and honour_range:
            start = int(rng.split("=")[1].rstrip("-"))
            if start >= len(body):
                return FakeResponse(416, headers={"Content-Range": f"bytes */{len(body)}"})
            part = body[start:]
            return FakeResponse(206, body=part, headers={
                "Content-Range": f"bytes {start}-{len(body) - 1}/{len(body)}",
                "Content-Length": str(len(part)), "ETag": etag})
        fail = None
        if fail_first_after is not None and not state["dropped"]:
            state["dropped"] = True
            fail = fail_first_after
        return FakeResponse(200, body=body, fail_after=fail, headers={
            "Content-Length": str(len(body)), "ETag": etag})

    return FakeSession(handler)


def completed(args: list[str], returncode: int = 0, stdout: str = "",
              stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args, returncode, stdout, stderr)
