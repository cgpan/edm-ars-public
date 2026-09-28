"""Shared fixtures and fakes for the providers/datasets/toolchain/LSAR tests.

Fixtures are imported into each test module by name (pytest discovers
them there). The stand-ins this file once registered for sibling modules
(model, paths, settings, proc, secrets, ui) are gone: the real modules
exist, and the tests run against them.
"""

from __future__ import annotations

import socket
import subprocess
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]


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
