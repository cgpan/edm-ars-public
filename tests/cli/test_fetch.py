"""The shared download helper: resume edge cases, limits and hashing."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from tests.cli.infra_helpers import (  # noqa: F401 - fixtures
    FakeResponse,
    FakeSession,
    edmars_home,
    no_network,
    serve_bytes,
)

from edmars import fetch  # noqa: E402

pytestmark = pytest.mark.usefixtures("edmars_home", "no_network")

URL = "https://downloads.example.com/file.zip"
BODY = bytes(range(256)) * 40


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fetch, "_sleep", lambda s: None)


def _prime(dest: Path, data: bytes) -> None:
    """Leave a partial download behind, as an interrupted run would."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.with_name(dest.name + ".part").write_bytes(data)
    dest.with_name(dest.name + ".part.json").write_text(
        '{"url": "%s", "etag": "\\"v1\\""}' % URL, encoding="utf-8")


def test_plain_download_asks_for_unencoded_bytes(tmp_path: Path) -> None:
    session = serve_bytes(BODY)
    dest = fetch.download_file(URL, tmp_path / "f.zip", session=session)
    assert dest.read_bytes() == BODY
    assert session.calls[0]["headers"]["Accept-Encoding"] == "identity"
    assert session.calls[0]["stream"] is True
    assert not (tmp_path / "f.zip.part").exists()
    assert not (tmp_path / "f.zip.part.json").exists()


def test_a_resume_at_the_wrong_offset_starts_over(tmp_path: Path) -> None:
    dest = tmp_path / "f.zip"
    _prime(dest, BODY[:100])

    def handler(url: str, headers: dict[str, str], params: dict[str, Any]) -> FakeResponse:
        if "Range" in headers:  # answers with the wrong range
            return FakeResponse(206, body=BODY[50:], headers={
                "Content-Range": f"bytes 50-{len(BODY) - 1}/{len(BODY)}"})
        return FakeResponse(200, body=BODY, headers={"Content-Length": str(len(BODY))})

    fetch.download_file(URL, dest, session=FakeSession(handler))
    assert dest.read_bytes() == BODY


def test_a_complete_partial_file_is_finished_on_416(tmp_path: Path) -> None:
    dest = tmp_path / "f.zip"
    _prime(dest, BODY)
    session = FakeSession(lambda u, h, p: FakeResponse(
        416, headers={"Content-Range": f"bytes */{len(BODY)}"}))
    fetch.download_file(URL, dest, session=session)
    assert dest.read_bytes() == BODY
    assert session.calls[0]["headers"]["If-Range"] == '"v1"'


def test_a_partial_file_from_another_url_is_discarded(tmp_path: Path) -> None:
    dest = tmp_path / "f.zip"
    _prime(dest, b"x" * 100)
    session = serve_bytes(BODY)
    fetch.download_file("https://downloads.example.com/other.zip", dest, session=session)
    assert "Range" not in session.calls[0]["headers"]
    assert dest.read_bytes() == BODY


def test_oversized_downloads_are_refused_and_not_kept(tmp_path: Path) -> None:
    dest = tmp_path / "f.zip"
    with pytest.raises(fetch.DownloadError, match="larger than expected"):
        fetch.download_file(URL, dest, session=serve_bytes(BODY), max_bytes=1000)
    no_length = FakeSession(lambda u, h, p: FakeResponse(200, body=BODY))
    with pytest.raises(fetch.DownloadError, match="exceeded"):
        fetch.download_file(URL, dest, session=no_length, max_bytes=1000)
    assert not dest.exists()
    assert not dest.with_name("f.zip.part").exists()


def test_server_errors_are_retried_then_reported(tmp_path: Path) -> None:
    session = FakeSession(lambda u, h, p: FakeResponse(503))
    with pytest.raises(fetch.DownloadError, match="HTTP 503"):
        fetch.download_file(URL, tmp_path / "f.zip", session=session, attempts=3)
    assert len(session.calls) == 3


def test_sha256_and_progress(tmp_path: Path) -> None:
    path = tmp_path / "blob"
    path.write_bytes(BODY * 700)  # > 1 chunk
    seen: list[tuple[int, int | None]] = []
    digest = fetch.sha256_file(path, progress=lambda d, t: seen.append((d, t)))
    assert digest == hashlib.sha256(BODY * 700).hexdigest()
    assert seen[-1] == (len(BODY) * 700, len(BODY) * 700)

    def broken(done: int, total: int | None, phase: str) -> None:
        raise RuntimeError("progress bar crashed")

    assert fetch.sha256_file(path, progress=broken) == digest


def test_human_bytes() -> None:
    assert fetch.human_bytes(512) == "512 bytes"
    assert fetch.human_bytes(2_147_483_648) == "2.1 GB"
    assert fetch.human_bytes(297_000_000) == "297.0 MB"
