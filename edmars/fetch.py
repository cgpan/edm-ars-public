"""Streamed, resumable HTTP downloads and file hashing.

Used only for downloads the user has approved: a dataset from NCES, the
LSAR reviewer from GitHub, and the TinyTeX installer. Nothing here runs on
its own, and nothing here sends anything but an ordinary GET request.

A download is written to ``<dest>.part`` and moved into place only when it
is complete, so an interrupted download never leaves a file that looks
finished. A later call resumes it with an HTTP ``Range`` request; the
server's ETag (or Last-Modified date) is sent as ``If-Range`` so a file
that changed on the server in the meantime is fetched again from the
start instead of being spliced onto the old bytes.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable

import requests

#: ``progress(done_bytes, total_bytes_or_None[, phase])``. The third
#: argument is passed only to callables that accept it, so both
#: ``cb(done, total)`` and ``cb(done, total, phase)`` work.
ProgressFn = Callable[..., None]

CHUNK_BYTES = 1 << 20
#: (connect, read) timeouts in seconds for one request.
DEFAULT_TIMEOUT: tuple[float, float] = (15.0, 60.0)
DEFAULT_ATTEMPTS = 4

#: Monkeypatched by tests so retries do not sleep.
_sleep: Callable[[float], None] = time.sleep


class DownloadError(RuntimeError):
    """A download failed in a way a retry did not fix."""


def user_agent() -> str:
    """The User-Agent sent with every CLI download."""
    try:
        from edmars import __version__ as version  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - version is cosmetic here
        version = "0"
    return f"edm-ars-cli/{version} (+https://github.com/cgpan/edm-ars-public)"


def _accepts_phase(progress: ProgressFn) -> bool:
    try:
        params = list(inspect.signature(progress).parameters.values())
    except (TypeError, ValueError):
        return False
    positional = [
        p for p in params
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    if any(p.kind is p.VAR_POSITIONAL for p in params):
        return True
    return len(positional) >= 3


def emit_progress(
    progress: ProgressFn | None, done: int, total: int | None, phase: str
) -> None:
    """Call a progress callback; a callback that raises never stops the work."""
    if progress is None:
        return
    try:
        if _accepts_phase(progress):
            progress(done, total, phase)
        else:
            progress(done, total)
    except Exception:  # noqa: BLE001 - a broken progress bar is not fatal
        pass


def _content_range_total(value: str | None) -> int | None:
    """Total size from ``Content-Range: bytes a-b/total`` (or ``*/total``)."""
    if not value:
        return None
    match = re.search(r"/(\d+)\s*$", value)
    return int(match.group(1)) if match else None


def _read_meta(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_meta(path: Path, meta: dict[str, Any]) -> None:
    try:
        path.write_text(json.dumps(meta), encoding="utf-8")
    except OSError:
        pass


def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def download_file(
    url: str,
    dest: Path,
    *,
    progress: ProgressFn | None = None,
    session: Any | None = None,
    timeout: tuple[float, float] | float = DEFAULT_TIMEOUT,
    attempts: int = DEFAULT_ATTEMPTS,
    max_bytes: int | None = None,
    phase: str = "download",
) -> Path:
    """Download ``url`` to ``dest``, resuming a previous partial download.

    Returns ``dest``. Raises :class:`DownloadError` when the server refuses
    the request (4xx), when ``max_bytes`` is exceeded, or when every
    attempt failed on a connection problem.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    meta_path = dest.with_name(dest.name + ".part.json")
    http = session if session is not None else requests
    meta = _read_meta(meta_path)
    if meta.get("url") != url:
        # A partial file from a different URL is not ours to continue.
        _unlink(part)
        meta = {}

    last_error: str | None = None
    for attempt in range(max(1, attempts)):
        if attempt:
            _sleep(min(2.0 ** attempt, 15.0))
        have = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": user_agent()}
        if have > 0:
            headers["Range"] = f"bytes={have}-"
            validator = meta.get("etag") or meta.get("last_modified")
            if validator:
                headers["If-Range"] = str(validator)
        try:
            response = http.get(url, headers=headers, stream=True, timeout=timeout)
            with response:
                status = int(response.status_code)
                if status == 416 and have > 0:
                    total = _content_range_total(response.headers.get("Content-Range"))
                    if total is not None and total == have:
                        break  # the partial file was already complete
                    _unlink(part)
                    meta = {}
                    last_error = "the server rejected the resume request"
                    continue
                if status == 206 and have > 0:
                    mode = "ab"
                    start = have
                    total = _content_range_total(response.headers.get("Content-Range"))
                    if total is None:
                        length = response.headers.get("Content-Length")
                        total = have + int(length) if length and length.isdigit() else None
                elif status == 200:
                    mode = "wb"
                    start = 0
                    length = response.headers.get("Content-Length")
                    total = int(length) if length and length.isdigit() else None
                elif 500 <= status < 600 or status == 429:
                    last_error = f"the server answered HTTP {status}"
                    continue
                else:
                    raise DownloadError(
                        f"The server refused the download (HTTP {status}) for {url}."
                    )
                meta = {
                    "url": url,
                    "etag": response.headers.get("ETag"),
                    "last_modified": response.headers.get("Last-Modified"),
                    "total": total,
                }
                _write_meta(meta_path, meta)
                if max_bytes is not None and total is not None and total > max_bytes:
                    raise DownloadError(
                        f"The file at {url} is larger than expected "
                        f"({total:,} bytes; limit {max_bytes:,})."
                    )
                done = start
                emit_progress(progress, done, total, phase)
                with open(part, mode) as out:
                    for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
                        if not chunk:
                            continue
                        out.write(chunk)
                        done += len(chunk)
                        if max_bytes is not None and done > max_bytes:
                            raise DownloadError(
                                f"The download from {url} exceeded {max_bytes:,} bytes."
                            )
                        emit_progress(progress, done, total, phase)
                if total is not None and done < total:
                    last_error = (
                        f"the connection closed early ({done:,} of {total:,} bytes)"
                    )
                    continue
                break
        except DownloadError:
            raise
        except requests.exceptions.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            continue
    else:
        raise DownloadError(
            f"Could not download {url} after {max(1, attempts)} attempts "
            f"({last_error or 'unknown error'}). Run the same command again to "
            "resume; the part already downloaded is kept."
        )

    os.replace(part, dest)
    _unlink(meta_path)
    return dest


def sha256_file(
    path: Path, progress: ProgressFn | None = None, phase: str = "verify"
) -> str:
    """SHA-256 of a file, read in 1 MiB chunks (safe for multi-GB files)."""
    digest = hashlib.sha256()
    path = Path(path)
    total = path.stat().st_size
    done = 0
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            done += len(chunk)
            emit_progress(progress, done, total, phase)
    return digest.hexdigest()


def human_bytes(n: float) -> str:
    """``2_147_483_648`` -> ``"2.1 GB"`` (decimal units, as disks are sold)."""
    value = float(n)
    for unit in ("bytes", "KB", "MB", "GB", "TB"):
        if abs(value) < 1000 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "bytes" else f"{value:.1f} {unit}"
        value /= 1000.0
    return f"{value:.1f} TB"  # pragma: no cover - loop always returns
