"""``python -m edmars``: the entry point and its UTF-8 streams."""

from __future__ import annotations

import io
import os
import subprocess
import sys
from pathlib import Path

import pytest

from edmars import __main__ as entry

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_streams_are_switched_to_utf8(monkeypatch: pytest.MonkeyPatch) -> None:
    out = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    err = io.TextIOWrapper(io.BytesIO(), encoding="ascii")
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    entry._force_utf8_streams()
    assert out.encoding.lower().replace("-", "") == "utf8"
    assert err.encoding.lower().replace("-", "") == "utf8"
    out.write("\u2713 \u00e9")  # would raise under cp1252/ascii
    out.flush()


def test_streams_without_reconfigure_are_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    class Bare:
        pass

    monkeypatch.setattr(sys, "stdout", Bare())
    entry._force_utf8_streams()


def test_module_entry_point_runs(edmars_home: Path) -> None:
    env = dict(os.environ, EDMARS_HOME=str(edmars_home), PYTHONIOENCODING="ascii")
    result = subprocess.run(
        [sys.executable, "-m", "edmars", "explain", "omega"],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    assert b"McDonald's omega" in result.stdout


def test_importing_the_cli_is_light_and_side_effect_free() -> None:
    code = (
        "import sys, edmars.cli; "
        "heavy = sorted(m for m in sys.modules if m == 'src' or m.startswith('src.') "
        "or m in ('keyring', 'questionary', 'requests', 'psutil'))\n"
        "print(','.join(heavy))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""
