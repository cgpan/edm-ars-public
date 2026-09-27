"""Generated-code output must survive non-ASCII characters on Windows.

With the ANSI code page as the child's stdout encoding, LLM-written code
died with UnicodeEncodeError on its first print of a check mark, an arrow
or a Greek letter -- usually at the end of a long script, costing a retry
(an archived ITR run lost an Analyst attempt to a final ``print`` with
'->' as a real arrow). In the other direction, a byte the parent's locale
codec could not decode killed the reader thread and the captured stream
came back as ``None``, which the agents' retry prompts then sliced.

The executor now forces UTF-8 on the child and decodes as UTF-8 with
replacement. On macOS/Linux these tests pass trivially; the parent env is
set to cp1252 to reproduce the Windows condition anywhere.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from src.sandbox import SubprocessExecutor, child_env


def test_child_env_forces_utf8_stdio(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTHONIOENCODING", "cp1252")
    monkeypatch.delenv("PYTHONUTF8", raising=False)
    env = child_env()
    assert env["PYTHONUTF8"] == "1"
    assert env["PYTHONIOENCODING"] == "utf-8"


def test_child_env_with_an_explicit_base_is_utf8_too() -> None:
    env = child_env({"OUTPUT_DIR": "/workspace"})
    assert env["PYTHONUTF8"] == "1"
    assert env["PYTHONIOENCODING"] == "utf-8"


def test_non_ascii_prints_do_not_crash_the_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTHONIOENCODING", "cp1252")
    code = "print('done \\u2713 \\u03c7\\u00b2 \\u2192 \\u2265 \\u03b1')\n"
    result = SubprocessExecutor().run(code, output_dir=str(tmp_path), timeout_s=60)
    assert result["returncode"] == 0, result["stderr"]
    assert "\u2713 \u03c7\u00b2 \u2192 \u2265 \u03b1" in result["stdout"]


def test_undecodable_bytes_never_lose_the_stream(tmp_path: Path) -> None:
    code = (
        "import sys\n"
        "sys.stdout.write('before\\n'); sys.stdout.flush()\n"
        "sys.stdout.buffer.write(b'\\xc3\\x81 \\xff\\xfe\\x81\\n'); sys.stdout.buffer.flush()\n"
        "sys.stderr.buffer.write(b'err \\xff\\x81\\n'); sys.stderr.buffer.flush()\n"
    )
    result = SubprocessExecutor().run(code, output_dir=str(tmp_path), timeout_s=60)
    assert result["returncode"] == 0
    assert isinstance(result["stdout"], str)
    assert isinstance(result["stderr"], str)
    assert "before" in result["stdout"]
    assert "\u00c1" in result["stdout"]
    assert "\ufffd" in result["stdout"]
    assert "err" in result["stderr"]


def test_the_executor_decodes_as_utf8_with_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def _fake(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen.update(kwargs)
        return subprocess.CompletedProcess(argv, 0, stdout=None, stderr=None)

    monkeypatch.setattr("src.sandbox.subprocess.run", _fake)
    result = SubprocessExecutor().run("print(1)", output_dir=str(tmp_path))
    assert seen["encoding"] == "utf-8"
    assert seen["errors"] == "replace"
    assert seen["env"]["PYTHONUTF8"] == "1"
    # A None stream is normalised, never handed to the retry prompt.
    assert result["stdout"] == "" and result["stderr"] == ""


# ---------------------------------------------------------------------------
# No __pycache__ in the study folder
# ---------------------------------------------------------------------------


def test_child_env_writes_no_bytecode() -> None:
    assert child_env()["PYTHONDONTWRITEBYTECODE"] == "1"
    assert child_env({"OUTPUT_DIR": "/workspace"})["PYTHONDONTWRITEBYTECODE"] == "1"


def test_importing_a_helper_leaves_no_pycache_in_the_study_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The owner's Mac test (round 2) found __pycache__/ in the study
    folder: generated code imports analysis_helpers.py from it."""
    monkeypatch.delenv("PYTHONDONTWRITEBYTECODE", raising=False)
    (tmp_path / "analysis_helpers.py").write_text(
        "def answer():\n    return 42\n", encoding="utf-8"
    )
    code = "import analysis_helpers\nprint(analysis_helpers.answer())\n"
    result = SubprocessExecutor().run(code, output_dir=str(tmp_path), timeout_s=60)
    assert result["returncode"] == 0, result["stderr"]
    assert "42" in result["stdout"]
    assert not (tmp_path / "__pycache__").exists()

