"""Generated code must run under the interpreter that runs the pipeline.

The executor used to launch a bare ``python``. On Windows a venv's
python.exe is a redirector for the base install, and CreateProcess looks in
the running image's own directory before PATH, so ``python`` resolved to
the BASE interpreter even inside the activated venv the README tells users
to create -- one without pandas/sklearn. The DataEngineer then spent its
LLM retries rewriting code that was never at fault. On macOS a bare
``python`` often does not exist, and the FileNotFoundError escaped the
executor and aborted the stage with "[Errno 2] ... 'python'".
"""
from __future__ import annotations

import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.sandbox import (
    INTERPRETER_NOT_STARTED,
    DockerSandbox,
    SubprocessExecutor,
    create_executor,
)


class _Recorder:
    """Stand-in for subprocess.run that records argv and succeeds."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append((list(argv), kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout="ok\n", stderr="")


def test_the_script_runs_under_sys_executable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rec = _Recorder()
    monkeypatch.setattr("src.sandbox.subprocess.run", rec)
    SubprocessExecutor().run("print(1)", output_dir=str(tmp_path))
    argv = rec.calls[0][0]
    assert argv[0] == sys.executable
    assert argv[0] != "python"


def test_the_config_override_is_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rec = _Recorder()
    monkeypatch.setattr("src.sandbox.subprocess.run", rec)
    SubprocessExecutor(python_executable="/opt/analysis/bin/python").run(
        "print(1)", output_dir=str(tmp_path)
    )
    assert rec.calls[0][0][0] == "/opt/analysis/bin/python"


def test_the_child_sees_the_parents_installation(tmp_path: Path) -> None:
    """What actually broke under a Windows venv: a different sys.prefix."""
    result = SubprocessExecutor().run(
        "import sys; print(sys.prefix)", output_dir=str(tmp_path), timeout_s=60
    )
    assert result["returncode"] == 0, result["stderr"]
    assert Path(result["stdout"].strip()) == Path(sys.prefix)


def test_an_interpreter_that_cannot_start_is_a_returncode_not_a_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _missing(argv: list[str], **kwargs: Any) -> None:
        raise FileNotFoundError(2, "No such file or directory", argv[0])

    monkeypatch.setattr("src.sandbox.subprocess.run", _missing)
    result = SubprocessExecutor(python_executable="python9").run(
        "print(1)", output_dir=str(tmp_path)
    )
    assert result["returncode"] == INTERPRETER_NOT_STARTED == 127
    assert "python9" in result["stderr"]
    assert "sandbox.python_executable" in result["stderr"]
    assert result["stdout"] == ""
    # The temp script is still cleaned up.
    assert not (tmp_path / "_generated_script.py").exists()


def test_a_real_missing_interpreter_is_reported(tmp_path: Path) -> None:
    missing = str(tmp_path / "no_such_dir" / "python-missing")
    result = SubprocessExecutor(python_executable=missing).run(
        "print(1)", output_dir=str(tmp_path), timeout_s=10
    )
    assert result["returncode"] == 127
    assert "python-missing" in result["stderr"]


def test_a_bad_output_dir_still_raises(tmp_path: Path) -> None:
    """Only the interpreter launch is mapped to 127; a missing working
    directory is a caller bug and keeps raising as before."""
    with pytest.raises(OSError):
        SubprocessExecutor().run(
            "print(1)", output_dir=str(tmp_path / "does_not_exist"), timeout_s=10
        )


def test_create_executor_carries_the_interpreter_setting() -> None:
    ex = create_executor(
        {"sandbox": {"enabled": False, "python_executable": "/opt/py/bin/python"}}
    )
    assert isinstance(ex, SubprocessExecutor)
    assert ex.interpreter() == "/opt/py/bin/python"


@pytest.mark.parametrize("value", [None, ""])
def test_an_unset_interpreter_setting_means_sys_executable(value: Any) -> None:
    ex = create_executor({"sandbox": {"enabled": False, "python_executable": value}})
    assert ex.interpreter() == sys.executable


def test_a_null_sandbox_block_is_tolerated() -> None:
    """`sandbox:` with nothing under it loads as None, not {}."""
    ex = create_executor({"sandbox": None})
    assert isinstance(ex, SubprocessExecutor)


def test_the_docker_fallback_uses_the_configured_interpreter() -> None:
    config: dict[str, Any] = {
        "sandbox": {"enabled": True, "python_executable": "/opt/py/bin/python"}
    }
    with patch("src.sandbox.docker") as mock_docker:
        mock_docker.from_env.side_effect = Exception("Docker not running")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ex = create_executor(config)
    assert isinstance(ex, SubprocessExecutor)
    assert ex.interpreter() == "/opt/py/bin/python"


def test_a_docker_sandbox_falls_back_under_the_configured_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rec = _Recorder()
    monkeypatch.setattr("src.sandbox.subprocess.run", rec)
    ds = DockerSandbox(python_executable="/opt/py/bin/python")
    ds._get_client = MagicMock(side_effect=RuntimeError("daemon gone"))  # type: ignore[method-assign]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds.run("print(1)", output_dir=str(tmp_path))
    assert rec.calls[0][0][0] == "/opt/py/bin/python"
