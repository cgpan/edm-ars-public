"""Unit and integration tests for src/sandbox.py.

Unit tests mock the Docker SDK entirely — no daemon required.
Integration tests (marked @pytest.mark.integration) require a running Docker
daemon and the edm-ars-sandbox:latest image.

Run unit tests:
    pytest tests/test_sandbox.py -v -k "not integration"

Run integration tests:
    pytest tests/test_sandbox.py -v --run-integration
"""
from __future__ import annotations

import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.sandbox import DockerSandbox, SubprocessExecutor, create_executor


# ---------------------------------------------------------------------------
# SubprocessExecutor — unit tests (real subprocess, no mocks needed)
# ---------------------------------------------------------------------------


def test_subprocess_executor_runs_simple_code() -> None:
    ex = SubprocessExecutor()
    result = ex.run("print('hello')", output_dir=".", timeout_s=10)
    assert result["returncode"] == 0
    assert "hello" in result["stdout"]


def test_subprocess_executor_timeout() -> None:
    ex = SubprocessExecutor()
    result = ex.run("import time; time.sleep(999)", output_dir=".", timeout_s=1)
    assert result["returncode"] == -1
    assert "Timeout" in result["stderr"]


def test_subprocess_executor_syntax_error() -> None:
    ex = SubprocessExecutor()
    result = ex.run("this is not python", output_dir=".", timeout_s=10)
    assert result["returncode"] != 0


def test_subprocess_executor_returns_stderr() -> None:
    ex = SubprocessExecutor()
    result = ex.run("import sys; sys.stderr.write('err msg')", output_dir=".", timeout_s=10)
    assert result["returncode"] == 0
    assert "err msg" in result["stderr"]


def test_subprocess_executor_raw_data_path_ignored(tmp_path: Path) -> None:
    """raw_data_path kwarg is accepted but has no effect in SubprocessExecutor."""
    ex = SubprocessExecutor()
    result = ex.run(
        "print('ok')",
        output_dir=str(tmp_path),
        raw_data_path="/nonexistent/path.csv",
        timeout_s=10,
    )
    assert result["returncode"] == 0


# ---------------------------------------------------------------------------
# create_executor — unit tests
# ---------------------------------------------------------------------------


def test_create_executor_disabled() -> None:
    config: dict = {"sandbox": {"enabled": False}}
    ex = create_executor(config)
    assert isinstance(ex, SubprocessExecutor)


def test_create_executor_no_sandbox_key() -> None:
    ex = create_executor({})
    assert isinstance(ex, SubprocessExecutor)


def test_create_executor_docker_unavailable() -> None:
    """When docker.from_env() raises, fall back with a RuntimeWarning."""
    config: dict = {"sandbox": {"enabled": True}}
    with patch("src.sandbox.docker") as mock_docker:
        mock_docker.from_env.side_effect = Exception("Docker not running")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ex = create_executor(config)
        assert isinstance(ex, SubprocessExecutor)
        assert any(issubclass(w.category, RuntimeWarning) for w in caught)


def test_create_executor_docker_available() -> None:
    config: dict = {
        "sandbox": {
            "enabled": True,
            "image": "edm-ars-sandbox:latest",
            "memory_limit": "4g",
            "cpu_count": 2,
            "network_disabled": True,
            "auto_build": True,
        }
    }
    with patch("src.sandbox._DOCKER_AVAILABLE", True), \
         patch("src.sandbox.docker") as mock_docker:
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        mock_client.ping.return_value = True
        ex = create_executor(config)
    assert isinstance(ex, DockerSandbox)
    assert ex.image == "edm-ars-sandbox:latest"
    assert ex.memory_limit == "4g"
    assert ex.cpu_count == 2
    assert ex.network_disabled is True
    assert ex.auto_build is True


def test_create_executor_docker_sdk_not_installed() -> None:
    """If docker SDK is missing (_DOCKER_AVAILABLE=False), fall back gracefully."""
    config: dict = {"sandbox": {"enabled": True}}
    with patch("src.sandbox._DOCKER_AVAILABLE", False):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ex = create_executor(config)
        assert isinstance(ex, SubprocessExecutor)
        assert any(issubclass(w.category, RuntimeWarning) for w in caught)


# ---------------------------------------------------------------------------
# DockerSandbox — init defaults
# ---------------------------------------------------------------------------


def test_docker_sandbox_init_defaults() -> None:
    ds = DockerSandbox()
    assert ds.image == "edm-ars-sandbox:latest"
    assert ds.memory_limit == "4g"
    assert ds.cpu_count == 2
    assert ds.network_disabled is True
    assert ds.auto_build is True


def test_docker_sandbox_init_custom() -> None:
    ds = DockerSandbox(
        image="my-image:v2",
        memory_limit="2g",
        cpu_count=4,
        network_disabled=False,
        auto_build=False,
    )
    assert ds.image == "my-image:v2"
    assert ds.memory_limit == "2g"
    assert ds.cpu_count == 4
    assert ds.network_disabled is False
    assert ds.auto_build is False


# ---------------------------------------------------------------------------
# DockerSandbox.run() — mocked unit tests
# ---------------------------------------------------------------------------


def _make_mock_client(
    stdout: bytes = b"stdout output",
    stderr: bytes = b"",
    exit_code: int = 0,
) -> MagicMock:
    """Return a mock docker client whose container.wait() resolves immediately."""
    mock_client = MagicMock()
    mock_container = MagicMock()
    mock_client.containers.create.return_value = mock_container
    mock_container.wait.return_value = {"StatusCode": exit_code}
    # logs(stdout=True, stderr=False) → stdout bytes
    # logs(stdout=False, stderr=True) → stderr bytes
    mock_container.logs.side_effect = lambda stdout=True, stderr=True: (
        stdout_bytes if stdout else stderr_bytes
        for stdout_bytes, stderr_bytes in [
            (mock_container._stdout, mock_container._stderr)
        ]
    ).__next__()
    mock_container._stdout = stdout
    mock_container._stderr = stderr
    # Simpler approach: use a real side_effect function
    def _logs(stdout: bool = True, stderr: bool = True) -> bytes:
        if stdout and not stderr:
            return mock_container._stdout
        if stderr and not stdout:
            return mock_container._stderr
        return mock_container._stdout + mock_container._stderr

    mock_container.logs.side_effect = _logs
    return mock_client


def test_docker_sandbox_run_success(tmp_path: Path) -> None:
    mock_client = _make_mock_client(stdout=b"hello sandbox\n", stderr=b"", exit_code=0)
    ds = DockerSandbox(auto_build=False)
    ds._client = mock_client

    result = ds.run("print('hello sandbox')", output_dir=str(tmp_path), timeout_s=30)

    assert result["returncode"] == 0
    assert "hello sandbox" in result["stdout"]
    # Container must always be removed
    mock_client.containers.create.return_value.remove.assert_called_once_with(force=True)


def test_docker_sandbox_run_passes_env_vars(tmp_path: Path) -> None:
    mock_client = _make_mock_client()
    ds = DockerSandbox(auto_build=False)
    ds._client = mock_client

    ds.run(
        "print('ok')",
        output_dir=str(tmp_path),
        raw_data_path="/some/dir/data.csv",
        timeout_s=30,
    )

    create_kwargs = mock_client.containers.create.call_args
    env = create_kwargs[1].get("environment") or create_kwargs[0][2]
    # environment may be positional or keyword depending on impl
    create_kwargs_all = mock_client.containers.create.call_args
    _, kwargs = create_kwargs_all
    environment = kwargs.get("environment", {})
    assert environment.get("RAW_DATA_PATH", "").endswith("data.csv")
    assert environment.get("OUTPUT_DIR") == "/workspace"


def test_docker_sandbox_run_timeout(tmp_path: Path) -> None:
    mock_client = MagicMock()
    mock_container = MagicMock()
    mock_client.containers.create.return_value = mock_container
    # Simulate timeout on wait()
    mock_container.wait.side_effect = Exception("Read timeout")

    ds = DockerSandbox(auto_build=False)
    ds._client = mock_client

    result = ds.run("import time; time.sleep(999)", output_dir=str(tmp_path), timeout_s=1)

    assert result["returncode"] == -1
    assert "Timeout" in result["stderr"]
    mock_container.kill.assert_called_once()
    mock_container.remove.assert_called_once_with(force=True)


def test_docker_sandbox_run_docker_exception_fallback(tmp_path: Path) -> None:
    """Any DockerException → fall back to subprocess with warning."""
    # Create a mock DockerException class for use without the real docker SDK
    class FakeDockerException(Exception):
        pass

    class FakeImageNotFound(FakeDockerException):
        pass

    mock_docker_errors = MagicMock()
    mock_docker_errors.DockerException = FakeDockerException
    mock_docker_errors.ImageNotFound = FakeImageNotFound

    mock_client = MagicMock()
    mock_client.containers.create.side_effect = FakeDockerException("boom")
    ds = DockerSandbox(auto_build=False)
    ds._client = mock_client

    with patch("src.sandbox.docker") as mock_docker_module, \
         patch("src.sandbox._DOCKER_AVAILABLE", True):
        mock_docker_module.errors = mock_docker_errors
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = ds.run("print('fallback')", output_dir=str(tmp_path), timeout_s=10)

    assert result["returncode"] == 0
    assert "fallback" in result["stdout"]
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)


def test_docker_sandbox_run_image_not_found_no_auto_build(tmp_path: Path) -> None:
    """ImageNotFound + auto_build=False → subprocess fallback with warning."""
    # Create mock exception classes that mirror docker.errors hierarchy
    class FakeDockerException(Exception):
        pass

    class FakeImageNotFound(FakeDockerException):
        pass

    mock_docker_errors = MagicMock()
    mock_docker_errors.DockerException = FakeDockerException
    mock_docker_errors.ImageNotFound = FakeImageNotFound

    mock_client = MagicMock()
    mock_client.containers.create.side_effect = FakeImageNotFound("no image")
    ds = DockerSandbox(auto_build=False)
    ds._client = mock_client

    with patch("src.sandbox.docker") as mock_docker_module, \
         patch("src.sandbox._DOCKER_AVAILABLE", True):
        mock_docker_module.errors = mock_docker_errors
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = ds.run("print('no_image_fallback')", output_dir=str(tmp_path), timeout_s=10)

    assert result["returncode"] == 0
    assert "no_image_fallback" in result["stdout"]
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)


def test_docker_sandbox_container_always_removed_on_exception(tmp_path: Path) -> None:
    """Container.remove(force=True) is called even when wait() raises."""
    mock_client = MagicMock()
    mock_container = MagicMock()
    mock_client.containers.create.return_value = mock_container
    mock_container.wait.side_effect = Exception("some error")

    ds = DockerSandbox(auto_build=False)
    ds._client = mock_client

    ds.run("print('x')", output_dir=str(tmp_path), timeout_s=5)

    mock_container.remove.assert_called_once_with(force=True)


# ---------------------------------------------------------------------------
# Integration tests — require running Docker daemon
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_docker_sandbox_real_execution(tmp_path: Path) -> None:
    ds = DockerSandbox()
    result = ds.run("print('sandbox ok')", output_dir=str(tmp_path), timeout_s=30)
    assert result["returncode"] == 0
    assert "sandbox ok" in result["stdout"]


@pytest.mark.integration
def test_docker_sandbox_real_timeout(tmp_path: Path) -> None:
    ds = DockerSandbox()
    result = ds.run("import time; time.sleep(999)", output_dir=str(tmp_path), timeout_s=5)
    assert result["returncode"] == -1


@pytest.mark.integration
def test_docker_sandbox_file_io(tmp_path: Path) -> None:
    ds = DockerSandbox()
    result = ds.run(
        "open('/workspace/test_output.txt', 'w').write('ok')",
        output_dir=str(tmp_path),
        timeout_s=30,
    )
    assert result["returncode"] == 0
    assert (tmp_path / "test_output.txt").exists()


@pytest.mark.integration
def test_docker_sandbox_no_network(tmp_path: Path) -> None:
    """Container must not be able to reach the internet (network_disabled=True)."""
    ds = DockerSandbox(network_disabled=True)
    result = ds.run(
        "import urllib.request; urllib.request.urlopen('http://example.com', timeout=5)",
        output_dir=str(tmp_path),
        timeout_s=20,
    )
    # Expect a network error (non-zero exit)
    assert result["returncode"] != 0
