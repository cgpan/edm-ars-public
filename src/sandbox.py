"""Sandboxed code execution for EDM-ARS.

Provides two executor implementations:
- SubprocessExecutor: bare subprocess.run() (fast, no isolation)
- DockerSandbox: Docker-container-based execution (isolated, resource-limited)

Use create_executor(config) to get the appropriate executor based on config.yaml.
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import warnings
from typing import TYPE_CHECKING, Any

try:
    import docker  # type: ignore[import-not-found]
    import docker.errors  # type: ignore[import-not-found]
    import requests.exceptions  # type: ignore[import-not-found]
    _DOCKER_AVAILABLE = True
except ImportError:
    _DOCKER_AVAILABLE = False
    docker = None  # type: ignore[assignment]


class SubprocessExecutor:
    """Execute LLM-generated code via bare subprocess.run().

    Mirrors the interface of DockerSandbox.run() so the two are interchangeable.
    """

    def run(
        self,
        code: str,
        output_dir: str,
        raw_data_path: str | None = None,
        timeout_s: int = 300,
    ) -> dict[str, Any]:
        """Run *code* as a Python -c invocation in *output_dir*.

        Parameters
        ----------
        code:
            Python source string to execute.
        output_dir:
            Working directory for the subprocess (files written here).
        raw_data_path:
            Ignored by SubprocessExecutor (path is embedded in the code by the
            agent). Accepted for interface parity with DockerSandbox.
        timeout_s:
            Hard timeout in seconds; returns returncode -1 on expiry.

        Returns
        -------
        dict with keys: stdout, stderr, returncode
        """
        # Write code to a temp file instead of passing via -c to avoid
        # Windows command-line length limit (WinError 206, ~32k char cap).
        script_path = os.path.join(output_dir, "_generated_script.py")
        try:
            with open(script_path, "w", encoding="utf-8") as fh:
                fh.write(code)
            result = subprocess.run(
                ["python", script_path],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=output_dir,
            )
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": f"Timeout after {timeout_s}s",
                "returncode": -1,
            }
        finally:
            if os.path.exists(script_path):
                os.remove(script_path)


class DockerSandbox:
    """Execute LLM-generated code inside a Docker container.

    The container runs as a non-root ``sandbox`` user with no network access,
    capped memory and CPU, and ephemeral filesystem state (only the bind-mounted
    volumes persist).

    Volume mounts
    -------------
    * output_dir  → /workspace  (read-write)
    * dirname(raw_data_path)  → /data/raw  (read-only)

    Environment variables injected
    --------------------------------
    * RAW_DATA_PATH=/data/raw/{filename}
    * OUTPUT_DIR=/workspace
    """

    def __init__(
        self,
        image: str = "edm-ars-sandbox:latest",
        memory_limit: str = "4g",
        cpu_count: int = 2,
        network_disabled: bool = True,
        auto_build: bool = True,
    ) -> None:
        self.image = image
        self.memory_limit = memory_limit
        self.cpu_count = cpu_count
        self.network_disabled = network_disabled
        self.auto_build = auto_build
        self._client: Any = None  # lazy-initialised on first run()

    def _get_client(self) -> Any:
        """Return (and cache) a docker.DockerClient."""
        if self._client is None:
            if not _DOCKER_AVAILABLE:
                raise RuntimeError("docker Python SDK is not installed")
            self._client = docker.from_env()
        return self._client

    def _build_image(self) -> None:
        """Build the sandbox image from the project-root Dockerfile."""
        project_root = str(pathlib.Path(__file__).parent.parent)
        client = self._get_client()
        print(f"[DockerSandbox] Building image {self.image} from {project_root} ...")
        client.images.build(path=project_root, tag=self.image, rm=True)
        print(f"[DockerSandbox] Image {self.image} built successfully.")

    def run(
        self,
        code: str,
        output_dir: str,
        raw_data_path: str | None = None,
        timeout_s: int = 300,
    ) -> dict[str, Any]:
        """Execute *code* inside a Docker container.

        Falls back to SubprocessExecutor on:
        - docker.errors.ImageNotFound (and auto-build fails)
        - Any docker.errors.DockerException

        Parameters
        ----------
        code:
            Python source string passed to ``python -c <code>``.
        output_dir:
            Host path mounted at /workspace inside the container (rw).
        raw_data_path:
            Optional host path to the raw data file. Its parent directory is
            mounted read-only at /data/raw. Injects RAW_DATA_PATH env var.
        timeout_s:
            Seconds to wait for the container to finish before killing it.

        Returns
        -------
        dict with keys: stdout, stderr, returncode
        """
        try:
            client = self._get_client()
        except Exception as exc:
            warnings.warn(
                f"DockerSandbox: cannot connect to Docker daemon ({exc}). "
                "Falling back to subprocess.",
                RuntimeWarning,
                stacklevel=2,
            )
            return SubprocessExecutor().run(
                code=code, output_dir=output_dir,
                raw_data_path=raw_data_path, timeout_s=timeout_s,
            )

        # Build volume-mount dict
        volumes: dict[str, dict[str, str]] = {
            os.path.abspath(output_dir): {"bind": "/workspace", "mode": "rw"},
        }
        environment: dict[str, str] = {"OUTPUT_DIR": "/workspace"}

        if raw_data_path is not None:
            raw_data_abs = os.path.abspath(raw_data_path)
            raw_data_dir = os.path.dirname(raw_data_abs)
            raw_filename = os.path.basename(raw_data_abs)
            volumes[raw_data_dir] = {"bind": "/data/raw", "mode": "ro"}
            environment["RAW_DATA_PATH"] = f"/data/raw/{raw_filename}"

        # nano_cpus: Docker API takes CPU quota as integer nanoseconds per second
        nano_cpus = int(self.cpu_count * 1e9)

        container: Any = None
        try:
            container = client.containers.create(
                self.image,
                command=code,
                volumes=volumes,
                environment=environment,
                mem_limit=self.memory_limit,
                nano_cpus=nano_cpus,
                network_disabled=self.network_disabled,
                working_dir="/workspace",
                user="sandbox",
            )
            container.start()

            try:
                exit_result = container.wait(timeout=timeout_s)
                returncode: int = exit_result.get("StatusCode", -1)
            except Exception:
                # Covers requests.exceptions.ReadTimeout and docker APIError
                try:
                    container.kill()
                except Exception:
                    pass
                return {
                    "stdout": "",
                    "stderr": f"Timeout after {timeout_s}s",
                    "returncode": -1,
                }

            stdout_bytes: bytes = container.logs(stdout=True, stderr=False)
            stderr_bytes: bytes = container.logs(stdout=False, stderr=True)
            return {
                "stdout": stdout_bytes.decode("utf-8", errors="replace"),
                "stderr": stderr_bytes.decode("utf-8", errors="replace"),
                "returncode": returncode,
            }

        except Exception as exc:
            # Catch docker.errors.ImageNotFound and docker.errors.DockerException.
            # We avoid referencing docker.errors.* directly so that this code remains
            # safe even when the docker SDK is partially mocked in tests.
            exc_type_name = type(exc).__name__
            if exc_type_name == "ImageNotFound":
                if self.auto_build:
                    try:
                        self._build_image()
                        # Retry once after building
                        return self.run(
                            code=code, output_dir=output_dir,
                            raw_data_path=raw_data_path, timeout_s=timeout_s,
                        )
                    except Exception as build_exc:
                        warnings.warn(
                            f"DockerSandbox: image build failed ({build_exc}). "
                            "Falling back to subprocess.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                else:
                    warnings.warn(
                        f"DockerSandbox: image {self.image!r} not found and auto_build=False. "
                        "Falling back to subprocess.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                return SubprocessExecutor().run(
                    code=code, output_dir=output_dir,
                    raw_data_path=raw_data_path, timeout_s=timeout_s,
                )
            elif _DOCKER_AVAILABLE and isinstance(exc, docker.errors.DockerException):
                warnings.warn(
                    f"DockerSandbox: DockerException ({exc}). Falling back to subprocess.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return SubprocessExecutor().run(
                    code=code, output_dir=output_dir,
                    raw_data_path=raw_data_path, timeout_s=timeout_s,
                )
            else:
                # Re-raise unexpected exceptions
                raise

        finally:
            if container is not None:
                try:
                    container.remove(force=True)
                except Exception:
                    pass


def compile_latex(output_dir: str, tex_file: str = "paper.tex", timeout_s: int = 120) -> dict[str, Any]:
    """Run the full pdflatex → bibtex → pdflatex → pdflatex compilation sequence.

    Args:
        output_dir: Directory containing the .tex and .bib files (used as cwd).
        tex_file:   Name of the main .tex file (no path prefix).
        timeout_s:  Timeout per individual command in seconds.

    Returns:
        dict with keys:
          ``success`` (bool), ``steps`` (list of step result dicts with
          ``cmd``, ``returncode``, ``stdout``, ``stderr``).
    """
    base = tex_file.replace(".tex", "")
    steps_results: list[dict[str, Any]] = []

    def _run(cmd: list[str]) -> dict[str, Any]:
        try:
            proc = subprocess.run(
                cmd,
                cwd=output_dir,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            return {
                "cmd": " ".join(cmd),
                "returncode": proc.returncode,
                "stdout": proc.stdout[-2000:] if proc.stdout else "",
                "stderr": proc.stderr[-2000:] if proc.stderr else "",
            }
        except FileNotFoundError:
            return {
                "cmd": " ".join(cmd),
                "returncode": -1,
                "stdout": "",
                "stderr": f"{cmd[0]!r} not found — is it installed and on PATH?",
            }
        except subprocess.TimeoutExpired:
            return {
                "cmd": " ".join(cmd),
                "returncode": -2,
                "stdout": "",
                "stderr": f"Timed out after {timeout_s}s",
            }

    pdflatex_cmd = ["pdflatex", "-interaction=nonstopmode", tex_file]

    for cmd in [
        pdflatex_cmd,
        ["bibtex", base],
        pdflatex_cmd,
        pdflatex_cmd,
    ]:
        result = _run(cmd)
        steps_results.append(result)
        # If pdflatex exits non-zero on first pass, abort early
        if result["returncode"] not in (0, 1) and cmd == pdflatex_cmd:
            break

    success = all(s["returncode"] in (0, 1) for s in steps_results)
    return {"success": success, "steps": steps_results}


def create_executor(config: dict[str, Any]) -> DockerSandbox | SubprocessExecutor:
    """Return the appropriate executor based on ``config["sandbox"]``.

    If sandbox.enabled is False (or the key is absent), returns SubprocessExecutor.
    If Docker daemon is not reachable, emits a RuntimeWarning and returns
    SubprocessExecutor.
    """
    sandbox_cfg: dict[str, Any] = config.get("sandbox", {})
    if not sandbox_cfg.get("enabled", False):
        return SubprocessExecutor()

    if not _DOCKER_AVAILABLE:
        warnings.warn(
            "sandbox.enabled is true but the docker Python SDK is not installed. "
            "Falling back to subprocess.",
            RuntimeWarning,
            stacklevel=2,
        )
        return SubprocessExecutor()

    try:
        client = docker.from_env()
        client.ping()
    except Exception as exc:
        warnings.warn(
            f"sandbox.enabled is true but Docker daemon not reachable ({exc}). "
            "Falling back to subprocess.",
            RuntimeWarning,
            stacklevel=2,
        )
        return SubprocessExecutor()

    return DockerSandbox(
        image=sandbox_cfg.get("image", "edm-ars-sandbox:latest"),
        memory_limit=sandbox_cfg.get("memory_limit", "4g"),
        cpu_count=sandbox_cfg.get("cpu_count", 2),
        network_disabled=sandbox_cfg.get("network_disabled", True),
        auto_build=sandbox_cfg.get("auto_build", True),
    )
