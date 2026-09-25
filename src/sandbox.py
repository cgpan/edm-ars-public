"""Sandboxed code execution for EDM-ARS.

Provides two executor implementations:
- SubprocessExecutor: bare subprocess.run() (fast, no isolation)
- DockerSandbox: Docker-container-based execution (isolated, resource-limited)

Use create_executor(config) to get the appropriate executor based on config.yaml.
"""
from __future__ import annotations

import os
import re
import pathlib
import subprocess
import sys
import warnings
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

try:
    import docker  # type: ignore[import-not-found]
    import docker.errors  # type: ignore[import-not-found]
    import requests.exceptions  # type: ignore[import-not-found]
    _DOCKER_AVAILABLE = True
except ImportError:
    _DOCKER_AVAILABLE = False
    docker = None  # type: ignore[assignment]



#: Constructs that fail on at least one supported host, mapped to what to
#: do instead. Each entry earned its place by breaking a real run.
UNPORTABLE_CONSTRUCTS: dict[str, str] = {
    "signal.SIGALRM": (
        "signal.SIGALRM is Unix-only and raises AttributeError on Windows. "
        "The executor already enforces a hard timeout, so generated code "
        "must not set its own."
    ),
    "signal.alarm": (
        "signal.alarm is Unix-only. The executor already enforces a hard "
        "timeout; remove the in-code timeout entirely."
    ),
    "signal.setitimer": (
        "signal.setitimer is Unix-only. The executor enforces the timeout."
    ),
    "os.fork": (
        "os.fork is unavailable on Windows. Use joblib or "
        "concurrent.futures if parallelism is genuinely needed."
    ),
    "resource.setrlimit": (
        "the resource module is Unix-only. Memory limits are enforced by "
        "the sandbox, not by generated code."
    ),
}


#: Patterns that run without error but do something other than what the
#: surrounding code claims. Unlike UNPORTABLE_CONSTRUCTS these do not
#: crash -- that is exactly why they need a machine to catch them.
SILENT_MISBEHAVIOUR = [
    (
        # IterativeImputer models each feature from the OTHERS. Given one
        # column there are no others, so it degenerates to the column
        # mean -- while the surrounding code records the method as
        # "IterativeImputer" and the manuscript repeats it. Five papers
        # named a multivariate imputer and performed mean-fill on
        # variables missing 28-36%.
        re.compile(
            r"IterativeImputer\([^)]*\)[\s\S]{0,400}?\.fit_transform\(\s*"
            r"[A-Za-z_][A-Za-z0-9_]*\s*\[\s*\[\s*[A-Za-z_'\"]"
        ),
        "IterativeImputer appears to be fitted on a single-column "
        "selection (df[[col]]). With one column it has no other features "
        "to model from and silently degenerates to MEAN imputation, while "
        "the recorded imputation_method still says IterativeImputer. Fit "
        "it once on the full numeric predictor block (training rows only), "
        "or record the method that actually ran.",
    ),
]


#: BLAS/OpenMP thread-pool controls, read once at import time by the numeric
#: backends. Setting them after numpy is imported is a no-op, which is why
#: they are injected into the CHILD process environment rather than set here.
BLAS_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

#: Threads per worker process. 2 is deliberate, not tuned: the generated
#: analysis code fits models with n_jobs=-1, so scikit-learn already spawns
#: one worker per core. Leaving the inner pools uncapped means each of those
#: workers spawns its own core-count of BLAS threads on top.
DEFAULT_INNER_THREADS = 2


#: Environment variable NAMES that say they hold a credential. Generated
#: code never needs one: it makes no network calls and talks to no
#: provider. What it can do is print os.environ while debugging, and its
#: stdout/stderr go into the retry prompt sent to the provider and into
#: prompts/<agent>/.../rendered_prompt.txt in the run folder.
#:
#: A denylist, not an allowlist, on purpose: Python needs SYSTEMROOT on
#: Windows, R needs R_HOME/R_LIBS*, the bridge needs EDM_ARS_RSCRIPT and
#: EDM_ARS_R_HELPERS, and conda/venv activation leaves a dozen more. None
#: of those names matches below.
_SECRET_NAME = re.compile(
    r"(?i)(?:API_?KEY|ACCESS_?KEY|PRIVATE_?KEY|SECRET|PASSWORD|PASSWD"
    r"|CREDENTIALS?|CONNECTION_?STRING"
    r"|(?:^|_)TOKENS?(?:_|$)|(?:^|_)PAT$|(?:^|_)KEYS?$)"
)


def is_secret_name(name: str) -> bool:
    """True when an environment variable name marks it as a credential."""
    return bool(_SECRET_NAME.search(name))


def scrub_secrets(env: Mapping[str, str]) -> dict[str, str]:
    """Return a copy of *env* without the credential-named variables."""
    return {k: v for k, v in env.items() if not is_secret_name(k)}


def blas_thread_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Cap the inner BLAS/OpenMP pools for generated analysis code.

    Observed on a 28-core Windows host: the generated script fits with
    n_jobs=-1, scikit-learn spawns one worker per core, and each worker's
    BLAS spawns its own threads inside that. 32 processes contended for 28
    cores and the battery ran SLOWER than a capped run -- so the analysis
    stage hit its execution timeout and burned all three retries, 21
    minutes apart, without ever being genuinely compute-bound.

    Capping the inner pools leaves n_jobs=-1 parallelising ACROSS models,
    which is the level that actually helps, and stops the nesting.

    This also matters under Docker: nano_cpus caps the container's CPU
    quota, but scikit-learn still reads the host core count and sizes its
    pool from that, so it oversubscribes inside the quota.

    An operator who has already set any of these vars keeps their value.

    Credential-named variables are dropped (see ``_SECRET_NAME``) whether
    the base is the host environment or an explicit mapping, so no caller
    can hand a key to generated code by accident.
    """
    env = scrub_secrets(base if base is not None else os.environ)
    threads = os.environ.get("EDMARS_INNER_THREADS", str(DEFAULT_INNER_THREADS))
    for var in BLAS_THREAD_VARS:
        env.setdefault(var, threads)
    return env


def child_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment LLM-generated code runs with.

    ``blas_thread_env`` (credentials dropped, inner thread pools capped)
    plus one thing the executor relies on:

    * UTF-8 stdio. On a Windows host the child's stdout defaults to the
      ANSI code page, so a generated script died with UnicodeEncodeError
      at its first ``print`` of a check mark, an arrow or a Greek letter --
      usually at the very end of a long run, costing a retry. The executor
      decodes the pipes as UTF-8, so the child must write UTF-8; these are
      forced, not defaulted, because a stray PYTHONIOENCODING=cp1252 in
      the host env would bring the crash back.

    Rscript, started by the bridge from inside the child, inherits all of
    this -- including the missing credentials.
    """
    env = blas_thread_env(dict(base) if base is not None else None)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def check_silent_misbehaviour(code: str) -> list[str]:
    """Return a message for each pattern that misreports what it does.

    Separate from portability because the failure mode is the opposite:
    these run cleanly and produce plausible output. Nothing downstream
    can notice, which is why the check has to happen before execution.
    """
    return [msg for pattern, msg in SILENT_MISBEHAVIOUR if pattern.search(code)]


def check_portability(code: str) -> list[str]:
    """Return a message for every unportable construct found in *code*.

    Empty list means nothing known-unportable was found. This is a
    substring scan, not a parse: the point is to catch the handful of
    idioms that have actually broken runs, cheaply, before execution.
    """
    findings: list[str] = []
    for construct, guidance in UNPORTABLE_CONSTRUCTS.items():
        if construct in code:
            findings.append(f"{construct}: {guidance}")
    return findings


#: Returned when the interpreter itself cannot be started -- the shell's
#: "command not found" code, so it cannot be mistaken for a script error.
INTERPRETER_NOT_STARTED = 127


class SubprocessExecutor:
    """Execute LLM-generated code via bare subprocess.run().

    Mirrors the interface of DockerSandbox.run() so the two are interchangeable.

    The script runs under the SAME interpreter as the pipeline
    (``sys.executable``), not whatever ``python`` the OS finds first. On
    Windows a venv's python.exe is a redirector that starts the base
    install, and CreateProcess searches the running image's own directory
    before PATH, so a bare ``python`` resolved to the base interpreter even
    inside an activated venv -- one without the packages the README had
    just installed. On macOS a bare ``python`` often does not exist at all.
    ``python_executable`` (config ``sandbox.python_executable``) overrides
    the choice for operators who deliberately run analysis code elsewhere.
    """

    def __init__(self, python_executable: str | None = None) -> None:
        self.python_executable: str | None = (
            os.path.expanduser(os.path.expandvars(str(python_executable)))
            if python_executable else None
        )

    def interpreter(self) -> str:
        """The Python the generated script will run under."""
        return self.python_executable or sys.executable or "python"

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
        dict with keys: stdout, stderr, returncode. stdout and stderr are
        always ``str``; returncode is 127 (``INTERPRETER_NOT_STARTED``)
        when the interpreter could not be started at all.
        """
        # Write code to a temp file instead of passing via -c to avoid
        # Windows command-line length limit (WinError 206, ~32k char cap).
        problems = check_portability(code)
        if problems:
            # Fail loudly and early rather than letting each model raise
            # AttributeError separately into results.errors, where the
            # Analyst files them away and the pipeline writes a paper
            # about an analysis that never ran.
            detail = "; ".join(problems)
            return {
                "stdout": "",
                "stderr": (
                    "PORTABILITY CHECK FAILED before execution. "
                    f"{detail} Rewrite the code without it and retry."
                ),
                "returncode": 2,
            }

        misbehaviour = check_silent_misbehaviour(code)
        if misbehaviour:
            # Blocked rather than warned. This code runs cleanly and
            # produces plausible output while doing something other than
            # what it records, so letting it proceed means a manuscript
            # that misstates its own methods -- and nothing downstream can
            # tell. The stage has a retry path; a clear message there is
            # cheaper than a wrong paper.
            detail = " ".join(misbehaviour)
            return {
                "stdout": "",
                "stderr": (
                    "SILENT-MISBEHAVIOUR CHECK FAILED before execution. "
                    f"{detail}"
                ),
                "returncode": 3,
            }

        script_path = os.path.join(output_dir, "_generated_script.py")
        exe = self.interpreter()
        try:
            with open(script_path, "w", encoding="utf-8") as fh:
                fh.write(code)
            try:
                result = subprocess.run(
                    [exe, script_path],
                    capture_output=True,
                    # UTF-8 with replacement, never the locale codec: a
                    # byte the ANSI code page cannot decode killed the
                    # reader thread and came back as stdout=None, which the
                    # agents' retry prompts then sliced into a TypeError.
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=timeout_s,
                    cwd=output_dir,
                    env=child_env(),
                )
            except OSError as exc:
                # Only the interpreter launch lands here: writing the
                # script above is outside this try on purpose, so a bad
                # output_dir still raises as it always did.
                return {
                    "stdout": "",
                    "stderr": (
                        f"Could not start the Python interpreter {exe!r}: "
                        f"{exc}. Set sandbox.python_executable in config.yaml "
                        "to a Python that has this project's requirements "
                        "installed, or leave it unset to use the interpreter "
                        "running the pipeline."
                    ),
                    "returncode": INTERPRETER_NOT_STARTED,
                }
            return {
                "stdout": result.stdout or "",
                "stderr": result.stderr or "",
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
        python_executable: str | None = None,
    ) -> None:
        self.image = image
        self.memory_limit = memory_limit
        self.cpu_count = cpu_count
        self.network_disabled = network_disabled
        self.auto_build = auto_build
        self._client: Any = None  # lazy-initialised on first run()
        # Every "fall back to subprocess" path uses this one, so a fallback
        # runs under the configured interpreter, not the default one.
        self._fallback = SubprocessExecutor(python_executable=python_executable)

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
            return self._fallback.run(
                code=code, output_dir=output_dir,
                raw_data_path=raw_data_path, timeout_s=timeout_s,
            )

        # Build volume-mount dict
        volumes: dict[str, dict[str, str]] = {
            os.path.abspath(output_dir): {"bind": "/workspace", "mode": "rw"},
        }
        # child_env with an explicit base: the container gets only
        # OUTPUT_DIR plus the thread caps and UTF-8 stdio, never a copy of
        # the host env.
        environment: dict[str, str] = child_env({"OUTPUT_DIR": "/workspace"})

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
                return self._fallback.run(
                    code=code, output_dir=output_dir,
                    raw_data_path=raw_data_path, timeout_s=timeout_s,
                )
            elif _DOCKER_AVAILABLE and isinstance(exc, docker.errors.DockerException):
                warnings.warn(
                    f"DockerSandbox: DockerException ({exc}). Falling back to subprocess.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return self._fallback.run(
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

    # V4 wave-2: apa7 journal manuscripts use biblatex/biber
    # (\addbibresource + \printbibliography); ACM conference papers use
    # bibtex. Pick the bibliography engine from the tex source.
    bib_engine = ["bibtex", base]
    try:
        with open(os.path.join(output_dir, tex_file), encoding="utf-8") as f:
            _tex_src = f.read()
        if "biblatex" in _tex_src or "\\addbibresource" in _tex_src:
            bib_engine = ["biber", base]
    except OSError:
        pass

    for cmd in [
        pdflatex_cmd,
        bib_engine,
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

    Every executor returned -- including the subprocess one a Docker
    sandbox falls back to -- carries ``sandbox.python_executable`` (null =
    the interpreter running the pipeline).
    """
    sandbox_cfg: dict[str, Any] = config.get("sandbox") or {}
    python_executable = sandbox_cfg.get("python_executable") or None

    def _subprocess() -> SubprocessExecutor:
        return SubprocessExecutor(python_executable=python_executable)

    if not sandbox_cfg.get("enabled", False):
        return _subprocess()

    if not _DOCKER_AVAILABLE:
        warnings.warn(
            "sandbox.enabled is true but the docker Python SDK is not installed. "
            "Falling back to subprocess.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _subprocess()

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
        return _subprocess()

    return DockerSandbox(
        image=sandbox_cfg.get("image", "edm-ars-sandbox:latest"),
        memory_limit=sandbox_cfg.get("memory_limit", "4g"),
        cpu_count=sandbox_cfg.get("cpu_count", 2),
        network_disabled=sandbox_cfg.get("network_disabled", True),
        auto_build=sandbox_cfg.get("auto_build", True),
        python_executable=python_executable,
    )
