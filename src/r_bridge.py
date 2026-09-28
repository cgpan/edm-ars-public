"""V4 psychometrics — deterministic R bridge.

Runs FIXED, certified R scripts from ``r_helpers/`` via ``Rscript
--vanilla`` with JSON files for input and output. Design rules:

- Generated (LLM) code never writes raw R; it calls
  ``analysis_helpers.psy_*`` wrappers, which call :func:`run_r_script`
  with a script NAME from the certified set. Passing a path outside
  ``r_helpers/`` raises.
- No rpy2: subprocess + JSON is version-robust and matches the
  executor model (and rpy2 is fragile on Windows).
- R is not installed in the Docker sandbox image, and ``r_helpers/`` is
  not mounted there, so psychometrics runs need the subprocess executor
  (``sandbox.enabled: false``).
- R is located lazily: :func:`find_rscript` runs the first time a
  ``psy_*`` wrapper calls :func:`run_r_script`. :func:`missing_r_packages`
  is the up-front check a preflight (or a test gate) can call. Both raise
  :class:`RBridgeError` with remediation text when R is unusable.
- Standard library only: this file is copied flat into run output dirs
  and imported from there by generated code.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from shutil import which

def _find_r_helpers_dir() -> Path:
    """Locate r_helpers/ robustly.

    This module is COPIED into run output dirs alongside
    analysis_helpers.py (generated code imports it flat), so
    __file__-relative resolution alone is not enough. Order:
    env EDM_ARS_R_HELPERS -> package-relative (src/..) -> upward walk
    from __file__ and from cwd (output dirs live a few levels below
    the project root).
    """
    env = os.environ.get("EDM_ARS_R_HELPERS")
    if env and Path(env).is_dir():
        return Path(env)
    pkg = Path(__file__).resolve().parent.parent / "r_helpers"
    if pkg.is_dir():
        return pkg
    for base in (Path(__file__).resolve().parent, Path.cwd()):
        cur = base
        for _ in range(6):
            cand = cur / "r_helpers"
            if cand.is_dir():
                return cand
            cur = cur.parent
    return pkg  # nonexistent; run_r_script raises a clear error


R_HELPERS_DIR = _find_r_helpers_dir()

#: Packages the certified scripts load. jsonlite is used by every one of
#: them (outside their tryCatch) and is NOT part of base R; MASS ships
#: with R but is listed so a broken library is still caught.
REQUIRED_R_PACKAGES: tuple[str, ...] = ("jsonlite", "lavaan", "mirt", "CDM", "MASS")

#: Checked on macOS/Linux after PATH, in this order. A GUI-launched
#: process on macOS often lacks /usr/local/bin and /opt/homebrew/bin on
#: PATH even though R is installed there.
_POSIX_RSCRIPT_PATHS: tuple[str, ...] = (
    "/Library/Frameworks/R.framework/Resources/bin/Rscript",  # CRAN .pkg
    "/opt/homebrew/bin/Rscript",  # Homebrew, Apple silicon
    "/usr/local/bin/Rscript",  # Homebrew (Intel), CRAN symlink
    "/usr/bin/Rscript",  # Linux distribution packages
)

_R_VERSION_DIR = re.compile(r"^R-(\d+)\.(\d+)(?:\.(\d+))?")
_R_PACKAGE_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9.]*$")
#: R's message for a package that is not installed. The quotes are ASCII
#: from Rscript on Windows but may be typographic in a UTF-8 locale.
_NO_PACKAGE = re.compile(
    r"there is no package called [\u2018\u2019'\"`]?([A-Za-z][A-Za-z0-9.]*)"
)


class RBridgeError(RuntimeError):
    """Raised when the R bridge cannot run or the script fails."""


def _rscript_exe_name() -> str:
    return "Rscript.exe" if os.name == "nt" else "Rscript"


def _resolve_override(value: str, source: str) -> str:
    """Turn a user-supplied Rscript setting into an executable path.

    Forgiving about the shapes people actually paste -- surrounding
    quotes, ``~`` or ``%VARS%``, R's ``bin`` folder, the R home folder, a
    Windows path without ``.exe``, a bare command name on PATH -- and
    strict about everything else: a setting that resolves to nothing
    raises instead of falling through to some other R. Falling through
    made a typo look like a working override while a different, possibly
    package-less, R ran; pointing at the ``bin`` folder used to reach
    subprocess and fail with a bare "Access is denied".
    """
    raw = value.strip().strip('"').strip("'").strip()
    expanded = os.path.expanduser(os.path.expandvars(raw))
    exe = _rscript_exe_name()
    p = Path(expanded)
    if p.is_file():
        return str(p)
    if p.is_dir():
        for cand in (p / exe, p / "bin" / exe):
            if cand.is_file():
                return str(cand)
        raise RBridgeError(
            f"{source} points at the folder {raw!r}, which has no {exe} "
            f"(looked in it and in its bin subfolder). Point it at the "
            f"{exe} executable itself, e.g. <R home>/bin/{exe}."
        )
    if os.name == "nt" and not p.suffix and Path(expanded + ".exe").is_file():
        return expanded + ".exe"
    if "/" not in raw and "\\" not in raw:
        on_path = which(expanded)
        if on_path:
            return on_path
    raise RBridgeError(
        f"{source} is set to {raw!r}, but no Rscript exists there. Fix or "
        "unset it. No other R was tried, so the one you configured is never "
        "silently replaced by a different one."
    )


def _version_key(dirname: str) -> tuple[int, int, int]:
    m = _R_VERSION_DIR.match(dirname)
    if not m:
        return (-1, -1, -1)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3) or 0))


def _newest_first(roots: Iterable[Path], exe_name: str) -> list[str]:
    """``<root>/R-*/bin/<exe_name>`` under every root, newest R first.

    Versions compare as numbers (R-4.10.0 beats R-4.9.3); on a tie the
    earlier root wins. Folders not named R-<version> sort last.
    """
    found: list[tuple[tuple[int, int, int], int, str]] = []
    for idx, root in enumerate(roots):
        try:
            children = sorted(root.glob("R-*"))
        except OSError:
            continue
        for child in children:
            cand = child / "bin" / exe_name
            if cand.is_file():
                found.append((_version_key(child.name), idx, str(cand)))
    found.sort(key=lambda t: (tuple(-v for v in t[0]), t[1], t[2]))
    return [path for _, _, path in found]


def _windows_r_roots() -> list[Path]:
    """Where the CRAN installer puts R on Windows: Program Files (admin
    install) and %LOCALAPPDATA%/Programs (per-user install, no admin)."""
    roots: list[Path] = []
    for var in ("ProgramFiles", "ProgramW6432"):
        val = os.environ.get(var)
        if val:
            roots.append(Path(val) / "R")
    if not roots:
        roots.append(Path("C:/Program Files/R"))
    local = os.environ.get("LOCALAPPDATA")
    if local:
        roots.append(Path(local) / "Programs" / "R")
    unique: list[Path] = []
    for r in roots:
        if r not in unique:
            unique.append(r)
    return unique


def _discovered_installs() -> list[str]:
    """Rscript executables in the standard install locations, best first."""
    if os.name == "nt":
        return _newest_first(_windows_r_roots(), "Rscript.exe")
    return [p for p in _POSIX_RSCRIPT_PATHS if Path(p).is_file()]


def find_rscript(explicit_path: str | None = None) -> str:
    """Locate Rscript. Raises RBridgeError with remediation if absent.

    Resolution order:

    1. ``explicit_path`` (the ``rscript_path`` argument);
    2. the ``EDM_ARS_RSCRIPT`` environment variable. Pipeline runs also
       receive config ``r_bridge.rscript_path`` this way: the subprocess
       executor exports it to generated code as EDM_ARS_RSCRIPT unless
       the operator has set the variable themselves;
    3. ``Rscript`` on PATH;
    4. the newest R in the standard install folders -- on Windows
       ``Program Files/R/R-*`` and ``%LOCALAPPDATA%/Programs/R/R-*`` (the
       CRAN installer does not put R on PATH); on macOS/Linux the
       R.framework, Homebrew and distribution locations.

    An override (1 or 2) that does not resolve to an Rscript raises
    instead of falling through; one that points at R's ``bin`` (or home)
    folder is resolved to the executable inside it.
    """
    if explicit_path:
        return _resolve_override(explicit_path, "rscript_path")
    env = os.environ.get("EDM_ARS_RSCRIPT")
    if env and env.strip():
        return _resolve_override(env, "EDM_ARS_RSCRIPT")
    on_path = which("Rscript")
    if on_path:
        return on_path
    installs = _discovered_installs()
    if installs:
        return installs[0]
    looked = (
        [str(r) for r in _windows_r_roots()]
        if os.name == "nt" else list(_POSIX_RSCRIPT_PATHS)
    )
    raise RBridgeError(
        "Rscript not found. Install R (>= 4.4) from https://cran.r-project.org, "
        "or set EDM_ARS_RSCRIPT (for pipeline runs, config.yaml "
        "r_bridge.rscript_path) to the Rscript executable. "
        f"Looked on PATH and in: {looked}"
    )


def _run_rscript(argv: list[str], timeout_s: int) -> subprocess.CompletedProcess[str]:
    """The bridge's one process launch.

    Output is decoded as UTF-8 with replacement: R >= 4.2 on Windows
    writes UTF-8, and the locale codec either garbled it or, on a byte it
    could not decode, silently returned no output at all.
    """
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_s,
    )


def _missing_package_hint(text: str, rscript: str) -> str:
    """Remediation for R's "there is no package called 'X'", else ''."""
    m = _NO_PACKAGE.search(text or "")
    if not m:
        return ""
    pkg = m.group(1)
    return (
        f" Remediation: the R at {rscript} does not have {pkg}. Install it "
        f'there (in that R: install.packages("{pkg}")), or point '
        "EDM_ARS_RSCRIPT at an R that already has it."
    )


def missing_r_packages(
    packages: Sequence[str] = REQUIRED_R_PACKAGES,
    rscript_path: str | None = None,
    timeout_s: int = 120,
) -> list[str]:
    """Return the packages in *packages* that this R cannot load.

    Uses base R only (``requireNamespace``), so it works even when
    jsonlite itself is missing. Raises RBridgeError when Rscript cannot be
    found or started, or when the check does not run to completion.
    """
    names = list(packages)
    bad = [p for p in names if not _R_PACKAGE_NAME.match(p)]
    if bad:
        raise ValueError(f"not R package names: {bad}")
    rscript = find_rscript(rscript_path)
    if not names:
        return []
    quoted = ", ".join(f"'{p}'" for p in names)
    # Single quotes and no backslashes: Rscript.exe re-quotes its -e
    # argument when it starts R on Windows, and escapes do not survive.
    expr = (
        f"for (p in c({quoted})) "
        "if (!requireNamespace(p, quietly = TRUE)) writeLines(paste('MISSING', p)); "
        "writeLines('EDM_ARS_R_CHECK_DONE')"
    )
    try:
        proc = _run_rscript([rscript, "--vanilla", "-e", expr], timeout_s)
    except OSError as exc:
        raise RBridgeError(f"Could not start Rscript at {rscript!r}: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RBridgeError(
            f"R package check with {rscript!r} timed out after {timeout_s}s"
        ) from exc
    out = proc.stdout or ""
    if proc.returncode != 0 or "EDM_ARS_R_CHECK_DONE" not in out:
        raise RBridgeError(
            f"R package check with {rscript!r} did not complete "
            f"(exit {proc.returncode}). stderr (tail): "
            f"{(proc.stderr or '')[-1000:]}"
        )
    return [
        line.split(None, 1)[1].strip()
        for line in out.splitlines()
        if line.startswith("MISSING ")
    ]


def run_r_script(
    script_name: str,
    payload: dict,
    timeout_s: int = 600,
    rscript_path: str | None = None,
) -> dict:
    """Run a certified r_helpers script with a JSON payload.

    The script receives two argv entries: input JSON path and output
    JSON path. It must write a JSON object to the output path; a
    top-level ``"error"`` key marks failure.
    """
    if "/" in script_name or "\\" in script_name or ".." in script_name:
        raise RBridgeError(
            f"script_name must be a bare name inside r_helpers/, got "
            f"{script_name!r}"
        )
    script = R_HELPERS_DIR / script_name
    if not script.exists():
        raise RBridgeError(
            f"Certified R helper not found: {script}. Available: "
            f"{sorted(p.name for p in R_HELPERS_DIR.glob('*.R'))}"
        )
    rscript = find_rscript(rscript_path)

    with tempfile.TemporaryDirectory(prefix="edm_ars_r_") as td:
        in_path = Path(td) / "in.json"
        out_path = Path(td) / "out.json"
        in_path.write_text(json.dumps(payload), encoding="utf-8")
        try:
            proc = _run_rscript(
                [rscript, "--vanilla", str(script), str(in_path), str(out_path)],
                timeout_s,
            )
        except OSError as exc:
            raise RBridgeError(
                f"Could not start Rscript at {rscript!r}: {exc}"
            ) from exc
        stderr = proc.stderr or ""
        stdout = proc.stdout or ""
        if proc.returncode != 0:
            raise RBridgeError(
                f"R script {script_name} exited {proc.returncode}. "
                f"stderr (tail): {stderr[-2000:]}"
                + _missing_package_hint(stderr, rscript)
            )
        if not out_path.exists():
            raise RBridgeError(
                f"R script {script_name} wrote no output JSON. "
                f"stdout (tail): {stdout[-1000:]}"
            )
        result = json.loads(out_path.read_text(encoding="utf-8"))
    if isinstance(result, dict) and result.get("error"):
        raise RBridgeError(
            f"R script {script_name} error: {result['error']}"
            + _missing_package_hint(str(result["error"]), rscript)
        )
    return result
