"""V4 psychometrics — deterministic R bridge.

Runs FIXED, certified R scripts from ``r_helpers/`` via ``Rscript
--vanilla`` with JSON files for input and output. Design rules:

- Generated (LLM) code never writes raw R; it calls
  ``analysis_helpers.psy_*`` wrappers, which call :func:`run_r_script`
  with a script NAME from the certified set. Passing a path outside
  ``r_helpers/`` raises.
- No rpy2: subprocess + JSON is version-robust and matches the
  executor model (and rpy2 is fragile on Windows).
- R is not installed in the Docker sandbox image; psychometrics runs
  use the subprocess executor (the current default). ``ensure_r()``
  fails loudly with remediation text when R is missing.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

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

# Resolution order: env var -> config-style explicit path (passed by
# caller) -> common Windows install locations -> PATH.
_COMMON_RSCRIPT_PATHS = [
    r"C:\Program Files\R\R-4.4.1\bin\Rscript.exe",
    r"C:\Program Files\R\R-4.4.2\bin\Rscript.exe",
    r"C:\Program Files\R\R-4.5.0\bin\Rscript.exe",
]


class RBridgeError(RuntimeError):
    """Raised when the R bridge cannot run or the script fails."""


def find_rscript(explicit_path: str | None = None) -> str:
    """Locate Rscript. Raises RBridgeError with remediation if absent."""
    candidates: list[str] = []
    if explicit_path:
        candidates.append(explicit_path)
    env = os.environ.get("EDM_ARS_RSCRIPT")
    if env:
        candidates.append(env)
    candidates += _COMMON_RSCRIPT_PATHS
    for c in candidates:
        if c and Path(c).exists():
            return c
    from shutil import which

    on_path = which("Rscript")
    if on_path:
        return on_path
    raise RBridgeError(
        "Rscript not found. Install R (>= 4.4) or set EDM_ARS_RSCRIPT / "
        "config r_bridge.rscript_path to the Rscript executable. "
        f"Checked: {candidates}"
    )


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
        proc = subprocess.run(
            [rscript, "--vanilla", str(script), str(in_path), str(out_path)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if proc.returncode != 0:
            raise RBridgeError(
                f"R script {script_name} exited {proc.returncode}. "
                f"stderr (tail): {proc.stderr[-2000:]}"
            )
        if not out_path.exists():
            raise RBridgeError(
                f"R script {script_name} wrote no output JSON. "
                f"stdout (tail): {proc.stdout[-1000:]}"
            )
        result = json.loads(out_path.read_text(encoding="utf-8"))
    if isinstance(result, dict) and result.get("error"):
        raise RBridgeError(f"R script {script_name} error: {result['error']}")
    return result
