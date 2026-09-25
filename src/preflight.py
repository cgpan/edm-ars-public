"""Checks that a run can finish, made before it spends anything.

Every item here used to surface only after paid stages had run: a missing
API key as a traceback from agent construction, a missing data file from
inside LLM-written DataEngineer code after the literature search and the
ProblemFormulator, a missing R package inside the Analyst's repair loop,
a missing pdflatex as a run that finished INCOMPLETE with no PDF.

:func:`check_run_prerequisites` returns findings; it never raises for a
failed check and never prints. ``fail`` means a real run would not get
far enough to be worth starting, so ``src.main`` stops before any LLM
call. ``warn`` means the run can finish but something the user probably
wants (a PDF, the review gate) will be missing.

No subprocess is started here. The R package probe goes through
``src.r_bridge.run_r_script`` -- the same ``Rscript --vanilla`` call the
certified helpers use -- with ``r_helpers/preflight_packages.R``.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path
from typing import NamedTuple

from src.config import PROJECT_ROOT


class Finding(NamedTuple):
    """One pre-flight result: ``(code, severity, message, fix)``."""

    code: str
    severity: str  # "fail" | "warn"
    message: str
    fix: str


FAIL = "fail"
WARN = "warn"

#: Environment variable each provider reads its key from (see
#: ``BaseAgent.__init__``).
PROVIDER_KEY_ENV: dict[str, str] = {
    "deepseek": "DEEPSEEK_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "minimax": "MINIMAX_API_KEY",
}

#: Python SDK module each provider's client is built from.
_PROVIDER_SDK: dict[str, str] = {
    "deepseek": "openai",
    "openai": "openai",
    "anthropic": "anthropic",
    "minimax": "anthropic",
}

#: The stages that always call an LLM.
_CORE_STAGES = ("problem_formulator", "data_engineer", "analyst", "critic", "writer")

#: R packages each psychometrics method loads through its certified helper
#: (r_helpers/*.R). P1 (CTT) is computed in Python; P2 (omega) reads the
#: P3 CFA result. jsonlite is needed by every helper to read its input.
_R_PACKAGES_BY_METHOD: dict[str, tuple[str, ...]] = {
    "P1": (),
    "P2": ("lavaan",),
    "P3": ("lavaan",),
    "P4": ("mirt",),
    "P5": ("MASS",),
    "P6": ("lavaan",),
    "P7": ("CDM",),
}
_ALL_R_PACKAGES = ("jsonlite", "lavaan", "mirt", "CDM", "MASS")

#: Where each dataset comes from, for the missing-data message.
_DATA_SOURCES: dict[str, str] = {
    "hsls09_public": (
        "Download the HSLS:09 public-use student file (CSV, labelled "
        "values) from https://nces.ed.gov/surveys/hsls09/"
    ),
    "els_2002": (
        "Download the ELS:2002 public-use BY-F3 student file as CSV from "
        "https://nces.ed.gov/surveys/els2002/ (EDAT export)"
    ),
    "assistments_0910": (
        "Download the ASSISTments 2009-10 skill-builder CSV from "
        "https://sites.google.com/site/assistmentsdata/"
    ),
    "did_els_hsls_panel": (
        "This file is built, not downloaded: put the HSLS:09 and ELS:2002 "
        "files in place, then run: python scripts/harmonize_els_hsls.py"
    ),
}


def check_run_prerequisites(
    config: dict,
    task_type: str,
    dataset: str,
    raw_data_path: str,
    review_gate_enabled: bool,
    *,
    locked_spec: dict | None = None,
    probe_r_packages: bool = True,
) -> list[Finding]:
    """Return every pre-flight finding for a run with these settings.

    ``locked_spec`` narrows the R packages a psychometrics run needs to
    the methods its ``method_battery`` declares (all of them when it
    declares none). ``probe_r_packages=False`` skips starting R (tests).
    """
    findings: list[Finding] = []
    findings += _check_provider_keys(config)
    findings += _check_install(config, dataset)
    findings += _check_raw_data(dataset, raw_data_path)
    findings += _check_latex(config)
    if task_type == "psychometrics":
        findings += _check_r(config, locked_spec, probe_r_packages)
    if review_gate_enabled:
        findings += _check_lsar(config)
    return findings


def has_failures(findings: list[Finding]) -> bool:
    return any(f.severity == FAIL for f in findings)


# ---------------------------------------------------------------------------
# Provider keys
# ---------------------------------------------------------------------------


def llm_stages(config: dict) -> list[str]:
    """The agent keys a run with this config will call an LLM for."""
    stages = list(_CORE_STAGES)
    if (config.get("writer") or {}).get("outline_first", True):
        stages.append("outline_agent")
    if (config.get("verification") or {}).get("judge_enabled", False):
        stages.append("verifier")
    return stages


def _set_key_fix(env_var: str) -> str:
    return (
        f"Put the line {env_var}=<your key> in a file named .env in the "
        f"repository folder, or set it in the shell that starts the run "
        f"(PowerShell: $env:{env_var} = \"<your key>\"; macOS/Linux: "
        f"export {env_var}=<your key>)."
    )


def _check_provider_keys(config: dict) -> list[Finding]:
    from src.agents.provider_resolver import (
        ProviderConfigError,
        resolve_provider_for_stage,
    )

    findings: list[Finding] = []
    needed: dict[str, list[str]] = {}
    for stage in llm_stages(config):
        try:
            provider = resolve_provider_for_stage(stage, config).name
        except ProviderConfigError as exc:
            findings.append(Finding(
                "PROVIDER_CONFIG_INVALID", FAIL,
                f"The AI provider setting for {stage} is not valid: {exc}",
                "Fix llm_provider / per_stage_providers in the config file.",
            ))
            continue
        needed.setdefault(provider, []).append(stage)

    for provider, stages in needed.items():
        env_var = PROVIDER_KEY_ENV.get(provider, "ANTHROPIC_API_KEY")
        if not os.environ.get(env_var):
            findings.append(Finding(
                "KEY_MISSING", FAIL,
                f"{env_var} is not set; the {provider} provider needs it "
                f"for: {', '.join(stages)}.",
                _set_key_fix(env_var),
            ))
        sdk = _PROVIDER_SDK.get(provider)
        if sdk and importlib.util.find_spec(sdk) is None:
            findings.append(Finding(
                "SDK_MISSING", FAIL,
                f"The {provider} provider needs the Python package "
                f"'{sdk}', which is not installed.",
                f"Install it: {Path(sys.executable).name} -m pip install {sdk}",
            ))
    return findings


# ---------------------------------------------------------------------------
# Installation (prompts, registry)
# ---------------------------------------------------------------------------


def _check_install(config: dict, dataset: str) -> list[Finding]:
    paths = config.get("paths") or {}
    findings: list[Finding] = []

    prompts_dir = paths.get("agent_prompts") or ""
    missing_prompts = [
        name for name in _CORE_STAGES
        if not os.path.isfile(os.path.join(prompts_dir, f"{name}.yaml"))
    ]
    if missing_prompts:
        findings.append(Finding(
            "INSTALL_INCOMPLETE", FAIL,
            f"Agent prompt files are missing from {prompts_dir or '(unset)'}: "
            f"{', '.join(m + '.yaml' for m in missing_prompts)}. Every agent "
            "would run on a one-line placeholder prompt.",
            "Point paths.agent_prompts in the config at the repository's "
            "agent_prompts/ folder, or re-download the repository.",
        ))

    registry = os.path.join(
        paths.get("data_registry") or "", "datasets", f"{dataset}.yaml"
    )
    if not os.path.isfile(registry):
        findings.append(Finding(
            "INSTALL_INCOMPLETE", FAIL,
            f"The variable registry for dataset {dataset!r} was not found "
            f"at {registry}.",
            "Point paths.data_registry in the config at the repository's "
            "data_registry/ folder.",
        ))

    skills = PROJECT_ROOT / "skills"
    if not skills.is_dir() or next(skills.rglob("SKILL.md"), None) is None:
        findings.append(Finding(
            "INSTALL_INCOMPLETE", FAIL,
            f"No skills were found under {skills}.",
            "Re-download the repository; the skills/ folder is part of it.",
        ))
    return findings


# ---------------------------------------------------------------------------
# Raw data
# ---------------------------------------------------------------------------


def _check_raw_data(dataset: str, raw_data_path: str) -> list[Finding]:
    if os.path.isfile(raw_data_path):
        return []
    where = Path(raw_data_path)
    source = _DATA_SOURCES.get(dataset, "Obtain the file from the dataset's source")
    fix = f"{source} and save it as {where}"
    if dataset == "did_els_hsls_panel":
        fix = f"{source} (it writes {where})."
    else:
        fix += " (exactly this file name)."
    nearby = _csv_files_near(where)
    if nearby:
        fix += f" CSV files already in {where.parent}: {', '.join(nearby)}."
    return [Finding(
        "DATA_MISSING", FAIL,
        f"The data file for {dataset} was not found at {where}.",
        fix,
    )]


def _csv_files_near(expected: Path, limit: int = 5) -> list[str]:
    try:
        names = sorted(p.name for p in expected.parent.glob("*.csv"))
    except OSError:
        return []
    return names[:limit]


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------


def _check_latex(config: dict) -> list[Finding]:
    findings: list[Finding] = []
    if shutil.which("pdflatex") is None:
        findings.append(Finding(
            "LATEX_MISSING", WARN,
            "pdflatex was not found on PATH, so no PDF will be produced. "
            "paper.tex is still written, but the final check that requires "
            "a PDF will mark the run INCOMPLETE.",
            "Install a TeX distribution (MiKTeX or TeX Live on Windows, "
            "MacTeX on macOS, texlive on Linux) and open a new terminal.",
        ))
        return findings
    journal = (config.get("writer") or {}).get("venue_format") == "journal"
    engine = "biber" if journal else "bibtex"
    if shutil.which(engine) is None:
        findings.append(Finding(
            "LATEX_MISSING", WARN,
            f"{engine} was not found on PATH, so the PDF's citations and "
            "reference list will be missing.",
            f"Install {engine} with your TeX distribution's package manager.",
        ))
    return findings


# ---------------------------------------------------------------------------
# R (psychometrics)
# ---------------------------------------------------------------------------


def required_r_packages(locked_spec: dict | None) -> tuple[list[str], list[str]]:
    """Return ``(required, optional)`` R packages for a psychometrics spec."""
    battery = [
        m for m in ((locked_spec or {}).get("method_battery") or [])
        if isinstance(m, str)
    ]
    if not battery or any(m not in _R_PACKAGES_BY_METHOD for m in battery):
        return list(_ALL_R_PACKAGES), []
    needed: set[str] = set()
    for method in battery:
        needed.update(_R_PACKAGES_BY_METHOD[method])
    if needed:
        needed.add("jsonlite")
    required = [p for p in _ALL_R_PACKAGES if p in needed]
    optional = [p for p in _ALL_R_PACKAGES if p not in needed]
    return required, optional


def _check_r(
    config: dict, locked_spec: dict | None, probe: bool
) -> list[Finding]:
    from src.r_bridge import RBridgeError, find_rscript, run_r_script

    required, optional = required_r_packages(locked_spec)
    if not required:
        return []
    findings: list[Finding] = []
    if (config.get("sandbox") or {}).get("enabled"):
        findings.append(Finding(
            "R_IN_SANDBOX", WARN,
            "sandbox.enabled is true: if Docker is available, the analysis "
            "code runs in a container that has no R, and every R-based "
            "method will fail there.",
            "Set sandbox.enabled: false for psychometrics runs.",
        ))

    explicit = (config.get("r_bridge") or {}).get("rscript_path") or None
    try:
        rscript = find_rscript(explicit)
    except RBridgeError as exc:
        findings.append(Finding(
            "R_MISSING", FAIL,
            "Rscript was not found; psychometrics methods run in R. "
            f"({exc})",
            "Install R 4.4 or newer from https://cran.r-project.org/ and "
            "set EDM_ARS_RSCRIPT to the full path of Rscript (Rscript.exe "
            "on Windows) if it is not on PATH.",
        ))
        return findings
    if not probe:
        return findings

    try:
        result = run_r_script(
            "preflight_packages.R", {}, timeout_s=180, rscript_path=rscript
        )
    except Exception as exc:  # noqa: BLE001 - any probe failure is reported
        findings.append(Finding(
            "R_PROBE_FAILED", FAIL,
            f"Could not ask R ({rscript}) which packages are installed: {exc}",
            "Check that this Rscript runs: Rscript --vanilla -e \"1\"",
        ))
        return findings

    missing = set(result.get("missing") or []) if isinstance(result, dict) else set()
    missing_required = [p for p in required if p in missing]
    missing_optional = [p for p in optional if p in missing]
    if missing_required:
        findings.append(Finding(
            "R_PACKAGES_MISSING", FAIL,
            f"R ({rscript}) is missing package(s) this study needs: "
            f"{', '.join(missing_required)}.",
            _r_install_fix(rscript, missing_required),
        ))
    if missing_optional:
        findings.append(Finding(
            "R_PACKAGES_MISSING", WARN,
            f"R is missing package(s) needed only by methods this study "
            f"does not declare: {', '.join(missing_optional)}.",
            _r_install_fix(rscript, missing_optional),
        ))
    return findings


def _r_install_fix(rscript: str, packages: list[str]) -> str:
    vector = ", ".join(f"'{p}'" for p in packages)
    return (
        f"Install them: \"{rscript}\" -e \"install.packages(c({vector}), "
        "repos='https://cloud.r-project.org')\""
    )


# ---------------------------------------------------------------------------
# LSAR review gate
# ---------------------------------------------------------------------------


def _check_lsar(config: dict) -> list[Finding]:
    rg = config.get("review_gate") or {}
    root = str(rg.get("lsar_project_path") or "")
    fix = (
        "Clone https://github.com/cgpan/LSAR-public next to this "
        "repository (or anywhere, then set LSAR_HOME to its folder) and "
        "install its requirements, or set review_gate.enabled: false."
    )
    if not root or not os.path.isdir(root):
        return [Finding(
            "LSAR_NOT_FOUND", WARN,
            f"The review gate is enabled but LSAR was not found at "
            f"{root or '(unset)'}; the gate will not run.",
            fix,
        )]
    if not os.path.isfile(os.path.join(root, "lsar", "pipeline.py")):
        return [Finding(
            "LSAR_NOT_FOUND", WARN,
            f"{root} does not look like an LSAR checkout (no "
            "lsar/pipeline.py); the review gate will not run.",
            fix,
        )]
    findings: list[Finding] = []
    error = _try_import_lsar(root)
    if error:
        findings.append(Finding(
            "LSAR_IMPORT_FAILED", WARN,
            f"LSAR at {root} could not be imported ({error}); the review "
            "gate will not run.",
            f"Install LSAR's requirements into this Python: "
            f"{Path(sys.executable).name} -m pip install -r "
            f"{os.path.join(root, 'requirements.txt')}",
        ))
    if not os.environ.get("DEEPSEEK_API_KEY"):
        findings.append(Finding(
            "LSAR_KEY_MISSING", WARN,
            "DEEPSEEK_API_KEY is not set; LSAR's review and scoring stages "
            "are pinned to DeepSeek, so the review gate will fail.",
            _set_key_fix("DEEPSEEK_API_KEY"),
        ))
    return findings


def _try_import_lsar(root: str) -> str | None:
    """Import ``lsar.pipeline`` the way ReviewGate does; return the error."""
    added = root not in sys.path
    if added:
        sys.path.insert(0, root)
    try:
        importlib.import_module("lsar.pipeline")
    except Exception as exc:  # noqa: BLE001 - reported, never raised
        return f"{type(exc).__name__}: {exc}"
    finally:
        if added:
            try:
                sys.path.remove(root)
            except ValueError:
                pass
    return None

