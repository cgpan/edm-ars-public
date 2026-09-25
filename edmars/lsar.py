"""Install and check LSAR, the automated reviewer (github.com/cgpan/LSAR-public).

Install = download the GitHub source archive for ``LSAR_REF``, unpack it
safely into ``<user data dir>/lsar/LSAR-public-<ref>``, and install LSAR's
Python requirements into the SAME interpreter that runs EDM-ARS, because
the pipeline's review gate imports ``lsar.pipeline`` in-process (defects
G2/G6: without those packages the gate silently skips itself).

LSAR is not pip-installed as a package: its ``pyproject.toml`` names a
build backend that does not exist (defect G1), and it resolves
``config.yaml``, ``calibration/`` and ``venue_criteria/`` relative to the
checkout. The runner points ``LSAR_HOME`` at the unpacked folder instead.

Requirements are installed conservatively. A requirement that is already
installed is left alone, even when its version is outside LSAR's pin, so
that installing the reviewer can never break the pipeline's own packages
(e.g. LSAR pins ``pymupdf4llm<1.0`` while a newer one may already be
present). Before anything is installed, a dry run lists every existing
package the install would change; if there is any, nothing is installed
unless the caller passes ``allow_changes=True`` after asking the user.
Versions kept outside LSAR's pins are recorded and reported by
:func:`checks`.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

from edmars import fetch
from edmars.model import Check

LSAR_REPO = "https://github.com/cgpan/LSAR-public"
# TODO(release): pin a tagged LSAR release (or a commit id) once the LSAR
# fix branch is merged and tagged; "master" follows whatever is published.
LSAR_REF = "master"

#: Files the review gate reads under LSAR_HOME (EDM config.yaml review_gate).
REQUIRED_FILES: tuple[str, ...] = (
    "config.yaml",
    "calibration/anchors_edm.yaml",
    "lsar/pipeline.py",
)

#: Import name -> distribution name, for what LSAR imports at run time.
RUNTIME_MODULES: dict[str, str] = {
    "tenacity": "tenacity",
    "jinja2": "Jinja2",
    "httpx": "httpx",
    "yaml": "PyYAML",
    "dotenv": "python-dotenv",
    "openai": "openai",
    "anthropic": "anthropic",
    "fitz": "PyMuPDF",
    "pymupdf4llm": "pymupdf4llm",
    "arxiv": "arxiv",
}

#: Lines of LSAR's requirements.txt that are development tools, not needed
#: to run a review.
DEV_ONLY: frozenset[str] = frozenset(
    {"pytest", "pytest-mock", "pytest-cov", "ruff", "mypy", "black"}
)

#: Model ids the provider no longer serves (defects E4/G5): LSAR's released
#: config still routes three stages to one, and those stages then degrade
#: silently. Reported by :func:`checks`; never changed by the CLI, because
#: LSAR's review/scoring settings are calibration-pinned.
RETIRED_MODEL_IDS: frozenset[str] = frozenset({"deepseek-v4-flash"})

#: The GitHub archive is ~2 MB; anything far bigger is not LSAR.
MAX_ARCHIVE_BYTES = 100_000_000
MAX_UNPACKED_BYTES = 300_000_000
INSTALL_RECORD = ".edmars-install.json"


class LsarInstallError(fetch.UserFacingError):
    """LSAR could not be installed; ``plan`` says what pip would change."""

    def __init__(self, message: str, plan: "RequirementPlan | None" = None) -> None:
        super().__init__(message)
        self.plan = plan


@dataclass
class RequirementPlan:
    """What installing LSAR's requirements would do to this interpreter."""

    #: Requirement lines that will be passed to pip.
    to_install: list[str] = field(default_factory=list)
    #: "name installed -> new" for packages the install would change.
    changes: list[str] = field(default_factory=list)
    #: "name installed (LSAR asks spec)" for kept, out-of-pin packages.
    unmet_pins: list[str] = field(default_factory=list)
    #: How the dry run was done: "pip", "uv", "pins" or "none".
    method: str = "none"


# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------


def archive_url(ref: str = LSAR_REF) -> str:
    return f"{LSAR_REPO}/archive/{ref}.tar.gz"


def _safe_ref(ref: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", ref).strip("-.") or "ref"


def lsar_root() -> Path:
    from edmars import paths

    return Path(paths.data_dir()) / "lsar"


def home_for_ref(ref: str = LSAR_REF) -> Path:
    return lsar_root() / f"LSAR-public-{_safe_ref(ref)}"


def _get(settings: Mapping[str, Any] | None, dotted: str) -> Any:
    node: Any = settings or {}
    for part in dotted.split("."):
        if not isinstance(node, Mapping):
            return None
        node = node.get(part)
    return node


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _child_env() -> dict[str, str]:
    """Environment of a child that must see what the pipeline child sees."""
    env = {
        k: v for k, v in os.environ.items()
        if k.upper() not in ("PYTHONPATH", "PYTHONHOME")
        and not any(h in k.upper() for h in ("API_KEY", "TOKEN", "SECRET", "PASSWORD"))
    }
    env.update({
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        # The runner starts the pipeline with PYTHONNOUSERSITE=1; a package
        # that only exists in the user site would pass here and fail there.
        "PYTHONNOUSERSITE": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_INPUT": "1",
        # Never fall back to a --user install the pipeline child cannot see.
        "PIP_USER": "0",
    })
    return env


def _run(args: Sequence[str], *, timeout: float, cwd: str | Path | None = None) -> tuple[int | None, str]:
    """(returncode or None, combined output) via ``edmars.proc``."""
    from edmars import proc

    try:
        done = proc.run(list(args), timeout=timeout,
                        cwd=str(cwd) if cwd is not None else None, env=_child_env())
    except Exception as exc:  # noqa: BLE001 - classify below, re-raise the rest
        if type(exc).__name__ == "TimeoutExpired":
            return None, f"timed out after {timeout:.0f} s"
        if isinstance(exc, OSError):
            return None, f"could not start {Path(args[0]).name}: {exc}"
        raise
    return done.returncode, (done.stdout or "") + (done.stderr or "")


def _tail(text: str, lines: int = 4) -> str:
    kept = [ln.strip() for ln in (text or "").strip().splitlines() if ln.strip()]
    return " / ".join(kept[-lines:])[:500]


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

_IMPORT_CHECK = "import sys; sys.path.insert(0, sys.argv[1]); import lsar.pipeline"


def _file_problems(home: Path) -> list[str]:
    if not home.is_dir():
        return [f"{home} does not exist."]
    problems = [f"{rel} is missing from {home}." for rel in REQUIRED_FILES
                if not (home / rel).is_file()]
    if problems:
        return problems
    try:
        import yaml

        data = yaml.safe_load((home / "calibration" / "anchors_edm.yaml").read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - any parse failure is a problem
        return [f"calibration/anchors_edm.yaml could not be read: {exc}"]
    if not isinstance(data, dict) or "overall_p25_full" not in data:
        return ["calibration/anchors_edm.yaml has no overall_p25_full value, so the "
                "review gate has no benchmark."]
    return []


def verify(home: Path) -> list[str]:
    """Problems with the LSAR install at ``home`` (empty list = ready).

    Checks the files the review gate reads, then imports ``lsar.pipeline``
    in a child Python started like the pipeline's own run (no user site,
    no PYTHONPATH), so a missing dependency shows up here and not as a
    silently skipped review at the end of a paid run.
    """
    home = Path(home)
    problems = _file_problems(home)
    if problems:
        return problems
    code, output = _run([sys.executable, "-c", _IMPORT_CHECK, str(home)],
                        timeout=180, cwd=home)
    if code != 0:
        missing = re.findall(r"No module named '([^'.]+)", output)
        if missing:
            problems.append(
                f"LSAR needs the Python package '{missing[-1]}', which is not installed "
                "for the Python that runs EDM-ARS."
            )
        else:
            problems.append(f"Python could not load LSAR: {_tail(output)}")
    return problems


def retired_model_stages(home: Path) -> list[str]:
    """``stage: model`` entries of LSAR's config that use a retired model id."""
    try:
        import yaml

        config = yaml.safe_load((Path(home) / "config.yaml").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - a missing/broken config is reported elsewhere
        return []
    llm = config.get("llm") if isinstance(config, dict) else None
    if not isinstance(llm, dict):
        return []
    found: list[str] = []
    if str(llm.get("model")) in RETIRED_MODEL_IDS:
        found.append(f"default: {llm.get('model')}")
    stages = llm.get("stage_models")
    if isinstance(stages, dict):
        for stage, model in stages.items():
            if str(model) in RETIRED_MODEL_IDS:
                found.append(f"{stage}: {model}")
    return found


def _read_install_record(home: Path) -> dict[str, Any]:
    try:
        data = json.loads((Path(home) / INSTALL_RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def checks(settings: Mapping[str, Any], *, deep: bool = False) -> list[Check]:
    """Is LSAR ready to review? ``deep`` also imports it in a child Python.

    Anything that would make a review skip itself -- not installed, files
    missing, a runtime package missing -- is ``fail`` whether or not the
    reviewer is switched on: callers ask this before turning it on, and
    "no fail" must mean "ready". (The DeepSeek key is the caller's check:
    the doctor and the wizard both look it up in the key store.)
    """
    title = "Automated reviewer (LSAR)"
    enabled = bool(_get(settings, "lsar.enabled"))
    home_value = _get(settings, "lsar.home")
    if not home_value:
        return [Check(title, "fail",
                      "Not installed. It scores each finished paper against a benchmark "
                      "and adds about 20-40 minutes per study.",
                      fix="edmars setup reviewer")]
    home = Path(str(home_value))
    problems = _file_problems(home)
    if problems:
        return [Check(title, "fail", " ".join(problems), fix="edmars setup reviewer")]

    ref = str(_get(settings, "lsar.ref") or LSAR_REF)
    state = "on" if enabled else "installed but turned off"
    out = [Check(title, "ok", f"{state}; version {ref[:12]} at {home}")]

    missing = [dist for module, dist in RUNTIME_MODULES.items()
               if importlib.util.find_spec(module) is None]
    unmet = _read_install_record(home).get("unmet_pins") or []
    if missing:
        out.append(Check(
            "LSAR's Python packages", "fail",
            "Missing: " + ", ".join(missing) + ". Without them every review is "
            "skipped.",
            fix="edmars setup reviewer",
        ))
    elif unmet:
        out.append(Check(
            "LSAR's Python packages", "warn",
            "Installed, but these were kept at versions outside LSAR's own pins so "
            "that EDM-ARS keeps working: " + "; ".join(str(u) for u in unmet) + ".",
        ))
    else:
        out.append(Check("LSAR's Python packages", "ok", "All present."))

    retired = retired_model_stages(home)
    if retired:
        out.append(Check(
            "LSAR model settings", "warn",
            "LSAR's config.yaml sends these steps to a model id the provider no "
            f"longer serves ({'; '.join(retired)}). Those steps fail quietly and the "
            "related-work part of each review is thinner than it should be.",
            fix="Update LSAR when a fixed version is published: edmars setup reviewer",
        ))

    if deep:
        deep_problems = verify(home)
        if deep_problems:
            out.append(Check("LSAR loads in Python", "fail", " ".join(deep_problems),
                             fix="edmars setup reviewer"))
        else:
            out.append(Check("LSAR loads in Python", "ok",
                             "lsar.pipeline imports in a fresh Python."))
    return out


# ---------------------------------------------------------------------------
# Archive handling
# ---------------------------------------------------------------------------


def _safe_extract(archive: Path, dest: Path) -> tuple[Path, str | None]:
    """Unpack a GitHub source archive into ``dest``; refuse anything unsafe.

    Returns (the single top-level folder, the commit id GitHub records in
    the archive's pax header, if any). Absolute paths, ``..`` components,
    drive letters and backslashes are refused outright; links, devices and
    FIFOs are skipped (LSAR has none).
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    with tarfile.open(archive, "r:gz") as tf:
        commit = tf.pax_headers.get("comment") if tf.pax_headers else None
        members: list[tarfile.TarInfo] = []
        tops: set[str] = set()
        total = 0
        for member in tf.getmembers():
            name = member.name
            parts = PurePosixPath(name).parts
            if (not name or name.startswith(("/", "\\")) or "\\" in name
                    or ".." in parts or any(":" in p for p in parts)):
                raise LsarInstallError(f"The LSAR archive contains an unsafe path: {name!r}")
            if not (member.isfile() or member.isdir()):
                continue
            target = (root / name).resolve()
            if target != root and root not in target.parents:
                raise LsarInstallError(f"The LSAR archive contains an unsafe path: {name!r}")
            total += max(0, member.size)
            if total > MAX_UNPACKED_BYTES:
                raise LsarInstallError("The LSAR archive unpacks to far more than expected.")
            tops.add(parts[0])
            members.append(member)
        if len(tops) != 1:
            raise LsarInstallError(
                f"The LSAR archive should hold one folder; it holds {sorted(tops)[:5]}."
            )
        if hasattr(tarfile, "data_filter"):
            tf.extractall(root, members=members, filter="data")
        else:  # pragma: no cover - Python < 3.11.4
            tf.extractall(root, members=members)
    return root / tops.pop(), commit


# ---------------------------------------------------------------------------
# Requirements
# ---------------------------------------------------------------------------


def runtime_requirements(home: Path) -> list[str]:
    """LSAR's requirements.txt without comments, options and dev tools."""
    path = Path(home) / "requirements.txt"
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        if _req_name(line).lower() in DEV_ONLY or not _applies(line):
            continue
        lines.append(line)
    return lines


def _req_name(line: str) -> str:
    match = re.match(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)", line)
    return match.group(1) if match else line.strip()


def _installed_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception:  # noqa: BLE001 - broken metadata counts as absent
        return None


def _requirement(line: str) -> Any | None:
    """``packaging`` Requirement for ``line``, or None when it cannot be parsed."""
    try:
        from packaging.requirements import Requirement
    except ImportError:  # pragma: no cover - packaging ships with pip/matplotlib
        try:
            from pip._vendor.packaging.requirements import Requirement  # type: ignore[no-redef, assignment]
        except ImportError:
            return None
    try:
        return Requirement(line)
    except Exception:  # noqa: BLE001 - unparsable line
        return None


def _applies(line: str) -> bool:
    """False for a requirement whose environment marker excludes this Python."""
    req = _requirement(line)
    if req is None or req.marker is None:
        return True
    try:
        return bool(req.marker.evaluate())
    except Exception:  # noqa: BLE001 - an odd marker is not our call to make
        return True


def _satisfies(line: str, version: str) -> bool | None:
    """Does ``version`` meet the requirement ``line``? None = cannot tell."""
    req = _requirement(line)
    if req is None:
        return None
    try:
        return bool(req.specifier.contains(version, prereleases=True))
    except Exception:  # noqa: BLE001 - unparsable version
        return None


def _installer() -> tuple[list[str], str]:
    """Base command that installs into ``sys.executable`` and its flavour.

    Prefers the interpreter's own pip. A venv made by ``uv venv`` has no
    pip, so ``uv pip --python <this python>`` is used when uv is available;
    otherwise pip is bootstrapped with ``ensurepip``.
    """
    from edmars import proc

    code, _ = _run([sys.executable, "-m", "pip", "--version"], timeout=60)
    if code == 0:
        return [sys.executable, "-m", "pip", "install"], "pip"
    uv = os.environ.get("EDMARS_UV") or proc.which("uv")
    if uv:
        return [uv, "pip", "install", "--python", sys.executable], "uv"
    code, output = _run([sys.executable, "-m", "ensurepip", "--upgrade"], timeout=300)
    if code == 0:
        return [sys.executable, "-m", "pip", "install"], "pip"
    raise LsarInstallError(
        "This Python has no pip, and uv was not found, so LSAR's packages cannot "
        f"be installed ({_tail(output, 2)})."
    )


def _dry_run_changes(base: list[str], flavour: str, req_file: Path) -> tuple[list[str], str]:
    """(packages the install would change, method). Raises on a failed dry run."""
    if flavour == "pip":
        report = req_file.with_name("pip-report.json")
        code, output = _run([*base, "--dry-run", "--quiet", "--report", str(report),
                             "-r", str(req_file)], timeout=600)
        if code != 0 or not report.is_file():
            raise LsarInstallError(
                "Checking LSAR's requirements failed before anything was installed: "
                + _tail(output)
            )
        data = json.loads(report.read_text(encoding="utf-8"))
        changes: list[str] = []
        for item in data.get("install") or []:
            meta = item.get("metadata") or {}
            name, new = str(meta.get("name") or ""), str(meta.get("version") or "")
            old = _installed_version(name) if name else None
            if old is not None and old != new:
                changes.append(f"{name} {old} -> {new}")
        return changes, "pip"
    code, output = _run([*base, "--dry-run", "-r", str(req_file)], timeout=600)
    if code != 0:
        raise LsarInstallError(
            "Checking LSAR's requirements failed before anything was installed: "
            + _tail(output)
        )
    removed = dict(re.findall(r"^\s*-\s+([A-Za-z0-9._-]+)==(\S+)", output, flags=re.MULTILINE))
    added = dict(re.findall(r"^\s*\+\s+([A-Za-z0-9._-]+)==(\S+)", output, flags=re.MULTILINE))
    changes = [f"{name} {old} -> {added.get(name, 'removed')}" for name, old in removed.items()]
    return changes, "uv"


def plan_requirements(
    home: Path, *, allow_changes: bool = False, dry_run: bool = True
) -> RequirementPlan:
    """Work out which requirement lines to install and what that would change."""
    plan = RequirementPlan()
    for line in runtime_requirements(home):
        name = _req_name(line)
        installed = _installed_version(name)
        if installed is None:
            plan.to_install.append(line)
            continue
        fits = _satisfies(line, installed)
        if fits is False:
            if allow_changes:
                plan.to_install.append(line)
            else:
                spec = line[len(name):].strip()
                plan.unmet_pins.append(f"{name} {installed} (LSAR asks {spec})")
    if plan.to_install and dry_run:
        base, flavour = _installer()
        with tempfile.TemporaryDirectory(prefix="edmars-lsar-req-") as tmp:
            req_file = Path(tmp) / "requirements.txt"
            req_file.write_text("\n".join(plan.to_install) + "\n", encoding="utf-8")
            plan.changes, plan.method = _dry_run_changes(base, flavour, req_file)
    elif plan.to_install:
        plan.method = "pins"
    return plan


def _install_requirements(plan: RequirementPlan) -> None:
    if not plan.to_install:
        return
    base, _ = _installer()
    with tempfile.TemporaryDirectory(prefix="edmars-lsar-req-") as tmp:
        req_file = Path(tmp) / "requirements.txt"
        req_file.write_text("\n".join(plan.to_install) + "\n", encoding="utf-8")
        code, output = _run([*base, "-r", str(req_file)], timeout=1800)
    if code != 0:
        raise LsarInstallError(
            "Installing LSAR's Python packages failed; nothing else was changed: "
            + _tail(output, 6),
            plan,
        )


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------


def _rmtree(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def install(
    settings: dict[str, Any],
    *,
    ref: str | None = None,
    allow_changes: bool = False,
    session: Any | None = None,
    on_step: Callable[[str], None] | None = None,
) -> Path:
    """Download, unpack, install requirements for, and verify LSAR.

    Returns the LSAR home folder and saves ``lsar.home`` / ``lsar.ref``
    into ``settings``. Raises :class:`LsarInstallError` with a plain
    message (and the requirement plan when relevant). A failed install
    leaves an earlier working install in place.
    """
    ref = ref or LSAR_REF

    def step(text: str) -> None:
        if on_step is not None:
            try:
                on_step(text)
            except Exception:  # noqa: BLE001 - a status line is not fatal
                pass

    root = lsar_root()
    root.mkdir(parents=True, exist_ok=True)
    staging = root / f".staging-{uuid.uuid4().hex[:8]}"
    archive = root / ".downloads" / f"LSAR-public-{_safe_ref(ref)}.tar.gz"
    try:
        step("Downloading LSAR from GitHub")
        try:
            fetch.download_file(archive_url(ref), archive, session=session,
                                max_bytes=MAX_ARCHIVE_BYTES)
        except fetch.DownloadError as exc:
            raise LsarInstallError(str(exc)) from exc
        step("Unpacking LSAR")
        try:
            unpacked, commit = _safe_extract(archive, staging)
        except (tarfile.TarError, EOFError, OSError) as exc:
            raise LsarInstallError(f"The LSAR archive could not be unpacked: {exc}") from exc
        problems = _file_problems(unpacked)
        if problems:
            raise LsarInstallError("The downloaded LSAR is incomplete: " + " ".join(problems))

        step("Checking LSAR's Python packages against the ones EDM-ARS uses")
        plan = plan_requirements(unpacked, allow_changes=allow_changes)
        if plan.changes and not allow_changes:
            raise LsarInstallError(
                "Installing LSAR's Python packages would change packages EDM-ARS "
                "already uses: " + "; ".join(plan.changes) + ". Nothing was "
                "installed or changed.",
                plan,
            )
        if plan.to_install:
            step("Installing LSAR's Python packages")
        _install_requirements(plan)

        # Prove the new copy loads BEFORE it replaces a working install.
        step("Checking that LSAR loads")
        problems = verify(unpacked)
        if problems:
            raise LsarInstallError(
                "LSAR was downloaded but does not load: " + " ".join(problems), plan
            )
        record = {
            "ref": ref,
            "commit": commit,
            "installed_at": _now(),
            "unmet_pins": plan.unmet_pins,
            "installed_requirements": plan.to_install,
            "dry_run": plan.method,
        }
        try:
            (unpacked / INSTALL_RECORD).write_text(json.dumps(record, indent=2),
                                                   encoding="utf-8")
        except OSError:
            pass

        home = home_for_ref(ref)
        old = root / f".old-{uuid.uuid4().hex[:8]}"
        try:
            if home.exists():
                os.replace(home, old)
            try:
                os.replace(unpacked, home)
            except OSError:
                if old.exists() and not home.exists():
                    os.replace(old, home)
                raise
        except OSError as exc:
            raise LsarInstallError(
                f"Could not replace the LSAR folder {home} ({exc}). If a study is "
                "running, try again after it finishes."
            ) from exc
        _rmtree(old)
    finally:
        _rmtree(staging)
        try:
            archive.unlink()
        except OSError:
            pass

    from edmars import settings as settings_mod

    settings_mod.set_(settings, "lsar.home", str(home))
    settings_mod.set_(settings, "lsar.ref", commit or ref)
    settings_mod.save(settings)
    return home


# ---------------------------------------------------------------------------
# Using LSAR: the run config block, and `edmars review RUN`
# ---------------------------------------------------------------------------

#: Venues with a calibrated benchmark in calibration/anchors_edm.yaml.
_VENUE_ALIASES: dict[str, str] = {"AERA OPEN": "AERA_OPEN", "AERAOPEN": "AERA_OPEN"}

REVIEW_FOLDER = "lsar_review_manual"
#: Secrets a manual review may need; LSAR's scoring is DeepSeek-pinned.
REVIEW_SECRETS: tuple[str, ...] = ("DEEPSEEK_API_KEY", "TAVILY_API_KEY",
                                   "SEMANTIC_SCHOLAR_API_KEY")


class LsarReviewError(fetch.UserFacingError):
    """A manual review could not run or did not produce a report."""


def lsar_venue(venue: str | None) -> str:
    """EDM-ARS venue name -> LSAR's venue name ("AERA Open" -> "AERA_OPEN")."""
    value = (venue or "EDM").strip().upper()
    return _VENUE_ALIASES.get(value, value.replace(" ", "_"))


def gate_config(settings: Mapping[str, Any], venue: str | None = "EDM") -> dict[str, Any]:
    """The ``review_gate`` keys a run needs to use this LSAR install.

    Absolute paths (never ``${LSAR_HOME}``-relative), so the gate cannot
    fall back to the cwd-relative ``../LSAR`` default (defect G4). The
    runner merges this into the run config; keys not listed here keep the
    shipped config's values.
    """
    home_value = _get(settings, "lsar.home")
    if not home_value:
        return {"enabled": False}
    home = Path(str(home_value))
    return {
        "enabled": True,
        "lsar_project_path": str(home),
        "lsar_config_path": str(home / "config.yaml"),
        "calibration_path": str(home / "calibration" / "anchors_edm.yaml"),
        "venue": lsar_venue(venue),
    }


def benchmark_for(home: Path, venue: str | None) -> float | None:
    """The calibrated benchmark (P25 of accepted papers) for ``venue``, if any."""
    try:
        import yaml

        data = yaml.safe_load((Path(home) / "calibration" / "anchors_edm.yaml")
                              .read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - no benchmark rather than a crash
        return None
    if not isinstance(data, dict):
        return None
    name = lsar_venue(venue)
    if name == "EDM":
        value = data.get("overall_p25_full")
    else:
        entry = (data.get("venues") or {}).get(name)
        value = entry.get("p25") if isinstance(entry, dict) else None
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _run_venue(run_dir: Path) -> str | None:
    """The venue a run was written for (run_config.yaml, then runner.json)."""
    try:
        import yaml

        config = yaml.safe_load((run_dir / "run_config.yaml").read_text(encoding="utf-8"))
        venue = ((config or {}).get("review_gate") or {}).get("venue")
        if venue:
            return str(venue)
    except Exception:  # noqa: BLE001 - fall through to runner.json
        pass
    try:
        runner = json.loads((run_dir / "runner.json").read_text(encoding="utf-8"))
        study = runner.get("study") or {}
        return str(study.get("venue")) if study.get("venue") else None
    except Exception:  # noqa: BLE001 - no venue recorded
        return None


def _review_env(names: Sequence[str]) -> dict[str, str]:
    env = _child_env()
    try:
        from edmars import secrets

        if hasattr(secrets, "child_secrets"):
            env.update(secrets.child_secrets(names))
        else:  # pragma: no cover - older secrets module
            for name in names:
                value = secrets.get_secret(name)
                if value:
                    env[name] = value
    except ImportError:  # pragma: no cover - secrets module always ships
        for name in names:
            if os.environ.get(name):
                env[name] = os.environ[name]
    return env


def review_paper(
    run_dir: str | Path,
    settings: Mapping[str, Any],
    *,
    venue: str | None = None,
    timeout_s: float = 5400,
) -> dict[str, Any]:
    """Review a finished run's paper with LSAR and return the result.

    Runs LSAR's own ``scripts/run_review.py`` in a child Python on the
    run's ``paper_for_review.pdf`` (the gate's citation-cleaned copy) or
    ``paper.pdf``. Nothing in the run is modified except the new
    ``<run>/lsar_review_manual/`` folder, which receives LSAR's report and
    a ``console.log``. Takes 10-40 minutes and costs a few cents of
    DeepSeek credit.

    Returns ``{"output_dir", "pdf", "venue", "score", "recommendation",
    "benchmark", "passed", "report_md", "report_json"}``; ``benchmark`` and
    ``passed`` are None for venues without a calibrated benchmark.
    """
    from edmars import proc

    run_dir = Path(run_dir)
    home_value = _get(settings, "lsar.home")
    if not home_value:
        raise LsarReviewError("LSAR is not installed. Run: edmars setup reviewer")
    home = Path(str(home_value))
    problems = _file_problems(home)
    if problems:
        raise LsarReviewError("LSAR is not ready: " + " ".join(problems))
    script = home / "scripts" / "run_review.py"
    if not script.is_file():
        raise LsarReviewError(f"LSAR's review script is missing: {script}")
    pdf = next((run_dir / n for n in ("paper_for_review.pdf", "paper.pdf")
                if (run_dir / n).is_file()), None)
    if pdf is None:
        raise LsarReviewError(
            f"{run_dir} has no paper PDF (paper.pdf), so there is nothing to review. "
            "Check `edmars doctor` for the PDF tools, then resume the study."
        )
    env = _review_env(REVIEW_SECRETS)
    if not env.get("DEEPSEEK_API_KEY"):
        raise LsarReviewError(
            "LSAR needs a DeepSeek key (its scoring is calibrated on DeepSeek). "
            "Add one with: edmars setup reviewer"
        )
    name = lsar_venue(venue or _run_venue(run_dir))
    out_dir = run_dir / REVIEW_FOLDER
    out_dir.mkdir(parents=True, exist_ok=True)
    args = [sys.executable, str(script), str(pdf), "--venue", name,
            "--config", str(home / "config.yaml"), "--output-dir", str(out_dir), "--force"]
    try:
        done = proc.run(args, timeout=timeout_s, cwd=str(home), env=env)
        code, output = done.returncode, (done.stdout or "") + (done.stderr or "")
    except Exception as exc:  # noqa: BLE001 - classify below, re-raise the rest
        if type(exc).__name__ == "TimeoutExpired":
            code, output = None, f"LSAR did not finish within {timeout_s / 60:.0f} minutes."
        elif isinstance(exc, OSError):
            code, output = None, f"Could not start LSAR: {exc}"
        else:
            raise
    log_path = out_dir / "console.log"
    try:
        from edmars import secrets

        text = secrets.redact(output) if hasattr(secrets, "redact") else output
    except ImportError:  # pragma: no cover
        text = output
    for name_ in REVIEW_SECRETS:
        if env.get(name_):
            text = text.replace(env[name_], "[redacted]")
    log_path.write_text(text, encoding="utf-8", errors="replace")

    report_json = out_dir / "LSAR_Review_Report.json"
    if code != 0 or not report_json.is_file():
        raise LsarReviewError(
            f"LSAR did not produce a report (exit code {code}). Details: {log_path}. "
            f"Last lines: {_tail(text, 3)}"
        )
    try:
        report = json.loads(report_json.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise LsarReviewError(f"LSAR's report could not be read: {exc}") from exc
    scores = report.get("scores") if isinstance(report, dict) else None
    scores = scores if isinstance(scores, dict) else {}
    raw_score = scores.get("overall_score")
    try:
        score: float | None = float(raw_score) if raw_score is not None else None
    except (TypeError, ValueError):
        score = None
    benchmark = benchmark_for(home, name)
    return {
        "output_dir": out_dir,
        "pdf": pdf,
        "venue": name,
        "score": score,
        "recommendation": scores.get("recommendation"),
        "benchmark": benchmark,
        "passed": (score >= benchmark) if (score is not None and benchmark is not None) else None,
        "report_md": out_dir / "LSAR_Review_Report.md",
        "report_json": report_json,
    }


def review_run(
    run_dir: str | Path,
    settings: Mapping[str, Any],
    *,
    venue: str | None = None,
) -> int:
    """``edmars review RUN``: review the paper and tell the user how it went.

    Returns 0 when a report was written, 1 otherwise; every problem is
    shown as a plain message (nothing is raised for expected failures).
    """
    from edmars import ui

    run_dir = Path(run_dir)
    note = ("Reviewing the paper with LSAR. This usually takes 10-40 minutes; "
            "you can leave this window open.")
    try:
        status_cm = getattr(ui, "status", None)
        if status_cm is not None:
            with status_cm(note):
                result = review_paper(run_dir, settings, venue=venue)
        else:  # pragma: no cover - ui without a spinner
            ui.info(note)
            result = review_paper(run_dir, settings, venue=venue)
    except LsarReviewError as exc:
        ui.fail(str(exc))
        return 1
    score, benchmark = result["score"], result["benchmark"]
    lines = [f"Venue: {result['venue']}"]
    if score is not None:
        lines.append(f"Score: {score:.1f} / 10"
                     + (f" ({result['recommendation']})" if result["recommendation"] else ""))
    if benchmark is not None and score is not None:
        verdict = "at or above" if result["passed"] else "below"
        lines.append(f"Benchmark for this venue: {benchmark:.2f} (the score is {verdict} it)")
    else:
        lines.append("No calibrated benchmark for this venue: the score is shown on its own.")
    lines.append(f"Full review: {result['report_md']}")
    lines.append("Two readings of the same paper can differ by about 2 points. "
                 "Treat the score as a rough signal, not a verdict or a prediction "
                 "of acceptance.")
    ui.panel("Automated review (LSAR)", "\n".join(lines))
    ui.ok(f"Review saved in {result['output_dir']}")
    return 0
