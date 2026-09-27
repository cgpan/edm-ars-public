"""Start, stop and resume a study as a detached child process.

The pipeline runs as ``<python> -m src.main --config <run>/run_config.yaml
--output-dir <run> --dataset <d> [--research-spec ...] [--prompt ...]``
with the application root as its working directory. The CLI never
imports the pipeline to run it; it writes a per-run config, launches the
child, and from then on only reads the run folder.

What this module writes into a run folder:

* ``run_config.yaml`` -- the shipped ``config.yaml`` deep-merged with the
  user's settings and the study's choices. Absolute paths, sandbox off,
  no secrets (keys travel only in the child's environment).
* ``research_spec.locked.json`` -- the study plan, when there is one.
* ``runner.json`` -- how the run was launched (argv, pid, times), so
  ``edmars stop`` / ``edmars resume`` / the live view can find it again.
* ``STOP`` -- a flag file written by ``stop()``. The pipeline watches for
  it and stops the way it does on SIGTERM (checkpoint and status saved).

One study runs at a time: ``<data dir>/active_run.json`` holds the
running study's pid and folder, and is treated as stale once that pid is
gone.
"""
from __future__ import annotations

import copy
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Literal

import yaml

from edmars import paths, proc
from edmars import secrets as edsecrets
from edmars.runstate import as_dict, load_state, parse_ts, process_alive

if TYPE_CHECKING:  # pragma: no cover
    from edmars.model import StudyPlan


class RunnerError(RuntimeError):
    """A launch/stop/resume request that cannot be carried out; the message
    is written for the person at the keyboard."""


#: Provider id -> the environment variable holding its key. The wizard's
#: provider catalog is authoritative; this is the fallback.
_PROVIDER_ENV = {
    "deepseek": "DEEPSEEK_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "local": "OPENAI_API_KEY",
    "minimax": "MINIMAX_API_KEY",
}

#: Keys passed to the pipeline when present, whatever the provider.
_OPTIONAL_KEYS = ("SEMANTIC_SCHOLAR_API_KEY", "OPENALEX_API_KEY", "TAVILY_API_KEY")

_LOCK_NAME = "active_run.json"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _sget(settings: dict[str, Any] | None, dotted: str, default: Any = None) -> Any:
    node: Any = settings or {}
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return default if node is None else node


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    os.replace(tmp, path)


def _write_json(path: Path, data: Any) -> None:
    _write_text_atomic(path, json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _read_runner(run_dir: Path) -> dict[str, Any]:
    return _read_json(run_dir / "runner.json")


def _app_version() -> str | None:
    try:
        import edmars

        version = getattr(edmars, "__version__", None)
        return str(version) if version else None
    except Exception:  # noqa: BLE001
        return None


def _create_time(pid: int) -> float | None:
    try:
        import psutil

        return float(psutil.Process(pid).create_time())
    except Exception:  # noqa: BLE001
        return None


def _load_settings() -> dict[str, Any]:
    from edmars import settings as settings_mod

    loaded = settings_mod.load()
    return loaded if isinstance(loaded, dict) else {}


# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------


def studies_dir(settings: dict[str, Any]) -> Path:
    configured = _sget(settings, "studies_dir")
    base = Path(str(configured)).expanduser() if configured else Path(paths.default_studies_dir())
    return base.absolute()


def raw_data_dir(settings: dict[str, Any]) -> Path:
    try:
        from edmars import datasets
    except ImportError:
        return (Path(paths.data_dir()) / "data" / "raw").absolute()
    return Path(datasets.raw_data_dir(settings)).absolute()


def _lock_path() -> Path:
    return Path(paths.data_dir()) / _LOCK_NAME


# ---------------------------------------------------------------------------
# Effective config
# ---------------------------------------------------------------------------


def _provider_models(provider: str, settings: dict[str, Any]) -> dict[str, str]:
    defaults: dict[str, str] = {}
    try:
        from edmars import providers

        got = providers.default_models(provider)
        if isinstance(got, dict):
            defaults = {str(k): str(v) for k, v in got.items() if v}
    except (ImportError, AttributeError, KeyError):
        defaults = {}
    overrides = _sget(settings, "models", {}) or {}
    if isinstance(overrides, dict):
        defaults.update({str(k): str(v) for k, v in overrides.items() if v})
    return defaults


def _expand_lsar_home(node: Any, home: str) -> Any:
    if isinstance(node, str):
        return node.replace("${LSAR_HOME}", home).replace("$LSAR_HOME", home)
    if isinstance(node, dict):
        return {k: _expand_lsar_home(v, home) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand_lsar_home(v, home) for v in node]
    return node


def _lsar_home(settings: dict[str, Any]) -> Path | None:
    home = _sget(settings, "lsar.home")
    if not home:
        return None
    path = Path(str(home)).expanduser()
    return path.absolute() if path.is_dir() else None


def review_enabled(settings: dict[str, Any], plan: "StudyPlan") -> bool:
    """The LSAR gate runs only when the study asks for it, LSAR is set up
    and enabled, and the provider is not an experimental local model."""
    if not getattr(plan, "review", False):
        return False
    if str(_sget(settings, "provider", "deepseek")) == "local":
        return False
    if _sget(settings, "lsar.enabled", True) is False:
        return False
    return _lsar_home(settings) is not None


def build_effective_config(settings: dict[str, Any], plan: "StudyPlan") -> dict[str, Any]:
    """The shipped ``config.yaml`` with this user's and this study's choices.

    Never contains a secret: API keys reach the pipeline through the child
    environment only.
    """
    root = Path(paths.app_root())
    with open(root / "config.yaml", encoding="utf-8") as fh:
        base = yaml.safe_load(fh) or {}
    cfg: dict[str, Any] = copy.deepcopy(base)

    provider = str(_sget(settings, "provider", "deepseek") or "deepseek")
    llm_provider = "openai" if provider == "local" else provider
    cfg["llm_provider"] = llm_provider
    models = _provider_models(provider, settings)
    base_url = _sget(settings, "provider_base_url")
    writer_model: str | None = None
    if llm_provider == "anthropic":
        cfg["models"] = {**(cfg.get("models") or {}), **models}
        writer_model = cfg["models"].get("writer")
        if writer_model and not cfg["models"].get("outline_agent"):
            # The shipped top-level ``models`` block has no outline_agent
            # entry, and the Anthropic path has no default for it: the
            # outline step would call the API with an empty model id.
            cfg["models"]["outline_agent"] = writer_model
    else:
        block = cfg.setdefault(llm_provider, {}) or {}
        cfg[llm_provider] = block
        block["models"] = {**(block.get("models") or {}), **models}
        if base_url and provider in ("local", "openai"):
            block["base_url"] = str(base_url)
        writer_model = block["models"].get("writer")
        if writer_model and not block["models"].get("revision_writer"):
            # Without this the review gate's reviser falls back to
            # review_gate.revision_model, which names a DeepSeek model.
            block["models"]["revision_writer"] = writer_model
        writer_model = block["models"].get("revision_writer") or writer_model

    # ---- review gate (LSAR) -------------------------------------------------
    rg = cfg.setdefault("review_gate", {}) or {}
    cfg["review_gate"] = rg
    rg["venue"] = str(getattr(plan, "venue", None) or _sget(settings, "defaults.venue", "EDM"))
    home = _lsar_home(settings)
    if review_enabled(settings, plan) and home is not None:
        gate: dict[str, Any] = {}
        try:
            from edmars import lsar

            gate = dict(lsar.gate_config({**settings, "lsar": {
                **as_dict(_sget(settings, "lsar", {})), "home": str(home)}}, rg["venue"]))
        except Exception:  # noqa: BLE001 -- the fallback below writes the same paths
            gate = {}
        rg["enabled"] = True
        rg["lsar_project_path"] = str(gate.get("lsar_project_path") or home)
        rg["lsar_config_path"] = str(gate.get("lsar_config_path") or home / "config.yaml")
        rg["calibration_path"] = str(
            gate.get("calibration_path") or home / "calibration" / "anchors_edm.yaml")
        if gate.get("venue"):
            # LSAR's own spelling of the venue ("AERA Open" -> "AERA_OPEN").
            rg["venue"] = str(gate["venue"])
        if writer_model:
            rg["revision_model"] = writer_model
    else:
        rg["enabled"] = False
    if home is not None:
        cfg = _expand_lsar_home(cfg, str(home))

    # ---- paths ----------------------------------------------------------------
    cfg_paths = cfg.setdefault("paths", {}) or {}
    cfg["paths"] = cfg_paths
    cfg_paths["raw_data"] = str(raw_data_dir(settings)) + os.sep
    cfg_paths["output_base"] = str(studies_dir(settings)) + os.sep
    memory = cfg.setdefault("findings_memory", {}) or {}
    cfg["findings_memory"] = memory
    memory["path"] = str((Path(paths.data_dir()) / "findings_memory" / "memory.yaml").absolute())

    # ---- execution --------------------------------------------------------------
    sandbox = cfg.setdefault("sandbox", {}) or {}
    cfg["sandbox"] = sandbox
    sandbox["enabled"] = False
    rscript = _sget(settings, "r.rscript")
    if rscript:
        cfg.setdefault("r_bridge", {})["rscript_path"] = str(rscript)

    # ---- the study itself ----------------------------------------------------------
    pipeline = cfg.setdefault("pipeline", {}) or {}
    cfg["pipeline"] = pipeline
    pipeline["task_type"] = str(plan.task_type)
    budget = _sget(settings, "defaults.budget_usd")
    if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0:
        pipeline["cost_budget_usd"] = float(budget)
    writer = cfg.setdefault("writer", {}) or {}
    cfg["writer"] = writer
    writer["venue_format"] = "journal" if getattr(plan, "paper_format", "conference") == "journal" else "conference"
    author = _sget(settings, "author.name")
    if isinstance(author, str) and author.strip():
        cfg.setdefault("paper", {})["authors"] = [author.strip(), "EDM-ARS"]
    mailto = _sget(settings, "literature.crossref_mailto")
    if mailto:
        cfg.setdefault("semantic_scholar", {})["crossref_mailto"] = str(mailto)
    return cfg


# ---------------------------------------------------------------------------
# Child environment
# ---------------------------------------------------------------------------


def _provider_env_var(provider: str) -> str:
    try:
        from edmars import providers

        info = providers.PROVIDERS.get(provider)
        env_var = getattr(info, "env_var", None) if info is not None else None
        if isinstance(env_var, str) and env_var:
            return env_var
    except (ImportError, AttributeError):
        pass
    return _PROVIDER_ENV.get(provider, "DEEPSEEK_API_KEY")


def _latex_bin_dir(settings: dict[str, Any]) -> str | None:
    if str(_sget(settings, "latex.mode", "") or "") == "none":
        return None
    try:
        from edmars import toolchain

        return toolchain.latex_bin_dir(settings)
    except Exception:  # noqa: BLE001 -- PATH as it is still works for a system TeX
        return None


def _rscript(settings: dict[str, Any]) -> str | None:
    """The Rscript setup saved. Without one the pipeline's own R search
    (src/r_bridge.py: newest version first) takes over."""
    saved = _sget(settings, "r.rscript")
    return str(saved) if saved else None


def child_env(
    settings: dict[str, Any],
    *,
    provider: str,
    review: bool,
    run_id: str,
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Environment for the pipeline process.

    Starts from the current environment, adds the keys this study needs
    from the keychain, forces UTF-8, keeps user site-packages and a
    foreign PYTHONPATH out, and puts this interpreter's folder first on
    PATH so any ``python`` the pipeline starts is this one.
    """
    env = dict(os.environ if base_env is None else base_env)
    names: list[str] = [_provider_env_var(provider)]
    if review:
        names.append("DEEPSEEK_API_KEY")  # LSAR scoring is calibrated on DeepSeek
    names.extend(_OPTIONAL_KEYS)
    seen: list[str] = []
    for name in names:
        if name not in seen:
            seen.append(name)
    env.update({k: v for k, v in edsecrets.child_secrets(seen).items() if v})
    if provider == "local" and not env.get("OPENAI_API_KEY"):
        env["OPENAI_API_KEY"] = "local"  # OpenAI-compatible servers want a non-empty key
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONNOUSERSITE"] = "1"
    for var in ("PYTHONPATH", "PYTHONHOME"):
        for key in [k for k in env if k.upper() == var]:
            env.pop(key, None)
    path_key = next((k for k in env if k.upper() == "PATH"), "PATH")
    exe_dir = str(Path(sys.executable).parent)
    first = [exe_dir]
    tex_dir = _latex_bin_dir(settings)
    if tex_dir:
        # The pipeline runs a bare ``pdflatex``; a TinyTeX install (or a TeX
        # found outside PATH) is only visible to it through PATH.
        first.append(tex_dir)
    current = env.get(path_key, "")
    parts = [p for p in current.split(os.pathsep) if p]
    env[path_key] = os.pathsep.join(first + [p for p in parts if p not in first])
    rscript = _rscript(settings)
    if rscript:
        env["EDM_ARS_RSCRIPT"] = str(rscript)
    base_url = _sget(settings, "provider_base_url")
    if provider == "local" and base_url:
        # OPENAI_BASE_URL in the environment beats the config's base_url
        # (defect E3), so a stray one would send a local study elsewhere.
        env["OPENAI_BASE_URL"] = str(base_url)
    home = _lsar_home(settings)
    if review and home is not None:
        env["LSAR_HOME"] = str(home)
    env["EDMARS_RUN_ID"] = run_id
    return {str(k): str(v) for k, v in env.items()}


# ---------------------------------------------------------------------------
# Run folders
# ---------------------------------------------------------------------------


def slugify(text: str, max_len: int = 40) -> str:
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    slug = ""
    for word in words:
        candidate = f"{slug}-{word}" if slug else word
        if len(candidate) > max_len:
            break
        slug = candidate
    return slug


def _new_run_dir(parent: Path, plan: "StudyPlan") -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    base = slugify(getattr(plan, "research_question", "") or "") \
        or slugify(getattr(plan, "example_id", "") or "") \
        or slugify(str(plan.task_type)) or "study"
    for _ in range(50):
        candidate = parent / f"{stamp}_{base}_{os.urandom(2).hex()}"
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise RunnerError(f"Could not create a new study folder in {parent}.")


def _plan_dataset(plan: "StudyPlan") -> str:
    spec = getattr(plan, "spec", None)
    if isinstance(spec, dict) and spec.get("dataset"):
        return str(spec["dataset"])  # the spec's dataset always wins
    return str(plan.dataset)


def _plan_prompt(plan: "StudyPlan") -> str | None:
    spec = getattr(plan, "spec", None)
    if isinstance(spec, dict) and spec.get("research_question"):
        return str(spec["research_question"])
    prompt = getattr(plan, "prompt", None) or getattr(plan, "research_question", None)
    return str(prompt) if prompt else None


def build_argv(plan: "StudyPlan", run_dir: Path) -> list[str]:
    argv = [
        sys.executable, "-m", "src.main",
        "--config", str(run_dir / "run_config.yaml"),
        "--output-dir", str(run_dir),
        "--dataset", _plan_dataset(plan),
    ]
    if isinstance(getattr(plan, "spec", None), dict):
        argv += ["--research-spec", str(run_dir / "research_spec.locked.json")]
    prompt = _plan_prompt(plan)
    if prompt:
        argv += ["--prompt", prompt]
    return argv


def _study_summary(settings: dict[str, Any], plan: "StudyPlan") -> dict[str, Any]:
    wanted = bool(getattr(plan, "review", False))
    enabled = review_enabled(settings, plan)
    return {
        "task_type": str(plan.task_type),
        "dataset": _plan_dataset(plan),
        "research_question": str(getattr(plan, "research_question", "") or _plan_prompt(plan) or ""),
        "example_id": getattr(plan, "example_id", None),
        "experimental": bool(getattr(plan, "experimental", False)),
        "venue": str(getattr(plan, "venue", "EDM")),
        "paper_format": str(getattr(plan, "paper_format", "conference")),
        "review": enabled,
        "review_requested": wanted,
        # The study asked for automated review but LSAR is not set up:
        # the end screen says "not reviewed" instead of staying silent.
        "review_unavailable": wanted and not enabled,
        "provider": str(_sget(settings, "provider", "deepseek")),
    }


def prepare_run(settings: dict[str, Any], plan: "StudyPlan") -> Path:
    """Create a fresh study folder with run_config.yaml, the locked spec
    (if any) and runner.json. Nothing is started."""
    parent = studies_dir(settings)
    parent.mkdir(parents=True, exist_ok=True)
    run_dir = _new_run_dir(parent, plan)
    cfg = build_effective_config(settings, plan)
    _write_text_atomic(run_dir / "run_config.yaml",
                       yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    spec = getattr(plan, "spec", None)
    if isinstance(spec, dict):
        _write_json(run_dir / "research_spec.locked.json", spec)
    runner = {
        "schema": 1,
        "version": _app_version(),
        "argv": build_argv(plan, run_dir),
        "pid": None,
        "create_time": None,
        "created_at": _utc_now(),
        "started_at": None,
        "python": sys.executable,
        "app_root": str(Path(paths.app_root()).absolute()),
        "study": _study_summary(settings, plan),
    }
    _write_json(run_dir / "runner.json", runner)
    return run_dir


# ---------------------------------------------------------------------------
# The pipeline's own pre-flight
# ---------------------------------------------------------------------------

#: What the pipeline's findings are called on the check list.
_FINDING_TITLES = {
    "KEY_MISSING": "AI service key",
    "SDK_MISSING": "Python packages",
    "PROVIDER_CONFIG_INVALID": "AI models",
    "INSTALL_INCOMPLETE": "Installation",
    "DATA_MISSING": "Dataset file",
    "LATEX_MISSING": "PDF typesetting",
    "R_MISSING": "R",
    "R_PACKAGES_MISSING": "R packages",
    "LSAR_NOT_FOUND": "Automated reviewer",
    "LSAR_IMPORT_FAILED": "Automated reviewer",
    "LSAR_KEY_MISSING": "Automated reviewer key",
}


#: The edmars command that fixes a finding. The pipeline's own fix text
#: speaks to people who run ``python -m src.main`` (".env in the repository
#: folder"), which is wrong advice for an installed copy.
_FINDING_FIXES = {
    "KEY_MISSING": "Run `edmars setup ai` to add the key.",
    "PROVIDER_CONFIG_INVALID": "Run `edmars setup ai` (and `edmars setup advanced` for per-step models).",
    "SDK_MISSING": "Reinstall EDM-ARS, then run `edmars doctor`.",
    "INSTALL_INCOMPLETE": "Reinstall EDM-ARS, then run `edmars doctor`.",
    "DATA_MISSING": "Run `{get_data}`.",
    "LATEX_MISSING": "Run `edmars setup pdf`.",
    "R_MISSING": "Run `edmars setup r`.",
    "R_PACKAGES_MISSING": "Run `edmars setup r`.",
    "LSAR_NOT_FOUND": "Run `edmars setup reviewer`.",
    "LSAR_IMPORT_FAILED": "Run `edmars setup reviewer`.",
    "LSAR_KEY_MISSING": "Run `edmars setup reviewer` to add a DeepSeek key.",
}


#: Findings after which the pipeline skips the review gate ("the review
#: gate will not run"): a study that asked for the review goes ahead
#: without one, and the confirmation card has to say so.
REVIEW_OFF_CODES = frozenset({"LSAR_NOT_FOUND", "LSAR_IMPORT_FAILED"})


#: Provider ids as people know them.
_PROVIDER_NAMES = {
    "deepseek": "DeepSeek",
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "minimax": "MiniMax",
}


def _plain_finding(code: str, message: str) -> str:
    """The pipeline's finding in words for a non-programmer.

    The pipeline names its internal step ids ("problem_formulator,
    data_engineer, ...") and environment variables; the user only needs
    to know which service's key or setting is missing.
    """
    from edmars.wizard import STAGE_LABELS

    if code == "KEY_MISSING":
        match = re.search(r"the (\w+) provider needs it for: (.+?)\.?\s*$", message)
        if match:
            provider = _PROVIDER_NAMES.get(match.group(1), match.group(1))
            steps = [s.strip() for s in match.group(2).split(",") if s.strip()]
            labels = [STAGE_LABELS.get(s, s.replace("_", " ")) for s in steps]
            return (
                f"No {provider} key is saved on this computer, and the study "
                f"needs one for {len(labels)} step(s): {', '.join(labels)}."
            )
    if code == "LSAR_KEY_MISSING":
        return (
            "No DeepSeek key is saved on this computer. The automated reviewer "
            "always uses DeepSeek, so the review would fail."
        )
    if code == "PROVIDER_CONFIG_INVALID":
        match = re.search(r"setting for (\w+) is not valid: (.+)$", message)
        if match:
            step = STAGE_LABELS.get(match.group(1), match.group(1).replace("_", " "))
            return f"The AI model setting for the step '{step}' is not valid: {match.group(2)}"
    return message


def pipeline_check(settings: dict[str, Any], plan: "StudyPlan", *,
                   timeout_s: float = 300) -> list[Any]:
    """Run the pipeline's own ``--dry-run`` with exactly the config, spec and
    environment a launch would use, and return its findings as checks.

    This is the check the pipeline makes at start-up (keys, models the
    agents would refuse, the data file, R for measurement studies, LSAR).
    Running it here means a study that would stop in its first second is
    refused before a study folder exists, with the pipeline's own words.
    Nothing is created outside a temporary folder, and nothing is sent.
    """
    import tempfile

    from edmars.model import Check

    root = Path(paths.app_root()).absolute()
    with tempfile.TemporaryDirectory(prefix="edmars-check-") as tmp:
        work = Path(tmp)
        cfg = build_effective_config(settings, plan)
        _write_text_atomic(work / "run_config.yaml",
                           yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
        argv = build_argv(plan, work / "run")
        # build_argv points --config / --research-spec into the run folder;
        # here they live next to it, and the run folder is never created.
        fixed: list[str] = []
        for i, token in enumerate(argv):
            if i and argv[i - 1] == "--config":
                token = str(work / "run_config.yaml")
            elif i and argv[i - 1] == "--research-spec":
                token = str(work / "research_spec.locked.json")
            fixed.append(token)
        spec = getattr(plan, "spec", None)
        if isinstance(spec, dict):
            _write_json(work / "research_spec.locked.json", spec)
        fixed += ["--dry-run", "--json-summary", "--quiet"]
        env = child_env(
            settings,
            provider=str(_sget(settings, "provider", "deepseek")),
            review=review_enabled(settings, plan),
            run_id="preflight",
        )
        try:
            result = proc.run(fixed, timeout=timeout_s, env=env, cwd=root)
        except Exception as exc:  # noqa: BLE001 -- reported as a check
            return [Check("Pipeline check", "fail",
                          f"The pipeline's own check could not run: {edsecrets.redact(str(exc))}",
                          "Run `edmars doctor`.")]
    summary: dict[str, Any] = {}
    for line in reversed((result.stdout or "").strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                summary = json.loads(line)
            except ValueError:
                summary = {}
            break
    if not summary:
        tail = [ln for ln in (result.stderr or "").strip().splitlines() if ln.strip()]
        detail = edsecrets.redact(tail[-1].strip()) if tail else f"exit code {result.returncode}"
        return [Check("Pipeline check", "fail",
                      f"The pipeline could not check this study: {detail}",
                      "Run `edmars doctor`; if it finds nothing, run "
                      "`edmars doctor --bundle` and attach the file to an issue.")]
    checks: list[Any] = []
    for item in summary.get("checks") or []:
        if not isinstance(item, dict):
            continue
        code = str(item.get("code") or "")
        severity: Literal["fail", "warn"] = "fail" if str(item.get("severity")) == "fail" else "warn"
        fix = _FINDING_FIXES.get(code)
        if fix:
            from edmars.datasets import get_command

            dataset = _plan_dataset(plan)
            fix = fix.format(dataset=dataset, get_data=get_command(dataset))
        else:
            fix = str(item.get("fix") or "")
        checks.append(Check(
            _FINDING_TITLES.get(code, code.replace("_", " ").capitalize() or "Pipeline check"),
            severity,
            edsecrets.redact(_plain_finding(code, str(item.get("message") or code))),
            edsecrets.redact(fix) or None,
            code=code or None,
        ))
    if not checks:
        checks.append(Check("Pipeline check", "ok", "The pipeline's own start-up check passed"))
    return checks


# ---------------------------------------------------------------------------
# The one-active-run lock
# ---------------------------------------------------------------------------


def _lock_holder() -> dict[str, Any]:
    return _read_json(_lock_path())


def active_run() -> Path | None:
    """The study that is running now, or None. Clears a stale lock."""
    path = _lock_path()
    data = _lock_holder()
    if not data:
        if path.exists():
            # Unreadable (half-written) lock: leave it if it is brand new.
            try:
                if datetime.now().timestamp() - path.stat().st_mtime > 60:
                    path.unlink()
            except OSError:
                pass
        return None
    pid = data.get("pid")
    alive = process_alive(int(pid), data.get("create_time")) if isinstance(pid, int) else False
    if alive:
        run_dir = data.get("run_dir")
        return Path(run_dir) if run_dir else None
    try:
        path.unlink()
    except OSError:
        pass
    return None


def _acquire_lock(run_dir: Path) -> None:
    path = _lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(3):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            holder = active_run()
            if holder is not None and Path(holder) != Path(run_dir):
                raise RunnerError(
                    f"Another study is still running ({Path(holder).name}). One study runs "
                    "at a time: wait for it to finish, or stop it with: edmars stop"
                ) from None
            if holder is not None:
                raise RunnerError("This study is already running.") from None
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({
                "pid": os.getpid(),
                "create_time": _create_time(os.getpid()),
                "run_dir": str(run_dir),
                "state": "starting",
                "at": _utc_now(),
            }, fh)
        return
    raise RunnerError(f"Could not take the run lock at {path}; remove it if no study is running.")


def _set_lock(run_dir: Path, pid: int, create_time: float | None) -> None:
    _write_json(_lock_path(), {
        "pid": pid,
        "create_time": create_time,
        "run_dir": str(run_dir),
        "state": "running",
        "at": _utc_now(),
    })


def _release_lock(run_dir: Path) -> None:
    data = _lock_holder()
    if data and data.get("run_dir") and Path(data["run_dir"]) != Path(run_dir):
        return
    try:
        _lock_path().unlink()
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Launch / stop / resume
# ---------------------------------------------------------------------------


def _spawn(run_dir: Path, argv: list[str], env: dict[str, str], settings: dict[str, Any]) -> tuple[int, float | None]:
    root = Path(paths.app_root()).absolute()
    try:
        pid = int(proc.spawn_detached(argv, cwd=root, env=env, log_path=run_dir / "console.log"))
    except Exception as exc:  # noqa: BLE001
        _release_lock(run_dir)
        raise RunnerError(f"Could not start the study: {edsecrets.redact(str(exc))}") from exc
    create_time = _create_time(pid)
    _set_lock(run_dir, pid, create_time)
    if _sget(settings, "defaults.keep_awake", True):
        try:
            proc.keep_awake(pid)
        except Exception:  # noqa: BLE001 -- staying awake is a convenience
            pass
    return pid, create_time


def launch(settings: dict[str, Any], plan: "StudyPlan") -> Path:
    """Create the study folder and start the pipeline in the background."""
    holder = active_run()
    if holder is not None:
        raise RunnerError(
            f"Another study is still running ({holder.name}). One study runs at a time: "
            "wait for it to finish, or stop it with: edmars stop"
        )
    run_dir = prepare_run(settings, plan)
    try:
        _acquire_lock(run_dir)
    except RunnerError:
        # Another launch won the race. The folder we just made holds only
        # our own config files; do not leave an empty study behind.
        shutil.rmtree(run_dir, ignore_errors=True)
        raise
    runner = _read_runner(run_dir)
    env = child_env(
        settings,
        provider=str(_sget(settings, "provider", "deepseek")),
        review=bool(as_dict(runner.get("study")).get("review")),
        run_id=run_dir.name,
    )
    pid, create_time = _spawn(run_dir, list(runner["argv"]), env, settings)
    runner.update({"pid": pid, "create_time": create_time, "started_at": _utc_now()})
    _write_json(run_dir / "runner.json", runner)
    return run_dir


def stop(run_dir: Path | str) -> None:
    """Ask the study to stop, then end its process tree after a grace period.

    The STOP file asks the pipeline to wind down and save its state (on
    macOS/Linux it also gets SIGTERM). Whatever is still running 30 s
    later is ended. Finished steps stay on disk; ``resume`` continues from
    the last one.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise RunnerError(f"There is no study folder at {run_dir}.")
    runner = _read_runner(run_dir)
    pid = runner.get("pid")
    if not (isinstance(pid, int) and process_alive(pid, runner.get("create_time"))):
        # Never signal a pid that no longer belongs to this study.
        _release_lock(run_dir)
        raise RunnerError("This study is not running, so there is nothing to stop.")
    _write_text_atomic(run_dir / "STOP", _utc_now() + "\n")
    proc.terminate_tree(pid, grace_s=30)
    if runner:
        runner["stopped_at"] = _utc_now()
        runner["stopped_by_user"] = True
        _write_json(run_dir / "runner.json", runner)
    _release_lock(run_dir)


#: What a study folder may say about its dataset, venue and failed step.
#: Anything else is ignored: the folder is data, not instructions.
_SAFE_DATASET = re.compile(r"[a-z0-9][a-z0-9_]{0,63}")
_SAFE_VENUE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _-]{0,39}")
_PIPELINE_STEPS = ("FORMULATING", "ENGINEERING", "ANALYZING", "CRITIQUING", "REVISING",
                   "WRITING", "REVIEWING", "VERIFYING")


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        return as_dict(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (OSError, yaml.YAMLError):
        return {}


def _argv_value(argv: Any, flag: str) -> str | None:
    """The value given for ``flag`` in a saved argv (``--flag V`` or ``--flag=V``)."""
    if not isinstance(argv, list):
        return None
    tokens = [str(t) for t in argv]
    for i, token in enumerate(tokens):
        if token == flag and i + 1 < len(tokens):
            return tokens[i + 1]
        if token.startswith(flag + "="):
            return token[len(flag) + 1:]
    return None


def _resume_study(run_dir: Path, runner: dict[str, Any]) -> dict[str, Any]:
    """The study's own choices, read from the folder and checked.

    A study folder can come from anywhere (a shared drive, a colleague),
    so only plain values are taken from it, each checked against what
    edmars itself writes; anything else falls back to a safe default.
    The pipeline takes the dataset, study type and spec of a resumed run
    from its checkpoint anyway.
    """
    from edmars.model import TASK_TYPES

    study = as_dict(runner.get("study"))
    old_cfg = _read_yaml(run_dir / "run_config.yaml")
    checkpoint = _read_json(run_dir / "checkpoint.json")
    gate = as_dict(old_cfg.get("review_gate"))

    task_type = str(study.get("task_type") or checkpoint.get("task_type")
                    or as_dict(old_cfg.get("pipeline")).get("task_type") or "")
    if task_type not in TASK_TYPES:
        task_type = "prediction"
    dataset = str(study.get("dataset") or checkpoint.get("dataset_name")
                  or _argv_value(runner.get("argv"), "--dataset") or "")
    venue = str(study.get("venue") or gate.get("venue") or "EDM")
    if not _SAFE_VENUE.fullmatch(venue):
        venue = "EDM"
    paper_format = study.get("paper_format") or as_dict(old_cfg.get("writer")).get("venue_format")
    if "review_requested" in study:
        review = study.get("review_requested") is True
    elif "review" in study:
        review = study.get("review") is True
    else:
        review = gate.get("enabled") is True
    prompt = _argv_value(runner.get("argv"), "--prompt")
    return {
        "task_type": task_type,
        "dataset": dataset if _SAFE_DATASET.fullmatch(dataset) else None,
        "venue": venue,
        "paper_format": "journal" if paper_format == "journal" else "conference",
        "review": review,
        "prompt": prompt if prompt and prompt.strip() else None,
    }


def _resume_argv(run_dir: Path, study: dict[str, Any]) -> list[str]:
    """The pipeline command for a resumed study, built here, never read
    from the folder: runner.json's saved argv could name any program."""
    argv = [sys.executable, "-m", "src.main",
            "--config", str(run_dir / "run_config.yaml"),
            "--output-dir", str(run_dir)]
    if study.get("dataset"):
        argv += ["--dataset", str(study["dataset"])]
    locked = run_dir / "research_spec.locked.json"
    if locked.is_file():
        argv += ["--research-spec", str(locked)]
    if study.get("prompt"):
        # One token, so a question that starts with "-" is not read as a flag.
        argv.append("--prompt=" + str(study["prompt"]))
    return argv


def _refresh_run_config(run_dir: Path, settings: dict[str, Any], study: dict[str, Any]) -> dict[str, Any]:
    """Write run_config.yaml again from this computer's settings.

    The folder's copy could point the pipeline at another Python, another
    Rscript, another LSAR (imported into the process that holds the keys)
    or another server to send the keys to. The replaced copy is kept as
    run_config.previous.yaml when it differs.
    """
    from edmars.model import StudyPlan

    plan = StudyPlan(
        task_type=str(study["task_type"]),
        dataset=str(study.get("dataset") or ""),
        research_question=str(study.get("prompt") or ""),
        venue=str(study["venue"]),
        paper_format=str(study["paper_format"]),
        review=bool(study["review"]),
    )
    cfg = build_effective_config(settings, plan)
    path = run_dir / "run_config.yaml"
    text = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
    try:
        old = path.read_text(encoding="utf-8")
    except OSError:
        old = None
    if old is not None and old != text:
        _write_text_atomic(run_dir / "run_config.previous.yaml", old)
    _write_text_atomic(path, text)
    return cfg


def retry_stage_support(app_root: Path | None = None) -> tuple[bool, bool]:
    """(supported, takes_a_value) for ``src.main --retry-stage``.

    Read from the pipeline's own argument definitions; falls back to its
    ``--help`` text when the source cannot be read.
    """
    root = Path(app_root or paths.app_root())
    text = ""
    try:
        text = (root / "src" / "main.py").read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = ""
    if text:
        idx = text.find('"--retry-stage"')
        if idx < 0:
            idx = text.find("'--retry-stage'")
        if idx < 0:
            return False, False
        window = text[idx: idx + 600]
        end = window.find("add_argument(", 1)
        window = window if end < 0 else window[:end]
        return True, "store_true" not in window
    try:
        result = proc.run([sys.executable, "-m", "src.main", "--help"], timeout=120, cwd=root)
    except Exception:  # noqa: BLE001
        return False, False
    help_text = result.stdout or ""
    m = re.search(r"--retry-stage(?:[ =]([A-Z_\[]\S*))?", help_text)
    if not m:
        return False, False
    return True, bool(m.group(1))


def _failed_stage(run_dir: Path) -> str | None:
    status = _read_json(run_dir / "run_status.json")
    abort = status.get("abort") if isinstance(status.get("abort"), dict) else None
    if abort and abort.get("stage"):
        return str(abort["stage"])
    state = load_state(run_dir)
    if isinstance(state.abort, dict) and state.abort.get("stage"):
        return str(state.abort["stage"])
    for st in state.stages:
        if st.status == "failed":
            return st.key
    checkpoint = _read_json(run_dir / "checkpoint.json")
    done = set(checkpoint.get("completed_stages") or [])
    for key in ("FORMULATING", "ENGINEERING", "ANALYZING", "CRITIQUING", "WRITING", "REVIEWING", "VERIFYING"):
        if key not in done:
            return key
    return None


def resume(run_dir: Path | str) -> None:
    """Start a stopped study again from its last finished step.

    The command and run_config.yaml are rebuilt from this computer's
    settings; the folder only supplies plain, checked values (study type,
    dataset name, venue, the research question), because a study folder
    can come from someone else.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise RunnerError(f"There is no study folder at {run_dir}.")
    runner = _read_runner(run_dir)
    pid = runner.get("pid")
    if isinstance(pid, int) and process_alive(pid, runner.get("create_time")):
        raise RunnerError("This study is still running. Watch it with: edmars status")

    state = load_state(run_dir)
    if state.final_state in ("COMPLETED", "INCOMPLETE"):
        raise RunnerError(
            "This study already finished, so there is nothing to resume. "
            "See it with: edmars results   Start another with: edmars new"
        )
    study = _resume_study(run_dir, runner)
    argv = _resume_argv(run_dir, study)
    argv.append("--resume")

    if state.final_state == "ABORTED":
        from edmars import endstates

        outcome = endstates.classify(run_dir)
        if outcome.resumable is False:
            raise RunnerError(
                f"This study cannot be resumed: {outcome.title}. {outcome.fix}".strip()
            )
        # This install's pipeline is the one that runs, whatever
        # runner.json says the study was started with.
        supported, takes_value = retry_stage_support(Path(paths.app_root()))
        if not supported:
            raise RunnerError(
                "This study stopped with an error, and this version of the pipeline "
                "cannot retry a failed step. Fix the cause, then start a new study "
                "with: edmars new"
            )
        argv.append("--retry-stage")
        if takes_value:
            stage = _failed_stage(run_dir)
            if stage in _PIPELINE_STEPS:
                argv.append(stage)

    settings = _load_settings()
    holder = active_run()
    if holder is not None and Path(holder) != run_dir:
        raise RunnerError(
            f"Another study is still running ({Path(holder).name}). Wait for it to "
            "finish, or stop it with: edmars stop"
        )
    _acquire_lock(run_dir)
    try:
        config = _refresh_run_config(run_dir, settings, study)
    except (OSError, yaml.YAMLError, ValueError) as exc:
        _release_lock(run_dir)
        raise RunnerError(f"Could not write the study's settings file: {exc}") from exc
    review = bool(as_dict(config.get("review_gate")).get("enabled"))
    provider = str(_sget(settings, "provider", "deepseek"))
    env = child_env(settings, provider=provider, review=review, run_id=run_dir.name)

    # Keep the earlier console output instead of letting the new process
    # overwrite it.
    console = run_dir / "console.log"
    if console.exists():
        n = 1
        while (run_dir / f"console.{n}.log").exists():
            n += 1
        try:
            console.rename(run_dir / f"console.{n}.log")
        except OSError:
            pass
    try:
        (run_dir / "STOP").unlink()
    except OSError:
        pass

    new_pid, create_time = _spawn(run_dir, argv, env, settings)
    runner = runner or {"schema": 1, "created_at": _utc_now(), "study": {
        "task_type": study["task_type"], "dataset": study.get("dataset"),
        "venue": study["venue"], "paper_format": study["paper_format"]}}
    runner_study = as_dict(runner.get("study"))
    runner_study["provider"] = provider  # the service the resumed part uses
    runner_study["review"] = review
    runner["study"] = runner_study
    now = _utc_now()
    runner.update({
        "argv": argv,
        "pid": new_pid,
        "create_time": create_time,
        "resumed_at": now,
        "python": sys.executable,
        "app_root": runner.get("app_root") or str(Path(paths.app_root()).absolute()),
    })
    runner.setdefault("started_at", now)
    runner.setdefault("resumes", []).append(now)
    runner.pop("stopped_by_user", None)
    runner.pop("stopped_at", None)
    _write_json(run_dir / "runner.json", runner)


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------


def _is_run_dir(path: Path) -> bool:
    return path.is_dir() and any(
        (path / name).exists() for name in ("runner.json", "checkpoint.json", "pipeline.log", "events.jsonl")
    )


def _started(path: Path) -> datetime:
    runner = _read_runner(path)
    for key in ("started_at", "created_at"):
        ts = parse_ts(runner.get(key))
        if ts is not None:
            return ts
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
    except OSError:
        return datetime.fromtimestamp(0, timezone.utc)


def _candidates(settings: dict[str, Any]) -> list[Path]:
    root = studies_dir(settings)
    try:
        entries: Iterable[Path] = list(root.iterdir())
    except OSError:
        return []
    runs = [p for p in entries if _is_run_dir(p)]
    runs.sort(key=_started, reverse=True)
    return runs


def list_runs(settings: dict[str, Any]) -> list[dict[str, Any]]:
    """Every study folder, newest first, with its current label."""
    from edmars import endstates

    active = active_run()
    out: list[dict[str, Any]] = []
    for path in _candidates(settings):
        runner = _read_runner(path)
        study = as_dict(runner.get("study"))
        try:
            outcome = endstates.classify(path)
            label, kind = outcome.label, outcome.kind
        except Exception:  # noqa: BLE001 -- one damaged folder must not hide the rest
            label, kind = "Unreadable", "stopped"
        question = study.get("research_question")
        if not question:
            spec = _read_json(path / "research_spec.json")
            question = spec.get("research_question")
        out.append({
            "path": str(path),
            "name": path.name,
            "started_at": _started(path).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "question": question or "",
            "task_type": study.get("task_type") or _read_json(path / "checkpoint.json").get("task_type") or "",
            "dataset": study.get("dataset") or "",
            "label": label,
            "kind": kind,
            "active": active is not None and Path(active) == path,
        })
    return out


def latest_run(settings: dict[str, Any]) -> Path | None:
    """The running study if there is one, else the most recent one."""
    active = active_run()
    if active is not None:
        return active
    runs = _candidates(settings)
    return runs[0] if runs else None


__all__ = [
    "REVIEW_OFF_CODES",
    "RunnerError",
    "active_run",
    "build_argv",
    "build_effective_config",
    "child_env",
    "latest_run",
    "launch",
    "pipeline_check",
    "list_runs",
    "prepare_run",
    "resume",
    "retry_stage_support",
    "review_enabled",
    "slugify",
    "stop",
    "studies_dir",
]
