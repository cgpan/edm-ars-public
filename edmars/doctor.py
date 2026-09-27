"""`edmars doctor`: check that everything a study needs is in place.

Every check returns an ``edmars.model.Check`` (ok / warn / fail / info) with
a plain-English detail and, where something can be done, the exact fix.
A check never prints, never prompts and never shows a secret: key checks
report WHERE a key comes from (environment, credential store, file), never
its value. ``--deep`` adds the checks that cost a network round trip or a
few minutes (live key checks, retired-model detection, a LaTeX test
compile).

``make_bundle`` writes a support zip a user can attach to an issue:
``doctor --json``, the settings with personal fields and home paths
removed, version information, and the latest study's ``pipeline.log``,
``run_status.json`` and ``console.log``. Everything written into it passes
through ``edmars.secrets.redact`` plus a pattern and exact-value scrub.
It never contains ``prompts/`` or any CSV.

The modules this one uses (settings, secrets, providers, datasets,
toolchain, lsar, paths, proc, disclosure, ui) are imported lazily inside
functions, so importing ``edmars.doctor`` is cheap and has no side effects.
Nothing here starts a process directly; tool lookups go through
``edmars.proc`` or through the toolchain module.
"""

from __future__ import annotations

import contextlib
import copy
import importlib
import importlib.util
import json
import os
import platform
import re
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only
    from edmars.model import Check

#: Oldest Python the pipeline supports, and the newest one it was tested on.
MIN_PYTHON: tuple[int, int] = (3, 11)
TESTED_PYTHON_MAX: tuple[int, int] = (3, 12)

#: Memory below this gets a warning (large studies can run out of memory).
RAM_RECOMMENDED_GB: float = 16.0

#: Free-space thresholds for the studies and data folders.
DISK_FAIL_GB: float = 1.0
DISK_WARN_GB: float = 5.0
#: What the HSLS:09 download needs while it is being unpacked.
HSLS_DOWNLOAD_GB: float = 2.5

#: Import names the pipeline and the CLI cannot run without.
CORE_PACKAGES: tuple[str, ...] = (
    "numpy", "pandas", "sklearn", "scipy", "xgboost", "shap", "matplotlib",
    "yaml", "requests", "openai", "anthropic", "typer", "rich", "questionary",
    "keyring", "platformdirs", "psutil",
)

#: Distribution names reported in the support bundle's versions.json.
VERSION_PACKAGES: tuple[str, ...] = (
    "numpy", "pandas", "scikit-learn", "scipy", "xgboost", "shap", "dowhy",
    "matplotlib", "PyYAML", "requests", "openai", "anthropic", "typer", "rich",
    "questionary", "keyring", "platformdirs", "psutil", "pymupdf",
)

SEMANTIC_SCHOLAR_ENV = "SEMANTIC_SCHOLAR_API_KEY"
DEEPSEEK_ENV = "DEEPSEEK_API_KEY"

#: Every secret EDM-ARS may hold. Their values are scrubbed from the bundle.
KNOWN_SECRET_NAMES: tuple[str, ...] = (
    "DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
    "SEMANTIC_SCHOLAR_API_KEY", "TAVILY_API_KEY", "MINIMAX_API_KEY",
)

#: The only files taken from a study folder into a support bundle.
BUNDLE_RUN_FILES: tuple[str, ...] = ("pipeline.log", "run_status.json", "console.log")
#: Each bundled log keeps at most its last this-many bytes.
BUNDLE_MAX_BYTES: int = 2_000_000

#: Fallback provider facts, used when ``edmars.providers.PROVIDERS`` lacks a
#: field. The providers module is the source of truth (CLI_SPEC section 4).
PROVIDER_FALLBACK: dict[str, dict[str, str | None]] = {
    "deepseek": {
        "label": "DeepSeek",
        "env_var": "DEEPSEEK_API_KEY",
        "key_page": "https://platform.deepseek.com/api_keys",
        "topup": "https://platform.deepseek.com/top_up",
    },
    "openai": {
        "label": "OpenAI",
        "env_var": "OPENAI_API_KEY",
        "key_page": "https://platform.openai.com/api-keys",
        "topup": None,
    },
    "anthropic": {
        "label": "Anthropic",
        "env_var": "ANTHROPIC_API_KEY",
        "key_page": "https://platform.claude.com/settings/keys",
        "topup": None,
    },
    "local": {
        "label": "your own model server",
        "env_var": "OPENAI_API_KEY",
        "key_page": None,
        "topup": None,
    },
}

GLYPHS: dict[str, str] = {"ok": "✓", "warn": "!", "fail": "✗", "info": "·"}
PLAIN_GLYPHS: dict[str, str] = {"ok": "[ok]", "warn": "[!]", "fail": "[x]", "info": "[i]"}
_STYLES: dict[str, str] = {"ok": "green", "warn": "yellow", "fail": "red", "info": "dim"}


# ---------------------------------------------------------------------------
# Small shared helpers (the wizard uses these too)
# ---------------------------------------------------------------------------

def make_check(name: str, status: str, detail: str, fix: str | None = None) -> "Check":
    """Build an ``edmars.model.Check``; the one place this module creates one."""
    from edmars.model import Check

    return Check(name=name, status=status, detail=detail, fix=fix)  # type: ignore[arg-type]


def cfg(settings: dict[str, Any], dotted: str, default: Any = None) -> Any:
    """Read ``a.b.c`` from a settings dict without ever raising."""
    try:
        from edmars import settings as st

        value = st.get(settings, dotted, default)
    except Exception:
        value = settings
        for part in dotted.split("."):
            if not isinstance(value, dict) or part not in value:
                return default
            value = value[part]
    return default if value is None else value


def provider_meta(provider_id: str) -> dict[str, str | None]:
    """Label, key env var, key page and top-up page for a provider.

    Reads ``edmars.providers.PROVIDERS`` and fills anything it does not
    carry from :data:`PROVIDER_FALLBACK`. ``label`` is the short name used
    in sentences ("DeepSeek"); ``menu_label`` is the catalog label.
    """
    base: dict[str, str | None] = dict(PROVIDER_FALLBACK.get(provider_id, {
        "label": provider_id, "env_var": None, "key_page": None, "topup": None,
    }))
    info: Any = None
    try:
        from edmars import providers

        info = providers.PROVIDERS.get(provider_id)
    except Exception:
        info = None

    def pick(*names: str) -> str | None:
        for attr in names:
            if info is None:
                return None
            value = info.get(attr) if isinstance(info, dict) else getattr(info, attr, None)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    menu_label = pick("label", "name", "title") or base["label"]
    base["menu_label"] = menu_label
    if provider_id != "local" and menu_label:
        short = re.split(r"\s+\(|\s+[—-]\s+", menu_label, maxsplit=1)[0].strip()
        base["label"] = short or base["label"]
    base["env_var"] = pick("env_var", "env", "key_env", "env_name") or base["env_var"]
    base["key_page"] = pick("key_page", "key_url", "keys_url", "api_keys_url") or base["key_page"]
    base["topup"] = pick("topup_url", "top_up_url", "topup", "billing_url") or base["topup"]
    return base


def store_label(source: str | None, env_var: str | None = None) -> str:
    """Where a secret lives, in words a user understands. Never the value."""
    if source == "env":
        return f"the environment variable {env_var}" if env_var else "an environment variable"
    if source == "keyring":
        if sys.platform == "win32":
            return "Windows Credential Manager"
        if sys.platform == "darwin":
            return "your macOS Keychain"
        return "your system keyring"
    if source == "file":
        try:
            from edmars import paths

            where = paths.config_dir() / "secrets.env"
            return f"a private file only your user account can read ({where})"
        except Exception:
            return "a private file only your user account can read"
    return "nowhere yet"


def glyph(status: str, plain: bool) -> str:
    table = PLAIN_GLYPHS if plain else GLYPHS
    return table.get(status, PLAIN_GLYPHS["info"] if plain else GLYPHS["info"])


def check_to_dict(check: Any) -> dict[str, Any]:
    return {
        "name": str(getattr(check, "name", "")),
        "status": str(getattr(check, "status", "info")),
        "detail": str(getattr(check, "detail", "")),
        "fix": getattr(check, "fix", None),
    }


def _is_plain() -> bool:
    try:
        from edmars import ui

        return bool(ui.is_plain())
    except Exception:
        return True


def _out(text: str) -> None:
    """Print one literal line through the shared console (no markup)."""
    try:
        from edmars import ui

        ui.console.print(text, markup=False, highlight=False)
    except Exception:
        print(text)


def render_checks(checks: Sequence[Any], *, title: str | None = None) -> None:
    """Show checks as a table (rich) or as ``[ok] name: detail`` lines (plain).

    Status is never carried by colour alone: every line has a glyph.
    """
    plain = _is_plain()
    if plain:
        if title:
            _out(title)
        for chk in checks:
            status = str(getattr(chk, "status", "info"))
            _out(f"{glyph(status, True)} {chk.name}: {chk.detail}")
            if getattr(chk, "fix", None):
                _out(f"     To fix: {chk.fix}")
        return
    try:
        from rich.table import Table
        from rich.text import Text

        from edmars import ui

        table = Table(title=title, show_header=False, box=None, pad_edge=False,
                      expand=False, title_justify="left")
        table.add_column(width=1, no_wrap=True)
        table.add_column(style="bold", no_wrap=False, overflow="fold")
        table.add_column(overflow="fold")
        for chk in checks:
            status = str(getattr(chk, "status", "info"))
            body = Text(str(chk.detail))
            if getattr(chk, "fix", None):
                body.append("\n→ " + str(chk.fix), style="dim")
            table.add_row(Text(glyph(status, False), style=_STYLES.get(status, "")),
                          Text(str(chk.name)), body)
        ui.console.print(table)
    except Exception:
        for chk in checks:
            status = str(getattr(chk, "status", "info"))
            _out(f"{glyph(status, True)} {chk.name}: {chk.detail}")
            if getattr(chk, "fix", None):
                _out(f"     To fix: {chk.fix}")


def summarize(checks: Sequence[Any]) -> dict[str, int]:
    counts = {"ok": 0, "warn": 0, "fail": 0, "info": 0}
    for chk in checks:
        status = str(getattr(chk, "status", "info"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def _safe(label: str, fn: Callable[[], list["Check"]]) -> list["Check"]:
    """Run one check group; a crash becomes a warning, never a traceback."""
    try:
        return list(fn())
    except Exception as exc:  # noqa: BLE001 - doctor must always finish
        return [make_check(label, "warn",
                           f"This check could not run: {redact(_one_line(exc))}",
                           "Run `edmars doctor --bundle` and attach the file to an issue.")]


def _one_line(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()
    first = text[0] if text else ""
    name = type(exc).__name__
    return f"{name}: {first}"[:300] if first else name


def _expand(path: str | os.PathLike[str] | None) -> Path | None:
    if not path:
        return None
    return Path(os.path.expandvars(str(path))).expanduser()


def studies_dir(settings: dict[str, Any]) -> Path:
    configured = _expand(cfg(settings, "studies_dir", None))
    if configured is not None:
        return configured
    from edmars import paths

    return paths.default_studies_dir()


def _data_dir(settings: dict[str, Any]) -> Path:
    try:
        from edmars import datasets

        return Path(datasets.raw_data_dir(settings))
    except Exception:
        from edmars import paths

        return paths.data_dir()


def _existing_parent(path: Path) -> Path:
    probe = path
    while not probe.exists():
        if probe.parent == probe:
            break
        probe = probe.parent
    return probe


def _gb(n_bytes: float) -> float:
    return n_bytes / (1024 ** 3)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _os_label() -> str:
    system = platform.system() or sys.platform
    release = platform.release()
    if system == "Windows":
        try:
            build = sys.getwindowsversion().build  # type: ignore[attr-defined]
            release = "11" if build >= 22000 else release
        except Exception:
            pass
    elif system == "Darwin":
        system = "macOS"
        release = platform.mac_ver()[0] or release
    bits = "64-bit" if sys.maxsize > 2 ** 32 else "32-bit"
    machine = platform.machine() or "?"
    return f"{system} {release} ({bits}, {machine})"


def check_os() -> list["Check"]:
    label = _os_label()
    if sys.platform in ("win32", "darwin"):
        return [make_check("Computer", "ok", label)]
    if sys.platform.startswith("linux"):
        return [make_check("Computer", "ok", f"{label}. Linux is supported on a best-effort basis.")]
    return [make_check("Computer", "warn", f"{label}. This system has not been tested.")]


def check_python() -> list["Check"]:
    ver = sys.version_info
    shown = f"{ver.major}.{ver.minor}.{ver.micro}"
    in_env = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    where = f"{sys.executable}" + (" (EDM-ARS's own environment)" if in_env else "")
    if (ver.major, ver.minor) < MIN_PYTHON:
        return [make_check("Python", "fail",
                           f"Python {shown} is too old; EDM-ARS needs {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer. Running: {where}",
                           "Reinstall EDM-ARS with the installer; it brings its own Python.")]
    if (ver.major, ver.minor) > TESTED_PYTHON_MAX:
        return [make_check("Python", "warn",
                           f"Python {shown} is newer than the versions EDM-ARS was tested on "
                           f"({MIN_PYTHON[0]}.{MIN_PYTHON[1]}-{TESTED_PYTHON_MAX[0]}.{TESTED_PYTHON_MAX[1]}). Running: {where}",
                           "Reinstall EDM-ARS with the installer; it brings a tested Python.")]
    return [make_check("Python", "ok", f"Python {shown} at {where}")]


def check_packages() -> list["Check"]:
    missing = [name for name in CORE_PACKAGES if importlib.util.find_spec(name) is None]
    if missing:
        return [make_check("Python packages", "fail",
                           "Missing: " + ", ".join(missing),
                           "Reinstall EDM-ARS with the installer (developer checkout: "
                           "`pip install -r requirements.txt -r requirements-cli.txt`).")]
    return [make_check("Python packages", "ok", f"All {len(CORE_PACKAGES)} core packages are installed")]


def check_xgboost() -> list["Check"]:
    """XGBoost can be installed and still not load (on macOS: no OpenMP library)."""
    from edmars import toolchain

    return list(toolchain.xgboost_checks())


def check_app() -> list["Check"]:
    from edmars import paths

    root = paths.app_root()
    problems: list[str] = []
    if not (root / "src" / "main.py").is_file():
        problems.append("src/main.py is missing")
    config_path = root / "config.yaml"
    if not config_path.is_file():
        problems.append("config.yaml is missing")
    else:
        try:
            import yaml

            with open(config_path, encoding="utf-8") as fh:
                loaded = yaml.safe_load(fh)
            if not isinstance(loaded, dict):
                problems.append("config.yaml is not a settings file")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"config.yaml can't be read ({_one_line(exc)})")
    if problems:
        return [make_check("EDM-ARS files", "fail", f"{root}: " + "; ".join(problems),
                           "Reinstall EDM-ARS with the installer.")]
    return [make_check("EDM-ARS files", "ok", str(root))]


def check_settings_file(settings: dict[str, Any]) -> list["Check"]:
    from edmars import paths

    path = paths.settings_path()
    if not Path(path).is_file():
        return [make_check("Settings", "warn", f"Setup has not been run yet (no settings file at {path})",
                           "Run `edmars setup`.")]
    out = [make_check("Settings", "ok", str(path))]
    last = cfg(settings, "setup_progress.last_completed_screen", None)
    if last and last != "S11":
        out.append(make_check("Setup", "info", f"Setup was not finished (it stopped after step {str(last).lstrip('S')})",
                              "Run `edmars setup` to continue where you stopped."))
    return out


def check_disclosure(settings: dict[str, Any]) -> list["Check"]:
    from edmars import disclosure

    version = getattr(disclosure, "ACK_VERSION", "?")
    if disclosure.is_acknowledged(settings):
        return [make_check("Notice accepted", "ok", f"You accepted the notice about what leaves your computer (version {version})")]
    return [make_check("Notice accepted", "fail",
                       f"You have not accepted the current notice about what leaves your computer (version {version}); "
                       "studies can't start until you do",
                       "Run `edmars setup disclosure` (read it first with `edmars privacy` and `edmars disclaimer`).")]


def configured_models(settings: dict[str, Any], provider_id: str) -> list[str]:
    """The distinct model ids a study will call: shipped defaults plus overrides."""
    merged: dict[str, Any] = {}
    try:
        from edmars import providers

        merged.update(providers.default_models(provider_id) or {})
    except Exception:
        pass
    overrides = cfg(settings, "models", None)
    if isinstance(overrides, dict):
        merged.update(overrides)
    values = [str(v) for v in merged.values() if v]
    seen: list[str] = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen


def _key_status_check(name: str, label: str, status: str, message: str, balance: str | None,
                      fix_page: str | None) -> "Check":
    message = redact(message or "")
    if status == "OK":
        detail = f"{label} accepted the key" + (f"; balance {balance}" if balance else "")
        return make_check(name, "ok", detail)
    if status == "REJECTED":
        return make_check(name, "fail", f"{label} did not accept the saved key",
                          "Paste a new key with `edmars setup ai`" + (f" (keys: {fix_page})" if fix_page else "") + ".")
    if status == "NO_CREDIT":
        return make_check(name, "fail", f"The key works, but your {label} account has no credit",
                          f"Add credit{(' at ' + fix_page) if fix_page else ''}, then run `edmars doctor --deep` again.")
    if status == "NETWORK":
        return make_check(name, "fail", f"Couldn't reach {label}. {message}".strip(),
                          "Check your internet connection. University networks sometimes block AI services; "
                          "a VPN or another network may help.")
    return make_check(name, "warn", f"{label} answered in an unexpected way. {message}".strip(),
                      "Try again later; if it keeps happening, run `edmars doctor --bundle` and open an issue.")


def check_provider_key(settings: dict[str, Any], *, deep: bool = False) -> list["Check"]:
    from edmars import secrets

    provider_id = str(cfg(settings, "provider", "deepseek") or "deepseek")
    meta = provider_meta(provider_id)
    label = meta["label"] or provider_id
    env_var = meta["env_var"] or ""
    out: list["Check"] = []

    if provider_id == "local":
        base_url = cfg(settings, "provider_base_url", None)
        if not base_url:
            return [make_check("AI service", "fail", "A local model server was chosen, but no server address is saved",
                               "Run `edmars setup ai`.")]
        out.append(make_check("AI service", "ok", f"Your own model server at {base_url} (experimental)"))
        if deep:
            from edmars import providers

            key = secrets.get_secret(env_var) if env_var else None
            result = providers.check_key("local", key or "local", base_url=str(base_url))
            status = str(getattr(result, "status", "UNKNOWN"))
            if status == "OK":
                available = [str(m) for m in (getattr(result, "models", None) or [])]
                wanted = configured_models(settings, "local")
                missing = [m for m in wanted if available and m not in available]
                if missing:
                    out.append(make_check("AI models", "fail", "Your server does not offer: " + ", ".join(missing),
                                          "Load the model in your server, or pick another with `edmars setup ai`."))
                else:
                    out.append(make_check("AI models", "ok", f"Your server answered and offers {len(available)} model(s)"))
            else:
                out.append(_key_status_check("AI service check", "your model server", status,
                                             str(getattr(result, "message", "")), None, None))
        return out

    source = secrets.secret_source(env_var) if env_var else None
    if source is None:
        return [make_check("AI service key", "fail", f"No {label} key found ({env_var})",
                           "Run `edmars setup ai`" + (f"; get a key at {meta['key_page']}" if meta["key_page"] else "") + ".")]
    out.append(make_check("AI service key", "ok", f"{label} key found in {store_label(source, env_var)}"))
    if not configured_models(settings, provider_id):
        # OpenAI ships no model list and the pipeline refuses to guess one.
        out.append(make_check("AI models", "fail", f"No {label} model is chosen, so studies can't start",
                              "Run `edmars setup ai` and choose a model."))
    if not deep:
        return out

    from edmars import providers

    key = secrets.get_secret(env_var)
    if not key:
        out.append(make_check("AI service check", "fail", f"The {label} key could not be read",
                              "Run `edmars setup ai` and paste the key again."))
        return out
    base_url = None  # provider_base_url belongs to a local server only
    result = providers.check_key(provider_id, key, base_url=base_url)
    status = str(getattr(result, "status", "UNKNOWN"))
    fix_page = meta["topup"] if status == "NO_CREDIT" else meta["key_page"]
    out.append(_key_status_check("AI service check", label, status, str(getattr(result, "message", "")),
                                 getattr(result, "balance", None), fix_page))
    if status in ("OK", "NO_CREDIT"):
        wanted = configured_models(settings, provider_id)
        if wanted:
            missing = providers.missing_models(provider_id, key, wanted, base_url=base_url)
            if missing:
                out.append(make_check("AI models", "fail",
                                      f"{label} no longer offers: " + ", ".join(missing),
                                      "Run `edmars update` for new defaults, or choose other models in "
                                      "`edmars setup advanced`."))
            else:
                out.append(make_check("AI models", "ok", f"{label} offers all {len(wanted)} model(s) EDM-ARS will use"))
    return out


def check_semantic_scholar(settings: dict[str, Any], *, deep: bool = False) -> list["Check"]:
    from edmars import secrets

    source = secrets.secret_source(SEMANTIC_SCHOLAR_ENV)
    if source is None:
        return [make_check("Literature search", "warn",
                           "No Semantic Scholar key. Semantic Scholar often turns away keyless searches; arXiv, "
                           "the other search, needs no key but can refuse requests outright, so papers can end up "
                           "with few real citations. A free Semantic Scholar key is the reliable fix",
                           "Request a free key at https://www.semanticscholar.org/product/api#api-key-form, "
                           "then run `edmars setup literature`.")]
    out = [make_check("Literature search", "ok", f"Semantic Scholar key found in {store_label(source, SEMANTIC_SCHOLAR_ENV)}")]
    if deep:
        from edmars import providers

        key = secrets.get_secret(SEMANTIC_SCHOLAR_ENV)
        if key:
            result = providers.check_semantic_scholar(key)
            status = str(getattr(result, "status", "UNKNOWN"))
            if status == "OK":
                out.append(make_check("Semantic Scholar check", "ok", "Semantic Scholar accepted the key"))
            elif status == "REJECTED":
                out.append(make_check("Semantic Scholar check", "fail", "Semantic Scholar did not accept the saved key",
                                      "Run `edmars setup literature` and paste the key again."))
            else:
                out.append(make_check("Semantic Scholar check", "warn",
                                      "Couldn't confirm the key: " + redact(str(getattr(result, "message", "")) or status),
                                      "Check your internet connection and try again later."))
    return out


def check_datasets(settings: dict[str, Any]) -> list["Check"]:
    """One line per dataset; a missing optional dataset is information only.

    A dataset the user installed that no longer validates stays a failure.
    It is also a failure when no dataset at all is ready.
    """
    from edmars import datasets

    recorded = cfg(settings, "datasets", {}) or {}
    out: list["Check"] = []
    ready = 0
    for name in list(getattr(datasets, "CATALOG", {}) or {}):
        chk = datasets.status(name, settings)
        status = str(getattr(chk, "status", "info"))
        if status == "ok":
            ready += 1
            out.append(chk)
            continue
        installed = isinstance(recorded, dict) and name in recorded
        if status == "fail" and not installed:
            out.append(make_check(str(chk.name), "info", str(chk.detail), getattr(chk, "fix", None)))
        else:
            out.append(chk)
    if ready == 0:
        out.insert(0, make_check("Datasets", "fail", "No dataset is ready, so no study can start",
                                 "Run `edmars setup datasets` (HSLS:09 is the recommended first dataset)."))
    return out


def _downgrade(checks: Iterable["Check"], to: str = "warn") -> list["Check"]:
    out: list["Check"] = []
    for chk in checks:
        if str(getattr(chk, "status", "")) == "fail":
            out.append(make_check(str(chk.name), to, str(chk.detail), getattr(chk, "fix", None)))
        else:
            out.append(chk)
    return out


def check_latex(settings: dict[str, Any], *, deep: bool = False) -> list["Check"]:
    from edmars import toolchain

    mode = str(cfg(settings, "latex.mode", "") or "")
    checks = list(toolchain.latex_checks(settings))
    if mode == "none":
        checks = _downgrade(checks)
        checks.insert(0, make_check("PDF typesetting", "warn",
                                    "Turned off in setup: studies produce LaTeX source and figures but no PDF, "
                                    "and the automated reviewer can't run",
                                    "Run `edmars setup pdf` to turn it on."))
        return checks
    if deep:
        if any(str(getattr(chk, "status", "")) == "fail" for chk in checks):
            # The test documents need what the failing line above says is
            # missing, so they could only fail for the same reason and
            # count one problem two or three times (no LaTeX made plain
            # doctor say 1 problem and --deep say 3).
            checks.append(make_check("PDF test", "info",
                                     "Not run: the LaTeX problem above has to be fixed first"))
        else:
            checks.extend(toolchain.test_compile(timeout_s=120, settings=settings))
    return checks


def check_r(settings: dict[str, Any]) -> list["Check"]:
    from edmars import toolchain

    rscript = cfg(settings, "r.rscript", None) or toolchain.find_rscript(settings)
    if not rscript:
        return [make_check("R", "info", "R not found. It is only needed for measurement (psychometrics) studies",
                           "Run `edmars setup r` when you want to run a measurement study.")]
    # R is optional for every other study type, so its problems are warnings;
    # a measurement study's own pre-flight blocks the launch.
    return _downgrade(toolchain.r_checks(settings))


def check_lsar(settings: dict[str, Any], *, deep: bool = False) -> list["Check"]:
    if not cfg(settings, "lsar.enabled", False):
        return [make_check("Automated reviewer", "info", "Off (LSAR is not turned on)",
                           "Run `edmars setup reviewer` to turn it on.")]
    from edmars import lsar, secrets

    # deep: load LSAR and its PDF layout model in a child Python, the one
    # check that catches a reviewer that would skip or quietly change reviews.
    out = list(lsar.checks(settings, deep=deep))
    if secrets.secret_source(DEEPSEEK_ENV) is None:
        out.append(make_check("Reviewer key", "fail",
                              "The automated reviewer needs a DeepSeek key (its scoring was calibrated with DeepSeek), "
                              "and none is saved",
                              "Run `edmars setup reviewer`."))
    if str(cfg(settings, "latex.mode", "") or "") == "none":
        out.append(make_check("Reviewer needs PDFs", "warn",
                              "The reviewer reads the PDF, but PDF typesetting is off, so reviews will be skipped",
                              "Run `edmars setup pdf`."))
    return out


def check_disk(settings: dict[str, Any]) -> list["Check"]:
    out: list["Check"] = []
    seen: set[tuple[int, int]] = set()
    try:
        from edmars import datasets

        hsls_ready = str(getattr(datasets.status("hsls09_public", settings), "status", "")) == "ok"
    except Exception:
        hsls_ready = True  # unknown: do not add the download requirement
    for label, folder in (("studies folder", studies_dir(settings)), ("data folder", _data_dir(settings))):
        probe = _existing_parent(folder)
        usage = shutil.disk_usage(probe)
        key = (usage.total, usage.free // (1024 ** 2))
        if key in seen:
            continue
        seen.add(key)
        free = _gb(usage.free)
        need = DISK_WARN_GB + (HSLS_DOWNLOAD_GB if label == "data folder" and not hsls_ready else 0.0)
        text = f"{free:.1f} GB free for the {label} ({probe})"
        if free < DISK_FAIL_GB:
            out.append(make_check("Disk space", "fail", text, "Free up disk space, or choose a folder on another drive in `edmars setup folders`."))
        elif free < need:
            out.append(make_check("Disk space", "warn", text + f"; EDM-ARS works best with {need:.0f} GB or more",
                                  "Free up some disk space."))
        else:
            out.append(make_check("Disk space", "ok", text))
    return out


def check_memory() -> list["Check"]:
    try:
        import psutil
    except Exception:
        return [make_check("Memory", "info", "Could not measure memory (psutil is missing)")]
    total = _gb(psutil.virtual_memory().total)
    text = f"{total:.0f} GB memory"
    if total < RAM_RECOMMENDED_GB - 0.5:
        return [make_check("Memory", "warn",
                           f"{text}. EDM-ARS works best with {RAM_RECOMMENDED_GB:.0f} GB or more; big studies may be slow or fail",
                           "Close other programs while a study runs.")]
    return [make_check("Memory", "ok", text)]


def check_sync_folders(settings: dict[str, Any]) -> list["Check"]:
    from edmars import paths

    out: list["Check"] = []
    places = (("Your studies folder", studies_dir(settings), "Choose a local folder with `edmars setup folders`."),
              ("The data folder", _data_dir(settings), "Move EDM-ARS's data folder out of the synced folder (set EDMARS_HOME), or pause syncing during studies."),
              ("The EDM-ARS program folder", paths.app_root(), "Reinstall EDM-ARS into a local folder (the installer's default location)."))
    for label, folder, fix in places:
        provider = paths.sync_provider(Path(folder))
        if provider:
            out.append(make_check("Cloud sync", "warn",
                                  f"{label} is inside {provider} ({folder}). Syncing can slow studies down a lot "
                                  "and uploads your working files to the cloud", fix))
    if not out:
        out.append(make_check("Cloud sync", "ok", "Studies and data are in local folders"))
    return out


def check_encoding() -> list["Check"]:
    encoding = (getattr(sys.stdout, "encoding", None) or "unknown")
    normalized = encoding.lower().replace("-", "").replace("_", "")
    plain = _is_plain()
    mode = "plain text mode" if plain else "full display"
    if normalized in ("utf8", "utf8sig"):
        return [make_check("Terminal", "ok", f"UTF-8 output, {mode}")]
    return [make_check("Terminal", "warn",
                       f"This terminal uses {encoding}; some symbols may show as '?'. Status words are always printed too",
                       "Use Windows Terminal or PowerShell, or run commands with --plain.")]


def check_shell() -> list["Check"]:
    if os.environ.get("MSYSTEM"):
        return [make_check("Shell", "info",
                           "Git Bash detected. Arrow-key menus may fall back to typed answers here",
                           "PowerShell or Windows Terminal gives the best experience.")]
    return []


def check_docker() -> list["Check"]:
    from edmars import toolchain

    chk = toolchain.docker_info()
    detail = str(getattr(chk, "detail", "")) or "Docker information unavailable"
    return [make_check("Docker", "info", detail + " (EDM-ARS does not use Docker; nothing to do)")]


def _active_run_lock() -> Path:
    from edmars import paths

    return paths.data_dir() / "active_run.json"


def check_active_run() -> list["Check"]:
    lock = _active_run_lock()
    if not lock.is_file():
        return [make_check("Running study", "ok", "No study is running")]
    try:
        data = json.loads(lock.read_text(encoding="utf-8"))
    except Exception:
        return [make_check("Running study", "warn", f"The study lock file is damaged ({lock})",
                           "If no study is running, delete that file.")]
    pid = data.get("pid") if isinstance(data, dict) else None
    run_dir = None
    if isinstance(data, dict):
        run_dir = data.get("run_dir") or data.get("run") or data.get("path")
    from edmars import proc

    alive = False
    started = data.get("create_time") if isinstance(data, dict) else None
    try:
        # The runner records the process's start time, so a pid Windows has
        # since handed to another program does not look like the study.
        alive = isinstance(pid, int) and proc.pid_alive(
            pid, started_at=started if isinstance(started, (int, float)) else None)
    except Exception:
        alive = False
    if alive:
        return [make_check("Running study", "info", f"A study is running: {run_dir} (process {pid})",
                           "Watch it with `edmars status`.")]
    return [make_check("Running study", "warn",
                       f"A study stopped without clearing its lock ({run_dir or 'unknown folder'})",
                       "Check it with `edmars status`; resume it with `edmars resume`. "
                       "Starting a new study clears the stale lock.")]


def _where_short(source: str | None) -> str:
    if source == "env":
        return "set in your environment"
    if source == "file":
        return "in a private file"
    return "in " + store_label(source)


def check_keys_found() -> list["Check"]:
    """Which keys already exist, and where (for the setup wizard's S2)."""
    from edmars import secrets

    found: list[str] = []
    seen: set[str] = set()
    for provider_id in ("deepseek", "openai", "anthropic"):
        env_var = provider_meta(provider_id)["env_var"]
        if not env_var or env_var in seen:
            continue
        seen.add(env_var)
        source = secrets.secret_source(env_var)
        if source:
            found.append(f"{env_var} ({_where_short(source)})")
    source = secrets.secret_source(SEMANTIC_SCHOLAR_ENV)
    if source:
        found.append(f"{SEMANTIC_SCHOLAR_ENV} ({_where_short(source)})")
    if not found:
        return [make_check("Saved keys", "info", "No AI service keys found yet. You'll add one in a later step")]
    return [make_check("Saved keys", "ok", "Found " + "; ".join(found) + ". You can reuse them in the next steps")]


def check_old_dotenv() -> list["Check"]:
    from edmars import paths

    dotenv = paths.app_root() / ".env"
    if dotenv.is_file():
        return [make_check("Old .env file", "info",
                           f"Found {dotenv}. In the AI service step you can move its keys into secure storage",
                           None)]
    return []


def check_latex_quick(settings: dict[str, Any] | None = None) -> list["Check"]:
    from edmars import toolchain

    # toolchain also finds a TinyTeX that is not on PATH yet.
    pdflatex = toolchain.find_tex_tool("pdflatex", settings)
    if pdflatex:
        flavor = "MiKTeX" if toolchain.find_tex_tool("initexmf", settings) else "LaTeX"
        return [make_check("PDF typesetting", "ok", f"{flavor} found ({pdflatex})")]
    return [make_check("PDF typesetting", "info", "LaTeX not found. You can add it in a later step")]


def check_r_quick(settings: dict[str, Any]) -> list["Check"]:
    from edmars import toolchain

    rscript = cfg(settings, "r.rscript", None) or toolchain.find_rscript(settings)
    if rscript:
        return [make_check("R", "ok", f"R found ({rscript})")]
    return [make_check("R", "info", "R not found. Only needed for measurement (psychometrics) studies")]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def system_checks(settings: dict[str, Any]) -> list["Check"]:
    """The quick, offline checks the setup wizard shows on its S2 screen."""
    groups: list[tuple[str, Callable[[], list["Check"]]]] = [
        ("Computer", check_os),
        ("Disk space", lambda: check_disk(settings)),
        ("Memory", check_memory),
        ("Python", check_python),
        ("PDF typesetting", lambda: check_latex_quick(settings)),
        ("R", lambda: check_r_quick(settings)),
        ("Saved keys", check_keys_found),
        ("Cloud sync", lambda: check_sync_folders(settings)),
        ("Shell", check_shell),
        ("Old .env file", check_old_dotenv),
    ]
    out: list["Check"] = []
    for label, fn in groups:
        out.extend(_safe(label, fn))
    return out


def run_checks(settings: dict[str, Any], *, deep: bool = False) -> list["Check"]:
    """Every doctor check, in reading order. Never raises, never prompts.

    ``deep=True`` adds live key checks (each key goes only to its own
    service), retired-model detection, a LaTeX test compile and a load
    test of the automated reviewer.
    """
    groups: list[tuple[str, Callable[[], list["Check"]]]] = [
        ("Computer", check_os),
        ("Python", check_python),
        ("Python packages", check_packages),
        ("XGBoost", check_xgboost),
        ("EDM-ARS files", check_app),
        ("Settings", lambda: check_settings_file(settings)),
        ("Notice accepted", lambda: check_disclosure(settings)),
        ("AI service key", lambda: check_provider_key(settings, deep=deep)),
        ("Literature search", lambda: check_semantic_scholar(settings, deep=deep)),
        ("Datasets", lambda: check_datasets(settings)),
        ("PDF typesetting", lambda: check_latex(settings, deep=deep)),
        ("R", lambda: check_r(settings)),
        ("Automated reviewer", lambda: check_lsar(settings, deep=deep)),
        ("Disk space", lambda: check_disk(settings)),
        ("Memory", check_memory),
        ("Cloud sync", lambda: check_sync_folders(settings)),
        ("Terminal", check_encoding),
        ("Shell", check_shell),
        ("Running study", check_active_run),
        ("Docker", check_docker),
    ]
    out: list["Check"] = []
    for label, fn in groups:
        out.extend(_safe(label, fn))
    return out


def load_settings_for_doctor() -> tuple[dict[str, Any], "Check | None"]:
    """Load settings; a damaged file becomes a failing check, not a crash."""
    from edmars import settings as st

    try:
        return st.load(), None
    except Exception as exc:  # noqa: BLE001
        try:
            from edmars import paths

            where = str(paths.settings_path())
        except Exception:
            where = "settings.yaml"
        problem = make_check("Settings", "fail", f"The settings file can't be read: {where} ({redact(_one_line(exc))})",
                             "Move that file aside and run `edmars setup` again.")
        return copy.deepcopy(getattr(st, "DEFAULTS", {})), problem


def _edmars_version() -> str:
    try:
        module = importlib.import_module("edmars")
        version = getattr(module, "__version__", None)
        if version:
            return str(version)
    except Exception:
        pass
    try:
        from importlib import metadata

        return metadata.version("edmars")
    except Exception:
        return "unknown"


def _payload(checks: Sequence[Any], *, deep: bool, bundle: Path | None = None) -> dict[str, Any]:
    counts = summarize(checks)
    payload: dict[str, Any] = {
        "schema": 1,
        "tool": "edmars doctor",
        "edmars_version": _edmars_version(),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "deep": deep,
        "ok": counts.get("fail", 0) == 0,
        "counts": counts,
        "checks": [check_to_dict(c) for c in checks],
    }
    if bundle is not None:
        payload["bundle"] = str(bundle)
    return payload


def quick_checks() -> list["Check"]:
    """Installation sanity only (the installer's `edmars doctor --quick`).

    Offline and fast; nothing that setup has not configured yet can fail it.
    """
    groups: list[tuple[str, Callable[[], list["Check"]]]] = [
        ("Computer", check_os),
        ("Python", check_python),
        ("Python packages", check_packages),
        ("XGBoost", check_xgboost),
        ("EDM-ARS files", check_app),
        ("Terminal", check_encoding),
    ]
    out: list["Check"] = []
    for label, fn in groups:
        out.extend(_safe(label, fn))
    return out


def _collect(*, deep: bool, quick: bool) -> tuple[dict[str, Any], list[Any]]:
    settings, problem = load_settings_for_doctor()
    if quick:
        return settings, quick_checks()
    return settings, ([problem] if problem else []) + run_checks(settings, deep=deep)


def main(*, deep: bool = False, json_out: bool = False, bundle: bool = False, quick: bool = False) -> int:
    """`edmars doctor`. Returns 0 when nothing failed, 1 when anything did.

    With ``bundle`` the command's job is the support file, so it returns 0
    once the file is written and 1 only when it could not be: the Mac
    test's `doctor --bundle` wrote its bundle and exited 1 because doctor
    had found a problem, which a script reads as "the bundle failed".

    ``quick`` (an extra beyond the CLI_SPEC section-18 signature) runs only
    the installation checks, for the installer's smoke test.
    """
    from edmars import ui

    if json_out:
        # Only the JSON document may reach stdout; anything a check prints
        # (for example a keyring warning) goes to stderr instead.
        bundle_path: Path | None = None
        with contextlib.redirect_stdout(sys.stderr):
            settings, checks = _collect(deep=deep, quick=quick)
            if bundle:
                try:
                    bundle_path = make_bundle(default_bundle_path(settings), checks=checks, announce=False)
                except OSError as exc:
                    checks.append(make_check("Support bundle", "fail",
                                             f"Couldn't write the support bundle: {redact(_one_line(exc))}"))
        payload = _payload(checks, deep=deep, bundle=bundle_path)
        text = redact(json.dumps(payload, indent=2, ensure_ascii=False))
        sys.stdout.write(text + "\n")
        sys.stdout.flush()
        if bundle:
            return 0 if bundle_path is not None else 1
        return 0 if payload["ok"] else 1

    if deep and not quick:
        ui.info("Deep check: each saved key is sent only to its own service to confirm it works, "
                "and LaTeX compiles two small test documents (up to a few minutes).")
    settings, checks = _collect(deep=deep, quick=quick)
    render_checks(checks, title="EDM-ARS installation check" if quick else "EDM-ARS check")
    counts = summarize(checks)
    if counts.get("fail", 0):
        ui.fail(f"{counts['fail']} problem(s) to fix and {counts.get('warn', 0)} warning(s). The fixes are listed above.")
    elif counts.get("warn", 0):
        ui.warn(f"No problems, {counts['warn']} warning(s) worth reading.")
    else:
        ui.ok("Everything checked out.")
    if not quick:
        try:
            from edmars import paths

            _out(f"Settings file: {paths.settings_path()}")
        except Exception:
            pass
    if bundle:
        include_run = bool(ui.confirm(
            "Include the latest study's log files (pipeline.log, run_status.json, console.log)? "
            "They can contain your research question and short data excerpts from error messages.",
            default=True))
        try:
            make_bundle(default_bundle_path(settings), include_run=include_run, checks=checks)
        except OSError as exc:
            ui.fail(f"Couldn't write the support bundle: {redact(_one_line(exc))}")
            return 1
        return 0
    return 1 if counts.get("fail", 0) else 0


# ---------------------------------------------------------------------------
# Support bundle
# ---------------------------------------------------------------------------

_KEY_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"sk-[A-Za-z0-9_-]{8,}"), "sk-<redacted>"),
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{8,}"), r"\1<redacted>"),
    (re.compile(r"(?i)((?:x-api-key|api[_-]?key|authorization|access[_-]?token|secret)[\"']?\s*[:=]\s*[\"']?)"
                r"[^\s\"',;]{8,}"), r"\1<redacted>"),
)


def _secret_values() -> list[str]:
    values: set[str] = set()
    for name in KNOWN_SECRET_NAMES:
        env_value = os.environ.get(name)
        if env_value and len(env_value) >= 8:
            values.add(env_value)
        try:
            from edmars import secrets

            stored = secrets.get_secret(name)
        except Exception:
            stored = None
        if stored and len(stored) >= 8:
            values.add(stored)
    return sorted(values, key=len, reverse=True)


def redact(text: str, extra: Iterable[str] = ()) -> str:
    """Remove secrets by value (``edmars.secrets.redact`` plus ``extra``) and by shape."""
    if not text:
        return text
    try:
        from edmars import secrets

        text = secrets.redact(text)
    except Exception:
        pass
    for value in extra:
        if value and len(value) >= 3:
            text = text.replace(value, "<removed>")
    for pattern, repl in _KEY_PATTERNS:
        text = pattern.sub(repl, text)
    return text


def _scrub(text: str, extra: Iterable[str] = ()) -> str:
    """:func:`redact`, then replace the home-folder path with ``~`` (bundle files)."""
    if not text:
        return text
    text = redact(text, extra)
    home = str(Path.home())
    if len(home) > 3:
        variants = {home, home.replace("\\", "/"), home.replace("\\", "\\\\")}
        for variant in sorted(variants, key=len, reverse=True):
            text = re.sub(re.escape(variant), "~", text, flags=re.IGNORECASE)
    return text


def _redacted_settings(settings: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """A copy with personal fields replaced; also returns their values to scrub."""
    clean = copy.deepcopy(settings) if isinstance(settings, dict) else {}
    personal: list[str] = []
    for dotted in ("author.name", "author.affiliation", "literature.crossref_mailto"):
        node: Any = clean
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.get(part) if isinstance(node, dict) else None
        if isinstance(node, dict) and node.get(parts[-1]):
            personal.append(str(node[parts[-1]]))
            node[parts[-1]] = "<set>"
    return clean, personal


def _read_tail(path: Path, limit: int = BUNDLE_MAX_BYTES) -> str:
    size = path.stat().st_size
    with open(path, "rb") as fh:
        if size > limit:
            fh.seek(size - limit)
            data = fh.read()
            head = f"[... first {size - limit} bytes omitted ...]\n".encode()
            data = head + data.split(b"\n", 1)[-1]
        else:
            data = fh.read()
    return data.decode("utf-8", errors="replace")


def _latest_run(settings: dict[str, Any]) -> Path | None:
    try:
        from edmars import runner

        found = runner.latest_run(settings)
        if found:
            return Path(found)
    except Exception:
        pass
    root = studies_dir(settings)
    if not root.is_dir():
        return None
    candidates = [p for p in root.iterdir() if p.is_dir() and (p / "pipeline.log").exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: (p / "pipeline.log").stat().st_mtime)


def _versions() -> dict[str, Any]:
    from importlib import metadata

    packages: dict[str, str | None] = {}
    for dist in VERSION_PACKAGES:
        try:
            packages[dist] = metadata.version(dist)
        except Exception:
            packages[dist] = None
    return {
        "edmars": _edmars_version(),
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "os": _os_label(),
        "packages": packages,
    }


def default_bundle_path(settings: dict[str, Any]) -> Path:
    """``<studies folder's parent>/edmars-support-<timestamp>.zip`` (home as fallback)."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    parent = studies_dir(settings).parent
    if not parent.is_dir():
        parent = Path.home()
    return parent / f"edmars-support-{stamp}.zip"


def _allowed_in_bundle(name: str) -> bool:
    lowered = name.replace("\\", "/").lower()
    return not (lowered.endswith(".csv") or "prompts/" in lowered or lowered.startswith("prompts"))


def make_bundle(path: str | os.PathLike[str], *, include_run: bool = True, announce: bool = True,
                checks: Sequence[Any] | None = None) -> Path:
    """Write a support zip and return its path.

    ``path`` is the zip to create, or an existing folder to create it in.
    Contents: ``doctor.json``, ``settings.yaml`` (personal fields replaced),
    ``versions.json``, ``README.txt`` and, when ``include_run``, the latest
    study's ``pipeline.log`` / ``run_status.json`` / ``console.log``.
    Every text is scrubbed of secrets and of the home-folder path. Never
    ``prompts/``, never a CSV.
    """
    import yaml

    dest = Path(path).expanduser()
    if dest.is_dir() or not dest.suffix:
        dest = dest / f"edmars-support-{datetime.now().strftime('%Y%m%d-%H%M%S')}.zip"
    dest.parent.mkdir(parents=True, exist_ok=True)

    settings, problem = load_settings_for_doctor()
    if checks is None:
        checks = ([problem] if problem else []) + run_checks(settings)
    clean_settings, personal = _redacted_settings(settings)
    extra = personal + _secret_values()

    entries: dict[str, str] = {}
    entries["doctor.json"] = json.dumps(_payload(checks, deep=False), indent=2, ensure_ascii=False)
    entries["settings.yaml"] = yaml.safe_dump(clean_settings, sort_keys=True, allow_unicode=True)
    entries["versions.json"] = json.dumps(_versions(), indent=2)

    run_note = "No study log was included."
    if include_run:
        run = _latest_run(settings)
        if run is not None:
            added: list[str] = []
            for fname in BUNDLE_RUN_FILES:
                candidate = run / fname
                if candidate.is_file():
                    entries[f"last_study/{fname}"] = _read_tail(candidate)
                    added.append(fname)
            run_note = (f"Latest study: {run.name} ({', '.join(added)})" if added
                        else f"Latest study {run.name} had no log files yet.")
        else:
            run_note = "No study was found."

    readme = (
        "EDM-ARS support bundle\n"
        "======================\n\n"
        "Made by `edmars doctor --bundle`. Attach it to an issue at\n"
        "https://github.com/cgpan/edm-ars-public/issues\n\n"
        "It contains the doctor report, your settings (your name, affiliation and\n"
        "contact email replaced by <set>), version numbers, and the latest study's\n"
        "log files if you agreed to include them. API keys and the path of your\n"
        "home folder were removed from every file. It never contains data files\n"
        "(CSV), the prompts/ folder, or anything else from your studies.\n\n"
        f"{run_note}\n"
        "Please look through the files before you share them.\n"
    )
    entries["README.txt"] = readme

    fd, tmp_name = tempfile.mkstemp(prefix=".edmars-bundle-", suffix=".zip", dir=str(dest.parent))
    os.close(fd)
    try:
        with zipfile.ZipFile(tmp_name, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for name, text in entries.items():
                if not _allowed_in_bundle(name):
                    continue
                zf.writestr(name, _scrub(text, extra))
        os.replace(tmp_name, dest)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise

    if announce:
        from edmars import ui

        ui.ok(f"Saved a support bundle: {dest}")
        _out("It contains:")
        for name in entries:
            if _allowed_in_bundle(name):
                _out(f"  - {name}")
        _out("API keys and your home-folder path were removed. It has no data files and no prompts.")
        _out("Look through it before you share it.")
    return dest


__all__ = [
    "run_checks", "system_checks", "quick_checks", "main", "make_bundle", "render_checks", "summarize",
    "provider_meta", "store_label", "make_check", "cfg", "default_bundle_path", "redact", "configured_models",
]
