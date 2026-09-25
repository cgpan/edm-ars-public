"""The user's non-secret settings (``settings.yaml``).

The file lives in :func:`edmars.paths.settings_path`. It never holds an
API key: keys go to the operating system's credential store through
:mod:`edmars.secrets`, and :func:`save` refuses to write anything shaped
like one.

:func:`load` always returns a complete dictionary: every key in
:data:`DEFAULTS` is present even when the file is missing, older than
this version, or was edited by hand and lost a section. Keys the file has
that this version does not know are kept, so a newer version's settings
survive a round trip through an older one.

:func:`save` is atomic: it writes a temporary file in the same folder and
swaps it in with ``os.replace``, so a crash or a full disk leaves the
previous settings intact rather than a half-written file.
"""

from __future__ import annotations

import copy
import os
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path, PurePath
from typing import Any

import yaml

from edmars import paths

SCHEMA = 1

DEFAULTS: dict[str, Any] = {
    "schema": SCHEMA,
    # Disclosure acknowledgement: which text version was accepted, and when.
    "acknowledged": {"version": None, "at": None},
    "studies_dir": "~/EDM-ARS/studies",
    # deepseek | openai | anthropic | local
    "provider": "deepseek",
    # Only for "local" and other OpenAI-compatible servers.
    "provider_base_url": None,
    # Per-stage model overrides; empty means the shipped defaults.
    "models": {},
    "literature": {"semantic_scholar_key_set": False, "crossref_mailto": None},
    # name -> {path, sha256, verified_at, ...}
    "datasets": {},
    # mode: tinytex | system | none (None = not set up yet)
    "latex": {"mode": None, "pdflatex": None},
    "r": {"rscript": None, "packages_ok": False},
    "lsar": {"enabled": False, "auto_review": False, "home": None, "ref": None},
    "defaults": {
        "venue": "EDM",
        "paper_format": "conference",
        "budget_usd": None,
        "keep_awake": True,
    },
    "author": {"name": None, "affiliation": None},
    "setup_progress": {"last_completed_screen": None},
}

_HEADER = (
    "# EDM-ARS settings, written by `edmars setup`. You may edit this file.\n"
    "# It holds no API keys: those are kept in your system's credential store.\n"
)

#: Shapes that must never be written into settings.yaml. A key pasted
#: into the wrong wizard field would otherwise sit in a plain-text file
#: that support bundles and backups copy around.
_SECRET_SHAPES = (
    re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"(?<![A-Za-z0-9])tvly-[A-Za-z0-9_-]{16,}"),
)


class SettingsError(RuntimeError):
    """Raised when settings cannot be saved; the message is for the user."""


def defaults() -> dict[str, Any]:
    """Return a fresh copy of :data:`DEFAULTS` for this environment.

    The only difference from the constant: under ``EDMARS_HOME`` the
    default studies folder lives inside that home, so tests and portable
    installs never create ``~/EDM-ARS`` in the real profile.
    """
    base = copy.deepcopy(DEFAULTS)
    if paths.home_override() is not None:
        base["studies_dir"] = str(paths.default_studies_dir())
    return base


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge ``override`` onto a copy of ``base``.

    A section that should be a mapping but was replaced by something else
    in the file (``models: null``, a stray string) keeps its default, so a
    hand edit cannot make ``settings["models"]["analyst"]`` crash later.
    """
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict):
            if isinstance(value, dict):
                out[key] = _merge(out[key], value)
            continue
        out[key] = copy.deepcopy(value)
    return out


def _warn(message: str) -> None:
    try:
        from edmars import ui

        ui.warn(message)
    except Exception:  # pragma: no cover - output must never break loading
        pass


def _set_aside(path: Path, reason: str) -> None:
    """Rename an unreadable settings file so it is kept but not reused."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = path.with_name(f"{path.name}.broken-{stamp}")
    try:
        os.replace(path, target)
        where = f"it was kept as {target.name}"
    except OSError:
        where = "it was left in place"
    _warn(
        f"Your settings file could not be read ({reason}); {where} and the "
        "default settings are used. Run `edmars setup` to set things up again."
    )


def load() -> dict[str, Any]:
    """Return the saved settings merged over :func:`defaults`."""
    path = paths.settings_path()
    base = defaults()
    if not path.exists():
        return base
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        # Locked by a sync client or antivirus: do not rename it, just
        # carry on with defaults for this command.
        _warn(f"Your settings file could not be opened ({exc.strerror or exc}); "
              "using the default settings for now.")
        return base
    except UnicodeDecodeError:
        _set_aside(path, "it is not UTF-8 text")
        return base
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f"line {mark.line + 1}" if mark is not None else "a syntax error"
        _set_aside(path, f"YAML error at {where}")
        return base
    if raw is None:
        return base
    if not isinstance(raw, dict):
        _set_aside(path, "it does not contain a list of settings")
        return base
    return _merge(base, raw)


def _to_plain(value: Any) -> Any:
    """Convert values YAML's safe dumper cannot represent (Path, tuple)."""
    if isinstance(value, dict):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    if isinstance(value, PurePath):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _contains_secret_shape(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_secret_shape(v) for v in value.values())
    if isinstance(value, list):
        return any(_contains_secret_shape(v) for v in value)
    if isinstance(value, str):
        return any(p.search(value) for p in _SECRET_SHAPES)
    return False


def _replace_with_retry(src: str, dst: Path, attempts: int = 6) -> None:
    """``os.replace`` that tolerates a brief lock on Windows.

    Antivirus scanners and sync clients open freshly written files for a
    moment; ``os.replace`` then fails with a sharing violation that goes
    away within a fraction of a second.
    """
    delay = 0.05
    for attempt in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay)
            delay *= 2


def save(settings: dict[str, Any]) -> None:
    """Write ``settings`` to :func:`edmars.paths.settings_path` atomically."""
    if not isinstance(settings, dict):
        raise TypeError("settings must be a dict")
    plain = _to_plain(settings)
    if _contains_secret_shape(plain):
        raise SettingsError(
            "Refusing to save settings: one value looks like an API key. "
            "Keys are stored in the system credential store, not in settings.yaml."
        )
    path = paths.settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(plain, sort_keys=False, allow_unicode=True, default_flow_style=False)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".settings-", suffix=".yaml.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(_HEADER)
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        _replace_with_retry(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def get(settings: dict[str, Any], dotted: str, default: Any = None) -> Any:
    """Read ``settings["a"]["b"]["c"]`` as ``get(settings, "a.b.c")``.

    ``default`` is returned only when a key is missing; a stored ``None``
    is returned as ``None`` (it usually means "deliberately not set").
    """
    current: Any = settings
    for part in _split(dotted):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default
    return current


def set_(settings: dict[str, Any], dotted: str, value: Any) -> None:
    """Set ``settings["a"]["b"]["c"] = value``, creating sections as needed."""
    parts = _split(dotted)
    current = settings
    for part in parts[:-1]:
        nxt = current.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            current[part] = nxt
        current = nxt
    current[parts[-1]] = value


def _split(dotted: str) -> list[str]:
    parts = dotted.split(".")
    if not dotted or any(not p for p in parts):
        raise ValueError(f"not a settings key: {dotted!r}")
    return parts


def studies_dir(settings: dict[str, Any]) -> Path:
    """The studies folder as an absolute path (``~`` expanded)."""
    raw = get(settings, "studies_dir") or str(paths.default_studies_dir())
    return Path(str(raw)).expanduser()


def utc_now() -> str:
    """Current UTC time as ISO 8601 with a ``Z`` suffix (for timestamps)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
