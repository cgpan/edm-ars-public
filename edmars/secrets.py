"""API keys: where they are kept, how they are found, and how they are hidden.

Keys are stored in the operating system's credential store (Windows
Credential Manager, macOS Keychain, the Linux Secret Service) through
``keyring``, under the service name ``edm-ars`` and the environment
variable's name as the user name (``DEEPSEEK_API_KEY`` and so on). They
are never written to settings.yaml, a study folder or a repository
``.env`` file.

Lookup order (:func:`get_secret`): the environment variable, then the
credential store, then the fallback file. The environment wins so a
one-off ``DEEPSEEK_API_KEY=... edmars run`` works, but when it differs
from the stored key the user is warned once, because a forgotten old
variable silently overriding a fresh key is a classic "my key doesn't
work" support case.

The fallback file (``secrets.env`` in the config folder, readable only by
the user) exists for machines without a working credential store, such
as a headless Linux server. It is used ONLY when the caller passes
``allow_file=True`` after the user agreed to it; otherwise
:func:`set_secret` raises :class:`SecretStoreError`.

:func:`redact` removes secrets from any text before the application
writes or prints it: every key value this process has seen, key-shaped
environment variables, and common key patterns (``sk-...``, ``Bearer
...``, ``API_KEY=...``).
"""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from edmars import paths

#: keyring service name. Changing it orphans every stored key.
SERVICE = "edm-ars"

SECRETS_FILE_NAME = "secrets.env"

#: Keys the application knows about (used by redaction and uninstall).
KNOWN_SECRET_NAMES: tuple[str, ...] = (
    "DEEPSEEK_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "SEMANTIC_SCHOLAR_API_KEY",
    "OPENALEX_API_KEY",
    "TAVILY_API_KEY",
    "MINIMAX_API_KEY",
)

_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")

#: Environment variable names whose values are treated as secrets when
#: redacting, even if the application never looked them up.
_SECRET_ENV_NAME = re.compile(r"(API_?KEY|TOKEN|SECRET|PASSWORD|PASSWD)", re.IGNORECASE)

#: Values that are clearly template placeholders, not keys. Sending
#: "your-key-here" to a provider as a credential produces a confusing 401
#: and, for optional services, a silently degraded run.
_PLACEHOLDER = re.compile(
    r"^(?:your[\s_-].*|<.*>|changeme|change[\s_-]?me|replace[\s_-]?me.*|x{4,}|sk-x{4,}|\*+|\.\.\.)$",
    re.IGNORECASE,
)

#: Exact values shorter than this are not redacted by value (they would
#: blank out ordinary words); pattern redaction still applies.
_MIN_REDACT_LEN = 8

_REDACTION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{8,}"), "sk-[REDACTED]"),
    (re.compile(r"(?<![A-Za-z0-9])tvly-[A-Za-z0-9_-]{8,}"), "tvly-[REDACTED]"),
    (re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}"), "Bearer [REDACTED]"),
    (
        re.compile(
            r"(?i)\b([A-Z0-9_]*(?:API[_-]?KEY|ACCESS[_-]?TOKEN|AUTH[_-]?TOKEN|SECRET|PASSWORD)"
            r"|x-api-key|apikey)(\s*[:=]\s*[\"']?)([^\s\"'&,;]{8,})"
        ),
        r"\1\2[REDACTED]",
    ),
)

REDACTED = "[REDACTED]"

#: Every secret value this process has read or stored, for exact-value
#: redaction. Never printed.
_SEEN: set[str] = set()

#: Names already warned about (env shadowing a stored key).
_WARNED: set[str] = set()


class SecretStoreError(RuntimeError):
    """The credential store could not keep a key and no fallback was allowed.

    ``reason`` is the short cause (an error class name such as
    ``NoKeyringError``), for callers that write their own sentence.
    """

    def __init__(self, message: str, reason: str = "") -> None:
        super().__init__(message)
        self.reason = reason


def _keyring() -> Any:
    """The ``keyring`` module (indirection so tests can substitute a fake)."""
    import keyring

    return keyring


def _check_name(name: str) -> None:
    if not _NAME.match(name or ""):
        raise ValueError(f"not an environment-variable style key name: {name!r}")


def _clean(value: str | None) -> str | None:
    """Strip a value; empty strings and template placeholders become None."""
    if value is None:
        return None
    value = value.strip()
    if not value or _PLACEHOLDER.match(value):
        return None
    return value


def _remember(value: str | None) -> None:
    if value and len(value) >= _MIN_REDACT_LEN:
        _SEEN.add(value)


# --- Credential store ---------------------------------------------------------


def _keyring_get(name: str) -> str | None:
    try:
        return _clean(_keyring().get_password(SERVICE, name))
    except Exception:
        return None  # no usable backend: behave as "not stored"


def keyring_backend() -> str | None:
    """Name of the active keyring backend, or None when none is usable.

    ``keyring`` falls back to "fail" or "null" backends when the OS store
    is missing; those cannot hold a key, so they count as unusable.
    """
    try:
        backend = _keyring().get_keyring()
    except Exception:
        return None
    cls = type(backend)
    label = f"{cls.__module__}.{cls.__name__}"
    lowered = label.lower()
    if lowered.startswith(("keyring.backends.fail", "keyring.backends.null")):
        return None
    try:
        if float(getattr(backend, "priority", 1)) <= 0:
            return None
    except Exception:
        pass
    return label


# --- Fallback file --------------------------------------------------------------


def secrets_file() -> Path:
    """Path of the fallback secrets file (it may not exist)."""
    return paths.config_dir() / SECRETS_FILE_NAME


def _file_read() -> dict[str, str]:
    path = secrets_file()
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return {}
    values: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if _NAME.match(name):
            values[name] = value
    return values


def _restrict_permissions(path: Path) -> None:
    """Make ``path`` readable only by the current user (best effort)."""
    if os.name != "nt":
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        return
    # Windows: grant the user full control first, and only then drop the
    # inherited entries, so a failure can never lock the user out.
    try:
        from edmars import proc

        user = os.environ.get("USERNAME", "")
        domain = os.environ.get("USERDOMAIN", "")
        if not user:
            return
        principal = f"{domain}\\{user}" if domain else user
        granted = proc.run(["icacls", str(path), "/grant:r", f"{principal}:F"], timeout=30)
        if granted.returncode == 0:
            proc.run(["icacls", str(path), "/inheritance:r"], timeout=30)
    except Exception:
        pass


def _file_write(values: dict[str, str]) -> None:
    path = secrets_file()
    if not values:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# EDM-ARS keys (fallback store, used because no system credential store worked).",
        "# Readable only by your user account. Do not share or back up this file.",
    ]
    lines += [f"{name}={value}" for name, value in sorted(values.items())]
    # mkstemp creates the file with owner-only permissions on POSIX, so the
    # key is never briefly world-readable.
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".secrets-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("\n".join(lines) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    _restrict_permissions(path)


def _file_get(name: str) -> str | None:
    return _clean(_file_read().get(name))


def _file_delete(name: str) -> None:
    values = _file_read()
    if name in values:
        del values[name]
        _file_write(values)


# --- Public API -----------------------------------------------------------------


def _stored(name: str) -> tuple[str | None, str | None]:
    """The stored value and where it came from ("keyring" or "file")."""
    value = _keyring_get(name)
    if value:
        return value, "keyring"
    value = _file_get(name)
    if value:
        return value, "file"
    return None, None


def _warn_shadow(name: str, store: str) -> None:
    if name in _WARNED:
        return
    _WARNED.add(name)
    where = "your system's credential store" if store == "keyring" else "the EDM-ARS keys file"
    message = (
        f"The environment variable {name} is set and differs from the key saved in "
        f"{where}. The environment variable is used. If that is not what you want, "
        f"remove {name} from your environment (or run `edmars setup` to save the "
        "key you mean)."
    )
    try:
        from edmars import ui

        ui.warn(message)
    except Exception:  # pragma: no cover - never let a warning break a lookup
        pass


def get_secret(name: str) -> str | None:
    """The key called ``name``: environment, then credential store, then file."""
    _check_name(name)
    from_env = _clean(os.environ.get(name))
    stored, store = _stored(name)
    if from_env:
        if stored and store and stored != from_env:
            _warn_shadow(name, store)
        _remember(from_env)
        return from_env
    _remember(stored)
    return stored


def secret_source(name: str) -> str | None:
    """Where :func:`get_secret` would find ``name``: "env", "keyring", "file" or None."""
    _check_name(name)
    if _clean(os.environ.get(name)):
        return "env"
    return _stored(name)[1]


def stored_location(name: str) -> str | None:
    """Where EDM-ARS itself stored ``name`` ("keyring" or "file"), ignoring the environment."""
    _check_name(name)
    return _stored(name)[1]


def set_secret(name: str, value: str, *, allow_file: bool = False) -> str:
    """Store a key and return where it went: "keyring" or "file".

    The value is read back from the credential store to prove it was
    kept (some backends accept a write and drop it). If the store fails,
    the fallback file is used only when ``allow_file`` is True; otherwise
    :class:`SecretStoreError` explains the situation so the caller can ask
    the user for consent and call again.
    """
    _check_name(name)
    value = (value or "").strip()
    if not value:
        raise ValueError("the key is empty")
    if "\n" in value or "\r" in value:
        raise ValueError("the key contains a line break; paste it again as one line")
    problem = "the credential store did not keep the key"
    try:
        store = _keyring()
        store.set_password(SERVICE, name, value)
        if store.get_password(SERVICE, name) == value:
            _remember(value)
            _file_delete(name)  # one source of truth
            return "keyring"
    except Exception as exc:
        problem = type(exc).__name__
    if not allow_file:
        raise SecretStoreError(
            f"Could not save {name} in this computer's credential store ({problem}). "
            "EDM-ARS can keep it in a file readable only by your user account instead, "
            "if you agree to that.",
            reason=problem,
        )
    values = _file_read()
    values[name] = value
    _file_write(values)
    _remember(value)
    return "file"


def delete_secret(name: str) -> None:
    """Remove ``name`` from the credential store and the fallback file.

    An environment variable of the same name is not touched (it belongs to
    the user's shell, not to EDM-ARS).
    """
    _check_name(name)
    try:
        _keyring().delete_password(SERVICE, name)
    except Exception:
        pass  # not stored there, or no usable backend
    _file_delete(name)


def child_secrets(names: Iterable[str]) -> dict[str, str]:
    """``{name: value}`` for each of ``names`` that has a key (for a child's env)."""
    found: dict[str, str] = {}
    for name in names:
        value = get_secret(name)
        if value:
            found[name] = value
    return found


def register_secret(value: str) -> None:
    """Add a value to the exact-match redaction list (e.g. a key just typed)."""
    _remember((value or "").strip())


def _known_values() -> set[str]:
    values = set(_SEEN)
    for name, value in os.environ.items():
        if name.upper() in KNOWN_SECRET_NAMES or _SECRET_ENV_NAME.search(name):
            cleaned = _clean(value)
            if cleaned and len(cleaned) >= _MIN_REDACT_LEN:
                values.add(cleaned)
    return values


def redact(text: str) -> str:
    """Return ``text`` with every secret replaced by ``[REDACTED]``."""
    if not text:
        return text
    out = text
    for value in sorted(_known_values(), key=len, reverse=True):
        out = out.replace(value, REDACTED)
    for pattern, replacement in _REDACTION_PATTERNS:
        out = pattern.sub(replacement, out)
    return out
