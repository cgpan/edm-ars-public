"""Where edmars keeps things on this computer.

Every location comes from ``platformdirs`` with the application name
``edm-ars``, so the app follows each operating system's conventions:

=========  ==========================================  =====================================
           Windows                                     macOS / Linux
=========  ==========================================  =====================================
config     ``%LOCALAPPDATA%\\edm-ars``                   ``~/Library/Application Support/edm-ars``
                                                       / ``~/.config/edm-ars``
data       ``%LOCALAPPDATA%\\edm-ars``                   ``~/Library/Application Support/edm-ars``
                                                       / ``~/.local/share/edm-ars``
cache      ``%LOCALAPPDATA%\\edm-ars\\Cache``             ``~/Library/Caches/edm-ars``
                                                       / ``~/.cache/edm-ars``
=========  ==========================================  =====================================

Setting the environment variable ``EDMARS_HOME`` moves all three under
one folder (``$EDMARS_HOME``, ``$EDMARS_HOME/data``, ``$EDMARS_HOME/cache``)
and moves the default studies folder to ``$EDMARS_HOME/studies``. Tests
use it so they never read or write the real profile.

None of these functions create anything; call :func:`ensure_dir` when a
folder has to exist.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import platformdirs

APP_NAME = "edm-ars"

#: Environment variable that relocates config, data and cache (tests, CI,
#: portable installs).
HOME_ENV = "EDMARS_HOME"

#: Environment variable the launcher sets to the installed application.
APP_ROOT_ENV = "EDMARS_APP_ROOT"


def home_override() -> Path | None:
    """Return ``$EDMARS_HOME`` as a path, or None when it is unset or empty."""
    raw = os.environ.get(HOME_ENV, "").strip()
    if not raw:
        return None
    return Path(raw).expanduser()


def _looks_like_app_root(path: Path) -> bool:
    return (path / "src").is_dir() and (path / "edmars").is_dir()


def app_root() -> Path:
    """Return the folder holding the pipeline (``src/``) and this package.

    The installer's launcher sets ``EDMARS_APP_ROOT``; otherwise this is
    the checkout the running ``edmars`` package lives in. The pipeline
    child process is started with this folder as its working directory.
    """
    raw = os.environ.get(APP_ROOT_ENV, "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    here = Path(__file__).resolve().parent.parent
    if _looks_like_app_root(here):
        return here
    # An unusual layout (e.g. the package copied elsewhere). Walk up from
    # the working directory before giving up, so a developer running from
    # the checkout still gets the right answer.
    for candidate in (Path.cwd(), *Path.cwd().parents):
        if _looks_like_app_root(candidate):
            return candidate
    return here


def config_dir() -> Path:
    """Folder for ``settings.yaml`` and the (rare) fallback secrets file."""
    override = home_override()
    if override is not None:
        return override
    return Path(platformdirs.user_config_dir(APP_NAME, appauthor=False))


def data_dir() -> Path:
    """Folder for datasets, the LSAR install, findings memory and run locks."""
    override = home_override()
    if override is not None:
        return override / "data"
    return Path(platformdirs.user_data_dir(APP_NAME, appauthor=False))


def cache_dir() -> Path:
    """Folder for disposable files (probe cache, crash notes)."""
    override = home_override()
    if override is not None:
        return override / "cache"
    return Path(platformdirs.user_cache_dir(APP_NAME, appauthor=False))


def settings_path() -> Path:
    """Path of the non-secret settings file."""
    return config_dir() / "settings.yaml"


def default_studies_dir() -> Path:
    """The studies folder offered in setup: ``~/EDM-ARS/studies``.

    Under ``EDMARS_HOME`` it is ``$EDMARS_HOME/studies`` instead, so tests
    never create folders in the real home directory.
    """
    override = home_override()
    if override is not None:
        return override / "studies"
    return Path.home() / "EDM-ARS" / "studies"


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_within(path: Path, parent: Path) -> bool:
    """True when ``path`` is ``parent`` or somewhere below it.

    Both are resolved first, so ``..`` segments and symlinks cannot make a
    path outside ``parent`` look like it is inside.
    """
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


# --- Cloud-sync folders -------------------------------------------------
#
# A study writes thousands of small files and a 2 GB dataset is read
# repeatedly. Inside a folder that a sync client mirrors, files get locked
# mid-write, replaced by online-only placeholders, or uploaded to an
# institution's cloud (which may breach a data-use rule). So setup warns
# and offers a local folder instead.

_SPLIT = re.compile(r"[\\/]+")

#: Folder names (compared case-insensitively) that belong to a sync client.
_PART_MARKERS: tuple[tuple[str, str], ...] = (
    ("my drive", "Google Drive"),
    ("shared drives", "Google Drive"),
    ("google drive", "Google Drive"),
    ("googledrive", "Google Drive"),
    ("dropbox", "Dropbox"),
    ("mobile documents", "iCloud Drive"),
    ("icloud drive", "iCloud Drive"),
    ("iclouddrive", "iCloud Drive"),
)

#: Prefixes of folder names that belong to a sync client. OneDrive names
#: its folders "OneDrive", "OneDrive - <Org>" or (macOS) "OneDrive-<Org>";
#: macOS File Provider mounts Google Drive as "GoogleDrive-<account>" and
#: Dropbox as "Dropbox" or "Dropbox (<Team>)".
_PREFIX_MARKERS: tuple[tuple[str, str], ...] = (
    ("onedrive", "OneDrive"),
    ("googledrive-", "Google Drive"),
    ("dropbox (", "Dropbox"),
    ("dropbox-", "Dropbox"),
)

#: A Google Drive for desktop virtual drive (e.g. ``G:\``) carries this
#: folder at its root.
_GOOGLE_DRIVE_ROOT_MARKER = ".shortcut-targets-by-id"


def _parts(path: Path | str) -> list[str]:
    """Split a path on either separator, whatever the running platform.

    ``Path.parts`` on POSIX does not split a Windows path, and paths reach
    this function from settings files written on other machines.
    """
    return [p for p in _SPLIT.split(str(path)) if p]


def _norm(path: Path | str) -> str:
    return "/".join(_parts(path)).casefold()


def _onedrive_roots() -> list[str]:
    """Values of the ``OneDrive*`` environment variables the client sets.

    Windows stores environment variable names in upper case, so the prefix
    is compared case-insensitively.
    """
    roots = []
    for name, value in os.environ.items():
        if name.upper().startswith("ONEDRIVE") and value.strip():
            roots.append(_norm(value.strip()))
    return roots


def _is_google_drive_root(anchor: str) -> bool:
    """True when the drive holding a path is a Google Drive virtual drive."""
    if not anchor:
        return False
    try:
        return (Path(anchor) / _GOOGLE_DRIVE_ROOT_MARKER).exists()
    except OSError:
        return False


def sync_provider(path: Path) -> str | None:
    """Name the cloud-sync service that mirrors ``path``, or None.

    Returns one of ``"OneDrive"``, ``"Google Drive"``, ``"Dropbox"`` or
    ``"iCloud Drive"``. The test is by name and location only (no network),
    so it is cheap and works for folders that do not exist yet.
    """
    normalized = _norm(path)
    for root in _onedrive_roots():
        if normalized == root or normalized.startswith(root + "/"):
            return "OneDrive"

    for part in _parts(path):
        lowered = part.casefold()
        for marker, provider in _PART_MARKERS:
            if lowered == marker:
                return provider
        for prefix, provider in _PREFIX_MARKERS:
            if lowered.startswith(prefix):
                return provider

    anchor = Path(str(path)).anchor
    if _is_google_drive_root(anchor):
        return "Google Drive"
    return None
