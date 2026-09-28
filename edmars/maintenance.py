"""``edmars update`` and ``edmars uninstall``.

Neither is in the fixed module list of the CLI spec (section 18), but
both commands are (section 12), and PRIVACY.md promises that ``edmars
uninstall`` removes the application's settings and stored keys and asks
separately about datasets and studies. They live here so ``cli.py`` stays
a thin layer.

``update`` only CHECKS and prints the installer command: it never
downloads or runs anything by itself. ``uninstall`` deletes only inside
EDM-ARS's own folders, never a study folder it did not create, and never
the running program (which Windows would not allow anyway); it prints
what is left for the user to delete.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from edmars import __version__

REPO = "cgpan/edm-ars-public"
RELEASES_API = f"https://api.github.com/repos/{REPO}/releases/latest"
RELEASES_PAGE = f"https://github.com/{REPO}/releases"
INSTALL_BASE = f"https://github.com/{REPO}/releases/latest/download"

#: Files that mark a folder as a study EDM-ARS made (and so may delete).
_STUDY_MARKERS = ("runner.json", "checkpoint.json", "run_status.json", "pipeline.log")

#: On macOS `edmars setup pdf` installs TinyTeX with --no-path (see
#: toolchain.tinytex_installer_command), so there is no PATH change to
#: undo, and tlmgr is not on PATH to undo one with.
_TINYTEX_PATH_UNCHANGED = sys.platform == "darwin"


# --- update -------------------------------------------------------------------


def version_tuple(version: str) -> tuple[int, ...]:
    """``"v0.10.2"`` -> ``(0, 10, 2)``; anything unparseable -> ``()``."""
    match = re.match(r"\s*[vV]?(\d+(?:\.\d+)*)", version or "")
    if not match:
        return ()
    return tuple(int(part) for part in match.group(1).split("."))


def latest_release(timeout: float = 15) -> str | None:
    """The newest released version on GitHub, or None when there is none."""
    import requests

    response = requests.get(
        RELEASES_API,
        headers={"Accept": "application/vnd.github+json", "User-Agent": f"edmars/{__version__}"},
        timeout=timeout,
    )
    if response.status_code == 404:
        return None
    response.raise_for_status()
    tag = str(response.json().get("tag_name") or "").strip()
    return tag.lstrip("vV") or None


def install_command() -> str:
    """The one-line command that installs the latest release on this system."""
    if os.name == "nt":
        return (
            "powershell -NoProfile -ExecutionPolicy Bypass -Command "
            f'"irm {INSTALL_BASE}/install.ps1 | iex"'
        )
    return f"curl -fsSL {INSTALL_BASE}/install.sh | sh"


def update(*, check_only: bool = False) -> int:
    """Check GitHub for a newer release and explain how to install it."""
    from edmars import paths, ui

    ui.info(f"You have EDM-ARS {__version__}.")
    try:
        latest = latest_release()
    except Exception as exc:
        ui.fail(
            f"Could not check for updates ({type(exc).__name__}). Check your internet "
            f"connection, or look at {RELEASES_PAGE}"
        )
        return 1
    if latest is None:
        ui.info(f"No released version was found at {RELEASES_PAGE}.")
        return 0
    if version_tuple(latest) <= version_tuple(__version__):
        ui.ok("You have the latest version.")
        return 0
    ui.info(f"Version {latest} is available.")
    if check_only:
        ui.info("Run `edmars update` to see how to install it.")
        return 0
    if (paths.app_root() / ".git").exists():
        ui.info(
            "This copy of EDM-ARS is a source checkout: update it with `git pull`, "
            "then reinstall the requirements."
        )
        return 0
    ui.info(
        "To install it, run this in a new terminal window. Your settings, keys, "
        "datasets and studies are kept:"
    )
    ui.say("    " + install_command())
    ui.info(f"What changed: {RELEASES_PAGE}")
    return 0


# --- after an install -------------------------------------------------------------


def setup_state() -> str:
    """``none`` (no settings file), ``partial`` or ``done`` (setup finished)."""
    from edmars import paths
    from edmars import settings as settings_mod

    if not paths.settings_path().is_file():
        return "none"
    last = settings_mod.get(settings_mod.load(), "setup_progress.last_completed_screen")
    return "done" if last == "S11" else "partial"


def after_install(state_file: str | os.PathLike[str] | None = None) -> int:
    """``edmars after-install``: the installer's last step, run by the new command.

    The installer builds ``venv-<version>`` from scratch on every install,
    so an update loses the packages ``edmars setup reviewer`` added to the
    previous one. On the owner's Mac that silently turned every automated
    review off. When the settings record an LSAR folder (on or off: `edmars
    review` uses it too), this reinstalls LSAR's packages into the new
    environment and checks that LSAR loads. It also tells the installer
    whether setup was already done, so an update does not end with "Next,
    set it up". Being the new ``edmars`` itself, it finds the settings
    exactly where every other command does.

    A new release can also pin a newer LSAR than the one installed: the
    settings keep the commit the previous release installed, and nothing
    else ever replaced it. Being the new version, this command's
    ``lsar.LSAR_REF`` is the new pin, so when the installed commit differs
    it first runs the same update as ``edmars setup reviewer``. When that
    fails (no network, or a study still running), the old LSAR is made
    ready in the new environment instead and keeps reviewing.

    Writes ``setup=<none|partial|done>`` and ``reviewer=<none|ok|repaired|
    updated|outdated|failed>`` lines to ``state_file``; ``outdated`` is an
    older LSAR that works but could not be updated. Returns 0, or 1 when
    the reviewer needs ``edmars setup reviewer`` before it can review.
    """
    from edmars import lsar, ui
    from edmars import settings as settings_mod

    def reason(exc: Exception) -> str:
        return str(exc) if isinstance(exc, lsar.LsarInstallError) else f"{type(exc).__name__}: {exc}"

    state = setup_state()
    settings = settings_mod.load()
    reviewer = "none"
    older = lsar.outdated(settings) if settings_mod.get(settings, "lsar.home") else None
    if older:
        ui.info(f"The automated reviewer (LSAR) is set up with an older version ({older[:12]}) than "
                f"this release was tested with ({lsar.LSAR_REF[:12]}): updating it (a small download "
                "from GitHub) and checking that it loads...")
        try:
            lsar.update(settings)
        except Exception as exc:  # noqa: BLE001 - the old version is still in place
            ui.warn(f"The automated reviewer could not be updated: {reason(exc)} "
                    "The version you have is kept.")
        else:
            reviewer = "updated"
            ui.ok("The automated reviewer is updated and ready.")
    if reviewer == "none" and settings_mod.get(settings, "lsar.home"):
        ui.info("The automated reviewer (LSAR) is set up: installing its Python packages "
                "into the new environment and checking that it loads...")
        try:
            plan = lsar.reinstall_requirements(settings)
        except Exception as exc:  # noqa: BLE001 - the installer must get its answer
            reviewer = "failed"
            ui.warn(f"The automated reviewer could not be made ready: {reason(exc)} "
                    "Repair it with `edmars setup reviewer`; until then its reviews are skipped.")
        else:
            again = len(plan.to_install)
            reviewer = "outdated" if older else ("repaired" if again else "ok")
            ui.ok("The automated reviewer is ready"
                  + (f" ({again} of its packages installed again)" if again else "")
                  + (", in the older version; update it later with `edmars setup reviewer`."
                     if older else "."))
    if state_file is not None:
        Path(state_file).write_text(f"setup={state}\nreviewer={reviewer}\n", encoding="utf-8")
    return 1 if reviewer == "failed" else 0


# --- uninstall ------------------------------------------------------------------


def _on_rm_error(func: Callable[..., Any], path: str, _exc: Any) -> None:
    """rmtree helper: clear the read-only bit Windows puts on some files, retry."""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except OSError:
        pass


def _remove(path: Path) -> bool:
    """Delete a file or folder; True when it is gone afterwards."""
    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path, onerror=_on_rm_error)
        elif path.exists() or path.is_symlink():
            path.unlink()
    except OSError:
        pass
    return not path.exists()


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def _human(size: int) -> str:
    value = float(size)
    for unit in ("bytes", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "bytes" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} bytes"  # pragma: no cover


def _owned_roots() -> list[Path]:
    """EDM-ARS's own folders; nothing outside them is deleted automatically."""
    from edmars import paths

    return [paths.config_dir(), paths.data_dir(), paths.cache_dir()]


def _is_owned(path: Path, *, allow_root: bool = False) -> bool:
    """True when ``path`` is inside one of EDM-ARS's own folders.

    ``path`` must be strictly below the folder unless ``allow_root`` (used
    for the cache folder, which is entirely EDM-ARS's).

    Those folders are named ``edm-ars`` by platformdirs (or sit under
    ``EDMARS_HOME``); a root that is neither is not trusted, so a broken
    environment can never turn uninstall into "delete my home folder".
    """
    from edmars import paths

    home = paths.home_override()
    for root in _owned_roots():
        trusted = paths.APP_NAME in {p.casefold() for p in root.parts} or (
            home is not None and paths.is_within(root, home)
        )
        if not trusted:
            continue
        if paths.is_within(path, root) and (allow_root or path.resolve() != root.resolve()):
            return True
    return False


def _running_study() -> str | None:
    """The folder (or pid) of a running study, from the runner's lock file."""
    from edmars import paths, proc

    lock = paths.data_dir() / "active_run.json"
    try:
        info = json.loads(lock.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(info, dict):
        return None
    pid = info.get("pid")
    if isinstance(pid, int) and proc.pid_alive(pid):
        return str(info.get("run_dir") or f"process {pid}")
    return None


def _study_folders(studies: Path) -> list[Path]:
    if not studies.is_dir():
        return []
    return sorted(
        child
        for child in studies.iterdir()
        if child.is_dir() and any((child / marker).exists() for marker in _STUDY_MARKERS)
    )


def uninstall(
    *,
    assume_yes: bool = False,
    remove_datasets: bool | None = None,
    remove_studies: bool | None = None,
) -> int:
    """Remove settings, stored keys, the LSAR install, caches, and optionally data.

    ``assume_yes`` skips the main confirmation (``--yes``). Datasets and
    studies are kept unless ``remove_datasets`` / ``remove_studies`` is True
    or the user says yes to the separate question about each.
    """
    from edmars import paths, secrets, ui
    from edmars import settings as settings_mod

    running = _running_study()
    if running:
        ui.fail(
            f"A study is still running ({running}). Stop it first with `edmars stop`, "
            "then run `edmars uninstall` again."
        )
        return 1

    current = settings_mod.load()
    config, data, cache = paths.config_dir(), paths.data_dir(), paths.cache_dir()
    keys = [(n, secrets.stored_location(n)) for n in secrets.KNOWN_SECRET_NAMES]
    keys = [(n, where) for n, where in keys if where]
    files = [
        paths.settings_path(),
        secrets.secrets_file(),
        *sorted(config.glob("settings.yaml.broken-*")),
        data / "lsar",
        data / "findings_memory",
        data / "active_run.json",
        cache,
    ]
    files = [p for p in files if p.exists() and _is_owned(p, allow_root=p == cache)]
    datasets_dir = data / "data"
    has_datasets = datasets_dir.exists() and _is_owned(datasets_dir)
    studies_dir = settings_mod.studies_dir(current)
    studies = _study_folders(studies_dir)

    if not keys and not files and not has_datasets and not studies:
        ui.ok("Nothing of EDM-ARS's is stored on this computer (apart from the program).")
        _explain_left_in_place(current)
        _explain_program_removal()
        return 0

    ui.say("This removes from this computer:")
    for name, where in keys:
        label = "credential store" if where == "keyring" else "keys file"
        ui.say(f"  - the stored key {name} ({label})")
    for path in files:
        ui.say(f"  - {path}")
    if has_datasets or studies:
        ui.say("It asks separately about:")
        if has_datasets:
            ui.say(f"  - downloaded datasets ({_human(_size(datasets_dir))}): {datasets_dir}")
        if studies:
            ui.say(f"  - your {len(studies)} study folder(s) in {studies_dir}")

    if not assume_yes:
        if not ui.is_interactive():
            raise ui.NonInteractiveError(
                "Remove these now?", hint="Pass --yes to remove them without being asked."
            )
        if not ui.confirm("Remove these now?", default=False):
            ui.info("Nothing was removed.")
            return 0

    if has_datasets and remove_datasets is None:
        remove_datasets = (
            ui.confirm("Also delete the downloaded datasets? You can download them again later.",
                       default=False)
            if ui.is_interactive() else False
        )
    if studies and remove_studies is None:
        remove_studies = (
            ui.confirm(
                f"Also delete your {len(studies)} study folder(s), including the papers? "
                "This cannot be undone.",
                default=False,
            )
            if ui.is_interactive() else False
        )

    problems: list[Path] = []
    for name, _where in keys:
        secrets.delete_secret(name)
        if secrets.stored_location(name):
            ui.warn(f"Could not remove the stored key {name}; remove it in your credential store.")
    for path in files:
        if not _remove(path):
            problems.append(path)
    if has_datasets and remove_datasets:
        if not _remove(datasets_dir):
            problems.append(datasets_dir)
    elif has_datasets:
        ui.info(f"Datasets kept in {datasets_dir}")
    if studies and remove_studies:
        for folder in studies:
            if not _remove(folder):
                problems.append(folder)
        try:
            studies_dir.rmdir()  # only succeeds when nothing else is in it
        except OSError:
            pass
    elif studies:
        ui.info(f"Studies kept in {studies_dir}")

    for path in problems:
        ui.warn(f"Could not remove {path} (it may be open in another program). Delete it yourself.")
    if not problems:
        ui.ok("EDM-ARS's settings, stored keys, automated reviewer and caches were removed.")
    _explain_left_in_place(current)
    _explain_program_removal()
    return 1 if problems else 0


def _explain_left_in_place(current: dict[str, Any]) -> None:
    """Name what setup installed outside EDM-ARS's folders and uninstall keeps.

    TinyTeX (from `edmars setup pdf`) lives in the folder the official
    TinyTeX installer uses, where other programs can use it too, so it is
    listed for the user to delete rather than deleted. R packages from
    `edmars setup r` went into the user's own R library, shared with their
    other R work: they are named, with the library and an R command that
    removes them, when setup recorded them (``r.added_packages``), and only
    mentioned otherwise.
    """
    from edmars import settings as settings_mod
    from edmars import toolchain, ui

    if str(settings_mod.get(current, "latex.mode", "") or "") == "tinytex":
        root = toolchain.tinytex_root()
        if root.is_dir():
            ui.info(
                f"TinyTeX (the LaTeX that `edmars setup pdf` installed, {_human(_size(root))}) is still in "
                f"{root}. Other programs can use it, so it was not removed. If nothing else needs it, "
                + ("" if _TINYTEX_PATH_UNCHANGED else "run `tlmgr path remove` (this undoes the PATH "
                   "change TinyTeX's installer made), ") +
                "then delete that folder."
            )
    added = settings_mod.get(current, "r.added_packages", {}) or {}
    listed = False
    if isinstance(added, dict):
        for library, names in added.items():
            names = [str(n) for n in (names or []) if n]
            if not names:
                continue
            listed = True
            lib = str(library).replace("\\", "/").replace("'", "\\'")
            ui.info(
                f"Uninstall does not touch R. `edmars setup r` added {len(names)} R packages to "
                f"{library}; your other R work may use them, so they stay. "
                "If nothing else needs them, remove them in R with: "
                f"remove.packages(c({', '.join(repr(n) for n in names)}), lib = '{lib}')"
            )
    if not listed and settings_mod.get(current, "r.packages_ok", False):
        ui.info("Uninstall does not touch R: any R packages `edmars setup r` added stay in your R "
                "library, where your other R work may use them.")


def _install_record(root: Path) -> tuple[Path, dict[str, Any]] | None:
    """The installer's ``install.json`` for the copy running from ``root``.

    The installer puts the app in ``<install dir>/app/<version>`` and
    records there everything it created (the venv, a private Python and uv,
    the launcher, the PATH change).
    """
    if root.parent.name != "app":
        return None
    base = root.parent.parent
    try:
        record = json.loads((base / "install.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        record = None
    return base, record if isinstance(record, dict) else {}


def _explain_program_removal() -> None:
    """Say how to delete the program itself (it cannot delete itself while running).

    Only what the installer created is listed. The install folder is not
    named as a whole: on Windows it is also where settings, datasets and
    LSAR live (%LOCALAPPDATA%\\edm-ars), and those are handled above.
    """
    from edmars import paths, ui

    root = paths.app_root()
    found = _install_record(root)
    if found is None:
        ui.info(
            f"This copy of EDM-ARS runs from {root}; delete that folder yourself "
            "if you no longer need it."
        )
        return
    base, record = found
    items: list[Path] = [base / "app"]
    items += sorted(base.glob("venv-*"))
    items += [base / name for name in ("python", "uv", "install.json", "versions.txt")]
    if record.get("uv_private") and record.get("uv"):
        items.append(Path(str(record["uv"])))
    launcher = record.get("launcher")
    items.append(Path(str(launcher)) if launcher else
                 Path.home() / ".local" / "bin" / ("edmars.cmd" if os.name == "nt" else "edmars"))
    if record.get("sh_launcher"):  # Windows: the same command for Git Bash
        items.append(Path(str(record["sh_launcher"])))
    seen: set[str] = set()
    shown: list[Path] = []
    for item in items:
        key = str(item).casefold()
        if key in seen or not item.exists():
            continue
        # A private uv inside <base>/uv is already covered by that folder.
        if any(paths.is_within(item, parent) and item != parent for parent in shown):
            continue
        seen.add(key)
        shown.append(item)
    ui.info("To remove the program itself, close this window and delete:")
    for item in shown:
        ui.say(f"  - {item}")
    if record.get("path_modified"):
        bin_dir = record.get("bin_dir") or (Path(str(launcher)).parent if launcher else None)
        files = [str(f) for f in (record.get("path_files") or []) if f]
        if files:
            ui.say(f"  - and the lines between '# >>> edm-ars >>>' and '# <<< edm-ars <<<' in: {', '.join(files)}")
        elif os.name == "nt" and bin_dir:
            ui.say(f"  - and remove {bin_dir} from your user PATH (Settings > System > About > "
                   "Advanced system settings > Environment Variables)")
