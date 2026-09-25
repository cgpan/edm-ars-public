"""Where things live, and cloud-sync folder detection (synthetic paths only)."""

from __future__ import annotations

import os
from pathlib import Path

import platformdirs
import pytest

from edmars import paths

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_edmars_home_relocates_everything(edmars_home: Path) -> None:
    assert paths.home_override() == edmars_home
    assert paths.config_dir() == edmars_home
    assert paths.data_dir() == edmars_home / "data"
    assert paths.cache_dir() == edmars_home / "cache"
    assert paths.settings_path() == edmars_home / "settings.yaml"
    assert paths.default_studies_dir() == edmars_home / "studies"
    assert not edmars_home.exists()  # nothing is created just by asking


def test_without_override_platformdirs_decides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EDMARS_HOME")
    assert paths.home_override() is None
    assert paths.config_dir() == Path(platformdirs.user_config_dir("edm-ars", appauthor=False))
    assert paths.data_dir() == Path(platformdirs.user_data_dir("edm-ars", appauthor=False))
    assert paths.cache_dir() == Path(platformdirs.user_cache_dir("edm-ars", appauthor=False))
    assert paths.default_studies_dir() == Path.home() / "EDM-ARS" / "studies"
    assert "edm-ars" in paths.config_dir().parts


def test_blank_override_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EDMARS_HOME", "   ")
    assert paths.home_override() is None


def test_app_root_is_this_checkout() -> None:
    root = paths.app_root()
    assert root == REPO_ROOT
    assert (root / "src").is_dir() and (root / "edmars").is_dir()


def test_app_root_honours_the_launcher_variable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EDMARS_APP_ROOT", str(tmp_path))
    assert paths.app_root() == tmp_path.resolve()


def test_ensure_dir_and_is_within(tmp_path: Path) -> None:
    target = paths.ensure_dir(tmp_path / "a" / "b")
    assert target.is_dir()
    assert paths.is_within(target, tmp_path)
    assert paths.is_within(tmp_path, tmp_path)
    assert not paths.is_within(tmp_path, target)
    assert not paths.is_within(tmp_path / "a" / ".." / "..", tmp_path)


@pytest.fixture
def no_onedrive_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os.environ):
        if name.upper().startswith("ONEDRIVE"):
            monkeypatch.delenv(name)


@pytest.mark.parametrize(
    ("path", "provider"),
    [
        ("C:/Users/<you>/OneDrive/EDM-ARS/studies", "OneDrive"),
        ("C:/Users/<you>/OneDrive - Example University/Research", "OneDrive"),
        ("/Users/<you>/Library/CloudStorage/OneDrive-Personal/studies", "OneDrive"),
        ("C:\\Users\\<you>\\OneDrive\\studies", "OneDrive"),
        ("G:/My Drive/EDM-ARS", "Google Drive"),  # synthetic example; audit-allow-path
        ("G:\\Shared drives\\Lab\\data", "Google Drive"),
        ("/Users/<you>/Google Drive/data", "Google Drive"),
        ("/Users/<you>/Library/CloudStorage/GoogleDrive-account/My Drive/x", "Google Drive"),
        ("/home/user/Dropbox/studies", "Dropbox"),
        ("C:/Users/<you>/Dropbox (Personal)/studies", "Dropbox"),
        ("/Users/<you>/Library/Mobile Documents/com~apple~CloudDocs/EDM", "iCloud Drive"),
        ("C:/Users/<you>/iCloudDrive/EDM", "iCloud Drive"),
        ("C:/Users/<you>/Documents/EDM-ARS/studies", None),
        ("/home/user/research/dropboxing-notes", None),
        ("/home/user/my-drive-backup", None),
        ("relative/folder", None),
    ],
)
def test_sync_provider_by_folder_name(path: str, provider: str | None, no_onedrive_env: None) -> None:
    assert paths.sync_provider(Path(path)) == provider


def test_sync_provider_by_onedrive_environment(
    monkeypatch: pytest.MonkeyPatch, no_onedrive_env: None
) -> None:
    # A business OneDrive can be redirected to a folder whose name does not
    # say OneDrive; the client's environment variable still gives it away.
    monkeypatch.setenv("OneDriveCommercial", "D:/Sync/Work Files")
    assert paths.sync_provider(Path("D:/Sync/Work Files/EDM-ARS")) == "OneDrive"
    assert paths.sync_provider(Path("d:\\sync\\work files")) == "OneDrive"
    assert paths.sync_provider(Path("D:/Sync/Work Files Old")) is None


def test_google_drive_virtual_drive_marker(tmp_path: Path, no_onedrive_env: None) -> None:
    assert not paths._is_google_drive_root(str(tmp_path))
    (tmp_path / ".shortcut-targets-by-id").mkdir()
    assert paths._is_google_drive_root(str(tmp_path))
    assert not paths._is_google_drive_root("")


def test_sync_provider_consults_the_drive_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_onedrive_env: None
) -> None:
    monkeypatch.setattr(paths, "_is_google_drive_root", lambda anchor: bool(anchor))
    assert paths.sync_provider(tmp_path / "studies") == "Google Drive"
