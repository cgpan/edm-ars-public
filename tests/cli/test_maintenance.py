"""``edmars update`` (check only, fake HTTP) and ``edmars uninstall`` (temp home)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
import requests

from edmars import __version__, maintenance, paths, secrets, settings, ui
from tests.cli.conftest import FakeKeyring


class FakeResponse:
    def __init__(self, status: int, payload: dict[str, Any] | None = None) -> None:
        self.status_code = status
        self._payload = payload or {}

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


def fake_get(monkeypatch: pytest.MonkeyPatch, response: FakeResponse | Exception) -> list[str]:
    urls: list[str] = []

    def get(url: str, **_kwargs: Any) -> FakeResponse:
        urls.append(url)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(requests, "get", get)
    return urls


def test_version_tuple() -> None:
    assert maintenance.version_tuple("v0.10.2") == (0, 10, 2)
    assert maintenance.version_tuple("0.2.0rc1") == (0, 2, 0)
    assert maintenance.version_tuple("nonsense") == ()
    assert maintenance.version_tuple("0.10.0") > maintenance.version_tuple("0.9.9")


def test_update_reports_a_newer_release(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    urls = fake_get(monkeypatch, FakeResponse(200, {"tag_name": "v99.0.0"}))
    assert maintenance.update(check_only=True) == 0
    out = capsys.readouterr().out
    assert "Version 99.0.0 is available" in out
    assert urls == [maintenance.RELEASES_API]


def test_update_on_the_latest_version(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    fake_get(monkeypatch, FakeResponse(200, {"tag_name": f"v{__version__}"}))
    assert maintenance.update() == 0
    assert "latest version" in capsys.readouterr().out


def test_update_explains_how_to_install(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
                                       tmp_path: Path) -> None:
    fake_get(monkeypatch, FakeResponse(200, {"tag_name": "v99.0.0"}))
    app = tmp_path / "app" / "99.0.0"
    app.mkdir(parents=True)
    monkeypatch.setattr(paths, "app_root", lambda: app)
    assert maintenance.update() == 0
    out = capsys.readouterr().out
    assert maintenance.install_command() in out


def test_update_from_a_checkout_says_git_pull(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
                                            tmp_path: Path) -> None:
    fake_get(monkeypatch, FakeResponse(200, {"tag_name": "v99.0.0"}))
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(paths, "app_root", lambda: tmp_path)
    assert maintenance.update() == 0
    assert "git pull" in capsys.readouterr().out


def test_update_without_network(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    fake_get(monkeypatch, requests.ConnectionError("offline"))
    assert maintenance.update() == 1
    assert "Could not check for updates" in capsys.readouterr().err


def test_update_with_no_releases(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_get(monkeypatch, FakeResponse(404))
    assert maintenance.update() == 0


def _populate(edmars_home: Path, fake_keyring: FakeKeyring) -> dict[str, Path]:
    current = settings.load()
    settings.save(current)
    fake_keyring.set_password(secrets.SERVICE, "DEEPSEEK_API_KEY", "sk-fake-uninstall-0123456789")
    secrets._file_write({"TAVILY_API_KEY": "tvly-fake-uninstall-0123456789"})
    data = paths.data_dir()
    made = {
        "lsar": data / "lsar" / "LSAR-public-master",
        "memory": data / "findings_memory",
        "cache": paths.cache_dir(),
        "dataset": data / "data" / "raw" / "hsls.csv",
        "study": paths.default_studies_dir() / "2026-09-25_1402_gpa_ab12",
        "notes": paths.default_studies_dir() / "my own notes",
    }
    for key in ("lsar", "memory", "cache", "study", "notes"):
        made[key].mkdir(parents=True)
    made["dataset"].parent.mkdir(parents=True)
    made["dataset"].write_text("X1SEX\n", encoding="utf-8")
    (made["study"] / "runner.json").write_text("{}", encoding="utf-8")
    (made["notes"] / "ideas.txt").write_text("mine", encoding="utf-8")
    (made["cache"] / "tier1").mkdir()
    return made


def test_uninstall_removes_settings_keys_and_downloads_but_keeps_data(
    edmars_home: Path, fake_keyring: FakeKeyring
) -> None:
    made = _populate(edmars_home, fake_keyring)
    assert maintenance.uninstall(assume_yes=True) == 0
    assert not paths.settings_path().exists()
    assert not secrets.secrets_file().exists()
    assert fake_keyring.store == {}
    assert not made["lsar"].exists() and not made["memory"].exists() and not made["cache"].exists()
    # Without a yes to the separate questions, data and studies stay.
    assert made["dataset"].exists()
    assert made["study"].exists() and made["notes"].exists()


def test_uninstall_names_the_tinytex_it_leaves_in_place(
    edmars_home: Path, fake_keyring: FakeKeyring, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # `edmars setup pdf` installs TinyTeX (about 300 MB) outside EDM-ARS's
    # folders. Uninstall said "settings, keys and downloads were removed"
    # and never mentioned it.
    from edmars import toolchain

    _populate(edmars_home, fake_keyring)
    current = settings.load()
    settings.set_(current, "latex.mode", "tinytex")
    settings.set_(current, "r.packages_ok", True)
    settings.save(current)
    tinytex = tmp_path / "appdata" / "TinyTeX"
    (tinytex / "bin" / "windows").mkdir(parents=True)
    (tinytex / "bin" / "windows" / "pdflatex.exe").write_bytes(b"x" * 2048)
    monkeypatch.setattr(toolchain, "tinytex_root", lambda: tinytex)

    assert maintenance.uninstall(assume_yes=True) == 0
    out = " ".join(capsys.readouterr().out.split())
    assert tinytex.is_dir()  # other programs may use it; the user decides
    assert str(tinytex) in out and "tlmgr path remove" in out
    assert "R library" in out
    assert "downloads were removed" not in out
    assert "automated reviewer and caches were removed" in out


def test_uninstall_names_the_r_packages_setup_added_and_where(
    edmars_home: Path, fake_keyring: FakeKeyring, capsys: pytest.CaptureFixture[str],
) -> None:
    # "any R packages `edmars setup r` added stay in your R library" named
    # neither the packages nor the library; a real install added 74.
    _populate(edmars_home, fake_keyring)
    current = settings.load()
    lib = "C:/Users/O'Neil/AppData/Local/R/win-library/4.4"
    settings.set_(current, "r.packages_ok", True)
    settings.set_(current, "r.added_packages", {lib: ["CDM", "Deriv", "mirt"]})
    settings.save(current)

    assert maintenance.uninstall(assume_yes=True) == 0
    out = " ".join(capsys.readouterr().out.split())
    assert f"added 3 R packages to {lib};" in out
    assert "remove.packages(c('CDM', 'Deriv', 'mirt'), lib = 'C:/Users/O\\'Neil/" in out
    assert "any R packages" not in out


def test_uninstall_can_remove_data_but_only_study_folders(
    edmars_home: Path, fake_keyring: FakeKeyring
) -> None:
    made = _populate(edmars_home, fake_keyring)
    assert maintenance.uninstall(assume_yes=True, remove_datasets=True, remove_studies=True) == 0
    assert not made["dataset"].exists()
    assert not made["study"].exists()
    # A folder EDM-ARS did not create is never deleted, and so the studies
    # folder itself stays.
    assert (made["notes"] / "ideas.txt").exists()


def test_uninstall_refuses_while_a_study_runs(edmars_home: Path, fake_keyring: FakeKeyring) -> None:
    _populate(edmars_home, fake_keyring)
    lock = paths.data_dir() / "active_run.json"
    lock.write_text(json.dumps({"pid": os.getpid(), "run_dir": "somewhere"}), encoding="utf-8")
    assert maintenance.uninstall(assume_yes=True) == 1
    assert paths.settings_path().exists()


def test_uninstall_ignores_a_stale_lock(edmars_home: Path, fake_keyring: FakeKeyring) -> None:
    _populate(edmars_home, fake_keyring)
    lock = paths.data_dir() / "active_run.json"
    lock.write_text(json.dumps({"pid": 0, "run_dir": "gone"}), encoding="utf-8")
    assert maintenance.uninstall(assume_yes=True) == 0
    assert not lock.exists()


def test_uninstall_asks_before_removing(edmars_home: Path, fake_keyring: FakeKeyring,
                                        monkeypatch: pytest.MonkeyPatch) -> None:
    _populate(edmars_home, fake_keyring)
    with pytest.raises(ui.NonInteractiveError):
        maintenance.uninstall()
    monkeypatch.setattr(ui, "is_interactive", lambda: True)
    monkeypatch.setattr(ui, "confirm", lambda message, default=True: False)
    assert maintenance.uninstall() == 0
    assert paths.settings_path().exists()


def test_uninstall_with_nothing_stored(edmars_home: Path) -> None:
    assert maintenance.uninstall(assume_yes=True) == 0


def test_ownership_check_refuses_outside_paths(edmars_home: Path, tmp_path: Path) -> None:
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    assert not maintenance._is_owned(outside)
    assert not maintenance._is_owned(edmars_home)  # the root itself is never deleted
    assert maintenance._is_owned(edmars_home / "settings.yaml")
    assert maintenance._is_owned(paths.cache_dir(), allow_root=True)


def test_uninstall_lists_what_the_installer_recorded(edmars_home: Path, tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch,
                                                     capsys: pytest.CaptureFixture[str]) -> None:
    base = tmp_path / "install"
    app = base / "app" / "0.1.0"
    app.mkdir(parents=True)
    for name in ("venv-0.1.0", "python", "uv"):
        (base / name).mkdir()
    (base / "data").mkdir()  # settings/datasets can share this folder on Windows
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    launcher = bin_dir / "edmars.cmd"
    launcher.write_text("@echo off\n", encoding="utf-8")
    sh_launcher = bin_dir / "edmars"  # install.ps1's launcher for Git Bash
    sh_launcher.write_text("#!/bin/sh\n", encoding="utf-8")
    (base / "install.json").write_text(json.dumps({
        "schema": 1, "install_dir": str(base), "app_root": str(app), "uv": str(base / "uv" / "uv.exe"),
        "uv_private": True, "bin_dir": str(bin_dir), "launcher": str(launcher),
        "sh_launcher": str(sh_launcher),
        "path_modified": True, "path_files": [str(tmp_path / ".profile")],
    }), encoding="utf-8")
    monkeypatch.setattr(paths, "app_root", lambda: app)
    assert maintenance.uninstall(assume_yes=True) == 0
    out = capsys.readouterr().out
    for item in (base / "app", base / "venv-0.1.0", base / "python", base / "uv", launcher, sh_launcher):
        assert f"  - {item}" in out, item
    assert str(tmp_path / ".profile") in out  # the PATH lines the installer added
    # The folder as a whole is never named: datasets may live in it.
    assert f"  - {base}\n" not in out
