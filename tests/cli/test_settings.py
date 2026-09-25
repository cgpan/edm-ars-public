"""settings.yaml: defaults, round trip, tolerance of hand edits, atomic save."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from edmars import paths, settings


def test_missing_file_gives_complete_defaults(edmars_home: Path) -> None:
    loaded = settings.load()
    assert set(settings.DEFAULTS) <= set(loaded)
    assert loaded["defaults"]["venue"] == "EDM"
    assert loaded["lsar"] == {"enabled": False, "auto_review": False, "home": None, "ref": None}
    # Under EDMARS_HOME the studies folder never points into the real home.
    assert Path(loaded["studies_dir"]) == edmars_home / "studies"
    # Loading creates nothing.
    assert not paths.settings_path().exists()


def test_defaults_are_a_fresh_copy() -> None:
    first = settings.load()
    first["defaults"]["venue"] = "changed"
    first["models"]["analyst"] = "x"
    assert settings.DEFAULTS["defaults"]["venue"] == "EDM"
    assert settings.DEFAULTS["models"] == {}
    assert settings.load()["defaults"]["venue"] == "EDM"


def test_round_trip_including_paths(edmars_home: Path) -> None:
    data = settings.load()
    settings.set_(data, "provider", "openai")
    settings.set_(data, "datasets.hsls09_public.path", edmars_home / "data" / "hsls.csv")
    settings.set_(data, "author.name", "Ada Researcher")
    settings.save(data)

    again = settings.load()
    assert again["provider"] == "openai"
    assert again["datasets"]["hsls09_public"]["path"] == str(edmars_home / "data" / "hsls.csv")
    assert again["author"]["name"] == "Ada Researcher"
    text = paths.settings_path().read_text(encoding="utf-8")
    assert text.startswith("# EDM-ARS settings")


def test_partial_file_is_merged_with_defaults() -> None:
    path = paths.settings_path()
    path.parent.mkdir(parents=True)
    path.write_text("provider: anthropic\ndefaults:\n  venue: AERA Open\n", encoding="utf-8")
    loaded = settings.load()
    assert loaded["provider"] == "anthropic"
    assert loaded["defaults"]["venue"] == "AERA Open"
    assert loaded["defaults"]["paper_format"] == "conference"
    assert loaded["r"] == {"rscript": None, "packages_ok": False}


def test_unknown_keys_survive_a_round_trip() -> None:
    path = paths.settings_path()
    path.parent.mkdir(parents=True)
    path.write_text("future_feature: 3\nlsar:\n  new_knob: on\n", encoding="utf-8")
    settings.save(settings.load())
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert raw["future_feature"] == 3
    assert raw["lsar"]["new_knob"] is True
    assert raw["lsar"]["enabled"] is False


def test_section_with_wrong_type_keeps_its_default() -> None:
    path = paths.settings_path()
    path.parent.mkdir(parents=True)
    path.write_text("models: null\nliterature: just a string\n", encoding="utf-8")
    loaded = settings.load()
    assert loaded["models"] == {}
    assert loaded["literature"]["semantic_scholar_key_set"] is False


def test_corrupt_file_is_set_aside_with_a_warning(capsys: pytest.CaptureFixture[str]) -> None:
    path = paths.settings_path()
    path.parent.mkdir(parents=True)
    path.write_text("provider: [unclosed\n", encoding="utf-8")
    loaded = settings.load()
    assert loaded == settings.defaults()
    assert not path.exists()
    kept = list(path.parent.glob("settings.yaml.broken-*"))
    assert len(kept) == 1
    assert "could not be read" in capsys.readouterr().err


def test_empty_file_is_defaults() -> None:
    path = paths.settings_path()
    path.parent.mkdir(parents=True)
    path.write_text("", encoding="utf-8")
    assert settings.load() == settings.defaults()


def test_failed_save_leaves_previous_file_and_no_temp_files(monkeypatch: pytest.MonkeyPatch) -> None:
    data = settings.load()
    settings.set_(data, "provider", "deepseek")
    settings.save(data)
    before = paths.settings_path().read_bytes()

    def broken_replace(src: str, dst: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(settings.os, "replace", broken_replace)
    settings.set_(data, "provider", "openai")
    with pytest.raises(OSError):
        settings.save(data)
    assert paths.settings_path().read_bytes() == before
    leftovers = [p.name for p in paths.config_dir().iterdir() if p.name.startswith(".settings-")]
    assert leftovers == []


def test_save_retries_a_brief_windows_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    real_replace = os.replace
    calls = {"n": 0}

    def flaky_replace(src: str, dst: object) -> None:
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError("sharing violation")
        real_replace(src, dst)  # type: ignore[arg-type]

    monkeypatch.setattr(settings.os, "replace", flaky_replace)
    monkeypatch.setattr(settings.time, "sleep", lambda _s: None)
    settings.save(settings.load())
    assert calls["n"] == 3
    assert paths.settings_path().exists()


def test_save_refuses_a_value_shaped_like_an_api_key() -> None:
    data = settings.load()
    settings.set_(data, "provider_base_url", "sk-fake-0123456789abcdefghij")
    with pytest.raises(settings.SettingsError):
        settings.save(data)
    assert not paths.settings_path().exists()


def test_save_accepts_paths_that_merely_contain_sk_dash(edmars_home: Path) -> None:
    data = settings.load()
    settings.set_(data, "studies_dir", str(edmars_home / "desk-top-research-folder-2026"))
    settings.save(data)
    assert settings.load()["studies_dir"].endswith("desk-top-research-folder-2026")


def test_get_and_set_dotted_keys() -> None:
    data: dict = {"a": {"b": None}, "x": 5}
    assert settings.get(data, "a.b", "fallback") is None
    assert settings.get(data, "a.c", "fallback") == "fallback"
    assert settings.get(data, "x.y", 1) == 1
    settings.set_(data, "a.c.d", 2)
    assert data["a"]["c"] == {"d": 2}
    settings.set_(data, "x.y", 3)  # a scalar in the way is replaced by a section
    assert data["x"] == {"y": 3}
    with pytest.raises(ValueError):
        settings.set_(data, "a..b", 1)
    with pytest.raises(ValueError):
        settings.get(data, "")


def test_studies_dir_expands_the_home_folder() -> None:
    data = {"studies_dir": "~/EDM-ARS/studies"}
    assert settings.studies_dir(data) == Path.home() / "EDM-ARS" / "studies"


def test_utc_now_is_iso_with_z() -> None:
    stamp = settings.utc_now()
    assert stamp.endswith("Z") and "T" in stamp
