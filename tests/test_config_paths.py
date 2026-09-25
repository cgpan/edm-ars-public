"""C5: config and the paths in it resolve against the repository, not the cwd.

Before, ``python -m src.main`` started from any directory other than the
repository root raised FileNotFoundError for config.yaml, and with an
explicit --config it ran with zero skills, one-line agent prompts and a
raw-data path under the wrong folder.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

import src.config as config_mod
from src.config import load_config, resolve_config_path, resolve_repo_path

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def _clean_lsar_home(monkeypatch: pytest.MonkeyPatch) -> None:
    # setenv (not delenv) so monkeypatch restores the absent state even
    # though load_config writes LSAR_HOME.
    monkeypatch.setenv("LSAR_HOME", "")


def test_default_config_loads_from_another_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _clean_lsar_home: None
) -> None:
    monkeypatch.chdir(tmp_path)
    config = load_config()
    for key in ("data_registry", "raw_data", "output_base", "agent_prompts",
                "paper_template"):
        value = config["paths"][key]
        assert os.path.isabs(value), key
        assert Path(value).resolve().is_relative_to(REPO_ROOT.resolve()), key
    assert Path(config["paths"]["agent_prompts"], "critic.yaml").exists()
    assert Path(config["paths"]["data_registry"], "datasets",
                "hsls09_public.yaml").exists()
    memory = Path(config["findings_memory"]["path"])
    assert memory.is_absolute()
    assert memory.parent == REPO_ROOT / "findings_memory"


def test_relative_config_path_falls_back_to_the_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert Path(resolve_config_path("config.yaml")) == REPO_ROOT / "config.yaml"
    assert Path(resolve_config_path(None)) == REPO_ROOT / "config.yaml"


def test_a_relative_path_that_exists_from_the_cwd_keeps_its_meaning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _clean_lsar_home: None
) -> None:
    (tmp_path / "mydata").mkdir()
    cfg = tmp_path / "my_config.yaml"
    text = (REPO_ROOT / "config.yaml").read_text(encoding="utf-8")
    cfg.write_text(text.replace("raw_data: data/raw/", "raw_data: mydata/"),
                   encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    config = load_config("my_config.yaml")
    raw = config["paths"]["raw_data"]
    assert Path(raw) == tmp_path / "mydata"
    # String concatenation callers rely on the trailing separator.
    assert raw.endswith(os.sep)
    # data_registry does not exist under tmp_path, so it is the repo's.
    assert Path(config["paths"]["data_registry"]) == REPO_ROOT / "data_registry"


def test_repo_root_invocation_resolves_to_the_same_files(
    monkeypatch: pytest.MonkeyPatch, _clean_lsar_home: None
) -> None:
    monkeypatch.chdir(REPO_ROOT)
    config = load_config("config.yaml")
    assert Path(config["paths"]["agent_prompts"]) == REPO_ROOT / "agent_prompts"
    assert config["paths"]["agent_prompts"].endswith(os.sep)


def test_absolute_paths_are_unchanged(tmp_path: Path) -> None:
    assert resolve_repo_path(str(tmp_path)) == str(tmp_path)


def test_missing_config_names_the_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError) as info:
        load_config("no_such_config.yaml")
    assert "no_such_config.yaml" in str(info.value)


def test_lsar_home_default_prefers_an_existing_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = tmp_path / "edm-ars-public"
    app.mkdir()
    monkeypatch.setattr(config_mod, "PROJECT_ROOT", app)
    assert Path(config_mod.default_lsar_home()) == tmp_path / "LSAR"

    (tmp_path / "LSAR-public").mkdir()
    assert Path(config_mod.default_lsar_home()) == tmp_path / "LSAR-public"

    (tmp_path / "LSAR").mkdir()
    assert Path(config_mod.default_lsar_home()) == tmp_path / "LSAR"


def test_lsar_home_from_the_environment_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LSAR_HOME", str(tmp_path / "somewhere"))
    config = load_config(str(REPO_ROOT / "config.yaml"))
    assert config["review_gate"]["lsar_project_path"] == str(tmp_path / "somewhere")


def test_lsar_home_default_is_absolute(
    monkeypatch: pytest.MonkeyPatch, _clean_lsar_home: None
) -> None:
    config = load_config(str(REPO_ROOT / "config.yaml"))
    assert os.path.isabs(config["review_gate"]["lsar_project_path"])
