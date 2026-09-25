"""A run started outside the repository finds its own files (C5).

fix/wp-main anchored config.yaml, paths.* and the findings memory at the
repository; three lookups the orchestrator and the Critic make were
still relative to the working directory: skills/ (the registry then
loaded zero skills without a word), r_helpers/ (psychometrics helpers
not found) and the Critic's methodological checklist (FileNotFoundError
at CRITIQUING).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.config import PROJECT_ROOT
from src.task_template import create_task_template
from tests.test_orchestrator_terminal import _config, _orch


def test_orchestrator_built_from_another_folder_finds_skills_and_r_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.delenv("EDM_ARS_R_HELPERS", raising=False)

    orch = _orch(run_dir, _config(run_dir))

    assert orch.skill_registry.count() > 0
    assert os.environ["EDM_ARS_R_HELPERS"] == str(PROJECT_ROOT / "r_helpers")


@pytest.mark.parametrize(
    "task_type", ["prediction", "causal_soo", "causal_itr", "causal_did", "psychometrics"]
)
def test_critic_checklist_path_does_not_depend_on_the_working_directory(
    task_type: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    path = create_task_template(task_type).get_critic_checklist_path()
    assert os.path.isabs(path)
    assert os.path.isfile(path)
