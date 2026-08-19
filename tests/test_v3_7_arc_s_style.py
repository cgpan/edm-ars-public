"""V3.7 Arc S — narrative archetypes + formulaic-construction ban."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.orchestrator import _resolve_skill_caps
from src.skills import SkillRegistry, format_skills_for_prompt

SKILLS_ROOT = Path(__file__).parent.parent / "skills"


@pytest.fixture(scope="module")
def registry() -> SkillRegistry:
    return SkillRegistry(SKILLS_ROOT)


@pytest.mark.parametrize("stage", ["Writer", "OutlineAgent"])
@pytest.mark.parametrize("task_type", ["prediction", "causal_soo", "causal_itr"])
def test_archetypes_render_everywhere(
    registry: SkillRegistry, stage: str, task_type: str
) -> None:
    m = registry.match_and_compose(
        stage=stage, task_type=task_type, dataset="hsls09_public",
        context="paper narrative style",
        top_k_per_layer=_resolve_skill_caps(task_type),
    )
    block = format_skills_for_prompt(m)
    assert "Selection rule (apply in order" in block
    assert "Null-result paper" in block


@pytest.mark.parametrize("task_type", ["prediction", "causal_itr"])
def test_ban_list_renders_at_writer(
    registry: SkillRegistry, task_type: str
) -> None:
    m = registry.match_and_compose(
        stage="Writer", task_type=task_type, dataset="hsls09_public",
        context="paper style prose",
        top_k_per_layer=_resolve_skill_caps(task_type),
    )
    block = format_skills_for_prompt(m)
    for marker in (
        "Stock phrases (never use)",
        "Rule-of-three padding",
        "More research is needed",
        "Concreteness requirements",
    ):
        assert marker in block, marker


def test_both_skills_mandatory() -> None:
    for name in ("paper-narrative-archetypes", "formulaic-construction-ban"):
        body = (SKILLS_ROOT / "writing" / name / "SKILL.md").read_text(encoding="utf-8")
        assert "rule_severity: mandatory" in body, name
