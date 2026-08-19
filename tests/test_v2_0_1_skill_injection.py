"""V2.0.1 — verify {{SKILLS}} placeholder reaches the V1 monolithic prompts.

The substitution path itself (BaseAgent.render_system_prompt) is already
covered by tests/test_orchestrator_skill_wiring.py. This file adds two
V2.0.1-specific checks:

  - Per-agent integration: each of the four V1 YAMLs (PF, Analyst, Critic,
    Writer) renders cleanly with the production SkillRegistry — no
    stranded {{SKILLS}} token, the binding-rules banner is present, and
    matched skill content actually appears in the rendered prompt.

  - Stage-mapping regression: every agent stage referenced by the
    orchestrator's _match_skills_for_stage call appears in at least one
    SKILL.md's `applicable_stages` field. Catches typos / drift between
    agent class names and skill frontmatter.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.skills import SkillRegistry, format_skills_for_prompt

# Stages the orchestrator dispatches via _inject_skills(). Source of
# truth: src/orchestrator.py call sites (each agent's _run_* method).
ORCHESTRATOR_STAGES: tuple[str, ...] = (
    "ProblemFormulator",
    "DataEngineer",
    "Analyst",
    "Critic",
    "OutlineAgent",
    "Writer",
)

# V1 monolithic agents that V2.0.1 wires {{SKILLS}} into. DE was slimmed
# in V2.0; OutlineAgent has no V1 to patch (it's small enough to ship
# unchanged).
V1_AGENTS_WIRED_IN_V2_0_1: tuple[str, ...] = (
    "problem_formulator",
    "analyst",
    "critic",
    "writer",
)

# Agent-name (lowercase, snake_case) → stage (orchestrator class-name form)
# used by Orchestrator._match_skills_for_stage.
STAGE_BY_AGENT_NAME: dict[str, str] = {
    "problem_formulator": "ProblemFormulator",
    "data_engineer": "DataEngineer",
    "analyst": "Analyst",
    "critic": "Critic",
    "outline_agent": "OutlineAgent",
    "writer": "Writer",
}


def _load_v1_system_prompt(agent: str) -> str:
    """Read the agent's system_prompt out of agent_prompts/<agent>.yaml."""
    path = Path(__file__).parent.parent / "agent_prompts" / f"{agent}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data["system_prompt"]  # type: ignore[no-any-return]


def _registry() -> SkillRegistry:
    return SkillRegistry(skills_root=Path(__file__).parent.parent / "skills")


# ---------------------------------------------------------------------------
# Test #5 — per-agent integration with the real SkillRegistry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("agent_name", V1_AGENTS_WIRED_IN_V2_0_1)
def test_v1_prompt_renders_with_real_registry(agent_name: str) -> None:
    """For each V1 agent, load its prompt + match real skills + render.

    Asserts:
      - {{SKILLS}} placeholder is present in the loaded prompt (V2.0.1 patch
        applied correctly to the YAML)
      - Rendered prompt has no remaining {{SKILLS}} token (substitution worked)
      - Rendered prompt contains the # Binding Rules section (placement
        survived rendering)
      - Rendered prompt actually contains skill content — both the canonical
        skill header convention from format_skills_for_prompt and at least
        one matched skill name
    """
    raw_prompt = _load_v1_system_prompt(agent_name)
    assert "{{SKILLS}}" in raw_prompt, (
        f"{agent_name}.yaml missing the V2.0.1 {{SKILLS}} placeholder"
    )

    stage = STAGE_BY_AGENT_NAME[agent_name]
    registry = _registry()
    skills = registry.match_and_compose(
        stage=stage,
        task_type="prediction",
        dataset="hsls09_public",
        context=(
            "Predict 11th-grade math achievement from 9th-grade factors with "
            "SHAP and subgroup fairness"
        ),
    )
    # Every V1-wired agent should match at least one skill in the production
    # registry. (If this regresses, the inventory in the V2.0.1 memo entry
    # needs an update.)
    assert len(skills) > 0, (
        f"{agent_name} matched 0 skills against the production registry; "
        "stage filter or trigger keywords likely broken"
    )

    skills_block = format_skills_for_prompt(skills).rstrip()
    rendered = raw_prompt.replace("{{SKILLS}}", skills_block)

    assert "{{SKILLS}}" not in rendered, "placeholder still present after substitution"
    assert "# Binding Rules" in rendered, (
        f"{agent_name}: binding-rules section header missing from rendered prompt"
    )
    # format_skills_for_prompt renders mandatory skills with this exact header
    # and recommended skills with `## Guidance: ...`. At least one of the two
    # must appear if any skill was matched.
    assert ("## MANDATORY RULE: " in rendered) or ("## Guidance: " in rendered), (
        f"{agent_name}: no rendered skill body found in prompt"
    )
    # Spot-check: at least one matched skill's name appears in the rendered text.
    assert any(s.name in rendered for s in skills), (
        f"{agent_name}: no matched skill name appears in rendered prompt"
    )


# ---------------------------------------------------------------------------
# Test #6 — stage-mapping regression
# ---------------------------------------------------------------------------


def test_orchestrator_stages_appear_in_skill_registry() -> None:
    """Every stage the orchestrator dispatches must be claimed by ≥1 skill.

    Catches drift between Orchestrator._match_skills_for_stage call sites
    and SKILL.md frontmatter `applicable_stages` values. A skill that
    declares an unmatched stage is fine (just unused for that stage); but
    an orchestrator stage with zero declaring skills is a wiring bug.
    """
    registry = _registry()
    # Build the union of all `applicable_stages` declared across the registry.
    declared_stages: set[str] = set()
    for skill in registry.all():
        if skill.applicable_stages:
            declared_stages.update(skill.applicable_stages)
        else:
            # Empty applicable_stages = applies to ALL stages, so this
            # skill effectively claims every orchestrator stage.
            declared_stages.update(ORCHESTRATOR_STAGES)

    for stage in ORCHESTRATOR_STAGES:
        assert stage in declared_stages, (
            f"orchestrator stage {stage!r} is not claimed by any SKILL.md "
            f"applicable_stages field; either the orchestrator should not "
            f"dispatch to this stage, or the registry is missing skills for it"
        )


# ---------------------------------------------------------------------------
# Auxiliary: prevent re-emergence of the V2.0 substitution-path duplication
# ---------------------------------------------------------------------------


def test_no_duplicate_substitution_path_in_orchestrator() -> None:
    """V2.0.1 reuses BaseAgent.render_system_prompt rather than adding a
    parallel render_agent_prompt in the orchestrator. Document and lock
    that decision so future refactors don't regress to two paths.
    """
    orchestrator_src = (
        Path(__file__).parent.parent / "src" / "orchestrator.py"
    ).read_text(encoding="utf-8")
    # Negative assertion: no top-level render_agent_prompt definition.
    assert "def render_agent_prompt" not in orchestrator_src, (
        "Orchestrator gained a parallel render_agent_prompt definition; "
        "V2.0.1 chose to keep BaseAgent.render_system_prompt as the single "
        "substitution chokepoint. Consolidate before merging."
    )
