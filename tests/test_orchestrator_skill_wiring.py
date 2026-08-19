"""Phase 2c — orchestrator + agent skill-wiring tests.

Verify the {{SKILLS}} placeholder rendering, the orchestrator's registry
load, and the per-stage context construction. None of these tests make a
real LLM call — agents that would call the API are stubbed.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.skills import Skill, SkillRegistry

# Tests in this module exercise BaseAgent and the Orchestrator directly,
# but stub the LLM client so no API calls happen. Anthropic and MiniMax
# are constructed in BaseAgent.__init__; conftest already sets fake API
# key envs, so client construction is safe.


# ---------------------------------------------------------------------------
# render_system_prompt
# ---------------------------------------------------------------------------


def _make_base_agent(monkeypatch: pytest.MonkeyPatch, system_prompt: str):
    """Construct a minimal BaseAgent subclass without going through agent_prompts/."""
    from src.agents.base import BaseAgent

    class _StubAgent(BaseAgent):
        def run(self, **kwargs):  # type: ignore[override]
            return None

    # Patch the prompt loader so the stub's system prompt comes from the test.
    monkeypatch.setattr(
        "src.agents.base.load_prompt",
        lambda name, config, **kwargs: {
            "system_prompt": system_prompt, "temperature": 0.0,
        },
    )
    # Patch out the executor + Anthropic client so init succeeds fast.
    monkeypatch.setattr("src.sandbox.create_executor", lambda config: object())
    with patch("anthropic.Anthropic"):
        agent = _StubAgent(
            context=_FakeCtx(),
            agent_name="StubAgent",
            config=_FAKE_CONFIG,
        )
    return agent


class _FakeCtx:
    dataset_name = "hsls09_public"
    task_type = "prediction"
    output_dir = "/tmp"
    log: list = []


_FAKE_CONFIG: dict = {
    "llm_provider": "minimax",
    "models": {
        "problem_formulator": "claude-sonnet-4-6",
        "data_engineer": "claude-sonnet-4-6",
        "analyst": "claude-sonnet-4-6",
        "critic": "claude-opus-4-6",
        "writer": "claude-sonnet-4-6",
        "stubagent": "claude-sonnet-4-6",
    },
    "minimax": {
        "base_url": "https://example.invalid/anthropic",
        "models": {"stubagent": "MiniMax-stub"},
    },
    "paths": {"data_registry": "data_registry/", "agent_prompts": "agent_prompts/"},
    "sandbox": {"enabled": False},
}


class TestRenderSystemPrompt:
    def test_render_no_placeholder_returns_prompt_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _make_base_agent(monkeypatch, "You are an agent. Do the task.")
        agent.skills = [_make_skill("any", "methodology")]
        rendered = agent.render_system_prompt()
        assert rendered == "You are an agent. Do the task."

    def test_render_with_placeholder_but_no_skills_removes_placeholder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _make_base_agent(
            monkeypatch,
            "You are an agent.\n\n{{SKILLS}}\n\nDo the task.",
        )
        agent.skills = []  # explicit empty
        rendered = agent.render_system_prompt()
        assert "{{SKILLS}}" not in rendered
        assert "You are an agent." in rendered
        assert "Do the task." in rendered

    def test_render_with_placeholder_and_skills_splices_content(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _make_base_agent(
            monkeypatch,
            "You are an agent.\n\n{{SKILLS}}\n\nDo the task.",
        )
        agent.skills = [
            _make_skill("skill-a", "methodology", body="Skill A body."),
            _make_skill("skill-b", "writing", body="Skill B body."),
        ]
        rendered = agent.render_system_prompt()
        assert "{{SKILLS}}" not in rendered
        assert "## Guidance: skill-a" in rendered
        assert "## Guidance: skill-b" in rendered
        assert "Skill A body." in rendered
        assert "Skill B body." in rendered

    def test_render_with_no_skills_attribute_returns_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Backward-compat: pre-Phase-2c agents that never had .skills set."""
        agent = _make_base_agent(monkeypatch, "You are an agent. {{SKILLS}}")
        # Don't touch agent.skills at all — should default to None and pass through.
        rendered = agent.render_system_prompt()
        # When skills is None, treat like empty: placeholder should be removed
        # rather than leaked into the LLM prompt.
        assert "{{SKILLS}}" not in rendered


# ---------------------------------------------------------------------------
# Orchestrator wiring
# ---------------------------------------------------------------------------


class TestOrchestratorRegistry:
    def test_orchestrator_loads_skill_registry_at_init(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Import lazily so module-level Anthropic patches stay clean.
        from src.context import PipelineContext
        from src.orchestrator import Orchestrator

        ctx = PipelineContext(
            dataset_name="hsls09_public",
            raw_data_path="/nonexistent/raw.csv",
            output_dir=str(Path.cwd() / "tmp_orch_test"),
            task_type="prediction",
            max_revision_cycles=0,
        )
        # Stub all five agent classes' __init__ + run so no LLM-construction happens.
        with patch("src.agents.problem_formulator.ProblemFormulator.__init__", return_value=None), \
             patch("src.agents.data_engineer.DataEngineer.__init__", return_value=None), \
             patch("src.agents.analyst.Analyst.__init__", return_value=None), \
             patch("src.agents.critic.Critic.__init__", return_value=None), \
             patch("src.agents.writer.Writer.__init__", return_value=None), \
             patch("src.sandbox.create_executor", return_value=object()):
            config = {
                "llm_provider": "minimax",
                "models": {k: "x" for k in
                           ("problem_formulator", "data_engineer", "analyst", "critic", "writer")},
                "minimax": {"base_url": "x", "models": {}},
                "pipeline": {"task_type": "prediction", "max_revision_cycles": 0},
                "findings_memory": {"enabled": False},
                "review_gate": {"enabled": False},
                "paths": {"data_registry": "data_registry/", "agent_prompts": "agent_prompts/"},
                "sandbox": {"enabled": False},
            }
            orch = Orchestrator(ctx, config)
        assert isinstance(orch.skill_registry, SkillRegistry)
        # Should pick up: 41 V2.0/V2.0.1 + 6 V3.0 Phase 3b.1 (G1-G5
        # methodology + D1 dataset) + 5 V3.0 Phase 3b.2 method skills
        # (M1 regression-adjustment, M2 PSM, M3 IPW, M4 AIPW/TMLE,
        # M5 causal-forest-cate) + 1 V3.0 Phase 3b.12 (causal-data-
        # engineer-contract). Total 53.
        assert orch.skill_registry.count() == 70  # ... +assistments-conventions +natural-academic-prose (E2)

    def test_stage_context_for_analyst_pulls_expected_skills(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.context import PipelineContext
        from src.orchestrator import Orchestrator

        ctx = PipelineContext(
            dataset_name="hsls09_public",
            raw_data_path="/nonexistent/raw.csv",
            output_dir=str(Path.cwd() / "tmp_orch_test2"),
            task_type="prediction",
            max_revision_cycles=0,
        )
        ctx.research_spec = {
            "research_question": "Predict 11th-grade math achievement with SHAP feature importance",
        }

        with patch("src.agents.problem_formulator.ProblemFormulator.__init__", return_value=None), \
             patch("src.agents.data_engineer.DataEngineer.__init__", return_value=None), \
             patch("src.agents.analyst.Analyst.__init__", return_value=None), \
             patch("src.agents.critic.Critic.__init__", return_value=None), \
             patch("src.agents.writer.Writer.__init__", return_value=None), \
             patch("src.sandbox.create_executor", return_value=object()):
            config = {
                "llm_provider": "minimax",
                "models": {k: "x" for k in
                           ("problem_formulator", "data_engineer", "analyst", "critic", "writer")},
                "minimax": {"base_url": "x", "models": {}},
                "pipeline": {"task_type": "prediction", "max_revision_cycles": 0},
                "findings_memory": {"enabled": False},
                "review_gate": {"enabled": False},
                "paths": {"data_registry": "data_registry/", "agent_prompts": "agent_prompts/"},
                "sandbox": {"enabled": False},
            }
            orch = Orchestrator(ctx, config)

        skills = orch._match_skills_for_stage("Analyst")
        names = {s.name for s in skills}
        # Must include the SHAP methodology skill (keyword-driven match) and
        # the prediction model battery (task-type match).
        assert "shap-explainer-selection" in names
        assert "prediction-model-battery" in names
        # Composition pulls in all six per-family model skills.
        for family in (
            "model-logistic-regression",
            "model-random-forest",
            "model-xgboost",
            "model-elasticnet",
            "model-mlp",
            "model-stacking-ensemble",
        ):
            assert family in names


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_skill(name: str, layer: str, *, body: str = "") -> Skill:
    return Skill(
        name=name,
        layer=layer,
        description=f"Stub skill {name}.",
        body=body or f"Body of {name}.",
    )
