"""V3.0 Phase 3b.6 / 6.1 — matcher fix verification.

Anchors the requirement that ALL FIVE M-skills (M1 regression-adjustment,
M2 PSM, M3 IPW, M4 AIPW/TMLE, M5 causal-forest-cate) reach the Analyst
stage when ``task_type=causal_soo``, alongside G1-G5 + D1.

Phase 3b.5's report flagged this as the buried lede: M-skills declared
applicable_task_types=[causal_soo] and applicable_stages=[Analyst] but
were de-prioritized out of the matcher's per-layer cap (methodology=5).
The Analyst improvised from G1-G5 + D1 without the precise method
mechanics. F-AIPW-NARROW-CI and F-COVARIATE-SET-MISMATCH may both be
downstream consequences.

3b.6's fix: bump M1-M5 priority from 2 to 1 in their SKILL.md
frontmatter; raise the methodology layer cap to 12 for task_type=
causal_soo via a task-type-specific override in the orchestrator.
This file tests both the matcher behavior and the orchestrator
integration.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.skills import SkillRegistry


SKILLS_ROOT = Path(__file__).parent.parent / "skills"

EXPECTED_M_SKILLS: frozenset[str] = frozenset({
    "causal-regression-adjustment",            # M1
    "causal-propensity-score-matching",        # M2
    "causal-inverse-probability-weighting",    # M3
    "causal-aipw-tmle",                        # M4
    "causal-forest-cate",                      # M5
})

EXPECTED_G_AND_D_SKILLS: frozenset[str] = frozenset({
    "causal-dag-identification",               # G1
    "causal-estimand-definition",              # G2
    "causal-positivity-diagnostics",           # G3
    "causal-balance-diagnostics",              # G4
    "causal-sensitivity-unmeasured-confounding",  # G5
    "hsls09-causal-conventions",               # D1
})


@pytest.fixture(scope="module")
def registry() -> SkillRegistry:
    return SkillRegistry(SKILLS_ROOT)


# ---------------------------------------------------------------------------
# 6.1 acceptance — M-skills reach Analyst for causal_soo
# ---------------------------------------------------------------------------


class TestMethodSkillsReachAnalystForCausalSOO:
    def test_all_five_m_skills_match_at_analyst_for_causal_soo(
        self, registry: SkillRegistry
    ) -> None:
        # Use the same per-layer caps the orchestrator uses for causal_soo.
        from src.orchestrator import _resolve_skill_caps

        caps = _resolve_skill_caps("causal_soo")
        matched = registry.match(
            stage="Analyst",
            task_type="causal_soo",
            dataset="hsls09_public",
            top_k_per_layer=caps,
        )
        names = {s.name for s in matched}
        missing = EXPECTED_M_SKILLS - names
        assert not missing, (
            f"M-skills missing from Analyst@causal_soo match: {sorted(missing)}. "
            f"Got: {sorted(names)}"
        )

    def test_g_and_d_skills_still_match_at_analyst_for_causal_soo(
        self, registry: SkillRegistry
    ) -> None:
        """Regression: G1-G5 + D1 must remain matched after the cap bump."""
        from src.orchestrator import _resolve_skill_caps

        caps = _resolve_skill_caps("causal_soo")
        matched = registry.match(
            stage="Analyst",
            task_type="causal_soo",
            dataset="hsls09_public",
            top_k_per_layer=caps,
        )
        names = {s.name for s in matched}
        missing = EXPECTED_G_AND_D_SKILLS - names
        assert not missing, (
            f"G+D skills missing after 6.1 changes: {sorted(missing)}"
        )

    def test_m_skills_priority_bumped_to_one(
        self, registry: SkillRegistry
    ) -> None:
        """M1-M5 frontmatter priority must be 1 for the cap-fit ranking."""
        for skill_name in EXPECTED_M_SKILLS:
            skill = registry.get(skill_name)
            assert skill is not None, f"Skill {skill_name!r} not loaded"
            assert skill.priority == 1, (
                f"{skill_name} priority should be 1, got {skill.priority}"
            )


# ---------------------------------------------------------------------------
# 6.1 acceptance — prediction codepath unchanged (regression protection)
# ---------------------------------------------------------------------------


class TestPredictionMatcherUnchanged:
    """The Option-A unblock contract from 3b.4: prediction-task behavior
    MUST be byte-identical to pre-change. The cap bump is task-type-
    scoped (causal_soo only); prediction's caps come from the unchanged
    _DEFAULT_SKILL_CAPS path.
    """

    def test_prediction_uses_default_caps(self) -> None:
        from src.orchestrator import _DEFAULT_SKILL_CAPS, _resolve_skill_caps

        # _resolve_skill_caps('prediction') falls through to defaults.
        caps = _resolve_skill_caps("prediction")
        assert caps == _DEFAULT_SKILL_CAPS

    def test_unknown_task_type_falls_back_to_defaults(self) -> None:
        from src.orchestrator import _DEFAULT_SKILL_CAPS, _resolve_skill_caps

        # Any unknown task type falls back to the prediction-shaped
        # default caps. (Future task types should be added explicitly
        # to _SKILL_CAPS_BY_TASK_TYPE rather than relying on this.)
        caps = _resolve_skill_caps("not_a_real_task_type")
        assert caps == _DEFAULT_SKILL_CAPS

    def test_prediction_analyst_match_excludes_m_skills(
        self, registry: SkillRegistry
    ) -> None:
        """Sanity: M-skills declare applicable_task_types=[causal_soo] —
        they should NOT appear in prediction-task matching at all,
        regardless of cap.
        """
        from src.orchestrator import _resolve_skill_caps

        caps = _resolve_skill_caps("prediction")
        matched = registry.match(
            stage="Analyst",
            task_type="prediction",
            dataset="hsls09_public",
            top_k_per_layer=caps,
        )
        names = {s.name for s in matched}
        leaked = EXPECTED_M_SKILLS & names
        assert not leaked, (
            f"M-skills (causal_soo only) leaked into prediction match: "
            f"{sorted(leaked)}"
        )


# ---------------------------------------------------------------------------
# 6.1 acceptance — orchestrator integration
# ---------------------------------------------------------------------------


class TestOrchestratorAttachesMSkillsForCausalSOO:
    """End-to-end: instantiate Orchestrator with task_type='causal_soo'
    and confirm _match_skills_for_stage('Analyst') includes M1-M5.
    """

    def test_orchestrator_match_skills_for_analyst_includes_m_skills(
        self, tmp_path: Path
    ) -> None:
        from src.context import PipelineContext
        from src.orchestrator import Orchestrator

        ctx = PipelineContext(
            dataset_name="hsls09_public",
            raw_data_path=str(tmp_path / "raw.csv"),
            output_dir=str(tmp_path / "orch_out"),
            task_type="causal_soo",
            max_revision_cycles=0,
        )

        config = {
            "llm_provider": "minimax",
            "models": {
                k: "x"
                for k in (
                    "problem_formulator",
                    "data_engineer",
                    "analyst",
                    "critic",
                    "writer",
                )
            },
            "minimax": {"base_url": "x", "models": {}},
            "pipeline": {
                "task_type": "causal_soo",
                "max_revision_cycles": 0,
            },
            "findings_memory": {"enabled": False},
            "review_gate": {"enabled": False},
            "paths": {
                "data_registry": "data_registry/",
                "agent_prompts": "agent_prompts/",
            },
            "sandbox": {"enabled": False},
        }

        with patch(
            "src.agents.problem_formulator.ProblemFormulator.__init__",
            return_value=None,
        ), patch(
            "src.agents.data_engineer.DataEngineer.__init__",
            return_value=None,
        ), patch(
            "src.agents.analyst.Analyst.__init__", return_value=None
        ), patch(
            "src.agents.critic.Critic.__init__", return_value=None
        ), patch(
            "src.agents.writer.Writer.__init__", return_value=None
        ), patch(
            "src.sandbox.create_executor", return_value=object()
        ):
            orch = Orchestrator(ctx, config)

        skills = orch._match_skills_for_stage("Analyst")
        names = {s.name for s in skills}
        missing = EXPECTED_M_SKILLS - names
        assert not missing, (
            f"Orchestrator._match_skills_for_stage('Analyst') missing "
            f"M-skills for causal_soo: {sorted(missing)}"
        )
