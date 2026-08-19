"""V3.0 Phase 3b.4 / B2-B4 — task-type branching for V1 agent prompts.

These tests anchor the additive prompt-file lookup added to
``src.agents.base.load_prompt`` and the three new causal_soo prompt
files (``problem_formulator_causal_soo.yaml``, ``analyst_causal_soo.yaml``,
``writer_causal_soo.yaml``).

Strict scope (as amended by the V2.1 slim migration arc): the BASE
prompt files MUST remain the ones served on the prediction codepath,
and the legacy no-task_type path must return the same file. The
original 3b.4 wording ("V1 prediction prompts MUST remain unchanged")
predates V2.1; migrated agents (problem_formulator as of 3b.22) serve
their slim form on the same routing. Routing invariants — not V1 body
content — are what these tests anchor. Rendered-prompt content
preservation for migrated agents is asserted in the per-phase
test_v2_1_phase_3b2N_*_slim.py files.
"""
from __future__ import annotations

from typing import Any

import pytest

from src.agents.base import load_prompt


_FAKE_CONFIG: dict = {
    "paths": {"agent_prompts": "agent_prompts/"},
}


# ---------------------------------------------------------------------------
# B2 — ProblemFormulator
# ---------------------------------------------------------------------------


class TestProblemFormulatorBranching:
    def test_prediction_loads_base_prompt(self) -> None:
        # Post-3b.22 (V2.1 slim migration): the base prompt is the slim
        # form; the forbidding-causal rule now reaches the rendered
        # prompt via the prediction-research-question-design skill
        # (asserted in test_v2_1_phase_3b22_pf_slim.py, rule-08 marker).
        # This test's remaining job is ROUTING: prediction must load the
        # base file, with no causal-mode markers leaking in.
        prompt = load_prompt(
            "problem_formulator", _FAKE_CONFIG, task_type="prediction"
        )
        assert "system_prompt" in prompt
        assert "research_spec" in prompt["system_prompt"]
        # And no causal-mode-specific markers should appear.
        assert "CAUSAL_SOO" not in prompt["system_prompt"]

    def test_causal_soo_loads_causal_prompt(self) -> None:
        prompt = load_prompt(
            "problem_formulator", _FAKE_CONFIG, task_type="causal_soo"
        )
        assert "system_prompt" in prompt
        body = prompt["system_prompt"]
        # Causal-mode markers must be present.
        assert "CAUSAL_SOO" in body or "causal" in body.lower()
        assert "estimand" in body.lower()
        # And the V1 prediction prompt's forbidding-causal line must NOT
        # appear (this would be the smoking-gun for accidental fall-through).
        assert "Do not propose causal" not in body


# ---------------------------------------------------------------------------
# B3 — Analyst
# ---------------------------------------------------------------------------


class TestAnalystBranching:
    def test_prediction_loads_unmodified_v1_prompt(self) -> None:
        prompt = load_prompt("analyst", _FAKE_CONFIG, task_type="prediction")
        assert "system_prompt" in prompt
        body = prompt["system_prompt"]
        # V1 analyst prompt is prediction-coupled: AUC, SHAP, etc.
        assert "AUC" in body or "SHAP" in body
        assert "CAUSAL_SOO" not in body

    def test_causal_soo_loads_causal_prompt(self) -> None:
        prompt = load_prompt("analyst", _FAKE_CONFIG, task_type="causal_soo")
        body = prompt["system_prompt"]
        # Causal-mode markers
        assert "estimand" in body.lower()
        assert (
            "ATE" in body or "ATT" in body
        ), "causal Analyst prompt must reference an estimand label"
        # Forbidden patterns must be enumerated
        assert "SHAP" in body, "must explicitly forbid SHAP-as-causal"
        assert "AUC" in body, "must explicitly forbid AUC reporting"


# ---------------------------------------------------------------------------
# B4 — Writer
# ---------------------------------------------------------------------------


class TestWriterBranching:
    def test_prediction_loads_unmodified_v1_prompt(self) -> None:
        prompt = load_prompt("writer", _FAKE_CONFIG, task_type="prediction")
        body = prompt["system_prompt"]
        # V1 writer prompt is prediction-coupled: refers to acmart and
        # the prediction paper structure.
        assert "acmart" in body or "ACM" in body
        assert "CAUSAL_SOO" not in body

    def test_causal_soo_loads_causal_prompt(self) -> None:
        prompt = load_prompt("writer", _FAKE_CONFIG, task_type="causal_soo")
        body = prompt["system_prompt"]
        assert "estimand" in body.lower()
        assert "Identification Strategy" in body
        assert (
            "Sensitivity" in body or "E-value" in body
        ), "causal Writer prompt must require sensitivity analyses"


# ---------------------------------------------------------------------------
# Unknown task_type fails loudly
# ---------------------------------------------------------------------------


class TestUnknownTaskType:
    def test_unknown_task_type_raises_valueerror(self) -> None:
        with pytest.raises(ValueError, match="unknown task_type"):
            load_prompt(
                "problem_formulator",
                _FAKE_CONFIG,
                task_type="not_a_real_task_type",
            )

    def test_no_task_type_legacy_loads_default(self) -> None:
        # Legacy callers (no task_type kwarg) must continue to work and
        # must return the same base file the prediction path returns.
        # (Post-3b.22 the base file is the slim form, so this asserts
        # path equivalence rather than V1 body content.)
        prompt = load_prompt("problem_formulator", _FAKE_CONFIG)
        assert "system_prompt" in prompt
        via_prediction = load_prompt(
            "problem_formulator", _FAKE_CONFIG, task_type="prediction"
        )
        assert prompt["system_prompt"] == via_prediction["system_prompt"]
        assert "CAUSAL_SOO" not in prompt["system_prompt"]


# ---------------------------------------------------------------------------
# Regression: V1 prediction prompts are unchanged
# ---------------------------------------------------------------------------


class TestPredictionPromptsUnchanged:
    """The Option-A unblock contract: V1 prediction prompts MUST be
    untouched on the prediction codepath. This test cross-checks that
    ``load_prompt(agent, task_type='prediction')`` returns byte-identical
    content to the legacy ``load_prompt(agent)`` path.
    """

    @pytest.mark.parametrize(
        "agent_name", ["problem_formulator", "analyst", "writer"]
    )
    def test_prediction_path_byte_identical_to_legacy(
        self, agent_name: str
    ) -> None:
        legacy = load_prompt(agent_name, _FAKE_CONFIG)
        explicit = load_prompt(
            agent_name, _FAKE_CONFIG, task_type="prediction"
        )
        assert (
            legacy == explicit
        ), f"prediction path for {agent_name} drifted from legacy"
