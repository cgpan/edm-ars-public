"""V3.0 Phase 3b.4 / B1 — CausalSOOTemplate registration + dispatch tests.

These tests anchor the structural contract added in 3b.4 sub-wave 1:

  1. Template registers under the ``"causal_soo"`` key in
     ``_TASK_REGISTRY`` and instantiates without raising.
  2. ``validate_research_spec`` accepts the locked smoke-test fixture
     and rejects malformed specs with informative messages.
  3. ``dispatch_analysis`` returns one config per method in
     ``primary + comparator + secondary``, with the method-specific
     skill appended to each config.
  4. The orchestrator can be instantiated with ``task_type='causal_soo'``
     without raising — the ``ValueError`` from the missing template
     registration that blocked Phase 3b.3 no longer occurs.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.task_template import (
    CausalSOOTemplate,
    PredictionTemplate,
    TaskTemplate,
    create_task_template,
)


FIXTURE_PATH = (
    Path(__file__).parent.parent
    / "runs"
    / "fixtures"
    / "spec_x1mtheff_x4college.json"
)


@pytest.fixture
def locked_spec() -> dict:
    """Load the Phase 3b.4 / 3b.5 smoke-test locked spec."""
    with open(FIXTURE_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 1. Registration
# ---------------------------------------------------------------------------


class TestCausalSOOTemplateRegistration:
    def test_registers_under_causal_soo_key(self) -> None:
        template = create_task_template("causal_soo")
        assert isinstance(template, CausalSOOTemplate)
        assert isinstance(template, TaskTemplate)
        assert template.get_name() == "causal_soo"

    def test_prediction_still_routes_correctly(self) -> None:
        # Regression: causal_soo registration must not displace prediction.
        template = create_task_template("prediction")
        assert isinstance(template, PredictionTemplate)
        assert template.get_name() == "prediction"

    def test_unknown_still_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown task_type"):
            create_task_template("nonexistent_task_type")


# ---------------------------------------------------------------------------
# 2. Structural validation
# ---------------------------------------------------------------------------


class TestValidateResearchSpec:
    def setup_method(self) -> None:
        self.template = CausalSOOTemplate()

    def test_locked_smoke_test_spec_validates_clean(
        self, locked_spec: dict
    ) -> None:
        """The Phase 3b.5 smoke-test fixture must pass structural validation."""
        warnings = self.template.validate_research_spec(locked_spec)
        assert warnings == [], (
            f"Locked smoke-test spec at {FIXTURE_PATH} failed validation: "
            f"{warnings}"
        )

    def test_missing_treatment_block_flagged(self) -> None:
        spec = {
            "task_type": "causal_soo",
            "outcome": {"variable": "X4EVRATNDCLG", "type": "binary"},
            "target_estimand_hint": "ATT",
            "primary_method": "M2",
        }
        warnings = self.template.validate_research_spec(spec)
        assert any("treatment" in w for w in warnings)

    def test_missing_outcome_block_flagged(self) -> None:
        spec = {
            "task_type": "causal_soo",
            "treatment": {
                "variable": "X1MTHEFF",
                "operationalization": "median_split_binary",
            },
            "target_estimand_hint": "ATT",
            "primary_method": "M2",
        }
        warnings = self.template.validate_research_spec(spec)
        assert any("outcome" in w for w in warnings)

    def test_wrong_task_type_flagged(self) -> None:
        spec = {
            "task_type": "prediction",
            "treatment": {
                "variable": "X1MTHEFF",
                "operationalization": "median_split_binary",
            },
            "outcome": {"variable": "X4EVRATNDCLG", "type": "binary"},
            "target_estimand_hint": "ATT",
            "primary_method": "M2",
        }
        warnings = self.template.validate_research_spec(spec)
        assert any("task_type" in w and "causal_soo" in w for w in warnings)

    def test_unsupported_primary_method_flagged(self) -> None:
        spec = {
            "task_type": "causal_soo",
            "treatment": {
                "variable": "X1MTHEFF",
                "operationalization": "median_split_binary",
            },
            "outcome": {"variable": "X4EVRATNDCLG", "type": "binary"},
            "target_estimand_hint": "ATT",
            "primary_method": "M99",
        }
        warnings = self.template.validate_research_spec(spec)
        assert any("primary_method" in w and "M99" in w for w in warnings)

    def test_missing_primary_method_flagged(self) -> None:
        spec = {
            "task_type": "causal_soo",
            "treatment": {
                "variable": "X1MTHEFF",
                "operationalization": "median_split_binary",
            },
            "outcome": {"variable": "X4EVRATNDCLG", "type": "binary"},
            "target_estimand_hint": "ATT",
        }
        warnings = self.template.validate_research_spec(spec)
        assert any("primary_method" in w for w in warnings)


# ---------------------------------------------------------------------------
# 3. Dispatch methods
# ---------------------------------------------------------------------------


class TestDispatchAnalysis:
    def setup_method(self) -> None:
        self.template = CausalSOOTemplate()

    def test_returns_one_config_per_method(self, locked_spec: dict) -> None:
        configs = self.template.dispatch_analysis(locked_spec)
        # 3b.7 locked spec: primary=M2, comparator=M1,
        # secondary=[M3, M4, M5], exclude=[] → expect 5 configs in
        # order M2, M1, M3, M4, M5.
        assert [c["method_id"] for c in configs] == [
            "M2", "M1", "M3", "M4", "M5",
        ]

    def test_excluded_methods_dropped_with_handcrafted_spec(self) -> None:
        """Generic exclusion test using a hand-crafted spec; decoupled
        from the live fixture (which now has empty exclude_methods for
        3b.7)."""
        spec = {
            "task_type": "causal_soo",
            "primary_method": "M2",
            "comparator_method": "M1",
            "secondary_methods": ["M3", "M5"],
            "exclude_methods": ["M5"],
        }
        configs = self.template.dispatch_analysis(spec)
        method_ids = [c["method_id"] for c in configs]
        assert method_ids == ["M2", "M1", "M3"]
        assert "M5" not in method_ids

    def test_method_skill_attached_per_config(
        self, locked_spec: dict
    ) -> None:
        configs = self.template.dispatch_analysis(locked_spec)
        skill_by_method = {c["method_id"]: c["method_skill"] for c in configs}
        assert skill_by_method["M1"] == "causal-regression-adjustment"
        assert skill_by_method["M2"] == "causal-propensity-score-matching"
        assert skill_by_method["M3"] == "causal-inverse-probability-weighting"
        assert skill_by_method["M4"] == "causal-aipw-tmle"
        assert skill_by_method["M5"] == "causal-forest-cate"

    def test_each_config_includes_standing_methodology_skills(
        self, locked_spec: dict
    ) -> None:
        configs = self.template.dispatch_analysis(locked_spec)
        for cfg in configs:
            skills = cfg["skills"]
            # Standing methodology skills attached at the Analyst stage
            # per CausalSOOTemplate._STAGE_SKILLS["Analyst"].
            assert "causal-dag-identification" in skills
            assert "causal-estimand-definition" in skills
            assert "causal-positivity-diagnostics" in skills
            assert "causal-balance-diagnostics" in skills
            assert "causal-sensitivity-unmeasured-confounding" in skills
            assert "hsls09-causal-conventions" in skills
            # And the method-specific skill.
            assert cfg["method_skill"] in skills

    def test_dispatch_data_engineering_passes_through_spec(
        self, locked_spec: dict
    ) -> None:
        cfg = self.template.dispatch_data_engineering(locked_spec)
        assert cfg["treatment"] == locked_spec["treatment"]
        assert cfg["outcome"] == locked_spec["outcome"]
        assert "hsls09-causal-conventions" in cfg["skills"]


# ---------------------------------------------------------------------------
# 4. Orchestrator instantiation regression
# ---------------------------------------------------------------------------


class TestOrchestratorInstantiatesForCausalSOO:
    """The blocker from Phase 3b.3: Orchestrator.__init__ raised
    ValueError on task_type='causal_soo' because no template was
    registered. This test asserts the unblock landed.
    """

    def test_orchestrator_init_with_causal_soo_does_not_raise(
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

        # Stub all five agents and the executor so init doesn't reach
        # the LLM construction path.
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

        assert isinstance(orch.task_template, CausalSOOTemplate)
        assert orch.task_template.get_name() == "causal_soo"
