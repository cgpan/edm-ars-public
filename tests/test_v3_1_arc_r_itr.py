"""V3.1 Arc R — causal_itr task type: offline verification + synthetic gate.

R1: registration, spec validation, gate widening, skill matching,
rendered-prompt markers per stage, no cross-task leaks.
R2: the synthetic-DGP gate (known-optimal-rule recovery + null-DGP
no-false-rule) — the standing discipline for every new estimator
battery from the V4 roadmap onward.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.agents.base import load_prompt
from src.orchestrator import _resolve_skill_caps
from src.skills import SkillRegistry, format_skills_for_prompt
from src.task_template import create_task_template

PROJECT_ROOT = Path(__file__).parent.parent
SKILLS_ROOT = PROJECT_ROOT / "skills"
_CONFIG: dict[str, Any] = {
    "paths": {"agent_prompts": str(PROJECT_ROOT / "agent_prompts") + "/"}
}
_CONTEXT = "For whom does raising math self-efficacy change college attendance — learn a targeting rule"


def _spec() -> dict:
    return {
        "task_type": "causal_itr",
        "treatment": {"variable": "X1MTHEFF", "operationalization": "median_split_binary"},
        "outcome": {"variable": "X4EVRATNDCLG", "type": "binary"},
        "target_estimand_hint": "policy value of a learned rule vs best constant",
        "adjustment_set": ["X1SES", "X1TXMTSCOR", "X1SEX", "X1SCHOOLBEL"],
        "rule_covariates": ["X1SES", "X1TXMTSCOR", "X1SEX"],
        "primary_method": "M6",
        "secondary_methods": ["M5", "M1"],
    }


@pytest.fixture(scope="module")
def registry() -> SkillRegistry:
    return SkillRegistry(SKILLS_ROOT)


class TestTaskTypeRegistration:
    def test_template_registered(self) -> None:
        t = create_task_template("causal_itr")
        assert t.get_name() == "causal_itr"
        assert t.get_evaluation_metrics("binary")["primary"] == "POLICY_VALUE"

    def test_valid_spec_passes(self) -> None:
        assert create_task_template("causal_itr").validate_research_spec(_spec()) == []

    @pytest.mark.parametrize(
        "mutation,expected",
        [
            ({"rule_covariates": None}, "rule_covariates is required"),
            ({"rule_covariates": ["X1PAREDU"]}, "not in adjustment_set"),
            ({"primary_method": "M2"}, "primary_method must be 'M6'"),
            ({"task_type": "causal_soo"}, "task_type must be 'causal_itr'"),
        ],
    )
    def test_invalid_specs_flagged(self, mutation: dict, expected: str) -> None:
        spec = {**_spec(), **mutation}
        warnings = create_task_template("causal_itr").validate_research_spec(spec)
        assert any(expected in w for w in warnings), warnings


class TestGateWidening:
    def test_contract_applies_to_itr(self, tmp_path: Path) -> None:
        import pandas as pd

        from src.causal_data_contract import (
            CausalDataContractError,
            assert_causal_soo_data_contract,
        )

        pd.DataFrame({"X1SES": [1.0, 2.0]}).to_csv(
            tmp_path / "train_X.csv", index=False
        )
        with pytest.raises(CausalDataContractError):
            assert_causal_soo_data_contract(tmp_path / "train_X.csv", _spec())

    def test_pre_critic_accepts_itr(self, tmp_path: Path) -> None:
        from types import SimpleNamespace

        from src.pre_critic_checks import run_pre_critic_checks

        ctx = SimpleNamespace(
            research_spec=_spec(),
            results_object={"sensitivity": {}},
            data_report={"validation_passed": True},
        )
        result = run_pre_critic_checks(ctx, str(tmp_path), task_type="causal_itr")
        # The refuter assertion fires for ITR too (empty sensitivity).
        assert any(f.check_id == "pcc_c01" for f in result.failures)


class TestVariantPromptsRoute:
    @pytest.mark.parametrize("agent", ["problem_formulator", "analyst", "writer"])
    def test_itr_variant_loads(self, agent: str) -> None:
        body = load_prompt(agent, _CONFIG, task_type="causal_itr")["system_prompt"]
        assert "CAUSAL_ITR" in body
        assert "{{SKILLS}}" in body


class TestRenderedPromptsPerStage:
    @pytest.mark.parametrize(
        "stage,agent,markers",
        [
            ("Analyst", "analyst", [
                "DR pseudo-outcome", "weighted classification",
                "value_gain_vs_best_constant", "cluster_bootstrap",
                "rule_covariates", "POLICY_VALUE",
            ]),
            ("Critic", "critic", [
                "itr_01", "itr_05", "cross-fitted", "subgroup_value_parity",
            ]),
            ("Writer", "writer", [
                "rule card", "Policy-value table", "no-detectable-benefit",
            ]),
        ],
    )
    def test_markers_reach_rendered_prompt(
        self, registry: SkillRegistry, stage: str, agent: str, markers: list[str]
    ) -> None:
        prompt = load_prompt(agent, _CONFIG, task_type="causal_itr")["system_prompt"]
        matched = registry.match_and_compose(
            stage=stage,
            task_type="causal_itr",
            dataset="hsls09_public",
            context=_CONTEXT,
            top_k_per_layer=_resolve_skill_caps("causal_itr"),
        )
        rendered = prompt.replace(
            "{{SKILLS}}", format_skills_for_prompt(matched).rstrip()
        )
        for m in markers:
            assert m in rendered, f"{stage}: marker {m!r} missing"

    def test_no_itr_leak_into_soo_or_prediction(
        self, registry: SkillRegistry
    ) -> None:
        for tt in ("causal_soo", "prediction"):
            matched = registry.match_and_compose(
                stage="Analyst",
                task_type=tt,
                dataset="hsls09_public",
                context=_CONTEXT,
                top_k_per_layer=_resolve_skill_caps(tt),
            )
            names = {s.name for s in matched}
            assert "causal-itr-policy-learning" not in names, tt
            assert "critic-checklist-causal-itr" not in names, tt


class TestSyntheticGate:
    """R2 — smaller n for test speed; thresholds unchanged."""

    def test_gate_passes(self) -> None:
        from scripts.itr_synthetic_gate import run_gate

        result = run_gate(n=4000)
        assert result["heterogeneous"]["passed"], result
        assert result["null"]["passed"], result
        assert result["gate_passed"]

    def test_heterogeneous_recovers_oracle(self) -> None:
        from scripts.itr_synthetic_gate import (
            _dr_pseudo_outcomes,
            learn_policy_tree,
            make_dgp,
        )

        df = make_dgp("heterogeneous", n=4000)
        tree = learn_policy_tree(df, _dr_pseudo_outcomes(df))
        pi = tree.predict(df[["X1SES", "X1TXMTSCOR"]].to_numpy())
        oracle = (df["_tau"].to_numpy() > 0).astype(int)
        assert (pi == oracle).mean() >= 0.80
