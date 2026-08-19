"""V2.1 Phase 3b.25 (V4 Arc H / H3) — Analyst slim apply verification.

The largest migration (605 → 115 lines). Phase-specific locks:
- prediction-model-battery carries the authoritative hyperparameter
  grids (harvested from SPEC §4.3) and is mandatory — the per-model
  skills cap-drop at Analyst/prediction (measured: elasticnet, mlp,
  stacking dropped under the 30K non-mandatory budget), so the grids
  must live in a cap-immune carrier.
- clustered-bootstrap-ci-and-icc retagged mandatory (ICC + clustered
  CIs are SPEC-mandated reporting; its body was cap-dropped pre-tag).
- The causal Analyst path (variant prompt + M1-M5 skills) untouched.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.agents.base import load_prompt
from src.orchestrator import _resolve_skill_caps
from src.skills import SkillRegistry, format_skills_for_prompt


PROJECT_ROOT = Path(__file__).parent.parent
SKILLS_ROOT = PROJECT_ROOT / "skills"
PROMPTS_DIR = PROJECT_ROOT / "agent_prompts"
_CONFIG: dict[str, Any] = {"paths": {"agent_prompts": str(PROMPTS_DIR) + "/"}}
_CONTEXT = "Do non-cognitive factors predict college attendance beyond achievement and SES?"


@pytest.fixture(scope="module")
def registry() -> SkillRegistry:
    return SkillRegistry(SKILLS_ROOT)


@pytest.fixture(scope="module")
def slim_analyst_prompt() -> str:
    return load_prompt("analyst", _CONFIG, task_type="prediction")["system_prompt"]


def _render(registry: SkillRegistry, prompt: str, task_type: str) -> str:
    matched = registry.match_and_compose(
        stage="Analyst",
        task_type=task_type,
        dataset="hsls09_public",
        context=_CONTEXT,
        top_k_per_layer=_resolve_skill_caps(task_type),
    )
    return prompt.replace("{{SKILLS}}", format_skills_for_prompt(matched).rstrip())


class TestAnalystSlimApplied:
    def test_line_count(self, slim_analyst_prompt: str) -> None:
        assert slim_analyst_prompt.count("\n") + 1 <= 125  # V1 body ~601

    def test_no_v1_residue(self, slim_analyst_prompt: str) -> None:
        for marker in (
            "## Pilot Model Battery",
            "## Hyperparameter Tuning",
            "## SHAP Interpretability Protocol",
            "## Model Quality Gate",
        ):
            assert marker not in slim_analyst_prompt

    def test_placeholder_and_data_loading_kept(
        self, slim_analyst_prompt: str
    ) -> None:
        assert "{{SKILLS}}" in slim_analyst_prompt
        # The defensive outcome-column rule survives in the slim body.
        assert "target_col" in slim_analyst_prompt


class TestBatteryGridsCapImmune:
    def test_battery_skill_mandatory_with_grids(self) -> None:
        body = (
            SKILLS_ROOT / "task-type" / "prediction-model-battery" / "SKILL.md"
        ).read_text(encoding="utf-8")
        assert "rule_severity: mandatory" in body
        assert "n_estimators` ∈ {100, 300, 500}" in body
        assert "hidden_layer_sizes" in body
        assert "RidgeCV" in body

    def test_icc_skill_mandatory(self) -> None:
        body = (
            SKILLS_ROOT
            / "methodology"
            / "clustered-bootstrap-ci-and-icc"
            / "SKILL.md"
        ).read_text(encoding="utf-8")
        assert "rule_severity: mandatory" in body

    @pytest.mark.parametrize(
        "marker",
        [
            "n_estimators` ∈ {100, 300, 500}",
            "hidden_layer_sizes",
            "RidgeCV",
            "Clustered Bootstrap CIs and ICC",
            "KernelExplainer",
            "shap_summary.png",
            "subgroup_performance",
            "SMOTE",
        ],
    )
    def test_marker_in_rendered_prediction_prompt(
        self, registry: SkillRegistry, slim_analyst_prompt: str, marker: str
    ) -> None:
        rendered = _render(registry, slim_analyst_prompt, "prediction")
        assert marker in rendered, f"marker {marker!r} missing"


class TestCausalPathUntouched:
    def test_causal_variant_routes(self) -> None:
        causal = load_prompt("analyst", _CONFIG, task_type="causal_soo")["system_prompt"]
        base = load_prompt("analyst", _CONFIG, task_type="prediction")["system_prompt"]
        assert causal != base and "{{SKILLS}}" in causal

    def test_no_battery_leak_into_causal(self, registry: SkillRegistry) -> None:
        matched = registry.match_and_compose(
            stage="Analyst",
            task_type="causal_soo",
            dataset="hsls09_public",
            context="ATT propensity matching",
            top_k_per_layer=_resolve_skill_caps("causal_soo"),
        )
        names = {s.name for s in matched}
        assert "prediction-model-battery" not in names
        assert any(n.startswith("causal-") for n in names)


class TestAnalystRoleAndBackup:
    def test_role_preserved(self) -> None:
        d = load_prompt("analyst", _CONFIG, task_type="prediction")
        assert d["agent_name"] == "Analyst"
        assert d["temperature"] == 0.0

    def test_v1_backup(self) -> None:
        text = (PROMPTS_DIR / "analyst.v1.yaml.bak").read_text(encoding="utf-8")
        assert "## Pilot Model Battery" in text
        assert "## Model Quality Gate" in text
