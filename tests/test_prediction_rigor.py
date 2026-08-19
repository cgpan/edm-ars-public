"""V4 stream-2 — prediction rigor extensions: helpers + skill wiring."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _synthetic(n: int = 2500, interaction: float = 0.6, seed: int = 0):
    rng = np.random.default_rng(seed)
    ses = rng.normal(0, 1, n)
    exp1 = (rng.random(n) < 0.5).astype(float)
    cov = rng.normal(0, 1, n)
    logit = -0.3 + 0.5 * exp1 + 0.4 * ses + interaction * exp1 * ses
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(float)
    X = pd.DataFrame({"EXP_1": exp1, "SES": ses, "COV": cov})
    return X, y


class TestModerationAnalysis:
    def test_detects_true_interaction(self) -> None:
        from src.analysis_helpers import run_moderation_analysis

        X, y = _synthetic(interaction=0.6)
        out = run_moderation_analysis(X, y, ["EXP_1"], "SES", n_boot=40)
        assert out["status"] == "computed"
        assert out["lrt_p"] < 0.01
        assert "significant" in out["interpretation"]

    def test_null_interaction_not_flagged(self) -> None:
        from src.analysis_helpers import run_moderation_analysis

        # replicated over seeds: null should rarely cross alpha
        flagged = 0
        for seed in range(4):
            X, y = _synthetic(interaction=0.0, seed=seed)
            out = run_moderation_analysis(X, y, ["EXP_1"], "SES", n_boot=40)
            if out["lrt_p"] < 0.05:
                flagged += 1
        assert flagged <= 1

    def test_missing_columns_skips_gracefully(self) -> None:
        from src.analysis_helpers import run_moderation_analysis

        X, y = _synthetic()
        out = run_moderation_analysis(X, y, ["NOPE"], "SES")
        assert out["status"] == "skipped"


class TestGroupShapByParent:
    def test_groups_dummies_and_sorts(self) -> None:
        from src.analysis_helpers import group_shap_by_parent

        out = group_shap_by_parent(
            ["BYSTEXP_5.0", "BYSTEXP_6.0", "BYSES1", "BYRACE_White"],
            [0.5, 0.4, 0.33, 0.2],
        )
        assert out[0]["parent"] == "BYSTEXP"
        assert out[0]["total_shap_mean_abs"] == pytest.approx(0.9)
        assert out[0]["n_columns"] == 2
        assert [d["parent"] for d in out] == ["BYSTEXP", "BYSES1", "BYRACE"]


class TestBootstrapAucDifference:
    def test_real_difference_significant(self) -> None:
        from src.analysis_helpers import bootstrap_auc_difference

        rng = np.random.default_rng(1)
        n = 2000
        signal = rng.normal(0, 1, n)
        y = (rng.random(n) < 1 / (1 + np.exp(-signal))).astype(float)
        pa = 1 / (1 + np.exp(-(signal + rng.normal(0, 0.3, n))))
        pb = rng.random(n)  # noise model
        out = bootstrap_auc_difference(y, pa, pb, n_boot=300)
        assert out["auc_diff"] > 0.1
        assert out["significant"] is True
        assert out["se_method"] == "bootstrap"

    def test_cluster_aware_mode(self) -> None:
        from src.analysis_helpers import bootstrap_auc_difference

        rng = np.random.default_rng(2)
        n = 1200
        sid = rng.integers(0, 60, n)
        signal = rng.normal(0, 1, n)
        y = (rng.random(n) < 1 / (1 + np.exp(-signal))).astype(float)
        pa = 1 / (1 + np.exp(-(signal + rng.normal(0, 0.3, n))))
        out = bootstrap_auc_difference(y, pa, pa, school_ids=sid, n_boot=200)
        assert out["se_method"] == "cluster_bootstrap"
        assert out["significant"] is False  # identical models: no difference
        assert out["ci_lower"] <= 0.0 <= out["ci_upper"]


class TestCalibrationMetrics:
    def test_well_calibrated_probs(self) -> None:
        from src.analysis_helpers import compute_calibration_metrics

        rng = np.random.default_rng(3)
        p = rng.uniform(0.05, 0.95, 5000)
        y = (rng.random(5000) < p).astype(float)
        out = compute_calibration_metrics(y, p)
        assert out["ece"] < 0.05
        assert 0.85 < out["calibration_slope"] < 1.15

    def test_overconfident_probs_flagged_by_slope(self) -> None:
        from src.analysis_helpers import compute_calibration_metrics

        rng = np.random.default_rng(4)
        base = rng.uniform(0.2, 0.8, 5000)
        y = (rng.random(5000) < base).astype(float)
        # push probabilities toward extremes -> slope < 1
        over = np.clip(base + np.sign(base - 0.5) * 0.18, 0.01, 0.99)
        out = compute_calibration_metrics(y, over)
        assert out["calibration_slope"] < 0.85


class TestRigorSkillWiring:
    @pytest.mark.parametrize("stage", ["Analyst", "Critic", "Writer"])
    def test_rigor_skill_reaches_stage(self, stage: str) -> None:
        from src.orchestrator import _resolve_skill_caps
        from src.skills import SkillRegistry
        from src.skills.composer import format_skills_for_prompt

        registry = SkillRegistry(str(PROJECT_ROOT / "skills"))
        skills = registry.match_and_compose(
            task_type="prediction",
            dataset="els_2002",
            stage=stage,
            context="moderation calibration SHAP rigor",
            top_k_per_layer=_resolve_skill_caps("prediction"),
        )
        names = [s.name for s in skills]
        assert "prediction-rigor-extensions" in names
        rendered = format_skills_for_prompt(skills)
        assert "run_moderation_analysis" in rendered
        assert "group_shap_by_parent" in rendered

    def test_els_conventions_carries_cluster_recipe(self) -> None:
        from src.skills import SkillRegistry

        registry = SkillRegistry(str(PROJECT_ROOT / "skills"))
        skills = registry.match_and_compose(
            task_type="prediction",
            dataset="els_2002",
            stage="DataEngineer",
            context="ELS sentinel school cluster split",
            top_k_per_layer=None,
        )
        conventions = next(s for s in skills if s.name == "els-2002-conventions")
        assert "F1SCH_ID" in conventions.body
        assert "singleton" in conventions.body

    def test_analyst_prompt_carries_rigor_schema(self) -> None:
        from src.agents.base import load_prompt

        config = {"paths": {"agent_prompts": str(PROJECT_ROOT / "agent_prompts")}}
        body = load_prompt("analyst", config)["system_prompt"]
        for field in ("moderation_analysis", "top_feature_groups",
                      "model_comparison_test", "calibration"):
            assert field in body, field
