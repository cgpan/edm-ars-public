"""Phase B — causal_did task type: helpers, template, adapter, variants,
and orchestrator-path skill rendering."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# M8 helpers — synthetic recovery (replicated over seeds, per the Arc Q rule)
# ---------------------------------------------------------------------------

def _make_panel(seed: int, true_did: float, n_per_cell: int = 4000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for cohort in (0, 1):
        for group in (0, 1):
            base = 50.0 + (-10.0 if group else 10.0)
            if cohort and group:
                base += true_did
            y = rng.normal(base, 8.0, n_per_cell)
            rows.append(pd.DataFrame({
                "cohort": cohort, "low_ses": group,
                "rank_base": y,
                "rank_follow": y + rng.normal(0, 2.0, n_per_cell),
            }))
    return pd.concat(rows, ignore_index=True)


class TestDIDHelpers:
    def test_recovers_true_gap_change_across_seeds(self) -> None:
        from src.analysis_helpers import did_gap_in_gaps

        errs = []
        ci_covers = 0
        for seed in range(5):
            df = _make_panel(seed, true_did=-3.0)
            est = did_gap_in_gaps(
                df, "rank_base", "low_ses", "cohort",
                n_boot=200, random_state=seed,
            )
            errs.append(abs(est["point_estimate"] - (-3.0)))
            if est["ci_lower"] <= -3.0 <= est["ci_upper"]:
                ci_covers += 1
            assert est["ci_lower"] < est["point_estimate"] < est["ci_upper"]
            assert est["se_method"] == "stratified_bootstrap"
        # DiD SE at n=4000/cell, sd=8 is ~0.25; mean |err| ~ 0.20
        assert float(np.mean(errs)) < 0.5, f"mean abs error {np.mean(errs)}"
        assert ci_covers >= 4  # the CI is the certification, not the point

    def test_null_effect_ci_covers_zero(self) -> None:
        from src.analysis_helpers import did_gap_in_gaps

        covered = 0
        for seed in range(5):
            df = _make_panel(seed, true_did=0.0)
            est = did_gap_in_gaps(
                df, "rank_base", "low_ses", "cohort",
                n_boot=200, random_state=seed,
            )
            if est["ci_lower"] <= 0.0 <= est["ci_upper"]:
                covered += 1
        assert covered >= 4  # 95% CI should almost always cover the null

    def test_placebo_probe_flags_divergent_follow_wave(self) -> None:
        from src.analysis_helpers import did_placebo_follow_wave

        df = _make_panel(0, true_did=-3.0)
        # stable follow wave -> no flag
        probe = did_placebo_follow_wave(
            df, "rank_base", "rank_follow", "low_ses", "cohort",
            n_boot=200, random_state=0,
        )
        assert probe["wave_instability_flag"] is False
        # inject a large divergence in the follow wave for post-group1
        df2 = df.copy()
        mask = (df2["cohort"] == 1) & (df2["low_ses"] == 1)
        df2.loc[mask, "rank_follow"] += 15.0
        probe2 = did_placebo_follow_wave(
            df2, "rank_base", "rank_follow", "low_ses", "cohort",
            n_boot=200, random_state=0,
        )
        assert probe2["wave_instability_flag"] is True


# ---------------------------------------------------------------------------
# Task type / adapter / selector wiring
# ---------------------------------------------------------------------------

class TestCausalDIDWiring:
    def test_template_registered_and_validates(self) -> None:
        from src.task_template import create_task_template

        t = create_task_template("causal_did")
        assert t.get_name() == "causal_did"
        spec = {
            "task_type": "causal_did", "outcome": {"variable": "rank_base"},
            "group_variable": "low_ses", "post_variable": "cohort",
            "primary_method": "M8",
        }
        assert t.validate_research_spec(spec, {}, None) == []
        bad = dict(spec, primary_method="M1")
        assert any("M8" in w for w in t.validate_research_spec(bad, {}, None))

    def test_panel_adapter_registered(self) -> None:
        from src.dataset_adapter import create_dataset_adapter

        a = create_dataset_adapter("did_els_hsls_panel")
        assert a.get_raw_data_filename().endswith("panel.csv")
        assert a.get_protected_attributes() == ["low_ses", "female"]

    def test_panel_registry_loads(self) -> None:
        path = PROJECT_ROOT / "data_registry" / "datasets" / "did_els_hsls_panel.yaml"
        with open(path, encoding="utf-8") as f:
            reg = yaml.safe_load(f)
        assert reg["design_feasibility"]["multi_cohort_partner"]
        pitfall_ids = {p["id"] for p in reg["common_pitfalls"]}
        assert {"non_equated_tests", "single_policy_attribution"} <= pitfall_ids

    def test_prompt_variants_exist_and_resolve(self) -> None:
        from src.agents.base import load_prompt

        config = {"paths": {"agent_prompts": str(PROJECT_ROOT / "agent_prompts")}}
        for agent in ("problem_formulator", "analyst", "writer"):
            prompt = load_prompt(agent, config, task_type="causal_did")
            body = prompt["system_prompt"]
            assert "CAUSAL_DID" in body, agent
            assert "{{SKILLS}}" in body, agent

    def test_pre_critic_supports_causal_did(self) -> None:
        from src.pre_critic_checks import _SUPPORTED_TASK_TYPES

        assert "causal_did" in _SUPPORTED_TASK_TYPES


# ---------------------------------------------------------------------------
# Orchestrator-path skill rendering (never bare match())
# ---------------------------------------------------------------------------

class TestDIDSkillRendering:
    @pytest.fixture(scope="class")
    def registry(self):
        from src.skills import SkillRegistry

        return SkillRegistry(str(PROJECT_ROOT / "skills"))

    @pytest.mark.parametrize(
        "stage", ["ProblemFormulator", "Analyst", "Critic", "Writer"]
    )
    def test_did_skill_reaches_stage(self, registry, stage: str) -> None:
        from src.orchestrator import _resolve_skill_caps
        from src.skills.composer import format_skills_for_prompt

        skills = registry.match_and_compose(
            task_type="causal_did",
            dataset="did_els_hsls_panel",
            stage=stage,
            context="difference-in-differences cohort gap",
            top_k_per_layer=_resolve_skill_caps("causal_did"),
        )
        names = [s.name for s in skills]
        assert "causal-did-cross-cohort" in names
        rendered = format_skills_for_prompt(skills)
        assert "MANDATORY" in rendered
        # wrap-safe case-exact markers from the skill body
        assert "stability probe" in rendered
        assert "did_gap_in_gaps" in rendered


# ---------------------------------------------------------------------------
# Phase A regression: split-sanity pre-flight (F-A1-ELS-EMPTY-TEST-SPLIT)
# ---------------------------------------------------------------------------

class TestSplitSanityPreflight:
    def _orch_stub(self, tmp_path, task_type: str, n_test: int, test_rows: int):
        from src.orchestrator import Orchestrator

        o = object.__new__(Orchestrator)  # no __init__: probe only

        class Ctx:
            pass

        o.ctx = Ctx()
        o.ctx.research_spec = {"task_type": task_type}
        o.ctx.data_report = {"analytic_n": 10000, "n_test": n_test}
        o.ctx.output_dir = str(tmp_path)
        o._log = lambda *a, **k: None
        if test_rows >= 0:
            import pandas as pd

            pd.DataFrame({"x": range(test_rows)}).to_csv(
                tmp_path / "test_X.csv", index=False
            )
        elif test_rows == -1:
            # unparseable: zero-byte file (the ELS attempt-1 shape)
            (tmp_path / "test_X.csv").write_text("")
        return o

    def test_empty_test_set_is_violation(self, tmp_path) -> None:
        o = self._orch_stub(tmp_path, "prediction", n_test=0, test_rows=0)
        v = o._run_post_de_preflight()
        assert v is not None and "degenerate" in v

    def test_unreadable_test_file_is_violation(self, tmp_path) -> None:
        o = self._orch_stub(tmp_path, "prediction", n_test=2000, test_rows=-1)
        v = o._run_post_de_preflight()
        assert v is not None and "degenerate" in v

    def test_healthy_split_passes(self, tmp_path) -> None:
        o = self._orch_stub(tmp_path, "prediction", n_test=2000, test_rows=2000)
        assert o._run_post_de_preflight() is None

    def test_causal_did_exempt_from_split_check(self, tmp_path) -> None:
        o = self._orch_stub(tmp_path, "causal_did", n_test=0, test_rows=0)
        assert o._run_post_de_preflight() is None

    def test_psychometrics_exempt_from_split_check(self, tmp_path) -> None:
        # F-P1: item-matrix runs pin n_test=0 by design
        o = self._orch_stub(tmp_path, "psychometrics", n_test=0, test_rows=0)
        assert o._run_post_de_preflight() is None


# ---------------------------------------------------------------------------
# causal_did DE panel contract
# ---------------------------------------------------------------------------

class TestPanelValidator:
    def _de_stub(self, tmp_path):
        from src.agents.data_engineer import DataEngineer

        de = object.__new__(DataEngineer)

        class Ctx:
            pass

        class Adapter:
            def get_multilevel_warning(self):
                return None

        de.ctx = Ctx()
        de.ctx.output_dir = str(tmp_path)
        de.ctx.research_spec = {
            "task_type": "causal_did",
            "group_variable": "low_ses",
            "post_variable": "cohort",
            "outcome": {"variable": "rank_base"},
        }
        de.dataset_adapter = Adapter()
        return de

    def _write_panel(self, tmp_path, n_per_cell=300, break_cell=False):
        rows = []
        for c in (0, 1):
            for g in (0, 1):
                n = 10 if (break_cell and c and g) else n_per_cell
                rows.append(pd.DataFrame({
                    "cohort": c, "low_ses": g,
                    "rank_base": np.linspace(1, 99, n),
                    "rank_follow": np.linspace(1, 99, n),
                }))
        pd.concat(rows, ignore_index=True).to_csv(
            tmp_path / "panel_analytic.csv", index=False
        )

    def test_healthy_panel_passes(self, tmp_path) -> None:
        de = self._de_stub(tmp_path)
        self._write_panel(tmp_path)
        report = de._validate_outputs({"validation_passed": True})
        assert report["validation_passed"] is True
        assert report["analytic_n"] == 1200
        assert report["n_test"] == 0

    def test_missing_panel_fails(self, tmp_path) -> None:
        de = self._de_stub(tmp_path)
        report = de._validate_outputs({"validation_passed": True})
        assert report["validation_passed"] is False
        assert any("panel_analytic.csv" in w for w in report["warnings"])

    def test_degenerate_cell_fails(self, tmp_path) -> None:
        de = self._de_stub(tmp_path)
        self._write_panel(tmp_path, break_cell=True)
        report = de._validate_outputs({"validation_passed": True})
        assert report["validation_passed"] is False
        assert any("Degenerate 2x2" in w for w in report["warnings"])

    def test_de_variant_prompt_resolves(self) -> None:
        from src.agents.base import load_prompt

        config = {"paths": {"agent_prompts": str(PROJECT_ROOT / "agent_prompts")}}
        prompt = load_prompt("data_engineer", config, task_type="causal_did")
        body = prompt["system_prompt"]
        assert "panel_analytic.csv" in body
        assert "{{SKILLS}}" in body
        assert "train_test_split" in body  # forbidden-patterns marker (wrap-safe)


class TestStderrHint:
    def test_duplicate_label_signature_hinted(self) -> None:
        from src.agents.data_engineer import DataEngineer

        hint = DataEngineer._stderr_hint(
            "TypeError: arg must be a list, tuple, 1-d array, or Series"
        )
        assert "DUPLICATE" in hint and "dict.fromkeys" in hint

    def test_clean_stderr_no_hint(self) -> None:
        from src.agents.data_engineer import DataEngineer

        assert DataEngineer._stderr_hint("KeyError: 'X1FOO'") == ""


class TestCriticJsonRetry:
    def test_malformed_then_valid_json_recovers(self, tmp_path, monkeypatch) -> None:
        from src.agents.critic import Critic

        c = object.__new__(Critic)

        class Ctx:
            pass

        c.ctx = Ctx()
        c.ctx.output_dir = str(tmp_path)
        c.ctx.log = []
        c.agent_name = "Critic"
        responses = iter([
            '```json\n{"overall_verdict": "PASS", "broken": "unterminated\n```',
            '```json\n{"overall_verdict": "PASS", "overall_quality_score": 8,\n'
            ' "problem_formulation_review": {"score": 8, "issues": []},\n'
            ' "data_preparation_review": {"score": 8, "issues": []},\n'
            ' "analysis_review": {"score": 8, "issues": []},\n'
            ' "substantive_review": {"score": 8, "issues": []},\n'
            ' "revision_instructions": {}}\n```',
        ])
        c.call_llm = lambda msg, **kw: next(responses)
        c._build_user_message = lambda **kw: "msg"
        c.load_registry = lambda: {}
        c.load_task_template = lambda: {}
        c._load_checklist = lambda: ""
        # exercise just the parse-retry block via run()'s core sequence
        from src.agents.base import parse_llm_json
        import json as _json
        llm_response = c.call_llm("msg")
        json_text = c._extract_last_json_block(llm_response)
        try:
            report = parse_llm_json(json_text)
        except _json.JSONDecodeError:
            retry = c.call_llm("msg-retry")
            report = parse_llm_json(c._extract_last_json_block(retry))
        assert report["overall_verdict"] == "PASS"


# ---------------------------------------------------------------------------
# Stream-1 v2: M9/M10 certification gate + wiring
# ---------------------------------------------------------------------------

class TestDidV2Certification:
    def test_certification_gate_passes_at_certified_defaults(self) -> None:
        # Standing rule (Arc R/Q): tests run the gate itself, no downscaling.
        import warnings

        from scripts.quasi_experimental_gates import run_did_v2_gate

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = run_did_v2_gate()
        assert report["did_dr"]["passed"], report["did_dr"]
        assert report["did_het"]["passed"], report["did_het"]
        assert report["passed"]


class TestDidV2Wiring:
    def test_template_accepts_m9_primary(self) -> None:
        from src.task_template import create_task_template

        t = create_task_template("causal_did")
        spec = {
            "task_type": "causal_did", "outcome": {"variable": "rank_base"},
            "group_variable": "low_ses", "post_variable": "cohort",
            "primary_method": "M9",
        }
        assert t.validate_research_spec(spec, {}, None) == []

    def test_v2_fixture_loads_through_cli_path(self) -> None:
        from src.main import load_locked_research_spec

        spec = load_locked_research_spec("runs/fixtures/spec_did_ses_gap_v2.json")
        assert spec["primary_method"] == "M9"
        assert spec["adjustment_covariates"] == [
            "race5", "pared3", "expect_ba", "female"]

    def test_skill_v11_carries_m9_m10_rules(self) -> None:
        from src.orchestrator import _resolve_skill_caps
        from src.skills import SkillRegistry
        from src.skills.composer import format_skills_for_prompt

        registry = SkillRegistry(str(PROJECT_ROOT / "skills"))
        skills = registry.match_and_compose(
            task_type="causal_did", dataset="did_els_hsls_panel",
            stage="Analyst", context="composition adjusted heterogeneity",
            top_k_per_layer=_resolve_skill_caps("causal_did"),
        )
        rendered = format_skills_for_prompt(skills)
        assert "did_dr_gap_change" in rendered
        assert "did_ml_heterogeneity" in rendered
        assert "contrast" in rendered  # contrast-based inference rule
        assert "did_09" in rendered and "did_11" in rendered

    def test_harmonizer_v2_columns_in_registry(self) -> None:
        with open(PROJECT_ROOT / "data_registry" / "datasets"
                  / "did_els_hsls_panel.yaml", encoding="utf-8") as f:
            reg = yaml.safe_load(f)
        names = []
        for group in reg["variables"]["predictors"].values():
            names += [v["name"] for v in group]
        for c in ("race5", "pared3", "expect_ba", "ses_std"):
            assert c in names, c
