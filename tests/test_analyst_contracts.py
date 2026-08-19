"""Analyst output-contract tests (Arc P residuals G3 + G4).

Two defects, both offline-reproducible:

G4 (F-P5-PSY-SCHEMA-KEYS) — ``_REQUIRED_KEYS`` was a hardcoded PREDICTION
schema applied to every task type, so a psychometrics or causal run got a
phantom ``"results.json is missing required keys: [...]"`` error injected
into ``results.json.errors`` and handed to the Critic as a genuine
analysis failure.

G3 (F-P5-BATTERY-SCOPE-CREEP) — the locked ``research_spec`` said
``method_battery: ["P1","P7"]`` but in 1 of 2 observed runs the Analyst
produced the full P1-P7 battery (24 wasted minutes + a CFA timeout). The
Analyst now runs a deterministic post-analysis scope assertion.

Offline: no network, no live LLM. Run:
    pytest tests/test_analyst_contracts.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.analyst import (
    _MINIMUM_REQUIRED_KEYS,
    _REQUIRED_KEYS,
    Analyst,
)
from src.config import load_config
from src.context import PipelineContext
from src.task_template import (
    CausalSOOTemplate,
    PredictionTemplate,
    PsychometricsTemplate,
)

CONFIG_PATH = str(Path(__file__).parent.parent / "config.yaml")


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_agent(tmp_path: Path, task_template: Any) -> Analyst:
    """Construct an Analyst bound to a specific TaskTemplate, offline."""
    config = load_config(CONFIG_PATH)
    ctx = PipelineContext(
        dataset_name="hsls09_public",
        raw_data_path="data/raw/nonexistent.csv",
        output_dir=str(tmp_path),
        max_revision_cycles=2,
    )
    with patch("anthropic.Anthropic"):
        return Analyst(ctx, "analyst", config, task_template=task_template)


def _psychometrics_results() -> dict:
    """A realistic psychometrics results.json (matches analyst_psychometrics.yaml).

    Deliberately carries NONE of the prediction keys — that is the point.
    """
    return {
        "task_id": "psy_hsls_matheff",
        "scale_name": "math_self_efficacy",
        "measurement_results": {
            "P1_ctt": {"alpha": 0.86, "n_complete": 18240},
            "P7_cdm": {"selected": "G-DINA", "degenerate_single_attribute": False},
        },
        "headline": "The scale functions equivalently across sex groups.",
        "group_mean_comparison_permitted": True,
        "items_missingness": {"S1MTESTS": 19.0},
        "warnings": [],
        "validation_passed": True,
    }


def _causal_results() -> dict:
    return {
        "estimand": "ATT",
        "primary_estimate": {"point": 0.12, "ci_lower": 0.03, "ci_upper": 0.21},
        "sensitivity": {"dowhy_refuters": {"placebo": {"status": "ran"}}},
        "errors": [],
        "warnings": [],
    }


def _prediction_results() -> dict:
    return {
        "best_model": "XGBoost",
        "best_metric_value": 0.61,
        "primary_metric": "RMSE",
        "all_models": {"XGBoost": {"rmse": 0.61}},
        "top_features": [{"feature": "X1TXMTSC", "shap_mean_abs": 0.18,
                          "direction": "positive"}],
        "subgroup_performance": {},
        "figures_generated": ["shap_summary.png"],
        "tables_generated": ["model_comparison.csv"],
        "errors": [],
        "warnings": [],
    }


def _missing_keys_errors(results: dict) -> list[str]:
    return [e for e in results.get("errors", []) if "missing required keys" in e]


class _DeclaringTemplate(PsychometricsTemplate):
    """A template that declares its own results contract via the hook."""

    def get_required_results_keys(self) -> set[str]:
        return {"measurement_results", "headline", "errors", "warnings"}


class _AttributeTemplate(PsychometricsTemplate):
    """A template that declares its contract as a class attribute."""

    REQUIRED_RESULTS_KEYS = frozenset({"measurement_results", "errors", "warnings"})


class _BrokenHookTemplate(PsychometricsTemplate):
    """A template whose hook raises — must not break a healthy run."""

    def get_required_results_keys(self) -> set[str]:
        raise RuntimeError("template hook is broken")


class _NamelessTemplate(PsychometricsTemplate):
    """Template whose get_name() raises (defensive-path probe).

    Swapped in AFTER construction — BaseAgent legitimately needs a name to
    pick the prompt file.
    """

    def get_name(self) -> str:
        raise RuntimeError("no name")


# ---------------------------------------------------------------------------
# G4 — required-results-keys sourcing
# ---------------------------------------------------------------------------


class TestRequiredResultsKeys:
    def test_prediction_keeps_full_spec_contract(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path, PredictionTemplate())
        assert agent._required_results_keys() == _REQUIRED_KEYS

    def test_psychometrics_falls_back_to_minimum(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path, PsychometricsTemplate())
        assert agent._required_results_keys() == _MINIMUM_REQUIRED_KEYS

    def test_causal_soo_falls_back_to_minimum(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path, CausalSOOTemplate())
        assert agent._required_results_keys() == _MINIMUM_REQUIRED_KEYS

    def test_minimum_is_not_the_prediction_set(self) -> None:
        assert _MINIMUM_REQUIRED_KEYS == {"errors", "warnings"}
        assert _MINIMUM_REQUIRED_KEYS < _REQUIRED_KEYS

    def test_template_hook_wins(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path, _DeclaringTemplate())
        assert agent._required_results_keys() == {
            "measurement_results", "headline", "errors", "warnings",
        }

    def test_template_attribute_wins(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path, _AttributeTemplate())
        assert agent._required_results_keys() == {
            "measurement_results", "errors", "warnings",
        }

    def test_broken_hook_falls_back_without_raising(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path, _BrokenHookTemplate())
        assert agent._required_results_keys() == _MINIMUM_REQUIRED_KEYS

    def test_unnameable_template_falls_back_without_raising(
        self, tmp_path: Path
    ) -> None:
        agent = _make_agent(tmp_path, PsychometricsTemplate())
        agent.task_template = _NamelessTemplate()
        assert agent._required_results_keys() == _MINIMUM_REQUIRED_KEYS

    def test_template_without_the_hook_falls_back_by_name(
        self, tmp_path: Path
    ) -> None:
        """No template declares the hook today — prediction must still work."""
        agent = _make_agent(tmp_path, PredictionTemplate())
        assert not hasattr(agent.task_template, "get_required_results_keys")
        assert agent._required_results_keys() == _REQUIRED_KEYS


class TestNoPhantomSchemaErrors:
    """The exact defect: a phantom error handed to the Critic."""

    def test_psychometrics_results_get_no_missing_keys_error(
        self, tmp_path: Path
    ) -> None:
        agent = _make_agent(tmp_path, PsychometricsTemplate())
        validated = agent._validate_results(_psychometrics_results())
        assert _missing_keys_errors(validated) == [], (
            f"phantom schema error injected: {validated['errors']}"
        )
        assert validated["errors"] == []

    def test_causal_results_get_no_missing_keys_error(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path, CausalSOOTemplate())
        validated = agent._validate_results(_causal_results())
        assert _missing_keys_errors(validated) == []

    def test_psychometrics_empty_results_still_clean(self, tmp_path: Path) -> None:
        """Degenerate case: even an empty dict must not raise the phantom.

        errors/warnings are setdefault-ed before the check, so the minimum
        contract is always satisfied — by design, per the G4 brief.
        """
        agent = _make_agent(tmp_path, PsychometricsTemplate())
        validated = agent._validate_results({})
        assert _missing_keys_errors(validated) == []

    def test_prediction_missing_keys_still_reported(self, tmp_path: Path) -> None:
        """Prediction behaviour must NOT change."""
        agent = _make_agent(tmp_path, PredictionTemplate())
        validated = agent._validate_results({"best_model": "XGBoost"})
        errors = _missing_keys_errors(validated)
        assert len(errors) == 1
        for key in _REQUIRED_KEYS - {"best_model", "errors", "warnings"}:
            assert key in errors[0]

    def test_prediction_complete_results_clean(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path, PredictionTemplate())
        validated = agent._validate_results(_prediction_results())
        assert validated["errors"] == []
        assert validated["warnings"] == []

    def test_declared_contract_is_enforced(self, tmp_path: Path) -> None:
        """A template that declares keys gets them checked (forward-compat)."""
        agent = _make_agent(tmp_path, _DeclaringTemplate())
        results = _psychometrics_results()
        del results["headline"]
        validated = agent._validate_results(results)
        errors = _missing_keys_errors(validated)
        assert len(errors) == 1
        assert "headline" in errors[0]
        assert "measurement_results" not in errors[0]


class TestTopFeaturesPhantomWarning:
    """Same phantom class: 'top_features is missing' on a measurement run."""

    def test_psychometrics_absent_top_features_no_warning(
        self, tmp_path: Path
    ) -> None:
        agent = _make_agent(tmp_path, PsychometricsTemplate())
        validated = agent._validate_results(_psychometrics_results())
        assert not any("top_features" in w for w in validated["warnings"])
        assert "top_features" not in validated

    def test_prediction_absent_top_features_still_warns(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path, PredictionTemplate())
        results = _prediction_results()
        del results["top_features"]
        validated = agent._validate_results(results)
        assert any("top_features" in w for w in validated["warnings"])
        assert validated["top_features"] == []

    def test_wrong_type_top_features_warns_on_any_task_type(
        self, tmp_path: Path
    ) -> None:
        agent = _make_agent(tmp_path, PsychometricsTemplate())
        results = _psychometrics_results()
        results["top_features"] = "not a list"
        validated = agent._validate_results(results)
        assert any("top_features" in w for w in validated["warnings"])
        assert validated["top_features"] == []


# ---------------------------------------------------------------------------
# G3 — method-battery scope assertion
# ---------------------------------------------------------------------------


def _psy_spec(battery: list[str]) -> dict:
    return {
        "task_type": "psychometrics",
        "scale_name": "math_self_efficacy",
        "item_columns": ["S1MTESTS", "S1MTEXTBOOK", "S1MSKILLS", "S1MASSEXCL"],
        "method_battery": battery,
    }


def _blocks(*names: str) -> dict:
    return {name: {"ok": True} for name in names}


class TestMethodBatteryScope:
    def test_noop_when_spec_declares_no_battery(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path, PredictionTemplate())
        results = _prediction_results()
        checked = agent._check_method_battery_scope(results, {"task_type": "prediction"})
        assert checked["errors"] == []
        assert checked["warnings"] == []

    def test_noop_when_battery_is_empty_list(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path, PsychometricsTemplate())
        results = _psychometrics_results()
        checked = agent._check_method_battery_scope(results, _psy_spec([]))
        assert checked["errors"] == []
        assert checked["warnings"] == []

    def test_noop_when_spec_is_empty(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path, PsychometricsTemplate())
        checked = agent._check_method_battery_scope(_psychometrics_results(), {})
        assert checked["errors"] == []
        assert checked["warnings"] == []

    def test_exact_match_is_clean_and_logged(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path, PsychometricsTemplate())
        results = _psychometrics_results()
        checked = agent._check_method_battery_scope(results, _psy_spec(["P1", "P7"]))
        assert checked["errors"] == []
        assert checked["warnings"] == []
        assert any(
            "scope check passed" in entry.get("message", "")
            for entry in agent.ctx.log
        ), agent.ctx.log

    def test_extra_blocks_are_a_warning_not_an_error(self, tmp_path: Path) -> None:
        """The observed defect: locked ['P1','P7'], produced the full battery."""
        agent = _make_agent(tmp_path, PsychometricsTemplate())
        results = _psychometrics_results()
        results["measurement_results"] = _blocks(
            "P1_ctt", "P2_omega", "P3_cfa", "P4_grm", "P5_dif", "P6_invariance",
            "P7_cdm",
        )
        checked = agent._check_method_battery_scope(results, _psy_spec(["P1", "P7"]))
        assert checked["errors"] == []
        assert len(checked["warnings"]) == 1
        warning = checked["warnings"][0]
        assert "SCOPE CREEP" in warning
        for unrequested in ("P2_omega", "P3_cfa", "P4_grm", "P5_dif",
                            "P6_invariance"):
            assert unrequested in warning
        assert "P1_ctt" not in warning
        assert "P7_cdm" not in warning
        assert any(
            "SCOPE CREEP" in entry.get("message", "") for entry in agent.ctx.log
        )

    def test_missing_blocks_are_an_error(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path, PsychometricsTemplate())
        results = _psychometrics_results()
        results["measurement_results"] = _blocks("P1_ctt")
        checked = agent._check_method_battery_scope(
            results, _psy_spec(["P1", "P2", "P7"])
        )
        assert len(checked["errors"]) == 1
        error = checked["errors"][0]
        assert "SCOPE VIOLATION" in error
        assert "'P2'" in error and "'P7'" in error
        assert checked["warnings"] == []

    def test_missing_and_extra_report_independently(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path, PsychometricsTemplate())
        results = _psychometrics_results()
        results["measurement_results"] = _blocks("P1_ctt", "P3_cfa")
        checked = agent._check_method_battery_scope(results, _psy_spec(["P1", "P7"]))
        assert len(checked["errors"]) == 1
        assert "'P7'" in checked["errors"][0]
        assert len(checked["warnings"]) == 1
        assert "P3_cfa" in checked["warnings"][0]

    def test_absent_measurement_results_is_an_error(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path, PsychometricsTemplate())
        results = {"errors": [], "warnings": []}
        checked = agent._check_method_battery_scope(results, _psy_spec(["P1", "P7"]))
        assert len(checked["errors"]) == 1
        assert "no 'measurement_results'" in checked["errors"][0]

    def test_measurement_results_wrong_type_is_an_error(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path, PsychometricsTemplate())
        results = {"measurement_results": ["P1_ctt"], "errors": [], "warnings": []}
        checked = agent._check_method_battery_scope(results, _psy_spec(["P1"]))
        assert len(checked["errors"]) == 1

    def test_case_and_whitespace_insensitive(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path, PsychometricsTemplate())
        results = _psychometrics_results()
        results["measurement_results"] = _blocks("p1_ctt", " P7_cdm")
        checked = agent._check_method_battery_scope(
            results, _psy_spec([" p1 ", "P7"])
        )
        assert checked["errors"] == []
        assert checked["warnings"] == []

    def test_bare_method_id_block_key_matches(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path, PsychometricsTemplate())
        results = _psychometrics_results()
        results["measurement_results"] = _blocks("P1", "P7")
        checked = agent._check_method_battery_scope(results, _psy_spec(["P1", "P7"]))
        assert checked["errors"] == []
        assert checked["warnings"] == []

    def test_duplicate_battery_entries_are_deduped(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path, PsychometricsTemplate())
        results = _psychometrics_results()
        checked = agent._check_method_battery_scope(
            results, _psy_spec(["P1", "P1", "P7"])
        )
        assert checked["errors"] == []
        assert checked["warnings"] == []

    def test_two_blocks_for_one_method_is_not_scope_creep(
        self, tmp_path: Path
    ) -> None:
        """P5/P6 run once per grouping var; block naming may vary."""
        agent = _make_agent(tmp_path, PsychometricsTemplate())
        results = _psychometrics_results()
        results["measurement_results"] = _blocks("P1_ctt", "P1_ctt_alt", "P7_cdm")
        checked = agent._check_method_battery_scope(results, _psy_spec(["P1", "P7"]))
        assert checked["errors"] == []
        assert checked["warnings"] == []

    def test_non_method_block_key_is_ignored(self, tmp_path: Path) -> None:
        """Ancillary keys must not generate their own phantom complaint."""
        agent = _make_agent(tmp_path, PsychometricsTemplate())
        results = _psychometrics_results()
        results["measurement_results"] = _blocks("P1_ctt", "P7_cdm", "notes")
        checked = agent._check_method_battery_scope(results, _psy_spec(["P1", "P7"]))
        assert checked["errors"] == []
        assert checked["warnings"] == []

    def test_misnamed_block_surfaces_as_missing(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path, PsychometricsTemplate())
        results = _psychometrics_results()
        results["measurement_results"] = _blocks("P1_ctt", "cdm_for_P7")
        checked = agent._check_method_battery_scope(results, _psy_spec(["P1", "P7"]))
        assert len(checked["errors"]) == 1
        assert "'P7'" in checked["errors"][0]
        # the actual key list is in the message so the operator can act
        assert "cdm_for_P7" in checked["errors"][0]

    def test_scope_check_never_crashes_the_run(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path, PsychometricsTemplate())
        # battery entries of odd types must not raise
        checked = agent._check_method_battery_scope(
            _psychometrics_results(), {"method_battery": [None, 7, "P1"]}
        )
        assert isinstance(checked["errors"], list)


# ---------------------------------------------------------------------------
# End-to-end through Analyst.run() (offline: LLM + executor stubbed)
# ---------------------------------------------------------------------------


class TestRunWritesContractCheckedResults:
    def _stub(self, agent: Analyst) -> None:
        agent.call_llm = MagicMock(return_value="```python\n# no-op\n```")
        agent.execute_code = MagicMock(
            return_value={"returncode": 0, "stdout": "", "stderr": ""}
        )

    def test_psychometrics_run_produces_clean_results_json(
        self, tmp_path: Path
    ) -> None:
        agent = _make_agent(tmp_path, PsychometricsTemplate())
        self._stub(agent)
        (tmp_path / "results.json").write_text(
            json.dumps(_psychometrics_results()), encoding="utf-8"
        )

        results = agent.run(
            data_report={"validation_passed": True, "analytic_n": 18240},
            research_spec=_psy_spec(["P1", "P7"]),
        )

        assert results["errors"] == []
        assert results["warnings"] == []
        on_disk = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
        assert on_disk["errors"] == []

    def test_psychometrics_run_records_scope_creep(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path, PsychometricsTemplate())
        self._stub(agent)
        overrun = _psychometrics_results()
        overrun["measurement_results"] = _blocks(
            "P1_ctt", "P2_omega", "P3_cfa", "P7_cdm"
        )
        (tmp_path / "results.json").write_text(
            json.dumps(overrun), encoding="utf-8"
        )

        results = agent.run(
            data_report={"validation_passed": True},
            research_spec=_psy_spec(["P1", "P7"]),
        )

        assert results["errors"] == []
        assert any("SCOPE CREEP" in w for w in results["warnings"])
        on_disk = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
        assert any("SCOPE CREEP" in w for w in on_disk["warnings"])

    def test_prediction_run_unaffected(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path, PredictionTemplate())
        self._stub(agent)
        (tmp_path / "results.json").write_text(
            json.dumps(_prediction_results()), encoding="utf-8"
        )

        results = agent.run(
            data_report={"validation_passed": True},
            research_spec={"task_type": "prediction", "outcome_variable": "X3TGPAMAT"},
        )

        assert results["errors"] == []
        assert results["warnings"] == []


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
