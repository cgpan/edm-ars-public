"""Metrics must never come from the model's response when the code did not run.

A1, observed on a real JEDM run. The Analyst's generated script hit the
executor timeout, so ``subprocess.TimeoutExpired`` returned empty stdout
and nothing wrote results.json. The pipeline fell back to parsing a JSON
block the model had appended after its code — a block it AUTHORED rather
than computed. The resulting results.json reported five trained models
(AUC 0.823 / 0.83 / 0.81 / 0.79 / 0.78), calibration, an ablation and
eight figures. Zero figures existed on disk, no model had been fitted,
and ``errors`` was empty. The run was recorded as normal and a paper was
written from it.

An earlier attempt at the same study failed honestly with
``all_models: {}``, which was obviously unusable. That contrast is the
whole point: a crash is safe, a plausible fabrication is not.

Two independent guards, because either alone leaves a hole:
  1. execution status gates the response fallback;
  2. every claimed figure is checked against disk regardless.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.agents.analyst import Analyst
from src.context import PipelineContext


def _config() -> dict:
    return {
        "models": {"analyst": "deepseek-v4-pro"},
        "llm_provider": "deepseek",
        "deepseek": {"models": {"analyst": "deepseek-v4-pro"}},
        "pipeline": {"random_state": 42},
        "paths": {"agent_prompts": "agent_prompts/", "data_registry": "data_registry/"},
        "sandbox": {"enabled": False},
    }


def _agent(tmp_path: Path) -> Analyst:
    ctx = PipelineContext(
        dataset_name="hsls09_public",
        raw_data_path=str(tmp_path / "raw.csv"),
        output_dir=str(tmp_path),
    )
    with patch("anthropic.Anthropic"):
        return Analyst(ctx, "analyst", _config())


#: The shape of the block the model appended after its code on the real run.
FABRICATED = {
    "best_model": "XGBoost",
    "best_metric_value": 0.83,
    "primary_metric": "AUC",
    "all_models": {
        "LogisticRegression": {"auc": 0.823},
        "RandomForest": {"auc": 0.83},
        "XGBoost": {"auc": 0.81},
        "ElasticNet": {"auc": 0.79},
        "MLP": {"auc": 0.78},
    },
    "calibration": {"brier": 0.11},
    "ablation": {"without_smote": {"auc": 0.80}},
    "figures_generated": [
        "roc_curves.png", "shap_summary.png", "shap_importance.png",
        "calibration_curve.png", "confusion_matrix.png",
        "pdp_a.png", "pdp_b.png", "pdp_c.png",
    ],
    "tables_generated": ["model_comparison.csv"],
    "errors": [],
    "warnings": [],
}


def _response_with_fabricated_block() -> str:
    return (
        "Here is the analysis.\n```python\nprint('work')\n```\n"
        f"Results:\n```json\n{json.dumps(FABRICATED)}\n```\n"
    )


def test_timeout_does_not_import_metrics_from_the_response(tmp_path: Path) -> None:
    """The exact defect: no results.json, execution failed, model claims five models."""
    agent = _agent(tmp_path)
    out = agent._read_results(
        _response_with_fabricated_block(),
        execution_ok=False,
        execution_detail="returncode=-1, stdout=empty (TimeoutExpired)",
    )
    assert out["all_models"] == {}, "metrics were imported from a failed run"
    assert out["best_model"] == ""
    assert out["figures_generated"] == []
    assert out["errors"], "a failed execution must be recorded, not silent"
    assert any("not taken from the model's response" in e.lower()
               or "were not taken" in e.lower()
               or "response text" in e.lower() for e in out["errors"])


def test_a_real_results_file_on_disk_is_always_preferred(tmp_path: Path) -> None:
    """The guard must not block the normal path."""
    real = dict(FABRICATED, figures_generated=[], errors=[], warnings=[])
    (tmp_path / "results.json").write_text(json.dumps(real), encoding="utf-8")
    agent = _agent(tmp_path)
    out = agent._read_results("irrelevant", execution_ok=False)
    assert out["best_model"] == "XGBoost"


def test_successful_execution_may_still_use_the_block_but_flags_it(tmp_path: Path) -> None:
    """Execution ran, artifact missing: usable, but not silently."""
    agent = _agent(tmp_path)
    out = agent._read_results(_response_with_fabricated_block(), execution_ok=True)
    assert out["all_models"], "a successful run should not be discarded"
    assert any("unverified" in w.lower() for w in out["warnings"])


def test_claimed_figures_that_do_not_exist_are_removed(tmp_path: Path) -> None:
    """0 of 8 figures existed on the real fabricated run."""
    agent = _agent(tmp_path)
    out = agent._verify_figures_on_disk(dict(FABRICATED))
    assert out["figures_generated"] == []
    assert out["errors"]
    assert "do not exist on disk" in out["errors"][0]


def test_figures_that_do_exist_are_kept(tmp_path: Path) -> None:
    (tmp_path / "roc_curves.png").write_bytes(b"\x89PNG\r\n")
    (tmp_path / "shap_summary.png").write_bytes(b"\x89PNG\r\n")
    agent = _agent(tmp_path)
    out = agent._verify_figures_on_disk(
        {"figures_generated": ["roc_curves.png", "shap_summary.png"]}
    )
    assert set(out["figures_generated"]) == {"roc_curves.png", "shap_summary.png"}
    assert not out.get("errors")


def test_partially_present_figures_keep_only_the_real_ones(tmp_path: Path) -> None:
    (tmp_path / "roc_curves.png").write_bytes(b"\x89PNG\r\n")
    agent = _agent(tmp_path)
    out = agent._verify_figures_on_disk(
        {"figures_generated": ["roc_curves.png", "ghost.png"]}
    )
    assert out["figures_generated"] == ["roc_curves.png"]
    assert "ghost.png" in out["errors"][0]


@pytest.mark.parametrize(
    ("returncode", "stdout", "expected_ok"),
    [
        (0, "trained 5 models\n", True),
        (0, "", False),      # succeeded but produced nothing: a timeout signature
        (0, "   \n", False),
        (1, "partial\n", False),
        (-1, "", False),     # TimeoutExpired
        (137, "", False),    # OOM kill
    ],
    ids=["clean", "empty-stdout", "whitespace", "nonzero-rc", "timeout", "oom"],
)
def test_execution_ok_is_computed_from_both_signals(
    returncode: int, stdout: str, expected_ok: bool
) -> None:
    """Return code alone is not enough: the observed timeout returned rc 0
    from one executor path with nothing on stdout."""
    ok = returncode == 0 and bool(stdout.strip())
    assert ok is expected_ok


# --- the stage must actually fail, not just report nothing --------------

def test_an_empty_battery_is_critical_not_merely_major() -> None:
    """Acceptance criterion 2: a failed analysis produces a FAILED stage.

    pcc_02 treated "no models" as the bottom of a continuum with "too few
    models", so it returned major -> REVISE. REVISE spends its cycles and
    then writes an UNVERIFIED paper about an empty results object, which
    is the exact artifact this guard exists to prevent. Zero is not a
    small number here; it means the analysis never ran.
    """
    from src.pre_critic_checks import PreCriticResult, _check_model_count

    ctx = SimpleNamespace(results_object={"all_models": {}})
    result = PreCriticResult(failures=[])
    _check_model_count(ctx, result)

    assert result.failures, "an empty battery must be reported"
    assert result.failures[0].severity == "critical"
    assert result.has_critical


def test_too_few_models_is_still_only_major() -> None:
    """The escalation must not swallow the ordinary under-count case."""
    from src.pre_critic_checks import PreCriticResult, _check_model_count

    ctx = SimpleNamespace(
        results_object={"all_models": {"LogisticRegression": {}, "RandomForest": {}}}
    )
    result = PreCriticResult(failures=[])
    _check_model_count(ctx, result)

    assert result.failures[0].severity == "major"
    assert not result.has_critical


def test_a_full_battery_still_passes() -> None:
    from src.pre_critic_checks import PreCriticResult, _check_model_count

    ctx = SimpleNamespace(
        results_object={
            "all_models": {
                "LogisticRegression": {}, "RandomForest": {}, "XGBoost": {},
                "ElasticNet": {}, "MLP": {}, "StackingEnsemble": {},
            }
        }
    )
    result = PreCriticResult(failures=[])
    _check_model_count(ctx, result)
    assert not result.failures


def test_a_battery_of_only_stacking_counts_as_empty() -> None:
    """StackingEnsemble is not an individual model; alone it is still zero."""
    from src.pre_critic_checks import PreCriticResult, _check_model_count

    ctx = SimpleNamespace(results_object={"all_models": {"StackingEnsemble": {}}})
    result = PreCriticResult(failures=[])
    _check_model_count(ctx, result)
    assert result.failures[0].severity == "critical"
