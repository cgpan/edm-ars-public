"""results.json says which "best model" it names, and which scored best.

Observed on the owner's Mac (round 3, 2026-09-27): results.json named
XGBoost (AUC 0.801488285622901) as best_model while StackingEnsemble had
0.802020230289461 in the same all_models. By design -- SPEC section 4.3
takes every interpretability output from the best INDIVIDUAL model -- but
nothing in the file said so, and the paper wrote "XGBoost achieved the
best discrimination ... followed closely by the Stacking Ensemble (AUC =
0.802)". INV_SUPERLATIVE_CONTRADICTED flagged it.

The contract these tests pin:

* after ANALYZING (and an Analyst revision) results.json carries
  best_model_scope, best_overall_model and best_overall_metric_value,
  computed from all_models in the primary metric's direction;
* best_model itself is never rewritten; a best_model that is not first
  in its own scope gets a warning;
* the Writer's skills tell it to say "best individual model" and to
  report the ensemble when it scored higher.

No provider is called; nothing needs TeX.
"""
from __future__ import annotations

import copy
import json
import types
from pathlib import Path
from typing import Any

import pytest

from src.best_model import annotate, annotate_best_model
from src.context import PipelineState
from src.invariants import RunArtifacts, check_superlative_contradicted
from tests.test_orchestrator_terminal import _config, _orch, _wire

#: all_models AUCs of the round-3 study, to the last digit.
ROUND3_AUC = {
    "LogisticRegression": 0.7884831280985126,
    "RandomForest": 0.7985067167759475,
    "XGBoost": 0.801488285622901,
    "ElasticNet": 0.7812581960658883,
    "StackingEnsemble": 0.802020230289461,
}

ROUND3_RESULTS: dict = {
    "best_model": "XGBoost",
    "best_metric_value": 0.801488285622901,
    "primary_metric": "AUC",
    "all_models": {
        name: {"auc": auc, "auc_ci_lower": auc - 0.015, "auc_ci_upper": auc + 0.015}
        for name, auc in ROUND3_AUC.items()
    },
    "shap_model": "XGBoost",
    "top_features": [],
    "subgroup_performance": {},
    "figures_generated": [],
    "tables_generated": [],
    "errors": [],
    "warnings": ["High missingness: X1STUEDEXPCT has 27.4% missing values"],
}


def _rmse_results(**rmse: float) -> dict:
    return {
        "best_model": "XGBoost",
        "best_metric_value": rmse["XGBoost"],
        "primary_metric": "RMSE",
        "all_models": {n: {"rmse": v, "r2": 0.3} for n, v in rmse.items()},
        "warnings": [],
    }


# ---------------------------------------------------------------------------
# The annotation
# ---------------------------------------------------------------------------


def test_the_observed_study_names_its_scope_and_the_ensemble() -> None:
    out, warnings = annotate(copy.deepcopy(ROUND3_RESULTS))

    assert out["best_model"] == "XGBoost"
    assert out["best_metric_value"] == 0.801488285622901
    assert out["best_model_scope"] == "individual"
    assert out["best_overall_model"] == "StackingEnsemble"
    assert out["best_overall_metric_value"] == 0.802020230289461
    assert warnings == []


def test_lower_is_better_for_rmse() -> None:
    out, warnings = annotate(
        _rmse_results(LinearRegression=0.71, XGBoost=0.61, StackingEnsemble=0.60)
    )

    assert out["best_model_scope"] == "individual"
    assert out["best_overall_model"] == "StackingEnsemble"
    assert out["best_overall_metric_value"] == 0.60
    assert warnings == []


def test_when_the_best_individual_model_is_best_overall_both_name_it() -> None:
    results = copy.deepcopy(ROUND3_RESULTS)
    results["all_models"]["StackingEnsemble"]["auc"] = 0.79

    out, _ = annotate(results)

    assert out["best_overall_model"] == "XGBoost"
    assert out["best_overall_metric_value"] == 0.801488285622901


def test_a_tie_with_the_ensemble_goes_to_the_individual_model() -> None:
    out, _ = annotate(
        _rmse_results(LinearRegression=0.71, XGBoost=0.61, StackingEnsemble=0.61)
    )
    assert out["best_overall_model"] == "XGBoost"


def test_a_best_model_that_is_not_first_is_kept_and_warned_about() -> None:
    results = copy.deepcopy(ROUND3_RESULTS)
    results["best_model"] = "RandomForest"

    out, warnings = annotate(results)

    assert out["best_model"] == "RandomForest"
    assert out["best_model_scope"] == "individual"
    [warning] = warnings
    assert "results.best_model is RandomForest" in warning
    assert "XGBoost first among the individual models" in warning


def test_an_ensemble_named_best_model_is_scoped_overall() -> None:
    results = copy.deepcopy(ROUND3_RESULTS)
    results["best_model"] = "StackingEnsemble"

    out, warnings = annotate(results)

    assert out["best_model_scope"] == "overall"
    assert out["best_overall_model"] == "StackingEnsemble"
    assert warnings == []


@pytest.mark.parametrize(
    "results",
    [
        {"best_model": "XGBoost", "primary_metric": "AUC"},
        {"best_model": "XGBoost", "primary_metric": "AUC", "all_models": []},
        {"best_model": "XGBoost", "primary_metric": "AUC",
         "all_models": {"XGBoost": {"rmse": 0.6}}},
        {"best_model": "XGBoost", "all_models": {"XGBoost": {"auc": 0.8}}},
        # Causal and psychometric results carry no all_models at all.
        {"estimates": {"ate": 0.1}},
    ],
)
def test_results_without_per_model_values_are_left_alone(results: dict) -> None:
    out, warnings = annotate(results)
    assert out is results and warnings == []


def test_the_context_and_the_file_are_both_updated(tmp_path: Path) -> None:
    (tmp_path / "results.json").write_text(json.dumps(ROUND3_RESULTS), encoding="utf-8")
    ctx = types.SimpleNamespace(output_dir=str(tmp_path),
                                results_object=copy.deepcopy(ROUND3_RESULTS))

    annotate_best_model(ctx, log=lambda _m: None)

    on_disk = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert on_disk == ctx.results_object
    assert on_disk["best_overall_model"] == "StackingEnsemble"
    assert on_disk["warnings"] == ROUND3_RESULTS["warnings"]


def test_a_warning_lands_in_results_warnings_once(tmp_path: Path) -> None:
    results = {**copy.deepcopy(ROUND3_RESULTS), "best_model": "RandomForest"}
    ctx = types.SimpleNamespace(output_dir=str(tmp_path), results_object=results)
    logged: list[str] = []

    annotate_best_model(ctx, log=logged.append)
    annotate_best_model(ctx, log=logged.append)

    added = [w for w in ctx.results_object["warnings"]
             if w.startswith("results.best_model is RandomForest")]
    assert len(added) == 1
    assert logged and logged[0] == added[0]


def test_the_final_check_accepts_a_sentence_that_says_individual(tmp_path: Path) -> None:
    """The wording the Writer skill now asks for passes the check that
    flagged the round-3 sentence; the round-3 sentence still fails it."""
    (tmp_path / "results.json").write_text(json.dumps(ROUND3_RESULTS), encoding="utf-8")
    body = (
        "\\documentclass{acmart}\\begin{document}\n{}\n\\end{document}\n"
    )
    old = ("XGBoost achieved the best discrimination (AUC $= 0.801$) and "
           "outperformed logistic regression.")
    new = ("XGBoost achieved the highest AUC among the individual models "
           "(AUC $= 0.801$); the stacking ensemble's AUC (0.802) was marginally "
           "higher.")
    (tmp_path / "paper.tex").write_text(body.replace("{}", old), encoding="utf-8")
    assert [f.code for f in check_superlative_contradicted(RunArtifacts(str(tmp_path)))] == [
        "INV_SUPERLATIVE_CONTRADICTED"
    ]
    (tmp_path / "paper.tex").write_text(body.replace("{}", new), encoding="utf-8")
    assert check_superlative_contradicted(RunArtifacts(str(tmp_path))) == []


# ---------------------------------------------------------------------------
# In the pipeline
# ---------------------------------------------------------------------------


def _analyst_round3(orch: Any) -> None:
    out = Path(orch.ctx.output_dir)

    def analyst(**_kw: Any) -> dict:
        (out / "results.json").write_text(json.dumps(ROUND3_RESULTS), encoding="utf-8")
        return copy.deepcopy(ROUND3_RESULTS)

    orch.analyst.run = analyst


def test_analyzing_writes_the_scope_before_the_critic_and_writer_read_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The stub study's paper and results disagree on purpose; the release
    # decision is not what this test is about.
    monkeypatch.setattr("src.invariants.run_invariants", lambda _d: [])
    orch = _orch(tmp_path, _config(tmp_path))
    _wire(orch)
    _analyst_round3(orch)
    seen: dict[str, Any] = {}
    writer_run = orch.writer.run

    def writer(**kw: Any) -> Any:
        seen["writer"] = copy.deepcopy(orch.ctx.results_object)
        return writer_run(**kw)

    orch.writer.run = writer

    ctx = orch.run()

    assert ctx.current_state == PipelineState.COMPLETED, ctx.errors
    on_disk = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    for results in (on_disk, seen["writer"]):
        assert results["best_model"] == "XGBoost"
        assert results["best_model_scope"] == "individual"
        assert results["best_overall_model"] == "StackingEnsemble"


def test_an_analyst_revision_is_annotated_too(tmp_path: Path) -> None:
    orch = _orch(tmp_path, _config(tmp_path))
    _wire(orch)
    _analyst_round3(orch)
    orch.ctx.current_state = PipelineState.REVISING

    orch._run_agent("Analyst", revision_instructions="add the incremental test")

    assert orch.ctx.results_object["best_overall_model"] == "StackingEnsemble"
    on_disk = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert on_disk["best_model_scope"] == "individual"


# ---------------------------------------------------------------------------
# What the Writer and the Analyst are told
# ---------------------------------------------------------------------------


def test_the_writer_is_told_to_say_best_individual_model() -> None:
    from tests.test_v2_1_phase_3b24_writer_slim import (
        SKILLS_ROOT,
        _CONFIG,
        _render,
    )
    from src.agents.base import load_prompt
    from src.skills import SkillRegistry

    prompt = load_prompt("writer", _CONFIG, task_type="prediction")["system_prompt"]
    rendered = _render(SkillRegistry(SKILLS_ROOT), prompt, "prediction")

    assert "The best individual model with 95% CI" in rendered
    assert '`best_model_scope: "individual"`' in rendered
    assert "When `best_overall_model` is not `best_model`" in rendered
    assert "Best model with 95% CI" not in rendered


def test_the_analyst_skills_say_best_model_excludes_the_ensemble() -> None:
    root = Path(__file__).resolve().parents[1] / "skills" / "task-type"
    for name in ("prediction-evaluation-classification",
                 "prediction-evaluation-regression"):
        text = (root / name / "SKILL.md").read_text(encoding="utf-8")
        assert "**individual** model family" in text, name
        assert "StackingEnsemble is never `best_model`" in text, name
        assert "do not write those three" in text, name
