"""A research question that promises a test must be backed by that test.

A9, observed on real runs. The ordinal-attainment paper's entire premise
was whether modelling attainment as ORDERED beats treating it as nominal.
No ordinal model was ever fitted — no proportional-odds, no ordinal
forest. LSAR scored it 2/10 on Methodological Rigor and rejected it. Two
other papers showed the pattern more mildly: a stated incremental-validity
test never directly implemented, and no nested-model comparison.

This is the largest driver of low rigor scores that is genuinely the
system's fault, and it is invisible in the artifact: the paper compiles,
scores, and reads fluently. Only a reader who compares the question to
the analysis notices.

The check is deliberately narrow. It fires on phrases that COMMIT the
paper to a named analysis, never on vague ones like "explore" or
"examine" — a check that cries wolf gets ignored, and then it is not
running on the day the central question really does go untested.
"""

from __future__ import annotations

import types

import pytest

from src.pre_critic_checks import (
    PreCriticResult,
    _check_research_question_is_answered,
    run_pre_critic_checks,
)


def _run(question: str, results: object) -> list:
    ctx = types.SimpleNamespace(
        research_spec={"research_question": question},
        results_object=results,
        data_report={},
    )
    result = PreCriticResult()
    _check_research_question_is_answered(ctx, result)
    return result.failures


def test_the_ordinal_paper_that_was_rejected() -> None:
    """The observed case, verbatim in shape."""
    failures = _run(
        "Does modelling postsecondary attainment as an ordinal outcome "
        "improve prediction over treating it as nominal?",
        {"all_models": {"RandomForest": {"auc": 0.8}, "XGBoost": {"auc": 0.81}}},
    )
    assert len(failures) == 1
    assert failures[0].check_id == "pcc_07"
    assert failures[0].severity == "critical"
    assert "ordinal" in failures[0].message


def test_an_ordinal_model_satisfies_it() -> None:
    failures = _run(
        "Does modelling attainment as an ordinal outcome improve prediction?",
        {"all_models": {"OrdinalForest": {"auc": 0.82}}},
    )
    assert failures == []


@pytest.mark.parametrize(
    ("question", "results", "should_fire"),
    [
        ("Does X predict Y above and beyond SES?", {"all_models": {}}, True),
        ("Does X predict Y above and beyond SES?",
         {"incremental_validity": {"status": "ok", "delta_auc": 0.01}}, False),
        ("Does school climate mediate the SES effect?", {"estimates": {}}, True),
        ("Does school climate mediate the SES effect?",
         {"mediation": {"indirect_effect": 0.1}}, False),
        ("Does the effect vary by sex?", {"all_models": {}}, True),
        ("Does the effect vary by sex?",
         {"subgroup_performance": {"Male": {}}}, False),
        ("Is the model well calibrated?", {"all_models": {}}, True),
        ("Is the model well calibrated?", {"calibration": {"brier": 0.1}}, False),
    ],
    ids=[
        "incremental-missing", "incremental-present",
        "mediation-missing", "mediation-present",
        "moderation-missing", "moderation-present",
        "calibration-missing", "calibration-present",
    ],
)
def test_each_commitment_is_checked_both_ways(
    question: str, results: dict, should_fire: bool
) -> None:
    assert bool(_run(question, results)) is should_fire


@pytest.mark.parametrize(
    "question",
    [
        "What predicts postsecondary enrolment?",
        "We explore the relationship between motivation and achievement.",
        "This study examines dropout risk factors.",
        "Which features are most important for enrolment prediction?",
    ],
    ids=["what-predicts", "explore", "examines", "which-features"],
)
def test_vague_questions_do_not_fire(question: str) -> None:
    """Noise is how a check gets switched off."""
    assert _run(question, {"all_models": {"RF": {"auc": 0.8}}}) == []


def test_evidence_nested_anywhere_counts() -> None:
    """A model name inside all_models is as good as a top-level key."""
    assert _run(
        "Does an ordinal specification help?",
        {"all_models": {"OrderedLogit": {"auc": 0.7}}},
    ) == []


def test_missing_question_is_not_a_failure() -> None:
    assert _run("", {"all_models": {}}) == []


def test_non_dict_results_are_survivable() -> None:
    """The check must never be the thing that breaks a run."""
    ctx = types.SimpleNamespace(
        research_spec={"research_question": "an ordinal question"},
        results_object=None,
        data_report={},
    )
    result = PreCriticResult()
    _check_research_question_is_answered(ctx, result)  # must not raise


def test_the_check_runs_for_every_task_type(tmp_path) -> None:
    """The defect spanned task types, so the check is universal."""
    for task_type in ("prediction", "causal_soo", "causal_itr",
                      "causal_did", "psychometrics"):
        ctx = types.SimpleNamespace(
            research_spec={
                "research_question": "Does an ordinal model beat a nominal one?"
            },
            results_object={"all_models": {}},
            data_report={"validation_passed": True},
        )
        result = run_pre_critic_checks(ctx, str(tmp_path), task_type=task_type)
        assert any(f.check_id == "pcc_07" for f in result.failures), (
            f"pcc_07 did not fire for {task_type}"
        )


# ---------------------------------------------------------------------------
# A record that says the test did not run is not the test
# ---------------------------------------------------------------------------

_ABOVE_AND_BEYOND = (
    "Do ninth-grade non-cognitive factors predict college enrollment above "
    "and beyond academic achievement and socioeconomic status?"
)


@pytest.mark.parametrize(
    "status", ["skipped", "failed", "error", "not_run", "Skipped "],
)
def test_a_not_run_incremental_record_does_not_satisfy_it(status: str) -> None:
    """run_incremental_validity returns {"status": "skipped", ...} when its
    column lists match nothing. The key name used to count as evidence, so
    the question's central test could be skipped and the paper written."""
    failures = _run(
        _ABOVE_AND_BEYOND,
        {"incremental_validity": {"status": status,
                                  "reason": "no focal column present"}},
    )
    assert [f.check_id for f in failures] == ["pcc_07"]


def test_a_sentence_about_the_comparison_is_not_the_comparison() -> None:
    failures = _run(
        _ABOVE_AND_BEYOND,
        {
            "all_models": {"LogisticRegression": {"auc": 0.81}},
            "warnings": ["incremental_validity was not computed"],
            "errors": ["nested_model comparison raised ValueError"],
        },
    )
    assert [f.check_id for f in failures] == ["pcc_07"]


def test_a_computed_incremental_record_still_satisfies_it() -> None:
    assert _run(
        _ABOVE_AND_BEYOND,
        {
            "incremental_validity": {
                "status": "ok", "baseline_auc": 0.78, "full_auc": 0.81,
                "delta_auc": 0.03, "ci_lower": 0.01, "ci_upper": 0.05,
            },
            "warnings": [],
        },
    ) == []


@pytest.mark.parametrize(
    ("question", "results"),
    [
        ("Does the effect vary by sex?",
         {"moderation_analysis": {"status": "skipped",
                                  "reason": "moderator not in matrix"}}),
        ("Is the model well calibrated?",
         {"calibration": {"status": "skipped",
                          "reason": "not applicable to regression"}}),
    ],
    ids=["moderation-descoped", "calibration-regression"],
)
def test_descoped_records_the_contract_allows_still_count(
    question: str, results: dict
) -> None:
    """prediction-rigor-extensions lets moderation be recorded as skipped
    and descoped, and the Analyst prompt tells regression runs to record
    calibration as skipped. Only the incremental promise is strict."""
    assert _run(question, results) == []
