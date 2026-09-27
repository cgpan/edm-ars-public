"""What the pre-review check does with a critical finding.

Observed on a real study (macOS test, 2026-09-26): the ProblemFormulator
asked whether non-cognitive factors predict college enrolment "ABOVE AND
BEYOND academic achievement and socioeconomic status", the Analyst ran
no nested-model comparison, and pcc_07 caught it -- correctly. But every
critical pre-review finding became ABORT, so a study one Analyst revision
could have finished stopped as PRE_CRITIC_ABORT, not resumable, and the
result screen said "a revision cannot fix" it.

The contract these tests pin:

* a critical finding a revision can fix (pcc_07, pcc_02 with no models)
  goes back to its target agent through the REVISING cascade, with a
  concrete instruction;
* a finding no revision can fix (pcc_01 confirmed leakage, pcc_06 failed
  validation) still stops the run at once, even next to a revisable one;
* a revisable finding still failing when the cycles run out stops the
  run as PRE_CRITIC_UNRESOLVED with no paper written;
* the abort record carries every finding, so an interface can show why;
* checkpoints and --resume work across a pre-review revision.

Every agent is a stub; nothing calls a provider or needs TeX.
"""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Any

import pytest

from src.context import PipelineState
from src.errors import ProviderError, is_resumable
from src.pre_critic_checks import (
    CheckFailure,
    PreCriticResult,
    _check_model_count,
    run_pre_critic_checks,
)
from tests.test_end_to_end import _DATA_REPORT, _PASS_REVIEW, _RESULTS
from tests.test_orchestrator_terminal import (
    _config,
    _events,
    _fake_compile_ok,
    _orch,
    _status,
    _wire,
)

_QUESTION = (
    "Do ninth-grade non-cognitive factors (math identity, math "
    "self-efficacy, school belonging) and college-going expectations predict "
    "college enrollment by February 2016 ABOVE AND BEYOND academic "
    "achievement and socioeconomic status, and does the predictive validity "
    "of these factors vary across sex and SES subgroups?"
)

_SPEC: dict = {
    "research_question": _QUESTION,
    "outcome_variable": "X4EVRATNDCLG",
    "outcome_type": "binary",
    "predictor_set": [
        {"variable": v, "rationale": "r", "wave": "base_year"}
        for v in ("X1TXMTSCOR", "X1SES", "X1MTHID", "X1MTHEFF",
                  "X1SCHOOLBEL", "X1STUEDEXPCT")
    ],
    "target_population": "full sample",
    "subgroup_analyses": ["X1SEX", "X1RACE"],
    "expected_contribution": "c",
    "potential_limitations": [],
    "novelty_score_self_assessment": 4,
}

_BINARY_REPORT: dict = {
    **_DATA_REPORT,
    "outcome_variable": "X4EVRATNDCLG",
    "outcome_type": "binary",
}

_INCREMENTAL_OK = {
    "status": "ok", "baseline_auc": 0.78, "full_auc": 0.81,
    "delta_auc": 0.03, "ci_lower": 0.01, "ci_upper": 0.05,
    "significant": True, "focal_cols": ["X1MTHID"],
    "baseline_cols": ["X1TXMTSCOR", "X1SES"],
}

_INCREMENTAL_SKIPPED = {
    "status": "skipped",
    "reason": "no focal column present in the design matrix",
}

#: What an archived GPA run wrote when its helper call raised: a null
#: record beside a warning. Not the test, and not the Analyst's word that
#: the test cannot run either.
_NULL_WITH_WARNING = object()


@pytest.fixture(autouse=True)
def _compile_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.orchestrator.compile_latex", _fake_compile_ok)


def _stage_study(
    orch: Any,
    *,
    revised_incremental: Any = _INCREMENTAL_OK,
    first_incremental: Any = None,
    leak: bool = False,
    question: str = _QUESTION,
) -> tuple[dict[str, int], list[str | None]]:
    """Wire the stub agents for the observed study.

    The first analysis records ``first_incremental`` (None: no record). A
    revision records ``revised_incremental`` (None: adds nothing).
    ``_NULL_WITH_WARNING`` for either writes a null record and a failure
    warning. ``leak`` writes the outcome into train_X.csv.
    """
    calls = _wire(orch, review=_PASS_REVIEW)
    out = orch.ctx.output_dir
    spec = {**_SPEC, "research_question": question}
    instructions: list[str | None] = []

    def pf(**_kw: Any) -> dict:
        calls["pf"] += 1
        return {"research_spec": copy.deepcopy(spec), "literature_context": None}

    def de(**_kw: Any) -> dict:
        calls["de"] += 1
        Path(out, "data_report.json").write_text(
            json.dumps(_BINARY_REPORT), encoding="utf-8"
        )
        headers = ["X1TXMTSCOR", "X1SES", "X1MTHID"]
        if leak:
            headers.append("X4EVRATNDCLG")
        with open(Path(out, "train_X.csv"), "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerows([headers, ["1"] * len(headers)])
        return copy.deepcopy(_BINARY_REPORT)

    def analyst(revision_instructions: str | None = None, **_kw: Any) -> dict:
        calls["analyst"] += 1
        instructions.append(revision_instructions)
        results = copy.deepcopy(_RESULTS)
        record = revised_incremental if revision_instructions else first_incremental
        if record is _NULL_WITH_WARNING:
            results["incremental_validity"] = None
            results["warnings"] = [
                "run_incremental_validity failed: Unknown label type: continuous."
            ]
        elif record is not None:
            results["incremental_validity"] = copy.deepcopy(record)
        Path(out, "results.json").write_text(json.dumps(results), encoding="utf-8")
        return results

    orch.problem_formulator.run = pf
    orch.data_engineer.run = de
    orch.analyst.run = analyst
    return calls, instructions


def _cp(out: Path) -> dict:
    return json.loads((out / "checkpoint.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# A revisable finding goes back to its agent
# ---------------------------------------------------------------------------


def test_the_observed_study_is_revised_not_aborted(tmp_path: Path) -> None:
    orch = _orch(tmp_path, _config(tmp_path))
    calls, instructions = _stage_study(orch)

    ctx = orch.run()

    assert ctx.current_state == PipelineState.COMPLETED
    assert calls == {"pf": 1, "de": 1, "analyst": 2, "critic": 1, "writer": 1}
    assert ctx.revision_cycle == 1
    assert _status(tmp_path)["state"] == "COMPLETED"

    first, revision = instructions
    assert first is None
    assert "[pcc_07, REQUIRED]" in revision
    assert "analysis_helpers.run_incremental_validity(" in revision
    assert "results['incremental_validity']" in revision
    # The baseline the question names, and the variables to map it to.
    assert '"academic achievement and socioeconomic status"' in revision
    assert "X1TXMTSCOR, X1SES, X1MTHID" in revision

    events = _events(tmp_path)
    verdicts = [e["data"] for e in events if e["type"] == "verdict"]
    assert verdicts[0]["verdict"] == "REVISE"
    assert verdicts[0]["source"] == "pre_critic"
    warned = [e["data"] for e in events if e["type"] == "warning"
              and e["data"].get("code") == "PRE_CRITIC_REVISE"]
    assert len(warned) == 1
    assert "Analyst" in warned[0]["message"] and "pcc_07" in warned[0]["message"]
    assert not [e for e in events if e["type"] == "error"]


def test_the_revision_goes_to_the_analyst_only(tmp_path: Path) -> None:
    """The stub results miss X1RACE, a DataEngineer-targeted major
    (pcc_05). It must not widen a cascade the critical finding starts at
    the Analyst; the Critic sees it next cycle. Analyst-targeted majors
    ride along with the required fix."""
    orch = _orch(tmp_path, _config(tmp_path))
    _stage_study(orch)
    orch.ctx.research_spec = copy.deepcopy(_SPEC)
    orch.ctx.data_report = copy.deepcopy(_BINARY_REPORT)
    orch.ctx.results_object = copy.deepcopy(_RESULTS)
    pre = run_pre_critic_checks(orch.ctx, str(tmp_path), task_type="prediction")
    assert any(f.check_id == "pcc_05" and f.target_agent == "DataEngineer"
               for f in pre.failures)

    report = orch._synthesize_pre_critic_report(pre)

    assert report["overall_verdict"] == "REVISE"
    ri = report["revision_instructions"]
    assert ri["ProblemFormulator"] is None
    assert ri["DataEngineer"] is None
    assert "[pcc_07, REQUIRED]" in ri["Analyst"]
    assert "[pcc_03, also fix]" in ri["Analyst"]
    assert {f["check_id"] for f in report["pre_critic_findings"]} >= {"pcc_07", "pcc_05"}


def test_an_empty_battery_is_revised(tmp_path: Path) -> None:
    """pcc_02 with no models: an execution failure, not a flaw in the
    question or the data, so the Analyst gets another run."""
    orch = _orch(tmp_path, _config(tmp_path))
    calls, instructions = _stage_study(orch, question="What predicts enrolment?")
    real = orch.analyst.run

    def analyst(revision_instructions: str | None = None, **kw: Any) -> dict:
        if revision_instructions is None:
            calls["analyst"] += 1
            instructions.append(None)
            return {"all_models": {}, "errors": ["Analysis code timed out"],
                    "warnings": []}
        return real(revision_instructions=revision_instructions, **kw)

    orch.analyst.run = analyst

    assert orch.run().current_state == PipelineState.COMPLETED
    assert calls["analyst"] == 2
    assert "[pcc_02, REQUIRED]" in instructions[1]
    assert "Analysis code timed out" in instructions[1]


def test_pcc_02_zero_and_pcc_07_are_revisable_pcc_01_and_06_are_not(
    tmp_path: Path,
) -> None:
    class Ctx:
        research_spec = {**_SPEC, "outcome_variable": "X4EVRATNDCLG"}
        results_object: dict = {"all_models": {}}
        data_report = {"validation_passed": False}

    with open(tmp_path / "train_X.csv", "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows([["X4EVRATNDCLG"], ["1"]])
    pre = run_pre_critic_checks(Ctx(), str(tmp_path), task_type="prediction")
    critical = {f.check_id: f.revisable for f in pre.failures
                if f.severity == "critical"}
    assert critical == {"pcc_01": False, "pcc_06": False,
                        "pcc_02": True, "pcc_07": True}
    for f in pre.revisable_failures:
        assert f.revision_instruction and f.revision_instruction != f.message

    too_few = PreCriticResult()
    _check_model_count(
        type("C", (), {"results_object": {"all_models": {"LR": {}}}})(), too_few
    )
    assert too_few.failures[0].severity == "major"


def test_a_check_that_says_nothing_still_stops_the_run() -> None:
    """A new critical check keeps the old behaviour until someone decides."""
    failure = CheckFailure("pcc_new", "critical", "m", "Analyst")
    assert failure.revisable is False
    assert PreCriticResult([failure]).fatal_failures == [failure]


# ---------------------------------------------------------------------------
# A finding no revision can fix still stops the run
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question", [_QUESTION, "What predicts enrolment?"],
    ids=["next-to-a-revisable-finding", "alone"],
)
def test_confirmed_leakage_still_aborts(tmp_path: Path, question: str) -> None:
    orch = _orch(tmp_path, _config(tmp_path))
    calls, _ = _stage_study(orch, leak=True, question=question)

    assert orch.run().current_state == PipelineState.ABORTED
    assert calls["analyst"] == 1 and calls["critic"] == 0 and calls["writer"] == 0

    abort = _status(tmp_path)["abort"]
    assert abort["code"] == "PRE_CRITIC_ABORT"
    assert abort["resumable"] is False
    assert abort["message"].startswith("pcc_01: Outcome variable 'X4EVRATNDCLG'")
    ids = {c["check_id"]: c["revisable"] for c in abort["checks"]}
    assert ids["pcc_01"] is False
    if question == _QUESTION:
        assert ids["pcc_07"] is True
    assert not (tmp_path / "paper.tex").exists()


# ---------------------------------------------------------------------------
# A revisable finding that outlasts the cycles stops the run, with no paper
# ---------------------------------------------------------------------------


def test_a_finding_still_failing_after_the_last_cycle_stops_the_run(
    tmp_path: Path,
) -> None:
    """Both revisions come back with a null record and a failure warning,
    which is not the test and not the Analyst's word that it cannot run,
    so each goes back until the cycles run out. No paper may be written
    around the missing result."""
    cfg = _config(tmp_path)
    orch = _orch(tmp_path, cfg)
    calls, instructions = _stage_study(orch, revised_incremental=_NULL_WITH_WARNING)

    ctx = orch.run()

    assert ctx.current_state == PipelineState.ABORTED
    assert calls == {"pf": 1, "de": 1, "analyst": 3, "critic": 0, "writer": 0}
    assert "2 more revision cycle" not in instructions[1]
    assert "1 more revision cycle(s) remain" in instructions[1]
    assert "this is the last revision cycle" in instructions[2]
    assert not (tmp_path / "paper.tex").exists()
    assert not (tmp_path / "paper.pdf").exists()

    status = _status(tmp_path)
    assert status["state"] == "ABORTED"
    assert status["released"] is False
    abort = status["abort"]
    assert abort["code"] == "PRE_CRITIC_UNRESOLVED"
    assert abort["stage"] == "CRITIQUING"
    assert abort["resumable"] is False
    assert abort["message"].startswith(
        "pcc_07 was still failing when the revision cycles ran out (2 of 2 used): "
        "The research question says 'above and beyond'"
    )
    assert [c["check_id"] for c in abort["checks"]] == ["pcc_07"]
    assert "incremental-validity" in abort["checks"][0]["message"]
    assert ctx.review_report["overall_verdict"] == "ABORT"
    assert ctx.review_report["stop_code"] == "PRE_CRITIC_UNRESOLVED"

    ends = [e["data"] for e in _events(tmp_path) if e["type"] == "run.end"]
    assert ends[-1]["exit_code"] == 3
    errors = [e["data"] for e in _events(tmp_path) if e["type"] == "error"]
    assert errors[-1]["code"] == "PRE_CRITIC_UNRESOLVED"

    # --resume cannot help: the cycle count comes back from the checkpoint.
    assert not is_resumable("PRE_CRITIC_UNRESOLVED")
    second = _orch(tmp_path, cfg)
    again, _ = _stage_study(second)
    assert second.run().current_state == PipelineState.ABORTED
    assert sum(again.values()) == 0


def test_with_no_revision_cycles_the_finding_stops_the_run_at_once(
    tmp_path: Path,
) -> None:
    orch = _orch(tmp_path, _config(tmp_path), max_revision_cycles=0)
    calls, _ = _stage_study(orch)

    assert orch.run().current_state == PipelineState.ABORTED
    assert calls["analyst"] == 1 and calls["writer"] == 0
    abort = _status(tmp_path)["abort"]
    assert abort["code"] == "PRE_CRITIC_UNRESOLVED"
    assert "(0 of 0 used)" in abort["message"]


def test_a_first_analysis_with_a_null_record_is_revised(tmp_path: Path) -> None:
    """The archived GPA shape: the helper call raised, the Analyst wrote
    incremental_validity: null and a warning. The key name used to pass
    for the test, so this went straight to the Critic and the Writer."""
    orch = _orch(tmp_path, _config(tmp_path))
    calls, instructions = _stage_study(orch, first_incremental=_NULL_WITH_WARNING)

    assert orch.run().current_state == PipelineState.COMPLETED
    assert calls["analyst"] == 2
    assert "[pcc_07, REQUIRED]" in instructions[1]


# ---------------------------------------------------------------------------
# A revision that comes back saying another will not help stops the run
# ---------------------------------------------------------------------------


def test_a_revision_that_says_the_test_cannot_run_stops_the_run(
    tmp_path: Path,
) -> None:
    """The instruction tells the Analyst to leave the helper's skipped
    record with its reason when the test truly cannot run, and says the
    study then stops. It used to be sent the identical instruction until
    every cycle was spent: four Analyst runs with three cycles."""
    orch = _orch(tmp_path, _config(tmp_path), max_revision_cycles=3)
    calls, instructions = _stage_study(orch, revised_incremental=_INCREMENTAL_SKIPPED)

    ctx = orch.run()

    assert ctx.current_state == PipelineState.ABORTED
    assert calls == {"pf": 1, "de": 1, "analyst": 2, "critic": 0, "writer": 0}
    assert "leave the helper's record, with its reason" in instructions[1]
    assert not (tmp_path / "paper.tex").exists()
    abort = _status(tmp_path)["abort"]
    assert abort["code"] == "PRE_CRITIC_UNRESOLVED"
    assert abort["resumable"] is False
    assert abort["message"].startswith(
        "pcc_07 was still failing after revision 1 of 3, and another revision "
        "would not change it: the Analyst recorded that the test cannot run: "
        "no focal column present in the design matrix."
    )
    assert [c["check_id"] for c in abort["checks"]] == ["pcc_07"]


def test_a_first_analysis_that_says_it_cannot_run_still_gets_a_revision(
    tmp_path: Path,
) -> None:
    """The helper's own skipped record in the FIRST analysis is usually a
    wrong column list, which the revision's instruction fixes. Only a
    revision that returns it again stops the run."""
    orch = _orch(tmp_path, _config(tmp_path))
    calls, _ = _stage_study(orch, first_incremental=_INCREMENTAL_SKIPPED)

    assert orch.run().current_state == PipelineState.COMPLETED
    assert calls["analyst"] == 2 and calls["writer"] == 1


def _empty_battery(calls: dict, instructions: list, errors_by_run: list[list[str]],
                   staged: Any) -> Any:
    """An Analyst whose first ``len(errors_by_run)`` runs train nothing."""

    def analyst(revision_instructions: str | None = None, **kw: Any) -> dict:
        run = calls["analyst"]
        if run < len(errors_by_run):
            calls["analyst"] += 1
            instructions.append(revision_instructions)
            return {"all_models": {}, "errors": list(errors_by_run[run]),
                    "warnings": []}
        return staged(revision_instructions=revision_instructions, **kw)

    return analyst


_TIMED_OUT = (
    "Analysis code did not execute successfully and wrote no results.json. "
    "returncode=-1, stdout=empty, stderr=Timeout after 600s"
)


def test_an_analysis_that_times_out_twice_stops_the_run(tmp_path: Path) -> None:
    """A try/except cannot make code finish inside the time limit. The
    instruction now says so, and a second timeout stops the run instead
    of spending the last cycle on a third."""
    orch = _orch(tmp_path, _config(tmp_path))
    calls, instructions = _stage_study(orch, question="What predicts enrolment?")
    orch.analyst.run = _empty_battery(
        calls, instructions, [[_TIMED_OUT], [_TIMED_OUT], [_TIMED_OUT]],
        orch.analyst.run,
    )

    assert orch.run().current_state == PipelineState.ABORTED
    assert calls["analyst"] == 2 and calls["writer"] == 0
    assert "The code ran out of time" in instructions[1]
    abort = _status(tmp_path)["abort"]
    assert abort["code"] == "PRE_CRITIC_UNRESOLVED"
    assert abort["message"].startswith(
        "pcc_02 was still failing after revision 1 of 2, and another revision "
        "would not change it: the analysis code ran out of time"
    )


def test_one_timeout_still_gets_its_revision(tmp_path: Path) -> None:
    orch = _orch(tmp_path, _config(tmp_path))
    calls, instructions = _stage_study(orch, question="What predicts enrolment?")
    orch.analyst.run = _empty_battery(
        calls, instructions, [[_TIMED_OUT]], orch.analyst.run
    )

    assert orch.run().current_state == PipelineState.COMPLETED
    assert calls["analyst"] == 2
    assert "The code ran out of time" in instructions[1]


def test_an_empty_battery_without_a_timeout_uses_every_cycle(
    tmp_path: Path,
) -> None:
    """A code failure is not a declaration: a fresh run may fix it."""
    orch = _orch(tmp_path, _config(tmp_path))
    calls, instructions = _stage_study(orch, question="What predicts enrolment?")
    failed = ["KeyError: 'X1SES'"]
    orch.analyst.run = _empty_battery(
        calls, instructions, [failed, failed, failed], orch.analyst.run
    )

    assert orch.run().current_state == PipelineState.ABORTED
    assert calls["analyst"] == 3
    assert "ran out of time" not in instructions[1]
    abort = _status(tmp_path)["abort"]
    assert abort["message"].startswith(
        "pcc_02 was still failing when the revision cycles ran out (2 of 2 used)"
    )


# ---------------------------------------------------------------------------
# Checkpoints and --resume across a pre-review revision
# ---------------------------------------------------------------------------


def test_ctrl_c_during_the_revision_resumes_the_same_revision(
    tmp_path: Path,
) -> None:
    cfg = _config(tmp_path)
    first = _orch(tmp_path, cfg)
    _, instructions = _stage_study(first)
    staged = first.analyst.run

    def analyst(revision_instructions: str | None = None, **kw: Any) -> dict:
        if revision_instructions:
            raise KeyboardInterrupt
        return staged(revision_instructions=revision_instructions, **kw)

    first.analyst.run = analyst
    with pytest.raises(KeyboardInterrupt):
        first.run()

    cp = _cp(tmp_path)
    assert cp["current_state"] == "REVISING"
    assert cp["revision_cycle"] == 1
    assert cp["review_report"]["_source"] == "pre_critic_short_circuit"
    saved = cp["review_report"]["revision_instructions"]["Analyst"]
    assert "run_incremental_validity" in saved

    second = _orch(tmp_path, cfg)
    calls, seen = _stage_study(second)
    ctx = second.run()

    assert ctx.current_state == PipelineState.COMPLETED
    assert calls == {"pf": 0, "de": 0, "analyst": 1, "critic": 1, "writer": 1}
    assert seen == [saved]
    assert ctx.revision_cycle == 1


def test_a_failed_revision_stops_resumably_instead_of_writing(
    tmp_path: Path,
) -> None:
    """A revision the check required that raises must not fall back to
    WRITING (UNVERIFIED): that writes the paper around the missing test."""
    cfg = _config(tmp_path)
    first = _orch(tmp_path, cfg)
    calls, _ = _stage_study(first)
    staged = first.analyst.run

    def analyst(revision_instructions: str | None = None, **kw: Any) -> dict:
        if revision_instructions:
            raise ProviderError("NETWORK", "connection reset")
        return staged(revision_instructions=revision_instructions, **kw)

    first.analyst.run = analyst

    assert first.run().current_state == PipelineState.ABORTED
    assert calls["writer"] == 0
    assert not (tmp_path / "paper.tex").exists()
    abort = _status(tmp_path)["abort"]
    assert abort["stage"] == "REVISING"
    assert abort["code"] == "NETWORK"
    assert abort["resumable"] is True

    second = _orch(tmp_path, cfg)
    again, _ = _stage_study(second)
    assert second.run().current_state == PipelineState.COMPLETED
    assert again == {"pf": 0, "de": 0, "analyst": 1, "critic": 1, "writer": 1}


def test_a_critic_revise_that_fails_still_writes_unverified(tmp_path: Path) -> None:
    """The fallback the guard above bypasses is unchanged for the Critic."""
    from tests.test_end_to_end import _REVISE_REVIEW

    orch = _orch(tmp_path, _config(tmp_path))
    calls = _wire(orch, review=_REVISE_REVIEW)

    def de(**kw: Any) -> dict:
        calls["de"] += 1
        if kw.get("revision_instructions"):
            raise ProviderError("NETWORK", "connection reset")
        Path(tmp_path, "data_report.json").write_text(
            json.dumps(_DATA_REPORT), encoding="utf-8"
        )
        return copy.deepcopy(_DATA_REPORT)

    orch.data_engineer.run = de
    ctx = orch.run()
    assert calls["writer"] == 1
    assert ctx.review_report.get("unverified") is True
