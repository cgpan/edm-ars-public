"""What edmars shows when the pipeline removed the outcome from the predictors.

On the owner's Mac (round 2) the data preparation code left the outcome
X4EVRATNDCLG among the predictors, the analysis fitted every model with
it (AUC 1.0), and the automatic checks stopped the study. fix/r2-leakage
checks right after data preparation: the Data Engineer gets one retry,
and a column still there after it is removed by name before the
analysis. The pipeline says so with an OUTCOME_REMOVED_FROM_PREDICTORS
warning whose message names files and columns ("Removed the outcome from
the predictors: after the DataEngineer's targeted retry, train_X.csv had
X4EVRATNDCLG; ..."). The live view and the result screen say it in
plain words, and the result screen says what removing a column cannot
undo.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.cli._run_support import (  # also installs stand-ins
    DATA_REPORT,
    event,
    log_lines,
    make_run,
    v2_status,
)

from edmars.endstates import classify, outcome_removed_concern
from edmars.results import render_summary_html, result_text
from edmars.runstate import fold, load_state, parse_log_line
from edmars.view import render_screen

LINE = ("The data preparation step left the outcome among the predictors; "
        "EDM-ARS removed it before the analysis")
PIPELINE_MESSAGE = (
    "Removed the outcome from the predictors: after the DataEngineer's targeted retry, "
    "train_X.csv had X4EVRATNDCLG; test_X.csv had X4EVRATNDCLG. The orchestrator dropped "
    "X4EVRATNDCLG from the predictor matrices and changed nothing else; the outcome stays in "
    "train_y.csv and test_y.csv. If the data preparation code also used the outcome while "
    "imputing or scaling other predictors, those columns may still carry it, which no check "
    "of column names can see."
)
CONCERN_START = ("The data preparation step left the outcome (X4EVRATNDCLG) among the "
                 "predictors; EDM-ARS removed it before the analysis.")
RECORD = {"after": "the DataEngineer's targeted retry",
          "removed": {"train_X.csv": ["X4EVRATNDCLG"], "test_X.csv": ["X4EVRATNDCLG"]}}


def _removed(seq: int, minute: int) -> dict[str, Any]:
    return event(seq, "warning", minute, stage="ENGINEERING",
                 code="OUTCOME_REMOVED_FROM_PREDICTORS", message=PIPELINE_MESSAGE)


def _through_engineering() -> list[dict[str, Any]]:
    return [
        event(1, "run.start", 0, task_type="prediction", dataset="hsls09_public", provider="deepseek"),
        event(2, "stage.start", 0, stage="ENGINEERING"),
        _removed(3, 4),
        event(4, "stage.end", 4, stage="ENGINEERING", outcome="ok"),
        event(5, "stage.start", 4, stage="ANALYZING"),
    ]


# ---------------------------------------------------------------------------
# The live view
# ---------------------------------------------------------------------------


def test_the_removal_is_said_in_plain_words() -> None:
    state = fold(_through_engineering())
    assert state.recent == [LINE]
    assert state.warnings == [LINE]
    assert state.notices == [LINE]
    assert state.outcome_removed is True
    joined = " ".join(state.recent)
    assert "train_X.csv" not in joined and "X4EVRATNDCLG" not in joined
    assert "orchestrator" not in joined.lower()


def test_other_warnings_do_not_count_as_a_removal() -> None:
    state = fold([event(1, "warning", 0, code="SOMETHING_ELSE", message="a note")])
    assert state.outcome_removed is False
    assert state.notices == []


@pytest.mark.parametrize("plain", [True, False])
def test_the_live_screen_shows_the_line_whole_at_eighty_columns(plain: bool) -> None:
    state = fold(_through_engineering())
    lines = [text for text, _style in render_screen(state, width=80, plain=plain)]
    at = next(i for i, text in enumerate(lines) if "The data preparation step left" in text)
    shown = " ".join(text.strip() for text in lines[at:at + 3])
    assert "EDM-ARS removed it before the analysis" in shown
    assert all(len(text) <= 80 for text in lines)


def test_the_log_line_of_a_run_without_events_is_the_same_line() -> None:
    line = log_lines((4, PIPELINE_MESSAGE)).strip()
    events = parse_log_line(line)
    assert [e["type"] for e in events] == ["warning"]
    assert events[0]["data"]["code"] == "OUTCOME_REMOVED_FROM_PREDICTORS"
    state = fold(events)
    assert state.recent == [LINE] and state.outcome_removed is True


# ---------------------------------------------------------------------------
# The result screen
# ---------------------------------------------------------------------------


def test_the_result_screen_says_what_removing_it_cannot_undo(run_home: Path) -> None:
    report = {**DATA_REPORT, "outcome_variable": "X4EVRATNDCLG", "post_de_outcome_check": RECORD}
    run = make_run(run_home, events=_through_engineering(), data_report=report,
                   status=v2_status(), log=None)
    outcome = classify(run)
    [concern] = [c for c in outcome.concerns if c.startswith("The data preparation step")]
    assert concern.startswith(CONCERN_START)
    assert "fill in other predictors' missing values" in concern
    text = result_text(outcome, load_state(run), run)
    assert "Please check" in text
    assert "EDM-ARS removed it before the analysis" in text
    html = render_summary_html(outcome, load_state(run), run)
    assert "EDM-ARS removed it before the analysis" in html


def test_a_later_clean_data_preparation_leaves_nothing_to_check(run_home: Path) -> None:
    # A DataEngineer revision regenerated the files without the outcome:
    # the paper uses those, so the earlier removal is history, not a caveat.
    later = {"after": "a DataEngineer revision", "removed": {}}
    report = {**DATA_REPORT, "outcome_variable": "X4EVRATNDCLG", "post_de_outcome_check": later}
    run = make_run(run_home, events=_through_engineering(), data_report=report,
                   status=v2_status(), log=None)
    assert outcome_removed_concern(run, load_state(run)) is None
    assert not [c for c in classify(run).concerns if "outcome" in c]


def test_without_the_record_the_warning_event_decides(run_home: Path) -> None:
    run = make_run(run_home, events=_through_engineering(), status=v2_status(), log=None,
                   extra={"research_spec.json": '{"outcome_variable": "X4EVRATNDCLG"}'})
    concern = outcome_removed_concern(run, load_state(run))
    assert concern is not None and concern.startswith(CONCERN_START)

    quiet = make_run(run_home, name="quiet", status=v2_status(), log=None, events=[
        event(1, "run.start", 0, task_type="prediction", dataset="hsls09_public")])
    assert outcome_removed_concern(quiet, load_state(quiet)) is None


def test_a_stopped_study_lists_it_too(run_home: Path) -> None:
    report = {**DATA_REPORT, "outcome_variable": "X4EVRATNDCLG", "post_de_outcome_check": RECORD}
    stop = {"code": "CRITIC_ABORT", "stage": "CRITIQUING",
            "message": "Critic verdict ABORT: AUC 0.99 is too good to be true."}
    run = make_run(run_home, data_report=report, pdf=False, log=None,
                   status=v2_status("ABORTED", released=False, reason_code="ABORTED", abort=stop),
                   events=_through_engineering() + [
                       event(6, "error", 9, stage="CRITIQUING", code="CRITIC_ABORT",
                             message=stop["message"]),
                       event(7, "run.end", 9, state="ABORTED", abort=stop)])
    outcome = classify(run)
    assert outcome.kind == "stopped"
    assert outcome.concerns and outcome.concerns[0].startswith(CONCERN_START)
    assert "EDM-ARS removed it before the analysis" in result_text(outcome, load_state(run), run)


# ---------------------------------------------------------------------------
# Against the real pipeline: the observed leak, retried, then removed
# ---------------------------------------------------------------------------


def _pipeline_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, leaks: list[bool]) -> Path:
    """Run the orchestrator with stub agents (no provider call, no TeX);
    the DataEngineer's n-th run leaves the outcome in when ``leaks[n]``."""
    from tests.test_orchestrator_terminal import _config, _fake_compile_ok, _orch
    from tests.test_outcome_guard import _stage

    monkeypatch.setattr("src.orchestrator.compile_latex", _fake_compile_ok)
    # The clients are built but never called; the CLI tests clear real keys.
    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline-test-placeholder")
    run = tmp_path / "run"
    run.mkdir()
    orch = _orch(run, _config(run))
    _stage(orch, leaks)
    orch.run()
    return run


def test_a_study_whose_retry_leaked_again_from_the_real_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, run_home: Path,
) -> None:
    run = _pipeline_run(tmp_path, monkeypatch, [True, True])
    state = load_state(run)
    assert state.final_state == "COMPLETED"
    assert LINE in state.recent and state.outcome_removed
    assert not [line for line in state.recent if "train_X.csv" in line]
    outcome = classify(run)
    assert [c for c in outcome.concerns if c.startswith(CONCERN_START)]
    text = result_text(outcome, state, run)
    assert "EDM-ARS removed it before the analysis" in text
    assert "EDM-ARS removed it before the analysis" in render_summary_html(outcome, state, run)


def test_a_study_whose_retry_was_clean_says_nothing_from_the_real_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, run_home: Path,
) -> None:
    run = _pipeline_run(tmp_path, monkeypatch, [True, False])
    state = load_state(run)
    assert state.final_state == "COMPLETED"
    assert LINE not in state.recent and not state.outcome_removed
    assert not [c for c in classify(run).concerns if "outcome" in c]
