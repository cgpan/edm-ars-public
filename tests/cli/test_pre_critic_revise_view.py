"""What the live view shows when the automatic checks send the work back.

fix/pcc-revise turned a critical pre-review finding a revision can fix
(pcc_07: the analysis did not run the comparison the question promises)
from a stopped study into a revision. The pipeline says so with a
PRE_CRITIC_REVISE warning whose message is the check's own text, and a
verdict event carrying the short-circuit report's placeholder score of 1.
Before this, the recent list showed

    Sent back to Analyst (revision 1 of 2): pcc_07: The research question
    says 'above and beyond', which commits the paper to an ...

then "Automatic pre-review check: REVISE", and the review step read
"Score 1/10 - asked for changes", a score no reviewer gave.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.cli._run_support import alive_pid, event, log_lines, make_run  # also installs stand-ins

from edmars.results import render_summary_html, result_text
from edmars.endstates import classify
from edmars.runstate import (
    describe_now,
    fold,
    load_state,
    parse_log_line,
    pre_critic_revise_words,
)
from edmars.view import render_screen

MAC_LINE = ("Automatic check: the analysis did not run the comparison the question promises; "
            "sending it back to the analysis step (revision 1 of 2)")
PCC_07_TEXT = ("The research question says 'above and beyond', which commits the paper to an "
               "incremental-validity / nested-model comparison, but no such analysis appears in "
               "results.json.")


def _revise(seq: int, minute: int, revision: int = 1, *, bare: bool = False) -> dict[str, Any]:
    """The pipeline's PRE_CRITIC_REVISE warning; ``bare`` without the
    structured fields, as the message alone."""
    data: dict[str, Any] = {
        "code": "PRE_CRITIC_REVISE",
        "message": f"Sent back to Analyst (revision {revision} of 2): pcc_07: {PCC_07_TEXT}",
    }
    if not bare:
        data.update(checks=["pcc_07"], targets=["Analyst"], revision=revision, max_revisions=2)
    return event(seq, "warning", minute, stage="CRITIQUING", **data)


def _to_revision() -> list[dict[str, Any]]:
    return [
        event(1, "run.start", 0, task_type="prediction", dataset="hsls09_public", provider="deepseek"),
        event(2, "stage.start", 0, stage="ANALYZING"),
        event(3, "stage.end", 5, stage="ANALYZING", outcome="ok"),
        event(4, "stage.start", 5, stage="CRITIQUING", cycle=0),
        _revise(5, 5),
        event(6, "verdict", 5, stage="CRITIQUING", cycle=1, plain="Automatic pre-review check: REVISE",
              critic_score=1, verdict="REVISE", unverified=False, source="pre_critic"),
        event(7, "stage.end", 5, stage="CRITIQUING", outcome="ok"),
        event(8, "stage.start", 5, stage="REVISING", cycle=1),
    ]


# ---------------------------------------------------------------------------
# The warning in plain words
# ---------------------------------------------------------------------------


def test_the_revision_is_said_in_plain_words() -> None:
    state = fold(_to_revision())
    assert MAC_LINE in state.recent
    assert state.warnings == [MAC_LINE]
    joined = " ".join(state.recent)
    assert "pcc_07" not in joined and "results.json" not in joined
    assert "Automatic pre-review check" not in joined  # the verdict adds nothing the line did not say


def test_an_event_without_the_fields_is_worded_from_its_message() -> None:
    ev = _revise(5, 5, bare=True)
    assert set(ev["data"]) == {"code", "message"}
    assert pre_critic_revise_words(ev["data"]) == MAC_LINE


@pytest.mark.parametrize(("data", "expected"), [
    ({"checks": ["pcc_02", "pcc_07"], "targets": ["Analyst"], "revision": 2, "max_revisions": 2},
     "Automatic checks: the analysis produced no trained model and the analysis did not run the "
     "comparison the question promises; sending it back to the analysis step (revision 2 of 2)"),
    ({"checks": ["pcc_99"], "targets": ["DataEngineer", "Analyst"], "revision": 1, "max_revisions": 3},
     "Automatic check: the results have a problem a revision can fix; sending it back to the data "
     "preparation step (revision 1 of 3)"),
])
def test_several_or_unknown_checks_and_an_earlier_step(data: dict[str, Any], expected: str) -> None:
    assert pre_critic_revise_words({"code": "PRE_CRITIC_REVISE", "message": "x", **data}) == expected


def test_a_message_that_cannot_be_read_is_shown_as_it_is() -> None:
    state = fold([event(1, "warning", 0, code="PRE_CRITIC_REVISE", message="something else")])
    assert state.recent == ["something else"]


def test_the_log_line_of_a_run_without_events_is_the_same_notice() -> None:
    line = log_lines((5, "Pre-Critic guard: revision cycle 1 of 2 re-runs Analyst for pcc_07")).strip()
    events = parse_log_line(line)
    assert [e["type"] for e in events] == ["warning"]
    assert events[0]["data"]["code"] == "PRE_CRITIC_REVISE"
    assert fold(events).recent == [MAC_LINE]


def test_a_stop_read_from_the_log_is_not_given_the_wrong_title() -> None:
    # The log's short-circuit ABORT line comes before the line naming the
    # code, so it cannot know whether the stop was PRE_CRITIC_UNRESOLVED.
    lines = log_lines(
        (9, "Pre-Critic guard found critical failures → short-circuit verdict: ABORT"),
        (9, "Pre-Critic guard stopped the run [PRE_CRITIC_UNRESOLVED]: pcc_07 was still failing "
            "when the revision cycles ran out (2 of 2 used): " + PCC_07_TEXT),
    )
    state = fold([ev for line in lines.splitlines() for ev in parse_log_line(line)])
    assert not [line for line in state.recent if line.startswith("Automatic checks stopped")]
    assert state.abort is not None and state.abort["code"] == "PRE_CRITIC_UNRESOLVED"


# ---------------------------------------------------------------------------
# The live screen
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("plain", [True, False])
def test_the_live_screen_shows_the_notice_whole_at_eighty_columns(plain: bool) -> None:
    state = fold(_to_revision())
    lines = [text for text, _style in render_screen(state, width=80, plain=plain)]
    at = next(i for i, text in enumerate(lines) if "Automatic check:" in text)
    shown = " ".join(text.strip() for text in lines[at:at + 3])
    assert "sending it back to the analysis step (revision 1 of 2)" in shown
    assert all(len(text) <= 80 for text in lines)


def test_the_review_step_does_not_show_the_placeholder_score(run_home: Path) -> None:
    # A review_report.json from an earlier reviewer round must not win
    # either: the checks decided the latest round.
    earlier = {"overall_verdict": "REVISE", "overall_quality_score": 6}
    run = make_run(run_home, events=_to_revision(), pid=alive_pid(), log=None, pdf=False,
                   review=earlier)
    state = load_state(run)
    review = next(s for s in state.stages if s.key == "CRITIQUING")
    assert review.detail == "automatic checks sent the work back"
    assert "critic_score" not in state.metrics


def test_the_now_line_says_the_checks_sent_it_back_not_the_reviewer() -> None:
    state = fold(_to_revision())
    assert describe_now(state) == ("The automatic checks sent the work back; the affected steps "
                                   "are being redone.")
    reviewer = fold([e for e in _to_revision() if e["type"] != "verdict"]
                    + [event(9, "verdict", 6, stage="CRITIQUING", verdict="REVISE", critic_score=6)])
    assert describe_now(reviewer).startswith("The reviewer asked for changes")


# ---------------------------------------------------------------------------
# Against the real pipeline: the observed study, revised and stopped
# ---------------------------------------------------------------------------


def _pipeline_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, resolved: bool) -> Path:
    """Run the orchestrator with stub agents (no provider call, no TeX)."""
    from tests.test_orchestrator_terminal import _config, _fake_compile_ok, _orch
    from tests.test_pre_critic_revise import _INCREMENTAL_OK, _INCREMENTAL_SKIPPED, _stage_study

    monkeypatch.setattr("src.orchestrator.compile_latex", _fake_compile_ok)
    # The clients are built but never called; the CLI tests clear real keys.
    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline-test-placeholder")
    run = tmp_path / "run"
    run.mkdir()
    orch = _orch(run, _config(run))
    _stage_study(orch, revised_incremental=_INCREMENTAL_OK if resolved else _INCREMENTAL_SKIPPED)
    orch.run()
    return run


def test_a_revised_study_from_the_real_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, run_home: Path,
) -> None:
    run = _pipeline_run(tmp_path, monkeypatch, resolved=True)
    state = load_state(run)
    assert state.final_state == "COMPLETED"
    assert MAC_LINE in state.recent
    assert not [line for line in state.recent if "pcc_" in line or "pre-review check" in line]
    review = next(s for s in state.stages if s.key == "CRITIQUING")
    assert review.detail.endswith("passed")  # the reviewer's own round, after the revision


def test_an_unresolved_study_from_the_real_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, run_home: Path,
) -> None:
    run = _pipeline_run(tmp_path, monkeypatch, resolved=False)
    state = load_state(run)
    assert state.final_state == "ABORTED"
    assert [line for line in state.recent if line.startswith("Automatic check:")] == [
        MAC_LINE, MAC_LINE.replace("revision 1 of 2", "revision 2 of 2")]
    assert "Automatic checks stopped the study after its revisions" in state.recent
    review = next(s for s in state.stages if s.key == "CRITIQUING")
    assert review.detail == "automatic checks stopped the study"

    outcome = classify(run)
    assert outcome.code == "PRE_CRITIC_UNRESOLVED" and outcome.resumable is False
    text = result_text(outcome, state, run)
    assert "1/10" not in text
    html = render_summary_html(outcome, state, run)
    assert "1/10" not in html
    assert "Internal methods review: not scored; the automatic checks stopped the study before the review" in html
    status = json.loads((run / "run_status.json").read_text(encoding="utf-8"))
    assert status["abort"]["code"] == "PRE_CRITIC_UNRESOLVED"
