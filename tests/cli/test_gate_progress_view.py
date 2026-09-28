"""The review step in the live view: which review is running, and what a
failed revision means.

On the round-3 Mac study the automated peer review (the LSAR review
gate) ran for 42 minutes behind one unchanging line. Its first score
(5.7) was close to the 6.3 benchmark, so LSAR reviewed the paper three
times; the paper revision then failed ("Could not extract LaTeX from LLM
response; keeping original"), and the gate still paid for three more
reviews of the unchanged paper, whose lower median (5.1) became the
final score. fix/r3-gate ends the gate when the revision fails. The live
view now says which review is running ("review 2 of up to 6"), when the
paper is being revised, and that a failed revision means no further
paid reviews, or, with a pipeline from before that fix, that the
unchanged paper is reviewed again.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tests.cli._run_support import alive_pid, event, make_run, ts

from edmars.runstate import RunState, cost_line, describe_now, fold, gate_reviews_up_to, load_state
from edmars.view import PlainPrinter, screen_text

NO_MORE = ("The paper revision failed, so the paper is unchanged: no further paid reviews; "
           "the review ends with the scores it has")
AGAIN = ("The paper revision failed, so the paper is unchanged; this version of EDM-ARS "
         "reviews the unchanged paper again, at a cost")
NO_OP = ("Revision was a no-op (LLM failed or returned the original); paper.tex left unchanged. "
         "The next cycle still recompiles and re-reviews the unchanged manuscript.")


class Gate:
    """Builds the review gate's events as the pipeline writes them."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = [
            event(1, "run.start", 0, task_type="prediction", dataset="hsls09_public",
                  provider="deepseek"),
            event(2, "stage.start", 0, stage="WRITING"),
            event(3, "stage.end", 5, stage="WRITING", outcome="ok"),
            event(4, "stage.start", 5, stage="REVIEWING"),
        ]
        self.minute = 5

    def add(self, etype: str, **kw: Any) -> "Gate":
        self.minute += 1
        self.events.append(event(len(self.events) + 1, etype, min(self.minute, 59),
                                 stage="REVIEWING", **kw))
        return self

    def log(self, message: str) -> "Gate":
        return self.add("log", agent="ReviewGate", message=message)

    def state(self) -> RunState:
        state = fold(self.events)
        state.metrics.setdefault("gate_max_cycles", 2)
        state.metrics.setdefault("gate_median_samples", 3)
        from edmars import runstate

        runstate._stage_details(state)
        return state


def _detail(state: RunState) -> str:
    return next(s.detail for s in state.stages if s.key == "REVIEWING")


def _borderline_first_cycle() -> Gate:
    g = Gate()
    g.log("Starting review gate (max_cycles=2, threshold=6.3, floor=4.0)")
    g.log("--- Review gate cycle 1/2 ---")
    g.add("gate.cycle", plain="Review gate: cycle 1 of 2", cycle=1, max_cycles=2)
    g.log("Running LSAR review (cycle 1, venue=EDM)")
    return g


def test_the_review_step_says_which_review_is_running() -> None:
    g = _borderline_first_cycle()
    state = g.state()
    assert _detail(state) == "review 1 of up to 6"
    assert describe_now(state).endswith("This is review 1 of up to 6.")
    for _ in range(7):  # the first review, metered when it ends (fix/r3-gate)
        g.add("llm.end", agent="LSAR", ok=True, cost_usd=0.005, component="review")
    g.add("gate.cost", cost_usd=0.035, n_calls=7)
    g.log("LSAR review complete (cycle 1): overall_score=5.7")
    g.log("Borderline score 5.7 within ±1.5 of threshold 6.3 → median sampling (3 total reviews)")
    g.log("Running LSAR review (cycle 102, venue=EDM)")
    state = g.state()
    assert _detail(state) == "review 2 of up to 6"
    assert cost_line(state) == ("Cost so far: US$0.035 (7 AI calls, including US$0.035 for the "
                                "automated peer review; the review under way is not yet counted)")
    assert "LSAR review 1 finished, score 5.7" in state.recent
    assert "Close to the benchmark: 2 more reviews, and the middle score counts" in state.recent


def test_a_cycle_with_one_review_lowers_the_most_there_can_be() -> None:
    g = _borderline_first_cycle()
    g.log("LSAR review complete (cycle 1): overall_score=3.2")
    g.log("Review gate FAILED (cycle 1): score=3.20, rec=Reject, failing=['novelty']")
    assert gate_reviews_up_to(g.state()) == 4
    g.log("Calling LLM for section-scoped revision: 2 section(s) [Introduction, Discussion], "
          "9000 of 40000 chars sent")
    g.add("llm.start", agent="ReviewGate", plain="Waiting for deepseek-v4-pro to revise the paper",
          model="deepseek-v4-pro")
    state = g.state()
    assert _detail(state) == "revising the paper"
    assert describe_now(state).startswith("Revising the paper from the reviewer's comments")
    assert "waiting for the AI's reply" in describe_now(state)
    g.add("llm.end", agent="ReviewGate", ok=True, cost_usd=0.004, component="review")
    g.log("Section-scoped revision spliced 2 of 2 section(s): ['Introduction', 'Discussion']")
    g.log("Revised paper.tex written; recompiling LaTeX")
    g.log("--- Review gate cycle 2/2 ---")
    g.log("Running LSAR review (cycle 2, venue=EDM)")
    assert _detail(g.state()) == "review 2 of up to 4"


def _mac_until_revision() -> Gate:
    g = _borderline_first_cycle()
    for sample, score in ((1, "5.7"), (102, "5.4"), (103, "6.0")):
        if sample != 1:
            g.log(f"Running LSAR review (cycle {sample}, venue=EDM)")
        g.log(f"LSAR review complete (cycle {sample}): overall_score={score}")
        if sample == 1:
            g.log("Borderline score 5.7 within ±1.5 of threshold 6.3 → median sampling "
                  "(3 total reviews)")
    g.log("Median sampling: scores=[5.4, 5.7, 6.0] → gating on median 5.7")
    g.add("gate.review", cycle=1, plain="Review gate cycle 1: score 5.7, not passed", score=5.7,
          passed=False, threshold=6.3)
    g.log("Review gate FAILED (cycle 1): score=5.70, rec=Borderline, failing=['clarity']")
    g.log("Calling LLM for whole-document paper revision (LSAR feedback)")
    return g


def test_a_failed_revision_says_no_further_paid_reviews() -> None:
    g = _mac_until_revision()
    assert gate_reviews_up_to(g.state()) == 6
    g.log("Could not extract LaTeX from LLM response; keeping original")
    state = g.state()
    assert _detail(state) == "revision failed · no further reviews"
    assert describe_now(state) == NO_MORE + "."
    assert gate_reviews_up_to(state) == 3
    assert NO_MORE in state.notices and state.recent[-1] == NO_MORE
    text = screen_text(state, width=100)
    assert "revision failed - no further reviews" in text
    assert "no further paid reviews" in " ".join(text.split())


def test_an_older_pipeline_that_reviews_the_unchanged_paper_again_is_described_so() -> None:
    # Before fix/r3-gate the gate went on to a second cycle and paid for
    # reviews of the unchanged paper; it said so right after the failure.
    g = _mac_until_revision()
    g.log("Could not extract LaTeX from LLM response; keeping original")
    g.log(NO_OP)
    state = g.state()
    assert NO_MORE not in state.notices and NO_MORE not in state.recent
    assert state.notices[-1] == AGAIN and state.recent[-1] == AGAIN
    assert describe_now(state) == AGAIN + "."
    g.log("--- Review gate cycle 2/2 ---")
    g.log("Running LSAR review (cycle 2, venue=EDM)")
    state = g.state()
    assert _detail(state) == "review 4 of up to 6"
    assert describe_now(state).endswith("This is review 4 of up to 6.")


def test_the_plain_view_prints_the_failed_revision_once() -> None:
    g = _mac_until_revision()
    printer = PlainPrinter(width=100)
    printer.lines(g.state())
    g.log("Could not extract LaTeX from LLM response; keeping original")
    printed = " ".join(" ".join(printer.lines(g.state())).split())
    assert printed.count(NO_MORE) == 1
    g.log("Revision was a no-op (LLM failed or returned the original); paper.tex left unchanged.")
    assert not [line for line in printer.lines(g.state()) if "revision failed" in line.lower()]


def test_the_review_step_is_followed_from_pipeline_log_too(run_home: Path) -> None:
    # A run without events.jsonl: the same lines in pipeline.log.
    lines = [
        (0, "Orchestrator", "Starting WRITING stage"),
        (5, "Orchestrator", "WRITING stage complete → REVIEWING"),
        (5, "Orchestrator", "Starting REVIEWING stage (LSAR quality gate)"),
        (5, "ReviewGate", "--- Review gate cycle 1/2 ---"),
        (6, "ReviewGate", "Running LSAR review (cycle 1, venue=EDM)"),
        (12, "ReviewGate", "LSAR review complete (cycle 1): overall_score=5.7"),
        (12, "ReviewGate", "Borderline score 5.7 within ±1.5 of threshold 6.3 → median sampling "
                           "(3 total reviews)"),
        (12, "ReviewGate", "Running LSAR review (cycle 102, venue=EDM)"),
    ]
    log = "".join(f"{ts(m)} [{who}] {msg}\n" for m, who, msg in lines)
    usage = json.dumps({"agent": "Analyst", "model": "deepseek-v4-pro", "prompt_tokens": 1000,
                        "completion_tokens": 100, "cached_prompt_tokens": 0}) + "\n"
    run = make_run(run_home, pid=alive_pid(), log=log, pdf=False, review_gate_enabled=True,
                   extra={"token_usage.jsonl": usage * 15})
    state = load_state(run)
    assert _detail(state) == "review 2 of up to 6"
    assert "This is review 2 of up to 6." in describe_now(state)
    assert "reviews not yet counted" in screen_text(state, width=100)


#: fix/r3-gate's own words when it ends the gate (log line and the
#: GATE_REVISION_FAILED warning's plain text).
GATE_STOP = ("The review gate could not revise the paper after cycle 1 (no LaTeX could be taken "
             "from the reply (the reply was cut off at the token limit)). The gate ends with "
             "cycle 1's score; no further review of the unchanged paper was paid for.")


def test_the_fixed_gates_stop_is_said_once_in_plain_words() -> None:
    g = _mac_until_revision()
    g.log("Could not extract LaTeX from LLM response (the reply was cut off at the token limit); "
          "keeping original")
    g.log(GATE_STOP)
    g.add("warning", cycle=1, plain=GATE_STOP, code="GATE_REVISION_FAILED", message=GATE_STOP)
    g.add("stage.end", outcome="ok")
    state = g.state()
    assert state.recent.count(NO_MORE) == 1 and state.notices.count(NO_MORE) == 1
    assert GATE_STOP not in state.recent
    assert NO_MORE in state.warnings


def test_a_revision_that_changed_nothing_visible_ends_the_reviews_too() -> None:
    # fix/r3-gate ends the gate when the revision changed only whitespace
    # or LaTeX comments: its stop line is the only one about it.
    g = _mac_until_revision()
    g.log("Post-revision reconciliation: 12 cited, 0 back-filled, 0 invented keys stripped")
    g.log("The review gate could not revise the paper after cycle 1 (the revision changed only "
          "whitespace or LaTeX comments, which a reviewer cannot see). The gate ends with cycle "
          "1's score; no further review of the unchanged paper was paid for.")
    state = g.state()
    assert _detail(state) == "revision failed · no further reviews"
    assert state.recent[-1] == NO_MORE
    assert gate_reviews_up_to(state) == 3
