"""The automated peer review's cost, on every screen that shows a cost.

The round-3 Mac study showed US$0.300 (15 AI calls) while the DeepSeek
balance fell by US$0.57: the review gate's six LSAR reviews (42 calls)
and its paper-revision call were in no count, and the live cost stood
still at US$0.30015 for the gate's 42 minutes. fix/r3-gate meters them:
each gate call is an ``llm.end`` event with ``component: "review"``, a
``gate.cost`` event carries the gate's running total after each review,
the rows go to token_usage.jsonl as component "review", and
run_cost.json gains ``by_component`` subtotals (``pipeline``, ``review``)
beside the whole run's figures. The live view, the result screen,
summary.html and ``edmars runs --json`` include that share, name it, and
say so when a gate ran and nothing counted its reviews.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.cli._run_support import (  # also installs stand-ins
    PREDICTION_RESULTS,
    alive_pid,
    event,
    invariants_file,
    log_lines,
    make_run,
    v2_status,
    write_json,
)

from edmars import runner
from edmars.endstates import classify
from edmars.results import render_summary_html, result_text
from edmars.runstate import (
    cost_line,
    cost_parts,
    fold,
    load_state,
    reviews_not_counted,
)
from edmars.view import screen_text

GATE = {"enabled": True, "ran": True, "skip_reason": None, "passed": False, "score": 5.1,
        "threshold": 6.3, "advisory": False, "venue": "EDM"}


def _pipeline(calls: int = 15, each: float = 0.02) -> list[dict[str, Any]]:
    """A run up to the review gate: ``calls`` answered pipeline calls."""
    evs = [event(1, "run.start", 0, task_type="prediction", dataset="hsls09_public",
                 provider="deepseek")]
    seq = 2
    for stage in ("FORMULATING", "ENGINEERING", "ANALYZING", "CRITIQUING", "WRITING"):
        evs.append(event(seq, "stage.start", 1, stage=stage))
        seq += 1
        evs.append(event(seq, "stage.end", 2, stage=stage, outcome="ok"))
        seq += 1
    for _ in range(calls):
        evs.append(event(seq, "llm.end", 2, agent="Analyst", ok=True, cost_usd=each))
        seq += 1
    evs.append(event(seq, "stage.start", 3, stage="REVIEWING"))
    return evs


def _seq(evs: list[dict[str, Any]]) -> int:
    return max(int(e["seq"]) for e in evs) + 1


def _review_calls(evs: list[dict[str, Any]], n: int, each: float, minute: int,
                  *, total_calls: int, total_cost: float) -> list[dict[str, Any]]:
    """One metered LSAR review as fix/r3-gate announces it."""
    seq = _seq(evs)
    out = [event(seq + i, "llm.end", minute, stage="REVIEWING", agent="LSAR", ok=True,
                 cost_usd=each, component="review") for i in range(n)]
    out.append(event(seq + n, "gate.cost", minute, stage="REVIEWING",
                     plain=f"Review 1 of cycle 1 used {n} AI calls; the review gate has cost "
                           f"US${total_cost:.2f} so far ({total_calls} AI calls)",
                     cost_usd=total_cost, n_calls=total_calls, unpriced_calls=0))
    return out


# ---------------------------------------------------------------------------
# The live figures
# ---------------------------------------------------------------------------


def test_while_the_gate_has_reported_nothing_the_line_says_reviews_are_not_yet_counted() -> None:
    state = fold(_pipeline())
    assert reviews_not_counted(state) == "running"
    assert cost_line(state) == "Cost so far: US$0.300 (15 AI calls; reviews not yet counted)"


def test_the_gates_calls_are_counted_and_named_as_they_are_reported() -> None:
    evs = _pipeline()
    evs += _review_calls(evs, 7, 0.01, 10, total_calls=7, total_cost=0.07)
    state = fold(evs)
    assert state.llm_calls == 22 and state.cost_usd == pytest.approx(0.37)
    assert state.review_calls == 7 and state.review_cost_usd == pytest.approx(0.07)
    assert cost_line(state) == ("Cost so far: US$0.370 (22 AI calls, including US$0.070 for "
                                "the automated peer review)")
    # The gate's running total is not news for the recent list: the cost
    # line carries the number, and the review's own line its end.
    assert not any("used 7 AI calls" in line for line in state.recent)


def test_a_running_total_alone_is_added_once() -> None:
    # A pipeline that sends only gate.cost totals (no llm.end per call):
    # the totals are added, and a repeated total adds nothing.
    evs = _pipeline()
    seq = _seq(evs)
    evs += [event(seq, "gate.cost", 10, cost_usd=0.07, n_calls=7),
            event(seq + 1, "gate.cost", 17, cost_usd=0.07, n_calls=7),
            event(seq + 2, "gate.cost", 24, cost_usd=0.15, n_calls=14)]
    state = fold(evs)
    assert state.llm_calls == 29 and state.cost_usd == pytest.approx(0.45)
    assert state.review_calls == 14 and state.review_cost_usd == pytest.approx(0.15)


def test_a_resumed_gate_counts_its_new_reviews_on_top_of_the_old_ones() -> None:
    # The gate starts again after a resume and its running total restarts
    # at zero; the reviews paid for before the stop stay counted.
    evs = _pipeline()
    evs += _review_calls(evs, 7, 0.01, 10, total_calls=7, total_cost=0.07)
    seq = _seq(evs)
    evs += [event(seq, "run.start", 20, resumed=True),
            event(seq + 1, "stage.start", 20, stage="REVIEWING"),
            event(seq + 2, "gate.cost", 27, cost_usd=0.06, n_calls=6)]
    state = fold(evs)
    assert state.review_calls == 13 and state.review_cost_usd == pytest.approx(0.13)
    assert state.llm_calls == 28 and state.cost_usd == pytest.approx(0.43)


def test_the_live_view_shows_the_review_share(run_home: Path) -> None:
    evs = _pipeline()
    run = make_run(run_home, pid=alive_pid(), log=None, events=evs, pdf=False,
                   review_gate_enabled=True)
    assert "reviews not yet counted" in screen_text(load_state(run), width=100)
    evs += _review_calls(evs, 7, 0.01, 10, total_calls=7, total_cost=0.07)
    (run / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in evs), encoding="utf-8")
    text = screen_text(load_state(run), width=100)
    assert "Cost so far: US$0.370 (22 AI calls, including US$0.070 for the automated peer" in text


def test_tail_mode_counts_the_review_rows_of_token_usage(run_home: Path) -> None:
    # A run without events.jsonl: the gate's rows in token_usage.jsonl
    # carry component "review".
    row = {"agent": "Analyst", "model": "deepseek-v4-pro", "prompt_tokens": 1000,
           "completion_tokens": 100, "cached_prompt_tokens": 0}
    rows = [row] * 3 + [dict(row, agent="LSAR", component="review", stage="REVIEWING")] * 2
    run = make_run(run_home, pid=alive_pid(), pdf=False, review_gate_enabled=True,
                   log=log_lines((0, "Starting WRITING stage"), (5, "WRITING stage complete"),
                                 (5, "Starting REVIEWING stage (LSAR quality gate)")),
                   extra={"token_usage.jsonl": "".join(json.dumps(r) + "\n" for r in rows)})
    state = load_state(run)
    each = (1000 * 0.28 + 100 * 0.42) / 1e6
    assert state.llm_calls == 5 and state.review_calls == 2
    assert state.review_cost_usd == pytest.approx(2 * each)
    assert cost_line(state).startswith("Cost so far: US$0.002 (5 AI calls, including US$0.001 for "
                                       "the automated peer review")


# ---------------------------------------------------------------------------
# run_cost.json, old and new
# ---------------------------------------------------------------------------

#: src/cost.py from fix/r3-gate: the whole run at the top, subtotals in
#: by_component.
NEW_COST = {
    "n_calls": 58, "cost_usd": 0.504, "cost_status": "measured",
    "pipeline_cost_usd": 0.3, "review_cost_usd": 0.204,
    "by_component": {
        "pipeline": {"n_calls": 15, "cost_usd": 0.3, "cost_status": "measured", "unpriced_calls": 0},
        "review": {"n_calls": 43, "cost_usd": 0.204, "cost_status": "measured", "unpriced_calls": 0},
    },
}
OLD_COST = {"n_calls": 15, "cost_usd": 0.30015, "cost_status": "measured"}


@pytest.mark.parametrize("cost_file, expected", [
    (NEW_COST, {"n_calls": 58, "cost_usd": 0.504,
                "review": {"n_calls": 43, "cost_usd": 0.204, "unpriced": False}}),
    (OLD_COST, {"n_calls": 15, "cost_usd": 0.30015, "review": None}),
    # "components", with top-level figures that are the pipeline's alone:
    # the subtotals are added up.
    ({"n_calls": 15, "cost_usd": 0.3,
      "components": {"pipeline": {"n_calls": 15, "cost_usd": 0.3},
                     "review": {"n_calls": 43, "cost_usd": 0.204}}},
     {"n_calls": 58, "cost_usd": 0.504,
      "review": {"n_calls": 43, "cost_usd": 0.204, "unpriced": False}}),
    # A review subtotal some of whose calls have no price.
    ({"n_calls": 20, "cost_usd": 0.3,
      "by_component": {"pipeline": {"n_calls": 15, "cost_usd": 0.3},
                       "review": {"n_calls": 5, "cost_usd": None, "cost_status": "unpriced",
                                  "unpriced_calls": 5}}},
     {"n_calls": 20, "cost_usd": 0.3,
      "review": {"n_calls": 5, "cost_usd": None, "unpriced": True}}),
    (None, None),
])
def test_cost_parts_reads_both_shapes(cost_file: Any, expected: Any) -> None:
    parts = cost_parts(cost_file)
    if expected is None:
        assert parts is None
        return
    assert parts is not None
    assert parts["n_calls"] == expected["n_calls"]
    assert parts["cost_usd"] == pytest.approx(expected["cost_usd"])
    if expected["review"] is None:
        assert parts["review"] is None
    else:
        assert parts["review"]["n_calls"] == expected["review"]["n_calls"]
        assert parts["review"]["unpriced"] is expected["review"]["unpriced"]
        if expected["review"]["cost_usd"] is None:
            assert parts["review"]["cost_usd"] is None
        else:
            assert parts["review"]["cost_usd"] == pytest.approx(expected["review"]["cost_usd"])


def _finished(root: Path, cost: dict[str, Any] | None, *, gate: bool = True,
              name: str = "2026-09-25_1300_study_abcd") -> Path:
    log = log_lines((0, "Starting FORMULATING stage"), (1, "FORMULATING stage complete"),
                    (17, "VERIFYING stage complete → COMPLETED (review gate did not pass)"))
    kw: dict[str, Any] = {}
    if gate:
        kw = dict(review_gate_enabled=True,
                  gate_summary={"ran": True, "cycles_used": 2, "final_score": 5.1, "passed": False,
                                "threshold_used": 6.3, "advisory_mode": False, "venue": "EDM"},
                  status=v2_status(reason_code="GATE_FAILED", gate=GATE))
    else:
        kw = dict(status=v2_status())
    extra = {"run_cost.json": json.dumps(cost)} if cost is not None else {}
    return make_run(root, name=name, log=log, results=PREDICTION_RESULTS,
                    review={"overall_quality_score": 9, "overall_verdict": "PASS"},
                    invariants=invariants_file([]), extra=extra, **kw)


def test_a_finished_study_names_the_reviews_share_on_every_screen(run_home: Path) -> None:
    run = _finished(run_home, NEW_COST)
    state = load_state(run)
    line = "Cost: US$0.504 (58 AI calls, including US$0.204 for the automated peer review)"
    assert cost_line(state) == line
    outcome = classify(run, state=state)
    assert line in " ".join(result_text(outcome, state, run, width=120).split())
    assert line in render_summary_html(outcome, state, run)
    assert "did not count the automated peer review" not in render_summary_html(outcome, state, run)


def test_an_older_pipelines_uncounted_reviews_make_the_cost_a_lower_bound(run_home: Path) -> None:
    # The round-3 Mac study as its folder is: run_cost.json without the
    # reviews, a gate that ran.
    run = _finished(run_home, OLD_COST)
    state = load_state(run)
    assert reviews_not_counted(state) == "ended"
    line = "Cost: at least US$0.300 (15 AI calls; the automated peer reviews are not counted)"
    assert cost_line(state) == line
    outcome = classify(run, state=state)
    flat = " ".join(result_text(outcome, state, run, width=120).split())
    assert line in flat
    assert "did not count the automated peer review's AI calls, so the real cost is higher" in flat
    html = render_summary_html(outcome, state, run)
    assert line in html and "did not count the automated peer review" in html


def test_without_a_review_gate_the_cost_line_is_as_before(run_home: Path) -> None:
    run = _finished(run_home, OLD_COST, gate=False)
    state = load_state(run)
    assert reviews_not_counted(state) is None
    assert cost_line(state) == "Cost: US$0.300 (15 AI calls)"


def test_runs_json_carries_the_cost_with_the_review(run_home: Path, tmp_path: Path) -> None:
    from edmars import settings as settings_mod

    studies = tmp_path / "studies"
    _finished(studies, NEW_COST, name="2026-09-25_1300_new_aaaa")
    _finished(studies, OLD_COST, name="2026-09-24_1300_old_bbbb")
    current = settings_mod.load()
    current["studies_dir"] = str(studies)
    listed = {item["name"]: item for item in runner.list_runs(current)}
    new, old = listed["2026-09-25_1300_new_aaaa"], listed["2026-09-24_1300_old_bbbb"]
    assert new["cost_usd"] == pytest.approx(0.504) and new["n_calls"] == 58
    assert new["review_cost_usd"] == pytest.approx(0.204) and new["review_n_calls"] == 43
    assert new["cost_complete"] is True
    assert new["cost_text"] == ("Cost: US$0.504 (58 AI calls, including US$0.204 for the "
                                "automated peer review)")
    assert old["cost_usd"] == pytest.approx(0.30015) and old["review_n_calls"] is None
    assert old["cost_complete"] is False
    assert old["cost_text"].endswith("the automated peer reviews are not counted)")
    json.dumps([new, old])  # the --json output


def test_a_stale_cost_file_from_before_a_resume_is_not_read(run_home: Path) -> None:
    # Unchanged rule: run_cost.json older than the latest start belongs to
    # an earlier attempt. Its review share is ignored with it.
    import os
    import time

    run = _finished(run_home, NEW_COST)
    info = json.loads((run / "runner.json").read_text(encoding="utf-8"))
    info["started_at"] = "2099-01-01T00:00:00Z"
    write_json(run / "runner.json", info)
    old = time.time() - 3600
    os.utime(run / "run_cost.json", (old, old))
    assert load_state(run).review_calls is None
