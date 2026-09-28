"""RunState: folding events.jsonl, and the tail adapter for older runs."""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tests.cli._run_support import (  # also installs stand-ins
    FULL_LOG,
    PREDICTION_RESULTS,
    alive_pid,
    dead_pid,
    event,
    log_lines,
    make_run,
    write_json,
)

from edmars import runstate
from edmars.runstate import (
    JsonlTail,
    RunState,
    StateReader,
    fold,
    load_state,
    parse_log_line,
    progress,
    step_position,
    usage_events,
)


def _stage(state: RunState, key: str) -> runstate.StageState:
    return next(s for s in state.stages if s.key == key)


# ---------------------------------------------------------------------------
# fold() over schema-v1 events
# ---------------------------------------------------------------------------

EVENTS = [
    event(1, "run.start", 0, task_type="prediction", dataset="hsls09_public", provider="deepseek"),
    event(2, "stage.start", 0, stage="FORMULATING", plain="Framing the question"),
    event(3, "lit.progress", 0, stage="FORMULATING", source="semantic_scholar", papers_found=30),
    event(4, "lit.progress", 1, stage="FORMULATING", source="arxiv", papers_found=8),
    event(5, "llm.start", 1, agent="ProblemFormulator", model="deepseek-v4-pro"),
    event(6, "llm.end", 1, agent="ProblemFormulator", cost_usd=0.004, prompt_tokens=10),
    event(7, "stage.end", 1, stage="FORMULATING", outcome="ok", duration_s=72),
    event(8, "stage.start", 1, stage="ENGINEERING"),
    event(9, "attempt.start", 2, stage="ENGINEERING", attempt=1, max_attempts=4, timeout_s=600),
    event(10, "attempt.end", 3, stage="ENGINEERING", attempt=1, returncode=0),
    event(11, "metric", 3, key="analytic_n", value=17335, label="students"),
    event(12, "stage.end", 4, stage="ENGINEERING", outcome="ok"),
    event(13, "stage.start", 4, stage="ANALYZING"),
    event(14, "brand.new.type", 5, plain="something newer pipelines say", whatever=1),
    event(15, "llm.start", 5, agent="Analyst"),
]


def test_fold_tracks_stages_cost_and_waiting() -> None:
    state = fold(EVENTS)
    assert state.task_type == "prediction"
    assert state.provider == "deepseek"
    assert _stage(state, "FORMULATING").status == "done"
    assert _stage(state, "ENGINEERING").status == "done"
    assert _stage(state, "ANALYZING").status == "running"
    assert state.current_stage == "ANALYZING"
    assert state.metrics["papers_found"] == 38
    assert state.metrics["reported"]["analytic_n"]["value"] == 17335
    assert state.llm_calls == 1
    assert state.cost_usd == pytest.approx(0.004)
    assert state.waiting_ai is True and state.waiting_agent == "Analyst"
    # plain text of an unknown event type still reaches "recent"
    assert "something newer pipelines say" in state.recent
    assert state.finished is False


def test_fold_is_pure_and_incremental() -> None:
    first = fold(EVENTS[:7])
    snapshot = json.dumps([s.status for s in first.stages])
    second = fold(EVENTS[7:], first)
    assert json.dumps([s.status for s in first.stages]) == snapshot  # base untouched
    assert fold(EVENTS).stages[2].status == second.stages[2].status == "running"
    # re-folding already seen events changes nothing (seq de-duplication)
    again = fold(EVENTS, second)
    assert again.llm_calls == second.llm_calls


def test_fold_run_end_and_failed_stage() -> None:
    state = fold(EVENTS + [
        event(16, "error", 6, stage="ANALYZING", code="NO_CREDIT", message="Insufficient Balance"),
        event(17, "run.end", 6, state="ABORTED", released=False, reason_code="ABORTED",
              exit_code=3, cost_usd=0.02),
    ])
    assert state.finished and state.final_state == "ABORTED"
    assert _stage(state, "ANALYZING").status == "failed"
    assert state.abort and state.abort["code"] == "NO_CREDIT"
    assert state.cost_usd == pytest.approx(0.02)
    assert state.exit_code == 3


def test_fold_critic_rounds_and_revising() -> None:
    evs = [
        event(1, "stage.start", 0, stage="CRITIQUING", cycle=0),
        event(2, "verdict", 1, stage="CRITIQUING", verdict="REVISE", critic_score=5, cycle=0),
        event(3, "stage.start", 1, stage="REVISING", cycle=1),
        event(4, "stage.end", 5, stage="REVISING", outcome="ok"),
        event(5, "stage.start", 5, stage="CRITIQUING", cycle=1),
    ]
    state = fold(evs)
    crit = _stage(state, "CRITIQUING")
    assert crit.status == "running" and crit.rounds == 2
    assert _stage(state, "REVISING").status == "done"
    assert runstate.stage_title(state, crit).endswith("(round 2 of up to 3)")


def test_fold_tolerates_garbage() -> None:
    state = fold([{}, {"type": None}, {"type": "stage.start"}, "not a dict",  # type: ignore[list-item]
                  {"type": "llm.end", "data": "oops"}, {"type": "metric", "data": {"key": None}}])
    assert state.llm_calls == 1
    assert state.finished is False


def test_jsonl_tail_holds_back_partial_line(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps({"type": "a", "seq": 1}) + "\n" + '{"type": "b", "se', encoding="utf-8")
    tail = JsonlTail(path)
    assert [e["type"] for e in tail.read_new()] == ["a"]
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('q": 2}\n{"type": "c"}\n')
    assert [e["type"] for e in tail.read_new()] == ["b", "c"]
    assert tail.read_new() == []
    # missing file: nothing, no error
    assert JsonlTail(tmp_path / "absent.jsonl").read_new() == []


def test_one_shot_read_accepts_complete_last_line_without_newline(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text('{"type": "a"}\n{"type": "b"}', encoding="utf-8")
    assert [e["type"] for e in runstate.read_events(path)] == ["a", "b"]
    path.write_text('{"type": "a"}\n{"type": "b', encoding="utf-8")
    assert [e["type"] for e in runstate.read_events(path)] == ["a"]


# ---------------------------------------------------------------------------
# Tail adapter (pipeline.log / token_usage.jsonl / artifacts)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "line, etypes",
    [
        ("Starting ENGINEERING stage", ["stage.start"]),
        ("Starting CRITIQUING stage (cycle 1)", ["stage.start"]),
        ("ENGINEERING stage complete", ["stage.end"]),
        ("Critic verdict: PASS → proceeding to WRITING", ["verdict"]),
        ("Critic verdict: PASS (UNVERIFIED) → proceeding to WRITING", ["verdict"]),
        ("Critic verdict: REVISE but max cycles exhausted → WRITING (UNVERIFIED)", ["verdict"]),
        ("Critic verdict: ABORT → pipeline aborted", ["verdict", "run.end"]),
        ("LaTeX compilation succeeded → paper.pdf written", ["compile.end"]),
        ("LaTeX compilation had errors — check pipeline.log for details", ["compile.end"]),
        ("VERIFYING stage complete → COMPLETED (clean)", ["stage.end", "verify.end", "run.end"]),
        ("VERIFYING stage complete -> COMPLETED", ["stage.end", "verify.end", "run.end"]),
        ("VERIFYING: release BLOCKED → INCOMPLETE (1 critical invariant finding(s))",
         ["stage.end", "verify.end", "run.end"]),
        ("Run cost: $0.0286 over 9 LLM calls (1 in / 2 out; 0 cached) -> run_cost.json", ["cost.total"]),
        ("ABORTED: ENGINEERING failed: boom", ["error", "run.end"]),
        ("Resumed from checkpoint (state=PipelineState.ANALYZING)", ["run.start"]),
        ("Code executor: SubprocessExecutor", ["heartbeat"]),
    ],
)
def test_parse_log_line(line: str, etypes: list[str]) -> None:
    events = parse_log_line(f"2026-09-25T13:28:12.540058 [Orchestrator] {line}")
    assert [e["type"] for e in events] == etypes


def test_parse_log_line_ignores_non_log_lines() -> None:
    assert parse_log_line("  continuation of a multi-line stderr") == []
    assert parse_log_line("") == []


def test_compile_step_reports_missing_tool() -> None:
    [ev] = parse_log_line(
        "2026-09-25T13:28:13.464682 [Orchestrator] LaTeX compile step failed: pdflatex "
        "-interaction=nonstopmode paper.tex (rc=-1): 'pdflatex' not found — is it installed and on PATH?"
    )
    assert ev["type"] == "compile.step" and ev["data"]["missing_tool"] == "pdflatex"


def _lit(seq: int, source: str, index: int, found: int, status: str = "ok", **extra: object) -> dict:
    return event(seq, "lit.progress", 0, stage="FORMULATING", source=source, query_index=index,
                 n_queries=3, papers_found=found, status=status, **extra)


def _formulating_detail(events: list[dict]) -> str:
    state = fold([event(1, "run.start", 0), event(2, "stage.start", 0, stage="FORMULATING"), *events])
    runstate._stage_details(state)
    return _stage(state, "FORMULATING").detail


def test_papers_found_add_up_over_every_query_of_every_source() -> None:
    # The handler kept only each source's LAST query: three arXiv queries
    # returning 10, 9 and 8 showed as 8.
    events = [_lit(3, "semantic_scholar", 1, 20), _lit(4, "semantic_scholar", 2, 15),
              _lit(5, "semantic_scholar", 3, 0, "rate_limited"),
              _lit(6, "arxiv", 1, 10), _lit(7, "arxiv", 2, 9), _lit(8, "arxiv", 3, 8)]
    state = fold([event(1, "run.start", 0), *events])
    assert state.metrics["lit_sources"] == {"semantic_scholar": 35, "arxiv": 27}
    assert state.metrics["papers_found"] == 62
    assert state.metrics["lit_status"] == {"semantic_scholar": "ok", "arxiv": "ok"}
    assert _formulating_detail(events) == "62 papers found"


def test_a_search_that_runs_again_replaces_its_counts() -> None:
    # A resume or a revision re-runs the question step's search: the same
    # query indexes come again and must not be added a second time.
    first = [_lit(3, "arxiv", 1, 10), _lit(4, "arxiv", 2, 9)]
    again = [_lit(10, "arxiv", 1, 7), _lit(11, "arxiv", 2, 6)]
    state = fold([event(1, "run.start", 0), *first, *again])
    assert state.metrics["papers_found"] == 13


def test_an_arxiv_refusal_is_said_as_a_refusal() -> None:
    # The Mac test: arXiv's front end answered HTTP 406, the other two
    # queries were not sent, and all three Semantic Scholar searches were
    # rate-limited.
    events = [_lit(3, "semantic_scholar", 1, 0, "rate_limited"),
              _lit(4, "semantic_scholar", 2, 0, "rate_limited"),
              _lit(5, "semantic_scholar", 3, 0, "rate_limited"),
              _lit(6, "arxiv", 1, 0, "refused", http_status=406),
              _lit(7, "arxiv", 2, 0, "skipped"), _lit(8, "arxiv", 3, 0, "skipped")]
    state = fold([event(1, "run.start", 0), *events])
    assert state.metrics["lit_status"] == {"semantic_scholar": "rate_limited", "arxiv": "refused",
                                           "arxiv_http_status": 406}
    detail = _formulating_detail(events)
    assert detail.startswith("0 papers found")
    assert "arXiv refused our requests (HTTP 406)" in detail
    assert "Semantic Scholar turned our searches away" in detail
    assert "failed" not in detail


def test_literature_notes_follow_the_pipelines_retrieval_status() -> None:
    notes = runstate.literature_notes
    assert notes({"semantic_scholar": "ok", "arxiv": "refused", "arxiv_http_status": 406}) == [
        "arXiv refused our requests (HTTP 406)"]
    assert notes({"arxiv": "refused"}) == ["arXiv refused our requests"]
    assert notes({"arxiv": "failed", "arxiv_http_status": 503}) == ["arXiv search failed (HTTP 503)"]
    assert notes({"semantic_scholar": "ok", "arxiv": "ok"}) == []
    assert notes({"arxiv": "disabled"}) == [] and notes(None) == []
    # One arXiv query answered: retrieval_status says "ok", and so does this.
    ok_then_refused = {"arxiv": {"1": {"status": "ok", "found": 5}, "2": {"status": "refused", "http_status": 406}}}
    assert runstate.lit_retrieval_status(ok_then_refused) == {"arxiv": "ok"}


def test_openalex_papers_are_tallied_with_the_other_sources() -> None:
    # arXiv refused every Python client on the Mac (HTTP 406); the pipeline
    # then asks OpenAlex with the same words. Its rows must count, and the
    # live view must not read as if only Semantic Scholar had answered.
    events = [_lit(3, "semantic_scholar", 1, 20), _lit(4, "semantic_scholar", 2, 15),
              _lit(5, "semantic_scholar", 3, 0, "rate_limited"),
              _lit(6, "arxiv", 1, 0, "refused", http_status=406),
              _lit(7, "arxiv", 2, 0, "skipped"), _lit(8, "arxiv", 3, 0, "skipped"),
              _lit(9, "openalex", 1, 10), _lit(10, "openalex", 2, 3), _lit(11, "openalex", 3, 10)]
    state = fold([event(1, "run.start", 0), *events])
    assert state.metrics["lit_sources"] == {"semantic_scholar": 35, "arxiv": 0, "openalex": 23}
    assert state.metrics["papers_found"] == 58
    assert state.metrics["lit_status"] == {"semantic_scholar": "ok", "arxiv": "refused",
                                           "arxiv_http_status": 406, "openalex": "ok", "n_openalex": 23}
    detail = _formulating_detail(events)
    assert detail == "58 papers found · arXiv refused our requests (HTTP 406); OpenAlex supplied 23 papers instead"


def test_an_openalex_that_also_turned_us_away_is_said_after_arxiv() -> None:
    events = [_lit(3, "semantic_scholar", 1, 0, "rate_limited"),
              _lit(4, "arxiv", 1, 0, "refused", http_status=406),
              _lit(5, "openalex", 1, 0, "rate_limited", http_status=429),
              _lit(6, "openalex", 2, 0, "skipped"), _lit(7, "openalex", 3, 0, "skipped")]
    state = fold([event(1, "run.start", 0), *events])
    assert state.metrics["lit_status"] == {"semantic_scholar": "rate_limited", "arxiv": "refused",
                                           "arxiv_http_status": 406, "openalex": "rate_limited",
                                           "n_openalex": 0, "openalex_http_status": 429}
    assert _formulating_detail(events) == (
        "0 papers found · Semantic Scholar turned our searches away (too many requests) · arXiv refused "
        "our requests (HTTP 406); OpenAlex, asked instead, turned our searches away (too many requests)")


def test_literature_notes_read_openalex_from_the_pipelines_retrieval_status() -> None:
    notes = runstate.literature_notes
    refused = {"semantic_scholar": "ok", "arxiv": "refused", "arxiv_http_status": 406}
    assert notes({**refused, "openalex": "ok", "n_openalex": 1}) == [
        "arXiv refused our requests (HTTP 406); OpenAlex supplied 1 paper instead"]
    assert notes({**refused, "openalex": "ok", "n_openalex": 0}) == [
        "arXiv refused our requests (HTTP 406); OpenAlex, asked instead, found none"]
    assert notes({**refused, "openalex": "refused", "openalex_http_status": 403}) == [
        "arXiv refused our requests (HTTP 406); OpenAlex, asked instead, refused our requests (HTTP 403)"]
    assert notes({"arxiv": "failed", "openalex": "failed", "openalex_http_status": 503}) == [
        "arXiv search failed; OpenAlex, asked instead, failed (HTTP 503)"]
    # Not asked (arXiv answered) or turned off: nothing to say about it.
    assert notes({"semantic_scholar": "ok", "arxiv": "ok", "openalex": "not_needed", "n_openalex": 0}) == []
    assert notes({**refused, "openalex": "disabled", "n_openalex": 0}) == [
        "arXiv refused our requests (HTTP 406)"]
    # Asked alongside an arXiv that answered (openalex.when: always).
    assert notes({"arxiv": "ok", "openalex": "failed", "openalex_http_status": 502}) == [
        "OpenAlex search failed (HTTP 502)"]


def test_usage_events_are_priced_like_src_cost() -> None:
    pricing = {"deepseek-v4-pro": {"input": 0.28, "cached_input": 0.028, "output": 0.42}}
    [ev] = usage_events([{"agent": "Analyst", "model": "deepseek-v4-pro", "prompt_tokens": 1_000_000,
                          "cached_prompt_tokens": 500_000, "completion_tokens": 1_000_000}], pricing)
    assert ev["data"]["cost_usd"] == pytest.approx(0.5 * 0.28 + 0.5 * 0.028 + 0.42)
    [unpriced] = usage_events([{"model": "mystery"}], pricing)
    assert unpriced["data"]["cost_usd"] is None


def test_usage_events_are_priced_by_the_hour_like_run_cost_json() -> None:
    # config.yaml now prices DeepSeek by the hour: the top-level rates are
    # the peak ones, and the flat formula charged every call at them.
    from src.cost import TokenUsage, cost_usd

    pricing = {"deepseek-v4-pro": {
        "input": 1.32, "cached_input": 0.044, "output": 3.96,
        "off_peak": {"input": 0.66, "cached_input": 0.022, "output": 1.98},
        "peak_windows_utc": {"days": ["mon", "tue", "wed", "thu", "fri"],
                             "hours": ["01:00-04:00", "06:00-10:00"]},
    }}
    row = {"agent": "Analyst", "model": "deepseek-v4-pro", "prompt_tokens": 1_000_000,
           "cached_prompt_tokens": 500_000, "completion_tokens": 1_000_000}
    saturday = {**row, "timestamp": "2026-09-26T13:50:00"}  # the Mac study: off-peak
    friday_peak = {**row, "timestamp": "2026-09-25T02:30:00"}
    untimed = dict(row)
    off, peak, unknown = (ev["data"]["cost_usd"] for ev in usage_events([saturday, friday_peak, untimed], pricing))
    assert off == pytest.approx(0.5 * 0.66 + 0.5 * 0.022 + 1.98)
    assert peak == pytest.approx(0.5 * 1.32 + 0.5 * 0.044 + 3.96)
    assert unknown == pytest.approx(peak)  # an unknown time is never priced low
    for got, source in ((off, saturday), (peak, friday_peak)):
        usage = TokenUsage(agent="Analyst", model="deepseek-v4-pro", provider="deepseek",
                           prompt_tokens=1_000_000, completion_tokens=1_000_000,
                           cached_prompt_tokens=500_000, timestamp=source["timestamp"])
        assert got == pytest.approx(cost_usd(usage, pricing))


def test_tail_state_of_a_finished_run(run_home: Path) -> None:
    run = make_run(run_home, results=PREDICTION_RESULTS,
                   review={"overall_quality_score": 8, "overall_verdict": "PASS", "unverified": False},
                   invariants={"counts": {"critical": 0, "major": 0, "minor": 0}, "findings": []},
                   extra={"retrieved_literature.json": json.dumps({"papers": [{}] * 38})})
    state = load_state(run)
    assert state.source == "tail"
    assert state.finished and state.final_state == "COMPLETED"
    assert all(_stage(state, k).status == "done" for k in
               ("FORMULATING", "ENGINEERING", "ANALYZING", "CRITIQUING", "WRITING", "VERIFYING"))
    assert _stage(state, "REVISING").status == "skipped"
    assert state.cost_usd == pytest.approx(0.0286) and state.llm_calls == 9
    assert state.metrics["best_model"] == "XGBoost"
    assert state.metrics["best_ci"] == [0.762, 0.80]
    assert _stage(state, "FORMULATING").detail == "38 papers found"
    assert _stage(state, "ENGINEERING").detail == "17,335 students · 42 predictors"
    assert _stage(state, "ANALYZING").detail == "Best: XGBoost, AUC 0.78 [0.76–0.80]"
    assert _stage(state, "CRITIQUING").detail == "Score 8/10 · passed"
    assert _stage(state, "WRITING").detail == "PDF ready"
    assert _stage(state, "VERIFYING").detail == "no problems found"
    assert _stage(state, "ANALYZING").duration_s() == pytest.approx(8 * 60)


def test_tail_running_run_waits_on_ai_and_counts_usage(run_home: Path) -> None:
    log = log_lines((0, "Starting FORMULATING stage"), (1, "FORMULATING stage complete"),
                    (1, "Starting ENGINEERING stage"))
    usage = "".join(json.dumps({"agent": "data_engineer", "model": "deepseek-v4-pro",
                                "prompt_tokens": 1000, "completion_tokens": 100,
                                "cached_prompt_tokens": 0}) + "\n" for _ in range(3))
    run = make_run(run_home, pid=alive_pid(), log=log, data_report=None, pdf=False,
                   extra={"token_usage.jsonl": usage,
                          "prompts/data_engineer/cycle_0/response_raw.txt": "old"})
    prompt = run / "prompts" / "data_engineer" / "cycle_0" / "rendered_prompt.txt"
    prompt.write_text("new prompt", encoding="utf-8")
    later = time.time() + 5
    os.utime(prompt, (later, later))
    state = load_state(run)
    assert not state.finished
    assert _stage(state, "ENGINEERING").status == "running"
    assert state.waiting_ai is True and state.waiting_agent == "data_engineer"
    assert state.llm_calls == 3
    assert state.cost_usd == pytest.approx(3 * (1000 * 0.28 + 100 * 0.42) / 1e6)
    assert "waiting for the AI" in state.now_text
    # generated code running
    (run / "_generated_script.py").write_text("print(1)", encoding="utf-8")
    state = load_state(run)
    assert state.code_running is True
    assert "Running the data-preparation code" in state.now_text


@pytest.mark.parametrize(("agent", "what"), [
    ("analyst", "Running the analysis code"),
    ("data_engineer", "Running the data-preparation code"),
    (None, "Running the data-preparation code"),  # an event without the agent: as before
])
def test_a_revision_names_the_code_of_the_agent_it_re_runs(agent: str | None, what: str) -> None:
    # A revision re-runs the Analyst's code under the REVISING step, and
    # the Now line took any step but ANALYZING for data preparation. The
    # automatic checks' pcc_07 revision re-runs only the Analyst.
    state = fold([
        event(1, "run.start", 0, task_type="prediction"),
        event(2, "stage.start", 0, stage="REVISING", cycle=1),
        event(3, "attempt.start", 1, stage="REVISING", agent=agent, attempt=1, max_attempts=3,
              timeout_s=1200),
    ])
    assert runstate.describe_now(state).startswith(f"{what} (attempt 1 of 3)")


def test_tail_reader_is_incremental_and_does_not_double_count_cost(run_home: Path) -> None:
    run = make_run(run_home, pid=alive_pid(), log=log_lines((0, "Starting FORMULATING stage")),
                   pdf=False, data_report=None,
                   extra={"token_usage.jsonl": json.dumps({"model": "deepseek-v4-pro",
                                                           "prompt_tokens": 1000}) + "\n"})
    reader = StateReader(run)
    assert reader.refresh().llm_calls == 1
    with open(run / "pipeline.log", "a", encoding="utf-8") as fh:
        fh.write(FULL_LOG)
    state = reader.refresh()
    # the logged total (9 calls, $0.0286) wins over the per-row sum
    assert state.llm_calls == 9 and state.cost_usd == pytest.approx(0.0286)
    assert reader.refresh().cost_usd == pytest.approx(0.0286)


def test_dead_process_without_end_is_crashed(run_home: Path) -> None:
    log = log_lines((0, "Starting FORMULATING stage"), (1, "FORMULATING stage complete"),
                    (1, "Starting ENGINEERING stage"))
    run = make_run(run_home, pid=dead_pid(), log=log, pdf=False, data_report=None)
    state = load_state(run)
    assert state.finished and state.final_state == "CRASHED"
    assert _stage(state, "ENGINEERING").status == "failed"
    (run / "STOP").write_text("x", encoding="utf-8")
    assert load_state(run).final_state == "STOPPED"


def test_run_without_pid_and_no_output_for_hours_is_not_running(run_home: Path) -> None:
    old = (datetime.now(timezone.utc) - timedelta(hours=10)).replace(tzinfo=None).isoformat()
    run = make_run(run_home, runner=False, log=f"{old} [Orchestrator] Starting FORMULATING stage\n",
                   pdf=False, data_report=None)
    assert load_state(run).finished is True
    fresh = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
    (run / "pipeline.log").write_text(f"{fresh} [Orchestrator] Starting FORMULATING stage\n",
                                      encoding="utf-8")
    assert load_state(run).finished is False


def test_checkpoint_terminal_state_ends_an_old_run(run_home: Path) -> None:
    log = log_lines((0, "Starting FORMULATING stage"), (1, "FORMULATING stage complete"),
                    (1, "Starting ENGINEERING stage"), (2, "ABORTED: ENGINEERING failed: boom"))
    run = make_run(run_home, runner=False, log=log, pdf=False, data_report=None,
                   checkpoint={"current_state": "ABORTED", "completed_stages": ["FORMULATING"],
                               "errors": ["ENGINEERING failed: boom"], "dataset_name": "hsls09_public",
                               "task_type": "prediction"})
    state = load_state(run)
    assert state.finished and state.final_state == "ABORTED"
    assert _stage(state, "ENGINEERING").status == "failed"
    assert state.dataset == "hsls09_public"


def test_stale_run_status_from_before_a_resume_is_ignored(run_home: Path) -> None:
    run = make_run(run_home, pid=alive_pid(), log=log_lines((0, "Starting WRITING stage")),
                   status={"released": True, "reason": "clean"})
    old = time.time() - 3600
    os.utime(run / "run_status.json", (old, old))
    info = json.loads((run / "runner.json").read_text(encoding="utf-8"))
    info["resumed_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    write_json(run / "runner.json", info)
    assert load_state(run).finished is False


def test_events_mode_is_preferred_and_merges_artifacts(run_home: Path) -> None:
    run = make_run(run_home, pid=alive_pid(), log=FULL_LOG, events=EVENTS, results=PREDICTION_RESULTS,
                   pdf=False)
    state = load_state(run)
    assert state.source == "events"
    assert not state.finished  # the log says COMPLETED, the events (source of truth) do not
    assert state.metrics["analytic_n"] == 17335  # from data_report.json
    assert state.question.startswith("Which ninth-graders")


def test_causal_and_psychometric_metrics() -> None:
    causal = runstate.causal_metrics(
        {"estimand": "ATE", "estimates": {
            "M1": {"point_estimate": 0.1, "ci_lower": 0.0, "ci_upper": 0.2},
            "M4": {"method_name": "M4 AIPW", "point_estimate": -0.12, "ci_lower": -0.2, "ci_upper": -0.04}}},
        "M4")
    assert causal["method"] == "M4" and causal["estimate"] == -0.12
    assert runstate.key_result(causal, "causal_soo") == "ATE: -0.12 [-0.20 to -0.04] (M4)"
    itr = runstate.causal_metrics({"estimates": {"M6": {"policy_rule_text": "x"},
                                                 "M7": {"value_gain_vs_best_constant": 0.03,
                                                        "gain_ci_lower": -0.01, "gain_ci_upper": 0.07}}},
                                  "M6")
    assert itr["method"] == "M7" and itr["estimate_kind"] == "policy_gain"
    psy = runstate.psychometric_metrics({"measurement_results": {
        "P2_omega": {"omega_total": 0.83}, "P3_cfa": {"fit": {"cfi": 0.97, "rmsea": 0.05}}}})
    assert psy["fit"] == {"omega": 0.83, "CFI": 0.97, "RMSEA": 0.05}
    assert runstate.psychometric_metrics({"headline": "The scale works  the same."})["headline"] == \
        "The scale works the same."


def test_progress_and_eta_ranges() -> None:
    state = fold(EVENTS)
    now = datetime(2026, 9, 25, 13, 6, tzinfo=timezone.utc)
    fraction, low, high = progress(state, now)
    assert 0.2 < fraction < 0.7
    assert low is not None and high is not None and now < low < high
    assert step_position(state) == (3, 7)  # REVIEWING hidden while LSAR is off
    state.lsar_enabled = True
    assert step_position(state)[1] == 8
    # with LSAR the remaining time is much longer
    _, low2, high2 = progress(state, now)
    assert high2 - high > timedelta(minutes=20)


def test_minority_share() -> None:
    assert runstate.minority_share({"class_0": 0.69, "class_1": 0.31}) == pytest.approx(0.31)
    assert runstate.minority_share({"0": 900, "1": 100}) == pytest.approx(0.1)
    assert runstate.minority_share(None) is None
    assert runstate.minority_share({"only": 1}) is None


def test_process_alive_checks_start_time() -> None:
    pid, created = alive_pid()
    assert runstate.process_alive(pid, created) is True
    assert runstate.process_alive(pid, 1.0) is False
    assert runstate.process_alive(None) is None


def test_the_analysis_step_names_the_model_its_metrics_put_first(run_home: Path) -> None:
    # results.json's best_model is the analysis's claim; the round-3 Mac
    # study's claim was contradicted by its own all_models.
    found = dict(PREDICTION_RESULTS, all_models={
        **PREDICTION_RESULTS["all_models"],
        "RandomForest": {"auc": 0.812, "auc_ci_lower": 0.794, "auc_ci_upper": 0.83}})
    state = load_state(make_run(run_home, results=found))
    assert state.metrics["best_model"] == "RandomForest"
    assert state.metrics["claimed_best_model"] == "XGBoost"
    assert state.metrics["claimed_metric_value"] == pytest.approx(0.781)
    assert state.metrics["best_ci"] == [0.794, 0.83]
    assert _stage(state, "ANALYZING").detail == \
        "Best by AUC: RandomForest, AUC 0.81 [0.79–0.83] (the analysis named XGBoost)"


def test_the_analysis_step_names_the_best_single_model_and_a_higher_ensemble(run_home: Path) -> None:
    # results.json's best_model is the best single model by design; the
    # round-3 Mac study's stacking ensemble scored higher.
    found = dict(PREDICTION_RESULTS, all_models={
        **PREDICTION_RESULTS["all_models"], "StackingEnsemble": {"auc": 0.79}})
    state = load_state(make_run(run_home, results=found))
    assert state.metrics["best_model"] == "XGBoost"
    assert state.metrics["ensemble_model"] == "StackingEnsemble"
    assert "claimed_best_model" not in state.metrics
    assert _stage(state, "ANALYZING").detail == \
        "Best single model: XGBoost, AUC 0.78 [0.76–0.80]; the stacking ensemble 0.79"


def test_best_by_metric_respects_the_metrics_direction() -> None:
    models = {"A": {"auc": 0.7, "rmse": 0.5}, "B": {"AUC": 0.8, "rmse": 0.6}, "C": {"note": "failed"}}
    assert runstate.best_by_metric(models, "AUC") == (["B"], 0.8)
    assert runstate.best_by_metric(models, "RMSE") == (["A"], 0.5)
    assert runstate.best_by_metric(models, "something_else") is None
    assert runstate.best_by_metric({"A": {}}, "AUC") is None
