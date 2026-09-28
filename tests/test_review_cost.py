"""The review gate's spending is part of the run's cost.

The owner's Mac study (round 3) reported US$0.300 over 15 calls while the
DeepSeek balance fell by US$0.57: the review gate's six LSAR reviews (42
calls, written by LSAR to token_usage.json in each review folder) and its
paper-revision call reached neither token_usage.jsonl nor run_cost.json.

These tests pin how src/cost.py turns an LSAR review's usage into run
rows (with or without per-call timestamps), and how run_cost.json shows
the pipeline and the review as separate subtotals.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.cost import (
    REVIEW_COMPONENT,
    TokenUsage,
    load_review_usage,
    load_usage,
    load_usage_from_checkpoint,
    record_usage,
    review_usages,
    summarize,
    window_period,
    write_review_window,
    write_summary,
)

TOD = {
    "tod-model": {
        "input": 2.0, "cached_input": 0.5, "output": 4.0,
        "off_peak": {"input": 1.0, "cached_input": 0.25, "output": 2.0},
        "peak_windows_utc": {
            "days": ["mon", "tue", "wed", "thu", "fri"],
            "hours": ["01:00-04:00", "06:00-10:00"],
        },
    },
}
CONFIG = {"pricing": {"per_million_tokens": TOD}}
PEAK = 2.0 + 4.0      # 1M uncached input + 1M output at the peak rate
OFF_PEAK = 1.0 + 2.0

# 2026-09-26 is a Saturday (off-peak all day); 2026-09-28 a Monday.
SAT_START, SAT_END = "2026-09-26T13:41:00", "2026-09-26T13:48:00"


def _lsar_payload(n: int = 2, *, timestamps: list | None = None) -> dict:
    """What LSAR's _write_review_cost writes (lsar/pipeline.py)."""
    calls = []
    for i in range(n):
        call = {"provider": "deepseek", "model": "tod-model", "stage": "review",
                "prompt_tokens": 1_000_000, "completion_tokens": 1_000_000,
                "cached_prompt_tokens": 0}
        if timestamps is not None:
            call["timestamp"] = timestamps[i]
        calls.append(call)
    return {
        "n_calls": n,
        "prompt_tokens": n * 1_000_000,
        "completion_tokens": n * 1_000_000,
        "cached_prompt_tokens": 0,
        "by_model": {"tod-model": {"n_calls": n, "prompt_tokens": n * 1_000_000,
                                   "completion_tokens": n * 1_000_000}},
        "calls": calls,
    }


class TestWindowPeriod:
    def test_a_window_inside_one_period_has_that_period(self) -> None:
        rates = TOD["tod-model"]
        assert window_period(SAT_START, SAT_END, rates) == "off_peak"
        assert window_period("2026-09-28T07:00:00", "2026-09-28T07:40:00", rates) == "peak"
        assert window_period("2026-09-28T11:00:00", "2026-09-28T11:30:00", rates) == "off_peak"

    def test_a_window_crossing_a_boundary_is_mixed(self) -> None:
        rates = TOD["tod-model"]
        assert window_period("2026-09-28T09:55:00", "2026-09-28T10:05:00", rates) == "mixed"
        # Off-peak at both ends, peak in the middle.
        assert window_period("2026-09-28T00:50:00", "2026-09-28T04:10:00", rates) == "mixed"

    def test_unknown_or_reversed_ends_are_untimed(self) -> None:
        rates = TOD["tod-model"]
        assert window_period(None, SAT_END, rates) == "untimed"
        assert window_period(SAT_END, SAT_START, rates) == "untimed"

    def test_flat_rates_need_no_window(self) -> None:
        assert window_period(None, None, {"input": 1.0, "output": 1.0}) == "flat"


class TestReviewUsages:
    def test_untimed_calls_take_the_review_window(self) -> None:
        rows = review_usages(_lsar_payload(), started=SAT_START, ended=SAT_END,
                             pricing=TOD)
        assert len(rows) == 2
        assert all(r.component == REVIEW_COMPONENT for r in rows)
        assert all(r.agent == "LSAR" and r.stage == "REVIEWING" for r in rows)
        assert all(r.timestamp == SAT_START for r in rows)
        assert all(r.time_source == "review_window" for r in rows)
        s = summarize(rows, TOD)
        assert s.cost_usd == 2 * OFF_PEAK
        assert s.cost_status == "measured"

    def test_a_call_stamped_by_lsar_keeps_its_own_time(self) -> None:
        """The LSAR that stamps each call: a peak-hour call is peak even
        when the rest of the review was not."""
        payload = _lsar_payload(timestamps=["2026-09-28T09:59:00", "2026-09-28T10:03:00"])
        rows = review_usages(payload, started="2026-09-28T09:55:00",
                             ended="2026-09-28T10:05:00", pricing=TOD)
        assert [r.time_source for r in rows] == [None, None]
        assert summarize(rows, TOD).cost_usd == PEAK + OFF_PEAK

    def test_the_rows_lsar_now_writes_are_read_as_they_are(self) -> None:
        """LSAR fix/released-issues (f6bc00c, f0ac437): each row carries an
        aware UTC timestamp, its stage, reasoning tokens and an outcome."""
        payload = _lsar_payload(1)
        payload["calls"][0].update(timestamp="2026-09-28T07:30:00+00:00",
                                   reasoning_tokens=1234, outcome="truncated")
        [row] = review_usages(payload, pricing=TOD)
        assert row.time_source is None and row.reasoning_tokens == 1234
        # A refused answer was still billed: it is priced like any other.
        assert summarize([row], TOD).cost_usd == PEAK

    def test_a_window_across_a_peak_boundary_is_charged_peak_and_estimated(self) -> None:
        rows = review_usages(_lsar_payload(), started="2026-09-28T09:55:00",
                             ended="2026-09-28T10:05:00", pricing=TOD)
        assert all(r.timestamp is None and r.time_source == "unknown" for r in rows)
        s = summarize(rows, TOD)
        assert s.cost_usd == 2 * PEAK
        assert s.cost_status == "estimated"

    def test_no_window_is_charged_peak_and_estimated(self) -> None:
        rows = review_usages(_lsar_payload(), pricing=TOD)
        s = summarize(rows, TOD)
        assert s.cost_usd == 2 * PEAK and s.untimed_calls == 2
        assert s.cost_status == "estimated"

    def test_totals_without_calls_keep_the_call_count_and_the_cost(self) -> None:
        payload = _lsar_payload(3)
        del payload["calls"]
        rows = review_usages(payload, started=SAT_START, ended=SAT_END, pricing=TOD)
        assert len(rows) == 3
        assert sum(r.prompt_tokens for r in rows) == 3_000_000
        assert summarize(rows, TOD).cost_usd == 3 * OFF_PEAK

    def test_a_call_row_without_a_model_takes_the_only_model(self) -> None:
        payload = _lsar_payload(1)
        del payload["calls"][0]["model"]
        [row] = review_usages(payload, started=SAT_START, ended=SAT_END, pricing=TOD)
        assert row.model == "tod-model"

    def test_garbage_is_no_rows(self) -> None:
        assert review_usages(None) == []
        assert review_usages({"calls": ["x"]}) == []

    def test_a_review_folder_is_read_with_its_window(self, tmp_path: Path) -> None:
        (tmp_path / "token_usage.json").write_text(
            json.dumps(_lsar_payload()), encoding="utf-8")
        assert load_review_usage(tmp_path, TOD)[0].time_source == "unknown"
        write_review_window(tmp_path, SAT_START, SAT_END)
        rows = load_review_usage(tmp_path, TOD)
        assert [r.time_source for r in rows] == ["review_window"] * 2
        assert load_review_usage(tmp_path / "missing", TOD) == []


class TestRunCostSubtotals:
    def _run(self, tmp_path: Path) -> dict:
        record_usage(str(tmp_path), TokenUsage(
            "writer", "tod-model", "deepseek", 1_000_000, 1_000_000,
            timestamp=SAT_START))
        for row in review_usages(_lsar_payload(), started=SAT_START,
                                 ended=SAT_END, pricing=TOD):
            record_usage(str(tmp_path), row)
        return write_summary(str(tmp_path), CONFIG)

    def test_run_cost_counts_the_review_and_shows_both_subtotals(
        self, tmp_path: Path
    ) -> None:
        payload = self._run(tmp_path)
        assert payload["n_calls"] == 3
        assert payload["cost_usd"] == 3 * OFF_PEAK
        assert payload["pipeline_cost_usd"] == OFF_PEAK
        assert payload["review_cost_usd"] == 2 * OFF_PEAK
        review = payload["by_component"]["review"]
        assert review["n_calls"] == 2 and review["cost_status"] == "measured"
        assert payload["by_component"]["pipeline"]["n_calls"] == 1
        assert "review gate made 2 of the 3 calls" in payload["note"]
        on_disk = json.loads((tmp_path / "run_cost.json").read_text(encoding="utf-8"))
        assert on_disk["review_cost_usd"] == 2 * OFF_PEAK

    def test_a_run_without_a_review_shows_a_zero_review_subtotal(
        self, tmp_path: Path
    ) -> None:
        record_usage(str(tmp_path), TokenUsage(
            "writer", "tod-model", "deepseek", 1_000_000, 0, timestamp=SAT_START))
        payload = write_summary(str(tmp_path), CONFIG)
        assert payload["review_cost_usd"] == 0.0
        assert payload["by_component"]["review"]["n_calls"] == 0
        assert "review gate" not in payload["note"]

    def test_pipeline_rows_are_written_as_before(self, tmp_path: Path) -> None:
        record_usage(str(tmp_path), TokenUsage("writer", "m", "p", 1, 1))
        line = json.loads((tmp_path / "token_usage.jsonl").read_text(encoding="utf-8"))
        assert "component" not in line and "time_source" not in line

    def test_review_rows_round_trip(self, tmp_path: Path) -> None:
        self._run(tmp_path)
        rows = load_usage(str(tmp_path))
        assert [r.component for r in rows] == [None, "review", "review"]
        assert rows[1].time_source == "review_window"

    def test_the_checkpoint_copy_keeps_the_component(self, tmp_path: Path) -> None:
        (tmp_path / "checkpoint.json").write_text(json.dumps({"log": [
            {"agent": "LSAR", "model": "tod-model", "prompt_tokens": 1_000_000,
             "completion_tokens": 1_000_000, "cached_prompt_tokens": 0,
             "timestamp": SAT_START, "component": "review",
             "time_source": "review_window"}]}), encoding="utf-8")
        [row] = load_usage_from_checkpoint(str(tmp_path))
        assert row.component == "review"
        assert summarize([row], TOD).by_component["review"]["cost_usd"] == OFF_PEAK
