"""K1 — token metering and cost accounting.

Before this, every LLM call logged one number: prompt + completion
SUMMED. That cannot be priced, because input and output cost different
amounts and DeepSeek prices a prompt-cache hit about 10x below a miss.
The orchestrator's budget check compounded it by multiplying the sum by
a hardcoded 0.000015 — $15 per million, an Anthropic-era rate left
behind when the stack moved to DeepSeek, overstating cost ~40x.

The rules these tests pin:
  - an unpriced model yields None, never a silent zero;
  - counts are stored raw so a rate change re-prices history;
  - metering never raises into the pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from src.cost import (
    CostSummary,
    TokenUsage,
    cost_usd,
    extract_usage,
    load_pricing,
    load_usage,
    record_usage,
    summarize,
    write_summary,
)

# Synthetic rates keyed by arbitrary model ids. "deepseek-v4-flash" is
# the retired flash id, used here only as a name for a cheaper tier;
# the shipped config prices and routes to "deepseek-flash".
PRICING = {
    "deepseek-v4-pro": {"input": 0.28, "cached_input": 0.028, "output": 0.42},
    "deepseek-v4-flash": {"input": 0.07, "cached_input": 0.007, "output": 0.28},
}


class TestExtractUsage:
    def test_openai_shape(self) -> None:
        r = SimpleNamespace(usage=SimpleNamespace(
            prompt_tokens=1000, completion_tokens=250))
        u = extract_usage(r, "writer", "deepseek-v4-pro", "deepseek")
        assert (u.prompt_tokens, u.completion_tokens) == (1000, 250)
        assert u.total_tokens == 1250

    def test_anthropic_shape(self) -> None:
        r = SimpleNamespace(usage=SimpleNamespace(
            input_tokens=800, output_tokens=200))
        u = extract_usage(r, "critic", "claude-opus-4-6", "anthropic")
        assert (u.prompt_tokens, u.completion_tokens) == (800, 200)

    def test_deepseek_cache_hit_tokens(self) -> None:
        r = SimpleNamespace(usage=SimpleNamespace(
            prompt_tokens=5000, completion_tokens=100,
            prompt_cache_hit_tokens=4000))
        u = extract_usage(r, "analyst", "deepseek-v4-pro", "deepseek")
        assert u.cached_prompt_tokens == 4000

    def test_openai_nested_cached_tokens(self) -> None:
        r = SimpleNamespace(usage=SimpleNamespace(
            prompt_tokens=5000, completion_tokens=100,
            prompt_tokens_details=SimpleNamespace(cached_tokens=3000)))
        u = extract_usage(r, "a", "m", "openai")
        assert u.cached_prompt_tokens == 3000

    def test_missing_usage_is_zeros_not_an_error(self) -> None:
        u = extract_usage(SimpleNamespace(), "a", "m", "p")
        assert u.total_tokens == 0

    def test_usage_none_is_zeros(self) -> None:
        u = extract_usage(SimpleNamespace(usage=None), "a", "m", "p")
        assert u.total_tokens == 0


class TestCost:
    def test_input_and_output_priced_separately(self) -> None:
        """The whole point: a summed count cannot produce this number."""
        u = TokenUsage("writer", "deepseek-v4-pro", "deepseek",
                       prompt_tokens=1_000_000, completion_tokens=1_000_000)
        assert cost_usd(u, PRICING) == 0.28 + 0.42

    def test_cache_hits_priced_lower(self) -> None:
        cached = TokenUsage("a", "deepseek-v4-pro", "d",
                            prompt_tokens=1_000_000,
                            cached_prompt_tokens=1_000_000)
        uncached = TokenUsage("a", "deepseek-v4-pro", "d",
                              prompt_tokens=1_000_000)
        assert cost_usd(cached, PRICING) == 0.028
        assert cost_usd(uncached, PRICING) == 0.28
        assert cost_usd(cached, PRICING) < cost_usd(uncached, PRICING)

    def test_unpriced_model_returns_none_not_zero(self) -> None:
        u = TokenUsage("a", "some-unlisted-model", "p", prompt_tokens=10_000)
        assert cost_usd(u, PRICING) is None

    def test_empty_pricing_returns_none(self) -> None:
        u = TokenUsage("a", "deepseek-v4-pro", "d", prompt_tokens=10_000)
        assert cost_usd(u, {}) is None

    def test_flash_is_cheaper_than_pro(self) -> None:
        """Model tiering only pays off if the rates differ."""
        args = dict(prompt_tokens=100_000, completion_tokens=10_000)
        pro = cost_usd(TokenUsage("a", "deepseek-v4-pro", "d", **args), PRICING)
        flash = cost_usd(TokenUsage("a", "deepseek-v4-flash", "d", **args), PRICING)
        assert flash < pro


class TestSummarize:
    def _usages(self) -> list[TokenUsage]:
        return [
            TokenUsage("writer", "deepseek-v4-pro", "d", 100_000, 20_000),
            TokenUsage("writer", "deepseek-v4-pro", "d", 50_000, 10_000),
            TokenUsage("outline_agent", "deepseek-v4-flash", "d", 20_000, 5_000),
        ]

    def test_totals_and_breakdown(self) -> None:
        s = summarize(self._usages(), PRICING)
        assert s.n_calls == 3
        assert s.prompt_tokens == 170_000
        assert s.completion_tokens == 35_000
        assert s.total_tokens == 205_000
        assert s.by_agent["writer"]["n_calls"] == 2
        assert set(s.by_model) == {"deepseek-v4-pro", "deepseek-v4-flash"}

    def test_cost_is_sum_of_parts(self) -> None:
        us = self._usages()
        s = summarize(us, PRICING)
        assert abs(s.cost_usd - sum(cost_usd(u, PRICING) for u in us)) < 1e-9

    def test_partially_priced_run_flags_the_gap(self) -> None:
        """A run mixing a priced and an unpriced model must report the
        unpriced one, so the total is read as a LOWER BOUND."""
        us = self._usages() + [TokenUsage("x", "mystery-model", "p", 1_000, 1_000)]
        s = summarize(us, PRICING)
        assert s.unpriced_models == ["mystery-model"]
        assert s.cost_usd is not None  # partial total still reported

    def test_fully_unpriced_run_has_none_cost(self) -> None:
        s = summarize(self._usages(), {})
        assert s.cost_usd is None
        assert len(s.unpriced_models) == 2

    def test_empty_run(self) -> None:
        s = summarize([], PRICING)
        assert s == CostSummary()


class TestPersistence:
    def test_record_and_load_round_trip(self, tmp_path: Path) -> None:
        u = TokenUsage("writer", "deepseek-v4-pro", "d", 123, 45, 67,
                       stage="WRITING", timestamp="2026-08-08T00:00:00")
        record_usage(str(tmp_path), u)
        record_usage(str(tmp_path), u)
        back = load_usage(str(tmp_path))
        assert len(back) == 2
        assert back[0].prompt_tokens == 123
        assert back[0].cached_prompt_tokens == 67
        assert back[0].stage == "WRITING"

    def test_record_with_no_output_dir_is_a_noop(self) -> None:
        record_usage(None, TokenUsage("a", "m", "p"))  # must not raise

    def test_load_missing_file_is_empty(self, tmp_path: Path) -> None:
        assert load_usage(str(tmp_path)) == []

    def test_corrupt_line_is_skipped_not_fatal(self, tmp_path: Path) -> None:
        p = tmp_path / "token_usage.jsonl"
        p.write_text(
            '{"agent":"a","model":"m","provider":"p","prompt_tokens":5}\n'
            "NOT JSON\n"
            '{"agent":"b","model":"m","provider":"p","prompt_tokens":7}\n',
            encoding="utf-8",
        )
        back = load_usage(str(tmp_path))
        assert [u.prompt_tokens for u in back] == [5, 7]

    def test_write_summary_records_pricing_provenance(self, tmp_path: Path) -> None:
        record_usage(str(tmp_path),
                     TokenUsage("w", "deepseek-v4-pro", "d", 1000, 100))
        payload = write_summary(str(tmp_path),
                                {"pricing": {"per_million_tokens": PRICING}})
        assert payload["cost_usd"] is not None
        assert "config.yaml" in payload["pricing_source"]
        on_disk = json.loads((tmp_path / "run_cost.json").read_text(encoding="utf-8"))
        assert on_disk["n_calls"] == 1

    def test_unpriced_model_is_declared_not_hidden(self, tmp_path: Path) -> None:
        """A model with no rate anywhere must read as 'not priced',
        never as $0.00."""
        record_usage(str(tmp_path),
                     TokenUsage("w", "no-such-model-anywhere", "d", 1000, 100))
        payload = write_summary(str(tmp_path), {})
        assert payload["cost_usd"] is None
        assert payload["unpriced_models"] == ["no-such-model-anywhere"]

    def test_run_config_without_rates_falls_back_to_repo_config(
        self, tmp_path: Path
    ) -> None:
        """Rates belong to the provider, not the run. A per-run config
        that omits them must still price — otherwise the one run config
        that forgot silently reports itself as unpriced."""
        record_usage(str(tmp_path),
                     TokenUsage("w", "deepseek-v4-pro", "d", 1_000_000, 0))
        payload = write_summary(str(tmp_path), {})  # no pricing block
        assert payload["cost_usd"] is not None
        assert payload["cost_usd"] > 0

    def test_summary_of_empty_run_is_none(self, tmp_path: Path) -> None:
        assert write_summary(str(tmp_path), {}) is None


class TestConfigWiring:
    def test_repo_config_prices_the_models_it_uses(self) -> None:
        """The shipped config must price every model the shipped config
        routes to, or real runs come out partially unpriced."""
        import yaml

        cfg = yaml.safe_load(
            (Path(__file__).resolve().parent.parent / "config.yaml").read_text(
                encoding="utf-8"
            )
        )
        pricing = load_pricing(cfg)
        assert pricing, "config.yaml has no pricing.per_million_tokens block"
        routed = set((cfg.get("deepseek") or {}).get("models", {}).values())
        missing = routed - set(pricing)
        assert not missing, f"models routed but unpriced: {sorted(missing)}"

    def test_rates_have_all_three_components(self) -> None:
        import yaml

        cfg = yaml.safe_load(
            (Path(__file__).resolve().parent.parent / "config.yaml").read_text(
                encoding="utf-8"
            )
        )
        for model, rates in load_pricing(cfg).items():
            for key in ("input", "cached_input", "output"):
                assert key in rates, f"{model} missing {key} rate"
            assert rates["output"] > 0 and rates["input"] > 0
            assert rates["cached_input"] <= rates["input"], model


class TestRedundantRecord:
    """K1 writes each call to BOTH token_usage.jsonl and ctx.log (which
    is re-serialized into checkpoint.json every stage). The jsonl is an
    append-only file that a crash mid-write, a resumed run, or a stray
    command can truncate; the checkpoint is rewritten whole from memory.
    Costing must take whichever record is more complete, because an
    undercount reads exactly like a cheap run."""

    def _checkpoint(self, tmp_path: Path, n: int) -> None:
        entries = [
            {"agent": "writer", "model": "deepseek-v4-pro", "tokens_used": 1100,
             "prompt_tokens": 1000, "completion_tokens": 100,
             "cached_prompt_tokens": 400, "timestamp": f"t{i}"}
            for i in range(n)
        ]
        entries.insert(0, {"agent": "x", "message": "not a usage entry"})
        (tmp_path / "checkpoint.json").write_text(
            json.dumps({"log": entries}), encoding="utf-8")

    def test_recovers_usage_from_checkpoint(self, tmp_path: Path) -> None:
        from src.cost import load_usage_from_checkpoint

        self._checkpoint(tmp_path, 3)
        rows = load_usage_from_checkpoint(str(tmp_path))
        assert len(rows) == 3
        assert rows[0].prompt_tokens == 1000
        assert rows[0].cached_prompt_tokens == 400

    def test_prefers_the_more_complete_source(self, tmp_path: Path) -> None:
        """The exact recovery: the jsonl lost rows, the checkpoint kept
        them all."""
        from src.cost import load_usage_best

        self._checkpoint(tmp_path, 6)
        record_usage(str(tmp_path),
                     TokenUsage("writer", "deepseek-v4-pro", "d", 1000, 100))
        assert len(load_usage_best(str(tmp_path))) == 6

    def test_prefers_jsonl_when_it_is_richer(self, tmp_path: Path) -> None:
        from src.cost import load_usage_best

        self._checkpoint(tmp_path, 1)
        for _ in range(4):
            record_usage(str(tmp_path),
                         TokenUsage("w", "deepseek-v4-pro", "d", 10, 1))
        assert len(load_usage_best(str(tmp_path))) == 4

    def test_no_checkpoint_is_not_an_error(self, tmp_path: Path) -> None:
        from src.cost import load_usage_from_checkpoint

        assert load_usage_from_checkpoint(str(tmp_path)) == []

    def test_corrupt_checkpoint_is_not_an_error(self, tmp_path: Path) -> None:
        from src.cost import load_usage_from_checkpoint

        (tmp_path / "checkpoint.json").write_text("{ broken", encoding="utf-8")
        assert load_usage_from_checkpoint(str(tmp_path)) == []


class TestUnpricedAndUnverifiedRates:
    """E5: an unpriced entry in the breakdowns read as $0.00, and a rate
    nobody had checked read exactly like a measured one."""

    def test_unpriced_breakdown_entries_are_none_not_zero(self) -> None:
        us = [
            TokenUsage("writer", "deepseek-v4-pro", "d", 1_000, 1_000),
            TokenUsage("critic", "claude-opus-4-6", "a", 1_000, 1_000),
        ]
        s = summarize(us, PRICING)
        assert s.by_model["claude-opus-4-6"]["cost_usd"] is None
        assert s.by_model["claude-opus-4-6"]["unpriced_calls"] == 1
        assert s.by_agent["critic"]["cost_usd"] is None
        assert s.by_agent["writer"]["cost_usd"] == cost_usd(us[0], PRICING)
        assert s.cost_usd == cost_usd(us[0], PRICING)
        assert s.unpriced_calls == 1
        assert s.cost_status == "partial"

    def test_mixed_entry_keeps_its_priced_part_and_counts_the_gap(self) -> None:
        us = [
            TokenUsage("writer", "deepseek-v4-pro", "d", 1_000, 0),
            TokenUsage("writer", "mystery", "d", 1_000, 0),
        ]
        entry = summarize(us, PRICING).by_agent["writer"]
        assert entry["cost_usd"] == cost_usd(us[0], PRICING)
        assert entry["unpriced_calls"] == 1

    def test_fully_verified_run_is_measured(self) -> None:
        s = summarize([TokenUsage("w", "deepseek-v4-pro", "d", 10, 10)], PRICING)
        assert s.cost_status == "measured"
        assert s.unverified_rate_models == []

    def test_unverified_rate_makes_the_cost_an_estimate(self, tmp_path: Path) -> None:
        pricing = {
            **PRICING,
            "deepseek-flash": {"input": 0.07, "cached_input": 0.007,
                               "output": 0.28, "unverified": True},
        }
        record_usage(str(tmp_path), TokenUsage("outline_agent", "deepseek-flash", "d", 1_000, 100))
        record_usage(str(tmp_path), TokenUsage("writer", "deepseek-v4-pro", "d", 1_000, 100))
        payload = write_summary(str(tmp_path), {"pricing": {"per_million_tokens": pricing}})
        assert payload["cost_usd"] is not None
        assert payload["cost_status"] == "estimated"
        assert payload["unverified_rate_models"] == ["deepseek-flash"]
        assert "ESTIMATE" in payload["note"]

    def test_verified_false_is_the_same_flag(self) -> None:
        from src.cost import rate_is_unverified

        assert rate_is_unverified({"input": 1, "verified": False})
        assert rate_is_unverified({"input": 1, "unverified": True})
        assert not rate_is_unverified({"input": 1})
        assert not rate_is_unverified(None)

    def test_flag_keys_do_not_change_the_price(self) -> None:
        flagged = {"m": {"input": 1.0, "output": 2.0, "unverified": True}}
        plain = {"m": {"input": 1.0, "output": 2.0}}
        u = TokenUsage("a", "m", "d", 1_000_000, 1_000_000)
        assert cost_usd(u, flagged) == cost_usd(u, plain) == 3.0


class TestReasoningTokens:
    """DeepSeek documents ``completion_tokens_details.reasoning_tokens`` as a
    breakdown OF ``completion_tokens`` (OpenAI and Anthropic count thinking
    the same way), and bills all of ``completion_tokens`` as output."""

    def test_deepseek_shape_keeps_reasoning_inside_completion(self) -> None:
        r = SimpleNamespace(usage=SimpleNamespace(
            prompt_tokens=5000, completion_tokens=1200,
            prompt_cache_hit_tokens=4000, prompt_cache_miss_tokens=1000,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=900)))
        u = extract_usage(r, "analyst", "deepseek-v4-pro", "deepseek")
        assert u.completion_tokens == 1200
        assert u.reasoning_tokens == 900

    def test_reasoning_is_not_charged_twice(self) -> None:
        plain = TokenUsage("a", "deepseek-v4-pro", "d", 1_000, 1_000_000)
        thinking = TokenUsage("a", "deepseek-v4-pro", "d", 1_000, 1_000_000,
                              reasoning_tokens=800_000)
        assert cost_usd(thinking, PRICING) == cost_usd(plain, PRICING)


#: A time-of-day entry shaped like the shipped DeepSeek ones, with round
#: numbers so the expected costs are obvious.
TOD = {
    "tod-model": {
        "input": 2.0, "cached_input": 0.2, "output": 4.0,
        "off_peak": {"input": 1.0, "cached_input": 0.1, "output": 2.0},
        "peak_windows_utc": {
            "days": ["mon", "tue", "wed", "thu", "fri"],
            "hours": ["01:00-04:00", "06:00-10:00"],
        },
    },
}
PEAK = 2.0 + 4.0      # 1M uncached input + 1M output at the peak rate
OFF_PEAK = 1.0 + 2.0


def _tod_call(timestamp: object) -> TokenUsage:
    return TokenUsage("writer", "tod-model", "deepseek",
                      prompt_tokens=1_000_000, completion_tokens=1_000_000,
                      timestamp=timestamp)  # type: ignore[arg-type]


class TestTimeOfDayRates:
    """DeepSeek charges twice as much from 01:00 to 04:00 and from 06:00 to
    10:00 UTC on weekdays as at any other time. With one flat rate, a
    study run on a Saturday and one run on a Monday morning read the same,
    and the rate that had been configured matched neither."""

    def test_weekday_peak_hour_is_charged_the_peak_rate(self) -> None:
        # 2026-09-28 is a Monday.
        assert cost_usd(_tod_call("2026-09-28T07:30:00"), TOD) == PEAK

    def test_weekday_outside_the_windows_is_off_peak(self) -> None:
        assert cost_usd(_tod_call("2026-09-28T12:00:00"), TOD) == OFF_PEAK
        assert cost_usd(_tod_call("2026-09-28T05:00:00"), TOD) == OFF_PEAK

    def test_window_start_is_inside_and_end_is_outside(self) -> None:
        assert cost_usd(_tod_call("2026-09-28T01:00:00"), TOD) == PEAK
        assert cost_usd(_tod_call("2026-09-28T03:59:59"), TOD) == PEAK
        assert cost_usd(_tod_call("2026-09-28T04:00:00"), TOD) == OFF_PEAK

    def test_weekend_is_off_peak_all_day(self) -> None:
        """The macOS test study ran on Saturday 2026-09-26, 13:41-13:55 UTC."""
        for ts in ("2026-09-26T07:30:00", "2026-09-26T13:50:00",
                   "2026-09-27T02:00:00"):
            assert cost_usd(_tod_call(ts), TOD) == OFF_PEAK, ts

    def test_aware_timestamps_are_converted_to_utc(self) -> None:
        # 03:30 at UTC-4 is 07:30 UTC, inside the Monday window.
        assert cost_usd(_tod_call("2026-09-28T03:30:00-04:00"), TOD) == PEAK
        # 07:30 at UTC+9 on Monday is 22:30 UTC on Sunday.
        assert cost_usd(_tod_call("2026-09-28T07:30:00+09:00"), TOD) == OFF_PEAK
        assert cost_usd(_tod_call("2026-09-28T07:30:00Z"), TOD) == PEAK

    def test_unknown_time_is_charged_the_peak_rate(self) -> None:
        """Never price a call below what it can have cost."""
        for ts in (None, "", "t0", "not a date"):
            assert cost_usd(_tod_call(ts), TOD) == PEAK, ts

    def test_unreadable_schedule_is_charged_the_peak_rate(self) -> None:
        for schedule in (None, "weekdays", {"days": ["mon"], "hours": ["7-9"]},
                         {"days": ["someday"], "hours": ["01:00-04:00"]},
                         {"days": [], "hours": ["01:00-04:00"]}):
            rates = {"tod-model": {**TOD["tod-model"], "peak_windows_utc": schedule}}
            assert cost_usd(_tod_call("2026-09-26T13:50:00"), rates) == PEAK, schedule

    def test_flat_entries_ignore_the_time(self) -> None:
        u = TokenUsage("a", "deepseek-v4-pro", "d", 1_000_000, 0,
                       timestamp="2026-09-28T07:30:00")
        assert cost_usd(u, PRICING) == 0.28

    def test_summary_counts_the_periods_and_stays_measured(self) -> None:
        from src.cost import rate_period

        calls = [_tod_call("2026-09-28T07:30:00"), _tod_call("2026-09-26T13:50:00"),
                 _tod_call("2026-09-28T12:00:00")]
        assert [rate_period(u, TOD["tod-model"]) for u in calls] == [
            "peak", "off_peak", "off_peak"]
        s = summarize(calls, TOD)
        assert (s.peak_calls, s.off_peak_calls, s.untimed_calls) == (1, 2, 0)
        assert s.cost_usd == PEAK + 2 * OFF_PEAK
        assert s.cost_status == "measured"

    def test_a_call_of_unknown_time_makes_the_run_an_estimate(
        self, tmp_path: Path
    ) -> None:
        record_usage(str(tmp_path), _tod_call("2026-09-26T13:50:00"))
        record_usage(str(tmp_path), _tod_call(None))
        payload = write_summary(str(tmp_path),
                                {"pricing": {"per_million_tokens": TOD}})
        assert payload["untimed_calls"] == 1
        assert payload["off_peak_calls"] == 1
        assert payload["cost_status"] == "estimated"
        assert payload["cost_usd"] == OFF_PEAK + PEAK
        assert "ESTIMATE" in payload["note"] and "peak rate" in payload["note"]

    def test_note_says_how_many_calls_were_peak(self, tmp_path: Path) -> None:
        record_usage(str(tmp_path), _tod_call("2026-09-28T07:30:00"))
        payload = write_summary(str(tmp_path),
                                {"pricing": {"per_million_tokens": TOD}})
        assert payload["cost_status"] == "measured"
        assert "1 call(s) fell in its peak window" in payload["note"]
        assert "public holiday" in payload["note"]

    def test_checkpoint_rows_keep_their_time(self, tmp_path: Path) -> None:
        """The recovery path must price at the hour too, not fall to peak."""
        from src.cost import load_usage_from_checkpoint

        (tmp_path / "checkpoint.json").write_text(json.dumps({"log": [
            {"agent": "writer", "model": "tod-model", "prompt_tokens": 1_000_000,
             "completion_tokens": 1_000_000, "cached_prompt_tokens": 0,
             "timestamp": "2026-09-26T13:50:00"}]}), encoding="utf-8")
        rows = load_usage_from_checkpoint(str(tmp_path))
        assert summarize(rows, TOD).cost_usd == OFF_PEAK


class TestShippedDeepSeekRates:
    """The rates config.yaml ships, checked against how DeepSeek states them
    (https://api-docs.deepseek.com/quick_start/pricing, 2026-09-26)."""

    @staticmethod
    def _pricing() -> dict:
        import yaml

        cfg = yaml.safe_load(
            (Path(__file__).resolve().parent.parent / "config.yaml").read_text(
                encoding="utf-8"
            )
        )
        return load_pricing(cfg)

    def test_off_peak_is_half_of_peak(self) -> None:
        for model, rates in self._pricing().items():
            if "off_peak" not in rates:
                continue
            for key in ("input", "cached_input", "output"):
                assert rates["off_peak"][key] * 2 == rates[key], (model, key)

    def test_every_time_of_day_entry_has_a_readable_schedule(self) -> None:
        """An unreadable schedule would charge every call the peak rate."""
        from src.cost import rate_period

        for model, rates in self._pricing().items():
            if "off_peak" not in rates:
                continue
            probe = TokenUsage("a", model, "deepseek", 1, 1,
                               timestamp="2026-09-26T13:50:00")
            assert rate_period(probe, rates) == "off_peak", model

    def test_the_macos_study_is_priced_off_peak(self) -> None:
        """The 2026-09-26 macOS study: 232,105 prompt tokens (114,560 cached)
        and 42,175 completion tokens on deepseek-v4-pro, on a Saturday.
        The old flat rates said US$0.0538; the account balance fell by
        about US$0.20, which also paid for one call cut off by a stop."""
        u = TokenUsage("x", "deepseek-v4-pro", "deepseek", 232_105, 42_175,
                       114_560, timestamp="2026-09-26T13:50:00")
        pricing = self._pricing()
        rates = pricing["deepseek-v4-pro"]["off_peak"]
        expected = ((232_105 - 114_560) * rates["input"]
                    + 114_560 * rates["cached_input"]
                    + 42_175 * rates["output"]) / 1_000_000
        assert cost_usd(u, pricing) == expected
        assert 0.10 < expected < 0.20
