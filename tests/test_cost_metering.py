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
