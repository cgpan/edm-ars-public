"""The review gate meters what it spends, while it spends it.

On the owner's Mac (round 3) the live cost sat at US$0.30015 for the
gate's 42 minutes and run_cost.json said US$0.300 over 15 calls; the
DeepSeek balance fell US$0.57. The gate's six LSAR reviews (42 calls) and
its revision call were never recorded. Every one of those calls now goes
to token_usage.jsonl as component "review" when the review (or the
revision) ends, and is announced as an llm.end event, the event a live
view already adds to its running cost, followed by a gate.cost event with
the gate's running total.

LSAR is faked here: no paid call is made.
"""
from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from src.cost import load_usage, load_usage_from_checkpoint, summarize, write_summary
from src.review_gate import ReviewGate

TOD = {
    "tod-model": {
        "input": 2.0, "cached_input": 0.5, "output": 4.0,
        "off_peak": {"input": 1.0, "cached_input": 0.25, "output": 2.0},
        "peak_windows_utc": {
            "days": ["mon", "tue", "wed", "thu", "fri"],
            "hours": ["01:00-04:00", "06:00-10:00"],
        },
    },
    "reviser-model": {"input": 1.0, "cached_input": 1.0, "output": 1.0},
}
PEAK = 2.0 + 4.0


def _call(**extra: Any) -> dict:
    return {"provider": "deepseek", "model": "tod-model", "stage": "review",
            "prompt_tokens": 1_000_000, "completion_tokens": 1_000_000,
            "cached_prompt_tokens": 0, **extra}


def _usage_file(calls: list[dict]) -> dict:
    """LSAR's token_usage.json (lsar/pipeline.py::_write_review_cost)."""
    return {
        "n_calls": len(calls),
        "prompt_tokens": sum(c["prompt_tokens"] for c in calls),
        "completion_tokens": sum(c["completion_tokens"] for c in calls),
        "cached_prompt_tokens": 0,
        "by_model": {"tod-model": {"n_calls": len(calls), "prompt_tokens": 0,
                                   "completion_tokens": 0}},
        "calls": calls,
    }


REPORT = {"scores": {"overall_score": 7.0, "recommendation": "Accept",
                     "dimensions": [{"name": "Novelty", "score": 7}]}}


def _install_fake_lsar(
    monkeypatch: pytest.MonkeyPatch, run: Callable[..., Any], leftover: list
) -> None:
    """lsar.pipeline.LSARPipeline and lsar.utils.llm_client.drain_usage_log."""
    pkg = types.ModuleType("lsar")
    pkg.__path__ = []  # type: ignore[attr-defined]
    utils = types.ModuleType("lsar.utils")
    utils.__path__ = []  # type: ignore[attr-defined]
    client = types.ModuleType("lsar.utils.llm_client")

    def drain_usage_log() -> list:
        out = list(leftover)
        leftover.clear()
        return out

    client.drain_usage_log = drain_usage_log  # type: ignore[attr-defined]
    pipeline_mod = types.ModuleType("lsar.pipeline")

    class LSARPipeline:
        def __init__(self, config_path: Any = None) -> None:
            pass

        def run(self, **kw: Any) -> Any:
            return run(**kw)

    pipeline_mod.LSARPipeline = LSARPipeline  # type: ignore[attr-defined]
    for name, mod in (("lsar", pkg), ("lsar.utils", utils),
                      ("lsar.utils.llm_client", client),
                      ("lsar.pipeline", pipeline_mod)):
        monkeypatch.setitem(sys.modules, name, mod)


def _gate(tmp_path: Path, **review_gate: Any) -> tuple[ReviewGate, list]:
    lsar_dir = tmp_path / "LSAR"
    lsar_dir.mkdir(exist_ok=True)
    cfg = {
        "llm_provider": "deepseek",
        "deepseek": {"models": {"revision_writer": "reviser-model"}},
        "pricing": {"per_million_tokens": TOD},
        "review_gate": {"pass_threshold": 5.5, "dimension_floor": 3,
                        "max_cycles": 1, "median_samples": 1,
                        "lsar_project_path": str(lsar_dir), **review_gate},
    }
    gate = ReviewGate(cfg, str(tmp_path), log_fn=lambda *_: None)
    gate.lsar_project_path = lsar_dir
    seen: list = []
    gate.event_fn = lambda t, **kw: seen.append((t, kw))
    return gate, seen


def _review_writes(calls: list[dict]) -> Callable[..., Any]:
    def run(**kw: Any) -> Any:
        out = Path(kw["output_dir"])
        (out / "token_usage.json").write_text(json.dumps(_usage_file(calls)),
                                              encoding="utf-8")
        return "# report", json.loads(json.dumps(REPORT))
    return run


class TestLsarReviewsAreMetered:
    def test_a_review_s_calls_reach_token_usage_and_the_live_view(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_lsar(monkeypatch, _review_writes([_call(), _call()]), [])
        gate, seen = _gate(tmp_path)

        assert gate.run_lsar(tmp_path / "paper.pdf", cycle=1) is not None

        rows = load_usage(str(tmp_path))
        assert [(r.agent, r.component) for r in rows] == [("LSAR", "review")] * 2
        window = json.loads((tmp_path / "lsar_review" / "cycle_1" /
                             "review_window.json").read_text(encoding="utf-8"))
        assert window["started_utc"] <= window["ended_utc"]
        ends = [kw for t, kw in seen if t == "llm.end"]
        assert len(ends) == 2
        assert all(kw["cost_usd"] is not None and kw["component"] == "review"
                   for kw in ends)
        [cost] = [kw for t, kw in seen if t == "gate.cost"]
        assert cost["n_calls"] == 2
        assert cost["cost_usd"] == pytest.approx(sum(kw["cost_usd"] for kw in ends))
        assert "US$" in cost["plain"]

    def test_lsar_s_own_timestamps_are_used(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 2026-09-28T07:30 is a Monday inside the peak window.
        stamped = [_call(timestamp="2026-09-28T07:30:00")]
        _install_fake_lsar(monkeypatch, _review_writes(stamped), [])
        gate, _ = _gate(tmp_path)
        gate.run_lsar(tmp_path / "paper.pdf", cycle=1)
        [row] = load_usage(str(tmp_path))
        assert row.timestamp == "2026-09-28T07:30:00" and row.time_source is None
        assert summarize([row], TOD).cost_usd == PEAK

    def test_a_review_that_raised_still_counts_what_it_spent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """LSAR writes token_usage.json only after a review completes; a
        scoring failure leaves the calls in its in-process log."""
        leftover = [_call(), _call(), _call()]

        def run(**_kw: Any) -> Any:
            raise RuntimeError("scoring failed")

        _install_fake_lsar(monkeypatch, run, leftover)
        gate, seen = _gate(tmp_path)

        assert gate.run_lsar(tmp_path / "paper.pdf", cycle=1) is None
        assert len(load_usage(str(tmp_path))) == 3
        written = json.loads((tmp_path / "lsar_review" / "cycle_1" /
                              "token_usage.json").read_text(encoding="utf-8"))
        assert len(written["calls"]) == 3
        assert [t for t, _ in seen].count("llm.end") == 3

    def test_an_earlier_review_s_usage_in_the_same_folder_is_not_counted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        folder = tmp_path / "lsar_review" / "cycle_1"
        folder.mkdir(parents=True)
        stale = folder / "token_usage.json"
        stale.write_text(json.dumps(_usage_file([_call()] * 5)), encoding="utf-8")
        old = stale.stat().st_mtime - 3600
        os.utime(stale, (old, old))

        def run(**_kw: Any) -> Any:
            raise RuntimeError("stage 1 failed before any call")

        _install_fake_lsar(monkeypatch, run, [])
        gate, _ = _gate(tmp_path)
        gate.run_lsar(tmp_path / "paper.pdf", cycle=1)
        assert load_usage(str(tmp_path)) == []

    def test_the_gate_summary_and_run_cost_carry_the_review(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_lsar(monkeypatch, _review_writes([_call(), _call()]), [])
        gate, _ = _gate(tmp_path)
        pdf = tmp_path / "paper_for_review.pdf"
        pdf.write_bytes(b"%PDF-1.5 stub")
        gate.prepare_pdf = lambda *_a, **_k: pdf  # type: ignore[method-assign]
        gate._honesty_blockers = lambda: []  # type: ignore[method-assign]

        summary = gate.run_gate()

        assert summary["cost"]["n_calls"] == 2
        assert [r["review_dir"] for r in summary["cost"]["reviews"]] == ["cycle_1"]
        payload = write_summary(str(tmp_path), gate.config)
        assert payload["by_component"]["review"]["n_calls"] == 2
        assert payload["review_cost_usd"] == summary["cost"]["cost_usd"]
        assert payload["pipeline_cost_usd"] == 0.0


class _Reviser:
    """OpenAI-compatible client whose reply carries provider usage."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **_kw: Any) -> Any:
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.text),
                                     finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=1_000_000, completion_tokens=1_000_000,
                                  prompt_cache_hit_tokens=0),
        )


class TestRevisionCallsAreMetered:
    def test_the_revision_call_is_a_review_row_and_an_llm_call(
        self, tmp_path: Path
    ) -> None:
        gate, seen = _gate(tmp_path)
        gate._llm_client = _Reviser("```latex\n\\documentclass{x}\\end{document}\n```")

        assert gate._call_revision_llm("prompt") is not None

        [row] = load_usage(str(tmp_path))
        assert (row.agent, row.model, row.component) == (
            "ReviewGate", "reviser-model", "review")
        assert row.timestamp
        types_seen = [t for t, _ in seen]
        assert types_seen.index("llm.start") < types_seen.index("llm.end")
        [end] = [kw for t, kw in seen if t == "llm.end"]
        assert end["cost_usd"] == 2.0


class TestTheCheckpointKeepsTheGateCalls:
    def test_the_orchestrator_copies_the_gate_rows_into_ctx_log(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.orchestrator import Orchestrator

        _install_fake_lsar(monkeypatch, _review_writes([_call(), _call()]), [])
        gate, _ = _gate(tmp_path)
        gate.run_lsar(tmp_path / "paper.pdf", cycle=1)
        fake = SimpleNamespace(ctx=SimpleNamespace(log=[]))

        Orchestrator._keep_gate_usage(fake, gate)  # type: ignore[arg-type]
        Orchestrator._keep_gate_usage(fake, None)  # type: ignore[arg-type]

        (tmp_path / "checkpoint.json").write_text(
            json.dumps({"log": fake.ctx.log}), encoding="utf-8")
        rows = load_usage_from_checkpoint(str(tmp_path))
        assert [r.component for r in rows] == ["review", "review"]
        assert [r.time_source for r in rows] == [
            r.time_source for r in load_usage(str(tmp_path))]
