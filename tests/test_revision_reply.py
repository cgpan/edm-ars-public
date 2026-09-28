"""How the review gate asks for a revision, and what it keeps of the reply.

The Mac study (round 3) logged "Could not extract LaTeX from LLM response;
keeping original" and nothing else: the reply was not saved, and how it
ended was not asked. The gate's DeepSeek request was also the only
DeepSeek request in the pipeline sent without ``thinking: disabled``, so
the model's reasoning, billed as output and counted against max_tokens,
shared the 16,000-token budget with the manuscript it had to write back.

The client is a local stub throughout: no paid call is made.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import pytest

from src.review_gate import ReviewGate


class _Chat:
    """OpenAI-compatible stub replying from a list of (text, finish_reason)."""

    def __init__(self, replies: list[tuple[str, Optional[str]]]) -> None:
        self.replies = list(replies)
        self.requests: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        text, finish = self.replies.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text),
                                     finish_reason=finish)],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20),
        )


class _Stream:
    def __init__(self, final: Any) -> None:
        self.final = final

    def __enter__(self) -> "_Stream":
        return self

    def __exit__(self, *_a: Any) -> None:
        return None

    def get_final_message(self) -> Any:
        return self.final


class _Messages:
    """Anthropic-SDK stub (messages.stream)."""

    def __init__(self, replies: list[tuple[str, str]]) -> None:
        self.replies = list(replies)
        self.requests: list[dict] = []
        self.messages = SimpleNamespace(stream=self._stream)

    def _stream(self, **kwargs: Any) -> _Stream:
        self.requests.append(kwargs)
        text, stop = self.replies.pop(0)
        return _Stream(SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)],
            stop_reason=stop,
            usage=SimpleNamespace(input_tokens=10, output_tokens=20),
        ))


def _gate(tmp_path: Path, provider: str = "deepseek", **review_gate: Any) -> ReviewGate:
    cfg = {
        "llm_provider": "deepseek",
        "deepseek": {"models": {"revision_writer": "deepseek-v4-pro"}},
        "review_gate": {"revision_max_tokens": 1000, **review_gate},
    }
    gate = ReviewGate(cfg, str(tmp_path), log_fn=lambda *_: None)
    gate._llm_provider = provider
    return gate


class TestTheRequest:
    def test_deepseek_is_asked_without_thinking(self, tmp_path: Path) -> None:
        gate = _gate(tmp_path)
        gate._llm_client = client = _Chat([("reply", "stop")])
        assert gate._call_revision_llm("prompt") == "reply"
        [req] = client.requests
        assert req["extra_body"] == {"thinking": {"type": "disabled"}}
        assert req["max_tokens"] == 1000

    def test_openai_gets_max_completion_tokens(self, tmp_path: Path) -> None:
        gate = _gate(tmp_path, provider="openai")
        gate._llm_client = client = _Chat([("reply", "stop")])
        gate._call_revision_llm("prompt")
        [req] = client.requests
        assert req["max_completion_tokens"] == 1000
        assert "max_tokens" not in req and "extra_body" not in req


class TestACutOffReply:
    def test_is_asked_for_once_more_with_a_larger_budget(self, tmp_path: Path) -> None:
        gate = _gate(tmp_path)
        gate._llm_client = client = _Chat([("```latex\n\\documentclass", "length"),
                                           ("complete", "stop")])
        assert gate._call_revision_llm("prompt") == "complete"
        assert [r["max_tokens"] for r in client.requests] == [1000, 2000]
        assert gate._last_reply["retried"] is True
        assert gate._last_reply["truncated"] is False
        assert gate._last_reply["max_tokens"] == 2000

    def test_is_retried_only_once(self, tmp_path: Path) -> None:
        gate = _gate(tmp_path, revision_retry_max_tokens=3000)
        gate._llm_client = client = _Chat([("part", "length"), ("longer part", "length")])
        assert gate._call_revision_llm("prompt") == "longer part"
        assert [r["max_tokens"] for r in client.requests] == [1000, 3000]
        assert gate._last_reply["truncated"] is True

    def test_a_complete_reply_is_not_retried(self, tmp_path: Path) -> None:
        gate = _gate(tmp_path)
        gate._llm_client = client = _Chat([("done", "stop")])
        gate._call_revision_llm("prompt")
        assert len(client.requests) == 1
        assert gate._last_reply == {"finish_reason": "stop", "truncated": False,
                                    "retried": False, "max_tokens": 1000}

    def test_anthropic_max_tokens_stop_is_a_cut_off(self, tmp_path: Path) -> None:
        gate = _gate(tmp_path, provider="anthropic")
        gate._llm_client = client = _Messages([("part", "max_tokens"),
                                               ("whole", "end_turn")])
        assert gate._call_revision_llm("prompt") == "whole"
        assert [r["max_tokens"] for r in client.requests] == [1000, 2000]

    def test_a_failed_retry_keeps_the_first_reply(self, tmp_path: Path) -> None:
        gate = _gate(tmp_path)
        client = _Chat([("part", "length")])

        def create(**kwargs: Any) -> Any:
            if client.requests:
                raise RuntimeError("upstream 502")
            return client._create(**kwargs)

        client.chat.completions.create = create
        gate._llm_client = client
        assert gate._call_revision_llm("prompt") == "part"
        assert gate._last_reply["truncated"] is True


class TestTheRawReplyIsKept:
    def test_each_reply_is_saved_in_the_cycle_folder(self, tmp_path: Path) -> None:
        gate = _gate(tmp_path)
        gate._revision_dir = tmp_path / "lsar_review" / "cycle_1"
        gate._llm_client = _Chat([("first reply", "length"), ("second reply", "stop")])
        gate._call_revision_llm("prompt")
        folder = tmp_path / "lsar_review" / "cycle_1"
        assert (folder / "revision_raw.txt").read_text(encoding="utf-8") == "first reply"
        assert (folder / "revision_raw_retry.txt").read_text(
            encoding="utf-8") == "second reply"

    def test_run_gate_saves_the_reply_under_the_cycle_it_revises(
        self, tmp_path: Path
    ) -> None:
        gate = _gate(tmp_path, max_cycles=2, median_samples=1)
        gate.pass_threshold = 9.0
        (tmp_path / "paper.tex").write_text(
            "\\documentclass{article}\\begin{document}\n\\section{Introduction}\n"
            "Short.\n\\end{document}\n", encoding="utf-8")
        pdf = tmp_path / "paper_for_review.pdf"
        pdf.write_bytes(b"%PDF-1.5 stub")
        gate.prepare_pdf = lambda *_a, **_k: pdf  # type: ignore[method-assign]
        gate.run_lsar = lambda *_a, **_k: {  # type: ignore[method-assign]
            "scores": {"overall_score": 4.0, "recommendation": "Reject",
                       "dimensions": [{"name": "Novelty", "score": 4}]},
            "review": {}}
        gate._honesty_blockers = lambda: []  # type: ignore[method-assign]
        gate._compile_full_latex = lambda *_a: None  # type: ignore[method-assign]
        gate._llm_client = _Chat([("Sorry, I cannot help with that.", "stop")] * 2)

        gate.run_gate()

        raw = tmp_path / "lsar_review" / "cycle_1" / "revision_raw.txt"
        assert raw.read_text(encoding="utf-8") == "Sorry, I cannot help with that."
        assert gate._revision_dir is None


def test_a_failed_call_closes_its_llm_start(tmp_path: Path) -> None:
    gate = _gate(tmp_path)
    seen: list = []
    gate.event_fn = lambda t, **kw: seen.append((t, kw))
    client = _Chat([])

    def create(**_kwargs: Any) -> Any:
        raise RuntimeError("boom")

    client.chat.completions.create = create
    gate._llm_client = client
    assert gate._call_revision_llm("prompt") is None
    assert [t for t, _ in seen] == ["llm.start", "llm.end"]
    assert seen[1][1]["ok"] is False


@pytest.mark.parametrize("finish", [None, "stop", "end_turn", "content_filter"])
def test_other_endings_are_not_cut_offs(tmp_path: Path, finish: Any) -> None:
    gate = _gate(tmp_path)
    gate._llm_client = client = _Chat([("reply", finish)])
    gate._call_revision_llm("prompt")
    assert len(client.requests) == 1
