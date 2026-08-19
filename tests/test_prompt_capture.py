"""V3.0 Phase 3b.7 / sub-phase A.2 — per-stage rendered-prompt capture.

Verifies that ``BaseAgent.call_llm`` writes the rendered system prompt,
user message, and raw response to disk so 3b.7's report can cite exact
prompt content per stage. The capture is best-effort — disk-IO failures
must not break the LLM call.

Layout (per the 3b.7 hand-off):

    {output_dir}/prompts/{agent_name}/cycle_{N}/rendered_prompt.txt
    {output_dir}/prompts/{agent_name}/cycle_{N}/response_raw.txt

These tests do NOT make real LLM calls — anthropic.Anthropic is patched
so ``BaseAgent.call_llm`` returns a synthetic response.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


_FAKE_CONFIG: dict = {
    "llm_provider": "minimax",
    "models": {"stubagent": "claude-sonnet-4-6"},
    "minimax": {
        "base_url": "https://example.invalid/anthropic",
        "models": {"stubagent": "MiniMax-stub"},
    },
    "paths": {"data_registry": "data_registry/", "agent_prompts": "agent_prompts/"},
    "sandbox": {"enabled": False},
}


class _FakeCtx:
    def __init__(self, output_dir: str) -> None:
        self.dataset_name = "hsls09_public"
        self.task_type = "prediction"
        self.output_dir = output_dir
        self.revision_cycle = 0
        self.log: list = []


def _make_agent(monkeypatch: pytest.MonkeyPatch, output_dir: str) -> Any:
    """Build a minimal BaseAgent with a stubbed prompt loader + executor."""
    from src.agents.base import BaseAgent

    class _StubAgent(BaseAgent):
        def run(self, **kwargs):  # type: ignore[override]
            return None

    monkeypatch.setattr(
        "src.agents.base.load_prompt",
        lambda name, config, **kwargs: {
            "system_prompt": "System body for capture test.",
            "temperature": 0.0,
        },
    )
    monkeypatch.setattr("src.sandbox.create_executor", lambda config: object())

    with patch("anthropic.Anthropic"):
        agent = _StubAgent(
            context=_FakeCtx(output_dir=output_dir),
            agent_name="StubAgent",
            config=_FAKE_CONFIG,
        )
    return agent


# ---------------------------------------------------------------------------
# Capture path resolution
# ---------------------------------------------------------------------------


def test_capture_dir_layout_matches_handoff_spec(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    agent = _make_agent(monkeypatch, str(tmp_path))
    capture_dir = agent._capture_prompt_dir()
    assert capture_dir is not None
    expected = tmp_path / "prompts" / "stubagent" / "cycle_0"
    assert Path(capture_dir).resolve() == expected.resolve()
    assert os.path.isdir(capture_dir)


def test_capture_dir_uses_revision_cycle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    agent = _make_agent(monkeypatch, str(tmp_path))
    agent.ctx.revision_cycle = 2
    capture_dir = agent._capture_prompt_dir()
    assert capture_dir is not None
    assert Path(capture_dir).name == "cycle_2"


def test_capture_dir_returns_none_when_no_output_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    agent = _make_agent(monkeypatch, str(tmp_path))
    agent.ctx.output_dir = None  # type: ignore[assignment]
    assert agent._capture_prompt_dir() is None


# ---------------------------------------------------------------------------
# call_llm writes prompt + response to disk
# ---------------------------------------------------------------------------


def _stub_anthropic_stream_response(text: str) -> Any:
    """Build a mock that mimics anthropic.Anthropic().messages.stream() ctx.

    Returns a context manager whose get_final_message() yields a message
    with content blocks of type 'text' (plus a 'thinking' block to
    exercise the post-3b.7 thinking-aware text extraction).
    """
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = text

    thinking_block = MagicMock()
    thinking_block.type = "thinking"
    thinking_block.text = None

    final_message = MagicMock()
    final_message.content = [thinking_block, text_block]
    final_message.usage = MagicMock(input_tokens=10, output_tokens=20)

    stream_ctx = MagicMock()
    stream_ctx.get_final_message.return_value = final_message
    stream_ctx.__enter__.return_value = stream_ctx
    stream_ctx.__exit__.return_value = False
    return stream_ctx


def test_call_llm_writes_rendered_prompt_and_response(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    agent = _make_agent(monkeypatch, str(tmp_path))
    # Replace the client with one whose .messages.stream returns the stub.
    agent.client = MagicMock()
    agent.client.messages.stream.return_value = _stub_anthropic_stream_response(
        "stubbed response body"
    )

    response = agent.call_llm("user prompt body for capture", max_tokens=100)
    assert response == "stubbed response body"

    capture_dir = tmp_path / "prompts" / "stubagent" / "cycle_0"
    rendered = (capture_dir / "rendered_prompt.txt").read_text(encoding="utf-8")
    raw = (capture_dir / "response_raw.txt").read_text(encoding="utf-8")

    # Rendered prompt must include both the system prompt and the user message
    # in their canonical sections.
    assert "=== SYSTEM PROMPT ===" in rendered
    assert "System body for capture test." in rendered
    assert "=== USER MESSAGE ===" in rendered
    assert "user prompt body for capture" in rendered

    # Response capture is the raw text returned to the caller.
    assert "stubbed response body" in raw


def test_call_llm_appends_on_second_call_in_same_cycle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Multi-branch / retry calls within a cycle should append rather than
    silently clobber the prior capture."""
    agent = _make_agent(monkeypatch, str(tmp_path))
    agent.client = MagicMock()
    agent.client.messages.stream.side_effect = [
        _stub_anthropic_stream_response("first response"),
        _stub_anthropic_stream_response("second response"),
    ]

    agent.call_llm("first user message", max_tokens=50)
    agent.call_llm("second user message", max_tokens=50)

    capture_dir = tmp_path / "prompts" / "stubagent" / "cycle_0"
    rendered = (capture_dir / "rendered_prompt.txt").read_text(encoding="utf-8")
    raw = (capture_dir / "response_raw.txt").read_text(encoding="utf-8")

    assert "first user message" in rendered
    assert "second user message" in rendered
    assert "first response" in raw
    assert "second response" in raw
    # Marker between calls.
    assert "--- additional call within same cycle ---" in rendered


def test_call_llm_does_not_crash_when_capture_dir_unwritable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Capture is best-effort. Disk-IO failures must NOT break the LLM call."""
    agent = _make_agent(monkeypatch, str(tmp_path))
    agent.client = MagicMock()
    agent.client.messages.stream.return_value = _stub_anthropic_stream_response(
        "still got the response"
    )

    # Force the capture writer to fail.
    monkeypatch.setattr(
        agent, "_capture_prompt_dir", lambda: tmp_path / "does_not_exist_yet" / "x"
    )
    # Also mock _write_prompt_capture / _write_response_capture to raise.
    monkeypatch.setattr(
        agent, "_write_prompt_capture",
        lambda *a, **kw: (_ for _ in ()).throw(OSError("disk full")),
    )

    # Without the broad except on capture writes this would crash.
    response = agent.call_llm("hello", max_tokens=50)
    assert response == "still got the response"
