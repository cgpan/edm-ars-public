"""V3.0 Phase 3b.7 / sub-phase A.1 — MiniMax-M2.7 provider smoke test.

The codebase already supports MiniMax via the Anthropic-SDK-compatible
endpoint at https://api.minimax.io/anthropic (see
src/agents/base.py::BaseAgent.__init__ when provider=='minimax'). This
test verifies the provider class is callable end-to-end with the model
string `MiniMax-M2.7` and that auth resolves from MINIMAX_API_KEY.

This is the ONLY LLM call permitted in Phase 3b.7 sub-phase A.
The hand-off A.3 acceptance gate specified ≤50 tokens, but
empirically MiniMax-M2.7 emits a "thinking" content block before
"text" (similar to Anthropic extended thinking). At max_tokens=50
the model exhausts its budget on the thinking block alone and
returns no text. We bump to 500 tokens here to accommodate the
thinking overhead — still bounded, still trivial. Documented as
F-MINIMAX-THINKING-OVERHEAD in the 3b.7 failure-mode catalog
(narrow exception #4 — provider runtime issue surfaced under load).
Marked as integration so it runs only with --run-integration.
"""
from __future__ import annotations

import os

import pytest


@pytest.mark.integration
def test_minimax_m27_smoke_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reach MiniMax-M2.7 via the project's BaseAgent abstraction; confirm
    a coherent response on a trivial prompt."""
    # The conftest autouse fixture installs a fake MINIMAX_API_KEY for
    # unit tests. For this integration test we need the real key from
    # .env, so load it and override the fake.
    from dotenv import load_dotenv

    # override=True is required because conftest's autouse fixture has
    # already set the fake MINIMAX_API_KEY by this point.
    load_dotenv(override=True)
    real_key = os.environ.get("MINIMAX_API_KEY")
    if not real_key or real_key.startswith("sk-minimax-fake-"):
        pytest.skip("MINIMAX_API_KEY not set in .env (real key required)")
    monkeypatch.setenv("MINIMAX_API_KEY", real_key)
    real_base = os.environ.get(
        "MINIMAX_BASE_URL", "https://api.minimax.io/anthropic"
    )
    monkeypatch.setenv("MINIMAX_BASE_URL", real_base)

    # Use the same provider construction path the orchestrator uses, so
    # this test exercises the real abstraction (not a private SDK call).
    from src.agents.base import BaseAgent

    class _SmokeAgent(BaseAgent):
        def run(self, **kwargs):  # type: ignore[override]
            return None

    config: dict = {
        "llm_provider": "minimax",
        "models": {"smokeagent": "claude-sonnet-4-6"},  # unused under minimax
        "minimax": {
            "base_url": "https://api.minimax.io/anthropic",
            "models": {"smokeagent": "MiniMax-M2.7"},
        },
        "paths": {"data_registry": "data_registry/", "agent_prompts": "agent_prompts/"},
        "sandbox": {"enabled": False},
    }

    class _FakeCtx:
        dataset_name = "hsls09_public"
        task_type = "prediction"
        output_dir = "/tmp"
        log: list = []

    agent = _SmokeAgent(
        context=_FakeCtx(),
        agent_name="SmokeAgent",
        config=config,
    )

    # Confirm the provider wiring resolved to MiniMax-M2.7.
    assert agent.model == "MiniMax-M2.7", (
        f"Expected model='MiniMax-M2.7', got {agent.model!r}"
    )

    # Live call. 500 tokens to accommodate MiniMax-M2.7 thinking overhead
    # (see module docstring). The exact text is not asserted — we only
    # require a non-empty string response, which proves the round trip
    # AND that the post-3b.7 base.py text-extraction handles thinking
    # content blocks correctly.
    response = agent.call_llm(
        "Reply with the single word OK and nothing else.",
        max_tokens=500,
    )
    assert isinstance(response, str)
    assert len(response.strip()) > 0, (
        "Empty response — likely the thinking-block extraction in "
        "BaseAgent.call_llm regressed; or the model's text block was "
        "empty (would indicate F-MINIMAX-THINKING-OVERHEAD with "
        "insufficient max_tokens)."
    )
