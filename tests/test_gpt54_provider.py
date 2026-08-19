"""V3.0 Phase 3b.10 / §10.1 — gpt-5.4 provider verification.

The codebase already supports OpenAI via BaseAgent's ``provider=='openai'``
branch (added in/around 3b.5; the 3b.5 manifest.json shows gpt-5.4 was
the active model for that run). 3b.7 added MiniMax additively without
removing OpenAI. This file verifies the OpenAI / gpt-5.4 path still
works post-3b.6 numpy/scipy/statsmodels reshuffle and post-3b.7
MiniMax-thinking-mode handling.

These tests are integration-marked: the live smoke call is the only
LLM call permitted in 3b.10 sub-wave 1. Capped at 50 tokens per the
hand-off acceptance criterion.
"""
from __future__ import annotations

import os

import pytest


# ---------------------------------------------------------------------------
# Static checks (no LLM calls; always run)
# ---------------------------------------------------------------------------


class TestOpenAIProviderClassPresent:
    def test_openai_provider_branch_exists_in_base_agent(self) -> None:
        """BaseAgent.__init__ has a provider=='openai' branch."""
        from src.agents import base as base_module

        # Pull the source so we can grep for the branch literal — easier
        # than reflecting over the constructor.
        import inspect

        source = inspect.getsource(base_module.BaseAgent)
        assert 'provider == "openai"' in source or "elif provider == 'openai'" in source, (
            "BaseAgent.__init__ should retain the 'openai' provider branch "
            "(added in 3b.5; preserved in 3b.7 additive MiniMax integration)."
        )

    def test_openai_sdk_importable(self) -> None:
        """If the OpenAI SDK isn't installed, the provider class won't
        construct. Surface this early."""
        try:
            import openai  # type: ignore[import-not-found]  # noqa: F401
        except ImportError:
            pytest.fail(
                "openai SDK not installed in current environment. Phase "
                "3b.10's gpt-5.4 provider needs it. "
                "pip install openai>=1.40 should resolve this."
            )


# ---------------------------------------------------------------------------
# Live smoke test (integration; ≤50 tokens)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_gpt54_smoke_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reach gpt-5.4 via the project's BaseAgent abstraction; confirm a
    coherent response on a trivial prompt.

    This is the only LLM call permitted in Phase 3b.10 sub-wave 1.
    Capped at 50 tokens per the hand-off § A.3 acceptance gate.
    """
    # The conftest autouse fixture installs a fake OPENAI_API_KEY for
    # unit tests. For this integration test we need the real key from
    # .env, so load it and override the fake.
    from dotenv import load_dotenv

    load_dotenv(override=True)
    real_key = os.environ.get("OPENAI_API_KEY")
    if not real_key or real_key.startswith("sk-openai-fake-") or real_key.startswith("sk-fake-"):
        pytest.skip("OPENAI_API_KEY not set in .env (real key required)")
    monkeypatch.setenv("OPENAI_API_KEY", real_key)

    from src.agents.base import BaseAgent

    class _SmokeAgent(BaseAgent):
        def run(self, **kwargs):  # type: ignore[override]
            return None

    config: dict = {
        "llm_provider": "openai",
        "models": {"smokeagent": "claude-sonnet-4-6"},
        "openai": {
            "models": {"smokeagent": "gpt-5.4"},
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
    assert agent.model == "gpt-5.4", (
        f"Expected model='gpt-5.4', got {agent.model!r}"
    )

    response = agent.call_llm(
        "Reply with the single word OK and nothing else.",
        max_tokens=50,
    )
    assert isinstance(response, str)
    assert len(response.strip()) > 0, (
        "Empty response from gpt-5.4 — check OpenAI SDK + auth."
    )
