"""V3.0 Phase 3b.10.5 — DeepSeek-V4-Pro provider verification.

DeepSeek-V4-Pro replaces MiniMax-M2.7 as the project default for all
non-OpenAI stages going forward. The MiniMax branch is retained for
backward-compat with 3b.5 / 3b.7 / 3b.9 run artifacts.

DeepSeek's API is OpenAI-compatible at https://api.deepseek.com.
Key facts the integration relies on:
  - Env var: DEEPSEEK_API_KEY
  - Base URL: https://api.deepseek.com (OpenAI format)
  - Model: deepseek-v4-pro (current; deepseek-v4-flash also available)
  - Thinking mode: ENABLED by default in the API; DISABLED by the
    project's BaseAgent path via extra_body={"thinking": {"type":
    "disabled"}}. Avoids the F-3b9 thinking-block-overhead recurrence.

Smoke test is integration-marked: live call ≤50 tokens, runs only
under --run-integration.
"""
from __future__ import annotations

import os

import pytest


# ---------------------------------------------------------------------------
# Static checks (no LLM calls; always run)
# ---------------------------------------------------------------------------


class TestDeepSeekProviderClassPresent:
    def test_deepseek_provider_branch_exists_in_base_agent(self) -> None:
        """BaseAgent.__init__ has a provider=='deepseek' branch."""
        from src.agents import base as base_module
        import inspect

        source = inspect.getsource(base_module.BaseAgent)
        assert (
            'provider == "deepseek"' in source
            or "elif provider == 'deepseek'" in source
        ), (
            "BaseAgent.__init__ should have the 'deepseek' provider "
            "branch added in 3b.10.5."
        )

    def test_deepseek_in_known_providers(self) -> None:
        from src.agents.provider_resolver import _KNOWN_PROVIDERS
        assert "deepseek" in _KNOWN_PROVIDERS

    def test_deepseek_call_path_disables_thinking_mode(self) -> None:
        """The deepseek branch in call_llm passes
        extra_body={'thinking': {'type': 'disabled'}}. This is the
        F-3b9 prevention rule."""
        from src.agents import base as base_module
        import inspect

        source = inspect.getsource(base_module.BaseAgent.call_llm)
        assert 'thinking' in source and 'disabled' in source, (
            "BaseAgent.call_llm's deepseek branch must pass "
            "extra_body={'thinking': {'type': 'disabled'}} to avoid "
            "thinking-block overhead (F-3b9 recurrence prevention)."
        )


# ---------------------------------------------------------------------------
# Resolver round-trip — DeepSeek registered, accepts deepseek-v4-pro
# ---------------------------------------------------------------------------


class TestDeepSeekResolverIntegration:
    def test_resolve_provider_for_deepseek(self) -> None:
        from src.agents.provider_resolver import resolve_provider_for_stage

        config: dict = {
            "llm_provider": "deepseek",
            "deepseek": {
                "models": {
                    "analyst": "deepseek-v4-pro",
                    "data_engineer": "deepseek-v4-pro",
                },
            },
        }
        cfg = resolve_provider_for_stage("analyst", config)
        assert cfg.name == "deepseek"
        assert cfg.model == "deepseek-v4-pro"

    def test_per_stage_override_to_deepseek(self) -> None:
        """Stages can be routed to DeepSeek via per_stage_providers
        (e.g., 'critic + DE on DeepSeek; Analyst + Writer on OpenAI')."""
        from src.agents.provider_resolver import resolve_provider_for_stage

        config: dict = {
            "llm_provider": "openai",
            "openai": {"models": {"analyst": "gpt-5.4"}},
            "per_stage_providers": {
                "critic": {"provider": "deepseek", "model": "deepseek-v4-pro"},
                "data_engineer": {
                    "provider": "deepseek", "model": "deepseek-v4-pro",
                },
            },
        }
        critic = resolve_provider_for_stage("critic", config)
        assert critic.name == "deepseek"
        assert critic.model == "deepseek-v4-pro"
        analyst = resolve_provider_for_stage("analyst", config)
        assert analyst.name == "openai"


# ---------------------------------------------------------------------------
# Live smoke test (integration; ≤50 tokens)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_deepseek_v4_pro_smoke_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reach DeepSeek-V4-Pro via the project's BaseAgent abstraction;
    confirm a coherent response on a trivial prompt.

    This is the only LLM call permitted in Phase 3b.10.5. Capped at 50
    tokens per the project's smoke-test budget convention.
    """
    from dotenv import load_dotenv

    load_dotenv(override=True)
    real_key = os.environ.get("DEEPSEEK_API_KEY")
    if not real_key or real_key.startswith("sk-fake-"):
        pytest.skip("DEEPSEEK_API_KEY not set in .env (real key required)")
    monkeypatch.setenv("DEEPSEEK_API_KEY", real_key)

    from src.agents.base import BaseAgent

    class _SmokeAgent(BaseAgent):
        def run(self, **kwargs):  # type: ignore[override]
            return None

    config: dict = {
        "llm_provider": "deepseek",
        "models": {"smokeagent": "claude-sonnet-4-6"},
        "deepseek": {
            "models": {"smokeagent": "deepseek-v4-pro"},
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
    assert agent.model == "deepseek-v4-pro", (
        f"Expected model='deepseek-v4-pro', got {agent.model!r}"
    )
    assert agent._provider == "deepseek"

    response = agent.call_llm(
        "Reply with the single word OK and nothing else.",
        max_tokens=50,
    )
    assert isinstance(response, str)
    assert len(response.strip()) > 0, (
        "Empty response from deepseek-v4-pro — check thinking mode is "
        "disabled (otherwise the model may exhaust 50-token budget on "
        "thinking content alone, the same F-3b9 pattern)."
    )
