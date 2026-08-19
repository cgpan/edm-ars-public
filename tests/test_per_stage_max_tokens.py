"""V3.0 Phase 3b.10 / §10.2 — per-stage max_tokens routing.

The resolver in src/agents/provider_resolver.py exposes
``resolve_max_tokens_for_stage``. These tests pin its behavior with no
LLM calls.

Per the 3b.10 hand-off, the resolution chain is:
  1. config['per_stage_max_tokens'][<agent_key>]
  2. config['default_max_tokens']
  3. fallback (caller-supplied; BaseAgent passes prompt_data.max_tokens
     or 8192).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agents.provider_resolver import (
    ProviderConfigError,
    resolve_max_tokens_for_stage,
)


class TestResolveMaxTokensForStage:
    def test_default_max_tokens_used_when_no_per_stage_override(self) -> None:
        config: dict = {"default_max_tokens": 8000}
        assert resolve_max_tokens_for_stage("data_engineer", config) == 8000

    def test_per_stage_max_tokens_override_wins(self) -> None:
        config: dict = {
            "default_max_tokens": 8000,
            "per_stage_max_tokens": {"analyst": 32000, "writer": 32000},
        }
        assert resolve_max_tokens_for_stage("analyst", config) == 32000
        assert resolve_max_tokens_for_stage("writer", config) == 32000
        # Stage without an override falls back to default.
        assert resolve_max_tokens_for_stage("critic", config) == 8000

    def test_fallback_used_when_neither_per_stage_nor_default(self) -> None:
        # Empty config → caller-supplied fallback wins (BaseAgent passes
        # prompt_data.max_tokens here).
        config: dict = {}
        assert resolve_max_tokens_for_stage("analyst", config, fallback=4096) == 4096

    def test_default_overrides_fallback(self) -> None:
        config: dict = {"default_max_tokens": 16000}
        # config default beats caller fallback.
        assert resolve_max_tokens_for_stage("analyst", config, fallback=4096) == 16000

    def test_per_stage_overrides_default_and_fallback(self) -> None:
        config: dict = {
            "default_max_tokens": 16000,
            "per_stage_max_tokens": {"analyst": 32000},
        }
        assert resolve_max_tokens_for_stage("analyst", config, fallback=4096) == 32000

    def test_3b11_recommended_values(self) -> None:
        """The hand-off § 10.2.2 recommended-values check."""
        config: dict = {
            "default_max_tokens": 8000,
            "per_stage_max_tokens": {
                "analyst": 32000,
                "writer": 32000,
                "critic": 16000,
                "problem_formulator": 16000,
                "data_engineer": 12000,
            },
        }
        assert resolve_max_tokens_for_stage("analyst", config) == 32000
        assert resolve_max_tokens_for_stage("writer", config) == 32000
        assert resolve_max_tokens_for_stage("critic", config) == 16000
        assert resolve_max_tokens_for_stage("problem_formulator", config) == 16000
        assert resolve_max_tokens_for_stage("data_engineer", config) == 12000
        # Other stages (e.g., outline_agent) fall back to the default.
        assert resolve_max_tokens_for_stage("outline_agent", config) == 8000

    def test_negative_max_tokens_raises(self) -> None:
        with pytest.raises(ProviderConfigError, match="positive int"):
            resolve_max_tokens_for_stage(
                "analyst", {"per_stage_max_tokens": {"analyst": -1}}
            )

    def test_zero_max_tokens_raises(self) -> None:
        with pytest.raises(ProviderConfigError, match="positive int"):
            resolve_max_tokens_for_stage(
                "analyst", {"default_max_tokens": 0}
            )


# ---------------------------------------------------------------------------
# BaseAgent integration: resolved max_tokens reaches the call site
# ---------------------------------------------------------------------------


class TestMaxTokensFlowsThroughToBaseAgent:
    """BaseAgent.__init__ must use the resolver to compute self.max_tokens.
    The 3b.7 sub-phase A.2 prompt-capture instrumentation shows that
    BaseAgent.call_llm uses self.max_tokens as the default; per-stage
    config overrides should reach the call.
    """

    def test_base_agent_picks_up_per_stage_max_tokens(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.agents.base import BaseAgent

        class _StubAgent(BaseAgent):
            def run(self, **kwargs):  # type: ignore[override]
                return None

        # Patch out external dependencies so init doesn't make any LLM/IO.
        monkeypatch.setattr(
            "src.agents.base.load_prompt",
            lambda name, config, **kwargs: {
                "system_prompt": "x", "temperature": 0.0,
            },
        )
        monkeypatch.setattr("src.sandbox.create_executor", lambda config: object())

        config: dict = {
            "llm_provider": "minimax",
            "minimax": {"models": {"stubagent": "MiniMax-stub"}},
            "per_stage_max_tokens": {"stubagent": 32000},
            "paths": {
                "data_registry": "data_registry/",
                "agent_prompts": "agent_prompts/",
            },
            "sandbox": {"enabled": False},
        }

        class _FakeCtx:
            dataset_name = "hsls09_public"
            task_type = "prediction"
            output_dir = None
            log: list = []

        with patch("anthropic.Anthropic"):
            agent = _StubAgent(_FakeCtx(), "StubAgent", config)

        assert agent.max_tokens == 32000, (
            f"BaseAgent should pick up per_stage_max_tokens override; "
            f"got {agent.max_tokens}"
        )

    def test_base_agent_falls_back_to_prompt_yaml_max_tokens(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When neither per_stage_max_tokens nor default_max_tokens is
        set, BaseAgent uses the prompt YAML's max_tokens (legacy
        per-prompt default)."""
        from src.agents.base import BaseAgent

        class _StubAgent(BaseAgent):
            def run(self, **kwargs):  # type: ignore[override]
                return None

        monkeypatch.setattr(
            "src.agents.base.load_prompt",
            lambda name, config, **kwargs: {
                "system_prompt": "x", "temperature": 0.0, "max_tokens": 12345,
            },
        )
        monkeypatch.setattr("src.sandbox.create_executor", lambda config: object())

        config: dict = {
            "llm_provider": "minimax",
            "minimax": {"models": {"stubagent": "MiniMax-stub"}},
            "paths": {
                "data_registry": "data_registry/",
                "agent_prompts": "agent_prompts/",
            },
            "sandbox": {"enabled": False},
        }

        class _FakeCtx:
            dataset_name = "hsls09_public"
            task_type = "prediction"
            output_dir = None
            log: list = []

        with patch("anthropic.Anthropic"):
            agent = _StubAgent(_FakeCtx(), "StubAgent", config)

        # Resolver fallback chain: no per-stage, no default → fallback
        # which BaseAgent supplies as prompt_data['max_tokens'] (12345).
        assert agent.max_tokens == 12345
