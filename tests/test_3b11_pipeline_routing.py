"""V3.0 Phase 3b.10 / §10.4 + §10.5 — 3b.11 config validation +
end-to-end mocked routing test.

Phase 3b.10.5 update: DeepSeek-V4-Pro replaced MiniMax-M2.7 as the
default provider for non-OpenAI stages.

Confirms that ``runs/configs/smoketest_3b11.yaml`` resolves to:
  - Analyst → openai / gpt-5.4 / 32000 max_tokens
  - Writer  → openai / gpt-5.4 / 32000 max_tokens
  - Critic  → deepseek / deepseek-v4-pro / 16000 max_tokens
  - PF      → deepseek / deepseek-v4-pro / 16000 max_tokens
  - DE      → deepseek / deepseek-v4-pro / 12000 max_tokens
  - Default fallback for any unenumerated stage → deepseek / 8000

End-to-end test: mock both anthropic.Anthropic and openai.OpenAI; build
real BaseAgent instances against the 3b.11 config; confirm each agent's
self.client / self.model / self.max_tokens reflect the per-stage config.
The DeepSeek path uses the openai SDK under the hood (DeepSeek's API
is OpenAI-compatible at https://api.deepseek.com).

No real LLM calls.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from src.agents.provider_resolver import (
    resolve_max_tokens_for_stage,
    resolve_provider_for_stage,
)


PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_3B11 = PROJECT_ROOT / "runs" / "configs" / "smoketest_3b11.yaml"


def _load_3b11_config() -> dict:
    with open(CONFIG_3B11, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# §10.4 — config file loadability + resolver round-trip
# ---------------------------------------------------------------------------


class TestConfig3b11LoadsAndResolves:
    def test_3b11_config_loads(self) -> None:
        config = _load_3b11_config()
        # Phase 3b.10.5 default switch: deepseek (was minimax).
        assert config["llm_provider"] == "deepseek"
        per_stage = config["per_stage_providers"]
        assert per_stage["analyst"]["provider"] == "openai"
        assert per_stage["analyst"]["model"] == "gpt-5.4"
        assert per_stage["writer"]["provider"] == "openai"
        assert per_stage["writer"]["model"] == "gpt-5.4"
        per_max = config["per_stage_max_tokens"]
        assert per_max["analyst"] == 32000
        assert per_max["writer"] == 32000
        assert per_max["critic"] == 16000
        assert per_max["problem_formulator"] == 16000
        assert per_max["data_engineer"] == 12000

    def test_3b11_provider_resolution_per_stage(self) -> None:
        config = _load_3b11_config()
        analyst = resolve_provider_for_stage("analyst", config)
        writer = resolve_provider_for_stage("writer", config)
        critic = resolve_provider_for_stage("critic", config)
        de = resolve_provider_for_stage("data_engineer", config)
        pf = resolve_provider_for_stage("problem_formulator", config)
        outline = resolve_provider_for_stage("outline_agent", config)

        assert (analyst.name, analyst.model) == ("openai", "gpt-5.4")
        assert (writer.name, writer.model) == ("openai", "gpt-5.4")
        # Phase 3b.10.5: non-OpenAI stages now route to DeepSeek-V4-Pro.
        assert (critic.name, critic.model) == ("deepseek", "deepseek-v4-pro")
        assert (de.name, de.model) == ("deepseek", "deepseek-v4-pro")
        assert (pf.name, pf.model) == ("deepseek", "deepseek-v4-pro")
        # outline_agent has no per-stage override; falls back to legacy
        # config.deepseek.models lookup.
        assert outline.name == "deepseek"

    def test_3b11_max_tokens_resolution_per_stage(self) -> None:
        config = _load_3b11_config()
        assert resolve_max_tokens_for_stage("analyst", config) == 32000
        assert resolve_max_tokens_for_stage("writer", config) == 32000
        assert resolve_max_tokens_for_stage("critic", config) == 16000
        assert resolve_max_tokens_for_stage("problem_formulator", config) == 16000
        assert resolve_max_tokens_for_stage("data_engineer", config) == 12000
        # Unenumerated stage → default_max_tokens.
        assert resolve_max_tokens_for_stage("outline_agent", config) == 8000


# ---------------------------------------------------------------------------
# §10.5 — end-to-end mocked routing through BaseAgent
# ---------------------------------------------------------------------------


def _make_agent_with_3b11(
    monkeypatch: pytest.MonkeyPatch,
    agent_name: str,
) -> Any:
    """Build a BaseAgent subclass instance under the 3b.11 config.

    Mocks:
      - anthropic.Anthropic + openai.OpenAI (so client construction
        doesn't reach out to the real APIs).
      - load_prompt (so we don't depend on agent_prompts/*.yaml).
      - sandbox.create_executor.
    """
    from src.agents.base import BaseAgent

    class _StubAgent(BaseAgent):
        def run(self, **kwargs):  # type: ignore[override]
            return None

    config = _load_3b11_config()

    # The conftest autouse fixture sets fake ANTHROPIC + MINIMAX keys;
    # OPENAI_API_KEY and DEEPSEEK_API_KEY aren't covered there.
    # Set fakes for both. (DeepSeek-API-key requirement landed in
    # 3b.10.5 when DeepSeek replaced MiniMax as the default for
    # non-OpenAI stages.)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-openai-for-3b10-routing-test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake-deepseek-for-3b10-routing-test")

    monkeypatch.setattr(
        "src.agents.base.load_prompt",
        lambda name, cfg, **kwargs: {
            "system_prompt": f"You are {name}.",
            "temperature": 0.0,
        },
    )
    monkeypatch.setattr("src.sandbox.create_executor", lambda cfg: object())

    class _FakeCtx:
        dataset_name = "hsls09_public"
        task_type = "causal_soo"
        output_dir = None
        log: list = []

    # Patch BOTH provider clients so init succeeds for any provider
    # the resolver picks.
    with patch("anthropic.Anthropic") as anthropic_cls, patch(
        "openai.OpenAI"
    ) as openai_cls:
        agent = _StubAgent(
            context=_FakeCtx(),
            agent_name=agent_name,
            config=config,
        )
        # Tag the mocks so tests can identify which class was used.
        anthropic_cls.return_value._provider_tag = "anthropic"
        openai_cls.return_value._provider_tag = "openai"
    return agent


class TestPipelineRoutingUnder3b11Config:
    """Each stage's BaseAgent picks up the right provider + model +
    max_tokens under the 3b.11 config."""

    # NOTE on agent_name: BaseAgent.__init__ uses
    # ``agent_key = agent_name.lower().replace(" ", "_")``, which assumes
    # the orchestrator passes either lowercase+underscored names directly
    # (orchestrator.py does — see "problem_formulator", "data_engineer",
    # etc.) or names with spaces (legacy convention). It does NOT
    # camel-case-split, so passing "ProblemFormulator" here would map to
    # "problemformulator" — a key that is NOT in per_stage_providers /
    # per_stage_max_tokens / minimax.models. Tests below pass the same
    # canonical names the orchestrator passes.

    def test_analyst_routes_to_openai_gpt54_with_32k_tokens(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _make_agent_with_3b11(monkeypatch, agent_name="analyst")
        assert agent._provider == "openai"
        assert agent.model == "gpt-5.4"
        assert agent.max_tokens == 32000

    def test_writer_routes_to_openai_gpt54_with_32k_tokens(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _make_agent_with_3b11(monkeypatch, agent_name="writer")
        assert agent._provider == "openai"
        assert agent.model == "gpt-5.4"
        assert agent.max_tokens == 32000

    def test_critic_routes_to_deepseek_v4_pro_with_16k_tokens(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _make_agent_with_3b11(monkeypatch, agent_name="critic")
        assert agent._provider == "deepseek"
        assert agent.model == "deepseek-v4-pro"
        assert agent.max_tokens == 16000

    def test_problem_formulator_routes_to_deepseek_v4_pro_with_16k_tokens(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _make_agent_with_3b11(
            monkeypatch, agent_name="problem_formulator"
        )
        assert agent._provider == "deepseek"
        assert agent.model == "deepseek-v4-pro"
        assert agent.max_tokens == 16000

    def test_data_engineer_routes_to_deepseek_v4_pro_with_12k_tokens(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _make_agent_with_3b11(monkeypatch, agent_name="data_engineer")
        assert agent._provider == "deepseek"
        assert agent.model == "deepseek-v4-pro"
        assert agent.max_tokens == 12000


# ---------------------------------------------------------------------------
# Backward compat — 3b.9 config still routes the legacy single-provider way
# ---------------------------------------------------------------------------


class TestBackwardCompat3b9ConfigStillWorks:
    def test_3b9_config_resolves_all_stages_to_minimax(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The post-3b.10 BaseAgent + resolver must keep working under
        3b.9's config shape (legacy llm_provider + per-provider models;
        no per_stage_providers / per_stage_max_tokens)."""
        from src.agents.base import BaseAgent

        class _StubAgent(BaseAgent):
            def run(self, **kwargs):  # type: ignore[override]
                return None

        # Mirror 3b.9's config shape exactly (no per-stage overrides).
        config = {
            "llm_provider": "minimax",
            "models": {
                "stubagent": "claude-sonnet-4-6",
                "problem_formulator": "claude-sonnet-4-6",
                "data_engineer": "claude-sonnet-4-6",
                "analyst": "claude-sonnet-4-6",
                "critic": "claude-opus-4-6",
                "writer": "claude-sonnet-4-6",
            },
            "minimax": {
                "base_url": "https://api.minimax.io/anthropic",
                "models": {
                    "stubagent": "MiniMax-M2.7",
                    "analyst": "MiniMax-M2.7",
                    "writer": "MiniMax-M2.7",
                },
            },
            "paths": {
                "data_registry": "data_registry/",
                "agent_prompts": "agent_prompts/",
            },
            "sandbox": {"enabled": False},
        }
        monkeypatch.setattr(
            "src.agents.base.load_prompt",
            lambda name, cfg, **kwargs: {
                "system_prompt": "x", "temperature": 0.0, "max_tokens": 8192,
            },
        )
        monkeypatch.setattr("src.sandbox.create_executor", lambda cfg: object())

        class _FakeCtx:
            dataset_name = "hsls09_public"
            task_type = "causal_soo"
            output_dir = None
            log: list = []

        with patch("anthropic.Anthropic"):
            agent = _StubAgent(_FakeCtx(), "StubAgent", config)
        assert agent._provider == "minimax"
        assert agent.model == "MiniMax-M2.7"
        # No per-stage / no default max_tokens → falls through to
        # prompt_data['max_tokens']=8192 fallback.
        assert agent.max_tokens == 8192
