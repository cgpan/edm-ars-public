"""V3.0 Phase 3b.10 / §10.1.3 — per-stage provider routing.

The resolver (``src/agents/provider_resolver.py``) is a pure function of
the run config. These tests pin its behavior with no LLM calls and no
agent-construction side effects.
"""
from __future__ import annotations

import pytest

from src.agents.provider_resolver import (
    ProviderConfig,
    ProviderConfigError,
    resolve_provider_for_stage,
)


# ---------------------------------------------------------------------------
# Per-stage override + fall-back behavior
# ---------------------------------------------------------------------------


class TestResolveProviderForStage:
    def test_default_provider_used_when_no_per_stage_override(self) -> None:
        config: dict = {
            "llm_provider": "minimax",
            "minimax": {"models": {"analyst": "MiniMax-M2.7"}},
        }
        cfg = resolve_provider_for_stage("analyst", config)
        assert cfg.name == "minimax"
        assert cfg.model == "MiniMax-M2.7"

    def test_per_stage_override_wins_over_default(self) -> None:
        config: dict = {
            "llm_provider": "minimax",
            "minimax": {"models": {"analyst": "MiniMax-M2.7"}},
            "per_stage_providers": {
                "analyst": {"provider": "openai", "model": "gpt-5.4"},
            },
        }
        # Analyst gets the override.
        analyst_cfg = resolve_provider_for_stage("analyst", config)
        assert analyst_cfg.name == "openai"
        assert analyst_cfg.model == "gpt-5.4"
        # Other stages fall back to the default.
        de_cfg = resolve_provider_for_stage("data_engineer", config)
        assert de_cfg.name == "minimax"

    def test_per_stage_override_for_multiple_stages(self) -> None:
        """3b.11's intended config: Analyst + Writer → OpenAI; rest → MiniMax."""
        config: dict = {
            "llm_provider": "minimax",
            "minimax": {"models": {}},
            "per_stage_providers": {
                "analyst": {"provider": "openai", "model": "gpt-5.4"},
                "writer": {"provider": "openai", "model": "gpt-5.4"},
            },
        }
        for routed_stage in ("analyst", "writer"):
            cfg = resolve_provider_for_stage(routed_stage, config)
            assert cfg.name == "openai"
            assert cfg.model == "gpt-5.4"
        for default_stage in ("problem_formulator", "data_engineer", "critic"):
            cfg = resolve_provider_for_stage(default_stage, config)
            assert cfg.name == "minimax"

    def test_unknown_provider_in_override_raises(self) -> None:
        config: dict = {
            "per_stage_providers": {
                "analyst": {"provider": "nonexistent", "model": "x"},
            },
        }
        with pytest.raises(ProviderConfigError, match="Unknown provider"):
            resolve_provider_for_stage("analyst", config)

    def test_unknown_default_provider_raises(self) -> None:
        config: dict = {"llm_provider": "totally_made_up"}
        with pytest.raises(ProviderConfigError, match="Unknown llm_provider"):
            resolve_provider_for_stage("analyst", config)

    def test_per_stage_override_missing_provider_raises(self) -> None:
        config: dict = {
            "per_stage_providers": {"analyst": {"model": "gpt-5.4"}},
        }
        with pytest.raises(ProviderConfigError, match="provider is required"):
            resolve_provider_for_stage("analyst", config)

    def test_per_stage_override_missing_model_raises(self) -> None:
        config: dict = {
            "per_stage_providers": {"analyst": {"provider": "openai"}},
        }
        with pytest.raises(ProviderConfigError, match="model is required"):
            resolve_provider_for_stage("analyst", config)

    def test_per_stage_override_non_dict_raises(self) -> None:
        config: dict = {
            "per_stage_providers": {"analyst": "not-a-dict"},
        }
        with pytest.raises(ProviderConfigError, match="must be a dict"):
            resolve_provider_for_stage("analyst", config)

    def test_returns_provider_config_dataclass(self) -> None:
        config: dict = {
            "per_stage_providers": {
                "analyst": {
                    "provider": "openai",
                    "model": "gpt-5.4",
                    "base_url": "https://example.invalid/v1",
                },
            },
        }
        cfg = resolve_provider_for_stage("analyst", config)
        assert isinstance(cfg, ProviderConfig)
        assert cfg.base_url == "https://example.invalid/v1"


# ---------------------------------------------------------------------------
# Backward compatibility — legacy schema still works
# ---------------------------------------------------------------------------


class TestLegacyConfigSchemaBackwardCompat:
    """3b.5 / 3b.7 / 3b.9 configs use llm_provider + per-provider models
    blocks. They must keep working post-3b.10."""

    def test_3b9_config_shape_resolves(self) -> None:
        """Mirror runs/v3_0_smoketest_mtheff_college_20260427_3b7/config.yaml
        shape (the same shape 3b.9 used)."""
        config: dict = {
            "llm_provider": "minimax",
            "minimax": {
                "base_url": "https://api.minimax.io/anthropic",
                "models": {
                    "problem_formulator": "MiniMax-M2.7",
                    "data_engineer": "MiniMax-M2.7",
                    "analyst": "MiniMax-M2.7",
                    "critic": "MiniMax-M2.7",
                    "writer": "MiniMax-M2.7",
                },
            },
        }
        for stage in (
            "problem_formulator", "data_engineer", "analyst",
            "critic", "writer",
        ):
            cfg = resolve_provider_for_stage(stage, config)
            assert cfg.name == "minimax"
            assert cfg.model == "MiniMax-M2.7"

    def test_3b5_openai_config_shape_resolves(self) -> None:
        """Mirror 3b.5's gpt-5.4 config shape (legacy single-provider
        OpenAI)."""
        config: dict = {
            "llm_provider": "openai",
            "openai": {
                "models": {
                    "analyst": "gpt-5.4",
                    "writer": "gpt-5.4",
                },
            },
        }
        cfg = resolve_provider_for_stage("analyst", config)
        assert cfg.name == "openai"
        assert cfg.model == "gpt-5.4"

    def test_anthropic_direct_config_shape_resolves(self) -> None:
        """Pre-3b.5 anthropic-direct path: config.models is the
        per-agent map; no per-provider block."""
        config: dict = {
            "llm_provider": "anthropic",
            "models": {"analyst": "claude-sonnet-4-6"},
        }
        cfg = resolve_provider_for_stage("analyst", config)
        assert cfg.name == "anthropic"
        assert cfg.model == "claude-sonnet-4-6"
