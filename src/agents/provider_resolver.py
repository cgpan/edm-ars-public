"""Per-stage provider + max_tokens resolution for agent stages.

Phase 3b.10 / §10.1.3 + §10.2. Centralises the logic that picks which
LLM provider + model + max_tokens an agent should use, given the
run config.

The resolution rules:

  Provider/model:
    1. If config['per_stage_providers'][<agent_key>] is set, use that
       (its 'provider' + 'model' fields).
    2. Else, fall back to the legacy config schema:
       - provider = config['llm_provider'] (default 'anthropic')
       - model = config[<provider>]['models'][<agent_key>] if present,
         otherwise config['models'][<agent_key>] (Anthropic-direct path),
         otherwise the provider's hardcoded default.

  Max tokens:
    1. If config['per_stage_max_tokens'][<agent_key>] is set, use that.
    2. Else, fall back to config['default_max_tokens'] if set.
    3. Else, fall back to whatever the agent's prompt YAML / call site
       passes (handled at call time by BaseAgent.call_llm; this resolver
       returns ``None`` to mean "use the per-call default").

The resolver is a pure function of the config dict — no I/O, no env
reads, no client construction. That makes it easy to unit-test without
LLM calls.
"""
from __future__ import annotations

from dataclasses import dataclass


_KNOWN_PROVIDERS: frozenset[str] = frozenset(
    {"anthropic", "minimax", "openai", "deepseek"}
)


class ProviderConfigError(ValueError):
    """Raised when the run config has a malformed provider override."""


@dataclass(frozen=True)
class ProviderConfig:
    """Resolved provider configuration for a single agent stage.

    Attributes
    ----------
    name:
        Provider identifier — one of ``_KNOWN_PROVIDERS``.
    model:
        Model string (e.g., "gpt-5.4", "MiniMax-M2.7", "claude-sonnet-4-6").
    base_url:
        Optional base URL override. When ``None``, the provider's default
        endpoint is used.
    """

    name: str
    model: str
    base_url: str | None = None


def resolve_provider_for_stage(
    agent_key: str,
    config: dict,
) -> ProviderConfig:
    """Return the resolved provider for a given agent stage.

    Parameters
    ----------
    agent_key:
        Lowercase, underscored agent identifier — e.g., 'analyst',
        'data_engineer', 'problem_formulator', 'writer', 'critic'.
    config:
        The run config dict (loaded from YAML).

    Raises
    ------
    ProviderConfigError:
        If the per-stage override declares an unknown provider name, or
        if a per-stage override is missing required fields.
    """
    per_stage = config.get("per_stage_providers", {}) or {}
    override = per_stage.get(agent_key)

    if override is not None:
        if not isinstance(override, dict):
            raise ProviderConfigError(
                f"per_stage_providers[{agent_key!r}] must be a dict, "
                f"got {type(override).__name__}"
            )
        provider = override.get("provider")
        model = override.get("model")
        if not provider:
            raise ProviderConfigError(
                f"per_stage_providers[{agent_key!r}].provider is required"
            )
        if not model:
            raise ProviderConfigError(
                f"per_stage_providers[{agent_key!r}].model is required"
            )
        if provider not in _KNOWN_PROVIDERS:
            raise ProviderConfigError(
                f"Unknown provider {provider!r} for stage {agent_key!r}. "
                f"Known providers: {sorted(_KNOWN_PROVIDERS)}"
            )
        return ProviderConfig(
            name=provider,
            model=model,
            base_url=override.get("base_url"),
        )

    # Fall back to legacy single-provider schema.
    provider = config.get("llm_provider", "anthropic")
    if provider not in _KNOWN_PROVIDERS:
        raise ProviderConfigError(
            f"Unknown llm_provider {provider!r}. "
            f"Known providers: {sorted(_KNOWN_PROVIDERS)}"
        )

    if provider in ("minimax", "openai", "deepseek"):
        # Per-provider per-agent model override: config[<provider>][models][<agent_key>]
        provider_block = config.get(provider, {}) or {}
        model = (provider_block.get("models", {}) or {}).get(agent_key)
        if not model:
            # Provider-class hardcoded defaults are applied in BaseAgent
            # if model is empty; we surface the empty here so callers can
            # decide whether to fall back further.
            model = ""
        base_url = provider_block.get("base_url")
        return ProviderConfig(name=provider, model=model, base_url=base_url)

    # Anthropic-direct path uses config["models"][agent_key].
    model = (config.get("models", {}) or {}).get(agent_key, "")
    return ProviderConfig(name=provider, model=model, base_url=None)


def resolve_max_tokens_for_stage(
    agent_key: str,
    config: dict,
    fallback: int = 8192,
) -> int:
    """Return the max_tokens budget for a given agent stage.

    Resolution order:
      1. ``config['per_stage_max_tokens'][<agent_key>]`` if set.
      2. ``config['default_max_tokens']`` if set.
      3. ``fallback`` (default 8192 — matches the BaseAgent legacy default).
    """
    per_stage = config.get("per_stage_max_tokens", {}) or {}
    if agent_key in per_stage:
        value = per_stage[agent_key]
        if not isinstance(value, int) or value <= 0:
            raise ProviderConfigError(
                f"per_stage_max_tokens[{agent_key!r}] must be a positive "
                f"int, got {value!r}"
            )
        return value
    default = config.get("default_max_tokens")
    if default is not None:
        if not isinstance(default, int) or default <= 0:
            raise ProviderConfigError(
                f"default_max_tokens must be a positive int, got {default!r}"
            )
        return default
    return fallback
