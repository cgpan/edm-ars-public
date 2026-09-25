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
       - model = config[<provider>]['models'][<agent_key>] for the
         minimax / openai / deepseek providers, and
         config['models'][<agent_key>] for anthropic. A missing entry
         comes back as "" -- the resolver never guesses. BaseAgent then
         applies the provider default for deepseek and minimax, and
         refuses to start for openai, which ships no default block
         (defect E1: every stage used to run silently on gpt-4o).

  ``ProviderConfig.model_key`` records WHERE the model was (or would
  have been) read from, e.g. ``deepseek.models.critic``, so an error
  about a retired or missing model can name the line to edit.

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
    #: Dotted config path the model id comes from (for error messages).
    model_key: str = ""
    #: True when the stage has its own ``per_stage_providers`` entry. Its
    #: ``base_url`` then names one stage explicitly and outranks the
    #: provider-wide ``<PROVIDER>_BASE_URL`` environment variable.
    per_stage: bool = False


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
            model_key=f"per_stage_providers.{agent_key}.model",
            per_stage=True,
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
        return ProviderConfig(
            name=provider,
            model=model,
            base_url=base_url,
            model_key=f"{provider}.models.{agent_key}",
        )

    # Anthropic-direct path uses config["models"][agent_key].
    model = (config.get("models", {}) or {}).get(agent_key, "")
    return ProviderConfig(
        name=provider, model=model, base_url=None,
        model_key=f"models.{agent_key}",
    )


#: Models BaseAgent falls back to when a stage has no entry. openai has
#: none on purpose (E1): config.yaml ships no openai block, and the old
#: silent gpt-4o default put every stage, the Critic included, on it.
_AGENT_DEFAULTS: dict[str, str] = {
    "deepseek": "deepseek-v4-pro",
    "minimax": "MiniMax-M2.5",
}


def effective_model(provider_cfg: ProviderConfig, agent_key: str) -> str:
    """The model id an agent stage will call.

    The configured model when there is one; else the provider default
    for deepseek and minimax; for openai a ProviderConfigError naming the
    missing key; for anthropic whatever ``models.<agent_key>`` held (the
    resolver already read it), possibly "".
    """
    if provider_cfg.model:
        return provider_cfg.model
    if provider_cfg.name in _AGENT_DEFAULTS:
        return _AGENT_DEFAULTS[provider_cfg.name]
    if provider_cfg.name == "openai":
        raise ProviderConfigError(
            f"llm_provider is 'openai' but no model is configured for the "
            f"'{agent_key}' stage. Add openai.models.{agent_key} to "
            f"config.yaml (or a per_stage_providers.{agent_key} entry "
            f"with provider and model). EDM-ARS no longer falls back to "
            f"gpt-4o silently."
        )
    return provider_cfg.model


#: Last-resort revision models for providers that have a sensible one.
#: openai and anthropic deliberately have none: the only id the config
#: could offer them is ``review_gate.revision_model``, which ships as a
#: DeepSeek id and fails with model-not-found anywhere else (defect E2).
_REVISION_DEFAULTS: dict[str, str] = {
    "deepseek": "deepseek-v4-pro",
    "minimax": "MiniMax-M2.7",
}


def resolve_revision_writer(config: dict) -> ProviderConfig:
    """Provider and model for the review gate's manuscript reviser.

    Resolution order:
      1. ``per_stage_providers.revision_writer`` (provider + model).
      2. ``<provider>.models.revision_writer`` (``models.revision_writer``
         for anthropic).
      3. ``<provider>.models.writer`` (``models.writer`` for anthropic):
         the reviser rewrites the Writer's manuscript, so the Writer's
         model is the natural fallback.
      4. ``review_gate.revision_model`` -- ONLY when the provider is
         deepseek. That key ships as a DeepSeek id; sending it to
         OpenAI, Anthropic or a local server can only fail.
      5. The provider's built-in default (deepseek, minimax), else "".

    An empty ``model`` means no reviser can be configured; the caller
    decides what to do (the gate disables revision and says so).
    """
    primary = resolve_provider_for_stage("revision_writer", config)
    if primary.per_stage or primary.model:
        return primary
    # The provider-wide writer entry, not a per_stage_providers.writer
    # override: that one may name a different provider altogether.
    if primary.name == "anthropic":
        writer_model = (config.get("models") or {}).get("writer") or ""
        writer_key = "models.writer"
    else:
        block = config.get(primary.name) or {}
        writer_model = (block.get("models") or {}).get("writer") or ""
        writer_key = f"{primary.name}.models.writer"
    if writer_model:
        return ProviderConfig(
            name=primary.name,
            model=str(writer_model),
            base_url=primary.base_url,
            model_key=writer_key,
        )
    if primary.name == "deepseek":
        rv = ((config.get("review_gate") or {}).get("revision_model") or "")
        if rv:
            return ProviderConfig(
                name=primary.name,
                model=str(rv),
                base_url=primary.base_url,
                model_key="review_gate.revision_model",
            )
    return ProviderConfig(
        name=primary.name,
        model=_REVISION_DEFAULTS.get(primary.name, ""),
        base_url=primary.base_url,
        model_key=primary.model_key,
    )


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
