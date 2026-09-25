"""Building LLM clients, and deciding what a failed provider call means.

BaseAgent and the review gate each used to build their own clients, and
they disagreed: the gate ignored ``per_stage_providers`` and the
``DEEPSEEK_BASE_URL`` fallback, sent a DeepSeek model id to whichever
provider was active (E2/E3), and neither passed a timeout. Both now go
through this module, so a request goes to the same place, waits the
same bounded time and fails with the same classified error wherever it
starts.

Three rules:

* Every client gets an explicit timeout (``llm.request_timeout_s``,
  default 600 s) and SDK-internal retries switched OFF. Retrying happens
  here, where every wait is announced (D6) instead of spent silently.
* Transient failures -- connection errors, timeouts, HTTP 408/409/5xx,
  an overloaded server -- are retried ``llm.max_network_retries`` times
  (default 3) with a 30/60/120 s backoff, so a Wi-Fi drop or a laptop
  waking from sleep does not kill an overnight run. HTTP 429 keeps the
  schedule it always had: wait 60 s, then 120 s, then give up.
* Failures a person has to act on -- a rejected key, an empty account, a
  retired model id -- are never retried. They raise
  :class:`src.errors.ProviderError` with a code from ``ABORT_CODES`` and
  a message that names the provider, the model, the environment variable
  or config key involved, and what to do next.

Base URL precedence (E3), the same for every agent and for the gate:

  1. ``per_stage_providers.<stage>.base_url`` -- it names one stage;
  2. the ``<PROVIDER>_BASE_URL`` environment variable (DEEPSEEK_BASE_URL,
     OPENAI_BASE_URL, MINIMAX_BASE_URL) -- an explicit operator override
     that must work without editing the shipped config.yaml;
  3. ``<provider>.base_url`` in config.yaml;
  4. the provider default (the SDK's own for openai and anthropic; the
     anthropic SDK still reads ANTHROPIC_BASE_URL itself).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, TypeVar

from src.agents.provider_resolver import (
    ProviderConfig,
    ProviderConfigError,
    effective_model,
    resolve_provider_for_stage,
    resolve_revision_writer,
)
from src.errors import ProviderError

T = TypeVar("T")

DEFAULT_REQUEST_TIMEOUT_S: float = 600.0
DEFAULT_MAX_NETWORK_RETRIES: int = 3
#: Connect timeout, capped separately: a host that does not accept a TCP
#: connection within this long is not going to, and a connect stall
#: should not cost the full read budget.
CONNECT_TIMEOUT_CAP_S: float = 10.0
#: Wait before retry n of a transient failure (the last value repeats).
NETWORK_BACKOFF_S: tuple[int, ...] = (30, 60, 120)
#: Wait before retry n of an HTTP 429. Unchanged from the original
#: call_llm loop: three attempts, 60 s then 120 s apart.
RATE_LIMIT_WAITS_S: tuple[int, ...] = (60, 120)

API_KEY_ENV: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "minimax": "MINIMAX_API_KEY",
    "openai": "OPENAI_API_KEY",
}
BASE_URL_ENV: dict[str, str] = {
    "deepseek": "DEEPSEEK_BASE_URL",
    "minimax": "MINIMAX_BASE_URL",
    "openai": "OPENAI_BASE_URL",
}
DEFAULT_BASE_URL: dict[str, str] = {
    "deepseek": "https://api.deepseek.com",
    "minimax": "https://api.minimax.io/anthropic",
}

#: Wait callback: (seconds, failed_attempt_number, reason, message).
WaitCallback = Callable[[float, int, str, str], None]


@dataclass(frozen=True)
class LLMSettings:
    """Transport limits shared by every client (config block ``llm``)."""

    request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S
    max_network_retries: int = DEFAULT_MAX_NETWORK_RETRIES


def llm_settings(config: dict | None) -> LLMSettings:
    """Read ``llm.request_timeout_s`` and ``llm.max_network_retries``.

    Absent or null keys take the defaults. A value that is present but
    unusable is a configuration error, raised before any request is made,
    rather than a silently unbounded wait.
    """
    block = (config or {}).get("llm") or {}
    if not isinstance(block, dict):
        raise ProviderConfigError(f"config 'llm' must be a mapping, got {type(block).__name__}")
    raw_timeout = block.get("request_timeout_s")
    raw_retries = block.get("max_network_retries")
    try:
        timeout = DEFAULT_REQUEST_TIMEOUT_S if raw_timeout is None else float(raw_timeout)
    except (TypeError, ValueError) as exc:
        raise ProviderConfigError(
            f"llm.request_timeout_s must be a number of seconds, got {raw_timeout!r}"
        ) from exc
    if timeout <= 0:
        raise ProviderConfigError(f"llm.request_timeout_s must be positive, got {raw_timeout!r}")
    if raw_retries is None:
        retries = DEFAULT_MAX_NETWORK_RETRIES
    elif isinstance(raw_retries, bool) or not isinstance(raw_retries, int) or raw_retries < 0:
        raise ProviderConfigError(
            f"llm.max_network_retries must be a non-negative integer, got {raw_retries!r}"
        )
    else:
        retries = raw_retries
    return LLMSettings(request_timeout_s=timeout, max_network_retries=retries)


def resolve_base_url(provider_cfg: ProviderConfig) -> Optional[str]:
    """Effective base URL for *provider_cfg*; see the module docstring."""
    if provider_cfg.per_stage and provider_cfg.base_url:
        return provider_cfg.base_url
    env_name = BASE_URL_ENV.get(provider_cfg.name)
    if env_name:
        env_value = (os.environ.get(env_name) or "").strip()
        if env_value:
            return env_value
    if provider_cfg.base_url:
        return provider_cfg.base_url
    return DEFAULT_BASE_URL.get(provider_cfg.name)


def _timeout(settings: LLMSettings) -> Any:
    """An httpx timeout: the full budget for reads, a short one to connect."""
    seconds = float(settings.request_timeout_s)
    try:
        import httpx  # both SDKs depend on it

        return httpx.Timeout(seconds, connect=min(seconds, CONNECT_TIMEOUT_CAP_S))
    except Exception:  # noqa: BLE001 -- a float is accepted too
        return seconds


def build_client(provider_cfg: ProviderConfig, settings: LLMSettings) -> Any:
    """Construct the SDK client for *provider_cfg*.

    Raises ``EnvironmentError`` when the provider's API key is not set or
    its SDK is not installed (the messages name the variable / package).
    """
    import anthropic  # type: ignore[import-not-found]

    provider = provider_cfg.name
    env_name = API_KEY_ENV.get(provider, "ANTHROPIC_API_KEY")
    api_key = os.environ.get(env_name)
    if not api_key:
        if provider == "anthropic":
            raise EnvironmentError(
                "ANTHROPIC_API_KEY environment variable is not set. "
                "Export it before running the pipeline: "
                "export ANTHROPIC_API_KEY=sk-ant-..."
            )
        raise EnvironmentError(
            f"provider '{provider}' selected but {env_name} is not set. "
            "Add it to a .env file or export it in your shell."
        )
    base_url = resolve_base_url(provider_cfg)
    common: dict[str, Any] = {"timeout": _timeout(settings), "max_retries": 0}

    if provider in ("openai", "deepseek"):
        try:
            import openai  # type: ignore[import-not-found]
        except ImportError as exc:
            extra = (
                " (DeepSeek uses the OpenAI-compatible endpoint)"
                if provider == "deepseek" else ""
            )
            raise EnvironmentError(
                f"provider '{provider}' selected but the openai SDK is not "
                f"installed{extra}. Install it with: pip install openai"
            ) from exc
        kwargs: dict[str, Any] = {"api_key": api_key, **common}
        if base_url:
            kwargs["base_url"] = base_url
        return openai.OpenAI(**kwargs)

    kwargs = {"api_key": api_key, **common}
    if base_url:
        kwargs["base_url"] = base_url
    return anthropic.Anthropic(**kwargs)


# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------

#: Actions: retry after a rate-limit wait, retry after a network backoff,
#: raise a ProviderError now, or re-raise the exception untouched (it is
#: not a provider failure -- a bug, a parse error, a KeyboardInterrupt).
RATE_LIMIT = "rate_limit"
TRANSIENT = "transient"
FATAL = "fatal"
PASSTHROUGH = "passthrough"

_PROVIDER_MODULES = ("openai", "anthropic", "httpx", "httpcore")

_NO_CREDIT_MARKERS = (
    "insufficient balance",       # DeepSeek, HTTP 402
    "insufficient_balance",
    "insufficient_quota",         # OpenAI: HTTP 429 with this code = no credit
    "credit balance is too low",  # Anthropic, HTTP 400
    "exceeded your current quota",
)
_MODEL_GONE_MARKERS = (
    "model_not_found",
    "model not exist",            # DeepSeek, HTTP 400
    "model does not exist",
    "unknown model",
    "invalid model",
    "no such model",
)


def _status_of(exc: BaseException) -> Optional[int]:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def _text_of(exc: BaseException) -> str:
    parts = [str(exc)]
    body = getattr(exc, "body", None)
    if body is not None:
        parts.append(str(body))
    for attr in ("code", "type"):
        value = getattr(exc, attr, None)
        if isinstance(value, str):
            parts.append(value)
    return " ".join(parts).lower()


def _is_provider_exception(exc: BaseException) -> bool:
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    for klass in type(exc).__mro__:
        module = (getattr(klass, "__module__", "") or "").split(".", 1)[0]
        if module in _PROVIDER_MODULES:
            return True
    return False


def _class_names(exc: BaseException) -> set[str]:
    return {klass.__name__ for klass in type(exc).__mro__}


def classify_exception(exc: BaseException) -> tuple[str, str]:
    """Return ``(action, code)`` for an exception raised by a provider call.

    ``code`` is an ``ABORT_CODES`` key; it is meaningful for every action
    except PASSTHROUGH.
    """
    if not isinstance(exc, Exception) or isinstance(exc, ProviderError):
        return PASSTHROUGH, ""
    if not _is_provider_exception(exc):
        return PASSTHROUGH, ""
    names = _class_names(exc)
    status = _status_of(exc)
    text = _text_of(exc)

    # Out of credit first: OpenAI reports it as a 429, which must not be
    # mistaken for a rate limit worth waiting out.
    if status == 402 or any(m in text for m in _NO_CREDIT_MARKERS):
        return FATAL, "NO_CREDIT"
    if status in (401, 403) or names & {"AuthenticationError", "PermissionDeniedError"}:
        return FATAL, "KEY_REJECTED"
    if status == 404 or "NotFoundError" in names or any(m in text for m in _MODEL_GONE_MARKERS):
        return FATAL, "MODEL_GONE"
    if status == 429 or "RateLimitError" in names:
        return RATE_LIMIT, "RATE_LIMITED"
    if (
        status == 408
        or isinstance(exc, TimeoutError)
        or any("Timeout" in n for n in names)
    ):
        return TRANSIENT, "TIMEOUT"
    if (
        "APIConnectionError" in names
        or "TransportError" in names  # httpx network/protocol failures
        or isinstance(exc, ConnectionError)
    ):
        return TRANSIENT, "NETWORK"
    if (
        (status is not None and (status >= 500 or status == 409))
        or "InternalServerError" in names
        or "overloaded" in text
    ):
        return TRANSIENT, "PROVIDER_ERROR"
    if status is not None or "APIError" in names:
        # 400/413/422 and friends: the request itself is wrong; retrying
        # cannot help, but the code tells the operator it was the provider.
        return FATAL, "PROVIDER_ERROR"
    return PASSTHROUGH, ""


def _short(exc: BaseException, limit: int = 300) -> str:
    text = " ".join(str(exc).split())
    if not text:
        text = type(exc).__name__
    return text if len(text) <= limit else text[: limit - 3] + "..."


def describe_failure(
    code: str,
    exc: BaseException,
    provider_cfg: ProviderConfig,
    *,
    model: str,
    settings: LLMSettings,
    retries: int = 0,
    waited_s: float = 0.0,
) -> str:
    """One actionable sentence for a provider failure (no secrets)."""
    provider = provider_cfg.name
    status = _status_of(exc)
    http = f"HTTP {status}" if status is not None else type(exc).__name__
    detail = _short(exc)
    env_name = API_KEY_ENV.get(provider, "the API key variable")
    model_key = provider_cfg.model_key or "the model setting"
    base_url = resolve_base_url(provider_cfg) or "the provider's default endpoint"
    if code == "KEY_REJECTED":
        return (
            f"{provider} rejected the API key in {env_name} ({http}) for model "
            f"{model!r}. Check or replace the key, then resume the run. [{detail}]"
        )
    if code == "NO_CREDIT":
        return (
            f"{provider} refused the request for lack of credit ({http}). Top up "
            f"the {provider} account, then resume the run. [{detail}]"
        )
    if code == "MODEL_GONE":
        return (
            f"{provider} does not serve model {model!r} ({http}). Set a current "
            f"model id at {model_key} in config.yaml (or check the endpoint "
            f"{base_url}), then resume the run. [{detail}]"
        )
    if code == "RATE_LIMITED":
        return (
            f"{provider} was still rate-limiting model {model!r} after {retries} "
            f"waits ({waited_s:.0f} s in total). Resume the run later. [{detail}]"
        )
    if code == "TIMEOUT":
        return (
            f"{provider} did not answer within {settings.request_timeout_s:.0f} s, "
            f"{retries + 1} tries in a row, for model {model!r}. Resume the run "
            f"later, or raise llm.request_timeout_s in config.yaml. [{detail}]"
        )
    if code == "NETWORK":
        return (
            f"Could not reach {provider} at {base_url} after {retries + 1} tries. "
            f"Check the network connection, then resume the run. [{detail}]"
        )
    if retries:
        return (
            f"{provider} kept failing ({http}) for model {model!r} after "
            f"{retries + 1} tries. Resume the run later. [{detail}]"
        )
    return f"{provider} rejected the request for model {model!r} ({http}). [{detail}]"


def call_with_retries(
    call: Callable[[], T],
    *,
    provider_cfg: ProviderConfig,
    model: str,
    settings: LLMSettings,
    on_wait: Optional[WaitCallback] = None,
    sleep: Optional[Callable[[float], None]] = None,
) -> T:
    """Run *call*, retrying provider failures under the module's policy.

    ``on_wait`` is told about every wait BEFORE the sleep starts, so a
    log line or event reaches the operator while the run is paused, not
    after. It must not raise; any exception from it is ignored.
    """
    do_sleep = sleep if sleep is not None else time.sleep
    rate_waits = 0
    net_retries = 0
    waited = 0.0
    while True:
        try:
            return call()
        except Exception as exc:
            action, code = classify_exception(exc)
            if action == PASSTHROUGH:
                raise
            attempt = rate_waits + net_retries + 1
            if action == RATE_LIMIT and rate_waits < len(RATE_LIMIT_WAITS_S):
                wait_s = float(RATE_LIMIT_WAITS_S[rate_waits])
                rate_waits += 1
                reason = "rate_limit"
                message = (
                    f"Rate limit hit (attempt {rate_waits}/{len(RATE_LIMIT_WAITS_S) + 1}); "
                    f"waiting {wait_s:.0f}s before retry. ({_short(exc)})"
                )
            elif action == TRANSIENT and net_retries < settings.max_network_retries:
                wait_s = float(NETWORK_BACKOFF_S[min(net_retries, len(NETWORK_BACKOFF_S) - 1)])
                net_retries += 1
                reason = code.lower()
                message = (
                    f"{provider_cfg.name} call failed ({code}: {_short(exc, 200)}); "
                    f"retry {net_retries}/{settings.max_network_retries} in {wait_s:.0f}s."
                )
            else:
                retries = rate_waits if action == RATE_LIMIT else net_retries
                raise ProviderError(
                    code,
                    describe_failure(
                        code, exc, provider_cfg, model=model, settings=settings,
                        retries=retries, waited_s=waited,
                    ),
                    provider=provider_cfg.name,
                    model=model,
                    status=_status_of(exc),
                ) from exc
            if on_wait is not None:
                try:
                    on_wait(wait_s, attempt, reason, message)
                except Exception:  # noqa: BLE001 -- announcing must not fail the call
                    pass
            waited += wait_s
            do_sleep(wait_s)


#: Stages ``describe_routing`` reports by default: the five SPEC agents,
#: the two later ones, and the review gate's reviser.
ROUTED_STAGES: tuple[str, ...] = (
    "problem_formulator", "data_engineer", "analyst", "critic", "writer",
    "outline_agent", "verifier", "revision_writer",
)


def describe_routing(
    config: dict, stages: tuple[str, ...] = ROUTED_STAGES
) -> list[dict[str, Any]]:
    """Where each stage's requests would go, without building a client.

    One row per stage: ``{"stage", "provider", "model", "model_key",
    "base_url", "key_env", "key_set", "error"}``. ``error`` is None or
    the configuration error that stage would raise at start-up, so a
    pre-flight (``--dry-run``) can print the whole table instead of
    stopping at the first problem. Reads the environment, never the
    network.
    """
    rows: list[dict[str, Any]] = []
    for stage in stages:
        row: dict[str, Any] = {"stage": stage, "provider": None, "model": None,
                               "model_key": None, "base_url": None,
                               "key_env": None, "key_set": None, "error": None}
        try:
            if stage == "revision_writer":
                cfg = resolve_revision_writer(config)
                model = cfg.model
                if not model:
                    raise ProviderConfigError(
                        f"no revision model for provider {cfg.name!r}; set "
                        f"{'models' if cfg.name == 'anthropic' else cfg.name + '.models'}"
                        ".revision_writer (or .writer)"
                    )
            else:
                cfg = resolve_provider_for_stage(stage, config)
                model = effective_model(cfg, stage)
            key_env = API_KEY_ENV.get(cfg.name)
            row.update(
                provider=cfg.name, model=model, model_key=cfg.model_key,
                base_url=resolve_base_url(cfg), key_env=key_env,
                key_set=bool(key_env and os.environ.get(key_env)),
            )
            if not model:
                # anthropic has no built-in default: the stage would send
                # an empty model id and fail at its first call.
                row["error"] = (
                    f"no model configured for the '{stage}' stage; set "
                    f"{cfg.model_key or 'its model'} in config.yaml"
                )
        except ProviderConfigError as exc:
            row["error"] = str(exc)
        rows.append(row)
    return rows
