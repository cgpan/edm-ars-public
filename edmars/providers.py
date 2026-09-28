"""AI-service catalog and live key checks.

A key check sends the key to the provider's own model-listing endpoint
(and, for DeepSeek, its balance endpoint) and maps the answer to one of
five plain outcomes:

    OK         the key works (and, for DeepSeek, the account has balance)
    REJECTED   HTTP 401/403 -- the provider does not accept the key
    NO_CREDIT  HTTP 402, or DeepSeek reports ``is_available: false``
    NETWORK    nothing answered (offline, proxy, firewall, server down)
    UNKNOWN    any other answer; the message names the HTTP status

The key only ever travels in a request header to the provider it belongs
to. It is never logged, never put in a URL, and never included in a
returned message (provider error texts are scrubbed before use).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping
from urllib.parse import urlsplit

import requests

from edmars import estimates

KeyStatus = Literal["OK", "REJECTED", "NO_CREDIT", "NETWORK", "UNKNOWN"]
KEY_STATUSES: tuple[str, ...] = ("OK", "REJECTED", "NO_CREDIT", "NETWORK", "UNKNOWN")
DEFAULT_TIMEOUT = 15.0
ANTHROPIC_VERSION = "2023-06-01"


@dataclass(frozen=True)
class ProviderInfo:
    """One AI service the wizard can offer."""

    id: str
    label: str
    env_var: str
    key_page: str
    #: API base URL written into the run config (None = the user supplies it).
    base_url: str | None
    #: "recommended", "works" (less tested) or "experimental".
    support: Literal["recommended", "works", "experimental"]
    note: str = ""
    top_up_url: str | None = None
    #: Key prefix a genuine key starts with (used only to explain a rejection).
    key_prefix: str | None = None
    #: (label, base URL) presets offered for a local/self-hosted server.
    presets: tuple[tuple[str, str], ...] = field(default_factory=tuple)


PROVIDERS: dict[str, ProviderInfo] = {
    "deepseek": ProviderInfo(
        id="deepseek",
        label="DeepSeek (recommended)",
        env_var="DEEPSEEK_API_KEY",
        key_page="https://platform.deepseek.com/api_keys",
        base_url="https://api.deepseek.com",
        support="recommended",
        note=(
            "The configuration EDM-ARS is tested with. Pay-as-you-go; a study "
            "costs " + estimates.COST_DEEPSEEK_WITH_REVIEW
        ),
        top_up_url="https://platform.deepseek.com/top_up",
        key_prefix="sk-",
    ),
    "openai": ProviderInfo(
        id="openai",
        label="OpenAI (ChatGPT API) - works, less tested",
        env_var="OPENAI_API_KEY",
        key_page="https://platform.openai.com/api-keys",
        base_url="https://api.openai.com/v1",
        support="works",
        note=(
            "Needs an API key from the OpenAI platform. A ChatGPT Plus "
            "subscription is NOT an API key and does not include API credit."
        ),
        top_up_url="https://platform.openai.com/settings/organization/billing",
        key_prefix="sk-",
    ),
    "anthropic": ProviderInfo(
        id="anthropic",
        label="Anthropic (Claude API) - works, less tested",
        env_var="ANTHROPIC_API_KEY",
        key_page="https://platform.claude.com/settings/keys",
        base_url="https://api.anthropic.com",
        support="works",
        note=(
            "Needs an API key from the Claude developer platform. A Claude Pro "
            "subscription is NOT an API key and does not include API credit."
        ),
        top_up_url="https://platform.claude.com/settings/billing",
        key_prefix="sk-ant-",
    ),
    "local": ProviderInfo(
        id="local",
        label="A model on my own computer/server - experimental",
        env_var="OPENAI_API_KEY",
        key_page="",
        base_url=None,
        support="experimental",
        note=(
            "Any OpenAI-compatible server (Ollama, LM Studio, vLLM). The model "
            "needs a context window of at least 128k tokens; paper quality with "
            "local models is untested, and the automated reviewer (LSAR) stays "
            "off with this choice."
        ),
        presets=(
            ("Ollama", "http://localhost:11434/v1"),
            ("LM Studio", "http://localhost:1234/v1"),
            ("vLLM", "http://localhost:8000/v1"),
        ),
    ),
}

#: Placeholder key written for a local server that needs none.
LOCAL_PLACEHOLDER_KEY = "local"

SEMANTIC_SCHOLAR_ENV = "SEMANTIC_SCHOLAR_API_KEY"
SEMANTIC_SCHOLAR_KEY_PAGE = "https://www.semanticscholar.org/product/api#api-key-form"
SEMANTIC_SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

#: Model ids the shipped config used for DeepSeek on 2026-09-25; used only
#: when the app's config.yaml cannot be read.
_FALLBACK_DEEPSEEK_MODELS: dict[str, str] = {
    "problem_formulator": "deepseek-v4-pro",
    "data_engineer": "deepseek-v4-pro",
    "analyst": "deepseek-v4-pro",
    "critic": "deepseek-v4-pro",
    "writer": "deepseek-v4-pro",
    "revision_writer": "deepseek-v4-pro",
    "outline_agent": "deepseek-flash",
    "verifier": "deepseek-flash",
}

_SK_PATTERN = re.compile(r"sk-[A-Za-z0-9_\-*.]{4,}")
_BEARER_PATTERN = re.compile(r"(?i)bearer\s+\S+")


@dataclass
class KeyCheck:
    """Outcome of a key check. ``message`` is plain English, safe to print."""

    status: str
    message: str
    balance: str | None = None
    models: list[str] | None = None

    @property
    def ok(self) -> bool:
        return self.status == "OK"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _provider(provider_id: str) -> ProviderInfo:
    try:
        return PROVIDERS[provider_id]
    except KeyError:
        known = ", ".join(PROVIDERS)
        raise ValueError(f"Unknown AI service {provider_id!r}; choose one of: {known}.") from None


def clean_key(key: str | None) -> str:
    """Strip what a paste commonly drags along: spaces, newlines, quotes."""
    value = (key or "").strip().strip("\"'").strip()
    if value.lower().startswith("bearer "):
        value = value[7:].strip()
    return value


def _scrub(text: str, key: str | None) -> str:
    """Remove the key (and anything key-shaped) from provider text."""
    out = text or ""
    if key:
        out = out.replace(key, "[key]")
    out = _SK_PATTERN.sub("[key]", out)
    out = _BEARER_PATTERN.sub("Bearer [key]", out)
    return out[:300]


def _host(url: str) -> str:
    return urlsplit(url).netloc or url


def _error_text(response: Any, key: str | None) -> str:
    try:
        body = response.json()
    except Exception:  # noqa: BLE001 - not JSON
        body = None
    message = ""
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            message = str(err.get("message") or err.get("type") or "")
        elif isinstance(err, str):
            message = err
        elif body.get("message"):
            message = str(body["message"])
    if not message:
        message = str(getattr(response, "text", "") or "")[:200]
    return _scrub(message.strip(), key)


def _network_message(label: str, url: str, exc: Exception, timeout: float) -> str:
    host = _host(url)
    if isinstance(exc, requests.exceptions.Timeout):
        why = f"it did not answer within {timeout:.0f} seconds"
    elif isinstance(exc, requests.exceptions.SSLError):
        why = ("the secure connection failed; a proxy, VPN or antivirus may be "
               "inspecting HTTPS traffic")
    elif isinstance(exc, requests.exceptions.ProxyError):
        why = "the proxy refused the connection"
    else:
        why = "the connection failed"
    return (f"Could not reach {label} at {host}: {why}. Check your internet "
            "connection, VPN or proxy, then try again.")


def _get(
    session: Any | None,
    url: str,
    headers: Mapping[str, str],
    timeout: float,
    params: Mapping[str, Any] | None = None,
) -> Any:
    http = session if session is not None else requests
    kwargs: dict[str, Any] = {"headers": dict(headers), "timeout": timeout}
    if params:
        kwargs["params"] = dict(params)
    return http.get(url, **kwargs)


def _models_url(provider_id: str, base_url: str | None) -> str:
    info = _provider(provider_id)
    base = (base_url or info.base_url or "").rstrip("/")
    if not base:
        raise ValueError("A server address (base URL) is needed for a local model.")
    if provider_id == "anthropic":
        if not base.endswith("/v1"):
            base = base + "/v1"
        return base + "/models"
    return base + "/models"


def _auth_headers(provider_id: str, key: str) -> dict[str, str]:
    if provider_id == "anthropic":
        return {"x-api-key": key, "anthropic-version": ANTHROPIC_VERSION}
    return {"Authorization": f"Bearer {key or LOCAL_PLACEHOLDER_KEY}"}


def _model_ids(body: Any) -> list[str]:
    items = body.get("data") if isinstance(body, dict) else body
    ids: list[str] = []
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict) and item.get("id"):
                ids.append(str(item["id"]))
            elif isinstance(item, str):
                ids.append(item)
    return ids


def _key_hint(provider_id: str, key: str) -> str:
    """Explain a rejection when the key obviously belongs somewhere else."""
    if provider_id != "anthropic" and key.startswith("sk-ant-"):
        return " This looks like an Anthropic (Claude) key, not a key for this service."
    if provider_id == "anthropic" and key.startswith("sk-") and not key.startswith("sk-ant-"):
        return " Anthropic keys start with 'sk-ant-'; this looks like a key for another service."
    if provider_id == "openai" and key.startswith("sk-or-"):
        return " This looks like an OpenRouter key, not an OpenAI key."
    return ""


# ---------------------------------------------------------------------------
# Key checks
# ---------------------------------------------------------------------------


def list_models(
    provider_id: str,
    key: str | None,
    base_url: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    *,
    session: Any | None = None,
) -> KeyCheck:
    """GET the provider's model list; ``models`` is filled when status is OK."""
    info = _provider(provider_id)
    key = clean_key(key)
    name = info.label.split(" (")[0].split(" - ")[0]
    if not key and provider_id != "local":
        return KeyCheck("REJECTED", f"No {name} key was entered.")
    try:
        url = _models_url(provider_id, base_url)
    except ValueError as exc:
        return KeyCheck("UNKNOWN", str(exc))
    headers = _auth_headers(provider_id, key)
    params = {"limit": 1000} if provider_id == "anthropic" else None
    models: list[str] = []
    try:
        for _page in range(10):
            response = _get(session, url, headers, timeout, params)
            status = int(response.status_code)
            if status != 200:
                return _status_check(provider_id, status, response, key, url)
            try:
                body = response.json()
            except ValueError:
                return KeyCheck(
                    "UNKNOWN",
                    f"{_host(url)} answered, but not with a model list. Is "
                    f"{url.rsplit('/models', 1)[0]} the right address?",
                )
            models.extend(_model_ids(body))
            if (provider_id == "anthropic" and isinstance(body, dict)
                    and body.get("has_more") and body.get("last_id")):
                params = {"limit": 1000, "after_id": body["last_id"]}
                continue
            break
    except (UnicodeError, requests.exceptions.InvalidHeader):
        return KeyCheck(
            "REJECTED",
            "The key contains characters that no API key has; the paste probably "
            "picked up extra text. Copy the key again and paste only the key.",
        )
    except requests.exceptions.RequestException as exc:
        if provider_id == "local":
            return KeyCheck(
                "NETWORK",
                f"Nothing answered at {url.rsplit('/models', 1)[0]}. Start the model "
                "server (for example Ollama or LM Studio's local server) and check "
                "the address and port.",
            )
        return KeyCheck("NETWORK", _network_message(name, url, exc, timeout))
    message = f"The key works; {len(models)} models are available."
    if provider_id == "local":
        message = (f"The server answered with {len(models)} models."
                   if models else
                   "The server answered but lists no models; load or pull a model first.")
    return KeyCheck("OK", message, models=models)


def _status_check(
    provider_id: str, status: int, response: Any, key: str, url: str
) -> KeyCheck:
    info = _provider(provider_id)
    name = info.label.split(" (")[0].split(" - ")[0]
    if status in (401, 403):
        page = f" Check that you copied the whole key from {info.key_page}." if info.key_page else ""
        return KeyCheck(
            "REJECTED",
            f"{name} did not accept this key (HTTP {status}).{page}{_key_hint(provider_id, key)}",
        )
    if status == 402:
        top_up = f" Add credit at {info.top_up_url}." if info.top_up_url else ""
        return KeyCheck(
            "NO_CREDIT",
            f"The key is valid, but the {name} account has no credit (HTTP 402).{top_up}",
        )
    if status == 404 and provider_id == "local":
        return KeyCheck(
            "UNKNOWN",
            f"The server at {_host(url)} has no /models endpoint (HTTP 404). OpenAI-"
            "compatible addresses usually end in /v1.",
        )
    if status == 429:
        return KeyCheck(
            "UNKNOWN",
            f"{name} is limiting requests right now (HTTP 429); the key could not be "
            "confirmed. Wait a minute and try again.",
        )
    detail = _error_text(response, key)
    tail = f" ({detail})" if detail else ""
    return KeyCheck(
        "UNKNOWN",
        f"{name} gave an unexpected answer (HTTP {status}){tail}. Try again later.",
    )


def _format_balance(infos: Any) -> str | None:
    if not isinstance(infos, list):
        return None
    parts: list[str] = []
    for item in infos:
        if not isinstance(item, dict):
            continue
        amount = item.get("total_balance")
        currency = str(item.get("currency") or "").upper()
        if amount is None:
            continue
        parts.append(f"US${amount}" if currency == "USD" else f"{currency} {amount}".strip())
    return " + ".join(parts) if parts else None


def _deepseek_balance(
    key: str, base_url: str | None, timeout: float, session: Any | None
) -> tuple[bool | None, str | None]:
    """(is_available, balance text); (None, None) when it could not be read."""
    base = base_url or PROVIDERS["deepseek"].base_url or ""
    parts = urlsplit(base)
    url = f"{parts.scheme or 'https'}://{parts.netloc or 'api.deepseek.com'}/user/balance"
    try:
        response = _get(session, url, _auth_headers("deepseek", key), timeout)
        if int(response.status_code) != 200:
            return None, None
        body = response.json()
    except (requests.exceptions.RequestException, ValueError, UnicodeError):
        return None, None
    if not isinstance(body, dict):
        return None, None
    available = body.get("is_available")
    return (bool(available) if available is not None else None,
            _format_balance(body.get("balance_infos")))


def check_key(
    provider_id: str,
    key: str | None,
    base_url: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    *,
    session: Any | None = None,
) -> KeyCheck:
    """Check ``key`` against ``provider_id``'s API. Never raises on HTTP/network."""
    info = _provider(provider_id)
    result = list_models(provider_id, key, base_url, timeout, session=session)
    if provider_id != "deepseek" or result.status != "OK":
        return result
    available, balance = _deepseek_balance(clean_key(key), base_url, timeout, session)
    result.balance = balance
    if available is False:
        shown = balance or "0"
        return KeyCheck(
            "NO_CREDIT",
            f"The key is valid, but the DeepSeek account has no usable balance "
            f"({shown}). Top up at {info.top_up_url} and check again.",
            balance=balance,
            models=result.models,
        )
    if balance:
        result.message = f"The key works. Account balance: {balance}."
    else:
        result.message = "The key works (the account balance could not be read)."
    return result


def check_semantic_scholar(
    key: str | None,
    timeout: float = DEFAULT_TIMEOUT,
    *,
    session: Any | None = None,
) -> KeyCheck:
    """One search request with the key in ``x-api-key``."""
    key = clean_key(key)
    if not key:
        return KeyCheck("REJECTED", "No Semantic Scholar key was entered.")
    params = {"query": "educational data mining", "limit": 1, "fields": "title"}
    try:
        response = _get(session, SEMANTIC_SCHOLAR_SEARCH_URL, {"x-api-key": key},
                        timeout, params)
    except (UnicodeError, requests.exceptions.InvalidHeader):
        return KeyCheck(
            "REJECTED",
            "The key contains characters that no API key has; copy it again and "
            "paste only the key.",
        )
    except requests.exceptions.RequestException as exc:
        return KeyCheck("NETWORK", _network_message(
            "Semantic Scholar", SEMANTIC_SCHOLAR_SEARCH_URL, exc, timeout))
    status = int(response.status_code)
    if status == 200:
        return KeyCheck("OK", "The Semantic Scholar key works.")
    if status in (401, 403):
        return KeyCheck(
            "REJECTED",
            f"Semantic Scholar did not accept this key (HTTP {status}). Keys arrive "
            f"by email after you apply at {SEMANTIC_SCHOLAR_KEY_PAGE}.",
        )
    if status == 429:
        return KeyCheck(
            "UNKNOWN",
            "Semantic Scholar is limiting requests right now (HTTP 429), so the key "
            "could not be confirmed. Try again in a minute; the key is kept.",
        )
    detail = _error_text(response, key)
    return KeyCheck(
        "UNKNOWN",
        f"Semantic Scholar gave an unexpected answer (HTTP {status})"
        f"{f' ({detail})' if detail else ''}.",
    )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def _shipped_config() -> dict[str, Any]:
    try:
        import yaml

        from edmars import paths

        path = Path(paths.app_root()) / "config.yaml"
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - fall back to the built-in table
        return {}
    return data if isinstance(data, dict) else {}


def default_models(provider_id: str) -> dict[str, str]:
    """Per-stage model ids the shipped config.yaml uses for ``provider_id``.

    DeepSeek: ``deepseek.models``. OpenAI: ``openai.models`` when the
    shipped config has that block. Anthropic: ``anthropic.models``, else
    the top-level ``models`` block (the pipeline's Anthropic-direct path).
    Local: always empty -- the wizard asks the user to pick from the
    server's own ``/models`` list. An empty dict means "ask the user".
    """
    _provider(provider_id)
    if provider_id == "local":
        return {}
    config = _shipped_config()
    block = config.get(provider_id)
    models = block.get("models") if isinstance(block, dict) else None
    if not models and provider_id == "anthropic":
        models = config.get("models")
    if not isinstance(models, dict) or not models:
        return dict(_FALLBACK_DEEPSEEK_MODELS) if provider_id == "deepseek" else {}
    return {str(k): str(v) for k, v in models.items() if v}


def _served(model: str, listed: set[str]) -> bool:
    if model in listed:
        return True
    # Anthropic lists dated snapshots ("...-20250929") for an alias.
    return any(item.startswith(model + "-") for item in listed)


def missing_models(
    provider_id: str,
    key: str | None,
    models: Mapping[str, str] | Iterable[str],
    base_url: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    *,
    session: Any | None = None,
) -> list[str]:
    """Model ids in ``models`` that the provider's ``/models`` does not list.

    Catches retired ids before a run spends money. Returns ``[]`` when the
    list could not be fetched (use :func:`list_models` to tell "all
    present" from "could not check").
    """
    wanted = list(models.values()) if isinstance(models, Mapping) else list(models)
    listing = list_models(provider_id, key, base_url, timeout, session=session)
    if listing.status != "OK" or listing.models is None:
        return []
    listed = set(listing.models)
    missing: list[str] = []
    for model in wanted:
        if model and not _served(str(model), listed) and model not in missing:
            missing.append(str(model))
    return missing


def provider_choices() -> list[tuple[str, str]]:
    """(value, label) pairs in display order, for ``ui.select``."""
    return [(pid, info.label) for pid, info in PROVIDERS.items()]
