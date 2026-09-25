"""Provider key checks map HTTP answers to plain outcomes and never leak the key."""

from __future__ import annotations

from typing import Any

import pytest
import requests
import yaml

from tests.cli.infra_helpers import (  # noqa: F401 - fixtures
    REPO_ROOT,
    FakeResponse,
    FakeSession,
    edmars_home,
    no_network,
)

from edmars import providers  # noqa: E402

pytestmark = pytest.mark.usefixtures("edmars_home", "no_network")

# Built at run time so no key-shaped literal sits in the tracked file.
KEY = "sk" + "-" + "t3st" * 6
ANT_KEY = "sk" + "-ant-" + "t3st" * 6

MODELS = {"object": "list", "data": [{"id": "deepseek-v4-pro"}, {"id": "deepseek-flash"}]}


def _deepseek(balance: Any = None, models_status: int = 200) -> FakeSession:
    def handler(url: str, headers: dict[str, str], params: dict[str, Any]) -> Any:
        if url.endswith("/models"):
            return FakeResponse(models_status, json_data=MODELS if models_status == 200 else
                                {"error": {"message": "bad"}})
        if url.endswith("/user/balance"):
            if balance is None:
                return FakeResponse(500, json_data={"error": "down"})
            return FakeResponse(200, json_data=balance)
        raise AssertionError(url)

    return FakeSession(handler)


def test_deepseek_ok_reports_balance_and_models() -> None:
    session = _deepseek({"is_available": True, "balance_infos": [
        {"currency": "USD", "total_balance": "4.87", "granted_balance": "0.00",
         "topped_up_balance": "4.87"}]})
    result = providers.check_key("deepseek", f"  '{KEY}'\n", session=session)
    assert result.status == "OK"
    assert result.balance == "US$4.87"
    assert result.models == ["deepseek-v4-pro", "deepseek-flash"]
    assert "US$4.87" in result.message
    urls = [c["url"] for c in session.calls]
    assert urls == ["https://api.deepseek.com/models", "https://api.deepseek.com/user/balance"]
    for call in session.calls:
        assert call["headers"]["Authorization"] == f"Bearer {KEY}"  # cleaned paste
        assert call["timeout"] == 15


def test_deepseek_without_balance_is_no_credit() -> None:
    session = _deepseek({"is_available": False, "balance_infos": [
        {"currency": "CNY", "total_balance": "0.00"}]})
    result = providers.check_key("deepseek", KEY, session=session)
    assert result.status == "NO_CREDIT"
    assert result.balance == "CNY 0.00"
    assert "platform.deepseek.com/top_up" in result.message


def test_deepseek_balance_unreadable_is_still_ok() -> None:
    result = providers.check_key("deepseek", KEY, session=_deepseek(None))
    assert result.status == "OK"
    assert result.balance is None
    assert "could not be read" in result.message


@pytest.mark.parametrize(
    ("status", "expected"),
    [(401, "REJECTED"), (403, "REJECTED"), (402, "NO_CREDIT"), (429, "UNKNOWN"),
     (500, "UNKNOWN"), (404, "UNKNOWN")],
)
def test_http_status_mapping(status: int, expected: str) -> None:
    session = FakeSession(lambda u, h, p: FakeResponse(
        status, json_data={"error": {"message": f"Incorrect API key provided: {KEY}"}}))
    result = providers.check_key("openai", KEY, session=session)
    assert result.status == expected
    assert f"HTTP {status}" in result.message
    assert KEY not in result.message
    assert "sk-" not in result.message


@pytest.mark.parametrize(
    "exc",
    [requests.exceptions.ConnectionError("refused"), requests.exceptions.ReadTimeout("slow"),
     requests.exceptions.SSLError("intercepted")],
)
def test_network_failures_are_network(exc: Exception) -> None:
    session = FakeSession(lambda u, h, p: exc)
    result = providers.check_key("deepseek", KEY, session=session)
    assert result.status == "NETWORK"
    assert "api.deepseek.com" in result.message
    assert KEY not in result.message
    if isinstance(exc, requests.exceptions.Timeout):
        assert "15 seconds" in result.message


def test_anthropic_uses_its_own_headers_and_paginates() -> None:
    pages = [
        {"data": [{"id": "claude-sonnet-4-6-20260101"}], "has_more": True, "last_id": "x1"},
        {"data": [{"id": "claude-opus-4-6"}], "has_more": False},
    ]

    def handler(url: str, headers: dict[str, str], params: dict[str, Any]) -> FakeResponse:
        assert url == "https://api.anthropic.com/v1/models"
        assert headers["x-api-key"] == ANT_KEY
        assert headers["anthropic-version"] == "2023-06-01"
        assert "Authorization" not in headers
        return FakeResponse(200, json_data=pages[1] if params.get("after_id") else pages[0])

    result = providers.check_key("anthropic", ANT_KEY, session=FakeSession(handler))
    assert result.status == "OK"
    assert result.models == ["claude-sonnet-4-6-20260101", "claude-opus-4-6"]


def test_rejection_explains_a_key_for_the_wrong_service() -> None:
    session = FakeSession(lambda u, h, p: FakeResponse(401, json_data={}))
    result = providers.check_key("openai", ANT_KEY, session=session)
    assert result.status == "REJECTED"
    assert "Anthropic" in result.message
    assert "platform.openai.com/api-keys" in result.message


def test_empty_key_is_rejected_without_a_request() -> None:
    session = FakeSession(lambda u, h, p: FakeResponse(200, json_data=MODELS))
    assert providers.check_key("deepseek", "   ", session=session).status == "REJECTED"
    assert session.calls == []


def test_local_server_uses_placeholder_key_and_explains_nothing_listening() -> None:
    seen: list[dict[str, str]] = []

    def handler(url: str, headers: dict[str, str], params: dict[str, Any]) -> Any:
        seen.append(headers)
        assert url == "http://localhost:11434/v1/models"
        return requests.exceptions.ConnectionError("refused")

    result = providers.check_key("local", "", base_url="http://localhost:11434/v1/",
                                 session=FakeSession(handler))
    assert result.status == "NETWORK"
    assert "Ollama" in result.message
    assert seen[0]["Authorization"] == "Bearer local"


def test_local_server_with_no_models_says_so() -> None:
    session = FakeSession(lambda u, h, p: FakeResponse(200, json_data={"data": []}))
    result = providers.check_key("local", None, base_url="http://localhost:1234/v1",
                                 session=session)
    assert result.status == "OK"
    assert "no models" in result.message


def test_local_without_address_is_unknown_not_a_crash() -> None:
    result = providers.check_key("local", None, session=FakeSession(lambda u, h, p: None))
    assert result.status == "UNKNOWN"


def test_non_json_models_answer_is_unknown() -> None:
    session = FakeSession(lambda u, h, p: FakeResponse(200, body=b"<html>portal</html>"))
    result = providers.check_key("openai", KEY, base_url="https://proxy.example.com/v1",
                                 session=session)
    assert result.status == "UNKNOWN"
    assert "right address" in result.message


def test_unknown_provider_raises_value_error() -> None:
    with pytest.raises(ValueError):
        providers.check_key("nope", KEY)


def test_clean_key_strips_paste_debris() -> None:
    assert providers.clean_key(f' "{KEY}"\r\n') == KEY
    assert providers.clean_key(f"Bearer {KEY}") == KEY
    assert providers.clean_key(None) == ""


@pytest.mark.parametrize(("status", "expected"), [(200, "OK"), (403, "REJECTED"),
                                                   (401, "REJECTED"), (429, "UNKNOWN")])
def test_semantic_scholar_check(status: int, expected: str) -> None:
    session = FakeSession(lambda u, h, p: FakeResponse(status, json_data={"data": []}))
    result = providers.check_semantic_scholar(KEY, session=session)
    assert result.status == expected
    call = session.calls[0]
    assert call["url"] == providers.SEMANTIC_SCHOLAR_SEARCH_URL
    assert call["headers"] == {"x-api-key": KEY}
    assert call["params"]["limit"] == 1
    assert KEY not in result.message


def test_semantic_scholar_network() -> None:
    session = FakeSession(lambda u, h, p: requests.exceptions.ConnectTimeout("x"))
    assert providers.check_semantic_scholar(KEY, session=session).status == "NETWORK"


def test_missing_models_flags_retired_ids_and_accepts_dated_aliases() -> None:
    session = FakeSession(lambda u, h, p: FakeResponse(200, json_data=MODELS))
    missing = providers.missing_models(
        "deepseek", KEY,
        {"writer": "deepseek-v4-pro", "outline_agent": "deepseek-v4-flash",
         "verifier": "deepseek-v4-flash"},
        session=session,
    )
    assert missing == ["deepseek-v4-flash"]

    ant = FakeSession(lambda u, h, p: FakeResponse(200, json_data={
        "data": [{"id": "claude-sonnet-4-6-20260101"}], "has_more": False}))
    assert providers.missing_models("anthropic", ANT_KEY, ["claude-sonnet-4-6"],
                                    session=ant) == []


def test_missing_models_is_empty_when_the_list_cannot_be_fetched() -> None:
    session = FakeSession(lambda u, h, p: requests.exceptions.ConnectionError("x"))
    assert providers.missing_models("deepseek", KEY, ["anything"], session=session) == []


def test_default_models_follow_the_shipped_config() -> None:
    shipped = yaml.safe_load((REPO_ROOT / "config.yaml").read_text(encoding="utf-8"))
    assert providers.default_models("deepseek") == {
        k: str(v) for k, v in shipped["deepseek"]["models"].items()}
    assert providers.default_models("local") == {}
    openai_block = (shipped.get("openai") or {}).get("models") or {}
    assert providers.default_models("openai") == {k: str(v) for k, v in openai_block.items()}
    anthropic = providers.default_models("anthropic")
    expected = (shipped.get("anthropic") or {}).get("models") or shipped.get("models") or {}
    assert anthropic == {k: str(v) for k, v in expected.items()}


def test_default_models_fall_back_when_config_is_unreadable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    monkeypatch.setenv("EDMARS_APP_ROOT", str(tmp_path))
    from edmars import paths

    if getattr(paths, "__edmars_test_stub__", False) is False:
        monkeypatch.setattr(paths, "app_root", lambda: tmp_path)
    models = providers.default_models("deepseek")
    assert models["writer"] == "deepseek-v4-pro"
    assert models["outline_agent"] == "deepseek-flash"
    assert providers.default_models("openai") == {}


def test_catalog_is_complete() -> None:
    assert set(providers.PROVIDERS) == {"deepseek", "openai", "anthropic", "local"}
    assert providers.PROVIDERS["local"].env_var == "OPENAI_API_KEY"
    assert [p[1] for p in providers.PROVIDERS["local"].presets] == [
        "http://localhost:11434/v1", "http://localhost:1234/v1", "http://localhost:8000/v1"]
    assert providers.provider_choices()[0] == ("deepseek", "DeepSeek (recommended)")
