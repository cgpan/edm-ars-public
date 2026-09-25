"""Provider failures, waits and routing (defects D5, D6, E1, E2, E3).

Every test here runs offline: SDK clients are replaced by fakes whose
``create()`` raises the exception class the real SDK would raise, and
``time.sleep`` is recorded instead of slept.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httpx
import openai
import pytest

from src.agents.base import BaseAgent
from src.agents.llm_client import (
    NETWORK_BACKOFF_S,
    RATE_LIMIT_WAITS_S,
    classify_exception,
    llm_settings,
    resolve_base_url,
)
from src.agents.provider_resolver import (
    ProviderConfigError,
    resolve_provider_for_stage,
    resolve_revision_writer,
)
from src.context import PipelineContext
from src.errors import ProviderError, code_for_exception
from src.events import attach

ROOT = Path(__file__).resolve().parent.parent
_REQ = httpx.Request("POST", "https://api.deepseek.com/chat/completions")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class _Probe(BaseAgent):
    def run(self, **kwargs: Any) -> None:  # type: ignore[override]
        return None


def _config(provider: str = "deepseek", **extra: Any) -> dict:
    cfg: dict = {
        "llm_provider": provider,
        "models": {"critic": "claude-opus-4-6", "writer": "claude-sonnet-4-6",
                   "probe": "claude-sonnet-4-6"},
        "deepseek": {
            "base_url": "https://api.deepseek.com",
            "models": {"probe": "deepseek-v4-pro", "writer": "deepseek-v4-pro"},
        },
        "paths": {
            "data_registry": str(ROOT / "data_registry"),
            "agent_prompts": str(ROOT / "agent_prompts"),
        },
        "sandbox": {"enabled": False},
        "pricing": {"per_million_tokens": {
            "deepseek-v4-pro": {"input": 1.0, "cached_input": 0.1, "output": 2.0},
        }},
    }
    cfg.update(extra)
    return cfg


def _ctx(tmp_path: Path) -> PipelineContext:
    ctx = PipelineContext(
        dataset_name="hsls09_public",
        raw_data_path=str(tmp_path / "raw.csv"),
        output_dir=str(tmp_path),
    )
    attach(ctx, str(tmp_path))
    return ctx


def _agent(tmp_path: Path, config: dict | None = None, name: str = "Probe") -> _Probe:
    return _Probe(_ctx(tmp_path), name, config or _config())


def _events(tmp_path: Path, etype: str | None = None) -> list[dict]:
    path = tmp_path / "events.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return [r for r in rows if etype is None or r["type"] == etype]


def _status_error(cls: type, status: int, message: str, body: Any = None) -> Exception:
    return cls(message, response=httpx.Response(status, request=_REQ), body=body)


def _ok_response(text: str = "hello") -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=text))]
    resp.usage = MagicMock(
        prompt_tokens=1000, completion_tokens=500,
        prompt_cache_hit_tokens=0, prompt_tokens_details=None,
        completion_tokens_details=None,
    )
    return resp


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    recorded: list[float] = []
    monkeypatch.setattr("time.sleep", lambda s: recorded.append(s))
    return recorded


# ---------------------------------------------------------------------------
# D5 -- explicit timeouts, SDK retries off, classified failures
# ---------------------------------------------------------------------------


class TestTransportSettings:
    def test_defaults(self) -> None:
        s = llm_settings({})
        assert s.request_timeout_s == 600
        assert s.max_network_retries == 3

    def test_config_values_are_read(self) -> None:
        s = llm_settings({"llm": {"request_timeout_s": 120, "max_network_retries": 1}})
        assert (s.request_timeout_s, s.max_network_retries) == (120, 1)

    @pytest.mark.parametrize("block", [
        {"request_timeout_s": 0}, {"request_timeout_s": "soon"},
        {"max_network_retries": -1}, {"max_network_retries": 1.5},
    ])
    def test_unusable_values_are_a_config_error(self, block: dict) -> None:
        with pytest.raises(ProviderConfigError):
            llm_settings({"llm": block})

    def test_clients_get_the_timeout_and_no_hidden_retries(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict = {}

        def _fake(**kwargs: Any) -> MagicMock:
            captured.update(kwargs)
            return MagicMock()

        monkeypatch.setattr(openai, "OpenAI", _fake)
        _agent(tmp_path, _config(llm={"request_timeout_s": 90}))
        timeout = captured["timeout"]
        assert isinstance(timeout, httpx.Timeout)
        assert timeout.read == 90
        assert timeout.connect <= 10
        assert captured["max_retries"] == 0

    def test_anthropic_client_too(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import anthropic

        captured: dict = {}
        monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: captured.update(kw) or MagicMock())
        agent = _agent(tmp_path, _config("anthropic"))
        assert agent.model == "claude-sonnet-4-6"
        assert captured["timeout"].read == 600
        assert captured["max_retries"] == 0


class TestClassifiedFailures:
    def _call_with(self, tmp_path: Path, side_effect: Any, **cfg: Any) -> _Probe:
        agent = _agent(tmp_path, _config(**cfg))
        agent.client = MagicMock()
        agent.client.chat.completions.create.side_effect = side_effect
        return agent

    def test_rejected_key_is_not_retried_and_names_the_variable(
        self, tmp_path: Path, sleeps: list[float]
    ) -> None:
        agent = self._call_with(tmp_path, _status_error(
            openai.AuthenticationError, 401, "Authentication Fails (invalid key)"))
        with pytest.raises(ProviderError) as info:
            agent.call_llm("hi")
        assert info.value.code == "KEY_REJECTED"
        assert "DEEPSEEK_API_KEY" in str(info.value)
        assert code_for_exception(info.value) == "KEY_REJECTED"
        assert sleeps == []
        assert agent.client.chat.completions.create.call_count == 1

    def test_deepseek_insufficient_balance_is_no_credit(
        self, tmp_path: Path, sleeps: list[float]
    ) -> None:
        agent = self._call_with(tmp_path, _status_error(
            openai.APIStatusError, 402, "Insufficient Balance",
            body={"error": {"message": "Insufficient Balance"}}))
        with pytest.raises(ProviderError) as info:
            agent.call_llm("hi")
        assert info.value.code == "NO_CREDIT"
        assert "Top up" in str(info.value)
        assert sleeps == []

    def test_openai_insufficient_quota_429_is_no_credit_not_a_wait(
        self, tmp_path: Path, sleeps: list[float]
    ) -> None:
        agent = self._call_with(tmp_path, _status_error(
            openai.RateLimitError, 429, "You exceeded your current quota",
            body={"error": {"code": "insufficient_quota"}}))
        with pytest.raises(ProviderError) as info:
            agent.call_llm("hi")
        assert info.value.code == "NO_CREDIT"
        assert sleeps == []

    @pytest.mark.parametrize("exc", [
        _status_error(openai.NotFoundError, 404, "The model does not exist"),
        _status_error(openai.BadRequestError, 400, "Model Not Exist",
                      body={"error": {"message": "Model Not Exist"}}),
    ])
    def test_retired_model_names_the_config_key(
        self, tmp_path: Path, sleeps: list[float], exc: Exception
    ) -> None:
        agent = self._call_with(tmp_path, exc)
        with pytest.raises(ProviderError) as info:
            agent.call_llm("hi")
        assert info.value.code == "MODEL_GONE"
        assert "deepseek.models.probe" in str(info.value)
        assert info.value.model == "deepseek-v4-pro"
        assert sleeps == []

    def test_other_bad_request_is_provider_error_without_retry(
        self, tmp_path: Path, sleeps: list[float]
    ) -> None:
        agent = self._call_with(tmp_path, _status_error(
            openai.BadRequestError, 400, "max_tokens is too large"))
        with pytest.raises(ProviderError) as info:
            agent.call_llm("hi")
        assert info.value.code == "PROVIDER_ERROR"
        assert sleeps == []

    def test_connection_blips_are_waited_out(
        self, tmp_path: Path, sleeps: list[float]
    ) -> None:
        agent = self._call_with(tmp_path, [
            openai.APIConnectionError(request=_REQ),
            openai.APIConnectionError(request=_REQ),
            _ok_response("recovered"),
        ])
        assert agent.call_llm("hi") == "recovered"
        assert sleeps == list(NETWORK_BACKOFF_S[:2])

    def test_network_retries_are_bounded(
        self, tmp_path: Path, sleeps: list[float]
    ) -> None:
        agent = self._call_with(
            tmp_path, openai.APIConnectionError(request=_REQ),
            llm={"max_network_retries": 2},
        )
        with pytest.raises(ProviderError) as info:
            agent.call_llm("hi")
        assert info.value.code == "NETWORK"
        assert "https://api.deepseek.com" in str(info.value)
        assert len(sleeps) == 2
        assert agent.client.chat.completions.create.call_count == 3

    def test_timeouts_end_as_timeout(self, tmp_path: Path, sleeps: list[float]) -> None:
        agent = self._call_with(
            tmp_path, openai.APITimeoutError(request=_REQ),
            llm={"max_network_retries": 1, "request_timeout_s": 45},
        )
        with pytest.raises(ProviderError) as info:
            agent.call_llm("hi")
        assert info.value.code == "TIMEOUT"
        assert "45 s" in str(info.value)
        assert len(sleeps) == 1

    def test_server_errors_are_retried_then_reported(
        self, tmp_path: Path, sleeps: list[float]
    ) -> None:
        agent = self._call_with(
            tmp_path,
            _status_error(openai.InternalServerError, 503, "Server overloaded"),
            llm={"max_network_retries": 1},
        )
        with pytest.raises(ProviderError) as info:
            agent.call_llm("hi")
        assert info.value.code == "PROVIDER_ERROR"
        assert info.value.status == 503
        assert len(sleeps) == 1

    def test_rate_limit_keeps_its_60_then_120_schedule(
        self, tmp_path: Path, sleeps: list[float]
    ) -> None:
        agent = self._call_with(
            tmp_path, _status_error(openai.RateLimitError, 429, "Rate limit reached"))
        with pytest.raises(ProviderError) as info:
            agent.call_llm("hi")
        assert info.value.code == "RATE_LIMITED"
        assert sleeps == list(RATE_LIMIT_WAITS_S) == [60, 120]
        assert agent.client.chat.completions.create.call_count == 3

    def test_non_provider_exceptions_pass_through_untouched(
        self, tmp_path: Path, sleeps: list[float]
    ) -> None:
        agent = self._call_with(tmp_path, ValueError("a bug of ours"))
        with pytest.raises(ValueError, match="a bug of ours"):
            agent.call_llm("hi")
        assert sleeps == []

    def test_anthropic_overloaded_stream_error_is_transient(self) -> None:
        import anthropic

        exc = anthropic.APIStatusError(
            "Overloaded",
            response=httpx.Response(200, request=_REQ),
            body={"type": "error", "error": {"type": "overloaded_error"}},
        )
        assert classify_exception(exc) == ("transient", "PROVIDER_ERROR")

    def test_mid_stream_transport_errors_are_network(self) -> None:
        assert classify_exception(httpx.RemoteProtocolError("peer closed")) == (
            "transient", "NETWORK")
        assert classify_exception(httpx.ReadTimeout("slow")) == ("transient", "TIMEOUT")


# ---------------------------------------------------------------------------
# D6 -- a wait is visible while it happens; CONTRACT section 2 events
# ---------------------------------------------------------------------------


class TestWaitsAndEvents:
    def test_rate_limit_wait_is_on_disk_before_the_sleep(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _agent(tmp_path)
        agent.client = MagicMock()
        agent.client.chat.completions.create.side_effect = [
            _status_error(openai.RateLimitError, 429, "Rate limit reached"),
            _ok_response("ok"),
        ]
        seen_at_sleep: dict = {}

        def _sleep(seconds: float) -> None:
            seen_at_sleep["log"] = (tmp_path / "pipeline.log").read_text(encoding="utf-8")
            seen_at_sleep["waits"] = _events(tmp_path, "llm.wait")

        monkeypatch.setattr("time.sleep", _sleep)
        assert agent.call_llm("hi") == "ok"

        line = seen_at_sleep["log"].strip().splitlines()[-1]
        # The orchestrator's exact format: "<iso timestamp> [<agent>] <message>"
        timestamp, rest = line.split(" ", 1)
        assert "T" in timestamp
        assert rest.startswith("[Probe] Rate limit hit (attempt 1/3); waiting 60s")
        (wait,) = seen_at_sleep["waits"]
        assert wait["data"]["seconds"] == 60
        assert wait["data"]["attempt"] == 1
        assert wait["data"]["reason"] == "rate_limit"

    def test_llm_start_and_end_carry_tokens_and_cost(self, tmp_path: Path) -> None:
        agent = _agent(tmp_path)
        agent.client = MagicMock()
        agent.client.chat.completions.create.return_value = _ok_response()
        agent.call_llm("hi")
        (start,) = _events(tmp_path, "llm.start")
        (end,) = _events(tmp_path, "llm.end")
        assert start["data"] == {"model": "deepseek-v4-pro", "provider": "deepseek"}
        assert start["agent"] == "Probe"
        data = end["data"]
        assert data["ok"] is True
        assert (data["prompt_tokens"], data["completion_tokens"]) == (1000, 500)
        assert data["cached_tokens"] == 0
        # 1000 * $1/M + 500 * $2/M
        assert data["cost_usd"] == pytest.approx(0.002)
        assert data["cost_estimated"] is False
        assert data["duration_s"] >= 0

    def test_failed_call_still_closes_with_llm_end(
        self, tmp_path: Path, sleeps: list[float]
    ) -> None:
        agent = _agent(tmp_path)
        agent.client = MagicMock()
        agent.client.chat.completions.create.side_effect = _status_error(
            openai.APIStatusError, 402, "Insufficient Balance")
        with pytest.raises(ProviderError):
            agent.call_llm("hi")
        (end,) = _events(tmp_path, "llm.end")
        assert end["data"]["ok"] is False
        assert end["data"]["error_code"] == "NO_CREDIT"
        assert end["data"]["cost_usd"] is None

    def test_deepseek_happy_path_parameters_are_unchanged(self, tmp_path: Path) -> None:
        agent = _agent(tmp_path)
        agent.client = MagicMock()
        agent.client.chat.completions.create.return_value = _ok_response()
        agent.call_llm("hi", max_tokens=123)
        kwargs = agent.client.chat.completions.create.call_args.kwargs
        assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
        assert kwargs["max_tokens"] == 123
        assert kwargs["model"] == "deepseek-v4-pro"

    def test_attempt_events_name_the_failure(self, tmp_path: Path) -> None:
        agent = _agent(tmp_path)
        agent.execute_code = MagicMock(return_value={  # type: ignore[method-assign]
            "returncode": 1, "stdout": "",
            "stderr": "Traceback (most recent call last):\n  File x\nKeyError: 'X1SES'\n",
        })
        agent.execute_code_attempt("print(1)", attempt=2, max_attempts=3)
        agent.execute_code.assert_called_once_with("print(1)")
        (start,) = _events(tmp_path, "attempt.start")
        (end,) = _events(tmp_path, "attempt.end")
        assert start["data"]["attempt"] == 2 and start["data"]["max_attempts"] == 3
        assert start["data"]["timeout_s"] == 300
        assert end["data"]["returncode"] == 1
        assert end["data"]["error_class"] == "KeyError"

    def test_attempt_events_report_a_timeout(self, tmp_path: Path) -> None:
        agent = _agent(tmp_path)
        agent.execute_code = MagicMock(return_value={  # type: ignore[method-assign]
            "returncode": -1, "stdout": "", "stderr": "Timeout after 600s"})
        agent.execute_code_attempt("x", attempt=1, max_attempts=3, timeout_s=600)
        agent.execute_code.assert_called_once_with("x", timeout_s=600)
        (end,) = _events(tmp_path, "attempt.end")
        assert end["data"]["error_class"] == "Timeout"
        assert end["data"]["timeout_s"] == 600


# ---------------------------------------------------------------------------
# E1 -- no silent gpt-4o
# ---------------------------------------------------------------------------


class TestOpenAIModelRequired:
    def test_missing_openai_model_is_a_named_config_error(self, tmp_path: Path) -> None:
        cfg = _config("openai")
        with pytest.raises(ProviderConfigError, match=r"openai\.models\.critic"):
            _Probe(_ctx(tmp_path), "Critic", cfg)

    def test_shipped_config_switched_to_openai_fails_loudly(
        self, tmp_path: Path
    ) -> None:
        from src.config import load_config

        cfg = load_config(str(ROOT / "config.yaml"))
        cfg["llm_provider"] = "openai"
        with pytest.raises(ProviderConfigError, match="gpt-4o"):
            _Probe(_ctx(tmp_path), "Critic", cfg)

    def test_configured_openai_model_is_used(self, tmp_path: Path) -> None:
        cfg = _config("openai", openai={"models": {"critic": "gpt-5.4"}})
        agent = _Probe(_ctx(tmp_path), "Critic", cfg)
        assert agent.model == "gpt-5.4"
        assert agent._provider == "openai"

    def test_deepseek_keeps_its_default(self, tmp_path: Path) -> None:
        agent = _Probe(_ctx(tmp_path), "Unlisted", _config())
        assert agent.model == "deepseek-v4-pro"

    def test_missing_model_is_reported_before_a_missing_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ProviderConfigError, match=r"openai\.models\.critic"):
            _Probe(_ctx(tmp_path), "Critic", _config("openai"))


# ---------------------------------------------------------------------------
# E2 -- the gate's reviser uses a model the active provider serves
# ---------------------------------------------------------------------------


class TestRevisionModel:
    def _gate(self, tmp_path: Path, cfg: dict) -> Any:
        from src.review_gate import ReviewGate

        logs: list[str] = []
        gate = ReviewGate(cfg, str(tmp_path), log_fn=lambda _a, m: logs.append(m))
        gate._test_logs = logs
        return gate

    def test_openai_uses_the_writer_model_not_the_deepseek_id(self, tmp_path: Path) -> None:
        from src.config import load_config

        cfg = load_config(str(ROOT / "config.yaml"))
        cfg["llm_provider"] = "openai"
        cfg["openai"] = {"models": {"writer": "gpt-5.4"}}
        gate = self._gate(tmp_path, cfg)
        assert gate._llm_provider == "openai"
        assert gate._llm_model == "gpt-5.4"
        assert cfg["review_gate"]["revision_model"] == "deepseek-v4-pro"

    def test_revision_writer_entry_wins(self, tmp_path: Path) -> None:
        cfg = _config("openai", openai={"models": {"writer": "gpt-5.4",
                                                     "revision_writer": "gpt-5.4-mini"}})
        assert self._gate(tmp_path, cfg)._llm_model == "gpt-5.4-mini"

    def test_per_stage_revision_writer_is_honoured(self, tmp_path: Path) -> None:
        cfg = _config("deepseek", per_stage_providers={
            "revision_writer": {"provider": "openai", "model": "gpt-5.4"}})
        gate = self._gate(tmp_path, cfg)
        assert (gate._llm_provider, gate._llm_model) == ("openai", "gpt-5.4")

    def test_no_model_disables_revision_and_says_why(self, tmp_path: Path) -> None:
        cfg = _config("openai", review_gate={"revision_model": "deepseek-v4-pro"})
        gate = self._gate(tmp_path, cfg)
        assert gate._llm_model == ""
        assert gate._llm_client is None
        assert "openai.models.revision_writer" in gate.revision_unavailable_reason
        assert any("revision disabled" in m for m in gate._test_logs)
        assert gate._call_revision_llm("prompt") is None

    def test_malformed_reviser_setting_disables_revision_only(self, tmp_path: Path) -> None:
        cfg = _config("deepseek", per_stage_providers={
            "revision_writer": {"provider": "nope", "model": "x"}})
        gate = self._gate(tmp_path, cfg)
        assert gate._llm_client is None
        assert "invalid reviser configuration" in gate.revision_unavailable_reason
        assert gate._call_revision_llm("prompt") is None

    def test_deepseek_still_falls_back_to_revision_model(self) -> None:
        cfg = {"llm_provider": "deepseek", "review_gate": {"revision_model": "deepseek-flash"}}
        assert resolve_revision_writer(cfg).model == "deepseek-flash"

    def test_revision_failure_is_classified_and_kept(
        self, tmp_path: Path, sleeps: list[float]
    ) -> None:
        gate = self._gate(tmp_path, _config())
        gate._llm_client = MagicMock()
        gate._llm_client.chat.completions.create.side_effect = _status_error(
            openai.APIStatusError, 402, "Insufficient Balance")
        assert gate._call_revision_llm("prompt") is None
        assert gate.revision_failures[0]["code"] == "NO_CREDIT"
        assert any("[NO_CREDIT]" in m for m in gate._test_logs)
        assert sleeps == []


# ---------------------------------------------------------------------------
# E3 -- one base-URL rule for agents and gate
# ---------------------------------------------------------------------------


class TestBaseUrlPrecedence:
    def test_env_var_beats_the_shipped_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The shipped config sets deepseek.base_url explicitly, which used
        # to make DEEPSEEK_BASE_URL dead for every stage.
        monkeypatch.setenv("DEEPSEEK_BASE_URL", "http://proxy.example/v1")
        agent = _agent(tmp_path, _config())
        assert str(agent.client.base_url).rstrip("/") == "http://proxy.example/v1"

    def test_the_gate_goes_where_the_agents_go(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.review_gate import ReviewGate

        monkeypatch.setenv("DEEPSEEK_BASE_URL", "http://proxy.example/v1")
        cfg = _config()
        agent = _agent(tmp_path, cfg)
        gate = ReviewGate(cfg, str(tmp_path), log_fn=None)
        assert str(gate._llm_client.base_url) == str(agent.client.base_url)
        monkeypatch.delenv("DEEPSEEK_BASE_URL")
        del cfg["deepseek"]["base_url"]
        gate = ReviewGate(cfg, str(tmp_path), log_fn=None)
        assert str(gate._llm_client.base_url).rstrip("/") == "https://api.deepseek.com"

    def test_config_used_when_env_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
        cfg = _config(deepseek={"base_url": "https://cfg.example", "models": {}})
        assert resolve_base_url(resolve_provider_for_stage("probe", cfg)) == "https://cfg.example"

    def test_per_stage_base_url_names_one_stage_and_wins(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_BASE_URL", "http://env.example/v1")
        cfg = _config(per_stage_providers={"probe": {
            "provider": "openai", "model": "local", "base_url": "http://stage.example/v1"}})
        assert resolve_base_url(resolve_provider_for_stage("probe", cfg)) == "http://stage.example/v1"

    def test_minimax_env_var_is_live_again(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MINIMAX_BASE_URL", "http://mm.example")
        cfg = {"llm_provider": "minimax", "minimax": {"base_url": "https://api.minimax.io/anthropic"}}
        assert resolve_base_url(resolve_provider_for_stage("probe", cfg)) == "http://mm.example"


class TestRoutingTable:
    """A pre-flight can show every stage's route without building clients."""

    def test_shipped_config_routes_every_stage(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.agents.llm_client import describe_routing
        from src.config import load_config

        monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
        rows = {r["stage"]: r for r in describe_routing(load_config(str(ROOT / "config.yaml")))}
        assert rows["critic"]["model"] == "deepseek-v4-pro"
        assert rows["outline_agent"]["model"] == "deepseek-flash"
        assert rows["revision_writer"]["model"] == "deepseek-v4-pro"
        assert all(r["error"] is None for r in rows.values())
        assert rows["writer"]["base_url"] == "https://api.deepseek.com"
        assert rows["writer"]["key_env"] == "DEEPSEEK_API_KEY"

    def test_openai_without_models_reports_every_gap(self) -> None:
        from src.agents.llm_client import describe_routing
        from src.config import load_config

        cfg = load_config(str(ROOT / "config.yaml"))
        cfg["llm_provider"] = "openai"
        rows = describe_routing(cfg)
        assert all(r["error"] for r in rows)
        assert "openai.models.critic" in next(r for r in rows if r["stage"] == "critic")["error"]

    def test_anthropic_stage_without_a_model_is_flagged(self) -> None:
        from src.agents.llm_client import describe_routing

        cfg = {"llm_provider": "anthropic",
               "models": {"critic": "claude-opus-4-6", "writer": "claude-sonnet-4-6"}}
        rows = {r["stage"]: r for r in describe_routing(cfg, ("critic", "outline_agent"))}
        assert rows["critic"]["error"] is None
        assert "models.outline_agent" in rows["outline_agent"]["error"]
