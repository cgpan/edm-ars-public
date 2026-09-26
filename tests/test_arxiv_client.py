"""The arXiv client: address, headers, pacing and refusals.

On a Mac every arXiv query came back HTTP 406, and the user saw only
"arXiv failed (0 papers)". arXiv's front end answers 406 with an empty
body to some clients whose request misses its cache, whatever headers
they send. These tests pin what the client can control: it asks the
https address (the http one answers 301, so each query cost two
requests), identifies itself, waits three seconds between requests as
arXiv's API terms ask, stops asking once refused, and carries the HTTP
status into ``retrieval_status``, ``lit.progress`` and the warning.

The HTTP layer is a fake; nothing here touches the network.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.agents.problem_formulator import ProblemFormulator
from src.context import PipelineContext
from src.orchestrator import _literature_warning
from tests.test_literature_status import _agent, _events

_FEED = """<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/{aid}v1</id>
    <title>Predicting college enrollment {aid}</title>
    <summary>An abstract.</summary>
    <published>2024-05-01T00:00:00Z</published>
    <author><name>A. Author</name></author>
  </entry>
</feed>
"""


def _resp(status: int, text: str = "") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.text = text
    resp.json.return_value = {}
    return resp


class FakeHTTP:
    """Scripted responses for arXiv; Semantic Scholar always 429."""

    def __init__(self, arxiv: list[Any]) -> None:
        self.arxiv = list(arxiv)
        self.arxiv_calls: list[dict[str, Any]] = []

    def __call__(self, url: str, **kwargs: Any) -> MagicMock:
        if "semanticscholar" in url:
            return _resp(429)
        self.arxiv_calls.append({"url": url, **kwargs})
        step = self.arxiv.pop(0)
        if isinstance(step, BaseException):
            raise step
        if step == 200:
            return _resp(200, _FEED.format(aid=f"2405.0{len(self.arxiv_calls)}"))
        return _resp(step)


def _arxiv_progress(tmp_path: Path) -> list[dict]:
    return [e for e in _events(tmp_path, "lit.progress") if e["data"]["source"] == "arxiv"]


QUERIES = ["q one", "q two", "q three"]


class TestRequest:
    def test_https_address_and_explicit_headers(self, tmp_path: Path) -> None:
        agent = _agent(tmp_path)
        fake = FakeHTTP([200])
        with patch("requests.get", side_effect=fake), patch("time.sleep"):
            papers = agent._search_arxiv(["college enrollment"], max_results_per_query=5)

        (call,) = fake.arxiv_calls
        assert call["url"] == "https://export.arxiv.org/api/query"
        assert call["params"]["search_query"] == "all:college enrollment"
        assert call["params"]["max_results"] == 5
        assert call["headers"]["User-Agent"].startswith("EDM-ARS (+https://")
        assert "application/atom+xml" in call["headers"]["Accept"]
        assert call["timeout"] == 15
        assert [p["paperId"] for p in papers] == ["arxiv:2405.01v1"]
        assert agent._arxiv_query_outcomes == ["ok"]

    def test_three_seconds_between_requests(self, tmp_path: Path) -> None:
        agent = _agent(tmp_path)
        fake = FakeHTTP([200, 200, 200])
        with patch("requests.get", side_effect=fake), patch("time.sleep") as sleep:
            agent._search_arxiv(QUERIES)
        assert len(fake.arxiv_calls) == 3
        # No wait before the first request, three seconds before each other.
        assert [c.args[0] for c in sleep.call_args_list] == [3.0, 3.0]


class TestRefusal:
    def test_a_406_stops_the_search_and_is_recorded(self, tmp_path: Path) -> None:
        agent = _agent(tmp_path)
        fake = FakeHTTP([406, 200, 200])
        with patch("requests.get", side_effect=fake), patch("time.sleep") as sleep:
            papers = agent._search_arxiv(QUERIES)

        assert papers == []
        assert len(fake.arxiv_calls) == 1  # the other two were never sent
        sleep.assert_not_called()
        assert agent._arxiv_query_outcomes == ["refused", "skipped", "skipped"]
        assert agent._arxiv_http_status == 406

        rows = _arxiv_progress(tmp_path)
        assert [r["data"]["status"] for r in rows] == ["refused", "skipped", "skipped"]
        assert rows[0]["data"]["http_status"] == 406
        assert rows[0]["plain"] == "arXiv refused the request (HTTP 406)"
        assert all("http_status" not in r["data"] for r in rows[1:])
        assert [r["data"]["query_index"] for r in rows] == [1, 2, 3]

        log = (tmp_path / "pipeline.log").read_text(encoding="utf-8")
        assert "arXiv refused query 1/3 'q one' with HTTP 406; " \
               "not sending the other 2 arXiv queries" in log

    @pytest.mark.parametrize("code", [403, 429])
    def test_other_refusals(self, tmp_path: Path, code: int) -> None:
        agent = _agent(tmp_path)
        fake = FakeHTTP([code, 200])
        with patch("requests.get", side_effect=fake), patch("time.sleep"):
            agent._search_arxiv(QUERIES[:2])
        assert len(fake.arxiv_calls) == 1
        assert agent._arxiv_query_outcomes == ["refused", "skipped"]

    def test_a_server_error_is_not_a_refusal(self, tmp_path: Path) -> None:
        agent = _agent(tmp_path)
        fake = FakeHTTP([500, 200])
        with patch("requests.get", side_effect=fake), patch("time.sleep"):
            papers = agent._search_arxiv(QUERIES[:2])
        assert len(fake.arxiv_calls) == 2
        assert agent._arxiv_query_outcomes == ["failed", "ok"]
        assert len(papers) == 1
        rows = _arxiv_progress(tmp_path)
        assert rows[0]["data"]["http_status"] == 500
        assert rows[0]["data"]["status"] == "failed"

    def test_a_network_error_has_no_status(self, tmp_path: Path) -> None:
        agent = _agent(tmp_path)
        fake = FakeHTTP([requests.ConnectionError("no route"), 200])
        with patch("requests.get", side_effect=fake), patch("time.sleep"):
            agent._search_arxiv(QUERIES[:2])
        assert agent._arxiv_query_outcomes == ["failed", "ok"]
        assert agent._arxiv_http_status is None
        assert "http_status" not in _arxiv_progress(tmp_path)[0]["data"]


class TestWarning:
    """What the user reads when the search came back empty."""

    def _search(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, arxiv: list[Any]
    ) -> tuple[ProblemFormulator, dict]:
        monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
        agent = _agent(tmp_path)
        with patch("requests.get", side_effect=FakeHTTP(arxiv)), patch("time.sleep"):
            result = agent._search_literature(None)
        return agent, result

    def test_the_warning_says_arxiv_refused_with_its_status(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, result = self._search(tmp_path, monkeypatch, [406])
        status = result["retrieval_status"]
        assert status["arxiv"] == "refused"
        assert status["arxiv_http_status"] == 406
        assert status["degraded"] is True

        (warning,) = _events(tmp_path, "warning")
        message = warning["data"]["message"]
        assert "arXiv refused the request (HTTP 406, 0 papers)" in message
        assert "arXiv failed" not in message
        assert warning["data"]["retrieval_status"]["arxiv_http_status"] == 406

        # The orchestrator's run-level sentence carries the status too.
        ctx = PipelineContext(dataset_name="hsls09_public", raw_data_path="x",
                              output_dir=str(tmp_path))
        ctx.literature_context = {"papers": [], "retrieval_status": status}
        sentence = _literature_warning(ctx)
        assert sentence is not None
        assert "arxiv=refused" in sentence and "arxiv_http_status=406" in sentence

    def test_an_http_failure_is_named_as_a_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, result = self._search(tmp_path, monkeypatch, [502, 502, 502])
        status = result["retrieval_status"]
        assert status["arxiv"] == "failed"
        assert status["arxiv_http_status"] == 502
        (warning,) = _events(tmp_path, "warning")
        assert "arXiv failed (HTTP 502, 0 papers)" in warning["data"]["message"]

    def test_a_refusal_after_a_failure_is_named_by_its_own_status(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, result = self._search(tmp_path, monkeypatch, [500, 406])
        status = result["retrieval_status"]
        assert status["arxiv"] == "refused"
        assert status["arxiv_http_status"] == 406
        (warning,) = _events(tmp_path, "warning")
        assert "arXiv refused the request (HTTP 406, 0 papers)" in warning["data"]["message"]

    def test_a_partly_answered_search_carries_no_status(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, result = self._search(tmp_path, monkeypatch, [500, 200, 200])
        status = result["retrieval_status"]
        assert status["arxiv"] == "ok"
        assert "arxiv_http_status" not in status
