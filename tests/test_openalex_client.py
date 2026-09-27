"""OpenAlex as a stand-in for an arXiv that refuses Python clients.

On the Mac test arXiv's front end answered HTTP 406 to every Python HTTP
client (requests, urllib, httpx) while curl got 200, so the literature
search lost arXiv entirely, twice. These tests pin the OpenAlex client
that is asked instead: when it is asked (only when arXiv did not answer,
unless ``openalex.when: always``), what it sends (the same search words,
no contact email, a key only in a header), how a work becomes a pool
record (a BibTeX-safe ``paperId``, the abstract rebuilt from OpenAlex's
inverted index, the DOI, the arXiv id of a preprint), how the records
are deduplicated against the pool, and what ``retrieval_status``,
``lit.progress`` and the warnings say.

The HTTP layer is a fake; nothing here touches the network.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests
import yaml

from src.agents.problem_formulator import (
    ProblemFormulator,
    _openalex_abstract,
    _openalex_paper,
)
from src.citations import build_bib_entry, sanitize_key
from src.context import PipelineContext
from src.orchestrator import _literature_warning
from tests.test_literature_status import ROOT, _agent, _events

QUERIES = ["q one", "q two", "q three"]
OPENALEX_URL = "https://api.openalex.org/works"

_ARXIV_FEED = """<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2405.00001v1</id>
    <title>An arXiv preprint on dropout</title>
    <summary>Abstract.</summary>
    <published>2024-05-01T00:00:00Z</published>
    <author><name>A. Author</name></author>
  </entry>
</feed>
"""


def _work(n: int, title: str | None = None, doi: str | None = None, **extra: Any) -> dict:
    """One OpenAlex work as the API returns it (fields abridged from a
    live response to the pipeline's own request)."""
    work: dict[str, Any] = {
        "id": f"https://openalex.org/W{4293192350 + n}",
        "doi": f"https://doi.org/{doi}" if doi else None,
        "display_name": title or f"Predicting college enrollment, study {n}",
        "publication_year": 2022,
        "publication_date": "2022-01-01",
        "type": "article",
        "authorships": [
            {"author": {"display_name": "Arturo Patungan"}},
            {"author": {"display_name": "Mari Francia"}},
        ],
        "primary_location": {
            "landing_page_url": f"https://doi.org/{doi}" if doi else None,
            "source": {"display_name": "Journal of Advanced Academics", "type": "journal"},
            "raw_source_name": "Journal of Advanced Academics",
        },
        "abstract_inverted_index": {"Predicting": [0], "enrollment": [1], "matters.": [2]},
        "cited_by_count": 4,
        "open_access": {"is_oa": False},
    }
    work.update(extra)
    return work


def _resp(status: int, payload: dict | None = None, text: str = "") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = payload or {}
    resp.text = text
    return resp


class FakeHTTP:
    """Semantic Scholar, arXiv and OpenAlex, each with its own script.

    A script step is an HTTP status (200 answers with the source's own
    body), a dict (an OpenAlex 200 body) or an exception to raise.
    """

    def __init__(
        self,
        arxiv: list[Any] | None = None,
        openalex: list[Any] | None = None,
        s2: dict | None = None,
    ) -> None:
        self.arxiv = list(arxiv or [])
        self.openalex = list(openalex or [])
        self.s2 = s2
        self.calls: dict[str, list[dict[str, Any]]] = {"s2": [], "arxiv": [], "openalex": []}

    def __call__(self, url: str, **kwargs: Any) -> MagicMock:
        if "semanticscholar" in url:
            self.calls["s2"].append({"url": url, **kwargs})
            return _resp(200, self.s2) if self.s2 is not None else _resp(429)
        if "arxiv" in url:
            self.calls["arxiv"].append({"url": url, **kwargs})
            step = self.arxiv.pop(0) if self.arxiv else 503
            if isinstance(step, BaseException):
                raise step
            return _resp(200, text=_ARXIV_FEED) if step == 200 else _resp(step)
        if "openalex" in url:
            self.calls["openalex"].append({"url": url, **kwargs})
            step = self.openalex.pop(0) if self.openalex else 503
            if isinstance(step, BaseException):
                raise step
            if isinstance(step, dict):
                return _resp(200, step)
            return _resp(step, {"results": []})
        raise AssertionError(f"unexpected request to {url}")


def _oa_progress(tmp_path: Path) -> list[dict]:
    return [e for e in _events(tmp_path, "lit.progress") if e["data"]["source"] == "openalex"]


def _page(*works: dict) -> dict:
    return {"meta": {"count": len(works)}, "results": list(works)}


def _search(
    tmp_path: Path,
    fake: FakeHTTP,
    queries: list[str] | None = None,
    **openalex_cfg: Any,
) -> tuple[ProblemFormulator, dict]:
    agent = _agent(tmp_path)
    agent._generate_search_queries = MagicMock(  # type: ignore[method-assign]
        return_value=list(queries or QUERIES))
    if openalex_cfg:
        agent.config["openalex"] = {**agent.config.get("openalex", {}), **openalex_cfg}
    with patch("requests.get", side_effect=fake), patch("time.sleep"):
        result = agent._search_literature("predict college enrollment")
    return agent, result


# ---------------------------------------------------------------------------
# A work becomes a pool record
# ---------------------------------------------------------------------------


class TestRecord:
    def test_a_work_maps_to_the_pool_record_shape(self) -> None:
        paper = _openalex_paper(_work(7, doi="10.1177/1932202X211064543"), "q one", 3)
        assert paper is not None
        assert paper["paperId"] == "openalex_W4293192357"
        assert paper["title"] == "Predicting college enrollment, study 7"
        assert paper["authors"] == ["Arturo Patungan", "Mari Francia"]
        assert paper["year"] == 2022
        assert paper["abstract"] == "Predicting enrollment matters."
        assert paper["venue"] == "Journal of Advanced Academics"
        assert paper["doi"] == "10.1177/1932202X211064543"
        assert paper["citationCount"] == 4
        assert paper["publicationTypes"] == ["JournalArticle"]
        assert paper["matched_query"] == "q one"
        assert paper["retrieval_rank"] == 3
        assert paper["source"] == "openalex"
        assert "arxiv_id" not in paper

    def test_the_paper_id_is_already_a_bibtex_key(self) -> None:
        paper = _openalex_paper(_work(1, doi="10.1177/1932202X211064543"), "q", 0)
        assert paper is not None
        assert sanitize_key(paper["paperId"]) == paper["paperId"]
        entry = build_bib_entry(paper)
        assert entry.startswith("@article{openalex_W4293192351,")
        assert "journal = {Journal of Advanced Academics}" in entry
        assert "doi       = {10.1177/1932202X211064543}" in entry

    def test_the_abstract_is_rebuilt_in_word_order(self) -> None:
        inverted = {"of": [2, 5], "The": [0], "role": [1], "belonging": [3],
                    "in": [4], "enrollment.": [6]}
        assert _openalex_abstract(inverted) == "The role of belonging in of enrollment."
        assert _openalex_abstract(None) == ""
        assert _openalex_abstract({"word": "not a list"}) == ""

    def test_an_arxiv_preprint_carries_its_arxiv_id(self) -> None:
        work = _work(
            2, title="Next-Term Student Performance Prediction",
            doi="10.48550/arxiv.1604.01840", type="preprint",
            primary_location={
                "landing_page_url": "https://arxiv.org/abs/1604.01840",
                "source": {"display_name": "arXiv (Cornell University)", "type": "repository"},
            },
        )
        paper = _openalex_paper(work, "q", 0)
        assert paper is not None
        assert paper["arxiv_id"] == "1604.01840"
        assert paper["publicationTypes"] == []
        assert build_bib_entry(paper).startswith("@misc{openalex_W4293192352,")

    def test_an_arxiv_id_is_read_from_the_landing_page_without_a_doi(self) -> None:
        work = _work(3, primary_location={
            "landing_page_url": "http://arxiv.org/abs/2401.12345v2",
            "source": {"display_name": "arXiv (Cornell University)", "type": "repository"},
        })
        paper = _openalex_paper(work, "q", 0)
        assert paper is not None and paper["arxiv_id"] == "2401.12345v2"

    @pytest.mark.parametrize(("work_type", "source_type", "expected"), [
        ("conference-paper", "journal", ["Conference"]),
        ("article", "conference", ["Conference"]),
        ("review", "journal", ["Review"]),
        ("article", "repository", []),
        ("book-chapter", "book series", ["BookSection"]),
        ("dissertation", "repository", []),
    ])
    def test_types_are_named_only_when_clear(
        self, work_type: str, source_type: str, expected: list[str]
    ) -> None:
        work = _work(4, type=work_type, primary_location={
            "source": {"display_name": "Some venue", "type": source_type}})
        paper = _openalex_paper(work, "q", 0)
        assert paper is not None and paper["publicationTypes"] == expected

    def test_markup_in_a_title_is_removed(self) -> None:
        paper = _openalex_paper(_work(5, title="Predicting <i>GED</i>\n passers"), "q", 0)
        assert paper is not None and paper["title"] == "Predicting GED passers"

    @pytest.mark.parametrize("work", [
        None, "W1", {"id": "", "display_name": "T"},
        {"id": "https://openalex.org/W1", "display_name": ""},
        {"id": "https://openalex.org/A123", "display_name": "An author, not a work"},
    ])
    def test_a_work_without_an_id_or_title_is_dropped(self, work: Any) -> None:
        assert _openalex_paper(work, "q", 0) is None

    def test_a_work_without_a_source_uses_the_raw_source_name(self) -> None:
        work = _work(6, primary_location={
            "source": None, "raw_source_name": "Proceedings of the 14th LAK Conference"})
        paper = _openalex_paper(work, "q", 0)
        assert paper is not None
        assert paper["venue"] == "Proceedings of the 14th LAK Conference"


# ---------------------------------------------------------------------------
# The request
# ---------------------------------------------------------------------------


class TestRequest:
    def test_address_words_and_headers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
        # A Crossref contact address is not passed on: OpenAlex retired
        # its mailto pool, and sending it would only share the address.
        monkeypatch.setenv("CROSSREF_MAILTO", "ops" + "@" + "example.invalid")
        agent = _agent(tmp_path)
        fake = FakeHTTP(openalex=[_page(_work(1))])
        with patch("requests.get", side_effect=fake), patch("time.sleep"):
            papers = agent._search_openalex(
                ["college enrollment prediction machine learning"], max_results_per_query=7)

        (call,) = fake.calls["openalex"]
        assert call["url"] == OPENALEX_URL
        params = call["params"]
        assert params["filter"] == (
            "title_and_abstract.search:college enrollment prediction machine learning")
        assert "search" not in params
        assert params["per_page"] == 7
        assert "abstract_inverted_index" in params["select"].split(",")
        assert not any("mailto" in str(k) or "mailto" in str(v) for k, v in params.items())
        headers = call["headers"]
        assert headers["User-Agent"].startswith("EDM-ARS (+https://")
        assert "mailto" not in headers["User-Agent"]
        assert headers["Accept"] == "application/json"
        assert "Authorization" not in headers
        assert call["timeout"] == 15
        assert [p["paperId"] for p in papers] == ["openalex_W4293192351"]
        assert agent._openalex_query_outcomes == ["ok"]

    def test_a_key_goes_in_a_header_never_in_the_url(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        key = "oa-" + "test-key-not-real"
        monkeypatch.setenv("OPENALEX_API_KEY", key)
        agent = _agent(tmp_path)
        fake = FakeHTTP(openalex=[requests.ConnectionError("no route")])
        with patch("requests.get", side_effect=fake), patch("time.sleep"):
            agent._search_openalex(["q"])
        (call,) = fake.calls["openalex"]
        assert call["headers"]["Authorization"] == f"Bearer {key}"
        assert key not in repr(call["params"]) and key not in call["url"]
        assert key not in (tmp_path / "pipeline.log").read_text(encoding="utf-8")
        assert key not in (tmp_path / "events.jsonl").read_text(encoding="utf-8")

    def test_filter_syntax_is_removed_from_the_words(self, tmp_path: Path) -> None:
        agent = _agent(tmp_path)
        params = agent._openalex_params('fairness, "subgroup": a|b (accuracy)', 10,
                                        "title_and_abstract")
        assert params["filter"] == "title_and_abstract.search:fairness subgroup a b accuracy"

    def test_search_in_all_uses_the_search_parameter(self, tmp_path: Path) -> None:
        agent = _agent(tmp_path)
        params = agent._openalex_params("college, enrollment", 10, "all")
        assert params["search"] == "college, enrollment"
        assert "filter" not in params

    def test_one_second_between_requests(self, tmp_path: Path) -> None:
        agent = _agent(tmp_path)
        fake = FakeHTTP(openalex=[_page(_work(1)), _page(_work(2)), _page(_work(3))])
        with patch("requests.get", side_effect=fake), patch("time.sleep") as sleep:
            papers = agent._search_openalex(QUERIES, delay_s=1.0)
        assert len(fake.calls["openalex"]) == 3
        assert [c.args[0] for c in sleep.call_args_list] == [1.0, 1.0]
        assert len(papers) == 3

    def test_a_work_found_twice_is_kept_once(self, tmp_path: Path) -> None:
        agent = _agent(tmp_path)
        fake = FakeHTTP(openalex=[_page(_work(1), _work(2)), _page(_work(2), _work(3))])
        with patch("requests.get", side_effect=fake), patch("time.sleep"):
            papers = agent._search_openalex(QUERIES[:2])
        assert [p["paperId"][-1] for p in papers] == ["1", "2", "3"]
        counts = [r["data"]["papers_found"] for r in _oa_progress(tmp_path)]
        assert counts == [2, 1]


class TestTurnedAway:
    def test_a_429_stops_the_search_and_is_recorded(self, tmp_path: Path) -> None:
        agent = _agent(tmp_path)
        fake = FakeHTTP(openalex=[429, _page(_work(1)), _page(_work(2))])
        with patch("requests.get", side_effect=fake), patch("time.sleep") as sleep:
            papers = agent._search_openalex(QUERIES)
        assert papers == []
        assert len(fake.calls["openalex"]) == 1
        sleep.assert_not_called()
        assert agent._openalex_query_outcomes == ["rate_limited", "skipped", "skipped"]
        assert agent._openalex_http_status == 429
        rows = _oa_progress(tmp_path)
        assert [r["data"]["status"] for r in rows] == ["rate_limited", "skipped", "skipped"]
        assert rows[0]["data"]["http_status"] == 429
        assert rows[0]["plain"] == "OpenAlex turned the request away: too many requests (HTTP 429)"
        assert rows[1]["plain"] == (
            "Skipped OpenAlex (2/3): OpenAlex turned away an earlier request as too many")
        log = (tmp_path / "pipeline.log").read_text(encoding="utf-8")
        assert ("OpenAlex turned away query 1/3 'q one' with HTTP 429; "
                "not sending the other 2 OpenAlex queries") in log

    @pytest.mark.parametrize("code", [401, 403, 406])
    def test_a_refusal_stops_the_search(self, tmp_path: Path, code: int) -> None:
        agent = _agent(tmp_path)
        fake = FakeHTTP(openalex=[code, _page(_work(1))])
        with patch("requests.get", side_effect=fake), patch("time.sleep"):
            agent._search_openalex(QUERIES[:2])
        assert len(fake.calls["openalex"]) == 1
        assert agent._openalex_query_outcomes == ["refused", "skipped"]
        assert _oa_progress(tmp_path)[1]["plain"] == (
            "Skipped OpenAlex (2/2): OpenAlex refused an earlier request")

    def test_a_server_error_or_a_broken_body_is_a_failure_and_the_search_goes_on(
        self, tmp_path: Path
    ) -> None:
        agent = _agent(tmp_path)
        broken = _resp(200)
        broken.json.side_effect = ValueError("not JSON")
        fake = FakeHTTP(openalex=[500, requests.Timeout("slow"), _page(_work(1))])
        with patch("requests.get", side_effect=fake), patch("time.sleep"):
            papers = agent._search_openalex(QUERIES)
        assert agent._openalex_query_outcomes == ["failed", "failed", "ok"]
        assert agent._openalex_http_status == 500
        assert len(papers) == 1
        rows = _oa_progress(tmp_path)
        assert rows[0]["data"]["http_status"] == 500
        assert "http_status" not in rows[1]["data"]

        (tmp_path / "b").mkdir()
        agent2 = _agent(tmp_path / "b")
        with patch("requests.get", return_value=broken), patch("time.sleep"):
            assert agent2._search_openalex(["q"]) == []
        assert agent2._openalex_query_outcomes == ["failed"]


# ---------------------------------------------------------------------------
# When OpenAlex is asked, and what the run records
# ---------------------------------------------------------------------------


class TestFallback:
    def test_a_refused_arxiv_is_replaced_by_openalex(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
        s2 = {"data": [{"paperId": "s2a", "title": "Predicting college enrollment, study 1",
                        "authors": [], "year": 2020, "externalIds": {}},
                       {"paperId": "s2b", "title": "A different title entirely",
                        "authors": [], "year": 2021,
                        "externalIds": {"DOI": "10.1000/SAME-DOI"}}]}
        fake = FakeHTTP(
            s2=s2,
            arxiv=[406],
            openalex=[
                # study 1 duplicates an S2 title; W…52 an S2 DOI (another case).
                _page(_work(1), _work(2, title="Something else", doi="10.1000/same-doi")),
                _page(_work(3), _work(4, title="Predicting college enrollment, study 3.")),
                _page(_work(5)),
            ],
        )
        agent, result = _search(tmp_path, fake)

        oa_calls = fake.calls["openalex"]
        assert [c["params"]["filter"].split(":", 1)[1] for c in oa_calls] == QUERIES
        ids = [p["paperId"] for p in result["papers"]]
        assert "openalex_W4293192353" in ids and "openalex_W4293192355" in ids
        assert "openalex_W4293192351" not in ids  # same title as an S2 record
        assert "openalex_W4293192352" not in ids  # same DOI as an S2 record
        assert "openalex_W4293192354" not in ids  # same title as study 3

        status = result["retrieval_status"]
        assert status["arxiv"] == "refused"
        assert status["arxiv_http_status"] == 406
        assert status["openalex"] == "ok"
        assert status["n_openalex"] == 2
        assert status["n_semantic_scholar"] == 2
        assert status["n_papers"] == 4
        assert status["degraded"] is False
        assert "openalex_http_status" not in status

        rows = _oa_progress(tmp_path)
        assert [r["data"]["query_index"] for r in rows] == [1, 2, 3]
        assert all(r["data"]["n_queries"] == 3 for r in rows)
        log = (tmp_path / "pipeline.log").read_text(encoding="utf-8")
        assert "arXiv refused; asking OpenAlex with the same 3 queries" in log
        merged = [e["message"] for e in agent.ctx.log if "Literature search merged" in e["message"]]
        assert merged == [
            "Literature search merged: 2 S2 + 0 arXiv (deduped) + 2 OpenAlex (deduped) "
            "= 4 total papers"]

    def test_an_arxiv_that_answered_is_not_replaced(self, tmp_path: Path) -> None:
        fake = FakeHTTP(arxiv=[200, 200, 200], openalex=[_page(_work(1))])
        _, result = _search(tmp_path, fake)
        assert fake.calls["openalex"] == []
        status = result["retrieval_status"]
        assert status["arxiv"] == "ok"
        assert status["openalex"] == "not_needed"
        assert status["n_openalex"] == 0
        assert _oa_progress(tmp_path) == []

    def test_a_failed_arxiv_is_replaced_too(self, tmp_path: Path) -> None:
        fake = FakeHTTP(arxiv=[502, 502, 502], openalex=[_page(_work(1))] * 3)
        _, result = _search(tmp_path, fake)
        assert len(fake.calls["openalex"]) == 3
        assert result["retrieval_status"]["openalex"] == "ok"
        log = (tmp_path / "pipeline.log").read_text(encoding="utf-8")
        assert "arXiv failed; asking OpenAlex" in log

    def test_turned_off_it_is_never_asked(self, tmp_path: Path) -> None:
        fake = FakeHTTP(arxiv=[406], openalex=[_page(_work(1))])
        _, result = _search(tmp_path, fake, enabled=False)
        assert fake.calls["openalex"] == []
        assert result["retrieval_status"]["openalex"] == "disabled"

    def test_arxiv_turned_off_is_not_arxiv_unavailable(self, tmp_path: Path) -> None:
        agent = _agent(tmp_path)
        agent.config["arxiv"] = {"enabled": False}
        agent._generate_search_queries = MagicMock(  # type: ignore[method-assign]
            return_value=list(QUERIES))
        fake = FakeHTTP(openalex=[_page(_work(1))])
        with patch("requests.get", side_effect=fake), patch("time.sleep"):
            result = agent._search_literature("x")
        assert fake.calls["arxiv"] == [] and fake.calls["openalex"] == []
        assert result["retrieval_status"]["openalex"] == "not_needed"

    def test_always_asks_it_as_a_third_source(self, tmp_path: Path) -> None:
        fake = FakeHTTP(arxiv=[200, 200, 200], openalex=[_page(_work(1))] * 3)
        agent, result = _search(tmp_path, fake, when="always")
        assert len(fake.calls["openalex"]) == 3
        status = result["retrieval_status"]
        assert status["arxiv"] == "ok" and status["openalex"] == "ok"
        assert status["n_openalex"] == 1
        # One set of search words for every source: no second model call.
        assert agent._generate_search_queries.call_count == 1
        log = (tmp_path / "pipeline.log").read_text(encoding="utf-8")
        assert "asking OpenAlex" not in log  # not standing in for anything

    def test_always_with_arxiv_turned_off(self, tmp_path: Path) -> None:
        agent = _agent(tmp_path)
        agent.config["arxiv"] = {"enabled": False}
        agent.config["openalex"] = {"enabled": True, "when": "always"}
        agent._generate_search_queries = MagicMock(  # type: ignore[method-assign]
            return_value=list(QUERIES))
        fake = FakeHTTP(openalex=[_page(_work(1))] * 3)
        with patch("requests.get", side_effect=fake), patch("time.sleep"):
            result = agent._search_literature("x")
        assert fake.calls["arxiv"] == [] and len(fake.calls["openalex"]) == 3
        assert result["retrieval_status"]["arxiv"] == "disabled"
        assert result["retrieval_status"]["openalex"] == "ok"


class TestWarning:
    def test_the_degraded_warning_names_openalex(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
        fake = FakeHTTP(arxiv=[406], openalex=[_page(_work(1), _work(2)), 429])
        _, result = _search(tmp_path, fake)
        status = result["retrieval_status"]
        # Semantic Scholar supplied nothing, so the pool stays degraded.
        assert status["degraded"] is True
        assert status["openalex"] == "ok" and status["n_openalex"] == 2
        (warning,) = _events(tmp_path, "warning")
        message = warning["data"]["message"]
        assert ("arXiv refused the request (HTTP 406, 0 papers), "
                "OpenAlex ok (2 papers), 2 papers in total.") in message

        ctx = PipelineContext(dataset_name="hsls09_public", raw_data_path="x",
                              output_dir=str(tmp_path))
        ctx.literature_context = {"papers": [], "retrieval_status": status}
        sentence = _literature_warning(ctx)
        assert sentence is not None and "openalex=ok" in sentence

    def test_an_openalex_limit_is_named_with_its_status(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
        fake = FakeHTTP(arxiv=[406], openalex=[429])
        _, result = _search(tmp_path, fake)
        status = result["retrieval_status"]
        assert status["openalex"] == "rate_limited"
        assert status["openalex_http_status"] == 429
        (warning,) = _events(tmp_path, "warning")
        assert "OpenAlex rate_limited (HTTP 429, 0 papers)" in warning["data"]["message"]

    def test_the_warning_says_nothing_of_an_openalex_not_asked(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
        fake = FakeHTTP(arxiv=[406], openalex=[_page(_work(1))])
        _search(tmp_path, fake, enabled=False)
        (warning,) = _events(tmp_path, "warning")
        assert "OpenAlex" not in warning["data"]["message"]


# ---------------------------------------------------------------------------
# Settings and documentation
# ---------------------------------------------------------------------------


class TestSettings:
    def test_defaults_without_a_block(self, tmp_path: Path) -> None:
        agent = _agent(tmp_path)
        agent.config.pop("openalex", None)
        assert agent._openalex_settings() == {
            "enabled": True, "when": "arxiv_unavailable", "max_results_per_query": 10,
            "request_delay_s": 1.0, "search_in": "title_and_abstract",
        }

    def test_odd_values_fall_back_rather_than_stop_the_step(self, tmp_path: Path) -> None:
        agent = _agent(tmp_path)
        agent.config["openalex"] = {
            "when": "sometimes", "max_results_per_query": "lots",
            "request_delay_s": "soon", "search_in": "everywhere",
        }
        settings = agent._openalex_settings()
        assert settings["when"] == "arxiv_unavailable"
        assert settings["max_results_per_query"] == 10
        assert settings["request_delay_s"] == 1.0
        assert settings["search_in"] == "title_and_abstract"
        agent.config["openalex"] = {"max_results_per_query": 500}
        assert agent._openalex_settings()["max_results_per_query"] == 100

    def test_config_yaml_ships_the_fallback(self) -> None:
        cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
        assert cfg["openalex"]["enabled"] is True
        assert cfg["openalex"]["when"] == "arxiv_unavailable"
        assert cfg["openalex"]["search_in"] == "title_and_abstract"

    def test_privacy_page_names_openalex_as_a_recipient(self) -> None:
        privacy = (ROOT / "PRIVACY.md").read_text(encoding="utf-8")
        rows = [line for line in privacy.splitlines()
                if line.startswith("| **") and "OpenAlex" in line]
        assert rows, "PRIVACY.md does not say that search words go to OpenAlex"
        assert "OPENALEX_API_KEY" in (ROOT / ".env.example").read_text(encoding="utf-8")
