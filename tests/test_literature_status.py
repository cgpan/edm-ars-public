"""Literature retrieval status and Crossref politeness (defect E9).

A run whose Semantic Scholar search failed used to finish COMPLETED with
the failure recorded only in checkpoint.json. These tests pin the
``retrieval_status`` block the ProblemFormulator now writes (CONTRACT
section 6), the pipeline.log line and warning event for a degraded
search, the ``lit.progress`` events, and the Crossref ``mailto``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.agents.problem_formulator import (
    ProblemFormulator,
    _crossref_mailto,
    _verify_paper_three_layers,
)
from src.config import load_config
from src.context import PipelineContext
from src.events import attach

ROOT = Path(__file__).resolve().parent.parent
# Built at runtime so no address-shaped literal sits in the source.
_MAILTO = "ops" + "@" + "example.invalid"


def _agent(tmp_path: Path, **s2_overrides: Any) -> ProblemFormulator:
    config = load_config(str(ROOT / "config.yaml"))
    config["semantic_scholar"].update({"max_retries": 0, **s2_overrides})
    ctx = PipelineContext(
        dataset_name="hsls09_public",
        raw_data_path=str(tmp_path / "raw.csv"),
        output_dir=str(tmp_path),
    )
    attach(ctx, str(tmp_path))
    return ProblemFormulator(ctx, "problem_formulator", config)


def _events(tmp_path: Path, etype: str) -> list[dict]:
    path = tmp_path / "events.jsonl"
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]
    return [r for r in rows if r["type"] == etype]


def _response(status: int, payload: dict | None = None, text: str = "") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = payload or {}
    resp.text = text
    return resp


# ---------------------------------------------------------------------------
# Crossref
# ---------------------------------------------------------------------------


class TestCrossrefPoliteness:
    def test_mailto_and_user_agent_are_sent(self) -> None:
        with patch("requests.get", return_value=_response(200, {"message": {"items": []}})) as get:
            _verify_paper_three_layers({"title": "A title"}, set(), [], crossref_mailto=_MAILTO)
        kwargs = get.call_args.kwargs
        assert kwargs["params"]["mailto"] == _MAILTO
        assert f"mailto:{_MAILTO}" in kwargs["headers"]["User-Agent"]
        assert kwargs["params"]["query.title"] == "A title"

    def test_without_an_address_nothing_is_invented(self) -> None:
        with patch("requests.get", return_value=_response(200, {"message": {"items": []}})) as get:
            _verify_paper_three_layers({"title": "A title"}, set(), [])
        kwargs = get.call_args.kwargs
        assert "mailto" not in kwargs["params"]
        assert "mailto" not in kwargs["headers"]["User-Agent"]
        assert kwargs["headers"]["User-Agent"].startswith("EDM-ARS")

    def test_env_overrides_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = {"semantic_scholar": {"crossref_mailto": "config" + "@" + "example.invalid"}}
        monkeypatch.delenv("CROSSREF_MAILTO", raising=False)
        assert _crossref_mailto(cfg) == cfg["semantic_scholar"]["crossref_mailto"]
        monkeypatch.setenv("CROSSREF_MAILTO", _MAILTO)
        assert _crossref_mailto(cfg) == _MAILTO
        monkeypatch.delenv("CROSSREF_MAILTO")
        assert _crossref_mailto({"semantic_scholar": {"crossref_mailto": None}}) is None

    def test_filter_passes_the_configured_address(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CROSSREF_MAILTO", _MAILTO)
        agent = _agent(tmp_path)
        s2 = {"papers": [{"paperId": "p1", "title": "Known paper"}]}
        lit = {"papers": [{"paperId": "zz", "title": "Unknown paper title here"}]}
        with patch("requests.get", return_value=_response(200, {"message": {"items": []}})) as get:
            agent._filter_hallucinated_papers(lit, s2)
        assert get.call_args.kwargs["params"]["mailto"] == _MAILTO


# ---------------------------------------------------------------------------
# retrieval_status
# ---------------------------------------------------------------------------


class TestRetrievalStatus:
    def test_rate_limited_search_is_degraded_and_says_so(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
        agent = _agent(tmp_path)

        def _get(url: str, **_kw: Any) -> MagicMock:
            if "semanticscholar" in url:
                return _response(429)
            return _response(503)  # arXiv down too

        with patch("requests.get", side_effect=_get), patch("time.sleep"):
            result = agent._search_literature(None)

        status = result["retrieval_status"]
        assert status["semantic_scholar"] == "rate_limited"
        assert status["arxiv"] == "failed"
        assert status["n_papers"] == 0
        assert status["degraded"] is True

        log = (tmp_path / "pipeline.log").read_text(encoding="utf-8")
        assert "[problem_formulator] Literature retrieval degraded" in log
        assert "SEMANTIC_SCHOLAR_API_KEY" in log
        (warning,) = _events(tmp_path, "warning")
        assert warning["data"]["code"] == "LITERATURE_DEGRADED"
        progress = _events(tmp_path, "lit.progress")
        s2_rows = [e["data"] for e in progress if e["data"]["source"] == "semantic_scholar"]
        assert s2_rows and all(r["status"] == "rate_limited" for r in s2_rows)
        assert [r["query_index"] for r in s2_rows] == list(range(1, len(s2_rows) + 1))
        assert all(r["n_queries"] == len(s2_rows) for r in s2_rows)
        assert any(e["data"]["source"] == "arxiv" and e["data"]["status"] == "failed"
                   for e in progress)

    def test_arxiv_only_pool_is_still_degraded(self, tmp_path: Path) -> None:
        agent = _agent(tmp_path)
        agent._search_semantic_scholar = MagicMock(  # type: ignore[method-assign]
            return_value={"search_query": "q", "papers": [], "novelty_evidence": ""})
        agent._search_arxiv = MagicMock(return_value=[  # type: ignore[method-assign]
            {"paperId": "arxiv:1", "title": "Preprint", "year": 2025, "source": "arxiv"}])
        result = agent._search_literature(None)
        status = result["retrieval_status"]
        assert status["n_papers"] == 1
        assert status["n_arxiv"] == 1
        assert status["semantic_scholar"] == "failed"
        assert status["degraded"] is True

    def test_healthy_search_is_not_degraded(self, tmp_path: Path) -> None:
        agent = _agent(tmp_path)
        agent.config["arxiv"] = {"enabled": False}
        paper = {"paperId": "p1", "title": "T", "authors": [], "year": 2022, "abstract": ""}
        with patch("requests.get", return_value=_response(200, {"data": [paper]})), \
             patch("time.sleep"):
            result = agent._search_literature(None)
        status = result["retrieval_status"]
        assert status == {
            "semantic_scholar": "ok", "arxiv": "disabled", "n_papers": 1,
            "degraded": False, "n_semantic_scholar": 1, "n_arxiv": 0,
        }
        assert not (tmp_path / "events.jsonl").exists() or not _events(tmp_path, "warning")

    def test_run_carries_the_status_onto_the_stored_context(self, tmp_path: Path) -> None:
        agent = _agent(tmp_path)
        status = {"semantic_scholar": "ok", "arxiv": "ok", "n_papers": 1,
                  "degraded": False, "n_semantic_scholar": 1, "n_arxiv": 0}
        s2_context = {
            "search_query": "q",
            "papers": [{"paperId": "s2_001", "title": "Real paper", "year": 2024}],
            "novelty_evidence": "",
            "retrieval_status": status,
        }
        agent._search_literature = MagicMock(return_value=s2_context)  # type: ignore[method-assign]
        agent.call_llm = MagicMock(return_value=json.dumps({  # type: ignore[method-assign]
            "research_spec": {"research_question": "q", "outcome_variable": "X3TGPAMAT",
                              "predictor_set": []},
            "literature_context": {"search_query": "q", "novelty_evidence": "n",
                                   "papers": [{"paperId": "s2_001", "title": "Real paper"}]},
        }))
        with patch("requests.get", side_effect=Exception("offline")):
            out = agent.run()
        assert out["literature_context"]["retrieval_status"] == status
        assert out["retrieved_literature"]["retrieval_status"] == status
        # Bookkeeping, not literature: the model's prompt does not carry it.
        prompt = agent.call_llm.call_args.args[0]
        assert "retrieval_status" not in prompt
        assert "s2_001" in prompt


class TestProviderFailureIsNotSwallowed:
    def test_query_generation_reraises_a_provider_error(self, tmp_path: Path) -> None:
        from src.errors import ProviderError

        agent = _agent(tmp_path)
        agent.call_llm = MagicMock(  # type: ignore[method-assign]
            side_effect=ProviderError("NO_CREDIT", "top up", provider="deepseek"))
        with pytest.raises(ProviderError):
            agent._generate_search_queries("predict math GPA")

    def test_other_query_generation_failures_still_fall_back(self, tmp_path: Path) -> None:
        agent = _agent(tmp_path)
        agent.call_llm = MagicMock(return_value="not json")  # type: ignore[method-assign]
        assert agent._generate_search_queries("x") == agent._DEFAULT_S2_QUERIES
