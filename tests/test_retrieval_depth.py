"""Retrieval-depth tests for F-P5-DEPTH-RECENCY-SKEW (G1, retrieval half).

The defect: the Semantic Scholar request carried ``year=<current-10>-<current>``
derived from ``semantic_scholar.year_filter: 10``, so every record older than
ten years was excluded *at the HTTP layer*. A real run retrieved 2024x4 /
2025x79 / 2026x17 — oldest paper 2024 — while every anchor venue profile needs
~27% of its references older than ten years and 94% of published anchors cite
something older than fifteen. No client-side ranking can recover a record the
request never asked for.

These tests pin the three retrieval-side fixes:
  1. the ranking/metadata fields are REQUESTED *and* survive the hand-written
     record mapping (requesting without mapping is a silent no-op),
  2. the topical queries no longer carry a rolling recency floor, and one
     extra unwindowed "seminal work" request is issued,
  3. the retrieved old records are not immediately thrown away again by the
     year-descending trims.

Everything is offline: ``requests.get`` is patched, no LLM is called.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.problem_formulator import ProblemFormulator
from src.config import load_config
from src.context import PipelineContext

CONFIG_PATH = str(Path(__file__).parent.parent / "config.yaml")

# getattr with a literal default so a regression that deletes the tag fails
# these tests one by one instead of blowing up collection for the whole file.
_SEMINAL = getattr(ProblemFormulator, "_SEMINAL_QUERY_TAG", "__seminal__")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(tmp_path: Path, queries: list[str] | None = None) -> ProblemFormulator:
    config = load_config(CONFIG_PATH)
    ctx = PipelineContext(
        dataset_name="hsls09_public",
        raw_data_path="data/raw/nonexistent.csv",
        output_dir=str(tmp_path),
        max_revision_cycles=2,
    )
    with patch("anthropic.Anthropic"):
        agent = ProblemFormulator(ctx, "problem_formulator", config)
    # Never let a test reach the query-generation LLM call.
    agent._generate_search_queries = MagicMock(
        return_value=list(queries) if queries else ["edm prediction test query"]
    )
    return agent


def _s2_item(paper_id: str, year: int, **extra: Any) -> dict:
    """One raw S2 API response row (as S2 shapes it, not as we store it)."""
    item = {
        "paperId": paper_id,
        "title": f"Paper {paper_id}",
        "authors": [{"name": "Author A"}],
        "year": year,
        "abstract": "Abstract.",
    }
    item.update(extra)
    return item


def _resp(items: list[dict]) -> MagicMock:
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"data": items}
    return resp


class _Router:
    """side_effect for ``requests.get`` that answers topical vs seminal calls.

    A request carrying ``minCitationCount`` is the seminal-work query; every
    other request is a topical query. Records every ``params`` dict sent.
    """

    def __init__(
        self,
        topical: list[dict],
        seminal: list[dict] | None = None,
        seminal_status: int | None = None,
    ) -> None:
        self.topical = topical
        self.seminal = seminal or []
        self.seminal_status = seminal_status
        self.params_sent: list[dict] = []

    def __call__(self, url: str, **kwargs: Any) -> MagicMock:
        params = kwargs.get("params") or {}
        self.params_sent.append(params)
        if "minCitationCount" in params:
            if self.seminal_status is not None:
                return MagicMock(status_code=self.seminal_status)
            return _resp(self.seminal)
        return _resp(self.topical)

    @property
    def topical_params(self) -> list[dict]:
        return [p for p in self.params_sent if "minCitationCount" not in p]

    @property
    def seminal_params(self) -> list[dict]:
        return [p for p in self.params_sent if "minCitationCount" in p]


def _year_lower_bound(year_param: str) -> int:
    return int(str(year_param).split("-")[0])


# ---------------------------------------------------------------------------
# A. Request fields + the load-bearing record mapping
# ---------------------------------------------------------------------------


class TestRequestFieldsAndMapping:
    def test_s2_request_asks_for_ranking_fields(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        router = _Router(topical=[])
        with patch("requests.get", side_effect=router), patch("time.sleep"):
            agent._search_semantic_scholar("test query")

        fields = router.params_sent[0]["fields"].split(",")
        for name in (
            "citationCount",
            "influentialCitationCount",
            "publicationDate",
            "fieldsOfStudy",
            "publicationTypes",
        ):
            assert name in fields, f"{name} not requested; fields={fields}"
        # Arc P3 contract must survive.
        for name in ("paperId", "title", "authors", "year", "abstract", "venue",
                     "externalIds"):
            assert name in fields

    def test_new_fields_survive_the_record_mapping(self, tmp_path: Path) -> None:
        """LOAD-BEARING: the response is hand-mapped into a small dict.

        Adding names to the request without extending that mapping changes
        nothing at all — the fields are fetched and then dropped on the floor.
        """
        agent = _make_agent(tmp_path)
        item = _s2_item(
            "p1",
            2011,
            citationCount=1234,
            influentialCitationCount=99,
            referenceCount=41,
            publicationDate="2011-03-15",
            fieldsOfStudy=["Education", "Psychology"],
            publicationTypes=["JournalArticle"],
            isOpenAccess=True,
            externalIds={"DOI": "10.1007/s11336-011-9207-7"},
            venue="Psychometrika",
        )
        router = _Router(topical=[item])
        with patch("requests.get", side_effect=router), patch("time.sleep"):
            result = agent._search_semantic_scholar("test query")

        rec = result["papers"][0]
        assert rec["citationCount"] == 1234
        assert rec["influentialCitationCount"] == 99
        assert rec["referenceCount"] == 41
        assert rec["publicationDate"] == "2011-03-15"
        assert rec["fieldsOfStudy"] == ["Education", "Psychology"]
        assert rec["publicationTypes"] == ["JournalArticle"]
        assert rec["isOpenAccess"] is True
        # Pre-existing contract untouched.
        assert rec["doi"] == "10.1007/s11336-011-9207-7"
        assert rec["venue"] == "Psychometrika"
        assert rec["authors"] == ["Author A"]

    def test_missing_counts_map_to_none_not_zero(self, tmp_path: Path) -> None:
        """Degradation: S2 omitting a count must not read as "uncited".

        A 0 default would rank every record S2 has no data for dead last,
        which is a different bias, not an absence of one.
        """
        agent = _make_agent(tmp_path)
        router = _Router(topical=[_s2_item("p1", 2020)])
        with patch("requests.get", side_effect=router), patch("time.sleep"):
            result = agent._search_semantic_scholar("test query")

        rec = result["papers"][0]
        assert rec["citationCount"] is None
        assert rec["influentialCitationCount"] is None
        assert rec["referenceCount"] is None
        # Collection-valued fields degrade to empty, not None.
        assert rec["fieldsOfStudy"] == []
        assert rec["publicationTypes"] == []
        assert rec["publicationDate"] == ""

    def test_retrieval_rank_and_matched_query_are_stamped(self, tmp_path: Path) -> None:
        """S2 returns RELEVANCE order; the provenance must not be lost."""
        agent = _make_agent(tmp_path, queries=["query alpha"])
        router = _Router(
            topical=[_s2_item("p1", 2020), _s2_item("p2", 2019), _s2_item("p3", 2018)]
        )
        with patch("requests.get", side_effect=router), patch("time.sleep"):
            result = agent._search_semantic_scholar("test query")

        by_id = {p["paperId"]: p for p in result["papers"]}
        assert by_id["p1"]["retrieval_rank"] == 0
        assert by_id["p2"]["retrieval_rank"] == 1
        assert by_id["p3"]["retrieval_rank"] == 2
        assert by_id["p1"]["matched_query"] == "query alpha"
        assert by_id["p1"]["source"] == "s2"


# ---------------------------------------------------------------------------
# B. The year window
# ---------------------------------------------------------------------------


class TestYearWindow:
    def test_rolling_window_still_supported(self) -> None:
        assert ProblemFormulator._build_year_range(10, 1900, 2026) == "2016-2026"

    def test_null_year_filter_uses_absolute_floor(self) -> None:
        assert ProblemFormulator._build_year_range(None, 1900, 2026) == "1900-2026"

    def test_null_filter_and_null_floor_omit_the_year_param(self) -> None:
        assert ProblemFormulator._build_year_range(None, None, 2026) is None

    @pytest.mark.parametrize("bad", ["", "abc", [], {}])
    def test_garbage_year_filter_degrades_to_the_floor(self, bad: Any) -> None:
        assert ProblemFormulator._build_year_range(bad, 1900, 2026) == "1900-2026"

    def test_garbage_year_floor_omits_the_year_param(self) -> None:
        assert ProblemFormulator._build_year_range(None, "not-a-year", 2026) is None

    def test_zero_year_filter_is_not_a_zero_width_window(self) -> None:
        """``year_filter: 0`` must not send ``year=2026-2026``."""
        assert ProblemFormulator._build_year_range(0, 1900, 2026) == "1900-2026"

    def test_shipped_config_has_no_recency_floor(self, tmp_path: Path) -> None:
        """Pins the config default: pre-2016 work must be REACHABLE.

        With ``year_filter: 10`` the request was ``year=2016-2026`` and
        Tatsuoka (1983), Junker & Sijtsma (2001) and de la Torre (2011) were
        unretrievable no matter what any ranker did downstream.
        """
        agent = _make_agent(tmp_path)
        router = _Router(topical=[])
        with patch("requests.get", side_effect=router), patch("time.sleep"):
            agent._search_semantic_scholar("test query")

        year_param = router.topical_params[0].get("year")
        assert year_param is not None, "topical query lost its year param entirely"
        lower = _year_lower_bound(year_param)
        assert lower <= 1983, (
            f"topical query still floors retrieval at {lower}; "
            "foundational work remains unreachable"
        )

    def test_seminal_query_sends_no_year_param_at_all(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path, queries=["q1"])
        router = _Router(
            topical=[_s2_item(f"p{i}", 2025) for i in range(3)],
            seminal=[_s2_item("old1", 1983)],
        )
        with patch("requests.get", side_effect=router), patch("time.sleep"):
            agent._search_semantic_scholar("test query")

        assert len(router.seminal_params) == 1
        assert "year" not in router.seminal_params[0]

    def test_rolling_window_config_still_reaches_the_request(
        self, tmp_path: Path
    ) -> None:
        """A legacy snapshot pinning ``year_filter: 10`` keeps its behaviour."""
        agent = _make_agent(tmp_path)
        agent.config["semantic_scholar"]["year_filter"] = 10
        router = _Router(topical=[])
        with patch("requests.get", side_effect=router), patch("time.sleep"):
            agent._search_semantic_scholar("test query")

        current_year = datetime.utcnow().year
        assert router.topical_params[0]["year"] == f"{current_year - 10}-{current_year}"


# ---------------------------------------------------------------------------
# C. The seminal-work query
# ---------------------------------------------------------------------------


class TestSeminalQuery:
    def _three_query_agent(self, tmp_path: Path) -> ProblemFormulator:
        return _make_agent(tmp_path, queries=["q1", "q2", "q3"])

    def test_seminal_query_is_exactly_one_extra_request(self, tmp_path: Path) -> None:
        agent = self._three_query_agent(tmp_path)
        router = _Router(
            topical=[_s2_item(f"p{i}", 2025) for i in range(3)],
            seminal=[_s2_item("old1", 1983)],
        )
        with patch("requests.get", side_effect=router) as mock_get, patch("time.sleep"):
            agent._search_semantic_scholar("test query")

        assert mock_get.call_count == 4, "3 topical + 1 seminal"
        assert len(router.seminal_params) == 1
        sem_cfg = agent.config["semantic_scholar"]["seminal_query"]
        # min_citations is deliberately NOT pinned to a literal — it is an
        # uncalibrated judgment call (spec §7.5) and is expected to move.
        assert router.seminal_params[0]["minCitationCount"] == sem_cfg["min_citations"]
        assert router.seminal_params[0]["minCitationCount"] > 0
        assert router.seminal_params[0]["limit"] == sem_cfg["limit"]
        assert router.seminal_params[0]["query"] == "q1"

    def test_foundational_work_reaches_the_pool(self, tmp_path: Path) -> None:
        """The headline regression for F-P5-DEPTH-RECENCY-SKEW.

        Before the fix the retrieved pool's oldest record was 2024 and a
        perfect re-rank of it yielded ZERO references older than five years.
        """
        agent = self._three_query_agent(tmp_path)
        router = _Router(
            topical=[_s2_item(f"p{i}", 2025) for i in range(5)],
            seminal=[
                _s2_item("tatsuoka1983", 1983, citationCount=2100),
                _s2_item("junker2001", 2001, citationCount=1500),
                _s2_item("delatorre2011", 2011, citationCount=1800),
            ],
        )
        with patch("requests.get", side_effect=router), patch("time.sleep"):
            result = agent._search_semantic_scholar("cognitive diagnosis models")

        ids = {p["paperId"] for p in result["papers"]}
        assert {"tatsuoka1983", "junker2001", "delatorre2011"} <= ids
        current_year = datetime.utcnow().year
        oldest = min(p["year"] for p in result["papers"])
        assert current_year - oldest > 15

        seminal_records = [
            p for p in result["papers"] if p.get("matched_query") == _SEMINAL
        ]
        assert len(seminal_records) == 3, "seminal provenance must be stamped"

    def test_seminal_failure_is_non_fatal(self, tmp_path: Path) -> None:
        """A 429 storm on the extra request must not lose the topical pool."""
        agent = self._three_query_agent(tmp_path)
        router = _Router(
            topical=[_s2_item(f"p{i}", 2025) for i in range(3)],
            seminal_status=429,
        )
        with patch("requests.get", side_effect=router), patch("time.sleep"):
            result = agent._search_semantic_scholar("test query")

        assert len(result["papers"]) == 3
        assert {p["paperId"] for p in result["papers"]} == {"p0", "p1", "p2"}

    def test_seminal_config_error_is_non_fatal(self, tmp_path: Path) -> None:
        agent = self._three_query_agent(tmp_path)
        agent.config["semantic_scholar"]["seminal_query"]["min_citations"] = "fifty"
        router = _Router(topical=[_s2_item(f"p{i}", 2025) for i in range(3)])
        with patch("requests.get", side_effect=router), patch("time.sleep"):
            result = agent._search_semantic_scholar("test query")

        assert len(result["papers"]) == 3
        assert any(
            "seminal query failed" in e.get("message", "").lower()
            for e in agent.ctx.log
        )

    def test_garbage_numeric_config_never_aborts_retrieval(
        self, tmp_path: Path
    ) -> None:
        """A typo in config.yaml must not take down the FORMULATING stage."""
        agent = self._three_query_agent(tmp_path)
        agent.config["semantic_scholar"]["seminal_query"].update(
            {"min_primary_pool": "three", "reserved_pool_slots": None, "limit": "many"}
        )
        router = _Router(
            topical=[_s2_item(f"p{i}", 2025) for i in range(3)],
            seminal=[_s2_item("old1", 1983)],
        )
        with patch("requests.get", side_effect=router), patch("time.sleep"):
            result = agent._search_semantic_scholar("test query")

        assert len(result["papers"]) >= 3

    def test_seminal_query_can_be_disabled(self, tmp_path: Path) -> None:
        agent = self._three_query_agent(tmp_path)
        agent.config["semantic_scholar"]["seminal_query"]["enabled"] = False
        router = _Router(topical=[_s2_item(f"p{i}", 2025) for i in range(3)])
        with patch("requests.get", side_effect=router) as mock_get, patch("time.sleep"):
            agent._search_semantic_scholar("test query")

        assert mock_get.call_count == 3
        assert router.seminal_params == []

    def test_seminal_query_defaults_on_for_legacy_config_snapshots(
        self, tmp_path: Path
    ) -> None:
        agent = self._three_query_agent(tmp_path)
        agent.config["semantic_scholar"].pop("seminal_query", None)
        router = _Router(
            topical=[_s2_item(f"p{i}", 2025) for i in range(3)],
            seminal=[_s2_item("old1", 1983)],
        )
        with patch("requests.get", side_effect=router) as mock_get, patch("time.sleep"):
            result = agent._search_semantic_scholar("test query")

        assert mock_get.call_count == 4
        assert "old1" in {p["paperId"] for p in result["papers"]}

    def test_seminal_query_skipped_when_topical_pool_is_degenerate(
        self, tmp_path: Path
    ) -> None:
        """Don't compound 429 exposure for a pool that is unusable anyway."""
        agent = self._three_query_agent(tmp_path)
        router = _Router(topical=[_s2_item("only", 2025)])
        with patch("requests.get", side_effect=router) as mock_get, patch("time.sleep"):
            agent._search_semantic_scholar("test query")

        assert mock_get.call_count == 3
        assert router.seminal_params == []

    def test_seminal_query_skipped_when_s2_is_down(self, tmp_path: Path) -> None:
        agent = self._three_query_agent(tmp_path)
        mock_500 = MagicMock(status_code=500)
        with patch("requests.get", return_value=mock_500) as mock_get, patch("time.sleep"):
            result = agent._search_semantic_scholar("test query")

        assert result["papers"] == []
        # 3 queries x (1 + max_retries) attempts, and no seminal request.
        max_retries = agent.config["semantic_scholar"]["max_retries"]
        assert mock_get.call_count == 3 * (max_retries + 1)


# ---------------------------------------------------------------------------
# D. The retrieved old records must survive the year-descending trims
# ---------------------------------------------------------------------------


class TestSeminalRecordsSurviveTrimming:
    def test_seminal_records_survive_the_s2_pool_trim(self, tmp_path: Path) -> None:
        """Retrieving old work and then trimming it away is a no-op fix.

        The pool trim runs off a year-descending sort, so the seminal records
        are by construction the first casualties.
        """
        agent = _make_agent(tmp_path, queries=["q1", "q2", "q3"])
        agent.config["semantic_scholar"]["max_results"] = 4
        router = _Router(
            topical=[_s2_item(f"p{i}", 2025) for i in range(6)],
            seminal=[_s2_item("old1", 1983), _s2_item("old2", 2001)],
        )
        with patch("requests.get", side_effect=router), patch("time.sleep"):
            result = agent._search_semantic_scholar("test query")

        ids = {p["paperId"] for p in result["papers"]}
        assert len(result["papers"]) == 4
        assert {"old1", "old2"} <= ids

    def test_trim_evicts_the_least_relevant_records_not_the_newest(self) -> None:
        pool = [
            {"paperId": "new_top", "year": 2026, "retrieval_rank": 0,
             "matched_query": "q1"},
            {"paperId": "new_tail", "year": 2025, "retrieval_rank": 90,
             "matched_query": "q1"},
            {"paperId": "old1", "year": 1983, "retrieval_rank": 0,
             "matched_query": _SEMINAL},
        ]
        kept = ProblemFormulator._trim_pool_preserving_seminal(pool, 2, reserve=20)
        ids = [p["paperId"] for p in kept]
        assert len(kept) == 2
        assert "old1" in ids
        assert "new_top" in ids, "the most relevant recent record was evicted"

    def test_trim_tolerates_unusable_retrieval_ranks(self) -> None:
        """Legacy / JSON-round-tripped pools carry no rank, or a string one."""
        pool = [
            {"paperId": "no_rank", "year": 2026, "matched_query": "q1"},
            {"paperId": "str_rank", "year": 2025, "retrieval_rank": "3",
             "matched_query": "q1"},
            {"paperId": "int_rank", "year": 2024, "retrieval_rank": 0,
             "matched_query": "q1"},
            {"paperId": "old1", "year": 1983, "retrieval_rank": 0,
             "matched_query": _SEMINAL},
        ]
        kept = ProblemFormulator._trim_pool_preserving_seminal(pool, 3, reserve=20)
        ids = [p["paperId"] for p in kept]
        assert len(kept) == 3
        assert "old1" in ids
        # The record with no usable rank is the one evicted.
        assert "no_rank" not in ids
        assert {"str_rank", "int_rank"} <= set(ids)

    def test_trim_is_a_no_op_when_reserve_is_zero(self) -> None:
        pool = [
            {"paperId": "n1", "year": 2026, "matched_query": "q1"},
            {"paperId": "old1", "year": 1983, "matched_query": _SEMINAL},
        ]
        kept = ProblemFormulator._trim_pool_preserving_seminal(pool, 1, reserve=0)
        assert [p["paperId"] for p in kept] == ["n1"]

    def test_trim_never_grows_or_duplicates_the_pool(self) -> None:
        pool = [
            {"paperId": f"n{i}", "year": 2026, "retrieval_rank": i,
             "matched_query": "q1"}
            for i in range(10)
        ] + [
            {"paperId": f"o{i}", "year": 1990 + i, "retrieval_rank": i,
             "matched_query": _SEMINAL}
            for i in range(5)
        ]
        kept = ProblemFormulator._trim_pool_preserving_seminal(pool, 6, reserve=20)
        ids = [p["paperId"] for p in kept]
        assert len(ids) == 6
        assert len(set(ids)) == 6

    def test_trim_handles_more_seminal_records_than_slots(self) -> None:
        pool = [{"paperId": f"o{i}", "year": 1990, "matched_query": _SEMINAL}
                for i in range(5)]
        kept = ProblemFormulator._trim_pool_preserving_seminal(pool, 2, reserve=20)
        assert len(kept) == 2

    def test_seminal_records_survive_the_arxiv_merge_trim(self, tmp_path: Path) -> None:
        """Second trim site: arXiv preprints are all recent and would win."""
        agent = _make_agent(tmp_path, queries=["q1"])
        agent.config["semantic_scholar"]["max_results"] = 5
        s2_papers = [
            {"paperId": f"s{i}", "title": f"S2 Paper {i}", "authors": ["A"],
             "year": 2025, "abstract": "", "retrieval_rank": i,
             "matched_query": "q1", "source": "s2"}
            for i in range(5)
        ] + [
            {"paperId": "old1", "title": "Rule Space Model", "authors": ["Tatsuoka"],
             "year": 1983, "abstract": "", "retrieval_rank": 0,
             "matched_query": _SEMINAL, "source": "s2"}
        ]
        agent._search_semantic_scholar = MagicMock(return_value={
            "search_query": "q1", "papers": s2_papers, "novelty_evidence": "",
        })
        agent._search_arxiv = MagicMock(return_value=[
            {"paperId": "arxiv:2601.1", "title": "A Recent Preprint",
             "authors": ["X"], "year": 2026, "abstract": "", "source": "arxiv"},
            {"paperId": "arxiv:2601.2", "title": "Another Recent Preprint",
             "authors": ["Y"], "year": 2026, "abstract": "", "source": "arxiv"},
        ])

        result = agent._search_literature("test prompt")

        ids = {p["paperId"] for p in result["papers"]}
        assert len(result["papers"]) == 5
        assert "old1" in ids, "the seminal record was trimmed away by the arXiv merge"
