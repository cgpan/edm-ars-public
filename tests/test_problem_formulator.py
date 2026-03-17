"""Tests for the ProblemFormulator agent.

Unit tests run without any API calls.
Integration tests (marked @pytest.mark.integration) require ANTHROPIC_API_KEY.

Run unit tests:
    pytest tests/test_problem_formulator.py -v -k "not integration"

Run integration tests:
    pytest tests/test_problem_formulator.py -v --run-integration
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.problem_formulator import (
    TEMPORAL_ORDER,
    ProblemFormulator,
    _build_registry_var_map,
    _get_tier3_exact_matches,
    _jaccard_similarity,
    _tokenize_title,
    _verify_paper_three_layers,
)
from src.config import load_config
from src.context import PipelineContext

CONFIG_PATH = str(Path(__file__).parent.parent / "config.yaml")

# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------

_REGISTRY_SNIPPET: dict = {
    "name": "hsls09_public",
    "levels": {"student": 23503, "school": 944},
    "temporal_order": [
        "base_year",
        "first_follow_up",
        "second_follow_up",
        "update_panel",
    ],
    "tier3_exclusion_rules": {
        "exact_matches": ["STU_ID", "SCH_ID", "W1STRATUM"],
        "prefix_patterns": ["W[0-9]"],
        "suffix_patterns": ["_IM$"],
    },
    "variables": {
        "outcomes": [
            {
                "name": "X3TGPAMAT",
                "label": "GPA in math courses",
                "type": "continuous",
                "wave": "second_follow_up",
                "pct_missing": 17.5,
            },
            {
                "name": "X4EVRATNDCLG",
                "label": "Ever attended college",
                "type": "binary",
                "wave": "update_panel",
                "pct_missing": 22.8,
            },
            {
                "name": "dropout_derived",
                "label": "HS dropout indicator",
                "type": "binary",
                "wave": "second_follow_up",
                "pct_missing": 14.1,
            },
        ],
        "predictors": {
            "academic": [
                {
                    "name": "X1TXMTSC",
                    "label": "Math theta score (9th grade)",
                    "type": "continuous",
                    "wave": "base_year",
                    "pct_missing": 5.4,
                },
                {
                    "name": "X1MTHEFF",
                    "label": "Math self-efficacy",
                    "type": "continuous",
                    "wave": "base_year",
                    "pct_missing": 0.0,
                },
            ],
            "demographic": [
                {
                    "name": "X1SEX",
                    "label": "Student sex",
                    "type": "categorical",
                    "wave": "base_year",
                    "pct_missing": 0.0,
                    "protected_attribute": True,
                },
                {
                    "name": "X1SES",
                    "label": "SES composite",
                    "type": "continuous",
                    "wave": "base_year",
                    "pct_missing": 0.0,
                },
            ],
            "academic_followup": [
                {
                    "name": "X2TXMTSC",
                    "label": "Math theta score (11th grade)",
                    "type": "continuous",
                    "wave": "first_follow_up",
                    "pct_missing": 24.3,
                },
            ],
        },
    },
}

_VALID_RESEARCH_SPEC: dict = {
    "research_question": (
        "Can we predict students' 12th-grade math GPA using "
        "9th-grade academic and motivational factors?"
    ),
    "outcome_variable": "X3TGPAMAT",
    "outcome_type": "continuous",
    "predictor_set": [
        {"variable": "X1TXMTSC", "rationale": "Prior math achievement.", "wave": "base_year"},
        {"variable": "X1MTHEFF", "rationale": "Math self-efficacy.", "wave": "base_year"},
        {"variable": "X1SES", "rationale": "Socioeconomic status.", "wave": "base_year"},
        {"variable": "X1SEX", "rationale": "Sex differences in GPA.", "wave": "base_year"},
    ],
    "target_population": "Full HSLS:09 public-use sample",
    "subgroup_analyses": ["X1SEX"],
    "expected_contribution": "Novel motivational predictor combination.",
    "potential_limitations": ["Multilevel structure not modeled"],
    "novelty_score_self_assessment": 4,
}

_VALID_LIT_CONTEXT: dict = {
    "search_query": "math GPA prediction educational data mining",
    "papers": [
        {
            "paperId": "abc123",
            "title": "Predicting Math GPA",
            "authors": ["Smith, J."],
            "year": 2022,
            "abstract": "We predict math GPA.",
        }
    ],
    "novelty_evidence": "Prior work uses test scores; we use GPA.",
}


def _make_ctx(tmp_path: Path) -> PipelineContext:
    return PipelineContext(
        dataset_name="hsls09_public",
        raw_data_path="data/raw/nonexistent.csv",
        output_dir=str(tmp_path),
        max_revision_cycles=2,
    )


def _make_agent(tmp_path: Path) -> ProblemFormulator:
    config = load_config(CONFIG_PATH)
    ctx = _make_ctx(tmp_path)
    with patch("anthropic.Anthropic"):
        return ProblemFormulator(ctx, "problem_formulator", config)


# ---------------------------------------------------------------------------
# Unit tests — registry helpers
# ---------------------------------------------------------------------------


class TestRegistryHelpers:
    def test_build_var_map_includes_outcomes(self) -> None:
        var_map = _build_registry_var_map(_REGISTRY_SNIPPET)
        assert "X3TGPAMAT" in var_map
        assert var_map["X3TGPAMAT"]["wave"] == "second_follow_up"

    def test_build_var_map_includes_predictors(self) -> None:
        var_map = _build_registry_var_map(_REGISTRY_SNIPPET)
        assert "X1TXMTSC" in var_map
        assert var_map["X1TXMTSC"]["wave"] == "base_year"
        assert "X2TXMTSC" in var_map
        assert var_map["X2TXMTSC"]["wave"] == "first_follow_up"

    def test_build_var_map_empty_registry(self) -> None:
        var_map = _build_registry_var_map({})
        assert var_map == {}

    def test_tier3_exact_matches(self) -> None:
        excluded = _get_tier3_exact_matches(_REGISTRY_SNIPPET)
        assert "STU_ID" in excluded
        assert "SCH_ID" in excluded
        assert "X1TXMTSC" not in excluded  # valid predictor

    def test_tier3_empty(self) -> None:
        excluded = _get_tier3_exact_matches({})
        assert excluded == set()


# ---------------------------------------------------------------------------
# Unit tests — _validate_spec()
# ---------------------------------------------------------------------------


class TestValidateSpec:
    def _agent(self, tmp_path: Path) -> ProblemFormulator:
        return _make_agent(tmp_path)

    def test_valid_spec_no_warnings(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        warnings = agent._validate_spec(_VALID_RESEARCH_SPEC, _REGISTRY_SNIPPET)
        temporal_warnings = [w for w in warnings if "TEMPORAL" in w]
        assert not temporal_warnings, f"Unexpected temporal warnings: {temporal_warnings}"

    def test_temporal_violation_same_wave(self, tmp_path: Path) -> None:
        """Predictor at same wave as outcome triggers a warning."""
        agent = self._agent(tmp_path)
        spec = {
            **_VALID_RESEARCH_SPEC,
            "outcome_variable": "X3TGPAMAT",  # wave=second_follow_up
            "predictor_set": [
                {
                    "variable": "X1TXMTSC",
                    "rationale": "Prior math.",
                    "wave": "second_follow_up",  # same wave as outcome → violation
                },
            ],
        }
        warnings = agent._validate_spec(spec, _REGISTRY_SNIPPET)
        assert any("TEMPORAL VIOLATION" in w for w in warnings), (
            f"Expected TEMPORAL VIOLATION warning; got: {warnings}"
        )

    def test_temporal_violation_future_wave(self, tmp_path: Path) -> None:
        """Predictor from a later wave than outcome triggers a warning."""
        agent = self._agent(tmp_path)
        spec = {
            **_VALID_RESEARCH_SPEC,
            "outcome_variable": "X3TGPAMAT",  # wave=second_follow_up (idx=2)
            "predictor_set": [
                {
                    "variable": "X4EVRATNDCLG",
                    "rationale": "College attendance.",
                    "wave": "update_panel",  # idx=3 > idx=2 → violation
                },
            ],
        }
        warnings = agent._validate_spec(spec, _REGISTRY_SNIPPET)
        assert any("TEMPORAL VIOLATION" in w for w in warnings)

    def test_valid_first_followup_predictor_for_x3_outcome(self, tmp_path: Path) -> None:
        """first_follow_up predictor is valid for second_follow_up outcome."""
        agent = self._agent(tmp_path)
        spec = {
            **_VALID_RESEARCH_SPEC,
            "outcome_variable": "X3TGPAMAT",  # second_follow_up
            "predictor_set": [
                {
                    "variable": "X2TXMTSC",
                    "rationale": "11th-grade math.",
                    "wave": "first_follow_up",  # idx=1 < idx=2 → valid
                },
            ],
        }
        warnings = agent._validate_spec(spec, _REGISTRY_SNIPPET)
        temporal_violations = [w for w in warnings if "TEMPORAL VIOLATION" in w]
        assert not temporal_violations

    def test_unknown_predictor_wave_warns(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        spec = {
            **_VALID_RESEARCH_SPEC,
            "predictor_set": [
                {
                    "variable": "X1TXMTSC",
                    "rationale": "Prior math.",
                    "wave": "unknown_wave",
                },
            ],
        }
        warnings = agent._validate_spec(spec, _REGISTRY_SNIPPET)
        assert any("unknown wave" in w for w in warnings)

    def test_outcome_not_in_registry_warns(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        spec = {**_VALID_RESEARCH_SPEC, "outcome_variable": "NONEXISTENT_VAR"}
        warnings = agent._validate_spec(spec, _REGISTRY_SNIPPET)
        assert any("not found in registry" in w for w in warnings)

    def test_novelty_score_below_3_warns(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        spec = {**_VALID_RESEARCH_SPEC, "novelty_score_self_assessment": 2}
        warnings = agent._validate_spec(spec, _REGISTRY_SNIPPET)
        assert any("novelty_score_self_assessment" in w for w in warnings)

    def test_novelty_score_exactly_3_no_warning(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        spec = {**_VALID_RESEARCH_SPEC, "novelty_score_self_assessment": 3}
        warnings = agent._validate_spec(spec, _REGISTRY_SNIPPET)
        novelty_warnings = [w for w in warnings if "novelty_score_self_assessment" in w]
        assert not novelty_warnings

    def test_feasibility_low_retention_warns(self, tmp_path: Path) -> None:
        """Very high aggregate missingness should trigger a feasibility warning."""
        agent = self._agent(tmp_path)
        # Craft a spec with high-missingness predictors
        spec = {
            **_VALID_RESEARCH_SPEC,
            "outcome_variable": "X4EVRATNDCLG",  # 22.8% missing
            "predictor_set": [
                # X2TXMTSC has 24.3% missing; outcome has 22.8%
                # total ~47% missing → estimated_n ≈ 0.53 * 23503 ≈ 12456, but
                # use a registry where missingness is very high
                {
                    "variable": "X2TXMTSC",
                    "rationale": "11th grade math.",
                    "wave": "first_follow_up",
                },
            ],
        }
        # Patch registry to have very high missingness so the estimate is < 10000
        high_missing_registry = {
            **_REGISTRY_SNIPPET,
            "variables": {
                "outcomes": [
                    {
                        "name": "X4EVRATNDCLG",
                        "wave": "update_panel",
                        "pct_missing": 60.0,  # 60% missing on outcome
                    }
                ],
                "predictors": {
                    "academic_followup": [
                        {
                            "name": "X2TXMTSC",
                            "wave": "first_follow_up",
                            "pct_missing": 50.0,  # 50% missing on predictor
                        }
                    ]
                },
            },
        }
        warnings = agent._validate_spec(spec, high_missing_registry)
        assert any("below 10,000" in w or "10,000" in w for w in warnings), (
            f"Expected feasibility warning; got: {warnings}"
        )

    def test_empty_predictor_set_no_temporal_warnings(self, tmp_path: Path) -> None:
        """Empty predictor set should produce no temporal warnings."""
        agent = self._agent(tmp_path)
        spec = {**_VALID_RESEARCH_SPEC, "predictor_set": []}
        warnings = agent._validate_spec(spec, _REGISTRY_SNIPPET)
        temporal_warnings = [w for w in warnings if "TEMPORAL VIOLATION" in w]
        assert not temporal_warnings


# ---------------------------------------------------------------------------
# Unit tests — _search_semantic_scholar()
# ---------------------------------------------------------------------------


class TestSearchSemanticScholar:
    """Tests for _search_semantic_scholar() and _run_single_s2_query() HTTP behavior.

    Each test stubs ``_generate_search_queries`` to return a single query so that
    only one HTTP call-chain is exercised per test.  Multi-query / dedup behavior
    has its own test class (TestMultiQuerySearchBehavior) below.
    """

    def _agent(self, tmp_path: Path) -> ProblemFormulator:
        agent = _make_agent(tmp_path)
        # Stub query generation → single query so HTTP side_effects stay predictable
        agent._generate_search_queries = MagicMock(return_value=["edm prediction test query"])
        return agent

    def test_success_returns_papers(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {
                    "paperId": "p1",
                    "title": "EDM Study",
                    "authors": [{"name": "Author A"}],
                    "year": 2022,
                    "abstract": "Abstract here.",
                }
            ]
        }
        with patch("requests.get", return_value=mock_response):
            result = agent._search_semantic_scholar("math GPA prediction")

        assert result["papers"]
        assert result["papers"][0]["paperId"] == "p1"
        assert result["papers"][0]["authors"] == ["Author A"]

    def test_non_200_response_returns_empty_papers(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        mock_429 = MagicMock()
        mock_429.status_code = 429
        with patch("requests.get", return_value=mock_429), patch("time.sleep"):
            result = agent._search_semantic_scholar("some query")

        assert result["papers"] == []
        assert "no results" in result["novelty_evidence"].lower()

    def test_connection_error_returns_empty_papers(self, tmp_path: Path) -> None:
        import requests as req_module
        agent = self._agent(tmp_path)

        with patch("requests.get", side_effect=req_module.ConnectionError("unreachable")):
            result = agent._search_semantic_scholar("some query")

        assert result["papers"] == []
        # search_query is the first generated query (stubbed above)
        assert result["search_query"] == "edm prediction test query"

    def test_timeout_returns_empty_papers(self, tmp_path: Path) -> None:
        import requests as req_module
        agent = self._agent(tmp_path)

        with patch("requests.get", side_effect=req_module.Timeout("timed out")), \
             patch("time.sleep"):
            result = agent._search_semantic_scholar(None)

        assert result["papers"] == []

    def test_default_query_used_when_user_prompt_none(self, tmp_path: Path) -> None:
        # Use a real agent (no generate stub) to verify defaults are used
        agent = _make_agent(tmp_path)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": []}
        with patch("requests.get", return_value=mock_response), patch("time.sleep"):
            result = agent._search_semantic_scholar(None)

        assert result["search_query"]
        assert len(result["search_query"]) > 10

    def test_year_filter_applied_in_request(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": []}
        with patch("requests.get", return_value=mock_response) as mock_get, \
             patch("time.sleep"):
            agent._search_semantic_scholar("test query")

        call_kwargs = mock_get.call_args
        params = call_kwargs.kwargs.get("params", {}) if call_kwargs.kwargs else {}
        assert "year" in params

    def test_retry_on_429_succeeds_on_second_attempt(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        mock_429 = MagicMock(status_code=429)
        mock_200 = MagicMock(status_code=200)
        mock_200.json.return_value = {"data": [{"paperId": "p1", "title": "T", "authors": [], "year": 2022, "abstract": ""}]}

        with patch("requests.get", side_effect=[mock_429, mock_200]) as mock_get, \
             patch("time.sleep"):
            result = agent._search_semantic_scholar("test")

        assert result["papers"]
        assert mock_get.call_count == 2
        assert any("retry" in e.get("message", "").lower() for e in agent.ctx.log)

    def test_retry_on_500_succeeds_on_third_attempt(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        mock_500 = MagicMock(status_code=500)
        mock_503 = MagicMock(status_code=503)
        mock_200 = MagicMock(status_code=200)
        mock_200.json.return_value = {"data": [{"paperId": "p2", "title": "T2", "authors": [], "year": 2023, "abstract": ""}]}

        with patch("requests.get", side_effect=[mock_500, mock_503, mock_200]) as mock_get, \
             patch("time.sleep"):
            result = agent._search_semantic_scholar("test")

        assert result["papers"]
        assert mock_get.call_count == 3

    def test_no_retry_on_400(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        mock_400 = MagicMock(status_code=400)

        with patch("requests.get", return_value=mock_400) as mock_get, \
             patch("time.sleep"):
            result = agent._search_semantic_scholar("test")

        assert result["papers"] == []
        assert mock_get.call_count == 1

    def test_no_retry_on_403(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        mock_403 = MagicMock(status_code=403)

        with patch("requests.get", return_value=mock_403) as mock_get, \
             patch("time.sleep"):
            result = agent._search_semantic_scholar("test")

        assert result["papers"] == []
        assert mock_get.call_count == 1

    def test_all_retries_exhausted(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        max_retries = agent.config.get("semantic_scholar", {}).get("max_retries", 3)
        mock_429 = MagicMock(status_code=429)

        with patch("requests.get", return_value=mock_429) as mock_get, \
             patch("time.sleep"):
            result = agent._search_semantic_scholar("test")

        assert result["papers"] == []
        assert mock_get.call_count == max_retries + 1
        # novelty_evidence describes zero-results condition
        assert "no results" in result["novelty_evidence"].lower()
        assert any("failed" in e.get("message", "").lower() for e in agent.ctx.log)

    def test_retry_on_connection_error_then_success(self, tmp_path: Path) -> None:
        import requests as req_module
        agent = self._agent(tmp_path)
        mock_200 = MagicMock(status_code=200)
        mock_200.json.return_value = {"data": [{"paperId": "p3", "title": "T3", "authors": [], "year": 2021, "abstract": ""}]}

        with patch("requests.get", side_effect=[req_module.ConnectionError("down"), mock_200]) as mock_get, \
             patch("time.sleep"):
            result = agent._search_semantic_scholar("test")

        assert result["papers"]
        assert mock_get.call_count == 2

    def test_retry_on_timeout_then_success(self, tmp_path: Path) -> None:
        import requests as req_module
        agent = self._agent(tmp_path)
        mock_200 = MagicMock(status_code=200)
        mock_200.json.return_value = {"data": [{"paperId": "p4", "title": "T4", "authors": [], "year": 2020, "abstract": ""}]}

        with patch("requests.get", side_effect=[req_module.Timeout("timed out"), mock_200]) as mock_get, \
             patch("time.sleep"):
            result = agent._search_semantic_scholar("test")

        assert result["papers"]
        assert mock_get.call_count == 2

    def test_backoff_delay_increases(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        agent.config["semantic_scholar"]["max_retries"] = 2
        agent.config["semantic_scholar"]["backoff_base_s"] = 1.0
        agent.config["semantic_scholar"]["backoff_factor"] = 2.0
        agent.config["semantic_scholar"]["backoff_jitter"] = False
        mock_429 = MagicMock(status_code=429)

        sleep_calls: list[float] = []
        with patch("requests.get", return_value=mock_429), \
             patch("time.sleep", side_effect=lambda s: sleep_calls.append(s)):
            agent._search_semantic_scholar("test")

        # sleep_calls[0] = initial request_delay_s (0.5s)
        # sleep_calls[1] = retry 1: 1.0s, sleep_calls[2] = retry 2: 2.0s
        assert len(sleep_calls) == 3
        assert sleep_calls[1] == pytest.approx(1.0, rel=0.01)
        assert sleep_calls[2] == pytest.approx(2.0, rel=0.01)


# ---------------------------------------------------------------------------
# Unit tests — _generate_search_queries()
# ---------------------------------------------------------------------------


class TestGenerateSearchQueries:
    """Tests for the LLM-based query generation helper."""

    def _agent(self, tmp_path: Path) -> ProblemFormulator:
        return _make_agent(tmp_path)

    def test_returns_defaults_when_no_user_prompt(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        queries = agent._generate_search_queries(None)
        assert queries == ProblemFormulator._DEFAULT_S2_QUERIES

    def test_returns_defaults_on_empty_string(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        queries = agent._generate_search_queries("")
        assert queries == ProblemFormulator._DEFAULT_S2_QUERIES

    def test_llm_valid_json_array_returned(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        agent.call_llm = MagicMock(return_value='["college enrollment prediction", "machine learning education", "high school outcomes"]')
        queries = agent._generate_search_queries("Predict college enrollment")
        assert queries == ["college enrollment prediction", "machine learning education", "high school outcomes"]

    def test_llm_json_with_fences_parsed(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        agent.call_llm = MagicMock(return_value='```json\n["q1", "q2", "q3"]\n```')
        queries = agent._generate_search_queries("some topic")
        assert queries == ["q1", "q2", "q3"]

    def test_fallback_to_defaults_on_invalid_json(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        agent.call_llm = MagicMock(return_value="not valid json at all")
        queries = agent._generate_search_queries("some topic")
        assert queries == ProblemFormulator._DEFAULT_S2_QUERIES

    def test_fallback_to_defaults_on_llm_exception(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        agent.call_llm = MagicMock(side_effect=RuntimeError("api down"))
        queries = agent._generate_search_queries("some topic")
        assert queries == ProblemFormulator._DEFAULT_S2_QUERIES
        assert any("Query generation failed" in e.get("message", "") for e in agent.ctx.log)

    def test_successful_queries_logged(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        agent.call_llm = MagicMock(return_value='["query one", "query two", "query three"]')
        agent._generate_search_queries("topic")
        assert any("S2 keyword queries" in e.get("message", "") for e in agent.ctx.log)

    def test_trims_to_max_three_queries(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        agent.call_llm = MagicMock(return_value='["q1", "q2", "q3", "q4", "q5"]')
        queries = agent._generate_search_queries("topic")
        assert len(queries) <= 3

    def test_ignores_empty_strings_in_llm_response(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        agent.call_llm = MagicMock(return_value='["valid query", "", "  ", "another valid"]')
        queries = agent._generate_search_queries("topic")
        assert all(q.strip() for q in queries)


# ---------------------------------------------------------------------------
# Unit tests — multi-query merge / dedup behavior
# ---------------------------------------------------------------------------


class TestMultiQuerySearchBehavior:
    """Tests for the 3-query dedup / merge / year-sort behavior added in v1.1."""

    def _agent(self, tmp_path: Path) -> ProblemFormulator:
        return _make_agent(tmp_path)

    def _make_mock_200(self, papers: list[dict]) -> MagicMock:
        m = MagicMock(status_code=200)
        m.json.return_value = {"data": papers}
        return m

    def test_three_queries_each_produce_one_http_call(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        agent._generate_search_queries = MagicMock(return_value=["q1", "q2", "q3"])
        mock_200 = self._make_mock_200([])
        with patch("requests.get", return_value=mock_200) as mock_get, \
             patch("time.sleep"):
            agent._search_semantic_scholar("test topic")
        assert mock_get.call_count == 3

    def test_duplicate_papers_deduped_by_paper_id(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        agent._generate_search_queries = MagicMock(return_value=["q1", "q2"])
        paper = {"paperId": "dup001", "title": "Duplicate Paper", "authors": [], "year": 2022, "abstract": ""}
        mock_200 = self._make_mock_200([paper])
        with patch("requests.get", return_value=mock_200), patch("time.sleep"):
            result = agent._search_semantic_scholar("test")
        # Same paper from both queries → only 1 unique result
        assert len(result["papers"]) == 1
        assert result["papers"][0]["paperId"] == "dup001"

    def test_results_merged_from_multiple_queries(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        agent._generate_search_queries = MagicMock(return_value=["q1", "q2"])
        papers_q1 = [{"paperId": "a1", "title": "Paper A", "authors": [], "year": 2022, "abstract": ""}]
        papers_q2 = [{"paperId": "b2", "title": "Paper B", "authors": [], "year": 2021, "abstract": ""}]
        m1 = self._make_mock_200(papers_q1)
        m2 = self._make_mock_200(papers_q2)
        with patch("requests.get", side_effect=[m1, m2]), patch("time.sleep"):
            result = agent._search_semantic_scholar("test")
        ids = {p["paperId"] for p in result["papers"]}
        assert "a1" in ids
        assert "b2" in ids

    def test_results_sorted_by_year_descending(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        agent._generate_search_queries = MagicMock(return_value=["q1", "q2"])
        p_old = {"paperId": "old", "title": "Old Paper", "authors": [], "year": 2015, "abstract": ""}
        p_new = {"paperId": "new", "title": "New Paper", "authors": [], "year": 2023, "abstract": ""}
        m1 = self._make_mock_200([p_old])
        m2 = self._make_mock_200([p_new])
        with patch("requests.get", side_effect=[m1, m2]), patch("time.sleep"):
            result = agent._search_semantic_scholar("test")
        years = [p["year"] for p in result["papers"]]
        assert years == sorted(years, reverse=True)

    def test_inter_query_sleep_called_between_queries(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        agent._generate_search_queries = MagicMock(return_value=["q1", "q2", "q3"])
        mock_200 = self._make_mock_200([])
        sleep_calls: list[float] = []
        with patch("requests.get", return_value=mock_200), \
             patch("time.sleep", side_effect=lambda s: sleep_calls.append(s)):
            agent._search_semantic_scholar("test")
        # 1 initial delay (delay_s) per query + 2 inter-query 1.0s sleeps
        inter_query_sleeps = [s for s in sleep_calls if abs(s - 1.0) < 0.01]
        assert len(inter_query_sleeps) == 2

    def test_primary_query_stored_as_search_query(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        agent._generate_search_queries = MagicMock(return_value=["first query", "second query"])
        mock_200 = self._make_mock_200([{"paperId": "p1", "title": "T", "authors": [], "year": 2022, "abstract": ""}])
        with patch("requests.get", return_value=mock_200), patch("time.sleep"):
            result = agent._search_semantic_scholar("test")
        assert result["search_query"] == "first query"

    def test_per_query_counts_logged(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        agent._generate_search_queries = MagicMock(return_value=["q1", "q2"])
        m1 = self._make_mock_200([{"paperId": "p1", "title": "T", "authors": [], "year": 2022, "abstract": ""}])
        m2 = self._make_mock_200([])
        with patch("requests.get", side_effect=[m1, m2]), patch("time.sleep"):
            agent._search_semantic_scholar("test")
        log_msgs = [e.get("message", "") for e in agent.ctx.log]
        assert any("1/2" in m for m in log_msgs)
        assert any("2/2" in m for m in log_msgs)


# ---------------------------------------------------------------------------
# Unit tests — run() orchestration
# ---------------------------------------------------------------------------


class TestRunOrchestration:
    def _agent(self, tmp_path: Path) -> ProblemFormulator:
        return _make_agent(tmp_path)

    def _make_llm_response(self) -> str:
        payload = {
            "research_spec": _VALID_RESEARCH_SPEC,
            "literature_context": _VALID_LIT_CONTEXT,
        }
        return f"```json\n{json.dumps(payload)}\n```"

    def test_run_returns_research_spec_and_lit_context_keys(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        agent.call_llm = MagicMock(return_value=self._make_llm_response())
        with patch("requests.get", side_effect=Exception("no S2")):
            result = agent.run()

        assert "research_spec" in result
        assert "literature_context" in result

    def test_run_uses_llm_literature_context(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        agent.call_llm = MagicMock(return_value=self._make_llm_response())
        with patch("requests.get", side_effect=Exception("no S2")):
            result = agent.run()

        assert result["literature_context"]["search_query"] == _VALID_LIT_CONTEXT["search_query"]

    def test_run_falls_back_to_s2_when_llm_omits_lit_context(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        # LLM response without literature_context key
        payload = {"research_spec": _VALID_RESEARCH_SPEC}
        agent.call_llm = MagicMock(return_value=f"```json\n{json.dumps(payload)}\n```")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"paperId": "s2_paper", "title": "S2 Result", "authors": [], "year": 2022, "abstract": ""}]
        }
        with patch("requests.get", return_value=mock_response):
            result = agent.run()

        # Falls back to s2_context (papers from S2)
        assert result["literature_context"]["papers"][0]["paperId"] == "s2_paper"

    def test_run_with_revision_instructions_includes_in_message(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        agent.call_llm = MagicMock(return_value=self._make_llm_response())
        with patch("requests.get", side_effect=Exception("no S2")):
            agent.run(revision_instructions="Fix the temporal leakage in predictor X2TXMTSC.")

        call_args = agent.call_llm.call_args
        user_message = call_args[0][0]
        assert "Fix the temporal leakage" in user_message

    def test_run_with_user_prompt_includes_in_message(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        agent.call_llm = MagicMock(return_value=self._make_llm_response())
        with patch("requests.get", side_effect=Exception("no S2")):
            agent.run(user_prompt="Focus on STEM dropout prediction.")

        user_message = agent.call_llm.call_args[0][0]
        assert "STEM dropout prediction" in user_message

    def test_run_logs_temporal_violations(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        # Build a response with a temporal violation
        bad_spec = {
            **_VALID_RESEARCH_SPEC,
            "predictor_set": [
                {
                    "variable": "X1TXMTSC",
                    "rationale": "future wave predictor",
                    "wave": "update_panel",  # violation: update_panel > second_follow_up
                }
            ],
        }
        payload = {"research_spec": bad_spec, "literature_context": _VALID_LIT_CONTEXT}
        agent.call_llm = MagicMock(return_value=f"```json\n{json.dumps(payload)}\n```")
        with patch("requests.get", side_effect=Exception("no S2")):
            agent.run()

        log_messages = [entry.get("message", "") for entry in agent.ctx.log]
        assert any("TEMPORAL VIOLATION" in msg for msg in log_messages), (
            f"Expected TEMPORAL VIOLATION in log; got: {log_messages}"
        )

    def test_run_calls_call_llm_once(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        agent.call_llm = MagicMock(return_value=self._make_llm_response())
        with patch("requests.get", side_effect=Exception("no S2")):
            agent.run()

        assert agent.call_llm.call_count == 1


# ---------------------------------------------------------------------------
# Integration test — full ProblemFormulator run with real LLM
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_problem_formulator_full_run(tmp_path: Path) -> None:
    """Full integration: real LLM call + S2 API call.

    Requires ANTHROPIC_API_KEY in environment.
    Run with:  pytest tests/test_problem_formulator.py -v --run-integration
    """
    config = load_config(CONFIG_PATH)
    ctx = PipelineContext(
        dataset_name="hsls09_public",
        raw_data_path="data/raw/nonexistent.csv",
        output_dir=str(tmp_path),
        max_revision_cycles=2,
    )
    agent = ProblemFormulator(ctx, "problem_formulator", config)
    result = agent.run(user_prompt="Predict high school dropout from 9th-grade factors.")

    # --- required keys present ---
    assert "research_spec" in result
    assert "literature_context" in result

    spec = result["research_spec"]
    lit = result["literature_context"]

    # --- research_spec schema ---
    assert spec.get("research_question"), "research_question must be non-empty"
    assert spec.get("outcome_variable"), "outcome_variable must be non-empty"
    assert spec.get("outcome_type") in ("binary", "continuous")
    predictor_set = spec.get("predictor_set", [])
    assert len(predictor_set) >= 3, "At least 3 predictors required"
    for pred in predictor_set:
        assert "variable" in pred
        assert "rationale" in pred
        assert "wave" in pred
        assert pred["wave"] in TEMPORAL_ORDER

    # --- novelty score ---
    assert spec.get("novelty_score_self_assessment", 0) >= 3, (
        "novelty_score_self_assessment must be ≥ 3"
    )

    # --- temporal ordering ---
    warnings = agent._validate_spec(spec, agent.load_registry())
    temporal_violations = [w for w in warnings if "TEMPORAL VIOLATION" in w]
    assert not temporal_violations, (
        f"Temporal violations in returned spec: {temporal_violations}"
    )

    # --- literature_context schema ---
    assert "search_query" in lit
    assert "papers" in lit
    assert isinstance(lit["papers"], list)
    assert "novelty_evidence" in lit


# ---------------------------------------------------------------------------
# Unit tests — N-branch / FindingsMemory integration
# ---------------------------------------------------------------------------


class TestMultiBranchOrchestration:
    """Tests for the N-branch candidate generation path."""

    def _agent(self, tmp_path: Path) -> ProblemFormulator:
        return _make_agent(tmp_path)

    def _make_llm_response(self, outcome: str = "X3TGPAMAT", novelty: int = 4) -> str:
        spec = {**_VALID_RESEARCH_SPEC, "outcome_variable": outcome, "novelty_score_self_assessment": novelty}
        payload = {"research_spec": spec, "literature_context": _VALID_LIT_CONTEXT}
        return f"```json\n{json.dumps(payload)}\n```"

    def test_run_single_unchanged_calls_llm_once(self, tmp_path: Path) -> None:
        """n_candidate_specs=1 must call call_llm exactly once (existing behavior)."""
        agent = self._agent(tmp_path)
        agent.call_llm = MagicMock(return_value=self._make_llm_response())
        with patch("requests.get", side_effect=Exception("no S2")):
            agent.run(n_candidate_specs=1)
        assert agent.call_llm.call_count == 1

    def test_run_multi_branch_calls_llm_n_times(self, tmp_path: Path) -> None:
        """n_candidate_specs=3 must call call_llm exactly 3 times."""
        agent = self._agent(tmp_path)
        agent.call_llm = MagicMock(return_value=self._make_llm_response())
        with patch("requests.get", side_effect=Exception("no S2")):
            agent.run(n_candidate_specs=3)
        assert agent.call_llm.call_count == 3

    def test_run_multi_branch_returns_best_unstudied_outcome(self, tmp_path: Path) -> None:
        """When one candidate studies an already-studied outcome, prefer the other."""
        agent = self._agent(tmp_path)
        # Candidate 1: already studied outcome (X3TGPAMAT, novelty=4)
        # Candidate 2: unstudied outcome (X4EVRATNDCLG, novelty=3)
        responses = [
            self._make_llm_response(outcome="X3TGPAMAT", novelty=4),
            self._make_llm_response(outcome="X4EVRATNDCLG", novelty=3),
        ]
        agent.call_llm = MagicMock(side_effect=responses)
        with patch("requests.get", side_effect=Exception("no S2")):
            result = agent.run(
                n_candidate_specs=2,
                studied_outcomes=["X3TGPAMAT"],
            )
        # X4EVRATNDCLG is unstudied → scores +2; should beat X3TGPAMAT (studied)
        assert result["research_spec"]["outcome_variable"] == "X4EVRATNDCLG"

    def test_memory_summary_injected_in_message(self, tmp_path: Path) -> None:
        """findings_memory_summary should appear in the user message passed to call_llm."""
        agent = self._agent(tmp_path)
        agent.call_llm = MagicMock(return_value=self._make_llm_response())
        with patch("requests.get", side_effect=Exception("no S2")):
            agent.run(findings_memory_summary="Prior run: outcome=X2TXMTSCOR")
        user_message = agent.call_llm.call_args[0][0]
        assert "Prior run: outcome=X2TXMTSCOR" in user_message

    def test_studied_outcomes_injected_in_message(self, tmp_path: Path) -> None:
        """studied_outcomes should appear in the user message."""
        agent = self._agent(tmp_path)
        agent.call_llm = MagicMock(return_value=self._make_llm_response())
        with patch("requests.get", side_effect=Exception("no S2")):
            agent.run(studied_outcomes=["X2TXMTSCOR", "X3TGPAMAT"])
        user_message = agent.call_llm.call_args[0][0]
        assert "X2TXMTSCOR" in user_message
        assert "X3TGPAMAT" in user_message

    def test_multi_branch_prior_specs_injected_in_subsequent_calls(self, tmp_path: Path) -> None:
        """Each call after the first should include prior candidate summaries."""
        agent = self._agent(tmp_path)
        captured_messages: list[str] = []

        def capture_call(msg: str, **kwargs) -> str:
            captured_messages.append(msg)
            return self._make_llm_response()

        agent.call_llm = MagicMock(side_effect=capture_call)
        with patch("requests.get", side_effect=Exception("no S2")):
            agent.run(n_candidate_specs=3)

        # First call has no prior specs section; second and third should
        assert "Prior Candidate Specs" not in captured_messages[0]
        assert "Prior Candidate Specs" in captured_messages[1]
        assert "Prior Candidate Specs" in captured_messages[2]

    def test_multi_branch_fallback_on_all_parse_failures(self, tmp_path: Path) -> None:
        """If all branches fail to parse, fall back to single-branch (one more call)."""
        agent = self._agent(tmp_path)
        responses = ["not valid json", "also bad", self._make_llm_response()]
        agent.call_llm = MagicMock(side_effect=responses)
        with patch("requests.get", side_effect=Exception("no S2")):
            result = agent.run(n_candidate_specs=2)
        # 2 failed branches + 1 single-branch fallback = 3 calls
        assert agent.call_llm.call_count == 3
        assert "research_spec" in result


# ---------------------------------------------------------------------------
# Unit tests — three-layer citation verification helpers
# ---------------------------------------------------------------------------


class TestTokenizeTitle:
    def test_basic_tokenization(self) -> None:
        tokens = _tokenize_title("Predicting Math GPA")
        assert tokens == {"predicting", "math", "gpa"}

    def test_punctuation_removed(self) -> None:
        tokens = _tokenize_title("Learning Analytics: A Survey")
        assert "analytics" in tokens
        assert ":" not in str(tokens)

    def test_empty_string(self) -> None:
        assert _tokenize_title("") == set()

    def test_deduplication(self) -> None:
        tokens = _tokenize_title("math math math")
        assert len(tokens) == 1


class TestJaccardSimilarity:
    def test_identical_sets(self) -> None:
        s = {"a", "b", "c"}
        assert _jaccard_similarity(s, s) == 1.0

    def test_disjoint_sets(self) -> None:
        assert _jaccard_similarity({"a"}, {"b"}) == 0.0

    def test_partial_overlap(self) -> None:
        a = {"a", "b", "c"}
        b = {"b", "c", "d"}
        # intersection=2, union=4
        assert _jaccard_similarity(a, b) == pytest.approx(0.5)

    def test_empty_set_returns_zero(self) -> None:
        assert _jaccard_similarity(set(), {"a", "b"}) == 0.0
        assert _jaccard_similarity({"a"}, set()) == 0.0


class TestVerifyPaperThreeLayers:
    def _real_ids(self) -> set[str]:
        return {"abc123", "def456"}

    def _real_title_tokens(self) -> list[tuple[set[str], dict]]:
        return [
            (_tokenize_title("Predicting STEM Outcomes in High School"), {"paperId": "abc123"}),
            (_tokenize_title("Machine Learning for Educational Data Mining"), {"paperId": "def456"}),
        ]

    def test_layer1_exact_id_returns_verified(self) -> None:
        paper = {"paperId": "abc123", "title": "Some Title"}
        status = _verify_paper_three_layers(paper, self._real_ids(), self._real_title_tokens())
        assert status == "VERIFIED"

    def test_layer3_high_jaccard_returns_suspicious(self) -> None:
        # Title very similar to "Predicting STEM Outcomes in High School" (> 0.80 overlap)
        paper = {"paperId": "unknown999", "title": "Predicting STEM Outcomes in High School"}
        # Mock CrossRef to fail (so layer 3 determines result)
        with patch("requests.get", side_effect=Exception("no crossref")):
            status = _verify_paper_three_layers(paper, self._real_ids(), self._real_title_tokens())
        assert status == "SUSPICIOUS"

    def test_hallucinated_paper_returns_hallucinated(self) -> None:
        paper = {"paperId": "zzz999", "title": "Completely Unrelated Paper About Cooking"}
        with patch("requests.get", side_effect=Exception("no crossref")):
            status = _verify_paper_three_layers(paper, self._real_ids(), self._real_title_tokens())
        assert status == "HALLUCINATED"

    def test_paper_without_id_can_still_match_by_title(self) -> None:
        paper = {"title": "Predicting STEM Outcomes in High School"}  # no paperId
        with patch("requests.get", side_effect=Exception("no crossref")):
            status = _verify_paper_three_layers(paper, self._real_ids(), self._real_title_tokens())
        assert status == "SUSPICIOUS"

    def test_no_matching_id_or_title_returns_hallucinated(self) -> None:
        """Paper with unrecognized ID and unrelated title is HALLUCINATED."""
        paper = {"paperId": "unknown999", "title": "A Completely Unrelated Subject"}
        with patch("requests.get", side_effect=Exception("no crossref")):
            status = _verify_paper_three_layers(paper, set(), [])
        assert status == "HALLUCINATED"


class TestFilterHallucinatedPapers:
    """Tests for ProblemFormulator._filter_hallucinated_papers() (3-layer wrapper)."""

    def _agent(self, tmp_path: Path) -> ProblemFormulator:
        config = load_config(CONFIG_PATH)
        ctx = _make_ctx(tmp_path)
        with patch("anthropic.Anthropic"):
            return ProblemFormulator(ctx, "problem_formulator", config)

    def _s2_context(self) -> dict:
        return {
            "papers": [
                {
                    "paperId": "real001",
                    "title": "Predicting Math GPA Using Machine Learning",
                    "authors": ["Smith, J."],
                    "year": 2022,
                },
            ]
        }

    def test_exact_id_match_verified(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        lit = {
            "search_query": "test",
            "papers": [{"paperId": "real001", "title": "Predicting Math GPA Using Machine Learning"}],
            "novelty_evidence": "",
        }
        result = agent._filter_hallucinated_papers(lit, self._s2_context())
        assert len(result["papers"]) == 1
        assert result["papers"][0]["verification_status"] == "VERIFIED"

    def test_hallucinated_paper_dropped(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        lit = {
            "search_query": "test",
            "papers": [{"paperId": "fake999", "title": "An Entirely Fabricated Paper About Nothing"}],
            "novelty_evidence": "",
        }
        with patch("requests.get", side_effect=Exception("no crossref")):
            result = agent._filter_hallucinated_papers(lit, self._s2_context())
        assert len(result["papers"]) == 0

    def test_s2_empty_returns_empty_papers(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        lit = {
            "search_query": "test",
            "papers": [{"paperId": "real001", "title": "Predicting Math GPA"}],
            "novelty_evidence": "prior work",
        }
        s2_empty = {"papers": []}
        result = agent._filter_hallucinated_papers(lit, s2_empty)
        assert result["papers"] == []

    def test_no_crossref_call_when_s2_empty(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        lit = {"search_query": "", "papers": [{"paperId": "x", "title": "T"}], "novelty_evidence": ""}
        with patch("requests.get") as mock_get:
            agent._filter_hallucinated_papers(lit, {"papers": []})
        mock_get.assert_not_called()

    def test_jaccard_title_match_marked_suspicious(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        # Paper not in S2 IDs but title matches via Layer 3 Jaccard
        lit = {
            "search_query": "test",
            "papers": [{"paperId": "unknown", "title": "Predicting Math GPA Using Machine Learning"}],
            "novelty_evidence": "",
        }
        with patch("requests.get", side_effect=Exception("no crossref")):
            result = agent._filter_hallucinated_papers(lit, self._s2_context())
        # Should be SUSPICIOUS (title matches Layer 3)
        assert len(result["papers"]) == 1
        assert result["papers"][0]["verification_status"] == "SUSPICIOUS"


# ---------------------------------------------------------------------------
# arXiv search tests
# ---------------------------------------------------------------------------

_ARXIV_XML_RESPONSE = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.12345v1</id>
    <title>Predicting Student Achievement with Machine Learning</title>
    <summary>This paper studies ML prediction of student outcomes.</summary>
    <published>2024-01-15T00:00:00Z</published>
    <author><name>Jane Smith</name></author>
    <author><name>John Doe</name></author>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2403.67890v2</id>
    <title>Deep Learning for Educational Data Mining</title>
    <summary>A survey of deep learning methods in EDM.</summary>
    <published>2024-03-20T00:00:00Z</published>
    <author><name>Alice Wang</name></author>
  </entry>
</feed>
"""


class TestArxivSearch:
    """Tests for _search_arxiv method."""

    def _agent(self, tmp_path: Path) -> ProblemFormulator:
        return _make_agent(tmp_path)

    def test_parses_arxiv_xml_correctly(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = _ARXIV_XML_RESPONSE

        with patch("requests.get", return_value=mock_resp):
            papers = agent._search_arxiv(["test query"], max_results_per_query=10)

        assert len(papers) == 2
        assert papers[0]["paperId"] == "arxiv:2401.12345v1"
        assert papers[0]["title"] == "Predicting Student Achievement with Machine Learning"
        assert papers[0]["year"] == 2024
        assert papers[0]["authors"] == ["Jane Smith", "John Doe"]
        assert papers[0]["source"] == "arxiv"

    def test_deduplicates_across_queries(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = _ARXIV_XML_RESPONSE

        with patch("requests.get", return_value=mock_resp):
            # Same XML for both queries — should deduplicate
            papers = agent._search_arxiv(["query1", "query2"], max_results_per_query=10)

        assert len(papers) == 2  # not 4

    def test_handles_http_error_gracefully(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        mock_resp = MagicMock()
        mock_resp.status_code = 500

        with patch("requests.get", return_value=mock_resp):
            papers = agent._search_arxiv(["test query"], max_results_per_query=10)

        assert papers == []

    def test_handles_network_error_gracefully(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)

        with patch("requests.get", side_effect=Exception("connection refused")):
            papers = agent._search_arxiv(["test query"], max_results_per_query=10)

        assert papers == []

    def test_logs_results_count(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = _ARXIV_XML_RESPONSE

        with patch("requests.get", return_value=mock_resp):
            agent._search_arxiv(["test query"], max_results_per_query=10)

        log_msgs = [e["message"] for e in agent.ctx.log]
        assert any("arXiv query" in m and "2 results" in m for m in log_msgs)


class TestSearchLiterature:
    """Tests for _search_literature (combined S2 + arXiv)."""

    def _agent(self, tmp_path: Path) -> ProblemFormulator:
        return _make_agent(tmp_path)

    def _mock_s2_context(self) -> dict:
        return {
            "search_query": "test query",
            "papers": [
                {"paperId": "s2_001", "title": "S2 Paper One", "authors": ["A"], "year": 2024, "abstract": "..."},
                {"paperId": "s2_002", "title": "S2 Paper Two", "authors": ["B"], "year": 2023, "abstract": "..."},
            ],
            "novelty_evidence": "",
        }

    def test_merges_s2_and_arxiv_results(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        agent._search_semantic_scholar = MagicMock(return_value=self._mock_s2_context())
        agent._generate_search_queries = MagicMock(return_value=["test query"])

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = _ARXIV_XML_RESPONSE

        with patch("requests.get", return_value=mock_resp):
            result = agent._search_literature("test prompt")

        assert len(result["papers"]) == 4  # 2 S2 + 2 arXiv

    def test_deduplicates_arxiv_against_s2_by_title(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        # S2 has a paper with nearly identical title to one arXiv paper
        s2_ctx = self._mock_s2_context()
        s2_ctx["papers"][0]["title"] = "Predicting Student Achievement with Machine Learning"
        agent._search_semantic_scholar = MagicMock(return_value=s2_ctx)
        agent._generate_search_queries = MagicMock(return_value=["test query"])

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = _ARXIV_XML_RESPONSE

        with patch("requests.get", return_value=mock_resp):
            result = agent._search_literature("test prompt")

        # Should be 3: 2 S2 + 1 arXiv (one arXiv deduped against S2)
        assert len(result["papers"]) == 3

    def test_arxiv_disabled_returns_s2_only(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        agent.config["arxiv"] = {"enabled": False}
        agent._search_semantic_scholar = MagicMock(return_value=self._mock_s2_context())

        result = agent._search_literature("test prompt")

        assert len(result["papers"]) == 2  # S2 only

    def test_arxiv_failure_returns_s2_only(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        agent._search_semantic_scholar = MagicMock(return_value=self._mock_s2_context())
        agent._generate_search_queries = MagicMock(return_value=["test query"])

        with patch("requests.get", side_effect=Exception("arXiv down")):
            result = agent._search_literature("test prompt")

        assert len(result["papers"]) == 2  # S2 only

    def test_logs_merged_count(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        agent._search_semantic_scholar = MagicMock(return_value=self._mock_s2_context())
        agent._generate_search_queries = MagicMock(return_value=["test query"])

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = _ARXIV_XML_RESPONSE

        with patch("requests.get", return_value=mock_resp):
            agent._search_literature("test prompt")

        log_msgs = [e["message"] for e in agent.ctx.log]
        assert any("Literature search merged" in m for m in log_msgs)
