"""Tests for the Writer agent.

Unit tests run without any API calls.
Integration tests (marked @pytest.mark.integration) require ANTHROPIC_API_KEY.

Run unit tests:
    pytest tests/test_writer.py -v -k "not integration"

Run integration tests:
    pytest tests/test_writer.py -v --run-integration
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.writer import Writer, _S2_FAILURE_BIB_COMMENT, _MINIMAL_STUB_TEX
from src.config import load_config
from src.context import PipelineContext

CONFIG_PATH = str(Path(__file__).parent.parent / "config.yaml")

# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------

_RESEARCH_SPEC: dict = {
    "research_question": "Predict 12th-grade math GPA from 9th-grade factors.",
    "outcome_variable": "X3TGPAMAT",
    "outcome_type": "continuous",
    "predictor_set": [
        {"variable": "X1TXMTSC", "rationale": "Prior math achievement.", "wave": "base_year"},
        {"variable": "X1MTHEFF", "rationale": "Math self-efficacy.", "wave": "base_year"},
    ],
    "target_population": "Full HSLS:09 sample",
    "subgroup_analyses": ["X1SEX"],
    "expected_contribution": "Novel motivational combination.",
    "potential_limitations": ["Multilevel structure not modeled"],
    "novelty_score_self_assessment": 4,
}

_LIT_CONTEXT: dict = {
    "search_query": "math GPA prediction EDM",
    "papers": [
        {
            "paperId": "paper001",
            "title": "Predicting Math Achievement",
            "authors": ["Smith, John", "Doe, Jane"],
            "year": 2022,
            "abstract": "We study math achievement prediction.",
            "venue": "Proceedings of EDM 2022",
        },
        {
            "paperId": "paper002",
            "title": "Machine Learning for GPA Prediction",
            "authors": ["Johnson, A."],
            "year": 2021,
            "abstract": "XGBoost for GPA.",
            "venue": "Journal of Educational Data Mining",
        },
    ],
    "novelty_evidence": "Prior work uses test scores, not transcript-based GPA.",
}

_DATA_REPORT: dict = {
    "dataset": "hsls09_public",
    "original_n": 23503,
    "analytic_n": 19240,
    "n_train": 15392,
    "n_test": 3848,
    "outcome_variable": "X3TGPAMAT",
    "outcome_type": "continuous",
    "class_balance": None,
    "n_predictors_raw": 2,
    "n_predictors_encoded": 3,
    "missingness_summary": {},
    "variables_flagged": [],
    "validation_passed": True,
    "warnings": [
        "Multilevel structure (students nested in schools) is not modeled. This is a limitation."
    ],
}

_RESULTS_OBJECT: dict = {
    "best_model": "XGBoost",
    "best_metric_value": 0.61,
    "primary_metric": "RMSE",
    "all_models": {
        "LinearRegression": {"rmse": 0.71, "rmse_ci_lower": 0.69, "rmse_ci_upper": 0.73},
        "XGBoost": {"rmse": 0.61, "rmse_ci_lower": 0.59, "rmse_ci_upper": 0.63},
    },
    "top_features": [
        {"feature": "X1TXMTSC", "shap_mean_abs": 0.18, "direction": "positive"},
    ],
    "subgroup_performance": {
        "X1SEX": {"Male": {"rmse": 0.63, "n": 7800}, "Female": {"rmse": 0.59, "n": 7592}}
    },
    "figures_generated": ["shap_summary.png", "shap_importance.png", "residual_plot.png"],
    "tables_generated": ["model_comparison.csv"],
    "errors": [],
    "warnings": [],
}

_PASS_REVIEW: dict = {
    "overall_verdict": "PASS",
    "overall_quality_score": 8,
    "problem_formulation_review": {"score": 8, "issues": []},
    "data_preparation_review": {"score": 8, "issues": []},
    "analysis_review": {"score": 8, "issues": []},
    "substantive_review": {"score": 8, "educational_meaningfulness": "Good.", "issues": []},
    "revision_instructions": {"ProblemFormulator": None, "DataEngineer": None, "Analyst": None},
}

_SAMPLE_TEX = r"""
\documentclass[sigconf]{acmart}
\usepackage{booktabs}
\usepackage{graphicx}
\title{Test Paper}
\begin{document}
\maketitle
\begin{abstract}
This is the abstract.
\end{abstract}
\section{Introduction}
This paper studies math GPA prediction \cite{paper001}.
\bibliographystyle{ACM-Reference-Format}
\bibliography{references}
\end{document}
""".strip()

_SAMPLE_BIB = """
@inproceedings{paper001,
  author    = {Smith, John and Doe, Jane},
  title     = {Predicting Math Achievement},
  year      = {2022},
  booktitle = {Proceedings of EDM 2022},
}
""".strip()


def _make_ctx(tmp_path: Path) -> PipelineContext:
    ctx = PipelineContext(
        dataset_name="hsls09_public",
        raw_data_path="data/raw/nonexistent.csv",
        output_dir=str(tmp_path),
        max_revision_cycles=2,
    )
    ctx.research_spec = _RESEARCH_SPEC
    ctx.literature_context = _LIT_CONTEXT
    ctx.data_report = _DATA_REPORT
    ctx.results_object = _RESULTS_OBJECT
    ctx.review_report = _PASS_REVIEW
    return ctx


def _make_agent(tmp_path: Path) -> Writer:
    config = load_config(CONFIG_PATH)
    ctx = _make_ctx(tmp_path)
    with patch("anthropic.Anthropic"):
        return Writer(ctx, "writer", config)


def _llm_response_with_both_blocks() -> str:
    """LLM response containing both a ```latex and a ```bibtex block."""
    return (
        "Here is the paper:\n"
        f"```latex\n{_SAMPLE_TEX}\n```\n\n"
        "And the references:\n"
        f"```bibtex\n{_SAMPLE_BIB}\n```\n"
    )


# ---------------------------------------------------------------------------
# Unit tests — _build_bibtex()
# ---------------------------------------------------------------------------


class TestBuildBibtex:
    def _agent(self, tmp_path: Path) -> Writer:
        return _make_agent(tmp_path)

    def test_with_valid_papers_produces_entries(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        bibtex = agent._build_bibtex(_LIT_CONTEXT)
        assert "@inproceedings{paper001" in bibtex
        assert "Smith, John" in bibtex
        assert "2022" in bibtex

    def test_journal_venue_produces_article_entry(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        bibtex = agent._build_bibtex(_LIT_CONTEXT)
        # paper002 has "Journal of Educational Data Mining" as venue
        assert "@article{paper002" in bibtex
        assert "journal" in bibtex.lower()

    def test_conference_venue_produces_inproceedings(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        bibtex = agent._build_bibtex(_LIT_CONTEXT)
        assert "@inproceedings{paper001" in bibtex

    def test_with_none_returns_placeholder_comment(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        bibtex = agent._build_bibtex(None)
        assert _S2_FAILURE_BIB_COMMENT.strip() in bibtex

    def test_with_empty_papers_returns_placeholder_comment(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        bibtex = agent._build_bibtex({"search_query": "test", "papers": [], "novelty_evidence": ""})
        assert _S2_FAILURE_BIB_COMMENT.strip() in bibtex

    def test_missing_venue_falls_back_to_default(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        lit_no_venue = {
            "papers": [
                {
                    "paperId": "no_venue_paper",
                    "title": "Some Study",
                    "authors": ["Author, A."],
                    "year": 2023,
                    "abstract": "",
                    # no "venue" key
                }
            ]
        }
        bibtex = agent._build_bibtex(lit_no_venue)
        assert "Educational Data Mining" in bibtex

    def test_empty_authors_list(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        lit = {
            "papers": [
                {
                    "paperId": "anon_paper",
                    "title": "Anonymous Work",
                    "authors": [],
                    "year": 2023,
                    "abstract": "",
                    "venue": "",
                }
            ]
        }
        bibtex = agent._build_bibtex(lit)
        assert "Unknown Author" in bibtex

    def test_paper_id_used_as_bibtex_key(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        bibtex = agent._build_bibtex(_LIT_CONTEXT)
        assert "{paper001," in bibtex or "@inproceedings{paper001" in bibtex


# ---------------------------------------------------------------------------
# Unit tests — _extract_latex()
# ---------------------------------------------------------------------------


class TestExtractLatex:
    def test_extracts_latex_fenced_block(self) -> None:
        text = f"Some text.\n```latex\n{_SAMPLE_TEX}\n```\nMore text."
        result = Writer._extract_latex(text)
        assert r"\documentclass[sigconf]{acmart}" in result
        assert r"\end{document}" in result

    def test_falls_back_to_documentclass_span(self) -> None:
        text = (
            "No fenced block here, but:\n"
            r"\documentclass[sigconf]{acmart}" + "\n"
            r"\begin{document}" + "\n"
            r"\end{document}"
        )
        result = Writer._extract_latex(text)
        assert r"\documentclass" in result
        assert r"\end{document}" in result

    def test_returns_stub_when_no_latex_found(self) -> None:
        result = Writer._extract_latex("No LaTeX content here at all.")
        assert r"\documentclass" in result  # stub has documentclass
        assert result == _MINIMAL_STUB_TEX

    def test_strips_surrounding_whitespace(self) -> None:
        text = f"```latex\n\n{_SAMPLE_TEX}\n\n```"
        result = Writer._extract_latex(text)
        assert not result.startswith("\n")
        assert not result.endswith("\n\n")

    def test_nested_code_blocks_handled_correctly(self) -> None:
        """The first ```latex block should be extracted, not a later one."""
        first_tex = r"\documentclass{article}" + "\n" + r"\end{document}"
        second_tex = r"\documentclass{beamer}" + "\n" + r"\end{document}"
        text = f"```latex\n{first_tex}\n```\nMore prose.\n```latex\n{second_tex}\n```"
        result = Writer._extract_latex(text)
        assert "article" in result
        # Should NOT pick up the second block
        assert "beamer" not in result


# ---------------------------------------------------------------------------
# Unit tests — _extract_bibtex()
# ---------------------------------------------------------------------------


class TestExtractBibtex:
    def test_extracts_bibtex_fenced_block(self) -> None:
        text = f"References:\n```bibtex\n{_SAMPLE_BIB}\n```\n"
        result = Writer._extract_bibtex(text)
        assert "@inproceedings{paper001" in result

    def test_returns_empty_string_when_no_bibtex_block(self) -> None:
        result = Writer._extract_bibtex("No BibTeX here.")
        assert result == ""

    def test_trailing_newline_added(self) -> None:
        text = f"```bibtex\n{_SAMPLE_BIB}\n```"
        result = Writer._extract_bibtex(text)
        assert result.endswith("\n")


# ---------------------------------------------------------------------------
# Unit tests — run() orchestration
# ---------------------------------------------------------------------------


class TestRunOrchestration:
    def _agent(self, tmp_path: Path) -> Writer:
        return _make_agent(tmp_path)

    def test_run_writes_paper_tex(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        agent.call_llm = MagicMock(return_value=_llm_response_with_both_blocks())

        agent.run()

        assert (tmp_path / "paper.tex").exists(), "paper.tex was not written"

    def test_run_writes_references_bib(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        agent.call_llm = MagicMock(return_value=_llm_response_with_both_blocks())

        agent.run()

        assert (tmp_path / "references.bib").exists(), "references.bib was not written"

    def test_run_returns_paper_text_as_string(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        agent.call_llm = MagicMock(return_value=_llm_response_with_both_blocks())

        result = agent.run()

        assert isinstance(result, str), "run() must return a string"
        assert r"\documentclass" in result

    def test_run_paper_tex_matches_return_value(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        agent.call_llm = MagicMock(return_value=_llm_response_with_both_blocks())

        returned_text = agent.run()
        on_disk = (tmp_path / "paper.tex").read_text(encoding="utf-8")
        assert returned_text == on_disk

    def test_run_uses_bibtex_from_llm_response(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        agent.call_llm = MagicMock(return_value=_llm_response_with_both_blocks())

        agent.run()

        bib_text = (tmp_path / "references.bib").read_text(encoding="utf-8")
        assert "paper001" in bib_text

    def test_run_falls_back_to_prebuild_bibtex_when_llm_omits_it(self, tmp_path: Path) -> None:
        """If LLM response has no ```bibtex block, pre-built BibTeX from S2 is used."""
        agent = self._agent(tmp_path)
        latex_only_response = f"```latex\n{_SAMPLE_TEX}\n```\n(no bibtex block)"
        agent.call_llm = MagicMock(return_value=latex_only_response)

        agent.run()

        bib_text = (tmp_path / "references.bib").read_text(encoding="utf-8")
        # The pre-built bibtex from _LIT_CONTEXT should be present
        assert "paper001" in bib_text

    def test_run_uses_ctx_inputs_when_no_kwargs(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        agent.call_llm = MagicMock(return_value=_llm_response_with_both_blocks())

        # No explicit kwargs — reads from ctx
        result = agent.run()
        assert r"\documentclass" in result

    def test_run_calls_call_llm_exactly_once(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        agent.call_llm = MagicMock(return_value=_llm_response_with_both_blocks())

        agent.run()
        assert agent.call_llm.call_count == 1

    def test_run_includes_figures_in_user_message(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        agent.call_llm = MagicMock(return_value=_llm_response_with_both_blocks())

        agent.run(results_object=_RESULTS_OBJECT)

        user_message = agent.call_llm.call_args[0][0]
        assert "shap_summary.png" in user_message

    def test_run_handles_stub_latex_on_llm_failure(self, tmp_path: Path) -> None:
        """If LLM returns no parseable LaTeX, the stub fallback is written."""
        agent = self._agent(tmp_path)
        agent.call_llm = MagicMock(return_value="Sorry, I cannot write that paper.")
        # Patch template loading to avoid filesystem dependency in this test
        agent._load_template = MagicMock(return_value=_MINIMAL_STUB_TEX)

        result = agent.run()

        assert r"\documentclass" in result
        on_disk = (tmp_path / "paper.tex").read_text(encoding="utf-8")
        assert on_disk == result

    def test_run_s2_failure_bib_fallback_when_no_papers(self, tmp_path: Path) -> None:
        """With no S2 papers and no bibtex block from LLM, uses placeholder comment."""
        agent = self._agent(tmp_path)
        agent.ctx.literature_context = {"papers": [], "search_query": "", "novelty_evidence": ""}
        agent.call_llm = MagicMock(return_value=f"```latex\n{_SAMPLE_TEX}\n```")

        agent.run()

        bib_text = (tmp_path / "references.bib").read_text(encoding="utf-8")
        assert "Semantic Scholar" in bib_text or "placeholder" in bib_text.lower()


# ---------------------------------------------------------------------------
# Unit tests — UNVERIFIED flag behaviour (via user message content check)
# ---------------------------------------------------------------------------


class TestUnverifiedFlag:
    """The UNVERIFIED flag is inserted by the LLM per the Writer system prompt.

    These tests verify that the user message clearly signals to the LLM when the
    review_report verdict is not PASS, so it knows to include the UNVERIFIED block.
    """

    def _agent(self, tmp_path: Path) -> Writer:
        return _make_agent(tmp_path)

    def test_non_pass_verdict_in_user_message(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        revise_review = {**_PASS_REVIEW, "overall_verdict": "REVISE"}
        agent.ctx.review_report = revise_review
        agent.call_llm = MagicMock(return_value=_llm_response_with_both_blocks())

        agent.run(review_report=revise_review)

        user_message = agent.call_llm.call_args[0][0]
        # The review_report JSON should appear in the prompt with REVISE verdict
        assert "REVISE" in user_message

    def test_pass_verdict_in_user_message(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        agent.call_llm = MagicMock(return_value=_llm_response_with_both_blocks())

        agent.run(review_report=_PASS_REVIEW)

        user_message = agent.call_llm.call_args[0][0]
        assert "PASS" in user_message


# ---------------------------------------------------------------------------
# Unit tests — template file
# ---------------------------------------------------------------------------


class TestTemplateFile:
    """Tests that verify the LaTeX template file structure."""

    _TEMPLATE_PATH = Path(__file__).parent.parent / "templates" / "paper_template.tex"

    def test_template_file_exists(self) -> None:
        """The LaTeX template file must exist at the expected path."""
        assert self._TEMPLATE_PATH.exists(), "templates/paper_template.tex not found"

    def test_template_has_required_structure(self) -> None:
        """Template contains all required ACM sigconf structural elements."""
        content = self._TEMPLATE_PATH.read_text(encoding="utf-8")
        assert r"\documentclass[sigconf]{acmart}" in content
        assert r"\begin{document}" in content
        assert r"\begin{abstract}" in content
        assert r"\maketitle" in content
        assert r"\begin{acks}" in content
        assert r"\bibliographystyle{ACM-Reference-Format}" in content
        assert r"\bibliography{references}" in content
        assert r"\end{document}" in content
        # Fixed authors
        assert "EDM-ARS" in content
        assert "Claude AI" in content
        assert "Chenguang Pan" in content
        assert "cp3280@tc.columbia.edu" in content
        # Abstract must be inside \begin{document} and before \maketitle
        doc_start = content.index(r"\begin{document}")
        abstract_start = content.index(r"\begin{abstract}")
        maketitle_pos = content.index(r"\maketitle")
        assert abstract_start > doc_start, r"\begin{abstract} must be inside \begin{document}"
        assert abstract_start < maketitle_pos, r"\begin{abstract} must precede \maketitle"

    def test_template_has_all_placeholders(self) -> None:
        """Template contains all expected placeholder markers."""
        content = self._TEMPLATE_PATH.read_text(encoding="utf-8")
        expected_placeholders = [
            "%%PLACEHOLDER:TITLE%%",
            "%%PLACEHOLDER:ABSTRACT%%",
            "%%PLACEHOLDER:KEYWORDS%%",
            "%%PLACEHOLDER:INTRODUCTION%%",
            "%%PLACEHOLDER:RELATED_WORK%%",
            "%%PLACEHOLDER:METHODS_DATA%%",
            "%%PLACEHOLDER:METHODS_VARIABLES%%",
            "%%PLACEHOLDER:METHODS_MISSING_DATA%%",
            "%%PLACEHOLDER:METHODS_MODELS%%",
            "%%PLACEHOLDER:METHODS_TUNING%%",
            "%%PLACEHOLDER:METHODS_EVALUATION%%",
            "%%PLACEHOLDER:METHODS_INTERPRETABILITY%%",
            "%%PLACEHOLDER:RESULTS_MODEL_COMPARISON%%",
            "%%PLACEHOLDER:RESULTS_FEATURE_IMPORTANCE%%",
            "%%PLACEHOLDER:RESULTS_SUBGROUP%%",
            "%%PLACEHOLDER:DISCUSSION_SUMMARY%%",
            "%%PLACEHOLDER:DISCUSSION_IMPLICATIONS%%",
            "%%PLACEHOLDER:DISCUSSION_LIMITATIONS%%",
            "%%PLACEHOLDER:DISCUSSION_FUTURE%%",
            "%%PLACEHOLDER:APPENDIX%%",
        ]
        for ph in expected_placeholders:
            assert ph in content, f"Missing placeholder: {ph}"


# ---------------------------------------------------------------------------
# Unit tests — _validate_template_structure()
# ---------------------------------------------------------------------------


class TestValidateTemplateStructure:
    def test_valid_latex_returns_no_warnings(self) -> None:
        valid = (
            r"\documentclass[sigconf]{acmart}" + "\n"
            r"\begin{document}" + "\n"
            r"\begin{abstract}text\end{abstract}" + "\n"
            r"\maketitle" + "\n"
            r"\begin{acks}thanks\end{acks}" + "\n"
            r"\bibliographystyle{ACM-Reference-Format}" + "\n"
            "Chenguang Pan\n"
            r"\end{document}"
        )
        warnings = Writer._validate_template_structure(valid)
        assert warnings == []

    def test_abstract_after_maketitle_flagged(self) -> None:
        bad = (
            r"\documentclass[sigconf]{acmart}" + "\n"
            r"\begin{document}" + "\n"
            r"\maketitle" + "\n"
            r"\begin{abstract}text\end{abstract}" + "\n"
            r"\begin{acks}thanks\end{acks}" + "\n"
            r"\bibliographystyle{ACM-Reference-Format}" + "\n"
            "Chenguang Pan\n"
        )
        warnings = Writer._validate_template_structure(bad)
        assert any("abstract" in w.lower() and "maketitle" in w.lower() for w in warnings)

    def test_unfilled_placeholders_flagged(self) -> None:
        tex_with_placeholder = (
            r"\documentclass[sigconf]{acmart}" + "\n"
            r"\begin{document}" + "\n"
            r"\begin{abstract}%%PLACEHOLDER:ABSTRACT%%\end{abstract}" + "\n"
            r"\maketitle" + "\n"
            r"\begin{acks}thanks\end{acks}" + "\n"
            r"\bibliographystyle{ACM-Reference-Format}" + "\n"
            "Chenguang Pan\n"
        )
        warnings = Writer._validate_template_structure(tex_with_placeholder)
        assert any("PLACEHOLDER" in w for w in warnings)


# ---------------------------------------------------------------------------
# Unit tests — LaTeX quality gate integration
# ---------------------------------------------------------------------------


class TestLatexQualityGate:
    """Verify that quality warnings are logged when crutch phrases are present."""

    def _agent(self, tmp_path: Path) -> Writer:
        return _make_agent(tmp_path)

    def _tex_with_crutch(self, phrase: str) -> str:
        """Build a minimal valid LaTeX body that contains a crutch phrase."""
        return (
            r"\documentclass[sigconf]{acmart}" + "\n"
            r"\begin{document}" + "\n"
            r"\begin{abstract}Abstract text here.\end{abstract}" + "\n"
            r"\maketitle" + "\n"
            rf"\section{{Results}}{phrase} \cite{{paper001}}" + "\n"
            r"\begin{acks}Thanks.\end{acks}" + "\n"
            r"\bibliographystyle{ACM-Reference-Format}" + "\n"
            "Chenguang Pan\n"
            r"\end{document}"
        )

    def test_todo_crutch_phrase_logged(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        crutch_tex = self._tex_with_crutch("TODO: add more results here.")
        agent.call_llm = MagicMock(return_value=f"```latex\n{crutch_tex}\n```")

        agent.run()

        warning_messages = [
            entry["message"] for entry in agent.ctx.log if "LaTeX quality warning" in entry.get("message", "")
        ]
        assert warning_messages, "Expected at least one LaTeX quality warning in ctx.log"
        assert any("lq_03" in m for m in warning_messages)

    def test_not_shown_crutch_phrase_logged(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        crutch_tex = self._tex_with_crutch(r"Figure \ref{fig:shap} (not shown) omits details.")
        agent.call_llm = MagicMock(return_value=f"```latex\n{crutch_tex}\n```")

        agent.run()

        warning_messages = [
            entry["message"] for entry in agent.ctx.log if "LaTeX quality warning" in entry.get("message", "")
        ]
        assert warning_messages
        assert any("lq_01" in m for m in warning_messages)

    def test_clean_latex_no_quality_warnings(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path)
        # _llm_response_with_both_blocks() produces clean LaTeX with no crutch phrases
        agent.call_llm = MagicMock(return_value=_llm_response_with_both_blocks())

        agent.run()

        quality_warnings = [
            entry for entry in agent.ctx.log if "LaTeX quality warning" in entry.get("message", "")
        ]
        assert not quality_warnings, f"Unexpected quality warnings: {quality_warnings}"


# ---------------------------------------------------------------------------
# Paper ID sanitization tests
# ---------------------------------------------------------------------------


class TestSanitizePaperIds:
    """Tests for Writer._sanitize_paper_ids (arXiv colon → underscore)."""

    def test_replaces_colons_in_paper_ids(self) -> None:
        lit = {
            "papers": [
                {"paperId": "arxiv:2401.12345", "title": "Test"},
                {"paperId": "abc123", "title": "Test2"},
            ]
        }
        result = Writer._sanitize_paper_ids(lit)
        assert result["papers"][0]["paperId"] == "arxiv_2401.12345"
        assert result["papers"][1]["paperId"] == "abc123"

    def test_returns_none_for_none_input(self) -> None:
        assert Writer._sanitize_paper_ids(None) is None

    def test_returns_unchanged_when_no_papers(self) -> None:
        lit = {"papers": []}
        result = Writer._sanitize_paper_ids(lit)
        assert result["papers"] == []

    def test_bibtex_uses_sanitized_arxiv_id(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        lit = {
            "papers": [
                {"paperId": "arxiv:2401.12345", "title": "Test Paper", "year": 2024,
                 "authors": ["Smith, J."], "venue": ""},
            ]
        }
        sanitized = Writer._sanitize_paper_ids(lit)
        bib = agent._build_bibtex(sanitized)
        assert "arxiv_2401.12345" in bib
        assert "@misc" in bib
        assert "arXiv preprint" in bib


# ---------------------------------------------------------------------------
# Integration test — full Writer run with real LLM
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_writer_full_run(tmp_path: Path) -> None:
    """Full integration: real LLM (claude-sonnet-4-6) writes a complete paper.

    Requires ANTHROPIC_API_KEY in environment.
    Run with:  pytest tests/test_writer.py -v --run-integration
    """
    config = load_config(CONFIG_PATH)
    ctx = PipelineContext(
        dataset_name="hsls09_public",
        raw_data_path="data/raw/nonexistent.csv",
        output_dir=str(tmp_path),
        max_revision_cycles=2,
    )

    agent = Writer(ctx, "writer", config)
    paper_text = agent.run(
        research_spec=_RESEARCH_SPEC,
        literature_context=_LIT_CONTEXT,
        data_report=_DATA_REPORT,
        results_object=_RESULTS_OBJECT,
        review_report=_PASS_REVIEW,
    )

    # --- outputs written to disk ---
    assert (tmp_path / "paper.tex").exists(), "paper.tex not written"
    assert (tmp_path / "references.bib").exists(), "references.bib not written"

    # --- paper content ---
    assert isinstance(paper_text, str), "run() must return a string"
    assert r"\documentclass" in paper_text, "paper.tex must contain \\documentclass"
    assert r"\end{document}" in paper_text, "paper.tex must contain \\end{document}"
    assert r"\begin{abstract}" in paper_text, "Abstract section required"
    assert r"\section" in paper_text, "At least one section required"

    # --- automated-generation disclosure sentence (SPEC §4.5) ---
    assert "EDM-ARS" in paper_text, "Automated generation disclosure sentence required"

    # --- references.bib ---
    bib_text = (tmp_path / "references.bib").read_text(encoding="utf-8")
    assert bib_text, "references.bib must not be empty"
