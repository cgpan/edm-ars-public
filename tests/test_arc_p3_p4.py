"""Arc P3 (citation depth + bib reconciliation) and P4 (revise-resubmit).

P3 fixes F-E2A-SECTIONWISE-BIB-DRIFT (22 cited keys with no bib entry on a
real shipped manuscript), venue fabrication (29 invented
EDM-proceedings booktitles), and citation depth (4-26 references vs venue
norms of 15/47/54).

P4 turns the existing single revision cycle into one that is actually
driven by evidence — deterministic linter defects plus the weakest LSAR
dimension — and guards it against the failure modes an unguarded
whole-document rewrite has: truncation, no-ops, and edited results.

Everything here is offline: the P3 core is deterministic by design.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from src.citations import (
    S2_FAILURE_BIB_COMMENT,
    build_bib_entry,
    build_bibtex,
    expand_literature_pool,
    format_citation_key_block,
    reconcile_citations,
    venue_citation_target,
)
from src.manuscript_linter import cited_keys

ROOT = Path(__file__).resolve().parent.parent


def _paper(pid: str, year: int = 2023, title: str | None = None, **kw) -> dict:
    return {
        "paperId": pid,
        "title": title or f"Study {pid}",
        "authors": ["Author, A."],
        "year": year,
        "abstract": "",
        **kw,
    }


# ---------------------------------------------------------------------------
# P3 — BibTeX honesty
# ---------------------------------------------------------------------------


class TestBibHonesty:
    def test_absent_venue_is_declared_not_invented(self) -> None:
        entry = build_bib_entry(_paper("p1"))
        assert "@misc{p1" in entry
        assert "Venue metadata unavailable" in entry
        assert "Educational Data Mining" not in entry

    def test_journal_venue_becomes_article(self) -> None:
        entry = build_bib_entry(_paper("p2", venue="Journal of Learning Analytics"))
        assert "@article{p2" in entry
        assert "journal = {Journal of Learning Analytics}" in entry

    def test_conference_venue_becomes_inproceedings(self) -> None:
        entry = build_bib_entry(
            _paper("p3", venue="Proceedings of the 17th International Conference on EDM")
        )
        assert "@inproceedings{p3" in entry
        assert "17th International Conference" in entry

    def test_arxiv_id_sanitized_and_marked_preprint(self) -> None:
        entry = build_bib_entry(_paper("arxiv:2401.12345"))
        assert "@misc{arxiv_2401.12345" in entry
        assert "arXiv preprint" in entry

    def test_empty_papers_yields_honest_failure_comment(self) -> None:
        assert build_bibtex([]) == S2_FAILURE_BIB_COMMENT

    def test_non_ascii_authors_survive(self) -> None:
        # Backfilling ~50 references pulls in accented names; the bib is
        # written UTF-8 and must not be mangled or dropped.
        entry = build_bib_entry(
            {**_paper("p4"), "authors": ["Şahin, A.", "Müller, J.", "Cláudia, R."]}
        )
        assert "Şahin" in entry and "Müller" in entry


# ---------------------------------------------------------------------------
# P3 — reconciliation
# ---------------------------------------------------------------------------


class TestReconciliation:
    TEX = (
        r"\documentclass{article}\begin{document}"
        r"Prior work \cite{real1} and \parencite{ghost99} agree."
        r"\end{document}"
    )

    def test_invented_key_is_stripped(self) -> None:
        bib = build_bibtex([_paper("real1")])
        tex, out_bib, stats = reconcile_citations(self.TEX, bib, [_paper("real1")])
        assert "ghost99" not in tex
        assert "real1" in tex
        assert stats["stripped"] == 1
        assert stats["invented_keys"] == ["ghost99"]

    def test_pool_key_is_backfilled_not_stripped(self) -> None:
        papers = [_paper("real1"), _paper("real2")]
        bib = build_bibtex([_paper("real1")])  # bib missing real2
        tex = self.TEX.replace("ghost99", "real2")
        out_tex, out_bib, stats = reconcile_citations(tex, bib, papers)
        assert "real2" in out_tex, "a retrievable key must be kept, not stripped"
        assert "@misc{real2" in out_bib
        assert stats["backfilled"] == 1 and stats["stripped"] == 0

    def test_every_cited_key_ends_up_in_the_bib(self) -> None:
        """The F-E2A-SECTIONWISE-BIB-DRIFT contract, stated directly."""
        papers = [_paper(f"p{i}") for i in range(6)]
        tex = (
            r"\begin{document}"
            + " ".join(rf"\parencite{{p{i}}}" for i in range(6))
            + r"\cite{invented2020}\end{document}"
        )
        out_tex, out_bib, _ = reconcile_citations(tex, "", papers)
        from src.manuscript_linter import _BIB_ENTRY

        assert cited_keys(out_tex) <= set(_BIB_ENTRY.findall(out_bib))
        assert "invented2020" not in out_tex

    def test_multi_key_command_keeps_survivors(self) -> None:
        papers = [_paper("real1")]
        tex = r"\begin{document}Work \parencite{real1,ghost1}.\end{document}"
        out_tex, _, stats = reconcile_citations(tex, build_bibtex(papers), papers)
        assert "real1" in out_tex and "ghost1" not in out_tex
        assert r"\parencite{real1}" in out_tex
        assert stats["stripped"] == 1

    def test_placeholder_keys_stripped_when_real_papers_exist(self) -> None:
        papers = [_paper("real1")]
        tex = r"\begin{document}A \cite{placeholder_1} B \cite{real1}.\end{document}"
        out_tex, _, stats = reconcile_citations(tex, build_bibtex(papers), papers)
        assert "placeholder_1" not in out_tex
        assert "real1" in out_tex

    def test_no_papers_is_a_no_op(self) -> None:
        tex = r"\cite{placeholder_1}"
        out_tex, out_bib, stats = reconcile_citations(
            tex, S2_FAILURE_BIB_COMMENT, []
        )
        assert out_tex == tex
        assert out_bib == S2_FAILURE_BIB_COMMENT
        assert stats["skipped"] == "no retrieved papers"

    def test_never_strips_every_citation(self) -> None:
        """An empty bibliography is worse than a dangling key."""
        papers = [_paper("real1")]
        tex = r"\begin{document}All invented \cite{ghost1} \cite{ghost2}.\end{document}"
        out_tex, _, stats = reconcile_citations(tex, build_bibtex(papers), papers)
        assert out_tex == tex
        assert stats["skipped"] == "would remove all citations"

    def test_stripping_tidies_space_before_punctuation(self) -> None:
        papers = [_paper("real1")]
        tex = r"\begin{document}Claim \cite{real1} and more \cite{ghost1} .\end{document}"
        out_tex, _, _ = reconcile_citations(tex, build_bibtex(papers), papers)
        assert "  " not in out_tex.replace(r"\begin", "")
        assert " ." not in out_tex


class TestStrippedCitationStillCompiles:
    """A stripped \\cite must leave compilable LaTeX, not a wrecked sentence."""

    @pytest.mark.skipif(
        subprocess.run(
            ["pdflatex", "--version"], capture_output=True
        ).returncode != 0,
        reason="pdflatex not available",
    )
    def test_compiles_after_reconciliation(self, tmp_path: Path) -> None:
        papers = [_paper("real1", venue="Journal of Learning Analytics")]
        tex = (
            "\\documentclass{article}\n\\begin{document}\n"
            "Prior work \\cite{real1} and \\cite{ghost1} agree, as others "
            "note \\cite{ghost2} .\n"
            "\\bibliographystyle{plain}\n\\bibliography{references}\n"
            "\\end{document}\n"
        )
        out_tex, out_bib, _ = reconcile_citations(tex, build_bibtex(papers), papers)
        (tmp_path / "p.tex").write_text(out_tex, encoding="utf-8")
        (tmp_path / "references.bib").write_text(out_bib, encoding="utf-8")
        for _ in range(2):
            r = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "p.tex"],
                cwd=tmp_path, capture_output=True, timeout=120,
            )
        assert (tmp_path / "p.pdf").exists(), r.stdout[-800:]
        log = (tmp_path / "p.log").read_text(encoding="utf-8", errors="replace")
        assert "ghost1" not in log


# ---------------------------------------------------------------------------
# P3 — depth
# ---------------------------------------------------------------------------


class TestCitationDepth:
    def test_pool_tops_up_to_target(self) -> None:
        selected = [_paper(f"s{i}") for i in range(10)]
        pool = [_paper(f"pool{i}") for i in range(80)]
        out = expand_literature_pool(selected, pool, target=54)
        assert len(out) == 54
        assert [p["paperId"] for p in out[:10]] == [p["paperId"] for p in selected]

    def test_dedups_by_id_and_by_title(self) -> None:
        selected = [_paper("s1", title="Predicting dropout with machine learning")]
        pool = [
            _paper("s1"),  # same id
            _paper("x9", title="Predicting dropout with machine learning"),  # same title
            _paper("x10", title="A completely different investigation entirely"),
        ]
        out = expand_literature_pool(selected, pool, target=10)
        assert [p["paperId"] for p in out] == ["s1", "x10"]

    def test_requires_title_and_year_to_be_citable(self) -> None:
        selected: list[dict] = []
        pool = [
            {"paperId": "a", "title": "", "year": 2020},
            {"paperId": "b", "title": "Fine", "year": None},
            {"paperId": "c", "title": "Good one", "year": 2021},
        ]
        out = expand_literature_pool(selected, pool, target=5)
        assert [p["paperId"] for p in out] == ["c"]

    def test_selected_metadata_comes_from_the_pool_not_the_echo(self) -> None:
        """literature_context.json is the LLM's *echo* of the pool records.

        Measured on runs/arc_p_validation_20260711: the echo kept venue but
        dropped doi on 10/10 papers, while the pool held 97/100 with both.
        A dropped venue silently demotes a real reference to
        "@misc / Venue metadata unavailable", so the authoritative record
        must win — while the selected ORDER (LLM-judged relevance) stands.
        """
        echoed = {
            "paperId": "s1",
            "title": "Predicting dropout",
            "authors": ["Author, A."],
            "year": 2023,
            "abstract": "",
        }  # no venue, no doi — exactly the observed transcription loss
        authoritative = _paper(
            "s1",
            title="Predicting dropout",
            venue="Journal of Learning Analytics",
            doi="10.1234/jla.2023.1",
        )
        selected = [_paper("only-in-echo"), echoed]
        pool = [_paper("pool0"), authoritative]

        out = expand_literature_pool(selected, pool, target=3)

        assert [p["paperId"] for p in out] == ["only-in-echo", "s1", "pool0"]
        assert out[1] is authoritative
        assert out[1]["venue"] == "Journal of Learning Analytics"
        assert out[1]["doi"] == "10.1234/jla.2023.1"
        # A paper the pool does not contain is passed through untouched.
        assert out[0] is selected[0]
        # …and the substituted record now renders as a real reference.
        assert "@article{s1" in build_bib_entry(out[1])

    def test_never_shrinks_the_selection(self) -> None:
        selected = [_paper(f"s{i}") for i in range(20)]
        out = expand_literature_pool(selected, [], target=5)
        assert len(out) == 20

    @pytest.mark.parametrize("venue,floor", [("JEDM", 50), ("JLA", 45), ("EDM", 25)])
    def test_targets_come_from_mined_anchor_norms(self, venue: str, floor: int) -> None:
        target = venue_citation_target(venue)
        assert target is not None and target >= floor

    def test_unknown_venue_has_no_target(self) -> None:
        assert venue_citation_target("NOPE") is None
        assert venue_citation_target(None) is None

    def test_key_block_lists_keys_and_forbids_invention(self) -> None:
        block = format_citation_key_block([_paper("k1"), _paper("arxiv:9")], target=54)
        assert "k1" in block and "arxiv_9" in block
        assert "54" in block
        assert "Never invent" in block

    def test_key_block_empty_without_papers(self) -> None:
        assert format_citation_key_block([]) == ""


# ---------------------------------------------------------------------------
# P3 — Writer integration (offline)
# ---------------------------------------------------------------------------


class TestWriterProducesConsistentArtifacts:
    def _writer(self, tmp_path: Path):
        from src.config import load_config
        from src.context import PipelineContext
        from src.agents.writer import Writer

        config = load_config(str(ROOT / "config.yaml"))
        ctx = PipelineContext(
            dataset_name="hsls09_public",
            raw_data_path="x.csv",
            output_dir=str(tmp_path),
        )
        ctx.research_spec = {"research_question": "Q"}
        ctx.literature_context = {"papers": [_paper("real1"), _paper("real2")]}
        ctx.data_report = {}
        ctx.results_object = {}
        ctx.review_report = {"overall_verdict": "PASS"}
        with patch("anthropic.Anthropic"):
            return Writer(ctx, "writer", config)

    def test_model_invented_keys_never_reach_disk(self, tmp_path: Path) -> None:
        writer = self._writer(tmp_path)
        tex = (
            "\\documentclass[sigconf]{acmart}\n\\begin{document}\n"
            "\\title{T}\\maketitle\n"
            "Prior work \\cite{real1} and \\cite{hallucinated2021} agree.\n"
            "\\bibliographystyle{ACM-Reference-Format}\n"
            "\\bibliography{references}\n\\end{document}\n"
        )
        writer.call_llm = lambda *a, **k: f"```latex\n{tex}\n```"
        writer.run(outline=None)

        paper = (tmp_path / "paper.tex").read_text(encoding="utf-8")
        bib = (tmp_path / "references.bib").read_text(encoding="utf-8")
        from src.manuscript_linter import _BIB_ENTRY

        assert "hallucinated2021" not in paper
        assert cited_keys(paper) <= set(_BIB_ENTRY.findall(bib)), (
            "every cited key must have a bib entry (F-E2A-SECTIONWISE-BIB-DRIFT)"
        )

    def test_linter_reports_clean_citations_after_writer(self, tmp_path: Path) -> None:
        writer = self._writer(tmp_path)
        tex = (
            "\\documentclass[sigconf]{acmart}\n\\begin{document}\n"
            "A \\cite{real1} B \\cite{nope99}\n"
            "\\bibliography{references}\n\\end{document}\n"
        )
        writer.call_llm = lambda *a, **k: f"```latex\n{tex}\n```"
        writer.run(outline=None)

        from src.manuscript_linter import lint_manuscript

        report = lint_manuscript(tmp_path, write_json=False)
        codes = [d.code for d in report.errors]
        assert "cited-key-missing-from-bib" not in codes
        assert "placeholder-citations" not in codes

    def test_reconciliation_is_logged(self, tmp_path: Path) -> None:
        writer = self._writer(tmp_path)
        writer.call_llm = lambda *a, **k: (
            "```latex\n\\documentclass{article}\\begin{document}"
            "\\cite{real1}\\bibliography{references}\\end{document}\n```"
        )
        writer.run(outline=None)
        msgs = [e.get("message", "") for e in writer.ctx.log]
        assert any("Bib reconciliation:" in m for m in msgs)
        # must not masquerade as a LaTeX quality warning (pinned elsewhere)
        assert not any("LaTeX quality warning" in m for m in msgs)


# ---------------------------------------------------------------------------
# P4 — revision guards
# ---------------------------------------------------------------------------


def _gate(tmp_path: Path):
    from src.review_gate import ReviewGate

    cfg = {
        "llm_provider": "deepseek",
        "deepseek": {"models": {"revision_writer": "deepseek-v4-pro"}},
        "review_gate": {"venue": "JEDM", "revision_model": "deepseek-v4-pro"},
    }
    return ReviewGate(cfg, str(tmp_path), log_fn=lambda *_: None)


_DOC = (
    "\\documentclass{article}\\begin{document}\n"
    "Intro prose here.\n"
    "\\begin{table}\\caption{R}\\begin{tabular}{ll}AUC & 0.82\\\\\\end{tabular}\\end{table}\n"
    "\\includegraphics{fig1.png}\n"
    "Discussion prose.\n\\end{document}\n"
)


class TestRevisionGuards:
    def test_accepts_prose_only_revision(self, tmp_path: Path) -> None:
        gate = _gate(tmp_path)
        revised = _DOC.replace("Discussion prose.", "Discussion prose, expanded a lot.")
        ok, reason = gate._revision_is_safe(_DOC, revised)
        assert ok, reason

    def test_rejects_truncated_revision(self, tmp_path: Path) -> None:
        gate = _gate(tmp_path)
        ok, reason = gate._revision_is_safe(_DOC, _DOC[: len(_DOC) // 3])
        assert not ok and "truncat" in reason.lower()

    def test_rejects_edited_results_table(self, tmp_path: Path) -> None:
        gate = _gate(tmp_path)
        tampered = _DOC.replace("0.82", "0.91")
        ok, reason = gate._revision_is_safe(_DOC, tampered)
        assert not ok and "tables/figures" in reason

    def test_rejects_dropped_figure(self, tmp_path: Path) -> None:
        gate = _gate(tmp_path)
        dropped = _DOC.replace("\\includegraphics{fig1.png}\n", "")
        ok, reason = gate._revision_is_safe(_DOC, dropped)
        assert not ok

    def test_rejects_empty_revision(self, tmp_path: Path) -> None:
        gate = _gate(tmp_path)
        ok, _ = gate._revision_is_safe(_DOC, "")
        assert not ok

    def test_whitespace_reflow_is_not_tampering(self, tmp_path: Path) -> None:
        gate = _gate(tmp_path)
        reflowed = _DOC.replace(
            "\\begin{table}\\caption{R}", "\\begin{table}\n  \\caption{R}"
        )
        ok, reason = gate._revision_is_safe(_DOC, reflowed)
        assert ok, reason


class TestRevisionPromptCarriesEvidence:
    def _diagnosis(self, weakest: str = "Novelty") -> dict:
        return {
            "overall_score": 4.8,
            "dimension_scores": {weakest: 3, "Clarity of Communication": 7},
            "suggested_focus_areas": [
                {"dimension": weakest, "score": "3", "target_agent": "Writer"}
            ],
        }

    def _lint(self, tmp_path: Path):
        from src.manuscript_linter import lint_manuscript

        (tmp_path / "paper.tex").write_text(
            r"\begin{document}\cite{ghost}\ref{tab:none}\end{document}",
            encoding="utf-8",
        )
        (tmp_path / "references.bib").write_text("", encoding="utf-8")
        return lint_manuscript(tmp_path, venue="JEDM", write_json=False)

    def test_prompt_lists_linter_defects(self, tmp_path: Path) -> None:
        gate = _gate(tmp_path)
        prompt = gate._build_revision_prompt(
            paper_tex="\\begin{document}x\\end{document}",
            strengths=[], weaknesses=[], suggestions=[], questions=[],
            focus_dims=["Novelty"], diagnosis=self._diagnosis(),
            lint_report=self._lint(tmp_path),
        )
        assert "Deterministic format defects" in prompt
        assert "cited-key-missing-from-bib" in prompt or "dangling-crossref" in prompt
        assert "re-checked" in prompt

    def test_prompt_states_citation_target(self, tmp_path: Path) -> None:
        gate = _gate(tmp_path)
        prompt = gate._build_revision_prompt(
            paper_tex="x", strengths=[], weaknesses=[], suggestions=[],
            questions=[], focus_dims=["Novelty"], diagnosis=self._diagnosis(),
            lint_report=self._lint(tmp_path),
        )
        assert "Citation target" in prompt
        assert "Never invent a citation key" in prompt

    def test_prompt_targets_sections_for_weakest_dimension(self, tmp_path: Path) -> None:
        gate = _gate(tmp_path)
        prompt = gate._build_revision_prompt(
            paper_tex="x", strengths=[], weaknesses=[], suggestions=[],
            questions=[], focus_dims=["Novelty"], diagnosis=self._diagnosis(),
        )
        assert "Related Work" in prompt
        assert "Concentrate your edits" in prompt

    def test_analyst_dimensions_route_to_limitations_not_results(self) -> None:
        from src.review_gate import DIMENSION_AGENT_MAP, DIMENSION_SECTION_MAP

        for dim, agent in DIMENSION_AGENT_MAP.items():
            if agent == "Analyst":
                assert DIMENSION_SECTION_MAP.get(dim) == ("Limitations",), (
                    f"{dim} routes to the Analyst; a prose reviser must not "
                    "be pointed at Methods/Results"
                )

    def test_prompt_survives_missing_lint_report(self, tmp_path: Path) -> None:
        gate = _gate(tmp_path)
        prompt = gate._build_revision_prompt(
            paper_tex="x", strengths=[], weaknesses=[], suggestions=[],
            questions=[], focus_dims=[], diagnosis={"overall_score": 5},
            lint_report=None,
        )
        assert "Deterministic format defects" not in prompt
        assert "## Task" in prompt


class TestLintReportIsAvailableToReviser:
    def test_lint_returns_report_and_stashes_it(self, tmp_path: Path) -> None:
        gate = _gate(tmp_path)
        (tmp_path / "paper.tex").write_text(
            r"\begin{document}\cite{a}\end{document}", encoding="utf-8"
        )
        (tmp_path / "references.bib").write_text("", encoding="utf-8")
        report = gate._lint_manuscript(tmp_path, cycle=1)
        assert report is not None
        assert gate._last_lint is report
        assert (tmp_path / "lsar_review" / "cycle_1" / "manuscript_lint.json").exists()

    def test_last_lint_defaults_to_none(self, tmp_path: Path) -> None:
        assert _gate(tmp_path)._last_lint is None


# ---------------------------------------------------------------------------
# F-P5-EMPTY-ABSTRACT — front-matter completeness
# ---------------------------------------------------------------------------


class TestFrontMatterCompleteness:
    """The linter must catch a structurally incomplete manuscript.

    Live regression: the Arc P validation run shipped a journal paper with
    a literally empty ``\abstract{}`` (reassembly matched only the
    ``\begin{abstract}`` environment while apa7 uses ``\abstract{...}``).
    It compiled, the linter said format_clean=True, and LSAR then refused
    to review it at all -- the whole gate was lost to a defect a static
    check should have caught first.
    """

    def _run(self, tmp_path: Path, tex: str):
        from src.manuscript_linter import lint_manuscript

        (tmp_path / "paper.tex").write_text(tex, encoding="utf-8")
        (tmp_path / "references.bib").write_text("", encoding="utf-8")
        return lint_manuscript(tmp_path, write_json=False)

    def test_empty_apa_abstract_macro_is_an_error(self, tmp_path: Path) -> None:
        rep = self._run(tmp_path, r"\title{T}\abstract{}\begin{document}x\end{document}")
        assert not rep.format_clean
        assert any(d.code == "empty-abstract" for d in rep.errors)

    def test_missing_abstract_is_an_error(self, tmp_path: Path) -> None:
        rep = self._run(tmp_path, r"\title{T}\begin{document}x\end{document}")
        assert any(d.code == "missing-abstract" for d in rep.errors)

    def test_populated_apa_macro_passes(self, tmp_path: Path) -> None:
        rep = self._run(
            tmp_path,
            r"\title{T}\abstract{" + "word " * 40 + r"}\begin{document}x\end{document}",
        )
        assert not any(
            d.code in ("empty-abstract", "missing-abstract") for d in rep.defects
        )
        assert rep.metrics["abstract_chars"] > 50

    def test_nested_braces_do_not_truncate(self, tmp_path: Path) -> None:
        body = "We fit \emph{DINA} models with $x_{1}$ terms. " + "word " * 30
        rep = self._run(
            tmp_path,
            r"\title{T}\abstract{" + body + r"}\begin{document}x\end{document}",
        )
        assert rep.metrics["abstract_chars"] >= len(body) - 2

    def test_environment_form_still_works(self, tmp_path: Path) -> None:
        rep = self._run(
            tmp_path,
            r"\begin{abstract}" + "word " * 40 + r"\end{abstract}"
            r"\begin{document}x\end{document}",
        )
        assert not any(d.code.endswith("abstract") for d in rep.errors)

    def test_unfilled_placeholder_is_an_error(self, tmp_path: Path) -> None:
        rep = self._run(
            tmp_path,
            r"\title{%%PLACEHOLDER:TITLE%%}\abstract{" + "w " * 40
            + r"}\begin{document}x\end{document}",
        )
        assert any(d.code == "unfilled-placeholder" for d in rep.errors)

    def test_writer_extracts_apa_macro_abstract(self) -> None:
        """The producer side of the same defect."""
        from src.agents.writer import _extract_braced_arg

        got = _extract_braced_arg(
            r"\title{T}" + "\n" + r"\abstract{Nested \emph{x} and $y_{2}$ here.}",
            r"\abstract",
        )
        assert got == r"Nested \emph{x} and $y_{2}$ here."
