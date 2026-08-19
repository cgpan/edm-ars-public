"""Arc P / P1+P2 — manuscript linter + venue norms tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.manuscript_linter import (
    LintReport,
    cited_keys,
    lint_manuscript,
    load_venue_norms,
)

ROOT = Path(__file__).resolve().parent.parent

# NOTE (2026-07-11): this fixture gained an abstract when the linter
# learned to check front-matter completeness. It previously modelled a
# "clean" manuscript that had no abstract at all -- exactly the defect
# that got a real journal run rejected by the reviewer (F-P5-EMPTY-ABSTRACT).
CLEAN_TEX = r"""
\documentclass{article}
\title{A Study of Something}
\begin{abstract}
This manuscript reports a study with a genuine abstract of sufficient
length to satisfy both the linter and a reviewer sanity check.
\end{abstract}
\begin{document}
\section{Introduction}
Prior work \cite{smith2020, jones2021} and \parencite{lee2022} agree.
Table~\ref{tab:results} shows it.
\begin{table}\caption{R}\label{tab:results}\end{table}
\bibliography{references}
\end{document}
"""

CLEAN_BIB = """
@article{smith2020, title={A}, year={2020}}
@article{jones2021, title={B}, year={2021}}
@inproceedings{lee2022, title={C}, year={2022}}
"""


def _write_run(tmp_path: Path, tex: str = CLEAN_TEX, bib: str = CLEAN_BIB,
               log: str = "", blg: str | None = None) -> Path:
    (tmp_path / "paper.tex").write_text(tex, encoding="utf-8")
    (tmp_path / "references.bib").write_text(bib, encoding="utf-8")
    if log:
        (tmp_path / "paper.log").write_text(log, encoding="utf-8")
    if blg is not None:
        (tmp_path / "paper.blg").write_text(blg, encoding="utf-8")
    return tmp_path


class TestCitationExtraction:
    def test_all_citation_command_variants(self) -> None:
        tex = (r"\cite{a} \citep{b} \citet[p.~3]{c} \parencite{d,e} "
               r"\textcite{f} \autocite{g}")
        assert cited_keys(tex) == {"a", "b", "c", "d", "e", "f", "g"}


class TestLinterDefects:
    def test_clean_manuscript_is_format_clean(self, tmp_path: Path) -> None:
        report = lint_manuscript(_write_run(tmp_path))
        assert report.format_clean, [d.__dict__ for d in report.defects]
        assert report.metrics["n_citations_distinct"] == 3
        assert (tmp_path / "manuscript_lint.json").exists()

    def test_placeholder_citations_are_error(self, tmp_path: Path) -> None:
        tex = CLEAN_TEX.replace("smith2020", "placeholder_1")
        report = lint_manuscript(_write_run(tmp_path, tex=tex))
        assert any(d.code == "placeholder-citations" for d in report.errors)

    def test_cited_key_missing_from_bib(self, tmp_path: Path) -> None:
        bib = CLEAN_BIB.replace("@article{jones2021, title={B}, year={2021}}", "")
        report = lint_manuscript(_write_run(tmp_path, bib=bib))
        assert any(d.code == "cited-key-missing-from-bib" for d in report.errors)

    def test_dangling_crossref(self, tmp_path: Path) -> None:
        tex = CLEAN_TEX.replace(r"\label{tab:results}", "")
        report = lint_manuscript(_write_run(tmp_path, tex=tex))
        assert any(d.code == "dangling-crossref" for d in report.errors)

    def test_unreferenced_float_is_warn(self, tmp_path: Path) -> None:
        tex = CLEAN_TEX.replace(r"Table~\ref{tab:results} shows it.", "")
        report = lint_manuscript(_write_run(tmp_path, tex=tex))
        assert any(d.code == "unreferenced-float" for d in report.defects)
        assert report.format_clean  # warn, not error

    def test_undefined_citation_in_log(self, tmp_path: Path) -> None:
        log = ("LaTeX Warning: Citation `ghost2019' on page 2 undefined "
               "on input line 40.\nThere were undefined references.\n")
        report = lint_manuscript(_write_run(tmp_path, log=log))
        assert any(d.code == "undefined-citation" for d in report.errors)

    def test_overfull_boxes_warn_when_heavy(self, tmp_path: Path) -> None:
        log = "Overfull \\hbox (45.3pt too wide) in paragraph\n"
        report = lint_manuscript(_write_run(tmp_path, log=log))
        assert any(d.code == "overfull-hboxes" for d in report.defects)
        assert report.metrics["worst_overfull_pt"] == pytest.approx(45.3)

    def test_biber_error_is_error(self, tmp_path: Path) -> None:
        blg = "[123] Utils.pm:123> ERROR - BibTeX subsystem: fatal thing\n"
        report = lint_manuscript(_write_run(tmp_path, blg=blg))
        assert any(d.code == "biber-error" for d in report.errors)

    def test_missing_tex_is_error_not_crash(self, tmp_path: Path) -> None:
        report = lint_manuscript(tmp_path)
        assert isinstance(report, LintReport)
        assert any(d.code == "missing-tex" for d in report.errors)


class TestVenueNorms:
    def test_norms_file_has_all_three_venues(self) -> None:
        norms = load_venue_norms()
        for venue in ("EDM", "JEDM", "JLA"):
            assert venue in norms, f"{venue} missing from venue_norms.yaml"
            assert norms[venue]["refs"]["p25"] > 0
            assert norms[venue]["n_anchors"] >= 9

    def test_journal_norms_reflect_citation_depth(self) -> None:
        # The Arc-P motivation: journals demand far more citations than
        # our papers currently carry (~10-15).
        norms = load_venue_norms()
        assert norms["JEDM"]["refs"]["p25"] >= 40
        assert norms["JLA"]["refs"]["p25"] >= 40

    def test_low_citation_count_flags_against_jedm(self, tmp_path: Path) -> None:
        report = lint_manuscript(_write_run(tmp_path), venue="JEDM")
        codes = [d.code for d in report.defects]
        assert "citations-below-venue-norm" in codes
        assert report.format_clean  # advisory warn, not error

    def test_unknown_venue_applies_no_norms(self, tmp_path: Path) -> None:
        report = lint_manuscript(_write_run(tmp_path), venue="NOPE")
        assert report.metrics.get("venue_norms_applied") is False


class TestLiveRunArtifact:
    def test_lints_a_real_past_run(self) -> None:
        """Smoke the linter against a real pipeline output dir if one is
        on disk (no fixture drift — the artifact contract is the test)."""
        candidates = sorted(ROOT.glob("runs/*/output*/paper.tex"))
        if not candidates:
            pytest.skip("no past run artifacts on this machine")
        run_dir = candidates[-1].parent
        report = lint_manuscript(run_dir, venue="EDM", write_json=False)
        assert isinstance(report.metrics.get("n_citations_distinct"), int)
        assert report.metrics["n_citations_distinct"] > 0
