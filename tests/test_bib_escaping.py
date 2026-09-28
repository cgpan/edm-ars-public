r"""references.bib fields have LaTeX's specials escaped, exactly once.

Observed on the owner's Mac (round 3, 2026-09-27): references.bib held
three bare ampersands, most from the new OpenAlex records --
"Arthritis Research & Therapy", "... A Journal of Crime Conflict & World
Order", "... using Data Mining & Machine Learning Approaches". The
reference list printed "Data Mining  Machine Learning" with the & gone,
and INV_BIB_AMPERSAND flagged the file.

The contract these tests pin: build_bib_entry escapes & % # _ $ in the
title, author and venue fields; decodes HTML entities first; never
escapes twice; keeps $...$ mathematics in arXiv-style titles; leaves the
DOI and URL raw.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src.citations import build_bib_entry, build_bibtex, latex_escape_field
from src.invariants import RunArtifacts, check_bibliography_ampersands

#: The three records the round-3 bib broke on, as OpenAlex supplied them
#: (titles and venues paraphrased where they were not in the report).
ROUND3_PAPERS = [
    {
        "paperId": "openalex_W100",
        "title": "Machine learning in rheumatology",
        "authors": ["A. Author"],
        "year": 2021,
        "venue": "Arthritis Research & Therapy",
        "publicationTypes": ["JournalArticle"],
        "doi": "10.1186/s13075-021-00000-0",
    },
    {
        "paperId": "openalex_W200",
        "title": "Predicting Indian Master's and PhD Enrollment using Data "
                 "Mining & Machine Learning Approaches",
        "authors": ["Bhimasen Moharana", "Vinay Kumar Singh"],
        "year": 2024,
        "venue": "2024 4th International Conference on Technological "
                 "Advancements in Computational Sciences (ICTACS)",
        "publicationTypes": ["Conference"],
        "doi": "10.1109/ictacs62700.2024.10841260",
    },
    {
        "paperId": "openalex_W300",
        "title": "School pathways",
        "authors": ["C. Author"],
        "year": 2020,
        "venue": "Social Justice A Journal of Crime Conflict & World Order",
        "publicationTypes": ["JournalArticle"],
    },
]


@pytest.mark.parametrize(
    "raw, escaped",
    [
        ("Arthritis Research & Therapy", "Arthritis Research \\& Therapy"),
        ("Computers &amp; Education", "Computers \\& Education"),
        ("Science &#38; Education", "Science \\& Education"),
        ("100% of grade_9 students ranked #1", "100\\% of grade\\_9 students ranked \\#1"),
        ("A $5 voucher", "A \\$5 voucher"),
        ("Costs from $15 to $20", "Costs from \\$15 to \\$20"),
        ("$k$-means clustering of $\\ell_1$ paths", "$k$-means clustering of $\\ell_1$ paths"),
        ("Already \\& escaped \\_ here", "Already \\& escaped \\_ here"),
        ("Plain title", "Plain title"),
    ],
)
def test_field_text_is_escaped_once(raw: str, escaped: str) -> None:
    assert latex_escape_field(raw) == escaped
    assert latex_escape_field(escaped) == escaped


def test_the_round3_entries_carry_no_bare_ampersand() -> None:
    bib = build_bibtex(ROUND3_PAPERS)

    assert "journal = {Arthritis Research \\& Therapy}" in bib
    assert "Data Mining \\& Machine Learning Approaches" in bib
    assert "Crime Conflict \\& World Order" in bib
    assert "& " not in bib.replace("\\&", "")


def test_the_final_check_passes_the_rebuilt_bib(tmp_path: Path) -> None:
    (tmp_path / "references.bib").write_text(build_bibtex(ROUND3_PAPERS), encoding="utf-8")
    assert check_bibliography_ampersands(RunArtifacts(str(tmp_path))) == []


def test_authors_are_escaped_and_the_doi_and_url_are_not() -> None:
    entry = build_bib_entry({
        "paperId": "p1",
        "title": "T",
        "authors": ["Research & Development Group", "Jane Doe"],
        "year": 2020,
        "venue": "Journal of X",
        "doi": "10.1000/abc_def%20#x",
    })
    assert "author    = {Research \\& Development Group and Jane Doe}" in entry
    assert "doi       = {10.1000/abc_def%20#x}" in entry

    entry = build_bib_entry({
        "paperId": "p2", "title": "T", "authors": ["A"], "year": 2020,
        "venue": "Journal of X", "url": "https://example.org/a_b?x=1&y=2",
    })
    assert "url       = {https://example.org/a_b?x=1&y=2}" in entry


@pytest.mark.requires_tools("pdflatex", "bibtex")
def test_the_reference_list_compiles_with_the_ampersands(tmp_path: Path) -> None:
    (tmp_path / "references.bib").write_text(build_bibtex(ROUND3_PAPERS), encoding="utf-8")
    (tmp_path / "p.tex").write_text(
        "\\documentclass{article}\n\\begin{document}\nText.\n\\nocite{*}\n"
        "\\bibliographystyle{plain}\n\\bibliography{references}\n\\end{document}\n",
        encoding="utf-8",
    )
    for argv in (["pdflatex", "-interaction=nonstopmode", "p.tex"], ["bibtex", "p"],
                 ["pdflatex", "-interaction=nonstopmode", "p.tex"]):
        subprocess.run(argv, cwd=tmp_path, capture_output=True, timeout=300)
    log = (tmp_path / "p.log").read_text(encoding="latin-1", errors="replace")
    assert [ln for ln in log.splitlines() if ln.startswith("!")] == []
    bbl = (tmp_path / "p.bbl").read_text(encoding="latin-1", errors="replace")
    # plain.bst lowercases titles.
    assert "data mining \\& machine learning" in " ".join(bbl.split()).lower()
