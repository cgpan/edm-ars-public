r"""The running head is a short title, not the whole title.

Observed on the owner's Mac (round 3, 2026-09-27): the 154-character title
"Do Ninth-Grade Non-Cognitive Factors Improve Prediction of College
Enrollment Beyond Achievement and SES? A School-Aware Machine Learning
Analysis of HSLS:09" ran across the whole header of every odd page and
printed over "Anonymous Conference". The conference template gave acmart
no short title, so acmart used the full one; the SHORTTITLE value the
reassembly computed went only to the journal template.

The contract these tests pin: the conference template sets \shorttitle
from a SHORTTITLE slot; reassembly fills it with the Writer's own short
title when that fits, else with the title's first clause, else with the
title cut at a word boundary (60 characters, 50 for APA 7); \title keeps
the full title, so the linter and the review gate still read it.
"""
from __future__ import annotations

import re
import subprocess
import types
from pathlib import Path

import pytest

from src.agents.writer import SHORT_TITLE_LIMIT, Writer, short_title
from src.manuscript_linter import _braced_arg

ROOT = Path(__file__).resolve().parents[1]
ROUND3_TITLE = (
    "Do Ninth-Grade Non-Cognitive Factors Improve Prediction of College "
    "Enrollment Beyond Achievement and SES? A School-Aware Machine Learning "
    "Analysis of HSLS:09"
)


@pytest.mark.parametrize(
    "title, limit, expected",
    [
        (ROUND3_TITLE, 60, "Do Ninth-Grade Non-Cognitive Factors Improve Prediction..."),
        ("Predicting Dropout: A School-Aware Analysis of HSLS:09 With Many More "
         "Words Than Fit", 60, "Predicting Dropout"),
        # "HSLS:09" is not a clause break.
        ("Using HSLS:09 to Predict College Enrollment From Ninth-Grade "
         "Non-Cognitive Factors", 60, "Using HSLS:09 to Predict College Enrollment..."),
        ("Predicting \\textbf{STEM Achievement} in Ninth Grade From Motivation "
         "and Other Factors", 60, "Predicting STEM Achievement in Ninth Grade..."),
        ("A Long Title About Cognitive Diagnosis in Tutoring Systems", 50,
         "A Long Title About Cognitive Diagnosis..."),
        ("Short Enough Title", 60, "Short Enough Title"),
        ("Two\\\\Lines of Title", 60, "Two Lines of Title"),
    ],
)
def test_short_title(title: str, limit: int, expected: str) -> None:
    got = short_title(title, limit)
    assert got == expected
    assert len(got) <= limit


def test_a_truncated_short_title_never_ends_on_a_little_word() -> None:
    got = short_title("Grades and Scores and Grades and Scores and Grades and "
                      "Scores and More", 40)
    assert not re.search(r"\b(and|of|the)\.\.\.$", got)


def _writer() -> Writer:
    w = object.__new__(Writer)
    w.agent_name = "Writer"
    w.ctx = types.SimpleNamespace(output_dir="", log=[], errors=[], dataset_name="hsls09_public")
    w.config = {}
    return w


def _llm_paper(title: str, extra: str = "") -> str:
    return (
        "\\documentclass[sigconf]{acmart}\n\\begin{document}\n"
        f"\\title{{{title}}}\n{extra}\n"
        "\\begin{abstract}An abstract.\\end{abstract}\n\\keywords{a, b}\n"
        "\\maketitle\n\\section{Introduction}\nText.\n"
        "\\bibliographystyle{ACM-Reference-Format}\n\\bibliography{references}\n"
        "\\end{document}\n"
    )


def _template(name: str = "paper_template_v2.tex") -> str:
    return (ROOT / "templates" / name).read_text(encoding="utf-8")


def test_the_conference_template_carries_a_short_title_slot() -> None:
    for path in (ROOT / "templates" / "paper_template_v2.tex",
                 ROOT / "skills" / "writing" / "acm-acmart-sigconf-template"
                 / "paper_template_v2.tex"):
        text = path.read_text(encoding="utf-8")
        assert "\\title{%%PLACEHOLDER:TITLE%%}" in text, path
        assert "\\renewcommand{\\shorttitle}{%%PLACEHOLDER:SHORTTITLE%%}" in text, path


def test_reassembly_sets_the_short_title_and_keeps_the_full_one() -> None:
    out = _writer()._reassemble_from_template(_llm_paper(ROUND3_TITLE), _template())

    assert "%%PLACEHOLDER" not in out
    assert ("\\renewcommand{\\shorttitle}{Do Ninth-Grade Non-Cognitive Factors "
            "Improve Prediction...}") in out
    # The linter and the review gate read \title{...}: it is the full title.
    assert _braced_arg(out, r"\title") == ROUND3_TITLE


@pytest.mark.parametrize(
    "extra",
    ["\\renewcommand{\\shorttitle}{Non-Cognitive Factors and College Enrollment}",
     "\\shorttitle{Non-Cognitive Factors and College Enrollment}"],
)
def test_the_writers_own_short_title_is_used_when_it_fits(extra: str) -> None:
    out = _writer()._reassemble_from_template(_llm_paper(ROUND3_TITLE, extra), _template())
    assert "\\renewcommand{\\shorttitle}{Non-Cognitive Factors and College Enrollment}" in out


def test_a_writers_short_title_that_is_too_long_is_replaced() -> None:
    extra = "\\renewcommand{\\shorttitle}{" + ROUND3_TITLE[:100] + "}"
    out = _writer()._reassemble_from_template(_llm_paper(ROUND3_TITLE, extra), _template())
    match = re.search(r"\\renewcommand\{\\shorttitle\}\{([^{}]*)\}", out)
    assert match and len(match.group(1)) <= SHORT_TITLE_LIMIT


def test_the_journal_template_keeps_the_apa_limit() -> None:
    out = _writer()._reassemble_from_template(
        _llm_paper(ROUND3_TITLE), _template("paper_template_journal.tex")
    )
    match = re.search(r"\\shorttitle\{([^{}]*)\}", out)
    assert match and len(match.group(1)) <= 50


@pytest.mark.requires_tools("pdflatex")
def test_the_running_head_no_longer_runs_into_the_conference_line(tmp_path: Path) -> None:
    fitz = pytest.importorskip("fitz")
    body = "\\section{Introduction}\n" + "\n\n".join(
        ["Students were followed from ninth grade to early adulthood. " * 40] * 25
    )
    paper = _llm_paper(ROUND3_TITLE).replace("\\section{Introduction}\nText.\n", body + "\n")
    tex = _writer()._reassemble_from_template(paper, _template())
    (tmp_path / "paper.tex").write_text(tex, encoding="utf-8")
    subprocess.run(
        ["pdflatex", "-no-shell-escape", "-interaction=nonstopmode", "paper.tex"],
        cwd=tmp_path, capture_output=True, timeout=300,
    )
    if not (tmp_path / "paper.pdf").exists():
        pytest.skip("this TeX installation could not build the acmart template")
    doc = fitz.open(str(tmp_path / "paper.pdf"))
    assert len(doc) >= 3
    # Odd pages after the first carry the short title on the left and the
    # conference on the right, on one line.
    words = [w for w in doc[2].get_text("words") if w[3] < 75]
    line = " ".join(w[4] for w in words)
    assert "Do Ninth-Grade" in line and "HSLS:09" not in line
    title_right = max(w[2] for w in words if w[4] != "Anonymous" and w[0] < 300)
    conference_left = min(w[0] for w in words if w[4] == "Anonymous")
    assert title_right < conference_left
