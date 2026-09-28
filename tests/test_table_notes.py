r"""Table notes are put inside a threeparttable before paper.tex is written.

Observed on the owner's Mac (round 3, 2026-09-27): paper.log recorded 18
LaTeX errors, among them "You can't use \prevdepth in restricted
horizontal mode", "Lonely \item--perhaps a missing list environment" and
"\begin{table} on input line 280 ended by \end{tablenotes}". The subgroup
table (tab:subgroup) never rendered, and the text read "Table ?? reports
AUC separately by sex". Compiling the candidate shapes reproduces that
error list exactly for a tablenotes block inside a \resizebox argument
with no threeparttable, the shape the table skill led to ("wrap the
entire threeparttable block inside \resizebox", minus the threeparttable).

The contract these tests pin:

* notes inside a box argument without a threeparttable of its own move
  to just after the box, and the table gets a threeparttable around it;
* a table with notes and no threeparttable gets one, caption and label
  kept where they were;
* a table already in a working shape is left byte for byte;
* the Writer applies it on every path before paper.tex is written;
* with pdflatex on PATH: the broken table fails to compile and loses its
  label, the repaired one compiles cleanly and the reference resolves.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.latex_quality import repair_table_notes

TAB = (
    "\\begin{tabular}{llrrrr}\n"
    "\\toprule\n"
    "Attribute & Group & AUC & CI Low & CI High & $n$ \\\\\n"
    "\\midrule\n"
    "Sex & Female & 0.803 & 0.781 & 0.824 & 1{,}860 \\\\\n"
    "Sex & Male & 0.790 & 0.766 & 0.812 & 1{,}702 \\\\\n"
    "\\bottomrule\n"
    "\\end{tabular}"
)
NOTES = (
    "\\begin{tablenotes}\n"
    "\\footnotesize\n"
    "\\item Cells with fewer than 30 students are not interpreted.\n"
    "\\end{tablenotes}"
)
HEAD = (
    "\\begin{table}[t]\n"
    "\\caption{AUC by subgroup on the held-out test set.}\n"
    "\\label{tab:subgroup}\n"
)
TAIL = "\n\\end{table}"

#: What round 3's Writer produced: the notes inside the \resizebox.
NOTES_IN_BOX = (
    HEAD + "\\resizebox{\\columnwidth}{!}{%\n" + TAB + "%\n" + NOTES + "\n}" + TAIL
)
#: Notes after the box, no threeparttable (compiles, but notes float free).
NOTES_AFTER_BOX = HEAD + "\\resizebox{\\columnwidth}{!}{%\n" + TAB + "%\n}\n" + NOTES + TAIL
#: Notes after a plain tabular, no threeparttable.
NOTES_AFTER_TABULAR = HEAD + TAB + "\n" + NOTES + TAIL
#: The documented shape.
CORRECT = (
    HEAD + "\\begin{threeparttable}\n\\resizebox{\\columnwidth}{!}{%\n" + TAB
    + "%\n}\n" + NOTES + "\n\\end{threeparttable}" + TAIL
)
#: A whole threeparttable inside the box: compiles, left alone.
TPT_IN_BOX = (
    HEAD + "\\resizebox{\\columnwidth}{!}{%\n\\begin{threeparttable}\n" + TAB + "\n"
    + NOTES + "\n\\end{threeparttable}%\n}" + TAIL
)
#: A threeparttable around a box that swallowed the notes.
NOTES_IN_BOX_IN_TPT = (
    HEAD + "\\begin{threeparttable}\n\\resizebox{\\columnwidth}{!}{%\n" + TAB
    + "%\n" + NOTES + "\n}\n\\end{threeparttable}" + TAIL
)

MODEL_TABLE = (
    "\\begin{table}[t]\n\\caption{Model comparison.}\n\\label{tab:models}\n"
    "\\resizebox{\\columnwidth}{!}{%\n" + TAB + "%\n}\n\\end{table}"
)


def _shape_ok(table: str) -> None:
    """tablenotes inside a threeparttable, outside every box; label kept."""
    assert table.count("\\begin{threeparttable}") == 1
    assert table.count("\\end{threeparttable}") == 1
    tpt = table[table.index("\\begin{threeparttable}"): table.index("\\end{threeparttable}")]
    assert "\\begin{tablenotes}" in tpt and "\\end{tablenotes}" in tpt
    box = table.find("\\resizebox")
    if box >= 0:
        # The box's argument closes before the notes begin.
        assert table.index("}\n\\begin{tablenotes}") > box
    assert "\\label{tab:subgroup}" in table
    assert "\\caption{AUC by subgroup" in table


@pytest.mark.parametrize(
    "broken", [NOTES_IN_BOX, NOTES_AFTER_BOX, NOTES_AFTER_TABULAR, NOTES_IN_BOX_IN_TPT],
    ids=["in-box", "after-box", "after-tabular", "in-box-in-threeparttable"],
)
def test_a_table_with_loose_notes_is_repaired(broken: str) -> None:
    fixed, n = repair_table_notes(broken)
    assert n == 1
    _shape_ok(fixed)
    # Every line of the table is still there.
    for line in (TAB + "\n" + NOTES).splitlines():
        assert line in fixed


def test_the_round3_shape_becomes_the_documented_shape() -> None:
    fixed, _ = repair_table_notes(NOTES_IN_BOX)
    assert fixed == CORRECT


@pytest.mark.parametrize("ok", [CORRECT, TPT_IN_BOX, MODEL_TABLE],
                         ids=["documented", "threeparttable-in-box", "no-notes"])
def test_a_table_in_a_working_shape_is_left_alone(ok: str) -> None:
    assert repair_table_notes(ok) == (ok, 0)


def test_only_the_broken_table_of_a_paper_changes_and_twice_is_once() -> None:
    body = (
        "Text before \\ref{tab:subgroup}.\n\n" + MODEL_TABLE + "\n\n"
        + NOTES_IN_BOX.replace("\\begin{table}", "\\begin{table*}").replace(
            "\\end{table}", "\\end{table*}")
        + "\n\n\\begin{figure}\\centering x\\end{figure}\n"
    )
    fixed, n = repair_table_notes(body)
    assert n == 1
    assert MODEL_TABLE in fixed
    assert "\\begin{figure}\\centering x\\end{figure}" in fixed
    assert repair_table_notes(fixed) == (fixed, 0)


def test_the_writer_repairs_the_table_before_paper_tex_is_written(tmp_path: Path) -> None:
    from tests.test_writer import _SAMPLE_BIB, _SAMPLE_TEX, _make_agent

    agent = _make_agent(tmp_path)
    # Inside the body: reassembly keeps what lies between \maketitle and
    # \bibliographystyle.
    tex = _SAMPLE_TEX.replace(
        "\\bibliographystyle", NOTES_IN_BOX + "\n\\bibliographystyle"
    )
    assert "\\begin{tablenotes}" in tex
    agent.call_llm = MagicMock(
        return_value=f"```latex\n{tex}\n```\n```bibtex\n{_SAMPLE_BIB}\n```\n"
    )

    agent.run()

    written = (tmp_path / "paper.tex").read_text(encoding="utf-8")
    assert CORRECT in written
    assert any("threeparttable" in str(e.get("message")) for e in agent.ctx.log)


def test_the_table_skill_teaches_the_shape_the_repair_keeps() -> None:
    """The skill said to wrap the whole threeparttable in \\resizebox; the
    model dropped the threeparttable and kept the box. Its examples with
    notes must be shapes the repair leaves alone."""
    import re

    text = (
        Path(__file__).resolve().parents[1]
        / "skills" / "writing" / "latex-table-discipline" / "SKILL.md"
    ).read_text(encoding="utf-8")
    assert "wrap the entire" not in text
    assert "NEVER put `tablenotes` inside the `\\resizebox` argument" in text
    # A "\n" in "\noindent" once became a real newline in this file.
    assert "\noindent" not in text
    assert "`\\noindent{\\small text}`" in text
    examples = [b for b in re.findall(r"```latex\n(.*?)```", text, re.DOTALL)
                if "tablenotes" in b]
    assert len(examples) == 2
    assert any("\\resizebox" in b for b in examples)
    for block in examples:
        assert repair_table_notes(block) == (block, 0)


# ---------------------------------------------------------------------------
# What LaTeX makes of it
# ---------------------------------------------------------------------------

_DOC = (
    "\\documentclass[sigconf]{acmart}\n"
    "\\setcopyright{none}\n\\settopmatter{printacmref=false}\n"
    "\\usepackage{booktabs}\n\\usepackage{graphicx}\n\\usepackage{threeparttable}\n"
    "\\begin{document}\n\\title{T}\n\\author{A}\n"
    "\\begin{abstract}\nx\n\\end{abstract}\n\\maketitle\n\\section{Results}\n"
    "Table~\\ref{tab:subgroup} reports AUC separately by sex.\n\n{}\n"
    "\\end{document}\n"
)


def _compile(tmp_path: Path, name: str, table: str) -> tuple[list[str], str]:
    (tmp_path / f"{name}.tex").write_text(_DOC.replace("{}", table), encoding="utf-8")
    for _ in range(2):
        subprocess.run(
            ["pdflatex", "-no-shell-escape", "-interaction=nonstopmode", f"{name}.tex"],
            cwd=tmp_path, capture_output=True, timeout=300,
        )
    log = (tmp_path / f"{name}.log").read_text(encoding="latin-1", errors="replace")
    return [ln for ln in log.splitlines() if ln.startswith("!")], log


@pytest.mark.requires_tools("pdflatex")
def test_the_round3_table_fails_to_compile_and_the_repaired_one_does_not(
    tmp_path: Path,
) -> None:
    errors, log = _compile(tmp_path, "broken", NOTES_IN_BOX)
    if any("File `acmart.cls' not found" in ln or ".sty' not found" in ln
           for ln in log.splitlines()):
        pytest.skip("this TeX installation lacks acmart or its packages")
    assert "! LaTeX Error: Lonely \\item--perhaps a missing list environment." in errors
    assert any("ended by \\end{tablenotes}" in e for e in errors)
    assert "Reference `tab:subgroup' on page" in log

    fixed, _ = repair_table_notes(NOTES_IN_BOX)
    errors, log = _compile(tmp_path, "fixed", fixed)
    assert errors == []
    assert "Reference `tab:subgroup' on page" not in log
    assert (tmp_path / "fixed.pdf").exists()
