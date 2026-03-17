"""Tests for src/latex_quality.py — crutch-phrase / placeholder quality gate."""
from __future__ import annotations

import pytest

from src.latex_quality import LatexQualityReport, check_latex_quality


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_CLEAN_BODY = r"""
\section{Introduction}
This study examines factors predicting STEM outcomes among high school students.
Using data from the HSLS:09 longitudinal survey, we apply a six-model machine
learning battery to predict X4RFDGMJSTEM.

\section{Results}
XGBoost achieved AUC = 0.72, 95\% CI [0.69, 0.75], outperforming all individual models.

\begin{table}[h]
\caption{Model comparison.}
\label{tab:models}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lrrrrr}
\toprule
Model & AUC & Acc. & Prec. & Recall & F1 \\
\midrule
XGBoost & 0.72 & 0.66 & 0.64 & 0.60 & 0.62 \\
\bottomrule
\end{tabular}%
}
\end{table}
"""


# ---------------------------------------------------------------------------
# Individual pattern tests
# ---------------------------------------------------------------------------


def test_not_shown_detected() -> None:
    latex = r"Figure \ref{fig:shap} (not shown) displays the SHAP beeswarm plot."
    report = check_latex_quality(latex)
    ids = {i.pattern_id for i in report.issues}
    assert "lq_01" in ids
    assert any(i.severity == "error" for i in report.issues if i.pattern_id == "lq_01")


def test_insert_figure_placeholder_detected() -> None:
    latex = r"[Insert figure here showing model comparison]"
    report = check_latex_quality(latex)
    ids = {i.pattern_id for i in report.issues}
    assert "lq_02" in ids


def test_insert_table_placeholder_detected() -> None:
    latex = r"[Insert table with results]"
    report = check_latex_quality(latex)
    ids = {i.pattern_id for i in report.issues}
    assert "lq_02" in ids


def test_todo_marker_detected() -> None:
    latex = r"TODO: add discussion of limitations."
    report = check_latex_quality(latex)
    ids = {i.pattern_id for i in report.issues}
    assert "lq_03" in ids
    assert any(i.severity == "error" for i in report.issues if i.pattern_id == "lq_03")


def test_fixme_marker_detected() -> None:
    latex = r"FIXME: check this value."
    report = check_latex_quality(latex)
    ids = {i.pattern_id for i in report.issues}
    assert "lq_04" in ids


def test_unfilled_template_placeholder_detected() -> None:
    latex = r"%%PLACEHOLDER:INTRODUCTION%%"
    report = check_latex_quality(latex)
    ids = {i.pattern_id for i in report.issues}
    assert "lq_09" in ids
    assert report.has_errors


def test_author_year_citation_placeholder_detected() -> None:
    latex = r"as shown by [Author, Year], this approach is effective."
    report = check_latex_quality(latex)
    ids = {i.pattern_id for i in report.issues}
    assert "lq_06" in ids


def test_citation_needed_placeholder_detected() -> None:
    latex = r"This is well established [Citation needed]."
    report = check_latex_quality(latex)
    ids = {i.pattern_id for i in report.issues}
    assert "lq_06" in ids


def test_needs_citation_placeholder_detected() -> None:
    latex = r"Prior work [NEEDS CITATION] has established this."
    report = check_latex_quality(latex)
    ids = {i.pattern_id for i in report.issues}
    assert "lq_11" in ids


def test_omitted_for_brevity_is_warning() -> None:
    latex = r"Full results are omitted for brevity."
    report = check_latex_quality(latex)
    matching = [i for i in report.issues if i.pattern_id == "lq_07"]
    assert matching
    assert all(i.severity == "warning" for i in matching)


def test_will_be_discussed_is_warning() -> None:
    latex = r"This finding will be discussed in detail."
    report = check_latex_quality(latex)
    matching = [i for i in report.issues if i.pattern_id == "lq_08"]
    assert matching


def test_clean_latex_no_issues() -> None:
    report = check_latex_quality(_CLEAN_BODY)
    assert not report.issues
    assert not report.has_errors
    assert not report.exceeds_ratio_threshold


# ---------------------------------------------------------------------------
# Aggregate / report tests
# ---------------------------------------------------------------------------


def test_has_errors_false_when_only_warnings() -> None:
    latex = r"Full results are omitted for brevity."
    report = check_latex_quality(latex)
    assert not report.has_errors  # lq_07 is a warning, not an error


def test_has_errors_true_when_error_present() -> None:
    report = check_latex_quality(r"TODO: fix this")
    assert report.has_errors


def test_template_ratio_calculation() -> None:
    # A large latex body with a single short TODO
    body = "x" * 10000 + " TODO "
    report = check_latex_quality(body)
    assert report.total_chars == len(body)
    assert report.matched_chars == len("TODO")  # exact match of the pattern
    assert report.template_ratio == pytest.approx(len("TODO") / len(body), rel=1e-3)


def test_template_ratio_exceeds_threshold() -> None:
    # Fill paper with many TODO markers so ratio > 0.5%
    body = "TODO " * 500  # 2500 chars of "TODO " — well above 0.5% threshold for itself
    report = check_latex_quality(body)
    assert report.exceeds_ratio_threshold


def test_template_ratio_not_exceeded_for_clean() -> None:
    report = check_latex_quality(_CLEAN_BODY)
    assert not report.exceeds_ratio_threshold


def test_to_warning_strings_format() -> None:
    report = check_latex_quality(r"TODO: fix")
    strings = report.to_warning_strings()
    assert strings
    assert any("lq_03" in s for s in strings)
    assert any("ERROR" in s for s in strings)
    assert any("line" in s.lower() for s in strings)


def test_issue_has_correct_line_number() -> None:
    latex = "line one\nline two\nTODO: fix"
    report = check_latex_quality(latex)
    todo_issues = [i for i in report.issues if i.pattern_id == "lq_03"]
    assert todo_issues
    assert todo_issues[0].line_number == 3


def test_case_insensitive_matching() -> None:
    # Patterns use IGNORECASE
    report = check_latex_quality(r"todo: fix this")
    ids = {i.pattern_id for i in report.issues}
    assert "lq_03" in ids


def test_empty_latex_no_errors() -> None:
    report = check_latex_quality("")
    assert not report.issues
    assert report.template_ratio == 0.0


# ---------------------------------------------------------------------------
# Structural checks (lq_13, lq_14)
# ---------------------------------------------------------------------------


def test_lq13_no_cite_with_bibliography() -> None:
    """lq_13: bibliography present but zero \\cite commands → error."""
    latex = (
        r"\documentclass{article}\begin{document}"
        r"\section{Intro}Some text."
        r"\bibliographystyle{ACM-Reference-Format}"
        r"\bibliography{references}"
        r"\end{document}"
    )
    report = check_latex_quality(latex)
    ids = {i.pattern_id for i in report.issues}
    assert "lq_13" in ids
    assert any(i.severity == "error" for i in report.issues if i.pattern_id == "lq_13")


def test_lq13_not_triggered_when_cite_present() -> None:
    """lq_13: should NOT fire when \\cite commands exist."""
    latex = (
        r"\documentclass{article}\begin{document}"
        r"\section{Intro}As shown by \cite{smith2024}."
        r"\bibliographystyle{ACM-Reference-Format}"
        r"\bibliography{references}"
        r"\end{document}"
    )
    report = check_latex_quality(latex)
    ids = {i.pattern_id for i in report.issues}
    assert "lq_13" not in ids


def test_lq13_not_triggered_without_bibliography() -> None:
    """lq_13: should NOT fire when there is no \\bibliography command."""
    latex = r"\documentclass{article}\begin{document}\section{Intro}Text.\end{document}"
    report = check_latex_quality(latex)
    ids = {i.pattern_id for i in report.issues}
    assert "lq_13" not in ids


def test_lq14_resizebox_on_narrow_table() -> None:
    """lq_14: resizebox wrapping a 3-column table → warning."""
    latex = (
        r"\resizebox{\columnwidth}{!}{"
        r"\begin{tabular}{lrr}"
        r"\toprule Model & AUC & F1 \\"
        r"\bottomrule"
        r"\end{tabular}}"
    )
    report = check_latex_quality(latex)
    ids = {i.pattern_id for i in report.issues}
    assert "lq_14" in ids
    assert any(i.severity == "warning" for i in report.issues if i.pattern_id == "lq_14")


def test_lq14_not_triggered_on_wide_table() -> None:
    """lq_14: resizebox on a 6-column table should NOT fire."""
    latex = (
        r"\resizebox{\columnwidth}{!}{"
        r"\begin{tabular}{lrrrrr}"
        r"\toprule Model & AUC & Acc & Prec & Rec & F1 \\"
        r"\bottomrule"
        r"\end{tabular}}"
    )
    report = check_latex_quality(latex)
    ids = {i.pattern_id for i in report.issues}
    assert "lq_14" not in ids


def test_lq15_missing_backslash_includegraphics() -> None:
    """lq_15: includegraphics without backslash → error."""
    latex = (
        r"\begin{figure}[h]"
        "\n"
        r"\centering"
        "\n"
        "includegraphics[width=\\linewidth]{calibration_curve.png}"
        "\n"
        r"\end{figure}"
    )
    report = check_latex_quality(latex)
    ids = {i.pattern_id for i in report.issues}
    assert "lq_15" in ids
    assert any(i.severity == "error" for i in report.issues if i.pattern_id == "lq_15")


def test_lq15_not_triggered_with_backslash() -> None:
    """lq_15: proper \\includegraphics should NOT fire."""
    latex = (
        r"\begin{figure}[h]"
        "\n"
        r"\centering"
        "\n"
        r"\includegraphics[width=\linewidth]{calibration_curve.png}"
        "\n"
        r"\end{figure}"
    )
    report = check_latex_quality(latex)
    ids = {i.pattern_id for i in report.issues}
    assert "lq_15" not in ids
