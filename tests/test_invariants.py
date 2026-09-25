"""Fixture-pinned tests for the deterministic invariant battery.

Every check here ships with a fixture that makes the artifacts it needs
and asserts it fires, plus a near-miss fixture asserting it does not. A
check whose false-positive behaviour nobody has written down is a check
nobody should switch to blocking.

Where an archived run is available on this machine, a second test pins
the check to the real ``(run_dir, defect_id)`` pair it was written
against. Those tests skip when the archive is absent, so the suite stays
green on a fresh clone, and they are the ones that would catch a check
that still passes its synthetic fixture while having drifted off the
real data.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.invariants import (
    CHECKS,
    Finding,
    RunArtifacts,
    findings_to_json,
    run_invariants,
)


pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _run(tmp_path: Path, **files) -> str:
    """Write a minimal run directory. dicts are JSON, str are verbatim."""
    d = tmp_path / "run"
    d.mkdir(exist_ok=True)
    for name, content in files.items():
        name = name.replace("__", ".")
        p = d / name
        if isinstance(content, (dict, list)):
            p.write_text(json.dumps(content), encoding="utf-8")
        else:
            p.write_text(content, encoding="utf-8")
    return str(d)


def _codes(run_dir: str) -> set[str]:
    return {f.code for f in run_invariants(run_dir)}


def _by_code(run_dir: str, code: str) -> list[Finding]:
    return [f for f in run_invariants(run_dir) if f.code == code]


# ---------------------------------------------------------------------------
# INV_MACRO_METRIC_MISLABEL
# ---------------------------------------------------------------------------


_MACRO_RESULTS = {
    "primary_metric": "AUC",
    "all_models": {
        "XGBoost": {
            "auc": 0.73,
            "recall": 0.5071271057609023,
            "balanced_accuracy": 0.5071271057609023,
            "precision": 0.6279046712665557,
        }
    },
}


def test_macro_metric_is_critical_when_the_prose_reads_it_as_positive_class(tmp_path):
    run = _run(
        tmp_path,
        results__json=_MACRO_RESULTS,
        paper__tex=(
            r"\begin{document} The model correctly identifies 341 of the 506 "
            r"true dropout episodes, a recall of 0.51. \end{document}"
        ),
    )
    hits = _by_code(run, "INV_MACRO_METRIC_MISLABEL")
    assert len(hits) == 1
    assert hits[0].severity == "critical"
    assert hits[0].evidence["manuscript_reads_as_positive_class"] is True


def test_macro_metric_is_only_minor_when_the_prose_never_interprets_it(tmp_path):
    """The measured false positive: metrics listed once in Methods.

    One archived paper lists precision/recall/F1/F2/balanced accuracy in
    a single Methods sentence and never reads any of them as a
    positive-class quantity. The artifact condition holds; the
    manuscript defect does not.
    """
    run = _run(
        tmp_path,
        results__json=_MACRO_RESULTS,
        paper__tex=(
            r"\begin{document} We also report accuracy, precision, recall, F1, "
            r"F2, and balanced accuracy for the imbalanced classification "
            r"context. \end{document}"
        ),
    )
    hits = _by_code(run, "INV_MACRO_METRIC_MISLABEL")
    assert len(hits) == 1
    assert hits[0].severity == "minor"
    assert hits[0].defect_ids == ()


def test_macro_metric_silent_when_recall_is_genuinely_positive_class(tmp_path):
    run = _run(
        tmp_path,
        results__json={
            "all_models": {
                "XGBoost": {
                    "auc": 0.73,
                    "recall": 0.018348623853211,
                    "balanced_accuracy": 0.5071271057609023,
                }
            }
        },
        paper__tex=r"\begin{document} identifies 10 of the 545 true episodes. \end{document}",
    )
    assert "INV_MACRO_METRIC_MISLABEL" not in _codes(run)


# ---------------------------------------------------------------------------
# INV_COMPARATOR_MISNAMED / INV_COMPARATOR_UNNAMED
# ---------------------------------------------------------------------------


_TWO_MODEL_RESULTS = {
    "all_models": {
        "XGBoost": {"auc": 0.7295270544978876},
        "LogisticRegression": {"auc": 0.7150381821074403},
        "RandomForest": {"auc": 0.7247359513},
    },
    "model_comparison_test": {"auc_diff": 0.014488872390447272},
}


def test_comparator_misnamed_is_critical(tmp_path):
    run = _run(
        tmp_path,
        results__json=_TWO_MODEL_RESULTS,
        paper__tex=(
            r"\begin{document} The paired cluster-bootstrap comparison between "
            r"XGBoost and the next-best individual model (RandomForest) yields "
            r"an AUC difference of 0.0145. \end{document}"
        ),
    )
    hits = _by_code(run, "INV_COMPARATOR_MISNAMED")
    assert len(hits) == 1 and hits[0].severity == "critical"
    assert hits[0].evidence["actual_contrast"] == ["XGBoost", "LogisticRegression"]


def test_comparator_not_misnamed_when_the_paper_names_the_real_pair(tmp_path):
    run = _run(
        tmp_path,
        results__json=_TWO_MODEL_RESULTS,
        paper__tex=(
            r"\begin{document} The comparison between XGBoost and "
            r"LogisticRegression yields an AUC difference of 0.0145. "
            r"\end{document}"
        ),
    )
    assert "INV_COMPARATOR_MISNAMED" not in _codes(run)


def test_unnamed_comparands_is_only_minor(tmp_path):
    """Measured base rate 17/18 -- a probe, not a detector.

    The bare "are the comparands recorded?" question fired on 17 of 18
    archived runs, because the helper had no such parameter until this
    arc. At that base rate it carries almost no information, so it stays
    advisory and exists to stop the gap recurring.
    """
    run = _run(tmp_path, results__json=_TWO_MODEL_RESULTS)
    hits = _by_code(run, "INV_COMPARATOR_UNNAMED")
    assert len(hits) == 1 and hits[0].severity == "minor"


def test_named_comparands_produce_no_finding(tmp_path):
    res = json.loads(json.dumps(_TWO_MODEL_RESULTS))
    res["model_comparison_test"].update(
        {"model_a": "XGBoost", "model_b": "LogisticRegression"}
    )
    run = _run(tmp_path, results__json=res)
    assert "INV_COMPARATOR_UNNAMED" not in _codes(run)


# ---------------------------------------------------------------------------
# INV_SUBGROUP_FALSE_UNAVAILABLE  (0 FP in 18 archived firings)
# ---------------------------------------------------------------------------


def test_false_unavailable_subgroup_fires(tmp_path):
    run = _run(
        tmp_path,
        results__json={
            "warnings": ["Subgroup attribute X1SEX not found in test_protected.csv; skipping."]
        },
        test_protected__csv="X1SEX,X1RACE\nFemale,White\nMale,White\n",
    )
    hits = _by_code(run, "INV_SUBGROUP_FALSE_UNAVAILABLE")
    assert len(hits) == 1 and hits[0].severity == "critical"


def test_false_unavailable_escalates_when_the_paper_repeats_it(tmp_path):
    run = _run(
        tmp_path,
        results__json={
            "warnings": ["Subgroup attribute X1SEX not found in test_protected.csv; skipping."]
        },
        test_protected__csv="X1SEX,X1RACE\nFemale,White\nMale,White\n",
        paper__tex=(
            r"\begin{document} However, X1SEX was not carried into the "
            r"protected-attribute test extract, so sex-specific performance "
            r"could not be computed. \end{document}"
        ),
    )
    hit = _by_code(run, "INV_SUBGROUP_FALSE_UNAVAILABLE")[0]
    assert hit.evidence["repeated_in_manuscript"] is True
    assert hit.defect_ids


def test_genuinely_absent_subgroup_is_not_flagged(tmp_path):
    run = _run(
        tmp_path,
        results__json={"warnings": ["Subgroup attribute X1SEX not found; skipping."]},
        test_protected__csv="X1RACE,X1SES\nWhite,0.1\n",
    )
    assert "INV_SUBGROUP_FALSE_UNAVAILABLE" not in _codes(run)


def test_the_fixed_pipeline_warning_is_not_itself_a_finding(tmp_path):
    """The repaired helper's own disclosure must not read as the defect."""
    run = _run(
        tmp_path,
        results__json={
            "warnings": [
                "PIPELINE: subgroup attribute 'X1SEX' is not a column of "
                "test_protected.csv (columns present: X1RACE, X1SES); subgroup "
                "analysis for 'X1SEX' skipped. This is a limitation of this "
                "run's data preparation, NOT of the dataset."
            ]
        },
        test_protected__csv="X1RACE,X1SES\nWhite,0.1\n",
    )
    assert "INV_SUBGROUP_FALSE_UNAVAILABLE" not in _codes(run)


# ---------------------------------------------------------------------------
# INV_DUMMY_CARDINALITY
# ---------------------------------------------------------------------------


def test_dummy_cardinality_fires_when_a_category_column_is_missing(tmp_path):
    run = _run(
        tmp_path,
        test_protected__csv=(
            "BYRACE\nWhite\nBlack\nAsian\nHispanic\nMultiracial\nAmerIndian\nPacific\n"
        ),
        test_X__csv="BYRACE_Black,BYRACE_Asian,BYRACE_Hispanic,BYRACE_Multiracial,BYRACE_White\n1,0,0,0,0\n",
    )
    hits = _by_code(run, "INV_DUMMY_CARDINALITY")
    assert len(hits) == 1 and hits[0].severity == "critical"
    assert hits[0].evidence["actual"] == 5


def test_dummy_cardinality_accepts_k_minus_one(tmp_path):
    run = _run(
        tmp_path,
        test_protected__csv="BYSEX\nMale\nFemale\n",
        test_X__csv="BYSEX_Male\n1\n",
    )
    assert "INV_DUMMY_CARDINALITY" not in _codes(run)


# ---------------------------------------------------------------------------
# INV_FIGURES_ORPHANED  (zero-figures form only)
# ---------------------------------------------------------------------------


def test_zero_embedded_figures_is_critical(tmp_path):
    run = _run(tmp_path, paper__tex=r"\begin{document}No figures here.\end{document}")
    (Path(run) / "love_plot.png").write_bytes(b"\x89PNG\r\n")
    hits = _by_code(run, "INV_FIGURES_ORPHANED")
    assert len(hits) == 1 and hits[0].severity == "critical"


def test_partial_orphaning_is_only_minor(tmp_path):
    """Measured 3 FP of 4: leaving two PDPs out is editorial selection."""
    run = _run(
        tmp_path,
        paper__tex=r"\begin{document}\includegraphics{roc_curves.png}\end{document}",
    )
    for n in ("roc_curves.png", "pdp_a.png", "pdp_b.png"):
        (Path(run) / n).write_bytes(b"\x89PNG\r\n")
    codes = _codes(run)
    assert "INV_FIGURES_ORPHANED" not in codes
    assert "INV_FIGURES_PARTIALLY_ORPHANED" in codes


# ---------------------------------------------------------------------------
# INV_PROSE_NUMERAL_UNBOUND
# ---------------------------------------------------------------------------


def test_unbound_count_claim_fires(tmp_path):
    run = _run(
        tmp_path,
        results__json={"all_models": {"XGBoost": {"auc": 0.73}},
                       "confusion": {"tp": 370, "fn": 175, "fp": 1375, "tn": 2777}},
        paper__tex=(
            r"\begin{document} The model correctly identifies 341 of the 506 "
            r"true dropout episodes, but it also flags 1550 non-dropouts as "
            r"at-risk. \end{document}"
        ),
    )
    hits = _by_code(run, "INV_PROSE_NUMERAL_UNBOUND")
    assert len(hits) == 1 and hits[0].severity == "critical"
    values = {u["value"] for u in hits[0].evidence["unmatched"]}
    # 341 and 506 are the catalogued fabrications. 1550 is deliberately
    # NOT asserted here: the grounding pool derives sibling sums, and
    # fp + fn = 1375 + 175 = 1550 in this fixture, so it legitimately
    # binds. On the real manuscript, whose artifacts do not offer that
    # sum, 1,550 does fire -- see the archive-pinned test below.
    assert {"341", "506"} <= values, values


def test_a_cited_paper_s_numbers_are_not_bound_to_this_run(tmp_path):
    """Related Work reports other people's results by construction."""
    run = _run(
        tmp_path,
        results__json={"all_models": {"XGBoost": {"auc": 0.73}}},
        paper__tex=(
            r"\begin{document} Their logistic regression model achieved an AUC "
            r"of 0.990 \parencite{abc123}, but their outcome differed. "
            r"\end{document}"
        ),
    )
    assert "INV_PROSE_NUMERAL_UNBOUND" not in _codes(run)


def test_binding_reports_when_it_could_not_check(tmp_path):
    """A skipped reconciliation must not look like a clean one."""
    run = _run(tmp_path, paper__tex=r"\begin{document}AUC = 0.82.\end{document}")
    assert "INV_NUMERAL_BINDING_SKIPPED" in _codes(run)


# ---------------------------------------------------------------------------
# presentation and honesty
# ---------------------------------------------------------------------------


def test_alt_text_as_body_fires_outside_acmart(tmp_path):
    run = _run(
        tmp_path,
        paper__tex=(
            "\\documentclass[man]{apa7}\n\\begin{document}"
            r"\Description{Bar chart of mean absolute SHAP values by feature.}"
            "\\end{document}"
        ),
    )
    assert "INV_ALT_TEXT_AS_BODY" in _codes(run)


def test_alt_text_is_fine_under_acmart(tmp_path):
    run = _run(
        tmp_path,
        paper__tex=(
            "\\documentclass[sigconf]{acmart}\n\\begin{document}"
            r"\Description{Bar chart.}"
            "\\end{document}"
        ),
    )
    assert "INV_ALT_TEXT_AS_BODY" not in _codes(run)


def test_unused_class_option_is_reported(tmp_path):
    """The floatsintex typo, as LaTeX itself reports it."""
    run = _run(
        tmp_path,
        paper__log="LaTeX Warning: Unused global option(s):\n    [floatsintex].\n",
    )
    hits = _by_code(run, "INV_UNUSED_CLASS_OPTION")
    assert len(hits) == 1
    assert "floatsintex" in hits[0].evidence["unused_options"][0]


# ---------------------------------------------------------------------------
# INV_LATEX_NO_PDF -- a compile that produced nothing is not a "major"
# ---------------------------------------------------------------------------


_FATAL_LOG = (
    "! File ended while scanning use of \\next.\n"
    "<inserted text>\n"
    "! Emergency stop.\n"
    "<*> ./paper.tex\n"
    "!  ==> Fatal error occurred, no output PDF file produced!\n"
)


def test_fatal_compile_with_no_pdf_is_critical(tmp_path):
    """The exact shape that shipped: released clean, no PDF anywhere."""
    run = _run(tmp_path, paper__log=_FATAL_LOG)
    hits = _by_code(run, "INV_LATEX_NO_PDF")
    assert len(hits) == 1
    assert hits[0].severity == "critical"
    assert "no output PDF file produced" in hits[0].evidence["fatal_markers"]
    assert hits[0].evidence["paper_pdf_present"] is False
    # The same errors must not also be reported as a separate major:
    # one failure, one finding.
    assert "INV_LATEX_COMPILE_ERROR" not in _codes(run)


def test_errors_with_a_pdf_beside_them_stay_major(tmp_path):
    """Three errors and a PDF is a different event from an abort."""
    run = _run(
        tmp_path,
        paper__log="! Undefined control sequence.\n! Missing $ inserted.\n",
        paper__pdf="%PDF-1.5 stub",
    )
    codes = _codes(run)
    assert "INV_LATEX_COMPILE_ERROR" in codes
    assert "INV_LATEX_NO_PDF" not in codes


def test_a_missing_pdf_is_reported_even_without_a_fatal_marker(tmp_path):
    """The log is evidence a compile was attempted; the PDF is evidence
    of what it produced."""
    run = _run(tmp_path, paper__log="This is pdfTeX, Version 3.14\n")
    hits = _by_code(run, "INV_LATEX_NO_PDF")
    assert len(hits) == 1
    assert hits[0].evidence["fatal_markers"] == []
    assert "no fatal marker" in hits[0].message


def test_a_clean_compile_is_silent(tmp_path):
    run = _run(
        tmp_path,
        paper__log="This is pdfTeX, Version 3.14\nOutput written on paper.pdf.\n",
        paper__pdf="%PDF-1.5 stub",
    )
    codes = _codes(run)
    assert "INV_LATEX_NO_PDF" not in codes
    assert "INV_LATEX_COMPILE_ERROR" not in codes


def test_no_log_at_all_claims_nothing(tmp_path):
    """No log means no compile was attempted -- not a failed one."""
    run = _run(tmp_path, paper__tex=r"\begin{document}Body.\end{document}")
    assert "INV_LATEX_NO_PDF" not in _codes(run)


# ---------------------------------------------------------------------------
# INV_UNDEFINED_CITATION -- against the lines pdflatex actually writes
# ---------------------------------------------------------------------------
#
# Captured from MiKTeX pdflatex 2025 on minimal documents (kernel \cite,
# natbib \citep/\citet, biblatex+biber with a key missing from the .bib).
# The previous pattern required "' undefined" right after the key and so
# matched none of these: every real log reads "on page N" in between.

_KERNEL_UNDEFINED = (
    "LaTeX Warning: Citation `foo2020' on page 1 undefined on input line 3.\n"
    "\n"
    "\n"
    # TeX hard-wraps the log at 79 characters, splitting a long key.
    "LaTeX Warning: Citation `averyveryveryveryveryveryverylongcitationkeyname2020ab\n"
    "cdefgh' on page 1 undefined on input line 3.\n"
    "\n"
    "LaTeX Warning: There were undefined references.\n"
)
_NATBIB_UNDEFINED = (
    "Package natbib Warning: Citation `foo2020' on page 1 undefined on input line 4.\n"
    "\n"
    "Package natbib Warning: Citation `bar2019' on page 1 undefined on input line 4.\n"
    "\n"
    "Package natbib Warning: There were undefined citations.\n"
)
_BIBLATEX_UNDEFINED = (
    "LaTeX Warning: Citation 'foo2020' on page 1 undefined on input line 5.\n"
    "\n"
    "LaTeX Warning: Citation 'bar2019' on page 1 undefined on input line 5.\n"
    "\n"
    "LaTeX Warning: Empty bibliography on input line 6.\n"
)
_BIBLATEX_OLD_MISSING_ENTRY = (
    "Package biblatex Warning: The following entry could not be found\n"
    "(biblatex)                in the database:\n"
    "(biblatex)                ghost2019\n"
    "(biblatex)                Please verify the spelling and rerun\n"
    "(biblatex)                LaTeX afterwards.\n"
)


@pytest.mark.parametrize(
    "log, keys",
    [
        (
            _KERNEL_UNDEFINED,
            {"foo2020", "averyveryveryveryveryveryverylongcitationkeyname2020abcdefgh"},
        ),
        (_NATBIB_UNDEFINED, {"foo2020", "bar2019"}),
        (_BIBLATEX_UNDEFINED, {"foo2020", "bar2019"}),
        (_BIBLATEX_OLD_MISSING_ENTRY, {"ghost2019"}),
    ],
    ids=["kernel", "natbib", "biblatex", "biblatex-missing-entry"],
)
def test_undefined_citations_in_real_log_lines_fire(tmp_path, log, keys):
    run = _run(tmp_path, paper__log=log, paper__pdf="%PDF-1.5 stub")
    hits = _by_code(run, "INV_UNDEFINED_CITATION")
    assert len(hits) == 1
    assert hits[0].severity == "critical"
    assert set(hits[0].evidence["undefined"]) == keys


def test_a_log_with_only_the_summary_line_names_no_key(tmp_path):
    """"There were undefined references" alone is not a key to report."""
    run = _run(
        tmp_path,
        paper__log="LaTeX Warning: There were undefined references.\n",
        paper__pdf="%PDF-1.5 stub",
    )
    assert "INV_UNDEFINED_CITATION" not in _codes(run)


def test_a_resolved_bibliography_is_silent(tmp_path):
    run = _run(
        tmp_path,
        paper__log=(
            "This is pdfTeX, Version 3.14\n"
            "(./paper.bbl)\n"
            "Output written on paper.pdf (1 page).\n"
        ),
        paper__pdf="%PDF-1.5 stub",
    )
    assert "INV_UNDEFINED_CITATION" not in _codes(run)


# ---------------------------------------------------------------------------
# INV_LATEX_ENVIRONMENT_UNBALANCED
# ---------------------------------------------------------------------------


def test_an_unclosed_environment_is_caught(tmp_path):
    r"""The CCSXML shape: the XML tag closes, the environment does not.

    ``</CCSXML>`` looks like a closing tag and is not one. LaTeX keeps
    reading to the end of the file and then reports a runaway argument
    850 lines from the line that actually opened the group.
    """
    run = _run(
        tmp_path,
        paper__tex=(
            "\\documentclass[sigconf]{acmart}\n"
            "\\begin{CCSXML}\n<ccs2012></ccs2012>\n</CCSXML>\n"
            "\\begin{document}Body.\\end{document}\n"
        ),
    )
    hits = _by_code(run, "INV_LATEX_ENVIRONMENT_UNBALANCED")
    assert len(hits) == 1
    assert hits[0].severity == "critical"
    assert hits[0].evidence["unbalanced"] == {"CCSXML": 1}
    assert "never closed" in hits[0].message


def test_an_extra_end_is_caught_too(tmp_path):
    run = _run(
        tmp_path,
        paper__tex=(
            "\\begin{document}\\begin{table}A\\end{table}\\end{table}"
            "\\end{document}"
        ),
    )
    hits = _by_code(run, "INV_LATEX_ENVIRONMENT_UNBALANCED")
    assert len(hits) == 1
    assert hits[0].evidence["unbalanced"] == {"table": -1}
    assert "more than it was opened" in hits[0].message


def test_a_balanced_document_is_silent(tmp_path):
    run = _run(
        tmp_path,
        paper__tex=(
            "\\begin{document}\n"
            "\\begin{table}\\begin{tabular}{ll}a&b\\end{tabular}\\end{table}\n"
            "\\begin{figure}\\includegraphics{x.png}\\end{figure}\n"
            "\\end{document}\n"
        ),
    )
    assert "INV_LATEX_ENVIRONMENT_UNBALANCED" not in _codes(run)


def test_a_commented_begin_opens_nothing(tmp_path):
    """A checker that cries wolf on correct content gets switched off."""
    run = _run(
        tmp_path,
        paper__tex=(
            "\\begin{document}\n"
            "% \\begin{table} -- kept for reference, not used\n"
            "Body.\n\\end{document}\n"
        ),
    )
    assert "INV_LATEX_ENVIRONMENT_UNBALANCED" not in _codes(run)


def test_an_environment_definition_is_not_a_use(tmp_path):
    r"""``\newenvironment`` balances across two arguments this counter
    never sees as a pair."""
    run = _run(
        tmp_path,
        paper__tex=(
            "\\newenvironment{myfig}{\\begin{figure}}{\\end{figure}}\n"
            "\\begin{document}Body.\\end{document}\n"
        ),
    )
    assert "INV_LATEX_ENVIRONMENT_UNBALANCED" not in _codes(run)


def test_a_listing_may_print_a_begin(tmp_path):
    run = _run(
        tmp_path,
        paper__tex=(
            "\\begin{document}\n"
            "\\begin{verbatim}\n\\begin{table}\n\\end{verbatim}\n"
            "\\end{document}\n"
        ),
    )
    assert "INV_LATEX_ENVIRONMENT_UNBALANCED" not in _codes(run)


def test_the_two_latex_checks_see_the_same_failure_from_both_sides(tmp_path):
    """One reads the log, one reads the source, and they agree.

    The source-side check is the one that names the cause. The log only
    ever reports where TeX gave up.
    """
    run = _run(
        tmp_path,
        paper__log=_FATAL_LOG,
        paper__tex=(
            "\\begin{CCSXML}\n</CCSXML>\n\\begin{document}B.\\end{document}\n"
        ),
    )
    codes = _codes(run)
    assert "INV_LATEX_NO_PDF" in codes
    assert "INV_LATEX_ENVIRONMENT_UNBALANCED" in codes
    named = _by_code(run, "INV_LATEX_ENVIRONMENT_UNBALANCED")[0]
    assert "CCSXML" in named.evidence["unbalanced"]


def test_unverified_block_missing_is_critical(tmp_path):
    run = _run(
        tmp_path,
        review_report__json={"overall_verdict": "REVISE", "unverified": True},
        paper__tex=r"\begin{document}A perfectly ordinary paper.\end{document}",
    )
    assert "INV_UNVERIFIED_BLOCK_MISSING" in _codes(run)


def test_unverified_block_present_is_accepted(tmp_path):
    run = _run(
        tmp_path,
        review_report__json={"overall_verdict": "REVISE", "unverified": True},
        paper__tex=(
            r"\begin{document}\textbf{WARNING: This paper has unresolved "
            r"methodological issues identified by automated review.} "
            r"\end{document}"
        ),
    )
    assert "INV_UNVERIFIED_BLOCK_MISSING" not in _codes(run)


def test_prose_authority_uncited_fires(tmp_path):
    run = _run(
        tmp_path,
        paper__tex=(
            r"\begin{document} We apply Chen's (2007) fit-change criteria. "
            r"Chen (2007) sets DCFI at -0.01. Following Chen (2007), scalar "
            r"invariance holds. \end{document}"
        ),
        references__bib="@article{x, author={Someone Else}, year={2020}}",
    )
    hits = _by_code(run, "INV_PROSE_AUTHORITY_UNCITED")
    assert len(hits) == 1
    assert hits[0].evidence["surname"] == "Chen"
    assert hits[0].evidence["occurrences"] == 3


def test_prose_authority_with_a_bib_entry_is_fine(tmp_path):
    run = _run(
        tmp_path,
        paper__tex=r"\begin{document} Chen's (2007) criteria. \end{document}",
        references__bib="@article{chen2007, author={Chen, Fang Fang}, year={2007}}",
    )
    assert "INV_PROSE_AUTHORITY_UNCITED" not in _codes(run)


def test_group_label_nan_is_reported(tmp_path):
    run = _run(
        tmp_path,
        results__json={"subgroup_performance": {"X1SESQ5": {"nan": {"n": 10778}}}},
    )
    hits = _by_code(run, "INV_GROUP_LABEL_IS_MISSING")
    assert len(hits) == 1 and hits[0].severity == "major"


def test_post_match_balance_worse_is_critical(tmp_path):
    run = _run(
        tmp_path,
        results__json={
            "balance_diagnostics": {"M2": {"smd_max_pre": 0.56, "smd_max_post": 2.624}}
        },
    )
    hits = _by_code(run, "INV_POST_MATCH_BALANCE_WORSE")
    assert len(hits) == 1 and hits[0].severity == "critical"


def test_good_matching_is_not_flagged(tmp_path):
    run = _run(
        tmp_path,
        results__json={
            "balance_diagnostics": {"M2": {"smd_max_pre": 0.56, "smd_max_post": 0.04}}
        },
    )
    assert "INV_POST_MATCH_BALANCE_WORSE" not in _codes(run)


def test_omega_pooled_over_factors_is_critical(tmp_path):
    run = _run(
        tmp_path,
        results__json={
            "P2_omega": {
                "omega_total": 0.9223395,
                "n_factors": 2,
                "omega_by_factor": {"F1": 0.90, "F2": 0.83},
            }
        },
    )
    hits = _by_code(run, "INV_OMEGA_UNIDIMENSIONAL_POOLING")
    assert len(hits) == 1 and hits[0].severity == "critical"


def test_omega_with_phi_recorded_is_accepted(tmp_path):
    run = _run(
        tmp_path,
        results__json={
            "P2_omega": {
                "omega_total": 0.9065,
                "n_factors": 2,
                "omega_by_factor": {"F1": 0.90, "F2": 0.83},
                "factor_cor_used": {"F1~~F2": 0.42},
            }
        },
    )
    assert "INV_OMEGA_UNIDIMENSIONAL_POOLING" not in _codes(run)


# ---------------------------------------------------------------------------
# battery contract
# ---------------------------------------------------------------------------


def test_every_check_is_side_effect_free_on_an_empty_directory(tmp_path):
    run = _run(tmp_path)
    before = set(os.listdir(run))
    findings = run_invariants(run)
    assert set(os.listdir(run)) == before
    assert all(f.code != "INV_CHECK_ERROR" for f in findings), [
        f.message for f in findings if f.code == "INV_CHECK_ERROR"
    ]


def test_a_raising_check_becomes_a_finding_not_a_crash(tmp_path):
    def exploding(_a: RunArtifacts) -> list[Finding]:
        raise RuntimeError("boom")

    findings = run_invariants(_run(tmp_path), checks=[exploding])
    assert len(findings) == 1
    assert findings[0].code == "INV_CHECK_ERROR"
    assert "boom" in findings[0].message


def test_findings_serialize_with_counts(tmp_path):
    run = _run(
        tmp_path,
        results__json=_MACRO_RESULTS,
        paper__tex=r"\begin{document} identifies 341 of the 506 true episodes. \end{document}",
    )
    payload = findings_to_json(run_invariants(run))
    assert payload["n_findings"] == len(payload["findings"])
    assert set(payload["counts"]) >= {"critical", "major", "minor"}
    assert json.dumps(payload)  # must be serializable as written


def test_the_as_produced_manuscript_wins_over_a_repaired_one(tmp_path):
    """Measure what the system produced, not what somebody fixed.

    A QA suite in this project returned all-clear across nine records on
    a corpus with 15 confirmed defects because it read the rebuilt
    artifact.
    """
    run = _run(
        tmp_path,
        paper__tex=r"\begin{document}repaired\end{document}",
    )
    (Path(run) / "paper.tex.orig-latex").write_text(
        r"\begin{document}as produced\end{document}", encoding="utf-8"
    )
    a = RunArtifacts(run)
    assert a.paper_name == "paper.tex.orig-latex"
    assert "as produced" in a.paper


def test_checks_registry_has_no_duplicates():
    names = [c.__name__ for c in CHECKS]
    assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# arithmetic claims about the paper's own numbers
# ---------------------------------------------------------------------------


def test_stated_gap_that_is_not_the_difference(tmp_path):
    run = _run(
        tmp_path,
        paper__tex=(
            r"\begin{document} AUC was highest for White students (0.788) and "
            r"lowest for Native Hawaiian/Pacific Islander students (0.444), a "
            r"gap of 0.528. \end{document}"
        ),
    )
    hits = _by_code(run, "INV_STATED_GAP_ARITHMETIC")
    assert len(hits) == 1 and hits[0].severity == "major"
    assert hits[0].evidence["available_differences"][0] == 0.344


def test_a_correct_gap_is_not_flagged(tmp_path):
    run = _run(
        tmp_path,
        paper__tex=(
            r"\begin{document} AUC ranged from 0.669 for the lowest group to "
            r"0.778 for the highest, a gap of 0.109. \end{document}"
        ),
    )
    assert "INV_STATED_GAP_ARITHMETIC" not in _codes(run)


def test_an_auc_difference_is_not_a_gap_claim(tmp_path):
    """The measured false-positive driver.

    An earlier version also matched "difference", "range" and "spread",
    and fired 20 times across four papers against two real defects: an
    "AUC difference of 0.0145" is an artifact value, not a subtraction
    between the numbers in its sentence.
    """
    run = _run(
        tmp_path,
        paper__tex=(
            r"\begin{document} The best model reached 0.7295 against the "
            r"baseline's 0.7150, an AUC difference of 0.9999. \end{document}"
        ),
    )
    assert "INV_STATED_GAP_ARITHMETIC" not in _codes(run)


def _school_ids(path: Path, sizes: list[int]) -> None:
    rows = ["school_id"]
    for i, n in enumerate(sizes):
        rows += [f"S{i}"] * n
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_harmonic_mean_reported_as_mean_cluster_size(tmp_path):
    run = _run(tmp_path, paper__tex="")
    sizes = [1, 2, 40, 40]          # arithmetic 20.75, harmonic 2.58
    _school_ids(Path(run) / "train_school_ids.csv", sizes)
    (Path(run) / "paper.tex").write_text(
        r"\begin{document} The mean cluster size was 2.58. \end{document}",
        encoding="utf-8",
    )
    hits = _by_code(run, "INV_HARMONIC_MEAN_AS_MEAN")
    assert len(hits) == 1
    assert hits[0].evidence["arithmetic_mean"] == 20.75


def test_the_arithmetic_mean_cluster_size_is_accepted(tmp_path):
    run = _run(tmp_path, paper__tex="")
    _school_ids(Path(run) / "train_school_ids.csv", [1, 2, 40, 40])
    (Path(run) / "paper.tex").write_text(
        r"\begin{document} The mean cluster size was 20.75. \end{document}",
        encoding="utf-8",
    )
    codes = _codes(run)
    assert "INV_HARMONIC_MEAN_AS_MEAN" not in codes
    assert "INV_CLUSTER_SIZE_UNRECONCILED" not in codes


def test_flagged_variable_count_mismatch(tmp_path):
    run = _run(
        tmp_path,
        data_report__json={
            "missingness_summary": {
                "A": {"pct_missing": 25.0}, "B": {"pct_missing": 22.0},
                "C": {"pct_missing": 21.0}, "D": {"pct_missing": 20.5},
                "E": {"pct_missing": 3.0},
            }
        },
        paper__tex=(
            r"\begin{document} Three predictors exceeded 20\% missingness. "
            r"\end{document}"
        ),
    )
    hits = _by_code(run, "INV_FLAGGED_COUNT_MISMATCH")
    assert len(hits) == 1
    assert hits[0].evidence["actual"] == 4


def test_a_correct_flagged_count_is_accepted(tmp_path):
    run = _run(
        tmp_path,
        data_report__json={
            "missingness_summary": {
                "A": {"pct_missing": 25.0}, "B": {"pct_missing": 3.0},
            }
        },
        paper__tex=r"\begin{document} One predictor exceeded 20\% missingness. \end{document}",
    )
    assert "INV_FLAGGED_COUNT_MISMATCH" not in _codes(run)


def test_imputation_method_mismatch(tmp_path):
    run = _run(
        tmp_path,
        data_report__json={
            "missingness_summary": {
                "X2STUEDEXPCT": {"pct_missing": 12.0,
                                 "imputation_method": "IterativeImputer"}
            }
        },
        paper__tex=(
            r"\begin{document} Categorical variables (X2STUEDEXPCT, X1SEX) "
            r"were imputed with the mode. \end{document}"
        ),
    )
    hits = _by_code(run, "INV_IMPUTATION_METHOD_MISMATCH")
    assert len(hits) == 1 and hits[0].evidence["claimed"] == "mode"


def test_a_two_clause_methods_sentence_attributes_by_clause(tmp_path):
    """The measured false positive, and the reason for clause attribution.

    "imputed missing values using IterativeImputer for the continuous and
    ordinal variables (BYSES1, BYPARED) and mode imputation for BYSEX" --
    BYSES1 is 88 characters past the start of the first claim and 38
    before the second, so nearest-by-distance handed it to `mode` and
    produced four spurious findings on one paper.
    """
    run = _run(
        tmp_path,
        data_report__json={
            "missingness_summary": {
                "BYSES1": {"pct_missing": 5.4, "imputation_method": "IterativeImputer"},
                "BYSEX": {"pct_missing": 4.8, "imputation_method": "mode"},
            }
        },
        paper__tex=(
            r"\begin{document} We imputed missing values using IterativeImputer "
            r"for the continuous and ordinal variables (BYSES1, BYPARED) and "
            r"mode imputation for BYSEX and BYSCHPRG. \end{document}"
        ),
    )
    assert "INV_IMPUTATION_METHOD_MISMATCH" not in _codes(run)


# ---------------------------------------------------------------------------
# a paper that says nothing passes every other check in this module
# ---------------------------------------------------------------------------


def test_an_empty_manuscript_is_critical(tmp_path):
    r"""The 287-byte paper.tex that shipped as COMPLETED.

    The Writer produced a complete 69 KB manuscript, the response was cut
    off mid-sentence at the token ceiling, the reassembler's body regex
    found no closing boundary and substituted an empty body into the
    template. What reached disk was a title, \bibliographystyle,
    \bibliography and \end{document}.
    """
    run = _run(
        tmp_path,
        paper__tex=(
            r"\documentclass{acmart}\n\title{Predicting Dropout}\n"
            r"\begin{document}\maketitle\n"
            r"\bibliographystyle{ACM-Reference-Format}\n"
            r"\bibliography{references}\n\end{document}\n"
        ),
    )
    hits = _by_code(run, "INV_MANUSCRIPT_EMPTY")
    assert len(hits) == 1 and hits[0].severity == "critical"
    assert hits[0].evidence["body_words"] < 20


def test_a_real_length_manuscript_is_not_flagged(tmp_path):
    body = " ".join(["students persisted through the follow up wave"] * 100)
    run = _run(
        tmp_path,
        paper__tex=(
            "\\begin{document}\\maketitle\\n"
            + body
            + "\\n\\end{document}"
        ),
    )
    assert "INV_MANUSCRIPT_EMPTY" not in _codes(run)


def test_bibliography_ampersand_is_reported(tmp_path):
    run = _run(
        tmp_path,
        references__bib="@article{x, journal={Science & Education}, year={2020}}",
    )
    hits = _by_code(run, "INV_BIB_AMPERSAND")
    assert len(hits) == 1
    assert hits[0].evidence["unescaped_ampersands"] == 1


def test_an_escaped_ampersand_is_fine(tmp_path):
    run = _run(
        tmp_path,
        references__bib=r"@article{x, journal={Science \& Education}, year={2020}}",
    )
    assert "INV_BIB_AMPERSAND" not in _codes(run)


def test_a_boolean_dummy_column_is_not_reported_as_constant(tmp_path):
    """pandas writes one-hot dummies as True/False, not 0/1.

    The float parser rejected those cells, which left the uniqueness set
    empty and the sum at 0.0 -- so a perfectly varying column was
    reported as "constant across all 4,697 test rows (sum 0.0)". One
    baseline run produced 37 such findings, one per dummy, with exactly
    one of them real. A metric dominated by that would have read as a
    37-point improvement from a single fix.
    """
    run = _run(
        tmp_path,
        test_X__csv="X1SEX_1.0,X1SEX_2.0\nTrue,False\nFalse,False\nTrue,False\n",
        feature_importance__csv=(
            "feature,shap_mean_abs\nX1SEX_1.0,0.105\nX1SEX_2.0,0.094\n"
        ),
    )
    hits = _by_code(run, "INV_DEGENERATE_FEATURE_WEIGHT")
    assert [h.evidence["column"] for h in hits] == ["X1SEX_2.0"], [
        h.evidence for h in hits
    ]


# ---------------------------------------------------------------------------
# INV_PERCENTAGE_FROM_SPEC_NOT_RUN
# ---------------------------------------------------------------------------


def test_a_percentage_from_the_spec_alone_is_flagged(tmp_path):
    """The paper printed the plan instead of the result.

    research_spec.json carries pre-analysis estimates -- an anticipated
    class split, expected missingness. One paper wrote "approximately
    15% non-persisters" where the outcome CSVs give 20.0%, and 15
    appears only in the spec.
    """
    run = _run(
        tmp_path,
        research_spec__json={
            "potential_limitations": ["roughly 15% of the sample are non-persisters"]
        },
        data_report__json={"analytic_n": 12918},
        train_y__csv="y\n" + "\n".join(["1"] * 80 + ["0"] * 20),
        paper__tex=(
            r"\begin{document} The analytic sample has approximately 15\% "
            r"non-persisters. \end{document}"
        ),
    )
    hits = _by_code(run, "INV_PERCENTAGE_FROM_SPEC_NOT_RUN")
    assert len(hits) == 1 and hits[0].evidence["value"] == "15"


def test_a_percentage_the_run_computed_is_not_flagged(tmp_path):
    run = _run(
        tmp_path,
        research_spec__json={"potential_limitations": ["roughly 15% non-persisters"]},
        data_report__json={"analytic_n": 100},
        train_y__csv="y\n" + "\n".join(["1"] * 80 + ["0"] * 20),
        paper__tex=r"\begin{document} 20\% of the sample are non-persisters. \end{document}",
    )
    assert "INV_PERCENTAGE_FROM_SPEC_NOT_RUN" not in _codes(run)


def test_a_percentage_derivable_from_the_outcome_csvs_is_not_flagged(tmp_path):
    """The measured false-positive driver.

    Three correct sentences in three papers -- "11% base rate",
    "5,100 students (38.5%)", "39.0% attended a two-year public
    institution" -- are all derivable from train_y + test_y and appear
    in no JSON. The rubric licenses arithmetic on artifacts; without the
    y CSVs in the computed set, all three read as spec-only.
    """
    run = _run(
        tmp_path,
        research_spec__json={"note": "we expect around 20 percent"},
        data_report__json={"analytic_n": 100},
        train_y__csv="y\n" + "\n".join(["1"] * 20 + ["0"] * 80),
        paper__tex=r"\begin{document} The base rate is 20\%. \end{document}",
    )
    assert "INV_PERCENTAGE_FROM_SPEC_NOT_RUN" not in _codes(run)


# ---------------------------------------------------------------------------
# INV_CLASS_BALANCE_WRONG_SAMPLE
# ---------------------------------------------------------------------------


def test_class_balance_counts_that_describe_the_training_set(tmp_path):
    run = _run(
        tmp_path,
        data_report__json={
            "analytic_n": 12918,
            "n_train": 10289,
            "n_test": 2629,
            "class_balance": {"class_0": 2070, "class_1": 8219},
        },
    )
    hits = _by_code(run, "INV_CLASS_BALANCE_WRONG_SAMPLE")
    assert len(hits) == 1 and hits[0].evidence["sum"] == 10289


def test_class_balance_over_the_analytic_sample_is_fine(tmp_path):
    run = _run(
        tmp_path,
        data_report__json={
            "analytic_n": 12918,
            "n_train": 10289,
            "class_balance": {"class_0": 2587, "class_1": 10331},
        },
    )
    assert "INV_CLASS_BALANCE_WRONG_SAMPLE" not in _codes(run)


def test_class_balance_as_proportions_is_not_a_count_claim(tmp_path):
    run = _run(
        tmp_path,
        data_report__json={
            "analytic_n": 23503,
            "n_train": 18806,
            "class_balance": {"class_0": 0.892, "class_1": 0.108},
        },
    )
    assert "INV_CLASS_BALANCE_WRONG_SAMPLE" not in _codes(run)


# ---------------------------------------------------------------------------
# INV_SUPERLATIVE_CONTRADICTED
# ---------------------------------------------------------------------------

_RANKED = {
    "all_models": {
        "XGBoost": {"auc": 0.7041860483168286},
        "StackingEnsemble": {"auc": 0.7045036583370043},
        "RandomForest": {"auc": 0.6994},
        "LogisticRegression": {"auc": 0.6991},
    }
}


def test_a_superlative_the_table_contradicts(tmp_path):
    """J38's shape: five models named, the claim belongs to the first."""
    run = _run(
        tmp_path,
        results__json=_RANKED,
        paper__tex=(
            r"\begin{document} XGBoost achieved the highest point estimate "
            r"(AUC = 0.704), followed closely by the stacking ensemble "
            r"(AUC = 0.705), random forest (AUC = 0.699), and logistic "
            r"regression (AUC = 0.699). \end{document}"
        ),
    )
    hits = _by_code(run, "INV_SUPERLATIVE_CONTRADICTED")
    assert len(hits) == 1
    assert hits[0].evidence["claimed"] == "XGBoost"
    assert hits[0].evidence["actual"] == "StackingEnsemble"


def test_the_individual_qualifier_excludes_the_ensemble(tmp_path):
    """This pipeline's own selection rule excludes the stacking model."""
    run = _run(
        tmp_path,
        results__json=_RANKED,
        paper__tex=(
            r"\begin{document} XGBoost was the best individual model "
            r"(AUC = 0.704). \end{document}"
        ),
    )
    assert "INV_SUPERLATIVE_CONTRADICTED" not in _codes(run)


def test_an_ordinal_is_not_a_claim_about_the_maximum(tmp_path):
    """"the next-best individual model (RandomForest)" is about SECOND."""
    run = _run(
        tmp_path,
        results__json=_RANKED,
        paper__tex=(
            r"\begin{document} The comparison between XGBoost and the "
            r"next-best individual model (RandomForest) yields an AUC "
            r"difference of 0.0047. \end{document}"
        ),
    )
    assert "INV_SUPERLATIVE_CONTRADICTED" not in _codes(run)


def test_a_cited_papers_ranking_is_not_this_papers_claim(tmp_path):
    run = _run(
        tmp_path,
        results__json=_RANKED,
        paper__tex=(
            r"\begin{document} \textcite{abc} found that random forest "
            r"achieved the highest classification accuracy at 67.73\%. "
            r"\end{document}"
        ),
    )
    assert "INV_SUPERLATIVE_CONTRADICTED" not in _codes(run)


def test_lowest_rmse_means_the_minimum_rmse(tmp_path):
    """A literal direction word is not a quality word.

    Treating "lowest" as "the good end" and then flipping again because
    RMSE is lower-is-better expects the WORST model, and reports a
    correct sentence as contradicted.
    """
    run = _run(
        tmp_path,
        results__json={
            "all_models": {
                "RandomForest": {"rmse": 0.756},
                "LinearRegression": {"rmse": 0.834},
            }
        },
        paper__tex=(
            r"\begin{document} Random Forest achieved the lowest root mean "
            r"squared error (RMSE = 0.756). \end{document}"
        ),
    )
    assert "INV_SUPERLATIVE_CONTRADICTED" not in _codes(run)


def test_a_generic_best_model_phrase_names_nothing(tmp_path):
    run = _run(
        tmp_path,
        results__json=_RANKED,
        paper__tex=(
            r"\begin{document} Model comparison used a paired bootstrap test "
            r"of the AUC difference between the best model and the logistic "
            r"regression baseline. \end{document}"
        ),
    )
    assert "INV_SUPERLATIVE_CONTRADICTED" not in _codes(run)

