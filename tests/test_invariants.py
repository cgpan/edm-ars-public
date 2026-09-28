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


#: The first finished paper from the owner's Mac test (round 3): five
#: models, and a comparison test recorded as XGBoost minus
#: LogisticRegression. Values copied from that run's results.json.
_R3_RESULTS = {
    "best_model": "XGBoost",
    "all_models": {
        "LogisticRegression": {"auc": 0.7884831280985126},
        "RandomForest": {"auc": 0.7985067167759475},
        "XGBoost": {"auc": 0.801488285622901},
        "ElasticNet": {"auc": 0.7812581960658883},
        "StackingEnsemble": {"auc": 0.802020230289461},
    },
    "model_comparison_test": {
        "auc_diff": 0.013005157524388355,
        "model_a": "XGBoost",
        "model_b": "LogisticRegression",
        "contrast": "XGBoost - LogisticRegression",
    },
}

#: Table 1 of that paper, then the sentence that follows it. The header
#: row ends in "Bal. Acc.", so a split on terminal punctuation fell there
#: and glued every model-name row to the comparison sentence.
_R3_TABLE_THEN_COMPARISON = r"""\begin{document}
\subsection{Model Comparison}
Table~\ref{tab:models} reports the performance of all five models.

\begin{table}
\caption{Model comparison on the held-out test set ($n = 3{,}562$). AUC is the primary metric; 95\% confidence intervals are bootstrap (1,000 iterations).}
\label{tab:models}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lrrrrr}
\toprule
Model & AUC & CI Low & CI High & Acc. & Bal. Acc. \\
\midrule
LogisticRegression & 0.788 & 0.773 & 0.804 & 0.761 & 0.616 \\
RandomForest & 0.799 & 0.784 & 0.814 & 0.759 & 0.617 \\
XGBoost & 0.801 & 0.787 & 0.817 & 0.765 & 0.623 \\
ElasticNet & 0.781 & 0.766 & 0.797 & 0.753 & 0.576 \\
StackingEnsemble & 0.802 & 0.787 & 0.817 & 0.765 & 0.630 \\
\bottomrule
\end{tabular}%
}
\end{table}

The paired comparison between XGBoost and Logistic Regression showed a statistically significant difference: AUC difference $= 0.013$, 95\% CI [0.006, 0.021], cluster-bootstrap, $p < 0.05$. The advantage is small in magnitude.
\end{document}
"""


def test_a_table_above_the_comparison_sentence_is_not_part_of_it(tmp_path):
    """Round-3 Mac paper: a correct sentence reported as critical.

    The sentence names XGBoost and Logistic Regression, which is the pair
    results.json recorded. The finding named RandomForest, ElasticNet and
    StackingEnsemble -- rows of the table above it.
    """
    run = _run(tmp_path, results__json=_R3_RESULTS, paper__tex=_R3_TABLE_THEN_COMPARISON)
    assert "INV_COMPARATOR_MISNAMED" not in _codes(run)


def test_the_comparison_sentence_still_fires_when_it_names_the_wrong_model(tmp_path):
    """The same table and layout, with the sentence itself wrong."""
    tex = _R3_TABLE_THEN_COMPARISON.replace(
        "between XGBoost and Logistic Regression", "between XGBoost and Random Forest"
    )
    run = _run(tmp_path, results__json=_R3_RESULTS, paper__tex=tex)
    hits = _by_code(run, "INV_COMPARATOR_MISNAMED")
    assert len(hits) == 1
    assert hits[0].evidence["named_in_paper"] == ["RandomForest", "XGBoost"]
    assert "midrule" not in hits[0].evidence["sentence"]


def test_a_comparator_spelled_as_prose_is_still_read(tmp_path):
    """J53: the one sentence that named the wrong model spelled it out.

    "...between Logistic Regression and the runner-up (Random Forest)"
    where the test was LogisticRegression minus XGBoost. The CamelCase
    key never appears in that sentence; only the table above it had
    matched before, and for the wrong reason.
    """
    run = _run(
        tmp_path,
        results__json={
            "all_models": {
                "LogisticRegression": {"auc": 0.823899450821177},
                "XGBoost": {"auc": 0.8184131758811203},
                "RandomForest": {"auc": 0.8174343},
            },
            "model_comparison_test": {"auc_diff": 0.005486274940056712},
        },
        paper__tex=(
            r"\begin{document} The paired cluster-bootstrap test of the AUC "
            r"difference between Logistic Regression and the runner-up (Random "
            r"Forest) yielded $\Delta$AUC = 0.005, 95\% CI [0.002, 0.009]. "
            r"\end{document}"
        ),
    )
    hits = _by_code(run, "INV_COMPARATOR_MISNAMED")
    assert len(hits) == 1
    assert hits[0].evidence["named_in_paper"] == ["LogisticRegression", "RandomForest"]


def test_a_range_that_equals_the_difference_is_not_a_test_report(tmp_path):
    """"All five models performed within 0.010 AUC of one another: ..."

    0.010 is also the rounded auc_diff, so the sentence was read as the
    test report and every model it lists as a comparator. The paper's
    real test sentence names the right pair.
    """
    run = _run(
        tmp_path,
        results__json={
            "all_models": {
                "LogisticRegression": {"auc": 0.749},
                "ElasticNet": {"auc": 0.759486570710109389},
                "XGBoost": {"auc": 0.756},
            },
            "model_comparison_test": {
                "auc_diff": 0.759486570710109389 - 0.749,
                "model_a": "ElasticNet",
                "model_b": "LogisticRegression",
            },
        },
        paper__tex=(
            r"\begin{document} All five models performed within 0.010 AUC of one "
            r"another: Logistic Regression (0.749), XGBoost (0.756) and "
            r"ElasticNet (0.760). The paired cluster-bootstrap comparison between "
            r"ElasticNet and Logistic Regression yielded $\Delta$AUC = 0.010. "
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


#: The front matter of the ACM template the Writer fills
#: (templates/paper_template_v2.tex), then one of the round-3 paper's
#: seven figures. Its 19 "unescaped underscores" were the 12 in this
#: CCSXML block and one in each figure's file name.
_ACM_FRONT_MATTER = r"""\documentclass[sigconf]{acmart}
\begin{document}
\title{Do Ninth-Grade Non-Cognitive Factors Improve Prediction of College Enrollment?}
\begin{CCSXML}
<ccs2012>
 <concept>
  <concept_id>10010147.10010178</concept_id>
  <concept_desc>Computing methodologies~Machine learning</concept_desc>
  <concept_significance>500</concept_significance>
 </concept>
 <concept>
  <concept_id>10003456.10003457.10003527</concept_id>
  <concept_desc>Social and professional topics~Student assessment</concept_desc>
  <concept_significance>500</concept_significance>
 </concept>
</ccs2012>
\end{CCSXML}
\ccsdesc[500]{Computing methodologies~Machine learning}
\maketitle
We trained five model families with drop\_first=True encoding.
\begin{figure}
\includegraphics[width=\columnwidth]{roc_curves.png}
\caption{ROC curves for all five models on the held-out test set.}
\Description{ROC curves for five models.}
\label{fig:roc_curves}
\end{figure}
"""


def test_ccsxml_and_figure_file_names_are_not_prose(tmp_path):
    """Round-3 Mac paper: 19 reported underscores, none of them in its text."""
    run = _run(tmp_path, paper__tex=_ACM_FRONT_MATTER + r"\end{document}")
    assert "INV_UNESCAPED_LATEX_SPECIAL" not in _codes(run)


def test_display_math_subscripts_are_not_prose(tmp_path):
    run = _run(
        tmp_path,
        paper__tex=(
            r"\begin{document} The estimand is \begin{equation} \hat{\mu}_{g,t} "
            r"= \text{low\_ses}_{1} \end{equation} \begin{align*} a &= b_1 \\ "
            r"c &= d_2 \end{align*} \end{document}"
        ),
    )
    assert "INV_UNESCAPED_LATEX_SPECIAL" not in _codes(run)


def test_a_bare_underscore_in_prose_still_fires(tmp_path):
    """T14: "(F1SCH_ID)" in prose ran a page of text together in italic."""
    run = _run(
        tmp_path,
        paper__tex=(
            _ACM_FRONT_MATTER
            + r"we used the first follow-up school identifier (F1SCH_ID), which "
            r"carries 752 real school IDs. \end{document}"
        ),
    )
    hits = _by_code(run, "INV_UNESCAPED_LATEX_SPECIAL")
    assert len(hits) == 1 and hits[0].severity == "major"
    # The only underscore reported is the one in the prose.
    assert len(hits[0].evidence["_"]) == 1
    assert "F1SCH_ID" in hits[0].evidence["_"][0]


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


def test_a_manuscript_with_no_log_and_no_pdf_is_critical(tmp_path):
    """pdflatex never ran (not installed / not on PATH): no paper.log and
    no paper.pdf. This used to claim nothing -- "no log means no compile
    was attempted" -- but the orchestrator compiles every manuscript it
    writes, so the one blocking code could not fire and a run with no PDF
    was released as clean (B1)."""
    run = _run(tmp_path, paper__tex=r"\begin{document}Body.\end{document}")
    hits = _by_code(run, "INV_LATEX_NO_PDF")
    assert len(hits) == 1
    assert hits[0].severity == "critical"
    assert hits[0].evidence["paper_log_present"] is False
    assert hits[0].evidence["compile_ran"] is False
    assert "never ran" in hits[0].message


def test_no_log_names_the_missing_tool_from_the_compile_record(tmp_path):
    """latex_compile.json is the only place the reason survives."""
    run = _run(
        tmp_path,
        paper__tex=r"\begin{document}Body.\end{document}",
        latex_compile__json={
            "success": False,
            "pdf_exists": False,
            "missing_tool": "pdflatex",
            "steps": [
                {
                    "cmd": "pdflatex -interaction=nonstopmode paper.tex",
                    "returncode": -1,
                    "stderr": "'pdflatex' not found - is it installed and on PATH?",
                }
            ],
        },
    )
    hits = _by_code(run, "INV_LATEX_NO_PDF")
    assert len(hits) == 1
    assert hits[0].evidence["missing_tool"] == "pdflatex"
    assert "pdflatex was not found" in hits[0].message


def test_no_log_but_a_pdf_claims_nothing(tmp_path):
    """A PDF with its log cleaned up afterwards is a delivered paper."""
    run = _run(
        tmp_path,
        paper__tex=r"\begin{document}Body.\end{document}",
        paper__pdf="%PDF-1.5 stub",
    )
    assert "INV_LATEX_NO_PDF" not in _codes(run)


def test_no_manuscript_no_log_claims_nothing(tmp_path):
    """An aborted run wrote no paper, so there is nothing to compile."""
    run = _run(tmp_path, results__json={"best_model": "X"})
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


#: The round-3 Mac paper's two subgroup sentences, verbatim. Both are
#: right: 0.835 - 0.743 = 0.092, and the SES cells are 0.8215 and 0.6784
#: unrounded, 0.1431 apart.
_R3_RACE_GAP = (
    r"Among adequately sized groups, White students had the highest AUC "
    r"(0.835, $n = 1{,}864$) and Hispanic students (race specified) the lowest "
    r"(0.743, $n = 488$), a gap of 9.2 percentage points that exceeds the "
    r"fairness threshold."
)
_R3_SES_GAP = (
    r"For SES quintiles, AUC ranged from 0.678 (lowest quintile) to 0.822 "
    r"(highest quintile), a gap of 14.3 percentage points."
)


def test_a_gap_in_percentage_points_between_proportions(tmp_path):
    run = _run(
        tmp_path,
        paper__tex=r"\begin{document} " + _R3_RACE_GAP + " " + _R3_SES_GAP + r" \end{document}",
    )
    assert "INV_STATED_GAP_ARITHMETIC" not in _codes(run)


def test_a_wrong_gap_in_percentage_points_still_fires(tmp_path):
    run = _run(
        tmp_path,
        paper__tex=(
            r"\begin{document} "
            + _R3_RACE_GAP.replace("9.2 percentage points", "12.5 percentage points")
            + r" \end{document}"
        ),
    )
    hits = _by_code(run, "INV_STATED_GAP_ARITHMETIC")
    assert len(hits) == 1
    assert hits[0].evidence["stated_gap"] == "12.5"
    assert hits[0].evidence["stated_in_points"] is True


def test_the_points_scale_needs_a_unit(tmp_path):
    """"a gap of 9.2" between 0.835 and 0.743 names no scale: still wrong."""
    run = _run(
        tmp_path,
        paper__tex=(
            r"\begin{document} "
            + _R3_RACE_GAP.replace("9.2 percentage points", "9.2")
            + r" \end{document}"
        ),
    )
    assert len(_by_code(run, "INV_STATED_GAP_ARITHMETIC")) == 1


def test_a_gap_off_by_more_than_rounding_still_fires(tmp_path):
    """J39, verbatim: the cells are 0.6690 and 0.7778, 0.1088 apart."""
    run = _run(
        tmp_path,
        paper__tex=(
            r"\begin{document} The range across the interpretable cells is "
            r"0.669 to 0.778, a gap of 0.112. \end{document}"
        ),
    )
    assert len(_by_code(run, "INV_STATED_GAP_ARITHMETIC")) == 1


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


_R3_IMPUTATION_SENTENCE = (
    r"\begin{document} Two predictors exceeded the 20\% missingness threshold. "
    r"Continuous predictors were imputed using IterativeImputer; categorical "
    r"predictors (X1RACE, X1SEX) were imputed using the mode. \end{document}"
)


def test_a_variable_takes_the_method_of_its_own_clause(tmp_path):
    """Round-3 Mac paper: two major findings on a correct sentence.

    The claim that precedes X1RACE ("imputed using IterativeImputer")
    belongs to the clause before the semicolon; the variables' own
    clause names the mode, which is what data_report recorded.
    """
    run = _run(
        tmp_path,
        data_report__json={
            "missingness_summary": {
                "X1RACE": {"pct_missing": 4.3, "imputation_method": "mode"},
                "X1SEX": {"pct_missing": 0.02, "imputation_method": "mode"},
                "X1SES": {"pct_missing": 8.8, "imputation_method": "IterativeImputer"},
            }
        },
        paper__tex=_R3_IMPUTATION_SENTENCE,
    )
    assert "INV_IMPUTATION_METHOD_MISMATCH" not in _codes(run)


def test_the_clause_method_is_still_held_against_the_data_report(tmp_path):
    """Same sentence, but the data report says X1RACE was not mode-imputed."""
    run = _run(
        tmp_path,
        data_report__json={
            "missingness_summary": {
                "X1RACE": {"pct_missing": 4.3, "imputation_method": "IterativeImputer"},
                "X1SEX": {"pct_missing": 0.02, "imputation_method": "mode"},
            }
        },
        paper__tex=_R3_IMPUTATION_SENTENCE,
    )
    hits = _by_code(run, "INV_IMPUTATION_METHOD_MISMATCH")
    assert [(h.evidence["variable"], h.evidence["claimed"]) for h in hits] == [
        ("X1RACE", "mode")
    ]


def test_a_comma_and_clause_is_a_clause_too(tmp_path):
    run = _run(
        tmp_path,
        data_report__json={
            "missingness_summary": {
                "X1RACE": {"pct_missing": 4.3, "imputation_method": "mode"},
            }
        },
        paper__tex=(
            r"\begin{document} Continuous predictors were imputed using "
            r"IterativeImputer, and categorical predictors (X1RACE, X1SEX, and "
            r"X1LOCALE) were imputed using the mode. \end{document}"
        ),
    )
    assert "INV_IMPUTATION_METHOD_MISMATCH" not in _codes(run)


_ACCESS_MISSINGNESS = {
    "BYSTEXP": {"pct_missing": 9.0, "imputation_method": "IterativeImputer"},
    "BYSES1": {"pct_missing": 5.4, "imputation_method": "IterativeImputer"},
    "BYRACE": {"pct_missing": 5.1, "imputation_method": "IterativeImputer"},
    "BYTXMSTD": {"pct_missing": 3.0, "imputation_method": "median"},
    "BYSEX": {"pct_missing": 4.8, "imputation_method": "mode"},
}

#: An archived paper's methods sentence, verbatim. It is correct, and the
#: check reported five of its variables as median-imputed.
_ACCESS_SENTENCE = (
    r"\begin{document} We imputed missing values using methods appropriate to "
    r"each variable's type: iterative imputation (IterativeImputer) for the "
    r"continuous and ordered-categorical variables with substantial "
    r"missingness (BYSTEXP, BYSES1, BYMATHSE, BYRISKFC, BYRACE), median "
    r"imputation for the two achievement scores (BYTXMSTD, BYTXRSTD), and "
    r"mode imputation for the categorical variables with low missingness "
    r"(BYPARED, BYSCHPRG, BYSEX). \end{document}"
)


def test_iterative_imputation_names_the_iterative_imputer(tmp_path):
    run = _run(
        tmp_path,
        data_report__json={"missingness_summary": _ACCESS_MISSINGNESS},
        paper__tex=_ACCESS_SENTENCE,
    )
    assert "INV_IMPUTATION_METHOD_MISMATCH" not in _codes(run)


def test_iterative_imputation_is_still_a_claim_the_report_can_refute(tmp_path):
    miss = json.loads(json.dumps(_ACCESS_MISSINGNESS))
    miss["BYSES1"]["imputation_method"] = "median"
    run = _run(
        tmp_path,
        data_report__json={"missingness_summary": miss},
        paper__tex=_ACCESS_SENTENCE,
    )
    hits = _by_code(run, "INV_IMPUTATION_METHOD_MISMATCH")
    assert [(h.evidence["variable"], h.evidence["claimed"]) for h in hits] == [
        ("BYSES1", "iterative")
    ]


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


#: The round-3 Mac run's sample counts, from its data_report.json.
_R3_DATA_REPORT = {
    "original_n": 23503,
    "analytic_n": 17335,
    "n_train": 13773,
    "n_test": 3562,
    "class_balance": {"class_0": 3319, "class_1": 10454},
    "missingness_summary": {
        "X1STUEDEXPCT": {"pct_missing": 27.4, "imputation_method": "IterativeImputer"},
        "X1PAREDU": {"pct_missing": 24.2, "imputation_method": "IterativeImputer"},
        "X1RACE": {"pct_missing": 4.3, "imputation_method": "mode"},
        "X1SEX": {"pct_missing": 0.02, "imputation_method": "mode"},
    },
}


def test_a_percentage_derivable_from_the_sample_counts_is_not_flagged(tmp_path):
    """Round-3 Mac paper: "approximately 26% missingness" is 1 - 17,335/23,503.

    The spec guessed the same figure before the run, which is why the
    check looked at it; the run's own counts give 26.2%, which prints as
    26 at the precision the sentence uses.
    """
    run = _run(
        tmp_path,
        research_spec__json={
            "potential_limitations": [
                "X4EVRATNDCLG has approximately 26% missingness; complete-case "
                "analysis on the outcome may introduce bias"
            ]
        },
        data_report__json=_R3_DATA_REPORT,
        results__json={"best_metric_value": 0.801488285622901},
        paper__tex=(
            r"\begin{document} However, the college enrollment outcome itself "
            r"has approximately 26\% missingness, which may be non-random (MNAR); "
            r"complete-case analysis may bias estimates. \end{document}"
        ),
    )
    assert "INV_PERCENTAGE_FROM_SPEC_NOT_RUN" not in _codes(run)


def test_a_count_ratio_is_held_to_the_printed_precision(tmp_path):
    """26.2 derives from the counts; 26.8 and 25 do not, and stay flagged."""
    for printed, fires in (("26.2", False), ("26.8", True), ("25", True)):
        run = _run(
            tmp_path,
            research_spec__json={"note": f"expect about {printed}% missing outcome"},
            data_report__json=_R3_DATA_REPORT,
            paper__tex=(
                r"\begin{document} The outcome has " + printed
                + r"\% missingness. \end{document}"
            ),
        )
        assert ("INV_PERCENTAGE_FROM_SPEC_NOT_RUN" in _codes(run)) is fires, printed


def test_a_split_share_does_not_excuse_a_planned_whole_number(tmp_path):
    """n_test / analytic_n is about 20% in every run; "20%" is not therefore derived.

    Here the test share is 4,786 / 23,503 = 20.36% (an archived run's
    counts), which prints as 20 at whole-number precision. A spec that
    planned "about 20% non-completers" and a paper that prints it are
    still flagged.
    """
    run = _run(
        tmp_path,
        research_spec__json={"note": "we expect about 20% non-completers"},
        data_report__json={
            "original_n": 23503, "analytic_n": 23503,
            "n_train": 18717, "n_test": 4786,
        },
        paper__tex=r"\begin{document} About 20\% of students did not complete. \end{document}",
    )
    assert "INV_PERCENTAGE_FROM_SPEC_NOT_RUN" in _codes(run)


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

