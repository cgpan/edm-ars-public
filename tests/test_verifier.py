"""Tests for the post-Writer verification judge's validator.

The judge itself is an LLM and is not tested here. What IS tested is
everything around it: the contract that decides which of its findings
anybody is allowed to read. A judge whose output nothing filters is a
judge whose false positives are the system's false positives.

The measured precision context, recorded so nobody re-derives it: the
rater contract in ``agent_prompts/verifier_rubric.md`` scored 98% flag
precision against the built-ins' 67% -- CROSS-FAMILY, a Kimi rater
against a DeepSeek generator. This agent runs same-family. Expect less,
and report the gap rather than borrowing the number.
"""

from __future__ import annotations

import types

import pytest

from src.agents.verifier import (
    MAX_FINDINGS,
    Verifier,
    _normalize_for_quote_match,
    _SELF_NEGATING,
)

PAPER = (
    r"\section{Results}" "\n"
    r"The model correctly identifies 341 of the 506 true dropout episodes "
    r"in the test set, but it also flags 1{,}550 non-dropouts as at-risk." "\n"
    r"The paired comparison between XGBoost and RandomForest yields an "
    r"AUC difference of 0.0145." "\n"
)


def _verifier() -> Verifier:
    v = Verifier.__new__(Verifier)
    v.ctx = types.SimpleNamespace(output_dir=".", log=[], errors=[])
    v.config = {}
    return v


def _finding(**over) -> dict:
    base = {
        "id": "F1",
        "dimension": "D2",
        "severity": "major",
        "location": "Results",
        "quote": "correctly identifies 341 of the 506 true dropout episodes",
        "problem": "The counts appear in no artifact.",
        "evidence": "confusion_matrix.png shows 370/545/1375.",
        "direction": "flatters",
    }
    base.update(over)
    return base


def _reason(f: dict, paper: str = PAPER, n_kept: int = 0):
    v = _verifier()
    return v._rejection_reason(f, _normalize_for_quote_match(paper), n_kept)


# ---------------------------------------------------------------------------
# the quote rule -- the one that cannot be argued with
# ---------------------------------------------------------------------------


def test_a_well_formed_finding_survives():
    assert _reason(_finding()) is None


def test_a_quote_that_is_not_in_the_paper_is_dropped():
    """Observed live, twice in one run.

    The judge paraphrased the comparator sentence instead of quoting it.
    A paraphrase cannot be opened and checked by a human, which is the
    whole contract.
    """
    f = _finding(
        quote="The paired cluster-bootstrap comparison between XGBoost and "
        "the next-best individual model"
    )
    assert _reason(f) == "quote is not a literal substring of the manuscript"


def test_quoting_survives_latex_wrapping_and_spacing():
    """4{,}312 in the source is 4,312 to a reader."""
    paper = r"The analyzed n was 4{,}312 students\\" "\n" r"across both groups."
    f = _finding(quote="The analyzed n was 4,312 students across both groups")
    assert _reason(f, paper=paper) is None


def test_an_empty_quote_is_dropped():
    assert _reason(_finding(quote="")) == "no quote"


def test_an_over_long_quote_is_dropped():
    f = _finding(quote=" ".join(["word"] * 60))
    assert "over the 40-word limit" in _reason(f)


def test_a_two_word_quote_is_too_short_to_locate():
    assert _reason(_finding(quote="the model")) == "quote too short to locate"


# ---------------------------------------------------------------------------
# contract rules
# ---------------------------------------------------------------------------


def test_no_evidence_no_finding():
    assert "no evidence" in _reason(_finding(evidence="  "))


def test_a_dimension_outside_the_rubric_is_dropped():
    assert "not in the rubric" in _reason(_finding(dimension="D9"))


def test_an_invented_severity_is_dropped():
    assert "not critical/major/minor" in _reason(_finding(severity="blocker"))


@pytest.mark.parametrize(
    "text",
    [
        "The contribution lacks novelty relative to prior work.",
        "They should have used a mixed-effects model here.",
        "The writing style is hard to follow.",
    ],
)
def test_opinions_are_out_of_contract(text):
    assert "out of contract" in _reason(_finding(problem=text))


def test_the_finding_cap_holds():
    assert f"over the {MAX_FINDINGS}-finding cap" in _reason(
        _finding(), n_kept=MAX_FINDINGS
    )


# ---------------------------------------------------------------------------
# the self-negating guard
# ---------------------------------------------------------------------------


def test_a_finding_that_concludes_the_values_agree_is_dropped():
    """Observed live.

    "The gap is stated as 0.094 ... and as '0.746 vs. 0.652', which is a
    difference of 0.094 - consistent." The model did the arithmetic,
    found agreement, and filed it as a finding anyway.
    """
    f = _finding(
        problem="The gap is stated as 0.094 and as 0.746 vs 0.652, which is "
        "a difference of 0.094 - consistent."
    )
    assert "concludes the values agree" in _reason(f)


@pytest.mark.parametrize(
    "text",
    [
        "The confusion-matrix counts are internally inconsistent with the recall.",
        "The values do not agree: 0.788 minus 0.444 is 0.344.",
        "The clustered CI does not match the artefact value.",
        "The reported precision is 0.576 while the positive class is 0.212.",
    ],
)
def test_a_real_discrepancy_is_not_read_as_self_negating(text):
    assert not _SELF_NEGATING.search(text), text


# ---------------------------------------------------------------------------
# parse and assemble
# ---------------------------------------------------------------------------


def test_a_fenced_response_parses_and_validates():
    v = _verifier()
    raw = (
        "Here is my analysis.\n```json\n"
        '{"findings": [' + _json(_finding()) + ", " + _json(_finding(dimension="D9")) + "]}"
        "\n```"
    )
    kept, dropped, err = v._parse_and_validate(raw, PAPER)
    assert err is None
    assert len(kept) == 1 and len(dropped) == 1
    assert kept[0]["source"] == "verifier"


def test_an_unparseable_response_reports_the_error_not_silence():
    v = _verifier()
    kept, dropped, err = v._parse_and_validate("I could not comply.", PAPER)
    assert kept == [] and err


def test_a_response_with_no_findings_list_is_an_error_not_a_clean_paper():
    v = _verifier()
    kept, dropped, err = v._parse_and_validate('{"summary": "looks fine"}', PAPER)
    assert kept == []
    assert err == "response had no findings list"


# ---------------------------------------------------------------------------
# the artifact digest
# ---------------------------------------------------------------------------


def test_self_certified_booleans_are_stripped_from_the_digest():
    """The run's own opinion of the run is not evidence about the run.

    One archived paper asserted "the estimand-match check passed for all
    five methods" from a field computed by comparing a string to itself.
    A judge handed that field inherits the assertion instead of checking
    it.
    """
    v = _verifier()
    stripped = v._strip_self_certified(
        {
            "causal_estimand_check": {"declared": "ATT", "match": True},
            "data": {"validation_passed": True, "analytic_n": 17335},
            "models": [{"auc": 0.73, "significant": True}],
        }
    )
    assert "match" not in stripped["causal_estimand_check"]
    assert "validation_passed" not in stripped["data"]
    assert "significant" not in stripped["models"][0]
    # and everything else survives
    assert stripped["causal_estimand_check"]["declared"] == "ATT"
    assert stripped["data"]["analytic_n"] == 17335
    assert stripped["models"][0]["auc"] == 0.73


def test_caption_is_found_after_the_graphic():
    cap = Verifier._caption_near(
        r"\includegraphics{love_plot.png}" "\n" r"\caption{Love Plot of "
        r"Covariate Balance}" "\n" r"\label{fig:love}",
        0,
    )
    assert cap == "Love Plot of Covariate Balance"


def test_caption_is_found_before_the_graphic():
    tex = r"\caption{Balance after matching}" "\n" r"\includegraphics{lp.png}"
    cap = Verifier._caption_near(tex, tex.index(r"\includegraphics"))
    assert cap == "Balance after matching"


def _json(obj) -> str:
    import json

    return json.dumps(obj)


def test_the_verifier_reads_from_source_dir_and_writes_to_output_dir(tmp_path):
    """A scoring pass must leave no marks on what it scores.

    The first version of the offline harness pointed ctx.output_dir at an
    archived run, so running it wrote verification_report.json,
    verification_raw.txt and a prompts/verifier/ directory INTO the
    evaluation archive -- the only surviving record of roughly seventy
    hours of adversarial verification, whose builder can no longer run.
    """
    source = tmp_path / "archived_run"
    source.mkdir()
    (source / "paper.tex").write_text(PAPER, encoding="utf-8")
    before = set(p.name for p in source.iterdir())

    dest = tmp_path / "scratch"
    dest.mkdir()

    v = Verifier.__new__(Verifier)
    v.ctx = types.SimpleNamespace(output_dir=str(dest), log=[], errors=[])
    v.config = {}
    v.source_dir = str(source)

    assert "341 of the 506" in v._read("paper.tex")
    v._persist({"ran": True, "findings": []}, "raw response")

    assert set(p.name for p in source.iterdir()) == before
    assert (dest / "verification_report.json").exists()
    assert (dest / "verification_raw.txt").exists()
