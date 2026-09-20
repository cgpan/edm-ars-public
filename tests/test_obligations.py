"""Tests for the Writer-obligation channel.

The channel exists because the old one deleted its contents. Across 19
archived runs the Critic addressed 41 findings to the Writer -- a third
of everything it found -- and its own validator dropped every one before
the report was saved.
"""

from __future__ import annotations

import json

from src.obligations import (
    Obligation,
    derive_obligations,
    evaluate_obligations,
    render_for_writer,
    summarize,
)


# ---------------------------------------------------------------------------
# derivation
# ---------------------------------------------------------------------------


def test_explicit_critic_obligations_are_used_verbatim():
    rr = {
        "writer_obligations": [
            {
                "id": "WO01",
                "severity": "major",
                "instruction": "State the DIF n as 4,312.",
                "assert_kind": "tex_contains",
                "assert_values": ["4,312"],
            }
        ]
    }
    obs = derive_obligations(rr)
    assert len(obs) == 1
    assert obs[0].assert_kind == "tex_contains"
    assert obs[0].assert_values == ["4,312"]
    assert obs[0].source == "critic.writer_obligations"


def test_a_free_text_instruction_gets_its_numerals_as_the_test():
    rr = {
        "revision_instructions": {
            "Writer": "State the analyzed n for the X1SEX DIF as 4,312 "
            "(2,118 + 2,194), not 9,870."
        }
    }
    obs = derive_obligations(rr)
    assert len(obs) == 1
    assert obs[0].assert_kind == "tex_contains"
    assert "4,312" in obs[0].assert_values


def test_years_are_not_treated_as_required_values():
    """"Cite Chen (2007)" must not demand the string 2007 be present.

    A year binds to almost any manuscript, so requiring one makes the
    test pass for the wrong reason.
    """
    rr = {"revision_instructions": {"Writer": "Cite Chen (2007) properly."}}
    obs = derive_obligations(rr)
    assert obs[0].assert_values == []
    assert obs[0].assert_kind == "manual"


def test_a_removal_instruction_does_not_demand_its_numbers_appear():
    rr = {
        "revision_instructions": {
            "Writer": "Do not report the M2 estimate of 0.0148 as the primary result."
        }
    }
    obs = derive_obligations(rr)
    assert obs[0].assert_kind == "manual", obs[0].assert_values


def test_writer_targeted_issues_become_obligations():
    rr = {
        "analysis_review": {
            "issues": [
                {
                    "severity": "minor",
                    "target_agent": "Writer",
                    "recommendation": "Label the SES bins by quintile, not by "
                    "numeric range.",
                },
                {
                    "severity": "major",
                    "target_agent": "Analyst",
                    "recommendation": "Re-run matching.",
                },
            ]
        }
    }
    obs = derive_obligations(rr)
    assert len(obs) == 1
    assert "SES bins" in obs[0].instruction


def test_duplicate_instructions_collapse():
    text = "State the analyzed n as 4,312."
    rr = {
        "revision_instructions": {"Writer": text},
        "analysis_review": {
            "issues": [{"target_agent": "Writer", "recommendation": text}]
        },
    }
    assert len(derive_obligations(rr)) == 1


def test_no_review_report_yields_nothing():
    assert derive_obligations(None) == []
    assert derive_obligations({}) == []


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def _ob(values, kind="tex_contains"):
    return [
        Obligation(
            id="WO01",
            source="test",
            severity="major",
            instruction="x",
            assert_kind=kind,
            assert_values=values,
        )
    ]


def test_a_present_value_closes_the_obligation():
    out = evaluate_obligations(_ob(["4,312"]), r"The analyzed n was 4,312 students.")
    assert out[0].status == "closed"


def test_an_absent_value_leaves_it_open():
    out = evaluate_obligations(_ob(["4,312"]), r"The analyzed n was 11,540 students.")
    assert out[0].status == "open"
    assert "4,312" in out[0].detail


def test_latex_thin_space_does_not_hide_the_number():
    """LaTeX writes 4{,}312; a reader sees 4,312 and `in` sees neither."""
    out = evaluate_obligations(_ob(["4,312"]), r"n = 4{,}312 students")
    assert out[0].status == "closed"


def test_an_unpunctuated_number_still_counts():
    out = evaluate_obligations(_ob(["4,312"]), r"n = 4312 students")
    assert out[0].status == "closed"


def test_tex_absent_closes_when_the_value_is_gone():
    out = evaluate_obligations(_ob(["0.0148"], kind="tex_absent"), "no such number here")
    assert out[0].status == "closed"
    out = evaluate_obligations(_ob(["0.0148"], kind="tex_absent"), "the ATT was 0.0148")
    assert out[0].status == "open"


def test_an_untestable_obligation_reports_itself_as_unchecked():
    """Not a pass. A thing nobody checked, said out loud."""
    out = evaluate_obligations(_ob([], kind="manual"), "any paper at all")
    assert out[0].status == "unchecked"
    assert "requires a reader" in out[0].detail


def test_no_manuscript_is_unknown_not_closed():
    out = evaluate_obligations(_ob(["4,312"]), None)
    assert out[0].status == "unknown"


def test_summary_counts_and_rate():
    obs = evaluate_obligations(
        _ob(["4,312"]) + _ob(["11,540"]), "n = 11,540 and nothing else"
    )
    s = summarize(obs)
    assert s["n_obligations"] == 2
    assert s["by_status"] == {"open": 1, "closed": 1}
    assert s["compliance_rate"] == 0.5
    assert json.dumps(s)


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def test_the_writer_block_names_what_must_appear():
    block = render_for_writer(
        derive_obligations(
            {"revision_instructions": {"Writer": "State the DIF n as 4,312."}}
        )
    )
    assert "Obligations from the Critic" in block
    assert "4,312" in block
    assert "must contain" in block


def test_no_obligations_renders_nothing():
    assert render_for_writer([]) == ""


# ---------------------------------------------------------------------------
# the archived case this exists for
# ---------------------------------------------------------------------------


def test_the_critic_validator_no_longer_deletes_writer_instructions():
    """The receipt six archived runs carry must stop being generated."""
    from src.agents.critic import Critic

    class _Template:
        @staticmethod
        def get_agent_order():
            return ["ProblemFormulator", "DataEngineer", "Analyst"]

    critic = Critic.__new__(Critic)
    critic.task_template = _Template()
    report = critic._validate_review_report(
        {
            "overall_verdict": "PASS",
            "revision_instructions": {"Writer": "State the DIF n as 4,312."},
        }
    )
    assert report["revision_instructions"].get("Writer") == "State the DIF n as 4,312."
    assert not any(
        "unknown agent 'Writer'" in e
        for e in report.get("_validation_errors", [])
    )
