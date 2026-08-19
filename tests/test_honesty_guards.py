"""I1/I2/I3/I5 — deterministic honesty guards from the AERA_OPEN audit.

The routed AERA_OPEN run (runs/aera_open_routed_did_20260711) passed the
calibrated 7.3 gate at 7.5 while (a) the Writer had invented values for
every null input field, and (b) the mandatory SPEC section 4.5 UNVERIFIED
block was missing after a NaN serialization crash killed the REVISING
stage. Every defense involved was an LLM-obedience prompt rule. These
tests pin the deterministic replacements.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import patch

import pytest

from src.agents.analyst import _sanitize_nonfinite
from src.agents.writer import NOT_AVAILABLE_MARKER, Writer, _mark_null_values
from src.config import load_config
from src.context import PipelineContext
from src.manuscript_linter import (
    UNVERIFIED_BLOCK,
    UNVERIFIED_MARKER,
    LintDefect,
    LintReport,
    _ground_candidates,
    _matches,
    lint_manuscript,
    run_is_unverified,
)

CONFIG_PATH = str(Path(__file__).resolve().parent.parent / "config.yaml")


# ---------------------------------------------------------------------------
# I3 — NaN sanitization before results serialization
# ---------------------------------------------------------------------------


class TestSanitizeNonfinite:
    def test_nan_and_inf_become_none_with_paths(self) -> None:
        obj = {
            "a": float("nan"),
            "b": {"c": [1.0, float("inf"), {"d": float("-inf")}]},
            "ok": 2.5,
        }
        clean, paths = _sanitize_nonfinite(obj)
        assert clean["a"] is None
        assert clean["b"]["c"][1] is None
        assert clean["b"]["c"][2]["d"] is None
        assert clean["ok"] == 2.5
        assert set(paths) == {"$.a", "$.b.c[1]", "$.b.c[2].d"}

    def test_serializes_strictly_after_sanitization(self) -> None:
        """The exact crash: json.dumps(..., allow_nan=False) on a NaN.

        The AERA_OPEN run's follow-wave probe was NaN; the streaming
        json.dump raised mid-write, truncating results.json on disk."""
        obj = {"estimates": {"follow_wave_estimate": float("nan")}}
        clean, paths = _sanitize_nonfinite(obj)
        json.dumps(clean, allow_nan=False)  # must not raise
        assert paths == ["$.estimates.follow_wave_estimate"]

    def test_ints_bools_strings_untouched(self) -> None:
        obj = {"n": 16862, "flag": True, "s": "NaN", "f": 0.0}
        clean, paths = _sanitize_nonfinite(obj)
        assert clean == obj
        assert paths == []


# ---------------------------------------------------------------------------
# I1a — null values render as loud markers in the Writer prompt
# ---------------------------------------------------------------------------


class TestMarkNullValues:
    def test_nested_nulls_become_markers(self) -> None:
        obj = {"cell_means": None, "m10": {"cis": [None, 1.2]}}
        marked = _mark_null_values(obj)
        assert marked["cell_means"] == NOT_AVAILABLE_MARKER
        assert marked["m10"]["cis"][0] == NOT_AVAILABLE_MARKER
        assert marked["m10"]["cis"][1] == 1.2

    def test_original_not_mutated(self) -> None:
        obj = {"x": None}
        _mark_null_values(obj)
        assert obj["x"] is None

    def test_marker_reaches_both_prompt_builders(self, tmp_path: Path) -> None:
        config = load_config(CONFIG_PATH)
        ctx = PipelineContext(
            dataset_name="hsls09_public",
            raw_data_path="data/raw/none.csv",
            output_dir=str(tmp_path),
        )
        ctx.results_object = {"estimates": {"cell_means": None}}
        with patch("anthropic.Anthropic"):
            agent = Writer(ctx, "writer", config)
        msg_v1 = agent._build_user_message(
            research_spec={}, literature_context=None, data_report={},
            results_object=ctx.results_object, review_report={},
        )
        msg_v2 = agent._build_user_message_with_outline(
            outline={"sections": []}, research_spec={},
            literature_context=None, data_report={},
            results_object=ctx.results_object, review_report={},
        )
        assert NOT_AVAILABLE_MARKER in msg_v1
        assert NOT_AVAILABLE_MARKER in msg_v2
        assert '"cell_means": null' not in msg_v1


# ---------------------------------------------------------------------------
# I2 — deterministic UNVERIFIED block
# ---------------------------------------------------------------------------


def _writer(tmp_path: Path) -> Writer:
    config = load_config(CONFIG_PATH)
    ctx = PipelineContext(
        dataset_name="hsls09_public",
        raw_data_path="data/raw/none.csv",
        output_dir=str(tmp_path),
    )
    with patch("anthropic.Anthropic"):
        return Writer(ctx, "writer", config)


_TEX = (
    "\\documentclass{article}\n\\begin{document}\n"
    "\\section{Introduction}\nBody prose.\n\\end{document}\n"
)


class TestRunIsUnverified:
    def test_pass_verdict_is_verified(self) -> None:
        assert not run_is_unverified({"overall_verdict": "PASS"})

    def test_revise_verdict_is_unverified(self) -> None:
        assert run_is_unverified({"overall_verdict": "REVISE"})

    def test_explicit_flag_wins_even_on_pass(self) -> None:
        """The orchestrator sets unverified=True on the REVISING-crash
        fallback even when a later evaluator would say PASS."""
        assert run_is_unverified({"overall_verdict": "PASS", "unverified": True})

    def test_none_and_empty_are_verified(self) -> None:
        assert not run_is_unverified(None)
        assert not run_is_unverified({})


class TestInjectUnverifiedFlag:
    def test_flagged_run_gets_block_before_first_section(self, tmp_path: Path) -> None:
        w = _writer(tmp_path)
        out = w._inject_unverified_flag(_TEX, {"overall_verdict": "REVISE"})
        assert UNVERIFIED_MARKER in out
        assert out.index(UNVERIFIED_MARKER) < out.index("\\section{Introduction}")
        assert "Appendix: Automated Critic Review Report" in out
        assert out.index("Appendix:") < out.index("\\end{document}")

    def test_unflagged_run_untouched(self, tmp_path: Path) -> None:
        w = _writer(tmp_path)
        assert w._inject_unverified_flag(_TEX, {"overall_verdict": "PASS"}) == _TEX

    def test_idempotent(self, tmp_path: Path) -> None:
        w = _writer(tmp_path)
        review = {"overall_verdict": "REVISE"}
        once = w._inject_unverified_flag(_TEX, review)
        twice = w._inject_unverified_flag(once, review)
        assert twice == once
        assert twice.count(UNVERIFIED_MARKER.split(".")[0].split(":")[0]) >= 1

    def test_no_section_falls_back_to_begin_document(self, tmp_path: Path) -> None:
        w = _writer(tmp_path)
        tex = "\\documentclass{article}\n\\begin{document}\nProse.\n\\end{document}\n"
        out = w._inject_unverified_flag(tex, {"unverified": True})
        assert out.index("\\begin{document}") < out.index(UNVERIFIED_MARKER)
        assert UNVERIFIED_MARKER in out

    def test_block_constant_contains_marker(self) -> None:
        """Writer injects UNVERIFIED_BLOCK; linter greps UNVERIFIED_MARKER.
        The contract is that one contains the other."""
        assert UNVERIFIED_MARKER in UNVERIFIED_BLOCK


class TestLinterUnverifiedCheck:
    def _run_dir(self, tmp_path: Path, tex: str, review: dict) -> Path:
        (tmp_path / "paper.tex").write_text(tex, encoding="utf-8")
        (tmp_path / "checkpoint.json").write_text(
            json.dumps({"review_report": review}), encoding="utf-8"
        )
        return tmp_path

    def test_flagged_run_without_block_errors(self, tmp_path: Path) -> None:
        d = self._run_dir(tmp_path, _TEX, {"overall_verdict": "REVISE"})
        report = lint_manuscript(d, write_json=False)
        assert any(x.code == "unverified-block-missing" for x in report.errors)

    def test_flagged_run_with_block_is_clean(self, tmp_path: Path) -> None:
        tex = _TEX.replace("\\section{Introduction}", UNVERIFIED_BLOCK + "\\section{Introduction}")
        d = self._run_dir(tmp_path, tex, {"overall_verdict": "REVISE", "unverified": True})
        report = lint_manuscript(d, write_json=False)
        assert not any(x.code == "unverified-block-missing" for x in report.defects)

    def test_pass_verdict_never_requires_block(self, tmp_path: Path) -> None:
        d = self._run_dir(tmp_path, _TEX, {"overall_verdict": "PASS"})
        report = lint_manuscript(d, write_json=False)
        assert not any(x.code == "unverified-block-missing" for x in report.defects)

    def test_no_checkpoint_skips_check(self, tmp_path: Path) -> None:
        (tmp_path / "paper.tex").write_text(_TEX, encoding="utf-8")
        report = lint_manuscript(tmp_path, write_json=False)
        assert report.metrics.get("unverified_flag_checked") is False


# ---------------------------------------------------------------------------
# I1b — numeric reconciliation
# ---------------------------------------------------------------------------


_RESULTS = {
    "estimates": {
        "point_estimate": -1.904982835,
        "ci_lower": -3.456958661,
        "ci_upper": -0.376760808,
        "cell_means": {
            "pre_group0": 63.8015535,
            "post_group0": 62.3351290,
            "pre_group1": 33.5613964,
            "post_group1": 30.1899891,
        },
        "cell_ns": {"a": 4301, "b": 5519, "c": 3608, "d": 3434},
    },
    "share": 0.205,
}


def _reconcile_dir(tmp_path: Path, tex: str) -> Path:
    (tmp_path / "paper.tex").write_text(tex, encoding="utf-8")
    (tmp_path / "results.json").write_text(json.dumps(_RESULTS), encoding="utf-8")
    return tmp_path


def _doc(body: str) -> str:
    return (
        "\\documentclass{article}\n\\begin{document}\n"
        "\\section{Results}\n" + body + "\n\\end{document}\n"
    )


class TestNumericReconciliation:
    def test_honest_table_is_clean(self, tmp_path: Path) -> None:
        body = (
            "\\begin{tabular}{lrr}\n"
            "Pre & 63.80 & 33.56\\\\\nPost & 62.34 & 30.19\\\\\n"
            "N & 4,301 & 3,608\\\\\n"
            "\\end{tabular}\n"
        )
        report = lint_manuscript(_reconcile_dir(tmp_path, _doc(body)), write_json=False)
        assert not any("unreconciled" in x.code for x in report.defects)

    def test_marginal_sums_and_contrasts_are_legitimate(self, tmp_path: Path) -> None:
        """9,820 = 4,301 + 5,519 (sibling sum); -1.47 = 62.34 - 63.80
        rounds from the sibling difference. The first version of this
        check flagged the routed run's honest tab:cell_counts marginals."""
        body = (
            "\\begin{tabular}{lr}\nRow total & 9,820\\\\\n"
            "Change & -1.47\\\\\n\\end{tabular}\n"
        )
        report = lint_manuscript(_reconcile_dir(tmp_path, _doc(body)), write_json=False)
        assert not any("unreconciled" in x.code for x in report.defects)

    def test_fabricated_table_errors(self, tmp_path: Path) -> None:
        """The tab:m8_2x2 signature: mostly-invented cell values."""
        body = (
            "\\begin{tabular}{lrr}\n"
            "High & 56.19 & 53.77\\\\\nLow & 44.31 & 43.79\\\\\n"
            "Gap & 11.88 & 9.98\\\\\n\\end{tabular}\n"
        )
        report = lint_manuscript(_reconcile_dir(tmp_path, _doc(body)), write_json=False)
        errs = [x for x in report.errors if x.code == "unreconciled-table-numerals"]
        assert errs and "56.19" in errs[0].message

    def test_fully_invented_ci_in_prose_errors(self, tmp_path: Path) -> None:
        body = "The follow-wave contrast is $-3.42$ (95\\% CI $[-14.19, +7.36]$)."
        report = lint_manuscript(_reconcile_dir(tmp_path, _doc(body)), write_json=False)
        assert any(x.code == "unreconciled-ci-interval" for x in report.errors)

    def test_honest_ci_in_prose_is_clean(self, tmp_path: Path) -> None:
        body = "The DiD estimate is $-1.90$ (95\\% CI $[-3.46, -0.38]$)."
        report = lint_manuscript(_reconcile_dir(tmp_path, _doc(body)), write_json=False)
        assert not any("unreconciled" in x.code for x in report.defects)

    def test_half_matched_ci_warns_not_errors(self, tmp_path: Path) -> None:
        """The fabricated follow-wave CI [-2.61, +0.93] half-escaped:
        one endpoint collided with an unrelated value at print
        tolerance. One-miss intervals warn."""
        body = "Probe CI $[-3.46, +99.77]$."
        report = lint_manuscript(_reconcile_dir(tmp_path, _doc(body)), write_json=False)
        hits = [x for x in report.defects if x.code == "unreconciled-ci-interval"]
        assert hits and hits[0].severity == "warn"

    def test_percentage_variant_matches(self, tmp_path: Path) -> None:
        body = "\\begin{tabular}{lr}\nShare & 20.5 & 33.56\\\\\nX & 62.34 & 63.80\\\\\n\\end{tabular}\n"
        report = lint_manuscript(_reconcile_dir(tmp_path, _doc(body)), write_json=False)
        assert not any("unreconciled" in x.code for x in report.defects)

    def test_years_and_small_ints_exempt(self, tmp_path: Path) -> None:
        body = "\\begin{tabular}{lr}\nCohort & 2002 & 2009\\\\\nWave & 1 & 2\\\\\n\\end{tabular}\n"
        report = lint_manuscript(_reconcile_dir(tmp_path, _doc(body)), write_json=False)
        assert not any("unreconciled" in x.code for x in report.defects)

    def test_truncated_results_json_still_contributes(self, tmp_path: Path) -> None:
        """The AERA_OPEN run's results.json was invalid JSON (truncated
        mid-serialization); the salvage regex must still recover its
        numerals as ground truth."""
        (tmp_path / "paper.tex").write_text(
            _doc("\\begin{tabular}{lr}\nA & 63.80 & 33.56\\\\\nB & 62.34 & 30.19\\\\\n\\end{tabular}\n"),
            encoding="utf-8",
        )
        valid = json.dumps(_RESULTS)
        (tmp_path / "results.json").write_text(
            valid[: len(valid) // 2], encoding="utf-8"  # truncated → invalid
        )
        report = lint_manuscript(tmp_path, write_json=False)
        assert report.metrics.get("numeric_reconciliation_checked") is True

    def test_no_artifacts_skips_check(self, tmp_path: Path) -> None:
        (tmp_path / "paper.tex").write_text(_doc("Prose 12.34."), encoding="utf-8")
        report = lint_manuscript(tmp_path, write_json=False)
        assert report.metrics.get("numeric_reconciliation_checked") is False
        assert not any("unreconciled" in x.code for x in report.defects)

    def test_match_precision_respects_printed_decimals(self, tmp_path: Path) -> None:
        cand, _ = _ground_candidates(_reconcile_dir(tmp_path, _doc("")))
        assert _matches(cand, "-1.90")     # rounds from -1.904982835
        assert _matches(cand, "-1.905")
        assert not _matches(cand, "-1.92")  # off by more than print rounding


# ---------------------------------------------------------------------------
# I1/I2 gate blocking + I5 provenance
# ---------------------------------------------------------------------------


def _gate(tmp_path: Path):
    from src.review_gate import ReviewGate

    cfg = {
        "llm_provider": "deepseek",
        "deepseek": {"models": {"revision_writer": "deepseek-v4-pro"}},
        "review_gate": {"venue": "JEDM", "revision_model": "deepseek-v4-pro"},
    }
    return ReviewGate(cfg, str(tmp_path), log_fn=lambda *_: None)


class TestHonestyBlockers:
    def test_error_severity_honesty_codes_block(self, tmp_path: Path) -> None:
        gate = _gate(tmp_path)
        report = LintReport()
        report.add("error", "unreconciled-table-numerals", "tab:x fabricated")
        report.add("error", "no-foundational-references", "old refs missing")
        gate._last_lint = report
        blockers = gate._honesty_blockers()
        assert len(blockers) == 1
        assert blockers[0].startswith("unreconciled-table-numerals")

    def test_warn_severity_never_blocks(self, tmp_path: Path) -> None:
        gate = _gate(tmp_path)
        report = LintReport()
        report.add("warn", "unreconciled-ci-interval", "one endpoint off")
        gate._last_lint = report
        assert gate._honesty_blockers() == []

    def test_no_lint_never_blocks(self, tmp_path: Path) -> None:
        gate = _gate(tmp_path)
        gate._last_lint = None
        assert gate._honesty_blockers() == []

    def test_blocking_codes_are_the_audit_trio(self, tmp_path: Path) -> None:
        gate = _gate(tmp_path)
        assert gate.HONESTY_BLOCKING_CODES == {
            "unreconciled-table-numerals",
            "unreconciled-ci-interval",
            "unverified-block-missing",
        }


class TestMedianSampleProvenance:
    def test_gated_sample_dir_recorded(self, tmp_path: Path) -> None:
        """I5: gate_summary attributed the median 7.5 to cycle_1 (whose
        on-disk report said 7.2); the gated sample dir is now explicit."""
        gate = _gate(tmp_path)
        gate.pass_threshold = 7.3
        reports = {
            1: {"scores": {"overall_score": 7.2}},
            102: {"scores": {"overall_score": 7.5}},
            103: {"scores": {"overall_score": 7.5}},
        }
        gate.run_lsar = lambda pdf, cycle: reports[cycle]  # type: ignore[method-assign]
        gate.config.setdefault("review_gate", {})["median_samples"] = 3
        out = gate._maybe_median_sample(reports[1], tmp_path / "p.pdf", cycle=1)
        ms = out["scores"]["median_sampling"]
        assert ms["all_scores"] == [7.2, 7.5, 7.5]
        assert ms["gated_sample_dir"] in ("cycle_102", "cycle_103")
        assert out["scores"]["overall_score"] == 7.5


# ---------------------------------------------------------------------------
# Adversarial-review regressions (2026-08-06 diff review)
# ---------------------------------------------------------------------------


class TestReviewRegressions:
    def test_digit_grouping_not_fragmented(self, tmp_path: Path) -> None:
        """CRITICAL review finding: 23{,}503 tokenized as exempt '23' +
        guaranteed-miss '503' — an honest sample-size table gate-blocked."""
        body = (
            "\begin{tabular}{lr}\nN & 4{,}301 & 5{,}519\\\n"
            "M & 3{,}608 & 3{,}434\\\n\end{tabular}\n"
        )
        report = lint_manuscript(_reconcile_dir(tmp_path, _doc(body)), write_json=False)
        assert not any("unreconciled" in x.code for x in report.defects)

    def test_column_spec_dimensions_not_numerals(self, tmp_path: Path) -> None:
        """MAJOR review finding: p{2.5cm}/p{0.24\columnwidth} widths were
        checked as data and could gate-block a mostly-textual table."""
        body = (
            "\begin{tabular}{l p{2.5cm} p{0.24\columnwidth} p{1.5cm}}\n"
            "Item & text one & text two & text three\\\n"
            "\rule{0pt}{2.5ex}More & a & b & c\\\n\end{tabular}\n"
        )
        report = lint_manuscript(_reconcile_dir(tmp_path, _doc(body)), write_json=False)
        assert not any("unreconciled" in x.code for x in report.defects)

    def test_scientific_notation_skipped(self, tmp_path: Path) -> None:
        """MINOR review finding: 1.2e-3 / 1.2 \times 10^{-3} misparsed
        to the mantissa and counted as misses. Sci-notation is fail-open."""
        body = (
            "\begin{tabular}{lr}\np & $1.2 \times 10^{-3}$\\\n"
            "q & 3.1e-05\\\nr & 7.7e-04\\\n\end{tabular}\n"
        )
        report = lint_manuscript(_reconcile_dir(tmp_path, _doc(body)), write_json=False)
        assert not any("unreconciled" in x.code for x in report.defects)

    def test_csv_values_are_ground_truth(self, tmp_path: Path) -> None:
        """Residual-risk closure: tables rendered from Analyst CSVs
        (values absent from results.json) must not flag."""
        (tmp_path / "model_comparison.csv").write_text(
            "model,auc\nLR,0.7123\nRF,0.7456\nXGB,0.7789\n", encoding="utf-8"
        )
        body = (
            "\begin{tabular}{lr}\nLR & 0.7123\\\nRF & 0.7456\\\n"
            "XGB & 0.7789\\\n\end{tabular}\n"
        )
        report = lint_manuscript(_reconcile_dir(tmp_path, _doc(body)), write_json=False)
        assert not any("unreconciled" in x.code for x in report.defects)

    def test_explicit_unverified_false_wins_over_revise_verdict(self) -> None:
        """MAJOR review finding (evaluator-override path): raw LLM verdict
        'REVISE' + evaluator-computed effective PASS writes unverified:
        False — the block must NOT be stamped."""
        assert not run_is_unverified(
            {"overall_verdict": "REVISE", "unverified": False}
        )

    def test_sanitize_handles_numpy_scalars_and_bad_keys(self) -> None:
        """MINOR review finding: np.float32 NaN and non-finite float dict
        keys still crashed strict serialization."""
        np = pytest.importorskip("numpy")
        obj = {
            "a": np.float32("nan"),
            "b": np.float64(1.5),
            float("nan"): "keyed",
        }
        clean, paths = _sanitize_nonfinite(obj)
        json.dumps(clean, allow_nan=False)  # must not raise
        assert clean["a"] is None
        assert clean["b"] == 1.5
        assert "non-finite-key" in clean

    def test_prepare_pdf_resets_stale_lint(self, tmp_path: Path) -> None:
        """MAJOR review finding: a compile-failure fallback kept the
        PREVIOUS cycle's lint as a gate-blocking input."""
        gate = _gate(tmp_path)
        stale = LintReport()
        stale.add("error", "unreconciled-table-numerals", "from cycle 1")
        gate._last_lint = stale
        gate._compile_review_tex = lambda run_dir, name: False  # type: ignore[method-assign]
        gate._build_review_tex = lambda run_dir: None  # type: ignore[method-assign]
        try:
            gate.prepare_pdf(tmp_path, cycle=2)
        except Exception:
            pass
        assert gate._last_lint is not stale


# ---------------------------------------------------------------------------
# J1 consumer-side guard: the gate verifies the review it received
# ---------------------------------------------------------------------------


class TestGateRejectsDegenerateReviewSamples:
    """LSAR scored reviews whose generation was cut off at the
    Strengths/Weaknesses boundary; its scorer reads only those lists, so
    the paper was graded on praise alone (+0.83). cycle_102 of the
    routed AERA_OPEN run passed at 7.5 on a review with zero of both."""

    def _report(self, overall: float, review: dict | None) -> dict:
        r = {
            "scores": {
                "overall_score": overall,
                "recommendation": "Accept",
                "dimensions": [{"name": "Relevance", "score": 8}],
            }
        }
        if review is not None:
            r["review"] = review
        return r

    def test_healthy_review_passes(self, tmp_path: Path) -> None:
        gate = _gate(tmp_path)
        gate.pass_threshold = 5.0
        passed, diag = gate.evaluate_gate(
            self._report(7.5, {"strengths": ["a"], "weaknesses": ["b"]})
        )
        assert passed is True
        assert diag["review_health_problem"] is None

    def test_review_with_no_weaknesses_cannot_pass(self, tmp_path: Path) -> None:
        gate = _gate(tmp_path)
        gate.pass_threshold = 5.0
        passed, diag = gate.evaluate_gate(
            self._report(7.5, {"strengths": ["a", "b"], "weaknesses": []})
        )
        assert passed is False
        assert "weaknesses" in diag["review_health_problem"]

    def test_fully_empty_review_cannot_pass(self, tmp_path: Path) -> None:
        """The exact cycle_102 shape: 0 strengths, 0 weaknesses, 7.5."""
        gate = _gate(tmp_path)
        gate.pass_threshold = 7.3
        passed, diag = gate.evaluate_gate(
            self._report(7.5, {"strengths": [], "weaknesses": []})
        )
        assert passed is False
        assert "strengths" in diag["review_health_problem"]

    def test_advisory_mode_cannot_rescue_a_degenerate_review(
        self, tmp_path: Path
    ) -> None:
        """Advisory means the VENUE THRESHOLD is untrusted; a degenerate
        review means the SCORE is untrusted. The second is not
        rescuable by the first."""
        gate = _gate(tmp_path)
        gate.advisory_mode = True
        gate.pass_threshold = 5.0
        passed, _ = gate.evaluate_gate(
            self._report(7.5, {"strengths": [], "weaknesses": []})
        )
        assert passed is False
        # sanity: advisory still force-passes a healthy low score
        passed_ok, _ = gate.evaluate_gate(
            self._report(1.0, {"strengths": ["a"], "weaknesses": ["b"]})
        )
        assert passed_ok is True

    def test_report_without_a_review_block_is_unknown_not_broken(
        self, tmp_path: Path
    ) -> None:
        """Older LSAR reports do not persist review sections. Treating
        absence as emptiness made an audit flag every historical run."""
        gate = _gate(tmp_path)
        gate.pass_threshold = 5.0
        passed, diag = gate.evaluate_gate(self._report(7.5, None))
        assert passed is True
        assert diag["review_health_problem"] is None

    def test_review_block_without_the_keys_is_unknown(self, tmp_path: Path) -> None:
        gate = _gate(tmp_path)
        gate.pass_threshold = 5.0
        passed, diag = gate.evaluate_gate(
            self._report(7.5, {"paper_summary": "prose only"})
        )
        assert passed is True
        assert diag["review_health_problem"] is None
