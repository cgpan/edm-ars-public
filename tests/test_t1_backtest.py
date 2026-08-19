"""Arc T V2 backtest - tests for scripts/backtest_ranker.py.

The backtest is the falsification gate for Arc T's ranker, so its
arithmetic has to be pinned independently of the archive it happens to
read today. Every correlation number asserted below is either
hand-computable or was cross-checked against ``scipy.stats.spearmanr``
(agreement to 1.1e-16 on all six cases); scipy is NOT a dependency of
the script or of these tests.

Offline by construction: no network, no LLM, no provider SDK. The
synthetic fixtures are built in-process or in ``tmp_path``.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import backtest_ranker as B  # noqa: E402

RUNS_DIR = REPO_ROOT / "runs"


# ==========================================================================
# Ranking and correlation arithmetic
# ==========================================================================


def test_rank_with_ties_is_one_based_and_averages_ties() -> None:
    assert B.rank_with_ties([10, 20, 30]) == [1.0, 2.0, 3.0]
    assert B.rank_with_ties([30, 10, 20]) == [3.0, 1.0, 2.0]
    # two-way tie at the bottom -> both get (1+2)/2
    assert B.rank_with_ties([1, 1, 2, 3]) == [1.5, 1.5, 3.0, 4.0]
    # three-way tie in the middle -> all get (2+3+4)/3
    assert B.rank_with_ties([1, 5, 5, 5, 9]) == [1.0, 3.0, 3.0, 3.0, 5.0]
    # all tied
    assert B.rank_with_ties([7, 7, 7]) == [2.0, 2.0, 2.0]


def test_pearson_hand_computation() -> None:
    # r = 1 for an exact positive linear relation, regardless of scale
    assert B.pearson([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert B.pearson([1, 2, 3], [30, 20, 10]) == pytest.approx(-1.0)
    # sum dxdy = 1, sum dx^2 = 2, sum dy^2 = 2  ->  r = 0.5
    assert B.pearson([1, 2, 3], [1, 3, 2]) == pytest.approx(0.5)


def test_pearson_is_nan_without_variance() -> None:
    assert math.isnan(B.pearson([1, 1, 1], [1, 2, 3]))
    assert math.isnan(B.pearson([1, 2, 3], [5, 5, 5]))
    assert math.isnan(B.pearson([1], [1]))


def test_spearman_perfect_orderings() -> None:
    assert B.spearman([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == pytest.approx(1.0)
    assert B.spearman([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) == pytest.approx(-1.0)
    # monotone but non-linear: Spearman is 1, Pearson is not
    assert B.spearman([1, 2, 3, 4], [1, 8, 27, 64]) == pytest.approx(1.0)


def test_spearman_known_value_no_ties() -> None:
    """rho = 1 - 6*sum(d^2)/(n(n^2-1)) = 1 - 24/120 = 0.8."""
    assert B.spearman([1, 2, 3, 4, 5], [2, 1, 4, 3, 5]) == pytest.approx(0.8)


def test_spearman_known_value_with_ties() -> None:
    """x ranks [1.5,1.5,3,4] vs y ranks [1,2,3,4].

    sum dxdy = 4.5, sum dx^2 = 4.5, sum dy^2 = 5.0
    rho = 4.5 / sqrt(22.5) = 0.9486832980505138
    """
    assert B.spearman([1, 1, 2, 3], [1, 2, 3, 4]) == pytest.approx(
        4.5 / math.sqrt(22.5)
    )


def test_spearman_matches_the_shape_of_the_real_primary_population() -> None:
    """The n=5 configuration the archive actually produces.

    x has two 2-way ties; rho = 7.5 / sqrt(9.0 * 10.0) = 0.7905694150420948.
    """
    x = [0.5, 1.5, 0.5, 1.5, 3.0]
    y = [6.2, 7.0, 6.6, 7.5, 7.3]
    assert B.spearman(x, y) == pytest.approx(7.5 / math.sqrt(90.0))
    assert B.spearman(x, y) == pytest.approx(0.7905694150420948)


def test_spearman_is_nan_when_a_margin_is_constant() -> None:
    assert math.isnan(B.spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))


# ==========================================================================
# Permutation test
# ==========================================================================


def test_permutation_exact_on_a_perfect_ordering() -> None:
    result = B.permutation_test([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    assert result.method == "exact"
    assert result.n_permutations == 120
    assert result.rho == pytest.approx(1.0)
    # exactly one of the 120 permutations reproduces rho = 1
    assert result.p_greater == pytest.approx(1 / 120)
    # rho = -1 is equally extreme, so two-sided counts 2 of 120
    assert result.p_two_sided == pytest.approx(2 / 120)
    assert result.min_attainable_p_greater == pytest.approx(1 / 120)
    assert result.seed is None


def test_permutation_exact_known_p_for_rho_0_8() -> None:
    """8 of the 120 permutations of [1..5] reach rho >= 0.8."""
    result = B.permutation_test([1, 2, 3, 4, 5], [2, 1, 4, 3, 5])
    assert result.rho == pytest.approx(0.8)
    assert result.p_greater == pytest.approx(8 / 120)
    assert result.p_two_sided == pytest.approx(16 / 120)


def test_permutation_min_attainable_p_rises_when_x_has_ties() -> None:
    """Ties cost power: the best achievable one-sided p goes 1/120 -> 4/120.

    This is the honest answer to "is n big enough" - with two 2-way ties
    in the deterministic score, four distinct permutations all realise the
    maximum rho, so no ordering of 5 papers can ever produce p < 0.033.
    """
    tied = B.permutation_test([0.5, 1.5, 0.5, 1.5, 3.0], [6.2, 7.0, 6.6, 7.5, 7.3])
    untied = B.permutation_test([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    assert tied.min_attainable_p_greater == pytest.approx(4 / 120)
    assert untied.min_attainable_p_greater == pytest.approx(1 / 120)
    assert tied.min_attainable_p_greater > untied.min_attainable_p_greater
    # and the observed configuration is not significant at 0.05
    assert tied.rho == pytest.approx(0.7905694150420948)
    assert tied.p_greater == pytest.approx(8 / 120)
    assert tied.p_greater > 0.05


def test_permutation_switches_to_seeded_monte_carlo_above_max_exact_n() -> None:
    x = list(range(12))
    y = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8]
    first = B.permutation_test(x, y, n_draws=2000, seed=42)
    second = B.permutation_test(x, y, n_draws=2000, seed=42)
    third = B.permutation_test(x, y, n_draws=2000, seed=7)
    assert first.method == "monte_carlo"
    assert first.seed == 42
    assert first.n_permutations == 2000
    assert first.p_greater == second.p_greater  # deterministic given the seed
    assert first.rho == pytest.approx(second.rho) == pytest.approx(third.rho)


def test_permutation_p_values_are_probabilities() -> None:
    result = B.permutation_test([1, 2, 3, 4, 5, 6], [6, 2, 4, 1, 5, 3])
    for value in (result.p_greater, result.p_two_sided,
                  result.min_attainable_p_greater):
        assert 0.0 < value <= 1.0


def test_permutation_is_undefined_without_variance() -> None:
    result = B.permutation_test([1, 1, 1, 1], [1, 2, 3, 4])
    assert result.method == "undefined"
    assert math.isnan(result.p_greater)


# ==========================================================================
# Bootstrap and power planning
# ==========================================================================


def test_bootstrap_ci_is_seed_deterministic_and_bounded() -> None:
    x = [0.5, 1.5, 0.5, 1.5, 3.0]
    y = [6.2, 7.0, 6.6, 7.5, 7.3]
    first = B.bootstrap_ci(x, y, n_boot=500, seed=42)
    second = B.bootstrap_ci(x, y, n_boot=500, seed=42)
    assert first == second
    assert -1.0 <= first["lo"] <= first["hi"] <= 1.0
    assert first["n_valid"] + first["n_degenerate"] == 500
    # at n=5 with ties, a real share of resamples collapse a margin
    assert first["n_degenerate"] > 0


def test_required_n_is_monotone_in_rho() -> None:
    ns = [B.required_n_for_rho(r) for r in (0.2, 0.35, 0.5, 0.7, 0.9)]
    assert ns == sorted(ns, reverse=True)
    assert all(n > 3 for n in ns)


def test_required_n_pins_the_planning_numbers_quoted_in_the_report() -> None:
    # Fisher-z, alpha=0.05 one-sided, 80% power, 1.06 Spearman inflation
    assert B.required_n_for_rho(0.5) == 25
    assert B.required_n_for_rho(0.7905694150420948) == 9
    assert B.required_n_for_rho(1.0) == -1  # undefined at the boundary
    assert B.required_n_for_rho(0.0) == -1


# ==========================================================================
# analyse() on synthetic fixtures with KNOWN correlations
# ==========================================================================


def _row(run: str, det: float, gate: float, *, penalty: float = 0.0,
         venue_fit: float | None = None) -> B.Row:
    return B.Row(
        run=run, spec_dir="output", canonical=True,
        dataset="synthetic", task_type="prediction", timestamp=None,
        feasibility_verdict="CLEAN", feasibility_penalty=penalty,
        venue_fit_default=(det if venue_fit is None else venue_fit),
        deterministic_score=det, gate_score=gate, gate_n_samples=3,
        gate_samples=[gate, gate, gate], gate_venue="EDM", in_ledger=True,
    )


def test_analyse_reports_rho_one_on_a_perfectly_ordered_fixture() -> None:
    rows = [_row(f"R{i}", det=float(i), gate=5.0 + i) for i in range(5)]
    out = B.analyse(rows, label="synthetic perfect")
    assert out["n"] == 5
    assert out["rho"] == pytest.approx(1.0)
    assert out["permutation"]["p_greater"] == pytest.approx(1 / 120)
    assert out["permutation"]["method"] == "exact"


def test_analyse_reports_rho_minus_one_on_an_inverted_fixture() -> None:
    rows = [_row(f"R{i}", det=float(i), gate=9.0 - i) for i in range(5)]
    out = B.analyse(rows, label="synthetic inverted")
    assert out["rho"] == pytest.approx(-1.0)
    # a perfectly inverted ranker cannot be rescued by the one-sided test
    assert out["permutation"]["p_greater"] == pytest.approx(1.0)


def test_analyse_reports_the_known_tied_rho() -> None:
    """Same shape as the real primary population: rho = 7.5/sqrt(90)."""
    pairs = [(0.5, 6.2), (1.5, 7.0), (0.5, 6.6), (1.5, 7.5), (3.0, 7.3)]
    rows = [_row(f"R{i}", det=d, gate=g) for i, (d, g) in enumerate(pairs)]
    out = B.analyse(rows, label="synthetic tied")
    assert out["rho"] == pytest.approx(0.7905694150420948)
    assert out["x_distinct_values"] == [0.5, 1.5, 3.0]
    assert out["permutation"]["p_greater"] == pytest.approx(8 / 120)


def test_analyse_reports_penalty_variance_so_a_dead_term_is_visible() -> None:
    rows = [_row(f"R{i}", det=float(i), gate=5.0 + i) for i in range(5)]
    out = B.analyse(rows, label="dead penalty term")
    assert out["penalty_distinct_values"] == [0.0]

    rows[0].feasibility_penalty = 1.5
    out2 = B.analyse(rows, label="live penalty term")
    assert out2["penalty_distinct_values"] == [0.0, 1.5]


def test_analyse_is_degenerate_when_the_deterministic_score_is_constant() -> None:
    rows = [_row(f"R{i}", det=1.0, gate=5.0 + i) for i in range(5)]
    out = B.analyse(rows, label="constant score")
    assert math.isnan(out["rho"])
    assert out["permutation"] is None
    assert "degenerate" in out["note"]


def test_analyse_can_score_the_venue_fit_term_alone() -> None:
    rows = [_row(f"R{i}", det=float(i), gate=5.0 + i, venue_fit=float(-i))
            for i in range(5)]
    composite = B.analyse(rows, label="composite")
    vf_only = B.analyse(rows, label="vf only", score_attr="venue_fit_default")
    assert composite["rho"] == pytest.approx(1.0)
    assert vf_only["rho"] == pytest.approx(-1.0)


# ==========================================================================
# Robustness diagnostics: leave-one-out and duplicate ideas
# ==========================================================================


def test_leave_one_out_drops_exactly_one_row_each_time() -> None:
    rows = [_row(f"R{i}", det=float(i), gate=5.0 + i) for i in range(5)]
    loo = B.leave_one_out(rows)
    assert [item["dropped"] for item in loo] == ["R0", "R1", "R2", "R3", "R4"]
    assert all(item["n"] == 4 for item in loo)
    assert all(item["rho"] == pytest.approx(1.0) for item in loo)


def test_leave_one_out_exposes_a_correlation_carried_by_one_row() -> None:
    """Four tied papers plus one outlier: drop the outlier and rho vanishes."""
    rows = [_row(f"R{i}", det=1.0, gate=6.0 + 0.1 * i) for i in range(4)]
    rows.append(_row("outlier", det=9.0, gate=9.0))
    loo = {item["dropped"]: item["rho"] for item in B.leave_one_out(rows)}
    assert math.isnan(loo["outlier"])  # x is constant once it is gone
    assert not math.isnan(loo["R0"])


def test_resolved_target_covers_every_task_type_shape() -> None:
    assert B.resolved_target({"outcome_variable": "X4EVRATNDCLG"}) == "X4EVRATNDCLG"
    assert B.resolved_target({"outcome": {"variable": "rank_base"}}) == "rank_base"
    assert B.resolved_target({"treatment": {"variable": "algebra1"}}) == "algebra1"
    assert B.resolved_target({"scale_name": "math self-efficacy"}) == (
        "math self-efficacy"
    )
    assert B.resolved_target({}) is None


def test_duplicate_idea_groups_flags_one_idea_run_twice() -> None:
    a = _row("phase_a", det=0.15, gate=6.2)
    b = _row("stream2", det=0.15, gate=6.6)
    c = _row("did_v2", det=0.45, gate=7.0)
    for row in (a, b):
        row.dataset, row.task_type, row.target = "els_2002", "prediction", "F2EVRATT"
    c.dataset, c.task_type, c.target = "did_panel", "causal_did", "rank_base"

    groups = B.duplicate_idea_groups([a, b, c])
    assert len(groups) == 1
    assert groups[0]["runs"] == ["phase_a", "stream2"]
    assert groups[0]["target"] == "F2EVRATT"
    assert groups[0]["gate_scores"] == [6.2, 6.6]


def test_duplicate_idea_groups_ignores_rows_without_a_target() -> None:
    a = _row("a", det=1.0, gate=7.0)
    b = _row("b", det=1.0, gate=7.0)
    a.target = b.target = None
    assert B.duplicate_idea_groups([a, b]) == []


# ==========================================================================
# Verdict logic - the falsification rules, applied mechanically
# ==========================================================================


def _perm(p: float, floor: float = 0.001) -> dict:
    return {"p_greater": p, "min_attainable_p_greater": floor}


def test_verdict_inverted_rho_is_falsified_and_exits_nonzero() -> None:
    verdict = B._verdict(-0.4, _perm(0.9), {"separated": True}, alpha=0.05, n=5)
    assert verdict["direction"] == "INVERTED"
    assert verdict["overall"] == "FALSIFIED_INVERTED"
    assert verdict["exit_code"] == 1
    assert "feasibility screen ALONE" in verdict["recommendation"]


def test_verdict_zero_rho_counts_as_falsified() -> None:
    """The spec's rule is rho <= 0, not rho < 0."""
    verdict = B._verdict(0.0, _perm(0.5), {"separated": True}, alpha=0.05, n=5)
    assert verdict["direction"] == "NULL"
    assert verdict["overall"] == "FALSIFIED_INVERTED"
    assert verdict["exit_code"] == 1


def test_verdict_unseparated_pair_is_falsified_even_with_positive_rho() -> None:
    verdict = B._verdict(0.9, _perm(0.01), {"separated": False}, alpha=0.05, n=12)
    assert verdict["overall"] == "FALSIFIED_PAIR"
    assert verdict["exit_code"] == 1


def test_verdict_positive_but_underpowered_does_not_fail_the_regression() -> None:
    verdict = B._verdict(0.79, _perm(0.0667, 0.0333), {"separated": True},
                         alpha=0.05, n=5)
    assert verdict["direction"] == "POSITIVE"
    assert verdict["power"] == "UNDERPOWERED"
    assert verdict["overall"] == "POSITIVE_BUT_UNDERPOWERED"
    assert verdict["exit_code"] == 0
    assert "Advisory use only" in verdict["recommendation"]


def test_verdict_flags_a_design_that_can_never_reach_alpha() -> None:
    verdict = B._verdict(0.6, _perm(0.20, 0.20), {"separated": True},
                         alpha=0.05, n=4)
    assert verdict["power"] == "IMPOSSIBLE_AT_THIS_N"
    assert verdict["exit_code"] == 0


def test_verdict_usable_when_the_ordering_beats_chance() -> None:
    verdict = B._verdict(0.8, _perm(0.01), {"separated": True}, alpha=0.05, n=20)
    assert verdict["overall"] == "USABLE"
    assert verdict["power"] == "SUFFICIENT"
    assert verdict["exit_code"] == 0


def test_verdict_not_computable_on_a_degenerate_population() -> None:
    verdict = B._verdict(float("nan"), {}, {"separated": None}, alpha=0.05, n=1)
    assert verdict["overall"] == "NOT_COMPUTABLE"
    assert verdict["exit_code"] == 2


# ==========================================================================
# Gate-outcome recovery - revision cycles vs sample cycles
# ==========================================================================


def _write_cycle(run_dir: Path, name: str, score: float,
                 venue: str = "EDM") -> None:
    cycle = run_dir / "output" / "lsar_review" / name
    cycle.mkdir(parents=True, exist_ok=True)
    (cycle / "scores.json").write_text(
        json.dumps({"overall_score": score, "recommendation": "x"}),
        encoding="utf-8",
    )
    (cycle / "venue_classification.json").write_text(
        json.dumps({"venue": venue}), encoding="utf-8"
    )


def _write_summary(run_dir: Path, cycles_used: int, final_score: float) -> None:
    (run_dir / "output" / "lsar_review").mkdir(parents=True, exist_ok=True)
    (run_dir / "output" / "lsar_review" / "gate_summary.json").write_text(
        json.dumps({"cycles_used": cycles_used, "final_score": final_score}),
        encoding="utf-8",
    )


def test_recover_gate_outcome_reads_a_median_of_three(tmp_path: Path) -> None:
    run = tmp_path / "run_a"
    _write_cycle(run, "cycle_1", 6.6)
    _write_cycle(run, "cycle_102", 7.2)
    _write_cycle(run, "cycle_103", 7.0)
    _write_summary(run, 1, 7.0)

    gate = B.recover_gate_outcome(run)
    assert gate.samples == [6.6, 7.2, 7.0]
    assert gate.cycles == ["cycle_1", "cycle_102", "cycle_103"]
    assert gate.score == pytest.approx(7.0)
    assert gate.score == gate.summary_final_score
    assert gate.venue == "EDM"


def test_recover_gate_outcome_uses_the_final_revision_cycle(tmp_path: Path) -> None:
    """cycle_1/2/3 are DIFFERENT manuscripts; the gate's verdict is the last.

    Reading cycle_1 here would report 6.5 for a run whose gate actually
    ended at 4.2 - the arc_p_validation_20260711 shape.
    """
    run = tmp_path / "run_b"
    _write_cycle(run, "cycle_1", 6.5)
    _write_cycle(run, "cycle_2", 3.1)
    _write_cycle(run, "cycle_3", 4.2)
    _write_summary(run, 3, 4.2)

    gate = B.recover_gate_outcome(run)
    assert gate.samples == [4.2]
    assert gate.cycles == ["cycle_3"]
    assert gate.final_cycle == 3
    assert gate.score == pytest.approx(4.2)


def test_recover_gate_outcome_pairs_samples_with_their_own_revision(
    tmp_path: Path,
) -> None:
    """Samples of revision 2 live at cycle_202/203, not cycle_102/103."""
    run = tmp_path / "run_c"
    _write_cycle(run, "cycle_1", 5.0)
    _write_cycle(run, "cycle_102", 5.2)
    _write_cycle(run, "cycle_2", 6.0)
    _write_cycle(run, "cycle_202", 6.4)
    _write_cycle(run, "cycle_203", 6.2)
    _write_summary(run, 2, 6.2)

    gate = B.recover_gate_outcome(run)
    assert gate.cycles == ["cycle_2", "cycle_202", "cycle_203"]
    assert gate.samples == [6.0, 6.4, 6.2]
    assert gate.score == pytest.approx(6.2)


def test_recover_gate_outcome_survives_a_missing_summary(tmp_path: Path) -> None:
    run = tmp_path / "run_d"
    _write_cycle(run, "cycle_1", 6.1)
    _write_cycle(run, "cycle_2", 6.9)
    gate = B.recover_gate_outcome(run)
    assert gate.final_cycle == 2
    assert gate.samples == [6.9]
    assert gate.summary_final_score is None


def test_recover_gate_outcome_on_a_run_with_no_review(tmp_path: Path) -> None:
    gate = B.recover_gate_outcome(tmp_path / "nothing_here")
    assert gate.samples == []
    assert gate.score is None
    assert gate.venue is None


# ==========================================================================
# Population selection
# ==========================================================================


def test_assign_population_keeps_a_clean_median_of_three_edm_ledger_run() -> None:
    row = _row("good", det=1.0, gate=7.0)
    B.assign_population([row])
    assert row.included is True
    assert row.exclusion_reasons == []


@pytest.mark.parametrize(
    "mutate, fragment",
    [
        (lambda r: setattr(r, "gate_n_samples", 1), "single-review"),
        (lambda r: setattr(r, "gate_n_samples", 0), "no LSAR gate score"),
        (lambda r: setattr(r, "in_ledger", False), "ledger.json"),
        (lambda r: setattr(r, "gate_venue", "JEDM"), "uncalibrated"),
        (lambda r: setattr(r, "canonical", False), "aborted-attempt"),
    ],
)
def test_assign_population_excludes_with_a_named_reason(mutate, fragment) -> None:
    row = _row("candidate", det=1.0, gate=7.0)
    mutate(row)
    B.assign_population([row])
    assert row.included is False
    assert any(fragment in reason for reason in row.exclusion_reasons), (
        row.exclusion_reasons
    )


def test_every_excluded_row_records_at_least_one_reason() -> None:
    rows = [
        _row("a", 1.0, 7.0),
        _row("b", 1.0, 7.0),
        _row("c", 1.0, 7.0),
    ]
    rows[1].gate_n_samples = 1
    rows[2].gate_venue = "JLA"
    B.assign_population(rows)
    for row in rows:
        assert row.included == (not row.exclusion_reasons)


# ==========================================================================
# Ledger join
# ==========================================================================


def test_load_ledger_keys_on_run_id_not_run_dir(tmp_path: Path) -> None:
    """Three ledger run_dir values carry a literal vertical tab from an
    unescaped ``\\v4`` (spec sec. 1.4), so run_dir is unusable as a key."""
    ledger = tmp_path / "ledger.json"
    ledger.write_text(
        json.dumps({"papers": [
            {"run_id": "v4_psy_paper1_20260708",
             "run_dir": "runs\v4_psy_paper1_20260708", "lsar_overall": 7.5},
            {"run_id": "no_dir_at_all", "lsar_overall": 6.0},
        ]}),
        encoding="utf-8",
    )
    loaded = B.load_ledger(ledger)
    assert set(loaded) == {"v4_psy_paper1_20260708", "no_dir_at_all"}
    assert "\v" in loaded["v4_psy_paper1_20260708"]["run_dir"]


def test_load_ledger_on_a_missing_file_is_empty(tmp_path: Path) -> None:
    assert B.load_ledger(tmp_path / "absent.json") == {}


# ==========================================================================
# Offline guarantee
# ==========================================================================


def test_backtest_script_contains_no_network_or_llm_call_sites() -> None:
    source = (SCRIPTS / "backtest_ranker.py").read_text(encoding="utf-8")
    for forbidden in ("requests", "urllib", "socket", "anthropic", "openai",
                      "call_llm", "subprocess"):
        assert forbidden not in source, forbidden


# ==========================================================================
# Real-archive invariants (skipped when the archive is not on this machine)
#
# Deliberately NOT pinning the archive's rho: a new gated run should move
# that number, and a test that forbids it would just get muted. What is
# pinned is the falsification surface - direction and the pre-registered
# pair - which is what the gate is actually about.
# ==========================================================================


@pytest.mark.skipif(
    not (RUNS_DIR / "stream1_did_v2_20260708" / "output"
         / "research_spec.json").exists(),
    reason="archived runs not present on this machine",
)
def test_real_archive_backtest_runs_and_is_not_inverted() -> None:
    result = B.run_backtest(RUNS_DIR, REPO_ROOT / "evaluation" / "ledger.json")
    primary = result["results"]["primary"]
    assert primary["n"] >= 5
    assert not math.isnan(primary["rho"])
    assert primary["rho"] > 0, (
        "V2 falsification condition met: the deterministic ranker is "
        "inverted. Ship the feasibility screen alone (spec sec. 6 V2)."
    )
    assert result["verdict"]["exit_code"] == 0


@pytest.mark.skipif(
    not (RUNS_DIR / "phase_b_did_20260704" / "output"
         / "research_spec.json").exists(),
    reason="archived DiD runs not present on this machine",
)
def test_real_archive_pre_registered_pair_is_separated() -> None:
    result = B.run_backtest(RUNS_DIR, REPO_ROOT / "evaluation" / "ledger.json")
    pair = result["pair"]
    assert pair["available"] is True
    assert pair["separated"] is True
    assert "VF-01" in pair["low"]["rules_fired"]
    assert "VF-04" in pair["high"]["rules_fired"]
    assert pair["low"]["venue_fit"] < pair["high"]["venue_fit"]


@pytest.mark.skipif(
    not (RUNS_DIR / "stream1_did_v2_20260708" / "output"
         / "research_spec.json").exists(),
    reason="archived runs not present on this machine",
)
def test_real_archive_ledger_agrees_with_the_on_disk_gate_scores() -> None:
    result = B.run_backtest(RUNS_DIR, REPO_ROOT / "evaluation" / "ledger.json")
    assert result["ledger_disk_disagreements"] == []
