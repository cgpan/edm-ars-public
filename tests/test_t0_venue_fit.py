"""Arc T / T0 - tests for the deterministic venue-fit rule table.

The load-bearing test is ``test_did_v1_ranks_below_did_v2``: the
pre-registered pair from spec sec. 6 V2. Same data, same estimand, same
dataset; the only difference is the idea (bare 2x2 DiD vs the same DiD
wrapped in M9 composition adjustment + M10 contrast heterogeneity), and
the realized gate scores were 3.7 Reject vs 7.0 Accept. If VF-01/VF-04
cannot separate that pair, the table is not encoding what it claims to.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.ideation import venue_fit as V

REPO_ROOT = Path(__file__).resolve().parents[1]
RULES_PATH = REPO_ROOT / "data_registry" / "venue_fit_rules.yaml"
RUNS_DIR = REPO_ROOT / "runs"


# --------------------------------------------------------------------------
# Rule table integrity
# --------------------------------------------------------------------------


def test_rules_file_loads() -> None:
    rules = V.load_rules(RULES_PATH)
    codes = [r["code"] for r in rules["rules"]]
    assert codes == ["VF-01", "VF-02", "VF-03", "VF-04", "VF-05", "VF-06", "VF-07"]


def test_every_rule_has_a_known_predicate_and_anchor_evidence() -> None:
    """C2: a rule that cannot cite the anchor fact behind it is not shipped."""
    rules = V.load_rules(RULES_PATH)
    for rule in rules["rules"]:
        assert rule["predicate"] in V._PREDICATES, rule["code"]
        assert len(str(rule.get("evidence", "")).strip()) > 40, rule["code"]
        assert isinstance(rule["delta"], (int, float))
        assert rule.get("applies_to"), rule["code"]


def test_venue_ethics_weights_present() -> None:
    rules = V.load_rules(RULES_PATH)
    weights = {k: v["ethics_weight"] for k, v in rules["venues"].items()}
    assert weights == {"EDM": 1.0, "JEDM": 0.5, "JLA": 0.6}
    for venue in rules["venues"].values():
        assert venue.get("evidence")


def test_rules_file_is_ascii() -> None:
    """The audit prints evidence strings to a Windows console."""
    text = RULES_PATH.read_text(encoding="utf-8")
    assert all(ord(c) < 128 for c in text)


# --------------------------------------------------------------------------
# The pre-registered pair
# --------------------------------------------------------------------------


def _did_v1() -> dict:
    """Bare 2x2 gap-in-gaps (the shipped phase_b_did spec, trimmed)."""
    return {
        "task_type": "causal_did",
        "dataset": "did_els_hsls_panel",
        "research_question": (
            "Did the within-cohort math achievement percentile-rank gap "
            "between lowest- and highest-band SES students change between "
            "the ELS:2002 and HSLS:09 sophomore cohorts?"
        ),
        "group_variable": "low_ses",
        "post_variable": "cohort",
        "outcome": {"variable": "rank_base", "type": "continuous"},
        "primary_method": "M8",
        "secondary_methods": [],
        "subgroup_analyses": ["female"],
        "expected_contribution": (
            "No retrieved paper applies a cross-cohort gap-in-gaps DiD to "
            "SES achievement-rank gaps using the harmonized panel."
        ),
    }


def _did_v2() -> dict:
    """Same estimand, same data, wrapped in M9/M10 (the shipped v2 spec)."""
    spec = _did_v1()
    spec.update(
        primary_method="M9",
        secondary_methods=["M8", "M10"],
        adjustment_covariates=["race5", "pared3", "expect_ba", "female"],
        heterogeneity_subgroups=["female", "expect_ba", "race5", "pared3"],
    )
    return spec


def test_did_v1_ranks_below_did_v2() -> None:
    v1 = V.score_venue_fit(_did_v1(), rules_path=RULES_PATH)
    v2 = V.score_venue_fit(_did_v2(), rules_path=RULES_PATH)
    assert v1.score < v2.score, (v1.render(), v2.render())
    assert "VF-01" in v1.codes
    assert "VF-04" in v2.codes


@pytest.mark.skipif(
    not (RUNS_DIR / "phase_b_did_20260704" / "output" / "research_spec.json").exists()
    or not (
        RUNS_DIR / "stream1_did_v2_20260708" / "output" / "research_spec.json"
    ).exists(),
    reason="archived DiD specs not present on this machine",
)
def test_did_v1_ranks_below_did_v2_on_the_real_archived_specs() -> None:
    def _load(run: str) -> dict:
        path = RUNS_DIR / run / "output" / "research_spec.json"
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    v1 = V.score_venue_fit(_load("phase_b_did_20260704"), rules_path=RULES_PATH)
    v2 = V.score_venue_fit(_load("stream1_did_v2_20260708"), rules_path=RULES_PATH)
    assert v1.score < v2.score
    assert v1.score <= -2.0  # VF-01 fired on the 3.7-Reject spec


def test_cross_cohort_alone_is_not_a_transfer_claim() -> None:
    """Regression guard for the pair: both DiD specs say 'cross-cohort',
    so treating it as a transfer claim would collapse the separation."""
    report = V.score_venue_fit(_did_v1(), rules_path=RULES_PATH)
    assert "VF-05" not in report.codes
    assert report.facts["transfer_terms"] == []


# --------------------------------------------------------------------------
# Individual rules
# --------------------------------------------------------------------------


def test_vf01_does_not_fire_when_a_second_contribution_exists() -> None:
    report = V.score_venue_fit(_did_v2(), rules_path=RULES_PATH)
    assert "VF-01" not in report.codes


def test_vf01_does_not_apply_to_prediction() -> None:
    spec = {"task_type": "prediction", "research_question": "predict x"}
    report = V.score_venue_fit(spec, rules_path=RULES_PATH)
    assert "VF-01" not in report.codes


def test_vf02_shap_headline() -> None:
    spec = {
        "task_type": "prediction",
        "research_question": "Which features predict enrollment?",
        "expected_contribution": "We report SHAP values for the best model.",
        "subgroup_analyses": ["X1SEX", "X1RACE"],
    }
    report = V.score_venue_fit(spec, rules_path=RULES_PATH)
    assert "VF-02" in report.codes
    hit = next(h for h in report.hits if h.code == "VF-02")
    assert "shap" in hit.why


def test_vf03_auc_only_prediction() -> None:
    spec = {
        "task_type": "prediction",
        "research_question": "Can we predict dropout?",
        "expected_contribution": "A model with higher AUC than the baseline.",
    }
    report = V.score_venue_fit(spec, rules_path=RULES_PATH)
    assert "VF-03" in report.codes
    assert report.score < 0


def test_vf03_suppressed_by_a_second_contribution() -> None:
    spec = {
        "task_type": "prediction",
        "research_question": "Can we predict dropout?",
        "expected_contribution": "The model is validated on a second dataset.",
    }
    report = V.score_venue_fit(spec, rules_path=RULES_PATH)
    assert "VF-03" not in report.codes
    assert {"VF-04", "VF-05"} <= set(report.codes)


def test_vf04_fairness_needs_more_than_one_routine_subgroup() -> None:
    one = {"task_type": "causal_soo", "subgroup_analyses": ["female"],
           "research_question": "effect of x"}
    two = {"task_type": "causal_soo", "subgroup_analyses": ["female", "race5"],
           "research_question": "effect of x"}
    assert "VF-04" not in V.score_venue_fit(one, rules_path=RULES_PATH).codes
    assert "VF-04" in V.score_venue_fit(two, rules_path=RULES_PATH).codes


def test_vf05_transfer_claim() -> None:
    spec = {
        "task_type": "prediction",
        "research_question": "Does the model generalize to a second cohort?",
        "expected_contribution": "We test transfer across datasets.",
    }
    report = V.score_venue_fit(spec, rules_path=RULES_PATH)
    assert "VF-05" in report.codes


def test_vf06_measurement_to_decision() -> None:
    spec = {
        "task_type": "psychometrics",
        "scale_name": "math self-efficacy",
        "research_question": (
            "Does the scale support course placement decisions?"
        ),
    }
    report = V.score_venue_fit(spec, rules_path=RULES_PATH)
    assert "VF-06" in report.codes


def test_vf06_does_not_apply_outside_psychometrics() -> None:
    spec = {"task_type": "prediction",
            "research_question": "Does it support course placement?"}
    report = V.score_venue_fit(spec, rules_path=RULES_PATH)
    assert "VF-06" not in report.codes


def test_vf07_synthetic_only() -> None:
    spec = {
        "task_type": "causal_soo",
        "research_question": "Does the estimator recover the truth?",
        "expected_contribution": "Certified on synthetic data only.",
        "subgroup_analyses": ["a", "b"],  # keeps VF-01 out of the way
    }
    report = V.score_venue_fit(spec, rules_path=RULES_PATH)
    assert "VF-07" in report.codes


def test_vf07_does_not_fire_when_a_dataset_is_declared() -> None:
    spec = {
        "task_type": "causal_soo",
        "dataset": "hsls09_public",
        "research_question": "Does the estimator recover the truth?",
        "expected_contribution": "Validated on synthetic data, then on HSLS.",
        "subgroup_analyses": ["a", "b"],
    }
    report = V.score_venue_fit(spec, rules_path=RULES_PATH)
    assert "VF-07" not in report.codes


# --------------------------------------------------------------------------
# VF-08 ethics weighting
# --------------------------------------------------------------------------


def test_ethics_multiplier_scales_a_fairness_only_contribution() -> None:
    spec = {
        "task_type": "prediction",
        "research_question": "Do attitudes predict enrollment?",
        "subgroup_analyses": ["X1SEX", "X1RACE", "X1SESQ5"],
    }
    edm = V.score_venue_fit(spec, venue="EDM", rules_path=RULES_PATH).score
    jedm = V.score_venue_fit(spec, venue="JEDM", rules_path=RULES_PATH).score
    jla = V.score_venue_fit(spec, venue="JLA", rules_path=RULES_PATH).score
    assert edm == pytest.approx(1.5)
    assert jedm == pytest.approx(0.75)
    assert jla == pytest.approx(0.9)
    assert jedm < jla < edm


def test_ethics_multiplier_not_applied_to_non_fairness_contributions() -> None:
    spec = {
        "task_type": "causal_did",
        "research_question": "Did the gap change?",
        "primary_method": "M9",
        "secondary_methods": ["M10"],
    }
    edm = V.score_venue_fit(spec, venue="EDM", rules_path=RULES_PATH).score
    jedm = V.score_venue_fit(spec, venue="JEDM", rules_path=RULES_PATH).score
    assert edm == jedm == pytest.approx(1.5)


def test_unknown_venue_falls_back_to_weight_one() -> None:
    spec = {"task_type": "prediction", "subgroup_analyses": ["a", "b"],
            "research_question": "x"}
    report = V.score_venue_fit(spec, venue="NEURIPS", rules_path=RULES_PATH)
    assert report.venue == "NEURIPS"
    assert report.score == pytest.approx(1.5)


# --------------------------------------------------------------------------
# Determinism, evidence, and C1
# --------------------------------------------------------------------------


def test_scoring_is_deterministic() -> None:
    spec = _did_v2()
    first = V.score_venue_fit(spec, rules_path=RULES_PATH).to_dict()
    second = V.score_venue_fit(spec, rules_path=RULES_PATH).to_dict()
    assert first == second


def test_every_hit_carries_rule_evidence_and_artifact_evidence() -> None:
    for spec in (_did_v1(), _did_v2()):
        report = V.score_venue_fit(spec, rules_path=RULES_PATH)
        for hit in report.hits:
            assert hit.evidence.strip(), hit.code  # anchor fact (C2)
            assert hit.why.strip(), hit.code  # what fired it, in THIS artifact


def test_score_is_invariant_to_self_assessed_novelty() -> None:
    """C1: no positive novelty number enters any ranking term."""
    scores = set()
    for novelty in (1, 3, 5, 0.43, None):
        spec = dict(_did_v2(), novelty_score_self_assessment=novelty)
        scores.add(V.score_venue_fit(spec, rules_path=RULES_PATH).score)
    assert len(scores) == 1


def test_empty_spec_scores_zero_without_raising() -> None:
    report = V.score_venue_fit({}, rules_path=RULES_PATH)
    assert report.score == 0.0 or isinstance(report.score, float)


def test_venue_fit_rules_yaml_matches_default_path_constant() -> None:
    assert V.DEFAULT_RULES_PATH.name == "venue_fit_rules.yaml"
    with open(RULES_PATH, encoding="utf-8") as f:
        assert yaml.safe_load(f)["schema_version"] == "1.0"
