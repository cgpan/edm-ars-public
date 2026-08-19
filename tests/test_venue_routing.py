"""Arc T / H8 + H2 Option C - v2 clause evaluator, venue router, config.

All offline: no LLM, no network, no raw data. Three obligations:

1. **H8, the blocker.** ``src/ideation/venue_fit.py`` must evaluate the
   v2 declarative rule table (clause enum: field_regex / task_type_in /
   dataset_in composed by any_of / all_of / none_of), with the
   de-hyphenation pass that kills the ``"shap- ing"`` PDF artifact, while
   the loud-failure guard still raises on a genuinely unknown predicate
   vocabulary.
2. **Option C routing.** ``route_idea`` sends causal work to the
   policy-causal family WITH evidence, psychometrics to JEDM, marks
   AERA_OPEN as advisory-uncalibrated, and VF2-01/02 are penalties ONLY
   under computational-edm routing.
3. **Config.** The ``ideation:`` block in config.yaml is read by the
   tournament / judge / router, and its shipped values equal the code
   fallbacks (H5 moved defaults, it did not change behavior).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ideation import judge as J  # noqa: E402
from src.ideation import priorart as P  # noqa: E402
from src.ideation import tournament as T  # noqa: E402
from src.ideation import venue_fit as V  # noqa: E402
from src.ideation import venue_router as R  # noqa: E402

V1_RULES_PATH = REPO_ROOT / "data_registry" / "venue_fit_rules.yaml"
V2_RULES_PATH = REPO_ROOT / "data_registry" / "venue_fit_rules_v2.yaml"
CONFIG_PATH = REPO_ROOT / "config.yaml"


@pytest.fixture(scope="module")
def v2_table() -> dict:
    with open(V2_RULES_PATH, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _card(**kwargs: Any) -> dict:
    base: dict[str, Any] = {
        "candidate_id": "C-01",
        "research_question": "Does something happen for students?",
        "why_it_matters": "It would be useful to know.",
        "what_we_would_do": "Fit a model.",
        "what_counts_as_the_result": "A number.",
        "cell": {"dataset": "assistments_0910", "task_type": "prediction"},
    }
    cell = kwargs.pop("cell", None)
    base.update(kwargs)
    if cell:
        base["cell"] = {**base["cell"], **cell}
    return base


# --------------------------------------------------------------------------
# 1. The v2 clause evaluator in src (H8) - every clause kind
# --------------------------------------------------------------------------


def test_field_regex_fires_and_names_what_it_read() -> None:
    clause = {
        "kind": "field_regex",
        "fields": ["research_question"],
        "patterns": ["widget"],
    }
    fired, why = V.evaluate_predicate(clause, _card(research_question="Widget use."))
    assert fired
    assert "research_question" in why and "widget" in why
    assert not V.evaluate_predicate(clause, _card())[0]


def test_field_regex_reads_card_before_spec_and_joins_lists() -> None:
    clause = {
        "kind": "field_regex",
        "fields": ["what_we_would_do"],
        "patterns": ["propensity score"],
    }
    spec = {"what_we_would_do": ["estimate a", "propensity score model"]}
    fired, _ = V.evaluate_predicate(clause, {}, spec)
    assert fired


def test_task_type_in_resolves_cell_first() -> None:
    clause = {"kind": "task_type_in", "values": ["causal_did"]}
    fired, why = V.evaluate_predicate(clause, _card(cell={"task_type": "causal_did"}))
    assert fired and "causal_did" in why
    # cell wins over spec
    fired, _ = V.evaluate_predicate(
        clause, _card(cell={"task_type": "prediction"}), {"task_type": "causal_did"}
    )
    assert not fired


def test_dataset_in_clause() -> None:
    clause = {"kind": "dataset_in", "values": ["hsls09_public"]}
    assert V.evaluate_predicate(clause, _card(cell={"dataset": "hsls09_public"}))[0]
    assert not V.evaluate_predicate(clause, _card())[0]


def test_composition_semantics_match_the_reference() -> None:
    regex_clause = {
        "kind": "field_regex",
        "fields": ["research_question"],
        "patterns": ["widget"],
    }
    task_clause = {"kind": "task_type_in", "values": ["prediction"]}
    card = _card(research_question="A study of widget use.")
    assert V.evaluate_predicate(
        {"kind": "any_of", "clauses": [regex_clause, task_clause]}, card
    )[0]
    assert V.evaluate_predicate(
        {"kind": "all_of", "clauses": [regex_clause, task_clause]}, card
    )[0]
    assert not V.evaluate_predicate(
        {"kind": "none_of", "clauses": [regex_clause, task_clause]}, card
    )[0]
    other = _card(
        research_question="Nothing here.", cell={"task_type": "psychometrics"}
    )
    assert not V.evaluate_predicate(
        {"kind": "any_of", "clauses": [regex_clause, task_clause]}, other
    )[0]
    assert V.evaluate_predicate(
        {"kind": "none_of", "clauses": [regex_clause, task_clause]}, other
    )[0]


def test_empty_all_of_does_not_fire() -> None:
    assert not V.evaluate_predicate({"kind": "all_of", "clauses": []}, _card())[0]


def test_unknown_clause_kind_raises() -> None:
    with pytest.raises(ValueError, match="unknown predicate clause kind"):
        V.evaluate_predicate({"kind": "eval_python", "code": "1"}, _card())


def test_unknown_kind_nested_in_a_composite_still_raises() -> None:
    clause = {
        "kind": "any_of",
        "clauses": [
            {"kind": "task_type_in", "values": ["prediction"]},
            {"kind": "not_a_kind"},
        ],
    }
    with pytest.raises(ValueError, match="unknown predicate clause kind"):
        V.evaluate_predicate(clause, _card())


# --------------------------------------------------------------------------
# 2. De-hyphenation (the "shap- ing" artifact)
# --------------------------------------------------------------------------


def test_dehyphenate_rejoins_pdf_linewrap() -> None:
    assert V.dehyphenate("shap- ing") == "shaping"
    assert V.dehyphenate("shap-\ning") == "shaping"


def test_dehyphenate_preserves_real_compounds() -> None:
    assert V.dehyphenate("difference-in-differences") == "difference-in-differences"
    assert V.dehyphenate("cross-context") == "cross-context"


def test_shap_pattern_does_not_fire_on_the_known_artifact() -> None:
    """The derivation false hit: 'shap- ing' matched \\bshap\\b."""
    clause = {
        "kind": "field_regex",
        "fields": ["why_it_matters"],
        "patterns": [r"\bshap\b"],
    }
    artifact = _card(why_it_matters="We are shap- ing the curriculum.")
    assert not V.evaluate_predicate(clause, artifact)[0]
    genuine = _card(why_it_matters="A shap ranking of predictors.")
    assert V.evaluate_predicate(clause, genuine)[0]


def test_v1_keyword_blob_is_also_dehyphenated() -> None:
    """The v1 phrase keywords see de-hyphenated, whitespace-collapsed
    text: a PDF line-wrap inside a phrase no longer hides a genuine
    match. (The bare 'shap' v1 keyword is SUBSTRING-based, so the
    'shap- ing' -> 'shaping' rejoin cannot un-hit it there - only the
    v2 word-bounded patterns fix that artifact; see the test above.)"""
    spec = {
        "task_type": "prediction",
        "research_question": "We report the feature import- ance ranking.",
    }
    facts = V.extract_facts(spec)
    assert "feature importance" in facts["shap_terms"]
    spec["research_question"] = "A shap analysis of student futures."
    assert "shap" in V.extract_facts(spec)["shap_terms"]


def test_hyphenated_pattern_still_matches_after_normalization(v2_table: dict) -> None:
    rule = next(r for r in v2_table["rules"] if r["code"] == "VF2-01")
    card = _card(what_we_would_do="Run a difference-in-differences analysis.")
    fired, why = V.evaluate_predicate(rule["predicate"], card)
    assert fired and "difference" in why


# --------------------------------------------------------------------------
# 3. The v2 table scores through the src scorer (H8, the blocker)
# --------------------------------------------------------------------------


def test_v2_table_no_longer_raises_and_scores(v2_table: dict) -> None:
    """Before H8 this exact call raised the loud-failure guard."""
    report = V.score_venue_fit({}, rules=v2_table)
    assert report.score == 0.0
    assert report.hits == []


def test_vf2_06_is_now_scoreable(v2_table: dict) -> None:
    """The H2 finding: nothing was scored against VF2-06 at all."""
    card = _card(
        what_counts_as_the_result="The model generalizes to a second dataset."
    )
    report = V.score_venue_fit(
        {"task_type": "prediction"}, card=card, rules=v2_table
    )
    assert "VF2-06" in report.codes
    hit = next(h for h in report.hits if h.code == "VF2-06")
    assert hit.delta == 0.5
    assert hit.why and hit.evidence


def test_vf2_07_is_now_scoreable(v2_table: dict) -> None:
    card = _card(
        why_it_matters="It would change course placement for entering students."
    )
    report = V.score_venue_fit(
        {"task_type": "prediction"}, card=card, rules=v2_table
    )
    assert "VF2-07" in report.codes


def test_vf2_01_fires_on_a_causal_task_type_without_routing(v2_table: dict) -> None:
    card = _card(cell={"task_type": "causal_did"})
    report = V.score_venue_fit(
        {"task_type": "causal_did"}, card=card, rules=v2_table
    )
    hit = next(h for h in report.hits if h.code == "VF2-01")
    assert hit.delta == -1.5
    assert hit.role == "scored"


def test_v2_venue_delta_applies_only_at_its_venue(v2_table: dict) -> None:
    card = _card(what_we_would_do="Use a large language model to code responses.")
    jedm = V.score_venue_fit({}, card=card, rules=v2_table, venue="JEDM")
    edm = V.score_venue_fit({}, card=card, rules=v2_table, venue="EDM")
    assert jedm.score > edm.score


def test_v2_scoring_is_deterministic(v2_table: dict) -> None:
    card = _card(cell={"dataset": "hsls09_public", "task_type": "causal_soo"})
    first = V.score_venue_fit(
        {"task_type": "causal_soo"}, card=card, rules=v2_table
    ).to_dict()
    second = V.score_venue_fit(
        {"task_type": "causal_soo"}, card=card, rules=v2_table
    ).to_dict()
    assert first == second


# --------------------------------------------------------------------------
# 4. The loud-failure guard survives H8
# --------------------------------------------------------------------------


def test_guard_raises_on_all_unknown_named_predicates() -> None:
    table = {
        "rules": [
            {"code": "X-01", "predicate": "not_a_predicate", "delta": 1.0},
            {"code": "X-02", "predicate": "also_unknown", "delta": 1.0},
        ]
    }
    with pytest.raises(ValueError, match="NONE of its"):
        V.score_venue_fit({}, rules=table)


def test_guard_raises_on_all_unknown_clause_kinds() -> None:
    table = {
        "rules": [
            {"code": "X-01", "predicate": {"kind": "eval_python"}, "delta": 1.0},
        ]
    }
    with pytest.raises(ValueError, match="NONE of its"):
        V.score_venue_fit({}, rules=table)


def test_partially_unknown_table_still_scores_the_known_rules() -> None:
    """Preserved v1 semantics: individual unknown rules are skipped; the
    guard fires only when the WHOLE vocabulary is unknown."""
    table = {
        "rules": [
            {
                "code": "K-01",
                "predicate": {"kind": "task_type_in", "values": ["prediction"]},
                "delta": 0.5,
                "applies_to": ["prediction"],
            },
            {"code": "X-01", "predicate": "not_a_predicate", "delta": -9.0},
        ]
    }
    report = V.score_venue_fit(
        {"task_type": "prediction"},
        card=_card(),
        rules=table,
    )
    assert report.codes == ["K-01"]
    assert report.score == 0.5


def test_unknown_nested_kind_in_an_otherwise_known_table_raises_loudly() -> None:
    """An unknown clause kind must never silently contribute 0.0."""
    table = {
        "rules": [
            {
                "code": "K-01",
                "predicate": {
                    "kind": "any_of",
                    "clauses": [{"kind": "spooky_new_kind"}],
                },
                "delta": 0.5,
            },
        ]
    }
    with pytest.raises(ValueError):
        V.score_venue_fit({}, card=_card(), rules=table)


# --------------------------------------------------------------------------
# 5. Routing hook in the scorer: penalty vs routing signal
# --------------------------------------------------------------------------


def test_vf2_01_is_a_penalty_without_routing_and_a_signal_under_policy(
    v2_table: dict,
) -> None:
    card = _card(cell={"task_type": "causal_soo", "dataset": "assistments_0910"})
    spec = {"task_type": "causal_soo"}
    plain = V.score_venue_fit(spec, card=card, rules=v2_table)
    routed = V.score_venue_fit(
        spec, card=card, rules=v2_table, routing_family=V.FAMILY_POLICY
    )
    plain_hit = next(h for h in plain.hits if h.code == "VF2-01")
    routed_hit = next(h for h in routed.hits if h.code == "VF2-01")
    assert plain_hit.delta == -1.5 and plain_hit.role == "scored"
    assert routed_hit.delta == 0.0 and routed_hit.role == "routing_signal"
    assert "ROUTING SIGNAL" in routed_hit.why
    assert routed.score == plain.score + 1.5


def test_vf2_01_remains_a_penalty_under_computational_routing(v2_table: dict) -> None:
    """Dual-targeting: routed to computational-edm, the penalty stands."""
    card = _card(cell={"task_type": "causal_soo", "dataset": "assistments_0910"})
    report = V.score_venue_fit(
        {"task_type": "causal_soo"},
        card=card,
        rules=v2_table,
        routing_family=V.FAMILY_COMPUTATIONAL,
    )
    hit = next(h for h in report.hits if h.code == "VF2-01")
    assert hit.delta == -1.5 and hit.role == "scored"


def test_vf2_02_neutralized_under_policy_routing(v2_table: dict) -> None:
    card = _card(cell={"dataset": "hsls09_public"})
    routed = V.score_venue_fit(
        {}, card=card, rules=v2_table, routing_family=V.FAMILY_POLICY
    )
    hit = next(h for h in routed.hits if h.code == "VF2-02")
    assert hit.delta == 0.0 and hit.role == "routing_signal"


def test_non_routing_codes_still_penalize_under_policy_routing(v2_table: dict) -> None:
    """Only VF-01/VF2-01/VF2-02 are routing signals; VF2-03 is a real
    defect at every destination and keeps its delta."""
    card = _card(
        cell={"task_type": "causal_soo"},
        what_counts_as_the_result="A SHAP ranking of the top predictors.",
    )
    report = V.score_venue_fit(
        {"task_type": "causal_soo"},
        card=card,
        rules=v2_table,
        routing_family=V.FAMILY_POLICY,
    )
    shap_hit = next(h for h in report.hits if h.code == "VF2-03")
    assert shap_hit.delta == -1.0 and shap_hit.role == "scored"


def test_v1_vf01_is_also_neutralized_under_policy_routing() -> None:
    spec = {
        "task_type": "causal_did",
        "primary_method": "M8",
        "research_question": "A bare causal question.",
    }
    plain = V.score_venue_fit(spec, rules_path=V1_RULES_PATH)
    routed = V.score_venue_fit(
        spec, rules_path=V1_RULES_PATH, routing_family=V.FAMILY_POLICY
    )
    assert next(h for h in plain.hits if h.code == "VF-01").delta == -2.0
    routed_hit = next(h for h in routed.hits if h.code == "VF-01")
    assert routed_hit.delta == 0.0 and routed_hit.role == "routing_signal"


# --------------------------------------------------------------------------
# 6. The router
# --------------------------------------------------------------------------


def test_causal_soo_routes_policy_with_evidence() -> None:
    spec = {"task_type": "causal_soo", "dataset": "hsls09_public"}
    route = R.route_idea(spec)
    assert route["family"] == "policy-causal"
    assert route["venue"] == "AERA_OPEN"
    assert route["gate_status"] == "advisory-uncalibrated-venue"
    assert route["rule"] == "R1-observational-causal"
    assert "task_type=causal_soo" in route["evidence"]
    codes = {s["code"]: s for s in route["signals"]}
    assert codes["VF2-01"]["role"] == "routing"
    assert codes["VF2-01"]["evidence"]  # C2: anchor/counter-corpus fact
    # the survey extract is ALSO routing evidence at this destination
    assert codes["VF2-02"]["role"] == "routing"


@pytest.mark.parametrize("task_type", ["causal_soo", "causal_itr", "causal_did"])
def test_every_causal_task_type_routes_policy(task_type: str) -> None:
    assert R.route_idea({"task_type": task_type})["family"] == "policy-causal"


def test_estimator_language_on_a_prediction_card_routes_policy() -> None:
    """VF2-01-style facts route even when the task_type vocabulary does
    not: a causal claim smuggled into a prediction card must not escape."""
    card = _card(
        what_we_would_do=(
            "Estimate the average treatment effect using propensity "
            "score weighting."
        )
    )
    route = R.route_idea({"task_type": "prediction"}, card)
    assert route["family"] == "policy-causal"
    assert "propensity" in route["evidence"] or "average treatment" in route["evidence"]


def test_psychometrics_routes_to_jedm_calibrated() -> None:
    route = R.route_idea({"task_type": "psychometrics"})
    assert route["family"] == "computational-edm"
    assert route["venue"] == "JEDM"
    assert route["gate_status"] == "calibrated"
    assert route["rule"] == "R2-psychometrics"


def test_prediction_with_fairness_contribution_stays_computational() -> None:
    spec = {
        "task_type": "prediction",
        "dataset": "hsls09_public",
        "heterogeneity_subgroups": ["X1SEX", "X1RACE"],
    }
    route = R.route_idea(spec)
    assert route["family"] == "computational-edm"
    assert route["rule"] == "R3-prediction-contribution"
    assert "fairness" in route["evidence"]
    # VF2-02 fired (survey dataset) and remains a PENALTY at this destination
    codes = {s["code"]: s for s in route["signals"]}
    assert codes["VF2-02"]["role"] == "penalty"


def test_population_description_on_a_national_survey_routes_policy() -> None:
    card = _card(
        cell={"dataset": "hsls09_public"},
        what_counts_as_the_result=(
            "A population-level description of course-taking in a "
            "nationally representative cohort."
        ),
    )
    route = R.route_idea({"task_type": "prediction"}, card)
    assert route["family"] == "policy-causal"
    assert route["rule"] == "R4-population-description-national-survey"
    assert route["gate_status"] == "advisory-uncalibrated-venue"


def test_undecidable_prediction_defaults_computational_and_says_so() -> None:
    route = R.route_idea({"task_type": "prediction", "dataset": "assistments_0910"})
    assert route["family"] == "computational-edm"
    assert route["rule"] == "R5-default-undecidable"
    assert "UNDECIDABLE" in route["evidence"]
    assert "default" in route["evidence"].lower()


def test_missing_task_type_defaults_computational_and_says_so() -> None:
    route = R.route_idea({})
    assert route["family"] == "computational-edm"
    assert "UNDECIDABLE" in route["evidence"]


def test_router_is_deterministic() -> None:
    spec = {"task_type": "causal_did", "dataset": "did_els_hsls_panel"}
    assert R.route_idea(spec) == R.route_idea(spec)


def test_router_config_override_keeps_uncalibrated_status() -> None:
    cfg = {"ideation": {"routing": {"policy_venue": "JREE"}}}
    route = R.route_idea({"task_type": "causal_did"}, config=cfg)
    assert route["venue"] == "JREE"
    assert route["gate_status"] == "advisory-uncalibrated-venue"


def test_router_clauses_stay_in_sync_with_the_v2_table(v2_table: dict) -> None:
    """The router mirrors VF2-01/VF2-02 verbatim; drift is a defect."""
    by_code = {r["code"]: r for r in v2_table["rules"]}
    assert R.OBSERVATIONAL_CAUSAL_PREDICATE == by_code["VF2-01"]["predicate"]
    assert R.NATIONAL_SURVEY_PREDICATE == by_code["VF2-02"]["predicate"]
    assert list(R.CARD_TEXT_FIELDS) == list(v2_table["card_text_fields"])


# --------------------------------------------------------------------------
# 7. Tournament integration + config (offline; no judge, no prior art)
# --------------------------------------------------------------------------


def _record(
    cid: str,
    *,
    task_type: str = "prediction",
    dataset: str = "hsls09_public",
    venue_fit: float = 0.0,
    card_overrides: dict | None = None,
    spec_overrides: dict | None = None,
) -> dict:
    card = _card(
        candidate_id=cid,
        cell={"dataset": dataset, "task_type": task_type},
        **(card_overrides or {}),
    )
    spec = {
        "task_id": cid,
        "task_type": task_type,
        "dataset": dataset,
        "outcome_variable": f"OUTCOME_{cid[-1]}",
        "research_question": f"Question for {cid}.",
    }
    spec.update(spec_overrides or {})
    return {
        "candidate_id": cid,
        "card": card,
        "spec": spec,
        "feasibility": {
            "candidate_id": cid,
            "verdict": "CLEAN",
            "dataset": dataset,
            "task_type": task_type,
            "analytic_n_estimate": 12960,
            "penalty": 0.0,
            "checks": [],
        },
        "venue_fit": {"score": venue_fit, "venue": "EDM", "hits": [], "facts": {}},
    }


def _config(**tournament_overrides: Any) -> dict:
    tournament = {
        "judge_samples": 5,
        "weight_venue_fit": 0.42,
        "weight_feasibility_penalty": 0.17,
        "bt_prior_sd": 0.7,
    }
    tournament.update(tournament_overrides)
    return {
        "ideation": {
            "venue_fit": {"rules_path": str(V2_RULES_PATH)},
            "tournament": tournament,
        }
    }


def _run(records: list[dict], config: dict | None = None) -> Any:
    return T.run_cascade(
        records,
        tournament_id="T-ROUTE",
        config=config,
        judged=False,
        prior_art=False,
        run_shuffle_control=False,
    )


def test_cascade_reads_the_ideation_config_block() -> None:
    result = _run([_record("C-01"), _record("C-02")], config=_config())
    weights = result.ranking["weights"]
    assert weights["venue_fit"] == 0.42
    assert weights["feasibility_penalty"] == 0.17
    assert weights["bt_prior_sd"] == 0.7
    judge_stage = next(
        s for s in result.ranking["cascade"] if s["stage"] == "pairwise_judge"
    )
    assert judge_stage["samples_per_orientation"] == 5
    fit_stage = next(
        s for s in result.ranking["cascade"] if s["stage"] == "venue_fit"
    )
    assert fit_stage["rules_path"].endswith("venue_fit_rules_v2.yaml")


def test_cascade_stamps_family_venue_gate_status_into_ranking_rows() -> None:
    records = [
        _record("C-CAUSAL", task_type="causal_soo", dataset="hsls09_public"),
        _record("C-PSY", task_type="psychometrics", dataset="els_2002"),
        _record("C-PRED", task_type="prediction", dataset="assistments_0910"),
    ]
    result = _run(records, config=_config())
    rows = {
        row["candidate_id"]: row
        for row in result.ranking["ranking_deterministic"]
    }
    causal = rows["C-CAUSAL"]["venue_routing"]
    assert causal["family"] == "policy-causal"
    assert causal["venue"] == "AERA_OPEN"
    assert causal["gate_status"] == "advisory-uncalibrated-venue"
    psy = rows["C-PSY"]["venue_routing"]
    assert psy["family"] == "computational-edm"
    assert psy["venue"] == "JEDM"
    assert psy["gate_status"] == "calibrated"
    pred = rows["C-PRED"]["venue_routing"]
    assert pred["family"] == "computational-edm"
    # the judged ordering carries the same stamp
    for row in result.ranking["ranking"]:
        assert row["venue_routing"]["family"] in (
            "policy-causal",
            "computational-edm",
        )


def test_cascade_ranking_carries_the_routing_block_and_uncalibrated_note() -> None:
    result = _run(
        [_record("C-CAUSAL", task_type="causal_did")], config=_config()
    )
    block = result.ranking["venue_routing"]
    assert "AERA_OPEN" in block["uncalibrated_note"]
    assert "advisory-uncalibrated-venue" in block["uncalibrated_note"]
    assert block["by_candidate"]["C-CAUSAL"]["family"] == "policy-causal"
    routing_stage = next(
        s for s in result.ranking["cascade"] if s["stage"] == "venue_routing"
    )
    assert routing_stage["families"].get("policy-causal") == 1
    # the digest prints the routing table
    digest = result.digest()
    assert "Venue routing" in digest
    assert "advisory-uncalibrated-venue" in digest


def test_policy_routed_candidate_is_rescored_without_the_routing_penalties() -> None:
    """The stale hand-built venue_fit (scored without routing) must be
    rescored: VF2-01/02 fire as routing signals with delta 0.0."""
    record = _record("C-CAUSAL", task_type="causal_soo", venue_fit=-99.0)
    result = _run([record], config=_config())
    candidate = result.candidates[0]
    assert candidate.venue_fit["facts"]["routing_family"] == "policy-causal"
    hits = {h["code"]: h for h in candidate.venue_fit["hits"]}
    assert hits["VF2-01"]["role"] == "routing_signal"
    assert hits["VF2-01"]["delta"] == 0.0
    assert hits["VF2-02"]["role"] == "routing_signal"
    assert candidate.venue_fit["score"] == 0.0
    # C2: the routing evidence is in the row's evidence block
    row = result.ranking["ranking_deterministic"][0]
    assert any(
        "R1-observational-causal" in line
        for line in row["evidence"]["venue_routing"]
    )


def test_computational_routed_candidate_keeps_its_cached_venue_fit() -> None:
    """A cached report is NOT stale at a computational destination."""
    record = _record("C-PRED", task_type="prediction", venue_fit=2.5)
    result = _run([record])  # no config: v1 default table would apply anyway
    candidate = result.candidates[0]
    assert candidate.venue_fit["score"] == 2.5  # untouched fixture


def test_tournament_remains_advisory_after_routing() -> None:
    """Routing does not clear V2. The refusal is untouched."""
    records = [_record("C-01"), _record("C-02")]
    result = _run(records, config=_config())
    assert result.ranking["advisory"] is True
    assert result.ranking["authorized_for_live_selection"] is False
    with pytest.raises(T.LiveSelectionNotAuthorized):
        T.run_cascade(
            records,
            tournament_id="T-ROUTE-LIVE",
            judged=False,
            prior_art=False,
            allow_live_selection=True,
        )


# --------------------------------------------------------------------------
# 8. The shipped config.yaml block (H5: defaults moved, behavior kept)
# --------------------------------------------------------------------------


def test_config_yaml_ships_the_ideation_block() -> None:
    with open(CONFIG_PATH, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    ideation = config["ideation"]
    assert ideation["venue_fit"]["rules_path"] == (
        "data_registry/venue_fit_rules_v2.yaml"
    )
    assert ideation["routing"]["policy_venue"] == "AERA_OPEN"
    tournament = ideation["tournament"]
    # H5 contract: the shipped values EQUAL the code fallbacks, so the
    # move to config changes no behavior.
    assert tournament["judge_samples"] == J.DEFAULT_SAMPLES
    assert tournament["judge_temperature"] is None
    assert tournament["weight_venue_fit"] == T.DEFAULT_WEIGHT_VENUE_FIT
    assert tournament["weight_feasibility_penalty"] == T.DEFAULT_WEIGHT_PENALTY
    assert tournament["bt_prior_sd"] == T.DEFAULT_PRIOR_SD
    assert tournament["max_survivors_to_tournament"] == T.DEFAULT_MAX_SURVIVORS
    assert tournament["swiss_rounds"] == T.DEFAULT_SWISS_ROUNDS
    priorart = ideation["priorart"]
    assert priorart["purpose_coverage_min"] == P.PURPOSE_COVERAGE_MIN
    assert priorart["anchor_corpus"] == P.DEFAULT_ANCHOR_CORPUS


def test_judge_reads_samples_and_temperature_from_the_ideation_block() -> None:
    cfg = {
        "ideation": {
            "tournament": {"judge_samples": 7, "judge_temperature": 0.4}
        }
    }
    assert J.judge_samples(cfg) == 7
    assert J.judge_temperature(cfg) == 0.4
    assert J.judge_samples({}) == J.DEFAULT_SAMPLES
    assert J.judge_temperature({}) is None
