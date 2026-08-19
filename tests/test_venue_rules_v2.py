"""Arc T -- tests for the blind-derived venue-fit rule table (v2).

Three layers, all offline, no LLM, no network:

1. Integrity + blindness. The table must declare every rule
   ``provenance: external``, must not smuggle in an outcome-derived
   justification, and must obey its own stated derivation policy. These
   run everywhere.
2. Predicate semantics. The declarative clause language has one reference
   evaluator (``scripts.derive_venue_rules.evaluate_predicate``) and these
   tests pin it. Every rule gets a constructed positive control and a
   negative control -- this is the only validation VF2-01 can have, since
   it fires on 0 of 34 anchors.
3. Corpus reproduction. Skipped when the anchor corpus is not on this
   machine (it lives outside the repo).
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.derive_venue_rules import (  # noqa: E402
    BANDED_SIGNS,
    Anchor,
    audit_table,
    bundle_sizes,
    count,
    evaluate_predicate,
    load_corpus,
    policy_delta,
    score_card,
    venue_of,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RULES_PATH = REPO_ROOT / "data_registry" / "venue_fit_rules_v2.yaml"
V1_RULES_PATH = REPO_ROOT / "data_registry" / "venue_fit_rules.yaml"


@pytest.fixture(scope="module")
def table() -> dict:
    with open(RULES_PATH, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _corpus_path(table: dict) -> Path:
    return Path((table.get("corpus") or {}).get("path", ""))


# --------------------------------------------------------------------------
# 1. Integrity and blindness
# --------------------------------------------------------------------------


def test_rules_file_is_ascii() -> None:
    """Project rule: data_registry/*.yaml is ASCII only. An em dash broke
    the suite once already."""
    text = RULES_PATH.read_text(encoding="utf-8")
    offenders = [(i, line) for i, line in enumerate(text.splitlines(), 1)
                 if not all(ord(c) < 128 for c in line)]
    assert offenders == [], offenders[:3]


def test_table_declares_itself_external(table: dict) -> None:
    assert table["provenance"] == "external"
    assert table["schema_version"] == "2.0"
    assert table["table_id"] == "venue_fit_rules_v2"


def test_every_rule_is_provenance_external(table: dict) -> None:
    """The whole point of v2: no rule may rest on our own run outcomes."""
    for rule in table["rules"]:
        assert rule["provenance"] == "external", rule["code"]


def test_rule_codes_are_unique_and_do_not_collide_with_v1(table: dict) -> None:
    codes = [rule["code"] for rule in table["rules"]]
    assert len(codes) == len(set(codes))
    # v1 uses VF-NN; v2 uses VF2-NN. A shared code would make an audit
    # trail ambiguous about which table produced a hit.
    assert all(code.startswith("VF2-") for code in codes)


def test_no_rule_evidence_cites_our_own_outcomes(table: dict) -> None:
    """Blindness guard.

    An ``evidence`` string may cite anchor counts and nothing else. Words
    naming this project's own measured outcomes are exactly what made the
    v1 table unidentified. (``caveats`` are exempt: they are allowed, and
    required, to discuss provenance and contamination risk.)
    """
    forbidden = re.compile(
        r"\b(ledger|gate score|gate_summary|lsar|reject\b|accept\b|rho|"
        r"backtest|our own run|realized score)\b",
        re.IGNORECASE,
    )
    for rule in table["rules"]:
        hit = forbidden.search(str(rule["evidence"]))
        assert hit is None, f"{rule['code']} evidence cites an outcome: {hit.group(0)!r}"
    for rejected in table.get("rejected_rules") or []:
        hit = forbidden.search(str(rejected.get("rejected_because", "")))
        assert hit is None, f"rejected[{rejected['name']}]: {hit.group(0)!r}"


def test_no_rule_reuses_a_v1_in_sample_predicate(table: dict) -> None:
    """v1's VF-01 and VF-04 are tagged in_sample. v2 must not re-import
    their predicate names, which would silently reattach the contaminated
    logic to a table claiming to be external."""
    banned = {"bare_observational_causal", "second_contribution"}
    text = RULES_PATH.read_text(encoding="utf-8")
    for name in banned:
        assert name not in text


def test_every_rule_has_the_required_fields(table: dict) -> None:
    required = (
        "code", "name", "provenance", "sign", "delta", "applies_to",
        "summary", "evidence", "measurement", "predicate",
        "corpus_separation", "detector_validated_on_anchors",
    )
    known_task_types = set(table["known_task_types"])
    for rule in table["rules"]:
        for key in required:
            assert key in rule, f"{rule['code']} missing {key}"
        assert rule["sign"] in BANDED_SIGNS, rule["code"]
        assert isinstance(rule["delta"], (int, float))
        assert rule["applies_to"], rule["code"]
        assert set(rule["applies_to"]) <= known_task_types, rule["code"]
        # C2: a component that cannot cite what it read is not shipped.
        assert len(str(rule["evidence"]).strip()) > 80, rule["code"]
        assert re.search(r"\d+ of 34|\d+/34|0 of 34", str(rule["evidence"])), rule["code"]


def test_every_measurement_block_is_recomputable(table: dict) -> None:
    """Each rule must carry the field, the patterns, and the counts, so
    scripts/derive_venue_rules.py can reproduce the evidence string."""
    for rule in table["rules"]:
        measurement = rule["measurement"]
        assert measurement["field"] in ("abstract", "full_text"), rule["code"]
        assert measurement["patterns"], rule["code"]
        observed = measurement["observed"]
        assert observed["n"] == 34
        assert observed["total"] == sum(observed[v] for v in ("EDM", "JEDM", "JLA"))
        for pattern in measurement["patterns"]:
            re.compile(pattern)  # raises if the table ships a bad regex


def test_every_delta_obeys_the_stated_policy(table: dict) -> None:
    """A rule cannot be smuggled in with an off-policy magnitude."""
    for rule in table["rules"]:
        total = rule["measurement"]["observed"]["total"]
        licensed = policy_delta(rule["sign"], total)
        assert licensed is not None, f"{rule['code']} count {total} licenses no rule"
        assert float(rule["delta"]) == licensed, rule["code"]


def test_venue_overrides_clear_the_six_anchor_floor(table: dict) -> None:
    for rule in table["rules"]:
        for venue, delta in (rule.get("venue_delta") or {}).items():
            venue_count = rule["measurement"]["observed"][venue]
            assert venue_count >= 6, f"{rule['code']} override on {venue} rests on {venue_count}"
            assert float(delta) == 1.0, rule["code"]


def test_negative_field_policy_takes_the_smaller_magnitude(table: dict) -> None:
    """Stated policy: for a negative rule, where the other field would
    license a larger magnitude, the declared field wins."""
    for rule in table["rules"]:
        if rule["sign"] != "negative":
            continue
        cross = rule["measurement"].get("cross_field_check")
        if not cross:
            continue
        other = policy_delta("negative", cross["observed_total"])
        if other is None:
            continue
        assert abs(float(rule["delta"])) <= abs(other), rule["code"]


def test_degenerate_rules_are_flagged_not_hidden(table: dict) -> None:
    """A rule whose detector fires on 0 or all 34 anchors gets no
    within-corpus separation. It may still ship (a zero base rate IS the
    evidence for a negative rule) but it must say so, must be capped, and
    must carry caveats."""
    for rule in table["rules"]:
        total = rule["measurement"]["observed"]["total"]
        degenerate = total in (0, 34)
        if not degenerate:
            assert rule["corpus_separation"] == "separated", rule["code"]
            assert rule["detector_validated_on_anchors"] is True, rule["code"]
            continue
        assert rule["sign"] == "negative", (
            f"{rule['code']}: a POSITIVE rule at a degenerate count is not derivable"
        )
        assert rule["corpus_separation"] == "none", rule["code"]
        assert rule["detector_validated_on_anchors"] is False, rule["code"]
        assert abs(float(rule["delta"])) <= 1.5, rule["code"]
        caveats = " ".join(rule.get("caveats") or [])
        assert "NOT VALIDATED" in caveats.upper(), rule["code"]


def test_stricter_predicates_declare_and_explain_their_strictness(table: dict) -> None:
    """When a predicate is narrower than the measurement that justifies
    it, the gap is a live threat to the evidence and must be written down."""
    for rule in table["rules"]:
        strictness = rule.get("predicate_strictness")
        assert strictness in ("same_as_measurement", "stricter_than_measurement"), rule["code"]
        if strictness == "stricter_than_measurement":
            assert len(str(rule.get("strictness_note", "")).strip()) > 60, rule["code"]


def test_rejected_rules_are_recorded_with_their_counts(table: dict) -> None:
    """A dropped rule must leave a trace, or someone re-derives it by
    accident next quarter."""
    rejected = table["rejected_rules"]
    assert len(rejected) >= 4
    for entry in rejected:
        assert entry["name"]
        assert len(str(entry["rejected_because"]).strip()) > 80, entry["name"]
        assert entry.get("measurement"), entry["name"]


def test_no_ethics_multiplier_is_shipped(table: dict) -> None:
    """v1 carries EDM 1.0 / JEDM 0.5 / JLA 0.6. It is not reproducible
    from this corpus, so v2 must not carry it."""
    for venue_cfg in table["venues"].values():
        assert "ethics_weight" not in venue_cfg
    names = {entry["name"] for entry in table["rejected_rules"]}
    assert "venue_ethics_multiplier" in names


def test_v1_table_is_untouched() -> None:
    """Strict file ownership: v2 is a new file, v1 is somebody else's."""
    assert V1_RULES_PATH.exists()
    with open(V1_RULES_PATH, encoding="utf-8") as handle:
        v1 = yaml.safe_load(handle)
    assert [r["code"] for r in v1["rules"]][0] == "VF-01"


def _code_string_literals(path: Path) -> list[str]:
    """Every str literal in a module except the doc/comment prose.

    Docstrings are excluded on purpose: the module docstring names the
    forbidden artifacts in order to state that it does not read them.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstring_nodes: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                if isinstance(body[0].value.value, str):
                    docstring_nodes.add(id(body[0].value))
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstring_nodes
    ]


def test_audit_script_never_opens_a_review_artifact() -> None:
    """C7 rail. The derivation reads published papers only; LSAR's reviews
    of those papers are outcomes and are out of bounds. Checks executable
    string literals, not prose."""
    literals = _code_string_literals(REPO_ROOT / "scripts" / "derive_venue_rules.py")
    forbidden = ("scores.json", "review.json", "review.md", "LSAR_Review_Report",
                 "ledger.json", "gate_summary", "research_spec.json", "lsar_review")
    for literal in literals:
        for artifact in forbidden:
            assert artifact.lower() not in literal.lower(), (artifact, literal)


def test_audit_script_only_reads_the_two_permitted_filenames() -> None:
    """Whitelist, not blacklist: the corpus loader must touch exactly
    paper.md and metadata.json."""
    literals = _code_string_literals(REPO_ROOT / "scripts" / "derive_venue_rules.py")
    filenames = {lit for lit in literals if re.fullmatch(r"[\w.-]+\.(json|md|yaml|csv|txt)", lit)}
    assert filenames <= {"paper.md", "metadata.json", "venue_fit_rules_v2.yaml"}, filenames


# --------------------------------------------------------------------------
# 2. Predicate semantics (the reference evaluator)
# --------------------------------------------------------------------------


def _card(**kwargs) -> dict:
    base = {
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


# One positive control per rule. For VF2-01 this is the ONLY validation
# available, because the detector fires on 0 of 34 anchors.
POSITIVE_CONTROLS = {
    "VF2-01": _card(cell={"task_type": "causal_did"}),
    "VF2-02": _card(cell={"dataset": "hsls09_public"}),
    "VF2-03": _card(what_counts_as_the_result="A SHAP ranking of the top predictors."),
    "VF2-04": _card(what_we_would_do="Use a large language model to code responses."),
    "VF2-05": _card(what_counts_as_the_result="Cohen's kappa against human coders."),
    "VF2-06": _card(what_counts_as_the_result="The model generalizes to a second dataset."),
    "VF2-07": _card(why_it_matters="It would change course placement for entering students."),
    "VF2-08": _card(what_we_would_do="Run a thematic analysis with human-in-the-loop coding."),
}


def test_every_rule_has_a_positive_control(table: dict) -> None:
    assert {rule["code"] for rule in table["rules"]} == set(POSITIVE_CONTROLS)


@pytest.mark.parametrize("code", sorted(POSITIVE_CONTROLS))
def test_predicate_fires_on_its_positive_control(table: dict, code: str) -> None:
    rule = next(r for r in table["rules"] if r["code"] == code)
    fired, why = evaluate_predicate(rule["predicate"], POSITIVE_CONTROLS[code])
    assert fired, f"{code} did not fire on its own positive control"
    assert why, f"{code} fired without naming what it read (C2)"


@pytest.mark.parametrize("code", sorted(POSITIVE_CONTROLS))
def test_predicate_does_not_fire_on_the_neutral_card(table: dict, code: str) -> None:
    """A rule that fires on a contentless card fires on everything."""
    rule = next(r for r in table["rules"] if r["code"] == code)
    fired, _ = evaluate_predicate(rule["predicate"], _card())
    assert not fired, f"{code} fired on a neutral card"


def test_vf2_01_fires_on_estimator_language_without_a_causal_task_type(table: dict) -> None:
    """The detector must key on the claim, not only on our task_type
    vocabulary, or a causal claim smuggled into a prediction card escapes."""
    rule = next(r for r in table["rules"] if r["code"] == "VF2-01")
    card = _card(
        cell={"task_type": "prediction"},
        what_we_would_do="Estimate the average treatment effect using propensity score weighting.",
    )
    fired, why = evaluate_predicate(rule["predicate"], card)
    assert fired
    assert "propensity" in why or "average treatment effect" in why


def test_vf2_02_does_not_fire_on_the_log_dataset(table: dict) -> None:
    rule = next(r for r in table["rules"] if r["code"] == "VF2-02")
    fired, _ = evaluate_predicate(rule["predicate"], _card(cell={"dataset": "assistments_0910"}))
    assert not fired


def test_vf2_06_does_not_fire_on_the_bare_word_transfer(table: dict) -> None:
    """Documented strictness: 'transfer' alone is in the measurement but
    deliberately not in the predicate."""
    rule = next(r for r in table["rules"] if r["code"] == "VF2-06")
    card = _card(why_it_matters="Knowledge transfer is a core learning construct.")
    fired, _ = evaluate_predicate(rule["predicate"], card)
    assert not fired


def test_vf2_07_does_not_fire_on_generic_practice_language(table: dict) -> None:
    """The stated risk on VF2-07 is that it becomes a constant. The
    predicate must reject the generic tokens the measurement contains."""
    rule = next(r for r in table["rules"] if r["code"] == "VF2-07")
    card = _card(
        why_it_matters="The findings have implications for practice and are actionable "
        "for educators, informing intervention and yielding recommendations.",
    )
    fired, _ = evaluate_predicate(rule["predicate"], card)
    assert not fired


def test_unknown_clause_kind_raises() -> None:
    with pytest.raises(ValueError):
        evaluate_predicate({"kind": "eval_python", "code": "1"}, _card())


def test_predicate_composition_semantics() -> None:
    regex_clause = {
        "kind": "field_regex",
        "fields": ["research_question"],
        "patterns": ["widget"],
    }
    task_clause = {"kind": "task_type_in", "values": ["prediction"]}
    card = _card(research_question="A study of widget use.")
    assert evaluate_predicate({"kind": "any_of", "clauses": [regex_clause, task_clause]}, card)[0]
    assert evaluate_predicate({"kind": "all_of", "clauses": [regex_clause, task_clause]}, card)[0]
    assert not evaluate_predicate(
        {"kind": "none_of", "clauses": [regex_clause, task_clause]}, card
    )[0]
    other = _card(research_question="Nothing here.", cell={"task_type": "psychometrics"})
    assert not evaluate_predicate({"kind": "any_of", "clauses": [regex_clause, task_clause]}, other)[0]
    assert evaluate_predicate({"kind": "none_of", "clauses": [regex_clause, task_clause]}, other)[0]


def test_scoring_is_deterministic_and_carries_evidence(table: dict) -> None:
    card = POSITIVE_CONTROLS["VF2-04"]
    first = score_card(table, card, venue="JEDM")
    second = score_card(table, card, venue="JEDM")
    assert first == second
    assert first["hits"], "expected at least one hit"
    for hit in first["hits"]:
        assert hit["evidence"], "C2: every hit names the anchor fact behind the rule"
        assert hit["why"], "C2: every hit names what it read in THIS card"


def test_venue_delta_applies_only_at_its_venue(table: dict) -> None:
    card = POSITIVE_CONTROLS["VF2-04"]
    jedm = score_card(table, card, venue="JEDM")
    edm = score_card(table, card, venue="EDM")
    assert jedm["score"] > edm["score"]


def test_score_is_invariant_to_self_assessed_novelty(table: dict) -> None:
    """C1: no positive novelty score is ever computed, stored, or ranked
    on. Perturbing the recorded field across its whole range must not
    move a single point."""
    base = POSITIVE_CONTROLS["VF2-06"]
    scores = set()
    for value in (1, 2, 3, 4, 5, 7, 10, 0.43, None, {"score": 5}):
        card = dict(base)
        card["novelty_score_self_assessment"] = value
        scores.add(score_card(table, card)["score"])
    assert len(scores) == 1


def test_empty_card_scores_zero_without_raising(table: dict) -> None:
    result = score_card(table, {})
    assert result["score"] == 0.0
    assert result["hits"] == []


def test_policy_delta_bands() -> None:
    assert policy_delta("negative", 0) == -1.5
    assert policy_delta("negative", 1) == -1.0
    assert policy_delta("negative", 3) == -1.0
    assert policy_delta("negative", 4) is None
    assert policy_delta("positive", 5) is None
    assert policy_delta("positive", 6) == 0.5
    assert policy_delta("positive", 22) == 0.5
    assert policy_delta("positive", 23) is None
    assert policy_delta("multiplier", 8) is None
    with pytest.raises(ValueError):
        policy_delta("sideways", 1)


def test_venue_of_never_reads_semantics_from_the_stem() -> None:
    """The EDM anchor stems are rotated relative to their contents, so the
    stem may decide the venue and nothing else."""
    assert venue_of("jedm_974_20260710_145529") == "JEDM"
    assert venue_of("jla_9099_20260710_163857") == "JLA"
    assert venue_of("dkt_interpretability_20260703_150612") == "EDM"


def test_count_and_bundle_helpers_on_a_synthetic_corpus() -> None:
    anchors = [
        Anchor("a_20260703_1", "EDM", "we propose a method", "we propose a method with shap"),
        Anchor("jedm_x", "JEDM", "human coders agreed", "human coders agreed, code on github.com"),
        Anchor("jla_y", "JLA", "nothing", "nothing"),
    ]
    tally = count(anchors, [r"\bshap\b"], "full_text")
    assert tally == {"EDM": 1, "JEDM": 0, "JLA": 0, "total": 1, "n": 3}
    bundles = bundle_sizes(
        anchors,
        {"method": [r"we propose"], "validation": [r"human coder"]},
        "abstract",
    )
    assert bundles["type_counts"] == {"method": 1, "validation": 1}
    assert bundles["distribution"] == {0: 1, 1: 2}
    assert bundles["at_least_two"] == 0


# --------------------------------------------------------------------------
# 3. Corpus reproduction (skipped when the anchor corpus is not present)
# --------------------------------------------------------------------------


def test_every_asserted_anchor_count_reproduces(table: dict) -> None:
    corpus_path = _corpus_path(table)
    if not corpus_path.is_dir():
        pytest.skip(f"anchor corpus not on this machine: {corpus_path}")
    anchors = load_corpus(corpus_path, (table["corpus"].get("excluded_dirs") or ()))
    audit = audit_table(table, anchors)
    assert audit.failures == [], [c["label"] for c in audit.failures]


def test_corpus_shape(table: dict) -> None:
    corpus_path = _corpus_path(table)
    if not corpus_path.is_dir():
        pytest.skip(f"anchor corpus not on this machine: {corpus_path}")
    anchors = load_corpus(corpus_path, (table["corpus"].get("excluded_dirs") or ()))
    assert len(anchors) == 34
    by_venue = {v: sum(1 for a in anchors if a.venue == v) for v in ("EDM", "JEDM", "JLA")}
    assert by_venue == {"EDM": 15, "JEDM": 10, "JLA": 9}
    # dedupe actually did something: the raw directory count is larger
    assert len(list(corpus_path.iterdir())) > len(anchors)


def test_editorial_is_excluded(table: dict) -> None:
    corpus_path = _corpus_path(table)
    if not corpus_path.is_dir():
        pytest.skip(f"anchor corpus not on this machine: {corpus_path}")
    anchors = load_corpus(corpus_path, (table["corpus"].get("excluded_dirs") or ()))
    assert not any(a.stem.startswith("jla_9743") for a in anchors)
