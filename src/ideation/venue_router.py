"""Arc T / H2 Option C - deterministic venue routing (dual-target).

Owner decision (2026-08-06, docs/v5_arc_t_h2_capability_roadmap.md
sec. 5.3, Option C): EDM-ARS dual-targets two venue families.

* ``computational-edm`` - EDM / JEDM / JLA. Measurement, psychometrics,
  and prediction-METHOD work stays here.
* ``policy-causal`` - AERA Open first (open access). Observational
  causal work and population-level national-survey work routes here.

Under this decision VF2-01 (observational causal estimand) and VF2-02
(national-survey extract) stop being flat idea penalties and become
ROUTING signals: they say where an idea goes, not whether it is good.
When an idea is routed to ``computational-edm`` they remain penalties in
the venue-fit score - that is what dual-targeting means. The score-side
half of this contract lives in ``venue_fit.score_venue_fit``'s
``routing_family`` hook.

Everything here is deterministic: no LLM, no network, no randomness.
Every routing verdict carries the artifact-side fact that produced it
(design commitment C2) plus the anchor/counter-corpus evidence behind
the rule.

CALIBRATION HONESTY
-------------------
AERA_OPEN has NO LSAR calibration (no anchor corpus, no P25 gate - H2
roadmap sec. 6 step 6 is the plan to build one). Every policy-routed
idea therefore carries ``gate_status: "advisory-uncalibrated-venue"`` so
no caller believes a calibrated gate exists there yet. EDM / JEDM / JLA
carry LSAR P25 calibrations (EDM 6.3 in
``LSAR/calibration/anchors_edm.yaml``; JEDM 5.15; JLA 5.4) and are
marked ``"calibrated"``.
"""
from __future__ import annotations

from typing import Any

from src.ideation import venue_fit as _vf

__all__ = [
    "CALIBRATED_VENUES",
    "CAUSAL_TASK_TYPES",
    "DEFAULT_COMPUTATIONAL_VENUE",
    "DEFAULT_POLICY_VENUE",
    "DEFAULT_PSYCHOMETRICS_VENUE",
    "FAMILY_COMPUTATIONAL",
    "FAMILY_POLICY",
    "GATE_STATUS_CALIBRATED",
    "GATE_STATUS_UNCALIBRATED",
    "NATIONAL_SURVEY_DATASETS",
    "route_idea",
]

FAMILY_COMPUTATIONAL = _vf.FAMILY_COMPUTATIONAL
FAMILY_POLICY = _vf.FAMILY_POLICY

CAUSAL_TASK_TYPES = frozenset({"causal_soo", "causal_itr", "causal_did"})

#: Mirrors VF2-02's dataset_in clause (data_registry/venue_fit_rules_v2.yaml)
#: and design_feasibility across the registries: every one is a national
#: survey extract; assistments_0910 is the only non-survey dataset held.
NATIONAL_SURVEY_DATASETS = frozenset(
    {"hsls09_public", "els_2002", "did_els_hsls_panel"}
)

DEFAULT_POLICY_VENUE = "AERA_OPEN"
DEFAULT_COMPUTATIONAL_VENUE = "EDM"
DEFAULT_PSYCHOMETRICS_VENUE = "JEDM"

GATE_STATUS_CALIBRATED = "calibrated"
GATE_STATUS_UNCALIBRATED = "advisory-uncalibrated-venue"

#: Venues with a live LSAR P25 calibration. EDM: P25 6.3
#: (LSAR/calibration/anchors_edm.yaml, wired via config.yaml
#: review_gate.calibration_path). JEDM: P25 5.15, JLA: P25 5.4 (LSAR
#: v0.4.0 journal calibration, 2026-07-10). AERA_OPEN is deliberately
#: absent: no anchors have been fetched and no gate exists.
CALIBRATED_VENUES = frozenset({"EDM", "JEDM", "JLA"})

#: The card/spec text fields the routing clauses read. Must match the
#: ``card_text_fields`` anchor in data_registry/venue_fit_rules_v2.yaml;
#: a sync test enforces it.
CARD_TEXT_FIELDS: tuple[str, ...] = (
    "research_question",
    "why_it_matters",
    "what_we_would_do",
    "what_counts_as_the_result",
    "expected_contribution",
    "rationale_for_method_set",
)

#: VF2-01's declarative predicate, verbatim from
#: data_registry/venue_fit_rules_v2.yaml (a sync test asserts equality).
#: Fires on a causal task type OR on observational-causal estimator
#: language smuggled into any card text field. Anchor evidence: 0 of 34
#: anchors mention this estimator machinery anywhere in full text.
#: Counter-corpus evidence (H2 roadmap sec. 5.1): AERA Open random n=30
#: full texts, 9/30 = 30.0% fire the same detector.
OBSERVATIONAL_CAUSAL_PREDICATE: dict[str, Any] = {
    "kind": "any_of",
    "clauses": [
        {
            "kind": "task_type_in",
            "values": ["causal_soo", "causal_itr", "causal_did"],
        },
        {
            "kind": "field_regex",
            "fields": list(CARD_TEXT_FIELDS),
            "patterns": [
                "propensity score",
                "difference-in-differences",
                "difference in differences",
                "regression discontinuity",
                "instrumental variable",
                "doubly robust",
                "inverse probability weight",
                "average treatment effect",
                "causal effect of",
                "causal forest",
            ],
        },
    ],
}

#: VF2-02's declarative predicate, verbatim from the v2 table (sync test
#: asserts equality). Anchor evidence: 2/34 lexical, 1/34 genuine data
#: source. Counter-corpus evidence: AERA Open n=30, 15/30 = 50.0% fire;
#: 3/30 use HSLS:09 - our exact dataset.
NATIONAL_SURVEY_PREDICATE: dict[str, Any] = {
    "kind": "any_of",
    "clauses": [
        {
            "kind": "dataset_in",
            "values": ["hsls09_public", "els_2002", "did_els_hsls_panel"],
        },
        {
            "kind": "field_regex",
            "fields": list(CARD_TEXT_FIELDS),
            "patterns": [
                r"\bhsls\b",
                r"\bnels\b",
                r"\bpisa\b",
                r"\bnaep\b",
                r"\becls\b",
                "nationally representative",
                "education longitudinal study",
            ],
        },
    ],
}

#: Population-level description detector for the prediction branch.
#: Deliberately narrow: a prediction card only routes policy-causal when
#: it BOTH sits on a national survey AND frames its result as a
#: population-level description. Router-local, not from the v2 table.
POPULATION_DESCRIPTION_PREDICATE: dict[str, Any] = {
    "kind": "field_regex",
    "fields": list(CARD_TEXT_FIELDS),
    "patterns": [
        "population-level",
        "nationally representative",
        "national (estimate|portrait|landscape)",
        "prevalence",
        "descriptive (portrait|profile|account)",
        "how (common|widespread)",
    ],
}

_EVIDENCE_VF2_01 = (
    "anchors 0/34 full texts contain observational-causal estimator "
    "machinery; AERA Open counter-corpus (random n=30 full texts, seed "
    "42) 9/30 = 30.0% do (H2 roadmap sec. 1 + 5.1)"
)
_EVIDENCE_VF2_02 = (
    "anchors 2/34 lexical national-survey references (1/34 genuine data "
    "source); AERA Open counter-corpus n=30: 15/30 = 50.0%, and 3/30 use "
    "HSLS:09, our exact dataset (H2 roadmap sec. 1 + 5.1)"
)
_PROVENANCE = (
    "Option C dual-target routing, owner decision 2026-08-06 "
    "(docs/v5_arc_t_h2_capability_roadmap.md sec. 5.3)"
)


def _routing_cfg(config: dict | None) -> dict:
    ideation = (config or {}).get("ideation") or {}
    routing = ideation.get("routing")
    return routing if isinstance(routing, dict) else {}


def _gate_status(venue: str) -> str:
    return (
        GATE_STATUS_CALIBRATED
        if venue in CALIBRATED_VENUES
        else GATE_STATUS_UNCALIBRATED
    )


def route_idea(
    spec: dict,
    card: dict | None = None,
    *,
    config: dict | None = None,
) -> dict:
    """Deterministic venue routing for one idea. Returns the verdict dict.

    Shape::

        {
          "family": "computational-edm" | "policy-causal",
          "venue": str,
          "gate_status": "calibrated" | "advisory-uncalibrated-venue",
          "rule": str,             # which routing rule decided
          "task_type": str|None,
          "dataset": str|None,
          "signals": [ {code, role, why, evidence}, ... ],
          "evidence": str,         # C2: the facts behind THIS decision
          "provenance": str,
        }

    Rules, in order:

    * R1: causal task type OR VF2-01-style observational-causal estimand
      -> policy-causal (AERA_OPEN).
    * R2: psychometrics -> computational-edm (JEDM).
    * R3: prediction with a method / fairness / measurement / transfer
      contribution -> computational-edm.
    * R4: prediction that is a population-level description on a
      national survey -> policy-causal.
    * R5: undecidable -> computational-edm by default, and the evidence
      string says so.
    """
    spec = spec if isinstance(spec, dict) else {}
    card_dict = card if isinstance(card, dict) else {}
    cfg = _routing_cfg(config)
    policy_venue = str(cfg.get("policy_venue") or DEFAULT_POLICY_VENUE)
    computational_venue = str(
        cfg.get("computational_venue") or DEFAULT_COMPUTATIONAL_VENUE
    )
    psychometrics_venue = str(
        cfg.get("psychometrics_venue") or DEFAULT_PSYCHOMETRICS_VENUE
    )

    task_type = _vf.resolve_task_type(card_dict, spec)
    dataset = _vf.resolve_dataset(card_dict, spec)

    causal_fired, causal_why = _vf.evaluate_predicate(
        OBSERVATIONAL_CAUSAL_PREDICATE, card_dict, spec
    )
    survey_fired, survey_why = _vf.evaluate_predicate(
        NATIONAL_SURVEY_PREDICATE, card_dict, spec
    )

    rule: str
    family: str
    venue: str
    evidence: str

    if causal_fired:
        rule = "R1-observational-causal"
        family = FAMILY_POLICY
        venue = policy_venue
        evidence = (
            f"routed policy-causal by R1: {causal_why}. VF2-01 is a venue "
            f"fact, not an idea defect, under {_PROVENANCE}."
        )
    elif task_type == "psychometrics":
        rule = "R2-psychometrics"
        family = FAMILY_COMPUTATIONAL
        venue = psychometrics_venue
        evidence = (
            f"routed computational-edm by R2: task_type={task_type}. "
            "Measurement and psychometrics work stays with the "
            f"computational family ({_PROVENANCE})."
        )
    elif task_type == "prediction":
        facts = _vf.extract_facts(spec, card_dict or None)
        reasons = list(facts.get("second_contribution_reasons") or [])
        pop_fired, pop_why = _vf.evaluate_predicate(
            POPULATION_DESCRIPTION_PREDICATE, card_dict, spec
        )
        if reasons:
            rule = "R3-prediction-contribution"
            family = FAMILY_COMPUTATIONAL
            venue = computational_venue
            evidence = (
                "routed computational-edm by R3: prediction carrying a "
                f"contribution beyond accuracy ({reasons}). Method / "
                "fairness / measurement / transfer contributions stay "
                f"with the computational family ({_PROVENANCE})."
            )
        elif survey_fired and pop_fired:
            rule = "R4-population-description-national-survey"
            family = FAMILY_POLICY
            venue = policy_venue
            evidence = (
                "routed policy-causal by R4: population-level description "
                f"({pop_why}) on a national-survey extract ({survey_why}). "
                f"{_PROVENANCE}."
            )
        else:
            rule = "R5-default-undecidable"
            family = FAMILY_COMPUTATIONAL
            venue = computational_venue
            evidence = (
                "UNDECIDABLE on the deterministic facts: prediction with "
                "no method/fairness/measurement/transfer contribution "
                "detected and no population-description-on-national-survey "
                "signal (survey_fired="
                f"{survey_fired}, population_description={pop_fired}). "
                "Defaulting to computational-edm per the routing rule; "
                "this default is stated, not silent."
            )
    else:
        rule = "R5-default-undecidable"
        family = FAMILY_COMPUTATIONAL
        venue = computational_venue
        evidence = (
            f"UNDECIDABLE: task_type={task_type or 'MISSING'} matches no "
            "routing rule. Defaulting to computational-edm; this default "
            "is stated, not silent."
        )

    # VF2-01/02 firings always appear in signals. Their role depends on
    # the destination: routing evidence at policy-causal, penalty at
    # computational-edm (dual-targeting).
    signal_role = "routing" if family == FAMILY_POLICY else "penalty"
    signals: list[dict] = []
    if causal_fired:
        signals.append(
            {
                "code": "VF2-01",
                "role": signal_role,
                "why": causal_why,
                "evidence": _EVIDENCE_VF2_01,
            }
        )
    if survey_fired:
        signals.append(
            {
                "code": "VF2-02",
                "role": signal_role,
                "why": survey_why,
                "evidence": _EVIDENCE_VF2_02,
            }
        )

    return {
        "family": family,
        "venue": venue,
        "gate_status": _gate_status(venue),
        "rule": rule,
        "task_type": task_type or None,
        "dataset": dataset or None,
        "signals": signals,
        "evidence": evidence,
        "provenance": _PROVENANCE,
    }
