"""Arc T / T0 + H8 - deterministic venue-fit scoring.

A rule table, not a judge. Two table schemas are supported:

* **v1** (``data_registry/venue_fit_rules.yaml``): each rule keys a named
  Python predicate implemented here (``_PREDICATES``), so the YAML never
  carries executable expressions.
* **v2** (``data_registry/venue_fit_rules_v2.yaml``, blind-derived): each
  rule carries a *declarative* predicate - a closed clause enum
  (``field_regex`` / ``task_type_in`` / ``dataset_in``, composed by
  ``any_of`` / ``all_of`` / ``none_of``) evaluated by
  :func:`evaluate_predicate`. The clause language is data, never eval'd.
  The evaluator is a port of the reference implementation in
  ``scripts/derive_venue_rules.py::evaluate_predicate`` (H8: src must not
  import scripts), plus a de-hyphenation pass on text blobs - PDF
  line-wrap artifacts like ``"shap- ing"`` produced false hits during
  derivation.

No LLM, no network. Deterministic given (spec, card, venue).

The load-bearing test is the pre-registered pair from spec sec. 6 V2: the
bare 2x2 DiD (``phase_b_did_20260704``, gate 3.7 Reject) must score
strictly below the same estimand on the same data wrapped in M9/M10
(``stream1_did_v2_20260708``, gate 7.0 Accept). If VF-01/VF-04 cannot
separate that pair, the table is not encoding what it claims to.

ROUTING HOOK (H2 Option C, owner decision 2026-08-06)
-----------------------------------------------------
Under dual-targeting (the v5 capability roadmap (internal) sec. 5.3),
VF-01 / VF2-01 (observational causal) and VF2-02 (national-survey
extract) are venue facts, not idea defects. When the caller passes
``routing_family="policy-causal"`` (from
``src.ideation.venue_router.route_idea``), those hits are recorded with
``role="routing_signal"`` and a delta of 0.0: they stay visible as
evidence but no longer penalize the idea. Routed to the computational-EDM
family (or with no routing at all) they remain penalties - that is what
dual-targeting means.
"""
from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

DEFAULT_RULES_PATH = Path("data_registry") / "venue_fit_rules.yaml"
V2_RULES_PATH = Path("data_registry") / "venue_fit_rules_v2.yaml"

# Venue families under H2 Option C dual-targeting. Defined here (not in
# venue_router) because venue_router imports this module and the scorer
# needs the policy family name for the routing hook.
FAMILY_COMPUTATIONAL = "computational-edm"
FAMILY_POLICY = "policy-causal"

#: Rule codes that are ROUTING SIGNALS, not idea penalties, when the
#: destination family is policy-causal. VF-01 is the v1 twin of VF2-01.
ROUTING_SIGNAL_CODES: tuple[str, ...] = ("VF-01", "VF2-01", "VF2-02")

# Keyword sets. Deliberately narrow; each entry is a phrase whose
# presence is a claim, not a topic.
#
# "cross-cohort" is NOT a transfer keyword: on causal_did it describes
# the design itself, and both shipped DiD specs contain it - including
# it would give the bare-DiD spec a transfer contribution and destroy
# the pre-registered pair separation.
_TRANSFER_KEYWORDS = (
    "generaliz",
    "transfer",
    "replicat",
    "external validation",
    "another dataset",
    "second dataset",
    "out-of-sample context",
    "cross-context",
    "cross-dataset",
    "portability",
)
_SHAP_KEYWORDS = ("shap", "feature importance", "variable importance")
_DECISION_KEYWORDS = (
    "placement",
    "screening",
    "advising",
    "admission",
    "intervention assignment",
    "institutional decision",
    "decision rule",
    "flagging",
    "targeting decision",
    "who to",
)
_SYNTHETIC_KEYWORDS = ("synthetic data", "simulated data", "synthetic dgp")

# Methods that constitute a *second* methodological contribution rather
# than the standard battery. M5 (causal forest CATE) is deliberately
# excluded: it appears as a secondary method in most archived causal
# specs, so counting it would make VF-01 unfireable.
_SECOND_LAYER_METHODS = {"M9", "M10"}


@dataclass(frozen=True)
class VenueFitHit:
    code: str
    delta: float
    summary: str
    evidence: str  # C2 - the anchor fact behind the rule
    why: str  # what in THIS artifact triggered it
    #: "scored" for a hit contributing its delta; "routing_signal" for a
    #: hit reclassified by the H2 Option C routing hook (delta 0.0, kept
    #: visible as evidence of where the idea belongs).
    role: str = "scored"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VenueFitReport:
    score: float
    venue: str
    hits: list[VenueFitHit] = field(default_factory=list)
    facts: dict = field(default_factory=dict)

    @property
    def codes(self) -> list[str]:
        return [h.code for h in self.hits]

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "venue": self.venue,
            "hits": [h.to_dict() for h in self.hits],
            "facts": self.facts,
        }

    def render(self) -> str:
        lines = [f"venue_fit({self.venue}) = {self.score:+.2f}"]
        for hit in self.hits:
            lines.append(f"  {hit.code} {hit.delta:+.2f}  {hit.summary}")
            lines.append(f"      why: {hit.why}")
            lines.append(f"      evidence: {hit.evidence}")
        return "\n".join(lines)


def load_rules(path: str | os.PathLike[str] | None = None) -> dict:
    rules_path = Path(path) if path is not None else DEFAULT_RULES_PATH
    with open(rules_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# --------------------------------------------------------------------------
# Text normalization
# --------------------------------------------------------------------------

# A word character, a hyphen, whitespace, a word character: the signature
# of a PDF line-wrap hyphenation ("shap-\ning" -> "shap- ing" after
# whitespace collapse). Rejoining kills the false hit that pattern
# produced during derivation (H2 roadmap sec. 1, VF2-03 row: two
# "shap- ing" artifacts in the AERA Open counter-corpus). Deliberate
# hyphenated compounds ("difference-in-differences", "cross-context")
# carry no whitespace after the hyphen and are untouched.
_DEHYPHENATION_RE = re.compile(r"(?<=\w)-\s+(?=\w)")


def dehyphenate(text: str) -> str:
    """Rejoin PDF line-wrap artifacts: ``"shap- ing"`` -> ``"shaping"``."""
    return _DEHYPHENATION_RE.sub("", text)


def _normalize_blob(text: str) -> str:
    """De-hyphenate, collapse whitespace, lowercase. Applied to every
    text blob BEFORE any keyword or regex matching."""
    return re.sub(r"\s+", " ", dehyphenate(text)).lower()


# --------------------------------------------------------------------------
# Fact extraction
# --------------------------------------------------------------------------


def _text_blob(spec: dict, card: dict | None = None) -> str:
    parts = [
        str(spec.get("research_question") or ""),
        str(spec.get("expected_contribution") or ""),
        str(spec.get("target_population") or ""),
        str(spec.get("rationale_for_method_set") or ""),
    ]
    memo = spec.get("design_memo")
    if isinstance(memo, dict):
        parts.extend(str(memo.get(k) or "") for k in ("chosen_design", "feasibility_evidence"))
    if card:
        parts.extend(
            str(card.get(k) or "")
            for k in (
                "research_question",
                "why_it_matters",
                "what_we_would_do",
                "what_counts_as_the_result",
            )
        )
    return _normalize_blob(" ".join(parts))


def _methods(spec: dict) -> set[str]:
    out: set[str] = set()
    for key in ("primary_method", "comparator_method"):
        value = spec.get(key)
        if isinstance(value, str):
            out.add(value.strip().upper())
    for key in ("secondary_methods", "method_battery", "methods"):
        for value in spec.get(key) or []:
            if isinstance(value, str):
                out.add(value.strip().upper())
    return out


def _subgroup_names(spec: dict) -> list[str]:
    names: list[str] = []
    for key in ("subgroup_analyses", "grouping_vars", "heterogeneity_subgroups"):
        for value in spec.get(key) or []:
            if isinstance(value, str):
                names.append(value)
    return names


def extract_facts(spec: dict, card: dict | None = None) -> dict:
    """Deterministic facts the predicates read. Auditable on its own."""
    blob = _text_blob(spec, card)
    methods = _methods(spec)
    heterogeneity = [
        v for v in (spec.get("heterogeneity_subgroups") or []) if isinstance(v, str)
    ]
    subgroups = _subgroup_names(spec)

    reasons: list[str] = []
    declared = (card or {}).get("second_contribution") or spec.get("second_contribution")
    if isinstance(declared, str) and declared.strip():
        reasons.append(declared.strip().lower())

    # fairness: a genuine second contribution, not the routine single
    # subgroup column every spec carries.
    if heterogeneity or len(set(subgroups)) >= 2:
        reasons.append("fairness")
    if any(k in blob for k in _TRANSFER_KEYWORDS):
        reasons.append("transfer")
    if methods & _SECOND_LAYER_METHODS:
        reasons.append("method_second_layer")
    if spec.get("task_type") == "psychometrics" and any(
        k in blob for k in _DECISION_KEYWORDS
    ):
        reasons.append("measurement")

    return {
        "task_type": spec.get("task_type") or "prediction",
        "methods": sorted(methods),
        "subgroup_variables": sorted(set(subgroups)),
        "heterogeneity_subgroups": heterogeneity,
        "second_contribution_reasons": sorted(set(reasons)),
        "shap_terms": [k for k in _SHAP_KEYWORDS if k in blob],
        "transfer_terms": [k for k in _TRANSFER_KEYWORDS if k in blob],
        "decision_terms": [k for k in _DECISION_KEYWORDS if k in blob],
        "synthetic_terms": [k for k in _SYNTHETIC_KEYWORDS if k in blob],
        "dataset": spec.get("dataset") or (card or {}).get("cell", {}).get("dataset"),
    }


# --------------------------------------------------------------------------
# Predicates - return (fired, why) ; `why` is the artifact-side evidence
# --------------------------------------------------------------------------

Predicate = Callable[[dict, dict], "tuple[bool, str]"]


def _p_bare_observational_causal(facts: dict, spec: dict) -> tuple[bool, str]:
    if not str(facts["task_type"]).startswith("causal"):
        return False, ""
    reasons = facts["second_contribution_reasons"]
    if reasons:
        return False, ""
    return True, (
        f"task_type={facts['task_type']} with methods {facts['methods']} and "
        f"no second contribution (subgroup vars: "
        f"{facts['subgroup_variables'] or 'none'}, heterogeneity: "
        f"{facts['heterogeneity_subgroups'] or 'none'})"
    )


def _p_shap_headline(facts: dict, spec: dict) -> tuple[bool, str]:
    terms = facts["shap_terms"]
    if not terms:
        return False, ""
    return True, f"contribution text names {terms}"


def _p_auc_only_prediction(facts: dict, spec: dict) -> tuple[bool, str]:
    if facts["task_type"] != "prediction":
        return False, ""
    if facts["second_contribution_reasons"]:
        return False, ""
    return True, (
        "prediction study with no transfer / fairness / measurement / "
        "replication second contribution"
    )


def _p_second_contribution(facts: dict, spec: dict) -> tuple[bool, str]:
    reasons = facts["second_contribution_reasons"]
    if not reasons:
        return False, ""
    return True, f"second contribution(s): {reasons}"


def _p_transfer_claim(facts: dict, spec: dict) -> tuple[bool, str]:
    terms = facts["transfer_terms"]
    if not terms:
        return False, ""
    return True, f"transfer/generalizability language: {terms}"


def _p_measurement_to_decision(facts: dict, spec: dict) -> tuple[bool, str]:
    if facts["task_type"] != "psychometrics":
        return False, ""
    terms = facts["decision_terms"]
    if not terms:
        return False, ""
    return True, f"measurement study naming a downstream decision: {terms}"


def _p_synthetic_only(facts: dict, spec: dict) -> tuple[bool, str]:
    terms = facts["synthetic_terms"]
    if not terms:
        return False, ""
    if facts["dataset"]:
        return False, ""
    return True, (
        f"synthetic-only empirical support ({terms}) with no dataset declared"
    )


_PREDICATES: dict[str, Predicate] = {
    "bare_observational_causal": _p_bare_observational_causal,
    "shap_headline": _p_shap_headline,
    "auc_only_prediction": _p_auc_only_prediction,
    "second_contribution": _p_second_contribution,
    "transfer_claim": _p_transfer_claim,
    "measurement_to_decision": _p_measurement_to_decision,
    "synthetic_only": _p_synthetic_only,
}


# --------------------------------------------------------------------------
# v2 declarative clause evaluator (H8)
#
# Port of scripts/derive_venue_rules.py::evaluate_predicate - src must not
# import scripts/, and tests/test_venue_rules_v2.py pins the reference
# semantics. One deliberate addition over the reference: text is
# de-hyphenated before matching (see ``dehyphenate``).
#
# The clause language is a closed enum. Nothing in the YAML is ever
# eval'd; regex strings are data.
#
#   {kind: any_of|all_of|none_of, clauses: [...]}
#   {kind: field_regex, fields: [...], patterns: [...]}
#   {kind: task_type_in, values: [...]}
#   {kind: dataset_in,   values: [...]}
# --------------------------------------------------------------------------

_COMPOSITE_KINDS = ("any_of", "all_of", "none_of")
_LEAF_KINDS = ("field_regex", "task_type_in", "dataset_in")


def _card_field(card: dict, spec: dict, name: str) -> str:
    for source in (card, spec):
        value = source.get(name)
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)):
            return " ".join(str(v) for v in value)
    return ""


def resolve_task_type(card: dict, spec: dict) -> str:
    cell = card.get("cell")
    if isinstance(cell, dict) and cell.get("task_type"):
        return str(cell["task_type"])
    return str(card.get("task_type") or spec.get("task_type") or "")


def resolve_dataset(card: dict, spec: dict) -> str:
    cell = card.get("cell")
    if isinstance(cell, dict) and cell.get("dataset"):
        return str(cell["dataset"])
    return str(card.get("dataset") or spec.get("dataset") or "")


def clause_is_known(clause: Any) -> bool:
    """True when every ``kind`` in the clause tree is in the closed enum.

    Used by the loud-failure guard to decide whether a v2-style rule is
    evaluable at all; :func:`evaluate_predicate` still raises on the
    first unknown kind it reaches, so an unknown vocabulary can never
    silently score 0.0.
    """
    if not isinstance(clause, dict):
        return False
    kind = str(clause.get("kind", ""))
    if kind in _COMPOSITE_KINDS:
        return all(clause_is_known(c) for c in clause.get("clauses") or [])
    return kind in _LEAF_KINDS


def evaluate_predicate(
    clause: dict, card: dict, spec: dict | None = None
) -> tuple[bool, str]:
    """Evaluate one declarative clause. Returns ``(fired, why)``.

    ``why`` is the artifact-side evidence string required by design
    commitment C2: it names the fact in THIS card that made the clause
    fire. Raises ``ValueError`` on an unknown clause kind - the loud
    failure that replaced the silent-zero defect.
    """
    spec = spec or {}
    kind = str(clause.get("kind", ""))

    if kind in _COMPOSITE_KINDS:
        results = [
            evaluate_predicate(c, card, spec) for c in clause.get("clauses") or []
        ]
        fired_flags = [r[0] for r in results]
        reasons = [r[1] for r in results if r[0]]
        if kind == "any_of":
            return (any(fired_flags), "; ".join(reasons))
        if kind == "all_of":
            if fired_flags and all(fired_flags):
                return True, "; ".join(reasons)
            return False, ""
        return (not any(fired_flags), "no clause matched")

    if kind == "field_regex":
        fields = clause.get("fields") or []
        patterns = clause.get("patterns") or []
        for name in fields:
            text = _normalize_blob(_card_field(card, spec, str(name)))
            if not text:
                continue
            for pattern in patterns:
                found = re.search(pattern, text)
                if found:
                    return True, f"{name} matches /{pattern}/ at {found.group(0)!r}"
        return False, ""

    if kind == "task_type_in":
        task_type = resolve_task_type(card, spec)
        if task_type in {str(v) for v in clause.get("values") or []}:
            return True, f"task_type={task_type}"
        return False, ""

    if kind == "dataset_in":
        dataset = resolve_dataset(card, spec)
        if dataset in {str(v) for v in clause.get("values") or []}:
            return True, f"dataset={dataset}"
        return False, ""

    raise ValueError(f"unknown predicate clause kind: {kind!r}")


def _predicate_label(declared: Any) -> str:
    """Printable name for a rule's predicate, for the guard's error."""
    if isinstance(declared, dict):
        return f"clause kind {declared.get('kind')!r}"
    return repr(declared)


def score_venue_fit(
    spec: dict,
    *,
    venue: str | None = None,
    card: dict | None = None,
    rules_path: str | os.PathLike[str] | None = None,
    rules: dict | None = None,
    routing_family: str | None = None,
) -> VenueFitReport:
    """Deterministic venue-fit score for a spec (or a spec + idea card).

    Handles both table schemas: v1 rules name a Python predicate in
    ``_PREDICATES``; v2 rules carry a declarative clause evaluated by
    :func:`evaluate_predicate`.

    ``routing_family`` is the H2 Option C routing hook: pass the family
    from ``venue_router.route_idea``. When it is ``FAMILY_POLICY``, hits
    whose code is in :data:`ROUTING_SIGNAL_CODES` are recorded with
    ``role="routing_signal"`` and delta 0.0 - routing evidence, not a
    penalty. Any other family (or None) leaves them as penalties.
    """
    table = rules if rules is not None else load_rules(rules_path)
    venues = table.get("venues") or {}
    venue_name = venue or table.get("default_venue") or "EDM"
    venue_cfg = venues.get(venue_name) or {}
    ethics_weight = float(venue_cfg.get("ethics_weight", 1.0))

    spec = spec if isinstance(spec, dict) else {}
    card_dict = card if isinstance(card, dict) else {}
    facts = extract_facts(spec, card)
    facts["venue"] = venue_name
    facts["ethics_weight"] = ethics_weight
    facts["routing_family"] = routing_family

    hits: list[VenueFitHit] = []
    score = 0.0
    # LOUD-FAILURE GUARD. A table whose predicates this module cannot
    # evaluate would score 0.0 with no hits and no error --
    # indistinguishable from 'nothing fired'. Pointing --venue-rules at a
    # table built for an unknown predicate vocabulary must therefore
    # raise, not yield a silently INERT venue-fit term. A rule is
    # evaluable iff its predicate is a name in ``_PREDICATES`` (v1) or a
    # declarative clause whose kinds are all in the closed enum (v2).
    _declared = [r.get("predicate") for r in (table.get("rules") or [])]
    _known = [
        d
        for d in _declared
        if (isinstance(d, str) and d in _PREDICATES)
        or (isinstance(d, dict) and clause_is_known(d))
    ]
    if _declared and not _known:
        labels = sorted({_predicate_label(d) for d in _declared})[:5]
        raise ValueError(
            "venue-fit rule table declares %d rules but NONE of its "
            "predicates are implemented here (%s). This would score every "
            "candidate 0.0 silently. Implement the predicates or point at "
            "a compatible table." % (len(_declared), labels)
        )
    for rule in table.get("rules") or []:
        code = str(rule.get("code"))
        declared = rule.get("predicate")
        applies_to = rule.get("applies_to") or []
        if isinstance(declared, dict):
            # v2 declarative rule. Task type resolves card-first (cell),
            # matching the reference implementation; an unknown clause
            # kind raises inside evaluate_predicate.
            task_type = resolve_task_type(card_dict, spec)
            if applies_to and task_type and task_type not in applies_to:
                continue
            fired, why = evaluate_predicate(declared, card_dict, spec)
            if not fired:
                continue
            delta = float(
                (rule.get("venue_delta") or {}).get(venue_name, rule.get("delta", 0.0))
            )
        else:
            predicate = _PREDICATES.get(str(declared))
            if predicate is None:
                continue
            if applies_to and facts["task_type"] not in applies_to:
                continue
            fired, why = predicate(facts, spec)
            if not fired:
                continue
            delta = float(rule.get("delta", 0.0))
            weighted_reasons = set(rule.get("ethics_weighted_reasons") or [])
            if weighted_reasons and delta > 0:
                fairness_only = (
                    set(facts["second_contribution_reasons"]) <= weighted_reasons
                )
                if fairness_only and ethics_weight != 1.0:
                    delta *= ethics_weight
                    why += (
                        f"; VF-08 ethics multiplier {ethics_weight} applied for "
                        f"venue {venue_name}"
                    )
        role = "scored"
        if (
            routing_family == FAMILY_POLICY
            and code in ROUTING_SIGNAL_CODES
            and delta < 0
        ):
            # H2 Option C: at a policy-causal destination this fact says
            # where the idea goes, not whether it is good.
            role = "routing_signal"
            why += (
                "; reclassified as a ROUTING SIGNAL (destination family "
                f"{FAMILY_POLICY}): delta suppressed from "
                f"{delta:+.2f} to +0.00 under Option C dual-targeting "
                "(the v5 capability roadmap (internal) sec. 5.3)"
            )
            delta = 0.0
        hits.append(
            VenueFitHit(
                code=code,
                delta=delta,
                summary=str(rule.get("summary") or "").strip(),
                evidence=str(rule.get("evidence") or "").strip(),
                why=why,
                role=role,
            )
        )
        score += delta

    return VenueFitReport(score=score, venue=venue_name, hits=hits, facts=facts)
