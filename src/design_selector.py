"""V3.2 Arc D — deterministic design-selection layer.

Given a dataset registry (carrying a ``design_feasibility`` block) and
an optional question intent, decide which identification strategies are
FEASIBLE on this dataset and rank them; every infeasible design gets an
explicit reason. No LLM in this layer — the predicates are pure data
checks. The LLM's role (via the ``design-selection-memo`` skill) is to
consume this report and argue the choice in ``research_spec.design_memo``;
it may not overrule a deterministic infeasibility.

Design vocabulary (extends as Arc Q lands): ``prediction``,
``causal_soo``, ``causal_itr``, ``rd``, ``iv``, ``did``.
Only the first three are currently EXECUTABLE task types; rd/iv/did
predicates exist now so the selector can say "infeasible on this
dataset, here's why" (the roadmap's honesty-is-a-feature rule) and so
Arc Q can flip them on by populating the registry block.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Intents the question classifier recognizes (deterministic keywords;
# the caller may also pass an explicit intent).
_TARGETING_KEYWORDS = (
    "for whom", "targeting", "treatment rule", "who should",
    "personalized", "individualized", "regime",
)
_CAUSAL_KEYWORDS = (
    "effect of", "causal", "impact of", "att", "ate", "treatment",
    "intervention", "does raising", "consequence of",
)

# Keyword words at least this long may match a morphological tail
# ("causal" -> "causally", "treatment rule" -> "treatment rules"). The
# 3-letter estimand acronyms "att" and "ate" fall below the threshold and
# match strictly, which is the whole point of the word-boundary fix: bare
# substring matching classified *attitudes*, *attainment*, *attendance*,
# *climate* and *estimate* as causal questions.
_SUFFIX_TAIL_MIN_LEN = 4


def _keyword_to_pattern(keyword: str) -> re.Pattern[str]:
    """Compile one intent keyword into a word-boundary regex."""
    parts: list[str] = []
    for word in keyword.split():
        escaped = re.escape(word)
        if len(word) >= _SUFFIX_TAIL_MIN_LEN and word.isalpha():
            parts.append(rf"\b{escaped}[a-z]*\b")
        else:
            parts.append(rf"\b{escaped}\b")
    return re.compile(r"\s+".join(parts))


_TARGETING_PATTERNS = tuple(_keyword_to_pattern(k) for k in _TARGETING_KEYWORDS)
_CAUSAL_PATTERNS = tuple(_keyword_to_pattern(k) for k in _CAUSAL_KEYWORDS)


@dataclass
class DesignVerdict:
    design: str
    feasible: bool
    reasons: list[str] = field(default_factory=list)
    executable_task_type: str | None = None  # None => not yet implemented


def _block(registry: dict) -> dict:
    return (registry or {}).get("design_feasibility") or {}


def rd_feasible(registry: dict) -> DesignVerdict:
    rv = _block(registry).get("running_variables") or []
    usable = [r for r in rv if isinstance(r, dict) and r.get("cutoff") is not None]
    if usable:
        return DesignVerdict("rd", True, [
            f"running variable(s) with documented cutoffs: "
            f"{[r.get('name') for r in usable]}"
        ], executable_task_type=None)
    return DesignVerdict("rd", False, [
        "no running variable with a documented cutoff in the registry's "
        "design_feasibility block (public-use files typically coarsen or "
        "suppress assignment variables)"
    ])


def iv_feasible(registry: dict) -> DesignVerdict:
    instruments = _block(registry).get("candidate_instruments") or []
    justified = [
        i for i in instruments
        if isinstance(i, dict) and i.get("exclusion_restriction_justification")
    ]
    if justified:
        return DesignVerdict("iv", True, [
            f"curated instrument(s) with written exclusion-restriction "
            f"justifications: {[i.get('name') for i in justified]}"
        ], executable_task_type=None)
    return DesignVerdict("iv", False, [
        "no curated instrument with a written exclusion-restriction "
        "justification — IV feasibility is an argumentation question, and "
        "the registry carries none for this dataset"
    ])


def did_feasible(registry: dict) -> DesignVerdict:
    """DiD feasibility.

    Executable ONLY when the dataset itself carries the two-by-two
    structure: either it IS a harmonized multi-cohort panel
    (``design_feasibility.panel_ready``) or it declares policy-timing
    variables. A ``multi_cohort_partner`` pointer is a *harmonization
    lead*, not a runnable design — both live DiD runs
    (``runs/phase_b_did_20260704``, ``runs/stream1_did_v2_20260708``)
    executed on ``did_els_hsls_panel``, never on the cohort registries
    that name each other as partners.
    """
    b = _block(registry)
    panel_ready = bool(b.get("panel_ready"))
    timing = b.get("policy_timing_variables") or []
    partner = b.get("multi_cohort_partner")

    if panel_ready or timing:
        reasons: list[str] = []
        if panel_ready:
            reasons.append(
                "design_feasibility.panel_ready is true — this dataset is "
                "itself a harmonized multi-cohort panel"
            )
        if timing:
            reasons.append(
                f"documented policy-timing variable(s): {list(timing)}"
            )
        return DesignVerdict("did", True, reasons,
                             executable_task_type="causal_did")

    if partner:
        return DesignVerdict("did", True, [
            f"a partner cohort ({partner}) is documented, so DiD is "
            "data-feasible in principle — but this registry is neither a "
            "harmonized panel (design_feasibility.panel_ready) nor does it "
            "declare policy-timing variables, so causal_did is NOT "
            "executable on this dataset as-is; it must first be harmonized "
            "onto a panel dataset (see did_els_hsls_panel)"
        ], executable_task_type=None)

    return DesignVerdict("did", False, [
        "single-cohort dataset with no documented policy-timing variation; "
        "DiD needs cross-cohort or policy-shock structure (roadmap pairs "
        "this with the ELS:2002 onboarding)"
    ])


def soo_feasible(registry: dict) -> DesignVerdict:
    # SOO is feasible whenever the registry documents a rich pre-treatment
    # covariate set — which the Tier-1 registry structure itself attests.
    has_predictors = bool(((registry or {}).get("variables") or {}).get("predictors"))
    if has_predictors:
        return DesignVerdict("causal_soo", True, [
            "rich pre-treatment covariate registry supports "
            "selection-on-observables (with the usual no-unmeasured-"
            "confounding caveat)"
        ], executable_task_type="causal_soo")
    return DesignVerdict("causal_soo", False, ["registry lists no predictors"])


def itr_feasible(registry: dict) -> DesignVerdict:
    """ITR feasibility = SOO-feasible AND curator-declared ``itr_ready``.

    ``design_feasibility.itr_ready`` is written by
    ``scripts/onboard_dataset.py`` and was previously never read, so
    every dataset with any predictor list reported ITR-feasible —
    including ``assistments_0910``, which declares ``itr_ready: false``
    and carries zero protected attributes.
    """
    base = soo_feasible(registry)
    if not base.feasible:
        return DesignVerdict("causal_itr", False, base.reasons)

    b = _block(registry)
    if "itr_ready" not in b:
        return DesignVerdict("causal_itr", False, [
            "registry's design_feasibility block does not declare "
            "itr_ready; ITR readiness is a curation decision (covariates "
            "observable at decision time + a certified M6/M7 battery) and "
            "is never inferred from the covariate list alone"
        ])
    if not b.get("itr_ready"):
        return DesignVerdict("causal_itr", False, [
            "design_feasibility.itr_ready is false for this dataset — a "
            "rich covariate set alone does not make a learned treatment "
            "rule estimable here"
        ])

    return DesignVerdict("causal_itr", True, [
        "SOO-feasible and design_feasibility.itr_ready is true — registry "
        "covariates provide decision-time rule candidates (M6/M7 battery, "
        "synthetic-gate certified)"
    ], executable_task_type="causal_itr")


def classify_intent(question: str | None) -> str:
    """Classify a research question as prediction | causal | targeting.

    Matching is word-boundary anchored: bare substring matching made
    *attitudes*, *attainment*, *attendance* ("att") and *climate*,
    *estimate* ("ate") read as causal-intent questions.
    """
    q = (question or "").lower()
    if any(p.search(q) for p in _TARGETING_PATTERNS):
        return "targeting"
    if any(p.search(q) for p in _CAUSAL_PATTERNS):
        return "causal"
    return "prediction"


def select_design(
    registry: dict,
    question: str | None = None,
    intent: str | None = None,
) -> dict:
    """Return the full feasibility report + a recommendation.

    ``intent`` overrides keyword classification when provided
    ("prediction" | "causal" | "targeting").
    """
    intent = intent or classify_intent(question)
    verdicts = {
        v.design: v
        for v in (
            soo_feasible(registry),
            itr_feasible(registry),
            rd_feasible(registry),
            iv_feasible(registry),
            did_feasible(registry),
        )
    }

    if intent == "prediction":
        recommended = "prediction"
        rationale = "question intent is predictive; supervised ML applies"
    elif intent == "targeting":
        recommended = (
            "causal_itr" if verdicts["causal_itr"].feasible else "causal_soo"
        )
        rationale = "targeting question → learned-rule estimand (M6/M7)"
    else:  # causal average-effect intent
        # Prefer stronger designs when the registry supports them;
        # fall back to SOO with the explicit statement of why the
        # stronger designs are infeasible.
        for stronger in ("rd", "iv", "did"):
            if verdicts[stronger].feasible and verdicts[stronger].executable_task_type:
                recommended = verdicts[stronger].executable_task_type
                rationale = f"{stronger} feasible and executable"
                break
        else:
            recommended = "causal_soo"
            unavailable = [
                f"{d}: {verdicts[d].reasons[0]}" for d in ("rd", "iv", "did")
            ]
            rationale = (
                "no stronger quasi-experimental design is both feasible and "
                "executable on this dataset — " + "; ".join(unavailable)
            )

    return {
        "intent": intent,
        "recommended_task_type": recommended,
        "rationale": rationale,
        "verdicts": {
            d: {
                "feasible": v.feasible,
                "executable_task_type": v.executable_task_type,
                "reasons": v.reasons,
            }
            for d, v in verdicts.items()
        },
    }


def format_design_report(report: dict) -> str:
    """Render the selector report for injection into the PF user message."""
    lines = [
        "## Design Feasibility Report (deterministic — may not be overruled)",
        f"- Question intent: {report['intent']}",
        f"- Recommended task type: **{report['recommended_task_type']}**"
        f" ({report['rationale']})",
        "- Per-design verdicts:",
    ]
    for design, v in report["verdicts"].items():
        if not v["feasible"]:
            status = "infeasible"
        elif v["executable_task_type"]:
            status = "FEASIBLE"
        else:
            # New state introduced with the did_feasible fix: the design
            # is defensible on the data but no executable task type can
            # run it on THIS dataset.
            status = "FEASIBLE but NOT EXECUTABLE here"
        lines.append(f"  - {design}: {status} — {v['reasons'][0]}")
    lines.append(
        "Include a `design_memo` object in research_spec: "
        '{"chosen_design": ..., "feasibility_evidence": ..., '
        '"rejected_alternatives": [{"design": ..., "reason": ...}]} '
        "consistent with this report."
    )
    return "\n".join(lines)
