"""Deterministic Critic verdict evaluator (Phase 3b.10 / §10.3).

Background. Pre-3b.10, the orchestrator trusted the Critic LLM's
self-reported ``overall_verdict`` field. The 3b.5 report flagged
F-CRITIC-PASSED-WITH-LOW-SCORE: Critic emitted verdict=PASS with
quality_score=5 and 1 critical + 1 major issue, despite the
documented thresholds requiring quality≥7 and zero critical issues
for PASS. 3b.9 confirmed the recurrence: cycle-1 PASS with
quality_score=6, 1 critical, 1 major issue, while cycles=max — paper
advanced to Writer without UNVERIFIED flagging.

Fix (failure mode (a) per the 3b.10 hand-off): enforce thresholds at
the evaluator regardless of LLM-reported verdict. The LLM's verdict
is a cross-check; if it disagrees with the deterministic evaluator,
log a WARNING but use the deterministic outcome.

Documented thresholds (per agent_prompts/critic.yaml lines 165–166):

  PASS:   No critical issues; ≤ 2 major issues; overall_quality_score ≥ 7
  REVISE: Any critical issue OR > 2 major issues OR quality_score < 7

  Special case (UNVERIFIED): when cycles are exhausted and the
  deterministic verdict would be REVISE, the orchestrator advances
  to WRITING with the UNVERIFIED flag set on the review_report. The
  evaluator surfaces this via the ``unverified`` field on the
  result; the orchestrator is responsible for the flag wiring.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CriticVerdictResult:
    """Result of deterministic verdict evaluation.

    Attributes
    ----------
    verdict:
        The verdict the orchestrator should act on.
        One of ``"PASS"``, ``"REVISE"``, ``"ABORT"``.
    deterministic_verdict:
        The evaluator's strict computation BEFORE the cycles-exhausted
        downgrade. May be ``"REVISE"`` while ``verdict == "PASS"`` and
        ``unverified == True`` (the cycles-exhausted case).
    llm_verdict:
        The Critic LLM's self-reported ``overall_verdict`` (informational).
    llm_disagreement:
        True if ``llm_verdict != verdict`` after evaluation.
    quality_score:
        The extracted ``overall_quality_score`` (or ``quality_score``).
    n_critical:
        Count of issues with severity == "critical".
    n_major:
        Count of issues with severity == "major".
    unverified:
        True when the deterministic verdict was REVISE but cycles are
        exhausted; orchestrator should set ``review_report['unverified']
        = True`` and advance to WRITING.
    rationale:
        Human-readable explanation of which threshold(s) the review
        violated; present only when ``deterministic_verdict != "PASS"``.
    """

    verdict: str
    deterministic_verdict: str
    llm_verdict: str
    llm_disagreement: bool
    quality_score: int
    n_critical: int
    n_major: int
    unverified: bool
    rationale: str = ""


# Thresholds (matching agent_prompts/critic.yaml § "Verdict Criteria").
_PASS_QUALITY_FLOOR = 7
_PASS_MAX_MAJOR = 2
_PASS_MAX_CRITICAL = 0


def _extract_quality_score(review: dict) -> int:
    """Pull the quality score from either of the two known schemas.

    Production review_report.json uses ``overall_quality_score``; the
    flat test schema (per the 3b.10 hand-off § 10.3 tests) uses
    ``quality_score``. We accept both and prefer ``overall_*``.
    """
    raw: Any = review.get("overall_quality_score")
    if raw is None:
        raw = review.get("quality_score")
    if raw is None:
        return 0
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def _extract_issue_counts(review: dict) -> tuple[int, int]:
    """Return (n_critical, n_major) from either review schema.

    Schema A (production review_report.json): nested per-section
    ``*_review.issues[].severity`` lists.
    Schema B (flat hand-off-test schema): top-level
    ``critical_issues[]`` and ``major_issues[]`` lists.

    Both are accepted; counts are summed if both are present (defensive).
    """
    n_critical = 0
    n_major = 0

    # Schema A — walk nested per-section issues.
    for k, v in review.items():
        if not isinstance(v, dict):
            continue
        issues = v.get("issues")
        if not isinstance(issues, list):
            continue
        for issue in issues:
            if not isinstance(issue, dict):
                continue
            sev = issue.get("severity")
            if sev == "critical":
                n_critical += 1
            elif sev == "major":
                n_major += 1

    # Schema B — flat top-level lists.
    flat_critical = review.get("critical_issues")
    if isinstance(flat_critical, list):
        n_critical += len(flat_critical)
    flat_major = review.get("major_issues")
    if isinstance(flat_major, list):
        n_major += len(flat_major)

    return n_critical, n_major


def _extract_llm_verdict(review: dict) -> str:
    """Pull the LLM's self-reported verdict from either schema."""
    raw = review.get("overall_verdict") or review.get("llm_reported_verdict")
    return str(raw) if raw is not None else ""


def evaluate_critic_verdict(
    review: dict,
    *,
    revision_cycle: int = 0,
    max_revision_cycles: int = 0,
) -> CriticVerdictResult:
    """Compute the deterministic verdict from a Critic review.

    Parameters
    ----------
    review:
        The Critic's review_report dict (production schema) or a flat
        ``{quality_score, critical_issues, major_issues, ...}`` dict.
    revision_cycle:
        The current revision cycle (0 for initial). Used for the
        UNVERIFIED downgrade when cycles are exhausted.
    max_revision_cycles:
        The configured cap. When ``revision_cycle >=
        max_revision_cycles`` and the deterministic verdict is REVISE,
        the result downgrades to ``verdict="PASS"`` with
        ``unverified=True`` so the orchestrator can advance to WRITING
        with the UNVERIFIED flag.

    Returns
    -------
    CriticVerdictResult: the verdict + metadata. The orchestrator
    should consult ``result.verdict`` for state-machine routing and
    ``result.unverified`` for the UNVERIFIED-flag wiring.

    Side effects
    ------------
    Logs a WARNING when ``llm_verdict != verdict`` (post-evaluation).
    The deterministic evaluator wins regardless.
    """
    quality_score = _extract_quality_score(review)
    n_critical, n_major = _extract_issue_counts(review)
    llm_verdict = _extract_llm_verdict(review)

    # Apply thresholds.
    threshold_failures: list[str] = []
    if n_critical > _PASS_MAX_CRITICAL:
        threshold_failures.append(
            f"n_critical={n_critical} (max {_PASS_MAX_CRITICAL})"
        )
    if n_major > _PASS_MAX_MAJOR:
        threshold_failures.append(
            f"n_major={n_major} (max {_PASS_MAX_MAJOR})"
        )
    if quality_score < _PASS_QUALITY_FLOOR:
        threshold_failures.append(
            f"quality_score={quality_score} (min {_PASS_QUALITY_FLOOR})"
        )

    if not threshold_failures:
        deterministic = "PASS"
        rationale = ""
    else:
        deterministic = "REVISE"
        rationale = (
            "Threshold violations: " + "; ".join(threshold_failures)
        )

    # V4 Arc H (3b.23.7): supported-ABORT pass-through. Pre-3b.23.7 the
    # evaluator could never emit ABORT, which made the orchestrator's
    # ABORT branch dead code and silently disabled the SPEC §8 safety
    # valve ("Critic verdict = ABORT → ABORTED"). Rule: the LLM's ABORT
    # is honored ONLY when backed by at least one critical issue — an
    # unsupported ABORT (no criticals) stays downgraded per the
    # no-invented-issues philosophy. ABORT is about unfixable flaws, so
    # the cycles-exhaustion UNVERIFIED downgrade does NOT rescue it.
    if llm_verdict == "ABORT" and n_critical >= 1:
        deterministic = "ABORT"
        rationale = (
            f"LLM ABORT supported by n_critical={n_critical}"
            + (f"; {rationale}" if rationale else "")
        )

    # UNVERIFIED downgrade: cycles exhausted + deterministic REVISE → effective PASS.
    cycles_exhausted = (
        max_revision_cycles > 0
        and revision_cycle >= max_revision_cycles
    )
    if deterministic == "REVISE" and cycles_exhausted:
        verdict = "PASS"
        unverified = True
    else:
        verdict = deterministic
        unverified = False

    # LLM verdict cross-check.
    disagreement = bool(llm_verdict) and llm_verdict != verdict
    if disagreement:
        logger.warning(
            "Critic verdict-evaluator: LLM reported %r but evaluator "
            "computed %r (deterministic=%r, unverified=%s, %s). "
            "Evaluator wins; LLM verdict was overridden.",
            llm_verdict,
            verdict,
            deterministic,
            unverified,
            rationale or "thresholds satisfied",
        )

    return CriticVerdictResult(
        verdict=verdict,
        deterministic_verdict=deterministic,
        llm_verdict=llm_verdict,
        llm_disagreement=disagreement,
        quality_score=quality_score,
        n_critical=n_critical,
        n_major=n_major,
        unverified=unverified,
        rationale=rationale,
    )
