"""Arc T (research taste) - idea screening and ranking.

T0 ships the deterministic half: a feasibility screen computed entirely
from registries, task templates and data on disk (``feasibility``), the
Tier-1 column cache it reads (``probe_cache``), and the anchor-derived
venue-fit rule table (``venue_fit``). No module in this package makes an
LLM call or a network request.

Re-exports are lazy (PEP 562) so that
``python -m src.ideation.feasibility`` does not import the module twice
and trip runpy's double-import warning.
"""
from __future__ import annotations

from typing import Any

__all__ = [
    "CLEAN",
    "DATASET_TASK_MATRIX",
    "KILL",
    "OK",
    "WARN",
    "CheckResult",
    "FeasibilityReport",
    "ScreenContext",
    "VenueFitHit",
    "VenueFitReport",
    "make_context",
    "probe",
    "rank_key",
    "score_venue_fit",
    "screen",
]

_FEASIBILITY_EXPORTS = {
    "CLEAN",
    "DATASET_TASK_MATRIX",
    "KILL",
    "OK",
    "WARN",
    "CheckResult",
    "FeasibilityReport",
    "ScreenContext",
    "make_context",
    "probe",
    "rank_key",
    "screen",
}
_VENUE_FIT_EXPORTS = {"VenueFitHit", "VenueFitReport", "score_venue_fit"}


def __getattr__(name: str) -> Any:
    if name in _FEASIBILITY_EXPORTS:
        from src.ideation import feasibility

        return getattr(feasibility, name)
    if name in _VENUE_FIT_EXPORTS:
        from src.ideation import venue_fit

        return getattr(venue_fit, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
