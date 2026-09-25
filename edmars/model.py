"""Small data types shared by every part of the edmars application.

These are deliberately plain dataclasses with no behaviour that touches
the disk or the network, so any module can import them without cost.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

#: The four outcomes a check can have. ``info`` is for facts that are
#: neither good nor bad (for example "Docker is installed but not used").
CheckStatus = Literal["ok", "warn", "fail", "info"]

CHECK_STATUSES: tuple[str, ...] = ("ok", "warn", "fail", "info")

#: Task types the pipeline can run, in the order menus show them.
TASK_TYPES: tuple[str, ...] = (
    "prediction",
    "causal_soo",
    "causal_itr",
    "causal_did",
    "psychometrics",
)


@dataclass
class StudyPlan:
    """Everything needed to start one study.

    ``spec`` is a locked research spec (a dict the pipeline's
    ``load_locked_research_spec`` accepts) for causal and psychometrics
    studies; prediction studies usually carry only ``prompt``.
    ``experimental`` marks a spec built from menus rather than one of the
    proven examples, and is shown as a badge wherever the plan is shown.
    """

    task_type: str
    dataset: str
    research_question: str
    prompt: str | None = None
    spec: dict | None = None
    example_id: str | None = None
    experimental: bool = False
    venue: str = "EDM"
    paper_format: str = "conference"
    review: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable copy (used for runner.json)."""
        return asdict(self)


@dataclass
class Check:
    """One line of a checklist: what was checked, how it went, what to do.

    ``detail`` is a plain-language sentence for a non-programmer; ``fix``
    is the exact thing to do next (often a command) when the status is
    ``warn`` or ``fail``.
    """

    name: str
    status: CheckStatus
    detail: str
    fix: str | None = None

    def __post_init__(self) -> None:
        if self.status not in CHECK_STATUSES:
            raise ValueError(
                f"Check status must be one of {CHECK_STATUSES}, got {self.status!r}"
            )

    @property
    def ok(self) -> bool:
        """True unless the check failed; warnings do not block anything."""
        return self.status != "fail"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable copy (used by ``doctor --json``)."""
        return asdict(self)
