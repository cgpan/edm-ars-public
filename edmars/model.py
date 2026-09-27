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

#: Exit codes of every command that ends on a result screen (`edmars
#: results`, `edmars status`, and `new`, `run` and `resume` while they
#: watch): one scheme, so a script reads them the same way everywhere.
EXIT_READY = 0  # the paper is ready (with or without issues), or it is still running
EXIT_ERROR = 1  # something went wrong: no matching study, a bad option
EXIT_NOT_READY = 2  # the study ran to its end, but the paper is not ready
EXIT_STOPPED = 3  # the study stopped before it finished

#: The same scheme in words, for those commands' --help.
EXIT_CODES_HELP = (
    "Exit codes: 0 the paper is ready, or the study is still running (you "
    "left the view); 2 the study finished but the paper is not ready; 3 the "
    "study stopped before it finished, also when you stop it from the view; "
    "1 something went wrong, such as no matching study. A command line that "
    "could not be read (a mistyped option) gives 2, as with most commands."
)

#: `edmars doctor --help`. With --bundle the command's job is the support
#: file, so its exit code says whether that was written, not what the
#: checks inside it found.
DOCTOR_EXIT_CODES_HELP = (
    "Exit codes: 0 no check failed; 1 at least one check failed. With "
    "--bundle: 0 the support file was written (the checks in it may still "
    "have found problems), 1 it could not be written."
)

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
    ``warn`` or ``fail``. ``code`` is the pipeline's own finding code
    (``LSAR_IMPORT_FAILED``) when the check came from its pre-start check,
    so a screen can act on the finding without parsing the sentence.
    """

    name: str
    status: CheckStatus
    detail: str
    fix: str | None = None
    code: str | None = None

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
