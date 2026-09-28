"""Failure codes shared by the pipeline, run_status.json and user interfaces.

A stage that fails raises (or records) one of these codes so the terminal
status can say *why* in a form a person and a program can both act on,
instead of a raw exception string. The plain-language wording lives with
the user interface; this module only fixes the vocabulary.
"""
from __future__ import annotations

import re

#: Every code a run may end with. ``resumable`` says whether ``--resume``
#: can continue once the cause is fixed (for example after topping up an
#: account), as opposed to needing a changed question or configuration.
ABORT_CODES: dict[str, dict[str, bool]] = {
    "KEY_MISSING": {"resumable": True},
    "KEY_REJECTED": {"resumable": True},
    "NO_CREDIT": {"resumable": True},
    "RATE_LIMITED": {"resumable": True},
    "NETWORK": {"resumable": True},
    "TIMEOUT": {"resumable": True},
    "MODEL_GONE": {"resumable": True},
    "PROVIDER_ERROR": {"resumable": True},
    "DATA_MISSING": {"resumable": True},
    "DE_VALIDATION_FAILED": {"resumable": True},
    "SAMPLE_TOO_SMALL": {"resumable": False},
    "DATA_CONTRACT_FAILED": {"resumable": True},
    "ANALYSIS_FAILED": {"resumable": True},
    # A pre-review finding no revision can fix (confirmed leakage, a data
    # report that failed validation). One recorded before findings were
    # classified may be resumable after all: see reopened_pre_critic_stop.
    "PRE_CRITIC_ABORT": {"resumable": False},
    # A pre-review finding a revision could have fixed was still failing
    # when the revision cycles ran out. Not resumable: a resume re-enters
    # the same check with the cycle count restored from the checkpoint.
    "PRE_CRITIC_UNRESOLVED": {"resumable": False},
    "CRITIC_ABORT": {"resumable": False},
    "LLM_OUTPUT_UNPARSEABLE": {"resumable": True},
    "INTERRUPTED": {"resumable": True},
    "CRASHED": {"resumable": True},
    "UNKNOWN": {"resumable": True},
}


class ProviderError(RuntimeError):
    """An LLM provider call failed for a reason a user can act on.

    ``code`` is one of KEY_REJECTED, NO_CREDIT, MODEL_GONE, NETWORK,
    TIMEOUT, RATE_LIMITED or PROVIDER_ERROR.
    """

    def __init__(self, code: str, message: str, *, provider: str | None = None,
                 model: str | None = None, status: int | None = None) -> None:
        super().__init__(message)
        self.code = code if code in ABORT_CODES else "PROVIDER_ERROR"
        self.provider = provider
        self.model = model
        self.status = status


def code_for_exception(exc: BaseException) -> str:
    """Best-effort mapping from an exception to an abort code."""
    code = getattr(exc, "code", None)
    if isinstance(code, str) and code in ABORT_CODES:
        return code
    if isinstance(exc, KeyboardInterrupt):
        return "INTERRUPTED"
    text = f"{type(exc).__name__}: {exc}"
    lowered = text.lower()
    if (
        "jsondecodeerror" in lowered
        or "expecting value" in lowered
        or "unterminated string" in lowered
        or ("json" in lowered and "decode" in lowered)
    ):
        return "LLM_OUTPUT_UNPARSEABLE"
    if "no python code block" in lowered:
        return "LLM_OUTPUT_UNPARSEABLE"
    if "api_key" in lowered and "not set" in lowered:
        return "KEY_MISSING"
    return "UNKNOWN"


def is_resumable(code: str) -> bool:
    return ABORT_CODES.get(code, {"resumable": True})["resumable"]


#: Pre-review checks whose critical finding a revision can fix
#: (``revisable`` in src/pre_critic_checks.py).
REVISABLE_PRE_CRITIC_CHECKS: frozenset[str] = frozenset({"pcc_02", "pcc_07"})

_LEAD_CHECK = re.compile(r"^(pcc_[a-z0-9]+):")


def reopened_pre_critic_stop(abort_info: object) -> bool:
    """True for a PRE_CRITIC_ABORT a revision can now fix.

    Before 2026-09-27 every critical pre-review finding stopped the run as
    PRE_CRITIC_ABORT, which is not resumable, so a study stopped for a
    missing "above and beyond" comparison (pcc_07) could only be run
    again from the start after the fix that revises such findings. Such a
    record is recognised by what it lacks: every later PRE_CRITIC_* record
    lists its findings in ``checks``, and a later PRE_CRITIC_ABORT always
    holds one no revision can fix. Its message is led by the first
    critical finding, and the checks ran in the order pcc_01, pcc_06,
    pcc_07, pcc_02, so a message led by pcc_07 or pcc_02 means neither
    leakage (pcc_01) nor a failed validation (pcc_06) was found. Resuming
    it retries CRITIQUING, where the checks run again under the current
    rules and a revisable finding is sent back for revision.
    """
    if not isinstance(abort_info, dict):
        return False
    if abort_info.get("code") != "PRE_CRITIC_ABORT":
        return False
    if abort_info.get("stage") != "CRITIQUING" or "checks" in abort_info:
        return False
    lead = _LEAD_CHECK.match(str(abort_info.get("message") or "").strip())
    return lead is not None and lead.group(1) in REVISABLE_PRE_CRITIC_CHECKS


def abort_is_resumable(abort_info: object) -> bool:
    """Whether ``--resume`` can continue a run stopped with ``abort_info``."""
    if not isinstance(abort_info, dict):
        return False
    code = str(abort_info.get("code") or "UNKNOWN")
    return is_resumable(code) or reopened_pre_critic_stop(abort_info)
