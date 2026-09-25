"""Failure codes shared by the pipeline, run_status.json and user interfaces.

A stage that fails raises (or records) one of these codes so the terminal
status can say *why* in a form a person and a program can both act on,
instead of a raw exception string. The plain-language wording lives with
the user interface; this module only fixes the vocabulary.
"""
from __future__ import annotations

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
    "PRE_CRITIC_ABORT": {"resumable": False},
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
