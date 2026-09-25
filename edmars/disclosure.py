"""The disclaimer, the privacy notice and the setup acknowledgement.

The full texts ship in ``edmars/texts/`` as packaged copies of the
repository's ``DISCLAIMER.md`` and ``PRIVACY.md`` (a test asserts the
copies are identical, so the page on GitHub and the text the program
prints cannot drift apart).

The short acknowledgement shown in setup is versioned. A user who
accepted an older text is asked again: :func:`is_acknowledged` compares
the stored version with :data:`ACK_VERSION`, and ``edmars new`` / ``edmars
run`` refuse to start a study until the current version is accepted.

To change the acknowledgement: write ``acknowledgement_v<N>.md``, point
:data:`ACK_FILE` at it, bump :data:`ACK_VERSION`, and update
:data:`ACK_SHA256` (a test fails with the new value if you forget, which
is the point: a silent text change would otherwise not be re-asked).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from edmars import settings as settings_mod

#: Version of the acknowledgement text currently shown in setup.
ACK_VERSION = "2026-09-25"

#: File in ``edmars/texts/`` holding that text.
ACK_FILE = "acknowledgement_v1.md"

#: SHA-256 of the acknowledgement text with LF line endings. Pinned so a
#: wording change cannot ship without a version bump.
ACK_SHA256 = "6053324c1b0428e64749d6a231fef0778cd1b1e604ed800806fc2d457897e0f4"

#: The reminder every result screen ends with.
AI_DRAFT_REMINDER = (
    "This is an AI-generated draft. Check every number and citation before "
    "sharing, and follow your venue's rules on disclosing AI use. "
    "See `edmars disclaimer`."
)

TEXTS_DIR = Path(__file__).resolve().parent / "texts"


def _read(name: str) -> str:
    return (TEXTS_DIR / name).read_text(encoding="utf-8")


def ack_text() -> str:
    """The short "before you start" text shown and accepted in setup."""
    return _read(ACK_FILE)


def disclaimer_text() -> str:
    """The full disclaimer (same text as the repository's DISCLAIMER.md)."""
    return _read("disclaimer.md")


def privacy_text() -> str:
    """The privacy notice (same text as the repository's PRIVACY.md)."""
    return _read("privacy.md")


def ack_digest() -> str:
    """SHA-256 of :func:`ack_text` with normalised line endings."""
    text = ack_text().replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_acknowledged(settings: dict[str, Any]) -> bool:
    """True when the user accepted the CURRENT acknowledgement version."""
    return settings_mod.get(settings, "acknowledged.version") == ACK_VERSION


def record_ack(settings: dict[str, Any], *, persist: bool = True) -> None:
    """Record acceptance of the current text in ``settings``.

    With ``persist`` (the default) the settings are saved immediately, so
    an acceptance is never lost to a later crash in the same command.
    """
    settings_mod.set_(
        settings,
        "acknowledged",
        {"version": ACK_VERSION, "at": settings_mod.utc_now(), "sha256": ack_digest()},
    )
    if persist:
        settings_mod.save(settings)


def ensure_acknowledged(settings: dict[str, Any], *, accept: bool = False) -> bool:
    """Make sure the current text is accepted before a study starts.

    Returns True when it already was, when ``accept`` is set (the
    ``--accept-disclosure`` flag), or when the user accepts it now at an
    interactive prompt. Returns False otherwise; the caller explains how
    to accept and stops.
    """
    if is_acknowledged(settings):
        return True
    if accept:
        record_ack(settings)
        return True

    from edmars import ui

    if not ui.is_interactive():
        return False
    ui.panel("Please read this first", ack_text())
    if ui.confirm("I have read this and accept it", default=False):
        record_ack(settings)
        return True
    return False
