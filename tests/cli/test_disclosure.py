"""The disclaimer, privacy notice and the versioned setup acknowledgement."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from edmars import disclosure, settings, ui

REPO_ROOT = Path(__file__).resolve().parents[2]
TEXTS = REPO_ROOT / "edmars" / "texts"


@pytest.mark.parametrize(
    ("root_name", "packaged_name"),
    [("DISCLAIMER.md", "disclaimer.md"), ("PRIVACY.md", "privacy.md")],
)
def test_packaged_texts_are_identical_to_the_repository_copies(root_name: str, packaged_name: str) -> None:
    root = (REPO_ROOT / root_name).read_bytes()
    packaged = (TEXTS / packaged_name).read_bytes()
    assert root == packaged, (
        f"edmars/texts/{packaged_name} differs from {root_name}: edit {root_name} and copy it "
        "over byte for byte, so the program prints what the repository publishes."
    )


def test_acknowledgement_text_is_pinned_to_its_version() -> None:
    assert disclosure.ack_digest() == disclosure.ACK_SHA256, (
        "The acknowledgement text changed. Bump ACK_VERSION (so everyone who accepted the "
        "old text is asked again) and set ACK_SHA256 to " + disclosure.ack_digest()
    )


def test_texts_say_what_they_must() -> None:
    def flat(text: str) -> str:
        return " ".join(text.split())

    disclaimer = flat(disclosure.disclaimer_text())
    privacy = flat(disclosure.privacy_text())
    ack = flat(disclosure.ack_text())
    assert "without an isolated sandbox" in disclaimer
    assert "without warranty" in disclaimer
    assert "no telemetry" in privacy.lower()
    assert "never uploaded" in privacy
    assert "WHAT LEAVES YOUR COMPUTER" in ack
    assert "edmars disclaimer" in ack and "edmars privacy" in ack
    assert "edmars disclaimer" in disclosure.AI_DRAFT_REMINDER


def test_privacy_notice_does_not_imply_stored_keys_are_out_of_reach() -> None:
    """The AI-written code runs in the edmars Python, which has keyring
    installed, so keys in the credential store are as reachable as the
    fallback file. The notice once named only the file, and pointed at a
    section title that no longer existed."""
    privacy = disclosure.privacy_text()
    flat = " ".join(privacy.split())
    edmars_part = flat[flat.index("## If you use the `edmars` command"):]
    assert "it can still read your keys: from your credential store" in edmars_part
    assert "not a security barrier" in edmars_part
    headings = {line.lstrip("#").strip() for line in privacy.splitlines() if line.startswith("#")}
    for ref in re.findall(r"as described in \*([^*]+)\*", flat):
        assert ref in headings, f"the privacy notice points at a section that does not exist: {ref!r}"


def test_not_acknowledged_by_default() -> None:
    assert not disclosure.is_acknowledged(settings.load())


def test_record_ack_persists_the_current_version() -> None:
    current = settings.load()
    disclosure.record_ack(current)
    assert disclosure.is_acknowledged(current)
    again = settings.load()
    assert disclosure.is_acknowledged(again)
    assert again["acknowledged"]["version"] == disclosure.ACK_VERSION
    assert again["acknowledged"]["sha256"] == disclosure.ACK_SHA256
    assert again["acknowledged"]["at"].endswith("Z")


def test_an_older_acknowledgement_is_asked_again() -> None:
    current = settings.load()
    settings.set_(current, "acknowledged", {"version": "2020-01-01", "at": "2020-01-01T00:00:00Z"})
    assert not disclosure.is_acknowledged(current)


def test_record_ack_without_persist_does_not_write() -> None:
    current = settings.load()
    disclosure.record_ack(current, persist=False)
    assert disclosure.is_acknowledged(current)
    assert not disclosure.is_acknowledged(settings.load())


def test_ensure_acknowledged_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    # No terminal, no flag: refuse.
    assert not disclosure.ensure_acknowledged(settings.load())
    # The flag accepts and records.
    assert disclosure.ensure_acknowledged(settings.load(), accept=True)
    assert disclosure.is_acknowledged(settings.load())


def test_ensure_acknowledged_asks_a_person(monkeypatch: pytest.MonkeyPatch) -> None:
    shown: list[str] = []
    monkeypatch.setattr(ui, "is_interactive", lambda: True)
    monkeypatch.setattr(ui, "panel", lambda title, body: shown.append(body))
    monkeypatch.setattr(ui, "confirm", lambda message, default=True: False)
    assert not disclosure.ensure_acknowledged(settings.load())
    assert shown and "WHAT LEAVES YOUR COMPUTER" in shown[0]
    monkeypatch.setattr(ui, "confirm", lambda message, default=True: True)
    assert disclosure.ensure_acknowledged(settings.load())
    assert disclosure.is_acknowledged(settings.load())
