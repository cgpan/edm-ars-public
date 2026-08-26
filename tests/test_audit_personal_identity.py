"""The release audit must catch names and affiliations, not just paths.

The audit checked machine paths, emails and credentials. It had no notion
of a personal name, so a byline check keyed to a real person's name, and
commented-out template blocks naming that person and their institution,
all passed an audit that reported clean.

The names cannot live in the script. An audit that hardcodes the identity
it protects publishes that identity to every reader of the script -- it
would leak precisely what it exists to catch. They are supplied from
outside the repository instead, and a repo with none configured must SAY
personal-name checking is off rather than quietly reporting clean.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_public_paths import (  # noqa: E402
    IDENTITY_EXEMPT_FILES,
    identity_patterns,
    scan,
)

AUDIT = Path(__file__).resolve().parents[1] / "scripts" / "audit_public_paths.py"


# --- the script must not contain what it protects -----------------------

def test_the_audit_script_names_no_real_person() -> None:
    """The self-defeating failure mode, asserted directly."""
    text = AUDIT.read_text(encoding="utf-8")
    # A capitalised two-word run assigned into a pattern list would be a
    # hardcoded identity. The script should reference only variables.
    assert "identity_patterns" in text
    for line in text.splitlines():
        if "personal-name" in line and "re.compile(" in line:
            assert "escaped" in line or "pattern" in line, (
                f"personal-name pattern built from a literal: {line.strip()}"
            )


# --- sources of identity ------------------------------------------------

def test_names_come_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EDMARS_AUDIT_IDENTITIES", "Ada Lovelace")
    patterns = identity_patterns(tmp_path)
    assert any(p.search("by Ada Lovelace") for _, p in patterns)


def test_the_environment_variable_holds_several_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "EDMARS_AUDIT_IDENTITIES", os.pathsep.join(["Ada Lovelace", "Some Institute"])
    )
    patterns = identity_patterns(tmp_path)
    assert any(p.search("Ada Lovelace") for _, p in patterns)
    assert any(p.search("Some Institute") for _, p in patterns)


def test_names_come_from_a_gitignored_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("EDMARS_AUDIT_IDENTITIES", raising=False)
    (tmp_path / ".audit_identities").write_text(
        "# one per line\nAda Lovelace\n\n", encoding="utf-8"
    )
    patterns = identity_patterns(tmp_path)
    assert any(p.search("Ada Lovelace") for _, p in patterns)


def test_comments_and_blank_lines_are_not_treated_as_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("EDMARS_AUDIT_IDENTITIES", raising=False)
    (tmp_path / ".audit_identities").write_text("# a comment\n\n", encoding="utf-8")
    names = [p.pattern for _, p in identity_patterns(tmp_path)]
    assert not any("comment" in n for n in names)


def test_the_same_name_from_two_sources_is_reported_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EDMARS_AUDIT_IDENTITIES", "Ada Lovelace")
    (tmp_path / ".audit_identities").write_text("Ada Lovelace\n", encoding="utf-8")
    patterns = identity_patterns(tmp_path, extra_names=["Ada Lovelace"])
    matching = [p for _, p in patterns if p.search("Ada Lovelace")]
    assert len(matching) == 1, "duplicate patterns produce duplicate findings"


# --- matching behaviour -------------------------------------------------

def _hits(tmp_path: Path, content: str, name: str, filename: str = "f.py") -> list:
    target = tmp_path / filename
    target.write_text(content, encoding="utf-8")
    escaped = r"\s+".join(re.escape(part) for part in name.split())
    patterns = [("personal-name", re.compile(escaped, re.IGNORECASE))]
    return scan([target], tmp_path, patterns)


def test_a_name_in_source_code_is_caught(tmp_path: Path) -> None:
    """The exact miss: a byline check keyed to a person's name."""
    hits = _hits(tmp_path, 'if "Ada Lovelace" not in latex:\n', "Ada Lovelace")
    assert len(hits) == 1
    assert hits[0][0] == "personal-name"


def test_a_name_in_a_comment_is_still_caught(tmp_path: Path) -> None:
    """Commenting a line out does not unpublish it."""
    assert _hits(tmp_path, "% \\author{Ada Lovelace}\n", "Ada Lovelace")


def test_a_wrapped_name_is_caught(tmp_path: Path) -> None:
    """Whitespace between parts may be a newline once a file wraps."""
    assert _hits(tmp_path, "Ada\n  Lovelace\n", "Ada Lovelace")


def test_matching_ignores_case(tmp_path: Path) -> None:
    assert _hits(tmp_path, "ada lovelace\n", "Ada Lovelace")


def test_an_unrelated_file_is_not_flagged(tmp_path: Path) -> None:
    assert not _hits(tmp_path, "import pandas as pd\n", "Ada Lovelace")


# --- exemptions ---------------------------------------------------------

@pytest.mark.parametrize("filename", sorted(IDENTITY_EXEMPT_FILES))
def test_a_copyright_line_is_not_a_leak(tmp_path: Path, filename: str) -> None:
    """A licence naming its author is the point of the file."""
    assert not _hits(
        tmp_path, "Copyright (c) 2026 Ada Lovelace\n", "Ada Lovelace", filename
    )


def test_the_exemption_does_not_extend_to_other_files(tmp_path: Path) -> None:
    """LICENSE is exempt; a source file with the same text is not."""
    assert _hits(tmp_path, "Copyright (c) 2026 Ada Lovelace\n", "Ada Lovelace", "s.py")


def test_the_exemption_covers_names_only(tmp_path: Path) -> None:
    """A credential in a LICENSE is still a credential."""
    target = tmp_path / "LICENSE"
    # Deliberately NOT self-announcing as fake: is_obvious_fixture would
    # exempt it and the test would pass for the wrong reason. The allow
    # marker suppresses it in THIS source file only -- the temp file the
    # test writes carries no marker, so the scan still has to catch it.
    secret = "API_KEY = 'abcdefghijklmnopqrstuvwxyz123456'\n"  # audit-allow-path
    target.write_text(secret, encoding="utf-8")
    assert scan([target], tmp_path)


# --- the off state must be visible --------------------------------------

def test_no_identities_configured_is_announced(tmp_path: Path) -> None:
    """Silence here would read as 'checked and clean'."""
    subprocess.run(
        ["git", "init", "-q", str(tmp_path)], check=False, capture_output=True
    )
    env = dict(os.environ)
    env.pop("EDMARS_AUDIT_IDENTITIES", None)
    env["GIT_CONFIG_GLOBAL"] = str(tmp_path / "nonexistent_gitconfig")
    env["GIT_CONFIG_SYSTEM"] = str(tmp_path / "nonexistent_gitconfig")
    result = subprocess.run(
        [sys.executable, str(AUDIT), "--root", str(tmp_path)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert "no identities configured" in result.stderr.lower()


def test_the_identity_file_is_gitignored() -> None:
    """A file listing the names to redact must not itself be published."""
    root = Path(__file__).resolve().parents[1]
    ignored = subprocess.run(
        ["git", "-C", str(root), "check-ignore", ".audit_identities"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert ignored.returncode == 0, ".audit_identities is not gitignored"


# --- short and embedded names -------------------------------------------

def test_a_one_character_identity_is_ignored() -> None:
    """`git config user.name` is often an initial or a handle.

    Compiled case-insensitively, "T" matches the t inside "path", so every
    file in the repository becomes a finding and the audit stops meaning
    anything. Caught by an existing test whose fixture repo sets user.name
    to "T".
    """
    from scripts.audit_public_paths import build_identity_patterns

    assert build_identity_patterns(["T"]) == []


def test_a_short_handle_is_ignored() -> None:
    from scripts.audit_public_paths import build_identity_patterns

    assert build_identity_patterns(["cgp"]) == []


def test_a_real_name_is_still_compiled() -> None:
    from scripts.audit_public_paths import build_identity_patterns

    assert len(build_identity_patterns(["Ada Lovelace"])) == 1


def test_a_short_name_with_spaces_counts_its_letters() -> None:
    """"A B" is two letters, not three; spaces are not length."""
    from scripts.audit_public_paths import build_identity_patterns

    assert build_identity_patterns(["A B"]) == []


def test_a_name_inside_a_longer_word_is_not_a_hit(tmp_path: Path) -> None:
    """Word boundaries, or a surname matches every word containing it."""
    from scripts.audit_public_paths import build_identity_patterns

    target = tmp_path / "f.py"
    target.write_text("XAda LovelaceY\n", encoding="utf-8")
    assert not scan([target], tmp_path, build_identity_patterns(["Ada Lovelace"]))


def test_the_same_name_standing_alone_is_a_hit(tmp_path: Path) -> None:
    from scripts.audit_public_paths import build_identity_patterns

    target = tmp_path / "f.py"
    target.write_text("written by Ada Lovelace\n", encoding="utf-8")
    assert scan([target], tmp_path, build_identity_patterns(["Ada Lovelace"]))
