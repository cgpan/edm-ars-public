"""Tracked text files must not contain stray C0 control bytes.

Every occurrence found so far came from the same accident: a backslash
sequence written through a shell heredoc, where the shell resolved
``\\f`` to a form feed and ``\\a`` to a BEL before Python ever saw it. The
corrupted byte is invisible in most editors and in ``git diff``, so it
gets committed and then silently changes meaning:

  skills/writing/latex-table-discipline/SKILL.md
      ``\\footnotesize`` became FF + "ootnotesize". That file is injected
      into the Writer's prompt, so a rule marked CRITICAL was naming a
      LaTeX command that does not exist.

  src/review_gate.py
      ``\\addbibresource`` became BEL + "ddbibresource" in a biblatex
      detection clause, which could therefore never match.

Neither failed loudly. A test is the only thing that sees them.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

#: C0 controls that never legitimately appear in source or documentation.
#: Tab (9), newline (10) and carriage return (13) are excluded because
#: they are ordinary whitespace.
FORBIDDEN = {
    0x00: "NUL",
    0x07: "BEL",
    0x08: "BS",
    0x0B: "VT",
    0x0C: "FF",
    0x1A: "SUB",
    0x1B: "ESC",
}


#: Directories holding EXTRACTED or GENERATED document text rather than
#: authored source. A form feed in a PDF-to-Markdown extraction is the
#: page-break character doing its job, and NUL bytes turn up in text
#: lifted out of real PDFs. Flagging those is a false positive, and a
#: check that reports hundreds of them is a check nobody reads.
DATA_DIR_PREFIXES = (
    "outputs/",
    "outputs_",
    "Papers/",
    "runs/",
    "regression/",
)


def _is_authored_source(rel: str) -> bool:
    return not any(rel.startswith(prefix) for prefix in DATA_DIR_PREFIXES)


def _tracked_files() -> list[Path]:
    out = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "-z"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [
        REPO_ROOT / name
        for name in out.stdout.split("\0")
        if name and _is_authored_source(name)
    ]


def test_no_tracked_text_file_contains_a_stray_control_byte() -> None:
    offenders: list[str] = []
    for path in _tracked_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary: control bytes are expected and meaningless here
        for lineno, line in enumerate(text.split("\n"), start=1):
            for code, name in FORBIDDEN.items():
                if chr(code) in line:
                    rel = path.relative_to(REPO_ROOT).as_posix()
                    context = line.strip()[:70]
                    offenders.append(f"{rel}:{lineno} contains {name}: {context!r}")
    assert not offenders, "stray control bytes:\n  " + "\n  ".join(offenders)


@pytest.mark.parametrize(
    "rel",
    ["outputs/x/paper.md", "outputs_minimax_era/y/paper.md", "Papers/z.md", "runs/r/out.md"],
)
def test_extracted_document_text_is_out_of_scope(rel: str) -> None:
    """Page breaks in a PDF extraction are content, not corruption."""
    assert not _is_authored_source(rel)


@pytest.mark.parametrize(
    "rel", ["lsar/utils/llm_client.py", "scripts/run_review.py", "docs/j1.md", "config.yaml"]
)
def test_authored_source_is_in_scope(rel: str) -> None:
    assert _is_authored_source(rel)


@pytest.mark.parametrize(
    ("code", "name"),
    [(0x07, "BEL"), (0x0C, "FF"), (0x1B, "ESC")],
    ids=["bel-from-backslash-a", "ff-from-backslash-f", "esc-from-backslash-e"],
)
def test_the_check_actually_detects_each_byte(tmp_path: Path, code: int, name: str) -> None:
    """Positive control: the sweep is worthless if it cannot see the byte.

    These three are the ones a shell produces from ``\\a``, ``\\f`` and
    ``\\e`` -- the escapes that appear in LaTeX commands the project
    writes constantly (``\\author``, ``\\footnotesize``, ``\\email``).
    """
    planted = tmp_path / "sample.md"
    planted.write_text(f"a line with {chr(code)}ootnotesize in it\n", encoding="utf-8")
    text = planted.read_text(encoding="utf-8")
    assert any(chr(c) in text for c in FORBIDDEN), f"{name} not detectable"
    assert chr(code) in text
