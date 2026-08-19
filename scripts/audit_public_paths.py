#!/usr/bin/env python
"""Scan tracked files for machine-specific paths and personal identifiers.

This repository is published. Anything git tracks is world-readable, so a
developer's home directory, a mapped drive letter, or an email address in
a config file leaks whether or not it is secret.

The check is deliberately mechanical. An earlier hand-written scan used a
character class that matched exactly ONE path separator, so a
YAML-escaped ``H:\\\\My Drive`` -- two literal backslashes in the file --
did not match, and ``config.yaml`` shipped with a real local path anyway.
Separator runs are therefore matched with ``+``, not a single character,
and the patterns are unit-tested against that exact escaping.

Usage:
    python scripts/audit_public_paths.py           # audit tracked files
    python scripts/audit_public_paths.py --all     # audit the whole tree

Exit code is 1 when anything is found, so CI can gate on it.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

BS = chr(92)

#: One or more separators, either flavour. The ``+`` is the whole point:
#: YAML escaping doubles backslashes, so the count is not predictable.
SEP = "[" + BS + BS + "/]+"

PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # A mapped Windows drive plus a cloud-sync folder: H:\My Drive\...
    ("drive-path", re.compile("[A-Za-z]:" + SEP + r"My Drive")),
    # Any Windows user profile directory.
    ("windows-home", re.compile("[A-Za-z]:" + SEP + "Users" + SEP + r"[A-Za-z0-9._-]+")),
    # A POSIX home directory. /home/sandbox and /home/user are container
    # paths that belong in a Dockerfile, so they are allowed by name.
    ("posix-home", re.compile(r"/(?:home|Users)/(?!sandbox\b|user\b)[a-z][a-z0-9._-]*")),
    ("personal-email", re.compile(r"[A-Za-z0-9._%+-]+@(?:gmail|outlook|yahoo|hotmail)\.com")),
]

#: Directories never worth scanning even in --all mode.
SKIP_DIRS = frozenset({".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"})

#: Documentation may need to SHOW a bad path to explain why it is bad.
#: Such a line must say so explicitly, on the same line.
ALLOW_MARKER = "audit-allow-path"


def tracked_files(root: Path) -> list[Path]:
    """Return the files git tracks, which is exactly what publishing exposes."""
    out = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [root / name for name in out.stdout.split("\0") if name]


def all_files(root: Path) -> list[Path]:
    return [
        p
        for p in root.rglob("*")
        if p.is_file() and not any(part in SKIP_DIRS for part in p.parts)
    ]


def scan(paths: list[Path], root: Path) -> list[tuple[str, Path, int, str]]:
    """Return (label, path, line_number, matched_text) for every hit."""
    hits: list[tuple[str, Path, int, str]] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary or unreadable: nothing textual to leak
        lines = text.split("\n")
        for label, pattern in PATTERNS:
            for match in pattern.finditer(text):
                lineno = text[: match.start()].count("\n")
                if ALLOW_MARKER in lines[lineno]:
                    continue
                hits.append((label, path.relative_to(root), lineno + 1, match.group(0)))
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="repository root to audit")
    parser.add_argument(
        "--all",
        action="store_true",
        help="scan the whole tree, not just tracked files",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    paths = all_files(root) if args.all else tracked_files(root)
    hits = scan(paths, root)

    for label, rel, lineno, text in hits:
        print(f"[{label}] {rel}:{lineno}: {text}")

    scope = "all files" if args.all else "tracked files"
    print(f"\n{len(hits)} machine-specific path(s) across {len(paths)} {scope}.")
    if hits:
        print(
            "Replace them with ${LSAR_HOME} (expanded by src.config.load_config) "
            f"or a relative path. To keep an intentional example, add "
            f"'{ALLOW_MARKER}' to that line."
        )
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
