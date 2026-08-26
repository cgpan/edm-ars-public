#!/usr/bin/env python
"""Scan tracked files for machine-specific paths and personal identifiers.

This repository is published. Anything git tracks is world-readable, so a
developer's home directory, a mapped drive letter, or an email address in
a config file leaks whether or not it is secret.

The check is deliberately mechanical. An earlier hand-written scan used a
character class that matched exactly ONE path separator, so a
YAML-escaped ``H:\\\\My Drive`` -- two literal backslashes in the file --  audit-allow-path
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
import os
import re
import subprocess
import sys
from pathlib import Path

BS = chr(92)

#: One or more separators, either flavour. The ``+`` is the whole point:
#: YAML escaping doubles backslashes, so the count is not predictable.
SEP = "[" + BS + BS + "/]+"

PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # A mapped Windows drive plus a cloud-sync folder: H:\My Drive\...  audit-allow-path
    ("drive-path", re.compile("[A-Za-z]:" + SEP + r"My Drive")),
    # Any Windows user profile directory.
    ("windows-home", re.compile("[A-Za-z]:" + SEP + "Users" + SEP + r"[A-Za-z0-9._-]+")),
    # A POSIX home directory. /home/sandbox and /home/user are container
    # paths that belong in a Dockerfile, so they are allowed by name.
    ("posix-home", re.compile(r"/(?:home|Users)/(?!sandbox\b|user\b)[a-z][a-z0-9._-]*")),
    # Any real address, not just freemail. The original pattern listed four
    # consumer domains and so sailed past a real institutional address -- which
    # is more identifying than a consumer one, not less, since it names an
    # affiliation too. Reserved and obviously-fake domains (RFC 2606)
    # plus the repo's own byline placeholder are excluded by name, so a
    # template can still show what an address looks like.
    (
        "personal-email",
        re.compile(
            r"[A-Za-z0-9._%+-]+@"
            # Placeholder second-level domains, rejected at a fixed position.
            r"(?!(?:example|localhost|institution|yourdomain|your-domain"
            r"|domain|email|host|server)\.)"
            # Reserved TLDs (RFC 2606 / RFC 6761) anywhere in the domain. This
            # is a LOOKAHEAD, not a trailing lookbehind: a lookbehind at the end
            # is defeated by backtracking -- the engine matches one character
            # less of the TLD and the exclusion never applies.
            r"(?![A-Za-z0-9.-]*\.(?:invalid|test|local|localdomain|localhost"
            r"|example)(?![A-Za-z0-9-]))"
            r"(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}"
        ),
    ),
    # --- Credentials -------------------------------------------------
    # Vendor-prefixed keys. The character class MUST allow - and _:
    # a previous version used [A-Za-z0-9] and therefore could not match
    # a real key of the form sk-api-z-<mixed alnum, hyphens, underscores>.
    ("api-key-prefixed", re.compile(
        r"\b(?:sk|tvly|pk|rk|xox[baprs]|shpat|glpat|hf)[-_]"
        r"[A-Za-z0-9_-]{16,}"
    )),
    ("github-token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr|github_pat)_[A-Za-z0-9_]{20,}")),
    ("aws-key-id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("google-api-key", re.compile(r"\bAIza[A-Za-z0-9_-]{30,}\b")),
    ("private-key-block", re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----")),
    # The one that actually matters. Two of the four leaked keys had no
    # vendor prefix -- bare hex and base62 -- so only shape-in-context
    # finds them: a credential-named variable assigned a long value.
    # 20 chars is above every placeholder in use ("your-key-here" is 13)
    # and below every real key seen (shortest was 40).
    ("credential-assignment", re.compile(
        r"(?i)\b[A-Z0-9_]*(?:API_?KEY|ACCESS_?TOKEN|AUTH_?TOKEN|SECRET|PASSWORD|PASSWD)"
        r"\s*[=:]\s*[\"']?"
        r"(?!your[-_]|<|\$\{|None\b|null\b|placeholder|xxx|changeme|example)"
        r"[A-Za-z0-9_\-\.]{20,}"
    )),
]

#: Directories never worth scanning even in --all mode.
SKIP_DIRS = frozenset({".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"})

#: Labels whose hits are credentials rather than identifiers.
CREDENTIAL_LABELS = frozenset(
    {
        "api-key-prefixed",
        "github-token",
        "aws-key-id",
        "google-api-key",
        "private-key-block",
        "credential-assignment",
    }
)

#: A credential-shaped string containing one of these is a test fixture.
#: Applied to CREDENTIAL_LABELS only -- a path or an email containing the
#: word "test" is still a real finding.
_OBVIOUSLY_FAKE = (
    "fake",
    "dummy",
    "placeholder",
    "changeme",
    "your-key",
    "your_key",
    "example",
    "sample",
    "notreal",
    "for-unit-test",
    "unit-testing",
    "redacted",
)


#: Files where a personal name is the correct content, not a leak. A
#: copyright line naming the author is the whole point of the file, and
#: flagging it trains the reader to ignore the audit's output.
IDENTITY_EXEMPT_FILES = frozenset({"LICENSE", "LICENSE.md", "LICENSE.txt",
                                   "COPYRIGHT", "NOTICE", "CITATION.cff"})


def is_obvious_fixture(label: str, matched: str) -> bool:
    """True when a credential-shaped match announces itself as fake."""
    if label not in CREDENTIAL_LABELS:
        return False
    lowered = matched.lower()
    return any(marker in lowered for marker in _OBVIOUSLY_FAKE)


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


def history_blobs(root: Path) -> list[tuple[str, str]]:
    """Return (sha, path) for every blob reachable from any ref.

    Sanitising the working tree does NOT unpublish anything. A string
    removed in commit N is still served by ``git show N-1:file`` to
    anyone who clones, and to GitHub's web UI, forever. This repository
    learned that the hard way: config.yaml was cleaned at HEAD while two
    earlier blobs kept the original absolute path.
    """
    out = subprocess.run(
        ["git", "-C", str(root), "rev-list", "--objects", "--all"],
        capture_output=True,
        text=True,
        check=True,
    )
    blobs: list[tuple[str, str]] = []
    for line in out.stdout.splitlines():
        sha, _, path = line.partition(" ")
        if not path:
            continue  # commits and trees carry no path; only blobs interest us
        blobs.append((sha, path))
    return blobs


def scan_history(root: Path) -> list[tuple[str, Path, int, str]]:
    """Scan every historical blob, returning hits keyed by ``sha:path``."""
    hits: list[tuple[str, Path, int, str]] = []
    for sha, path in history_blobs(root):
        raw = subprocess.run(
            ["git", "-C", str(root), "cat-file", "-p", sha],
            capture_output=True,
            check=False,
        )
        if raw.returncode != 0:
            continue
        try:
            text = raw.stdout.decode("utf-8")
        except UnicodeDecodeError:
            continue  # binary blob: nothing textual to leak
        lines = text.split("\n")
        for label, pattern in PATTERNS:
            for match in pattern.finditer(text):
                lineno = text[: match.start()].count("\n")
                if ALLOW_MARKER in lines[lineno]:
                    continue
                if is_obvious_fixture(label, match.group(0)):
                    continue
                hits.append(
                    (label, Path(f"{sha[:8]}:{path}"), lineno + 1, match.group(0))
                )
    return hits


IDENTITY_FILE = ".audit_identities"


def identity_patterns(
    root: Path, extra_names: list[str] | None = None
) -> list[tuple[str, re.Pattern[str]]]:
    """Patterns for names and affiliations that must not appear publicly.

    The names are NOT written here. An audit script that hardcodes the
    identity it is protecting publishes that identity to every reader of
    the script -- the check would leak exactly what it exists to catch.

    They are read instead from, in order: --identity arguments, the
    EDMARS_AUDIT_IDENTITIES environment variable (os.pathsep-separated),
    the gitignored .audit_identities file, and finally `git config
    user.name`. A repository with none of these configured audits paths
    and credentials as before and reports that name checking is off,
    rather than silently passing.

    This gap is why a mirror shipped a byline check keyed to a real
    person's name, plus commented-out template blocks naming that person
    and their institution, through an audit that reported clean.
    """
    names: list[str] = list(extra_names or [])

    from_env = os.environ.get("EDMARS_AUDIT_IDENTITIES", "")
    names.extend(part for part in from_env.split(os.pathsep) if part.strip())

    identity_path = root / IDENTITY_FILE
    if identity_path.exists():
        try:
            for line in identity_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    names.append(line)
        except OSError:
            pass

    try:
        configured = subprocess.run(
            ["git", "config", "user.name"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        if configured:
            names.append(configured)
    except OSError:
        pass

    return build_identity_patterns(names)


#: Shortest identity worth matching. `git config user.name` is often an
#: initial or a handle, and a one-character name compiled case-insensitively
#: matches a letter inside ordinary words -- "T" hits the t in "path" -- so
#: every file becomes a finding and the audit is worthless.
MIN_IDENTITY_LENGTH = 4


def build_identity_patterns(names: list[str]) -> list[tuple[str, re.Pattern[str]]]:
    """Compile de-duplicated, word-bounded patterns for the given names."""
    patterns: list[tuple[str, re.Pattern[str]]] = []
    for name in dict.fromkeys(n.strip() for n in names if n.strip()):
        if len(name.replace(" ", "")) < MIN_IDENTITY_LENGTH:
            print(
                f"NOTE: ignoring identity {name!r}: shorter than "
                f"{MIN_IDENTITY_LENGTH} characters, which would match inside "
                "ordinary words.",
                file=sys.stderr,
            )
            continue
        # Whitespace in a name may be any run of it once a file wraps.
        escaped = r"\s+".join(re.escape(part) for part in name.split())
        # Bounded so a name is not found inside a longer word.
        patterns.append(
            ("personal-name", re.compile(rf"\b{escaped}\b", re.IGNORECASE))
        )
    return patterns


def scan(
    paths: list[Path],
    root: Path,
    extra_patterns: list[tuple[str, re.Pattern[str]]] | None = None,
) -> list[tuple[str, Path, int, str]]:
    """Return (label, path, line_number, matched_text) for every hit."""
    patterns = PATTERNS + list(extra_patterns or [])
    hits: list[tuple[str, Path, int, str]] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary or unreadable: nothing textual to leak
        lines = text.split("\n")
        for label, pattern in patterns:
            for match in pattern.finditer(text):
                lineno = text[: match.start()].count("\n")
                if ALLOW_MARKER in lines[lineno]:
                    continue
                if is_obvious_fixture(label, match.group(0)):
                    continue
                if label == "personal-name" and path.name in IDENTITY_EXEMPT_FILES:
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
    parser.add_argument(
        "--history",
        action="store_true",
        help="scan every blob in git history instead of the working tree",
    )
    parser.add_argument(
        "--identity",
        action="append",
        default=[],
        metavar="NAME",
        help=(
            "a personal name or affiliation that must not appear in a "
            "published file. Repeatable. Also read from "
            "EDMARS_AUDIT_IDENTITIES, the gitignored .audit_identities "
            "file, and git config user.name."
        ),
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()

    identities = identity_patterns(root, extra_names=args.identity)
    if not identities:
        print(
            "NOTE: no identities configured, so personal names and "
            "affiliations were NOT checked. Set EDMARS_AUDIT_IDENTITIES, "
            "add a .audit_identities file, or pass --identity.",
            file=sys.stderr,
        )

    if args.history:
        hits = scan_history(root)
        count = len(history_blobs(root))
        scope = "historical blobs"
    else:
        paths = all_files(root) if args.all else tracked_files(root)
        hits = scan(paths, root, identities)
        count = len(paths)
        scope = "all files" if args.all else "tracked files"

    for label, rel, lineno, text in hits:
        print(f"[{label}] {rel}:{lineno}: {text}")

    print(f"\n{len(hits)} finding(s) across {count} {scope}.")
    if hits and args.history:
        print(
            "History cannot be cleaned by editing files -- it needs a rewrite "
            "(git filter-repo / rebase) and a force push, which is destructive "
            "and does not reach anyone who already cloned or forked."
        )
    if hits:
        print(
            "Paths: replace with an environment variable or a relative path. "
            "Credentials: move the value to an untracked .env and leave a "
            "placeholder. To keep an intentional example, add "
            f"'{ALLOW_MARKER}' to that line."
        )
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
