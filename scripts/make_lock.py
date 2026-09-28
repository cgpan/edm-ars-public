#!/usr/bin/env python
"""Regenerate ``requirements.lock`` from an environment that passed the tests.

``requirements.txt`` only states floors, so two installs a week apart can
get different numpy, pandas or scikit-learn releases, and the combination a
user ends up with may never have been tested. The installer and CI install
from ``requirements.lock`` instead: every package pinned to the exact
version that was present when the full offline test suite passed.

How the lock is made:

1. Build a clean Python 3.11 environment from ``requirements.txt`` (minus
   docker) plus ``requirements-cli.txt`` and pytest, and run the full
   offline suite in it.
2. Freeze that environment (``uv pip freeze``).
3. Re-resolve the runtime requirements with the frozen versions as
   constraints, for every platform at once (``uv pip compile --universal``).
   This drops test-only packages (pytest and its helpers) because nothing
   at runtime needs them, keeps every tested version exactly, and adds
   platform markers -- the freeze happened on one OS, and a plain freeze
   would pin Windows-only packages for everyone and miss the Linux-only
   keyring backends.

Usage::

    python scripts/make_lock.py --tested-python PATH/TO/venv/python \\
        --suite-result "2482 passed, 32 skipped"

Needs ``uv`` on PATH and network access to PyPI (for step 3).
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Present in the tested environment only to run the tests.
TEST_ONLY = frozenset({"pytest", "pluggy", "iniconfig"})

#: Never part of the lock. docker is optional (the sandbox is off by
#: default); torch and lightgbm are not dependencies at all, and are listed
#: so that a developer's environment that happens to have them cannot leak
#: them into the lock.
EXCLUDED = frozenset({"docker", "torch", "lightgbm"})

#: Packages the freeze picks up on Windows only. Their marker keeps them
#: from being constrained (or installed) on other platforms.
WINDOWS_ONLY = frozenset({"colorama", "pywin32-ctypes", "pywin32"})

PYTHON_VERSION = "3.11"


def normalize(name: str) -> str:
    """PEP 503 normalized project name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_name(line: str) -> str | None:
    """Project name at the start of a requirements line, or None."""
    stripped = line.strip()
    if not stripped or stripped.startswith(("#", "-")):
        return None
    match = re.match(r"([A-Za-z0-9][A-Za-z0-9._-]*)", stripped)
    return normalize(match.group(1)) if match else None


def freeze(python: str) -> dict[str, str]:
    """``{normalized name: version}`` for every package in the environment."""
    out = subprocess.run(
        ["uv", "pip", "freeze", "--python", python],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    pins: dict[str, str] = {}
    for line in out.splitlines():
        match = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==(\S+)$", line.strip())
        if match:
            pins[normalize(match.group(1))] = match.group(2)
    return pins


def python_version(python: str) -> str:
    return subprocess.run(
        [python, "-c", "import sys; print(f'{sys.version.split()[0]} ({sys.platform})')"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def runtime_requirements(path: Path) -> list[str]:
    """Lines of a requirements file with the excluded packages removed."""
    kept: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if requirement_name(line) in EXCLUDED:
            continue
        kept.append(line)
    return kept


def header(tested: str, suite_result: str, extras: list[str]) -> str:
    extra_note = "\n".join(
        textwrap.wrap(
            "Resolved for the pinned versions but NOT part of the tested "
            f"environment: {', '.join(extras) if extras else 'none'}.",
            width=74,
            initial_indent="# ",
            subsequent_indent="# ",
        )
    )
    return f"""\
# requirements.lock -- exact package versions for installing EDM-ARS.
#
# Frozen from a clean Python {tested} environment that passed the full
# offline test suite ({suite_result}). That environment was built from
# requirements.txt (without docker) plus requirements-cli.txt and pytest;
# the test-only packages (pytest, pluggy, iniconfig) are left out here.
#
# Deliberately NOT included:
#   docker    the Docker sandbox is optional and off by default. Install it
#             yourself (pip install "docker>=7.0") only if you turn it on.
#   torch, lightgbm   not dependencies of EDM-ARS; generated analysis code
#             must not rely on them.
#
# One environment on one operating system was tested. A line with a
# `sys_platform` marker is installed on that platform only.
{extra_note}
#
# Used by install/install.sh, install/install.ps1 and CI:
#     uv pip install -r requirements.lock -r requirements-cli.txt
# Regenerate with scripts/make_lock.py after the suite passes in the new
# environment; do not edit versions by hand.
"""


def compile_lock(
    pins: dict[str, str], workdir: Path, requirements: Path, cli_requirements: Path
) -> list[str]:
    """Run ``uv pip compile`` and return the output lines."""
    (workdir / "requirements.txt").write_text(
        "\n".join(runtime_requirements(requirements)) + "\n", encoding="utf-8"
    )
    shutil.copyfile(cli_requirements, workdir / "requirements-cli.txt")
    constraints: list[str] = []
    for name, version in sorted(pins.items()):
        if name in TEST_ONLY or name in EXCLUDED:
            continue
        marker = " ; sys_platform == 'win32'" if name in WINDOWS_ONLY else ""
        constraints.append(f"{name}=={version}{marker}")
    (workdir / "tested-constraints.txt").write_text(
        "\n".join(constraints) + "\n", encoding="utf-8"
    )
    result = subprocess.run(
        [
            "uv", "pip", "compile",
            "requirements.txt", "requirements-cli.txt",
            "-c", "tested-constraints.txt",
            "--universal",
            "--python-version", PYTHON_VERSION,
            "--annotation-style", "line",
            "--no-header",
            "-o", "lock.txt",
        ],
        cwd=workdir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"uv pip compile failed:\n{result.stderr}")
    return (workdir / "lock.txt").read_text(encoding="utf-8").splitlines()


def clean_annotation(line: str) -> str:
    """Drop the constraints file from ``# via`` annotations; it says nothing."""
    code, sep, comment = line.partition("#")
    if not sep:
        return line.rstrip()
    sources = [
        s.strip()
        for s in comment.strip().removeprefix("via").split(",")
        if s.strip() and s.strip() != "-c tested-constraints.txt"
    ]
    if not sources:
        return code.rstrip()
    return f"{code.rstrip()}  # via {', '.join(sources)}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--tested-python", required=True,
                        help="python executable of the environment that passed the suite")
    parser.add_argument("--suite-result", required=True,
                        help='what the suite reported there, e.g. "2482 passed, 32 skipped"')
    parser.add_argument("--output", default=str(REPO_ROOT / "requirements.lock"))
    args = parser.parse_args(argv)

    if shutil.which("uv") is None:
        print("uv is not on PATH; see https://docs.astral.sh/uv/", file=sys.stderr)
        return 1
    pins = freeze(args.tested_python)
    tested = python_version(args.tested_python)
    with tempfile.TemporaryDirectory() as tmp:
        lines = compile_lock(
            pins,
            Path(tmp),
            REPO_ROOT / "requirements.txt",
            REPO_ROOT / "requirements-cli.txt",
        )

    body: list[str] = []
    extras: list[str] = []
    for line in lines:
        name = requirement_name(line)
        if name is None:
            continue
        version = re.search(r"==([^\s;]+)", line)
        if name in pins:
            if version is None or version.group(1) != pins[name]:
                print(f"{name}: lock and tested environment disagree", file=sys.stderr)
                return 1
        else:
            extras.append(name)
        body.append(clean_annotation(line))
    Path(args.output).write_text(
        header(tested, args.suite_result, extras) + "\n".join(body) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"wrote {args.output}: {len(body)} packages ({len(extras)} not in the tested env)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
