"""The published tree must not carry anyone's local filesystem layout.

The regression this guards against is specific. During the public release
a hand-written shell scan reported the tree clean, and it was not:
``config.yaml`` still held ``lsar_project_path: "H:\\\\My Drive\\\\LSAR"``.
The scan's character class matched a single separator, YAML escaping had
doubled it, and a check that looked thorough silently passed. These tests
pin the escaping cases the scan has to survive.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_public_paths import PATTERNS, scan, tracked_files  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
BS = chr(92)


def _match_labels(text: str) -> set[str]:
    return {label for label, pattern in PATTERNS if pattern.search(text)}


def _origin_url(root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""
    return out.stdout.strip()


#: The tree-wide assertion is about PUBLISHING. The private development
#: repository legitimately holds absolute paths -- run configs record the
#: machine a run actually happened on -- so asserting there would fail for
#: a correct checkout. Gate on the remote rather than on a hand-maintained
#: flag: the thing that makes a path dangerous is that this remote is public.
_IS_PUBLIC_MIRROR = "edm-ars-public" in _origin_url(REPO_ROOT)


@pytest.mark.skipif(
    not _IS_PUBLIC_MIRROR,
    reason="tree-wide path audit applies to the public mirror; this checkout is private",
)
def test_tracked_files_carry_no_machine_specific_paths() -> None:
    """The property that actually matters: nothing published leaks a path."""
    hits = scan(tracked_files(REPO_ROOT), REPO_ROOT)
    rendered = "\n".join(f"  [{lbl}] {p}:{n}: {t}" for lbl, p, n, t in hits)
    assert not hits, f"machine-specific paths in tracked files:\n{rendered}"


@pytest.mark.parametrize(
    "separator",
    [BS, BS + BS, "/", "//"],
    ids=["one-backslash", "yaml-escaped-backslash", "one-slash", "double-slash"],
)
def test_drive_path_is_caught_at_every_escaping(separator: str) -> None:
    """A doubled backslash is what the original scan missed."""
    line = f'lsar_project_path: "H:{separator}My Drive{separator}LSAR"'
    assert "drive-path" in _match_labels(line), f"missed: {line}"


@pytest.mark.parametrize("separator", [BS, BS + BS, "/"])
def test_windows_home_is_caught_at_every_escaping(separator: str) -> None:
    line = f"C:{separator}Users{separator}someone{separator}scratch"
    assert "windows-home" in _match_labels(line)


def test_container_paths_are_not_flagged() -> None:
    """A Dockerfile legitimately names paths inside the image."""
    assert not _match_labels("WORKDIR /home/sandbox")
    assert not _match_labels("USER /home/user")


def test_personal_email_is_caught() -> None:
    assert "personal-email" in _match_labels("contact: someone@gmail.com")


def test_env_var_form_is_not_flagged() -> None:
    """The sanctioned replacement must not trip the check that demanded it."""
    assert not _match_labels('lsar_config_path: "${LSAR_HOME}/config.yaml"')


def test_allow_marker_suppresses_an_intentional_example() -> None:
    """Docs explaining a bad path need to be able to show one."""
    bad = f'H:{BS}My Drive{BS}LSAR'
    doc = REPO_ROOT / "tests" / "_audit_allow_fixture.tmp"
    doc.write_text(f"Do not write {bad} -- audit-allow-path\n", encoding="utf-8")
    try:
        assert not scan([doc], REPO_ROOT)
    finally:
        doc.unlink()


def test_audit_script_exits_nonzero_when_it_finds_something(tmp_path: Path) -> None:
    """CI gates on the exit code, so the exit code has to be real."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    offender = tmp_path / "conf.yaml"
    offender.write_text(f'path: "H:{BS}{BS}My Drive{BS}{BS}LSAR"\n', encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "conf.yaml"], check=True)

    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "audit_public_paths.py"),
         "--root", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1, result.stdout
    assert "drive-path" in result.stdout
