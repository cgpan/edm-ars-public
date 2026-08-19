"""The published tree must not carry anyone's local filesystem layout.

The regression this guards against is specific. During the public release
a hand-written shell scan reported the tree clean, and it was not:
``config.yaml`` still held ``lsar_project_path: "H:\\\\My Drive\\\\LSAR"``.  audit-allow-path
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

from scripts.audit_public_paths import (  # noqa: E402
    PATTERNS,
    history_blobs,
    scan,
    scan_history,
    tracked_files,
)

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
    """The property that actually matters: nothing published leaks a path.

    Scope note: this reads ``git ls-files``, so a brand-new file that has
    not been added yet is invisible to it and the test passes vacuously
    for that file. That is not a gap in the check -- untracked files are
    not published -- but it does mean a green run before ``git add``
    proves less than it appears to. This suite's own two files were
    flagged on the first run after they were committed.
    """
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
    assert "personal-email" in _match_labels("contact: someone@gmail.com")  # audit-allow-path


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


def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    for key, value in (("user.email", "t@example.invalid"), ("user.name", "T")):
        subprocess.run(["git", "-C", str(path), "config", key, value], check=True)


def _commit(path: Path, message: str) -> None:
    subprocess.run(["git", "-C", str(path), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-q", "-m", message], check=True)


def test_history_scan_finds_a_path_that_was_cleaned_at_head(tmp_path: Path) -> None:
    """The exact failure this repository shipped.

    config.yaml was sanitised at HEAD while two earlier blobs kept the
    original absolute path. A working-tree scan calls that clean; anyone
    can still run ``git show <old>:config.yaml`` and read it.
    """
    _init_repo(tmp_path)
    conf = tmp_path / "conf.yaml"

    conf.write_text(f'path: "H:{BS}{BS}My Drive{BS}{BS}LSAR"\n', encoding="utf-8")
    _commit(tmp_path, "leak the path")

    conf.write_text('path: "${LSAR_HOME}"\n', encoding="utf-8")
    _commit(tmp_path, "clean it up at HEAD")

    assert not scan(tracked_files(tmp_path), tmp_path), "working tree should look clean"

    history = scan_history(tmp_path)
    assert history, "history scan missed a path that only exists in an older blob"
    assert any("drive-path" == label for label, _, _, _ in history)


def test_history_scan_reports_the_blob_it_found(tmp_path: Path) -> None:
    """A finding is only actionable if it names where to look."""
    _init_repo(tmp_path)
    (tmp_path / "conf.yaml").write_text(
        f'path: "C:{BS}Users{BS}someone{BS}scratch"\n', encoding="utf-8"
    )
    _commit(tmp_path, "leak a home directory")

    (label, located, lineno, text) = scan_history(tmp_path)[0]
    assert label == "windows-home"
    assert "conf.yaml" in str(located)
    assert ":" in str(located), "should be reported as <sha>:<path>"
    assert lineno == 1
    assert "someone" in text


def test_history_scan_honours_the_allow_marker(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "notes.md").write_text(
        f"never write H:{BS}My Drive here -- audit-allow-path\n", encoding="utf-8"
    )
    _commit(tmp_path, "documented example")
    assert not scan_history(tmp_path)


def test_history_blobs_skips_commits_and_trees(tmp_path: Path) -> None:
    """Only blobs carry content; counting trees would inflate the denominator."""
    _init_repo(tmp_path)
    (tmp_path / "a.txt").write_text("hello\n", encoding="utf-8")
    _commit(tmp_path, "one file")

    blobs = history_blobs(tmp_path)
    assert [path for _, path in blobs] == ["a.txt"]


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


def test_audit_script_history_mode_is_wired_to_the_cli(tmp_path: Path) -> None:
    """--history is what a release actually runs, so exercise the flag itself."""
    _init_repo(tmp_path)
    conf = tmp_path / "conf.yaml"
    conf.write_text(f'path: "H:{BS}{BS}My Drive{BS}{BS}LSAR"\n', encoding="utf-8")
    _commit(tmp_path, "leak")
    conf.write_text('path: "${LSAR_HOME}"\n', encoding="utf-8")
    _commit(tmp_path, "clean at HEAD")

    script = str(REPO_ROOT / "scripts" / "audit_public_paths.py")

    tree = subprocess.run(
        [sys.executable, script, "--root", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert tree.returncode == 0, "working tree is genuinely clean here"

    history = subprocess.run(
        [sys.executable, script, "--root", str(tmp_path), "--history"],
        capture_output=True,
        text=True,
    )
    assert history.returncode == 1, history.stdout
    assert "drive-path" in history.stdout
    assert "force push" in history.stdout, "must say editing files cannot fix history"
