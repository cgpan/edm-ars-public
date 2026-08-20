"""The published tree must not carry anyone's local filesystem layout.

The regression this guards against is specific. During the public release
a hand-written shell scan reported the tree clean, and it was not:
``config.yaml`` still held ``lsar_project_path: "H:\\\\My Drive\\\\LSAR"``.  audit-allow-path
The scan's character class matched a single separator, YAML escaping had
doubled it, and a check that looked thorough silently passed. These tests
pin the escaping cases the scan has to survive.
"""

from __future__ import annotations

import re
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


@pytest.mark.parametrize(
    "address",
    [
        "someone@gmail.com",  # audit-allow-path
        "someone@outlook.com",  # audit-allow-path
        "abc1234@tc.someuniversity.edu",  # audit-allow-path
        "a.person@sub.dept.university.ac.uk",  # audit-allow-path
        "first.last+tag@company.io",  # audit-allow-path
    ],
    ids=["freemail", "freemail-2", "institutional", "multi-label", "plus-tag"],
)
def test_real_email_is_caught_whatever_the_domain(address: str) -> None:
    """The original pattern listed four freemail domains and missed the rest.

    An institutional address is more identifying than a consumer one, not
    less: it names an affiliation as well as a person.
    """
    assert "personal-email" in _match_labels(f"contact: {address}")  # audit-allow-path


@pytest.mark.parametrize(
    "address",
    [
        "email@institution.edu",
        "you@example.com",
        "t@example.invalid",
        "user@yourdomain.org",
        "dev@localhost.localdomain",  # audit-allow-path
    ],
    ids=["byline-placeholder", "rfc2606", "reserved-invalid", "generic", "localhost"],
)
def test_placeholder_addresses_are_not_flagged(address: str) -> None:
    """Templates must still be able to show the shape of an address."""
    assert "personal-email" not in _match_labels(f"contact: {address}")  # audit-allow-path


def test_reserved_tld_exclusion_survives_backtracking() -> None:
    """A trailing negative lookbehind is not an exclusion.

    The first attempt ended the pattern with ``(?<!\\.localdomain)``. The
    engine answered by letting the TLD match one character less, so
    ``dev@localhost.localdomain`` matched as ``dev@localhost.localdomai``
    and the exclusion never fired. Excluding at a fixed position after
    the ``@`` is what actually holds, so assert on the FULL match text,
    not merely on whether something matched.
    """
    pattern = dict(PATTERNS)["personal-email"]
    for address in ("dev@localhost.localdomain", "svc@my.test", "n@box.local"):  # audit-allow-path
        match = pattern.search(address)
        assert match is None, f"matched a reserved domain as {match.group(0)!r}"


def test_paper_templates_carry_no_real_address() -> None:
    """Regression pin for the byline that shipped a real institutional address.

    Asserted as a PROPERTY, not as the literal string. Pinning the exact
    address would mean re-publishing it in this file, which is precisely
    what removing it was meant to stop -- and it would only ever catch
    that one value, not the next one someone pastes into a byline.
    """
    for rel in (
        "templates/paper_template_v2.tex",
        "skills/writing/acm-acmart-sigconf-template/paper_template_v2.tex",
        "templates/paper_template_journal.tex",
        "templates/paper_template.tex",
    ):
        path = REPO_ROOT / rel
        if not path.exists():
            continue
        hits = scan([path], REPO_ROOT)
        emails = [h for h in hits if h[0] == "personal-email"]
        assert not emails, f"{rel} carries a real address: {emails}"


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


#: Where the copyright holder's real name is allowed to appear, and why.
#: LICENSE: an MIT grant needs a named holder or it grants nothing.
#: README.md: the BibTeX entry cites an external published paper, and
#: anonymising a citation misattributes real work.
_NAME_ALLOWED_IN = frozenset({"LICENSE", "README.md"})


def _copyright_holder() -> str:
    """Read the real name out of LICENSE rather than hardcoding it here.

    Writing the name into this file would add a THIRD published copy of
    the very string the anonymisation removed -- the same mistake as
    pinning a real email address in the test that deletes it. Deriving it
    also means the check follows the name if the holder ever changes.
    """
    match = re.search(
        r"Copyright \(c\)\s+\d{4}\s+(.+)", (REPO_ROOT / "LICENSE").read_text(encoding="utf-8")
    )
    assert match, "LICENSE has no parseable copyright line"
    return match.group(1).strip().rstrip(".")


def _name_variants(full_name: str) -> list[str]:
    """Forms the same person's name is written in across a repo."""
    variants = [full_name]
    parts = full_name.split()
    if len(parts) >= 2:
        variants.append(f"{parts[-1]}, {' '.join(parts[:-1])}")  # BibTeX "Last, First"
        variants.append(parts[0])  # bare first name, as in "flagged for <first>"
    return variants


@pytest.mark.skipif(
    not _IS_PUBLIC_MIRROR,
    reason="author anonymisation applies to the public mirror; this checkout is private",
)
def test_owner_name_appears_only_where_it_must() -> None:
    """The private repo keeps the real byline; the public mirror must not.

    That divergence is deliberate -- the owner publishes papers under
    their own name -- which is exactly why it needs a test. A future sync
    that copies templates or the Writer's author assertion across would
    otherwise reintroduce the name silently.
    """
    variants = _name_variants(_copyright_holder())
    offenders: list[str] = []
    for path in tracked_files(REPO_ROOT):
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel in _NAME_ALLOWED_IN:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for variant in variants:
            if variant in text:
                offenders.append(f"{rel} (as {variant!r})")
                break
    assert not offenders, "owner name outside LICENSE/README: " + ", ".join(offenders)


_TEMPLATES = [
    "templates/paper_template_v2.tex",
    "templates/paper_template_journal.tex",
    "skills/writing/acm-acmart-sigconf-template/paper_template_v2.tex",
]


@pytest.mark.skipif(
    not _IS_PUBLIC_MIRROR,
    reason="placeholders are a public-mirror property; the private repo keeps real names",
)
@pytest.mark.parametrize("rel", _TEMPLATES)
def test_paper_templates_use_author_placeholders(rel: str) -> None:
    """A user of this repo fills these in; they must not ship someone else's name."""
    path = REPO_ROOT / rel
    if not path.exists():
        pytest.skip(f"{rel} not in this checkout")
    text = path.read_text(encoding="utf-8")
    # On a compiled line the underscore is escaped, so accept either form.
    for token in ("Human_Author_Name", "AI_Name"):
        escaped = token.replace("_", BS + "_")
        assert token in text or escaped in text, f"{rel} lost the {token} placeholder"


@pytest.mark.parametrize("rel", _TEMPLATES)
def test_placeholders_on_compiled_lines_escape_their_underscores(rel: str) -> None:
    """An underscore is a math-mode character, and these templates get compiled.

    ``\\authorsnames{EDM-ARS, AI_Name, Human_Author_Name}`` is not
    commented out, and pdflatex answers a bare underscore there with
    "Missing $ inserted" -- every journal-format paper would fail to
    build. Verified both ways against a minimal document.

    Scoped to the placeholder tokens on purpose. A blanket ban on bare
    underscores flags ACM's own CCS concept XML (``<concept_id>``,
    ``<concept_significance>``), where they are perfectly legal -- a
    checker that cries wolf on correct content gets switched off.
    Commented lines are exempt: nothing compiles them, and an escaped
    underscore would only look wrong to someone reading the template.
    """
    path = REPO_ROOT / rel
    if not path.exists():
        pytest.skip(f"{rel} not in this checkout")

    tokens = [
        "AI_Name",
        "AI_Institution",
        "Human_Author_Name",
        "Human_Author_Institution",
        "Human_Author_City",
        "Human_Author_State",
        "Human_Author_Country",
    ]
    offenders: list[str] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").split("\n"), start=1):
        if line.lstrip().startswith("%"):
            continue
        for token in tokens:
            if token in line:  # the bare, unescaped spelling
                offenders.append(f"{rel}:{lineno}: {line.strip()[:70]}")
                break
    assert not offenders, (
        "placeholder with an unescaped underscore on a compiled line "
        "(pdflatex: Missing $ inserted):\n  " + "\n  ".join(offenders)
    )


def test_the_acm_template_and_its_skill_copy_stay_identical() -> None:
    """Two copies that drift are two chances to reintroduce a name."""
    a = REPO_ROOT / "templates/paper_template_v2.tex"
    b = REPO_ROOT / "skills/writing/acm-acmart-sigconf-template/paper_template_v2.tex"
    if not (a.exists() and b.exists()):
        pytest.skip("template pair not in this checkout")
    assert a.read_text(encoding="utf-8") == b.read_text(encoding="utf-8")


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
