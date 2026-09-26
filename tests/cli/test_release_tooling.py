"""requirements.lock and scripts/make_release.py.

The lock is what every installed copy and CI run gets, so these tests pin
that it is exact, covers every runtime requirement at a version the floors
allow, and never carries test tools or the packages it promises to leave
out. The release builder is exercised against a throwaway git repository:
what goes into the tarball, the SHA256SUMS the installers verify, and that
building twice gives the same bytes.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import make_release  # noqa: E402

LOCK = REPO_ROOT / "requirements.lock"
PIN_LINE = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)==(?P<version>[^\s;#]+)"
    r"(?:\s*;\s*(?P<marker>[^#]+?))?\s*(?:#.*)?$"
)


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EDMARS_HOME", str(tmp_path / "edmars-home"))


def _lock_pins() -> dict[str, tuple[str, str | None]]:
    pins: dict[str, tuple[str, str | None]] = {}
    for line in LOCK.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = PIN_LINE.match(line.strip())
        assert match, f"requirements.lock line is not an exact pin: {line!r}"
        name = canonicalize_name(match.group("name"))
        assert name not in pins, f"{name} is pinned twice"
        marker = match.group("marker")
        pins[name] = (match.group("version"), marker.strip() if marker else None)
    return pins


def _requirements(path: Path) -> list[Requirement]:
    reqs: list[Requirement] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.split("#", 1)[0].strip()
        if text:
            reqs.append(Requirement(text))
    return reqs


# --- requirements.lock ------------------------------------------------------------


def test_lock_is_all_exact_pins() -> None:
    pins = _lock_pins()
    assert len(pins) > 50  # the scientific stack, not a stub


def test_lock_satisfies_every_runtime_requirement() -> None:
    pins = _lock_pins()
    for req_file in ("requirements.txt", "requirements-cli.txt"):
        for req in _requirements(REPO_ROOT / req_file):
            name = canonicalize_name(req.name)
            if name == "docker":
                continue  # optional sandbox; deliberately not locked
            assert name in pins, f"{req_file} needs {req.name}, which the lock does not pin"
            version, marker = pins[name]
            assert marker is None, f"{name} is required on every platform but pinned with {marker!r}"
            assert req.specifier.contains(version, prereleases=True), (
                f"lock pins {name}=={version}, outside {req_file}'s {req.specifier}"
            )


def test_lock_leaves_out_test_tools_and_excluded_packages() -> None:
    pins = _lock_pins()
    for name in ("pytest", "pluggy", "iniconfig", "docker", "torch", "lightgbm"):
        assert name not in pins, f"{name} must not be in requirements.lock"


def test_lock_marks_platform_only_packages() -> None:
    pins = _lock_pins()
    for name in ("pywin32-ctypes", "colorama"):
        assert name in pins
        assert pins[name][1] is not None and "win32" in pins[name][1]
    for name in ("secretstorage", "jeepney"):
        assert name in pins
        assert pins[name][1] is not None and "linux" in pins[name][1]


def test_locked_questionary_works_with_the_locked_prompt_toolkit() -> None:
    # questionary 2.1.0 reaches into prompt-toolkit's prompt layout, which
    # prompt-toolkit 3.0.52 changed: its select menu then raises
    # AttributeError ('VSplit' object has no attribute 'content'), and every
    # menu of `edmars setup` and `edmars new` crashed. 2.1.1 fixes it. The
    # suite's own questionary tests only catch this when they run in the
    # locked environment, so the pair is also checked here.
    pins = _lock_pins()
    toolkit = Version(pins["prompt-toolkit"][0])
    questionary = Version(pins["questionary"][0])
    if toolkit >= Version("3.0.52"):
        assert questionary >= Version("2.1.1"), (
            f"questionary {questionary} crashes with prompt-toolkit {toolkit}; lock 2.1.1 or later")
    for req in _requirements(REPO_ROOT / "requirements-cli.txt"):
        if canonicalize_name(req.name) == "questionary":
            assert not req.specifier.contains("2.1.0"), "requirements-cli.txt must exclude questionary 2.1.0"


def test_lock_header_says_what_was_tested() -> None:
    header = LOCK.read_text(encoding="utf-8").split("\n\n", 1)[0].lower()
    assert "python 3.11" in header
    assert "passed" in header
    assert "docker" in header and "torch" in header and "lightgbm" in header


def test_cli_requirements_are_the_agreed_list() -> None:
    # The foundation branch writes the same file; both must stay identical.
    lines = (REPO_ROOT / "requirements-cli.txt").read_text(encoding="utf-8").splitlines()
    assert lines == [
        "typer>=0.12",
        "rich>=13.7",
        "questionary>=2.1.1",
        "keyring>=23.0",
        "platformdirs>=4.0",
        "psutil>=5.9",
    ]


# --- make_release -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "excluded"),
    [
        ("data/raw/hsls.csv", True),
        ("output/run/paper.tex", True),
        ("runs/x/output/results.json", True),
        ("runs/x/outputs_old/results.json", True),
        ("runs/fixtures/spec.json", False),
        ("runs/configs/a.yaml", False),
        (".github/workflows/ci.yml", True),
        ("dist/edm-ars-1.0.0.tar.gz", True),
        ("src/main.py", False),
        ("data_registry/datasets/hsls09_public.yaml", False),
        ("config/local.env", True),
        (".env", True),
        ("install/install.sh", False),
    ],
)
def test_release_exclusions(path: str, excluded: bool) -> None:
    assert make_release.is_excluded(path) is excluded


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=Release Test", "-c", "user.email=release@example.com",
         "-c", "core.autocrlf=false", "-C", str(repo), *args],
        capture_output=True, text=True, check=True,
    ).stdout


@pytest.fixture
def release_repo(tmp_path: Path) -> Path:
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    repo = tmp_path / "repo"
    files = {
        "src/main.py": "print('pipeline')\n",
        "edmars/__init__.py": '"""edmars."""\n\n__version__ = "1.2.3"\n',
        "edmars/__main__.py": "print('edmars')\n",
        "requirements.lock": "numpy==1.26.4\n",
        "requirements-cli.txt": "typer>=0.12\n",
        "install/install.sh": "#!/bin/sh\nset -eu\necho install\n",
        "install/install.ps1": "Write-Host 'install'\r\n",
        "runs/fixtures/spec.json": "{}\n",
        "runs/demo/output/results.json": "{}\n",
        "data/raw/students.csv": "id\n1\n",
        "output/old/paper.tex": "x\n",
        ".github/workflows/ci.yml": "name: CI\n",
    }
    for rel, text in files.items():
        target = repo / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(text.encode("utf-8"))
    _git(repo, "init", "-q")
    _git(repo, "add", "-A", "-f")
    _git(repo, "commit", "-q", "-m", "release test")
    return repo


def _members(tarball: Path) -> dict[str, tarfile.TarInfo]:
    with tarfile.open(tarball, "r:gz") as tar:
        return {m.name: m for m in tar.getmembers()}


def test_release_tarball_contents_and_checksums(release_repo: Path, tmp_path: Path) -> None:
    out = tmp_path / "dist"
    files = make_release.build_release(release_repo, out)
    names = sorted(p.name for p in files)
    assert names == ["SHA256SUMS", "edm-ars-1.2.3.tar.gz", "install.ps1", "install.sh"]

    members = _members(out / "edm-ars-1.2.3.tar.gz")
    assert all(n == "edm-ars-1.2.3" or n.startswith("edm-ars-1.2.3/") for n in members)
    for kept in ("src/main.py", "edmars/__main__.py", "requirements.lock",
                 "runs/fixtures/spec.json", "install/install.sh", "BUILD_INFO.json"):
        assert f"edm-ars-1.2.3/{kept}" in members, kept
    for dropped in ("data/raw/students.csv", "output/old/paper.tex",
                    "runs/demo/output/results.json", ".github/workflows/ci.yml"):
        assert f"edm-ars-1.2.3/{dropped}" not in members, dropped

    with tarfile.open(out / "edm-ars-1.2.3.tar.gz", "r:gz") as tar:
        extracted = tar.extractfile("edm-ars-1.2.3/BUILD_INFO.json")
        assert extracted is not None
        info = json.loads(extracted.read())
    assert info["version"] == "1.2.3"
    assert info["commit"] == _git(release_repo, "rev-parse", "HEAD").strip()

    # SHA256SUMS: the format `sha256sum -c` and both installers parse.
    lines = (out / "SHA256SUMS").read_text(encoding="ascii").splitlines()
    assert len(lines) == 3
    for line in lines:
        digest, name = line.split("  ", 1)
        assert re.fullmatch(r"[0-9a-f]{64}", digest)
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest

    # The installers are the committed bytes, line endings untouched.
    assert (out / "install.sh").read_bytes() == b"#!/bin/sh\nset -eu\necho install\n"
    assert (out / "install.ps1").read_bytes() == b"Write-Host 'install'\r\n"


def test_release_build_is_reproducible(release_repo: Path, tmp_path: Path) -> None:
    first = make_release.build_release(release_repo, tmp_path / "a")
    second = make_release.build_release(release_repo, tmp_path / "b")
    for a, b in zip(sorted(first), sorted(second)):
        assert a.read_bytes() == b.read_bytes(), a.name
    # The gzip header carries no timestamp or file name.
    raw = (tmp_path / "a" / "edm-ars-1.2.3.tar.gz").read_bytes()
    assert raw[4:8] == b"\x00\x00\x00\x00"
    with gzip.open(io.BytesIO(raw)) as gz:
        assert gz.read(1)


def test_release_version_must_match_the_package(release_repo: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not match"):
        make_release.build_release(release_repo, tmp_path / "dist", version="9.9.9")


def test_release_version_comes_from_the_tag(release_repo: Path, tmp_path: Path) -> None:
    _git(release_repo, "tag", "v1.2.3")
    files = make_release.build_release(release_repo, tmp_path / "dist")
    assert any(p.name == "edm-ars-1.2.3.tar.gz" for p in files)


@pytest.mark.parametrize("bad", ["1.2", "../1.2.3", "1.2.3 beta", "latest"])
def test_release_rejects_malformed_versions(release_repo: Path, tmp_path: Path, bad: str) -> None:
    with pytest.raises(ValueError):
        make_release.build_release(release_repo, tmp_path / "dist", version=bad)


def test_release_refuses_uncommitted_installers(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    repo = tmp_path / "bare"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "main.py").write_text("x\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "no installers")
    with pytest.raises(ValueError, match="not committed"):
        make_release.build_release(repo, tmp_path / "dist", version="0.0.1")
