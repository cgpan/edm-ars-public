#!/usr/bin/env python
"""Build the files the installers download from a GitHub release.

Writes into ``dist/``:

* ``edm-ars-<version>.tar.gz`` -- the committed tree at ``--ref`` (default
  HEAD) under a single top-level folder ``edm-ars-<version>/``, without
  raw data or run outputs, plus a small ``BUILD_INFO.json``;
* ``install.sh`` and ``install.ps1`` -- the installers from the same commit;
* ``SHA256SUMS`` -- one ``<sha256>  <file name>`` line per file above, in
  the format ``sha256sum -c`` reads. The installers refuse a tarball whose
  hash does not match this file.

The archive comes from ``git archive``, so only committed content can ship:
an untracked scratch file or a local ``.env`` in the working tree is never
included, whatever state the checkout is in. The output is reproducible:
building the same commit twice gives byte-identical files.

Usage::

    python scripts/make_release.py                  # version from the tag at HEAD
    python scripts/make_release.py --version 0.1.0  # explicit
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Release versions become folder names and URL path segments, so keep
#: them to a plain dotted shape with an optional pre-release suffix.
VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[-.+][0-9A-Za-z.]+)?$")

#: Top-level entries that never belong in an installed copy. ``data`` holds
#: raw survey files, the rest are outputs or repository plumbing.
EXCLUDED_TOP = frozenset(
    {
        "data",
        "output",
        "dist",
        "cache",
        "ideas",
        "files",
        "tmp_orch_test",
        "tmp_orch_test2",
        ".github",
    }
)

INSTALLERS = ("install/install.sh", "install/install.ps1")


def git(repo: Path, *args: str) -> bytes:
    # git archive applies the checkout line-ending conversion, so on a
    # Windows machine with core.autocrlf=true every file (install.sh
    # included, which sh then cannot run) would ship with CRLF. Pin the
    # conversion off so a release built anywhere is the committed bytes.
    return subprocess.run(
        ["git", "-c", "core.autocrlf=false", "-c", "core.eol=lf", "-C", str(repo), *args],
        capture_output=True,
        check=True,
    ).stdout


def is_excluded(path: str) -> bool:
    """True for archive paths (relative to the repo root) left out of a release."""
    parts = PurePosixPath(path).parts
    if not parts:
        return False
    if parts[0] in EXCLUDED_TOP:
        return True
    # runs/<name>/output*/... is evidence from past runs, not something the
    # app reads; runs/fixtures and runs/configs stay (study examples).
    if parts[0] == "runs" and len(parts) >= 3 and parts[2].startswith("output"):
        return True
    name = parts[-1]
    return name == ".env" or name.endswith(".env")


def version_from_tag(repo: Path, ref: str) -> str | None:
    try:
        tag = git(repo, "describe", "--tags", "--exact-match", ref).decode().strip()
    except subprocess.CalledProcessError:
        return None
    return tag[1:] if tag.startswith("v") else tag


def version_in_package(repo: Path, ref: str) -> str | None:
    """``__version__`` from ``edmars/__init__.py`` at ``ref``, if present."""
    try:
        text = git(repo, "show", f"{ref}:edmars/__init__.py").decode("utf-8")
    except subprocess.CalledProcessError:
        return None
    match = re.search(r"""^__version__\s*=\s*["']([^"']+)["']""", text, re.MULTILINE)
    return match.group(1) if match else None


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_tarball(repo: Path, ref: str, version: str, out: Path) -> Path:
    commit = git(repo, "rev-parse", ref).decode().strip()
    commit_time = int(git(repo, "show", "-s", "--format=%ct", commit).decode().strip())
    raw = git(repo, "archive", "--format=tar", commit)
    prefix = f"edm-ars-{version}"
    target = out / f"{prefix}.tar.gz"

    build_info = json.dumps(
        {"name": "edm-ars", "version": version, "commit": commit, "commit_time": commit_time},
        indent=2,
        sort_keys=True,
    ).encode("utf-8") + b"\n"

    with target.open("wb") as fh:
        # mtime=0 and an empty file name make the gzip header deterministic.
        with gzip.GzipFile(filename="", mode="wb", fileobj=fh, mtime=0, compresslevel=9) as gz:
            with tarfile.open(
                fileobj=gz,
                mode="w",
                format=tarfile.PAX_FORMAT,
                pax_headers={"comment": commit},
            ) as dst:
                top = tarfile.TarInfo(prefix)
                top.type = tarfile.DIRTYPE
                top.mode = 0o755
                top.mtime = commit_time
                dst.addfile(top)
                with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as src:
                    for member in src.getmembers():
                        if member.type == tarfile.XGLTYPE or is_excluded(member.name):
                            continue
                        if not (member.isfile() or member.isdir() or member.issym()):
                            continue
                        clone = tarfile.TarInfo(f"{prefix}/{member.name.rstrip('/')}")
                        clone.type = member.type
                        clone.mode = member.mode
                        clone.mtime = commit_time
                        clone.linkname = member.linkname
                        clone.size = member.size if member.isfile() else 0
                        data = src.extractfile(member) if member.isfile() else None
                        dst.addfile(clone, data)
                info = tarfile.TarInfo(f"{prefix}/BUILD_INFO.json")
                info.size = len(build_info)
                info.mode = 0o644
                info.mtime = commit_time
                dst.addfile(info, io.BytesIO(build_info))
    return target


def build_release(
    repo: Path,
    out: Path,
    version: str | None = None,
    ref: str = "HEAD",
) -> list[Path]:
    """Build the release files for ``ref`` into ``out`` and return their paths."""
    if version is None:
        version = version_from_tag(repo, ref) or version_in_package(repo, ref)
    if version is None:
        raise ValueError(
            "no version: pass --version, tag the commit (vX.Y.Z), or set "
            "__version__ in edmars/__init__.py"
        )
    version = version[1:] if version.startswith("v") else version
    if not VERSION_RE.match(version):
        raise ValueError(f"version {version!r} is not of the form X.Y.Z[-suffix]")
    packaged = version_in_package(repo, ref)
    if packaged is not None and packaged != version:
        raise ValueError(
            f"release version {version} does not match edmars.__version__ "
            f"{packaged} at {ref}; `edmars version` would report the wrong number"
        )

    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("edm-ars-*.tar.gz"):
        stale.unlink()
    files = [build_tarball(repo, ref, version, out)]
    for rel in INSTALLERS:
        try:
            content = git(repo, "show", f"{ref}:{rel}")
        except subprocess.CalledProcessError as exc:
            raise ValueError(f"{rel} is not committed at {ref}") from exc
        dest = out / PurePosixPath(rel).name
        dest.write_bytes(content)
        files.append(dest)

    sums = "".join(f"{sha256_of(p)}  {p.name}\n" for p in sorted(files, key=lambda p: p.name))
    sums_path = out / "SHA256SUMS"
    sums_path.write_bytes(sums.encode("ascii"))
    return [*files, sums_path]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build EDM-ARS release files into dist/.")
    parser.add_argument("--version", help="release version, e.g. 0.1.0 (default: tag at --ref)")
    parser.add_argument("--ref", default="HEAD", help="commit to package (default HEAD)")
    parser.add_argument("--out", default=str(REPO_ROOT / "dist"), help="output folder")
    args = parser.parse_args(argv)

    if shutil.which("git") is None:
        print("git is not on PATH", file=sys.stderr)
        return 1
    try:
        files = build_release(REPO_ROOT, Path(args.out), args.version, args.ref)
    except (ValueError, subprocess.CalledProcessError) as exc:
        print(f"make_release: {exc}", file=sys.stderr)
        return 1
    for path in files:
        print(f"{path.stat().st_size:>12,}  {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
