"""The Docker sandbox must be able to run the code the pipeline generates.

Three defects that the fully-mocked Docker tests could not see:

* The script went to ``containers.create`` as a string command. docker-py
  shlex-splits a string, so with ``ENTRYPOINT ["python", "-c"]`` the
  container ran ``python -c import`` (SyntaxError), and an apostrophe in a
  comment raised "No closing quotation" before any container existed.
* There was no .dockerignore, so the auto-build uploaded the whole project
  -- data/raw (~2 GB of HSLS alone), outputs, .git and the .env holding
  API keys -- to the Docker daemon as build context.
* The image lacked dowhy and statsmodels, which the causal skills make
  mandatory inside generated code.
"""
from __future__ import annotations

import fnmatch
import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.sandbox import DockerSandbox

ROOT = Path(__file__).resolve().parent.parent


# --- the command -----------------------------------------------------------

def _client() -> MagicMock:
    client = MagicMock()
    container = client.containers.create.return_value
    container.wait.return_value = {"StatusCode": 0}
    container.logs.return_value = b""
    return client


@pytest.mark.parametrize(
    "code",
    [
        "import pandas as pd\nprint('ok')\n",
        "# don't impute the outcome\nprint(\"it's fine\")\n",
        "x = 'unbalanced \" quote'\n",
    ],
)
def test_the_script_is_passed_as_one_argv_element(tmp_path: Path, code: str) -> None:
    client = _client()
    ds = DockerSandbox(auto_build=False)
    ds._client = client
    ds.run(code, output_dir=str(tmp_path), timeout_s=5)
    command = client.containers.create.call_args.kwargs["command"]
    assert command == [code]
    assert not isinstance(command, str)


def test_the_container_env_is_utf8_and_carries_no_host_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key")
    monkeypatch.setenv("EDM_ARS_RSCRIPT", "host-only")
    client = _client()
    ds = DockerSandbox(auto_build=False, rscript_path="host/Rscript")
    ds._client = client
    ds.run("print(1)", output_dir=str(tmp_path), timeout_s=5)
    env = client.containers.create.call_args.kwargs["environment"]
    assert env["OUTPUT_DIR"] == "/workspace"
    assert env["PYTHONUTF8"] == "1"
    assert "DEEPSEEK_API_KEY" not in env
    assert "PATH" not in env
    assert "EDM_ARS_RSCRIPT" not in env


# --- the build context -----------------------------------------------------

def _dockerignore_patterns() -> list[str]:
    text = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    return [
        line.strip() for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _excluded(path: str, patterns: list[str]) -> bool:
    """Docker's rule: the last matching pattern wins; a pattern that
    matches a parent directory matches everything under it."""
    parts = path.split("/")
    prefixes = ["/".join(parts[: i + 1]) for i in range(len(parts))]
    excluded = False
    for pat in patterns:
        negate = pat.startswith("!")
        body = pat[1:] if negate else pat
        if any(fnmatch.fnmatchcase(p, body) for p in prefixes):
            excluded = not negate
    return excluded


def _dockerfile_copy_sources() -> list[str]:
    sources: list[str] = []
    for line in (ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*(COPY|ADD)\s+(.*)$", line, re.IGNORECASE)
        if not m:
            continue
        args = [a for a in m.group(2).split() if not a.startswith("--")]
        sources.extend(args[:-1])
    return sources


def test_a_dockerignore_exists_and_excludes_by_default() -> None:
    patterns = _dockerignore_patterns()
    assert patterns and patterns[0] == "*"


@pytest.mark.parametrize(
    "path",
    [
        "data/raw/hsls_17_student_pets_sr_v1_0.csv",
        "output/run_20260101_000000/paper.pdf",
        "runs/some_run/output/panel_analytic.csv",
        ".venv/Lib/site-packages/x.py",
        ".env",
        ".git/config",
        "anything.csv",
        "config.yaml",
        "src/sandbox.py",
    ],
)
def test_data_outputs_and_secrets_stay_out_of_the_build_context(path: str) -> None:
    assert _excluded(path, _dockerignore_patterns())


def test_every_file_the_dockerfile_copies_is_in_the_context() -> None:
    patterns = _dockerignore_patterns()
    sources = _dockerfile_copy_sources()
    assert sources, "expected at least one COPY in the Dockerfile"
    for src in sources:
        assert not _excluded(src, patterns), f"{src} is excluded by .dockerignore"


# --- the image's packages --------------------------------------------------

def _pins(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        m = re.match(r"^([A-Za-z0-9_.-]+)\s*==\s*([^\s;]+)$", line)
        if m:
            pins[m.group(1).lower()] = m.group(2)
    return pins


def test_the_causal_packages_are_pinned_in_the_image() -> None:
    pins = _pins(ROOT / "requirements-sandbox.txt")
    assert "dowhy" in pins
    assert "statsmodels" in pins


def test_the_image_dowhy_is_inside_the_host_range() -> None:
    specifiers = pytest.importorskip("packaging.specifiers")
    host = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    m = re.search(r"^dowhy\s*([^#\n]+)", host, re.MULTILINE)
    assert m, "requirements.txt no longer names dowhy"
    spec = specifiers.SpecifierSet(m.group(1).strip())
    assert _pins(ROOT / "requirements-sandbox.txt")["dowhy"] in spec


def test_the_image_build_smoke_imports_them() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    smoke = [line for line in dockerfile.splitlines() if "import pandas" in line]
    assert smoke and "dowhy" in smoke[0] and "statsmodels" in smoke[0]
