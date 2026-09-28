"""LLM-generated code must not see the host's credentials.

The subprocess executor copied os.environ wholesale into the child, after
main.py's load_dotenv() had put every provider key there. Generated code
that printed os.environ while debugging (a common idiom) sent the keys
into the retry prompt -- to whichever provider served it -- and into
prompts/<agent>/.../rendered_prompt.txt in the run folder. Rscript is a
grandchild of that process and inherited the same environment.

The scrub is a denylist on NAMES: Python on Windows needs SYSTEMROOT, R
needs its R_* variables, and the bridge needs EDM_ARS_*.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.sandbox import (
    SubprocessExecutor,
    blas_thread_env,
    is_secret_name,
    scrub_secrets,
)

SECRET_NAMES = [
    "DEEPSEEK_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "MINIMAX_API_KEY",
    "SEMANTIC_SCHOLAR_API_KEY",
    "TAVILY_API_KEY",
    "LSAR_DEEPSEEK_KEY",
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "HF_TOKEN",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "AZURE_STORAGE_CONNECTION_STRING",
    "GITHUB_PAT",
    "DB_PASSWORD",
    "CLIENT_SECRET",
    "some_api_key",
    "APIKEY",
]

#: Variables a child Python or Rscript genuinely needs, or that the
#: pipeline passes on on purpose. None may be dropped.
ESSENTIAL_NAMES = [
    "PATH",
    "PATHEXT",
    "SYSTEMROOT",
    "WINDIR",
    "COMSPEC",
    "TEMP",
    "TMP",
    "HOME",
    "USERPROFILE",
    "APPDATA",
    "LOCALAPPDATA",
    "LANG",
    "LC_ALL",
    "R_HOME",
    "R_LIBS",
    "R_LIBS_USER",
    "R_USER",
    "EDM_ARS_RSCRIPT",
    "EDM_ARS_R_HELPERS",
    "EDMARS_INNER_THREADS",
    "EDMARS_RUN_ID",
    "LSAR_HOME",
    "VIRTUAL_ENV",
    "CONDA_PREFIX",
    "PYTHONPATH",
    "MPLBACKEND",
    "OMP_NUM_THREADS",
    "TOKENIZERS_PARALLELISM",
    "SSL_CERT_FILE",
    "REQUESTS_CA_BUNDLE",
    "NUMBER_OF_PROCESSORS",
]


@pytest.mark.parametrize("name", SECRET_NAMES)
def test_credential_names_are_recognised(name: str) -> None:
    assert is_secret_name(name)


@pytest.mark.parametrize("name", ESSENTIAL_NAMES)
def test_essential_names_are_kept(name: str) -> None:
    assert not is_secret_name(name)


def test_the_default_env_drops_keys_and_keeps_the_rest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in SECRET_NAMES:
        monkeypatch.setenv(name, "fake-value")
    monkeypatch.setenv("R_LIBS_USER", "rlib")
    monkeypatch.setenv("EDM_ARS_R_HELPERS", "helpers")
    env = blas_thread_env()
    for name in SECRET_NAMES:
        assert name not in env and name.upper() not in env, name
    assert "PATH" in env or "Path" in env
    assert env["R_LIBS_USER"] == "rlib"
    assert env["EDM_ARS_R_HELPERS"] == "helpers"
    if os.name == "nt":
        assert any(k.upper() == "SYSTEMROOT" for k in env)


def test_an_explicit_base_is_scrubbed_too() -> None:
    env = blas_thread_env({"OUTPUT_DIR": "/workspace", "DEEPSEEK_API_KEY": "fake"})
    assert env["OUTPUT_DIR"] == "/workspace"
    assert "DEEPSEEK_API_KEY" not in env


def test_scrub_does_not_modify_its_input() -> None:
    src = {"DEEPSEEK_API_KEY": "fake", "PATH": "p"}
    out = scrub_secrets(src)
    assert out == {"PATH": "p"}
    assert "DEEPSEEK_API_KEY" in src


def test_generated_code_cannot_read_a_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-deepseek")
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "fake-s2")
    result = SubprocessExecutor().run(
        "import os\n"
        "print(sorted(k for k in os.environ if 'KEY' in k.upper()))\n"
        "print(os.environ.get('DEEPSEEK_API_KEY'))\n",
        output_dir=str(tmp_path),
        timeout_s=60,
    )
    assert result["returncode"] == 0, result["stderr"]
    assert "fake-deepseek" not in result["stdout"]
    assert "fake-s2" not in result["stdout"]
    lines = result["stdout"].splitlines()
    assert lines[0] == "[]"
    assert lines[1] == "None"


def test_the_pipeline_process_keeps_its_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scrubbing builds a copy; the orchestrator still needs the keys."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-deepseek")
    blas_thread_env()
    assert os.environ["DEEPSEEK_API_KEY"] == "fake-deepseek"
