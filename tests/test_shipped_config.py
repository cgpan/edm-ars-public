"""The shipped config.yaml and .env.example say what the code does.

Each test here pins a defect that reached the public release:

* ``sandbox.enabled: true`` shipped although every validated run config
  turned it off, so anyone with Docker running got an executor with no R,
  no r_helpers and a memory cap below a full HSLS:09 load (A3).
* The ``task_type`` comment offered ``causal_inference``, which raises at
  start-up, and another comment pointed at a ``set_env.ps1`` that is not
  shipped (E7).
* The flash tier's retired model id stayed in the config after the API
  stopped serving it (E4).
* There was no ``.env.example``; the obvious one (LSAR's) used placeholder
  values that were then sent to services as real keys.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterator

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"
ENV_EXAMPLE = ROOT / ".env.example"

#: Model ids a provider has stopped serving. A config that routes a stage
#: to one of these fails every call to that stage, and several call sites
#: swallow the failure.
RETIRED_MODEL_IDS = frozenset({"deepseek-v4-flash"})


@pytest.fixture(scope="module")
def config_text() -> str:
    return CONFIG_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def config(config_text: str) -> dict:
    return yaml.safe_load(config_text)


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield str(k)
            yield from _strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v)


# ---------------------------------------------------------------------------
# CONTRACT section 8: every new key exists with the default the code assumes
# ---------------------------------------------------------------------------


class TestContractKeys:
    def test_sandbox_ships_off(self, config: dict) -> None:
        assert config["sandbox"]["enabled"] is False

    def test_sandbox_python_executable_defaults_to_running_python(
        self, config: dict
    ) -> None:
        assert "python_executable" in config["sandbox"]
        assert config["sandbox"]["python_executable"] is None

    def test_r_bridge_rscript_path_is_declared(self, config: dict) -> None:
        assert config["r_bridge"] == {"rscript_path": None}

    def test_llm_network_settings(self, config: dict) -> None:
        assert config["llm"]["request_timeout_s"] == 600
        assert config["llm"]["max_network_retries"] == 3

    def test_crossref_mailto_is_declared_and_empty(self, config: dict) -> None:
        assert "crossref_mailto" in config["semantic_scholar"]
        assert config["semantic_scholar"]["crossref_mailto"] is None

    def test_loader_accepts_the_shipped_file(self) -> None:
        from src.config import load_config

        loaded = load_config(str(CONFIG_PATH))
        assert loaded["sandbox"]["enabled"] is False
        assert loaded["llm"]["request_timeout_s"] == 600

    def test_sandbox_comment_is_honest_about_the_docker_path(
        self, config_text: str
    ) -> None:
        block = config_text.split("\nsandbox:", 1)[0].rsplit("\n\n", 1)[-1]
        assert "EXPERIMENTAL" in block
        assert "no R" in block
        assert "4g" in block


# ---------------------------------------------------------------------------
# E7: comments must not send users to values or files that do not exist
# ---------------------------------------------------------------------------


class TestComments:
    def test_task_type_comment_lists_exactly_the_registered_types(
        self, config_text: str, config: dict
    ) -> None:
        from src.task_template import _TASK_REGISTRY

        match = re.search(r"#\s*Study type:\s*(.+)", config_text)
        assert match, "config.yaml lost the task_type option list"
        listed = {t.strip() for t in match.group(1).split("|")}
        assert listed == set(_TASK_REGISTRY)
        assert config["pipeline"]["task_type"] in _TASK_REGISTRY

    def test_no_comment_names_the_unshipped_set_env_script(
        self, config_text: str
    ) -> None:
        assert "set_env.ps1" not in config_text

    def test_repo_paths_named_in_config_exist(self, config_text: str) -> None:
        pattern = re.compile(
            r"\b((?:src|scripts|agent_prompts|data_registry|templates|r_helpers)"
            r"/[\w./-]+\.(?:py|yaml|md|tex|R))\b"
        )
        missing = sorted(
            {p for p in pattern.findall(config_text) if not (ROOT / p).exists()}
        )
        assert not missing, f"config.yaml names files that do not exist: {missing}"


# ---------------------------------------------------------------------------
# E4: no config routes a stage to a model the provider no longer serves
# ---------------------------------------------------------------------------


class TestRetiredModels:
    def test_shipped_config_mentions_no_retired_id(self, config_text: str) -> None:
        for model_id in RETIRED_MODEL_IDS:
            assert model_id not in config_text

    @pytest.mark.parametrize(
        "path",
        sorted((ROOT / "runs" / "configs").glob("*.yaml")),
        ids=lambda p: p.name,
    )
    def test_run_configs_route_nothing_to_a_retired_id(self, path: Path) -> None:
        # Comments in these archived configs may quote the retired id as
        # history; only parsed VALUES are routed to a provider.
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        used = RETIRED_MODEL_IDS.intersection(_strings(data))
        assert not used, f"{path.name} routes to retired model id(s) {sorted(used)}"

    def test_unverified_rates_are_marked(self, config: dict) -> None:
        rates = config["pricing"]["per_million_tokens"]
        assert rates["deepseek-flash"].get("verified") is False


# ---------------------------------------------------------------------------
# .env.example
# ---------------------------------------------------------------------------


def _env_lines() -> tuple[dict[str, str], set[str]]:
    """(active assignments, commented-out assignment names)."""
    active: dict[str, str] = {}
    commented: set[str] = set()
    for raw in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        m = re.fullmatch(r"([A-Z][A-Z0-9_]*)\s*=(.*)", line)
        if m:
            active[m.group(1)] = m.group(2).strip()
            continue
        m = re.fullmatch(r"#\s*([A-Z][A-Z0-9_]*)=\s*", line)
        if m:
            commented.add(m.group(1))
    return active, commented


class TestEnvExample:
    REQUIRED = {
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "ANTHROPIC_API_KEY",
        "SEMANTIC_SCHOLAR_API_KEY",
        "CROSSREF_MAILTO",
        "LSAR_HOME",
        "EDM_ARS_RSCRIPT",
    }
    #: python-dotenv loads ``NAME=`` as an empty string. For these an empty
    #: string is harmful, not neutral: openai.OpenAI() takes "" from
    #: OPENAI_BASE_URL as its base URL, and "" in LSAR_HOME defeats
    #: src/config.py's os.environ.setdefault default. They must stay
    #: commented out until the user gives a value.
    MUST_BE_COMMENTED = {"OPENAI_BASE_URL", "LSAR_HOME"}

    def test_every_variable_is_listed(self) -> None:
        active, commented = _env_lines()
        missing = self.REQUIRED - set(active) - commented
        assert not missing, f".env.example does not mention {sorted(missing)}"

    def test_no_value_is_filled_in(self) -> None:
        active, _ = _env_lines()
        filled = {k: v for k, v in active.items() if v not in ("", '""', "''")}
        assert not filled, (
            f".env.example must carry empty values only, found {sorted(filled)}: "
            "a placeholder is sent to the service as if it were a real key"
        )

    def test_url_and_path_settings_are_commented_out(self) -> None:
        active, commented = _env_lines()
        assert not (self.MUST_BE_COMMENTED & set(active))
        assert self.MUST_BE_COMMENTED <= commented

    def test_parses_to_empty_values_with_python_dotenv(self) -> None:
        dotenv = pytest.importorskip("dotenv")
        values = dotenv.dotenv_values(ENV_EXAMPLE)
        assert all(v in ("", None) for v in values.values())
        assert not (self.MUST_BE_COMMENTED & set(values))

    @pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
    def test_env_is_ignored_and_example_is_not(self) -> None:
        def ignored(name: str) -> bool:
            proc = subprocess.run(
                ["git", "-C", str(ROOT), "check-ignore", "-q", name],
                capture_output=True,
                check=False,
            )
            if proc.returncode not in (0, 1):
                pytest.skip("not a git checkout")
            return proc.returncode == 0

        assert ignored(".env")
        assert not ignored(".env.example")
