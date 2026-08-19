import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.agents.base import BaseAgent

CONFIG_PATH = str(Path(__file__).parent.parent / "config.yaml")


def test_load_config_defaults() -> None:
    config = load_config(CONFIG_PATH)
    for key in ("models", "pipeline", "semantic_scholar", "paths"):
        assert key in config, f"Missing top-level key: {key}"


def test_load_config_missing_key(tmp_path) -> None:
    bad_config = tmp_path / "bad_config.yaml"
    bad_config.write_text("pipeline:\n  max_revision_cycles: 2\n")
    with pytest.raises(ValueError, match="models"):
        load_config(str(bad_config))


def test_model_names() -> None:
    config = load_config(CONFIG_PATH)
    for key in ("problem_formulator", "data_engineer", "analyst", "critic", "writer"):
        assert key in config["models"], f"Missing model key: {key}"


def test_base_agent_raises_on_missing_api_key(tmp_path, monkeypatch) -> None:
    """BaseAgent.__init__ must raise EnvironmentError with a clear message when
    the active provider's API key is not set."""

    # Strip every provider's API key — whichever one config.yaml is
    # currently set to should surface the expected env var name in the
    # error.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    class _ConcreteAgent(BaseAgent):
        def run(self, **kwargs: Any) -> Any:  # pragma: no cover
            return {}

    config = load_config(CONFIG_PATH)

    class _FakeCtx:
        log: list = []
        dataset_name = "hsls09_public"
        output_dir = str(tmp_path)

    # Error message depends on active provider. Phase 3b.10.5: deepseek
    # is now the project default, so the expected env var is usually
    # DEEPSEEK_API_KEY.
    provider = config.get("llm_provider", "anthropic")
    expected_key = {
        "deepseek": "DEEPSEEK_API_KEY",
        "minimax": "MINIMAX_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }.get(provider, "ANTHROPIC_API_KEY")
    with pytest.raises(EnvironmentError, match=expected_key):
        _ConcreteAgent(_FakeCtx(), "problem_formulator", config)
