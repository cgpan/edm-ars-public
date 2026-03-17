"""Pytest configuration: registers custom markers and handles integration test skipping."""
import os

import pytest


@pytest.fixture(autouse=True)
def _set_fake_llm_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure API keys are always set so unit tests can instantiate agents.

    Unit tests mock ``anthropic.Anthropic`` and never make real API calls, but
    ``BaseAgent.__init__`` validates the env var at construction time regardless
    of which provider (anthropic or minimax) is active in config.yaml.
    Integration tests override this with the real key via the environment.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-key-for-unit-testing")
    if not os.environ.get("MINIMAX_API_KEY"):
        monkeypatch.setenv("MINIMAX_API_KEY", "sk-minimax-fake-key-for-unit-testing")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that require ANTHROPIC_API_KEY and make real API calls",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: mark test as an integration test requiring ANTHROPIC_API_KEY",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if not config.getoption("--run-integration"):
        skip_marker = pytest.mark.skip(
            reason="Integration test: pass --run-integration flag to run"
        )
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_marker)
