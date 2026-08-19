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
    # 3b.10.5 added the deepseek + openai providers; without fakes for
    # them, any test constructing an agent under a deepseek/openai config
    # fails when run in isolation (the full suite only passed because an
    # earlier test imported src.main, whose module-level load_dotenv()
    # leaked the real keys into the process — an import-order dependence
    # fixed here in V4 Arc H / 3b.23.7).
    if not os.environ.get("DEEPSEEK_API_KEY"):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-fake-key-for-unit-testing")
    if not os.environ.get("OPENAI_API_KEY"):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-fake-key-for-unit-testing")


@pytest.fixture(autouse=True)
def _no_live_review_gate(request: pytest.FixtureRequest,
                         monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the orchestrator's REVIEWING stage offline in unit tests.

    Proven leak (2026-07-11, Arc P4): ``config.yaml`` ships
    ``review_gate.enabled: true``, and tests/test_end_to_end.py loads that
    real config while stubbing only the agents' ``run()`` methods. A
    socket probe showed each e2e test opening live HTTPS connections from
    ``orchestrator._run_writing -> OutlineAgent.run -> call_llm`` and from
    LSAR's own ``metadata_extractor``. They survive only because conftest
    injects a fake key that 401s -- but ``src/main.py`` calls
    ``load_dotenv()`` at import, so once any test imports it the REAL key
    is in ``os.environ`` and those become billed requests. Arc P4 adds
    revision cycles, multiplying them.

    Tests that genuinely exercise the gate construct ``ReviewGate``
    directly (tests/test_calibrated_gate.py, tests/test_arc_p3_p4.py) and
    are unaffected; this only neutralizes the orchestrator-level
    integration path. Opt out with ``@pytest.mark.live_review_gate``.
    """
    if request.node.get_closest_marker("live_review_gate"):
        return
    try:
        import src.orchestrator as orch
    except Exception:  # pragma: no cover - src not importable in some units
        return

    class _OfflineReviewGate:
        def __init__(self, *a: object, **kw: object) -> None:
            pass

        def run_gate(self) -> dict:
            return {
                "cycles_used": 0, "max_cycles": 0, "final_score": 0.0,
                "final_recommendation": "Skipped (offline test)",
                "per_cycle_scores": [], "final_review_path": None,
                "passed": False, "offline_stub": True,
            }

    monkeypatch.setattr(orch, "ReviewGate", _OfflineReviewGate)

    try:
        from src.agents.outline_agent import OutlineAgent

        monkeypatch.setattr(
            OutlineAgent, "run",
            lambda self, *a, **kw: (_ for _ in ()).throw(
                RuntimeError("OutlineAgent disabled in offline tests")
            ),
        )
    except Exception:  # pragma: no cover
        pass


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
    config.addinivalue_line(
        "markers",
        "live_review_gate: opt out of the offline ReviewGate stub (makes "
        "real LSAR + provider calls; use only with --run-integration)",
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
