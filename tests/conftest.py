"""Pytest configuration: registers custom markers and handles integration test skipping."""
import os
import shutil
from pathlib import Path
from typing import Any

import pytest

#: The findings memory a real run reads and writes. The shipped config
#: enables it, so any test that drives the Orchestrator to a terminal state
#: with that config would otherwise append a fake run to it.
_LIVE_FINDINGS_MEMORY_DIR = (
    Path(__file__).resolve().parents[1] / "findings_memory"
).resolve()


@pytest.fixture(autouse=True)
def _isolate_findings_memory(
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Send every read and write of the live findings memory to a temp dir.

    Before this, ``pytest tests/`` -- the README's install check -- left
    runs such as ``test_happy_path_completes0`` in
    ``findings_memory/memory.yaml``, and the ProblemFormulator of the
    user's first real run was told X3TGPAMAT had already been studied by
    a dozen runs that never happened. Paths outside the live directory
    (the tests of FindingsMemory itself use tmp_path) pass through.
    """
    try:
        from src import findings_memory as fm
    except Exception:  # pragma: no cover - src not importable in some units
        return

    sandbox: dict[str, Path] = {}

    def _redirect(path: str) -> str:
        try:
            resolved = Path(path).resolve()
        except (OSError, TypeError, ValueError):
            return path
        if resolved.parent != _LIVE_FINDINGS_MEMORY_DIR:
            return path
        if "dir" not in sandbox:
            sandbox["dir"] = tmp_path_factory.mktemp("findings_memory")
        return str(sandbox["dir"] / resolved.name)

    original_init = fm.FindingsMemory.__init__
    original_load = fm.FindingsMemory.load.__func__

    # Extra arguments pass through, so a later signature (for example a
    # lock timeout) keeps working under the redirect.
    def _init(self: "fm.FindingsMemory", path: str, *args: Any, **kwargs: Any) -> None:
        original_init(self, _redirect(path), *args, **kwargs)

    def _load(cls: type, path: str, *args: Any, **kwargs: Any) -> "fm.FindingsMemory":
        return original_load(cls, _redirect(path), *args, **kwargs)

    monkeypatch.setattr(fm.FindingsMemory, "__init__", _init)
    monkeypatch.setattr(fm.FindingsMemory, "load", classmethod(_load))


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
    config.addinivalue_line(
        "markers",
        "requires_tools(*names): skip unless every named command-line tool "
        "(pdflatex, bibtex, biber, ...) is on PATH; looked up with "
        "shutil.which after collection, never run while a module is imported",
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
    _skip_tests_whose_tools_are_missing(items)


def _skip_tests_whose_tools_are_missing(items: list[pytest.Item]) -> None:
    """Skip each ``requires_tools`` test whose tools are not on PATH.

    shutil.which only looks, it never starts the tool, so this cannot fail
    the way a tool call in a skipif condition does: the first CI run had
    no TeX, ``subprocess.run(["pdflatex", ...])`` in a class-level skipif
    raised FileNotFoundError while tests/test_arc_p3_p4.py was being
    imported, and pytest stopped before it ran a single test. The skip is
    a marker, so -rs reports it at the test, with the missing tools named.
    """
    found: dict[str, bool] = {}
    for item in items:
        names = [
            name
            for marker in item.iter_markers(name="requires_tools")
            for name in marker.args
        ]
        missing: list[str] = []
        for name in names:
            if name not in found:
                found[name] = shutil.which(name) is not None
            if not found[name] and name not in missing:
                missing.append(name)
        if missing:
            item.add_marker(pytest.mark.skip(
                reason=f"{', '.join(missing)} not found on PATH; this test "
                "runs the real tool"
            ))
