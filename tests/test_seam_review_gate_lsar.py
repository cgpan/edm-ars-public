"""The review gate against the LSAR fix branch, and the gate's reviser.

* LSAR's ``ScoringFailedError`` (raised instead of fabricated 5/10
  scores) is a gate that did not run, with its own skip reason.
* LSAR's ``configure_logging`` routes its INFO chatter to a per-cycle
  ``lsar.log``; the console keeps warnings unless the operator chose a
  level. An LSAR without it still reviews.
* A reviser that was unavailable or whose calls failed reaches the run's
  errors, but only when a revision was actually due.
"""
from __future__ import annotations

import logging
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from src.review_gate import ReviewGate


class ScoringFailedError(RuntimeError):
    """Stand-in for lsar.stage5_scoring.ScoringFailedError (matched by name)."""


def _install_fake_lsar(
    monkeypatch: pytest.MonkeyPatch, run: Any, with_configure: bool = True
) -> logging.Logger:
    lsar_logger = logging.getLogger("lsar")
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.NOTSET)
    monkeypatch.setattr(lsar_logger, "handlers", [console])

    def configure_logging(level: Any = None, *, console: Any = None,
                          log_file: Any = None) -> logging.Logger:
        if log_file is not None:
            lsar_logger.addHandler(logging.FileHandler(str(log_file), encoding="utf-8"))
        return lsar_logger

    pkg = types.ModuleType("lsar")
    pkg.__path__ = []  # type: ignore[attr-defined]
    utils = types.ModuleType("lsar.utils")
    utils.__path__ = []  # type: ignore[attr-defined]
    logger_mod = types.ModuleType("lsar.utils.logger")
    if with_configure:
        logger_mod.configure_logging = configure_logging  # type: ignore[attr-defined]
    utils.logger = logger_mod  # type: ignore[attr-defined]
    pipeline_mod = types.ModuleType("lsar.pipeline")

    class LSARPipeline:
        def __init__(self, config_path: Any = None) -> None:
            pass

        def run(self, **kw: Any) -> Any:
            logging.getLogger("lsar.stage1").info("stage chatter")
            return run(**kw)

    pipeline_mod.LSARPipeline = LSARPipeline  # type: ignore[attr-defined]
    for name, mod in (("lsar", pkg), ("lsar.utils", utils),
                      ("lsar.utils.logger", logger_mod),
                      ("lsar.pipeline", pipeline_mod)):
        monkeypatch.setitem(sys.modules, name, mod)
    return lsar_logger


def _gate(tmp_path: Path) -> ReviewGate:
    lsar_dir = tmp_path / "LSAR"
    lsar_dir.mkdir(exist_ok=True)
    cfg = {"review_gate": {"pass_threshold": 5.5, "dimension_floor": 3,
                           "max_cycles": 1, "lsar_project_path": str(lsar_dir)}}
    gate = ReviewGate(cfg, str(tmp_path), log_fn=lambda *_: None)
    gate.lsar_project_path = lsar_dir
    pdf = tmp_path / "paper_for_review.pdf"
    pdf.write_bytes(b"%PDF-1.5 stub")
    gate.prepare_pdf = lambda *_a, **_k: pdf  # type: ignore[method-assign]
    return gate


def test_a_scoring_failure_is_a_gate_that_did_not_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def run(**_kw: Any) -> Any:
        raise ScoringFailedError("J1 guard refused a review with no weaknesses")

    _install_fake_lsar(monkeypatch, run)
    summary = _gate(tmp_path).run_gate()

    assert summary["ran"] is False
    assert summary["final_score"] is None
    assert summary["skip_reason"].startswith("lsar_scoring_failed: J1 guard")
    assert "revision_failures" in summary
    assert "revision_unavailable_reason" in summary


def test_lsar_chatter_goes_to_the_cycle_log_not_the_console(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LSAR_LOG_LEVEL", raising=False)
    monkeypatch.delenv("LSAR_QUIET", raising=False)

    def run(**_kw: Any) -> Any:
        return "# report", {"scores": {"overall_score": 7.0,
                                        "recommendation": "Accept",
                                        "dimensions": []}}

    lsar_logger = _install_fake_lsar(monkeypatch, run)
    gate = _gate(tmp_path)
    report = gate.run_lsar(tmp_path / "paper_for_review.pdf", cycle=1)

    assert report is not None
    console = [h for h in lsar_logger.handlers
               if not isinstance(h, logging.FileHandler)]
    assert console and all(h.level >= logging.WARNING for h in console)
    # The per-cycle file handler is detached once the cycle ends.
    assert not [h for h in lsar_logger.handlers if isinstance(h, logging.FileHandler)]
    assert (tmp_path / "lsar_review" / "cycle_1" / "lsar.log").exists()


def test_an_operator_lsar_log_level_keeps_the_console(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LSAR_LOG_LEVEL", "DEBUG")

    def run(**_kw: Any) -> Any:
        return "# report", {"scores": {"overall_score": 7.0}}

    lsar_logger = _install_fake_lsar(monkeypatch, run)
    _gate(tmp_path).run_lsar(tmp_path / "paper_for_review.pdf", cycle=1)
    console = [h for h in lsar_logger.handlers
               if not isinstance(h, logging.FileHandler)]
    assert console and all(h.level == logging.NOTSET for h in console)


def test_an_older_lsar_without_configure_logging_still_reviews(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def run(**_kw: Any) -> Any:
        return "# report", {"scores": {"overall_score": 6.0}}

    _install_fake_lsar(monkeypatch, run, with_configure=False)
    report = _gate(tmp_path).run_lsar(tmp_path / "paper_for_review.pdf", cycle=1)
    assert report == {"scores": {"overall_score": 6.0}}


def test_revision_problems_reach_the_run_errors_only_when_they_mattered() -> None:
    from src.orchestrator import _revision_problem

    base = {"ran": True, "passed": False, "max_cycles": 2,
            "revision_unavailable_reason": None, "revision_failures": []}
    assert _revision_problem(base) is None
    unavailable = {**base, "revision_unavailable_reason": "no revision model for provider 'openai'"}
    assert "could not revise" in _revision_problem(unavailable)
    failed = {**base, "revision_failures": [{"code": "MODEL_GONE", "message": "404"}]}
    assert "(MODEL_GONE)" in _revision_problem(failed)
    # A pass, a single cycle, or a gate that never ran had nothing to revise.
    assert _revision_problem({**unavailable, "passed": True}) is None
    assert _revision_problem({**unavailable, "max_cycles": 1}) is None
    assert _revision_problem({**unavailable, "ran": False}) is None
