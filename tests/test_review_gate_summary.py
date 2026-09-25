"""The review gate's summary says whether a review happened at all.

A gate that could not run -- LSAR not found, LSAR not importable, no PDF
to review -- used to report ``passed: false, final_score: 0.0``, the same
record as a paper that was reviewed and judged worthless (B2). The summary
now carries ``ran`` and ``skip_reason`` and a null score when nothing was
reviewed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from src.review_gate import ReviewGate


def _gate(tmp_path: Path, lsar_path: Path) -> Any:
    cfg = {
        "review_gate": {
            "pass_threshold": 5.5,
            "dimension_floor": 3,
            "max_cycles": 2,
            "lsar_project_path": str(lsar_path),
        }
    }
    gate = ReviewGate(cfg, str(tmp_path), log_fn=lambda *_: None)
    gate.lsar_project_path = lsar_path
    return gate


def test_gate_with_lsar_missing_reports_not_run(tmp_path: Path) -> None:
    gate = _gate(tmp_path, tmp_path / "no_such_LSAR")
    seen: list[tuple[str, dict]] = []
    gate.event_fn = lambda t, **kw: seen.append((t, kw))
    pdf = tmp_path / "paper_for_review.pdf"
    pdf.write_bytes(b"%PDF-1.5 stub")
    gate.prepare_pdf = lambda *_a, **_k: pdf

    summary = gate.run_gate()

    assert summary["ran"] is False
    assert summary["skip_reason"].startswith("lsar_not_found")
    assert summary["final_score"] is None
    assert summary["passed"] is None
    on_disk = json.loads(
        (tmp_path / "lsar_review" / "gate_summary.json").read_text(encoding="utf-8")
    )
    assert on_disk["ran"] is False and on_disk["final_score"] is None
    assert ("gate.skipped" in [t for t, _ in seen])


def test_gate_with_lsar_unimportable_reports_the_import_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lsar_dir = tmp_path / "LSAR"
    lsar_dir.mkdir()
    # None in sys.modules makes the import fail without touching the disk.
    monkeypatch.setitem(sys.modules, "lsar", None)
    monkeypatch.setitem(sys.modules, "lsar.pipeline", None)
    gate = _gate(tmp_path, lsar_dir)
    pdf = tmp_path / "paper_for_review.pdf"
    pdf.write_bytes(b"%PDF-1.5 stub")
    gate.prepare_pdf = lambda *_a, **_k: pdf

    summary = gate.run_gate()

    assert summary["ran"] is False
    assert summary["skip_reason"].startswith("lsar_import_failed")


def test_gate_with_no_pdf_reports_no_pdf(tmp_path: Path) -> None:
    gate = _gate(tmp_path, tmp_path)
    gate.prepare_pdf = lambda *_a, **_k: None
    summary = gate.run_gate()
    assert summary["ran"] is False
    assert summary["skip_reason"] == "no_pdf"
    assert summary["final_recommendation"] == "Not reviewed"


def test_gate_that_reviewed_reports_ran_and_score(tmp_path: Path) -> None:
    gate = _gate(tmp_path, tmp_path)
    pdf = tmp_path / "paper_for_review.pdf"
    pdf.write_bytes(b"%PDF-1.5 stub")
    gate.prepare_pdf = lambda *_a, **_k: pdf
    gate.run_lsar = lambda *_a, **_k: {
        "scores": {"overall_score": 7.0, "recommendation": "Accept",
                   "dimensions": [{"name": "Novelty", "score": 7}]}
    }
    gate._honesty_blockers = lambda: []
    seen: list[str] = []
    gate.event_fn = lambda t, **kw: seen.append(t)

    summary = gate.run_gate()

    assert summary["ran"] is True
    assert summary["skip_reason"] is None
    assert summary["final_score"] == 7.0
    assert summary["passed"] is True
    assert seen[:2] == ["gate.cycle", "gate.review"]


def _two_cycle_gate(tmp_path: Path, second: str) -> tuple[Any, list[tuple[str, dict]]]:
    """Cycle 1 scores 5.0 against a 9.0 threshold and the paper is revised;
    cycle 2 then reviews nothing: ``second`` is "lsar" (LSAR raised, as it
    now does instead of inventing a score) or "pdf" (the revised paper did
    not compile)."""
    gate = _gate(tmp_path, tmp_path)
    gate.pass_threshold = 9.0
    (tmp_path / "paper.tex").write_text("original manuscript\n", encoding="utf-8")
    pdf = tmp_path / "paper_for_review.pdf"
    pdf.write_bytes(b"%PDF-1.5 stub")

    def prepare_pdf(*_a: Any, cycle: int = 1, **_k: Any) -> Any:
        return pdf if (cycle == 1 or second != "pdf") else None

    def run_lsar(_pdf: Any, cycle: int) -> Any:
        if cycle == 1:
            return {"scores": {"overall_score": 5.0, "recommendation": "Reject",
                               "dimensions": [{"name": "Novelty", "score": 5}]}}
        gate._last_lsar_failure = "lsar_scoring_failed: simulated"
        return None

    gate.prepare_pdf = prepare_pdf
    gate.run_lsar = run_lsar
    gate._honesty_blockers = lambda: []
    gate.revise_from_review = lambda **_k: "revised manuscript\n"
    gate._revision_is_safe = lambda *_a: (True, "")
    gate._compile_full_latex = lambda *_a: None
    seen: list[tuple[str, dict]] = []
    gate.event_fn = lambda t, **kw: seen.append((t, kw))
    return gate, seen


@pytest.mark.parametrize("second, reason", [
    ("lsar", "lsar_scoring_failed: simulated"),
    ("pdf", "no_pdf"),
])
def test_a_failed_later_cycle_is_not_passed_off_as_the_final_review(
    tmp_path: Path, second: str, reason: str
) -> None:
    gate, seen = _two_cycle_gate(tmp_path, second)

    summary = gate.run_gate()

    assert (tmp_path / "paper.tex").read_text(encoding="utf-8") == "revised manuscript\n"
    assert summary["ran"] is True
    assert summary["cycles_used"] == 1
    assert summary["final_score"] == 5.0
    assert summary["passed"] is False
    assert summary["final_manuscript_reviewed"] is False
    assert summary["last_cycle_failure"] == reason
    warnings = [kw for t, kw in seen if t == "warning"]
    assert warnings and warnings[0]["code"] == "GATE_CYCLE_NOT_REVIEWED"
    assert "not re-reviewed" in warnings[0]["message"]
    on_disk = json.loads(
        (tmp_path / "lsar_review" / "gate_summary.json").read_text(encoding="utf-8")
    )
    assert on_disk["final_manuscript_reviewed"] is False
    assert on_disk["last_cycle_failure"] == reason


def test_a_scored_last_cycle_reports_the_manuscript_as_reviewed(tmp_path: Path) -> None:
    gate = _gate(tmp_path, tmp_path)
    pdf = tmp_path / "paper_for_review.pdf"
    pdf.write_bytes(b"%PDF-1.5 stub")
    gate.prepare_pdf = lambda *_a, **_k: pdf
    gate.run_lsar = lambda *_a, **_k: {
        "scores": {"overall_score": 7.0, "recommendation": "Accept",
                   "dimensions": [{"name": "Novelty", "score": 7}]}
    }
    gate._honesty_blockers = lambda: []

    summary = gate.run_gate()

    assert summary["final_manuscript_reviewed"] is True
    assert summary["last_cycle_failure"] is None
