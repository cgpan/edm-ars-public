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
