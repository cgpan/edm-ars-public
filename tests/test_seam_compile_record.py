"""compile_latex (sandbox package) -> the orchestrator's compile record.

``compile_latex`` now judges whether THIS compile wrote the PDF and says
why not in ``message``. The orchestrator's ``latex_compile.json`` and its
pipeline.log line used bare file existence and a fixed sentence; they
now use both keys, so a PDF the stale-output cleanup could not remove is
not recorded as this compile's product.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.context import PipelineState
from src.orchestrator import _summarize_compile
from tests.test_orchestrator_terminal import _config, _orch, _wire


def test_summary_trusts_compile_latex_over_a_pdf_left_on_disk(tmp_path: Path) -> None:
    """A paper.pdf the stale-output cleanup could not remove must not be
    recorded as this compile's product when compile_latex says it is not."""
    (tmp_path / "paper.pdf").write_bytes(b"%PDF-1.5 old")
    result = {
        "success": False,
        "pdf_exists": False,
        "stale_pdf": True,
        "missing_tool": None,
        "failed_step": "pdflatex -interaction=nonstopmode paper.tex",
        "message": "LaTeX produced no new paper.pdf; the one on disk is left "
        "over from an earlier compile. See paper.log.",
        "steps": [{"cmd": "pdflatex -interaction=nonstopmode paper.tex",
                   "returncode": 1, "stdout": "", "stderr": ""}],
    }
    summary = _summarize_compile(str(tmp_path), result)
    assert summary["pdf_exists"] is False
    assert summary["stale_pdf"] is True
    assert summary["message"].startswith("LaTeX produced no new paper.pdf")


def test_summary_without_the_new_keys_falls_back_to_the_file(tmp_path: Path) -> None:
    (tmp_path / "paper.pdf").write_bytes(b"%PDF-1.5")
    summary = _summarize_compile(str(tmp_path), {"success": True, "steps": []})
    assert summary["pdf_exists"] is True
    assert summary["stale_pdf"] is False
    assert summary["message"] is None


def test_no_pdf_log_line_carries_compile_latex_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_compile(output_dir: str, *_a: Any, **_k: Any) -> dict:
        return {
            "success": False,
            "pdf_exists": False,
            "stale_pdf": False,
            "missing_tool": None,
            "failed_step": "pdflatex -interaction=nonstopmode paper.tex",
            "message": "LaTeX produced no paper.pdf. See paper.log.",
            "steps": [{"cmd": "pdflatex -interaction=nonstopmode paper.tex",
                       "returncode": 1, "stdout": "", "stderr": ""}],
        }

    monkeypatch.setattr("src.orchestrator.compile_latex", fake_compile)
    orch = _orch(tmp_path, _config(tmp_path))
    _wire(orch)

    ctx = orch.run()

    assert ctx.current_state == PipelineState.INCOMPLETE
    log = (tmp_path / "pipeline.log").read_text(encoding="utf-8")
    assert "LaTeX compilation produced NO paper.pdf: LaTeX produced no paper.pdf." in log
    record = json.loads((tmp_path / "latex_compile.json").read_text(encoding="utf-8"))
    assert record["message"] == "LaTeX produced no paper.pdf. See paper.log."
