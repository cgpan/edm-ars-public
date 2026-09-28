"""compile_latex must say whether it produced a PDF, not guess from exit codes.

pdflatex in nonstopmode exits 1 both for recoverable errors and for a
fatal abort that writes nothing. ``success`` was ``all(rc in (0, 1))``, so
a paper.tex with a missing ``\\input`` compiled with step codes [1, 0, 1, 1],
success=True and no paper.pdf -- and pipeline.log said "paper.pdf
written". A missing pdflatex was an rc of -1 among the steps with nothing
naming the tool.
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import pytest

from src.sandbox import compile_latex

TEX = "\\documentclass{article}\n\\begin{document}\nHi.\n\\end{document}\n"


def _fake_run(
    tmp_path: Path,
    *,
    write_pdf_on: int | None = None,
    rc: int = 1,
    missing: tuple[str, ...] = (),
) -> Callable[..., subprocess.CompletedProcess[str]]:
    """A subprocess.run stand-in: every step exits *rc*; the call numbered
    *write_pdf_on* (0-based) writes paper.pdf; tools in *missing* raise
    FileNotFoundError like an uninstalled program."""
    calls = {"n": 0}

    def run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        n = calls["n"]
        calls["n"] += 1
        assert kwargs.get("encoding") == "utf-8"
        assert kwargs.get("errors") == "replace"
        if cmd[0] in missing:
            raise FileNotFoundError(2, "No such file or directory", cmd[0])
        if write_pdf_on is not None and n == write_pdf_on:
            (tmp_path / "paper.pdf").write_bytes(b"%PDF-1.5 fresh " + str(n).encode())
        return subprocess.CompletedProcess(cmd, rc, stdout="", stderr="")

    return run


def _tex(tmp_path: Path, body: str = TEX) -> None:
    (tmp_path / "paper.tex").write_text(body, encoding="utf-8")


def test_exit_codes_of_one_and_no_pdf_is_not_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tex(tmp_path)
    monkeypatch.setattr("src.sandbox.subprocess.run", _fake_run(tmp_path, rc=1))
    result = compile_latex(str(tmp_path))
    assert [s["returncode"] for s in result["steps"]] == [1, 1, 1, 1]
    assert result["success"] is False
    assert result["pdf_exists"] is False
    assert result["stale_pdf"] is False
    assert result["missing_tool"] is None
    assert result["failed_step"] == "pdflatex -no-shell-escape -interaction=nonstopmode paper.tex"
    assert "no paper.pdf" in result["message"]


def test_a_pdf_written_by_this_compile_is_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tex(tmp_path)
    monkeypatch.setattr(
        "src.sandbox.subprocess.run", _fake_run(tmp_path, write_pdf_on=0, rc=0)
    )
    result = compile_latex(str(tmp_path))
    assert result["success"] is True
    assert result["pdf_exists"] is True
    assert result["failed_step"] is None
    assert result["missing_tool"] is None
    assert result["pdf_path"] == os.path.join(str(tmp_path), "paper.pdf")
    assert result["message"] == "paper.pdf written."


def test_a_stale_pdf_from_an_earlier_compile_does_not_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tex(tmp_path)
    stale = tmp_path / "paper.pdf"
    stale.write_bytes(b"%PDF-1.5 old")
    old = time.time() - 3600
    os.utime(stale, (old, old))
    monkeypatch.setattr("src.sandbox.subprocess.run", _fake_run(tmp_path, rc=1))
    result = compile_latex(str(tmp_path))
    assert result["success"] is False
    assert result["pdf_exists"] is False
    assert result["stale_pdf"] is True
    assert "earlier compile" in result["message"]


def test_a_rewritten_pdf_counts_even_if_one_existed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tex(tmp_path)
    stale = tmp_path / "paper.pdf"
    stale.write_bytes(b"%PDF-1.5 old")
    old = time.time() - 3600
    os.utime(stale, (old, old))
    monkeypatch.setattr(
        "src.sandbox.subprocess.run", _fake_run(tmp_path, write_pdf_on=3, rc=1)
    )
    result = compile_latex(str(tmp_path))
    assert result["pdf_exists"] is True
    assert result["stale_pdf"] is False
    assert result["success"] is True  # rc 1 = warnings, and a PDF appeared


def test_a_missing_pdflatex_is_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tex(tmp_path)
    monkeypatch.setattr(
        "src.sandbox.subprocess.run", _fake_run(tmp_path, missing=("pdflatex",))
    )
    result = compile_latex(str(tmp_path))
    assert result["missing_tool"] == "pdflatex"
    assert result["success"] is False
    assert result["pdf_exists"] is False
    assert len(result["steps"]) == 1  # stops after the first pass
    assert result["failed_step"] == "pdflatex -no-shell-escape -interaction=nonstopmode paper.tex"
    assert "TeX distribution" in result["message"]


def test_a_missing_biber_is_named_even_when_a_pdf_appears(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tex(tmp_path, "\\usepackage{biblatex}\\addbibresource{r.bib}\n" + TEX)
    monkeypatch.setattr(
        "src.sandbox.subprocess.run",
        _fake_run(tmp_path, write_pdf_on=0, rc=0, missing=("biber",)),
    )
    result = compile_latex(str(tmp_path))
    assert result["missing_tool"] == "biber"
    assert result["pdf_exists"] is True
    assert result["success"] is False
    assert result["failed_step"] == "biber paper"
    assert "biber" in result["message"]


def test_the_existing_keys_are_still_there(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tex(tmp_path)
    monkeypatch.setattr("src.sandbox.subprocess.run", _fake_run(tmp_path, rc=0))
    result = compile_latex(str(tmp_path))
    for key in ("success", "steps", "pdf_exists", "missing_tool", "failed_step"):
        assert key in result
    for step in result["steps"]:
        assert set(step) == {"cmd", "returncode", "stdout", "stderr"}


@pytest.mark.requires_tools("pdflatex", "bibtex")
def test_real_fatal_abort_is_not_success(tmp_path: Path) -> None:
    """The reproduction from the verification: a missing \\input."""
    _tex(
        tmp_path,
        "\\documentclass{article}\n\\begin{document}\n"
        "\\input{definitely_missing_file_xyz}\n\\end{document}\n",
    )
    result = compile_latex(str(tmp_path), timeout_s=120)
    assert not (tmp_path / "paper.pdf").exists()
    assert result["pdf_exists"] is False
    assert result["success"] is False
    assert result["failed_step"] is not None
