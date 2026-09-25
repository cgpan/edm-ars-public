"""The manuscript is model output; compiling it must not hand over the keys.

The Writer's and the gate reviser's LaTeX were compiled by the pipeline
process with its full environment, API keys included, and with the
restricted \\write18 every TeX distribution enables by default. Restricted
mode still runs kpsewhich, so a paper.tex holding
``\\input|"kpsewhich -var-value=DEEPSEEK_API_KEY"`` typeset the key into
paper.pdf, the file users share and the review copy sent to LSAR. The
generated Python and R already ran without the keys; LaTeX did not.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.review_gate import ReviewGate
from src.sandbox import compile_latex, is_secret_name, pdflatex_argv

# Built at runtime so no key-shaped literal sits in the source.
_FAKE = "fake" + "value" + "0123456789"

_PROBE_TEX = r"""\documentclass{article}
\newread\probe
\begin{document}
\openin\probe=|"kpsewhich -var-value=%(var)s"
\ifeof\probe\typeout{PROBE-CLOSED}\else
\read\probe to\probeval\typeout{PROBE:\probeval}\closein\probe\fi
Text.
\end{document}
"""


def _recording_run(calls: list[dict[str, Any]]) -> Any:
    def run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append({"cmd": list(cmd), "env": kwargs.get("env")})
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    return run


def _assert_safe(calls: list[dict[str, Any]]) -> None:
    assert calls
    for call in calls:
        env = call["env"]
        assert env is not None, f"{call['cmd']} inherited the full environment"
        assert "DEEPSEEK_API_KEY" not in env
        assert "FAKE_TEST_API_KEY" not in env
        assert not [k for k in env if is_secret_name(k)]
        assert "PATH" in {k.upper() for k in env}
        if call["cmd"][0] == "pdflatex":
            assert "-no-shell-escape" in call["cmd"]


def test_compile_latex_scrubs_keys_and_turns_shell_escape_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", _FAKE)
    monkeypatch.setenv("FAKE_TEST_API_KEY", _FAKE)
    (tmp_path / "paper.tex").write_text(
        "\\documentclass{article}\\begin{document}x\\end{document}\n",
        encoding="utf-8",
    )
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr("src.sandbox.subprocess.run", _recording_run(calls))
    compile_latex(str(tmp_path))
    assert [c["cmd"][0] for c in calls] == ["pdflatex", "bibtex", "pdflatex", "pdflatex"]
    _assert_safe(calls)


@pytest.mark.parametrize("biblatex", [False, True])
def test_the_gate_review_compile_does_the_same(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, biblatex: bool
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", _FAKE)
    monkeypatch.setenv("FAKE_TEST_API_KEY", _FAKE)
    head = "\\usepackage{biblatex}\\addbibresource{r.bib}\n" if biblatex else ""
    (tmp_path / "p.tex").write_text(
        "\\documentclass{article}" + head + "\\begin{document}x\\end{document}\n",
        encoding="utf-8",
    )
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr("src.review_gate.subprocess.run", _recording_run(calls))
    gate = SimpleNamespace(_log=lambda _m: None)
    ReviewGate._compile_review_tex(gate, tmp_path, "p.tex")  # type: ignore[arg-type]
    assert ("biber" in [c["cmd"][0] for c in calls]) is biblatex
    _assert_safe(calls)


def test_argv_keeps_the_mode_every_caller_relies_on() -> None:
    argv = pdflatex_argv("paper.tex")
    assert argv[0] == "pdflatex" and argv[-1] == "paper.tex"
    assert "-interaction=nonstopmode" in argv
    assert "-no-shell-escape" in argv


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex not installed")
@pytest.mark.parametrize("var", ["FAKE_TEST_API_KEY", "EDMARS_PROBE_VALUE"])
def test_a_real_compile_cannot_read_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, var: str
) -> None:
    """End to end with the installed TeX. EDMARS_PROBE_VALUE is not a
    credential name, so only the missing shell escape keeps it out; the
    API-key name is kept out by both defences."""
    monkeypatch.setenv(var, _FAKE)
    (tmp_path / "paper.tex").write_text(_PROBE_TEX % {"var": var}, encoding="utf-8")
    result = compile_latex(str(tmp_path), timeout_s=180)
    log = (tmp_path / "paper.log").read_text(encoding="utf-8", errors="replace")
    assert "PROBE-CLOSED" in log, log[-1500:]
    assert result["pdf_exists"] is True, result["message"]
    for path in tmp_path.iterdir():
        if path.is_file():
            assert _FAKE.encode() not in path.read_bytes(), path.name
