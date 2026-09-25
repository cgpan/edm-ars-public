"""Result screens (success and failure) and summary.html."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.cli._run_support import (  # also installs stand-ins
    PREDICTION_RESULTS,
    alive_pid,
    invariants_file,
    log_lines,
    make_run,
    v2_status,
)

from edmars import results

REMINDER_START = "This is an AI-generated draft. Check every number and citation before sharing"


def _ready(root: Path, **kw: object) -> Path:
    base: dict = dict(results=PREDICTION_RESULTS,
                      review={"overall_quality_score": 8, "overall_verdict": "PASS"},
                      invariants=invariants_file([]), status=v2_status(),
                      extra={"shap_summary.png": "png", "roc_curves.png": "png"})
    base.update(kw)
    return make_run(root, **base)


def _show(run: Path, capsys: pytest.CaptureFixture[str], **kw: object) -> tuple[int, str]:
    code = results.show(run, **kw)  # type: ignore[arg-type]
    return code, capsys.readouterr().out


def test_success_screen(run_home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run = _ready(run_home)
    code, out = _show(run, capsys)
    assert code == 0
    assert out.startswith("[ok] Ready")
    assert "Key result: Best model: XGBoost, AUC 0.78 (95% CI 0.76-0.80), on 3,467 students" in out
    assert "Internal methods review: 8/10, passed" in out
    assert "Final checks: no problems found" in out
    assert "paper.pdf" in out and "Your paper" in out
    assert "2 figure file(s)" in out
    assert "edmars results" in out and "--open pdf" in out
    assert "edmars new" in out
    assert REMINDER_START in " ".join(out.split())
    assert "Release:" not in out
    assert (run / "summary.html").exists()


def test_success_with_serious_issues_lists_them(run_home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run = _ready(run_home,
                 invariants=invariants_file([("INV_DANGLING_CITATION_KEY", "critical"),
                                             ("INV_PROMISED_ANALYSIS_MISSING", "major")]),
                 status=v2_status(reason_code="ADVISORY_FINDINGS", counts={"critical": 1, "major": 1}))
    code, out = _show(run, capsys)
    assert code == 0
    assert out.startswith("[!] Ready, with 1 serious issue to check")
    assert "Please check" in out
    flat = " ".join(out.split())
    assert "[x] The paper cites a reference that is not in the reference list [INV_DANGLING_CITATION_KEY]" in flat
    assert "[!] An analysis the study plan promised is not in the paper" in flat
    assert "Final checks: 1 serious, 1 to check" in out
    assert "Release:" not in out and "YES" not in out


def test_gate_scores_on_success_screen(run_home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    gate = {"enabled": True, "ran": True, "skip_reason": None, "passed": False, "score": 5.1,
            "threshold": 6.3, "advisory": False, "venue": "EDM"}
    run = _ready(run_home, review_gate_enabled=True,
                 gate_summary={"cycles_used": 2, "final_score": 5.1, "passed": False, "threshold_used": 6.3,
                               "advisory_mode": False, "venue": "EDM", "ran": True},
                 status=v2_status(reason_code="GATE_FAILED", gate=gate))
    _, out = _show(run, capsys)
    assert "Ready, below the review benchmark" in out
    assert "Automated peer review (LSAR): 5.1 out of 10, below the benchmark of 6.3" in " ".join(out.split())


@pytest.mark.parametrize("code, expected_rc", [("NO_CREDIT", 3), ("DATA_MISSING", 3)])
def test_failure_screen(run_home: Path, capsys: pytest.CaptureFixture[str], code: str, expected_rc: int) -> None:
    run = make_run(run_home, pdf=False,
                   status=v2_status("ABORTED", released=False, reason_code="ABORTED",
                                    abort={"stage": "FORMULATING", "code": code, "message": "x",
                                           "resumable": True}))
    rc, out = _show(run, capsys)
    assert rc == expected_rc
    assert out.startswith("[x] Stopped:")
    assert "What happened:" in out and "Why:" in out and "What to do:" in out
    assert "Type" in out and "edmars" in out
    assert "Your finished steps are saved." in out
    assert REMINDER_START in " ".join(out.split())


def test_not_ready_screen(run_home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run = make_run(run_home, pdf=False,
                   invariants=invariants_file([("INV_LATEX_NO_PDF", "critical")]),
                   status=v2_status("INCOMPLETE", released=False, reason_code="BLOCKING_FINDINGS",
                                    counts={"critical": 1}, blocking_findings=["INV_LATEX_NO_PDF"]))
    rc, out = _show(run, capsys)
    assert rc == 2
    assert out.startswith("[x] Not ready: The PDF could not be made")
    assert "No PDF was produced" in out


def test_running_screen(run_home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run = make_run(run_home, pid=alive_pid(), pdf=False, log=log_lines((0, "Starting FORMULATING stage")))
    rc, out = _show(run, capsys)
    assert rc == 0
    assert "Still running" in out and "edmars status" in out
    assert not (run / "summary.html").exists()


def test_open_actions(run_home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from edmars import ui

    opened: list[Path] = []
    run = _ready(run_home)
    orig = ui.open_path
    try:
        ui.open_path = lambda p: opened.append(Path(p))  # type: ignore[assignment]
        assert results.show(run, open_="pdf") == 0
        assert results.show(run, open_="folder") == 0
        assert results.show(run, open_="summary") == 0
        assert results.show(run, open_="bogus") == 1
    finally:
        ui.open_path = orig  # type: ignore[assignment]
    assert opened == [run / "paper.pdf", run, run / "summary.html"]
    capsys.readouterr()


def test_summary_html_is_self_contained_and_escaped(run_home: Path) -> None:
    run = _ready(run_home, question='Does <script>alert("x")</script> & friends predict GPA?',
                 invariants=invariants_file([("INV_PROSE_NUMERAL_UNBOUND", "critical")]),
                 status=v2_status(reason_code="ADVISORY_FINDINGS", counts={"critical": 1}))
    path = results.write_summary_html(run)
    html_text = path.read_text(encoding="utf-8")
    assert "<script>" not in html_text
    assert "&lt;script&gt;" in html_text and "&amp; friends" in html_text
    assert '<img src="roc_curves.png"' in html_text and '<img src="shap_summary.png"' in html_text
    assert 'href="paper.pdf"' in html_text
    assert str(run_home) not in html_text  # relative links only, no local paths
    assert not re.search(r'(src|href)="(https?:|file:|/)', html_text)  # nothing external
    assert "INV_PROSE_NUMERAL_UNBOUND evidence &lt;here&gt;" in html_text
    assert "AI-generated draft" in html_text
    assert "Release:" not in html_text


@pytest.mark.parametrize("width", [60, 80, 120])
def test_result_text_wraps(run_home: Path, width: int) -> None:
    from edmars.endstates import classify
    from edmars.runstate import load_state

    run = _ready(run_home)
    text = results.result_text(classify(run), load_state(run), run, plain=True, width=width)
    long_lines = [ln for ln in text.splitlines() if len(ln) > width and " " in ln.strip()
                  and str(run) not in ln]
    assert not long_lines


def test_causal_and_psychometric_result_sentences(run_home: Path) -> None:
    from edmars.runstate import load_state

    causal = make_run(run_home, name="c", task_type="causal_soo",
                      results={"estimand": "ATT", "estimates": {"M4": {
                          "method_name": "M4 AIPW", "point_estimate": 0.12, "ci_lower": 0.05, "ci_upper": 0.19}}},
                      extra={"research_spec.json": '{"primary_method": "M4", "research_question": "q"}'})
    sentence = results.result_sentence(load_state(causal))
    assert sentence == "Estimated effect (ATT): 0.12 (95% CI 0.05–0.19), from M4 AIPW."
    psy = make_run(run_home, name="p", task_type="psychometrics",
                   results={"headline": "The scale functions equivalently across sex.",
                            "measurement_results": {}})
    assert results.result_sentence(load_state(psy)) == "The scale functions equivalently across sex."
