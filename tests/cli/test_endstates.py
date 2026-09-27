"""End-state classification over synthetic run folders, and the catalog."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tests.cli._run_support import (  # also installs stand-ins
    PREDICTION_RESULTS,
    REPO_ROOT,
    alive_pid,
    dead_pid,
    invariants_file,
    log_lines,
    make_run,
    v2_status,
)

from edmars import endstates
from edmars.endstates import classify, code_from_text, messages


# ---------------------------------------------------------------------------
# The catalog covers every code the pipeline can produce
# ---------------------------------------------------------------------------


def test_every_abort_code_has_a_plain_entry() -> None:
    from src.errors import ABORT_CODES

    failures = messages()["failures"]
    required = set(ABORT_CODES) | {
        "LATEX_MISSING", "NO_PDF", "R_MISSING", "LSAR_MISSING", "R_PACKAGES_MISSING",
        "DATA_WRONG_FORMAT", "LEAKAGE_SUSPECTED", "LSAR_FAILED",
    }
    missing = sorted(required - set(failures))
    assert not missing, f"messages.yaml failures lack: {missing}"
    for code, entry in failures.items():
        for key in ("title", "why", "fix", "command"):
            assert isinstance(entry.get(key), str) and entry[key].strip(), (code, key)


def test_every_invariant_code_has_a_title() -> None:
    source = (REPO_ROOT / "src" / "invariants.py").read_text(encoding="utf-8")
    codes = set(re.findall(r'"(INV_[A-Z0-9_]+)"', source))
    titles = messages()["invariants"]
    assert codes, "no invariant codes found in src/invariants.py"
    missing = sorted(codes - set(titles))
    assert not missing, f"messages.yaml invariants lack: {missing}"
    for code, title in titles.items():
        assert title and len(title) < 110 and "INV_" not in title, code


def test_stage_titles_and_reminder_present() -> None:
    msgs = messages()
    for key in ("FORMULATING", "ENGINEERING", "ANALYZING", "CRITIQUING", "REVISING",
                "WRITING", "REVIEWING", "VERIFYING"):
        assert msgs["stages"][key]["title"]
    assert "AI-generated draft" in msgs["reminder"]
    assert "edmars disclaimer" in msgs["reminder"]


def test_printed_paths_keep_their_backslashes_in_git_bash(monkeypatch: pytest.MonkeyPatch) -> None:
    # Unquoted, Git Bash turns D:\EDM-ARS\studies\run into D:EDM-ARSstudiesrun,
    # so a pasted `edmars resume ...` found no study.
    monkeypatch.setattr(endstates.os, "name", "nt")
    study = r"D:\EDM-ARS\studies\2026-09-25_1200_gpa_ab12"
    assert endstates.quote_path(study) == f'"{study}"'
    assert endstates.quote_path("D:/EDM-ARS/studies/run") == "D:/EDM-ARS/studies/run"
    monkeypatch.setattr(endstates.os, "name", "posix")
    assert endstates.quote_path("/srv/studies/run") == "/srv/studies/run"
    assert endstates.quote_path("/srv/my studies/a\\b") == "'/srv/my studies/a\\b'"


def test_fill_leaves_unknown_placeholders() -> None:
    assert endstates.fill("a {x} b {y}", x=1) == "a 1 b {y}"


# ---------------------------------------------------------------------------
# Text heuristics for older runs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, code",
    [
        ("FORMULATING failed: Error code: 402 - {'error': {'message': 'Insufficient Balance'}}", "NO_CREDIT"),
        ("FORMULATING failed: Error code: 401 - {'error': {'message': 'Authentication Fails'}}", "KEY_REJECTED"),
        ("The api_key client option must be set either by passing api_key", "KEY_MISSING"),
        ("ANALYZING failed: Error code: 429 - rate limit reached", "RATE_LIMITED"),
        ("Error code: 404 - model_not_found: deepseek-v4-flash", "MODEL_GONE"),
        ("ENGINEERING failed: [Errno 2] No such file or directory: 'data/raw/hsls_17_student_pets_sr_v1_0.csv'",
         "DATA_MISSING"),
        ("ENGINEERING aborted (validation retry exhausted): validation_passed=False. Warnings: [timeout]",
         "DE_VALIDATION_FAILED"),
        ("ENGINEERING aborted: analytic_n=412 < 1000", "SAMPLE_TOO_SMALL"),
        ("ENGINEERING aborted (causal data contract, post-retry): missing T", "DATA_CONTRACT_FAILED"),
        ("ANALYZING failed: Expecting value: line 1 column 1 (char 0)", "LLM_OUTPUT_UNPARSEABLE"),
        ("ANALYZING failed: something odd", "ANALYSIS_FAILED"),
        ("openai.APIConnectionError: Connection error.", "NETWORK"),
        ("Critic issued ABORT verdict: {...}", "CRITIC_ABORT"),
        ("Pre-Critic guard found critical failures → short-circuit verdict: ABORT", "PRE_CRITIC_ABORT"),
        # fix/pcc-revise's last-cycle stop also logs the short-circuit line.
        ("Pre-Critic guard found critical failures → short-circuit verdict: ABORT\n"
         "Pre-Critic guard stopped the run [PRE_CRITIC_UNRESOLVED]: pcc_07 was still failing",
         "PRE_CRITIC_UNRESOLVED"),
        ("pcc_07 was still failing when the revision cycles ran out (2 of 2 used): x", "PRE_CRITIC_UNRESOLVED"),
        ('{"code": "NO_CREDIT", "message": "x"}', "NO_CREDIT"),
        ("Error in library(mirt) : there is no package called 'mirt'", "R_PACKAGES_MISSING"),
        ("nothing recognisable here", None),
    ],
)
def test_code_from_text(text: str, code: str | None) -> None:
    assert code_from_text(text) == code


def test_the_word_network_alone_is_not_a_network_failure() -> None:
    assert code_from_text("The network of schools was not modelled.") is None


# ---------------------------------------------------------------------------
# classify(): one synthetic folder per outcome
# ---------------------------------------------------------------------------


def _ready_run(root: Path, **kw: object) -> Path:
    defaults: dict = dict(results=PREDICTION_RESULTS,
                          review={"overall_quality_score": 8, "overall_verdict": "PASS"},
                          invariants=invariants_file([]))
    defaults.update(kw)
    return make_run(root, **defaults)


def test_ready(run_home: Path) -> None:
    run = _ready_run(run_home, status=v2_status())
    out = classify(run)
    assert out.kind == "ready" and out.label == "Ready"
    assert out.code is None and out.findings == [] and out.concerns == []


def test_ready_with_serious_issues(run_home: Path) -> None:
    findings = [("INV_PROSE_NUMERAL_UNBOUND", "critical"), ("INV_DANGLING_CITATION_KEY", "critical"),
                ("INV_DANGLING_CITATION_KEY", "critical"), ("INV_PROMISED_ANALYSIS_MISSING", "major"),
                ("INV_HANDTYPED_CROSSREF", "minor")]
    run = _ready_run(run_home, invariants=invariants_file(findings),
                     status=v2_status(reason_code="ADVISORY_FINDINGS",
                                      counts={"critical": 3, "major": 1, "minor": 1}))
    out = classify(run)
    assert out.kind == "ready_with_issues"
    assert out.label == "Ready, with 3 serious issues to check"
    codes = [f["code"] for f in out.findings]
    assert codes == ["INV_DANGLING_CITATION_KEY", "INV_PROSE_NUMERAL_UNBOUND", "INV_PROMISED_ANALYSIS_MISSING"]
    assert out.findings[0]["count"] == 2
    assert out.findings[0]["title"] == messages()["invariants"]["INV_DANGLING_CITATION_KEY"]
    assert "INV_HANDTYPED_CROSSREF" not in codes  # minor findings are not in "Please check"


def test_unknown_invariant_code_is_shown_verbatim(run_home: Path) -> None:
    run = _ready_run(run_home, invariants=invariants_file([("INV_FROM_THE_FUTURE", "critical")]),
                     status=v2_status(reason_code="ADVISORY_FINDINGS", counts={"critical": 1}))
    out = classify(run)
    assert out.label == "Ready, with 1 serious issue to check"
    assert out.findings[0]["title"] == "INV_FROM_THE_FUTURE"


def test_ready_unverified(run_home: Path) -> None:
    run = _ready_run(run_home, status=v2_status(reason_code="CRITIC_UNVERIFIED", critic_unverified=True))
    out = classify(run)
    assert out.kind == "ready_with_issues" and out.label == "Ready, with unresolved concerns"
    assert any("not all resolved" in c for c in out.concerns)


def test_ready_gate_failed(run_home: Path) -> None:
    gate = {"enabled": True, "ran": True, "skip_reason": None, "passed": False, "score": 5.1,
            "threshold": 6.3, "advisory": False, "venue": "EDM"}
    run = _ready_run(run_home, status=v2_status(reason_code="GATE_FAILED", gate=gate))
    out = classify(run)
    assert out.label == "Ready, below the review benchmark"
    assert "5.1" in out.why and "6.3" in out.why and "EDM" in out.why


def test_ready_names_the_literature_source_that_refused(run_home: Path) -> None:
    # arXiv refused (HTTP 406) while Semantic Scholar answered: not
    # "degraded", but the related work rests on one source.
    lit = {"degraded": False, "n_papers": 12,
           "sources": {"semantic_scholar": "ok", "arxiv": "refused", "arxiv_http_status": 406,
                       "n_semantic_scholar": 12, "n_arxiv": 0}}
    out = classify(_ready_run(run_home, status=v2_status(literature=lit)))
    [line] = out.concerns
    assert line.startswith("arXiv refused our requests (HTTP 406), so the related papers all come "
                           "from Semantic Scholar.")
    assert "failed" not in line


def test_few_papers_says_which_sources_turned_the_search_away(run_home: Path) -> None:
    # The Mac test's literature step: every search turned away.
    lit = {"degraded": True, "n_papers": 0,
           "sources": {"semantic_scholar": "rate_limited", "arxiv": "refused", "arxiv_http_status": 406,
                       "n_semantic_scholar": 0, "n_arxiv": 0}}
    out = classify(_ready_run(run_home, status=v2_status(literature=lit)))
    [line] = out.concerns
    assert line.startswith("Few related papers were found: arXiv refused our requests (HTTP 406) and "
                           "Semantic Scholar turned our searches away (too many requests).")
    assert "edmars setup literature" in line
    assert "partly unavailable" not in line
    # Without a per-source status (an older pipeline) the old sentence stays.
    assert endstates.literature_concerns({"degraded": True, "n_papers": 0, "sources": {}}) == [
        "Few related papers were found (the literature search was partly unavailable). "
        "Check the related-work section and the references."]


def test_openalex_standing_in_for_arxiv_is_named_with_its_papers(run_home: Path) -> None:
    # arXiv refused (HTTP 406) and OpenAlex answered in its place: the old
    # line said the related papers "all come from Semantic Scholar".
    lit = {"degraded": False, "n_papers": 21,
           "sources": {"semantic_scholar": "ok", "arxiv": "refused", "arxiv_http_status": 406,
                       "openalex": "ok", "n_semantic_scholar": 12, "n_arxiv": 0, "n_openalex": 9}}
    run = _ready_run(run_home, status=v2_status(literature=lit))
    out = classify(run)
    [line] = out.concerns
    assert line == ("arXiv refused our requests (HTTP 406); OpenAlex supplied 9 papers instead, so the "
                    "related papers come from Semantic Scholar and OpenAlex. Check the related-work "
                    "section and the references.")
    from edmars.results import render_summary_html, result_text
    from edmars.runstate import load_state

    assert "OpenAlex supplied 9 papers instead" in " ".join(result_text(out, load_state(run), run).split())
    html = render_summary_html(out, load_state(run), run)
    assert "OpenAlex supplied 9 papers instead" in html and "all come from Semantic Scholar" not in html
    # The live view's line for the step names OpenAlex too.
    assert "OpenAlex, if arXiv refuses" in " ".join(messages()["stages"]["FORMULATING"]["now"].split())


def test_openalex_alone_is_not_called_few_papers(run_home: Path) -> None:
    # Round 1 on the Mac: Semantic Scholar rate-limited and arXiv refused.
    # With OpenAlex answering, the pool is OpenAlex's, not empty.
    lit = {"degraded": True, "n_papers": 28,
           "sources": {"semantic_scholar": "rate_limited", "arxiv": "refused", "arxiv_http_status": 406,
                       "openalex": "ok", "n_semantic_scholar": 0, "n_arxiv": 0, "n_openalex": 28}}
    [line] = endstates.literature_concerns(lit)
    assert line.startswith("Semantic Scholar turned our searches away (too many requests) and arXiv refused "
                           "our requests (HTTP 406); OpenAlex supplied 28 papers instead, so the related "
                           "papers all come from OpenAlex. Check the related-work section")
    assert "Few related papers" not in line
    assert "edmars setup literature" in line
    # OpenAlex turned away too: few papers, and each source is named.
    lit["n_papers"] = 0
    lit["sources"] = {**lit["sources"], "openalex": "rate_limited", "openalex_http_status": 429,
                      "n_openalex": 0}
    [line] = endstates.literature_concerns(lit)
    assert line.startswith("Few related papers were found: Semantic Scholar turned our searches away (too "
                           "many requests) and arXiv refused our requests (HTTP 406); OpenAlex, asked "
                           "instead, turned our searches away (too many requests).")


def test_ready_gate_not_run_is_never_a_zero_score(run_home: Path) -> None:
    gate = {"enabled": True, "ran": False, "skip_reason": "lsar_not_found", "passed": None,
            "score": None, "threshold": None, "advisory": None, "venue": "EDM"}
    run = _ready_run(run_home, status=v2_status(reason_code="GATE_NOT_RUN", gate=gate))
    out = classify(run)
    assert out.label == "Ready, not reviewed"
    assert out.code == "LSAR_MISSING"
    assert "edmars setup lsar" in (out.command or "")
    assert "0.0" not in out.headline + out.why


def test_a_review_that_could_not_be_scored_is_explained_in_plain_words(run_home: Path) -> None:
    # review_gate.py records "lsar_scoring_failed: <up to 200 characters>";
    # the headline used to be that raw text, cut off mid-sentence, and the
    # explanation blamed the DeepSeek key or the network.
    detail = "Stage 5 scoring failed (the scoring model's answer could not be used: No valid JSON" + "x" * 150
    gate = {"enabled": True, "ran": False, "skip_reason": f"lsar_scoring_failed: {detail}", "passed": None,
            "score": None, "threshold": None, "advisory": None, "venue": "EDM"}
    run = _ready_run(run_home, status=v2_status(reason_code="GATE_NOT_RUN", gate=gate))
    out = classify(run)
    assert out.label == "Ready, not reviewed" and out.code == "LSAR_SCORING_FAILED"
    assert out.headline == ("Your paper is written, but the automated peer review gave no score: "
                            "LSAR wrote a review but could not score it, so there is no score.")
    assert "lsar_scoring_failed" not in out.headline + out.why + out.fix
    assert "key" not in out.why and "network" not in out.why
    assert "edmars review" in (out.command or "")
    no_result = dict(gate, skip_reason="lsar_no_result")
    run2 = _ready_run(run_home / "b", status=v2_status(reason_code="GATE_NOT_RUN", gate=no_result))
    assert classify(run2).headline.endswith("LSAR finished without producing a review.")


def test_released_pipeline_gate_that_could_not_run_is_not_reviewed(run_home: Path) -> None:
    # Released code records a gate that never ran as passed=false, score 0.0.
    run = _ready_run(run_home, review_gate_enabled=True,
                     status={"released": True, "reason": "review gate did not pass",
                             "invariant_counts": {"critical": 0, "major": 0, "minor": 0},
                             "review_gate_passed": False, "review_gate_score": 0.0},
                     gate_summary={"cycles_used": 0, "final_score": 0.0, "passed": False})
    out = classify(run)
    assert out.label == "Ready, not reviewed"


def test_released_pipeline_gate_failed(run_home: Path) -> None:
    run = _ready_run(run_home, review_gate_enabled=True,
                     status={"released": True, "reason": "review gate did not pass",
                             "invariant_counts": {"critical": 0, "major": 0, "minor": 0},
                             "review_gate_passed": False, "review_gate_score": 5.2},
                     gate_summary={"cycles_used": 2, "final_score": 5.2, "passed": False,
                                   "threshold_used": 6.3, "advisory_mode": False, "venue": "EDM"})
    assert classify(run).label == "Ready, below the review benchmark"


def test_released_pipeline_critical_findings(run_home: Path) -> None:
    run = _ready_run(run_home, invariants=invariants_file([("INV_MANUSCRIPT_EMPTY", "critical")]),
                     status={"released": True, "reason": "1 critical invariant finding(s)",
                             "invariant_counts": {"critical": 1, "major": 0, "minor": 0}})
    out = classify(run)
    assert out.label == "Ready, with 1 serious issue to check"


def test_crashed_battery_is_not_clean(run_home: Path) -> None:
    run = _ready_run(run_home, invariants={"enabled": True, "error": "RuntimeError: detector crashed"},
                     status={"released": True, "reason": "clean",
                             "invariant_counts": {}})
    out = classify(run)
    assert out.label == "Ready, final checks did not run"
    assert out.kind == "ready_with_issues"


def test_review_requested_but_lsar_missing_at_launch(run_home: Path) -> None:
    run = _ready_run(run_home, status={"released": True, "reason": "clean",
                                       "invariant_counts": {"critical": 0}},
                     study={"review_requested": True, "review_unavailable": True})
    out = classify(run)
    assert out.label == "Ready, not reviewed" and out.code == "LSAR_MISSING"


@pytest.mark.parametrize("latex_missing, code", [(True, "LATEX_MISSING"), (False, "NO_PDF")])
def test_incomplete_without_pdf(run_home: Path, latex_missing: bool, code: str) -> None:
    lines = [(15, "Compiling paper.tex (pdflatex → bibtex → pdflatex → pdflatex)")]
    if latex_missing:
        lines.append((15, "LaTeX compile step failed: pdflatex -interaction=nonstopmode paper.tex "
                          "(rc=-1): 'pdflatex' not found — is it installed and on PATH?"))
    lines += [(15, "LaTeX compilation had errors — check pipeline.log for details"),
              (16, "VERIFYING: release BLOCKED → INCOMPLETE (1 critical invariant finding(s))")]
    run = make_run(run_home, log=log_lines(*lines), pdf=False,
                   invariants=invariants_file([("INV_LATEX_NO_PDF", "critical")]),
                   status=v2_status("INCOMPLETE", released=False, reason_code="BLOCKING_FINDINGS",
                                    counts={"critical": 1}, blocking_findings=["INV_LATEX_NO_PDF"]))
    out = classify(run)
    assert out.kind == "not_ready" and out.label == "Not ready"
    assert out.code == code
    assert out.commands


def test_released_clean_without_a_pdf_is_not_ready(run_home: Path) -> None:
    # The released pipeline marks this run "clean": pdflatex was missing,
    # so there was no paper.log for the no-PDF check to read (defect B1).
    log = log_lines(
        (15, "LaTeX compile step failed: pdflatex -interaction=nonstopmode paper.tex (rc=-1): "
             "'pdflatex' not found — is it installed and on PATH?"),
        (15, "LaTeX compilation had errors — check pipeline.log for details"),
        (16, "VERIFYING stage complete → COMPLETED (clean)"))
    run = make_run(run_home, log=log, pdf=False, invariants=invariants_file([]),
                   status={"released": True, "reason": "clean", "invariant_counts": {"critical": 0}})
    (run / "paper.tex").write_text("x", encoding="utf-8")
    out = classify(run)
    assert out.kind == "not_ready" and out.code == "LATEX_MISSING"


def test_incomplete_from_checkpoint_errors_only(run_home: Path) -> None:
    run = make_run(run_home, pdf=False, log=log_lines((16, "VERIFYING: release BLOCKED → INCOMPLETE (x)")),
                   checkpoint={"current_state": "INCOMPLETE", "completed_stages": ["VERIFYING"],
                               "errors": ["Release blocked by 1 critical invariant finding(s): INV_LATEX_NO_PDF"]})
    assert classify(run).code == "NO_PDF"


@pytest.mark.parametrize("code", ["NO_CREDIT", "KEY_REJECTED", "DATA_MISSING", "DE_VALIDATION_FAILED",
                                  "CRITIC_ABORT"])
def test_aborted_with_code_from_run_status(run_home: Path, code: str) -> None:
    run = make_run(run_home, pdf=False, log=log_lines((3, "ABORTED: something")),
                   status=v2_status("ABORTED", released=False, reason_code="ABORTED",
                                    abort={"stage": "ENGINEERING", "code": code, "message": "m",
                                           "resumable": True}))
    out = classify(run)
    assert out.kind == "stopped" and out.label == "Stopped"
    assert out.code == code
    entry = messages()["failures"][code]
    assert out.title == entry["title"]
    # The step's title from the step list, not the pipeline's state name.
    assert out.headline == f"{entry['title']} (during: preparing the data)"
    assert out.why and out.fix and out.commands
    assert "{run}" not in (out.command or "") and str(run.name) in (out.command or "") \
        or code in ("CRITIC_ABORT",)


@pytest.mark.parametrize(
    "abort_line, code",
    [
        ("FORMULATING failed: Error code: 402 - {'error': {'message': 'Insufficient Balance'}}", "NO_CREDIT"),
        ("FORMULATING failed: Error code: 401 - Authentication Fails", "KEY_REJECTED"),
        ("ENGINEERING failed: [Errno 2] No such file or directory: 'x/hsls_17_student_pets_sr_v1_0.csv'",
         "DATA_MISSING"),
        ("ENGINEERING aborted: analytic_n=300 < 1000", "SAMPLE_TOO_SMALL"),
        ("ANALYZING failed: weird", "ANALYSIS_FAILED"),
    ],
)
def test_aborted_released_pipeline_from_log_and_checkpoint(run_home: Path, abort_line: str, code: str) -> None:
    run = make_run(run_home, pdf=False, runner=False,
                   log=log_lines((0, "Starting FORMULATING stage"), (1, f"ABORTED: {abort_line}")),
                   checkpoint={"current_state": "ABORTED", "completed_stages": [],
                               "errors": [abort_line], "dataset_name": "hsls09_public"})
    out = classify(run)
    assert out.kind == "stopped" and out.code == code


def test_sample_too_small_is_not_resumable(run_home: Path) -> None:
    run = make_run(run_home, pdf=False,
                   status=v2_status("ABORTED", released=False, reason_code="ABORTED",
                                    abort={"stage": "ENGINEERING", "code": "SAMPLE_TOO_SMALL",
                                           "message": "", "resumable": False}))
    out = classify(run)
    assert out.resumable is False
    assert out.command == "edmars new"


def test_critic_abort_citing_leakage(run_home: Path) -> None:
    run = make_run(run_home, pdf=False, runner=False,
                   log=log_lines((12, "Critic verdict: ABORT → pipeline aborted")),
                   checkpoint={"current_state": "ABORTED", "errors": ["Critic issued ABORT verdict: {}"]},
                   review={"overall_verdict": "ABORT",
                           "analysis_review": {"issues": [{"description": "AUC 0.99 suggests target leakage"}]}})
    assert classify(run).code == "LEAKAGE_SUSPECTED"


def test_interrupted_from_run_status(run_home: Path) -> None:
    run = make_run(run_home, pdf=False,
                   status=v2_status("INTERRUPTED", released=False, reason_code="INTERRUPTED",
                                    abort={"stage": "ANALYZING", "code": "INTERRUPTED", "message": "",
                                           "resumable": True}))
    out = classify(run)
    assert out.kind == "stopped" and out.code == "INTERRUPTED"
    assert "edmars resume" in (out.command or "")


def test_stopped_by_user_and_crashed(run_home: Path) -> None:
    log = log_lines((0, "Starting FORMULATING stage"), (1, "FORMULATING stage complete"),
                    (1, "Starting ENGINEERING stage"))
    run = make_run(run_home, pid=dead_pid(), log=log, pdf=False)
    out = classify(run)
    assert out.code == "CRASHED" and out.label == "Stopped"
    (run / "STOP").write_text("x", encoding="utf-8")
    assert classify(run).code == "INTERRUPTED"


def test_crash_log_names_the_cause(run_home: Path) -> None:
    run = make_run(run_home, pid=dead_pid(), pdf=False,
                   log=log_lines((0, "Starting FORMULATING stage")),
                   extra={"crash.log": "Traceback...\nopenai.APIStatusError: Error code: 402 - Insufficient Balance\n"})
    assert classify(run).code == "NO_CREDIT"


def test_process_that_died_at_import_names_the_missing_part(run_home: Path) -> None:
    console = ("Traceback (most recent call last):\n  File \"<frozen runpy>\", line 198\n"
               "ModuleNotFoundError: No module named 'xgboost'\n")
    run = make_run(run_home, pid=dead_pid(), pdf=False, log=None, data_report=None,
                   extra={"console.log": console})
    out = classify(run)
    assert out.code == "INSTALL_BROKEN"
    assert "No module named 'xgboost'" in out.why
    assert out.commands[0] == "edmars doctor"


def test_quoted_errors_are_redacted(run_home: Path) -> None:
    secret = "sk-proj-" + "A1b2C3d4" * 4
    console = f"openai.AuthenticationError: Incorrect API key provided: {secret}. Bearer {secret}\n"
    run = make_run(run_home, pid=dead_pid(), pdf=False, log=None, data_report=None,
                   extra={"console.log": console})
    out = classify(run)
    blob = " ".join([out.headline, out.why, out.fix, out.command or ""])
    assert secret not in blob and "A1b2C3d4" not in blob
    # The shared redactor in edmars.secrets writes "[REDACTED]"; the
    # endstates fallback writes "[redacted]". Either is fine.
    redacted = endstates.redact(f"api_key={secret}")
    assert secret not in redacted and redacted.lower() == "api_key=[redacted]"


def test_last_error_line() -> None:
    from edmars.endstates import last_error_line

    assert last_error_line("x\nValueError: config.yaml missing required keys\n") == \
        "ValueError: config.yaml missing required keys"
    assert last_error_line("openai.AuthenticationError: Error code: 401") == \
        "openai.AuthenticationError: Error code: 401"
    assert last_error_line("all good") is None


def test_running(run_home: Path) -> None:
    run = make_run(run_home, pid=alive_pid(), pdf=False,
                   log=log_lines((0, "Starting FORMULATING stage"), (1, "FORMULATING stage complete"),
                                 (1, "Starting ENGINEERING stage")))
    out = classify(run)
    assert out.kind == "running" and out.label == "Still running"
    assert "step 2 of 7" in out.headline
    assert out.command and out.command.startswith("edmars status")


def test_missing_folder(tmp_path: Path) -> None:
    out = classify(tmp_path / "nope")
    assert out.kind == "stopped" and out.command == "edmars runs"


def test_labels_never_say_release_yes(run_home: Path) -> None:
    run = _ready_run(run_home, invariants=invariants_file([("INV_MANUSCRIPT_EMPTY", "critical")]),
                     status=v2_status(reason_code="ADVISORY_FINDINGS", counts={"critical": 1}))
    out = classify(run)
    blob = " ".join([out.label, out.headline, out.why, out.fix, out.command or ""])
    assert "Release:" not in blob and "YES" not in blob


@pytest.mark.parametrize(("dataset", "first_command"), [
    ("hsls09_public", "edmars data install hsls09_public"),
    # ASSISTments cannot be downloaded yet: `install` only says "coming later".
    ("assistments_0910", "edmars data import assistments_0910 <path to the .csv file>"),
])
def test_data_missing_names_a_command_that_works(run_home: Path, dataset: str,
                                                  first_command: str) -> None:
    run = make_run(run_home, pdf=False, log=log_lines((3, "ABORTED: something")),
                   study={"dataset": dataset},
                   status=v2_status("ABORTED", released=False, reason_code="ABORTED",
                                    abort={"stage": "ENGINEERING", "code": "DATA_MISSING",
                                           "message": "m", "resumable": True}))
    out = classify(run)
    assert out.commands[0] == first_command
    assert out.commands[-1].startswith("edmars resume")



# ---------------------------------------------------------------------------
# A study the automatic checks or the reviewer stopped: say what they found
# ---------------------------------------------------------------------------

PCC_07 = ("pcc_07: The research question says 'above and beyond', which commits the paper to an "
          "incremental-validity / nested-model comparison, but no such analysis appears in "
          "results.json. Either run it, or change the research question so it does not promise a "
          "test the study never performed.")
USER_Q = ("Which ninth-grade factors predict whether a student enrolls in college by 2016, and is "
          "the prediction equally accurate across sex and socioeconomic groups?")
WORKED_Q = ("Do ninth-grade non-cognitive factors predict college enrollment by February 2016 "
            "ABOVE AND BEYOND academic achievement and socioeconomic status?")


def _pre_critic_run(run_home: Path, *, resumable: bool = False, second: bool = True,
                    code: str = "PRE_CRITIC_ABORT", message: str = PCC_07,
                    status: bool = True) -> Path:
    issues = [{"severity": "critical", "category": "pcc_07", "description": PCC_07.split(": ", 1)[1],
               "recommendation": "x", "target_agent": "Analyst"}]
    if second:
        issues.append({"severity": "critical", "category": "pcc_02",
                       "description": "No individual models are present in results.json.",
                       "recommendation": "x", "target_agent": "Analyst"})
    issues.append({"severity": "major", "category": "pcc_05", "description": "subgroup gap",
                   "recommendation": "x", "target_agent": "Analyst"})
    report = {"overall_verdict": "ABORT", "overall_quality_score": 1, "_source": "pre_critic_short_circuit",
              "analysis_review": {"score": 1, "issues": issues}}
    run_status = v2_status("ABORTED", released=False, reason_code="ABORTED",
                           abort={"stage": "CRITIQUING", "code": code,
                                  "message": message, "resumable": resumable})
    log = log_lines((9, "Pre-Critic guard found critical failures → short-circuit verdict: ABORT"),
                    (9, f"Pre-Critic guard stopped the run [{code}]: {message}"))
    return make_run(run_home, pdf=False, question=USER_Q, log=log,
                    status=run_status if status else None,
                    checkpoint={"current_state": "ABORTED", "review_report": report,
                                "errors": [f"Pre-Critic guard: {code} with 2 of 2 revision cycles used"]},
                    extra={"research_spec.json": json.dumps({"research_question": WORKED_Q})})


def test_a_pre_critic_abort_says_what_the_checks_found_and_advice_that_fits(run_home: Path) -> None:
    # The Mac study: the screen gave only a generic reason and advised
    # "Start a new study, possibly with a simpler question", although the
    # researcher's question was simple and pcc_07's sentence said exactly
    # what was missing.
    # Its record predates abort.checks and is led by pcc_07, which this
    # version sends back for revision, so the study can now be resumed
    # (src/errors.py reopened_pre_critic_stop); the checks' own advice
    # follows for the case it stops again.
    out = classify(_pre_critic_run(run_home))
    assert out.code == "PRE_CRITIC_ABORT" and out.resumable is True
    assert out.details_heading == "What the automatic checks found:"
    assert out.details[0] == PCC_07.split(": ", 1)[1]  # in full, without "pcc_07:"
    assert out.details[1] == "No individual models are present in results.json."
    assert len(out.details) == 2  # a major finding did not stop the study
    assert out.note == f'The study worded your question as: "{WORKED_Q}"'
    assert "simpler question" not in out.fix
    assert out.fix.startswith("This version of EDM-ARS sends this finding back to be fixed "
                              "instead of stopping the study. Resume the study")
    assert "If it stops again: the question the study worked from promised a comparison" in out.fix
    assert "no trained models" in out.fix  # pcc_02's advice too
    assert out.command is not None and out.command.startswith("edmars resume ")


@pytest.mark.parametrize("lead", ["pcc_01", "pcc_06", "pcc_99"])
def test_an_old_stop_no_revision_can_fix_stays_final(run_home: Path, lead: str) -> None:
    out = classify(_pre_critic_run(run_home, second=False, message=f"{lead}: something"))
    assert out.resumable is False and out.command == "edmars new"


def test_an_old_stop_read_from_the_log_alone_stays_final(run_home: Path) -> None:
    # Without run_status.json there is no abort record to recognise it by.
    out = classify(_pre_critic_run(run_home, second=False, status=False))
    assert out.code == "PRE_CRITIC_ABORT" and out.resumable is False


def test_a_pre_critic_abort_says_what_was_found_in_its_why(run_home: Path) -> None:
    # The "why" claimed "a problem a revision cannot fix" for every check,
    # although pcc_07 is one a revision can fix (fix/pcc-revise sends it
    # back to the Analyst). It now names what the checks found.
    out = classify(_pre_critic_run(run_home))
    assert out.why == ("Built-in checks that run before the methods review found a test the question "
                       "promises missing from the analysis and no trained model in the results, so a "
                       "paper written from these results would mislead.")
    assert "cannot fix" not in out.why and "{" not in out.why
    only_07 = classify(_pre_critic_run(run_home / "b", second=False))
    assert "found a test the question promises missing from the analysis, so" in only_07.why


def test_an_unknown_check_is_pointed_to_not_invented(run_home: Path) -> None:
    run = _pre_critic_run(run_home, second=False, message="pcc_99: something new")
    review = json.loads((run / "checkpoint.json").read_text(encoding="utf-8"))
    review["review_report"]["analysis_review"]["issues"][0]["category"] = "pcc_99"
    (run / "checkpoint.json").write_text(json.dumps(review), encoding="utf-8")
    out = classify(run)
    assert "found a problem with these results (listed below)" in out.why
    assert out.fix == "Read what the checks found, then start a new study that avoids it."


UNRESOLVED = ("pcc_07 was still failing when the revision cycles ran out (2 of 2 used): "
              + PCC_07.split(": ", 1)[1])


def test_a_finding_still_failing_after_the_revisions_is_not_resumable(run_home: Path) -> None:
    # fix/pcc-revise: pcc_07 goes back to the Analyst; when the revision
    # rounds run out it stops as PRE_CRITIC_UNRESOLVED, with no paper.
    # Without an entry the CLI fell back to UNKNOWN: "Resume once".
    out = classify(_pre_critic_run(run_home, code="PRE_CRITIC_UNRESOLVED", message=UNRESOLVED,
                                   second=False))
    assert out.code == "PRE_CRITIC_UNRESOLVED" and out.resumable is False
    assert out.title == "Automatic checks stopped the study after its revisions"
    assert "a test the question promises missing from the analysis." in out.why
    assert "the revisions did not fix it" in out.why and "no paper was written" in out.why
    assert out.details_heading == "What the automatic checks found:"
    assert out.details == [PCC_07.split(": ", 1)[1]]  # without the "pcc_07 was still failing" lead
    assert out.fix.startswith("Resuming would not help, because no revision rounds are left. "
                              "The question the study worked from promised")
    assert out.command == "edmars new"
    assert out.note == f'The study worded your question as: "{WORKED_Q}"'


def test_a_stop_before_the_rounds_ran_out_does_not_claim_they_did(run_home: Path) -> None:
    # The pipeline stops with rounds left when the revision comes back
    # saying another would not fix it; "no revision rounds are left" and
    # "used all its revision rounds" were both untrue of that stop.
    early = ("pcc_07 was still failing after revision 1 of 2, and another revision would not "
             "change it: the Analyst recorded that the test cannot run: baseline block is empty. "
             + PCC_07.split(": ", 1)[1])
    out = classify(_pre_critic_run(run_home, code="PRE_CRITIC_UNRESOLVED", message=early,
                                   second=False))
    assert out.code == "PRE_CRITIC_UNRESOLVED" and out.resumable is False
    assert out.fix.startswith("Resuming would not help: the step the work went back to "
                              "reported that another revision would not fix it. The question")
    assert "rounds" not in out.why and "rounds are left" not in out.fix


PCC_01_TEXT = "Outcome variable 'X4EVRATNDCLG' found as a column in train_X.csv - confirmed target leakage."
PCC_07_FULL = (PCC_07.split(": ", 1)[1] + " A paper whose central question is never tested reads "
               "fluently and is rejected on rigor.")


def _abort_checks_run(run_home: Path) -> Path:
    """A PRE_CRITIC_ABORT as fix/pcc-revise writes it: run_status.json's
    abort.checks holds every critical finding in full, and the checkpoint
    holds no short-circuit review (a resumed or cleaned-up run)."""
    checks = [
        {"check_id": "pcc_07", "severity": "critical", "message": PCC_07_FULL,
         "target_agent": "Analyst", "revisable": True},
        {"check_id": "pcc_01", "severity": "critical", "message": PCC_01_TEXT,
         "target_agent": "DataEngineer", "revisable": False},
    ]
    run_status = v2_status("ABORTED", released=False, reason_code="ABORTED",
                           abort={"stage": "CRITIQUING", "code": "PRE_CRITIC_ABORT",
                                  "message": f"pcc_01: {PCC_01_TEXT}", "resumable": False,
                                  "checks": checks})
    return make_run(run_home, pdf=False, question=USER_Q, log=None, status=run_status,
                    checkpoint={"current_state": "ABORTED"},
                    extra={"research_spec.json": json.dumps({"research_question": WORKED_Q})})


def test_the_result_screen_lists_every_finding_in_abort_checks(run_home: Path) -> None:
    # The screen took the first finding from abort.message and the rest
    # from the checkpoint's short-circuit review; without that review it
    # showed one finding of two. run_status.json has had them all since
    # fix/pcc-revise.
    run = _abort_checks_run(run_home)
    out = classify(run)
    assert out.code == "PRE_CRITIC_ABORT" and out.resumable is False
    assert out.details == [PCC_01_TEXT, PCC_07_FULL]  # the stop's own finding first, each in full
    assert "the outcome among the predictors (data leakage)" in out.why
    assert "a test the question promises missing from the analysis" in out.why
    # The leakage stopped the study; pcc_07 would have been sent back for
    # revision, so its advice is not what to do about this stop.
    assert out.fix.startswith("The outcome itself ended up among the predictors")
    assert "ask in plain words" not in out.fix

    from edmars.results import render_summary_html
    from edmars.runstate import load_state

    html = render_summary_html(out, load_state(run), run)
    assert "rejected on rigor." in html and "confirmed target leakage." in html


def test_abort_checks_is_read_only_for_the_automatic_checks_stops(run_home: Path) -> None:
    assert endstates.abort_checks(None) == []
    assert endstates.abort_checks({"abort": {"code": "PRE_CRITIC_ABORT", "checks": "not a list"}}) == []
    assert endstates.abort_checks({"abort": {"checks": [{"check_id": "pcc_07"}, 3]}}) == []


def test_an_unresolved_stop_read_from_the_log_alone_is_not_resumable(run_home: Path) -> None:
    # No run_status.json: the code comes from checkpoint errors and
    # pipeline.log, and resumability from _NOT_RESUMABLE.
    out = classify(_pre_critic_run(run_home, code="PRE_CRITIC_UNRESOLVED", message=UNRESOLVED,
                                   second=False, status=False))
    assert out.code == "PRE_CRITIC_UNRESOLVED"
    assert out.resumable is False and out.command == "edmars new"


def test_a_pre_critic_abort_the_pipeline_calls_resumable_says_resume(run_home: Path) -> None:
    out = classify(_pre_critic_run(run_home, resumable=True, second=False))
    assert out.resumable is True
    assert out.command is not None and out.command.startswith("edmars resume ")
    assert out.fix.startswith("Resume the study to let it try again")
    assert "the question the study worked from promised" in out.fix


def test_a_critic_abort_lists_the_reviewers_critical_notes(run_home: Path) -> None:
    review = {"overall_verdict": "ABORT", "overall_quality_score": 2,
              "problem_formulation_review": {"issues": [
                  {"severity": "critical", "description": "The outcome is measured before the predictors."},
                  {"severity": "minor", "description": "Wording."}]}}
    run = make_run(run_home, pdf=False, review=review,
                   status=v2_status("ABORTED", released=False, reason_code="ABORTED",
                                    abort={"stage": "CRITIQUING", "code": "CRITIC_ABORT",
                                           "message": "Critic ABORT: The outcome is measured before the predictors.",
                                           "resumable": False}))
    out = classify(run)
    assert out.code == "CRITIC_ABORT"
    assert out.details == ["The outcome is measured before the predictors."]
    assert out.details_heading == "What the reviewer found:"
    assert out.note == "" and out.command == "edmars new"


def test_other_stops_carry_no_review_details(run_home: Path) -> None:
    run = make_run(run_home, pdf=False,
                   status=v2_status("ABORTED", released=False, reason_code="ABORTED",
                                    abort={"stage": "ENGINEERING", "code": "NO_CREDIT", "message": "m",
                                           "resumable": True}))
    out = classify(run)
    assert out.details == [] and out.note == ""
