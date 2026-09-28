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


@pytest.mark.parametrize("severity", ["critical", "major"])
def test_the_why_line_points_at_the_list_that_is_above_it(
        run_home: Path, capsys: pytest.CaptureFixture[str], severity: str) -> None:
    # The round-3 screen said "Why: ... Each item below is something the
    # paper states ...", under the "Please check" list it meant. A serious
    # finding moves the list above the scores; either way it is above Why.
    run = _ready(run_home, invariants=invariants_file([("INV_SUPERLATIVE_CONTRADICTED", severity)]),
                 status=v2_status(reason_code="ADVISORY_FINDINGS",
                                  counts={"critical": int(severity == "critical"),
                                          "major": int(severity == "major")}))
    _, out = _show(run, capsys)
    lines = [line.strip() for line in out.splitlines()]
    why = next(i for i, line in enumerate(lines) if line.startswith("Why:"))
    assert lines.index("Please check") < why
    flat = " ".join(out.split())
    said = flat[flat.index("Why: "):flat.index("What to do: ")]
    assert "Each item in the list above is something the paper states" in said
    assert "below" not in said

    html_text = (run / "summary.html").read_text(encoding="utf-8")
    assert "Each item in the list above is something the paper states" in html_text
    assert html_text.index("<h2>Please check</h2>") < html_text.index("Each item in the list above")
    assert "Each item below" not in html_text


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


def test_an_experimental_plan_is_labelled_on_every_result(run_home: Path,
                                                         capsys: pytest.CaptureFixture[str]) -> None:
    # runner.json recorded study.experimental, but only the confirmation
    # card ever showed it: a finished menu-built study read as a plain "Ready".
    from edmars import view
    from edmars.endstates import classify
    from edmars.runstate import load_state

    run = _ready(run_home, study={"experimental": True})
    _, out = _show(run, capsys)
    lines = out.splitlines()
    assert lines[0] == "[ok] Ready"
    assert lines[1].startswith("[EXPERIMENTAL] not a tested example study.")
    state = load_state(run)
    html_text = results.render_summary_html(classify(run), state, run)
    assert "[EXPERIMENTAL] not a tested example study" in html_text
    assert "[EXPERIMENTAL]" in view.screen_text(state, width=80, plain=True)

    plain = _ready(run_home / "b")
    _, out = _show(plain, capsys)
    assert "EXPERIMENTAL" not in out
    assert "EXPERIMENTAL" not in results.render_summary_html(classify(plain), load_state(plain), plain)


def test_a_review_that_did_not_run_is_explained_and_setup_comes_first(
    run_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    gate = {"enabled": True, "ran": False, "skip_reason": "lsar_not_found: /x/y", "passed": None,
            "score": None, "threshold": None, "advisory": None, "venue": "EDM"}
    run = _ready(run_home, review_gate_enabled=True,
                 gate_summary={"ran": False, "skip_reason": "lsar_not_found: /x/y", "venue": "EDM"},
                 status=v2_status(reason_code="GATE_NOT_RUN", gate=gate))
    _, out = _show(run, capsys)
    flat = " ".join(out.split())
    assert ("Automated peer review (LSAR): did not run. LSAR is not installed where "
            "EDM-ARS expects it.") in flat
    assert "lsar_not_found" not in out and "/x/y" not in out
    lines = [line.strip() for line in out.splitlines()]
    assert lines.index("First set up LSAR:") < lines.index("Automated peer review:")


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
    from edmars.endstates import classify, quote_path
    from edmars.runstate import load_state

    run = _ready(run_home)
    text = results.result_text(classify(run), load_state(run), run, plain=True, width=width)
    long_lines = [ln for ln in text.splitlines() if len(ln) > width and " " in ln.strip()
                  and not ln.lstrip().startswith("edmars ")]
    assert not long_lines
    # commands are whole lines, so they can be copied as they are
    assert f"    edmars results {quote_path(run)} --open pdf" in text.splitlines()


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


def test_an_experimental_plan_is_labelled_on_the_result_and_summary(
        run_home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run = _ready(run_home, study={"experimental": True})
    code, out = _show(run, capsys)
    assert code == 0
    lines = out.splitlines()
    assert lines[0].startswith("[ok] Ready")
    assert lines[1].startswith("[EXPERIMENTAL] not a tested example study")
    html_text = (run / "summary.html").read_text(encoding="utf-8")
    assert "[EXPERIMENTAL] not a tested example study" in html_text
    assert "not been run end to end" in html_text
    failed = make_run(run_home, name="2026-09-25_1400_failed_ef01", pdf=False,
                      study={"experimental": True},
                      status=v2_status("ABORTED", released=False, reason_code="ABORTED",
                                       abort={"stage": "FORMULATING", "code": "NO_CREDIT",
                                              "message": "x", "resumable": True}))
    _, out = _show(failed, capsys)
    assert out.splitlines()[1].startswith("[EXPERIMENTAL] not a tested example study")


def test_a_tested_plan_carries_no_experimental_label(
        run_home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run = _ready(run_home)
    _, out = _show(run, capsys)
    assert "EXPERIMENTAL" not in out
    assert "EXPERIMENTAL" not in (run / "summary.html").read_text(encoding="utf-8")



def test_a_study_the_checks_stopped_shows_what_they_found(run_home: Path,
                                                          capsys: pytest.CaptureFixture[str]) -> None:
    from tests.cli.test_endstates import PCC_07, WORKED_Q, _pre_critic_run

    run = _pre_critic_run(run_home, second=False)
    code, out = _show(run, capsys)
    assert code == 3
    flat = " ".join(out.split())
    assert "What the automatic checks found: - " + PCC_07.split(": ", 1)[1] in flat
    assert f'The study worded your question as: "{WORKED_Q}"' in flat
    assert "simpler question" not in flat
    # The record predates abort.checks and is led by pcc_07, which a
    # resume now sends back for revision.
    assert "What to do: This version of EDM-ARS sends this finding back to be fixed" in flat
    assert "If it stops again: the question the study worked from promised a comparison" in flat
    html = (run / "summary.html").read_text(encoding="utf-8")
    assert "What the automatic checks found:" in html and "above and beyond" in html



def test_the_result_screen_and_summary_word_the_cost_as_the_live_view(
    run_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The Mac study: the live view said "at least US$0.054 (11 AI calls)",
    # summary.html "US$0.054 (11 AI calls)", and the result screen nothing.
    # run_cost.json counts answered calls only (10), so the cut-off call
    # stays on top of it.
    import json as _json

    from tests.cli._run_support import event

    events = [
        event(1, "run.start", 0, task_type="prediction", dataset="hsls09_public", provider="deepseek"),
        event(2, "stage.start", 0, stage="ANALYZING"),
        event(3, "llm.end", 1, ok=True, cost_usd=0.02),
        event(4, "llm.end", 2, ok=False, error_class="_StopRequested", cost_usd=None),
        event(5, "stage.end", 2, stage="ANALYZING", outcome="interrupted"),
        event(6, "run.end", 2, state="INTERRUPTED"),
    ]
    run = make_run(run_home, pdf=False, log=None, events=events,
                   status=v2_status("INTERRUPTED", released=False, reason_code="INTERRUPTED",
                                    abort={"stage": "ANALYZING", "code": "INTERRUPTED", "message": "",
                                           "resumable": True}),
                   extra={"run_cost.json": _json.dumps({"n_calls": 1, "cost_usd": 0.0216})})
    code, out = _show(run, capsys)
    assert code == 3
    line = "Cost: at least US$0.022 (2 AI calls, 1 cut off when the study was stopped)"
    flat = " ".join(out.split())
    assert line in flat and "may still be billed by the AI service" in flat
    html = (run / "summary.html").read_text(encoding="utf-8")
    assert line in html and "may still be billed by the AI service" in html


# The round-3 Mac study: results.json's best_model named a model its own
# all_models did not put first, the paper repeated the claim, and so did
# the result screen's "Key result" line.
CONTRADICTED_RESULTS = {
    "best_model": "XGBoost",
    "best_metric_value": 0.781,
    "primary_metric": "AUC",
    "all_models": {
        "LogisticRegression": {"auc": 0.74, "auc_ci_lower": 0.72, "auc_ci_upper": 0.76},
        "RandomForest": {"auc": 0.812, "auc_ci_lower": 0.794, "auc_ci_upper": 0.83},
        "XGBoost": {"auc": 0.781, "auc_ci_lower": 0.762, "auc_ci_upper": 0.80},
    },
}


def test_the_key_result_comes_from_the_metrics_when_the_claim_contradicts_them(
        run_home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run = _ready(run_home, results=CONTRADICTED_RESULTS)
    _, out = _show(run, capsys)
    flat = " ".join(out.split())
    expected = ("Key result: Best by AUC: RandomForest (AUC 0.81, 95% CI 0.79-0.83), on 3,467 "
                "students held out for testing; the analysis named XGBoost (AUC 0.78) as its best "
                "model - check the paper's claim.")
    assert expected in flat
    assert "Best model: XGBoost" not in flat
    html_text = (run / "summary.html").read_text(encoding="utf-8")
    assert "Best by AUC: RandomForest (AUC 0.81, 95% CI 0.79–0.83)" in html_text
    assert "the analysis named XGBoost (AUC 0.78) as its best model — check the paper&#x27;s claim." \
        in html_text


def test_a_lower_is_better_metric_picks_the_smallest_error(run_home: Path) -> None:
    from edmars.runstate import load_state

    run = _ready(run_home, results={
        "best_model": "LinearRegression", "best_metric_value": 0.71, "primary_metric": "RMSE",
        "all_models": {"LinearRegression": {"rmse": 0.71, "rmse_ci_lower": 0.69, "rmse_ci_upper": 0.73},
                       "XGBoost": {"rmse": 0.61, "rmse_ci_lower": 0.59, "rmse_ci_upper": 0.63},
                       "RandomForest": {"rmse": 0.65}}})
    assert results.result_sentence(load_state(run)) == (
        "Best by RMSE: XGBoost (RMSE 0.61, 95% CI 0.59–0.63), on 3,467 students held out for "
        "testing; the analysis named LinearRegression (RMSE 0.71) as its best model — check the "
        "paper's claim.")


@pytest.mark.parametrize("found, expected", [
    # A tie the analysis broke in favour of the simpler model: no contradiction.
    ({"best_model": "LogisticRegression", "primary_metric": "AUC",
      "all_models": {"LogisticRegression": {"auc": 0.8}, "XGBoost": {"auc": 0.8}}},
     "Best model: LogisticRegression, AUC 0.80, on 3,467 students held out for testing."),
    # The claim's value in best_metric_value is not the model's metric: the metric wins.
    ({"best_model": "XGBoost", "best_metric_value": 0.9, "primary_metric": "auc_roc",
      "all_models": {"XGBoost": {"test_auc": 0.78}, "MLP": {"test_auc": 0.7}}},
     "Best model: XGBoost, AUC 0.78, on 3,467 students held out for testing."),
    # A metric whose direction is unknown is not compared: the claim stands.
    ({"best_model": "XGBoost", "best_metric_value": 0.4, "primary_metric": "custom_score",
      "all_models": {"XGBoost": {"custom_score": 0.4}, "MLP": {"custom_score": 0.9}}},
     "Best model: XGBoost, custom_score 0.40, on 3,467 students held out for testing."),
    # No per-model values: the claim is all there is.
    ({"best_model": "XGBoost", "best_metric_value": 0.78, "primary_metric": "AUC"},
     "Best model: XGBoost, AUC 0.78, on 3,467 students held out for testing."),
])
def test_the_key_result_keeps_the_claim_when_the_metrics_agree_or_cannot_say(
        run_home: Path, found: dict, expected: str) -> None:
    from edmars.runstate import load_state

    assert results.result_sentence(load_state(_ready(run_home, results=found))) == expected


# The round-3 Mac study's results.json (AUC values and XGBoost's interval
# as the run wrote them), with the scope fields the orchestrator now adds
# (src/best_model.py). best_model is the best single model by design; the
# stacking ensemble scored higher, which is not a contradiction.
R3_RESULTS = {
    "best_model": "XGBoost",
    "best_metric_value": 0.801488285622901,
    "primary_metric": "AUC",
    "all_models": {
        "LogisticRegression": {"auc": 0.7884831280985126, "auc_ci_lower": 0.7728, "auc_ci_upper": 0.8042},
        "RandomForest": {"auc": 0.7985067167759475, "auc_ci_lower": 0.7836, "auc_ci_upper": 0.8144},
        "XGBoost": {"auc": 0.801488285622901, "auc_ci_lower": 0.7868867153371972,
                    "auc_ci_upper": 0.8172237882287067},
        "ElasticNet": {"auc": 0.7812581960658883, "auc_ci_lower": 0.7656, "auc_ci_upper": 0.7975},
        "StackingEnsemble": {"auc": 0.802020230289461, "auc_ci_lower": 0.7868,
                             "auc_ci_upper": 0.8175},
    },
    "best_model_scope": "individual",
    "best_overall_model": "StackingEnsemble",
    "best_overall_metric_value": 0.802020230289461,
}
R3_DATA_REPORT = {
    "dataset": "hsls09_public", "original_n": 23503, "analytic_n": 17335,
    "n_train": 13773, "n_test": 3562, "n_predictors_encoded": 42, "validation_passed": True,
}
R3_KEY_RESULT = ("Key result: Best single model: XGBoost, AUC 0.80 (95% CI 0.79-0.82); the "
                 "stacking ensemble, which combines the models, scored 0.80. Both were measured "
                 "on 3,562 students held out for testing.")


def test_the_key_result_names_the_best_single_model_and_the_ensemble_beside_it(
        run_home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # The round-3 Mac study: the screen said "check the paper's claim"
    # about a best_model that is the best single model, as designed.
    run = _ready(run_home, results=R3_RESULTS, data_report=R3_DATA_REPORT)
    _, out = _show(run, capsys)
    flat = " ".join(out.split())
    assert R3_KEY_RESULT in flat
    assert "check the paper's claim" not in flat
    assert "the analysis named" not in flat
    html_text = (run / "summary.html").read_text(encoding="utf-8")
    assert ("Best single model: XGBoost, AUC 0.80 (95% CI 0.79–0.82); the stacking ensemble, "
            "which combines the models, scored 0.80. Both were measured on 3,562 students held "
            "out for testing.") in html_text
    assert "check the paper" not in html_text


def test_an_older_results_file_without_the_scope_fields_reads_the_same(run_home: Path) -> None:
    from edmars.runstate import load_state

    older = {k: v for k, v in R3_RESULTS.items()
             if k not in ("best_model_scope", "best_overall_model", "best_overall_metric_value")}
    state = load_state(_ready(run_home, results=older, data_report=R3_DATA_REPORT))
    assert f"Key result: {results.result_sentence(state)}".replace("–", "-") == R3_KEY_RESULT
    assert state.metrics["ensemble_model"] == "StackingEnsemble"
    assert "claimed_best_model" not in state.metrics


def test_an_ensemble_behind_the_best_single_model_is_not_mentioned(run_home: Path) -> None:
    from edmars.runstate import load_state

    found = dict(R3_RESULTS, best_overall_model="XGBoost", best_overall_metric_value=0.8015,
                 all_models={**R3_RESULTS["all_models"], "StackingEnsemble": {"auc": 0.79}})
    assert results.result_sentence(load_state(_ready(run_home, results=found,
                                                     data_report=R3_DATA_REPORT))) == (
        "Best model: XGBoost, AUC 0.80 (95% CI 0.79–0.82), on 3,562 students held out for testing.")


def test_an_ensemble_the_analysis_named_as_its_best_is_compared_with_every_model(
        run_home: Path) -> None:
    from edmars.runstate import load_state

    found = dict(R3_RESULTS, best_model="StackingEnsemble", best_metric_value=0.802020230289461,
                 best_model_scope="overall")
    state = load_state(_ready(run_home, results=found, data_report=R3_DATA_REPORT))
    assert results.result_sentence(state) == (
        "Best model: StackingEnsemble, AUC 0.80 (95% CI 0.79–0.82), on 3,562 students held out "
        "for testing.")
    assert "ensemble_model" not in state.metrics


def test_a_claim_contradicted_among_the_single_models_still_says_so(run_home: Path) -> None:
    # The claim is compared with the single models: RandomForest beat
    # XGBoost there, and the ensemble beat both.
    from edmars.runstate import load_state

    found = dict(CONTRADICTED_RESULTS, best_model_scope="individual",
                 best_overall_model="StackingEnsemble", best_overall_metric_value=0.83,
                 all_models={**CONTRADICTED_RESULTS["all_models"], "StackingEnsemble": {"auc": 0.83}})
    assert results.result_sentence(load_state(_ready(run_home, results=found))) == (
        "Best single model by AUC: RandomForest (AUC 0.81, 95% CI 0.79–0.83), on 3,467 students "
        "held out for testing; the stacking ensemble, which combines the models, scored 0.83; "
        "the analysis named XGBoost (AUC 0.78) as its best model — check the paper's claim.")


#: The six final checks that fired falsely on the round-3 Mac paper.
R3_FALSE_FINDINGS = ("INV_COMPARATOR_MISNAMED", "INV_IMPUTATION_METHOD_MISMATCH",
                     "INV_STATED_GAP_ARITHMETIC", "INV_PERCENTAGE_FROM_SPEC_NOT_RUN",
                     "INV_UNESCAPED_LATEX_SPECIAL", "INV_SCAFFOLDING_LEAKED")


def test_the_round_3_paper_screen_lists_its_real_findings_and_not_the_false_ones(
        run_home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # The round-3 Mac study as far as its evidence rebuilds it (the
    # sentences in tests/test_invariants.py), checked by the real final
    # checks rather than a hand-written invariants.json. Its result screen
    # listed six findings that were false; two were real and still are.
    import copy
    import json

    from src.invariants import findings_to_json, run_invariants
    from tests.cli._run_support import write_json
    from tests.test_invariants import (
        _ACM_FRONT_MATTER,
        _R3_DATA_REPORT,
        _R3_IMPUTATION_SENTENCE,
        _R3_RACE_GAP,
        _R3_RESULTS,
        _R3_SES_GAP,
        _R3_TABLE_THEN_COMPARISON,
    )

    def body(fragment: str) -> str:
        return fragment.replace(r"\begin{document}", "").replace(r"\end{document}", "")

    # The rebuilt sentences are far shorter than the paper; a neutral
    # paragraph brings the body over the empty-manuscript floor, as the
    # real paper was.
    filler = "We describe the students, the questions and the models in the sections below. " * 40
    paper = (
        _ACM_FRONT_MATTER + r"\Description{ROC curves.}" * 6 + "\n" + filler + "\n"
        + "\nXGBoost achieved the best discrimination (AUC $= 0.801$, 95\\% clustered CI "
          "[0.781, 0.820]) and outperformed logistic regression by a small but statistically "
          "detectable margin (AUC difference $= 0.013$, 95\\% CI [0.006, 0.021]).\n\n"
        + body(_R3_IMPUTATION_SENTENCE) + "\n\n" + body(_R3_TABLE_THEN_COMPARISON) + "\n\n"
        + _R3_RACE_GAP + "\n\n" + _R3_SES_GAP + "\n\n"
        + r"However, the college enrollment outcome itself has approximately 26\% missingness, "
          r"which may be non-random (MNAR); complete-case analysis may bias estimates."
        + "\n\\end{document}\n"
    )
    found = copy.deepcopy(_R3_RESULTS)
    found.update({key: R3_RESULTS[key] for key in (
        "primary_metric", "best_metric_value", "best_model_scope", "best_overall_model",
        "best_overall_metric_value")})
    found["all_models"] = {**found["all_models"], **R3_RESULTS["all_models"]}
    run = _ready(run_home, results=found, invariants=None, status=None,
                 data_report=dict(R3_DATA_REPORT, **_R3_DATA_REPORT),
                 extra={"paper.tex": paper, "roc_curves.png": "png", "research_spec.json": json.dumps(
                     {"potential_limitations": ["X4EVRATNDCLG has approximately 26% missingness"]})})
    checks = {"enabled": True, **findings_to_json(run_invariants(str(run)))}
    write_json(run / "invariants.json", checks)
    write_json(run / "run_status.json", v2_status(
        reason_code="ADVISORY_FINDINGS", counts=checks["counts"], invariant_codes=checks["codes"]))

    _, out = _show(run, capsys)
    flat = " ".join(out.split())
    for code in R3_FALSE_FINDINGS:
        assert code not in checks["codes"], code
        assert f"[{code}]" not in flat, code
    assert checks["codes"] == ["INV_CLASS_BALANCE_WRONG_SAMPLE", "INV_SUPERLATIVE_CONTRADICTED"]
    assert "[INV_CLASS_BALANCE_WRONG_SAMPLE]" in flat
    assert "[INV_SUPERLATIVE_CONTRADICTED]" in flat
    assert R3_KEY_RESULT in flat
    html_text = (run / "summary.html").read_text(encoding="utf-8")
    for code in R3_FALSE_FINDINGS:
        assert code not in html_text, code


def test_a_serious_finding_is_listed_before_the_scores_under_a_review_label(
        run_home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # The round-3 Mac study: "Ready, below the review benchmark", with a
    # serious finding that only the list below the scores mentioned.
    from tests.cli.test_endstates import MAC_R3_FINDINGS, MAC_R3_GATE

    run = _ready(run_home, review_gate_enabled=True, invariants=invariants_file(MAC_R3_FINDINGS),
                 gate_summary={"cycles_used": 2, "final_score": 5.1, "passed": False,
                               "threshold_used": 6.3, "advisory_mode": False, "venue": "EDM",
                               "ran": True},
                 status=v2_status(reason_code="GATE_FAILED", gate=MAC_R3_GATE,
                                  counts={"critical": 1, "major": 2, "minor": 0}))
    _, out = _show(run, capsys)
    lines = [line.strip() for line in out.splitlines()]
    assert lines[0] == "[!] Ready, below the review benchmark - 1 serious issue to check"
    assert lines.index("Please check") < lines.index("Scores")
    after = " ".join(" ".join(lines[lines.index("Please check") + 1:]).split())
    assert after.startswith("[x] The paper names the wrong models in its main comparison "
                            "[INV_COMPARATOR_MISNAMED]")
    html_text = (run / "summary.html").read_text(encoding="utf-8")
    assert "Ready, below the review benchmark — 1 serious issue to check" in html_text
    assert html_text.index("<h2>Please check</h2>") < html_text.index("<h2>Scores</h2>")


def test_without_a_serious_finding_the_scores_come_first(
        run_home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run = _ready(run_home, invariants=invariants_file([("INV_HANDTYPED_CROSSREF", "major")]),
                 status=v2_status(counts={"critical": 0, "major": 1, "minor": 0}))
    _, out = _show(run, capsys)
    lines = [line.strip() for line in out.splitlines()]
    assert lines[0] == "[ok] Ready"
    assert lines.index("Scores") < lines.index("Please check")
