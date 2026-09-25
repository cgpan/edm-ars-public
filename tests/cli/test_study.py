"""The guided new-study flow (edmars/study.py, CLI_SPEC sections 14 and 18).

Offline: no network, no keyring, no data download. EDMARS_HOME points at a
tmp dir; where the flow looks for raw data, the probe cache and Rscript is
monkeypatched per test so nothing on the host machine is read or written.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import pytest

from edmars import study
from edmars.model import Check, StudyPlan

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_FILES = {
    "hsls09_public": "hsls_17_student_pets_sr_v1_0.csv",
    "els_2002": "els_2002/els_02_12_byf3pststu_v1_0.csv",
    "did_els_hsls_panel": "did_els_hsls_panel/panel.csv",
    "assistments_0910": "assistments_0910/skill_builder_0910.csv",
}


# --------------------------------------------------------------------------
# Fixtures and helpers
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """EDMARS_HOME in tmp; raw data, cache and R lookups redirected."""
    monkeypatch.setenv("EDMARS_HOME", str(tmp_path / "home"))
    raw = tmp_path / "raw"
    raw.mkdir()
    monkeypatch.setattr(study, "_raw_data_dir", lambda settings: raw)
    monkeypatch.setattr(study, "_cache_dir", lambda: tmp_path / "cache")
    monkeypatch.setattr(study, "_find_rscript", lambda settings: None)
    return raw


@pytest.fixture
def raw_dir(_isolated: Path) -> Path:
    return _isolated


def install_data(raw: Path, dataset: str, content: str = "") -> Path:
    path = raw / DATA_FILES[dataset]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def load_with_pipeline_loader(spec: dict, tmp_path: Path, dataset: str) -> dict:
    """Round-trip through the pipeline's own src.main loader."""
    # Via the study module's accessor: importing src.main directly would run
    # its module-level load_dotenv() and could leak a local .env into tests.
    load_locked_research_spec = study._spec_loader()

    path = tmp_path / "research_spec.locked.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return load_locked_research_spec(
        str(path), dataset=dataset, registry_dir=str(REPO_ROOT / "data_registry")
    )


def stage0(spec: dict, dataset: str, tmp_path: Path) -> Any:
    from src.ideation.feasibility import screen

    return screen(
        spec,
        dataset=dataset,
        task_type=spec["task_type"],
        registry_dir=REPO_ROOT / "data_registry" / "datasets",
        raw_data_dir=tmp_path / "no-data-here",
        cache_dir=tmp_path / "cache",
        run_probes=False,
    )


DEFAULT = object()


class FakeUI:
    """Scripted stand-in for edmars.ui. Each prompt pops the next answer."""

    def __init__(self, answers: list[Any]) -> None:
        self.answers = list(answers)
        self.log: list[tuple[str, ...]] = []
        self.console = None

    def is_plain(self) -> bool:
        return True

    def _next(self, kind: str, message: str) -> Any:
        if not self.answers:
            raise AssertionError(f"unexpected {kind} prompt: {message!r}")
        return self.answers.pop(0)

    def select(
        self, message: str, choices: list[tuple[str, str]], default: str | None = None
    ) -> str:
        answer = self._next("select", message)
        if answer is DEFAULT:
            answer = default
        values = [value for value, _ in choices]
        assert answer in values, f"{answer!r} not offered for {message!r}: {values}"
        self.log.append(("select", message, str(answer)))
        return str(answer)

    def text(self, message: str, default: str | None = None, validate: Any = None) -> str:
        answer = self._next("text", message)
        if answer is DEFAULT:
            answer = default or ""
        self.log.append(("text", message, str(answer)))
        return str(answer)

    def confirm(self, message: str, default: bool = True) -> bool:
        answer = self._next("confirm", message)
        if answer is DEFAULT:
            answer = default
        self.log.append(("confirm", message, str(answer)))
        return bool(answer)

    def secret(self, message: str) -> str:  # pragma: no cover - never asked here
        raise AssertionError("the study flow never asks for a secret")

    def ok(self, msg: str) -> None:
        self.log.append(("ok", msg))

    def info(self, msg: str) -> None:
        self.log.append(("info", msg))

    def warn(self, msg: str) -> None:
        self.log.append(("warn", msg))

    def fail(self, msg: str) -> None:
        self.log.append(("fail", msg))

    def panel(self, title: str, body: str) -> None:
        self.log.append(("panel", title, body))

    def said(self, kind: str) -> str:
        return "\n".join(" ".join(entry[1:]) for entry in self.log if entry[0] == kind)


def run_flow(monkeypatch: pytest.MonkeyPatch, answers: list[Any],
             settings: dict | None = None) -> tuple[StudyPlan | None, FakeUI]:
    ui = FakeUI(answers)
    monkeypatch.setattr(study, "_ui", lambda: ui)
    plan = study.new_study_interactive(settings or {})
    assert not ui.answers, f"unused scripted answers: {ui.answers}"
    return plan, ui


def ok_preflight(monkeypatch: pytest.MonkeyPatch) -> list[StudyPlan]:
    seen: list[StudyPlan] = []

    def fake(plan: StudyPlan, settings: Any, **_: Any) -> list[Check]:
        seen.append(plan)
        return [Check("stub", "ok", "stubbed preflight")]

    monkeypatch.setattr(study, "preflight", fake)
    return seen


# --------------------------------------------------------------------------
# Example studies
# --------------------------------------------------------------------------

SHIPPED_EXAMPLES = sorted(
    p.stem[5:] for p in (REPO_ROOT / "runs" / "fixtures").glob("spec_*.json")
)


def test_every_shipped_fixture_is_an_example() -> None:
    assert SHIPPED_EXAMPLES, "runs/fixtures has no example specs"
    assert sorted(study.EXAMPLES) == SHIPPED_EXAMPLES
    for ex in study.EXAMPLES.values():
        assert ex.task_type in study.TASK_TYPES
        assert ex.dataset == ex.spec["dataset"]
        assert ex.research_question == ex.spec["research_question"]
        # No reviewer scores or run numbers in what the user is shown.
        assert not re.search(r"\d\.\d", ex.title), ex.title


@pytest.mark.parametrize("example_id", SHIPPED_EXAMPLES)
def test_example_plan_passes_the_pipeline_loader(example_id: str, tmp_path: Path) -> None:
    ex = study.EXAMPLES[example_id]
    plan = study.plan_from_flags({}, task_type=ex.task_type, example=example_id)
    assert plan.task_type == ex.task_type
    assert plan.dataset == ex.spec["dataset"]  # the dataset comes from the spec
    assert plan.spec is not None and plan.spec["dataset"] == plan.dataset
    assert plan.example_id == example_id
    assert plan.experimental is False
    assert plan.prompt == plan.research_question == ex.research_question
    loaded = load_with_pipeline_loader(plan.spec, tmp_path, plan.dataset)
    assert loaded["task_type"] == ex.task_type


@pytest.mark.parametrize("ref", ["x1mtheff_itr", "spec_x1mtheff_itr",
                                 "spec_x1mtheff_itr.json", "X1MTHEFF_ITR"])
def test_example_ids_resolve_loosely(ref: str) -> None:
    ex = study.find_example(ref)
    assert ex is not None and ex.id == "x1mtheff_itr"


@pytest.mark.parametrize("example_id", SHIPPED_EXAMPLES)
def test_examples_pass_stage0_screen_without_data(example_id: str, tmp_path: Path) -> None:
    ex = study.EXAMPLES[example_id]
    report = stage0(ex.spec, ex.dataset, tmp_path)
    assert report.verdict != "KILL", report.render()


@pytest.mark.parametrize("example_id", SHIPPED_EXAMPLES)
def test_example_preflight_without_data_fails_only_on_the_data_file(
    example_id: str,
) -> None:
    plan = study.plan_from_flags({}, example=example_id)
    checks = study.preflight(plan, {})
    fails = [c for c in checks if c.status == "fail"]
    names = {c.name for c in fails}
    allowed = {"The data file is on this computer",
               "R is installed (needed for the measurement models)"}
    assert names <= allowed, [(c.name, c.detail) for c in fails]
    data = next(c for c in checks if c.name == "The data file is on this computer")
    assert data.status == "fail"
    assert data.fix and "edmars data" in data.fix
    assert study.blocking(checks)
    loader = next(c for c in checks if c.name == "The pipeline accepts the study plan")
    assert loader.status == "ok"


def test_example_preflight_with_data_present_is_clear(raw_dir: Path) -> None:
    install_data(
        raw_dir, "hsls09_public",
        "X1MTHEFF,X4EVRATNDCLG,X1SEX\n0.5,Yes,Male\n-0.2,No,Female\n",
    )
    plan = study.plan_from_flags({}, example="x1mtheff_x4college")
    checks = study.preflight(plan, {})
    assert not study.blocking(checks), [(c.name, c.detail) for c in checks
                                        if c.status == "fail"]
    names = {c.name for c in checks}
    assert "Every variable is in the data file" in names  # header was read
    assert "Enough students with usable data" in names  # a probe ran


# --------------------------------------------------------------------------
# Build my own (EXPERIMENTAL)
# --------------------------------------------------------------------------


def test_menu_causal_soo_on_hsls_passes_loader_and_screen(tmp_path: Path) -> None:
    spec = study.build_menu_spec(
        {}, "causal_soo", "hsls09_public",
        {"treatment": "X1MTHEFF", "outcome": "X4EVRATNDCLG"},
    )
    assert spec["task_type"] == "causal_soo"
    assert spec["dataset"] == "hsls09_public"
    assert spec["treatment"] == {
        "variable": "X1MTHEFF",
        "operationalization": "median_split_binary",
        "rationale_for_PF": spec["treatment"]["rationale_for_PF"],
    }
    assert spec["outcome"]["variable"] == "X4EVRATNDCLG"
    assert spec["primary_method"] == "M2"
    # Recommended adjustment set: all base-year, never the treatment.
    assert spec["adjustment_set"] and "X1MTHEFF" not in spec["adjustment_set"]
    assert all(name.startswith("X1") for name in spec["adjustment_set"])
    assert spec["subgroup_analyses"] == ["X1SEX"]
    assert spec["note"] == study.MENU_NOTE
    assert spec["compiled_by"]["tournament_id"] == "edmars-cli"
    # No local directory layout leaks into the saved spec.
    assert spec["compiled_by"]["registry_source"] == (
        "data_registry/datasets/hsls09_public.yaml"
    )
    load_with_pipeline_loader(spec, tmp_path, "hsls09_public")
    report = stage0(spec, "hsls09_public", tmp_path)
    assert report.verdict == "CLEAN", report.render()


def test_menu_psychometrics_on_hsls_passes_loader_and_screen(tmp_path: Path) -> None:
    spec = study.build_menu_spec(
        {}, "psychometrics", "hsls09_public",
        {"item_banks": ["math_self_efficacy", "math_utility"],
         "grouping_vars": ["X1SEX"]},
    )
    assert spec["item_columns"] == [
        "S1MTESTS", "S1MTEXTBOOK", "S1MSKILLS", "S1MASSEXCL",
        "S1MUSELIFE", "S1MUSECLG", "S1MUSEJOB",
    ]
    factors = spec["factor_model"].splitlines()
    assert len(factors) == 2 and all("=~" in f for f in factors)
    assert spec["method_battery"] == ["P1", "P2", "P3", "P5", "P6"]
    assert spec["grouping_vars"] == ["X1SEX"]
    assert spec["response_labels"]["Strongly agree"] == 4
    assert "X1SEX codes: 1=Male, 2=Female" in spec["grouping_notes"]
    load_with_pipeline_loader(spec, tmp_path, "hsls09_public")
    report = stage0(spec, "hsls09_public", tmp_path)
    assert report.verdict == "CLEAN", report.render()


def test_menu_psychometrics_reverse_items_carried(tmp_path: Path) -> None:
    spec = study.build_menu_spec(
        {}, "psychometrics", "hsls09_public", {"item_banks": ["math_interest"]}
    )
    assert spec["reverse_items"] == ["S1MWASTE", "S1MBORING"]
    assert spec["method_battery"] == ["P1", "P2", "P3"]
    load_with_pipeline_loader(spec, tmp_path, "hsls09_public")


@pytest.mark.parametrize(
    ("task_type", "dataset", "choices"),
    [
        ("causal_itr", "hsls09_public",
         {"treatment": "X1MTHEFF", "outcome": "X4EVRATNDCLG"}),
        ("causal_soo", "els_2002", {"treatment": "BYTXMSTD", "outcome": "F2EVRATT"}),
        ("psychometrics", "els_2002", {"grouping_vars": ["BYSEX"], "irt": True}),
        ("causal_did", "did_els_hsls_panel", {"group": "low_ses"}),
    ],
)
def test_other_menu_specs_pass_loader_and_screen(
    task_type: str, dataset: str, choices: dict, tmp_path: Path
) -> None:
    spec = study.build_menu_spec({}, task_type, dataset, choices)
    assert spec["task_type"] == task_type and spec["dataset"] == dataset
    load_with_pipeline_loader(spec, tmp_path, dataset)
    assert stage0(spec, dataset, tmp_path).verdict != "KILL"
    if task_type == "causal_itr":
        assert spec["primary_method"] == "M6"
        assert set(spec["rule_covariates"]) <= set(spec["adjustment_set"])
    if task_type == "causal_did":
        assert spec["post_variable"] == "cohort"
        assert spec["primary_method"] == "M8"


def test_menu_offers_no_post_treatment_covariates() -> None:
    offered = {o.value for o in study.covariate_options(
        "hsls09_public", "X1MTHEFF", "X4EVRATNDCLG")}
    assert "X1SES" in offered
    assert not any(name.startswith(("X2", "X3", "X4", "S3")) for name in offered)
    outcomes = {o.value for o in study.outcome_options("hsls09_public", "X1MTHEFF")}
    assert "X4EVRATNDCLG" in outcomes and "dropout_derived" not in outcomes
    treatments = {o.value for o in study.treatment_options("hsls09_public")}
    assert "X1MTHEFF" in treatments
    assert "X1SES" not in treatments  # protected attributes are never a "cause"


@pytest.mark.parametrize(
    ("choices", "fragment"),
    [
        ({"treatment": "NOT_A_VAR", "outcome": "X4EVRATNDCLG"}, "not an available"),
        ({"treatment": "X1MTHEFF", "outcome": "X1SES"}, "not an available"),
        ({"treatment": "X1MTHEFF", "outcome": "X4EVRATNDCLG",
          "covariates": ["X2SES"]}, "not an available"),
    ],
)
def test_menu_rejects_choices_the_menus_do_not_offer(choices: dict, fragment: str) -> None:
    with pytest.raises(study.StudyError, match=fragment):
        study.build_menu_spec({}, "causal_soo", "hsls09_public", choices)


def test_menu_refuses_unsupported_cells_and_two_item_banks() -> None:
    with pytest.raises(study.StudyError, match="single group"):
        study.build_menu_spec({}, "causal_did", "hsls09_public", {"group": "X1SEX"})
    with pytest.raises(study.StudyError, match="too few"):
        study.build_menu_spec({}, "psychometrics", "hsls09_public",
                              {"item_banks": ["math_identity"]})
    with pytest.raises(study.StudyError, match="grouping variable"):
        study.build_menu_spec({}, "psychometrics", "hsls09_public",
                              {"item_banks": ["math_utility"], "methods": ["P1", "P5"]})
    assert study.menu_unavailable_reason("psychometrics", "assistments_0910")
    assert study.menu_unavailable_reason("causal_soo", "hsls09_public") is None


def test_check_spec_reports_the_loader_error_without_temp_paths() -> None:
    problems, _ = study.check_spec(
        {"task_type": "psychometrics", "dataset": "hsls09_public",
         "research_question": "q", "scale_name": "s",
         "item_columns": ["S1MTESTS", "S1MSKILLS"], "method_battery": ["P3"]},
        "hsls09_public",
    )
    assert problems and "CFA (P3) needs >= 3 items" in problems[0]
    assert "research_spec.locked.json" not in problems[0]


# --------------------------------------------------------------------------
# preflight for free-text prediction and plan mismatches
# --------------------------------------------------------------------------


def test_prediction_preflight_checks_compat_and_data(raw_dir: Path) -> None:
    plan = study.plan_from_flags(
        {}, prompt="Which ninth-grade factors best predict college attendance?"
    )
    checks = study.preflight(plan, {})
    by_name = {c.name: c for c in checks}
    assert by_name["Study type works with this dataset"].status == "ok"
    data = by_name["The data file is on this computer"]
    assert data.status == "fail" and data.fix == "edmars data install hsls09_public"
    assert "hsls_17_student_pets_sr_v1_0.csv" in data.detail
    install_data(raw_dir, "hsls09_public")
    assert not study.blocking(study.preflight(plan, {}))


def test_preflight_blocks_incompatible_and_specless_plans(raw_dir: Path) -> None:
    install_data(raw_dir, "hsls09_public")
    did_on_hsls = StudyPlan("causal_did", "hsls09_public", "Did the gap change?")
    checks = study.preflight(did_on_hsls, {})
    failed = {c.name for c in checks if c.status == "fail"}
    assert "Study type works with this dataset" in failed
    assert "The study plan is complete" in failed  # causal type with no spec


def test_preflight_blocks_a_spec_whose_dataset_differs() -> None:
    ex = study.EXAMPLES["did_ses_gap"]
    plan = StudyPlan("causal_did", "hsls09_public", ex.research_question,
                     spec=dict(ex.spec))
    failed = {c.name for c in study.preflight(plan, {}) if c.status == "fail"}
    assert "The dataset matches the study plan" in failed


def test_preflight_requires_r_for_measurement_models(
    raw_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_data(raw_dir, "hsls09_public")
    plan = study.plan_from_flags({}, example="psy_hsls_matheff_invariance")
    r_check = next(c for c in study.preflight(plan, {}, run_probes=False)
                   if c.name.startswith("R is installed"))
    assert r_check.status == "fail" and r_check.fix == "edmars setup r"
    monkeypatch.setattr(study, "_find_rscript", lambda settings: "Rscript")
    r_check = next(c for c in study.preflight(plan, {}, run_probes=False)
                   if c.name.startswith("R is installed"))
    assert r_check.status == "ok"


def test_preflight_adds_a_language_note() -> None:
    plan = StudyPlan(
        "prediction", "hsls09_public",
        "¿Qué factores de noveno grado predicen la asistencia a la universidad?",
    )
    notes = [c for c in study.preflight(plan, {}) if c.name == "Language"]
    assert notes and notes[0].status == "info" and "English" in notes[0].detail


def _late_treatment_plan() -> StudyPlan:
    """A causal plan whose cause (11th grade) comes after its outcome (9th)."""
    spec = study.build_menu_spec(
        {}, "causal_soo", "hsls09_public",
        {"treatment": "X1MTHEFF", "outcome": "X4EVRATNDCLG"},
    )
    spec["treatment"]["variable"] = "X2MTHEFF"
    spec["outcome"]["variable"] = "X1TXMTSCOR"
    return StudyPlan("causal_soo", "hsls09_public", spec["research_question"],
                     spec=spec, experimental=True)


_DEVELOPER_WORDS = ("Why:", "temporal_order", "registry wave", "role=", "['",
                    "Tier-", "predicate", "dispatch", ".yaml")


def test_preflight_explains_a_blocking_check_in_plain_words() -> None:
    checks = study.preflight(_late_treatment_plan(), {}, run_probes=False)
    order = next(c for c in checks if c.name == "Earlier measures come before the outcome")
    assert order.status == "fail"
    assert "X2MTHEFF is measured in 2012 (11th grade), after the outcome" in order.detail
    assert "The outcome X1TXMTSCOR is measured in 2009 (9th grade)" in order.detail
    assert order.fix == "Choose an outcome measured after the other variables."
    for check in checks:
        for word in _DEVELOPER_WORDS:
            assert word not in check.detail, (check.name, check.detail)


def test_preflight_technical_detail_only_in_debug_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EDMARS_DEBUG", "1")
    checks = study.preflight(_late_treatment_plan(), {}, run_probes=False)
    order = next(c for c in checks if c.name == "Earlier measures come before the outcome")
    assert "[Technical detail: F-TEMPORAL-ORDER:" in order.detail


def test_passing_screen_checks_carry_no_developer_text(raw_dir: Path) -> None:
    install_data(
        raw_dir, "hsls09_public",
        "X1MTHEFF,X4EVRATNDCLG,X1SEX\n0.5,Yes,Male\n-0.2,No,Female\n",
    )
    plan = study.plan_from_flags({}, example="x1mtheff_x4college")
    for check in study.preflight(plan, {}):
        for word in _DEVELOPER_WORDS + ("Analytic n", "Minority class"):
            assert word not in check.detail, (check.name, check.detail)


@pytest.mark.parametrize(
    ("code", "message", "fragment"),
    [
        ("F-VAR-ABSENT", "Variable(s) do not exist in this dataset: FOO, BAR.",
         "These variables are not in HSLS:09: FOO, BAR."),
        ("F-TIER3-EXCLUDED", "Tier-3 excluded name(s) used as study variables: "
         "W1STUDENT. These are weights, sampling/administrative IDs, or "
         "processing flags.", "not measures of students: W1STUDENT."),
        ("F-DEAD-VARIABLE", "Variable(s) carry no usable data: X1ASIAN "
         "(100.0% missing).", "suppressed or empty): X1ASIAN (100.0% missing)."),
        ("F-ESTIMATOR-UNCERTIFIED", "Estimator(s) RD are certified on synthetic "
         "DGPs but shelved: no executable task type implements them.",
         "cannot run these methods for this kind of study yet: RD."),
        ("F-SPEC-INCOMPLETE", "Spec cannot be dispatched as causal_soo: missing "
         "treatment, outcome.", "missing parts the pipeline needs: treatment, outcome."),
    ],
)
def test_blocking_checks_name_the_offending_parts(
    code: str, message: str, fragment: str
) -> None:
    from src.ideation.feasibility import KILL, CheckResult, FeasibilityReport

    report = FeasibilityReport(
        "t", KILL, [CheckResult(code, KILL, message, "evidence ['x']")],
        dataset="hsls09_public", task_type="causal_soo",
    )
    (check,) = study._map_report(report)
    assert check.status == "fail"
    assert fragment in check.detail
    assert "evidence" not in check.detail


def test_run_shows_only_names_for_passing_checks(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from edmars import cli, ui

    ui.set_plain(True)
    cli._show_study_checks([
        Check("Enough students with usable data", "ok", "Passed."),
        Check("Earlier measures come before the outcome", "fail", "X is too late.",
              "Choose an outcome measured after the other variables."),
    ])
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "Enough students with usable data" in out and "Passed." not in out
    assert "X is too late." in out and "Choose an outcome" in out


# --------------------------------------------------------------------------
# Out of scope, language, "Not sure"
# --------------------------------------------------------------------------

OUT_OF_SCOPE = [
    ("I want to interview teachers about why students drop out", "qualitative"),
    ("A qualitative study of first-generation college students", "qualitative"),
    ("Run focus groups with parents about homework", "qualitative"),
    ("An ethnographic account of a rural high school", "qualitative"),
    ("A meta-analysis of tutoring effects on math achievement", "meta-analysis"),
    ("Systematic review of dropout early-warning systems", "meta-analysis"),
    ("Write a literature review on self-efficacy", "literature review"),
    ("Design a survey to measure school belonging", "new survey"),
    ("Develop a new questionnaire about math anxiety", "new survey"),
    ("Predict dropout using my own data from our district", "your own data"),
    ("How does tutoring work with our district data", "your own data"),
    ("Can I upload a CSV of my students' grades?", "your own data"),
    ("Which of my students will fail algebra next year?", "your own data"),
    ("Use our own records to predict attendance", "your own data"),
    ("Design a randomized controlled trial of a reading program", "experiment"),
    ("How many schools do I need for an RCT?", "experiment"),
]

IN_SCOPE = [
    "Does 9th-grade math self-efficacy affect college attendance?",
    "Which ninth-grade factors best predict dropping out of high school?",
    "Do the math self-efficacy items measure the same construct for boys and girls?",
    "How well can survey responses about school belonging predict GPA?",
    "What is the effect of attending a private school on college enrollment in HSLS?",
    "Predict college attendance and summarise the literature on its predictors",
    "Can we reliably predict which students will take AP math?",
    "Are our schools failing low-income students, and which factors predict it?",
    "Without random assignment, does tutoring affect later grades?",
    "How can we help our students succeed in algebra, based on HSLS:09?",
]


@pytest.mark.parametrize(("text", "fragment"), OUT_OF_SCOPE)
def test_out_of_scope_explains_and_offers_a_framing(text: str, fragment: str) -> None:
    message = study.out_of_scope(text)
    assert message is not None, text
    assert fragment in message.lower()
    assert "Nearest thing EDM-ARS can do:" in message


@pytest.mark.parametrize("text", IN_SCOPE)
def test_in_scope_questions_pass(text: str) -> None:
    assert study.out_of_scope(text) is None


@pytest.mark.parametrize(
    "text",
    [
        "¿Cuál es el efecto de la autoeficacia en matemáticas sobre la asistencia "
        "a la universidad?",
        "Quels facteurs prédisent le décrochage scolaire des élèves?",
        "数学自我效能感如何影响大学入学？",
        "Welche Faktoren sagen den Schulabbruch der Schüler voraus und wie stark?",
    ],
)
def test_non_english_is_detected(text: str) -> None:
    assert study.looks_non_english(text)
    assert study.language_note(text) == study.LANGUAGE_NOTE


@pytest.mark.parametrize("text", IN_SCOPE + ["GPA", "math self-efficacy college"])
def test_english_is_not_flagged(text: str) -> None:
    assert not study.looks_non_english(text)


@pytest.mark.parametrize(
    ("text", "task_type"),
    [
        ("What is the effect of math self-efficacy on college attendance?",
         "causal_soo"),
        ("For whom does a growth mindset raise math grades the most?", "causal_itr"),
        ("Which students are likely to drop out of high school?", "prediction"),
        ("Do the self-efficacy items show measurement invariance across sex?",
         "psychometrics"),
        ("Did the gap between low- and high-SES students change between the 2002 "
         "and 2009 cohorts?", "causal_did"),
        # "attendance" must not read as the estimand "att" (word boundaries).
        ("Which attitudes predict college attendance?", "prediction"),
    ],
)
def test_suggest_study_type(text: str, task_type: str) -> None:
    assert study.suggest_study_type(text).task_type == task_type


@pytest.mark.parametrize(
    ("text", "task_type"),
    [
        # Everyday cause-and-effect wording the pipeline keywords miss.
        ("Does taking algebra in 8th grade cause higher college enrollment?",
         "causal_soo"),
        ("Does math self-efficacy affect college attendance?", "causal_soo"),
        ("Does taking calculus lead to higher GPA?", "causal_soo"),
        ("Does taking calculus influence college enrollment?", "causal_soo"),
        # The family menu's own words: "does one thing change another?"
        ("Does tutoring change math scores?", "causal_soo"),
        ("Does tutoring improve math scores?", "causal_soo"),
        # The causal-kind menu's own words: "who would benefit most from X?"
        ("Who would benefit most from tutoring?", "causal_itr"),
        ("Which students benefit most from taking advanced math?", "causal_itr"),
        # Forecasting wording stays a prediction even with a change verb.
        ("Can 9th-grade scores predict whether GPA will increase?", "prediction"),
        ("Which students are likely to reduce their course load?", "prediction"),
        # "because" is not "cause".
        ("Which students drop out because of low grades, and can we predict it?",
         "prediction"),
    ],
)
def test_suggest_study_type_plain_causal_wording(text: str, task_type: str) -> None:
    suggestion = study.suggest_study_type(text)
    assert suggestion.task_type == task_type


# --------------------------------------------------------------------------
# Menus: datasets and causal kinds
# --------------------------------------------------------------------------


def test_dataset_options_grey_out_unsupported_and_missing(raw_dir: Path) -> None:
    install_data(raw_dir, "hsls09_public")
    options = {o.value: o for o in study.dataset_options({}, "causal_did")}
    assert "single group" in (options["hsls09_public"].disabled or "")
    assert "not on this computer" in (options["did_els_hsls_panel"].disabled or "")
    options = {o.value: o for o in study.dataset_options({}, "prediction")}
    assert options["hsls09_public"].disabled is None
    assert "edmars data install els_2002" in (options["els_2002"].disabled or "")
    assert "import" in (options["assistments_0910"].disabled or "")


def test_cohort_comparison_only_with_the_panel(raw_dir: Path) -> None:
    kinds = {o.value: o for o in study.causal_kind_options({})}
    assert kinds["causal_did"].disabled and kinds["causal_soo"].disabled is None
    install_data(raw_dir, "did_els_hsls_panel")
    kinds = {o.value: o for o in study.causal_kind_options({})}
    assert kinds["causal_did"].disabled is None


def test_parse_selection() -> None:
    assert study.parse_selection("1,3 5-7", 8) == [1, 3, 5, 6, 7]
    assert study.parse_selection("2,2,1", 3) == [2, 1]
    for bad in ("0", "9", "3-1", "a", "1;2", ""):
        assert study.parse_selection(bad, 8) is None


# --------------------------------------------------------------------------
# plan_from_flags
# --------------------------------------------------------------------------


def test_flags_refuse_causal_or_measurement_without_a_spec() -> None:
    for task_type in ("causal_soo", "causal_itr", "causal_did", "psychometrics"):
        with pytest.raises(study.StudyError, match="--example"):
            study.plan_from_flags({}, task_type=task_type, prompt="Does X change Y?")


def test_flags_never_override_the_spec_dataset_or_type() -> None:
    with pytest.raises(study.StudyError, match="did_els_hsls_panel"):
        study.plan_from_flags({}, example="did_ses_gap", dataset="hsls09_public")
    with pytest.raises(study.StudyError, match="causal_soo"):
        study.plan_from_flags({}, example="x1mtheff_x4college", task_type="causal_itr")
    plan = study.plan_from_flags({}, example="did_ses_gap")
    assert plan.dataset == "did_els_hsls_panel"


def test_flags_prediction_rules() -> None:
    with pytest.raises(study.StudyError, match="--prompt"):
        study.plan_from_flags({}, task_type="prediction")
    with pytest.raises(study.StudyError, match="qualitative"):
        study.plan_from_flags({}, prompt="Interview teachers about dropout")
    with pytest.raises(study.StudyError, match="No example"):
        study.plan_from_flags({}, example="nope")
    with pytest.raises(study.StudyError, match="either"):
        study.plan_from_flags({}, example="did_ses_gap", spec_path="x.json")
    with pytest.raises(study.StudyError, match="only for prediction"):
        study.plan_from_flags({}, example="did_ses_gap", prompt="Anything?")
    with pytest.raises(study.StudyError, match="Unknown venue"):
        study.plan_from_flags({}, prompt="Which factors predict GPA?", venue="NeurIPS")
    plan = study.plan_from_flags(
        {}, prompt="  Which factors predict GPA in 12th grade?  ",
        dataset="els_2002", venue="jedm",
    )
    assert plan.task_type == "prediction" and plan.dataset == "els_2002"
    assert plan.research_question == plan.prompt == "Which factors predict GPA in 12th grade?"
    assert plan.spec is None and plan.venue == "JEDM" and plan.paper_format == "journal"


def test_flags_spec_file(tmp_path: Path) -> None:
    fixture = REPO_ROOT / "runs" / "fixtures" / "spec_x1mtheff_x4college.json"
    plan = study.plan_from_flags({}, spec_path=fixture)
    assert plan.example_id == "x1mtheff_x4college" and plan.experimental is False

    custom = json.loads(fixture.read_text(encoding="utf-8"))
    custom["research_question"] = "Does math self-efficacy change college going?"
    custom.pop("dataset")
    path = tmp_path / "mine.json"
    path.write_text(json.dumps(custom), encoding="utf-8")
    plan = study.plan_from_flags({}, spec_path=path)
    assert plan.experimental is True and plan.example_id is None
    assert plan.dataset == "hsls09_public" and plan.spec["dataset"] == "hsls09_public"

    custom["primary_method"] = "M99"
    path.write_text(json.dumps(custom), encoding="utf-8")
    with pytest.raises(study.StudyError, match="not a valid study plan"):
        study.plan_from_flags({}, spec_path=path)
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(study.StudyError, match="not valid JSON"):
        study.plan_from_flags({}, spec_path=path)


def test_flags_take_defaults_from_settings() -> None:
    settings = {"defaults": {"venue": "JLA", "paper_format": "journal"},
                "lsar": {"enabled": True, "auto_review": True}}
    plan = study.plan_from_flags(settings, example="x1mtheff_itr")
    assert (plan.venue, plan.paper_format, plan.review) == ("JLA", "journal", True)
    settings["provider"] = "local"  # LSAR is off for local models
    assert study.plan_from_flags(settings, example="x1mtheff_itr").review is False


# --------------------------------------------------------------------------
# Confirmation card (R6)
# --------------------------------------------------------------------------


def test_card_shows_experimental_badge_only_when_experimental() -> None:
    spec = study.build_menu_spec(
        {}, "causal_soo", "hsls09_public",
        {"treatment": "X1MTHEFF", "outcome": "X4EVRATNDCLG"},
    )
    menu_plan = StudyPlan("causal_soo", "hsls09_public", spec["research_question"],
                          spec=spec, experimental=True)
    card = study.confirmation_card(menu_plan, {"provider": "deepseek"})
    assert card.startswith(study.EXPERIMENTAL_BADGE)
    assert card.count("EXPERIMENTAL") >= 2
    assert "built with the menus" in card

    example_plan = study.plan_from_flags({}, example="x1mtheff_x4college")
    card = study.confirmation_card(example_plan, {"provider": "deepseek"})
    assert "EXPERIMENTAL" not in card
    assert "tested example 'x1mtheff_x4college'" in card


def test_card_content(tmp_path: Path) -> None:
    plan = study.plan_from_flags({}, prompt="Which ninth-grade factors predict GPA?")
    card = study.confirmation_card(plan, {"provider": "deepseek"}, balance="US$4.20")
    assert "Which ninth-grade factors predict GPA?" in card
    assert study.TIME_WITHOUT_REVIEW in card
    assert "US$0.05-0.20" in card and "Balance:" in card
    assert "What is sent where" in card and "To DeepSeek:" in card
    assert "never uploaded" in card
    assert "automated review" not in card.split("What is sent where")[1]
    card.encode("ascii")  # --plain safe when the question itself is ASCII

    reviewed = StudyPlan(**{**plan.__dict__, "review": True})
    card = study.confirmation_card(reviewed, {"provider": "openai",
                                              "lsar": {"enabled": True}})
    assert study.TIME_WITH_REVIEW.split(",")[0] in card
    assert "not estimated" in card
    assert "To OpenAI:" in card and "To DeepSeek, for the automated review" in card
    assert "benchmark" in card


def test_card_for_a_local_model() -> None:
    plan = study.plan_from_flags({}, prompt="Which ninth-grade factors predict GPA?")
    card = study.confirmation_card(
        plan, {"provider": "local", "provider_base_url": "http://localhost:11434/v1"}
    )
    assert "your own model server" in card and "localhost:11434" in card
    assert study.COST_LOCAL.split(";")[0] in card
    assert "US$0.05" not in card


def test_venue_benchmark_reads_the_installed_calibration(tmp_path: Path) -> None:
    assert study.venue_benchmarked("EDM", {}) is True
    assert study.venue_benchmarked("JEDM", {}) is False  # no LSAR installed
    home = tmp_path / "lsar"
    (home / "calibration").mkdir(parents=True)
    (home / "calibration" / "anchors_edm.yaml").write_text(
        "venues:\n  JEDM: {p25: 1.0}\n  JLA: {p25: null}\n", encoding="utf-8")
    settings = {"lsar": {"home": str(home)}}
    assert study.venue_benchmarked("JEDM", settings) is True
    assert study.venue_benchmarked("JLA", settings) is False


# --------------------------------------------------------------------------
# Interactive flow (scripted UI)
# --------------------------------------------------------------------------


def test_interactive_prediction_with_edit(
    monkeypatch: pytest.MonkeyPatch, raw_dir: Path
) -> None:
    install_data(raw_dir, "hsls09_public")
    plan, ui = run_flow(monkeypatch, [
        "prediction",
        "hsls09_public",
        "Which ninth-grade factors best predict college attendance by 2016?",
        DEFAULT, DEFAULT,           # venue, paper format
        "edit", "question",
        "Which ninth-grade attitudes best predict college attendance by 2016?",
        "start",
    ])
    assert plan is not None
    assert plan.task_type == "prediction" and plan.spec is None
    assert plan.research_question.startswith("Which ninth-grade attitudes")
    assert plan.prompt == plan.research_question
    assert (plan.venue, plan.paper_format, plan.review) == ("EDM", "conference", False)
    assert "automated peer review is not set up" in ui.said("info")
    assert any(entry[:2] == ("panel", "Check your study") for entry in ui.log)
    # Greyed datasets are explained, not hidden.
    assert "Not available:" in ui.said("info")


def test_interactive_out_of_scope_then_cancel(
    monkeypatch: pytest.MonkeyPatch, raw_dir: Path
) -> None:
    install_data(raw_dir, "hsls09_public")
    plan, ui = run_flow(monkeypatch, [
        "prediction", "hsls09_public",
        "I want to interview teachers about why students leave school",
        "",                          # empty question -> back to the start
        "cancel",
    ])
    assert plan is None
    assert any(e[0] == "panel" and "outside" in e[1] for e in ui.log)


def test_interactive_not_sure_to_example(
    monkeypatch: pytest.MonkeyPatch, raw_dir: Path
) -> None:
    install_data(raw_dir, "hsls09_public")
    seen = ok_preflight(monkeypatch)
    plan, ui = run_flow(monkeypatch, [
        "unsure",
        "What is the effect of ninth-grade math self-efficacy on college attendance?",
        True,                        # accept the suggested type
        "hsls09_public",
        "example:x1mtheff_x4college",
        DEFAULT, DEFAULT,
        "start",
    ])
    assert plan is not None and plan.example_id == "x1mtheff_x4college"
    assert plan.task_type == "causal_soo" and plan.experimental is False
    assert seen and seen[0].example_id == "x1mtheff_x4college"
    assert "Cause and effect: the average effect" in ui.said("info")


def test_interactive_not_sure_non_english_chooses_manually(
    monkeypatch: pytest.MonkeyPatch, raw_dir: Path
) -> None:
    install_data(raw_dir, "hsls09_public")
    plan, ui = run_flow(monkeypatch, [
        "unsure",
        "¿Qué factores predicen el abandono escolar de los estudiantes?",
        "cancel",                    # the manual family menu
    ])
    assert plan is None
    assert study.LANGUAGE_NOTE_UNSURE in ui.said("info")


def test_interactive_build_my_own_psychometrics(
    monkeypatch: pytest.MonkeyPatch, raw_dir: Path
) -> None:
    install_data(raw_dir, "hsls09_public")
    ok_preflight(monkeypatch)
    plan, ui = run_flow(monkeypatch, [
        "measurement",
        "hsls09_public",
        "build",
        DEFAULT,                     # item banks: the recommended one
        "X1SEX",                     # grouping
        False,                       # no IRT model
        DEFAULT,                     # keep the generated question
        DEFAULT, DEFAULT,
        "start",
    ])
    assert plan is not None and plan.experimental is True
    assert plan.spec is not None and plan.spec["task_type"] == "psychometrics"
    assert plan.spec["item_columns"] == ["S1MTESTS", "S1MTEXTBOOK", "S1MSKILLS",
                                         "S1MASSEXCL"]
    assert plan.spec["method_battery"] == ["P1", "P2", "P3", "P5", "P6"]
    assert plan.research_question == plan.spec["research_question"]
    assert "Not available: math identity (2 items" in ui.said("info")  # greyed, with why


def test_interactive_build_my_own_causal_soo(
    monkeypatch: pytest.MonkeyPatch, raw_dir: Path
) -> None:
    install_data(raw_dir, "hsls09_public")
    ok_preflight(monkeypatch)
    plan, _ = run_flow(monkeypatch, [
        "causal", "causal_soo", "hsls09_public", "build",
        "X1MTHEFF", "X4EVRATNDCLG",
        "",                          # recommended covariates
        "Does math self-efficacy change whether students attend college?",
        DEFAULT, DEFAULT,
        "start",
    ])
    assert plan is not None and plan.experimental
    assert plan.research_question == (
        "Does math self-efficacy change whether students attend college?"
    )
    assert plan.spec is not None
    assert plan.spec["research_question"] == plan.research_question
    assert plan.spec["adjustment_set"] == study.default_covariates(
        "hsls09_public", "X1MTHEFF", "X4EVRATNDCLG")


def test_interactive_blocking_preflight_offers_start_over(
    monkeypatch: pytest.MonkeyPatch, raw_dir: Path
) -> None:
    install_data(raw_dir, "hsls09_public")
    monkeypatch.setattr(
        study, "preflight",
        lambda plan, settings, **_: [Check("Data", "fail", "missing", "edmars data")],
    )
    plan, ui = run_flow(monkeypatch, [
        "prediction", "hsls09_public",
        "Which ninth-grade factors best predict college attendance?",
        "cancel",
    ])
    assert plan is None
    assert "What to do: edmars data" in ui.said("info")


def test_interactive_nothing_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    plan, ui = run_flow(monkeypatch, ["prediction", "cancel"])
    assert plan is None
    assert "No dataset on this computer" in ui.said("warn")


# --------------------------------------------------------------------------
# Hygiene
# --------------------------------------------------------------------------


def test_study_module_spawns_no_processes() -> None:
    source = Path(study.__file__).read_text(encoding="utf-8")
    assert "subprocess" not in source and "os.system" not in source
    assert "Popen" not in source


def test_importing_the_loader_does_not_leak_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = tmp_path / "edmars_leaky_probe.py"
    module.write_text(
        "import os\nos.environ['EDMARS_LEAK_PROBE'] = 'from-a-dotenv'\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delenv("EDMARS_LEAK_PROBE", raising=False)
    try:
        study._import_without_env_side_effects("edmars_leaky_probe")
        assert "EDMARS_LEAK_PROBE" not in os.environ
    finally:
        sys.modules.pop("edmars_leaky_probe", None)
