"""The outcome among the predictors is caught right after data preparation.

Observed on the owner's Mac (round 2, 2026-09-27): in a prediction study
the DataEngineer's generated code left the outcome X4EVRATNDCLG as the
14th column of train_X.csv and test_X.csv. Nothing looked until pcc_01,
after the Analyst had spent two paid calls fitting every model with the
answer as an input (AUC 1.0), and pcc_01 stopped the study. Round 1 had
answered the same question (AUC 0.81), so the question was feasible.

The contract these tests pin:

* right after ENGINEERING, an outcome column (by name, as an encoded
  level, or under the name train_y.csv gives it) sends the DataEngineer
  one targeted retry that names the columns, before any Analyst call;
* if the retry still leaves them, exactly those columns are dropped from
  train_X.csv and test_X.csv -- every other cell keeps its text -- and the
  run says so in pipeline.log, a warning event and data_report;
* a DataEngineer revision gets the same check;
* an outcome column that still reaches the review stops the study
  (pcc_01, not revisable), and its message says whether the check ran.

Every agent is a stub; nothing calls a provider or needs TeX.
"""
from __future__ import annotations

import copy
import csv
import json
import types
from pathlib import Path
from typing import Any

import pytest

from src import events
from src.context import PipelineState
from src.outcome_guard import (
    RECORD_KEY,
    REPAIR_CODE,
    describe_guard,
    drop_columns,
    find_outcome_columns,
    guard_outcome_in_predictors,
)
from src.pre_critic_checks import PreCriticResult, _check_outcome_not_in_train_x
from tests.test_orchestrator_terminal import (
    _config,
    _events,
    _fake_compile_ok,
    _orch,
    _status,
    _wire,
)
from tests.test_pre_critic_revise import _BINARY_REPORT, _SPEC

OUTCOME = "X4EVRATNDCLG"

#: The observed study's 13 predictors.
PREDICTORS = [
    "X1TXMTSCOR", "X1SES", "X1MTHID", "X1MTHEFF", "X1SCIID", "X1SCIEFF",
    "X1SCHOOLBEL", "X1SCHOOLENG", "X1SEX_Female", "X1RACE_Black",
    "X1RACE_Hispanic", "X1STUEDEXPCT", "X1PAREDU",
]


@pytest.fixture(autouse=True)
def _compile_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.orchestrator.compile_latex", _fake_compile_ok)


def _write(path: Path, header: list[str], n: int = 200) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for i in range(n):
            # Varying values, so the constant-column check stays quiet;
            # "0.10"-style text shows that a repair keeps every cell as is.
            w.writerow([f"{(i * (j + 3)) % 97}.10" for j in range(len(header))])


def _matrices(out: Path, x_cols: list[str], y_name: str = OUTCOME) -> None:
    for split in ("train", "test"):
        _write(out / f"{split}_X.csv", x_cols)
        with open(out / f"{split}_y.csv", "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerows([[y_name]] + [[str(i % 2)] for i in range(200)])


def _header(path: Path) -> list[str]:
    with open(path, newline="", encoding="utf-8") as fh:
        return next(csv.reader(fh))


def _spec(**kw: Any) -> dict:
    return {**copy.deepcopy(_SPEC), "research_question": "What predicts enrolment?", **kw}


# ---------------------------------------------------------------------------
# Which columns are the outcome's
# ---------------------------------------------------------------------------


def test_the_observed_study_is_found_in_both_matrices(tmp_path: Path) -> None:
    _matrices(tmp_path, PREDICTORS + [OUTCOME])
    assert find_outcome_columns(str(tmp_path), _spec()) == {
        "train_X.csv": [OUTCOME], "test_X.csv": [OUTCOME],
    }


@pytest.mark.parametrize(
    "column",
    [f"{OUTCOME}_Yes", f"{OUTCOME}=Yes", f"{OUTCOME}_1.0", f"cat__{OUTCOME}_Yes",
     f"remainder__{OUTCOME}", OUTCOME.lower()],
    ids=["get_dummies", "dictvectorizer", "numeric-level",
         "columntransformer", "passthrough", "lower-case"],
)
def test_an_encoded_outcome_is_the_outcome(tmp_path: Path, column: str) -> None:
    _matrices(tmp_path, PREDICTORS + [column])
    assert find_outcome_columns(str(tmp_path), _spec())["train_X.csv"] == [column]


def test_the_name_train_y_gives_the_outcome_counts(tmp_path: Path) -> None:
    """A derived outcome carries its own name into train_y.csv."""
    _matrices(tmp_path, PREDICTORS + ["enrolled_college"], y_name="enrolled_college")
    assert find_outcome_columns(str(tmp_path), _spec())["test_X.csv"] == [
        "enrolled_college"
    ]


def test_a_train_y_with_two_named_columns_names_neither(tmp_path: Path) -> None:
    """A stray id written beside y cannot be told from y by name, and a
    predictor of that name must not be dropped for it."""
    _matrices(tmp_path, PREDICTORS + ["pseudo_school_id"])
    with open(tmp_path / "train_y.csv", "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows([["pseudo_school_id", "enrolled"], ["1", "0"]])
    assert find_outcome_columns(str(tmp_path), _spec()) == {}


def test_a_clean_matrix_has_nothing(tmp_path: Path) -> None:
    _matrices(tmp_path, PREDICTORS)
    assert find_outcome_columns(str(tmp_path), _spec()) == {}


def test_an_unnamed_train_y_column_names_nothing(tmp_path: Path) -> None:
    """An unnamed Series is written with the header "0"; a matrix saved
    from an array has a column "0" too. That is a position, not the
    outcome."""
    _matrices(tmp_path, ["0", "1", "2"], y_name="0")
    assert find_outcome_columns(str(tmp_path), _spec()) == {}


def test_a_longer_declared_predictor_is_not_the_outcomes_level(tmp_path: Path) -> None:
    """With the outcome X1SES, the column X1SES_U is the predictor X1SES_U
    (SES with urbanicity), not a level of X1SES."""
    spec = _spec(
        outcome_variable="X1SES",
        predictor_set=[{"variable": "X1SES_U"}, {"variable": "X1TXMTSCOR"}],
    )
    _matrices(tmp_path, ["X1SES_U", "X1TXMTSCOR", "X1SES_high"], y_name="X1SES")
    assert find_outcome_columns(str(tmp_path), spec) == {
        "train_X.csv": ["X1SES_high"], "test_X.csv": ["X1SES_high"],
    }


def test_a_name_that_only_starts_like_the_outcome_is_kept(tmp_path: Path) -> None:
    _matrices(tmp_path, PREDICTORS + [f"{OUTCOME}2", f"PRE{OUTCOME}"])
    assert find_outcome_columns(str(tmp_path), _spec()) == {}


def test_missing_files_are_not_a_finding(tmp_path: Path) -> None:
    assert find_outcome_columns(str(tmp_path), _spec()) == {}


# ---------------------------------------------------------------------------
# The repair removes those columns and nothing else
# ---------------------------------------------------------------------------


def test_the_repair_drops_exactly_the_named_columns(tmp_path: Path) -> None:
    _matrices(tmp_path, ["A", OUTCOME, "B"])
    before = (tmp_path / "train_X.csv").read_text(encoding="utf-8").splitlines()

    left = drop_columns(str(tmp_path), {"train_X.csv": [OUTCOME], "test_X.csv": [OUTCOME]})

    assert left == {"train_X.csv": 2, "test_X.csv": 2}
    after = (tmp_path / "train_X.csv").read_text(encoding="utf-8").splitlines()
    assert after[0] == "A,B"
    for old, new in zip(before, after):
        a, _, b = old.split(",")
        assert new == f"{a},{b}"  # every other cell keeps its text ("0.10")
    assert _header(tmp_path / "test_X.csv") == ["A", "B"]
    assert (tmp_path / "train_y.csv").read_text(encoding="utf-8").startswith(OUTCOME)


def _ctx(tmp_path: Path, spec: dict, report: dict | None) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        research_spec=spec,
        data_report=report,
        output_dir=str(tmp_path),
        current_state=PipelineState.ENGINEERING,
        event_sink=events.EventSink(str(tmp_path)),
    )


def test_before_the_retry_it_is_a_violation_naming_the_columns(tmp_path: Path) -> None:
    _matrices(tmp_path, PREDICTORS + [OUTCOME])
    report = copy.deepcopy(_BINARY_REPORT)
    ctx = _ctx(tmp_path, _spec(), report)
    lines: list[str] = []

    violation = guard_outcome_in_predictors(
        ctx, repair=False, after="data preparation", log=lines.append
    )

    assert violation is not None
    assert "train_X.csv has X4EVRATNDCLG; test_X.csv has X4EVRATNDCLG" in violation
    assert "only in train_y.csv and test_y.csv" in violation
    assert "BEFORE imputation" in violation
    assert OUTCOME in _header(tmp_path / "train_X.csv")  # nothing touched yet
    assert ctx.data_report is report and RECORD_KEY not in report


def test_after_the_retry_it_is_removed_and_said(tmp_path: Path) -> None:
    _matrices(tmp_path, PREDICTORS + [f"{OUTCOME}_Yes"])
    report = {
        **copy.deepcopy(_BINARY_REPORT),
        "n_predictors_encoded": 14,
        "encoding_report": {OUTCOME: {"categories_fitted_on_train": ["No", "Yes"]},
                            "X1SEX": {"categories_fitted_on_train": ["Male", "Female"]}},
    }
    (tmp_path / "data_report.json").write_text(json.dumps(report), encoding="utf-8")
    original = copy.deepcopy(report)
    ctx = _ctx(tmp_path, _spec(), report)
    lines: list[str] = []

    assert guard_outcome_in_predictors(
        ctx, repair=True, after="the DataEngineer's targeted retry", log=lines.append
    ) is None

    assert _header(tmp_path / "train_X.csv") == PREDICTORS
    assert _header(tmp_path / "test_X.csv") == PREDICTORS
    assert report == original  # the agent's dict is replaced, never mutated
    new = ctx.data_report
    assert new[RECORD_KEY]["removed"] == {
        "train_X.csv": [f"{OUTCOME}_Yes"], "test_X.csv": [f"{OUTCOME}_Yes"],
    }
    assert new["n_predictors_encoded"] == 13
    assert set(new["encoding_report"]) == {"X1SEX"}
    note = new["warnings"][-1]
    assert note.startswith("Removed the outcome from the predictors: after the "
                           "DataEngineer's targeted retry")
    assert "imputing" in note
    assert json.loads((tmp_path / "data_report.json").read_text(encoding="utf-8")) == new
    assert lines == [note]
    [warning] = [e for e in _events(tmp_path) if e["type"] == "warning"]
    assert warning["data"]["code"] == REPAIR_CODE
    assert warning["data"]["message"] == note


def test_a_clean_run_is_recorded_without_a_warning(tmp_path: Path) -> None:
    _matrices(tmp_path, PREDICTORS)
    report = copy.deepcopy(_BINARY_REPORT)
    ctx = _ctx(tmp_path, _spec(), report)

    assert guard_outcome_in_predictors(
        ctx, repair=False, after="data preparation", log=lambda _m: None
    ) is None

    assert ctx.data_report[RECORD_KEY] == {"after": "data preparation", "removed": {}}
    assert ctx.data_report["warnings"] == report["warnings"]
    assert not _events(tmp_path)


@pytest.mark.parametrize("task_type", ["causal_soo", "causal_itr", "psychometrics"])
def test_other_task_types_are_left_alone(tmp_path: Path, task_type: str) -> None:
    _matrices(tmp_path, PREDICTORS + [OUTCOME])
    ctx = _ctx(tmp_path, _spec(task_type=task_type), copy.deepcopy(_BINARY_REPORT))

    assert guard_outcome_in_predictors(
        ctx, repair=True, after="data preparation", log=lambda _m: None
    ) is None
    assert OUTCOME in _header(tmp_path / "train_X.csv")
    assert RECORD_KEY not in ctx.data_report


# ---------------------------------------------------------------------------
# In the pipeline
# ---------------------------------------------------------------------------


def _stage(orch: Any, leaks: list[bool]) -> tuple[dict[str, int], list[str | None]]:
    """Stub agents; the DataEngineer's n-th run leaks when ``leaks[n]``."""
    calls = _wire(orch)
    out = Path(orch.ctx.output_dir)
    instructions: list[str | None] = []
    report = {**copy.deepcopy(_BINARY_REPORT), "analytic_n": 1000, "n_test": 200}

    def pf(**_kw: Any) -> dict:
        calls["pf"] += 1
        return {"research_spec": _spec(), "literature_context": None}

    def de(revision_instructions: str | None = None, **_kw: Any) -> dict:
        n = calls["de"]
        calls["de"] += 1
        instructions.append(revision_instructions)
        leak = leaks[min(n, len(leaks) - 1)]
        _matrices(out, PREDICTORS + ([OUTCOME] if leak else []))
        (out / "data_report.json").write_text(json.dumps(report), encoding="utf-8")
        return copy.deepcopy(report)

    orch.problem_formulator.run = pf
    orch.data_engineer.run = de
    return calls, instructions


def test_the_observed_study_gets_one_retry_before_any_analysis(tmp_path: Path) -> None:
    orch = _orch(tmp_path, _config(tmp_path))
    calls, instructions = _stage(orch, [True, False])

    ctx = orch.run()

    assert ctx.current_state == PipelineState.COMPLETED
    assert calls["de"] == 2 and calls["analyst"] == 1
    first, retry = instructions
    assert first is None
    assert "POST-DE PRE-FLIGHT CONTRACT VIOLATION" in retry
    assert "train_X.csv has X4EVRATNDCLG; test_X.csv has X4EVRATNDCLG" in retry
    assert ctx.data_report[RECORD_KEY]["removed"] == {}
    assert ctx.data_report[RECORD_KEY]["after"] == "the DataEngineer's targeted retry"
    assert not [e for e in _events(tmp_path) if e["data"].get("code") == REPAIR_CODE]


def test_a_retry_that_leaks_again_is_repaired_and_the_study_goes_on(
    tmp_path: Path,
) -> None:
    orch = _orch(tmp_path, _config(tmp_path))
    calls, _ = _stage(orch, [True, True])

    ctx = orch.run()

    assert ctx.current_state == PipelineState.COMPLETED
    assert calls == {"pf": 1, "de": 2, "analyst": 1, "critic": 1, "writer": 1}
    assert OUTCOME not in _header(tmp_path / "train_X.csv")
    assert OUTCOME not in _header(tmp_path / "test_X.csv")
    on_disk = json.loads((tmp_path / "data_report.json").read_text(encoding="utf-8"))
    assert any(w.startswith("Removed the outcome from the predictors")
               for w in on_disk["warnings"])
    assert "Removed the outcome from the predictors" in (
        tmp_path / "pipeline.log"
    ).read_text(encoding="utf-8")
    warned = [e for e in _events(tmp_path) if e["type"] == "warning"
              and e["data"].get("code") == REPAIR_CODE]
    assert len(warned) == 1 and warned[0]["stage"] == "ENGINEERING"
    assert _status(tmp_path)["state"] == "COMPLETED"


def test_a_dataengineer_revision_is_checked_too(tmp_path: Path) -> None:
    """A Critic REVISE aimed at the DataEngineer regenerates the code, and
    the Analyst runs next: the same check, repairing at once."""
    orch = _orch(tmp_path, _config(tmp_path))
    _stage(orch, [True])
    orch.ctx.research_spec = _spec()
    orch.ctx.current_state = PipelineState.REVISING

    orch._run_agent("DataEngineer", revision_instructions="fix the imputation")

    assert OUTCOME not in _header(tmp_path / "train_X.csv")
    assert orch.ctx.data_report[RECORD_KEY]["after"] == "a DataEngineer revision"
    [warning] = [e for e in _events(tmp_path) if e["data"].get("code") == REPAIR_CODE]
    assert warning["stage"] == "REVISING"


# ---------------------------------------------------------------------------
# pcc_01 says whether the check ran
# ---------------------------------------------------------------------------


def _pcc_01(tmp_path: Path, report: dict | None, task_type: str = "prediction") -> str:
    _matrices(tmp_path, PREDICTORS + [OUTCOME])
    ctx = types.SimpleNamespace(research_spec=_spec(), data_report=report)
    result = PreCriticResult()
    _check_outcome_not_in_train_x(ctx, str(tmp_path), result, task_type=task_type)
    [failure] = result.failures
    assert failure.check_id == "pcc_01" and failure.revisable is False
    return failure.message


def test_pcc_01_says_the_check_ran_and_found_nothing(tmp_path: Path) -> None:
    message = _pcc_01(
        tmp_path, {RECORD_KEY: {"after": "data preparation", "removed": {}}}
    )
    assert message.startswith(f"Outcome variable '{OUTCOME}' found as a column")
    assert "ran after data preparation and found no outcome column" in message
    assert "written after it ran" in message


def test_pcc_01_says_the_check_did_not_run(tmp_path: Path) -> None:
    message = _pcc_01(tmp_path, {"validation_passed": True})
    assert "did not run on these files" in message


def test_pcc_01_says_other_task_types_have_no_check(tmp_path: Path) -> None:
    assert "this causal_soo study has none" in _pcc_01(tmp_path, None, "causal_soo")


def test_describe_guard_names_what_was_removed() -> None:
    text = describe_guard(
        {RECORD_KEY: {"after": "a DataEngineer revision",
                      "removed": {"train_X.csv": [OUTCOME]}}},
        "prediction",
    )
    assert f"ran after a DataEngineer revision and removed {OUTCOME}" in text
