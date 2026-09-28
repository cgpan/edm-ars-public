"""data_report's class balance is counted from the y files, per sample.

Observed on the owner's Mac (round 3, 2026-09-27), outcome X4EVRATNDCLG:
the paper's Methods said "10,454 students (60.3%) had enrolled ..., and
3,319 (19.1%) had not; the remaining students were excluded due to missing
outcome data". 10,454 + 3,319 = 13,773 = n_train; the analytic sample was
17,335. data_report.class_balance was {class_0: 3319, class_1: 10454},
computed by generated code on y_train, and the Writer read it as the
analytic sample and explained the missing 20% away.

The contract these tests pin:

* after ENGINEERING (and after a DataEngineer revision) the orchestrator
  writes class_balance (analytic sample), class_balance_train and
  class_balance_test, each with counts and shares, counted from
  train_y.csv and test_y.csv;
* a DataEngineer value that says something else is kept as
  class_balance_reported_by_de and the correction is in warnings;
* the Critic, OutlineAgent and Writer read the corrected report;
* continuous outcomes and studies without both y files are untouched.

No provider is called; nothing needs TeX.
"""
from __future__ import annotations

import copy
import csv
import json
import types
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from src.class_balance import REPORTED_KEY, correct_class_balance
from src.context import PipelineState
from src.invariants import RunArtifacts, check_class_balance_sample
from tests.test_orchestrator_terminal import _config, _orch, _wire
from tests.test_pre_critic_revise import _SPEC

OUTCOME = "X4EVRATNDCLG"

#: The round-3 study: train 10,454 / 3,319; test 2,600 / 962 (results.json
#: n_positive_true = 2,600 of 3,562).
TRAIN = {"1.0": 10454, "0.0": 3319}
TEST = {"1.0": 2600, "0.0": 962}

#: What the DataEngineer wrote.
ROUND3_REPORT: dict = {
    "dataset": "hsls09_public",
    "original_n": 23503,
    "analytic_n": 17335,
    "n_train": 13773,
    "n_test": 3562,
    "outcome_variable": OUTCOME,
    "outcome_type": "binary",
    "class_balance": {"class_0": 3319, "class_1": 10454},
    "is_imbalanced": False,
    "missingness_summary": {
        "X1RACE": {"pct_missing": 4.3, "imputation_method": "mode"},
        "X1SEX": {"pct_missing": 0.02, "imputation_method": "mode"},
    },
    "validation_passed": True,
    "warnings": [
        "Multilevel structure (students nested in schools) is not modeled. "
        "This is a limitation."
    ],
}


def _write_y(path: Path, counts: dict[str, int], header: list[str] | None = None,
             index: bool = False) -> None:
    rows: list[list[str]] = []
    i = 0
    for value, n in counts.items():
        for _ in range(n):
            rows.append(([str(i)] if index else []) + [value])
            i += 1
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header or ([""] if index else []) + [OUTCOME])
        w.writerows(rows)


def _round3_files(out: Path, **kw: Any) -> None:
    _write_y(out / "train_y.csv", TRAIN, **kw)
    _write_y(out / "test_y.csv", TEST, **kw)


def _ctx(out: Path, report: dict | None) -> types.SimpleNamespace:
    report = copy.deepcopy(report)
    if report is not None:
        (out / "data_report.json").write_text(json.dumps(report), encoding="utf-8")
    return types.SimpleNamespace(
        output_dir=str(out),
        data_report=report,
        research_spec={"outcome_variable": OUTCOME, "task_type": "prediction"},
    )


def _run(ctx: Any) -> tuple[str | None, list[str]]:
    logged: list[str] = []
    note = correct_class_balance(ctx, log=logged.append)
    return note, logged


# ---------------------------------------------------------------------------
# The recount
# ---------------------------------------------------------------------------


def test_the_observed_study_gets_the_analytic_sample_and_both_splits(
    tmp_path: Path,
) -> None:
    _round3_files(tmp_path)
    ctx = _ctx(tmp_path, ROUND3_REPORT)

    note, _ = _run(ctx)

    report = ctx.data_report
    assert report["class_balance"] == {
        "sample": "analytic sample (train + test)",
        "n": 17335,
        "counts": {"class_0": 4281, "class_1": 13054},
        "shares": {"class_0": 0.247, "class_1": 0.753},
    }
    assert report["class_balance_train"]["counts"] == {"class_0": 3319, "class_1": 10454}
    assert report["class_balance_train"]["n"] == 13773
    assert report["class_balance_train"]["sample"] == "training split"
    assert report["class_balance_test"]["counts"] == {"class_0": 962, "class_1": 2600}
    assert report["class_balance_test"]["n"] == 3562
    assert report["class_balance_test"]["sample"] == "test split"
    # What the DataEngineer said is kept, and the correction is said.
    assert report[REPORTED_KEY] == {"class_0": 3319, "class_1": 10454}
    assert note is not None and note in report["warnings"]
    assert "training split's (n_train = 13,773)" in note
    assert "17,335 students" in note
    assert "class_1 = 13,054 (75.3%)" in note
    # The file the Critic, the checks and a resumed run read says the same.
    on_disk = json.loads((tmp_path / "data_report.json").read_text(encoding="utf-8"))
    assert on_disk == report
    # Nothing else in the report moved.
    for key in ("analytic_n", "n_train", "n_test", "missingness_summary",
                "is_imbalanced", "outcome_type"):
        assert report[key] == ROUND3_REPORT[key]


def test_the_final_check_that_flagged_the_study_no_longer_fires(tmp_path: Path) -> None:
    _round3_files(tmp_path)
    ctx = _ctx(tmp_path, ROUND3_REPORT)
    before = check_class_balance_sample(RunArtifacts(str(tmp_path)))
    assert [f.code for f in before] == ["INV_CLASS_BALANCE_WRONG_SAMPLE"]

    _run(ctx)

    assert check_class_balance_sample(RunArtifacts(str(tmp_path))) == []


def test_a_value_that_already_describes_the_analytic_sample_is_not_a_correction(
    tmp_path: Path,
) -> None:
    _round3_files(tmp_path)
    ctx = _ctx(tmp_path, {**ROUND3_REPORT,
                          "class_balance": {"class_0": 4281, "class_1": 13054}})

    note, _ = _run(ctx)

    assert note is None
    assert REPORTED_KEY not in ctx.data_report
    assert ctx.data_report["warnings"] == ROUND3_REPORT["warnings"]
    assert ctx.data_report["class_balance"]["counts"] == {"class_0": 4281, "class_1": 13054}


@pytest.mark.parametrize(
    "reported, corrected",
    [
        ({"class_0": 0.25, "class_1": 0.75}, False),     # analytic, 2 decimals
        ({"class_0": 0.247, "class_1": 0.753}, False),   # analytic, 3 decimals
        ({"0": 24.7, "1": 75.3}, False),                 # analytic, percent
        ({"class_0": 0.241, "class_1": 0.759}, True),    # training shares
        ({"No": 0.5, "Yes": 0.5}, True),                 # other labels
    ],
)
def test_shares_are_compared_at_the_precision_they_were_reported(
    tmp_path: Path, reported: dict, corrected: bool
) -> None:
    _round3_files(tmp_path)
    ctx = _ctx(tmp_path, {**ROUND3_REPORT, "class_balance": reported})

    note, _ = _run(ctx)

    assert (note is not None) is corrected
    assert (REPORTED_KEY in ctx.data_report) is corrected


def test_running_twice_changes_nothing_the_second_time(tmp_path: Path) -> None:
    _round3_files(tmp_path)
    ctx = _ctx(tmp_path, ROUND3_REPORT)
    _run(ctx)
    first = copy.deepcopy(ctx.data_report)

    note, _ = _run(ctx)

    assert note is None
    assert ctx.data_report == first


def test_a_y_file_written_with_its_index_is_read_by_its_outcome_column(
    tmp_path: Path,
) -> None:
    _round3_files(tmp_path, index=True)
    ctx = _ctx(tmp_path, ROUND3_REPORT)

    _run(ctx)

    assert ctx.data_report["class_balance"]["counts"] == {"class_0": 4281, "class_1": 13054}


def test_a_y_file_with_two_named_columns_is_not_guessed_at(tmp_path: Path) -> None:
    for split, counts in (("train", TRAIN), ("test", TEST)):
        with open(tmp_path / f"{split}_y.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["pseudo_school_id", "enrolled"])
            for value, n in counts.items():
                w.writerows([["7", value]] * n)
    ctx = _ctx(tmp_path, ROUND3_REPORT)

    note, logged = _run(ctx)

    assert note is None
    assert ctx.data_report == ROUND3_REPORT
    assert any("does not have one outcome column" in m for m in logged)


def test_a_continuous_outcome_is_left_alone(tmp_path: Path) -> None:
    _write_y(tmp_path / "train_y.csv", {"3.5": 10, "2.0": 10})
    _write_y(tmp_path / "test_y.csv", {"3.5": 5, "2.0": 5})
    report = {**ROUND3_REPORT, "outcome_type": "continuous", "class_balance": None}
    ctx = _ctx(tmp_path, report)

    assert _run(ctx) == (None, [])
    assert ctx.data_report == report


def test_a_study_without_both_y_files_is_left_alone(tmp_path: Path) -> None:
    _write_y(tmp_path / "train_y.csv", TRAIN)
    ctx = _ctx(tmp_path, ROUND3_REPORT)

    assert _run(ctx) == (None, [])
    assert ctx.data_report == ROUND3_REPORT


def test_a_report_that_is_not_a_dict_is_not_an_error(tmp_path: Path) -> None:
    _round3_files(tmp_path)
    ctx = types.SimpleNamespace(output_dir=str(tmp_path), data_report=None,
                                research_spec=None)
    assert _run(ctx) == (None, [])


# ---------------------------------------------------------------------------
# In the pipeline
# ---------------------------------------------------------------------------


def _de_writing_round3(orch: Any) -> None:
    out = Path(orch.ctx.output_dir)

    def de(**_kw: Any) -> dict:
        _round3_files(out)
        (out / "data_report.json").write_text(json.dumps(ROUND3_REPORT), encoding="utf-8")
        return copy.deepcopy(ROUND3_REPORT)

    orch.data_engineer.run = de


def test_the_critic_and_the_writer_see_the_analytic_sample(tmp_path: Path) -> None:
    orch = _orch(tmp_path, _config(tmp_path))
    _wire(orch)
    _de_writing_round3(orch)
    seen: dict[str, Any] = {}
    critic_run, writer_run = orch.critic.run, orch.writer.run

    def critic(**kw: Any) -> Any:
        seen["critic"] = copy.deepcopy(orch.ctx.data_report)
        return critic_run(**kw)

    def writer(**kw: Any) -> Any:
        seen["writer"] = copy.deepcopy(orch.ctx.data_report)
        return writer_run(**kw)

    orch.critic.run, orch.writer.run = critic, writer
    orch.problem_formulator.run = lambda **_kw: {
        "research_spec": {**copy.deepcopy(_SPEC),
                          "research_question": "What predicts enrolment?"},
        "literature_context": None,
    }

    ctx = orch.run()

    assert ctx.current_state == PipelineState.COMPLETED, ctx.errors
    for stage in ("critic", "writer"):
        report = seen[stage]
        assert report["class_balance"]["n"] == 17335, stage
        assert report["class_balance"]["counts"] == {"class_0": 4281, "class_1": 13054}
        assert report[REPORTED_KEY] == {"class_0": 3319, "class_1": 10454}
    assert "class_balance was recounted" in (tmp_path / "pipeline.log").read_text(
        encoding="utf-8"
    )


def test_a_dataengineer_revision_is_recounted_too(tmp_path: Path) -> None:
    orch = _orch(tmp_path, _config(tmp_path))
    _wire(orch)
    _de_writing_round3(orch)
    orch.ctx.research_spec = {**copy.deepcopy(_SPEC), "outcome_variable": OUTCOME}
    orch.ctx.current_state = PipelineState.REVISING

    orch._run_agent("DataEngineer", revision_instructions="fix the imputation")

    assert orch.ctx.data_report["class_balance"]["n"] == 17335
    assert orch.ctx.data_report[REPORTED_KEY] == {"class_0": 3319, "class_1": 10454}


def test_the_writer_and_outline_prompts_carry_the_labelled_fields(tmp_path: Path) -> None:
    """Both read data_report from the context; what they are sent is what
    the recount wrote, labels included."""
    from src.agents.outline_agent import OutlineAgent
    from tests.test_writer import _llm_response_with_both_blocks, _make_agent

    _round3_files(tmp_path)
    ctx = _ctx(tmp_path, ROUND3_REPORT)
    _run(ctx)

    writer = _make_agent(tmp_path)
    writer.ctx.data_report = ctx.data_report
    sent: list[str] = []

    def fake_llm(message: str, **_kw: Any) -> str:
        sent.append(message)
        return _llm_response_with_both_blocks()

    writer.call_llm = fake_llm
    writer.run()
    with patch("anthropic.Anthropic"):
        outline = OutlineAgent(writer.ctx, "outline_agent", writer.config)
    outline_message = outline._build_user_message(
        research_spec=writer.ctx.research_spec,
        data_report=writer.ctx.data_report,
        results_object=writer.ctx.results_object,
        triggers={},
    )

    for message in (sent[0], outline_message):
        assert '"sample": "analytic sample (train + test)"' in message
        assert '"class_1": 13054' in message
        assert '"sample": "training split"' in message
        assert REPORTED_KEY in message
        assert "class_balance was recounted" in message

