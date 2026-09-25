"""src/main.py: how a run is asked for, checked, started and reported.

Covers the launch defects C1-C5, C7, the resume/output-folder defects and
B4 (the end-of-run summary):
every test drives ``main(argv)`` in-process with a stubbed Orchestrator, so
nothing here calls a provider or runs a pipeline stage.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

import src.main as main_mod
from src.context import PipelineState
from src.main import main
from src.preflight import Finding

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "runs" / "fixtures"
HSLS_FILE = "hsls_17_student_pets_sr_v1_0.csv"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """A config pointing raw data and output at tmp, run from a foreign cwd."""
    monkeypatch.setenv("LSAR_HOME", "")
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / HSLS_FILE).write_text("X1SEX\nMale\n", encoding="utf-8")
    out_base = tmp_path / "output"
    with open(REPO_ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"]["raw_data"] = str(raw) + os.sep
    cfg["paths"]["output_base"] = str(out_base) + os.sep
    cfg["findings_memory"]["enabled"] = False
    cfg["review_gate"]["enabled"] = False
    cfg["sandbox"]["enabled"] = False
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    return {"root": tmp_path, "config": config_path, "raw": raw,
            "out_base": out_base, "cwd": cwd}


def _write_config(env: dict[str, Path], **pipeline: Any) -> Path:
    with open(env["config"], encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["pipeline"].update(pipeline)
    path = env["root"] / "config_variant.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


class _NoOrchestrator:
    def __init__(self, *a: Any, **kw: Any) -> None:
        raise AssertionError("the Orchestrator must not be constructed here")


class _StubOrchestrator:
    """Records the context main() built; run() ends in ``final_state`` and
    writes ``status`` as run_status.json when one is given."""

    instances: list["_StubOrchestrator"] = []
    final_state: PipelineState = PipelineState.COMPLETED
    status: dict | None = None
    errors: list[str] = []

    def __init__(self, ctx: Any, config: dict, config_path: str = "") -> None:
        self.ctx = ctx
        self.config = config
        self.config_path = config_path
        os.makedirs(ctx.output_dir, exist_ok=True)
        _StubOrchestrator.instances.append(self)

    def run(self, user_prompt: str | None = None) -> Any:
        self.ctx.current_state = self.final_state
        self.ctx.errors = list(self.errors)
        if self.status is not None:
            path = os.path.join(self.ctx.output_dir, "run_status.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.status, f)
        return self.ctx


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> type[_StubOrchestrator]:
    _StubOrchestrator.instances = []
    _StubOrchestrator.final_state = PipelineState.COMPLETED
    _StubOrchestrator.status = None
    _StubOrchestrator.errors = []
    monkeypatch.setattr(main_mod, "Orchestrator", _StubOrchestrator)
    monkeypatch.setattr(main_mod, "check_run_prerequisites", lambda *a, **k: [])
    return _StubOrchestrator


@pytest.fixture
def no_orchestrator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main_mod, "Orchestrator", _NoOrchestrator)


def _tree(path: Path) -> list[str]:
    return sorted(str(p.relative_to(path)) for p in path.rglob("*"))


def _checkpoint(folder: Path, **fields: Any) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    data = {
        "dataset_name": "hsls09_public", "raw_data_path": "x",
        "output_dir": str(folder), "task_type": "prediction",
        "current_state": "ANALYZING", "locked_research_spec": None,
    }
    data.update(fields)
    path = folder / "checkpoint.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# C3: the dry run is side-effect free and needs no key
# ---------------------------------------------------------------------------


def test_dry_run_without_a_key_reports_it_instead_of_crashing(
    env: dict[str, Path], no_orchestrator: None, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for var in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "MINIMAX_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    run_dir = env["root"] / "old_run"
    ckpt = _checkpoint(run_dir)
    before = _tree(env["root"])

    code = main(["--dry-run", "--config", str(env["config"]),
                 "--output-dir", str(run_dir)])

    out, err = capsys.readouterr()
    assert code == 1
    assert "DEEPSEEK_API_KEY" in out and "KEY_MISSING" in out
    assert "Traceback" not in out + err
    assert ckpt.exists()  # the resume point survives a dry run
    assert _tree(env["root"]) == before  # nothing created or deleted


def test_dry_run_with_everything_in_place_passes_and_writes_nothing(
    env: dict[str, Path], no_orchestrator: None, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(main_mod, "check_run_prerequisites", lambda *a, **k: [])
    before = _tree(env["root"])
    code = main(["--dry-run", "--config", str(env["config"])])
    out = capsys.readouterr().out
    assert code == 0
    assert "a real run would start" in out
    assert "skill_registry count: 0" not in out  # skills found from a foreign cwd
    assert _tree(env["root"]) == before
    assert not env["out_base"].exists()  # no output/run_* folder either


def test_dry_run_fails_when_the_data_file_is_missing(
    env: dict[str, Path], no_orchestrator: None, capsys: pytest.CaptureFixture[str],
) -> None:
    (env["raw"] / HSLS_FILE).unlink()
    code = main(["--dry-run", "--config", str(env["config"])])
    out = capsys.readouterr().out
    assert code == 1
    assert "DATA_MISSING" in out and HSLS_FILE in out
    assert "raw_data exists:      False" in out


# ---------------------------------------------------------------------------
# C1: non-prediction study types need a locked spec; causal prompts get a note
# ---------------------------------------------------------------------------


def test_causal_task_type_without_a_spec_is_refused(
    env: dict[str, Path], no_orchestrator: None, capsys: pytest.CaptureFixture[str],
) -> None:
    config = _write_config(env, task_type="causal_soo")
    code = main(["--config", str(config)])
    err = capsys.readouterr().err
    assert code == 1
    assert "--research-spec" in err and "causal_soo" in err
    assert "runs/fixtures/spec_x1mtheff_x4college.json" in err
    assert "Traceback" not in err


def test_unregistered_task_type_is_a_usage_error(
    env: dict[str, Path], no_orchestrator: None, capsys: pytest.CaptureFixture[str],
) -> None:
    config = _write_config(env, task_type="causal_inference")
    assert main(["--dry-run", "--config", str(config)]) == 1
    err = capsys.readouterr().err
    assert "causal_inference" in err and "psychometrics" in err


def test_a_causal_prompt_gets_a_notice_and_still_runs_as_prediction(
    env: dict[str, Path], no_orchestrator: None, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(main_mod, "check_run_prerequisites", lambda *a, **k: [])
    code = main(["--dry-run", "--config", str(env["config"]), "--prompt",
                 "What is the effect of ninth-grade math self-efficacy on "
                 "college enrollment?"])
    out, err = capsys.readouterr()
    assert code == 0
    assert "NOTE" in err and "causal_soo" in err
    assert "task_type:            prediction" in out


def test_a_predictive_prompt_gets_no_notice(
    env: dict[str, Path], no_orchestrator: None, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(main_mod, "check_run_prerequisites", lambda *a, **k: [])
    main(["--dry-run", "--config", str(env["config"]), "--prompt",
          "Which ninth-grade attitudes predict math GPA?"])
    assert "NOTE" not in capsys.readouterr().err


# ---------------------------------------------------------------------------
# C2: the locked spec's dataset wins; an explicit disagreement is an error
# ---------------------------------------------------------------------------


def test_the_spec_dataset_is_used_when_dataset_is_omitted(
    env: dict[str, Path], stub: type[_StubOrchestrator],
) -> None:
    code = main(["--config", str(env["config"]), "--research-spec",
                 str(FIXTURES / "spec_did_ses_gap.json"),
                 "--output-dir", str(env["root"] / "did")])
    assert code == 0
    ctx = stub.instances[0].ctx
    assert ctx.dataset_name == "did_els_hsls_panel"
    assert ctx.task_type == "causal_did"
    assert ctx.raw_data_path.endswith(os.path.join("did_els_hsls_panel", "panel.csv"))


def test_a_conflicting_dataset_names_both(
    env: dict[str, Path], no_orchestrator: None, capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["--dry-run", "--config", str(env["config"]), "--dataset",
                 "hsls09_public", "--research-spec",
                 str(FIXTURES / "spec_did_ses_gap.json")])
    err = capsys.readouterr().err
    assert code == 1
    assert "hsls09_public" in err and "did_els_hsls_panel" in err


def test_a_spec_without_a_dataset_keeps_the_flag(
    env: dict[str, Path], stub: type[_StubOrchestrator], tmp_path: Path,
) -> None:
    spec = json.loads((FIXTURES / "spec_x1mtheff_x4college.json").read_text(encoding="utf-8"))
    spec.pop("dataset", None)
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    assert main(["--config", str(env["config"]), "--dataset", "hsls09_public",
                 "--research-spec", str(path),
                 "--output-dir", str(tmp_path / "soo")]) == 0
    assert stub.instances[0].ctx.dataset_name == "hsls09_public"


# ---------------------------------------------------------------------------
# C7: setup errors are one readable line, exit 1, no traceback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("argv, needle", [
    (["--dataset", "hsls09"], "unknown dataset 'hsls09'"),
    (["--config", "no_such.yaml"], "no_such.yaml"),
])
def test_setup_errors_are_one_line(
    env: dict[str, Path], no_orchestrator: None, argv: list[str], needle: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    if "--config" not in argv:
        argv = argv + ["--config", str(env["config"])]
    assert main(argv + ["--dry-run"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ") and needle in err
    assert "Traceback" not in err


def test_malformed_spec_names_the_file_and_position(
    env: dict[str, Path], no_orchestrator: None, tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bad = tmp_path / "bad_spec.json"
    bad.write_text('{"task_type": "causal_soo",}', encoding="utf-8")
    assert main(["--dry-run", "--config", str(env["config"]),
                 "--research-spec", str(bad)]) == 1
    err = capsys.readouterr().err
    assert "bad_spec.json" in err and "line 1" in err and "column" in err
    assert "Traceback" not in err


def test_invalid_spec_is_a_usage_error(
    env: dict[str, Path], no_orchestrator: None, tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "no_treatment.json"
    path.write_text(json.dumps({"task_type": "causal_soo", "primary_method": "M2"}),
                    encoding="utf-8")
    assert main(["--dry-run", "--config", str(env["config"]),
                 "--research-spec", str(path)]) == 1
    err = capsys.readouterr().err
    assert "failed structural validation" in err and "Traceback" not in err


def test_debug_shows_the_traceback(
    env: dict[str, Path], no_orchestrator: None, capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--dry-run", "--debug", "--dataset", "nope",
                 "--config", str(env["config"])]) == 1
    assert "Traceback" in capsys.readouterr().err


def test_a_bad_flag_exits_1_not_2(env: dict[str, Path]) -> None:
    with pytest.raises(SystemExit) as info:
        main(["--no-such-flag"])
    assert info.value.code == 1


# ---------------------------------------------------------------------------
# Pre-flight gates a real run
# ---------------------------------------------------------------------------


def test_a_failed_preflight_starts_nothing_and_deletes_nothing(
    env: dict[str, Path], no_orchestrator: None, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(main_mod, "check_run_prerequisites", lambda *a, **k: [
        Finding("DATA_MISSING", "fail", "The data file was not found.", "Get it."),
        Finding("LATEX_MISSING", "warn", "No pdflatex.", "Install TeX."),
    ])
    run_dir = env["root"] / "old"
    ckpt = _checkpoint(run_dir)
    code = main(["--config", str(env["config"]), "--output-dir", str(run_dir),
                 "--overwrite"])
    err = capsys.readouterr().err
    assert code == 1
    assert "DATA_MISSING" in err and "LATEX_MISSING" in err
    assert "nothing was sent or spent" in err
    assert ckpt.exists()  # --overwrite only acts once the run can start


# ---------------------------------------------------------------------------
# C4: an output folder that holds a run is refused, or deliberately cleared
# ---------------------------------------------------------------------------


def test_a_folder_with_a_checkpoint_is_refused_without_resume(
    env: dict[str, Path], no_orchestrator: None, capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = env["root"] / "run"
    ckpt = _checkpoint(run_dir, current_state="WRITING")
    code = main(["--config", str(env["config"]), "--output-dir", str(run_dir)])
    err = capsys.readouterr().err
    assert code == 1
    assert "WRITING" in err and "--resume" in err and "--overwrite" in err
    assert ckpt.exists()


def test_a_folder_with_leftovers_but_no_checkpoint_is_refused(
    env: dict[str, Path], no_orchestrator: None, capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = env["root"] / "run"
    run_dir.mkdir()
    (run_dir / "run_status.json").write_text('{"released": true}', encoding="utf-8")
    assert main(["--config", str(env["config"]), "--output-dir", str(run_dir)]) == 1
    err = capsys.readouterr().err
    assert "run_status.json" in err and "--overwrite" in err


def test_a_front_end_prepared_folder_is_not_mistaken_for_a_run(
    env: dict[str, Path], stub: type[_StubOrchestrator],
) -> None:
    run_dir = env["root"] / "study"
    run_dir.mkdir()
    for name in ("run_config.yaml", "runner.json", "console.log",
                 "research_spec.locked.json"):
        (run_dir / name).write_text("x", encoding="utf-8")
    assert main(["--config", str(env["config"]), "--output-dir", str(run_dir)]) == 0


def test_overwrite_removes_only_the_earlier_runs_files(
    env: dict[str, Path], stub: type[_StubOrchestrator],
) -> None:
    run_dir = env["root"] / "run"
    _checkpoint(run_dir)
    for name in ("run_status.json", "invariants.json", "obligations.json",
                 "events.jsonl", "live_status.json", "paper.pdf", "paper.log",
                 "paper.tex", "token_usage.jsonl", "pipeline.log",
                 "train_school_ids.csv", "checkpoint.json.4242.tmp",
                 "live_status.json.tmp",
                 # An earlier run's figures: the Analyst and Writer adopt
                 # every image in the folder, so one left behind would be
                 # embedded in the new paper.
                 "love_plot.png", "cate_distribution.pdf", "roc_curves.png",
                 "run_config.yaml", "my_notes.txt"):
        (run_dir / name).write_text("old", encoding="utf-8")
    (run_dir / "prompts" / "writer").mkdir(parents=True)
    (run_dir / "prompts" / "writer" / "x.txt").write_text("old", encoding="utf-8")

    assert main(["--config", str(env["config"]), "--output-dir", str(run_dir),
                 "--overwrite"]) == 0
    assert sorted(p.name for p in run_dir.iterdir()) == ["my_notes.txt", "run_config.yaml"]


def test_resume_and_overwrite_together_are_refused(
    env: dict[str, Path], no_orchestrator: None, capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = env["root"] / "run"
    _checkpoint(run_dir)
    assert main(["--config", str(env["config"]), "--output-dir", str(run_dir),
                 "--resume", "--overwrite"]) == 1
    assert "one or the other" in capsys.readouterr().err


def test_a_new_run_folder_never_reuses_an_existing_one(
    env: dict[str, Path], stub: type[_StubOrchestrator], monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Frozen(main_mod.datetime):  # type: ignore[misc,name-defined]
        @classmethod
        def now(cls, tz: Any = None) -> Any:
            return main_mod.datetime(2026, 1, 2, 3, 4, 5)

    monkeypatch.setattr(main_mod, "datetime", _Frozen)
    taken = env["out_base"] / "run_20260102_030405"
    taken.mkdir(parents=True)
    (taken / "checkpoint.json").write_text("{}", encoding="utf-8")
    assert main(["--config", str(env["config"])]) == 0
    used = Path(stub.instances[0].ctx.output_dir)
    assert used.parent == env["out_base"] and used != taken
    assert (taken / "checkpoint.json").read_text(encoding="utf-8") == "{}"


# ---------------------------------------------------------------------------
# --resume keeps what the run started with
# ---------------------------------------------------------------------------


def test_resume_takes_type_dataset_and_spec_from_the_checkpoint(
    env: dict[str, Path], stub: type[_StubOrchestrator],
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec = json.loads((FIXTURES / "spec_did_ses_gap.json").read_text(encoding="utf-8"))
    run_dir = env["root"] / "did_run"
    _checkpoint(run_dir, task_type="causal_did", dataset_name="did_els_hsls_panel",
                locked_research_spec=spec, current_state="ANALYZING")
    # The README's resume example: --dataset hsls09_public, no spec.
    code = main(["--config", str(env["config"]), "--dataset", "hsls09_public",
                 "--output-dir", str(run_dir), "--resume"])
    err = capsys.readouterr().err
    assert code == 0
    ctx = stub.instances[0].ctx
    assert (ctx.task_type, ctx.dataset_name) == ("causal_did", "did_els_hsls_panel")
    assert ctx.locked_research_spec == spec
    assert "--dataset hsls09_public is ignored" in err
    assert "continues as one" in err


def test_resume_without_output_dir_does_not_start_a_new_run(
    env: dict[str, Path], no_orchestrator: None, capsys: pytest.CaptureFixture[str],
) -> None:
    only = env["out_base"] / "run_20260101_000000"
    _checkpoint(only, current_state="CRITIQUING")
    _checkpoint(env["out_base"] / "run_done", current_state="COMPLETED")
    code = main(["--config", str(env["config"]), "--resume"])
    err = capsys.readouterr().err
    assert code == 1
    assert "--resume needs --output-dir" in err
    assert "run_20260101_000000" in err and "CRITIQUING" in err
    assert "run_done" not in err


def test_resume_of_a_folder_without_checkpoint_is_refused(
    env: dict[str, Path], no_orchestrator: None, capsys: pytest.CaptureFixture[str],
) -> None:
    empty = env["root"] / "empty"
    empty.mkdir()
    assert main(["--config", str(env["config"]), "--output-dir", str(empty),
                 "--resume"]) == 1
    assert "nothing to resume" in capsys.readouterr().err


def test_resume_past_engineering_only_warns_about_missing_data(
    env: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    (env["raw"] / HSLS_FILE).unlink()
    run_dir = env["root"] / "late"
    _checkpoint(run_dir, current_state="WRITING")
    plan = main_mod._plan_run(main_mod._build_parser().parse_args(
        ["--config", str(env["config"]), "--output-dir", str(run_dir), "--resume"]))
    monkeypatch.setattr(main_mod, "check_run_prerequisites", lambda *a, **k: [
        Finding("DATA_MISSING", "fail", "The data file was not found.", "Get it.")])
    [finding] = main_mod._preflight(plan)
    assert finding.severity == "warn"
    assert "back to data preparation" in finding.message

# ---------------------------------------------------------------------------
# B4 and stale run_status: how the end of a run is reported
#
# The console used to print ``Final state: PipelineState.COMPLETED`` and
# ``Release: YES (1 critical invariant finding(s); review gate did not
# pass)`` -- an enum repr, and a release line followed by what read as its
# blockers -- and it printed whatever run_status.json sat in the folder,
# including an earlier run's.
# ---------------------------------------------------------------------------


def _status_v2(**fields: Any) -> dict:
    status = {
        "schema": 2, "state": "COMPLETED", "released": True, "reason": "clean",
        "reason_code": "CLEAN", "advisories": [], "abort": None,
        "gate": {"enabled": False, "ran": False, "skip_reason": "disabled",
                 "passed": None, "score": None, "threshold": None,
                 "advisory": None, "venue": "EDM"},
        "invariant_counts": {"critical": 0, "major": 0, "minor": 0},
        "blocking_findings": [],
        "written_at": "2999-01-01T00:00:00Z",
    }
    status.update(fields)
    return status


def _run(env: dict[str, Path], *extra: str) -> int:
    return main(["--config", str(env["config"]),
                 "--output-dir", str(env["root"] / "run"), *extra])


def test_released_with_advisories_is_not_a_contradiction(
    env: dict[str, Path], stub: type[_StubOrchestrator],
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The released (v1) status shape: the reason lists what did not block.
    stub.status = {
        "released": True,
        "reason": "1 critical invariant finding(s); review gate did not pass",
        "invariant_counts": {"critical": 1, "major": 2, "minor": 0},
        "blocking_findings": [],
    }
    assert _run(env) == 0
    out = capsys.readouterr().out
    assert "PipelineState." not in out
    assert "Run finished: COMPLETED" in out
    assert "Released: yes" in out
    for line in out.splitlines():
        assert not re.match(r"^Release[d]?: (YES|yes) \(.*critical", line), line
    assert "Did not block release, but worth checking: 1 critical" in out
    assert "Final checks: 1 critical, 2 major, 0 minor" in out


def test_blocked_release_says_what_blocked_it(
    env: dict[str, Path], stub: type[_StubOrchestrator],
    capsys: pytest.CaptureFixture[str],
) -> None:
    stub.final_state = PipelineState.INCOMPLETE
    stub.status = _status_v2(
        state="INCOMPLETE", released=False, reason_code="BLOCKING_FINDINGS",
        reason="1 critical invariant finding(s)",
        blocking_findings=["INV_LATEX_NO_PDF"],
    )
    stub.errors = ["Release blocked by 1 critical invariant finding(s): INV_LATEX_NO_PDF"]
    assert _run(env) == 2
    out, err = capsys.readouterr()
    assert "Released: no - a final check blocked release: INV_LATEX_NO_PDF" in out
    assert "Paper: none was written" in out
    # Errors one per line, not a Python list repr.
    assert "  - Release blocked by 1 critical" in err
    assert "['Release" not in err


def test_gate_that_did_not_run_is_said_in_words(
    env: dict[str, Path], stub: type[_StubOrchestrator],
    capsys: pytest.CaptureFixture[str],
) -> None:
    stub.status = _status_v2(
        reason_code="GATE_NOT_RUN",
        advisories=["review gate did not run (lsar_not_found)"],
        gate={"enabled": True, "ran": False, "skip_reason": "lsar_not_found",
              "passed": None, "score": None, "threshold": None,
              "advisory": None, "venue": "EDM"},
    )
    assert _run(env) == 0
    out = capsys.readouterr().out
    assert "Automated peer review (LSAR): did not run (lsar_not_found)" in out
    assert "0.0" not in out


def test_gate_score_is_shown_against_its_benchmark(
    env: dict[str, Path], stub: type[_StubOrchestrator],
    capsys: pytest.CaptureFixture[str],
) -> None:
    stub.status = _status_v2(
        reason_code="GATE_FAILED", advisories=["review gate did not pass"],
        gate={"enabled": True, "ran": True, "skip_reason": None, "passed": False,
              "score": 5.9, "threshold": 6.3, "advisory": False, "venue": "EDM"},
    )
    _run(env)
    out = capsys.readouterr().out
    assert "score 5.90, benchmark 6.30 - below the benchmark" in out


def test_a_paper_pdf_is_pointed_to(
    env: dict[str, Path], stub: type[_StubOrchestrator],
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    original_run = _StubOrchestrator.run

    def _run_with_pdf(self: _StubOrchestrator, user_prompt: str | None = None) -> Any:
        Path(self.ctx.output_dir, "paper.pdf").write_bytes(b"%PDF-1.5")
        return original_run(self, user_prompt)

    monkeypatch.setattr(_StubOrchestrator, "run", _run_with_pdf)
    stub.status = _status_v2()
    _run(env)
    out = capsys.readouterr().out
    assert f"Paper: {env['root'] / 'run' / 'paper.pdf'}" in out


def test_an_aborted_run_never_prints_an_earlier_release(
    env: dict[str, Path], stub: type[_StubOrchestrator],
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = env["root"] / "run"
    _checkpoint(run_dir, current_state="ENGINEERING")
    stale = run_dir / "run_status.json"
    stale.write_text(json.dumps({
        "released": True, "reason": "clean",
        "timestamp": "2020-01-01T00:00:00",
    }), encoding="utf-8")
    old = time.time() - 3600
    os.utime(stale, (old, old))
    stub.final_state = PipelineState.ABORTED
    assert _run(env, "--resume") == 3
    out = capsys.readouterr().out
    assert "Released: yes" not in out
    assert "Released: no - the run stopped before it finished" in out


def test_abort_reason_and_resume_command_when_resumable(
    env: dict[str, Path], stub: type[_StubOrchestrator],
    capsys: pytest.CaptureFixture[str],
) -> None:
    stub.final_state = PipelineState.ABORTED
    stub.status = _status_v2(
        state="ABORTED", released=False, reason_code="ABORTED",
        abort={"stage": "FORMULATING", "code": "NO_CREDIT",
               "message": "the DeepSeek account has no balance", "resumable": True},
    )
    assert _run(env) == 3
    out = capsys.readouterr().out
    assert "Run stopped: ABORTED during FORMULATING" in out
    assert "NO_CREDIT - the DeepSeek account has no balance" in out
    assert "After fixing the cause, continue the run with:" in out
    assert "--resume" in out


def test_no_resume_command_when_the_abort_is_final(
    env: dict[str, Path], stub: type[_StubOrchestrator],
    capsys: pytest.CaptureFixture[str],
) -> None:
    stub.final_state = PipelineState.ABORTED
    stub.status = _status_v2(
        state="ABORTED", released=False, reason_code="ABORTED",
        abort={"stage": "CRITIQUING", "code": "CRITIC_ABORT",
               "message": "confirmed leakage", "resumable": False},
    )
    _run(env)
    assert "--resume" not in capsys.readouterr().out


def test_json_summary_is_one_object_on_stdout(
    env: dict[str, Path], stub: type[_StubOrchestrator],
    capsys: pytest.CaptureFixture[str],
) -> None:
    stub.status = _status_v2()
    assert _run(env, "--json-summary") == 0
    out, err = capsys.readouterr()
    payload = json.loads(out)
    assert payload["state"] == "COMPLETED"
    assert payload["exit_code"] == 0
    assert payload["released"] is True
    assert payload["run_status_path"] == str(env["root"] / "run" / "run_status.json")
    assert payload["run_status"]["reason_code"] == "CLEAN"
    assert "Released: yes" in err  # the readable summary moved to stderr


def test_json_summary_for_dry_run_and_usage_errors(
    env: dict[str, Path], no_orchestrator: None, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert main(["--dry-run", "--json-summary", "--config", str(env["config"])]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True and payload["would_start"] is False
    assert "KEY_MISSING" in [c["code"] for c in payload["checks"]]

    assert main(["--json-summary", "--dataset", "nope",
                 "--config", str(env["config"])]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"error": payload["error"], "exit_code": 1}
    assert "nope" in payload["error"]


def test_resuming_a_finished_run_reports_it_without_starting_anything(
    env: dict[str, Path], no_orchestrator: None, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)  # not needed
    run_dir = env["root"] / "run"
    _checkpoint(run_dir, current_state="COMPLETED")
    (run_dir / "run_status.json").write_text(json.dumps(_status_v2()), encoding="utf-8")
    assert _run(env, "--resume") == 0
    out = capsys.readouterr().out
    assert "already finished (COMPLETED)" in out
    assert "Released: yes" in out


# ---------------------------------------------------------------------------
# Unit level
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("state, status, code", [
    ("COMPLETED", {"released": True}, 0),
    ("COMPLETED", None, 0),
    ("COMPLETED", {"released": False}, 2),
    ("INCOMPLETE", {"released": False}, 2),
    ("ABORTED", None, 3),
    ("ABORTED", {"abort": {"code": "CRASHED"}}, 5),
    ("INTERRUPTED", None, 4),
    ("CRASHED", None, 5),
    ("ANALYZING", None, 3),
])
def test_exit_codes(state: str, status: dict | None, code: int) -> None:
    assert main_mod._exit_code(state, status) == code


def test_state_names_are_plain() -> None:
    assert main_mod._state_name(PipelineState.COMPLETED) == "COMPLETED"
    assert main_mod._state_name("PipelineState.ABORTED") == "ABORTED"
    assert main_mod._state_name("INCOMPLETE") == "INCOMPLETE"


def test_status_from_this_run_is_recognised(tmp_path: Path) -> None:
    path = tmp_path / "run_status.json"
    path.write_text(json.dumps({"released": True,
                                "written_at": "2026-09-25T10:00:00Z"}), encoding="utf-8")
    old = time.time() - 3600
    os.utime(path, (old, old))
    now = time.time()
    # Old file, stamped before this run started: not ours.
    assert main_mod._current_status(str(tmp_path), "2026-09-25T11:00:00+00:00", now)[1] is None
    # Stamped after the (resumed) run's original start: ours.
    assert main_mod._current_status(str(tmp_path), "2026-09-25T09:00:00", now)[1] is not None
    # Written during this invocation: ours.
    os.utime(path, None)
    assert main_mod._current_status(str(tmp_path), "", now - 1)[1] is not None


def test_a_real_orchestrator_run_is_summarised_from_its_own_status(
    env: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The real Orchestrator with the end-to-end suite's stub agents."""
    from src.orchestrator import Orchestrator
    from tests.test_end_to_end import _wire_stubs

    class _Wired(Orchestrator):
        def __init__(self, *a: Any, **kw: Any) -> None:
            super().__init__(*a, **kw)
            _wire_stubs(self)

    monkeypatch.setattr(main_mod, "Orchestrator", _Wired)
    monkeypatch.setattr(main_mod, "check_run_prerequisites", lambda *a, **k: [])
    run_dir = env["root"] / "real"
    code = main(["--config", str(env["config"]), "--output-dir", str(run_dir)])
    out = capsys.readouterr().out
    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    state = "COMPLETED" if status["released"] else "INCOMPLETE"
    assert code == (0 if status["released"] else 2)
    assert f"Run finished: {state}" in out
    assert ("Released: yes" if status["released"] else "Released: no") in out
    assert "PipelineState." not in out
    assert f"Run folder: {run_dir}" in out


# ---------------------------------------------------------------------------
# D4: Ctrl-C, termination signals and crashes
# ---------------------------------------------------------------------------


class _StoppingOrchestrator(_StubOrchestrator):
    """run() raises ``raise_in_run``; records finalize_interrupted calls."""

    raise_in_run: BaseException | None = None
    finalized: list[tuple[str, str]] = []

    def run(self, user_prompt: str | None = None) -> Any:
        self.ctx.current_state = PipelineState.ANALYZING
        exc = type(self).__dict__.get("raise_in_run")  # a function stays unbound
        if callable(exc) and not isinstance(exc, BaseException):
            exc()
        assert exc is not None
        raise exc

    def finalize_interrupted(self, code: str, message: str) -> None:
        _StoppingOrchestrator.finalized.append((code, message))


@pytest.fixture
def stopping(monkeypatch: pytest.MonkeyPatch) -> type[_StoppingOrchestrator]:
    _StoppingOrchestrator.instances = []
    _StoppingOrchestrator.finalized = []
    monkeypatch.setattr(main_mod, "Orchestrator", _StoppingOrchestrator)
    monkeypatch.setattr(main_mod, "check_run_prerequisites", lambda *a, **k: [])
    return _StoppingOrchestrator


def test_ctrl_c_during_a_run_exits_4_with_the_resume_command(
    env: dict[str, Path], stopping: type[_StoppingOrchestrator],
    capsys: pytest.CaptureFixture[str],
) -> None:
    stopping.raise_in_run = KeyboardInterrupt()
    assert _run(env) == 4
    out, err = capsys.readouterr()
    assert stopping.finalized == [("INTERRUPTED", "Stopped by Ctrl-C")]
    assert "Run interrupted during ANALYZING" in out
    assert "--resume" in out and "Traceback" not in out + err


def test_a_crash_writes_crash_log_and_exits_5(
    env: dict[str, Path], stopping: type[_StoppingOrchestrator],
    capsys: pytest.CaptureFixture[str],
) -> None:
    stopping.raise_in_run = RuntimeError("boom in the analyst")
    assert _run(env) == 5
    out, err = capsys.readouterr()
    log = (env["root"] / "run" / "crash.log").read_text(encoding="utf-8")
    assert "Traceback" in log and "RuntimeError: boom in the analyst" in log
    assert "during ANALYZING" in log
    assert stopping.finalized == [("CRASHED", "RuntimeError: boom in the analyst")]
    assert err.count("\n") >= 1 and "Traceback" not in err
    assert "crash.log" in err and "boom in the analyst" in err
    assert "--resume" in out


def test_debug_prints_the_crash_traceback(
    env: dict[str, Path], stopping: type[_StoppingOrchestrator],
    capsys: pytest.CaptureFixture[str],
) -> None:
    stopping.raise_in_run = RuntimeError("boom")
    assert _run(env, "--debug") == 5
    assert "Traceback" in capsys.readouterr().err


def test_sigterm_is_handled_like_ctrl_c_and_handlers_are_restored(
    env: dict[str, Path], stopping: type[_StoppingOrchestrator],
    capsys: pytest.CaptureFixture[str],
) -> None:
    import signal

    before = signal.getsignal(signal.SIGTERM)
    stopping.raise_in_run = lambda: signal.raise_signal(signal.SIGTERM)  # type: ignore[assignment]
    assert _run(env) == 4
    assert stopping.finalized[0][0] == "INTERRUPTED"
    assert "SIGTERM" in stopping.finalized[0][1]
    assert signal.getsignal(signal.SIGTERM) == before


def test_an_orchestrator_without_finalize_still_exits_4(
    env: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[str] = []

    class _Old(_StubOrchestrator):
        def run(self, user_prompt: str | None = None) -> Any:
            self.ctx.current_state = PipelineState.WRITING
            raise KeyboardInterrupt

        def _log(self, agent: str, message: str) -> None:
            calls.append(message)

        def _write_cost_summary(self) -> None:
            calls.append("cost")

    monkeypatch.setattr(main_mod, "Orchestrator", _Old)
    monkeypatch.setattr(main_mod, "check_run_prerequisites", lambda *a, **k: [])
    assert _run(env) == 4
    assert calls[-1] == "cost" and "INTERRUPTED during WRITING" in calls[0]
    assert "Run interrupted during WRITING" in capsys.readouterr().out


def test_a_second_ctrl_c_while_saving_still_exits_4(
    env: dict[str, Path], stopping: type[_StoppingOrchestrator],
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    def _finalize(self: Any, code: str, message: str) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(_StoppingOrchestrator, "finalize_interrupted", _finalize)
    stopping.raise_in_run = KeyboardInterrupt()
    assert _run(env) == 4
    assert "Stopped again while saving" in capsys.readouterr().err


def test_ctrl_c_before_the_run_starts(
    env: dict[str, Path], no_orchestrator: None, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def _stop(*a: Any, **k: Any) -> list:
        raise KeyboardInterrupt

    monkeypatch.setattr(main_mod, "check_run_prerequisites", _stop)
    assert _run(env) == 4
    assert "before the run started" in capsys.readouterr().err


def test_a_finished_run_folder_is_refused_without_offering_resume(
    env: dict[str, Path], no_orchestrator: None, capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = env["root"] / "done"
    _checkpoint(run_dir, current_state="COMPLETED")
    assert main(["--config", str(env["config"]), "--output-dir", str(run_dir)]) == 1
    err = capsys.readouterr().err
    assert "finished run (COMPLETED)" in err and "--resume" not in err


def test_the_runs_own_spec_in_the_folder_is_not_an_earlier_run(
    env: dict[str, Path], stub: type[_StubOrchestrator],
) -> None:
    run_dir = env["root"] / "study"
    run_dir.mkdir()
    spec = run_dir / "research_spec.json"
    spec.write_text((FIXTURES / "spec_x1mtheff_x4college.json").read_text(
        encoding="utf-8"), encoding="utf-8")
    assert main(["--config", str(env["config"]), "--output-dir", str(run_dir),
                 "--research-spec", str(spec)]) == 0
    assert spec.exists()


def test_resuming_an_abort_judges_the_data_need_by_the_retried_stage(
    env: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = env["root"] / "aborted"
    _checkpoint(run_dir, current_state="ABORTED", abort_info={
        "stage": "WRITING", "code": "NETWORK", "message": "x", "resumable": True})
    plan = main_mod._plan_run(main_mod._build_parser().parse_args(
        ["--config", str(env["config"]), "--output-dir", str(run_dir), "--resume"]))
    assert plan.retry_stage == "WRITING"
    monkeypatch.setattr(main_mod, "check_run_prerequisites", lambda *a, **k: [
        Finding("DATA_MISSING", "fail", "The data file was not found.", "Get it.")])
    assert [f.severity for f in main_mod._preflight(plan)] == ["warn"]


@pytest.mark.parametrize("prompt, suggested", [
    ("Does taking algebra in 9th grade cause higher college enrollment?", "causal_soo"),
    ("For whom does counseling raise college enrollment?", "causal_itr"),
    ("Is the math self-efficacy scale invariant across sex (DIF)?", "psychometrics"),
    ("How reliable is the science identity scale?", "psychometrics"),
])
def test_prompt_notices_name_the_study_type_that_fits(
    env: dict[str, Path], no_orchestrator: None, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], prompt: str, suggested: str,
) -> None:
    monkeypatch.setattr(main_mod, "check_run_prerequisites", lambda *a, **k: [])
    assert main(["--dry-run", "--config", str(env["config"]), "--prompt", prompt]) == 0
    err = capsys.readouterr().err
    assert f"locked {suggested} spec" in err and "runs/fixtures/" in err


def test_because_is_not_a_causal_question() -> None:
    assert main_mod._prompt_intent("Because of low attendance, who drops out?") == "prediction"
