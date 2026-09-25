"""src/main.py: how a run is asked for, checked and started.

Covers the launch defects C1-C5, C7 and the resume/output-folder defects:
every test drives ``main(argv)`` in-process with a stubbed Orchestrator, so
nothing here calls a provider or runs a pipeline stage.
"""
from __future__ import annotations

import json
import os
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
    """Records the context main() built; run() ends in ``final_state``."""

    instances: list["_StubOrchestrator"] = []
    final_state: PipelineState = PipelineState.COMPLETED

    def __init__(self, ctx: Any, config: dict, config_path: str = "") -> None:
        self.ctx = ctx
        self.config = config
        self.config_path = config_path
        os.makedirs(ctx.output_dir, exist_ok=True)
        _StubOrchestrator.instances.append(self)

    def run(self, user_prompt: str | None = None) -> Any:
        self.ctx.current_state = self.final_state
        return self.ctx


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> type[_StubOrchestrator]:
    _StubOrchestrator.instances = []
    _StubOrchestrator.final_state = PipelineState.COMPLETED
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
    (env["out_base"] / "run_20260102_030405").mkdir(parents=True)
    assert main(["--config", str(env["config"])]) == 0
    assert Path(stub.instances[0].ctx.output_dir).name == "run_20260102_030405_2"


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
