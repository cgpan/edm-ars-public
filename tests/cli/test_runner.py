"""Effective config, run folders, detached launch, the lock, stop and resume."""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.cli._run_support import (  # also installs stand-ins
    REPO_ROOT,
    alive_pid,
    dead_pid,
    log_lines,
    make_run,
    v2_status,
    write_json,
)

from edmars import paths, proc, runner
from edmars import secrets as edsecrets
from edmars.model import StudyPlan
from edmars.runner import RunnerError

FAKE_KEY = "sk-" + "Z" * 32
FAKE_S2 = "s2-secret-value-1234567890"


@pytest.fixture
def settings(run_home: Path, tmp_path: Path) -> dict[str, Any]:
    return {
        "schema": 1,
        "studies_dir": str(tmp_path / "studies"),
        "provider": "deepseek",
        "provider_base_url": None,
        "models": {},
        "lsar": {"enabled": False, "auto_review": False, "home": None},
        "defaults": {"venue": "EDM", "paper_format": "conference", "budget_usd": None, "keep_awake": True},
        "author": {"name": None},
        "r": {"rscript": None},
    }


@pytest.fixture
def lsar_home(tmp_path: Path) -> Path:
    home = tmp_path / "lsar" / "LSAR-public-abc123"
    (home / "calibration").mkdir(parents=True)
    (home / "config.yaml").write_text("x: 1\n", encoding="utf-8")
    (home / "calibration" / "anchors_edm.yaml").write_text("overall_p25_full: 6.3\n", encoding="utf-8")
    return home


@pytest.fixture
def fake_keys(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    store = {"DEEPSEEK_API_KEY": FAKE_KEY, "SEMANTIC_SCHOLAR_API_KEY": FAKE_S2}
    monkeypatch.setattr(edsecrets, "child_secrets", lambda names: {n: store[n] for n in names if n in store})
    return store


class Spawner:
    """Records detached launches instead of starting processes."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.pid, self.create_time = alive_pid()

    def __call__(self, args: list[str], *, cwd: Path, env: dict[str, str], log_path: Path) -> int:
        self.calls.append({"args": list(args), "cwd": Path(cwd), "env": dict(env), "log_path": Path(log_path)})
        return self.pid


@pytest.fixture
def spawner(monkeypatch: pytest.MonkeyPatch) -> Spawner:
    sp = Spawner()
    monkeypatch.setattr(proc, "spawn_detached", sp)
    monkeypatch.setattr(proc, "keep_awake", lambda pid: None)
    return sp


def _plan(**kw: Any) -> StudyPlan:
    base: dict[str, Any] = dict(task_type="prediction", dataset="hsls09_public",
                                research_question="Which ninth-graders are at risk of not attending college?")
    base.update(kw)
    return StudyPlan(**base)


# ---------------------------------------------------------------------------
# build_effective_config
# ---------------------------------------------------------------------------


def test_effective_config_paths_sandbox_and_study(settings: dict[str, Any]) -> None:
    settings["defaults"]["budget_usd"] = 2.5
    settings["author"]["name"] = "Ada Researcher"
    cfg = runner.build_effective_config(settings, _plan(venue="JEDM", paper_format="journal"))
    assert cfg["llm_provider"] == "deepseek"
    assert cfg["sandbox"]["enabled"] is False
    for key in ("raw_data", "output_base"):
        assert Path(cfg["paths"][key]).is_absolute(), key
        assert cfg["paths"][key].endswith(os.sep)
    assert Path(cfg["findings_memory"]["path"]).is_absolute()
    assert Path(cfg["findings_memory"]["path"]).parent.parent == Path(paths.data_dir()).absolute()
    assert cfg["pipeline"]["task_type"] == "prediction"
    assert cfg["pipeline"]["cost_budget_usd"] == 2.5
    assert cfg["writer"]["venue_format"] == "journal"
    assert cfg["review_gate"]["venue"] == "JEDM"
    assert cfg["review_gate"]["enabled"] is False
    assert cfg["paper"]["authors"] == ["Ada Researcher", "EDM-ARS"]
    # shipped values survive the merge
    assert cfg["deepseek"]["models"]["critic"] == "deepseek-v4-pro"
    assert cfg["deepseek"]["models"]["revision_writer"]
    assert cfg["verification"]["blocking_codes"] == ["INV_LATEX_NO_PDF"]


def test_effective_config_with_lsar(settings: dict[str, Any], lsar_home: Path) -> None:
    settings["lsar"].update(enabled=True, home=str(lsar_home))
    cfg = runner.build_effective_config(settings, _plan(review=True))
    rg = cfg["review_gate"]
    assert rg["enabled"] is True
    assert Path(rg["lsar_project_path"]) == lsar_home.absolute()
    assert Path(rg["lsar_config_path"]) == lsar_home.absolute() / "config.yaml"
    assert Path(rg["calibration_path"]) == lsar_home.absolute() / "calibration" / "anchors_edm.yaml"
    assert rg["revision_model"] == cfg["deepseek"]["models"]["revision_writer"]
    assert "${LSAR_HOME}" not in yaml.safe_dump(cfg)


def test_review_needs_lsar_installed(settings: dict[str, Any], tmp_path: Path) -> None:
    settings["lsar"].update(enabled=True, home=str(tmp_path / "missing"))
    assert runner.build_effective_config(settings, _plan(review=True))["review_gate"]["enabled"] is False
    run = runner.prepare_run(settings, _plan(review=True))
    study = json.loads((run / "runner.json").read_text(encoding="utf-8"))["study"]
    assert study["review_requested"] is True and study["review_unavailable"] is True


def test_local_provider_goes_through_openai_block(settings: dict[str, Any], lsar_home: Path) -> None:
    settings.update(provider="local", provider_base_url="http://localhost:11434/v1",
                    models={"problem_formulator": "qwen3", "writer": "qwen3"})
    settings["lsar"].update(enabled=True, home=str(lsar_home))
    cfg = runner.build_effective_config(settings, _plan(review=True))
    assert cfg["llm_provider"] == "openai"
    assert cfg["openai"]["base_url"] == "http://localhost:11434/v1"
    assert cfg["openai"]["models"]["writer"] == "qwen3"
    assert cfg["openai"]["models"]["revision_writer"] == "qwen3"
    assert cfg["review_gate"]["enabled"] is False  # LSAR is off for local models


def test_anthropic_models_go_to_top_level(settings: dict[str, Any]) -> None:
    settings.update(provider="anthropic", models={"writer": "claude-x", "critic": "claude-y"})
    cfg = runner.build_effective_config(settings, _plan())
    assert cfg["llm_provider"] == "anthropic"
    assert cfg["models"]["writer"] == "claude-x" and cfg["models"]["critic"] == "claude-y"


def test_no_secret_is_written_anywhere(settings: dict[str, Any], lsar_home: Path, spawner: Spawner,
                                       fake_keys: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_KEY)
    settings["lsar"].update(enabled=True, home=str(lsar_home))
    run = runner.launch(settings, _plan(review=True))
    for path in run.rglob("*"):
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="replace")
            assert FAKE_KEY not in text and FAKE_S2 not in text, path
    lock = Path(paths.data_dir()) / "active_run.json"
    assert FAKE_KEY not in lock.read_text(encoding="utf-8")
    # ... but the child gets them
    env = spawner.calls[0]["env"]
    assert env["DEEPSEEK_API_KEY"] == FAKE_KEY and env["SEMANTIC_SCHOLAR_API_KEY"] == FAKE_S2


# ---------------------------------------------------------------------------
# prepare_run / launch
# ---------------------------------------------------------------------------

_NAME = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{4}_[a-z0-9-]+_[0-9a-f]{4}$")


def test_prepare_run_creates_a_fresh_folder(settings: dict[str, Any]) -> None:
    a = runner.prepare_run(settings, _plan())
    b = runner.prepare_run(settings, _plan())
    assert a != b
    for run in (a, b):
        assert _NAME.match(run.name), run.name
        assert "which-ninth-graders-are-at-risk-of-not" in run.name
        assert (run / "run_config.yaml").exists() and (run / "runner.json").exists()
        assert not (run / "research_spec.locked.json").exists()
    info = json.loads((a / "runner.json").read_text(encoding="utf-8"))
    argv = info["argv"]
    assert argv[:3] == [sys.executable, "-m", "src.main"]
    assert argv[argv.index("--config") + 1] == str(a / "run_config.yaml")
    assert argv[argv.index("--output-dir") + 1] == str(a)
    assert argv[argv.index("--dataset") + 1] == "hsls09_public"
    assert argv[argv.index("--prompt") + 1].startswith("Which ninth-graders")
    assert "--research-spec" not in argv
    assert info["pid"] is None and info["study"]["task_type"] == "prediction"


def test_locked_spec_sets_dataset_and_prompt(settings: dict[str, Any]) -> None:
    spec = {"task_type": "causal_soo", "dataset": "els_2002", "research_question": "Does X cause Y?",
            "primary_method": "M4"}
    run = runner.prepare_run(settings, _plan(task_type="causal_soo", dataset="hsls09_public", spec=spec,
                                             research_question="Does X cause Y?", experimental=True))
    assert json.loads((run / "research_spec.locked.json").read_text(encoding="utf-8")) == spec
    argv = json.loads((run / "runner.json").read_text(encoding="utf-8"))["argv"]
    assert argv[argv.index("--dataset") + 1] == "els_2002"  # the spec wins
    assert argv[argv.index("--research-spec") + 1] == str(run / "research_spec.locked.json")
    assert argv[argv.index("--prompt") + 1] == "Does X cause Y?"
    cfg = yaml.safe_load((run / "run_config.yaml").read_text(encoding="utf-8"))
    assert cfg["pipeline"]["task_type"] == "causal_soo"


def test_slugify() -> None:
    assert runner.slugify("Does SES (X1SES) predict GPA?!") == "does-ses-x1ses-predict-gpa"
    assert runner.slugify("数学の自己効力感") == ""
    assert len(runner.slugify("word " * 40)) <= 40


def test_launch_spawns_detached_child_and_takes_the_lock(settings: dict[str, Any], spawner: Spawner,
                                                         fake_keys: dict[str, str],
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTHONPATH", "/somewhere/else")
    monkeypatch.setenv("PYTHONHOME", "/elsewhere")
    run = runner.launch(settings, _plan())
    [call] = spawner.calls
    assert call["args"][:3] == [sys.executable, "-m", "src.main"]
    assert call["cwd"] == Path(paths.app_root()).absolute()
    assert call["log_path"] == run / "console.log"
    env = call["env"]
    assert env["PYTHONUTF8"] == "1" and env["PYTHONIOENCODING"] == "utf-8" and env["PYTHONNOUSERSITE"] == "1"
    assert not any(k.upper() in ("PYTHONPATH", "PYTHONHOME") for k in env)
    path_key = next(k for k in env if k.upper() == "PATH")
    assert env[path_key].split(os.pathsep)[0] == str(Path(sys.executable).parent)
    assert env["EDMARS_RUN_ID"] == run.name
    assert "LSAR_HOME" not in env
    info = json.loads((run / "runner.json").read_text(encoding="utf-8"))
    assert info["pid"] == spawner.pid and info["started_at"]
    assert runner.active_run() == run
    with pytest.raises(RunnerError, match="Another study is still running"):
        runner.launch(settings, _plan())
    assert len(spawner.calls) == 1


def test_launch_failure_releases_the_lock(settings: dict[str, Any], fake_keys: dict[str, str],
                                          monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a: Any, **k: Any) -> int:
        raise OSError(f"cannot start with key {FAKE_KEY}")

    monkeypatch.setattr(proc, "spawn_detached", boom)
    monkeypatch.setattr(edsecrets, "redact", lambda text: text.replace(FAKE_KEY, "[redacted]"))
    with pytest.raises(RunnerError) as info:
        runner.launch(settings, _plan())
    assert FAKE_KEY not in str(info.value)
    assert runner.active_run() is None


def test_stale_lock_is_cleared(run_home: Path) -> None:
    pid, _ = dead_pid()
    lock = Path(paths.data_dir()) / "active_run.json"
    write_json(lock, {"pid": pid, "create_time": 1.0, "run_dir": str(run_home / "x")})
    assert runner.active_run() is None
    assert not lock.exists()


def test_rscript_and_lsar_reach_the_child(settings: dict[str, Any], lsar_home: Path, fake_keys: dict[str, str]) -> None:
    settings["r"]["rscript"] = "C:/R/bin/Rscript.exe"
    settings["lsar"].update(enabled=True, home=str(lsar_home))
    env = runner.child_env(settings, provider="openai", review=True, run_id="r1", base_env={"PATH": "p"})
    assert env["EDM_ARS_RSCRIPT"] == "C:/R/bin/Rscript.exe"
    assert env["LSAR_HOME"] == str(lsar_home.absolute())
    assert env["DEEPSEEK_API_KEY"] == FAKE_KEY  # LSAR scoring needs DeepSeek even for OpenAI runs
    cfg = runner.build_effective_config(settings, _plan(review=True))
    assert cfg["r_bridge"]["rscript_path"] == "C:/R/bin/Rscript.exe"


# ---------------------------------------------------------------------------
# stop / resume
# ---------------------------------------------------------------------------


def test_stop_flags_and_terminates(run_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    killed: list[tuple[int, float]] = []
    monkeypatch.setattr(proc, "terminate_tree", lambda pid, grace_s=30: killed.append((pid, grace_s)))
    run = make_run(run_home / "studies", pid=alive_pid(), pdf=False, log=log_lines((0, "Starting FORMULATING stage")))
    write_json(Path(paths.data_dir()) / "active_run.json",
               {"pid": alive_pid()[0], "create_time": alive_pid()[1], "run_dir": str(run)})
    runner.stop(run)
    assert (run / "STOP").exists()
    assert killed == [(os.getpid(), 30)]
    info = json.loads((run / "runner.json").read_text(encoding="utf-8"))
    assert info["stopped_by_user"] is True
    assert runner.active_run() is None


#: A detached child that stands in for the pipeline: it uses the
#: pipeline's own stop handling and records how it ended.
_STOPPABLE_CHILD = """
import pathlib, sys, time
sys.path.insert(0, sys.argv[2])
from src.main import _stop_signals_interrupt
run = pathlib.Path(sys.argv[1])
try:
    with _stop_signals_interrupt(str(run)):
        (run / "child_ready").write_text("1", encoding="utf-8")
        while True:
            time.sleep(0.05)
except KeyboardInterrupt as exc:
    (run / "child_stopped").write_text(
        type(exc).__name__ + " " + str(getattr(exc, "by_stop_file", "")), encoding="utf-8")
"""


def test_stop_lets_a_detached_run_wind_down_instead_of_killing_it(run_home: Path) -> None:
    # On Windows no signal reaches a detached run, so before the pipeline
    # watched the STOP file every `edmars stop` waited out the grace
    # period and then killed the run without saving its state.
    import time

    import psutil

    run = make_run(run_home / "studies", pdf=False, log=log_lines((0, "Starting FORMULATING stage")))
    pid = proc.spawn_detached([sys.executable, "-c", _STOPPABLE_CHILD, str(run), str(REPO_ROOT)],
                              cwd=run, env=None, log_path=run / "child.log")
    try:
        deadline = time.monotonic() + 90
        while not (run / "child_ready").exists():
            assert time.monotonic() < deadline, (run / "child.log").read_text(errors="replace")
            assert proc.pid_alive(pid), (run / "child.log").read_text(errors="replace")
            time.sleep(0.1)
        info = json.loads((run / "runner.json").read_text(encoding="utf-8"))
        info.update({"pid": pid, "create_time": psutil.Process(pid).create_time()})
        write_json(run / "runner.json", info)

        started = time.monotonic()
        runner.stop(run)
        assert time.monotonic() - started < 20
        stopped = (run / "child_stopped").read_text(encoding="utf-8")
        if sys.platform == "win32":
            assert stopped == "_StopRequested True"  # only the STOP file can reach it
        else:
            # macOS/Linux also get SIGTERM, which usually arrives before the
            # file watcher's next look. Either way the run wound itself down.
            assert stopped in ("_StopRequested True", "_StopRequested False")
    finally:
        proc.terminate_tree(pid, grace_s=0)


def test_stop_does_not_kill_a_reused_pid(run_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    killed: list[int] = []
    monkeypatch.setattr(proc, "terminate_tree", lambda pid, grace_s=30: killed.append(pid))
    run = make_run(run_home / "studies", pid=dead_pid(), pdf=False, log=log_lines((0, "Starting FORMULATING stage")))
    with pytest.raises(RunnerError, match="not running"):
        runner.stop(run)
    assert killed == []
    assert not (run / "STOP").exists()


def _crashed_run(root: Path) -> Path:
    log = log_lines((0, "Starting FORMULATING stage"), (1, "FORMULATING stage complete"),
                    (1, "Starting ENGINEERING stage"))
    run = make_run(root, pid=dead_pid(), pdf=False, log=log,
                   checkpoint={"current_state": "ENGINEERING", "completed_stages": ["FORMULATING"],
                               "dataset_name": "hsls09_public"})
    (run / "console.log").write_text("first attempt output", encoding="utf-8")
    (run / "STOP").write_text("x", encoding="utf-8")
    return run


def test_resume_relaunches_with_resume_flag(run_home: Path, spawner: Spawner, fake_keys: dict[str, str]) -> None:
    run = _crashed_run(run_home / "studies")
    runner.resume(run)
    [call] = spawner.calls
    args = call["args"]
    assert args[-1] == "--resume" and args.count("--resume") == 1
    assert args[:3] == [sys.executable, "-m", "src.main"]
    assert args[args.index("--output-dir") + 1] == str(run)
    assert (run / "console.1.log").read_text(encoding="utf-8") == "first attempt output"
    assert not (run / "STOP").exists()
    info = json.loads((run / "runner.json").read_text(encoding="utf-8"))
    assert info["pid"] == spawner.pid and info["resumed_at"] and info["resumes"]
    assert runner.active_run() == run


def test_resume_refuses_finished_or_running(run_home: Path, spawner: Spawner) -> None:
    done = make_run(run_home / "studies", name="done", status=v2_status())
    with pytest.raises(RunnerError, match="already finished"):
        runner.resume(done)
    running = make_run(run_home / "studies", name="running", pid=alive_pid(), pdf=False,
                       log=log_lines((0, "Starting FORMULATING stage")))
    with pytest.raises(RunnerError, match="still running"):
        runner.resume(running)
    assert spawner.calls == []


def _aborted(root: Path, code: str) -> Path:
    return make_run(root, name=f"aborted_{code}", pdf=False,
                    log=log_lines((0, "Starting ENGINEERING stage"), (3, "ABORTED: ENGINEERING failed")),
                    status=v2_status("ABORTED", released=False, reason_code="ABORTED",
                                     abort={"stage": "ENGINEERING", "code": code, "message": "m",
                                            "resumable": code != "SAMPLE_TOO_SMALL"}))


def test_resume_aborted_needs_retry_stage_support(run_home: Path, spawner: Spawner,
                                                   monkeypatch: pytest.MonkeyPatch,
                                                   fake_keys: dict[str, str]) -> None:
    run = _aborted(run_home / "studies", "NO_CREDIT")
    monkeypatch.setattr(runner, "retry_stage_support", lambda root=None: (False, False))
    with pytest.raises(RunnerError, match="cannot retry a failed step"):
        runner.resume(run)
    monkeypatch.setattr(runner, "retry_stage_support", lambda root=None: (True, True))
    runner.resume(run)
    args = spawner.calls[-1]["args"]
    assert args[-3:] == ["--resume", "--retry-stage", "ENGINEERING"]


def test_resume_refuses_non_resumable_abort(run_home: Path, spawner: Spawner,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    run = _aborted(run_home / "studies", "SAMPLE_TOO_SMALL")
    monkeypatch.setattr(runner, "retry_stage_support", lambda root=None: (True, True))
    with pytest.raises(RunnerError, match="cannot be resumed"):
        runner.resume(run)
    assert spawner.calls == []


def test_resume_reopens_an_old_pre_review_stop_a_revision_can_fix(
    run_home: Path, spawner: Spawner, monkeypatch: pytest.MonkeyPatch,
    fake_keys: dict[str, str],
) -> None:
    # The Mac study: stopped as PRE_CRITIC_ABORT (resumable: false) for
    # pcc_07 before the pipeline sent such findings back for revision.
    # The pipeline now resumes it at CRITIQUING (src/errors.py
    # reopened_pre_critic_stop); `edmars resume` refused it.
    def stopped(name: str, message: str, **extra: object) -> Path:
        return make_run(run_home / "studies", name=name, pdf=False, log=None,
                        status=v2_status("ABORTED", released=False, reason_code="ABORTED",
                                         abort={"stage": "CRITIQUING", "code": "PRE_CRITIC_ABORT",
                                                "message": message, "resumable": False, **extra}))

    monkeypatch.setattr(runner, "retry_stage_support", lambda root=None: (True, True))
    runner.resume(stopped("mac", "pcc_07: The research question says 'above and beyond'"))
    assert spawner.calls[-1]["args"][-3:] == ["--resume", "--retry-stage", "CRITIQUING"]

    for run in (stopped("leak", "pcc_01: Outcome variable found in train_X.csv"),
                stopped("new", "pcc_07: x", checks=[{"check_id": "pcc_01", "revisable": False}])):
        with pytest.raises(RunnerError, match="cannot be resumed"):
            runner.resume(run)
    assert len(spawner.calls) == 1


def test_retry_stage_support_reads_the_pipeline(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    main = tmp_path / "src" / "main.py"
    main.write_text('parser.add_argument("--resume", action="store_true")\n', encoding="utf-8")
    assert runner.retry_stage_support(tmp_path) == (False, False)
    main.write_text('parser.add_argument(\n    "--retry-stage",\n    action="store_true",\n)\n'
                    'parser.add_argument("--x")\n', encoding="utf-8")
    assert runner.retry_stage_support(tmp_path) == (True, False)
    main.write_text('parser.add_argument("--retry-stage", default=None, metavar="STAGE")\n', encoding="utf-8")
    assert runner.retry_stage_support(tmp_path) == (True, True)
    # the shipped pipeline on this branch
    supported, _ = runner.retry_stage_support(REPO_ROOT)
    assert supported == ("--retry-stage" in (REPO_ROOT / "src" / "main.py").read_text(encoding="utf-8"))


def test_resume_never_runs_what_the_study_folder_names(run_home: Path, spawner: Spawner,
                                                       fake_keys: dict[str, str]) -> None:
    # A study folder can come from a colleague or a shared drive. Its
    # runner.json and run_config.yaml must not choose the program that
    # runs, the Python or Rscript it uses, the LSAR it imports, or the
    # server the keys go to.
    run = _crashed_run(run_home / "studies")
    other_program = run / "other_program.exe"
    other_program.write_bytes(b"MZ")
    info = json.loads((run / "runner.json").read_text(encoding="utf-8"))
    info["argv"] = [str(other_program), "-c", "print('hello')", "--dataset", "../../x",
                    "--prompt", "-starts with a dash"]
    info["study"]["venue"] = "../../venue"
    write_json(run / "runner.json", info)
    (run / "run_config.yaml").write_text(yaml.safe_dump({
        "llm_provider": "deepseek",
        "deepseek": {"base_url": "https://collector.example/v1"},
        "semantic_scholar": {"base_url": "https://collector.example/s2"},
        "sandbox": {"enabled": False, "python_executable": str(other_program)},
        "r_bridge": {"rscript_path": str(other_program)},
        "review_gate": {"enabled": True, "lsar_project_path": str(run), "venue": "EDM"},
        "paths": {"agent_prompts": str(run)},
    }), encoding="utf-8")

    runner.resume(run)

    args = spawner.calls[-1]["args"]
    assert args[:3] == [sys.executable, "-m", "src.main"]
    assert str(other_program) not in args and "-c" not in args
    assert args[args.index("--config") + 1] == str(run / "run_config.yaml")
    assert args[args.index("--dataset") + 1] == "hsls09_public"  # the study's, not the argv's
    assert "--prompt=-starts with a dash" in args
    cfg = yaml.safe_load((run / "run_config.yaml").read_text(encoding="utf-8"))
    shipped = yaml.safe_load((REPO_ROOT / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["deepseek"]["base_url"] == shipped["deepseek"]["base_url"]
    assert cfg["semantic_scholar"]["base_url"] == shipped["semantic_scholar"]["base_url"]
    assert cfg["sandbox"]["python_executable"] is None
    assert not cfg.get("r_bridge", {}).get("rscript_path")
    assert cfg["review_gate"]["enabled"] is False  # no LSAR set up on this computer
    assert cfg["review_gate"]["venue"] == "EDM"
    assert cfg["paths"]["agent_prompts"] == shipped["paths"]["agent_prompts"]
    # The replaced file is kept, not lost.
    assert "collector.example" in (run / "run_config.previous.yaml").read_text(encoding="utf-8")


def test_resume_ignores_a_retry_step_the_folder_made_up(run_home: Path, spawner: Spawner,
                                                         monkeypatch: pytest.MonkeyPatch,
                                                         fake_keys: dict[str, str]) -> None:
    run = _aborted(run_home / "studies", "NO_CREDIT")
    status = json.loads((run / "run_status.json").read_text(encoding="utf-8"))
    status["abort"]["stage"] = "--overwrite"
    write_json(run / "run_status.json", status)
    monkeypatch.setattr(runner, "retry_stage_support", lambda root=None: (True, True))
    runner.resume(run)
    args = spawner.calls[-1]["args"]
    assert "--overwrite" not in args
    assert args[args.index("--resume"):][:2] == ["--resume", "--retry-stage"]


# ---------------------------------------------------------------------------
# listing
# ---------------------------------------------------------------------------


def test_list_and_latest_runs(settings: dict[str, Any]) -> None:
    root = Path(settings["studies_dir"])
    older = make_run(root, name="2026-09-24_0900_old_aaaa", status=v2_status())
    info = json.loads((older / "runner.json").read_text(encoding="utf-8"))
    info["started_at"] = "2026-09-24T09:00:00Z"
    write_json(older / "runner.json", info)
    newer = make_run(root, name="2026-09-25_1300_new_bbbb", status=v2_status())
    (root / "not-a-run").mkdir()
    runs = runner.list_runs(settings)
    assert [r["name"] for r in runs] == [newer.name, older.name]
    assert runs[0]["label"] == "Ready" and runs[0]["kind"] == "ready"
    assert runs[0]["question"].startswith("Which ninth-graders")
    assert runner.latest_run(settings) == newer


def test_list_runs_on_missing_folder(settings: dict[str, Any]) -> None:
    assert runner.list_runs(settings) == []
    assert runner.latest_run(settings) is None
