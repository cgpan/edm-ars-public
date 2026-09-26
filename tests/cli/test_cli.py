"""The ``edmars`` command line, driven through typer's CliRunner.

Commands delegate to modules other branches provide (doctor, wizard,
study, runner, view, results, datasets, lsar). These tests install small
fake modules in ``sys.modules`` so they check the CLI's own behaviour
(argument handling, confirmation before spending, exit codes, error
messages) whether or not the real modules are present.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

from edmars import __version__, disclosure, paths, settings
from edmars.cli import app
from edmars.model import Check, StudyPlan

runner = CliRunner()

SPEC_COMMANDS = {
    "setup", "doctor", "new", "run", "status", "runs", "results", "stop", "resume",
    "review", "data", "explain", "privacy", "disclaimer", "version", "update", "uninstall",
}
DATA_COMMANDS = {"list", "install", "import", "verify"}


def invoke(*args: str, input: str | None = None) -> Any:
    return runner.invoke(app, list(args), input=input)


def fake_module(monkeypatch: pytest.MonkeyPatch, name: str, **attrs: Any) -> types.ModuleType:
    module = types.ModuleType(f"edmars.{name}")
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, f"edmars.{name}", module)
    return module


def accept_disclosure() -> None:
    disclosure.record_ack(settings.load())


def make_study(name: str = "2026-09-25_1402_gpa_ab12") -> Path:
    folder = paths.default_studies_dir() / name
    folder.mkdir(parents=True)
    (folder / "runner.json").write_text("{}", encoding="utf-8")
    return folder


# --- help and information --------------------------------------------------------------


def test_every_spec_command_exists() -> None:
    group = typer.main.get_command(app)
    assert SPEC_COMMANDS <= set(group.commands)  # type: ignore[attr-defined]
    assert DATA_COMMANDS <= set(group.commands["data"].commands)  # type: ignore[attr-defined]


@pytest.mark.parametrize("command", sorted(SPEC_COMMANDS))
def test_help_for_every_command(command: str) -> None:
    result = invoke(command, "--help")
    assert result.exit_code == 0, result.output
    assert "Usage: edmars " + command in result.output


@pytest.mark.parametrize("command", sorted(DATA_COMMANDS))
def test_help_for_every_data_command(command: str) -> None:
    result = invoke("data", command, "--help")
    assert result.exit_code == 0, result.output
    assert f"Usage: edmars data {command}" in result.output


def test_root_help_and_no_arguments() -> None:
    result = invoke("--help")
    assert result.exit_code == 0
    assert "edmars setup" in result.output
    result = invoke()
    assert result.exit_code == 0
    assert "First time here?" in result.output and "edmars setup" in result.output
    accept_disclosure()
    assert "First time here?" not in invoke().output


def test_version_command_and_flag() -> None:
    for args in (["version"], ["--version"]):
        result = invoke(*args)
        assert result.exit_code == 0
        assert result.output.startswith(f"edmars {__version__} (Python ")


def _flat(text: str) -> str:
    return " ".join(text.split())


def test_disclaimer_prints_the_full_text() -> None:
    result = invoke("disclaimer")
    assert result.exit_code == 0
    assert _flat(result.stdout) == _flat(disclosure.disclaimer_text())


def test_privacy_prints_the_full_text() -> None:
    result = invoke("privacy", "--plain")
    assert result.exit_code == 0
    assert _flat(result.stdout) == _flat(disclosure.privacy_text())


def test_explain_known_and_unknown_terms() -> None:
    result = invoke("explain", "propensity", "score")
    assert result.exit_code == 0
    assert "Propensity score" in result.output
    result = invoke("explain", "flibbertigibbet")
    assert result.exit_code == 1
    assert "Terms I can explain" in result.output
    listing = invoke("explain")
    assert listing.exit_code == 0
    assert "AUC" in listing.output and "edmars explain AUC" in listing.output


# --- missing parts and crashes ------------------------------------------------------------


def test_a_missing_module_is_explained(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "edmars.doctor", None)
    result = invoke("doctor")
    assert result.exit_code == 1
    assert "edmars.doctor" in result.output and "not included" in result.output
    assert "Traceback" not in result.output


def test_a_crash_becomes_a_message_and_a_redacted_log(monkeypatch: pytest.MonkeyPatch) -> None:
    def main(**_kwargs: Any) -> int:
        raise RuntimeError("provider said no to sk-fake-crashkey-0123456789")

    fake_module(monkeypatch, "doctor", main=main)
    result = invoke("doctor")
    assert result.exit_code == 1
    assert "Something went wrong" in result.output
    assert "sk-fake-crashkey-0123456789" not in result.output
    assert "Traceback" not in result.output
    log = paths.cache_dir() / "last_error.txt"
    text = log.read_text(encoding="utf-8")
    assert "RuntimeError" in text and "sk-fake-crashkey-0123456789" not in text


def test_debug_mode_shows_the_real_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    def main(**_kwargs: Any) -> int:
        raise RuntimeError("boom")

    fake_module(monkeypatch, "doctor", main=main)
    monkeypatch.setenv("EDMARS_DEBUG", "1")
    result = invoke("doctor")
    assert isinstance(result.exception, RuntimeError)


# --- delegation ----------------------------------------------------------------------------


def test_doctor_passes_flags_and_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def main(**kwargs: Any) -> int:
        calls.append(kwargs)
        return 3

    fake_module(monkeypatch, "doctor", main=main)
    assert invoke("doctor", "--deep", "--json").exit_code == 3
    assert calls[-1] == {"deep": True, "json_out": True, "bundle": False, "quick": False}
    # --quick is the installer's smoke test: installation checks only.
    invoke("doctor", "--quick", "--deep", "--bundle")
    assert calls[-1] == {"deep": False, "json_out": False, "bundle": True, "quick": True}


def test_setup_passes_section_and_options(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Any, ...]] = []

    def run_setup(section: str | None = None, *, non_interactive: bool = False,
                  options: dict | None = None) -> int:
        calls.append((section, non_interactive, options))
        return 0

    fake_module(monkeypatch, "wizard", run_setup=run_setup)
    result = invoke("setup", "ai", "--yes", "-o", "provider=deepseek", "-o", "lsar=no",
                    "--accept-disclosure")
    assert result.exit_code == 0, result.output
    assert calls[-1] == ("ai", True, {"provider": "deepseek", "lsar": False, "accept_disclosure": True})
    # --yes before the command name works too.
    assert invoke("--yes", "setup").exit_code == 0
    assert calls[-1] == (None, True, None)
    assert invoke("setup", "-o", "novalue").exit_code == 2


def _study_fakes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                 checks: list[Check] | None = None,
                 pipeline_checks: list[Check] | None = None) -> dict[str, list[Any]]:
    record: dict[str, list[Any]] = {"launch": [], "plan": []}
    plan = StudyPlan(task_type="prediction", dataset="hsls09_public", research_question="Q?", prompt="Q?")

    def plan_from_flags(settings: dict, **kwargs: Any) -> StudyPlan:
        record["plan"].append(kwargs)
        if kwargs.get("example") == "missing":
            raise ValueError("There is no example study called 'missing'.")
        return plan

    fake_module(
        monkeypatch,
        "study",
        plan_from_flags=plan_from_flags,
        preflight=lambda plan, settings: checks if checks is not None else [Check("Data", "ok", "found")],
        blocking=lambda found: any(c.status == "fail" for c in found),
        confirmation_card=lambda plan, settings: "Question: Q?",
    )

    def launch(settings: dict, plan: StudyPlan) -> Path:
        record["launch"].append(plan)
        return tmp_path / "study"

    def pipeline_check(settings: dict, plan: StudyPlan) -> list[Check]:
        record.setdefault("pipeline_check", []).append(plan)
        return list(pipeline_checks or [Check("Pipeline check", "ok", "passed")])

    fake_module(monkeypatch, "runner", launch=launch, pipeline_check=pipeline_check)
    return record


def test_run_requires_a_type() -> None:
    result = invoke("run")
    assert result.exit_code == 2
    assert "--type" in result.output


def test_run_requires_the_disclosure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    record = _study_fakes(monkeypatch, tmp_path)
    result = invoke("run", "--type", "prediction", "--prompt", "Q?", "--yes", "--no-watch")
    assert result.exit_code == 1
    assert "accept" in result.output
    assert record["launch"] == []


def test_run_starts_with_yes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    record = _study_fakes(monkeypatch, tmp_path)
    result = invoke("run", "-t", "prediction", "--prompt", "Q?", "--dataset", "hsls09_public",
                    "--yes", "--no-watch", "--accept-disclosure")
    assert result.exit_code == 0, result.output
    assert len(record["launch"]) == 1
    assert record["plan"][0]["task_type"] == "prediction"
    assert record["plan"][0]["dataset"] == "hsls09_public"
    assert "The study has started" in result.output
    assert disclosure.is_acknowledged(settings.load())


def test_run_never_spends_money_on_a_default_answer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    accept_disclosure()
    record = _study_fakes(monkeypatch, tmp_path)
    result = invoke("run", "--type", "prediction", "--prompt", "Q?", "--no-watch")
    assert result.exit_code == 1
    assert "--yes" in result.output
    assert record["launch"] == []


def test_run_stops_on_a_failed_preflight(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    accept_disclosure()
    record = _study_fakes(
        monkeypatch, tmp_path, checks=[Check("Data", "fail", "HSLS is not installed", fix="edmars data install hsls09_public")]
    )
    result = invoke("run", "--type", "prediction", "--prompt", "Q?", "--yes", "--no-watch")
    assert result.exit_code == 1
    assert "HSLS is not installed" in result.output
    assert "edmars data install hsls09_public" in result.output
    assert record["launch"] == []


def test_run_stops_when_the_pipeline_would_refuse(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    accept_disclosure()
    record = _study_fakes(
        monkeypatch, tmp_path,
        pipeline_checks=[Check("AI service key", "fail", "DEEPSEEK_API_KEY is not set",
                               fix="Run `edmars setup ai` to add the key.")],
    )
    result = invoke("run", "--type", "prediction", "--prompt", "Q?", "--yes", "--no-watch")
    assert result.exit_code == 1
    assert "DEEPSEEK_API_KEY is not set" in result.output
    assert "edmars setup ai" in result.output
    assert record["launch"] == []


def test_run_passes_paper_format_and_review(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    accept_disclosure()
    record = _study_fakes(monkeypatch, tmp_path)
    result = invoke("run", "--type", "prediction", "--prompt", "Q?", "--paper-format", "journal",
                    "--no-review", "--yes", "--no-watch")
    assert result.exit_code == 0, result.output
    assert record["launch"][0].paper_format == "journal"
    assert record["launch"][0].review is False
    # --review needs a reviewer that is set up.
    result = invoke("run", "--type", "prediction", "--prompt", "Q?", "--review", "--yes", "--no-watch")
    assert result.exit_code == 1
    assert "edmars setup reviewer" in result.output
    assert len(record["launch"]) == 1


def test_setup_named_flags_become_options(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Any, ...]] = []

    def run_setup(section: str | None = None, *, non_interactive: bool = False,
                  options: dict | None = None) -> int:
        calls.append((section, non_interactive, options))
        return 0

    fake_module(monkeypatch, "wizard", run_setup=run_setup,
                NONINTERACTIVE_OPTIONS={"provider": ("EDMARS_PROVIDER", "which AI service")})
    result = invoke("setup", "--yes", "--provider", "openai", "--key-env", "MY_KEY",
                    "--no-key-check", "--lsar-action", "skip", "--latex-action", "skip")
    assert result.exit_code == 0, result.output
    assert calls[-1] == (None, True, {"provider": "openai", "key_env": "MY_KEY", "check_keys": False,
                                      "lsar_action": "skip", "latex_action": "skip"})
    listed = invoke("setup", "--list-options")
    assert listed.exit_code == 0
    assert "provider (or EDMARS_PROVIDER): which AI service" in listed.output


def test_user_facing_errors_are_not_crash_reports(monkeypatch: pytest.MonkeyPatch) -> None:
    class Refused(RuntimeError):
        user_facing = True

    def boom(*_args: Any, **_kwargs: Any) -> list[Any]:
        raise Refused("The file is not the labelled HSLS:09 CSV.")

    source = paths.data_dir() / "some.csv"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("x", encoding="utf-8")
    fake_module(monkeypatch, "datasets", CATALOG={"hsls09_public": object()}, import_file=boom)
    result = invoke("data", "import", "hsls09_public", str(source))
    assert result.exit_code == 1
    assert "The file is not the labelled HSLS:09 CSV." in result.output
    assert "Something went wrong" not in result.output


def test_run_reports_a_bad_plan_plainly(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    accept_disclosure()
    _study_fakes(monkeypatch, tmp_path)
    result = invoke("run", "--type", "causal_soo", "--example", "missing", "--yes")
    assert result.exit_code == 1
    assert "There is no example study called 'missing'." in result.output
    assert "Traceback" not in result.output


def test_new_needs_a_terminal() -> None:
    result = invoke("new")
    assert result.exit_code == 1
    assert "edmars run --type" in result.output


def test_new_stops_before_any_question_without_an_ai_key(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from edmars import secrets as edsecrets
    from edmars import ui

    accept_disclosure()
    asked: list[Any] = []
    fake_module(monkeypatch, "study", new_study_interactive=lambda s: asked.append(s))
    monkeypatch.setattr(ui, "is_interactive", lambda: True)
    result = invoke("new")
    assert result.exit_code == 1
    assert "no DeepSeek key" in result.output and "edmars setup ai" in result.output
    assert asked == []  # not one question was asked
    edsecrets.set_secret("DEEPSEEK_API_KEY", "sk-" + "t" * 32)
    result = invoke("new")
    assert asked, result.output  # with a key, the questions start
    assert "sk-" not in result.output


def test_status_watches_then_shows_results(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    shown: list[Any] = []
    fake_module(monkeypatch, "runner", active_run=lambda: tmp_path, latest_run=lambda s: None)
    fake_module(monkeypatch, "view", watch=lambda run_dir, plain=False: 0)
    fake_module(monkeypatch, "results", show=lambda run_dir, open_=None: shown.append(run_dir) or 2)
    result = invoke("status")
    assert result.exit_code == 2
    assert shown == [tmp_path]


def test_status_when_the_user_leaves_it_running(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_module(monkeypatch, "runner", active_run=lambda: tmp_path, latest_run=lambda s: None)
    fake_module(monkeypatch, "view", watch=lambda run_dir, plain=False: 10)
    result = invoke("status")
    assert result.exit_code == 0
    assert "keeps running" in result.output


def test_status_without_any_study(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module(monkeypatch, "runner", active_run=lambda: None, latest_run=lambda s: None)
    result = invoke("status")
    assert result.exit_code == 1
    assert "no studies yet" in result.output


def test_results_finds_a_study_by_part_of_its_name(monkeypatch: pytest.MonkeyPatch) -> None:
    folder = make_study()
    make_study("2026-09-26_0900_dropout_cd34")
    calls: list[tuple[Any, ...]] = []
    fake_module(monkeypatch, "results", show=lambda run_dir, open_=None: calls.append((run_dir, open_)) or 0)
    result = invoke("results", "gpa", "--open", "pdf")
    assert result.exit_code == 0, result.output
    assert calls == [(folder, "pdf")]
    assert invoke("results", "2026-09").exit_code == 1  # ambiguous
    assert invoke("results", "nothing-like-this").exit_code == 1


def test_stop_asks_first_and_needs_yes_without_a_terminal(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    stopped: list[Path] = []
    fake_module(monkeypatch, "runner", active_run=lambda: tmp_path, stop=stopped.append)
    result = invoke("stop")
    assert result.exit_code == 1
    assert stopped == []
    result = invoke("stop", "--yes")
    assert result.exit_code == 0, result.output
    assert stopped == [tmp_path]


def test_stop_when_nothing_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module(monkeypatch, "runner", active_run=lambda: None, stop=lambda d: None)
    result = invoke("stop", "--yes")
    assert result.exit_code == 1
    assert "No study is running" in result.output


def test_resume_confirms_spending(monkeypatch: pytest.MonkeyPatch) -> None:
    accept_disclosure()
    folder = make_study()
    resumed: list[Path] = []
    fake_module(monkeypatch, "runner", resume=resumed.append, latest_run=lambda s: folder)
    assert invoke("resume").exit_code == 1
    assert resumed == []
    result = invoke("resume", "--yes", "--no-watch")
    assert result.exit_code == 0, result.output
    assert resumed == [folder]


def test_resume_warns_about_a_folder_from_outside_the_studies_folder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    accept_disclosure()
    resumed: list[Path] = []
    fake_module(monkeypatch, "runner", resume=resumed.append)
    own = make_study()
    result = invoke("resume", str(own), "--yes", "--no-watch")
    assert result.exit_code == 0, result.output
    assert "not in your studies folder" not in result.output
    shared = tmp_path / "shared" / "2026-09-25_1402_gpa_ab12"
    shared.mkdir(parents=True)
    result = invoke("resume", str(shared), "--yes", "--no-watch")
    assert result.exit_code == 0, result.output
    assert "not in your studies folder" in " ".join(result.output.split())
    assert resumed == [own.resolve(), shared.resolve()]


def test_review_adapts_to_the_lsar_function_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    accept_disclosure()
    folder = make_study()
    seen: list[tuple[Any, Any]] = []

    def review(settings: dict, run_dir: Path) -> int:
        seen.append((settings["defaults"]["venue"], run_dir))
        return 0

    fake_module(monkeypatch, "lsar", review=review)
    result = invoke("review", folder.name, "--yes")
    assert result.exit_code == 0, result.output
    assert seen == [("EDM", folder)]


def test_review_requires_the_disclosure(monkeypatch: pytest.MonkeyPatch) -> None:
    folder = make_study()
    reviewed: list[Path] = []
    fake_module(monkeypatch, "lsar", review=lambda run_dir: reviewed.append(run_dir) or 0)
    result = invoke("review", folder.name, "--yes")
    assert result.exit_code == 1
    assert "accept" in result.output
    assert reviewed == []  # the paper was not sent anywhere
    result = invoke("review", folder.name, "--yes", "--accept-disclosure")
    assert result.exit_code == 0, result.output
    assert reviewed == [folder]


def test_review_without_a_review_function(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module(monkeypatch, "lsar")
    result = invoke("review", "--yes")
    assert result.exit_code == 1
    assert "not included" in result.output


def test_runs_lists_and_prints_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    items = [{"run_dir": tmp_path / "2026-09-25_1402_gpa_ab12", "started_at": "2026-09-25 14:02",
              "label": "Ready", "question": "Does ninth-grade math identity predict GPA?"}]
    fake_module(monkeypatch, "runner", list_runs=lambda s: items)
    result = invoke("runs")
    assert result.exit_code == 0
    assert "2026-09-25_1402_gpa_ab12" in result.output and "Ready" in result.output
    result = invoke("runs", "--json")
    parsed = json.loads(result.stdout)
    assert parsed[0]["label"] == "Ready"
    fake_module(monkeypatch, "runner", list_runs=lambda s: [])
    assert "No studies yet" in invoke("runs").output


def test_runs_shows_start_times_in_local_time(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # The live view shows local times; `edmars runs` used to print the raw
    # UTC value (2026-09-25T21:02:01Z) for the same study.
    from datetime import datetime, timezone

    utc = "2026-09-25T21:02:01Z"
    local = datetime(2026, 9, 25, 21, 2, 1, tzinfo=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")
    items = [{"name": "2026-09-25_1702_gpa_ab12", "started_at": utc, "label": "Ready", "question": "Q?"}]
    fake_module(monkeypatch, "runner", list_runs=lambda s: items)
    result = invoke("runs")
    assert result.exit_code == 0
    assert local in result.output and utc not in result.output
    assert json.loads(invoke("runs", "--json").stdout)[0]["started_at"] == utc


# --- data ----------------------------------------------------------------------------------


def _dataset_fakes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[str]:
    """Stand-ins with the real edmars.datasets signatures (no network)."""
    downloads: list[str] = []
    info = types.SimpleNamespace(label="HSLS:09 public-use file", terms="Use it for research only.")
    verified: list[str] = []

    def install(name: str, settings_: dict, progress: Any = None) -> Path:
        downloads.append(name)
        if progress is not None:
            # The real HSLS:09 sequence: download the zip, check its
            # SHA-256, convert the CSV inside it to the labelled one.
            progress(50, 100, "download")
            progress(100, 100, "download")
            progress(100, 100, "verify")
            progress(100, 100, "convert")
        target = tmp_path / "hsls.csv"
        target.write_text("X1SEX\nMale\n", encoding="utf-8")
        settings.set_(settings_, f"datasets.{name}.sha256", "ab" * 32)
        settings.set_(settings_, f"datasets.{name}.verified_at", "2026-09-25T00:00:00Z")
        return target

    def accept_terms(name: str, settings_: dict) -> None:
        settings.set_(settings_, f"datasets.{name}.terms_accepted_at", "2026-09-25T00:00:00Z")

    def status(name: str, settings_: dict) -> Check:
        if settings.get(settings_, f"datasets.{name}.path"):
            return Check(name, "ok", "installed")
        return Check(name, "warn", "not downloaded yet", fix="edmars data install " + name)

    def verify(name: str, settings_: dict, progress: Any = None) -> Check:
        verified.append(name)
        return Check("HSLS:09", "ok", "labelled values found. SHA-256 abcd...")

    fake_module(
        monkeypatch,
        "datasets",
        CATALOG={"hsls09_public": info},
        install=install,
        terms_text=lambda name: info.terms,
        terms_accepted=lambda name, s: bool(settings.get(s, f"datasets.{name}.terms_accepted_at")),
        accept_terms=accept_terms,
        raw_data_dir=lambda s: tmp_path,
        expected_path=lambda name, s=None: tmp_path / "hsls.csv",
        validate_file=lambda name, path: Check("HSLS:09", "ok", "labelled values found"),
        status=status,
        verify=verify,
        import_file=lambda name, path, s: path,
    )
    return downloads


def test_data_install_requires_accepting_the_terms(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    downloads = _dataset_fakes(monkeypatch, tmp_path)
    result = invoke("data", "install", "hsls09_public")
    assert result.exit_code == 1
    assert "--accept-terms" in result.output
    assert downloads == []


def test_data_install_of_a_manual_dataset_points_to_import() -> None:
    # The real catalog: ASSISTments has no automatic download yet.
    result = invoke("data", "install", "assistments_0910", "--accept-terms")
    assert result.exit_code == 1
    assert "Downloading" not in result.output
    assert "edmars data import assistments_0910" in " ".join(result.output.split())


def test_data_install_records_the_dataset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    downloads = _dataset_fakes(monkeypatch, tmp_path)
    result = invoke("data", "install", "hsls09_public", "--accept-terms")
    assert result.exit_code == 0, result.output
    assert downloads == ["hsls09_public"]
    assert "downloaded 100%" in result.output
    assert "checked 100%" in result.output
    assert "converted 100%" in result.output
    entry = settings.load()["datasets"]["hsls09_public"]
    assert entry["path"] == str(tmp_path / "hsls.csv")
    assert entry["verified_at"].endswith("Z")
    assert entry["sha256"] == "ab" * 32  # the record datasets.install made is saved
    assert entry["terms_accepted_at"].endswith("Z")
    assert "Downloading." in result.output
    # Terms already accepted: not asked again, even without --accept-terms.
    again = invoke("data", "install", "hsls09_public")
    assert again.exit_code == 0, again.output
    # Already installed: the file is re-checked, and nothing says it downloads.
    assert "Already installed" in again.output
    assert "Downloading" not in again.output


def test_data_install_unknown_name(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _dataset_fakes(monkeypatch, tmp_path)
    result = invoke("data", "install", "pisa_2099", "--accept-terms")
    assert result.exit_code == 1
    assert "Known datasets: hsls09_public" in result.output


def test_data_list_import_and_verify(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _dataset_fakes(monkeypatch, tmp_path)
    listed = invoke("data", "list")
    assert listed.exit_code == 0
    assert "hsls09_public (HSLS:09 public-use file): not downloaded yet" in listed.output
    source = tmp_path / "mine.csv"
    source.write_text("X1SEX\nFemale\n", encoding="utf-8")
    imported = invoke("data", "import", "hsls09_public", str(source))
    assert imported.exit_code == 0, imported.output
    entry = settings.load()["datasets"]["hsls09_public"]
    assert entry["path"] == str(source)
    assert "terms_accepted_at" not in entry  # the user was never shown terms here
    verified = invoke("data", "verify")
    assert verified.exit_code == 0
    assert "labelled values found" in verified.output


# --- maintenance -------------------------------------------------------------------------------


def test_uninstall_needs_yes_without_a_terminal() -> None:
    settings.save(settings.load())
    result = invoke("uninstall")
    assert result.exit_code == 1
    assert paths.settings_path().exists()
    result = invoke("uninstall", "--yes")
    assert result.exit_code == 0, result.output
    assert not paths.settings_path().exists()


def test_a_module_returning_nothing_exits_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module(monkeypatch, "doctor", main=lambda **_kwargs: None)
    assert invoke("doctor").exit_code == 0



def test_review_of_a_study_without_a_paper_stops_before_asking_anything() -> None:
    # Before, the confirmation ("... usually takes 10-40 minutes") and the
    # notice came first, and only then "no paper PDF".
    from tests.cli._run_support import make_run, v2_status

    run = make_run(paths.default_studies_dir(), pdf=False,
                   status=v2_status("ABORTED", released=False, reason_code="ABORTED",
                                    abort={"stage": "CRITIQUING", "code": "PRE_CRITIC_ABORT",
                                           "message": "pcc_07: x", "resumable": False}))
    result = invoke("review", run.name)
    assert result.exit_code == 1
    out = " ".join(result.output.split())
    assert "stopped before its paper was written" in out and "edmars results" in out
    assert "10-40" not in out and "accept" not in out
