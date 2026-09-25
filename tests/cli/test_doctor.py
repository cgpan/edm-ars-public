"""`edmars doctor` (edmars/doctor.py): checks, output formats, support bundle.

All collaborators are fakes from ``wizard_fakes``; disk and memory figures
are pinned so the results do not depend on the machine running the tests.
"""

from __future__ import annotations

import io
import json
import shutil
import types
import zipfile
from collections import namedtuple
from pathlib import Path
from typing import Any

import pytest

from tests.cli.wizard_fakes import Fakes, KeyCheck, install_fakes

GOOD_KEY = "sk-fake-deepseek-0123456789abcdef"
S2_KEY = "s2-fake-key-0123456789"
ACK = "2026-09-25"
GB = 1024 ** 3

_Usage = namedtuple("_Usage", "total used free")


@pytest.fixture
def fx(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Fakes:
    fakes = install_fakes(monkeypatch, tmp_path)
    from edmars import doctor

    monkeypatch.setattr(doctor, "CORE_PACKAGES", ("json", "os"))
    monkeypatch.setattr(shutil, "disk_usage", lambda p: _Usage(500 * GB, 300 * GB, 200 * GB))
    set_memory(monkeypatch, 32)
    return fakes


def set_memory(monkeypatch: pytest.MonkeyPatch, gigabytes: float) -> None:
    import psutil

    monkeypatch.setattr(psutil, "virtual_memory", lambda: types.SimpleNamespace(total=int(gigabytes * GB)))


def healthy(fx: Fakes, **over: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "acknowledged": {"version": ACK, "at": "2026-09-25T10:00:00Z"},
        "provider": "deepseek",
        "studies_dir": str(fx.home / "EDM-ARS" / "studies"),
        "latex": {"mode": "system", "pdflatex": "pdflatex"},
        "setup_progress": {"last_completed_screen": "S11"},
    }
    values.update(over)
    fx.write_settings(**values)
    fx.secrets.store["DEEPSEEK_API_KEY"] = GOOD_KEY
    fx.secrets.store["SEMANTIC_SCHOLAR_API_KEY"] = S2_KEY
    fx.datasets.ready.add("hsls09_public")
    return fx.settings.load()


def by_name(checks: list[Any]) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    for chk in checks:
        out.setdefault(chk.name, []).append(chk)
    return out


def statuses(checks: list[Any], name: str) -> list[str]:
    return [c.status for c in by_name(checks).get(name, [])]


# ---------------------------------------------------------------------------
# run_checks
# ---------------------------------------------------------------------------

def test_openai_without_a_model_fails_the_check(fx: Fakes) -> None:
    from edmars.doctor import run_checks

    settings = healthy(fx, provider="openai")
    fx.secrets.store["OPENAI_API_KEY"] = GOOD_KEY
    models = by_name(run_checks(settings)).get("AI models", [])
    assert [c.status for c in models] == ["fail"]
    assert "edmars setup ai" in (models[0].fix or "")
    settings = healthy(fx, provider="openai", models={"writer": "gpt-test"})
    assert "AI models" not in by_name(run_checks(settings))


def test_healthy_setup_has_no_failures(fx: Fakes) -> None:
    from edmars.doctor import run_checks

    checks = run_checks(healthy(fx))
    failures = [(c.name, c.detail) for c in checks if c.status == "fail"]
    assert failures == []
    names = set(by_name(checks))
    for expected in ("Computer", "Python", "Python packages", "EDM-ARS files", "Settings", "Notice accepted",
                     "AI service key", "Literature search", "Dataset: HSLS:09", "LaTeX", "R",
                     "Automated reviewer", "Disk space", "Memory", "Cloud sync", "Terminal", "Running study",
                     "Docker"):
        assert expected in names, expected
    key = by_name(checks)["AI service key"][0]
    assert key.status == "ok" and "DeepSeek key found" in key.detail
    assert GOOD_KEY not in json.dumps([c.__dict__ for c in checks])
    assert statuses(checks, "Docker") == ["info"]


def test_empty_home_json_reports_what_is_missing(fx: Fakes, capsys: pytest.CaptureFixture[str]) -> None:
    from edmars.doctor import main

    assert main(json_out=True) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False and payload["schema"] == 1
    checks = {c["name"]: c for c in payload["checks"]}
    assert checks["AI service key"]["status"] == "fail"
    assert "edmars setup ai" in checks["AI service key"]["fix"]
    assert checks["Notice accepted"]["status"] == "fail"
    assert checks["Datasets"]["status"] == "fail"
    assert checks["Settings"]["status"] == "warn"
    assert payload["counts"]["fail"] >= 3


def test_json_stdout_stays_pure_even_when_a_check_prints(
        fx: Fakes, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from edmars.doctor import main

    healthy(fx)

    def noisy_docker() -> Any:
        print("noise from a check")
        return fx.toolchain.docker_info()

    monkeypatch.setattr(fx.modules["toolchain"], "docker_info", noisy_docker)
    main(json_out=True)
    captured = capsys.readouterr()
    json.loads(captured.out)
    assert "noise from a check" in captured.err


def test_deep_check_reports_a_rejected_key_and_runs_a_test_compile(fx: Fakes) -> None:
    from edmars.doctor import run_checks

    settings = healthy(fx)
    fx.providers.results[GOOD_KEY] = KeyCheck("REJECTED", "401 Unauthorized")
    checks = run_checks(settings, deep=True)
    assert statuses(checks, "AI service check") == ["fail"]
    assert "did not accept" in by_name(checks)["AI service check"][0].detail
    assert fx.toolchain.compile_calls == 1


def test_deep_check_finds_retired_models(fx: Fakes) -> None:
    from edmars.doctor import run_checks

    settings = healthy(fx, models={"writer": "deepseek-v4-flash"})
    fx.providers.retired = {"deepseek-v4-flash"}
    checks = run_checks(settings, deep=True)
    models = by_name(checks)["AI models"][0]
    assert models.status == "fail" and "deepseek-v4-flash" in models.detail


def test_deep_check_never_prints_the_key_even_when_the_service_echoes_it(
        fx: Fakes, capsys: pytest.CaptureFixture[str]) -> None:
    from edmars.doctor import main

    healthy(fx)
    fx.providers.results[GOOD_KEY] = KeyCheck("UNKNOWN", f"unexpected reply mentioning {GOOD_KEY}")
    main(deep=True, json_out=True)
    out = capsys.readouterr().out
    assert GOOD_KEY not in out
    assert "AI service check" in out


def test_no_credit_points_to_top_up(fx: Fakes) -> None:
    from edmars.doctor import run_checks

    settings = healthy(fx)
    fx.providers.results[GOOD_KEY] = KeyCheck("NO_CREDIT", "402")
    chk = by_name(run_checks(settings, deep=True))["AI service check"][0]
    assert chk.status == "fail" and "https://platform.deepseek.com/top_up" in (chk.fix or "")


def test_missing_optional_dataset_is_information_only(fx: Fakes) -> None:
    from edmars.doctor import run_checks

    checks = run_checks(healthy(fx))
    assert statuses(checks, "Dataset: ELS:2002") == ["info"]
    assert "Datasets" not in by_name(checks)


def test_an_installed_dataset_that_broke_stays_a_failure(fx: Fakes) -> None:
    from edmars.doctor import run_checks

    settings = healthy(fx, datasets={"els_2002": {"path": "gone.csv"}})
    assert statuses(run_checks(settings), "Dataset: ELS:2002") == ["fail"]


def test_no_dataset_ready_is_a_failure(fx: Fakes) -> None:
    from edmars.doctor import run_checks

    settings = healthy(fx)
    fx.datasets.ready.clear()
    assert statuses(run_checks(settings), "Datasets") == ["fail"]


def test_little_memory_is_a_warning(fx: Fakes, monkeypatch: pytest.MonkeyPatch) -> None:
    from edmars.doctor import run_checks

    set_memory(monkeypatch, 8)
    chk = by_name(run_checks(healthy(fx)))["Memory"][0]
    assert chk.status == "warn" and "16 GB" in chk.detail


def test_low_disk_space(fx: Fakes, monkeypatch: pytest.MonkeyPatch) -> None:
    from edmars.doctor import run_checks

    monkeypatch.setattr(shutil, "disk_usage", lambda p: _Usage(100 * GB, 99.5 * GB, int(0.5 * GB)))
    assert "fail" in statuses(run_checks(healthy(fx)), "Disk space")


def test_synced_studies_folder_is_a_warning(fx: Fakes, tmp_path: Path) -> None:
    from edmars.doctor import run_checks

    settings = healthy(fx, studies_dir=str(tmp_path / "OneDrive" / "EDM-ARS" / "studies"))
    sync = by_name(run_checks(settings))["Cloud sync"]
    assert [c.status for c in sync] == ["warn"] and "OneDrive" in sync[0].detail


def test_stale_and_live_run_locks(fx: Fakes) -> None:
    from edmars.doctor import run_checks

    settings = healthy(fx)
    lock = fx.paths.data_dir() / "active_run.json"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps({"pid": 424242, "run_dir": "studies/run1"}), encoding="utf-8")
    assert statuses(run_checks(settings), "Running study") == ["warn"]
    fx.proc.alive.add(424242)
    assert statuses(run_checks(settings), "Running study") == ["info"]


def test_a_crashing_check_becomes_a_warning_without_secrets(fx: Fakes) -> None:
    from edmars.doctor import run_checks

    settings = healthy(fx)
    fx.toolchain.latex_error = RuntimeError(f"boom while using {GOOD_KEY}")
    checks = run_checks(settings)
    latex = by_name(checks)["PDF typesetting"][0]
    assert latex.status == "warn" and "could not run" in latex.detail
    assert GOOD_KEY not in latex.detail
    assert "Docker" in by_name(checks)  # later checks still ran


def test_latex_turned_off_is_a_warning_not_a_failure(fx: Fakes) -> None:
    from edmars.doctor import run_checks
    from tests.cli.wizard_fakes import Check

    settings = healthy(fx, latex={"mode": "none", "pdflatex": None})
    fx.toolchain.latex = [Check("LaTeX", "fail", "pdflatex not found")]
    checks = run_checks(settings)
    assert statuses(checks, "LaTeX") == ["warn"]
    assert statuses(checks, "PDF typesetting") == ["warn"]


def test_r_is_optional(fx: Fakes) -> None:
    from edmars.doctor import run_checks

    settings = healthy(fx)
    assert statuses(run_checks(settings), "R") == ["info"]
    fx.toolchain.rscript = "Rscript"
    fx.toolchain.packages_missing = True
    assert statuses(run_checks(settings), "R packages") == ["warn"]


def test_reviewer_without_a_deepseek_key_fails(fx: Fakes) -> None:
    from edmars.doctor import run_checks

    settings = healthy(fx, provider="openai", lsar={"enabled": True, "auto_review": True})
    fx.secrets.store.pop("DEEPSEEK_API_KEY")
    fx.secrets.store["OPENAI_API_KEY"] = "sk-fake-openai-0123456789abcdef"
    checks = run_checks(settings)
    assert statuses(checks, "Reviewer key") == ["fail"]
    assert statuses(checks, "Automated reviewer") == ["fail"]  # not installed (fake)


def test_local_server_without_an_address_fails(fx: Fakes) -> None:
    from edmars.doctor import run_checks

    settings = healthy(fx, provider="local")
    assert statuses(run_checks(settings), "AI service") == ["fail"]


def test_damaged_settings_file_is_reported(fx: Fakes, capsys: pytest.CaptureFixture[str]) -> None:
    from edmars.doctor import main

    fx.paths.settings_path().write_text("- [broken\n", encoding="utf-8")
    assert main(json_out=True) == 1
    checks = json.loads(capsys.readouterr().out)["checks"]
    assert checks[0]["name"] == "Settings" and checks[0]["status"] == "fail"


def test_table_output_in_plain_mode(fx: Fakes) -> None:
    from edmars.doctor import main

    healthy(fx)
    assert main() == 0
    out = fx.ui.output
    assert "[ok] Python:" in out
    assert "Settings file:" in out


@pytest.mark.parametrize("width", [60, 80, 120])
def test_table_output_in_rich_mode(fx: Fakes, monkeypatch: pytest.MonkeyPatch, width: int) -> None:
    from rich.console import Console

    from edmars.doctor import render_checks, run_checks

    buffer = io.StringIO()
    fx.ui.plain = False
    monkeypatch.setattr(fx.modules["ui"], "console", Console(file=buffer, width=width, color_system=None))
    render_checks(run_checks(healthy(fx)), title="EDM-ARS check")
    text = buffer.getvalue()
    assert "✓" in text and "Python" in text


# ---------------------------------------------------------------------------
# Support bundle
# ---------------------------------------------------------------------------

def make_study(fx: Fakes, home_text: str) -> Path:
    run = fx.home / "EDM-ARS" / "studies" / "2026-09-25_1400_test_ab12"
    (run / "prompts" / "analyst").mkdir(parents=True)
    (run / "prompts" / "analyst" / "rendered_prompt.txt").write_text("PROMPT BODY", encoding="utf-8")
    (run / "train_X.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (run / "pipeline.log").write_text(
        f"2026-09-25 [Analyst] used {GOOD_KEY}\n"
        "2026-09-25 [Writer] header Authorization: Bearer abcdefghijklmnopqrst\n"
        "2026-09-25 [Critic] leaked sk-anotherfakesecret0123456789\n"
        f"2026-09-25 [Orchestrator] wrote {home_text}\n",
        encoding="utf-8")
    (run / "run_status.json").write_text(json.dumps({"state": "COMPLETED", "schema": 2}), encoding="utf-8")
    (run / "console.log").write_text("console line\n", encoding="utf-8")
    fx.runner.latest = run
    return run


def zip_texts(path: Path) -> dict[str, str]:
    with zipfile.ZipFile(path) as zf:
        return {name: zf.read(name).decode("utf-8") for name in zf.namelist()}


def test_bundle_contains_only_the_listed_files_and_no_secrets(fx: Fakes, tmp_path: Path) -> None:
    from edmars.doctor import make_bundle

    healthy(fx, author={"name": "Ada Lovelace", "affiliation": "Example University"},
            literature={"semantic_scholar_key_set": True, "crossref_mailto": "ada@example.org"})
    home_text = str(Path.home() / "EDM-ARS" / "studies" / "x")
    make_study(fx, home_text)

    bundle = make_bundle(tmp_path / "out")
    assert bundle.parent == tmp_path / "out" and bundle.suffix == ".zip"
    texts = zip_texts(bundle)
    assert set(texts) == {"doctor.json", "settings.yaml", "versions.json", "README.txt",
                          "last_study/pipeline.log", "last_study/run_status.json", "last_study/console.log"}
    everything = "\n".join(texts.values())
    for secret in (GOOD_KEY, S2_KEY, "abcdefghijklmnopqrst", "sk-anotherfakesecret0123456789"):
        assert secret not in everything, secret
    assert "PROMPT BODY" not in everything
    assert str(Path.home()) not in everything
    assert "Ada Lovelace" not in everything and "ada@example.org" not in everything
    assert "<set>" in texts["settings.yaml"]
    json.loads(texts["doctor.json"])
    assert json.loads(texts["versions.json"])["python"]
    listed = fx.ui.output
    assert "last_study/pipeline.log" in listed and "doctor.json" in listed


def test_bundle_without_study_logs(fx: Fakes, tmp_path: Path) -> None:
    from edmars.doctor import make_bundle

    healthy(fx)
    make_study(fx, "x")
    texts = zip_texts(make_bundle(tmp_path / "support.zip", include_run=False))
    assert not any(name.startswith("last_study/") for name in texts)
    assert "No study log was included" in texts["README.txt"]


def test_doctor_bundle_asks_before_including_study_logs(fx: Fakes) -> None:
    from edmars.doctor import main

    healthy(fx)
    make_study(fx, "x")
    fx.ui.script = [False]
    assert main(bundle=True) == 0
    bundles = list((fx.home / "EDM-ARS").glob("edmars-support-*.zip"))
    assert len(bundles) == 1
    assert not any(n.startswith("last_study/") for n in zip_texts(bundles[0]))
    assert "Saved a support bundle" in fx.ui.output


def test_json_with_bundle_reports_the_path(fx: Fakes, capsys: pytest.CaptureFixture[str]) -> None:
    from edmars.doctor import main

    healthy(fx)
    make_study(fx, "x")
    assert main(json_out=True, bundle=True) == 0
    payload = json.loads(capsys.readouterr().out)
    assert Path(payload["bundle"]).is_file()


def test_bundle_write_failure_is_reported(fx: Fakes, monkeypatch: pytest.MonkeyPatch) -> None:
    import edmars.doctor as doctor

    healthy(fx)

    def refuse(*_: Any, **__: Any) -> Path:
        raise PermissionError("access denied")

    monkeypatch.setattr(doctor, "make_bundle", refuse)
    fx.ui.script = [True]
    assert doctor.main(bundle=True) == 1
    assert "Couldn't write the support bundle" in fx.ui.output


# ---------------------------------------------------------------------------
# Quick mode and the setup wizard's computer check
# ---------------------------------------------------------------------------

def test_quick_mode_checks_only_the_installation(fx: Fakes, capsys: pytest.CaptureFixture[str]) -> None:
    from edmars.doctor import main

    # nothing is set up yet, and that must not fail the installer's smoke test
    assert main(quick=True, json_out=True) == 0
    names = {c["name"] for c in json.loads(capsys.readouterr().out)["checks"]}
    assert names == {"Computer", "Python", "Python packages", "EDM-ARS files", "Terminal"}


def test_quick_mode_fails_on_a_broken_installation(fx: Fakes, capsys: pytest.CaptureFixture[str]) -> None:
    from edmars.doctor import main

    (fx.app / "config.yaml").unlink()
    assert main(quick=True, json_out=True) == 1
    checks = {c["name"]: c for c in json.loads(capsys.readouterr().out)["checks"]}
    assert checks["EDM-ARS files"]["status"] == "fail"
    assert "config.yaml is missing" in checks["EDM-ARS files"]["detail"]


def test_system_checks_say_where_keys_are_never_what_they_are(
        fx: Fakes, monkeypatch: pytest.MonkeyPatch) -> None:
    from edmars.doctor import system_checks

    monkeypatch.setenv("DEEPSEEK_API_KEY", GOOD_KEY)
    fx.secrets.store["SEMANTIC_SCHOLAR_API_KEY"] = S2_KEY
    (fx.app / ".env").write_text("SOMETHING=1\n", encoding="utf-8")
    monkeypatch.setenv("MSYSTEM", "MINGW64")
    checks = system_checks(fx.settings.load())
    keys = by_name(checks)["Saved keys"][0]
    assert "DEEPSEEK_API_KEY (set in your environment)" in keys.detail
    assert "SEMANTIC_SCHOLAR_API_KEY" in keys.detail
    assert GOOD_KEY not in keys.detail and S2_KEY not in keys.detail
    assert "Old .env file" in by_name(checks)
    assert "Git Bash" in by_name(checks)["Shell"][0].detail
