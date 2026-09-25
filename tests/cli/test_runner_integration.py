"""How the runner uses the other edmars modules: LSAR's gate config, the
TeX that setup found, a local server's address, and the pipeline's own
pre-flight (``python -m src.main --dry-run``)."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from edmars import proc, runner, toolchain
from edmars import secrets as edsecrets
from edmars.model import StudyPlan


@pytest.fixture
def settings(run_home: Path, tmp_path: Path) -> dict[str, Any]:
    return {
        "schema": 1,
        "studies_dir": str(tmp_path / "studies"),
        "provider": "deepseek",
        "provider_base_url": None,
        "models": {},
        "latex": {"mode": "system", "pdflatex": None},
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


def _plan(**kw: Any) -> StudyPlan:
    base: dict[str, Any] = dict(task_type="prediction", dataset="hsls09_public",
                                research_question="Which ninth-graders are at risk of not attending college?")
    base.update(kw)
    return StudyPlan(**base)


def test_the_gate_uses_lsars_spelling_of_the_venue(settings: dict[str, Any], lsar_home: Path) -> None:
    settings["lsar"].update(enabled=True, home=str(lsar_home))
    cfg = runner.build_effective_config(settings, _plan(review=True, venue="AERA Open"))
    assert cfg["review_gate"]["enabled"] is True
    assert cfg["review_gate"]["venue"] == "AERA_OPEN"


def test_tinytex_found_off_path_goes_first_on_the_studys_path(settings: dict[str, Any], tmp_path: Path,
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    tex_bin = tmp_path / "TinyTeX" / "bin" / "windows"
    monkeypatch.setattr(toolchain, "latex_bin_dir", lambda s=None: str(tex_bin))
    env = runner.child_env(settings, provider="deepseek", review=False, run_id="r1", base_env={"PATH": "/usr/bin"})
    parts = env["PATH"].split(runner.os.pathsep)
    assert parts[1] == str(tex_bin) and parts[-1] == "/usr/bin"
    # PDFs switched off in setup: nothing is added.
    settings["latex"]["mode"] = "none"
    env = runner.child_env(settings, provider="deepseek", review=False, run_id="r1", base_env={"PATH": "/usr/bin"})
    assert str(tex_bin) not in env["PATH"]


def test_a_local_server_address_beats_a_stray_openai_base_url(settings: dict[str, Any],
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(edsecrets, "child_secrets", lambda names: {})
    settings.update(provider="local", provider_base_url="http://localhost:11434/v1")
    env = runner.child_env(settings, provider="local", review=False, run_id="r1",
                           base_env={"PATH": "", "OPENAI_BASE_URL": "https://api.openai.com/v1"})
    assert env["OPENAI_BASE_URL"] == "http://localhost:11434/v1"
    assert env["OPENAI_API_KEY"] == "local"


def test_anthropic_gets_a_model_for_the_outline_step(settings: dict[str, Any]) -> None:
    settings.update(provider="anthropic")
    cfg = runner.build_effective_config(settings, _plan())
    # The shipped top-level models block has no outline_agent, and the
    # Anthropic path has no default for it.
    assert cfg["models"]["outline_agent"] == cfg["models"]["writer"]


class DryRun:
    """Stands in for ``python -m src.main --dry-run --json-summary``."""

    def __init__(self, stdout: str, returncode: int = 1, stderr: str = "") -> None:
        self.stdout, self.returncode, self.stderr = stdout, returncode, stderr
        self.calls: list[dict[str, Any]] = []

    def __call__(self, args: list[str], *, timeout: float | None = None, env: Any = None,
                 cwd: Any = None, input: str | None = None) -> subprocess.CompletedProcess[str]:
        config = Path(args[args.index("--config") + 1])
        self.calls.append({"args": list(args), "env": dict(env or {}), "cwd": cwd,
                           "config_existed": config.is_file(),
                           "output_dir": Path(args[args.index("--output-dir") + 1])})
        return subprocess.CompletedProcess(args, self.returncode, self.stdout, self.stderr)


def test_pipeline_check_reports_the_pipelines_findings_in_edmars_words(
        settings: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    summary = {"dry_run": True, "would_start": False, "checks": [
        {"code": "KEY_MISSING", "severity": "fail", "message": "DEEPSEEK_API_KEY is not set",
         "fix": "Put the line DEEPSEEK_API_KEY=<your key> in a file named .env in the repository folder"},
        {"code": "DATA_MISSING", "severity": "fail", "message": "The data file was not found", "fix": "..."},
        {"code": "LATEX_MISSING", "severity": "warn", "message": "pdflatex was not found", "fix": "..."},
    ]}
    fake = DryRun(json.dumps(summary) + "\n")
    monkeypatch.setattr(proc, "run", fake)
    checks = runner.pipeline_check(settings, _plan())
    assert [(c.name, c.status) for c in checks] == [
        ("AI service key", "fail"), ("Dataset file", "fail"), ("PDF typesetting", "warn")]
    assert checks[0].fix == "Run `edmars setup ai` to add the key."  # never ".env in the repository folder"
    assert checks[1].fix and "edmars data install hsls09_public" in checks[1].fix
    call = fake.calls[0]
    assert call["args"][1:3] == ["-m", "src.main"]
    assert "--dry-run" in call["args"] and "--json-summary" in call["args"]
    assert call["config_existed"]  # the same effective config a launch writes
    assert not call["output_dir"].exists()  # no study folder is made
    assert not any(Path(settings["studies_dir"]).glob("*"))


def test_pipeline_check_passes_and_explains_a_crash(settings: dict[str, Any],
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(proc, "run", DryRun(json.dumps({"checks": []}), returncode=0))
    assert [c.status for c in runner.pipeline_check(settings, _plan())] == ["ok"]
    secret = "sk-" + "Q" * 30
    monkeypatch.setattr(proc, "run", DryRun("", returncode=1,
                                            stderr=f"Traceback...\nImportError: bad key {secret}\n"))
    [check] = runner.pipeline_check(settings, _plan())
    assert check.status == "fail" and "ImportError" in check.detail
    assert secret not in check.detail


def test_pipeline_check_passes_the_locked_spec(settings: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    fake = DryRun(json.dumps({"checks": []}), returncode=0)
    monkeypatch.setattr(proc, "run", fake)
    spec = {"dataset": "hsls09_public", "research_question": "Does X affect Y?", "task_type": "causal_soo"}
    runner.pipeline_check(settings, _plan(task_type="causal_soo", spec=spec))
    args = fake.calls[0]["args"]
    assert Path(args[args.index("--research-spec") + 1]).name == "research_spec.locked.json"
    assert args[args.index("--prompt") + 1] == "Does X affect Y?"
