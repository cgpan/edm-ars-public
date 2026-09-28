"""Shared support for the run/live-view/results tests.

Fixtures that build synthetic run folders for every way a run can end.
The tests run against the real sibling modules (``edmars.paths``,
``proc``, ``secrets``, ``ui``, ``model``, ``settings``) and patch
behaviour (spawning, keyring) explicitly.

Nothing here touches the network, the real keyring or the real home
directory: ``EDMARS_HOME`` points at a temporary folder.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def run_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An isolated EDMARS_HOME; no real keyring, no network."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("EDMARS_HOME", str(home))
    monkeypatch.setenv("EDMARS_APP_ROOT", str(REPO_ROOT))
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.null.Keyring")

    def _no_network(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("network access is disabled in the CLI tests")

    try:
        import requests

        monkeypatch.setattr(requests.sessions.Session, "request", _no_network)
    except ImportError:  # pragma: no cover
        pass
    from edmars import ui

    # Plain output for assertions; monkeypatch restores the real function,
    # so no global display state leaks into other tests.
    monkeypatch.setattr(ui, "is_plain", lambda: True)
    return home


def alive_pid() -> tuple[int, float]:
    """This test process: a pid that is certainly alive, with its start time."""
    import psutil

    return os.getpid(), psutil.Process().create_time()


def dead_pid() -> tuple[int, float]:
    """A pid whose recorded start time does not match: treated as gone."""
    return os.getpid(), 1.0


# ---------------------------------------------------------------------------
# Synthetic run folders
# ---------------------------------------------------------------------------

T0 = "2026-09-25T13:00:00"


def ts(minute: int, second: int = 0) -> str:
    return f"2026-09-25T13:{minute:02d}:{second:02d}.000000"


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def log_lines(*entries: tuple[int, str]) -> str:
    return "".join(f"{ts(m)} [Orchestrator] {msg}\n" for m, msg in entries)


FULL_LOG = log_lines(
    (0, "Code executor: SubprocessExecutor"),
    (0, "Starting FORMULATING stage"),
    (1, "FORMULATING stage complete"),
    (1, "Starting ENGINEERING stage"),
    (4, "ENGINEERING stage complete"),
    (4, "Starting ANALYZING stage"),
    (12, "ANALYZING stage complete"),
    (12, "Starting CRITIQUING stage (cycle 0)"),
    (14, "Critic verdict: PASS → proceeding to WRITING"),
    (14, "Starting WRITING stage"),
    (15, "Compiling paper.tex (pdflatex → bibtex → pdflatex → pdflatex)"),
    (17, "LaTeX compilation succeeded → paper.pdf written"),
    (17, "WRITING stage complete → VERIFYING"),
    (17, "Starting VERIFYING stage (invariant battery)"),
    (17, "Invariant battery: 0 critical, 0 major, 0 minor (none)"),
    (17, "VERIFYING stage complete → COMPLETED (clean)"),
    (17, "Run cost: $0.0286 over 9 LLM calls (100 in / 50 out; 10 cached) -> run_cost.json"),
)

PREDICTION_RESULTS = {
    "best_model": "XGBoost",
    "best_metric_value": 0.781,
    "primary_metric": "AUC",
    "all_models": {
        "LogisticRegression": {"auc": 0.74, "auc_ci_lower": 0.72, "auc_ci_upper": 0.76},
        "XGBoost": {"auc": 0.781, "auc_ci_lower": 0.762, "auc_ci_upper": 0.80},
    },
}

DATA_REPORT = {
    "dataset": "hsls09_public", "original_n": 23503, "analytic_n": 17335,
    "n_train": 13868, "n_test": 3467, "n_predictors_encoded": 42, "validation_passed": True,
}


def make_run(
    root: Path,
    name: str = "2026-09-25_1300_study_abcd",
    *,
    task_type: str = "prediction",
    question: str = "Which ninth-graders are at risk of not attending college?",
    pid: tuple[int, float] | None = None,
    log: str | None = FULL_LOG,
    events: list[dict[str, Any]] | None = None,
    status: dict[str, Any] | None = None,
    checkpoint: dict[str, Any] | None = None,
    results: dict[str, Any] | None = None,
    data_report: dict[str, Any] | None = DATA_REPORT,
    review: dict[str, Any] | None = None,
    invariants: dict[str, Any] | None = None,
    gate_summary: dict[str, Any] | None = None,
    pdf: bool = True,
    runner: bool = True,
    review_gate_enabled: bool = False,
    extra: dict[str, str] | None = None,
    study: dict[str, Any] | None = None,
) -> Path:
    run = root / name
    run.mkdir(parents=True, exist_ok=True)
    if runner:
        info: dict[str, Any] = {
            "schema": 1,
            "argv": [sys.executable, "-m", "src.main", "--config", str(run / "run_config.yaml"),
                     "--output-dir", str(run), "--dataset", "hsls09_public", "--prompt", question],
            "pid": pid[0] if pid else None,
            "create_time": pid[1] if pid else None,
            "started_at": "2026-09-25T13:00:00Z",
            "app_root": str(REPO_ROOT),
            "study": {"task_type": task_type, "dataset": "hsls09_public",
                      "research_question": question, "provider": "deepseek",
                      "review": review_gate_enabled, **(study or {})},
        }
        write_json(run / "runner.json", info)
        (run / "run_config.yaml").write_text(
            "llm_provider: deepseek\n"
            f"pipeline: {{task_type: {task_type}, max_revision_cycles: 2}}\n"
            f"review_gate: {{enabled: {'true' if review_gate_enabled else 'false'}, venue: EDM}}\n"
            "pricing:\n  per_million_tokens:\n    deepseek-v4-pro: {input: 0.28, cached_input: 0.028, output: 0.42}\n",
            encoding="utf-8",
        )
    if log is not None:
        (run / "pipeline.log").write_text(log, encoding="utf-8")
    if events is not None:
        (run / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    if status is not None:
        write_json(run / "run_status.json", status)
    if checkpoint is not None:
        write_json(run / "checkpoint.json", checkpoint)
    if results is not None:
        write_json(run / "results.json", results)
    if data_report is not None:
        write_json(run / "data_report.json", data_report)
    if review is not None:
        write_json(run / "review_report.json", review)
    if invariants is not None:
        write_json(run / "invariants.json", invariants)
    if gate_summary is not None:
        write_json(run / "lsar_review" / "gate_summary.json", gate_summary)
    if pdf:
        (run / "paper.pdf").write_bytes(b"%PDF-1.5\n")
        (run / "paper.tex").write_text("\\documentclass{article}\n", encoding="utf-8")
    for rel, text in (extra or {}).items():
        path = run / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return run


def v2_status(state: str = "COMPLETED", *, released: bool = True, reason_code: str = "CLEAN",
              counts: dict[str, int] | None = None, gate: dict[str, Any] | None = None,
              abort: dict[str, Any] | None = None, **extra: Any) -> dict[str, Any]:
    return {
        "schema": 2,
        "state": state,
        "released": released,
        "reason": extra.pop("reason", "clean"),
        "reason_code": reason_code,
        "abort": abort,
        "gate": gate or {"enabled": False, "ran": False, "skip_reason": "disabled", "passed": None,
                         "score": None, "threshold": None, "advisory": None, "venue": "EDM"},
        "literature": extra.pop("literature", None),
        "invariant_counts": counts or {"critical": 0, "major": 0, "minor": 0},
        "invariant_codes": extra.pop("invariant_codes", []),
        "blocking_findings": extra.pop("blocking_findings", []),
        "critic_unverified": extra.pop("critic_unverified", False),
        "run_id": "run",
        "written_at": "2026-09-25T13:17:00Z",
        **extra,
    }


def invariants_file(findings: list[tuple[str, str]]) -> dict[str, Any]:
    counts = {"critical": 0, "major": 0, "minor": 0}
    rows = []
    for code, severity in findings:
        counts[severity] += 1
        rows.append({"code": code, "severity": severity, "message": f"{code} evidence <here>"})
    return {"enabled": True, "n_findings": len(rows), "counts": counts,
            "codes": sorted({c for c, _ in findings}), "findings": rows}


def event(seq: int, etype: str, minute: int, *, stage: str | None = None, cycle: int | None = None,
          plain: str | None = None, agent: str | None = None, **data: Any) -> dict[str, Any]:
    return {"v": 1, "seq": seq, "ts": f"2026-09-25T13:{minute:02d}:00.000Z", "run_id": "run",
            "type": etype, "stage": stage, "cycle": cycle, "agent": agent, "plain": plain, "data": data}
