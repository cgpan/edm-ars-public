"""How a run ends, and what it leaves behind when it does.

Covers the orchestrator side of the release-honesty and resilience
repairs: run_status.json at every terminal state (B3), a gate that did not
run reported as such (B2), no PDF without a pdflatex log (B1), abort codes,
resuming an aborted or interrupted run (D3), checkpoint atomicity and
identity (D1, D2), the live event stream (D8) and the literature
status (E9).

Every test stubs the agents and the LaTeX compile, so nothing here calls a
provider or needs TeX installed.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import pytest

from src.config import load_config
from src.context import PipelineContext, PipelineState
from src.errors import ProviderError
from src.orchestrator import CheckpointCorruptError, Orchestrator
from src.pre_critic_checks import CheckFailure, PreCriticResult
from tests.test_end_to_end import (
    _ABORT_REVIEW,
    _DATA_REPORT,
    _PASS_REVIEW,
    _analyst_stub,
    _critic_stub,
    _de_stub,
    _pf_stub,
    _writer_stub,
)

CONFIG_PATH = str(Path(__file__).parent.parent / "config.yaml")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _fake_compile_ok(output_dir: str, *_a: Any, **_k: Any) -> dict:
    Path(output_dir, "paper.pdf").write_bytes(b"%PDF-1.5 stub")
    Path(output_dir, "paper.log").write_text(
        "This is pdfTeX, Version 3.14\nOutput written on paper.pdf (1 page).\n",
        encoding="utf-8",
    )
    return {
        "success": True,
        "steps": [
            {"cmd": "pdflatex -interaction=nonstopmode paper.tex",
             "returncode": 0, "stdout": "", "stderr": ""}
        ],
    }


def _fake_compile_no_pdflatex(output_dir: str, *_a: Any, **_k: Any) -> dict:
    """What compile_latex returns when pdflatex is not installed: rc -1,
    and nothing at all written to the run directory."""
    return {
        "success": False,
        "steps": [
            {"cmd": "pdflatex -interaction=nonstopmode paper.tex",
             "returncode": -1, "stdout": "",
             "stderr": "'pdflatex' not found - is it installed and on PATH?"}
        ],
    }


@pytest.fixture(autouse=True)
def _compile_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default: a compile that writes a clean log and a PDF."""
    monkeypatch.setattr("src.orchestrator.compile_latex", _fake_compile_ok)


def _config(tmp_path: Path, **overrides: Any) -> dict:
    cfg = copy.deepcopy(load_config(CONFIG_PATH))
    # Keep the cross-run memory inside the test's own directory.
    cfg.setdefault("findings_memory", {})["path"] = str(
        tmp_path / "memory" / "memory.yaml"
    )
    cfg.setdefault("review_gate", {})["enabled"] = False
    for dotted, value in overrides.items():
        node = cfg
        keys = dotted.split("__")
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
    return cfg


def _ctx(out: Path, **kw: Any) -> PipelineContext:
    return PipelineContext(
        dataset_name=kw.pop("dataset_name", "hsls09_public"),
        raw_data_path=kw.pop("raw_data_path", str(out / "no_such_raw.csv")),
        output_dir=str(out),
        max_revision_cycles=kw.pop("max_revision_cycles", 2),
        **kw,
    )


def _orch(out: Path, cfg: dict, **ctx_kw: Any) -> Orchestrator:
    return Orchestrator(_ctx(out, **ctx_kw), cfg, config_path=CONFIG_PATH)


def _wire(orch: Orchestrator, review: dict = _PASS_REVIEW) -> dict[str, int]:
    out = orch.ctx.output_dir
    calls = {"pf": 0, "de": 0, "analyst": 0, "critic": 0, "writer": 0}

    def counted(name: str, fn: Any) -> Any:
        def run(**kw: Any) -> Any:
            calls[name] += 1
            return fn(**kw)
        return run

    orch.problem_formulator.run = counted("pf", lambda **kw: _pf_stub(out, **kw))
    orch.data_engineer.run = counted("de", lambda **kw: _de_stub(out, **kw))
    orch.analyst.run = counted("analyst", lambda **kw: _analyst_stub(out, **kw))
    orch.critic.run = counted("critic", lambda **kw: _critic_stub(out, review, **kw))
    orch.writer.run = counted("writer", lambda **kw: _writer_stub(out, **kw))
    return calls


def _status(out: Path) -> dict:
    return json.loads((out / "run_status.json").read_text(encoding="utf-8"))


def _events(out: Path) -> list[dict]:
    path = out / "events.jsonl"
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _no_invariants(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate the release decision from the stub paper's own findings."""
    monkeypatch.setattr("src.invariants.run_invariants", lambda _d: [])


# ---------------------------------------------------------------------------
# B1 -- no pdflatex, no log, no PDF: the run must not be released
# ---------------------------------------------------------------------------


def test_a_run_whose_compile_never_ran_is_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.orchestrator.compile_latex", _fake_compile_no_pdflatex)
    orch = _orch(tmp_path, _config(tmp_path))
    _wire(orch)

    ctx = orch.run()

    assert ctx.current_state == PipelineState.INCOMPLETE
    assert not (tmp_path / "paper.pdf").exists()
    assert not (tmp_path / "paper.log").exists()
    status = _status(tmp_path)
    assert status["released"] is False
    assert status["state"] == "INCOMPLETE"
    assert status["blocking_findings"] == ["INV_LATEX_NO_PDF"]
    assert status["reason_code"] == "BLOCKING_FINDINGS"
    record = json.loads((tmp_path / "latex_compile.json").read_text(encoding="utf-8"))
    assert record["pdf_exists"] is False
    assert record["missing_tool"] == "pdflatex"
    assert any("pdflatex was not found" in e for e in ctx.errors)
    compile_end = [e for e in _events(tmp_path) if e["type"] == "compile.end"]
    assert compile_end and compile_end[-1]["data"]["pdf_exists"] is False
    assert compile_end[-1]["data"]["missing_tool"] == "pdflatex"
    log = (tmp_path / "pipeline.log").read_text(encoding="utf-8")
    assert "paper.pdf written" not in log


def test_a_stale_pdf_from_an_earlier_run_does_not_mask_a_missing_compile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reused output dir: the previous run's paper.pdf/paper.log must not
    stand in for a compile that never happened this time."""
    (tmp_path / "paper.pdf").write_bytes(b"%PDF-1.5 old")
    (tmp_path / "paper.log").write_text("Output written on paper.pdf\n", encoding="utf-8")
    monkeypatch.setattr("src.orchestrator.compile_latex", _fake_compile_no_pdflatex)
    orch = _orch(tmp_path, _config(tmp_path))
    _wire(orch)

    assert orch.run().current_state == PipelineState.INCOMPLETE
    assert not (tmp_path / "paper.pdf").exists()


# ---------------------------------------------------------------------------
# run_status.json v2 at every terminal state (B3)
# ---------------------------------------------------------------------------


def test_completed_run_writes_schema_2_status(tmp_path: Path) -> None:
    orch = _orch(tmp_path, _config(tmp_path))
    _wire(orch)

    assert orch.run().current_state == PipelineState.COMPLETED

    status = _status(tmp_path)
    assert status["schema"] == 2
    assert status["state"] == "COMPLETED"
    assert status["released"] is True
    assert status["abort"] is None
    assert status["run_id"] == os.path.basename(str(tmp_path))
    assert status["written_at"].endswith("+00:00")
    assert status["gate"] == {
        "enabled": False, "ran": False, "skip_reason": "disabled",
        "passed": None, "score": None, "threshold": None, "advisory": None,
        "venue": status["gate"]["venue"],
    }
    assert status["review_gate_passed"] is None
    assert status["review_gate_score"] is None
    assert status["verification"]["ran"] is True
    assert status["literature"] == {"degraded": False, "n_papers": 1, "sources": {}}


def test_critic_abort_writes_a_status_with_the_abort_block(tmp_path: Path) -> None:
    orch = _orch(tmp_path, _config(tmp_path))
    _wire(orch, review=_ABORT_REVIEW)

    assert orch.run().current_state == PipelineState.ABORTED

    status = _status(tmp_path)
    assert status["state"] == "ABORTED"
    assert status["released"] is False
    assert status["reason_code"] == "ABORTED"
    assert status["abort"]["stage"] == "CRITIQUING"
    assert status["abort"]["code"] == "CRITIC_ABORT"
    assert status["abort"]["resumable"] is False
    assert "Outcome variable found in predictor set" in status["abort"]["message"]
    end = [e for e in _events(tmp_path) if e["type"] == "run.end"]
    assert len(end) == 1
    assert end[0]["data"]["state"] == "ABORTED"
    assert end[0]["data"]["exit_code"] == 3


def test_an_abort_replaces_a_stale_released_status(tmp_path: Path) -> None:
    """A reused --output-dir used to show the previous run's verdict."""
    (tmp_path / "run_status.json").write_text(
        json.dumps({"released": True, "reason": "clean"}), encoding="utf-8"
    )
    orch = _orch(tmp_path, _config(tmp_path))
    _wire(orch, review=_ABORT_REVIEW)
    orch.run()

    status = _status(tmp_path)
    assert status["released"] is False
    assert status["state"] == "ABORTED"


def test_pre_critic_abort_is_coded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "src.orchestrator.run_pre_critic_checks",
        lambda *a, **k: PreCriticResult(
            failures=[CheckFailure("pcc_x", "critical", "results.json missing", "Analyst")]
        ),
    )
    orch = _orch(tmp_path, _config(tmp_path))
    calls = _wire(orch)

    assert orch.run().current_state == PipelineState.ABORTED
    assert calls["critic"] == 0
    abort = _status(tmp_path)["abort"]
    assert abort["code"] == "PRE_CRITIC_ABORT"
    assert "pcc_x" in abort["message"]


# ---------------------------------------------------------------------------
# Abort codes from the DataEngineer paths
# ---------------------------------------------------------------------------


def test_validation_failure_on_a_missing_raw_file_is_data_missing(tmp_path: Path) -> None:
    orch = _orch(tmp_path, _config(tmp_path))
    _wire(orch)
    failed = {
        **_DATA_REPORT,
        "validation_passed": False,
        "execution_failed": True,
        "warnings": [
            "DataEngineer code did not execute successfully; Last error: "
            "FileNotFoundError: [Errno 2] No such file or directory: 'no_such_raw.csv'"
        ],
    }
    orch.data_engineer.run = lambda **kw: failed

    orch.run()

    assert orch.ctx.abort_info["code"] == "DATA_MISSING"
    assert orch.ctx.abort_info["stage"] == "ENGINEERING"
    assert _status(tmp_path)["abort"]["resumable"] is True


def test_validation_failure_with_data_present_is_de_validation_failed(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw.csv"
    raw.write_text("a\n1\n", encoding="utf-8")
    orch = _orch(tmp_path, _config(tmp_path), raw_data_path=str(raw))
    _wire(orch)
    orch.data_engineer.run = lambda **kw: {
        **_DATA_REPORT, "validation_passed": False, "warnings": ["NaN cells remain"],
    }

    orch.run()

    assert orch.ctx.abort_info["code"] == "DE_VALIDATION_FAILED"


def test_small_sample_is_sample_too_small(tmp_path: Path) -> None:
    orch = _orch(tmp_path, _config(tmp_path))
    _wire(orch)
    orch.data_engineer.run = lambda **kw: {**_DATA_REPORT, "analytic_n": 800}

    orch.run()

    assert orch.ctx.abort_info["code"] == "SAMPLE_TOO_SMALL"
    assert orch.ctx.abort_info["resumable"] is False


def test_contract_violation_after_retry_is_data_contract_failed(tmp_path: Path) -> None:
    orch = _orch(tmp_path, _config(tmp_path))
    _wire(orch)
    orch._run_post_de_preflight = lambda: "treatment is not binary"  # type: ignore[method-assign]

    orch.run()

    assert orch.ctx.abort_info["code"] == "DATA_CONTRACT_FAILED"


def test_a_provider_error_keeps_its_code(tmp_path: Path) -> None:
    orch = _orch(tmp_path, _config(tmp_path))
    _wire(orch)

    def broke(**_kw: Any) -> dict:
        raise ProviderError("NO_CREDIT", "Insufficient Balance", provider="deepseek")

    orch.analyst.run = broke
    orch.run()

    assert orch.ctx.abort_info["code"] == "NO_CREDIT"
    assert orch.ctx.abort_info["stage"] == "ANALYZING"
    errors = [e for e in _events(tmp_path) if e["type"] == "error"]
    assert errors and errors[-1]["data"]["code"] == "NO_CREDIT"


# ---------------------------------------------------------------------------
# D3 -- resuming an aborted run
# ---------------------------------------------------------------------------


def test_resume_retries_the_stage_that_aborted(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    first = _orch(tmp_path, cfg)
    _wire(first)

    def flaky(**_kw: Any) -> dict:
        raise ProviderError("NETWORK", "connection reset")

    first.data_engineer.run = flaky
    assert first.run().current_state == PipelineState.ABORTED
    assert first.ctx.abort_info["stage"] == "ENGINEERING"

    second = _orch(tmp_path, cfg)
    assert second.ctx.current_state == PipelineState.ABORTED  # nothing yet
    calls = _wire(second)
    result = second.run()

    assert result.current_state == PipelineState.COMPLETED
    assert calls == {"pf": 0, "de": 1, "analyst": 1, "critic": 1, "writer": 1}
    assert result.abort_info is None
    assert _status(tmp_path)["state"] == "COMPLETED"
    starts = [e for e in _events(tmp_path) if e["type"] == "run.start"]
    assert starts[-1]["data"]["resumed"] is True


def test_resume_does_not_rerun_a_passed_critic_after_a_writing_abort(
    tmp_path: Path,
) -> None:
    cfg = _config(tmp_path)
    first = _orch(tmp_path, cfg)
    _wire(first)

    def boom(**_kw: Any) -> str:
        raise ProviderError("TIMEOUT", "read timed out")

    first.writer.run = boom
    assert first.run().current_state == PipelineState.ABORTED

    second = _orch(tmp_path, cfg)
    calls = _wire(second)
    assert second.run().current_state == PipelineState.COMPLETED
    assert calls["critic"] == 0
    assert calls["writer"] == 1


def test_a_non_resumable_abort_stays_aborted(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    first = _orch(tmp_path, cfg)
    _wire(first, review=_ABORT_REVIEW)
    first.run()

    second = _orch(tmp_path, cfg)
    calls = _wire(second)
    assert second.run().current_state == PipelineState.ABORTED
    assert sum(calls.values()) == 0
    assert _status(tmp_path)["abort"]["code"] == "CRITIC_ABORT"


def test_completed_run_stays_terminal_on_resume(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    first = _orch(tmp_path, cfg)
    _wire(first)
    first.run()
    before = _status(tmp_path)

    second = _orch(tmp_path, cfg)
    calls = _wire(second)
    assert second.run().current_state == PipelineState.COMPLETED
    assert sum(calls.values()) == 0
    assert _status(tmp_path)["written_at"] == before["written_at"]


# ---------------------------------------------------------------------------
# Interrupts and crashes (finalize_interrupted)
# ---------------------------------------------------------------------------


def test_ctrl_c_leaves_a_resumable_checkpoint_and_status(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    orch = _orch(tmp_path, cfg)
    _wire(orch)

    def interrupted(**_kw: Any) -> dict:
        raise KeyboardInterrupt

    orch.analyst.run = interrupted
    with pytest.raises(KeyboardInterrupt):
        orch.run()

    cp = json.loads((tmp_path / "checkpoint.json").read_text(encoding="utf-8"))
    assert cp["current_state"] == "ANALYZING"
    assert cp["abort_info"]["code"] == "INTERRUPTED"
    status = _status(tmp_path)
    assert status["state"] == "INTERRUPTED"
    assert status["reason_code"] == "INTERRUPTED"
    assert status["abort"]["resumable"] is True
    ends = [e for e in _events(tmp_path) if e["type"] == "run.end"]
    assert len(ends) == 1 and ends[0]["data"]["exit_code"] == 4
    stage_ends = [e for e in _events(tmp_path) if e["type"] == "stage.end"]
    assert stage_ends[-1]["data"]["outcome"] == "interrupted"

    # The entry point's own handler calling again is a no-op.
    orch.finalize_interrupted("INTERRUPTED", "second call")
    assert len([e for e in _events(tmp_path) if e["type"] == "run.end"]) == 1

    # And --resume picks up at ANALYZING.
    second = _orch(tmp_path, cfg)
    calls = _wire(second)
    assert second.run().current_state == PipelineState.COMPLETED
    assert calls["de"] == 0 and calls["analyst"] == 1
    assert _status(tmp_path)["state"] == "COMPLETED"


def test_an_escaped_exception_is_recorded_as_crashed(tmp_path: Path) -> None:
    orch = _orch(tmp_path, _config(tmp_path))
    _wire(orch)

    def broken() -> None:
        raise RuntimeError("verifier exploded")

    orch._run_verifying = broken  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        orch.run()

    status = _status(tmp_path)
    assert status["state"] == "ABORTED"
    assert status["abort"]["code"] == "CRASHED"
    assert status["abort"]["stage"] == "VERIFYING"
    cp = json.loads((tmp_path / "checkpoint.json").read_text(encoding="utf-8"))
    assert cp["current_state"] == "VERIFYING"
    end = [e for e in _events(tmp_path) if e["type"] == "run.end"][-1]
    assert end["data"]["exit_code"] == 5


# ---------------------------------------------------------------------------
# B2 -- the review gate: not run vs failed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "gate_enabled, result, ran, skip, passed, score, code",
    [
        (False, None, False, "disabled", None, None, "CLEAN"),
        (True, {"error": "boom", "passed": False, "cycles_used": 0},
         False, "exception: boom", None, None, "GATE_NOT_RUN"),
        # The shape the old gate (and the offline test stub) produced when
        # LSAR was missing: 0.0 that nobody scored.
        (True, {"passed": False, "cycles_used": 0, "final_score": 0.0},
         False, "unknown", None, None, "GATE_NOT_RUN"),
        (True, {"ran": False, "skip_reason": "lsar_not_found: ../LSAR",
                "passed": None, "cycles_used": 0, "final_score": None},
         False, "lsar_not_found: ../LSAR", None, None, "GATE_NOT_RUN"),
        (True, {"ran": True, "passed": False, "cycles_used": 2,
                "final_score": 5.1, "threshold_used": 6.3},
         True, None, False, 5.1, "GATE_FAILED"),
        (True, {"passed": True, "advisory_mode": True, "final_score": 4.0,
                "cycles_used": 1},
         True, None, True, 4.0, "CLEAN"),
    ],
    ids=["disabled", "init-error", "legacy-zero", "lsar-missing", "failed", "advisory"],
)
def test_gate_outcomes_are_distinguishable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gate_enabled: bool,
    result: dict | None,
    ran: bool,
    skip: str | None,
    passed: bool | None,
    score: float | None,
    code: str,
) -> None:
    _no_invariants(monkeypatch)
    orch = _orch(tmp_path, _config(tmp_path, review_gate__enabled=gate_enabled))
    orch.ctx.review_gate_result = result
    orch.ctx.current_state = PipelineState.VERIFYING

    orch._run_verifying()

    status = _status(tmp_path)
    gate = status["gate"]
    assert gate["enabled"] is gate_enabled
    assert gate["ran"] is ran
    assert gate["skip_reason"] == skip
    assert gate["passed"] == passed
    assert gate["score"] == score
    assert status["review_gate_score"] == score
    assert status["reason_code"] == code
    assert status["released"] is True  # the gate is advisory either way
    if not ran:
        assert status["review_gate_score"] is None


def test_an_unverified_paper_outranks_a_gate_that_did_not_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The headline code names the manuscript problem, not the tooling
    one; ``reason`` still lists both."""
    _no_invariants(monkeypatch)
    orch = _orch(tmp_path, _config(tmp_path, review_gate__enabled=True))
    orch.ctx.review_report = {"overall_verdict": "REVISE", "unverified": True}
    orch.ctx.review_gate_result = {
        "ran": False, "skip_reason": "no_pdf", "passed": None,
        "cycles_used": 0, "final_score": None,
    }
    orch.ctx.current_state = PipelineState.VERIFYING

    orch._run_verifying()

    status = _status(tmp_path)
    assert status["reason_code"] == "CRITIC_UNVERIFIED"
    assert "review gate did not run (no_pdf)" in status["reason"]
    assert "critic verdict was not PASS" in status["reason"]


def test_a_malformed_gate_result_still_gets_a_status(tmp_path: Path) -> None:
    orch = _orch(tmp_path, _config(tmp_path, review_gate__enabled=True))
    orch.ctx.review_gate_result = {"cycles_used": "two", "passed": False}
    assert orch._gate_block()["ran"] is False

    orch.ctx.review_report = "not a dict"  # type: ignore[assignment]
    status = orch._write_run_status({"state": "ABORTED", "released": False})
    on_disk = _status(tmp_path)
    assert on_disk["state"] == "ABORTED" and on_disk["released"] is False
    assert "status_error" in on_disk and status["schema"] == 2


def test_offline_gate_run_is_not_recorded_as_a_zero_score(tmp_path: Path) -> None:
    """End to end with the gate enabled: conftest's offline gate never
    reviews anything, and the record must say so."""
    orch = _orch(tmp_path, _config(tmp_path, review_gate__enabled=True))
    _wire(orch)
    orch.run()

    status = _status(tmp_path)
    assert status["gate"]["ran"] is False
    assert status["review_gate_score"] is None
    assert status["reason_code"] in ("GATE_NOT_RUN", "BLOCKING_FINDINGS", "VERIFICATION_NOT_RUN")
    log = (tmp_path / "pipeline.log").read_text(encoding="utf-8")
    assert "score=0.00" not in log
    assert "did NOT run" in log


# ---------------------------------------------------------------------------
# Verification that did not run is never "clean"
# ---------------------------------------------------------------------------


def _verify_only(tmp_path: Path, cfg: dict) -> Orchestrator:
    orch = _orch(tmp_path, cfg)
    orch.ctx.current_state = PipelineState.VERIFYING
    orch._run_verifying()
    return orch


def _battery_crashes(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(_d: str) -> list:
        raise RuntimeError("detector crashed")

    monkeypatch.setattr("src.invariants.run_invariants", boom)


def test_a_crashed_battery_blocks_when_blocking_is_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _battery_crashes(monkeypatch)
    orch = _verify_only(
        tmp_path, _config(tmp_path, verification__blocking_codes=["INV_LATEX_NO_PDF"])
    )
    status = _status(tmp_path)
    assert orch.ctx.current_state == PipelineState.INCOMPLETE
    assert status["released"] is False
    assert status["reason_code"] == "VERIFICATION_NOT_RUN"
    assert "detector crashed" in status["reason"]


def test_a_crashed_battery_in_advisory_mode_is_released_but_not_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _battery_crashes(monkeypatch)
    orch = _verify_only(
        tmp_path,
        _config(tmp_path, verification__blocking_codes=[], verification__blocking=False),
    )
    status = _status(tmp_path)
    assert orch.ctx.current_state == PipelineState.COMPLETED
    assert status["released"] is True
    assert status["reason_code"] == "VERIFICATION_NOT_RUN"
    assert status["reason"] != "clean"


def test_disabled_verification_is_not_clean(tmp_path: Path) -> None:
    _verify_only(tmp_path, _config(tmp_path, verification__enabled=False))
    status = _status(tmp_path)
    assert status["released"] is True
    assert status["reason_code"] == "VERIFICATION_NOT_RUN"
    assert status["verification"] == {"enabled": False, "ran": False, "error": None}
    assert status["reason"] == "verification is disabled"


# ---------------------------------------------------------------------------
# E9 -- degraded literature retrieval is visible
# ---------------------------------------------------------------------------


def test_degraded_literature_reaches_log_errors_and_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_invariants(monkeypatch)
    orch = _orch(tmp_path, _config(tmp_path))
    _wire(orch)
    out = orch.ctx.output_dir

    def pf(**_kw: Any) -> dict:
        res = _pf_stub(out)
        lit = dict(res["literature_context"])
        lit["papers"] = []
        lit["retrieval_status"] = {
            "semantic_scholar": "rate_limited", "arxiv": "ok",
            "n_papers": 0, "degraded": True,
        }
        return {**res, "literature_context": lit}

    orch.problem_formulator.run = pf
    ctx = orch.run()

    assert ctx.current_state == PipelineState.COMPLETED
    status = _status(tmp_path)
    assert status["literature"] == {
        "degraded": True, "n_papers": 0,
        "sources": {"semantic_scholar": "rate_limited", "arxiv": "ok"},
    }
    assert status["reason_code"] == "ADVISORY_FINDINGS"
    assert "literature retrieval degraded" in status["reason"]
    assert any("Literature retrieval degraded" in e for e in ctx.errors)
    assert "semantic_scholar=rate_limited" in (tmp_path / "pipeline.log").read_text(
        encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# D8 -- the event stream
# ---------------------------------------------------------------------------


def test_events_cover_the_run_and_mirror_each_log_line_once(tmp_path: Path) -> None:
    orch = _orch(tmp_path, _config(tmp_path))
    _wire(orch)
    orch.run()

    evs = _events(tmp_path)
    types = [e["type"] for e in evs]
    assert types[0] in ("log", "run.start")
    assert types[-1] == "run.end"
    assert types.count("run.start") == 1 and types.count("run.end") == 1
    started = [e["stage"] for e in evs if e["type"] == "stage.start"]
    ended = [e["stage"] for e in evs if e["type"] == "stage.end"]
    assert started == ended
    assert started == [
        "FORMULATING", "ENGINEERING", "ANALYZING", "CRITIQUING", "WRITING", "VERIFYING",
    ]
    for t in ("metric", "verdict", "compile.end", "verify.end", "gate.skipped"):
        assert t in types, t
    metrics = {e["data"]["key"] for e in evs if e["type"] == "metric"}
    assert {"analytic_n", "RMSE", "critic_score"} <= metrics
    verdict = next(e for e in evs if e["type"] == "verdict")
    assert verdict["data"]["verdict"] == "PASS"
    end = evs[-1]["data"]
    assert end["state"] == "COMPLETED" and end["exit_code"] == 0
    assert end["released"] is True

    # Exactly one "log" event per pipeline.log line, and no orchestrator
    # line mirrored a second time as an agent note.
    n_lines = len(
        (tmp_path / "pipeline.log").read_text(encoding="utf-8").splitlines()
    )
    assert types.count("log") == n_lines
    assert not [
        e for e in evs if e["type"] == "agent.note" and e["agent"] == "Orchestrator"
    ]
    # ctx.log still holds each orchestrator line once (checkpoint fallback).
    cp = json.loads((tmp_path / "checkpoint.json").read_text(encoding="utf-8"))
    msgs = [x.get("message") for x in cp["log"] if x.get("agent") == "Orchestrator"]
    assert len(msgs) == len(set(zip(msgs, range(len(msgs)))))
    assert msgs.count("ENGINEERING stage complete") == 1


def test_a_revision_cycle_reports_its_new_numbers(tmp_path: Path) -> None:
    """Re-run agents in a revision replace the sample and headline metric;
    the event stream must carry the new values, not only cycle 0's."""
    orch = _orch(tmp_path, _config(tmp_path))
    _wire(orch)

    orch._run_agent("DataEngineer")
    orch._run_agent("Analyst")

    revised = [
        e for e in _events(tmp_path) if e["type"] == "metric" and e["stage"] == "REVISING"
    ]
    assert {e["data"]["key"] for e in revised} == {"analytic_n", "RMSE"}


def test_event_sink_failure_never_breaks_a_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    orch = _orch(tmp_path, _config(tmp_path))
    _wire(orch)

    def explode(*_a: Any, **_k: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(orch.ctx.event_sink, "_update_status", explode)
    assert orch.run().current_state == PipelineState.COMPLETED


# ---------------------------------------------------------------------------
# D1 -- atomic checkpoint, readable failure on a corrupt one
# ---------------------------------------------------------------------------


def test_checkpoint_survives_a_failed_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    orch = _orch(tmp_path, _config(tmp_path))
    orch.ctx.research_spec = {"q": "good"}
    orch._save_checkpoint()
    good = (tmp_path / "checkpoint.json").read_text(encoding="utf-8")

    # 1. An unserialisable value fails before the file is touched.
    orch.ctx.research_spec = {"q": object()}
    with pytest.raises(TypeError):
        orch._save_checkpoint()
    assert (tmp_path / "checkpoint.json").read_text(encoding="utf-8") == good

    # 2. A failure mid-write (here: at fsync) leaves the old file intact
    #    and no temp file behind.
    orch.ctx.research_spec = {"q": "newer"}

    def fsync_fails(_fd: int) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "fsync", fsync_fails)
    with pytest.raises(OSError):
        orch._save_checkpoint()
    assert (tmp_path / "checkpoint.json").read_text(encoding="utf-8") == good
    assert not [p for p in os.listdir(tmp_path) if p.endswith(".tmp")]


def test_corrupt_checkpoint_raises_a_named_error(tmp_path: Path) -> None:
    (tmp_path / "checkpoint.json").write_text('{"current_state": "ANA', encoding="utf-8")
    with pytest.raises(CheckpointCorruptError) as info:
        _orch(tmp_path, _config(tmp_path))
    assert "checkpoint.json" in str(info.value)


# ---------------------------------------------------------------------------
# D2 -- the checkpoint, not the command line, says what run this is
# ---------------------------------------------------------------------------


def test_resume_adopts_the_checkpoints_identity(tmp_path: Path) -> None:
    locked = {"task_type": "causal_did", "research_question": "gap change?"}
    cp = PipelineContext(
        dataset_name="did_els_hsls_panel",
        raw_data_path=str(tmp_path / "panel.csv"),
        output_dir=str(tmp_path),
        task_type="causal_did",
        locked_research_spec=locked,
    )
    cp.current_state = PipelineState.ANALYZING
    cp.completed_stages = ["FORMULATING", "ENGINEERING"]
    cp.paper_outline = {"sections": ["intro"]}
    cp.run_start_time = "2026-09-01T10:00:00+00:00"
    cp.abort_info = None
    (tmp_path / "checkpoint.json").write_text(json.dumps(cp.to_dict()), encoding="utf-8")

    with pytest.warns(RuntimeWarning):
        orch = _orch(tmp_path, _config(tmp_path))  # CLI said hsls09_public/prediction

    assert orch.ctx.dataset_name == "did_els_hsls_panel"
    assert orch.ctx.task_type == "causal_did"
    assert orch.ctx.raw_data_path == str(tmp_path / "panel.csv")
    assert orch.ctx.locked_research_spec == locked
    assert type(orch.task_template).__name__ == "CausalDIDTemplate"
    assert orch.ctx.paper_outline == {"sections": ["intro"]}
    assert orch.ctx.run_start_time == "2026-09-01T10:00:00+00:00"
    log = (tmp_path / "pipeline.log").read_text(encoding="utf-8")
    assert "disagrees with the checkpoint" in log


def test_checkpoint_round_trip_restores_every_field(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    orch = _orch(tmp_path, cfg)
    sentinels = {
        "research_spec": {"s": 1},
        "literature_context": {"papers": [{"paperId": "p"}]},
        "retrieved_literature": {"papers": []},
        "data_report": {"analytic_n": 5000},
        "results_object": {"best_model": "X"},
        "review_report": {"overall_verdict": "PASS"},
        "paper_text": "tex",
        "paper_outline": {"o": 1},
        "review_gate_result": {"ran": False},
        "completed_stages": ["FORMULATING"],
        "revision_cycle": 1,
        "errors": ["e1"],
        "abort_info": {"stage": "ENGINEERING", "code": "NETWORK"},
        "run_start_time": "2026-09-01T00:00:00+00:00",
    }
    for k, v in sentinels.items():
        setattr(orch.ctx, k, v)
    orch.ctx.current_state = PipelineState.ENGINEERING
    orch._save_checkpoint()

    again = _orch(tmp_path, cfg)
    for k, v in sentinels.items():
        assert getattr(again.ctx, k) == v, k
    assert again.ctx.current_state == PipelineState.ENGINEERING
