"""The pipeline.log cost line says when the figure is not a measurement.

fix/wp-prov marks run_cost.json `cost_status: estimated` when a rate is
flagged unverified in config.yaml, or when a call to a time-of-day priced
model has no readable time, and `partial` when some calls have no rate.
The orchestrator's "Run cost: $X" line still read as measured.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.test_orchestrator_terminal import _config, _orch


def _payload(status: str, cost: Any = 0.0123) -> dict:
    return {"cost_usd": cost, "cost_status": status, "n_calls": 3,
            "prompt_tokens": 10, "completion_tokens": 5,
            "cached_prompt_tokens": 0}


@pytest.mark.parametrize(
    "status, expected",
    [
        ("measured", "Run cost: $0.0123 over"),
        ("estimated", "Run cost: $0.0123 (estimated: a rate is unverified) over"),
        ("partial", "Run cost: $0.0123 (lower bound: some calls have no rate) over"),
    ],
)
def test_cost_line_names_an_estimate(
    status: str, expected: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.cost.write_summary", lambda _d, _c: _payload(status))
    orch = _orch(tmp_path, _config(tmp_path))
    orch._write_cost_summary()
    log = (tmp_path / "pipeline.log").read_text(encoding="utf-8")
    assert expected in log


def test_estimate_from_calls_of_unknown_time_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rates are verified, but a call with no timestamp on a time-of-day
    priced model was charged the peak rate: the line must not blame a
    rate."""
    payload = {**_payload("estimated"), "untimed_calls": 2,
               "unverified_rate_models": []}
    monkeypatch.setattr("src.cost.write_summary", lambda _d, _c: payload)
    orch = _orch(tmp_path, _config(tmp_path))
    orch._write_cost_summary()
    log = (tmp_path / "pipeline.log").read_text(encoding="utf-8")
    assert "Run cost: $0.0123 (estimated: some calls have no time, priced at peak) over" in log


def test_unpriced_run_still_says_not_priced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.cost.write_summary",
                        lambda _d, _c: _payload("unpriced", cost=None))
    orch = _orch(tmp_path, _config(tmp_path))
    orch._write_cost_summary()
    assert "Run cost: not priced over" in (tmp_path / "pipeline.log").read_text(encoding="utf-8")
