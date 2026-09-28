"""The pre-flight (main package) against the providers and R bridge.

* ``describe_routing`` (providers) knows a configuration the agents will
  refuse at start-up -- llm_provider: openai with no openai.models.<stage>
  has no abort code -- so the pre-flight reports it before any spend.
* The pre-flight probes the R the run will actually use, in the
  executor's precedence: an operator's EDM_ARS_RSCRIPT beats config
  ``r_bridge.rscript_path``.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from src import preflight
from src.config import load_config

CONFIG_PATH = str(Path(__file__).parent.parent / "config.yaml")


def test_preflight_reports_an_openai_stage_with_no_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """E1 made the agents refuse llm_provider: openai without
    openai.models.<stage>; that refusal has no abort code, so the
    pre-flight must catch it before a run starts."""
    cfg = copy.deepcopy(load_config(CONFIG_PATH))
    cfg["llm_provider"] = "openai"
    cfg.pop("openai", None)
    cfg["per_stage_providers"] = {}
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder")

    findings = preflight._check_provider_keys(cfg)

    invalid = [f for f in findings if f.code == "PROVIDER_CONFIG_INVALID"]
    assert invalid, findings
    assert all(f.severity == preflight.FAIL for f in invalid)
    assert any("openai.models." in f.message for f in invalid)


def test_preflight_default_config_has_no_routing_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = copy.deepcopy(load_config(CONFIG_PATH))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-placeholder")
    findings = preflight._check_provider_keys(cfg)
    assert [f for f in findings if f.code == "PROVIDER_CONFIG_INVALID"] == []


def _capture_find_rscript(monkeypatch: pytest.MonkeyPatch) -> list:
    from src import r_bridge

    seen: list = []

    def fake_find(explicit: Any = None) -> str:
        seen.append(explicit)
        return "Rscript"

    monkeypatch.setattr(r_bridge, "find_rscript", fake_find)
    return seen


def test_preflight_probes_the_operators_rscript_before_the_config_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _capture_find_rscript(monkeypatch)
    monkeypatch.setenv("EDM_ARS_RSCRIPT", "operator-Rscript")
    cfg = {"r_bridge": {"rscript_path": "config-Rscript"}, "sandbox": {"enabled": False}}
    preflight._check_r(cfg, None, probe=False)
    # None lets find_rscript read EDM_ARS_RSCRIPT, as the executor would.
    assert seen == [None]


def test_preflight_uses_the_config_rscript_when_the_operator_set_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _capture_find_rscript(monkeypatch)
    monkeypatch.delenv("EDM_ARS_RSCRIPT", raising=False)
    cfg = {"r_bridge": {"rscript_path": "config-Rscript"}, "sandbox": {"enabled": False}}
    preflight._check_r(cfg, None, probe=False)
    assert seen == ["config-Rscript"]
