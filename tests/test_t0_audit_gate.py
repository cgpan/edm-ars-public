"""The V1 gate must reach the EXIT CODE, not just the printed banner.

`scripts/audit_feasibility.py` originally returned non-zero only on false
kills, so a missed mutant printed "V1 GATE: FAIL" and still exited 0 — CI
wired to the exit code would have waved through a regression that dulled
the screen's teeth. The spec (docs/v5_arc_t_spec.md section 6, V1) makes
BOTH conditions blocking.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "audit_feasibility.py"


def _load_audit_module():
    """Import the script as a module.

    It must be registered in sys.modules BEFORE exec: @dataclass resolves
    its own module from sys.modules, and an unregistered module makes the
    decorator raise AttributeError on None.
    """
    spec = importlib.util.spec_from_file_location("_audit_feasibility", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["_audit_feasibility"] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop("_audit_feasibility", None)
    return module


@pytest.fixture(scope="module")
def audit():
    return _load_audit_module()


class TestGateExitCodes:
    def _exit_code(self, audit, monkeypatch, result: dict) -> int:
        monkeypatch.setattr(audit, "run_audit", lambda *a, **kw: result)
        monkeypatch.setattr(audit, "render", lambda r: "", raising=False)
        monkeypatch.setattr(sys, "argv", ["audit_feasibility.py"])
        return audit.main()

    def test_clean_audit_exits_zero(self, audit, monkeypatch) -> None:
        assert self._exit_code(
            audit, monkeypatch, {"false_kills": [], "mutants_missed": []}
        ) == 0

    def test_false_kill_exits_one(self, audit, monkeypatch) -> None:
        """A destroyed research question — the worse failure."""
        assert self._exit_code(
            audit, monkeypatch,
            {"false_kills": [{"spec": "x", "codes": ["F-VAR-ABSENT"]}],
             "mutants_missed": []},
        ) == 1

    def test_missed_mutant_exits_two(self, audit, monkeypatch) -> None:
        """The screen's teeth are dulling — blocking, distinct code."""
        assert self._exit_code(
            audit, monkeypatch,
            {"false_kills": [], "mutants_missed": [{"code": "F-DEAD-VARIABLE"}]},
        ) == 2

    def test_false_kill_takes_precedence(self, audit, monkeypatch) -> None:
        assert self._exit_code(
            audit, monkeypatch,
            {"false_kills": [{"spec": "x", "codes": ["F-VAR-ABSENT"]}],
             "mutants_missed": [{"code": "F-DEAD-VARIABLE"}]},
        ) == 1

    def test_result_key_name_matches_the_producer(self, audit) -> None:
        """Guards the exact bug I introduced: the consumer read
        `missed_mutants` while the producer writes `mutants_missed`, so the
        check silently never fired."""
        source = SCRIPT.read_text(encoding="utf-8")
        assert "missed_mutants" not in source, (
            "key drift: the audit result uses 'mutants_missed'"
        )
        assert source.count("mutants_missed") >= 2


class TestUnresolvableRegistryNeverKills:
    """KILL is for facts established, never for facts we could not look up.

    Live regression (2026-07-11): 6 of 26 real archived specs were killed by
    F-NO-PROTECTED-ATTRS because their `dataset` field is absent, so the
    registry could not load, so `var_map` was empty, so every protected
    attribute looked missing. The evidence string read literally
    "None.yaml declares no variable with protected_attribute: true" — the
    check reporting its own failure to load as a property of the dataset.
    ELS demonstrably HAS protected attributes; one of the killed runs
    shipped a fairness analysis and scored 6.6.
    """

    def _spec(self) -> dict:
        return {
            "research_question": "Do college enrollment gaps differ by sex and race?",
            "outcome_variable": "F2EVERAT",
            "predictor_set": [{"variable": "BYTXMSTD", "rationale": "prior achievement"}],
            "subgroup_analyses": ["BYSEX", "BYRACE"],
        }

    def test_absent_dataset_warns_instead_of_killing(self) -> None:
        from src.ideation.feasibility import screen

        report = screen(self._spec())  # no `dataset` key at all
        checks = {c.code: c for c in report.checks}
        pa = checks.get("F-NO-PROTECTED-ATTRS")
        assert pa is not None
        assert pa.status == "WARN", (
            f"unresolvable registry must WARN, not {pa.status}: {pa.evidence}"
        )
        assert report.verdict != "KILL"

    def test_evidence_names_the_uncertainty_not_a_false_fact(self) -> None:
        from src.ideation.feasibility import screen

        pa = {c.code: c for c in screen(self._spec()).checks}["F-NO-PROTECTED-ATTRS"]
        assert "cannot establish absence" in pa.evidence
        assert "None.yaml" not in pa.evidence, (
            "must not report a failed registry load as a dataset property"
        )

    def test_a_real_dataset_with_protected_attrs_still_passes(self) -> None:
        from src.ideation.feasibility import screen

        report = screen(dict(self._spec(), dataset="els_2002", task_type="prediction"))
        pa = {c.code: c for c in report.checks}["F-NO-PROTECTED-ATTRS"]
        assert pa.status == "OK", pa.evidence

    def test_genuine_absence_still_kills(self) -> None:
        """The check must keep its teeth where the fact IS established."""
        from src.ideation.feasibility import screen

        spec = dict(
            self._spec(), dataset="assistments_0910", task_type="prediction",
            subgroup_analyses=["gender"],
        )
        pa = {c.code: c for c in screen(spec).checks}["F-NO-PROTECTED-ATTRS"]
        assert pa.status == "KILL", (
            "ASSISTments genuinely carries zero protected attributes; an "
            f"equity question there is infeasible. Got {pa.status}."
        )
