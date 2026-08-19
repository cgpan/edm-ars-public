"""V3.0 Phase 3b.4 / B6 — --research-spec CLI flag tests.

These tests cover the load + structural-validate path added to
``src/main.py``. The smoke-test fixture at
``runs/fixtures/spec_x1mtheff_x4college.json`` is the canonical
locked spec; it must round-trip through ``load_locked_research_spec``
without warnings.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.main import load_locked_research_spec


FIXTURE_PATH = (
    Path(__file__).parent.parent
    / "runs"
    / "fixtures"
    / "spec_x1mtheff_x4college.json"
)


class TestLoadLockedResearchSpec:
    def test_loads_smoke_test_fixture(self) -> None:
        spec = load_locked_research_spec(str(FIXTURE_PATH))
        assert spec["task_type"] == "causal_soo"
        assert spec["treatment"]["variable"] == "X1MTHEFF"
        assert spec["outcome"]["variable"] == "X4EVRATNDCLG"
        assert spec["primary_method"] == "M2"

    def test_invalid_path_raises_filenotfounderror(
        self, tmp_path: Path
    ) -> None:
        missing = tmp_path / "does_not_exist.json"
        with pytest.raises(FileNotFoundError):
            load_locked_research_spec(str(missing))

    def test_malformed_json_raises_decodeerror(
        self, tmp_path: Path
    ) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            load_locked_research_spec(str(bad))

    def test_missing_task_type_raises_valueerror(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "no_task_type.json"
        path.write_text(json.dumps({"foo": "bar"}), encoding="utf-8")
        with pytest.raises(ValueError, match="task_type"):
            load_locked_research_spec(str(path))

    def test_structural_validation_failure_raises(
        self, tmp_path: Path
    ) -> None:
        # task_type is causal_soo but treatment block missing → must
        # fail structural validation under CausalSOOTemplate.
        path = tmp_path / "bad_spec.json"
        path.write_text(
            json.dumps(
                {
                    "task_type": "causal_soo",
                    "outcome": {"variable": "X", "type": "binary"},
                    "primary_method": "M2",
                    "target_estimand_hint": "ATT",
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="treatment"):
            load_locked_research_spec(str(path))


# ---------------------------------------------------------------------------
# 3b.6 / 6.8 — locked-spec fixture must include M5 for 3b.7 re-run
# ---------------------------------------------------------------------------


class TestLockedSpecFixtureFor3b7:
    """The fixture is the immutable input to 3b.7's re-run. 6.8 bumps
    it to include M5 (was excluded in 3b.5) and a new task_id so 3b.7's
    run directory does not collide with 3b.5's.
    """

    def test_locked_spec_includes_m5(self) -> None:
        spec = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        assert "M5" in spec["secondary_methods"], (
            "6.8 acceptance: M5 must appear in secondary_methods to "
            "exercise the causal-forest CATE pathway in 3b.7"
        )

    def test_locked_spec_excludes_nothing(self) -> None:
        spec = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        assert spec["exclude_methods"] == [], (
            "6.8 acceptance: 3b.5 had exclude=['M5']; 3b.7 includes "
            "all five methods so exclude_methods must be empty"
        )

    def test_locked_spec_task_id_bumped_from_3b5(self) -> None:
        spec = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        assert spec["task_id"] != "v3_0_smoketest_mtheff_college", (
            "6.8 acceptance: task_id must be bumped from 3b.5's value "
            "so 3b.7's run directory does not collide"
        )

    def test_locked_spec_validates_against_template(self) -> None:
        """Whatever the new fixture contents, structural validation must
        still pass under CausalSOOTemplate."""
        spec = load_locked_research_spec(str(FIXTURE_PATH))
        assert spec["task_type"] == "causal_soo"
