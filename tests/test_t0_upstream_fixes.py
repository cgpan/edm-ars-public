"""Arc T slice T0 — the six upstream defect fixes.

Every test in this file is deterministic and offline: no LLM call, no
network, no raw-data read. Tests that touch the archived run corpus skip
cleanly when it is absent.

Each defect is documented with the measured BEFORE value (taken against
``git show HEAD:`` copies of the pre-fix modules on 2026-07-25) so the
regression is legible without re-running the forensics.

Grounding: docs/v5_arc_t_spec.md §1.4, §3, §9 (T0).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.dataset_adapter import create_dataset_adapter
from src.design_selector import (
    classify_intent,
    did_feasible,
    itr_feasible,
    select_design,
)
from src.main import (
    _ADVISORY_WARNING_MARKERS,
    load_locked_research_spec,
)
from src.registry import RegistryLoader, is_excluded_variable
from src.task_template import PredictionTemplate

PROJECT_ROOT = Path(__file__).parent.parent
REGISTRY_DIR = PROJECT_ROOT / "data_registry" / "datasets"


def _registry(name: str) -> dict:
    with open(REGISTRY_DIR / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def hsls() -> dict:
    return _registry("hsls09_public")


@pytest.fixture(scope="module")
def els() -> dict:
    return _registry("els_2002")


@pytest.fixture(scope="module")
def panel() -> dict:
    return _registry("did_els_hsls_panel")


@pytest.fixture(scope="module")
def assistments() -> dict:
    return _registry("assistments_0910")


# ---------------------------------------------------------------------------
# Defect 1 — did_feasible
# ---------------------------------------------------------------------------


class TestDidFeasible:
    """BEFORE: any registry naming a ``multi_cohort_partner`` reported
    ``feasible=True, executable_task_type='causal_did'``, so
    ``select_design(hsls09_public, intent='causal')`` recommended
    ``causal_did`` — a task type that has never executed on HSLS. Both
    live DiD runs (runs/phase_b_did_20260704,
    runs/stream1_did_v2_20260708) ran on ``did_els_hsls_panel``.
    """

    def test_partner_alone_is_feasible_but_not_executable(self, hsls: dict) -> None:
        v = did_feasible(hsls)
        assert v.feasible is True          # the lead is real
        assert v.executable_task_type is None   # BEFORE: "causal_did"
        assert "harmonized" in " ".join(v.reasons)

    def test_causal_intent_no_longer_routes_hsls_to_causal_did(
        self, hsls: dict
    ) -> None:
        report = select_design(hsls, intent="causal")
        # BEFORE: "causal_did" (unexecutable on this dataset)
        assert report["recommended_task_type"] == "causal_soo"
        assert report["verdicts"]["did"]["executable_task_type"] is None

    def test_els_partner_pointer_also_not_executable(self, els: dict) -> None:
        assert did_feasible(els).executable_task_type is None

    def test_harmonized_panel_is_executable(self, panel: dict) -> None:
        v = did_feasible(panel)
        assert v.feasible is True
        assert v.executable_task_type == "causal_did"
        assert "cohort" in " ".join(v.reasons)
        assert select_design(panel, intent="causal")[
            "recommended_task_type"
        ] == "causal_did"

    def test_panel_ready_flag_makes_did_executable(self) -> None:
        """A future harmonized dataset flips DiD on via panel_ready
        alone — the predicate is registry-driven, not hardcoded."""
        reg = {
            "variables": {"predictors": {"any": [{"name": "X"}]}},
            "design_feasibility": {"panel_ready": True},
        }
        v = did_feasible(reg)
        assert v.feasible is True
        assert v.executable_task_type == "causal_did"

    def test_no_structure_is_infeasible(self, assistments: dict) -> None:
        v = did_feasible(assistments)
        assert v.feasible is False
        assert v.reasons

    @pytest.mark.parametrize("reg", [{}, {"design_feasibility": {}}, None])
    def test_degrades_on_missing_metadata(self, reg: dict | None) -> None:
        v = did_feasible(reg)  # type: ignore[arg-type]
        assert v.feasible is False
        assert v.reasons  # never a silent empty verdict


# ---------------------------------------------------------------------------
# Defect 2 — itr_feasible
# ---------------------------------------------------------------------------


class TestItrFeasible:
    """BEFORE: ``itr_ready`` (written by scripts/onboard_dataset.py) was
    never read, so every registry with any predictor list reported
    ITR-feasible — including assistments_0910 (itr_ready: false, zero
    protected attributes) and did_els_hsls_panel (itr_ready: false).
    """

    def test_declared_itr_ready_false_is_infeasible(
        self, assistments: dict
    ) -> None:
        v = itr_feasible(assistments)
        assert v.feasible is False        # BEFORE: True
        assert v.executable_task_type is None
        assert "itr_ready" in " ".join(v.reasons)

    def test_panel_itr_ready_false_is_infeasible(self, panel: dict) -> None:
        assert itr_feasible(panel).feasible is False   # BEFORE: True

    def test_targeting_intent_no_longer_routes_assistments_to_itr(
        self, assistments: dict
    ) -> None:
        report = select_design(assistments, intent="targeting")
        # BEFORE: "causal_itr" on a dataset that declares itr_ready: false
        assert report["recommended_task_type"] != "causal_itr"

    def test_itr_ready_true_stays_feasible(self, hsls: dict, els: dict) -> None:
        for reg in (hsls, els):
            v = itr_feasible(reg)
            assert v.feasible is True
            assert v.executable_task_type == "causal_itr"

    def test_absent_itr_ready_key_is_infeasible_not_assumed(self) -> None:
        """Degradation: a registry that never declares itr_ready must not
        be assumed ready (readiness is a curation decision)."""
        reg = {"variables": {"predictors": {"any": [{"name": "X"}]}}}
        v = itr_feasible(reg)
        assert v.feasible is False
        assert "does not declare" in " ".join(v.reasons)

    def test_no_predictors_is_infeasible(self) -> None:
        v = itr_feasible({"design_feasibility": {"itr_ready": True}})
        assert v.feasible is False


# ---------------------------------------------------------------------------
# Defect 3 — classify_intent word boundaries
# ---------------------------------------------------------------------------


class TestClassifyIntentWordBoundaries:
    """BEFORE: substring matching on "att" and "ate" classified
    *attitudes*, *attainment*, *attendance*, *climate* and *estimate* as
    causal-intent questions."""

    @pytest.mark.parametrize(
        "question",
        [
            "How do math attitudes predict college attendance?",   # att
            "Does school climate predict dropout risk?",            # ate
            "Which students are at risk of non-attainment?",        # att
            "Predict grade-12 attendance from base-year covariates",
            "Estimate dropout probability from 9th-grade data",     # ate
        ],
    )
    def test_substring_misfires_are_now_prediction(self, question: str) -> None:
        assert classify_intent(question) == "prediction"  # BEFORE: "causal"

    @pytest.mark.parametrize(
        "question",
        [
            "What is the ATE of tutoring on GPA?",
            "Estimate the ATT for treated students",
            "Does self-efficacy causally affect enrollment?",
            "What is the effect of tutoring on GPA?",
            "What are the effects of counseling on enrollment?",
            "Which treatments work best overall?",
            "the consequences of grade retention",
        ],
    )
    def test_genuine_causal_phrasings_still_route_to_causal(
        self, question: str
    ) -> None:
        assert classify_intent(question) == "causal"

    def test_morphological_tail_is_matched(self) -> None:
        """"impact of" -> "impacts of": bare substring matching missed the
        plural entirely (BEFORE: "prediction")."""
        assert classify_intent(
            "What are the impacts of counseling on enrollment?"
        ) == "causal"

    @pytest.mark.parametrize(
        "question",
        [
            "For whom does tutoring work?",
            "Who should receive the intervention?",
            "What treatment rules maximize enrollment?",
        ],
    )
    def test_targeting_still_wins(self, question: str) -> None:
        assert classify_intent(question) == "targeting"

    @pytest.mark.parametrize("question", [None, "", "   "])
    def test_degrades_to_prediction(self, question: str | None) -> None:
        assert classify_intent(question) == "prediction"


# ---------------------------------------------------------------------------
# Defect 5 — temporal wave resolved FROM THE REGISTRY
# ---------------------------------------------------------------------------


def _predict_spec(outcome: str, predictors: list[tuple[str, str]]) -> dict:
    return {
        "task_type": "prediction",
        "outcome_variable": outcome,
        "predictor_set": [
            {"variable": name, "wave": wave} for name, wave in predictors
        ],
        "novelty_score_self_assessment": 4,
    }


class TestRegistryResolvedTemporalWave:
    """BEFORE: the predictor wave came from ``pred.get("wave")`` — the
    LLM's own claim — so declaring the second-follow-up variable
    X3TGPAMAT as ``wave: base_year`` produced 0 warnings."""

    def setup_method(self) -> None:
        self.template = PredictionTemplate()
        self.adapter = create_dataset_adapter("hsls09_public")

    def test_misdeclared_wave_now_fires_temporal_violation(
        self, hsls: dict
    ) -> None:
        spec = _predict_spec(
            "X2TXMTSCOR", [("X3TGPAMAT", "base_year")]
        )
        warnings = self.template.validate_research_spec(
            spec, hsls, self.adapter
        )
        violations = [w for w in warnings if "TEMPORAL VIOLATION" in w]
        assert violations, "BEFORE: 0 warnings for a mis-declared wave"
        assert "registry wave=second_follow_up" in violations[0]  # truth
        assert "declared wave='base_year'" in violations[0]  # the claim

    def test_declared_wave_violation_still_caught(self, hsls: dict) -> None:
        """The registry check is ADDITIVE: a spec that openly declares a
        post-outcome wave for a variable the registry does not cover is
        still flagged (this is the pre-existing check, preserved)."""
        spec = _predict_spec("X2TXMTSCOR", [("X1NOTINREGISTRY", "update_panel")])
        warnings = self.template.validate_research_spec(
            spec, hsls, self.adapter
        )
        assert any("TEMPORAL VIOLATION" in w for w in warnings), warnings
        assert any("not in the dataset registry" in w for w in warnings)

    def test_honest_declaration_still_passes(self, hsls: dict) -> None:
        spec = _predict_spec(
            "X4EVRATNDCLG",
            [("X1TXMTSCOR", "base_year"), ("X1SES", "base_year")],
        )
        assert self.template.validate_research_spec(
            spec, hsls, self.adapter
        ) == []

    def test_predictor_absent_from_registry_is_flagged(
        self, hsls: dict
    ) -> None:
        spec = _predict_spec("X4EVRATNDCLG", [("X1INVENTED", "base_year")])
        warnings = self.template.validate_research_spec(
            spec, hsls, self.adapter
        )
        assert any("not in the dataset registry" in w for w in warnings)

    def test_degrades_when_registry_has_no_variables(self) -> None:
        """Empty registry: falls back to the adapter's temporal order and
        says what it could not verify, instead of crashing."""
        spec = _predict_spec("X4EVRATNDCLG", [("X1SES", "base_year")])
        warnings = self.template.validate_research_spec(
            spec, {}, self.adapter
        )
        assert any("not found in registry" in w for w in warnings)


# ---------------------------------------------------------------------------
# Defect 6 — RegistryLoader.is_excluded wired in
# ---------------------------------------------------------------------------


class TestTier3ExclusionWiring:
    """BEFORE: the Tier-3 machinery had zero production call sites — the
    spec's probe list ['W1STUDENT', 'STU_ID'] produced 0 warnings."""

    def setup_method(self) -> None:
        self.template = PredictionTemplate()
        self.adapter = create_dataset_adapter("hsls09_public")

    def test_weight_and_id_predictors_are_flagged(self, hsls: dict) -> None:
        spec = _predict_spec(
            "X4EVRATNDCLG",
            [("W1STUDENT", "base_year"), ("STU_ID", "base_year")],
        )
        warnings = self.template.validate_research_spec(
            spec, hsls, self.adapter
        )
        flagged = [w for w in warnings if "TIER-3 EXCLUDED" in w]
        assert len(flagged) == 2, f"BEFORE: 0 warnings; got {warnings}"
        assert any("W1STUDENT" in w for w in flagged)
        assert any("STU_ID" in w for w in flagged)

    def test_tier3_outcome_is_flagged(self, hsls: dict) -> None:
        spec = _predict_spec("W1STUDENT", [("X1SES", "base_year")])
        warnings = self.template.validate_research_spec(
            spec, hsls, self.adapter
        )
        assert any(
            "TIER-3 EXCLUDED" in w and "outcome" in w for w in warnings
        )

    def test_tier3_reported_even_when_outcome_wave_unresolvable(
        self, hsls: dict
    ) -> None:
        """The Tier-3 pass must not hide behind the temporal block, which
        is skipped whenever the outcome cannot be resolved."""
        spec = _predict_spec("X1NOT_A_REAL_OUTCOME", [("W1STUDENT", "base_year")])
        warnings = self.template.validate_research_spec(
            spec, hsls, self.adapter
        )
        assert any("TIER-3 EXCLUDED" in w for w in warnings), warnings

    def test_curated_flag_variable_is_not_excluded(self, hsls: dict) -> None:
        """FALSE-KILL GUARD. X1IEPFLAG is a curated Tier-1 predictor (IEP
        status) that matches the 'FLAG$' suffix rule. Curation must win:
        the pattern rules target the auto-profiled Tier-2/3 name space."""
        assert is_excluded_variable(
            "X1IEPFLAG", hsls.get("tier3_exclusion_rules")
        ) is True, "the raw pattern rule does match — that is the trap"

        spec = _predict_spec("X4EVRATNDCLG", [("X1IEPFLAG", "base_year")])
        warnings = self.template.validate_research_spec(
            spec, hsls, self.adapter
        )
        assert not any("TIER-3" in w for w in warnings), warnings

    def test_registry_without_tier3_rules_flags_nothing(self) -> None:
        reg = {
            "temporal_order": ["base_year", "first_follow_up"],
            "levels": {"student": 23503},
            "variables": {
                "outcomes": [
                    {"name": "Y", "wave": "first_follow_up", "pct_missing": 1.0}
                ],
                "predictors": {},
            },
        }
        spec = _predict_spec("Y", [("W1STUDENT", "base_year")])
        warnings = self.template.validate_research_spec(
            spec, reg, self.adapter
        )
        assert not any("TIER-3" in w for w in warnings)

    def test_module_predicate_matches_loader(self, hsls: dict) -> None:
        loader = RegistryLoader(str(REGISTRY_DIR / "hsls09_public.yaml"))
        rules = hsls.get("tier3_exclusion_rules")
        for name in (
            "W1STUDENT", "STU_ID", "X1TXMTSC_IM", "X1TXMTSCOR", "X1SES",
        ):
            assert is_excluded_variable(name, rules) == loader.is_excluded(name)

    def test_from_dict_parity(self, hsls: dict) -> None:
        from_disk = RegistryLoader(str(REGISTRY_DIR / "hsls09_public.yaml"))
        from_dict = RegistryLoader.from_dict(hsls)
        for name in ("W1STUDENT", "X1SES", "STU_ID"):
            assert from_dict.is_excluded(name) == from_disk.is_excluded(name)
        assert from_dict.is_protected_attribute("X1SEX") is True

    def test_is_excluded_variable_degrades_without_rules(self) -> None:
        assert is_excluded_variable("W1STUDENT") is False
        assert is_excluded_variable("W1STUDENT", None) is False
        assert is_excluded_variable("W1STUDENT", {}) is False


# ---------------------------------------------------------------------------
# Defect 4 — src/main.py:48 validate_research_spec call
# ---------------------------------------------------------------------------


def _write(tmp_path: Path, name: str, spec: dict) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(spec), encoding="utf-8")
    return str(path)


class TestLoadLockedPredictionSpec:
    """BEFORE: ``template.validate_research_spec(spec)`` raised
    ``TypeError: validate_research_spec() missing 2 required positional
    arguments: 'registry' and 'dataset_adapter'`` for every
    ``task_type: "prediction"`` locked spec — the Arc T winner-spec seam
    runs through this function."""

    def test_prediction_spec_loads(self, tmp_path: Path) -> None:
        spec = _predict_spec(
            "X4EVRATNDCLG",
            [("X1TXMTSCOR", "base_year"), ("X1SES", "base_year")],
        )
        spec["dataset"] = "hsls09_public"
        loaded = load_locked_research_spec(_write(tmp_path, "p.json", spec))
        assert loaded["task_type"] == "prediction"   # BEFORE: TypeError

    def test_temporal_violation_blocks_the_load(self, tmp_path: Path) -> None:
        spec = _predict_spec("X2TXMTSCOR", [("X3TGPAMAT", "base_year")])
        spec["dataset"] = "hsls09_public"
        with pytest.raises(ValueError, match="TEMPORAL VIOLATION"):
            load_locked_research_spec(_write(tmp_path, "bad.json", spec))

    def test_tier3_predictor_blocks_the_load(self, tmp_path: Path) -> None:
        spec = _predict_spec("X4EVRATNDCLG", [("W1STUDENT", "base_year")])
        spec["dataset"] = "hsls09_public"
        with pytest.raises(ValueError, match="TIER-3 EXCLUDED"):
            load_locked_research_spec(_write(tmp_path, "w.json", spec))

    def test_dataset_resolved_from_the_spec(self, tmp_path: Path) -> None:
        """An ELS spec validates against the ELS registry even when the
        caller passes the CLI's hsls default."""
        spec = _predict_spec(
            "F2EVRATT",
            [("BYTXMSTD", "base_year"), ("BYSES1", "base_year")],
        )
        spec["dataset"] = "els_2002"
        loaded = load_locked_research_spec(
            _write(tmp_path, "els.json", spec), dataset="hsls09_public"
        )
        assert loaded["dataset"] == "els_2002"

    def test_analytic_n_heuristic_is_advisory_not_blocking(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The sum-of-pct_missing retention rule is known-broken
        (measured: estimates 0 where the executed runs carried
        analytic_n = 14,039 / 17,335). It must not hard-fail a load."""
        spec = _predict_spec(
            "X4EVRATNDCLG",
            [
                ("X1PAR2EDU", "base_year"),      # 43.99% missing
                ("X1FAMINCOME", "base_year"),    # 28.69%
                ("X1TXMTSCOR", "base_year"),     # 8.76%
            ],
        )
        spec["dataset"] = "hsls09_public"
        loaded = load_locked_research_spec(_write(tmp_path, "a.json", spec))
        assert loaded["outcome_variable"] == "X4EVRATNDCLG"
        assert "ADVISORY" in capsys.readouterr().err

    def test_advisory_marker_still_matches_template_prose(
        self, hsls: dict
    ) -> None:
        """Drift guard: the advisory filter is prose-coupled, so assert
        the marker still appears in PredictionTemplate's output."""
        spec = _predict_spec(
            "X4EVRATNDCLG",
            [("X1PAR2EDU", "base_year"), ("X1FAMINCOME", "base_year"),
             ("X1TXMTSCOR", "base_year")],
        )
        warnings = PredictionTemplate().validate_research_spec(
            spec, hsls, create_dataset_adapter("hsls09_public")
        )
        assert any(
            m in w for w in warnings for m in _ADVISORY_WARNING_MARKERS
        ), warnings

    def test_unknown_dataset_raises_valueerror(self, tmp_path: Path) -> None:
        spec = _predict_spec("X4EVRATNDCLG", [("X1SES", "base_year")])
        spec["dataset"] = "no_such_dataset"
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_locked_research_spec(_write(tmp_path, "u.json", spec))

    def test_missing_registry_dir_raises_named_error(
        self, tmp_path: Path
    ) -> None:
        spec = _predict_spec("X4EVRATNDCLG", [("X1SES", "base_year")])
        spec["dataset"] = "hsls09_public"
        with pytest.raises(ValueError, match="No registry YAML"):
            load_locked_research_spec(
                _write(tmp_path, "m.json", spec),
                registry_dir=str(tmp_path / "nowhere"),
            )

    def test_causal_fixture_still_loads_unchanged(self) -> None:
        fixture = PROJECT_ROOT / "runs" / "fixtures" / "spec_x1mtheff_x4college.json"
        if not fixture.exists():          # pragma: no cover - repo layout
            pytest.skip("locked-spec fixture not present")
        spec = load_locked_research_spec(str(fixture))
        assert spec["task_type"] == "causal_soo"


# ---------------------------------------------------------------------------
# V1-style false-kill guard over the archived corpus
# ---------------------------------------------------------------------------


class TestNoFalseWarningsOnArchivedPredictionSpecs:
    """docs/v5_arc_t_spec.md §6 V1: a real archived spec must never be
    rejected by the new checks. Temporal + Tier-3 warnings must stay at
    zero on all archived prediction specs; only the known-broken
    analytic_n heuristic may fire (and it is advisory)."""

    def test_archived_specs_carry_no_new_warnings(self) -> None:
        runs = PROJECT_ROOT / "runs"
        if not runs.is_dir():             # pragma: no cover - repo layout
            pytest.skip("archived run corpus not present")
        specs = sorted(runs.glob("*/output/research_spec.json"))
        checked = 0
        for path in specs:
            spec = json.loads(path.read_text(encoding="utf-8"))
            if spec.get("task_type") not in (None, "prediction"):
                continue
            if not spec.get("predictor_set"):
                continue
            dataset = spec.get("dataset") or (
                "els_2002" if "els" in path.as_posix() else "hsls09_public"
            )
            spec = dict(spec, task_type="prediction")
            warnings = PredictionTemplate().validate_research_spec(
                spec, _registry(dataset), create_dataset_adapter(dataset)
            )
            offenders = [
                w
                for w in warnings
                if not any(m in w for m in _ADVISORY_WARNING_MARKERS)
            ]
            assert offenders == [], f"{path}: {offenders}"
            checked += 1
        if checked == 0:                  # pragma: no cover - repo layout
            pytest.skip("no archived prediction specs found")
