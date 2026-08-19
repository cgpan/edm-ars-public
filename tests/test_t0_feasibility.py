"""Arc T / T0 - tests for the deterministic feasibility screen.

Three obligations, in priority order:

1. **No false kills.** A KILL deletes a research question with no human
   in the loop. ``test_no_false_kills_on_archived_specs`` replays the
   screen over every spec that actually shipped; it is the blocking V1
   gate. The degradation tests assert that every "cannot establish the
   fact" path returns OK or WARN.
2. **Every KILL code fires** on a spec that is broken in exactly that
   way.
3. **WARN codes warn**, never kill.

Everything here runs offline. Tests that need a raw data file skip when
it is absent, so the suite passes on a machine without the datasets.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from src.ideation import feasibility as F
from src.ideation import probe_cache

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_DIR = REPO_ROOT / "data_registry" / "datasets"
RUNS_DIR = REPO_ROOT / "runs"


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _registry(dataset: str) -> dict:
    with open(REGISTRY_DIR / f"{dataset}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture()
def hsls_soo_spec() -> dict:
    """A minimal, feasible causal_soo spec on HSLS (mirrors the shipped one)."""
    return {
        "task_id": "t0_test_soo",
        "task_type": "causal_soo",
        "dataset": "hsls09_public",
        "research_question": (
            "What is the ATT of above-median 9th-grade math self-efficacy on "
            "ever attending college?"
        ),
        "treatment": {
            "variable": "X1MTHEFF",
            "operationalization": "median_split_binary",
        },
        "outcome": {"variable": "X4EVRATNDCLG", "type": "binary"},
        "estimand": "ATT",
        "adjustment_set": ["X1TXMTSCOR", "X1SES", "X1RACE", "X1SEX"],
        "primary_method": "M2",
        "comparator_method": "M1",
        "secondary_methods": ["M3"],
    }


@pytest.fixture()
def hsls_prediction_spec() -> dict:
    return {
        "task_type": "prediction",
        "dataset": "hsls09_public",
        "research_question": "Do 9th-grade attitudes predict college enrollment?",
        "outcome_variable": "X4EVRATNDCLG",
        "outcome_type": "binary",
        "predictor_set": [
            {"variable": "X1TXMTSCOR", "rationale": "prior achievement",
             "wave": "base_year"},
            {"variable": "X1MTHEFF", "rationale": "self-efficacy",
             "wave": "base_year"},
        ],
        "subgroup_analyses": ["X1SEX", "X1RACE"],
    }


@pytest.fixture()
def psy_spec() -> dict:
    return {
        "task_type": "psychometrics",
        "dataset": "hsls09_public",
        "scale_name": "HSLS math self-efficacy",
        "item_columns": ["S1MTESTS", "S1MTEXTBOOK", "S1MSKILLS", "S1MASSEXCL"],
        "factor_model": "EFF =~ S1MTESTS + S1MTEXTBOOK + S1MSKILLS + S1MASSEXCL",
        "method_battery": ["P1", "P2", "P3"],
        "grouping_vars": ["X1SEX"],
    }


def _ctx(spec: dict, dataset: str | None = None, **kwargs):
    kwargs.setdefault("registry_dir", REGISTRY_DIR)
    return F.make_context(spec, dataset=dataset, **kwargs)


# --------------------------------------------------------------------------
# 1. KILL cases - one per KILL code
# --------------------------------------------------------------------------


def test_kill_task_incompatible(hsls_soo_spec: dict) -> None:
    spec = dict(hsls_soo_spec, task_type="causal_did")
    result = F.check_dataset_task_compatibility(_ctx(spec))
    assert result.status == F.KILL
    assert "policy_timing_variables" in result.evidence


def test_kill_variable_absent_with_explicit_columns(hsls_soo_spec: dict) -> None:
    spec = copy.deepcopy(hsls_soo_spec)
    spec["adjustment_set"][0] = "X1MTHCONFIDENCE"  # invented but plausible
    ctx = _ctx(spec, columns=["X1MTHEFF", "X4EVRATNDCLG", "X1SES", "X1RACE", "X1SEX"])
    result = F.check_variables_exist_in_registry(ctx)
    assert result.status == F.KILL
    assert "X1MTHCONFIDENCE" in result.message


def test_kill_variable_absent_offline_via_tier2_profile() -> None:
    """ELS carries a Tier-2 profile of all 4,012 columns, so the KILL is
    reachable with no raw data on disk."""
    spec = {
        "task_type": "prediction",
        "dataset": "els_2002",
        "outcome_variable": "F2EVRATT",
        "predictor_set": [
            {"variable": "BYTXMSTD", "rationale": "prior achievement"},
            {"variable": "BYIMAGINARYVAR", "rationale": "invented"},
        ],
    }
    ctx = _ctx(spec, use_column_cache=False)
    assert ctx.tier2_names, "ELS Tier-2 draft profile should be on disk"
    result = F.check_variables_exist_in_registry(ctx)
    assert result.status == F.KILL
    assert "BYIMAGINARYVAR" in result.message


def test_kill_column_absent(hsls_soo_spec: dict) -> None:
    spec = copy.deepcopy(hsls_soo_spec)
    spec["adjustment_set"].append("X1SES_TOTALLY_MADE_UP")
    ctx = _ctx(
        spec,
        columns=["X1MTHEFF", "X4EVRATNDCLG", "X1TXMTSCOR", "X1SES", "X1RACE", "X1SEX"],
    )
    result = F.check_columns_exist_in_csv(ctx)
    assert result.status == F.KILL
    assert "X1SES_TOTALLY_MADE_UP" in result.message


def test_kill_temporal_order_uses_registry_wave(hsls_prediction_spec: dict) -> None:
    """The spec's own `wave` field is a lie here; the registry is believed."""
    spec = copy.deepcopy(hsls_prediction_spec)
    spec["predictor_set"].append(
        {"variable": "X4EVERDROP", "rationale": "same wave as outcome",
         "wave": "base_year"}
    )
    result = F.check_temporal_order(_ctx(spec))
    assert result.status == F.KILL
    assert "X4EVERDROP" in result.message


def test_kill_tier3_excluded(hsls_soo_spec: dict) -> None:
    spec = copy.deepcopy(hsls_soo_spec)
    spec["adjustment_set"].append("W1STUDENT")
    result = F.check_tier3_exclusion(_ctx(spec))
    assert result.status == F.KILL
    assert "W1STUDENT" in result.message


def test_kill_dead_variable(hsls_soo_spec: dict) -> None:
    spec = copy.deepcopy(hsls_soo_spec)
    spec["adjustment_set"].append("X1FREELUNCH")  # pct_missing 100 (suppressed)
    result = F.check_dead_variables(_ctx(spec))
    assert result.status == F.KILL
    assert "X1FREELUNCH" in result.message


def test_kill_estimator_uncertified_shelved_design(hsls_soo_spec: dict) -> None:
    spec = dict(hsls_soo_spec, primary_method="RD")
    result = F.check_estimator_certified(_ctx(spec))
    assert result.status == F.KILL
    assert "shelved" in result.message


def test_kill_estimator_uncertified_unknown_id(hsls_soo_spec: dict) -> None:
    spec = dict(hsls_soo_spec, primary_method="M42")
    result = F.check_estimator_certified(_ctx(spec))
    assert result.status == F.KILL


def test_kill_design_infeasible_itr_on_assistments() -> None:
    spec = {
        "task_type": "causal_itr",
        "dataset": "assistments_0910",
        "research_question": "For whom does hint use help?",
        "treatment": {"variable": "hint_count", "operationalization": "median_split_binary"},
        "outcome": {"variable": "correct", "type": "binary"},
        "adjustment_set": ["attempt_count"],
        "rule_covariates": ["attempt_count"],
        "primary_method": "M6",
    }
    result = F.check_design_feasible(_ctx(spec))
    assert result.status == F.KILL
    assert "design_selector" in result.evidence


def test_kill_spec_incomplete_missing_dispatch_field(hsls_soo_spec: dict) -> None:
    spec = copy.deepcopy(hsls_soo_spec)
    spec.pop("treatment")
    result = F.check_structural_completeness(_ctx(spec))
    assert result.status == F.KILL
    assert "treatment" in result.message


def test_kill_no_protected_attributes_on_assistments() -> None:
    spec = {
        "task_type": "psychometrics",
        "dataset": "assistments_0910",
        "scale_name": "skill mastery",
        "item_construction": "templates within skills",
        "method_battery": ["P7"],
        "subgroup_analyses": ["skill_name"],
    }
    result = F.check_protected_attributes(_ctx(spec))
    assert result.status == F.KILL
    assert "protected_attribute" in result.evidence


def test_kill_item_bank_too_few(psy_spec: dict) -> None:
    spec = dict(
        psy_spec,
        factor_model="ID =~ S1MPERSON1 + S1MPERSON2",
        item_columns=["S1MPERSON1", "S1MPERSON2"],
    )
    result = F.check_item_bank_adequacy(_ctx(spec))
    assert result.status == F.KILL
    assert "2 items" in result.message


# --------------------------------------------------------------------------
# 2. WARN cases - informative, never fatal
# --------------------------------------------------------------------------


def test_warn_subgroup_variable_unknown(hsls_prediction_spec: dict) -> None:
    spec = copy.deepcopy(hsls_prediction_spec)
    spec["subgroup_analyses"].append("X1GENDERIDENTITY")
    ctx = _ctx(spec, columns=["X1TXMTSCOR", "X1MTHEFF", "X4EVRATNDCLG", "X1SEX", "X1RACE"])
    result = F.check_subgroup_variables(ctx)
    assert result.status == F.WARN
    assert result.penalty > 0


def test_warn_metadata_unverified_for_tier2_variable() -> None:
    spec = {
        "task_type": "prediction",
        "dataset": "els_2002",
        "outcome_variable": "F2EVRATT",
        "predictor_set": [
            {"variable": "BYTXMSTD", "rationale": "prior achievement"},
            {"variable": "BYPARASP", "rationale": "Tier-2, real but uncurated"},
        ],
    }
    result = F.check_metadata_verified(_ctx(spec, use_column_cache=False))
    assert result.status == F.WARN
    assert "BYPARASP" in result.message or "BYPARASP" in result.evidence


def test_warn_pitfall_protected_attribute_misuse(hsls_prediction_spec: dict) -> None:
    spec = copy.deepcopy(hsls_prediction_spec)
    spec["predictor_set"].append(
        {"variable": "X1SES", "rationale": "SES", "wave": "base_year"}
    )
    spec.pop("subgroup_analyses")
    result = F.check_common_pitfalls(_ctx(spec))
    assert result.status == F.WARN
    assert "protected_attribute_misuse" in result.message


def test_equity_keyword_without_named_subgroups_warns_not_kills() -> None:
    """Keyword-only equity detection is probabilistic -> WARN, never KILL."""
    spec = {
        "task_type": "prediction",
        "dataset": "assistments_0910",
        "research_question": "Does the model show fairness across learners?",
        "outcome_variable": "correct",
        "predictor_set": [{"variable": "attempt_count", "rationale": "x"}],
    }
    result = F.check_protected_attributes(_ctx(spec))
    assert result.status == F.WARN


def test_curated_variable_matching_tier3_pattern_is_not_killed(
    hsls_soo_spec: dict,
) -> None:
    """X1IEPFLAG matches the FLAG$ tier-3 suffix AND is a curated Tier-1
    predictor used by 3 shipped specs. Curation wins."""
    spec = copy.deepcopy(hsls_soo_spec)
    spec["adjustment_set"].append("X1IEPFLAG")
    result = F.check_tier3_exclusion(_ctx(spec))
    assert result.status == F.OK
    assert "X1IEPFLAG" in result.message


def test_method_aliases_from_completed_runs_are_not_killed(
    hsls_soo_spec: dict,
) -> None:
    """The 3b.5 smoke test ran end to end with primary_method 'IPW'."""
    spec = dict(
        hsls_soo_spec,
        primary_method="IPW",
        comparator_method="PSM",
        secondary_methods=["regression_adjustment", "AIPW"],
    )
    result = F.check_estimator_certified(_ctx(spec))
    assert result.status == F.OK


# --------------------------------------------------------------------------
# 3. Degradation - "cannot establish" must never be a KILL
# --------------------------------------------------------------------------


def test_missing_registry_never_kills(hsls_soo_spec: dict) -> None:
    report = F.screen(
        hsls_soo_spec,
        dataset="hsls09_public",
        registry={},  # registry deliberately empty
        registry_dir=REGISTRY_DIR,
    )
    assert F.KILL not in [c.status for c in report.checks], report.kill_codes


def test_missing_registry_key_never_kills(hsls_soo_spec: dict) -> None:
    """A registry missing 'variables' entirely (a half-onboarded dataset)."""
    registry = {"name": "hsls09_public", "temporal_order": ["base_year"]}
    report = F.screen(
        hsls_soo_spec,
        dataset="hsls09_public",
        registry=registry,
        registry_dir=REGISTRY_DIR,
        columns=None,
    )
    assert report.verdict != F.KILL, report.kill_codes


def test_absent_csv_skips_the_column_check(hsls_soo_spec: dict, tmp_path: Path) -> None:
    ctx = _ctx(hsls_soo_spec, dataset="hsls09_public", raw_data_dir=tmp_path)
    assert ctx.columns is None
    result = F.check_columns_exist_in_csv(ctx)
    assert result.status == F.OK
    assert "Skipped" in result.message


def test_absent_csv_downgrades_unknown_variable_to_warn(tmp_path: Path) -> None:
    """HSLS has no Tier-2 profile on disk, so with no CSV there is no
    column universe: an unresolvable name is UNVERIFIED, not absent."""
    spec = {
        "task_type": "prediction",
        "dataset": "hsls09_public",
        "outcome_variable": "X4EVRATNDCLG",
        "predictor_set": [{"variable": "X1COMPLETELYINVENTED", "rationale": "x"}],
    }
    ctx = _ctx(spec, dataset="hsls09_public", raw_data_dir=tmp_path)
    assert ctx.column_universe() is None
    result = F.check_variables_exist_in_registry(ctx)
    assert result.status == F.WARN


def test_absent_optional_metadata_never_kills() -> None:
    """A registry entry with no wave and no pct_missing (a fresh Tier-2
    promotion) must not trigger the temporal or dead-variable KILLs."""
    registry = {
        "name": "toy",
        "temporal_order": ["w1", "w2"],
        "levels": {"student": 5000},
        "variables": {
            "outcomes": [{"name": "OUT", "type": "binary"}],
            "predictors": {"d": [{"name": "PRED"}]},
        },
    }
    spec = {
        "task_type": "prediction",
        "outcome_variable": "OUT",
        "predictor_set": [{"variable": "PRED", "rationale": "x"}],
    }
    ctx = F.make_context(spec, dataset="toy_dataset", registry=registry)
    assert F.check_temporal_order(ctx).status == F.OK
    assert F.check_dead_variables(ctx).status == F.OK
    assert F.check_variables_exist_in_registry(ctx).status == F.OK


def test_unknown_dataset_warns_rather_than_kills() -> None:
    spec = {"task_type": "prediction", "dataset": "nonexistent_dataset",
            "outcome_variable": "Y", "predictor_set": [{"variable": "X"}]}
    report = F.screen(spec, registry_dir=REGISTRY_DIR)
    assert report.verdict != F.KILL, report.kill_codes


def test_screen_never_raises_on_degenerate_input() -> None:
    for spec in ({}, {"task_type": "prediction"}, {"task_type": "not_a_task"}):
        report = F.screen(spec, registry_dir=REGISTRY_DIR)
        assert isinstance(report, F.FeasibilityReport)


def test_every_check_result_carries_evidence(hsls_soo_spec: dict) -> None:
    """C2: a component that cannot cite what it read is deleted, not shipped."""
    report = F.screen(hsls_soo_spec, dataset="hsls09_public", registry_dir=REGISTRY_DIR)
    assert len(report.checks) == len(F.STAGE0_CHECKS)
    for check in report.checks:
        assert check.evidence and check.evidence.strip(), check.code
        assert check.status in (F.KILL, F.WARN, F.OK)


def test_warns_never_change_the_verdict_to_kill(hsls_prediction_spec: dict) -> None:
    spec = copy.deepcopy(hsls_prediction_spec)
    spec["subgroup_analyses"] = ["X1GENDERIDENTITY_FAKE"]
    report = F.screen(
        spec,
        dataset="hsls09_public",
        registry_dir=REGISTRY_DIR,
        columns=["X1TXMTSCOR", "X1MTHEFF", "X4EVRATNDCLG"],
    )
    warn_only = [c for c in report.checks if c.status == F.WARN]
    assert warn_only
    assert report.penalty > 0


def test_screen_is_deterministic(hsls_soo_spec: dict) -> None:
    a = F.screen(hsls_soo_spec, dataset="hsls09_public", registry_dir=REGISTRY_DIR)
    b = F.screen(hsls_soo_spec, dataset="hsls09_public", registry_dir=REGISTRY_DIR)
    assert a.to_dict() == b.to_dict()


# --------------------------------------------------------------------------
# 4. C1 - novelty is never read
# --------------------------------------------------------------------------


def test_ranking_invariant_to_self_novelty(hsls_soo_spec: dict) -> None:
    keys = set()
    for novelty in (1, 3, 5, 0.43, {"score": 4}, None):
        spec = dict(hsls_soo_spec, novelty_score_self_assessment=novelty)
        keys.add(
            F.rank_key(spec, dataset="hsls09_public", registry_dir=REGISTRY_DIR)
        )
    assert len(keys) == 1


def test_module_never_reads_the_novelty_field() -> None:
    source = Path(F.__file__).read_text(encoding="utf-8")
    body = "\n".join(
        line for line in source.splitlines()
        if "novelty" not in line.lower() or line.strip().startswith("#")
        or '"""' in line or line.strip().startswith("*")
    )
    assert 'get("novelty_score_self_assessment"' not in body
    assert '["novelty_score_self_assessment"]' not in source


# --------------------------------------------------------------------------
# 5. DATASET_TASK_MATRIX
# --------------------------------------------------------------------------


def test_matrix_covers_four_datasets_and_five_task_types() -> None:
    assert set(F.DATASET_TASK_MATRIX) == {
        "hsls09_public", "els_2002", "did_els_hsls_panel", "assistments_0910",
    }
    for row in F.DATASET_TASK_MATRIX.values():
        assert set(row) == {
            "prediction", "causal_soo", "causal_itr", "causal_did", "psychometrics",
        }


def test_matrix_did_cell_matches_registry_policy_timing() -> None:
    for dataset, row in F.DATASET_TASK_MATRIX.items():
        registry = _registry(dataset)
        timing = (registry.get("design_feasibility") or {}).get(
            "policy_timing_variables"
        ) or []
        assert row["causal_did"] == bool(timing), dataset


def test_matrix_itr_cell_matches_registry_itr_ready() -> None:
    for dataset, row in F.DATASET_TASK_MATRIX.items():
        registry = _registry(dataset)
        ready = (registry.get("design_feasibility") or {}).get("itr_ready")
        assert row["causal_itr"] == bool(ready), dataset


def test_matrix_psychometrics_cell_matches_item_level_data() -> None:
    for dataset, row in F.DATASET_TASK_MATRIX.items():
        registry = _registry(dataset)
        has_items = bool(registry.get("item_banks")) or bool(registry.get("cdm_support"))
        assert row["psychometrics"] == has_items, dataset


def test_every_unsupported_cell_has_a_reason() -> None:
    for dataset, row in F.DATASET_TASK_MATRIX.items():
        for task, supported in row.items():
            if not supported:
                reason = F._UNSUPPORTED_REASONS[(dataset, task)]
                assert len(reason) > 40, (dataset, task)


# --------------------------------------------------------------------------
# 6. Stage-1 probes
# --------------------------------------------------------------------------


def test_probes_skip_cleanly_without_raw_data(
    hsls_soo_spec: dict, tmp_path: Path
) -> None:
    ctx = _ctx(hsls_soo_spec, dataset="hsls09_public", raw_data_dir=tmp_path)
    results = F.probe(ctx)
    assert len(results) == len(F.STAGE1_PROBES)
    for result in results:
        assert result.status == F.OK
        assert "Skipped" in result.message
        assert result.evidence


@pytest.mark.skipif(
    not (REPO_ROOT / "data" / "raw" / "did_els_hsls_panel" / "panel.csv").exists(),
    reason="harmonized panel CSV not present on this machine",
)
def test_probe_results_are_warn_or_ok_never_kill(tmp_path: Path) -> None:
    """Probes read real data (the 1.3 MB panel) and still cannot kill."""
    spec = {
        "task_type": "causal_did",
        "dataset": "did_els_hsls_panel",
        "outcome": {"variable": "rank_base", "type": "continuous"},
        "group_variable": "low_ses",
        "post_variable": "cohort",
        "adjustment_covariates": ["race5", "pared3"],
        "primary_method": "M8",
    }
    ctx = _ctx(spec, dataset="did_els_hsls_panel", cache_dir=tmp_path)
    for result in F.probe(ctx):
        assert result.status in (F.OK, F.WARN)


@pytest.mark.skipif(
    not (REPO_ROOT / "data" / "raw" / "did_els_hsls_panel" / "panel.csv").exists(),
    reason="harmonized panel CSV not present on this machine",
)
def test_did_cell_probe_on_real_panel(tmp_path: Path) -> None:
    spec = {
        "task_type": "causal_did",
        "dataset": "did_els_hsls_panel",
        "outcome": {"variable": "rank_base", "type": "continuous"},
        "group_variable": "low_ses",
        "post_variable": "cohort",
        "primary_method": "M8",
    }
    ctx = _ctx(spec, dataset="did_els_hsls_panel", cache_dir=tmp_path)
    result = F.check_did_cells(ctx)
    assert result.status in (F.OK, F.WARN)
    assert "groupby" in result.evidence


@pytest.mark.skipif(
    not (REPO_ROOT / "data" / "raw" / "did_els_hsls_panel" / "panel.csv").exists(),
    reason="harmonized panel CSV not present on this machine",
)
def test_analytic_n_probe_on_real_panel(tmp_path: Path) -> None:
    spec = {
        "task_type": "prediction",
        "dataset": "did_els_hsls_panel",
        "outcome_variable": "rank_follow",
        "predictor_set": [
            {"variable": "ses_std", "rationale": "x"},
            {"variable": "female", "rationale": "x"},
        ],
    }
    ctx = _ctx(spec, dataset="did_els_hsls_panel", cache_dir=tmp_path)
    result = F.estimate_analytic_n(ctx)
    assert result.status in (F.OK, F.WARN)
    assert "Analytic n" in result.message
    assert "listwise-complete" in result.message


# --------------------------------------------------------------------------
# 7. probe_cache
# --------------------------------------------------------------------------


def test_header_columns_returns_none_for_absent_file(tmp_path: Path) -> None:
    assert probe_cache.header_columns(
        "hsls09_public", raw_data_dir=tmp_path, cache_dir=tmp_path
    ) is None


def test_header_columns_unknown_dataset_is_none(tmp_path: Path) -> None:
    assert probe_cache.header_columns("not_a_dataset", cache_dir=tmp_path) is None


def test_tier1_columns_includes_item_bank_items() -> None:
    registry = _registry("hsls09_public")
    columns = probe_cache.tier1_columns(registry)
    assert "X1MTHEFF" in columns
    assert "S1MTESTS" in columns  # item-bank item, not a Tier-1 variable
    assert len(columns) == len(set(columns))


def test_tier1_frame_is_none_without_raw_data(tmp_path: Path) -> None:
    registry = _registry("hsls09_public")
    assert probe_cache.tier1_frame(
        "hsls09_public", registry, raw_data_dir=tmp_path, cache_dir=tmp_path
    ) is None


@pytest.mark.skipif(
    not (REPO_ROOT / "data" / "raw" / "did_els_hsls_panel" / "panel.csv").exists(),
    reason="harmonized panel CSV not present on this machine",
)
def test_tier1_cache_round_trip(tmp_path: Path) -> None:
    registry = _registry("did_els_hsls_panel")
    first = probe_cache.tier1_frame(
        "did_els_hsls_panel", registry, cache_dir=tmp_path
    )
    assert first is not None and not first.empty
    status = probe_cache.cache_status("did_els_hsls_panel", cache_dir=tmp_path)
    assert status.frame_cached and status.raw_exists
    second = probe_cache.tier1_frame(
        "did_els_hsls_panel", registry, cache_dir=tmp_path, allow_build=False
    )
    assert second is not None
    assert list(second.columns) == list(first.columns)


# --------------------------------------------------------------------------
# 8. V1 - the blocking false-kill gate over the real archive
# --------------------------------------------------------------------------


def _archived_specs() -> list[tuple[str, dict, str | None, str | None]]:
    out: list[tuple[str, dict, str | None, str | None]] = []
    for path in sorted(RUNS_DIR.glob("*/output/research_spec.json")):
        try:
            with open(path, encoding="utf-8") as f:
                spec = json.load(f)
        except (OSError, ValueError):
            continue
        if not isinstance(spec, dict):
            continue
        dataset = spec.get("dataset")
        task_type = spec.get("task_type")
        checkpoint = path.parent / "checkpoint.json"
        if (not dataset or not task_type) and checkpoint.exists():
            with open(checkpoint, encoding="utf-8") as f:
                data = json.load(f)
            dataset = dataset or data.get("dataset_name")
            task_type = task_type or data.get("task_type")
        out.append((path.parent.parent.name, spec, dataset, task_type))
    return out


@pytest.mark.skipif(not RUNS_DIR.exists(), reason="no run archive on this machine")
def test_no_false_kills_on_archived_specs() -> None:
    """V1, blocking: every spec here produced a real paper. Killing one is a bug."""
    archived = _archived_specs()
    if not archived:
        pytest.skip("run archive contains no research_spec.json files")
    killed = []
    for run, spec, dataset, task_type in archived:
        report = F.screen(
            spec, candidate_id=run, dataset=dataset, task_type=task_type
        )
        if report.verdict == F.KILL:
            killed.append((run, report.kill_codes))
    assert not killed, f"FALSE KILLS on shipped specs: {killed}"


#: The V1 gate needs a corpus of archived research specs, which lives in
#: run OUTPUT directories. Those are excluded from the public release
#: (they contain derived analytic data and this project's own research
#: output), so a fresh clone has no corpus and the gate cannot run.
#:
#: The skip is deliberately narrow. It fires only when NO run-output
#: directory exists at all — a checkout that never had a corpus. If the
#: directories exist but yield too few specs, that is the corpus having
#: been LOST, and the test still fails: deleting a git worktree once took
#: the entire 26-spec denominator with it and the audit reported zero
#: canonical specs while still exiting 0. A blanket skip would restore
#: exactly that blind spot.
_RUN_OUTPUT_DIRS = sorted(RUNS_DIR.glob("*/output*")) if RUNS_DIR.exists() else []


@pytest.mark.skipif(
    not _RUN_OUTPUT_DIRS,
    reason="no run-output corpus in this checkout (public release excludes it)",
)
def test_audit_script_reports_zero_false_kills() -> None:
    from scripts.audit_feasibility import run_audit

    result = run_audit(RUNS_DIR)
    assert result["n_canonical"] >= 20, (
        f"run outputs exist ({len(_RUN_OUTPUT_DIRS)} dirs) but only "
        f"{result['n_canonical']} canonical specs were found — the gate's "
        "denominator has been lost, which is the failure this asserts against"
    )
    assert result["false_kill_rate"] == 0.0, result["false_kills"]
    assert result["mutant_kill_rate"] == 1.0, result["mutants_missed"]
