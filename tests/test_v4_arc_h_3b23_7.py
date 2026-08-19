"""V4 Arc H / Phase 3b.23.7 — DE hardening verification.

Sub-wave 1: post-DE matrix-level pre-flight (this file's first classes).
Sub-wave 2: M1 dtype discipline skill amendment.
Sub-wave 3: refuters unconditional-with-fallback + pre-critic assertion.

All offline (Pattern A) — no LLM calls. The live validation run is the
phase's Pattern-B closure.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.causal_data_contract import (
    CausalDataContractError,
    assert_causal_soo_matrix_contract,
)
from src.orchestrator import Orchestrator


PROJECT_ROOT = Path(__file__).parent.parent
SKILLS_ROOT = PROJECT_ROOT / "skills"

RNG = np.random.default_rng(42)


def _spec(task_type: str = "causal_soo") -> dict:
    return {
        "task_type": task_type,
        "treatment": {
            "variable": "X1MTHEFF",
            "operationalization": "median_split_binary",
        },
        "adjustment_set": ["X1SES", "X1TXMTSCOR", "X1SEX"],
    }


def _registry() -> dict:
    return {
        "variables": {
            "predictors": {
                "demographic": [
                    {"name": "X1SES", "type": "continuous"},
                    {"name": "X1SEX", "type": "binary"},
                ],
                "academic": [
                    {"name": "X1TXMTSCOR", "type": "continuous"},
                ],
            }
        }
    }


def _good_matrix(n: int = 600) -> pd.DataFrame:
    """Binary treatment, numeric covariates, healthy overlap."""
    x1 = RNG.normal(size=n)
    x2 = RNG.normal(size=n)
    x3 = RNG.integers(0, 2, size=n)
    # Weak selection: treatment correlated with x1 but far from
    # deterministic -> propensity well inside (0.05, 0.95).
    logits = 0.5 * x1
    t = (RNG.random(n) < 1 / (1 + np.exp(-logits))).astype(int)
    return pd.DataFrame(
        {
            "X1MTHEFF_binary": t,
            "X1SES": x1,
            "X1TXMTSCOR": x2,
            "X1SEX": x3,
        }
    )


def _write(tmp_path: Path, df: pd.DataFrame) -> Path:
    out = tmp_path / "run"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "train_X.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# Sub-wave 2 — M1 dtype discipline reaches the rendered Analyst prompt
# ---------------------------------------------------------------------------


class TestM1DtypeDiscipline:
    """F-3b21.5-M1-DTYPE-ERROR (2-run pattern): the M1 skill now carries
    prescriptive Series/array casting rules; they must reach the live
    Analyst prompt via the orchestrator-equivalent rendering path."""

    @pytest.mark.parametrize(
        "marker",
        [
            "Dtype discipline (MANDATORY",
            "1-D float array, never a DataFrame",
            'apply(pd.to_numeric, errors="raise")',
            'select_dtypes(exclude="number")',
            "Row-align everything through one dropna",
        ],
    )
    def test_dtype_rules_in_rendered_analyst_prompt(self, marker: str) -> None:
        from src.orchestrator import _resolve_skill_caps
        from src.skills import SkillRegistry, format_skills_for_prompt

        registry = SkillRegistry(SKILLS_ROOT)
        matched = registry.match_and_compose(
            stage="Analyst",
            task_type="causal_soo",
            dataset="hsls09_public",
            context="ATT regression adjustment causal",
            top_k_per_layer=_resolve_skill_caps("causal_soo"),
        )
        block = format_skills_for_prompt(matched)
        assert marker in block, (
            f"M1 dtype-discipline marker {marker!r} did not reach the "
            f"rendered Analyst skills block."
        )


# ---------------------------------------------------------------------------
# Sub-wave 3 — refuter-attempt assertion (pcc_c01) + skill contract
# ---------------------------------------------------------------------------


class TestRefuterAttemptAssertion:
    """pcc_c01: the first causal-specific pre-critic check. Absent or
    empty sensitivity.dowhy_refuters -> major failure targeting the
    Analyst (the F-3b23.5 silent-omission shape). Documented failures
    are acceptable; silence is not."""

    def _ctx(self, sensitivity: dict | None) -> SimpleNamespace:
        results: dict[str, Any] = {"estimand": "ATT"}
        if sensitivity is not None:
            results["sensitivity"] = sensitivity
        return SimpleNamespace(
            research_spec={"task_type": "causal_soo"},
            results_object=results,
            data_report={"validation_passed": True},
        )

    def _run(self, sensitivity: dict | None, tmp_path: Path):
        from src.pre_critic_checks import run_pre_critic_checks

        return run_pre_critic_checks(
            self._ctx(sensitivity), str(tmp_path), task_type="causal_soo"
        )

    def _c01(self, result) -> list:
        return [f for f in result.failures if f.check_id == "pcc_c01"]

    def test_absent_refuters_fails(self, tmp_path: Path) -> None:
        result = self._run({"e_value_point": 1.3}, tmp_path)
        fails = self._c01(result)
        assert len(fails) == 1
        assert fails[0].severity == "major"
        assert fails[0].target_agent == "Analyst"
        assert "never attempted" in fails[0].message

    def test_missing_sensitivity_entirely_fails(self, tmp_path: Path) -> None:
        result = self._run(None, tmp_path)
        assert len(self._c01(result)) == 1

    def test_healthy_3b19_shape_passes(self, tmp_path: Path) -> None:
        sensitivity = {
            "dowhy_refuters": {
                "random_common_cause": {
                    "new_effect": 0.0246, "p_value": 1.0,
                    "status": "ran", "error": None,
                },
                "placebo_treatment_refuter": {
                    "new_effect": -1.55e-15, "p_value": 0.0,
                    "status": "ran", "error": None,
                },
            }
        }
        result = self._run(sensitivity, tmp_path)
        assert self._c01(result) == []

    def test_documented_failures_pass(self, tmp_path: Path) -> None:
        """Fallback path: both refuters attempted, both failed WITH
        documented errors — the contract is attempt-and-document."""
        sensitivity = {
            "dowhy_refuters": {
                "random_common_cause": {"status": "failed", "error": "boom"},
                "placebo_treatment_refuter": {"status": "failed", "error": "boom"},
            }
        }
        result = self._run(sensitivity, tmp_path)
        assert self._c01(result) == []

    def test_single_refuter_fails(self, tmp_path: Path) -> None:
        sensitivity = {
            "dowhy_refuters": {
                "random_common_cause": {"status": "ran", "error": None},
            }
        }
        result = self._run(sensitivity, tmp_path)
        assert len(self._c01(result)) == 1
        assert "at least two" in self._c01(result)[0].message

    def test_status_less_entries_fail(self, tmp_path: Path) -> None:
        sensitivity = {
            "dowhy_refuters": {
                "random_common_cause": {"new_effect": 0.1},
                "placebo_treatment_refuter": {"new_effect": 0.0},
            }
        }
        result = self._run(sensitivity, tmp_path)
        assert len(self._c01(result)) == 1
        assert "status" in self._c01(result)[0].message

    def test_prediction_task_unaffected(self, tmp_path: Path) -> None:
        from src.pre_critic_checks import run_pre_critic_checks

        ctx = SimpleNamespace(
            research_spec={"task_type": "prediction"},
            results_object={"all_models": {f"M{i}": {} for i in range(5)}},
            data_report={"validation_passed": True},
        )
        result = run_pre_critic_checks(ctx, str(tmp_path), task_type="prediction")
        assert self._c01(result) == []


class TestRefuterStatusContractInSkill:
    """The status contract must reach the rendered Analyst prompt for
    causal runs (the sensitivity skill is mandatory, so it bypasses the
    formatter cap)."""

    @pytest.mark.parametrize(
        "marker",
        [
            "Refuter execution status contract",
            "invocation is unconditional",
            '"status": "failed"',
            "pcc_c01",
        ],
    )
    def test_contract_in_rendered_analyst_prompt(self, marker: str) -> None:
        from src.orchestrator import _resolve_skill_caps
        from src.skills import SkillRegistry, format_skills_for_prompt

        registry = SkillRegistry(SKILLS_ROOT)
        matched = registry.match_and_compose(
            stage="Analyst",
            task_type="causal_soo",
            dataset="hsls09_public",
            context="sensitivity refuters unmeasured confounding",
            top_k_per_layer=_resolve_skill_caps("causal_soo"),
        )
        block = format_skills_for_prompt(matched)
        assert marker in block


# ---------------------------------------------------------------------------
# Sub-wave 1 — matrix-level contract assertions
# ---------------------------------------------------------------------------


class TestMatrixContractCompliantPasses:
    def test_good_matrix_passes(self, tmp_path: Path) -> None:
        out = _write(tmp_path, _good_matrix())
        assert_causal_soo_matrix_contract(out, _spec(), _registry())

    def test_noop_for_prediction_task(self, tmp_path: Path) -> None:
        """Garbage matrix + prediction task type -> no-op (the contract
        is causal-only; the prediction DE contract is untouched)."""
        df = pd.DataFrame({"anything": ["a", "b", "c"]})
        out = _write(tmp_path, df)
        assert_causal_soo_matrix_contract(out, _spec("prediction"), _registry())


class TestTreatmentBinaryCheck:
    def test_three_valued_treatment_violates(self, tmp_path: Path) -> None:
        df = _good_matrix()
        df.loc[:10, "X1MTHEFF_binary"] = 2
        out = _write(tmp_path, df)
        with pytest.raises(CausalDataContractError, match="distinct non-null"):
            assert_causal_soo_matrix_contract(out, _spec(), _registry())

    def test_single_class_treatment_violates(self, tmp_path: Path) -> None:
        """A split that dropped one treatment class entirely — the
        F-3b17-adjacent degenerate-split shape."""
        df = _good_matrix()
        df["X1MTHEFF_binary"] = 1
        out = _write(tmp_path, df)
        with pytest.raises(CausalDataContractError, match="BOTH classes"):
            assert_causal_soo_matrix_contract(out, _spec(), _registry())


class TestObjectDtypeCheck:
    def test_label_passthrough_violates(self, tmp_path: Path) -> None:
        """Raw label strings = the D1 dispatch was skipped (the HSLS CSV
        stores labeled values; unencoded passthrough is the classic
        failure)."""
        df = _good_matrix()
        df["X1LOCALE"] = RNG.choice(["City", "Suburb", "Rural"], size=len(df))
        out = _write(tmp_path, df)
        with pytest.raises(CausalDataContractError, match="object-dtype"):
            assert_causal_soo_matrix_contract(out, _spec(), _registry())


class TestContinuousDispatchCheck:
    def test_onehot_exploded_continuous_violates(self, tmp_path: Path) -> None:
        """F-3b15-DE-CONTINUOUS-AS-CATEGORICAL: a continuous covariate
        one-hot-expanded into dummies instead of passing through."""
        df = _good_matrix()
        df["X1SES_low"] = (df["X1SES"] < -0.5).astype(int)
        df["X1SES_mid"] = (df["X1SES"].abs() <= 0.5).astype(int)
        df["X1SES_high"] = (df["X1SES"] > 0.5).astype(int)
        df = df.drop(columns=["X1SES"])
        out = _write(tmp_path, df)
        with pytest.raises(
            CausalDataContractError, match="one-hot-expanded"
        ):
            assert_causal_soo_matrix_contract(out, _spec(), _registry())

    def test_present_continuous_column_passes_even_with_dummies(
        self, tmp_path: Path
    ) -> None:
        """Derived helper columns alongside the intact continuous column
        are allowed — only absence + dummies is the violation."""
        df = _good_matrix()
        df["X1SES_squared"] = df["X1SES"] ** 2
        out = _write(tmp_path, df)
        assert_causal_soo_matrix_contract(out, _spec(), _registry())


class TestPositivitySanityCheck:
    def test_near_deterministic_treatment_violates(self, tmp_path: Path) -> None:
        """Treatment (almost) determined by a covariate -> propensity
        mass in the extreme tails >= 10% (the F-3b23.5 shape)."""
        n = 600
        x1 = RNG.normal(size=n)
        t = (x1 > 0).astype(int)
        flip = RNG.random(n) < 0.01
        t[flip] = 1 - t[flip]
        df = pd.DataFrame(
            {
                "X1MTHEFF_binary": t,
                "X1SES": x1,
                "X1TXMTSCOR": RNG.normal(size=n),
                "X1SEX": RNG.integers(0, 2, size=n),
            }
        )
        out = _write(tmp_path, df)
        with pytest.raises(
            CausalDataContractError, match="Propensity-overlap sanity"
        ):
            assert_causal_soo_matrix_contract(out, _spec(), _registry())

    def test_small_matrix_skips_ps_probe(self, tmp_path: Path) -> None:
        """Below the row floor the PS probe is skipped (never crash a
        tiny fixture run); other checks still apply."""
        df = _good_matrix(n=50)
        out = _write(tmp_path, df)
        assert_causal_soo_matrix_contract(out, _spec(), _registry())


class TestSupportedAbortPassThrough:
    """3b.23.7: the verdict evaluator honors an LLM ABORT backed by
    >= 1 critical issue (pre-3b.23.7 the evaluator could never emit
    ABORT, making the orchestrator's ABORT branch dead code and
    silently disabling the SPEC §8 safety valve)."""

    def _review(self, llm_verdict: str, n_critical: int, quality: int = 2) -> dict:
        issues = [
            {
                "severity": "critical",
                "category": "leakage",
                "description": "d",
                "recommendation": "r",
                "target_agent": "DataEngineer",
            }
            for _ in range(n_critical)
        ]
        return {
            "overall_verdict": llm_verdict,
            "overall_quality_score": quality,
            "problem_formulation_review": {"score": quality, "issues": issues},
            "data_preparation_review": {"score": quality, "issues": []},
            "analysis_review": {"score": quality, "issues": []},
            "substantive_review": {
                "score": quality,
                "educational_meaningfulness": "m",
                "issues": [],
            },
            "revision_instructions": {
                "ProblemFormulator": None,
                "DataEngineer": None,
                "Analyst": None,
            },
        }

    def test_supported_abort_is_honored(self) -> None:
        from src.agents.verdict_evaluator import evaluate_critic_verdict

        result = evaluate_critic_verdict(
            self._review("ABORT", n_critical=1),
            revision_cycle=0,
            max_revision_cycles=2,
        )
        assert result.verdict == "ABORT"
        assert result.unverified is False

    def test_unsupported_abort_is_downgraded(self) -> None:
        """LLM ABORT with ZERO critical issues stays overridden — the
        no-invented-issues philosophy."""
        from src.agents.verdict_evaluator import evaluate_critic_verdict

        result = evaluate_critic_verdict(
            self._review("ABORT", n_critical=0, quality=8),
            revision_cycle=0,
            max_revision_cycles=2,
        )
        assert result.verdict == "PASS"

    def test_cycles_exhaustion_does_not_rescue_abort(self) -> None:
        """ABORT is about unfixable flaws; the UNVERIFIED downgrade
        applies to REVISE only."""
        from src.agents.verdict_evaluator import evaluate_critic_verdict

        result = evaluate_critic_verdict(
            self._review("ABORT", n_critical=2),
            revision_cycle=2,
            max_revision_cycles=2,
        )
        assert result.verdict == "ABORT"
        assert result.unverified is False


class TestDummiedTreatmentRepair:
    """sw1c (added after attempt 2): the DE one-hot-encoded the binary
    treatment (X1MTHEFF_binary_0/_1) and repeated it on the targeted
    retry. The pre-flight now deterministically collapses a
    complementary dummy pair back to the single column."""

    def _dummied(self, n: int = 300) -> pd.DataFrame:
        t = RNG.integers(0, 2, size=n)
        return pd.DataFrame(
            {
                "X1MTHEFF_binary_0": (1 - t),
                "X1MTHEFF_binary_1": t,
                "X1SES": RNG.normal(size=n),
                "X1TXMTSCOR": RNG.normal(size=n),
            }
        )

    def test_complementary_pair_is_repaired(self, tmp_path: Path) -> None:
        from src.causal_data_contract import repair_dummied_treatment

        self._dummied().to_csv(tmp_path / "train_X.csv", index=False)
        self._dummied().to_csv(tmp_path / "test_X.csv", index=False)
        note = repair_dummied_treatment(tmp_path, _spec())
        assert note is not None and "X1MTHEFF_binary" in note
        for csv_name in ("train_X.csv", "test_X.csv"):
            df = pd.read_csv(tmp_path / csv_name)
            assert "X1MTHEFF_binary" in df.columns
            assert "X1MTHEFF_binary_0" not in df.columns
            assert "X1MTHEFF_binary_1" not in df.columns
            assert set(df["X1MTHEFF_binary"].unique()) <= {0, 1}

    def test_repaired_matrix_passes_contract(self, tmp_path: Path) -> None:
        from src.causal_data_contract import (
            assert_causal_soo_data_contract,
            repair_dummied_treatment,
        )

        self._dummied(600).to_csv(tmp_path / "train_X.csv", index=False)
        repair_dummied_treatment(tmp_path, _spec())
        assert_causal_soo_data_contract(tmp_path / "train_X.csv", _spec())

    def test_non_complementary_pair_not_repaired(self, tmp_path: Path) -> None:
        """A pair that is NOT one-hot-complementary must be left alone
        (the repair can never invent data)."""
        from src.causal_data_contract import repair_dummied_treatment

        df = self._dummied()
        df.loc[:20, "X1MTHEFF_binary_0"] = 1  # rows with both == 1
        df.to_csv(tmp_path / "train_X.csv", index=False)
        assert repair_dummied_treatment(tmp_path, _spec()) is None

    def test_noop_for_prediction(self, tmp_path: Path) -> None:
        from src.causal_data_contract import repair_dummied_treatment

        self._dummied().to_csv(tmp_path / "train_X.csv", index=False)
        assert repair_dummied_treatment(tmp_path, _spec("prediction")) is None


class TestOrchestratorPreflightWiring:
    """The retry glue in Orchestrator._run_engineering, exercised via
    unbound calls on a stub 'self' (no LLM, no real agents)."""

    def _fake_self(self, tmp_path: Path) -> SimpleNamespace:
        ctx = SimpleNamespace(
            research_spec=_spec(),
            output_dir=str(tmp_path),
            data_report=None,
            completed_stages=[],
            current_state=None,
            errors=[],
        )
        fake = SimpleNamespace(
            ctx=ctx,
            _log=MagicMock(),
            _abort=MagicMock(),
            _save_checkpoint=MagicMock(),
            _check_cost=MagicMock(),
            _inject_skills=MagicMock(),
            data_engineer=MagicMock(),
        )
        return fake

    def test_preflight_returns_none_for_prediction(self, tmp_path: Path) -> None:
        fake = self._fake_self(tmp_path)
        fake.ctx.research_spec = {"task_type": "prediction"}
        assert Orchestrator._run_post_de_preflight(fake) is None

    def test_preflight_reports_violation_message(self, tmp_path: Path) -> None:
        df = _good_matrix()
        df["X1MTHEFF_binary"] = 1  # single-class treatment
        df.to_csv(tmp_path / "train_X.csv", index=False)
        fake = self._fake_self(tmp_path)
        fake.data_engineer.load_registry.return_value = _registry()
        violation = Orchestrator._run_post_de_preflight(fake)
        assert violation is not None and "BOTH classes" in violation

    def test_preflight_passes_good_matrix(self, tmp_path: Path) -> None:
        _good_matrix().to_csv(tmp_path / "train_X.csv", index=False)
        fake = self._fake_self(tmp_path)
        fake.data_engineer.load_registry.return_value = _registry()
        assert Orchestrator._run_post_de_preflight(fake) is None

    def test_probe_error_is_nonfatal(self, tmp_path: Path) -> None:
        """A crashed probe (e.g. unreadable CSV) must be treated as a
        pass, not kill a healthy run."""
        fake = self._fake_self(tmp_path)  # no train_X.csv written at all
        fake.data_engineer.load_registry.return_value = _registry()
        assert Orchestrator._run_post_de_preflight(fake) is None
        assert fake._log.called

    def test_engineering_retry_flow(self, tmp_path: Path) -> None:
        """Violation on first pre-flight -> ONE targeted DE retry with
        the violation text injected -> second pre-flight clean -> stage
        completes without abort."""
        _good_matrix().to_csv(tmp_path / "train_X.csv", index=False)
        fake = self._fake_self(tmp_path)
        fake.data_engineer.run.return_value = {
            "validation_passed": True,
            "analytic_n": 15000,
        }
        fake._run_post_de_preflight = MagicMock(
            side_effect=["object-dtype violation XYZ", None]
        )
        Orchestrator._run_engineering(fake)

        assert fake._abort.call_count == 0
        # DE ran twice: initial + targeted retry.
        assert fake.data_engineer.run.call_count == 2
        retry_kwargs = fake.data_engineer.run.call_args_list[1].kwargs
        assert "object-dtype violation XYZ" in retry_kwargs.get(
            "revision_instructions", ""
        )
        assert "ENGINEERING" in fake.ctx.completed_stages

    def test_validation_failure_gets_targeted_retry(
        self, tmp_path: Path
    ) -> None:
        """sw1b (added after the first H1 validation run aborted on the
        F-3b17 NaN-cells shape): validation_passed=False earns ONE
        targeted retry with the warnings injected; a passing retry
        completes the stage."""
        _good_matrix().to_csv(tmp_path / "train_X.csv", index=False)
        fake = self._fake_self(tmp_path)
        fake.data_engineer.run.side_effect = [
            {
                "validation_passed": False,
                "warnings": ["NaN values remain in train_X: 312840 cells"],
            },
            {"validation_passed": True, "analytic_n": 15000},
        ]
        fake._run_post_de_preflight = MagicMock(return_value=None)
        Orchestrator._run_engineering(fake)

        assert fake._abort.call_count == 0
        assert fake.data_engineer.run.call_count == 2
        retry_kwargs = fake.data_engineer.run.call_args_list[1].kwargs
        assert "NaN values remain" in retry_kwargs.get(
            "revision_instructions", ""
        )
        assert "ENGINEERING" in fake.ctx.completed_stages

    def test_validation_failure_twice_aborts(self, tmp_path: Path) -> None:
        fake = self._fake_self(tmp_path)
        fake.data_engineer.run.side_effect = [
            {"validation_passed": False, "warnings": ["NaN cells"]},
            {"validation_passed": False, "warnings": ["NaN cells"]},
        ]
        Orchestrator._run_engineering(fake)
        assert fake._abort.call_count == 1
        assert "validation retry exhausted" in fake._abort.call_args.args[0]

    def test_engineering_aborts_after_second_violation(
        self, tmp_path: Path
    ) -> None:
        _good_matrix().to_csv(tmp_path / "train_X.csv", index=False)
        fake = self._fake_self(tmp_path)
        fake.data_engineer.run.return_value = {
            "validation_passed": True,
            "analytic_n": 15000,
        }
        fake._run_post_de_preflight = MagicMock(
            side_effect=["violation A", "violation B"]
        )
        Orchestrator._run_engineering(fake)
        assert fake._abort.call_count == 1
        assert "post-retry" in fake._abort.call_args.args[0]
