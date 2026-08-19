"""V3.0 Phase 3b.12 / §12.2 — Orchestrator fail-fast guardrail tests.

Asserts the new ``CausalDataContractError`` and
``assert_causal_soo_data_contract`` behave per §12.2.3:

  1. Happy path: treatment column present → no raise.
  2. Operationalized name accepted (e.g. ``X1MTHEFF_binary`` for
     ``median_split_binary``).
  3. F-3b11 case: treatment dropped, only adjustment_set + outcome
     present → CausalDataContractError citing the missing column name.
  4. Prediction-task no-op: assertion does not trigger when
     ``task_type != 'causal_soo'``.
  5. Error message cites the new skill so future debug sessions can
     find the contract source via grep / IDE search.
  6. Wire-up: the orchestrator's ``_run_engineering`` actually calls
     the assertion before transitioning to ANALYZING (without this
     test the helper could exist in code and silently never run —
     the same bug class as F-3b7-FORMATTER-TRUNCATES-METHOD-SKILLS:
     "matcher returns the right answer, but the formatter never asks").
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.causal_data_contract import (
    CausalDataContractError,
    assert_causal_soo_data_contract,
)


def _write_fake_train_X(tmp_path: Path, columns: list[str]) -> Path:
    """Write a 1-row CSV with the given column header to a tmp path.
    The guardrail only reads the header (``nrows=0``), so 1 dummy row
    is sufficient for the test."""
    df = pd.DataFrame([{c: 0 for c in columns}])
    csv_path = tmp_path / "train_X.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


# ---------------------------------------------------------------------------
# §12.2.3 (1) — happy path: treatment column present
# ---------------------------------------------------------------------------


class TestGuardrailHappyPath:
    def test_passes_when_treatment_column_present(
        self, tmp_path: Path,
    ) -> None:
        spec = {
            "task_type": "causal_soo",
            "treatment": {"variable": "X1MTHEFF"},
        }
        csv = _write_fake_train_X(
            tmp_path,
            columns=["X1MTHEFF", "X1RACE_1", "X1RACE_2", "X1SES"],
        )
        # Should not raise
        assert_causal_soo_data_contract(csv, spec)


# ---------------------------------------------------------------------------
# §12.2.3 (2) — operationalized name accepted
# ---------------------------------------------------------------------------


class TestGuardrailOperationalization:
    def test_accepts_median_split_binary_form(
        self, tmp_path: Path,
    ) -> None:
        """median_split_binary → X1MTHEFF_binary is acceptable."""
        spec = {
            "task_type": "causal_soo",
            "treatment": {
                "variable": "X1MTHEFF",
                "operationalization": "median_split_binary",
            },
        }
        csv = _write_fake_train_X(
            tmp_path,
            columns=["X1MTHEFF_binary", "X1RACE_1", "X1SES"],
        )
        # Original name absent but operationalized form present → ok
        assert_causal_soo_data_contract(csv, spec)

    def test_accepts_original_name_under_operationalization(
        self, tmp_path: Path,
    ) -> None:
        """If DE retains the raw column even under an operationalization,
        that's still acceptable — the contract requires SOMETHING to
        identify the treatment, not the binarized form specifically."""
        spec = {
            "task_type": "causal_soo",
            "treatment": {
                "variable": "X1MTHEFF",
                "operationalization": "median_split_binary",
            },
        }
        csv = _write_fake_train_X(
            tmp_path,
            columns=["X1MTHEFF", "X1RACE_1"],
        )
        assert_causal_soo_data_contract(csv, spec)

    def test_accepts_quartile_split_form(
        self, tmp_path: Path,
    ) -> None:
        spec = {
            "task_type": "causal_soo",
            "treatment": {
                "variable": "X1MTHEFF",
                "operationalization": "quartile_split",
            },
        }
        csv = _write_fake_train_X(
            tmp_path,
            columns=["X1MTHEFF_quartile", "X1RACE_1"],
        )
        assert_causal_soo_data_contract(csv, spec)


# ---------------------------------------------------------------------------
# §12.2.3 (3) — F-3b11 case: treatment dropped → raise
# ---------------------------------------------------------------------------


class TestGuardrailRaisesOnMissingTreatment:
    def test_raises_when_treatment_column_missing(
        self, tmp_path: Path,
    ) -> None:
        """Reproduce the 3b.11 failure case: train_X.csv contains the
        adjustment_set but the treatment column was dropped."""
        spec = {
            "task_type": "causal_soo",
            "treatment": {"variable": "X1MTHEFF"},
        }
        csv = _write_fake_train_X(
            tmp_path,
            columns=["X1MTHID", "X1MTHUTI", "X1RACE_1", "X1SES"],
        )
        with pytest.raises(CausalDataContractError, match="X1MTHEFF"):
            assert_causal_soo_data_contract(csv, spec)

    def test_raises_when_only_adjustment_set_present(
        self, tmp_path: Path,
    ) -> None:
        """Same shape as F-3b11 but with the locked-spec adjustment_set."""
        spec = {
            "task_type": "causal_soo",
            "treatment": {"variable": "X1MTHEFF"},
        }
        csv = _write_fake_train_X(
            tmp_path,
            columns=[
                "X1TXMTSCOR", "X1SES", "X1RACE_1", "X1RACE_2", "X1SEX_1",
                "X1PAREDU_1", "X1STUEDEXPCT_1", "X1MTHID", "X1MTHUTI",
                "X1SCHOOLBEL", "X1SCHOOLENG", "X1LOCALE_1", "X1CONTROL_1",
            ],
        )
        with pytest.raises(CausalDataContractError) as excinfo:
            assert_causal_soo_data_contract(csv, spec)
        # Error mentions the expected treatment
        assert "X1MTHEFF" in str(excinfo.value)


# ---------------------------------------------------------------------------
# §12.2.3 (4) — prediction-task no-op
# ---------------------------------------------------------------------------


class TestGuardrailNoopForPrediction:
    def test_noop_for_prediction_task(self, tmp_path: Path) -> None:
        """Regression: prediction tasks must not be affected. The
        prediction-task DE contract is unchanged by 3b.12."""
        spec = {
            "task_type": "prediction",
            "target": "X4EVRATNDCLG",
            # Prediction specs typically have no 'treatment' field at all.
        }
        csv = _write_fake_train_X(
            tmp_path,
            columns=["X1RACE_1", "X1MTHID", "X1SES"],
        )
        # Must not raise even though there's no "treatment" column
        # because the assertion is no-op for non-causal task types.
        assert_causal_soo_data_contract(csv, spec)

    def test_noop_when_task_type_missing(self, tmp_path: Path) -> None:
        """If task_type is absent (legacy spec), the guardrail still
        no-ops. Defensive — never raise on something other than the
        documented causal_soo case."""
        spec = {"target": "X4EVRATNDCLG"}
        csv = _write_fake_train_X(tmp_path, columns=["X1RACE_1", "X1SES"])
        assert_causal_soo_data_contract(csv, spec)

    def test_noop_when_treatment_field_missing(
        self, tmp_path: Path,
    ) -> None:
        """If task_type=causal_soo but the spec has no treatment field,
        the guardrail bails silently. That's a spec-validation problem,
        not a DE-output problem — surfaced upstream of this helper."""
        spec = {"task_type": "causal_soo"}
        csv = _write_fake_train_X(tmp_path, columns=["X1RACE_1"])
        # Should not raise — no treatment to check
        assert_causal_soo_data_contract(csv, spec)


# ---------------------------------------------------------------------------
# §12.2.3 (5) — error message cites the skill
# ---------------------------------------------------------------------------


class TestGuardrailErrorMessageCitesSkill:
    def test_error_message_cites_skill_name(self, tmp_path: Path) -> None:
        """The guardrail's error message must reference the
        causal-data-engineer-contract skill so a debug session
        starting from a stack trace can find the contract source."""
        spec = {
            "task_type": "causal_soo",
            "treatment": {"variable": "X1MTHEFF"},
        }
        csv = _write_fake_train_X(tmp_path, columns=["X1MTHID"])
        with pytest.raises(
            CausalDataContractError, match="causal-data-engineer-contract"
        ):
            assert_causal_soo_data_contract(csv, spec)

    def test_error_message_lists_actual_columns(
        self, tmp_path: Path,
    ) -> None:
        """The message should help debug by including a sample of the
        actual columns the DE produced."""
        spec = {
            "task_type": "causal_soo",
            "treatment": {"variable": "X1MTHEFF"},
        }
        csv = _write_fake_train_X(
            tmp_path, columns=["X1MTHID", "X1MTHUTI", "X1RACE_1"],
        )
        with pytest.raises(CausalDataContractError) as excinfo:
            assert_causal_soo_data_contract(csv, spec)
        msg = str(excinfo.value)
        # Confirms the error tells you what WAS there, not just what wasn't.
        assert "X1MTHID" in msg


# ---------------------------------------------------------------------------
# §12.2.3 (6) — orchestrator wire-up integration test
# ---------------------------------------------------------------------------


class TestOrchestratorCallsGuardrail:
    """If the assertion exists in code but ``_run_engineering`` doesn't
    invoke it, the guardrail is inert. This integration test reads the
    orchestrator's ENGINEERING stage runner and confirms the call site
    is wired up.

    Two layers of assertion: (a) the source-level wire-up (the import +
    call appear in src/orchestrator.py), and (b) a behavioral test
    that mocks the DE to produce a contract-violating CSV and confirms
    the orchestrator transitions to ABORTED rather than ANALYZING.
    """

    def test_orchestrator_imports_guardrail_helpers(self) -> None:
        """Source-level wire-up check: orchestrator.py imports
        ``CausalDataContractError`` and ``assert_causal_soo_data_contract``
        from ``src.causal_data_contract``. Catches the case where the
        helpers exist but are never imported by the runtime path."""
        import src.orchestrator as orch

        assert hasattr(orch, "CausalDataContractError"), (
            "orchestrator.py must import CausalDataContractError from "
            "src.causal_data_contract; the guardrail is inert without "
            "the import"
        )
        assert hasattr(orch, "assert_causal_soo_data_contract"), (
            "orchestrator.py must import assert_causal_soo_data_contract "
            "from src.causal_data_contract; the guardrail is inert "
            "without the import"
        )

    def test_orchestrator_run_engineering_invokes_guardrail(self) -> None:
        """Source-level wire-up check via inspect. Post-3b.23.7 the
        guardrail call sites live in ``_run_post_de_preflight`` (header
        check + matrix-level checks), which ``_run_engineering`` invokes
        for the violation-then-targeted-retry flow. If a future refactor
        removes either link in that chain, this test fails loudly."""
        import inspect
        import src.orchestrator as orch

        engineering_src = inspect.getsource(orch.Orchestrator._run_engineering)
        assert "_run_post_de_preflight" in engineering_src, (
            "_run_engineering must invoke _run_post_de_preflight after "
            "the DataEngineer stage; the guardrail is inert otherwise."
        )
        preflight_src = inspect.getsource(
            orch.Orchestrator._run_post_de_preflight
        )
        assert "assert_causal_soo_data_contract" in preflight_src
        assert "assert_causal_soo_matrix_contract" in preflight_src, (
            "_run_post_de_preflight must run the 3b.23.7 matrix-level "
            "checks alongside the 3b.12 header check."
        )

    def test_orchestrator_aborts_when_guardrail_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Behavioral integration test: mock the DataEngineer to produce
        a contract-violating train_X.csv (treatment column missing).
        The orchestrator's _run_engineering must catch the
        CausalDataContractError and transition to ABORTED; the Analyst
        must never run.

        This is the test that would have flagged the F-3b11 failure
        runtime-deterministically had the guardrail existed in 3b.11.
        """
        from src.context import PipelineContext, PipelineState
        from src.orchestrator import Orchestrator

        # Build a minimal context where the DE has "produced" a CSV
        # missing the treatment column.
        output_dir = tmp_path / "run"
        output_dir.mkdir()
        (output_dir / "train_X.csv").write_text(
            "X1MTHID,X1RACE_1,X1SES\n0,1,0.5\n", encoding="utf-8"
        )

        ctx = PipelineContext(
            dataset_name="hsls09_public",
            raw_data_path=str(tmp_path / "fake_data.csv"),
            output_dir=str(output_dir),
            task_type="causal_soo",
        )
        ctx.research_spec = {
            "task_type": "causal_soo",
            "treatment": {"variable": "X1MTHEFF"},
            "outcome": {"variable": "X4EVRATNDCLG"},
            "adjustment_set": ["X1MTHID", "X1RACE", "X1SES"],
        }

        # Stub the agent constructors so __init__ doesn't try to read
        # API keys. We patch BaseAgent.__init__ via the Orchestrator's
        # agent attributes after instantiation.
        config = {
            "llm_provider": "anthropic",
            "models": {
                "problem_formulator": "claude-sonnet-4-6",
                "data_engineer": "claude-sonnet-4-6",
                "analyst": "claude-sonnet-4-6",
                "critic": "claude-opus-4-6",
                "writer": "claude-sonnet-4-6",
            },
            "paths": {
                "data_registry": "data_registry/",
                "agent_prompts": "agent_prompts/",
                "output_base": str(tmp_path),
            },
            "sandbox": {"enabled": False},
            "pipeline": {
                "max_revision_cycles": 1,
                "random_state": 42,
                "cost_budget_usd": 100.0,
            },
            "review_gate": {"enabled": False},
        }

        # Set fake API keys so BaseAgent doesn't blow up.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake-3b12-guardrail-test")

        # Patch the DataEngineer.run to produce a "validated" data_report
        # without actually running. The CSV is already on disk above.
        with patch("anthropic.Anthropic"), \
             patch("src.agents.base.load_prompt", return_value={
                 "system_prompt": "stub", "temperature": 0.0,
             }), \
             patch("src.sandbox.create_executor", return_value=object()):
            orch = Orchestrator(ctx, config, config_path="config.yaml")

        # Mock DE.run to return a passing data_report; the CSV on disk
        # is the contract-violating one we wrote above.
        orch.data_engineer.run = MagicMock(return_value={
            "validation_passed": True,
            "analytic_n": 13773,
            "warnings": [],
        })
        # Stub _inject_skills to avoid touching the registry.
        orch._inject_skills = MagicMock()

        # Pre-flight: state should not yet be ABORTED.
        assert ctx.current_state != PipelineState.ABORTED

        # Run the engineering stage. The post-DE guardrail should fire.
        orch._run_engineering()

        # The orchestrator must have aborted because the contract failed.
        assert ctx.current_state == PipelineState.ABORTED, (
            f"Expected ABORTED after contract violation; got "
            f"{ctx.current_state}. The guardrail did not halt the "
            f"pipeline — F-3b11 would recur silently."
        )
        # The error message must reference the contract violation.
        assert any(
            "causal data contract" in str(e).lower() or "X1MTHEFF" in str(e)
            for e in ctx.errors
        ), f"errors list does not cite the contract violation: {ctx.errors}"
        # ENGINEERING must NOT be in completed_stages (we aborted before
        # checkpointing it).
        assert "ENGINEERING" not in ctx.completed_stages
