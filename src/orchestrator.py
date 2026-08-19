from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from src.agents.analyst import Analyst
from src.agents.base import BaseAgent
from src.agents.critic import Critic
from src.agents.data_engineer import DataEngineer
from src.agents.problem_formulator import ProblemFormulator
from src.agents.writer import Writer
from src.causal_data_contract import (
    CausalDataContractError,
    assert_causal_soo_data_contract,
    assert_causal_soo_matrix_contract,
    repair_dummied_treatment,
)
from src.context import PipelineContext, PipelineState
from src.dataset_adapter import create_dataset_adapter
from src.findings_memory import FindingsMemory, RunEntry
from src.pre_critic_checks import PreCriticResult, run_pre_critic_checks
from src.review_gate import ReviewGate
from src.sandbox import compile_latex, create_executor
from src.skills import Skill, SkillRegistry
from src.task_template import create_task_template

# Default per-layer caps for skill matching, sized so injected content
# stays comfortably under typical max-tokens budgets even when several
# skills compose via references.
_DEFAULT_SKILL_CAPS: dict[str, int] = {
    "task-type": 3,
    "dataset": 4,
    "methodology": 5,
    "writing": 5,
}

# Task-type-specific overrides (Phase 3b.6 / 6.1). Causal_soo runs need
# ALL FIVE M-skills (M1-M5: regression-adjustment, PSM, IPW, AIPW/TMLE,
# causal-forest) attached at the Analyst stage alongside G1-G5 + D1.
# That's 11 causal skills + ~4 generic methodology references = 15+;
# the prediction cap of 5 is too tight. Raising to 12 fits the 11
# causal-specific without overflowing the prompt budget.
_SKILL_CAPS_BY_TASK_TYPE: dict[str, dict[str, int]] = {
    "causal_soo": {
        "task-type": 3,
        "dataset": 4,
        "methodology": 12,
        "writing": 5,
    },
    # V3.1 Arc R: ITR inherits the causal budget; methodology cap +2
    # for the M6/M7 additions on top of the G-family + M5.
    "causal_itr": {
        "task-type": 3,
        "dataset": 4,
        "methodology": 14,
        "writing": 5,
    },
    "causal_did": {
        "task-type": 3,
        "dataset": 4,
        "methodology": 10,
        "writing": 5,
    },
    "psychometrics": {
        "task-type": 3,
        "dataset": 4,
        "methodology": 10,
        "writing": 5,
    },
}


def _resolve_skill_caps(task_type: str) -> dict[str, int]:
    """Return the per-layer skill cap for a given task type.

    Falls back to ``_DEFAULT_SKILL_CAPS`` when no task-type-specific
    override exists; this preserves byte-identical behavior for the
    prediction codepath.
    """
    return _SKILL_CAPS_BY_TASK_TYPE.get(task_type, _DEFAULT_SKILL_CAPS)


class Orchestrator:
    def __init__(
        self,
        ctx: PipelineContext,
        config: dict,
        config_path: str = "config.yaml",
    ) -> None:
        self.ctx = ctx
        self.config = config
        self._config_path = config_path
        self._user_prompt: Optional[str] = None

        os.makedirs(ctx.output_dir, exist_ok=True)

        # V4 psychometrics: executor subprocesses import the copied
        # r_bridge.py flat; give them a deterministic path to the
        # certified R scripts (inherited via os.environ).
        r_helpers = Path("r_helpers").resolve()
        if r_helpers.is_dir():
            os.environ.setdefault("EDM_ARS_R_HELPERS", str(r_helpers))

        # Copy config snapshot for reproducibility
        if os.path.exists(config_path):
            shutil.copy(
                config_path,
                os.path.join(ctx.output_dir, "config_snapshot.yaml"),
            )

        # Create task template and dataset adapter
        self.task_template = create_task_template(ctx.task_type)
        self.dataset_adapter = create_dataset_adapter(ctx.dataset_name)

        # V2.0 skill registry: load all SKILL.md files under skills/.
        # Inert during the transition (agents whose system prompts have
        # no {{SKILLS}} placeholder fall through to the original prompt).
        self.skill_registry = SkillRegistry(skills_root=Path("skills"))

        # Load findings memory if enabled (non-fatal on failure)
        self.findings_memory: FindingsMemory | None = None
        self._pending_memory_warning: str | None = None
        fm_cfg = config.get("findings_memory", {})
        if fm_cfg.get("enabled", False):
            try:
                mem_path = fm_cfg.get("path", "findings_memory/memory.yaml")
                self.findings_memory = FindingsMemory.load(mem_path)
            except Exception as exc:
                self.findings_memory = None
                # Log after executor/agents are set up — deferred to after __init__
                self._pending_memory_warning = f"FindingsMemory load failed (non-fatal): {exc}"

        # Create shared executor (Docker sandbox or subprocess fallback)
        self._executor = create_executor(config)
        executor_type = type(self._executor).__name__

        # Instantiate all agents (share ctx reference, executor, template, and adapter)
        # Type the dict as Any: mypy otherwise infers a narrow value type
        # that conflicts with BaseAgent's `skills: list[Skill] | None` parameter
        # added in Phase 2c (even though we never pass skills via kwargs here).
        agent_kwargs: dict[str, Any] = dict(
            executor=self._executor,
            task_template=self.task_template,
            dataset_adapter=self.dataset_adapter,
        )
        self.problem_formulator = ProblemFormulator(ctx, "problem_formulator", config, **agent_kwargs)
        self.data_engineer = DataEngineer(ctx, "data_engineer", config, **agent_kwargs)
        self.analyst = Analyst(ctx, "analyst", config, **agent_kwargs)
        self.critic = Critic(ctx, "critic", config, **agent_kwargs)
        self.writer = Writer(ctx, "writer", config, **agent_kwargs)

        # Resume from checkpoint if present
        self._load_checkpoint()
        self._log("Orchestrator", f"Code executor: {executor_type}")
        if self._pending_memory_warning:
            self._log("Orchestrator", self._pending_memory_warning)

    # ------------------------------------------------------------------
    # V2.0 skill injection helpers
    # ------------------------------------------------------------------

    def _stage_context_string(self) -> str:
        """Free-text context for keyword-based skill ranking.

        Prefers the research question (post-formulation) and falls back to
        the original user prompt. Returns empty string if neither is
        available, in which case the matcher falls back to priority-only
        ranking.
        """
        venue_hint = ""
        if self.config.get("writer", {}).get("venue_format") == "journal":
            venue_hint = " journal manuscript APA"
        if self.ctx.research_spec:
            q = self.ctx.research_spec.get("research_question") or ""
            if q:
                return q + venue_hint
        return (self._user_prompt or "") + venue_hint

    def _match_skills_for_stage(self, agent_name: str) -> list[Skill]:
        """Return the composed skill list for one stage (testable helper)."""
        return self.skill_registry.match_and_compose(
            stage=agent_name,
            task_type=self.ctx.task_type,
            dataset=self.ctx.dataset_name,
            context=self._stage_context_string(),
            top_k_per_layer=_resolve_skill_caps(self.ctx.task_type),
        )

    def _inject_skills(self, agent: BaseAgent, agent_name: str) -> None:
        """Match + attach skills to an agent immediately before invoking it."""
        agent.skills = self._match_skills_for_stage(agent_name)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, user_prompt: Optional[str] = None) -> PipelineContext:
        self._user_prompt = user_prompt
        while True:
            state = self.ctx.current_state
            if state in (PipelineState.INITIALIZED, PipelineState.FORMULATING):
                self._run_formulating()
            elif state == PipelineState.ENGINEERING:
                self._run_engineering()
            elif state == PipelineState.ANALYZING:
                self._run_analyzing()
            elif state == PipelineState.CRITIQUING:
                self._run_critiquing()
            elif state == PipelineState.REVISING:
                self._run_revising()
            elif state == PipelineState.WRITING:
                self._run_writing()
            elif state == PipelineState.REVIEWING:
                self._run_reviewing()
            elif state in (PipelineState.COMPLETED, PipelineState.ABORTED):
                break
            else:
                self._log("Orchestrator", f"Unknown state: {state}. Aborting.")
                self.ctx.current_state = PipelineState.ABORTED
                break
        self._write_cost_summary()
        return self.ctx

    def _write_cost_summary(self) -> None:
        """Aggregate the run's measured token usage into run_cost.json (K1).

        Runs on BOTH terminal states — an aborted run still spent money,
        and a cost record that only exists for successes understates the
        real cost of operating the system.
        """
        try:
            from src.cost import write_summary

            payload = write_summary(self.ctx.output_dir, self.config)
            if not payload:
                return
            cost = payload.get("cost_usd")
            cost_str = "not priced" if cost is None else f"${cost:.4f}"
            self._log(
                "Orchestrator",
                f"Run cost: {cost_str} over {payload['n_calls']} LLM calls "
                f"({payload['prompt_tokens']:,} in / "
                f"{payload['completion_tokens']:,} out; "
                f"{payload['cached_prompt_tokens']:,} cached) "
                "-> run_cost.json",
            )
        except Exception as exc:  # noqa: BLE001 — accounting is never fatal
            self._log("Orchestrator", f"Cost summary skipped: {exc}")

    # ------------------------------------------------------------------
    # Stage runners
    # ------------------------------------------------------------------

    def _run_formulating(self) -> None:
        self.ctx.current_state = PipelineState.FORMULATING
        if "FORMULATING" in self.ctx.completed_stages:
            self.ctx.current_state = PipelineState.ENGINEERING
            return
        self._log("Orchestrator", "Starting FORMULATING stage")
        try:
            fm_cfg = self.config.get("findings_memory", {})
            n_branches = (
                fm_cfg.get("n_candidate_specs", 1)
                if fm_cfg.get("enabled", False) and self.findings_memory is not None
                else 1
            )
            memory_summary = (
                self.findings_memory.to_summary_str()
                if self.findings_memory is not None
                else ""
            )
            studied_outcomes = (
                self.findings_memory.get_studied_outcomes()
                if self.findings_memory is not None
                else []
            )

            self._inject_skills(self.problem_formulator, "ProblemFormulator")
            result = self.problem_formulator.run(
                user_prompt=self._user_prompt,
                findings_memory_summary=memory_summary,
                n_candidate_specs=n_branches,
                studied_outcomes=studied_outcomes,
                # Phase 3b.4 / B6: pass locked spec when CLI provided one.
                # PF currently ignores via **kwargs; sub-wave 2 introduces
                # the causal "refine" branch that consumes this kwarg.
                locked_research_spec=self.ctx.locked_research_spec,
            )
            self.ctx.research_spec = result.get("research_spec")
            self.ctx.literature_context = result.get("literature_context")
            self.ctx.retrieved_literature = result.get("retrieved_literature")
            self._save_formulating_outputs()
            self.ctx.completed_stages.append("FORMULATING")
            self.ctx.current_state = PipelineState.ENGINEERING
            self._log("Orchestrator", "FORMULATING stage complete")
            self._save_checkpoint()
            self._check_cost()
        except Exception as e:
            self._abort(f"FORMULATING failed: {e}")

    def _run_post_de_preflight(self) -> str | None:
        """Run the causal-mode post-DE contract checks.

        Returns the violation message (str) or None when compliant.
        No-op (returns None) for non-causal task types and when the
        research spec is absent. Unexpected probe errors are logged and
        treated as non-violations — the pre-flight must never be the
        thing that breaks a healthy run.
        """
        spec = self.ctx.research_spec or {}

        # V4 Phase A (F-A1-ELS-EMPTY-TEST-SPLIT): task-type-agnostic split
        # sanity. On the first ELS run, HSLS-specific school-fingerprint
        # reconstruction degenerated to ONE pseudo-school, the school-aware
        # splitter put every row in train, and DE self-reported
        # validation_passed=True with n_test=0 — every model then failed
        # downstream. Check the FILE, not just the self-report. causal_did
        # runs estimate on the full panel (no split) — exempt.
        if spec.get("task_type") not in ("causal_did", "psychometrics"):
            dr = self.ctx.data_report or {}
            analytic_n = int(dr.get("analytic_n") or 0)
            n_test = int(dr.get("n_test") or 0)
            floor = max(1, int(0.15 * analytic_n))
            # Check the FILE too when it exists (a self-report can lie);
            # absent file -> report-only check (stubbed test runs).
            test_path = Path(self.ctx.output_dir) / "test_X.csv"
            test_rows = floor
            if test_path.exists():
                try:
                    import pandas as _pd

                    test_rows = len(_pd.read_csv(test_path))
                except Exception:
                    test_rows = 0
            if analytic_n > 0 and (n_test < floor or test_rows < floor):
                return (
                    f"Test split is degenerate: n_test={n_test}, actual "
                    f"test_X.csv rows={test_rows}, analytic_n={analytic_n} "
                    "(SPEC requires a stratified 20% test set). If "
                    "school-aware splitting produced fewer than 10 school "
                    "groups, or school IDs are unavailable/suppressed on "
                    "this dataset, use a PLAIN stratified 80/20 "
                    "train_test_split(random_state=42) and do NOT attempt "
                    "school-cluster reconstruction."
                )

        if spec.get("task_type") != "causal_soo":
            return None
        train_X_path = Path(self.ctx.output_dir) / "train_X.csv"
        try:
            # 3b.23.7 sw1c: deterministic repair for the dummied-treatment
            # shape BEFORE asserting — the pair encodes identical
            # information, so collapsing it is safe and saves a retry.
            repair_note = repair_dummied_treatment(self.ctx.output_dir, spec)
            if repair_note:
                self._log("Orchestrator", repair_note)
                if isinstance(self.ctx.data_report, dict):
                    self.ctx.data_report.setdefault("warnings", []).append(
                        repair_note
                    )
            assert_causal_soo_data_contract(train_X_path, spec)
            registry: dict = {}
            try:
                registry = self.data_engineer.load_registry() or {}
            except Exception:
                registry = {}
            assert_causal_soo_matrix_contract(
                self.ctx.output_dir, spec, registry
            )
        except CausalDataContractError as cdce:
            return str(cdce)
        except Exception as exc:  # probe robustness: never crash a run
            self._log(
                "Orchestrator",
                f"Post-DE pre-flight probe error (non-fatal, treated as "
                f"pass): {exc}",
            )
        return None

    def _run_engineering(self) -> None:
        if "ENGINEERING" in self.ctx.completed_stages:
            self.ctx.current_state = PipelineState.ANALYZING
            return
        self._log("Orchestrator", "Starting ENGINEERING stage")
        try:
            self._inject_skills(self.data_engineer, "DataEngineer")
            result = self.data_engineer.run()
            self.ctx.data_report = result
            if not result.get("validation_passed", False):
                # V4 Arc H (3b.23.7 sw1b): validation_passed=False is a
                # deterministic post-DE signal exactly like a contract
                # violation — grant the same single targeted retry with
                # the failed-validation warnings injected, instead of
                # aborting on the first roll of the codegen dice
                # (F-3b17 NaN-cells shape aborted a full run here).
                warnings_txt = "; ".join(
                    str(w) for w in result.get("warnings", [])
                )
                self._log(
                    "Orchestrator",
                    f"DE validation_passed=False -> targeted DataEngineer "
                    f"retry: {warnings_txt[:300]}",
                )
                self._inject_skills(self.data_engineer, "DataEngineer")
                result = self.data_engineer.run(
                    revision_instructions=(
                        "VALIDATION FAILURE on your previous output "
                        "(deterministic post-DE check -- not a Critic "
                        "opinion). Your code executed but the produced "
                        "artifacts failed validation. Regenerate the data "
                        "engineering code fixing exactly these problems "
                        "while keeping everything else unchanged:\n\n"
                        f"{warnings_txt}\n\n"
                        "In particular: NO NaN cells may remain in "
                        "train_X/test_X after imputation -- verify with "
                        "df.isna().sum().sum() == 0 before writing the "
                        "CSVs, and impute EVERY remaining column "
                        "(including one-hot dummies and passthrough "
                        "columns), not only the originally-listed ones."
                    )
                )
                self.ctx.data_report = result
                if not result.get("validation_passed", False):
                    self._abort(
                        f"ENGINEERING aborted (validation retry exhausted): "
                        f"validation_passed=False. Warnings: "
                        f"{result.get('warnings', [])}"
                    )
                    return
                self._log(
                    "Orchestrator",
                    "DE validation retry produced a passing data_report",
                )
            if result.get("analytic_n", 0) < 1000:
                self._abort(
                    f"ENGINEERING aborted: analytic_n={result.get('analytic_n')} < 1000"
                )
                return
            # V3.0 Phase 3b.12 / §12.2 + V4 Arc H (3b.23.7) — post-DE
            # pre-flight. Header check (3b.12) plus matrix-level D1
            # assertions (3b.23.7: treatment binary, no object dtypes,
            # continuous-vars-stay-continuous, propensity-overlap sanity).
            # V4 Phase A adds a task-type-agnostic test-split sanity check
            # (empty/degenerate test set -> violation; causal_did exempt).
            # On violation: ONE targeted DataEngineer retry with the
            # violation text injected, then abort if the retry still
            # violates (fail-fast preserved).
            violation = self._run_post_de_preflight()
            if violation is not None:
                self._log(
                    "Orchestrator",
                    f"Post-DE pre-flight violation -> targeted DataEngineer "
                    f"retry: {violation}",
                )
                self._inject_skills(self.data_engineer, "DataEngineer")
                result = self.data_engineer.run(
                    revision_instructions=(
                        "POST-DE PRE-FLIGHT CONTRACT VIOLATION on your "
                        "previous output (deterministic orchestrator check "
                        "-- not a Critic opinion). Regenerate the data "
                        "engineering code fixing exactly this violation "
                        "while keeping everything else unchanged:\n\n"
                        f"{violation}"
                    )
                )
                self.ctx.data_report = result
                if not result.get("validation_passed", False):
                    self._abort(
                        f"ENGINEERING aborted after pre-flight retry: "
                        f"validation_passed=False. Warnings: "
                        f"{result.get('warnings', [])}"
                    )
                    return
                second_violation = self._run_post_de_preflight()
                if second_violation is not None:
                    self._abort(
                        f"ENGINEERING aborted (causal data contract, "
                        f"post-retry): {second_violation}"
                    )
                    return
                self._log(
                    "Orchestrator",
                    "Post-DE pre-flight retry produced a compliant matrix",
                )
            self.ctx.completed_stages.append("ENGINEERING")
            self.ctx.current_state = PipelineState.ANALYZING
            self._log("Orchestrator", "ENGINEERING stage complete")
            self._save_checkpoint()
            self._check_cost()
        except Exception as e:
            self._abort(f"ENGINEERING failed: {e}")

    def _run_analyzing(self) -> None:
        if "ANALYZING" in self.ctx.completed_stages:
            self.ctx.current_state = PipelineState.CRITIQUING
            return
        self._log("Orchestrator", "Starting ANALYZING stage")
        try:
            self._inject_skills(self.analyst, "Analyst")
            result = self.analyst.run()
            self.ctx.results_object = result
            self.ctx.completed_stages.append("ANALYZING")
            self.ctx.current_state = PipelineState.CRITIQUING
            self._log("Orchestrator", "ANALYZING stage complete")
            self._save_checkpoint()
            self._check_cost()
        except Exception as e:
            self._abort(f"ANALYZING failed: {e}")

    def _run_critiquing(self) -> None:
        self._log("Orchestrator", f"Starting CRITIQUING stage (cycle {self.ctx.revision_cycle})")
        try:
            # --- Deterministic pre-Critic guard (inspired by AutoResearchClaw health.py) ---
            # 3b.6 / 6.2: task_type gates which structural checks fire so
            # prediction-shaped complaints (SHAP, top_features, etc.) do
            # not pollute causal_soo Critic input.
            pre_result = run_pre_critic_checks(
                self.ctx, self.ctx.output_dir, task_type=self.ctx.task_type,
            )
            if pre_result.failures:
                for f in pre_result.failures:
                    self._log(
                        "Orchestrator",
                        f"[PreCritic][{f.severity}] {f.check_id}: {f.message}",
                    )
            if pre_result.has_critical:
                # Short-circuit: synthesise REVISE/ABORT without an Opus call
                self.ctx.review_report = self._synthesize_pre_critic_report(pre_result)
                verdict = self.ctx.review_report["overall_verdict"]
                self._log(
                    "Orchestrator",
                    f"Pre-Critic guard found critical failures → short-circuit verdict: {verdict}",
                )
                if verdict == "REVISE" and self.ctx.revision_cycle < self.ctx.max_revision_cycles:
                    self.ctx.revision_cycle += 1
                    self.ctx.current_state = PipelineState.REVISING
                elif verdict == "ABORT":
                    self.ctx.errors.append(
                        f"Pre-Critic guard issued ABORT: {pre_result.failures}"
                    )
                    self.ctx.current_state = PipelineState.ABORTED
                else:
                    self.ctx.review_report["unverified"] = True
                    self.ctx.current_state = PipelineState.WRITING
                self._save_checkpoint()
                self._check_cost()
                return

            memory_summary = (
                self.findings_memory.to_summary_str()
                if self.findings_memory is not None
                else ""
            )
            self._inject_skills(self.critic, "Critic")
            result = self.critic.run(
                findings_memory_summary=memory_summary,
                pre_critic_failures=pre_result.failures,
            )
            self.ctx.review_report = result

            # Phase 3b.10 / §10.3: deterministic verdict evaluator.
            # Pre-3b.10 the orchestrator trusted result['overall_verdict']
            # directly. F-CRITIC-PASSED-WITH-LOW-SCORE (3b.5 + 3b.9
            # recurrence) showed the LLM emitted PASS even with
            # quality_score < 7 + critical issues present. The evaluator
            # below recomputes from (quality_score, n_critical, n_major)
            # per the documented thresholds, applies the cycles-exhausted
            # UNVERIFIED downgrade, and surfaces LLM-reported
            # disagreement at WARNING level.
            from src.agents.verdict_evaluator import evaluate_critic_verdict

            eval_result = evaluate_critic_verdict(
                result,
                revision_cycle=self.ctx.revision_cycle,
                max_revision_cycles=self.ctx.max_revision_cycles,
            )
            verdict = eval_result.verdict
            if eval_result.llm_disagreement:
                self._log(
                    "Orchestrator",
                    f"Critic verdict-evaluator overrode LLM: "
                    f"llm={eval_result.llm_verdict!r} → "
                    f"deterministic={eval_result.deterministic_verdict!r}, "
                    f"effective={verdict!r}, unverified={eval_result.unverified}. "
                    f"{eval_result.rationale}",
                )

            if verdict == "PASS":
                # Record the flag EXPLICITLY in both directions. On the
                # evaluator-override path the raw LLM verdict string can
                # be "REVISE" while the effective verdict is PASS —
                # run_is_unverified (Writer + linter) honors an explicit
                # unverified key, so an effective PASS must write False
                # or the paper gets a warning block the system itself
                # says it does not deserve.
                self.ctx.review_report["unverified"] = bool(
                    eval_result.unverified
                )
                self.ctx.completed_stages.append("CRITIQUING")
                self.ctx.current_state = PipelineState.WRITING
                pass_label = "PASS (UNVERIFIED)" if eval_result.unverified else "PASS"
                self._log(
                    "Orchestrator",
                    f"Critic verdict: {pass_label} → proceeding to WRITING",
                )
            elif verdict == "REVISE":
                if self.ctx.revision_cycle < self.ctx.max_revision_cycles:
                    self.ctx.revision_cycle += 1
                    self.ctx.current_state = PipelineState.REVISING
                    self._log(
                        "Orchestrator",
                        f"Critic verdict: REVISE → starting revision cycle {self.ctx.revision_cycle}",
                    )
                else:
                    self.ctx.review_report["unverified"] = True
                    self.ctx.current_state = PipelineState.WRITING
                    self._log(
                        "Orchestrator",
                        "Critic verdict: REVISE but max cycles exhausted → WRITING (UNVERIFIED)",
                    )
            elif verdict == "ABORT":
                self.ctx.errors.append(f"Critic issued ABORT verdict: {result}")
                self.ctx.current_state = PipelineState.ABORTED
                self._log("Orchestrator", "Critic verdict: ABORT → pipeline aborted")
            else:
                self._abort(f"Unknown critic verdict: {verdict}")
                return

            self._save_checkpoint()
            self._check_cost()
        except Exception as e:
            self._abort(f"CRITIQUING failed: {e}")

    def _run_revising(self) -> None:
        self._log("Orchestrator", f"Starting REVISING stage (cycle {self.ctx.revision_cycle})")
        try:
            self._execute_revisions()
            self.ctx.completed_stages.append("REVISING")
            self.ctx.current_state = PipelineState.CRITIQUING
            self._log("Orchestrator", "REVISING stage complete → back to CRITIQUING")
            self._save_checkpoint()
            self._check_cost()
        except Exception as e:
            # Revision failure is non-fatal: fall back to WRITING with UNVERIFIED flag
            # rather than aborting and discarding the existing analysis results.
            self._log("Orchestrator", f"REVISING failed ({e}); falling back to WRITING (UNVERIFIED)")
            if self.ctx.review_report is None:
                self.ctx.review_report = {}
            self.ctx.review_report["unverified"] = True
            self.ctx.errors.append(f"REVISING failed: {e}")
            self.ctx.current_state = PipelineState.WRITING
            self._save_checkpoint()

    def _run_writing(self) -> None:
        if "WRITING" in self.ctx.completed_stages:
            rg_enabled = self.config.get("review_gate", {}).get("enabled", False)
            if rg_enabled and "REVIEWING" not in self.ctx.completed_stages:
                self.ctx.current_state = PipelineState.REVIEWING
            else:
                self.ctx.current_state = PipelineState.COMPLETED
            return
        self._log("Orchestrator", "Starting WRITING stage")
        try:
            # Phase 1: generate outline if outline_first is enabled
            outline = None
            if self.config.get("writer", {}).get("outline_first", True):
                self._log("Orchestrator", "Running OutlineAgent (outline-first mode)")
                try:
                    from src.agents.outline_agent import OutlineAgent

                    agent_kwargs = dict(
                        executor=self._executor,
                        task_template=self.task_template,
                        dataset_adapter=self.dataset_adapter,
                    )
                    outline_agent = OutlineAgent(
                        self.ctx, "outline_agent", self.config, **agent_kwargs
                    )
                    self._inject_skills(outline_agent, "OutlineAgent")
                    outline = outline_agent.run()
                    self.ctx.paper_outline = outline
                    self._log("Orchestrator", "OutlineAgent complete")
                except Exception as e:
                    self._log(
                        "Orchestrator",
                        f"OutlineAgent failed (non-fatal, falling back to v1): {e}",
                    )
                    outline = None

            # Arc P3: top the reference list back up to the venue norm.
            # The ProblemFormulator retrieves ~100 papers and persists
            # only the 8-12 it selected, which is why manuscripts carried
            # 4-26 references against venue norms of 15 (EDM) / 47 (JLA) /
            # 54 (JEDM). Pure set arithmetic over already-retrieved
            # records: no new search, no LLM call, no network.
            self._expand_literature_for_depth()

            # Phase 2: generate prose
            self._inject_skills(self.writer, "Writer")
            result = self.writer.run(outline=outline)
            self.ctx.paper_text = result if isinstance(result, str) else result.get("paper_text", "")

            # Compile LaTeX: pdflatex → bibtex → pdflatex → pdflatex
            self._log("Orchestrator", "Compiling paper.tex (pdflatex → bibtex → pdflatex → pdflatex)")
            compile_result = compile_latex(self.ctx.output_dir)
            if compile_result["success"]:
                self._log("Orchestrator", "LaTeX compilation succeeded → paper.pdf written")
            else:
                failed = [s for s in compile_result["steps"] if s["returncode"] not in (0, 1)]
                for step in failed:
                    self._log("Orchestrator", f"LaTeX compile step failed: {step['cmd']} (rc={step['returncode']}): {step['stderr'][:500]}")
                self._log("Orchestrator", "LaTeX compilation had errors — check pipeline.log for details")

            self.ctx.completed_stages.append("WRITING")

            # Transition to REVIEWING if the review gate is enabled, else COMPLETED
            rg_enabled = self.config.get("review_gate", {}).get("enabled", False)
            if rg_enabled:
                self.ctx.current_state = PipelineState.REVIEWING
                self._log("Orchestrator", "WRITING stage complete → REVIEWING")
            else:
                self.ctx.current_state = PipelineState.COMPLETED
                self._log("Orchestrator", "WRITING stage complete → COMPLETED")
            self._save_checkpoint()
            self._check_cost()
            if not rg_enabled:
                self._update_findings_memory()
        except Exception as e:
            self._abort(f"WRITING failed: {e}")

    def _expand_literature_for_depth(self) -> None:
        """Arc P3: widen literature_context.papers toward the venue norm.

        Non-fatal by construction — a failure here must never cost the
        run its paper, so any problem degrades to the original selection.
        """
        try:
            # NOTE: composition_age_profile comes from src.citations, NOT
            # src.manuscript_linter — the linter's venue_age_profile returns
            # a 3-tuple and would raise here.
            from src.citations import (
                composition_age_profile,
                expand_literature_pool,
                venue_citation_target,
            )

            lit = self.ctx.literature_context or {}
            selected = lit.get("papers") or []
            pool = (getattr(self.ctx, "retrieved_literature", None) or {}).get(
                "papers"
            ) or []
            venue = self.config.get("review_gate", {}).get("venue")
            target = venue_citation_target(venue)
            if not target or not pool:
                self._log(
                    "Orchestrator",
                    f"Citation depth: no expansion (venue={venue}, "
                    f"target={target}, pool={len(pool)}, "
                    f"selected={len(selected)})",
                )
                return
            # Arc P5: compose across age bins rather than appending in the
            # pool's year-descending order. Without `profile=` the append
            # path returns the newest N, which is how a manuscript ended up
            # citing nothing published before 2024 while the citation COUNT
            # metric read green — and the linter now errors on exactly that.
            depth_stats: dict[str, Any] = {}
            expanded = expand_literature_pool(
                selected,
                pool,
                target,
                profile=composition_age_profile(venue),
                now_year=datetime.utcnow().year,
                stats=depth_stats,
            )
            if len(expanded) <= len(selected):
                self._log(
                    "Orchestrator",
                    f"Citation depth: pool exhausted at {len(expanded)} "
                    f"papers (target {target} for {venue})",
                )
            achieved = depth_stats.get("achieved") or {}
            if achieved:
                self._log(
                    "Orchestrator",
                    "Citation recency composition: "
                    + ", ".join(f"{b}={achieved.get(b, 0)}" for b in achieved)
                    + (f" (degraded signals: {depth_stats['degraded']})"
                       if depth_stats.get("degraded") else ""),
                )
                try:
                    path = os.path.join(
                        self.ctx.output_dir, "citation_depth_report.json"
                    )
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(depth_stats, f, indent=2, default=str)
                except OSError:
                    pass
            self.ctx.literature_context = {**lit, "papers": expanded}
            self._log(
                "Orchestrator",
                f"Citation depth: {len(selected)} selected + pool of "
                f"{len(pool)} -> {len(expanded)} available references "
                f"(target {target} for {venue})",
            )
            path = os.path.join(
                self.ctx.output_dir, "literature_context_expanded.json"
            )
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.ctx.literature_context, f, indent=2)
        except Exception as exc:  # noqa: BLE001
            self._log(
                "Orchestrator",
                f"Citation depth expansion failed (non-fatal): {exc}",
            )

    def _run_reviewing(self) -> None:
        if "REVIEWING" in self.ctx.completed_stages:
            self.ctx.current_state = PipelineState.COMPLETED
            return
        self._log("Orchestrator", "Starting REVIEWING stage (LSAR quality gate)")
        try:
            gate = ReviewGate(
                config=self.config,
                output_dir=Path(self.ctx.output_dir),
                log_fn=self._log,
            )
            summary = gate.run_gate()
            self.ctx.review_gate_result = summary

            # Log summary
            self._log(
                "Orchestrator",
                f"LSAR review gate: passed={summary['passed']}, "
                f"cycles={summary['cycles_used']}, "
                f"score={summary['final_score']:.2f}, "
                f"rec={summary['final_recommendation']}",
            )
        except Exception as e:
            self._log("Orchestrator", f"REVIEWING failed (non-fatal): {e}")
            self.ctx.review_gate_result = {
                "error": str(e),
                "passed": False,
                "cycles_used": 0,
            }

        # Always proceed to COMPLETED — the gate is diagnostic, not blocking
        self.ctx.completed_stages.append("REVIEWING")
        self.ctx.current_state = PipelineState.COMPLETED
        self._log("Orchestrator", "REVIEWING stage complete → COMPLETED")
        self._save_checkpoint()
        self._check_cost()
        self._update_findings_memory()

    # ------------------------------------------------------------------
    # Revision cascade (SPEC §5.3)
    # ------------------------------------------------------------------

    def _execute_revisions(self) -> None:
        if not self.ctx.review_report:
            return
        agent_order = self.task_template.get_agent_order()
        instructions = self.ctx.review_report["revision_instructions"]
        targeted = [a for a in agent_order if instructions.get(a)]
        if not targeted:
            return
        start_idx = agent_order.index(targeted[0])
        for agent_name in agent_order[start_idx:]:
            self._run_agent(agent_name, revision_instructions=instructions.get(agent_name))

    def _run_agent(
        self,
        agent_name: str,
        revision_instructions: Optional[str] = None,
    ) -> None:
        agent_map = {
            "ProblemFormulator": self.problem_formulator,
            "DataEngineer": self.data_engineer,
            "Analyst": self.analyst,
        }
        agent = agent_map[agent_name]
        self._inject_skills(agent, agent_name)
        if agent_name == "ProblemFormulator":
            # Phase 3b.6 / 6.6 wire-up: in revision mode, the PF must see
            # the cycle-0 spec to preserve its locked invariants (estimand,
            # method battery, methodological_concerns, etc.). 3b.5's
            # F-LOCKED-SPEC-INVARIANTS evidence: without this wiring, PF
            # cycle-1 re-derived everything from scratch and silently
            # dropped the cycle-0 ESC-07 flag, renamed methods, and
            # restricted the adjustment set without justification.
            #
            # Preference order for the spec to preserve:
            #   1. self.ctx.research_spec (the cycle-0 PF refinement) when
            #      it exists — captures both the original locked
            #      invariants AND PF's prior analysis (e.g., flagged
            #      ESC-07 concerns).
            #   2. self.ctx.locked_research_spec (the CLI-loaded spec) as
            #      a fallback for the unlikely case of revising before
            #      cycle 0 emitted anything.
            prior_spec = (
                self.ctx.research_spec or self.ctx.locked_research_spec
            )
            result = agent.run(
                revision_instructions=revision_instructions,
                locked_research_spec=prior_spec,
            )
        else:
            result = agent.run(revision_instructions=revision_instructions)
        if agent_name == "ProblemFormulator":
            self.ctx.research_spec = result.get("research_spec")
            self.ctx.literature_context = result.get("literature_context")
            self.ctx.retrieved_literature = result.get("retrieved_literature")
            self._save_formulating_outputs()
        elif agent_name == "DataEngineer":
            self.ctx.data_report = result
        elif agent_name == "Analyst":
            self.ctx.results_object = result

    # ------------------------------------------------------------------
    # Output file helpers
    # ------------------------------------------------------------------

    def _save_formulating_outputs(self) -> None:
        """Persist research_spec.json and literature_context.json to the run directory."""
        if self.ctx.research_spec is not None:
            path = os.path.join(self.ctx.output_dir, "research_spec.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.ctx.research_spec, f, indent=2)
        if self.ctx.literature_context is not None:
            path = os.path.join(self.ctx.output_dir, "literature_context.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.ctx.literature_context, f, indent=2)
        # Arc P3: the full retrieved pool the Writer draws depth from.
        # Persisted so a resumed run (and post-hoc analysis of how deep
        # the pool actually was) does not lose it.
        if getattr(self.ctx, "retrieved_literature", None) is not None:
            path = os.path.join(self.ctx.output_dir, "retrieved_literature.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.ctx.retrieved_literature, f, indent=2)

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _synthesize_pre_critic_report(self, pre_result: PreCriticResult) -> dict:
        """Build a minimal review_report from pre-critic failures without an LLM call."""
        verdict = "ABORT" if pre_result.has_critical else "REVISE"

        def _issues_for(agent: str) -> list[dict]:
            return [
                {
                    "severity": f.severity,
                    "category": f.check_id,
                    "description": f.message,
                    "recommendation": f.message,
                    "target_agent": agent,
                }
                for f in pre_result.failures
                if f.target_agent == agent
            ]

        ri: dict[str, Optional[str]] = {
            "ProblemFormulator": None,
            "DataEngineer": None,
            "Analyst": None,
        }
        for f in pre_result.failures:
            if f.target_agent in ri and ri[f.target_agent] is None:
                ri[f.target_agent] = f.message

        return {
            "overall_verdict": verdict,
            "overall_quality_score": 1,
            "problem_formulation_review": {"score": 5, "issues": _issues_for("ProblemFormulator")},
            "data_preparation_review": {"score": 1, "issues": _issues_for("DataEngineer")},
            "analysis_review": {"score": 1, "issues": _issues_for("Analyst")},
            "substantive_review": {
                "score": 1,
                "educational_meaningfulness": "Pre-Critic automated check failed before substantive review.",
                "issues": [],
            },
            "revision_instructions": ri,
            "_source": "pre_critic_short_circuit",
        }

    def _save_checkpoint(self) -> None:
        path = os.path.join(self.ctx.output_dir, "checkpoint.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.ctx.to_dict(), f, indent=2)

    def _load_checkpoint(self) -> None:
        path = os.path.join(self.ctx.output_dir, "checkpoint.json")
        if not os.path.exists(path):
            return
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        loaded = PipelineContext.from_dict(data)
        # Mutate in-place so agent references stay valid
        self.ctx.current_state = loaded.current_state
        self.ctx.completed_stages = loaded.completed_stages
        self.ctx.revision_cycle = loaded.revision_cycle
        self.ctx.research_spec = loaded.research_spec
        self.ctx.literature_context = loaded.literature_context
        self.ctx.retrieved_literature = loaded.retrieved_literature
        self.ctx.data_report = loaded.data_report
        self.ctx.results_object = loaded.results_object
        self.ctx.review_report = loaded.review_report
        self.ctx.paper_text = loaded.paper_text
        self.ctx.review_gate_result = loaded.review_gate_result
        self.ctx.errors = loaded.errors
        self.ctx.log = loaded.log
        self._log("Orchestrator", f"Resumed from checkpoint (state={loaded.current_state})")

    # ------------------------------------------------------------------
    # Logging and cost tracking
    # ------------------------------------------------------------------

    def _log(self, agent: str, message: str) -> None:
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "agent": agent,
            "message": message,
        }
        self.ctx.log.append(entry)
        log_path = os.path.join(self.ctx.output_dir, "pipeline.log")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{entry['timestamp']} [{agent}] {message}\n")

    def _check_cost(self) -> None:
        """Compare measured spend against the run budget (K1).

        This used to multiply the summed token count by a hardcoded
        0.000015 — $15 per million, an Anthropic-era rate left in the
        code after the stack moved to DeepSeek, where it overstates cost
        by roughly 40x. Rates now come from config.yaml and are applied
        to the prompt/completion split the provider actually reported.
        """
        from src.cost import load_pricing, load_usage_best, summarize

        budget = self.config["pipeline"].get("cost_budget_usd", 5.0)
        usages = load_usage_best(self.ctx.output_dir)
        if not usages:
            return
        summary = summarize(usages, load_pricing(self.config))
        if summary.cost_usd is None:
            self._log(
                "Orchestrator",
                f"Cost not priced: {summary.total_tokens:,} tokens across "
                f"{summary.n_calls} calls; no rate configured for "
                f"{', '.join(summary.unpriced_models)}. Add them under "
                "config.yaml pricing.per_million_tokens.",
            )
            return
        note = ""
        if summary.unpriced_models:
            note = (
                f" (LOWER BOUND — no rate for {', '.join(summary.unpriced_models)})"
            )
        if summary.cost_usd > budget:
            self._log(
                "Orchestrator",
                f"WARNING: measured cost ${summary.cost_usd:.4f} exceeds "
                f"budget ${budget:.2f}{note}",
            )

    def _update_findings_memory(self) -> None:
        """Persist this run's findings to the cross-run memory store (non-fatal)."""
        if self.findings_memory is None:
            return
        try:
            run_id = os.path.basename(self.ctx.output_dir)
            start_time = getattr(self.ctx, "run_start_time", "")
            runtime_minutes: float | None = None
            if start_time:
                try:
                    from datetime import timezone
                    start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
                    now_dt = datetime.now(timezone.utc)
                    runtime_minutes = (now_dt - start_dt).total_seconds() / 60.0
                except Exception:
                    pass
            entry = RunEntry.from_pipeline_context(
                ctx=self.ctx,
                run_id=run_id,
                runtime_minutes=runtime_minutes,
                api_cost_usd=None,
            )
            self.findings_memory.add_run(entry)
            self.findings_memory.save()
            self._log("Orchestrator", f"FindingsMemory updated: {run_id}")
        except Exception as exc:
            self._log("Orchestrator", f"FindingsMemory update failed (non-fatal): {exc}")

    def _abort(self, reason: str) -> None:
        self.ctx.errors.append(reason)
        self.ctx.current_state = PipelineState.ABORTED
        self._log("Orchestrator", f"ABORTED: {reason}")
        self._save_checkpoint()
        self._update_findings_memory()
