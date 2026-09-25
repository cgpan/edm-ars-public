from __future__ import annotations

import csv
import dataclasses
import json
import os
import shutil
import time
import warnings
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


#: Encoded design matrices wider than this are structural corruption, not
#: a modelling choice. A typical prediction spec of <=30 raw variables
#: encodes to ~50 columns; the observed failure reached 16,945.
MAX_ENCODED_COLUMNS = 500


def check_design_matrix_width(
    output_dir: str, max_columns: int = MAX_ENCODED_COLUMNS
) -> str | None:
    """Refuse a design matrix wide enough that nothing can train.

    A2. A continuous maths theta score (X1TXMTSCOR, 20,741 distinct
    values) was one-hot encoded into 16,705 dummy columns; the matrix
    reached 18,806 x 16,945 and 1.8 GB, and no model trained inside any
    timeout. Three consecutive runs failed, each misdiagnosed as
    "slowness".

    The generated guard existed and mutated the list it was iterating
    (``onehot_cols.remove(col)`` inside ``for col in onehot_cols``), which
    skips the next element and lets roughly half the offenders through.
    Prose guidance did not hold, so the ceiling is enforced here, where a
    violation triggers a targeted DataEngineer retry.

    A module-level function rather than a method: the pre-flight is
    exercised with lightweight stubs, and a new check should not require
    every caller to grow an attribute.

    Names the columns responsible rather than reporting only the total,
    because the actionable fact is WHICH variable exploded -- the failure
    presents as a timeout, which sends people to the model battery
    instead of the encoder.
    """
    train_X_path = Path(output_dir) / "train_X.csv"
    if not train_X_path.exists():
        return None
    try:
        import pandas as _pd

        header = _pd.read_csv(train_X_path, nrows=0)
    except Exception:  # noqa: BLE001
        # The pre-flight must never be the thing that breaks a healthy run.
        return None

    n_cols = len(header.columns)
    if n_cols <= max_columns:
        return None

    from collections import Counter

    parents = Counter(str(c).split("_")[0] for c in header.columns)
    worst = ", ".join(
        f"{name} -> {count} columns"
        for name, count in parents.most_common(3)
        if count > 1
    )
    return (
        f"Encoded design matrix has {n_cols} columns, above the ceiling of "
        f"{max_columns}. This is one-hot expansion of a CONTINUOUS "
        f"variable, not a modelling choice, and no model will train. Worst "
        f"offenders: {worst or 'unable to attribute'}. Decide categorical "
        "vs continuous from the DECLARED TYPE in the dataset registry, not "
        "from the pandas dtype after sentinel replacement -- mapping "
        "labelled sentinels to NaN turns a numeric column into object "
        "dtype and makes a continuous score look categorical. Build the "
        "one-hot list with a comprehension "
        "(`[c for c in cats if X[c].nunique() <= 100]`); never remove from "
        "a list while iterating over it."
    )


#: A handful of all-zero dummies is a rare category; a matrix where most
#: of them are constant is a broken encoding. The threshold exists so a
#: single sparse level does not abort a healthy run.
MAX_CONSTANT_TEST_COLUMNS = 3


def check_constant_test_columns(
    output_dir: str, max_constant: int = MAX_CONSTANT_TEST_COLUMNS
) -> str | None:
    """Refuse a test matrix whose columns carry no information.

    Returns a retry message naming the offending columns, or ``None``.

    A module-level function rather than a method, matching
    ``check_design_matrix_width``: the pre-flight is exercised with
    lightweight stubs and a new check should not require every caller to
    grow an attribute.
    """
    path = Path(output_dir) / "test_X.csv"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8", newline="") as f:
            rd = csv.reader(f)
            header = next(rd)
            firsts: list[str | None] = [None] * len(header)
            varies = [False] * len(header)
            n = 0
            for row in rd:
                n += 1
                for i in range(min(len(row), len(header))):
                    if varies[i]:
                        continue
                    if firsts[i] is None:
                        firsts[i] = row[i]
                    elif row[i] != firsts[i]:
                        varies[i] = True
    except (OSError, StopIteration, ValueError):
        # The pre-flight must never be the thing that breaks a healthy run.
        return None
    if not n:
        return None

    constant = [header[i] for i in range(len(header)) if not varies[i]]
    if len(constant) <= max_constant:
        return None

    return (
        f"{len(constant)} of {len(header)} columns in test_X.csv are CONSTANT "
        f"across all {n} test rows: {', '.join(constant[:10])}"
        + (" ..." if len(constant) > 10 else "")
        + ". A column with no variance in the test set cannot move a test "
        "prediction, yet it reaches the model and then a SHAP ranking the "
        "paper writes about. This is one-hot encoding fitted SEPARATELY on "
        "train and test and reconciled with reindex: the two splits drop "
        "different reference categories, so surviving columns land on the "
        "wrong labels and absent categories become all-zero. Fit the "
        "encoding ONCE on train and apply the same category mapping to "
        "test -- analysis_helpers.encode_categoricals(train_df, test_df, "
        "cat_cols) does this and returns an encoding_report naming each "
        "reference category. Compute constancy on BOTH splits, not just "
        "train."
    )


def _resolve_skill_caps(task_type: str) -> dict[str, int]:
    """Return the per-layer skill cap for a given task type.

    Falls back to ``_DEFAULT_SKILL_CAPS`` when no task-type-specific
    override exists; this preserves byte-identical behavior for the
    prediction codepath.
    """
    return _SKILL_CAPS_BY_TASK_TYPE.get(task_type, _DEFAULT_SKILL_CAPS)


class CheckpointCorruptError(ValueError):
    """``checkpoint.json`` exists but cannot be read back.

    Raised from ``Orchestrator.__init__`` instead of a bare
    ``JSONDecodeError`` so the message names the file and says what to
    do. With atomic saves this should only come from an external edit or
    a file written by an older version.
    """


def _atomic_write_text(path: str, text: str) -> None:
    """Replace ``path`` with ``text`` so a reader never sees half a file.

    ``open(path, "w")`` truncates first and writes in chunks: a kill, a
    full disk or a serialisation error part-way through used to leave a
    partial ``checkpoint.json`` -- the run's only resume point -- that the
    next ``--resume`` could not parse. The text is written to a sibling
    temp file, flushed to disk and moved over the target in one step.
    """
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        for attempt in range(5):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                # Windows refuses to replace a file another process has
                # open (an editor, a status viewer). Retry briefly.
                if attempt == 4:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


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

        # A resumed run is the run in the checkpoint, not whatever the
        # command line says this time. Adopt the checkpoint's dataset,
        # task type and locked spec BEFORE the template, adapter and
        # agents are built from them (D2) -- restoring them afterwards
        # would leave a causal run running prediction prompts.
        checkpoint = self._read_checkpoint()
        if checkpoint is not None:
            self._adopt_checkpoint_identity(checkpoint)

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
        if checkpoint is not None:
            self._load_checkpoint(checkpoint)
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
            elif state == PipelineState.VERIFYING:
                self._run_verifying()
            elif state in (
                PipelineState.COMPLETED,
                PipelineState.INCOMPLETE,
                PipelineState.ABORTED,
            ):
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
            self.ctx.research_spec = self._carry_locked_guidance(
                result.get("research_spec")
            )
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

    def _check_design_matrix_width(self) -> str | None:
        """Refuse a design matrix wide enough that nothing can train.

        Names the columns responsible rather than reporting only the
        total, because the actionable fact is WHICH variable exploded --
        the failure presents as a timeout, which sends people looking at
        the model battery instead of the encoder.
        """
        return check_design_matrix_width(self.ctx.output_dir)

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

        # A2: hard ceiling on the encoded design matrix.
        #
        # A continuous maths theta score (X1TXMTSCOR, 20,741 distinct
        # values) was one-hot encoded into 16,705 dummy columns; the
        # design matrix reached 18,806 x 16,945 and 1.8 GB, and no model
        # could train inside any timeout. Three consecutive runs failed
        # and were each misdiagnosed as "slowness".
        #
        # The generated guard existed but mutated the list it was
        # iterating (`onehot_cols.remove(col)` inside `for col in
        # onehot_cols`), which skips the next element and lets roughly
        # half the offenders through. Prose guidance alone clearly does
        # not hold, so the ceiling is enforced here where a violation
        # triggers a targeted retry.
        violation = check_design_matrix_width(self.ctx.output_dir)
        if violation:
            return violation

        # A column with no variance in TEST cannot move a test
        # prediction, yet it reaches the model and then a SHAP ranking
        # the paper writes about. `encode_categoricals` prevents this,
        # and the A/B measured that the DataEngineer does not call it --
        # so the guard lives here, in code, next to the width ceiling,
        # which exists for the same reason.
        violation = check_constant_test_columns(self.ctx.output_dir)
        if violation:
            return violation

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

    def _carry_locked_guidance(self, spec: dict | None) -> dict | None:
        """Keep the locked spec's free-text guidance in the emitted spec.

        The locked spec is handed to the ProblemFormulator, which returns
        a research_spec of its own, and that replaces the locked one
        wholesale. Anything the PF does not echo back is gone by the time
        the DataEngineer runs.

        A study spec proved what that costs. It carried an ENCODING
        CONTRACT -- "do NOT pass them to get_dummies ... under any
        circumstance" -- naming five continuous predictors. The prompts
        show it reached the ProblemFormulator (6 mentions) and NOTHING
        downstream: DataEngineer, Analyst and Critic all saw zero. X1SES
        was one-hot encoded into 5,514 columns and X1TXMTSCOR into 9,350,
        giving 15,008 features for 12,918 students, and SHAP for
        X1TXMTSCOR summed to 0.0 across its 9,350 dummies.

        The unrecognised-key warning tells authors to move such guidance
        into `additional_constraints`. That advice was itself broken
        until this function existed, because that key died at the same
        boundary as the ones it was recommending a retreat from.
        """
        if not isinstance(spec, dict):
            return spec
        locked = getattr(self.ctx, "locked_research_spec", None)
        if not isinstance(locked, dict):
            return spec
        carried = locked.get("additional_constraints")
        if carried and not spec.get("additional_constraints"):
            spec["additional_constraints"] = carried
            self._log(
                "Orchestrator",
                "Carried 'additional_constraints' from the locked spec into "
                "the ProblemFormulator's research_spec so it reaches the "
                "DataEngineer, Analyst and Critic.",
            )
        return spec

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

            # Re-persist now that the effective verdict and the
            # `unverified` flag are settled. Without this the on-disk
            # review_report.json keeps only the LLM's raw verdict, and
            # every downstream reader -- Writer, linter, audit, archive
            # -- is looking at a record the orchestrator has already
            # superseded in memory.
            self.ctx.review_report["effective_verdict"] = verdict
            self.ctx.review_report.setdefault(
                "unverified", bool(eval_result.unverified)
            )
            self.ctx.review_report["verdict_evaluation"] = {
                "llm_verdict": eval_result.llm_verdict,
                "deterministic_verdict": eval_result.deterministic_verdict,
                "llm_disagreement": bool(eval_result.llm_disagreement),
                "rationale": eval_result.rationale,
                "revision_cycle": self.ctx.revision_cycle,
                "max_revision_cycles": self.ctx.max_revision_cycles,
            }
            try:
                self.critic.persist_review_report(self.ctx.review_report)
            except Exception as exc:  # noqa: BLE001 - never lose the stage over a write
                self._log("Orchestrator", f"Could not re-persist review_report.json: {exc}")

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
                self.ctx.current_state = PipelineState.VERIFYING
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
                    # Non-fatal, but NOT invisible. This branch used to
                    # leave one log line and nothing else, and a config
                    # pointing the outline stage at a retired model id
                    # (`deepseek-v4-flash`, which the API stopped serving)
                    # therefore degraded ten shipped configs to the v1
                    # placeholder-filling template path with no record
                    # anywhere that the outline-first design had not run.
                    # A degradation the run does not disclose is the D7
                    # defect this project writes papers about.
                    self._log(
                        "Orchestrator",
                        f"OutlineAgent failed (non-fatal, falling back to v1): {e}",
                    )
                    self.ctx.errors.append(
                        f"OutlineAgent failed; paper written via the v1 "
                        f"template path instead of outline-first: {e}"
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

            # Transition to REVIEWING if the review gate is enabled, else
            # straight to VERIFYING. Neither branch ends the run: VERIFYING
            # is what decides between COMPLETED and INCOMPLETE.
            rg_enabled = self.config.get("review_gate", {}).get("enabled", False)
            if rg_enabled:
                self.ctx.current_state = PipelineState.REVIEWING
                self._log("Orchestrator", "WRITING stage complete → REVIEWING")
            else:
                self.ctx.current_state = PipelineState.VERIFYING
                self._log("Orchestrator", "WRITING stage complete → VERIFYING")
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
            self.ctx.current_state = PipelineState.VERIFYING
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

        # The gate records a verdict; VERIFYING is what decides whether
        # the run is releasable. Proceeding unconditionally here is fine
        # now -- it was not, when COMPLETED was the next state and
        # nothing anywhere read summary["passed"].
        self.ctx.completed_stages.append("REVIEWING")
        self.ctx.current_state = PipelineState.VERIFYING
        self._log("Orchestrator", "REVIEWING stage complete → VERIFYING")
        self._save_checkpoint()
        self._check_cost()
        self._update_findings_memory()

    # ------------------------------------------------------------------
    # VERIFYING — hold the finished manuscript against the run's own files
    # ------------------------------------------------------------------

    def _run_verifying(self) -> None:
        """Run the deterministic invariant battery and record a verdict.

        This is the first stage in the pipeline that sees the finished
        manuscript and the artifacts at the same time. The Critic runs
        before the Writer and has never seen a paper; the LSAR gate runs
        after but reads a 48,000-character head-slice with no figures in
        it, and until now nothing anywhere read its ``passed`` field.

        The stage always writes ``invariants.json`` and
        ``run_status.json``. Whether a critical finding actually stops
        the run is ``verification.blocking`` in config, default False --
        every one of the fourteen archived papers fails at least one of
        these checks, so switching it on globally on day one would
        terminate every run INCOMPLETE and teach nobody anything.
        Promote checks to blocking individually, on evidence.
        """
        if "VERIFYING" in self.ctx.completed_stages:
            self.ctx.current_state = PipelineState.COMPLETED
            return
        self._log("Orchestrator", "Starting VERIFYING stage (invariant battery)")

        cfg = self.config.get("verification", {}) or {}
        enabled = cfg.get("enabled", True)
        blocking = bool(cfg.get("blocking", False))
        blocking_codes = set(cfg.get("blocking_codes") or [])

        payload: dict[str, Any] = {"enabled": bool(enabled)}
        findings: list = []
        if enabled:
            try:
                from src.invariants import findings_to_json, run_invariants

                findings = run_invariants(self.ctx.output_dir)
                payload = {**payload, **findings_to_json(findings)}
            except Exception as exc:  # noqa: BLE001
                # A broken detector must not look like a clean run.
                payload["error"] = f"{type(exc).__name__}: {exc}"
                self._log("Orchestrator", f"VERIFYING battery failed: {exc}")

        try:
            with open(
                os.path.join(self.ctx.output_dir, "invariants.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(payload, f, indent=2, default=str)
        except OSError as exc:
            self._log("Orchestrator", f"Could not write invariants.json: {exc}")

        # Instruction compliance. Nothing in this pipeline has ever been
        # able to say whether a Critic instruction was acted on: the one
        # archived instruction whose disposition is documented was
        # applied WITHOUT DISCLOSURE, and two others were ignored, and
        # none of the three left a trace anywhere. An obligation closes
        # when its own test passes against the produced manuscript, never
        # because an agent reported compliance.
        obligations_summary: dict = {}
        try:
            from src.obligations import (
                derive_obligations,
                evaluate_obligations,
                summarize,
            )

            obligations = derive_obligations(self.ctx.review_report)
            if obligations:
                paper_path = os.path.join(self.ctx.output_dir, "paper.tex")
                paper = None
                if os.path.exists(paper_path):
                    with open(paper_path, encoding="utf-8", errors="replace") as f:
                        paper = f.read()
                obligations_summary = summarize(
                    evaluate_obligations(obligations, paper)
                )
                with open(
                    os.path.join(self.ctx.output_dir, "obligations.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(obligations_summary, f, indent=2, default=str)
                by = obligations_summary["by_status"]
                self._log(
                    "Orchestrator",
                    f"Writer obligations: {obligations_summary['n_obligations']} "
                    f"({', '.join(f'{k}={v}' for k, v in sorted(by.items()))})",
                )
        except Exception as exc:  # noqa: BLE001
            self._log("Orchestrator", f"Obligation evaluation failed: {exc}")

        # Optional LLM judge over the finished manuscript. Strictly
        # opt-in (`verification.judge_enabled`), and OFF by default.
        #
        # Off is the honest default for two reasons. It costs a call per
        # run plus one per figure, and -- more importantly -- an A/B that
        # measures whether the pipeline PRODUCES fewer defects must not
        # have a judge in one arm FINDING more of them. Its marginal
        # recovery over the deterministic battery is measured separately,
        # offline, against the answer key.
        verification_report: dict = {}
        if cfg.get("judge_enabled", False):
            try:
                from src.agents.verifier import Verifier

                verifier = Verifier(self.ctx, self.config)
                verifier.skills = self._match_skills_for_stage("Verifier")
                verification_report = verifier.run()
                vf = verification_report.get("findings") or []
                self._log(
                    "Orchestrator",
                    f"Verifier: {len(vf)} finding(s) kept, "
                    f"{verification_report.get('n_dropped_by_validator', 0)} "
                    "dropped by the validator",
                )
            except Exception as exc:  # noqa: BLE001
                self._log("Orchestrator", f"Verifier failed (non-fatal): {exc}")
                verification_report = {"ran": False, "error": str(exc)}

        criticals = [f for f in findings if f.severity == "critical"]
        gate = self.ctx.review_gate_result or {}
        gate_failed = gate.get("passed") is False
        review = self.ctx.review_report or {}
        unverified = bool(review.get("unverified"))

        if blocking_codes:
            blockers = [f for f in criticals if f.code in blocking_codes]
        elif blocking:
            blockers = criticals
        else:
            blockers = []

        released = not blockers
        status = {
            "released": released,
            "reason": (
                "clean"
                if released and not criticals and not gate_failed and not unverified
                else "; ".join(
                    filter(
                        None,
                        [
                            f"{len(criticals)} critical invariant finding(s)"
                            if criticals
                            else "",
                            "review gate did not pass" if gate_failed else "",
                            "critic verdict was not PASS" if unverified else "",
                        ],
                    )
                )
                or "clean"
            ),
            "blocking_mode": "codes" if blocking_codes else ("all" if blocking else "advisory"),
            "invariant_counts": payload.get("counts", {}),
            "invariant_codes": payload.get("codes", []),
            "blocking_findings": [f.code for f in blockers],
            "review_gate_passed": gate.get("passed"),
            "review_gate_score": gate.get("final_score"),
            "critic_verdict": review.get("effective_verdict") or review.get("overall_verdict"),
            "critic_unverified": unverified,
            "writer_obligations": {
                k: obligations_summary.get(k)
                for k in ("n_obligations", "by_status", "compliance_rate")
            }
            if obligations_summary
            else None,
            "verifier": {
                "ran": bool(verification_report.get("ran")),
                "n_findings": len(verification_report.get("findings") or []),
                "n_dropped_by_validator": verification_report.get(
                    "n_dropped_by_validator"
                ),
            }
            if verification_report
            else None,
            "run_dir": self.ctx.output_dir,
            "timestamp": datetime.utcnow().isoformat(),
        }
        try:
            with open(
                os.path.join(self.ctx.output_dir, "run_status.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(status, f, indent=2, default=str)
        except OSError as exc:
            self._log("Orchestrator", f"Could not write run_status.json: {exc}")

        counts = payload.get("counts", {})
        self._log(
            "Orchestrator",
            "Invariant battery: "
            f"{counts.get('critical', 0)} critical, {counts.get('major', 0)} major, "
            f"{counts.get('minor', 0)} minor "
            f"({', '.join(payload.get('codes', [])) or 'none'})",
        )

        self.ctx.completed_stages.append("VERIFYING")
        if blockers:
            self.ctx.current_state = PipelineState.INCOMPLETE
            self.ctx.errors.append(
                "Release blocked by "
                f"{len(blockers)} critical invariant finding(s): "
                + ", ".join(sorted({f.code for f in blockers}))
            )
            self._log(
                "Orchestrator",
                f"VERIFYING: release BLOCKED → INCOMPLETE ({status['reason']})",
            )
        else:
            self.ctx.current_state = PipelineState.COMPLETED
            self._log(
                "Orchestrator",
                f"VERIFYING stage complete → COMPLETED ({status['reason']})",
            )
        self._save_checkpoint()

    # ------------------------------------------------------------------
    # Revision cascade (SPEC §5.3)
    # ------------------------------------------------------------------

    def _execute_revisions(self) -> None:
        """Re-run the targeted agent and everything downstream of it.

        ``revision_instructions`` may now carry a ``Writer`` key, which
        this loop deliberately ignores: CRITIQUING precedes WRITING, so
        at revision time the Writer has not run and there is nothing to
        re-run. Those instructions are obligations on the manuscript the
        Writer is about to produce, and they are delivered at WRITING by
        :meth:`_writer_obligations`.
        """
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
            self.ctx.research_spec = self._carry_locked_guidance(
                result.get("research_spec")
            )
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
        """Persist the context atomically (D1).

        Serialised to a string first, so a value json cannot encode raises
        before anything on disk changes; then written to a temp file and
        moved over checkpoint.json in one step. A kill or a full disk
        mid-write can no longer leave the run's only resume point
        half-written.
        """
        path = os.path.join(self.ctx.output_dir, "checkpoint.json")
        text = json.dumps(self.ctx.to_dict(), indent=2)
        _atomic_write_text(path, text)

    def _read_checkpoint(self) -> Optional[dict]:
        """Return the parsed checkpoint, None when there is none.

        Raises :class:`CheckpointCorruptError` naming the file when it
        exists but cannot be parsed, instead of a bare JSONDecodeError out
        of the constructor.
        """
        path = os.path.join(self.ctx.output_dir, "checkpoint.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (ValueError, UnicodeDecodeError) as exc:
            raise CheckpointCorruptError(
                f"{path} exists but is not valid JSON ({exc}). It is this "
                "run's resume point and cannot be read. Move it aside to "
                "start this directory afresh, or restore it from a copy."
            ) from exc
        if not isinstance(data, dict) or "current_state" not in data:
            raise CheckpointCorruptError(
                f"{path} does not look like an EDM-ARS checkpoint (no "
                "current_state). Move it aside to start this directory afresh."
            )
        return data

    def _adopt_checkpoint_identity(self, data: dict) -> None:
        """Make the context describe the checkpointed run (D2).

        ``--resume`` used to take dataset, task type and locked spec from
        the command line while every artifact came from the checkpoint,
        so the README's resume example re-typed a locked causal run as an
        HSLS prediction run with nothing warning anyone. The checkpoint
        wins; a disagreeing flag is reported, not obeyed.
        """
        def _warn(message: str) -> None:
            self._log("Orchestrator", f"WARNING: {message}")
            warnings.warn(message, RuntimeWarning, stacklevel=3)

        ck_dataset = data.get("dataset_name")
        if ck_dataset and ck_dataset != self.ctx.dataset_name:
            _warn(
                f"--dataset {self.ctx.dataset_name!r} disagrees with the "
                f"checkpoint in {self.ctx.output_dir}, which is a "
                f"{ck_dataset!r} run. Resuming it as {ck_dataset!r}."
            )
            self.ctx.dataset_name = ck_dataset
            if data.get("raw_data_path"):
                self.ctx.raw_data_path = data["raw_data_path"]
        ck_task = data.get("task_type")
        if ck_task and ck_task != self.ctx.task_type:
            _warn(
                f"task type {self.ctx.task_type!r} (from config or "
                f"--research-spec) disagrees with the checkpoint, which is a "
                f"{ck_task!r} run. Resuming it as {ck_task!r}."
            )
            self.ctx.task_type = ck_task
        if "locked_research_spec" in data:
            ck_locked = data.get("locked_research_spec")
            if (
                self.ctx.locked_research_spec is not None
                and ck_locked != self.ctx.locked_research_spec
            ):
                _warn(
                    "--research-spec differs from the locked spec this run "
                    "started with; the checkpoint's spec is kept."
                )
            self.ctx.locked_research_spec = ck_locked

    #: Fields ``_load_checkpoint`` does not copy: where the run lives and
    #: the live event handle; the identity fields, which
    #: ``_adopt_checkpoint_identity`` already settled (a relocated raw
    #: data path for the SAME dataset is kept); and the revision budget,
    #: which follows the current config.
    _NOT_RESTORED = frozenset(
        {
            "output_dir",
            "event_sink",
            "dataset_name",
            "raw_data_path",
            "task_type",
            "locked_research_spec",
            "max_revision_cycles",
        }
    )

    def _load_checkpoint(self, data: Optional[dict] = None) -> None:
        if data is None:
            data = self._read_checkpoint()
            if data is None:
                return
        loaded = PipelineContext.from_dict(data)
        # Mutate in place so agent references stay valid. Every dataclass
        # field is copied, so a field added later cannot be silently left
        # behind the way paper_outline and run_start_time were.
        for f in dataclasses.fields(PipelineContext):
            if f.name in self._NOT_RESTORED:
                continue
            setattr(self.ctx, f.name, getattr(loaded, f.name))
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
