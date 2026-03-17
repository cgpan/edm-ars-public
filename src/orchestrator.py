from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from typing import Optional

from src.agents.analyst import Analyst
from src.agents.critic import Critic
from src.agents.data_engineer import DataEngineer
from src.agents.problem_formulator import ProblemFormulator
from src.agents.writer import Writer
from src.context import PipelineContext, PipelineState
from src.dataset_adapter import create_dataset_adapter
from src.findings_memory import FindingsMemory, RunEntry
from src.pre_critic_checks import CheckFailure, PreCriticResult, run_pre_critic_checks
from src.sandbox import compile_latex, create_executor
from src.task_template import create_task_template


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

        # Copy config snapshot for reproducibility
        if os.path.exists(config_path):
            shutil.copy(
                config_path,
                os.path.join(ctx.output_dir, "config_snapshot.yaml"),
            )

        # Create task template and dataset adapter
        self.task_template = create_task_template(ctx.task_type)
        self.dataset_adapter = create_dataset_adapter(ctx.dataset_name)

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
        agent_kwargs = dict(
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
            elif state in (PipelineState.COMPLETED, PipelineState.ABORTED):
                break
            else:
                self._log("Orchestrator", f"Unknown state: {state}. Aborting.")
                self.ctx.current_state = PipelineState.ABORTED
                break
        return self.ctx

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

            result = self.problem_formulator.run(
                user_prompt=self._user_prompt,
                findings_memory_summary=memory_summary,
                n_candidate_specs=n_branches,
                studied_outcomes=studied_outcomes,
            )
            self.ctx.research_spec = result.get("research_spec")
            self.ctx.literature_context = result.get("literature_context")
            self._save_formulating_outputs()
            self.ctx.completed_stages.append("FORMULATING")
            self.ctx.current_state = PipelineState.ENGINEERING
            self._log("Orchestrator", "FORMULATING stage complete")
            self._save_checkpoint()
            self._check_cost()
        except Exception as e:
            self._abort(f"FORMULATING failed: {e}")

    def _run_engineering(self) -> None:
        if "ENGINEERING" in self.ctx.completed_stages:
            self.ctx.current_state = PipelineState.ANALYZING
            return
        self._log("Orchestrator", "Starting ENGINEERING stage")
        try:
            result = self.data_engineer.run()
            self.ctx.data_report = result
            if not result.get("validation_passed", False):
                self._abort(
                    f"ENGINEERING aborted: validation_passed=False. "
                    f"Warnings: {result.get('warnings', [])}"
                )
                return
            if result.get("analytic_n", 0) < 1000:
                self._abort(
                    f"ENGINEERING aborted: analytic_n={result.get('analytic_n')} < 1000"
                )
                return
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
            pre_result = run_pre_critic_checks(self.ctx, self.ctx.output_dir)
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
            result = self.critic.run(
                findings_memory_summary=memory_summary,
                pre_critic_failures=pre_result.failures,
            )
            self.ctx.review_report = result
            verdict = result.get("overall_verdict", "ABORT")

            if verdict == "PASS":
                self.ctx.completed_stages.append("CRITIQUING")
                self.ctx.current_state = PipelineState.WRITING
                self._log("Orchestrator", "Critic verdict: PASS → proceeding to WRITING")
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
            self.ctx.current_state = PipelineState.COMPLETED
            return
        self._log("Orchestrator", "Starting WRITING stage")
        try:
            result = self.writer.run()
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
            self.ctx.current_state = PipelineState.COMPLETED
            self._log("Orchestrator", "WRITING stage complete → COMPLETED")
            self._save_checkpoint()
            self._check_cost()
            self._update_findings_memory()
        except Exception as e:
            self._abort(f"WRITING failed: {e}")

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
        result = agent.run(revision_instructions=revision_instructions)
        if agent_name == "ProblemFormulator":
            self.ctx.research_spec = result.get("research_spec")
            self.ctx.literature_context = result.get("literature_context")
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
        with open(path, "w") as f:
            json.dump(self.ctx.to_dict(), f, indent=2)

    def _load_checkpoint(self) -> None:
        path = os.path.join(self.ctx.output_dir, "checkpoint.json")
        if not os.path.exists(path):
            return
        with open(path) as f:
            data = json.load(f)
        loaded = PipelineContext.from_dict(data)
        # Mutate in-place so agent references stay valid
        self.ctx.current_state = loaded.current_state
        self.ctx.completed_stages = loaded.completed_stages
        self.ctx.revision_cycle = loaded.revision_cycle
        self.ctx.research_spec = loaded.research_spec
        self.ctx.literature_context = loaded.literature_context
        self.ctx.data_report = loaded.data_report
        self.ctx.results_object = loaded.results_object
        self.ctx.review_report = loaded.review_report
        self.ctx.paper_text = loaded.paper_text
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
        total_tokens = sum(e.get("tokens_used", 0) for e in self.ctx.log)
        estimated_cost = total_tokens * 0.000015
        budget = self.config["pipeline"].get("cost_budget_usd", 5.0)
        if estimated_cost > budget:
            self._log(
                "Orchestrator",
                f"WARNING: Estimated cost ${estimated_cost:.2f} exceeds budget ${budget:.2f}",
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
