"""Critic agent: reviews all pipeline outputs and issues PASS/REVISE/ABORT verdicts."""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any

import yaml

from src.agents.base import BaseAgent, literature_for_prompt, parse_llm_json

_VALID_VERDICTS = {"PASS", "REVISE", "ABORT"}
# Default valid agents for revision targeting (overridden by TaskTemplate at runtime)
_DEFAULT_VALID_AGENTS = {"ProblemFormulator", "DataEngineer", "Analyst"}

# Backward-compatible alias for tests that import this constant
_VALID_AGENTS = _DEFAULT_VALID_AGENTS

_REQUIRED_REVIEW_KEYS = {
    "overall_verdict",
    "overall_quality_score",
    "problem_formulation_review",
    "data_preparation_review",
    "analysis_review",
    "substantive_review",
    "revision_instructions",
}


_OPTIONAL_REVIEW_KEYS: set[str] = {"novelty_review"}


class Critic(BaseAgent):
    """Reviews research_spec, data_report, and results to issue a quality verdict."""

    def run(
        self,
        research_spec: dict | None = None,
        data_report: dict | None = None,
        results_object: dict | None = None,
        revision_cycle: int | None = None,
        findings_memory_summary: str = "",
        pre_critic_failures: list | None = None,
        **kwargs: Any,
    ) -> dict:
        """
        Args:
            research_spec: ProblemFormulator output (falls back to ctx if None).
            data_report: DataEngineer output (falls back to ctx if None).
            results_object: Analyst output (falls back to ctx if None).
            revision_cycle: Current revision cycle (falls back to ctx.revision_cycle).

        Returns:
            review_report dict matching the SPEC §4.4 schema.
        """
        spec = research_spec if research_spec is not None else self.ctx.research_spec
        report = data_report if data_report is not None else self.ctx.data_report
        results = results_object if results_object is not None else self.ctx.results_object
        cycle = revision_cycle if revision_cycle is not None else self.ctx.revision_cycle

        registry = self.load_registry()
        task_template = self.load_task_template()
        checklist = self._load_checklist()

        user_message = self._build_user_message(
            research_spec=spec,
            literature_context=self.ctx.literature_context,
            data_report=report,
            results_object=results,
            registry=registry,
            task_template=task_template,
            checklist=checklist,
            revision_cycle=cycle,
            findings_memory_summary=findings_memory_summary,
            pre_critic_failures=pre_critic_failures or [],
        )

        llm_response = self.call_llm(user_message)

        # Extract the LAST ```json block — response may have reasoning preamble
        # (LENS A/B/C sections) before the final JSON deliverable
        json_text = self._extract_last_json_block(llm_response)
        try:
            review_report = parse_llm_json(json_text)
        except json.JSONDecodeError as first_err:
            # F-B2-CRITIC-MALFORMED-JSON (Phase B first live run): the LLM
            # occasionally emits an unterminated string / truncated JSON.
            # One targeted re-prompt before the SPEC §8 abort — same
            # philosophy as the DE/Analyst fix loops.
            self.ctx.log.append({
                "timestamp": datetime.utcnow().isoformat(),
                "agent": self.agent_name,
                "message": (
                    f"Critic response JSON malformed ({first_err}); "
                    "one re-prompt retry before abort."
                ),
            })
            retry_response = self.call_llm(
                user_message
                + "\n\nIMPORTANT: Your previous response's JSON was malformed "
                f"({first_err}). Re-emit the COMPLETE review_report as a single "
                "valid ```json code block. Escape all quotes and newlines "
                "inside strings; do not truncate."
            )
            json_text = self._extract_last_json_block(retry_response)
            review_report = parse_llm_json(json_text)  # 2nd failure aborts per SPEC §8
            llm_response = retry_response
        review_report = self._validate_review_report(review_report)

        # Persist reasoning scratchpad for debugging (everything before the last ```json)
        json_fence_pos = llm_response.rfind("```json")
        if json_fence_pos > 0:
            reasoning = llm_response[:json_fence_pos].strip()
            if reasoning:
                reasoning_path = os.path.join(self.ctx.output_dir, "critic_reasoning.txt")
                with open(reasoning_path, "w", encoding="utf-8") as f:
                    f.write(reasoning)

        # Persist to disk. The orchestrator writes the effective verdict
        # and the `unverified` flag onto this dict AFTER this returns, so
        # persist_review_report() is called again from there; this write
        # is the crash-safe copy.
        self.persist_review_report(review_report)
        return review_report

    def persist_review_report(self, review_report: dict) -> str:
        """Write ``review_report.json``. Safe to call more than once.

        The file used to be written here and never again, while the
        effective verdict and the `unverified` flag were stamped onto the
        in-memory dict by the orchestrator a few lines later. So the
        persisted artifact -- the one every downstream reader, every
        audit and every archived run actually has -- carried only the
        LLM's raw verdict and no flag at all. Six recent runs on disk
        show `unverified` in checkpoint.json and absent from
        review_report.json beside it, one of them with the two
        disagreeing. The orchestrator calls this again once the verdict
        is settled.
        """
        report_path = os.path.join(self.ctx.output_dir, "review_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(review_report, f, indent=2)
        return report_path

    @staticmethod
    def _extract_last_json_block(text: str) -> str:
        """Extract the last ```json ... ``` block from an LLM response.

        The multi-persona reasoning protocol produces LENS A/B/C prose before the
        final JSON block. We want the *last* JSON block so the reasoning scratchpad
        doesn't confuse ``parse_llm_json``.
        """
        matches = list(re.finditer(r"```json\s*\n(.*?)```", text, re.DOTALL))
        if matches:
            return matches[-1].group(1).strip()
        # No fenced block — return the full response and let parse_llm_json handle it
        return text

    # ------------------------------------------------------------------
    # File loaders
    # ------------------------------------------------------------------

    def _load_checklist(self) -> dict:
        """Load the methodological checklist YAML used during review."""
        checklist_path = self.task_template.get_critic_checklist_path()
        with open(checklist_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    # ------------------------------------------------------------------
    # Message builders
    # ------------------------------------------------------------------

    #: Config settings that checklist rows are written as conditions on.
    #: Each earned its place by appearing in a rule the Critic is asked to
    #: apply: an_01 keys off mlp_enabled, the SMOTE rows off
    #: minority_threshold and ablation_enabled.
    CHECKLIST_CONFIG_KEYS: tuple[tuple[str, str], ...] = (
        ("pipeline", "mlp_enabled"),
        ("class_imbalance", "minority_threshold"),
        ("class_imbalance", "ablation_enabled"),
        ("pipeline", "random_state"),
        ("pipeline", "task_type"),
    )

    def _checklist_relevant_config(self) -> dict:
        """Report the settings the checklist's conditional rows depend on.

        The checklist says "at least 5 individual model families ... (When
        `mlp_enabled: false`, 4 individual + Stacking = 5 total is
        acceptable.)". The shipped default IS false, so four models is the
        correct outcome -- but the only occurrence of `mlp_enabled` in the
        rendered prompt was inside that rule's own text. The Critic was
        handed a condition and never told whether it held, so it could not
        apply the carve-out and flagged an_01 major on a run that had done
        nothing wrong. Every prediction run on the default config earned
        the same unavoidable finding.

        This is the same shape as guidance that reaches one agent and no
        other: the rule existed, the fact needed to evaluate it did not
        arrive.
        """
        config = getattr(self, "config", None) or {}
        relevant: dict = {}
        for section, key in self.CHECKLIST_CONFIG_KEYS:
            block = config.get(section)
            if isinstance(block, dict) and key in block:
                relevant[f"{section}.{key}"] = block[key]
        return relevant

    def _build_user_message(
        self,
        research_spec: dict | None,
        literature_context: dict | None,
        data_report: dict | None,
        results_object: dict | None,
        registry: dict,
        task_template: dict,
        checklist: dict,
        revision_cycle: int,
        findings_memory_summary: str = "",
        pre_critic_failures: list | None = None,
    ) -> str:
        """Assemble the Critic's user message.

        See ``_checklist_relevant_config`` for why the run's configuration
        is included: several checklist rows are conditional on settings the
        Critic could not otherwise observe.
        """
        max_cycles = self.ctx.max_revision_cycles
        at_max = revision_cycle >= max_cycles

        parts = [
            "## Run Configuration (settings this run's checklist depends on)",
            "```json",
            json.dumps(self._checklist_relevant_config(), indent=2),
            "```",
            "",
            "## research_spec.json",
            "```json",
            json.dumps(research_spec or {}, indent=2),
            "```",
            "",
            "## literature_context.json",
            "```json",
            json.dumps(literature_for_prompt(literature_context or {}), indent=2),
            "```",
            "",
            "## data_report.json",
            "```json",
            json.dumps(data_report or {}, indent=2),
            "```",
            "",
            "## results.json",
            "```json",
            json.dumps(results_object or {}, indent=2),
            "```",
            "",
            "## Dataset Registry (YAML)",
            "```yaml",
            yaml.dump(registry, default_flow_style=False, allow_unicode=True),
            "```",
            "",
            "## Task Template",
            "```yaml",
            yaml.dump(task_template, default_flow_style=False, allow_unicode=True),
            "```",
            "",
            "## Methodological Checklist",
            "```yaml",
            yaml.dump(checklist, default_flow_style=False, allow_unicode=True),
            "```",
            "",
            "## Revision Context",
            f"- current_revision_cycle: {revision_cycle}",
            f"- max_revision_cycles: {max_cycles}",
        ]

        if at_max:
            # Report the true verdict. This block used to say "If your
            # verdict would be REVISE, set it to PASS instead", and the
            # orchestrator does not need that: _run_critiquing already
            # routes REVISE-at-max-cycles to WRITING with unverified=True.
            # What the instruction did do was corrupt the only
            # LLM-vs-deterministic cross-check in the pipeline --
            # verdict_evaluator compares the LLM's verdict string against
            # its own recomputation and logs a disagreement, so telling
            # the model to relabel REVISE as PASS made `llm_disagreement`
            # read False in precisely the runs where the model disagreed.
            # One run shipped at PASS with quality_score 9/10 holding
            # eight critical defects.
            parts.append(
                "- NOTE: Max revision cycles have been reached, so no further "
                "revision is possible. Report your TRUE verdict anyway. A "
                "REVISE here does not abort the run: the orchestrator writes "
                "the paper with a mandatory UNVERIFIED warning block and "
                "attaches your full report. Do NOT soften the verdict to PASS "
                "-- the warning block is the only thing that tells a reader "
                "the issues were left unresolved."
            )

        if findings_memory_summary:
            parts += [
                "",
                "## Findings Memory Summary",
                findings_memory_summary,
            ]

        if pre_critic_failures:
            failure_lines = [
                f"- [{f.severity}] {f.check_id} (target: {f.target_agent}): {f.message}"
                for f in pre_critic_failures
            ]
            parts += [
                "",
                "## Pre-Critic Automated Checks (already confirmed — do not re-derive)",
                "The following issues were detected deterministically before this review:",
                "\n".join(failure_lines),
                "Treat these as confirmed findings when scoring the relevant sections.",
            ]

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_review_report(self, report: dict) -> dict:
        """
        Ensure the review_report conforms to the expected schema.

        - All required keys present (defaulted to None if missing).
        - overall_verdict is one of PASS / REVISE / ABORT.
        - revision_instructions only targets known agents.
        """
        # The cascade order is what _execute_revisions may RE-RUN. The
        # Writer is not in it and must not be: CRITIQUING precedes
        # WRITING, so at this point the Writer has not run yet and there
        # is nothing to re-run.
        #
        # But that list was also used to decide which instructions were
        # allowed to EXIST, and those are different questions. Across 19
        # archived reports the Critic addressed 41 findings to the Writer
        # -- a third of everything it found -- and every one was deleted
        # here, six runs carrying the receipt in _validation_errors. A
        # project skill (prediction-critic-checklist row sv_05) tells the
        # Critic to write into that key. It was writing into a hole.
        #
        # Writer-addressed instructions are now kept and carried forward
        # to the Writer's first invocation, where they are obligations on
        # the manuscript it is about to produce.
        cascade_agents = set(self.task_template.get_agent_order())
        valid_agents = cascade_agents | {"Writer"}

        # Default missing required keys
        for key in _REQUIRED_REVIEW_KEYS:
            if key not in report:
                report[key] = None

        # Validate verdict
        verdict = report.get("overall_verdict", "")
        if verdict not in _VALID_VERDICTS:
            report.setdefault("_validation_errors", []).append(
                f"Invalid verdict '{verdict}' replaced with ABORT"
            )
            report["overall_verdict"] = "ABORT"

        # Validate and normalise revision_instructions
        raw_ri = report.get("revision_instructions")
        if not isinstance(raw_ri, dict):
            report["revision_instructions"] = {a: None for a in valid_agents}
        else:
            valid_ri: dict[str, Any] = {}
            for agent, instr in raw_ri.items():
                if agent in valid_agents:
                    valid_ri[agent] = instr
                else:
                    report.setdefault("_validation_errors", []).append(
                        f"revision_instructions targeted unknown agent '{agent}'; ignored"
                    )
            # Ensure every valid agent has an entry (even if None)
            for agent in valid_agents:
                valid_ri.setdefault(agent, None)
            report["revision_instructions"] = valid_ri

        return report
