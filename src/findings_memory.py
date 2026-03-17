"""FindingsMemory: persistent cross-run knowledge store for EDM-ARS.

Accumulates RunEntry records from each completed pipeline run. Provides
a human-readable summary string that ProblemFormulator and Critic can
inject into their prompts to avoid repeating prior work and build on
open questions.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from src.context import PipelineContext


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RunEntry:
    """Snapshot of a single pipeline run, stored in findings memory."""

    run_id: str
    dataset: str
    task_type: str
    outcome_variable: str
    predictor_set: list[str]
    best_model: str
    best_metric_value: float
    primary_metric: str
    verdict: str                      # "PASS" | "REVISE" | "ABORT"
    quality_score: int | None
    top_features: list[str]           # top 5 SHAP feature names
    open_questions: list[str]         # from Critic / Writer
    research_question: str            # one-line summary
    timestamp: str                    # ISO-8601

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "dataset": self.dataset,
            "task_type": self.task_type,
            "outcome_variable": self.outcome_variable,
            "predictor_set": self.predictor_set,
            "best_model": self.best_model,
            "best_metric_value": self.best_metric_value,
            "primary_metric": self.primary_metric,
            "verdict": self.verdict,
            "quality_score": self.quality_score,
            "top_features": self.top_features,
            "open_questions": self.open_questions,
            "research_question": self.research_question,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunEntry:
        return cls(
            run_id=data.get("run_id", ""),
            dataset=data.get("dataset", ""),
            task_type=data.get("task_type", "prediction"),
            outcome_variable=data.get("outcome_variable", ""),
            predictor_set=data.get("predictor_set", []),
            best_model=data.get("best_model", ""),
            best_metric_value=float(data.get("best_metric_value", 0.0)),
            primary_metric=data.get("primary_metric", ""),
            verdict=data.get("verdict", ""),
            quality_score=data.get("quality_score"),
            top_features=data.get("top_features", []),
            open_questions=data.get("open_questions", []),
            research_question=data.get("research_question", ""),
            timestamp=data.get("timestamp", ""),
        )

    @classmethod
    def from_pipeline_context(
        cls,
        ctx: PipelineContext,
        run_id: str,
        runtime_minutes: float | None = None,
        api_cost_usd: float | None = None,
    ) -> RunEntry:
        """Build a RunEntry from a completed PipelineContext."""
        spec = ctx.research_spec or {}
        results = ctx.results_object or {}
        review = ctx.review_report or {}

        outcome_var = spec.get("outcome_variable", "")
        predictor_set = [
            p.get("variable", "") for p in spec.get("predictor_set", [])
        ]
        research_question = spec.get("research_question", "")

        best_model = results.get("best_model", "")
        best_metric_value = float(results.get("best_metric_value", 0.0))
        primary_metric = results.get("primary_metric", "")
        top_features = [
            f["feature"] for f in results.get("top_features", [])[:5]
            if isinstance(f, dict) and "feature" in f
        ]

        verdict = review.get("overall_verdict", "")
        quality_score = review.get("overall_quality_score")
        if isinstance(quality_score, (int, float)):
            quality_score = int(quality_score)
        else:
            quality_score = None

        # Gather open questions from Critic substantive_review
        open_questions: list[str] = []
        substantive = review.get("substantive_review", {})
        if isinstance(substantive, dict):
            for issue in substantive.get("issues", []):
                if isinstance(issue, dict) and issue.get("severity") in ("minor", "major"):
                    desc = issue.get("description", "")
                    if desc:
                        open_questions.append(desc)
        # Limit to 5 to keep memory compact
        open_questions = open_questions[:5]

        return cls(
            run_id=run_id,
            dataset=ctx.dataset_name,
            task_type=getattr(ctx, "task_type", "prediction"),
            outcome_variable=outcome_var,
            predictor_set=predictor_set,
            best_model=best_model,
            best_metric_value=best_metric_value,
            primary_metric=primary_metric,
            verdict=verdict,
            quality_score=quality_score,
            top_features=top_features,
            open_questions=open_questions,
            research_question=research_question,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )


@dataclass
class KnowledgeGraph:
    """Cross-run aggregate knowledge derived from all RunEntry records."""

    studied_outcomes: dict[str, list[str]] = field(default_factory=dict)
    # outcome_variable → list of run_ids that studied it
    strong_predictors: dict[str, float] = field(default_factory=dict)
    # feature_name → mean shap_mean_abs across runs (approximated from top_features counts)
    open_questions: list[str] = field(default_factory=list)
    last_updated: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "studied_outcomes": self.studied_outcomes,
            "strong_predictors": self.strong_predictors,
            "open_questions": self.open_questions,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KnowledgeGraph:
        return cls(
            studied_outcomes=data.get("studied_outcomes", {}),
            strong_predictors=data.get("strong_predictors", {}),
            open_questions=data.get("open_questions", []),
            last_updated=data.get("last_updated", ""),
        )


# ---------------------------------------------------------------------------
# FindingsMemory
# ---------------------------------------------------------------------------


class FindingsMemory:
    """Persistent cross-run knowledge store.

    Loaded at orchestrator startup; updated after each terminal state
    (COMPLETED or ABORTED). All I/O failures are non-fatal.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self.runs: list[RunEntry] = []
        self.knowledge_graph: KnowledgeGraph = KnowledgeGraph()

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str) -> FindingsMemory:
        """Load from YAML file; return empty instance if file missing or corrupt."""
        instance = cls(path)
        if not os.path.exists(path):
            return instance
        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            for run_data in data.get("runs", []):
                try:
                    instance.runs.append(RunEntry.from_dict(run_data))
                except Exception:
                    pass  # skip malformed entries
            kg_data = data.get("knowledge_graph", {})
            if kg_data:
                instance.knowledge_graph = KnowledgeGraph.from_dict(kg_data)
        except Exception:
            pass  # corrupt file — start fresh
        return instance

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_run(self, entry: RunEntry) -> None:
        """Append a new run and refresh the knowledge graph."""
        self.runs.append(entry)
        self._update_knowledge_graph(entry)

    def _update_knowledge_graph(self, entry: RunEntry) -> None:
        """Update aggregate knowledge from the new entry."""
        kg = self.knowledge_graph

        # Track studied outcomes
        if entry.outcome_variable:
            kg.studied_outcomes.setdefault(entry.outcome_variable, [])
            if entry.run_id not in kg.studied_outcomes[entry.outcome_variable]:
                kg.studied_outcomes[entry.outcome_variable].append(entry.run_id)

        # Track strong predictors (frequency-based proxy for importance)
        for feat in entry.top_features:
            kg.strong_predictors[feat] = kg.strong_predictors.get(feat, 0) + 1

        # Accumulate and deduplicate open questions (keep most recent 20)
        for q in entry.open_questions:
            if q and q not in kg.open_questions:
                kg.open_questions.append(q)
        kg.open_questions = kg.open_questions[-20:]

        kg.last_updated = datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Atomically write memory to YAML (tmp rename prevents corruption)."""
        os.makedirs(os.path.dirname(self.path) if os.path.dirname(self.path) else ".", exist_ok=True)
        data = {
            "runs": [r.to_dict() for r in self.runs],
            "knowledge_graph": self.knowledge_graph.to_dict(),
        }
        tmp_path = self.path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        os.replace(tmp_path, self.path)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_studied_outcomes(self) -> list[str]:
        """Return list of outcome variables studied in prior runs."""
        return list(self.knowledge_graph.studied_outcomes.keys())

    def get_open_questions(self) -> list[str]:
        """Return curated list of open questions from prior runs."""
        return list(self.knowledge_graph.open_questions)

    # ------------------------------------------------------------------
    # Summary for LLM injection
    # ------------------------------------------------------------------

    def to_summary_str(self) -> str:
        """Return a ~500-token human-readable summary for LLM prompt injection.

        Returns empty string if no runs are recorded.
        """
        if not self.runs:
            return ""

        lines: list[str] = [
            f"Prior runs recorded: {len(self.runs)}",
            "",
        ]

        # Studied outcomes
        studied = self.knowledge_graph.studied_outcomes
        if studied:
            lines.append("**Outcomes already studied:**")
            for outcome, run_ids in studied.items():
                lines.append(f"  - {outcome} (studied in {len(run_ids)} run(s))")
            lines.append("")

        # Strong predictors (top 10 by frequency)
        strong = sorted(
            self.knowledge_graph.strong_predictors.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:10]
        if strong:
            lines.append("**Frequently important predictors:**")
            for feat, count in strong:
                lines.append(f"  - {feat} (appeared in {int(count)} run(s))")
            lines.append("")

        # Recent run summaries (last 3)
        lines.append("**Recent run summaries:**")
        for entry in self.runs[-3:]:
            verdict_str = f"{entry.verdict}"
            if entry.quality_score is not None:
                verdict_str += f" (score={entry.quality_score})"
            metric_str = (
                f"{entry.primary_metric}={entry.best_metric_value:.3f}"
                if entry.best_metric_value
                else ""
            )
            lines.append(
                f"  - [{entry.run_id}] outcome={entry.outcome_variable}, "
                f"best_model={entry.best_model}, {metric_str}, verdict={verdict_str}"
            )
            if entry.research_question:
                lines.append(f"    RQ: {entry.research_question[:120]}")
        lines.append("")

        # Open questions
        questions = self.knowledge_graph.open_questions[-5:]
        if questions:
            lines.append("**Open questions from prior runs:**")
            for q in questions:
                lines.append(f"  - {q[:150]}")

        return "\n".join(lines)
