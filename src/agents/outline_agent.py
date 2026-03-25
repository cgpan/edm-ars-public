"""OutlineAgent: generates a paper-specific outline before prose generation."""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

from src.agents.base import BaseAgent, load_prompt, parse_llm_json


def _with_model_alias(config: dict, alias: str, source: str) -> dict:
    """Return a config copy where *alias* maps to the same model as *source*.

    This lets the OutlineAgent reuse the writer's model without adding a
    dedicated entry to config.yaml.
    """
    config = {**config}
    if "models" in config:
        models = dict(config["models"])
        if alias not in models and source in models:
            models[alias] = models[source]
        config["models"] = models
    if "minimax" in config:
        mm = dict(config["minimax"])
        mm_models = dict(mm.get("models", {}))
        if alias not in mm_models and source in mm_models:
            mm_models[alias] = mm_models[source]
        mm["models"] = mm_models
        config["minimax"] = mm
    return config


class OutlineAgent(BaseAgent):
    """Generates a paper-specific outline based on pipeline results."""

    def __init__(
        self,
        context: Any,
        agent_name: str,
        config: dict,
        **kwargs: Any,
    ) -> None:
        # OutlineAgent reuses the writer's model — inject mapping before
        # BaseAgent resolves it.
        prompt_data = load_prompt(agent_name, config)
        model_key = prompt_data.get("model_config_key", "writer")
        config = _with_model_alias(config, agent_name, model_key)
        super().__init__(context, agent_name, config, **kwargs)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        research_spec: dict | None = None,
        data_report: dict | None = None,
        results_object: dict | None = None,
        **kwargs: Any,
    ) -> dict:
        """Generate a paper outline from pipeline results.

        Returns:
            The outline dict (also saved as ``paper_outline.json``).
        """
        spec = research_spec if research_spec is not None else self.ctx.research_spec
        report = data_report if data_report is not None else self.ctx.data_report
        results = results_object if results_object is not None else self.ctx.results_object

        triggers = self._detect_emphasis_triggers(results or {}, report or {})

        user_message = self._build_user_message(
            research_spec=spec,
            data_report=report,
            results_object=results,
            triggers=triggers,
        )

        llm_response = self.call_llm(user_message, max_tokens=self.max_tokens)
        outline = parse_llm_json(llm_response)

        # Persist to output directory
        outline_path = os.path.join(self.ctx.output_dir, "paper_outline.json")
        with open(outline_path, "w", encoding="utf-8") as f:
            json.dump(outline, f, indent=2)

        return outline

    # ------------------------------------------------------------------
    # Emphasis trigger detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_emphasis_triggers(results: dict, data_report: dict) -> dict:
        """Analyze results to find structural emphasis signals."""
        triggers: dict[str, Any] = {}

        # --- Model convergence ---
        all_models = results.get("all_models", {})
        primary_metric = results.get("primary_metric", "AUC").lower()
        metric_key = _metric_key(primary_metric)
        metrics = [
            m.get(metric_key, 0.0)
            for m in all_models.values()
            if isinstance(m, dict) and metric_key in m
        ]
        if len(metrics) >= 2:
            metric_range = max(metrics) - min(metrics)
            triggers["models_similar"] = metric_range < 0.02
        else:
            triggers["models_similar"] = False

        # --- Surprising SHAP features ---
        obvious = {
            "X1TXMTSCOR", "X2TXMTSCOR", "X1SES", "X1SES_U",
            "X2STUEDEXPCT", "X1STUEDEXPCT", "X1PAREDU",
        }
        top5 = [f["feature"] for f in results.get("top_features", [])[:5]]
        # Strip one-hot suffixes (e.g. "X1RACE_3.0" → "X1RACE")
        raw_top5 = [f.split("_")[0] if "_" in f else f for f in top5]
        triggers["surprising_features"] = [
            feat for feat, raw in zip(top5, raw_top5) if raw not in obvious
        ]

        # --- Subgroup gaps > 5% ---
        subgroup_perf = results.get("subgroup_performance", {})
        gaps: list[dict[str, Any]] = []
        for attr, groups in subgroup_perf.items():
            if not isinstance(groups, dict):
                continue
            vals = [
                g.get(metric_key, g.get("auc", g.get("rmse", 0.0)))
                for g in groups.values()
                if isinstance(g, dict)
            ]
            if len(vals) >= 2:
                gap = max(vals) - min(vals)
                if gap > 0.05:
                    gaps.append({"attribute": attr, "gap": round(gap, 4)})
        triggers["subgroup_gaps_over_5pct"] = gaps

        # --- Sensitivity analysis significance ---
        sens = results.get("sensitivity_analysis")
        triggers["sensitivity_significant"] = bool(
            sens and sens.get("significant_change")
        )

        # --- ICC non-negligible ---
        icc = results.get("icc", {})
        triggers["icc_nonnegligible"] = icc.get("icc", 0) >= 0.05

        return triggers

    # ------------------------------------------------------------------
    # Message builder
    # ------------------------------------------------------------------

    def _build_user_message(
        self,
        research_spec: dict | None,
        data_report: dict | None,
        results_object: dict | None,
        triggers: dict,
    ) -> str:
        parts = [
            "## research_spec.json",
            "```json",
            json.dumps(research_spec or {}, indent=2),
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
            "## Emphasis Triggers (pre-computed)",
            "```json",
            json.dumps(triggers, indent=2),
            "```",
            "",
            "## Task",
            (
                "Design a paper-specific outline for this study. "
                "Use the emphasis triggers to decide which findings deserve "
                "expanded treatment and which can be compressed. "
                "Output ONLY the outline JSON in a ```json code block."
            ),
        ]
        return "\n".join(parts)


def _metric_key(primary_metric: str) -> str:
    """Map primary metric name to the key used in all_models dicts."""
    mapping = {
        "auc": "auc",
        "auc-roc": "auc",
        "rmse": "rmse",
        "r2": "r2",
    }
    return mapping.get(primary_metric.lower(), primary_metric.lower())
