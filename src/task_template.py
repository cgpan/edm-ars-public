"""TaskTemplate abstraction: encapsulates task-type-specific logic.

Each task type (prediction, causal inference, etc.) implements this ABC.
Agents and the orchestrator consume the template to get task-specific
configuration without hardcoding assumptions.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.dataset_adapter import DatasetAdapter


class TaskTemplate(ABC):
    """Abstract base for task-type-specific configuration and validation."""

    @abstractmethod
    def get_name(self) -> str:
        """Return the task type identifier (e.g. 'prediction')."""
        ...

    @abstractmethod
    def get_agent_order(self) -> list[str]:
        """Return the ordered list of agent names for the revision cascade."""
        ...

    @abstractmethod
    def get_evaluation_metrics(self, outcome_type: str) -> dict:
        """Return metric configuration for the given outcome type.

        Returns a dict with keys like 'primary', 'suspicion_threshold', etc.
        """
        ...

    @abstractmethod
    def get_critic_checklist_path(self) -> str:
        """Return the path to the methodological checklist YAML for this task type."""
        ...

    @abstractmethod
    def get_paper_template_path(self, config: dict) -> str:
        """Return the path to the LaTeX paper template."""
        ...

    @abstractmethod
    def validate_research_spec(
        self,
        spec: dict,
        registry: dict,
        dataset_adapter: DatasetAdapter,
    ) -> list[str]:
        """Task-specific validation of the research specification.

        Returns a list of warning strings (non-fatal; Critic enforces hard failures).
        """
        ...


class PredictionTemplate(TaskTemplate):
    """Prediction task type — the original EDM-ARS v1 workflow."""

    def get_name(self) -> str:
        return "prediction"

    def get_agent_order(self) -> list[str]:
        return ["ProblemFormulator", "DataEngineer", "Analyst"]

    def get_evaluation_metrics(self, outcome_type: str) -> dict:
        if outcome_type == "binary":
            return {
                "primary": "AUC",
                "suspicion_threshold": 0.95,
                "higher_is_better": True,
            }
        return {
            "primary": "RMSE",
            "suspicion_threshold": None,
            "higher_is_better": False,
        }

    def get_critic_checklist_path(self) -> str:
        return "data_registry/evaluation_rubrics/methodological_checklist.yaml"

    def get_paper_template_path(self, config: dict) -> str:
        return config["paths"]["paper_template"]

    def validate_research_spec(
        self,
        spec: dict,
        registry: dict,
        dataset_adapter: DatasetAdapter,
    ) -> list[str]:
        """Validate a prediction research spec.

        Checks:
          1. Temporal ordering — all predictor waves precede the outcome wave.
          2. Feasibility — estimated analytic_n >= 10,000.
          3. Novelty score — novelty_score_self_assessment >= 3.
        """
        from src.agents.problem_formulator import _build_registry_var_map

        warnings: list[str] = []
        temporal_order: list[str] = registry.get(
            "temporal_order", dataset_adapter.get_temporal_order()
        )
        var_map = _build_registry_var_map(registry)

        outcome_var: str = spec.get("outcome_variable", "")
        predictor_set: list[dict] = spec.get("predictor_set", [])

        # --- 1. Temporal ordering ---
        outcome_meta = var_map.get(outcome_var, {})
        outcome_wave: str | None = outcome_meta.get("wave")

        if not outcome_wave:
            warnings.append(
                f"Outcome variable '{outcome_var}' not found in registry or has no "
                "wave metadata; temporal ordering cannot be verified."
            )
        else:
            if outcome_wave not in temporal_order:
                warnings.append(
                    f"Outcome wave '{outcome_wave}' is not in temporal_order {temporal_order}."
                )
            else:
                outcome_idx = temporal_order.index(outcome_wave)
                for pred in predictor_set:
                    pred_var = pred.get("variable", "")
                    pred_wave = pred.get("wave", "")
                    if pred_wave not in temporal_order:
                        warnings.append(
                            f"Predictor '{pred_var}' has unknown wave '{pred_wave}'."
                        )
                        continue
                    pred_idx = temporal_order.index(pred_wave)
                    if pred_idx >= outcome_idx:
                        warnings.append(
                            f"TEMPORAL VIOLATION: predictor '{pred_var}' "
                            f"(wave={pred_wave}, idx={pred_idx}) does not precede "
                            f"outcome '{outcome_var}' (wave={outcome_wave}, "
                            f"idx={outcome_idx}). This predictor should be removed."
                        )

        # --- 2. Feasibility: estimated analytic_n >= 10,000 ---
        n_full: int = registry.get("levels", {}).get(
            "student", dataset_adapter.get_sample_size()
        )
        outcome_pct_missing: float = outcome_meta.get("pct_missing", 0.0) / 100.0

        total_predictor_missing: float = 0.0
        for pred in predictor_set:
            pred_meta = var_map.get(pred.get("variable", ""), {})
            total_predictor_missing += pred_meta.get("pct_missing", 0.0) / 100.0

        retention = max(
            0.0,
            1.0 - outcome_pct_missing - total_predictor_missing,
        )
        estimated_n = n_full * retention
        if estimated_n < 10_000:
            warnings.append(
                f"Estimated analytic_n ({estimated_n:.0f}) may fall below 10,000 "
                "based on registry missingness. Consider swapping high-missingness "
                "predictors or relaxing the predictor set."
            )

        # --- 3. Novelty score ---
        novelty_score = spec.get("novelty_score_self_assessment")
        if isinstance(novelty_score, (int, float)) and novelty_score < 3:
            warnings.append(
                f"novelty_score_self_assessment = {novelty_score} is below the "
                "minimum of 3. The research question lacks sufficient novelty."
            )

        return warnings


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_TASK_REGISTRY: dict[str, type[TaskTemplate]] = {
    "prediction": PredictionTemplate,
}


def create_task_template(task_type: str) -> TaskTemplate:
    """Create a TaskTemplate instance for the given task type."""
    cls = _TASK_REGISTRY.get(task_type)
    if cls is None:
        raise ValueError(
            f"Unknown task_type: {task_type!r}. "
            f"Available: {sorted(_TASK_REGISTRY.keys())}"
        )
    return cls()
