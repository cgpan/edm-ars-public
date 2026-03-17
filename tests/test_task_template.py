"""Tests for TaskTemplate abstraction and PredictionTemplate."""
import pytest

from src.task_template import (
    PredictionTemplate,
    TaskTemplate,
    create_task_template,
)


class TestPredictionTemplate:
    def setup_method(self) -> None:
        self.template = PredictionTemplate()

    def test_get_name(self) -> None:
        assert self.template.get_name() == "prediction"

    def test_get_agent_order(self) -> None:
        order = self.template.get_agent_order()
        assert order == ["ProblemFormulator", "DataEngineer", "Analyst"]

    def test_get_evaluation_metrics_binary(self) -> None:
        metrics = self.template.get_evaluation_metrics("binary")
        assert metrics["primary"] == "AUC"
        assert metrics["suspicion_threshold"] == 0.95
        assert metrics["higher_is_better"] is True

    def test_get_evaluation_metrics_continuous(self) -> None:
        metrics = self.template.get_evaluation_metrics("continuous")
        assert metrics["primary"] == "RMSE"
        assert metrics["suspicion_threshold"] is None
        assert metrics["higher_is_better"] is False

    def test_get_critic_checklist_path(self) -> None:
        path = self.template.get_critic_checklist_path()
        assert "methodological_checklist.yaml" in path

    def test_get_paper_template_path(self) -> None:
        config = {"paths": {"paper_template": "templates/paper_template.tex"}}
        path = self.template.get_paper_template_path(config)
        assert path == "templates/paper_template.tex"

    def test_is_task_template_subclass(self) -> None:
        assert isinstance(self.template, TaskTemplate)


class TestCreateTaskTemplate:
    def test_prediction(self) -> None:
        template = create_task_template("prediction")
        assert isinstance(template, PredictionTemplate)
        assert template.get_name() == "prediction"

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown task_type"):
            create_task_template("nonexistent_task")


class TestPredictionValidateResearchSpec:
    """Test the validation logic extracted from ProblemFormulator."""

    def setup_method(self) -> None:
        self.template = PredictionTemplate()

    def test_temporal_violation(self) -> None:
        from src.dataset_adapter import HSLS09Adapter

        adapter = HSLS09Adapter()
        registry = {
            "temporal_order": ["base_year", "first_follow_up", "second_follow_up"],
            "variables": {
                "outcomes": [{"name": "X2VAR", "wave": "first_follow_up"}],
                "predictors": {
                    "academic": [{"name": "X2PRED", "wave": "first_follow_up"}],
                },
            },
        }
        spec = {
            "outcome_variable": "X2VAR",
            "predictor_set": [{"variable": "X2PRED", "wave": "first_follow_up"}],
        }
        warnings = self.template.validate_research_spec(spec, registry, adapter)
        assert any("TEMPORAL VIOLATION" in w for w in warnings)

    def test_low_novelty_score(self) -> None:
        from src.dataset_adapter import HSLS09Adapter

        adapter = HSLS09Adapter()
        registry = {
            "temporal_order": ["base_year", "first_follow_up"],
            "variables": {"outcomes": [], "predictors": {}},
        }
        spec = {
            "outcome_variable": "X2VAR",
            "predictor_set": [],
            "novelty_score_self_assessment": 2,
        }
        warnings = self.template.validate_research_spec(spec, registry, adapter)
        assert any("novelty_score_self_assessment" in w for w in warnings)

    def test_clean_spec_no_warnings(self) -> None:
        from src.dataset_adapter import HSLS09Adapter

        adapter = HSLS09Adapter()
        registry = {
            "temporal_order": ["base_year", "first_follow_up"],
            "levels": {"student": 23503},
            "variables": {
                "outcomes": [
                    {"name": "X2VAR", "wave": "first_follow_up", "pct_missing": 5.0}
                ],
                "predictors": {
                    "academic": [
                        {"name": "X1PRED", "wave": "base_year", "pct_missing": 2.0}
                    ],
                },
            },
        }
        spec = {
            "outcome_variable": "X2VAR",
            "predictor_set": [{"variable": "X1PRED", "wave": "base_year"}],
            "novelty_score_self_assessment": 4,
        }
        warnings = self.template.validate_research_spec(spec, registry, adapter)
        assert warnings == []
