"""Deterministic pre-Critic validation checks for EDM-ARS.

Inspired by AutoResearchClaw health.py: a fast, zero-LLM validation layer that
catches obvious pipeline failures before the expensive Critic (Opus) call is made.

If critical failures are found, the Orchestrator can short-circuit and synthesise a
REVISE/ABORT review_report without burning an Opus API call.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CheckFailure:
    check_id: str
    severity: str  # "critical" | "major"
    message: str
    target_agent: str  # "ProblemFormulator" | "DataEngineer" | "Analyst"


@dataclass
class PreCriticResult:
    failures: list[CheckFailure] = field(default_factory=list)

    @property
    def has_critical(self) -> bool:
        return any(f.severity == "critical" for f in self.failures)

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.failures if f.severity == "critical")

    @property
    def major_count(self) -> int:
        return sum(1 for f in self.failures if f.severity == "major")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_pre_critic_checks(ctx: object, output_dir: str) -> PreCriticResult:
    """Run all deterministic pre-Critic checks and return a :class:`PreCriticResult`.

    Parameters
    ----------
    ctx:
        The ``PipelineContext`` object (typed as ``object`` to avoid circular import).
        Must expose ``.research_spec``, ``.results_object``, and ``.data_report`` attrs.
    output_dir:
        Absolute path to the run's output directory.
    """
    result = PreCriticResult()
    _check_outcome_not_in_train_x(ctx, output_dir, result)
    _check_model_count(ctx, result)
    _check_required_figures(output_dir, result)
    _check_top_features_present(ctx, result)
    _check_subgroup_performance_present(ctx, result)
    _check_data_report_validation_passed(ctx, result)
    return result


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_outcome_not_in_train_x(
    ctx: object, output_dir: str, result: PreCriticResult
) -> None:
    """pcc_01 (critical): outcome variable must NOT appear as a column in train_X.csv."""
    spec = getattr(ctx, "research_spec", None) or {}
    outcome = spec.get("outcome_variable", "")
    if not outcome:
        return

    train_x_path = os.path.join(output_dir, "train_X.csv")
    if not os.path.exists(train_x_path):
        return  # Missing file is caught by DataEngineer validation; not duplicated here

    try:
        with open(train_x_path, newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            headers = next(reader, [])
        if outcome in headers:
            result.failures.append(
                CheckFailure(
                    check_id="pcc_01",
                    severity="critical",
                    message=(
                        f"Outcome variable '{outcome}' found as a column in train_X.csv "
                        "— confirmed target leakage."
                    ),
                    target_agent="DataEngineer",
                )
            )
    except OSError:
        pass  # Can't read file — not a pre-critic error, pipeline will surface it


def _check_model_count(ctx: object, result: PreCriticResult) -> None:
    """pcc_02 (major): results.json must have at least 4 individual models."""
    results = getattr(ctx, "results_object", None) or {}
    all_models: dict = results.get("all_models") or {}
    # StackingEnsemble is not an individual model
    stacking_keys = {k for k in all_models if "stack" in k.lower()}
    individual_count = len(all_models) - len(stacking_keys)
    if individual_count < 4:
        result.failures.append(
            CheckFailure(
                check_id="pcc_02",
                severity="major",
                message=(
                    f"Only {individual_count} individual model(s) found in results.json "
                    "(minimum 4 required: LR, RF, XGBoost, ElasticNet, MLP)."
                ),
                target_agent="Analyst",
            )
        )


def _check_required_figures(output_dir: str, result: PreCriticResult) -> None:
    """pcc_03 (major): shap_summary.png and shap_importance.png must exist."""
    for fig in ("shap_summary.png", "shap_importance.png"):
        if not os.path.exists(os.path.join(output_dir, fig)):
            result.failures.append(
                CheckFailure(
                    check_id="pcc_03",
                    severity="major",
                    message=f"Required figure '{fig}' not found in output directory — SHAP may not have completed.",
                    target_agent="Analyst",
                )
            )


def _check_top_features_present(ctx: object, result: PreCriticResult) -> None:
    """pcc_04 (major): results.json.top_features must not be empty."""
    results = getattr(ctx, "results_object", None) or {}
    if not results.get("top_features"):
        result.failures.append(
            CheckFailure(
                check_id="pcc_04",
                severity="major",
                message="results.json.top_features is empty — SHAP feature importance analysis did not complete.",
                target_agent="Analyst",
            )
        )


def _check_subgroup_performance_present(ctx: object, result: PreCriticResult) -> None:
    """pcc_05 (major): results.json.subgroup_performance must not be empty."""
    results = getattr(ctx, "results_object", None) or {}
    if not results.get("subgroup_performance"):
        result.failures.append(
            CheckFailure(
                check_id="pcc_05",
                severity="major",
                message="results.json.subgroup_performance is empty — subgroup analysis did not run.",
                target_agent="Analyst",
            )
        )


def _check_data_report_validation_passed(ctx: object, result: PreCriticResult) -> None:
    """pcc_06 (critical): data_report.validation_passed must be True."""
    report = getattr(ctx, "data_report", None) or {}
    # If validation_passed is explicitly False (not just missing), flag it
    if report.get("validation_passed") is False:
        warnings_preview = str(report.get("warnings", []))[:200]
        result.failures.append(
            CheckFailure(
                check_id="pcc_06",
                severity="critical",
                message=(
                    f"data_report.validation_passed=False. "
                    f"Warnings: {warnings_preview}"
                ),
                target_agent="DataEngineer",
            )
        )
