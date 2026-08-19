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


_SUPPORTED_TASK_TYPES: frozenset[str] = frozenset({"prediction", "causal_soo", "causal_itr", "causal_did", "psychometrics"})


def run_pre_critic_checks(
    ctx: object,
    output_dir: str,
    task_type: str = "prediction",
) -> PreCriticResult:
    """Run all deterministic pre-Critic checks and return a :class:`PreCriticResult`.

    Parameters
    ----------
    ctx:
        The ``PipelineContext`` object (typed as ``object`` to avoid circular import).
        Must expose ``.research_spec``, ``.results_object``, and ``.data_report`` attrs.
    output_dir:
        Absolute path to the run's output directory.
    task_type:
        Phase 3b.6 / 6.2 — gates which structural checks fire. Prediction-
        specific checks (SHAP figures, top_features, all_models, subgroup
        AUC) are skipped under ``task_type='causal_soo'`` because the
        causal pipeline does not produce those artifacts. Universal checks
        (target leakage in train_X, data_report.validation_passed) fire
        regardless of task type.

    Raises
    ------
    ValueError
        If ``task_type`` is not in :data:`_SUPPORTED_TASK_TYPES`. We fail
        loudly rather than silently running the prediction-shaped checks
        for unknown types — the prior implicit fall-through is exactly
        what produced 3b.5's F-PRECRITIC-PREDICTION pollution.
    """
    if task_type not in _SUPPORTED_TASK_TYPES:
        raise ValueError(
            f"run_pre_critic_checks: unknown task_type {task_type!r}. "
            f"Supported: {sorted(_SUPPORTED_TASK_TYPES)}. "
            f"Add a structural-check policy for the new task type rather "
            f"than falling through to prediction-shaped checks."
        )
    result = PreCriticResult()

    # Universal checks (run for every task type)
    _check_outcome_not_in_train_x(ctx, output_dir, result)
    _check_data_report_validation_passed(ctx, result)

    if task_type == "prediction":
        # Prediction-shaped structural checks: model battery + SHAP +
        # top_features + subgroup AUC. None of these apply to causal
        # output; they fire as false-positive [major] issues under
        # causal_soo (per 3b.5 evidence — F-PRECRITIC-PREDICTION).
        _check_model_count(ctx, result)
        _check_required_figures(output_dir, result)
        _check_top_features_present(ctx, result)
        _check_subgroup_performance_present(ctx, result)
    elif task_type == "psychometrics":
        # Measurement runs: no split/refuter/model-battery checks; the
        # psychometrics-measurement-protocol skill carries the critic
        # rows (psy_01..). No structural pre-checks yet.
        pass
    elif task_type == "causal_did":
        # DiD has no propensity model -> the DoWhy refuter contract
        # (pcc_c01) does not apply; the M8 helpers carry their own
        # placebo probe. No causal_did-specific structural checks yet.
        pass
    elif task_type in ("causal_soo", "causal_itr"):
        # 3b.6 deliberately ran NO causal-specific structural checks.
        # V4 Arc H (3b.23.7) adds the first one: the refuter-attempt
        # assertion (pcc_c01), after 3b.23.5 shipped a paper whose
        # sensitivity package silently omitted the mandatory DoWhy
        # refuters. The fuller causal pre-critic checklist
        # (F-CAUSAL-PRECRITIC: balance, positivity, estimand gates)
        # remains future work — the Critic carries that load via the
        # injected G1-G5 + D1 skills.
        _check_refuters_attempted(ctx, result)

    return result


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_refuters_attempted(ctx: object, result: PreCriticResult) -> None:
    """pcc_c01 (major, causal_soo only): DoWhy refuters must be ATTEMPTED.

    The causal-sensitivity-unmeasured-confounding skill makes refuter
    invocation unconditional: ``sensitivity.dowhy_refuters`` must exist
    with at least two refuter entries, each carrying a ``status`` field
    ("ran" or "failed"). Failure is acceptable when documented
    (fallback), silence is not (F-3b23.5 shipped a paper with the key
    absent entirely; 3b.19's healthy shape has per-refuter dicts with
    status="ran").
    """
    results = getattr(ctx, "results_object", None) or {}
    sensitivity = results.get("sensitivity") or {}
    refuters = (
        sensitivity.get("dowhy_refuters")
        if isinstance(sensitivity, dict)
        else None
    )

    def _fail(message: str) -> None:
        result.failures.append(
            CheckFailure(
                check_id="pcc_c01",
                severity="major",
                message=message,
                target_agent="Analyst",
            )
        )

    if not isinstance(refuters, dict) or not refuters:
        _fail(
            "sensitivity.dowhy_refuters is absent or empty — the DoWhy "
            "refuters were never attempted. Refuter invocation is "
            "unconditional for causal_soo (attempt-and-document; a "
            "documented failure is acceptable, silence is not). See "
            "causal-sensitivity-unmeasured-confounding §Refuter "
            "execution status contract."
        )
        return

    entries = {
        name: entry
        for name, entry in refuters.items()
        if isinstance(entry, dict)
    }
    if len(entries) < 2:
        _fail(
            f"sensitivity.dowhy_refuters has {len(entries)} refuter "
            f"entry(ies); the skill requires at least two refuters "
            f"attempted (e.g. random_common_cause + "
            f"placebo_treatment_refuter)."
        )
        return

    missing_status = [
        name
        for name, entry in entries.items()
        if entry.get("status") not in ("ran", "failed")
    ]
    if missing_status:
        _fail(
            f"Refuter entry(ies) {missing_status} lack a valid status "
            f"('ran' | 'failed') — attempts must be documented with "
            f"their outcome; a failed refuter records status='failed' "
            f"plus the error text."
        )


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
