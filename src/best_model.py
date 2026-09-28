"""Which "best model" results.json names, said in the file itself.

Observed on the owner's Mac (round 3, 2026-09-27): results.json named
XGBoost (AUC 0.80149) as ``best_model`` while StackingEnsemble scored
0.80202 in the same file. That is the design -- SPEC section 4.3 takes
every interpretability output from the best INDIVIDUAL model and excludes
the ensemble -- but nothing in the file said so. The paper read the field
as the overall winner: "XGBoost achieved the best discrimination (AUC =
0.801) ... followed closely by the Stacking Ensemble (AUC = 0.802)", and
the final checks flagged the contradiction (INV_SUPERLATIVE_CONTRADICTED).

After ANALYZING (and after an Analyst revision) the orchestrator adds,
from ``all_models`` and the primary metric's direction -- not from the
model's code --

* ``best_model_scope``: ``"individual"`` when ``best_model`` is not an
  ensemble (the design), ``"overall"`` when the analysis named one;
* ``best_overall_model`` / ``best_overall_metric_value``: the best of all
  models, ensemble included.

A ``best_model`` that is not first within its own scope is not rewritten
(its intervals and SHAP were computed for it); a warning says so.

Never raises: a bookkeeping step must not be what breaks a finished
analysis.
"""
from __future__ import annotations

import json
import math
import os
import re
import tempfile
from typing import Any, Callable, Optional

#: Model names that are combinations of the others.
ENSEMBLE_NAME = re.compile(r"stack|ensemble|voting|blend", re.IGNORECASE)

#: Metric keys where smaller is better.
LOWER_IS_BETTER = ("rmse", "mae", "mse", "brier", "log_loss", "logloss", "error")

_METRIC_ALIASES = {
    "auc-roc": "auc", "roc_auc": "auc", "auroc": "auc", "roc-auc": "auc",
    "r^2": "r2", "r-squared": "r2",
}


def metric_key(primary_metric: Any) -> str:
    """The ``all_models`` key for a primary metric name (``"AUC"`` -> ``"auc"``)."""
    key = str(primary_metric or "").strip().lower()
    return _METRIC_ALIASES.get(key, key)


def is_ensemble(name: str) -> bool:
    return bool(ENSEMBLE_NAME.search(str(name)))


def _score(row: Any, key: str) -> Optional[float]:
    if not isinstance(row, dict):
        return None
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _best(scores: list[tuple[str, float]], lower: bool,
          prefer: tuple[str, ...]) -> Optional[tuple[str, float]]:
    """The best entry; a tie goes to the first name in *prefer*, then to
    ``all_models`` order (the battery lists simpler models first)."""
    if not scores:
        return None
    target = (min if lower else max)(v for _, v in scores)
    tied = [(n, v) for n, v in scores if v == target]
    for name in prefer:
        for n, v in tied:
            if n == name:
                return n, v
    return tied[0]


def annotate(results: dict) -> tuple[dict, list[str]]:
    """Return *results* with the scope fields added, and warnings to add.

    Leaves *results* unchanged (and returns no warnings) when it has no
    per-model values for its primary metric.
    """
    all_models = results.get("all_models")
    key = metric_key(results.get("primary_metric"))
    if not isinstance(all_models, dict) or not key:
        return results, []
    scores = [
        (str(name), s) for name, row in all_models.items()
        if (s := _score(row, key)) is not None
    ]
    if not scores:
        return results, []
    lower = any(w in key for w in LOWER_IS_BETTER)
    claimed = results.get("best_model")
    claimed = claimed if isinstance(claimed, str) else ""
    individual = [(n, v) for n, v in scores if not is_ensemble(n)]

    best_individual = _best(individual, lower, prefer=(claimed,))
    # Ties between an individual model and an ensemble go to the
    # individual model: it is the simpler one and the one interpreted.
    overall = _best(
        scores, lower,
        prefer=((best_individual[0],) if best_individual else ()) + (claimed,),
    )
    scope = "overall" if claimed and is_ensemble(claimed) else "individual"

    out = dict(results)
    out["best_model_scope"] = scope
    out["best_overall_model"] = overall[0]
    out["best_overall_metric_value"] = overall[1]

    warnings: list[str] = []
    expected = overall if scope == "overall" else best_individual
    metric = str(results.get("primary_metric") or key)
    if claimed and expected is not None and claimed != expected[0]:
        mine = _score(all_models.get(claimed), key)
        where = "of all models" if scope == "overall" else "among the individual models"
        mine_txt = f" ({metric} {mine:.4g})" if mine is not None else ""
        warnings.append(
            f"results.best_model is {claimed}{mine_txt}, but all_models puts "
            f"{expected[0]} first {where} ({metric} {expected[1]:.4g}). "
            "best_model was not changed; its intervals and interpretation "
            "outputs were computed for it."
        )
    return out, warnings


def _write_results(output_dir: str, results: dict) -> None:
    path = os.path.join(output_dir, "results.json")
    fd, tmp = tempfile.mkstemp(prefix=".results.", dir=output_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            json.dump(results, out, indent=2)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def annotate_best_model(ctx: Any, log: Callable[[str], None]) -> None:
    """Add the scope fields to ``ctx.results_object`` and results.json."""
    try:
        results = getattr(ctx, "results_object", None)
        if not isinstance(results, dict):
            return
        updated, warnings = annotate(results)
        if updated is results:
            return
        if warnings:
            existing = results.get("warnings")
            existing = list(existing) if isinstance(existing, list) else (
                [str(existing)] if existing else []
            )
            updated["warnings"] = existing + [w for w in warnings if w not in existing]
            for w in warnings:
                log(w)
        ctx.results_object = updated
        output_dir = str(getattr(ctx, "output_dir", "") or "")
        if output_dir and os.path.exists(os.path.join(output_dir, "results.json")):
            _write_results(output_dir, updated)
    except Exception as exc:  # noqa: BLE001
        log(f"Best-model scope annotation skipped (non-fatal): {exc}")
