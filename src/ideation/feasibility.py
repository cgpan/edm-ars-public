"""Arc T / T0 - the deterministic feasibility screen (Stage 0 + Stage 1).

Zero LLM calls, zero network. Everything here is computed from files on
disk: the dataset registries, the task templates, and (for Stage 1) the
Tier-1 column cache. The screen must keep working with every model in the
system switched off - that is the point of the slice.

THE KILL DISCIPLINE (spec sec. 3, risk R3)
--------------------------------------
``KILL`` is reserved for facts that are **logically dispositive**: the
column is not in the file, the estimator is not implemented, the task
type cannot execute on that dataset, a CFA factor has two items. Anything
probabilistic is a ``WARN`` carrying a penalty weight, and a WARN never
removes a candidate.

A false KILL deletes a legitimate research question with no human in the
loop and leaves no trace, so every check follows one rule:

    when a check cannot establish its fact with certainty - missing
    metadata, absent data file, ambiguous registry entry - it returns
    OK or WARN, never KILL.

Two consequences are visible in the code below and were forced by
measurement against the 26 archived specs (see
``scripts/audit_feasibility.py``):

* ``check_tier3_exclusion`` does not KILL a name that the registry
  *curates*. ``X1IEPFLAG`` matches the ``FLAG$`` tier-3 suffix pattern
  and is also a curated Tier-1 predictor used by 3 shipped specs;
  curation beats the pattern.
* ``check_structural_completeness`` KILLs only on dispatch-blocking
  absences, not on every warning ``validate_research_spec`` returns.
  13 shipped causal specs carry ``estimand`` rather than
  ``target_estimand_hint``; killing them would be a 50% false-kill rate.
* ``check_estimator_certified`` normalises the method aliases a
  COMPLETED run actually used (``IPW``/``AIPW``/``PSM``/
  ``regression_adjustment``) before testing membership.

C1: no novelty score is computed, stored, or ranked on anywhere in this
module. ``novelty_score_self_assessment`` is never read.
C2: every CheckResult carries an ``evidence`` string naming the artifact
fact it read.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

from src.ideation import probe_cache

# --------------------------------------------------------------------------
# Statuses and codes
# --------------------------------------------------------------------------

KILL = "KILL"
WARN = "WARN"
OK = "OK"

CLEAN = "CLEAN"

DEFAULT_REGISTRY_DIR = Path("data_registry") / "datasets"

# WARN penalty weights. These enter the tournament ranking as a prior
# offset (spec sec. 4.1); they never kill. Values are ordinal, not
# calibrated - the only claim is "a fabricated subgroup variable is worse
# than an uncurated one".
PENALTY = {
    "F-SUBGROUP-VAR-UNKNOWN": 1.0,
    "F-METADATA-UNVERIFIED": 0.5,
    "F-PITFALL-TOUCHED": 0.5,
    "F-VAR-UNVERIFIED": 0.75,
    "F-SPEC-INCOMPLETE": 0.5,
    "F-DESIGN-INFEASIBLE": 0.5,
    "P-ANALYTIC-N": 1.0,
    "P-CLASS-BALANCE": 0.5,
    "P-POSITIVITY": 1.5,
    "P-DID-CELLS": 1.0,
    "P-CDM-SCOPE": 0.5,
}


@dataclass(frozen=True)
class CheckResult:
    """One check outcome.

    ``evidence`` is mandatory (C2): it names the artifact fact the check
    read - a registry field, a CSV header, a template constant. A check
    that cannot say what it read does not ship.
    """

    code: str
    status: str  # KILL | WARN | OK
    message: str
    evidence: str
    penalty: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FeasibilityReport:
    candidate_id: str
    verdict: str  # KILL | WARN | CLEAN
    checks: list[CheckResult] = field(default_factory=list)
    analytic_n_estimate: int | None = None
    penalty: float = 0.0
    dataset: str | None = None
    task_type: str | None = None

    @property
    def kills(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == KILL]

    @property
    def warns(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == WARN]

    @property
    def kill_codes(self) -> list[str]:
        return [c.code for c in self.kills]

    @property
    def warn_codes(self) -> list[str]:
        return [c.code for c in self.warns]

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "verdict": self.verdict,
            "dataset": self.dataset,
            "task_type": self.task_type,
            "analytic_n_estimate": self.analytic_n_estimate,
            "penalty": round(self.penalty, 4),
            "checks": [c.to_dict() for c in self.checks],
        }

    def render(self) -> str:
        lines = [
            f"FeasibilityReport({self.candidate_id}) verdict={self.verdict} "
            f"dataset={self.dataset} task_type={self.task_type} "
            f"penalty={self.penalty:.2f} "
            f"analytic_n={self.analytic_n_estimate}"
        ]
        for check in self.checks:
            lines.append(f"  [{check.status:4}] {check.code}: {check.message}")
            lines.append(f"         evidence: {check.evidence}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# DATASET_TASK_MATRIX
# --------------------------------------------------------------------------
#
# Which of the 4 runnable datasets can execute which of the 5 task types.
# Nothing in the repo encoded this before, so
# `--dataset hsls09_public --task-type causal_did` was accepted and spent
# real money on a design the data cannot support.
#
# Derived from registry facts + task-template requirements, NOT from
# memory. The predicates, applied to data_registry/datasets/*.yaml:
#
#   prediction    registry declares >=1 outcome and >=1 predictor
#                 category. (All four pass.)
#   causal_soo    registry declares predictors, i.e. the covariate set
#                 selection-on-observables needs - the same predicate
#                 src/design_selector.py::soo_feasible applies.
#   causal_itr    design_feasibility.itr_ready is true. This is the
#                 registry's own explicit declaration, written by
#                 scripts/onboard_dataset.py and (before Arc T) never
#                 read by anything.
#   causal_did    design_feasibility.policy_timing_variables is
#                 non-empty. A DiD needs a post/timing dimension INSIDE
#                 the file; `multi_cohort_partner` alone names a
#                 *different* dataset and is not sufficient (spec sec. 1.4).
#   psychometrics registry declares item_banks or cdm_support, i.e.
#                 item-level responses exist. Measurement models cannot
#                 run on composites alone.
#
# Unsupported cells carry the dispositive reason, which becomes the
# check's evidence string.

_UNSUPPORTED_REASONS: dict[tuple[str, str], str] = {
    # HSLS:09 is a single 2009 cohort. design_feasibility declares
    # policy_timing_variables: [] - there is no post/timing variable in
    # the file. Its multi_cohort_partner (els_2002) is a different
    # dataset; the cross-cohort structure lives in the harmonized
    # did_els_hsls_panel, which is its own registry entry.
    ("hsls09_public", "causal_did"): (
        "hsls09_public.design_feasibility.policy_timing_variables is empty: "
        "single 2009 cohort, no in-file post/timing dimension. The DiD "
        "structure lives in the harmonized did_els_hsls_panel dataset "
        "(multi_cohort_partner names a partner file, it is not itself a "
        "post variable)."
    ),
    ("els_2002", "causal_did"): (
        "els_2002.design_feasibility.policy_timing_variables is empty: "
        "single 2002 cohort, no in-file post/timing dimension. Run the "
        "cross-cohort contrast on did_els_hsls_panel instead."
    ),
    ("assistments_0910", "causal_did"): (
        "assistments_0910.design_feasibility.policy_timing_variables is "
        "empty and multi_cohort_partner is null; the log covers a single "
        "school year (temporal_order: [single_year])."
    ),
    ("assistments_0910", "causal_itr"): (
        "assistments_0910.design_feasibility.itr_ready is false - the "
        "public log carries no pre-treatment student covariates (0 "
        "protected attributes, all predictors are within-interaction "
        "measures), so a decision-time rule has nothing to condition on."
    ),
    ("did_els_hsls_panel", "causal_itr"): (
        "did_els_hsls_panel.design_feasibility.itr_ready is false - the "
        "harmonized panel carries 7 design/covariate columns built for a "
        "2x2 gap contrast, not a decision-time covariate set."
    ),
    ("did_els_hsls_panel", "psychometrics"): (
        "did_els_hsls_panel declares neither item_banks nor cdm_support, "
        "and every variable in it is a composite or a within-cohort "
        "percentile rank. Measurement models need item-level responses; "
        "the harmonizer did not carry any."
    ),
}

_ALL_TASK_TYPES = (
    "prediction",
    "causal_soo",
    "causal_itr",
    "causal_did",
    "psychometrics",
)

_KNOWN_DATASETS = (
    "hsls09_public",
    "els_2002",
    "did_els_hsls_panel",
    "assistments_0910",
)

DATASET_TASK_MATRIX: dict[str, dict[str, bool]] = {
    dataset: {
        task: (dataset, task) not in _UNSUPPORTED_REASONS
        for task in _ALL_TASK_TYPES
    }
    for dataset in _KNOWN_DATASETS
}
# Resulting matrix (rows = dataset, cols = task type):
#                      prediction  causal_soo  causal_itr  causal_did  psychometrics
#   hsls09_public          y           y           y           n            y
#   els_2002               y           y           y           n            y
#   did_els_hsls_panel     y           y           n           y            n
#   assistments_0910       y           y           n           n            y
#
# The permissive cells are deliberate. `assistments_0910 x causal_soo`
# and `did_els_hsls_panel x prediction` are unusual but not logically
# impossible (a knowledge-tracing SOO contrast, a rank prediction), and
# KILL is reserved for the impossible. They are expected to lose on
# venue fit, not to be deleted here.


# --------------------------------------------------------------------------
# Certified estimator sets
# --------------------------------------------------------------------------
#
# The causal sets come from the task templates themselves
# (Template.SUPPORTED_METHODS) so they cannot drift. causal_did and
# psychometrics keep their method IDs inside the validate function rather
# than a class constant, so they are mirrored here with the source line
# named in the evidence string.
_DID_METHODS = frozenset({"M8", "M9", "M10"})  # task_template.py CausalDIDTemplate
_PSY_METHODS = frozenset({"P1", "P2", "P3", "P4", "P5", "P6", "P7"})  # PsychometricsTemplate

# Aliases seen in COMPLETED archived runs (e.g. the 3b.5 smoke test ran
# primary_method 'IPW' end to end). Normalised before the membership
# test so a legitimate spec is never killed over vocabulary.
_METHOD_ALIASES = {
    "regressionadjustment": "M1",
    "ra": "M1",
    "outcomeregression": "M1",
    "propensityscorematching": "M2",
    "psm": "M2",
    "matching": "M2",
    "ipw": "M3",
    "inverseprobabilityweighting": "M3",
    "aipw": "M4",
    "tmle": "M4",
    "doublyrobust": "M4",
    "causalforest": "M5",
    "cate": "M5",
    "policylearning": "M6",
    "policyvalue": "M7",
    "gapingaps": "M8",
    "did": "M8",
    "compositionaipw": "M9",
}

# Designs that are certified on synthetic DGPs but have no executable
# task type (docs/backlog.md: RD and IV are shelved).
_SHELVED_DESIGNS = {"rd", "rdd", "regressiondiscontinuity", "iv", "2sls", "iv2sls"}


# --------------------------------------------------------------------------
# Context
# --------------------------------------------------------------------------


@dataclass
class ScreenContext:
    """Everything the Stage-0 checks read, resolved once.

    Built by :func:`make_context`. Tests construct it directly (or via
    the helper) and call individual checks - every check is a pure
    function of this context.
    """

    spec: dict
    task_type: str
    dataset: str | None = None
    registry: dict = field(default_factory=dict)
    registry_path: str | None = None
    var_map: dict[str, dict] = field(default_factory=dict)
    item_bank_items: dict[str, str] = field(default_factory=dict)
    tier2_names: set[str] = field(default_factory=set)
    columns: set[str] | None = None  # None => column universe unknown
    columns_source: str = "unavailable"
    raw_data_dir: str | None = None
    cache_dir: str | None = None
    card: dict | None = None

    # --- derived helpers ------------------------------------------------
    @property
    def temporal_order(self) -> list[str]:
        return list(self.registry.get("temporal_order") or [])

    @property
    def sentinels(self) -> list[str]:
        raw = (self.registry.get("missingness") or {}).get(
            "sentinel_codes_or_labels"
        ) or []
        return [str(s).strip() for s in raw]

    def known_name(self, name: str) -> bool:
        """Is this name curated in the registry (Tier-1 or an item bank)?"""
        return name in self.var_map or name in self.item_bank_items

    def column_universe(self) -> set[str] | None:
        """Names that certainly exist in the data, or None if unknown.

        Falls back to the Tier-2 auto-profile when the raw CSV is absent:
        the ELS Tier-2 draft profiles all 4,012 columns, so it is a
        complete column universe even offline.
        """
        if self.columns is not None:
            return self.columns
        if self.tier2_names:
            return set(self.tier2_names) | set(self.var_map)
        return None


# Registry YAMLs are static during a screen but expensive to parse (the
# ELS Tier-2 draft profiles 4,012 variables, ~1.3 s). Memoised on
# (path, mtime, size) so an edited file still invalidates.
_YAML_MEMO: dict[tuple[str, int, int], dict] = {}


def _load_yaml_cached(path: Path) -> dict:
    try:
        stat = path.stat()
    except OSError:
        return {}
    key = (str(path), int(stat.st_mtime), int(stat.st_size))
    cached = _YAML_MEMO.get(key)
    if cached is not None:
        return cached
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        data = {}
    _YAML_MEMO[key] = data
    return data


def load_registry(
    dataset: str,
    registry_dir: str | os.PathLike[str] | None = None,
) -> tuple[dict, str | None]:
    """Load a dataset registry. Returns ``({}, None)`` if absent."""
    base = Path(registry_dir) if registry_dir is not None else DEFAULT_REGISTRY_DIR
    path = base / f"{dataset}.yaml"
    if not path.exists():
        return {}, None
    return _load_yaml_cached(path), str(path)


def build_var_map(registry: dict) -> dict[str, dict]:
    """Flat ``{variable_name: metadata}`` over outcomes + predictors.

    Mirrors ``problem_formulator._build_registry_var_map``; duplicated
    rather than imported because the feasibility screen must not import
    an agent module (spec sec. 1.4 moves that helper to ``src/registry.py``;
    this delegates to it once it lands).
    """
    try:  # prefer the shared home once the parallel fix lands
        from src.registry import build_var_map as _shared  # type: ignore

        return _shared(registry)
    except Exception:
        pass

    var_map: dict[str, dict] = {}
    variables = (registry or {}).get("variables") or {}
    for outcome in variables.get("outcomes") or []:
        if isinstance(outcome, dict) and "name" in outcome:
            var_map[str(outcome["name"])] = outcome
    predictors = variables.get("predictors") or {}
    if isinstance(predictors, dict):
        for var_list in predictors.values():
            for var in var_list or []:
                if isinstance(var, dict) and "name" in var:
                    var_map[str(var["name"])] = var
    return var_map


def _item_bank_items(registry: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for bank_name, bank in ((registry or {}).get("item_banks") or {}).items():
        if not isinstance(bank, dict):
            continue
        for item in bank.get("items") or []:
            out[str(item)] = str(bank_name)
    return out


def _tier2_names(
    registry: dict,
    registry_dir: str | os.PathLike[str] | None = None,
) -> set[str]:
    """Names from the dataset's auto-profiled Tier-2 file, if it exists.

    HSLS declares ``hsls09_tier2_auto_v2.yaml`` and the file is NOT on
    disk - that absence is exactly why `check_metadata_verified` exists,
    and why an unresolvable name there degrades to WARN.
    """
    cfg = (registry or {}).get("tier2_config") or {}
    filename = cfg.get("auto_generated_file")
    if not filename:
        return set()
    base = Path(registry_dir) if registry_dir is not None else DEFAULT_REGISTRY_DIR
    path = base / str(filename)
    if not path.exists():
        return set()
    data = _load_yaml_cached(path)
    names: set[str] = set()
    for group in (data.get("variables") or {}).values():
        for var in group or []:
            if isinstance(var, dict) and var.get("name"):
                names.add(str(var["name"]))
    return names


def make_context(
    spec: dict,
    *,
    dataset: str | None = None,
    task_type: str | None = None,
    registry: dict | None = None,
    registry_path: str | None = None,
    registry_dir: str | os.PathLike[str] | None = None,
    raw_data_dir: str | os.PathLike[str] | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    columns: Iterable[str] | None = None,
    card: dict | None = None,
    use_column_cache: bool = True,
) -> ScreenContext:
    """Resolve registry, column universe and task type for one spec.

    ``dataset`` falls back to ``spec['dataset']``; the 6 archived
    prediction specs carry neither ``dataset`` nor ``task_type``, so the
    caller (or the checkpoint) supplies them. ``task_type`` falls back to
    ``spec['task_type']`` then to ``"prediction"`` (config.yaml's
    ``pipeline.task_type`` default).
    """
    spec = spec if isinstance(spec, dict) else {}
    dataset = dataset or spec.get("dataset") or None
    task_type = task_type or spec.get("task_type") or "prediction"

    reg: dict = registry or {}
    reg_path = registry_path
    if not reg and dataset:
        reg, reg_path = load_registry(dataset, registry_dir)
    if reg_path is None and dataset:
        base = Path(registry_dir) if registry_dir is not None else DEFAULT_REGISTRY_DIR
        candidate = base / f"{dataset}.yaml"
        reg_path = str(candidate) if candidate.exists() else None

    resolved_columns: set[str] | None = None
    source = "unavailable"
    if columns is not None:
        resolved_columns = {str(c) for c in columns}
        source = "caller-supplied column list"
    elif dataset and use_column_cache:
        header = probe_cache.header_columns(
            dataset, raw_data_dir=raw_data_dir, cache_dir=cache_dir
        )
        if header is not None:
            resolved_columns = set(header)
            source = f"CSV header of {probe_cache.raw_data_path(dataset, raw_data_dir)}"

    return ScreenContext(
        spec=spec,
        task_type=str(task_type),
        dataset=dataset,
        registry=reg,
        registry_path=reg_path,
        var_map=build_var_map(reg),
        item_bank_items=_item_bank_items(reg),
        tier2_names=_tier2_names(reg, registry_dir),
        columns=resolved_columns,
        columns_source=source,
        raw_data_dir=str(raw_data_dir) if raw_data_dir is not None else None,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        card=card,
    )


# --------------------------------------------------------------------------
# Name extraction
# --------------------------------------------------------------------------

# role -> spec keys carrying variable names. Roles matter because the
# checks treat them differently (a DiD group variable is contemporaneous
# with the outcome by construction; a prediction predictor is not).
_LIST_KEYS: dict[str, tuple[str, ...]] = {
    "covariate": ("adjustment_set", "adjustment_covariates", "controls"),
    "rule_covariate": ("rule_covariates",),
    "item": ("item_columns",),
    "subgroup": (
        "subgroup_analyses",
        "grouping_vars",
        "heterogeneity_subgroups",
        "moderators",
    ),
}
_SCALAR_KEYS: dict[str, tuple[str, ...]] = {
    "outcome": ("outcome_variable", "placebo_outcome"),
    "group": ("group_variable",),
    "post": ("post_variable",),
    "cluster": ("cluster_variable",),
}


def _as_name(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, dict):
        for key in ("variable", "name"):
            inner = value.get(key)
            if isinstance(inner, str) and inner.strip():
                return inner.strip()
    return None


def spec_variable_names(spec: dict) -> list[tuple[str, str]]:
    """``[(name, role), ...]`` for every variable a spec names.

    Task-type agnostic: the four task-type families put their variables
    under different keys, and 20 of 26 archived specs carry no
    ``outcome_variable`` at all.
    """
    found: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def _add(value: object, role: str) -> None:
        name = _as_name(value)
        if name and (name, role) not in seen:
            seen.add((name, role))
            found.append((name, role))

    for role, keys in _SCALAR_KEYS.items():
        for key in keys:
            _add(spec.get(key), role)
    for role, keys in _LIST_KEYS.items():
        for key in keys:
            for item in spec.get(key) or []:
                _add(item, role)

    for pred in spec.get("predictor_set") or []:
        _add(pred, "predictor")

    treatment = spec.get("treatment")
    _add(treatment, "treatment")
    outcome = spec.get("outcome")
    _add(outcome, "outcome")

    return found


def _resolved_target(spec: dict) -> str | None:
    """Task-type-agnostic 'what is being studied' resolver (spec sec. 2.5)."""
    for candidate in (
        spec.get("outcome_variable"),
        _as_name(spec.get("outcome")),
        _as_name(spec.get("treatment")),
        spec.get("scale_name"),
    ):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _text_blob(spec: dict) -> str:
    parts = [
        str(spec.get("research_question") or ""),
        str(spec.get("expected_contribution") or ""),
        str(spec.get("target_population") or ""),
    ]
    memo = spec.get("design_memo")
    if isinstance(memo, dict):
        parts.append(str(memo.get("chosen_design") or ""))
        parts.append(str(memo.get("feasibility_evidence") or ""))
    return " ".join(parts).lower()


# --------------------------------------------------------------------------
# Stage 0 - deterministic hard filter
# --------------------------------------------------------------------------


def check_dataset_task_compatibility(ctx: ScreenContext) -> CheckResult:
    """KILL when the task type cannot execute on that dataset."""
    code = "F-TASK-INCOMPATIBLE"
    if not ctx.dataset:
        return CheckResult(
            code, WARN,
            "No dataset declared on the spec; dataset x task-type "
            "compatibility cannot be established.",
            "spec['dataset'] absent and no dataset argument supplied",
            PENALTY["F-METADATA-UNVERIFIED"],
        )
    row = DATASET_TASK_MATRIX.get(ctx.dataset)
    if row is None:
        return CheckResult(
            code, WARN,
            f"Dataset {ctx.dataset!r} is not in DATASET_TASK_MATRIX; "
            "compatibility unknown (a newly onboarded dataset must be "
            "added to the matrix).",
            f"DATASET_TASK_MATRIX has no row for {ctx.dataset!r}",
            PENALTY["F-METADATA-UNVERIFIED"],
        )
    if ctx.task_type not in row:
        return CheckResult(
            code, WARN,
            f"Task type {ctx.task_type!r} is not one of {list(_ALL_TASK_TYPES)}.",
            "DATASET_TASK_MATRIX column set",
            PENALTY["F-METADATA-UNVERIFIED"],
        )
    if row[ctx.task_type]:
        return CheckResult(
            code, OK,
            f"{ctx.dataset} supports {ctx.task_type}.",
            f"DATASET_TASK_MATRIX[{ctx.dataset!r}][{ctx.task_type!r}] is True",
        )
    reason = _UNSUPPORTED_REASONS[(ctx.dataset, ctx.task_type)]
    return CheckResult(
        code, KILL,
        f"{ctx.task_type} cannot execute on {ctx.dataset}.",
        reason,
    )


def check_variables_exist_in_registry(ctx: ScreenContext) -> CheckResult:
    """KILL for names that exist in neither the registry nor the data.

    Degradation rule: a name absent from the registry when no column
    universe is available is UNVERIFIED (WARN), not absent. The Tier-1
    registry is a curated subset, not a census - killing on it alone
    would delete every legitimate Tier-2 variable.
    """
    code = "F-VAR-ABSENT"
    names = spec_variable_names(ctx.spec)
    if not names:
        return CheckResult(
            code, OK, "Spec names no variables to resolve.",
            "no variable-bearing keys present in the spec",
        )
    if not ctx.registry:
        return CheckResult(
            code, WARN,
            "No registry loaded; variable existence unverified.",
            f"registry for dataset {ctx.dataset!r} could not be loaded",
            PENALTY["F-VAR-UNVERIFIED"],
        )

    universe = ctx.column_universe()
    constructed = _constructed_names(ctx.spec)
    unknown = [
        (n, role)
        for n, role in names
        if not ctx.known_name(n) and n not in ctx.tier2_names and n not in constructed
    ]
    if not unknown:
        return CheckResult(
            code, OK,
            f"All {len(names)} named variables resolve in the registry.",
            f"{ctx.dataset}.yaml variables/ + item_banks/ "
            f"({len(ctx.var_map)} curated names, "
            f"{len(ctx.item_bank_items)} item-bank items)",
        )

    if universe is None:
        listed = ", ".join(sorted({n for n, _ in unknown}))
        return CheckResult(
            code, WARN,
            f"Names absent from the curated registry and unverifiable "
            f"(no column universe available): {listed}.",
            f"{ctx.dataset}.yaml has no entry for these names; "
            f"neither the raw CSV header nor a Tier-2 profile is on disk",
            PENALTY["F-VAR-UNVERIFIED"],
        )

    fabricated = sorted({n for n, _ in unknown if n not in universe})
    if fabricated:
        return CheckResult(
            code, KILL,
            f"Variable(s) do not exist in this dataset: {', '.join(fabricated)}.",
            f"absent from {ctx.dataset}.yaml AND from the column universe "
            f"({ctx.columns_source if ctx.columns is not None else 'Tier-2 auto profile'}, "
            f"{len(universe)} columns)",
        )
    real_but_uncurated = sorted({n for n, _ in unknown})
    return CheckResult(
        code, WARN,
        f"Variable(s) exist in the data but are not curated in the "
        f"registry: {', '.join(real_but_uncurated)}.",
        f"present in the column universe, absent from {ctx.dataset}.yaml",
        PENALTY["F-METADATA-UNVERIFIED"],
    )


def _constructed_names(spec: dict) -> set[str]:
    """Names the pipeline derives rather than reads from the CSV.

    Operationalized treatments (``X1MTHEFF_binary``) and log-derived item
    matrices do not exist as columns; killing on them would be a false
    kill of exactly the kind R3 warns about.
    """
    names: set[str] = set()
    treatment = spec.get("treatment")
    if isinstance(treatment, dict):
        var = treatment.get("variable")
        if isinstance(var, str):
            for suffix in ("_binary", "_quartile", "_median", "_z"):
                names.add(var + suffix)
    if spec.get("item_construction"):
        for item in spec.get("item_columns") or []:
            if isinstance(item, str):
                names.add(item)
    return names


def check_columns_exist_in_csv(ctx: ScreenContext) -> CheckResult:
    """KILL when a named column is absent from the actual CSV header.

    Free insurance against registry drift: of 151 curated registry names
    across the four datasets, exactly one (``dropout_derived``, tagged
    ``derived: true``) is absent from its CSV - and derived names are
    skipped here.
    """
    code = "F-COL-ABSENT"
    if ctx.columns is None:
        return CheckResult(
            code, OK,
            "Skipped: raw CSV not available on this machine.",
            f"no readable raw data file for dataset {ctx.dataset!r} "
            f"({probe_cache.raw_data_path(ctx.dataset or '', ctx.raw_data_dir)})",
        )
    constructed = _constructed_names(ctx.spec)
    missing: list[str] = []
    for name, _role in spec_variable_names(ctx.spec):
        if name in ctx.columns or name in constructed:
            continue
        meta = ctx.var_map.get(name) or {}
        if meta.get("derived"):
            continue  # registry says the pipeline builds this one
        missing.append(name)
    if missing:
        return CheckResult(
            code, KILL,
            f"Column(s) not present in the raw data file: "
            f"{', '.join(sorted(set(missing)))}.",
            f"{ctx.columns_source} ({len(ctx.columns)} columns)",
        )
    return CheckResult(
        code, OK,
        "Every named column is present in the raw data file.",
        f"{ctx.columns_source} ({len(ctx.columns)} columns)",
    )


def check_temporal_order(ctx: ScreenContext) -> CheckResult:
    """KILL when a predictor does not strictly precede the outcome.

    Waves are resolved **from the registry**, not from the spec's own
    ``wave`` field: declaring ``X3TGPAMAT`` as ``wave: base_year``
    produces zero warnings under the pre-Arc-T rule.

    Not applicable, and therefore skipped with evidence, when:
      * the dataset declares a single wave (ASSISTments log data  - 
        ordering there is by timestamp, not by wave);
      * the task type is causal_did (group and post variables are
        contemporaneous with the outcome by construction: on the
        harmonized panel every column including ``rank_base`` sits in
        ``base_wave``, so a naive rule would kill both shipped DiD
        specs);
      * the task type is psychometrics (items are contemporaneous with
        the scale they measure).
    """
    code = "F-TEMPORAL-ORDER"
    order = ctx.temporal_order
    if len(order) < 2:
        return CheckResult(
            code, OK,
            "Skipped: dataset declares a single wave.",
            f"{ctx.dataset}.yaml temporal_order = {order}",
        )
    if ctx.task_type in ("causal_did", "psychometrics"):
        return CheckResult(
            code, OK,
            f"Skipped: wave ordering does not apply to {ctx.task_type}.",
            f"task_type {ctx.task_type} uses contemporaneous design/item "
            f"variables by construction",
        )

    target = _resolved_target(ctx.spec)
    if ctx.task_type.startswith("causal"):
        target = _as_name(ctx.spec.get("outcome")) or target
    outcome_meta = ctx.var_map.get(str(target) or "", {})
    outcome_wave = outcome_meta.get("wave")
    if outcome_wave not in order:
        return CheckResult(
            code, OK,
            f"Skipped: outcome wave for {target!r} is not resolvable "
            "from the registry.",
            f"{ctx.dataset}.yaml has no wave metadata for {target!r}",
        )
    outcome_idx = order.index(outcome_wave)

    roles = ("predictor", "treatment", "covariate", "rule_covariate")
    violations: list[str] = []
    checked = 0
    for name, role in spec_variable_names(ctx.spec):
        if role not in roles:
            continue
        wave = (ctx.var_map.get(name) or {}).get("wave")
        if wave not in order:
            continue
        checked += 1
        if order.index(wave) >= outcome_idx:
            violations.append(f"{name} (registry wave={wave}, role={role})")
    if violations:
        return CheckResult(
            code, KILL,
            f"Predictor(s) do not strictly precede outcome {target!r} "
            f"(wave={outcome_wave}): {', '.join(violations)}.",
            f"{ctx.dataset}.yaml per-variable 'wave' fields resolved "
            f"against temporal_order={order}",
        )
    return CheckResult(
        code, OK,
        f"All {checked} wave-resolvable predictors precede outcome "
        f"{target!r} (wave={outcome_wave}).",
        f"{ctx.dataset}.yaml per-variable 'wave' fields resolved against "
        f"temporal_order={order}",
    )


def check_tier3_exclusion(ctx: ScreenContext) -> CheckResult:
    """KILL when the spec uses a Tier-3 excluded name (weights, IDs, flags).

    Uses ``RegistryLoader.is_excluded``, which had zero production call
    sites before this. **Curation beats the pattern**: ``X1IEPFLAG``
    matches the ``FLAG$`` suffix rule and is also a curated Tier-1
    predictor used by 3 shipped specs, so a curated name is reported as
    OK-with-note rather than killed (and carries no penalty - the
    tension is a registry-rules defect, not a defect of the idea).
    """
    code = "F-TIER3-EXCLUDED"
    if not ctx.registry_path or not Path(ctx.registry_path).exists():
        return CheckResult(
            code, OK,
            "Skipped: no registry file path available for the Tier-3 rules.",
            f"registry_path={ctx.registry_path!r}",
        )
    try:
        from src.registry import RegistryLoader

        loader = RegistryLoader(ctx.registry_path)
    except Exception as exc:  # pragma: no cover - defensive
        return CheckResult(
            code, WARN,
            f"Tier-3 rules could not be loaded ({type(exc).__name__}).",
            f"RegistryLoader({ctx.registry_path!r}) raised",
            PENALTY["F-METADATA-UNVERIFIED"],
        )

    excluded: list[str] = []
    curated_hits: list[str] = []
    for name, _role in spec_variable_names(ctx.spec):
        if not loader.is_excluded(name):
            continue
        if ctx.known_name(name):
            curated_hits.append(name)
        else:
            excluded.append(name)
    rules = ctx.registry.get("tier3_exclusion_rules") or {}
    evidence = (
        f"{ctx.dataset}.yaml tier3_exclusion_rules "
        f"(prefix={rules.get('prefix_patterns')}, "
        f"suffix={rules.get('suffix_patterns')}, "
        f"{len(rules.get('exact_matches') or [])} exact matches)"
    )
    if excluded:
        return CheckResult(
            code, KILL,
            f"Tier-3 excluded name(s) used as study variables: "
            f"{', '.join(sorted(set(excluded)))}. These are weights, "
            f"sampling/administrative IDs, or processing flags.",
            evidence,
        )
    if curated_hits:
        return CheckResult(
            code, OK,
            f"Name(s) match a Tier-3 pattern but are curated Tier-1 "
            f"variables, so the curation wins: "
            f"{', '.join(sorted(set(curated_hits)))}.",
            evidence + " vs the curated variables/ section",
        )
    return CheckResult(
        code, OK, "No Tier-3 excluded names used.", evidence,
    )


def check_dead_variables(ctx: ScreenContext) -> CheckResult:
    """KILL when a named variable is suppressed/empty (pct_missing >= 99)."""
    code = "F-DEAD-VARIABLE"
    if not ctx.var_map:
        return CheckResult(
            code, OK, "Skipped: no registry metadata to read.",
            f"no var_map for dataset {ctx.dataset!r}",
        )
    dead: list[str] = []
    for name, _role in spec_variable_names(ctx.spec):
        meta = ctx.var_map.get(name)
        if not meta:
            continue
        pct = meta.get("pct_missing")
        if isinstance(pct, (int, float)) and pct >= 99:
            dead.append(f"{name} ({pct}% missing)")
    if dead:
        return CheckResult(
            code, KILL,
            f"Variable(s) carry no usable data: {', '.join(sorted(dead))}.",
            f"{ctx.dataset}.yaml pct_missing >= 99 (public-use suppression)",
        )
    return CheckResult(
        code, OK, "No suppressed/empty variables used.",
        f"{ctx.dataset}.yaml pct_missing fields",
    )


def _normalise_method(method: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(method).lower())


def _certified_methods(task_type: str) -> tuple[frozenset[str], str]:
    """(certified IDs, evidence source) for a task type."""
    if task_type in ("causal_soo", "causal_itr"):
        from src.task_template import create_task_template

        template = create_task_template(task_type)
        return (
            frozenset(getattr(template, "SUPPORTED_METHODS", frozenset())),
            f"{type(template).__name__}.SUPPORTED_METHODS",
        )
    if task_type == "causal_did":
        return _DID_METHODS, "src/task_template.py CausalDIDTemplate (M8/M9/M10)"
    if task_type == "psychometrics":
        return _PSY_METHODS, "src/task_template.py PsychometricsTemplate (P1-P7)"
    return frozenset(), "no estimator IDs apply to this task type"


def check_estimator_certified(ctx: ScreenContext) -> CheckResult:
    """KILL when a requested estimator is not implemented.

    RD and IV are synthetic-gate certified but shelved (no executable
    task type), so any RD/IV identification claim is infeasible *today*
    and that is checkable in under a millisecond.
    """
    code = "F-ESTIMATOR-UNCERTIFIED"
    requested: list[str] = []
    for key in ("primary_method", "comparator_method"):
        value = ctx.spec.get(key)
        if isinstance(value, str) and value.strip():
            requested.append(value.strip())
    for key in ("secondary_methods", "method_battery", "methods"):
        for value in ctx.spec.get(key) or []:
            if isinstance(value, str) and value.strip():
                requested.append(value.strip())
    if not requested:
        return CheckResult(
            code, OK, "Spec requests no explicit estimator IDs.",
            "no primary_method / secondary_methods / method_battery in the spec",
        )

    certified, source = _certified_methods(ctx.task_type)
    if not certified:
        return CheckResult(
            code, OK,
            f"No certified-method list applies to task type "
            f"{ctx.task_type!r}.",
            source,
        )

    shelved: list[str] = []
    uncertified: list[str] = []
    for method in requested:
        norm = _normalise_method(method)
        if norm in _SHELVED_DESIGNS:
            shelved.append(method)
            continue
        resolved = method.upper() if method.upper() in certified else _METHOD_ALIASES.get(norm)
        if resolved is None or resolved not in certified:
            uncertified.append(method)
    if shelved:
        return CheckResult(
            code, KILL,
            f"Estimator(s) {', '.join(shelved)} are certified on synthetic "
            f"DGPs but shelved: no executable task type implements them.",
            "docs/backlog.md - RD and IV are shelved; "
            + source,
        )
    if uncertified:
        return CheckResult(
            code, KILL,
            f"Estimator(s) not in the certified battery for "
            f"{ctx.task_type}: {', '.join(uncertified)}.",
            f"{source} = {sorted(certified)}",
        )
    return CheckResult(
        code, OK,
        f"All {len(requested)} requested estimators are certified.",
        f"{source} = {sorted(certified)}",
    )


def check_design_feasible(ctx: ScreenContext) -> CheckResult:
    """KILL when ``select_design`` says the identification claim is infeasible.

    Delegates to ``src.design_selector`` - the deterministic layer that
    already owns design feasibility. Any exception degrades to WARN.
    """
    code = "F-DESIGN-INFEASIBLE"
    design_for_task = {
        "causal_soo": "causal_soo",
        "causal_itr": "causal_itr",
        "causal_did": "did",
    }
    design = design_for_task.get(ctx.task_type)
    if design is None:
        return CheckResult(
            code, OK,
            f"Skipped: {ctx.task_type} makes no quasi-experimental "
            "identification claim.",
            "src/design_selector.py covers causal designs only",
        )
    # The selector reads two registry structures. If NEITHER is present
    # its "infeasible" verdict reports missing metadata, not a documented
    # negative - and missing metadata must never kill.
    has_design_block = bool((ctx.registry or {}).get("design_feasibility"))
    has_predictors = bool(
        ((ctx.registry or {}).get("variables") or {}).get("predictors")
    )
    if not (has_design_block or has_predictors):
        return CheckResult(
            code, WARN,
            "Design feasibility unverified: the registry carries neither a "
            "design_feasibility block nor a predictor section.",
            f"registry for {ctx.dataset!r} lacks design_feasibility and "
            f"variables.predictors - select_design would be reading absence, "
            f"not evidence",
            PENALTY["F-DESIGN-INFEASIBLE"],
        )
    try:
        from src.design_selector import select_design

        report = select_design(
            ctx.registry, question=ctx.spec.get("research_question")
        )
        verdict = (report.get("verdicts") or {}).get(design) or {}
    except Exception as exc:
        return CheckResult(
            code, WARN,
            f"select_design raised {type(exc).__name__}; design "
            "feasibility unverified.",
            f"src.design_selector.select_design({ctx.dataset!r}) raised: {exc}",
            PENALTY["F-DESIGN-INFEASIBLE"],
        )
    if not verdict:
        return CheckResult(
            code, WARN,
            f"select_design returned no verdict for design {design!r}.",
            "src.design_selector.select_design verdicts block",
            PENALTY["F-DESIGN-INFEASIBLE"],
        )
    reasons = "; ".join(verdict.get("reasons") or [])
    if not verdict.get("feasible"):
        return CheckResult(
            code, KILL,
            f"Design {design!r} is infeasible on {ctx.dataset}.",
            f"src.design_selector.select_design -> verdicts[{design!r}]: {reasons}",
        )
    executable = verdict.get("executable_task_type")
    if executable is not None and executable != ctx.task_type:
        return CheckResult(
            code, KILL,
            f"Design {design!r} is feasible but executes as "
            f"{executable!r}, not {ctx.task_type!r}.",
            f"src.design_selector.select_design -> "
            f"verdicts[{design!r}].executable_task_type={executable!r}",
        )
    if executable is None:
        return CheckResult(
            code, KILL,
            f"Design {design!r} is feasible in principle but has no "
            f"executable task type in this codebase.",
            f"src.design_selector.select_design -> "
            f"verdicts[{design!r}].executable_task_type is None ({reasons})",
        )
    return CheckResult(
        code, OK,
        f"Design {design!r} is feasible and executable as {executable!r}.",
        f"src.design_selector.select_design -> verdicts[{design!r}]: {reasons}",
    )


# Fields whose absence blocks dispatch outright, per task type. These are
# the only structural absences that KILL; every other warning
# validate_research_spec returns is reported as a WARN, because 13 of the
# 26 archived specs carry `estimand` instead of `target_estimand_hint`
# and killing them would be a 50% false-kill rate.
_DISPATCH_REQUIRED: dict[str, tuple[tuple[str, ...], ...]] = {
    "prediction": (("outcome_variable",), ("predictor_set",)),
    "causal_soo": (("treatment",), ("outcome",), ("primary_method",)),
    "causal_itr": (
        ("treatment",),
        ("outcome",),
        ("primary_method",),
        ("rule_covariates",),
    ),
    "causal_did": (
        ("outcome",),
        ("group_variable",),
        ("post_variable",),
        ("primary_method",),
    ),
    "psychometrics": (
        ("scale_name",),
        ("item_columns", "item_construction"),
        ("method_battery",),
    ),
}


def check_structural_completeness(ctx: ScreenContext) -> CheckResult:
    """KILL when a dispatch-blocking field is absent; WARN otherwise."""
    code = "F-SPEC-INCOMPLETE"
    required = _DISPATCH_REQUIRED.get(ctx.task_type, ())
    missing: list[str] = []
    for alternatives in required:
        if not any(ctx.spec.get(key) for key in alternatives):
            missing.append(" or ".join(alternatives))
    if missing:
        return CheckResult(
            code, KILL,
            f"Spec cannot be dispatched as {ctx.task_type}: missing "
            f"{', '.join(missing)}.",
            f"required-to-dispatch fields for {ctx.task_type} "
            f"(src/task_template.py {ctx.task_type} template + "
            f"src/orchestrator.py stage runners)",
        )

    warnings: list[str] = []
    try:
        from src.dataset_adapter import create_dataset_adapter
        from src.task_template import create_task_template

        template = create_task_template(ctx.task_type)
        try:
            adapter = create_dataset_adapter(ctx.dataset) if ctx.dataset else None
        except Exception:
            adapter = None  # unknown dataset: the registry still carries
            # temporal_order and levels, which is all the template needs
        # adapter may be None: the causal/psychometrics templates accept
        # it, and the prediction template only touches it when the
        # registry itself lacks temporal_order / levels.
        warnings = list(
            template.validate_research_spec(
                ctx.spec, ctx.registry, adapter  # type: ignore[arg-type]
            )
        )
    except Exception as exc:
        return CheckResult(
            code, WARN,
            f"validate_research_spec raised {type(exc).__name__}; only the "
            "dispatch-required fields were verified.",
            f"src.task_template.{ctx.task_type}.validate_research_spec "
            f"raised: {exc}",
            PENALTY["F-SPEC-INCOMPLETE"],
        )

    # These are handled by dedicated checks (or banned by C1) and must
    # not be double-counted here.
    ignore = ("novelty_score", "Estimated analytic_n", "TEMPORAL VIOLATION")
    residual = [w for w in warnings if not any(tok in w for tok in ignore)]
    if residual:
        return CheckResult(
            code, WARN,
            f"Template validation warnings (non-blocking): "
            f"{'; '.join(residual)}",
            f"src.task_template.{ctx.task_type}.validate_research_spec "
            f"returned {len(warnings)} warning(s)",
            PENALTY["F-SPEC-INCOMPLETE"] * len(residual),
        )
    return CheckResult(
        code, OK,
        f"Spec carries every field required to dispatch as {ctx.task_type}.",
        f"dispatch-required fields present; "
        f"src.task_template.{ctx.task_type}.validate_research_spec "
        f"returned {len(warnings)} warning(s)",
    )


_EQUITY_KEYWORDS = (
    "equity", "fairness", "disparit", "achievement gap", "bias across",
    "differential prediction", "subgroup gap",
)


def check_protected_attributes(ctx: ScreenContext) -> CheckResult:
    """KILL an equity question on a dataset with no protected attributes.

    Only an *explicit* equity claim kills: named subgroup variables, or a
    card carrying ``opportunity_pattern: equity_subgroup_gap``. A
    keyword-only signal is probabilistic, so it downgrades to WARN.
    """
    code = "F-NO-PROTECTED-ATTRS"
    protected = sorted(
        name
        for name, meta in ctx.var_map.items()
        if isinstance(meta, dict) and meta.get("protected_attribute")
    )
    named_subgroups = [
        n for n, role in spec_variable_names(ctx.spec) if role == "subgroup"
    ]
    pattern = ((ctx.card or {}).get("cell") or {}).get("opportunity_pattern")
    explicit = bool(named_subgroups) or pattern == "equity_subgroup_gap"
    keyword_only = any(k in _text_blob(ctx.spec) for k in _EQUITY_KEYWORDS)

    if not (explicit or keyword_only):
        return CheckResult(
            code, OK, "Spec makes no equity/subgroup claim.",
            "no subgroup variables named, no equity opportunity pattern",
        )
    if protected:
        return CheckResult(
            code, OK,
            f"Dataset carries {len(protected)} protected attribute(s): "
            f"{', '.join(protected)}.",
            f"{ctx.dataset}.yaml protected_attribute: true",
        )
    # An UNLOADABLE registry is not evidence of absence. When the dataset
    # cannot be resolved, var_map is empty and every attribute looks
    # missing -- which once killed 6 of 26 real archived specs whose
    # `dataset` field is absent (evidence read literally "None.yaml
    # declares no variable with protected_attribute: true"). KILL is
    # reserved for facts we established, never for facts we could not
    # look up.
    registry_unavailable = not ctx.dataset or not ctx.var_map
    if explicit and registry_unavailable:
        return CheckResult(
            code, WARN,
            "Equity/subgroup question, but the dataset registry could not "
            f"be resolved ({ctx.dataset or 'no dataset field'}), so the "
            "protected attributes are UNKNOWN, not absent.",
            f"dataset={ctx.dataset!r}, var_map entries={len(ctx.var_map)}; "
            "cannot establish absence",
            PENALTY["F-SUBGROUP-VAR-UNKNOWN"],
        )
    if explicit:
        return CheckResult(
            code, KILL,
            "Equity/subgroup question on a dataset with zero protected "
            f"attributes (named subgroups: {named_subgroups or pattern}).",
            f"{ctx.dataset}.yaml declares no variable with "
            f"protected_attribute: true",
        )
    return CheckResult(
        code, WARN,
        "Text suggests an equity framing but the dataset carries no "
        "protected attributes; the fairness claim cannot be supported.",
        f"{ctx.dataset}.yaml declares no protected_attribute; equity "
        "keyword matched in the research question / contribution text",
        PENALTY["F-SUBGROUP-VAR-UNKNOWN"],
    )


_FACTOR_LINE = re.compile(r"^\s*(\w+)\s*=~\s*(.+)$")
_CFA_METHODS = {"P3", "P6"}  # CFA and measurement invariance both need a factor


def _factor_items(factor_model: object) -> dict[str, list[str]]:
    factors: dict[str, list[str]] = {}
    if not isinstance(factor_model, str):
        return factors
    for line in factor_model.splitlines():
        match = _FACTOR_LINE.match(line)
        if not match:
            continue
        name, rhs = match.group(1), match.group(2)
        items = [tok.strip() for tok in rhs.split("+") if tok.strip()]
        factors[name] = items
    return factors


def check_item_bank_adequacy(ctx: ScreenContext) -> CheckResult:
    """KILL a CFA/invariance model with a factor carrying fewer than 3 items."""
    code = "F-ITEM-BANK-TOO-FEW"
    if ctx.task_type != "psychometrics":
        return CheckResult(
            code, OK, "Skipped: not a measurement study.",
            f"task_type={ctx.task_type}",
        )
    battery = {str(m).upper() for m in (ctx.spec.get("method_battery") or [])}
    needs_factor = bool(battery & _CFA_METHODS)
    factors = _factor_items(ctx.spec.get("factor_model"))
    items = [i for i in (ctx.spec.get("item_columns") or []) if isinstance(i, str)]

    if ctx.spec.get("item_construction") and not items:
        return CheckResult(
            code, OK,
            "Skipped: items are constructed from log data, not read from "
            "fixed item columns.",
            "spec.item_construction is set (log-data measurement study)",
        )
    if not needs_factor:
        return CheckResult(
            code, OK,
            "Skipped: battery requests no factor model "
            f"({sorted(battery) or 'empty battery'}).",
            f"method_battery does not include {sorted(_CFA_METHODS)}",
        )

    thin: list[str] = []
    if factors:
        thin = [f"{name} ({len(its)} items)" for name, its in factors.items() if len(its) < 3]
        evidence = f"spec.factor_model: {len(factors)} factor(s) parsed"
    else:
        thin = [f"single factor ({len(items)} items)"] if len(items) < 3 else []
        evidence = f"spec.item_columns: {len(items)} item(s), no factor_model given"

    bank_note = ""
    for bank_name, bank in ((ctx.registry.get("item_banks")) or {}).items():
        if not isinstance(bank, dict):
            continue
        bank_items = [i for i in bank.get("items") or [] if i in set(items)]
        if bank_items and len(bank.get("items") or []) < 3:
            bank_note = (
                f"; registry item bank {bank_name!r} declares "
                f"{len(bank.get('items') or [])} items"
            )
    if thin:
        return CheckResult(
            code, KILL,
            f"Factor(s) with fewer than 3 items requested under "
            f"{sorted(battery & _CFA_METHODS)}: {', '.join(thin)}.",
            evidence + bank_note,
        )
    return CheckResult(
        code, OK,
        "Every requested factor carries at least 3 items.",
        evidence + bank_note,
    )


def check_subgroup_variables(ctx: ScreenContext) -> CheckResult:
    """WARN when a named subgroup variable is not real."""
    code = "F-SUBGROUP-VAR-UNKNOWN"
    subgroups = [n for n, role in spec_variable_names(ctx.spec) if role == "subgroup"]
    if not subgroups:
        return CheckResult(
            code, OK, "Spec names no subgroup variables.",
            "no subgroup_analyses / grouping_vars / heterogeneity_subgroups",
        )
    universe = ctx.column_universe()
    unknown = [
        n for n in subgroups
        if not ctx.known_name(n)
        and n not in ctx.tier2_names
        and (universe is None or n not in universe)
    ]
    if unknown:
        return CheckResult(
            code, WARN,
            f"Subgroup variable(s) not found in registry or data: "
            f"{', '.join(sorted(set(unknown)))}.",
            f"{ctx.dataset}.yaml variables/ + "
            f"{ctx.columns_source if ctx.columns is not None else 'no column universe'}",
            PENALTY[code],
        )
    return CheckResult(
        code, OK,
        f"All {len(subgroups)} subgroup variable(s) resolve.",
        f"{ctx.dataset}.yaml variables/ (+ column universe)",
    )


def check_metadata_verified(ctx: ScreenContext) -> CheckResult:
    """WARN when the spec leans on uncurated / unverified metadata.

    Two known metadata defects make this necessary: the ELS Tier-2
    profile reports ``pct_missing: 0.00`` for all 4,012 variables because
    the profiler is not sentinel-aware, and the Tier-2 file HSLS's
    registry points at does not exist on disk.
    """
    code = "F-METADATA-UNVERIFIED"
    names = [n for n, _ in spec_variable_names(ctx.spec)]
    uncurated = sorted(
        {
            n for n in names
            if not ctx.known_name(n) and n not in _constructed_names(ctx.spec)
        }
    )
    cfg = (ctx.registry.get("tier2_config") or {})
    declared_file = cfg.get("auto_generated_file")
    tier2_missing = bool(declared_file) and not ctx.tier2_names

    if not uncurated:
        return CheckResult(
            code, OK,
            f"All {len(names)} named variables carry curated Tier-1 metadata.",
            f"{ctx.dataset}.yaml variables/ + item_banks/",
        )
    detail = f"uncurated names: {', '.join(uncurated)}"
    if tier2_missing:
        detail += (
            f"; the declared Tier-2 file {declared_file!r} is not on disk"
        )
    elif ctx.tier2_names:
        detail += (
            f"; Tier-2 auto profile ({len(ctx.tier2_names)} vars) is "
            "not sentinel-aware, so its pct_missing cannot be trusted"
        )
    return CheckResult(
        code, WARN,
        f"{len(uncurated)} variable(s) lack curated metadata; missingness "
        "and wave for them are unverified.",
        f"{ctx.dataset}.yaml tier2_config + variables/ - {detail}",
        PENALTY[code] * min(len(uncurated), 4),
    )


def check_common_pitfalls(ctx: ScreenContext) -> CheckResult:
    """WARN when the spec touches a registry-documented pitfall.

    Only pitfalls with a deterministic predicate fire; the rest of the
    registry's pitfall list is LLM-instruction-only and stays that way
    rather than producing a guess.
    """
    code = "F-PITFALL-TOUCHED"
    pitfalls = {
        str(p.get("id")): p
        for p in (ctx.registry.get("common_pitfalls") or [])
        if isinstance(p, dict)
    }
    if not pitfalls:
        return CheckResult(
            code, OK, "Registry documents no pitfalls for this dataset.",
            f"{ctx.dataset}.yaml common_pitfalls is empty/absent",
        )

    fired: list[str] = []

    # protected_attribute_misuse - a protected attribute is used as a
    # study variable but no subgroup analysis is declared.
    if "protected_attribute_misuse" in pitfalls:
        used_protected = [
            n for n, role in spec_variable_names(ctx.spec)
            if role in ("predictor", "covariate", "rule_covariate", "treatment")
            and (ctx.var_map.get(n) or {}).get("protected_attribute")
        ]
        declared = [n for n, role in spec_variable_names(ctx.spec) if role == "subgroup"]
        if used_protected and not declared:
            fired.append(
                f"protected_attribute_misuse: uses "
                f"{', '.join(sorted(set(used_protected)))} without declaring "
                f"any subgroup/fairness analysis"
            )

    # school_level_misinterpretation - multilevel claim on an extract
    # whose school IDs are suppressed.
    if "school_level_misinterpretation" in pitfalls:
        blob = _text_blob(ctx.spec)
        if any(k in blob for k in ("multilevel", "random effect", "school-level model")):
            fired.append(
                "school_level_misinterpretation: the spec claims a "
                "multilevel model, but the public-use extract suppresses "
                "school identifiers"
            )

    # public_use_suppression - a used variable whose only codebook codes
    # are suppression codes.
    if "public_use_suppression" in pitfalls:
        suppressed = []
        for name, _role in spec_variable_names(ctx.spec):
            codes = (ctx.var_map.get(name) or {}).get("codebook_codes") or {}
            labels = {str(v).lower() for v in codes.values()}
            if labels and labels <= {"data suppressed"}:
                suppressed.append(name)
        if suppressed:
            fired.append(
                f"public_use_suppression: {', '.join(sorted(set(suppressed)))} "
                "carry only suppression codes"
            )

    # non_equated_tests / single_policy_attribution (DiD panel) - the
    # spec must not claim absolute achievement change or a named policy.
    if "non_equated_tests" in pitfalls:
        blob = _text_blob(ctx.spec)
        if "achievement increase" in blob or "test scores rose" in blob:
            fired.append(
                "non_equated_tests: absolute achievement-change language on "
                "a panel where only rank-gap changes are estimable"
            )

    if fired:
        return CheckResult(
            code, WARN,
            "Registry-documented pitfall(s) touched: " + "; ".join(fired),
            f"{ctx.dataset}.yaml common_pitfalls "
            f"({', '.join(sorted(pitfalls))})",
            PENALTY[code] * len(fired),
        )
    return CheckResult(
        code, OK,
        f"No deterministic pitfall predicate fired "
        f"({len(pitfalls)} documented).",
        f"{ctx.dataset}.yaml common_pitfalls",
    )


STAGE0_CHECKS = (
    check_dataset_task_compatibility,
    check_variables_exist_in_registry,
    check_columns_exist_in_csv,
    check_temporal_order,
    check_tier3_exclusion,
    check_dead_variables,
    check_estimator_certified,
    check_design_feasible,
    check_structural_completeness,
    check_protected_attributes,
    check_item_bank_adequacy,
    check_subgroup_variables,
    check_metadata_verified,
    check_common_pitfalls,
)


# --------------------------------------------------------------------------
# Stage 1 - deterministic data probes
# --------------------------------------------------------------------------
#
# Probes never KILL. They read real data, but every operationalization
# they apply (median splits, sentinel rules, group construction) is an
# approximation of what the DataEngineer will actually build, so their
# findings are WARN + penalty by construction.

def _skip(code: str, ctx: ScreenContext, why: str) -> CheckResult:
    """A probe that could not run. Always OK - never a kill, never a warn."""
    return CheckResult(
        code, OK, f"Skipped: {why}.", f"{why} (dataset={ctx.dataset!r})",
    )


def _load_frame(ctx: ScreenContext, columns: list[str]) -> Any:
    if not ctx.dataset or not ctx.registry:
        return None
    return probe_cache.tier1_frame(
        ctx.dataset,
        ctx.registry,
        columns=columns or None,
        raw_data_dir=ctx.raw_data_dir,
        cache_dir=ctx.cache_dir,
    )


def _missing_mask(series: Any, meta: dict, sentinels: list[str]) -> Any:
    """Sentinel-aware missingness for one column.

    Uses the registry ``range`` for continuous variables (the naive
    "any negative is missing" rule understates n by 38% on HSLS, because
    several composites have valid negative values) and the registry's
    sentinel labels/codes otherwise.
    """
    import pandas as pd

    mask = series.isna()
    rng = meta.get("range")
    if (
        meta.get("type") == "continuous"
        and isinstance(rng, list)
        and len(rng) == 2
        and all(isinstance(x, (int, float)) for x in rng)
    ):
        numeric = pd.to_numeric(series, errors="coerce")
        mask = mask | numeric.isna() | (numeric < rng[0]) | (numeric > rng[1])
        return mask
    if sentinels:
        as_text = series.astype(str).str.strip()
        mask = mask | as_text.isin(sentinels)
    return mask


def _outcome_variable(ctx: ScreenContext) -> str | None:
    """The variable whose missingness sets the analytic sample.

    Covariates are imputed (SPEC sec. 4.2 missing-data protocol); the
    outcome never is ("NEVER impute the outcome variable. Drop rows with
    missing outcomes."). So the outcome alone determines analytic_n.
    """
    if ctx.task_type == "psychometrics":
        return None  # no single outcome column; items define the sample
    for candidate in (
        ctx.spec.get("outcome_variable"),
        _as_name(ctx.spec.get("outcome")),
    ):
        if isinstance(candidate, str) and candidate in ctx.var_map:
            return candidate
    return None


def estimate_analytic_n(ctx: ScreenContext) -> CheckResult:
    """Measured analytic n, sentinel-aware, from the Tier-1 cache.

    Reports two numbers, because they answer different questions and
    confusing them is how the pre-Arc-T rule produced a false abort
    warning:

    * **outcome-complete n** - rows with a usable outcome. This is the
      analytic sample the pipeline actually builds, because covariates
      are imputed and outcomes never are. Validated against the realized
      ``data_report.json`` of the archived runs (HSLS X4EVRATNDCLG:
      probe 17,335 vs realized 17,335).
    * **listwise-complete n** - rows complete on every named variable.
      A lower bound, reported for context only; the WARN thresholds are
      applied to the outcome-complete number.
    """
    code = "P-ANALYTIC-N"
    names = sorted({n for n, _ in spec_variable_names(ctx.spec)})
    resolvable = [n for n in names if n in ctx.var_map]
    if not resolvable:
        return _skip(code, ctx, "no registry-resolvable variables to probe")
    frame = _load_frame(ctx, resolvable)
    if frame is None or frame.empty:
        return _skip(code, ctx, "raw data file absent or unreadable")

    present = [c for c in resolvable if c in frame.columns]
    if not present:
        return _skip(code, ctx, "no named variable is in the Tier-1 cache")

    sentinels = ctx.sentinels
    keep = None
    for column in present:
        mask = _missing_mask(frame[column], ctx.var_map.get(column) or {}, sentinels)
        keep = (~mask) if keep is None else (keep & (~mask))
    listwise_n = int(keep.sum()) if keep is not None else 0
    total = int(len(frame))

    outcome = _outcome_variable(ctx)
    if outcome and outcome in frame.columns:
        outcome_mask = _missing_mask(
            frame[outcome], ctx.var_map.get(outcome) or {}, sentinels
        )
        n = int((~outcome_mask).sum())
        basis = f"outcome-complete on {outcome}"
    else:
        n = listwise_n
        basis = f"listwise-complete across {len(present)} variables (no single outcome column)"

    floor = 1000  # SPEC sec. 8: analytic_n < 1000 aborts the pipeline
    message = (
        f"Analytic n = {n:,} of {total:,} rows ({basis}); "
        f"listwise-complete across all {len(present)} named variables = "
        f"{listwise_n:,}."
    )
    evidence = (
        f"Tier-1 cache for {ctx.dataset} ({total:,} rows); sentinel-aware "
        f"missingness from {ctx.dataset}.yaml range/ + "
        f"missingness.sentinel_codes_or_labels"
    )
    if n < floor:
        return CheckResult(
            code, WARN,
            message + f" Below the {floor:,}-row pipeline abort floor.",
            evidence, PENALTY[code] * 3,
        )
    if n < 10_000 and ctx.task_type == "prediction":
        return CheckResult(
            code, WARN,
            message + " Below the 10,000-row prediction feasibility target.",
            evidence, PENALTY[code],
        )
    return CheckResult(code, OK, message, evidence)


def check_class_balance(ctx: ScreenContext) -> CheckResult:
    """Measured class balance for a binary outcome."""
    code = "P-CLASS-BALANCE"
    target = _resolved_target(ctx.spec)
    if ctx.task_type.startswith("causal"):
        target = _as_name(ctx.spec.get("outcome")) or target
    meta = ctx.var_map.get(str(target) or "") or {}
    if meta.get("type") != "binary":
        return _skip(code, ctx, f"outcome {target!r} is not binary in the registry")
    frame = _load_frame(ctx, [str(target)])
    if frame is None or str(target) not in getattr(frame, "columns", []):
        return _skip(code, ctx, "raw data file absent or outcome not cached")

    series = frame[str(target)]
    mask = _missing_mask(series, meta, ctx.sentinels)
    values = series[~mask]
    counts = values.value_counts(normalize=True)
    if counts.empty or len(counts) < 2:
        return _skip(code, ctx, f"outcome {target!r} has fewer than 2 observed levels")
    minority = float(counts.min())
    evidence = (
        f"Tier-1 cache column {target}: "
        + ", ".join(f"{k}={v:.3f}" for k, v in counts.items())
    )
    if minority < 0.05:
        return CheckResult(
            code, WARN,
            f"Severe class imbalance: minority class = {minority:.1%}.",
            evidence, PENALTY[code] * 2,
        )
    return CheckResult(
        code, OK, f"Minority class = {minority:.1%}.", evidence,
    )


def check_positivity(ctx: ScreenContext) -> CheckResult:
    """Pre-run propensity-overlap probe for causal task types.

    Same 0.10 extreme-tail threshold ``src/causal_data_contract.py``
    enforces after the DataEngineer stage - but at 0.1 s of compute,
    before ~$2-3 of pipeline spend.
    """
    code = "P-POSITIVITY"
    if ctx.task_type not in ("causal_soo", "causal_itr"):
        return _skip(code, ctx, f"positivity does not apply to {ctx.task_type}")
    treatment = _as_name(ctx.spec.get("treatment"))
    covariates = [
        n for n, role in spec_variable_names(ctx.spec)
        if role in ("covariate", "rule_covariate")
    ]
    if not treatment or not covariates:
        return _skip(code, ctx, "spec names no treatment or no adjustment set")
    frame = _load_frame(ctx, [treatment, *covariates])
    if frame is None or treatment not in getattr(frame, "columns", []):
        return _skip(code, ctx, "raw data file absent or treatment not cached")

    import numpy as np
    import pandas as pd

    meta = ctx.var_map.get(treatment) or {}
    t_raw = frame[treatment]
    t_mask = _missing_mask(t_raw, meta, ctx.sentinels)
    work = frame[~t_mask].copy()
    if len(work) < 200:  # mirrors causal_data_contract._PS_MIN_ROWS
        return _skip(code, ctx, "fewer than 200 complete treatment rows")

    t_values = work[treatment]
    if meta.get("type") == "continuous" or pd.api.types.is_numeric_dtype(t_values):
        numeric = pd.to_numeric(t_values, errors="coerce")
        if numeric.nunique(dropna=True) > 2:
            treated = (numeric > numeric.median()).astype(float)
            operationalization = "median split (registry type continuous)"
        else:
            codes = pd.Categorical(numeric).codes
            treated = pd.Series(codes, index=work.index).astype(float)
            operationalization = "numeric binary as-is"
    else:
        levels = t_values.astype(str)
        if levels.nunique() != 2:
            return _skip(
                code, ctx,
                f"treatment {treatment!r} has {levels.nunique()} label levels; "
                "no deterministic binarization",
            )
        treated = pd.Series(
            pd.Categorical(levels).codes, index=work.index
        ).astype(float)
        operationalization = "two-level categorical as-is"
    if treated.nunique() < 2:
        return _skip(code, ctx, "treatment is constant after binarization")

    design_cols: list[Any] = []
    used: list[str] = []
    for name in covariates:
        if name not in work.columns:
            continue
        cmeta = ctx.var_map.get(name) or {}
        column = work[name]
        cmask = _missing_mask(column, cmeta, ctx.sentinels)
        if cmeta.get("type") == "continuous":
            numeric = pd.to_numeric(column, errors="coerce")
            numeric = numeric.mask(cmask)
            design_cols.append(numeric.fillna(numeric.median()).rename(name))
            used.append(name)
        else:
            labels = column.astype(str).mask(cmask, "__missing__")
            if labels.nunique() > 25:  # one-hot cardinality guard
                continue
            dummies = pd.get_dummies(labels, prefix=name, drop_first=True)
            design_cols.append(dummies.astype(float))
            used.append(name)
    if not design_cols:
        return _skip(code, ctx, "no usable covariate columns for the probe")

    X = pd.concat(design_cols, axis=1).astype(float)
    std = X.std(numeric_only=True).replace(0, 1.0)
    X = (X - X.mean(numeric_only=True)) / std
    X = X.fillna(0.0)

    try:
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=200, C=1.0, random_state=42)
        model.fit(X.values, treated.values)
        ps = model.predict_proba(X.values)[:, 1]
    except Exception as exc:
        return _skip(code, ctx, f"propensity fit failed ({type(exc).__name__}: {exc})")

    try:
        from src.causal_data_contract import (
            _PS_TAIL_FRACTION_VIOLATION as threshold,
            _PS_TAIL_HIGH as high,
            _PS_TAIL_LOW as low,
        )
    except Exception:  # pragma: no cover - keep the probe self-sufficient
        low, high, threshold = 0.05, 0.95, 0.10
    tail = float(np.mean((ps < low) | (ps > high)))
    evidence = (
        f"bounded logistic propensity fit on the Tier-1 cache "
        f"({len(work):,} rows, treatment={treatment} via {operationalization}, "
        f"{len(used)} covariates); thresholds from "
        f"src/causal_data_contract.py (tails <{low} or >{high}, "
        f"violation tier {threshold:.0%})"
    )
    if tail >= threshold:
        return CheckResult(
            code, WARN,
            f"Pre-run positivity probe: {tail:.1%} of rows in the extreme "
            "propensity tails - this encoding is heading for a positivity "
            "violation.",
            evidence, PENALTY[code] * 2,
        )
    return CheckResult(
        code, OK, f"Positivity probe clean: {tail:.1%} in the extreme tails.",
        evidence,
    )


def check_did_cells(ctx: ScreenContext) -> CheckResult:
    """2x2 group x post cell counts for a DiD spec."""
    code = "P-DID-CELLS"
    if ctx.task_type != "causal_did":
        return _skip(code, ctx, f"cell counts do not apply to {ctx.task_type}")
    group = ctx.spec.get("group_variable")
    post = ctx.spec.get("post_variable")
    if not group or not post:
        return _skip(code, ctx, "spec names no group/post variable")
    frame = _load_frame(ctx, [str(group), str(post)])
    if frame is None or not {str(group), str(post)} <= set(getattr(frame, "columns", [])):
        return _skip(code, ctx, "raw data file absent or design columns not cached")

    counts = frame.groupby([str(group), str(post)]).size()
    cells = {f"{g}x{p}": int(n) for (g, p), n in counts.items()}
    evidence = f"Tier-1 cache groupby({group}, {post}) -> {cells}"
    if len(cells) < 4:
        return CheckResult(
            code, WARN,
            f"The 2x2 design has only {len(cells)} populated cell(s); a "
            "gap-in-gaps contrast needs all four.",
            evidence, PENALTY[code] * 2,
        )
    smallest = min(cells.values())
    if smallest < 100:
        return CheckResult(
            code, WARN,
            f"Smallest 2x2 cell has {smallest} rows.",
            evidence, PENALTY[code],
        )
    return CheckResult(
        code, OK, f"All four cells populated; smallest = {smallest:,} rows.",
        evidence,
    )


def check_cdm_scope(ctx: ScreenContext) -> CheckResult:
    """Does the log support the registry's recommended CDM/IRT scope?"""
    code = "P-CDM-SCOPE"
    support = ctx.registry.get("cdm_support") or {}
    if ctx.task_type != "psychometrics" or not support:
        return _skip(
            code, ctx,
            "no cdm_support block in the registry or not a measurement study",
        )
    if not ctx.spec.get("item_construction"):
        return _skip(code, ctx, "spec reads fixed item columns, not log data")

    path = probe_cache.raw_data_path(ctx.dataset or "", ctx.raw_data_dir)
    if path is None or not path.exists():
        return _skip(code, ctx, "raw log file absent")

    import pandas as pd

    item_unit = str(support.get("item_unit") or "template_id")
    try:
        log = pd.read_csv(
            path,
            usecols=[c for c in ("user_id", item_unit, "skill_id", "original") ],
            low_memory=False,
        )
    except Exception as exc:
        return _skip(code, ctx, f"log unreadable ({type(exc).__name__}: {exc})")

    if "original" in log.columns:
        log = log[log["original"] == 1]
    per_item = log.groupby(item_unit)["user_id"].nunique()
    usable_items = int((per_item >= 300).sum())
    n_skills = int(log["skill_id"].nunique()) if "skill_id" in log.columns else 0
    evidence = (
        f"{path.name}: {len(log):,} main-problem rows, {usable_items} "
        f"{item_unit}s with >=300 respondents, {n_skills} tagged skills; "
        f"registry cdm_support.recommended_scope = "
        f"{support.get('recommended_scope')!r}"
    )
    if usable_items < 8:
        return CheckResult(
            code, WARN,
            f"Only {usable_items} item(s) meet the >=300-response scope rule.",
            evidence, PENALTY[code] * 2,
        )
    return CheckResult(
        code, OK,
        f"{usable_items} items meet the >=300-response scope rule.",
        evidence,
    )


STAGE1_PROBES = (
    estimate_analytic_n,
    check_class_balance,
    check_positivity,
    check_did_cells,
    check_cdm_scope,
)


def probe(ctx: ScreenContext) -> list[CheckResult]:
    """Run every Stage-1 probe. Probes skip, they never raise."""
    results: list[CheckResult] = []
    for fn in STAGE1_PROBES:
        try:
            results.append(fn(ctx))
        except Exception as exc:  # pragma: no cover - probes must never crash
            results.append(
                CheckResult(
                    f"P-{fn.__name__.upper()}", OK,
                    f"Skipped: probe raised {type(exc).__name__}.",
                    f"{fn.__name__} raised: {exc}",
                )
            )
    return results


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------


def screen(
    spec: dict,
    *,
    candidate_id: str = "spec",
    dataset: str | None = None,
    task_type: str | None = None,
    registry: dict | None = None,
    registry_dir: str | os.PathLike[str] | None = None,
    raw_data_dir: str | os.PathLike[str] | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    columns: Iterable[str] | None = None,
    card: dict | None = None,
    run_probes: bool = False,
    context: ScreenContext | None = None,
) -> FeasibilityReport:
    """Run the deterministic screen over one research spec / idea card.

    Stage 0 always runs (free, no data load). Stage 1 runs only when
    ``run_probes`` is set, and skips cleanly when the raw data is absent.
    """
    ctx = context or make_context(
        spec,
        dataset=dataset,
        task_type=task_type,
        registry=registry,
        registry_dir=registry_dir,
        raw_data_dir=raw_data_dir,
        cache_dir=cache_dir,
        columns=columns,
        card=card,
    )

    checks: list[CheckResult] = []
    for fn in STAGE0_CHECKS:
        try:
            checks.append(fn(ctx))
        except Exception as exc:  # never let a check crash the screen
            checks.append(
                CheckResult(
                    f"F-CHECK-ERROR:{fn.__name__}", WARN,
                    f"Check raised {type(exc).__name__}; treated as "
                    "unverified (never as a kill).",
                    f"{fn.__name__} raised: {exc}",
                    PENALTY["F-METADATA-UNVERIFIED"],
                )
            )

    analytic_n: int | None = None
    if run_probes:
        probe_results = probe(ctx)
        checks.extend(probe_results)
        for result in probe_results:
            if result.code == "P-ANALYTIC-N":
                match = re.search(r"Analytic n = ([\d,]+)", result.message)
                if match:
                    analytic_n = int(match.group(1).replace(",", ""))

    penalty = sum(c.penalty for c in checks)
    if any(c.status == KILL for c in checks):
        verdict = KILL
    elif any(c.status == WARN for c in checks):
        verdict = WARN
    else:
        verdict = CLEAN

    return FeasibilityReport(
        candidate_id=candidate_id,
        verdict=verdict,
        checks=checks,
        analytic_n_estimate=analytic_n,
        penalty=penalty,
        dataset=ctx.dataset,
        task_type=ctx.task_type,
    )


def rank_key(spec: dict, **kwargs: Any) -> tuple:
    """Deterministic ranking key for a spec - lower sorts better.

    C1: novelty is not a term. This exists so
    ``ProblemFormulator._select_best_candidate`` can delegate to it
    (spec sec. 1.4) instead of ranking on ``novelty_score_self_assessment``,
    which measures r = -0.35 against the criterion it is meant to
    predict.
    """
    report = screen(spec, **kwargs)
    return (
        0 if report.verdict != KILL else 1,
        len(report.kills),
        round(report.penalty, 4),
        len(report.warns),
        str(spec.get("task_id") or _resolved_target(spec) or ""),
    )


def _cli() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Arc T deterministic feasibility screen. No LLM, no network."
        )
    )
    parser.add_argument("--spec", required=True, help="Path to a research_spec JSON")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--task-type", default=None, dest="task_type")
    parser.add_argument("--registry-dir", default=None, dest="registry_dir")
    parser.add_argument("--raw-data-dir", default=None, dest="raw_data_dir")
    parser.add_argument(
        "--probes", action="store_true",
        help="Also run the Stage-1 data probes (needs the raw data files)",
    )
    parser.add_argument("--json", default=None, help="Write the report as JSON here")
    args = parser.parse_args()

    with open(args.spec, encoding="utf-8") as f:
        spec = json.load(f)

    report = screen(
        spec,
        candidate_id=Path(args.spec).stem,
        dataset=args.dataset,
        task_type=args.task_type,
        registry_dir=args.registry_dir,
        raw_data_dir=args.raw_data_dir,
        run_probes=args.probes,
    )
    print(report.render())
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=1)
    return 1 if report.verdict == KILL else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
