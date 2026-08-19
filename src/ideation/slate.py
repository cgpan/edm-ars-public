"""Arc T / T1a - deterministic enumeration of the candidate space.

The slate is the whole diversity mechanism. Nothing downstream asks a
model to "be diverse": the (dataset x task_type x opportunity pattern x
persona x gap cell) assignment is computed here, before any LLM call,
from a seeded RNG and a set of quotas, and every decision is written to
``slate.json`` so a slate can be reproduced and audited.

Why structural, and not a prompt instruction
--------------------------------------------
Measured over the 26 archived runs: 100% are mathematics, 53.8% share
one outcome, and in the only free-generation regime the mean pairwise
predictor Jaccard between candidate specs was 0.837 (one pair a literal
subset). Those candidates came from one prompt asking for several
different ideas. A slate cannot produce that failure, because two
candidates cannot share a cell more than ``max_per_cell`` times and the
cells are drawn before generation starts.

Infeasible cells are never enumerated (two rules, both mirroring a
shipped KILL in ``feasibility.py``):

* ``S1`` ``DATASET_TASK_MATRIX[dataset][task_type]`` must be true -
  the same table ``check_dataset_task_compatibility`` kills on.
* ``S2`` the ``equity_subgroup_gap`` pattern requires the dataset to
  declare at least one ``protected_attribute: true`` variable - the
  same fact ``check_protected_attributes`` kills on. ASSISTments
  declares zero, so an equity card there is dead on arrival and there is
  no reason to spend a generation call on it.
* ``S3`` ``prediction`` requires a dataset with at least two waves in
  ``temporal_order``. Measured 2026-07-26: a compiled prediction spec on
  ``assistments_0910`` (``temporal_order: [single_year]``) is rejected by
  ``src.main.load_locked_research_spec`` with a blocking TEMPORAL
  VIOLATION for every predictor, because predictor and outcome
  necessarily share the single wave. ``DATASET_TASK_MATRIX`` marks that
  cell feasible; the loader disagrees, and the loader is the seam that
  matters. See the hand-off note - this is a defect in a file this slice
  does not own.

No LLM, no network, no data load.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from src.ideation import feasibility as _feas

# --------------------------------------------------------------------------
# Axes
# --------------------------------------------------------------------------

#: The eight core opportunity patterns (spec sec. 2.2). This is the
#: diversity axis that matters - topic diversity is cosmetic, framing
#: diversity is not.
OPPORTUNITY_PATTERNS: tuple[str, ...] = (
    "puzzle_anomaly",
    "explanation_gap",
    "measurement_bottleneck",
    "equity_subgroup_gap",
    "replication_transfer",
    "design_upgrade",
    "scope_extension",
    "robustification",
)

#: The ninth pattern, hard-capped. LLMs default to 47-64% bridge/synthesis
#: framings against 12.1% for humans, so it is allowed but rationed.
BRIDGE_PATTERN = "bridge_synthesis"

#: Ceiling on the bridge share of a slate, independent of the absolute
#: cap in config (``bridge_framing_quota``). Both bind; the smaller wins.
BRIDGE_MAX_SHARE = 0.15

PATTERN_BRIEFS: dict[str, str] = {
    "puzzle_anomaly": (
        "Start from a result that does not fit: a subgroup that behaves "
        "backwards, a predictor that flips sign, a distribution with a "
        "shape nobody explains."
    ),
    "explanation_gap": (
        "Something is well established descriptively and unexplained "
        "mechanistically. Name the competing explanations and what would "
        "separate them."
    ),
    "measurement_bottleneck": (
        "A construct everyone models is measured badly. Attack the "
        "instrument, not the outcome."
    ),
    "equity_subgroup_gap": (
        "A performance, measurement or effect difference across a "
        "protected attribute that the standard analysis averages away."
    ),
    "replication_transfer": (
        "Take a specific published finding and ask whether it survives in "
        "a different cohort, instrument or context. Name the finding."
    ),
    "design_upgrade": (
        "An existing question is answered with a weaker design than the "
        "data supports. Upgrade the identification, keep the question."
    ),
    "scope_extension": (
        "A method or claim is established on a narrow slice. Extend the "
        "scope and state what could break."
    ),
    "robustification": (
        "A widely used result depends on an unexamined choice. Vary the "
        "choice and report where the conclusion changes."
    ),
    BRIDGE_PATTERN: (
        "Combine two literatures that do not cite each other. Rationed: "
        "this framing is over-produced by language models and is capped "
        "in the slate."
    ),
}

#: Ordinary heterogeneous research roles (spec sec. 2.2). Deliberately
#: not celebrity-creative personas: those sample from a densely connected
#: region of the space and REDUCE diversity.
PERSONAS: tuple[str, ...] = (
    "psychometrician",
    "equity_researcher",
    "policy_analyst",
    "replication_methodologist",
    "measurement_to_decision_analyst",
    "causal_econometrician",
)

PERSONA_BRIEFS: dict[str, str] = {
    "psychometrician": (
        "You care whether the numbers mean what people say they mean. "
        "Reason first about the instrument, its items, and what its "
        "score is comparable across."
    ),
    "equity_researcher": (
        "You care about who is served badly by the average model. Reason "
        "first about which group the aggregate hides, and what the "
        "difference would imply for practice."
    ),
    "policy_analyst": (
        "You care about a decision someone actually makes. Reason first "
        "about the decision, its counterfactual, and the magnitude that "
        "would change it."
    ),
    "replication_methodologist": (
        "You care whether a published claim holds up. Reason first about "
        "which specific claim you are testing and what result would "
        "count as a failure to replicate."
    ),
    "measurement_to_decision_analyst": (
        "You care about the chain from a measurement to an action. Reason "
        "first about where the chain breaks and what it costs."
    ),
    "causal_econometrician": (
        "You care about identification. Reason first about the "
        "assumption that carries the claim and what would falsify it."
    ),
}

DEFAULT_N_CANDIDATES = 24
DEFAULT_SEED = 42
DEFAULT_BRIDGE_QUOTA = 3
DEFAULT_MAX_PER_CELL = 3

SCHEMA_VERSION = "1.0"


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SlateCell:
    """One fully specified candidate slot. Assigned before generation."""

    candidate_id: str
    dataset: str
    task_type: str
    opportunity_pattern: str
    persona: str
    gap_cell: tuple[str, str] | None = None

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "dataset": self.dataset,
            "task_type": self.task_type,
            "opportunity_pattern": self.opportunity_pattern,
            "persona": self.persona,
            "gap_cell": list(self.gap_cell) if self.gap_cell else None,
        }


@dataclass
class QuotaDecision:
    """One recorded decision. ``evidence`` names the fact it read (C2)."""

    rule: str
    decision: str
    evidence: str

    def to_dict(self) -> dict:
        return {"rule": self.rule, "decision": self.decision, "evidence": self.evidence}


@dataclass
class Slate:
    tournament_id: str
    random_state: int
    n_requested: int
    cells: list[SlateCell] = field(default_factory=list)
    dataset_task_cells: list[tuple[str, str]] = field(default_factory=list)
    excluded_cells: list[dict] = field(default_factory=list)
    quota_decisions: list[QuotaDecision] = field(default_factory=list)
    axes: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.cells)

    def by_id(self, candidate_id: str) -> SlateCell | None:
        for cell in self.cells:
            if cell.candidate_id == candidate_id:
                return cell
        return None

    def diversity_ledger(self) -> dict:
        """Counts per axis. Printed as a pass/fail line, never buried."""

        def _counts(key: str) -> dict[str, int]:
            out: dict[str, int] = {}
            for cell in self.cells:
                value = str(getattr(cell, key))
                out[value] = out.get(value, 0) + 1
            return dict(sorted(out.items()))

        patterns = _counts("opportunity_pattern")
        return {
            "n_candidates": len(self.cells),
            "datasets": _counts("dataset"),
            "task_types": _counts("task_type"),
            "opportunity_patterns": patterns,
            "personas": _counts("persona"),
            "distinct_dataset_task_cells": len(
                {(c.dataset, c.task_type) for c in self.cells}
            ),
            "bridge_share": (
                round(patterns.get(BRIDGE_PATTERN, 0) / len(self.cells), 4)
                if self.cells
                else 0.0
            ),
            "core_patterns_covered": sum(
                1 for p in OPPORTUNITY_PATTERNS if patterns.get(p, 0) > 0
            ),
            "core_patterns_total": len(OPPORTUNITY_PATTERNS),
        }

    def to_dict(self) -> dict:
        """Fully deterministic: no timestamps, no paths, no run state."""
        return {
            "schema_version": SCHEMA_VERSION,
            "tournament_id": self.tournament_id,
            "random_state": self.random_state,
            "n_requested": self.n_requested,
            "n_enumerated": len(self.cells),
            "axes": self.axes,
            "dataset_task_cells": [list(c) for c in self.dataset_task_cells],
            "excluded_cells": self.excluded_cells,
            "quota_decisions": [d.to_dict() for d in self.quota_decisions],
            "diversity_ledger": self.diversity_ledger(),
            "cells": [c.to_dict() for c in self.cells],
        }


# --------------------------------------------------------------------------
# Structural feasibility of a (dataset, task_type, pattern) triple
# --------------------------------------------------------------------------


def _unsupported_reason(dataset: str, task_type: str) -> str:
    """Dispositive reason from feasibility.py, with a safe fallback.

    Read through ``getattr`` because ``_UNSUPPORTED_REASONS`` is private
    to a shipped module this slice may not edit; if it is ever renamed
    the slate degrades to the matrix-level statement rather than
    breaking.
    """
    reasons = getattr(_feas, "_UNSUPPORTED_REASONS", {}) or {}
    reason = reasons.get((dataset, task_type))
    if reason:
        return str(reason)
    return (
        f"feasibility.DATASET_TASK_MATRIX[{dataset!r}][{task_type!r}] is False"
    )


#: Minimum number of declared waves a dataset needs before a prediction
#: spec compiled against it can survive the temporal-ordering check.
_MIN_WAVES_FOR_PREDICTION = 2


def cell_allowed(
    dataset: str,
    task_type: str,
    *,
    registry: dict | None = None,
    registry_dir: str | os.PathLike[str] | None = None,
) -> tuple[bool, str, str]:
    """``(allowed, rule, evidence)`` for one (dataset, task_type) cell."""
    row = _feas.DATASET_TASK_MATRIX.get(dataset)
    if row is None:
        return (
            False,
            "S1",
            f"{dataset!r} is not a row of feasibility.DATASET_TASK_MATRIX",
        )
    if not row.get(task_type):
        return False, "S1", _unsupported_reason(dataset, task_type)

    if task_type == "prediction":
        reg = registry
        if reg is None:
            reg, _path = _feas.load_registry(dataset, registry_dir)
        waves = [str(w) for w in ((reg or {}).get("temporal_order") or [])]
        if len(waves) < _MIN_WAVES_FOR_PREDICTION:
            return (
                False,
                "S3",
                (
                    f"{dataset}.yaml temporal_order declares {len(waves)} "
                    f"wave(s) ({waves}); predictors and outcome would share "
                    f"a wave, and PredictionTemplate.validate_research_spec "
                    f"then returns a blocking TEMPORAL VIOLATION for every "
                    f"predictor, so the compiled spec cannot pass "
                    f"src.main.load_locked_research_spec (measured "
                    f"2026-07-26)"
                ),
            )
    return True, "", f"DATASET_TASK_MATRIX[{dataset!r}][{task_type!r}] is True"


def protected_attributes(
    dataset: str,
    registry_dir: str | os.PathLike[str] | None = None,
    registry: dict | None = None,
) -> list[str]:
    """Names the dataset registry marks ``protected_attribute: true``."""
    reg = registry
    if reg is None:
        reg, _path = _feas.load_registry(dataset, registry_dir)
    var_map = _feas.build_var_map(reg or {})
    return sorted(
        str(name)
        for name, meta in var_map.items()
        if isinstance(meta, dict) and meta.get("protected_attribute")
    )


def pattern_allowed(
    pattern: str,
    dataset: str,
    *,
    registry_dir: str | os.PathLike[str] | None = None,
    protected_by_dataset: dict[str, list[str]] | None = None,
) -> tuple[bool, str]:
    """``(allowed, evidence)`` for one (pattern, dataset) pair - rule S2."""
    if pattern != "equity_subgroup_gap":
        return True, "no structural constraint on this pattern"
    if protected_by_dataset is not None:
        protected = protected_by_dataset.get(dataset, [])
    else:
        protected = protected_attributes(dataset, registry_dir)
    if protected:
        return True, (
            f"{dataset}.yaml declares {len(protected)} protected "
            f"attribute(s): {', '.join(protected)}"
        )
    return False, (
        f"{dataset}.yaml declares no variable with protected_attribute: "
        f"true; feasibility.check_protected_attributes KILLs an equity "
        f"card here"
    )


# --------------------------------------------------------------------------
# Allocation
# --------------------------------------------------------------------------


def allocate_patterns(
    n: int,
    *,
    bridge_quota: int = DEFAULT_BRIDGE_QUOTA,
    core_order: Sequence[str] | None = None,
) -> tuple[list[str], list[QuotaDecision]]:
    """Pattern multiset for ``n`` candidates. Deterministic given order.

    Bridge framings take ``min(bridge_quota, floor(n * BRIDGE_MAX_SHARE))``
    slots; the rest are spread over the eight core patterns as evenly as
    ``n`` allows, with the remainder handed out in ``core_order``.
    """
    core = list(core_order or OPPORTUNITY_PATTERNS)
    decisions: list[QuotaDecision] = []
    if n <= 0:
        return [], decisions

    bridge_slots = max(0, min(int(bridge_quota), int(n * BRIDGE_MAX_SHARE)))
    decisions.append(
        QuotaDecision(
            "Q-BRIDGE-CAP",
            f"bridge_synthesis allocated {bridge_slots} of {n} slots "
            f"({bridge_slots / n:.1%})",
            f"config bridge_framing_quota={bridge_quota}, structural ceiling "
            f"{BRIDGE_MAX_SHARE:.0%} of n={n}; the smaller binds",
        )
    )

    remaining = n - bridge_slots
    base, extra = divmod(remaining, len(core)) if core else (0, 0)
    counts = {pattern: base for pattern in core}
    for pattern in core[:extra]:
        counts[pattern] += 1
    decisions.append(
        QuotaDecision(
            "Q-PATTERN-ALLOC",
            f"core patterns allocated {base}-{base + (1 if extra else 0)} "
            f"slots each over {len(core)} patterns "
            f"({sorted(counts.items())})",
            f"n={n} minus {bridge_slots} bridge slots, divided evenly; the "
            f"remainder {extra} went to the first {extra} pattern(s) of the "
            f"seeded order",
        )
    )

    sequence: list[str] = []
    for pattern in core:
        sequence.extend([pattern] * counts[pattern])
    sequence.extend([BRIDGE_PATTERN] * bridge_slots)
    return sequence, decisions


def build_slate(
    tournament_id: str,
    *,
    n_candidates: int = DEFAULT_N_CANDIDATES,
    seed: int = DEFAULT_SEED,
    datasets: Iterable[str] | None = None,
    task_types: Iterable[str] | None = None,
    bridge_quota: int = DEFAULT_BRIDGE_QUOTA,
    max_per_cell: int = DEFAULT_MAX_PER_CELL,
    registry_dir: str | os.PathLike[str] | None = None,
    s2_context: dict | None = None,
    gap_cells: Sequence[tuple[str, str]] | None = None,
) -> Slate:
    """Enumerate the candidate space. Same inputs + seed -> same slate."""
    matrix = _feas.DATASET_TASK_MATRIX
    dataset_list = sorted(datasets) if datasets is not None else sorted(matrix)
    if task_types is not None:
        task_list = list(task_types)
    else:
        task_list = sorted({t for row in matrix.values() for t in row})

    decisions: list[QuotaDecision] = []
    feasible: list[tuple[str, str]] = []
    excluded: list[dict] = []
    registries: dict[str, dict] = {}
    for dataset in dataset_list:
        registry, _path = _feas.load_registry(dataset, registry_dir)
        registries[dataset] = registry or {}
        for task in task_list:
            allowed, rule, evidence = cell_allowed(
                dataset, task, registry=registries[dataset]
            )
            if allowed:
                feasible.append((dataset, task))
            else:
                excluded.append(
                    {
                        "dataset": dataset,
                        "task_type": task,
                        "rule": rule,
                        "reason": evidence,
                    }
                )
    decisions.append(
        QuotaDecision(
            "Q-MATRIX",
            f"{len(feasible)} of {len(dataset_list) * len(task_list)} "
            f"(dataset x task_type) cells enumerated; "
            f"{len(excluded)} excluded "
            f"({sum(1 for e in excluded if e['rule'] == 'S1')} by S1, "
            f"{sum(1 for e in excluded if e['rule'] == 'S3')} by S3)",
            "feasibility.DATASET_TASK_MATRIX (rule S1) + the >=2-wave "
            "requirement for prediction (rule S3); excluded cells carry "
            "their dispositive reason in slate.json excluded_cells",
        )
    )

    protected_by_dataset = {
        dataset: protected_attributes(
            dataset, registry_dir, registries.get(dataset)
        )
        for dataset in {d for d, _t in feasible}
    }
    for dataset, names in sorted(protected_by_dataset.items()):
        if not names:
            decisions.append(
                QuotaDecision(
                    "Q-EQUITY-PROTECTED",
                    f"pattern equity_subgroup_gap is not enumerable on "
                    f"{dataset}",
                    pattern_allowed(
                        "equity_subgroup_gap",
                        dataset,
                        protected_by_dataset=protected_by_dataset,
                    )[1],
                )
            )

    rng = random.Random(seed)
    core_order = list(OPPORTUNITY_PATTERNS)
    rng.shuffle(core_order)
    sequence, alloc_decisions = allocate_patterns(
        n_candidates, bridge_quota=bridge_quota, core_order=core_order
    )
    decisions.extend(alloc_decisions)
    rng.shuffle(sequence)

    cell_order = list(feasible)
    rng.shuffle(cell_order)
    cell_rank = {cell: i for i, cell in enumerate(cell_order)}
    usage: dict[tuple[str, str], int] = {cell: 0 for cell in cell_order}

    persona_order = list(PERSONAS)
    rng.shuffle(persona_order)

    gap_pool = list(gap_cells) if gap_cells is not None else _gap_cells(s2_context)
    gap_order = list(gap_pool)
    rng.shuffle(gap_order)

    cells: list[SlateCell] = []
    pattern_counts: dict[str, int] = {}
    for index, pattern in enumerate(sequence):
        chosen, pattern_used, event = _choose_cell(
            pattern,
            cell_order,
            usage,
            cell_rank,
            max_per_cell,
            protected_by_dataset,
            pattern_counts,
        )
        if event is not None:
            decisions.append(event)
        if chosen is None:
            continue
        usage[chosen] += 1
        pattern_counts[pattern_used] = pattern_counts.get(pattern_used, 0) + 1
        candidate_id = f"C-{len(cells) + 1:02d}"
        gap = gap_order[len(cells) % len(gap_order)] if gap_order else None
        cells.append(
            SlateCell(
                candidate_id=candidate_id,
                dataset=chosen[0],
                task_type=chosen[1],
                opportunity_pattern=pattern_used,
                persona=persona_order[len(cells) % len(persona_order)],
                gap_cell=(str(gap[0]), str(gap[1])) if gap else None,
            )
        )

    decisions.append(
        QuotaDecision(
            "Q-CELL-CAP",
            f"no (dataset x task_type) cell carries more than "
            f"{max(usage.values()) if usage else 0} candidate(s); cap is "
            f"{max_per_cell}",
            f"{len({c for c, u in usage.items() if u})} distinct cells used "
            f"of {len(cell_order)} enumerated",
        )
    )
    decisions.append(
        QuotaDecision(
            "Q-PERSONA-ROTATION",
            f"personas assigned round-robin over a seed-{seed} shuffle of "
            f"{len(persona_order)} roles",
            "src/ideation/slate.py PERSONAS; assignment index = position in "
            "the slate",
        )
    )
    decisions.append(
        QuotaDecision(
            "Q-GAP-ROTATION",
            f"gap cells assigned round-robin over {len(gap_order)} sparse "
            f"cell(s); no cell repeats before all are used",
            "src/gap_miner.py build_gap_matrix(...)['sparse_cells']"
            + ("" if s2_context else " with no retrieved corpus (all cells sparse)"),
        )
    )

    return Slate(
        tournament_id=tournament_id,
        random_state=seed,
        n_requested=n_candidates,
        cells=cells,
        dataset_task_cells=feasible,
        excluded_cells=excluded,
        quota_decisions=decisions,
        axes={
            "datasets": dataset_list,
            "task_types": task_list,
            "opportunity_patterns": list(OPPORTUNITY_PATTERNS),
            "bridge_pattern": BRIDGE_PATTERN,
            "personas": list(PERSONAS),
            "max_per_cell": max_per_cell,
            "bridge_quota": bridge_quota,
            "protected_attributes": {
                k: v for k, v in sorted(protected_by_dataset.items())
            },
        },
    )


def _choose_cell(
    pattern: str,
    cell_order: Sequence[tuple[str, str]],
    usage: dict[tuple[str, str], int],
    cell_rank: dict[tuple[str, str], int],
    max_per_cell: int,
    protected_by_dataset: dict[str, list[str]],
    pattern_counts: dict[str, int],
) -> tuple[tuple[str, str] | None, str, QuotaDecision | None]:
    """Least-used compatible cell for ``pattern``; substitute if none."""

    def _compatible(p: str) -> list[tuple[str, str]]:
        return [
            cell
            for cell in cell_order
            if pattern_allowed(
                p, cell[0], protected_by_dataset=protected_by_dataset
            )[0]
            and usage[cell] < max_per_cell
        ]

    options = _compatible(pattern)
    if options:
        chosen = min(options, key=lambda c: (usage[c], cell_rank[c]))
        return chosen, pattern, None

    # No compatible cell under the cap. Substitute the least-used core
    # pattern that does have one, and say so.
    for alternative in sorted(
        OPPORTUNITY_PATTERNS, key=lambda p: (pattern_counts.get(p, 0), p)
    ):
        if alternative == pattern:
            continue
        options = _compatible(alternative)
        if options:
            chosen = min(options, key=lambda c: (usage[c], cell_rank[c]))
            return (
                chosen,
                alternative,
                QuotaDecision(
                    "Q-PATTERN-SUBSTITUTION",
                    f"pattern {pattern!r} had no enumerable cell under the "
                    f"per-cell cap; substituted {alternative!r}",
                    pattern_allowed(
                        pattern,
                        chosen[0],
                        protected_by_dataset=protected_by_dataset,
                    )[1],
                ),
            )
    return (
        None,
        pattern,
        QuotaDecision(
            "Q-SLOT-DROPPED",
            f"no cell available for pattern {pattern!r} under the per-cell "
            f"cap; slot dropped and the slate is shorter than requested",
            f"every enumerated cell is at the cap ({max_per_cell})",
        ),
    )


def _gap_cells(s2_context: dict | None) -> list[tuple[str, str]]:
    """Sparse (outcome family, method family) cells from the gap miner.

    With no retrieved corpus every cell is sparse, which is the honest
    offline answer: nothing has been retrieved, so nothing is known to be
    crowded.
    """
    try:
        from src.gap_miner import build_gap_matrix

        gap = build_gap_matrix(s2_context)
        return [(str(o), str(m)) for o, m in gap.get("sparse_cells") or []]
    except Exception:
        return []


def format_slate(slate: Slate) -> str:
    """Human-readable digest for the console and the run log."""
    ledger = slate.diversity_ledger()
    lines = [
        f"slate {slate.tournament_id}: {len(slate)} candidates "
        f"(requested {slate.n_requested}, seed {slate.random_state})",
        f"  dataset x task cells enumerated: {len(slate.dataset_task_cells)}"
        f"  excluded: {len(slate.excluded_cells)}",
        f"  distinct cells used: {ledger['distinct_dataset_task_cells']}"
        f"  core patterns covered: {ledger['core_patterns_covered']}"
        f"/{ledger['core_patterns_total']}"
        f"  bridge share: {ledger['bridge_share']:.1%}",
    ]
    for cell in slate.cells:
        lines.append(
            f"  {cell.candidate_id}  {cell.task_type:<14} {cell.dataset:<20} "
            f"{cell.opportunity_pattern:<24} {cell.persona}"
        )
    return "\n".join(lines)
