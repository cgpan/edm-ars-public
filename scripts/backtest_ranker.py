"""Arc T validation V2 - rank-inversion backtest, DETERMINISTIC terms only.

Spec: ``docs/v5_arc_t_spec.md`` sec. 6 "V2". This is the falsification
gate for Arc T's ranker and it runs BEFORE the judged tournament layer
exists, so that an inverted ranker costs nothing further.

What it does
------------
1. Compiles every archived research spec (``runs/*/output*/research_spec.json``)
   into the deterministic feature set that T0 already ships:
   ``src.ideation.feasibility.screen`` (verdict + WARN penalty) and
   ``src.ideation.venue_fit.score_venue_fit`` (anchor-derived rule table).
   The archived spec fields ARE the idea-card fields, so no LLM and no
   network are involved anywhere in this script.
2. Joins to realized LSAR gate scores, recovered from disk
   (``runs/<run>/output/lsar_review/cycle_*/scores.json``) and
   cross-checked against ``evaluation/ledger.json``.
3. Computes tie-corrected Spearman rho between the deterministic score
   and the realized gate score, with an exact permutation p-value.
4. Checks the pre-registered pair: ``phase_b_did_20260704`` (3.7 Reject,
   bare 2x2 DiD) must score BELOW ``stream1_did_v2_20260708`` (7.0
   Accept, same data, same estimand, wrapped in M9/M10).

Population, and why it is small
-------------------------------
The spec restricts the primary population to papers with MEDIAN-OF-3
gate scores. Two facts about how those arise, both read from the code
and the artifacts rather than assumed:

* ``src/review_gate.py::_maybe_median_sample`` writes the extra samples
  to ``cycle_{cycle*100+extra}``, so a median-of-3 run has exactly the
  directories ``cycle_1``, ``cycle_102``, ``cycle_103``. Directories
  ``cycle_2``/``cycle_3`` are REVISION cycles of different manuscripts
  and are not samples of one manuscript.
* Median sampling is BORDERLINE-TRIGGERED: it only fires when the first
  review lands within ``median_trigger_band`` (1.5) of the pass
  threshold. The median-of-3 population is therefore selected on the
  criterion, which restricts its range. This is reported, not hidden.

Exit codes (this script is a regression check)
----------------------------------------------
0  no falsification condition met (includes the UNDERPOWERED case, which
   is a persistent property of the archive, not a regression)
1  FALSIFIED: rho <= 0 (ranker inverted) or the pre-registered pair is
   not separated
2  the backtest could not be computed (no usable population)

Usage:
    python scripts/backtest_ranker.py
    python scripts/backtest_ranker.py --json backtest.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from dataclasses import asdict, dataclass, field
from itertools import permutations
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ideation.feasibility import screen  # noqa: E402
from src.ideation.venue_fit import score_venue_fit  # noqa: E402

# Spec sec. 4.1 deterministic prior weights. On this archive the penalty
# term has zero variance (every canonical spec screens CLEAN with
# penalty 0.0), so the composite is a positive affine transform of the
# venue-fit term and carries exactly the same ranking. That is reported
# as a finding, not smoothed over.
W_VENUE_FIT = 0.30
W_PENALTY = 0.20

# The calibrated instrument. EDM is the only venue with a calibrated
# threshold (P25 6.3); JEDM/JLA scores in this archive are explicitly
# labelled advisory in evaluation/ledger.json.
CALIBRATED_VENUE = "EDM"

PAIR_LOW = "phase_b_did_20260704"
PAIR_HIGH = "stream1_did_v2_20260708"


# --------------------------------------------------------------------------
# Statistics - implemented here, not imported, so the test can pin the
# arithmetic on a fixture with a known answer. scipy is not in
# requirements.txt and this script must run without it.
# --------------------------------------------------------------------------


def rank_with_ties(values: Sequence[float]) -> list[float]:
    """Average ("fractional") ranks, 1-based. Ties share their mean rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        mean_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = mean_rank
        i = j + 1
    return ranks


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson r. Returns NaN when either input has zero variance."""
    n = len(x)
    if n < 2 or len(y) != n:
        return float("nan")
    mx = sum(x) / n
    my = sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx <= 0.0 or syy <= 0.0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Tie-corrected Spearman rho (Pearson on average ranks)."""
    return pearson(rank_with_ties(x), rank_with_ties(y))


@dataclass
class PermutationResult:
    rho: float
    method: str  # "exact" | "monte_carlo"
    n: int
    n_permutations: int
    p_greater: float  # H1: rho > 0
    p_two_sided: float
    min_attainable_p_greater: float
    seed: int | None

    def to_dict(self) -> dict:
        return asdict(self)


def permutation_test(
    x: Sequence[float],
    y: Sequence[float],
    *,
    max_exact_n: int = 9,
    n_draws: int = 200_000,
    seed: int = 42,
) -> PermutationResult:
    """Permutation test on Spearman rho, permuting y against a fixed x.

    Exact (full enumeration) when ``n! `` is affordable, Monte Carlo with
    a fixed seed otherwise. ``min_attainable_p_greater`` is the smallest
    one-sided p this design can possibly produce - with ties and a small
    n it is often above 0.05, which is the honest answer to "is n big
    enough".
    """
    n = len(x)
    rx = rank_with_ties(x)
    ry = rank_with_ties(y)
    obs = pearson(rx, ry)
    tol = 1e-12

    if math.isnan(obs) or n < 3:
        return PermutationResult(
            rho=obs, method="undefined", n=n, n_permutations=0,
            p_greater=float("nan"), p_two_sided=float("nan"),
            min_attainable_p_greater=float("nan"), seed=None,
        )

    if n <= max_exact_n:
        perms = list(permutations(ry))
        rhos = [pearson(rx, p) for p in perms]
        total = len(rhos)
        method, used_seed = "exact", None
    else:
        rng = random.Random(seed)
        shuffled = list(ry)
        rhos = []
        for _ in range(n_draws):
            rng.shuffle(shuffled)
            rhos.append(pearson(rx, shuffled))
        total = n_draws
        method, used_seed = "monte_carlo", seed

    n_ge = sum(1 for r in rhos if r >= obs - tol)
    n_abs = sum(1 for r in rhos if abs(r) >= abs(obs) - tol)
    best = max(rhos)
    n_best = sum(1 for r in rhos if r >= best - tol)

    return PermutationResult(
        rho=obs,
        method=method,
        n=n,
        n_permutations=total,
        p_greater=n_ge / total,
        p_two_sided=n_abs / total,
        min_attainable_p_greater=n_best / total,
        seed=used_seed,
    )


def bootstrap_ci(
    x: Sequence[float],
    y: Sequence[float],
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Percentile bootstrap CI for Spearman rho.

    At the n available here this is decoration, not inference: a large
    share of resamples are degenerate (zero variance in one margin) and
    are dropped. ``n_degenerate`` is returned so the caller can say so.
    """
    n = len(x)
    rng = random.Random(seed)
    stats: list[float] = []
    degenerate = 0
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        rho = spearman([x[i] for i in idx], [y[i] for i in idx])
        if math.isnan(rho):
            degenerate += 1
        else:
            stats.append(rho)
    if not stats:
        return {"lo": float("nan"), "hi": float("nan"), "n_valid": 0,
                "n_degenerate": degenerate, "n_boot": n_boot, "seed": seed}
    stats.sort()
    lo = stats[max(0, int(math.floor((alpha / 2) * len(stats))))]
    hi = stats[min(len(stats) - 1, int(math.ceil((1 - alpha / 2) * len(stats))) - 1)]
    return {"lo": lo, "hi": hi, "n_valid": len(stats),
            "n_degenerate": degenerate, "n_boot": n_boot, "seed": seed}


def required_n_for_rho(
    rho: float, *, alpha: float = 0.05, power: float = 0.80,
    one_sided: bool = True,
) -> int:
    """Fisher-z sample size to detect a true Spearman rho. APPROXIMATE.

    Uses the normal approximation with the standard 1.06 variance
    inflation for Spearman. It is a planning number, not a guarantee,
    and it assumes no ties - the real archive has heavy ties in the
    deterministic score, which costs additional power.
    """
    if not 0 < abs(rho) < 1:
        return -1
    nd = statistics.NormalDist()
    z_alpha = nd.inv_cdf(1 - (alpha if one_sided else alpha / 2))
    z_power = nd.inv_cdf(power)
    z_r = math.atanh(abs(rho))
    return int(math.ceil(1.06 * ((z_alpha + z_power) / z_r) ** 2 + 3))


# --------------------------------------------------------------------------
# Archive + outcome loading
# --------------------------------------------------------------------------


@dataclass
class Row:
    run: str
    spec_dir: str
    canonical: bool
    dataset: str | None
    task_type: str | None
    timestamp: str | None
    target: str | None = None
    # deterministic terms
    feasibility_verdict: str = ""
    feasibility_penalty: float = 0.0
    feasibility_warns: list[str] = field(default_factory=list)
    venue_fit_default: float = 0.0
    venue_fit_codes: list[str] = field(default_factory=list)
    venue_fit_rule_deltas: dict[str, float] = field(default_factory=dict)
    venue_fit_realized: float | None = None
    deterministic_score: float = 0.0
    # realized outcome
    gate_samples: list[float] = field(default_factory=list)
    gate_cycles: list[str] = field(default_factory=list)
    gate_n_samples: int = 0
    gate_score: float | None = None
    gate_summary_final_score: float | None = None
    gate_venue: str | None = None
    in_ledger: bool = False
    ledger_score: float | None = None
    ledger_n_samples: int | None = None
    # population bookkeeping
    included: bool = False
    exclusion_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _resolve_context(spec_dir: Path, spec: dict) -> tuple[str | None, str | None, str | None]:
    """dataset, task_type, timestamp - falling back to the run checkpoint.

    6 of the 26 canonical prediction specs declare neither dataset nor
    task_type; ``checkpoint.json`` carries both plus the run timestamp.
    """
    dataset = spec.get("dataset")
    task_type = spec.get("task_type")
    timestamp = None
    checkpoint = spec_dir / "checkpoint.json"
    if checkpoint.exists():
        try:
            with open(checkpoint, encoding="utf-8") as f:
                data = json.load(f)
            dataset = dataset or data.get("dataset_name")
            task_type = task_type or data.get("task_type")
            timestamp = data.get("timestamp")
        except (OSError, ValueError):
            pass
    return dataset, task_type, timestamp


@dataclass
class GateOutcome:
    samples: list[float] = field(default_factory=list)
    cycles: list[str] = field(default_factory=list)
    final_cycle: int | None = None
    venue: str | None = None
    summary_final_score: float | None = None

    @property
    def score(self) -> float | None:
        if self.samples:
            return statistics.median(self.samples)
        return self.summary_final_score


def _read_json(path: Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def recover_gate_outcome(run_dir: Path) -> GateOutcome:
    """Recover the realized LSAR outcome for a run, straight from disk.

    Two cycle numberings are in play and conflating them would invent
    median-of-3 populations that never existed:

    * REVISION cycles are 1, 2, 3 ... - each is a DIFFERENT manuscript.
      The gate's verdict is the LAST one (``gate_summary.cycles_used``).
    * SAMPLE cycles are ``cycle*100 + extra`` (``review_gate.py``
      ``_maybe_median_sample``) - repeated reviews of the SAME manuscript.

    So the samples for a run are ``cycle_C`` plus ``cycle_{C*100+k}``,
    where C is the final revision cycle.
    """
    lsar = run_dir / "output" / "lsar_review"
    out = GateOutcome()
    if not lsar.is_dir():
        return out

    summary = _read_json(lsar / "gate_summary.json") or {}
    out.summary_final_score = summary.get("final_score")
    final_cycle = summary.get("cycles_used")

    present: dict[int, Path] = {}
    for entry in sorted(lsar.iterdir()):
        if not entry.is_dir() or not entry.name.startswith("cycle_"):
            continue
        try:
            present[int(entry.name.split("_", 1)[1])] = entry
        except ValueError:
            continue
    if not present:
        return out
    if not isinstance(final_cycle, int) or final_cycle not in present:
        revisions = [n for n in present if n < 100]
        final_cycle = max(revisions) if revisions else min(present)
    out.final_cycle = final_cycle

    wanted = [final_cycle] + sorted(
        n for n in present if n // 100 == final_cycle and n >= 100
    )
    for number in wanted:
        maybe_dir = present.get(number)
        if maybe_dir is None:
            continue
        cycle_dir: Path = maybe_dir
        data = _read_json(cycle_dir / "scores.json") or {}
        value = data.get("overall_score")
        if value is None:
            continue
        out.samples.append(float(value))
        out.cycles.append(cycle_dir.name)
        if out.venue is None:
            venue_data = _read_json(cycle_dir / "venue_classification.json") or {}
            out.venue = venue_data.get("venue")
    return out


def load_ledger(path: Path) -> dict[str, dict]:
    """Ledger keyed by ``run_id``.

    Deliberately NOT keyed by ``run_dir``: three ledger ``run_dir``
    values contain a literal vertical tab from an unescaped ``\\v4``
    (spec sec. 1.4), so that field is unusable as a join key.
    """
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {p["run_id"]: p for p in data.get("papers", []) if p.get("run_id")}


def build_rows(
    runs_dir: Path,
    ledger: dict[str, dict],
    *,
    registry_dir: str | None = None,
    raw_data_dir: str | None = None,
) -> list[Row]:
    rows: list[Row] = []
    for spec_path in sorted(runs_dir.glob("*/output*/research_spec.json")):
        try:
            with open(spec_path, encoding="utf-8") as f:
                spec = json.load(f)
        except (OSError, ValueError):
            continue
        if not isinstance(spec, dict):
            continue
        spec_dir = spec_path.parent
        run_dir = spec_dir.parent
        dataset, task_type, timestamp = _resolve_context(spec_dir, spec)

        report = screen(
            spec,
            candidate_id=run_dir.name,
            dataset=dataset,
            task_type=task_type,
            registry_dir=registry_dir,
            raw_data_dir=raw_data_dir,
            run_probes=False,
        )
        vf_default = score_venue_fit(spec)
        gate = recover_gate_outcome(run_dir)
        vf_realized = (
            score_venue_fit(spec, venue=gate.venue).score if gate.venue else None
        )
        ledger_entry = ledger.get(run_dir.name)
        median_sampling = (ledger_entry or {}).get("median_sampling") or {}

        rows.append(
            Row(
                run=run_dir.name,
                spec_dir=spec_dir.name,
                canonical=(spec_dir.name == "output"),
                dataset=dataset,
                task_type=task_type,
                timestamp=timestamp,
                target=resolved_target(spec),
                feasibility_verdict=report.verdict,
                feasibility_penalty=round(report.penalty, 4),
                feasibility_warns=report.warn_codes,
                venue_fit_default=vf_default.score,
                venue_fit_codes=vf_default.codes,
                # Per-rule deltas. Without these the external-only
                # diagnostic silently returns None and prints 'n/a' --
                # which is exactly what happened when it shipped: the
                # guardrail the rule table's own header calls
                # load-bearing was never running.
                venue_fit_rule_deltas={h.code: h.delta for h in vf_default.hits},
                venue_fit_realized=vf_realized,
                deterministic_score=round(
                    W_VENUE_FIT * vf_default.score - W_PENALTY * report.penalty, 6
                ),
                gate_samples=gate.samples,
                gate_cycles=gate.cycles,
                gate_n_samples=len(gate.samples),
                gate_score=gate.score,
                gate_summary_final_score=gate.summary_final_score,
                gate_venue=gate.venue,
                in_ledger=ledger_entry is not None,
                ledger_score=(ledger_entry or {}).get("lsar_overall"),
                ledger_n_samples=median_sampling.get("n_samples"),
            )
        )
    return rows


# --------------------------------------------------------------------------
# Population selection - explicit, by name, with reasons
# --------------------------------------------------------------------------


def assign_population(rows: Iterable[Row]) -> None:
    """Set ``included`` / ``exclusion_reasons`` for the PRIMARY population.

    Primary = archived canonical spec, present in the ledger, gate score
    recovered as a median of 3 samples, scored by the CALIBRATED (EDM)
    reviewer persona. Every other run records why it is out.
    """
    for row in rows:
        reasons: list[str] = []
        if not row.canonical:
            reasons.append(
                "aborted-attempt spec (runs/<run>/output_attempt*/): that "
                "manuscript never reached the gate"
            )
        if row.gate_n_samples == 0:
            reasons.append("no LSAR gate score on disk")
        elif row.gate_n_samples < 3:
            reasons.append(
                f"single-review gate score (n_samples={row.gate_n_samples}); "
                "not on a common footing with median-of-3 (spec sec. 6 V2)"
            )
        if not row.in_ledger:
            reasons.append("not recorded in evaluation/ledger.json")
        if row.gate_venue and row.gate_venue != CALIBRATED_VENUE:
            reasons.append(
                f"venue {row.gate_venue}: uncalibrated journal persona, "
                "scores labelled advisory in the ledger"
            )
        row.exclusion_reasons = reasons
        row.included = not reasons


def population(rows: Sequence[Row], predicate: Any) -> list[Row]:
    return [r for r in rows if predicate(r)]


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------


def analyse(
    rows: Sequence[Row], *, label: str, score_attr: str = "deterministic_score"
) -> dict:
    """Correlate one deterministic term against the realized gate score.

    Rows without a gate score are dropped here rather than silently
    coerced; ``n`` in the output is always the number actually used.
    """
    rows = [r for r in rows if r.gate_score is not None]
    x = [float(getattr(r, score_attr)) for r in rows]
    y = [float(r.gate_score) for r in rows if r.gate_score is not None]
    n = len(rows)
    out: dict[str, Any] = {
        "label": label,
        "score_term": score_attr,
        "n": n,
        "runs": [r.run for r in rows],
        "x": x,
        "y": y,
        "x_distinct_values": sorted(set(x)),
        "x_variance": (statistics.pvariance(x) if n > 1 else 0.0),
        "y_variance": (statistics.pvariance(y) if n > 1 else 0.0),
        "penalty_distinct_values": sorted({r.feasibility_penalty for r in rows}),
        "venue_fit_distinct_values": sorted({r.venue_fit_default for r in rows}),
    }
    if n < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        out.update(rho=float("nan"), permutation=None, bootstrap=None,
                   note="degenerate: fewer than 3 points, or no variance in one margin")
        return out
    perm = permutation_test(x, y)
    out["rho"] = perm.rho
    out["permutation"] = perm.to_dict()
    out["bootstrap"] = bootstrap_ci(x, y)
    out["required_n_to_detect_observed_rho"] = required_n_for_rho(perm.rho)
    out["required_n_to_detect_rho_0.5"] = required_n_for_rho(0.5)
    out["leave_one_out"] = leave_one_out(rows, score_attr=score_attr)
    out["duplicate_idea_groups"] = duplicate_idea_groups(rows)
    return out


def leave_one_out(
    rows: Sequence[Row], *, score_attr: str = "deterministic_score"
) -> list[dict]:
    """rho recomputed with each row dropped in turn.

    At n=5 a single observation can carry the whole correlation. This is
    the cheapest honest way to show whether it does.
    """
    out: list[dict] = []
    for i, dropped in enumerate(rows):
        kept = [r for j, r in enumerate(rows) if j != i]
        x = [float(getattr(r, score_attr)) for r in kept]
        y = [float(r.gate_score) for r in kept if r.gate_score is not None]
        rho = spearman(x, y) if len(kept) >= 3 else float("nan")
        out.append({"dropped": dropped.run, "n": len(kept), "rho": rho})
    return out


def resolved_target(spec: dict) -> str | None:
    """Task-type-agnostic target name (spec sec. 2.5).

    ``outcome_variable`` exists in only 6 of the 26 archived specs, so a
    resolver keyed on it alone silently no-ops for causal and
    psychometrics runs.
    """
    for key in ("outcome_variable", "scale_name", "treatment_variable"):
        value = spec.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for key in ("outcome", "treatment"):
        block = spec.get(key)
        if isinstance(block, dict):
            value = block.get("variable")
            if isinstance(value, str) and value.strip():
                return value.strip()
        elif isinstance(block, str) and block.strip():
            return block.strip()
    return None


def duplicate_idea_groups(rows: Sequence[Row]) -> list[dict]:
    """Rows that are the SAME idea executed more than once.

    Keyed on (dataset, task_type, resolved target). Two runs sharing a
    key are not independent observations of "does the ranker order ideas
    well" - they are one idea measured twice, and at this n that matters.
    """
    groups: dict[tuple, list[Row]] = {}
    for row in rows:
        key = (row.dataset, row.task_type, row.target)
        if key[2] is None:
            continue
        groups.setdefault(key, []).append(row)
    return [
        {"dataset": k[0], "task_type": k[1], "target": k[2],
         "runs": [r.run for r in v],
         "deterministic_scores": [r.deterministic_score for r in v],
         "gate_scores": [r.gate_score for r in v]}
        for k, v in sorted(groups.items(), key=lambda kv: str(kv[0]))
        if len(v) > 1
    ]


def check_pair(rows: Sequence[Row]) -> dict:
    by_run = {r.run: r for r in rows if r.canonical}
    low = by_run.get(PAIR_LOW)
    high = by_run.get(PAIR_HIGH)
    if low is None or high is None:
        return {"available": False, "separated": None,
                "note": f"missing {PAIR_LOW if low is None else PAIR_HIGH}"}
    separated = low.venue_fit_default < high.venue_fit_default
    return {
        "available": True,
        "separated": separated,
        "low": {
            "run": low.run, "venue_fit": low.venue_fit_default,
            "rules_fired": low.venue_fit_codes,
            "gate_score": low.gate_score, "gate_n_samples": low.gate_n_samples,
            "in_primary_population": low.included,
            "exclusion_reasons": low.exclusion_reasons,
        },
        "high": {
            "run": high.run, "venue_fit": high.venue_fit_default,
            "rules_fired": high.venue_fit_codes,
            "gate_score": high.gate_score, "gate_n_samples": high.gate_n_samples,
            "in_primary_population": high.included,
            "exclusion_reasons": high.exclusion_reasons,
        },
        "margin": high.venue_fit_default - low.venue_fit_default,
    }


def run_backtest(
    runs_dir: Path,
    ledger_path: Path,
    *,
    registry_dir: str | None = None,
    raw_data_dir: str | None = None,
    alpha: float = 0.05,
) -> dict:
    ledger = load_ledger(ledger_path)
    rows = build_rows(
        runs_dir, ledger, registry_dir=registry_dir, raw_data_dir=raw_data_dir
    )
    assign_population(rows)

    primary = [r for r in rows if r.included]
    with_journal = [
        r for r in rows
        if r.canonical and r.in_ledger and r.gate_n_samples >= 3
    ]
    all_median3 = [r for r in rows if r.canonical and r.gate_n_samples >= 3]
    all_gated = [r for r in rows if r.canonical and r.gate_n_samples >= 1]

    results = {
        "primary": analyse(primary, label="PRIMARY: ledger + median-of-3 + EDM"),
        "plus_journal": analyse(
            with_journal,
            label="SENSITIVITY: + JEDM advisory (uncalibrated persona)",
        ),
        "all_median3": analyse(
            all_median3,
            label="SENSITIVITY: + superseded first attempts not in the ledger",
        ),
        "all_gated": analyse(
            all_gated,
            label="CONTEXT ONLY: every gated spec incl. single reviews "
                  "across the 3b.23.7 provider switch",
        ),
        "primary_venue_fit_only": analyse(
            primary, label="PRIMARY, venue-fit term alone",
            score_attr="venue_fit_default",
        ),
    }

    # Ledger-vs-disk agreement. The ledger is a hand-maintained file; the
    # cycle scores are what the gate actually wrote. Also cross-check the
    # recovered median against gate_summary.json's own final_score.
    disagreements = []
    for row in rows:
        if row.gate_score is None:
            continue
        if (
            row.in_ledger
            and row.ledger_score is not None
            and abs(row.ledger_score - row.gate_score) > 1e-9
        ):
            disagreements.append(
                {"kind": "ledger_vs_disk", "run": row.run,
                 "ledger": row.ledger_score, "disk_median": row.gate_score,
                 "disk_samples": row.gate_samples, "cycles": row.gate_cycles}
            )
        if (
            row.gate_summary_final_score is not None
            and abs(row.gate_summary_final_score - row.gate_score) > 1e-9
        ):
            disagreements.append(
                {"kind": "gate_summary_vs_recovered_median", "run": row.run,
                 "gate_summary_final_score": row.gate_summary_final_score,
                 "recovered_median": row.gate_score,
                 "samples": row.gate_samples, "cycles": row.gate_cycles}
            )

    penalty_values = sorted({r.feasibility_penalty for r in primary})
    pair = check_pair(rows)
    primary_result = results["primary"]
    rho = primary_result.get("rho")
    perm = primary_result.get("permutation") or {}

    # ------------------------------------------------------------------
    # DECONFOUNDING DIAGNOSTICS (2026-07-11)
    #
    # The headline rho is not a verdict. Two checks decide whether it means
    # anything, both established empirically on this very data:
    #
    #   1. CIRCULARITY. venue_fit rules tagged provenance: in_sample cite our
    #      own gate scores in their evidence, so they cannot validate a ranker
    #      against those scores. Measured on n=24: full table +0.377,
    #      external-only +0.002, in-sample-only +0.349. All of the apparent
    #      signal came from the two rules written knowing the answer.
    #
    #   2. CONFOUNDING. Run recency ("the pipeline improved over five days")
    #      scored rho +0.90 on the primary population vs the ranker's +0.79,
    #      and the two are 0.95-collinear. A baseline with zero idea content
    #      ranking as well means this design cannot attribute the correlation
    #      to the ranker at all.
    # ------------------------------------------------------------------
    def _external_only_rho(pop: list) -> float | None:
        """rho using ONLY rules whose evidence is external to our runs."""
        try:
            from src.ideation.venue_fit import load_rules
            in_sample = {
                r.get("code") for r in (load_rules() or {}).get("rules", [])
                if r.get("provenance") == "in_sample"
            }
        except Exception:
            return None
        if not in_sample:
            return None
        xs, ys = [], []
        for r in pop:
            if r.gate_score is None:
                continue
            deltas = getattr(r, "venue_fit_rule_deltas", None)
            if isinstance(deltas, dict):
                ext = sum(v for k, v in deltas.items() if k not in in_sample)
            else:
                # No per-rule breakdown: subtract nothing we cannot attribute,
                # and report None rather than a number we cannot defend.
                return None
            xs.append(ext)
            ys.append(r.gate_score)
        return spearman(xs, ys) if len(xs) >= 3 else None

    def _recency_rho(pop: list) -> float | None:
        xs, ys = [], []
        for i, r in enumerate(sorted(pop, key=lambda z: (z.timestamp or ""))):
            if r.gate_score is None:
                continue
            xs.append(float(i))
            ys.append(r.gate_score)
        return spearman(xs, ys) if len(xs) >= 3 else None

    ext_rho = _external_only_rho(primary)
    rec_rho = _recency_rho(primary)
    # The primary population is too small to expose circularity: an
    # in-sample rule that fires on ALL of it is constant and cancels
    # out of the ranks. The contamination is only visible on the wider
    # population, where v1 measured full +0.377 vs external-only +0.002.
    ext_rho_all = _external_only_rho(all_gated)
    rec_rho_all = _recency_rho(all_gated)
    diagnostics = {
        "external_only_rho": ext_rho,
        "external_only_rho_all_gated": ext_rho_all,
        "recency_baseline_rho": rec_rho,
        "recency_baseline_rho_all_gated": rec_rho_all,
        "beaten_by_confound": bool(
            rec_rho is not None and rho is not None and rec_rho >= rho
        ),
        "note": (
            "external_only_rho drops venue-fit rules tagged provenance: "
            "in_sample. recency_baseline_rho ranks by run order alone and "
            "contains NO idea content; if it matches or beats the ranker, "
            "this backtest cannot attribute the correlation to the ranker."
        ),
    }

    # Deconfounding inputs (2026-07-11): the headline rho alone is not a
    # verdict. `external_rho` drops the rules whose evidence cites our own
    # gate outcomes; `beaten_by_confound` is True when a baseline with no
    # idea content ranks at least as well.
    # Judge circularity on the wider population (see note above).
    external_rho = (
        (diagnostics or {}).get("external_only_rho_all_gated")
        if (diagnostics or {}).get("external_only_rho_all_gated") is not None
        else (diagnostics or {}).get("external_only_rho")
    )
    beaten_by_confound = bool((diagnostics or {}).get("beaten_by_confound"))
    verdict = _verdict(
        rho, perm, pair, alpha=alpha, n=primary_result["n"],
        external_rho=external_rho, beaten_by_confound=beaten_by_confound,
    )

    return {
        "runs_dir": str(runs_dir),
        "ledger_path": str(ledger_path),
        "n_specs_scanned": len(rows),
        "n_canonical": sum(1 for r in rows if r.canonical),
        "rows": [r.to_dict() for r in rows],
        "results": results,
        "pair": pair,
        "ledger_disk_disagreements": disagreements,
        "primary_penalty_distinct_values": penalty_values,
        "weights": {"venue_fit": W_VENUE_FIT, "penalty": W_PENALTY},
        "alpha": alpha,
        "diagnostics": diagnostics,
        "verdict": verdict,
    }


def _verdict(
    rho: float | None, perm: dict, pair: dict, *, alpha: float, n: int,
    external_rho: float | None = None, beaten_by_confound: bool = False,
) -> dict:
    """The falsification rules from the task, applied mechanically."""
    if rho is None or (isinstance(rho, float) and math.isnan(rho)):
        return {
            "direction": "UNDEFINED", "pair": pair.get("separated"),
            "power": "UNDEFINED", "overall": "NOT_COMPUTABLE",
            "recommendation": "The primary population is degenerate; no rho "
                              "exists. Ship the feasibility screen alone.",
            "exit_code": 2,
        }
    direction = "POSITIVE" if rho > 0 else ("NULL" if rho == 0 else "INVERTED")
    separated = pair.get("separated")
    p = perm.get("p_greater", float("nan"))
    floor = perm.get("min_attainable_p_greater", float("nan"))
    if not math.isnan(p) and p <= alpha:
        power = "SUFFICIENT"
    elif not math.isnan(floor) and floor > alpha:
        power = "IMPOSSIBLE_AT_THIS_N"
    else:
        power = "UNDERPOWERED"

    if direction != "POSITIVE":
        overall = "FALSIFIED_INVERTED"
        rec = ("rho <= 0. The deterministic ranker is INVERTED. Ship the "
               "feasibility screen ALONE and do not build the judged layer "
               "on top of this ordering. Do not reweight the rules to "
               "rescue it.")
        code = 1
    elif separated is False:
        overall = "FALSIFIED_PAIR"
        rec = ("The pre-registered pair is not separated: the rule table is "
               "not encoding what it claims to. Fix or shelve the venue-fit "
               "term; ship the feasibility screen alone.")
        code = 1
    elif power == "SUFFICIENT":
        overall = "USABLE"
        rec = ("rho > 0, the pre-registered pair is ordered correctly, and "
               "the ordering is distinguishable from chance at this n.")
        code = 0
    elif external_rho is not None and abs(external_rho) < 0.10:
        # The rules whose evidence cites OUR OWN gate outcomes cannot be
        # used to validate a ranker against those outcomes. When the
        # out-of-sample partition measures ~0, the headline rho is an
        # artifact of circularity no matter how large it looks.
        overall = "NULL_CIRCULAR"
        rec = ("The headline rho comes from in-sample rules (evidence citing "
               "our own gate scores). External rules alone measure "
               f"rho = {external_rho:+.3f}. This is a measurement of ZERO, "
               "not a power problem, and it does not improve with n. "
               "Advisory only. To fix it, re-derive the rule table from the "
               "anchor corpus WITHOUT consulting our run outcomes.")
        code = 0
    elif beaten_by_confound:
        overall = "NULL_CONFOUNDED"
        rec = ("A baseline with zero idea content ranks at least as well and "
               "is collinear with the ranker, so this backtest cannot "
               "attribute the correlation to the ranker. Advisory only.")
        code = 0
    else:
        overall = "POSITIVE_BUT_UNDERPOWERED"
        rec = ("rho > 0 and the pre-registered pair is ordered correctly, "
               "but n is too small to distinguish this ordering from "
               "chance. Advisory use only: print the deterministic ranking "
               "beside any judged ranking, and do not let it select a live "
               "spec on its own until n grows.")
        code = 0
    return {
        "direction": direction, "pair_separated": separated, "power": power,
        "overall": overall, "recommendation": rec, "exit_code": code,
        "n_primary": n,
    }


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _fmt(value: Any, spec: str = ".3f") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    if isinstance(value, float):
        return format(value, spec)
    return str(value)


def render(result: dict) -> str:
    lines: list[str] = []
    add = lines.append
    add("=" * 78)
    add("Arc T V2 backtest - DETERMINISTIC terms only (no LLM, no network)")
    add("=" * 78)
    add("")
    add(f"Specs scanned: {result['n_specs_scanned']} "
        f"({result['n_canonical']} canonical runs/<run>/output/)")
    add(f"Deterministic score = {result['weights']['venue_fit']} * venue_fit "
        f"- {result['weights']['penalty']} * feasibility_penalty  (spec sec. 4.1)")
    add("")

    # ---- population -----------------------------------------------------
    rows = [Row(**r) for r in result["rows"]]
    included = [r for r in rows if r.included]
    add("-" * 78)
    add("POPULATION - INCLUDED (primary)")
    add("-" * 78)
    for row in sorted(included, key=lambda r: r.run):
        add(f"  {row.run:44} {str(row.task_type):14} "
            f"gate={_fmt(row.gate_score, '.2f')} "
            f"samples={row.gate_samples} {row.gate_cycles} venue={row.gate_venue}")
        add(f"      venue_fit={row.venue_fit_default:+.2f} "
            f"{row.venue_fit_codes}  feasibility={row.feasibility_verdict} "
            f"penalty={row.feasibility_penalty:.2f}")
    add("")
    add("-" * 78)
    add("POPULATION - EXCLUDED, by name and reason")
    add("-" * 78)
    for row in sorted((r for r in rows if not r.included), key=lambda r: r.run):
        add(f"  {row.run}/{row.spec_dir}")
        for reason in row.exclusion_reasons:
            add(f"      - {reason}")
    add("")

    if result["ledger_disk_disagreements"]:
        add("!! ledger vs on-disk gate score disagreements:")
        for row in result["ledger_disk_disagreements"]:
            add(f"   {row}")
        add("")
    else:
        add("Ledger lsar_overall agrees with the on-disk median for every "
            "joined run.")
        add("")

    # ---- correlation ----------------------------------------------------
    add("-" * 78)
    add("SPEARMAN rho  (deterministic score vs realized gate score)")
    add("-" * 78)
    for key in ("primary", "primary_venue_fit_only", "plus_journal",
                "all_median3", "all_gated"):
        res = result["results"][key]
        add(f"  {res['label']}")
        add(f"      n = {res['n']}   runs = {res['runs']}")
        if res.get("permutation") is None:
            add(f"      rho = n/a  ({res.get('note', 'not computed')})")
            add("")
            continue
        perm = res["permutation"]
        boot = res["bootstrap"]
        add(f"      rho = {res['rho']:+.4f}")
        add(f"      permutation ({perm['method']}, {perm['n_permutations']} "
            f"permutations): p(one-sided, rho>0) = {perm['p_greater']:.4f}, "
            f"p(two-sided) = {perm['p_two_sided']:.4f}")
        add(f"      smallest one-sided p this n can produce = "
            f"{perm['min_attainable_p_greater']:.4f}")
        add(f"      bootstrap 95% CI = [{_fmt(boot['lo'])}, {_fmt(boot['hi'])}] "
            f"({boot['n_degenerate']}/{boot['n_boot']} resamples degenerate - "
            f"treat as decoration at this n)")
        add(f"      distinct deterministic scores: {res['x_distinct_values']}")
        add(f"      distinct venue_fit: {res['venue_fit_distinct_values']}   "
            f"distinct feasibility penalties: {res['penalty_distinct_values']}")
        add("")

    primary = result["results"]["primary"]
    if primary.get("permutation"):
        add("  PRIMARY leave-one-out rho (does one paper carry the whole "
            "correlation?):")
        for item in primary["leave_one_out"]:
            add(f"      drop {item['dropped']:44} n={item['n']} "
                f"rho={_fmt(item['rho'], '+.4f')}")
        add("")
        if primary["duplicate_idea_groups"]:
            add("  !! PRIMARY contains the SAME IDEA executed more than once "
                "- these are")
            add("     not independent observations of idea quality:")
            for group in primary["duplicate_idea_groups"]:
                add(f"      {group['dataset']} / {group['task_type']} / "
                    f"target {group['target']}")
                add(f"        runs = {group['runs']}")
                add(f"        deterministic = {group['deterministic_scores']}  "
                    f"gate = {group['gate_scores']}")
            n_distinct = primary["n"] - sum(
                len(g["runs"]) - 1 for g in primary["duplicate_idea_groups"]
            )
            add(f"      -> distinct IDEAS in the primary population: "
                f"{n_distinct} (rows: {primary['n']})")
            add("")
        add("  Fisher-z planning numbers (APPROXIMATE, assumes no ties):")
        add(f"      to detect the observed rho={primary['rho']:+.3f} at "
            f"alpha=0.05 one-sided, 80% power: n ~ "
            f"{primary['required_n_to_detect_observed_rho']}")
        add(f"      to detect a true rho=0.50 on the same terms: n ~ "
            f"{primary['required_n_to_detect_rho_0.5']}")
        add("")

    add(f"  Feasibility penalty values in the primary population: "
        f"{result['primary_penalty_distinct_values']}")
    if len(result["primary_penalty_distinct_values"]) <= 1:
        add("      -> ZERO VARIANCE. The feasibility term contributes nothing")
        add("         to this ordering; the composite is a positive affine")
        add("         transform of venue_fit and carries the same ranks. Only")
        add("         the venue-fit term is actually being tested here.")
    add("")

    # ---- pre-registered pair -------------------------------------------
    add("-" * 78)
    add("PRE-REGISTERED PAIR (spec sec. 6 V2)")
    add("-" * 78)
    pair = result["pair"]
    if not pair.get("available"):
        add(f"  NOT AVAILABLE: {pair.get('note')}")
    else:
        for side in ("low", "high"):
            item = pair[side]
            add(f"  {item['run']}")
            add(f"      venue_fit = {item['venue_fit']:+.2f}  rules fired: "
                f"{item['rules_fired'] or '(none)'}")
            add(f"      realized gate = {_fmt(item['gate_score'], '.2f')} "
                f"from {item['gate_n_samples']} sample(s); "
                f"in primary population: {item['in_primary_population']}")
            for reason in item["exclusion_reasons"]:
                add(f"        - excluded: {reason}")
        add(f"  SEPARATED: {pair['separated']}  (margin "
            f"{pair['margin']:+.2f} in favour of {PAIR_HIGH})")
    add("")

    # ---- verdict --------------------------------------------------------
    verdict = result["verdict"]
    add("=" * 78)
    diag = result.get("diagnostics") or {}
    if diag:
        add("")
        add("=" * 78)
        add("DECONFOUNDING DIAGNOSTICS (read these before the verdict)")
        add("=" * 78)
        add(f"  external-only rho, primary (n small)     : "
            f"{_fmt(diag.get('external_only_rho'))}")
        add(f"  external-only rho, ALL GATED  <-- the one : "
            f"{_fmt(diag.get('external_only_rho_all_gated'))}")
        add(f"  recency baseline, all gated              : "
            f"{_fmt(diag.get('recency_baseline_rho_all_gated'))}")
        add(f"  run-recency baseline rho (no idea content): "
            f"{_fmt(diag.get('recency_baseline_rho'))}")
        add(f"  ranker beaten by that baseline           : "
            f"{diag.get('beaten_by_confound')}")
        import textwrap
        for line in textwrap.wrap(str(diag.get("note", "")), 74):
            add("  " + line)
    add("")
    add(f"VERDICT: {verdict['overall']}")
    add("=" * 78)
    add(f"  direction      : {verdict['direction']}")
    add(f"  pair separated : {verdict['pair_separated']}")
    add(f"  power          : {verdict['power']}")
    add(f"  n (primary)    : {verdict['n_primary']}")
    add("")
    add("  " + verdict["recommendation"])
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default=str(REPO_ROOT / "runs"), dest="runs_dir")
    parser.add_argument(
        "--ledger", default=str(REPO_ROOT / "evaluation" / "ledger.json"),
    )
    parser.add_argument("--registry-dir", default=None, dest="registry_dir")
    parser.add_argument("--raw-data-dir", default=None, dest="raw_data_dir")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--json", default=None, help="write the full result as JSON")
    args = parser.parse_args(argv)

    result = run_backtest(
        Path(args.runs_dir),
        Path(args.ledger),
        registry_dir=args.registry_dir,
        raw_data_dir=args.raw_data_dir,
        alpha=args.alpha,
    )
    print(render(result))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1)
    return int(result["verdict"]["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
