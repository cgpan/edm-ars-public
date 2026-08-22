"""Arc T / T1b - the cascade: screen -> venue fit -> prior art -> judge -> BT.

    feasibility screen (T0)  ->  venue fit (T0)  ->  prior-art veto
        ->  pairwise judged matches  ->  Bradley-Terry ranking

ADVISORY MODE IS THE CURRENT HONEST STATE
-----------------------------------------
The V2 rank-inversion backtest (spec sec. 6) has **not** cleared. Its
headline number looked like a pass and the deconfounded finding is null,
so this module refuses to present itself as a validated selector:

* every ranking artifact carries BOTH the judged ordering and the
  deterministic-only ordering, side by side;
* every artifact carries :data:`V2_STATUS`, with the measured numbers
  and their n;
* ``ranking.json`` carries ``authorized_for_live_selection: false``, and
  :func:`run_cascade` raises if a caller passes
  ``allow_live_selection=True`` while V2 is uncleared.

Advisory is not a flag someone flips later once the code looks finished.
It flips when a re-derived, out-of-sample rule table measures something
above zero on a population with real spread. Until then a caller that
wants a spec for a live run should take the deterministic ordering and
know that is what it took.

C3: the judged layer is REMOVABLE
---------------------------------
With every judged verdict shuffled or deleted, this module still emits a
complete, deterministic ordering over the same candidates -
:func:`deterministic_ordering`, which reads only the feasibility report,
the venue-fit rule table, the prior-art verdict and the candidate id.
:func:`shuffle_control` measures how far the judged ordering actually
moves off that baseline, against the distribution produced by randomised
verdicts. If the judged displacement sits inside the shuffled
distribution, the judged layer is not contributing and spec sec. 6 V4
says to delete it.

Nothing in this module calls a model directly: judging goes through
``src.ideation.judge``, which goes through ``BaseAgent.call_llm``.
"""
from __future__ import annotations

import json
import os
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from src.ideation import bradley_terry as _bt
from src.ideation import cards as _cards
from src.ideation import feasibility as _feas
from src.ideation import judge as _judge
from src.ideation import venue_fit as _vf
from src.ideation import venue_router as _router

__all__ = [
    "Candidate",
    "TournamentResult",
    "V2_STATUS",
    "build_pairs",
    "deterministic_ordering",
    "format_ranking",
    "load_candidate_records",
    "prior_art_stage",
    "round_robin",
    "routing_stage",
    "run_cascade",
    "shuffle_control",
    "swiss_round",
]

SCHEMA_VERSION = "1.0"

# Spec sec. 4.1 defaults; mirrored in scripts/run_idea_tournament.py for
# the T1a deterministic-only ranking. Chosen so the deterministic prior
# can move a candidate about one rank but cannot override a unanimous
# judged sweep.
DEFAULT_WEIGHT_VENUE_FIT = 0.30
DEFAULT_WEIGHT_PENALTY = 0.20
DEFAULT_PRIOR_SD = 1.0

DEFAULT_MAX_SURVIVORS = 12
DEFAULT_ROUND_ROBIN_MAX = 7
DEFAULT_SWISS_ROUNDS = 5
DEFAULT_SEED = 42
DEFAULT_TOP_K = 2
DEFAULT_SHUFFLE_REPLICATES = 40


# --------------------------------------------------------------------------
# The V2 verdict, carried in every artifact
# --------------------------------------------------------------------------

#: Read off `the v5 ranker backtest verdict (internal)` (2026-07-11), which
#: is the adversarial verification of ``scripts/backtest_ranker.py``.
#: Every number here is a measurement with its n attached. None of them
#: is an estimate.
V2_STATUS: dict[str, Any] = {
    "cleared": False,
    "verdict": "NULL once deconfounded (not inverted, not positive)",
    "gate_text": (
        "spec sec. 9 T1b: 'V2 backtest must return rho > 0 on the 12 ledger "
        "papers, and must correctly order the phase_b_did v1/v2 pair, before "
        "the tournament is permitted to select a spec for a live run.'"
    ),
    "why_not_cleared": (
        "The gate is technically satisfied by the headline number and is "
        "not satisfied in substance: on the only population with real "
        "spread the externally-mined half of the venue-fit rule table "
        "measures zero, and on the primary population a predictor with no "
        "idea content in it at all outranks the ranker."
    ),
    "measured": [
        {
            "quantity": "headline Spearman rho, deterministic ranker vs realized gate score",
            "value": 0.7906,
            "n": 5,
            "note": (
                "primary population = ledger + median-of-3 + EDM-calibrated. "
                "Exact one-sided permutation p = 0.0667; the smallest p this "
                "n can produce is 0.0333, so it cannot reach 0.05."
            ),
        },
        {
            "quantity": "rho, OUT-OF-SAMPLE venue-fit rules only (VF-02/03/05/06/07)",
            "value": 0.0018,
            "n": 24,
            "note": "p = 0.501. This is a measurement of zero, not a power problem.",
        },
        {
            "quantity": "rho, IN-SAMPLE venue-fit rules only (VF-01, VF-04)",
            "value": 0.3486,
            "n": 24,
            "note": (
                "p = 0.049. Both rules were authored from the outcomes being "
                "tested: VF-01's evidence cites our own 3.7 Reject, VF-04's "
                "cites the 3.7 -> 7.0 recovery. All of the signal is here."
            ),
        },
        {
            "quantity": "rho, run-recency baseline (checkpoint timestamp, zero idea content)",
            "value": 0.9000,
            "n": 5,
            "note": (
                "one-sided p = 0.0417 - the only predictor on the primary "
                "population that reaches alpha = 0.05, and it beats the ranker."
            ),
        },
        {
            "quantity": "collinearity, ranker vs run recency",
            "value": 0.9487,
            "n": 5,
            "note": "+0.7976 on n=24; the backtest cannot attribute the correlation to the ranker.",
        },
        {
            "quantity": "partial rho (ranker, gate | recency)",
            "value": -0.4588,
            "n": 5,
            "note": "+0.1691, p=0.214 on n=24. At n=5 this reflects collinearity, not harm.",
        },
    ],
    "pre_registered_pair": {
        "separated": True,
        "note": (
            "phase_b_did_20260704 (3.7) below stream1_did_v2_20260708 (7.0), "
            "margin +3.50 - but both of its rules are in-sample, so this is "
            "a consistency check on the implementation, not evidence."
        ),
    },
    "consequence": (
        "The tournament runs in ADVISORY mode: it prints and persists both "
        "its judged ranking and the deterministic-only ranking, and it "
        "declines to authorize a live selection."
    ),
    "how_this_flips": (
        "Re-derive the venue-fit rule table BLIND (rules mined without "
        "seeing our own gate scores), re-run scripts/backtest_ranker.py "
        "with the in-sample/out-of-sample partition and the recency "
        "baseline printed beside the ranker, and require the out-of-sample "
        "half to measure above zero on a population with spread."
    ),
    "source": "the v5 ranker backtest verdict (internal) (2026-07-11)",
}


class LiveSelectionNotAuthorized(RuntimeError):
    """Raised when a caller asks this tournament to pick a spec for a live run."""


# --------------------------------------------------------------------------
# Candidates
# --------------------------------------------------------------------------


@dataclass
class Candidate:
    """One tournament entrant: the card, the compiled spec, the screens.

    Constructed from a ``candidates.jsonl`` row written by the T1a
    generate stage, or from a bare ``{"card": ..., "spec": ...}``.
    """

    candidate_id: str
    card: dict = field(default_factory=dict)
    spec: dict = field(default_factory=dict)
    feasibility: dict = field(default_factory=dict)
    venue_fit: dict = field(default_factory=dict)
    routing: dict = field(default_factory=dict)
    priorart: dict | None = None
    seam_check: dict | None = None
    notes: list[str] = field(default_factory=list)

    @classmethod
    def from_record(cls, record: dict) -> "Candidate":
        card = record.get("card") or {}
        spec = record.get("spec") or {}
        cid = str(
            record.get("candidate_id")
            or card.get("candidate_id")
            or spec.get("task_id")
            or ""
        )
        return cls(
            candidate_id=cid,
            card=dict(card),
            spec=dict(spec),
            feasibility=dict(record.get("feasibility") or {}),
            venue_fit=dict(record.get("venue_fit") or {}),
            routing=dict(record.get("routing") or {}),
            priorart=(dict(record["priorart"]) if record.get("priorart") else None),
            seam_check=(
                dict(record["seam_check"]) if record.get("seam_check") else None
            ),
            notes=[str(n) for n in (record.get("notes") or [])],
        )

    # --- deterministic facts -------------------------------------------
    @property
    def cell(self) -> dict:
        cell = self.card.get("cell")
        return dict(cell) if isinstance(cell, dict) else {}

    @property
    def dataset(self) -> str | None:
        return self.cell.get("dataset") or self.spec.get("dataset")

    @property
    def task_type(self) -> str | None:
        return self.cell.get("task_type") or self.spec.get("task_type")

    @property
    def opportunity_pattern(self) -> str | None:
        return self.cell.get("opportunity_pattern")

    @property
    def outcome_family(self) -> str | None:
        target = self.card.get("resolved_target")
        if isinstance(target, str) and target.strip():
            return target.strip()
        return _cards.resolve_target(self.spec)

    @property
    def penalty(self) -> float:
        try:
            return float(self.feasibility.get("penalty") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @property
    def venue_fit_score(self) -> float:
        try:
            return float(self.venue_fit.get("score") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @property
    def warn_codes(self) -> list[str]:
        return [
            str(c.get("code"))
            for c in (self.feasibility.get("checks") or [])
            if isinstance(c, dict) and c.get("status") == _feas.WARN
        ]

    @property
    def verdict(self) -> str | None:
        value = self.feasibility.get("verdict")
        return str(value) if value else None

    @property
    def priorart_verdict(self) -> str:
        if not isinstance(self.priorart, dict):
            return "NOT_RUN"
        return str(self.priorart.get("verdict") or "NOT_RUN")

    @property
    def spec_loads(self) -> bool | None:
        seam = self.seam_check or {}
        if not seam.get("checked"):
            return None
        return bool(seam.get("passed"))

    # --- H2 Option C venue routing -------------------------------------
    @property
    def family(self) -> str | None:
        value = self.routing.get("family")
        return str(value) if value else None

    @property
    def route_venue(self) -> str | None:
        value = self.routing.get("venue")
        return str(value) if value else None

    @property
    def gate_status(self) -> str | None:
        value = self.routing.get("gate_status")
        return str(value) if value else None

    def routing_stamp(self) -> dict:
        """The family/venue stamp carried into every ranking row."""
        return {
            "family": self.family,
            "venue": self.route_venue,
            "gate_status": self.gate_status,
            "rule": self.routing.get("rule"),
        }

    def deterministic_score(
        self,
        weight_vf: float = DEFAULT_WEIGHT_VENUE_FIT,
        weight_pen: float = DEFAULT_WEIGHT_PENALTY,
    ) -> float:
        return round(
            weight_vf * self.venue_fit_score - weight_pen * self.penalty, 6
        )

    def evidence(self) -> dict:
        """C2: every term names the artifact fact behind it."""
        return {
            "venue_routing": (
                [
                    f"{self.routing.get('rule')}: {self.routing.get('evidence')}"
                ]
                + [
                    f"{s.get('code')} as {s.get('role')}: {s.get('why')} "
                    f"[evidence: {s.get('evidence')}]"
                    for s in (self.routing.get("signals") or [])
                ]
                if self.routing
                else ["routing stage did not run for this candidate"]
            ),
            "venue_fit": [
                f"{hit.get('code')} {float(hit.get('delta', 0.0)):+.2f}: "
                f"{hit.get('why')} [anchor evidence: {hit.get('evidence')}]"
                for hit in (self.venue_fit.get("hits") or [])
            ],
            "feasibility_penalty": [
                f"{c.get('code')} +{float(c.get('penalty') or 0.0):.2f}: "
                f"{c.get('message')} [read: {c.get('evidence')}]"
                for c in (self.feasibility.get("checks") or [])
                if c.get("status") == _feas.WARN
            ],
            "prior_art": (
                [
                    f"{self.priorart_verdict}: "
                    f"{(self.priorart or {}).get('delta_sentence') or 'no delta sentence'}"
                ]
                if isinstance(self.priorart, dict)
                else ["prior-art stage did not run for this candidate"]
            ),
            "seam": (
                []
                if self.spec_loads is None
                else [
                    "compiled spec loads through "
                    f"{(self.seam_check or {}).get('loader')}"
                    if self.spec_loads
                    else "compiled spec REJECTED by "
                    f"{(self.seam_check or {}).get('loader')}: "
                    f"{(self.seam_check or {}).get('error')}"
                ]
            ),
        }


def load_candidate_records(path: str | os.PathLike[str]) -> list[dict]:
    """Read a ``candidates.jsonl`` written by the T1a generate stage."""
    rows: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


# --------------------------------------------------------------------------
# Stage 0/1 - deterministic screen (T0)
# --------------------------------------------------------------------------


def screen_stage(
    records: Iterable[Any],
    *,
    registry_dir: str | os.PathLike[str] | None = None,
    raw_data_dir: str | os.PathLike[str] | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    use_column_cache: bool = True,
    run_probes: bool = False,
    rescreen: bool = False,
) -> tuple[list[Candidate], list[dict]]:
    """Attach (or reuse) a feasibility report; KILL verdicts drop out.

    Reuses the report already on a T1a survivor row by default: the T1a
    stage ran the identical shipped ``feasibility.screen`` over the
    identical compiled spec, so re-running it would burn the probe cost
    to reproduce a value byte-for-byte. ``rescreen=True`` forces it.
    """
    survivors: list[Candidate] = []
    killed: list[dict] = []
    for record in records:
        candidate = (
            record
            if isinstance(record, Candidate)
            else Candidate.from_record(record)
        )
        if rescreen or not candidate.feasibility:
            context = _feas.make_context(
                candidate.spec,
                dataset=candidate.dataset,
                task_type=candidate.task_type,
                registry_dir=registry_dir,
                raw_data_dir=raw_data_dir,
                cache_dir=cache_dir,
                card=candidate.card or None,
                use_column_cache=use_column_cache,
            )
            report = _feas.screen(
                candidate.spec,
                candidate_id=candidate.candidate_id,
                context=context,
                run_probes=run_probes,
            )
            candidate.feasibility = report.to_dict()
        if candidate.verdict == _feas.KILL:
            kills = [
                c
                for c in (candidate.feasibility.get("checks") or [])
                if c.get("status") == _feas.KILL
            ]
            killed.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "stage": "feasibility_screen",
                    "kill_code": (kills[0].get("code") if kills else "F-UNKNOWN"),
                    "kill_codes": [c.get("code") for c in kills],
                    "evidence": "; ".join(
                        f"{c.get('code')}: {c.get('message')} "
                        f"[read: {c.get('evidence')}]"
                        for c in kills
                    ),
                    "cell": candidate.cell,
                    "card": candidate.card,
                    "detail": {"spec": candidate.spec},
                }
            )
            continue
        survivors.append(candidate)
    return survivors, killed


def routing_stage(
    candidates: Sequence[Candidate],
    *,
    config: dict | None = None,
) -> list[Candidate]:
    """Attach the deterministic venue-routing verdict (H2 Option C).

    Always recomputed - the router is cheap, deterministic, and its
    output feeds the venue-fit routing hook, so a stale cached verdict
    from an older record must not survive into the artifacts.
    """
    for candidate in candidates:
        candidate.routing = _router.route_idea(
            candidate.spec, candidate.card or None, config=config
        )
    return list(candidates)


def venue_fit_stage(
    candidates: Sequence[Candidate],
    *,
    venue: str | None = None,
    rules_path: str | os.PathLike[str] | None = None,
    rescore: bool = False,
) -> list[Candidate]:
    """Attach (or reuse) the deterministic venue-fit report.

    A cached report on a POLICY-routed candidate is reused only when it
    was scored under policy-causal routing: a report scored without the
    H2 routing hook would keep VF-01/VF2-01/VF2-02 as penalties on a
    policy-routed idea, which is exactly what Option C removed. A cached
    report on a computational-routed candidate is reused as-is - the
    routing hook changes nothing at that destination, so the cache is
    not stale.
    """
    for candidate in candidates:
        routing_family = candidate.family
        cached_family = ((candidate.venue_fit or {}).get("facts") or {}).get(
            "routing_family"
        )
        stale_routing = (
            routing_family == _vf.FAMILY_POLICY and cached_family != routing_family
        )
        if rescore or not candidate.venue_fit or stale_routing:
            report = _vf.score_venue_fit(
                candidate.spec,
                venue=venue,
                card=candidate.card or None,
                rules_path=rules_path,
                routing_family=routing_family,
            )
            candidate.venue_fit = report.to_dict()
    return list(candidates)


# --------------------------------------------------------------------------
# Stage 2 - prior-art veto (parallel agent's module; optional at import)
# --------------------------------------------------------------------------


def _load_priorart() -> tuple[Any, str]:
    """``(module_or_None, why)``.

    ``src.ideation.priorart`` is owned by a parallel slice. This is the
    structured call site: when the module is absent the stage is SKIPPED
    and says so in the artifacts. It is never treated as if every
    candidate had been cleared - an unrun veto is not a clean veto, the
    same reason UNVERIFIABLE is not CLEAR (spec sec. 3 Stage 2, R6).
    """
    try:
        from src.ideation import priorart  # type: ignore[attr-defined]
    except Exception as exc:  # ImportError, or an error inside the module
        return None, f"{type(exc).__name__}: {exc}"
    checker = getattr(priorart, "collision_check", None)
    if not callable(checker):
        return None, (
            "src.ideation.priorart exists but exposes no callable "
            "collision_check(card); spec sec. 3 Stage 2 defines that entry point"
        )
    return priorart, ""


def prior_art_stage(
    candidates: Sequence[Candidate],
    *,
    enabled: bool = True,
    checker: Callable[..., dict] | None = None,
    on_event: Callable[[str], None] | None = None,
    **checker_kwargs: Any,
) -> tuple[list[Candidate], list[dict], dict]:
    """Run the collision veto. Returns ``(survivors, killed, status)``.

    Only the NEGATIVE claim acts (C1): a COLLISION with a cited nearest
    prior work kills; UNVERIFIABLE is recorded as a distinct third state
    that is not CLEAR and enters only as tie-break rule 4; a CLEAR
    verdict grants nothing, because "no paper does this" is an
    absence-of-evidence claim and is exactly the claim that measured
    r = -0.35 against the criterion here.
    """
    status: dict[str, Any] = {
        "ran": False,
        "reason": "",
        "verdicts": {},
        "note": (
            "A veto only. No positive novelty number is computed, stored or "
            "ranked on anywhere in this cascade (C1)."
        ),
    }
    if not enabled:
        status["reason"] = "disabled by the caller"
        return list(candidates), [], status

    if checker is None:
        module, why = _load_priorart()
        if module is None:
            status["reason"] = (
                "src.ideation.priorart is not available: " + why + ". The "
                "stage is SKIPPED; no candidate was cleared by it, and the "
                "ranking's tie-break rule 4 therefore has no signal."
            )
            if on_event:
                on_event("prior-art: SKIPPED (" + why + ")")
            return list(candidates), [], status
        checker = getattr(module, "collision_check")

    status["ran"] = True
    survivors: list[Candidate] = []
    killed: list[dict] = []
    for candidate in candidates:
        try:
            verdict = checker(candidate.card or candidate.spec, **checker_kwargs)
        except Exception as exc:
            candidate.priorart = {
                "verdict": "UNVERIFIABLE",
                "nearest": [],
                "delta_sentence": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
            candidate.notes.append(
                "prior-art check raised; treated as UNVERIFIABLE, never as "
                f"CLEAR: {type(exc).__name__}: {exc}"
            )
            survivors.append(candidate)
            status["verdicts"][candidate.candidate_id] = "UNVERIFIABLE"
            continue
        verdict = dict(verdict or {})
        candidate.priorart = verdict
        name = str(verdict.get("verdict") or "UNVERIFIABLE").upper()
        status["verdicts"][candidate.candidate_id] = name
        if name == "COLLISION":
            nearest = (verdict.get("nearest") or [{}])[0]
            killed.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "stage": "prior_art_veto",
                    "kill_code": "T-PRIORART-COLLISION",
                    "kill_codes": ["T-PRIORART-COLLISION"],
                    "evidence": (
                        "nearest prior work "
                        f"{nearest.get('paperId') or '?'} "
                        f"({nearest.get('year') or '?'}) "
                        f"{nearest.get('title') or '?'}: "
                        f"{nearest.get('snippet') or 'no snippet supplied'}"
                    ),
                    "cell": candidate.cell,
                    "card": candidate.card,
                    "detail": {"priorart": verdict, "spec": candidate.spec},
                }
            )
            continue
        survivors.append(candidate)
    return survivors, killed, status


# --------------------------------------------------------------------------
# Field selection and pairing
# --------------------------------------------------------------------------


def select_field(
    candidates: Sequence[Candidate],
    max_survivors: int = DEFAULT_MAX_SURVIVORS,
    *,
    weight_vf: float = DEFAULT_WEIGHT_VENUE_FIT,
    weight_pen: float = DEFAULT_WEIGHT_PENALTY,
) -> tuple[list[Candidate], list[dict]]:
    """Truncate to the tournament's input size. Recorded, never silent.

    Spec sec. 4.1: >12 survivors truncate to the 12 with the lowest
    feasibility penalty. Ties on penalty fall through to the
    deterministic score and then to the candidate id, so the cut is
    reproducible.
    """
    ordered = sorted(
        candidates,
        key=lambda c: (
            c.penalty,
            -c.deterministic_score(weight_vf, weight_pen),
            c.candidate_id,
        ),
    )
    if max_survivors <= 0 or len(ordered) <= max_survivors:
        return list(ordered), []
    field_ = ordered[:max_survivors]
    deferred = [
        {
            "candidate_id": c.candidate_id,
            "reason": (
                f"field truncated to {max_survivors} entrants (spec sec. 4.1); "
                f"cut by feasibility penalty {c.penalty:.2f} then deterministic "
                f"score {c.deterministic_score(weight_vf, weight_pen):+.4f}"
            ),
            "penalty": round(c.penalty, 4),
            "deterministic_score": c.deterministic_score(weight_vf, weight_pen),
        }
        for c in ordered[max_survivors:]
    ]
    return field_, deferred


def round_robin(ids: Sequence[str]) -> list[tuple[str, str]]:
    """Every unordered pair, in a stable order."""
    items = list(ids)
    return [
        (items[i], items[j])
        for i in range(len(items))
        for j in range(i + 1, len(items))
    ]


def swiss_round(
    ids: Sequence[str],
    points: dict[str, float] | None = None,
    *,
    played: set[frozenset[str]] | None = None,
) -> list[tuple[str, str]]:
    """One Swiss round's pairings, given current points.

    Deterministic: candidates are ordered by (points desc, seed index),
    then paired greedily with the nearest opponent not already played.
    A repeat pairing is allowed only when no fresh opponent remains, and
    an odd field gives exactly one bye per round.
    """
    seed_index = {cid: i for i, cid in enumerate(ids)}
    scores = points or {}
    ordered = sorted(ids, key=lambda cid: (-scores.get(cid, 0.0), seed_index[cid]))
    seen = played or set()
    pairs: list[tuple[str, str]] = []
    unpaired = list(ordered)
    while len(unpaired) >= 2:
        first = unpaired.pop(0)
        chosen_index = None
        for index, other in enumerate(unpaired):
            if frozenset((first, other)) not in seen:
                chosen_index = index
                break
        if chosen_index is None:
            chosen_index = 0
        other = unpaired.pop(chosen_index)
        pairs.append((first, other))
        seen.add(frozenset((first, other)))
    return pairs


def build_pairs(
    ids: Sequence[str],
    *,
    round_robin_max: int = DEFAULT_ROUND_ROBIN_MAX,
) -> tuple[str, list[tuple[str, str]]]:
    """``(mode, pairs)``. Round-robin at <= 7 entrants, Swiss above.

    Swiss pairings after round 1 depend on judged results, so this
    returns only round 1 for the Swiss mode; :func:`run_cascade` drives
    the remaining rounds.
    """
    items = list(ids)
    if len(items) <= 1:
        return "none", []
    if len(items) <= round_robin_max:
        return "round_robin", round_robin(items)
    return "swiss", swiss_round(items, None)


# --------------------------------------------------------------------------
# Orderings
# --------------------------------------------------------------------------

_PRIORART_RANK = {"CLEAR": 0, "UNVERIFIABLE": 1, "NOT_RUN": 2, "COLLISION": 3}

TIE_BREAK_RULES: tuple[str, ...] = (
    "0. the compiled spec loads through src.main.load_locked_research_spec "
    "(a spec the pipeline cannot consume sorts last; it is NOT killed)",
    "1. primary term (BT posterior mean for the judged ordering; the "
    "deterministic score venue_fit*w_vf - penalty*w_pen for the "
    "deterministic-only ordering)",
    "2. fewer WARN codes in the feasibility report",
    "3. higher deterministic venue_fit score",
    "4. prior-art verdict CLEAR > UNVERIFIABLE > NOT_RUN",
    "5. opportunity-pattern diversity vs the previous tournament's winner "
    "(prefer the pattern not used last time)",
    "6. lexicographic candidate_id (seeded, so reproducible)",
)


def _diversity_key(candidate: Candidate, previous_winner: dict | None) -> int:
    """0 when this candidate breaks the last winner's pattern, else 1."""
    if not previous_winner:
        return 0
    last_pattern = (previous_winner.get("cell") or {}).get("opportunity_pattern")
    if not last_pattern:
        return 0
    return 1 if candidate.opportunity_pattern == last_pattern else 0


def _order_rows(
    candidates: Sequence[Candidate],
    primary: dict[str, float],
    *,
    method: str,
    weight_vf: float,
    weight_pen: float,
    previous_winner: dict | None,
    extra: dict[str, dict] | None = None,
) -> list[dict]:
    rows: list[dict] = []
    for candidate in candidates:
        cid = candidate.candidate_id
        key = (
            0 if candidate.spec_loads is not False else 1,
            -float(primary.get(cid, 0.0)),
            len(candidate.warn_codes),
            -candidate.venue_fit_score,
            _PRIORART_RANK.get(candidate.priorart_verdict, 2),
            _diversity_key(candidate, previous_winner),
            cid,
        )
        row = {
            "candidate_id": cid,
            "method": method,
            "primary_term": round(float(primary.get(cid, 0.0)), 6),
            "spec_loads": candidate.spec_loads,
            "deterministic_score": candidate.deterministic_score(
                weight_vf, weight_pen
            ),
            "venue_fit_score": candidate.venue_fit_score,
            "feasibility_penalty": round(candidate.penalty, 4),
            "feasibility_verdict": candidate.verdict,
            "warn_codes": candidate.warn_codes,
            "prior_art_verdict": candidate.priorart_verdict,
            "analytic_n_estimate": candidate.feasibility.get("analytic_n_estimate"),
            "cell": candidate.cell,
            "outcome_family": candidate.outcome_family,
            "venue_routing": candidate.routing_stamp(),
            "tie_break_trace": [
                f"rule 0 spec_loads={candidate.spec_loads}",
                f"rule 1 {method}_primary={float(primary.get(cid, 0.0)):+.6f}",
                f"rule 2 n_warn={len(candidate.warn_codes)}",
                f"rule 3 venue_fit={candidate.venue_fit_score:+.2f}",
                f"rule 4 prior_art={candidate.priorart_verdict}",
                f"rule 5 breaks_previous_pattern="
                f"{_diversity_key(candidate, previous_winner) == 0}",
                f"rule 6 candidate_id={cid}",
            ],
            "evidence": candidate.evidence(),
            "_key": key,
        }
        if extra and cid in extra:
            row.update(extra[cid])
        rows.append(row)
    rows.sort(key=lambda r: r["_key"])
    for index, row in enumerate(rows, start=1):
        row["rank"] = index
        row.pop("_key", None)
    return rows


def deterministic_ordering(
    candidates: Sequence[Candidate],
    *,
    weight_vf: float = DEFAULT_WEIGHT_VENUE_FIT,
    weight_pen: float = DEFAULT_WEIGHT_PENALTY,
    previous_winner: dict | None = None,
) -> list[dict]:
    """C3: a complete ordering with every judged verdict removed.

    Reads only the feasibility report, the venue-fit rule table, the
    prior-art verdict and the candidate id. No LLM output enters at any
    weight, so this ordering exists whether or not the judge ran, and it
    is the fallback the spec's own falsification conditions 1-3 hand
    back to.
    """
    primary = {
        c.candidate_id: c.deterministic_score(weight_vf, weight_pen)
        for c in candidates
    }
    return _order_rows(
        candidates,
        primary,
        method="deterministic",
        weight_vf=weight_vf,
        weight_pen=weight_pen,
        previous_winner=previous_winner,
    )


def judged_ordering(
    candidates: Sequence[Candidate],
    posterior: _bt.BTPosterior,
    membership: dict[str, float] | None = None,
    *,
    weight_vf: float = DEFAULT_WEIGHT_VENUE_FIT,
    weight_pen: float = DEFAULT_WEIGHT_PENALTY,
    previous_winner: dict | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """Ordering on the BT posterior mean, with the same tie-breaks."""
    primary = {c.candidate_id: posterior.strength.get(c.candidate_id, 0.0)
               for c in candidates}
    extra = {
        c.candidate_id: {
            "bt_strength": round(posterior.strength.get(c.candidate_id, 0.0), 6),
            "bt_sd": round(posterior.sd.get(c.candidate_id, 0.0), 6),
            "bt_prior_mean": round(
                posterior.prior_means.get(c.candidate_id, 0.0), 6
            ),
            "n_matches": posterior.matches_per_candidate.get(c.candidate_id, 0),
            f"p_top{top_k}": round(
                float((membership or {}).get(c.candidate_id, 0.0)), 4
            ),
        }
        for c in candidates
    }
    return _order_rows(
        candidates,
        primary,
        method="judged_bt",
        weight_vf=weight_vf,
        weight_pen=weight_pen,
        previous_winner=previous_winner,
        extra=extra,
    )


def _spearman(order_a: Sequence[str], order_b: Sequence[str]) -> float | None:
    """Rank correlation between two orderings of the same ids (no ties)."""
    common = [cid for cid in order_a if cid in set(order_b)]
    n = len(common)
    if n < 3:
        return None
    rank_a = {cid: i for i, cid in enumerate(order_a)}
    rank_b = {cid: i for i, cid in enumerate(order_b)}
    mean = (n - 1) / 2.0
    num = sum((rank_a[c] - mean) * (rank_b[c] - mean) for c in common)
    den_a = sum((rank_a[c] - mean) ** 2 for c in common)
    den_b = sum((rank_b[c] - mean) ** 2 for c in common)
    if den_a <= 0 or den_b <= 0:
        return None
    return round(num / ((den_a * den_b) ** 0.5), 6)


def _displacement(order_a: Sequence[str], order_b: Sequence[str]) -> float | None:
    common = [cid for cid in order_a if cid in set(order_b)]
    if not common:
        return None
    rank_a = {cid: i for i, cid in enumerate(order_a)}
    rank_b = {cid: i for i, cid in enumerate(order_b)}
    return round(
        sum(abs(rank_a[c] - rank_b[c]) for c in common) / len(common), 6
    )


def shuffle_control(
    candidates: Sequence[Candidate],
    observations: Sequence[dict],
    *,
    judged_order: Sequence[str],
    deterministic_order: Sequence[str],
    prior_means: dict[str, float],
    prior_sd: float = DEFAULT_PRIOR_SD,
    replicates: int = DEFAULT_SHUFFLE_REPLICATES,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Spec sec. 6 V4 control: does the JUDGE contribute, or the prior?

    Method, stated exactly because it deviates from the spec's one-word
    description ("randomly permuted"): permuting winners ACROSS pairs
    would produce outcomes naming candidates that were not in the match,
    which is not a control, it is corrupt data. Instead each observation
    keeps its own pair and has its outcome redrawn - tie with the run's
    own observed tie rate, otherwise a fair coin between the two
    candidates actually in that pair. Idea content is destroyed; match
    structure, tie rate and the deterministic prior are preserved.

    Read it this way: if the judged ordering's displacement from the
    deterministic ordering sits INSIDE the shuffled displacement
    distribution, the judged layer moved the ranking no more than noise
    would have, and spec sec. 6 V4 says delete it.
    """
    rows = [r for r in observations if r.get("pair")]
    n_obs = len(rows)
    ids = [c.candidate_id for c in candidates]
    if n_obs == 0 or len(ids) < 3:
        return {
            "ran": False,
            "reason": (
                f"needs >= 3 candidates and >= 1 observation; had "
                f"{len(ids)} candidates and {n_obs} observations"
            ),
        }
    tie_rate = sum(1 for r in rows if r.get("winner") is None) / n_obs
    rng = random.Random(seed)
    spearmans: list[float] = []
    displacements: list[float] = []
    for _ in range(max(1, replicates)):
        shuffled: list[dict] = []
        for row in rows:
            pair = list(row["pair"])
            if rng.random() < tie_rate:
                winner = None
            else:
                winner = pair[0] if rng.random() < 0.5 else pair[1]
            shuffled.append({"pair": pair, "winner": winner, "source": "shuffle"})
        posterior = _bt.fit(
            shuffled,
            prior_means=prior_means,
            prior_sd=prior_sd,
            candidates=ids,
        )
        order = _bt.strength_order(posterior)
        rho = _spearman(order, list(deterministic_order))
        if rho is not None:
            spearmans.append(rho)
        disp = _displacement(order, list(deterministic_order))
        if disp is not None:
            displacements.append(disp)

    judged_rho = _spearman(list(judged_order), list(deterministic_order))
    judged_disp = _displacement(list(judged_order), list(deterministic_order))
    inside = None
    if judged_disp is not None and displacements:
        lo, hi = min(displacements), max(displacements)
        inside = bool(lo <= judged_disp <= hi)
    return {
        "ran": True,
        "method": (
            "outcomes redrawn within each observed pair, preserving pair "
            "structure and the run's tie rate; deterministic prior held fixed"
        ),
        "replicates": max(1, replicates),
        "seed": seed,
        "observed_tie_rate": round(tie_rate, 4),
        "judged_vs_deterministic_rho": judged_rho,
        "judged_vs_deterministic_mean_rank_shift": judged_disp,
        "shuffled_vs_deterministic_rho_median": (
            round(statistics.median(spearmans), 6) if spearmans else None
        ),
        "shuffled_mean_rank_shift_median": (
            round(statistics.median(displacements), 6) if displacements else None
        ),
        "shuffled_mean_rank_shift_range": (
            [round(min(displacements), 6), round(max(displacements), 6)]
            if displacements
            else None
        ),
        "judged_displacement_inside_shuffled_range": inside,
        "reading": (
            "judged_displacement_inside_shuffled_range == true means the "
            "judged layer moved the ranking no further off the deterministic "
            "baseline than randomised verdicts did; spec sec. 6 V4 then says "
            "delete the judged layer. This is a diagnostic on ONE tournament, "
            "not a test - it has no n and no p."
        ),
    }


# --------------------------------------------------------------------------
# The cascade
# --------------------------------------------------------------------------


@dataclass
class TournamentResult:
    tournament_id: str
    ranking: dict = field(default_factory=dict)
    matches: list[dict] = field(default_factory=list)
    killed: list[dict] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)
    posterior: Any = None
    summary: dict = field(default_factory=dict)

    def digest(self) -> str:
        return format_ranking(self.ranking)

    def write(self, out_dir: str | os.PathLike[str]) -> dict[str, str]:
        """Write ``matches.jsonl``, ``ranking.json``, ``tournament.md``.

        ``ranking.json`` carries no timestamp and no path, so two runs
        over the same match set produce byte-identical files.
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths: dict[str, str] = {}

        matches_path = out / "matches.jsonl"
        with open(matches_path, "w", encoding="utf-8") as handle:
            for row in self.matches:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
        paths["matches"] = str(matches_path)

        ranking_path = out / "ranking.json"
        with open(ranking_path, "w", encoding="utf-8") as handle:
            json.dump(self.ranking, handle, indent=1, ensure_ascii=False)
            handle.write("\n")
        paths["ranking"] = str(ranking_path)

        digest_path = out / "tournament.md"
        with open(digest_path, "w", encoding="utf-8") as handle:
            handle.write(self.digest())
            handle.write("\n")
        paths["digest"] = str(digest_path)

        if self.killed:
            paths["killed"] = _append_kills(out / "killed.jsonl", self.killed)
        return paths


def _append_kills(path: Path, rows: Sequence[dict]) -> str:
    """Append to ``killed.jsonl``, skipping rows already recorded.

    killed.jsonl is append-only training data by design (spec sec. 1.3),
    and it may already hold the generate stage's kills. Dedupe is on
    (candidate_id, stage, kill_code) so re-running the tournament over
    the same field does not double-count.
    """
    seen: set[tuple[str, str, str]] = set()
    if path.exists():
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                seen.add(
                    (
                        str(row.get("candidate_id")),
                        str(row.get("stage")),
                        str(row.get("kill_code")),
                    )
                )
    fresh = [
        row
        for row in rows
        if (
            str(row.get("candidate_id")),
            str(row.get("stage")),
            str(row.get("kill_code")),
        )
        not in seen
    ]
    if fresh:
        with open(path, "a", encoding="utf-8") as handle:
            for row in fresh:
                handle.write(json.dumps(row, ensure_ascii=False))
                handle.write("\n")
    return str(path)


def _cfg(config: dict | None, *keys: str, default: Any = None) -> Any:
    node: Any = config or {}
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node if node is not None else default


def run_cascade(
    records: Iterable[Any],
    *,
    tournament_id: str,
    config: dict | None = None,
    call_llm: _judge.LLMCaller | None = None,
    judge_model: str = "",
    generator_model: str = "",
    judged: bool = True,
    samples: int | None = None,
    seed: int = DEFAULT_SEED,
    venue: str | None = None,
    venue_rules_path: str | os.PathLike[str] | None = None,
    registry_dir: str | os.PathLike[str] | None = None,
    raw_data_dir: str | os.PathLike[str] | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    use_column_cache: bool = True,
    run_probes: bool = False,
    rescreen: bool = False,
    prior_art: bool = True,
    prior_art_checker: Callable[..., dict] | None = None,
    prior_art_kwargs: dict | None = None,
    weight_vf: float | None = None,
    weight_pen: float | None = None,
    prior_sd: float | None = None,
    max_survivors: int | None = None,
    round_robin_max: int = DEFAULT_ROUND_ROBIN_MAX,
    n_swiss_rounds: int | None = None,
    top_k: int = DEFAULT_TOP_K,
    membership_draws: int = _bt.DEFAULT_N_DRAWS,
    previous_winner: dict | None = None,
    run_shuffle_control: bool = True,
    shuffle_replicates: int = DEFAULT_SHUFFLE_REPLICATES,
    allow_live_selection: bool = False,
    on_event: Callable[[str], None] | None = None,
) -> TournamentResult:
    """Run the full cascade and build every artifact payload.

    ``records`` are T1a ``candidates.jsonl`` rows (or ``Candidate``s).
    ``call_llm`` is injectable, so the whole cascade runs offline in
    tests against a stub and no provider SDK is constructed.

    Raises :class:`LiveSelectionNotAuthorized` when
    ``allow_live_selection=True`` while :data:`V2_STATUS` is uncleared.
    That is deliberate: advisory mode is enforced in code, not by a note
    in a docstring somebody can decide to ignore.
    """
    if allow_live_selection and not V2_STATUS["cleared"]:
        raise LiveSelectionNotAuthorized(
            "This tournament is ADVISORY. " + str(V2_STATUS["why_not_cleared"])
            + " " + str(V2_STATUS["how_this_flips"])
        )

    emit = on_event or (lambda message: None)
    weight_vf = (
        weight_vf
        if weight_vf is not None
        else float(
            _cfg(config, "ideation", "tournament", "weight_venue_fit",
                 default=DEFAULT_WEIGHT_VENUE_FIT)
        )
    )
    weight_pen = (
        weight_pen
        if weight_pen is not None
        else float(
            _cfg(config, "ideation", "tournament", "weight_feasibility_penalty",
                 default=DEFAULT_WEIGHT_PENALTY)
        )
    )
    max_survivors = (
        max_survivors
        if max_survivors is not None
        else int(
            _cfg(config, "ideation", "tournament", "max_survivors_to_tournament",
                 default=DEFAULT_MAX_SURVIVORS)
        )
    )
    n_swiss_rounds = (
        n_swiss_rounds
        if n_swiss_rounds is not None
        else int(
            _cfg(config, "ideation", "tournament", "swiss_rounds",
                 default=DEFAULT_SWISS_ROUNDS)
        )
    )
    prior_sd = (
        prior_sd
        if prior_sd is not None
        else float(
            _cfg(config, "ideation", "tournament", "bt_prior_sd",
                 default=DEFAULT_PRIOR_SD)
        )
    )
    # H8: the scored rule table is configurable; config.yaml points
    # ideation.venue_fit.rules_path at the blind-derived v2 table. An
    # explicit venue_rules_path argument wins; with neither, venue_fit's
    # module default (the v1 table) applies.
    if venue_rules_path is None:
        venue_rules_path = _cfg(
            config, "ideation", "venue_fit", "rules_path", default=None
        )
    k_samples = samples if samples is not None else _judge.judge_samples(config or {})

    stages: list[dict] = []
    killed: list[dict] = []

    # --- Stage 0/1: deterministic screen -------------------------------
    survivors, screen_kills = screen_stage(
        records,
        registry_dir=registry_dir,
        raw_data_dir=raw_data_dir,
        cache_dir=cache_dir,
        use_column_cache=use_column_cache,
        run_probes=run_probes,
        rescreen=rescreen,
    )
    killed.extend(screen_kills)
    stages.append(
        {
            "stage": "feasibility_screen",
            "module": "src.ideation.feasibility.screen (T0)",
            "in": len(screen_kills) + len(survivors),
            "out": len(survivors),
            "killed": len(screen_kills),
            "kill_codes": sorted({str(k["kill_code"]) for k in screen_kills}),
        }
    )
    emit(f"screen: {len(survivors)} survived, {len(screen_kills)} killed")

    # --- Stage 2a: venue routing (H2 Option C) -------------------------
    survivors = routing_stage(survivors, config=config)
    families: dict[str, int] = {}
    for candidate in survivors:
        key = str(candidate.family)
        families[key] = families.get(key, 0) + 1
    stages.append(
        {
            "stage": "venue_routing",
            "module": "src.ideation.venue_router.route_idea (H2 Option C)",
            "in": len(survivors),
            "out": len(survivors),
            "families": dict(sorted(families.items())),
            "note": (
                "deterministic dual-target routing; policy-routed "
                "candidates carry gate_status "
                f"{_router.GATE_STATUS_UNCALIBRATED!r} because "
                "AERA_OPEN has no LSAR calibration"
            ),
        }
    )

    # --- Stage 2b: venue fit -------------------------------------------
    survivors = venue_fit_stage(
        survivors, venue=venue, rules_path=venue_rules_path
    )
    stages.append(
        {
            "stage": "venue_fit",
            "module": "src.ideation.venue_fit.score_venue_fit (T0)",
            "in": len(survivors),
            "out": len(survivors),
            "rules_path": (
                str(venue_rules_path)
                if venue_rules_path is not None
                else str(_vf.DEFAULT_RULES_PATH)
            ),
            "note": (
                "deterministic rule table; under policy-causal routing "
                "VF-01/VF2-01/VF2-02 hits are recorded as routing signals "
                "with delta 0.0 (H2 Option C). The v1 table's VF-01/VF-04 "
                "were authored from our own gate scores and are in-sample "
                "- see V2_STATUS"
            ),
        }
    )

    # --- Stage 2: prior-art veto ---------------------------------------
    survivors, priorart_kills, priorart_status = prior_art_stage(
        survivors,
        enabled=prior_art,
        checker=prior_art_checker,
        on_event=emit,
        # Forwarded verbatim to collision_check: the retriever, the
        # anchor corpus, the config. Passing `anchors=[]` keeps the
        # stage offline (tests), passing `retrieve=` turns it on.
        **(prior_art_kwargs or {}),
    )
    killed.extend(priorart_kills)
    stages.append(
        {
            "stage": "prior_art_veto",
            "module": "src.ideation.priorart.collision_check",
            "ran": priorart_status["ran"],
            "reason": priorart_status["reason"],
            "in": len(survivors) + len(priorart_kills),
            "out": len(survivors),
            "killed": len(priorart_kills),
        }
    )

    # --- field selection ------------------------------------------------
    field_, deferred = select_field(
        survivors, max_survivors, weight_vf=weight_vf, weight_pen=weight_pen
    )
    stages.append(
        {
            "stage": "field_selection",
            "in": len(survivors),
            "out": len(field_),
            "deferred": deferred,
        }
    )
    by_id = {c.candidate_id: c for c in field_}
    ids = [c.candidate_id for c in field_]

    prior_means = {
        c.candidate_id: c.deterministic_score(weight_vf, weight_pen) for c in field_
    }

    # --- the deterministic-only ordering, ALWAYS computed (C3) ---------
    det_rows = deterministic_ordering(
        field_,
        weight_vf=weight_vf,
        weight_pen=weight_pen,
        previous_winner=previous_winner,
    )
    det_order = [row["candidate_id"] for row in det_rows]

    # --- Stage 3: pairwise judging -------------------------------------
    match_records: list[dict] = []
    observations: list[dict] = []
    judge_summary: dict = {
        "ran": False,
        "reason": "judging disabled by the caller" if not judged else "",
    }
    pair_reports: list[dict] = []
    mode = "none"
    pairs_played: list[tuple[str, str]] = []

    if judged and call_llm is not None and len(ids) >= 2:
        mode, first_round = build_pairs(ids, round_robin_max=round_robin_max)
        cards = {cid: by_id[cid].card for cid in ids}
        feas = {cid: by_id[cid].feasibility for cid in ids}
        points: dict[str, float] = {cid: 0.0 for cid in ids}
        played: set[frozenset[str]] = set()
        runs: list[_judge.JudgeRun] = []

        # Round-robin is one "round" of every pair; Swiss adds rounds as
        # results arrive, because its pairings depend on them.
        rounds: list[list[tuple[str, str]]] = [first_round]
        if mode == "swiss":
            played.update(frozenset(p) for p in first_round)

        round_index = 0
        while rounds:
            current = rounds.pop(0)
            round_index += 1
            run = _judge.run_matches(
                current,
                cards,
                call_llm=call_llm,
                feasibility=feas,
                samples=k_samples,
                judge_model=judge_model,
                generator_model=generator_model,
                on_event=emit,
            )
            runs.append(run)
            for pair_result in run.pairs:
                pair_reports.append(pair_result.to_dict())
                overall = pair_result.overall
                if overall is None:
                    continue
                a, b = pair_result.pair
                if overall.winner == a:
                    points[a] = points.get(a, 0.0) + 1.0
                elif overall.winner == b:
                    points[b] = points.get(b, 0.0) + 1.0
                else:
                    points[a] = points.get(a, 0.0) + 0.5
                    points[b] = points.get(b, 0.0) + 0.5
            pairs_played.extend(current)
            if mode == "swiss" and round_index < max(1, n_swiss_rounds):
                nxt = swiss_round(ids, points, played=played)
                if nxt:
                    rounds.append(nxt)

        for run in runs:
            match_records.extend(run.match_records())
            observations.extend(run.bt_observations())
        merged = _judge.JudgeRun(
            pairs=[p for run in runs for p in run.pairs],
            judge_model=judge_model,
            generator_model=generator_model,
            samples=k_samples,
        )
        judge_summary = merged.summary()
        judge_summary["ran"] = True
        judge_summary["mode"] = mode
        judge_summary["rounds"] = round_index
        judge_summary["dimensions_judged"] = list(_judge.ALL_KEYS)
        judge_summary["dimensions_not_judged"] = [
            "novelty (C1: veto only, handled in priorart.py)",
            "feasibility (computed deterministically in Stage 0/1)",
        ]
    elif judged and call_llm is None:
        judge_summary["reason"] = (
            "no judge caller was supplied; the judged layer did not run and "
            "the ranking below is the deterministic ordering"
        )
    elif len(ids) < 2:
        judge_summary["reason"] = (
            f"only {len(ids)} candidate(s) reached the tournament; there is "
            "nothing to compare"
        )

    stages.append(
        {
            "stage": "pairwise_judge",
            "module": "src.ideation.judge.run_matches",
            "ran": judge_summary.get("ran", False),
            "mode": mode,
            "pairs": len(pairs_played),
            "samples_per_orientation": k_samples,
            "orientations": list(_judge.ORIENTATIONS),
        }
    )

    # --- Stage 3b: Bradley-Terry ----------------------------------------
    posterior = _bt.fit(
        observations,
        prior_means=prior_means,
        prior_sd=prior_sd,
        candidates=ids,
        seed=seed,
    )
    membership = _bt.top_k_membership(
        posterior, k=top_k, n_draws=membership_draws, seed=seed
    )
    judged_rows = judged_ordering(
        field_,
        posterior,
        membership,
        weight_vf=weight_vf,
        weight_pen=weight_pen,
        previous_winner=previous_winner,
        top_k=top_k,
    )
    judged_order = [row["candidate_id"] for row in judged_rows]
    stages.append(
        {
            "stage": "bradley_terry",
            "module": "src.ideation.bradley_terry.fit",
            "candidates": len(ids),
            "observations": len(observations),
            "converged": posterior.converged,
            "iterations": posterior.iterations,
        }
    )

    # --- V4 shuffle control ---------------------------------------------
    control: dict = {"ran": False, "reason": "not requested"}
    if run_shuffle_control and observations:
        control = shuffle_control(
            field_,
            observations,
            judged_order=judged_order,
            deterministic_order=det_order,
            prior_means=prior_means,
            prior_sd=prior_sd,
            replicates=shuffle_replicates,
            seed=seed,
        )
    elif run_shuffle_control:
        control = {
            "ran": False,
            "reason": "no judged observations to shuffle",
        }

    ranking = {
        "schema_version": SCHEMA_VERSION,
        "tournament_id": tournament_id,
        "stage": "tournament",
        # The mandatory honesty block. First key a reader hits after the id.
        "advisory": True,
        "authorized_for_live_selection": False,
        "v2_status": V2_STATUS,
        "how_to_read_this_file": (
            "Two orderings are published side by side and NEITHER is "
            "validated. 'ranking' is the judged Bradley-Terry ordering; "
            "'ranking_deterministic' is the same field ordered with every "
            "judged verdict removed (C3). The judged layer has not been "
            "shown to add anything - see v2_status and shuffle_control. "
            "Treat rank 1 as a suggestion with its evidence attached, not "
            "as a selection."
        ),
        "seed": seed,
        "weights": {
            "venue_fit": weight_vf,
            "feasibility_penalty": weight_pen,
            "bt_prior_sd": prior_sd,
        },
        "tie_breaks": list(TIE_BREAK_RULES),
        "cascade": stages,
        "judge": judge_summary,
        "bradley_terry": posterior.to_dict(),
        f"p_top{top_k}": {
            cid: round(float(membership.get(cid, 0.0)), 4) for cid in ids
        },
        "ranking": judged_rows,
        "ranking_deterministic": det_rows,
        "ranking_agreement": {
            "spearman_judged_vs_deterministic": _spearman(judged_order, det_order),
            "mean_absolute_rank_shift": _displacement(judged_order, det_order),
            "rank1_judged": judged_order[0] if judged_order else None,
            "rank1_deterministic": det_order[0] if det_order else None,
            "rank1_agrees": (
                bool(judged_order and det_order and judged_order[0] == det_order[0])
            ),
        },
        "shuffle_control": control,
        "venue_routing": {
            "policy": (
                "Option C dual-target routing (owner decision 2026-08-06, "
                "the v5 capability roadmap (internal) sec. 5.3): causal "
                "and national-survey work routes to the policy-causal "
                "family; measurement, psychometrics and prediction-method "
                "work stays computational-edm. VF2-01/VF2-02 are routing "
                "signals at a policy destination and penalties at a "
                "computational one."
            ),
            "uncalibrated_note": (
                f"{_router.DEFAULT_POLICY_VENUE} has NO LSAR calibration; "
                "every policy-routed candidate carries gate_status "
                f"{_router.GATE_STATUS_UNCALIBRATED!r}. No calibrated "
                "gate exists at that venue yet."
            ),
            "by_candidate": {
                c.candidate_id: c.routing_stamp() for c in field_
            },
        },
        "diversity_ledger": _diversity_ledger(field_, judged_order, det_order,
                                              previous_winner),
        "prior_art": priorart_status,
        "killed_this_stage": [
            {
                "candidate_id": row["candidate_id"],
                "stage": row["stage"],
                "kill_code": row["kill_code"],
                "evidence": row["evidence"],
            }
            for row in killed
        ],
        "winner_spec_written": False,
        "winner_spec_note": (
            "T1b does not write winner_spec.json. Writing it would put a "
            "file on disk whose only purpose is to be fed to a live run, "
            "which is the exact thing V2 has not authorized. Compile it "
            "explicitly from the ranking when a human decides to."
        ),
    }

    result = TournamentResult(
        tournament_id=tournament_id,
        ranking=ranking,
        matches=match_records,
        killed=killed,
        candidates=field_,
        posterior=posterior,
        summary={
            "tournament_id": tournament_id,
            "advisory": True,
            "entrants": len(ids),
            "killed": len(killed),
            "pairs": len(pairs_played),
            "judge_calls": judge_summary.get("calls", 0),
            "rank1_judged": judged_order[0] if judged_order else None,
            "rank1_deterministic": det_order[0] if det_order else None,
        },
    )
    return result


def _diversity_ledger(
    candidates: Sequence[Candidate],
    judged_order: Sequence[str],
    deterministic_order: Sequence[str],
    previous_winner: dict | None,
) -> dict:
    """Spec sec. 6 V4 metric-vs-artifact audit, printed not buried."""
    by_id = {c.candidate_id: c for c in candidates}
    top = [by_id[cid] for cid in list(judged_order)[:5] if cid in by_id]

    def _count(values: Iterable[Any]) -> dict[str, int]:
        out: dict[str, int] = {}
        for value in values:
            key = str(value)
            out[key] = out.get(key, 0) + 1
        return dict(sorted(out.items()))

    targets = _count(c.outcome_family for c in top)
    known = {k: v for k, v in targets.items() if k != "None"}
    rank1 = by_id.get(judged_order[0]) if judged_order else None
    previous_pair = None
    repeat = None
    if previous_winner and rank1 is not None:
        previous_pair = [
            (previous_winner.get("cell") or {}).get("dataset"),
            previous_winner.get("outcome_family"),
        ]
        repeat = bool(
            previous_pair[0] == rank1.dataset
            and previous_pair[1] == rank1.outcome_family
        )
    return {
        "n_entrants": len(candidates),
        "top_k": len(top),
        "top_datasets": _count(c.dataset for c in top),
        "top_task_types": _count(c.task_type for c in top),
        "top_opportunity_patterns": _count(c.opportunity_pattern for c in top),
        "top_outcome_families": targets,
        "collapsed_to_one_dataset": len(_count(c.dataset for c in top)) <= 1
        and len(top) > 1,
        "collapsed_to_one_outcome_family": len(known) == 1
        and len(top) > 1
        and sum(known.values()) == len(top),
        "previous_winner_dataset_outcome": previous_pair,
        # Spec sec. 4.2 rule 5: no two consecutive tournaments may put the
        # same (dataset, outcome-family) at rank 1. It is applied as a
        # tie-break, so a decisive margin can still repeat - and when it
        # does, that is printed here rather than being quietly allowed.
        "rank1_repeats_previous_dataset_outcome": repeat,
        "judged_and_deterministic_agree_on_rank1": (
            bool(judged_order and deterministic_order
                 and judged_order[0] == deterministic_order[0])
        ),
    }


# --------------------------------------------------------------------------
# Human digest
# --------------------------------------------------------------------------


def format_ranking(ranking: dict) -> str:
    """The side-by-side digest. Both orderings, always, in that order."""
    lines: list[str] = []
    tid = ranking.get("tournament_id", "?")
    lines.append(f"# Idea tournament {tid} - ADVISORY")
    lines.append("")
    lines.append("## Status: the ranker is NOT validated")
    lines.append("")
    status = ranking.get("v2_status") or {}
    lines.append(f"V2 backtest: **{status.get('verdict', 'unknown')}**. "
                 f"cleared = {status.get('cleared')}")
    lines.append("")
    lines.append(str(status.get("why_not_cleared", "")))
    lines.append("")
    for row in status.get("measured", []):
        lines.append(
            f"- {row.get('quantity')}: **{row.get('value')}** (n = "
            f"{row.get('n')}). {row.get('note', '')}"
        )
    lines.append("")
    lines.append(f"How this flips: {status.get('how_this_flips', '')}")
    lines.append("")
    lines.append(str(ranking.get("how_to_read_this_file", "")))
    lines.append("")

    judged = ranking.get("ranking") or []
    det = ranking.get("ranking_deterministic") or []
    det_rank = {row["candidate_id"]: row["rank"] for row in det}

    lines.append("## Both orderings, side by side")
    lines.append("")
    lines.append(
        "| judged rank | candidate | BT strength (SD) | P(top-2) | "
        "deterministic rank | det. score | prior art |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for row in judged:
        cid = row["candidate_id"]
        strength = row.get("bt_strength")
        sd = row.get("bt_sd")
        ptop = next(
            (v for k, v in row.items() if k.startswith("p_top")), None
        )
        lines.append(
            f"| {row['rank']} | {cid} | "
            f"{strength if strength is not None else '-'} "
            f"({sd if sd is not None else '-'}) | "
            f"{ptop if ptop is not None else '-'} | "
            f"{det_rank.get(cid, '-')} | {row.get('deterministic_score')} | "
            f"{row.get('prior_art_verdict')} |"
        )
    lines.append("")

    agreement = ranking.get("ranking_agreement") or {}
    lines.append(
        f"Spearman(judged, deterministic) = "
        f"{agreement.get('spearman_judged_vs_deterministic')}; mean absolute "
        f"rank shift = {agreement.get('mean_absolute_rank_shift')}; rank 1 "
        f"agrees = {agreement.get('rank1_agrees')}."
    )
    lines.append("")

    routing = ranking.get("venue_routing") or {}
    lines.append("## Venue routing (H2 Option C dual-target)")
    lines.append("")
    lines.append(str(routing.get("policy", "")))
    lines.append("")
    lines.append(f"**{routing.get('uncalibrated_note', '')}**")
    lines.append("")
    by_candidate = routing.get("by_candidate") or {}
    if by_candidate:
        lines.append("| candidate | family | venue | gate status | rule |")
        lines.append("|---|---|---|---|---|")
        for cid in sorted(by_candidate):
            stamp = by_candidate[cid] or {}
            lines.append(
                f"| {cid} | {stamp.get('family')} | {stamp.get('venue')} | "
                f"{stamp.get('gate_status')} | {stamp.get('rule')} |"
            )
    else:
        lines.append("- no candidates were routed")
    lines.append("")

    control = ranking.get("shuffle_control") or {}
    lines.append("## Shuffle control (does the judge contribute?)")
    lines.append("")
    if control.get("ran"):
        lines.append(f"- method: {control.get('method')}")
        lines.append(f"- replicates: {control.get('replicates')}")
        lines.append(
            "- judged mean rank shift off the deterministic baseline: "
            f"{control.get('judged_vs_deterministic_mean_rank_shift')}"
        )
        lines.append(
            "- shuffled mean rank shift (median, range): "
            f"{control.get('shuffled_mean_rank_shift_median')}, "
            f"{control.get('shuffled_mean_rank_shift_range')}"
        )
        lines.append(
            "- judged displacement inside the shuffled range: "
            f"**{control.get('judged_displacement_inside_shuffled_range')}**"
        )
        lines.append(f"- {control.get('reading')}")
    else:
        lines.append(f"- did not run: {control.get('reason')}")
    lines.append("")

    judge = ranking.get("judge") or {}
    lines.append("## Judge (C2 / C4 audit)")
    lines.append("")
    if judge.get("ran"):
        lines.append(f"- pairs: {judge.get('n_pairs')}, mode {judge.get('mode')}, "
                     f"calls {judge.get('calls')}, errors {judge.get('errors')}")
        lines.append(
            f"- orientations {judge.get('orientations')} x "
            f"{judge.get('samples_per_orientation')} samples, median-aggregated"
        )
        lines.append(
            f"- position-bias rate (orientations disagree): "
            f"**{judge.get('position_bias_rate')}** "
            f"(strict: {judge.get('position_bias_strict_rate')}), pairs "
            f"{judge.get('position_bias_pairs')}"
        )
        lines.append(f"- tie rate on the overall verdict: "
                     f"{judge.get('tie_rate_overall')}")
        lines.append(
            f"- evidence present on {judge.get('evidence_present_rate')} of "
            f"verdicts; written BEFORE the verdict on "
            f"{judge.get('evidence_first_rate')}"
        )
        lines.append(
            f"- judge model {judge.get('judge_model')!r} differs from "
            f"generator {judge.get('generator_model')!r}: "
            f"{judge.get('judge_differs_from_generator')}"
        )
        lines.append(f"- judged: {judge.get('dimensions_judged')}")
        lines.append(f"- NOT judged: {judge.get('dimensions_not_judged')}")
    else:
        lines.append(f"- did not run: {judge.get('reason')}")
    lines.append("")

    prior = ranking.get("prior_art") or {}
    lines.append("## Prior-art veto")
    lines.append("")
    lines.append(
        f"- ran: {prior.get('ran')}"
        + (f" ({prior.get('reason')})" if prior.get("reason") else "")
    )
    lines.append(f"- {prior.get('note')}")
    lines.append("")

    ledger = ranking.get("diversity_ledger") or {}
    lines.append("## Diversity ledger (V4 metric-vs-artifact audit)")
    lines.append("")
    for key in (
        "top_datasets",
        "top_task_types",
        "top_opportunity_patterns",
        "top_outcome_families",
        "collapsed_to_one_dataset",
        "collapsed_to_one_outcome_family",
        "rank1_repeats_previous_dataset_outcome",
    ):
        lines.append(f"- {key}: {ledger.get(key)}")
    if ledger.get("collapsed_to_one_dataset") or ledger.get(
        "collapsed_to_one_outcome_family"
    ):
        lines.append("")
        lines.append(
            "**FAILURE LINE: the top 5 collapsed onto one dataset or one "
            "outcome family. That is a red flag regardless of score (R2).**"
        )
    lines.append("")

    killed = ranking.get("killed_this_stage") or []
    lines.append(f"## Killed at this stage ({len(killed)})")
    lines.append("")
    for row in killed:
        lines.append(
            f"- {row['candidate_id']} [{row['kill_code']}] "
            f"{row['stage']}: {row['evidence']}"
        )
    if not killed:
        lines.append("- none")
    lines.append("")
    lines.append(str(ranking.get("winner_spec_note", "")))
    return "\n".join(lines)
