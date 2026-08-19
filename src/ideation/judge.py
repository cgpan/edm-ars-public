"""Arc T / T1b - the PAIRWISE idea judge.

Why pairwise and not a score
----------------------------
Absolute LLM scores are the thing that failed in this system: an
identical manuscript scored 6.5 and 3.1, and LSAR's own test-retest MAD
is 1.9 on 4 pairs. Spec sec. 3 Stage 3 therefore judges ideas only in
pairs, and only on dimensions that are not computable:

    significance | venue_conversation_fit | clarity_bottleneck | framing_so_what

Novelty and feasibility are deliberately absent (C1 and Stage 0/1). A
model asked for a positive novelty judgement in this repo produced a
number that measured r = -0.35 against the LSAR Novelty it was supposed
to predict, so novelty survives only as a veto with a citation, computed
in ``priorart.py``, never here.

Noise control, all of it mandatory (C4)
---------------------------------------
* **Both orientations.** Every pair is judged twice: once with A shown
  first, once with B shown first. The model never sees a candidate id -
  the cards are labelled "Idea 1" and "Idea 2" - so the only thing the
  swap changes is position.
* **k samples per orientation** (default 3), median-aggregated. A single
  sample is banned.
* **Orientation disagreement is RECORDED, not hidden.** When the two
  orientations reach different majorities that is position bias, and it
  is written into the match record, the pair aggregate, and the run
  summary. Averaging it away silently would turn a measurement of judge
  unreliability into a false consensus.
* **Evidence before verdict.** The prompt requires the evidence string
  first; this module records, per verdict, whether the model actually
  complied (``evidence_first``), read off the raw response text.
* **Length carries no signal**: both cards are rendered through the same
  fixed template and truncated to the same word cap BEFORE judging.
* **Judge model must differ from the generator model** (self-enhancement
  is +10-25% on a model's own output). The run summary records both and
  flags them when they are equal; it does not silently proceed as if
  that were fine.

LLM routing: production goes through ``BaseAgent.call_llm`` via
:class:`IdeaJudgeAgent`, whose system prompt lives in
``agent_prompts/idea_judge.yaml``. Every function here takes the caller
as a plain ``Callable[[str], str]``, so tests run fully offline against
a stub and no provider SDK is ever constructed.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Sequence

from src.ideation import cards as _cards
from src.ideation.cards import IdeaCard

LLMCaller = Callable[[str], str]

AGENT_KEY = "idea_judge"

#: Spec sec. 3 Stage 3. Novelty and feasibility are deliberately absent.
DIMENSIONS: tuple[str, ...] = (
    "significance",
    "venue_conversation_fit",
    "clarity_bottleneck",
    "framing_so_what",
)

#: The dimension the Bradley-Terry fit consumes. The four above are kept
#: per-dimension for the taste layer (spec sec. 5.6 path B fits a
#: per-dimension offset) and for the evidence trail.
OVERALL = "overall"

ALL_KEYS: tuple[str, ...] = DIMENSIONS + (OVERALL,)

#: C1 rail, enforced in code and not only in the prompt: any key a model
#: invents that looks like a novelty or feasibility judgement is dropped
#: before it can reach an aggregate.
_BANNED_DIMENSION_SUBSTRINGS = ("novel", "feasib", "original", "priority", "score")

DEFAULT_SAMPLES = 3
DEFAULT_WORD_CAP = _cards.RENDER_WORD_CAP
ORIENTATIONS: tuple[str, str] = ("AB", "BA")

TIE = "tie"

for _name in ALL_KEYS:  # pragma: no cover - module-level invariant
    if any(bad in _name for bad in _BANNED_DIMENSION_SUBSTRINGS):
        raise AssertionError(
            f"judged dimension {_name!r} names a banned construct; C1 forbids "
            "judging novelty and Stage 0/1 computes feasibility deterministically"
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------
# Rendering the two sides
# --------------------------------------------------------------------------


def _as_card(obj: Any) -> IdeaCard:
    if isinstance(obj, IdeaCard):
        return obj
    if isinstance(obj, dict):
        return IdeaCard.from_dict(obj)
    raise TypeError(
        f"expected an IdeaCard or a card dict, got {type(obj).__name__}"
    )


def render_side(
    card: Any,
    *,
    label: str,
    feasibility: Any = None,
    word_cap: int = DEFAULT_WORD_CAP,
) -> str:
    """Render one card for the judge, with its candidate id removed.

    ``IdeaCard.render`` puts ``[C-07] prediction | els_2002 | pattern`` on
    the first line. The cell facets are legitimate context, the id is
    not: a judge that can see ids can develop a preference over them,
    and the id also survives the orientation swap, which would defeat
    the swap. So the header is rebuilt around the neutral label.
    """
    idea = _as_card(card)
    body = idea.render(feasibility, word_cap).split("\n")
    facets = " | ".join(
        str(part)
        for part in (idea.task_type, idea.dataset, idea.opportunity_pattern)
        if part
    )
    header = f"{label}" + (f"  ({facets})" if facets else "")
    return "\n".join([header] + body[1:])


def build_user_message(
    left: Any,
    right: Any,
    *,
    left_feasibility: Any = None,
    right_feasibility: Any = None,
    word_cap: int = DEFAULT_WORD_CAP,
) -> str:
    """The judge's user message. Deterministic given the two cards."""
    block_1 = render_side(
        left, label="Idea 1", feasibility=left_feasibility, word_cap=word_cap
    )
    block_2 = render_side(
        right, label="Idea 2", feasibility=right_feasibility, word_cap=word_cap
    )
    return "\n\n".join(
        [
            "# Compare these two research ideas",
            "Both cards were written to the same fixed template and "
            "truncated to the same word budget. Length carries no signal.",
            block_1,
            block_2,
            "# Your answer",
            "For each of significance, venue_conversation_fit, "
            "clarity_bottleneck, framing_so_what and overall: write the "
            'evidence first, then the winner ("1", "2" or "tie"). Return '
            "one JSON object and nothing else.",
        ]
    )


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _first_json_object(text: str) -> dict:
    stripped = _FENCE.sub("", str(text or "").strip()).strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except (ValueError, TypeError):
        pass
    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, ch in enumerate(stripped):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                candidate = stripped[start : i + 1]
                try:
                    parsed = json.loads(candidate)
                except ValueError:
                    start = -1
                    continue
                if isinstance(parsed, dict):
                    return parsed
                start = -1
    raise ValueError("no JSON object found in the judge response")


def _normalise_choice(value: Any) -> str:
    text = str(value if value is not None else "").strip().lower()
    if text in {"1", "idea 1", "idea1", "a", "left", "first"}:
        return "1"
    if text in {"2", "idea 2", "idea2", "b", "right", "second"}:
        return "2"
    if text in {"tie", "draw", "equal", "neither", "none", ""}:
        return TIE
    if text.startswith("idea 1"):
        return "1"
    if text.startswith("idea 2"):
        return "2"
    return TIE


def _evidence_precedes_winner(raw: str, dimension: str) -> bool:
    """Did the model write ``evidence`` before ``winner`` for this key?

    Read off the RAW text, not the parsed dict, because ``json.loads``
    preserves order but a re-serialised dict would not tell us what the
    model actually emitted. A false here is a prompt-compliance
    measurement, never a reason to discard the verdict.
    """
    text = str(raw or "")
    key = re.search(rf'"{re.escape(dimension)}"\s*:\s*{{', text)
    if not key:
        return False
    tail = text[key.end() :]
    close = tail.find("}")
    block = tail[:close] if close >= 0 else tail
    e_at = block.find('"evidence"')
    w_at = block.find('"winner"')
    if e_at < 0 or w_at < 0:
        return False
    return e_at < w_at


def parse_response(text: str) -> dict:
    """``{dimension: {"choice": "1"|"2"|"tie", "evidence": str,
    "evidence_first": bool}}`` for every key the model supplied.

    Unknown keys are dropped, and any key that names a banned construct
    (novelty, feasibility, an absolute score) is dropped explicitly and
    counted, so a model cannot smuggle back the judgement C1 removed.
    """
    payload = _first_json_object(text)
    out: dict[str, dict] = {}
    banned: list[str] = []
    for key, value in payload.items():
        name = str(key).strip().lower().replace(" ", "_")
        if any(bad in name for bad in _BANNED_DIMENSION_SUBSTRINGS):
            banned.append(str(key))
            continue
        if name not in ALL_KEYS:
            continue
        if isinstance(value, dict):
            choice = _normalise_choice(
                value.get("winner", value.get("choice", value.get("verdict")))
            )
            evidence = str(value.get("evidence") or "").strip()
        else:
            choice = _normalise_choice(value)
            evidence = ""
        out[name] = {
            "choice": choice,
            "evidence": evidence,
            "evidence_first": _evidence_precedes_winner(text, name),
        }
    if banned:
        out.setdefault("_banned_keys", {})["keys"] = sorted(banned)
    if not any(k in out for k in ALL_KEYS):
        raise ValueError(
            "judge response carried none of the expected dimensions "
            f"{ALL_KEYS}"
        )
    return out


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Verdict:
    """One (pair, orientation, sample, dimension) outcome."""

    pair: tuple[str, str]  # canonical order, independent of orientation
    orientation: str  # which candidate was shown as "Idea 1"
    sample: int
    dimension: str
    winner: str | None  # candidate id, or None for a tie
    evidence: str
    judge_model: str
    evidence_first: bool = False
    raw_choice: str = TIE
    ts: str = ""

    def to_dict(self) -> dict:
        """Spec Appendix B shape, plus the C2/C4 bookkeeping fields."""
        return {
            "pair": list(self.pair),
            "orientation": self.orientation,
            "sample": self.sample,
            "dimension": self.dimension,
            "winner": self.winner,
            "evidence": self.evidence,
            "judge_model": self.judge_model,
            "evidence_first": self.evidence_first,
            "raw_choice": self.raw_choice,
            "shown_first": self.pair[0] if self.orientation == "AB" else self.pair[1],
            "ts": self.ts,
        }


@dataclass
class DimensionAggregate:
    """Median over 2 orientations x k samples, with the split recorded."""

    dimension: str
    pair: tuple[str, str]
    votes: dict[str, int] = field(default_factory=dict)  # candidate -> votes
    ties: int = 0
    winner: str | None = None
    orientation_winners: dict[str, str | None] = field(default_factory=dict)
    position_bias: bool = False
    position_bias_strict: bool = False
    n_votes: int = 0
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "dimension": self.dimension,
            "pair": list(self.pair),
            "votes": dict(self.votes),
            "ties": self.ties,
            "winner": self.winner,
            "orientation_winners": dict(self.orientation_winners),
            "position_bias": self.position_bias,
            "position_bias_strict": self.position_bias_strict,
            "n_votes": self.n_votes,
            "evidence": list(self.evidence),
        }


def _majority(votes: dict[str, int], a: str, b: str) -> str | None:
    if votes.get(a, 0) > votes.get(b, 0):
        return a
    if votes.get(b, 0) > votes.get(a, 0):
        return b
    return None


def aggregate(
    verdicts: Sequence[Verdict], dimension: str, pair: tuple[str, str]
) -> DimensionAggregate:
    """Median verdict for one dimension, with the orientation split kept.

    For a binary outcome the median IS the majority, so this is the
    median C4 asks for. Ties and undecided majorities both resolve to
    ``winner = None``: a 3-3 split is a real measurement of "these two
    are not separable by this judge", and calling it a win for whoever
    happens to sort first would be inventing a preference.
    """
    a, b = pair
    rows = [v for v in verdicts if v.dimension == dimension]
    votes = {a: 0, b: 0}
    ties = 0
    per_orientation: dict[str, dict[str, int]] = {
        o: {a: 0, b: 0} for o in ORIENTATIONS
    }
    evidence: list[str] = []
    for row in rows:
        if row.winner is None:
            ties += 1
        elif row.winner in votes:
            votes[row.winner] += 1
            per_orientation.setdefault(row.orientation, {a: 0, b: 0})
            per_orientation[row.orientation][row.winner] += 1
        if row.evidence:
            evidence.append(
                f"[{row.orientation}#{row.sample}] "
                f"{row.winner or 'tie'}: {row.evidence}"
            )

    orientation_winners = {
        o: _majority(per_orientation.get(o, {}), a, b) for o in ORIENTATIONS
    }
    ab, ba = orientation_winners.get("AB"), orientation_winners.get("BA")
    return DimensionAggregate(
        dimension=dimension,
        pair=pair,
        votes=votes,
        ties=ties,
        winner=_majority(votes, a, b),
        orientation_winners=orientation_winners,
        position_bias=ab != ba,
        position_bias_strict=(ab is not None and ba is not None and ab != ba),
        n_votes=len(rows),
        evidence=evidence,
    )


@dataclass
class PairResult:
    pair: tuple[str, str]
    verdicts: list[Verdict] = field(default_factory=list)
    aggregates: dict[str, DimensionAggregate] = field(default_factory=dict)
    errors: list[dict] = field(default_factory=list)
    calls: int = 0
    judge_model: str = ""

    @property
    def overall(self) -> DimensionAggregate | None:
        return self.aggregates.get(OVERALL)

    @property
    def position_bias_dimensions(self) -> list[str]:
        return sorted(k for k, v in self.aggregates.items() if v.position_bias)

    def bt_observations(self) -> list[dict]:
        """One BT observation per overall vote: 2 orientations x k samples.

        Both orientations enter the fit. Position bias therefore averages
        out of the strength estimate instead of being suppressed before
        it, and it is still recorded separately on the aggregate.
        """
        return [
            {
                "pair": list(self.pair),
                "winner": v.winner,
                "orientation": v.orientation,
                "sample": v.sample,
                "dimension": v.dimension,
                "source": "judge",
            }
            for v in self.verdicts
            if v.dimension == OVERALL
        ]

    def to_dict(self) -> dict:
        return {
            "pair": list(self.pair),
            "judge_model": self.judge_model,
            "calls": self.calls,
            "n_verdicts": len(self.verdicts),
            "aggregates": {k: v.to_dict() for k, v in self.aggregates.items()},
            "position_bias_dimensions": self.position_bias_dimensions,
            "errors": self.errors,
        }


# --------------------------------------------------------------------------
# Judging
# --------------------------------------------------------------------------


def judge_pair(
    left: Any,
    right: Any,
    *,
    call_llm: LLMCaller,
    left_id: str | None = None,
    right_id: str | None = None,
    left_feasibility: Any = None,
    right_feasibility: Any = None,
    samples: int = DEFAULT_SAMPLES,
    word_cap: int = DEFAULT_WORD_CAP,
    judge_model: str = "",
    orientations: Sequence[str] = ORIENTATIONS,
    on_event: Callable[[str], None] | None = None,
) -> PairResult:
    """Judge one pair in BOTH orientations, ``samples`` times each.

    ``left``/``right`` are IdeaCards or card dicts. Ids are taken from
    the cards unless overridden. Nothing about the ordering of the two
    arguments reaches the model: orientation "AB" shows ``left`` first,
    "BA" shows ``right`` first, and the returned ``pair`` is always
    ``(left_id, right_id)`` so aggregates are comparable across calls.
    """
    a = str(left_id or _as_card(left).candidate_id)
    b = str(right_id or _as_card(right).candidate_id)
    if a == b:
        raise ValueError(f"cannot judge a candidate against itself ({a!r})")
    pair = (a, b)
    result = PairResult(pair=pair, judge_model=judge_model)

    sides = {
        "AB": (left, right, left_feasibility, right_feasibility),
        "BA": (right, left, right_feasibility, left_feasibility),
    }
    for orientation in orientations:
        if orientation not in sides:
            raise ValueError(f"unknown orientation {orientation!r}")
        first, second, first_feas, second_feas = sides[orientation]
        message = build_user_message(
            first,
            second,
            left_feasibility=first_feas,
            right_feasibility=second_feas,
            word_cap=word_cap,
        )
        # Which candidate the model's "1" and "2" refer to in THIS
        # orientation. Recomputed per orientation rather than flipped
        # later, so a mapping bug cannot look like position bias.
        shown = (a, b) if orientation == "AB" else (b, a)

        for sample in range(1, max(1, samples) + 1):
            result.calls += 1
            try:
                raw = call_llm(message)
            except Exception as exc:
                result.errors.append(
                    {
                        "pair": list(pair),
                        "orientation": orientation,
                        "sample": sample,
                        "stage": "call",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                if on_event:
                    on_event(f"{a} vs {b} [{orientation}#{sample}]: call failed")
                continue
            try:
                parsed = parse_response(raw)
            except ValueError as exc:
                result.errors.append(
                    {
                        "pair": list(pair),
                        "orientation": orientation,
                        "sample": sample,
                        "stage": "parse",
                        "error": f"{type(exc).__name__}: {exc}",
                        "raw_head": str(raw)[:280],
                    }
                )
                if on_event:
                    on_event(f"{a} vs {b} [{orientation}#{sample}]: unparseable")
                continue
            banned = (parsed.get("_banned_keys") or {}).get("keys")
            if banned:
                result.errors.append(
                    {
                        "pair": list(pair),
                        "orientation": orientation,
                        "sample": sample,
                        "stage": "c1_guard",
                        "error": (
                            "dropped model-invented key(s) naming a banned "
                            f"construct: {banned}"
                        ),
                    }
                )
            ts = _utc_now()
            for dimension in ALL_KEYS:
                row = parsed.get(dimension)
                if not row:
                    continue
                choice = row["choice"]
                winner: str | None
                if choice == "1":
                    winner = shown[0]
                elif choice == "2":
                    winner = shown[1]
                else:
                    winner = None
                result.verdicts.append(
                    Verdict(
                        pair=pair,
                        orientation=orientation,
                        sample=sample,
                        dimension=dimension,
                        winner=winner,
                        evidence=row["evidence"],
                        judge_model=judge_model,
                        evidence_first=bool(row["evidence_first"]),
                        raw_choice=choice,
                        ts=ts,
                    )
                )

    for dimension in ALL_KEYS:
        result.aggregates[dimension] = aggregate(result.verdicts, dimension, pair)
    if on_event:
        overall = result.overall
        on_event(
            f"{a} vs {b}: overall={overall.winner if overall else None} "
            f"votes={overall.votes if overall else {}} "
            f"position_bias={overall.position_bias if overall else None}"
        )
    return result


@dataclass
class JudgeRun:
    pairs: list[PairResult] = field(default_factory=list)
    judge_model: str = ""
    generator_model: str = ""
    samples: int = DEFAULT_SAMPLES
    word_cap: int = DEFAULT_WORD_CAP

    def match_records(self) -> list[dict]:
        return [v.to_dict() for pair in self.pairs for v in pair.verdicts]

    def bt_observations(self) -> list[dict]:
        return [row for pair in self.pairs for row in pair.bt_observations()]

    def summary(self) -> dict:
        verdicts = [v for p in self.pairs for v in p.verdicts]
        overall = [v for v in verdicts if v.dimension == OVERALL]
        decided = [v for v in overall if v.winner is not None]
        biased = [p for p in self.pairs if p.aggregates.get(OVERALL, None)
                  and p.aggregates[OVERALL].position_bias]
        strict = [p for p in self.pairs if p.aggregates.get(OVERALL, None)
                  and p.aggregates[OVERALL].position_bias_strict]
        n_pairs = len(self.pairs)
        return {
            "n_pairs": n_pairs,
            "samples_per_orientation": self.samples,
            "orientations": list(ORIENTATIONS),
            "calls": sum(p.calls for p in self.pairs),
            "errors": sum(len(p.errors) for p in self.pairs),
            "n_verdicts": len(verdicts),
            "n_overall_votes": len(overall),
            "tie_rate_overall": (
                round(1.0 - len(decided) / len(overall), 4) if overall else None
            ),
            "evidence_first_rate": (
                round(sum(1 for v in verdicts if v.evidence_first) / len(verdicts), 4)
                if verdicts
                else None
            ),
            "evidence_present_rate": (
                round(sum(1 for v in verdicts if v.evidence.strip()) / len(verdicts), 4)
                if verdicts
                else None
            ),
            # C4: reported, never smoothed away.
            "position_bias_pairs": sorted("|".join(p.pair) for p in biased),
            "position_bias_rate": (
                round(len(biased) / n_pairs, 4) if n_pairs else None
            ),
            "position_bias_strict_rate": (
                round(len(strict) / n_pairs, 4) if n_pairs else None
            ),
            "judge_model": self.judge_model,
            "generator_model": self.generator_model,
            "judge_differs_from_generator": (
                None
                if not (self.judge_model and self.generator_model)
                else self.judge_model != self.generator_model
            ),
        }


def run_matches(
    pairs: Iterable[tuple[str, str]],
    cards: dict[str, Any],
    *,
    call_llm: LLMCaller,
    feasibility: dict[str, Any] | None = None,
    samples: int = DEFAULT_SAMPLES,
    word_cap: int = DEFAULT_WORD_CAP,
    judge_model: str = "",
    generator_model: str = "",
    on_event: Callable[[str], None] | None = None,
) -> JudgeRun:
    """Judge every pair. Missing cards are skipped with a recorded error."""
    run = JudgeRun(
        judge_model=judge_model,
        generator_model=generator_model,
        samples=samples,
        word_cap=word_cap,
    )
    feas = feasibility or {}
    for a, b in pairs:
        if a not in cards or b not in cards:
            missing = [x for x in (a, b) if x not in cards]
            run.pairs.append(
                PairResult(
                    pair=(a, b),
                    judge_model=judge_model,
                    errors=[
                        {
                            "pair": [a, b],
                            "stage": "setup",
                            "error": f"no card for {missing}",
                        }
                    ],
                )
            )
            continue
        run.pairs.append(
            judge_pair(
                cards[a],
                cards[b],
                call_llm=call_llm,
                left_id=a,
                right_id=b,
                left_feasibility=feas.get(a),
                right_feasibility=feas.get(b),
                samples=samples,
                word_cap=word_cap,
                judge_model=judge_model,
                on_event=on_event,
            )
        )
    return run


# --------------------------------------------------------------------------
# Production LLM routing (BaseAgent only)
# --------------------------------------------------------------------------


@dataclass
class _JudgeContext:
    """The minimal surface BaseAgent reads off a pipeline context."""

    dataset_name: str
    task_type: str = "prediction"
    output_dir: str | None = None
    revision_cycle: int = 0
    log: list = field(default_factory=list)


class _NoExecutor:
    def run(self, *args: Any, **kwargs: Any) -> dict:
        raise RuntimeError(
            "The idea judge does not execute code. If this was reached, "
            "something routed a sandbox job to the wrong agent."
        )


def resolve_judge_model(config: dict) -> str | None:
    """``ideation.models.judge`` if configured, else None.

    None means "leave BaseAgent's per-stage resolution alone". No model
    id is hardcoded in this module.
    """
    ideation = (config or {}).get("ideation") or {}
    model = (ideation.get("models") or {}).get("judge")
    return str(model) if model else None


def judge_samples(config: dict) -> int:
    tournament = ((config or {}).get("ideation") or {}).get("tournament") or {}
    try:
        value = int(tournament.get("judge_samples", DEFAULT_SAMPLES))
    except (TypeError, ValueError):
        return DEFAULT_SAMPLES
    return max(1, value)


def judge_temperature(config: dict) -> float | None:
    """``ideation.tournament.judge_temperature``, else None.

    None means "use the temperature in ``agent_prompts/idea_judge.yaml``",
    which is the source of truth. That file sets a small positive value
    on purpose: at temperature 0 the k samples C4 requires would be
    three copies of one draw, i.e. triple the cost for no variance
    reduction at all.
    """
    tournament = ((config or {}).get("ideation") or {}).get("tournament") or {}
    value = tournament.get("judge_temperature")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class IdeaJudgeAgent:
    """Thin BaseAgent wrapper, constructed lazily so importing this
    module never touches a provider SDK or an API key."""

    AGENT_NAME = AGENT_KEY

    def __init__(
        self,
        config: dict,
        *,
        dataset: str,
        task_type: str = "prediction",
        output_dir: str | None = None,
        model: str | None = None,
    ) -> None:
        from src.agents.base import BaseAgent

        class _Agent(BaseAgent):  # local subclass: BaseAgent.run is abstract
            def run(self, **kwargs: Any) -> dict:
                raise NotImplementedError(
                    "The idea judge is driven by "
                    "src.ideation.judge.run_matches, not by a pipeline stage "
                    "runner."
                )

        context = _JudgeContext(
            dataset_name=dataset, task_type=task_type, output_dir=output_dir
        )
        self.agent = _Agent(context, self.AGENT_NAME, config, executor=_NoExecutor())
        if model:
            self.agent.model = model
        self.model = self.agent.model

    def __call__(self, user_message: str, temperature: float | None = None) -> str:
        return self.agent.call_llm(user_message, temperature_override=temperature)


def make_llm_caller(
    config: dict,
    *,
    dataset: str,
    task_type: str = "prediction",
    output_dir: str | None = None,
) -> tuple[LLMCaller, str]:
    """``(caller, model_id)`` routed through ``BaseAgent.call_llm``."""
    agent = IdeaJudgeAgent(
        config,
        dataset=dataset,
        task_type=task_type,
        output_dir=output_dir,
        model=resolve_judge_model(config),
    )
    temperature = judge_temperature(config)

    def _call(user_message: str) -> str:
        return agent(user_message, temperature)

    return _call, agent.model


# --------------------------------------------------------------------------
# Offline stub
# --------------------------------------------------------------------------

OFFLINE_MODEL_ID = "offline-tie-stub"

_OFFLINE_EVIDENCE = (
    "OFFLINE STUB: no model was called. This is not a judgement about "
    "either idea and must not be read as one."
)


def offline_caller() -> LLMCaller:
    """A judge that always answers "tie", loudly.

    Deliberately NOT a hash-based pseudo-verdict: a fake preference
    would flow into the Bradley-Terry fit and produce an ordering that
    looks judged and is not. Answering tie everywhere leaves the
    posterior at its deterministic prior, which is exactly what the
    C3 fallback says should happen when the judged layer contributes
    nothing.
    """

    def _call(user_message: str) -> str:
        return json.dumps(
            {
                key: {"evidence": _OFFLINE_EVIDENCE, "winner": TIE}
                for key in ALL_KEYS
            }
        )

    return _call
