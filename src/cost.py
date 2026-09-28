"""K1 — token metering and cost accounting.

Every run before 2026-08-08 logged a single ``tokens_used`` number per
LLM call: prompt and completion SUMMED. That cannot be costed, because
input and output are priced differently (typically 3-5x apart), and
DeepSeek prices a cache HIT on input at a fraction of a cache miss. A
per-run dollar figure derived from the sum would have been a modelled
number wearing the clothes of a measured one.

This module records what the provider actually reports — prompt,
completion, cached-input and reasoning tokens, per call, per agent —
and converts it to USD using rates that live in ``config.yaml`` rather
than in code, so a rate change is a config edit and never a silent
constant drift.

Design rules:
- Rates come from config. An unpriced model yields ``None`` cost, never
  a guess. A run that cannot be priced says so.
- Token counts are ground truth and are stored raw, so a later rate
  correction re-prices historical runs without re-running them.
- Metering is best-effort at the call site: a provider that omits usage
  must never break the pipeline.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

#: Filename written per run, one JSON object per LLM call.
USAGE_FILENAME = "token_usage.jsonl"
#: Aggregated summary written at the end of a run.
SUMMARY_FILENAME = "run_cost.json"
#: What LSAR writes into each review's folder: its measured calls.
LSAR_USAGE_FILENAME = "token_usage.json"
#: What the review gate writes beside it: when that review started and
#: ended (UTC), so calls LSAR did not time-stamp can still be priced at
#: the right hour.
REVIEW_WINDOW_FILENAME = "review_window.json"

#: The part of a run that paid for a call. Agents' rows carry no
#: component and count as the pipeline's.
PIPELINE_COMPONENT = "pipeline"
#: The review gate: LSAR's reviews and the gate's own paper revisions.
REVIEW_COMPONENT = "review"


@dataclass
class TokenUsage:
    """One LLM call's measured usage."""

    agent: str
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    #: Input tokens served from the provider's prompt cache. Priced far
    #: below a cache miss on DeepSeek, so keeping them separate is the
    #: difference between a real cost and a pessimistic one.
    cached_prompt_tokens: int = 0
    #: Reasoning/thinking tokens where the provider breaks them out.
    #: Informational only: DeepSeek, OpenAI and Anthropic all count them
    #: INSIDE completion_tokens and bill them as output, so pricing them
    #: again on top of completion_tokens would charge them twice.
    reasoning_tokens: int = 0
    stage: Optional[str] = None
    timestamp: Optional[str] = None
    #: Which part of the run paid for the call (``REVIEW_COMPONENT`` for
    #: the review gate). None is the pipeline's agents.
    component: Optional[str] = None
    #: Set only when ``timestamp`` is not the call's own time:
    #: ``"review_window"`` (the call carried no time; the whole review it
    #: belongs to fell inside one rate period, and ``timestamp`` is that
    #: review's start) or ``"unknown"`` (no time could be placed; the call
    #: is priced at the peak rate and the total is an estimate).
    time_source: Optional[str] = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict:
        d = asdict(self)
        # The pipeline's rows stay as they were before the review gate
        # was metered: the two optional fields appear only when set.
        for key in ("component", "time_source"):
            if d.get(key) is None:
                d.pop(key, None)
        d["total_tokens"] = self.total_tokens
        return d


def extract_usage(
    response: Any, agent: str, model: str, provider: str
) -> TokenUsage:
    """Pull usage off a provider response object.

    Handles the OpenAI-compatible shape (``usage.prompt_tokens``) and the
    Anthropic shape (``usage.input_tokens``). Anything missing reads as
    zero rather than raising — a metering failure must not cost a run.
    """
    usage = getattr(response, "usage", None)
    out = TokenUsage(agent=agent, model=model, provider=provider)
    if usage is None:
        return out

    def _get(*names: str) -> int:
        for n in names:
            v = getattr(usage, n, None)
            if v is None and isinstance(usage, dict):
                v = usage.get(n)
            if isinstance(v, (int, float)):
                return int(v)
        return 0

    out.prompt_tokens = _get("prompt_tokens", "input_tokens")
    out.completion_tokens = _get("completion_tokens", "output_tokens")
    # DeepSeek reports cache hits at the top level; OpenAI nests them
    # under prompt_tokens_details.cached_tokens.
    cached = _get("prompt_cache_hit_tokens", "cache_read_input_tokens")
    if not cached:
        details = getattr(usage, "prompt_tokens_details", None)
        if details is not None:
            cached = int(getattr(details, "cached_tokens", 0) or 0)
    out.cached_prompt_tokens = cached
    details = getattr(usage, "completion_tokens_details", None)
    if details is not None:
        out.reasoning_tokens = int(getattr(details, "reasoning_tokens", 0) or 0)
    return out


def load_pricing(config: dict) -> dict:
    """Per-1M-token USD rates from config, keyed by model id.

    Shape (config.yaml)::

        pricing:
          currency: USD
          per_million_tokens:
            some-model: {input: 1.0, cached_input: 0.1, output: 2.0}
            deepseek-v4-pro:
              input: 1.32            # the PEAK (full) rates
              cached_input: 0.044
              output: 3.96
              off_peak: {input: 0.66, cached_input: 0.022, output: 1.98}
              peak_windows_utc:
                days: [mon, tue, wed, thu, fri]
                hours: ["01:00-04:00", "06:00-10:00"]

    An entry without ``off_peak`` has one flat rate. An entry with it is
    priced per call from the call's UTC timestamp (see ``rate_period``).

    Returns an empty dict when unconfigured, which makes every cost
    ``None`` — deliberately. A missing rate must surface as "not priced",
    not as zero dollars.

    Falls back to the repo-root ``config.yaml`` when the active (often
    per-run) config carries no pricing block. A rate is a property of
    the PROVIDER, not of one run, so every run config would otherwise
    have to repeat it — and the one that forgot would silently report
    an unpriced run.
    """
    block = (config or {}).get("pricing") or {}
    rates = block.get("per_million_tokens") or {}
    if rates:
        return rates
    try:
        import yaml

        root_cfg = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config.yaml",
        )
        with open(root_cfg, encoding="utf-8") as fh:
            base = yaml.safe_load(fh) or {}
        return (base.get("pricing") or {}).get("per_million_tokens") or {}
    except Exception:  # noqa: BLE001 — unpriced is a valid outcome
        return {}


def rate_is_unverified(rates: Any) -> bool:
    """True when a pricing entry is flagged as not checked against the
    provider's price list (``unverified: true`` or ``verified: false``).

    Such a rate still prices the call -- the figure is the best available
    -- but everything built from it is labelled an ESTIMATE, never a
    measured cost (defect E5: the flash tier's rates were carried over
    from a retired model id and would otherwise read as fact).
    """
    if not isinstance(rates, dict):
        return False
    if rates.get("unverified"):
        return True
    return rates.get("verified") is False


_WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


def _utc(timestamp: Any) -> Optional[datetime]:
    """A call's timestamp as naive UTC, or None when it cannot be read.

    ``BaseAgent._meter`` writes ``datetime.utcnow().isoformat()`` (naive,
    already UTC); an aware timestamp is converted.
    """
    if not isinstance(timestamp, str) or not timestamp.strip():
        return None
    try:
        dt = datetime.fromisoformat(timestamp.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _minute_of_day(hhmm: str) -> int:
    hours, minutes = hhmm.strip().split(":")
    value = int(hours) * 60 + int(minutes)
    if not 0 <= value <= 24 * 60:
        raise ValueError(hhmm)
    return value


def _peak_windows(schedule: Any) -> Optional[tuple[set, list]]:
    """(weekday names, [(start, end) in minutes]), or None if unreadable."""
    if not isinstance(schedule, dict):
        return None
    try:
        days = {str(d).strip().lower()[:3] for d in schedule.get("days") or ()}
        spans = []
        for span in schedule.get("hours") or ():
            start, end = str(span).split("-")
            spans.append((_minute_of_day(start), _minute_of_day(end)))
    except (TypeError, ValueError):
        return None
    if not days or not spans or not days <= set(_WEEKDAYS):
        return None
    return days, spans


def _period_at(when: datetime, windows: tuple[set, list]) -> str:
    """``"peak"`` or ``"off_peak"`` for a naive-UTC instant."""
    days, spans = windows
    if _WEEKDAYS[when.weekday()] not in days:
        return "off_peak"
    minute = when.hour * 60 + when.minute
    if any(start <= minute < end for start, end in spans):
        return "peak"
    return "off_peak"


def rate_period(usage: TokenUsage, rates: Any) -> str:
    """Which of a model's rates applies to this call.

    ``"flat"``      the entry has one rate (no ``off_peak`` block);
    ``"peak"``      the call's timestamp is inside ``peak_windows_utc``;
    ``"off_peak"``  it is outside them;
    ``"untimed"``   the entry is time-of-day priced but the call has no
                    readable timestamp, or the schedule cannot be read.
                    Such a call is charged the PEAK rate: an unknown time
                    must never price a call below what it can have cost.

    DeepSeek's peak window excludes Chinese public holidays, which this
    does not model, so a weekday-peak call on such a holiday is charged
    the peak rate here and half of it by DeepSeek: an over-estimate,
    never an under-estimate.
    """
    if not isinstance(rates, dict) or not isinstance(rates.get("off_peak"), dict):
        return "flat"
    when = _utc(getattr(usage, "timestamp", None))
    windows = _peak_windows(rates.get("peak_windows_utc"))
    if when is None or windows is None:
        return "untimed"
    return _period_at(when, windows)


#: A window longer than this is not walked minute by minute; it is
#: treated as spanning both periods (no review takes two days).
_MAX_WINDOW = timedelta(days=2)


def window_period(started: Any, ended: Any, rates: Any) -> str:
    """The rate period of a call known only to lie between two instants.

    ``"flat"``, ``"peak"`` or ``"off_peak"`` when every minute from
    *started* to *ended* is priced the same; ``"mixed"`` when the window
    crosses a peak boundary; ``"untimed"`` when either end is unreadable
    (or they are reversed) or the schedule cannot be read. Peak windows
    start and end on whole minutes, so checking *started* and each minute
    boundary up to *ended* sees every change.
    """
    if not isinstance(rates, dict) or not isinstance(rates.get("off_peak"), dict):
        return "flat"
    start, end = _utc(started), _utc(ended)
    windows = _peak_windows(rates.get("peak_windows_utc"))
    if start is None or end is None or windows is None or end < start:
        return "untimed"
    if end - start > _MAX_WINDOW:
        return "mixed"
    first = _period_at(start, windows)
    tick = start.replace(second=0, microsecond=0) + timedelta(minutes=1)
    while tick <= end:
        if _period_at(tick, windows) != first:
            return "mixed"
        tick += timedelta(minutes=1)
    return first


def cost_usd(usage: TokenUsage, pricing: dict) -> Optional[float]:
    """USD for one call, or None when the model has no configured rate.

    The billed quantities are the provider's own counts: uncached input,
    cached input and ``completion_tokens``. Reasoning tokens are already
    inside ``completion_tokens`` and are not added again.
    """
    rates = (pricing or {}).get(usage.model)
    if not rates:
        return None
    if rate_period(usage, rates) == "off_peak":
        rates = rates["off_peak"]
    uncached = max(usage.prompt_tokens - usage.cached_prompt_tokens, 0)
    cached_rate = rates.get("cached_input", rates.get("input", 0.0))
    total = (
        uncached * float(rates.get("input", 0.0))
        + usage.cached_prompt_tokens * float(cached_rate)
        + usage.completion_tokens * float(rates.get("output", 0.0))
    )
    return total / 1_000_000.0


def record_usage(output_dir: Optional[str], usage: TokenUsage) -> None:
    """Append one usage record to the run's ``token_usage.jsonl``.

    Best-effort by contract: metering never raises into the caller.
    """
    if not output_dir:
        return
    try:
        path = os.path.join(output_dir, USAGE_FILENAME)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(usage.to_dict(), allow_nan=False) + "\n")
    except OSError:
        pass


def load_usage(output_dir: str) -> list[TokenUsage]:
    """Read back every recorded call for a run."""
    path = os.path.join(output_dir, USAGE_FILENAME)
    out: list[TokenUsage] = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            d.pop("total_tokens", None)
            try:
                out.append(TokenUsage(**d))
            except TypeError:
                continue
    return out


def load_usage_from_checkpoint(output_dir: str) -> list[TokenUsage]:
    """Rebuild usage from ``checkpoint.json``'s in-memory log.

    ``_meter`` deliberately writes each call to TWO places: the run's
    ``token_usage.jsonl`` and ``ctx.log``, which is re-serialized into
    the checkpoint at every stage boundary. That redundancy is what
    makes the accounting survivable — the jsonl is an append-only file
    on disk that a stray command, a crash mid-write, or a resumed run
    can truncate, while the checkpoint is rewritten whole from memory.
    (It earned its keep on 2026-08-08, when a mistaken ``rm`` removed a
    live run's usage file and every row came back from the checkpoint.)
    """
    path = os.path.join(output_dir, "checkpoint.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as fh:
            ck = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return []
    out: list[TokenUsage] = []
    for e in ck.get("log") or []:
        if "prompt_tokens" not in e:
            continue
        out.append(
            TokenUsage(
                agent=e.get("agent", "?"),
                model=e.get("model", "?"),
                provider="",
                prompt_tokens=int(e.get("prompt_tokens") or 0),
                completion_tokens=int(e.get("completion_tokens") or 0),
                cached_prompt_tokens=int(e.get("cached_prompt_tokens") or 0),
                timestamp=e.get("timestamp"),
                component=e.get("component"),
                time_source=e.get("time_source"),
            )
        )
    return out


def load_usage_best(output_dir: str) -> list[TokenUsage]:
    """Whichever record of this run is more complete.

    Costing must not silently under-report because one of the two sinks
    lost rows: an undercount reads exactly like a cheap run.
    """
    jsonl = load_usage(output_dir)
    ckpt = load_usage_from_checkpoint(output_dir)
    return ckpt if len(ckpt) > len(jsonl) else jsonl


def _count(value: Any) -> int:
    """A token or call count from a JSON value; anything else reads 0."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return max(int(value), 0)


def review_usages(
    payload: Any,
    *,
    started: Optional[str] = None,
    ended: Optional[str] = None,
    pricing: Optional[dict] = None,
    agent: str = "LSAR",
    stage: str = "REVIEWING",
) -> list[TokenUsage]:
    """One LSAR review's LLM calls as ``token_usage.jsonl`` rows.

    *payload* is what LSAR writes to ``token_usage.json`` in the review's
    folder: ``calls`` (one row per call: ``model``, ``provider``,
    ``stage``, ``prompt_tokens``, ``completion_tokens``,
    ``cached_prompt_tokens`` and, from the LSAR that stamps them,
    ``timestamp`` (UTC, with its offset) and ``reasoning_tokens``) beside
    per-review totals and ``by_model``. A payload
    with totals but no ``calls`` is split evenly over each model's
    ``n_calls``, so the call count and the cost both come out right.

    A call with its own timestamp is priced at that time. One without is
    placed in the review's *started*..*ended* window: when every minute of
    the window has the same rate, the call is stamped with *started* and
    ``time_source="review_window"``; when the window crosses a peak
    boundary, or is unknown, the call gets no timestamp and
    ``time_source="unknown"``, which prices it at the peak rate and makes
    the run's cost an estimate.
    """
    if not isinstance(payload, dict):
        return []
    raw_by_model = payload.get("by_model")
    by_model: dict = raw_by_model if isinstance(raw_by_model, dict) else {}
    only_model = next(iter(by_model)) if len(by_model) == 1 else None
    rows: list[dict] = []
    calls = payload.get("calls")
    if isinstance(calls, list) and calls:
        for call in calls:
            if not isinstance(call, dict):
                continue
            rows.append({
                "model": str(call.get("model") or only_model or "unknown"),
                "provider": str(call.get("provider") or ""),
                "prompt": _count(call.get("prompt_tokens")),
                "completion": _count(call.get("completion_tokens")),
                "cached": _count(call.get("cached_prompt_tokens")),
                "reasoning": _count(call.get("reasoning_tokens")),
                "timestamp": call.get("timestamp"),
            })
    else:
        totals = by_model or {
            "unknown": {
                "n_calls": payload.get("n_calls"),
                "prompt_tokens": payload.get("prompt_tokens"),
                "completion_tokens": payload.get("completion_tokens"),
            }
        }
        for model, entry in totals.items():
            if not isinstance(entry, dict):
                continue
            n = max(_count(entry.get("n_calls")), 1)
            prompt = _count(entry.get("prompt_tokens"))
            completion = _count(entry.get("completion_tokens"))
            # by_model carries no cache split. With one model the review's
            # cached count is that model's; with several it is left out,
            # which prices those tokens at the uncached rate: too high,
            # never too low.
            cached = _count(entry.get("cached_prompt_tokens"))
            if not cached and (only_model == model or not by_model):
                cached = _count(payload.get("cached_prompt_tokens"))
            if not (prompt or completion):
                continue
            for i in range(n):
                first = i == 0
                rows.append({
                    "model": str(model),
                    "provider": "",
                    "prompt": prompt // n + (prompt % n if first else 0),
                    "completion": completion // n + (completion % n if first else 0),
                    "cached": cached // n + (cached % n if first else 0),
                    "reasoning": 0,
                    "timestamp": None,
                })

    out: list[TokenUsage] = []
    for row in rows:
        usage = TokenUsage(
            agent=agent,
            model=row["model"],
            provider=row["provider"],
            prompt_tokens=row["prompt"],
            completion_tokens=row["completion"],
            cached_prompt_tokens=min(row["cached"], row["prompt"]),
            reasoning_tokens=row["reasoning"],
            stage=stage,
            component=REVIEW_COMPONENT,
        )
        stamp = row["timestamp"]
        if _utc(stamp) is not None:
            usage.timestamp = str(stamp)
        else:
            rates = (pricing or {}).get(usage.model)
            period = window_period(started, ended, rates)
            if period in ("flat", "peak", "off_peak") and _utc(started) is not None:
                usage.timestamp = str(started)
                usage.time_source = "review_window"
            else:
                usage.time_source = "unknown"
        out.append(usage)
    return out


def write_review_window(review_dir: Any, started: str, ended: str) -> None:
    """Record when one review ran, beside LSAR's ``token_usage.json``."""
    try:
        path = Path(review_dir) / REVIEW_WINDOW_FILENAME
        path.write_text(
            json.dumps({"started_utc": started, "ended_utc": ended}, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass


def load_review_usage(
    review_dir: Any, pricing: Optional[dict] = None, *, agent: str = "LSAR"
) -> list[TokenUsage]:
    """The rows :func:`review_usages` makes from one review folder.

    Reads LSAR's ``token_usage.json`` and the gate's
    ``review_window.json``; a folder without the window (a review made
    before the gate recorded it) prices its untimed calls at the peak
    rate. Returns [] when LSAR recorded no usage there.
    """
    folder = Path(review_dir)
    try:
        payload = json.loads(
            (folder / LSAR_USAGE_FILENAME).read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return []
    started = ended = None
    try:
        window = json.loads(
            (folder / REVIEW_WINDOW_FILENAME).read_text(encoding="utf-8")
        )
        if isinstance(window, dict):
            started, ended = window.get("started_utc"), window.get("ended_utc")
    except (OSError, ValueError):
        pass
    return review_usages(
        payload, started=started, ended=ended, pricing=pricing, agent=agent
    )


@dataclass
class CostSummary:
    n_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_prompt_tokens: int = 0
    total_tokens: int = 0
    cost_usd: Optional[float] = None
    #: Calls whose model had no configured rate. A non-empty list means
    #: cost_usd covers only PART of the run and must be reported as a
    #: lower bound.
    unpriced_models: list = field(default_factory=list)
    #: Models priced from a rate flagged unverified in config.yaml. A
    #: non-empty list makes cost_usd an ESTIMATE.
    unverified_rate_models: list = field(default_factory=list)
    #: Calls with no configured rate (their tokens are in the totals,
    #: their dollars are not).
    unpriced_calls: int = 0
    #: Calls priced at a time-of-day model's peak / off-peak rate (see
    #: ``rate_period``). Calls to flat-rate models are in neither.
    peak_calls: int = 0
    off_peak_calls: int = 0
    #: Calls to a time-of-day model with no readable timestamp. They are
    #: charged the peak rate, which makes cost_usd an ESTIMATE.
    untimed_calls: int = 0
    #: "measured" (every call priced from a verified rate at a known
    #: time), "estimated" (some rate is unverified, or some call's time is
    #: unknown), "partial" (some calls unpriced) or "unpriced" (no call
    #: priced). "partial" wins over "estimated".
    cost_status: str = "unpriced"
    #: Per-agent / per-model breakdowns. Each entry's ``cost_usd`` is None
    #: when none of its calls could be priced -- never a silent 0.0 --
    #: and ``unpriced_calls`` says how many of its calls are missing.
    by_agent: dict = field(default_factory=dict)
    by_model: dict = field(default_factory=dict)
    #: Subtotals for the pipeline's agents and for the review gate (LSAR's
    #: reviews and the gate's revisions). Both keys are present whenever
    #: the run made a call; each has its own ``cost_status``, worked out
    #: as for the whole run.
    by_component: dict = field(default_factory=dict)


def _cost_status(any_priced: bool, unpriced: bool, estimated: bool) -> str:
    if not any_priced:
        return "unpriced"
    if unpriced:
        return "partial"
    if estimated:
        return "estimated"
    return "measured"


def _component_entry() -> dict:
    return {"n_calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "cached_prompt_tokens": 0, "cost_usd": None,
            "unpriced_calls": 0, "untimed_calls": 0, "_unverified": False}


def summarize(usages: list[TokenUsage], pricing: dict) -> CostSummary:
    """Aggregate a run's calls into totals, by agent, model and component."""
    s = CostSummary(n_calls=len(usages))
    priced_total = 0.0
    any_priced = False
    unpriced: set[str] = set()
    unverified: set[str] = set()
    components: dict[str, dict] = {
        PIPELINE_COMPONENT: _component_entry(),
        REVIEW_COMPONENT: _component_entry(),
    }

    def _add(entry: dict, c: Optional[float]) -> None:
        if c is None:
            entry["unpriced_calls"] += 1
        else:
            entry["cost_usd"] = (entry["cost_usd"] or 0.0) + c

    for u in usages:
        s.prompt_tokens += u.prompt_tokens
        s.completion_tokens += u.completion_tokens
        s.cached_prompt_tokens += u.cached_prompt_tokens
        comp = components.setdefault(
            u.component or PIPELINE_COMPONENT, _component_entry()
        )
        comp["n_calls"] += 1
        comp["prompt_tokens"] += u.prompt_tokens
        comp["completion_tokens"] += u.completion_tokens
        comp["cached_prompt_tokens"] += u.cached_prompt_tokens
        c = cost_usd(u, pricing)
        _add(comp, c)
        if c is None:
            unpriced.add(u.model)
            s.unpriced_calls += 1
        else:
            any_priced = True
            priced_total += c
            rates = (pricing or {}).get(u.model)
            if rate_is_unverified(rates):
                unverified.add(u.model)
                comp["_unverified"] = True
            period = rate_period(u, rates)
            if period == "peak":
                s.peak_calls += 1
            elif period == "off_peak":
                s.off_peak_calls += 1
            elif period == "untimed":
                s.untimed_calls += 1
                comp["untimed_calls"] += 1
        agent = s.by_agent.setdefault(
            u.agent, {"n_calls": 0, "prompt_tokens": 0,
                      "completion_tokens": 0, "cost_usd": None,
                      "unpriced_calls": 0}
        )
        agent["n_calls"] += 1
        agent["prompt_tokens"] += u.prompt_tokens
        agent["completion_tokens"] += u.completion_tokens
        _add(agent, c)
        model = s.by_model.setdefault(
            u.model, {"n_calls": 0, "total_tokens": 0, "cost_usd": None,
                      "unpriced_calls": 0}
        )
        model["n_calls"] += 1
        model["total_tokens"] += u.total_tokens
        _add(model, c)
    for entry in list(s.by_agent.values()) + list(s.by_model.values()):
        if entry["cost_usd"] is not None:
            entry["cost_usd"] = round(entry["cost_usd"], 6)
    for comp in components.values():
        unverified_here = comp.pop("_unverified")
        if comp["n_calls"] == 0:
            # Nothing was spent here: a real zero, not an unpriced gap.
            comp["cost_usd"] = 0.0
            comp["cost_status"] = "measured"
            continue
        priced_here = comp["cost_usd"] is not None
        if priced_here:
            comp["cost_usd"] = round(comp["cost_usd"], 6)
        comp["cost_status"] = _cost_status(
            priced_here,
            bool(comp["unpriced_calls"]),
            unverified_here or bool(comp["untimed_calls"]),
        )
    # A run with no calls has nothing to subtotal (and equals an empty
    # CostSummary, as it always has).
    s.by_component = components if usages else {}
    s.total_tokens = s.prompt_tokens + s.completion_tokens
    s.cost_usd = round(priced_total, 6) if any_priced else None
    s.unpriced_models = sorted(unpriced)
    s.unverified_rate_models = sorted(unverified)
    s.cost_status = _cost_status(
        any_priced, bool(unpriced), bool(unverified or s.untimed_calls)
    )
    return s


def write_summary(output_dir: str, config: dict) -> Optional[dict]:
    """Aggregate the run and write ``run_cost.json``. Returns the dict."""
    usages = load_usage_best(output_dir)
    if not usages:
        return None
    pricing = load_pricing(config)
    summary = summarize(usages, pricing)
    payload = asdict(summary)
    payload["pricing_source"] = (
        "config.yaml pricing.per_million_tokens" if pricing else "NOT CONFIGURED"
    )
    pipeline = summary.by_component.get(PIPELINE_COMPONENT) or {}
    review = summary.by_component.get(REVIEW_COMPONENT) or {}
    payload["pipeline_cost_usd"] = pipeline.get("cost_usd")
    payload["review_cost_usd"] = review.get("cost_usd")
    payload["note"] = (
        "Token counts are MEASURED from provider usage reports. Cost is "
        "those counts multiplied by the configured rate; if rates change, "
        "re-price from the raw counts rather than re-running."
    )
    if review.get("n_calls"):
        payload["note"] += (
            f" The review gate made {review['n_calls']} of the "
            f"{summary.n_calls} calls (LSAR's reviews and the gate's paper "
            "revisions); by_component gives the pipeline and review "
            "subtotals."
        )
    if summary.peak_calls or summary.off_peak_calls:
        payload["note"] += (
            f" The provider bills by time of day: {summary.peak_calls} "
            "call(s) fell in its peak window (UTC) and were priced at the "
            f"peak rate, {summary.off_peak_calls} at the off-peak rate."
        )
        if summary.peak_calls:
            payload["note"] += (
                " DeepSeek bills a peak-window call on a Chinese public "
                "holiday at the off-peak rate, half the figure used here."
            )
    if summary.untimed_calls:
        payload["note"] += (
            f" cost_usd is an ESTIMATE: {summary.untimed_calls} call(s) "
            "have no readable time, so they were priced at the peak rate, "
            "up to twice what they cost."
        )
    if summary.unverified_rate_models:
        payload["note"] += (
            " cost_usd is an ESTIMATE: the configured rate for "
            + ", ".join(summary.unverified_rate_models)
            + " is marked unverified in config.yaml pricing."
        )
    if summary.unpriced_models:
        payload["note"] += (
            " cost_usd is a LOWER BOUND: no rate is configured for "
            + ", ".join(summary.unpriced_models)
            + ", so those calls are counted in tokens but not in dollars."
        )
    try:
        with open(
            os.path.join(output_dir, SUMMARY_FILENAME), "w", encoding="utf-8"
        ) as fh:
            json.dump(payload, fh, indent=2, allow_nan=False)
    except OSError:
        pass
    return payload
