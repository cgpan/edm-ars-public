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
from typing import Any, Optional

#: Filename written per run, one JSON object per LLM call.
USAGE_FILENAME = "token_usage.jsonl"
#: Aggregated summary written at the end of a run.
SUMMARY_FILENAME = "run_cost.json"


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
    #: Reasoning/thinking tokens where the provider reports them
    #: separately; they bill as output.
    reasoning_tokens: int = 0
    stage: Optional[str] = None
    timestamp: Optional[str] = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict:
        d = asdict(self)
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
            deepseek-v4-pro:   {input: 0.28, cached_input: 0.028, output: 0.42}
            deepseek-v4-flash: {input: 0.07, cached_input: 0.007, output: 0.28}

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


def cost_usd(usage: TokenUsage, pricing: dict) -> Optional[float]:
    """USD for one call, or None when the model has no configured rate."""
    rates = (pricing or {}).get(usage.model)
    if not rates:
        return None
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
    #: "measured" (every call priced from a verified rate), "estimated"
    #: (some rate is unverified), "partial" (some calls unpriced) or
    #: "unpriced" (no call priced). "partial" wins over "estimated".
    cost_status: str = "unpriced"
    #: Per-agent / per-model breakdowns. Each entry's ``cost_usd`` is None
    #: when none of its calls could be priced -- never a silent 0.0 --
    #: and ``unpriced_calls`` says how many of its calls are missing.
    by_agent: dict = field(default_factory=dict)
    by_model: dict = field(default_factory=dict)


def summarize(usages: list[TokenUsage], pricing: dict) -> CostSummary:
    """Aggregate a run's calls into totals, by agent and by model."""
    s = CostSummary(n_calls=len(usages))
    priced_total = 0.0
    any_priced = False
    unpriced: set[str] = set()
    unverified: set[str] = set()

    def _add(entry: dict, c: Optional[float]) -> None:
        if c is None:
            entry["unpriced_calls"] += 1
        else:
            entry["cost_usd"] = (entry["cost_usd"] or 0.0) + c

    for u in usages:
        s.prompt_tokens += u.prompt_tokens
        s.completion_tokens += u.completion_tokens
        s.cached_prompt_tokens += u.cached_prompt_tokens
        c = cost_usd(u, pricing)
        if c is None:
            unpriced.add(u.model)
            s.unpriced_calls += 1
        else:
            any_priced = True
            priced_total += c
            if rate_is_unverified((pricing or {}).get(u.model)):
                unverified.add(u.model)
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
    s.total_tokens = s.prompt_tokens + s.completion_tokens
    s.cost_usd = round(priced_total, 6) if any_priced else None
    s.unpriced_models = sorted(unpriced)
    s.unverified_rate_models = sorted(unverified)
    if not any_priced:
        s.cost_status = "unpriced"
    elif unpriced:
        s.cost_status = "partial"
    elif unverified:
        s.cost_status = "estimated"
    else:
        s.cost_status = "measured"
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
    payload["note"] = (
        "Token counts are MEASURED from provider usage reports. Cost is "
        "those counts multiplied by the configured rate; if rates change, "
        "re-price from the raw counts rather than re-running."
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
