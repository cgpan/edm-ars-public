"""Where a study is right now, read from its run folder.

The pipeline runs as a separate process and the CLI only READS its run
folder. Two sources describe progress:

* ``events.jsonl`` -- the structured side channel (``src/events.py``,
  schema v1). Pipelines that carry the event stream write it.
* older runs, or a pipeline without the event stream, leave only
  ``pipeline.log`` (``<utc ts> [Agent] message`` lines),
  ``token_usage.jsonl`` and the stage artifacts.

Both are reduced to the same :class:`RunState` by one pure function,
:func:`fold`. The "tail adapter" turns ``pipeline.log`` lines and usage
rows into synthetic events, so there is one set of rules for what a line
of progress means, not two that drift apart.

Artifacts on disk (``data_report.json``, ``results.json``,
``review_report.json``, ``lsar_review/gate_summary.json``,
``run_status.json`` ...) are the source of truth for numbers and for the
final outcome; events are the source of truth for "what is happening
now". :func:`load_state` combines them.

Everything here is best-effort: a half-written JSON file, a partial last
line or a missing folder degrades what is shown, it never raises into
the live view.
"""
from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

#: Stage keys in pipeline order. ``REVIEWING`` only exists when the LSAR
#: review gate is on; ``REVISING`` only runs when the Critic asks for it.
STAGE_ORDER: tuple[str, ...] = (
    "FORMULATING",
    "ENGINEERING",
    "ANALYZING",
    "CRITIQUING",
    "REVISING",
    "WRITING",
    "REVIEWING",
    "VERIFYING",
)

#: Typical minutes per stage, used for the progress bar and the ETA
#: range. Measured on a handful of DeepSeek runs; deliberately rough.
STAGE_WEIGHTS: dict[str, float] = {
    "FORMULATING": 1.5,
    "ENGINEERING": 3.0,
    "ANALYZING": 6.0,
    "CRITIQUING": 2.5,
    "REVISING": 8.0,
    "WRITING": 3.0,
    "REVIEWING": 20.0,
    "VERIFYING": 0.2,
}

TERMINAL_STATES = frozenset({"COMPLETED", "INCOMPLETE", "ABORTED", "INTERRUPTED"})

#: A run with no process to check and no new output for this long is
#: treated as no longer running (the LSAR step, the longest quiet one,
#: rarely exceeds two hours).
STALE_AFTER_HOURS = 6.0

#: Fallback titles; the user-facing wording lives in edmars/messages.yaml.
_FALLBACK_TITLES: dict[str, str] = {
    "FORMULATING": "Framing the question & finding related studies",
    "ENGINEERING": "Preparing the data",
    "ANALYZING": "Running the analysis",
    "CRITIQUING": "Internal methods review",
    "REVISING": "Revising (only if the reviewer asks for changes)",
    "WRITING": "Writing the paper and making the PDF",
    "REVIEWING": "Automated peer review (LSAR)",
    "VERIFYING": "Final checks",
}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class StageState:
    """One step of the study as the live view shows it."""

    key: str
    title: str
    status: str = "pending"  # pending | running | done | failed | skipped
    started: datetime | None = None
    ended: datetime | None = None
    #: Seconds spent in rounds that already ended (CRITIQUING and
    #: REVISING can run more than once).
    elapsed_s: float = 0.0
    rounds: int = 0
    cycle: int | None = None
    detail: str = ""

    def duration_s(self, now: datetime | None = None) -> float | None:
        """Total seconds in this stage, including a round still running."""
        if self.status == "running" and self.started is not None:
            ref = now or datetime.now(timezone.utc)
            return self.elapsed_s + max(0.0, (ref - self.started).total_seconds())
        if self.rounds or self.elapsed_s:
            return self.elapsed_s
        return None


def _new_stages() -> list[StageState]:
    return [StageState(key=k, title=_FALLBACK_TITLES[k]) for k in STAGE_ORDER]


@dataclass
class RunState:
    """Everything the live view, ``edmars status`` and the result screens
    need to know about one run, folded from events and artifacts."""

    question: str = ""
    task_type: str = ""
    dataset: str = ""
    provider: str = ""
    stages: list[StageState] = field(default_factory=_new_stages)
    current_stage: str | None = None
    now_text: str = ""
    cost_usd: float | None = None
    llm_calls: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    recent: list[str] = field(default_factory=list)
    finished: bool = False
    final_state: str | None = None
    pid: int | None = None
    started: datetime | None = None
    updated: datetime | None = None
    source: str = "tail"  # "events" | "tail"

    # -- extras (not part of the fixed API, used by view/endstates) --
    run_dir: str = ""
    cycle: int = 0
    max_rounds: int = 3
    lsar_enabled: bool = False
    waiting_ai: bool = False
    waiting_agent: str | None = None
    waiting_since: datetime | None = None
    llm_wait: dict[str, Any] | None = None
    attempt: dict[str, Any] | None = None
    code_running: bool = False
    released: bool | None = None
    reason_code: str | None = None
    exit_code: int | None = None
    abort: dict[str, Any] | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    verdict: dict[str, Any] | None = None
    compile: dict[str, Any] | None = None
    gate: dict[str, Any] | None = None
    verify: dict[str, Any] | None = None
    cost_unpriced: bool = False
    last_seq: int = 0
    last_plain: str = ""
    alive: bool | None = None
    stop_requested: bool = False
    resumed: int = 0
    #: When the latest process started (the last ``run.start``); files
    #: older than this belong to an earlier attempt.
    run_started_at: datetime | None = None

    def stage(self, key: str) -> StageState:
        for st in self.stages:
            if st.key == key:
                return st
        st = StageState(key=key, title=_FALLBACK_TITLES.get(key, key.title()))
        self.stages.append(st)
        return st

    def visible_stages(self) -> list[StageState]:
        """Stages the view shows. REVIEWING is hidden while LSAR is off."""
        out = []
        for st in self.stages:
            if st.key == "REVIEWING" and not self.lsar_enabled and st.status in ("pending", "skipped"):
                continue
            out.append(st)
        return out


# ---------------------------------------------------------------------------
# Small parsing helpers
# ---------------------------------------------------------------------------


def parse_ts(value: Any) -> datetime | None:
    """ISO timestamp -> aware UTC datetime. Naive values are UTC (the
    pipeline writes ``datetime.utcnow().isoformat()`` into its log)."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, str):
        try:
            f = float(value.strip())
        except ValueError:
            return None
        return f if math.isfinite(f) else None
    return None


def as_dict(value: Any) -> dict[str, Any]:
    """``value`` when it is a dict, else an empty dict (for JSON read from disk)."""
    return value if isinstance(value, dict) else {}


def _int(value: Any) -> int | None:
    f = _num(value)
    return int(f) if f is not None else None


def _plain_stage(stage: Any) -> str | None:
    if stage is None:
        return None
    text = str(stage)
    if text.startswith("PipelineState."):
        text = text.split(".", 1)[1]
    return text.upper() or None


# ---------------------------------------------------------------------------
# Number formatting (shared by the view, the result screens and summary.html)
# ---------------------------------------------------------------------------


def fmt_num(value: Any) -> str:
    """A number as a person would write it in a results sentence."""
    f = _num(value)
    if f is None:
        return str(value)
    if f == int(f):
        return f"{int(f):,}"
    a = abs(f)
    if a >= 100:
        return f"{f:,.0f}"
    if a >= 10:
        return f"{f:.1f}"
    if a >= 0.1 or a == 0:
        return f"{f:.2f}"
    return f"{f:.2g}"  # small effects keep two significant digits: 0.012, 0.05


def fmt_score(value: Any) -> str:
    """A review score (0-10): one decimal, or a whole number."""
    f = _num(value)
    if f is None:
        return str(value)
    return f"{int(f)}" if f == int(f) else f"{f:.1f}"


def fmt_ci(lo: Any, hi: Any) -> str | None:
    lo_f, hi_f = _num(lo), _num(hi)
    if lo_f is None or hi_f is None:
        return None
    if lo_f >= 0 and hi_f >= 0:
        return f"[{fmt_num(lo_f)}–{fmt_num(hi_f)}]"
    return f"[{fmt_num(lo_f)} to {fmt_num(hi_f)}]"


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return ""
    s = int(max(0, round(seconds)))
    if s >= 3600:
        return f"{s // 3600}h{(s % 3600) // 60:02d}m"
    return f"{s // 60}m{s % 60:02d}s"


# ---------------------------------------------------------------------------
# fold: events -> RunState (pure)
# ---------------------------------------------------------------------------

#: Event types whose ``plain`` text is not "recent news": too frequent,
#: or (stage boundaries) already shown as the step rows themselves.
_QUIET_TYPES = frozenset({"llm.start", "llm.end", "heartbeat", "agent.note", "stage.start", "stage.end"})


def _add_recent(state: RunState, line: str | None) -> None:
    if not line:
        return
    line = " ".join(str(line).split())
    if not line:
        return
    if state.recent and state.recent[-1] == line:
        return
    state.recent.append(line)
    if len(state.recent) > 50:
        del state.recent[: len(state.recent) - 50]


def _stage_start(state: RunState, key: str, ts: datetime | None, cycle: int | None) -> None:
    # A new stage implicitly ends any other stage still marked running;
    # the old pipeline logs no "complete" line for CRITIQUING at all.
    for other in state.stages:
        if other.key != key and other.status == "running":
            _stage_end(state, other.key, ts, "ok")
    st = state.stage(key)
    st.status = "running"
    st.started = ts
    st.ended = None
    st.rounds += 1
    if cycle is not None:
        st.cycle = cycle
        state.cycle = max(state.cycle, int(cycle))
    state.current_stage = key
    state.attempt = None
    state.code_running = False
    state.waiting_ai = False
    state.waiting_agent = None
    state.llm_wait = None
    # Everything before a running stage is done or was not needed.
    idx = STAGE_ORDER.index(key) if key in STAGE_ORDER else -1
    for prev in state.stages:
        if prev.key in STAGE_ORDER and STAGE_ORDER.index(prev.key) < idx:
            if prev.status == "pending":
                prev.status = "skipped" if prev.key in ("REVISING", "REVIEWING") else "done"


def _stage_end(state: RunState, key: str, ts: datetime | None, outcome: Any) -> None:
    st = state.stage(key)
    if st.status == "running" and st.started is not None and ts is not None:
        st.elapsed_s += max(0.0, (ts - st.started).total_seconds())
    if st.rounds == 0:
        st.rounds = 1
    st.ended = ts
    text = str(outcome or "ok").lower()
    st.status = "failed" if text in ("failed", "fail", "error", "aborted", "abort") else "done"
    if key == "ENGINEERING" or key == "ANALYZING":
        state.code_running = False


def _apply(state: RunState, ev: dict[str, Any]) -> None:
    etype = str(ev.get("type") or "")
    seq = ev.get("seq")
    if isinstance(seq, int) and not isinstance(seq, bool):
        if seq <= state.last_seq:
            return
        state.last_seq = seq
    ts = parse_ts(ev.get("ts"))
    if ts is not None:
        if state.updated is None or ts > state.updated:
            state.updated = ts
        if state.started is None:
            state.started = ts
    data = ev.get("data")
    if not isinstance(data, dict):
        data = {}
    stage = _plain_stage(ev.get("stage"))
    plain = ev.get("plain") if isinstance(ev.get("plain"), str) else None
    cycle = _int(ev.get("cycle"))

    if plain and etype not in _QUIET_TYPES:
        _add_recent(state, plain)
        state.last_plain = plain

    if etype == "run.start":
        if state.finished or data.get("resumed"):
            state.resumed += 1
        state.run_started_at = ts or state.run_started_at
        state.finished = False
        state.final_state = None
        state.exit_code = None
        state.released = None
        state.reason_code = None
        state.abort = None
        for attr in ("task_type", "dataset", "provider"):
            value = data.get(attr)
            if isinstance(value, str) and value:
                setattr(state, attr, value)
        if data.get("started_at") and state.started is None:
            state.started = parse_ts(data.get("started_at"))
    elif etype == "stage.start" and stage:
        _stage_start(state, stage, ts, cycle)
        if stage == "REVIEWING":
            state.lsar_enabled = True
    elif etype == "stage.end" and stage:
        _stage_end(state, stage, ts, data.get("outcome"))
    elif etype == "llm.start":
        state.waiting_ai = True
        state.waiting_agent = ev.get("agent") if isinstance(ev.get("agent"), str) else None
        state.waiting_since = ts
        state.llm_wait = None
    elif etype == "llm.end":
        state.waiting_ai = False
        state.waiting_agent = None
        state.llm_wait = None
        state.llm_calls += 1
        cost = _num(data.get("cost_usd"))
        if cost is None:
            state.cost_unpriced = True
        else:
            state.cost_usd = round((state.cost_usd or 0.0) + cost, 6)
    elif etype == "llm.wait":
        state.llm_wait = {
            "seconds": _num(data.get("seconds")),
            "attempt": _int(data.get("attempt")),
            "reason": data.get("reason"),
            "since": ts,
        }
    elif etype == "attempt.start":
        state.attempt = {
            "attempt": _int(data.get("attempt")),
            "max_attempts": _int(data.get("max_attempts")),
            "timeout_s": _num(data.get("timeout_s")),
            "since": ts,
            "stage": stage or state.current_stage,
        }
        state.code_running = True
    elif etype == "attempt.end":
        state.code_running = False
        if state.attempt is not None:
            state.attempt["ended"] = ts
            state.attempt["returncode"] = data.get("returncode")
    elif etype == "lit.progress":
        lit = state.metrics.setdefault("lit_sources", {})
        found = _int(data.get("papers_found"))
        source = str(data.get("source") or "search")
        if found is not None:
            lit[source] = found
            state.metrics["papers_found"] = sum(v for v in lit.values() if isinstance(v, int))
    elif etype == "metric":
        key = data.get("key")
        if key:
            reported = state.metrics.setdefault("reported", {})
            reported[str(key)] = {
                "value": data.get("value"),
                "ci": data.get("ci"),
                "label": data.get("label"),
            }
    elif etype == "verdict":
        state.verdict = dict(data)
        if stage == "CRITIQUING" or stage is None:
            st = state.stage("CRITIQUING")
            if st.status == "running":
                _stage_end(state, "CRITIQUING", ts, "ok")
    elif etype == "compile.end":
        state.compile = dict(data)
    elif etype in ("gate.cycle", "gate.review"):
        gate = dict(state.gate or {})
        gate.update(data)
        gate["ran"] = True
        state.gate = gate
        state.lsar_enabled = True
    elif etype == "gate.skipped":
        state.gate = {
            "ran": False,
            "skip_reason": data.get("skip_reason") or data.get("reason") or plain,
        }
    elif etype == "verify.end":
        state.verify = dict(data)
        if "released" in data:
            state.released = data.get("released")
        if data.get("reason_code"):
            state.reason_code = str(data.get("reason_code"))
    elif etype == "warning":
        msg = plain or data.get("message")
        if msg:
            state.warnings.append(str(msg))
            _add_recent(state, str(msg))
    elif etype == "error":
        msg = plain or data.get("message")
        if msg:
            state.errors.append(str(msg))
            _add_recent(state, str(msg))
        if data.get("code"):
            state.abort = {
                "code": data.get("code"),
                "message": data.get("message"),
                "stage": stage,
            }
    elif etype == "run.end":
        state.finished = True
        final = data.get("state")
        state.final_state = _plain_stage(final) if final else state.final_state
        if "released" in data:
            state.released = data.get("released")
        if data.get("reason_code"):
            state.reason_code = str(data.get("reason_code"))
        if data.get("exit_code") is not None:
            state.exit_code = _int(data.get("exit_code"))
        cost = _num(data.get("cost_usd"))
        if cost is not None:
            state.cost_usd = cost
        if isinstance(data.get("abort"), dict):
            state.abort = dict(data["abort"])
        state.waiting_ai = False
        state.code_running = False
        for st in state.stages:
            if st.status == "running":
                failed = state.final_state in ("ABORTED", "INTERRUPTED", "CRASHED")
                _stage_end(state, st.key, ts, "failed" if failed else "ok")


def fold(events: Iterable[dict[str, Any]], base: RunState | None = None) -> RunState:
    """Reduce events to a :class:`RunState`. Pure: ``base`` is not modified.

    Unknown event types and fields are ignored, so a newer pipeline can
    add events without breaking an older CLI.
    """
    state = copy.deepcopy(base) if base is not None else RunState()
    for ev in events:
        if isinstance(ev, dict):
            try:
                _apply(state, ev)
            except Exception:  # noqa: BLE001 -- one odd event must not blank the view
                continue
    return state


# ---------------------------------------------------------------------------
# Reading events.jsonl incrementally
# ---------------------------------------------------------------------------


class JsonlTail:
    """Read complete new lines from a growing JSONL file.

    Keeps a byte offset, so the live view reads each line once. A partial
    last line (the writer is mid-append) is held back until it is
    complete. Lines that are not JSON objects are skipped.
    """

    def __init__(self, path: Path, *, final: bool = False) -> None:
        self.path = Path(path)
        self.offset = 0
        self._partial = b""
        #: One-shot reads (a finished run) also accept a last line that
        #: has no newline, provided it is a complete JSON object.
        self.final = final

    def read_new(self) -> list[dict[str, Any]]:
        try:
            size = self.path.stat().st_size
        except OSError:
            return []
        if size < self.offset:  # file replaced or truncated: start over
            self.offset = 0
            self._partial = b""
        if size == self.offset:
            return []
        try:
            with open(self.path, "rb") as fh:
                fh.seek(self.offset)
                chunk = fh.read()
        except OSError:
            return []
        self.offset += len(chunk)
        data = self._partial + chunk
        lines = data.split(b"\n")
        self._partial = lines.pop()  # b"" when data ended with a newline
        if self.final and self._partial.strip():
            try:
                if isinstance(json.loads(self._partial.decode("utf-8", errors="replace")), dict):
                    lines.append(self._partial)
                    self._partial = b""
            except ValueError:
                pass  # a torn last line: ignore it
        out: list[dict[str, Any]] = []
        for raw in lines:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw.decode("utf-8", errors="replace"))
            except ValueError:
                continue
            if isinstance(obj, dict):
                out.append(obj)
        return out


def read_events(path: Path) -> list[dict[str, Any]]:
    """All complete events in ``events.jsonl`` (empty list if absent)."""
    return JsonlTail(Path(path), final=True).read_new()


# ---------------------------------------------------------------------------
# Tail adapter: pipeline.log / token_usage.jsonl -> synthetic events
# ---------------------------------------------------------------------------

_LOG_LINE = re.compile(r"^(\d{4}-\d{2}-\d{2}T[\d:.]+(?:Z|[+-]\d{2}:\d{2})?)\s+\[([^\]]+)\]\s?(.*)$")
_ARROW = r"(?:→|->)"
_RE_START = re.compile(r"^Starting ([A-Z]+) stage(?: \(cycle (\d+)\))?")
_RE_COMPLETE = re.compile(r"^([A-Z]+) stage complete\b(?:\s*" + _ARROW + r"\s*([A-Za-z_ ]+))?")
_RE_VERIFY_DONE = re.compile(r"^VERIFYING stage complete\s*" + _ARROW + r"\s*([A-Z]+)(?:\s*\((.*)\))?")
_RE_VERIFY_BLOCKED = re.compile(r"^VERIFYING: release BLOCKED\s*" + _ARROW + r"\s*([A-Z]+)(?:\s*\((.*)\))?")
_RE_ABORTED = re.compile(r"^ABORTED: (.*)")
_RE_VERDICT = re.compile(r"^Critic verdict: (PASS \(UNVERIFIED\)|PASS|REVISE|ABORT)\b(.*)")
_RE_PRECRITIC = re.compile(r"^Pre-Critic guard found critical failures\s*" + _ARROW + r"\s*short-circuit verdict: (\w+)")
_RE_COST = re.compile(r"^Run cost: (?:\$([\d.]+)|not priced) over (\d+) LLM calls")
_RE_RESUMED = re.compile(r"^Resumed from checkpoint \(state=(?:PipelineState\.)?(\w+)\)")
_RE_GATE = re.compile(r"^LSAR review gate: passed=(\w+), cycles=(\d+), score=([\d.]+)")
_RE_GATE_FAIL = re.compile(r"^REVIEWING failed \(non-fatal\): (.*)")
_RE_INVARIANTS = re.compile(r"^Invariant battery: (\d+) critical, (\d+) major, (\d+) minor")
_RE_COMPILE_STEP = re.compile(r"^LaTeX compile step failed: (.*?) \(rc=(-?\d+)\): (.*)")
_RE_REVISING_FAILED = re.compile(r"^REVISING failed \((.*)\); falling back to WRITING")


def parse_log_line(line: str) -> list[dict[str, Any]]:
    """Translate one ``pipeline.log`` line into zero or more events.

    Recognises the orchestrator's stage lines; any other timestamped line
    becomes a ``heartbeat`` so "last update" still moves.
    """
    m = _LOG_LINE.match(line.rstrip("\r\n"))
    if not m:
        return []
    ts, agent, msg = m.group(1), m.group(2), m.group(3).strip()
    ev: dict[str, Any] = {"ts": ts, "agent": agent}

    def make(etype: str, *, stage: str | None = None, cycle: int | None = None,
             plain: str | None = None, **data: Any) -> dict[str, Any]:
        return {**ev, "type": etype, "stage": stage, "cycle": cycle, "plain": plain, "data": data}

    mm = _RE_VERIFY_DONE.match(msg)
    if mm:
        final = mm.group(1)
        return [
            make("stage.end", stage="VERIFYING", outcome="ok"),
            make("verify.end", reason=mm.group(2), state=final),
            make("run.end", state=final),
        ]
    mm = _RE_VERIFY_BLOCKED.match(msg)
    if mm:
        final = mm.group(1)
        return [
            make("stage.end", stage="VERIFYING", outcome="ok"),
            make("verify.end", released=False, reason=mm.group(2), state=final),
            make("run.end", state=final, released=False),
        ]
    mm = _RE_START.match(msg)
    if mm:
        cyc = int(mm.group(2)) if mm.group(2) else None
        return [make("stage.start", stage=mm.group(1), cycle=cyc)]
    mm = _RE_REVISING_FAILED.match(msg)
    if mm:
        return [make("stage.end", stage="REVISING", outcome="ok",
                     plain="The revision step failed; the paper is written from the last version",
                     error=mm.group(1))]
    mm = _RE_COMPLETE.match(msg)
    if mm:
        return [make("stage.end", stage=mm.group(1), outcome="ok")]
    mm = _RE_ABORTED.match(msg)
    if mm:
        return [
            make("error", message=mm.group(1)),
            make("run.end", state="ABORTED", message=mm.group(1)),
        ]
    mm = _RE_PRECRITIC.match(msg)
    if mm:
        verdict = mm.group(1).upper()
        out = [make("verdict", stage="CRITIQUING", verdict=verdict, source="pre_critic")]
        if verdict == "ABORT":
            out.append(make("run.end", state="ABORTED",
                            abort={"code": "PRE_CRITIC_ABORT", "stage": "CRITIQUING"}))
        return out
    mm = _RE_VERDICT.match(msg)
    if mm:
        label = mm.group(1)
        rest = mm.group(2) or ""
        verdict = label.split()[0]
        unverified = "UNVERIFIED" in label or "UNVERIFIED" in rest
        out = [make("verdict", stage="CRITIQUING", verdict=verdict, unverified=unverified)]
        if verdict == "ABORT":
            out.append(make("run.end", state="ABORTED",
                            abort={"code": "CRITIC_ABORT", "stage": "CRITIQUING"}))
        return out
    mm = _RE_COST.match(msg)
    if mm:
        cost = float(mm.group(1)) if mm.group(1) else None
        return [make("cost.total", cost_usd=cost, n_calls=int(mm.group(2)))]
    mm = _RE_RESUMED.match(msg)
    if mm:
        return [make("run.start", resumed=True, plain="Resumed from the last finished step",
                     checkpoint_state=mm.group(1))]
    mm = _RE_GATE.match(msg)
    if mm:
        return [make("gate.review", passed=mm.group(1) == "True",
                     cycles=int(mm.group(2)), score=float(mm.group(3)))]
    mm = _RE_GATE_FAIL.match(msg)
    if mm:
        return [make("gate.skipped", skip_reason=f"exception: {mm.group(1)}")]
    mm = _RE_INVARIANTS.match(msg)
    if mm:
        return [make("verify.counts", critical=int(mm.group(1)),
                     major=int(mm.group(2)), minor=int(mm.group(3)))]
    if msg.startswith("LaTeX compilation succeeded"):
        return [make("compile.end", pdf_exists=True, plain="The PDF was made")]
    mm = _RE_COMPILE_STEP.match(msg)
    if mm:
        text = mm.group(3)
        missing = None
        tool = re.search(r"'([A-Za-z0-9_.-]+)' not found", text)
        if tool:
            missing = tool.group(1)
        return [make("compile.step", cmd=mm.group(1), returncode=int(mm.group(2)),
                     missing_tool=missing)]
    if msg.startswith("LaTeX compilation had errors"):
        return [make("compile.end", pdf_exists=None, failed=True)]
    if msg.startswith("Compiling paper.tex"):
        return [make("log", plain="Turning the paper into a PDF with LaTeX")]
    if msg.startswith("Running OutlineAgent"):
        return [make("log", plain="Planning the paper's outline")]
    return [make("heartbeat")]


def _apply_tail_extras(state: RunState, events: list[dict[str, Any]]) -> None:
    """Tail-only event types that fold() does not know."""
    for ev in events:
        etype = ev.get("type")
        data = ev.get("data") or {}
        if etype == "cost.total":
            state.metrics["log_cost_total"] = {
                "cost": _num(data.get("cost_usd")),
                "n": _int(data.get("n_calls")),
            }
        elif etype == "verify.counts":
            verify = dict(state.verify or {})
            verify["counts"] = {k: data.get(k) for k in ("critical", "major", "minor")}
            state.verify = verify
        elif etype == "compile.step":
            comp = dict(state.compile or {})
            if data.get("missing_tool"):
                comp["missing_tool"] = data.get("missing_tool")
            comp.setdefault("failed_step", data.get("cmd"))
            state.compile = comp
        elif etype == "compile.end" and state.compile:
            # fold() replaced compile with the end record; keep what the
            # step lines said.
            pass


def tail_events_from_log(text: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in text.splitlines():
        out.extend(parse_log_line(line))
    return out


def load_pricing(config: dict[str, Any] | None) -> dict[str, Any]:
    block = (config or {}).get("pricing") or {}
    rates = block.get("per_million_tokens") if isinstance(block, dict) else None
    return rates if isinstance(rates, dict) else {}


def usage_cost(row: dict[str, Any], pricing: dict[str, Any]) -> float | None:
    """USD for one ``token_usage.jsonl`` row, same formula as src/cost.py."""
    rates = pricing.get(str(row.get("model"))) if pricing else None
    if not isinstance(rates, dict):
        return None
    prompt = _num(row.get("prompt_tokens")) or 0.0
    cached = _num(row.get("cached_prompt_tokens")) or 0.0
    completion = _num(row.get("completion_tokens")) or 0.0
    rate_in = _num(rates.get("input")) or 0.0
    rate_cached = _num(rates.get("cached_input"))
    rate_cached = rate_in if rate_cached is None else rate_cached
    rate_out = _num(rates.get("output")) or 0.0
    total = max(prompt - cached, 0.0) * rate_in + cached * rate_cached + completion * rate_out
    return total / 1_000_000.0


def usage_events(rows: Iterable[dict[str, Any]], pricing: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        out.append({
            "type": "llm.end",
            "ts": row.get("timestamp"),
            "agent": row.get("agent"),
            "stage": row.get("stage"),
            "data": {
                "model": row.get("model"),
                "prompt_tokens": row.get("prompt_tokens"),
                "completion_tokens": row.get("completion_tokens"),
                "cached_tokens": row.get("cached_prompt_tokens"),
                "cost_usd": usage_cost(row, pricing),
            },
        })
    return out


# ---------------------------------------------------------------------------
# Artifact readers (cached by mtime)
# ---------------------------------------------------------------------------


class _FileCache:
    """Parse a file again only when its mtime or size changed. A file that
    is mid-write and fails to parse keeps its last good value."""

    def __init__(self) -> None:
        self._cache: dict[str, tuple[float, int, Any]] = {}

    def load(self, path: Path, kind: str = "json") -> Any:
        key = str(path)
        try:
            st = path.stat()
        except OSError:
            self._cache.pop(key, None)
            return None
        sig = (st.st_mtime, st.st_size)
        hit = self._cache.get(key)
        if hit is not None and (hit[0], hit[1]) == sig:
            return hit[2]
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            value = json.loads(text) if kind == "json" else (
                yaml.safe_load(text) if kind == "yaml" else text
            )
        except (OSError, ValueError, yaml.YAMLError):
            return hit[2] if hit is not None else None
        self._cache[key] = (sig[0], sig[1], value)
        return value


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Metrics from artifacts
# ---------------------------------------------------------------------------


def _find_ci(row: dict[str, Any], metric: str) -> tuple[Any, Any]:
    m = metric.lower().replace("-", "_").replace(" ", "_")
    candidates = [m, m.replace("_roc", ""), "auc" if "auc" in m else m]
    for base in candidates:
        lo, hi = row.get(f"{base}_ci_lower"), row.get(f"{base}_ci_upper")
        if lo is not None and hi is not None:
            return lo, hi
        pair = row.get(f"{base}_ci")
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            return pair[0], pair[1]
    if row.get("ci_lower") is not None and row.get("ci_upper") is not None:
        return row.get("ci_lower"), row.get("ci_upper")
    for key, lo in row.items():
        if isinstance(key, str) and key.endswith("_ci_lower"):
            base = key[: -len("_ci_lower")]
            if base and base in m and row.get(f"{base}_ci_upper") is not None:
                return lo, row.get(f"{base}_ci_upper")
    return None, None


def _metric_label(metric: str) -> str:
    m = metric.strip()
    lower = m.lower()
    if lower in ("auc", "auc_roc", "roc_auc", "auc-roc"):
        return "AUC"
    if lower in ("rmse", "mae", "r2", "f1", "f2"):
        return {"r2": "R²"}.get(lower, lower.upper())
    return m


def prediction_metrics(results: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    best = results.get("best_model")
    metric = results.get("primary_metric")
    value = results.get("best_metric_value")
    all_models = as_dict(results.get("all_models"))
    if best:
        out["best_model"] = str(best)
    if metric:
        out["primary_metric"] = _metric_label(str(metric))
    row = all_models.get(best) if isinstance(best, str) else None
    if not isinstance(row, dict) and isinstance(best, str):
        for name, candidate in all_models.items():
            if isinstance(name, str) and name.lower() == best.lower() and isinstance(candidate, dict):
                row = candidate
                break
    if value is None and isinstance(row, dict) and metric:
        value = row.get(str(metric).lower())
    if _num(value) is not None:
        out["best_metric_value"] = _num(value)
    if isinstance(row, dict) and metric:
        lo, hi = _find_ci(row, str(metric))
        if _num(lo) is not None and _num(hi) is not None:
            out["best_ci"] = [_num(lo), _num(hi)]
    out["n_models"] = len(all_models)
    return out


_METHOD_ORDER = ("M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10")


def causal_metrics(results: dict[str, Any], primary: str | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    estimates = as_dict(results.get("estimates")) or as_dict(results.get("all_estimators"))
    if results.get("estimand"):
        out["estimand"] = str(results.get("estimand"))
    chosen: str | None = None
    if primary and isinstance(estimates.get(primary), dict):
        chosen = primary
    # ITR: the policy value evaluation (M7) is the headline number.
    if chosen == "M6" and isinstance(estimates.get("M7"), dict):
        chosen = "M7"
    if chosen is None:
        for key in list(_METHOD_ORDER) + sorted(estimates):
            block = estimates.get(key)
            if isinstance(block, dict) and (
                _num(block.get("point_estimate")) is not None
                or _num(block.get("value_gain_vs_best_constant")) is not None
            ):
                chosen = key
                break
    if chosen is None:
        return out
    block = estimates[chosen]
    out["method"] = chosen
    name = block.get("method_name")
    if isinstance(name, str) and name:
        out["method_name"] = name
    if _num(block.get("point_estimate")) is not None:
        out["estimate"] = _num(block.get("point_estimate"))
        lo, hi = block.get("ci_lower"), block.get("ci_upper")
    else:
        out["estimate"] = _num(block.get("value_gain_vs_best_constant"))
        out["estimate_kind"] = "policy_gain"
        lo, hi = block.get("gain_ci_lower"), block.get("gain_ci_upper")
    if _num(lo) is not None and _num(hi) is not None:
        out["estimate_ci"] = [_num(lo), _num(hi)]
    return out


def _deep_find(node: Any, keys: tuple[str, ...], depth: int = 0) -> float | None:
    if depth > 6:
        return None
    if isinstance(node, dict):
        for k in keys:
            if k in node and _num(node[k]) is not None:
                return _num(node[k])
        for v in node.values():
            found = _deep_find(v, keys, depth + 1)
            if found is not None:
                return found
    elif isinstance(node, list):
        for v in node[:50]:
            found = _deep_find(v, keys, depth + 1)
            if found is not None:
                return found
    return None


def psychometric_metrics(results: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    headline = results.get("headline")
    if isinstance(headline, str) and headline.strip():
        out["headline"] = " ".join(headline.split())
    blocks = results.get("measurement_results")
    if isinstance(blocks, dict):
        found: dict[str, float] = {}
        for label, keys in (
            ("omega", ("omega_total", "omega")),
            ("alpha", ("alpha", "cronbach_alpha")),
            ("CFI", ("cfi", "CFI")),
            ("RMSEA", ("rmsea", "RMSEA")),
            ("reliability", ("marginal_reliability",)),
        ):
            value = _deep_find(blocks, keys)
            if value is not None:
                found[label] = value
        if found:
            out["fit"] = found
    return out


def key_result(metrics: dict[str, Any], task_type: str) -> str | None:
    """One short line with the study's main number, or None if unknown."""
    if metrics.get("best_model"):
        text = f"Best: {metrics['best_model']}"
        if metrics.get("primary_metric") and metrics.get("best_metric_value") is not None:
            text += f", {metrics['primary_metric']} {fmt_num(metrics['best_metric_value'])}"
            ci = metrics.get("best_ci")
            if isinstance(ci, list) and len(ci) == 2:
                text += f" {fmt_ci(ci[0], ci[1])}"
        return text
    if metrics.get("estimate") is not None:
        if metrics.get("estimate_kind") == "policy_gain":
            text = f"Gain from targeting: {fmt_num(metrics['estimate'])}"
        else:
            label = metrics.get("estimand") or "Effect"
            text = f"{label}: {fmt_num(metrics['estimate'])}"
        ci = metrics.get("estimate_ci")
        if isinstance(ci, list) and len(ci) == 2:
            text += f" {fmt_ci(ci[0], ci[1])}"
        if metrics.get("method"):
            text += f" ({metrics['method']})"
        return text
    if metrics.get("headline"):
        return str(metrics["headline"])
    fit = metrics.get("fit")
    if isinstance(fit, dict) and fit:
        return ", ".join(f"{k} {fmt_num(v)}" for k, v in fit.items())
    return None


# ---------------------------------------------------------------------------
# load_state: events (or tail) + artifacts
# ---------------------------------------------------------------------------


class StateReader:
    """Incremental reader for one run folder, used by the live view.

    ``refresh()`` folds only the events written since the last call and
    re-derives artifact-based fields, so polling once a second stays cheap
    on a long run.
    """

    def __init__(self, run_dir: Path | str, *, final: bool = False) -> None:
        self.run_dir = Path(run_dir)
        self.final = final
        self._files = _FileCache()
        self._base: RunState | None = None
        self._mode: str | None = None
        self._reset_tails()

    def _reset_tails(self) -> None:
        self._events_tail = JsonlTail(self.run_dir / "events.jsonl", final=self.final)
        self._log_tail = _TextTail(self.run_dir / "pipeline.log", final=self.final)
        self._usage_tail = JsonlTail(self.run_dir / "token_usage.jsonl", final=self.final)
        self._usage_calls = 0
        self._usage_cost: float | None = None
        self._usage_unpriced = False

    def refresh(self) -> RunState:
        run_dir = self.run_dir
        mode = "events" if (run_dir / "events.jsonl").exists() else "tail"
        if mode != self._mode:
            # Switching source (an old run resumed under a newer pipeline):
            # rebuild from scratch rather than mixing two sources.
            self._mode = mode
            self._base = RunState(source=mode, run_dir=str(run_dir))
            self._reset_tails()
        base = self._base or RunState(source=mode, run_dir=str(run_dir))
        if mode == "events":
            base = fold(self._events_tail.read_new(), base)
        else:
            log_events: list[dict[str, Any]] = []
            for line in self._log_tail.read_new_lines():
                log_events.extend(parse_log_line(line))
            base = fold(log_events, base)
            _apply_tail_extras(base, log_events)
            # Usage rows are counted apart from the log: a "Run cost" line
            # (written once, at the end) is the authoritative total and
            # must not be added on top of the per-call rows.
            rows = self._usage_tail.read_new()
            if rows:
                pricing = load_pricing(self._run_config()) or load_pricing(self._app_config())
                for ev in usage_events(rows, pricing):
                    self._usage_calls += 1
                    cost = ev["data"].get("cost_usd")
                    if cost is None:
                        self._usage_unpriced = True
                    else:
                        self._usage_cost = (self._usage_cost or 0.0) + cost
            logged = base.metrics.get("log_cost_total")
            if isinstance(logged, dict) and int(logged.get("n") or 0) >= self._usage_calls:
                base.llm_calls = int(logged.get("n") or 0)
                base.cost_usd = logged.get("cost")
            else:
                # No total yet, or a resumed run has made calls since the
                # last "Run cost" line: the per-call rows are newer.
                base.llm_calls = self._usage_calls
                base.cost_usd = round(self._usage_cost, 6) if self._usage_cost is not None else None
            base.cost_unpriced = self._usage_unpriced
        base.source = mode
        base.run_dir = str(run_dir)
        self._base = base
        state = copy.deepcopy(base)
        _enrich(state, run_dir, self._files, tail=(mode == "tail"))
        return state

    def _run_config(self) -> dict[str, Any]:
        for name in ("run_config.yaml", "config_snapshot.yaml"):
            cfg = self._files.load(self.run_dir / name, "yaml")
            if isinstance(cfg, dict):
                return cfg
        return {}

    def _app_config(self) -> dict[str, Any]:
        runner = self._files.load(self.run_dir / "runner.json")
        root = runner.get("app_root") if isinstance(runner, dict) else None
        if not root:
            root = str(Path(__file__).resolve().parents[1])
        cfg = self._files.load(Path(root) / "config.yaml", "yaml")
        return cfg if isinstance(cfg, dict) else {}


class _TextTail:
    """Like :class:`JsonlTail`, for plain text lines."""

    def __init__(self, path: Path, *, final: bool = False) -> None:
        self.path = Path(path)
        self.offset = 0
        self._partial = b""
        self.final = final

    def read_new_lines(self) -> list[str]:
        try:
            size = self.path.stat().st_size
        except OSError:
            return []
        if size < self.offset:
            self.offset = 0
            self._partial = b""
        if size == self.offset:
            return []
        try:
            with open(self.path, "rb") as fh:
                fh.seek(self.offset)
                chunk = fh.read()
        except OSError:
            return []
        self.offset += len(chunk)
        data = self._partial + chunk
        parts = data.split(b"\n")
        self._partial = parts.pop()
        if self.final and self._partial.strip():
            parts.append(self._partial)
            self._partial = b""
        return [p.decode("utf-8", errors="replace").rstrip("\r") for p in parts]


def load_state(run_dir: Path | str) -> RunState:
    """Read a run folder once and return its current :class:`RunState`."""
    return StateReader(run_dir, final=True).refresh()


def _read_runner(run_dir: Path, files: _FileCache) -> dict[str, Any]:
    data = files.load(run_dir / "runner.json")
    return data if isinstance(data, dict) else {}


def process_alive(pid: int | None, create_time: float | None = None) -> bool | None:
    """True/False when the pid can be checked, None when it cannot.

    ``create_time`` guards against pid reuse: a pid that now belongs to
    another program is not our run.
    """
    if not pid:
        return None
    try:
        import psutil

        try:
            proc = psutil.Process(int(pid))
        except (psutil.NoSuchProcess, ValueError):
            return False
        if proc.status() == psutil.STATUS_ZOMBIE:
            return False
        if create_time is not None:
            try:
                if abs(proc.create_time() - float(create_time)) > 2.0:
                    return False
            except (psutil.Error, TypeError, ValueError):
                pass
        return True
    except ImportError:
        pass
    except Exception:  # noqa: BLE001 -- access denied etc.: assume alive
        return True
    try:
        from edmars import proc as _proc

        return bool(_proc.pid_alive(int(pid)))
    except Exception:  # noqa: BLE001
        return None


def _stage_done_marks(state: RunState, completed: Iterable[Any]) -> None:
    for key in completed or []:
        k = _plain_stage(key)
        if not k:
            continue
        st = state.stage(k)
        if st.status in ("pending", "running") and not (st.status == "running" and not state.finished):
            st.status = "done"


def _waiting_on_ai(run_dir: Path) -> tuple[bool, str | None, float | None]:
    """Tail mode: a rendered prompt newer than its response means the
    pipeline is waiting for the AI's answer."""
    prompts = run_dir / "prompts"
    if not prompts.is_dir():
        return False, None, None
    newest: tuple[float, str] | None = None
    try:
        for agent_dir in prompts.iterdir():
            if not agent_dir.is_dir():
                continue
            for cycle_dir in agent_dir.iterdir():
                sent = _mtime(cycle_dir / "rendered_prompt.txt")
                if sent is None:
                    continue
                got = _mtime(cycle_dir / "response_raw.txt")
                if got is None or sent > got:
                    if newest is None or sent > newest[0]:
                        newest = (sent, agent_dir.name)
    except OSError:
        return False, None, None
    if newest is None:
        return False, None, None
    return True, newest[1], newest[0]


def _enrich(state: RunState, run_dir: Path, files: _FileCache, *, tail: bool) -> None:
    runner = _read_runner(run_dir, files)
    study = as_dict(runner.get("study"))

    # ---- who/what ------------------------------------------------------
    spec = files.load(run_dir / "research_spec.json")
    locked = files.load(run_dir / "research_spec.locked.json")
    checkpoint = files.load(run_dir / "checkpoint.json")
    checkpoint = checkpoint if isinstance(checkpoint, dict) else {}
    config: dict[str, Any] = {}
    for name in ("run_config.yaml", "config_snapshot.yaml"):
        cfg = files.load(run_dir / name, "yaml")
        if isinstance(cfg, dict):
            config = cfg
            break

    question = study.get("research_question") or study.get("question")
    for source in (spec, locked, checkpoint.get("locked_research_spec")):
        if not question and isinstance(source, dict):
            question = source.get("research_question")
    if question:
        state.question = str(question)
    state.task_type = str(
        study.get("task_type")
        or state.task_type
        or checkpoint.get("task_type")
        or (config.get("pipeline") or {}).get("task_type")
        or ""
    )
    state.dataset = str(study.get("dataset") or state.dataset or checkpoint.get("dataset_name") or "")
    state.provider = str(study.get("provider") or state.provider or config.get("llm_provider") or "")
    rg = as_dict(config.get("review_gate"))
    if rg.get("enabled"):
        state.lsar_enabled = True
    max_cycles = _int((config.get("pipeline") or {}).get("max_revision_cycles"))
    if max_cycles is not None:
        state.max_rounds = max_cycles + 1

    started = parse_ts(runner.get("started_at"))
    if started is not None:
        state.started = started

    # ---- process -------------------------------------------------------
    pid = _int(runner.get("pid"))
    if not pid:
        live = files.load(run_dir / "live_status.json")
        if isinstance(live, dict):
            pid = _int(live.get("pid"))
    state.pid = pid
    state.stop_requested = (run_dir / "STOP").exists() or bool(runner.get("stopped_by_user"))
    alive = process_alive(pid, _num(runner.get("create_time"))) if pid else None
    state.alive = alive

    # ---- stages from the checkpoint --------------------------------------
    ck_state = _plain_stage(checkpoint.get("current_state"))
    completed = checkpoint.get("completed_stages") or []
    _stage_done_marks(state, completed)
    if isinstance(checkpoint.get("revision_cycle"), int):
        state.cycle = max(state.cycle, checkpoint["revision_cycle"])

    # ---- metrics from artifacts -----------------------------------------
    metrics = state.metrics
    retrieved = files.load(run_dir / "retrieved_literature.json")
    lit = files.load(run_dir / "literature_context.json")
    for source in (retrieved, lit):
        papers = source.get("papers") if isinstance(source, dict) else None
        if isinstance(papers, list) and papers:
            metrics["papers_found"] = max(len(papers), _int(metrics.get("papers_found")) or 0)
            break
    if isinstance(lit, dict) and isinstance(lit.get("papers"), list):
        metrics["papers_selected"] = len(lit["papers"])

    report = files.load(run_dir / "data_report.json")
    if isinstance(report, dict):
        for key in ("analytic_n", "n_train", "n_test", "original_n"):
            if _int(report.get(key)) is not None:
                metrics[key] = _int(report.get(key))
        npred = _int(report.get("n_predictors_encoded")) or _int(report.get("n_predictors_raw"))
        if npred is not None:
            metrics["n_predictors"] = npred
        if isinstance(report.get("class_balance"), dict):
            metrics["class_balance"] = report["class_balance"]
        if report.get("outcome_variable"):
            metrics["outcome_variable"] = report["outcome_variable"]

    results = files.load(run_dir / "results.json")
    task = state.task_type or "prediction"
    if isinstance(results, dict):
        if task.startswith("causal") or isinstance(results.get("estimates"), dict):
            primary = None
            for source in (spec, locked):
                if isinstance(source, dict) and source.get("primary_method"):
                    primary = str(source["primary_method"])
                    break
            metrics.update(causal_metrics(results, primary))
        elif task == "psychometrics" or "measurement_results" in results or "headline" in results:
            metrics.update(psychometric_metrics(results))
        else:
            metrics.update(prediction_metrics(results))

    review = files.load(run_dir / "review_report.json")
    if isinstance(review, dict):
        score = _num(review.get("overall_quality_score"))
        if score is not None:
            metrics["critic_score"] = score
        verdict = review.get("effective_verdict") or review.get("overall_verdict")
        if verdict:
            metrics["critic_verdict"] = str(verdict)
        if "unverified" in review:
            metrics["critic_unverified"] = bool(review.get("unverified"))
    if state.verdict:
        if state.verdict.get("critic_score") is not None and "critic_score" not in metrics:
            metrics["critic_score"] = _num(state.verdict.get("critic_score"))
        if state.verdict.get("verdict") and "critic_verdict" not in metrics:
            metrics["critic_verdict"] = str(state.verdict.get("verdict"))
        if state.verdict.get("unverified") is not None and "critic_unverified" not in metrics:
            metrics["critic_unverified"] = bool(state.verdict.get("unverified"))

    gate = files.load(run_dir / "lsar_review" / "gate_summary.json")
    if isinstance(gate, dict):
        merged = dict(state.gate or {})
        merged.update({k: v for k, v in gate.items() if k != "per_cycle_scores"})
        if "ran" not in gate:
            # Released pipelines record a gate that never ran as a
            # failed review scored 0.0 with no cycles. Do not read that
            # as a score.
            merged["ran"] = bool(gate.get("cycles_used")) and gate.get("final_score") is not None
        state.gate = merged
    if state.gate:
        g = state.gate
        if g.get("ran") is not False:
            score = _num(g.get("final_score")) if g.get("final_score") is not None else _num(g.get("score"))
            if score is not None:
                metrics["gate_score"] = score
            threshold = _num(g.get("threshold_used")) if g.get("threshold_used") is not None else _num(g.get("threshold"))
            if threshold is not None:
                metrics["gate_threshold"] = threshold
            if g.get("passed") is not None:
                metrics["gate_passed"] = bool(g.get("passed"))
            advisory = g.get("advisory_mode") if g.get("advisory_mode") is not None else g.get("advisory")
            if advisory is not None:
                metrics["gate_advisory"] = bool(advisory)
        metrics["gate_ran"] = g.get("ran")
        if g.get("skip_reason"):
            metrics["gate_skip_reason"] = g.get("skip_reason")

    inv = files.load(run_dir / "invariants.json")
    if isinstance(inv, dict) and isinstance(inv.get("counts"), dict):
        metrics["invariant_counts"] = inv["counts"]
    elif state.verify and isinstance(state.verify.get("counts"), dict):
        metrics["invariant_counts"] = state.verify["counts"]

    metrics["pdf"] = (run_dir / "paper.pdf").exists()

    # ---- cost ------------------------------------------------------------
    cost_file = files.load(run_dir / "run_cost.json")
    run_started_ts = state.started.timestamp() if state.started else None
    cost_mtime = _mtime(run_dir / "run_cost.json")
    if isinstance(cost_file, dict) and (
        run_started_ts is None or (cost_mtime is not None and cost_mtime >= run_started_ts - 1)
    ):
        n = _int(cost_file.get("n_calls"))
        if n is not None and n >= state.llm_calls:
            state.llm_calls = n
            cost = _num(cost_file.get("cost_usd"))
            if cost is not None:
                state.cost_usd = cost

    if state.updated is None:
        for name in ("pipeline.log", "events.jsonl", "runner.json"):
            m = _mtime(run_dir / name)
            if m is not None:
                state.updated = datetime.fromtimestamp(m, timezone.utc)
                break

    # ---- terminal status -------------------------------------------------
    status = files.load(run_dir / "run_status.json")
    status_fresh = isinstance(status, dict)
    if status_fresh:
        starts = [t for t in (parse_ts(runner.get("resumed_at")), started, state.run_started_at) if t]
        latest_start = max(starts) if starts else None
        status_mtime = _mtime(run_dir / "run_status.json")
        if latest_start is not None and status_mtime is not None and status_mtime < latest_start.timestamp() - 1:
            status_fresh = False  # left over from before this (re)start
    if alive is True and not state.finished:
        status_fresh = status_fresh and not tail  # a live tail run may have a stale file
    if status_fresh and isinstance(status, dict):
        st_state = _plain_stage(status.get("state"))
        if not st_state:
            st_state = "COMPLETED" if status.get("released", True) else "INCOMPLETE"
        if not state.finished or state.final_state is None:
            state.finished = True
            state.final_state = st_state
        if status.get("released") is not None:
            state.released = bool(status.get("released"))
        if status.get("reason_code"):
            state.reason_code = str(status.get("reason_code"))
        if isinstance(status.get("abort"), dict):
            state.abort = dict(status["abort"])
    if not state.finished and ck_state in TERMINAL_STATES and alive is not True:
        state.finished = True
        state.final_state = ck_state

    if (run_dir / "crash.log").exists() and not state.finished and alive is not True:
        state.finished = True
        state.final_state = "CRASHED"

    if not state.finished and alive is False:
        state.finished = True
        state.final_state = "STOPPED" if state.stop_requested else "CRASHED"
    if not state.finished and alive is None and state.updated is not None:
        # No process to ask (a run started outside edmars, or on another
        # machine) and nothing written for hours: it is not running.
        quiet = datetime.now(timezone.utc) - state.updated
        if quiet > timedelta(hours=STALE_AFTER_HOURS):
            state.finished = True
            state.final_state = "STOPPED" if state.stop_requested else "CRASHED"
    if state.finished:
        state.waiting_ai = False
        state.code_running = False
        for st in state.stages:
            if st.status == "running":
                st.status = "failed" if state.final_state in ("ABORTED", "CRASHED", "STOPPED", "INTERRUPTED") else "done"
        if state.final_state in ("COMPLETED", "INCOMPLETE"):
            for st in state.stages:
                if st.status == "pending" and st.key in ("REVISING", "REVIEWING"):
                    st.status = "skipped"

    # ---- live activity (tail mode guesses from files) ---------------------
    if tail and not state.finished:
        waiting, agent, since = _waiting_on_ai(run_dir)
        state.waiting_ai = waiting
        state.waiting_agent = agent
        state.waiting_since = datetime.fromtimestamp(since, timezone.utc) if since else None
        state.code_running = (run_dir / "_generated_script.py").exists()

    # REVISING that never ran is "not needed" once later stages started.
    later_started = any(
        st.status != "pending" for st in state.stages if st.key in ("WRITING", "REVIEWING", "VERIFYING")
    )
    rev = state.stage("REVISING")
    if later_started and rev.status == "pending":
        rev.status = "skipped"

    _stage_details(state)
    state.now_text = describe_now(state)


# ---------------------------------------------------------------------------
# Plain descriptions
# ---------------------------------------------------------------------------


def _messages() -> dict[str, Any]:
    try:
        from edmars.endstates import messages

        return messages()
    except Exception:  # noqa: BLE001 -- wording is optional here
        return {}


def stage_title(state: RunState, st: StageState) -> str:
    msgs = _messages().get("stages") or {}
    entry = msgs.get(st.key) if isinstance(msgs, dict) else None
    title = entry.get("title") if isinstance(entry, dict) else None
    title = title or st.title
    if st.key == "CRITIQUING" and (st.status == "running" or st.rounds > 1):
        rnd = max(st.rounds, 1)
        title = f"{title} (round {rnd} of up to {state.max_rounds})"
    return str(title)


def _stage_details(state: RunState) -> None:
    m = state.metrics
    for st in state.stages:
        detail = ""
        if st.key == "FORMULATING" and m.get("papers_found"):
            n = m["papers_found"]
            detail = f"{n} paper{'s' if n != 1 else ''} found"
        elif st.key == "ENGINEERING" and m.get("analytic_n") is not None:
            detail = f"{m['analytic_n']:,} students"
            if m.get("n_predictors"):
                detail += f" · {m['n_predictors']} predictors"
            share = minority_share(m.get("class_balance"))
            if share is not None:
                detail += f" · smaller outcome group {share:.0%}"
        elif st.key == "ANALYZING" and st.status in ("done", "failed"):
            detail = key_result(m, state.task_type) or ""
        elif st.key == "CRITIQUING" and m.get("critic_verdict"):
            verdict = str(m.get("critic_verdict")).upper()
            words = {"PASS": "passed", "REVISE": "asked for changes", "ABORT": "stopped the study"}
            detail = words.get(verdict.split()[0], verdict.lower())
            if m.get("critic_unverified") and verdict.startswith(("PASS", "REVISE")) and st.status != "running":
                detail = "concerns not fully resolved"
            if m.get("critic_score") is not None:
                detail = f"Score {fmt_score(m['critic_score'])}/10 · {detail}"
        elif st.key == "REVISING":
            if st.status == "skipped":
                detail = "not needed"
            elif st.rounds:
                detail = f"round {st.rounds}"
        elif st.key == "WRITING" and st.status in ("done", "failed"):
            if m.get("pdf"):
                detail = "PDF ready"
            elif state.compile and state.compile.get("missing_tool"):
                detail = "no PDF (LaTeX not installed)"
            else:
                detail = "no PDF"
        elif st.key == "REVIEWING":
            if m.get("gate_ran") is False and m.get("gate_skip_reason"):
                detail = "did not run"
            elif m.get("gate_score") is not None:
                detail = f"Score {fmt_score(m['gate_score'])}"
                if m.get("gate_advisory"):
                    detail += " · score only, no benchmark"
                elif m.get("gate_threshold") is not None:
                    mark = "above" if m.get("gate_passed") else "below"
                    detail += f" · {mark} benchmark {fmt_score(m['gate_threshold'])}"
        elif st.key == "VERIFYING" and isinstance(m.get("invariant_counts"), dict) and st.status == "done":
            c = m["invariant_counts"]
            crit, major = _int(c.get("critical")) or 0, _int(c.get("major")) or 0
            if crit or major:
                parts = []
                if crit:
                    parts.append(f"{crit} serious")
                if major:
                    parts.append(f"{major} to check")
                detail = " · ".join(parts)
            else:
                detail = "no problems found"
        st.detail = detail


def minority_share(balance: Any) -> float | None:
    """The smaller class's share from ``data_report.class_balance``, which
    holds either proportions or counts."""
    if not isinstance(balance, dict) or len(balance) < 2:
        return None
    values = [v for v in (_num(x) for x in balance.values()) if v is not None and v >= 0]
    total = sum(values)
    if len(values) < 2 or total <= 0:
        return None
    return min(values) / total


def _stage_now_text(state: RunState, key: str | None) -> str:
    msgs = _messages().get("stages") or {}
    entry = msgs.get(key) if isinstance(msgs, dict) and key else None
    if isinstance(entry, dict):
        now = entry.get("now")
        if isinstance(now, dict):
            text = now.get(state.task_type) or now.get("default")
        else:
            text = now
        if text:
            return str(text)
    return _FALLBACK_TITLES.get(key or "", "Working")


def running_stage(state: RunState) -> StageState | None:
    for st in state.stages:
        if st.status == "running":
            return st
    return None


def describe_now(state: RunState, now: datetime | None = None) -> str:
    """One plain sentence: what the study is doing at this moment."""
    ref = now or datetime.now(timezone.utc)
    if state.finished:
        return {
            "COMPLETED": "The study has finished.",
            "INCOMPLETE": "The study finished, but the paper is not ready.",
            "ABORTED": "The study stopped before finishing.",
            "INTERRUPTED": "The study was interrupted.",
            "STOPPED": "You stopped the study.",
            "CRASHED": "The study stopped unexpectedly.",
        }.get(state.final_state or "", "The study is not running.")
    st = running_stage(state)
    if st is None:
        if not any(s.status != "pending" for s in state.stages):
            return "Starting up…"
        return "Moving on to the next step…"
    base = _stage_now_text(state, st.key)
    if state.llm_wait and state.llm_wait.get("seconds"):
        secs = int(state.llm_wait["seconds"] or 0)
        return (f"The AI service asked us to slow down; waiting {secs} s before "
                f"trying again.")
    if state.code_running or (state.attempt and not state.attempt.get("ended")):
        att = state.attempt or {}
        since = att.get("since")
        elapsed = fmt_duration((ref - since).total_seconds()) if isinstance(since, datetime) else ""
        limit = att.get("timeout_s")
        limit_min = int(math.ceil(float(limit) / 60)) if limit else 20
        what = "the analysis code" if st.key == "ANALYZING" else "the data-preparation code"
        text = f"Running {what}"
        if att.get("attempt"):
            text += f" (attempt {att['attempt']} of {att.get('max_attempts') or 4})"
        if elapsed:
            text += f" — {elapsed} so far"
        text += f"; this step can take up to {limit_min} minutes"
        return text
    if state.waiting_ai:
        return f"{base} — waiting for the AI's reply…"
    return base


# ---------------------------------------------------------------------------
# Progress and ETA
# ---------------------------------------------------------------------------


#: Final states of a study that ran to its end, whether or not the paper
#: was released.
RAN_TO_END_STATES: tuple[str, ...] = ("COMPLETED", "INCOMPLETE")


def stopped_early(state: RunState) -> bool:
    """True for a study that ended before its last step: stopped by the
    user, crashed, or stopped by an error."""
    return state.finished and state.final_state is not None \
        and state.final_state not in RAN_TO_END_STATES


def progress(state: RunState, now: datetime | None = None) -> tuple[float, datetime | None, datetime | None]:
    """(fraction done, earliest finish, latest finish) from stage weights.

    REVISING counts only once it has started; REVIEWING only when LSAR is
    on. The range is deliberately wide: ``[0.7x, 1.6x]`` the remaining
    typical time.
    """
    ref = now or datetime.now(timezone.utc)
    total = 0.0
    done = 0.0
    completed = 0.0  # steps that finished, not ones that failed
    remaining = 0.0
    for st in state.stages:
        if st.key == "REVISING" and st.status in ("pending", "skipped"):
            continue
        if st.key == "REVIEWING" and not state.lsar_enabled:
            continue
        weight = STAGE_WEIGHTS.get(st.key, 1.0)
        if st.status == "skipped":
            continue
        total += weight
        if st.status in ("done", "failed"):
            done += weight
            if st.status == "done":
                completed += weight
        elif st.status == "running":
            spent = (st.duration_s(ref) or 0.0) / 60.0
            left = max(weight - spent, 0.2 * weight)
            done += max(weight - left, 0.0)
            remaining += left
        else:
            remaining += weight
    if total <= 0:
        return 0.0, None, None
    fraction = min(done / total, 1.0)
    if stopped_early(state):
        return min(completed / total, 1.0), None, None
    if state.finished:
        return 1.0, None, None
    low = ref + timedelta(minutes=remaining * 0.7)
    high = ref + timedelta(minutes=max(remaining * 1.6, remaining + 2))
    return fraction, low, high


def step_position(state: RunState) -> tuple[int, int]:
    """(current step number, number of visible steps).

    For a study that stopped early, the step it stopped at.
    """
    visible = state.visible_stages()
    total = len(visible)
    if stopped_early(state):
        for i, st in enumerate(visible, start=1):
            if st.status == "failed":
                return i, total
        done = sum(1 for st in visible if st.status in ("done", "skipped"))
        return min(done + 1, total), total
    for i, st in enumerate(visible, start=1):
        if st.status == "running":
            return i, total
    done = sum(1 for st in visible if st.status in ("done", "failed", "skipped"))
    return min(done + 1, total) if not state.finished else total, total


def load_json(path: Path) -> Any:
    """Read a JSON file; None when it is missing or unreadable."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError):
        return None


__all__ = [
    "RunState",
    "as_dict",
    "StageState",
    "StateReader",
    "STAGE_ORDER",
    "STAGE_WEIGHTS",
    "describe_now",
    "fmt_ci",
    "fmt_duration",
    "fmt_num",
    "fmt_score",
    "fold",
    "key_result",
    "load_state",
    "parse_log_line",
    "parse_ts",
    "progress",
    "read_events",
    "stage_title",
    "step_position",
    "stopped_early",
    "tail_events_from_log",
    "usage_events",
]

