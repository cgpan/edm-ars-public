"""Structured run events: a side channel a user interface can follow live.

The pipeline appends one JSON object per line to ``<run>/events.jsonl``
and keeps ``<run>/live_status.json`` as a small snapshot of "where are we
now". Nothing in the pipeline reads either file back; they exist so that a
terminal view, ``edmars status`` or a future web page can show progress
without parsing ``pipeline.log``.

Contract (schema v1)::

    {"v": 1, "seq": 12, "ts": "2026-09-25T13:05:11.203Z", "run_id": "...",
     "type": "stage.start", "stage": "ENGINEERING", "cycle": 0,
     "agent": null, "plain": "Preparing the data", "data": {...}}

Rules, the same ones ``BaseAgent._meter`` follows for token accounting:

* emitting an event NEVER raises -- a UI side channel must not be able to
  abort a paid run;
* the file is opened, appended and closed per event, so a reader can tail
  it and a crash leaves every earlier line intact;
* the snapshot is replaced atomically (temp file + ``os.replace``).
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from typing import Any, Callable, Optional

SCHEMA_VERSION = 1

#: Called with each record after it is written, when set. The command
#: line sets it to print a short progress line per stage, wait, retry and
#: warning (D8: the console was otherwise silent for the whole run, which
#: a first-time user cannot tell from a hang). Process-wide, like the
#: console it writes to; errors in it are swallowed like every other part
#: of this side channel.
_echo: Optional[Callable[[dict[str, Any]], None]] = None


def set_echo(
    fn: Optional[Callable[[dict[str, Any]], None]],
) -> Optional[Callable[[dict[str, Any]], None]]:
    """Install *fn* as the console echo; return the previous one."""
    global _echo
    previous = _echo
    _echo = fn
    return previous

#: The event types a reader should expect. Readers must still tolerate
#: types that are not listed here.
EVENT_TYPES = frozenset({
    "run.start", "run.end",
    "stage.start", "stage.end",
    "log", "agent.note",
    "llm.start", "llm.end", "llm.wait",
    "attempt.start", "attempt.end",
    "lit.progress", "metric", "verdict",
    "compile.end",
    "gate.cycle", "gate.review", "gate.skipped", "gate.cost",
    "verify.end", "warning", "error",
})


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class EventSink:
    """Appends events for one run directory. Thread-safe, never raises."""

    def __init__(self, run_dir: str, run_id: str | None = None) -> None:
        self.run_dir = run_dir
        self.run_id = run_id or os.path.basename(os.path.normpath(run_dir))
        self.path = os.path.join(run_dir, "events.jsonl")
        self.status_path = os.path.join(run_dir, "live_status.json")
        self._lock = threading.Lock()
        self._seq = self._last_seq()
        self._status: dict[str, Any] = {"v": SCHEMA_VERSION, "run_id": self.run_id, "pid": os.getpid()}

    def _last_seq(self) -> int:
        """Continue numbering after a resume instead of restarting at 0."""
        try:
            with open(self.path, "rb") as fh:
                fh.seek(0, os.SEEK_END)
                size = fh.tell()
                fh.seek(max(0, size - 4096))
                tail = fh.read().decode("utf-8", errors="replace").strip().splitlines()
            for line in reversed(tail):
                try:
                    return int(json.loads(line).get("seq", 0))
                except (ValueError, AttributeError):
                    continue
        except OSError:
            pass
        return 0

    def emit(
        self,
        type: str,
        *,
        stage: str | None = None,
        cycle: int | None = None,
        agent: str | None = None,
        plain: str | None = None,
        **data: Any,
    ) -> None:
        try:
            with self._lock:
                self._seq += 1
                record = {
                    "v": SCHEMA_VERSION,
                    "seq": self._seq,
                    "ts": _utc_now(),
                    "run_id": self.run_id,
                    "type": type,
                    "stage": _plain_state(stage),
                    "cycle": cycle,
                    "agent": agent,
                    "plain": plain,
                    "data": data,
                }
                line = json.dumps(record, ensure_ascii=False, default=str)
                os.makedirs(self.run_dir, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
                self._update_status(record)
        except Exception:  # noqa: BLE001 -- a UI side channel must never raise
            return
        echo = _echo
        if echo is not None:
            try:
                echo(record)
            except Exception:  # noqa: BLE001
                pass

    def _update_status(self, record: dict[str, Any]) -> None:
        status = self._status
        status["seq"] = record["seq"]
        status["updated"] = record["ts"]
        status["pid"] = os.getpid()
        etype = record["type"]
        if etype == "stage.start":
            status["stage"] = record["stage"]
            status["cycle"] = record["cycle"]
            status["stage_started"] = record["ts"]
        if record.get("plain"):
            status["now"] = record["plain"]
        if etype == "llm.end":
            status["llm_calls"] = int(status.get("llm_calls", 0)) + 1
            cost = record["data"].get("cost_usd")
            if isinstance(cost, (int, float)):
                status["cost_usd"] = round(float(status.get("cost_usd", 0.0)) + float(cost), 6)
        if etype == "metric":
            metrics = status.setdefault("metrics", {})
            key = record["data"].get("key")
            if key:
                metrics[str(key)] = record["data"].get("value")
        if etype == "run.end":
            status["state"] = record["data"].get("state")
            status["finished"] = record["ts"]
        tmp = self.status_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(status, fh, ensure_ascii=False, default=str)
        os.replace(tmp, self.status_path)


def _plain_state(stage: Any) -> str | None:
    """``PipelineState.ANALYZING`` -> ``"ANALYZING"``; strings pass through."""
    if stage is None:
        return None
    value = getattr(stage, "value", stage)
    text = str(value)
    return text.split(".", 1)[1] if text.startswith("PipelineState.") else text


def emit(ctx: Any, type: str, **kwargs: Any) -> None:
    """Emit through ``ctx.event_sink`` when one is attached; otherwise no-op.

    Agents and helpers call this instead of touching the sink directly, so
    code paths that run without a sink (unit tests, offline scripts) need no
    special casing.
    """
    sink = getattr(ctx, "event_sink", None)
    if sink is None:
        return
    try:
        sink.emit(type, **kwargs)
    except Exception:  # noqa: BLE001
        pass


class EventLogList(list):
    """A ``ctx.log`` list that also mirrors each appended entry as an event.

    About thirty call sites append dicts like ``{"timestamp", "agent",
    "message"}`` to ``ctx.log``; before this, those notes reached disk only
    when a stage finished and the checkpoint was saved. Wrapping the list
    surfaces them immediately without editing every call site.

    Token-accounting rows (entries with ``tokens_used``) are skipped: the
    ``llm.end`` event already carries them.
    """

    def __init__(self, iterable: Any = (), sink: EventSink | None = None) -> None:
        super().__init__(iterable)
        self._sink = sink

    def append(self, item: Any) -> None:  # type: ignore[override]
        super().append(item)
        sink = self._sink
        if sink is None:
            return
        try:
            if isinstance(item, dict):
                if "tokens_used" in item and "message" not in item:
                    return
                sink.emit(
                    "agent.note",
                    agent=str(item.get("agent")) if item.get("agent") is not None else None,
                    plain=None,
                    message=str(item.get("message", ""))[:2000],
                )
            else:
                sink.emit("agent.note", message=str(item)[:2000])
        except Exception:  # noqa: BLE001
            pass


def attach(ctx: Any, run_dir: str) -> EventSink | None:
    """Attach a sink to ``ctx`` and wrap ``ctx.log``. Safe to call again
    after a checkpoint load replaced ``ctx.log`` with a plain list."""
    try:
        sink = getattr(ctx, "event_sink", None)
        if sink is None or getattr(sink, "run_dir", None) != run_dir:
            sink = EventSink(run_dir)
            ctx.event_sink = sink
        if not isinstance(ctx.log, EventLogList) or ctx.log._sink is not sink:
            ctx.log = EventLogList(ctx.log, sink=sink)
        return sink
    except Exception:  # noqa: BLE001
        return None
