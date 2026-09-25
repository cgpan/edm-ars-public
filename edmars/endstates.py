"""How a study ended, in words a researcher can act on.

``classify(run_dir)`` reads the run folder and returns an
:class:`Outcome`: a label ("Ready", "Ready, below the review benchmark",
"Stopped" ...), what happened, why, what to do and the exact command.

Sources, most trusted first:

1. ``run_status.json`` schema 2 (``state``, ``reason_code``, ``abort``,
   ``gate``), written by pipelines that carry the release-honesty fixes;
2. the event stream / ``RunState`` (``run.end``, ``error`` events);
3. for older runs: ``run_status.json`` v1, ``checkpoint.json`` errors,
   and the tails of ``pipeline.log`` / ``console.log`` / ``crash.log``,
   matched against patterns for the common failures.

The wording lives in ``edmars/messages.yaml``. The labels never say
"Release: YES": a run with open critical findings is "Ready, with N
serious issues to check", which is what it is.
"""
from __future__ import annotations

import os
import re
import shlex
import string
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from edmars.runstate import (
    RunState,
    as_dict,
    fmt_score,
    load_json,
    load_state,
    parse_ts,
    stage_title,
    step_position,
)

_MESSAGES_PATH = Path(__file__).with_name("messages.yaml")

READY_KINDS = frozenset({"ready", "ready_with_issues"})


@dataclass
class Outcome:
    """The end screen for one run."""

    label: str
    headline: str
    why: str
    fix: str
    command: str | None
    kind: str  # ready | ready_with_issues | not_ready | stopped | running
    code: str | None
    findings: list[dict[str, Any]] = field(default_factory=list)
    #: Other things the reader should check (unresolved review concerns,
    #: a skipped review, few related papers found ...), one plain line each.
    concerns: list[str] = field(default_factory=list)
    title: str = ""
    final_state: str | None = None
    resumable: bool | None = None
    run_dir: str = ""

    @property
    def commands(self) -> list[str]:
        return [c for c in (self.command or "").splitlines() if c.strip()]


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_messages() -> dict[str, Any]:
    with open(_MESSAGES_PATH, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data if isinstance(data, dict) else {}


def messages() -> dict[str, Any]:
    """The parsed ``edmars/messages.yaml`` (cached; do not mutate)."""
    return _load_messages()


class _Keep(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def fill(template: Any, **values: Any) -> str:
    """Fill ``{placeholders}``; unknown ones are left visible, never an error."""
    if not isinstance(template, str):
        return ""
    try:
        return string.Formatter().vformat(template, (), _Keep(**{k: v for k, v in values.items() if v is not None}))
    except (ValueError, IndexError):
        return template


def failure_entry(code: str | None) -> dict[str, str]:
    catalog = messages().get("failures") or {}
    entry = catalog.get(code or "") if isinstance(catalog, dict) else None
    if not isinstance(entry, dict):
        entry = catalog.get("UNKNOWN") if isinstance(catalog, dict) else None
    return entry if isinstance(entry, dict) else {}


def invariant_title(code: str) -> str:
    titles = messages().get("invariants") or {}
    title = titles.get(code) if isinstance(titles, dict) else None
    return str(title) if title else code


def quote_path(path: Path | str) -> str:
    """A path as it should be typed in a terminal command."""
    text = str(path)
    if re.fullmatch(r"[A-Za-z0-9_./:\\-]+", text):
        return text
    if os.name == "nt":
        return f'"{text}"'
    return shlex.quote(text)


# ---------------------------------------------------------------------------
# Failure codes from free text (older runs)
# ---------------------------------------------------------------------------

#: Pipeline-specific markers first: their messages can quote arbitrary
#: warnings, so a provider-looking word inside them must not win.
_PIPELINE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("SAMPLE_TOO_SMALL", re.compile(r"analytic_n=\d+ < 1000")),
    ("DATA_CONTRACT_FAILED", re.compile(r"causal data contract", re.I)),
    ("DE_VALIDATION_FAILED", re.compile(r"validation_passed=False")),
    ("PRE_CRITIC_ABORT", re.compile(r"Pre-Critic guard issued ABORT|short-circuit verdict: ABORT")),
    ("CRITIC_ABORT", re.compile(r"Critic issued ABORT verdict|Critic verdict: ABORT")),
]

_ENVIRONMENT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("NO_CREDIT", re.compile(
        r"insufficient[ _]balance|error code: 402|payment required|insufficient_quota", re.I)),
    ("KEY_MISSING", re.compile(
        r"api[_ ]key[^\n]{0,60}\b(not set|missing|is required|must be set)|no api key", re.I)),
    ("KEY_REJECTED", re.compile(
        r"error code: 40[13]\b|authentication[ _]?(fails|failed|error)|invalid[ _]api[ _]key|"
        r"incorrect api key|invalid x-api-key", re.I)),
    ("MODEL_GONE", re.compile(
        r"model[_ ]not[_ ]found|the model [`'\"]?[\w.-]+[`'\"]? does not exist|unknown model", re.I)),
    ("RATE_LIMITED", re.compile(r"error code: 429|rate[ _]limit|too many requests", re.I)),
    ("R_PACKAGES_MISSING", re.compile(r"there is no package called", re.I)),
    ("R_MISSING", re.compile(
        r"rscript[^\n]{0,80}(not found|no such file|cannot find)|could not find rscript", re.I)),
    ("DATA_MISSING", re.compile(
        r"(filenotfounderror|no such file or directory)[^\n]{0,300}\.csv", re.I)),
    ("LLM_OUTPUT_UNPARSEABLE", re.compile(
        r"jsondecodeerror|expecting value: line|unterminated string|no python code block|"
        r"LLM_OUTPUT_UNPARSEABLE", re.I)),
    ("NETWORK", re.compile(
        r"apiconnectionerror|connection ?error|failed to establish a new connection|"
        r"name or service not known|getaddrinfo failed|temporary failure in name resolution|"
        r"connection (refused|reset|aborted)|network is unreachable", re.I)),
    ("TIMEOUT", re.compile(r"timed out|timeout after|apitimeouterror|readtimeout", re.I)),
    ("PROVIDER_ERROR", re.compile(
        r"error code: 5\d\d|internal server error|service unavailable|server is overloaded",
        re.I)),
]

_STAGE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("ANALYSIS_FAILED", re.compile(r"ANALYZING failed")),
    ("INTERRUPTED", re.compile(r"KeyboardInterrupt")),
]


#: A code the pipeline wrote itself ("code": "NO_CREDIT", code=NO_CREDIT).
_EXPLICIT_CODE = re.compile(r"""\bcode["']?\s*[:=]\s*["']?([A-Z][A-Z_]{3,})\b""")


def code_from_text(text: str) -> str | None:
    """Best guess at a failure code from an error message or log tail."""
    if not text:
        return None
    catalog = messages().get("failures") or {}
    for m in _EXPLICIT_CODE.finditer(text):
        if m.group(1) in catalog:
            return m.group(1)
    for group in (_PIPELINE_PATTERNS, _ENVIRONMENT_PATTERNS, _STAGE_PATTERNS):
        for code, pattern in group:
            if pattern.search(text):
                return code
    return None


def _tail(path: Path, max_bytes: int = 16_000) -> str:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - max_bytes))
            return fh.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _last_abort_line(log_tail: str) -> str | None:
    for line in reversed(log_tail.splitlines()):
        m = re.search(r"\] ABORTED: (.*)$", line)
        if m:
            return m.group(1).strip()
    return None


def _abort_details(run_dir: Path, state: RunState, status: dict[str, Any] | None) -> tuple[str, str, str | None]:
    """(code, message, stage) for a run that stopped."""
    abort = status.get("abort") if isinstance(status, dict) else None
    if not isinstance(abort, dict):
        abort = state.abort if isinstance(state.abort, dict) else None
    message = ""
    stage = None
    if isinstance(abort, dict):
        message = str(abort.get("message") or "")
        stage = abort.get("stage")
        code = abort.get("code")
        if isinstance(code, str) and code:
            if code == "CRITIC_ABORT" and _mentions_leakage(run_dir):
                return "LEAKAGE_SUSPECTED", message, stage
            return code, message, stage

    checkpoint = load_json(run_dir / "checkpoint.json")
    errors = checkpoint.get("errors") if isinstance(checkpoint, dict) else None
    texts: list[str] = []
    if isinstance(errors, list):
        texts.extend(str(e) for e in errors[-3:])
    log_tail = _tail(run_dir / "pipeline.log")
    abort_line = _last_abort_line(log_tail)
    if abort_line:
        texts.insert(0, abort_line)
        message = message or abort_line
    if not message and texts:
        message = texts[-1]
    for extra in ("crash.log", "console.log"):
        texts.append(_tail(run_dir / extra, 8000))
    texts.append(log_tail[-4000:])

    code = None
    for text in texts:
        code = code_from_text(text)
        if code:
            break
    if code == "CRITIC_ABORT" and _mentions_leakage(run_dir):
        code = "LEAKAGE_SUSPECTED"
    if stage is None and message:
        m = re.match(r"([A-Z]+) (?:failed|aborted)", message)
        if m:
            stage = m.group(1)
    return code or "UNKNOWN", message, stage


def _mentions_leakage(run_dir: Path) -> bool:
    review = load_json(run_dir / "review_report.json")
    if not isinstance(review, dict):
        return False
    text = str(review).lower()
    return "leakage" in text or "leak " in text or "too good to be true" in text


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------

_SEVERITY_ORDER = {"critical": 0, "major": 1, "minor": 2}


def load_findings(run_dir: Path, status: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Critical and major invariant findings, one entry per code."""
    inv = load_json(Path(run_dir) / "invariants.json")
    by_code: dict[str, dict[str, Any]] = {}
    if isinstance(inv, dict) and isinstance(inv.get("findings"), list):
        for f in inv["findings"]:
            if not isinstance(f, dict):
                continue
            code = str(f.get("code") or "UNKNOWN_CHECK")
            severity = str(f.get("severity") or "").lower()
            if severity not in ("critical", "major"):
                continue
            entry = by_code.get(code)
            if entry is None:
                by_code[code] = {
                    "code": code,
                    "severity": severity,
                    "title": invariant_title(code),
                    "message": str(f.get("message") or ""),
                    "count": 1,
                }
            else:
                entry["count"] += 1
                if _SEVERITY_ORDER.get(severity, 3) < _SEVERITY_ORDER.get(entry["severity"], 3):
                    entry["severity"] = severity
    elif isinstance(status, dict) and isinstance(status.get("invariant_codes"), list):
        # No per-finding file: codes only, severity unknown.
        for code in status["invariant_codes"]:
            code = str(code)
            by_code.setdefault(code, {
                "code": code, "severity": "unknown", "title": invariant_title(code),
                "message": "", "count": 1,
            })
    return sorted(by_code.values(), key=lambda f: (_SEVERITY_ORDER.get(f["severity"], 3), f["code"]))


# ---------------------------------------------------------------------------
# run_status.json
# ---------------------------------------------------------------------------


def _status(run_dir: Path, state: RunState) -> dict[str, Any] | None:
    """run_status.json when it belongs to the latest (re)start of the run."""
    path = run_dir / "run_status.json"
    status = load_json(path)
    if not isinstance(status, dict):
        return None
    runner = as_dict(load_json(run_dir / "runner.json"))
    starts = [t for t in (parse_ts(runner.get("resumed_at")), parse_ts(runner.get("started_at")),
                          state.run_started_at) if t]
    if starts:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = None
        if mtime is not None and mtime < max(starts).timestamp() - 1:
            return None  # left over from before the latest (re)start
    return status


def _gate_facts(run_dir: Path, state: RunState, status: dict[str, Any] | None) -> dict[str, Any]:
    """Whether the LSAR gate was wanted, whether it ran, and its verdict."""
    gate: dict[str, Any] = {}
    if isinstance(status, dict) and isinstance(status.get("gate"), dict):
        gate.update(status["gate"])
    else:
        m = state.metrics
        cfg = load_json(run_dir / "runner.json")
        study = cfg.get("study") if isinstance(cfg, dict) else None
        enabled = state.lsar_enabled
        gate["enabled"] = bool(enabled)
        gate["ran"] = m.get("gate_ran") if m.get("gate_ran") is not None else (
            m.get("gate_score") is not None
        )
        gate["passed"] = m.get("gate_passed")
        gate["score"] = m.get("gate_score")
        gate["threshold"] = m.get("gate_threshold")
        gate["advisory"] = m.get("gate_advisory")
        gate["skip_reason"] = m.get("gate_skip_reason")
        gate["venue"] = (state.gate or {}).get("venue")
        if enabled and not gate["ran"] and not gate.get("skip_reason"):
            gate["skip_reason"] = "exception"
        if isinstance(study, dict) and study.get("review_unavailable"):
            gate["enabled"] = True
            gate["ran"] = False
            gate["skip_reason"] = "not_available"
        if isinstance(status, dict) and status.get("review_gate_passed") is not None and gate["ran"]:
            gate["passed"] = bool(status.get("review_gate_passed"))
    if gate.get("ran") is False:
        gate["score"] = None  # a gate that did not run has no score
    return gate


def _gate_skip_text(reason: Any) -> str:
    text = str(reason or "")
    table = messages().get("gate_skip_reasons") or {}
    key = text.split(":", 1)[0].strip()
    plain = table.get(key) if isinstance(table, dict) else None
    return str(plain) if plain else (text or "reason not recorded")


def _gate_code(reason: Any) -> str:
    text = str(reason or "")
    if text.startswith(("lsar_not_found", "lsar_import_failed", "not_available")):
        return "LSAR_MISSING"
    return "LSAR_FAILED"


def _latex_missing(run_dir: Path, state: RunState) -> bool:
    if state.compile and state.compile.get("missing_tool"):
        return True
    tail = _tail(run_dir / "pipeline.log", 64_000)
    return bool(re.search(r"'(pdflatex|bibtex|biber)' not found", tail))


# ---------------------------------------------------------------------------
# classify
# ---------------------------------------------------------------------------


def _ctx(run_dir: Path, state: RunState, **extra: Any) -> dict[str, Any]:
    providers = messages().get("providers") or {}
    provider = providers.get(state.provider, state.provider) if isinstance(providers, dict) else state.provider
    return {
        "run": quote_path(run_dir),
        "provider": provider or "the AI service",
        "dataset": state.dataset or "hsls09_public",
        **extra,
    }


#: Codes a resume cannot fix: the question or the data must change.
#: Mirrors ``resumable: False`` in src/errors.py, plus the CLI-only
#: LEAKAGE_SUSPECTED; kept here so the CLI works without importing the
#: pipeline package.
_NOT_RESUMABLE = frozenset({"SAMPLE_TOO_SMALL", "PRE_CRITIC_ABORT", "CRITIC_ABORT", "LEAKAGE_SUSPECTED"})


def _resumable(code: str, status: dict[str, Any] | None) -> bool:
    abort = as_dict(status.get("abort")) if isinstance(status, dict) else {}
    if abort.get("code") == code and isinstance(abort.get("resumable"), bool):
        return bool(abort["resumable"])
    return code not in _NOT_RESUMABLE


def _stopped(run_dir: Path, state: RunState, code: str, message: str,
             stage: str | None, final: str | None,
             status: dict[str, Any] | None = None) -> Outcome:
    entry = failure_entry(code)
    ctx = _ctx(run_dir, state, reason=message or "no details recorded", stage=stage)
    title = fill(entry.get("title"), **ctx) or code
    labels = messages().get("outcomes") or {}
    resumable = _resumable(code, status)
    headline = title
    if stage:
        headline = f"{title} (during: {stage.lower()})"
    return Outcome(
        label=str(labels.get("stopped", "Stopped")),
        headline=headline,
        why=fill(entry.get("why"), **ctx),
        fix=fill(entry.get("fix"), **ctx),
        command=fill(entry.get("command"), **ctx) or None,
        kind="stopped",
        code=code,
        title=title,
        final_state=final,
        resumable=resumable,
        run_dir=str(run_dir),
    )


def classify(run_dir: Path | str) -> Outcome:
    """Decide how the run in ``run_dir`` ended (or that it is running)."""
    run_dir = Path(run_dir)
    labels = messages().get("outcomes") or {}
    if not run_dir.is_dir():
        return Outcome(
            label="Not found",
            headline=f"There is no study folder at {run_dir}.",
            why="The folder does not exist or was moved.",
            fix="List your studies to find the right one.",
            command="edmars runs",
            kind="stopped",
            code=None,
            run_dir=str(run_dir),
        )
    state = load_state(run_dir)

    if not state.finished:
        step, total = step_position(state)
        running = next((s for s in state.stages if s.status == "running"), None)
        what = stage_title(state, running) if running else "starting"
        return Outcome(
            label=str(labels.get("running", "Still running")),
            headline=f"This study is still running (step {step} of {total}: {what}).",
            why="",
            fix="Watch it live, or come back later. It keeps running if you close this window.",
            command=f"edmars status {quote_path(run_dir)}",
            kind="running",
            code=None,
            final_state=None,
            run_dir=str(run_dir),
        )

    status = _status(run_dir, state)
    final = (str(status.get("state")).upper() if isinstance(status, dict) and status.get("state") else None) \
        or state.final_state
    reason_code = (status.get("reason_code") if isinstance(status, dict) else None) or state.reason_code

    if final == "STOPPED":
        return _stopped(run_dir, state, "INTERRUPTED", "You stopped the study.", None, final)
    if final == "CRASHED":
        code, message, stage = _abort_details(run_dir, state, status)
        if code == "UNKNOWN":
            code = "CRASHED"
        return _stopped(run_dir, state, code, message, stage, final)
    if final == "INTERRUPTED" or reason_code == "INTERRUPTED":
        abort = status.get("abort") if isinstance(status, dict) else None
        stage = abort.get("stage") if isinstance(abort, dict) else None
        return _stopped(run_dir, state, "INTERRUPTED", "", stage, "INTERRUPTED", status)
    if final == "ABORTED" or reason_code == "ABORTED":
        code, message, stage = _abort_details(run_dir, state, status)
        return _stopped(run_dir, state, code, message, stage, "ABORTED", status)

    findings = load_findings(run_dir, status)
    released = status.get("released") if isinstance(status, dict) else state.released
    if final == "INCOMPLETE" or released is False:
        return _not_ready(run_dir, state, status, findings, reason_code)
    if not (run_dir / "paper.pdf").exists():
        # Released pipelines can mark a run clean although pdflatex never
        # ran (no paper.log, so the no-PDF check had nothing to read).
        # Without a PDF there is no paper to hand over: never "Ready".
        return _not_ready(run_dir, state, status, findings, reason_code, pdf_missing=True)
    return _ready(run_dir, state, status, findings, reason_code)


def _not_ready(run_dir: Path, state: RunState, status: dict[str, Any] | None,
               findings: list[dict[str, Any]], reason_code: str | None,
               *, pdf_missing: bool = False) -> Outcome:
    labels = messages().get("outcomes") or {}
    blocking: list[str] = []
    if isinstance(status, dict) and isinstance(status.get("blocking_findings"), list):
        blocking = [str(c) for c in status["blocking_findings"]]
    if not blocking:
        checkpoint = load_json(run_dir / "checkpoint.json")
        for err in (checkpoint.get("errors") or []) if isinstance(checkpoint, dict) else []:
            m = re.search(r"Release blocked by .*?: (.*)$", str(err))
            if m:
                blocking = [c.strip() for c in m.group(1).split(",") if c.strip()]
    if pdf_missing or "INV_LATEX_NO_PDF" in blocking or (
        not blocking and not (run_dir / "paper.pdf").exists() and (run_dir / "paper.tex").exists()
    ):
        code = "LATEX_MISSING" if _latex_missing(run_dir, state) else "NO_PDF"
    elif reason_code == "VERIFICATION_NOT_RUN":
        code = "VERIFICATION_NOT_RUN"
    else:
        code = "BLOCKING_FINDINGS"
    reason = ", ".join(invariant_title(c) for c in blocking) or "see invariants.json"
    entry = failure_entry(code)
    ctx = _ctx(run_dir, state, reason=reason)
    title = fill(entry.get("title"), **ctx)
    concerns = _concerns(run_dir, state, status, skip_gate=False)
    return Outcome(
        label=str(labels.get("not_ready", "Not ready")),
        headline=title,
        why=fill(entry.get("why"), **ctx),
        fix=fill(entry.get("fix"), **ctx),
        command=fill(entry.get("command"), **ctx) or None,
        kind="not_ready",
        code=code,
        findings=findings,
        concerns=concerns,
        title=title,
        final_state="INCOMPLETE",
        resumable=False,
        run_dir=str(run_dir),
    )


def _concerns(run_dir: Path, state: RunState, status: dict[str, Any] | None,
              *, skip_gate: bool) -> list[str]:
    """Plain lines for the "Please check" list besides invariant findings."""
    out: list[str] = []
    unverified = (status.get("critic_unverified") if isinstance(status, dict) else None)
    if unverified is None:
        unverified = state.metrics.get("critic_unverified")
    if unverified:
        out.append(
            "The internal methods reviewer's concerns were not all resolved; "
            "the paper carries a warning box. Read review_report.json."
        )
    if not skip_gate:
        gate = _gate_facts(run_dir, state, status)
        if gate.get("enabled") and gate.get("ran") is False:
            out.append(f"Automated peer review did not run: {_gate_skip_text(gate.get('skip_reason'))}")
        elif gate.get("ran") and gate.get("passed") is False and not gate.get("advisory"):
            out.append(_gate_sentence(gate))
    lit = status.get("literature") if isinstance(status, dict) else None
    if isinstance(lit, dict) and lit.get("degraded"):
        out.append(
            "Few related papers were found (the literature search was partly "
            "unavailable). Check the related-work section and the references."
        )
    return out


def _gate_sentence(gate: dict[str, Any]) -> str:
    score = gate.get("score")
    threshold = gate.get("threshold")
    venue = gate.get("venue") or "the venue"
    if score is None:
        return "Automated peer review: no score."
    if gate.get("advisory"):
        return (f"Automated peer review (LSAR) scored the paper {fmt_score(score)} out of 10. "
                f"There is no benchmark for {venue}, so this is a score only.")
    if threshold is None:
        return f"Automated peer review (LSAR) scored the paper {fmt_score(score)} out of 10."
    side = "at or above" if gate.get("passed") else "below"
    return (f"Automated peer review (LSAR) scored the paper {fmt_score(score)} out of 10, "
            f"{side} the benchmark of {fmt_score(threshold)} for {venue}. Scores vary by "
            f"about 2 points between reviews.")


def _ready(run_dir: Path, state: RunState, status: dict[str, Any] | None,
           findings: list[dict[str, Any]], reason_code: str | None) -> Outcome:
    labels = messages().get("outcomes") or {}
    gate = _gate_facts(run_dir, state, status)
    counts: dict[str, Any] = {}
    if isinstance(status, dict) and isinstance(status.get("invariant_counts"), dict):
        counts = status["invariant_counts"]
    elif isinstance(state.metrics.get("invariant_counts"), dict):
        counts = state.metrics["invariant_counts"]
    crit = counts.get("critical")
    if isinstance(crit, (int, float)) and not isinstance(crit, bool):
        n_critical = int(crit)
    else:
        n_critical = sum(1 for f in findings if f.get("severity") == "critical")
    unverified = status.get("critic_unverified") if isinstance(status, dict) else None
    if unverified is None:
        unverified = bool(state.metrics.get("critic_unverified"))

    code = reason_code if reason_code in (
        "CLEAN", "ADVISORY_FINDINGS", "CRITIC_UNVERIFIED", "GATE_FAILED",
        "GATE_NOT_RUN", "VERIFICATION_NOT_RUN",
    ) else None
    if code is None:
        inv = load_json(run_dir / "invariants.json")
        if isinstance(inv, dict) and (inv.get("error") or inv.get("enabled") is False):
            code = "VERIFICATION_NOT_RUN"  # a crashed battery is not a clean run
        elif not (run_dir / "invariants.json").exists():
            code = "VERIFICATION_NOT_RUN"  # a pipeline older than the checks
        if code is None:
            if n_critical > 0:
                code = "ADVISORY_FINDINGS"
            elif gate.get("ran") and gate.get("passed") is False and not gate.get("advisory"):
                code = "GATE_FAILED"
            elif unverified:
                code = "CRITIC_UNVERIFIED"
            elif gate.get("enabled") and gate.get("ran") is False:
                code = "GATE_NOT_RUN"
            else:
                code = "CLEAN"
    if code == "GATE_NOT_RUN" and gate.get("enabled") is False:
        code = "CLEAN"  # switched off by choice: nothing to warn about

    concerns = _concerns(run_dir, state, status, skip_gate=False)
    ctx = _ctx(run_dir, state)
    open_cmd = f"edmars results {quote_path(run_dir)} --open pdf"
    fix = "Open the paper and check it, starting with the list above."
    command: str | None = open_cmd
    out_code: str | None = None
    kind = "ready_with_issues"

    if code == "CLEAN":
        label = str(labels.get("ready", "Ready"))
        kind = "ready"
        headline = "Your paper is written and passed the final checks."
        why = ""
        fix = "Read the paper carefully before sharing it."
    elif code == "ADVISORY_FINDINGS":
        n = n_critical or sum(1 for f in findings if f.get("severity") == "critical")
        if n:
            key = "serious_one" if n == 1 else "serious_many"
        else:  # advisory findings below critical
            n = len(findings) or 1
            key = "issues_one" if n == 1 else "issues_many"
        label = fill(labels.get(key, "Ready, with {n} issues to check"), n=n)
        headline = "Your paper is written, but the final checks found problems to fix before sharing it."
        why = ("The final checks compare the paper with the study's own numbers, "
               "figures and citations. Each item below is something the paper states "
               "that its own results do not support, or a part that is missing.")
    elif code == "GATE_FAILED":
        label = str(labels.get("gate_failed", "Ready, below the review benchmark"))
        headline = "Your paper is written; the automated reviewer scored it below the benchmark."
        why = _gate_sentence(gate)
        fix = ("Read the review in the lsar_review folder. The score is a rough, noisy "
               "signal, not a prediction of acceptance.")
    elif code == "CRITIC_UNVERIFIED":
        label = str(labels.get("unverified", "Ready, with unresolved concerns"))
        headline = "Your paper is written, but the internal reviewer's concerns were not all resolved."
        why = ("The methods reviewer asked for changes and the allowed revision rounds "
               "ran out. The paper opens with a warning box and lists the concerns in an appendix.")
        fix = "Read review_report.json and check the paper against each concern."
    elif code == "GATE_NOT_RUN":
        label = str(labels.get("not_reviewed", "Ready, not reviewed"))
        reason = _gate_skip_text(gate.get("skip_reason"))
        headline = f"Your paper is written, but the automated peer review did not run: {reason}"
        out_code = _gate_code(gate.get("skip_reason"))
        entry = failure_entry(out_code)
        why = fill(entry.get("why"), **ctx)
        fix = fill(entry.get("fix"), **ctx)
        command = fill(entry.get("command"), **ctx) or open_cmd
        concerns = [c for c in concerns if not c.startswith("Automated peer review did not run")]
    elif code == "VERIFICATION_NOT_RUN":
        label = str(labels.get("not_checked", "Ready, final checks did not run"))
        entry = failure_entry("VERIFICATION_NOT_RUN")
        headline = "Your paper is written, but the final checks did not run."
        why = fill(entry.get("why"), **ctx)
        fix = fill(entry.get("fix"), **ctx)
        out_code = "VERIFICATION_NOT_RUN"
    else:  # pragma: no cover - exhaustive above
        label = str(labels.get("ready", "Ready"))
        headline = "Your paper is written."
        why = ""

    if kind == "ready" and (findings or concerns):
        # Clean release, but majors or other notes remain worth a look.
        fix = "Read the paper carefully, starting with the items listed above."
    return Outcome(
        label=label,
        headline=headline,
        why=why,
        fix=fix,
        command=command,
        kind=kind,
        code=out_code or (None if code == "CLEAN" else code),
        findings=findings,
        concerns=concerns,
        title=label,
        final_state="COMPLETED",
        resumable=None,
        run_dir=str(run_dir),
    )


__all__ = [
    "Outcome",
    "READY_KINDS",
    "classify",
    "code_from_text",
    "failure_entry",
    "fill",
    "invariant_title",
    "load_findings",
    "messages",
    "quote_path",
]
