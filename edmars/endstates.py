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
    literature_notes,
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
    #: For a study the checks or the reviewer stopped: what they found, one
    #: plain line each, and the heading to show above them.
    details: list[str] = field(default_factory=list)
    details_heading: str = ""
    #: A further line, such as the question as the study worded it.
    note: str = ""

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
    """A path as it should be typed in a terminal command.

    A backslash is never left bare: Git Bash drops it (D:\\studies\\x arrives
    as D:studiesx), and cmd, PowerShell and Git Bash all keep a double-quoted
    Windows path intact.
    """
    text = str(path)
    if re.fullmatch(r"[A-Za-z0-9_./:-]+", text):
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
    # Before PRE_CRITIC_ABORT: such a run also logs "short-circuit verdict: ABORT".
    ("PRE_CRITIC_UNRESOLVED", re.compile(
        r"PRE_CRITIC_UNRESOLVED|still failing when the revision cycles ran out")),
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
    ("INSTALL_BROKEN", re.compile(r"ModuleNotFoundError: No module named|DLL load failed")),
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


_KEY_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_*-]{8,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"(?i)(api[_ -]?key[\"':= ]+)[A-Za-z0-9._*-]{8,}"),
)


def redact(text: str) -> str:
    """Remove anything key-shaped from text quoted out of a log.

    Provider errors can echo part of the key ("Incorrect API key provided:
    sk-proj-****abcd"). The shared redactor in edmars.secrets also knows
    the exact stored values; the patterns here are the fallback.
    """
    try:
        from edmars.secrets import redact as shared_redact

        text = shared_redact(text)
    except Exception:  # noqa: BLE001 -- the fallback below still runs
        pass
    for pattern in _KEY_PATTERNS:
        text = pattern.sub(lambda m: (m.group(1) if m.lastindex else "") + "[redacted]", text)
    return text


def _tail(path: Path, max_bytes: int = 16_000) -> str:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - max_bytes))
            return fh.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


_ERROR_LINE = re.compile(r"^(?:[\w.]+\.)?\w+(?:Error|Exception|Interrupt)\b:?.*$")


def last_error_line(text: str) -> str | None:
    """The last ``SomethingError: message`` line of a traceback, if any."""
    for line in reversed(text.splitlines()):
        line = line.strip()
        if _ERROR_LINE.match(line):
            return line[:300]
    return None


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
                return "LEAKAGE_SUSPECTED", redact(message), stage
            return code, redact(message), stage

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
        tail = _tail(run_dir / extra, 8000)
        texts.append(tail)
        if not message:
            message = last_error_line(tail) or ""
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
    return code or "UNKNOWN", redact(message), stage


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


def gate_skip_text(reason: Any) -> str:
    """Why the automated peer review did not run, in plain words."""
    text = str(reason or "")
    table = messages().get("gate_skip_reasons") or {}
    key = text.split(":", 1)[0].strip()
    plain = table.get(key) if isinstance(table, dict) else None
    return str(plain) if plain else (text or "reason not recorded")


def _gate_code(reason: Any) -> str:
    text = str(reason or "")
    if text.startswith(("lsar_not_found", "lsar_import_failed", "not_available")):
        return "LSAR_MISSING"
    if text.startswith("lsar_scoring_failed"):
        return "LSAR_SCORING_FAILED"
    return "LSAR_FAILED"


def _latex_missing(run_dir: Path, state: RunState) -> bool:
    if state.compile and state.compile.get("missing_tool"):
        return True
    tail = _tail(run_dir / "pipeline.log", 64_000)
    return bool(re.search(r"'(pdflatex|bibtex|biber)' not found", tail))


# ---------------------------------------------------------------------------
# classify
# ---------------------------------------------------------------------------


def _get_data_command(dataset: str) -> str:
    """install, or import for a dataset EDM-ARS cannot download yet."""
    try:
        from edmars.datasets import get_command
    except Exception:  # noqa: BLE001 -- the wording must never crash a screen
        return f"edmars data install {dataset}"
    return get_command(dataset)


def _ctx(run_dir: Path, state: RunState, **extra: Any) -> dict[str, Any]:
    providers = messages().get("providers") or {}
    provider = providers.get(state.provider, state.provider) if isinstance(providers, dict) else state.provider
    return {
        "run": quote_path(run_dir),
        "provider": provider or "the AI service",
        "dataset": state.dataset or "hsls09_public",
        "get_data": _get_data_command(state.dataset or "hsls09_public"),
        **extra,
    }


#: Codes a resume cannot fix: the question or the data must change.
#: Mirrors ``resumable: False`` in src/errors.py, plus the CLI-only
#: LEAKAGE_SUSPECTED; kept here so the CLI works without importing the
#: pipeline package. PRE_CRITIC_UNRESOLVED (a revisable pre-review finding
#: still failing when the revision rounds ran out) is here before the
#: pipeline has it, so a log-only reading never tells the user to resume.
_NOT_RESUMABLE = frozenset({"SAMPLE_TOO_SMALL", "PRE_CRITIC_ABORT", "PRE_CRITIC_UNRESOLVED",
                            "CRITIC_ABORT", "LEAKAGE_SUSPECTED"})


#: Checks whose critical finding the pipeline sends back for revision
#: rather than stopping on (``REVISABLE_PRE_CRITIC_CHECKS`` in
#: src/errors.py).
_REVISABLE_CHECKS = frozenset({"pcc_02", "pcc_07"})
_LEAD_CHECK = re.compile(r"^(pcc_[a-z0-9]+):")


def _reopened(code: str, status: dict[str, Any] | None) -> bool:
    """A PRE_CRITIC_ABORT the pipeline now resumes into a revision.

    Mirrors ``reopened_pre_critic_stop`` in src/errors.py: a stop written
    before the checks told findings a revision can fix from ones it
    cannot, recognised by having no ``abort.checks`` (every later one has
    them) and a message led by pcc_07 or pcc_02. The Mac study that
    prompted fix/pcc-revise is one; its run_status.json says
    ``resumable: false``, which that release made untrue.
    """
    abort = as_dict(status.get("abort")) if isinstance(status, dict) else {}
    if code != "PRE_CRITIC_ABORT" or abort.get("code") != code:
        return False
    if abort.get("stage") != "CRITIQUING" or "checks" in abort:
        return False
    lead = _LEAD_CHECK.match(str(abort.get("message") or "").strip())
    return lead is not None and lead.group(1) in _REVISABLE_CHECKS


def _resumable(code: str, status: dict[str, Any] | None) -> bool:
    abort = as_dict(status.get("abort")) if isinstance(status, dict) else {}
    if _reopened(code, status):
        return True
    if abort.get("code") == code and isinstance(abort.get("resumable"), bool):
        return bool(abort["resumable"])
    return code not in _NOT_RESUMABLE


def step_words(stage: str | None) -> str | None:
    """A pipeline state name as the step's title in the step list, for use
    mid-sentence: ENGINEERING -> "preparing the data"."""
    if not stage:
        return None
    stages = messages().get("stages") or {}
    entry = stages.get(str(stage).upper()) if isinstance(stages, dict) else None
    title = entry.get("title") if isinstance(entry, dict) else None
    if not title:
        return str(stage).replace("_", " ").lower()
    title = str(title)
    return title[:1].lower() + title[1:]


#: Abort codes whose message is a check's or the reviewer's finding, worth
#: showing in full on the result screen.
_REVIEW_ABORTS = frozenset({"PRE_CRITIC_ABORT", "PRE_CRITIC_UNRESOLVED", "CRITIC_ABORT", "LEAKAGE_SUSPECTED"})

#: The stops the automatic pre-review checks make (src/pre_critic_checks.py).
_PRE_CRITIC_CODES = frozenset({"PRE_CRITIC_ABORT", "PRE_CRITIC_UNRESOLVED"})

#: How src/orchestrator.py words a PRE_CRITIC_UNRESOLVED that came before
#: the revision rounds ran out: the revision returned the agent's own
#: word that another would not fix it ("pcc_07 was still failing after
#: revision 1 of 2, and another revision would not change it: ...").
_STOPPED_EARLY = "another revision would not change it"

#: "pcc_07: <sentence>", or "pcc_07 was still failing when the revision
#: cycles ran out (2 of 2 used): <sentence>" for PRE_CRITIC_UNRESOLVED.
_CHECK_MESSAGE = re.compile(r"^(pcc_[a-z0-9]+)\b[^:]*:\s*(.+)$", re.DOTALL)
_REVIEW_SECTIONS = ("problem_formulation_review", "data_preparation_review",
                    "analysis_review", "substantive_review")


def _latest_review(run_dir: Path) -> dict[str, Any] | None:
    """The review the run ended with: the checkpoint's copy (it also holds
    the automatic checks' short-circuit report, which is never written to
    review_report.json), else review_report.json."""
    checkpoint = load_json(run_dir / "checkpoint.json")
    review = checkpoint.get("review_report") if isinstance(checkpoint, dict) else None
    if isinstance(review, dict):
        return review
    review = load_json(run_dir / "review_report.json")
    return review if isinstance(review, dict) else None


def abort_checks(status: dict[str, Any] | None) -> list[dict[str, Any]]:
    """run_status.json's ``abort.checks``: every critical pre-review
    finding behind a PRE_CRITIC_* stop (check_id, severity, message,
    target_agent, revisable), each message in full. Empty for other stops
    and for runs written before the record had them."""
    abort = as_dict(status.get("abort")) if isinstance(status, dict) else {}
    checks = abort.get("checks")
    if not isinstance(checks, list):
        return []
    return [c for c in checks if isinstance(c, dict) and c.get("message")]


def review_findings(run_dir: Path | str, code: str, message: str,
                    checks: list[dict[str, Any]] | None = None) -> tuple[list[str], list[str]]:
    """(check ids, plain lines) saying what stopped a study at the review.

    ``checks`` is :func:`abort_checks`: when the pipeline recorded them,
    they are the findings, in full, the one the abort message names first.
    Otherwise the abort message gives the first finding (cut to a line)
    and the review the run ended with the rest. Lines are the checks' and
    the reviewer's own sentences, without their "pcc_07:" or "Critic
    ABORT:" prefix, at most five.
    """
    ids: list[str] = []
    lines: list[str] = []

    def add(check_id: Any, text: Any) -> None:
        flat = " ".join(str(text or "").split())
        if not flat:
            return
        if flat not in lines:
            lines.append(flat)
        if isinstance(check_id, str) and check_id.startswith("pcc_") and check_id not in ids:
            ids.append(check_id)

    found = _CHECK_MESSAGE.match((message or "").strip())
    if checks and code in _PRE_CRITIC_CODES:
        lead = found.group(1) if found else None
        for check in sorted(checks, key=lambda c: c.get("check_id") != lead):
            add(check.get("check_id"), redact(str(check.get("message") or "")))
        return ids, lines[:5]
    if found:
        add(found.group(1), found.group(2))
    elif message:
        add(None, re.sub(r"^Critic ABORT:\s*", "", message.strip()))
    review = _latest_review(Path(run_dir))
    if review is not None:
        automatic = review.get("_source") == "pre_critic_short_circuit"
        if automatic == (code in _PRE_CRITIC_CODES):
            for section in _REVIEW_SECTIONS:
                block = review.get(section)
                for issue in (block.get("issues") or []) if isinstance(block, dict) else []:
                    if isinstance(issue, dict) and str(issue.get("severity") or "").lower() == "critical":
                        add(issue.get("category") if automatic else None,
                            redact(str(issue.get("description") or "")))
    return ids, lines[:5]


def _reworded_question(run_dir: Path, state: RunState) -> str | None:
    """The research question the study worked from, when it differs from
    the one the user typed (the first step rewrites it)."""
    for name in ("research_spec.json", "research_spec.locked.json"):
        spec = load_json(run_dir / name)
        worked = spec.get("research_question") if isinstance(spec, dict) else None
        if isinstance(worked, str) and worked.strip():
            if " ".join(worked.split()).casefold() != " ".join(state.question.split()).casefold():
                return " ".join(worked.split())
            return None
    return None


def checks_in_words(ids: list[str]) -> str:
    """What the automatic checks found, for ``{checks}`` in the
    PRE_CRITIC_* "why": each check's ``what`` from messages.yaml, joined
    ("the outcome among the predictors (data leakage) and no trained model
    in the results"), or a pointer to the list for an unknown check."""
    advice = messages().get("pre_critic_checks") or {}
    whats = [str(advice[i]["what"]) for i in ids
             if isinstance(advice, dict) and isinstance(advice.get(i), dict) and advice[i].get("what")]
    if not whats:
        return "a problem with these results (listed below)"
    return whats[0] if len(whats) == 1 else ", ".join(whats[:-1]) + " and " + whats[-1]


def _review_abort_advice(run_dir: Path, state: RunState, code: str, message: str,
                         entry: dict[str, str], ctx: dict[str, Any], resumable: bool,
                         status: dict[str, Any] | None = None,
                         ) -> tuple[list[str], str, str, str, str | None]:
    """(details, heading, note, fix, command) for a study the automatic
    checks or the reviewer stopped: what they found, and advice that fits
    the check instead of "start again with a simpler question".

    For the automatic checks it also sets ``ctx["checks"]``, which the
    "why" names."""
    checks = abort_checks(status)
    ids, details = review_findings(run_dir, code, message, checks)
    heading = ("What the automatic checks found:" if code in _PRE_CRITIC_CODES
               else "What the reviewer found:")
    if code in _PRE_CRITIC_CODES:
        ctx["checks"] = checks_in_words(ids)
    advice = messages().get("pre_critic_checks") or {}
    # A PRE_CRITIC_ABORT is caused by the findings no revision can fix;
    # a revisable one beside it (it would have been sent back) is listed
    # but its advice ("ask for the comparison in your question") is not
    # what to do about this stop.
    stopping = [str(c.get("check_id")) for c in checks if c.get("revisable") is False]
    fix_ids = [i for i in ids if i in stopping] if code == "PRE_CRITIC_ABORT" and stopping else ids
    fixes = [str(advice[i]["fix"]) for i in fix_ids
             if isinstance(advice, dict) and isinstance(advice.get(i), dict) and advice[i].get("fix")]
    fix = " ".join(" ".join(f.split()) for f in fixes) or fill(entry.get("fix"), **ctx)
    stops = messages().get("pre_critic_stops") or {}
    if fixes and code == "PRE_CRITIC_UNRESOLVED":
        early = _STOPPED_EARLY in (message or "")
        lead = stops.get("unresolved_early" if early else "unresolved_rounds_used")
        if lead:
            fix = f"{' '.join(str(lead).split())} {fix}"
    command = fill(entry.get("command"), **ctx) or None
    note = ""
    if "pcc_07" in ids:
        worded = _reworded_question(run_dir, state)
        if worded:
            note = f'The study worded your question as: "{worded}"'
    abort = as_dict(status.get("abort")) if isinstance(status, dict) else {}
    if resumable and _reopened(code, status) and abort.get("resumable") is not True:
        # Stopped by a rule this version no longer has: a resume sends
        # the finding back for revision (src/errors.py).
        lead = " ".join(str(stops.get("reopened") or "").split())
        fix = f"{lead} If it stops again: {fix[:1].lower()}{fix[1:]}".strip()
        command = f"edmars resume {ctx['run']}"
    elif resumable:
        # The pipeline recorded that a resume can deal with it (run_status.json).
        fix = ("Resume the study to let it try again from the step that failed. If it "
               f"stops the same way again: {fix[:1].lower()}{fix[1:]}")
        command = f"edmars resume {ctx['run']}"
    return details, heading, note, fix, command


def _stopped(run_dir: Path, state: RunState, code: str, message: str,
             stage: str | None, final: str | None,
             status: dict[str, Any] | None = None) -> Outcome:
    entry = failure_entry(code)
    step = step_words(stage)
    ctx = _ctx(run_dir, state, reason=message or "no details recorded", stage=step)
    title = fill(entry.get("title"), **ctx) or code
    labels = messages().get("outcomes") or {}
    resumable = _resumable(code, status)
    headline = title
    if step:
        headline = f"{title} (during: {step})"
    fix = fill(entry.get("fix"), **ctx)
    command = fill(entry.get("command"), **ctx) or None
    details: list[str] = []
    heading = note = ""
    if code in _REVIEW_ABORTS:
        details, heading, note, fix, command = _review_abort_advice(
            run_dir, state, code, message, entry, ctx, resumable, status)
    removed = outcome_removed_concern(run_dir, state)
    return Outcome(
        label=str(labels.get("stopped", "Stopped")),
        headline=headline,
        why=fill(entry.get("why"), **ctx),
        fix=fix,
        command=command,
        kind="stopped",
        code=code,
        title=title,
        final_state=final,
        resumable=resumable,
        run_dir=str(run_dir),
        details=details,
        details_heading=heading,
        note=note,
        concerns=[removed] if removed else [],
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


#: data_report.json's record of the pipeline's outcome check after data
#: preparation (src/outcome_guard.py RECORD_KEY): {"after": ..., "removed":
#: {"train_X.csv": [columns], ...}}.
_OUTCOME_CHECK_KEY = "post_de_outcome_check"


def outcome_removed_concern(run_dir: Path, state: RunState) -> str | None:
    """The "Please check" line when EDM-ARS removed the outcome from the
    predictors, else None.

    On the round-2 Mac test the data preparation left the outcome among
    the predictors and the study stopped after the paid analysis; the
    pipeline now removes such a column and goes on. Removing it cannot
    undo its use in filling in other predictors, so the reader is told.

    data_report.json says what the data the paper uses went through: a
    later data preparation (a revision) that left no outcome column
    replaces the record, and then there is nothing to say. Without the
    record, the warning event decides.
    """
    report = load_json(run_dir / "data_report.json")
    report = report if isinstance(report, dict) else {}
    record = report.get(_OUTCOME_CHECK_KEY)
    if isinstance(record, dict):
        if not record.get("removed"):
            return None
    elif not state.outcome_removed:
        return None
    outcome = str(report.get("outcome_variable") or "").strip()
    if not outcome:
        spec = load_json(run_dir / "research_spec.json")
        outcome = str(spec.get("outcome_variable") or "").strip() if isinstance(spec, dict) else ""
    words = messages().get("outcome_removed")
    template = str((words.get("concern") if isinstance(words, dict) else None) or (
        "The data preparation step left the outcome ({outcome}) among the predictors; "
        "EDM-ARS removed it before the analysis. If that step also used the outcome to "
        "fill in other predictors' missing values, those predictors can still carry it, "
        "so be wary of results that look too good."))
    if not outcome:
        template = template.replace(" ({outcome})", "")
    return " ".join(fill(template, outcome=outcome).split())


def _concerns(run_dir: Path, state: RunState, status: dict[str, Any] | None,
              *, skip_gate: bool) -> list[str]:
    """Plain lines for the "Please check" list besides invariant findings."""
    out: list[str] = []
    removed = outcome_removed_concern(run_dir, state)
    if removed:
        out.append(removed)
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
            out.append(f"Automated peer review did not run: {gate_skip_text(gate.get('skip_reason'))}")
        elif gate.get("ran") and gate.get("passed") is False and not gate.get("advisory"):
            out.append(_gate_sentence(gate))
    lit = status.get("literature") if isinstance(status, dict) else None
    if isinstance(lit, dict):
        out.extend(literature_concerns(lit))
    return out


def literature_concerns(lit: dict[str, Any]) -> list[str]:
    """The "Please check" line for run_status.json's ``literature`` block.

    Names the source that did not answer and how: the Mac test's arXiv
    search was refused (HTTP 406) and its Semantic Scholar searches
    rate-limited, and the old line said only that the search was "partly
    unavailable". When OpenAlex stood in for arXiv the line says so and
    how many papers it supplied, instead of claiming the papers all came
    from Semantic Scholar.
    """
    raw = lit.get("sources")
    sources: dict[str, Any] = raw if isinstance(raw, dict) else {}
    notes = literature_notes(sources)
    key_hint = (" A free Semantic Scholar key (`edmars setup literature`) makes that much "
                "less likely." if sources.get("semantic_scholar") == "rate_limited" else "")
    from_openalex = sources.get("openalex") == "ok" and bool(_count(sources.get("n_openalex")))
    from_s2 = sources.get("semantic_scholar") == "ok" and _count(sources.get("n_semantic_scholar")) != 0
    if lit.get("degraded"):
        if from_openalex and notes and not from_s2 and sources.get("arxiv") != "ok":
            # Semantic Scholar gave nothing, and OpenAlex filled in for
            # arXiv: not "few" papers, but one source's papers.
            return [f"{' and '.join(notes)}, so the related papers all come from OpenAlex. "
                    f"Check the related-work section and the references.{key_hint}"]
        why = (": " + " and ".join(notes) + "." if notes
               else " (the literature search was partly unavailable).")
        return [f"Few related papers were found{why} Check the related-work section and "
                f"the references.{key_hint}"]
    if notes:
        if from_s2 and from_openalex:
            only = "come from Semantic Scholar and OpenAlex"
        elif from_s2 and sources.get("arxiv") != "ok":
            only = "all come from Semantic Scholar"
        else:
            only = "come from fewer sources than usual"
        return [f"{'; '.join(notes)}, so the related papers {only}. "
                "Check the related-work section and the references."]
    return []


def _count(value: Any) -> int | None:
    """A paper count from run_status.json; None when absent or not a number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


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
        reason = gate_skip_text(gate.get("skip_reason"))
        out_code = _gate_code(gate.get("skip_reason"))
        verb = "gave no score" if out_code == "LSAR_SCORING_FAILED" else "did not run"
        headline = f"Your paper is written, but the automated peer review {verb}: {reason}"
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
    "gate_skip_text",
    "invariant_title",
    "load_findings",
    "messages",
    "outcome_removed_concern",
    "quote_path",
    "step_words",
    "redact",
    "abort_checks",
    "review_findings",
    "checks_in_words",
]
