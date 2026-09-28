"""Result screens and summary.html for a finished (or stopped) study.

``show(run_dir)`` prints either

* the success screen: the label, the main result in one sentence, the
  review scores, a "Please check" list (serious and notable findings of
  the final checks, with plain titles), what each file is, what to do
  next; or
* the failure screen: what happened, why, what to do, the exact command,
  and that finished steps are saved.

Every screen ends with the AI-draft reminder. No screen prints the
pipeline's internal "Release: YES/NO" line: a paper with open critical
findings is "Ready, with N serious issues to check".

``write_summary_html(run_dir)`` writes a self-contained ``summary.html``
into the run folder (inline CSS, relative links to the PDF and figures,
every piece of text HTML-escaped).
"""
from __future__ import annotations

import html
import os
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

from edmars.endstates import READY_KINDS, Outcome, classify, gate_skip_text, messages, quote_path
from edmars.model import EXIT_ERROR, EXIT_NOT_READY, EXIT_READY, EXIT_STOPPED
from edmars.runstate import (
    CUT_OFF_NOTE,
    EXPERIMENTAL_LINE,
    EXPERIMENTAL_NOTE,
    EXPERIMENTAL_WHY,
    RunState,
    fmt_ci,
    fmt_num,
    cost_line,
    fmt_score,
    load_state,
)

Line = tuple[str, str]  # (text, rich style)

_KIND_GLYPH = {
    "ready": ("✓", "[ok]", "bold green"),
    "ready_with_issues": ("!", "[!]", "bold yellow"),
    "not_ready": ("✗", "[x]", "bold red"),
    "stopped": ("✗", "[x]", "bold red"),
    "running": ("◐", "[..]", "bold yellow"),
}

_FIGURE_EXTS = (".png", ".jpg", ".jpeg", ".svg")


def reminder() -> str:
    text = messages().get("reminder") or (
        "This is an AI-generated draft. Check every number and citation before sharing, "
        "and follow your venue's rules on disclosing AI use. See `edmars disclaimer`."
    )
    return " ".join(str(text).split())


# ---------------------------------------------------------------------------
# Sentences
# ---------------------------------------------------------------------------


def _ci_words(ci: Any) -> str:
    if not (isinstance(ci, (list, tuple)) and len(ci) == 2):
        return ""
    text = fmt_ci(ci[0], ci[1])
    return f" (95% CI {text.strip('[]')})" if text else ""


def result_sentence(state: RunState) -> str | None:
    """The study's main result as one sentence.

    For a prediction study the best model is the one results.json's
    metrics put first (edmars.runstate.prediction_metrics). When the
    analysis named another model as its best, both are shown: the paper
    is likely to repeat the analysis's claim.
    """
    m = state.metrics
    if m.get("best_model"):
        metric = m.get("primary_metric")
        value = m.get("best_metric_value")
        claimed = m.get("claimed_best_model")
        held_out = ""
        if isinstance(m.get("n_test"), int) and m["n_test"] > 0:
            held_out = f", on {m['n_test']:,} students held out for testing"
        if claimed and metric and value is not None:
            ci = _ci_words(m.get("best_ci")).strip().strip("()")
            text = f"Best by {metric}: {m['best_model']} ({metric} {fmt_num(value)}"
            text += f", {ci})" if ci else ")"
            text += held_out
            text += f"; the analysis named {claimed}"
            if m.get("claimed_metric_value") is not None:
                text += f" ({metric} {fmt_num(m['claimed_metric_value'])})"
            return text + " as its best model — check the paper's claim."
        text = f"Best model: {m['best_model']}"
        if metric and value is not None:
            text += f", {metric} {fmt_num(value)}"
            text += _ci_words(m.get("best_ci"))
        return text + held_out + "."
    if m.get("estimate") is not None:
        ci_text = _ci_words(m.get("estimate_ci"))
        method = m.get("method_name") or m.get("method")
        if m.get("estimate_kind") == "policy_gain":
            lead = "Gain from the targeting rule over the best single policy"
        else:
            lead = f"Estimated effect ({m['estimand']})" if m.get("estimand") else "Estimated effect"
        text = f"{lead}: {fmt_num(m['estimate'])}{ci_text}"
        if method:
            text += f", from {method}"
        return text + "."
    if m.get("headline"):
        return str(m["headline"])
    fit = m.get("fit")
    if isinstance(fit, dict) and fit:
        return "Measurement results: " + ", ".join(f"{k} {fmt_num(v)}" for k, v in fit.items()) + "."
    return None


def _critic_line(state: RunState) -> str | None:
    m = state.metrics
    if m.get("critic_verdict") is None and m.get("critic_score") is None:
        return None
    verdict = str(m.get("critic_verdict") or "").upper()
    if m.get("critic_by_checks"):
        # The automatic pre-review checks decided the last round; the
        # reviewer did not score it (the "1/10" in their report is a
        # placeholder).
        did = "stopped the study" if verdict.startswith("ABORT") else "sent the work back"
        return f"Internal methods review: not scored; the automatic checks {did} before the review"
    words = {"PASS": "passed", "REVISE": "asked for changes", "ABORT": "stopped the study"}
    text = words.get(verdict.split()[0], verdict.lower()) if verdict else ""
    if m.get("critic_unverified") and verdict.startswith(("PASS", "REVISE")):
        text = "concerns not fully resolved"
    score = f"{fmt_score(m['critic_score'])}/10" if m.get("critic_score") is not None else ""
    return "Internal methods review: " + ", ".join(p for p in (score, text) if p)


def _gate_line(state: RunState, outcome: Outcome) -> str | None:
    m = state.metrics
    if m.get("gate_ran") is False:
        reason = m.get("gate_skip_reason")
        verb = "no score" if outcome.code == "LSAR_SCORING_FAILED" else "did not run"
        text = f"Automated peer review (LSAR): {verb}."
        return f"{text} {gate_skip_text(reason)}" if reason else text
    if m.get("gate_score") is None:
        if state.lsar_enabled:
            return "Automated peer review (LSAR): no score"
        return None
    text = f"Automated peer review (LSAR): {fmt_score(m['gate_score'])} out of 10"
    if m.get("gate_advisory"):
        text += " (score only; there is no benchmark for this venue)"
    elif m.get("gate_threshold") is not None:
        side = "at or above" if m.get("gate_passed") else "below"
        text += f", {side} the benchmark of {fmt_score(m['gate_threshold'])}"
    return text + ". Scores vary by about 2 points between reviews."


def _checks_line(state: RunState, outcome: Outcome) -> str | None:
    counts = state.metrics.get("invariant_counts")
    if not isinstance(counts, dict):
        if outcome.code == "VERIFICATION_NOT_RUN":
            return "Final checks: did not run"
        return None
    crit, major = int(counts.get("critical") or 0), int(counts.get("major") or 0)
    minor = int(counts.get("minor") or 0)
    if not (crit or major or minor):
        return "Final checks: no problems found"
    parts = []
    if crit:
        parts.append(f"{crit} serious")
    if major:
        parts.append(f"{major} to check")
    if minor:
        parts.append(f"{minor} minor")
    return "Final checks: " + ", ".join(parts)


def _files(run_dir: Path) -> list[tuple[str, str]]:
    meanings = messages().get("files") or {}
    out: list[tuple[str, str]] = []
    order = ["paper.pdf", "paper.tex", "references.bib", "summary.html", "results.json",
             "data_report.json", "review_report.json", "invariants.json", "lsar_review", "prompts"]
    for name in order:
        path = run_dir / name
        if path.exists():
            label = name + ("/" if path.is_dir() else "")
            out.append((label, str(meanings.get(name, ""))))
    figures = _figures(run_dir)
    if figures:
        out.insert(min(3, len(out)), (f"{len(figures)} figure file(s)", "Images the analysis produced"))
    return out


def _figures(run_dir: Path) -> list[Path]:
    try:
        return sorted(p for p in run_dir.iterdir() if p.is_file() and p.suffix.lower() in _FIGURE_EXTS)
    except OSError:
        return []


# ---------------------------------------------------------------------------
# Screens
# ---------------------------------------------------------------------------


def render_result(outcome: Outcome, state: RunState, run_dir: Path, *, plain: bool = False) -> list[Line]:
    """The result screen as (text, style) lines."""
    glyph, plain_glyph, style = _KIND_GLYPH.get(outcome.kind, ("·", "[ ]", "bold"))
    g = plain_glyph if plain else glyph
    run = quote_path(run_dir)
    out: list[Line] = []

    def add(text: str = "", st: str = "") -> None:
        out.append((text, st))

    def experimental() -> None:
        if state.experimental:
            add(f"{EXPERIMENTAL_LINE}. {EXPERIMENTAL_NOTE}", "bold yellow")

    if outcome.kind in READY_KINDS:
        add(f"{g} {outcome.label}", style)
        experimental()
        if state.question:
            add(f'"{state.question}"', "italic")
        add(outcome.headline)
        sentence = result_sentence(state)
        if sentence:
            add()
            add(f"Key result: {sentence}", "bold")
        score_lines = [x for x in (_critic_line(state), _gate_line(state, outcome), _checks_line(state, outcome)) if x]
        if score_lines:
            add()
            add("Scores", "bold")
            for line in score_lines:
                add(f"  {line}")
        _please_check(outcome, add, plain)
        if outcome.why:
            add()
            add(f"Why: {outcome.why}")
        if outcome.fix:
            add(f"What to do: {outcome.fix}")
        files = _files(run_dir)
        if files:
            add()
            add(f"Files (in {run_dir})", "bold")
            width = max(len(n) for n, _ in files) + 2
            for name, meaning in files:
                add(f"  {name.ljust(width)}{meaning}")
        add()
        add("What next", "bold")
        actions = [
            ("Open the paper", f"edmars results {run} --open pdf"),
            ("Open the folder", f"edmars results {run} --open folder"),
            ("Open the summary", f"edmars results {run} --open summary"),
        ]
        if state.metrics.get("gate_score") is None:
            actions.append(("Automated peer review", f"edmars review {run}"))
        elif outcome.code in ("LSAR_MISSING", "LSAR_FAILED", "LSAR_SCORING_FAILED"):
            actions.append(("Automated peer review", f"edmars review {run}"))
        if outcome.code in ("LSAR_MISSING",):
            # Setting up the reviewer comes before asking it for a review.
            review_at = next((i for i, (label, _) in enumerate(actions)
                              if label == "Automated peer review"), len(actions))
            actions.insert(review_at, ("First set up LSAR", "edmars setup lsar"))
        actions.append(("Start a new study", "edmars new"))
        for label, cmd in actions:
            add(f"  {label}:")
            add(f"    {cmd}", "bold")
    elif outcome.kind == "running":
        add(f"{g} {outcome.label}", style)
        experimental()
        add(outcome.headline)
        if outcome.fix:
            add(outcome.fix)
        for cmd in outcome.commands:
            add(f"  {cmd}", "bold")
        return out
    else:
        add(f"{g} {outcome.label}: {outcome.title or outcome.headline}", style)
        experimental()
        if state.question:
            add(f'"{state.question}"', "italic")
        add()
        add(f"What happened: {outcome.headline}")
        if outcome.why:
            add(f"Why: {outcome.why}")
        if outcome.details:
            add(outcome.details_heading or "Details:")
            for line in outcome.details:
                add(f"  - {line}")
        if outcome.note:
            add(outcome.note)
        if outcome.fix:
            add(f"What to do: {outcome.fix}")
        if outcome.commands:
            add("Type:" if len(outcome.commands) == 1 else "Type, one after the other:")
            for cmd in outcome.commands:
                add(f"  {cmd}", "bold")
        _please_check(outcome, add, plain)
        add()
        add(str(messages().get("saved_note") or "Your finished steps are saved."))
        add(f"Study folder: {run_dir}", "dim")
    _cost(state, add)
    add()
    add(reminder(), "italic")
    return out


def _cost(state: RunState, add: Any) -> None:
    """The cost, worded as the live view words it, and why it may be low."""
    if not (state.llm_calls or state.cost_usd is not None):
        return
    add()
    add(cost_line(state), "dim")
    if state.calls_cut_off:
        add(CUT_OFF_NOTE, "dim")


def _please_check(outcome: Outcome, add: Any, plain: bool) -> None:
    if not outcome.findings and not outcome.concerns:
        return
    add()
    add("Please check", "bold")
    serious = "[x]" if plain else "✗"
    notable = "[!]" if plain else "!"
    for f in outcome.findings:
        mark = serious if f.get("severity") == "critical" else notable
        times = f" ({f['count']} places)" if int(f.get("count") or 1) > 1 else ""
        add(f"  {mark} {f.get('title') or f.get('code')}{times} [{f.get('code')}]",
            "red" if f.get("severity") == "critical" else "yellow")
    for concern in outcome.concerns:
        add(f"  {notable} {concern}", "yellow")


def result_text(outcome: Outcome, state: RunState, run_dir: Path, *, plain: bool = True,
                width: int = 80) -> str:
    from edmars.view import to_ascii

    lines: list[str] = []
    for text, _style in render_result(outcome, state, run_dir, plain=plain):
        if plain:
            text = to_ascii(text)
        if not text:
            lines.append("")
            continue
        if text.lstrip().startswith("edmars "):
            lines.append(text)  # a command to copy: never wrapped
            continue
        indent = " " * (len(text) - len(text.lstrip(" ")) + 2)
        lines.extend(textwrap.wrap(text, width=max(width, 40), subsequent_indent=indent,
                                   break_long_words=False, break_on_hyphens=False) or [""])
    return "\n".join(lines)


def show(run_dir: Path | str, open_: str | None = None) -> int:
    """Print the result screen; optionally open the PDF, folder or summary.

    Returns the exit code of edmars.model's scheme: EXIT_READY (0) for a
    ready or still running study, EXIT_NOT_READY (2) when the paper is not
    ready, EXIT_STOPPED (3) when the study stopped, and EXIT_ERROR (1) for
    a bad ``--open`` value.
    """
    from edmars import ui

    run_dir = Path(run_dir)
    outcome = classify(run_dir)
    state = load_state(run_dir) if run_dir.is_dir() else RunState()
    plain = bool(ui.is_plain())
    summary: Path | None = None
    if run_dir.is_dir() and outcome.kind != "running":
        try:
            summary = write_summary_html(run_dir, outcome=outcome, state=state)
        except OSError:
            summary = None
    if plain:
        print(result_text(outcome, state, run_dir, plain=True, width=_width()), flush=True)
    else:
        from rich.text import Text

        for text, style in render_result(outcome, state, run_dir, plain=False):
            ui.console.print(Text(text, style=style), highlight=False)
    code = {"ready": EXIT_READY, "ready_with_issues": EXIT_READY, "running": EXIT_READY,
            "not_ready": EXIT_NOT_READY, "stopped": EXIT_STOPPED}.get(outcome.kind, EXIT_READY)
    if open_:
        targets = {"pdf": run_dir / "paper.pdf", "folder": run_dir, "summary": summary or run_dir / "summary.html"}
        target = targets.get(open_)
        if target is None:
            ui.warn(f"Unknown --open value {open_!r}; use pdf, folder or summary.")
            return EXIT_ERROR
        if not target.exists():
            what = {"pdf": "There is no PDF for this study.", "summary": "There is no summary yet."}
            ui.warn(what.get(open_, f"{target} does not exist."))
        else:
            ui.open_path(target)
    return code


def _width() -> int:
    try:
        import shutil

        return min(shutil.get_terminal_size((80, 24)).columns, 100)
    except Exception:  # noqa: BLE001
        return 80


# ---------------------------------------------------------------------------
# summary.html
# ---------------------------------------------------------------------------

_CSS = """
:root { --bg:#ffffff; --fg:#1d2433; --muted:#5b6475; --line:#d9dde5; --ok:#1c7c3a;
        --warn:#9a6200; --bad:#b3261e; --card:#f6f7f9; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#14171c; --fg:#e7e9ee; --muted:#a3aab8; --line:#2c323c; --ok:#5cc47f;
          --warn:#e0a84a; --bad:#f07068; --card:#1c2027; }
}
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg);
       font: 16px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
main { max-width: 880px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 1.6rem; margin: 0 0 4px; }
h2 { font-size: 1.1rem; margin: 28px 0 8px; border-bottom: 1px solid var(--line); padding-bottom: 4px; }
.q { color: var(--muted); font-style: italic; margin: 0 0 12px; }
.badge { display:inline-block; padding: 2px 10px; border-radius: 999px; font-weight: 600;
         border: 1px solid currentColor; }
.ready { color: var(--ok); } .ready_with_issues { color: var(--warn); }
.not_ready, .stopped { color: var(--bad); } .running { color: var(--warn); }
.key { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px; }
ul { padding-left: 1.2rem; } li { margin: 4px 0; }
li.critical::marker { color: var(--bad); } li.major::marker { color: var(--warn); }
code { font-family: ui-monospace, Consolas, monospace; font-size: .9em; }
.files td { padding: 3px 12px 3px 0; vertical-align: top; }
figure { margin: 16px 0; } figure img { max-width: 100%; height: auto; border: 1px solid var(--line);
         border-radius: 6px; background: #fff; }
figcaption { color: var(--muted); font-size: .9rem; }
.note { border-left: 4px solid var(--warn); padding: 8px 12px; background: var(--card); margin-top: 28px; }
.muted { color: var(--muted); font-size: .9rem; }
"""


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _href(name: str) -> str:
    from urllib.parse import quote

    return _e(quote(name))


def render_summary_html(outcome: Outcome, state: RunState, run_dir: Path) -> str:
    parts: list[str] = []
    add = parts.append
    add("<!DOCTYPE html>")
    add('<html lang="en"><head><meta charset="utf-8">')
    add('<meta name="viewport" content="width=device-width, initial-scale=1">')
    add("<title>EDM-ARS study summary</title>")
    add(f"<style>{_CSS}</style></head><body><main>")
    add('<h1>Study summary</h1>')
    if state.question:
        add(f'<p class="q">&ldquo;{_e(state.question)}&rdquo;</p>')
    add(f'<p><span class="badge {_e(outcome.kind)}">{_e(outcome.label)}</span></p>')
    if state.experimental:
        add(f'<p class="note"><strong>{_e(EXPERIMENTAL_LINE)}.</strong> '
            f"{_e(EXPERIMENTAL_WHY)}</p>")
    add(f"<p>{_e(outcome.headline)}</p>")
    sentence = result_sentence(state)
    if sentence and outcome.kind in READY_KINDS:
        add(f'<div class="key"><strong>Key result.</strong> {_e(sentence)}</div>')
    if outcome.kind not in READY_KINDS:
        if outcome.why:
            add(f"<p><strong>Why:</strong> {_e(outcome.why)}</p>")
        if outcome.details:
            add(f"<p><strong>{_e(outcome.details_heading or 'Details:')}</strong></p><ul>")
            for line in outcome.details:
                add(f"<li>{_e(line)}</li>")
            add("</ul>")
        if outcome.note:
            add(f"<p>{_e(outcome.note)}</p>")
        if outcome.fix:
            add(f"<p><strong>What to do:</strong> {_e(outcome.fix)}</p>")
        if outcome.commands:
            add("<p>" + "<br>".join(f"<code>{_e(c)}</code>" for c in outcome.commands) + "</p>")
    scores = [x for x in (_critic_line(state), _gate_line(state, outcome), _checks_line(state, outcome)) if x]
    if scores:
        add("<h2>Scores</h2><ul>")
        for line in scores:
            add(f"<li>{_e(line)}</li>")
        add("</ul>")
    if outcome.findings or outcome.concerns:
        add("<h2>Please check</h2><ul>")
        for f in outcome.findings:
            sev = f.get("severity") or ""
            label = "serious" if sev == "critical" else "check"
            count = int(f.get("count") or 1)
            times = f" ({count} places)" if count > 1 else ""
            add(f'<li class="{_e(sev)}"><strong>{_e(label)}:</strong> {_e(f.get("title") or f.get("code"))}'
                f'{_e(times)} <code>{_e(f.get("code"))}</code>')
            if f.get("message"):
                add(f'<details><summary class="muted">Technical detail</summary>'
                    f'<p class="muted">{_e(f.get("message"))}</p></details>')
            add("</li>")
        for concern in outcome.concerns:
            add(f'<li class="major">{_e(concern)}</li>')
        add("</ul>")
    if outcome.kind in READY_KINDS and outcome.why:
        add(f"<p>{_e(outcome.why)}</p>")
    files = [(n, m) for n, m in _files(run_dir) if not n[0].isdigit()]
    if files:
        add('<h2>Files</h2><table class="files">')
        for name, meaning in files:
            target = name.rstrip("/")
            add(f'<tr><td><a href="{_href(target)}">{_e(name)}</a></td><td>{_e(meaning)}</td></tr>')
        add("</table>")
    figures = _figures(run_dir)
    if figures:
        add("<h2>Figures</h2>")
        for fig in figures[:24]:
            add(f'<figure><img src="{_href(fig.name)}" alt="{_e(fig.stem)}" loading="lazy">'
                f"<figcaption>{_e(fig.name)}</figcaption></figure>")
    cost = state.cost_usd
    meta = []
    if state.dataset:
        meta.append(f"Dataset: {state.dataset}")
    if state.provider:
        meta.append(f"AI service: {state.provider}")
    if cost is not None or state.llm_calls:
        meta.append(cost_line(state))
    meta.append(f"Summary written {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    add(f'<p class="muted">{_e(" · ".join(meta))}</p>')
    if state.calls_cut_off:
        add(f'<p class="muted">{_e(CUT_OFF_NOTE)}</p>')
    add(f'<p class="note">{_e(reminder())}</p>')
    add("</main></body></html>")
    return "\n".join(parts) + "\n"


def write_summary_html(run_dir: Path | str, *, outcome: Outcome | None = None,
                       state: RunState | None = None) -> Path:
    """Write ``<run>/summary.html`` and return its path."""
    run_dir = Path(run_dir)
    outcome = outcome or classify(run_dir)
    state = state or load_state(run_dir)
    text = render_summary_html(outcome, state, run_dir)
    path = run_dir / "summary.html"
    tmp = run_dir / "summary.html.tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    os.replace(tmp, path)
    return path


__all__ = [
    "render_result",
    "render_summary_html",
    "reminder",
    "result_sentence",
    "result_text",
    "show",
    "write_summary_html",
]
