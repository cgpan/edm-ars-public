"""Arc P / P1 — deterministic manuscript linter.

Post-compile verifier battery for generated papers. Mines the artifacts
a compile already produces (paper.tex, references.bib, pdflatex .log,
biber .blg, optionally the PDF text) for presentation defects that
"compiled successfully" does not catch, and compares citation depth and
body length against per-venue norms mined from the LSAR anchor corpus
(data_registry/venue_norms.yaml, scripts/mine_venue_norms.py).

Severities:
    error — reader-visible defect (undefined citation/reference, ``??``
            in the PDF, placeholder citations, cited key missing from
            the .bib, a bibliography with no prior-decade scholarship).
    warn  — below venue norms or sloppy-but-readable (unreferenced
            floats, heavy overfull boxes, reference-age distribution
            off the venue profile, abstract framing off every measured
            venue's norms — feature-importance headlined, no named
            practice/decision in the closing sentences).

Some checks record INFO-level metrics only and never add a defect —
e.g. whether a manuscript with a school-aware split states the
within/cross-context contrast (``school_aware_split_stated`` /
``within_cross_contrast_stated``).

The linter never raises on malformed inputs; anything unreadable is
reported as a defect. It is pure inspection — no LLM calls, no edits.
"""
from __future__ import annotations

import json
import re
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

# ---------------------------------------------------------------------------
# Report model
# ---------------------------------------------------------------------------


@dataclass
class LintDefect:
    severity: str  # "error" | "warn"
    code: str
    message: str


@dataclass
class LintReport:
    defects: list[LintDefect] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    venue: Optional[str] = None

    @property
    def errors(self) -> list[LintDefect]:
        return [d for d in self.defects if d.severity == "error"]

    @property
    def format_clean(self) -> bool:
        return not self.errors

    def add(self, severity: str, code: str, message: str) -> None:
        self.defects.append(LintDefect(severity, code, message))

    def to_dict(self) -> dict:
        return {
            "format_clean": self.format_clean,
            "venue": self.venue,
            "defects": [asdict(d) for d in self.defects],
            "metrics": self.metrics,
        }


# ---------------------------------------------------------------------------
# UNVERIFIED flag (SPEC section 4.5) — canonical text shared with the Writer
# ---------------------------------------------------------------------------

# I2 (AERA_OPEN audit): the mandatory warning block existed only as a
# Writer prompt rule and was silently omitted after a REVISING crash.
# The Writer now injects this block DETERMINISTICALLY (src/agents/
# writer.py imports these constants) and the linter errors when a
# flagged run's manuscript lacks the marker.

UNVERIFIED_MARKER = (
    "WARNING: This paper has unresolved methodological issues "
    "identified by automated review"
)

UNVERIFIED_BLOCK = (
    "\\begin{quote}\n"
    "\\textbf{" + UNVERIFIED_MARKER + ". The issues listed in the "
    "appendix were not resolved within the allowed revision cycles. "
    "Use with caution.}\n"
    "\\end{quote}\n"
)


def run_is_unverified(review_report: Optional[dict]) -> bool:
    """SPEC section 4.5 condition, shared by Writer and linter.

    An explicit ``unverified`` key wins in BOTH directions: the
    orchestrator's verdict evaluator can compute an effective PASS from
    a raw LLM "REVISE" ("evaluator wins"), and stamps ``unverified:
    False`` to say so — without that, the raw verdict string would put
    the warning block on a paper the system itself judged passing
    (review finding, evaluator-override path).
    """
    if not review_report:
        return False
    if "unverified" in review_report:
        return bool(review_report["unverified"])
    verdict = review_report.get("overall_verdict")
    return bool(verdict) and verdict != "PASS"


def _check_unverified_flag(report: LintReport, tex: str, run_dir: Path) -> None:
    """ERROR when the run is flagged UNVERIFIED but the manuscript
    carries no warning block (I2)."""
    ckpt_path = run_dir / "checkpoint.json"
    if not ckpt_path.exists():
        report.metrics["unverified_flag_checked"] = False
        return
    try:
        ckpt = json.loads(ckpt_path.read_text(encoding="utf-8", errors="replace"))
    except (json.JSONDecodeError, OSError):
        report.metrics["unverified_flag_checked"] = False
        return
    review = ckpt.get("review_report")
    flagged = run_is_unverified(review)
    report.metrics["unverified_flag_checked"] = True
    report.metrics["run_unverified"] = flagged
    if flagged and UNVERIFIED_MARKER not in tex:
        report.add(
            "error",
            "unverified-block-missing",
            "checkpoint review_report marks this run UNVERIFIED "
            "(verdict != PASS or unverified=true) but the manuscript "
            "carries no SPEC section 4.5 warning block — the paper "
            "presents itself as clean",
        )


# ---------------------------------------------------------------------------
# Numeric reconciliation (I1) — manuscript numerals vs analysis artifacts
# ---------------------------------------------------------------------------

_NUMERAL = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+\.\d+|-?\d+")
_CI_INTERVAL = re.compile(
    r"\[\s*([-+\u2212]?\d+\.\d+)\s*,\s*([-+\u2212]?\d+\.\d+)\s*\]"
)
# Known limitation (fail-open): a tabular nested inside a SAME-name
# tabular truncates the match at the inner \end, so numerals in the
# outer table's tail escape reconciliation. Different-name nesting
# (tabular inside tabularx) is handled by the backreference.
_TABULAR_ENV = re.compile(
    r"\\begin\{(tabular[xy*]?|longtable)\}.*?\\end\{\1\}", re.DOTALL
)


def _numeric_leaves(obj: object) -> list[float]:
    """All int/float leaves of a JSON-like structure (bools excluded)."""
    out: list[float] = []
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        if isinstance(obj, float) and obj != obj:
            return out
        out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(_numeric_leaves(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(_numeric_leaves(v))
    return out


def _is_num(v: object) -> bool:
    return (
        isinstance(v, (int, float))
        and not isinstance(v, bool)
        and not (isinstance(v, float) and v != v)
    )


def _pairwise(group: list[float], cap: int = 30) -> list[float]:
    """Differences AND sums of a sibling group: a contrast is a
    difference of sibling estimates; a marginal is a sum of sibling
    counts. Both are legitimate reader-facing derivations."""
    out: list[float] = []
    if 2 <= len(group) <= cap:
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                out.append(a - b)
                out.append(a + b)
    return out


def _sibling_diffs(obj: object, cap: int = 30) -> list[float]:
    """Pairwise differences/sums of numeric values that live in the SAME
    container — the shapes a reported contrast or marginal legitimately
    takes (tau_mean differences between subgroup levels; row/column
    totals of cell counts).

    Sibling groups: direct numeric children of one dict; the same-named
    numeric field across a list of dicts; and the same-named numeric
    field across a dict of dicts (subgroup levels keyed by name).
    Restricting derivation to siblings keeps the candidate set sparse;
    whole-artifact pairwise combination is dense enough to match almost
    any fabricated numeral, which would defeat the check.
    """
    diffs: list[float] = []
    if isinstance(obj, dict):
        direct = [float(v) for v in obj.values() if _is_num(v)]
        diffs.extend(_pairwise(direct, cap))
        child_dicts = [v for v in obj.values() if isinstance(v, dict)]
        if len(child_dicts) >= 2:
            keys: set[str] = set()
            for v in child_dicts:
                keys.update(v.keys())
            for k in keys:
                group = [
                    float(v[k]) for v in child_dicts if _is_num(v.get(k))
                ]
                diffs.extend(_pairwise(group, cap))
        for v in obj.values():
            diffs.extend(_sibling_diffs(v, cap))
    elif isinstance(obj, list):
        if obj and all(isinstance(v, dict) for v in obj):
            keys = set()
            for v in obj:
                keys.update(v.keys())
            for k in keys:
                group = [float(v[k]) for v in obj if _is_num(v.get(k))]
                diffs.extend(_pairwise(group, cap))
        for v in obj:
            diffs.extend(_sibling_diffs(v, cap))
    return diffs


def _ground_candidates(run_dir: Path) -> tuple[list[float], dict]:
    """Collect every number the analysis actually produced.

    Sources: results.json (regex-salvaged so a truncated/invalid file
    still contributes — the AERA_OPEN run's results.json was cut off
    mid-serialization), checkpoint.json results_object + data_report,
    data_report.json. Adds sign flips, x100 percent variants of
    proportions, and sibling differences (legitimate contrasts).
    """
    base: list[float] = []
    info: dict = {"sources": []}
    structured: list[object] = []
    res_path = run_dir / "results.json"
    if res_path.exists():
        raw = res_path.read_text(encoding="utf-8", errors="replace")
        # Regex salvage: numerals survive even when the file is invalid
        # JSON (the AERA_OPEN run's results.json was truncated
        # mid-serialization by the NaN crash).
        for tok in re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", raw):
            try:
                base.append(float(tok))
            except ValueError:
                pass
        info["sources"].append("results.json")
        # Structural parse (when valid) feeds sibling-derivation too —
        # marginal sums and contrasts of results.json values are as
        # legitimate as those of checkpoint values.
        try:
            structured.append(json.loads(raw))
            info["sources"].append("results.json:structured")
        except json.JSONDecodeError:
            pass
    ckpt_path = run_dir / "checkpoint.json"
    if ckpt_path.exists():
        try:
            ckpt = json.loads(
                ckpt_path.read_text(encoding="utf-8", errors="replace")
            )
            for key in ("results_object", "data_report"):
                if isinstance(ckpt.get(key), dict):
                    structured.append(ckpt[key])
                    info["sources"].append(f"checkpoint.{key}")
        except (json.JSONDecodeError, OSError):
            pass
    for name in ("data_report.json", "research_spec.json"):
        p = run_dir / name
        if p.exists():
            try:
                structured.append(
                    json.loads(p.read_text(encoding="utf-8", errors="replace"))
                )
                info["sources"].append(name)
            except (json.JSONDecodeError, OSError):
                pass
    # SUMMARY CSV outputs (model_comparison, feature_importance,
    # subgroup_performance, ...): tables are sometimes rendered from
    # values that exist only there. Only SMALL files qualify — a paper
    # table renders summaries, and salvaging a row-level data file
    # (panel_analytic.csv added ~19k numerals here) densifies the
    # candidate space until fabricated values match by accident,
    # which guts the check.
    _CSV_SIZE_CAP = 10_000  # bytes; summary CSVs are well under this
    for p in sorted(run_dir.glob("*.csv")):
        try:
            if p.stat().st_size > _CSV_SIZE_CAP:
                info.setdefault("csv_skipped_rowlevel", []).append(p.name)
                continue
            raw = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for tok in re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", raw):
            try:
                base.append(float(tok))
            except ValueError:
                pass
        info["sources"].append(p.name)
    for s in structured:
        base.extend(_numeric_leaves(s))
    derived: list[float] = []
    for s in structured:
        derived.extend(_sibling_diffs(s))
    cand: set[float] = set()
    for g in base:
        cand.add(g)
        cand.add(-g)
        if abs(g) <= 1.0:
            cand.add(g * 100.0)
            cand.add(-g * 100.0)
    for g in derived:
        cand.add(g)
        cand.add(-g)
    info["n_base"] = len(base)
    info["n_derived"] = len(derived)
    info["n_candidates"] = len(cand)
    return sorted(cand), info


def _matches(sorted_cand: list[float], text_value: str) -> bool:
    """True when some ground candidate rounds to the printed numeral at
    its printed precision."""
    import bisect

    tok = text_value.replace(",", "").replace("\u2212", "-").replace("+", "")
    try:
        x = float(tok)
    except ValueError:
        return True  # unparseable → do not flag
    decimals = len(tok.split(".")[1]) if "." in tok else 0
    tol = 0.5 * 10 ** (-decimals) * 1.02 + 1e-12
    lo = bisect.bisect_left(sorted_cand, x - tol)
    return lo < len(sorted_cand) and sorted_cand[lo] <= x + tol


_DIMEN = re.compile(
    r"-?\d*\.?\d+\s*(?:cm|mm|in|pt|em|ex|pc|bp|sp|mu)\b"
)
_NUM_MACRO = re.compile(r"-?\d*\.?\d+\s*\\[a-zA-Z@]+")
_SCI_NOTATION = re.compile(
    r"-?\d+\.?\d*\s*(?:[eE][-+]?\d+|\\times\s*10\s*\^)"
)


def _normalize_table_text(body: str) -> str:
    """Remove LaTeX markup whose numbers are layout, not data.

    Review-found false positives (all empirically reproduced): the
    tabular column-spec preamble (``p{2.5cm}``, ``p{0.24\\columnwidth}``),
    dimension arguments of \\rule/\\multirow/\\resizebox, numbers glued
    to macros (``0.9\\columnwidth``), and LaTeX digit-grouping
    ``23{,}503`` which fragmented into an exempt ``23`` plus a
    guaranteed-miss ``503`` — an honest sample-size table could
    gate-block. Scientific-notation mantissas are removed too (their
    printed value cannot be reconstructed reliably; fail-open).
    """
    body = body.replace("{,}", "")  # digit grouping: 23{,}503 -> 23503
    # drop the column-spec preamble: first brace group after \begin{...}
    body = re.sub(
        r"(\\begin\{(?:tabular[xy*]?|longtable)\})(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})",
        r"\1",
        body,
    )
    body = _SCI_NOTATION.sub(" ", body)
    body = _DIMEN.sub(" ", body)
    body = _NUM_MACRO.sub(" ", body)
    body = re.sub(r"\\multicolumn\{\d+\}\{[^}]*\}", "", body)
    body = re.sub(r"\\[a-zA-Z@]+\*?(?:\[[^\]]*\])?", " ", body)
    return body


def _table_numerals(block: str) -> list[str]:
    """Checkable numerals in one tabular block: decimals, thousands-
    separated integers, and bare integers >= 100 that are not years."""
    body = _normalize_table_text(block)
    out: list[str] = []
    for tok in _NUMERAL.findall(body):
        plain = tok.replace(",", "")
        if "." in plain:
            out.append(tok)
        else:
            try:
                iv = int(plain)
            except ValueError:
                continue
            if abs(iv) >= 100 and not (1900 <= iv <= 2035):
                out.append(tok)
    return out


def _check_numeric_reconciliation(
    report: LintReport, tex: str, run_dir: Path
) -> None:
    """I1: every numeral the manuscript's tables and CI intervals print
    must be producible from the analysis artifacts. The AERA_OPEN run's
    Writer invented four cell means, a follow-wave CI, and five pairwise
    CIs — none existed anywhere on disk, and no reviewer could know."""
    cand, info = _ground_candidates(run_dir)
    if not cand:
        report.metrics["numeric_reconciliation_checked"] = False
        return
    report.metrics["numeric_reconciliation_checked"] = True
    report.metrics["numeric_ground"] = info

    checked = unmatched_total = 0
    flagged_tables: list[str] = []
    for t_idx, m in enumerate(_TABULAR_ENV.finditer(tex)):
        block = m.group(0)
        # Label attribution: prefer tab:-prefixed labels and the one
        # CLOSEST to this tabular (review finding: first-match-in-window
        # could name a preceding section/figure label).
        window = tex[max(0, m.start() - 500):m.end() + 200]
        labels = re.findall(r"\\label\{([^}]+)\}", window)
        tab_labels = [x for x in labels if x.startswith("tab")]
        name = (
            tab_labels[-1] if tab_labels
            else labels[-1] if labels
            else f"table#{t_idx + 1}"
        )
        numerals = _table_numerals(block)
        misses = [n for n in numerals if not _matches(cand, n)]
        checked += len(numerals)
        unmatched_total += len(misses)
        if len(misses) >= 3 and numerals and len(misses) / len(numerals) >= 0.4:
            flagged_tables.append(name)
            report.add(
                "error",
                "unreconciled-table-numerals",
                f"{name}: {len(misses)} of {len(numerals)} numerals match "
                "nothing in the analysis artifacts (values: "
                + ", ".join(misses[:6])
                + ") — numbers a reader will take as computed results "
                "that the analysis never produced",
            )
        elif misses:
            report.add(
                "warn",
                "unreconciled-table-numerals",
                f"{name}: unmatched numeral(s) " + ", ".join(misses[:6]),
            )

    # CI intervals in prose (outside tables): both endpoints invented
    # is the fabricated-interval signature.
    prose = _TABULAR_ENV.sub(" ", tex)
    prose = prose.replace("{,}", "")
    prose = prose.replace("$", "").replace("\\(", "").replace("\\)", "")
    ci_flagged = 0
    for m in _CI_INTERVAL.finditer(prose):
        lo_s, hi_s = m.group(1), m.group(2)
        checked += 2
        lo_ok, hi_ok = _matches(cand, lo_s), _matches(cand, hi_s)
        unmatched_total += (not lo_ok) + (not hi_ok)
        if not lo_ok and not hi_ok:
            ci_flagged += 1
            report.add(
                "error",
                "unreconciled-ci-interval",
                f"confidence interval [{lo_s}, {hi_s}] matches nothing "
                "in the analysis artifacts — a fully invented interval",
            )
        elif not lo_ok or not hi_ok:
            # A CI printed from real artifacts matches on BOTH endpoints;
            # one miss is either fabrication with an accidental match
            # (the AERA_OPEN run's invented [-2.61, +0.93] collided with
            # an unrelated 0.9322 at print tolerance) or a value edited
            # away from its source. Worth a human look either way.
            ci_flagged += 1
            report.add(
                "warn",
                "unreconciled-ci-interval",
                f"confidence interval [{lo_s}, {hi_s}]: endpoint "
                f"{lo_s if not lo_ok else hi_s} matches nothing in the "
                "analysis artifacts",
            )
    report.metrics["numerals_checked"] = checked
    report.metrics["numerals_unmatched"] = unmatched_total
    report.metrics["tables_numerically_flagged"] = flagged_tables
    report.metrics["ci_intervals_flagged"] = ci_flagged


# ---------------------------------------------------------------------------
# TeX source checks
# ---------------------------------------------------------------------------

_CITE_CMD = re.compile(
    r"\\(?:no)?(?:cite[pt]?|parencite|textcite|autocite|citeauthor|citeyear)"
    r"\*?\s*(?:\[[^\]]*\]\s*)*\{([^}]+)\}"
)
_BIB_ENTRY = re.compile(r"@\w+\s*\{\s*([^,\s]+)\s*,")
_LABEL = re.compile(r"\\label\{([^}]+)\}")
_REF = re.compile(r"\\(?:auto|c|C|page|eq)?ref\*?\{([^}]+)\}")
# \nocite{*} forces every .bib entry into the rendered bibliography; it is
# the only construct that makes an *uncited* entry reader-visible.
_NOCITE_ALL = re.compile(r"\\nocite\s*\{\s*\*\s*\}")


def cited_keys(tex: str) -> set[str]:
    keys: set[str] = set()
    for group in _CITE_CMD.findall(tex):
        keys.update(k.strip() for k in group.split(",") if k.strip())
    return keys


def _real_cited_keys(tex: str) -> set[str]:
    """``cited_keys`` minus the ``\\nocite{*}`` wildcard.

    ``*`` is not a citation key: left in, it inflates
    ``n_citations_distinct`` and fires a bogus
    ``cited-key-missing-from-bib`` on any manuscript using ``\\nocite{*}``.
    """
    return {k for k in cited_keys(tex) if k != "*"}


def _strip_tex(text: str) -> str:
    text = re.sub(r"(?<!\\)%.*", "", text)
    text = re.sub(r"\\begin\{(?:table|figure|equation|align)\*?\}.*?"
                  r"\\end\{(?:table|figure|equation|align)\*?\}", " ", text,
                  flags=re.DOTALL)
    text = re.sub(r"\\[a-zA-Z@]+\s*(?:\[[^\]]*\])?(?:\{[^{}]*\})*", " ", text)
    return re.sub(r"[{}~$&_^\\]", " ", text)


def _braced_arg(text: str, command: str) -> Optional[str]:
    """Balanced-brace argument of ``command``; None when absent."""
    idx = text.find(command + "{")
    if idx == -1:
        return None
    start = idx + len(command) + 1
    depth = 1
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{" and text[i - 1] != "\\":
            depth += 1
        elif ch == "}" and text[i - 1] != "\\":
            depth -= 1
            if depth == 0:
                return text[start:i]
    return None


def _extract_abstract(tex: str) -> Optional[str]:
    """Raw abstract text (env or ``\\abstract{}`` macro); None if absent."""
    env = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.DOTALL)
    if env:
        return env.group(1)
    return _braced_arg(tex, r"\abstract")


def _check_front_matter(report: LintReport, tex: str) -> None:
    """Title / abstract / keywords must exist and be non-empty.

    Added after F-P5-EMPTY-ABSTRACT: a journal manuscript shipped with a
    literally empty ``\\abstract{}`` because reassembly matched only the
    ``\\begin{abstract}`` environment. It compiled fine, this linter
    reported ``format_clean=True``, and the reviewer then rejected the
    paper outright for having no abstract -- burning the entire review.
    Structural completeness is exactly what a deterministic checker
    should catch, and it must be caught BEFORE the expensive stage.
    """
    abstract = _extract_abstract(tex)
    if abstract is None:
        report.add("error", "missing-abstract",
                   "no \\abstract{} or \\begin{abstract} block in the manuscript")
    else:
        n = len(abstract.strip())
        report.metrics["abstract_chars"] = n
        if n < 50:
            report.add(
                "error", "empty-abstract",
                f"abstract is empty or too short ({n} chars); reviewers and "
                "sanity checkers reject papers without one",
            )

    title = _braced_arg(tex, r"\title")
    if title is not None and not title.strip():
        report.add("error", "empty-title", "\\title{} is empty")

    leftover = sorted(set(re.findall(r"%%PLACEHOLDER:([A-Z_]+)%%", tex)))
    if leftover:
        report.add(
            "error", "unfilled-placeholder",
            "template placeholders were never substituted: "
            + ", ".join(leftover),
        )


def _check_tex(report: LintReport, tex: str, bib: Optional[str]) -> None:
    cited = _real_cited_keys(tex)
    report.metrics["n_citations_distinct"] = len(cited)

    placeholders = sorted(k for k in cited if k.startswith("placeholder"))
    if placeholders:
        report.add(
            "error", "placeholder-citations",
            f"{len(placeholders)} placeholder citation key(s) in paper.tex "
            f"(e.g. {placeholders[0]}) — literature context missing or unmapped",
        )

    if bib is not None:
        defined = set(_BIB_ENTRY.findall(bib))
        report.metrics["n_bib_entries"] = len(defined)
        missing = sorted(cited - defined - set(placeholders))
        if missing:
            report.add(
                "error", "cited-key-missing-from-bib",
                f"{len(missing)} cited key(s) absent from references.bib: "
                + ", ".join(missing[:8])
                + ("…" if len(missing) > 8 else ""),
            )
        uncited = sorted(defined - cited)
        report.metrics["n_bib_uncited"] = len(uncited)
        # Arc P residual G5. BibTeX/biblatex typeset only the entries a
        # document actually cites, so an over-provisioned references.bib
        # (Arc P3 deliberately ships a superset the reviser may draw
        # from — 62 entries, 19 cited) never reaches the PDF and is not
        # reader-visible. Firing on every run made the code pure noise
        # and trained the reviser to ignore the defect list. Scope it to
        # the one construct that *does* render them: \nocite{*}. The
        # count survives as a metric for the Arc I ledger.
        report.metrics["nocite_all"] = bool(_NOCITE_ALL.search(tex))
        if uncited and report.metrics["nocite_all"]:
            report.add(
                "warn", "many-uncited-bib-entries",
                f"{len(uncited)} bib entries reach the rendered bibliography "
                "via \\nocite{*} but are never cited in the text",
            )

    labels = set(_LABEL.findall(tex))
    refs = set(_REF.findall(tex))
    dangling = sorted(refs - labels)
    if dangling:
        report.add(
            "error", "dangling-crossref",
            f"{len(dangling)} \\ref target(s) with no \\label: "
            + ", ".join(dangling[:8]),
        )
    float_labels = {
        lab for lab in labels
        if lab.split(":", 1)[0].lower() in ("tab", "table", "fig", "figure")
    }
    unreferenced = sorted(float_labels - refs)
    report.metrics["n_floats_labeled"] = len(float_labels)
    if unreferenced:
        report.add(
            "warn", "unreferenced-float",
            f"{len(unreferenced)} labeled table(s)/figure(s) never referenced "
            "in the text: " + ", ".join(unreferenced[:8]),
        )

    body = tex
    begin = tex.find(r"\begin{document}")
    if begin != -1:
        body = tex[begin:]
    report.metrics["body_words"] = len(_strip_tex(body).split())


# ---------------------------------------------------------------------------
# Abstract content  (V5 Arc T H2 framing fixes: VF2-03, VF2-07 tier 1)
# ---------------------------------------------------------------------------
#
# Evidence base (docs/v5_arc_t_h2_capability_roadmap.md §1-2):
#   VF2-03 — 0 of 1,135 measured abstracts (34 EDM/JEDM/JLA anchors +
#   1,101 policy/ed-psych venue abstracts) and 0 of 30 AERA Open full
#   texts headline a feature-importance ranking. Promoting SHAP output
#   to the abstract/title is a defect at every measured venue.
#   VF2-07 tier 1 — anchor abstracts name a specific practice, decision
#   or artifact in 7/34; ours in 1/13.
# The mandatory writing rule lives in
# skills/writing/paper-section-content-prediction; these checks feed the
# Arc P4 revision loop. Both are heuristics over the abstract text, so
# they warn rather than error.

_FEATURE_IMPORTANCE_CLAIM = re.compile(
    r"\bSHAP\b"
    r"|feature[- ]importance"
    r"|(?:variable|predictor)[- ]importance"
    r"|importance\s+rank\w*"
    r"|rank\w*\s+of\s+(?:feature|predictor|variable)s"
    r"|(?:most|top)\s+(?:important|predictive|influential)\s+"
    r"(?:feature|predictor|variable|factor)s?"
    r"|top[- ]\d*\s*(?:features?|predictors?)\b",
    re.IGNORECASE,
)

#: Deliberately broad actor/decision/design lexicon: a false pass (a
#: vague closing sentence that happens to contain "design") is cheaper
#: than a false fire that trains the reviser to ignore the defect list
#: (Arc P residual G5 lesson).
_DECISION_TERMS = re.compile(
    r"\b(?:advis\w*|counsel\w*|teacher\w*|instructor\w*|tutor\w*"
    r"|educator\w*|district\w*|school leader\w*|administrat\w*"
    r"|practitioner\w*|policymaker\w*|policy|policies"
    r"|admission\w*|placement\w*|intervention\w*|interven\w*"
    r"|screening|screen\w*|early[- ]warning|alert\w*|triage\w*"
    r"|curricul\w*|instruction\w*|advising|deploy\w*|decision\w*"
    r"|design\w*|reweight\w*|prioriti[sz]\w*|target\w*|flag\w*"
    r"|monitor\w*|allocat\w*|outreach|referral\w*|tutoring)\b",
    re.IGNORECASE,
)

# Sentence boundary: terminal punctuation followed by whitespace and an
# uppercase/digit/bracket opener — avoids splitting on decimals and on
# lowercase abbreviations ("e.g. foo").
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


def _check_abstract_content(report: LintReport, tex: str) -> None:
    """Framing checks on the abstract (and title). Warn-level heuristics."""
    raw = _extract_abstract(tex)
    if raw is None or len(raw.strip()) < 50:
        # Missing/empty abstract is already an error in _check_front_matter.
        report.metrics["abstract_content_checked"] = False
        return
    report.metrics["abstract_content_checked"] = True
    text = " ".join(_strip_tex(raw).split())
    title_text = " ".join(_strip_tex(_braced_arg(tex, r"\title") or "").split())

    fi_abstract = bool(_FEATURE_IMPORTANCE_CLAIM.search(text))
    fi_title = bool(_FEATURE_IMPORTANCE_CLAIM.search(title_text))
    report.metrics["abstract_feature_importance_headline"] = (
        fi_abstract or fi_title
    )
    if fi_abstract or fi_title:
        where = (
            "title and abstract" if (fi_abstract and fi_title)
            else ("title" if fi_title else "abstract")
        )
        report.add(
            "warn", "abstract-headlines-feature-importance",
            f"feature-importance ranking language in the {where} — 0 of "
            "1,135 measured abstracts and 0 of 30 AERA Open full texts "
            "headline an importance ranking at any venue; state the "
            "substantive finding or the decision it feeds, and keep "
            "SHAP output as supporting evidence in Results",
        )

    sentences = [s for s in _SENT_SPLIT.split(text) if s.strip()]
    tail = " ".join(sentences[-2:]) if sentences else text
    names_decision = bool(_DECISION_TERMS.search(tail))
    report.metrics["abstract_names_decision"] = names_decision
    if not names_decision:
        report.add(
            "warn", "abstract-names-no-decision",
            "the closing abstract sentences name no specific practice, "
            "decision, or design the result feeds (anchors do this in "
            "7/34 abstracts; our papers in 1/13) — end with the concrete "
            "use, e.g. 'suggests reweighting engagement indicators in "
            "ninth-grade advising', without overclaiming beyond the "
            "evidence",
        )


# ---------------------------------------------------------------------------
# School-aware split contrast  (V5 Arc T H2 / VF2-06) — INFO metrics only
# ---------------------------------------------------------------------------
#
# Every school-aware prediction run already computes its headline metric
# over students in schools never seen during training — a cross-context
# generalization estimate. The replication is invisible unless the
# within-context number is reported beside it. These are metrics, never
# defects: the analysis may legitimately not have computed the
# within-context estimate, and its absence is an opportunity, not an
# error. Matching runs over _strip_tex output, so a contrast stated
# only inside a table environment is not detected.

_SCHOOL_AWARE_SPLIT = re.compile(
    r"school[- ]aware"
    r"|no school (?:appears|is present|occurs) in both"
    r"|schools?\s+(?:were\s+|was\s+)?(?:never|not)\s+(?:seen|present|included)"
    r"|(?:unseen|held[- ]out)\s+schools?"
    r"|grouped\s+(?:train[/-]test\s+)?split"
    r"|GroupShuffleSplit|StratifiedGroupKFold",
    re.IGNORECASE,
)
_WITHIN_CONTEXT = re.compile(
    r"within[- ](?:school|context|institution|cohort|sample)"
    r"|random(?:ly)?[- ]split",
    re.IGNORECASE,
)
_CROSS_CONTEXT = re.compile(
    r"(?:cross|out[- ]of)[- ](?:school|context|institution|cohort|sample)s?"
    r"|across\s+(?:schools|contexts|institutions|cohorts)",
    re.IGNORECASE,
)


def _check_split_contrast(report: LintReport, tex: str) -> None:
    """Record whether the manuscript states a school-aware split and,
    beside it, the within/cross-context contrast. Never adds a defect."""
    body = tex
    begin = tex.find(r"\begin{document}")
    if begin != -1:
        body = tex[begin:]
    text = " ".join(_strip_tex(body).split())
    school_aware = bool(_SCHOOL_AWARE_SPLIT.search(text))
    contrast = bool(_WITHIN_CONTEXT.search(text)) and bool(
        _CROSS_CONTEXT.search(text)
    )
    report.metrics["school_aware_split_stated"] = school_aware
    report.metrics["within_cross_contrast_stated"] = contrast
    report.metrics["school_aware_contrast_reported"] = school_aware and contrast


# ---------------------------------------------------------------------------
# Reference recency  (F-P5-DEPTH-RECENCY-SKEW)
# ---------------------------------------------------------------------------
#
# Arc P3 raised *how many* references reach the manuscript without
# constraining *which*. The shipped Arc P validation paper cited 19 of a
# 62-entry bibliography in which every single entry was dated 2024-2026:
# the count metric read "62 of 62 available references" (green) while the
# prose attributed DINA to Junker & Sijtsma and de la Torre with no
# citation at all, because no paper older than two years was reachable.
# A count is not a distribution. These checks measure the distribution.
#
# Profile source: docs/f_p5_citation_recency_spec.md §1, measured on 34
# LSAR anchors / 1,769 dated references.

_YEAR_FIELD = re.compile(r"\byear\s*=\s*[{\"']?\s*((?:19|20)\d{2})")

AGE_BUCKETS: tuple[str, ...] = ("le2", "3_5", "6_10", "11_20", "gt20")
#: Scarcest bin first — used by the composition side (src/citations.py) so
#: a tight budget protects the historical tail rather than the abundant
#: new work. Exported here so producer and checker cannot disagree.
FILL_ORDER: tuple[str, ...] = ("gt20", "11_20", "6_10", "3_5", "le2")

#: Pooled profile over all 34 anchors; used when the venue is unknown or
#: carries no ``ref_age`` block in data_registry/venue_norms.yaml.
DEFAULT_REF_AGE_PROFILE: dict[str, float] = {
    "le2": 0.246, "3_5": 0.237, "6_10": 0.241, "11_20": 0.170, "gt20": 0.106,
}
DEFAULT_REF_AGE_TOLERANCE_PP = 12.0

# An honestly short bibliography must never be punished: EDM has genuine
# 13-reference anchors, and a bin fraction over 8 references is noise.
MIN_REFS_FOR_DISTRIBUTION_CHECKS = 10
MIN_REFS_FOR_FOUNDATIONAL_FLOOR = 15

# §1.3 floors, near-universal in the anchor corpus.
FRAC_LE2_COLLAPSE = 0.90          # 1.00 on the shipped run
MIN_FRAC_OLDER_THAN_10 = 0.15     # forgiving floor; anchors sit at 26-29%


def bucket_of_age(age: int) -> str:
    """Age (years) -> fixed bin id. Shared by linter, ranker and miner."""
    if age <= 2:
        return "le2"
    if age <= 5:
        return "3_5"
    if age <= 10:
        return "6_10"
    if age <= 20:
        return "11_20"
    return "gt20"


def bib_entry_years(bib: str) -> dict[str, Optional[int]]:
    """``{citation key -> publication year or None}`` from references.bib.

    Entry bodies are sliced between successive ``_BIB_ENTRY`` matches
    rather than matched by a second regex, so an ``@`` that is indented
    or embedded in a field cannot swallow the entries that follow it.
    Never raises: an unparseable year is reported as ``None``.
    """
    out: dict[str, Optional[int]] = {}
    if not bib:
        return out
    matches = list(_BIB_ENTRY.finditer(bib))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(bib)
        body = bib[m.end():end]
        found = _YEAR_FIELD.search(body)
        year = int(found.group(1)) if found else None
        # Duplicate keys (biber warns, keeps one): take the first
        # occurrence that actually carries a year, so a stray undated
        # duplicate cannot erase a date we do have.
        if out.get(m.group(1)) is None:
            out[m.group(1)] = year
    return out


def _bucket_counts(ages: Iterable[int]) -> dict[str, int]:
    counts = {b: 0 for b in AGE_BUCKETS}
    for age in ages:
        counts[bucket_of_age(age)] += 1
    return counts


def _bucket_fractions(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    if not total:
        return {b: 0.0 for b in AGE_BUCKETS}
    return {b: counts[b] / total for b in AGE_BUCKETS}


def venue_age_profile(
    venue: Optional[str], norms: Optional[dict] = None
) -> tuple[dict[str, float], float, str]:
    """(profile, tolerance in pp, source label) for ``venue``.

    Falls back to the pooled default whenever the venue is unknown or its
    ``ref_age.buckets`` block is absent/incomplete/non-numeric — the
    venue_norms.yaml regeneration that adds these blocks lands separately
    (spec §5.5, step P5.1), so the fallback is the live path today.
    """
    block = ((norms or {}).get(venue or "") or {}).get("ref_age") or {}
    buckets = block.get("buckets")
    if isinstance(buckets, dict):
        try:
            profile = {b: float(buckets[b]) for b in AGE_BUCKETS}
        except (KeyError, TypeError, ValueError):
            profile = None
        if profile and sum(profile.values()) > 0:
            try:
                tol = float(block.get("tolerance_pp",
                                      DEFAULT_REF_AGE_TOLERANCE_PP))
            except (TypeError, ValueError):
                tol = DEFAULT_REF_AGE_TOLERANCE_PP
            return profile, tol, str(venue)
    return (dict(DEFAULT_REF_AGE_PROFILE), DEFAULT_REF_AGE_TOLERANCE_PP,
            "default")


def _check_reference_recency(
    report: LintReport,
    cited: set[str],
    bib: Optional[str],
    venue: Optional[str],
    norms: Optional[dict] = None,
    now_year: Optional[int] = None,
) -> None:
    """Age distribution of the CITED references vs the venue profile.

    Severity rationale: below-profile recency is structurally valid but
    worse scholarship, so it warns — the same treatment
    ``citations-below-venue-norm`` gets. The two §1.3 floors are errors:
    a bibliography with nothing older than a decade, or nothing older
    than 15 years, is the failure mode that burned a real review, and
    errors are surfaced first in the Arc P4 revision prompt.
    """
    if bib is None:
        return
    years = bib_entry_years(bib)
    if not years:
        return

    year_now = int(now_year) if now_year else datetime.now(timezone.utc).year
    profile, tolerance_pp, source = venue_age_profile(venue, norms)
    report.metrics["ref_age_target_fractions"] = {
        b: round(profile[b], 4) for b in AGE_BUCKETS
    }
    report.metrics["ref_age_profile_source"] = source
    report.metrics["ref_age_tolerance_pp"] = tolerance_pp
    report.metrics["ref_age_now_year"] = year_now

    def _age(year: int) -> int:
        return max(0, year_now - year)

    # --- whole .bib: "what was available" ---------------------------------
    bib_ages = [_age(y) for y in years.values() if y is not None]
    bib_counts = _bucket_counts(bib_ages)
    bib_fracs = _bucket_fractions(bib_counts)
    report.metrics["bib_age_buckets"] = bib_counts
    report.metrics["bib_age_fractions"] = {
        b: round(bib_fracs[b], 4) for b in AGE_BUCKETS
    }

    # --- cited subset: "what the reader sees" -----------------------------
    cited_in_bib = [k for k in years if k in cited]
    parsed = [y for y in (years[k] for k in cited_in_bib) if y is not None]
    ages = sorted(_age(y) for y in parsed)
    n = len(ages)
    counts = _bucket_counts(ages)
    fracs = _bucket_fractions(counts)
    n_gt10 = sum(1 for a in ages if a > 10)
    n_gt15 = sum(1 for a in ages if a > 15)
    frac_gt10 = (n_gt10 / n) if n else 0.0

    report.metrics["ref_year_parsed"] = n
    report.metrics["ref_year_missing"] = len(cited_in_bib) - n
    report.metrics["ref_age_median"] = statistics.median(ages) if ages else None
    report.metrics["ref_age_mean"] = (
        round(statistics.fmean(ages), 2) if ages else None
    )
    report.metrics["ref_age_max"] = max(ages) if ages else None
    report.metrics["ref_age_buckets"] = counts
    report.metrics["ref_age_fractions"] = {
        b: round(fracs[b], 4) for b in AGE_BUCKETS
    }
    report.metrics["n_refs_older_than_10"] = n_gt10
    report.metrics["n_refs_older_than_15"] = n_gt15
    report.metrics["frac_refs_older_than_10"] = round(frac_gt10, 4)
    report.metrics["n_refs_pre_2000"] = sum(1 for y in parsed if y < 2000)

    if n >= MIN_REFS_FOR_DISTRIBUTION_CHECKS:
        if fracs["le2"] >= FRAC_LE2_COLLAPSE:
            report.add(
                "error", "reference-recency-collapse",
                f"{fracs['le2'] * 100:.0f}% of the {n} dated references are "
                f"<= 2 years old (target {profile['le2'] * 100:.0f}%) — the "
                "bibliography cites no prior-decade scholarship, so "
                "foundational work is described in prose without a citation",
            )
        offenders = [
            (b, (fracs[b] - profile[b]) * 100.0)
            for b in AGE_BUCKETS
            if abs((fracs[b] - profile[b]) * 100.0) > tolerance_pp
        ]
        if offenders:
            detail = ", ".join(
                f"{b} {delta:+.1f}pp "
                f"({fracs[b] * 100:.0f}% vs {profile[b] * 100:.0f}%)"
                for b, delta in offenders
            )
            report.add(
                "warn", "reference-recency-skew",
                f"reference age distribution is off the {source} profile by "
                f"more than {tolerance_pp:g}pp in {len(offenders)} bin(s): "
                + detail,
            )
        if frac_gt10 < MIN_FRAC_OLDER_THAN_10:
            report.add(
                "warn", "thin-historical-tail",
                f"only {frac_gt10 * 100:.0f}% of references are older than 10 "
                f"years ({n_gt10}/{n}); anchor papers at every venue sit at "
                "26-29%",
            )

    if n >= MIN_REFS_FOR_FOUNDATIONAL_FLOOR and n_gt15 == 0:
        report.add(
            "error", "no-foundational-references",
            f"none of the {n} dated references is older than 15 years; 94% of "
            "anchor papers (and 100% of the journals) cite older work, with a "
            "median of 7-8 such references",
        )

    # The pool, not just the paper: catches the upstream retrieval skew
    # even when the model happened to cite the one old record it had.
    n_uncited = len(years) - len(cited_in_bib)
    if (
        n_uncited > 0
        and len(bib_ages) >= MIN_REFS_FOR_DISTRIBUTION_CHECKS
        and bib_fracs["le2"] >= FRAC_LE2_COLLAPSE
    ):
        report.add(
            "warn", "bib-recency-collapse",
            f"{bib_fracs['le2'] * 100:.0f}% of the {len(bib_ages)} dated "
            "references.bib entries are <= 2 years old — the retrieved pool "
            "itself contains no older work, so no revision can cite any",
        )


# ---------------------------------------------------------------------------
# Compile-log checks
# ---------------------------------------------------------------------------

_UNDEF_CITATION = re.compile(r"LaTeX Warning: Citation [`']([^'\s]+)'")
_UNDEF_REFERENCE = re.compile(r"LaTeX Warning: Reference [`']([^'\s]+)'")
_OVERFULL = re.compile(r"Overfull \\hbox \((\d+(?:\.\d+)?)pt")


def _check_latex_log(report: LintReport, log: str) -> None:
    undef_c = sorted(set(_UNDEF_CITATION.findall(log)))
    if undef_c:
        report.add(
            "error", "undefined-citation",
            f"{len(undef_c)} citation(s) undefined at compile time: "
            + ", ".join(undef_c[:8]) + ("…" if len(undef_c) > 8 else ""),
        )
    undef_r = sorted(set(_UNDEF_REFERENCE.findall(log)))
    if undef_r:
        report.add(
            "error", "undefined-reference",
            f"{len(undef_r)} cross-reference(s) undefined: "
            + ", ".join(undef_r[:8]),
        )
    if "There were undefined references" in log and not (undef_c or undef_r):
        report.add("error", "undefined-references-flag",
                   "compiler reports undefined references")
    over = [float(x) for x in _OVERFULL.findall(log)]
    report.metrics["n_overfull_hboxes"] = len(over)
    report.metrics["worst_overfull_pt"] = max(over) if over else 0.0
    if over and (len(over) > 10 or max(over) > 30):
        report.add(
            "warn", "overfull-hboxes",
            f"{len(over)} overfull hbox(es), worst {max(over):.0f}pt — "
            "text runs into the margin",
        )


def _check_biber_log(report: LintReport, blg: str) -> None:
    warns = re.findall(r"^.*WARN - (.+)$", blg, re.MULTILINE)
    errors = re.findall(r"^.*ERROR - (.+)$", blg, re.MULTILINE)
    report.metrics["n_biber_warnings"] = len(warns)
    for msg in errors[:5]:
        report.add("error", "biber-error", msg.strip())
    if len(warns) > 0:
        report.add(
            "warn", "biber-warnings",
            f"{len(warns)} biber warning(s), e.g.: {warns[0].strip()}",
        )


def _check_pdf_text(report: LintReport, pdf_path: Path) -> None:
    try:
        import fitz  # type: ignore[import-not-found]
    except ImportError:
        report.metrics["pdf_text_checked"] = False
        return
    try:
        doc = fitz.open(pdf_path)
        text = "".join(page.get_text() for page in doc)
        doc.close()
    except Exception as exc:  # unreadable PDF is itself a defect
        report.add("error", "pdf-unreadable", f"could not read PDF: {exc}")
        return
    report.metrics["pdf_text_checked"] = True
    n_qq = len(re.findall(r"\?\?", text))
    report.metrics["n_question_marks"] = n_qq
    if n_qq:
        report.add(
            "error", "question-mark-refs",
            f"{n_qq} '??' occurrence(s) in the rendered PDF "
            "(unresolved citations or cross-references)",
        )


# ---------------------------------------------------------------------------
# Venue norms
# ---------------------------------------------------------------------------

DEFAULT_NORMS_PATH = (
    Path(__file__).resolve().parent.parent / "data_registry" / "venue_norms.yaml"
)


def load_venue_norms(path: Optional[Path] = None) -> dict:
    import yaml

    norms_path = Path(path) if path else DEFAULT_NORMS_PATH
    if not norms_path.exists():
        return {}
    with open(norms_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("venues", {}) or {}


def _check_venue_norms(report: LintReport, venue: str, norms: dict) -> None:
    block = norms.get(venue)
    if not block:
        report.metrics["venue_norms_applied"] = False
        return
    report.metrics["venue_norms_applied"] = True
    refs_floor = (block.get("refs") or {}).get("p25")
    n_cited = report.metrics.get("n_citations_distinct")
    if refs_floor is not None and n_cited is not None:
        report.metrics["venue_refs_p25"] = refs_floor
        if n_cited < refs_floor:
            report.add(
                "warn", "citations-below-venue-norm",
                f"{n_cited} distinct citations vs {venue} anchor P25 of "
                f"{refs_floor:g} (median "
                f"{(block.get('refs') or {}).get('median', '?')}) — real "
                f"{venue} papers cite substantially more prior work",
            )
    words_floor = (block.get("body_words") or {}).get("p25")
    body_words = report.metrics.get("body_words")
    if words_floor is not None and body_words is not None:
        report.metrics["venue_body_words_p25"] = words_floor
        if body_words < 0.8 * words_floor:
            report.add(
                "warn", "length-below-venue-norm",
                f"~{body_words} body words vs {venue} anchor P25 of "
                f"{words_floor} — manuscript is short for the venue",
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def lint_manuscript(
    run_dir: Path,
    venue: Optional[str] = None,
    norms_path: Optional[Path] = None,
    tex_name: str = "paper.tex",
    write_json: bool = True,
    now_year: Optional[int] = None,
) -> LintReport:
    """Lint a compiled manuscript in ``run_dir``; returns a LintReport.

    Looks for ``paper.tex``/``references.bib`` plus any ``.log``/``.blg``
    and ``paper*.pdf`` produced by the compile. Missing artifacts skip
    their checks (recorded in metrics) — the linter itself never raises.

    ``now_year`` overrides the manuscript year used for reference-age
    arithmetic (default: current UTC year); it exists for test
    determinism only.
    """
    run_dir = Path(run_dir)
    report = LintReport(venue=venue)
    # Hoisted above the tex branch: the recency check needs the norms
    # whether or not a venue was supplied (unknown venue -> pooled default).
    norms = load_venue_norms(norms_path)

    tex_path = run_dir / tex_name
    if not tex_path.exists():
        report.add("error", "missing-tex", f"{tex_name} not found in {run_dir}")
    else:
        tex = tex_path.read_text(encoding="utf-8", errors="replace")
        bib_path = run_dir / "references.bib"
        bib = (
            bib_path.read_text(encoding="utf-8", errors="replace")
            if bib_path.exists()
            else None
        )
        _check_front_matter(report, tex)
        _check_tex(report, tex, bib)
        _check_abstract_content(report, tex)
        _check_split_contrast(report, tex)
        _check_reference_recency(
            report, _real_cited_keys(tex), bib, venue, norms, now_year
        )
        _check_unverified_flag(report, tex, run_dir)
        _check_numeric_reconciliation(report, tex, run_dir)

    logs = sorted(run_dir.glob("*.log"))
    latex_logs = [p for p in logs if p.stem.startswith("paper")]
    for log_path in latex_logs:
        _check_latex_log(
            report, log_path.read_text(encoding="utf-8", errors="replace")
        )
    report.metrics["n_compile_logs_checked"] = len(latex_logs)

    for blg_path in sorted(run_dir.glob("*.blg")):
        _check_biber_log(
            report, blg_path.read_text(encoding="utf-8", errors="replace")
        )

    pdfs = sorted(run_dir.glob("paper*.pdf"))
    if pdfs:
        _check_pdf_text(report, pdfs[-1])

    if venue:
        _check_venue_norms(report, venue, norms)

    # Dedupe: paper.log and paper_for_review.log often carry the same
    # warnings; identical defects collapse to one.
    seen: set[tuple[str, str, str]] = set()
    unique: list[LintDefect] = []
    for d in report.defects:
        key = (d.severity, d.code, d.message)
        if key not in seen:
            seen.add(key)
            unique.append(d)
    report.defects = unique

    if write_json:
        try:
            (run_dir / "manuscript_lint.json").write_text(
                json.dumps(report.to_dict(), indent=2), encoding="utf-8"
            )
        except OSError:
            pass
    return report
