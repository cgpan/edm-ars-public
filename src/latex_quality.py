"""Regex-based LaTeX content quality gate for EDM-ARS Writer output.

Inspired by AutoResearchClaw quality.py: 12 patterns catch placeholder/crutch content
in generated LaTeX that would produce an unfinished or hollow paper.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class LatexQualityIssue:
    pattern_id: str
    severity: str  # "error" | "warning"
    matched_text: str
    context: str  # ~60 chars surrounding the match for diagnosis
    line_number: int


@dataclass
class LatexQualityReport:
    issues: list[LatexQualityIssue] = field(default_factory=list)
    template_ratio: float = 0.0
    total_chars: int = 0
    matched_chars: int = 0

    @property
    def has_errors(self) -> bool:
        return any(i.severity == "error" for i in self.issues)

    @property
    def exceeds_ratio_threshold(self) -> bool:
        return self.template_ratio > _TEMPLATE_RATIO_THRESHOLD

    def to_warning_strings(self) -> list[str]:
        out: list[str] = []
        for issue in self.issues:
            out.append(
                f"[{issue.severity.upper()}] {issue.pattern_id} (line {issue.line_number}): "
                f"'{issue.matched_text[:60]}' — context: '{issue.context}'"
            )
        if self.exceeds_ratio_threshold:
            out.append(
                f"Template ratio {self.template_ratio:.4f} exceeds threshold "
                f"{_TEMPLATE_RATIO_THRESHOLD} "
                f"({self.matched_chars}/{self.total_chars} chars matched by crutch patterns)"
            )
        return out


# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

# Fraction of total chars matched by crutch patterns above which we flag globally.
# Mirrors AutoResearchClaw's quality gate default of 5% (we use 0.5% since EDM papers
# have specific known-good content and any crutch phrase is a signal).
_TEMPLATE_RATIO_THRESHOLD = 0.005

# (pattern_id, regex_string, severity)
# Severity "error" = must fix before paper is acceptable
# Severity "warning" = worth noting, may be intentional
_RAW_PATTERNS: list[tuple[str, str, str]] = [
    # Suppressed / hidden results
    ("lq_01", r"\(not shown\)", "error"),
    # Unfilled structural placeholders
    ("lq_02", r"\[Insert\s+(?:table|figure|result|graph|chart|plot|image|diagram)[^\]]*\]", "error"),
    # Development markers
    ("lq_03", r"\bTODO\b", "error"),
    ("lq_04", r"\bFIXME\b", "error"),
    # Ellipsis with completion comment (e.g. \ldots % fill in later)
    ("lq_05", r"\\ldots\s*%\s*(?:fill|complete|expand|todo|add)", "error"),
    # Unfilled citation placeholders
    ("lq_06", r"\[(?:Author(?:,?\s*Year)?|Citation\s+needed|REF)\]", "error"),
    # Content suppression excuses
    ("lq_07", r"(?:omitted\s+for\s+brevity|results?\s+not\s+shown\s+here?)", "warning"),
    # Future-tense completion placeholders
    ("lq_08", r"(?:will\s+be\s+discussed|to\s+be\s+determined|to\s+be\s+added|to\s+be\s+filled\s+in)", "warning"),
    # Unfilled %%PLACEHOLDER%% template markers
    ("lq_09", r"%%PLACEHOLDER:\w+%%", "error"),
    # References to supplementary/appendix that may not exist
    ("lq_10", r"(?:see\s+(?:the\s+)?(?:appendix|supplementary\s+material)\s+for\s+details?)", "warning"),
    # Explicit needs-citation marker
    ("lq_11", r"\[NEEDS\s+CITATION\]", "error"),
    # Vague cross-references hiding missing content
    ("lq_12", r"(?:described\s+in\s+detail\s+elsewhere|as\s+described\s+elsewhere)", "warning"),
]

# Compile with IGNORECASE
_COMPILED_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    (pid, re.compile(pat, re.IGNORECASE), sev)
    for pid, pat, sev in _RAW_PATTERNS
]

#: Any citation command, natbib or biblatex. Kept identical to
#: ``src.invariants._CITE_CMD``; see the note there.
_CITE_CMD = re.compile(r"\\[a-zA-Z]*cite[a-zA-Z]*\*?\s*(?:\[[^\]]*\]\s*)*\{([^}]*)\}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_latex_quality(latex: str) -> LatexQualityReport:
    """Scan a LaTeX document string for placeholder and crutch content.

    Returns a :class:`LatexQualityReport` with a list of issues and aggregate
    statistics. Does NOT raise exceptions — all errors are captured in the report.
    """
    report = LatexQualityReport(total_chars=len(latex))
    total_matched = 0

    for pid, pattern, severity in _COMPILED_PATTERNS:
        for match in pattern.finditer(latex):
            line_no = latex[: match.start()].count("\n") + 1
            ctx_start = max(0, match.start() - 30)
            ctx_end = min(len(latex), match.end() + 30)
            context = latex[ctx_start:ctx_end].replace("\n", " ").strip()

            report.issues.append(
                LatexQualityIssue(
                    pattern_id=pid,
                    severity=severity,
                    matched_text=match.group(0),
                    context=context,
                    line_number=line_no,
                )
            )
            total_matched += len(match.group(0))

    report.matched_chars = total_matched
    if report.total_chars > 0:
        report.template_ratio = total_matched / report.total_chars

    # --- Structural checks (not regex-match-per-instance) ---

    # lq_13: Zero citation commands when document has \bibliography.
    #
    # This tested for `\cite{` alone. The journal template loads biblatex
    # and the Writer emits `\parencite{...}`, which contains "cite" but
    # does not start with it -- so a fully-cited journal paper tripped
    # this as an ERROR, and a genuinely uncited one would have been
    # indistinguishable from it. Match any citation command.
    if r"\bibliography{" in latex and not _CITE_CMD.search(latex):
        report.issues.append(
            LatexQualityIssue(
                pattern_id="lq_13",
                severity="error",
                matched_text="(no \\cite{} commands found)",
                context="Paper has \\bibliography but zero \\cite{} commands — references will be empty",
                line_number=0,
            )
        )

    # lq_15: includegraphics without leading backslash (broken image)
    for m in re.finditer(r"(?<!\\)(?:^|\n)\s*includegraphics\b", latex):
        line_no = latex[: m.start()].count("\n") + 1
        report.issues.append(
            LatexQualityIssue(
                pattern_id="lq_15",
                severity="error",
                matched_text=m.group(0).strip(),
                context="Missing backslash: should be \\includegraphics",
                line_number=line_no,
            )
        )

    # lq_16: broken \ref — e.g. "Table \ foo" or "Table \ref" without braces
    for m in re.finditer(
        r"(?:Table|Figure|Section|Equation)\s+\\?\s+[a-zA-Z_]+(?!.*\\ref\{)",
        latex,
    ):
        # Only flag if the line does NOT contain a proper \ref{...}
        line_start = latex.rfind("\n", 0, m.start()) + 1
        line_end = latex.find("\n", m.end())
        if line_end == -1:
            line_end = len(latex)
        line_text = latex[line_start:line_end]
        if r"\ref{" not in line_text and r"\label{" not in line_text:
            line_no = latex[: m.start()].count("\n") + 1
            report.issues.append(
                LatexQualityIssue(
                    pattern_id="lq_16",
                    severity="warning",
                    matched_text=m.group(0)[:60],
                    context="Possible broken cross-reference — expected \\ref{label}",
                    line_number=line_no,
                )
            )

    # lq_14: \resizebox on narrow tables (fewer than 5 columns)
    for m in re.finditer(
        r"\\resizebox\{[^}]*\}\{[^}]*\}\{[^}]*\\begin\{tabular\}\{([^}]*)\}",
        latex,
        re.DOTALL,
    ):
        col_spec = re.sub(r"[^lrcpLRCPmMbBXSd]", "", m.group(1))
        if len(col_spec) < 5:
            line_no = latex[: m.start()].count("\n") + 1
            report.issues.append(
                LatexQualityIssue(
                    pattern_id="lq_14",
                    severity="warning",
                    matched_text=m.group(0)[:80],
                    context=f"\\resizebox on {len(col_spec)}-column table makes font too large",
                    line_number=line_no,
                )
            )

    return report


# ---------------------------------------------------------------------------
# Deterministic repair: table notes outside a threeparttable
# ---------------------------------------------------------------------------
#
# Round 3 on the owner's Mac: paper.log recorded 18 errors, from "You can't
# use \prevdepth in restricted horizontal mode" to "\begin{table} on input
# line 280 ended by \end{tablenotes}", and the subgroup table never
# rendered -- its label went with it, so the text read "Table ?? reports
# AUC separately by sex". Compiling the candidate shapes one by one
# reproduces that error list exactly for a tablenotes block INSIDE a
# \resizebox argument with no threeparttable around it: \resizebox sets its
# argument in restricted horizontal mode, and a tablenotes list cannot live
# there. The table skill taught "wrap the entire threeparttable block
# inside \resizebox"; drop the threeparttable from that and this is what
# is left.
#
# tablenotes belongs in a threeparttable, after the (possibly resized)
# tabular and outside any box. That is decidable from the source, so it is
# repaired here rather than asked of the model.

_TABLE_FLOAT = re.compile(r"(\\begin\{(table\*?)\})(.*?)(\\end\{\2\})", re.DOTALL)
_TABLENOTES = re.compile(r"\\begin\{tablenotes\}.*?\\end\{tablenotes\}", re.DOTALL)
#: Box commands that set their last mandatory argument in restricted
#: horizontal mode, and how many mandatory arguments each takes.
_BOX_ARITY = {"resizebox": 3, "scalebox": 2, "adjustbox": 2}
_BOX_START = re.compile(r"\\(resizebox|scalebox|adjustbox)\*?(?![A-Za-z])")
_TABLE_START = re.compile(
    r"\\(?:resizebox|scalebox|adjustbox)\*?(?![A-Za-z])|\\begin\{tabular[x*]?\}"
)


def _group_end(text: str, i: int, open_ch: str, close_ch: str) -> int | None:
    """Index just past the group that opens at ``text[i]``, or None."""
    depth = 0
    j = i
    while j < len(text):
        ch = text[j]
        if ch == "\\":
            j += 2
            continue
        if ch == "%":  # a comment runs to the end of its line
            nl = text.find("\n", j)
            j = len(text) if nl < 0 else nl + 1
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return j + 1
        j += 1
    return None


def _box_argument(text: str, match: re.Match) -> tuple[int, int, int] | None:
    """``(content_start, content_end, box_end)`` of a box command's last
    mandatory argument, or None when the source does not parse."""
    wanted = _BOX_ARITY[match.group(1)]
    i = match.end()
    seen = 0
    content = None
    while seen < wanted:
        while i < len(text) and text[i] in " \t\r\n":
            i += 1
        if i >= len(text):
            return None
        if text[i] == "[":
            end = _group_end(text, i, "[", "]")
            if end is None:
                return None
            i = end
            continue
        if text[i] != "{":
            return None
        end = _group_end(text, i, "{", "}")
        if end is None:
            return None
        seen += 1
        content = (i + 1, end - 1, end)
        i = end
    return content


def _move_notes_out_of_boxes(body: str) -> tuple[str, bool]:
    """Take tablenotes out of a box argument that holds no threeparttable
    of its own, and put them right after the box."""
    changed = False
    pos = 0
    while True:
        m = _BOX_START.search(body, pos)
        if m is None:
            return body, changed
        arg = _box_argument(body, m)
        if arg is None:
            pos = m.end()
            continue
        start, end, box_end = arg
        content = body[start:end]
        notes = [n.group(0) for n in _TABLENOTES.finditer(content)]
        if not notes or "\\begin{threeparttable}" in content:
            pos = box_end
            continue
        inner = _TABLENOTES.sub("", content).rstrip()
        if not inner.endswith("%"):
            inner += "%"
        body = (
            body[:start] + inner + "\n" + body[end:box_end]
            + "\n" + "\n".join(notes) + body[box_end:]
        )
        changed = True
        pos = start


def repair_table_notes(latex: str) -> tuple[str, int]:
    """Put every table's ``tablenotes`` inside a ``threeparttable``.

    For each table float that uses ``tablenotes``:

    * notes inside a ``\\resizebox`` / ``\\scalebox`` / ``\\adjustbox``
      argument that holds no threeparttable of its own move to just after
      the box;
    * a float with no ``threeparttable`` gets one around its table, from
      the first box or ``tabular`` to the last ``tablenotes``, so the
      caption and label stay where the Writer put them.

    A table already in a working shape -- including a whole threeparttable
    inside a ``\\resizebox`` -- is left as it is. Returns the text and the
    number of tables changed.
    """
    if "\\begin{tablenotes}" not in latex:
        return latex, 0
    repaired = 0

    def fix(m: re.Match) -> str:
        nonlocal repaired
        begin, _, body, end = m.groups()
        if "\\begin{tablenotes}" not in body:
            return m.group(0)
        new_body, moved = _move_notes_out_of_boxes(body)
        wrapped = False
        if "\\begin{threeparttable}" not in new_body:
            first = _TABLE_START.search(new_body)
            last = None
            for last in _TABLENOTES.finditer(new_body):
                pass
            if first is not None and last is not None and first.start() < last.start():
                new_body = (
                    new_body[: first.start()]
                    + "\\begin{threeparttable}\n"
                    + new_body[first.start(): last.end()]
                    + "\n\\end{threeparttable}"
                    + new_body[last.end():]
                )
                wrapped = True
        if moved or wrapped:
            repaired += 1
            return begin + new_body + end
        return m.group(0)

    return _TABLE_FLOAT.sub(fix, latex), repaired
