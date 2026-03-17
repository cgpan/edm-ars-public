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

    # lq_13: Zero \cite{} commands when document has \bibliography
    if r"\bibliography{" in latex and not re.search(r"\\cite\{", latex):
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
