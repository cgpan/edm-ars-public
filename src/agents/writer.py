"""Writer agent: synthesises all pipeline outputs into a complete LaTeX research paper."""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any

from src.agents.base import BaseAgent
from src.citations import (
    build_bibtex,
    format_citation_key_block,
    reconcile_citations,
    venue_citation_target,
)
from src.latex_quality import LatexQualityReport, check_latex_quality
from src.manuscript_linter import (
    UNVERIFIED_BLOCK,
    UNVERIFIED_MARKER,
    run_is_unverified,
)

#: I1a (AERA_OPEN audit): the Writer invented values for every null field
#: in its rendered results JSON. Nulls are now rendered as this loud
#: marker so "make something up" is never the path of least resistance.
NOT_AVAILABLE_MARKER = (
    "NOT AVAILABLE - this value was not computed; "
    "do NOT report a number for it"
)


def _mark_null_values(obj: Any) -> Any:
    """Deep-copy a JSON-like structure replacing None leaves with
    :data:`NOT_AVAILABLE_MARKER` (prompt rendering only — artifacts on
    disk keep real nulls)."""
    if obj is None:
        return NOT_AVAILABLE_MARKER
    if isinstance(obj, dict):
        return {k: _mark_null_values(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_mark_null_values(v) for v in obj]
    return obj

def _count_tabular_spec_cols(spec: str) -> int:
    """Count column specifiers in a LaTeX tabular spec string, e.g. 'lrrrr' → 5."""
    # Remove brace groups (e.g. p{3cm}, @{}, >{}) and vertical bars
    cleaned = re.sub(r"\{[^}]*\}", "", spec)
    return len(re.findall(r"[lrcpmb]", cleaned))


def _check_tabular_column_counts(latex: str) -> list[str]:
    """Return warnings for any tabular environment where spec column count != row column count."""
    warnings: list[str] = []
    tabular_re = re.compile(
        r"\\begin\{tabular\}\{([^}]+)\}(.*?)\\end\{tabular\}", re.DOTALL
    )
    # Lines that are not data rows
    skip_re = re.compile(
        r"^\s*\\(toprule|midrule|bottomrule|hline|cline|multicolumn|caption|label)\b"
    )
    for m in tabular_re.finditer(latex):
        spec = m.group(1)
        body = m.group(2)
        spec_cols = _count_tabular_spec_cols(spec)
        for raw_row in body.split("\\\\"):
            row = raw_row.strip()
            if not row or skip_re.match(row):
                continue
            # Count unescaped & characters
            clean = re.sub(r"\\&", "", row)
            row_cols = clean.count("&") + 1
            if row_cols != spec_cols:
                snippet = row[:60].replace("\n", " ")
                warnings.append(
                    f"Tabular column mismatch: spec '{{{spec}}}' = {spec_cols} cols "
                    f"but row has {row_cols} cols. Row: '{snippet}'"
                )
    return warnings


def _check_figure_ref_label_pairs(latex: str) -> list[str]:
    """Warn when a \\ref{fig:xxx} has no matching \\label{fig:xxx} in a figure environment."""
    warnings: list[str] = []
    refs = set(re.findall(r"\\ref\{(fig:[^}]+)\}", latex))
    labels = set(re.findall(r"\\label\{(fig:[^}]+)\}", latex))
    for ref in sorted(refs):
        if ref not in labels:
            warnings.append(
                f"\\ref{{{ref}}} has no matching \\label{{{ref}}} — will render as '??' in PDF. "
                "Add a full \\begin{figure}...\\end{figure} block with this label."
            )
    return warnings


def _check_wide_table_resizebox(latex: str) -> list[str]:
    """Warn when a tabular with 5+ columns is not wrapped in \\resizebox."""
    warnings: list[str] = []
    tabular_re = re.compile(r"\\begin\{tabular\}\{([^}]+)\}", re.DOTALL)
    for m in tabular_re.finditer(latex):
        spec = m.group(1)
        n_cols = _count_tabular_spec_cols(spec)
        if n_cols >= 5:
            # Check if \resizebox appears within ~200 chars before this tabular
            start = max(0, m.start() - 200)
            context_before = latex[start : m.start()]
            if r"\resizebox" not in context_before:
                warnings.append(
                    f"Wide table ({n_cols} columns, spec '{{{spec}}}') is not wrapped in "
                    r"\resizebox{\columnwidth}{!}{...} — will overflow column width in sigconf layout."
                )
    return warnings


_S2_FAILURE_BIB_COMMENT = (
    "% Semantic Scholar API was unavailable; citations are placeholders only.\n"
)

def _extract_braced_arg(text: str, command: str) -> str:
    """Return the balanced-brace argument of ``command`` (e.g. ``\\abstract``).

    A plain regex cannot do this: abstracts contain nested braces
    (``\\emph{...}``, ``$x_{1}$``), and a non-greedy match stops at the
    first closing brace, silently truncating the text.
    """
    idx = text.find(command + "{")
    if idx == -1:
        return ""
    start = idx + len(command) + 1
    depth = 1
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{" and text[i - 1] != "\\":
            depth += 1
        elif ch == "}" and text[i - 1] != "\\":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
    return ""


# Sentinel used when template cannot be loaded at all
_MINIMAL_STUB_TEX = (
    r"\documentclass[sigconf]{acmart}" + "\n"
    r"\usepackage{booktabs}" + "\n"
    r"\usepackage{graphicx}" + "\n"
    r"\begin{document}" + "\n"
    r"\begin{abstract}" + "\n"
    r"% Writer agent could not parse LaTeX from LLM response." + "\n"
    r"\end{abstract}" + "\n"
    r"\maketitle" + "\n"
    r"\end{document}" + "\n"
)


class Writer(BaseAgent):
    """Generates paper.tex (ACM acmart sigconf) and references.bib from pipeline outputs."""

    def run(
        self,
        research_spec: dict | None = None,
        literature_context: dict | None = None,
        data_report: dict | None = None,
        results_object: dict | None = None,
        review_report: dict | None = None,
        outline: dict | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Args:
            research_spec: ProblemFormulator output (falls back to ctx).
            literature_context: S2 papers + novelty evidence (falls back to ctx).
            data_report: DataEngineer output (falls back to ctx).
            results_object: Analyst output (falls back to ctx).
            review_report: Critic output (falls back to ctx).
            outline: OutlineAgent output (None → v1 placeholder-filling path).

        Returns:
            The full LaTeX paper text (also written to ``paper.tex``).
        """
        spec = research_spec if research_spec is not None else self.ctx.research_spec
        lit = literature_context if literature_context is not None else self.ctx.literature_context
        report = data_report if data_report is not None else self.ctx.data_report
        results = results_object if results_object is not None else self.ctx.results_object
        review = review_report if review_report is not None else self.ctx.review_report

        # Fall back to ctx outline if not provided directly
        if outline is None:
            outline = getattr(self.ctx, "paper_outline", None)

        # Sanitize paper IDs for BibTeX compatibility (: → _ in keys)
        lit = self._sanitize_paper_ids(lit)

        # Build BibTeX pre-populated from S2 metadata (LLM may refine/override)
        fallback_bibtex = self._build_bibtex(lit)

        # Choose template and message builder based on outline availability
        # V4 wave-2: journal venue format selects the APA7 manuscript
        # template and an ~8000-word budget (config writer.venue_format).
        venue_format = self.config.get("writer", {}).get(
            "venue_format", "conference")
        if outline is not None:
            template_text = self._load_template(
                version="journal" if venue_format == "journal" else "v2")
            user_message = self._build_user_message_with_outline(
                outline=outline,
                research_spec=spec,
                literature_context=lit,
                data_report=report,
                results_object=results,
                review_report=review,
                template_text=template_text,
            )
        else:
            template_text = self._load_template()
            user_message = self._build_user_message(
                research_spec=spec,
                literature_context=lit,
                data_report=report,
                results_object=results,
                review_report=review,
                template_text=template_text,
            )

        if venue_format == "journal" and outline is not None:
            # E2a: journal manuscripts are generated SECTION BY SECTION
            # (a single 16k-token call cannot reach ~8000 words), then
            # assembled into a synthetic full-latex document that flows
            # through the same reassembly path below.
            paper_tex, bibtex = self._run_journal_sectionwise(
                base_context=user_message, fallback_bibtex=fallback_bibtex,
            )
        else:
            llm_response = self.call_llm(user_message, max_tokens=self.max_tokens)
            paper_tex = self._extract_latex(llm_response)
            bibtex = self._extract_bibtex(llm_response) or fallback_bibtex

        # v2 path: reassemble from clean template to prevent preamble corruption.
        # The LLM often modifies \makeatletter / \renewcommand\@copyrightpermission
        # blocks, causing broken first pages in the compiled PDF.
        if outline is not None and paper_tex not in (_MINIMAL_STUB_TEX,):
            paper_tex = self._reassemble_from_template(paper_tex, template_text)

        # Validate template structure and log any warnings
        if paper_tex not in (_MINIMAL_STUB_TEX, template_text):
            structure_warnings = self._validate_template_structure(paper_tex)
            for w in structure_warnings:
                self.ctx.log.append(
                    {
                        "timestamp": datetime.utcnow().isoformat(),
                        "agent": self.agent_name,
                        "message": f"LaTeX structure warning: {w}",
                    }
                )

            # Deterministic fix: LLM sometimes drops backslash on \includegraphics
            paper_tex = re.sub(
                r"(?m)^(\s*)includegraphics\b",
                r"\1\\includegraphics",
                paper_tex,
            )

            # Crutch-phrase / placeholder quality scan (inspired by AutoResearchClaw quality.py)
            quality_report = check_latex_quality(paper_tex)
            for w in quality_report.to_warning_strings():
                self.ctx.log.append(
                    {
                        "timestamp": datetime.utcnow().isoformat(),
                        "agent": self.agent_name,
                        "message": f"LaTeX quality warning: {w}",
                    }
                )
            # One repair attempt if fixable errors found
            if quality_report.has_errors:
                repair_msg = self._build_quality_repair_prompt(paper_tex, quality_report)
                repaired_response = self.call_llm(repair_msg, max_tokens=self.max_tokens)
                repaired_tex = self._extract_latex(repaired_response)
                repaired_bib = self._extract_bibtex(repaired_response)
                # Only accept repair if it is a complete document (has \documentclass)
                if (
                    repaired_tex not in (_MINIMAL_STUB_TEX, template_text)
                    and r"\documentclass" in repaired_tex
                ):
                    paper_tex = repaired_tex
                    if repaired_bib:
                        bibtex = repaired_bib

        # F-A5-MISSING-BIBLIOGRAPHY (Phase A attempt 3): deterministic
        # guard on every path — a paper without \bibliography compiles to
        # a PDF with no References section, which the LSAR sanity check
        # rightly rejects. Inject the standard commands before
        # \end{document} when both bibliography forms are absent.
        if (
            "\\bibliography{" not in paper_tex
            and "\\begin{thebibliography}" not in paper_tex
            and "\\printbibliography" not in paper_tex  # biblatex (journal)
            and "\\end{document}" in paper_tex
        ):
            paper_tex = paper_tex.replace(
                "\\end{document}",
                "\\bibliographystyle{ACM-Reference-Format}\n"
                "\\bibliography{references}\n\n"
                "\\end{document}",
                1,
            )
            self.ctx.log.append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "agent": self.agent_name,
                    "message": (
                        "Injected missing \\bibliographystyle/\\bibliography "
                        "before \\end{document} (F-A5 deterministic guard)."
                    ),
                }
            )

        # Arc P3: deterministic citation reconciliation. This is the last
        # point at which paper_tex and bibtex are both final and still in
        # memory — after the sectionwise assembly, the template
        # reassembly, the quality-repair branch (which can replace
        # `bibtex`) and the F-A5 guard, but before either touches disk.
        #
        # When real retrieved papers exist, the bibliography is rebuilt
        # DETERMINISTICALLY from that metadata rather than trusting the
        # model's own bib output: an LLM-authored reference list invents
        # entries (29 fabricated venues found across shipped papers), and
        # the sectionwise path leaves the two artifacts causally
        # independent (F-E2A-SECTIONWISE-BIB-DRIFT: 22 dangling keys).
        pool_papers = (lit or {}).get("papers") or []
        if pool_papers:
            bibtex = build_bibtex(pool_papers)
        paper_tex, bibtex, cite_stats = reconcile_citations(
            paper_tex, bibtex, pool_papers
        )
        self.ctx.log.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "agent": self.agent_name,
                "message": (
                    "Bib reconciliation: "
                    f"{cite_stats['cited']} keys cited, "
                    f"{cite_stats['bib_entries']} bib entries, "
                    f"{cite_stats['backfilled']} back-filled, "
                    f"{cite_stats['stripped']} invented keys stripped"
                    + (f" ({cite_stats['skipped']})" if cite_stats["skipped"] else "")
                ),
            }
        )

        # I2 (AERA_OPEN audit): the SPEC section 4.5 UNVERIFIED block is
        # injected DETERMINISTICALLY — it used to be prompt rule 6 only,
        # and the one run that needed it omitted it while also
        # fabricating numbers. Never again an LLM-obedience rule.
        paper_tex = self._inject_unverified_flag(paper_tex, review)

        # Write outputs
        paper_path = os.path.join(self.ctx.output_dir, "paper.tex")
        with open(paper_path, "w", encoding="utf-8") as f:
            f.write(paper_tex)

        bib_path = os.path.join(self.ctx.output_dir, "references.bib")
        with open(bib_path, "w", encoding="utf-8") as f:
            f.write(bibtex)

        return paper_tex

    # ------------------------------------------------------------------
    # I2: deterministic UNVERIFIED flag (SPEC section 4.5)
    # ------------------------------------------------------------------

    def _inject_unverified_flag(
        self, paper_tex: str, review_report: dict | None
    ) -> str:
        """Prepend the SPEC section 4.5 warning block and append the
        Critic report appendix when the run is flagged UNVERIFIED.

        Deterministic on every path (conference AND journal templates);
        idempotent (skips when the marker is already present). The block
        goes immediately before the first ``\\section`` so it opens the
        paper body; fallback is right after ``\\begin{document}``.
        """
        if not run_is_unverified(review_report):
            return paper_tex
        if UNVERIFIED_MARKER in paper_tex:
            return paper_tex

        first_section = re.search(r"\\section\*?\{", paper_tex)
        if first_section:
            i = first_section.start()
            paper_tex = paper_tex[:i] + UNVERIFIED_BLOCK + "\n" + paper_tex[i:]
        elif "\\begin{document}" in paper_tex:
            paper_tex = paper_tex.replace(
                "\\begin{document}",
                "\\begin{document}\n" + UNVERIFIED_BLOCK,
                1,
            )
        else:
            paper_tex = UNVERIFIED_BLOCK + paper_tex

        if "\\end{document}" in paper_tex:
            review_json = json.dumps(
                review_report or {}, indent=1, default=str
            ).replace("\\end{verbatim}", "\\end~{verbatim}")
            appendix = (
                "\\section*{Appendix: Automated Critic Review Report}\n"
                "{\\small\\begin{verbatim}\n"
                + review_json
                + "\n\\end{verbatim}}\n\n"
            )
            paper_tex = paper_tex.replace(
                "\\end{document}", appendix + "\\end{document}", 1
            )

        self.ctx.log.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "agent": self.agent_name,
                "message": (
                    "Injected UNVERIFIED warning block + Critic appendix "
                    "(deterministic, SPEC section 4.5 / I2 guard)"
                ),
            }
        )
        return paper_tex

    # ------------------------------------------------------------------
    # E2a: sectionwise journal generation
    # ------------------------------------------------------------------

    JOURNAL_SECTIONS: list[tuple[str, int]] = [
        ("Introduction", 1400),
        ("Related Work", 1750),
        ("Methods", 1750),
        ("Results", 1500),
        ("Discussion", 1200),
        ("Limitations and Future Directions", 500),
    ]

    def _run_journal_sectionwise(
        self, base_context: str, fallback_bibtex: str
    ) -> tuple[str, str]:
        """Generate a journal manuscript section by section (~8000 words).

        Each section is its own LLM call carrying the full run context
        plus brief summaries of the sections already written (for
        coherence without token blow-up). The pieces are assembled into
        a synthetic full-latex document compatible with
        :meth:`_reassemble_from_template`.
        """
        # 1) Front matter: title / abstract / keywords
        front_resp = self.call_llm(
            base_context
            + "\n\n## TASK (front matter only)\n"
            "Write ONLY the manuscript front matter as LaTeX: a"
            " \\title{...} line, a \\begin{abstract}...\\end{abstract}"
            " block (150-250 words), and a \\keywords{...} line (4-6"
            " keywords). Nothing else - no sections, no preamble.",
            max_tokens=4000,
        )
        front = self._extract_latex(front_resp)
        if not front or front == _MINIMAL_STUB_TEX:
            front = front_resp

        sections_tex: list[str] = []
        summaries: list[str] = []
        for name, words in self.JOURNAL_SECTIONS:
            prior = (
                "\n".join(f"- {s}" for s in summaries)
                if summaries else "(none yet)"
            )
            resp = self.call_llm(
                base_context
                + "\n\n## SECTIONS ALREADY WRITTEN (one-line summaries)\n"
                + prior
                + f"\n\n## TASK (one section only)\n"
                f"Write ONLY the \\section{{{name}}} of the journal "
                f"manuscript as LaTeX body text, targeting ~{words} words "
                "(subsections allowed). Use \\parencite/\\textcite for "
                "citations keyed to the literature context. Do NOT repeat "
                "content summarized above; do NOT write any other "
                "section; no preamble, no \\end{document}. Then, after "
                "the LaTeX block, output one line starting with "
                "'SUMMARY:' - a single sentence summarizing what this "
                "section covered (for the next section's context).",
                max_tokens=8000,
            )
            body = self._extract_latex(resp)
            if body == _MINIMAL_STUB_TEX:
                body = ""
            if not body or "\\section" not in body:
                # salvage: wrap raw text under the section heading
                body = f"\\section{{{name}}}\n" + (body or resp)
            sections_tex.append(body)
            m = re.search(r"SUMMARY:\s*(.+)", resp)
            summaries.append(
                f"{name}: {m.group(1).strip()[:200]}" if m else name
            )
            self.ctx.log.append({
                "timestamp": datetime.utcnow().isoformat(),
                "agent": self.agent_name,
                "message": (
                    f"Journal sectionwise: '{name}' written "
                    f"(~{len(body.split())} words)"
                ),
            })

        # 3) Bibliography from a dedicated call (or fallback builder)
        bib_resp = self.call_llm(
            base_context
            + "\n\n## TASK (bibliography only)\n"
            "Output ONLY a ```bibtex code block containing every entry "
            "cited in a manuscript about this study, keyed to the "
            "literature context paperIds.",
            max_tokens=6000,
        )
        bibtex = self._extract_bibtex(bib_resp) or fallback_bibtex

        synthetic = (
            front.strip()
            + "\n\\maketitle\n\n"
            + "\n\n".join(sections_tex)
            + "\n\n\\end{document}\n"
        )
        return synthetic, bibtex

    # ------------------------------------------------------------------
    # Template reassembly (v2 preamble protection)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_braced_arg(latex: str, command: str) -> str | None:
        """Extract the braced argument of a LaTeX command, handling nested braces.

        Example: ``_extract_braced_arg(text, r"\\title")`` on
        ``\\title{Predicting \\textbf{STEM} Achievement}`` returns
        ``Predicting \\textbf{STEM} Achievement``.
        """
        pattern = re.escape(command) + r"(?:\[[^\]]*\])?\s*\{"
        match = re.search(pattern, latex)
        if not match:
            return None
        start = match.end()
        depth = 1
        i = start
        while i < len(latex) and depth > 0:
            if latex[i] == "{":
                depth += 1
            elif latex[i] == "}":
                depth -= 1
            i += 1
        if depth != 0:
            return None
        return latex[start : i - 1]

    def _reassemble_from_template(self, llm_latex: str, template: str) -> str:
        """Extract content from the LLM's LaTeX and insert it into the clean template.

        The LLM frequently corrupts the ACM preamble (e.g. dropping a backslash from
        ``\\renewcommand\\@copyrightpermission``).  By extracting only the *content*
        sections and substituting them into the pristine template, the preamble is
        guaranteed to remain intact.
        """
        # --- Title ---
        title = self._extract_braced_arg(llm_latex, r"\title") or "Untitled"

        # --- Abstract ---
        # Accept BOTH abstract forms. The prompt asks for the
        # \begin{abstract} environment, but apa7 (the journal template)
        # documents \abstract{...}, and the model follows the class it
        # can see. Matching only the environment produced a manuscript
        # with a literally empty \abstract{} that LSAR rejected outright
        # ("No abstract found"), costing the entire review (F-P5-EMPTY-ABSTRACT).
        abstract_match = re.search(
            r"\\begin\{abstract\}(.*?)\\end\{abstract\}", llm_latex, re.DOTALL
        )
        if abstract_match:
            abstract = abstract_match.group(1).strip()
        else:
            abstract = _extract_braced_arg(llm_latex, r"\abstract")

        # --- Keywords ---
        keywords = self._extract_braced_arg(llm_latex, r"\keywords") or ""

        # --- Body (between \maketitle and the first structural boundary) ---
        body_match = re.search(
            r"\\maketitle\s*(.*?)"
            r"(?=\\begin\{acks\}|\\appendix\b|\\bibliographystyle"
            r"|\\printbibliography|\\end\{document\})",
            llm_latex,
            re.DOTALL,
        )
        body = body_match.group(1).strip() if body_match else ""

        # --- Appendix (optional) ---
        appendix = ""
        appendix_match = re.search(
            r"(\\appendix\b.*?)(?=\\end\{document\})", llm_latex, re.DOTALL
        )
        if appendix_match:
            appendix = appendix_match.group(1).strip()

        # --- Substitute into clean template ---
        result = template
        result = result.replace("%%PLACEHOLDER:TITLE%%", title)
        short = title if len(title) <= 50 else title[:47].rstrip() + "..."
        result = result.replace("%%PLACEHOLDER:SHORTTITLE%%", short)
        result = result.replace("%%PLACEHOLDER:ABSTRACT%%", abstract)
        result = result.replace("%%PLACEHOLDER:KEYWORDS%%", keywords)
        result = result.replace("%%PLACEHOLDER:PAPER_BODY%%", body)
        result = result.replace("%%PLACEHOLDER:APPENDIX%%", appendix)

        return result

    # ------------------------------------------------------------------
    # Template loading
    # ------------------------------------------------------------------

    def _load_template(self, version: str = "v1") -> str:
        """Load the LaTeX paper template.

        Args:
            version: ``"v1"`` for the original placeholder template,
                     ``"v2"`` for the outline-first single-body template.
        """
        if version == "journal":
            default_path = "templates/paper_template_journal.tex"
        elif version == "v2":
            default_path = "templates/paper_template_v2.tex"
        else:
            default_path = "templates/paper_template.tex"
        # Try config-specified path first (relative to cwd / project root)
        template_path = self.config.get("paths", {}).get(
            "paper_template" if version == "v1" else "paper_template_v2",
            default_path,
        )
        # If relative, resolve from project root (two levels up from this file)
        if not os.path.isabs(template_path):
            project_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            abs_path = os.path.join(project_root, template_path)
            if os.path.exists(abs_path):
                template_path = abs_path
        try:
            with open(template_path, encoding="utf-8") as f:
                return f.read()
        except OSError:
            # F-A4-V1-TEMPLATE-MISSING (Phase A attempt 3): the repo ships
            # only paper_template_v2.tex; the v1 fallback path handed the
            # LLM a minimal stub, which produced a paper with a corrupted
            # preamble and NO bibliography. Fall back to the v2 template
            # text (correct preamble + \bibliography lines for the LLM to
            # mirror) before resorting to the stub.
            if version == "v1":
                v2_path = "templates/paper_template_v2.tex"
                if not os.path.isabs(v2_path):
                    project_root = os.path.dirname(
                        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    )
                    v2_abs = os.path.join(project_root, v2_path)
                else:
                    v2_abs = v2_path
                try:
                    with open(v2_abs, encoding="utf-8") as f:
                        self.ctx.log.append(
                            {
                                "timestamp": datetime.utcnow().isoformat(),
                                "agent": self.agent_name,
                                "message": (
                                    f"v1 template missing at {template_path}; "
                                    "using v2 template text as reference."
                                ),
                            }
                        )
                        return f.read()
                except OSError:
                    pass
            self.ctx.log.append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "agent": self.agent_name,
                    "message": f"Could not load paper template from {template_path}; using minimal stub.",
                }
            )
            return _MINIMAL_STUB_TEX

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_template_structure(latex: str) -> list[str]:
        """Check that critical ACM acmart structural elements are intact.

        Returns a list of warning strings (non-fatal; logged but not raised).
        """
        warnings: list[str] = []

        if r"\documentclass[sigconf]{acmart}" not in latex:
            warnings.append("Missing or modified \\documentclass[sigconf]{acmart}")
        if r"\begin{document}" not in latex:
            warnings.append("Missing \\begin{document}")
        if r"\begin{abstract}" not in latex:
            warnings.append("Missing \\begin{abstract}")
        if r"\maketitle" not in latex:
            warnings.append("Missing \\maketitle")

        # Abstract must come before \maketitle (ACM acmart requirement)
        if r"\begin{abstract}" in latex and r"\maketitle" in latex:
            abstract_pos = latex.index(r"\begin{abstract}")
            maketitle_pos = latex.index(r"\maketitle")
            if abstract_pos > maketitle_pos:
                warnings.append(
                    r"\begin{abstract} appears after \maketitle — will cause compile error"
                )

        # \begin{abstract} must be inside \begin{document}
        if r"\begin{document}" in latex and r"\begin{abstract}" in latex:
            doc_pos = latex.index(r"\begin{document}")
            abstract_pos = latex.index(r"\begin{abstract}")
            if abstract_pos < doc_pos:
                warnings.append(
                    r"\begin{abstract} is before \begin{document} — will cause compile error"
                )

        if r"\begin{acks}" not in latex:
            warnings.append("Missing \\begin{acks} environment (required by acmart)")
        if r"\bibliographystyle{ACM-Reference-Format}" not in latex:
            warnings.append("Missing \\bibliographystyle{ACM-Reference-Format}")
        # The invariant is that EDM-ARS remains credited as an author, not
        # that any particular person is named. The human and AI author
        # entries are placeholders a user of this repository fills in, so
        # asserting on their literal text would flag every customised
        # template as tampered with.
        if "EDM-ARS" not in latex:
            warnings.append("Fixed author block appears to have been removed or modified")

        # AI-generated paper disclaimer and copyright suppression
        if r"\setcopyright{none}" not in latex:
            warnings.append("Missing \\setcopyright{none} — ACM copyright text will appear")
        if r"\settopmatter{printacmref=false}" not in latex:
            warnings.append(
                "Missing \\settopmatter{printacmref=false} — "
                "'ACM Reference Format:' block will appear"
            )
        if "AI-Generated Research Paper" not in latex:
            warnings.append(
                "AI-generated paper disclaimer was removed or modified"
            )
        if "Anonymous Conference" not in latex:
            warnings.append(
                "\\acmConference was modified — must remain 'Anonymous Conference'"
            )

        # Check for unfilled placeholders
        remaining = re.findall(r"%%PLACEHOLDER:\w+%%", latex)
        if remaining:
            warnings.append(f"Unfilled placeholders remain: {remaining}")

        # Check tabular column count consistency
        warnings.extend(_check_tabular_column_counts(latex))

        # Check for \declaration{...} antipattern (declaration must be inside braces)
        declaration_antipatterns = re.findall(
            r"\\(small|footnotesize|large|Large|LARGE|normalsize|itshape|bfseries|ttfamily)\{",
            latex,
        )
        if declaration_antipatterns:
            warnings.append(
                f"Font/size declaration(s) used as commands (must be inside braces, not before them): "
                f"{declaration_antipatterns}. "
                r"Use {\small text} not \small{text}."
            )

        # Check for loose table footnotes placed after \end{table} instead of inside threeparttable
        loose_note = re.search(
            r"\\end\{table\}.*?\\noindent\s*\{?\\(small|footnotesize)",
            latex,
            re.DOTALL,
        )
        if loose_note:
            warnings.append(
                r"Loose table footnote detected after \end{table}. "
                "Use threeparttable with \\begin{tablenotes} inside the float instead — "
                "loose notes separate from the table when it floats."
            )

        # Check that every \ref{fig:xxx} has a matching \label{fig:xxx}
        warnings.extend(_check_figure_ref_label_pairs(latex))

        # Check for wide tables (5+ columns) missing \resizebox
        warnings.extend(_check_wide_table_resizebox(latex))

        return warnings

    # ------------------------------------------------------------------
    # BibTeX generation
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_paper_ids(literature_context: dict | None) -> dict | None:
        """Replace colons in paper IDs with underscores for BibTeX key compatibility."""
        if not literature_context:
            return literature_context
        papers = literature_context.get("papers")
        if not papers:
            return literature_context
        sanitized = []
        for p in papers:
            pid = p.get("paperId", "")
            if ":" in pid:
                sanitized.append({**p, "paperId": pid.replace(":", "_")})
            else:
                sanitized.append(p)
        return {**literature_context, "papers": sanitized}

    def _citation_key_block(self, literature_context: dict | None) -> str:
        """Arc P3 prompt block: the only citation keys the model may use.

        Reads the venue target from the mined anchor norms so the writer
        is told how deep the reference list must be for THIS venue
        (EDM ~34, JEDM ~62, JLA ~65 by anchor median).
        """
        papers = (literature_context or {}).get("papers") or []
        if not papers:
            return ""
        venue = self.config.get("review_gate", {}).get("venue")
        try:
            target = venue_citation_target(venue)
        except Exception:
            target = None
        return format_citation_key_block(papers, target=target)

    def _build_bibtex(self, literature_context: dict | None) -> str:
        """
        Generate BibTeX entries from S2 literature_context.papers.

        Delegates to the deterministic builder in ``src.citations`` so the
        Writer, the reconciler and the review gate can never disagree
        about entry formatting. Falls back to a placeholder comment if
        papers is empty or None.
        """
        if not literature_context:
            return _S2_FAILURE_BIB_COMMENT
        return build_bibtex(literature_context.get("papers") or [])

    def _build_quality_repair_prompt(
        self, paper_tex: str, quality_report: LatexQualityReport
    ) -> str:
        """Build a repair prompt listing all quality issues found in the generated LaTeX."""
        issues_str = "\n".join(
            f"  - {w}" for w in quality_report.to_warning_strings()
        )
        return (
            "The LaTeX paper you generated has quality issues that must be fixed:\n\n"
            f"{issues_str}\n\n"
            "## Required Fixes\n"
            "1. Replace every `(not shown)` phrase with the actual result value from the pipeline data.\n"
            "2. Replace every `[Insert ...]` placeholder with real content.\n"
            "3. Remove all `TODO` and `FIXME` markers.\n"
            "4. Replace every `%%PLACEHOLDER:X%%` with the appropriate content.\n"
            "5. Replace `[Author, Year]` citation placeholders with `\\cite{paperId}` if papers "
            "are available, otherwise remove the placeholder.\n\n"
            "Output the COMPLETE corrected paper in a ```latex code block. "
            "If you also have updated BibTeX, output it in a ```bibtex code block. "
            "Do NOT introduce any new placeholder phrases.\n\n"
            "## Original Paper\n"
            f"```latex\n{paper_tex}\n```"
        )

    # ------------------------------------------------------------------
    # Message builders
    # ------------------------------------------------------------------

    def _build_user_message(
        self,
        research_spec: dict | None,
        literature_context: dict | None,
        data_report: dict | None,
        results_object: dict | None,
        review_report: dict | None,
        template_text: str = "",
    ) -> str:
        figures = (results_object or {}).get("figures_generated") or []
        parts = [
            "## research_spec.json",
            "```json",
            json.dumps(research_spec or {}, indent=2),
            "```",
            "",
            "## literature_context.json",
            "```json",
            json.dumps(literature_context or {}, indent=2),
            "```",
            "",
            # Arc P3: enumerate the legal citation keys explicitly. The
            # JSON above was the only signal before, and the model
            # invented keys that no bib entry could satisfy.
            self._citation_key_block(literature_context),
            "## data_report.json",
            "```json",
            json.dumps(data_report or {}, indent=2),
            "```",
            "",
            "## results.json",
            "```json",
            # I1a: nulls render as loud NOT-AVAILABLE markers so the
            # model never pads a missing value with an invented number.
            json.dumps(_mark_null_values(results_object or {}), indent=2),
            "```",
            "",
            "## review_report.json",
            "```json",
            json.dumps(review_report or {}, indent=2),
            "```",
            "",
            "## Available Figures",
            "\n".join(f"- {fig}" for fig in figures) if figures else "(none)",
            "",
        ]
        if template_text:
            parts += [
                "--- TEMPLATE START ---",
                template_text,
                "--- TEMPLATE END ---",
                "",
            ]
        parts += [
            "## Task",
            (
                "Fill in all %%PLACEHOLDER:SLOT_NAME%% markers in the template above "
                "with the appropriate content based on all the inputs provided. "
                "Output the COMPLETE filled-in template in a ```latex code block. "
                "Also output the references.bib content in a ```bibtex code block. "
                "Follow all requirements in the system prompt exactly. "
                "Do NOT modify the template structure — only replace the placeholder markers."
            ),
        ]
        return "\n".join(parts)

    def _build_user_message_with_outline(
        self,
        outline: dict,
        research_spec: dict | None,
        literature_context: dict | None,
        data_report: dict | None,
        results_object: dict | None,
        review_report: dict | None,
        template_text: str = "",
    ) -> str:
        """Build the user message for outline-first paper generation."""
        figures = (results_object or {}).get("figures_generated") or []
        parts = [
            "## research_spec.json",
            "```json",
            json.dumps(research_spec or {}, indent=2),
            "```",
            "",
            "## literature_context.json",
            "```json",
            json.dumps(literature_context or {}, indent=2),
            "```",
            "",
            # Arc P3: enumerate the legal citation keys explicitly. The
            # JSON above was the only signal before, and the model
            # invented keys that no bib entry could satisfy.
            self._citation_key_block(literature_context),
            "## data_report.json",
            "```json",
            json.dumps(data_report or {}, indent=2),
            "```",
            "",
            "## results.json",
            "```json",
            # I1a: nulls render as loud NOT-AVAILABLE markers so the
            # model never pads a missing value with an invented number.
            json.dumps(_mark_null_values(results_object or {}), indent=2),
            "```",
            "",
            "## review_report.json",
            "```json",
            json.dumps(review_report or {}, indent=2),
            "```",
            "",
            "## Paper Outline",
            "```json",
            json.dumps(outline, indent=2),
            "```",
            "",
            "## Available Figures",
            "\n".join(f"- {fig}" for fig in figures) if figures else "(none)",
            "",
        ]
        if template_text:
            parts += [
                "--- TEMPLATE START ---",
                template_text,
                "--- TEMPLATE END ---",
                "",
            ]
        parts += [
            "## Task",
            (
                "Generate the paper following the outline above. "
                "Fill %%PLACEHOLDER:TITLE%%, %%PLACEHOLDER:ABSTRACT%%, and "
                "%%PLACEHOLDER:KEYWORDS%% in the template. "
                "For %%PLACEHOLDER:PAPER_BODY%%, generate ALL sections and subsections "
                "following the outline structure — use the section titles, emphasis levels, "
                "and word targets from the outline. Use \\section{} and \\subsection{} commands. "
                "Output the COMPLETE filled-in template in a ```latex code block. "
                "Also output the references.bib content in a ```bibtex code block. "
                "Follow all requirements in the system prompt exactly. "
                "Do NOT add sections not in the outline. "
                "The narrative_hook should inform the opening of the Introduction. "
                "CRITICAL: Do NOT modify the document preamble (everything before "
                "\\begin{document}). Copy it EXACTLY as-is from the template, including "
                "the \\makeatletter / \\renewcommand\\@copyrightpermission block. "
                "Do NOT change the author block, \\shortauthors, or CCS concepts."
            ),
        ]
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_latex(text: str) -> str:
        """
        Extract the LaTeX document from the LLM response.

        Priority:
          1. ```latex ... ``` fenced block
          2. \\documentclass ... \\end{document} span
          3. Minimal stub fallback
        """
        match = re.search(r"```latex\s*\n(.*?)```", text, re.DOTALL)
        if match:
            candidate = match.group(1).strip()
            # Only accept if it is a complete document; otherwise fall through
            if r"\documentclass" in candidate:
                return candidate
        # Fall back to any \documentclass...\end{document} span
        match = re.search(
            r"(\\documentclass.*?\\end\{document\})", text, re.DOTALL
        )
        if match:
            return match.group(1).strip()
        return _MINIMAL_STUB_TEX

    @staticmethod
    def _extract_bibtex(text: str) -> str:
        """
        Extract the BibTeX block from the LLM response.

        Returns an empty string if no ```bibtex block is found
        (caller uses the pre-built fallback in that case).
        """
        match = re.search(r"```bibtex\s*\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip() + "\n"
        return ""
