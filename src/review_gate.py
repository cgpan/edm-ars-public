"""LSAR-powered quality gate for EDM-ARS papers.

Integrates the LSAR (Learning Science Auto-Reviewer) pipeline as a post-writing
quality gate.  After the Writer agent produces paper.tex and it is compiled to
PDF, this module:

1. Prepares a clean PDF (fixing placeholder citations so pdflatex succeeds
   without bibtex).
2. Runs the LSAR pipeline programmatically to obtain dimensional review scores.
3. Evaluates a pass/fail gate based on configurable thresholds.
4. On failure, uses an LLM to revise the paper prose and loops back (up to
   *max_cycles* iterations).
5. Saves all LSAR artefacts alongside the EDM-ARS run output.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import anthropic  # type: ignore[import-not-found]


# ---------------------------------------------------------------------------
# Dimension → EDM-ARS agent mapping (for suggested_focus_areas)
# ---------------------------------------------------------------------------

DIMENSION_AGENT_MAP: dict[str, str] = {
    "Relevance": "ProblemFormulator",
    "Novelty": "ProblemFormulator",
    "Theoretical/Conceptual Grounding": "Writer",
    "Methodological Rigor": "Analyst",
    "Empirical Support / Results": "Analyst",
    "Empirical Support/Results": "Analyst",
    "Significance & Impact": "Writer",
    "Ethics, Fairness & Equity": "Writer",
    "Clarity of Communication": "Writer",
}


# Arc P4: which prose sections a reviewer dimension actually lives in.
# Methodological Rigor and Empirical Support are deliberately absent —
# DIMENSION_AGENT_MAP routes those to the Analyst, and a prose reviser
# must not "improve" them by editing numbers. For those the reviser gets
# the Discussion/Limitations framing instead, where an honest
# acknowledgment is the legitimate fix.
DIMENSION_SECTION_MAP: dict[str, tuple[str, ...]] = {
    "Relevance": ("Introduction", "Discussion"),
    "Novelty": ("Introduction", "Related Work"),
    "Theoretical/Conceptual Grounding": ("Introduction", "Related Work"),
    "Significance & Impact": ("Discussion", "Conclusion"),
    "Ethics, Fairness & Equity": ("Discussion", "Limitations"),
    "Clarity of Communication": ("Abstract", "Introduction", "Discussion"),
    "Methodological Rigor": ("Limitations",),
    "Empirical Support / Results": ("Limitations",),
    "Empirical Support/Results": ("Limitations",),
}


# ---------------------------------------------------------------------------
# Section-scoped revision
# ---------------------------------------------------------------------------
# Measured on runs/arc_p_validation_20260711: paper.tex is 67,167 chars
# (~14.2k tokens) against a 16,000-token response budget. The
# whole-document path therefore has only ~11% headroom *while the model
# is simultaneously asked to ADD prose and citations* — too thin to rely
# on, and P4's improvement layer has never once landed a revision in a
# live run (one no-op, one rejected for destroying 4 of 5 tables).
#
# So: anything that does not fit comfortably is revised SECTION BY
# SECTION and spliced back deterministically. Every byte outside the
# revised sections is untouched by construction, which is what makes the
# float-invariance guard nearly free.
_CHARS_PER_TOKEN = 4.7  # measured on shipped LaTeX manuscripts
_REVISION_GROWTH = 1.15  # a revision adds prose, so the reply is longer
_WHOLE_DOC_BUDGET_FRACTION = 0.75  # "comfortably" = 3/4 of the reply budget
_SECTION_BUDGET_FRACTION = 0.5  # sections sent per request

_HEADING_RE = re.compile(r"\\(section|subsection)(\*?)\s*\{")
# A section block must never swallow the document trailer: splicing a
# reply that omits \bibliography would silently delete the references.
_TRAILER_RE = re.compile(
    r"\\(?:appendix\b|bibliographystyle\b|bibliography\b|printbibliography\b"
    r"|begin\{thebibliography\}|end\{document\})"
)
_FENCE_RE = re.compile(r"```([^\n]*)\n(.*?)```", re.DOTALL)
_SECTION_MARKER_RE = re.compile(r"SECTION[\s:#_-]*(\d+)", re.IGNORECASE)
_LEVEL_RANK: dict[str, int] = {"abstract": 0, "section": 1, "subsection": 2}


@dataclass(frozen=True)
class TexBlock:
    """A contiguous, spliceable region of a LaTeX manuscript."""

    title: str
    level: str  # "abstract" | "section" | "subsection"
    start: int
    end: int
    text: str

    def overlaps(self, other: "TexBlock") -> bool:
        return self.start < other.end and other.start < self.end


def _normalize_title(title: str) -> str:
    """Lowercase alphanumeric form used for fuzzy heading matching."""
    stripped = re.sub(r"\\[a-zA-Z]+\s*", " ", title)
    return " ".join(re.sub(r"[^a-z0-9]+", " ", stripped.lower()).split())


def _titles_match(wanted: str, actual: str) -> bool:
    """Fuzzy heading match.

    Real manuscripts do not use the canonical names: "Limitations" ships
    as "Limitations and Future Directions", "Conclusion" as
    "Conclusions". Containment either way is the practical rule.
    """
    a, b = _normalize_title(wanted), _normalize_title(actual)
    if not a or not b:
        return False
    return a == b or a in b or b in a


def _braced_arg(tex: str, open_idx: int) -> tuple[str, int]:
    """Return ``(content, index_after_closing_brace)`` for ``tex[open_idx] == '{'``."""
    depth = 0
    i = open_idx
    n = len(tex)
    while i < n:
        ch = tex[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return tex[open_idx + 1 : i], i + 1
        i += 1
    return tex[open_idx + 1 :], n


def _is_commented_out(tex: str, idx: int) -> bool:
    line_start = tex.rfind("\n", 0, idx) + 1
    return re.search(r"(?<!\\)%", tex[line_start:idx]) is not None


def _brace_delta(text: str) -> int:
    """Unescaped ``{`` minus unescaped ``}`` — 0 for well-formed LaTeX."""
    bare = re.sub(r"\\.", "", text, flags=re.DOTALL)
    return bare.count("{") - bare.count("}")


def _environment_deltas(text: str) -> dict[str, int]:
    """Per-environment ``\\begin`` minus ``\\end`` counts (non-zero => truncated)."""
    deltas: dict[str, int] = {}
    for kind, name in re.findall(r"\\(begin|end)\{([^}]*)\}", text):
        deltas[name] = deltas.get(name, 0) + (1 if kind == "begin" else -1)
    return {k: v for k, v in deltas.items() if v != 0}


def _block_kind(text: str) -> str:
    """Coarse identity of what a block starts with (``\\section``, ``abstract`` …)."""
    body = text.lstrip()
    m = re.match(r"\\begin\{([A-Za-z@*]+)\}", body)
    if m:
        return f"env:{m.group(1)}"
    m = re.match(r"\\([A-Za-z@]+)\*?", body)
    if m:
        return f"cmd:{m.group(1)}"
    return "text"


def _leading_title(text: str) -> Optional[str]:
    """Title of the heading a returned block starts with, if any."""
    body = text.lstrip()
    if body.startswith("\\begin{abstract}") or body.startswith("\\abstract"):
        return "Abstract"
    m = _HEADING_RE.match(body)
    if m:
        title, _ = _braced_arg(body, m.end() - 1)
        return title.strip()
    return None


class ReviewGate:
    """LSAR-powered quality gate for EDM-ARS papers."""

    def __init__(
        self,
        config: dict,
        output_dir: Path,
        log_fn: Any = None,
    ) -> None:
        self.config = config
        self.output_dir = Path(output_dir)
        self._log_fn = log_fn

        rg_cfg = config.get("review_gate", {})
        self.lsar_project_path = Path(rg_cfg.get("lsar_project_path", ""))
        self.lsar_config_path = Path(rg_cfg.get("lsar_config_path", ""))
        self.venue: str = rg_cfg.get("venue", "EDM")
        # V4 wave-2: the P25 threshold is anchored on EDM-conference
        # papers ONLY. Reviews for any other venue (the journal
        # profiles) run in ADVISORY mode: the score and full report are
        # produced (median sampling included), but the run is not
        # failed against an uncalibrated threshold. Journal anchor
        # calibration is backlog C4.
        self.calibrated_venues = set(
            rg_cfg.get("calibrated_venues", ["EDM"]))
        self.advisory_mode: bool = self.venue not in self.calibrated_venues
        self.max_cycles: int = rg_cfg.get("max_cycles", 2)
        self.pass_threshold: float = rg_cfg.get("pass_threshold", 5.5)
        self.dimension_floor: float = rg_cfg.get("dimension_floor", 3)
        # V3.7 Arc L (user decision 2026-07-03): calibrated P25 gate.
        # When rg_cfg.calibration_path points at an anchors_edm.yaml
        # produced by LSAR's calibration_analyze.py (reviewed by the
        # SAME provider/model this gate runs), overall_p25_full
        # OVERRIDES pass_threshold. Per-dimension P25 values are kept
        # as ADVISORY (reported, not blocking) — the existing
        # dimension_floor stays the hard per-dimension gate. Missing or
        # unreadable calibration file -> the absolute threshold stands.
        self.calibration_source: str = "absolute (config pass_threshold)"
        self.calibrated_dimension_p25: dict = {}
        calib_path = rg_cfg.get("calibration_path")
        if calib_path:
            try:
                import yaml as _yaml

                with open(calib_path, encoding="utf-8") as fh:
                    calib = _yaml.safe_load(fh) or {}
                # E2c: per-venue calibrated thresholds. When the
                # calibration file carries an anchor block for THIS
                # venue, its P25 becomes the gate and the venue is
                # CALIBRATED (advisory mode off).
                venue_block = (calib.get("venues") or {}).get(self.venue)
                if venue_block and venue_block.get("p25") is not None:
                    self.pass_threshold = float(venue_block["p25"])
                    self.calibration_source = (
                        f"venue-anchored P25 ({self.venue}, "
                        f"n={venue_block.get('n_anchors')})"
                    )
                    self.calibrated_venues.add(self.venue)
                    self.advisory_mode = False
                p25 = calib.get("overall_p25_full")
                if not self.advisory_mode and "venue-anchored" in self.calibration_source:
                    p25 = None  # venue anchor wins over the conference P25
                if isinstance(p25, (int, float)):
                    self.pass_threshold = float(p25)
                    self.calibration_source = (
                        f"calibrated P25(full) from {calib_path} "
                        f"(n_anchors={calib.get('n_anchors')})"
                    )
                    self.calibrated_dimension_p25 = (
                        calib.get("per_dimension_p25_full") or {}
                    )
            except Exception as exc:
                # NOTE: this used to call self._log_fn(msg) with ONE
                # argument, but the orchestrator passes its two-argument
                # _log(agent, message). A malformed calibration file
                # therefore raised TypeError inside __init__, which the
                # orchestrator swallowed as "REVIEWING failed
                # (non-fatal)" — silently skipping the entire gate.
                self._log(
                    f"calibration file unreadable ({exc}); "
                    "falling back to absolute threshold"
                )
        # Arc P4: most recent manuscript lint, fed to the reviser.
        self._last_lint: Any = None
        self.revision_model: str = rg_cfg.get("revision_model", "claude-sonnet-4-6")
        self.revision_max_tokens: int = rg_cfg.get("revision_max_tokens", 16000)

        # Build LLM client (same pattern as BaseAgent — respects llm_provider)
        provider = config.get("llm_provider", "anthropic")
        self._llm_provider: str = provider
        if provider == "minimax":
            api_key = os.environ.get("MINIMAX_API_KEY", "")
            base_url = config.get("minimax", {}).get(
                "base_url", "https://api.minimax.io/anthropic"
            )
            self._llm_client = anthropic.Anthropic(api_key=api_key, base_url=base_url)
            minimax_models = config.get("minimax", {}).get("models", {})
            self._llm_model = minimax_models.get(
                "revision_writer", minimax_models.get("writer", "MiniMax-M2.7")
            )
        elif provider in ("deepseek", "openai"):
            # OpenAI-compatible chat.completions path. Model resolution:
            # <provider>.models.revision_writer wins (per-agent tiering),
            # then review_gate.revision_model.
            import openai  # deferred: anthropic-only envs need not install it

            provider_block = config.get(provider, {}) or {}
            if provider == "deepseek":
                api_key = os.environ.get("DEEPSEEK_API_KEY", "")
                base_url = provider_block.get(
                    "base_url", "https://api.deepseek.com"
                )
            else:
                api_key = os.environ.get("OPENAI_API_KEY", "")
                base_url = provider_block.get("base_url")
            client_kwargs: dict = {"api_key": api_key}
            if base_url:
                client_kwargs["base_url"] = base_url
            self._llm_client = openai.OpenAI(**client_kwargs)
            provider_models = provider_block.get("models", {}) or {}
            self._llm_model = (
                provider_models.get("revision_writer") or self.revision_model
            )
        else:
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            self._llm_client = anthropic.Anthropic(api_key=api_key)
            self._llm_model = self.revision_model

    # ------------------------------------------------------------------
    # Internal logging helper
    # ------------------------------------------------------------------

    def _log(self, message: str) -> None:
        if self._log_fn is not None:
            self._log_fn("ReviewGate", message)

    # ------------------------------------------------------------------
    # 1. Prepare a clean PDF for LSAR review
    # ------------------------------------------------------------------

    def prepare_pdf(self, run_dir: Path, cycle: Optional[int] = None) -> Optional[Path]:
        """Fix placeholder citations in paper.tex and compile to a clean PDF.

        - Replaces ``\\cite{placeholder_*}`` with ``[XX]``
        - Replaces ``\\citet{placeholder_*}`` with ``Author et al. [XX]``
        - Strips ``\\bibliography{references}`` so bibtex is not needed
        - Writes ``paper_for_review.tex`` and compiles with pdflatex (twice)

        Returns the path to paper_for_review.pdf, or None on failure.
        """
        run_dir = Path(run_dir)
        # Reset FIRST — _last_lint feeds gate-blocking honesty checks,
        # and EVERY exit path from this method (including early returns)
        # must invalidate the previous cycle's report (review finding:
        # stale lint could both falsely block a fixed manuscript and
        # falsely pass a newly fabricated one).
        self._last_lint = None
        tex_path = run_dir / "paper.tex"
        if not tex_path.exists():
            self._log(f"paper.tex not found at {tex_path}")
            return None

        tex = tex_path.read_text(encoding="utf-8")

        # Replace \citet{placeholder_*} → Author et al. [XX]
        tex = re.sub(
            r"\\citet\{placeholder_\w+\}",
            r"Author et al.\\ [XX]",
            tex,
        )
        # Replace \cite{placeholder_*} → [XX]
        tex = re.sub(r"\\cite\{placeholder_\w+\}", "[XX]", tex)
        # Also handle any remaining placeholder_ keys in \cite{...}
        tex = re.sub(r"\\cite\{[^}]*placeholder[^}]*\}", "[XX]", tex)

        # Replace bibtex-based bibliography with inline thebibliography block
        # so references appear in the PDF without running bibtex
        inline_bib = self._build_inline_bibliography(run_dir)
        tex = re.sub(r"\\bibliographystyle\{[^}]*\}\s*", "", tex)
        tex = re.sub(
            r"\\bibliography\{[^}]*\}",
            lambda _: inline_bib,
            tex,
        )

        review_tex_path = run_dir / "paper_for_review.tex"
        review_tex_path.write_text(tex, encoding="utf-8")

        # Compile with pdflatex twice (no bibtex needed)
        success = self._compile_review_tex(run_dir, "paper_for_review.tex")
        pdf_path = run_dir / "paper_for_review.pdf"
        if success and pdf_path.exists():
            self._log(f"Review PDF compiled: {pdf_path}")
            self._lint_manuscript(run_dir, cycle=cycle)
            return pdf_path
        else:
            self._log("Review PDF compilation failed; falling back to paper.pdf")
            fallback = run_dir / "paper.pdf"
            if fallback.exists():
                # The lint targets paper.tex, which exists regardless of
                # the review-tex compile — lint the fallback too so the
                # honesty checks always see the manuscript under review.
                self._lint_manuscript(run_dir, cycle=cycle)
                return fallback
            return None

    #: Lint error codes that BLOCK a gate pass regardless of the
    #: reviewer's score (I1/I2, AERA_OPEN audit): fabricated numbers and
    #: a missing UNVERIFIED flag are invisible to the reviewer by
    #: construction, so the reviewer's Accept cannot clear them.
    HONESTY_BLOCKING_CODES = frozenset(
        {
            "unreconciled-table-numerals",
            "unreconciled-ci-interval",
            "unverified-block-missing",
        }
    )

    def _honesty_blockers(self) -> list[str]:
        """Error-severity honesty lint defects from the current cycle's
        lint report; empty when lint did not run (never blocks blind)."""
        report = self._last_lint
        if report is None:
            return []
        try:
            return [
                f"{d.code}: {d.message[:120]}"
                for d in report.errors
                if d.code in self.HONESTY_BLOCKING_CODES
            ]
        except Exception:  # noqa: BLE001 — lint shape must never crash the gate
            return []

    def _lint_manuscript(self, run_dir: Path, cycle: Optional[int] = None) -> Any:
        """Arc P1/P4: deterministic post-compile lint.

        Writes manuscript_lint.json to the run dir, keeps a per-cycle
        copy under lsar_review/cycle_N/, stashes the report on
        ``self._last_lint`` for the Arc P4 revision prompt, and logs a
        summary. Mostly advisory for the verdict — the defects steer
        revision rather than fail the gate — EXCEPT the honesty codes in
        :data:`HONESTY_BLOCKING_CODES`, which override a pass in
        ``run_gate`` (I1/I2). Never raises: a lint crash must not take
        down the gate.
        """
        self._last_lint = None
        try:
            from src.manuscript_linter import lint_manuscript
        except ImportError:  # flat execution layout
            try:
                from manuscript_linter import lint_manuscript  # type: ignore
            except ImportError:
                self._log("[Lint] manuscript_linter unavailable; skipped")
                return None
        try:
            report = lint_manuscript(run_dir, venue=self.venue)
            self._last_lint = report
            if cycle is not None:
                try:
                    cdir = run_dir / "lsar_review" / f"cycle_{cycle}"
                    cdir.mkdir(parents=True, exist_ok=True)
                    (cdir / "manuscript_lint.json").write_text(
                        json.dumps(report.to_dict(), indent=2), encoding="utf-8"
                    )
                except OSError:
                    pass
            n_err = len(report.errors)
            n_warn = len(report.defects) - n_err
            cites = report.metrics.get("n_citations_distinct", "?")
            self._log(
                f"[Lint] format_clean={report.format_clean} "
                f"errors={n_err} warns={n_warn} citations={cites} "
                f"(report: manuscript_lint.json)"
            )
            for d in report.defects[:6]:
                self._log(f"[Lint] {d.severity.upper()} {d.code}: {d.message}")
            return report
        except Exception as exc:
            self._log(f"[Lint] failed (non-fatal): {exc}")
            return None

    def _build_inline_bibliography(self, run_dir: Path) -> str:
        """Convert references.bib to a \\thebibliography block for standalone compilation."""
        bib_path = run_dir / "references.bib"
        if not bib_path.exists():
            return "\\begin{thebibliography}{0}\n\\bibitem{placeholder} No references available.\n\\end{thebibliography}"

        bib_text = bib_path.read_text(encoding="utf-8")

        # Parse bibtex entries: extract key, author, title, year
        entries: list[str] = []
        for match in re.finditer(
            r"@\w+\{([^,]+),\s*(.*?)\n\}",
            bib_text,
            re.DOTALL,
        ):
            key = match.group(1).strip()
            body = match.group(2)

            def _field(name: str) -> str:
                m = re.search(
                    rf"{name}\s*=\s*\{{(.*?)\}}",
                    body,
                    re.DOTALL,
                )
                return m.group(1).strip() if m else ""

            author = _field("author")
            title = _field("title")
            year = _field("year")
            venue = _field("booktitle") or _field("journal") or _field("note")

            parts = [p for p in [author, title, venue, year] if p]
            entry_text = ". ".join(parts) + "." if parts else key
            entries.append(f"\\bibitem{{{key}}} {entry_text}")

        if not entries:
            return "\\begin{thebibliography}{0}\n\\bibitem{placeholder} No references available.\n\\end{thebibliography}"

        items = "\n".join(entries)
        return f"\\begin{{thebibliography}}{{{len(entries)}}}\n{items}\n\\end{{thebibliography}}"

    def _compile_review_tex(
        self,
        cwd: Path,
        tex_file: str,
        timeout_s: int = 120,
    ) -> bool:
        """Compile *tex_file*: pdflatex x2, with a biber pass for
        biblatex (journal apa7) documents so references render in the
        review copy (F-W2-GATE-BIBER). Returns True on success."""
        cmd = ["pdflatex", "-interaction=nonstopmode", tex_file]
        try:
            _src = (cwd / tex_file).read_text(encoding="utf-8")
        except OSError:
            _src = ""
        if "biblatex" in _src or "\\addbibresource" in _src:
            base = tex_file.replace(".tex", "")
            for c in ([*cmd], ["biber", base], [*cmd], [*cmd]):
                try:
                    proc = subprocess.run(c, cwd=str(cwd),
                                          capture_output=True, text=True,
                                          timeout=timeout_s)
                except Exception:
                    return False
            return (cwd / (base + ".pdf")).exists()
        for pass_num in range(2):
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(cwd),
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                )
                if proc.returncode not in (0, 1):
                    self._log(
                        f"pdflatex pass {pass_num + 1} failed (rc={proc.returncode}): "
                        f"{proc.stderr[-500:]}"
                    )
                    return False
            except FileNotFoundError:
                self._log("pdflatex not found on PATH")
                return False
            except subprocess.TimeoutExpired:
                self._log(f"pdflatex timed out after {timeout_s}s")
                return False
        return True

    # ------------------------------------------------------------------
    # 2. Run LSAR pipeline
    # ------------------------------------------------------------------

    def run_lsar(self, pdf_path: Path, cycle: int) -> Optional[dict]:
        """Run LSAR pipeline on *pdf_path* and return report_json.

        LSAR is imported by temporarily adding its project root to sys.path.
        Outputs are saved to ``{output_dir}/lsar_review/cycle_{cycle}/``.
        Returns ``None`` on any failure (LSAR crash, import error, etc.).
        """
        lsar_root = str(self.lsar_project_path)
        if not os.path.isdir(lsar_root):
            self._log(f"LSAR project path does not exist: {lsar_root}")
            return None

        cycle_dir = self.output_dir / "lsar_review" / f"cycle_{cycle}"
        cycle_dir.mkdir(parents=True, exist_ok=True)

        # Temporarily add LSAR to sys.path
        added_to_path = False
        if lsar_root not in sys.path:
            sys.path.insert(0, lsar_root)
            added_to_path = True

        try:
            from lsar.pipeline import LSARPipeline  # type: ignore[import-not-found]

            config_path: Optional[Path] = None
            if self.lsar_config_path and self.lsar_config_path.exists():
                config_path = self.lsar_config_path

            pipeline = LSARPipeline(config_path=config_path)
            self._log(f"Running LSAR review (cycle {cycle}, venue={self.venue})")

            report_md, report_json = pipeline.run(
                pdf_path=Path(pdf_path),
                venue=self.venue,
                force=True,
                output_dir=cycle_dir,
            )

            # Persist LSAR outputs alongside EDM-ARS artefacts
            (cycle_dir / "lsar_report.md").write_text(
                report_md, encoding="utf-8"
            )
            (cycle_dir / "lsar_report.json").write_text(
                json.dumps(report_json, indent=2, default=str), encoding="utf-8"
            )
            self._log(
                f"LSAR review complete (cycle {cycle}): "
                f"overall_score={report_json.get('scores', {}).get('overall_score', '?')}"
            )
            return report_json

        except Exception as exc:
            self._log(f"LSAR pipeline failed (cycle {cycle}): {exc}")
            return None
        finally:
            if added_to_path and lsar_root in sys.path:
                sys.path.remove(lsar_root)

    # ------------------------------------------------------------------
    # 3. Evaluate pass/fail gate
    # ------------------------------------------------------------------

    @staticmethod
    def _review_health(report_json: dict) -> Optional[str]:
        """Return a reason string when an LSAR report's review is
        unusable for scoring, else None (J1).

        Only reports that actually carry a review block are judged — a
        report that simply does not persist its sections must read as
        UNKNOWN, not as broken. (An audit that treated a missing file as
        an empty one flagged every historical run in this repo.)
        """
        review = report_json.get("review")
        if not isinstance(review, dict):
            return None  # nothing to judge; do not invent a failure
        has_keys = "strengths" in review or "weaknesses" in review
        if not has_keys:
            return None
        missing = [
            key for key in ("strengths", "weaknesses")
            if not (review.get(key) or [])
        ]
        if not missing:
            return None
        return "review has no " + " and no ".join(missing)

    def evaluate_gate(self, report_json: dict) -> tuple[bool, dict]:
        """Apply pass/fail logic to LSAR scores.

        Returns ``(passed, diagnosis)`` where *diagnosis* contains the
        overall score, per-dimension scores, failing dimensions, and
        suggested focus areas mapped to EDM-ARS agents.
        """
        scores_block = report_json.get("scores", {})
        overall_score: float = scores_block.get("overall_score", 0.0)
        recommendation: str = scores_block.get("recommendation", "Unknown")
        dimensions: list[dict] = scores_block.get("dimensions", [])

        # J1 (2026-08-07): LSAR used to score reviews whose generation
        # was cut off at the Strengths/Weaknesses boundary. Its scorer
        # reads only those two lists, so the paper got graded on its
        # praise alone — +0.83 on average, and cycle_102 of the routed
        # AERA_OPEN run passed this gate at 7.5 on a review with ZERO of
        # both. LSAR now refuses to produce such a review; this is the
        # consumer-side check, because a gate should verify what it
        # received rather than trust the producer.
        review_health = self._review_health(report_json)
        if review_health:
            self._log(
                f"UNUSABLE review sample ({review_health}): a review "
                "missing its strengths or weaknesses is scored on half "
                "the evidence and reads high. Treating this sample as a "
                "failure rather than gating on it."
            )

        dim_scores: dict[str, int] = {}
        failing_dims: list[str] = []
        for dim in dimensions:
            name = dim.get("name", "")
            score = dim.get("score", 0)
            dim_scores[name] = score
            if score < self.dimension_floor:
                failing_dims.append(name)

        passed = overall_score >= self.pass_threshold and len(failing_dims) == 0
        if self.advisory_mode:
            # Uncalibrated venue: report the would-be verdict, never block.
            self._log(
                f"ADVISORY (uncalibrated venue {self.venue}): score "
                f"{overall_score:.2f}; threshold {self.pass_threshold} not "
                f"enforced (would have {'passed' if passed else 'failed'})."
            )
            passed = True
        if review_health:
            # Applied AFTER the advisory branch on purpose. Advisory
            # mode means "this venue's threshold is not trustworthy";
            # an unusable review means "this SCORE is not trustworthy",
            # which no venue calibration can rescue.
            passed = False

        # Calibrated per-dimension P25 comparison — ADVISORY only
        # (reported in the gate summary; not blocking).
        below_calibrated_p25: list[str] = []
        for name, p25 in (self.calibrated_dimension_p25 or {}).items():
            if isinstance(p25, (int, float)) and name in dim_scores:
                if dim_scores[name] < p25:
                    below_calibrated_p25.append(f"{name} ({dim_scores[name]} < P25 {p25})")

        # Top 3 weakest dimensions → mapped to EDM-ARS agents
        sorted_dims = sorted(dimensions, key=lambda d: d.get("score", 10))
        focus_areas: list[dict[str, str]] = []
        for dim in sorted_dims[:3]:
            name = dim.get("name", "")
            focus_areas.append(
                {
                    "dimension": name,
                    "score": str(dim.get("score", "?")),
                    "target_agent": DIMENSION_AGENT_MAP.get(name, "Writer"),
                }
            )

        diagnosis: dict[str, Any] = {
            "overall_score": overall_score,
            "recommendation": recommendation,
            "dimension_scores": dim_scores,
            "failing_dimensions": failing_dims,
            "suggested_focus_areas": focus_areas,
            "passed": passed,
            "threshold_used": self.pass_threshold,
            "advisory_uncalibrated_venue": self.advisory_mode,
            "threshold_source": self.calibration_source,
            "below_calibrated_p25_advisory": below_calibrated_p25,
            "review_health_problem": review_health,
        }
        return passed, diagnosis

    # ------------------------------------------------------------------
    # 4. Revise paper from LSAR review feedback
    # ------------------------------------------------------------------

    def revise_from_review(
        self,
        paper_tex: str,
        report_json: dict,
        diagnosis: dict,
        lint_report: Any = None,
    ) -> str:
        """Use an LLM to revise paper.tex based on LSAR feedback.

        The LLM is instructed to only revise prose (introduction framing,
        related work positioning, discussion depth, limitation
        acknowledgment).  Data, results, and tables are never changed.

        Two paths:

        * **section-scoped** (default for real manuscripts) — only the
          sections that carry the weakest dimension are sent and
          required back; they are spliced in deterministically so every
          other byte is untouched by construction.
        * **whole-document** — kept for short manuscripts whose complete
          revision fits comfortably inside the reply budget.

        Returns the revised LaTeX string (the original on any failure).
        """
        review_block = report_json.get("review", {})
        strengths = review_block.get("strengths", [])
        weaknesses = review_block.get("weaknesses", [])
        suggestions = review_block.get("suggestions", [])
        questions = review_block.get("questions_for_authors", [])
        lint = lint_report if lint_report is not None else self._last_lint

        focus_dims = [fa["dimension"] for fa in diagnosis.get("suggested_focus_areas", [])]

        if not self._fits_whole_document(paper_tex):
            blocks = self._select_target_blocks(
                paper_tex, self._target_section_names(focus_dims)
            )
            blocks = self._fit_blocks_to_budget(blocks)
            if blocks:
                return self._revise_sections(
                    paper_tex=paper_tex,
                    blocks=blocks,
                    strengths=strengths,
                    weaknesses=weaknesses,
                    suggestions=suggestions,
                    questions=questions,
                    focus_dims=focus_dims,
                    diagnosis=diagnosis,
                    lint_report=lint,
                )
            self._log(
                "No target sections could be located in the manuscript; "
                f"falling back to a whole-document revision "
                f"({self._estimated_reply_tokens(paper_tex):.0f} est. reply "
                f"tokens vs a {self.revision_max_tokens}-token budget)"
            )

        prompt = self._build_revision_prompt(
            paper_tex=paper_tex,
            strengths=strengths,
            weaknesses=weaknesses,
            suggestions=suggestions,
            questions=questions,
            focus_dims=focus_dims,
            diagnosis=diagnosis,
            lint_report=lint,
        )

        self._log("Calling LLM for whole-document paper revision (LSAR feedback)")
        response_text = self._call_revision_llm(prompt)
        if response_text is None:
            return paper_tex  # Return original on failure

        revised_tex = self._extract_latex(response_text)
        if revised_tex:
            self._log("Paper revision complete")
            return revised_tex
        else:
            self._log("Could not extract LaTeX from LLM response; keeping original")
            return paper_tex

    # -- LLM plumbing shared by both revision paths ---------------------

    _REVISION_SYSTEM_TEXT = (
        "You are a skilled academic writer for educational data mining. "
        "You revise LaTeX papers to address reviewer feedback while "
        "preserving all data, results, tables, and figures exactly as-is."
    )

    def _call_revision_llm(self, prompt: str) -> Optional[str]:
        """Send *prompt* to the configured provider. ``None`` on failure."""
        try:
            if self._llm_provider in ("deepseek", "openai"):
                response = self._llm_client.chat.completions.create(
                    model=self._llm_model,
                    max_tokens=self.revision_max_tokens,
                    temperature=0.3,
                    messages=[
                        {"role": "system", "content": self._REVISION_SYSTEM_TEXT},
                        {"role": "user", "content": prompt},
                    ],
                )
                return response.choices[0].message.content or ""
            with self._llm_client.messages.stream(
                model=self._llm_model,
                max_tokens=self.revision_max_tokens,
                temperature=0.3,
                system=self._REVISION_SYSTEM_TEXT,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                return stream.get_final_text()
        except Exception as exc:
            self._log(f"LLM revision call failed: {exc}")
            return None

    # -- budget arithmetic ---------------------------------------------

    def _estimated_reply_tokens(self, tex: str) -> float:
        """Tokens a *revision* of *tex* is expected to cost to write back."""
        return (len(tex) / _CHARS_PER_TOKEN) * _REVISION_GROWTH

    def _fits_whole_document(self, tex: str) -> bool:
        """True when returning the COMPLETE revised document fits comfortably.

        "Comfortably" is 75% of the reply budget: the measured shipped
        manuscript needs ~103% of it, which is why the whole-document
        path never landed a revision.
        """
        budget = _WHOLE_DOC_BUDGET_FRACTION * float(self.revision_max_tokens)
        return self._estimated_reply_tokens(tex) <= budget

    # ------------------------------------------------------------------
    # 4a. Section-scoped revision
    # ------------------------------------------------------------------

    @staticmethod
    def _target_section_names(focus_dims: list[str]) -> list[str]:
        """Sections that carry the weakest dimensions, in priority order."""
        names: list[str] = []
        for dim in focus_dims:
            for sec in DIMENSION_SECTION_MAP.get(dim, ()):
                if sec not in names:
                    names.append(sec)
        return names or ["Introduction", "Discussion"]

    def _split_sections(self, tex: str) -> list[TexBlock]:
        """Split *tex* into abstract / section / subsection blocks.

        Blocks stop at the document trailer (``\\appendix``,
        ``\\bibliography``, ``\\end{document}`` …) so a spliced reply can
        never delete the bibliography, and commented-out headings are
        ignored.
        """
        blocks: list[TexBlock] = []

        abstract = re.search(
            r"\\begin\{abstract\}.*?\\end\{abstract\}", tex, re.DOTALL
        )
        if abstract:
            blocks.append(
                TexBlock("Abstract", "abstract", abstract.start(), abstract.end(),
                         abstract.group(0))
            )
        else:
            macro = re.search(r"\\abstract\s*\{", tex)
            if macro and not _is_commented_out(tex, macro.start()):
                _, end = _braced_arg(tex, macro.end() - 1)
                blocks.append(
                    TexBlock("Abstract", "abstract", macro.start(), end,
                             tex[macro.start():end])
                )

        heads: list[tuple[int, str, str]] = []
        for m in _HEADING_RE.finditer(tex):
            if _is_commented_out(tex, m.start()):
                continue
            title, _ = _braced_arg(tex, m.end() - 1)
            heads.append((m.start(), m.group(1), title.strip()))

        # Every trailer position, not just the first: the UNVERIFIED path
        # appends an appendix section AFTER \bibliography, and that block
        # must still stop before \end{document}.
        trailers = [m.start() for m in _TRAILER_RE.finditer(tex)]

        for i, (start, level, title) in enumerate(heads):
            end = len(tex)
            for nxt_start, nxt_level, _ in heads[i + 1:]:
                if _LEVEL_RANK[nxt_level] <= _LEVEL_RANK[level]:
                    end = nxt_start
                    break
            nxt_trailer = next((t for t in trailers if t > start), None)
            if nxt_trailer is not None:
                end = min(end, nxt_trailer)
            if end > start:
                blocks.append(TexBlock(title, level, start, end, tex[start:end]))

        blocks.sort(key=lambda b: b.start)
        return blocks

    def _select_target_blocks(self, tex: str, names: list[str]) -> list[TexBlock]:
        """Resolve section *names* to concrete blocks, in priority order.

        Sections win over subsections; among duplicate headings (the
        sectionwise writer emits ``\\section{Introduction}`` twice) the
        block with the most content wins; overlapping picks are dropped
        so a parent section and its own subsection are never both sent.
        """
        blocks = self._split_sections(tex)
        chosen: list[TexBlock] = []
        for name in names:
            cands = [
                b for b in blocks
                if b.level in ("section", "abstract") and _titles_match(name, b.title)
            ]
            if not cands:
                cands = [b for b in blocks if _titles_match(name, b.title)]
            if not cands:
                continue
            best = max(cands, key=lambda b: b.end - b.start)
            if any(best.overlaps(c) for c in chosen):
                continue
            chosen.append(best)
        return chosen

    def _fit_blocks_to_budget(self, blocks: list[TexBlock]) -> list[TexBlock]:
        """Trim the (priority-ordered) selection to the reply budget, then
        return it in document order."""
        budget = _SECTION_BUDGET_FRACTION * float(self.revision_max_tokens)
        kept: list[TexBlock] = []
        used = 0.0
        for blk in blocks:
            cost = self._estimated_reply_tokens(blk.text)
            if kept and used + cost > budget:
                self._log(
                    f"Section '{blk.title}' dropped from this revision "
                    "request (reply budget)"
                )
                continue
            kept.append(blk)
            used += cost
        kept.sort(key=lambda b: b.start)
        return kept

    def _revise_sections(
        self,
        paper_tex: str,
        blocks: list[TexBlock],
        strengths: list[str],
        weaknesses: list[str],
        suggestions: list[str],
        questions: list[str],
        focus_dims: list[str],
        diagnosis: dict,
        lint_report: Any = None,
    ) -> str:
        """Request only *blocks* back, guard each, and splice them in."""
        prompt = self._build_section_revision_prompt(
            paper_tex=paper_tex,
            blocks=blocks,
            strengths=strengths,
            weaknesses=weaknesses,
            suggestions=suggestions,
            questions=questions,
            focus_dims=focus_dims,
            diagnosis=diagnosis,
            lint_report=lint_report,
        )
        sent = sum(len(b.text) for b in blocks)
        self._log(
            "Calling LLM for section-scoped revision: "
            f"{len(blocks)} section(s) [{', '.join(b.title for b in blocks)}], "
            f"{sent} of {len(paper_tex)} chars sent"
        )
        response_text = self._call_revision_llm(prompt)
        if response_text is None:
            return paper_tex

        returned = self._parse_section_response(response_text, blocks)
        if not returned:
            self._log(
                "Could not extract any revised section from the LLM "
                "response; keeping original"
            )
            return paper_tex

        accepted: dict[int, str] = {}
        for idx, body in returned.items():
            original = blocks[idx].text
            safe, reason = self._section_revision_is_safe(original, body)
            if not safe:
                self._log(
                    f"Section '{blocks[idx].title}' REJECTED and discarded: "
                    f"{reason}. Keeping the original section."
                )
                continue
            if body.strip() == original.strip():
                continue
            accepted[idx] = body

        missing = [b.title for i, b in enumerate(blocks) if i not in returned]
        if missing:
            self._log(f"Sections not returned by the model (left as-is): {missing}")
        if not accepted:
            self._log(
                "No revised section survived the safety guards; keeping the "
                "original manuscript"
            )
            return paper_tex

        spliced = self._splice_sections(paper_tex, blocks, accepted)
        self._log(
            f"Section-scoped revision spliced {len(accepted)} of "
            f"{len(blocks)} section(s): "
            f"{[blocks[i].title for i in sorted(accepted)]}"
        )
        return spliced

    def _splice_sections(
        self, tex: str, blocks: list[TexBlock], accepted: dict[int, str]
    ) -> str:
        """Deterministically substitute *accepted* bodies into *tex*.

        Replacement happens back-to-front so earlier offsets stay valid;
        every byte outside the replaced ranges is preserved exactly.
        """
        out = tex
        for idx in sorted(accepted, key=lambda i: blocks[i].start, reverse=True):
            blk = blocks[idx]
            trailing = blk.text[len(blk.text.rstrip()):]
            out = out[: blk.start] + accepted[idx].strip() + trailing + out[blk.end:]
        return out

    def _parse_section_response(
        self, text: str, blocks: list[TexBlock]
    ) -> dict[int, str]:
        """Map returned section bodies onto target block indices.

        Heading text decides (the prompt requires it verbatim). A block
        whose heading names something else is DROPPED — never rescued by
        its ``SECTION n`` marker or by position, because doing so
        splices one section's prose over another's. Markers and the
        1-to-1 positional fallback only apply to a block with no
        recognizable heading at all, and that block still has to survive
        :meth:`_section_revision_is_safe`.
        """
        candidates: list[tuple[Optional[int], str]] = []
        for m in _FENCE_RE.finditer(text):
            info, body = m.group(1), m.group(2)
            prefix = text[max(0, m.start() - 200): m.start()]
            marks = _SECTION_MARKER_RE.findall(prefix + " " + info)
            marker = int(marks[-1]) - 1 if marks else None
            candidates.append((marker, body))
        if not candidates:
            for blk in self._split_sections(text):
                candidates.append((None, blk.text))

        out: dict[int, str] = {}
        leftovers: list[str] = []
        for marker, body in candidates:
            idx = self._match_block_by_heading(body, blocks, taken=set(out))
            if idx is None:
                title = _leading_title(body)
                if title is not None:
                    self._log(
                        f"Ignoring returned block '{title}': it does not match "
                        "any requested section (or that section was already "
                        "returned)"
                    )
                    continue
                if marker is not None and 0 <= marker < len(blocks) and marker not in out:
                    idx = marker
            if idx is None:
                leftovers.append(body)
                continue
            out[idx] = body

        remaining = [i for i in range(len(blocks)) if i not in out]
        if len(remaining) == 1 and len(leftovers) == 1:
            out[remaining[0]] = leftovers[0]
        return out

    @staticmethod
    def _match_block_by_heading(
        body: str, blocks: list[TexBlock], taken: set
    ) -> Optional[int]:
        title = _leading_title(body)
        if title is None:
            return None
        for i, blk in enumerate(blocks):
            if i in taken:
                continue
            if _titles_match(title, blk.title):
                return i
        return None

    def _section_revision_is_safe(
        self, original: str, revised: str
    ) -> tuple[bool, str]:
        """Per-section version of :meth:`_revision_is_safe`.

        A section has no ``\\end{document}`` to check, so truncation is
        detected structurally: unbalanced environments, unbalanced
        braces, or a large loss of length.
        """
        if not revised or not revised.strip():
            return False, "returned section was empty"
        if _block_kind(revised) != _block_kind(original):
            return False, (
                f"returned block starts with {_block_kind(revised)} but the "
                f"section starts with {_block_kind(original)} "
                "(wrong block or missing heading)"
            )
        n_before = len(_HEADING_RE.findall(original))
        n_after = len(_HEADING_RE.findall(revised))
        if n_after > n_before:
            return False, (
                f"returned block adds sectioning commands ({n_before} -> {n_after}); "
                "sections must not be split or merged"
            )
        env_deltas = _environment_deltas(revised)
        if env_deltas != _environment_deltas(original):
            return False, (
                f"revision is truncated (unbalanced environments: {env_deltas})"
            )
        if _brace_delta(revised) != _brace_delta(original):
            return False, "revision is truncated (unbalanced braces)"
        if len(revised.strip()) < 0.6 * len(original.strip()):
            return False, (
                f"revision lost {100 * (1 - len(revised.strip()) / max(1, len(original.strip()))):.0f}% "
                "of the section (likely truncated)"
            )
        before = self._float_environments(original)
        after = self._float_environments(revised)
        if before != after:
            return False, (
                f"revision altered tables/figures/graphics "
                f"({len(before)} -> {len(after)} float artifacts changed)"
            )
        return True, ""

    @staticmethod
    def _lint_evidence_text(lint_report: Any) -> tuple[str, str]:
        """``(defects_text, citation_text)`` for either revision prompt.

        Deterministic, machine-verified defects — not opinions; the
        linter re-checks them after the revision.
        """
        defects_text = ""
        citation_text = ""
        if lint_report is None:
            return defects_text, citation_text
        try:
            errors = [d for d in lint_report.defects if d.severity == "error"]
            warns = [d for d in lint_report.defects if d.severity != "error"]
            lines = [f"- [{d.code}] {d.message}" for d in errors + warns]
            if lines:
                defects_text = (
                    "\n## Deterministic format defects (MUST fix)\n\n"
                    "A static checker found these in the compiled PDF. They "
                    "are facts, not reviewer opinion, and will be re-checked "
                    "after your revision.\n\n" + "\n".join(lines) + "\n"
                )
            m = lint_report.metrics or {}
            n_cited = m.get("n_citations_distinct")
            p25 = m.get("venue_refs_p25")
            if n_cited is not None and p25:
                citation_text = (
                    "\n## Citation target\n\n"
                    f"This manuscript cites {n_cited} distinct works; "
                    f"published papers at this venue cite at least {p25:g} "
                    "(25th percentile of the anchor corpus). Strengthen the "
                    "Related Work and Discussion by engaging MORE of the "
                    "references already present in references.bib.\n\n"
                    "**Cite only keys that already exist in references.bib. "
                    "Never invent a citation key** — invented keys are "
                    "deleted automatically, removing the support from your "
                    "sentence.\n"
                )
        except Exception:  # noqa: BLE001 — never let lint shape-drift kill a revision
            return "", ""
        return defects_text, citation_text

    def _manuscript_digest(self, paper_tex: str, blocks: list[TexBlock]) -> str:
        """Context a section reviser needs: title, abstract, and the outline.

        Without this the model cannot position the sections it is given
        against the rest of the paper.
        """
        parts: list[str] = []
        title_m = re.search(r"\\title\s*\{", paper_tex)
        if title_m:
            title, _ = _braced_arg(paper_tex, title_m.end() - 1)
            parts.append(f"Title: {title.strip()[:300]}")
        all_blocks = self._split_sections(paper_tex)
        abstract = next((b for b in all_blocks if b.level == "abstract"), None)
        if abstract is not None and abstract not in blocks:
            body = re.sub(r"\s+", " ", abstract.text).strip()
            parts.append(f"Abstract (context only, do not return): {body[:1500]}")
        outline: list[str] = []
        targets = {(b.start, b.end) for b in blocks}
        for b in all_blocks:
            if b.level == "abstract":
                continue
            indent = "  " if b.level == "subsection" else ""
            mark = "  <-- REVISE THIS ONE" if (b.start, b.end) in targets else ""
            outline.append(f"{indent}- {b.title}{mark}")
        if outline:
            parts.append("Manuscript outline:\n" + "\n".join(outline))
        return "\n\n".join(parts)

    def _build_section_revision_prompt(
        self,
        paper_tex: str,
        blocks: list[TexBlock],
        strengths: list[str],
        weaknesses: list[str],
        suggestions: list[str],
        questions: list[str],
        focus_dims: list[str],
        diagnosis: dict,
        lint_report: Any = None,
    ) -> str:
        """Prompt that asks for ONLY *blocks* back, one fenced block each."""
        strengths_text = "\n".join(f"- {s}" for s in strengths) or "- (none listed)"
        weaknesses_text = "\n".join(f"- {w}" for w in weaknesses) or "- (none listed)"
        suggestions_text = "\n".join(f"- {s}" for s in suggestions) or "- (none listed)"
        questions_text = "\n".join(f"- {q}" for q in questions) or "- (none listed)"
        focus_text = ", ".join(focus_dims) if focus_dims else "general quality"

        dim_scores = diagnosis.get("dimension_scores", {})
        scores_text = "\n".join(
            f"- {name}: {score}/10" for name, score in dim_scores.items()
        )
        names_text = ", ".join(f"{i + 1}. {b.title}" for i, b in enumerate(blocks))
        defects_text, citation_text = self._lint_evidence_text(lint_report)
        digest = self._manuscript_digest(paper_tex, blocks)

        sections_text = "\n\n".join(
            f"### SECTION {i + 1}: {b.title}\n\n```latex\n{b.text.strip()}\n```"
            for i, b in enumerate(blocks)
        )

        return f"""## Task

Revise **only** the {len(blocks)} manuscript section(s) reproduced at the end of
this message ({names_text}). Your goal is to improve the paper's quality on the
weakest reviewer dimensions: **{focus_text}**.

The rest of the manuscript is NOT being revised and is not shown in full. Do not
ask for it, do not reproduce it, and do not return the preamble or
\\end{{document}}.

## Output format (strict)

For each section, return one fenced LaTeX block, in the same order, preceded by
its marker:

### SECTION 1
```latex
<the complete revised section, starting with its own heading command>
```

- Return the section COMPLETE, from its own heading command
  (e.g. \\section{{...}}) to its final sentence. A partial section is discarded.
- Keep each heading command and its title EXACTLY as given.
- Do not merge, split, add, or drop sections.
- Return nothing else: no preamble, no other sections, no commentary outside
  the fenced blocks.

## Constraints

- Do NOT change any data, results, numbers, tables, or figures.
- Do NOT add or remove \\begin{{table}}, \\begin{{figure}}, or \\includegraphics commands.
  Reproduce every float inside these sections byte-for-byte.
- Only revise the **prose**: introduction framing, related work positioning,
  discussion depth, limitation acknowledgment, and clarity of communication.
- You may add or rephrase sentences but must not fabricate results.
- If a weakness concerns the analysis itself, the honest fix is an explicit
  limitation, NOT a stronger claim and NOT edited numbers.
- Keep every \\label and \\ref intact; other sections still point at them.
{defects_text}{citation_text}
## Manuscript context (do not return this)

{digest}

## Reviewer Scores (1-10)

{scores_text}

Overall: {diagnosis.get('overall_score', '?')}/10

## Reviewer Strengths

{strengths_text}

## Reviewer Weaknesses

{weaknesses_text}

## Reviewer Suggestions

{suggestions_text}

## Questions for Authors

{questions_text}

## Sections to revise

{sections_text}
"""

    def _build_revision_prompt(
        self,
        paper_tex: str,
        strengths: list[str],
        weaknesses: list[str],
        suggestions: list[str],
        questions: list[str],
        focus_dims: list[str],
        diagnosis: dict,
        lint_report: Any = None,
    ) -> str:
        strengths_text = "\n".join(f"- {s}" for s in strengths) or "- (none listed)"
        weaknesses_text = "\n".join(f"- {w}" for w in weaknesses) or "- (none listed)"
        suggestions_text = "\n".join(f"- {s}" for s in suggestions) or "- (none listed)"
        questions_text = "\n".join(f"- {q}" for q in questions) or "- (none listed)"
        focus_text = ", ".join(focus_dims) if focus_dims else "general quality"

        dim_scores = diagnosis.get("dimension_scores", {})
        scores_text = "\n".join(
            f"- {name}: {score}/10" for name, score in dim_scores.items()
        )

        # Arc P4: name the sections that actually carry the weakest
        # dimension, so the reviser edits there instead of diffusing
        # changes across the whole manuscript.
        sections_text = ", ".join(self._target_section_names(focus_dims))

        # Arc P4: deterministic, machine-verified defects. These are not
        # opinions — the linter re-checks them after the revision.
        defects_text, citation_text = self._lint_evidence_text(lint_report)

        return f"""## Task

Revise the LaTeX paper below to address the reviewer feedback. Your goal is to
improve the paper's quality on the weakest dimensions: **{focus_text}**.

Concentrate your edits in these sections: **{sections_text}**. Leave the rest
of the manuscript byte-identical.

## Constraints

- Do NOT change any data, results, numbers, tables, or figures.
- Do NOT add or remove \\begin{{table}}, \\begin{{figure}}, or \\includegraphics commands.
- Only revise the **prose**: introduction framing, related work positioning,
  discussion depth, limitation acknowledgment, and clarity of communication.
- You may add or rephrase sentences but must not fabricate results.
- If a weakness concerns the analysis itself, the honest fix is an explicit
  limitation, NOT a stronger claim and NOT edited numbers.
- Return the COMPLETE revised paper.tex wrapped in a ```latex code block.
{defects_text}{citation_text}
## Reviewer Scores (1-10)

{scores_text}

Overall: {diagnosis.get('overall_score', '?')}/10

## Reviewer Strengths

{strengths_text}

## Reviewer Weaknesses

{weaknesses_text}

## Reviewer Suggestions

{suggestions_text}

## Questions for Authors

{questions_text}

## Current paper.tex

```latex
{paper_tex}
```
"""

    @staticmethod
    def _float_environments(tex: str) -> list[str]:
        """Normalized table/figure environments + graphics includes.

        Used to prove a prose revision did not touch the evidence.
        Whitespace is collapsed so pure reflowing is not flagged.
        """
        blocks = re.findall(
            r"(\\begin\{(?:table|figure)\*?\}.*?\\end\{(?:table|figure)\*?\})",
            tex,
            re.DOTALL,
        )
        # Whitespace is removed entirely, not merely collapsed: LaTeX is
        # whitespace-insensitive here, so re-indenting a table is a
        # legitimate prose-pass edit while "0.82" -> "0.91" is not.
        out = [re.sub(r"\s+", "", b) for b in blocks]
        out += sorted(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", tex))
        return sorted(out)

    def _revision_is_safe(self, original: str, revised: str) -> tuple[bool, str]:
        """Reject a revision that damaged the manuscript or its evidence.

        Three deterministic checks, each guarding a failure we can
        actually hit: truncation (the model returns a partial document
        under the token budget), evidence tampering (numbers inside
        tables/figures changed), and float churn.
        """
        if not revised or not revised.strip():
            return False, "revision was empty"
        if "\\end{document}" not in revised:
            return False, "revision is truncated (no \\end{document})"
        if len(revised) < 0.6 * len(original):
            return False, (
                f"revision lost {100 * (1 - len(revised) / max(1, len(original))):.0f}% "
                "of the manuscript (likely truncated)"
            )
        before, after = self._float_environments(original), self._float_environments(revised)
        if before != after:
            return False, (
                f"revision altered tables/figures/graphics "
                f"({len(before)} -> {len(after)} float artifacts changed)"
            )
        return True, ""

    def _extract_latex(self, text: str) -> Optional[str]:
        """Extract LaTeX from a ```latex ... ``` code block in LLM response."""
        match = re.search(
            r"```latex\s*\n(.*?)```",
            text,
            re.DOTALL,
        )
        if match:
            return match.group(1).strip()
        # Fallback: look for \documentclass ... \end{document}
        match = re.search(
            r"(\\documentclass.*?\\end\{document\})",
            text,
            re.DOTALL,
        )
        if match:
            return match.group(1).strip()
        return None

    # ------------------------------------------------------------------
    # 5. Full review gate loop
    # ------------------------------------------------------------------

    def _maybe_median_sample(
        self, first_report: dict, pdf_path: Path, cycle: int
    ) -> dict:
        """Borderline-triggered multi-sample median review.

        Config (review_gate): ``median_samples`` (total reviews when
        triggered; default 3; 1 disables) and ``median_trigger_band``
        (absolute distance from the pass threshold that triggers the
        extra samples; default 1.5 ≈ the measured test-retest MAD).
        The report whose overall score is the median of all samples is
        returned, so dimensions and prose stay internally consistent.
        """
        rg_cfg = self.config.get("review_gate", {})
        k = int(rg_cfg.get("median_samples", 3))
        band = float(rg_cfg.get("median_trigger_band", 1.5))
        first_score = (first_report.get("scores") or {}).get("overall_score")
        if k <= 1 or first_score is None:
            return first_report
        if abs(first_score - self.pass_threshold) > band:
            return first_report
        self._log(
            f"Borderline score {first_score} within ±{band} of threshold "
            f"{self.pass_threshold} → median sampling ({k} total reviews)"
        )
        samples: list[tuple[float, dict, int]] = [
            (first_score, first_report, cycle)
        ]
        for extra in range(2, k + 1):
            sample_id = cycle * 100 + extra
            rep = self.run_lsar(pdf_path, cycle=sample_id)
            score = ((rep or {}).get("scores") or {}).get("overall_score")
            if rep is not None and score is not None:
                samples.append((score, rep, sample_id))
        samples.sort(key=lambda triple: triple[0])
        median_score, median_report, median_id = samples[len(samples) // 2]
        self._log(
            f"Median sampling: scores={[round(s, 2) for s, _, _ in samples]} "
            f"→ gating on median {median_score}"
        )
        # I5 (AERA_OPEN audit): record WHICH sample the gate used — the
        # old summary attributed the median score to cycle_1 while
        # final_review_path pointed at a report with a different score,
        # and the sample set survived only in pipeline.log.
        median_report.setdefault("scores", {})["median_sampling"] = {
            "n_samples": len(samples),
            "all_scores": [s for s, _, _ in samples],
            "gated_sample_dir": f"cycle_{median_id}",
        }
        return median_report

    def run_gate(self) -> dict:
        """Execute the full review gate loop.

        Returns a summary dict with cycle details, final scores, and
        whether the paper passed.
        """
        self._log(
            f"Starting review gate (max_cycles={self.max_cycles}, "
            f"threshold={self.pass_threshold}, floor={self.dimension_floor})"
        )

        per_cycle_scores: list[dict] = []
        final_passed = False
        final_score: float = 0.0
        final_recommendation: str = "Unknown"
        final_review_path: Optional[str] = None

        for cycle in range(1, self.max_cycles + 1):
            self._log(f"--- Review gate cycle {cycle}/{self.max_cycles} ---")

            # 1. Prepare PDF
            pdf_path = self.prepare_pdf(self.output_dir, cycle=cycle)
            if pdf_path is None:
                self._log("Cannot prepare PDF; skipping review gate")
                break

            # 2. Run LSAR (with borderline-triggered median sampling —
            # Arc L follow-up: test-retest MAD 1.9 means one review is
            # unreliable near the threshold. When the first score lands
            # within ``median_trigger_band`` of the threshold, run
            # ``median_samples - 1`` more reviews and gate on the
            # median-score report.)
            report_json = self.run_lsar(pdf_path, cycle)
            if report_json is None:
                self._log("LSAR returned no result; skipping review gate")
                break
            report_json = self._maybe_median_sample(report_json, pdf_path, cycle)

            # 3. Evaluate gate
            passed, diagnosis = self.evaluate_gate(report_json)

            # I1/I2 (AERA_OPEN audit): a reviewer score cannot pass a
            # manuscript whose numbers the analysis never produced or
            # whose UNVERIFIED flag is missing. The reviewer has no way
            # to see either — this deterministic check does.
            honesty = self._honesty_blockers()
            reviewer_passed = passed
            if honesty and passed:
                self._log(
                    "Gate PASS OVERRIDDEN by manuscript honesty lint: "
                    + "; ".join(honesty)
                )
                passed = False
            elif honesty:
                self._log(
                    "Manuscript honesty lint blocking codes present: "
                    + "; ".join(honesty)
                )
            diagnosis["honesty_blockers"] = honesty

            final_score = diagnosis["overall_score"]
            final_recommendation = diagnosis["recommendation"]
            final_passed = passed

            median_info = (report_json.get("scores") or {}).get("median_sampling")
            gated_dir = (
                (median_info or {}).get("gated_sample_dir") or f"cycle_{cycle}"
            )
            cycle_dir = self.output_dir / "lsar_review" / gated_dir
            final_review_path = str(cycle_dir / "lsar_report.json")

            per_cycle_scores.append(
                {
                    "cycle": cycle,
                    "overall_score": final_score,
                    "recommendation": final_recommendation,
                    "passed": passed,
                    "failing_dimensions": diagnosis["failing_dimensions"],
                    "suggested_focus_areas": diagnosis["suggested_focus_areas"],
                    "median_sampling": median_info,
                    "honesty_blockers": honesty,
                }
            )

            if passed:
                self._log(
                    f"Review gate PASSED (cycle {cycle}): "
                    f"score={final_score:.2f}, rec={final_recommendation}"
                )
                break

            self._log(
                f"Review gate FAILED (cycle {cycle}): "
                f"score={final_score:.2f}, rec={final_recommendation}, "
                f"failing={diagnosis['failing_dimensions']}"
            )

            # When the reviewer PASSED and only honesty lint blocked,
            # revision is futile: the reviser cannot recompute analysis
            # values, and the P4 safety guard rightly rejects revisions
            # that touch tables — each extra cycle would recompile and
            # re-review the unchanged manuscript (review finding). Fail
            # now, loudly.
            if reviewer_passed and honesty:
                self._log(
                    "Honesty blockers cannot be fixed by prose revision "
                    "(fabricated numbers require re-analysis); ending the "
                    "gate without further cycles."
                )
                break

            # 4. If cycles remain, revise the paper
            if cycle < self.max_cycles:
                tex_path = self.output_dir / "paper.tex"
                if not tex_path.exists():
                    self._log("paper.tex not found; cannot revise")
                    break

                current_tex = tex_path.read_text(encoding="utf-8")

                # Build review markdown for the prompt
                review_md_path = cycle_dir / "lsar_report.md"
                review_md = ""
                if review_md_path.exists():
                    review_md = review_md_path.read_text(encoding="utf-8")

                # Keep the pre-revision manuscript so a bad revision is
                # always recoverable (and diffable after the run).
                try:
                    cycle_dir.mkdir(parents=True, exist_ok=True)
                    (cycle_dir / "paper_pre_revision.tex").write_text(
                        current_tex, encoding="utf-8"
                    )
                except OSError:
                    pass

                revised_tex = self.revise_from_review(
                    paper_tex=current_tex,
                    report_json=report_json,
                    diagnosis=diagnosis,
                    lint_report=self._last_lint,
                )

                # Arc P4 guards. revise_from_review returns the ORIGINAL
                # string on LLM failure, and the old code wrote it back
                # and recompiled anyway while logging "Revised paper.tex
                # written" — a no-op that looked like progress.
                if revised_tex == current_tex:
                    # The old message claimed it was "skipping rewrite and
                    # recompile". Only half of that was true: paper.tex is
                    # not rewritten, but the loop continues and the next
                    # cycle's prepare_pdf runs pdflatex again anyway (8.5s
                    # later in the live run). Say what actually happens.
                    self._log(
                        "Revision was a no-op (LLM failed or returned the "
                        "original); paper.tex left unchanged. The next cycle "
                        "still recompiles and re-reviews the unchanged "
                        "manuscript."
                    )
                    continue
                safe, reason = self._revision_is_safe(current_tex, revised_tex)
                if not safe:
                    self._log(
                        f"Revision REJECTED and discarded: {reason}. "
                        "Keeping the pre-revision manuscript."
                    )
                    continue

                # Arc P3 backstop: the reviser writes prose freely and can
                # introduce a citation key that references.bib does not
                # have. Reconcile before writing, or the reviewed PDF for
                # the next cycle renders it as [?].
                bib_path = self.output_dir / "references.bib"
                try:
                    from src.citations import reconcile_citations

                    lit_path = self.output_dir / "literature_context_expanded.json"
                    if not lit_path.exists():
                        lit_path = self.output_dir / "literature_context.json"
                    papers = []
                    if lit_path.exists():
                        papers = (
                            json.loads(lit_path.read_text(encoding="utf-8")).get(
                                "papers"
                            )
                            or []
                        )
                    current_bib = (
                        bib_path.read_text(encoding="utf-8")
                        if bib_path.exists()
                        else ""
                    )
                    revised_tex, revised_bib, cstats = reconcile_citations(
                        revised_tex, current_bib, papers
                    )
                    if revised_bib != current_bib:
                        bib_path.write_text(revised_bib, encoding="utf-8")
                    self._log(
                        f"Post-revision reconciliation: {cstats['cited']} cited, "
                        f"{cstats['backfilled']} back-filled, "
                        f"{cstats['stripped']} invented keys stripped"
                    )
                except Exception as exc:  # noqa: BLE001
                    self._log(f"Post-revision reconciliation skipped: {exc}")

                # Write revised paper.tex and recompile
                tex_path.write_text(revised_tex, encoding="utf-8")
                self._log("Revised paper.tex written; recompiling LaTeX")
                self._compile_full_latex(self.output_dir)

        # Build final summary. I5: threshold provenance and the median
        # sample set used to live only in pipeline.log — the summary now
        # carries everything needed to audit the verdict from disk.
        summary: dict[str, Any] = {
            "cycles_used": len(per_cycle_scores),
            "max_cycles": self.max_cycles,
            "final_score": final_score,
            "final_recommendation": final_recommendation,
            "per_cycle_scores": per_cycle_scores,
            "final_review_path": final_review_path,
            "passed": final_passed,
            "threshold_used": self.pass_threshold,
            "threshold_source": self.calibration_source,
            "advisory_mode": getattr(self, "advisory_mode", None),
            "venue": self.venue,
            "dimension_floor": self.dimension_floor,
        }

        # Persist summary
        summary_path = self.output_dir / "lsar_review" / "gate_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
        self._log(
            f"Review gate finished: passed={final_passed}, "
            f"cycles={len(per_cycle_scores)}, final_score={final_score:.2f}"
        )
        return summary

    def _compile_full_latex(self, run_dir: Path) -> None:
        """Run the standard pdflatex → bibtex → pdflatex → pdflatex sequence."""
        from src.sandbox import compile_latex

        result = compile_latex(str(run_dir))
        if result["success"]:
            self._log("LaTeX recompilation succeeded")
        else:
            failed = [s for s in result["steps"] if s["returncode"] not in (0, 1)]
            for step in failed:
                self._log(
                    f"LaTeX step failed: {step['cmd']} "
                    f"(rc={step['returncode']}): {step['stderr'][:300]}"
                )
