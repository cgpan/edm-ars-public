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
        self.max_cycles: int = rg_cfg.get("max_cycles", 2)
        self.pass_threshold: float = rg_cfg.get("pass_threshold", 5.5)
        self.dimension_floor: float = rg_cfg.get("dimension_floor", 3)
        self.revision_model: str = rg_cfg.get("revision_model", "claude-sonnet-4-6")
        self.revision_max_tokens: int = rg_cfg.get("revision_max_tokens", 16000)

        # Build LLM client (same pattern as BaseAgent — respects llm_provider)
        provider = config.get("llm_provider", "anthropic")
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

    def prepare_pdf(self, run_dir: Path) -> Optional[Path]:
        """Fix placeholder citations in paper.tex and compile to a clean PDF.

        - Replaces ``\\cite{placeholder_*}`` with ``[XX]``
        - Replaces ``\\citet{placeholder_*}`` with ``Author et al. [XX]``
        - Strips ``\\bibliography{references}`` so bibtex is not needed
        - Writes ``paper_for_review.tex`` and compiles with pdflatex (twice)

        Returns the path to paper_for_review.pdf, or None on failure.
        """
        run_dir = Path(run_dir)
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
            return pdf_path
        else:
            self._log("Review PDF compilation failed; falling back to paper.pdf")
            fallback = run_dir / "paper.pdf"
            if fallback.exists():
                return fallback
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
        """Run pdflatex twice on *tex_file* (no bibtex). Returns True on success."""
        cmd = ["pdflatex", "-interaction=nonstopmode", tex_file]
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

        dim_scores: dict[str, int] = {}
        failing_dims: list[str] = []
        for dim in dimensions:
            name = dim.get("name", "")
            score = dim.get("score", 0)
            dim_scores[name] = score
            if score < self.dimension_floor:
                failing_dims.append(name)

        passed = overall_score >= self.pass_threshold and len(failing_dims) == 0

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
    ) -> str:
        """Use an LLM to revise paper.tex based on LSAR feedback.

        The LLM is instructed to only revise prose (introduction framing,
        related work positioning, discussion depth, limitation
        acknowledgment).  Data, results, and tables are never changed.

        Returns the revised LaTeX string.
        """
        review_block = report_json.get("review", {})
        strengths = review_block.get("strengths", [])
        weaknesses = review_block.get("weaknesses", [])
        suggestions = review_block.get("suggestions", [])
        questions = review_block.get("questions_for_authors", [])

        focus_dims = [fa["dimension"] for fa in diagnosis.get("suggested_focus_areas", [])]

        prompt = self._build_revision_prompt(
            paper_tex=paper_tex,
            strengths=strengths,
            weaknesses=weaknesses,
            suggestions=suggestions,
            questions=questions,
            focus_dims=focus_dims,
            diagnosis=diagnosis,
        )

        self._log("Calling LLM for paper revision based on LSAR feedback")
        try:
            with self._llm_client.messages.stream(
                model=self._llm_model,
                max_tokens=self.revision_max_tokens,
                temperature=0.3,
                system=(
                    "You are a skilled academic writer for educational data mining. "
                    "You revise LaTeX papers to address reviewer feedback while "
                    "preserving all data, results, tables, and figures exactly as-is."
                ),
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                response_text = stream.get_final_text()
        except Exception as exc:
            self._log(f"LLM revision call failed: {exc}")
            return paper_tex  # Return original on failure

        revised_tex = self._extract_latex(response_text)
        if revised_tex:
            self._log("Paper revision complete")
            return revised_tex
        else:
            self._log("Could not extract LaTeX from LLM response; keeping original")
            return paper_tex

    def _build_revision_prompt(
        self,
        paper_tex: str,
        strengths: list[str],
        weaknesses: list[str],
        suggestions: list[str],
        questions: list[str],
        focus_dims: list[str],
        diagnosis: dict,
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

        return f"""## Task

Revise the LaTeX paper below to address the reviewer feedback. Your goal is to
improve the paper's quality on the weakest dimensions: **{focus_text}**.

## Constraints

- Do NOT change any data, results, numbers, tables, or figures.
- Do NOT add or remove \\begin{{table}}, \\begin{{figure}}, or \\includegraphics commands.
- Only revise the **prose**: introduction framing, related work positioning,
  discussion depth, limitation acknowledgment, and clarity of communication.
- You may add or rephrase sentences but must not fabricate results.
- Return the COMPLETE revised paper.tex wrapped in a ```latex code block.

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
            pdf_path = self.prepare_pdf(self.output_dir)
            if pdf_path is None:
                self._log("Cannot prepare PDF; skipping review gate")
                break

            # 2. Run LSAR
            report_json = self.run_lsar(pdf_path, cycle)
            if report_json is None:
                self._log("LSAR returned no result; skipping review gate")
                break

            # 3. Evaluate gate
            passed, diagnosis = self.evaluate_gate(report_json)
            final_score = diagnosis["overall_score"]
            final_recommendation = diagnosis["recommendation"]
            final_passed = passed

            cycle_dir = self.output_dir / "lsar_review" / f"cycle_{cycle}"
            final_review_path = str(cycle_dir / "lsar_report.json")

            per_cycle_scores.append(
                {
                    "cycle": cycle,
                    "overall_score": final_score,
                    "recommendation": final_recommendation,
                    "passed": passed,
                    "failing_dimensions": diagnosis["failing_dimensions"],
                    "suggested_focus_areas": diagnosis["suggested_focus_areas"],
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

                revised_tex = self.revise_from_review(
                    paper_tex=current_tex,
                    report_json=report_json,
                    diagnosis=diagnosis,
                )

                # Write revised paper.tex and recompile
                tex_path.write_text(revised_tex, encoding="utf-8")
                self._log("Revised paper.tex written; recompiling LaTeX")
                self._compile_full_latex(self.output_dir)

        # Build final summary
        summary: dict[str, Any] = {
            "cycles_used": len(per_cycle_scores),
            "max_cycles": self.max_cycles,
            "final_score": final_score,
            "final_recommendation": final_recommendation,
            "per_cycle_scores": per_cycle_scores,
            "final_review_path": final_review_path,
            "passed": final_passed,
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
