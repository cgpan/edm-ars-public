#!/usr/bin/env python
"""Prepare EDM-ARS paper.tex for LSAR review.

Creates a compilable copy of paper.tex by replacing any placeholder citations
with inline text markers and compiling to PDF via pdflatex.

Usage:
    python scripts/prepare_for_review.py
    python scripts/prepare_for_review.py --run-dir output/run_20260317_064410
    python scripts/prepare_for_review.py --output-pdf review.pdf
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path


def find_latest_run(output_dir: Path) -> Path:
    """Find the most recent run_* directory under output/."""
    run_dirs = sorted(
        [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("run_")],
        key=lambda d: d.name,
        reverse=True,
    )
    if not run_dirs:
        print(f"Error: No run_* directories found under {output_dir}", file=sys.stderr)
        sys.exit(1)
    return run_dirs[0]


def fix_placeholder_citations(tex: str) -> tuple[str, int]:
    """Replace placeholder citations with inline text markers.

    Handles \\citet{placeholder_XXX}, \\citep{placeholder_XXX}, and
    \\cite{placeholder_XXX}.  The number XXX is stripped of leading zeros
    and placed in square brackets.

    Returns:
        Tuple of (modified_tex, replacement_count).
    """
    count = 0

    def _make_replacer(prefix: str):
        def _replace(m: re.Match) -> str:
            nonlocal count
            count += 1
            num = m.group(1).lstrip("0") or "0"
            return f"{prefix}[{num}]"
        return _replace

    # Order matters: citet/citep before generic cite so the longer commands
    # are matched first.
    tex = re.sub(r"\\citet\{placeholder_(\d+)\}", _make_replacer("Author et al. "), tex)
    tex = re.sub(r"\\citep\{placeholder_(\d+)\}", _make_replacer(""), tex)
    tex = re.sub(r"\\cite\{placeholder_(\d+)\}", _make_replacer(""), tex)

    return tex, count


def remove_bibliography(tex: str) -> str:
    """Comment out \\bibliography and \\bibliographystyle commands."""
    tex = re.sub(
        r"^(\\bibliography\{.*?\})",
        r"% \1  % removed for review",
        tex,
        flags=re.MULTILINE,
    )
    tex = re.sub(
        r"^(\\bibliographystyle\{.*?\})",
        r"% \1  % removed for review",
        tex,
        flags=re.MULTILINE,
    )
    return tex


def ensure_url_package(tex: str) -> str:
    """Add \\usepackage{url} if not already present."""
    if re.search(r"\\usepackage(\[.*?\])?\{url\}", tex):
        return tex
    # Insert right after the \documentclass line
    tex = re.sub(
        r"(\\documentclass[^\n]*\n)",
        r"\1\\usepackage{url}\n",
        tex,
        count=1,
    )
    return tex


def compile_pdf(tex_file: Path, run_bibtex: bool = False) -> bool:
    """Compile a .tex file to PDF using pdflatex (and optionally bibtex).

    Runs the standard pdflatex [-> bibtex -> pdflatex] -> pdflatex sequence.

    Args:
        tex_file: Path to the .tex file to compile.
        run_bibtex: Whether to include a bibtex pass.

    Returns:
        True if the PDF was produced, False otherwise.
    """
    stem = tex_file.stem
    cwd = str(tex_file.parent)

    pdflatex_cmd = ["pdflatex", "-interaction=nonstopmode", tex_file.name]
    bibtex_cmd = ["bibtex", stem]

    steps: list[tuple[str, list[str]]] = [("pdflatex (pass 1)", pdflatex_cmd)]
    if run_bibtex:
        steps.append(("bibtex", bibtex_cmd))
        steps.append(("pdflatex (pass 2)", pdflatex_cmd))
    steps.append(("pdflatex (final)", pdflatex_cmd))

    last_result = None
    for label, cmd in steps:
        print(f"  [{label}] {' '.join(cmd)}")
        try:
            last_result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=120,
                shell=True,
            )
        except FileNotFoundError:
            print(f"Error: '{cmd[0]}' not found on PATH.", file=sys.stderr)
            return False
        except subprocess.TimeoutExpired:
            print(f"Error: {label} timed out after 120 s.", file=sys.stderr)
            return False

        # bibtex errors are warnings; pdflatex returncode 1 is usually just
        # warnings too, so we only hard-fail on returncode >= 2 for pdflatex.
        if "bibtex" not in label and last_result.returncode >= 2:
            print(f"Error: {label} exited with code {last_result.returncode}.", file=sys.stderr)

    # Check whether the PDF was produced
    pdf_path = tex_file.parent / f"{stem}.pdf"
    if pdf_path.exists():
        return True

    # Compilation failed — show diagnostics
    print("\nError: PDF was not produced.", file=sys.stderr)
    if last_result:
        stderr_tail = (last_result.stderr or "")[-2000:]
        stdout_tail = (last_result.stdout or "")[-2000:]
        if stderr_tail:
            print("pdflatex stderr (last 2000 chars):", file=sys.stderr)
            print(stderr_tail, file=sys.stderr)
        if stdout_tail:
            print("pdflatex stdout (last 2000 chars):", file=sys.stderr)
            print(stdout_tail, file=sys.stderr)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare EDM-ARS paper.tex for LSAR review",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Path to the run directory (default: most recent run under output/)",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=None,
        help="Path for the output PDF (default: paper_for_review.pdf in run dir)",
    )
    args = parser.parse_args()

    # Resolve run directory
    if args.run_dir:
        run_dir = args.run_dir.resolve()
    else:
        project_root = Path(__file__).resolve().parent.parent
        run_dir = find_latest_run(project_root / "output")

    if not run_dir.exists():
        print(f"Error: Run directory does not exist: {run_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Run directory: {run_dir}")

    # Read source file
    source_tex = run_dir / "paper.tex"
    if not source_tex.exists():
        print(f"Error: paper.tex not found in {run_dir}", file=sys.stderr)
        sys.exit(1)

    tex = source_tex.read_text(encoding="utf-8")

    # --- Fix placeholder citations ---
    tex, n_replaced = fix_placeholder_citations(tex)

    if n_replaced > 0:
        print(f"Replaced {n_replaced} placeholder citation(s) with inline markers")
        tex = remove_bibliography(tex)
        print("Removed bibliography commands (placeholders detected)")
        run_bibtex = False
    else:
        print("No placeholder citations found — keeping original citations")
        run_bibtex = True

    # --- Ensure \usepackage{url} ---
    tex = ensure_url_package(tex)

    # --- Write review copy ---
    review_tex = run_dir / "paper_for_review.tex"
    review_tex.write_text(tex, encoding="utf-8")
    print(f"Created review copy: {review_tex}")

    # --- Compile to PDF ---
    print("\nCompiling PDF...")
    if not compile_pdf(review_tex, run_bibtex=run_bibtex):
        print("\nPDF compilation failed. Suggestions:", file=sys.stderr)
        print("  - Ensure pdflatex is on your PATH", file=sys.stderr)
        print(f"  - Check the log: {run_dir / 'paper_for_review.log'}", file=sys.stderr)
        print("  - Look for missing packages in the log output", file=sys.stderr)
        sys.exit(1)

    # --- Optionally copy PDF to a custom location ---
    pdf_path = run_dir / "paper_for_review.pdf"
    if args.output_pdf:
        dest = args.output_pdf.resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pdf_path, dest)
        pdf_path = dest

    print(f"\nSuccess! PDF ready for review: {pdf_path}")


if __name__ == "__main__":
    main()
