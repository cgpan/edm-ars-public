#!/usr/bin/env python
"""One-command paper diagnosis: compile PDF then run LSAR review.

Convenience wrapper that chains prepare_for_review.py and run_lsar_review.py
to produce a full diagnostic report for an EDM-ARS generated paper.

Usage:
    python scripts/diagnose_paper.py
    python scripts/diagnose_paper.py --run-dir output/run_20260317_064410
    python scripts/diagnose_paper.py --venue LAK
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent


def run_step(label: str, cmd: list[str]) -> int:
    """Run a subprocess and stream its output, returning the exit code."""
    print(f"\n{'=' * 60}")
    print(f"  STEP: {label}")
    print("=" * 60 + "\n")

    result = subprocess.run(
        cmd,
        cwd=str(SCRIPTS_DIR.parent),
        shell=True,
    )
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose an EDM-ARS paper: compile to PDF, then run LSAR review",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="EDM-ARS run directory (default: most recent run under output/)",
    )
    parser.add_argument(
        "--venue",
        choices=["AIED", "EDM", "L@S", "LAK", "auto"],
        default="EDM",
        help="Target venue for LSAR review (default: EDM)",
    )
    args = parser.parse_args()

    python = sys.executable  # use the same interpreter

    # --- Step 1: Prepare PDF ---
    prepare_cmd = [python, str(SCRIPTS_DIR / "prepare_for_review.py")]
    if args.run_dir:
        prepare_cmd += ["--run-dir", str(args.run_dir)]

    rc = run_step("Prepare compilable PDF", prepare_cmd)
    if rc != 0:
        print("\nDiagnosis aborted: PDF compilation failed.", file=sys.stderr)
        print("Fix the LaTeX errors above, then re-run.", file=sys.stderr)
        sys.exit(1)

    # --- Step 2: Run LSAR review ---
    review_cmd = [python, str(SCRIPTS_DIR / "run_lsar_review.py")]
    review_cmd += ["--venue", args.venue]
    if args.run_dir:
        review_cmd += ["--run-dir", str(args.run_dir)]

    rc = run_step("Run LSAR review", review_cmd)
    if rc != 0:
        print("\nDiagnosis aborted: LSAR review failed.", file=sys.stderr)
        print("Check the error messages above for details.", file=sys.stderr)
        sys.exit(1)

    # --- Final banner ---
    print("\n" + "#" * 60)
    print("#" + " " * 58 + "#")
    print("#    DIAGNOSIS COMPLETE                                    #")
    print("#" + " " * 58 + "#")
    print("#" * 60)
    print("\nNext steps:")
    print("  1. Read the LSAR review report in the lsar_review/ folder")
    print("  2. Focus on dimensions scoring below 6 — those need work")
    print("  3. Address the top weaknesses listed in the summary above")
    print("  4. Re-run this script after making improvements\n")


if __name__ == "__main__":
    main()
