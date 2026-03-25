#!/usr/bin/env python
"""Run LSAR review on an EDM-ARS paper PDF.

Imports the LSAR pipeline and runs a structured review, saving results
into an ``lsar_review/`` subdirectory of the EDM-ARS run directory.

Usage:
    python scripts/run_lsar_review.py
    python scripts/run_lsar_review.py --run-dir output/run_20260317_064410
    python scripts/run_lsar_review.py --pdf path/to/paper.pdf --venue EDM
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# LSAR project location — adjust if your installation differs
LSAR_ROOT = Path(r"H:\My Drive\LSAR")


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


def resolve_pdf(run_dir: Path) -> Path:
    """Find the review-ready PDF in a run directory.

    Prefers ``paper_for_review.pdf`` (produced by prepare_for_review.py);
    falls back to ``paper.pdf``.
    """
    for name in ("paper_for_review.pdf", "paper.pdf"):
        candidate = run_dir / name
        if candidate.exists():
            return candidate
    print(f"Error: No PDF found in {run_dir}", file=sys.stderr)
    sys.exit(1)


def run_review(pdf_path: Path, venue: str | None, output_dir: Path) -> tuple[str, dict]:
    """Import LSAR and execute the review pipeline.

    Args:
        pdf_path: Path to the paper PDF.
        venue: Target venue (AIED/EDM/L@S/LAK) or None for auto-detect.
        output_dir: Directory to write LSAR outputs into.

    Returns:
        Tuple of (report_markdown, report_json).
    """
    # Make LSAR importable
    lsar_str = str(LSAR_ROOT)
    if lsar_str not in sys.path:
        sys.path.insert(0, lsar_str)

    from lsar.pipeline import LSARPipeline  # type: ignore[import-untyped]

    config_path = LSAR_ROOT / "config.yaml"
    if not config_path.exists():
        print(f"Error: LSAR config not found at {config_path}", file=sys.stderr)
        sys.exit(1)

    pipeline = LSARPipeline(config_path=config_path)
    return pipeline.run(
        pdf_path=pdf_path,
        venue=venue,
        force=True,
        output_dir=output_dir,
    )


def print_summary(report_json: dict) -> None:
    """Print a concise review summary to stdout."""
    scores = report_json.get("scores", {})
    review = report_json.get("review", {})

    overall = scores.get("overall_score", "N/A")
    recommendation = scores.get("recommendation", "N/A")
    ci = scores.get("confidence_interval", [])

    print("\n" + "=" * 60)
    print("  LSAR REVIEW SUMMARY")
    print("=" * 60)

    # Overall verdict
    print(f"\n  Overall Score:    {overall} / 10")
    if ci:
        print(f"  Confidence:       [{ci[0]:.1f}, {ci[1]:.1f}]")
    print(f"  Recommendation:   {recommendation}")

    # Dimensional scores
    dimensions = scores.get("dimensions", [])
    if dimensions:
        print(f"\n  {'Dimension':<30s} {'Score':>5s}")
        print("  " + "-" * 36)
        for dim in dimensions:
            name = dim.get("name", "unknown").replace("_", " ").title()
            score = dim.get("score", "?")
            bar = "#" * int(score) if isinstance(score, (int, float)) else ""
            print(f"  {name:<30s} {score:>5}  {bar}")

    # Top weaknesses
    weaknesses = review.get("weaknesses", [])
    if weaknesses:
        n = min(3, len(weaknesses))
        print(f"\n  Top {n} Weaknesses:")
        for i, w in enumerate(weaknesses[:n], 1):
            # Truncate long weakness text for the summary
            text = w if len(w) <= 120 else w[:117] + "..."
            print(f"    {i}. {text}")

    print("\n" + "=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LSAR review on an EDM-ARS paper",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="EDM-ARS run directory (default: most recent run under output/)",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=None,
        help="Explicit path to the paper PDF (overrides --run-dir lookup)",
    )
    parser.add_argument(
        "--venue",
        choices=["AIED", "EDM", "L@S", "LAK", "auto"],
        default="EDM",
        help="Target venue for the review (default: EDM)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Custom output directory for LSAR results (default: <run-dir>/lsar_review/)",
    )
    args = parser.parse_args()

    # Resolve PDF path
    if args.pdf:
        pdf_path = args.pdf.resolve()
        if not pdf_path.exists():
            print(f"Error: PDF not found: {pdf_path}", file=sys.stderr)
            sys.exit(1)
        # Default output next to the PDF
        default_output = pdf_path.parent / "lsar_review"
    else:
        if args.run_dir:
            run_dir = args.run_dir.resolve()
        else:
            project_root = Path(__file__).resolve().parent.parent
            run_dir = find_latest_run(project_root / "output")
        pdf_path = resolve_pdf(run_dir)
        default_output = run_dir / "lsar_review"

    output_dir = (args.output_dir or default_output).resolve()
    venue = None if args.venue == "auto" else args.venue

    print(f"PDF:        {pdf_path}")
    print(f"Venue:      {venue or 'auto-detect'}")
    print(f"Output dir: {output_dir}")

    # Run the review
    try:
        report_md, report_json = run_review(pdf_path, venue, output_dir)
    except Exception as e:
        print(f"\nLSAR pipeline failed: {e}", file=sys.stderr)

        # Try to identify which stage failed
        import traceback
        tb = traceback.format_exc()
        for stage_num in range(1, 7):
            if f"stage{stage_num}" in tb.lower():
                print(f"  -> Failure appears to be in Stage {stage_num}", file=sys.stderr)
                break

        print("\nFull traceback:", file=sys.stderr)
        print(tb, file=sys.stderr)
        sys.exit(1)

    # Save a local copy of the JSON for convenience
    local_json = output_dir / "review_summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    local_json.write_text(json.dumps(report_json, indent=2, ensure_ascii=False), encoding="utf-8")

    print_summary(report_json)
    print(f"\nFull report:  {output_dir / 'LSAR_Review_Report.md'}")
    print(f"JSON report:  {output_dir / 'LSAR_Review_Report.json'}")


if __name__ == "__main__":
    main()
