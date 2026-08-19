"""Summarize a gated pipeline run: citation depth, lint defects, gate deltas.

Built for the Arc P validation but written for any run that produced an
``lsar_review/`` directory. Answers the questions an arc close-out needs:

* did citation depth reach the venue norm, and where did the references
  come from (selected vs pool top-up)?
* does every cited key have a bib entry (the F-E2A-SECTIONWISE-BIB-DRIFT
  regression check)?
* did the linter's defect count fall across revision cycles?
* did the gate score move, and did any revision get rejected by the
  Arc P4 guards?

Usage:
    python scripts/analyze_gated_run.py runs/<run>/output [--json]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Optional


def _load(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def literature_summary(out: Path) -> dict:
    sel = _load(out / "literature_context.json") or {}
    exp = _load(out / "literature_context_expanded.json") or {}
    pool = _load(out / "retrieved_literature.json") or {}

    def papers(d: dict) -> list:
        return (d or {}).get("papers") or []

    used = papers(exp) or papers(sel)
    return {
        "selected": len(papers(sel)),
        "pool": len(papers(pool)),
        "available_to_writer": len(used),
        "with_venue": sum(1 for p in used if (p.get("venue") or "").strip()),
        "with_doi": sum(1 for p in used if (p.get("doi") or "").strip()),
    }


def citation_summary(out: Path) -> dict:
    """Cited-vs-bib consistency, computed the same way the linter does."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.manuscript_linter import _BIB_ENTRY, cited_keys

    tex_path, bib_path = out / "paper.tex", out / "references.bib"
    if not tex_path.exists():
        return {"error": "paper.tex missing"}
    tex = tex_path.read_text(encoding="utf-8", errors="replace")
    bib = bib_path.read_text(encoding="utf-8", errors="replace") if bib_path.exists() else ""
    cited = cited_keys(tex)
    entries = set(_BIB_ENTRY.findall(bib))
    fabricated = len(
        re.findall(r"Proceedings of the Educational Data Mining Conference", bib)
    )
    return {
        "distinct_citations": len(cited),
        "bib_entries": len(entries),
        "dangling_keys": sorted(cited - entries),
        "uncited_entries": len(entries - cited),
        "placeholder_keys": sorted(k for k in cited if k.startswith("placeholder")),
        "fabricated_edm_venues": fabricated,
        "honest_missing_venue": bib.count("Venue metadata unavailable"),
    }


def lint_by_cycle(out: Path) -> list[dict]:
    rows = []
    review_dir = out / "lsar_review"
    if review_dir.is_dir():
        for cdir in sorted(
            review_dir.glob("cycle_*"),
            key=lambda p: int(re.sub(r"\D", "", p.name) or 0),
        ):
            rep = _load(cdir / "manuscript_lint.json")
            if not rep:
                continue
            defects = rep.get("defects", [])
            rows.append({
                "cycle": cdir.name,
                "errors": sum(1 for d in defects if d.get("severity") == "error"),
                "warns": sum(1 for d in defects if d.get("severity") != "error"),
                "citations": (rep.get("metrics") or {}).get("n_citations_distinct"),
                "body_words": (rep.get("metrics") or {}).get("body_words"),
                "venue_p25": (rep.get("metrics") or {}).get("venue_refs_p25"),
                "codes": sorted({d.get("code") for d in defects}),
            })
    return rows


def gate_summary(out: Path) -> dict:
    s = _load(out / "lsar_review" / "gate_summary.json") or {}
    cycles = s.get("per_cycle_scores") or []
    scores = [c.get("overall_score") for c in cycles]
    return {
        "cycles_used": s.get("cycles_used"),
        "passed": s.get("passed"),
        "final_score": s.get("final_score"),
        "final_recommendation": s.get("final_recommendation"),
        "scores_by_cycle": scores,
        "delta_first_to_last": (
            round(scores[-1] - scores[0], 2)
            if len(scores) >= 2 and None not in (scores[0], scores[-1]) else None
        ),
        "weakest_by_cycle": [
            (c.get("suggested_focus_areas") or [{}])[0].get("dimension")
            for c in cycles
        ],
    }


def revision_events(out: Path) -> dict:
    """Read the pipeline log for Arc P3/P4 decisions."""
    log = out / "pipeline.log"
    if not log.exists():
        return {}
    text = log.read_text(encoding="utf-8", errors="replace")
    def grab(pattern: str) -> list[str]:
        return [m.strip() for m in re.findall(pattern, text)]
    return {
        "bib_reconciliation": grab(r"Bib reconciliation: ([^\n]+)"),
        "citation_depth": grab(r"Citation depth: ([^\n]+)"),
        "post_revision_reconciliation": grab(r"Post-revision reconciliation: ([^\n]+)"),
        "revisions_rejected": grab(r"Revision REJECTED[^\n]*"),
        "revisions_noop": grab(r"Revision was a no-op[^\n]*"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    out = Path(args.run_dir)

    report = {
        "run_dir": str(out),
        "literature": literature_summary(out),
        "citations": citation_summary(out),
        "lint_by_cycle": lint_by_cycle(out),
        "gate": gate_summary(out),
        "events": revision_events(out),
    }

    if args.json:
        print(json.dumps(report, indent=2))
        return

    lit, cit, gate = report["literature"], report["citations"], report["gate"]
    print(f"\n=== {out} ===\n")
    print("LITERATURE")
    print(f"  selected={lit['selected']}  pool={lit['pool']}  "
          f"available_to_writer={lit['available_to_writer']}")
    print(f"  with_venue={lit['with_venue']}  with_doi={lit['with_doi']}")
    print("\nCITATIONS (final manuscript)")
    if "error" in cit:
        print("  " + cit["error"])
    else:
        print(f"  distinct citations = {cit['distinct_citations']}  "
              f"bib entries = {cit['bib_entries']}")
        print(f"  dangling keys      = {len(cit['dangling_keys'])} "
              f"{cit['dangling_keys'][:6] if cit['dangling_keys'] else '(none - drift fixed)'}")
        print(f"  placeholder keys   = {len(cit['placeholder_keys'])}")
        print(f"  fabricated EDM venues = {cit['fabricated_edm_venues']} "
              f"(honest 'unavailable' notes = {cit['honest_missing_venue']})")
    print("\nLINT BY CYCLE")
    for row in report["lint_by_cycle"]:
        print(f"  {row['cycle']}: errors={row['errors']} warns={row['warns']} "
              f"citations={row['citations']} (venue P25={row['venue_p25']}) "
              f"words={row['body_words']}")
        print(f"      {', '.join(row['codes']) or '(clean)'}")
    if not report["lint_by_cycle"]:
        print("  (no per-cycle lint copies)")
    print("\nGATE")
    print(f"  cycles={gate['cycles_used']} passed={gate['passed']} "
          f"final={gate['final_score']} ({gate['final_recommendation']})")
    print(f"  scores by cycle = {gate['scores_by_cycle']}  "
          f"delta = {gate['delta_first_to_last']}")
    print(f"  weakest dimension by cycle = {gate['weakest_by_cycle']}")
    print("\nARC P EVENTS")
    for k, v in report["events"].items():
        if v:
            print(f"  {k}:")
            for line in v[:4]:
                print(f"      {line}")
    print()


if __name__ == "__main__":
    main()
