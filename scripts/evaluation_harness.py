"""V3.8 Arc E — evaluation harness.

Commands:
  collect   — build/extend the evaluation ledger from EDM-ARS run dirs
              (manifest.json + LSAR scores + paper stats).
  agreement — LSAR-vs-human agreement stats once human ratings exist.
  matrix    — print the remaining paper-matrix cells still to generate
              (gate G-E1) given the ledger.

The ledger is the single evaluation artifact: one entry per paper with
task type, provenance (run id + HEAD), LSAR dimension scores, token/
wall-clock cost, and paper stats.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

MATRIX = {"prediction": 4, "causal_soo": 3, "causal_itr": 3}


def _load_json(path: Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def collect(run_dirs: list[str], out: str) -> None:
    ledger_path = Path(out)
    ledger = _load_json(ledger_path) or {"papers": []}
    known = {p["run_id"] for p in ledger["papers"]}
    for run_dir in run_dirs:
        rd = Path(run_dir)
        manifest = _load_json(rd / "manifest.json") or {}
        run_id = manifest.get("run_id", rd.name)
        if run_id in known:
            print(f"skip (already in ledger): {run_id}")
            continue
        scores = None
        for candidate in (
            rd / "06_reviewgate_lsar" / "scores.json",
            rd / "output" / "lsar_review" / "cycle_1" / "scores.json",
        ):
            scores = _load_json(candidate)
            if scores:
                break
        tex = None
        for candidate in (rd / "05_writer" / "paper.tex", rd / "output" / "paper.tex"):
            if candidate.exists():
                tex = candidate.read_text(encoding="utf-8", errors="replace")
                break
        entry = {
            "run_id": run_id,
            "task_type": (manifest.get("task") or {}).get("task_type")
            or manifest.get("phase", "unknown"),
            "head": manifest.get("head_sha_short") or manifest.get("head"),
            "lsar_overall": (scores or {}).get("overall_score"),
            "lsar_recommendation": (scores or {}).get("recommendation"),
            "lsar_dimensions": {
                d["name"]: d["score"] for d in (scores or {}).get("dimensions", [])
            },
            "tokens_total": (manifest.get("tokens") or {}).get("total")
            or manifest.get("tokens_attempt3", {}).get("total"),
            "wall_clock_s": manifest.get("wall_clock_seconds"),
            "paper_words_approx": len(tex.split()) if tex else None,
            "run_dir": str(rd),
        }
        ledger["papers"].append(entry)
        print(f"added: {run_id} (task={entry['task_type']}, LSAR={entry['lsar_overall']})")
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    print(f"ledger: {ledger_path} ({len(ledger['papers'])} papers)")


def agreement(ledger_path: str, human_csv: str) -> None:
    ledger = _load_json(Path(ledger_path)) or {"papers": []}
    by_id = {p["run_id"]: p for p in ledger["papers"]}
    rows = list(csv.DictReader(open(human_csv, encoding="utf-8")))
    if not rows:
        print("no human ratings yet — gate G-E2")
        return
    pairs_overall: list[tuple[float, float]] = []
    dim_pairs: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        paper = by_id.get(row["paper_id"])
        if not paper:
            continue
        score = float(row["score"])
        if row["dimension"] == "overall" and paper.get("lsar_overall") is not None:
            pairs_overall.append((paper["lsar_overall"], score))
        elif row["dimension"] in paper.get("lsar_dimensions", {}):
            dim_pairs.setdefault(row["dimension"], []).append(
                (paper["lsar_dimensions"][row["dimension"]], score)
            )

    def _pearson(pairs: list[tuple[float, float]]) -> float | None:
        if len(pairs) < 3:
            return None
        xs, ys = zip(*pairs)
        try:
            return statistics.correlation(list(xs), list(ys))
        except statistics.StatisticsError:
            return None

    print(f"overall: n={len(pairs_overall)} r={_pearson(pairs_overall)}")
    if pairs_overall:
        mad = statistics.mean(abs(a - b) for a, b in pairs_overall)
        print(f"overall MAD (LSAR vs human): {mad:.3f}")
    for dim, pairs in sorted(dim_pairs.items()):
        print(f"  {dim}: n={len(pairs)} r={_pearson(pairs)}")


def matrix(ledger_path: str) -> None:
    ledger = _load_json(Path(ledger_path)) or {"papers": []}
    have: dict[str, int] = {}
    for p in ledger["papers"]:
        have[p["task_type"]] = have.get(p["task_type"], 0) + 1
    print("paper-matrix status (gate G-E1 for the remainder):")
    for tt, target in MATRIX.items():
        print(f"  {tt}: {have.get(tt, 0)}/{target}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--runs", nargs="+", required=True)
    c.add_argument("--out", default="evaluation/ledger.json")
    a = sub.add_parser("agreement")
    a.add_argument("--ledger", default="evaluation/ledger.json")
    a.add_argument("--human", required=True)
    m = sub.add_parser("matrix")
    m.add_argument("--ledger", default="evaluation/ledger.json")
    args = ap.parse_args()
    if args.cmd == "collect":
        collect(args.runs, args.out)
    elif args.cmd == "agreement":
        agreement(args.ledger, args.human)
    elif args.cmd == "matrix":
        matrix(args.ledger)


if __name__ == "__main__":
    main()
