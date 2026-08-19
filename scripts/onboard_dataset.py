"""V3.3 Arc G (G1) — dataset onboarding kit.

Three capabilities, per the roadmap's "onboarding a new dataset ≤ 1
phase" goal:

1. ``profile``: sweep a CSV → per-column stats.
2. ``draft``: profile → Tier-2-style registry YAML draft with typed
   variables, detected missing-value sentinels, tier-3 exclusion
   candidates, an honest-empty ``design_feasibility`` skeleton, and a
   curation checklist the human completes (types, waves, protected
   attributes, rationales).
3. ``parse-sps``: NCES-family fixed-width support — extract the
   ``DATA LIST`` layout (+ variable labels) from an SPSS syntax file
   so the ASCII ``.dat`` can be converted to a labeled CSV
   (``convert-nces``).

Usage:
  python scripts/onboard_dataset.py profile <csv> [--sample-rows N]
  python scripts/onboard_dataset.py draft <csv> --name <dataset_name> --out <yaml>
  python scripts/onboard_dataset.py parse-sps <file.sps>
  python scripts/onboard_dataset.py convert-nces <file.sps> <file.dat> --out <csv> [--columns A,B,...]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

SENTINEL_CANDIDATES = ["-9", "-8", "-7", "-6", "-5", "-1", "999", "9999"]
TIER3_NAME_PATTERNS = [
    r"^w[0-9]", r"^brr", r"_id$", r"^id$", r"_flg$", r"flag$", r"_wt$",
    r"weight", r"^psu", r"^strat",
]


def profile_csv(path: str | Path, sample_rows: int | None = None) -> dict:
    df = pd.read_csv(path, nrows=sample_rows, low_memory=False)
    cols: dict[str, dict] = {}
    n = len(df)
    for c in df.columns:
        s = df[c]
        non_null = s.dropna()
        numeric = pd.to_numeric(non_null, errors="coerce")
        numeric_share = float(numeric.notna().mean()) if len(non_null) else 0.0
        entry: dict = {
            "pct_missing": round(100 * (1 - len(non_null) / n), 2) if n else 0.0,
            "n_unique": int(non_null.nunique()),
            "numeric_share": round(numeric_share, 3),
        }
        if numeric_share > 0.99 and len(non_null):
            entry["range"] = [float(numeric.min()), float(numeric.max())]
            sentinels = [
                v for v in SENTINEL_CANDIDATES
                if (numeric == float(v)).sum() > max(3, 0.001 * n)
            ]
            if sentinels:
                entry["sentinel_candidates"] = sentinels
        else:
            entry["top_values"] = [
                str(v) for v in non_null.value_counts().head(6).index
            ]
        cols[str(c)] = entry
    return {"n_rows": n, "n_cols": len(df.columns), "columns": cols}


def _infer_type(entry: dict) -> str:
    if entry["n_unique"] <= 1:
        return "constant"
    if entry["n_unique"] == 2:
        return "binary"
    if entry.get("numeric_share", 0) > 0.99 and entry["n_unique"] > 15:
        return "continuous"
    return "categorical"


def draft_registry(profile: dict, name: str, source_note: str = "") -> dict:
    variables = []
    tier3_candidates = []
    for col, entry in profile["columns"].items():
        vtype = _infer_type(entry)
        var: dict = {
            "name": col,
            "type": vtype,
            "pct_missing": entry["pct_missing"],
            "wave": "UNKNOWN  # CURATE: assign wave / collection round",
        }
        if "range" in entry:
            var["range"] = entry["range"]
        if "sentinel_candidates" in entry:
            var["missing_sentinel_candidates"] = entry["sentinel_candidates"]
        if any(re.search(p, col.lower()) for p in TIER3_NAME_PATTERNS):
            tier3_candidates.append(col)
        variables.append(var)
    return {
        "name": name,
        "tier": 2,
        "auto_generated_by": "scripts/onboard_dataset.py (Arc G G1)",
        "source_note": source_note,
        "n_rows": profile["n_rows"],
        "curation_checklist": [
            "assign waves / collection rounds per variable",
            "mark protected attributes (protected_attribute: true)",
            "promote outcomes into an outcomes: section with rationales",
            "verify missing sentinels; map labeled-missing categories",
            "confirm tier-3 exclusions (weights/IDs/flags) below",
            "populate design_feasibility (cutoffs? instruments? cohorts?)",
            "write dataset skills (CSV quirks, conventions) per skills/README",
        ],
        "tier3_exclusion_candidates": tier3_candidates,
        "design_feasibility": {
            "running_variables": [],
            "candidate_instruments": [],
            "policy_timing_variables": [],
            "multi_cohort_partner": None,
            "itr_ready": False,
        },
        "variables": {"auto_profiled": variables},
    }


# --------------------------- NCES fixed-width ---------------------------

_DATA_LIST_ITEM = re.compile(r"([A-Za-z0-9_]+)\s+(\d+)\s*-\s*(\d+)")


def parse_sps_layout(sps_path: str | Path) -> list[tuple[str, int, int]]:
    """Extract (name, start, end) column layout from an SPSS DATA LIST.

    Handles the NCES style: a ``DATA LIST`` block terminated by a
    period, listing ``VAR start-end`` runs across many lines.
    """
    text = Path(sps_path).read_text(encoding="latin-1", errors="replace")
    m = re.search(r"DATA\s+LIST[^/]*/(.*?)\.", text, re.DOTALL | re.IGNORECASE)
    if not m:
        raise ValueError(f"No DATA LIST block found in {sps_path}")
    layout = [
        (name, int(start), int(end))
        for name, start, end in _DATA_LIST_ITEM.findall(m.group(1))
    ]
    if not layout:
        raise ValueError("DATA LIST block matched but no column specs parsed")
    return layout


def convert_nces_fixed_width(
    sps_path: str | Path,
    dat_path: str | Path,
    out_csv: str | Path,
    columns: list[str] | None = None,
    chunk_rows: int = 50_000,
) -> int:
    """Convert an NCES ASCII fixed-width file to CSV using the .sps layout.

    ``columns`` restricts output (recommended — the K-5 PUF has tens of
    thousands of columns; registry-relevant subsets keep files sane).
    Returns rows written.
    """
    layout = parse_sps_layout(sps_path)
    if columns:
        wanted = {c.upper() for c in columns}
        layout = [item for item in layout if item[0].upper() in wanted]
        if not layout:
            raise ValueError("None of the requested columns exist in the layout")
    colspecs = [(start - 1, end) for _, start, end in layout]
    names = [name for name, _, _ in layout]
    total = 0
    first = True
    for chunk in pd.read_fwf(
        dat_path, colspecs=colspecs, names=names, chunksize=chunk_rows,
        dtype=str,
    ):
        chunk.to_csv(out_csv, mode="w" if first else "a", header=first, index=False)
        total += len(chunk)
        first = False
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("profile")
    p.add_argument("csv")
    p.add_argument("--sample-rows", type=int, default=None)

    d = sub.add_parser("draft")
    d.add_argument("csv")
    d.add_argument("--name", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--sample-rows", type=int, default=None)
    d.add_argument("--source-note", default="")

    s = sub.add_parser("parse-sps")
    s.add_argument("sps")

    c = sub.add_parser("convert-nces")
    c.add_argument("sps")
    c.add_argument("dat")
    c.add_argument("--out", required=True)
    c.add_argument("--columns", default=None, help="comma-separated subset")

    args = ap.parse_args()
    if args.cmd == "profile":
        print(json.dumps(profile_csv(args.csv, args.sample_rows), indent=2)[:8000])
    elif args.cmd == "draft":
        reg = draft_registry(
            profile_csv(args.csv, args.sample_rows), args.name, args.source_note
        )
        Path(args.out).write_text(
            yaml.dump(reg, default_flow_style=False, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        print(f"draft written: {args.out} ({len(reg['variables']['auto_profiled'])} variables)")
    elif args.cmd == "parse-sps":
        layout = parse_sps_layout(args.sps)
        print(f"{len(layout)} columns; first 10: {layout[:10]}")
    elif args.cmd == "convert-nces":
        cols = args.columns.split(",") if args.columns else None
        n = convert_nces_fixed_width(args.sps, args.dat, args.out, cols)
        print(f"wrote {n} rows -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
