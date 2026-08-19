"""Arc P / P2 — mine per-venue manuscript norms from the LSAR anchor corpus.

Every anchor paper LSAR has reviewed leaves a converted ``paper.md`` +
``metadata.json`` in the LSAR outputs directory. Those anchors are REAL
published papers at their venues, so their reference counts and body
lengths are the empirical norms our manuscripts should meet. This script
measures them and writes ``data_registry/venue_norms.yaml``, which the
manuscript linter (src/manuscript_linter.py) consumes as thresholds.

Reference counting is heuristic (year-pattern occurrences inside the
References block; max over APA parenthesized-year, numbered-marker, and
ACM bare-year styles) — noisy per paper (~±10%), stable at the
P25/median level across a venue corpus. The method is recorded in the
output file for honesty.

Usage:
    python scripts/mine_venue_norms.py [--lsar-outputs "$LSAR_HOME/outputs"]
"""
from __future__ import annotations

import os

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Optional

DEFAULT_LSAR_OUTPUTS = Path(
    os.environ.get("LSAR_HOME", "../LSAR")
) / "outputs"
OUTPUT_FILE = Path(__file__).resolve().parent.parent / "data_registry" / "venue_norms.yaml"

# Anchors excluded from calibration stay excluded from norms.
EXCLUDED_STEMS = {
    "jla_9743",              # issue editorial, failed sections sanity
    "aera_open_ej1494736",   # stage-1 metadata doubly corrupted (math title
                             # + boilerplate abstract); excluded from the
                             # calibration P25 for the same reason
}


def venue_for_dir(name: str) -> Optional[str]:
    """Map an LSAR output dir name to its anchor venue.

    jedm_<id>_<ts> -> JEDM; jla_<id>_<ts> -> JLA; the 2026-07-03 batch
    (EDM-2024 conference anchors) -> EDM. Non-anchor dirs return None.
    """
    if name.startswith("jedm_"):
        return "JEDM"
    if name.startswith("jla_"):
        return "JLA"
    if name.startswith("aera_open_"):
        return "AERA_OPEN"
    if "_20260703_" in name:
        return "EDM"
    return None


def anchor_stem(name: str) -> str:
    """Dedupe key: dir name without the trailing _YYYYMMDD_HHMMSS."""
    return re.sub(r"_\d{8}_\d{6}$", "", name)


def refs_block(md: str) -> Optional[str]:
    """Slice the References section out of converted markdown."""
    m = re.search(
        r"^(?:#{1,3}\s*)?(?:\*\*)?references(?:\*\*)?\s*$",
        md,
        re.IGNORECASE | re.MULTILINE,
    )
    if not m:
        return None
    tail = md[m.end():]
    stop = re.search(
        r"^(?:#{1,3}\s+|\*\*)(?:appendix|acknowledg|supplement)",
        tail,
        re.IGNORECASE | re.MULTILINE,
    )
    return tail[: stop.start()] if stop else tail


def count_refs(block: str) -> int:
    """Estimate entry count: max over the three citation-style signals."""
    paren_years = len(re.findall(r"\((?:19|20)\d{2}[a-z]?\)", block))
    numbered = len(re.findall(r"^\s*\[\d{1,3}\]", block, re.MULTILINE))
    acm_years = len(re.findall(r"\.\s(?:19|20)\d{2}\.\s", block))
    return max(paren_years, numbered, acm_years)



# Publication year of each anchor corpus, needed to convert a reference's
# year into an AGE. Verified from the PDFs themselves: the JEDM running
# footer reads "Volume 18, No 1, 2026" and JLA self-DOIs are
# 10.18608/jla.2026.<id>. Recorded here (and echoed into the output file)
# because every ref_age number below is conditional on them: under a 2025
# assumption JEDM's le2 share moves 0.312 -> 0.431.
VENUE_PUB_YEAR: dict[str, int] = {
    "EDM": 2024, "JEDM": 2026, "JLA": 2026,
    "AERA_OPEN": 2025,  # all anchors are Vol. 11 (2025)
}

# Tolerance in percentage points, per spec 1.2: JEDM (n=10, 664 refs)
# has the soundest per-bin estimates; EDM carries lossy PDF conversions
# and JLA is the thinnest corpus, so both get a wider band.
VENUE_TOLERANCE_PP: dict[str, float] = {
    "EDM": 12.0, "JEDM": 10.0, "JLA": 12.0,
    "AERA_OPEN": 12.0,  # n=11, qualitative-leaning mix -> wide band
}

# Age bins, byte-identical to src/citations.py and src/manuscript_linter.py.
# NOTE: the keys MUST be quoted in YAML — YAML 1.1 reads bare 3_5 as the
# integer 35 (underscore digit separator).
AGE_BIN_ORDER = ("le2", "3_5", "6_10", "11_20", "gt20")


def bucket_of_age(age: int) -> str:
    if age <= 2:
        return "le2"
    if age <= 5:
        return "3_5"
    if age <= 10:
        return "6_10"
    if age <= 20:
        return "11_20"
    return "gt20"


def reference_years(block: str) -> list[int]:
    """Publication years appearing inside a References block."""
    return [int(y) for y in re.findall(r"(?:19|20)\d{2}", block)]


def measure_anchor(paper_md: Path, metadata_json: Path) -> Optional[dict]:
    md = paper_md.read_text(encoding="utf-8", errors="replace")
    block = refs_block(md)
    if block is None:
        return None
    n_refs = count_refs(block)
    if n_refs == 0:
        return None
    body = md[: md.find(block)]
    meta = {}
    if metadata_json.exists():
        meta = json.loads(metadata_json.read_text(encoding="utf-8"))
    return {
        "refs": n_refs,
        "body_words": len(body.split()),
        "pages": meta.get("page_count", 0),
        "ref_years": reference_years(block),
    }


def p25(values: list[float]) -> float:
    qs = statistics.quantiles(sorted(values), n=4, method="inclusive")
    return round(qs[0], 1)


def mine(lsar_outputs: Path) -> dict:
    # newest run per anchor stem wins (dir names sort by timestamp)
    latest: dict[str, Path] = {}
    for d in sorted(p for p in lsar_outputs.iterdir() if p.is_dir()):
        if venue_for_dir(d.name) is None:
            continue
        stem = anchor_stem(d.name)
        if stem in EXCLUDED_STEMS:
            continue
        if not (d / "paper.md").exists():
            continue
        latest[stem] = d

    per_venue: dict[str, list[dict]] = {}
    for stem, d in latest.items():
        venue = venue_for_dir(d.name)
        m = measure_anchor(d / "paper.md", d / "metadata.json")
        if m is None:
            print(f"  skip {stem}: no measurable references block")
            continue
        m["anchor"] = stem
        per_venue.setdefault(venue, []).append(m)

    norms: dict[str, dict] = {}
    for venue, rows in sorted(per_venue.items()):
        refs = [r["refs"] for r in rows]
        words = [r["body_words"] for r in rows]
        pub_year = VENUE_PUB_YEAR.get(venue)
        ages: list[int] = []
        if pub_year:
            for r in rows:
                for y in r.get("ref_years", []):
                    if 1900 <= y <= pub_year:
                        ages.append(pub_year - y)
        ref_age: dict = {}
        if ages:
            counts = {b: 0 for b in AGE_BIN_ORDER}
            for a in ages:
                counts[bucket_of_age(a)] += 1
            total = float(len(ages))
            ref_age = {
                "assumed_pub_year": pub_year,
                "n_refs_dated": len(ages),
                "median_age": int(statistics.median(ages)),
                # KEY NAME IS A CONTRACT: src/citations.py
                # composition_age_profile() and src/manuscript_linter.py
                # venue_age_profile() both read ``ref_age.buckets``.
                # Naming it anything else silently falls back to the
                # pooled default and the venue profiles go unused.
                "buckets": {b: round(counts[b] / total, 3) for b in AGE_BIN_ORDER},
                "tolerance_pp": VENUE_TOLERANCE_PP.get(venue, 12.0),
                "frac_older_than_10y": round(
                    (counts["11_20"] + counts["gt20"]) / total, 3),
            }
        norms[venue] = {
            "n_anchors": len(rows),
            "ref_age": ref_age,
            "refs": {
                "p25": p25(refs),
                "median": statistics.median(refs),
                "min": min(refs),
                "max": max(refs),
            },
            "body_words": {
                "p25": int(p25(words)),
                "median": int(statistics.median(words)),
            },
            "anchors": {
                r["anchor"]: {"refs": r["refs"], "body_words": r["body_words"]}
                for r in sorted(rows, key=lambda x: x["anchor"])
            },
        }
    return norms


def write_yaml(norms: dict, out_path: Path) -> None:
    import yaml

    header = (
        "# Per-venue manuscript norms mined from the LSAR anchor corpus\n"
        "# (real published papers at each venue). Generated by\n"
        "# scripts/mine_venue_norms.py — regenerate after adding anchors.\n"
        "#\n"
        "# refs counts are heuristic (year-pattern occurrences in the\n"
        "# References block; ~±10% per paper, stable at P25/median).\n"
        "# The manuscript linter uses refs.p25 as the citation floor and\n"
        "# body_words.p25 as the length floor for gate-advisory checks.\n"
    )
    out_path.write_text(
        header + yaml.safe_dump({"venues": norms}, sort_keys=True),
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lsar-outputs", default=str(DEFAULT_LSAR_OUTPUTS))
    ap.add_argument("--out", default=str(OUTPUT_FILE))
    args = ap.parse_args()

    norms = mine(Path(args.lsar_outputs))
    for venue, block in norms.items():
        print(
            f"{venue}: n={block['n_anchors']} refs P25={block['refs']['p25']} "
            f"median={block['refs']['median']} range=[{block['refs']['min']},"
            f"{block['refs']['max']}] body_words P25={block['body_words']['p25']}"
        )
    write_yaml(norms, Path(args.out))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
