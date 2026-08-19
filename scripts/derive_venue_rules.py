"""Arc T -- recompute and audit the blind venue-fit rule table (v2).

``data_registry/venue_fit_rules_v2.yaml`` asserts anchor counts in every
evidence string. This script rebuilds the anchor corpus from disk and
recomputes every one of those counts, so the table is reproducible rather
than merely asserted. It also checks that each shipped delta obeys the
table's own stated derivation policy, and provides the reference
evaluator for the table's declarative predicates.

No network, no LLM, no run outcomes. The corpus is published papers only:
``paper.md`` + ``metadata.json``. LSAR's reviews of those papers
(``scores.json``, ``review.json``, ``LSAR_Review_Report.*``) are never
opened -- the table is derived from what the venues publish, not from how
anything scored.

Usage
-----
    python scripts/derive_venue_rules.py
    python scripts/derive_venue_rules.py --corpus "$LSAR_HOME/outputs"
    python scripts/derive_venue_rules.py --json report.json

Exit code 0 iff every asserted count matches the recomputation and every
delta obeys the policy.
"""
from __future__ import annotations

import os

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RULES_PATH = REPO_ROOT / "data_registry" / "venue_fit_rules_v2.yaml"
DEFAULT_CORPUS_PATH = Path(
    os.environ.get("LSAR_HOME", "../LSAR")
) / "outputs"

VENUES = ("EDM", "JEDM", "JLA")


# --------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Anchor:
    """One published paper. Stem is provenance only, never semantics."""

    stem: str
    venue: str
    abstract: str
    full_text: str

    def text(self, field_name: str) -> str:
        if field_name == "abstract":
            return self.abstract
        if field_name == "full_text":
            return self.full_text
        raise ValueError(f"unknown corpus field: {field_name!r}")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).lower()


# The v2 rule table was derived from exactly these venues; anchors from any
# other venue must never silently join its corpus. The old bare-else made
# EDM the default for EVERY unknown prefix, so the first AERA Open
# calibration review that landed in LSAR/outputs was classified as an EDM
# anchor while a test suite was running -- corpus contamination by default.
RULE_CORPUS_VENUES = frozenset({"EDM", "JEDM", "JLA"})

_VENUE_PREFIXES: dict[str, str] = {
    "jedm_": "JEDM",
    "jla_": "JLA",
    "aera_open_": "AERA_OPEN",
}


def venue_of(stem: str) -> str | None:
    """Venue from the directory prefix; None for unrecognized directories.

    EDM is inferred only from the known 2026-07-03 conference batch
    timestamp, never as a fallback: defaulting unknowns to a venue is how
    a growing outputs/ directory silently changes a frozen corpus.
    """
    for prefix, venue in _VENUE_PREFIXES.items():
        if stem.startswith(prefix):
            return venue
    if "_20260703_" in stem:
        return "EDM"
    return None


def load_corpus(corpus_path: Path, excluded: Iterable[str] = ()) -> list[Anchor]:
    """Build the deduped anchor corpus.

    A directory contributes iff it holds both ``paper.md`` and
    ``metadata.json``. Duplicates (the same paper ingested twice under
    different timestamps) are removed by sha1 of ``paper.md``.
    """
    excluded_set = set(excluded)
    anchors: list[Anchor] = []
    seen: set[str] = set()
    if not corpus_path.is_dir():
        raise FileNotFoundError(f"anchor corpus not found: {corpus_path}")
    for entry in sorted(corpus_path.iterdir()):
        if not entry.is_dir() or entry.name in excluded_set:
            continue
        paper = entry / "paper.md"
        meta = entry / "metadata.json"
        if not (paper.is_file() and meta.is_file()):
            continue
        with open(paper, encoding="utf-8", errors="replace") as handle:
            raw = handle.read()
        digest = hashlib.sha1(raw.encode("utf-8", "replace")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        with open(meta, encoding="utf-8") as handle:
            abstract = json.load(handle).get("abstract") or ""
        venue = venue_of(entry.name)
        if venue not in RULE_CORPUS_VENUES:
            # Unknown prefix or a venue outside the frozen rule corpus
            # (e.g. AERA Open calibration reviews) -- excluded, never
            # defaulted into EDM.
            continue
        anchors.append(
            Anchor(
                stem=entry.name,
                venue=venue,
                abstract=_normalize(abstract),
                full_text=_normalize(raw),
            )
        )
    return anchors


# --------------------------------------------------------------------------
# Counting
# --------------------------------------------------------------------------


def matches(text: str, patterns: Sequence[str]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def count(anchors: Sequence[Anchor], patterns: Sequence[str], field_name: str) -> dict[str, int]:
    """Per-venue and total anchor counts for a pattern set on one field."""
    tally = {venue: 0 for venue in VENUES}
    total = 0
    for anchor in anchors:
        if matches(anchor.text(field_name), patterns):
            tally[anchor.venue] = tally.get(anchor.venue, 0) + 1
            total += 1
    tally["total"] = total
    tally["n"] = len(anchors)
    return tally


# --------------------------------------------------------------------------
# Declarative predicate evaluator (reference implementation)
# --------------------------------------------------------------------------
#
# The clause language is a closed enum. Nothing in the YAML is ever
# eval'd; regex strings are data.
#
#   {kind: any_of|all_of|none_of, clauses: [...]}
#   {kind: field_regex, fields: [...], patterns: [...]}
#   {kind: task_type_in, values: [...]}
#   {kind: dataset_in,   values: [...]}


def _card_field(card: dict, spec: dict, name: str) -> str:
    for source in (card, spec):
        value = source.get(name)
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)):
            return " ".join(str(v) for v in value)
    return ""


def _resolve_task_type(card: dict, spec: dict) -> str:
    cell = card.get("cell")
    if isinstance(cell, dict) and cell.get("task_type"):
        return str(cell["task_type"])
    return str(card.get("task_type") or spec.get("task_type") or "")


def _resolve_dataset(card: dict, spec: dict) -> str:
    cell = card.get("cell")
    if isinstance(cell, dict) and cell.get("dataset"):
        return str(cell["dataset"])
    return str(card.get("dataset") or spec.get("dataset") or "")


def evaluate_predicate(clause: dict, card: dict, spec: dict | None = None) -> tuple[bool, str]:
    """Evaluate one declarative clause. Returns (fired, why).

    ``why`` is the artifact-side evidence string required by design
    commitment C2: it names the fact in THIS card that made the clause
    fire.
    """
    spec = spec or {}
    kind = str(clause.get("kind", ""))

    if kind in ("any_of", "all_of", "none_of"):
        results = [evaluate_predicate(c, card, spec) for c in clause.get("clauses") or []]
        fired_flags = [r[0] for r in results]
        reasons = [r[1] for r in results if r[0]]
        if kind == "any_of":
            return (any(fired_flags), "; ".join(reasons))
        if kind == "all_of":
            if fired_flags and all(fired_flags):
                return True, "; ".join(reasons)
            return False, ""
        return (not any(fired_flags), "no clause matched")

    if kind == "field_regex":
        fields = clause.get("fields") or []
        patterns = clause.get("patterns") or []
        for name in fields:
            text = _normalize(_card_field(card, spec, str(name)))
            if not text:
                continue
            for pattern in patterns:
                found = re.search(pattern, text)
                if found:
                    return True, f"{name} matches /{pattern}/ at {found.group(0)!r}"
        return False, ""

    if kind == "task_type_in":
        task_type = _resolve_task_type(card, spec)
        if task_type in {str(v) for v in clause.get("values") or []}:
            return True, f"task_type={task_type}"
        return False, ""

    if kind == "dataset_in":
        dataset = _resolve_dataset(card, spec)
        if dataset in {str(v) for v in clause.get("values") or []}:
            return True, f"dataset={dataset}"
        return False, ""

    raise ValueError(f"unknown predicate clause kind: {kind!r}")


def score_card(table: dict, card: dict, spec: dict | None = None, venue: str | None = None) -> dict:
    """Deterministic venue-fit score for one idea card. No LLM, no network."""
    venue_name = venue or table.get("default_venue") or "EDM"
    task_type = _resolve_task_type(card, spec or {})
    hits: list[dict] = []
    score = 0.0
    for rule in table.get("rules") or []:
        applies_to = rule.get("applies_to") or []
        if applies_to and task_type and task_type not in applies_to:
            continue
        fired, why = evaluate_predicate(rule["predicate"], card, spec or {})
        if not fired:
            continue
        delta = float((rule.get("venue_delta") or {}).get(venue_name, rule["delta"]))
        score += delta
        hits.append(
            {
                "code": rule["code"],
                "delta": delta,
                "summary": str(rule.get("summary", "")).strip(),
                "evidence": str(rule.get("evidence", "")).strip(),
                "why": why,
            }
        )
    return {"score": round(score, 4), "venue": venue_name, "hits": hits}


# --------------------------------------------------------------------------
# Policy checks
# --------------------------------------------------------------------------


BANDED_SIGNS = ("positive", "negative")


def policy_delta(sign: str, total: int) -> float | None:
    """The delta the table's own derivation_policy licenses for a count.

    ``None`` means the policy licenses no rule at that count. Signs outside
    the two banded ones (e.g. ``multiplier``) have no band and always
    return ``None``: they must be justified on other grounds, and the
    table's rejected_rules block states those grounds in prose.
    """
    if sign == "negative":
        if total == 0:
            return -1.5
        if 1 <= total <= 3:
            return -1.0
        return None
    if sign == "positive":
        if 6 <= total <= 22:
            return 0.5
        return None
    if sign in ("multiplier", "none"):
        return None
    raise ValueError(f"unknown rule sign: {sign!r}")


# --------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------


@dataclass
class Audit:
    checks: list[dict] = field(default_factory=list)

    def record(self, label: str, expected: Any, observed: Any) -> None:
        self.checks.append(
            {
                "label": label,
                "expected": expected,
                "observed": observed,
                "ok": expected == observed,
            }
        )

    @property
    def failures(self) -> list[dict]:
        return [c for c in self.checks if not c["ok"]]


def _expected_subset(declared: dict, recomputed: dict) -> tuple[dict, dict]:
    """Compare only the keys the YAML actually asserts."""
    keys = [k for k in declared if k in recomputed]
    return ({k: declared[k] for k in keys}, {k: recomputed[k] for k in keys})


def audit_table(table: dict, anchors: Sequence[Anchor]) -> Audit:
    audit = Audit()

    corpus_cfg = table.get("corpus") or {}
    audit.record("corpus.n_total", corpus_cfg.get("n_total"), len(anchors))
    by_venue = {v: sum(1 for a in anchors if a.venue == v) for v in VENUES}
    audit.record("corpus.n_by_venue", dict(corpus_cfg.get("n_by_venue") or {}), by_venue)

    probe_library = table.get("probe_library") or {}

    for rule in table.get("rules") or []:
        code = rule["code"]
        measurement = rule["measurement"]
        field_name = measurement["field"]
        patterns = measurement["patterns"]

        recomputed = count(anchors, patterns, field_name)
        declared, observed = _expected_subset(dict(measurement["observed"]), recomputed)
        audit.record(f"{code}.measurement.observed", declared, observed)

        cross = measurement.get("cross_field_check")
        if cross:
            other = count(anchors, patterns, cross["field"])
            audit.record(
                f"{code}.cross_field_check[{cross['field']}].total",
                cross["observed_total"],
                other["total"],
            )

        neighbours = measurement.get("neighbour_controls")
        if neighbours:
            for pattern, expected_n in (neighbours.get("probes") or {}).items():
                got = count(anchors, [pattern], neighbours["field"])["total"]
                audit.record(f"{code}.neighbour[{pattern}]", expected_n, got)

        contrast = measurement.get("contrast_probes")
        if contrast:
            for name, expected_n in (contrast.get("expected") or {}).items():
                probe_patterns = probe_library.get(name)
                if probe_patterns is None:
                    audit.record(f"{code}.contrast[{name}]", "in probe_library", "MISSING")
                    continue
                got = count(anchors, probe_patterns, contrast["field"])["total"]
                audit.record(f"{code}.contrast[{name}]", expected_n, got)

        licensed = policy_delta(rule["sign"], recomputed["total"])
        audit.record(f"{code}.delta_obeys_policy", licensed, float(rule["delta"]))

        for venue, venue_delta in (rule.get("venue_delta") or {}).items():
            venue_count = recomputed.get(venue, 0)
            expected_override = 1.0 if venue_count >= 6 else None
            audit.record(
                f"{code}.venue_delta[{venue}] (venue count {venue_count})",
                expected_override,
                float(venue_delta),
            )

    for rejected in table.get("rejected_rules") or []:
        name = rejected["name"]
        measurement = rejected.get("measurement") or {}
        patterns = measurement.get("patterns")
        if patterns:
            recomputed = count(anchors, patterns, measurement["field"])
            declared, observed = _expected_subset(dict(measurement["observed"]), recomputed)
            audit.record(f"rejected[{name}].observed", declared, observed)

            also_abstract = measurement.get("also_abstract")
            if also_abstract:
                other = count(anchors, patterns, "abstract")
                declared2, observed2 = _expected_subset(dict(also_abstract), other)
                audit.record(f"rejected[{name}].also_abstract", declared2, observed2)

            also_full = measurement.get("also_full_text_total")
            if also_full is not None:
                other = count(anchors, patterns, "full_text")
                audit.record(f"rejected[{name}].also_full_text_total", also_full, other["total"])

            licensed = policy_delta(rejected["considered_sign"], recomputed["total"])
            audit.record(f"rejected[{name}].is_correctly_rejected", None, licensed)

        type_patterns = measurement.get("type_patterns")
        if type_patterns:
            bundle = bundle_sizes(anchors, type_patterns, measurement["field"])
            declared, observed = _expected_subset(
                dict(measurement.get("type_counts") or {}), bundle["type_counts"]
            )
            audit.record(f"rejected[{name}].type_counts", declared, observed)
            audit.record(
                f"rejected[{name}].bundle_distribution",
                {int(k): v for k, v in (measurement.get("bundle_distribution") or {}).items()},
                bundle["distribution"],
            )
            audit.record(
                f"rejected[{name}].mean_by_venue",
                dict(measurement.get("mean_by_venue") or {}),
                bundle["mean_by_venue"],
            )
            audit.record(
                f"rejected[{name}].at_least_two",
                measurement.get("at_least_two"),
                bundle["at_least_two"],
            )

    return audit


def bundle_sizes(
    anchors: Sequence[Anchor], type_patterns: dict[str, Sequence[str]], field_name: str
) -> dict:
    """How many distinct contribution types each anchor bundles."""
    per_anchor: dict[str, int] = {}
    type_counts = {name: 0 for name in type_patterns}
    for anchor in anchors:
        text = anchor.text(field_name)
        fired = [name for name, pats in type_patterns.items() if matches(text, pats)]
        for name in fired:
            type_counts[name] += 1
        per_anchor[anchor.stem] = len(fired)
    distribution: dict[int, int] = {}
    for size in per_anchor.values():
        distribution[size] = distribution.get(size, 0) + 1
    mean_by_venue: dict[str, float] = {}
    for venue in VENUES:
        sizes = [per_anchor[a.stem] for a in anchors if a.venue == venue]
        if sizes:
            mean_by_venue[venue] = round(sum(sizes) / len(sizes), 2)
    return {
        "type_counts": type_counts,
        "distribution": dict(sorted(distribution.items())),
        "mean_by_venue": mean_by_venue,
        "at_least_two": sum(1 for s in per_anchor.values() if s >= 2),
        "per_anchor": per_anchor,
    }


def load_table(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def firing_report(table: dict, anchors: Sequence[Anchor]) -> list[dict]:
    """Per-rule firing fraction on the corpus, for the usefulness check."""
    rows = []
    for rule in table.get("rules") or []:
        measurement = rule["measurement"]
        tally = count(anchors, measurement["patterns"], measurement["field"])
        total = tally["total"]
        rows.append(
            {
                "code": rule["code"],
                "name": rule["name"],
                "sign": rule["sign"],
                "delta": rule["delta"],
                "field": measurement["field"],
                "fires_on": total,
                "n": tally["n"],
                "fraction": round(total / tally["n"], 3) if tally["n"] else 0.0,
                "degenerate": total in (0, tally["n"]),
                "detector_validated_on_anchors": bool(
                    rule.get("detector_validated_on_anchors", False)
                ),
            }
        )
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rules", default=str(DEFAULT_RULES_PATH))
    parser.add_argument("--corpus", default=None)
    parser.add_argument("--json", default=None, help="write the full report here")
    args = parser.parse_args(argv)

    table = load_table(Path(args.rules))
    corpus_path = Path(args.corpus or (table.get("corpus") or {}).get("path") or DEFAULT_CORPUS_PATH)
    anchors = load_corpus(corpus_path, (table.get("corpus") or {}).get("excluded_dirs") or ())

    print(f"corpus: {corpus_path}")
    print(f"anchors: {len(anchors)}  " + "  ".join(
        f"{v}={sum(1 for a in anchors if a.venue == v)}" for v in VENUES
    ))
    print()

    print("RULE FIRING RATES ON THE ANCHOR CORPUS")
    print("-" * 78)
    rows = firing_report(table, anchors)
    for row in rows:
        flag = "  <- DEGENERATE" if row["degenerate"] else ""
        val = "" if row["detector_validated_on_anchors"] else "  [detector unvalidated]"
        print(
            f"  {row['code']}  {row['delta']:+.1f}  {row['fires_on']:2d}/{row['n']} "
            f"({row['fraction']:.2f}) {row['field']:9s} {row['name']}{flag}{val}"
        )
    print()

    audit = audit_table(table, anchors)
    print(f"AUDIT: {len(audit.checks) - len(audit.failures)}/{len(audit.checks)} checks passed")
    print("-" * 78)
    for check in audit.failures:
        print(f"  FAIL {check['label']}")
        print(f"       asserted:   {check['expected']}")
        print(f"       recomputed: {check['observed']}")
    if not audit.failures:
        print("  every asserted anchor count reproduces; every delta obeys the policy")

    if args.json:
        payload = {
            "corpus_path": str(corpus_path),
            "n_anchors": len(anchors),
            "anchors": [{"stem": a.stem, "venue": a.venue} for a in anchors],
            "firing": rows,
            "checks": audit.checks,
        }
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nwrote {args.json}")

    return 1 if audit.failures else 0


if __name__ == "__main__":
    sys.exit(main())
