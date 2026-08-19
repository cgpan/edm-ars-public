"""Compute a regression fingerprint for a pipeline run.

Captures SHA256 + structural summary for the canonical artifacts of a
Phase 2c regression run, plus paper line/word counts. Intended to be
invoked after each pipeline run (baseline + four checkpoints) so that
fingerprints can be diffed across the refactor.

Usage:
    python scripts/fingerprint_run.py <run_dir> > <run_dir>/fingerprint.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ARTIFACTS = [
    "research_spec.json",
    "literature_context.json",
    "data_report.json",
    "results.json",
    "review_report.json",
    "paper.tex",
    "references.bib",
    "model_comparison.csv",
    "feature_importance.csv",
    "subgroup_performance.csv",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _summarize_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}

    def _shape(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {
                k: _shape(v) if isinstance(v, (dict, list)) else type(v).__name__
                for k, v in obj.items()
            }
        if isinstance(obj, list):
            return {"_list_len": len(obj), "_item_shape": _shape(obj[0]) if obj else None}
        return type(obj).__name__

    return {"keys_top": sorted(data.keys()) if isinstance(data, dict) else None,
            "shape": _shape(data)}


def _csv_summary(path: Path) -> dict[str, Any]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        header = lines[0].split(",") if lines else []
        return {"n_rows": max(0, len(lines) - 1), "n_cols": len(header), "header": header}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def _tex_summary(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}
    lines = text.splitlines()
    words = text.split()
    return {
        "n_lines": len(lines),
        "n_words": len(words),
        "n_chars": len(text),
        "n_cite": text.count(r"\cite{"),
        "n_figure": text.count(r"\begin{figure}"),
        "n_table": text.count(r"\begin{table}"),
    }


def _bib_summary(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}
    return {
        "n_chars": len(text),
        "n_entries": text.count("@"),
    }


def fingerprint(run_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "run_dir": str(run_dir),
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": {},
    }
    for name in _ARTIFACTS:
        path = run_dir / name
        if not path.exists():
            out["artifacts"][name] = {"present": False}
            continue
        entry: dict[str, Any] = {
            "present": True,
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
        if name.endswith(".json"):
            entry["json_summary"] = _summarize_json(path)
        elif name.endswith(".csv"):
            entry["csv_summary"] = _csv_summary(path)
        elif name == "paper.tex":
            entry["tex_summary"] = _tex_summary(path)
        elif name == "references.bib":
            entry["bib_summary"] = _bib_summary(path)
        out["artifacts"][name] = entry
    return out


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: fingerprint_run.py <run_dir>", file=sys.stderr)
        sys.exit(2)
    run_dir = Path(sys.argv[1])
    if not run_dir.is_dir():
        print(f"not a directory: {run_dir}", file=sys.stderr)
        sys.exit(1)
    fp = fingerprint(run_dir)
    json.dump(fp, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
