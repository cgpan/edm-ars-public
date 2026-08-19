"""Arc T / T0 - deterministic column cache for the feasibility probes.

Two levels of caching, both keyed on (path, size, mtime) so a changed
source file invalidates automatically:

1. **Header cache** (``<dataset>.header.json``) - the CSV column list.
   Reading just the header of the 1.9 GB HSLS file costs 0.06 s, so the
   cache is a convenience rather than a necessity; it exists so that the
   Stage-0 screen can run thousands of times with zero I/O.
2. **Tier-1 frame cache** (``<dataset>.tier1.parquet``) - every curated
   Tier-1 registry column plus every item-bank item, materialised once.
   For HSLS that is ~128 of 9,614 columns: a few MB instead of 1.9 GB,
   which is what moves the Stage-1 data probes into the free tier.

Everything here is deterministic and offline. **No probe ever raises
because a data file is missing** - the accessors return ``None`` and the
caller degrades to a skipped check. That is a hard requirement: a screen
that crashes on a machine without the raw data would be worse than one
that reports "unverified".
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_CACHE_DIR = Path("cache") / "tier1"
DEFAULT_RAW_DATA_DIR = Path("data") / "raw"

# config.yaml: ideation.probe_cache.rebuild_if_older_than_days
DEFAULT_MAX_AGE_DAYS = 30

# In-process memoisation so a 24-candidate screen touches the disk once.
_HEADER_MEMO: dict[str, list[str]] = {}


@dataclass(frozen=True)
class CacheStatus:
    """What the cache knows about one dataset (for reporting/CLI)."""

    dataset: str
    raw_path: str | None
    raw_exists: bool
    header_cached: bool
    frame_cached: bool
    frame_age_days: float | None
    n_columns: int | None


def _dataset_filename(dataset: str) -> str | None:
    """Raw-data filename for a dataset, via the existing DatasetAdapter."""
    try:
        from src.dataset_adapter import create_dataset_adapter

        return create_dataset_adapter(dataset).get_raw_data_filename()
    except Exception:
        return None


def raw_data_path(
    dataset: str,
    raw_data_dir: str | os.PathLike[str] | None = None,
) -> Path | None:
    """Absolute path to the dataset's raw CSV, or ``None`` if unknown."""
    filename = _dataset_filename(dataset)
    if filename is None:
        return None
    base = Path(raw_data_dir) if raw_data_dir is not None else DEFAULT_RAW_DATA_DIR
    return (base / filename).resolve()


def _fingerprint(path: Path) -> dict[str, Any] | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return {"path": str(path), "size": st.st_size, "mtime": int(st.st_mtime)}


def _read_json(path: Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, sort_keys=True)


def _read_header(path: Path) -> list[str] | None:
    """Read only the CSV header. Never raises."""
    for encoding in ("utf-8", "latin-1"):
        try:
            frame = pd.read_csv(path, nrows=0, encoding=encoding, low_memory=False)
            return [str(c) for c in frame.columns]
        except UnicodeDecodeError:
            continue
        except Exception:
            return None
    return None


def header_columns(
    dataset: str,
    *,
    raw_data_dir: str | os.PathLike[str] | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    refresh: bool = False,
) -> list[str] | None:
    """Column names of the dataset's raw CSV, or ``None`` when unavailable.

    ``None`` means "cannot establish" - the raw file is absent or
    unreadable. Callers must treat that as *unknown*, never as *absent
    column* (see the KILL discipline in ``feasibility``).
    """
    path = raw_data_path(dataset, raw_data_dir)
    if path is None:
        return None

    memo_key = f"{dataset}|{path}"
    if not refresh and memo_key in _HEADER_MEMO:
        return list(_HEADER_MEMO[memo_key])

    fp = _fingerprint(path)
    if fp is None:
        return None

    cache_root = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cache_file = cache_root / f"{dataset}.header.json"

    if not refresh:
        cached = _read_json(cache_file)
        if cached and all(cached.get(k) == fp[k] for k in ("path", "size", "mtime")):
            columns = [str(c) for c in cached.get("columns", [])]
            _HEADER_MEMO[memo_key] = columns
            return list(columns)

    read = _read_header(path)
    if read is None:
        return None
    columns = read

    try:
        _write_json(cache_file, {**fp, "columns": columns})
    except OSError:
        pass  # a read-only cache dir must not break the screen
    _HEADER_MEMO[memo_key] = columns
    return list(columns)


def tier1_columns(registry: dict) -> list[str]:
    """Every curated Tier-1 variable name plus every item-bank item.

    Order is deterministic: outcomes, then predictors by category in
    registry order, then item-bank items.
    """
    names: list[str] = []
    seen: set[str] = set()

    def _add(name: object) -> None:
        if isinstance(name, str) and name and name not in seen:
            seen.add(name)
            names.append(name)

    variables = (registry or {}).get("variables") or {}
    for outcome in variables.get("outcomes") or []:
        if isinstance(outcome, dict):
            _add(outcome.get("name"))
    predictors = variables.get("predictors") or {}
    if isinstance(predictors, dict):
        for var_list in predictors.values():
            for var in var_list or []:
                if isinstance(var, dict):
                    _add(var.get("name"))

    for bank in ((registry or {}).get("item_banks") or {}).values():
        if isinstance(bank, dict):
            for item in bank.get("items") or []:
                _add(item)

    return names


def _frame_paths(dataset: str, cache_dir: Path) -> tuple[Path, Path, Path]:
    return (
        cache_dir / f"{dataset}.tier1.parquet",
        cache_dir / f"{dataset}.tier1.csv.gz",
        cache_dir / f"{dataset}.tier1.meta.json",
    )


def build_tier1_cache(
    dataset: str,
    registry: dict,
    *,
    raw_data_dir: str | os.PathLike[str] | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
) -> Path | None:
    """Materialise the Tier-1 column slice. Returns the cache file or None.

    Parquet when pyarrow is importable, gzipped CSV otherwise - the
    fallback keeps the cache working on an install without pyarrow
    (it is not in ``requirements.txt``).
    """
    path = raw_data_path(dataset, raw_data_dir)
    if path is None or not path.exists():
        return None
    header = header_columns(dataset, raw_data_dir=raw_data_dir, cache_dir=cache_dir)
    if header is None:
        return None

    wanted = [c for c in tier1_columns(registry) if c in set(header)]
    if not wanted:
        return None

    try:
        frame = pd.read_csv(path, usecols=wanted, low_memory=False)
    except Exception:
        try:
            frame = pd.read_csv(
                path, usecols=wanted, low_memory=False, encoding="latin-1"
            )
        except Exception:
            return None

    cache_root = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cache_root.mkdir(parents=True, exist_ok=True)
    parquet_path, csv_path, meta_path = _frame_paths(dataset, cache_root)

    written: Path
    try:
        frame.to_parquet(parquet_path, index=False)
        written = parquet_path
    except Exception:
        frame.to_csv(csv_path, index=False, encoding="utf-8")
        written = csv_path

    fp = _fingerprint(path) or {}
    _write_json(
        meta_path,
        {
            **fp,
            "built_at": time.time(),
            "columns": list(frame.columns.astype(str)),
            "n_rows": int(len(frame)),
            "cache_file": written.name,
        },
    )
    return written


def _cache_is_fresh(meta: dict | None, path: Path, max_age_days: float) -> bool:
    if not meta:
        return False
    fp = _fingerprint(path)
    if fp is None:
        return False
    if any(meta.get(k) != fp[k] for k in ("path", "size", "mtime")):
        return False
    built_at = meta.get("built_at")
    if not isinstance(built_at, (int, float)):
        return False
    age_days = (time.time() - float(built_at)) / 86400.0
    return age_days <= max_age_days


def tier1_frame(
    dataset: str,
    registry: dict,
    *,
    columns: list[str] | None = None,
    raw_data_dir: str | os.PathLike[str] | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    max_age_days: float = DEFAULT_MAX_AGE_DAYS,
    allow_build: bool = True,
) -> pd.DataFrame | None:
    """Return the cached Tier-1 slice, building it if needed.

    ``None`` when the raw file is absent (the probes then SKIP). Never
    raises. ``columns`` sub-selects; unknown names are ignored rather
    than raising, because a probe asking for a column the cache does not
    carry is a degradation, not an error.
    """
    path = raw_data_path(dataset, raw_data_dir)
    if path is None or not path.exists():
        return None

    cache_root = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    parquet_path, csv_path, meta_path = _frame_paths(dataset, cache_root)
    meta = _read_json(meta_path)

    cache_file: Path | None = None
    if _cache_is_fresh(meta, path, max_age_days):
        candidate = cache_root / str((meta or {}).get("cache_file", ""))
        if candidate.exists():
            cache_file = candidate
    if cache_file is None:
        if not allow_build:
            return None
        cache_file = build_tier1_cache(
            dataset, registry, raw_data_dir=raw_data_dir, cache_dir=cache_dir
        )
    if cache_file is None:
        return None

    try:
        if cache_file.suffix == ".parquet":
            frame = pd.read_parquet(cache_file)
        else:
            frame = pd.read_csv(cache_file, low_memory=False)
    except Exception:
        return None

    if columns is not None:
        keep = [c for c in columns if c in frame.columns]
        if not keep:
            return frame.iloc[:, :0]
        frame = frame[keep]
    return frame


def cache_status(
    dataset: str,
    *,
    raw_data_dir: str | os.PathLike[str] | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
) -> CacheStatus:
    """Introspection for the CLI / audit script. Never raises."""
    path = raw_data_path(dataset, raw_data_dir)
    cache_root = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    parquet_path, csv_path, meta_path = _frame_paths(dataset, cache_root)
    meta = _read_json(meta_path)
    built_at = (meta or {}).get("built_at")
    age = (
        (time.time() - float(built_at)) / 86400.0
        if isinstance(built_at, (int, float))
        else None
    )
    return CacheStatus(
        dataset=dataset,
        raw_path=str(path) if path else None,
        raw_exists=bool(path and path.exists()),
        header_cached=(cache_root / f"{dataset}.header.json").exists(),
        frame_cached=parquet_path.exists() or csv_path.exists(),
        frame_age_days=age,
        n_columns=len((meta or {}).get("columns") or []) or None,
    )
