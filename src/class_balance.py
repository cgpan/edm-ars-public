"""The outcome's class split, counted from the files the analysis reads.

Observed on the owner's Mac (round 3, 2026-09-27): the paper's Methods
said "10,454 students (60.3%) had enrolled in college by February 2016,
and 3,319 (19.1%) had not; the remaining students were excluded due to
missing outcome data". 10,454 + 3,319 = 13,773 is the TRAINING split, not
the analytic sample of 17,335. The percentages divide training counts by
the analytic n, and the Writer invented a reason for the missing 20%.

The source was ``data_report.class_balance``. The DataEngineer's generated
code computes it, and it computed it on ``y_train``; the field sits in a
report about the analytic sample and does not say which sample it counts.
The same shape was measured on two archived JEDM papers (INV_CLASS_BALANCE
_WRONG_SAMPLE).

So after data preparation the orchestrator counts the classes itself,
from ``train_y.csv`` and ``test_y.csv``, and writes three labelled fields::

    "class_balance":       analytic sample (train + test) -- the SPEC field
    "class_balance_train": the training split
    "class_balance_test":  the test split

each ``{"sample", "n", "counts", "shares"}``. When the DataEngineer's own
value says something else it is kept as ``class_balance_reported_by_de``
and ``warnings`` says what was corrected, so the Critic, the OutlineAgent
and the Writer (which all read data_report) see the correction and not
only the corrected number.

Deterministic, no LLM, and never raises: a check after data preparation
must not be what breaks a healthy run.
"""
from __future__ import annotations

import csv
import json
import os
import tempfile
from collections import Counter
from typing import Any, Callable, Optional

#: Outcome types with classes to count. Anything else (continuous, count,
#: an empty string) is left alone unless the DataEngineer itself reported
#: a class balance.
CATEGORICAL_OUTCOME_TYPES = frozenset(
    {"binary", "categorical", "multiclass", "ordinal", "classification"}
)

#: More distinct values than this is not a class variable, whatever the
#: report says; counting it would print a table of scores.
MAX_CLASSES = 20

#: Header names an index column gets when a frame is written with its
#: index (``df.to_csv(path)``), never the outcome.
_INDEX_HEADERS = frozenset({"", "unnamed: 0", "index"})

SAMPLE_LABELS = {
    "class_balance": "analytic sample (train + test)",
    "class_balance_train": "training split",
    "class_balance_test": "test split",
}

#: Where the DataEngineer's own value is kept when it differed.
REPORTED_KEY = "class_balance_reported_by_de"


def _read_outcome_column(path: str, outcome: str) -> Optional[list[str]]:
    """The outcome values in a y CSV, or None when the file cannot say.

    One column is the outcome. With an index column beside it, the other
    column is. With several named columns, only the one named like the
    outcome counts; otherwise the file is ambiguous and nothing is read.
    """
    try:
        with open(path, newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.reader(fh))
    except (OSError, UnicodeDecodeError, csv.Error):
        return None
    if len(rows) < 2:
        return None
    header = [h.strip() for h in rows[0]]
    candidates = [
        i for i, h in enumerate(header) if h.lower() not in _INDEX_HEADERS
    ]
    if len(candidates) != 1:
        named = [i for i, h in enumerate(header) if outcome and h == outcome]
        if len(named) != 1:
            return None
        candidates = named
    col = candidates[0]
    return [r[col].strip() for r in rows[1:] if len(r) > col]


def _label(value: str) -> str:
    """``"1.0"`` and ``"1"`` are the same class; ``"Yes"`` stays ``"Yes"``."""
    text = str(value).strip()
    if text.lower().startswith("class_"):
        text = text[len("class_"):]
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return str(int(number))
    return repr(number)


def _order(label: str) -> tuple:
    try:
        return (0, float(label), "")
    except ValueError:
        return (1, 0.0, label)


def _summary(counts: Counter, field: str) -> dict:
    n = sum(counts.values())
    labels = sorted(counts, key=_order)
    return {
        "sample": SAMPLE_LABELS[field],
        "n": n,
        "counts": {f"class_{k}": int(counts[k]) for k in labels},
        "shares": {
            f"class_{k}": round(counts[k] / n, 4) if n else 0.0 for k in labels
        },
    }


def _decimals(value: float) -> int:
    text = repr(float(value))
    if "e" in text or "." not in text:
        return 0
    return min(6, len(text.split(".", 1)[1]))


def _as_label_map(value: Any) -> Optional[dict[str, float]]:
    """A reported class balance as ``{label: number}``, or None."""
    if isinstance(value, dict) and isinstance(value.get("counts"), dict):
        value = value["counts"]
    if not isinstance(value, dict) or not value:
        return None
    out: dict[str, float] = {}
    for key, raw in value.items():
        if isinstance(raw, bool):
            return None
        try:
            out[_label(key)] = float(raw)
        except (TypeError, ValueError):
            return None
    return out


def _matches(reported: dict[str, float], summary: dict) -> bool:
    """Does a reported ``{label: number}`` say what *summary* says?

    Counts must be equal. Shares (every value in [0, 1], summing to about
    1) must agree to the precision the report used: 0.75 agrees with
    0.7530, 0.759 does not. Percentages (summing to about 100) likewise.
    """
    counts = {_label(k): v for k, v in summary["counts"].items()}
    shares = {_label(k): v for k, v in summary["shares"].items()}
    if set(reported) != set(counts):
        return False
    values = list(reported.values())
    for scale in (1.0, 100.0):
        if all(0.0 <= v <= scale for v in values) and abs(sum(values) - scale) < 0.02 * scale:
            return all(
                abs(reported[k] - scale * shares[k])
                <= 0.5 * 10 ** -_decimals(reported[k]) + scale * 5e-5 + 1e-9
                for k in reported
            )
    return all(abs(reported[k] - counts[k]) < 0.5 for k in reported)


def _fmt_counts(summary: dict) -> str:
    return ", ".join(
        f"{k} = {c:,} ({100 * summary['shares'][k]:.1f}%)"
        for k, c in summary["counts"].items()
    )


def _write_report(output_dir: str, report: dict) -> None:
    path = os.path.join(output_dir, "data_report.json")
    fd, tmp = tempfile.mkstemp(prefix=".data_report.", dir=output_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            json.dump(report, out, indent=2)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def correct_class_balance(ctx: Any, log: Callable[[str], None]) -> Optional[str]:
    """Recount the outcome's classes from the y files into data_report.

    Updates ``ctx.data_report`` and data_report.json. Returns the warning
    it added when the DataEngineer's value was corrected, else None.
    Does nothing for a continuous outcome, a study without both y files
    (causal_did panels, psychometrics), or a file it cannot read cleanly.
    Never raises.
    """
    try:
        return _correct(ctx, log)
    except Exception as exc:  # noqa: BLE001
        log(f"Class balance recount skipped (non-fatal): {exc}")
        return None


def _correct(ctx: Any, log: Callable[[str], None]) -> Optional[str]:
    report = getattr(ctx, "data_report", None)
    if not isinstance(report, dict):
        return None
    output_dir = str(getattr(ctx, "output_dir", "") or "")
    outcome_type = str(report.get("outcome_type") or "").strip().lower()
    reported = report.get("class_balance")
    if outcome_type not in CATEGORICAL_OUTCOME_TYPES and not (
        not outcome_type and isinstance(reported, dict) and reported
    ):
        return None

    spec = getattr(ctx, "research_spec", None) or {}
    outcome = str(
        report.get("outcome_variable")
        or (spec.get("outcome_variable") if isinstance(spec, dict) else "")
        or ""
    )
    columns = {}
    for split in ("train", "test"):
        path = os.path.join(output_dir, f"{split}_y.csv")
        if not os.path.exists(path):
            return None
        values = _read_outcome_column(path, outcome)
        if values is None:
            log(
                f"Class balance recount skipped: {split}_y.csv does not have "
                "one outcome column"
            )
            return None
        columns[split] = Counter(_label(v) for v in values if v != "")

    total = columns["train"] + columns["test"]
    if not total or len(total) > MAX_CLASSES:
        return None

    analytic = _summary(total, "class_balance")
    train = _summary(columns["train"], "class_balance_train")
    test = _summary(columns["test"], "class_balance_test")

    updated = dict(report)
    note = None
    prior = _as_label_map(reported)
    if prior is not None and not _matches(prior, analytic):
        if report.get(REPORTED_KEY) is None:
            updated[REPORTED_KEY] = reported
        if _matches(prior, train):
            what = f"which are the training split's (n_train = {train['n']:,})"
        elif _matches(prior, test):
            what = f"which are the test split's (n_test = {test['n']:,})"
        else:
            what = "which match neither the analytic sample nor either split"
        note = (
            "class_balance was recounted from train_y.csv and test_y.csv: "
            f"the analytic sample of {analytic['n']:,} students has "
            f"{_fmt_counts(analytic)}. The data preparation code reported "
            f"{json.dumps(reported)}, {what}; that value is kept as "
            f"{REPORTED_KEY}. class_balance is the analytic sample; "
            "class_balance_train and class_balance_test are the splits."
        )
    updated["class_balance"] = analytic
    updated["class_balance_train"] = train
    updated["class_balance_test"] = test
    if note:
        warnings = report.get("warnings")
        warnings = list(warnings) if isinstance(warnings, list) else (
            [str(warnings)] if warnings else []
        )
        if note not in warnings:
            warnings.append(note)
        updated["warnings"] = warnings
        log(note)

    ctx.data_report = updated
    try:
        if os.path.isdir(output_dir):
            _write_report(output_dir, updated)
    except OSError as exc:
        log(f"Could not rewrite data_report.json after the class balance recount: {exc}")
    return note
