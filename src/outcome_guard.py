"""The outcome must not be among the predictors: checked right after data
preparation, not after the paid analysis.

Observed on the owner's Mac (round 2, 2026-09-27): the DataEngineer's
generated code left the outcome X4EVRATNDCLG as a column of train_X.csv
and test_X.csv. Nothing looked until pre-critic check pcc_01, after the
Analyst had spent its two calls fitting every model with the answer as an
input (AUC 1.0, and a partial-dependence plot of the outcome against
itself). pcc_01 is confirmed leakage and stops the study, so a question
the same pipeline had answered in round 1 (AUC 0.81) ended with no paper.

The post-DE pre-flight (``Orchestrator._run_post_de_preflight``) now asks
this module first, for prediction studies:

* after data preparation, an outcome column is a contract violation: the
  DataEngineer gets its one targeted retry, told which columns to drop;
* after that retry (or after a DataEngineer revision), a column that is
  still there by name is dropped from train_X.csv and test_X.csv -- those
  columns and nothing else -- and the run says so in pipeline.log, a
  warning event and data_report.warnings.

Why dropping by name is safe: the outcome itself lives in train_y.csv and
test_y.csv, which are untouched, and a column whose name is the outcome's
(or the outcome's name plus an encoded level, ``X4EVRATNDCLG_Yes``) carries
nothing but the outcome. What a name cannot show -- a proxy variable, or
other predictors imputed with the outcome in the frame -- is a different
leak; the retry instruction tells the DataEngineer to remove the outcome
before imputation, and pcc_01 still stops a study whose outcome column
reaches the analysis.

Module-level functions, like the other post-DE checks: the pre-flight is
exercised with lightweight stand-ins for the orchestrator, and
``pre_critic_checks`` reads the record without importing the orchestrator.
"""
from __future__ import annotations

import csv
import json
import os
import re
import tempfile
from typing import Any, Callable

from src import events

#: Where the guard records that it ran, in data_report. pcc_01 reads it to
#: say whether a leak it finds got past the guard or never met it.
RECORD_KEY = "post_de_outcome_check"

#: The event code of the repair.
REPAIR_CODE = "OUTCOME_REMOVED_FROM_PREDICTORS"

#: The predictor matrices the guard reads and repairs.
PREDICTOR_FILES: tuple[str, ...] = ("train_X.csv", "test_X.csv")

#: Separators between a variable's name and an encoded level:
#: ``pd.get_dummies`` and ``OneHotEncoder`` write ``X_level``,
#: ``DictVectorizer`` writes ``X=level``.
_LEVEL_SEPARATORS: tuple[str, ...] = ("_", "=")

#: train_y.csv headers that name nothing: an unnamed Series ("0"), a saved
#: index. Matching those against train_X would compare positions, not
#: variables.
_UNNAMED = re.compile(r"^(?:\d+|unnamed:\s*\d+|index|)$", re.IGNORECASE)


def _header(path: str) -> list[str] | None:
    try:
        with open(path, newline="", encoding="utf-8-sig") as fh:
            return next(csv.reader(fh), [])
    except (OSError, UnicodeDecodeError, csv.Error):
        return None


def _declared_predictors(spec: dict, outcome: str) -> list[str]:
    """Every variable the spec puts on the predictor side."""
    names: list[str] = []
    for entry in spec.get("predictor_set") or []:
        name = entry.get("variable") if isinstance(entry, dict) else entry
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    for name in spec.get("subgroup_analyses") or []:
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return [n for n in names if n.casefold() != outcome.casefold()]


def _is_level_of(column: str, name: str) -> bool:
    col = column.casefold()
    return any(
        col.startswith((name + sep).casefold()) and len(col) > len(name) + 1
        for sep in _LEVEL_SEPARATORS
    )


def _belongs_to_outcome(
    column: str, outcome: str, exact: set[str], declared: list[str]
) -> bool:
    # sklearn's ColumnTransformer prefixes its transformer's name:
    # "cat__X4EVRATNDCLG_Yes", "remainder__X4EVRATNDCLG".
    candidates = {column}
    if "__" in column:
        candidates.add(column.split("__", 1)[1])
    for cand in candidates:
        folded = cand.casefold()
        if folded in exact:
            return True
        if outcome and _is_level_of(cand, outcome):
            # A longer declared name that also fits wins: with the
            # outcome X1SES, the column X1SES_U is the predictor X1SES_U.
            if any(
                len(d) > len(outcome)
                and (folded == d.casefold() or _is_level_of(cand, d))
                for d in declared
            ):
                continue
            return True
    return False


def find_outcome_columns(
    output_dir: str, spec: dict, data_report: dict | None = None
) -> dict[str, list[str]]:
    """The outcome's columns in each predictor matrix, by file name.

    A column is the outcome's when its name is the research_spec outcome,
    the outcome the data report names, or the name train_y.csv gives its
    column; or when it is the research_spec outcome's name followed by an
    encoded level (``OUTCOME_<level>``, ``OUTCOME=<level>``) and no longer
    declared predictor name fits it. Case is ignored. Empty when there is
    nothing to compare or no file to read.
    """
    outcome = str(spec.get("outcome_variable") or "").strip()
    exact = {outcome.casefold()} if outcome else set()
    declared = _declared_predictors(spec, outcome)
    declared_folded = {d.casefold() for d in declared}

    extra: list[str] = []
    if isinstance(data_report, dict):
        extra.append(str(data_report.get("outcome_variable") or ""))
    extra.extend(_header(os.path.join(output_dir, "train_y.csv")) or [])
    for name in extra:
        name = name.strip()
        if name and not _UNNAMED.match(name) and name.casefold() not in declared_folded:
            exact.add(name.casefold())
    if not exact:
        return {}

    found: dict[str, list[str]] = {}
    for fname in PREDICTOR_FILES:
        header = _header(os.path.join(output_dir, fname))
        if not header:
            continue
        cols = [c for c in header if _belongs_to_outcome(c, outcome, exact, declared)]
        if cols:
            found[fname] = cols
    return found


def _describe(found: dict[str, list[str]], verb: str = "has") -> str:
    return "; ".join(
        f"{fname} {verb} {', '.join(cols)}" for fname, cols in found.items()
    )


def violation_message(outcome: str, found: dict[str, list[str]]) -> str:
    """What the targeted DataEngineer retry is told."""
    cols = sorted({c for cs in found.values() for c in cs})
    name = outcome or cols[0]
    return (
        f"The outcome is among the predictors (target leakage): "
        f"{_describe(found)}. Drop {', '.join(cols)} from the predictor "
        f"matrices; the outcome must be only in train_y.csv and test_y.csv. "
        f"Build the predictor frame from research_spec.predictor_set and "
        f"remove outcome_variable '{name}', and every column made from it, "
        f"BEFORE imputation, scaling and encoding -- an outcome left in the "
        f"frame the imputer sees leaks into the other predictors too. Before "
        f"writing the CSVs, assert that no column of train_X or test_X is "
        f"'{name}' or starts with '{name}_'."
    )


def drop_columns(output_dir: str, found: dict[str, list[str]]) -> dict[str, int]:
    """Remove exactly ``found``'s columns from each file; nothing else changes.

    Rewritten with the csv module rather than pandas, so every other cell
    keeps its text as the DataEngineer wrote it, and replaced in one step.
    Returns the number of columns each file has left. Raises OSError when a
    file cannot be rewritten.
    """
    left: dict[str, int] = {}
    for fname, cols in found.items():
        path = os.path.join(output_dir, fname)
        with open(path, newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.reader(fh))
        if not rows:
            continue
        names = set(cols)
        drop = {i for i, c in enumerate(rows[0]) if c in names}
        fd, tmp = tempfile.mkstemp(prefix=f".{fname}.", dir=output_dir)
        try:
            with os.fdopen(fd, "w", newline="", encoding="utf-8") as out:
                writer = csv.writer(out, lineterminator="\n")
                for row in rows:
                    writer.writerow([v for i, v in enumerate(row) if i not in drop])
            os.replace(tmp, path)
        except BaseException:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise
        left[fname] = len(rows[0]) - len(drop)
    return left


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


def guard_outcome_in_predictors(
    ctx: Any,
    *,
    repair: bool,
    after: str,
    log: Callable[[str], None],
) -> str | None:
    """Check a prediction study's predictor matrices for the outcome.

    ``repair=False`` (right after data preparation): return the violation
    message when an outcome column is there, for the targeted DataEngineer
    retry. ``repair=True`` (after that retry, or after a DataEngineer
    revision): drop those columns and continue. ``after`` names what the
    check followed, in words ("the DataEngineer's targeted retry").

    Records what it did in ``data_report[RECORD_KEY]`` (and in
    data_report.json) whenever it ran to the end. Returns a message only
    for a violation to send back, or when a repair could not be written.
    Never raises: the pre-flight must not be what breaks a healthy run.
    """
    try:
        spec = getattr(ctx, "research_spec", None) or {}
        if not isinstance(spec, dict):
            return None
        task_type = spec.get("task_type") or getattr(ctx, "task_type", None) or "prediction"
        if task_type != "prediction":
            return None
        outcome = str(spec.get("outcome_variable") or "").strip()
        output_dir = str(getattr(ctx, "output_dir", "") or "")
        report = getattr(ctx, "data_report", None)
        report = report if isinstance(report, dict) else None
        found = find_outcome_columns(output_dir, spec, report)
    except Exception as exc:  # noqa: BLE001
        log(f"Outcome-in-predictors check error (non-fatal, treated as pass): {exc}")
        return None

    if found and not repair:
        return violation_message(outcome, found)

    note = ""
    left: dict[str, int] = {}
    if found:
        try:
            left = drop_columns(output_dir, found)
        except (OSError, csv.Error, UnicodeDecodeError) as exc:
            return (
                f"{violation_message(outcome, found)} The orchestrator could "
                f"not remove them itself: {exc}"
            )
        cols = sorted({c for cs in found.values() for c in cs})
        note = (
            f"Removed the outcome from the predictors: after {after}, "
            f"{_describe(found, 'had')}. The orchestrator dropped "
            f"{', '.join(cols)} from the predictor matrices and changed "
            f"nothing else; the outcome stays in train_y.csv and test_y.csv. "
            f"If the data preparation code also used the outcome while "
            f"imputing or scaling other predictors, those columns may still "
            f"carry it, which no check of column names can see."
        )
        log(note)
        events.emit(
            ctx,
            "warning",
            stage=getattr(ctx, "current_state", None) or "ENGINEERING",
            code=REPAIR_CODE,
            message=note,
        )

    if report is None:
        return None
    updated = {**report, RECORD_KEY: {"after": after, "removed": found}}
    if note:
        warnings = report.get("warnings")
        warnings = list(warnings) if isinstance(warnings, list) else (
            [str(warnings)] if warnings else []
        )
        updated["warnings"] = warnings + [note]
        if "train_X.csv" in left:
            updated["n_predictors_encoded"] = left["train_X.csv"]
        encoding = report.get("encoding_report")
        if isinstance(encoding, dict) and outcome in encoding:
            # Its dummies are gone; an entry left behind would read as a
            # variable whose columns went missing.
            updated["encoding_report"] = {
                k: v for k, v in encoding.items() if k != outcome
            }
    ctx.data_report = updated
    try:
        if os.path.exists(os.path.join(output_dir, "data_report.json")) or note:
            _write_report(output_dir, updated)
    except OSError as exc:
        log(f"Could not rewrite data_report.json after the outcome check: {exc}")
    return None


def describe_guard(data_report: Any, task_type: str) -> str:
    """One sentence for pcc_01: did the post-DE outcome check meet this leak?"""
    record = data_report.get(RECORD_KEY) if isinstance(data_report, dict) else None
    if isinstance(record, dict):
        after = str(record.get("after") or "data preparation")
        removed = record.get("removed") or {}
        if removed:
            cols = sorted({c for cs in removed.values() for c in cs})
            did = f"removed {', '.join(cols)}"
        else:
            did = "found no outcome column"
        return (
            f"The outcome check that follows data preparation ran after "
            f"{after} and {did}, so this column was written after it ran."
        )
    if task_type != "prediction":
        return (
            f"The outcome check that follows data preparation covers "
            f"prediction studies; this {task_type} study has none."
        )
    return (
        "The outcome check that follows data preparation did not run on "
        "these files (they were written before it existed, or by a step "
        "it does not follow)."
    )
