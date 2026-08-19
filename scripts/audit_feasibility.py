"""Arc T / T0 validation V1 - replay the feasibility screen over the archive.

Two populations, one blocking question each:

1. **Real archived specs** (``runs/*/output/research_spec.json``, the 26
   canonical ones, plus the aborted-attempt specs under
   ``output_attempt*`` reported separately). Question: *does the screen
   kill anything that actually shipped?* Target FALSE-KILL RATE = 0%.
   A false KILL removes a legitimate research question with no human in
   the loop, so any false kill is a release blocker (spec sec. 6 V1) and
   this script exits non-zero.

2. **Mutants** - one deliberately broken spec per KILL/WARN code, each
   derived from a real archived spec by a **single documented
   mutation**. Question: *does the intended code actually fire?* Target
   kill rate = 100%, because every mutation is logically dispositive by
   construction.

No LLM, no network. Stage-1 probes are off unless ``--probes`` is given
(they need the raw data files).

Usage:
    python scripts/audit_feasibility.py
    python scripts/audit_feasibility.py --json audit.json --probes
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ideation.feasibility import KILL, WARN, FeasibilityReport, screen  # noqa: E402


# --------------------------------------------------------------------------
# Archive loading
# --------------------------------------------------------------------------


@dataclass
class ArchivedSpec:
    run: str
    path: Path
    spec: dict
    dataset: str | None
    task_type: str | None
    canonical: bool

    @property
    def label(self) -> str:
        return f"{self.run}/{self.path.parent.name}"


def _resolve_context(path: Path, spec: dict) -> tuple[str | None, str | None]:
    """Dataset + task type for a spec.

    6 of the 26 archived prediction specs declare neither field; the run
    checkpoint carries both, so it is the fallback.
    """
    dataset = spec.get("dataset")
    task_type = spec.get("task_type")
    checkpoint = path.parent / "checkpoint.json"
    if (not dataset or not task_type) and checkpoint.exists():
        try:
            with open(checkpoint, encoding="utf-8") as f:
                data = json.load(f)
            dataset = dataset or data.get("dataset_name")
            task_type = task_type or data.get("task_type")
        except (OSError, ValueError):
            pass
    return dataset, task_type


def load_archive(runs_dir: Path) -> list[ArchivedSpec]:
    specs: list[ArchivedSpec] = []
    for path in sorted(runs_dir.glob("*/output*/research_spec.json")):
        try:
            with open(path, encoding="utf-8") as f:
                spec = json.load(f)
        except (OSError, ValueError):
            continue
        if not isinstance(spec, dict):
            continue
        dataset, task_type = _resolve_context(path, spec)
        specs.append(
            ArchivedSpec(
                run=path.parent.parent.name,
                path=path,
                spec=spec,
                dataset=dataset,
                task_type=task_type,
                canonical=(path.parent.name == "output"),
            )
        )
    return specs


def _find(
    archive: list[ArchivedSpec], predicate: Callable[[ArchivedSpec], bool]
) -> ArchivedSpec | None:
    for item in archive:
        if item.canonical and predicate(item):
            return item
    return None


# --------------------------------------------------------------------------
# Mutants - one per code, each a single documented mutation of a real spec
# --------------------------------------------------------------------------


@dataclass
class Mutant:
    code: str  # the code this mutant is designed to trigger
    base: str  # which archived spec it was derived from
    mutation: str  # the one thing that was changed
    spec: dict
    dataset: str | None
    task_type: str | None
    needs_raw_data: bool = False
    note: str = ""


def _by_dataset_task(
    archive: list[ArchivedSpec], dataset: str, task_type: str
) -> ArchivedSpec | None:
    return _find(
        archive,
        lambda s: s.dataset == dataset and (s.task_type or "prediction") == task_type,
    )


def build_mutants(archive: list[ArchivedSpec]) -> list[Mutant]:
    """One mutant per KILL/WARN code. Skips codes with no suitable base."""
    mutants: list[Mutant] = []

    def add(
        code: str,
        base: ArchivedSpec | None,
        mutation: str,
        mutate: Callable[[dict], None],
        *,
        task_type: str | None = None,
        needs_raw_data: bool = False,
        note: str = "",
    ) -> None:
        if base is None:
            return
        spec = copy.deepcopy(base.spec)
        mutate(spec)
        mutants.append(
            Mutant(
                code=code,
                base=base.label,
                mutation=mutation,
                spec=spec,
                dataset=base.dataset,
                task_type=task_type or base.task_type,
                needs_raw_data=needs_raw_data,
                note=note,
            )
        )

    hsls_soo = _by_dataset_task(archive, "hsls09_public", "causal_soo")
    hsls_pred = _by_dataset_task(archive, "hsls09_public", "prediction")
    els_pred = _by_dataset_task(archive, "els_2002", "prediction")
    hsls_psy = _by_dataset_task(archive, "hsls09_public", "psychometrics")
    assist_psy = _by_dataset_task(archive, "assistments_0910", "psychometrics")

    # --- KILL codes ------------------------------------------------------
    add(
        "F-TASK-INCOMPATIBLE", hsls_soo,
        "task_type causal_soo -> causal_did on hsls09_public (no in-file "
        "post/timing variable)",
        lambda s: s.update(task_type="causal_did"),
        task_type="causal_did",
    )
    add(
        "F-VAR-ABSENT", hsls_soo,
        "adjustment_set[0] renamed to the invented-but-plausible "
        "X1MTHCONFIDENCE",
        lambda s: s["adjustment_set"].__setitem__(0, "X1MTHCONFIDENCE"),
        needs_raw_data=True,
        note=(
            "co-fires F-COL-ABSENT: an invented name is absent from both "
            "registry and CSV"
        ),
    )
    add(
        "F-COL-ABSENT", hsls_soo,
        "adjustment_set[0] renamed to X1SES_TOTALLY_MADE_UP",
        lambda s: s["adjustment_set"].__setitem__(0, "X1SES_TOTALLY_MADE_UP"),
        needs_raw_data=True,
        note=(
            "no registry-only name exists outside `derived: true`, so this "
            "mutant necessarily co-fires F-VAR-ABSENT"
        ),
    )
    add(
        "F-TEMPORAL-ORDER", hsls_pred,
        "predictor_set gains X4EVERDROP (update_panel), the same wave as "
        "the X4 outcome",
        lambda s: s["predictor_set"].append(
            {
                "variable": "X4EVERDROP",
                "rationale": "MUTANT: same-wave predictor",
                "wave": "base_year",  # the lie the registry lookup catches
            }
        ),
    )
    add(
        "F-TIER3-EXCLUDED", hsls_soo,
        "adjustment_set gains W1STUDENT (a sampling weight)",
        lambda s: s["adjustment_set"].append("W1STUDENT"),
        note="co-fires F-VAR-ABSENT/F-COL-ABSENT if the weight is not a column",
    )
    add(
        "F-DEAD-VARIABLE", hsls_soo,
        "adjustment_set gains X1FREELUNCH (pct_missing 100, suppressed)",
        lambda s: s["adjustment_set"].append("X1FREELUNCH"),
    )
    add(
        "F-ESTIMATOR-UNCERTIFIED", hsls_soo,
        "primary_method M2 -> RD (synthetic-certified but shelved)",
        lambda s: s.update(primary_method="RD"),
    )
    add(
        "F-DESIGN-INFEASIBLE", assist_psy,
        "task_type psychometrics -> causal_itr on assistments_0910 "
        "(itr_ready: false)",
        lambda s: s.update(task_type="causal_itr"),
        task_type="causal_itr",
        note=(
            "depends on the parallel itr_feasible fix reading "
            "design_feasibility.itr_ready; co-fires F-TASK-INCOMPATIBLE"
        ),
    )
    add(
        "F-SPEC-INCOMPLETE", hsls_soo,
        "the treatment block is deleted",
        lambda s: s.pop("treatment", None),
    )
    add(
        "F-NO-PROTECTED-ATTRS", assist_psy,
        "subgroup_analyses set to ['skill_name'] on a dataset with zero "
        "protected attributes",
        lambda s: s.update(subgroup_analyses=["skill_name"]),
    )
    add(
        "F-ITEM-BANK-TOO-FEW", hsls_psy,
        "factor_model replaced with the 2-item math_identity bank while "
        "the battery still requests P3",
        lambda s: s.update(
            factor_model="ID =~ S1MPERSON1 + S1MPERSON2",
            item_columns=["S1MPERSON1", "S1MPERSON2"],
        ),
    )

    # --- WARN codes ------------------------------------------------------
    add(
        "F-SUBGROUP-VAR-UNKNOWN", hsls_pred,
        "subgroup_analyses gains X1GENDERIDENTITY (not a variable in HSLS)",
        lambda s: s.setdefault("subgroup_analyses", []).append("X1GENDERIDENTITY"),
    )
    add(
        "F-METADATA-UNVERIFIED", els_pred,
        "predictor_set gains BYPARASP, a real ELS column that is only in "
        "the Tier-2 auto profile",
        lambda s: s["predictor_set"].append(
            {
                "variable": "BYPARASP",
                "rationale": "MUTANT: uncurated Tier-2 variable",
                "wave": "base_year",
            }
        ),
    )
    add(
        "F-PITFALL-TOUCHED", hsls_pred,
        "subgroup_analyses deleted while protected attributes stay in the "
        "predictor set (registry pitfall protected_attribute_misuse)",
        lambda s: s.pop("subgroup_analyses", None),
    )
    return mutants


# --------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------


def run_audit(
    runs_dir: Path,
    *,
    run_probes: bool = False,
    registry_dir: str | None = None,
    raw_data_dir: str | None = None,
) -> dict[str, Any]:
    archive = load_archive(runs_dir)
    canonical = [a for a in archive if a.canonical]
    attempts = [a for a in archive if not a.canonical]

    kw: dict[str, Any] = {
        "registry_dir": registry_dir,
        "raw_data_dir": raw_data_dir,
        "run_probes": run_probes,
    }

    real_rows: list[dict] = []
    for item in archive:
        report: FeasibilityReport = screen(
            item.spec,
            candidate_id=item.label,
            dataset=item.dataset,
            task_type=item.task_type,
            **kw,
        )
        real_rows.append(
            {
                "label": item.label,
                "canonical": item.canonical,
                "dataset": item.dataset,
                "task_type": item.task_type or "prediction",
                "verdict": report.verdict,
                "kills": report.kill_codes,
                "warns": report.warn_codes,
                "penalty": round(report.penalty, 3),
                "analytic_n": report.analytic_n_estimate,
            }
        )

    mutants = build_mutants(archive)
    mutant_rows: list[dict] = []
    for mutant in mutants:
        report = screen(
            mutant.spec,
            candidate_id=f"MUTANT[{mutant.code}]",
            dataset=mutant.dataset,
            task_type=mutant.task_type,
            **kw,
        )
        fired = mutant.code in report.kill_codes or mutant.code in report.warn_codes
        expected_status = KILL if mutant.code in _KILL_CODES else WARN
        actual = (
            KILL if mutant.code in report.kill_codes
            else WARN if mutant.code in report.warn_codes
            else "NONE"
        )
        mutant_rows.append(
            {
                "code": mutant.code,
                "base": mutant.base,
                "mutation": mutant.mutation,
                "expected_status": expected_status,
                "actual_status": actual,
                "caught": fired and actual == expected_status,
                "verdict": report.verdict,
                "all_kills": report.kill_codes,
                "all_warns": report.warn_codes,
                "needs_raw_data": mutant.needs_raw_data,
                "note": mutant.note,
            }
        )

    false_kills = [r for r in real_rows if r["canonical"] and r["verdict"] == KILL]
    attempt_kills = [r for r in real_rows if not r["canonical"] and r["verdict"] == KILL]
    caught = [m for m in mutant_rows if m["caught"]]

    warn_histogram: dict[str, int] = {}
    for row in real_rows:
        if not row["canonical"]:
            continue
        for code in row["warns"]:
            warn_histogram[code] = warn_histogram.get(code, 0) + 1

    return {
        "n_canonical": len(canonical),
        "n_attempt_specs": len(attempts),
        "real": real_rows,
        "mutants": mutant_rows,
        "false_kill_rate": (len(false_kills) / len(canonical)) if canonical else 0.0,
        "false_kills": false_kills,
        "attempt_spec_kills": attempt_kills,
        "mutant_kill_rate": (len(caught) / len(mutant_rows)) if mutant_rows else 0.0,
        "mutants_missed": [m for m in mutant_rows if not m["caught"]],
        "warn_histogram": warn_histogram,
        "probes_run": run_probes,
    }


_KILL_CODES = {
    "F-TASK-INCOMPATIBLE",
    "F-VAR-ABSENT",
    "F-COL-ABSENT",
    "F-TEMPORAL-ORDER",
    "F-TIER3-EXCLUDED",
    "F-DEAD-VARIABLE",
    "F-ESTIMATOR-UNCERTIFIED",
    "F-DESIGN-INFEASIBLE",
    "F-SPEC-INCOMPLETE",
    "F-NO-PROTECTED-ATTRS",
    "F-ITEM-BANK-TOO-FEW",
}


def render(result: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("Arc T / T0 V1 audit - deterministic feasibility screen")
    lines.append("=" * 78)
    lines.append("")
    lines.append(
        f"Real archived specs: {result['n_canonical']} canonical "
        f"(runs/*/output/), {result['n_attempt_specs']} aborted-attempt "
        f"(runs/*/output_attempt*/)"
    )
    lines.append(
        f"FALSE-KILL RATE on canonical specs: "
        f"{result['false_kill_rate']:.1%}  (target 0%)"
    )
    for row in result["false_kills"]:
        lines.append(f"   !! {row['label']} killed by {row['kills']}")
    if result["attempt_spec_kills"]:
        lines.append(
            f"   ({len(result['attempt_spec_kills'])} aborted-attempt spec(s) "
            f"killed - reported, not counted: those runs did abort)"
        )
        for row in result["attempt_spec_kills"]:
            lines.append(f"      - {row['label']}: {row['kills']}")
    lines.append("")
    lines.append(
        f"MUTANT CATCH RATE: {result['mutant_kill_rate']:.1%} of "
        f"{len(result['mutants'])} mutants (target 100%)"
    )
    for row in result["mutants"]:
        mark = "OK " if row["caught"] else "MISS"
        lines.append(
            f"   [{mark}] {row['code']:24} expected={row['expected_status']:4} "
            f"actual={row['actual_status']:4}  base={row['base']}"
        )
        lines.append(f"          mutation: {row['mutation']}")
        if row["all_kills"] and row["all_kills"] != [row["code"]]:
            lines.append(f"          also fired KILL: {row['all_kills']}")
        if row["note"]:
            lines.append(f"          note: {row['note']}")
    lines.append("")
    lines.append("Defects the screen would now catch pre-spend, on specs that")
    lines.append("actually ran (WARN histogram over the canonical archive):")
    if result["warn_histogram"]:
        for code, count in sorted(
            result["warn_histogram"].items(), key=lambda kv: -kv[1]
        ):
            lines.append(f"   {code:26} {count:3} of {result['n_canonical']} specs")
    else:
        lines.append("   (none)")
    if not result["probes_run"]:
        lines.append("")
        lines.append(
            "Stage-1 data probes were NOT run (pass --probes; they need the "
            "raw data files)."
        )
    lines.append("")
    verdict = (
        "PASS" if not result["false_kills"] and not result["mutants_missed"]
        else "FAIL"
    )
    lines.append(f"V1 GATE: {verdict}")
    if result["false_kills"]:
        lines.append("  blocking: a real archived spec was killed (spec sec. 6 V1)")
    if result["mutants_missed"]:
        lines.append(
            "  non-blocking-but-reported: "
            + ", ".join(m["code"] for m in result["mutants_missed"])
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-dir", default=str(REPO_ROOT / "runs"), dest="runs_dir"
    )
    parser.add_argument("--registry-dir", default=None, dest="registry_dir")
    parser.add_argument("--raw-data-dir", default=None, dest="raw_data_dir")
    parser.add_argument(
        "--probes", action="store_true", help="also run the Stage-1 data probes"
    )
    parser.add_argument("--json", default=None, help="write the full result as JSON")
    args = parser.parse_args()

    result = run_audit(
        Path(args.runs_dir),
        run_probes=args.probes,
        registry_dir=args.registry_dir,
        raw_data_dir=args.raw_data_dir,
    )
    print(render(result))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1)
    # The V1 gate (spec section 6) makes BOTH conditions blocking, so both
    # must reach the exit code — CI reads this, not the printed banner.
    # Distinct codes so the two failure modes stay distinguishable:
    #   1 = false kill  — a legitimate research question was destroyed, and
    #       no human ever sees it. The worse failure.
    #   2 = missed mutant — the screen's teeth are dulling. Coverage gap,
    #       still blocking, because a screen that stops catching known-bad
    #       specs is how this silently becomes decorative.
    if result["false_kills"]:
        return 1
    if result.get("mutants_missed"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
