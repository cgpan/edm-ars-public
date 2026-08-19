"""V2.0 Phase 2b sanity check.

Six assertions on the post-2b registry:

1. Total count + per-layer breakdown match the extraction plan (41
   skills: 14 task-type + 7 dataset + 11 methodology + 9 writing).
2. The bundled HSLS variable registry resource resolves and is large
   (> 10 KB).
3. Decision-9 composition: an Analyst prediction match pulls in
   `prediction-model-battery` and all six per-family skills via
   `references_skills`, with no duplicates.
4. Decision-6 composition: a DataEngineer HSLS match pulls in
   `hsls09-school-fingerprints` and `cluster-id-reconstruction-from-fingerprints`
   via reference.
5. Cross-task filter: `task_type="causal_inference"` excludes ALL
   prediction task-type skills while still letting cross-cutting
   methodology + writing + dataset skills through.
6. No-dataset filter: `dataset="els2002"` excludes ALL HSLS-specific
   dataset/writing skills.

Run from the project root:

    python scripts/phase2b_sanity_check.py > scripts/phase2b_sanity_check_output.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

# Force UTF-8 stdout on Windows so any future Unicode survives the cp1252
# console codec (Phase 2b had to drop arrows/em-dashes for this reason).
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        # Older Python or non-reconfigurable streams: fall back to ASCII
        # which the script already uses.
        pass

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.skills import SkillRegistry  # noqa: E402


# Per-family skill names that prediction-model-battery should pull in.
EXPECTED_FAMILIES = {
    "model-logistic-regression",
    "model-random-forest",
    "model-xgboost",
    "model-elasticnet",
    "model-mlp",
    "model-stacking-ensemble",
}

# HSLS-coupled skills (any layer) that should not match a non-HSLS dataset.
HSLS_SKILL_NAMES = {
    # dataset layer (all HSLS)
    "hsls09-variable-registry",
    "hsls09-csv-format-quirks",
    "hsls09-missing-codes",
    "hsls09-temporal-ordering",
    "hsls09-school-fingerprints",
    "hsls09-tier3-exclusions",
    "hsls09-structural-mnar-outcomes",
    # writing layer with applicable_datasets=[hsls09_public]
    "hsls09-multilevel-limitations-paragraph",
    "hsls09-survey-weights-limitations-paragraph",
}


def _section(title: str) -> None:
    print()
    print(f"## {title}")


def _ok(label: str) -> str:
    return f"  OK    {label}"


def _fail(label: str, detail: str = "") -> str:
    return f"  FAIL  {label}" + (f" ({detail})" if detail else "")


def main() -> None:
    skills_root = _PROJECT_ROOT / "skills"
    registry = SkillRegistry(skills_root=skills_root)

    print("# Phase 2b Sanity Check")
    print()
    print(f"Skills root: {skills_root}")
    print(f"Loaded: {registry.count()}")
    by_layer = registry.count_by_layer()
    print("By layer:")
    for layer in ("task-type", "dataset", "methodology", "writing"):
        print(f"  {layer}: {by_layer.get(layer, 0)}")

    failures: list[str] = []

    # ----- 1. Total count and per-layer breakdown -----
    _section("Assertion 1 -- total count + per-layer breakdown")
    expected_total = 41
    expected_by_layer = {"task-type": 14, "dataset": 7, "methodology": 11, "writing": 9}
    if registry.count() == expected_total:
        print(_ok(f"total = {expected_total}"))
    else:
        msg = _fail(f"total = {registry.count()}", f"expected {expected_total}")
        print(msg)
        failures.append(msg)
    for layer, want in expected_by_layer.items():
        got = by_layer.get(layer, 0)
        if got == want:
            print(_ok(f"{layer} = {want}"))
        else:
            msg = _fail(f"{layer} = {got}", f"expected {want}")
            print(msg)
            failures.append(msg)

    # ----- 2. Bundled HSLS registry resource -----
    _section("Assertion 2 -- hsls09-variable-registry bundled resource")
    skill = registry.get("hsls09-variable-registry")
    if skill is None:
        msg = _fail("hsls09-variable-registry not loaded")
        print(msg)
        failures.append(msg)
    else:
        paths = skill.resource_paths
        if not paths:
            msg = _fail("no resource_paths on hsls09-variable-registry")
            print(msg)
            failures.append(msg)
        else:
            p = paths[0]
            size = p.stat().st_size if p.exists() else -1
            if p.exists() and size > 10_000:
                print(_ok(f"resource exists at {p.name} ({size:,} bytes)"))
            else:
                msg = _fail(
                    f"resource missing or too small at {p}", f"size={size}"
                )
                print(msg)
                failures.append(msg)

    # ----- 3. Decision-9 composition (model battery -> 6 families) -----
    _section("Assertion 3 -- Decision-9 composition (per-family expansion)")
    matched = registry.match_and_compose(
        stage="Analyst",
        task_type="prediction",
        dataset="hsls09_public",
        context="Train and evaluate the prediction model battery with SHAP",
        top_k_per_layer={"task-type": 4, "methodology": 2, "writing": 0, "dataset": 0},
    )
    matched_names = [s.name for s in matched]
    if "prediction-model-battery" in matched_names:
        print(_ok("prediction-model-battery matched"))
    else:
        msg = _fail("prediction-model-battery did NOT match")
        print(msg)
        failures.append(msg)
    missing_families = EXPECTED_FAMILIES - set(matched_names)
    if not missing_families:
        print(_ok("all 6 per-family skills present via composition"))
    else:
        msg = _fail(
            "missing per-family skills",
            f"{sorted(missing_families)}",
        )
        print(msg)
        failures.append(msg)
    if len(matched_names) == len(set(matched_names)):
        print(_ok(f"no duplicates in composed list ({len(matched_names)} skills)"))
    else:
        dupes = [n for n in matched_names if matched_names.count(n) > 1]
        msg = _fail("duplicates in composed list", f"{dupes}")
        print(msg)
        failures.append(msg)
    print("  composed:")
    for n in matched_names:
        print(f"    - {n}")

    # ----- 4. Decision-6 composition (HSLS school-fingerprints -> cluster recon) -----
    _section("Assertion 4 -- Decision-6 composition (HSLS -> cluster reconstruction)")
    matched_de = registry.match_and_compose(
        stage="DataEngineer",
        task_type="prediction",
        dataset="hsls09_public",
        context="Recover school cluster IDs from HSLS fingerprint variables",
        top_k_per_layer={"task-type": 1, "methodology": 1, "writing": 0, "dataset": 4},
    )
    matched_de_names = [s.name for s in matched_de]
    if "hsls09-school-fingerprints" in matched_de_names:
        print(_ok("hsls09-school-fingerprints matched"))
    else:
        msg = _fail("hsls09-school-fingerprints did NOT match")
        print(msg)
        failures.append(msg)
    if "cluster-id-reconstruction-from-fingerprints" in matched_de_names:
        print(_ok("cluster-id-reconstruction-from-fingerprints pulled in via reference"))
    else:
        msg = _fail("cluster-id-reconstruction-from-fingerprints NOT pulled in")
        print(msg)
        failures.append(msg)
    print("  composed:")
    for n in matched_de_names:
        print(f"    - {n}")

    # ----- 5. Cross-task filter (causal_inference excludes prediction skills) -----
    _section("Assertion 5 -- task_type filter excludes prediction skills")
    matched_causal = registry.match(
        stage="Analyst",
        task_type="causal_inference",
        dataset="hsls09_public",
        context="Estimate causal effect of treatment on outcome",
        top_k_per_layer={"task-type": 99, "dataset": 99, "methodology": 99, "writing": 99},
    )
    leaked_prediction: list[str] = []
    for s in matched_causal:
        if s.layer == "task-type" and "prediction" in s.applicable_task_types:
            leaked_prediction.append(s.name)
    if not leaked_prediction:
        print(_ok("no prediction task-type skills leaked into causal_inference query"))
    else:
        msg = _fail("prediction skills leaked", f"{leaked_prediction}")
        print(msg)
        failures.append(msg)
    # Cross-cutting layers should still match.
    methodology_count = sum(1 for s in matched_causal if s.layer == "methodology")
    if methodology_count > 0:
        print(_ok(f"cross-cutting methodology skills still match ({methodology_count} found)"))
    else:
        msg = _fail("no methodology skills matched for causal query")
        print(msg)
        failures.append(msg)

    # ----- 6. No-dataset filter (els2002 excludes HSLS skills) -----
    _section("Assertion 6 -- dataset filter excludes HSLS-specific skills")
    # Two-perspective query: Analyst sees task-type + methodology; Writer sees writing.
    matched_els_analyst = registry.match(
        stage="Analyst",
        task_type="prediction",
        dataset="els2002",
        context="Predict student outcomes using ELS:2002 data",
        top_k_per_layer={"task-type": 99, "dataset": 99, "methodology": 99, "writing": 99},
    )
    matched_els_writer = registry.match(
        stage="Writer",
        task_type="prediction",
        dataset="els2002",
        context="Write up the ELS:2002 prediction paper",
        top_k_per_layer={"task-type": 99, "dataset": 99, "methodology": 99, "writing": 99},
    )
    els_skills_by_name = {s.name: s for s in matched_els_analyst}
    els_skills_by_name.update({s.name: s for s in matched_els_writer})
    els_names = set(els_skills_by_name.keys())
    leaked_hsls = HSLS_SKILL_NAMES & els_names
    if not leaked_hsls:
        print(_ok("no HSLS-coupled skills leaked into els2002 query (Analyst+Writer)"))
    else:
        msg = _fail("HSLS skills leaked", f"{sorted(leaked_hsls)}")
        print(msg)
        failures.append(msg)
    task_type_count = sum(1 for s in els_skills_by_name.values() if s.layer == "task-type")
    writing_count = sum(1 for s in els_skills_by_name.values() if s.layer == "writing")
    if task_type_count > 0 and writing_count > 0:
        print(_ok(
            f"cross-cutting layers still match for els2002 query "
            f"(task-type={task_type_count}, writing={writing_count})"
        ))
    else:
        msg = _fail(
            "expected cross-cutting layers missing",
            f"task-type={task_type_count}, writing={writing_count}",
        )
        print(msg)
        failures.append(msg)

    print()
    print("## Summary")
    if failures:
        print(f"FAILED: {len(failures)} assertion(s) failed")
        for f in failures:
            print(f"  - {f.strip()}")
        sys.exit(1)
    else:
        print("PASS: all 6 assertions passed")


if __name__ == "__main__":
    main()
