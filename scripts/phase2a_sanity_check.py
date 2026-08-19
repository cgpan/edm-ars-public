"""V2.0 Phase 2a sanity check.

Loads the skill registry from `skills/`, prints counts and per-layer
breakdown, then runs a representative prediction-on-HSLS query and
prints the matched skills.

Run from the project root:

    python scripts/phase2a_sanity_check.py

Output is captured to `scripts/phase2a_sanity_check_output.txt` for the
commit by:

    python scripts/phase2a_sanity_check.py > scripts/phase2a_sanity_check_output.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable when invoking the script directly
# (e.g. `python scripts/phase2a_sanity_check.py`).
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.skills import SkillRegistry  # noqa: E402


def main() -> None:
    skills_root = Path(__file__).resolve().parents[1] / "skills"
    registry = SkillRegistry(skills_root=skills_root)
    registry.reload()

    print("# Phase 2a Sanity Check")
    print()
    print(f"Skills root: {skills_root}")
    print(f"Loaded: {registry.count()}")
    by_layer = registry.count_by_layer()
    print("By layer:")
    for layer in ("task-type", "dataset", "methodology", "writing"):
        print(f"  {layer}: {by_layer.get(layer, 0)}")
    print()

    print("## Match query")
    print("stage=Analyst, task_type=prediction, dataset=hsls09_public")
    print(
        'context="Predict 11th-grade math achievement with SHAP '
        'interpretability and subgroup fairness"'
    )
    print("top_k_per_layer={methodology: 4, writing: 2}")
    print()

    matched = registry.match_and_compose(
        stage="Analyst",
        task_type="prediction",
        dataset="hsls09_public",
        context=(
            "Predict 11th-grade math achievement with SHAP interpretability "
            "and subgroup fairness"
        ),
        top_k_per_layer={"methodology": 4, "writing": 2},
    )
    print("## Matched skills (post-composition)")
    for s in matched:
        print(f"  [{s.layer:11s}] {s.name} (priority={s.priority})")

    print()
    print("## Match query (Writer perspective)")
    print("stage=Writer, task_type=prediction, dataset=hsls09_public")
    print('context="ACM acmart sigconf paper with SHAP figures and tables"')
    print()
    matched_writer = registry.match_and_compose(
        stage="Writer",
        task_type="prediction",
        dataset="hsls09_public",
        context="ACM acmart sigconf paper with SHAP figures and tables",
        top_k_per_layer={"methodology": 2, "writing": 5},
    )
    for s in matched_writer:
        print(f"  [{s.layer:11s}] {s.name} (priority={s.priority})")


if __name__ == "__main__":
    main()
