"""Diagnostic: verify every skill reaches the stages it declares.

For each skill in the registry:
  1. Identify its `applicable_stages` (or all 5 stages if empty).
  2. For each declared stage, run match_and_compose with realistic
     defaults and check whether the skill appears in the matched set.
  3. Flag any skill that declares a stage but doesn't appear in that
     stage's matched set, with a categorized reason.

Permanent diagnostic — re-run whenever new skills are added.

Usage (from project root):

    python scripts/verify_skill_flow.py > scripts/skill_flow_verification.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        pass

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.skills import Skill, SkillRegistry  # noqa: E402
from src.skills.matcher import _score, _tokenize  # noqa: E402

# Stages the orchestrator runs (matches src/orchestrator.py and the
# audit's per-agent inventory).
ALL_STAGES = (
    "ProblemFormulator",
    "DataEngineer",
    "Analyst",
    "Critic",
    "OutlineAgent",
    "Writer",
)

# Defaults that mirror the orchestrator's _match_skills_for_stage call.
DEFAULT_TASK_TYPE = "prediction"
DEFAULT_DATASET = "hsls09_public"
DEFAULT_CONTEXT = (
    "Predict 11th-grade math achievement from 9th-grade factors with SHAP "
    "and subgroup fairness"
)
ORCHESTRATOR_CAPS = {
    "task-type": 3,
    "dataset": 4,
    "methodology": 5,
    "writing": 5,
}


def _classify_miss(
    skill: Skill,
    stage: str,
    registry: SkillRegistry,
    matched_names: set[str],
) -> str:
    """Return a short categorization explaining why a skill didn't reach a stage."""
    # Hard filters first.
    if skill.applicable_task_types and DEFAULT_TASK_TYPE not in skill.applicable_task_types:
        return f"task_type filter (allows {skill.applicable_task_types})"
    if skill.applicable_datasets and DEFAULT_DATASET not in skill.applicable_datasets:
        return f"dataset filter (allows {skill.applicable_datasets})"
    if skill.applicable_stages and stage not in skill.applicable_stages:
        # Self-inconsistency: caller asked about a stage that the skill
        # doesn't declare. We never enter this branch because we only
        # check stages the skill DOES declare.
        return "stage filter (skill does not declare this stage)"

    # Past hard filters: must be a ranking issue. Reconstruct the score
    # comparison for this layer.
    same_layer = [
        s for s in registry.all()
        if s.layer == skill.layer
        and s.applies_to_stage(stage)
        and s.applies_to_task_type(DEFAULT_TASK_TYPE)
        and s.applies_to_dataset(DEFAULT_DATASET)
    ]
    cap = ORCHESTRATOR_CAPS.get(skill.layer, len(same_layer))
    context_tokens = _tokenize(DEFAULT_CONTEXT) if DEFAULT_CONTEXT else set()
    scored = sorted(
        ((s, _score(s, context_tokens)) for s in same_layer),
        key=lambda pair: -pair[1],
    )
    surviving_names = {s.name for s, _ in scored[:cap]}
    if skill.name in surviving_names:
        # Survived the cap but got lost in composition / dedup. Should
        # be rare (only if a referenced skill brought it in then it got
        # dropped, or a test-only skill had a name collision).
        return "matched but lost in composition"
    # Score race: how far out of the cap did it land?
    rank_in_layer = next(
        (i for i, (s, _) in enumerate(scored) if s.name == skill.name),
        -1,
    )
    if rank_in_layer < 0:
        return "did not pass score sort (unexpected)"
    if cap >= len(scored):
        return "matched (no cap pressure)"
    # Determine whether the cause is keyword mismatch or priority loss.
    skill_score = next(s_score for s, s_score in scored if s.name == skill.name)
    survivor_min_score = scored[cap - 1][1] if cap > 0 else 0.0
    deficit = survivor_min_score - skill_score
    has_keywords = bool(skill.trigger_keywords)
    if not has_keywords:
        return (
            f"per-layer cap (no trigger_keywords; rank {rank_in_layer + 1}/{len(scored)} "
            f"in {skill.layer}; cap = {cap}; score {skill_score:.2f} vs "
            f"survivor cutoff {survivor_min_score:.2f})"
        )
    return (
        f"per-layer cap ({skill.layer} ties; rank {rank_in_layer + 1}/{len(scored)}; "
        f"cap = {cap}; score {skill_score:.2f} vs survivor cutoff "
        f"{survivor_min_score:.2f}; deficit {deficit:.2f})"
    )


def main() -> None:
    registry = SkillRegistry(skills_root=_PROJECT_ROOT / "skills")
    print("# Skill-Flow Verification")
    print()
    print(f"Skills root: {registry.skills_root}")
    print(f"Total skills loaded: {registry.count()}")
    print(f"Mandatory total: {sum(1 for s in registry.all() if s.rule_severity == 'mandatory')}")
    print(f"Default context: {DEFAULT_CONTEXT!r}")
    print(f"Default task_type: {DEFAULT_TASK_TYPE}")
    print(f"Default dataset: {DEFAULT_DATASET}")
    print(f"Per-layer caps: {ORCHESTRATOR_CAPS}")
    print()

    rows: list[tuple[str, str, str, str]] = []  # name, declared_stages, status, reason
    misfires: list[tuple[str, str, str]] = []  # skill_name, stage, reason

    for skill in sorted(registry.all(), key=lambda s: (s.layer, s.name)):
        declared = (
            list(skill.applicable_stages) if skill.applicable_stages else list(ALL_STAGES)
        )
        per_stage_results: list[str] = []
        for stage in declared:
            matched_skills = registry.match_and_compose(
                stage=stage,
                task_type=DEFAULT_TASK_TYPE,
                dataset=DEFAULT_DATASET,
                context=DEFAULT_CONTEXT,
                top_k_per_layer=ORCHESTRATOR_CAPS,
            )
            matched_names = {s.name for s in matched_skills}
            if skill.name in matched_names:
                per_stage_results.append(f"{stage}: YES")
            else:
                reason = _classify_miss(skill, stage, registry, matched_names)
                per_stage_results.append(f"{stage}: NO ({reason})")
                misfires.append((skill.name, stage, reason))
        rows.append(
            (
                skill.name,
                ", ".join(declared),
                "; ".join(per_stage_results),
                skill.rule_severity,
            )
        )

    # Print compact table.
    print("## Per-skill flow")
    print()
    print("| Skill | Severity | Declared Stages | Reaches Stage? |")
    print("|---|---|---|---|")
    for name, declared_str, status, severity in rows:
        # Render YES/NO without exploding the table when a skill targets all stages.
        short = status if "NO" in status else "all declared stages: YES"
        print(f"| `{name}` | `{severity}` | {declared_str} | {short} |")

    print()
    print("## Summary")
    print(f"Total skills audited: {len(rows)}")
    print(f"Total misfires: {len(misfires)}")
    if not misfires:
        print()
        print("No misfires. Every declared stage is reached.")
        return

    print()
    print("## Misfires (skill declares stage but is not matched)")
    print()
    print("| Skill | Stage | Reason |")
    print("|---|---|---|")
    for name, stage, reason in misfires:
        print(f"| `{name}` | {stage} | {reason} |")


if __name__ == "__main__":
    main()
