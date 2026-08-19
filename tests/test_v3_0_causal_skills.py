"""V3.0 Phase 3b.1 — verify causal_soo methodology + dataset skills load and wire correctly.

Six tests landing alongside the new causal-inference SKILL.md files
(G1, G2, G3, G4, G5, D1). Each maps directly to the 3b.1 hand-off
acceptance criteria:

  1. Skill loadability: every new skill ID parses and the frontmatter
     `name` matches the directory name.
  2. task_type=causal_soo presence: at least one skill declares it
     in `applicable_task_types`. The matcher has no whitelist (any
     string is accepted), so the existence of declaring skills IS
     the registration.
  3. Composition resolution: every reference in `references_skills`
     resolves to an existing skill — no dangling refs.
  4. Acyclicity: the full composition graph (existing 41 + 6 new)
     contains no cycles. Detected via DFS color-marking.
  5. Stage-filter expectations for `task_type=causal_soo` per the
     §4.2 attachment table — 3b.1 expectations only (Analyst will
     additionally include M1-M5 after 3b.2).
  6. Mandatory-tag inventory: exactly G2, G3, G5, D1 carry
     `rule_severity: mandatory` among the new skills; G1 and G4
     do not. Catches accidental tag drift.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.skills import SkillRegistry  # noqa: E402

SKILLS_ROOT = Path(__file__).parent.parent / "skills"

# The six new V3.0 Phase 3b.1 skills, with their directory paths.
NEW_SKILLS_PATHS: dict[str, Path] = {
    "causal-dag-identification":
        SKILLS_ROOT / "methodology" / "causal-dag-identification",
    "causal-estimand-definition":
        SKILLS_ROOT / "methodology" / "causal-estimand-definition",
    "causal-positivity-diagnostics":
        SKILLS_ROOT / "methodology" / "causal-positivity-diagnostics",
    "causal-balance-diagnostics":
        SKILLS_ROOT / "methodology" / "causal-balance-diagnostics",
    "causal-sensitivity-unmeasured-confounding":
        SKILLS_ROOT / "methodology" / "causal-sensitivity-unmeasured-confounding",
    "hsls09-causal-conventions":
        SKILLS_ROOT / "dataset" / "hsls09-causal-conventions",
}

# Per spec §3.x and §3a.1 R5: G2, G3, G5, D1 carry mandatory.
# G1 and G4 do NOT.
EXPECTED_MANDATORY: frozenset[str] = frozenset({
    "causal-estimand-definition",
    "causal-positivity-diagnostics",
    "causal-sensitivity-unmeasured-confounding",
    "hsls09-causal-conventions",
})
EXPECTED_NON_MANDATORY: frozenset[str] = frozenset({
    "causal-dag-identification",
    "causal-balance-diagnostics",
})


@pytest.fixture(scope="module")
def registry() -> SkillRegistry:
    """Production registry loaded once per module."""
    return SkillRegistry(SKILLS_ROOT)


# ---------------------------------------------------------------------------
# Test 1 — Skill loadability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("skill_name,skill_dir", list(NEW_SKILLS_PATHS.items()))
def test_new_causal_skill_loads_with_matching_name(
    registry: SkillRegistry, skill_name: str, skill_dir: Path,
) -> None:
    """Each of the 6 new SKILL.md files parses, and the frontmatter
    `name` matches the directory name. Catches typos in either place."""
    assert skill_dir.is_dir(), f"directory {skill_dir} does not exist"
    skill_md = skill_dir / "SKILL.md"
    assert skill_md.is_file(), f"{skill_md} does not exist"

    skill = registry.get(skill_name)
    assert skill is not None, (
        f"SkillRegistry did not load {skill_name!r}; check frontmatter parses"
    )
    assert skill.name == skill_name, (
        f"name/dir mismatch: frontmatter name={skill.name!r} "
        f"vs directory {skill_dir.name!r}"
    )


# ---------------------------------------------------------------------------
# Test 2 — task_type=causal_soo is registered (i.e. declared by ≥1 skill)
# ---------------------------------------------------------------------------


def test_causal_soo_task_type_is_declared(registry: SkillRegistry) -> None:
    """The matcher has no task_type whitelist — empty `applicable_task_types`
    means universal, non-empty restricts. So 'registration' of a task_type
    is established by ≥1 skill declaring it.

    All 6 new skills must declare causal_soo, so the union should contain
    causal_soo (and the matcher will hard-filter accordingly)."""
    declared: set[str] = set()
    for skill in registry.all():
        declared.update(skill.applicable_task_types)
    assert "causal_soo" in declared, (
        "no skill declares applicable_task_types: [causal_soo]; "
        "the V3.0 Phase 3b.1 skills are missing or mis-tagged"
    )

    # Every new skill must explicitly declare causal_soo.
    for skill_name in NEW_SKILLS_PATHS:
        skill = registry.get(skill_name)
        assert skill is not None
        assert "causal_soo" in skill.applicable_task_types, (
            f"{skill_name} should declare applicable_task_types: [causal_soo]; "
            f"got {skill.applicable_task_types}"
        )


# ---------------------------------------------------------------------------
# Test 3 — Composition resolution per skill (no dangling refs)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("skill_name", list(NEW_SKILLS_PATHS.keys()))
def test_new_skill_references_resolve(
    registry: SkillRegistry, skill_name: str,
) -> None:
    """Each entry in `references_skills` must resolve to a loaded skill."""
    skill = registry.get(skill_name)
    assert skill is not None
    for ref in skill.references_skills:
        assert registry.get(ref) is not None, (
            f"{skill_name} references {ref!r} which does not resolve to a "
            "loaded skill — dangling reference"
        )


# ---------------------------------------------------------------------------
# Test 4 — Acyclicity of the full composition graph
# ---------------------------------------------------------------------------


def test_composition_graph_is_acyclic(registry: SkillRegistry) -> None:
    """DFS color-marking over the full skill graph (41 existing + 6 new).
    A cycle here would mean two skills mutually reference each other (or
    a longer chain returns to a starting node), which the composer breaks
    with a warning at runtime — but that defense is for malformed user
    input, not a registry-shipped state. Treat any cycle in the shipped
    registry as a structural bug."""
    skills = {s.name: s for s in registry.all()}

    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {name: WHITE for name in skills}

    def visit(node: str, path: list[str]) -> None:
        color[node] = GRAY
        for ref in skills[node].references_skills:
            if ref not in skills:
                # Dangling reference — caught by test 3, not this test.
                continue
            if color[ref] == GRAY:
                cycle_start = path.index(ref) if ref in path else 0
                cycle = path[cycle_start:] + [ref]
                pytest.fail(
                    f"composition cycle detected: {' -> '.join(cycle)}"
                )
            if color[ref] == WHITE:
                visit(ref, path + [ref])
        color[node] = BLACK

    for name in skills:
        if color[name] == WHITE:
            visit(name, [name])


# ---------------------------------------------------------------------------
# Test 5 — Stage-filter expectations for task_type=causal_soo
# ---------------------------------------------------------------------------


# 3b.1 expectations per spec §4.2 (Analyst will additionally gain M1-M5
# after 3b.2). We assert the new V3.0 skills appear at the right stages;
# we do not assert the absence of unrelated existing skills (they may
# legitimately appear because they have empty applicable_task_types).
EXPECTED_NEW_SKILLS_PER_STAGE: dict[str, frozenset[str]] = {
    "ProblemFormulator": frozenset({
        "causal-dag-identification",
        "causal-estimand-definition",
        "hsls09-causal-conventions",
    }),
    "DataEngineer": frozenset({
        "hsls09-causal-conventions",
    }),
    "Analyst": frozenset({
        "causal-dag-identification",
        "causal-estimand-definition",
        "causal-positivity-diagnostics",
        "causal-balance-diagnostics",
        "causal-sensitivity-unmeasured-confounding",
        "hsls09-causal-conventions",
    }),
    "Critic": frozenset({
        "causal-dag-identification",
        "causal-estimand-definition",
        "causal-positivity-diagnostics",
        "causal-balance-diagnostics",
        "causal-sensitivity-unmeasured-confounding",
        "hsls09-causal-conventions",
    }),
    "Writer": frozenset({
        "causal-estimand-definition",
        "causal-balance-diagnostics",
        "causal-sensitivity-unmeasured-confounding",
        "hsls09-causal-conventions",
    }),
}


@pytest.mark.parametrize(
    "stage,expected_subset", list(EXPECTED_NEW_SKILLS_PER_STAGE.items()),
)
def test_causal_soo_stage_match_expectations(
    registry: SkillRegistry, stage: str, expected_subset: frozenset[str],
) -> None:
    """For each agent stage, the new-skill subset that should match
    on a `task_type=causal_soo, dataset=hsls09_public` request must be
    present in the matched output. We use a generous per-layer cap to
    avoid the cap dropping any of the new skills during this assertion
    (mandatory bypass already protects G2/G3/G5/D1 but G1/G4 are not
    mandatory and could be capped out without the override)."""
    matched = registry.match(
        stage=stage,
        task_type="causal_soo",
        dataset="hsls09_public",
        top_k_per_layer={
            "task-type": 20,
            "dataset": 20,
            "methodology": 20,
            "writing": 20,
        },
    )
    matched_names = {s.name for s in matched}
    missing = expected_subset - matched_names
    assert not missing, (
        f"stage {stage!r}: expected new V3.0 skills missing from match: "
        f"{sorted(missing)}; matched names = {sorted(matched_names)}"
    )


# ---------------------------------------------------------------------------
# Test 6 — Mandatory-tag inventory
# ---------------------------------------------------------------------------


def test_mandatory_tag_inventory_for_new_skills(registry: SkillRegistry) -> None:
    """Per spec: G2, G3, G5, D1 carry `rule_severity: mandatory`.
    G1 and G4 do not. Catches accidental tag drift in either direction."""
    actual_mandatory: set[str] = set()
    actual_non_mandatory: set[str] = set()
    for skill_name in NEW_SKILLS_PATHS:
        skill = registry.get(skill_name)
        assert skill is not None
        if skill.rule_severity == "mandatory":
            actual_mandatory.add(skill_name)
        else:
            actual_non_mandatory.add(skill_name)

    assert actual_mandatory == EXPECTED_MANDATORY, (
        f"mandatory-tag drift among new V3.0 skills:\n"
        f"  expected mandatory: {sorted(EXPECTED_MANDATORY)}\n"
        f"  actual   mandatory: {sorted(actual_mandatory)}"
    )
    assert actual_non_mandatory == EXPECTED_NON_MANDATORY, (
        f"non-mandatory-tag drift among new V3.0 skills:\n"
        f"  expected non-mandatory: {sorted(EXPECTED_NON_MANDATORY)}\n"
        f"  actual   non-mandatory: {sorted(actual_non_mandatory)}"
    )
