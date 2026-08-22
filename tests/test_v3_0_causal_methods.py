"""V3.0 Phase 3b.2 — verify the five causal method skills (M1-M5) load,
compose, attach only to Analyst, and render correctly into prompts.

Ten logical tests landing alongside M1-M5. Each maps to the 3b.2
hand-off acceptance criteria:

  1. Method-skill loadability (parametrized over M1-M5).
  2. Composition resolution per method (no dangling refs).
  3. M1 does not compose G3 (causal-positivity-diagnostics).
  4. Acyclicity of the full 52-skill composition graph.
  5. Analyst stage matches all 11 V3.0 skills for causal_soo.
  6. Method skills attach ONLY to Analyst (PF/DE/Critic/Writer
     must NOT match M1-M5).
  7. Per-agent match completeness (extends the 3b.1 stage-filter
     test with the post-3b.2 expected match table).
  8. Mandatory-tag inventory unchanged from 3b.1 (G2/G3/G5/D1
     mandatory; M1-M5 not mandatory).
  9. Dry-run prompt rendering for M1 contains M1-specific content.
 10. Dry-run prompt rendering for M5 contains the
     `CausalForestDML(honest=True)` mandate substring.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.skills import SkillRegistry  # noqa: E402

SKILLS_ROOT = Path(__file__).parent.parent / "skills"

# The five new V3.0 Phase 3b.2 method skills, with their directory paths.
# All five live under skills/methodology/ (the schema has no `method`
# layer; the M/G distinction is taxonomical within the spec, not a
# registry layer).
METHOD_SKILLS_PATHS: dict[str, Path] = {
    "causal-regression-adjustment":
        SKILLS_ROOT / "methodology" / "causal-regression-adjustment",
    "causal-propensity-score-matching":
        SKILLS_ROOT / "methodology" / "causal-propensity-score-matching",
    "causal-inverse-probability-weighting":
        SKILLS_ROOT / "methodology" / "causal-inverse-probability-weighting",
    "causal-aipw-tmle":
        SKILLS_ROOT / "methodology" / "causal-aipw-tmle",
    "causal-forest-cate":
        SKILLS_ROOT / "methodology" / "causal-forest-cate",
}

ALL_METHOD_NAMES: frozenset[str] = frozenset(METHOD_SKILLS_PATHS)

# 3b.1 methodology + dataset skills.
G1 = "causal-dag-identification"
G2 = "causal-estimand-definition"
G3 = "causal-positivity-diagnostics"
G4 = "causal-balance-diagnostics"
G5 = "causal-sensitivity-unmeasured-confounding"
D1 = "hsls09-causal-conventions"

ALL_V3_SKILLS: frozenset[str] = frozenset({
    G1, G2, G3, G4, G5, D1, *ALL_METHOD_NAMES,
})


@pytest.fixture(scope="module")
def registry() -> SkillRegistry:
    return SkillRegistry(SKILLS_ROOT)


# ---------------------------------------------------------------------------
# Test 1 — Method-skill loadability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("skill_name,skill_dir", list(METHOD_SKILLS_PATHS.items()))
def test_method_skill_loads_with_matching_name(
    registry: SkillRegistry, skill_name: str, skill_dir: Path,
) -> None:
    """Each M1-M5 SKILL.md parses; frontmatter `name` matches directory."""
    assert skill_dir.is_dir(), f"directory {skill_dir} missing"
    skill_md = skill_dir / "SKILL.md"
    assert skill_md.is_file(), f"{skill_md} missing"

    skill = registry.get(skill_name)
    assert skill is not None, f"registry did not load {skill_name!r}"
    assert skill.name == skill_name


# ---------------------------------------------------------------------------
# Test 2 — Composition resolution per method
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("skill_name", list(METHOD_SKILLS_PATHS.keys()))
def test_method_references_resolve(
    registry: SkillRegistry, skill_name: str,
) -> None:
    """Each method's references_skills entries resolve to loaded skills."""
    skill = registry.get(skill_name)
    assert skill is not None
    for ref in skill.references_skills:
        assert registry.get(ref) is not None, (
            f"{skill_name} -> {ref!r} is a dangling reference"
        )


# ---------------------------------------------------------------------------
# Test 3 — M1 does not compose G3 (regression has no propensity)
# ---------------------------------------------------------------------------


def test_m1_does_not_compose_positivity_diagnostics(
    registry: SkillRegistry,
) -> None:
    """Per spec §4.1: M1 (regression adjustment) does not estimate a
    propensity score, so it does not compose G3. M1's analogue of
    positivity is the regression-context covariate-overlap diagnostic
    in G4 (the §3a.1 R2 Path A resolution)."""
    m1 = registry.get("causal-regression-adjustment")
    assert m1 is not None
    assert G3 not in m1.references_skills, (
        f"M1 must NOT compose G3 ({G3}); spec §4.1 explicitly excludes it"
    )
    # Sanity: M1 SHOULD compose G1, G2, G4, G5, D1.
    assert set(m1.references_skills) == {G1, G2, G4, G5, D1}, (
        f"M1 references_skills mismatch: got {m1.references_skills}"
    )


# ---------------------------------------------------------------------------
# Test 4 — Acyclicity of the full 53-skill composition graph
# ---------------------------------------------------------------------------


def test_full_composition_graph_is_acyclic(registry: SkillRegistry) -> None:
    """DFS color-marking over the full 53-skill registry.

    Count history: 41 (V2.0) → 47 (V3.0 Phase 3b.1: +G1-G5+D1) → 52
    (V3.0 Phase 3b.2: +M1-M5) → 53 (V3.0 Phase 3b.12: +causal-data-
    engineer-contract).
    """
    skills = {s.name: s for s in registry.all()}
    assert len(skills) == 70, f"expected 70 skills, got {len(skills)}"  # +1 natural-academic-prose (E2)

    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {n: WHITE for n in skills}

    def visit(node: str, path: list[str]) -> None:
        color[node] = GRAY
        for ref in skills[node].references_skills:
            if ref not in skills:
                continue
            if color[ref] == GRAY:
                cycle_start = path.index(ref) if ref in path else 0
                cycle = path[cycle_start:] + [ref]
                pytest.fail(
                    f"cycle detected: {' -> '.join(cycle)}"
                )
            if color[ref] == WHITE:
                visit(ref, path + [ref])
        color[node] = BLACK

    for name in skills:
        if color[name] == WHITE:
            visit(name, [name])


# ---------------------------------------------------------------------------
# Test 5 — Analyst matches all 11 V3.0 skills for causal_soo
# ---------------------------------------------------------------------------


def test_analyst_matches_all_v3_skills_for_causal_soo(
    registry: SkillRegistry,
) -> None:
    """The Analyst is the only stage that touches every V3.0 skill —
    methodology + dataset + every method. With a generous per-layer
    cap, all 11 must appear."""
    matched = registry.match(
        stage="Analyst",
        task_type="causal_soo",
        dataset="hsls09_public",
        top_k_per_layer={
            "task-type": 20, "dataset": 20, "methodology": 20, "writing": 20,
        },
    )
    matched_names = {s.name for s in matched}
    missing = ALL_V3_SKILLS - matched_names
    assert not missing, (
        f"Analyst stage missing V3.0 skills for causal_soo: {sorted(missing)}"
    )


# ---------------------------------------------------------------------------
# Test 6 — Method skills attach only to Analyst
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stage", ["ProblemFormulator", "DataEngineer", "Critic", "Writer"],
)
def test_method_skills_do_not_match_non_analyst_stages(
    registry: SkillRegistry, stage: str,
) -> None:
    """M1-M5 declare applicable_stages: [Analyst] only — none of them
    should appear when matching for any other stage."""
    matched = registry.match(
        stage=stage,
        task_type="causal_soo",
        dataset="hsls09_public",
        top_k_per_layer={
            "task-type": 20, "dataset": 20, "methodology": 20, "writing": 20,
        },
    )
    matched_names = {s.name for s in matched}
    leaked_methods = ALL_METHOD_NAMES & matched_names
    assert not leaked_methods, (
        f"stage {stage!r} unexpectedly matched method skills: "
        f"{sorted(leaked_methods)}; M1-M5 must attach only to Analyst"
    )


# ---------------------------------------------------------------------------
# Test 7 — Per-agent match completeness (post-3b.2 expected table)
# ---------------------------------------------------------------------------


# Per spec §4.2 + 3b.2 attachment. The Analyst row gains all 5 methods;
# every other row is unchanged from 3b.1.
EXPECTED_NEW_SKILLS_PER_STAGE_POST_3B2: dict[str, frozenset[str]] = {
    "ProblemFormulator": frozenset({G1, G2, D1}),
    "DataEngineer": frozenset({D1}),
    "Analyst": frozenset({G1, G2, G3, G4, G5, D1, *ALL_METHOD_NAMES}),
    "Critic": frozenset({G1, G2, G3, G4, G5, D1}),
    "Writer": frozenset({G2, G4, G5, D1}),
}


@pytest.mark.parametrize(
    "stage,expected_subset",
    list(EXPECTED_NEW_SKILLS_PER_STAGE_POST_3B2.items()),
)
def test_post_3b2_stage_match_table(
    registry: SkillRegistry, stage: str, expected_subset: frozenset[str],
) -> None:
    """Post-3b.2 the stage filter must yield the table:
       PF -> {G1, G2, D1}
       DE -> {D1}
       Analyst -> {G1..G5, D1, M1..M5}
       Critic -> {G1..G5, D1}
       Writer -> {G2, G4, G5, D1}
    """
    matched = registry.match(
        stage=stage,
        task_type="causal_soo",
        dataset="hsls09_public",
        top_k_per_layer={
            "task-type": 20, "dataset": 20, "methodology": 20, "writing": 20,
        },
    )
    matched_names = {s.name for s in matched}
    missing = expected_subset - matched_names
    assert not missing, (
        f"stage {stage!r}: V3.0 skills missing from match: "
        f"{sorted(missing)}"
    )
    # Also assert that no method skill leaks to non-Analyst stages.
    if stage != "Analyst":
        leaked = ALL_METHOD_NAMES & matched_names
        assert not leaked, (
            f"stage {stage!r} matched method skills: {sorted(leaked)}"
        )


# ---------------------------------------------------------------------------
# Test 8 — Mandatory-tag inventory (updated post-3b.8)
# ---------------------------------------------------------------------------


# Phase 3b.8: M1-M5 promoted from recommended to mandatory. The change
# is documented in the v3.0 causal-methods specification (internal) § "Phase 3b.8
# amendment (post-3b.7 formatter discovery)". Original 3b.1/3b.2 mandatory
# inventory was {G2, G3, G5, D1}; post-3b.8 it is that set ∪ {M1-M5}.
EXPECTED_V3_MANDATORY: frozenset[str] = frozenset(
    {G2, G3, G5, D1, *ALL_METHOD_NAMES}
)


def test_v3_mandatory_inventory_post_3b8(
    registry: SkillRegistry,
) -> None:
    """Post-3b.8 mandatory inventory: G2, G3, G5, D1 (3b.1 originals)
    PLUS M1-M5 (promoted in 3b.8 to fix F-3b7-FORMATTER-TRUNCATES-METHOD-
    SKILLS). G1 and G4 remain recommended.
    """
    actual_mandatory: set[str] = set()
    for name in ALL_V3_SKILLS:
        skill = registry.get(name)
        assert skill is not None
        if skill.rule_severity == "mandatory":
            actual_mandatory.add(name)
    assert actual_mandatory == EXPECTED_V3_MANDATORY, (
        f"V3.0 mandatory inventory drift:\n"
        f"  expected: {sorted(EXPECTED_V3_MANDATORY)}\n"
        f"  actual:   {sorted(actual_mandatory)}"
    )
    # Explicit cross-check: every method skill IS now mandatory (3b.8).
    for m in ALL_METHOD_NAMES:
        skill = registry.get(m)
        assert skill is not None
        assert skill.rule_severity == "mandatory", (
            f"method skill {m} expected mandatory post-3b.8"
        )
    # G1, G4 explicitly NOT promoted (deliberate per the 3b.8 hand-off).
    for non_promoted in (G1, G4):
        skill = registry.get(non_promoted)
        assert skill is not None
        assert skill.rule_severity == "recommended", (
            f"{non_promoted} should remain recommended; promoting "
            f"G1/G4 in 3b.8 would have been a separate spec decision."
        )


# ---------------------------------------------------------------------------
# Test 9 — Dry-run prompt rendering for M1
# ---------------------------------------------------------------------------


def test_dry_run_render_m1_prompt_contains_m1_content(
    registry: SkillRegistry,
) -> None:
    """When the Analyst prompt is rendered for causal_soo, the rendered
    text must include M1's body content (we use the function-signature
    line as a stable substring that survives any reformatting)."""
    text = registry.format_for_prompt(
        stage="Analyst",
        task_type="causal_soo",
        dataset="hsls09_public",
        context="estimate ATE via regression adjustment with cluster-robust SEs",
        max_chars=200_000,
        top_k_per_layer={
            "task-type": 20, "dataset": 20, "methodology": 20, "writing": 20,
        },
    )
    # M1's section header in the rendered prompt.
    assert (
        "## Guidance: causal-regression-adjustment" in text
        or "## MANDATORY RULE: causal-regression-adjustment" in text
    ), "M1 skill header not present in rendered Analyst prompt"
    # M1-specific content from the body.
    assert "regression_adjustment_ate" in text, (
        "M1 function signature missing from rendered prompt"
    )
    # G4 regression-context invocation is the M1-distinctive content.
    assert "regression-context" in text or "regression_context" in text, (
        "M1's G4 regression-context invocation missing from rendered prompt"
    )


# ---------------------------------------------------------------------------
# Test 10 — Dry-run prompt rendering for M5 contains honest=True mandate
# ---------------------------------------------------------------------------


def test_dry_run_render_m5_prompt_contains_honest_mandate(
    registry: SkillRegistry,
) -> None:
    """M5's `CausalForestDML(honest=True)` mandate must appear in the
    rendered Analyst prompt verbatim (or as a stable substring)."""
    text = registry.format_for_prompt(
        stage="Analyst",
        task_type="causal_soo",
        dataset="hsls09_public",
        context="estimate CATE via causal forest with honest splitting",
        max_chars=200_000,
        top_k_per_layer={
            "task-type": 20, "dataset": 20, "methodology": 20, "writing": 20,
        },
    )
    assert (
        "## Guidance: causal-forest-cate" in text
        or "## MANDATORY RULE: causal-forest-cate" in text
    ), "M5 skill header not present in rendered Analyst prompt"
    # The honest=True mandate substring (transcribed verbatim from the spec).
    assert "honest=True" in text, (
        "M5's honest=True mandate missing from rendered prompt"
    )
    # ATE-on-overlap-population label discipline is the M5-distinctive content.
    assert "ATE-on-overlap-population" in text, (
        "M5's ATE-on-overlap-population label rule missing from rendered prompt"
    )
