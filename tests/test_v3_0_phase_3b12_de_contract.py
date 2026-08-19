"""V3.0 Phase 3b.12 / §12.1 + §12.4 — causal-data-engineer-contract skill.

Locks the post-3b.12 inventory: a new mandatory methodology skill
``causal-data-engineer-contract`` is registered and matches at the
DataEngineer + Analyst stages for ``task_type=causal_soo``. The skill
codifies the carve-out contract that prevents
F-3b11-DE-MISSING-TREATMENT-COLUMN.

Sub-wave 1 scope: skill loads cleanly, frontmatter is correct, the
matcher returns it for both stages, the composition graph stays
acyclic, and the V3.0 causal skill count goes from 11 → 12.

Sub-wave 3 (post-amend) extends the rendered-prompt verification in
``test_rendered_prompt_contains_all_mskills.py`` and bumps the post-3b.8
mandatory inventory in ``test_v3_0_phase_3b8_mandatory_promotion.py``.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.skills import SkillRegistry


SKILLS_ROOT = Path(__file__).parent.parent / "skills"
NEW_SKILL_NAME = "causal-data-engineer-contract"


@pytest.fixture(scope="module")
def registry() -> SkillRegistry:
    return SkillRegistry(SKILLS_ROOT)


# ---------------------------------------------------------------------------
# §12.4 — skill loadability + frontmatter contract
# ---------------------------------------------------------------------------


class TestCausalDataEngineerContractSkillLoads:
    def test_skill_directory_and_skillmd_exist(self) -> None:
        skill_dir = SKILLS_ROOT / "methodology" / NEW_SKILL_NAME
        assert skill_dir.is_dir(), f"{skill_dir} must exist"
        assert (skill_dir / "SKILL.md").is_file(), (
            f"{skill_dir / 'SKILL.md'} must exist"
        )

    def test_skill_loads_into_registry(
        self, registry: SkillRegistry
    ) -> None:
        skill = registry.get(NEW_SKILL_NAME)
        assert skill is not None, (
            f"{NEW_SKILL_NAME!r} did not load; check SKILL.md frontmatter"
        )

    def test_skill_layer_is_methodology(
        self, registry: SkillRegistry
    ) -> None:
        skill = registry.get(NEW_SKILL_NAME)
        assert skill is not None
        assert skill.layer == "methodology"

    def test_skill_applies_to_causal_soo(
        self, registry: SkillRegistry
    ) -> None:
        skill = registry.get(NEW_SKILL_NAME)
        assert skill is not None
        assert "causal_soo" in skill.applicable_task_types, (
            f"{NEW_SKILL_NAME} must declare applicable_task_types: [causal_soo]"
        )

    def test_skill_applies_to_dataengineer_and_analyst_stages(
        self, registry: SkillRegistry
    ) -> None:
        skill = registry.get(NEW_SKILL_NAME)
        assert skill is not None
        # Both stages required: DataEngineer produces the artifact;
        # Analyst reads it and benefits from knowing the contract.
        assert "DataEngineer" in skill.applicable_stages, (
            f"{NEW_SKILL_NAME} must include DataEngineer in applicable_stages"
        )
        assert "Analyst" in skill.applicable_stages, (
            f"{NEW_SKILL_NAME} must include Analyst in applicable_stages"
        )

    def test_skill_severity_is_mandatory(
        self, registry: SkillRegistry
    ) -> None:
        skill = registry.get(NEW_SKILL_NAME)
        assert skill is not None
        assert skill.rule_severity == "mandatory", (
            f"{NEW_SKILL_NAME} must be mandatory — its violation produces "
            f"silent treatment-column substitution which invalidates the "
            f"causal estimate (F-3b11-DE-MISSING-TREATMENT-COLUMN)"
        )

    def test_skill_priority_is_high(
        self, registry: SkillRegistry
    ) -> None:
        """priority=1 = highest; consistent with sibling mandatory causal
        skills (G2, G3, G5, D1, M1-M5 are all priority 1)."""
        skill = registry.get(NEW_SKILL_NAME)
        assert skill is not None
        assert skill.priority == 1

    def test_skill_references_resolve(
        self, registry: SkillRegistry
    ) -> None:
        """references_skills must resolve to loaded skills (no dangling refs)."""
        skill = registry.get(NEW_SKILL_NAME)
        assert skill is not None
        for ref in skill.references_skills:
            assert registry.get(ref) is not None, (
                f"{NEW_SKILL_NAME} references {ref!r} but that skill does "
                f"not load — dangling reference"
            )

    def test_skill_references_expected_companions(
        self, registry: SkillRegistry
    ) -> None:
        """Per §12.1 frontmatter spec: references G2 (estimand) + D1
        (encoded-column lookup)."""
        skill = registry.get(NEW_SKILL_NAME)
        assert skill is not None
        refs = set(skill.references_skills)
        assert "causal-estimand-definition" in refs, (
            "must reference G2 — the estimand declaration depends on the "
            "treatment being correctly identified"
        )
        assert "hsls09-causal-conventions" in refs, (
            "must reference D1 — its resolve_encoded_columns rule is "
            "downstream of the carve-out"
        )

    def test_skill_body_contains_carve_out_recipe(
        self, registry: SkillRegistry
    ) -> None:
        """The prescriptive Python recipe must be present in the body so
        that DE LLMs receive the literal carve-out shape."""
        skill = registry.get(NEW_SKILL_NAME)
        assert skill is not None
        assert "causal_soo_carve_out" in skill.body, (
            "the prescriptive Python recipe must appear in the SKILL.md body"
        )

    def test_skill_body_cites_failure_mode(
        self, registry: SkillRegistry
    ) -> None:
        """The body must cite F-3b11-DE-MISSING-TREATMENT-COLUMN by ID
        so that future debug sessions can trace the failure to its
        documented source."""
        skill = registry.get(NEW_SKILL_NAME)
        assert skill is not None
        assert "F-3b11-DE-MISSING-TREATMENT-COLUMN" in skill.body


# ---------------------------------------------------------------------------
# §12.4 — matcher returns the new skill at DE + Analyst stages
# ---------------------------------------------------------------------------


class TestCausalDataEngineerContractMatching:
    def test_matches_at_dataengineer_for_causal_soo(
        self, registry: SkillRegistry
    ) -> None:
        from src.orchestrator import _resolve_skill_caps

        caps = _resolve_skill_caps("causal_soo")
        matched = registry.match(
            stage="DataEngineer",
            task_type="causal_soo",
            dataset="hsls09_public",
            top_k_per_layer=caps,
        )
        names = {s.name for s in matched}
        assert NEW_SKILL_NAME in names, (
            f"{NEW_SKILL_NAME} did not match at DataEngineer stage for "
            f"causal_soo. Matched skills: {sorted(names)}"
        )

    def test_matches_at_analyst_for_causal_soo(
        self, registry: SkillRegistry
    ) -> None:
        from src.orchestrator import _resolve_skill_caps

        caps = _resolve_skill_caps("causal_soo")
        matched = registry.match(
            stage="Analyst",
            task_type="causal_soo",
            dataset="hsls09_public",
            top_k_per_layer=caps,
        )
        names = {s.name for s in matched}
        assert NEW_SKILL_NAME in names, (
            f"{NEW_SKILL_NAME} did not match at Analyst stage for "
            f"causal_soo. Matched skills: {sorted(names)}"
        )

    def test_does_not_match_at_dataengineer_for_prediction(
        self, registry: SkillRegistry
    ) -> None:
        """Regression: the new skill is causal_soo-scoped; prediction-task
        DE runs must not see it (would waste prompt budget + irrelevant)."""
        matched = registry.match(
            stage="DataEngineer",
            task_type="prediction",
            dataset="hsls09_public",
            top_k_per_layer={"task-type": 20, "dataset": 20, "methodology": 20, "writing": 20},
        )
        names = {s.name for s in matched}
        assert NEW_SKILL_NAME not in names, (
            f"{NEW_SKILL_NAME} leaked into prediction-task match; the "
            f"applicable_task_types filter is broken"
        )

    def test_mandatory_at_dataengineer_for_causal_soo(
        self, registry: SkillRegistry
    ) -> None:
        """Per §12.3.1: the new skill is the ONLY mandatory at DE for
        causal_soo (other V3.0 mandatory skills don't apply at DE)."""
        from src.orchestrator import _resolve_skill_caps

        caps = _resolve_skill_caps("causal_soo")
        matched = registry.match(
            stage="DataEngineer",
            task_type="causal_soo",
            dataset="hsls09_public",
            top_k_per_layer=caps,
        )
        mandatory_de = {
            s.name for s in matched if s.rule_severity == "mandatory"
        }
        assert NEW_SKILL_NAME in mandatory_de, (
            f"{NEW_SKILL_NAME} not mandatory at DE under causal_soo; "
            f"got mandatory matches: {sorted(mandatory_de)}"
        )


# ---------------------------------------------------------------------------
# §12.4 — V3.0 causal skill count: 11 → 12
# ---------------------------------------------------------------------------


class TestV3CausalSkillCount:
    def test_v3_causal_skill_count_is_twelve(
        self, registry: SkillRegistry
    ) -> None:
        """3b.12 brings the V3.0 causal-skill count to 12.

        Pre-3b.12 inventory (11): G1-G5 + D1 + M1-M5 — see
        docs/v3_0_causal_skill_specification.md §4.5.

        Post-3b.12 (12): the 11 above + causal-data-engineer-contract.
        """
        causal_skills = [
            s for s in registry.all()
            if "causal_soo" in s.applicable_task_types
        ]
        names = sorted(s.name for s in causal_skills)
        # The exact 12 expected (locks against accidental count drift).
        expected_twelve = {
            "causal-dag-identification",                     # G1
            "causal-estimand-definition",                    # G2
            "causal-positivity-diagnostics",                 # G3
            "causal-balance-diagnostics",                    # G4
            "causal-sensitivity-unmeasured-confounding",     # G5
            "hsls09-causal-conventions",                     # D1
            "causal-regression-adjustment",                  # M1
            "causal-propensity-score-matching",              # M2
            "causal-inverse-probability-weighting",          # M3
            "causal-aipw-tmle",                              # M4
            "causal-forest-cate",                            # M5
            "causal-data-engineer-contract",                 # 3b.12 new
        }
        actual = set(names)
        assert actual == expected_twelve, (
            f"V3.0 causal-skill inventory drift:\n"
            f"  expected (12): {sorted(expected_twelve)}\n"
            f"  actual ({len(actual)}): {sorted(actual)}\n"
            f"  missing: {sorted(expected_twelve - actual)}\n"
            f"  unexpected: {sorted(actual - expected_twelve)}"
        )


# ---------------------------------------------------------------------------
# §12.4 — composition graph stays acyclic after the new skill lands
# ---------------------------------------------------------------------------


class TestCompositionGraphAcyclicity:
    def test_full_graph_is_acyclic_post_3b12(
        self, registry: SkillRegistry
    ) -> None:
        """DFS color-marking; same pattern as test_v3_0_causal_skills.
        The new skill references G2 + D1 — no chain back to itself."""
        skills = {s.name: s for s in registry.all()}
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {name: WHITE for name in skills}

        def visit(node: str, path: list[str]) -> None:
            color[node] = GRAY
            for ref in skills[node].references_skills:
                if ref not in skills:
                    continue
                if color[ref] == GRAY:
                    cycle_start = path.index(ref) if ref in path else 0
                    cycle = path[cycle_start:] + [ref]
                    pytest.fail(
                        f"composition cycle detected post-3b.12: "
                        f"{' -> '.join(cycle)}"
                    )
                if color[ref] == WHITE:
                    visit(ref, path + [ref])
            color[node] = BLACK

        for name in skills:
            if color[name] == WHITE:
                visit(name, [name])
