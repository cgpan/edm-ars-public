"""V3.0 Phase 3b.8 / §6.2 — M1–M5 promoted to mandatory tier.

Locks the post-3b.8 mandatory inventory at the Analyst stage for
task_type=causal_soo. The 3b.7 report's F-3b7-FORMATTER-TRUNCATES-
METHOD-SKILLS finding explained why M-skills were dropped silently
under the legacy uniform-cap rule. The §6.1 fix protects mandatory
from drops; this §6.2 promotion ensures M-skill bodies are tagged
mandatory so the §6.1 protection applies.

The promotion is unconditional in the SKILL.md frontmatter (M-skills
already declare applicable_task_types=[causal_soo], so they only
get matched in causal contexts anyway).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.skills import SkillRegistry


SKILLS_ROOT = Path(__file__).parent.parent / "skills"

EXPECTED_M_SKILLS_MANDATORY: frozenset[str] = frozenset({
    "causal-regression-adjustment",            # M1
    "causal-propensity-score-matching",        # M2
    "causal-inverse-probability-weighting",    # M3
    "causal-aipw-tmle",                        # M4
    "causal-forest-cate",                      # M5
})

EXPECTED_G_AND_D_MANDATORY: frozenset[str] = frozenset({
    "causal-estimand-definition",              # G2
    "causal-positivity-diagnostics",           # G3
    "causal-sensitivity-unmeasured-confounding",  # G5
    "hsls09-causal-conventions",               # D1
})

EXPECTED_RECOMMENDED_CAUSAL_SKILLS: frozenset[str] = frozenset({
    "causal-dag-identification",               # G1
    "causal-balance-diagnostics",              # G4
})

# V3.0 Phase 3b.12 / §12.3.1 — new mandatory skill addressing
# F-3b11-DE-MISSING-TREATMENT-COLUMN. Bumps the post-3b.8 inventory
# from 9 → 10 at Analyst stage and seeds 1 mandatory at DE stage.
EXPECTED_3B12_NEW_MANDATORY: frozenset[str] = frozenset({
    "causal-data-engineer-contract",
})


@pytest.fixture(scope="module")
def registry() -> SkillRegistry:
    return SkillRegistry(SKILLS_ROOT)


# ---------------------------------------------------------------------------
# §6.2 acceptance — M-skill mandatory tags
# ---------------------------------------------------------------------------


class TestMSkillMandatoryPromotion:
    @pytest.mark.parametrize("skill_id", sorted(EXPECTED_M_SKILLS_MANDATORY))
    def test_each_m_skill_is_mandatory(
        self, registry: SkillRegistry, skill_id: str
    ) -> None:
        skill = registry.get(skill_id)
        assert skill is not None, f"Skill {skill_id!r} not loaded"
        assert skill.rule_severity == "mandatory", (
            f"{skill_id} expected mandatory after 3b.8 promotion, "
            f"got {skill.rule_severity}"
        )

    def test_g_and_d_mandatory_unchanged(
        self, registry: SkillRegistry
    ) -> None:
        """Regression: G2/G3/G5/D1 stay mandatory."""
        for skill_id in EXPECTED_G_AND_D_MANDATORY:
            skill = registry.get(skill_id)
            assert skill is not None
            assert skill.rule_severity == "mandatory", (
                f"{skill_id} mandatory tag changed unexpectedly"
            )

    def test_g1_g4_remain_recommended(
        self, registry: SkillRegistry
    ) -> None:
        """3b.8 deliberately did NOT promote G1 (DAG) or G4 (balance) —
        their original 3b.1 audit rationale stands. Verify they kept
        the recommended tier.
        """
        for skill_id in EXPECTED_RECOMMENDED_CAUSAL_SKILLS:
            skill = registry.get(skill_id)
            assert skill is not None
            assert skill.rule_severity == "recommended", (
                f"{skill_id} should remain recommended; got "
                f"{skill.rule_severity}. Promoting G1/G4 in 3b.8 "
                f"would be a separate spec decision (deliberately "
                f"out of scope per the hand-off)."
            )


# ---------------------------------------------------------------------------
# §6.2 acceptance — mandatory inventory at Analyst stage for causal_soo
# ---------------------------------------------------------------------------


class TestMandatoryInventoryForCausalSOO:
    def test_at_least_nine_mandatory_skills_at_analyst_for_causal_soo(
        self, registry: SkillRegistry
    ) -> None:
        """Post-3b.8 mandatory baseline: 9 causal-specific skills.

        G2, G3, G5, D1 (the 3b.1 mandatory four) plus M1, M2, M3, M4, M5
        (the 3b.8 promotions). All five M-skills declare applicable_
        task_types=[causal_soo] and applicable_stages=[Analyst], so they
        match at Analyst stage for causal_soo and only there.

        Phase 3b.12 added a 10th — see
        ``test_post_3b12_inventory_includes_de_contract_at_analyst``.
        """
        from src.orchestrator import _resolve_skill_caps

        caps = _resolve_skill_caps("causal_soo")
        matched = registry.match(
            stage="Analyst",
            task_type="causal_soo",
            dataset="hsls09_public",
            top_k_per_layer=caps,
        )
        mandatory_matched = {
            s.name for s in matched if s.rule_severity == "mandatory"
        }
        # Spec contract: the 9 causal-specific mandatory skills are
        # present. Other generic mandatory skills (e.g., subgroup-
        # fairness-analysis) may also match at Analyst stage; assert
        # OUR 9 are present, not that the total is exactly 9.
        causal_mandatory_subset = (
            EXPECTED_M_SKILLS_MANDATORY | EXPECTED_G_AND_D_MANDATORY
        )
        assert len(causal_mandatory_subset) == 9
        missing = causal_mandatory_subset - mandatory_matched
        assert not missing, (
            f"Causal mandatory inventory incomplete. Missing: "
            f"{sorted(missing)}. Got mandatory matches: "
            f"{sorted(mandatory_matched)}"
        )

    def test_post_3b12_inventory_includes_de_contract_at_analyst(
        self, registry: SkillRegistry
    ) -> None:
        """Phase 3b.12 / §12.3.1 — bump mandatory inventory to 10 at
        Analyst by adding ``causal-data-engineer-contract``. Locks the
        new skill into the Analyst's mandatory set so it cannot be
        accidentally dropped by a future formatter / matcher change."""
        from src.orchestrator import _resolve_skill_caps

        caps = _resolve_skill_caps("causal_soo")
        matched = registry.match(
            stage="Analyst",
            task_type="causal_soo",
            dataset="hsls09_public",
            top_k_per_layer=caps,
        )
        mandatory_matched = {
            s.name for s in matched if s.rule_severity == "mandatory"
        }
        # Post-3b.12: 9 (post-3b.8) + 1 (causal-data-engineer-contract) = 10.
        causal_mandatory_subset_post_3b12 = (
            EXPECTED_M_SKILLS_MANDATORY
            | EXPECTED_G_AND_D_MANDATORY
            | EXPECTED_3B12_NEW_MANDATORY
        )
        assert len(causal_mandatory_subset_post_3b12) == 10
        missing = causal_mandatory_subset_post_3b12 - mandatory_matched
        assert not missing, (
            f"Post-3b.12 causal mandatory inventory missing: "
            f"{sorted(missing)}. Got mandatory matches: "
            f"{sorted(mandatory_matched)}"
        )

    def test_post_3b12_inventory_includes_de_contract_at_dataengineer(
        self, registry: SkillRegistry
    ) -> None:
        """Phase 3b.12 / §12.3.1 — the new skill also registers at
        DataEngineer stage. Other V3.0 mandatory skills (G2/G3/G5/D1/
        M1-M5) do NOT apply at DE per their applicable_stages.
        """
        from src.orchestrator import _resolve_skill_caps

        caps = _resolve_skill_caps("causal_soo")
        matched = registry.match(
            stage="DataEngineer",
            task_type="causal_soo",
            dataset="hsls09_public",
            top_k_per_layer=caps,
        )
        mandatory_at_de = {
            s.name for s in matched if s.rule_severity == "mandatory"
        }
        # The new skill must be in the mandatory set at DE.
        assert "causal-data-engineer-contract" in mandatory_at_de, (
            f"causal-data-engineer-contract not mandatory at DE under "
            f"causal_soo; got {sorted(mandatory_at_de)}"
        )
        # And the M-skills must NOT leak to DE (they're Analyst-only).
        leaked_m = EXPECTED_M_SKILLS_MANDATORY & mandatory_at_de
        assert not leaked_m, (
            f"M-skills leaked to DataEngineer stage: {sorted(leaked_m)}. "
            f"M-skills declare applicable_stages=[Analyst] and should "
            f"not appear at DE."
        )

    def test_m_skills_NOT_mandatory_for_prediction_task(
        self, registry: SkillRegistry
    ) -> None:
        """Sanity: M-skills declare applicable_task_types=[causal_soo] only.
        For task_type=prediction they should not appear in the matched
        list at all — neither as mandatory nor as recommended.
        """
        from src.orchestrator import _resolve_skill_caps

        caps = _resolve_skill_caps("prediction")
        matched = registry.match(
            stage="Analyst",
            task_type="prediction",
            dataset="hsls09_public",
            top_k_per_layer=caps,
        )
        names = {s.name for s in matched}
        leaked = EXPECTED_M_SKILLS_MANDATORY & names
        assert not leaked, (
            f"M-skills leaked into prediction match after 3b.8 "
            f"promotion: {sorted(leaked)}. The promotion to mandatory "
            f"should not have changed task-type applicability."
        )
