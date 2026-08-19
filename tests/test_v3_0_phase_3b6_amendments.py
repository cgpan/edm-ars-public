"""V3.0 Phase 3b.6 sub-wave 2 — verify the four skill/prompt amendments
landed in their target files.

These are content-presence tests; behavioral verification of the new
guidance is deferred to 3b.7's LLM-call observation.

Tests cover:
  6.4 — D1 + M1-M4 contain the encoded-column-lookup rule + cross-ref.
  6.5 — M4 contains cluster-aware IF variance + sanity-check rule.
  6.6 — PF causal_soo prompt contains the locked-spec invariants section.
  6.7 — subgroup-fairness skill contains the causal-mode branch.
"""
from __future__ import annotations

from pathlib import Path

import pytest


SKILLS_ROOT = Path(__file__).parent.parent / "skills"
PROMPTS_ROOT = Path(__file__).parent.parent / "agent_prompts"


def _read_skill_body(skill_id: str) -> str:
    """Locate skill SKILL.md across known layers and return body text."""
    for layer_dir in (SKILLS_ROOT / "dataset", SKILLS_ROOT / "methodology",
                      SKILLS_ROOT / "task-type", SKILLS_ROOT / "writing"):
        candidate = layer_dir / skill_id / "SKILL.md"
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
    raise FileNotFoundError(f"SKILL.md not found for {skill_id!r}")


# ---------------------------------------------------------------------------
# 6.4 — D1 + M1-M4 encoded-column-lookup rule
# ---------------------------------------------------------------------------


class TestEncodedColumnLookupRule:
    def test_d1_contains_encoded_column_lookup_rule(self) -> None:
        body = _read_skill_body("hsls09-causal-conventions")
        assert "resolve_encoded_columns" in body, (
            "D1 must define resolve_encoded_columns helper for "
            "F-COVARIATE-SET-MISMATCH prevention"
        )
        # Prefix-match construct must be present in code form.
        assert "startswith(varname" in body, (
            "D1 must show the prefix-match construct in code"
        )
        # Failure-mode-prevented banner.
        assert "F-COVARIATE-SET-MISMATCH" in body

    @pytest.mark.parametrize(
        "skill_id",
        [
            "causal-regression-adjustment",            # M1
            "causal-propensity-score-matching",        # M2
            "causal-inverse-probability-weighting",    # M3
            "causal-aipw-tmle",                        # M4
        ],
    )
    def test_m1_through_m4_reference_d1_lookup(self, skill_id: str) -> None:
        body = _read_skill_body(skill_id)
        # Must reference the D1 helper either by name or by concept.
        has_helper_ref = "resolve_encoded_columns" in body
        has_concept_ref = "Encoded-column lookup" in body
        assert has_helper_ref or has_concept_ref, (
            f"{skill_id} must cross-reference D1's adjustment-set "
            f"resolution rule (helper name or section title)"
        )
        # Pointer to D1 must be explicit.
        assert (
            "hsls09-causal-conventions" in body
        ), f"{skill_id} must point to D1 by skill name"

    def test_m5_NOT_amended(self) -> None:
        """Per the hand-off: 'Do NOT amend M5 in this phase.'"""
        body = _read_skill_body("causal-forest-cate")
        assert "resolve_encoded_columns" not in body, (
            "M5 was deliberately NOT amended in 3b.6 — the M5 skill is "
            "tested as-written in 3b.7 first. Amending it now is "
            "premature."
        )


# ---------------------------------------------------------------------------
# 6.5 — M4 cluster-aware IF + sanity check
# ---------------------------------------------------------------------------


class TestM4ClusterAwareIF:
    def test_m4_specifies_cluster_aware_if_variance(self) -> None:
        body = _read_skill_body("causal-aipw-tmle")
        # Cluster-mean aggregation construct.
        cluster_terms_present = (
            "cluster_means" in body
            or "cluster-level mean" in body.lower()
        )
        assert cluster_terms_present, (
            "M4 must specify cluster-mean aggregation in IF variance"
        )
        # Degrees-of-freedom adjustment.
        assert "n_clusters" in body
        # Sanity-check rule for implausibly narrow CIs.
        sanity_present = (
            "0.5 * median_comparator_se" in body
            or "half the median comparator" in body.lower()
        )
        assert sanity_present, (
            "M4 must include the implausibly-narrow-SE sanity check"
        )

    def test_m4_cites_failure_mode_3b5(self) -> None:
        body = _read_skill_body("causal-aipw-tmle")
        assert "F-AIPW-NARROW-CI" in body, (
            "M4 amendment must cite the 3b.5 failure-mode being "
            "prevented for traceability"
        )


# ---------------------------------------------------------------------------
# 6.6 — PF causal_soo prompt invariants section
# ---------------------------------------------------------------------------


class TestPFLockedSpecInvariants:
    def test_pf_causal_soo_prompt_contains_locked_spec_invariants_section(
        self,
    ) -> None:
        prompt_path = PROMPTS_ROOT / "problem_formulator_causal_soo.yaml"
        body = prompt_path.read_text(encoding="utf-8")
        assert "Locked-spec invariants" in body
        assert "primary_method, comparator_method, secondary_methods" in body
        assert "methodological_concerns" in body
        # Either "byte-identical" or "preserve them" must be present
        # to convey the don't-mutate semantics.
        assert ("byte-identical" in body) or ("preserve them" in body)

    def test_pf_invariants_lists_required_fields(self) -> None:
        prompt_path = PROMPTS_ROOT / "problem_formulator_causal_soo.yaml"
        body = prompt_path.read_text(encoding="utf-8")
        for required_field in (
            "task_id",
            "task_type",
            "treatment.variable",
            "treatment.operationalization",
            "outcome.variable",
            "outcome.type",
        ):
            assert required_field in body, (
                f"Locked-spec invariants section must enumerate "
                f"{required_field!r}"
            )


# ---------------------------------------------------------------------------
# 6.7 — subgroup-fairness causal branch
# ---------------------------------------------------------------------------


class TestSubgroupFairnessCausalBranch:
    def test_subgroup_fairness_has_causal_mode_section(self) -> None:
        body = _read_skill_body("subgroup-fairness-analysis")
        assert "Causal mode" in body
        # Helper-function name OR conceptual description.
        helper_or_concept = (
            "causal_subgroup_analysis" in body
            or "subgroup ATE" in body.lower()
        )
        assert helper_or_concept, (
            "Causal-mode section must name the helper or describe "
            "subgroup-ATE estimation"
        )
        # FDR correction is mandatory per the spec text.
        assert "BH FDR" in body
        # Method-ID list must reference M1-M4.
        for method_id in ("M1", "M2", "M3", "M4"):
            assert method_id in body, (
                f"Causal-mode section must enumerate method-IDs "
                f"(missing {method_id})"
            )

    def test_subgroup_fairness_explicitly_excludes_m5(self) -> None:
        body = _read_skill_body("subgroup-fairness-analysis")
        # The hand-off text says M5 has its own native CATE path; the
        # causal-mode section must call this out so the Analyst doesn't
        # accidentally re-do CATE estimation here.
        m5_carve_out = (
            "M5 (causal-forest-cate)" in body
            or "M5 has native CATE" in body
            or "M5 has its own native" in body
        )
        assert m5_carve_out, (
            "Causal-mode section must carve out M5 — its CATE pathway "
            "lives in the M5 skill body"
        )
