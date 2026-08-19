"""V3.0 Phase 3b.18 — D1 registry-type-aware encoding amendment.

Sub-wave 2 isolated tests confirming:
  - D1 SKILL.md body carries the 3b.18 amendment markers.
  - The 3b.6 ``resolve_encoded_columns`` rule is preserved verbatim.
  - D1's frontmatter is unchanged (loadability regression).
  - D1 is still mandatory at the stages it applied to pre-3b.18.
  - V3.0 causal skill count is still 12; composition graph still acyclic.

The rendered-prompt verification lives in
``test_rendered_prompt_contains_all_mskills.py::TestRenderedPromptIncludes3b18D1EncodingAmendment``.
This file is the SKILL.md content-presence + loadability companion,
matching the 3b.14 / 3b.16 pattern.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.skills import SkillRegistry


SKILLS_ROOT = Path(__file__).parent.parent / "skills"
D1_NAME = "hsls09-causal-conventions"


@pytest.fixture(scope="module")
def registry() -> SkillRegistry:
    return SkillRegistry(SKILLS_ROOT)


# ---------------------------------------------------------------------------
# §18.1 — amendment content present in D1 SKILL.md body
# ---------------------------------------------------------------------------


class TestD1EncodingAmendmentContentPresent:
    """The amendment is a structural addition to D1's body. Content-
    presence checks pin the prescriptive section so a future edit
    that drops it gets caught by CI rather than at the next live run."""

    @pytest.mark.parametrize("marker", [
        # New section header
        "Encoding-type discipline (mandatory for DataEngineer)",
        # Mandatory dispatch rule (continuous branch)
        "type=continuous",
        "MUST NOT one-hot",
        # Categorical/binary branch
        "type=categorical",
        # Prescriptive Python recipe identifier
        "encode_for_causal_soo",
        # Function parameter name (the recipe takes registry-as-parameter)
        "variable_registry",
        # Failure-mode citation by ID
        "F-3b15-DE-CONTINUOUS-AS-CATEGORICAL",
        # Concrete examples (anchor variables the LLM has seen mis-encoded)
        "X1MTHUTI",
        "X1MTHID",
        # Cross-reference to the 3b.12 carve-out rule (upstream)
        "causal-data-engineer-contract",
    ])
    def test_marker_present_in_d1_body(
        self, registry: SkillRegistry, marker: str
    ) -> None:
        d1 = registry.get(D1_NAME)
        assert d1 is not None, f"{D1_NAME} did not load"
        assert marker in d1.body, (
            f"3b.18 amendment marker {marker!r} missing from D1 "
            f"SKILL.md body. The amendment has been partially or "
            f"fully reverted; F-3b15-DE-CONTINUOUS-AS-CATEGORICAL "
            f"will recur on the next live run."
        )

    def test_d1_preserves_3b6_resolve_encoded_columns_rule(
        self, registry: SkillRegistry
    ) -> None:
        """Regression: the 3b.6 Analyst-side `resolve_encoded_columns`
        rule must still be present in D1's body. The 3b.18 amendment
        is the *DataEngineer-side* encoding rule; the 3b.6 rule is the
        complementary *Analyst-side* read after encoding has happened.
        Both rules are needed."""
        d1 = registry.get(D1_NAME)
        assert d1 is not None
        assert "resolve_encoded_columns" in d1.body, (
            "The 3b.6 Analyst-side `resolve_encoded_columns` rule is "
            "missing from D1's body. 3b.18 should be ADDITIVE to D1, "
            "not a rewrite — the 3b.6 rule must be preserved."
        )

    def test_d1_examples_match_registry_continuous_types(
        self, registry: SkillRegistry
    ) -> None:
        """The amendment lists concrete continuous variables as
        examples. These should be variables the registry actually
        tags `type=continuous`. If a future registry edit retags one
        of these as categorical, the SKILL example becomes wrong and
        the rule's enforcement breaks."""
        import yaml
        registry_path = (
            Path(__file__).parent.parent / "data_registry" / "datasets"
            / "hsls09_public.yaml"
        )
        with open(registry_path, encoding="utf-8") as f:
            reg = yaml.safe_load(f)

        # Flatten predictors + outcomes into a single name -> entry map
        var_map = {}
        for outcome_var in reg.get("variables", {}).get("outcomes", []):
            var_map[outcome_var["name"]] = outcome_var
        for category, var_list in reg.get("variables", {}).get(
            "predictors", {}
        ).items():
            for v in var_list:
                var_map[v["name"]] = v

        # Variables D1's 3b.18 amendment lists as continuous examples
        d1_continuous_examples = [
            "X1MTHEFF", "X1MTHID", "X1MTHINT", "X1MTHUTI",
            "X1SCIID", "X1SCHOOLBEL", "X1SES", "X1TXMTSCOR",
        ]
        for var in d1_continuous_examples:
            entry = var_map.get(var)
            assert entry is not None, (
                f"D1's 3b.18 amendment lists {var!r} as a continuous "
                f"example, but it is not in the variable registry. "
                f"Update the example list or fix the registry."
            )
            assert entry.get("type") == "continuous", (
                f"D1's 3b.18 amendment lists {var!r} as continuous, "
                f"but the registry tags it as {entry.get('type')!r}. "
                f"The example list and the registry must agree, or "
                f"the amendment's dispatch logic fails for this "
                f"variable."
            )


# ---------------------------------------------------------------------------
# §18.3 — D1 loadability regression (frontmatter unchanged)
# ---------------------------------------------------------------------------


class TestD1FrontmatterUnchangedPost3b18:
    """3b.18 is a body-only amendment. D1's frontmatter must be
    byte-identical to its pre-3b.18 state."""

    def test_d1_layer_is_dataset(
        self, registry: SkillRegistry
    ) -> None:
        d1 = registry.get(D1_NAME)
        assert d1 is not None
        assert d1.layer == "dataset"

    def test_d1_applies_to_de_and_analyst(
        self, registry: SkillRegistry
    ) -> None:
        """D1's pre-3b.18 stages must remain. The 3b.18 amendment is
        most directly about DataEngineer behavior, but D1's existing
        attachment to Analyst (and other stages) is unchanged.
        """
        d1 = registry.get(D1_NAME)
        assert d1 is not None
        assert "DataEngineer" in d1.applicable_stages, (
            "3b.18 amendment depends on D1 reaching the DataEngineer's "
            "rendered prompt. If D1 is no longer attached to DE, the "
            "amendment is unreachable from the LLM."
        )
        assert "Analyst" in d1.applicable_stages, (
            "D1 must remain attached to Analyst (the 3b.6 "
            "resolve_encoded_columns rule is Analyst-side)."
        )

    def test_d1_severity_still_mandatory(
        self, registry: SkillRegistry
    ) -> None:
        d1 = registry.get(D1_NAME)
        assert d1 is not None
        assert d1.rule_severity == "mandatory"

    def test_d1_applies_to_causal_soo(
        self, registry: SkillRegistry
    ) -> None:
        d1 = registry.get(D1_NAME)
        assert d1 is not None
        assert "causal_soo" in d1.applicable_task_types

    def test_d1_applies_to_hsls09(
        self, registry: SkillRegistry
    ) -> None:
        """D1 is the HSLS-specific dataset skill; its
        applicable_datasets must include hsls09_public.
        """
        d1 = registry.get(D1_NAME)
        assert d1 is not None
        assert "hsls09_public" in d1.applicable_datasets


# ---------------------------------------------------------------------------
# §18.3 — invariants that must hold post-3b.18
# ---------------------------------------------------------------------------


class TestPost3b18Invariants:
    """3b.18 is body-only with no new skill, no frontmatter changes.
    Cross-skill invariants must hold byte-identical to the post-3b.16
    state."""

    def test_v3_causal_skill_count_still_twelve(
        self, registry: SkillRegistry
    ) -> None:
        """No new skill in 3b.18. Total V3.0 causal skills: 12."""
        causal_skills = [
            s for s in registry.all()
            if "causal_soo" in s.applicable_task_types
        ]
        assert len(causal_skills) == 12, (
            f"V3.0 causal-skill count changed in 3b.18: expected 12, "
            f"got {len(causal_skills)}. 3b.18 must be body-only; no "
            f"new skill."
        )

    def test_full_composition_graph_still_acyclic(
        self, registry: SkillRegistry
    ) -> None:
        """3b.18 amendment is body-only; references_skills is
        unchanged, so the graph cannot have cycles introduced. Sanity
        check."""
        skills = {s.name: s for s in registry.all()}
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
                        f"composition cycle detected post-3b.18: "
                        f"{' -> '.join(cycle)}"
                    )
                if color[ref] == WHITE:
                    visit(ref, path + [ref])
            color[node] = BLACK

        for name in skills:
            if color[name] == WHITE:
                visit(name, [name])
