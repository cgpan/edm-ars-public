"""V3.0 Phase 3b.14 — G5 DoWhy refuter invocation amendment.

Sub-wave 2 isolated tests confirming:
  - G5 SKILL.md content carries the 3b.14 amendment markers
    (prescriptive section additions; not just rendered-prompt
    verification).
  - G5's frontmatter is unchanged (loadability regression).
  - G5 is still mandatory at Analyst, Critic, Writer for causal_soo.
  - Composition graph remains acyclic; V3.0 causal skill count is
    still 12 (no new skill in 3b.14).

The rendered-prompt verification lives in
``test_rendered_prompt_contains_all_mskills.py::TestRenderedPromptIncludes3b14G5DoWhyAmendment``.
This file is the SKILL.md content-presence + loadability companion.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.skills import SkillRegistry


SKILLS_ROOT = Path(__file__).parent.parent / "skills"
G5_NAME = "causal-sensitivity-unmeasured-confounding"


@pytest.fixture(scope="module")
def registry() -> SkillRegistry:
    return SkillRegistry(SKILLS_ROOT)


# ---------------------------------------------------------------------------
# §14.1 — amendment content present in G5 SKILL.md body
# ---------------------------------------------------------------------------


class TestG5DoWhyAmendmentContentPresent:
    """The amendment is a structural addition to G5's body. Content-
    presence checks pin the prescriptive section so a future edit
    that drops it gets caught by CI rather than at the next live run."""

    @pytest.mark.parametrize("marker", [
        # The new section header
        "DoWhy refuter invocation (mandatory for causal_soo)",
        # The mandatory rule
        "DAG node names MUST match the column names",
        # The 3b.13 evidence citation
        "F-3b13-DOWHY-REFUTERS-GRAPH-FORMAT",
        # The four-step sequence — each step
        "build_dowhy_graph",
        "CausalModel",
        "identify_effect",
        "estimate_effect",
        "refute_estimate",
        # The output-schema key
        "dowhy_refuters",
        # The exception-handling guidance
        '"status": "failed"',
        # The Writer interpretation guidance section
        "Interpretation in the paper",
    ])
    def test_marker_present_in_g5_body(
        self, registry: SkillRegistry, marker: str
    ) -> None:
        g5 = registry.get(G5_NAME)
        assert g5 is not None, f"{G5_NAME} did not load"
        assert marker in g5.body, (
            f"3b.14 amendment marker {marker!r} missing from G5 "
            f"SKILL.md body. The amendment has been partially or fully "
            f"reverted; F-3b13-DOWHY-REFUTERS-GRAPH-FORMAT will recur. "
            f"(Note: the 3b.16 refinement preserves all 3b.14 markers; "
            f"this test still applies.)"
        )


# ---------------------------------------------------------------------------
# §16.1 — 3b.16 refinement content (NetworkX-DiGraph) is present
# ---------------------------------------------------------------------------


class TestG5DoWhy3b16NetworkXRefinement:
    """3b.16 refined the 3b.14 amendment's graph format from DOT-string
    to NetworkX-DiGraph after F-3b15-DOWHY-PYGRAPHVIZ-DEPENDENCY-GAP.
    Content-presence checks for the refinement, paired with absence-
    checks for the contradictory DOT-string prose 3b.16 removed."""

    @pytest.mark.parametrize("marker", [
        # NetworkX return type
        "nx.DiGraph",
        # NetworkX function-body idiom
        "g.add_edge",
        # 3b.15 failure-mode citation
        "F-3b15-DOWHY-PYGRAPHVIZ-DEPENDENCY-GAP",
        # The reason for the switch
        "pygraphviz",
        # Variable rename in the invocation sequence
        "graph_nx",
    ])
    def test_3b16_marker_present_in_g5_body(
        self, registry: SkillRegistry, marker: str
    ) -> None:
        g5 = registry.get(G5_NAME)
        assert g5 is not None, f"{G5_NAME} did not load"
        assert marker in g5.body, (
            f"3b.16 refinement marker {marker!r} missing from G5 "
            f"SKILL.md body. The NetworkX-DiGraph refinement has been "
            f"partially reverted; F-3b15-DOWHY-PYGRAPHVIZ-DEPENDENCY-GAP "
            f"will recur on the next live run."
        )

    def test_old_3b14_dot_string_prose_is_removed(
        self, registry: SkillRegistry
    ) -> None:
        """The 3b.14 sentence 'DOT strings are the safer default' was
        the OPPOSITE of the 3b.16 prescription. It MUST NOT remain in
        the body, or the LLM gets contradictory guidance.
        """
        g5 = registry.get(G5_NAME)
        assert g5 is not None
        assert "DOT strings are the safer default" not in g5.body, (
            "The 3b.14 'DOT strings are the safer default' prose still "
            "appears in G5's body. The 3b.16 refinement is incomplete "
            "— this prescription contradicts the NetworkX-DiGraph rule."
        )

    def test_old_3b14_graph_dot_variable_is_removed(
        self, registry: SkillRegistry
    ) -> None:
        """The 3b.14 sequence used a `graph_dot` variable. 3b.16 renamed
        it to `graph_nx`. The old name MUST NOT remain in the body, or
        the LLM may copy the half-refactored pattern (call the NetworkX
        builder, then bind result to a `graph_dot` variable).
        """
        g5 = registry.get(G5_NAME)
        assert g5 is not None
        assert "graph_dot" not in g5.body, (
            "The 3b.14 `graph_dot` variable name still appears in G5's "
            "body. The 3b.16 refinement is incomplete — it should be "
            "renamed to `graph_nx` throughout the invocation sequence."
        )


# ---------------------------------------------------------------------------
# §14.3 — G5 loadability regression (frontmatter unchanged)
# ---------------------------------------------------------------------------


class TestG5FrontmatterUnchangedPost3b14:
    """3b.14 is a body-only amendment. G5's frontmatter must be
    byte-identical to its post-3b.8 state: same layer, stages, task
    types, references, priority, severity."""

    def test_g5_layer_is_methodology(
        self, registry: SkillRegistry
    ) -> None:
        g5 = registry.get(G5_NAME)
        assert g5 is not None
        assert g5.layer == "methodology"

    def test_g5_applies_to_analyst_critic_writer(
        self, registry: SkillRegistry
    ) -> None:
        g5 = registry.get(G5_NAME)
        assert g5 is not None
        # Same as post-3b.8: Analyst + Critic + Writer (no DE; G5 is
        # post-estimation diagnostic content).
        assert "Analyst" in g5.applicable_stages
        assert "Critic" in g5.applicable_stages
        assert "Writer" in g5.applicable_stages
        # DE must NOT be added by 3b.14 — that would be scope creep.
        assert "DataEngineer" not in g5.applicable_stages, (
            "3b.14 must not add DataEngineer to G5's stages; the "
            "amendment is body-only. DE-stage attachment would be a "
            "scope-creep regression."
        )

    def test_g5_severity_still_mandatory(
        self, registry: SkillRegistry
    ) -> None:
        g5 = registry.get(G5_NAME)
        assert g5 is not None
        assert g5.rule_severity == "mandatory"

    def test_g5_applies_to_causal_soo(
        self, registry: SkillRegistry
    ) -> None:
        g5 = registry.get(G5_NAME)
        assert g5 is not None
        assert "causal_soo" in g5.applicable_task_types


# ---------------------------------------------------------------------------
# §14.3 — invariants that must hold post-3b.14
# ---------------------------------------------------------------------------


class TestPost3b14Invariants:
    """3b.14 is a body-only amendment with no new skill and no
    frontmatter changes. The following counts and graph properties
    must hold byte-identical to the post-3b.12 state."""

    def test_v3_causal_skill_count_still_twelve(
        self, registry: SkillRegistry
    ) -> None:
        """No new skill in 3b.14. Total V3.0 causal skills: 12
        (the 3b.12 count)."""
        causal_skills = [
            s for s in registry.all()
            if "causal_soo" in s.applicable_task_types
        ]
        assert len(causal_skills) == 12, (
            f"V3.0 causal-skill count changed in 3b.14: expected 12 "
            f"(post-3b.12 count), got {len(causal_skills)}. "
            f"3b.14 must be body-only; no new skill."
        )

    def test_full_composition_graph_still_acyclic(
        self, registry: SkillRegistry
    ) -> None:
        """The amendment is body-only; references_skills is unchanged
        so the graph cannot have cycles introduced. Sanity check."""
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
                        f"composition cycle detected post-3b.14: "
                        f"{' -> '.join(cycle)}"
                    )
                if color[ref] == WHITE:
                    visit(ref, path + [ref])
            color[node] = BLACK

        for name in skills:
            if color[name] == WHITE:
                visit(name, [name])
