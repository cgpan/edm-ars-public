"""V3.0 Phase 3b.8 / §6.3 — rendered-prompt verification.

The test class that 3b.7 surfaced as missing. Every prior phase's
"matcher returns the right skills" test passed; the actual rendered
Analyst prompt was silently missing the M-skill bodies because the
formatter's uniform cap dropped them. This file asserts directly
against the rendered prompt that all five M-skill bodies appear.

Pattern: any future cleanup phase that amends SKILL.md content for
causal_soo should add a marker string here. If a SKILL.md change is
supposed to affect LLM behavior, a rendered-prompt test must assert
the change is visible to the LLM.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.agents.base import load_prompt
from src.skills import SkillRegistry, format_skills_for_prompt


PROJECT_ROOT = Path(__file__).parent.parent
SKILLS_ROOT = PROJECT_ROOT / "skills"
FIXTURE_PATH = PROJECT_ROOT / "runs" / "fixtures" / "spec_x1mtheff_x4college.json"
_SKILLS_PLACEHOLDER = "{{SKILLS}}"


@pytest.fixture(scope="module")
def registry() -> SkillRegistry:
    return SkillRegistry(SKILLS_ROOT)


@pytest.fixture(scope="module")
def locked_spec() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _render_analyst_prompt_for_causal_soo(
    registry: SkillRegistry,
    research_question: str,
) -> str:
    """Reproduce the orchestrator's Analyst-stage prompt rendering path.

    Steps mirror src/agents/base.py BaseAgent + src/orchestrator.py
    _match_skills_for_stage:
      1. load_prompt('analyst', task_type='causal_soo')
      2. SkillRegistry.match_and_compose(stage=Analyst, ...)
      3. format_skills_for_prompt(matched)
      4. substitute {{SKILLS}} into the system_prompt
    """
    config: dict[str, Any] = {
        "paths": {"agent_prompts": str(PROJECT_ROOT / "agent_prompts") + "/"},
    }
    prompt_data = load_prompt("analyst", config, task_type="causal_soo")
    system_prompt = prompt_data["system_prompt"]

    # Use the same caps the orchestrator uses for causal_soo (post-3b.6).
    from src.orchestrator import _resolve_skill_caps

    caps = _resolve_skill_caps("causal_soo")
    matched = registry.match_and_compose(
        stage="Analyst",
        task_type="causal_soo",
        dataset="hsls09_public",
        context=research_question,
        top_k_per_layer=caps,
    )
    skills_block = format_skills_for_prompt(matched).rstrip()
    return system_prompt.replace(_SKILLS_PLACEHOLDER, skills_block)


def _render_data_engineer_prompt_for_causal_soo(
    registry: SkillRegistry,
    research_question: str,
) -> str:
    """Reproduce the orchestrator's DataEngineer-stage prompt rendering
    path.

    Same shape as the Analyst helper above but for the DE stage.
    DataEngineer has no causal-soo-specific YAML variant; it uses the
    default ``data_engineer.yaml`` (which already carries the
    ``{{SKILLS}}`` placeholder). Per Phase 3b.12, the new
    ``causal-data-engineer-contract`` skill must reach this rendered
    prompt body when ``task_type=causal_soo``.
    """
    config: dict[str, Any] = {
        "paths": {"agent_prompts": str(PROJECT_ROOT / "agent_prompts") + "/"},
    }
    # DE has no causal variant; load the default prompt.
    prompt_data = load_prompt("data_engineer", config, task_type="causal_soo")
    system_prompt = prompt_data["system_prompt"]

    from src.orchestrator import _resolve_skill_caps

    caps = _resolve_skill_caps("causal_soo")
    matched = registry.match_and_compose(
        stage="DataEngineer",
        task_type="causal_soo",
        dataset="hsls09_public",
        context=research_question,
        top_k_per_layer=caps,
    )
    skills_block = format_skills_for_prompt(matched).rstrip()
    return system_prompt.replace(_SKILLS_PLACEHOLDER, skills_block)


# ---------------------------------------------------------------------------
# §6.3 acceptance — M-skill bodies reach the rendered Analyst prompt
# ---------------------------------------------------------------------------


# Marker strings that uniquely identify each M-skill body. Choose
# things specific enough to be unique to that skill — function
# signatures and verbatim threshold values are good choices.
M_SKILL_MARKERS: dict[str, str] = {
    "M1 (regression-adjustment)": "Cook's D > 4/n",
    "M2 (PSM)": "n_control / n_treated < 5",
    "M3 (IPW)": "weights=stabilized_weights",
    "M4 (AIPW+TMLE)": "0.5 * median_comparator_se",
    "M5 (causal forest)": "honest=True",  # mandatory flag named in M5 §3.11
}


class TestRenderedPromptContainsAllMSkillBodies:
    """The regression test for F-3b7-FORMATTER-TRUNCATES-METHOD-SKILLS.

    Each M-skill body has a stable substring that should appear in the
    rendered Analyst prompt for the causal_soo task type. If the body
    is missing, the formatter or matcher dropped it silently — same bug
    class that 3b.7 burned a $30 LSAR run discovering.
    """

    def test_all_five_m_skill_body_markers_present_in_rendered_prompt(
        self, registry: SkillRegistry, locked_spec: dict
    ) -> None:
        rendered = _render_analyst_prompt_for_causal_soo(
            registry,
            research_question=locked_spec["research_question"],
        )
        missing = []
        for label, marker in M_SKILL_MARKERS.items():
            if marker not in rendered:
                missing.append((label, marker))
        assert not missing, (
            "M-skill body markers missing from rendered Analyst prompt — "
            "the same bug class as F-3b7-FORMATTER-TRUNCATES-METHOD-SKILLS:\n"
            + "\n".join(f"  - {label}: marker {marker!r} not found" for label, marker in missing)
        )

    @pytest.mark.parametrize(
        "label,marker",
        [(label, marker) for label, marker in M_SKILL_MARKERS.items()],
    )
    def test_specific_m_skill_body_present(
        self,
        registry: SkillRegistry,
        locked_spec: dict,
        label: str,
        marker: str,
    ) -> None:
        """Per-skill version of the above. Failure here pinpoints which
        M-skill is being dropped."""
        rendered = _render_analyst_prompt_for_causal_soo(
            registry,
            research_question=locked_spec["research_question"],
        )
        assert marker in rendered, (
            f"{label} body marker {marker!r} missing from rendered "
            f"Analyst prompt for causal_soo. The matcher unit test may "
            f"still pass (the skill is in the matched list); the failure "
            f"is in the formatter's per-tier cap. Check 3b.8's §6.1 "
            f"implementation in src/skills/composer.py."
        )


class TestRenderedPromptIncludes3b6Amendments:
    """3b.6's specific amendments must reach the rendered prompt. These
    amendments are the entire point of phases 3b.6 → 3b.7 → 3b.8 →
    3b.9; if they're invisible to the LLM, the cleanup work is
    unobservable.

    If this test had existed before 3b.7, the formatter gap would have
    been caught in 3b.6 verification rather than in 3b.7 LLM execution.
    """

    @pytest.mark.parametrize(
        "label,marker",
        [
            ("D1 encoded-column lookup (6.4)", "resolve_encoded_columns"),
            ("M4 cluster-aware IF (6.5)", "cluster_means"),
            ("subgroup causal mode (6.7)", "causal_subgroup_analysis"),
        ],
    )
    def test_3b6_amendment_visible_to_llm(
        self,
        registry: SkillRegistry,
        locked_spec: dict,
        label: str,
        marker: str,
    ) -> None:
        rendered = _render_analyst_prompt_for_causal_soo(
            registry,
            research_question=locked_spec["research_question"],
        )
        assert marker in rendered, (
            f"{label} amendment from 3b.6 missing from rendered prompt. "
            f"The skill content exists in the SKILL.md file, but the "
            f"formatter or matcher is not exposing it to the LLM. The "
            f"3b.6 work is invisible — same bug class as F-3b7-FORMATTER-"
            f"TRUNCATES-METHOD-SKILLS."
        )


class TestRenderedPromptIncludes3b12DEContract:
    """V3.0 Phase 3b.12 / §12.3.2 — the new
    ``causal-data-engineer-contract`` skill must reach BOTH the Analyst
    and DataEngineer rendered prompts at runtime. The Analyst stage
    needs it so the agent never silently substitutes a proxy for the
    declared treatment; the DataEngineer stage needs it so the carve-
    out actually includes the treatment column in the first place.

    If only one stage gets the rendered body, the contract is one-
    sided — same bug class as F-3b7-FORMATTER-TRUNCATES-METHOD-SKILLS:
    the matcher unit test passes (skill is in the matched list) while
    the formatter or placeholder substitution silently drops it.
    """

    DE_CONTRACT_MARKERS: tuple[str, ...] = (
        "causal_soo_carve_out",                  # the prescriptive Python recipe
        "F-3b11-DE-MISSING-TREATMENT-COLUMN",    # the failure mode it prevents
    )

    def test_de_contract_skill_body_at_analyst(
        self, registry: SkillRegistry, locked_spec: dict
    ) -> None:
        rendered = _render_analyst_prompt_for_causal_soo(
            registry,
            research_question=locked_spec["research_question"],
        )
        for marker in self.DE_CONTRACT_MARKERS:
            assert marker in rendered, (
                f"causal-data-engineer-contract marker {marker!r} "
                f"missing from rendered Analyst prompt for causal_soo. "
                f"The Analyst won't know the contract guarantees the "
                f"treatment column is present and may substitute a "
                f"proxy — F-3b11-DE-MISSING-TREATMENT-COLUMN recurs."
            )

    def test_de_contract_skill_body_at_data_engineer(
        self, registry: SkillRegistry, locked_spec: dict
    ) -> None:
        rendered = _render_data_engineer_prompt_for_causal_soo(
            registry,
            research_question=locked_spec["research_question"],
        )
        for marker in self.DE_CONTRACT_MARKERS:
            assert marker in rendered, (
                f"causal-data-engineer-contract marker {marker!r} "
                f"missing from rendered DataEngineer prompt for "
                f"causal_soo. The DE prompt won't carry the carve-out "
                f"recipe — even with the orchestrator guardrail in "
                f"place, the LLM has no positive guidance, so the "
                f"failure surfaces as a hard abort rather than a "
                f"correctly-shaped train_X.csv."
            )

    def test_de_prompt_has_skills_placeholder(self) -> None:
        """Wire-up sanity: data_engineer.yaml must contain {{SKILLS}}.
        If the placeholder is missing, no skill body — including this
        new mandatory one — reaches the DE LLM regardless of how the
        matcher / formatter behave upstream.
        """
        de_yaml = (PROJECT_ROOT / "agent_prompts" / "data_engineer.yaml").read_text(
            encoding="utf-8"
        )
        assert "{{SKILLS}}" in de_yaml, (
            "agent_prompts/data_engineer.yaml is missing the {{SKILLS}} "
            "placeholder. Without it, NO skill body (including the new "
            "3b.12 causal-data-engineer-contract) reaches the DE LLM. "
            "This is the V2.0.1-pattern wire-up the 3b.12 hand-off "
            "§12.3.3 anticipated — if this test fails, add {{SKILLS}} "
            "after the role/persona block, before task instructions."
        )


class TestRenderedPromptIncludes3b14G5DoWhyAmendment:
    """V3.0 Phase 3b.14 / §14.2 — the G5 (causal-sensitivity-unmeasured-
    confounding) SKILL.md gained a prescriptive "DoWhy refuter invocation"
    section. The amendment must reach the Analyst's rendered prompt so
    the LLM follows the four-step build → CausalModel → identify_effect
    → estimate_effect → refute_estimate sequence (instead of the 3b.13
    failure pattern of building a DOT graph with node aliases T/Y that
    don't match the column-name treatment passed to CausalModel).

    Markers chosen to be unique to the 3b.14 amendment, not shared with
    the existing declarative DoWhy section (which already rendered in
    3b.13 — but didn't carry the prescriptive form).
    """

    G5_DOWHY_INVOCATION_MARKERS: tuple[str, ...] = (
        "build_dowhy_graph",                  # the prescriptive Python recipe
        "DAG node names MUST match",          # the column-name-as-node-ID rule
        "F-3b13-DOWHY-REFUTERS-GRAPH-FORMAT", # the failure mode it prevents
        "dowhy_refuters",                     # the new output schema key
    )

    @pytest.mark.parametrize("marker", list(G5_DOWHY_INVOCATION_MARKERS))
    def test_g5_dowhy_invocation_marker_at_analyst(
        self, registry: SkillRegistry, locked_spec: dict, marker: str
    ) -> None:
        rendered = _render_analyst_prompt_for_causal_soo(
            registry,
            research_question=locked_spec["research_question"],
        )
        assert marker in rendered, (
            f"G5 DoWhy refuter invocation marker {marker!r} missing "
            f"from rendered Analyst prompt for causal_soo. The 3b.14 "
            f"amendment is invisible to the LLM; F-3b13-DOWHY-REFUTERS-"
            f"GRAPH-FORMAT will recur on the next run. Check that the "
            f"G5 SKILL.md amendment is in the file and that the "
            f"formatter is not truncating the new section."
        )

    def test_g5_dowhy_invocation_sequence_visible_at_analyst(
        self, registry: SkillRegistry, locked_spec: dict
    ) -> None:
        """The four-step sequence (build → CausalModel → identify_effect
        → estimate_effect → refute_estimate) must appear in the rendered
        prompt — not just one of the function names. The sequence is the
        prescriptive value-add of the amendment; partial rendering would
        leave the LLM with the same gap 3b.13 had.
        """
        rendered = _render_analyst_prompt_for_causal_soo(
            registry,
            research_question=locked_spec["research_question"],
        )
        ordered_step_markers = (
            "build_dowhy_graph",
            "CausalModel",
            "identify_effect",
            "estimate_effect",
            "refute_estimate",
        )
        for step in ordered_step_markers:
            assert step in rendered, (
                f"DoWhy invocation step {step!r} missing from rendered "
                f"Analyst prompt. The four-step sequence must be present "
                f"in full for the amendment to prevent recurrence of "
                f"F-3b13-DOWHY-REFUTERS-GRAPH-FORMAT."
            )

    def test_g5_exception_handling_guidance_visible_at_analyst(
        self, registry: SkillRegistry, locked_spec: dict
    ) -> None:
        """The exception-handling guidance (per-refuter try/except;
        status: 'failed' record) must reach the LLM. Without it, an
        Analyst whose refuter throws will silently emit refuter_results=[]
        — the exact 3b.13 shape.
        """
        rendered = _render_analyst_prompt_for_causal_soo(
            registry,
            research_question=locked_spec["research_question"],
        )
        # Either the explicit status: "failed" record or the
        # "try ... except" wrapper instruction is acceptable evidence.
        assert (
            '"status": "failed"' in rendered
            or 'status: "failed"' in rendered
            or "try:" in rendered and "except Exception" in rendered
        ), (
            "G5 exception-handling guidance missing from rendered "
            "Analyst prompt. Per-refuter try/except is mandatory "
            "(see 3b.14 amendment)."
        )


class TestRenderedPromptIncludes3b16G5NetworkXRefinement:
    """V3.0 Phase 3b.16 / §16.2 — the G5 DoWhy refuter invocation pattern
    was refined to use NetworkX-DiGraph instead of DOT-string after 3b.15
    surfaced F-3b15-DOWHY-PYGRAPHVIZ-DEPENDENCY-GAP. The refinement must
    reach the Analyst's rendered prompt; if the LLM falls back to DOT-
    string format the F-3b15 failure recurs.

    The 3b.14 markers (build_dowhy_graph, DAG node names MUST match,
    F-3b13 citation, dowhy_refuters, four-step sequence, exception
    handling) remain valid in their existing tests — those rules are
    preserved unchanged by 3b.16. This class adds the *new* markers
    that 3b.16 introduces.
    """

    G5_NETWORKX_MARKERS: tuple[str, ...] = (
        "nx.DiGraph",                            # the new return type
        "g.add_edge",                            # the new function-body pattern
        "F-3b15-DOWHY-PYGRAPHVIZ-DEPENDENCY-GAP",# the new failure-mode citation
        "pygraphviz",                            # the reason for the switch
    )

    @pytest.mark.parametrize("marker", list(G5_NETWORKX_MARKERS))
    def test_g5_networkx_refinement_marker_at_analyst(
        self, registry: SkillRegistry, locked_spec: dict, marker: str
    ) -> None:
        rendered = _render_analyst_prompt_for_causal_soo(
            registry,
            research_question=locked_spec["research_question"],
        )
        assert marker in rendered, (
            f"G5 NetworkX-DiGraph refinement marker {marker!r} missing "
            f"from rendered Analyst prompt. The 3b.16 refinement is "
            f"invisible to the LLM; F-3b15-DOWHY-PYGRAPHVIZ-DEPENDENCY-"
            f"GAP will recur — the Analyst will build a DOT-string graph "
            f"and DoWhy will raise the misleading 'Incorrect format' "
            f"error on identify_effect()."
        )

    def test_g5_old_dot_string_prescription_is_removed(
        self, registry: SkillRegistry, locked_spec: dict
    ) -> None:
        """3b.14 had a prose sentence: 'DOT strings are the safer
        default.' 3b.16 removed it (the opposite is now true). The
        sentence MUST NOT reappear in the rendered prompt or the LLM
        gets contradictory guidance.
        """
        rendered = _render_analyst_prompt_for_causal_soo(
            registry,
            research_question=locked_spec["research_question"],
        )
        assert "DOT strings are the safer default" not in rendered, (
            "The 3b.14 'DOT strings are the safer default' prose still "
            "appears in the rendered prompt. The 3b.16 refinement is "
            "incomplete — this prescription contradicts the NetworkX-"
            "DiGraph rule and must be removed."
        )

    def test_g5_invocation_uses_graph_nx_variable_name(
        self, registry: SkillRegistry, locked_spec: dict
    ) -> None:
        """The four-step sequence's Step 1 / Step 2 should use the
        ``graph_nx`` variable name (NetworkX DiGraph), not ``graph_dot``.
        Detects the half-refactored state where the function returns a
        DiGraph but the call site still names the variable as if it were
        a DOT string."""
        rendered = _render_analyst_prompt_for_causal_soo(
            registry,
            research_question=locked_spec["research_question"],
        )
        assert "graph_nx" in rendered, (
            "The four-step invocation should bind the build_dowhy_graph "
            "return value to `graph_nx` (NetworkX DiGraph). The 3b.16 "
            "refinement renamed the variable from graph_dot."
        )


class TestRenderedPromptIncludes3b18D1EncodingAmendment:
    """V3.0 Phase 3b.18 / §18.2 — the D1 (`hsls09-causal-conventions`)
    SKILL.md gained an "Encoding-type discipline (mandatory for
    DataEngineer)" section. The amendment must reach BOTH the
    DataEngineer's and Analyst's rendered prompts at runtime; the DE
    is where the encoding decision happens, and the Analyst benefits
    from knowing the encoding contract (it complements the existing
    3b.6 `resolve_encoded_columns` rule, which is Analyst-side).

    If only one stage gets the rendered body, the contract is one-
    sided — same risk class as F-3b7-FORMATTER-TRUNCATES-METHOD-SKILLS
    or 3b.12's DE-contract one-sided-attachment concern.
    """

    D1_ENCODING_MARKERS: tuple[str, ...] = (
        "Encoding-type discipline",              # new section header
        "type=continuous",                       # the dispatch keyword
        "encode_for_causal_soo",                 # prescriptive recipe identifier
        "F-3b15-DE-CONTINUOUS-AS-CATEGORICAL",   # failure-mode citation
        "MUST NOT one-hot",                      # the mandatory rule wording
    )

    @pytest.mark.parametrize("marker", list(D1_ENCODING_MARKERS))
    def test_d1_encoding_marker_at_data_engineer(
        self, registry: SkillRegistry, locked_spec: dict, marker: str
    ) -> None:
        rendered = _render_data_engineer_prompt_for_causal_soo(
            registry,
            research_question=locked_spec["research_question"],
        )
        assert marker in rendered, (
            f"D1 encoding-type-discipline marker {marker!r} missing "
            f"from rendered DataEngineer prompt. The 3b.18 amendment "
            f"is invisible to the DE LLM; F-3b15-DE-CONTINUOUS-AS-"
            f"CATEGORICAL will recur on the next live run."
        )

    @pytest.mark.parametrize("marker", list(D1_ENCODING_MARKERS))
    def test_d1_encoding_marker_at_analyst(
        self, registry: SkillRegistry, locked_spec: dict, marker: str
    ) -> None:
        """D1 also attaches at the Analyst stage; the encoding
        contract is informative for the Analyst (it explains the
        column-counting upstream of the 3b.6 prefix-match lookup)."""
        rendered = _render_analyst_prompt_for_causal_soo(
            registry,
            research_question=locked_spec["research_question"],
        )
        assert marker in rendered, (
            f"D1 encoding-type-discipline marker {marker!r} missing "
            f"from rendered Analyst prompt. The Analyst won't know "
            f"the DE-side encoding contract; it may still resolve "
            f"encoded columns correctly via 3b.6, but the upstream "
            f"rationale is lost."
        )

    def test_d1_3b6_rule_still_rendered_post_3b18(
        self, registry: SkillRegistry, locked_spec: dict
    ) -> None:
        """3b.18 must be additive to D1: the 3b.6 Analyst-side
        `resolve_encoded_columns` rule must still reach the Analyst
        rendered prompt unchanged. If 3b.18's additions pushed the
        3b.6 content out of the rendered prompt (budget overflow), the
        new amendment broke an older rule.
        """
        rendered = _render_analyst_prompt_for_causal_soo(
            registry,
            research_question=locked_spec["research_question"],
        )
        assert "resolve_encoded_columns" in rendered, (
            "The 3b.6 `resolve_encoded_columns` rule is missing from "
            "the rendered Analyst prompt post-3b.18. 3b.18 should be "
            "ADDITIVE to D1; if budget overflow pushed 3b.6 out, the "
            "amendment is breaking an older rule and needs a budget "
            "fix or formatter adjustment."
        )


class TestBudgetSufficiencyForCausalSOO:
    """§6.5 budget sufficiency check.

    Under the new per-tier cap, mandatory bodies render unconditionally.
    Recommended/reference compete for max_chars of their own. If the
    diagnostic comment line appears in the rendered prompt for the
    locked-spec causal_soo Analyst stage, some recommended skill is
    being dropped — either the budget is too small or a tier promotion
    is missing.
    """

    def test_no_drops_for_causal_soo_analyst_at_default_budget(
        self, registry: SkillRegistry, locked_spec: dict
    ) -> None:
        rendered = _render_analyst_prompt_for_causal_soo(
            registry,
            research_question=locked_spec["research_question"],
        )
        assert "Dropped from prompt due to budget" not in rendered, (
            "Recommended-tier skills are still being dropped under the "
            "new per-tier cap. Decision tree from §6.5:\n"
            "  (a) bump max_chars upward in format_skills_for_prompt's "
            "default, or pass a higher value from the caller for "
            "causal_soo; or\n"
            "  (b) accept that some recommended skills get dropped and "
            "treat the diagnostic-comment listing as input to future "
            "'should this skill be mandatory' decisions.\n"
            "The hand-off prefers (a) — cost is not the concern; "
            "performance is."
        )
