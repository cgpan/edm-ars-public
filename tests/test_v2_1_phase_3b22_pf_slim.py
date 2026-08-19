"""V2.1 Phase 3b.22 — ProblemFormulator slim apply verification.

Verifies the migration is structurally clean:

- The slim form was applied to the BASE (prediction-routed) prompt
  (line count ≤ 100; V1 was 207 lines) and V1 rule-block residue is
  gone from the prompt body.
- The {{SKILLS}} placeholder is present and substitutes at runtime.
- All V1 Validation Rules 1–14 reach the RENDERED prompt (slim body +
  injected skills) — the content moved into skills, it was not lost.
- The Literature Selection contract (8–12 papers, verbatim metadata,
  API-failure fallback) reaches the rendered prompt via the
  literature-search-s2-arxiv skill.
- Findings-Memory usage + diverse-candidates rules reach the rendered
  prompt via the findings-memory-novelty-cross-run skill.
- Output JSON schemas (research_spec + literature_context) stay in the
  slim prompt body itself (unlike OutlineAgent, where the schema moved
  into a skill) — field names must be byte-identical to V1 because
  downstream agents depend on them.
- The causal_soo task-type variant (problem_formulator_causal_soo.yaml)
  is untouched: still routed by load_prompt, still slim, still carries
  {{SKILLS}} (3b.20 §2.4: task-type branching RETAINED).
- V1 backup is in place per the CLAUDE.md V2.1 runbook.

Canonical Research Questions note: the V1 prompt's inline list is
intentionally NOT asserted anywhere. Sub-wave 0 verified the full
registry YAML — including its `canonical_research_questions:` field —
is dumped into the PF user message by _build_user_message()
(src/agents/problem_formulator.py), so the content travels via the
data path. (The V1 inline list also contained two variable names that
do not exist in the registry — X4ENTMJST, X3TGPAALL — so dropping it
is a correction, not a loss.)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.agents.base import load_prompt
from src.skills import SkillRegistry, format_skills_for_prompt


PROJECT_ROOT = Path(__file__).parent.parent
SKILLS_ROOT = PROJECT_ROOT / "skills"
PROMPTS_DIR = PROJECT_ROOT / "agent_prompts"
_SKILLS_PLACEHOLDER = "{{SKILLS}}"

_CONFIG: dict[str, Any] = {
    "paths": {"agent_prompts": str(PROMPTS_DIR) + "/"},
}


@pytest.fixture(scope="module")
def registry() -> SkillRegistry:
    return SkillRegistry(SKILLS_ROOT)


@pytest.fixture(scope="module")
def slim_pf_prompt() -> str:
    """Load the live base ProblemFormulator system_prompt (post-3b.22).

    load_prompt with task_type='prediction' falls through to the base
    file because no problem_formulator_prediction.yaml override exists
    — this mirrors the production routing exactly.
    """
    prompt_data = load_prompt("problem_formulator", _CONFIG, task_type="prediction")
    return prompt_data["system_prompt"]


@pytest.fixture(scope="module")
def causal_pf_prompt() -> str:
    """Load the causal_soo variant (must be untouched by 3b.22)."""
    prompt_data = load_prompt("problem_formulator", _CONFIG, task_type="causal_soo")
    return prompt_data["system_prompt"]


def _render_pf_prompt_for_prediction(
    registry: SkillRegistry,
    slim_pf_prompt: str,
) -> str:
    """Reproduce the orchestrator's ProblemFormulator skill-injection path."""
    matched = registry.match_and_compose(
        stage="ProblemFormulator",
        task_type="prediction",
        dataset="hsls09_public",
        context="research question design novelty gap literature",
    )
    skills_block = format_skills_for_prompt(matched).rstrip()
    return slim_pf_prompt.replace(_SKILLS_PLACEHOLDER, skills_block)


# ---------------------------------------------------------------------------
# §22.1 — slim form applied + placeholder present
# ---------------------------------------------------------------------------


class TestPFSlimApplied:
    def test_slim_line_count(self, slim_pf_prompt: str) -> None:
        """V1 was 207 lines; the staged slim draft is 95 lines total
        (system_prompt body ~91). After application, the system_prompt
        block should be substantially shorter than V1's."""
        line_count = slim_pf_prompt.count("\n") + 1
        assert line_count <= 100, (
            f"ProblemFormulator system_prompt has {line_count} lines; "
            f"expected slim form ≤ 100 (staged draft body is ~91 lines). "
            f"If this fails, the slim apply may have been reverted or "
            f"the V1 prompt restored."
        )

    def test_slim_does_not_contain_v1_rule_blocks(
        self, slim_pf_prompt: str
    ) -> None:
        """V1 had ~73 lines of Validation Rules 1–14 plus verbose
        Literature-Selection / Canonical-Questions / Findings-Memory
        sections. The slim form delegates all of them to skills. Their
        markers must be ABSENT from the slim prompt body (but PRESENT
        in the rendered prompt — checked in separate tests)."""
        v1_residue_markers = [
            "## Validation Rules (enforce before outputting)",
            "## Canonical Research Questions",
            "REJECTED patterns (too generic):",
            "You MUST include 8–12 papers",
            "## Generating Diverse Candidates",
        ]
        for marker in v1_residue_markers:
            assert marker not in slim_pf_prompt, (
                f"Slim ProblemFormulator prompt still contains V1 "
                f"residue marker {marker!r} — the rule-block migration "
                f"is incomplete."
            )

    def test_slim_has_skills_placeholder(self, slim_pf_prompt: str) -> None:
        assert _SKILLS_PLACEHOLDER in slim_pf_prompt, (
            "ProblemFormulator system_prompt does not contain the "
            "{{SKILLS}} placeholder. Skill injection cannot reach the "
            "rendered prompt without it."
        )

    def test_slim_pre_emit_validation_references_carrier_skill(
        self, slim_pf_prompt: str
    ) -> None:
        """The staged draft's Pre-Emit Validation block names the
        prediction-research-question-design skill explicitly. That
        skill is prediction-only (applicable_task_types: [prediction]),
        which is safe because the base prompt is only rendered for
        prediction tasks (causal_soo routes to its own variant file)."""
        assert "prediction-research-question-design" in slim_pf_prompt


# ---------------------------------------------------------------------------
# §22.2 — skill injection works at runtime
# ---------------------------------------------------------------------------


class TestPFSkillInjectionWorks:
    def test_carrier_skills_match_at_pf_prediction(
        self, registry: SkillRegistry
    ) -> None:
        """The three content-carrier skills must match at the
        ProblemFormulator stage for prediction tasks: the mandatory
        rule carrier + the literature contract + findings memory."""
        matched = registry.match(
            stage="ProblemFormulator",
            task_type="prediction",
            dataset="hsls09_public",
        )
        names = {s.name for s in matched}
        for required in (
            "prediction-research-question-design",
            "literature-search-s2-arxiv",
            "findings-memory-novelty-cross-run",
            "hsls09-temporal-ordering",
            "hsls09-tier3-exclusions",
        ):
            assert required in names, (
                f"Carrier skill {required!r} did not match at "
                f"ProblemFormulator/prediction. Matched: {sorted(names)}"
            )

    def test_cross_task_carriers_also_match_for_causal(
        self, registry: SkillRegistry
    ) -> None:
        """literature-search-s2-arxiv and findings-memory-novelty-
        cross-run are task-type-agnostic and must also match under
        causal_soo (the causal variant prompt depends on them too)."""
        # 3b.23 lesson (applied retroactively after Arc D added two
        # mandatory PF skills): probe via the ORCHESTRATOR path — bare
        # match() under default caps under-represents production.
        from src.orchestrator import _resolve_skill_caps

        matched = registry.match_and_compose(
            stage="ProblemFormulator",
            task_type="causal_soo",
            dataset="hsls09_public",
            context="causal effect literature novelty",
            top_k_per_layer=_resolve_skill_caps("causal_soo"),
        )
        names = {s.name for s in matched}
        assert "literature-search-s2-arxiv" in names
        assert "findings-memory-novelty-cross-run" in names
        # And the prediction-only carrier must NOT leak into causal.
        assert "prediction-research-question-design" not in names, (
            "prediction-research-question-design matched under "
            "causal_soo — its applicable_task_types narrowing is broken."
        )

    def test_placeholder_substitutes_in_rendered_prompt(
        self, registry: SkillRegistry, slim_pf_prompt: str
    ) -> None:
        rendered = _render_pf_prompt_for_prediction(registry, slim_pf_prompt)
        assert _SKILLS_PLACEHOLDER not in rendered

    def test_all_matched_skills_survive_formatter_cap(
        self, registry: SkillRegistry
    ) -> None:
        """Sub-wave 0 measured the PF/prediction skills block at
        ~24.3K chars — under the 30K non-mandatory budget, so nothing
        should be dropped. This test locks that in: if a future skill
        grows the block past the cap, the drop would silently remove
        V1-migrated content (the exact failure mode 3b.19's stdout
        showed at other stages)."""
        matched = registry.match(
            stage="ProblemFormulator",
            task_type="prediction",
            dataset="hsls09_public",
        )
        block = format_skills_for_prompt(matched)
        for skill in matched:
            assert skill.name in block, (
                f"Skill {skill.name!r} matched at ProblemFormulator/"
                f"prediction but was dropped by the formatter budget. "
                f"V1-migrated content is silently missing from the "
                f"rendered prompt."
            )


# ---------------------------------------------------------------------------
# §22.3 — V1 Validation Rules 1–14 preserved in the rendered prompt
# ---------------------------------------------------------------------------


class TestPFValidationRulesPreserved:
    """Each marker is a distinctive substring of the V1 rule's content
    as it now lives in a skill body (verified against the actual skill
    text in sub-wave 0). The rendered prompt (slim + injected skills)
    must contain every one."""

    @pytest.mark.parametrize(
        "rule,marker",
        [
            ("rule-01-temporal-ordering", "wave must come strictly"),
            ("rule-02-sample-size-floor", "10,000"),
            ("rule-03-novelty-floor", "Minimum acceptable"),
            ("rule-04-predictor-rationale", "is not a rationale"),
            ("rule-05-outcome-from-registry", "must be from the registry"),
            ("rule-06-protected-attributes", "MUST also appear in"),
            ("rule-07-redundant-composites", "X1SES_U"),
            ("rule-08-no-causal-framing", "causal or experimental designs"),
            ("rule-09-gap-driven-framing", "REJECTED patterns"),
            ("rule-09b-gap-accepted-examples", "BEYOND what academic achievement and SES explain"),
            ("rule-10-contrast-framing", "ABOVE AND BEYOND"),
            ("rule-11-surprising-predictor", "not bury them"),
            ("rule-12-predictor-coherence", "kitchen-sink"),
            ("rule-13-novelty-calibration", "would surprise a reviewer"),
            ("rule-14-findings-memory-build-on", "lose its predictive dominance"),
        ],
    )
    def test_rule_in_rendered_prompt(
        self,
        registry: SkillRegistry,
        slim_pf_prompt: str,
        rule: str,
        marker: str,
    ) -> None:
        rendered = _render_pf_prompt_for_prediction(registry, slim_pf_prompt)
        assert marker in rendered, (
            f"V1 {rule} marker {marker!r} missing from rendered "
            f"ProblemFormulator prompt. The rule was lost in the slim "
            f"migration — harvest it into the carrier skill per the "
            f"V2.1 runbook step 3."
        )


# ---------------------------------------------------------------------------
# §22.4 — Literature Selection contract preserved (via skill)
# ---------------------------------------------------------------------------


class TestPFLiteratureSelectionPreserved:
    @pytest.mark.parametrize(
        "marker",
        [
            "You MUST include 8–12 papers",
            "EXACTLY",
            # The skill body wraps this sentence across a line break
            # after "by the", so the marker stops at the wrap point.
            "will be filtered out by the",
            "fewer than 8 papers",
        ],
    )
    def test_literature_contract_in_rendered_prompt(
        self,
        registry: SkillRegistry,
        slim_pf_prompt: str,
        marker: str,
    ) -> None:
        rendered = _render_pf_prompt_for_prediction(registry, slim_pf_prompt)
        assert marker in rendered, (
            f"Literature Selection marker {marker!r} missing from the "
            f"rendered PF prompt. V1 carried this inline; the slim form "
            f"depends on literature-search-s2-arxiv reaching the prompt."
        )


# ---------------------------------------------------------------------------
# §22.5 — Findings Memory + diverse candidates preserved (via skill)
# ---------------------------------------------------------------------------


class TestPFFindingsMemoryPreserved:
    @pytest.mark.parametrize(
        "marker",
        [
            "catalogue of what has already been studied",
            "differ meaningfully from each prior candidate",
            "near-synonym",
        ],
    )
    def test_findings_memory_content_in_rendered_prompt(
        self,
        registry: SkillRegistry,
        slim_pf_prompt: str,
        marker: str,
    ) -> None:
        rendered = _render_pf_prompt_for_prediction(registry, slim_pf_prompt)
        assert marker in rendered, (
            f"Findings-Memory marker {marker!r} missing from the "
            f"rendered PF prompt. V1's Using-Findings-Memory / "
            f"Generating-Diverse-Candidates sections live in the "
            f"findings-memory-novelty-cross-run skill."
        )


# ---------------------------------------------------------------------------
# §22.6 — output JSON schemas preserved in the slim prompt body
# ---------------------------------------------------------------------------


class TestPFSchemaPreserved:
    """Unlike OutlineAgent (schema moved into a skill), the PF slim
    draft keeps both output schemas in the prompt body. Field names
    must be byte-identical to V1 — DataEngineer, Analyst, Critic, and
    Writer all consume research_spec fields by exact name."""

    V1_SCHEMA_FIELDS: tuple[str, ...] = (
        # research_spec
        "research_question",
        "outcome_variable",
        "outcome_type",
        "predictor_set",
        "rationale",
        "wave",
        "target_population",
        "subgroup_analyses",
        "expected_contribution",
        "potential_limitations",
        "novelty_score_self_assessment",
        # literature_context
        "search_query",
        "papers",
        "paperId",
        "novelty_evidence",
    )

    @pytest.mark.parametrize("field", list(V1_SCHEMA_FIELDS))
    def test_schema_field_in_slim_body(
        self, slim_pf_prompt: str, field: str
    ) -> None:
        assert field in slim_pf_prompt, (
            f"V1 schema field {field!r} missing from the slim PF "
            f"prompt body. The slim draft must carry both output "
            f"schemas inline — downstream agents depend on exact "
            f"field names."
        )

    def test_two_top_level_output_keys(self, slim_pf_prompt: str) -> None:
        """The output contract is exactly two top-level keys."""
        assert "research_spec" in slim_pf_prompt
        assert "literature_context" in slim_pf_prompt


# ---------------------------------------------------------------------------
# §22.7 — universal constraints preserved in the slim prompt body
# ---------------------------------------------------------------------------


class TestPFUniversalConstraintsPreserved:
    @pytest.mark.parametrize(
        "constraint,marker",
        [
            ("json-only-output", "Output ONLY valid JSON"),
            ("tier3-exclusions", "tier-3 exclusion"),
            ("outcome-not-in-predictors", "MUST NOT appear in `predictor_set`"),
            ("honest-novelty", "do not inflate"),
            ("revision-handling", "Do not regenerate"),
        ],
    )
    def test_constraint_in_slim_body(
        self, slim_pf_prompt: str, constraint: str, marker: str
    ) -> None:
        assert marker in slim_pf_prompt, (
            f"Universal constraint {constraint} (marker {marker!r}) "
            f"missing from the slim PF prompt body."
        )


# ---------------------------------------------------------------------------
# §22.8 — role / identity preservation
# ---------------------------------------------------------------------------


class TestPFRolePreserved:
    def test_agent_name_unchanged(self) -> None:
        prompt_data = load_prompt("problem_formulator", _CONFIG, task_type="prediction")
        assert prompt_data.get("agent_name") == "ProblemFormulator"

    def test_model_config_key_preserved(self) -> None:
        prompt_data = load_prompt("problem_formulator", _CONFIG, task_type="prediction")
        assert prompt_data.get("model_config_key") == "problem_formulator"

    def test_temperature_and_max_tokens_preserved(self) -> None:
        """V1 had temperature=0.7 (the one creative agent besides
        Writer) and max_tokens=8192. Both must survive."""
        prompt_data = load_prompt("problem_formulator", _CONFIG, task_type="prediction")
        assert prompt_data.get("temperature") == 0.7
        assert prompt_data.get("max_tokens") == 8192


# ---------------------------------------------------------------------------
# §22.9 — causal_soo task-type variant untouched (3b.20 §2.4: RETAINED)
# ---------------------------------------------------------------------------


class TestPFCausalVariantUntouched:
    def test_causal_variant_routes_via_load_prompt(
        self, causal_pf_prompt: str
    ) -> None:
        """load_prompt('problem_formulator', task_type='causal_soo')
        must return the causal variant, not the base prompt. The
        variant's distinctive marker is its operating-mode banner."""
        assert "CAUSAL_SOO" in causal_pf_prompt, (
            "load_prompt with task_type='causal_soo' did not return "
            "the causal variant — task-type routing broke."
        )

    def test_causal_variant_has_skills_placeholder(
        self, causal_pf_prompt: str
    ) -> None:
        assert _SKILLS_PLACEHOLDER in causal_pf_prompt

    def test_causal_variant_is_refine_mode(
        self, causal_pf_prompt: str
    ) -> None:
        """The causal variant operates in refine-a-locked-spec mode;
        3b.22 must not have overwritten it with the prediction slim."""
        assert "Locked Research Spec" in causal_pf_prompt or (
            "locked research_spec" in causal_pf_prompt
        )
        # The prediction slim's pre-emit block must NOT be here.
        assert "prediction-research-question-design" not in causal_pf_prompt


# ---------------------------------------------------------------------------
# §22.10 — V1 backup preserved
# ---------------------------------------------------------------------------


class TestV1BackupPreserved:
    def test_v1_backup_file_exists(self) -> None:
        bak = PROMPTS_DIR / "problem_formulator.v1.yaml.bak"
        assert bak.is_file(), (
            f"V1 backup file {bak} not present. Per the CLAUDE.md V2.1 "
            f"runbook, the 3b.22 migration must preserve V1 for "
            f"rollback."
        )

    def test_v1_backup_is_the_old_v1_content(self) -> None:
        """Spot-check the backup is V1, not a copy of the slim form."""
        bak = PROMPTS_DIR / "problem_formulator.v1.yaml.bak"
        bak_text = bak.read_text(encoding="utf-8")
        assert "## Validation Rules (enforce before outputting)" in bak_text, (
            "V1 backup doesn't contain the V1 Validation Rules block. "
            "The .bak may have been overwritten with the slim form."
        )
        assert "## Canonical Research Questions" in bak_text, (
            "V1 backup doesn't contain the Canonical Research "
            "Questions section."
        )
