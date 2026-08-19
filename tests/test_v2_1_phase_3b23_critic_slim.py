"""V2.1 Phase 3b.23 — Critic slim apply verification.

Verifies the migration is structurally clean:

- The slim form was applied (line count ≤ 150; V1 was 235 lines) and
  V1 checklist residue is gone from the prompt body.
- The {{SKILLS}} placeholder is present and substitutes at runtime.
- The prediction-critic-checklist skill matches at Critic/prediction,
  is tagged mandatory (3b.23 amendment per spec §3.3.4 — it bypasses
  the 30K formatter cap, where real drops occur at the Critic stage),
  and does NOT leak into causal_soo.
- The five 3b.23-harvested checklist rows (dp_08 group_overlap,
  dp_09 is_imbalanced consistency, an_11–an_13 ablation/SMOTE) reach
  the rendered prediction prompt — these were WEAKENED findings from
  the sub-wave-0 adversarial verification, harvested from V1 because
  their nominal carrier skills (school-aware-train-test-split,
  smote-imbalance-handling) are DataEngineer-/Analyst-stage only.
- Cross-task content (lens protocol, verdict criteria, severity
  definitions, revision-instructions guidance, novelty_review output
  contract, validation_passed rule) reaches BOTH task-type renderings.
  The novelty_review contract lives in the slim prompt BODY (not only
  in findings-memory-novelty-cross-run) because that skill is
  non-mandatory and is cap-dropped at Critic under causal_soo in live
  runs — the output contract must not depend on cap survival.
- review_report.json schema fields are byte-identical to V1 — the
  3b.10 deterministic verdict-evaluator and the orchestrator's
  revision cascade consume them by exact name.
- The verdict evaluator parses a slim-schema-shaped review correctly.
- Critic remains single-prompt (Option A from spec §3.3.3): no
  _causal_soo variant; load_prompt falls through to the base file for
  every task type.
- V1 backup is in place per the CLAUDE.md V2.1 runbook.

Rendering uses the ORCHESTRATOR-equivalent path (match_and_compose +
_resolve_skill_caps + a context string) rather than bare match() —
sub-wave 0 established that bare match() under-represents the
production skill set and produced a false LOST verdict.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.agents.base import load_prompt
from src.agents.verdict_evaluator import evaluate_critic_verdict
from src.orchestrator import _resolve_skill_caps
from src.skills import SkillRegistry, format_skills_for_prompt


PROJECT_ROOT = Path(__file__).parent.parent
SKILLS_ROOT = PROJECT_ROOT / "skills"
PROMPTS_DIR = PROJECT_ROOT / "agent_prompts"
_SKILLS_PLACEHOLDER = "{{SKILLS}}"

_CONFIG: dict[str, Any] = {
    "paths": {"agent_prompts": str(PROMPTS_DIR) + "/"},
}

_CONTEXT = "Do non-cognitive factors predict college attendance beyond achievement and SES?"


@pytest.fixture(scope="module")
def registry() -> SkillRegistry:
    return SkillRegistry(SKILLS_ROOT)


@pytest.fixture(scope="module")
def slim_critic_prompt() -> str:
    prompt_data = load_prompt("critic", _CONFIG)
    return prompt_data["system_prompt"]


def _render_critic_prompt(
    registry: SkillRegistry,
    slim_critic_prompt: str,
    task_type: str,
) -> str:
    """Reproduce the orchestrator's Critic skill-injection path
    (match_and_compose + task-type caps + context)."""
    matched = registry.match_and_compose(
        stage="Critic",
        task_type=task_type,
        dataset="hsls09_public",
        context=_CONTEXT,
        top_k_per_layer=_resolve_skill_caps(task_type),
    )
    skills_block = format_skills_for_prompt(matched).rstrip()
    return slim_critic_prompt.replace(_SKILLS_PLACEHOLDER, skills_block)


# ---------------------------------------------------------------------------
# §23.1 — slim form applied + placeholder present
# ---------------------------------------------------------------------------


class TestCriticSlimApplied:
    def test_slim_line_count(self, slim_critic_prompt: str) -> None:
        """V1 was 235 lines; the applied slim is 140 (staged 123 + the
        harvested Revision-Instructions block + the novelty_review
        output contract)."""
        line_count = slim_critic_prompt.count("\n") + 1
        assert line_count <= 150, (
            f"Critic system_prompt has {line_count} lines; expected the "
            f"slim form ≤ 150. The slim apply may have been reverted."
        )

    def test_slim_does_not_contain_v1_checklist_residue(
        self, slim_critic_prompt: str
    ) -> None:
        """The V1 prompt carried the four-section review checklist
        inline (~63 lines). Post-3b.23 it lives in the
        prediction-critic-checklist skill."""
        v1_residue_markers = [
            "## Review Checklist",
            "Research question is specific, answerable, and prediction-oriented",
            "## Optional: Novelty Review Against Prior Runs",
            "The task template (prediction.yaml)",
        ]
        for marker in v1_residue_markers:
            assert marker not in slim_critic_prompt, (
                f"Slim Critic prompt still contains V1 residue marker "
                f"{marker!r} — the checklist migration is incomplete."
            )

    def test_slim_has_skills_placeholder(self, slim_critic_prompt: str) -> None:
        assert _SKILLS_PLACEHOLDER in slim_critic_prompt


# ---------------------------------------------------------------------------
# §23.2 — skill injection works at runtime
# ---------------------------------------------------------------------------


class TestCriticSkillInjectionWorks:
    def test_checklist_skill_matches_at_critic_prediction(
        self, registry: SkillRegistry
    ) -> None:
        matched = registry.match_and_compose(
            stage="Critic",
            task_type="prediction",
            dataset="hsls09_public",
            context=_CONTEXT,
            top_k_per_layer=_resolve_skill_caps("prediction"),
        )
        names = {s.name for s in matched}
        assert "prediction-critic-checklist" in names, (
            f"prediction-critic-checklist did not match at "
            f"Critic/prediction. Matched: {sorted(names)}"
        )

    def test_checklist_skill_is_mandatory(self, registry: SkillRegistry) -> None:
        """3b.23 tagged the checklist skill mandatory (spec §3.3.4):
        real formatter-cap drops occur at the Critic stage in live runs
        (3b.19 / 3b.21.5 stdout), and a cap-dropped checklist would
        make the review structurally incomplete — silently."""
        matched = registry.match_and_compose(
            stage="Critic",
            task_type="prediction",
            dataset="hsls09_public",
            context=_CONTEXT,
            top_k_per_layer=_resolve_skill_caps("prediction"),
        )
        checklist = next(
            s for s in matched if s.name == "prediction-critic-checklist"
        )
        assert getattr(checklist, "rule_severity", None) == "mandatory"

    def test_checklist_does_not_leak_into_causal(
        self, registry: SkillRegistry
    ) -> None:
        matched = registry.match_and_compose(
            stage="Critic",
            task_type="causal_soo",
            dataset="hsls09_public",
            context=_CONTEXT,
            top_k_per_layer=_resolve_skill_caps("causal_soo"),
        )
        names = {s.name for s in matched}
        assert "prediction-critic-checklist" not in names, (
            "prediction-critic-checklist matched under causal_soo — "
            "its applicable_task_types narrowing is broken."
        )

    def test_checklist_survives_formatter_cap(
        self, registry: SkillRegistry
    ) -> None:
        """Mandatory severity bypasses the 30K cap; the full checklist
        body (incl. the harvested dp_08/dp_09/an_11–an_13 rows) must
        render IN at Critic/prediction."""
        matched = registry.match_and_compose(
            stage="Critic",
            task_type="prediction",
            dataset="hsls09_public",
            context=_CONTEXT,
            top_k_per_layer=_resolve_skill_caps("prediction"),
        )
        block = format_skills_for_prompt(matched)
        assert "dp_08" in block and "an_13" in block, (
            "prediction-critic-checklist body (harvested rows) was "
            "dropped by the formatter budget despite the mandatory tag."
        )

    def test_placeholder_substitutes_in_rendered_prompt(
        self, registry: SkillRegistry, slim_critic_prompt: str
    ) -> None:
        for tt in ("prediction", "causal_soo"):
            rendered = _render_critic_prompt(registry, slim_critic_prompt, tt)
            assert _SKILLS_PLACEHOLDER not in rendered


# ---------------------------------------------------------------------------
# §23.3 — prediction checklist content (incl. 3b.23 harvested rows)
# ---------------------------------------------------------------------------


class TestCriticChecklistPreserved:
    """Markers verified against the rendered prediction prompt.
    The last five entries are the rows harvested in 3b.23 from V1
    (previously WEAKENED in the sub-wave-0 adversarial verification)."""

    @pytest.mark.parametrize(
        "item,marker",
        [
            ("pf-temporal-ordering", "predictor wave appears strictly before outcome wave"),
            ("pf-feasibility-floor", "analytic_n ≥ 10,000"),
            ("dp-10p-rule", "10 × n_predictors_raw"),
            ("dp-imbalance-ratio", "> 9:1"),
            ("an-inner-cv", "inner CV on training data only"),
            ("an-shap-stacking-ban", "MUST NOT be StackingEnsemble"),
            ("an-kernel-cap", "sample cap ≤ 1,000"),
            ("an-subgroup-gap", "Subgroup gaps > 5%"),
            ("sv-auc-suspicious", "AUC > 0.95"),
            # 3b.23 harvested rows:
            ("dp08-school-disjoint", "group_overlap == 0"),
            ("dp09-imbalance-consistency", "`is_imbalanced` flag consistent"),
            ("an11-ablation-present", "Ablation present when data is imbalanced"),
            ("an12-smote-on-test", "SMOTE was applied to the test set"),
            ("an13-f2-balanced-acc", "balanced_accuracy"),
        ],
    )
    def test_checklist_item_in_rendered_prediction_prompt(
        self,
        registry: SkillRegistry,
        slim_critic_prompt: str,
        item: str,
        marker: str,
    ) -> None:
        rendered = _render_critic_prompt(
            registry, slim_critic_prompt, "prediction"
        )
        assert marker in rendered, (
            f"Checklist item {item} (marker {marker!r}) missing from "
            f"the rendered Critic/prediction prompt."
        )


# ---------------------------------------------------------------------------
# §23.4 — cross-task content in BOTH renderings
# ---------------------------------------------------------------------------


class TestCriticCrossTaskPreserved:
    @pytest.mark.parametrize(
        "item,marker",
        [
            ("lens-a", "LENS A — METHODOLOGIST"),
            ("lens-b", "LENS B — SKEPTIC"),
            ("lens-c", "LENS C — SYNTHESIZER"),
            ("verdict-exhausted-pass", "orchestrator will mark the paper UNVERIFIED"),
            ("verdict-never-abort", "Never"),
            ("severity-critical", "Invalidates the study"),
            ("revision-null-semantics", "Set to null for agents that do not need to revise"),
            ("revision-no-spec-override", "do not request SVM"),
            ("novelty-review-key", "novelty_review"),
            ("novelty-review-fields", "contribution_builds_on_memory"),
            ("novelty-review-never-null", '`"novelty_review": null`'),
            ("validation-failed-critical", "validation_passed == false"),
            ("no-invented-issues", "Never invent issues"),
        ],
    )
    def test_cross_task_content_in_both_renderings(
        self,
        registry: SkillRegistry,
        slim_critic_prompt: str,
        item: str,
        marker: str,
    ) -> None:
        for tt in ("prediction", "causal_soo"):
            rendered = _render_critic_prompt(registry, slim_critic_prompt, tt)
            assert marker in rendered, (
                f"Cross-task content {item} (marker {marker!r}) missing "
                f"from the rendered Critic prompt for {tt}."
            )


# ---------------------------------------------------------------------------
# §23.5 — review_report.json schema preserved in the slim prompt body
# ---------------------------------------------------------------------------


class TestCriticSchemaPreserved:
    """The 3b.10 deterministic verdict-evaluator and the orchestrator's
    revision cascade consume these fields by exact name."""

    V1_SCHEMA_FIELDS: tuple[str, ...] = (
        "overall_verdict",
        "overall_quality_score",
        "problem_formulation_review",
        "data_preparation_review",
        "analysis_review",
        "substantive_review",
        "educational_meaningfulness",
        "revision_instructions",
        "ProblemFormulator",
        "DataEngineer",
        "Analyst",
        "severity",
        "category",
        "recommendation",
        "target_agent",
    )

    @pytest.mark.parametrize("field", list(V1_SCHEMA_FIELDS))
    def test_schema_field_in_slim_body(
        self, slim_critic_prompt: str, field: str
    ) -> None:
        assert field in slim_critic_prompt, (
            f"review_report schema field {field!r} missing from the "
            f"slim Critic prompt body."
        )


# ---------------------------------------------------------------------------
# §23.6 — 3b.10 deterministic verdict-evaluator integration
# ---------------------------------------------------------------------------


class TestVerdictEvaluatorIntegration:
    """spec §3.3.7: confirm verdict_evaluator parses reviews shaped
    exactly like the slim prompt's schema example."""

    def _make_review(self, n_critical: int, n_major: int, quality: int) -> dict:
        def issues(n: int, sev: str) -> list[dict]:
            return [
                {
                    "severity": sev,
                    "category": "test",
                    "description": "d",
                    "recommendation": "r",
                    "target_agent": "Analyst",
                }
                for _ in range(n)
            ]

        return {
            "overall_verdict": "PASS",
            "overall_quality_score": quality,
            "problem_formulation_review": {"score": quality, "issues": issues(n_critical, "critical")},
            "data_preparation_review": {"score": quality, "issues": issues(n_major, "major")},
            "analysis_review": {"score": quality, "issues": []},
            "substantive_review": {
                "score": quality,
                "educational_meaningfulness": "m",
                "issues": [],
            },
            "revision_instructions": {
                "ProblemFormulator": None,
                "DataEngineer": None,
                "Analyst": None,
            },
        }

    def test_clean_review_passes(self) -> None:
        result = evaluate_critic_verdict(
            self._make_review(0, 0, 8), revision_cycle=0, max_revision_cycles=1
        )
        assert result.verdict == "PASS"
        assert result.n_critical == 0 and result.n_major == 0

    def test_critical_issue_forces_revise(self) -> None:
        result = evaluate_critic_verdict(
            self._make_review(1, 0, 8), revision_cycle=0, max_revision_cycles=1
        )
        assert result.verdict == "REVISE"
        assert result.n_critical == 1

    def test_exhausted_cycles_downgrade_to_unverified_pass(self) -> None:
        result = evaluate_critic_verdict(
            self._make_review(1, 0, 8), revision_cycle=1, max_revision_cycles=1
        )
        assert result.verdict == "PASS"
        assert result.unverified is True


# ---------------------------------------------------------------------------
# §23.7 — role / identity preservation
# ---------------------------------------------------------------------------


class TestCriticRolePreserved:
    def test_agent_identity(self) -> None:
        prompt_data = load_prompt("critic", _CONFIG)
        assert prompt_data.get("agent_name") == "Critic"
        assert prompt_data.get("model_config_key") == "critic"
        assert prompt_data.get("temperature") == 0.0
        assert prompt_data.get("max_tokens") == 8192


# ---------------------------------------------------------------------------
# §23.8 — no task-type variant (Option A confirmed)
# ---------------------------------------------------------------------------


class TestCriticNoTaskTypeVariant:
    def test_causal_soo_falls_through_to_base_prompt(self) -> None:
        """Critic is single-prompt (spec §3.3.5): no critic_causal_soo
        variant exists; load_prompt must fall through to the base
        file for every task type. Task-type behavior comes from skill
        matching + the deterministic verdict-evaluator."""
        base = load_prompt("critic", _CONFIG)
        via_causal = load_prompt("critic", _CONFIG, task_type="causal_soo")
        assert base["system_prompt"] == via_causal["system_prompt"]

    def test_no_variant_file_exists(self) -> None:
        assert not (PROMPTS_DIR / "critic_causal_soo.yaml").exists(), (
            "A critic_causal_soo.yaml variant appeared — spec §3.3.5 "
            "chose Option A (single prompt + task-type-aware skills). "
            "If a variant is now intended, update the spec and this test."
        )


# ---------------------------------------------------------------------------
# §23.9 — V1 backup preserved
# ---------------------------------------------------------------------------


class TestV1BackupPreserved:
    def test_v1_backup_file_exists(self) -> None:
        bak = PROMPTS_DIR / "critic.v1.yaml.bak"
        assert bak.is_file()

    def test_v1_backup_is_the_old_v1_content(self) -> None:
        bak = PROMPTS_DIR / "critic.v1.yaml.bak"
        bak_text = bak.read_text(encoding="utf-8")
        assert "## Review Checklist" in bak_text
        assert "## Optional: Novelty Review Against Prior Runs" in bak_text


# ---------------------------------------------------------------------------
# §23.10 — checklist-skill harvest rows landed
# ---------------------------------------------------------------------------


class TestChecklistSkillHarvest:
    """The 3b.23 harvest amendments to prediction-critic-checklist."""

    def test_harvested_rows_present(self) -> None:
        skill_md = (
            SKILLS_ROOT / "task-type" / "prediction-critic-checklist" / "SKILL.md"
        ).read_text(encoding="utf-8")
        for row_id in ("dp_08", "dp_09", "an_11", "an_12", "an_13"):
            assert f"`{row_id}`" in skill_md, (
                f"Harvested checklist row {row_id} missing from "
                f"prediction-critic-checklist/SKILL.md."
            )

    def test_mandatory_tag_present(self) -> None:
        skill_md = (
            SKILLS_ROOT / "task-type" / "prediction-critic-checklist" / "SKILL.md"
        ).read_text(encoding="utf-8")
        assert "rule_severity: mandatory" in skill_md
