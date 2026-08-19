"""V2.1 Phase 3b.21 — OutlineAgent slim apply + skill injection wire-up.

Verifies the migration is structurally clean:

- The slim form was applied (line count ≤ 40).
- The {{SKILLS}} placeholder is present in the raw prompt.
- Skill injection works at runtime (placeholder substitutes; the
  paper-narrative-outline skill body reaches the rendered prompt).
- Output JSON schema is preserved (the schema lives in the injected
  skill rather than the prompt itself; the rendered prompt contains it).
- Agent identity / role is preserved across the migration.
- V1 backup is in place (per CLAUDE.md V2.1 runbook convention).

F-3b13-OUTLINE-AGENT-TYPE-ERROR regression test is intentionally
omitted — Sub-wave 0 found the error is non-deterministic LLM-response-
shape stochasticity, not a deterministic bug. Closure is deferred to
a later phase if the error recurs after migration.
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


@pytest.fixture(scope="module")
def registry() -> SkillRegistry:
    return SkillRegistry(SKILLS_ROOT)


@pytest.fixture(scope="module")
def slim_outline_prompt() -> str:
    """Load the live OutlineAgent system_prompt (post-3b.21)."""
    config: dict[str, Any] = {
        "paths": {"agent_prompts": str(PROMPTS_DIR) + "/"},
    }
    prompt_data = load_prompt("outline_agent", config)
    return prompt_data["system_prompt"]


def _render_outline_prompt_for_causal_soo(
    registry: SkillRegistry,
    slim_outline_prompt: str,
) -> str:
    """Reproduce the orchestrator's OutlineAgent skill-injection path."""
    matched = registry.match_and_compose(
        stage="OutlineAgent",
        task_type="causal_soo",
        dataset="hsls09_public",
        context="paper outline narrative_hook sections",
    )
    skills_block = format_skills_for_prompt(matched).rstrip()
    return slim_outline_prompt.replace(_SKILLS_PLACEHOLDER, skills_block)


# ---------------------------------------------------------------------------
# §21.5 — slim form applied + placeholder present
# ---------------------------------------------------------------------------


class TestOutlineSlimApplied:
    def test_slim_line_count(self, slim_outline_prompt: str) -> None:
        """V1 was 93 lines; staged slim draft is 25 lines (system_prompt
        body alone is shorter). After application, the system_prompt
        block should be substantially shorter than V1's."""
        line_count = slim_outline_prompt.count("\n") + 1
        assert line_count <= 30, (
            f"OutlineAgent system_prompt has {line_count} lines; "
            f"expected slim form ≤ 30 (V2.0.1 staged draft is 25 lines, "
            f"of which ~19 are in the system_prompt body). If this test "
            f"fails, the slim apply may have been reverted or the V1 "
            f"prompt was restored."
        )

    def test_slim_does_not_contain_v1_schema_block(
        self, slim_outline_prompt: str
    ) -> None:
        """V1 had a verbose JSON schema block (~48 lines, lines 15-62)
        with example sections. The slim form delegates the schema to
        the paper-narrative-outline skill. The schema markers should
        be ABSENT from the slim prompt body (but PRESENT in the
        injected skill body — checked in a separate test)."""
        # The V1 prompt opened the schema block with a specific
        # example narrative_hook line.
        v1_schema_markers = [
            '"narrative_hook": "One sentence describing',
            '"id": "introduction"',
            '"emphasis": "compressed|standard|expanded"',
        ]
        for marker in v1_schema_markers:
            assert marker not in slim_outline_prompt, (
                f"Slim OutlineAgent prompt still contains V1 schema "
                f"marker {marker!r} — the schema-block migration is "
                f"incomplete. Expected: schema moves to the paper-"
                f"narrative-outline skill body."
            )

    def test_slim_has_skills_placeholder(
        self, slim_outline_prompt: str
    ) -> None:
        """The 3b.21 migration depends on the {{SKILLS}} placeholder
        being present so the matcher's skill injection reaches the
        rendered prompt. The V2.0.1 staged draft includes it; this
        test confirms the production prompt has it post-apply."""
        assert _SKILLS_PLACEHOLDER in slim_outline_prompt, (
            "OutlineAgent system_prompt does not contain the "
            "{{SKILLS}} placeholder. Skill injection cannot reach the "
            "rendered prompt without it. The 3b.21 migration is "
            "incomplete."
        )


# ---------------------------------------------------------------------------
# §21.5 — skill injection works at runtime
# ---------------------------------------------------------------------------


class TestOutlineSkillInjectionWorks:
    def test_paper_narrative_outline_skill_matches_at_outline_stage(
        self, registry: SkillRegistry
    ) -> None:
        """The paper-narrative-outline skill must match at OutlineAgent
        stage. Per its frontmatter: applicable_stages=[OutlineAgent,
        Writer], applicable_task_types=[] (cross-task)."""
        matched = registry.match(
            stage="OutlineAgent",
            task_type="causal_soo",  # the skill is task-type-agnostic
            dataset="hsls09_public",
        )
        names = {s.name for s in matched}
        assert "paper-narrative-outline" in names, (
            f"paper-narrative-outline skill did not match at "
            f"OutlineAgent stage. Matched: {sorted(names)}"
        )

    def test_paper_narrative_outline_also_matches_for_prediction(
        self, registry: SkillRegistry
    ) -> None:
        """The skill's applicable_task_types is empty (cross-task), so
        it must also match for prediction tasks. Catches accidental
        causal-only narrowing of the skill's frontmatter."""
        matched = registry.match(
            stage="OutlineAgent",
            task_type="prediction",
            dataset="hsls09_public",
        )
        names = {s.name for s in matched}
        assert "paper-narrative-outline" in names

    def test_placeholder_substitutes_in_rendered_prompt(
        self, registry: SkillRegistry, slim_outline_prompt: str
    ) -> None:
        """After the matcher returns skills and the composer formats
        them, the {{SKILLS}} placeholder must be replaced — not left
        as a literal string in the rendered prompt."""
        rendered = _render_outline_prompt_for_causal_soo(
            registry, slim_outline_prompt
        )
        assert _SKILLS_PLACEHOLDER not in rendered, (
            "{{SKILLS}} placeholder was not substituted in the "
            "rendered OutlineAgent prompt. The matcher/composer "
            "wire-up is incomplete or the format_skills_for_prompt "
            "call is failing silently."
        )


# ---------------------------------------------------------------------------
# §21.5 — output schema preservation (via injected skill)
# ---------------------------------------------------------------------------


class TestOutlineSchemaPreserved:
    """V1 OutlineAgent had the JSON output schema inline in the prompt
    (lines 15-62 of the V1 file). The slim form delegates it to the
    paper-narrative-outline skill body. The RENDERED prompt (slim +
    injected skill) must contain the same schema fields V1 had — that's
    the contract preservation."""

    V1_SCHEMA_FIELDS: tuple[str, ...] = (
        "narrative_hook",
        "sections",
        "subsections",
        "emphasis",
        "word_target",
        "guidance",
        "introduction",
        "related_work",
        "methods",
        "results",
        "discussion",
    )

    @pytest.mark.parametrize("field", list(V1_SCHEMA_FIELDS))
    def test_schema_field_in_rendered_prompt(
        self,
        registry: SkillRegistry,
        slim_outline_prompt: str,
        field: str,
    ) -> None:
        """Every V1 schema field must appear in the rendered prompt
        (post-injection). The skill body's schema example contains
        them all; this confirms it reached the LLM."""
        rendered = _render_outline_prompt_for_causal_soo(
            registry, slim_outline_prompt
        )
        assert field in rendered, (
            f"V1 schema field {field!r} missing from rendered "
            f"OutlineAgent prompt. The paper-narrative-outline skill "
            f"should carry the full schema; either the skill body was "
            f"trimmed or the skill isn't reaching the rendered prompt."
        )


# ---------------------------------------------------------------------------
# §21.5 — design rules preservation (the 6 numbered rules)
# ---------------------------------------------------------------------------


class TestOutlineDesignRulesPreserved:
    """V1 had 6 outline design rules (lines 64-90 of V1). The slim
    form delegates them to the paper-narrative-outline skill. The
    rendered prompt must contain the rule content."""

    @pytest.mark.parametrize("marker", [
        # Distinctive substrings from each of the 6 V1 rules
        "Sections always present",                     # Rule 1
        "Subsection titles should be descriptive",     # Rule 2
        "Emphasis allocation",                         # Rule 3 header
        "Methods can be compressed",                   # Rule 4
        "Discussion subsections should make arguments",# Rule 5
        "narrative_hook drives the paper",             # Rule 6
    ])
    def test_design_rule_in_rendered_prompt(
        self,
        registry: SkillRegistry,
        slim_outline_prompt: str,
        marker: str,
    ) -> None:
        rendered = _render_outline_prompt_for_causal_soo(
            registry, slim_outline_prompt
        )
        assert marker in rendered, (
            f"V1 outline design rule marker {marker!r} missing from "
            f"rendered OutlineAgent prompt. Methodology was lost in "
            f"the migration."
        )


# ---------------------------------------------------------------------------
# §21.5 — role / identity preservation
# ---------------------------------------------------------------------------


class TestOutlineRolePreserved:
    def test_agent_name_unchanged(self) -> None:
        """The agent_name in the YAML must still be 'OutlineAgent'.
        Downstream code (orchestrator's _inject_skills call site)
        passes this string."""
        config: dict[str, Any] = {
            "paths": {"agent_prompts": str(PROMPTS_DIR) + "/"},
        }
        prompt_data = load_prompt("outline_agent", config)
        assert prompt_data.get("agent_name") == "OutlineAgent"

    def test_model_config_key_preserved(self) -> None:
        """The 'writer' model_config_key (OutlineAgent re-uses Writer's
        model selection per V2.0 convention) must persist."""
        config: dict[str, Any] = {
            "paths": {"agent_prompts": str(PROMPTS_DIR) + "/"},
        }
        prompt_data = load_prompt("outline_agent", config)
        assert prompt_data.get("model_config_key") == "writer"

    def test_temperature_and_max_tokens_preserved(self) -> None:
        """V1 had temperature=0.5, max_tokens=4096. Slim draft preserves
        both — non-trivial because they affect LLM response shape."""
        config: dict[str, Any] = {
            "paths": {"agent_prompts": str(PROMPTS_DIR) + "/"},
        }
        prompt_data = load_prompt("outline_agent", config)
        assert prompt_data.get("temperature") == 0.5
        assert prompt_data.get("max_tokens") == 4096

    def test_role_statement_preserved(self, slim_outline_prompt: str) -> None:
        """The slim prompt's role block must identify the agent as
        OutlineAgent. The exact wording differs from V1 (V1 had a
        longer role paragraph) but the identity is preserved."""
        assert "OutlineAgent" in slim_outline_prompt
        assert "outline" in slim_outline_prompt.lower()


# ---------------------------------------------------------------------------
# §21.5 — V1 backup preserved
# ---------------------------------------------------------------------------


class TestV1BackupPreserved:
    def test_v1_backup_file_exists(self) -> None:
        """Per CLAUDE.md V2.1 runbook: 'back up the v1 to
        agent_prompts/<agent>.v1.yaml.bak'. The 3b.21 commit preserves
        the V1 monolithic prompt as a backup for rollback / reference."""
        bak = PROMPTS_DIR / "outline_agent.v1.yaml.bak"
        assert bak.is_file(), (
            f"V1 backup file {bak} not present. The 3b.21 migration "
            f"should preserve V1 per the V2.1 runbook convention. "
            f"Without it, the rollback path is gone."
        )

    def test_v1_backup_is_the_old_v1_content(self) -> None:
        """Spot-check that the backup is V1, not a copy of the new
        slim form. V1 had specific markers (the verbose schema +
        outline design rules) that the slim form drops."""
        bak = PROMPTS_DIR / "outline_agent.v1.yaml.bak"
        bak_text = bak.read_text(encoding="utf-8")
        # V1's schema block markers — should be in the backup
        assert '"narrative_hook":' in bak_text, (
            "V1 backup doesn't contain the V1 schema block. The .bak "
            "may have been overwritten with the slim form."
        )
        assert "Sections always present" in bak_text, (
            "V1 backup doesn't contain V1 outline design rules."
        )
