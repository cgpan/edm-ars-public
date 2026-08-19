"""V2.1 Phase 3b.24 (V4 Arc H / H2) — Writer slim apply verification.

Structure mirrors 3b.21/3b.22/3b.23. Rendering uses the orchestrator
path (match_and_compose + caps + context). Key phase-specific locks:

- Five writing skills retagged mandatory (acm template, figure
  discipline, bibtex, and the two HSLS limitation paragraphs) — live
  runs demonstrably cap-drop recommended writing skills at the Writer
  stage, and these carry compile-critical / structurally-required
  content.
- One new skill: paper-section-content-prediction (prediction-only)
  absorbing V1's title patterns, word budgets, per-section rules,
  limitations ordering, quality caveats, sensitivity reporting.
- fonts-in-braces harvested into latex-table-discipline.
- Marker strings are wrap-safe and case-exact (3b.22/3b.24 lesson:
  skill bodies line-wrap and capitalize; markers must match reality).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.agents.base import load_prompt
from src.orchestrator import _resolve_skill_caps
from src.skills import SkillRegistry, format_skills_for_prompt


PROJECT_ROOT = Path(__file__).parent.parent
SKILLS_ROOT = PROJECT_ROOT / "skills"
PROMPTS_DIR = PROJECT_ROOT / "agent_prompts"
_CONFIG: dict[str, Any] = {"paths": {"agent_prompts": str(PROMPTS_DIR) + "/"}}
_CONTEXT = "Do non-cognitive factors predict college attendance beyond achievement and SES?"


@pytest.fixture(scope="module")
def registry() -> SkillRegistry:
    return SkillRegistry(SKILLS_ROOT)


@pytest.fixture(scope="module")
def slim_writer_prompt() -> str:
    return load_prompt("writer", _CONFIG, task_type="prediction")["system_prompt"]


def _render(registry: SkillRegistry, prompt: str, task_type: str) -> str:
    matched = registry.match_and_compose(
        stage="Writer",
        task_type=task_type,
        dataset="hsls09_public",
        context=_CONTEXT,
        top_k_per_layer=_resolve_skill_caps(task_type),
    )
    return prompt.replace("{{SKILLS}}", format_skills_for_prompt(matched).rstrip())


class TestWriterSlimApplied:
    def test_line_count(self, slim_writer_prompt: str) -> None:
        assert slim_writer_prompt.count("\n") + 1 <= 85  # V1 body was ~481

    def test_no_v1_residue(self, slim_writer_prompt: str) -> None:
        for marker in (
            "Mandatory counting procedure",
            "## Title Format",
            "%%PLACEHOLDER:METHODS_DATA%%",
            "## Optional: Novelty Review",
        ):
            assert marker not in slim_writer_prompt

    def test_skills_placeholder_present(self, slim_writer_prompt: str) -> None:
        assert "{{SKILLS}}" in slim_writer_prompt

    def test_disclosure_sentence_kept_in_prompt(self, slim_writer_prompt: str) -> None:
        assert "performed programmatically" in slim_writer_prompt

    def test_v2_template_contract_kept(self, slim_writer_prompt: str) -> None:
        assert "paper_template_v2.tex" in slim_writer_prompt


class TestWriterMandatorySeverities:
    """The five retags + the new skill must be mandatory: recommended
    writing skills demonstrably cap-drop at the Writer stage in live
    runs, and these carry compile-critical or structurally-required
    content (design rule from §3.3.8: output contracts never live in
    cap-droppable skills)."""

    @pytest.mark.parametrize(
        "skill",
        [
            "acm-acmart-sigconf-template",
            "latex-figure-discipline",
            "bibtex-from-literature-context",
            "hsls09-multilevel-limitations-paragraph",
            "hsls09-survey-weights-limitations-paragraph",
            "paper-section-content-prediction",
        ],
    )
    def test_skill_is_mandatory(self, skill: str) -> None:
        body = (SKILLS_ROOT / "writing" / skill / "SKILL.md").read_text(
            encoding="utf-8"
        )
        assert "rule_severity: mandatory" in body


class TestWriterContentPreserved:
    """V1 content blocks reach the rendered prediction prompt."""

    @pytest.mark.parametrize(
        "marker",
        [
            "threeparttable",
            "resizebox",
            "inside braces",          # fonts harvest
            "\\Description",
            "Who Is at Risk?",        # title patterns (new skill)
            "roadmap sentence",
            "preserving the original", # SMOTE sentence
            "AUC (No SMOTE)",         # ablation table
            "F2 and Balanced Accuracy",
            "Most consequential first",  # limitations ordering
            "minimum performance",    # quality caveats (wrap-safe)
            "Survey Weights",         # case-exact
            "intraclass correlation",
            "unresolved methodological issues",  # UNVERIFIED
            "Active voice",
            "arxiv_",                 # bibtex key rule
            "PAPER_BODY",             # v2 template slot documented
        ],
    )
    def test_marker_in_rendered_prediction_prompt(
        self, registry: SkillRegistry, slim_writer_prompt: str, marker: str
    ) -> None:
        rendered = _render(registry, slim_writer_prompt, "prediction")
        assert marker in rendered, f"marker {marker!r} missing"

    def test_prediction_skill_does_not_leak_into_causal(
        self, registry: SkillRegistry
    ) -> None:
        matched = registry.match_and_compose(
            stage="Writer",
            task_type="causal_soo",
            dataset="hsls09_public",
            context=_CONTEXT,
            top_k_per_layer=_resolve_skill_caps("causal_soo"),
        )
        assert "paper-section-content-prediction" not in {s.name for s in matched}

    @pytest.mark.parametrize(
        "marker",
        ["threeparttable", "\\Description", "unresolved methodological issues", "Survey Weights"],
    )
    def test_cross_task_marker_in_causal_rendering(
        self, registry: SkillRegistry, slim_writer_prompt: str, marker: str
    ) -> None:
        rendered = _render(registry, slim_writer_prompt, "causal_soo")
        assert marker in rendered


class TestWriterRoleAndVariant:
    def test_role_preserved(self) -> None:
        d = load_prompt("writer", _CONFIG, task_type="prediction")
        assert d["agent_name"] == "Writer"
        assert d["temperature"] == 0.3
        assert d["max_tokens"] == 16384

    def test_causal_variant_untouched(self) -> None:
        causal = load_prompt("writer", _CONFIG, task_type="causal_soo")["system_prompt"]
        base = load_prompt("writer", _CONFIG, task_type="prediction")["system_prompt"]
        assert causal != base
        assert "{{SKILLS}}" in causal


class TestV1BackupPreserved:
    def test_backup_exists_with_v1_content(self) -> None:
        bak = PROMPTS_DIR / "writer.v1.yaml.bak"
        text = bak.read_text(encoding="utf-8")
        assert "Mandatory counting procedure" in text
        assert "## Title Format" in text
