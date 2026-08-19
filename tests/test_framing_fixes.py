"""V5 Arc T H2 — venue-independent framing fixes (offline).

Covers the three fixes from docs/v5_arc_t_h2_capability_roadmap.md §1-2:

- FIX 1 (VF2-03): feature-importance de-headlining. The abstract states
  the substantive finding and its use; SHAP/importance rankings are
  supporting evidence, never the stated contribution. Evidence cited in
  the skill: 0 of 1,135 measured abstracts + 0 of 30 AERA Open full
  texts headline an importance ranking.
- FIX 2 (VF2-07 tier 1): the final abstract sentence names the specific
  practice, decision, or design the result feeds — without overclaiming.
- FIX 3 (VF2-06 contrast): when school-aware splits are used, Results
  reports the within/cross-context contrast; the linter records
  INFO-level metrics (never defects) about whether it did.

Verification layers: skill files carry the rules (version-bumped, still
mandatory); the rules render in the composed Writer prompt through the
orchestrator path (match_and_compose + caps); the linter abstract checks
fire on synthetic non-compliant manuscripts and stay silent on
compliant ones. No network, no LLM.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.agents.base import load_prompt
from src.manuscript_linter import LintReport, lint_manuscript
from src.orchestrator import _resolve_skill_caps
from src.skills import SkillRegistry, format_skills_for_prompt

PROJECT_ROOT = Path(__file__).parent.parent
SKILLS_ROOT = PROJECT_ROOT / "skills"
PROMPTS_DIR = PROJECT_ROOT / "agent_prompts"
_CONFIG: dict[str, Any] = {"paths": {"agent_prompts": str(PROMPTS_DIR) + "/"}}
_CONTEXT = "Do non-cognitive factors predict college attendance beyond achievement and SES?"

SECTION_SKILL = SKILLS_ROOT / "writing" / "paper-section-content-prediction" / "SKILL.md"
OUTLINE_SKILL = SKILLS_ROOT / "writing" / "paper-narrative-outline" / "SKILL.md"


@pytest.fixture(scope="module")
def registry() -> SkillRegistry:
    return SkillRegistry(SKILLS_ROOT)


def _render_writer(registry: SkillRegistry, task_type: str = "prediction") -> str:
    prompt = load_prompt("writer", _CONFIG, task_type=task_type)["system_prompt"]
    matched = registry.match_and_compose(
        stage="Writer",
        task_type=task_type,
        dataset="hsls09_public",
        context=_CONTEXT,
        top_k_per_layer=_resolve_skill_caps(task_type),
    )
    return prompt.replace("{{SKILLS}}", format_skills_for_prompt(matched).rstrip())


# ---------------------------------------------------------------------------
# Skill-file level: the amendments landed, versions bumped, no new skill
# ---------------------------------------------------------------------------


class TestSkillAmendments:
    def test_section_skill_version_bumped_still_mandatory(self) -> None:
        body = SECTION_SKILL.read_text(encoding="utf-8")
        assert 'version: "1.1"' in body
        assert "rule_severity: mandatory" in body

    @pytest.mark.parametrize(
        "marker",
        [
            # FIX 1 — de-headlining rule + its load-bearing evidence
            "supporting evidence, never the stated contribution",
            "0 of 1,135 measured abstracts",
            # FIX 2 — final-sentence decision naming + overclaim ban
            "names the specific practice,",
            "NEVER overclaims beyond the evidence",
            # FIX 3 — cross-context claim + within/cross contrast rule
            "may claim cross-context evaluation",
            "never seen during training",
            "not seen during training",
            "invisible unless the contrast is printed",
        ],
    )
    def test_section_skill_carries_rule(self, marker: str) -> None:
        assert marker in SECTION_SKILL.read_text(encoding="utf-8")

    def test_contrast_rule_does_not_fabricate_capability(self) -> None:
        """The within/cross contrast is conditional on the analysis
        having computed both numbers — the skill must not instruct the
        Writer to invent the within-context estimate."""
        body = SECTION_SKILL.read_text(encoding="utf-8")
        assert "Never invent a comparison" in body

    def test_outline_skill_hook_rule(self) -> None:
        body = OUTLINE_SKILL.read_text(encoding="utf-8")
        assert 'version: "1.1"' in body
        assert "never a feature-importance ranking" in body

    def test_no_new_skill_directory_added(self) -> None:
        """Both count pins (test_orchestrator_skill_wiring,
        test_v3_0_causal_methods) stay at 70: this change amends
        existing skills only."""
        n = len(list(SKILLS_ROOT.glob("*/*/SKILL.md")))
        assert n == 70, f"expected 70 SKILL.md files, got {n}"


# ---------------------------------------------------------------------------
# Rendered-prompt level: rules reach the Writer via match_and_compose
# ---------------------------------------------------------------------------


class TestRulesReachWriterPrompt:
    @pytest.mark.parametrize(
        "marker",
        [
            "supporting evidence, never the stated contribution",
            "0 of 1,135 measured abstracts",
            "names the specific practice,",
            "NEVER overclaims beyond the evidence",
            "may claim cross-context evaluation",
            "invisible unless the contrast is printed",
        ],
    )
    def test_marker_in_rendered_prediction_writer_prompt(
        self, registry: SkillRegistry, marker: str
    ) -> None:
        rendered = _render_writer(registry, "prediction")
        assert marker in rendered, f"marker {marker!r} missing from Writer prompt"

    def test_hook_rule_reaches_outline_stage(self, registry: SkillRegistry) -> None:
        matched = registry.match_and_compose(
            stage="OutlineAgent",
            task_type="prediction",
            dataset="hsls09_public",
            context=_CONTEXT,
            top_k_per_layer=_resolve_skill_caps("prediction"),
        )
        block = format_skills_for_prompt(matched)
        assert "never a feature-importance ranking" in block


# ---------------------------------------------------------------------------
# Linter level: abstract checks fire / stay silent; contrast is INFO-only
# ---------------------------------------------------------------------------

BAD_ABSTRACT_TEX = r"""
\documentclass{article}
\title{Feature Importance Analysis of Dropout Prediction}
\begin{abstract}
We apply SHAP to identify the most important predictors of high school
dropout in a national longitudinal sample. Feature importance analysis
revealed that prior achievement ranks highest, followed by socioeconomic
status. The main contribution is a ranking of the top predictors of
dropout.
\end{abstract}
\begin{document}
\section{Introduction}
Body text.
\end{document}
"""

GOOD_ABSTRACT_TEX = r"""
\documentclass{article}
\title{Predicting College Enrollment from Ninth-Grade Engagement in a National Cohort}
\begin{abstract}
Using a national longitudinal sample (n = 19,240), we predict college
enrollment from ninth-grade engagement and achievement measures.
Gradient boosting reached AUC = 0.75, 95\% CI [0.73, 0.77], evaluated
on students in schools never seen during training. Engagement measures
added predictive value beyond achievement and socioeconomic status.
These results suggest reweighting engagement indicators in district
early-warning advising workflows for ninth graders.
\end{abstract}
\begin{document}
\section{Introduction}
Body text.
\end{document}
"""

CONTRAST_BOTH_TEX = r"""
\documentclass{article}
\title{A Study}
\begin{abstract}
An abstract of sufficient length for the front-matter check, closing
with a concrete suggestion for ninth-grade advising practice.
\end{abstract}
\begin{document}
\section{Results}
Our train/test split is school-aware: no school appears in both sets.
The cross-school AUC was 0.74 while the within-school AUC under a
random split was 0.78 (difference 0.04, 95\% CI [0.01, 0.07]).
\end{document}
"""

SCHOOL_AWARE_ONLY_TEX = r"""
\documentclass{article}
\title{A Study}
\begin{abstract}
An abstract of sufficient length for the front-matter check, closing
with a concrete suggestion for ninth-grade advising practice.
\end{abstract}
\begin{document}
\section{Results}
The train/test split is school-aware; the headline AUC is computed over
students in schools never seen during training.
\end{document}
"""

NO_SPLIT_STATEMENT_TEX = r"""
\documentclass{article}
\title{A Study}
\begin{abstract}
An abstract of sufficient length for the front-matter check, closing
with a concrete suggestion for ninth-grade advising practice.
\end{abstract}
\begin{document}
\section{Results}
The model reached an AUC of 0.75 on the held-out test set.
\end{document}
"""

NO_ABSTRACT_TEX = r"""
\documentclass{article}
\title{A Study}
\begin{document}
\section{Introduction}
Body text.
\end{document}
"""


def _lint(tmp_path: Path, tex: str) -> LintReport:
    (tmp_path / "paper.tex").write_text(tex, encoding="utf-8")
    return lint_manuscript(tmp_path, write_json=False)


def _codes(report: LintReport) -> list[str]:
    return [d.code for d in report.defects]


class TestLinterAbstractChecks:
    def test_feature_importance_headline_fires(self, tmp_path: Path) -> None:
        report = _lint(tmp_path, BAD_ABSTRACT_TEX)
        assert "abstract-headlines-feature-importance" in _codes(report)
        assert report.metrics["abstract_feature_importance_headline"] is True

    def test_no_decision_named_fires(self, tmp_path: Path) -> None:
        report = _lint(tmp_path, BAD_ABSTRACT_TEX)
        assert "abstract-names-no-decision" in _codes(report)
        assert report.metrics["abstract_names_decision"] is False

    def test_abstract_checks_are_warn_not_error(self, tmp_path: Path) -> None:
        report = _lint(tmp_path, BAD_ABSTRACT_TEX)
        for defect in report.defects:
            if defect.code.startswith("abstract-"):
                assert defect.severity == "warn", defect
        assert report.format_clean  # framing warns never block format_clean

    def test_compliant_abstract_is_silent(self, tmp_path: Path) -> None:
        report = _lint(tmp_path, GOOD_ABSTRACT_TEX)
        codes = _codes(report)
        assert "abstract-headlines-feature-importance" not in codes
        assert "abstract-names-no-decision" not in codes
        assert report.metrics["abstract_names_decision"] is True
        assert report.metrics["abstract_feature_importance_headline"] is False
        assert report.format_clean, codes

    def test_title_only_headline_fires(self, tmp_path: Path) -> None:
        tex = GOOD_ABSTRACT_TEX.replace(
            "Predicting College Enrollment from Ninth-Grade Engagement in a National Cohort",
            "The Most Important Predictors of College Enrollment",
        )
        report = _lint(tmp_path, tex)
        assert "abstract-headlines-feature-importance" in _codes(report)
        matching = [
            d for d in report.defects
            if d.code == "abstract-headlines-feature-importance"
        ]
        assert "title" in matching[0].message

    def test_missing_abstract_skips_content_checks(self, tmp_path: Path) -> None:
        report = _lint(tmp_path, NO_ABSTRACT_TEX)
        assert report.metrics["abstract_content_checked"] is False
        codes = _codes(report)
        assert "abstract-headlines-feature-importance" not in codes
        assert "abstract-names-no-decision" not in codes
        assert "missing-abstract" in codes  # front-matter error still fires


class TestLinterSplitContrastMetrics:
    def test_contrast_stated_both_metrics_true(self, tmp_path: Path) -> None:
        report = _lint(tmp_path, CONTRAST_BOTH_TEX)
        assert report.metrics["school_aware_split_stated"] is True
        assert report.metrics["within_cross_contrast_stated"] is True
        assert report.metrics["school_aware_contrast_reported"] is True

    def test_school_aware_without_contrast(self, tmp_path: Path) -> None:
        report = _lint(tmp_path, SCHOOL_AWARE_ONLY_TEX)
        assert report.metrics["school_aware_split_stated"] is True
        assert report.metrics["within_cross_contrast_stated"] is False
        assert report.metrics["school_aware_contrast_reported"] is False

    def test_no_split_statement(self, tmp_path: Path) -> None:
        report = _lint(tmp_path, NO_SPLIT_STATEMENT_TEX)
        assert report.metrics["school_aware_split_stated"] is False
        assert report.metrics["school_aware_contrast_reported"] is False

    @pytest.mark.parametrize(
        "tex", [CONTRAST_BOTH_TEX, SCHOOL_AWARE_ONLY_TEX, NO_SPLIT_STATEMENT_TEX]
    )
    def test_contrast_is_info_only_never_a_defect(
        self, tmp_path: Path, tex: str
    ) -> None:
        report = _lint(tmp_path, tex)
        assert not any(
            "school" in code or "contrast" in code for code in _codes(report)
        )
