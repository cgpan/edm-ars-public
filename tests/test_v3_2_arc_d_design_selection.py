"""V3.2 Arc D — design selection + gap mining: offline verification.

Exit probes per the v4 roadmap (internal) Arc D: the selector routes probe
questions to {SOO, ITR, infeasible-with-reasons}; the gap matrix and
design report reach the PF user message; the memo/gap/Critic skills
match at their stages.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from src.design_selector import classify_intent, select_design
from src.gap_miner import build_gap_matrix, format_gap_matrix

PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="module")
def hsls_registry() -> dict:
    with open(
        PROJECT_ROOT / "data_registry" / "datasets" / "hsls09_public.yaml",
        encoding="utf-8",
    ) as f:
        return yaml.safe_load(f)


class TestSelectorRouting:
    """The roadmap's three probe questions."""

    def test_targeting_question_routes_to_itr(self, hsls_registry: dict) -> None:
        report = select_design(
            hsls_registry,
            question="For whom should math self-efficacy be raised to "
            "improve college attendance?",
        )
        assert report["intent"] == "targeting"
        assert report["recommended_task_type"] == "causal_itr"

    def test_partner_cohort_alone_does_not_route_to_did(
        self, hsls_registry: dict
    ) -> None:
        # Arc T / T0 (the v5 ideation-layer specification (internal) §1.4, did_feasible row).
        # A `multi_cohort_partner` pointer is a harmonization LEAD, not a
        # runnable design: causal_did executes on the harmonized panel
        # (both live DiD runs used did_els_hsls_panel), never on HSLS
        # itself. This test previously asserted causal_did — that was the
        # defect, not the contract.
        report = select_design(
            hsls_registry,
            question="What is the effect of math self-efficacy on college attendance?",
        )
        assert report["recommended_task_type"] == "causal_soo"
        for d in ("rd", "iv"):
            assert report["verdicts"][d]["feasible"] is False
            assert report["verdicts"][d]["reasons"]
        # DiD stays honestly "feasible" (the partner data exists) but is
        # not executable on this dataset as-is.
        assert report["verdicts"]["did"]["feasible"] is True
        assert report["verdicts"]["did"]["executable_task_type"] is None

    def test_harmonized_panel_routes_to_did(self) -> None:
        with open(
            PROJECT_ROOT / "data_registry" / "datasets" / "did_els_hsls_panel.yaml",
            encoding="utf-8",
        ) as f:
            panel = yaml.safe_load(f)
        report = select_design(panel, intent="causal")
        assert report["recommended_task_type"] == "causal_did"
        assert report["verdicts"]["did"]["executable_task_type"] == "causal_did"

    def test_average_effect_question_falls_back_to_soo_without_partner(
        self, hsls_registry: dict
    ) -> None:
        # Honesty path: strip the partner-cohort marker and the selector
        # must fall back to SOO with per-design infeasibility reasons.
        import copy

        registry = copy.deepcopy(hsls_registry)
        registry.get("design_feasibility", {}).pop("multi_cohort_partner", None)
        registry.get("design_feasibility", {}).pop("policy_timing_variables", None)
        report = select_design(
            registry,
            question="What is the effect of math self-efficacy on college attendance?",
        )
        assert report["recommended_task_type"] == "causal_soo"
        for d in ("rd", "iv", "did"):
            assert report["verdicts"][d]["feasible"] is False
            assert report["verdicts"][d]["reasons"]
        assert "rd:" in report["rationale"]

    def test_prediction_question_routes_to_prediction(
        self, hsls_registry: dict
    ) -> None:
        report = select_design(
            hsls_registry,
            question="Which students are likely to enroll in college?",
        )
        assert report["recommended_task_type"] == "prediction"

    def test_registry_block_flips_feasibility(self) -> None:
        """A future dataset with a documented cutoff makes RD feasible —
        the predicate is data-driven, not hardcoded."""
        registry = {
            "variables": {"predictors": {"any": [{"name": "X"}]}},
            "design_feasibility": {
                "running_variables": [{"name": "TESTSCORE", "cutoff": 60.0}],
            },
        }
        report = select_design(registry, question="effect of scholarship")
        assert report["verdicts"]["rd"]["feasible"] is True

    def test_intent_override(self, hsls_registry: dict) -> None:
        report = select_design(hsls_registry, intent="targeting")
        assert report["recommended_task_type"] == "causal_itr"

    @pytest.mark.parametrize(
        "q,expected",
        [
            ("for whom does tutoring work", "targeting"),
            ("the effect of tutoring on GPA", "causal"),
            ("predict dropout risk", "prediction"),
        ],
    )
    def test_intent_classifier(self, q: str, expected: str) -> None:
        assert classify_intent(q) == expected


class TestGapMiner:
    def test_sparse_cells_detected(self) -> None:
        s2 = {
            "papers": [
                {"title": "Predicting college enrollment with random forest",
                 "abstract": "machine learning on HSLS"},
                {"title": "The causal effect of self-efficacy on GPA",
                 "abstract": "propensity score matching"},
            ]
        }
        gap = build_gap_matrix(s2)
        assert gap["n_papers"] == 2
        assert gap["matrix"]["college_enrollment"]["prediction_ml"] == 1
        # No retrieved paper does targeting on college enrollment:
        assert ("college_enrollment", "targeting_itr") in gap["sparse_cells"]

    def test_empty_corpus_is_honest(self) -> None:
        text = format_gap_matrix(build_gap_matrix(None))
        assert "No retrieved papers" in text

    def test_formatting_scopes_claims(self) -> None:
        gap = build_gap_matrix({"papers": [{"title": "causal gpa", "abstract": ""}]})
        text = format_gap_matrix(gap)
        assert "RETRIEVED corpus" in text
        assert "within the retrieved corpus" in text


class TestPFMessageInjection:
    """Both deterministic sections reach the PF user message on every
    build path (the injection lives inside _build_user_message)."""

    def test_sections_present(self, hsls_registry: dict) -> None:
        from src.agents.problem_formulator import ProblemFormulator

        pf = MagicMock(spec=ProblemFormulator)
        pf.ctx = MagicMock()
        pf.ctx.task_type = "causal_itr"
        msg = ProblemFormulator._build_user_message(
            pf,
            registry=hsls_registry,
            task_template={},
            s2_context={"papers": [{"title": "causal gpa", "abstract": "matching"}]},
            user_prompt="for whom should self-efficacy be raised?",
            revision_instructions=None,
        )
        assert "## Design Feasibility Report (deterministic" in msg
        assert "Recommended task type: **causal_itr**" in msg
        assert "## Gap Matrix (deterministic" in msg
        assert "design_memo" in msg

    def test_injection_failure_is_nonfatal(self) -> None:
        """A registry shaped in a way the selector chokes on must not
        kill message building."""
        from src.agents.problem_formulator import ProblemFormulator

        pf = MagicMock(spec=ProblemFormulator)
        pf.ctx = MagicMock()
        pf.ctx.task_type = "prediction"
        with patch(
            "src.design_selector.select_design", side_effect=RuntimeError("boom")
        ):
            msg = ProblemFormulator._build_user_message(
                pf,
                registry={"weird": True},
                task_template={},
                s2_context=None,
                user_prompt=None,
                revision_instructions=None,
            )
        assert "## Dataset Registry" in msg  # message still built


class TestArcDSkillsMatch:
    def test_memo_and_gap_skills_match_at_pf(self) -> None:
        from src.orchestrator import _resolve_skill_caps
        from src.skills import SkillRegistry

        reg = SkillRegistry(PROJECT_ROOT / "skills")
        for tt in ("prediction", "causal_soo", "causal_itr"):
            names = {
                s.name
                for s in reg.match_and_compose(
                    stage="ProblemFormulator",
                    task_type=tt,
                    dataset="hsls09_public",
                    context="design feasibility gap",
                    top_k_per_layer=_resolve_skill_caps(tt),
                )
            }
            assert "design-selection-memo" in names, tt
            assert "gap-driven-question-mining" in names, tt

    def test_design_gate_matches_at_critic(self) -> None:
        from src.orchestrator import _resolve_skill_caps
        from src.skills import SkillRegistry

        reg = SkillRegistry(PROJECT_ROOT / "skills")
        names = {
            s.name
            for s in reg.match_and_compose(
                stage="Critic",
                task_type="causal_itr",
                dataset="hsls09_public",
                context="design memo appropriateness",
                top_k_per_layer=_resolve_skill_caps("causal_itr"),
            )
        }
        assert "critic-design-appropriateness" in names
