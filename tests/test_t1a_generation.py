"""Arc T / T1a - slate, generator, cards, and the generate stage.

Everything here is OFFLINE. No test constructs a provider client, opens
a socket, or reads a raw data file: the LLM is a stub callable, and the
screen runs with ``use_column_cache=False`` so a machine without the
datasets gets the same result as one with them.

The obligations, in priority order:

1. **The seam holds.** ``compile_spec`` output must load through the
   REAL ``src.main.load_locked_research_spec`` - the function the CLI
   calls - for every (task_type, dataset) pair this repo can execute.
   Mocking the loader here would test nothing; the hand-off note says
   archived prediction specs carry neither ``task_type`` nor ``dataset``
   and the loader guards on it.
2. **Infeasible cells are never enumerated**, and the reason is written
   down.
3. **Diversity is structural**: quotas hold under a fixed seed, and the
   same seed reproduces the slate exactly.
4. **C1**: no novelty number is stored anywhere, and the deterministic
   ranking does not move when one is injected.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from src.ideation import cards as C  # noqa: E402
from src.ideation import feasibility as F  # noqa: E402
from src.ideation import generate as G  # noqa: E402
from src.ideation import slate as S  # noqa: E402
from src.ideation.cards import IdeaCard  # noqa: E402

import run_idea_tournament as R  # noqa: E402

REGISTRY_DIR = REPO_ROOT / "data_registry" / "datasets"

#: (task_type, dataset) pairs this repo can actually execute, measured
#: 2026-07-26 by compiling a card for each and loading it through
#: src.main.load_locked_research_spec. prediction x assistments_0910 is
#: deliberately absent: the loader rejects it (single-wave dataset), and
#: slate rule S3 stops it being enumerated.
SEAM_PAIRS: tuple[tuple[str, str], ...] = (
    ("prediction", "hsls09_public"),
    ("prediction", "els_2002"),
    ("prediction", "did_els_hsls_panel"),
    ("causal_soo", "hsls09_public"),
    ("causal_soo", "els_2002"),
    ("causal_soo", "did_els_hsls_panel"),
    ("causal_soo", "assistments_0910"),
    ("causal_itr", "hsls09_public"),
    ("causal_itr", "els_2002"),
    ("causal_did", "did_els_hsls_panel"),
    ("psychometrics", "hsls09_public"),
    ("psychometrics", "els_2002"),
    ("psychometrics", "assistments_0910"),
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def make_card(
    *,
    dataset: str = "hsls09_public",
    task_type: str = "prediction",
    pattern: str = "scope_extension",
    persona: str = "policy_analyst",
    candidate_id: str = "C-01",
    spec_draft: dict | None = None,
    **fields: Any,
) -> IdeaCard:
    payload: dict[str, Any] = {
        "candidate_id": candidate_id,
        "tournament_id": "T-0001",
        "cell": {
            "dataset": dataset,
            "task_type": task_type,
            "opportunity_pattern": pattern,
            "persona": persona,
            "gap_cell": ["college_enrollment", "fairness"],
        },
        "research_question": "Does prior achievement predict college entry?",
        "why_it_matters": "Advisers triage students on incomplete information.",
        "what_we_would_do": "Fit the standard battery and report subgroup error.",
        "what_counts_as_the_result": "A subgroup AUC gap wider than five points.",
        "spec_draft": spec_draft if spec_draft is not None else {},
    }
    payload.update(fields)
    return IdeaCard.from_dict(payload)


class StubCaller:
    """Records every prompt; returns a scripted or generated response."""

    def __init__(
        self,
        responder: Callable[[str, int], str] | None = None,
    ) -> None:
        self.prompts: list[str] = []
        self._responder = responder or _default_responder

    def __call__(self, user_message: str) -> str:
        index = len(self.prompts)
        self.prompts.append(user_message)
        return self._responder(user_message, index)


def _default_responder(user_message: str, index: int) -> str:
    cell = G._cell_from_message(user_message)
    return json.dumps(
        {
            "research_question": (
                f"Question {index}: how does {cell.get('dataset')} behave "
                f"under a {cell.get('opportunity_pattern')} framing?"
            ),
            "why_it_matters": f"Reason number {index} for caring about it.",
            "what_we_would_do": f"Procedure number {index} on the analytic file.",
            "what_counts_as_the_result": f"Outcome number {index} would settle it.",
            "method_family": "",
            "second_contribution": None,
            "spec_draft": {},
        }
    )


def screen_card(card: IdeaCard, spec: dict) -> F.FeasibilityReport:
    """Screen with the column universe switched off (hermetic)."""
    context = F.make_context(
        spec,
        dataset=card.dataset,
        task_type=card.task_type,
        registry_dir=REGISTRY_DIR,
        card=card.to_dict(),
        use_column_cache=False,
    )
    return F.screen(spec, candidate_id=card.candidate_id, context=context)


# ==========================================================================
# 1. Slate
# ==========================================================================


def test_slate_is_reproducible_under_a_fixed_seed() -> None:
    first = S.build_slate("T-0001", seed=42, registry_dir=REGISTRY_DIR)
    second = S.build_slate("T-0001", seed=42, registry_dir=REGISTRY_DIR)
    assert json.dumps(first.to_dict(), sort_keys=True) == json.dumps(
        second.to_dict(), sort_keys=True
    )


def test_slate_changes_with_the_seed() -> None:
    a = S.build_slate("T-0001", seed=42, registry_dir=REGISTRY_DIR)
    b = S.build_slate("T-0001", seed=7, registry_dir=REGISTRY_DIR)
    assert [c.to_dict() for c in a.cells] != [c.to_dict() for c in b.cells]


def test_slate_records_the_seed_it_used() -> None:
    slate = S.build_slate("T-0001", seed=1234, registry_dir=REGISTRY_DIR)
    assert slate.to_dict()["random_state"] == 1234


def test_no_enumerated_cell_is_infeasible_in_the_matrix() -> None:
    slate = S.build_slate("T-0001", registry_dir=REGISTRY_DIR)
    assert slate.cells
    for cell in slate.cells:
        assert F.DATASET_TASK_MATRIX[cell.dataset][cell.task_type] is True


def test_matrix_infeasible_cells_are_excluded_with_a_reason() -> None:
    slate = S.build_slate("T-0001", registry_dir=REGISTRY_DIR)
    excluded = {
        (row["dataset"], row["task_type"]): row for row in slate.excluded_cells
    }
    # A cell the shipped matrix marks False.
    assert ("hsls09_public", "causal_did") in excluded
    row = excluded[("hsls09_public", "causal_did")]
    assert row["rule"] == "S1"
    assert "policy_timing_variables" in row["reason"]
    assert ("hsls09_public", "causal_did") not in {
        (c.dataset, c.task_type) for c in slate.cells
    }


def test_single_wave_dataset_is_not_enumerated_for_prediction() -> None:
    """Rule S3 - measured, not assumed: the loader rejects that spec."""
    slate = S.build_slate("T-0001", registry_dir=REGISTRY_DIR)
    assert ("assistments_0910", "prediction") not in {
        (c.dataset, c.task_type) for c in slate.cells
    }
    row = next(
        r
        for r in slate.excluded_cells
        if (r["dataset"], r["task_type"]) == ("assistments_0910", "prediction")
    )
    assert row["rule"] == "S3"
    assert "temporal_order" in row["reason"]


def test_s3_reason_is_true_of_the_real_loader(tmp_path: Path) -> None:
    """Pin rule S3 to the behaviour it claims, through the real loader.

    If task_template.py ever resolves single-wave prediction, this test
    fails and rule S3 must be deleted rather than quietly kept.
    """
    from src.main import load_locked_research_spec

    card = make_card(dataset="assistments_0910", task_type="prediction")
    spec = C.compile_spec(card, registry_dir=REGISTRY_DIR)
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        load_locked_research_spec(str(path))
    assert "TEMPORAL VIOLATION" in str(excinfo.value)


def test_equity_pattern_never_lands_where_no_protected_attribute_exists() -> None:
    slate = S.build_slate("T-0001", registry_dir=REGISTRY_DIR)
    for cell in slate.cells:
        if cell.opportunity_pattern == "equity_subgroup_gap":
            assert S.protected_attributes(cell.dataset, REGISTRY_DIR), (
                f"{cell.candidate_id} puts an equity framing on "
                f"{cell.dataset}, which declares no protected attribute"
            )


def test_equity_on_a_dataset_without_protected_attributes_would_be_killed() -> None:
    """Rule S2 mirrors a real KILL - pin them together."""
    card = make_card(
        dataset="assistments_0910",
        task_type="psychometrics",
        pattern="equity_subgroup_gap",
    )
    report = screen_card(card, C.compile_spec(card, registry_dir=REGISTRY_DIR))
    assert report.verdict == F.KILL
    assert "F-NO-PROTECTED-ATTRS" in report.kill_codes
    allowed, evidence = S.pattern_allowed(
        "equity_subgroup_gap", "assistments_0910", registry_dir=REGISTRY_DIR
    )
    assert allowed is False
    assert "protected_attribute" in evidence


def test_bridge_framing_quota_enforced() -> None:
    slate = S.build_slate(
        "T-0001", n_candidates=24, bridge_quota=3, registry_dir=REGISTRY_DIR
    )
    ledger = slate.diversity_ledger()
    bridge = ledger["opportunity_patterns"].get(S.BRIDGE_PATTERN, 0)
    assert bridge <= 3
    assert ledger["bridge_share"] <= S.BRIDGE_MAX_SHARE


def test_bridge_quota_of_zero_produces_no_bridge_cards() -> None:
    slate = S.build_slate(
        "T-0001", n_candidates=24, bridge_quota=0, registry_dir=REGISTRY_DIR
    )
    assert slate.diversity_ledger()["opportunity_patterns"].get(
        S.BRIDGE_PATTERN, 0
    ) == 0


def test_per_cell_cap_enforced() -> None:
    slate = S.build_slate(
        "T-0001", n_candidates=24, max_per_cell=2, registry_dir=REGISTRY_DIR
    )
    counts: dict[tuple[str, str], int] = {}
    for cell in slate.cells:
        key = (cell.dataset, cell.task_type)
        counts[key] = counts.get(key, 0) + 1
    assert counts
    assert max(counts.values()) <= 2


def test_every_core_pattern_is_covered_at_the_default_size() -> None:
    slate = S.build_slate("T-0001", n_candidates=24, registry_dir=REGISTRY_DIR)
    ledger = slate.diversity_ledger()
    assert ledger["core_patterns_covered"] == ledger["core_patterns_total"]


def test_every_quota_decision_carries_evidence() -> None:
    slate = S.build_slate("T-0001", registry_dir=REGISTRY_DIR)
    assert slate.quota_decisions
    for decision in slate.quota_decisions:
        assert decision.rule.startswith("Q-")
        assert decision.decision.strip()
        assert decision.evidence.strip()


def test_slate_is_shorter_than_requested_rather_than_over_filling_a_cell() -> None:
    """2 enumerable cells x cap 3 = 6 slots; asking for 10 must not
    quietly put 5 candidates in one cell."""
    slate = S.build_slate(
        "T-0001",
        n_candidates=10,
        datasets=["assistments_0910"],
        max_per_cell=3,
        registry_dir=REGISTRY_DIR,
    )
    assert len(slate.cells) == 6
    assert slate.n_requested == 10
    dropped = [d for d in slate.quota_decisions if d.rule == "Q-SLOT-DROPPED"]
    assert len(dropped) == 4
    substituted = [
        d for d in slate.quota_decisions if d.rule == "Q-PATTERN-SUBSTITUTION"
    ]
    assert substituted
    assert "protected_attribute" in substituted[0].evidence


def test_slate_json_round_trips() -> None:
    slate = S.build_slate("T-0001", registry_dir=REGISTRY_DIR)
    payload = json.loads(json.dumps(slate.to_dict()))
    assert payload["n_enumerated"] == len(slate.cells)
    assert len(payload["cells"]) == len(slate.cells)
    assert payload["diversity_ledger"]["n_candidates"] == len(slate.cells)


def test_pattern_allocation_never_exceeds_the_requested_size() -> None:
    for n in (1, 5, 8, 13, 24, 40):
        sequence, _decisions = S.allocate_patterns(n, bridge_quota=3)
        assert len(sequence) == n
        assert sequence.count(S.BRIDGE_PATTERN) <= 3


# ==========================================================================
# 2. Cards: fixed-template render
# ==========================================================================


def test_render_has_every_section_label() -> None:
    card = make_card()
    rendered = card.render()
    for label in (
        "Question:",
        "Why it matters:",
        "What we would do:",
        "What would count as the result:",
        "Feasibility:",
    ):
        assert label in rendered


def test_render_word_cap_is_hard() -> None:
    long_text = " ".join(f"word{i}" for i in range(500))
    card = make_card(
        research_question=long_text,
        why_it_matters=long_text,
        what_we_would_do=long_text,
        what_counts_as_the_result=long_text,
    )
    assert card.render_word_count() <= C.RENDER_WORD_CAP
    # ... and the template survives truncation
    assert "What would count as the result:" in card.render()


def test_render_length_does_not_reward_verbosity() -> None:
    long_text = " ".join(f"word{i}" for i in range(500))
    verbose = make_card(research_question=long_text, why_it_matters=long_text)
    terse = make_card()
    assert verbose.render_word_count() <= C.RENDER_WORD_CAP
    assert terse.render_word_count() <= C.RENDER_WORD_CAP
    assert verbose.render().count("\n") == terse.render().count("\n")


def test_render_is_deterministic() -> None:
    card = make_card()
    assert card.render() == card.render()


def test_render_reports_the_measured_n_and_verdict() -> None:
    card = make_card()
    report = F.FeasibilityReport(
        candidate_id="C-01", verdict=F.CLEAN, analytic_n_estimate=12960
    )
    line = card.render(report).splitlines()[-1]
    assert "12,960" in line
    assert "CLEAN" in line


def test_render_says_not_screened_when_it_was_not() -> None:
    assert "not screened" in make_card().render()


def test_resolve_target_mirrors_the_feasibility_resolver() -> None:
    """cards.resolve_target duplicates a private helper on purpose; if
    the two ever disagree, dedupe silently changes meaning."""
    specs = [
        {"outcome_variable": "X4EVRATNDCLG"},
        {"outcome": {"variable": "F2EVRATT"}},
        {"treatment": {"variable": "X1MTHEFF"}},
        {"scale_name": "HSLS math self-efficacy"},
        {},
        {"outcome_variable": "A", "treatment": {"variable": "B"}},
    ]
    for spec in specs:
        assert C.resolve_target(spec) == F._resolved_target(spec)


# ==========================================================================
# 3. C1 - no novelty number, anywhere
# ==========================================================================


def test_from_dict_drops_a_generator_supplied_novelty_score() -> None:
    card = IdeaCard.from_dict(
        {
            "candidate_id": "C-01",
            "tournament_id": "T-0001",
            "cell": {"dataset": "hsls09_public", "task_type": "prediction"},
            "research_question": "q",
            "novelty_score_self_assessment": 5,
            "spec_draft": {"novelty_score_self_assessment": 4, "outcome_variable": "X1SES"},
        }
    )
    assert not hasattr(card, "novelty_score_self_assessment")
    assert "novelty_score_self_assessment" not in card.spec_draft
    assert any("C1" in note for note in card.notes)


def test_compiled_spec_carries_no_novelty_field() -> None:
    card = make_card(spec_draft={"novelty_score_self_assessment": 5})
    spec = C.compile_spec(card, registry_dir=REGISTRY_DIR)
    assert not [key for key in spec if "novel" in key.lower()]


def test_generator_prompt_forbids_a_novelty_number() -> None:
    cell = S.SlateCell(
        candidate_id="C-01",
        dataset="hsls09_public",
        task_type="prediction",
        opportunity_pattern="puzzle_anomaly",
        persona="policy_analyst",
    )
    message = G.build_user_message(cell, registry_dir=REGISTRY_DIR)
    assert "novelty score" in message.lower()
    assert "first study" in message.lower()


# ==========================================================================
# 4. compile_spec and the SEAM
# ==========================================================================


@pytest.mark.parametrize("task_type,dataset", SEAM_PAIRS)
def test_winner_spec_loads_unchanged(
    task_type: str, dataset: str, tmp_path: Path
) -> None:
    """The seam, tested with the exact function the CLI uses."""
    from src.main import load_locked_research_spec

    card = make_card(dataset=dataset, task_type=task_type)
    spec = C.compile_spec(card, registry_dir=REGISTRY_DIR)
    path = tmp_path / f"{task_type}_{dataset}.json"
    path.write_text(json.dumps(spec, indent=1), encoding="utf-8")
    loaded = load_locked_research_spec(str(path))
    assert loaded["task_type"] == task_type
    assert loaded["dataset"] == dataset


def test_compile_spec_always_emits_task_type_and_dataset() -> None:
    """The T0 hand-off seam warning: archived prediction specs carry
    neither, and load_locked_research_spec guards on task_type."""
    for task_type, dataset in SEAM_PAIRS:
        spec = C.compile_spec(
            make_card(dataset=dataset, task_type=task_type),
            registry_dir=REGISTRY_DIR,
        )
        assert spec["task_type"] == task_type
        assert spec["dataset"] == dataset


def test_compile_spec_with_a_realistic_draft_loads(tmp_path: Path) -> None:
    from src.main import load_locked_research_spec

    card = make_card(
        dataset="hsls09_public",
        task_type="causal_soo",
        spec_draft={
            "treatment": {
                "variable": "X1MTHEFF",
                "operationalization": "median_split_binary",
            },
            "outcome": {"variable": "X4EVRATNDCLG", "type": "binary"},
            "target_estimand_hint": "ATT",
            "primary_method": "M2",
            "secondary_methods": ["M3", "M4"],
            "adjustment_set": ["X1SES", "X1TXMTSCOR", "X1SEX"],
        },
    )
    spec = C.compile_spec(card, registry_dir=REGISTRY_DIR)
    assert spec["treatment"]["variable"] == "X1MTHEFF"
    assert spec["secondary_methods"] == ["M3", "M4"]
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    load_locked_research_spec(str(path))


def test_compile_spec_completes_but_does_not_correct_a_variable() -> None:
    card = make_card(
        spec_draft={
            "outcome_variable": "X4EVRATNDCLG",
            "outcome_type": "binary",
            "predictor_set": [
                {"variable": "X9NOTAREALVARIABLE", "wave": "base_year"},
                {"variable": "X1SES", "wave": "base_year"},
            ],
        }
    )
    spec = C.compile_spec(card, registry_dir=REGISTRY_DIR)
    named = [p["variable"] for p in spec["predictor_set"]]
    assert "X9NOTAREALVARIABLE" in named, (
        "compile_spec must not silently drop an invented variable: the "
        "screen has to be able to kill it"
    )
    report = screen_card(card, spec)
    assert "F-VAR-ABSENT" in report.kill_codes + report.warn_codes


def test_compile_spec_completes_but_does_not_correct_an_estimator() -> None:
    card = make_card(
        task_type="causal_soo", spec_draft={"primary_method": "RD"}
    )
    spec = C.compile_spec(card, registry_dir=REGISTRY_DIR)
    assert spec["primary_method"] == "RD"
    report = screen_card(card, spec)
    assert report.verdict == F.KILL
    assert "F-ESTIMATOR-UNCERTIFIED" in report.kill_codes


def test_compile_spec_fills_a_missing_wave_from_the_registry() -> None:
    card = make_card(
        spec_draft={
            "outcome_variable": "X4EVRATNDCLG",
            "predictor_set": [{"variable": "X1SES"}],
        }
    )
    spec = C.compile_spec(card, registry_dir=REGISTRY_DIR)
    assert spec["predictor_set"][0]["wave"] == "base_year"


def test_every_completion_note_names_what_it_read() -> None:
    for task_type, dataset in SEAM_PAIRS:
        spec = C.compile_spec(
            make_card(dataset=dataset, task_type=task_type),
            registry_dir=REGISTRY_DIR,
        )
        notes = spec["compiled_by"]["completion_notes"]
        assert notes, f"{task_type}/{dataset} completed nothing"
        for note in notes:
            assert "[read: " in note, note


def test_compiled_spec_carries_the_measured_n_when_probed() -> None:
    card = make_card()
    report = F.FeasibilityReport(
        candidate_id="C-01", verdict=F.CLEAN, analytic_n_estimate=12960
    )
    spec = C.compile_spec(card, report, registry_dir=REGISTRY_DIR)
    assert "12,960" in spec["expected_contribution"]


# ==========================================================================
# 5. Generation: one independent draw per cell
# ==========================================================================


@pytest.fixture()
def small_slate() -> S.Slate:
    return S.build_slate(
        "T-0001", n_candidates=6, seed=42, registry_dir=REGISTRY_DIR
    )


def test_one_call_per_cell_and_no_cross_candidate_context(
    small_slate: S.Slate,
) -> None:
    stub = StubCaller()
    result = G.generate_cards(
        small_slate,
        call_llm=stub,
        registry_dir=REGISTRY_DIR,
        dedupe_cosine=1.01,
    )
    assert len(stub.prompts) == len(small_slate.cells)
    assert len(result.cards) == len(small_slate.cells)
    # Each prompt names its own cell and no other candidate's card.
    for prompt, cell in zip(stub.prompts, small_slate.cells):
        assert f"dataset: {cell.dataset}" in prompt
        assert f"task_type: {cell.task_type}" in prompt
        assert "Question 0:" not in prompt  # no previous draw leaked in


def test_cards_inherit_the_cell_not_the_models_opinion(
    small_slate: S.Slate,
) -> None:
    def responder(message: str, index: int) -> str:
        return json.dumps(
            {
                "research_question": f"q{index}",
                "why_it_matters": "w",
                "what_we_would_do": "d",
                "what_counts_as_the_result": "r",
                "spec_draft": {
                    "dataset": "some_other_dataset",
                    "task_type": "causal_did",
                },
            }
        )

    result = G.generate_cards(
        small_slate,
        call_llm=StubCaller(responder),
        registry_dir=REGISTRY_DIR,
        dedupe_cosine=1.01,
    )
    for card, cell in zip(result.cards, small_slate.cells):
        assert card.dataset == cell.dataset
        assert card.task_type == cell.task_type
        assert card.spec_draft["dataset"] == cell.dataset
        assert card.spec_draft["task_type"] == cell.task_type


def test_unparseable_response_is_killed_with_a_code_and_evidence(
    small_slate: S.Slate,
) -> None:
    def responder(message: str, index: int) -> str:
        return "I'm afraid I can't do that." if index < 2 else _default_responder(
            message, index
        )

    result = G.generate_cards(
        small_slate,
        call_llm=StubCaller(responder),
        registry_dir=REGISTRY_DIR,
        dedupe_cosine=1.01,
        max_attempts=1,
    )
    killed = [row for row in result.killed if row["kill_code"] == "G-NO-CARD"]
    assert len(killed) == 2
    for row in killed:
        assert row["evidence"].strip()
        assert row["cell"]["dataset"]
    assert result.parse_failures == 2


def test_a_retry_is_attempted_before_killing(small_slate: S.Slate) -> None:
    state = {"failed": 0}

    def responder(message: str, index: int) -> str:
        if "could not be parsed" not in message and state["failed"] < 1:
            state["failed"] += 1
            return "not json"
        return _default_responder(message, index)

    stub = StubCaller(responder)
    result = G.generate_cards(
        small_slate, call_llm=stub, registry_dir=REGISTRY_DIR,
        dedupe_cosine=1.01, max_attempts=2,
    )
    assert len(result.cards) == len(small_slate.cells)
    assert len(stub.prompts) == len(small_slate.cells) + 1


def test_call_exception_is_killed_not_raised(small_slate: S.Slate) -> None:
    def responder(message: str, index: int) -> str:
        raise TimeoutError("provider timed out")

    result = G.generate_cards(
        small_slate,
        call_llm=StubCaller(responder),
        registry_dir=REGISTRY_DIR,
        max_attempts=1,
    )
    assert result.cards == []
    assert len(result.killed) == len(small_slate.cells)
    assert all("TimeoutError" in row["evidence"] for row in result.killed)


def test_empty_research_question_is_killed(small_slate: S.Slate) -> None:
    def responder(message: str, index: int) -> str:
        return json.dumps({"research_question": "", "why_it_matters": "w"})

    result = G.generate_cards(
        small_slate, call_llm=StubCaller(responder), registry_dir=REGISTRY_DIR
    )
    assert result.cards == []
    assert {row["kill_code"] for row in result.killed} == {"G-EMPTY-QUESTION"}


def test_parse_response_handles_fences_and_prose() -> None:
    assert G.parse_response('```json\n{"a": 1}\n```') == {"a": 1}
    assert G.parse_response('Here you go:\n{"a": {"b": 2}}\nHope that helps') == {
        "a": {"b": 2}
    }
    with pytest.raises(ValueError):
        G.parse_response("no json at all")


# ==========================================================================
# 6. Dedupe
# ==========================================================================


def test_structural_duplicates_are_removed() -> None:
    draft = {"outcome_variable": "X4EVRATNDCLG"}
    a = make_card(candidate_id="C-01", spec_draft=dict(draft))
    b = make_card(
        candidate_id="C-02",
        spec_draft=dict(draft),
        research_question="Completely different words about a different thing",
        why_it_matters="Nothing in common with the first card at all",
        what_we_would_do="An unrelated procedure entirely",
        what_counts_as_the_result="Some other observable",
    )
    kept, killed = G.dedupe([a, b])
    assert [card.candidate_id for card in kept] == ["C-01"]
    assert killed[0]["kill_code"] == "D-DUPLICATE"
    assert "structural key equal" in killed[0]["evidence"]
    assert killed[0]["detail"]["duplicate_of"] == "C-01"


def test_lexical_duplicates_are_removed() -> None:
    a = make_card(candidate_id="C-01", spec_draft={"outcome_variable": "A"})
    b = make_card(candidate_id="C-02", spec_draft={"outcome_variable": "B"})
    kept, killed = G.dedupe([a, b], threshold=0.80)
    assert [card.candidate_id for card in kept] == ["C-01"]
    assert "cosine" in killed[0]["evidence"]


def test_distinct_cards_both_survive() -> None:
    a = make_card(
        candidate_id="C-01",
        spec_draft={"outcome_variable": "X4EVRATNDCLG"},
        research_question="Which ninth graders never enrol in college?",
        why_it_matters="Advisers need an early list.",
        what_we_would_do="Fit a battery on baseline covariates.",
        what_counts_as_the_result="An AUC gap across SES quintiles.",
    )
    b = make_card(
        candidate_id="C-02",
        dataset="els_2002",
        task_type="psychometrics",
        spec_draft={"scale_name": "ELS math self-efficacy"},
        research_question=(
            "Do the five self-efficacy items measure one construct equally "
            "for boys and girls?"
        ),
        why_it_matters="Every downstream regression assumes they do.",
        what_we_would_do="Graded response calibration plus a DIF sweep.",
        what_counts_as_the_result="At least one item with large uniform DIF.",
    )
    kept, killed = G.dedupe([a, b], threshold=0.80)
    assert len(kept) == 2
    assert killed == []


def test_unknown_targets_are_not_treated_as_the_same_target() -> None:
    """Kill discipline: two unresolvable targets are unknown, not equal."""
    a = make_card(candidate_id="C-01")
    b = make_card(candidate_id="C-02")
    assert G.structural_key(a) is None
    assert G.structural_key(b) is None
    flagged, _why = G.is_duplicate(a, b, cosine=0.0)
    assert flagged is False


# ==========================================================================
# 7. The generate stage end to end
# ==========================================================================


def _run_stage(tmp_path: Path, **kwargs: Any) -> dict:
    defaults: dict[str, Any] = dict(
        tournament_id="T-0001",
        out_dir=tmp_path / "T-0001",
        n_candidates=6,
        seed=42,
        registry_dir=REGISTRY_DIR,
        use_column_cache=False,
        dedupe_cosine=1.01,
        call_llm=StubCaller(),
        generator_model="test-stub",
    )
    defaults.update(kwargs)
    return R.run_generate_stage(**defaults)


def test_generate_stage_writes_every_artifact(tmp_path: Path) -> None:
    summary = _run_stage(tmp_path)
    out = tmp_path / "T-0001"
    for name in (
        "slate.json",
        "candidates.jsonl",
        "killed.jsonl",
        "feasibility.json",
        "ranking_deterministic.json",
        "generate_summary.json",
        "rank1_spec.json",
    ):
        assert (out / name).exists(), f"{name} was not written"
    assert summary["survivors"] > 0
    assert summary["slate_cells"] == 6


def test_candidates_carry_card_spec_feasibility_and_venue_fit(
    tmp_path: Path,
) -> None:
    _run_stage(tmp_path)
    rows = [
        json.loads(line)
        for line in (tmp_path / "T-0001" / "candidates.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert rows
    for row in rows:
        assert row["card"]["render_word_count"] <= C.RENDER_WORD_CAP
        assert row["spec"]["task_type"] == row["card"]["cell"]["task_type"]
        assert row["spec"]["dataset"] == row["card"]["cell"]["dataset"]
        assert row["feasibility"]["verdict"] in (F.CLEAN, F.WARN)
        assert "score" in row["venue_fit"]
        assert row["rank"] >= 1


def test_killed_jsonl_records_the_code_and_the_evidence(tmp_path: Path) -> None:
    """No rejected idea has ever been written to disk here before. Every
    rejection path must land in the file with a code AND its evidence."""

    def responder(message: str, index: int) -> str:
        if index == 0:
            return "unparseable"
        if index == 1:
            # An uncertified estimator: a KILL that needs no data on disk.
            return json.dumps(
                {
                    "research_question": "Can we exploit a discontinuity?",
                    "why_it_matters": "w",
                    "what_we_would_do": "d",
                    "what_counts_as_the_result": "r",
                    "spec_draft": {"primary_method": "RD"},
                }
            )
        return _default_responder(message, index)

    summary = _run_stage(
        tmp_path, call_llm=StubCaller(responder), max_attempts=1
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "T-0001" / "killed.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    codes = {row["kill_code"] for row in rows}
    assert "G-NO-CARD" in codes
    assert "F-ESTIMATOR-UNCERTIFIED" in codes
    for row in rows:
        assert row["kill_code"]
        assert row["evidence"].strip()
        assert row["cell"]
    assert summary["killed"] == len(rows)
    screened = [row for row in rows if row["stage"] == "feasibility_screen"]
    assert screened and screened[0]["card"] is not None


def test_ranking_is_deterministic(tmp_path: Path) -> None:
    first = _run_stage(tmp_path / "a")["diversity_ledger"]
    second = _run_stage(tmp_path / "b")["diversity_ledger"]
    assert first == second
    ranking_a = json.loads(
        (tmp_path / "a" / "T-0001" / "ranking_deterministic.json").read_text(
            encoding="utf-8"
        )
    )["ranking"]
    ranking_b = json.loads(
        (tmp_path / "b" / "T-0001" / "ranking_deterministic.json").read_text(
            encoding="utf-8"
        )
    )["ranking"]
    assert ranking_a == ranking_b


def test_ranking_is_invariant_to_an_injected_novelty_score(
    tmp_path: Path,
) -> None:
    """C1 as an executable assertion: perturbing the banned field across
    its whole range must not move the ordering by one position."""

    def with_novelty(value: Any) -> Callable[[str, int], str]:
        def responder(message: str, index: int) -> str:
            payload = json.loads(_default_responder(message, index))
            payload["novelty_score_self_assessment"] = value
            payload["spec_draft"] = {"novelty_score_self_assessment": value}
            return json.dumps(payload)

        return responder

    rankings = []
    for i, value in enumerate([1, 3, 5, 10, 0.43, {"score": 5}]):
        out = tmp_path / f"n{i}"
        _run_stage(out, call_llm=StubCaller(with_novelty(value)))
        payload = json.loads(
            (out / "T-0001" / "ranking_deterministic.json").read_text(
                encoding="utf-8"
            )
        )
        rankings.append(
            [row["candidate_id"] for row in payload["ranking"]]
        )
    assert len(set(map(tuple, rankings))) == 1, rankings


def test_a_spec_the_pipeline_cannot_load_sorts_last_but_is_not_killed(
    tmp_path: Path,
) -> None:
    """R7 guard. An ITR card whose generator named M2 survives the screen
    (M2 is certified for causal_itr) but the loader rejects it, because
    the ITR template requires M6. It must be demoted, recorded, and kept.
    """

    def responder(message: str, index: int) -> str:
        payload = json.loads(_default_responder(message, index))
        if index == 0:
            payload["spec_draft"] = {"primary_method": "M2"}
        return json.dumps(payload)

    summary = _run_stage(
        tmp_path,
        n_candidates=4,
        datasets=["hsls09_public", "els_2002"],
        task_types=["causal_itr"],
        call_llm=StubCaller(responder),
    )
    payload = json.loads(
        (tmp_path / "T-0001" / "ranking_deterministic.json").read_text(
            encoding="utf-8"
        )
    )
    rows = payload["ranking"]
    assert len(rows) == 4, "nothing may be killed by this path"
    bad = next(row for row in rows if row["spec_loads"] is False)
    assert bad["candidate_id"] == "C-01"
    assert bad["rank"] == len(rows)
    assert any("REJECTED" in line for line in bad["evidence"]["seam"])
    assert summary["seam_check_rank1"]["passed"] is True


def test_rank1_spec_passes_the_real_loader(tmp_path: Path) -> None:
    summary = _run_stage(tmp_path)
    seam = summary["seam_check_rank1"]
    assert seam["checked"] is True
    assert seam["passed"] is True, seam.get("error")
    assert seam["loader"] == "src.main.load_locked_research_spec"


def test_ranking_records_the_evidence_behind_both_terms(
    tmp_path: Path,
) -> None:
    _run_stage(tmp_path)
    payload = json.loads(
        (tmp_path / "T-0001" / "ranking_deterministic.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["method"] == "deterministic_only"
    assert "novelty" in payload["method_note"].lower()
    for row in payload["ranking"]:
        assert set(row["evidence"]) == {
            "venue_fit",
            "feasibility_penalty",
            "seam",
        }
        for line in row["evidence"]["venue_fit"]:
            assert "anchor evidence:" in line
        for line in row["evidence"]["feasibility_penalty"]:
            assert "[read:" in line


def test_rank_survivors_orders_by_score_then_warns_then_id() -> None:
    def entry(cid: str, score: float, penalty: float, warns: int) -> dict:
        return {
            "candidate_id": cid,
            "feasibility": {
                "penalty": penalty,
                "verdict": F.WARN if warns else F.CLEAN,
                "checks": [
                    {
                        "code": f"F-W{i}",
                        "status": "WARN",
                        "message": "m",
                        "evidence": "e",
                        "penalty": 0.0,
                    }
                    for i in range(warns)
                ],
            },
            "venue_fit": {"score": score, "hits": []},
            "card": {"cell": {}},
        }

    rows = R.rank_survivors(
        [
            entry("C-03", 1.0, 0.0, 0),
            entry("C-01", 3.0, 0.0, 0),
            entry("C-02", 3.0, 0.0, 2),
            entry("C-04", 3.0, 0.0, 0),
        ],
        weight_vf=1.0,
        weight_pen=1.0,
    )
    assert [row["candidate_id"] for row in rows] == [
        "C-01",
        "C-04",
        "C-02",
        "C-03",
    ]
    assert rows[0]["rank"] == 1


def test_stage_survives_a_slate_with_no_enumerable_cell(tmp_path: Path) -> None:
    summary = _run_stage(
        tmp_path,
        datasets=["assistments_0910"],
        task_types=["causal_did"],
    )
    assert "error" in summary
    assert (tmp_path / "T-0001" / "slate.json").exists()


def test_offline_stage_needs_no_llm_and_no_api_key(tmp_path: Path) -> None:
    summary = R.run_generate_stage(
        tournament_id="T-0002",
        out_dir=tmp_path / "T-0002",
        n_candidates=4,
        seed=42,
        registry_dir=REGISTRY_DIR,
        use_column_cache=False,
        offline=True,
    )
    assert summary["generator_model"] == G.OFFLINE_MODEL_ID
    assert summary["survivors"] > 0
    assert any("offline mode" in note for note in summary["notes"])


# ==========================================================================
# 8. Production routing: BaseAgent.call_llm and nothing else
# ==========================================================================


def test_generator_model_comes_from_config_not_from_code() -> None:
    assert G.resolve_generator_model({}) is None
    assert (
        G.resolve_generator_model(
            {"ideation": {"models": {"generator": "some-model-v9"}}}
        )
        == "some-model-v9"
    )


def test_generation_temperature_is_fixed_unless_config_overrides() -> None:
    assert G.generation_temperature({}) == 0.9
    assert (
        G.generation_temperature(
            {"ideation": {"tournament": {"generation_temperature": 0.5}}}
        )
        == 0.5
    )


def test_production_caller_routes_through_base_agent_call_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No provider SDK is called here: BaseAgent.call_llm is patched out.

    What is being pinned is the routing rule - the production path must
    not reach a client directly - and the fixed 0.9 temperature.
    """
    pytest.importorskip("openai")
    from src.agents.base import BaseAgent

    seen: dict[str, Any] = {}

    def fake_call_llm(
        self: Any,
        user_message: str,
        max_tokens: int | None = None,
        temperature_override: float | None = None,
    ) -> str:
        seen["message"] = user_message
        seen["temperature"] = temperature_override
        seen["model"] = self.model
        return json.dumps({"research_question": "q"})

    monkeypatch.setattr(BaseAgent, "call_llm", fake_call_llm)

    config = {
        "llm_provider": "deepseek",
        "deepseek": {
            "base_url": "https://api.deepseek.com",
            "models": {"idea_generator": "configured-model-id"},
        },
        "models": {},
        "paths": {"agent_prompts": str(REPO_ROOT / "agent_prompts")},
        "sandbox": {"enabled": False},
    }
    caller, model = G.make_llm_caller(config, dataset="hsls09_public")
    assert model == "configured-model-id"
    assert caller("hello") == '{"research_question": "q"}'
    assert seen["temperature"] == G.GENERATION_TEMPERATURE
    assert seen["message"] == "hello"


def test_next_tournament_id_increments(tmp_path: Path) -> None:
    assert R.next_tournament_id(tmp_path) == "T-0001"
    (tmp_path / "T-0001").mkdir()
    (tmp_path / "T-0007").mkdir()
    assert R.next_tournament_id(tmp_path) == "T-0008"


def test_cli_parses_the_generate_stage() -> None:
    args = R.build_parser().parse_args(
        ["--stage", "generate", "--offline", "--n-candidates", "4"]
    )
    assert args.stage == "generate"
    assert args.offline is True
    assert args.n_candidates == 4
