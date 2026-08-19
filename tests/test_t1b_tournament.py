"""Arc T / T1b - pairwise judge, Bradley-Terry, and the cascade.

Everything here is OFFLINE. No test constructs a provider client, opens
a socket, or reads a raw data file: the judge is a stub callable, the
feasibility reports are hand-built, and no candidate is screened against
a CSV.

The obligations, in priority order:

1. **The judged layer is removable (C3).** With every judged verdict
   deleted, and again with every judged verdict shuffled, the cascade
   still emits a complete deterministic ordering over the same field.
2. **Advisory mode is enforced in code, not prose.** Every artifact
   carries both orderings and the uncleared V2 verdict, and a caller
   that asks for a live selection gets an exception.
3. **BT recovers a known ordering** from synthetic matches, and is
   byte-reproducible.
4. **Position bias is recorded, not hidden** (C4).
5. **C1**: no novelty judgement exists at any weight, and a model that
   tries to supply one is refused.
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

from src.ideation import bradley_terry as BT  # noqa: E402
from src.ideation import judge as J  # noqa: E402
from src.ideation import tournament as T  # noqa: E402
from src.ideation.cards import IdeaCard  # noqa: E402

PROMPT_PATH = REPO_ROOT / "agent_prompts" / "idea_judge.yaml"


# --------------------------------------------------------------------------
# Fixtures: hand-built candidates, no registry and no data on disk
# --------------------------------------------------------------------------


def make_card(
    cid: str,
    *,
    dataset: str = "hsls09_public",
    task_type: str = "prediction",
    pattern: str = "equity_subgroup_gap",
    question: str | None = None,
) -> dict:
    card = IdeaCard(
        candidate_id=cid,
        tournament_id="T-9999",
        cell={
            "dataset": dataset,
            "task_type": task_type,
            "opportunity_pattern": pattern,
            "persona": "equity_researcher",
        },
        research_question=question or f"Question for {cid} on {dataset}.",
        why_it_matters=f"{cid} matters because a named decision changes.",
        what_we_would_do=f"{cid} would fit the registry-default battery.",
        what_counts_as_the_result=f"{cid} reports an interval.",
        resolved_target=f"OUTCOME_{cid[-1]}",
        method_family="prediction_ml",
        generated_at="2026-07-25T00:00:00Z",
        generator_model="test-generator",
    )
    return card.to_dict()


def make_record(
    cid: str,
    *,
    venue_fit: float = 0.0,
    penalty: float = 0.0,
    warns: int = 0,
    verdict: str = "CLEAN",
    dataset: str = "hsls09_public",
    task_type: str = "prediction",
    pattern: str = "equity_subgroup_gap",
    spec_loads: bool | None = True,
) -> dict:
    checks = [
        {
            "code": f"F-TEST-WARN-{i}",
            "status": "WARN",
            "message": "synthetic warn",
            "evidence": "hand-built fixture, no registry was read",
            "penalty": 0.0,
        }
        for i in range(warns)
    ]
    record: dict[str, Any] = {
        "candidate_id": cid,
        "card": make_card(cid, dataset=dataset, task_type=task_type, pattern=pattern),
        "spec": {
            "task_id": cid,
            "task_type": task_type,
            "dataset": dataset,
            "outcome_variable": f"OUTCOME_{cid[-1]}",
            "research_question": f"Question for {cid}.",
        },
        "feasibility": {
            "candidate_id": cid,
            "verdict": verdict,
            "dataset": dataset,
            "task_type": task_type,
            "analytic_n_estimate": 12960,
            "penalty": penalty,
            "checks": checks,
        },
        "venue_fit": {"score": venue_fit, "venue": "EDM", "hits": [], "facts": {}},
    }
    if spec_loads is not None:
        record["seam_check"] = {
            "checked": True,
            "passed": spec_loads,
            "loader": "src.main.load_locked_research_spec",
        }
    return record


def field_of(n: int = 4) -> list[dict]:
    """A small field with distinct deterministic scores."""
    return [
        make_record(f"C-0{i}", venue_fit=float(n - i), penalty=0.0)
        for i in range(1, n + 1)
    ]


# --------------------------------------------------------------------------
# Stub judges
# --------------------------------------------------------------------------


def _verdict_json(choice: str, evidence: str = "stub evidence") -> str:
    return json.dumps(
        {key: {"evidence": evidence, "winner": choice} for key in J.ALL_KEYS}
    )


def preference_judge(
    order: list[str], *, position_bias_on: tuple[str, str] | None = None
) -> Callable[[str], str]:
    """A judge with a fixed, known preference over candidate ids.

    It reads the two ``Idea N  (facets)`` blocks out of the user message
    and answers for whichever candidate comes earlier in ``order``. The
    candidate id is not in the message by construction (that is the
    point of :func:`judge.render_side`), so the stub keys on the research
    question text, which does carry the id.
    """
    rank = {cid: i for i, cid in enumerate(order)}

    def _who(block: str) -> str | None:
        for cid in order:
            if cid in block:
                return cid
        return None

    def _call(user_message: str) -> str:
        blocks = user_message.split("Idea 2")
        first = _who(blocks[0])
        second = _who(blocks[1]) if len(blocks) > 1 else None
        if first is None or second is None:
            return _verdict_json(J.TIE, "stub could not identify both ideas")
        if position_bias_on and {first, second} == set(position_bias_on):
            # Always prefers whatever is shown FIRST: the pure position
            # bias case, which the aggregate must surface rather than
            # average away.
            return _verdict_json("1", "stub is position-biased on this pair")
        better = first if rank[first] < rank[second] else second
        return _verdict_json(
            "1" if better is first else "2",
            f"stub prefers {better} by fixed order",
        )

    return _call


# --------------------------------------------------------------------------
# 1. Bradley-Terry
# --------------------------------------------------------------------------


def test_bt_recovers_known_ordering_from_synthetic_matches() -> None:
    """A round-robin where a fixed order always wins is recovered exactly."""
    truth = ["A", "B", "C", "D", "E"]
    rank = {cid: i for i, cid in enumerate(truth)}
    matches = [
        {"pair": [x, y], "winner": x if rank[x] < rank[y] else y}
        for x, y in T.round_robin(truth)
        for _ in range(6)  # 2 orientations x k=3, as the judge would emit
    ]
    posterior = BT.fit(matches)
    assert BT.strength_order(posterior) == truth
    # Strengths are strictly decreasing along the true order.
    values = [posterior.strength[c] for c in truth]
    assert all(values[i] > values[i + 1] for i in range(len(values) - 1))
    assert posterior.n_matches == len(matches)
    assert all(posterior.sd[c] > 0 for c in truth)


def test_bt_is_deterministic_given_the_same_match_set() -> None:
    matches = [
        {"pair": ["A", "B"], "winner": "A"},
        {"pair": ["B", "C"], "winner": "B"},
        {"pair": ["A", "C"], "winner": "A"},
        {"pair": ["A", "C"], "winner": "C"},
    ]
    first = BT.fit(matches, prior_means={"A": 0.3, "B": 0.0, "C": -0.2})
    second = BT.fit(matches, prior_means={"A": 0.3, "B": 0.0, "C": -0.2})
    assert first.to_dict() == second.to_dict()
    assert json.dumps(first.to_dict(), sort_keys=True) == json.dumps(
        second.to_dict(), sort_keys=True
    )
    # And insensitive to the order the same matches arrive in.
    shuffled = [matches[i] for i in (2, 0, 3, 1)]
    third = BT.fit(shuffled, prior_means={"A": 0.3, "B": 0.0, "C": -0.2})
    for cid in ("A", "B", "C"):
        assert first.strength[cid] == pytest.approx(third.strength[cid], abs=1e-9)


def test_bt_prior_carries_a_candidate_with_no_matches() -> None:
    """A candidate nobody played still gets a strength: its prior."""
    posterior = BT.fit(
        [{"pair": ["A", "B"], "winner": "A"}],
        prior_means={"A": 0.0, "B": 0.0, "Z": 0.9},
        candidates=["A", "B", "Z"],
    )
    assert set(posterior.candidates) == {"A", "B", "Z"}
    assert posterior.strength["Z"] == pytest.approx(0.9, abs=1e-9)
    assert posterior.matches_per_candidate["Z"] == 0
    # Its SD is the prior SD exactly - no data, no shrinkage.
    assert posterior.sd["Z"] == pytest.approx(posterior.prior_sd, abs=1e-9)


def test_bt_ties_land_between_the_two_candidates() -> None:
    posterior = BT.fit([{"pair": ["A", "B"], "winner": None}] * 6)
    assert posterior.strength["A"] == pytest.approx(posterior.strength["B"], abs=1e-9)


def test_bt_weights_make_a_human_pair_outweigh_judge_matches() -> None:
    """Spec sec. 5.6 path A: one human pair at weight 5 beats 3 judge wins."""
    judge_rows = [
        {"pair": ["A", "B"], "winner": "A", "source": "judge"} for _ in range(3)
    ]
    human = [{"pair": ["A", "B"], "winner": "B", "source": "human"}]
    posterior = BT.fit(judge_rows + human, {"judge": 1.0, "human": 5.0})
    assert posterior.strength["B"] > posterior.strength["A"]


def test_bt_drops_unusable_rows_and_says_why() -> None:
    posterior = BT.fit(
        [
            {"pair": ["A", "B"], "winner": "A"},
            {"pair": ["A", "B"], "winner": "Q"},  # winner not in the pair
            {"pair": ["A", "A"], "winner": "A"},  # self-match
            {"pair": ["A", "B"]},  # no outcome at all
        ]
    )
    assert posterior.n_matches == 1
    assert len(posterior.dropped) == 3
    reasons = " ".join(row["reason"] for row in posterior.dropped)
    assert "winner names neither candidate" in reasons
    assert "self-match" in reasons
    assert "no winner/y field" in reasons


def test_top_k_membership_is_deterministic_and_ordered() -> None:
    truth = ["A", "B", "C", "D"]
    rank = {cid: i for i, cid in enumerate(truth)}
    matches = [
        {"pair": [x, y], "winner": x if rank[x] < rank[y] else y}
        for x, y in T.round_robin(truth)
        for _ in range(6)
    ]
    posterior = BT.fit(matches)
    first = BT.top_k_membership(posterior, k=2, n_draws=500, seed=42)
    second = BT.top_k_membership(posterior, k=2, n_draws=500, seed=42)
    assert first == second
    assert sum(first.values()) == pytest.approx(2.0, abs=1e-9)
    assert first["A"] > first["C"]
    assert first["A"] > first["D"]
    # A different seed is allowed to differ, but not wildly.
    other = BT.top_k_membership(posterior, k=2, n_draws=500, seed=7)
    assert abs(other["A"] - first["A"]) < 0.2


# --------------------------------------------------------------------------
# 2. The judge: orientation, sampling, evidence, C1
# --------------------------------------------------------------------------


def test_judge_prompt_lives_in_agent_prompts_and_is_ascii() -> None:
    assert PROMPT_PATH.exists(), "the judge system prompt must live in a YAML file"
    text = PROMPT_PATH.read_text(encoding="utf-8")
    assert text.isascii(), "non-ASCII in a prompt YAML has broken this repo before"
    import yaml

    data = yaml.safe_load(text)
    prompt = data["system_prompt"]
    assert "novelty" in prompt.lower(), "the C1 prohibition must be stated"
    assert "evidence" in prompt.lower()
    # C4: the k samples must be samples of something.
    assert float(data["temperature"]) > 0.0


def test_judge_never_shows_the_candidate_id_to_the_model() -> None:
    card = make_card("C-01")
    rendered = J.render_side(card, label="Idea 1")
    assert rendered.startswith("Idea 1")
    assert "[C-01]" not in rendered
    # The cell facets survive: they are legitimate context.
    assert "prediction" in rendered and "hsls09_public" in rendered


def test_judge_truncates_both_sides_to_the_same_word_budget() -> None:
    """A 400-word draft must not reach the judge as a 400-word card.

    Verbosity attacks work on weak judges, so length must carry zero
    signal: both sides go through the same fixed template and the same
    cap before anything is compared.
    """
    short = make_card("C-01")
    long_card = make_card("C-02", question="word " * 400)
    cap = 120
    rendered_short = J.render_side(short, label="Idea 1", word_cap=cap)
    rendered_long = J.render_side(long_card, label="Idea 2", word_cap=cap)
    # Body (everything after the neutral header) is inside the budget.
    for rendered in (rendered_short, rendered_long):
        body = " ".join(rendered.split("\n")[1:])
        assert len(body.split()) <= cap
    # The 400-word draft buys at most the template's own field cap.
    assert len(rendered_long.split()) - len(rendered_short.split()) < 40

    message = J.build_user_message(short, long_card, word_cap=cap)
    assert "word word word word word word word word word word" in message
    assert message.count("word") < 60  # not the original 400


def test_judge_runs_both_orientations_k_times() -> None:
    calls: list[str] = []

    def _call(message: str) -> str:
        calls.append(message)
        return _verdict_json("1")

    result = J.judge_pair(
        make_card("C-01"),
        make_card("C-02"),
        call_llm=_call,
        samples=3,
        judge_model="stub-judge",
    )
    assert result.calls == 6  # 2 orientations x 3 samples
    assert len({m for m in calls}) == 2  # exactly two distinct messages
    overall = result.overall
    assert overall is not None
    assert overall.n_votes == 6


def test_orientation_disagreement_is_recorded_as_position_bias() -> None:
    """A judge that always picks whatever is shown first must be caught."""
    result = J.judge_pair(
        make_card("C-01"),
        make_card("C-02"),
        call_llm=preference_judge(
            ["C-01", "C-02"], position_bias_on=("C-01", "C-02")
        ),
        samples=3,
        judge_model="stub-judge",
    )
    overall = result.overall
    assert overall is not None
    # 3 votes each way: the split is visible, and no winner is invented.
    assert overall.votes == {"C-01": 3, "C-02": 3}
    assert overall.winner is None
    assert overall.orientation_winners == {"AB": "C-01", "BA": "C-02"}
    assert overall.position_bias is True
    assert overall.position_bias_strict is True
    assert "significance" in result.position_bias_dimensions

    run = J.JudgeRun(pairs=[result], judge_model="stub-judge")
    summary = run.summary()
    assert summary["position_bias_rate"] == 1.0
    assert summary["position_bias_strict_rate"] == 1.0
    assert summary["position_bias_pairs"] == ["C-01|C-02"]


def test_a_consistent_judge_shows_no_position_bias() -> None:
    result = J.judge_pair(
        make_card("C-01"),
        make_card("C-02"),
        call_llm=preference_judge(["C-02", "C-01"]),
        samples=3,
        judge_model="stub-judge",
    )
    overall = result.overall
    assert overall is not None
    assert overall.winner == "C-02"
    assert overall.votes == {"C-01": 0, "C-02": 6}
    assert overall.position_bias is False


def test_every_verdict_carries_an_evidence_string() -> None:
    result = J.judge_pair(
        make_card("C-01"),
        make_card("C-02"),
        call_llm=preference_judge(["C-01", "C-02"]),
        samples=2,
        judge_model="stub-judge",
    )
    assert result.verdicts
    for verdict in result.verdicts:
        assert verdict.evidence.strip(), "C2: a verdict with no evidence does not ship"
    for record in (v.to_dict() for v in result.verdicts):
        for key in ("pair", "orientation", "sample", "dimension", "winner",
                    "evidence", "judge_model"):
            assert key in record


def test_evidence_before_verdict_is_measured_not_assumed() -> None:
    compliant = json.dumps(
        {k: {"evidence": "e", "winner": "1"} for k in J.ALL_KEYS}
    )
    reversed_order = json.dumps(
        {k: {"winner": "1", "evidence": "e"} for k in J.ALL_KEYS}
    )
    assert J.parse_response(compliant)["overall"]["evidence_first"] is True
    assert J.parse_response(reversed_order)["overall"]["evidence_first"] is False


def test_judge_refuses_a_novelty_dimension_the_model_invents() -> None:
    """C1 is enforced in code, not only in the prompt."""
    payload = json.dumps(
        {
            "significance": {"evidence": "e", "winner": "1"},
            "overall": {"evidence": "e", "winner": "1"},
            "novelty": {"evidence": "unprecedented", "winner": "2"},
            "feasibility": {"evidence": "n is large", "winner": "2"},
        }
    )
    parsed = J.parse_response(payload)
    assert "novelty" not in parsed
    assert "feasibility" not in parsed
    assert parsed["_banned_keys"]["keys"] == ["feasibility", "novelty"]

    result = J.judge_pair(
        make_card("C-01"),
        make_card("C-02"),
        call_llm=lambda _message: payload,
        samples=1,
        judge_model="stub-judge",
    )
    assert {v.dimension for v in result.verdicts} <= set(J.ALL_KEYS)
    assert any(e["stage"] == "c1_guard" for e in result.errors)


def test_judged_dimensions_exclude_novelty_and_feasibility() -> None:
    for name in J.ALL_KEYS:
        assert "novel" not in name
        assert "feasib" not in name


def test_judge_survives_a_broken_response_and_records_it() -> None:
    def _call(_message: str) -> str:
        return "I am not JSON."

    result = J.judge_pair(
        make_card("C-01"), make_card("C-02"), call_llm=_call, samples=2
    )
    assert result.verdicts == []
    assert len(result.errors) == 4
    assert all(e["stage"] == "parse" for e in result.errors)
    overall = result.overall
    assert overall is not None and overall.winner is None


def test_offline_judge_answers_tie_and_says_so() -> None:
    raw = J.offline_caller()("anything")
    parsed = J.parse_response(raw)
    assert parsed["overall"]["choice"] == J.TIE
    assert "OFFLINE STUB" in parsed["overall"]["evidence"]


# --------------------------------------------------------------------------
# 3. Pairing
# --------------------------------------------------------------------------


def test_round_robin_below_the_threshold_and_swiss_above() -> None:
    small = [f"C-{i:02d}" for i in range(1, 6)]
    mode, pairs = T.build_pairs(small)
    assert mode == "round_robin"
    assert len(pairs) == 10  # 5 choose 2

    big = [f"C-{i:02d}" for i in range(1, 13)]
    mode, pairs = T.build_pairs(big)
    assert mode == "swiss"
    assert len(pairs) == 6  # one round of 12 entrants
    assert len({cid for pair in pairs for cid in pair}) == 12


def test_swiss_cascade_runs_the_configured_number_of_rounds() -> None:
    """A 12-entrant field must not silently become a 66-pair round-robin."""
    order = [f"C-{i:02d}" for i in range(1, 13)]
    records = [
        make_record(cid, venue_fit=float(12 - i))
        for i, cid in enumerate(order)
    ]
    result = run(records, call_llm=preference_judge(order), n_swiss_rounds=5)
    judge_summary = result.ranking["judge"]
    assert judge_summary["mode"] == "swiss"
    assert judge_summary["rounds"] == 5
    assert judge_summary["n_pairs"] == 30  # 6 pairs x 5 rounds
    assert judge_summary["calls"] == 30 * 6  # x 2 orientations x 3 samples
    # Every entrant is still ranked, including any that drew few matches.
    assert len(result.ranking["ranking"]) == 12
    assert len(result.ranking["ranking_deterministic"]) == 12


def test_swiss_pairing_is_deterministic_and_avoids_rematches() -> None:
    ids = [f"C-{i:02d}" for i in range(1, 9)]
    first = T.swiss_round(ids, None)
    again = T.swiss_round(ids, None)
    assert first == again
    played = {frozenset(p) for p in first}
    second = T.swiss_round(ids, {"C-01": 1.0, "C-03": 1.0}, played=set(played))
    assert not ({frozenset(p) for p in second} & played)


# --------------------------------------------------------------------------
# 4. The cascade - advisory mode and C3
# --------------------------------------------------------------------------


def run(records: list[dict], **kwargs: Any) -> T.TournamentResult:
    defaults: dict[str, Any] = {
        "tournament_id": "T-9999",
        "prior_art": False,
        "use_column_cache": False,
    }
    defaults.update(kwargs)
    return T.run_cascade(records, **defaults)


def test_cascade_publishes_both_rankings_and_the_uncleared_v2_verdict() -> None:
    result = run(
        field_of(4), call_llm=preference_judge(["C-04", "C-03", "C-02", "C-01"])
    )
    ranking = result.ranking

    assert ranking["advisory"] is True
    assert ranking["authorized_for_live_selection"] is False
    assert ranking["v2_status"]["cleared"] is False
    # The measured numbers travel with their n - never a bare value.
    for row in ranking["v2_status"]["measured"]:
        assert isinstance(row["n"], int)
        assert isinstance(row["value"], float)

    assert ranking["ranking"], "judged ordering must be present"
    assert ranking["ranking_deterministic"], "deterministic ordering must be present"
    judged_ids = {row["candidate_id"] for row in ranking["ranking"]}
    det_ids = {row["candidate_id"] for row in ranking["ranking_deterministic"]}
    assert judged_ids == det_ids, "both orderings must cover the same field"

    digest = result.digest()
    assert "ADVISORY" in digest
    assert "NOT validated" in digest
    assert "deterministic rank" in digest


def test_live_selection_is_refused_while_v2_is_uncleared() -> None:
    with pytest.raises(T.LiveSelectionNotAuthorized):
        run(field_of(3), allow_live_selection=True)


def test_no_winner_spec_is_written() -> None:
    result = run(field_of(3), call_llm=preference_judge(["C-01", "C-02", "C-03"]))
    assert result.ranking["winner_spec_written"] is False


def test_deterministic_ordering_survives_with_the_judged_layer_removed() -> None:
    """C3, stated as an executable claim.

    Three regimes over the same field: judged, judged-with-no-caller,
    and judging switched off entirely. Every one produces a complete
    ordering, and the deterministic ordering is identical in all three.
    """
    records = field_of(5)
    with_judge = run(
        records, call_llm=preference_judge(["C-05", "C-04", "C-03", "C-02", "C-01"])
    )
    no_caller = run(records)
    judging_off = run(records, judged=False)

    expected = [row["candidate_id"] for row in with_judge.ranking[
        "ranking_deterministic"
    ]]
    assert len(expected) == 5
    for other in (no_caller, judging_off):
        got = [row["candidate_id"] for row in other.ranking["ranking_deterministic"]]
        assert got == expected
        # And a full judged ordering still exists: with no verdicts the BT
        # posterior is exactly the deterministic prior.
        judged = [row["candidate_id"] for row in other.ranking["ranking"]]
        assert judged == expected
        assert other.ranking["judge"]["ran"] is False
        assert other.ranking["judge"]["reason"]


def test_an_all_tie_judge_collapses_onto_the_deterministic_ordering() -> None:
    """The offline stub must not manufacture a ranking.

    Six ties per pair carry no preference, so the posterior stays at the
    deterministic prior (shrunk toward the field mean, order preserved).
    Measured on a 5-candidate field: judged order == deterministic order,
    Spearman 1.0.
    """
    result = run(field_of(5), call_llm=J.offline_caller())
    judged = [row["candidate_id"] for row in result.ranking["ranking"]]
    det = [row["candidate_id"] for row in result.ranking["ranking_deterministic"]]
    assert judged == det
    assert result.ranking["judge"]["tie_rate_overall"] == 1.0
    assert result.ranking["ranking_agreement"][
        "spearman_judged_vs_deterministic"
    ] == 1.0


def test_deterministic_ordering_follows_venue_fit_then_penalty() -> None:
    records = [
        make_record("C-01", venue_fit=1.5, penalty=0.0),
        make_record("C-02", venue_fit=3.0, penalty=0.0),
        make_record("C-03", venue_fit=3.0, penalty=1.5),
    ]
    rows = T.deterministic_ordering(
        [T.Candidate.from_record(r) for r in records]
    )
    assert [row["candidate_id"] for row in rows] == ["C-02", "C-03", "C-01"]
    assert rows[0]["tie_break_trace"][1].startswith("rule 1")


def test_a_spec_the_pipeline_cannot_load_sorts_last_but_is_not_killed() -> None:
    records = [
        make_record("C-01", venue_fit=9.0, spec_loads=False),
        make_record("C-02", venue_fit=0.0, spec_loads=True),
    ]
    rows = T.deterministic_ordering([T.Candidate.from_record(r) for r in records])
    assert [row["candidate_id"] for row in rows] == ["C-02", "C-01"]


def test_shuffled_verdicts_still_produce_a_complete_ordering() -> None:
    """C3's harder half: shuffle the verdicts, keep an ordering."""
    records = field_of(5)
    result = run(
        records,
        call_llm=preference_judge(["C-05", "C-04", "C-03", "C-02", "C-01"]),
        shuffle_replicates=10,
    )
    control = result.ranking["shuffle_control"]
    assert control["ran"] is True
    assert control["replicates"] == 10
    assert control["judged_vs_deterministic_mean_rank_shift"] is not None
    assert control["shuffled_mean_rank_shift_range"] is not None
    assert isinstance(control["judged_displacement_inside_shuffled_range"], bool)

    ids = [c.candidate_id for c in result.candidates]
    prior = {
        c.candidate_id: c.deterministic_score() for c in result.candidates
    }
    shuffled = BT.fit(
        [{"pair": [ids[0], ids[1]], "winner": None}],
        prior_means=prior,
        candidates=ids,
    )
    assert len(BT.strength_order(shuffled)) == len(ids)


def test_ranking_json_is_byte_stable_across_two_runs(tmp_path: Path) -> None:
    records = field_of(4)
    caller = preference_judge(["C-02", "C-01", "C-04", "C-03"])

    first_dir = tmp_path / "run_a"
    second_dir = tmp_path / "run_b"
    run(records, call_llm=caller, seed=42).write(first_dir)
    run(records, call_llm=caller, seed=42).write(second_dir)

    a = (first_dir / "ranking.json").read_bytes()
    b = (second_dir / "ranking.json").read_bytes()
    assert a == b, "ranking.json must not carry a timestamp or a path"
    # matches.jsonl differs only in the per-verdict timestamp, so compare
    # it with that field dropped.
    def _strip(path: Path) -> list[dict]:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for row in rows:
            row.pop("ts", None)
        return rows

    assert _strip(first_dir / "matches.jsonl") == _strip(second_dir / "matches.jsonl")


def test_artifacts_are_written_where_the_spec_says(tmp_path: Path) -> None:
    result = run(field_of(3), call_llm=preference_judge(["C-01", "C-02", "C-03"]))
    paths = result.write(tmp_path)
    assert (tmp_path / "matches.jsonl").exists()
    assert (tmp_path / "ranking.json").exists()
    assert (tmp_path / "tournament.md").exists()
    assert set(paths) >= {"matches", "ranking", "digest"}

    rows = [
        json.loads(line)
        for line in (tmp_path / "matches.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows
    for row in rows:  # spec Appendix B
        assert set(row) >= {
            "pair", "orientation", "sample", "dimension", "winner",
            "evidence", "judge_model", "ts",
        }
        assert row["orientation"] in J.ORIENTATIONS


def test_judged_ordering_can_differ_from_deterministic_and_both_are_kept() -> None:
    """The judge is allowed to move the ranking; the baseline is not lost."""
    records = [
        make_record("C-01", venue_fit=3.0),
        make_record("C-02", venue_fit=0.0),
    ]
    # A judge that unanimously prefers the deterministically weaker idea.
    result = run(records, call_llm=preference_judge(["C-02", "C-01"]), samples=3)
    judged = [row["candidate_id"] for row in result.ranking["ranking"]]
    det = [row["candidate_id"] for row in result.ranking["ranking_deterministic"]]
    assert det == ["C-01", "C-02"]
    assert judged == ["C-02", "C-01"]
    agreement = result.ranking["ranking_agreement"]
    assert agreement["rank1_agrees"] is False
    assert agreement["rank1_judged"] == "C-02"
    assert agreement["rank1_deterministic"] == "C-01"


def test_feasibility_kill_removes_a_candidate_and_records_the_evidence(
    tmp_path: Path,
) -> None:
    killed_record = make_record("C-09", verdict="KILL")
    killed_record["feasibility"]["checks"] = [
        {
            "code": "F-VAR-ABSENT",
            "status": "KILL",
            "message": "variable NOT_A_VAR is not in the registry",
            "evidence": "data_registry/datasets/hsls09_public.yaml var map",
            "penalty": 0.0,
        }
    ]
    result = run(
        field_of(3) + [killed_record],
        call_llm=preference_judge(["C-01", "C-02", "C-03"]),
    )
    entrants = {c.candidate_id for c in result.candidates}
    assert "C-09" not in entrants
    assert result.killed and result.killed[0]["kill_code"] == "F-VAR-ABSENT"
    assert "F-VAR-ABSENT" in result.ranking["killed_this_stage"][0]["evidence"]

    result.write(tmp_path)
    lines = (tmp_path / "killed.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    # Appending the same kill twice must not double-count.
    result.write(tmp_path)
    assert len((tmp_path / "killed.jsonl").read_text(encoding="utf-8").splitlines()) == 1


# --------------------------------------------------------------------------
# 5. Prior art: absent module, and the veto when it is present
# --------------------------------------------------------------------------


def test_prior_art_stage_wires_to_the_real_module_or_says_it_is_absent() -> None:
    """The call site is structured whether or not priorart.py has landed.

    ``anchors=[]`` keeps this offline: without it the real module reads
    the 34-anchor corpus off disk, which is not something a unit test
    should be doing.
    """
    result = run(
        field_of(3),
        prior_art=True,
        prior_art_kwargs={"anchors": []},
        call_llm=preference_judge(["C-01", "C-02", "C-03"]),
    )
    status = result.ranking["prior_art"]
    if status["ran"]:  # src/ideation/priorart.py is present
        assert set(status["verdicts"].values()) <= {
            "CLEAR", "UNVERIFIABLE", "COLLISION"
        }
        # With an empty corpus nothing can be verified, and UNVERIFIABLE
        # must not silently become CLEAR.
        for candidate in result.candidates:
            assert candidate.priorart_verdict in {
                "CLEAR", "UNVERIFIABLE", "COLLISION"
            }
    else:
        assert "SKIPPED" in status["reason"] or "not available" in status["reason"]
        for candidate in result.candidates:
            assert candidate.priorart_verdict == "NOT_RUN"
    assert "veto" in status["note"].lower()


def test_prior_art_module_exposes_the_entry_point_the_cascade_calls() -> None:
    """If priorart.py is on disk, its contract must be the one wired up."""
    module, why = T._load_priorart()
    if module is None:
        pytest.skip(f"src/ideation/priorart.py not available: {why}")
    verdict = module.collision_check(make_card("C-01"), anchors=[])
    assert set(verdict) >= {"verdict", "nearest", "delta_sentence"}
    assert verdict["verdict"] in {"CLEAR", "COLLISION", "UNVERIFIABLE"}


def test_prior_art_collision_kills_with_a_cited_snippet() -> None:
    def checker(card: dict) -> dict:
        if card.get("candidate_id") == "C-02":
            return {
                "verdict": "COLLISION",
                "nearest": [
                    {
                        "paperId": "abc123",
                        "title": "Exactly this study",
                        "year": 2024,
                        "snippet": "we predict the same outcome on the same data",
                    }
                ],
                "delta_sentence": None,
            }
        return {"verdict": "CLEAR", "nearest": [], "delta_sentence": "differs by X"}

    result = run(
        field_of(3),
        prior_art=True,
        prior_art_checker=checker,
        call_llm=preference_judge(["C-01", "C-03"]),
    )
    assert "C-02" not in {c.candidate_id for c in result.candidates}
    kill = next(k for k in result.killed if k["candidate_id"] == "C-02")
    assert kill["kill_code"] == "T-PRIORART-COLLISION"
    assert "we predict the same outcome" in kill["evidence"]


def test_unverifiable_is_a_third_state_and_only_breaks_ties() -> None:
    def checker(card: dict) -> dict:
        verdict = (
            "UNVERIFIABLE" if card.get("candidate_id") == "C-01" else "CLEAR"
        )
        return {"verdict": verdict, "nearest": [], "delta_sentence": None}

    records = [
        make_record("C-01", venue_fit=1.0),
        make_record("C-02", venue_fit=1.0),
    ]
    result = run(records, prior_art=True, prior_art_checker=checker)
    # Neither is killed; the tie between equal deterministic scores is
    # broken in favour of CLEAR (rule 4).
    assert {c.candidate_id for c in result.candidates} == {"C-01", "C-02"}
    det = [row["candidate_id"] for row in result.ranking["ranking_deterministic"]]
    assert det == ["C-02", "C-01"]


def test_prior_art_exception_is_unverifiable_never_clear() -> None:
    def checker(card: dict) -> dict:
        raise RuntimeError("S2 timed out")

    result = run(field_of(2), prior_art=True, prior_art_checker=checker)
    assert {c.priorart_verdict for c in result.candidates} == {"UNVERIFIABLE"}
    assert all("never as CLEAR" in " ".join(c.notes) for c in result.candidates)


# --------------------------------------------------------------------------
# 6. Diversity, truncation, and C1 invariance
# --------------------------------------------------------------------------


def test_field_truncation_is_recorded_not_silent() -> None:
    records = [
        make_record(f"C-{i:02d}", venue_fit=float(20 - i), penalty=float(i) * 0.1)
        for i in range(1, 16)
    ]
    result = run(records, max_survivors=12)
    assert len(result.candidates) == 12
    stage = next(
        s for s in result.ranking["cascade"] if s["stage"] == "field_selection"
    )
    assert len(stage["deferred"]) == 3
    assert all("field truncated" in row["reason"] for row in stage["deferred"])


def test_diversity_ledger_flags_a_collapsed_top_five() -> None:
    records = [
        make_record(f"C-0{i}", venue_fit=float(6 - i), dataset="hsls09_public")
        for i in range(1, 6)
    ]
    for record in records:  # force one outcome family across the field
        record["card"]["resolved_target"] = "X1MTHEFF"
        record["spec"]["outcome_variable"] = "X1MTHEFF"
    result = run(records)
    ledger = result.ranking["diversity_ledger"]
    assert ledger["collapsed_to_one_dataset"] is True
    assert ledger["collapsed_to_one_outcome_family"] is True
    assert "FAILURE LINE" in result.digest()


def test_tie_break_rule_5_prefers_a_new_opportunity_pattern() -> None:
    records = [
        make_record("C-01", venue_fit=1.0, pattern="equity_subgroup_gap"),
        make_record("C-02", venue_fit=1.0, pattern="replication_transfer"),
    ]
    previous = {
        "candidate_id": "C-99",
        "cell": {"dataset": "hsls09_public", "opportunity_pattern": "equity_subgroup_gap"},
        "outcome_family": "OUTCOME_9",
    }
    rows = T.deterministic_ordering(
        [T.Candidate.from_record(r) for r in records], previous_winner=previous
    )
    assert [row["candidate_id"] for row in rows] == ["C-02", "C-01"]


def test_rank1_repeating_the_previous_winner_is_reported() -> None:
    records = [make_record("C-01", venue_fit=5.0)]
    previous = {
        "candidate_id": "C-99",
        "cell": {"dataset": "hsls09_public", "opportunity_pattern": "equity_subgroup_gap"},
        "outcome_family": "OUTCOME_1",
    }
    result = run(records, previous_winner=previous)
    ledger = result.ranking["diversity_ledger"]
    assert ledger["rank1_repeats_previous_dataset_outcome"] is True


def test_ranking_is_invariant_to_an_injected_novelty_score() -> None:
    """C1: a novelty number changes nothing, at any value."""
    baseline = run(
        field_of(4), call_llm=preference_judge(["C-02", "C-01", "C-04", "C-03"])
    )
    reference = json.dumps(baseline.ranking, sort_keys=True)

    for value in (0, 1, 3, 5, 10, 0.43):
        records = field_of(4)
        for record in records:
            record["card"]["novelty_score_self_assessment"] = value
            record["spec"]["novelty_score_self_assessment"] = value
        polluted = run(
            records, call_llm=preference_judge(["C-02", "C-01", "C-04", "C-03"])
        )
        assert json.dumps(polluted.ranking, sort_keys=True) == reference, (
            f"the ranking moved when novelty_score_self_assessment = {value}"
        )


def test_no_module_ever_reads_a_novelty_field() -> None:
    """C1 as a property of the code, not of a run.

    Checked on the AST rather than by grepping text, so that the many
    places these modules EXPLAIN why novelty is absent do not have to be
    written around. The claim under test is narrow and exact: no module
    here ever reads a dict key whose name contains "novelty", by
    ``.get("...")`` or by subscript.
    """
    import ast

    for name in ("judge.py", "bradley_terry.py", "tournament.py"):
        source = (REPO_ROOT / "src" / "ideation" / name).read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr in {"get", "pop"}:
                    for arg in node.args:
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            assert "novelty" not in arg.value.lower(), (
                                f"{name}:{node.lineno} reads a novelty field"
                            )
            if isinstance(node, ast.Subscript):
                key = node.slice
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    assert "novelty" not in key.value.lower(), (
                        f"{name}:{node.lineno} subscripts a novelty field"
                    )
            if isinstance(node, (ast.Name, ast.Attribute)):
                ident = getattr(node, "id", None) or getattr(node, "attr", "")
                assert "novelty" not in str(ident).lower(), (
                    f"{name}:{node.lineno} names a novelty identifier"
                )


# --------------------------------------------------------------------------
# 7. Judge model routing (no provider SDK is constructed)
# --------------------------------------------------------------------------


def test_judge_model_is_read_from_config_never_hardcoded() -> None:
    assert J.resolve_judge_model({}) is None
    assert (
        J.resolve_judge_model({"ideation": {"models": {"judge": "some-flash"}}})
        == "some-flash"
    )
    assert J.judge_samples({}) == 3
    assert J.judge_samples({"ideation": {"tournament": {"judge_samples": 5}}}) == 5
    assert J.judge_temperature({}) is None
    assert (
        J.judge_temperature({"ideation": {"tournament": {"judge_temperature": 0.4}}})
        == 0.4
    )


def test_judge_differing_from_generator_is_reported() -> None:
    same = J.JudgeRun(judge_model="m", generator_model="m").summary()
    assert same["judge_differs_from_generator"] is False
    different = J.JudgeRun(judge_model="flash", generator_model="pro").summary()
    assert different["judge_differs_from_generator"] is True
    unknown = J.JudgeRun(judge_model="flash", generator_model="").summary()
    assert unknown["judge_differs_from_generator"] is None
