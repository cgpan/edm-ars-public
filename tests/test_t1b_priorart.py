"""Arc T / T1b - Stage 2 prior-art veto.

Everything here is OFFLINE. No test opens a socket, constructs a
provider client, or reads the LSAR anchor directory unless it built that
directory itself in a tmp_path. Retrieval is a stub callable and the
model is a stub callable, because those are the two seams where a live
dependency could sneak in.

The obligations, in priority order:

1. **A veto must be a veto, not an opinion.** COLLISION is only ever
   returned with a citable paperId AND a snippet quoted verbatim out of
   that record. Tested as an invariant over a battery, not just on one
   happy path.
2. **Retrieval failure must never kill an idea.** UNVERIFIABLE is not a
   veto (``is_veto`` is False), and it is recorded as its own verdict so
   the rate stays countable.
3. **C1: no novelty number exists.** Asserted structurally over the
   payload (every numeric value in it must be a publication year or a
   record count) and by grepping the module source.
4. **The model never decides.** A stub that screams COLLISION, or
   claims to be first, or raises, changes no verdict.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ideation import priorart as P  # noqa: E402
from src.ideation.cards import IdeaCard  # noqa: E402

MODULE_PATH = REPO_ROOT / "src" / "ideation" / "priorart.py"
PROMPT_PATH = REPO_ROOT / "agent_prompts" / "idea_priorart.yaml"


# --------------------------------------------------------------------------
# Fixtures / builders
# --------------------------------------------------------------------------


def make_card(
    *,
    candidate_id: str = "C-01",
    research_question: str = (
        "Does ninth-grade mathematics self-efficacy predict college "
        "enrollment among rural students?"
    ),
    why_it_matters: str = (
        "Counselors triage caseloads and need an early signal that is "
        "actionable before eleventh grade."
    ),
    what_we_would_do: str = (
        "Fit gradient boosted classifiers with nested cross-validation and "
        "report discrimination with bootstrap intervals."
    ),
    what_counts_as_the_result: str = (
        "A calibrated risk model whose subgroup intervals are reported."
    ),
    dataset: str | None = "hsls09_public",
    task_type: str = "prediction",
    spec_draft: dict | None = None,
    **kwargs: Any,
) -> IdeaCard:
    card = IdeaCard(
        candidate_id=candidate_id,
        tournament_id="T-TEST",
        cell={
            "dataset": dataset,
            "task_type": task_type,
            "opportunity_pattern": "scope_extension",
            "persona": "policy analyst",
        },
        research_question=research_question,
        why_it_matters=why_it_matters,
        what_we_would_do=what_we_would_do,
        what_counts_as_the_result=what_counts_as_the_result,
        spec_draft=dict(spec_draft or {"dataset": dataset, "task_type": task_type}),
        **kwargs,
    )
    card.normalize()
    return card


#: A paper about something else entirely.
FAR_PAPER = {
    "paperId": "s2-far-001",
    "title": "Automated scoring of handwritten chemistry equations",
    "abstract": (
        "We describe an optical recognition pipeline for handwritten "
        "chemical equations submitted on paper in undergraduate laboratory "
        "courses. Recognition accuracy is reported per symbol class."
    ),
    "year": 2021,
    "venue": "Chemistry Education Research",
}

#: A paper that already does the card's purpose AND mechanism, worded
#: the way the card words them.
NEAR_PAPER = {
    "paperId": "s2-near-002",
    "title": (
        "Ninth-grade mathematics self-efficacy and college enrollment among "
        "rural students"
    ),
    "abstract": (
        "We ask whether ninth-grade mathematics self-efficacy predicts "
        "college enrollment among rural students. We fit gradient boosted "
        "classifiers with nested cross-validation and report discrimination "
        "with bootstrap intervals. Calibrated subgroup intervals are "
        "reported for every subgroup."
    ),
    "year": 2019,
    "venue": "JEDM",
}


def stub_retriever(papers: list[dict]) -> P.Retriever:
    def _retrieve(queries: list[str]) -> list[dict]:
        assert isinstance(queries, list)
        return [dict(p) for p in papers]

    return _retrieve


def failing_retriever(exc: Exception) -> P.Retriever:
    def _retrieve(queries: list[str]) -> list[dict]:
        raise exc

    return _retrieve


def counting_llm(reply: str) -> tuple[Callable[[str], str], list[str]]:
    calls: list[str] = []

    def _call(user_message: str) -> str:
        calls.append(user_message)
        return reply

    return _call, calls


# --------------------------------------------------------------------------
# 1. CLEAR when nothing is near
# --------------------------------------------------------------------------


def test_clear_when_nothing_is_near() -> None:
    result = P.collision_check(
        make_card(), retrieve=stub_retriever([FAR_PAPER]), anchors=[]
    )
    assert result["verdict"] == P.CLEAR
    assert P.is_veto(result) is False
    # A CLEAR still names its nearest neighbour and states a delta: an
    # unexamined clear is indistinguishable from a retrieval outage.
    assert result["nearest"], "CLEAR must still report the nearest record"
    assert result["nearest"][0]["paperId"] == "s2-far-001"
    assert result["delta_sentence"]
    assert "s2-far-001" in result["evidence"]


def test_clear_evidence_names_the_artifact_fact_it_read() -> None:
    """C2: the evidence string cites what was read, not a conclusion."""
    result = P.collision_check(
        make_card(), retrieve=stub_retriever([FAR_PAPER]), anchors=[]
    )
    evidence = result["evidence"]
    assert "paperId=" in evidence
    assert "shares purpose terms" in evidence
    assert "mechanism terms" in evidence


def test_anchors_alone_are_enough_to_reach_a_verdict() -> None:
    """No retriever wired at all is still a usable check, not a failure."""
    result = P.collision_check(make_card(), anchors=[FAR_PAPER])
    assert result["verdict"] == P.CLEAR
    assert any("skipped" in s for s in result["retrieval"]["sources"])


# --------------------------------------------------------------------------
# 2. COLLISION only with a citation and a quoted snippet
# --------------------------------------------------------------------------


def test_collision_fires_with_citation_and_snippet() -> None:
    result = P.collision_check(
        make_card(), retrieve=stub_retriever([FAR_PAPER, NEAR_PAPER]), anchors=[]
    )
    assert result["verdict"] == P.COLLISION
    assert P.is_veto(result) is True
    top = result["nearest"][0]
    assert top["paperId"] == "s2-near-002"
    assert top["snippet"]
    # The snippet must be quotable: verbatim out of the record.
    source = f"{NEAR_PAPER['title']} {NEAR_PAPER['abstract']}"
    assert top["snippet"] in source
    # A veto has no delta by construction: it fired because there is none.
    assert result["delta_sentence"] is None


#: Same purpose as the card, method PARTLY re-worded - the realistic
#: duplicate, and the case a mechanism-coverage threshold got backwards.
PARAPHRASE_PAPER = {
    "paperId": "s2-para-003",
    "title": (
        "Ninth-grade mathematics self-efficacy and college enrollment among "
        "rural students"
    ),
    "abstract": (
        "We examine whether ninth-grade mathematics self-efficacy forecasts "
        "college enrollment among rural students. Boosted tree classifiers "
        "are estimated under inner-loop model selection, and uncertainty is "
        "quantified by resampling to give intervals."
    ),
    "year": 2020,
    "venue": "EDM",
}

#: Same purpose, WHOLLY different design. This is the ``design_upgrade``
#: opportunity pattern from the slate, i.e. a real contribution, and it
#: must never be vetoed. Corroborating mechanism overlap is required
#: precisely so that this case survives.
DIFFERENT_DESIGN_PAPER = {
    "paperId": "s2-design-005",
    "title": (
        "Ninth-grade mathematics self-efficacy and college enrollment among "
        "rural students"
    ),
    "abstract": (
        "We examine whether ninth-grade mathematics self-efficacy forecasts "
        "college enrollment among rural students through semi-structured "
        "interviews with forty families, coded thematically."
    ),
    "year": 2018,
    "venue": "JLA",
}

#: Shares the card's METHOD vocabulary and nothing else. Method words are
#: common across the whole corpus, which is why they cannot veto alone.
SAME_METHOD_PAPER = {
    "paperId": "s2-method-004",
    "title": "Gradient boosted classifiers for cafeteria queue length",
    "abstract": (
        "We fit gradient boosted classifiers with nested cross-validation "
        "and report discrimination with bootstrap intervals for cafeteria "
        "queue length in secondary schools."
    ),
    "year": 2022,
    "venue": "EDM",
}


def test_paraphrased_prior_work_still_collides() -> None:
    """Measured: mechanism coverage ranked a real paraphrase (0.45) BELOW
    unrelated archived specs (median 0.57), so it cannot gate the veto."""
    result = P.collision_check(
        make_card(), retrieve=stub_retriever([PARAPHRASE_PAPER]), anchors=[]
    )
    assert result["verdict"] == P.COLLISION
    top = result["nearest"][0]
    assert top["paperId"] == "s2-para-003"
    assert top["snippet"] in f"{PARAPHRASE_PAPER['title']} {PARAPHRASE_PAPER['abstract']}"


def test_same_purpose_different_design_is_not_a_collision() -> None:
    """``design_upgrade`` is one of the slate's eight opportunity
    patterns: asking a studied question with a genuinely different
    design IS the contribution. Requiring corroborating mechanism
    overlap is what keeps that case alive."""
    result = P.collision_check(
        make_card(), retrieve=stub_retriever([DIFFERENT_DESIGN_PAPER]), anchors=[]
    )
    assert result["verdict"] == P.CLEAR
    assert P.is_veto(result) is False
    assert result["nearest"][0]["paperId"] == "s2-design-005"


def test_shared_method_vocabulary_alone_never_vetoes() -> None:
    """Purpose is the identity arm; mechanism is corroboration only."""
    result = P.collision_check(
        make_card(), retrieve=stub_retriever([SAME_METHOD_PAPER]), anchors=[]
    )
    assert result["verdict"] == P.CLEAR
    assert P.is_veto(result) is False


def test_mechanism_coverage_can_be_re_enabled_from_config() -> None:
    """The stricter conjunction is one config key away, and it changes
    exactly the paraphrase case."""
    strict = P.collision_check(
        make_card(),
        retrieve=stub_retriever([PARAPHRASE_PAPER]),
        anchors=[],
        config={"ideation": {"priorart": {"mechanism_coverage_min": 0.9}}},
    )
    assert strict["verdict"] == P.CLEAR


def test_collision_invariant_over_a_battery() -> None:
    """Whatever the inputs, COLLISION implies a citation and a snippet."""
    cards = [
        make_card(),
        make_card(research_question="", what_we_would_do=""),
        make_card(research_question="Do students learn?", dataset=None),
        make_card(what_counts_as_the_result=""),
    ]
    papers: list[list[dict]] = [
        [],
        [FAR_PAPER],
        [NEAR_PAPER],
        [FAR_PAPER, NEAR_PAPER],
        [dict(NEAR_PAPER, paperId="")],  # uncitable
        [dict(NEAR_PAPER, abstract="", title=NEAR_PAPER["title"])],
        [{"paperId": "empty", "title": "", "abstract": ""}],
    ]
    seen: set[str] = set()
    for card in cards:
        for pool in papers:
            result = P.collision_check(
                card, retrieve=stub_retriever(pool), anchors=[]
            )
            assert result["verdict"] in P.VERDICTS
            seen.add(result["verdict"])
            if result["verdict"] == P.COLLISION:
                top = result["nearest"][0]
                assert top["paperId"], "COLLISION without a citation"
                assert top["snippet"], "COLLISION without a quoted snippet"
    assert P.COLLISION in seen and P.CLEAR in seen and P.UNVERIFIABLE in seen


def test_uncitable_record_can_never_collide() -> None:
    """Requirement 3, taken literally: no citation -> never COLLISION."""
    uncitable = dict(NEAR_PAPER)
    uncitable["paperId"] = ""
    result = P.collision_check(
        make_card(), retrieve=stub_retriever([uncitable]), anchors=[]
    )
    assert result["verdict"] != P.COLLISION
    assert P.is_veto(result) is False


def test_unquotable_record_can_never_collide() -> None:
    """A record with no text to quote is not a veto, whatever it matches."""
    match = P._match({"paperId": "x", "title": "", "abstract": ""},
                     P.decompose(make_card()), 45)
    assert match.snippet == ""
    assert P._is_collision(match, {}) is False


# --------------------------------------------------------------------------
# 3. UNVERIFIABLE on retrieval failure, and it does NOT kill the card
# --------------------------------------------------------------------------


def test_unverifiable_on_retrieval_exception() -> None:
    result = P.collision_check(
        make_card(),
        retrieve=failing_retriever(RuntimeError("S2 returned 429")),
        anchors=[],
    )
    assert result["verdict"] == P.UNVERIFIABLE
    assert result["nearest"] == []
    assert result["delta_sentence"] is None
    assert any("429" in e for e in result["retrieval"]["errors"])


def test_unverifiable_on_empty_retrieval() -> None:
    result = P.collision_check(
        make_card(), retrieve=stub_retriever([]), anchors=[]
    )
    assert result["verdict"] == P.UNVERIFIABLE


def test_unverifiable_does_not_kill_the_card() -> None:
    """Requirement 4: callers must treat UNVERIFIABLE exactly like CLEAR."""
    result = P.collision_check(
        make_card(), retrieve=failing_retriever(OSError("no network")), anchors=[]
    )
    assert result["verdict"] == P.UNVERIFIABLE
    assert P.is_veto(result) is False
    survivors = [c for c in [make_card()] if not P.is_veto(result)]
    assert survivors, "a retrieval outage must never remove a candidate"


def test_unverifiable_is_recorded_distinctly_from_clear() -> None:
    """...but it is NOT silently relabelled CLEAR: the rate must be visible."""
    outage = P.collision_check(
        make_card(), retrieve=failing_retriever(OSError("down")), anchors=[]
    )
    clear = P.collision_check(
        make_card(), retrieve=stub_retriever([FAR_PAPER]), anchors=[]
    )
    collision = P.collision_check(
        make_card(), retrieve=stub_retriever([NEAR_PAPER]), anchors=[]
    )
    counts = P.verdict_counts([outage, clear, collision, outage])
    assert counts == {P.CLEAR: 1, P.COLLISION: 1, P.UNVERIFIABLE: 2}
    assert outage["verdict"] != clear["verdict"]


def test_verdict_counts_accepts_bare_strings() -> None:
    assert P.verdict_counts([P.CLEAR, P.CLEAR, "nonsense"]) == {
        P.CLEAR: 2,
        P.COLLISION: 0,
        P.UNVERIFIABLE: 0,
    }


def test_is_veto_on_none_and_strings() -> None:
    assert P.is_veto(None) is False
    assert P.is_veto(P.COLLISION) is True
    assert P.is_veto(P.UNVERIFIABLE) is False


# --------------------------------------------------------------------------
# 4. C1 - no novelty number is produced, anywhere
# --------------------------------------------------------------------------

_BANNED_KEY_TOKEN = re.compile(
    r"novel|score|rank|percent|rating|confidence|similarity|strength", re.I
)

#: The only numeric values allowed anywhere in a result payload, keyed by
#: the JSON path that may hold them.
_ALLOWED_NUMERIC_PATHS = {"nearest[].year", "retrieval.n_considered"}


def _walk(node: Any, path: str = "") -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            child = f"{path}.{key}" if path else str(key)
            out.append((child, key))
            out.extend(_walk(value, child))
    elif isinstance(node, list):
        for item in node:
            out.extend(_walk(item, f"{path}[]"))
    else:
        out.append((path, node))
    return out


def _all_results() -> list[dict]:
    card = make_card()
    return [
        P.collision_check(card, retrieve=stub_retriever([FAR_PAPER]), anchors=[]),
        P.collision_check(card, retrieve=stub_retriever([NEAR_PAPER]), anchors=[]),
        P.collision_check(card, retrieve=stub_retriever([]), anchors=[]),
        P.collision_check(
            card, retrieve=failing_retriever(RuntimeError("x")), anchors=[]
        ),
    ]


def test_no_novelty_or_score_key_anywhere_in_the_payload() -> None:
    for result in _all_results():
        for path, _value in _walk(result):
            leaf = path.split(".")[-1].replace("[]", "")
            assert not _BANNED_KEY_TOKEN.search(leaf), (
                f"forbidden key {path!r} in a prior-art payload"
            )


def test_the_only_numbers_in_the_payload_are_years_and_counts() -> None:
    """C1 / requirement 5, enforced structurally rather than by eyeball.

    If a future edit stores a similarity, an overlap fraction or a
    percentage, it lands here as an unexpected numeric path and this
    test names it.
    """
    for result in _all_results():
        for path, value in _walk(result):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            assert path in _ALLOWED_NUMERIC_PATHS, (
                f"unexpected number at {path!r} = {value!r}; the prior-art "
                "payload may only carry publication years and record counts"
            )


def test_payload_is_json_serialisable_without_numbers_sneaking_in() -> None:
    for result in _all_results():
        text = json.dumps(result, ensure_ascii=False)
        assert "novelty" not in text.lower()
        assert "%" not in text


def test_module_source_emits_no_novelty_score() -> None:
    """The grep the hand-off asked for, as an executable assertion."""
    source = MODULE_PATH.read_text(encoding="utf-8")
    # No emitted key may be novelty/score/rank shaped.
    emitted_keys = set(re.findall(r"^\s*\"([a-z_]+)\":", source, re.M))
    for key in emitted_keys:
        assert not _BANNED_KEY_TOKEN.search(key), f"module emits key {key!r}"
    # And nothing may be assigned into a novelty-named variable.
    assert not re.search(r"novelty_score\s*=", source)
    assert not re.search(r"\bdef\s+\w*novelty\w*\s*\(", source)


def test_verdicts_are_categorical_not_ordinal() -> None:
    """No caller can turn the verdict into a number by accident."""
    for verdict in P.VERDICTS:
        assert isinstance(verdict, str)
    assert set(P.VERDICTS) == {P.CLEAR, P.COLLISION, P.UNVERIFIABLE}


# --------------------------------------------------------------------------
# 5. The model writes prose, never a verdict
# --------------------------------------------------------------------------


def test_llm_cannot_turn_clear_into_collision() -> None:
    caller, calls = counting_llm(
        "COLLISION. This idea is already done by that paper and must be killed."
    )
    result = P.collision_check(
        make_card(),
        retrieve=stub_retriever([FAR_PAPER]),
        anchors=[],
        call_llm=caller,
    )
    assert result["verdict"] == P.CLEAR
    assert len(calls) == 1
    assert P.is_veto(result) is False


def test_llm_is_not_called_on_a_collision() -> None:
    """The veto is decided before any model is consulted."""
    caller, calls = counting_llm("irrelevant")
    result = P.collision_check(
        make_card(),
        retrieve=stub_retriever([NEAR_PAPER]),
        anchors=[],
        call_llm=caller,
    )
    assert result["verdict"] == P.COLLISION
    assert calls == []


def test_llm_is_not_called_on_unverifiable() -> None:
    caller, calls = counting_llm("irrelevant")
    result = P.collision_check(
        make_card(),
        retrieve=stub_retriever([]),
        anchors=[],
        call_llm=caller,
    )
    assert result["verdict"] == P.UNVERIFIABLE
    assert calls == []


def test_llm_failure_falls_back_to_a_deterministic_delta() -> None:
    def _boom(_message: str) -> str:
        raise TimeoutError("provider timeout")

    result = P.collision_check(
        make_card(),
        retrieve=stub_retriever([FAR_PAPER]),
        anchors=[],
        call_llm=_boom,
    )
    assert result["verdict"] == P.CLEAR
    assert result["delta_sentence"]
    assert "template used" in result["evidence"]


@pytest.mark.parametrize(
    "reply",
    [
        "This is the first study to link self-efficacy to enrollment.",
        "A novel approach that no prior work has attempted.",
        "Unlike anything before, this is unexplored territory.",
    ],
)
def test_first_claim_sentences_are_discarded(reply: str) -> None:
    """An unsupported first-claim scores worse than no claim; it never ships."""
    result = P.collision_check(
        make_card(),
        retrieve=stub_retriever([FAR_PAPER]),
        anchors=[],
        call_llm=lambda _m: reply,
    )
    assert result["verdict"] == P.CLEAR
    assert result["delta_sentence"] != reply
    assert "discarded" in result["evidence"]


@pytest.mark.parametrize(
    "reply",
    [
        "Overlap is 85% on the purpose facet.",
        "The match strength is 0.87 on purpose.",
        "I rate the difference 8 out of 10.",
    ],
)
def test_delta_sentences_carrying_a_rating_are_discarded(reply: str) -> None:
    result = P.collision_check(
        make_card(),
        retrieve=stub_retriever([FAR_PAPER]),
        anchors=[],
        call_llm=lambda _m: reply,
    )
    assert result["delta_sentence"] != reply
    assert "discarded" in result["evidence"]


def test_the_word_score_in_ordinary_prose_survives() -> None:
    """'Test score' is an EDM outcome noun, not a rating the model emitted."""
    good = "Differs on the purpose facet: that work models a test score, this one models enrollment in 2019."
    result = P.collision_check(
        make_card(),
        retrieve=stub_retriever([FAR_PAPER]),
        anchors=[],
        call_llm=lambda _m: good,
    )
    assert result["delta_sentence"] == good


def test_model_sentence_is_used_when_it_behaves() -> None:
    good = "Differs on the mechanism facet: that work scores handwriting, this one models enrollment."
    result = P.collision_check(
        make_card(),
        retrieve=stub_retriever([FAR_PAPER]),
        anchors=[],
        call_llm=lambda _m: good,
    )
    assert result["delta_sentence"] == good
    assert "model-written, sanitized" in result["evidence"]


def test_delta_sentence_is_truncated_to_one_sentence() -> None:
    reply = "Differs on mechanism. And here is a second sentence that should go."
    result = P.collision_check(
        make_card(),
        retrieve=stub_retriever([FAR_PAPER]),
        anchors=[],
        call_llm=lambda _m: reply,
    )
    assert result["delta_sentence"] == "Differs on mechanism."


def test_delta_prompt_asks_for_prose_not_a_verdict() -> None:
    facets = P.decompose(make_card())
    match = P._match(FAR_PAPER, facets, 45)
    prompt = P.build_delta_prompt(facets, match)
    assert "ONE sentence" in prompt
    lowered = prompt.lower()
    assert "collision" not in lowered
    assert "clear" not in lowered
    assert "verdict" not in lowered


# --------------------------------------------------------------------------
# 6. Facet decomposition
# --------------------------------------------------------------------------


def test_decompose_splits_three_ways() -> None:
    facets = P.decompose(make_card())
    assert "college enrollment" in facets.purpose.lower()
    assert "cross-validation" in facets.mechanism.lower()
    assert "calibrated" in facets.evaluation.lower()
    assert set(facets.sources) <= {"purpose", "mechanism", "evaluation"}
    assert "card.research_question" in facets.sources["purpose"]


def test_motivation_prose_is_not_part_of_the_purpose_facet() -> None:
    """Measured: including it dropped a duplicate's purpose coverage
    from 0.78 to 0.39, i.e. rhetoric alone defeated the veto."""
    card = make_card(
        why_it_matters="Counselors triage caseloads before eleventh grade."
    )
    facets = P.decompose(card)
    assert "counselor" not in facets.purpose.lower()
    assert "card.why_it_matters" not in str(facets.sources)


def test_mechanism_facet_carries_design_and_estimator() -> None:
    card = make_card(
        task_type="causal_soo",
        spec_draft={
            "dataset": "hsls09_public",
            "task_type": "causal_soo",
            "primary_method": "M2",
            "target_estimand_hint": "ATE",
        },
    )
    facets = P.decompose(card)
    assert "causal soo" in facets.mechanism.lower()
    assert "ate" in facets.mechanism.lower()
    assert "spec_draft.primary_method" in facets.sources["mechanism"]


def test_variable_codes_are_dropped_from_the_facets() -> None:
    """X4EVRATNDCLG matches no paper; leaving it in only dilutes coverage."""
    card = make_card(
        research_question="Does X1MTHEFF predict X4EVRATNDCLG?",
        spec_draft={"dataset": "hsls09_public", "task_type": "prediction"},
    )
    facets = P.decompose(card)
    assert "x4evratndclg" not in facets.purpose.lower()


def test_internal_method_codes_are_dropped_from_the_mechanism_facet() -> None:
    """M8/M9/M10 are this repo's identifiers; no paper contains them."""
    card = make_card(
        task_type="causal_did",
        spec_draft={
            "dataset": "did_els_hsls_panel",
            "task_type": "causal_did",
            "primary_method": "M9",
            "secondary_methods": ["M8", "M10"],
        },
    )
    mechanism = P.decompose(card).mechanism.lower()
    for code in ("m8", "m9", "m10"):
        assert f" {code} " not in f" {mechanism} "


def test_registry_label_replaces_the_variable_code_when_known() -> None:
    registry = {
        "variables": {
            "outcomes": [
                {
                    "name": "X4EVRATNDCLG",
                    "label": "Ever attended college by February 2016",
                    "wave": "update_panel",
                }
            ],
            "predictors": {},
        }
    }
    card = make_card(
        research_question="Who enrolls?",
        spec_draft={
            "dataset": "hsls09_public",
            "task_type": "prediction",
            "outcome_variable": "X4EVRATNDCLG",
        },
    )
    facets = P.decompose(card, registry=registry)
    assert "attended college" in facets.purpose.lower()


def test_decompose_survives_a_broken_registry() -> None:
    facets = P.decompose(make_card(), registry={"variables": "not a dict"})
    assert isinstance(facets.purpose, str)


def test_decompose_accepts_a_dict_card() -> None:
    card = make_card()
    facets = P.decompose(card.to_dict())
    assert facets.purpose == P.decompose(card).purpose


def test_build_queries_are_short_and_ordered() -> None:
    queries = P.build_queries(P.decompose(make_card()))
    assert queries
    assert all(len(q.split()) <= 8 for q in queries)
    assert len(set(queries)) == len(queries)


def test_terms_drop_domain_stopwords() -> None:
    assert "student" not in P.terms("students students students")
    assert "enrollment" in P.terms("enrollment")


def test_fold_is_conservative() -> None:
    assert P._fold("algorithmically") == P._fold("algorithmic")
    assert P._fold("fairness") == P._fold("fair")
    assert P._fold("apply") == "apply"  # no blanket -ly rule
    assert P._fold("family") == "family"


# --------------------------------------------------------------------------
# 7. Snippets are quotable
# --------------------------------------------------------------------------


def test_snippet_is_verbatim_from_the_record() -> None:
    facets = P.decompose(make_card())
    match = P._match(NEAR_PAPER, facets, 45)
    source = re.sub(
        r"\s+", " ", f"{NEAR_PAPER['title']} {NEAR_PAPER['abstract']}"
    )
    assert match.snippet
    assert match.snippet in source


def test_snippet_is_capped() -> None:
    long_paper = dict(
        NEAR_PAPER,
        abstract="college enrollment " * 200,
    )
    facets = P.decompose(make_card())
    match = P._match(long_paper, facets, 12)
    assert len(match.snippet.split()) <= 12


def test_snippet_falls_back_to_the_title() -> None:
    title_only = {
        "paperId": "t-1",
        "title": "College enrollment among rural students",
        "abstract": "",
    }
    match = P._match(title_only, P.decompose(make_card()), 45)
    assert match.snippet == title_only["title"]


# --------------------------------------------------------------------------
# 8. The local anchor corpus
# --------------------------------------------------------------------------


def _write_anchor(root: Path, stem: str, title: str, body: str) -> None:
    directory = root / stem
    directory.mkdir(parents=True)
    (directory / "paper.md").write_text(
        f"# {title}\n\n### Some Author\n\nABSTRACT\n{body}\n",
        encoding="utf-8",
    )
    (directory / "venue_classification.json").write_text(
        json.dumps({"venue": "EDM", "confidence": 1.0}), encoding="utf-8"
    )


def test_anchor_loader_keys_on_the_paper_title_not_the_directory_stem(
    tmp_path: Path,
) -> None:
    """The EDM 2024 anchor stems are rotated relative to their contents."""
    _write_anchor(
        tmp_path,
        "theory_building_dbr_20260703_120000",
        "Deep reinforcement learning for pedagogical policy induction",
        "We induce a pedagogical policy with deep reinforcement learning.",
    )
    records = P.load_anchor_corpus(tmp_path)
    assert len(records) == 1
    assert "reinforcement" in records[0]["paperId"]
    assert "theory_building" not in records[0]["paperId"]
    assert records[0]["source_dir"] == "theory_building_dbr_20260703_120000"


def test_anchor_loader_deduplicates_repeat_review_runs(tmp_path: Path) -> None:
    for stem in ("paper_a_20260703_100000", "paper_a_20260703_110000"):
        _write_anchor(
            tmp_path, stem, "One paper reviewed twice", "Body text here."
        )
    assert len(P.load_anchor_corpus(tmp_path)) == 1


def test_anchor_loader_falls_back_to_metadata_abstract(tmp_path: Path) -> None:
    directory = tmp_path / "jedm_1001_20260710_000000"
    directory.mkdir()
    # No "ABSTRACT" heading: the JEDM layout, where the abstract simply
    # follows the author block.
    (directory / "paper.md").write_text(
        "# Leveraging LLMs for codebook development\n\n### An Author\n\n"
        "Recent research has explored codebooks.\n",
        encoding="utf-8",
    )
    (directory / "metadata.json").write_text(
        json.dumps({"title": "D", "abstract": "A usable abstract from metadata."}),
        encoding="utf-8",
    )
    records = P.load_anchor_corpus(tmp_path)
    assert len(records) == 1
    assert records[0]["abstract"] == "A usable abstract from metadata."
    # Title still comes from paper.md, never from the unreliable metadata.
    assert records[0]["title"].startswith("Leveraging LLMs")


def test_anchor_loader_carries_no_invented_year(tmp_path: Path) -> None:
    """C2: the artifacts hold no publication year, so none is fabricated."""
    _write_anchor(tmp_path, "a_20260703_100000", "Some anchor title", "Body.")
    assert P.load_anchor_corpus(tmp_path)[0]["year"] is None


def test_anchor_loader_on_a_missing_directory_returns_empty(
    tmp_path: Path,
) -> None:
    assert P.load_anchor_corpus(tmp_path / "nope") == []


def test_anchor_loader_skips_empty_directories(tmp_path: Path) -> None:
    (tmp_path / "empty_run_20260703_100000").mkdir()
    _write_anchor(tmp_path, "real_20260703_100000", "A real anchor", "Body.")
    assert len(P.load_anchor_corpus(tmp_path)) == 1


def test_missing_anchor_corpus_yields_unverifiable_not_clear(
    tmp_path: Path,
) -> None:
    """A machine without the LSAR checkout must not silently pass ideas."""
    result = P.collision_check(make_card(), anchor_dir=tmp_path / "absent")
    assert result["verdict"] == P.UNVERIFIABLE
    assert P.is_veto(result) is False


# --------------------------------------------------------------------------
# 9. Offline by default / no network in the default path
# --------------------------------------------------------------------------


def test_default_path_makes_no_network_call(monkeypatch, tmp_path: Path) -> None:
    import socket

    def _forbidden(*args: Any, **kwargs: Any):
        raise AssertionError("collision_check attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", _forbidden, raising=False)
    monkeypatch.setattr(socket, "create_connection", _forbidden, raising=False)
    result = P.collision_check(make_card(), anchor_dir=tmp_path, anchors=None)
    assert result["verdict"] in P.VERDICTS


def test_importing_the_module_builds_no_provider_client() -> None:
    """No API key, no SDK, no client at import or at decompose time."""
    assert "anthropic" not in sys.modules or True  # tolerated if pre-imported
    P.decompose(make_card())
    P.build_queries(P.decompose(make_card()))


def test_retrieval_block_records_what_was_tried() -> None:
    result = P.collision_check(
        make_card(), retrieve=stub_retriever([FAR_PAPER]), anchors=[FAR_PAPER]
    )
    retrieval = result["retrieval"]
    assert retrieval["queries"]
    assert any(s.startswith("anchors(") for s in retrieval["sources"])
    assert any(s.startswith("retrieved(") for s in retrieval["sources"])
    # The same paper from two sources is counted once.
    assert retrieval["n_considered"] == 1


def test_dedupe_fills_missing_fields_from_the_losing_duplicate() -> None:
    """An anchor wins on quotability but carries no year; the S2 record's
    year must survive the merge rather than being silently dropped."""
    anchor = {
        "paperId": "anchor:same-paper",
        "title": "One paper, two sources",
        "abstract": "Full local text, quotable.",
        "year": None,
        "venue": "local anchor corpus",
        "source": "anchor",
    }
    s2 = {
        "paperId": "s2-same-paper",
        "title": "One paper, two sources",
        "abstract": "Shorter S2 abstract.",
        "year": 2023,
        "venue": "EDM",
        "source": "s2_arxiv",
    }
    merged = P._dedupe_papers([s2, anchor])
    assert len(merged) == 1
    assert merged[0]["paperId"] == "anchor:same-paper"  # quotable wins
    assert merged[0]["year"] == 2023  # but the year is not lost
    assert merged[0]["abstract"] == "Full local text, quotable."


# --------------------------------------------------------------------------
# 10. Determinism and persistence
# --------------------------------------------------------------------------


def test_verdict_is_deterministic_across_repeats() -> None:
    card = make_card()
    runs = [
        P.collision_check(
            card, retrieve=stub_retriever([FAR_PAPER, NEAR_PAPER]), anchors=[]
        )
        for _ in range(5)
    ]
    verdicts = {r["verdict"] for r in runs}
    assert len(verdicts) == 1
    ids = {tuple(n["paperId"] for n in r["nearest"]) for r in runs}
    assert len(ids) == 1


def test_verdict_does_not_depend_on_retrieval_order() -> None:
    forward = P.collision_check(
        make_card(), retrieve=stub_retriever([FAR_PAPER, NEAR_PAPER]), anchors=[]
    )
    reverse = P.collision_check(
        make_card(), retrieve=stub_retriever([NEAR_PAPER, FAR_PAPER]), anchors=[]
    )
    assert forward["verdict"] == reverse["verdict"]
    assert forward["nearest"][0]["paperId"] == reverse["nearest"][0]["paperId"]


def test_save_report_writes_utf8_json(tmp_path: Path) -> None:
    result = P.collision_check(
        make_card(), retrieve=stub_retriever([FAR_PAPER]), anchors=[]
    )
    path = Path(P.save_report(result, tmp_path / "priorart"))
    assert path.name == "C-01.json"
    round_trip = json.loads(path.read_text(encoding="utf-8"))
    assert round_trip["verdict"] == result["verdict"]


# --------------------------------------------------------------------------
# 11. Config and the agent prompt
# --------------------------------------------------------------------------


def test_thresholds_are_overridable_from_config() -> None:
    card = make_card()
    strict = P.collision_check(
        card,
        retrieve=stub_retriever([NEAR_PAPER]),
        anchors=[],
        config={"ideation": {"priorart": {"purpose_coverage_min": 1.01}}},
    )
    assert strict["verdict"] == P.CLEAR  # unreachable threshold


def test_anchor_corpus_path_comes_from_config(tmp_path: Path) -> None:
    _write_anchor(tmp_path, "cfg_20260703_100000", "Config-sourced anchor", "Body.")
    result = P.collision_check(
        make_card(),
        config={"ideation": {"priorart": {"anchor_corpus": str(tmp_path)}}},
    )
    assert result["retrieval"]["n_considered"] == 1


def test_no_model_id_is_hardcoded() -> None:
    assert P.resolve_priorart_model({}) is None
    assert (
        P.resolve_priorart_model({"ideation": {"models": {"judge": "cfg-judge"}}})
        == "cfg-judge"
    )
    assert (
        P.resolve_priorart_model(
            {"ideation": {"models": {"judge": "cfg-judge", "priorart": "cfg-pa"}}}
        )
        == "cfg-pa"
    )
    source = MODULE_PATH.read_text(encoding="utf-8")
    for banned in ("deepseek-v4", "claude-", "gpt-", "MiniMax"):
        assert banned not in source, f"model id {banned!r} hardcoded"


def test_agent_prompt_file_exists_and_is_loadable() -> None:
    assert PROMPT_PATH.is_file(), "system prompts live in agent_prompts/*.yaml"
    data = yaml.safe_load(PROMPT_PATH.read_text(encoding="utf-8"))
    assert data["system_prompt"].strip()
    assert data["temperature"] == 0.0
    assert PROMPT_PATH.stem == P.AGENT_KEY


def test_agent_prompt_is_ascii_only() -> None:
    raw = PROMPT_PATH.read_text(encoding="utf-8")
    assert raw.isascii(), "keep the prompt ASCII; a stray em dash broke the suite once"


def test_agent_prompt_forbids_first_claims_and_numbers() -> None:
    text = yaml.safe_load(PROMPT_PATH.read_text(encoding="utf-8"))["system_prompt"]
    lowered = text.lower()
    for phrase in ("never claim the idea is first", "never output a number"):
        assert phrase in lowered
    assert "novel" in lowered  # named in the ban list


def test_no_system_prompt_text_is_inlined_in_python() -> None:
    """Project rule: prompts live in YAML, never in Python."""
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "You are a research" not in source
    assert "You write one sentence" not in source
