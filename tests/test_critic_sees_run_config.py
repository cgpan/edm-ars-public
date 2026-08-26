"""A conditional rule needs the fact its condition turns on.

The prediction checklist says: "At least 5 individual model families +
StackingEnsemble ... (When `mlp_enabled: false`, 4 individual + Stacking
= 5 total is acceptable.)"

The shipped default IS false, so four individual models is the correct
outcome. But in a real run's rendered Critic prompt the ONLY occurrence
of `mlp_enabled` was inside that rule's own text -- the Critic was handed
a condition and never told whether it held. It could not apply the
carve-out, flagged an_01 major, and drove a revision cycle over a run
that had done nothing wrong. Every prediction run on the default config
earned the same unavoidable finding.

Same shape as an instruction that reaches one agent and no other: the
rule existed, the fact needed to evaluate it did not arrive.
"""

from __future__ import annotations

import json

import pytest

from src.agents.critic import Critic


def _critic(config: dict | None) -> Critic:
    agent = object.__new__(Critic)
    if config is not None:
        agent.config = config
    return agent


FULL = {
    "pipeline": {"mlp_enabled": False, "random_state": 42, "task_type": "prediction"},
    "class_imbalance": {"minority_threshold": 0.2, "ablation_enabled": True},
}


def test_the_flag_the_carve_out_depends_on_is_reported() -> None:
    assert _critic(FULL)._checklist_relevant_config()["pipeline.mlp_enabled"] is False


def test_false_is_reported_rather_than_omitted() -> None:
    """A falsy value is the whole point here; it must not be dropped."""
    relevant = _critic(FULL)._checklist_relevant_config()
    assert "pipeline.mlp_enabled" in relevant


@pytest.mark.parametrize(
    "key",
    [
        "pipeline.mlp_enabled",
        "class_imbalance.minority_threshold",
        "class_imbalance.ablation_enabled",
    ],
)
def test_every_config_conditional_rule_gets_its_fact(key: str) -> None:
    assert key in _critic(FULL)._checklist_relevant_config()


def test_a_missing_section_is_skipped_not_guessed() -> None:
    relevant = _critic({"pipeline": {"mlp_enabled": True}})._checklist_relevant_config()
    assert relevant == {"pipeline.mlp_enabled": True}


def test_no_config_yields_an_empty_block() -> None:
    """Never fabricate settings the run did not declare."""
    assert _critic(None)._checklist_relevant_config() == {}


def test_a_non_dict_section_does_not_raise() -> None:
    assert _critic({"pipeline": "nonsense"})._checklist_relevant_config() == {}


def test_the_block_is_json_serialisable() -> None:
    """It is rendered into the prompt as JSON."""
    json.dumps(_critic(FULL)._checklist_relevant_config())


def test_the_keys_the_checklist_references_are_all_covered() -> None:
    """Guards against a rule gaining a condition with no fact behind it."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    checklist = root / "skills" / "task-type" / "prediction-critic-checklist" / "SKILL.md"
    text = checklist.read_text(encoding="utf-8")
    referenced = set(re.findall(r"mlp_enabled|minority_threshold|ablation_enabled", text))
    covered = {key for _, key in Critic.CHECKLIST_CONFIG_KEYS}
    assert referenced <= covered, f"checklist references {referenced - covered}"


def test_the_block_reaches_the_rendered_prompt() -> None:
    """Useless unless the Critic actually sees it."""
    import inspect

    source = inspect.getsource(Critic._build_user_message)
    assert "_checklist_relevant_config()" in source
    assert "Run Configuration" in source
