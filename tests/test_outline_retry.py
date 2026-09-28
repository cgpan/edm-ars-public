"""A cut-off outline answer is asked for again, with room, before the v1 fallback.

Observed on the owner's Mac (round 3, 2026-09-27): "OutlineAgent failed;
writing via the v1 template: Unterminated string starting at: line 109
column 23 (char 17064)". The outline stage runs on deepseek-flash with
thinking disabled (every DeepSeek request from BaseAgent says so), and its
prompt gave it max_tokens 4096; 17,064 characters of outline JSON is that
budget, spent. The answer was cut off mid-string, parse_llm_json raised,
and the Writer fell back to the v1 template without an outline.

The contract these tests pin:

* the outline stage asks for 8192 tokens, and every request to DeepSeek
  goes out with thinking disabled;
* BaseAgent records why the provider stopped ("length" for a cut-off);
* a cut-off or unparseable outline gets exactly one retry, with
  max(16000, twice the budget) and a note saying what went wrong; a
  second failure still raises, so the orchestrator falls back as before.

A fake client stands in for DeepSeek; nothing is sent anywhere.
"""
from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.agents.base import _finish_reason
from src.agents.outline_agent import OutlineAgent
from src.config import load_config
from src.context import PipelineContext
from tests.test_writer import CONFIG_PATH, _make_ctx

#: The real run(): tests/conftest.py replaces OutlineAgent.run for every
#: test so that no orchestrator test reaches a provider. It does that per
#: test, after this module was imported.
_REAL_RUN = OutlineAgent.run

OUTLINE = {
    "title": "Do Ninth-Grade Non-Cognitive Factors Improve Prediction?",
    "sections": [
        {"heading": f"Section {i}", "points": ["x" * 120] * 6} for i in range(20)
    ],
}
COMPLETE = "```json\n" + json.dumps(OUTLINE, indent=2) + "\n```"
#: Cut mid-string, as round 3's was.
TRUNCATED = COMPLETE[: COMPLETE.index("x" * 120, len(COMPLETE) // 2) + 50]


def _response(content: str, finish: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content=content), finish_reason=finish,
        )],
        usage=types.SimpleNamespace(prompt_tokens=1000, completion_tokens=500),
    )


def _agent(tmp_path: Path, *answers: tuple[str, str]) -> OutlineAgent:
    config = load_config(CONFIG_PATH)
    assert config["llm_provider"] == "deepseek"
    ctx: PipelineContext = _make_ctx(tmp_path)
    agent = OutlineAgent(ctx, "outline_agent", config)
    agent.client = MagicMock()
    agent.client.chat.completions.create.side_effect = [
        _response(c, f) for c, f in answers
    ]
    return agent


def _requests(agent: OutlineAgent) -> list[dict[str, Any]]:
    return [c.kwargs for c in agent.client.chat.completions.create.call_args_list]


def test_the_outline_stage_asks_for_8192_tokens_on_flash_with_thinking_off(
    tmp_path: Path,
) -> None:
    agent = _agent(tmp_path, (COMPLETE, "stop"))

    outline = _REAL_RUN(agent)

    assert outline == OUTLINE
    [request] = _requests(agent)
    assert request["model"] == "deepseek-flash"
    assert request["max_tokens"] == 8192
    assert request["extra_body"] == {"thinking": {"type": "disabled"}}
    assert agent.last_finish_reason == "stop"
    saved = json.loads((tmp_path / "paper_outline.json").read_text(encoding="utf-8"))
    assert saved == OUTLINE


def test_a_cut_off_outline_is_asked_for_once_more_with_room(tmp_path: Path) -> None:
    with pytest.raises(json.JSONDecodeError, match="Unterminated string"):
        json.loads(TRUNCATED.removeprefix("```json\n"))
    agent = _agent(tmp_path, (TRUNCATED, "length"), (COMPLETE, "stop"))

    outline = _REAL_RUN(agent)

    assert outline == OUTLINE
    first, retry = _requests(agent)
    assert (first["max_tokens"], retry["max_tokens"]) == (8192, 16384)
    for request in (first, retry):
        assert request["extra_body"] == {"thinking": {"type": "disabled"}}
    retry_text = retry["messages"][1]["content"]
    assert retry_text.startswith(first["messages"][1]["content"])
    assert "was cut off at the 8192-token limit" in retry_text
    log = (tmp_path / "pipeline.log").read_text(encoding="utf-8")
    assert "asking once more with max_tokens=16384" in log


def test_an_answer_that_is_not_json_is_asked_for_once_more(tmp_path: Path) -> None:
    agent = _agent(tmp_path, ("Here is the outline you asked for.", "stop"),
                   (COMPLETE, "stop"))

    assert _REAL_RUN(agent) == OUTLINE
    _, retry = _requests(agent)
    assert "was not valid JSON" in retry["messages"][1]["content"]


def test_a_second_cut_off_still_raises_for_the_v1_fallback(tmp_path: Path) -> None:
    agent = _agent(tmp_path, (TRUNCATED, "length"), (TRUNCATED, "length"))

    with pytest.raises(ValueError, match="cut off again at max_tokens=16384"):
        _REAL_RUN(agent)
    assert len(_requests(agent)) == 2
    assert not (tmp_path / "paper_outline.json").exists()


def test_the_retry_budget_doubles_a_larger_configured_budget(tmp_path: Path) -> None:
    agent = _agent(tmp_path, (TRUNCATED, "length"), (COMPLETE, "stop"))
    agent.max_tokens = 12000

    _REAL_RUN(agent)

    assert [r["max_tokens"] for r in _requests(agent)] == [12000, 24000]


@pytest.mark.parametrize(
    "raw, normalised",
    [("length", "length"), ("max_tokens", "length"), ("stop", "stop"),
     ("end_turn", "end_turn"), (None, None), ("", None), (MagicMock(), None)],
)
def test_the_stop_reason_is_normalised(raw: Any, normalised: str | None) -> None:
    assert _finish_reason(raw) == normalised
