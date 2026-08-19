"""Model tiering (2026-07-11): reasoning-light agents on
deepseek-v4-flash, reasoning-heavy agents on deepseek-v4-pro.

Pins (a) the shipped config.yaml tier assignment, and (b) the
ReviewGate revision-writer deepseek branch (previously the gate only
knew minimax/anthropic and would have sent a deepseek model ID to the
Anthropic endpoint once revision cycles were enabled).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.agents.provider_resolver import resolve_provider_for_stage

ROOT = Path(__file__).resolve().parent.parent

HEAVY_AGENTS = (
    "problem_formulator",
    "data_engineer",
    "analyst",
    "critic",
    "writer",
    "revision_writer",
)


@pytest.fixture(scope="module")
def repo_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestConfigTierAssignment:
    def test_outline_agent_routes_to_flash(self, repo_config: dict) -> None:
        cfg = resolve_provider_for_stage("outline_agent", repo_config)
        assert cfg.name == "deepseek"
        assert cfg.model == "deepseek-v4-flash"

    @pytest.mark.parametrize("agent_key", HEAVY_AGENTS)
    def test_heavy_agents_stay_on_pro(
        self, repo_config: dict, agent_key: str
    ) -> None:
        # Critic per SPEC uses the strongest tier; DE/Analyst generate
        # code (B7 reliability); Writer output is the product. None of
        # these may silently downgrade.
        cfg = resolve_provider_for_stage(agent_key, repo_config)
        assert cfg.name == "deepseek"
        assert cfg.model == "deepseek-v4-pro", agent_key


class TestReviewGateProviderRouting:
    def _base_cfg(self) -> dict:
        return {
            "llm_provider": "deepseek",
            "deepseek": {
                "base_url": "https://api.deepseek.com",
                "models": {"revision_writer": "deepseek-v4-pro"},
            },
            "review_gate": {
                "revision_model": "deepseek-v4-pro",
                "venue": "EDM",
            },
        }

    def test_deepseek_branch_builds_openai_client(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
        from src.review_gate import ReviewGate

        gate = ReviewGate(self._base_cfg(), str(tmp_path), log_fn=None)
        assert gate._llm_provider == "deepseek"
        assert gate._llm_model == "deepseek-v4-pro"
        # OpenAI-compatible client: chat.completions path exists,
        # anthropic messages.stream path does not.
        assert hasattr(gate._llm_client, "chat")
        assert not hasattr(gate._llm_client, "messages")

    def test_deepseek_model_falls_back_to_revision_model(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
        from src.review_gate import ReviewGate

        cfg = self._base_cfg()
        cfg["deepseek"].pop("models")
        cfg["review_gate"]["revision_model"] = "deepseek-v4-flash"
        gate = ReviewGate(cfg, str(tmp_path), log_fn=None)
        assert gate._llm_model == "deepseek-v4-flash"

    def test_anthropic_default_branch_unchanged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from src.review_gate import ReviewGate

        cfg = {
            "llm_provider": "anthropic",
            "review_gate": {"revision_model": "claude-sonnet-4-6"},
        }
        gate = ReviewGate(cfg, str(tmp_path), log_fn=None)
        assert gate._llm_provider == "anthropic"
        assert gate._llm_model == "claude-sonnet-4-6"
        assert hasattr(gate._llm_client, "messages")

    def test_revision_call_uses_chat_completions(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End-to-end through _revise_paper's deepseek branch with a
        stubbed client — the dormant-bug regression test."""
        from unittest.mock import MagicMock

        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
        from src.review_gate import ReviewGate

        gate = ReviewGate(self._base_cfg(), str(tmp_path), log_fn=None)
        fake = MagicMock()
        fake.chat.completions.create.return_value.choices = [
            MagicMock(
                message=MagicMock(
                    content=(
                        "```latex\n\\documentclass{article}"
                        "\\begin{document}revised\\end{document}\n```"
                    )
                )
            )
        ]
        gate._llm_client = fake
        revised = gate.revise_from_review(
            paper_tex="\\documentclass{article}\\begin{document}x\\end{document}",
            report_json={"review": {}},
            diagnosis={"suggested_focus_areas": []},
        )
        assert fake.chat.completions.create.called
        kwargs = fake.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == "deepseek-v4-pro"
        assert kwargs["messages"][0]["role"] == "system"
        assert "revised" in revised
