"""SkillRegistry — facade over loader + matcher + composer.

Agents call `match_and_compose()` (or the convenience `format_for_prompt()`)
to obtain the skill content that should be injected into their system
prompt for a given (stage, task_type, dataset) request.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from src.skills.composer import format_skills_for_prompt, resolve_references
from src.skills.loader import load_skills_from_directory
from src.skills.matcher import match_skills
from src.skills.schema import Skill


class SkillRegistry:
    """In-memory registry of all skills under a given root directory."""

    def __init__(self, skills_root: Path) -> None:
        self.skills_root: Path = Path(skills_root)
        self._skills: list[Skill] = []
        self._by_name: dict[str, Skill] = {}
        self.reload()

    def reload(self) -> None:
        self._skills = load_skills_from_directory(self.skills_root)
        self._by_name = {s.name: s for s in self._skills}

    def count(self) -> int:
        return len(self._skills)

    def count_by_layer(self) -> dict[str, int]:
        return dict(Counter(s.layer for s in self._skills))

    def get(self, name: str) -> Skill | None:
        return self._by_name.get(name)

    def all(self) -> list[Skill]:
        return list(self._skills)

    def match(
        self,
        *,
        stage: str,
        task_type: str,
        dataset: str,
        context: str = "",
        top_k_per_layer: dict[str, int] | None = None,
    ) -> list[Skill]:
        return match_skills(
            self._skills,
            stage=stage,
            task_type=task_type,
            dataset=dataset,
            context=context,
            top_k_per_layer=top_k_per_layer,
        )

    def match_and_compose(
        self,
        *,
        stage: str,
        task_type: str,
        dataset: str,
        context: str = "",
        top_k_per_layer: dict[str, int] | None = None,
    ) -> list[Skill]:
        matched = self.match(
            stage=stage,
            task_type=task_type,
            dataset=dataset,
            context=context,
            top_k_per_layer=top_k_per_layer,
        )
        return resolve_references(matched, self._by_name)

    def format_for_prompt(
        self,
        *,
        stage: str,
        task_type: str,
        dataset: str,
        context: str = "",
        max_chars: int = 30000,
        top_k_per_layer: dict[str, int] | None = None,
    ) -> str:
        composed = self.match_and_compose(
            stage=stage,
            task_type=task_type,
            dataset=dataset,
            context=context,
            top_k_per_layer=top_k_per_layer,
        )
        return format_skills_for_prompt(composed, max_chars=max_chars)
