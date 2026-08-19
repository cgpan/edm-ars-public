"""EDM-ARS skill registry.

Skills are markdown documents with YAML frontmatter that encode reusable
knowledge for the multi-agent pipeline. They are organized by layer
(task-type, dataset, methodology, writing) and selected at runtime by
agent name, task type, dataset, and free-text context.
"""
from __future__ import annotations

from src.skills.composer import format_skills_for_prompt, resolve_references
from src.skills.loader import load_skill_from_skillmd, load_skills_from_directory
from src.skills.matcher import match_skills
from src.skills.registry import SkillRegistry
from src.skills.schema import LAYERS, Skill

__all__ = [
    "LAYERS",
    "Skill",
    "SkillRegistry",
    "format_skills_for_prompt",
    "load_skill_from_skillmd",
    "load_skills_from_directory",
    "match_skills",
    "resolve_references",
]
