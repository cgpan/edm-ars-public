"""Skill dataclass and layer enumeration.

A Skill is a markdown document (SKILL.md) with YAML frontmatter, organized
under one of four layers. Layer is required and drives matching semantics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

LAYERS: tuple[str, ...] = ("task-type", "dataset", "methodology", "writing")
LAYER_ORDER: dict[str, int] = {layer: i for i, layer in enumerate(LAYERS)}

# rule_severity gates the rendering treatment in format_skills_for_prompt.
# - "mandatory":   skill is rendered first with a strong "MANDATORY RULE" header
#                  + a binding-rules notice; reserved for skills whose violation
#                  causes pipeline failure (e.g., the categorical pd.to_numeric
#                  trap that broke Phase 2c Checkpoint 3).
# - "recommended": default; rendered as "Guidance: ..." in normal position.
# - "reference":   purely informational; rendered last with a softer H3 header.
RULE_SEVERITIES: tuple[str, ...] = ("mandatory", "recommended", "reference")
RULE_SEVERITY_ORDER: dict[str, int] = {sev: i for i, sev in enumerate(RULE_SEVERITIES)}
RuleSeverity = Literal["mandatory", "recommended", "reference"]


@dataclass
class Skill:
    """A composable unit of agent knowledge.

    Required frontmatter:
      - name: kebab-case, unique within a layer
      - layer: one of LAYERS
      - description: one sentence used by the matcher
      - body: markdown content following the frontmatter

    Optional metadata (defaults applied below) controls when the skill
    is included in an agent's prompt and how it composes with siblings.
    """

    name: str
    layer: str
    description: str
    body: str

    trigger_keywords: list[str] = field(default_factory=list)
    applicable_task_types: list[str] = field(default_factory=list)
    applicable_datasets: list[str] = field(default_factory=list)
    applicable_stages: list[str] = field(default_factory=list)
    priority: int = 5
    references_skills: list[str] = field(default_factory=list)
    resources: list[str] = field(default_factory=list)
    version: str = "1.0"
    rule_severity: RuleSeverity = "recommended"

    source_dir: Path | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Skill.name is required")
        if not self.description:
            raise ValueError(f"Skill.description is required (skill={self.name!r})")
        if self.layer not in LAYERS:
            raise ValueError(
                f"Skill.layer must be one of {list(LAYERS)} "
                f"(got {self.layer!r}, skill={self.name!r})"
            )
        if self.rule_severity not in RULE_SEVERITIES:
            raise ValueError(
                f"Skill.rule_severity must be one of {list(RULE_SEVERITIES)} "
                f"(got {self.rule_severity!r}, skill={self.name!r})"
            )

    @property
    def resource_paths(self) -> list[Path]:
        """Absolute paths to bundled resources, resolved from source_dir."""
        if self.source_dir is None:
            return []
        return [(self.source_dir / r).resolve() for r in self.resources]

    def applies_to_stage(self, stage: str) -> bool:
        return not self.applicable_stages or stage in self.applicable_stages

    def applies_to_task_type(self, task_type: str) -> bool:
        return not self.applicable_task_types or task_type in self.applicable_task_types

    def applies_to_dataset(self, dataset: str) -> bool:
        return not self.applicable_datasets or dataset in self.applicable_datasets

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "layer": self.layer,
            "description": self.description,
            "body": self.body,
            "trigger_keywords": list(self.trigger_keywords),
            "applicable_task_types": list(self.applicable_task_types),
            "applicable_datasets": list(self.applicable_datasets),
            "applicable_stages": list(self.applicable_stages),
            "priority": self.priority,
            "references_skills": list(self.references_skills),
            "resources": list(self.resources),
            "version": self.version,
            "rule_severity": self.rule_severity,
            "source_dir": str(self.source_dir) if self.source_dir else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Skill:
        source_dir_raw = data.get("source_dir")
        source_dir = Path(source_dir_raw) if source_dir_raw else None
        return cls(
            name=data["name"],
            layer=data["layer"],
            description=data["description"],
            body=data.get("body", ""),
            trigger_keywords=list(data.get("trigger_keywords", []) or []),
            applicable_task_types=list(data.get("applicable_task_types", []) or []),
            applicable_datasets=list(data.get("applicable_datasets", []) or []),
            applicable_stages=list(data.get("applicable_stages", []) or []),
            priority=int(data.get("priority", 5)),
            references_skills=list(data.get("references_skills", []) or []),
            resources=list(data.get("resources", []) or []),
            version=str(data.get("version", "1.0")),
            rule_severity=data.get("rule_severity") or "recommended",
            source_dir=source_dir,
        )
