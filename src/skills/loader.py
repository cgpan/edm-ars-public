"""Load Skill objects from SKILL.md files on disk.

Convention:
    skills/<layer>/<skill_name>/SKILL.md (+ optional resource files alongside)

The loader is forgiving: malformed files log a warning and return None
instead of raising, so a single bad file does not break the registry.
"""
from __future__ import annotations

import logging
from pathlib import Path

import yaml  # type: ignore[import-untyped]

from src.skills.schema import LAYERS, Skill

logger = logging.getLogger(__name__)

_FRONTMATTER_DELIM = "---"


def _split_frontmatter(text: str) -> tuple[str, str] | None:
    """Return (frontmatter_yaml, body) or None if delimiters are missing."""
    stripped = text.lstrip()
    if not stripped.startswith(_FRONTMATTER_DELIM):
        return None
    # Find the closing delimiter on its own line, after the opening one.
    lines = text.splitlines()
    # Locate first non-empty line; it should be the opening ---.
    open_idx = next((i for i, ln in enumerate(lines) if ln.strip()), -1)
    if open_idx < 0 or lines[open_idx].strip() != _FRONTMATTER_DELIM:
        return None
    close_idx = next(
        (i for i in range(open_idx + 1, len(lines)) if lines[i].strip() == _FRONTMATTER_DELIM),
        -1,
    )
    if close_idx < 0:
        return None
    fm = "\n".join(lines[open_idx + 1 : close_idx])
    body = "\n".join(lines[close_idx + 1 :]).lstrip("\n")
    return fm, body


def load_skill_from_skillmd(path: Path) -> Skill | None:
    """Parse a single SKILL.md file into a Skill, or return None on failure."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Skill loader: cannot read %s: %s", path, exc)
        return None

    split = _split_frontmatter(text)
    if split is None:
        logger.warning("Skill loader: %s has no YAML frontmatter (--- delimiters)", path)
        return None
    fm_text, body = split

    try:
        fm = yaml.safe_load(fm_text) or {}
    except yaml.YAMLError as exc:
        logger.warning("Skill loader: invalid YAML frontmatter in %s: %s", path, exc)
        return None
    if not isinstance(fm, dict):
        logger.warning(
            "Skill loader: frontmatter in %s must be a mapping, got %s",
            path,
            type(fm).__name__,
        )
        return None

    # Reconcile frontmatter layer with directory layer.
    inferred_layer = _infer_layer_from_path(path)
    fm_layer = fm.get("layer")
    if fm_layer is None:
        if inferred_layer is None:
            logger.warning(
                "Skill loader: %s has no layer in frontmatter and none inferable from path",
                path,
            )
            return None
        fm["layer"] = inferred_layer
    elif inferred_layer is not None and fm_layer != inferred_layer:
        logger.warning(
            "Skill loader: %s layer mismatch (frontmatter=%r, dir=%r); using frontmatter",
            path,
            fm_layer,
            inferred_layer,
        )

    fm["body"] = body
    fm.setdefault("source_dir", str(path.parent))

    try:
        return Skill.from_dict(fm)
    except (KeyError, ValueError) as exc:
        logger.warning("Skill loader: %s failed schema validation: %s", path, exc)
        return None


def _infer_layer_from_path(path: Path) -> str | None:
    """Infer layer from the directory two levels above SKILL.md.

    Expected: <root>/<layer>/<skill_name>/SKILL.md → returns <layer>.
    """
    parts = path.resolve().parts
    # Look for any LAYERS token in the path; pick the closest to SKILL.md.
    for part in reversed(parts[:-1]):
        if part in LAYERS:
            return part
    return None


def load_skills_from_directory(root: Path) -> list[Skill]:
    """Recursively load every SKILL.md beneath root.

    Skills are returned in deterministic order (sorted by layer then name)
    so registry behavior is reproducible across runs.
    """
    if not root.exists():
        logger.warning("Skill loader: root %s does not exist", root)
        return []

    skills: list[Skill] = []
    for path in sorted(root.rglob("SKILL.md")):
        skill = load_skill_from_skillmd(path)
        if skill is not None:
            skills.append(skill)
    skills.sort(key=lambda s: (s.layer, s.name))
    return skills
