"""Resolve `references_skills` transitively and format skills for prompts.

A skill can name other skills in `references_skills`; those references are
walked transitively and appended to the matched output. Cycles are detected
and broken with a warning. Missing references warn but do not crash.
"""
from __future__ import annotations

import logging
from typing import Iterable

from src.skills.schema import RULE_SEVERITY_ORDER, Skill

_MANDATORY_BANNER = (
    "**The following rules are binding. Violating them will cause pipeline failure.**"
)

logger = logging.getLogger(__name__)

_TRUNCATION_NOTICE = (
    "<!-- skills truncated: dropped lowest-priority skills to fit max_chars -->"
)

# Phase 3b.8 / §6.1: replacement diagnostic for the per-tier-cap path.
# Format: "<!-- Dropped from prompt due to budget: name1 (N chars), name2 (M chars) -->"
_PER_TIER_DROP_PREFIX = "<!-- Dropped from prompt due to budget: "
_PER_TIER_DROP_SUFFIX = " -->"


def resolve_references(
    matched: Iterable[Skill],
    all_skills: dict[str, Skill],
) -> list[Skill]:
    """Return matched skills followed by their transitive `references_skills`.

    References appear after their referrer. The same skill is never included
    twice. Cycles are broken with a warning. Missing references warn.
    """
    output: list[Skill] = []
    in_output: set[str] = set()
    in_progress: set[str] = set()

    def visit(skill: Skill) -> None:
        # Order matters: check in_progress before in_output so we surface
        # cycles instead of silently deduping them.
        if skill.name in in_progress:
            logger.warning(
                "Skill composer: cycle detected at %r; breaking", skill.name
            )
            return
        if skill.name in in_output:
            return
        in_progress.add(skill.name)
        # Add the skill itself first so referrer comes before referenced.
        output.append(skill)
        in_output.add(skill.name)
        for ref_name in skill.references_skills:
            ref = all_skills.get(ref_name)
            if ref is None:
                logger.warning(
                    "Skill composer: %r references missing skill %r",
                    skill.name,
                    ref_name,
                )
                continue
            visit(ref)
        in_progress.discard(skill.name)

    for skill in matched:
        visit(skill)
    return output


def format_skills_for_prompt(
    skills: list[Skill],
    max_chars: int = 30000,
    *,
    mandatory_chars_unlimited: bool = True,
) -> str:
    """Concatenate skill bodies under stable headers.

    Skills are reordered by ``rule_severity`` (mandatory first, then
    recommended, then reference) so the LLM sees binding rules at the top
    of the injected block. Within each severity tier the original input
    order is preserved (stable sort).

    Tier discipline (Phase 3b.8 / §6.1):

      mandatory_chars_unlimited=True (DEFAULT, post-3b.8):
        - Mandatory-tier skills render in full regardless of ``max_chars``.
          Their byte cost is recorded but does NOT count against the
          recommended/reference tier budget.
        - Non-mandatory tiers (recommended + reference) compete for
          ``max_chars`` of room. If their total exceeds that budget,
          drops happen by lowest priority first (highest ``priority``
          integer = lowest rank in our convention).
        - When drops occur, a diagnostic comment line is appended listing
          the dropped skills with their char counts (for human inspection;
          downstream agents should not parse it).

      mandatory_chars_unlimited=False (LEGACY pre-3b.8 behavior):
        - All tiers compete for one shared ``max_chars`` budget. Mandatory
          is still protected from truncation, so when total > max_chars
          the formatter drops non-mandatory skills until either total
          fits or all non-mandatory are gone (then mandatory may push
          total over budget). This was the 3b.7 behavior that produced
          F-3b7-FORMATTER-TRUNCATES-METHOD-SKILLS — under causal_soo with
          7 mandatory skills exceeding 12000 chars, every recommended
          skill (M1-M5, G1, G4) was silently dropped.

    The legacy path is preserved for callers that want byte-uniform
    constraints (e.g., experiments comparing prompt sizes; regression
    tests). The default path (mandatory unlimited) is what production
    causal_soo runs use.
    """
    # 1. Stable severity-aware reorder.
    indexed = list(enumerate(skills))
    indexed.sort(key=lambda pair: (RULE_SEVERITY_ORDER.get(pair[1].rule_severity, 99), pair[0]))
    ordered = [skill for _, skill in indexed]

    rendered = [_render(skill) for skill in ordered]

    if not mandatory_chars_unlimited:
        return _legacy_uniform_cap(ordered, rendered, max_chars)

    return _per_tier_cap(ordered, rendered, max_chars)


def _per_tier_cap(
    ordered: list[Skill],
    rendered: list[str],
    max_chars: int,
) -> str:
    """Mandatory always renders; recommended/reference share max_chars.

    Implementation contract for Phase 3b.8 / §6.1:
      - Mandatory bodies are concatenated unconditionally.
      - Non-mandatory bodies are concatenated, then dropped by lowest
        priority first if total exceeds ``max_chars``.
      - Drop diagnostic appended as a comment line listing
        (skill_name, char_count) for each dropped skill, in drop order.
    """
    mandatory_idx = [i for i, s in enumerate(ordered) if s.rule_severity == "mandatory"]
    nonmandatory_idx = [
        i for i, s in enumerate(ordered) if s.rule_severity != "mandatory"
    ]

    rendered_mandatory = [rendered[i] for i in mandatory_idx]

    # Compute non-mandatory drops independently of mandatory cost.
    nonmandatory_total = sum(len(rendered[i]) for i in nonmandatory_idx)
    keep_nonmandatory = {i: True for i in nonmandatory_idx}
    drops: list[tuple[str, int]] = []  # (name, char_count) in drop order

    if nonmandatory_total > max_chars:
        # Drop lowest-priority first within non-mandatory; ties by
        # later-in-input-order (negative index sort).
        drop_order = sorted(
            nonmandatory_idx,
            key=lambda i: (-ordered[i].priority, -i),
        )
        for idx in drop_order:
            if nonmandatory_total <= max_chars:
                break
            n = len(rendered[idx])
            keep_nonmandatory[idx] = False
            nonmandatory_total -= n
            drops.append((ordered[idx].name, n))

    # Assemble: mandatory block first (in its sorted order), then surviving
    # non-mandatory in their sorted order.
    parts = list(rendered_mandatory)
    for i in nonmandatory_idx:
        if keep_nonmandatory[i]:
            parts.append(rendered[i])

    output = "".join(parts)
    if drops:
        drop_clause = ", ".join(f"{name} ({n} chars)" for name, n in drops)
        output += _PER_TIER_DROP_PREFIX + drop_clause + _PER_TIER_DROP_SUFFIX + "\n"
        # WARNING-level log so operators see the drop in the pipeline.log
        # without parsing the rendered prompt. Same shape as V2.0.1
        # empty-match logging.
        for name, n in drops:
            logger.warning(
                "format_skills_for_prompt: dropped non-mandatory skill %r "
                "(%d chars) due to budget (max_chars=%d, "
                "non-mandatory total exceeded budget)",
                name,
                n,
                max_chars,
            )
    return output


def _legacy_uniform_cap(
    ordered: list[Skill],
    rendered: list[str],
    max_chars: int,
) -> str:
    """Pre-3b.8 uniform-cap behavior. Retained for backward compatibility.

    All tiers compete for the same budget; mandatory is protected from
    drops. When mandatory alone exceeds budget, all non-mandatory drops
    and total still exceeds the cap.
    """
    total = sum(len(r) for r in rendered)
    if total <= max_chars:
        return "".join(rendered)

    drop_indexed = [(i, s) for i, s in enumerate(ordered) if s.rule_severity != "mandatory"]
    drop_order = sorted(drop_indexed, key=lambda pair: (-pair[1].priority, -pair[0]))
    keep_mask = [True] * len(ordered)
    for idx, _skill in drop_order:
        if total <= max_chars:
            break
        if not keep_mask[idx]:
            continue
        keep_mask[idx] = False
        total -= len(rendered[idx])

    kept = [rendered[i] for i, keep in enumerate(keep_mask) if keep]
    return "".join(kept) + _TRUNCATION_NOTICE + "\n"


def _render(skill: Skill) -> str:
    """Render a single skill with severity-appropriate framing.

    - mandatory   → "## MANDATORY RULE: <name>" + binding-rules banner
    - recommended → "## Guidance: <name>"
    - reference   → "### Reference: <name>" (softer; H3, no separator)
    """
    body = skill.body.rstrip()
    if skill.rule_severity == "mandatory":
        return (
            f"## MANDATORY RULE: {skill.name}\n\n"
            f"{_MANDATORY_BANNER}\n\n"
            f"{body}\n\n---\n"
        )
    if skill.rule_severity == "reference":
        return f"### Reference: {skill.name}\n\n{body}\n"
    # default: recommended
    return f"## Guidance: {skill.name}\n\n{body}\n\n---\n"
