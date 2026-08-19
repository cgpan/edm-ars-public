"""Match skills to a (stage, task_type, dataset, context) request.

Filtering is applied as hard gates first (stage → task_type → dataset),
then surviving skills are scored by keyword overlap and priority, and
finally capped per layer. Output is grouped by layer in a fixed order
so prompt assembly is deterministic.
"""
from __future__ import annotations

import re
from typing import Iterable

from src.skills.schema import LAYER_ORDER, LAYERS, Skill

DEFAULT_TOP_K_PER_LAYER: dict[str, int] = {
    "task-type": 2,
    "dataset": 2,
    "methodology": 4,
    "writing": 3,
}

_WORD_RE = re.compile(r"[A-Za-z0-9_-]+")

# Minimum length of the post-suffix-strip stem. Anything shorter is left
# unstemmed (avoids mapping `is` → `i`, `as` → `a`, `use` → `us`).
_MIN_STEM_LEN = 4

# Ordered list of suffixes to try stripping. Priority order per Phase 2c
# spec: s, es, ies, ing, ed. Trying `-s` first means "tables" -> "table"
# (rather than "tabl" via -es), which is what natural-language matching
# should produce. Words like "puppies" stem to "puppie" rather than
# "puppy" — acceptable given this is a coarse keyword-matching stemmer,
# not a full Porter algorithm.
_SUFFIXES: tuple[str, ...] = ("s", "es", "ies", "ing", "ed")


def _stem(token: str) -> str:
    """Naive plural/gerund-aware stemmer.

    Strips one trailing suffix from ``token`` if the remainder is at least
    ``_MIN_STEM_LEN`` characters. Returns the original token otherwise.
    Intentionally simple — enough to make "table"/"tables" and
    "analyze"/"analyzing" collide without pulling in a real NLP stemmer.
    """
    for suf in _SUFFIXES:
        if len(token) > len(suf) and token.endswith(suf):
            stem = token[: -len(suf)]
            if len(stem) >= _MIN_STEM_LEN:
                return stem
    return token


def _tokenize(text: str) -> set[str]:
    """Tokenize ``text`` into a set of lowercase, suffix-stripped stems."""
    return {_stem(tok.lower()) for tok in _WORD_RE.findall(text)}


def _priority_boost(priority: int) -> float:
    # Lower priority value = higher rank. Map priority 1..10 → 0.45..0.0.
    return max(0.0, (10 - priority) / 20.0)


def _score(skill: Skill, context_tokens: set[str]) -> float:
    """Compute a relevance score in roughly [0, 1.5].

    Both the skill's trigger_keywords and the context tokens are
    suffix-stripped before set intersection so that, e.g., a skill keyed
    on `table` still fires on a context that mentions "tables".
    """
    keyword_score = 0.0
    if skill.trigger_keywords:
        if context_tokens:
            kw_tokens = {_stem(kw.lower()) for kw in skill.trigger_keywords}
            overlap = len(kw_tokens & context_tokens)
            keyword_score = overlap / len(skill.trigger_keywords)
    return keyword_score + _priority_boost(skill.priority)


def match_skills(
    skills: Iterable[Skill],
    *,
    stage: str,
    task_type: str,
    dataset: str,
    context: str = "",
    top_k_per_layer: dict[str, int] | None = None,
) -> list[Skill]:
    """Return the skills that pass the hard filters and survive per-layer caps.

    Skills are ordered by layer (task-type → dataset → methodology → writing)
    then by descending score within each layer. The output never contains a
    duplicate skill (matched by ``name``).
    """
    caps = dict(DEFAULT_TOP_K_PER_LAYER)
    if top_k_per_layer:
        caps.update(top_k_per_layer)

    context_tokens = _tokenize(context) if context else set()

    surviving: list[tuple[Skill, float]] = []
    seen: set[str] = set()
    for skill in skills:
        if skill.name in seen:
            continue
        if not skill.applies_to_stage(stage):
            continue
        if not skill.applies_to_task_type(task_type):
            continue
        if not skill.applies_to_dataset(dataset):
            continue
        seen.add(skill.name)
        surviving.append((skill, _score(skill, context_tokens)))

    # Group by layer, sort each group, apply cap.
    # CORRECTNESS RULE (Phase 2c continuation): mandatory-tagged skills
    # bypass the per-layer cap. The cap exists to bound prompt size by
    # trimming the recommended/reference tail; mandatory rules are
    # crash-risk by definition and must reach the agent regardless of
    # how many sibling skills tied on score. Mandatory-skill count is
    # capped at the registry-authoring level (~8 total per recovery
    # spec), so unbounded inclusion is safe.
    by_layer: dict[str, list[tuple[Skill, float]]] = {layer: [] for layer in LAYERS}
    for entry in surviving:
        by_layer.setdefault(entry[0].layer, []).append(entry)

    output: list[Skill] = []
    for layer in sorted(by_layer.keys(), key=lambda layer_: LAYER_ORDER.get(layer_, 99)):
        # Stable sort by score desc; preserves insertion order for equal scores.
        ranked = sorted(by_layer[layer], key=lambda pair: -pair[1])
        cap = caps.get(layer, len(ranked))
        kept: list[Skill] = []
        kept_names: set[str] = set()
        # Pass 1: take the cap'd top of the score-ranked list.
        for skill, _score_val in ranked[:cap]:
            if skill.name in kept_names:
                continue
            kept.append(skill)
            kept_names.add(skill.name)
        # Pass 2: ensure every mandatory skill in this layer is included,
        # even if it fell outside the cap.
        for skill, _score_val in ranked[cap:]:
            if skill.rule_severity == "mandatory" and skill.name not in kept_names:
                kept.append(skill)
                kept_names.add(skill.name)
        output.extend(kept)
    return output
