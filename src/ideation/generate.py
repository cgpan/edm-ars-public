"""Arc T / T1a - one INDEPENDENT generation draw per slate cell.

Independence is the whole design. Measured: asking one prompt for
several ideas produced a mean pairwise predictor Jaccard of 0.837
across the candidate specs, all on one outcome. Agents that share a
prompt, a scratchpad, or a debate transcript converge, and the effect
gets worse with more communication rounds. So: one call per cell, no
cross-candidate context, no anti-repetition list injected at generation
time. The slate has already done that job, deterministically.

Everything that varies between draws comes from the slate cell, which
was fixed before any call was made. Temperature is the SAME for every
draw in a run (0.9 by default, settable once in config) - the previous
0.70/0.85/1.00 ramp confounded diversity with quality, because the
first and most conservative draw was also the one that won every tie
in the old selector.

LLM routing: production draws go through ``BaseAgent.call_llm`` via
:class:`IdeaGeneratorAgent`. Every function in this module takes the
caller as a plain ``Callable[[str], str]``, so tests run fully offline
with a stub and no provider SDK is touched.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from src.ideation import cards as _cards
from src.ideation import feasibility as _feas
from src.ideation import slate as _slate
from src.ideation.cards import IdeaCard
from src.ideation.slate import SlateCell

LLMCaller = Callable[[str], str]

#: Spec sec. 2.4. Fixed for every draw, deliberately.
GENERATION_TEMPERATURE = 0.9

#: Spec sec. 2.5 / config ``ideation.tournament.dedupe_cosine``.
DEFAULT_DEDUPE_COSINE = 0.80

AGENT_KEY = "idea_generator"

#: Used only when ``agent_prompts/idea_generator.yaml`` is absent.
#: BaseAgent loads that file automatically when it exists, which is
#: where this text belongs (project rule: system prompts live in
#: agent_prompts/*.yaml). The fallback keeps the module runnable before
#: that file ships; the whole task-specific instruction set is in the
#: user message either way.
FALLBACK_SYSTEM_PROMPT = (
    "You are a research-idea generator for an educational data mining "
    "group. You propose ONE research idea per request, grounded in the "
    "dataset facts you are given. You never rate your own novelty and "
    "never claim to be first at anything. You answer with JSON only."
)

_JSON_KEYS = (
    "research_question",
    "why_it_matters",
    "what_we_would_do",
    "what_counts_as_the_result",
    "method_family",
    "second_contribution",
    "spec_draft",
)


# --------------------------------------------------------------------------
# Prompt construction (deterministic, offline)
# --------------------------------------------------------------------------


def _fmt_var(meta: dict) -> str:
    bits = [str(meta.get("name"))]
    detail = [
        str(meta.get("type") or "?"),
        f"wave={meta.get('wave')}",
        f"pct_missing={meta.get('pct_missing')}",
    ]
    if meta.get("protected_attribute"):
        detail.append("protected")
    bits.append("(" + ", ".join(detail) + ")")
    label = str(meta.get("label") or "").strip()
    if label:
        bits.append(f"- {label}")
    return " ".join(bits)


def dataset_facts_block(
    dataset: str | None,
    *,
    registry_dir: str | os.PathLike[str] | None = None,
    registry: dict | None = None,
    max_predictors_per_category: int = 6,
) -> str:
    """Registry facts the draw must stay inside. Deterministic ordering."""
    if not dataset:
        return "No dataset registry available."
    facts = _cards._facts(dataset, registry, registry_dir)
    lines = [f"Registry: {facts.source}"]
    if facts.temporal_order:
        lines.append("Temporal order: " + " -> ".join(facts.temporal_order))
    lines.append("Outcomes available:")
    for meta in sorted(facts.outcomes, key=lambda m: str(m.get("name"))):
        if _cards._usable(meta):
            lines.append("  - " + _fmt_var(meta))
    by_category: dict[str, list[dict]] = {}
    for meta in facts.predictors:
        if _cards._usable(meta):
            by_category.setdefault(str(meta.get("_category")), []).append(meta)
    lines.append("Predictors available (by registry category):")
    for category in sorted(by_category):
        members = sorted(
            by_category[category],
            key=lambda m: (_cards.pct_missing(m), str(m.get("name"))),
        )[:max_predictors_per_category]
        lines.append(f"  {category}:")
        for meta in members:
            lines.append("    - " + _fmt_var(meta))
    lines.append(
        "Protected attributes: "
        + (", ".join(facts.protected) if facts.protected else "NONE DECLARED")
    )
    if facts.item_banks:
        lines.append("Item banks (item-level responses exist):")
        for name, bank in sorted(facts.item_banks.items()):
            items = (bank or {}).get("items") or []
            lines.append(f"  - {name}: {len(items)} items {list(items)}")
    if facts.cdm_support:
        lines.append(
            "Log-derived measurement support: "
            + str(facts.cdm_support.get("recommended_scope") or "")
        )
    return "\n".join(lines)


def build_user_message(
    cell: SlateCell,
    *,
    registry_dir: str | os.PathLike[str] | None = None,
    registry: dict | None = None,
    exemplars: str | None = None,
) -> str:
    """The per-cell user message. Pure function of the cell + registry."""
    pattern_brief = _slate.PATTERN_BRIEFS.get(cell.opportunity_pattern, "")
    persona_brief = _slate.PERSONA_BRIEFS.get(cell.persona, "")
    gap = (
        f"{cell.gap_cell[0]} x {cell.gap_cell[1]}"
        if cell.gap_cell
        else "none assigned"
    )
    caps = _cards.FIELD_CAPS
    schema = {
        "research_question": f"string, <= {caps['research_question']} words",
        "why_it_matters": f"string, <= {caps['why_it_matters']} words",
        "what_we_would_do": f"string, <= {caps['what_we_would_do']} words",
        "what_counts_as_the_result": (
            f"string, <= {caps['what_counts_as_the_result']} words; the "
            f"observable that would settle the question, including the "
            f"direction or magnitude that would count"
        ),
        "method_family": "one of " + str(sorted(_cards.KNOWN_METHOD_FAMILIES)),
        "second_contribution": (
            "one of "
            + str(sorted(_cards.SECOND_CONTRIBUTIONS))
            + " or null - a SECOND contribution beyond the headline result"
        ),
        "spec_draft": (
            "object - the research_spec fields you can already fill for "
            f"task_type {cell.task_type!r}; omit anything you cannot ground "
            "in the dataset facts above rather than inventing it"
        ),
    }
    parts = [
        "# Assignment (fixed - you may not change any of these)",
        f"dataset: {cell.dataset}",
        f"task_type: {cell.task_type}",
        f"opportunity pattern: {cell.opportunity_pattern}",
        f"  {pattern_brief}",
        f"persona: {cell.persona}",
        f"  {persona_brief}",
        f"under-retrieved gap cell to aim at: {gap}",
        "",
        "# Dataset facts",
        dataset_facts_block(
            cell.dataset, registry_dir=registry_dir, registry=registry
        ),
        "",
        "# Return JSON only, with exactly these keys",
        json.dumps(schema, indent=1),
        "",
        "# Hard rules",
        "1. Do not change the dataset or the task type. They are structural.",
        "2. Name only variables that appear in the dataset facts above. If "
        "the idea needs a variable that is not listed, choose a different "
        "idea.",
        "3. Do NOT include a novelty score, a self-rating, a confidence "
        "number, or any claim that this is the first study of anything. "
        "Unsupported first-claims are scored DOWN by reviewers and any "
        "such field is discarded before scoring.",
        "4. Respect the word caps. Longer answers are truncated before "
        "anyone reads them, so extra words are lost, not rewarded.",
        "5. State what would count as the result concretely enough that a "
        "null result would be recognisable as a null result.",
    ]
    if exemplars:
        parts.extend(["", "# Exemplars from previous sessions", exemplars])
    return "\n".join(parts)


# --------------------------------------------------------------------------
# Response parsing
# --------------------------------------------------------------------------

_FENCE = re.compile(r"^```(?:json)?\s*\n?|\n?```\s*$", re.MULTILINE)


def parse_response(text: str) -> dict:
    """Parse a generator response into a dict. Raises ValueError."""
    if not text or not text.strip():
        raise ValueError("empty response")
    stripped = _FENCE.sub("", text.strip()).strip()
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        data = _first_json_object(stripped)
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object, got {type(data).__name__}")
    return data


def _first_json_object(text: str) -> dict:
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found in the response")
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : index + 1])
    raise ValueError("unbalanced JSON object in the response")


def card_from_response(
    data: dict,
    cell: SlateCell,
    *,
    tournament_id: str,
    generator_model: str = "",
) -> IdeaCard:
    """Build an IdeaCard from a parsed response, cell facts winning."""
    payload = {key: data.get(key) for key in _JSON_KEYS}
    payload["candidate_id"] = cell.candidate_id
    payload["tournament_id"] = tournament_id
    payload["cell"] = cell.to_dict()
    payload["generator_model"] = generator_model
    return IdeaCard.from_dict(payload)


# --------------------------------------------------------------------------
# Dedupe (spec sec. 2.5 - two keys, either one is sufficient)
# --------------------------------------------------------------------------


def structural_key(card: IdeaCard) -> tuple[str, str, str] | None:
    """``(resolved_target, method_family, dataset)``, or None if unknown.

    Returns None when the target cannot be resolved: two cards whose
    targets are both unknown are not thereby the same card, and killing
    them would be a false positive of exactly the kind the kill
    discipline forbids.
    """
    if not card.resolved_target:
        return None
    return (card.resolved_target, card.method_family, card.dataset or "")


def lexical_similarity(texts: Sequence[str]) -> list[list[float]]:
    """TF-IDF char 4-gram cosine matrix. Deterministic, no model."""
    n = len(texts)
    if n < 2:
        return [[1.0] * n for _ in range(n)]
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as exc:  # pragma: no cover - sklearn is a hard dep
        raise RuntimeError(
            "src.ideation.generate needs scikit-learn for dedupe; it is a "
            "declared project dependency (requirements.txt)."
        ) from exc
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 4))
    matrix = vectorizer.fit_transform(texts)  # L2-normalised rows
    gram = (matrix @ matrix.T).toarray()
    return [[float(value) for value in row] for row in gram]


def is_duplicate(
    a: IdeaCard,
    b: IdeaCard,
    *,
    cosine: float | None = None,
    threshold: float = DEFAULT_DEDUPE_COSINE,
) -> tuple[bool, str]:
    """``(duplicate, evidence)`` for one pair."""
    key_a, key_b = structural_key(a), structural_key(b)
    if key_a is not None and key_a == key_b:
        return True, (
            f"structural key equal: (resolved_target, method_family, "
            f"dataset) = {key_a}"
        )
    if cosine is not None and cosine >= threshold:
        return True, (
            f"TF-IDF char 4-gram cosine over the rendered cards = "
            f"{cosine:.3f} >= {threshold:.2f}"
        )
    return False, ""


def dedupe(
    cards: Sequence[IdeaCard],
    *,
    threshold: float = DEFAULT_DEDUPE_COSINE,
) -> tuple[list[IdeaCard], list[dict]]:
    """Keep the first card of each duplicate group, in slate order."""
    kept: list[IdeaCard] = []
    killed: list[dict] = []
    if not cards:
        return kept, killed
    similarity = lexical_similarity([card.render() for card in cards])
    index_of = {card.candidate_id: i for i, card in enumerate(cards)}
    for card in cards:
        duplicate_of: IdeaCard | None = None
        evidence = ""
        for survivor in kept:
            cosine = similarity[index_of[card.candidate_id]][
                index_of[survivor.candidate_id]
            ]
            flagged, why = is_duplicate(
                card, survivor, cosine=cosine, threshold=threshold
            )
            if flagged:
                duplicate_of, evidence = survivor, why
                break
        if duplicate_of is None:
            kept.append(card)
        else:
            killed.append(
                kill_record(
                    card,
                    stage="dedupe",
                    kill_code="D-DUPLICATE",
                    evidence=(
                        f"near-identical to {duplicate_of.candidate_id}; "
                        f"{evidence}"
                    ),
                    detail={"duplicate_of": duplicate_of.candidate_id},
                )
            )
    return kept, killed


# --------------------------------------------------------------------------
# Kill records - killed.jsonl is the training data for everything downstream
# --------------------------------------------------------------------------


def kill_record(
    card: IdeaCard | None,
    *,
    stage: str,
    kill_code: str,
    evidence: str,
    cell: SlateCell | None = None,
    detail: dict | None = None,
) -> dict:
    """One killed.jsonl line. Always carries a code AND its evidence."""
    resolved_cell = cell.to_dict() if cell is not None else (
        dict(card.cell) if card is not None else {}
    )
    candidate_id = (
        card.candidate_id
        if card is not None
        else (cell.candidate_id if cell is not None else "")
    )
    return {
        "candidate_id": candidate_id,
        "stage": stage,
        "kill_code": kill_code,
        "kill_codes": [kill_code],
        "evidence": evidence,
        "cell": resolved_cell,
        "card": card.to_dict() if card is not None else None,
        "detail": detail or {},
    }


# --------------------------------------------------------------------------
# The generation loop
# --------------------------------------------------------------------------


@dataclass
class GenerationResult:
    cards: list[IdeaCard] = field(default_factory=list)
    killed: list[dict] = field(default_factory=list)
    calls: int = 0
    parse_failures: int = 0
    call_failures: int = 0

    def summary(self) -> dict:
        return {
            "cards": len(self.cards),
            "killed": len(self.killed),
            "calls": self.calls,
            "parse_failures": self.parse_failures,
            "call_failures": self.call_failures,
        }


_REPAIR_SUFFIX = (
    "\n\n# Your previous reply could not be parsed\n"
    "Return ONLY a single JSON object with the keys listed above. No "
    "prose before or after it, no markdown fence, no trailing commentary."
)


def generate_cards(
    slate_obj: _slate.Slate,
    *,
    call_llm: LLMCaller,
    generator_model: str = "",
    registry_dir: str | os.PathLike[str] | None = None,
    registries: dict[str, dict] | None = None,
    exemplars: str | None = None,
    dedupe_cosine: float = DEFAULT_DEDUPE_COSINE,
    max_attempts: int = 2,
    on_event: Callable[[str], None] | None = None,
) -> GenerationResult:
    """One independent draw per slate cell, then dedupe.

    ``call_llm`` takes the user message and returns raw text. In
    production it is :func:`make_llm_caller`'s closure over
    ``BaseAgent.call_llm``; in tests it is a stub.
    """
    result = GenerationResult()
    drawn: list[IdeaCard] = []

    for cell in slate_obj.cells:
        registry = (registries or {}).get(cell.dataset)
        message = build_user_message(
            cell,
            registry_dir=registry_dir,
            registry=registry,
            exemplars=exemplars,
        )
        card: IdeaCard | None = None
        last_error = ""
        for attempt in range(max(1, max_attempts)):
            prompt = message if attempt == 0 else message + _REPAIR_SUFFIX
            try:
                result.calls += 1
                raw = call_llm(prompt)
            except Exception as exc:  # provider error, timeout, ...
                result.call_failures += 1
                last_error = f"{type(exc).__name__}: {exc}"
                continue
            try:
                data = parse_response(raw)
            except (ValueError, json.JSONDecodeError) as exc:
                result.parse_failures += 1
                last_error = f"{type(exc).__name__}: {exc}"
                continue
            card = card_from_response(
                data,
                cell,
                tournament_id=slate_obj.tournament_id,
                generator_model=generator_model,
            )
            break

        if card is None:
            result.killed.append(
                kill_record(
                    None,
                    cell=cell,
                    stage="generation",
                    kill_code="G-NO-CARD",
                    evidence=(
                        f"{max(1, max_attempts)} attempt(s) produced no "
                        f"parseable card; last error: {last_error}"
                    ),
                )
            )
            if on_event:
                on_event(f"{cell.candidate_id}: no card ({last_error})")
            continue
        if not card.research_question:
            result.killed.append(
                kill_record(
                    card,
                    stage="generation",
                    kill_code="G-EMPTY-QUESTION",
                    evidence=(
                        "the parsed card carries no research_question; there "
                        "is nothing to screen or judge"
                    ),
                )
            )
            if on_event:
                on_event(f"{cell.candidate_id}: empty research_question")
            continue
        drawn.append(card)
        if on_event:
            on_event(f"{cell.candidate_id}: drew a card")

    kept, duplicates = dedupe(drawn, threshold=dedupe_cosine)
    result.cards = kept
    result.killed.extend(duplicates)
    return result


# --------------------------------------------------------------------------
# Production LLM routing (BaseAgent only)
# --------------------------------------------------------------------------


@dataclass
class _GeneratorContext:
    """The minimal surface BaseAgent reads off a pipeline context."""

    dataset_name: str
    task_type: str = "prediction"
    output_dir: str | None = None
    revision_cycle: int = 0
    log: list = field(default_factory=list)


class _NoExecutor:
    """The generator never runs code; this refuses instead of building
    a Docker client at construction time."""

    def run(self, *args: Any, **kwargs: Any) -> dict:
        raise RuntimeError(
            "The idea generator does not execute code. If this was reached, "
            "something routed a sandbox job to the wrong agent."
        )


def resolve_generator_model(config: dict) -> str | None:
    """``ideation.models.generator`` if configured, else None.

    None means "leave BaseAgent's per-stage resolution alone", which
    reads ``config[<provider>].models.idea_generator`` and falls back to
    the provider default. No model ID is hardcoded here.
    """
    ideation = (config or {}).get("ideation") or {}
    model = ((ideation.get("models") or {}).get("generator"))
    return str(model) if model else None


def generation_temperature(config: dict) -> float:
    tournament = ((config or {}).get("ideation") or {}).get("tournament") or {}
    value = tournament.get("generation_temperature", GENERATION_TEMPERATURE)
    try:
        return float(value)
    except (TypeError, ValueError):
        return GENERATION_TEMPERATURE


class IdeaGeneratorAgent:
    """Thin BaseAgent wrapper. Constructed lazily so that importing this
    module never touches a provider SDK or an API key."""

    AGENT_NAME = AGENT_KEY

    def __init__(
        self,
        config: dict,
        *,
        dataset: str,
        task_type: str = "prediction",
        output_dir: str | None = None,
        model: str | None = None,
    ) -> None:
        from src.agents.base import BaseAgent

        class _Agent(BaseAgent):  # local subclass: BaseAgent.run is abstract
            def run(self, **kwargs: Any) -> dict:
                raise NotImplementedError(
                    "The idea generator is driven by "
                    "src.ideation.generate.generate_cards, not by a pipeline "
                    "stage runner."
                )

        context = _GeneratorContext(
            dataset_name=dataset, task_type=task_type, output_dir=output_dir
        )
        self.agent = _Agent(
            context,
            self.AGENT_NAME,
            config,
            executor=_NoExecutor(),
        )
        if not self.agent.system_prompt or self.agent.system_prompt.startswith(
            "You are the idea_generator agent"
        ):
            # agent_prompts/idea_generator.yaml is not on disk yet; see the
            # module constant for why the text lives there and not here.
            self.agent.system_prompt = FALLBACK_SYSTEM_PROMPT
        if model:
            self.agent.model = model
        self.model = self.agent.model

    def __call__(self, user_message: str, temperature: float) -> str:
        return self.agent.call_llm(
            user_message, temperature_override=temperature
        )


def make_llm_caller(
    config: dict,
    *,
    dataset: str,
    task_type: str = "prediction",
    output_dir: str | None = None,
) -> tuple[LLMCaller, str]:
    """``(caller, model_id)`` routed through BaseAgent.call_llm."""
    agent = IdeaGeneratorAgent(
        config,
        dataset=dataset,
        task_type=task_type,
        output_dir=output_dir,
        model=resolve_generator_model(config),
    )
    temperature = generation_temperature(config)

    def _call(user_message: str) -> str:
        return agent(user_message, temperature)

    return _call, agent.model


# --------------------------------------------------------------------------
# Offline template generator (no LLM) - for plumbing smoke tests
# --------------------------------------------------------------------------

OFFLINE_MODEL_ID = "offline-template-stub"


def offline_caller(
    *, registry_dir: str | os.PathLike[str] | None = None
) -> LLMCaller:
    """A deterministic stand-in for the model.

    Emits a schema-valid card with an EMPTY ``spec_draft``, so
    ``compile_spec`` exercises its registry-completion path end to end.
    Cards produced this way are marked ``generator_model =
    'offline-template-stub'`` and must never be mistaken for real draws.
    """

    def _call(user_message: str) -> str:
        cell = _cell_from_message(user_message)
        dataset = cell.get("dataset", "the dataset")
        task_type = cell.get("task_type", "prediction")
        pattern = cell.get("opportunity_pattern", "unspecified pattern")
        persona = cell.get("persona", "analyst")
        return json.dumps(
            {
                "research_question": (
                    f"Offline template draw: what does a {task_type} study "
                    f"framed as {pattern} on {dataset} show?"
                ),
                "why_it_matters": (
                    f"Placeholder written without a model so the {persona} "
                    f"cell can be exercised end to end offline."
                ),
                "what_we_would_do": (
                    "Run the registry-derived default specification for this "
                    "task type on this dataset."
                ),
                "what_counts_as_the_result": (
                    "Any estimate reported with an interval that excludes no "
                    "effect, or a clearly reported null."
                ),
                "method_family": _cards.METHOD_FAMILY_BY_TASK.get(task_type, ""),
                "second_contribution": None,
                "spec_draft": {},
            }
        )

    return _call


_ASSIGNMENT_LINE = re.compile(r"^([a-z_ ]+):\s*(.+)$")


def _cell_from_message(user_message: str) -> dict:
    """Recover the assignment block from a built user message."""
    out: dict[str, str] = {}
    for line in user_message.splitlines():
        match = _ASSIGNMENT_LINE.match(line.strip())
        if not match:
            continue
        key, value = match.group(1).strip(), match.group(2).strip()
        key = key.replace(" ", "_")
        if key in ("dataset", "task_type", "opportunity_pattern", "persona"):
            out.setdefault(key, value)
    return out


def registries_for(
    datasets: Iterable[str],
    registry_dir: str | os.PathLike[str] | None = None,
) -> dict[str, dict]:
    """Preload registries once so 24 draws do not re-read four YAMLs."""
    out: dict[str, dict] = {}
    for dataset in sorted(set(datasets)):
        registry, _path = _feas.load_registry(dataset, registry_dir)
        if registry:
            out[dataset] = registry
    return out
