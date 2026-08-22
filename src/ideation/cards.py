"""Arc T / T1a - the IdeaCard, its fixed-template render, and compile_spec.

Three things live here and nothing else:

1. :class:`IdeaCard` - the schema of Appendix A of
   `the v5 ideation-layer specification (internal)`, minus one field (see C1 below).
2. :meth:`IdeaCard.render` - a FIXED template with a hard word cap. The
   template is fixed and the cap is hard for one reason: length carries
   no signal about idea quality, and a judge that sees a longer card
   rewards it. Every card is rendered by the same code path, so no
   generator can win by being verbose.
3. :func:`compile_spec` - turn a card into a locked ``research_spec``
   dict that ``src.main.load_locked_research_spec()`` accepts unchanged.

C1 (no positive novelty anywhere)
---------------------------------
Appendix A of the spec lists ``novelty_score_self_assessment`` on the
card as "RECORDED, NEVER READ". This module does not record it either:
:func:`IdeaCard.from_dict` DROPS the key if a generator emits it, and
:func:`compile_spec` never writes it. Rationale - a stored number gets
read eventually, and the measured facts about that particular number
are damning (r = -0.35 against the LSAR Novelty score it was meant to
predict; 8 of 11 recorded values are the constant hard-coded in the
prompt template). Nothing downstream needs it: ``PredictionTemplate``
only warns when the field is present AND below 3, so omitting it is
strictly safer than recording it. Deviation from Appendix A, taken
deliberately under C1, reported at hand-off.

COMPLETION, NOT CORRECTION (the rule that keeps the screen honest)
------------------------------------------------------------------
``compile_spec`` fills fields that are ABSENT from the card's
``spec_draft``, using facts read out of the dataset registry. It never
replaces a value the generator supplied. If the model invented a
variable name, that name survives into the compiled spec and
``feasibility.screen()`` kills it - which is the entire point of having
a screen. A compile step that silently repaired drafts would make the
screen unfalsifiable and would hide exactly the failure mode T0 exists
to catch.

Every completion decision is appended to ``spec['compiled_by']
['completion_notes']`` with the registry fact it read (C2).
"""
from __future__ import annotations

import copy
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.ideation import feasibility as _feas

# --------------------------------------------------------------------------
# Template constants
# --------------------------------------------------------------------------

#: Hard cap on the rendered card, counted over the five labelled
#: sections (the bracketed provenance header is metadata and is not
#: counted). Enforced by truncation inside :meth:`IdeaCard.render`.
RENDER_WORD_CAP = 120

#: Per-field caps from Appendix A. They sum to exactly RENDER_WORD_CAP,
#: which leaves no room for the section labels or the feasibility line,
#: so :meth:`IdeaCard.render` re-allocates the real budget in these
#: proportions after reserving the fixed text. A field is therefore
#: capped at ``min(FIELD_CAPS[f], allocation)``.
FIELD_CAPS: dict[str, int] = {
    "research_question": 30,
    "why_it_matters": 35,
    "what_we_would_do": 35,
    "what_counts_as_the_result": 20,
}

_SECTION_LABELS: tuple[tuple[str, str], ...] = (
    ("research_question", "Question:"),
    ("why_it_matters", "Why it matters:"),
    ("what_we_would_do", "What we would do:"),
    ("what_counts_as_the_result", "What would count as the result:"),
)

#: task_type -> default method family. Values are the gap_miner method
#: families (src/gap_miner.py METHOD_FAMILIES) plus ``measurement``,
#: which gap_miner does not model.
METHOD_FAMILY_BY_TASK: dict[str, str] = {
    "prediction": "prediction_ml",
    "causal_soo": "causal_average",
    "causal_did": "causal_average",
    "causal_itr": "targeting_itr",
    "psychometrics": "measurement",
}

KNOWN_METHOD_FAMILIES: frozenset[str] = frozenset(
    set(METHOD_FAMILY_BY_TASK.values()) | {"fairness"}
)

#: The four second contributions VF-04 recognises (venue_fit.yaml).
SECOND_CONTRIBUTIONS: frozenset[str] = frozenset(
    {"transfer", "fairness", "measurement", "replication"}
)

#: Keys a generator must never set. ``novelty_score_self_assessment`` is
#: banned by C1; the rest are computed downstream and a model-supplied
#: value would be a fabrication.
BANNED_CARD_KEYS: frozenset[str] = frozenset(
    {
        "novelty_score_self_assessment",
        "novelty_score",
        "novelty",
        "score",
        "rank",
        "feasibility",
        "analytic_n",
    }
)

_WS = re.compile(r"\s+")


def _clean(text: object) -> str:
    if text is None:
        return ""
    return _WS.sub(" ", str(text)).strip()


def _words(text: str) -> list[str]:
    return text.split()


def _truncate_words(text: str, limit: int) -> str:
    words = _words(text)
    if limit <= 0:
        return ""
    if len(words) <= limit:
        return text
    return " ".join(words[:limit])


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def resolve_target(spec: dict) -> str | None:
    """Task-type-agnostic 'what is being studied' resolver (spec sec. 2.5).

    Mirrors ``feasibility._resolved_target`` deliberately rather than
    importing it: that name is private to a module this slice may not
    edit, and a rename there must not break card dedupe here. The order
    is the same and a test pins the two together.
    """
    outcome = spec.get("outcome")
    treatment = spec.get("treatment")
    for candidate in (
        spec.get("outcome_variable"),
        outcome.get("variable") if isinstance(outcome, dict) else outcome,
        treatment.get("variable") if isinstance(treatment, dict) else treatment,
        spec.get("scale_name"),
    ):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


# --------------------------------------------------------------------------
# IdeaCard
# --------------------------------------------------------------------------


@dataclass
class IdeaCard:
    """One candidate research idea, structurally pinned to a slate cell.

    ``cell`` is the slate assignment (dataset, task_type,
    opportunity_pattern, persona, gap_cell) and is authoritative: the
    generator may not move an idea to a different dataset or task type,
    because the slate is what makes diversity a structural property
    rather than a request made of a model.
    """

    candidate_id: str
    tournament_id: str
    cell: dict[str, Any]
    research_question: str = ""
    why_it_matters: str = ""
    what_we_would_do: str = ""
    what_counts_as_the_result: str = ""
    resolved_target: str | None = None
    method_family: str = ""
    second_contribution: str | None = None
    spec_draft: dict[str, Any] = field(default_factory=dict)
    generated_at: str = ""
    generator_model: str = ""
    notes: list[str] = field(default_factory=list)

    # --- cell accessors -------------------------------------------------
    @property
    def dataset(self) -> str | None:
        value = self.cell.get("dataset")
        return str(value) if value else None

    @property
    def task_type(self) -> str:
        return str(self.cell.get("task_type") or "prediction")

    @property
    def opportunity_pattern(self) -> str | None:
        value = self.cell.get("opportunity_pattern")
        return str(value) if value else None

    @property
    def persona(self) -> str | None:
        value = self.cell.get("persona")
        return str(value) if value else None

    # --- render ---------------------------------------------------------
    def header(self) -> str:
        parts = [
            self.task_type,
            self.dataset or "no-dataset",
            self.opportunity_pattern or "no-pattern",
        ]
        return f"[{self.candidate_id}] " + " | ".join(parts)

    def feasibility_line(self, feasibility: Any = None) -> str:
        """One deterministic line: measured n and the screen's verdict."""
        if feasibility is None:
            return "Feasibility: not screened."
        verdict = getattr(feasibility, "verdict", None)
        n_est = getattr(feasibility, "analytic_n_estimate", None)
        warns = getattr(feasibility, "warn_codes", None)
        if verdict is None and isinstance(feasibility, dict):
            verdict = feasibility.get("verdict")
            n_est = feasibility.get("analytic_n_estimate")
            checks = feasibility.get("checks") or []
            warns = [
                c.get("code")
                for c in checks
                if isinstance(c, dict) and c.get("status") == "WARN"
            ]
        n_text = f"n={n_est:,}" if isinstance(n_est, int) else "n=not probed"
        n_warn = len(warns or [])
        return (
            f"Feasibility: {n_text}, screen {verdict or 'UNKNOWN'}, "
            f"{n_warn} warning(s)."
        )

    def render(
        self,
        feasibility: Any = None,
        word_cap: int = RENDER_WORD_CAP,
    ) -> str:
        """Fixed-template render, hard-capped at ``word_cap`` words.

        The cap counts the four content sections plus the feasibility
        line, including their labels; the bracketed provenance header is
        excluded as metadata. The budget left after the fixed text is
        split between the four fields in FIELD_CAPS proportions, so a
        1,000-word draft and a 20-word draft produce cards of the same
        maximum length.

        Exact statement of the guarantee: the rendered body is at most
        ``word_cap`` words WHENEVER ``word_cap`` leaves room for the
        fixed text (14 label words plus the feasibility line, so about
        22 at the default). Below that the fixed text wins and one word
        per section is kept - the cap is a content budget, not a
        truncation of the template itself.
        """
        feas_line = self.feasibility_line(feasibility)
        label_words = sum(len(_words(label)) for _, label in _SECTION_LABELS)
        fixed = label_words + len(_words(feas_line))
        available = max(word_cap - fixed, len(_SECTION_LABELS))

        total_caps = sum(FIELD_CAPS.values())
        allocation: dict[str, int] = {}
        for name, cap in FIELD_CAPS.items():
            allocation[name] = max(1, (available * cap) // total_caps)
        # Hand the rounding remainder out in template order so the
        # allocation is a pure function of (word_cap, feasibility line).
        spare = available - sum(allocation.values())
        for name, _label in _SECTION_LABELS:
            if spare <= 0:
                break
            room = FIELD_CAPS[name] - allocation[name]
            take = min(spare, max(room, 0))
            allocation[name] += take
            spare -= take

        lines = [self.header()]
        for name, label in _SECTION_LABELS:
            body = _truncate_words(
                _clean(getattr(self, name)) or "(not stated)",
                min(FIELD_CAPS[name], allocation[name]),
            )
            lines.append(f"{label} {body}")
        lines.append(feas_line)
        return "\n".join(lines)

    def render_word_count(
        self, feasibility: Any = None, word_cap: int = RENDER_WORD_CAP
    ) -> int:
        body = self.render(feasibility, word_cap).split("\n")[1:]
        return sum(len(_words(line)) for line in body)

    # --- serialisation --------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "tournament_id": self.tournament_id,
            "cell": dict(self.cell),
            "research_question": self.research_question,
            "why_it_matters": self.why_it_matters,
            "what_we_would_do": self.what_we_would_do,
            "what_counts_as_the_result": self.what_counts_as_the_result,
            "resolved_target": self.resolved_target,
            "method_family": self.method_family,
            "second_contribution": self.second_contribution,
            "spec_draft": copy.deepcopy(self.spec_draft),
            "generated_at": self.generated_at,
            "generator_model": self.generator_model,
            "notes": list(self.notes),
            "render": self.render(),
            "render_word_count": self.render_word_count(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "IdeaCard":
        payload = dict(data or {})
        dropped = sorted(k for k in payload if k in BANNED_CARD_KEYS)
        for key in dropped:
            payload.pop(key, None)
        spec_draft = payload.get("spec_draft")
        spec_draft = copy.deepcopy(spec_draft) if isinstance(spec_draft, dict) else {}
        for key in list(spec_draft):
            if key in BANNED_CARD_KEYS:
                spec_draft.pop(key, None)
                if f"spec_draft.{key}" not in dropped:
                    dropped.append(f"spec_draft.{key}")
        notes = [str(n) for n in (payload.get("notes") or [])]
        if dropped:
            notes.append(
                "C1: dropped banned key(s) supplied by the generator: "
                + ", ".join(dropped)
            )
        cell = payload.get("cell")
        card = cls(
            candidate_id=str(payload.get("candidate_id") or ""),
            tournament_id=str(payload.get("tournament_id") or ""),
            cell=dict(cell) if isinstance(cell, dict) else {},
            research_question=_clean(payload.get("research_question")),
            why_it_matters=_clean(payload.get("why_it_matters")),
            what_we_would_do=_clean(payload.get("what_we_would_do")),
            what_counts_as_the_result=_clean(
                payload.get("what_counts_as_the_result")
            ),
            resolved_target=(
                _clean(payload.get("resolved_target")) or None
            ),
            method_family=_clean(payload.get("method_family")),
            second_contribution=(
                _clean(payload.get("second_contribution")) or None
            ),
            spec_draft=spec_draft,
            generated_at=str(payload.get("generated_at") or ""),
            generator_model=str(payload.get("generator_model") or ""),
            notes=notes,
        )
        card.normalize()
        return card

    def normalize(self) -> None:
        """Deterministic post-parse cleanup. Idempotent."""
        if not self.generated_at:
            self.generated_at = _utc_now()
        family = self.method_family.strip().lower().replace(" ", "_")
        if family not in KNOWN_METHOD_FAMILIES:
            family = METHOD_FAMILY_BY_TASK.get(self.task_type, "prediction_ml")
        self.method_family = family
        second = (self.second_contribution or "").strip().lower()
        self.second_contribution = second if second in SECOND_CONTRIBUTIONS else None
        if not self.resolved_target:
            self.resolved_target = resolve_target(self.spec_draft)
        # The slate owns dataset/task_type. A generator that "moved" the
        # idea does not get to; the cell is copied back over the draft.
        if self.dataset:
            self.spec_draft.setdefault("dataset", self.dataset)
        self.spec_draft["task_type"] = self.task_type
        self.spec_draft["dataset"] = self.dataset or self.spec_draft.get("dataset")


# --------------------------------------------------------------------------
# Registry facts used by compile_spec's completion pass
# --------------------------------------------------------------------------


@dataclass
class _Facts:
    dataset: str | None
    registry: dict
    registry_path: str | None
    temporal_order: list[str]
    outcomes: list[dict]
    predictors: list[dict]  # each carries an injected "_category"
    var_map: dict[str, dict]
    item_banks: dict
    cdm_support: dict
    protected: list[str]

    def wave_index(self, meta: dict | None) -> int | None:
        wave = (meta or {}).get("wave")
        if wave in self.temporal_order:
            return self.temporal_order.index(str(wave))
        return None

    @property
    def source(self) -> str:
        return self.registry_path or f"{self.dataset}.yaml (not on disk)"


#: Preference order over registry predictor categories when compile_spec
#: has to invent a treatment. Attitudinal scales first: they are the
#: manipulable-ish constructs the archive's causal specs actually used
#: (X1MTHEFF in 3 shipped specs), and an achievement score as a
#: median-split "treatment" is the least defensible default available.
_TREATMENT_CATEGORY_PREFERENCE: tuple[str, ...] = (
    "math_attitudes",
    "noncognitive",
    "academic",
    "covariates_v2",
    "family",
    "interaction",
    "demographic",
)

_DEAD_PCT_MISSING = 99.0


def _facts(
    dataset: str | None,
    registry: dict | None = None,
    registry_dir: str | os.PathLike[str] | None = None,
) -> _Facts:
    reg: dict = registry or {}
    reg_path: str | None = None
    if not reg and dataset:
        reg, reg_path = _feas.load_registry(dataset, registry_dir)
    elif dataset:
        reg_path = f"{dataset}.yaml (caller-supplied registry)"

    variables = (reg.get("variables") or {}) if isinstance(reg, dict) else {}
    outcomes = [o for o in (variables.get("outcomes") or []) if isinstance(o, dict)]
    predictors: list[dict] = []
    raw_predictors = variables.get("predictors") or {}
    if isinstance(raw_predictors, dict):
        for category, var_list in raw_predictors.items():
            for var in var_list or []:
                if isinstance(var, dict) and var.get("name"):
                    enriched = dict(var)
                    enriched["_category"] = str(category)
                    predictors.append(enriched)

    var_map = _feas.build_var_map(reg)
    protected = sorted(
        str(name)
        for name, meta in var_map.items()
        if isinstance(meta, dict) and meta.get("protected_attribute")
    )
    return _Facts(
        dataset=dataset,
        registry=reg,
        registry_path=reg_path,
        temporal_order=[str(w) for w in (reg.get("temporal_order") or [])],
        outcomes=outcomes,
        predictors=predictors,
        var_map=var_map,
        item_banks=(reg.get("item_banks") or {}) if isinstance(reg, dict) else {},
        cdm_support=(reg.get("cdm_support") or {}) if isinstance(reg, dict) else {},
        protected=protected,
    )


def pct_missing(meta: dict) -> float:
    """Registry ``pct_missing`` as a float; unknown sorts last (100.0)."""
    value = (meta or {}).get("pct_missing")
    if isinstance(value, (int, float)):
        return float(value)
    return 100.0


def _usable(meta: dict) -> bool:
    """Skip variables the screen would kill on sight."""
    if meta.get("derived"):  # e.g. dropout_derived: not a CSV column
        return False
    pct = meta.get("pct_missing")
    if isinstance(pct, (int, float)) and pct >= _DEAD_PCT_MISSING:
        return False
    if str(meta.get("type") or "").lower() == "id":
        return False
    return True


def _pick_outcome(
    facts: _Facts,
    *,
    prefer_type: str | None = None,
    min_wave_index: int = 1,
) -> dict | None:
    def _sortable(meta: dict) -> tuple:
        return (
            0 if prefer_type and meta.get("type") == prefer_type else 1,
            pct_missing(meta),
            str(meta.get("name")),
        )

    for floor in (min_wave_index, 0):
        pool = [
            o
            for o in facts.outcomes
            if _usable(o)
            and (facts.wave_index(o) is not None)
            and (facts.wave_index(o) or 0) >= floor
        ]
        if pool:
            return sorted(pool, key=_sortable)[0]
    return None


def _earlier_predictors(
    facts: _Facts,
    outcome_index: int | None,
    *,
    exclude: set[str] | None = None,
    limit: int = 8,
) -> list[dict]:
    exclude = exclude or set()

    def _pool(strict: bool) -> list[dict]:
        out: list[dict] = []
        for meta in facts.predictors:
            if str(meta.get("name")) in exclude or not _usable(meta):
                continue
            idx = facts.wave_index(meta)
            if idx is None:
                continue
            if outcome_index is not None:
                if strict and idx >= outcome_index:
                    continue
                if not strict and idx > outcome_index:
                    continue
            out.append(meta)
        return out

    pool = _pool(strict=True) or _pool(strict=False)
    pool.sort(key=lambda m: (pct_missing(m), str(m.get("name"))))
    return pool[:limit]


def _pick_treatment(facts: _Facts) -> dict | None:
    pool = [
        m
        for m in facts.predictors
        if _usable(m)
        and str(m.get("type") or "").lower() == "continuous"
        and facts.wave_index(m) is not None
    ]
    if not pool:
        pool = [m for m in facts.predictors if _usable(m)]
    if not pool:
        return None

    def _rank(meta: dict) -> tuple:
        category = str(meta.get("_category") or "")
        try:
            cat_rank = _TREATMENT_CATEGORY_PREFERENCE.index(category)
        except ValueError:
            cat_rank = len(_TREATMENT_CATEGORY_PREFERENCE)
        return (
            facts.wave_index(meta) if facts.wave_index(meta) is not None else 99,
            cat_rank,
            pct_missing(meta),
            str(meta.get("name")),
        )

    return sorted(pool, key=_rank)[0]


# --------------------------------------------------------------------------
# compile_spec
# --------------------------------------------------------------------------


def compile_spec(
    card: IdeaCard,
    feasibility: Any = None,
    *,
    registry: dict | None = None,
    registry_dir: str | os.PathLike[str] | None = None,
    delta_sentence: str | None = None,
) -> dict:
    """Emit a locked ``research_spec`` dict for ``card``.

    The output must pass ``src.main.load_locked_research_spec()``
    unchanged; ``tests/test_t1a_generation.py`` asserts that through the
    real loader for every task type this repo can execute.

    SEAM NOTE (from the T0 hand-off): real archived prediction specs
    carry neither ``task_type`` nor ``dataset`` and the loader has a
    hard guard on ``task_type``. Both are written here, unconditionally,
    from the slate cell.

    ``feasibility`` is optional; when a FeasibilityReport is supplied its
    measured ``analytic_n_estimate`` is carried into
    ``expected_contribution`` so the downstream pipeline sees the number
    the screen actually measured rather than a model's guess.
    """
    spec: dict[str, Any] = copy.deepcopy(card.spec_draft) if card.spec_draft else {}
    for key in list(spec):
        if key in BANNED_CARD_KEYS:
            spec.pop(key, None)

    notes: list[str] = []
    facts = _facts(card.dataset, registry, registry_dir)

    # --- fields the seam requires, always written -----------------------
    spec["task_type"] = card.task_type
    spec["dataset"] = card.dataset
    if not spec.get("task_id"):
        spec["task_id"] = _task_id(card)
    if card.research_question:
        spec["research_question"] = card.research_question

    analytic_n = getattr(feasibility, "analytic_n_estimate", None)
    if analytic_n is None and isinstance(feasibility, dict):
        analytic_n = feasibility.get("analytic_n_estimate")
    spec["expected_contribution"] = _expected_contribution(
        card, analytic_n, delta_sentence, spec.get("expected_contribution")
    )

    completer = _COMPLETERS.get(card.task_type)
    if completer is not None:
        completer(spec, card, facts, notes)
    else:
        _note(
            notes,
            f"no completer for task_type {card.task_type!r}; the draft was "
            f"passed through untouched",
            f"src.ideation.cards._COMPLETERS covers "
            f"{sorted(_COMPLETERS)}",
        )

    spec["compiled_by"] = {
        "module": "src.ideation.cards.compile_spec",
        "tournament_id": card.tournament_id,
        "candidate_id": card.candidate_id,
        "cell": dict(card.cell),
        "registry_source": facts.source,
        "completion_notes": notes,
    }
    return spec


def _task_id(card: IdeaCard) -> str:
    raw = f"{card.tournament_id}_{card.candidate_id}".strip("_") or "idea"
    return re.sub(r"[^0-9a-zA-Z_]+", "_", raw).lower()


def _expected_contribution(
    card: IdeaCard,
    analytic_n: int | None,
    delta_sentence: str | None,
    drafted: object = None,
) -> str:
    """Card prose plus the measured facts; a drafted value is kept first."""
    parts: list[str] = []
    if isinstance(drafted, str) and drafted.strip():
        parts.append(_clean(drafted))
    if card.why_it_matters:
        parts.append(_clean(card.why_it_matters))
    if card.what_counts_as_the_result:
        parts.append(
            "What would count as the result: "
            + _clean(card.what_counts_as_the_result)
        )
    if delta_sentence:
        parts.append("Delta vs nearest prior work: " + _clean(delta_sentence))
    if isinstance(analytic_n, int):
        parts.append(
            f"Measured analytic n from the deterministic screen: {analytic_n:,}."
        )
    return " ".join(parts)


def _note(notes: list[str], what: str, evidence: str) -> None:
    notes.append(f"{what} [read: {evidence}]")


# --- prediction ------------------------------------------------------------


def _complete_prediction(
    spec: dict, card: IdeaCard, facts: _Facts, notes: list[str]
) -> None:
    outcome_name = spec.get("outcome_variable")
    if not outcome_name:
        picked = _pick_outcome(facts, min_wave_index=1)
        if picked is not None:
            outcome_name = str(picked["name"])
            spec["outcome_variable"] = outcome_name
            _note(
                notes,
                f"outcome_variable absent -> {outcome_name}",
                f"{facts.source} variables.outcomes (lowest pct_missing at a "
                f"wave later than base; pct_missing="
                f"{picked.get('pct_missing')})",
            )
    outcome_meta = facts.var_map.get(str(outcome_name or ""), {})
    if not spec.get("outcome_type") and outcome_meta.get("type"):
        spec["outcome_type"] = str(outcome_meta["type"])
        _note(
            notes,
            f"outcome_type absent -> {spec['outcome_type']}",
            f"{facts.source} {outcome_name}.type",
        )
    outcome_index = facts.wave_index(outcome_meta)

    predictors = spec.get("predictor_set")
    if isinstance(predictors, list) and predictors:
        normalised: list[dict] = []
        for entry in predictors:
            if isinstance(entry, str):
                entry = {"variable": entry}
            if not isinstance(entry, dict) or not entry.get("variable"):
                continue
            name = str(entry["variable"])
            meta = facts.var_map.get(name, {})
            if not entry.get("wave") and meta.get("wave"):
                entry["wave"] = str(meta["wave"])
                _note(
                    notes,
                    f"predictor {name} wave absent -> {entry['wave']}",
                    f"{facts.source} {name}.wave",
                )
            if not entry.get("rationale"):
                entry["rationale"] = (
                    f"Named by the generator for candidate {card.candidate_id}; "
                    f"no rationale supplied."
                )
            normalised.append(entry)
        spec["predictor_set"] = normalised
    else:
        derived = _earlier_predictors(
            facts, outcome_index, exclude={str(outcome_name or "")}, limit=8
        )
        spec["predictor_set"] = [
            {
                "variable": str(meta["name"]),
                "rationale": (
                    f"Registry-curated {meta.get('_category')} variable "
                    f"(pct_missing={meta.get('pct_missing')}) at wave "
                    f"{meta.get('wave')}, which precedes the outcome wave "
                    f"{outcome_meta.get('wave')}."
                ),
                "wave": str(meta.get("wave")),
            }
            for meta in derived
        ]
        _note(
            notes,
            f"predictor_set absent -> {len(derived)} registry-curated "
            f"predictors",
            f"{facts.source} variables.predictors, waves strictly before "
            f"{outcome_meta.get('wave')}",
        )

    if not spec.get("subgroup_analyses") and facts.protected:
        if card.opportunity_pattern == "equity_subgroup_gap":
            spec["subgroup_analyses"] = facts.protected[:2]
            _note(
                notes,
                f"subgroup_analyses absent on an equity card -> "
                f"{spec['subgroup_analyses']}",
                f"{facts.source} protected_attribute: true",
            )


# --- causal_soo / causal_itr ----------------------------------------------


def _complete_causal_soo(
    spec: dict, card: IdeaCard, facts: _Facts, notes: list[str]
) -> None:
    treatment = spec.get("treatment")
    if isinstance(treatment, str):
        treatment = {"variable": treatment}
    if not isinstance(treatment, dict) or not treatment.get("variable"):
        picked = _pick_treatment(facts)
        if picked is not None:
            treatment = {
                "variable": str(picked["name"]),
                "operationalization": "median_split_binary",
                "rationale_for_PF": (
                    "Registry-derived fallback: no treatment was named on the "
                    "idea card. A median split of a continuous scale is a "
                    "known weak operationalization (ESC-07) and PF must "
                    "either defend or replace it."
                ),
            }
            _note(
                notes,
                f"treatment absent -> {treatment['variable']}",
                f"{facts.source} variables.predictors "
                f"(category={picked.get('_category')}, "
                f"pct_missing={picked.get('pct_missing')})",
            )
    if isinstance(treatment, dict):
        if not treatment.get("operationalization"):
            treatment["operationalization"] = "median_split_binary"
            _note(
                notes,
                "treatment.operationalization absent -> median_split_binary",
                "src/task_template.py CausalSOOTemplate requires the field",
            )
        spec["treatment"] = treatment

    treatment_meta = facts.var_map.get(
        str((spec.get("treatment") or {}).get("variable") or ""), {}
    )
    treatment_index = facts.wave_index(treatment_meta)

    outcome = spec.get("outcome")
    if isinstance(outcome, str):
        outcome = {"variable": outcome}
    if not isinstance(outcome, dict) or not outcome.get("variable"):
        floor = (treatment_index + 1) if treatment_index is not None else 1
        picked = _pick_outcome(facts, prefer_type="binary", min_wave_index=floor)
        if picked is not None:
            outcome = {
                "variable": str(picked["name"]),
                "type": str(picked.get("type") or "binary"),
                "definition": str(picked.get("label") or picked["name"]),
            }
            _note(
                notes,
                f"outcome absent -> {outcome['variable']}",
                f"{facts.source} variables.outcomes at a wave after the "
                f"treatment wave ({treatment_meta.get('wave')})",
            )
    if isinstance(outcome, dict):
        if not outcome.get("type"):
            meta = facts.var_map.get(str(outcome.get("variable") or ""), {})
            outcome["type"] = str(meta.get("type") or "binary")
            _note(
                notes,
                f"outcome.type absent -> {outcome['type']}",
                f"{facts.source} {outcome.get('variable')}.type",
            )
        spec["outcome"] = outcome

    if not spec.get("target_estimand_hint"):
        spec["target_estimand_hint"] = (
            "ATT preferred (effect on the treated); PF must declare the "
            "estimand explicitly per G2."
        )
        _note(
            notes,
            "target_estimand_hint absent -> ATT default",
            "src/task_template.py CausalSOOTemplate requires the field",
        )

    # Read the certified set for THIS card's task type: causal_itr
    # inherits the SOO completer but certifies M6/M7 as well.
    #
    # An UNCERTIFIED method that the generator actually named is left
    # exactly as it is. Rewriting it to a certified default would hide
    # the infeasibility that F-ESTIMATOR-UNCERTIFIED exists to kill on,
    # and would silently turn an RD proposal into a PSM study.
    supported = _supported_methods(card.task_type)
    if not str(spec.get("primary_method") or "").strip():
        spec["primary_method"] = "M2"
        _note(
            notes,
            "primary_method absent -> M2 (PSM)",
            f"src/task_template.py {card.task_type} SUPPORTED_METHODS="
            f"{sorted(supported)}",
        )
    if not spec.get("comparator_method"):
        spec["comparator_method"] = "M1"
        _note(
            notes,
            "comparator_method absent -> M1 (regression adjustment)",
            "M1 is the always-reported comparator in every shipped causal spec",
        )
    primary = str(spec.get("primary_method") or "").upper()
    if not spec.get("secondary_methods"):
        spec["secondary_methods"] = [
            m for m in ("M3", "M4", "M1") if m != primary
        ][:1]
        _note(
            notes,
            f"secondary_methods absent -> {spec['secondary_methods']}",
            f"src/task_template.py {card.task_type} SUPPORTED_METHODS="
            f"{sorted(supported)}",
        )

    outcome_meta = facts.var_map.get(
        str((spec.get("outcome") or {}).get("variable") or ""), {}
    )
    outcome_index = facts.wave_index(outcome_meta)
    if not spec.get("adjustment_set"):
        derived = _earlier_predictors(
            facts,
            outcome_index,
            exclude={str((spec.get("treatment") or {}).get("variable") or "")},
            limit=6,
        )
        spec["adjustment_set"] = [str(m["name"]) for m in derived]
        _note(
            notes,
            f"adjustment_set absent -> {len(derived)} registry-curated "
            f"covariates",
            f"{facts.source} variables.predictors at waves at or before the "
            f"outcome wave, treatment excluded",
        )


def _complete_causal_itr(
    spec: dict, card: IdeaCard, facts: _Facts, notes: list[str]
) -> None:
    _complete_causal_soo(spec, card, facts, notes)
    # The SOO completer's default (M2) is a completion, not a choice the
    # generator made, so promoting it to M6 here is still completion. A
    # method the generator DID name is left alone - including a wrong
    # one, which the screen then kills.
    if spec.get("primary_method") == "M2" and any(
        "primary_method absent" in note for note in notes
    ):
        spec["primary_method"] = "M6"
        spec["secondary_methods"] = [
            m
            for m in (spec.get("secondary_methods") or [])
            if str(m).upper() != "M6"
        ] or ["M1"]
        _note(
            notes,
            "primary_method default -> M6 (policy learning)",
            "src/task_template.py CausalITRTemplate requires primary_method M6",
        )
    adjustment = [str(v) for v in (spec.get("adjustment_set") or [])]
    rule_covs = [str(v) for v in (spec.get("rule_covariates") or [])]
    if not rule_covs:
        rule_covs = adjustment[:3]
        _note(
            notes,
            f"rule_covariates absent -> {rule_covs}",
            "first three adjustment-set covariates; all are observable at "
            "decision time by construction",
        )
    extra = [c for c in rule_covs if c not in adjustment]
    if extra:
        adjustment = adjustment + extra
        _note(
            notes,
            f"adjustment_set extended with rule covariates {extra}",
            "src/task_template.py CausalITRTemplate requires "
            "rule_covariates to be a subset of adjustment_set",
        )
    spec["adjustment_set"] = adjustment
    spec["rule_covariates"] = rule_covs


# --- causal_did ------------------------------------------------------------


def _complete_causal_did(
    spec: dict, card: IdeaCard, facts: _Facts, notes: list[str]
) -> None:
    design = (facts.registry.get("design_feasibility") or {}) if facts.registry else {}
    timing = [str(v) for v in (design.get("policy_timing_variables") or [])]
    if not spec.get("post_variable") and timing:
        spec["post_variable"] = timing[0]
        _note(
            notes,
            f"post_variable absent -> {timing[0]}",
            f"{facts.source} design_feasibility.policy_timing_variables",
        )
    if not spec.get("group_variable"):
        post = str(spec.get("post_variable") or "")
        binaries = [
            m
            for m in facts.predictors
            if _usable(m)
            and str(m.get("type") or "").lower() == "binary"
            and str(m.get("name")) != post
        ]
        binaries.sort(
            key=lambda m: (
                0 if m.get("protected_attribute") else 1,
                str(m.get("name")),
            )
        )
        if binaries:
            spec["group_variable"] = str(binaries[0]["name"])
            _note(
                notes,
                f"group_variable absent -> {spec['group_variable']}",
                f"{facts.source} first binary predictor carrying "
                f"protected_attribute: true",
            )

    outcome = spec.get("outcome")
    if isinstance(outcome, str):
        outcome = {"variable": outcome}
    if not isinstance(outcome, dict) or not outcome.get("variable"):
        picked = _pick_outcome(facts, prefer_type="continuous", min_wave_index=0)
        if picked is not None:
            outcome = {
                "variable": str(picked["name"]),
                "type": str(picked.get("type") or "continuous"),
                "definition": str(picked.get("label") or picked["name"]),
            }
            _note(
                notes,
                f"outcome absent -> {outcome['variable']}",
                f"{facts.source} variables.outcomes (lowest pct_missing)",
            )
    if isinstance(outcome, dict):
        spec["outcome"] = outcome

    if not str(spec.get("primary_method") or "").strip():
        spec["primary_method"] = "M8"
        _note(
            notes,
            "primary_method absent -> M8 (raw gap-in-gaps)",
            "src/task_template.py CausalDIDTemplate accepts M8 or M9 only",
        )


# --- psychometrics ---------------------------------------------------------


def _complete_psychometrics(
    spec: dict, card: IdeaCard, facts: _Facts, notes: list[str]
) -> None:
    items = [i for i in (spec.get("item_columns") or []) if isinstance(i, str)]
    has_construction = bool(spec.get("item_construction"))

    if not items and not has_construction:
        bank_name, bank = _pick_item_bank(facts)
        if bank is not None and bank_name is not None:
            items = [str(i) for i in (bank.get("items") or [])]
            spec["item_columns"] = items
            spec.setdefault("scale_name", f"{facts.dataset} {bank_name}")
            if bank.get("response_labels"):
                spec.setdefault("response_labels", bank["response_labels"])
            if bank.get("reverse"):
                spec.setdefault(
                    "reverse_items", [str(i) for i in bank["reverse"]]
                )
            _note(
                notes,
                f"item_columns absent -> item bank {bank_name!r} "
                f"({len(items)} items)",
                f"{facts.source} item_banks.{bank_name}.items",
            )
        elif facts.cdm_support:
            spec["item_construction"] = {
                "unit": str(facts.cdm_support.get("item_unit") or "template_id"),
                "response": str(
                    facts.cdm_support.get("response_rule") or "first-attempt correct"
                ),
                "q_matrix": str(facts.cdm_support.get("q_matrix_source") or ""),
                "scope": str(facts.cdm_support.get("recommended_scope") or ""),
            }
            has_construction = True
            spec.setdefault(
                "scale_name", f"{facts.dataset} skill battery (log-derived)"
            )
            _note(
                notes,
                "item_columns absent, item_construction built from cdm_support",
                f"{facts.source} cdm_support.recommended_scope",
            )

    if not spec.get("scale_name"):
        spec["scale_name"] = f"{facts.dataset} measurement scale"
        _note(
            notes,
            "scale_name absent -> dataset-derived placeholder",
            "src/task_template.py PsychometricsTemplate requires scale_name",
        )

    if len(items) >= 3 and not spec.get("factor_model"):
        spec["factor_model"] = "F1 =~ " + " + ".join(items)
        _note(
            notes,
            f"factor_model absent -> single factor over {len(items)} items",
            f"{facts.source} item bank membership",
        )

    # As with the causal completers: a battery the generator supplied is
    # kept verbatim, uncertified IDs and all, so the screen can kill it.
    battery = [str(m).upper() for m in (spec.get("method_battery") or [])]
    if not battery:
        if has_construction and not items:
            battery = ["P1", "P7"]
            reason = "log-derived items: classical item analysis + CDM"
        elif len(items) >= 3:
            battery = ["P1", "P2", "P3"]
            reason = "survey scale with >= 3 items: CTT, omega, CFA"
        else:
            battery = ["P1"]
            reason = (
                f"only {len(items)} item(s) resolved: classical item "
                f"analysis only, no factor model"
            )
        _note(notes, f"method_battery absent -> {battery}", reason)
        spec["method_battery"] = battery
    else:
        spec["method_battery"] = battery

    needs_groups = any(m in spec["method_battery"] for m in ("P5", "P6"))
    if needs_groups and not spec.get("grouping_vars"):
        if facts.protected:
            spec["grouping_vars"] = facts.protected[:1]
            _note(
                notes,
                f"grouping_vars absent for {['P5', 'P6']} -> "
                f"{spec['grouping_vars']}",
                f"{facts.source} protected_attribute: true",
            )
        else:
            _note(
                notes,
                "P5/P6 requested but the dataset declares no protected "
                "attribute; grouping_vars left empty and the spec will not "
                "load. NOT auto-corrected: the card is infeasible and must "
                "say so.",
                f"{facts.source} declares no protected_attribute",
            )
    if (
        card.opportunity_pattern == "equity_subgroup_gap"
        and facts.protected
        and not spec.get("grouping_vars")
    ):
        spec["grouping_vars"] = facts.protected[:1]
        _note(
            notes,
            f"equity card -> grouping_vars {spec['grouping_vars']}",
            f"{facts.source} protected_attribute: true",
        )


def _pick_item_bank(facts: _Facts) -> tuple[str | None, dict | None]:
    banks = [
        (str(name), bank)
        for name, bank in (facts.item_banks or {}).items()
        if isinstance(bank, dict) and bank.get("items")
    ]
    if not banks:
        return None, None
    banks.sort(key=lambda pair: (-len(pair[1].get("items") or []), pair[0]))
    return banks[0]


def _supported_methods(task_type: str) -> frozenset[str]:
    """Certified method IDs, read from the task templates themselves."""
    try:
        from src.task_template import create_task_template

        template = create_task_template(task_type)
        methods = getattr(template, "SUPPORTED_METHODS", None)
        if methods:
            return frozenset(str(m).upper() for m in methods)
    except Exception:
        pass
    return {
        "causal_soo": frozenset({"M1", "M2", "M3", "M4", "M5"}),
        "causal_itr": frozenset({"M1", "M2", "M3", "M4", "M5", "M6", "M7"}),
        "causal_did": frozenset({"M8", "M9", "M10"}),
        "psychometrics": frozenset({"P1", "P2", "P3", "P4", "P5", "P6", "P7"}),
    }.get(task_type, frozenset())


_COMPLETERS: dict[str, Any] = {
    "prediction": _complete_prediction,
    "causal_soo": _complete_causal_soo,
    "causal_itr": _complete_causal_itr,
    "causal_did": _complete_causal_did,
    "psychometrics": _complete_psychometrics,
}
