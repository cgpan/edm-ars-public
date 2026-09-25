"""The guided "new study" flow (CLI_SPEC section 14).

Turns a researcher's intent into a :class:`~edmars.model.StudyPlan` the
runner can launch, and refuses the plans the pipeline cannot run before
any money is spent:

* Prediction studies start from a free-text question; the pipeline's
  ProblemFormulator turns it into a research spec.
* Cause-and-effect and measurement studies start from a tested example
  spec in ``runs/fixtures/`` or from registry-driven menus ("Build my
  own"), which are clearly labelled EXPERIMENTAL. A causal or
  measurement type with no spec is never launched: the pipeline would
  silently run a prediction-shaped study in causal clothing.
* The dataset always comes from the spec, never from a separate flag,
  so the spec is validated against the registry of the data the run
  actually loads.
* :func:`preflight` runs the deterministic feasibility screen
  (``src.ideation.feasibility.screen``), the pipeline's own locked-spec
  loader, and checks that the data file (and R, for measurement models)
  is on this computer.

Everything the flow knows about datasets and variables is read from the
registries under ``data_registry/datasets``; no model is called and no
network request is made. Nothing here spawns a process.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib
import io
import json
import os
import re
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Sequence

from edmars import paths
from edmars.model import Check, StudyPlan

# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------

TASK_TYPES: tuple[str, ...] = (
    "prediction",
    "causal_soo",
    "causal_itr",
    "causal_did",
    "psychometrics",
)
#: Task types a "Build my own" menu can produce.
MENU_TASK_TYPES: tuple[str, ...] = (
    "causal_soo",
    "causal_itr",
    "causal_did",
    "psychometrics",
)
DEFAULT_DATASET = "hsls09_public"

TASK_LABELS: dict[str, str] = {
    "prediction": "Prediction",
    "causal_soo": "Cause and effect: the average effect",
    "causal_itr": "Cause and effect: for whom it works",
    "causal_did": "Cause and effect: did a gap change between two cohorts",
    "psychometrics": "Measurement",
}

TASK_BLURBS: dict[str, str] = {
    "prediction": "machine-learning models that predict an outcome and show "
    "which factors matter",
    "causal_soo": "compares similar students who differ in one thing, to "
    "estimate its average effect",
    "causal_itr": "estimates who would benefit most, and tests a simple rule "
    "for targeting",
    "causal_did": "compares how a gap between two groups changed between two "
    "cohorts of students",
    "psychometrics": "checks how well survey items measure what they are meant "
    "to, and whether they work the same for different groups",
}

DATASET_LABELS: dict[str, str] = {
    "hsls09_public": "HSLS:09 - U.S. students followed from 9th grade (2009) "
    "into college and work",
    "els_2002": "ELS:2002 - U.S. students followed from 10th grade (2002) into "
    "college and work",
    "did_els_hsls_panel": "ELS:2002 + HSLS:09 combined - compares the two "
    "student cohorts",
    "assistments_0910": "ASSISTments 2009-10 - middle-school math practice logs",
}

DATASET_SHORT: dict[str, str] = {
    "hsls09_public": "HSLS:09",
    "els_2002": "ELS:2002",
    "did_els_hsls_panel": "ELS:2002 + HSLS:09 panel",
    "assistments_0910": "ASSISTments 2009-10",
}

#: Plain-language reasons for the dataset x task cells the feasibility
#: module's DATASET_TASK_MATRIX marks unsupported. The matrix is the
#: authority on WHICH cells are unsupported; this only words the reason.
_PLAIN_UNSUPPORTED: dict[tuple[str, str], str] = {
    ("hsls09_public", "causal_did"): "HSLS:09 is a single group of students "
    "(2009), so there is no second cohort to compare. The combined ELS:2002 + "
    "HSLS:09 data supports this kind of study.",
    ("els_2002", "causal_did"): "ELS:2002 is a single group of students "
    "(2002), so there is no second cohort to compare. The combined ELS:2002 + "
    "HSLS:09 data supports this kind of study.",
    ("assistments_0910", "causal_did"): "the ASSISTments log covers a single "
    "school year, with no second cohort.",
    ("assistments_0910", "causal_itr"): "the ASSISTments log has no student "
    "background information to base a 'for whom' rule on.",
    ("did_els_hsls_panel", "causal_itr"): "the combined panel carries only a "
    "few harmonized background variables, built for the cohort comparison.",
    ("did_els_hsls_panel", "psychometrics"): "the combined panel has no "
    "survey item responses, only summary scores.",
}

#: What to run when a dataset's file is not on this computer yet.
_INSTALL_HINTS: dict[str, str] = {
    "hsls09_public": "edmars data install hsls09_public",
    "els_2002": "edmars data install els_2002",
    "did_els_hsls_panel": "edmars data install did_els_hsls_panel   (it is "
    "built from ELS:2002 and HSLS:09, so install those two first)",
    "assistments_0910": "edmars data import assistments_0910 <the downloaded "
    "CSV file>   (ASSISTments is imported by hand for now)",
}

VENUES: dict[str, str] = {
    "EDM": "EDM - Educational Data Mining conference",
    "JEDM": "JEDM - Journal of Educational Data Mining",
    "JLA": "JLA - Journal of Learning Analytics",
    "AERA_OPEN": "AERA Open",
}
JOURNAL_VENUES: frozenset[str] = frozenset({"JEDM", "JLA", "AERA_OPEN"})

# R6 text constants. Ranges, not promises; ASCII so --plain output is safe.
TIME_WITHOUT_REVIEW = "usually 10-35 minutes"
TIME_WITH_REVIEW = (
    "usually 35-60 minutes with the automated review, occasionally about 2 hours"
)
COST_DEEPSEEK = (
    "roughly US$0.05-0.20 per study with DeepSeek, including the automated "
    "review (measured on only a few runs). You pay DeepSeek directly."
)
COST_OTHER = (
    "not estimated for this AI service; live token counts are shown while the "
    "study runs. You pay the service directly."
)
COST_LOCAL = (
    "no charge from an AI service (your own model server); live token counts "
    "are shown while the study runs."
)
EXPERIMENTAL_BADGE = "[EXPERIMENTAL]"
EXPERIMENTAL_TEXT = (
    "This study plan is not one of the tested examples. It has passed the "
    "automatic checks, but a plan like it has not been run end to end before, "
    "so expect rougher results and check them with extra care."
)
MENU_NOTE = (
    "Built with the edmars 'Build my own' menus (EXPERIMENTAL): variables were "
    "chosen from the dataset registry by the user; this plan has not been "
    "validated end to end."
)
PREFLIGHT_MESSAGE = (
    "Checking your study. The first check of a large data file can take a few "
    "minutes; later checks reuse a cache."
)
LANGUAGE_NOTE = (
    "Your question does not look like English. EDM-ARS writes the paper in "
    "English. The AI can usually work from a question in another language, but "
    "results are best when the question is in English."
)
LANGUAGE_NOTE_UNSURE = (
    "Your description does not look like English. The study-type helper only "
    "understands English keywords, so please choose the type yourself. (The "
    "paper is written in English either way.)"
)


class StudyError(ValueError):
    """A study request that cannot become a runnable plan.

    The message is plain English and safe to show to the user as is.
    """


@dataclass
class ExampleStudy:
    """One tested example spec from ``runs/fixtures``."""

    id: str
    path: Path
    task_type: str
    dataset: str
    research_question: str
    title: str
    spec: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MenuOption:
    """One entry of a menu. ``disabled`` holds the reason it is greyed out."""

    value: str
    label: str
    disabled: str | None = None


@dataclass(frozen=True)
class Suggestion:
    """The "Not sure" helper's guess at a study type."""

    task_type: str
    why: str
    intent: str


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _sget(settings: dict | None, dotted: str, default: Any = None) -> Any:
    node: Any = settings or {}
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return default if node is None else node


def registry_dir() -> Path:
    """The dataset registries shipped with the app (never the cwd)."""
    return paths.app_root() / "data_registry" / "datasets"


def _registry(dataset: str) -> dict:
    from src.ideation.feasibility import load_registry

    registry, _ = load_registry(dataset, registry_dir())
    return registry if isinstance(registry, dict) else {}


def _var_map(registry: dict) -> dict[str, dict]:
    from src.ideation.feasibility import build_var_map

    return build_var_map(registry)


def _predictors(registry: dict) -> list[dict]:
    """Registry predictors in registry order, each tagged with its category."""
    out: list[dict] = []
    groups = ((registry.get("variables") or {}).get("predictors")) or {}
    if isinstance(groups, dict):
        for category, members in groups.items():
            for meta in members or []:
                if isinstance(meta, dict) and meta.get("name"):
                    tagged = dict(meta)
                    tagged["_category"] = str(category)
                    out.append(tagged)
    return out


def _outcomes(registry: dict) -> list[dict]:
    items = ((registry.get("variables") or {}).get("outcomes")) or []
    return [o for o in items if isinstance(o, dict) and o.get("name")]


def _wave_index(registry: dict, meta: dict | None) -> int | None:
    order = [str(w) for w in (registry.get("temporal_order") or [])]
    wave = str((meta or {}).get("wave") or "")
    return order.index(wave) if wave in order else None


def _pct(meta: dict) -> float | None:
    value = meta.get("pct_missing")
    return float(value) if isinstance(value, (int, float)) else None


def _usable(meta: dict) -> bool:
    """A variable the screen would not kill on sight."""
    if meta.get("derived"):
        return False
    if str(meta.get("type") or "").lower() == "id":
        return False
    pct = _pct(meta)
    return pct is None or pct < 99.0


def plain_label(meta: dict) -> str:
    """A registry label without its NCES wave prefix ("X1 ", "S3 ")."""
    label = str(meta.get("label") or meta.get("name") or "").strip()
    label = re.sub(r"^[A-Z]\d\s+", "", label)
    return re.sub(r"\s+", " ", label)


def _lower_first(text: str) -> str:
    """Lower-case a leading capital, but leave acronyms ("BY math") alone."""
    if not text or (len(text) > 1 and text[1].isupper()):
        return text
    return text[:1].lower() + text[1:]


def _option_label(meta: dict) -> str:
    pct = _pct(meta)
    missing = f"{pct:.0f}% missing" if pct is not None else "missingness unknown"
    return f"{meta.get('name')} - {plain_label(meta)} ({missing})"


def _dataset_short(dataset: str) -> str:
    return DATASET_SHORT.get(dataset, dataset)


def _dataset_label(dataset: str) -> str:
    if dataset in DATASET_LABELS:
        return DATASET_LABELS[dataset]
    registry = _registry(dataset)
    return str(registry.get("full_name") or dataset)


def _matrix() -> dict[str, dict[str, bool]]:
    from src.ideation.feasibility import DATASET_TASK_MATRIX

    return DATASET_TASK_MATRIX


def unsupported_reason(dataset: str, task_type: str) -> str | None:
    """Why ``task_type`` cannot run on ``dataset``, or None when it can.

    The feasibility module's DATASET_TASK_MATRIX decides; this adds the
    plain-language reason.
    """
    row = _matrix().get(dataset)
    if row is None:
        return f"{dataset!r} is not one of the datasets EDM-ARS knows."
    if row.get(task_type, False):
        return None
    return _PLAIN_UNSUPPORTED.get(
        (dataset, task_type),
        f"{dataset} cannot support a {TASK_LABELS.get(task_type, task_type)} "
        f"study.",
    )


def known_datasets() -> list[str]:
    return list(_matrix())


def normalize_venue(venue: str) -> str:
    """Map user input ("aera open", "jedm") to a VENUES key."""
    key = re.sub(r"[\s\-]+", "_", str(venue or "").strip()).upper()
    if key in VENUES:
        return key
    raise StudyError(
        f"Unknown venue {venue!r}. Choose one of: {', '.join(VENUES)}."
    )


def _default_options(settings: dict | None) -> dict[str, Any]:
    try:
        venue = normalize_venue(_sget(settings, "defaults.venue", "EDM"))
    except StudyError:
        venue = "EDM"
    paper_format = str(_sget(settings, "defaults.paper_format", "conference"))
    if paper_format not in ("conference", "journal"):
        paper_format = "conference"
    review = bool(
        _review_available(settings) and _sget(settings, "lsar.auto_review", False)
    )
    return {"venue": venue, "paper_format": paper_format, "review": review}


def _review_available(settings: dict | None) -> bool:
    return bool(_sget(settings, "lsar.enabled", False)) and (
        _sget(settings, "provider", "deepseek") != "local"
    )


# --------------------------------------------------------------------------
# Machine-specific lookups (monkeypatched in tests)
# --------------------------------------------------------------------------


def _raw_data_dir(settings: dict | None) -> Path:
    """Where the pipeline will look for raw data (``paths.raw_data``)."""
    from edmars import datasets

    return Path(datasets.raw_data_dir(settings or {}))


def _cache_dir() -> Path:
    """The feasibility probe cache (CLI_SPEC section 2)."""
    return paths.cache_dir() / "tier1"


def _find_rscript(settings: dict | None) -> str | None:
    from edmars import toolchain

    return toolchain.find_rscript(settings or {})


def _ui() -> ModuleType:
    from edmars import ui

    return ui


def expected_data_path(dataset: str, settings: dict | None) -> Path | None:
    """The file the pipeline will read for ``dataset`` on this computer."""
    try:
        from src.dataset_adapter import create_dataset_adapter

        filename = create_dataset_adapter(dataset).get_raw_data_filename()
    except (ValueError, ImportError):
        return None
    return _raw_data_dir(settings) / filename


def data_installed(dataset: str, settings: dict | None) -> bool:
    path = expected_data_path(dataset, settings)
    return bool(path is not None and path.is_file())


# --------------------------------------------------------------------------
# Example studies (runs/fixtures)
# --------------------------------------------------------------------------

#: Short plain titles for the shipped examples. Unknown fixtures fall back
#: to their research question. Deliberately no scores or run numbers.
_EXAMPLE_TITLES: dict[str, str] = {
    "x1mtheff_x4college": "Math self-efficacy and college attendance "
    "(average effect)",
    "x1mtheff_itr": "Who gains most from high math self-efficacy? "
    "(a targeting rule for college attendance)",
    "did_ses_gap": "Did the SES gap in math rank change between the 2002 and "
    "2009 cohorts? (raw gap change)",
    "did_ses_gap_v2": "Did the SES gap in math rank change between the 2002 "
    "and 2009 cohorts? (composition-adjusted, with subgroup differences)",
    "psy_hsls_matheff_invariance": "Do the HSLS:09 math self-efficacy and "
    "utility scales work the same across sex and SES?",
    "psy_els_mathse_calibration": "How well does the ELS:2002 math "
    "self-efficacy scale measure? (item response calibration)",
    "psy_assistments_cdm": "Which math skills have ASSISTments students "
    "mastered? (cognitive diagnosis)",
}


def _example_id(stem: str) -> str:
    return stem[5:] if stem.startswith("spec_") else stem


def load_examples(fixtures_dir: Path | None = None) -> dict[str, ExampleStudy]:
    """Read every example spec in ``runs/fixtures``; skip unreadable files."""
    base = fixtures_dir or (paths.app_root() / "runs" / "fixtures")
    found: dict[str, ExampleStudy] = {}
    if not base.is_dir():
        return found
    for path in sorted(base.glob("*.json")):
        try:
            spec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(spec, dict):
            continue
        task_type = spec.get("task_type")
        question = str(spec.get("research_question") or "").strip()
        if task_type not in TASK_TYPES or not question:
            continue
        ex_id = _example_id(path.stem)
        found[ex_id] = ExampleStudy(
            id=ex_id,
            path=path,
            task_type=str(task_type),
            dataset=str(spec.get("dataset") or DEFAULT_DATASET),
            research_question=question,
            title=_EXAMPLE_TITLES.get(ex_id) or textwrap.shorten(question, 90),
            spec=spec,
        )
    return found


def _safe_load_examples() -> dict[str, ExampleStudy]:
    try:
        return load_examples()
    except Exception:  # pragma: no cover - a broken app root must not crash import
        return {}


#: The tested example studies, keyed by id (the fixture file stem without
#: its ``spec_`` prefix, e.g. ``x1mtheff_x4college``).
EXAMPLES: dict[str, ExampleStudy] = _safe_load_examples()


def find_example(ref: str) -> ExampleStudy | None:
    """Resolve ``x1mtheff_itr``, ``spec_x1mtheff_itr`` or ``...itr.json``."""
    key = Path(str(ref).strip()).name
    if key.lower().endswith(".json"):
        key = key[:-5]
    key = _example_id(key)
    if key in EXAMPLES:
        return EXAMPLES[key]
    lowered = {k.lower(): v for k, v in EXAMPLES.items()}
    return lowered.get(key.lower())


def examples_for(task_type: str, dataset: str | None = None) -> list[ExampleStudy]:
    return [
        ex
        for ex in EXAMPLES.values()
        if ex.task_type == task_type and (dataset is None or ex.dataset == dataset)
    ]


# --------------------------------------------------------------------------
# Out-of-scope and language checks (rule-based, English)
# --------------------------------------------------------------------------

_ANALYSIS_WORDS = re.compile(
    r"\b(predict\w*|effect\w*|impact\w*|caus\w*|model\w*|associat\w*|"
    r"relationship|hsls|els|assistments|dataset|data|scale|measure\w*)\b"
)

#: (pattern, why, nearest supported framing). First match wins.
_OUT_OF_SCOPE_RULES: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (
        re.compile(
            r"\b(qualitative|interview\w*|focus[- ]groups?|ethnograph\w*|"
            r"thematic analysis|lived experiences?|classroom observations?)\b"
        ),
        "This sounds like a qualitative study (interviews, focus groups, "
        "observation or fieldwork). EDM-ARS only runs quantitative analyses of "
        "large public-use datasets; it cannot collect or analyse interviews.",
        "a quantitative study of a related question in national survey data, "
        "for example which ninth-grade experiences predict the outcome you "
        "care about, or whether students' survey answers about it measure the "
        "same thing across groups.",
    ),
    (
        re.compile(r"\b(meta[- ]?analy\w*|systematic reviews?|scoping reviews?)\b"),
        "This sounds like a meta-analysis or systematic review. EDM-ARS "
        "analyses one dataset directly; it does not pool results from "
        "published studies.",
        "a single-dataset study of the same question, for example a prediction "
        "or average-effect study in HSLS:09. The paper will still include a "
        "short related-work section.",
    ),
    (
        re.compile(
            r"\b(randomi[sz]ed (controlled )?(trials?|experiments?)|rcts?|"
            r"field experiments?|power analysis)\b|"
            r"\b(design|plan|run|conduct)\w*\s+(an?\s+)?(new\s+)?"
            r"(experiment|trial|intervention study)\b"
        ),
        "This sounds like designing or running an experiment (such as a "
        "randomized trial). EDM-ARS analyses existing observational data; it "
        "cannot design, randomize or run a study.",
        "a cause-and-effect study on existing data: it compares similar "
        "students who differ in the thing you care about, under the stated "
        "assumption that the measured background explains who got it.",
    ),
    (
        re.compile(
            r"\b(design\w*|develop\w*|creat\w*|build\w*|writ\w*|construct\w*|"
            r"draft\w*)\s+(an?\s+|the\s+|my\s+|our\s+)?(new\s+)?"
            r"(survey|questionnaire|instrument|test items|survey items)\b|"
            r"\b(survey|questionnaire|instrument) (design|development|"
            r"construction)\b|\bitem writing\b"
        ),
        "This sounds like designing a new survey or test. EDM-ARS studies "
        "existing survey items; it does not write new ones or collect "
        "responses.",
        "a measurement study of an existing scale, for example how well the "
        "HSLS:09 math self-efficacy items measure and whether they work the "
        "same for different groups of students.",
    ),
    (
        # "our students" / "our schools" is often rhetorical ("are our
        # schools failing ..."), so only a possessive that names a data
        # source, or a first-person "my school/class", counts.
        re.compile(
            r"\b(my|our)\s+own\b|"
            r"\b(my|our)\s+(data(sets?)?|district|records|spreadsheets?|"
            r"files?|gradebooks?|lms)\b|"
            r"\bmy\s+(school|students|class|classes|classroom|survey)\b|"
            r"\bdistrict('s)? data\b|\b(csv|excel|spreadsheets?)\b|"
            r"\bupload\w*\b.{0,40}\b(data|files?|dataset)\b"
        ),
        "This sounds like it needs your own data. EDM-ARS works only with the "
        "public-use datasets it knows (HSLS:09, ELS:2002, the combined ELS + "
        "HSLS panel and ASSISTments) and never with local, restricted-use or "
        "identifiable student records.",
        "the same question asked of national data, for example in HSLS:09, "
        "which follows about 23,000 U.S. students from 9th grade onward.",
    ),
)

_LIT_REVIEW = re.compile(
    r"\b(literature reviews?|review of (the )?(research|literature)|"
    r"summari[sz]e (the )?(research|literature|studies))\b"
)


def out_of_scope(text: str) -> str | None:
    """Plain explanation + nearest supported framing, or None when in scope.

    Rule-based and English-only by design: it only catches requests that
    are clearly outside what the pipeline does (qualitative work,
    meta-analysis, literature-review-only, survey/instrument design,
    the user's own data, experiment design). Anything else is let
    through; the feasibility screen and the pipeline judge the rest.
    """
    lowered = " ".join(str(text or "").lower().split())
    if not lowered:
        return None
    for pattern, why, framing in _OUT_OF_SCOPE_RULES:
        if pattern.search(lowered):
            return f"{why}\nNearest thing EDM-ARS can do: {framing}"
    if _LIT_REVIEW.search(lowered) and not _ANALYSIS_WORDS.search(lowered):
        return (
            "This sounds like a literature review on its own. EDM-ARS writes "
            "empirical papers from a dataset; its related-work section is "
            "short and is not a review.\nNearest thing EDM-ARS can do: an "
            "empirical study on the topic, for example which factors predict "
            "the outcome you are reading about, in HSLS:09."
        )
    return None


_EN_STOPWORDS = frozenset(
    "the of and to in is are does do did what which how for on with by "
    "whether students student who why can from their than that this".split()
)
_FOREIGN_STOPWORDS = frozenset(
    # Spanish / Portuguese / French / German / Italian function words that
    # are not English words.
    "el los las del que por para una uno con sobre como cual es en la le "
    "les des une est dans qui avec pour sur der das und ist nicht mit wie "
    "welche den dem ein eine auf zu os uma dos das com não não entre il gli "
    "della degli sono alunos estudiantes élèves schüler".split()
)


def looks_non_english(text: str) -> bool:
    """Heuristic: mostly non-ASCII letters, or foreign function words only."""
    letters = [c for c in str(text or "") if c.isalpha()]
    if len(letters) < 3:
        return False
    non_ascii = sum(1 for c in letters if ord(c) > 127)
    if non_ascii / len(letters) >= 0.15:
        return True
    words = re.findall(r"[^\W\d_]+", str(text).lower())
    if len(words) < 4:
        return False
    english = sum(1 for w in words if w in _EN_STOPWORDS)
    foreign = sum(1 for w in words if w in _FOREIGN_STOPWORDS)
    return foreign >= 2 and foreign > 2 * english


def language_note(text: str) -> str | None:
    return LANGUAGE_NOTE if looks_non_english(text) else None


# --------------------------------------------------------------------------
# "Not sure": suggest a study type
# --------------------------------------------------------------------------

_MEASUREMENT_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p)
    for p in (
        r"\bmeasurement\b",
        r"\binvarian\w*",
        r"\bdifferential item functioning\b",
        r"\bdif\b",
        r"\breliability\b",
        r"\bfactor (structure|analysis|model)\w*",
        r"\bcfa\b",
        r"\birt\b",
        r"\bitem response\b",
        r"\bpsychometric\w*",
        r"\b(scale|questionnaire|survey) items?\b",
        r"\bitems?\b.*\b(measure|scale|construct)\w*",
        r"\bcognitive diagnos\w*",
        r"\bskill mastery\b",
        r"\bvalidity\b",
    )
)
_DID_GAP = re.compile(r"\bgaps?\b")
_DID_TIME = re.compile(
    r"\bcohorts?\b|\bbetween (19|20)\d\d and (19|20)\d\d\b|\bover (time|the years)\b"
)

_WHY: dict[str, str] = {
    "prediction": "It asks which students are likely to reach an outcome, or "
    "how well an outcome can be forecast.",
    "causal_soo": "It asks whether one thing changes another on average (an "
    "effect).",
    "causal_itr": "It asks for whom an effect is larger, or who would benefit "
    "most.",
    "causal_did": "It asks whether a gap between two groups of students "
    "changed between two cohorts.",
    "psychometrics": "It asks how well a set of survey items measures "
    "something, or whether the items work the same for different groups.",
}


def suggest_study_type(text: str, dataset: str = DEFAULT_DATASET) -> Suggestion:
    """Guess a task type from an English description.

    Uses ``src.design_selector.classify_intent`` + ``select_design`` (the
    same deterministic layer the pipeline uses) plus two keyword groups
    the selector does not cover: measurement and cross-cohort gap change.
    """
    lowered = " ".join(str(text or "").lower().split())
    if any(p.search(lowered) for p in _MEASUREMENT_PATTERNS):
        return Suggestion("psychometrics", _WHY["psychometrics"], "measurement")
    if _DID_GAP.search(lowered) and _DID_TIME.search(lowered):
        return Suggestion("causal_did", _WHY["causal_did"], "causal")

    from src.design_selector import classify_intent, select_design

    intent = classify_intent(text)
    try:
        report = select_design(_registry(dataset), question=text, intent=intent)
        recommended = str(report.get("recommended_task_type") or "")
    except Exception:
        recommended = ""
    if recommended not in TASK_TYPES:
        recommended = {
            "targeting": "causal_itr",
            "causal": "causal_soo",
        }.get(intent, "prediction")
    return Suggestion(recommended, _WHY[recommended], intent)


def _family_of(task_type: str) -> str:
    if task_type == "prediction":
        return "prediction"
    if task_type == "psychometrics":
        return "measurement"
    return "causal"


# --------------------------------------------------------------------------
# Dataset menu (R2)
# --------------------------------------------------------------------------


def dataset_options(settings: dict | None, task_type: str) -> list[MenuOption]:
    """Every known dataset, greyed out (with a reason) when it cannot be used."""
    options: list[MenuOption] = []
    for dataset in known_datasets():
        reason = unsupported_reason(dataset, task_type)
        if reason is None and not data_installed(dataset, settings):
            reason = "not on this computer yet - run: " + _INSTALL_HINTS.get(
                dataset, f"edmars data install {dataset}"
            )
        options.append(MenuOption(dataset, _dataset_label(dataset), reason))
    return options


def causal_kind_options(settings: dict | None) -> list[MenuOption]:
    """R1 cause-and-effect sub-choice; the cohort comparison needs the panel."""
    did_reason: str | None = None
    if not data_installed("did_els_hsls_panel", settings):
        did_reason = (
            "needs the combined ELS:2002 + HSLS:09 data, which is not on this "
            "computer yet - run: " + _INSTALL_HINTS["did_els_hsls_panel"]
        )
    return [
        MenuOption("causal_soo", "The average effect - does X change Y on "
                   "average?"),
        MenuOption("causal_itr", "For whom - who would benefit most from X?"),
        MenuOption("causal_did", "A gap change - did a gap between two groups "
                   "change between the 2002 and 2009 cohorts?", did_reason),
    ]


# --------------------------------------------------------------------------
# "Build my own" menus (EXPERIMENTAL), all read from the registry
# --------------------------------------------------------------------------

#: Order in which registry categories are offered as a possible cause:
#: attitudes first (what the tested examples use), achievement last.
_TREATMENT_CATEGORY_ORDER: tuple[str, ...] = (
    "math_attitudes",
    "noncognitive",
    "academic",
    "school_level",
    "teacher",
    "academic_followup",
    "course_taking",
    "family",
    "covariates_v2",
)

#: Recommended adjustment sets. The HSLS:09 list is the adjustment set of
#: the tested targeting example (runs/fixtures/spec_x1mtheff_itr.json)
#: plus X1MTHEFF for when another scale is the cause. Filtered against the
#: registry-derived covariate menu at use, so a stale name drops out.
_RECOMMENDED_COVARIATES: dict[str, tuple[str, ...]] = {
    "hsls09_public": (
        "X1SES", "X1TXMTSCOR", "X1SEX", "X1RACE", "X1SCHOOLBEL", "X1MTHID",
        "X1MTHUTI", "X1MTHINT", "X1MTHEFF", "X1PAREDU", "X1STUEDEXPCT",
        "X1LOCALE", "X1CONTROL",
    ),
    "els_2002": (
        "BYSES1", "BYTXMSTD", "BYSEX", "BYRACE", "BYPARED", "BYSTEXP",
        "BYSCHPRG", "BYTXRSTD",
    ),
    "did_els_hsls_panel": ("female", "race5", "pared3", "expect_ba", "ses_std"),
}
_RECOMMENDED_RULE_COVARIATES: dict[str, tuple[str, ...]] = {
    "hsls09_public": ("X1SES", "X1TXMTSCOR", "X1SEX"),
    "els_2002": ("BYSES1", "BYTXMSTD", "BYSEX"),
}

_PSY_METHODS = ("P1", "P2", "P3", "P4", "P5", "P6", "P7")
#: Every psychometric method except classical item analysis (P1) runs
#: through the R bridge (P2 omega is computed from the R CFA fit).
_R_METHODS = frozenset({"P2", "P3", "P4", "P5", "P6", "P7"})


def _find(options: Sequence[MenuOption], value: str) -> MenuOption | None:
    for option in options:
        if option.value == value:
            return option
    return None


def treatment_options(dataset: str) -> list[MenuOption]:
    """Continuous, non-protected scales measured before some outcome."""
    registry = _registry(dataset)
    outcome_waves = [
        _wave_index(registry, o)
        for o in _outcomes(registry)
        if _usable(o) and str(o.get("type")) in ("binary", "continuous")
    ]
    last_outcome = max([w for w in outcome_waves if w is not None], default=-1)
    candidates = []
    for meta in _predictors(registry):
        wave = _wave_index(registry, meta)
        if (
            not _usable(meta)
            or str(meta.get("type") or "").lower() != "continuous"
            or meta.get("protected_attribute")
            or meta["_category"] == "demographic"
            or wave is None
            or wave >= last_outcome
        ):
            continue
        try:
            rank = _TREATMENT_CATEGORY_ORDER.index(meta["_category"])
        except ValueError:
            rank = len(_TREATMENT_CATEGORY_ORDER)
        candidates.append(((wave, rank), meta))
    candidates.sort(key=lambda pair: pair[0])  # stable: registry order within
    return [MenuOption(str(m["name"]), _option_label(m)) for _, m in candidates]


def outcome_options(dataset: str, treatment: str) -> list[MenuOption]:
    """Binary or continuous outcomes measured after the treatment."""
    registry = _registry(dataset)
    t_wave = _wave_index(registry, _var_map(registry).get(treatment))
    if t_wave is None:
        return []
    out: list[MenuOption] = []
    for meta in _outcomes(registry):
        wave = _wave_index(registry, meta)
        if (
            _usable(meta)
            and str(meta.get("type")) in ("binary", "continuous")
            and wave is not None
            and wave > t_wave
            and meta["name"] != treatment
        ):
            out.append(MenuOption(str(meta["name"]), _option_label(meta)))
    return out


def covariate_options(dataset: str, treatment: str, outcome: str) -> list[MenuOption]:
    """Background variables measured no later than the treatment."""
    registry = _registry(dataset)
    t_wave = _wave_index(registry, _var_map(registry).get(treatment))
    if t_wave is None:
        return []
    out: list[MenuOption] = []
    for meta in _predictors(registry):
        wave = _wave_index(registry, meta)
        if (
            _usable(meta)
            and wave is not None
            and wave <= t_wave
            and meta["name"] not in (treatment, outcome)
            and str(meta.get("type")) in ("continuous", "binary", "categorical")
        ):
            out.append(MenuOption(str(meta["name"]), _option_label(meta)))
    return out


def _prefix_dedupe(names: Sequence[str]) -> list[str]:
    kept: list[str] = []
    for name in names:
        if not any(name.startswith(k) for k in kept):
            kept.append(name)
    return kept


def default_covariates(dataset: str, treatment: str, outcome: str) -> list[str]:
    """The recommended adjustment set, restricted to what the menu offers."""
    offered = [o.value for o in covariate_options(dataset, treatment, outcome)]
    recommended = [
        n for n in _RECOMMENDED_COVARIATES.get(dataset, ()) if n in offered
    ]
    if len(recommended) >= 3:
        return recommended
    # Generic fallback for a dataset without a curated list: its protected
    # attributes (near-duplicates such as X1SES_U dropped) plus the
    # least-missing variable of each other category.
    registry = _registry(dataset)
    var_map = _var_map(registry)
    protected = _prefix_dedupe(
        sorted(
            (n for n in offered if (var_map.get(n) or {}).get("protected_attribute")),
            key=lambda n: (_pct(var_map[n]) or 0.0, n),
        )
    )
    picked = list(protected)
    seen_categories: set[str] = set()
    for meta in sorted(
        (m for m in _predictors(registry) if m["name"] in offered),
        key=lambda m: (_pct(m) if _pct(m) is not None else 100.0, m["name"]),
    ):
        category = meta["_category"]
        pct = _pct(meta)
        if (
            category in seen_categories
            or meta["name"] in picked
            or (pct is not None and pct > 30.0)
        ):
            continue
        seen_categories.add(category)
        picked.append(str(meta["name"]))
    return [n for n in offered if n in picked][:10]


def default_rule_covariates(dataset: str, covariates: Sequence[str]) -> list[str]:
    recommended = [
        n for n in _RECOMMENDED_RULE_COVARIATES.get(dataset, ()) if n in covariates
    ]
    return recommended or list(covariates[:3])


def item_bank_options(dataset: str) -> list[MenuOption]:
    registry = _registry(dataset)
    out: list[MenuOption] = []
    for name, bank in (registry.get("item_banks") or {}).items():
        if not isinstance(bank, dict):
            continue
        items = [str(i) for i in bank.get("items") or []]
        title = str(name).replace("_", " ")
        label = f"{title} ({len(items)} items: {', '.join(items)})"
        disabled = None
        if len(items) < 3:
            disabled = (
                f"only {len(items)} items - too few for a scale of its own"
            )
        out.append(MenuOption(str(name), label, disabled))
    return out


def grouping_options(dataset: str) -> list[MenuOption]:
    """Protected attributes usable as two-group comparisons."""
    registry = _registry(dataset)
    out: list[MenuOption] = []
    for meta in _predictors(registry):
        if not meta.get("protected_attribute") or not _usable(meta):
            continue
        kind = str(meta.get("type") or "")
        if kind == "binary":
            out.append(MenuOption(str(meta["name"]), _option_label(meta)))
        elif kind == "categorical":
            out.append(
                MenuOption(
                    str(meta["name"]),
                    _option_label(meta),
                    "more than two groups - the menus compare two groups; "
                    "the example studies show multi-group designs",
                )
            )
    return out


def did_group_options(dataset: str) -> list[MenuOption]:
    registry = _registry(dataset)
    timing = {
        str(v)
        for v in ((registry.get("design_feasibility") or {}).get(
            "policy_timing_variables") or [])
    }
    return [
        MenuOption(str(m["name"]), _option_label(m))
        for m in _predictors(registry)
        if _usable(m) and str(m.get("type")) == "binary" and m["name"] not in timing
    ]


def did_outcome_options(dataset: str) -> list[MenuOption]:
    registry = _registry(dataset)
    return [
        MenuOption(str(o["name"]), _option_label(o))
        for o in _outcomes(registry)
        if _usable(o) and str(o.get("type")) == "continuous"
    ]


def menu_unavailable_reason(task_type: str, dataset: str) -> str | None:
    """Why "Build my own" cannot offer anything here, or None when it can."""
    if task_type not in MENU_TASK_TYPES:
        return "prediction studies start from your own question instead"
    reason = unsupported_reason(dataset, task_type)
    if reason is not None:
        return reason
    if task_type in ("causal_soo", "causal_itr"):
        for t in treatment_options(dataset):
            if outcome_options(dataset, t.value):
                return None
        return (
            "this dataset has no scale measured before a usable outcome, so "
            "there is no cause-then-effect pair to offer"
        )
    if task_type == "causal_did":
        if did_group_options(dataset) and did_outcome_options(dataset):
            return None
        return "this dataset has no two-group variable and outcome to compare"
    if any(o.disabled is None for o in item_bank_options(dataset)):
        return None
    return (
        "this dataset has no survey item bank with at least three items; use "
        "an example study instead"
    )


def menu_question(task_type: str, dataset: str, choices: dict) -> str:
    """The research question generated from menu choices."""
    registry = _registry(dataset)
    var_map = _var_map(registry)
    short = _dataset_short(dataset)

    def lab(name: str) -> str:
        # "bottom vs top SES band (...) - the DiD group variable" -> the part
        # before the curator's " - " aside.
        text = plain_label(var_map.get(name) or {"name": name}).split(" - ")[0]
        return _lower_first(text.strip())

    def the(name: str) -> str:
        text = lab(name)
        return text if text.startswith("the ") else f"the {text}"

    if task_type in ("causal_soo", "causal_itr"):
        t, o = str(choices.get("treatment")), str(choices.get("outcome"))
        cause = f"scoring above the median on {the(t)} ({t})"
        effect = f'the outcome "{lab(o)}" ({o})'
        if task_type == "causal_soo":
            return (
                f"Does {cause} change {effect} for {short} students on "
                f"average, after adjusting for measured background differences?"
            )
        rules = ", ".join(choices.get("rule_covariates") or [])
        return (
            f"For which {short} students does {cause} change {effect} the "
            f"most, and can a simple rule based on "
            f"{rules or 'baseline covariates'} target it better than treating "
            f"everyone the same, under the no-unmeasured-confounding assumption?"
        )
    if task_type == "causal_did":
        group, outcome = str(choices.get("group")), str(choices.get("outcome"))
        cohorts = (
            "the ELS:2002 and HSLS:09 cohorts"
            if dataset == "did_els_hsls_panel"
            else "the two cohorts"
        )
        return (
            f"Did the gap in {lab(outcome)} ({outcome}) between the groups "
            f"defined by {group} ({lab(group)}) change between {cohorts} (a "
            f"gap-in-gaps difference-in-differences)?"
        )
    banks = [str(b).replace("_", " ") for b in choices.get("item_banks") or []]
    groups = list(choices.get("grouping_vars") or [])
    question = (
        f"How well do the {short} {' and '.join(banks)} items measure what "
        f"they are meant to measure (reliability and factor structure)"
    )
    if groups:
        question += (
            ", and do they work the same way across the groups defined by "
            + " and ".join(f"{g} ({lab(g)})" for g in groups)
        )
    return question + "?"


def _require_choice(
    what: str, value: object, options: Sequence[MenuOption]
) -> str:
    name = str(value or "").strip()
    option = _find(options, name)
    if option is None:
        offered = ", ".join(o.value for o in options if o.disabled is None)
        raise StudyError(
            f"{name or '(nothing)'} is not an available {what}. "
            f"Available: {offered or 'none'}."
        )
    if option.disabled:
        raise StudyError(f"{name} cannot be used as the {what}: {option.disabled}.")
    return name


def _require_all(what: str, values: Sequence[str], options: Sequence[MenuOption]) -> list[str]:
    out: list[str] = []
    for value in values:
        name = _require_choice(what, value, options)
        if name not in out:
            out.append(name)
    return out


def _draft_causal(task_type: str, dataset: str, choices: dict) -> dict:
    registry = _registry(dataset)
    var_map = _var_map(registry)
    treatment = _require_choice(
        "cause ('treatment')", choices.get("treatment"), treatment_options(dataset)
    )
    outcome = _require_choice(
        "outcome", choices.get("outcome"), outcome_options(dataset, treatment)
    )
    cov_menu = covariate_options(dataset, treatment, outcome)
    covariates = _require_all(
        "adjustment variable",
        list(choices.get("covariates") or default_covariates(dataset, treatment, outcome)),
        cov_menu,
    )
    if not covariates:
        raise StudyError("Choose at least one background variable to adjust for.")
    o_meta = var_map.get(outcome) or {}
    draft: dict[str, Any] = {
        "treatment": {
            "variable": treatment,
            "operationalization": "median_split_binary",
            "rationale_for_PF": (
                "Chosen in the edmars menus (EXPERIMENTAL): students above the "
                "median count as treated. A median split of a continuous scale "
                "is a known weak operationalization (ESC-07); defend or refine "
                "it."
            ),
        },
        "outcome": {
            "variable": outcome,
            "type": str(o_meta.get("type") or "binary"),
            "definition": plain_label(o_meta) or outcome,
        },
        "comparator_method": "M1",
        "exclude_methods": [],
        "adjustment_set": covariates,
    }
    protected = [c for c in covariates if (var_map.get(c) or {}).get("protected_attribute")]
    protected_binary = [
        c for c in protected if str((var_map.get(c) or {}).get("type")) == "binary"
    ]
    if protected:
        # Protected attributes in the adjustment set call for subgroup
        # reporting (registry pitfall protected_attribute_misuse).
        draft["subgroup_analyses"] = (protected_binary or protected)[:1]
    if task_type == "causal_soo":
        draft.update(
            {
                "target_estimand_hint": (
                    "ATT preferred (effect on the treated students); declare "
                    "the estimand explicitly per G2."
                ),
                "primary_method": "M2",
                "secondary_methods": ["M3", "M4"],
                "rationale_for_method_set": (
                    "Menu default: matching (M2) as the primary estimate with "
                    "regression adjustment (M1), weighting (M3) and a doubly "
                    "robust estimate (M4) as the standard cross-checks."
                ),
            }
        )
    else:
        rules = _require_all(
            "rule variable",
            list(
                choices.get("rule_covariates")
                or default_rule_covariates(dataset, covariates)
            ),
            [o for o in cov_menu if o.value in covariates],
        )
        if not rules:
            raise StudyError("Choose at least one variable the rule may use.")
        draft.update(
            {
                "target_estimand_hint": (
                    "policy value of a learned rule vs the best constant policy"
                ),
                "primary_method": "M6",
                "secondary_methods": ["M5"] if protected_binary else [],
                "rule_covariates": rules,
                "rationale_for_method_set": (
                    "Menu default: policy learning (M6) with regression "
                    "adjustment (M1) as the comparator"
                    + (" and a causal forest (M5) for heterogeneity."
                       if protected_binary else ".")
                ),
            }
        )
        if protected_binary:
            draft["subgroup_of_interest_for_m5"] = protected_binary[0]
    return draft


def _draft_did(dataset: str, choices: dict) -> dict:
    registry = _registry(dataset)
    var_map = _var_map(registry)
    group = _require_choice("group variable", choices.get("group"), did_group_options(dataset))
    outcomes = did_outcome_options(dataset)
    outcome = _require_choice(
        "outcome",
        choices.get("outcome") or (outcomes[0].value if outcomes else ""),
        outcomes,
    )
    timing = [
        str(v)
        for v in ((registry.get("design_feasibility") or {}).get(
            "policy_timing_variables") or [])
    ]
    o_meta = var_map.get(outcome) or {}
    draft: dict[str, Any] = {
        "group_variable": group,
        "post_variable": timing[0] if timing else None,
        "outcome": {
            "variable": outcome,
            "type": "continuous",
            "definition": plain_label(o_meta) or outcome,
        },
        "estimand": "DID_GAP_CHANGE",
        "primary_method": "M8",
        "secondary_methods": [],
        "rationale_for_method_set": (
            "Menu default: the raw gap-in-gaps estimator (M8) with its "
            "stratified bootstrap, as in the tested example."
        ),
    }
    if draft["post_variable"] is None:
        del draft["post_variable"]
    later = [
        o.value
        for o in outcomes
        if o.value != outcome
        and (_wave_index(registry, var_map.get(o.value)) or 0)
        > (_wave_index(registry, o_meta) or 0)
    ]
    if later:
        draft["placebo_outcome"] = later[0]
    others = [o.value for o in did_group_options(dataset) if o.value != group]
    protected_others = [
        n for n in others if (var_map.get(n) or {}).get("protected_attribute")
    ]
    if protected_others:
        draft["subgroup_analyses"] = protected_others[:1]
    concerns = [
        f"{p.get('id')}: {p.get('description')}"
        for p in registry.get("common_pitfalls") or []
        if isinstance(p, dict) and p.get("id")
    ]
    if concerns:
        draft["known_concerns_to_flag"] = concerns
    return draft


def _draft_psychometrics(dataset: str, choices: dict) -> dict:
    registry = _registry(dataset)
    var_map = _var_map(registry)
    bank_menu = item_bank_options(dataset)
    requested = list(choices.get("item_banks") or [])
    if not requested:
        usable = [o for o in bank_menu if o.disabled is None]
        banks_all = registry.get("item_banks") or {}
        usable.sort(key=lambda o: -len((banks_all.get(o.value) or {}).get("items") or []))
        requested = [usable[0].value] if usable else []
    banks = _require_all("item bank", requested, bank_menu)
    if not banks:
        raise StudyError("This dataset has no item bank the menus can use.")
    groups = _require_all(
        "grouping variable", list(choices.get("grouping_vars") or []),
        grouping_options(dataset),
    )
    methods = [str(m).upper() for m in choices.get("methods") or []]
    if not methods:
        methods = ["P1", "P2", "P3"]
        if choices.get("irt"):
            methods.append("P4")
        if groups:
            methods += ["P5", "P6"]
    unknown = [m for m in methods if m not in _PSY_METHODS]
    if unknown:
        raise StudyError(
            f"Unknown method(s) {unknown}; the measurement methods are "
            f"{', '.join(_PSY_METHODS)}."
        )
    if any(m in ("P5", "P6") for m in methods) and not groups:
        raise StudyError(
            "Group comparisons (P5 item functioning, P6 invariance) need a "
            "grouping variable."
        )

    bank_data = registry.get("item_banks") or {}
    items: list[str] = []
    reverse: list[str] = []
    factor_lines: list[str] = []
    concerns: list[str] = []
    labels_seen: list[Any] = []
    codes_seen: list[Any] = []
    for name in banks:
        bank = bank_data.get(name) or {}
        bank_items = [str(i) for i in bank.get("items") or []]
        items += [i for i in bank_items if i not in items]
        reverse += [str(i) for i in bank.get("reverse") or [] if str(i) not in reverse]
        factor = re.sub(r"[^0-9A-Za-z_]", "_", name).upper()
        factor_lines.append(f"{factor} =~ " + " + ".join(bank_items))
        labels_seen.append(bank.get("response_labels"))
        codes_seen.append(bank.get("response_codes"))
        if bank.get("note"):
            concerns.append(f"{name}: {bank['note']}")
    titles = [b.replace("_", " ") for b in banks]
    draft: dict[str, Any] = {
        "scale_name": f"{_dataset_short(dataset)} " + " + ".join(titles),
        "item_columns": items,
        "reverse_items": reverse,
        "factor_model": "\n".join(factor_lines),
        "grouping_vars": groups,
        "method_battery": methods,
        "target_population": (
            f"{_dataset_short(dataset)} students who answered the "
            f"{' and '.join(titles)} items"
        ),
    }
    if labels_seen and labels_seen[0] and all(x == labels_seen[0] for x in labels_seen):
        draft["response_labels"] = labels_seen[0]
    if codes_seen and codes_seen[0] and all(x == codes_seen[0] for x in codes_seen):
        draft["response_codes"] = codes_seen[0]
    notes: list[str] = []
    for g in groups:
        codes = (var_map.get(g) or {}).get("codebook_codes") or {}
        valid = {
            str(k): str(v)
            for k, v in codes.items()
            if not str(k).lstrip().startswith("-")
        }
        if valid:
            notes.append(
                f"{g} codes: "
                + ", ".join(f"{k}={v}" for k, v in valid.items())
                + "; negative codes are missing-data sentinels."
            )
    if notes:
        draft["grouping_notes"] = " ".join(notes)
    concerns.append(
        "Menu-built (EXPERIMENTAL): each chosen item bank is modelled as one "
        "factor; that structure is an assumption to test, not a finding."
    )
    draft["known_concerns_to_flag"] = concerns
    return draft


def build_menu_spec(
    settings: dict | None, task_type: str, dataset: str, choices: dict
) -> dict:
    """Build a locked research spec from menu choices (EXPERIMENTAL).

    ``choices`` by task type (names are registry variable names):

    * ``causal_soo`` / ``causal_itr``: ``treatment``, ``outcome``, optional
      ``covariates`` (default: the recommended set), ``causal_itr`` also
      ``rule_covariates``;
    * ``causal_did``: ``group``, optional ``outcome``;
    * ``psychometrics``: ``item_banks`` (list), optional ``grouping_vars``,
      ``irt`` (bool) or an explicit ``methods`` list;
    * any: optional ``research_question`` (default: generated).

    The choices become an :class:`src.ideation.cards.IdeaCard`, which
    ``compile_spec`` completes from the registry. The result is checked
    with the pipeline's own ``load_locked_research_spec``; a spec that
    fails it raises :class:`StudyError` instead of being returned.
    """
    del settings  # the registries ship with the app; nothing per-user yet
    if task_type not in MENU_TASK_TYPES:
        raise StudyError(
            f"'Build my own' covers {', '.join(MENU_TASK_TYPES)}; "
            f"{task_type!r} is not one of them."
        )
    reason = unsupported_reason(dataset, task_type)
    if reason is not None:
        raise StudyError(f"{TASK_LABELS[task_type]} is not possible here: {reason}")
    if not _registry(dataset):
        raise StudyError(f"No registry found for dataset {dataset!r}.")

    if task_type in ("causal_soo", "causal_itr"):
        draft = _draft_causal(task_type, dataset, choices)
        question_choices = {
            "treatment": draft["treatment"]["variable"],
            "outcome": draft["outcome"]["variable"],
            "rule_covariates": draft.get("rule_covariates"),
        }
    elif task_type == "causal_did":
        draft = _draft_did(dataset, choices)
        question_choices = {
            "group": draft["group_variable"],
            "outcome": draft["outcome"]["variable"],
        }
    else:
        draft = _draft_psychometrics(dataset, choices)
        question_choices = {
            "item_banks": list(choices.get("item_banks") or [])
            or _banks_from_draft(dataset, draft),
            "grouping_vars": draft.get("grouping_vars"),
        }
    question = str(choices.get("research_question") or "").strip() or menu_question(
        task_type, dataset, question_choices
    )

    from src.ideation.cards import IdeaCard, compile_spec

    digest = hashlib.sha1(
        json.dumps([task_type, dataset, draft], sort_keys=True, default=str).encode(
            "utf-8"
        )
    ).hexdigest()[:8]
    card = IdeaCard(
        candidate_id=f"menu_{digest}",
        tournament_id="edmars-cli",
        cell={"dataset": dataset, "task_type": task_type},
        research_question=question,
        spec_draft=draft,
        generator_model="edmars menus (no model)",
        notes=[MENU_NOTE],
    )
    card.normalize()
    spec = compile_spec(card, registry_dir=registry_dir())
    spec["research_question"] = question
    spec["note"] = MENU_NOTE
    if not str(spec.get("expected_contribution") or "").strip():
        spec.pop("expected_contribution", None)  # the menus write no prose
    compiled_by = spec.get("compiled_by")
    if isinstance(compiled_by, dict):
        compiled_by["builder"] = "edmars.study.build_menu_spec"
        # compile_spec records the registry's absolute path. The spec is
        # saved in the study folder and shown to the AI, so keep the
        # user's directory layout out of it.
        compiled_by["registry_source"] = f"data_registry/datasets/{dataset}.yaml"

    problems, _ = check_spec(spec, dataset)
    if problems:
        raise StudyError(
            "The menu choices did not make a valid study plan:\n  - "
            + "\n  - ".join(problems)
        )
    return spec


def _banks_from_draft(dataset: str, draft: dict) -> list[str]:
    registry = _registry(dataset)
    items = set(draft.get("item_columns") or [])
    return [
        str(name)
        for name, bank in (registry.get("item_banks") or {}).items()
        if isinstance(bank, dict) and set(bank.get("items") or []) <= items
    ]


# --------------------------------------------------------------------------
# Spec validation through the pipeline's own loader
# --------------------------------------------------------------------------


def _import_without_env_side_effects(module_name: str) -> ModuleType:
    """Import a module, then drop any environment variables it added.

    ``src.main`` calls ``load_dotenv()`` at import. In a development
    checkout that loads a stray ``.env`` into this process, where it would
    shadow the keys stored in the OS keychain. Values that existed before
    are left alone (load_dotenv never overrides them).
    """
    before = set(os.environ)
    try:
        return importlib.import_module(module_name)
    finally:
        for key in set(os.environ) - before:
            os.environ.pop(key, None)


def _spec_loader() -> Callable[..., dict]:
    module = _import_without_env_side_effects("src.main")
    return module.load_locked_research_spec  # type: ignore[no-any-return]


_BUILDER_KEYS = frozenset({"compiled_by", "expected_contribution", "note"})


def check_spec(spec: dict, dataset: str | None = None) -> tuple[list[str], list[str]]:
    """Run ``src.main.load_locked_research_spec`` on ``spec``.

    Returns ``(problems, notes)``: problems are the loader's blocking
    errors (empty when the pipeline accepts the spec); notes are its
    non-blocking warnings about keys the pipeline will ignore, minus the
    provenance keys this module adds itself.
    """
    loader = _spec_loader()
    with tempfile.TemporaryDirectory(prefix="edmars-spec-") as tmp:
        path = Path(tmp) / "research_spec.locked.json"
        path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        captured = io.StringIO()
        try:
            with contextlib.redirect_stderr(captured):
                loader(
                    str(path),
                    dataset=dataset,
                    registry_dir=str(paths.app_root() / "data_registry"),
                )
        except (ValueError, OSError) as exc:
            message = str(exc).replace(repr(str(path)), "the study plan")
            message = message.replace(str(path), "the study plan")
            return [message], []
    notes: list[str] = []
    for line in captured.getvalue().splitlines():
        match = re.search(r"will ignore: (.+?)\. If they", line)
        if not match:
            continue
        keys = [k.strip() for k in match.group(1).split(",")]
        ignored = [k for k in keys if k and k not in _BUILDER_KEYS]
        if ignored:
            notes.append(
                "These fields in the study plan are ignored by the pipeline: "
                + ", ".join(ignored)
                + ". Put guidance the AI must follow in 'additional_constraints'."
            )
    return [], notes


# --------------------------------------------------------------------------
# R4 preflight
# --------------------------------------------------------------------------

_CHECK_TITLES: dict[str, str] = {
    "F-TASK-INCOMPATIBLE": "Study type works with this dataset",
    "F-VAR-ABSENT": "Every variable is in the dataset's catalogue",
    "F-COL-ABSENT": "Every variable is in the data file",
    "F-TEMPORAL-ORDER": "Earlier measures come before the outcome",
    "F-TIER3-EXCLUDED": "No survey weights, IDs or processing flags used",
    "F-DEAD-VARIABLE": "No suppressed or empty variables used",
    "F-ESTIMATOR-UNCERTIFIED": "Methods are ones EDM-ARS has tested",
    "F-DESIGN-INFEASIBLE": "The study design is possible with this data",
    "F-SPEC-INCOMPLETE": "The study plan is complete",
    "F-NO-PROTECTED-ATTRS": "Group comparisons are possible",
    "F-ITEM-BANK-TOO-FEW": "Each scale has enough items",
    "F-SUBGROUP-VAR-UNKNOWN": "Subgroup variables exist",
    "F-METADATA-UNVERIFIED": "Variables are documented",
    "F-PITFALL-TOUCHED": "Known pitfalls for this dataset",
    "P-ANALYTIC-N": "Enough students with usable data",
    "P-CLASS-BALANCE": "The outcome is not too rare",
    "P-POSITIVITY": "Treated and untreated students overlap",
    "P-DID-CELLS": "Every group-by-cohort cell has students",
    "P-CDM-SCOPE": "The skill data covers enough students",
}

_KILL_FIXES: dict[str, str] = {
    "F-TASK-INCOMPATIBLE": "Pick a dataset that supports this kind of study "
    "(`edmars new` lists them).",
    "F-VAR-ABSENT": "Replace the variable named above; the 'Build my own' "
    "menus only offer variables the dataset has.",
    "F-COL-ABSENT": "Replace the variable named above, or check that the data "
    "file is the right one (`edmars data verify`).",
    "F-TEMPORAL-ORDER": "Choose an outcome measured after the other variables.",
    "F-TIER3-EXCLUDED": "Remove the weight, ID or flag variable named above.",
    "F-DEAD-VARIABLE": "Replace the suppressed or empty variable named above.",
    "F-ESTIMATOR-UNCERTIFIED": "Use one of the tested methods (the example "
    "studies show them).",
    "F-SPEC-INCOMPLETE": "Start from an example study or the 'Build my own' "
    "menus, which fill every required field.",
    "F-ITEM-BANK-TOO-FEW": "Give each factor at least three items.",
}
_DEFAULT_KILL_FIX = (
    "Change the study (`edmars new`) or start from one of the example studies."
)


def _check_title(code: str) -> str:
    if code.startswith("F-CHECK-ERROR"):
        return "An automatic check could not run"
    return _CHECK_TITLES.get(code, code)


def _map_report(report: Any) -> list[Check]:
    from src.ideation.feasibility import KILL, WARN

    out: list[Check] = []
    skipped: list[str] = []
    for result in report.checks:
        title = _check_title(result.code)
        if result.status == KILL:
            out.append(
                Check(
                    title,
                    "fail",
                    f"{result.message} Why: {result.evidence}",
                    _KILL_FIXES.get(result.code, _DEFAULT_KILL_FIX),
                )
            )
        elif result.status == WARN:
            out.append(Check(title, "warn", result.message))
        elif str(result.message).startswith("Skipped"):
            skipped.append(re.sub(r"^Skipped:\s*", "", result.message).rstrip("."))
        else:
            out.append(Check(title, "ok", result.message))
    if skipped:
        shown = "; ".join(skipped[:3]) + ("; ..." if len(skipped) > 3 else "")
        out.append(
            Check(
                "Checks that did not apply",
                "info",
                f"{len(skipped)} automatic check(s) were skipped ({shown}).",
            )
        )
    return out


def _compat_check(dataset: str, task_type: str) -> Check:
    reason = unsupported_reason(dataset, task_type)
    title = _CHECK_TITLES["F-TASK-INCOMPATIBLE"]
    if reason is None:
        return Check(
            title, "ok",
            f"{_dataset_short(dataset)} supports {TASK_LABELS.get(task_type, task_type)} "
            f"studies.",
        )
    return Check(title, "fail", reason, _KILL_FIXES["F-TASK-INCOMPATIBLE"])


def _data_check(dataset: str, settings: dict | None) -> tuple[Check, bool]:
    path = expected_data_path(dataset, settings)
    title = "The data file is on this computer"
    if path is None:
        return Check(title, "fail", f"EDM-ARS does not know the dataset {dataset!r}.",
                     "Choose one of: " + ", ".join(known_datasets())), False
    if path.is_file():
        return Check(title, "ok", f"Found {path}"), True
    return (
        Check(
            title,
            "fail",
            f"The study needs {path}, which is not there yet. Without it the "
            f"study would fail after the first paid AI steps.",
            _INSTALL_HINTS.get(dataset, f"edmars data install {dataset}"),
        ),
        False,
    )


def _r_check(spec: dict | None, settings: dict | None) -> Check | None:
    battery = [str(m).upper() for m in (spec or {}).get("method_battery") or []]
    needs = sorted(m for m in battery if m in _R_METHODS)
    if not needs:
        return None
    title = "R is installed (needed for the measurement models)"
    try:
        rscript = _find_rscript(settings)
    except Exception:
        rscript = None
    if rscript:
        return Check(title, "ok", f"Found R at {rscript}")
    return Check(
        title,
        "fail",
        f"The measurement methods {', '.join(needs)} run in R, and R was not "
        f"found. Without it the study would fail after the first paid AI steps.",
        "edmars setup r",
    )


def preflight(
    plan: StudyPlan, settings: dict | None, *, run_probes: bool = True
) -> list[Check]:
    """The free feasibility check (R4). Any ``fail`` must block the launch.

    Free-text prediction plans have no spec yet (the pipeline writes it),
    so only the question's scope, the dataset x task compatibility and the
    data file are checked. Spec plans additionally go through the
    pipeline's own spec loader and ``src.ideation.feasibility.screen``
    with the Stage-1 data probes; a KILL there is a ``fail`` here. The
    first probe of a large CSV can take minutes (it builds a cache).
    """
    checks: list[Check] = []
    text = plan.research_question or plan.prompt or ""
    if plan.spec is None:
        scope = out_of_scope(text)
        if scope:
            first, _, rest = scope.partition("\n")
            checks.append(
                Check("The question fits what EDM-ARS does", "fail", first,
                      rest.strip() or None)
            )
    note = language_note(text)
    if note:
        checks.append(Check("Language", "info", note))

    if plan.task_type not in TASK_TYPES:
        checks.append(
            Check("Study type", "fail", f"Unknown study type {plan.task_type!r}.",
                  "Choose one of: " + ", ".join(TASK_TYPES))
        )
        return checks
    if plan.spec is None and plan.task_type != "prediction":
        checks.append(
            Check(
                "The study plan is complete",
                "fail",
                f"A {TASK_LABELS[plan.task_type]} study needs a study plan "
                f"(an example or a menu-built plan); without one the pipeline "
                f"would run a prediction study instead.",
                "Run `edmars new`, or pass --example ID or --spec FILE.",
            )
        )
    if plan.spec is None:
        checks.append(_compat_check(plan.dataset, plan.task_type))

    data_check, data_present = _data_check(plan.dataset, settings)
    checks.append(data_check)

    if plan.spec is not None:
        spec = plan.spec
        spec_dataset = spec.get("dataset")
        if spec_dataset and spec_dataset != plan.dataset:
            checks.append(
                Check(
                    "The dataset matches the study plan",
                    "fail",
                    f"The study plan is written for {spec_dataset}, but the run "
                    f"would load {plan.dataset}.",
                    "Leave the dataset to the study plan.",
                )
            )
        if spec.get("task_type") != plan.task_type:
            checks.append(
                Check(
                    "The study type matches the study plan",
                    "fail",
                    f"The study plan says {spec.get('task_type')!r}, but the run "
                    f"is set up as {plan.task_type!r}.",
                    "Leave the study type to the study plan.",
                )
            )
        problems, notes = check_spec(spec, plan.dataset)
        if problems:
            checks.append(
                Check(
                    "The pipeline accepts the study plan",
                    "fail",
                    " ".join(problems),
                    _KILL_FIXES["F-SPEC-INCOMPLETE"],
                )
            )
        else:
            checks.append(
                Check("The pipeline accepts the study plan", "ok",
                      "The pipeline's own study-plan check passed.")
            )
        checks += [Check("Study plan notes", "info", n) for n in notes]

        from src.ideation.feasibility import screen

        report = screen(
            spec,
            candidate_id=str(spec.get("task_id") or plan.example_id or "study"),
            dataset=plan.dataset,
            task_type=plan.task_type,
            registry_dir=registry_dir(),
            raw_data_dir=_raw_data_dir(settings),
            cache_dir=_cache_dir(),
            run_probes=run_probes and data_present,
        )
        checks += _map_report(report)

    if plan.task_type == "psychometrics":
        r_check = _r_check(plan.spec, settings)
        if r_check is not None:
            checks.append(r_check)
    return checks


def blocking(checks: Sequence[Check]) -> bool:
    """True when any check failed; the study must not start."""
    return any(c.status == "fail" for c in checks)


# --------------------------------------------------------------------------
# Plans from command-line flags
# --------------------------------------------------------------------------


def _plan_from_example(ex: ExampleStudy, options: dict[str, Any]) -> StudyPlan:
    spec = copy.deepcopy(ex.spec)
    spec.setdefault("dataset", ex.dataset)
    return StudyPlan(
        task_type=ex.task_type,
        dataset=ex.dataset,
        research_question=ex.research_question,
        prompt=ex.research_question,
        spec=spec,
        example_id=ex.id,
        experimental=False,
        **options,
    )


def _check_agrees(what: str, given: str | None, actual: str, flag: str, source: str) -> None:
    """Refuse a flag that contradicts the spec instead of silently overriding."""
    if given and given != actual:
        raise StudyError(
            f"The {source} uses {what} {actual!r}, but {flag} says {given!r}. "
            f"The {what} always comes from the study plan: drop {flag}, or "
            f"choose a matching example."
        )


def _read_spec_file(spec_path: str | os.PathLike[str]) -> dict:
    path = Path(spec_path).expanduser()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise StudyError(f"No study plan file at {path}.") from exc
    except OSError as exc:
        raise StudyError(f"Could not read {path}: {exc}.") from exc
    except ValueError as exc:
        raise StudyError(f"{path} is not valid JSON: {exc}.") from exc
    if not isinstance(data, dict):
        raise StudyError(f"{path} must contain one JSON object (a study plan).")
    return data


def _matching_example(spec: dict) -> str | None:
    canon = json.dumps(spec, sort_keys=True)
    for ex in EXAMPLES.values():
        candidate = copy.deepcopy(ex.spec)
        candidate.setdefault("dataset", ex.dataset)
        if json.dumps(candidate, sort_keys=True) == canon:
            return ex.id
    return None


def _example_list(task_type: str | None = None) -> str:
    items = [
        ex.id for ex in EXAMPLES.values()
        if task_type is None or ex.task_type == task_type
    ]
    return ", ".join(sorted(items)) or "none"


def plan_from_flags(
    settings: dict | None,
    *,
    task_type: str | None = None,
    prompt: str | None = None,
    example: str | None = None,
    spec_path: str | os.PathLike[str] | None = None,
    dataset: str | None = None,
    venue: str | None = None,
) -> StudyPlan:
    """A plan for ``edmars run`` (no prompts). Raises :class:`StudyError`.

    * ``--example ID`` / ``--spec FILE``: the dataset and study type come
      from the spec; a conflicting ``--dataset``/``--type`` is an error,
      never a silent override.
    * A causal or measurement type without a spec is refused: the
      pipeline would otherwise quietly run a prediction study.
    * Prediction needs ``--prompt``; out-of-scope questions are refused.
    """
    if example and spec_path:
        raise StudyError("Use either --example or --spec, not both.")
    if (example or spec_path) and prompt:
        raise StudyError(
            "--prompt is only for prediction studies; an example or a study "
            "plan file carries its own research question."
        )
    if task_type is not None and task_type not in TASK_TYPES:
        raise StudyError(
            f"Unknown study type {task_type!r}. Choose one of: "
            f"{', '.join(TASK_TYPES)}."
        )
    options = _default_options(settings)
    if venue:
        options["venue"] = normalize_venue(venue)
        if options["venue"] in JOURNAL_VENUES and not _sget(
            settings, "defaults.paper_format"
        ):
            options["paper_format"] = "journal"

    if example:
        ex = find_example(example)
        if ex is None:
            raise StudyError(
                f"No example study called {example!r}. Available: "
                f"{_example_list()}."
            )
        _check_agrees("study type", task_type, ex.task_type, "--type", "example")
        _check_agrees("dataset", dataset, ex.dataset, "--dataset", "example")
        return _plan_from_example(ex, options)

    if spec_path:
        spec = _read_spec_file(spec_path)
        spec_type = spec.get("task_type")
        if spec_type not in TASK_TYPES:
            raise StudyError(
                "The study plan file must name its study type in 'task_type' "
                f"(one of: {', '.join(TASK_TYPES)})."
            )
        _check_agrees("study type", task_type, str(spec_type), "--type", "study plan file")
        if spec.get("dataset"):
            _check_agrees("dataset", dataset, str(spec["dataset"]), "--dataset",
                          "study plan file")
            resolved = str(spec["dataset"])
        else:
            resolved = dataset or DEFAULT_DATASET
            spec = dict(spec, dataset=resolved)
        question = str(spec.get("research_question") or "").strip()
        if not question:
            raise StudyError("The study plan file has no 'research_question'.")
        problems, _ = check_spec(spec, resolved)
        if problems:
            raise StudyError(
                "The study plan file is not a valid study plan:\n  - "
                + "\n  - ".join(problems)
            )
        matched = _matching_example(spec)
        return StudyPlan(
            task_type=str(spec_type),
            dataset=resolved,
            research_question=question,
            prompt=question,
            spec=spec,
            example_id=matched,
            experimental=matched is None,
            **options,
        )

    resolved_type = task_type or "prediction"
    if resolved_type != "prediction":
        raise StudyError(
            f"A {TASK_LABELS[resolved_type]} study needs a tested example "
            f"(--example ID) or a study plan file (--spec FILE); without one the "
            f"pipeline would quietly run a prediction study instead. Examples "
            f"for this type: {_example_list(resolved_type)}. Or run `edmars new` "
            f"and choose 'Build my own'."
        )
    question = str(prompt or "").strip()
    if not question:
        raise StudyError('A prediction study needs a question: --prompt "...".')
    scope = out_of_scope(question)
    if scope:
        raise StudyError(scope)
    resolved = dataset or DEFAULT_DATASET
    reason = unsupported_reason(resolved, resolved_type)
    if reason is not None:
        raise StudyError(f"Cannot run a prediction study on {resolved}: {reason}")
    return StudyPlan(
        task_type="prediction",
        dataset=resolved,
        research_question=question,
        prompt=question,
        **options,
    )


# --------------------------------------------------------------------------
# R6 confirmation card
# --------------------------------------------------------------------------

_PROVIDER_NAMES: dict[str, str] = {
    "deepseek": "DeepSeek",
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "local": "your own model server",
}


def _models_text(settings: dict | None, provider: str) -> str:
    models = _sget(settings, "models", {}) or {}
    if isinstance(models, dict) and models:
        chosen = sorted({str(v) for v in models.values() if v})
        if chosen:
            return "your chosen models (" + ", ".join(chosen) + ")"
    try:
        from edmars import providers

        defaults = providers.default_models(provider)
        names = sorted({str(v) for v in (defaults or {}).values() if v})
        if names:
            return "the recommended models (" + ", ".join(names) + ")"
    except Exception:  # display only: a missing catalog must not block the card
        pass
    return "the recommended models"


def venue_benchmarked(venue: str, settings: dict | None) -> bool:
    """Whether the automated reviewer has a benchmark for ``venue``.

    EDM always does (the calibration file is built from EDM papers). Other
    venues count as benchmarked only when the installed reviewer's
    calibration file carries a threshold for them.
    """
    if venue == "EDM":
        return True
    home = _sget(settings, "lsar.home")
    if not home:
        return False
    calibration = Path(str(home)) / "calibration" / "anchors_edm.yaml"
    try:
        import yaml

        data = yaml.safe_load(calibration.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError, ImportError):
        return False
    block = ((data.get("venues") or {}) if isinstance(data, dict) else {}).get(venue)
    return isinstance(block, dict) and block.get("p25") is not None


def _wrap(label: str, text: str, width: int = 78) -> list[str]:
    head = f"{label:<11}"
    return textwrap.wrap(
        text, width=width, initial_indent=head, subsequent_indent=" " * 11
    ) or [head.rstrip()]


def confirmation_card(
    plan: StudyPlan, settings: dict | None, *, balance: str | None = None
) -> str:
    """The R6 card: what will run, how long, what it costs, what is sent where."""
    provider = str(_sget(settings, "provider", "deepseek"))
    provider_name = _PROVIDER_NAMES.get(provider, provider)
    lines: list[str] = []
    if plan.experimental:
        lines.append(f"{EXPERIMENTAL_BADGE} not a tested example study")
        lines.append("")
    lines += _wrap("Question:", plan.research_question)
    type_text = f"{TASK_LABELS.get(plan.task_type, plan.task_type)} - " + TASK_BLURBS.get(
        plan.task_type, ""
    )
    lines += _wrap("Type:", type_text.rstrip(" -"))
    if plan.example_id:
        source = f"tested example '{plan.example_id}'"
    elif plan.spec is not None and plan.spec.get("note") == MENU_NOTE:
        source = "built with the menus"
    elif plan.spec is not None:
        source = "your own study plan file"
    else:
        source = "the AI writes the study plan from your question"
    if plan.experimental:
        source += f" {EXPERIMENTAL_BADGE}"
    lines += _wrap("Plan:", source)
    lines += _wrap("Data:", f"{_dataset_label(plan.dataset)} ({plan.dataset})")
    venue_name = VENUES.get(plan.venue, plan.venue)
    lines += _wrap("Paper:", f"{venue_name}, {plan.paper_format} format")
    if plan.review:
        bench = (
            "benchmarked against papers published at this venue"
            if venue_benchmarked(plan.venue, settings)
            else "score only - there is no benchmark for this venue yet"
        )
        lines += _wrap(
            "Review:",
            f"automated peer review (LSAR) on - {bench}. Scores vary by about "
            f"2 points between runs.",
        )
    else:
        lines += _wrap("Review:", "automated peer review off")
    ai = f"{provider_name} with {_models_text(settings, provider)}"
    if provider == "local":
        base = _sget(settings, "provider_base_url")
        ai += f" at {base}" if base else ""
    lines += _wrap("AI:", ai)
    lines += _wrap("Time:", TIME_WITH_REVIEW if plan.review else TIME_WITHOUT_REVIEW)
    if provider == "deepseek":
        cost = COST_DEEPSEEK
    elif provider == "local":
        cost = COST_LOCAL
    else:
        cost = COST_OTHER
    lines += _wrap("Cost:", cost)
    if balance:
        lines += _wrap("Balance:", str(balance))
    budget = _sget(settings, "defaults.budget_usd")
    if isinstance(budget, (int, float)) and budget > 0:
        lines += _wrap("Budget:", f"warn above US${budget:.2f}")
    studies = _sget(settings, "studies_dir")
    if studies:
        lines += _wrap("Saved in:", f"a new folder inside {studies}")

    lines += ["", "What is sent where"]
    sent_to = (
        "your own model server" if provider == "local" else provider_name
    )
    bullets = [
        f"To {sent_to}: your question, variable names and summary statistics, "
        "the analysis code the AI writes, its error messages and printed output "
        "(which can include a few individual data values), and the draft paper.",
        "To Semantic Scholar, arXiv and Crossref: search words from your question.",
    ]
    if plan.review:
        bullets.append(
            "To DeepSeek, for the automated review: the finished paper's text."
        )
    bullets.append(
        "The dataset file itself is never uploaded. EDM-ARS has no telemetry. "
        "Details: `edmars privacy`."
    )
    for bullet in bullets:
        lines += textwrap.wrap(
            bullet, width=78, initial_indent=" - ", subsequent_indent="   "
        )
    if plan.experimental:
        lines += [""] + textwrap.wrap(
            f"{EXPERIMENTAL_BADGE} {EXPERIMENTAL_TEXT}", width=78
        )
    return "\n".join(lines)
