"""Deterministic pre-Critic validation checks for EDM-ARS.

Inspired by AutoResearchClaw health.py: a fast, zero-LLM validation layer that
catches obvious pipeline failures before the expensive Critic (Opus) call is made.

If critical failures are found, the Orchestrator short-circuits and synthesises a
REVISE/ABORT review_report without burning an Opus API call.

What a critical finding does
----------------------------
Every critical finding says whether a revision can fix it (``revisable``).
A revisable one sends its ``revision_instruction`` to ``target_agent``
through the ordinary REVISING cascade (SPEC §5.3); if it is still failing
when the revision cycles run out, the run stops with PRE_CRITIC_UNRESOLVED
and no paper is written. It stops the same way sooner when the revision
comes back with the agent's own word that another would not help
(``stop_on_repeat``: its not-run record with a reason, or a second
timeout). Any finding that is not revisable stops the run at
once with PRE_CRITIC_ABORT. A check that sets nothing is not revisable, so
a new critical check keeps the old stop-the-run behaviour until someone
decides otherwise.

    check   finding                                    on failure  target
    pcc_01  outcome is a column of train_X.csv         stop        DataEngineer
    pcc_06  data_report.validation_passed is False     stop        DataEngineer
    pcc_02  no individual model in results.json        revise      Analyst
    pcc_07  the analysis the question promises is      revise      Analyst
            missing from results.json

Why: SPEC §4.4 lists the ABORT conditions as a fundamental flaw
(unanswerable question, analytic_n < 1,000, confirmed leakage) and SPEC §8
aborts on ``validation_passed == false``. pcc_01 is confirmed leakage by
construction and pcc_06 is the §8 condition, on which the ENGINEERING
stage also stops after its one targeted retry. pcc_02 and pcc_07 are
neither: the data and the question are sound. pcc_07 means the Analyst
left out an analysis it can run on the same files (the helpers exist);
pcc_02 means its generated code failed on every attempt, which a fresh
Analyst run starts over from. Stopping on either throws away a study a
revision could finish. The ``major`` checks never short-circuit; the
Critic reads them.
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, NamedTuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CheckFailure:
    check_id: str
    severity: str  # "critical" | "major"
    message: str
    target_agent: str  # "ProblemFormulator" | "DataEngineer" | "Analyst"
    #: Critical findings only: True when re-running ``target_agent`` with
    #: ``revision_instruction`` can clear the finding. False stops the run.
    revisable: bool = False
    #: What ``target_agent`` must do, in terms it can act on. ``message``
    #: says what is wrong; this says what to change. Empty means the
    #: message is the instruction.
    revision_instruction: str = ""
    #: Revisable findings only: set when the target agent's own output
    #: says that running it again will not clear the finding -- its
    #: not-run record with a reason (what the instruction tells it to
    #: leave when the test truly cannot run), or a time limit hit. The
    #: orchestrator acts on it only when the previous cycle already sent
    #: this check back, so the agent always gets one revision first; after
    #: that it stops the run instead of repeating the same instruction.
    stop_on_repeat: str = ""

    @property
    def instruction(self) -> str:
        return self.revision_instruction or self.message

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "severity": self.severity,
            "message": self.message,
            "target_agent": self.target_agent,
            "revisable": bool(self.revisable),
        }


@dataclass
class PreCriticResult:
    failures: list[CheckFailure] = field(default_factory=list)

    @property
    def has_critical(self) -> bool:
        return any(f.severity == "critical" for f in self.failures)

    @property
    def fatal_failures(self) -> list[CheckFailure]:
        """Critical findings no revision can fix; any one stops the run."""
        return [
            f for f in self.failures
            if f.severity == "critical" and not f.revisable
        ]

    @property
    def revisable_failures(self) -> list[CheckFailure]:
        """Critical findings a targeted agent re-run can fix."""
        return [
            f for f in self.failures
            if f.severity == "critical" and f.revisable
        ]

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.failures if f.severity == "critical")

    @property
    def major_count(self) -> int:
        return sum(1 for f in self.failures if f.severity == "major")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


_SUPPORTED_TASK_TYPES: frozenset[str] = frozenset({"prediction", "causal_soo", "causal_itr", "causal_did", "psychometrics"})


def run_pre_critic_checks(
    ctx: object,
    output_dir: str,
    task_type: str = "prediction",
) -> PreCriticResult:
    """Run all deterministic pre-Critic checks and return a :class:`PreCriticResult`.

    Parameters
    ----------
    ctx:
        The ``PipelineContext`` object (typed as ``object`` to avoid circular import).
        Must expose ``.research_spec``, ``.results_object``, and ``.data_report`` attrs.
    output_dir:
        Absolute path to the run's output directory.
    task_type:
        Phase 3b.6 / 6.2 — gates which structural checks fire. Prediction-
        specific checks (SHAP figures, top_features, all_models, subgroup
        AUC) are skipped under ``task_type='causal_soo'`` because the
        causal pipeline does not produce those artifacts. Universal checks
        (target leakage in train_X, data_report.validation_passed) fire
        regardless of task type.

    Raises
    ------
    ValueError
        If ``task_type`` is not in :data:`_SUPPORTED_TASK_TYPES`. We fail
        loudly rather than silently running the prediction-shaped checks
        for unknown types — the prior implicit fall-through is exactly
        what produced 3b.5's F-PRECRITIC-PREDICTION pollution.
    """
    if task_type not in _SUPPORTED_TASK_TYPES:
        raise ValueError(
            f"run_pre_critic_checks: unknown task_type {task_type!r}. "
            f"Supported: {sorted(_SUPPORTED_TASK_TYPES)}. "
            f"Add a structural-check policy for the new task type rather "
            f"than falling through to prediction-shaped checks."
        )
    result = PreCriticResult()

    # Universal checks (run for every task type)
    _check_outcome_not_in_train_x(ctx, output_dir, result)
    _check_data_report_validation_passed(ctx, result)
    _check_research_question_is_answered(ctx, result, task_type=task_type)

    if task_type == "prediction":
        # Prediction-shaped structural checks: model battery + SHAP +
        # top_features + subgroup AUC. None of these apply to causal
        # output; they fire as false-positive [major] issues under
        # causal_soo (per 3b.5 evidence — F-PRECRITIC-PREDICTION).
        _check_model_count(ctx, result)
        _check_required_figures(output_dir, result)
        _check_top_features_present(ctx, result)
        _check_subgroup_performance_present(ctx, result)
    elif task_type == "psychometrics":
        # Measurement runs: no split/refuter/model-battery checks; the
        # psychometrics-measurement-protocol skill carries the critic
        # rows (psy_01..). No structural pre-checks yet.
        pass
    elif task_type == "causal_did":
        # DiD has no propensity model -> the DoWhy refuter contract
        # (pcc_c01) does not apply; the M8 helpers carry their own
        # placebo probe. No causal_did-specific structural checks yet.
        pass
    elif task_type in ("causal_soo", "causal_itr"):
        # 3b.6 deliberately ran NO causal-specific structural checks.
        # V4 Arc H (3b.23.7) adds the first one: the refuter-attempt
        # assertion (pcc_c01), after 3b.23.5 shipped a paper whose
        # sensitivity package silently omitted the mandatory DoWhy
        # refuters. The fuller causal pre-critic checklist
        # (F-CAUSAL-PRECRITIC: balance, positivity, estimand gates)
        # remains future work — the Critic carries that load via the
        # injected G1-G5 + D1 skills.
        _check_refuters_attempted(ctx, result)

    return result


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


#: Phrases in a research question that COMMIT the paper to a specific
#: analysis, mapped to the evidence that analysis leaves behind.
#:
#: A9. The ordinal-attainment paper's entire premise was whether modelling
#: attainment as ordered beats treating it as nominal. No ordinal model
#: was ever fitted -- no proportional-odds, no ordinal forest -- and LSAR
#: scored it 2/10 on Methodological Rigor and rejected it. Two other
#: papers showed the same pattern more mildly: a stated
#: incremental-validity test that was never directly implemented, and no
#: nested-model comparison.
#:
#: Deliberately short. Each entry is a phrase whose presence makes a
#: specific claim the reader will expect to see supported, paired with
#: keys or substrings that show the work was actually done. Vague phrases
#: are NOT listed: a check that fires on "explore" or "examine" would be
#: noise, and noise is how a check gets ignored.
#:
#: ``strict`` commitments accept only a record of the analysis having run
#: (see :func:`_strict_evidence`). Only the incremental-validity
#: promise is strict: the skill contract lets a moderation or calibration
#: analysis be recorded as skipped and descoped in the Limitations, and
#: the Analyst prompt tells regression runs to do exactly that for
#: calibration, so making those strict would stop runs the contract
#: allows.
class _Commitment(NamedTuple):
    kind: str
    phrases: tuple[str, ...]
    evidence_keys: tuple[str, ...]
    description: str
    strict: bool = False


_RQ_COMMITMENTS: tuple[_Commitment, ...] = (
    _Commitment(
        "ordinal",
        ("ordinal", "ordered categor", "proportional odds", "ordered logit"),
        ("ordinal", "proportional_odds", "polr", "ordered"),
        "an ordinal model (proportional-odds or ordinal forest)",
    ),
    _Commitment(
        "incremental",
        ("above and beyond", "over and above", "incremental valid",
         "incremental predictive", "beyond baseline"),
        ("incremental_validity", "nested_model", "delta_auc", "delta_r2"),
        "an incremental-validity / nested-model comparison",
        strict=True,
    ),
    _Commitment(
        "mediation",
        ("mediat",),
        ("mediation", "indirect_effect"),
        "a mediation analysis",
    ),
    _Commitment(
        "moderation",
        ("moderat", "interaction effect", "varies by", "vary by"),
        ("moderation", "interaction", "subgroup_heterogeneity",
         "subgroup_performance"),
        "a moderation / interaction analysis",
    ),
    _Commitment(
        "calibration",
        ("calibrat",),
        ("calibration",),
        "a calibration analysis",
    ),
)

#: ``status`` values with which a record says its analysis RAN
#: (run_incremental_validity returns "ok", run_moderation_analysis
#: "computed"). A strict commitment accepts only these. It used to reject
#: a list of not-run statuses instead, and every shape off the list
#: passed: ``null`` next to a "run_incremental_validity failed" warning
#: (what an archived GPA run wrote), ``{}``, a bare string,
#: "not_applicable", a record with a reason and no status. The ways to
#: say "did not run" are open-ended; the ways to say "ran" are not.
_COMPUTED_STATUSES: frozenset[str] = frozenset({"ok", "computed"})

#: Fields that hold the nested-model difference itself. A record without
#: a status counts when one of them is a number, and an "ok" record whose
#: difference is null does not count.
_DELTA_FIELDS: tuple[str, ...] = ("delta_auc", "delta_r2")

_LOG_KEYS: frozenset[str] = frozenset({"warnings", "errors"})


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _status_of(record: dict) -> str | None:
    status = record.get("status")
    if status is None:
        return None
    return str(status).strip().lower()


def _is_computed_record(value: Any) -> bool:
    """True when ``value`` records an analysis that ran."""
    if isinstance(value, list):
        return any(_is_computed_record(v) for v in value)
    if not isinstance(value, dict):
        return False
    deltas = [value[k] for k in _DELTA_FIELDS if k in value]
    status = _status_of(value)
    if status is None:
        return any(_is_number(d) for d in deltas)
    return status in _COMPUTED_STATUSES and all(_is_number(d) for d in deltas)


def _strict_evidence(value: Any, keys: tuple[str, ...], top: bool = True) -> bool:
    """Whether ``value`` holds, at any depth, a record of the analysis.

    A key that names the analysis (``incremental_validity``,
    ``nested_model_comparison``) counts only when it holds a record that
    ran (:func:`_is_computed_record`); a difference field
    (``delta_auc``, ``delta_r2``) counts only as a finite number. A key
    that merely contains the name (``incremental_validity_reason``) holds
    a string, so it does not count. Nothing inside a record whose status
    says it did not run counts, and neither do ``warnings`` and
    ``errors``: a sentence reporting that the comparison was skipped
    names the key without being the comparison.
    """
    if isinstance(value, list):
        return any(_strict_evidence(v, keys, top=False) for v in value)
    if not isinstance(value, dict):
        return False
    for key, item in value.items():
        name = str(key).strip().lower()
        if top and name in _LOG_KEYS:
            continue
        if name in _DELTA_FIELDS:
            if name in keys and _is_number(item):
                return True
            continue
        if any(name.startswith(k) for k in keys if k not in _DELTA_FIELDS):
            if _is_computed_record(item):
                return True
            continue
        if isinstance(item, dict):
            status = _status_of(item)
            if status is not None and status not in _COMPUTED_STATUSES:
                continue
        if _strict_evidence(item, keys, top=False):
            return True
    return False


def _declined_reason(results: dict, keys: tuple[str, ...]) -> str:
    """The reason in a not-run record under one of ``keys``, or "".

    This is the Analyst saying the analysis cannot run: the instruction
    tells it to leave the helper's skipped record, with its reason, when
    the test truly cannot run on its files.
    """
    names = [k for k in keys if k not in _DELTA_FIELDS]

    def walk(value: Any, top: bool) -> str:
        if isinstance(value, list):
            for item in value:
                found = walk(item, False)
                if found:
                    return found
            return ""
        if not isinstance(value, dict):
            return ""
        for key, item in value.items():
            name = str(key).strip().lower()
            if top and name in _LOG_KEYS:
                continue
            if isinstance(item, dict) and any(name.startswith(k) for k in names):
                status = _status_of(item)
                reason = item.get("reason")
                if (status is not None and status not in _COMPUTED_STATUSES
                        and isinstance(reason, str) and reason.strip()):
                    return " ".join(reason.split())[:300]
            found = walk(item, False)
            if found:
                return found
        return ""

    return walk(results, True)


def _evidence_text(results: dict) -> str:
    """The lower-cased text pcc_07 searches for a non-strict commitment.

    Evidence may sit at any depth (results.incremental_validity,
    results.all_models["OrdinalForest"], a key inside a sub-dict), so the
    serialised object is searched rather than a fixed set of top-level
    keys.
    """
    try:
        return json.dumps(results).lower()
    except (TypeError, ValueError):
        return str(results).lower()


def _check_research_question_is_answered(
    ctx: object, result: PreCriticResult, task_type: str = "prediction"
) -> None:
    """pcc_07 (critical, revisable): the analysis must contain the test the
    RQ promises.

    This is the single largest driver of low rigor scores that is
    genuinely the system's fault. A paper whose central question is never
    tested still compiles, still scores, and still reads fluently -- the
    absence is invisible in the artifact and obvious to a reviewer.

    Revisable, and aimed at the Analyst: the question and the data are
    sound, and the missing analysis runs on the files the Analyst already
    has. Rewording the question instead would restart the whole
    ProblemFormulator -> DataEngineer -> Analyst cascade to delete the
    paper's contribution; the instruction says how to run the test.
    """
    spec = getattr(ctx, "research_spec", None) or {}
    if not isinstance(spec, dict):
        return
    original = str(spec.get("research_question") or "")
    question = original.lower()
    if not question:
        return

    results = getattr(ctx, "results_object", None) or {}
    if not isinstance(results, dict):
        return

    haystack = _evidence_text(results)
    for commitment in _RQ_COMMITMENTS:
        phrases = commitment.phrases
        description = commitment.description
        if not any(p in question for p in phrases):
            continue
        if commitment.strict:
            if _strict_evidence(results, commitment.evidence_keys):
                continue
        elif any(k.lower() in haystack for k in commitment.evidence_keys):
            continue
        matched = next(p for p in phrases if p in question)
        declined = (
            _declined_reason(results, commitment.evidence_keys)
            if commitment.strict else ""
        )
        result.failures.append(
            CheckFailure(
                check_id="pcc_07",
                severity="critical",
                message=(
                    f"The research question says {matched!r}, which commits "
                    f"the paper to {description}, but no such analysis "
                    f"appears in results.json. Either run it, or change the "
                    f"research question so it does not promise a test the "
                    f"study never performed. A paper whose central question "
                    f"is never tested reads fluently and is rejected on "
                    f"rigor."
                ),
                target_agent="Analyst",
                revisable=True,
                revision_instruction=_commitment_instruction(
                    commitment, matched, original, ctx, task_type
                ),
                stop_on_repeat=(
                    f"the Analyst recorded that the test cannot run: {declined}"
                    if declined else ""
                ),
            )
        )


#: Where the clause holding a question's named baseline ends: the next
#: clause ("... above and beyond achievement and SES, and does ..."), or
#: the verb the baseline was inserted before ("..., over and above prior
#: achievement, predict ..."). A comma is NOT an end. Until 2026-09-27
#: every "," and "(" was, so "above and beyond academic achievement, SES,
#: and demographic controls" reached the Analyst as the baseline
#: "academic achievement", with SES and the controls left for the focal
#: block, where the comparison would credit their predictive power to
#: the constructs the question is about.
_CLAUSE_END = re.compile(
    r"[;?]|\.(?=\s|$)"
    r"|,?\s(?:and|or)\s+(?:does|do|did|is|are|was|were|can|could|will|would"
    r"|how|whether|to what|which|what)\b"
    r"|\s(?:(?:in|when|for)\s+)?(?:explains?|explaining|predicts?|predicting"
    r"|accounts? for)\b",
    re.IGNORECASE,
)

#: A comma-separated part that starts a new clause rather than naming one
#: more baseline item.
_NOT_A_LIST_ITEM = re.compile(
    r"^(?:(?:and|or)\s+)?(?:which|who|whose|that|while|whereas|controlling"
    r"|net of|after|among|within|across|when|where|because|since|although"
    r"|using|does|do|did|is|are|was|were|can|could|will|would|how|whether"
    r"|what|predicts?|predicting|explains?|explaining|accounts?|improves?"
    r"|adds?|contributes?|remains?|holds?)\b",
    re.IGNORECASE,
)

_CLOSES_A_LIST = re.compile(r"^(?:and|or)\s|\s(?:and|or)\s", re.IGNORECASE)

#: The words right before a named baseline. "incremental valid" is a
#: commitment phrase but names nothing after it ("the incremental
#: validity of X" used to give the baseline "ity of X").
_BASELINE_MARKER = re.compile(
    r"\b(?:above and beyond|over and above|beyond|relative to"
    r"|compared (?:with|to))\s+",
    re.IGNORECASE,
)


def _paren_depths(text: str) -> list[int]:
    depth, out = 0, []
    for ch in text:
        if ch == "(":
            depth += 1
        out.append(depth)
        if ch == ")" and depth:
            depth -= 1
    return out


def _named_after(question: str, phrase: str) -> str:
    """The baseline a question names after ``phrase``.

    "... ABOVE AND BEYOND academic achievement and socioeconomic status,
    and does ..." -> "academic achievement and socioeconomic status";
    "... above and beyond prior achievement (X1TXMTSCOR), SES (X1SES), and
    sex?" -> the whole list, parentheses included, since they often carry
    the variable names. A comma continues the baseline only inside a list
    that closes with "and"/"or"; otherwise it ends it. Empty when the
    question names no baseline ("the incremental validity of X").
    """
    at = question.lower().find(phrase)
    if at < 0:
        return ""
    marker = _BASELINE_MARKER.search(question, at)
    if marker is None:
        return ""
    tail = question[marker.end():]
    tail = re.sub(r"^\s*(?:what|that which|those of|the effects? of)\s+", "",
                  tail, flags=re.IGNORECASE)
    depths = _paren_depths(tail)
    cut = next(
        (m.start() for m in _CLAUSE_END.finditer(tail) if depths[m.start()] == 0),
        len(tail),
    )
    head = tail[:cut]
    depths = depths[:cut]
    parts: list[str] = []
    start = 0
    for i, ch in enumerate(head):
        if ch == "," and depths[i] == 0:
            parts.append(head[start:i].strip())
            start = i + 1
    parts.append(head[start:].strip())

    named = [parts[0]]
    rest = parts[1:]
    for j, part in enumerate(rest):
        if not part or _NOT_A_LIST_ITEM.match(part):
            break
        if _CLOSES_A_LIST.search(part):
            named.extend(rest[: j + 1])
            break
    return ", ".join(named).strip(" ,;:")[:240]


def _outcome_type(ctx: object) -> str:
    for source in ("data_report", "research_spec"):
        block = getattr(ctx, source, None)
        if isinstance(block, dict) and isinstance(block.get("outcome_type"), str):
            return block["outcome_type"].strip().lower()
    return ""


def _predictor_names(ctx: object, limit: int = 40) -> list[str]:
    spec = getattr(ctx, "research_spec", None)
    entries = spec.get("predictor_set") if isinstance(spec, dict) else None
    names: list[str] = []
    for entry in entries or []:
        name = entry.get("variable") if isinstance(entry, dict) else entry
        if isinstance(name, str) and name and name not in names:
            names.append(name)
    return names[:limit]


#: How an instruction for a commitment the skill contract lets be
#: descoped ends: a skipped record with its reason passes pcc_07 and
#: reaches the methods review.
_DESCOPE_ENDING = (
    "If it truly cannot run on these files, record {'status': 'skipped', "
    "'reason': <why>} under that key and say so in the paper's "
    "Limitations; the methods review then judges whether the question "
    "still stands. Keep every other part of the analysis as it was."
)

#: How the incremental-validity instruction ends. It says what the
#: orchestrator does: a not-run record with a reason, returned by the
#: revision this instruction ordered, stops the study at once
#: (stop_on_repeat); anything else not run is sent back while revision
#: cycles remain.
_STRICT_ENDING = (
    "Only a record with 'status': 'ok' and its numbers answers the "
    "question. null, an empty record, a string, a 'skipped' or 'error' "
    "record, or a sentence in warnings does not. If the helper returned "
    "'skipped' or 'error', fix what its reason names and call it again. If "
    "the test truly cannot run on these files, leave the helper's record, "
    "with its reason, under results['incremental_validity']: the study then "
    "stops without a paper instead of publishing a question it never "
    "tested. Keep every other part of the analysis as it was."
)


def _commitment_instruction(
    commitment: _Commitment,
    matched: str,
    question: str,
    ctx: object,
    task_type: str,
) -> str:
    """The concrete work pcc_07 asks the Analyst to add, per commitment."""
    prediction = task_type == "prediction"
    if commitment.kind == "incremental":
        return _incremental_instruction(matched, question, ctx, prediction)
    if commitment.kind == "ordinal" and prediction:
        return (
            f"The research question promises {commitment.description} "
            f"({matched!r}). Fit a proportional-odds model on the ordered "
            "outcome with statsmodels' OrderedModel (distr='logit') on the "
            "same train/test split and predictors as the other models, "
            "evaluate it on the held-out test set next to the nominal "
            "models, and record it as results['ordinal_model'] = {'status': "
            "'ok', <the same metrics as all_models>}. " + _DESCOPE_ENDING
        )
    if commitment.kind == "moderation" and prediction:
        return (
            f"The research question promises {commitment.description} "
            f"({matched!r}). Call analysis_helpers.run_moderation_analysis("
            "X=X_all, y=y_all, focal_cols=[encoded focal columns], "
            "moderator_col=<the encoded moderator the question names>) as "
            "prediction-rigor-extensions section 1 shows, and record the "
            "return value as results['moderation_analysis']. "
            + _DESCOPE_ENDING
        )
    if commitment.kind == "calibration" and prediction:
        return (
            f"The research question promises {commitment.description} "
            f"({matched!r}). Record results['calibration'] = "
            "analysis_helpers.compute_calibration_metrics(y_true=test_y_arr, "
            "y_prob=<best model's held-out probabilities>); never compute "
            "those fields by hand. " + _DESCOPE_ENDING
        )
    if commitment.kind == "mediation":
        return (
            f"The research question promises {commitment.description} "
            f"({matched!r}). Estimate the indirect effect through the "
            "mediator the question names (product of the a and b paths), "
            "with a 1000-resample bootstrap 95% CI (random_state=42; "
            "resample schools when school IDs exist), and record "
            "results['mediation'] = {'status': 'ok', 'mediator': ..., "
            "'indirect_effect': ..., 'ci_lower': ..., 'ci_upper': ...}. "
            + _DESCOPE_ENDING
        )
    return (
        f"The research question promises {commitment.description} "
        f"({matched!r}). Run that analysis on the existing analysis files "
        "and record it in results.json under a key that names it. "
        + _DESCOPE_ENDING
    )


def _incremental_instruction(
    matched: str, question: str, ctx: object, prediction: bool
) -> str:
    baseline = _named_after(question, matched)
    names = _predictor_names(ctx)
    outcome_type = _outcome_type(ctx)
    lines = [
        f"The research question promises an incremental-validity test "
        f"({matched!r}): the focal predictors must be shown to add "
        f"predictive power over a baseline block, in a nested-model "
        f"comparison on the held-out test set. A SHAP ranking inside one "
        f"model does not show this.",
    ]
    if baseline:
        lines.append(
            f"The words after {matched!r} in the question name the "
            f"baseline: \"{baseline}\". Check that against the question "
            "itself. baseline_cols = the encoded train_X columns of every "
            "predictor_set variable that measures something in that "
            "baseline, controls included. focal_cols = the encoded columns "
            "of only the constructs the question credits. Put no baseline "
            "or control variable in focal_cols: a column there is credited "
            "to the focal constructs, while a column in neither list is "
            "left out of both models. A one-hot variable's columns start "
            "with its name (X1RACE -> X1RACE_*)."
        )
    else:
        lines.append(
            "The question does not name the baseline block in words: "
            "focal_cols = the encoded columns of only the constructs the "
            "question credits; omit baseline_cols, which then defaults to "
            "every other column."
        )
    if names:
        lines.append("predictor_set: " + ", ".join(names) + ".")
    if prediction and outcome_type in ("binary", "continuous", ""):
        type_arg = (
            f"outcome_type={outcome_type!r}" if outcome_type
            else "outcome_type=<'binary' or 'continuous', from data_report.json>"
        )
        returns = {
            "binary": "baseline_auc, full_auc and delta_auc",
            "continuous": "baseline_r2, full_r2, delta_r2 and both RMSEs",
        }.get(
            outcome_type,
            "delta_auc for a binary outcome or delta_r2 for a continuous one",
        )
        lines.append(
            "Call the certified helper, which handles binary and continuous "
            "outcomes; do not reimplement it:\n"
            "    results['incremental_validity'] = "
            "analysis_helpers.run_incremental_validity(\n"
            "        train_X, train_y_arr, test_X, test_y_arr,\n"
            "        focal_cols=focal_cols, baseline_cols=baseline_cols,\n"
            "        school_ids=test_school_ids,  # pseudo_school_id from "
            "test_school_ids.csv; None if that file is absent\n"
            f"        {type_arg})\n"
            f"It returns {returns}, with a bootstrap CI on the difference. "
            "It does not raise on data it cannot use; it returns a record "
            "with a status and a reason, so do not wrap it in a try/except "
            "that writes null or a note instead."
        )
    elif prediction:
        lines.append(
            f"run_incremental_validity handles binary and continuous "
            f"outcomes; this outcome is {outcome_type}. Fit the nested pair "
            "(baseline_cols, then baseline_cols + focal_cols) with the "
            "study's own estimator on the training set, compare their "
            "held-out scores, bootstrap the difference over the test rows "
            "(1000 resamples, random_state=42; resample schools when "
            "test_school_ids.csv exists) and record "
            "results['incremental_validity'] = {'status': 'ok', the two "
            "scores, their difference, 'ci_lower', 'ci_upper', "
            "'baseline_cols', 'focal_cols'}."
        )
    else:
        lines.append(
            "Fit the nested pair (baseline, then baseline + focal) with the "
            "study's own estimator, compare them on the held-out data or "
            "with a likelihood-ratio test, and record the difference and "
            "its 95% CI as results['incremental_validity'] with "
            "'status': 'ok'."
        )
    lines.append(_STRICT_ENDING)
    return "\n".join(lines)


def _check_refuters_attempted(ctx: object, result: PreCriticResult) -> None:
    """pcc_c01 (major, causal_soo only): DoWhy refuters must be ATTEMPTED.

    The causal-sensitivity-unmeasured-confounding skill makes refuter
    invocation unconditional: ``sensitivity.dowhy_refuters`` must exist
    with at least two refuter entries, each carrying a ``status`` field
    ("ran" or "failed"). Failure is acceptable when documented
    (fallback), silence is not (F-3b23.5 shipped a paper with the key
    absent entirely; 3b.19's healthy shape has per-refuter dicts with
    status="ran").
    """
    results = getattr(ctx, "results_object", None) or {}
    sensitivity = results.get("sensitivity") or {}
    refuters = (
        sensitivity.get("dowhy_refuters")
        if isinstance(sensitivity, dict)
        else None
    )

    def _fail(message: str) -> None:
        result.failures.append(
            CheckFailure(
                check_id="pcc_c01",
                severity="major",
                message=message,
                target_agent="Analyst",
            )
        )

    if not isinstance(refuters, dict) or not refuters:
        _fail(
            "sensitivity.dowhy_refuters is absent or empty — the DoWhy "
            "refuters were never attempted. Refuter invocation is "
            "unconditional for causal_soo (attempt-and-document; a "
            "documented failure is acceptable, silence is not). See "
            "causal-sensitivity-unmeasured-confounding §Refuter "
            "execution status contract."
        )
        return

    entries = {
        name: entry
        for name, entry in refuters.items()
        if isinstance(entry, dict)
    }
    if len(entries) < 2:
        _fail(
            f"sensitivity.dowhy_refuters has {len(entries)} refuter "
            f"entry(ies); the skill requires at least two refuters "
            f"attempted (e.g. random_common_cause + "
            f"placebo_treatment_refuter)."
        )
        return

    missing_status = [
        name
        for name, entry in entries.items()
        if entry.get("status") not in ("ran", "failed")
    ]
    if missing_status:
        _fail(
            f"Refuter entry(ies) {missing_status} lack a valid status "
            f"('ran' | 'failed') — attempts must be documented with "
            f"their outcome; a failed refuter records status='failed' "
            f"plus the error text."
        )


def _check_outcome_not_in_train_x(
    ctx: object, output_dir: str, result: PreCriticResult
) -> None:
    """pcc_01 (critical): outcome variable must NOT appear as a column in train_X.csv."""
    spec = getattr(ctx, "research_spec", None) or {}
    outcome = spec.get("outcome_variable", "")
    if not outcome:
        return

    train_x_path = os.path.join(output_dir, "train_X.csv")
    if not os.path.exists(train_x_path):
        return  # Missing file is caught by DataEngineer validation; not duplicated here

    try:
        with open(train_x_path, newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            headers = next(reader, [])
        if outcome in headers:
            result.failures.append(
                CheckFailure(
                    check_id="pcc_01",
                    severity="critical",
                    message=(
                        f"Outcome variable '{outcome}' found as a column in train_X.csv "
                        "— confirmed target leakage."
                    ),
                    target_agent="DataEngineer",
                    # Not revisable: SPEC §4.4 names confirmed leakage as
                    # an ABORT condition, and every model, metric and SHAP
                    # value downstream was fitted with the answer as input.
                    revisable=False,
                )
            )
    except OSError:
        pass  # Can't read file — not a pre-critic error, pipeline will surface it


#: How a run out of wall-clock time reads in results.errors: the
#: executor's "Timeout after 600s", or "timed out".
_TIMEOUT = re.compile(r"\btimed?[\s-]?out\b", re.IGNORECASE)


def _check_model_count(ctx: object, result: PreCriticResult) -> None:
    """pcc_02 (major): results.json must have at least 4 individual models."""
    results = getattr(ctx, "results_object", None) or {}
    all_models: dict = results.get("all_models") or {}
    # StackingEnsemble is not an individual model
    stacking_keys = {k for k in all_models if "stack" in k.lower()}
    individual_count = len(all_models) - len(stacking_keys)
    if individual_count == 0:
        # Not "too few models" -- the analysis did not happen. As a major
        # finding it went to the Critic, whose REVISE spends its cycles
        # and then writes an UNVERIFIED paper about an empty results
        # object, which is the artifact this whole guard exists to
        # prevent. So it is critical: the paper is never written.
        #
        # Critical and revisable. An empty battery is what the Analyst
        # returns when its generated code failed on every attempt or timed
        # out -- an execution failure, not a flaw in the question or the
        # data, and not one of the SPEC's ABORT conditions. A fresh
        # Analyst run starts from new code and is worth its cost; if the
        # battery is still empty after the last revision cycle the run
        # stops (PRE_CRITIC_UNRESOLVED) and still writes no paper.
        errors = results.get("errors") if isinstance(results, dict) else None
        recorded = "; ".join(str(e) for e in (errors or [])[:2])
        # A wall-clock timeout is not a failing model, and wrapping each
        # model in try/except does not make the code finish in time. The
        # instruction says so, and a second timeout after that revision
        # stops the run instead of spending the remaining cycles, each up
        # to four executions at the full time limit, on the same outcome.
        timed_out = next(
            (str(e) for e in (errors or []) if _TIMEOUT.search(str(e))), ""
        )
        timeout_advice = (
            " The code ran out of time: every model then failed together, "
            "and a try/except does not change that. Keep the battery and "
            "its grids, but spend less time on them: run grid searches "
            "with n_jobs=-1, never fit the same model twice, and keep SHAP, "
            "bootstrap and permutation work at the sample sizes the skills "
            "give."
            if timed_out else ""
        )
        result.failures.append(
            CheckFailure(
                check_id="pcc_02",
                severity="critical",
                message=(
                    "No individual models are present in results.json. The "
                    "analysis stage produced no trained models at all, so "
                    "there are no results to report on."
                ),
                target_agent="Analyst",
                revisable=True,
                revision_instruction=(
                    "The previous analysis produced no trained model: "
                    "results.json all_models has no individual model."
                    + (f" It recorded: {recorded[:600]}" if recorded else "")
                    + " Write the analysis again from the start. Fit and "
                    "evaluate each model in the battery inside its own "
                    "try/except that appends the error to results['errors'] "
                    "and continues with the next model (SPEC section 8), so "
                    "one failing model cannot empty the battery, and write "
                    "results.json even when some models fail."
                    + timeout_advice
                ),
                stop_on_repeat=(
                    f"the analysis code ran out of time ({' '.join(timed_out.split())[:300]})"
                    if timed_out else ""
                ),
            )
        )
    elif individual_count < 4:
        result.failures.append(
            CheckFailure(
                check_id="pcc_02",
                severity="major",
                message=(
                    f"Only {individual_count} individual model(s) found in results.json "
                    "(minimum 4 required: LR, RF, XGBoost, ElasticNet, MLP)."
                ),
                target_agent="Analyst",
            )
        )


def _check_required_figures(output_dir: str, result: PreCriticResult) -> None:
    """pcc_03 (major): shap_summary.png and shap_importance.png must exist."""
    for fig in ("shap_summary.png", "shap_importance.png"):
        if not os.path.exists(os.path.join(output_dir, fig)):
            result.failures.append(
                CheckFailure(
                    check_id="pcc_03",
                    severity="major",
                    message=f"Required figure '{fig}' not found in output directory — SHAP may not have completed.",
                    target_agent="Analyst",
                )
            )


def _check_top_features_present(ctx: object, result: PreCriticResult) -> None:
    """pcc_04 (major): results.json.top_features must not be empty."""
    results = getattr(ctx, "results_object", None) or {}
    if not results.get("top_features"):
        result.failures.append(
            CheckFailure(
                check_id="pcc_04",
                severity="major",
                message="results.json.top_features is empty — SHAP feature importance analysis did not complete.",
                target_agent="Analyst",
            )
        )


def _check_subgroup_performance_present(ctx: object, result: PreCriticResult) -> None:
    """pcc_05 (major): every declared protected attribute must be reported.

    Emptiness was the only thing checked, so a run that reported two of
    three declared attributes passed. That is what happened live: the spec
    declared [X1SEX, X1RACE, X1SES], the DataEngineer wrote a
    test_protected.csv containing only X1RACE and X1SES, and the gender
    analysis was skipped with a warning nobody had to act on. The paper
    can still say subgroup analysis was conducted for protected
    attributes, which is the kind of claim that must not be able to go
    quietly half-true.
    """
    results = getattr(ctx, "results_object", None) or {}
    reported = results.get("subgroup_performance") or {}
    if not reported:
        result.failures.append(
            CheckFailure(
                check_id="pcc_05",
                severity="major",
                message="results.json.subgroup_performance is empty — subgroup analysis did not run.",
                target_agent="Analyst",
            )
        )
        return

    spec = getattr(ctx, "research_spec", None) or {}
    declared = spec.get("subgroup_analyses") or []
    missing = [attr for attr in declared if attr not in reported]
    if missing:
        result.failures.append(
            CheckFailure(
                check_id="pcc_05",
                severity="major",
                message=(
                    "research_spec.subgroup_analyses declares "
                    f"{sorted(declared)} but results.json.subgroup_performance "
                    f"reports only {sorted(reported)}. Missing: {sorted(missing)}. "
                    "Every declared protected attribute must be carried into "
                    "test_protected.csv and reported, or the fairness claim is "
                    "only partly supported."
                ),
                target_agent="DataEngineer",
            )
        )


def _check_data_report_validation_passed(ctx: object, result: PreCriticResult) -> None:
    """pcc_06 (critical): data_report.validation_passed must be True."""
    report = getattr(ctx, "data_report", None) or {}
    # If validation_passed is explicitly False (not just missing), flag it
    if report.get("validation_passed") is False:
        warnings_preview = str(report.get("warnings", []))[:200]
        result.failures.append(
            CheckFailure(
                check_id="pcc_06",
                severity="critical",
                message=(
                    f"data_report.validation_passed=False. "
                    f"Warnings: {warnings_preview}"
                ),
                target_agent="DataEngineer",
                # Not revisable: SPEC §8 aborts on validation_passed ==
                # false. The ENGINEERING stage does the same after its one
                # targeted DataEngineer retry; this check agrees with it.
                revisable=False,
            )
        )
