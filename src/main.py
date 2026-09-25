"""EDM-ARS: Educational Data Mining Automated Research System — CLI entry point.

Exit codes, for a batch harness or the ``edmars`` front end:

  0  COMPLETED and released
  1  could not start: a usage, configuration or pre-flight problem (no
     run was started and nothing was spent)
  2  INCOMPLETE: finished, but a final check blocked release
  3  ABORTED: a stage failed; run_status.json and pipeline.log say why
  4  INTERRUPTED: Ctrl-C or a termination signal; resumable
  5  CRASHED: an unexpected error inside the run; traceback in
     <run folder>/crash.log; resumable
"""

import argparse
import json
import os
import shutil
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, NoReturn

import yaml
from dotenv import load_dotenv

load_dotenv()

from src.config import PROJECT_ROOT, load_config, resolve_config_path
from src.context import PipelineContext
from src.dataset_adapter import _DATASET_REGISTRY, create_dataset_adapter
from src.orchestrator import Orchestrator
from src.preflight import FAIL, WARN, Finding, check_run_prerequisites, has_failures
from src.task_template import _TASK_REGISTRY, create_task_template


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_DATASET = "hsls09_public"

# Prose markers of validation warnings that are ADVISORY, not blocking.
# Measured 2026-07-25 over the archive: PredictionTemplate's
# sum-of-pct_missing retention rule fires on 6 of 6 archived prediction
# specs (estimated n = 0 for five of them, 1,663 for the sixth) where the
# executed runs carried analytic_n = 14,039 (ELS) and 17,335 (HSLS).
# Escalating it to a hard load failure would reject every real prediction
# spec, so it is printed and stepped over. docs/v5_arc_t_spec.md §1.4
# (task_template.py:146-167) replaces the rule with
# ``feasibility.estimate_analytic_n()``; this list shrinks to () then.
_ADVISORY_WARNING_MARKERS: tuple[str, ...] = ("Estimated analytic_n",)


def _is_advisory_warning(warning: str) -> bool:
    return any(marker in warning for marker in _ADVISORY_WARNING_MARKERS)


def _load_registry_for_dataset(dataset: str, registry_dir: str | None = None) -> dict:
    """Load a dataset registry YAML, trying cwd then the project root."""
    rel = os.path.join(registry_dir or "data_registry", "datasets", f"{dataset}.yaml")
    candidates = [rel, os.path.join(_PROJECT_ROOT, rel)]
    for candidate in candidates:
        if os.path.exists(candidate):
            with open(candidate, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    raise ValueError(
        f"No registry YAML for dataset {dataset!r}; looked in "
        f"{candidates}. A locked research_spec cannot be validated "
        f"without its dataset registry."
    )


#: Top-level keys a locked research_spec may declare.
#:
#: A8. `--research-spec` accepts arbitrary JSON, but the ProblemFormulator
#: rewrites the spec and preserves only a handful of fields. Everything
#: else was dropped WITHOUT WARNING — a user writing careful constraints
#: into a locked spec (`known_concerns_to_flag`, `required_reporting`) got
#: a run that behaved as though they had never written them. Grepping the
#: rendered DataEngineer prompt for those custom keys returned zero hits.
#:
#: Split into three groups so the warning can say WHY a key is being
#: ignored, which is more useful than "unrecognised".
SPEC_KEYS_HONOURED = {
    # Core, every task type.
    "task_type", "dataset", "research_question", "outcome_variable",
    "outcome_type", "predictor_set", "subgroup_analyses",
    "target_population", "additional_constraints",
    # Causal (selection-on-observables, ITR).
    "treatment", "outcome", "primary_method", "comparator_method",
    "secondary_methods", "exclude_methods", "target_estimand_hint",
    "subgroup_of_interest_for_m5", "confounder_guidance",
    "adjustment_set", "adjustment_covariates", "rule_covariates",
    "heterogeneity_subgroups",
    # Difference-in-differences.
    "estimand", "group_variable", "post_variable", "placebo_outcome",
    # Psychometrics.
    "method_battery", "item_columns", "factor_model", "grouping_vars",
    "reverse_items", "response_labels", "response_codes", "scale_name",
    "cdm_model", "item_construction",
}

#: Documentation and provenance. Read by humans, not by the pipeline.
SPEC_KEYS_METADATA = {
    "task_id", "study", "rationale_for_method_set", "rationale_for_PF",
    "rationale_for_topic_selection", "grouping_notes", "known_concerns_to_flag",
    "required_reporting", "cross_system_brief", "pct_missing_approx", "note",
}


def _warn_on_unrecognised_spec_keys(spec: dict, path: str) -> None:
    """Say plainly which locked-spec keys will not influence the run.

    Silence here is the trap: a spec that is read, validated and then
    quietly stripped looks exactly like one that was honoured.
    """
    unknown = sorted(set(spec) - SPEC_KEYS_HONOURED - SPEC_KEYS_METADATA)
    metadata = sorted(set(spec) & SPEC_KEYS_METADATA)

    if unknown:
        print(
            f"WARNING: locked research_spec at {path!r} declares "
            f"{len(unknown)} key(s) the pipeline does not recognise and "
            f"will ignore: {', '.join(unknown)}. If they are meant to "
            "steer the run, put the guidance in 'additional_constraints' "
            "(free text, passed to the DataEngineer and Analyst prompts).",
            file=sys.stderr,
        )
    if metadata:
        print(
            f"NOTE: {', '.join(metadata)} in {path!r} are recorded for "
            "provenance but do NOT steer the run. Use "
            "'additional_constraints' for guidance the agents must act on.",
            file=sys.stderr,
        )


def load_locked_research_spec(
    path: str,
    dataset: str | None = None,
    registry_dir: str | None = None,
) -> dict:
    """Load and structurally validate a locked research_spec JSON file.

    Used by the ``--research-spec`` CLI flag (Phase 3b.4 / B6) and
    callable directly by tests.

    The dataset whose registry the spec is validated against is resolved
    as ``spec["dataset"]`` -> the ``dataset`` argument -> ``hsls09_public``.
    ``prediction`` specs are validated against that registry (temporal
    ordering, Tier-3 exclusion); the causal/psychometrics templates
    ignore the registry and validate structurally.

    Raises:
        FileNotFoundError: if the path does not exist.
        json.JSONDecodeError: if the file is not valid JSON.
        ValueError: if the spec lacks ``task_type``, names an unknown
            dataset, has no loadable registry, or fails structural
            validation under the corresponding TaskTemplate.
    """
    with open(path, encoding="utf-8") as f:
        spec = json.load(f)

    if not isinstance(spec, dict):
        raise ValueError(
            f"Locked research_spec must be a JSON object (got {type(spec).__name__})"
        )

    task_type = spec.get("task_type")
    if not task_type:
        raise ValueError(
            "Locked research_spec must declare 'task_type' "
            "(e.g., 'causal_soo')"
        )

    _warn_on_unrecognised_spec_keys(spec, path)

    resolved_dataset = spec.get("dataset") or dataset or _DEFAULT_DATASET
    # create_dataset_adapter raises ValueError on an unknown dataset —
    # that is itself a legitimate structural failure of the locked spec.
    adapter = create_dataset_adapter(resolved_dataset)
    registry = _load_registry_for_dataset(resolved_dataset, registry_dir)

    template = create_task_template(task_type)
    warnings = template.validate_research_spec(spec, registry, adapter)

    blocking = [w for w in warnings if not _is_advisory_warning(w)]
    for advisory in (w for w in warnings if _is_advisory_warning(w)):
        print(
            f"ADVISORY (non-blocking) for {path!r}: {advisory}",
            file=sys.stderr,
        )
    if blocking:
        joined = "\n  - ".join(blocking)
        raise ValueError(
            f"Locked research_spec at {path!r} failed structural "
            f"validation:\n  - {joined}"
        )
    return spec


#: Exit codes (see the module docstring).
EXIT_RELEASED = 0
EXIT_USAGE = 1
EXIT_INCOMPLETE = 2
EXIT_ABORTED = 3
EXIT_INTERRUPTED = 4
EXIT_CRASHED = 5

_TERMINAL_FINISHED = ("COMPLETED", "INCOMPLETE")
#: Checkpoint states from which a resume no longer needs the raw data file,
#: unless a revision sends the run back to the DataEngineer.
_PAST_ENGINEERING = (
    "ANALYZING", "CRITIQUING", "REVISING", "WRITING", "REVIEWING",
    "VERIFYING", "COMPLETED", "INCOMPLETE",
)

#: Files the pipeline itself writes into a run folder. Their presence means
#: the folder already holds a run; ``--overwrite`` removes exactly these
#: (and the folders and name patterns below) and nothing else, so files a
#: user or a front end keeps there (run_config.yaml, runner.json,
#: console.log, a locked spec) survive.
_RUN_FILES = (
    "checkpoint.json", "run_status.json", "invariants.json", "obligations.json",
    "events.jsonl", "live_status.json", "live_status.json.tmp", "pipeline.log",
    "crash.log", "token_usage.jsonl", "run_cost.json", "config_snapshot.yaml",
    "verification_report.json", "verification_raw.txt", "manuscript_lint.json",
    "research_spec.json", "literature_context.json",
    "literature_context_expanded.json", "retrieved_literature.json",
    "citation_depth_report.json", "data_report.json", "results.json",
    "review_report.json", "critic_reasoning.txt", "paper_outline.json",
    "references.bib", "train_X.csv", "train_y.csv", "test_X.csv", "test_y.csv",
    "test_protected.csv", "panel_analytic.csv", "items_analytic.csv",
    "q_matrix.json", "data_engineer_generated.py", "_generated_script.py",
    "analysis_helpers.py", "r_bridge.py", "model_comparison.csv",
    "feature_importance.csv", "subgroup_performance.csv", "roc_curves.png",
    "shap_summary.png", "shap_importance.png",
)
_RUN_FILE_PATTERNS = ("paper.*", "paper_for_review.*", "pdp_*.png")
_RUN_DIRS = ("prompts", "lsar_review")


class UsageError(Exception):
    """A problem with how the run was asked for; reported in one message."""


class _Parser(argparse.ArgumentParser):
    """argparse exits 2 on a bad flag, but 2 means INCOMPLETE here."""

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(EXIT_USAGE, f"{self.prog}: error: {message}\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="python -m src.main",
        description="EDM-ARS: Educational Data Mining Automated Research System",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help=(
            "Dataset name: " + ", ".join(sorted(_DATASET_REGISTRY)) + ". "
            "Default: the locked research spec's dataset, else hsls09_public. "
            "An explicit value that disagrees with the spec is an error."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        dest="output_dir",
        help=(
            "Run folder (default: a new output/run_YYYYMMDD_HHMMSS). A "
            "folder that already holds a run is refused unless you pass "
            "--resume or --overwrite."
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to the config file (default: the repository's config.yaml)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue the run in --output-dir from its checkpoint. The run "
            "keeps the study type, dataset and research spec it started with."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Start over in an --output-dir that already holds a run: delete "
            "that run's files (checkpoint, results, paper, logs) first. "
            "Other files in the folder are left alone."
        ),
    )
    parser.add_argument(
        "--prompt", default=None, help="Optional research direction or question"
    )
    parser.add_argument(
        "--research-spec",
        default=None,
        dest="research_spec",
        help=(
            "Path to a JSON file containing a locked research_spec. "
            "If provided, ProblemFormulator runs in 'refine' mode "
            "against this spec rather than generating a new one from "
            "scratch. The spec's 'task_type' field overrides the "
            "config.yaml pipeline.task_type, and its 'dataset' field "
            "chooses the dataset. Required for every study type except "
            "prediction."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help=(
            "Check the configuration, the locked research_spec, the API "
            "key, the data file and the tools the run needs, show what "
            "would run, and exit. Needs no API key, sends nothing, and "
            "creates or deletes no files. Exit status 1 when a real run "
            "would not start."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show Python tracebacks for errors (also EDM_ARS_DEBUG=1).",
    )
    return parser


@dataclass
class _Plan:
    """Everything a run or a dry run was resolved to, before either starts."""

    config_path: str
    config: dict
    task_type: str
    dataset: str
    raw_data_path: str
    output_dir: str
    locked_spec: dict | None = None
    spec_source: str | None = None
    resume: bool = False
    start_state: str | None = None
    #: Files an earlier run left in output_dir (names relative to it).
    earlier_run_files: list[str] = field(default_factory=list)
    earlier_run_state: str | None = None
    #: Why a real run with these flags would refuse the output folder.
    occupied: str | None = None


# ---------------------------------------------------------------------------
# Setup: config, dataset, study type, locked spec, output folder
# ---------------------------------------------------------------------------


def _load_config_for_run(path_arg: str | None) -> tuple[str, dict]:
    path = resolve_config_path(path_arg)
    try:
        return path, load_config(path)
    except FileNotFoundError:
        where = path
        if path_arg and not os.path.isabs(path_arg):
            where = f"{path_arg} (looked in {os.path.abspath(path_arg)} and {path})"
        raise UsageError(f"config file not found: {where}") from None
    except yaml.YAMLError as exc:
        raise UsageError(
            f"config file {path} is not valid YAML: {_one_line(exc)}"
        ) from None
    except (ValueError, OSError) as exc:
        raise UsageError(f"config file {path}: {_one_line(exc)}") from None


def _read_spec_json(path: str) -> dict:
    """Read a research spec file with errors that name the file."""
    try:
        with open(path, encoding="utf-8") as f:
            spec = json.load(f)
    except FileNotFoundError:
        raise UsageError(
            f"research spec file not found: {os.path.abspath(path)}"
        ) from None
    except json.JSONDecodeError as exc:
        raise UsageError(
            f"research spec {os.path.abspath(path)} is not valid JSON: "
            f"{exc.msg} (line {exc.lineno}, column {exc.colno})"
        ) from None
    except (OSError, UnicodeDecodeError) as exc:
        raise UsageError(
            f"research spec {os.path.abspath(path)} cannot be read: {_one_line(exc)}"
        ) from None
    if not isinstance(spec, dict):
        raise UsageError(
            f"research spec {os.path.abspath(path)} must be a JSON object "
            f"(got {type(spec).__name__})"
        )
    return spec


def _check_dataset(dataset: str) -> None:
    if dataset not in _DATASET_REGISTRY:
        raise UsageError(
            f"unknown dataset {dataset!r}. Available: "
            f"{', '.join(sorted(_DATASET_REGISTRY))}"
        )


def _fixture_specs(task_type: str) -> list[str]:
    """Shipped example specs for a study type, as repo-relative paths."""
    found: list[str] = []
    for path in sorted((PROJECT_ROOT / "runs" / "fixtures").glob("*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and data.get("task_type") == task_type:
            found.append(path.relative_to(PROJECT_ROOT).as_posix())
    return found


def _needs_spec_message(task_type: str, config_path: str) -> str:
    examples = _fixture_specs(task_type)
    example = (
        f" Example for this type (in the repository folder): {examples[-1]}."
        if examples else ""
    )
    return (
        f"study type {task_type!r} (pipeline.task_type in {config_path}) needs "
        f"a locked research spec: add --research-spec <file>.{example} Only "
        "prediction studies can start from the config alone or a free-text "
        "--prompt."
    )


def _prompt_intent_notice(prompt: str) -> None:
    """Say so when a free-text prompt asks for something a prediction run
    does not do. The run itself is not rerouted."""
    try:
        from src.design_selector import classify_intent

        intent = classify_intent(prompt)
    except Exception:  # noqa: BLE001 - a notice must never stop a run
        return
    suggested = {"causal": "causal_soo", "targeting": "causal_itr"}.get(intent)
    if not suggested:
        return
    kind = (
        "who benefits from a treatment"
        if intent == "targeting" else "the effect of one thing on another"
    )
    examples = _fixture_specs(suggested)
    how = (
        f" start from a locked {suggested} spec, for example "
        f"--research-spec {examples[-1]}" if examples
        else f" use a locked {suggested} research spec (--research-spec)"
    )
    print(
        f"NOTE: your --prompt reads like a question about {kind}, but this "
        "run is a prediction study: it will find what predicts the outcome, "
        f"not estimate an effect. For that kind of answer,{how}.",
        file=sys.stderr,
    )


def _read_checkpoint(output_dir: str) -> dict | None:
    """The checkpoint in ``output_dir``, None when there is none."""
    path = os.path.join(output_dir, "checkpoint.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as exc:
        raise UsageError(
            f"{path} cannot be read ({_one_line(exc)}); it may have been cut "
            "short while being written, so the run cannot be resumed from it."
        ) from None
    if not isinstance(data, dict):
        raise UsageError(f"{path} is not a checkpoint (not a JSON object)")
    return data


def _state_name(state: Any) -> str:
    """``PipelineState.COMPLETED`` -> ``"COMPLETED"``; strings pass through."""
    value = getattr(state, "value", state)
    return str(value or "").rsplit(".", 1)[-1]


def _existing_run_files(output_dir: str) -> list[str]:
    """Names (relative to ``output_dir``) of files an earlier run left there."""
    if not os.path.isdir(output_dir):
        return []
    found: set[str] = set()
    for name in _RUN_FILES:
        if os.path.isfile(os.path.join(output_dir, name)):
            found.add(name)
    for pattern in _RUN_FILE_PATTERNS:
        for path in Path(output_dir).glob(pattern):
            if path.is_file():
                found.add(path.name)
    for name in _RUN_DIRS:
        if os.path.isdir(os.path.join(output_dir, name)):
            found.add(name + "/")
    return sorted(found)


def _remove_run_files(output_dir: str, names: list[str]) -> None:
    for name in names:
        path = os.path.join(output_dir, name.rstrip("/"))
        if name.endswith("/"):
            shutil.rmtree(path, ignore_errors=False)
        else:
            os.remove(path)


def _quote(path: str) -> str:
    return f'"{path}"' if any(c.isspace() for c in path) else path


def _resume_command(plan: "_Plan") -> str:
    parts = ["python -m src.main"]
    if os.path.normcase(os.path.abspath(plan.config_path)) != os.path.normcase(
        str(PROJECT_ROOT / "config.yaml")
    ):
        parts.append(f"--config {_quote(plan.config_path)}")
    parts.append(f"--output-dir {_quote(plan.output_dir)} --resume")
    return " ".join(parts)


def _occupied_message(plan: "_Plan") -> str:
    shown = ", ".join(plan.earlier_run_files[:6])
    more = len(plan.earlier_run_files) - 6
    if more > 0:
        shown += f" and {more} more"
    if plan.earlier_run_state:
        return (
            f"{plan.output_dir} already holds a run that stopped at "
            f"{plan.earlier_run_state}. To continue it: "
            f"{_resume_command(plan)}. To discard it and start again in this "
            "folder, add --overwrite. Or choose a new --output-dir."
        )
    return (
        f"{plan.output_dir} already holds files from an earlier run ({shown}) "
        "but no checkpoint to resume from. Choose a new --output-dir, or add "
        "--overwrite to delete that run's files and start again here."
    )


def _resumable_runs(output_base: str) -> list[tuple[str, str]]:
    """(folder, state) of runs under ``output_base`` that --resume can continue."""
    runs: list[tuple[float, str, str]] = []
    try:
        entries = list(os.scandir(output_base))
    except OSError:
        return []
    for entry in entries:
        path = os.path.join(entry.path, "checkpoint.json")
        if not entry.is_dir() or not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            continue
        state = _state_name(data.get("current_state"))
        abort = data.get("abort_info") or {}
        if state in _TERMINAL_FINISHED:
            continue
        if state == "ABORTED" and not abort.get("resumable"):
            continue
        runs.append((os.path.getmtime(path), entry.path, state))
    return [(p, s) for _, p, s in sorted(runs, reverse=True)]


def _resume_without_output_dir(config: dict) -> str:
    base = config["paths"]["output_base"]
    runs = _resumable_runs(base)
    msg = (
        "--resume needs --output-dir <run folder>: the folder of the run to "
        "continue. Without it a new run would start."
    )
    if len(runs) == 1:
        folder, state = runs[0]
        msg += (
            f" One run in {base} can be resumed (stopped at {state}): "
            f"python -m src.main --output-dir {_quote(folder)} --resume"
        )
    elif runs:
        folder, state = runs[0]
        msg += (
            f" {len(runs)} runs in {base} can be resumed; the most recent is "
            f"{folder} (stopped at {state})."
        )
    return msg


def _new_run_dir(output_base: str) -> str:
    stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    candidate = os.path.abspath(os.path.join(output_base, stamp))
    n = 2
    while os.path.exists(candidate):
        candidate = os.path.abspath(os.path.join(output_base, f"{stamp}_{n}"))
        n += 1
    return candidate


def _plan_run(args: argparse.Namespace) -> _Plan:
    """Resolve config, study type, dataset, spec and folder, or raise
    UsageError. Creates, changes and deletes nothing."""
    if args.resume and args.overwrite:
        raise UsageError(
            "--resume continues a run and --overwrite discards one; use one "
            "or the other."
        )
    config_path, config = _load_config_for_run(args.config)
    config_task_type = config["pipeline"].get("task_type", "prediction")
    locked_spec: dict | None = None

    if args.resume:
        if not args.output_dir:
            raise UsageError(_resume_without_output_dir(config))
        output_dir = os.path.abspath(args.output_dir)
        checkpoint = _read_checkpoint(output_dir)
        if checkpoint is None:
            raise UsageError(
                f"--resume: there is no checkpoint.json in {output_dir}, so "
                "there is nothing to resume. Check the folder name; to start "
                "a new run there, leave out --resume."
            )
        # The run keeps what it started with. Rebuilding it from today's
        # flags re-typed a resumed causal run as a prediction run on HSLS,
        # which is what the README's own resume example did.
        dataset = checkpoint.get("dataset_name") or _DEFAULT_DATASET
        task_type = checkpoint.get("task_type") or "prediction"
        locked_spec = checkpoint.get("locked_research_spec")
        _check_dataset(dataset)
        if args.dataset and args.dataset != dataset:
            _note(f"--dataset {args.dataset} is ignored: this run was started "
                  f"on {dataset} and continues on it.")
        if args.research_spec:
            try:
                given = _read_spec_json(args.research_spec)
            except UsageError:
                given = None
            if given != locked_spec:
                _note(f"--research-spec {args.research_spec} is ignored: this "
                      "run keeps the research spec it started with.")
        elif config_task_type != task_type:
            _note(f"pipeline.task_type in {config_path} is {config_task_type}; "
                  f"this run was started as {task_type} and continues as one.")
        plan = _Plan(
            config_path=config_path, config=config, task_type=task_type,
            dataset=dataset, raw_data_path="", output_dir=output_dir,
            locked_spec=locked_spec,
            spec_source="from the checkpoint" if locked_spec else None,
            resume=True,
            start_state=_state_name(checkpoint.get("current_state")) or None,
        )
    else:
        if args.research_spec:
            raw_spec = _read_spec_json(args.research_spec)
            spec_dataset = raw_spec.get("dataset")
            if args.dataset and spec_dataset and args.dataset != spec_dataset:
                raise UsageError(
                    f"--dataset {args.dataset} disagrees with the research "
                    f"spec {args.research_spec}, which declares dataset "
                    f"{spec_dataset}. Leave out --dataset to use the spec's, "
                    "or correct the spec."
                )
            dataset = args.dataset or spec_dataset or _DEFAULT_DATASET
            _check_dataset(dataset)
            try:
                locked_spec = load_locked_research_spec(
                    args.research_spec, dataset=dataset
                )
            except (ValueError, OSError) as exc:
                raise UsageError(str(exc)) from None
            task_type = locked_spec["task_type"]
        else:
            dataset = args.dataset or _DEFAULT_DATASET
            _check_dataset(dataset)
            task_type = config_task_type
            if task_type not in _TASK_REGISTRY:
                raise UsageError(
                    f"pipeline.task_type in {config_path} is {task_type!r}, "
                    "which is not a study type this version knows. Use one "
                    f"of: {', '.join(sorted(_TASK_REGISTRY))}."
                )
            if task_type != "prediction":
                raise UsageError(_needs_spec_message(task_type, config_path))
            if args.prompt:
                _prompt_intent_notice(args.prompt)

        if args.output_dir:
            output_dir = os.path.abspath(args.output_dir)
            if os.path.exists(output_dir) and not os.path.isdir(output_dir):
                raise UsageError(f"--output-dir {output_dir} is a file, not a folder")
        else:
            output_dir = _new_run_dir(config["paths"]["output_base"])
        plan = _Plan(
            config_path=config_path, config=config, task_type=task_type,
            dataset=dataset, raw_data_path="", output_dir=output_dir,
            locked_spec=locked_spec,
            spec_source=args.research_spec if locked_spec else None,
        )
        plan.earlier_run_files = _existing_run_files(output_dir)
        if plan.earlier_run_files:
            try:
                checkpoint = _read_checkpoint(output_dir)
            except UsageError:
                checkpoint = {"current_state": "an unreadable checkpoint"}
            if checkpoint is not None:
                plan.earlier_run_state = (
                    _state_name(checkpoint.get("current_state")) or "an unknown stage"
                )
            if not args.overwrite:
                plan.occupied = _occupied_message(plan)

    adapter = create_dataset_adapter(plan.dataset)
    plan.raw_data_path = os.path.abspath(os.path.join(
        config["paths"]["raw_data"], adapter.get_raw_data_filename(),
    ))
    return plan


def _preflight(plan: _Plan) -> list[Finding]:
    findings = check_run_prerequisites(
        plan.config,
        plan.task_type,
        plan.dataset,
        plan.raw_data_path,
        bool((plan.config.get("review_gate") or {}).get("enabled", False)),
        locked_spec=plan.locked_spec,
    )
    if plan.resume and plan.start_state in _PAST_ENGINEERING:
        findings = [
            f._replace(
                severity=WARN,
                message=f.message + " The data is only needed again if a "
                "revision sends the run back to data preparation.",
            ) if f.code == "DATA_MISSING" else f
            for f in findings
        ]
    return findings


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _one_line(exc: BaseException | str) -> str:
    text = str(exc).strip() or type(exc).__name__
    return " ".join(text.split())


def _note(message: str) -> None:
    print(f"NOTE: {message}", file=sys.stderr)


def _print_findings(findings: list[Finding], stream: Any) -> None:
    for f in findings:
        label = "[x] problem" if f.severity == FAIL else "[!] warning"
        print(f"  {label} ({f.code}): {f.message}", file=stream)
        if f.fix:
            print(f"      what to do: {f.fix}", file=stream)


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def _dry_run_context(plan: _Plan, prompt: str | None) -> str:
    """The free text the orchestrator ranks skills against, before any
    stage has run (mirrors Orchestrator._stage_context_string)."""
    venue = ""
    if (plan.config.get("writer") or {}).get("venue_format") == "journal":
        venue = " journal manuscript APA"
    question = (plan.locked_spec or {}).get("research_question") or ""
    return (question or prompt or "") + venue


def _dry_run(plan: _Plan, args: argparse.Namespace) -> int:
    """Report what a real run would do, without constructing it.

    The orchestrator is NOT built here: its constructor creates the run
    folder, copies the config into it, writes pipeline.log, probes Docker
    and builds every agent -- which raised without an API key. A dry run
    creates, changes and deletes nothing.
    """
    from src.orchestrator import _resolve_skill_caps
    from src.skills.registry import SkillRegistry

    findings = _preflight(plan)
    if plan.occupied:
        findings.insert(0, Finding(
            "OUTPUT_DIR_IN_USE", FAIL, plan.occupied,
            "Add --resume or --overwrite, or choose a new --output-dir.",
        ))

    out = sys.stdout
    print("DRY RUN - pre-flight summary (nothing is created, changed or sent):", file=out)
    print(f"  config:               {plan.config_path}", file=out)
    print(f"  llm_provider:         {plan.config.get('llm_provider')}", file=out)
    print(f"  task_type:            {plan.task_type}", file=out)
    print(f"  task_template:        {type(create_task_template(plan.task_type)).__name__}", file=out)
    print(f"  dataset:              {plan.dataset}", file=out)
    print(f"  raw_data_path:        {plan.raw_data_path}", file=out)
    print(f"  raw_data exists:      {os.path.isfile(plan.raw_data_path)}", file=out)
    print(f"  output_dir:           {plan.output_dir}", file=out)
    if plan.resume:
        print(f"  resume:               yes, from {plan.start_state}", file=out)
    elif plan.earlier_run_files:
        what = (
            f"a run stopped at {plan.earlier_run_state}"
            if plan.earlier_run_state else "files from an earlier run"
        )
        action = (
            f"--overwrite would delete {len(plan.earlier_run_files)} of them"
            if args.overwrite else "a real run would refuse it"
        )
        print(f"  output_dir holds:     {what} ({action})", file=out)
    print(
        "  locked_research_spec: "
        + (f"set ({plan.spec_source})" if plan.locked_spec else "none"),
        file=out,
    )
    registry = SkillRegistry(skills_root=PROJECT_ROOT / "skills")
    print(f"  skill_registry count: {registry.count()}", file=out)
    print(f"  skills by layer:      {registry.count_by_layer()}", file=out)
    context = _dry_run_context(plan, args.prompt)
    for stage in ("ProblemFormulator", "DataEngineer", "Analyst", "Critic", "Writer"):
        matched = registry.match_and_compose(
            stage=stage,
            task_type=plan.task_type,
            dataset=plan.dataset,
            context=context,
            top_k_per_layer=_resolve_skill_caps(plan.task_type),
        )
        print(f"  skills @ {stage}: {len(matched)} -> {[s.name for s in matched]}", file=out)

    print("Checks:", file=out)
    if findings:
        _print_findings(findings, out)
    else:
        print("  [ok] nothing to report", file=out)
    if has_failures(findings):
        n = sum(1 for f in findings if f.severity == FAIL)
        print(f"Result: a real run would not start ({n} problem(s) above).", file=out)
        return EXIT_USAGE
    print("Result: a real run would start.", file=out)
    return EXIT_RELEASED


# ---------------------------------------------------------------------------
# Real run
# ---------------------------------------------------------------------------


def _run(plan: _Plan, args: argparse.Namespace) -> int:
    if plan.occupied:
        raise UsageError(plan.occupied)

    findings = _preflight(plan)
    if findings:
        print("Pre-flight checks:", file=sys.stderr)
        _print_findings(findings, sys.stderr)
    if has_failures(findings):
        raise UsageError(
            "the run was not started because of the problem(s) above; "
            "nothing was sent or spent."
        )

    if args.overwrite and plan.earlier_run_files:
        try:
            _remove_run_files(plan.output_dir, plan.earlier_run_files)
        except OSError as exc:
            raise UsageError(
                f"could not remove the earlier run's files from "
                f"{plan.output_dir}: {_one_line(exc)}"
            ) from None
        print(
            f"Removed {len(plan.earlier_run_files)} file(s) of the earlier run "
            f"from {plan.output_dir}.",
            file=sys.stderr,
        )

    ctx = PipelineContext(
        dataset_name=plan.dataset,
        raw_data_path=plan.raw_data_path,
        output_dir=plan.output_dir,
        task_type=plan.task_type,
        max_revision_cycles=plan.config["pipeline"]["max_revision_cycles"],
        locked_research_spec=plan.locked_spec,
    )
    try:
        orchestrator = Orchestrator(ctx, plan.config, config_path=plan.config_path)
    except Exception as exc:  # noqa: BLE001 - reported in one line
        if _debug(args):
            traceback.print_exc()
        raise UsageError(
            f"the run could not be set up: {type(exc).__name__}: {_one_line(exc)}"
        ) from None

    print(f"Run folder: {plan.output_dir}")
    result_ctx = orchestrator.run(user_prompt=args.prompt)

    print(f"Pipeline complete. Final state: {_state_name(result_ctx.current_state)}")
    print(f"Output directory: {result_ctx.output_dir}")
    if result_ctx.errors:
        print(f"Errors: {result_ctx.errors}", file=sys.stderr)

    # Say out loud what the run decided about itself, and exit non-zero
    # when it decided against release.
    #
    # This process used to exit 0 unconditionally, with no sys.exit
    # anywhere in the file, so a run whose quality gate said
    # ``passed: false`` was indistinguishable from a clean one to any
    # wrapper script, CI job or batch harness. 23 archived runs carry a
    # failing gate under ``current_state: "COMPLETED"``.
    status_path = os.path.join(result_ctx.output_dir, "run_status.json")
    if os.path.exists(status_path):
        try:
            with open(status_path, encoding="utf-8") as f:
                status = json.load(f)
            counts = status.get("invariant_counts") or {}
            print(
                f"Release: {'YES' if status.get('released') else 'NO'} "
                f"({status.get('reason')})"
            )
            print(
                f"Invariant findings: {counts.get('critical', 0)} critical, "
                f"{counts.get('major', 0)} major, {counts.get('minor', 0)} minor"
            )
        except (OSError, ValueError):
            pass

    return _exit_code_for(result_ctx)


def _debug(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "debug", False)) or os.environ.get("EDM_ARS_DEBUG") == "1"


def main(argv: list[str] | None = None) -> int:
    """Run the pipeline and return the process exit code.

    The annotation matters: a batch harness branches on the codes in the
    module docstring, and ``-> None`` said the opposite of the thing this
    return value exists to provide.
    """
    args = _build_parser().parse_args(argv)
    try:
        plan = _plan_run(args)
        if args.dry_run:
            return _dry_run(plan, args)
        return _run(plan, args)
    except UsageError as exc:
        # One readable line (or a short list) instead of a traceback: an
        # unknown dataset, a malformed spec, a missing config or key read
        # as a crash to anyone who is not a Python programmer.
        if _debug(args):
            traceback.print_exc()
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE


def _exit_code_for(ctx: Any) -> int:
    state = _state_name(getattr(ctx, "current_state", ""))
    if state == "ABORTED":
        return EXIT_ABORTED
    if state == "INCOMPLETE":
        return EXIT_INCOMPLETE
    return EXIT_RELEASED


if __name__ == "__main__":
    sys.exit(main())
