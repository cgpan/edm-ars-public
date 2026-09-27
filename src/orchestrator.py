from __future__ import annotations

import csv
import dataclasses
import json
import os
import shutil
import time
import warnings
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from src import events
from src.config import PROJECT_ROOT
from src.agents.analyst import Analyst
from src.agents.base import BaseAgent
from src.agents.critic import Critic
from src.agents.data_engineer import DataEngineer
from src.agents.problem_formulator import ProblemFormulator
from src.agents.writer import Writer
from src.causal_data_contract import (
    CausalDataContractError,
    assert_causal_soo_data_contract,
    assert_causal_soo_matrix_contract,
    repair_dummied_treatment,
)
from src.context import PipelineContext, PipelineState
from src.dataset_adapter import create_dataset_adapter
from src.errors import code_for_exception, is_resumable, reopened_pre_critic_stop
from src.findings_memory import FindingsMemory, RunEntry
from src.pre_critic_checks import PreCriticResult, run_pre_critic_checks
from src.review_gate import ReviewGate
from src.sandbox import compile_latex, create_executor
from src.skills import Skill, SkillRegistry
from src.task_template import create_task_template

# Default per-layer caps for skill matching, sized so injected content
# stays comfortably under typical max-tokens budgets even when several
# skills compose via references.
_DEFAULT_SKILL_CAPS: dict[str, int] = {
    "task-type": 3,
    "dataset": 4,
    "methodology": 5,
    "writing": 5,
}

# Task-type-specific overrides (Phase 3b.6 / 6.1). Causal_soo runs need
# ALL FIVE M-skills (M1-M5: regression-adjustment, PSM, IPW, AIPW/TMLE,
# causal-forest) attached at the Analyst stage alongside G1-G5 + D1.
# That's 11 causal skills + ~4 generic methodology references = 15+;
# the prediction cap of 5 is too tight. Raising to 12 fits the 11
# causal-specific without overflowing the prompt budget.
_SKILL_CAPS_BY_TASK_TYPE: dict[str, dict[str, int]] = {
    "causal_soo": {
        "task-type": 3,
        "dataset": 4,
        "methodology": 12,
        "writing": 5,
    },
    # V3.1 Arc R: ITR inherits the causal budget; methodology cap +2
    # for the M6/M7 additions on top of the G-family + M5.
    "causal_itr": {
        "task-type": 3,
        "dataset": 4,
        "methodology": 14,
        "writing": 5,
    },
    "causal_did": {
        "task-type": 3,
        "dataset": 4,
        "methodology": 10,
        "writing": 5,
    },
    "psychometrics": {
        "task-type": 3,
        "dataset": 4,
        "methodology": 10,
        "writing": 5,
    },
}


#: Encoded design matrices wider than this are structural corruption, not
#: a modelling choice. A typical prediction spec of <=30 raw variables
#: encodes to ~50 columns; the observed failure reached 16,945.
MAX_ENCODED_COLUMNS = 500


def check_design_matrix_width(
    output_dir: str, max_columns: int = MAX_ENCODED_COLUMNS
) -> str | None:
    """Refuse a design matrix wide enough that nothing can train.

    A2. A continuous maths theta score (X1TXMTSCOR, 20,741 distinct
    values) was one-hot encoded into 16,705 dummy columns; the matrix
    reached 18,806 x 16,945 and 1.8 GB, and no model trained inside any
    timeout. Three consecutive runs failed, each misdiagnosed as
    "slowness".

    The generated guard existed and mutated the list it was iterating
    (``onehot_cols.remove(col)`` inside ``for col in onehot_cols``), which
    skips the next element and lets roughly half the offenders through.
    Prose guidance did not hold, so the ceiling is enforced here, where a
    violation triggers a targeted DataEngineer retry.

    A module-level function rather than a method: the pre-flight is
    exercised with lightweight stubs, and a new check should not require
    every caller to grow an attribute.

    Names the columns responsible rather than reporting only the total,
    because the actionable fact is WHICH variable exploded -- the failure
    presents as a timeout, which sends people to the model battery
    instead of the encoder.
    """
    train_X_path = Path(output_dir) / "train_X.csv"
    if not train_X_path.exists():
        return None
    try:
        import pandas as _pd

        header = _pd.read_csv(train_X_path, nrows=0)
    except Exception:  # noqa: BLE001
        # The pre-flight must never be the thing that breaks a healthy run.
        return None

    n_cols = len(header.columns)
    if n_cols <= max_columns:
        return None

    from collections import Counter

    parents = Counter(str(c).split("_")[0] for c in header.columns)
    worst = ", ".join(
        f"{name} -> {count} columns"
        for name, count in parents.most_common(3)
        if count > 1
    )
    return (
        f"Encoded design matrix has {n_cols} columns, above the ceiling of "
        f"{max_columns}. This is one-hot expansion of a CONTINUOUS "
        f"variable, not a modelling choice, and no model will train. Worst "
        f"offenders: {worst or 'unable to attribute'}. Decide categorical "
        "vs continuous from the DECLARED TYPE in the dataset registry, not "
        "from the pandas dtype after sentinel replacement -- mapping "
        "labelled sentinels to NaN turns a numeric column into object "
        "dtype and makes a continuous score look categorical. Build the "
        "one-hot list with a comprehension "
        "(`[c for c in cats if X[c].nunique() <= 100]`); never remove from "
        "a list while iterating over it."
    )


#: A handful of all-zero dummies is a rare category; a matrix where most
#: of them are constant is a broken encoding. The threshold exists so a
#: single sparse level does not abort a healthy run.
MAX_CONSTANT_TEST_COLUMNS = 3


def check_constant_test_columns(
    output_dir: str, max_constant: int = MAX_CONSTANT_TEST_COLUMNS
) -> str | None:
    """Refuse a test matrix whose columns carry no information.

    Returns a retry message naming the offending columns, or ``None``.

    A module-level function rather than a method, matching
    ``check_design_matrix_width``: the pre-flight is exercised with
    lightweight stubs and a new check should not require every caller to
    grow an attribute.
    """
    path = Path(output_dir) / "test_X.csv"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8", newline="") as f:
            rd = csv.reader(f)
            header = next(rd)
            firsts: list[str | None] = [None] * len(header)
            varies = [False] * len(header)
            n = 0
            for row in rd:
                n += 1
                for i in range(min(len(row), len(header))):
                    if varies[i]:
                        continue
                    if firsts[i] is None:
                        firsts[i] = row[i]
                    elif row[i] != firsts[i]:
                        varies[i] = True
    except (OSError, StopIteration, ValueError):
        # The pre-flight must never be the thing that breaks a healthy run.
        return None
    if not n:
        return None

    constant = [header[i] for i in range(len(header)) if not varies[i]]
    if len(constant) <= max_constant:
        return None

    return (
        f"{len(constant)} of {len(header)} columns in test_X.csv are CONSTANT "
        f"across all {n} test rows: {', '.join(constant[:10])}"
        + (" ..." if len(constant) > 10 else "")
        + ". A column with no variance in the test set cannot move a test "
        "prediction, yet it reaches the model and then a SHAP ranking the "
        "paper writes about. This is one-hot encoding fitted SEPARATELY on "
        "train and test and reconciled with reindex: the two splits drop "
        "different reference categories, so surviving columns land on the "
        "wrong labels and absent categories become all-zero. Fit the "
        "encoding ONCE on train and apply the same category mapping to "
        "test -- analysis_helpers.encode_categoricals(train_df, test_df, "
        "cat_cols) does this and returns an encoding_report naming each "
        "reference category. Compute constancy on BOTH splits, not just "
        "train."
    )


def _resolve_skill_caps(task_type: str) -> dict[str, int]:
    """Return the per-layer skill cap for a given task type.

    Falls back to ``_DEFAULT_SKILL_CAPS`` when no task-type-specific
    override exists; this preserves byte-identical behavior for the
    prediction codepath.
    """
    return _SKILL_CAPS_BY_TASK_TYPE.get(task_type, _DEFAULT_SKILL_CAPS)


# ----------------------------------------------------------------------
# Terminal status, abort records and events
# ----------------------------------------------------------------------

#: States the run loop stops on. A resumed run in one of these does no
#: work -- except ABORTED with a resumable cause, which is rewound to the
#: stage that failed (see ``Orchestrator._prepare_resume``).
_TERMINAL_STATES = (
    PipelineState.COMPLETED,
    PipelineState.INCOMPLETE,
    PipelineState.ABORTED,
)

#: Stages whose runner returns at once when the stage is already in
#: ``completed_stages``. CRITIQUING and REVISING have no such guard, so a
#: resumed run always re-enters them.
_SKIPPABLE_STAGES = frozenset(
    {"FORMULATING", "ENGINEERING", "ANALYZING", "WRITING", "REVIEWING", "VERIFYING"}
)

#: What a person watching the run should read for each stage.
_STAGE_PLAIN: dict[str, str] = {
    "FORMULATING": "Choosing the research question and searching the literature",
    "ENGINEERING": "Preparing the data",
    "ANALYZING": "Running the analysis",
    "CRITIQUING": "Reviewing the analysis",
    "REVISING": "Revising the analysis after review",
    "WRITING": "Writing and compiling the paper",
    "REVIEWING": "Running the review gate",
    "VERIFYING": "Checking the finished paper against the run's own files",
}

#: Version of the ``run_status.json`` layout this module writes. 2 adds
#: state, reason_code, abort, gate, literature, run_id and written_at.
RUN_STATUS_SCHEMA = 2

#: How long a finishing run waits for another run's findings-memory
#: write before giving up on its own (non-fatal) update.
_FINDINGS_LOCK_TIMEOUT_S = 30.0

#: Words a failure uses when the file it wanted is not there.
_MISSING_FILE_MARKERS = (
    "FileNotFoundError",
    "No such file or directory",
    "cannot find the file",
    "cannot find the path",
)


class CheckpointCorruptError(ValueError):
    """``checkpoint.json`` exists but cannot be read back.

    Raised from ``Orchestrator.__init__`` instead of a bare
    ``JSONDecodeError`` so the message names the file and says what to
    do. With atomic saves this should only come from an external edit or
    a file written by an older version.
    """


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _state_name(state: Any) -> str:
    """``PipelineState.ANALYZING`` -> ``"ANALYZING"``; strings pass through."""
    value = getattr(state, "value", state)
    text = str(value)
    return text.split(".", 1)[1] if text.startswith("PipelineState.") else text


def _one_line(text: Any, limit: int = 500) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


def _abort_record(
    stage: str, code: str, message: str, checks: Optional[list] = None
) -> dict:
    """The ``ctx.abort_info`` / ``run_status.abort`` record.

    ``checks`` lists the pre-review findings behind a PRE_CRITIC_* stop
    (check_id, severity, message, target_agent, revisable), so a reader
    can show every finding in full rather than the first one cut to a
    line.
    """
    record = {
        "stage": stage,
        "code": code,
        "message": _one_line(message),
        "resumable": is_resumable(code),
        "at": _utc_now_iso(),
    }
    if checks:
        record["checks"] = checks
    return record


def _data_missing(ctx: Any, text: str) -> bool:
    """True when a failure reads as a missing file AND the raw data file
    the run was pointed at does not exist.

    Generated DataEngineer code opens the raw CSV itself, so a missing
    download surfaces as a FileNotFoundError inside the sandbox's stderr,
    three retries later, looking like a code bug. Naming it DATA_MISSING
    tells a user the fix is a download, not a different model.
    """
    if not any(marker in text for marker in _MISSING_FILE_MARKERS):
        return False
    raw = getattr(ctx, "raw_data_path", None)
    return bool(raw) and not os.path.exists(str(raw))


def _engineering_failure_code(ctx: Any, report: Any) -> str:
    """Code for a DataEngineer report that never passed validation."""
    text = json.dumps(report, default=str) if isinstance(report, dict) else str(report)
    return "DATA_MISSING" if _data_missing(ctx, text) else "DE_VALIDATION_FAILED"


def _exception_code(ctx: Any, exc: BaseException, default: str = "UNKNOWN") -> str:
    code = code_for_exception(exc)
    if code == "UNKNOWN" and _data_missing(ctx, f"{type(exc).__name__}: {exc}"):
        return "DATA_MISSING"
    return default if code == "UNKNOWN" else code


def _emit_sample_metric(ctx: Any, report: Any, stage: str = "ENGINEERING") -> None:
    """One ``metric`` event for the analytic sample size, best effort.
    Never raises (see _emit_results_metric)."""
    if not isinstance(report, dict):
        return
    try:
        n = report.get("analytic_n")
        events.emit(
            ctx,
            "metric",
            stage=stage,
            plain=f"{n} students in the analytic sample",
            key="analytic_n",
            value=n,
            ci=None,
            label="Students in the analytic sample",
        )
    except Exception:  # noqa: BLE001
        return


def _emit_results_metric(ctx: Any, results: Any, stage: str = "ANALYZING") -> None:
    """One ``metric`` event for the analysis headline, best effort.

    Never raises. It runs inside the ANALYZING and REVISING stages after
    the stage's work is done, and results.json is model-written: a
    ``best_model`` that is a list or a dict, or an ``all_models`` that is
    not a mapping, used to raise here and turn a finished analysis into
    an abort (or a REVISING failure into an UNVERIFIED paper).
    """
    if not isinstance(results, dict):
        return
    try:
        value = results.get("best_metric_value")
        if not isinstance(value, (int, float)):
            return
        metric = str(results.get("primary_metric") or "metric")
        best = results.get("best_model")
        ci = None
        all_models = results.get("all_models")
        row = (
            all_models.get(best)
            if isinstance(best, str) and isinstance(all_models, dict)
            else None
        )
        if isinstance(row, dict):
            lo = row.get(f"{metric.lower()}_ci_lower")
            hi = row.get(f"{metric.lower()}_ci_upper")
            if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
                ci = [lo, hi]
        events.emit(
            ctx,
            "metric",
            stage=stage,
            plain=f"Best model {best}: {metric} = {value}",
            key=metric,
            value=value,
            ci=ci,
            label=f"{metric} of the best model ({best})",
        )
    except Exception:  # noqa: BLE001
        return


def _critic_abort_message(review: Any) -> str:
    """One line saying why the Critic aborted: its first critical issue."""
    if isinstance(review, dict):
        for section in (
            "problem_formulation_review",
            "data_preparation_review",
            "analysis_review",
            "substantive_review",
        ):
            block = review.get(section)
            issues = block.get("issues") if isinstance(block, dict) else None
            for issue in issues or []:
                if isinstance(issue, dict) and issue.get("severity") == "critical":
                    text = issue.get("description") or issue.get("msg") or ""
                    if text:
                        return f"Critic ABORT: {text}"
    return "The Critic judged the study fundamentally flawed (ABORT verdict)."


def _literature_warning(ctx: Any) -> Optional[str]:
    """A sentence saying literature retrieval came back degraded, or None.

    Continuing without papers is the SPEC s8 design; finishing COMPLETED
    with no record of it anywhere a user looks is the defect (E9). The
    ProblemFormulator records what each source returned under
    ``literature_context["retrieval_status"]``.
    """
    lit = getattr(ctx, "literature_context", None)
    if not isinstance(lit, dict):
        return None
    status = lit.get("retrieval_status")
    papers = lit.get("papers") or []
    if isinstance(status, dict):
        if not status.get("degraded"):
            return None
        sources = ", ".join(
            f"{k}={v}" for k, v in status.items() if k not in ("degraded", "n_papers")
        )
        return (
            "Literature retrieval degraded: "
            f"{status.get('n_papers', len(papers))} paper(s) retrieved"
            + (f" ({sources})" if sources else "")
            + ". Related work and citations rest on a thin or missing pool."
        )
    if not papers:
        return (
            "Literature retrieval returned no papers; the paper's citations "
            "will be placeholders."
        )
    return None


def _revision_problem(summary: Any) -> Optional[str]:
    """A sentence when a gate that did not pass could not revise the paper.

    Only a gate that ran and did not pass revises between cycles, so a
    reviser that was unavailable (no model for the provider, malformed
    settings) or whose calls failed matters only then.
    """
    if not isinstance(summary, dict) or not summary.get("ran") or summary.get("passed"):
        return None
    try:
        if int(summary.get("max_cycles") or 0) < 2:
            return None  # one cycle: there is never a revision to make
    except (TypeError, ValueError):
        return None
    why = summary.get("revision_unavailable_reason")
    raw = summary.get("revision_failures")
    failures = [f for f in raw if isinstance(f, dict)] if isinstance(raw, list) else []
    if why:
        return (
            "The review gate could not revise the paper between cycles: "
            f"{_one_line(why, 300)}"
        )
    if failures:
        codes = sorted({str(f.get("code") or "UNKNOWN") for f in failures})
        return (
            f"The review gate's revision failed {len(failures)} time(s) "
            f"({', '.join(codes)}); later cycles reviewed an unrevised paper."
        )
    return None


def _atomic_write_text(path: str, text: str) -> None:
    """Replace ``path`` with ``text`` so a reader never sees half a file.

    ``open(path, "w")`` truncates first and writes in chunks: a kill, a
    full disk or a serialisation error part-way through used to leave a
    partial ``checkpoint.json`` -- the run's only resume point -- that the
    next ``--resume`` could not parse. The text is written to a sibling
    temp file, flushed to disk and moved over the target in one step.
    """
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        for attempt in range(5):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                # Windows refuses to replace a file another process has
                # open (an editor, a status viewer). Retry briefly.
                if attempt == 4:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _acquire_lock(path: str, timeout_s: float, stale_s: float) -> Optional[int]:
    """Create ``path`` exclusively. Returns the fd, -1 when locking is not
    possible here (proceed unlocked), or None on timeout."""
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    except OSError:
        return -1
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(fd, f"{os.getpid()}\n".encode("ascii"))
            except OSError:
                pass
            return fd
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(path) > stale_s:
                    # A holder that died without cleaning up.
                    os.remove(path)
                    continue
            except OSError:
                pass
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.1)
        except OSError:
            return -1


@contextmanager
def _exclusive_lock(
    path: str, timeout_s: float = 30.0, stale_s: float = 300.0
) -> Iterator[bool]:
    """A lock file next to a shared resource. Yields False on timeout."""
    fd = _acquire_lock(path, timeout_s, stale_s)
    try:
        yield fd is not None
    finally:
        if fd is not None and fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                os.remove(path)
            except OSError:
                pass


def _remove_stale_compile_outputs(output_dir: str) -> None:
    """Drop the previous compile's PDF and log before compiling again.

    The release check reads ``paper.log`` and ``paper.pdf`` from the run
    directory. In a reused ``--output-dir`` an earlier run's files would
    otherwise stand in for a compile that never happened this time, and
    an old PDF would ship as the new paper.
    """
    for name in ("paper.pdf", "paper.log"):
        path = os.path.join(output_dir, name)
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass


def _summarize_compile(output_dir: str, result: Any) -> dict:
    """What a compile actually produced, judged by the file, not the rc.

    pdflatex in nonstopmode exits 1 both for recoverable errors and for a
    fatal stop that writes nothing, so ``success`` cannot say whether a
    PDF exists. ``missing_tool`` names a program that was never found
    (``compile_latex`` reports it as rc -1 "not found").
    """
    result = result if isinstance(result, dict) else {}
    steps = [s for s in (result.get("steps") or []) if isinstance(s, dict)]
    missing_tool = result.get("missing_tool")
    failed_step = result.get("failed_step")
    # compile_latex judges freshness itself (the PDF appeared or changed
    # during this compile); prefer that over bare existence, which a PDF
    # the stale-output cleanup could not delete would otherwise satisfy.
    on_disk = os.path.exists(os.path.join(output_dir, "paper.pdf"))
    pdf_exists = bool(result["pdf_exists"]) and on_disk if "pdf_exists" in result else on_disk
    for step in steps:
        rc = step.get("returncode")
        if rc in (0, 1):
            continue
        if failed_step is None:
            failed_step = step.get("cmd")
        if (
            missing_tool is None
            and rc == -1
            and "not found" in str(step.get("stderr") or "")
        ):
            missing_tool = str(step.get("cmd") or "").split(" ", 1)[0] or None
    return {
        "success": bool(result.get("success")),
        "pdf_exists": pdf_exists,
        "stale_pdf": bool(on_disk and not pdf_exists),
        "missing_tool": missing_tool,
        "failed_step": failed_step,
        "message": result.get("message"),
        "steps": [
            {
                "cmd": s.get("cmd"),
                "returncode": s.get("returncode"),
                "stderr": str(s.get("stderr") or "")[-1000:],
                "stdout_tail": str(s.get("stdout") or "")[-500:],
            }
            for s in steps
        ],
        "written_at": _utc_now_iso(),
    }


def _exit_code_for_status(state: str, abort_code: Optional[str]) -> int:
    """The exit code the command line uses for this outcome: 0 released,
    2 held back, 3 aborted, 4 interrupted, 5 crashed."""
    if state == "COMPLETED":
        return 0
    if state == "INCOMPLETE":
        return 2
    if state == "INTERRUPTED":
        return 4
    if state == "ABORTED":
        return 5 if abort_code == "CRASHED" else 3
    return 1


def _pipeline_version() -> Optional[str]:
    try:
        import src as _src_pkg

        version = getattr(_src_pkg, "__version__", None)
        return str(version) if version else None
    except Exception:  # noqa: BLE001
        return None


class Orchestrator:
    def __init__(
        self,
        ctx: PipelineContext,
        config: dict,
        config_path: str = "config.yaml",
    ) -> None:
        self.ctx = ctx
        self.config = config
        self._config_path = config_path
        self._user_prompt: Optional[str] = None
        # Terminal bookkeeping: whether run.end / the terminal status have
        # been written, the status this session wrote, the stage in flight.
        self._finalized = False
        self._last_status: Optional[dict] = None
        self._stage_clock: Optional[tuple[str, int, float]] = None
        self._resumed = False
        self._budget_warned = False
        self._findings_memory_path: Optional[str] = None

        os.makedirs(ctx.output_dir, exist_ok=True)
        # Live side channel (events.jsonl + live_status.json). Attached
        # before anything can log, and again after a checkpoint load,
        # which replaces ctx.log with a plain list.
        events.attach(ctx, ctx.output_dir)

        # A resumed run is the run in the checkpoint, not whatever the
        # command line says this time. Adopt the checkpoint's dataset,
        # task type and locked spec BEFORE the template, adapter and
        # agents are built from them (D2) -- restoring them afterwards
        # would leave a causal run running prediction prompts.
        checkpoint = self._read_checkpoint()
        if checkpoint is not None:
            self._adopt_checkpoint_identity(checkpoint)

        # V4 psychometrics: executor subprocesses import the copied
        # r_bridge.py flat; give them a deterministic path to the
        # certified R scripts (inherited via os.environ).
        # Anchored at the repository, not the working directory: a run
        # started from another folder found no r_helpers/ and no skills/
        # (C5), and the skill registry then loaded zero skills silently.
        r_helpers = PROJECT_ROOT / "r_helpers"
        if r_helpers.is_dir():
            os.environ.setdefault("EDM_ARS_R_HELPERS", str(r_helpers))

        # Copy config snapshot for reproducibility
        if os.path.exists(config_path):
            shutil.copy(
                config_path,
                os.path.join(ctx.output_dir, "config_snapshot.yaml"),
            )

        # Create task template and dataset adapter
        self.task_template = create_task_template(ctx.task_type)
        self.dataset_adapter = create_dataset_adapter(ctx.dataset_name)

        # V2.0 skill registry: load all SKILL.md files under skills/.
        # Inert during the transition (agents whose system prompts have
        # no {{SKILLS}} placeholder fall through to the original prompt).
        self.skill_registry = SkillRegistry(skills_root=PROJECT_ROOT / "skills")

        # Load findings memory if enabled (non-fatal on failure)
        self.findings_memory: FindingsMemory | None = None
        self._pending_memory_warning: str | None = None
        fm_cfg = config.get("findings_memory", {})
        if fm_cfg.get("enabled", False):
            try:
                mem_path = fm_cfg.get("path", "findings_memory/memory.yaml")
                self._findings_memory_path = mem_path
                self.findings_memory = FindingsMemory.load(mem_path)
            except Exception as exc:
                self.findings_memory = None
                # Log after executor/agents are set up — deferred to after __init__
                self._pending_memory_warning = f"FindingsMemory load failed (non-fatal): {exc}"

        # Create shared executor (Docker sandbox or subprocess fallback)
        self._executor = create_executor(config)
        executor_type = type(self._executor).__name__

        # Instantiate all agents (share ctx reference, executor, template, and adapter)
        # Type the dict as Any: mypy otherwise infers a narrow value type
        # that conflicts with BaseAgent's `skills: list[Skill] | None` parameter
        # added in Phase 2c (even though we never pass skills via kwargs here).
        agent_kwargs: dict[str, Any] = dict(
            executor=self._executor,
            task_template=self.task_template,
            dataset_adapter=self.dataset_adapter,
        )
        self.problem_formulator = ProblemFormulator(ctx, "problem_formulator", config, **agent_kwargs)
        self.data_engineer = DataEngineer(ctx, "data_engineer", config, **agent_kwargs)
        self.analyst = Analyst(ctx, "analyst", config, **agent_kwargs)
        self.critic = Critic(ctx, "critic", config, **agent_kwargs)
        self.writer = Writer(ctx, "writer", config, **agent_kwargs)

        # Resume from checkpoint if present
        if checkpoint is not None:
            self._load_checkpoint(checkpoint)
        self._log("Orchestrator", f"Code executor: {executor_type}")
        if self._pending_memory_warning:
            self._log("Orchestrator", self._pending_memory_warning)

    # ------------------------------------------------------------------
    # V2.0 skill injection helpers
    # ------------------------------------------------------------------

    def _stage_context_string(self) -> str:
        """Free-text context for keyword-based skill ranking.

        Prefers the research question (post-formulation) and falls back to
        the original user prompt. Returns empty string if neither is
        available, in which case the matcher falls back to priority-only
        ranking.
        """
        venue_hint = ""
        if self.config.get("writer", {}).get("venue_format") == "journal":
            venue_hint = " journal manuscript APA"
        if self.ctx.research_spec:
            q = self.ctx.research_spec.get("research_question") or ""
            if q:
                return q + venue_hint
        return (self._user_prompt or "") + venue_hint

    def _match_skills_for_stage(self, agent_name: str) -> list[Skill]:
        """Return the composed skill list for one stage (testable helper)."""
        return self.skill_registry.match_and_compose(
            stage=agent_name,
            task_type=self.ctx.task_type,
            dataset=self.ctx.dataset_name,
            context=self._stage_context_string(),
            top_k_per_layer=_resolve_skill_caps(self.ctx.task_type),
        )

    def _inject_skills(self, agent: BaseAgent, agent_name: str) -> None:
        """Match + attach skills to an agent immediately before invoking it."""
        agent.skills = self._match_skills_for_stage(agent_name)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def _stage_handler(self, state: Any) -> Optional[Callable[[], None]]:
        handlers: dict[Any, Callable[[], None]] = {
            PipelineState.INITIALIZED: self._run_formulating,
            PipelineState.FORMULATING: self._run_formulating,
            PipelineState.ENGINEERING: self._run_engineering,
            PipelineState.ANALYZING: self._run_analyzing,
            PipelineState.CRITIQUING: self._run_critiquing,
            PipelineState.REVISING: self._run_revising,
            PipelineState.WRITING: self._run_writing,
            PipelineState.REVIEWING: self._run_reviewing,
            PipelineState.VERIFYING: self._run_verifying,
        }
        return handlers.get(state)

    def run(self, user_prompt: Optional[str] = None) -> PipelineContext:
        """Drive the state machine to a terminal state.

        Every way out of this method leaves a terminal record: a normal
        end writes ``run_cost.json`` and (for ABORTED, which never reaches
        VERIFYING) ``run_status.json``; an exception or Ctrl-C escaping a
        stage goes through :meth:`finalize_interrupted`, which keeps the
        checkpoint resumable, and is then re-raised for the caller.
        """
        self._user_prompt = user_prompt
        self._finalized = False
        try:
            self._prepare_resume()
            events.emit(
                self.ctx,
                "run.start",
                stage=self.ctx.current_state,
                cycle=self.ctx.revision_cycle,
                plain="Run resumed" if self._resumed else "Run started",
                version=_pipeline_version(),
                task_type=self.ctx.task_type,
                dataset=self.ctx.dataset_name,
                provider=self.config.get("llm_provider"),
                resumed=self._resumed,
            )
            while True:
                state = self.ctx.current_state
                if state in _TERMINAL_STATES:
                    break
                handler = self._stage_handler(state)
                if handler is None:
                    reason = f"Unknown state: {state}. Aborting."
                    self._log("Orchestrator", reason)
                    self.ctx.abort_info = _abort_record(
                        _state_name(state), "UNKNOWN", reason
                    )
                    self.ctx.errors.append(reason)
                    self.ctx.current_state = PipelineState.ABORTED
                    self._save_checkpoint()
                    break
                self._run_stage(state, handler)
        except KeyboardInterrupt:
            self.finalize_interrupted(
                "INTERRUPTED", "Interrupted (Ctrl-C or a termination signal)"
            )
            raise
        except Exception as exc:
            # Stage runners catch their own failures and abort cleanly;
            # anything that reaches here escaped that net. Keep the run
            # resumable and leave a record, then let the caller report it.
            self.finalize_interrupted("CRASHED", f"{type(exc).__name__}: {exc}")
            raise
        self._finalize_terminal()
        return self.ctx

    def _run_stage(self, state: Any, handler: Callable[[], None]) -> None:
        """Run one stage runner between ``stage.start`` / ``stage.end`` events."""
        stage = (
            "FORMULATING"
            if state == PipelineState.INITIALIZED
            else _state_name(state)
        )
        if stage in _SKIPPABLE_STAGES and stage in self.ctx.completed_stages:
            handler()  # returns at once: already done in an earlier session
            return
        cycle = self.ctx.revision_cycle
        started = time.monotonic()
        self._stage_clock = (stage, cycle, started)
        events.emit(
            self.ctx,
            "stage.start",
            stage=stage,
            cycle=cycle,
            plain=_STAGE_PLAIN.get(stage, stage.title()),
        )
        handler()
        self._stage_clock = None
        new_state = self.ctx.current_state
        if new_state == PipelineState.ABORTED:
            outcome = "aborted"
        elif new_state == PipelineState.INCOMPLETE:
            outcome = "blocked"
        else:
            outcome = "ok"
        events.emit(
            self.ctx,
            "stage.end",
            stage=stage,
            cycle=cycle,
            outcome=outcome,
            duration_s=round(time.monotonic() - started, 3),
            next_state=_state_name(new_state),
        )

    # ------------------------------------------------------------------
    # Resume and terminal records
    # ------------------------------------------------------------------

    def _prepare_resume(self) -> None:
        """Decide what a run starting from this context should do.

        An ABORTED checkpoint used to end a resumed run immediately, so a
        run stopped by a transient failure (network, an exhausted balance,
        a data file not yet downloaded) could only be continued by hand-
        editing ``current_state``. The abort record now names the stage
        that failed and whether its cause is one a user can fix; a
        resumable abort is rewound to that stage, keeping every completed
        stage. COMPLETED and INCOMPLETE stay terminal (D3).
        """
        if self.ctx.current_state == PipelineState.ABORTED:
            self._rewind_aborted()
        if self.ctx.current_state in _TERMINAL_STATES:
            return
        # The run is about to do work. An earlier attempt's interrupt
        # record and verdict files describe that attempt, not this one,
        # and a verdict file left behind would be printed as this run's.
        self.ctx.abort_info = None
        for name in ("run_status.json", "invariants.json", "obligations.json"):
            path = os.path.join(self.ctx.output_dir, name)
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError as exc:
                self._log("Orchestrator", f"Could not remove stale {name}: {exc}")

    def _rewind_aborted(self) -> None:
        info = self.ctx.abort_info if isinstance(self.ctx.abort_info, dict) else {}
        stage = info.get("stage")
        code = str(info.get("code") or "UNKNOWN")
        retryable_stage = (
            isinstance(stage, str)
            and stage in PipelineState.__members__
            and PipelineState(stage) not in _TERMINAL_STATES
        )
        if not retryable_stage:
            self._log(
                "Orchestrator",
                "Checkpoint is ABORTED and records no stage to retry; nothing "
                "to resume. Start a new run.",
            )
            return
        reopened = reopened_pre_critic_stop(info)
        if not (is_resumable(code) or reopened):
            self._log(
                "Orchestrator",
                f"Checkpoint is ABORTED in {stage} with {code}, which a resume "
                "cannot fix (it needs a different question or configuration). "
                "Nothing to resume; start a new run.",
            )
            return
        if reopened:
            self._log(
                "Orchestrator",
                f"The run was stopped by a pre-review finding this version "
                f"sends back for revision instead ({info.get('message', '')}). "
                "Retrying CRITIQUING, where the checks run again.",
            )
        self._log(
            "Orchestrator",
            f"Resuming an ABORTED run: retrying {stage} (previous failure "
            f"{code}: {info.get('message', '')}). Completed stages are kept: "
            f"{', '.join(self.ctx.completed_stages) or 'none'}.",
        )
        self.ctx.current_state = PipelineState(stage)
        self.ctx.abort_info = None
        self._resumed = True

    def finalize_interrupted(self, code: str, message: str) -> None:
        """Leave a resumable, readable record of a run that did not finish.

        Called by :meth:`run` when Ctrl-C or an unexpected exception
        escapes a stage, and by the command-line entry point from its own
        handlers (a second call is a no-op). The checkpoint is saved
        atomically at the stage that was in progress, so ``--resume``
        continues from there; ``run_cost.json`` and ``run_status.json``
        are written (state INTERRUPTED for ``code == "INTERRUPTED"``,
        otherwise ABORTED) and ``run.end`` is emitted. Never raises.
        """
        if self._finalized:
            return
        if self.ctx.current_state in _TERMINAL_STATES:
            self._finalize_terminal()
            return
        self._finalized = True
        stage = _state_name(self.ctx.current_state)
        if stage == "INITIALIZED":
            stage = "FORMULATING"
        interrupted = code == "INTERRUPTED"
        try:
            self.ctx.abort_info = _abort_record(stage, code, message)
        except Exception:  # noqa: BLE001
            pass
        clock = self._stage_clock
        if clock is not None:
            events.emit(
                self.ctx,
                "stage.end",
                stage=clock[0],
                cycle=clock[1],
                outcome="interrupted" if interrupted else "crashed",
                duration_s=round(time.monotonic() - clock[2], 3),
                next_state=_state_name(self.ctx.current_state),
            )
            self._stage_clock = None
        try:
            self._log(
                "Orchestrator",
                f"{'INTERRUPTED' if interrupted else 'CRASHED'} during {stage}"
                f" ({code}): {_one_line(message)}. The checkpoint is kept at "
                f"{stage}; resume with --resume and the same --output-dir.",
            )
        except Exception:  # noqa: BLE001
            pass
        events.emit(self.ctx, "error", stage=stage, code=code, message=_one_line(message))
        try:
            self._save_checkpoint()
        except Exception as exc:  # noqa: BLE001
            try:
                self._log("Orchestrator", f"Could not save the checkpoint: {exc}")
            except Exception:  # noqa: BLE001
                pass
        cost = self._write_cost_summary()
        status = self._write_run_status(
            self._stopped_status_fields("INTERRUPTED" if interrupted else "ABORTED")
        )
        self._emit_run_end(status, cost)

    def _finalize_terminal(self) -> None:
        """Terminal record for a run that reached a terminal state."""
        if self._finalized:
            return
        self._finalized = True
        cost = self._write_cost_summary()
        state = self.ctx.current_state
        status: Optional[dict]
        if state == PipelineState.ABORTED:
            # ABORTED never reaches VERIFYING, which is where the status
            # file used to be written -- so an aborted run had none, and a
            # reused directory showed the previous run's verdict instead.
            status = self._write_run_status(self._stopped_status_fields("ABORTED"))
        elif self._last_status is not None:
            status = self._last_status
        else:
            status = self._read_run_status()
            if status is None:
                status = self._write_run_status(self._rebuilt_status_fields())
        self._emit_run_end(status, cost)

    def _emit_run_end(self, status: Optional[dict], cost: Optional[dict]) -> None:
        status = status or {}
        state = str(status.get("state") or _state_name(self.ctx.current_state))
        abort = status.get("abort") if isinstance(status.get("abort"), dict) else {}
        cost_usd = (cost or {}).get("cost_usd")
        events.emit(
            self.ctx,
            "run.end",
            stage=self.ctx.current_state,
            cycle=self.ctx.revision_cycle,
            plain=f"Run ended: {state}",
            state=state,
            released=bool(status.get("released")),
            reason_code=status.get("reason_code"),
            exit_code=_exit_code_for_status(state, (abort or {}).get("code")),
            cost_usd=cost_usd,
        )

    def _run_id(self) -> str:
        sink = getattr(self.ctx, "event_sink", None)
        run_id = getattr(sink, "run_id", None)
        return str(run_id or os.path.basename(os.path.normpath(self.ctx.output_dir)))

    def _gate_block(self) -> dict:
        """``run_status.gate``: did the review gate run, and what did it say.

        A gate that never ran (disabled, LSAR missing, no PDF, an
        exception) reports ``ran: false`` with ``score: null`` -- never a
        failed review scored 0.0, which is what the old record said (B2).

        A gate that ran also carries ``final_manuscript_reviewed`` (False
        when paper.tex was revised after the review ``score`` comes from)
        and ``last_cycle_failure`` (why the next cycle reviewed nothing),
        both None when the summary does not say.
        """
        rg_cfg = self.config.get("review_gate", {}) or {}
        enabled = bool(rg_cfg.get("enabled", False))
        block: dict[str, Any] = {
            "enabled": enabled,
            "ran": False,
            "skip_reason": None,
            "passed": None,
            "score": None,
            "threshold": None,
            "advisory": None,
            "venue": rg_cfg.get("venue"),
        }
        res = self.ctx.review_gate_result
        if not isinstance(res, dict):
            block["skip_reason"] = "disabled" if not enabled else "not_reached"
            return block
        ran = res.get("ran")
        if ran is None:
            # A summary from before the flag existed (or a stub): it ran
            # if it completed a cycle and did not error.
            cycles = res.get("cycles_used")
            ran = (
                isinstance(cycles, (int, float))
                and not isinstance(cycles, bool)
                and cycles > 0
                and not res.get("error")
            )
        block["ran"] = bool(ran)
        if ran:
            passed = res.get("passed")
            block["passed"] = passed if isinstance(passed, bool) else None
            score = res.get("final_score")
            block["score"] = (
                float(score)
                if isinstance(score, (int, float)) and not isinstance(score, bool)
                else None
            )
            reviewed = res.get("final_manuscript_reviewed")
            block["final_manuscript_reviewed"] = (
                reviewed if isinstance(reviewed, bool) else None
            )
            failure = res.get("last_cycle_failure")
            block["last_cycle_failure"] = (
                _one_line(failure, 200) if failure else None
            )
        else:
            skip = res.get("skip_reason")
            if not skip and res.get("error"):
                skip = f"exception: {_one_line(res['error'], 200)}"
            block["skip_reason"] = skip or "unknown"
        block["threshold"] = res.get("threshold_used")
        block["advisory"] = res.get("advisory_mode")
        block["venue"] = res.get("venue") or block["venue"]
        return block

    def _literature_block(self) -> Optional[dict]:
        """``run_status.literature`` from the ProblemFormulator's retrieval
        status (``literature_context["retrieval_status"]``), or a
        count-based fallback without one."""
        lit = self.ctx.literature_context
        if not isinstance(lit, dict):
            if "FORMULATING" in (self.ctx.completed_stages or []):
                return {"degraded": True, "n_papers": 0, "sources": {}}
            return None
        papers = lit.get("papers") or []
        status = lit.get("retrieval_status")
        if isinstance(status, dict):
            n = status.get("n_papers")
            return {
                "degraded": bool(status.get("degraded")),
                "n_papers": n if isinstance(n, int) else len(papers),
                "sources": {
                    k: v
                    for k, v in status.items()
                    if k not in ("degraded", "n_papers")
                },
            }
        return {"degraded": not papers, "n_papers": len(papers), "sources": {}}

    def _status_common(self) -> dict:
        gate = self._gate_block()
        review = self.ctx.review_report or {}
        return {
            "schema": RUN_STATUS_SCHEMA,
            "run_id": self._run_id(),
            "run_dir": self.ctx.output_dir,
            "written_at": _utc_now_iso(),
            "timestamp": datetime.utcnow().isoformat(),
            "abort": None,
            "gate": gate,
            "review_gate_passed": gate["passed"],
            "review_gate_score": gate["score"],
            "critic_verdict": review.get("effective_verdict") or review.get("overall_verdict"),
            "critic_unverified": bool(review.get("unverified")),
            "literature": self._literature_block(),
        }

    def _stopped_status_fields(self, state: str) -> dict:
        """Status fields for a run that stopped before VERIFYING."""
        info = self.ctx.abort_info if isinstance(self.ctx.abort_info, dict) else {}
        code = str(info.get("code") or ("INTERRUPTED" if state == "INTERRUPTED" else "UNKNOWN"))
        errors = self.ctx.errors or []
        message = info.get("message") or (_one_line(errors[-1]) if errors else "")
        stage = info.get("stage")
        abort = {
            "stage": stage,
            "code": code,
            "message": message,
            "resumable": bool(info.get("resumable", is_resumable(code))),
        }
        if isinstance(info.get("checks"), list):
            abort["checks"] = info["checks"]
        if state == "INTERRUPTED":
            reason = (
                f"interrupted during {stage or 'the run'}; resume with --resume"
            )
        else:
            reason = f"aborted during {stage or 'the run'} ({code}): {message}"
        return {
            "state": state,
            "released": False,
            "reason": reason,
            "reason_code": "INTERRUPTED" if state == "INTERRUPTED" else "ABORTED",
            "abort": abort,
            "advisories": [],
            "verification": {"enabled": None, "ran": False, "error": None},
            "blocking_mode": None,
            "invariant_counts": {},
            "invariant_codes": [],
            "blocking_findings": [],
            "writer_obligations": None,
            "verifier": None,
        }

    def _rebuilt_status_fields(self) -> dict:
        """A resumed terminal run whose status file has gone missing."""
        state = _state_name(self.ctx.current_state)
        return {
            "state": state,
            "released": state == "COMPLETED",
            "reason": (
                "run_status.json was missing on resume and was rebuilt from "
                "the checkpoint; the verification details are not available"
            ),
            "reason_code": "VERIFICATION_NOT_RUN",
            "advisories": [],
            "verification": {"enabled": None, "ran": False, "error": "status rebuilt"},
            "blocking_mode": None,
            "invariant_counts": {},
            "invariant_codes": [],
            "blocking_findings": [],
            "writer_obligations": None,
            "verifier": None,
        }

    def _write_run_status(self, fields: dict) -> dict:
        """Write ``run_status.json`` (schema 2) atomically. Never raises."""
        try:
            common = self._status_common()
        except Exception as exc:  # noqa: BLE001
            # A malformed gate result or review report must not cost the
            # run its terminal record; write what is certain.
            common = {
                "schema": RUN_STATUS_SCHEMA,
                "run_id": os.path.basename(os.path.normpath(self.ctx.output_dir)),
                "written_at": _utc_now_iso(),
                "abort": None,
                "gate": None,
                "literature": None,
                "status_error": f"{type(exc).__name__}: {exc}",
            }
        status = {**common, **fields}
        self._last_status = status
        try:
            _atomic_write_text(
                os.path.join(self.ctx.output_dir, "run_status.json"),
                json.dumps(status, indent=2, default=str),
            )
        except Exception as exc:  # noqa: BLE001
            try:
                self._log("Orchestrator", f"Could not write run_status.json: {exc}")
            except Exception:  # noqa: BLE001
                pass
        return status

    def _read_run_status(self) -> Optional[dict]:
        path = os.path.join(self.ctx.output_dir, "run_status.json")
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else None
        except (OSError, ValueError):
            return None

    def _write_cost_summary(self) -> Optional[dict]:
        """Aggregate the run's measured token usage into run_cost.json (K1).

        Runs on BOTH terminal states — an aborted run still spent money,
        and a cost record that only exists for successes understates the
        real cost of operating the system. Returns the payload (or None).
        """
        try:
            from src.cost import write_summary

            payload = write_summary(self.ctx.output_dir, self.config)
            if not payload:
                return None
            cost = payload.get("cost_usd")
            cost_str = "not priced" if cost is None else f"${cost:.4f}"
            status = payload.get("cost_status")
            if cost is not None and status in ("estimated", "partial"):
                # An unverified rate, a call of unknown time on a
                # time-of-day priced model, or calls with no rate at all
                # make the figure an estimate or a lower bound;
                # run_cost.json says which, and the log line should not
                # read as measured.
                if status == "partial":
                    cost_str += " (lower bound: some calls have no rate)"
                elif payload.get("untimed_calls") and not payload.get(
                    "unverified_rate_models"
                ):
                    cost_str += (
                        " (estimated: some calls have no time, priced at peak)"
                    )
                else:
                    cost_str += " (estimated: a rate is unverified)"
            self._log(
                "Orchestrator",
                f"Run cost: {cost_str} over {payload['n_calls']} LLM calls "
                f"({payload['prompt_tokens']:,} in / "
                f"{payload['completion_tokens']:,} out; "
                f"{payload['cached_prompt_tokens']:,} cached) "
                "-> run_cost.json",
            )
            return payload
        except Exception as exc:  # noqa: BLE001 — accounting is never fatal
            try:
                self._log("Orchestrator", f"Cost summary skipped: {exc}")
            except Exception:  # noqa: BLE001
                pass
            return None

    # ------------------------------------------------------------------
    # Stage runners
    # ------------------------------------------------------------------

    def _run_formulating(self) -> None:
        self.ctx.current_state = PipelineState.FORMULATING
        if "FORMULATING" in self.ctx.completed_stages:
            self.ctx.current_state = PipelineState.ENGINEERING
            return
        self._log("Orchestrator", "Starting FORMULATING stage")
        try:
            fm_cfg = self.config.get("findings_memory", {})
            n_branches = (
                fm_cfg.get("n_candidate_specs", 1)
                if fm_cfg.get("enabled", False) and self.findings_memory is not None
                else 1
            )
            memory_summary = (
                self.findings_memory.to_summary_str()
                if self.findings_memory is not None
                else ""
            )
            studied_outcomes = (
                self.findings_memory.get_studied_outcomes()
                if self.findings_memory is not None
                else []
            )

            self._inject_skills(self.problem_formulator, "ProblemFormulator")
            result = self.problem_formulator.run(
                user_prompt=self._user_prompt,
                findings_memory_summary=memory_summary,
                n_candidate_specs=n_branches,
                studied_outcomes=studied_outcomes,
                # Phase 3b.4 / B6: pass locked spec when CLI provided one.
                # PF currently ignores via **kwargs; sub-wave 2 introduces
                # the causal "refine" branch that consumes this kwarg.
                locked_research_spec=self.ctx.locked_research_spec,
            )
            self.ctx.research_spec = self._carry_locked_guidance(
                result.get("research_spec")
            )
            self.ctx.literature_context = result.get("literature_context")
            self.ctx.retrieved_literature = result.get("retrieved_literature")
            self._save_formulating_outputs()
            self._note_literature_status()
            self.ctx.completed_stages.append("FORMULATING")
            self.ctx.current_state = PipelineState.ENGINEERING
            self._log("Orchestrator", "FORMULATING stage complete")
            self._save_checkpoint()
            self._check_cost()
        except Exception as e:
            self._abort(f"FORMULATING failed: {e}", exc=e)

    def _note_literature_status(self) -> None:
        """Say so, where a user looks, when literature retrieval degraded (E9).

        The failure used to be recorded only through ``ctx.log`` inside the
        ProblemFormulator, which reaches disk only in checkpoint.json; the
        run then finished COMPLETED with placeholder citations and nothing
        in pipeline.log, ctx.errors or run_status.json saying why.
        """
        warning = _literature_warning(self.ctx)
        if not warning:
            return
        self._log("Orchestrator", f"WARNING: {warning}")
        lit = getattr(self.ctx, "literature_context", None)
        if not isinstance((lit or {}).get("retrieval_status"), dict):
            # The ProblemFormulator emits this warning event itself
            # whenever it writes a degraded retrieval_status; announcing
            # it here as well would put it in the event stream twice.
            # Only a context without that block (an older checkpoint, a
            # path that skipped the search) is announced from here.
            events.emit(
                self.ctx,
                "warning",
                stage=self.ctx.current_state,
                code="LITERATURE_DEGRADED",
                message=warning,
            )
        if warning not in self.ctx.errors:
            self.ctx.errors.append(warning)

    def _check_design_matrix_width(self) -> str | None:
        """Refuse a design matrix wide enough that nothing can train.

        Names the columns responsible rather than reporting only the
        total, because the actionable fact is WHICH variable exploded --
        the failure presents as a timeout, which sends people looking at
        the model battery instead of the encoder.
        """
        return check_design_matrix_width(self.ctx.output_dir)

    def _run_post_de_preflight(self) -> str | None:
        """Run the causal-mode post-DE contract checks.

        Returns the violation message (str) or None when compliant.
        No-op (returns None) for non-causal task types and when the
        research spec is absent. Unexpected probe errors are logged and
        treated as non-violations — the pre-flight must never be the
        thing that breaks a healthy run.
        """
        spec = self.ctx.research_spec or {}

        # V4 Phase A (F-A1-ELS-EMPTY-TEST-SPLIT): task-type-agnostic split
        # sanity. On the first ELS run, HSLS-specific school-fingerprint
        # reconstruction degenerated to ONE pseudo-school, the school-aware
        # splitter put every row in train, and DE self-reported
        # validation_passed=True with n_test=0 — every model then failed
        # downstream. Check the FILE, not just the self-report. causal_did
        # runs estimate on the full panel (no split) — exempt.
        if spec.get("task_type") not in ("causal_did", "psychometrics"):
            dr = self.ctx.data_report or {}
            analytic_n = int(dr.get("analytic_n") or 0)
            n_test = int(dr.get("n_test") or 0)
            floor = max(1, int(0.15 * analytic_n))
            # Check the FILE too when it exists (a self-report can lie);
            # absent file -> report-only check (stubbed test runs).
            test_path = Path(self.ctx.output_dir) / "test_X.csv"
            test_rows = floor
            if test_path.exists():
                try:
                    import pandas as _pd

                    test_rows = len(_pd.read_csv(test_path))
                except Exception:
                    test_rows = 0
            if analytic_n > 0 and (n_test < floor or test_rows < floor):
                return (
                    f"Test split is degenerate: n_test={n_test}, actual "
                    f"test_X.csv rows={test_rows}, analytic_n={analytic_n} "
                    "(SPEC requires a stratified 20% test set). If "
                    "school-aware splitting produced fewer than 10 school "
                    "groups, or school IDs are unavailable/suppressed on "
                    "this dataset, use a PLAIN stratified 80/20 "
                    "train_test_split(random_state=42) and do NOT attempt "
                    "school-cluster reconstruction."
                )

        # A2: hard ceiling on the encoded design matrix.
        #
        # A continuous maths theta score (X1TXMTSCOR, 20,741 distinct
        # values) was one-hot encoded into 16,705 dummy columns; the
        # design matrix reached 18,806 x 16,945 and 1.8 GB, and no model
        # could train inside any timeout. Three consecutive runs failed
        # and were each misdiagnosed as "slowness".
        #
        # The generated guard existed but mutated the list it was
        # iterating (`onehot_cols.remove(col)` inside `for col in
        # onehot_cols`), which skips the next element and lets roughly
        # half the offenders through. Prose guidance alone clearly does
        # not hold, so the ceiling is enforced here where a violation
        # triggers a targeted retry.
        violation = check_design_matrix_width(self.ctx.output_dir)
        if violation:
            return violation

        # A column with no variance in TEST cannot move a test
        # prediction, yet it reaches the model and then a SHAP ranking
        # the paper writes about. `encode_categoricals` prevents this,
        # and the A/B measured that the DataEngineer does not call it --
        # so the guard lives here, in code, next to the width ceiling,
        # which exists for the same reason.
        violation = check_constant_test_columns(self.ctx.output_dir)
        if violation:
            return violation

        if spec.get("task_type") != "causal_soo":
            return None
        train_X_path = Path(self.ctx.output_dir) / "train_X.csv"
        try:
            # 3b.23.7 sw1c: deterministic repair for the dummied-treatment
            # shape BEFORE asserting — the pair encodes identical
            # information, so collapsing it is safe and saves a retry.
            repair_note = repair_dummied_treatment(self.ctx.output_dir, spec)
            if repair_note:
                self._log("Orchestrator", repair_note)
                if isinstance(self.ctx.data_report, dict):
                    self.ctx.data_report.setdefault("warnings", []).append(
                        repair_note
                    )
            assert_causal_soo_data_contract(train_X_path, spec)
            registry: dict = {}
            try:
                registry = self.data_engineer.load_registry() or {}
            except Exception:
                registry = {}
            assert_causal_soo_matrix_contract(
                self.ctx.output_dir, spec, registry
            )
        except CausalDataContractError as cdce:
            return str(cdce)
        except Exception as exc:  # probe robustness: never crash a run
            self._log(
                "Orchestrator",
                f"Post-DE pre-flight probe error (non-fatal, treated as "
                f"pass): {exc}",
            )
        return None

    def _run_engineering(self) -> None:
        if "ENGINEERING" in self.ctx.completed_stages:
            self.ctx.current_state = PipelineState.ANALYZING
            return
        self._log("Orchestrator", "Starting ENGINEERING stage")
        try:
            self._inject_skills(self.data_engineer, "DataEngineer")
            result = self.data_engineer.run()
            self.ctx.data_report = result
            if not result.get("validation_passed", False):
                # V4 Arc H (3b.23.7 sw1b): validation_passed=False is a
                # deterministic post-DE signal exactly like a contract
                # violation — grant the same single targeted retry with
                # the failed-validation warnings injected, instead of
                # aborting on the first roll of the codegen dice
                # (F-3b17 NaN-cells shape aborted a full run here).
                warnings_txt = "; ".join(
                    str(w) for w in result.get("warnings", [])
                )
                self._log(
                    "Orchestrator",
                    f"DE validation_passed=False -> targeted DataEngineer "
                    f"retry: {warnings_txt[:300]}",
                )
                self._inject_skills(self.data_engineer, "DataEngineer")
                result = self.data_engineer.run(
                    revision_instructions=(
                        "VALIDATION FAILURE on your previous output "
                        "(deterministic post-DE check -- not a Critic "
                        "opinion). Your code executed but the produced "
                        "artifacts failed validation. Regenerate the data "
                        "engineering code fixing exactly these problems "
                        "while keeping everything else unchanged:\n\n"
                        f"{warnings_txt}\n\n"
                        "In particular: NO NaN cells may remain in "
                        "train_X/test_X after imputation -- verify with "
                        "df.isna().sum().sum() == 0 before writing the "
                        "CSVs, and impute EVERY remaining column "
                        "(including one-hot dummies and passthrough "
                        "columns), not only the originally-listed ones."
                    )
                )
                self.ctx.data_report = result
                if not result.get("validation_passed", False):
                    self._abort(
                        f"ENGINEERING aborted (validation retry exhausted): "
                        f"validation_passed=False. Warnings: "
                        f"{result.get('warnings', [])}",
                        code=_engineering_failure_code(self.ctx, result),
                    )
                    return
                self._log(
                    "Orchestrator",
                    "DE validation retry produced a passing data_report",
                )
            if result.get("analytic_n", 0) < 1000:
                self._abort(
                    f"ENGINEERING aborted: analytic_n={result.get('analytic_n')} < 1000",
                    code="SAMPLE_TOO_SMALL",
                )
                return
            # V3.0 Phase 3b.12 / §12.2 + V4 Arc H (3b.23.7) — post-DE
            # pre-flight. Header check (3b.12) plus matrix-level D1
            # assertions (3b.23.7: treatment binary, no object dtypes,
            # continuous-vars-stay-continuous, propensity-overlap sanity).
            # V4 Phase A adds a task-type-agnostic test-split sanity check
            # (empty/degenerate test set -> violation; causal_did exempt).
            # On violation: ONE targeted DataEngineer retry with the
            # violation text injected, then abort if the retry still
            # violates (fail-fast preserved).
            violation = self._run_post_de_preflight()
            if violation is not None:
                self._log(
                    "Orchestrator",
                    f"Post-DE pre-flight violation -> targeted DataEngineer "
                    f"retry: {violation}",
                )
                self._inject_skills(self.data_engineer, "DataEngineer")
                result = self.data_engineer.run(
                    revision_instructions=(
                        "POST-DE PRE-FLIGHT CONTRACT VIOLATION on your "
                        "previous output (deterministic orchestrator check "
                        "-- not a Critic opinion). Regenerate the data "
                        "engineering code fixing exactly this violation "
                        "while keeping everything else unchanged:\n\n"
                        f"{violation}"
                    )
                )
                self.ctx.data_report = result
                if not result.get("validation_passed", False):
                    self._abort(
                        f"ENGINEERING aborted after pre-flight retry: "
                        f"validation_passed=False. Warnings: "
                        f"{result.get('warnings', [])}",
                        code=_engineering_failure_code(self.ctx, result),
                    )
                    return
                second_violation = self._run_post_de_preflight()
                if second_violation is not None:
                    self._abort(
                        f"ENGINEERING aborted (causal data contract, "
                        f"post-retry): {second_violation}",
                        code="DATA_CONTRACT_FAILED",
                    )
                    return
                self._log(
                    "Orchestrator",
                    "Post-DE pre-flight retry produced a compliant matrix",
                )
            self.ctx.completed_stages.append("ENGINEERING")
            self.ctx.current_state = PipelineState.ANALYZING
            self._log("Orchestrator", "ENGINEERING stage complete")
            _emit_sample_metric(self.ctx, result)
            self._save_checkpoint()
            self._check_cost()
        except Exception as e:
            self._abort(
                f"ENGINEERING failed: {e}",
                code=_exception_code(self.ctx, e),
            )

    def _run_analyzing(self) -> None:
        if "ANALYZING" in self.ctx.completed_stages:
            self.ctx.current_state = PipelineState.CRITIQUING
            return
        self._log("Orchestrator", "Starting ANALYZING stage")
        try:
            self._inject_skills(self.analyst, "Analyst")
            result = self.analyst.run()
            self.ctx.results_object = result
            self.ctx.completed_stages.append("ANALYZING")
            self.ctx.current_state = PipelineState.CRITIQUING
            self._log("Orchestrator", "ANALYZING stage complete")
            _emit_results_metric(self.ctx, result)
            self._save_checkpoint()
            self._check_cost()
        except Exception as e:
            self._abort(
                f"ANALYZING failed: {e}",
                code=_exception_code(self.ctx, e, default="ANALYSIS_FAILED"),
            )

    def _carry_locked_guidance(self, spec: dict | None) -> dict | None:
        """Keep the locked spec's free-text guidance in the emitted spec.

        The locked spec is handed to the ProblemFormulator, which returns
        a research_spec of its own, and that replaces the locked one
        wholesale. Anything the PF does not echo back is gone by the time
        the DataEngineer runs.

        A study spec proved what that costs. It carried an ENCODING
        CONTRACT -- "do NOT pass them to get_dummies ... under any
        circumstance" -- naming five continuous predictors. The prompts
        show it reached the ProblemFormulator (6 mentions) and NOTHING
        downstream: DataEngineer, Analyst and Critic all saw zero. X1SES
        was one-hot encoded into 5,514 columns and X1TXMTSCOR into 9,350,
        giving 15,008 features for 12,918 students, and SHAP for
        X1TXMTSCOR summed to 0.0 across its 9,350 dummies.

        The unrecognised-key warning tells authors to move such guidance
        into `additional_constraints`. That advice was itself broken
        until this function existed, because that key died at the same
        boundary as the ones it was recommending a retreat from.
        """
        if not isinstance(spec, dict):
            return spec
        locked = getattr(self.ctx, "locked_research_spec", None)
        if not isinstance(locked, dict):
            return spec
        carried = locked.get("additional_constraints")
        if carried and not spec.get("additional_constraints"):
            spec["additional_constraints"] = carried
            self._log(
                "Orchestrator",
                "Carried 'additional_constraints' from the locked spec into "
                "the ProblemFormulator's research_spec so it reaches the "
                "DataEngineer, Analyst and Critic.",
            )
        return spec

    def _run_critiquing(self) -> None:
        self._log("Orchestrator", f"Starting CRITIQUING stage (cycle {self.ctx.revision_cycle})")
        try:
            # --- Deterministic pre-Critic guard (inspired by AutoResearchClaw health.py) ---
            # 3b.6 / 6.2: task_type gates which structural checks fire so
            # prediction-shaped complaints (SHAP, top_features, etc.) do
            # not pollute causal_soo Critic input.
            pre_result = run_pre_critic_checks(
                self.ctx, self.ctx.output_dir, task_type=self.ctx.task_type,
            )
            if pre_result.failures:
                for f in pre_result.failures:
                    self._log(
                        "Orchestrator",
                        f"[PreCritic][{f.severity}] {f.check_id}: {f.message}",
                    )
            if pre_result.has_critical:
                # Short-circuit without a Critic call: revise what a
                # revision can fix, stop on what it cannot.
                self._pre_critic_short_circuit(pre_result)
                self._save_checkpoint()
                self._check_cost()
                return

            memory_summary = (
                self.findings_memory.to_summary_str()
                if self.findings_memory is not None
                else ""
            )
            self._inject_skills(self.critic, "Critic")
            result = self.critic.run(
                findings_memory_summary=memory_summary,
                pre_critic_failures=pre_result.failures,
            )
            self.ctx.review_report = result

            # Phase 3b.10 / §10.3: deterministic verdict evaluator.
            # Pre-3b.10 the orchestrator trusted result['overall_verdict']
            # directly. F-CRITIC-PASSED-WITH-LOW-SCORE (3b.5 + 3b.9
            # recurrence) showed the LLM emitted PASS even with
            # quality_score < 7 + critical issues present. The evaluator
            # below recomputes from (quality_score, n_critical, n_major)
            # per the documented thresholds, applies the cycles-exhausted
            # UNVERIFIED downgrade, and surfaces LLM-reported
            # disagreement at WARNING level.
            from src.agents.verdict_evaluator import evaluate_critic_verdict

            eval_result = evaluate_critic_verdict(
                result,
                revision_cycle=self.ctx.revision_cycle,
                max_revision_cycles=self.ctx.max_revision_cycles,
            )
            verdict = eval_result.verdict
            if eval_result.llm_disagreement:
                self._log(
                    "Orchestrator",
                    f"Critic verdict-evaluator overrode LLM: "
                    f"llm={eval_result.llm_verdict!r} → "
                    f"deterministic={eval_result.deterministic_verdict!r}, "
                    f"effective={verdict!r}, unverified={eval_result.unverified}. "
                    f"{eval_result.rationale}",
                )

            if verdict == "PASS":
                # Record the flag EXPLICITLY in both directions. On the
                # evaluator-override path the raw LLM verdict string can
                # be "REVISE" while the effective verdict is PASS —
                # run_is_unverified (Writer + linter) honors an explicit
                # unverified key, so an effective PASS must write False
                # or the paper gets a warning block the system itself
                # says it does not deserve.
                self.ctx.review_report["unverified"] = bool(
                    eval_result.unverified
                )
                self.ctx.completed_stages.append("CRITIQUING")
                self.ctx.current_state = PipelineState.WRITING
                pass_label = "PASS (UNVERIFIED)" if eval_result.unverified else "PASS"
                self._log(
                    "Orchestrator",
                    f"Critic verdict: {pass_label} → proceeding to WRITING",
                )
            elif verdict == "REVISE":
                if self.ctx.revision_cycle < self.ctx.max_revision_cycles:
                    self.ctx.revision_cycle += 1
                    self.ctx.current_state = PipelineState.REVISING
                    self._log(
                        "Orchestrator",
                        f"Critic verdict: REVISE → starting revision cycle {self.ctx.revision_cycle}",
                    )
                else:
                    self.ctx.review_report["unverified"] = True
                    self.ctx.current_state = PipelineState.WRITING
                    self._log(
                        "Orchestrator",
                        "Critic verdict: REVISE but max cycles exhausted → WRITING (UNVERIFIED)",
                    )
            elif verdict == "ABORT":
                self.ctx.errors.append(f"Critic issued ABORT verdict: {result}")
                self.ctx.abort_info = _abort_record(
                    "CRITIQUING", "CRITIC_ABORT", _critic_abort_message(result)
                )
                events.emit(
                    self.ctx,
                    "error",
                    stage="CRITIQUING",
                    code="CRITIC_ABORT",
                    message=self.ctx.abort_info["message"],
                )
                self.ctx.current_state = PipelineState.ABORTED
                self._log("Orchestrator", "Critic verdict: ABORT → pipeline aborted")
            else:
                self._abort(f"Unknown critic verdict: {verdict}", code="UNKNOWN")
                return

            events.emit(
                self.ctx,
                "metric",
                stage="CRITIQUING",
                cycle=self.ctx.revision_cycle,
                key="critic_score",
                value=result.get("overall_quality_score")
                if isinstance(result, dict)
                else None,
                ci=None,
                label="Critic quality score (0-10)",
            )
            events.emit(
                self.ctx,
                "verdict",
                stage="CRITIQUING",
                cycle=self.ctx.revision_cycle,
                plain=f"Critic verdict: {verdict}",
                critic_score=result.get("overall_quality_score")
                if isinstance(result, dict)
                else None,
                verdict=verdict,
                unverified=bool(self.ctx.review_report.get("unverified", eval_result.unverified)),
            )

            # Re-persist now that the effective verdict and the
            # `unverified` flag are settled. Without this the on-disk
            # review_report.json keeps only the LLM's raw verdict, and
            # every downstream reader -- Writer, linter, audit, archive
            # -- is looking at a record the orchestrator has already
            # superseded in memory.
            self.ctx.review_report["effective_verdict"] = verdict
            self.ctx.review_report.setdefault(
                "unverified", bool(eval_result.unverified)
            )
            self.ctx.review_report["verdict_evaluation"] = {
                "llm_verdict": eval_result.llm_verdict,
                "deterministic_verdict": eval_result.deterministic_verdict,
                "llm_disagreement": bool(eval_result.llm_disagreement),
                "rationale": eval_result.rationale,
                "revision_cycle": self.ctx.revision_cycle,
                "max_revision_cycles": self.ctx.max_revision_cycles,
            }
            try:
                self.critic.persist_review_report(self.ctx.review_report)
            except Exception as exc:  # noqa: BLE001 - never lose the stage over a write
                self._log("Orchestrator", f"Could not re-persist review_report.json: {exc}")

            self._save_checkpoint()
            self._check_cost()
        except Exception as e:
            self._abort(f"CRITIQUING failed: {e}", exc=e)

    def _run_revising(self) -> None:
        self._log("Orchestrator", f"Starting REVISING stage (cycle {self.ctx.revision_cycle})")
        try:
            self._execute_revisions()
            self.ctx.completed_stages.append("REVISING")
            self.ctx.current_state = PipelineState.CRITIQUING
            self._log("Orchestrator", "REVISING stage complete → back to CRITIQUING")
            self._save_checkpoint()
            self._check_cost()
        except Exception as e:
            review = self.ctx.review_report if isinstance(self.ctx.review_report, dict) else {}
            if review.get("_source") == "pre_critic_short_circuit":
                # This revision was ordered by the pre-review check, whose
                # finding is that the existing analysis cannot carry the
                # paper (its central test or its models are missing).
                # Writing it UNVERIFIED is the outcome the check exists to
                # prevent, so stop instead. The code comes from the error
                # (a provider failure stays resumable) and the stage is
                # REVISING, so --resume retries this same revision.
                self._abort(
                    f"REVISING failed ({e}); the revision the pre-review "
                    f"check required did not run, so the paper is not written",
                    code=_exception_code(self.ctx, e),
                )
                return
            # Revision failure is non-fatal: fall back to WRITING with UNVERIFIED flag
            # rather than aborting and discarding the existing analysis results.
            self._log("Orchestrator", f"REVISING failed ({e}); falling back to WRITING (UNVERIFIED)")
            events.emit(
                self.ctx,
                "warning",
                stage="REVISING",
                code="REVISION_FAILED",
                message=_one_line(f"REVISING failed ({e}); writing the paper UNVERIFIED"),
            )
            if self.ctx.review_report is None:
                self.ctx.review_report = {}
            self.ctx.review_report["unverified"] = True
            self.ctx.errors.append(f"REVISING failed: {e}")
            self.ctx.current_state = PipelineState.WRITING
            self._save_checkpoint()

    def _run_writing(self) -> None:
        if "WRITING" in self.ctx.completed_stages:
            rg_enabled = self.config.get("review_gate", {}).get("enabled", False)
            if rg_enabled and "REVIEWING" not in self.ctx.completed_stages:
                self.ctx.current_state = PipelineState.REVIEWING
            else:
                self.ctx.current_state = PipelineState.VERIFYING
            return
        self._log("Orchestrator", "Starting WRITING stage")
        try:
            # Phase 1: generate outline if outline_first is enabled
            outline = None
            if self.config.get("writer", {}).get("outline_first", True):
                self._log("Orchestrator", "Running OutlineAgent (outline-first mode)")
                try:
                    from src.agents.outline_agent import OutlineAgent

                    agent_kwargs = dict(
                        executor=self._executor,
                        task_template=self.task_template,
                        dataset_adapter=self.dataset_adapter,
                    )
                    outline_agent = OutlineAgent(
                        self.ctx, "outline_agent", self.config, **agent_kwargs
                    )
                    self._inject_skills(outline_agent, "OutlineAgent")
                    outline = outline_agent.run()
                    self.ctx.paper_outline = outline
                    self._log("Orchestrator", "OutlineAgent complete")
                except Exception as e:
                    # Non-fatal, but NOT invisible. This branch used to
                    # leave one log line and nothing else, and a config
                    # pointing the outline stage at a retired model id
                    # (`deepseek-v4-flash`, which the API stopped serving)
                    # therefore degraded ten shipped configs to the v1
                    # placeholder-filling template path with no record
                    # anywhere that the outline-first design had not run.
                    # A degradation the run does not disclose is the D7
                    # defect this project writes papers about.
                    self._log(
                        "Orchestrator",
                        f"OutlineAgent failed (non-fatal, falling back to v1): {e}",
                    )
                    self.ctx.errors.append(
                        f"OutlineAgent failed; paper written via the v1 "
                        f"template path instead of outline-first: {e}"
                    )
                    events.emit(
                        self.ctx,
                        "warning",
                        stage="WRITING",
                        code="OUTLINE_FAILED",
                        message=_one_line(
                            f"OutlineAgent failed; writing via the v1 template: {e}"
                        ),
                    )
                    outline = None

            # Arc P3: top the reference list back up to the venue norm.
            # The ProblemFormulator retrieves ~100 papers and persists
            # only the 8-12 it selected, which is why manuscripts carried
            # 4-26 references against venue norms of 15 (EDM) / 47 (JLA) /
            # 54 (JEDM). Pure set arithmetic over already-retrieved
            # records: no new search, no LLM call, no network.
            self._expand_literature_for_depth()

            # Phase 2: generate prose
            self._inject_skills(self.writer, "Writer")
            result = self.writer.run(outline=outline)
            self.ctx.paper_text = result if isinstance(result, str) else result.get("paper_text", "")

            # Compile LaTeX: pdflatex → bibtex → pdflatex → pdflatex
            self._compile_paper()

            self.ctx.completed_stages.append("WRITING")

            # Transition to REVIEWING if the review gate is enabled, else
            # straight to VERIFYING. Neither branch ends the run: VERIFYING
            # is what decides between COMPLETED and INCOMPLETE.
            rg_enabled = self.config.get("review_gate", {}).get("enabled", False)
            if rg_enabled:
                self.ctx.current_state = PipelineState.REVIEWING
                self._log("Orchestrator", "WRITING stage complete → REVIEWING")
            else:
                self.ctx.current_state = PipelineState.VERIFYING
                self._log("Orchestrator", "WRITING stage complete → VERIFYING")
                events.emit(
                    self.ctx,
                    "gate.skipped",
                    stage="WRITING",
                    plain="Review gate is off; the paper is not reviewed",
                    reason="disabled",
                )
            self._save_checkpoint()
            self._check_cost()
            if not rg_enabled:
                self._update_findings_memory()
        except Exception as e:
            self._abort(f"WRITING failed: {e}", exc=e)

    def _compile_paper(self) -> dict:
        """Compile paper.tex and record what the compile really produced.

        Writes ``latex_compile.json`` -- every step's command, return code
        and stderr tail, plus whether ``paper.pdf`` exists and which tool,
        if any, was not found. Before this, a machine without pdflatex
        left no trace outside one pipeline.log line: no ``paper.log``, no
        ``paper.pdf``, and a release check that therefore never fired.
        The success line is decided by the PDF on disk, not by return
        codes, because nonstopmode pdflatex exits 1 for recoverable
        errors and for a fatal stop alike.
        """
        self._log("Orchestrator", "Compiling paper.tex (pdflatex → bibtex → pdflatex → pdflatex)")
        _remove_stale_compile_outputs(self.ctx.output_dir)
        compile_result = compile_latex(self.ctx.output_dir)
        summary = _summarize_compile(self.ctx.output_dir, compile_result)
        try:
            _atomic_write_text(
                os.path.join(self.ctx.output_dir, "latex_compile.json"),
                json.dumps(summary, indent=2, default=str),
            )
        except OSError as exc:
            self._log("Orchestrator", f"Could not write latex_compile.json: {exc}")
        for step in compile_result.get("steps", []) or []:
            if step.get("returncode") not in (0, 1):
                self._log(
                    "Orchestrator",
                    f"LaTeX compile step failed: {step.get('cmd')} "
                    f"(rc={step.get('returncode')}): {str(step.get('stderr') or '')[:500]}",
                )
        if summary["pdf_exists"]:
            self._log(
                "Orchestrator",
                "LaTeX compilation produced paper.pdf"
                + ("" if summary["success"] else " (with step errors; see above)"),
            )
        else:
            tool = summary.get("missing_tool")
            detail = summary.get("message")
            if tool:
                note = (
                    f"LaTeX compilation produced NO paper.pdf: {tool} was not found. "
                    "Install a TeX distribution and put it on PATH."
                )
            elif detail:
                note = (
                    f"LaTeX compilation produced NO paper.pdf: {detail} "
                    "Details in latex_compile.json."
                )
            else:
                note = (
                    "LaTeX compilation produced NO paper.pdf; see paper.log "
                    "and latex_compile.json."
                )
            self._log("Orchestrator", note)
            self.ctx.errors.append(note)
            events.emit(self.ctx, "warning", stage="WRITING", code="NO_PDF", message=note)
        if summary["pdf_exists"] and summary.get("missing_tool"):
            # e.g. biber/bibtex missing: a PDF exists but every citation
            # renders as [?]. Say so where the user reads errors.
            note = (
                f"{summary['missing_tool']} was not found; paper.pdf was built "
                "without its bibliography step, so citations render as [?]."
            )
            self._log("Orchestrator", note)
            self.ctx.errors.append(note)
        events.emit(
            self.ctx,
            "compile.end",
            stage="WRITING",
            plain="Paper compiled" if summary["pdf_exists"] else "Paper did not compile",
            pdf_exists=summary["pdf_exists"],
            missing_tool=summary.get("missing_tool"),
            failed_step=summary.get("failed_step"),
        )
        return summary

    def _expand_literature_for_depth(self) -> None:
        """Arc P3: widen literature_context.papers toward the venue norm.

        Non-fatal by construction — a failure here must never cost the
        run its paper, so any problem degrades to the original selection.
        """
        try:
            # NOTE: composition_age_profile comes from src.citations, NOT
            # src.manuscript_linter — the linter's venue_age_profile returns
            # a 3-tuple and would raise here.
            from src.citations import (
                composition_age_profile,
                expand_literature_pool,
                venue_citation_target,
            )

            lit = self.ctx.literature_context or {}
            selected = lit.get("papers") or []
            pool = (getattr(self.ctx, "retrieved_literature", None) or {}).get(
                "papers"
            ) or []
            venue = self.config.get("review_gate", {}).get("venue")
            target = venue_citation_target(venue)
            if not target or not pool:
                self._log(
                    "Orchestrator",
                    f"Citation depth: no expansion (venue={venue}, "
                    f"target={target}, pool={len(pool)}, "
                    f"selected={len(selected)})",
                )
                return
            # Arc P5: compose across age bins rather than appending in the
            # pool's year-descending order. Without `profile=` the append
            # path returns the newest N, which is how a manuscript ended up
            # citing nothing published before 2024 while the citation COUNT
            # metric read green — and the linter now errors on exactly that.
            depth_stats: dict[str, Any] = {}
            expanded = expand_literature_pool(
                selected,
                pool,
                target,
                profile=composition_age_profile(venue),
                now_year=datetime.utcnow().year,
                stats=depth_stats,
            )
            if len(expanded) <= len(selected):
                self._log(
                    "Orchestrator",
                    f"Citation depth: pool exhausted at {len(expanded)} "
                    f"papers (target {target} for {venue})",
                )
            achieved = depth_stats.get("achieved") or {}
            if achieved:
                self._log(
                    "Orchestrator",
                    "Citation recency composition: "
                    + ", ".join(f"{b}={achieved.get(b, 0)}" for b in achieved)
                    + (f" (degraded signals: {depth_stats['degraded']})"
                       if depth_stats.get("degraded") else ""),
                )
                try:
                    path = os.path.join(
                        self.ctx.output_dir, "citation_depth_report.json"
                    )
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(depth_stats, f, indent=2, default=str)
                except OSError:
                    pass
            self.ctx.literature_context = {**lit, "papers": expanded}
            self._log(
                "Orchestrator",
                f"Citation depth: {len(selected)} selected + pool of "
                f"{len(pool)} -> {len(expanded)} available references "
                f"(target {target} for {venue})",
            )
            path = os.path.join(
                self.ctx.output_dir, "literature_context_expanded.json"
            )
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.ctx.literature_context, f, indent=2)
        except Exception as exc:  # noqa: BLE001
            self._log(
                "Orchestrator",
                f"Citation depth expansion failed (non-fatal): {exc}",
            )

    def _run_reviewing(self) -> None:
        if "REVIEWING" in self.ctx.completed_stages:
            self.ctx.current_state = PipelineState.VERIFYING
            return
        self._log("Orchestrator", "Starting REVIEWING stage (LSAR quality gate)")
        try:
            gate = ReviewGate(
                config=self.config,
                output_dir=Path(self.ctx.output_dir),
                log_fn=self._log,
            )
            try:
                gate.event_fn = lambda etype, **kw: events.emit(
                    self.ctx, etype, stage="REVIEWING", **kw
                )
            except Exception:  # noqa: BLE001 - events are optional
                pass
            summary = gate.run_gate()
            self.ctx.review_gate_result = summary

            ran = summary.get("ran")
            if ran is None:  # a gate (or stub) that predates the flag
                ran = int(summary.get("cycles_used") or 0) > 0
                if not ran:
                    events.emit(
                        self.ctx,
                        "gate.skipped",
                        stage="REVIEWING",
                        plain="The review gate did not run",
                        reason=summary.get("skip_reason") or "unknown",
                    )
            if ran:
                score = summary.get("final_score")
                score_str = (
                    f"{score:.2f}" if isinstance(score, (int, float)) else str(score)
                )
                self._log(
                    "Orchestrator",
                    f"LSAR review gate: passed={summary.get('passed')}, "
                    f"cycles={summary.get('cycles_used')}, "
                    f"score={score_str}, "
                    f"rec={summary.get('final_recommendation')}",
                )
                note = _revision_problem(summary)
                if note:
                    # E2: a reviser that could not be configured or whose
                    # calls all failed used to leave only a gate log line;
                    # the paper was re-reviewed unrevised with no trace in
                    # the run's errors.
                    self._log("Orchestrator", f"WARNING: {note}")
                    self.ctx.errors.append(note)
            else:
                # Not a failed review: nobody reviewed the paper. The old
                # line here printed "passed=False, score=0.00".
                self._log(
                    "Orchestrator",
                    "LSAR review gate did NOT run "
                    f"({summary.get('skip_reason') or 'reason unknown'}); the "
                    "paper was not reviewed.",
                )
        except Exception as e:
            self._log("Orchestrator", f"REVIEWING failed (non-fatal): {e}")
            self.ctx.review_gate_result = {
                "error": str(e),
                "ran": False,
                "skip_reason": f"exception: {_one_line(e, 200)}",
                "passed": None,
                "final_score": None,
                "cycles_used": 0,
            }
            events.emit(
                self.ctx,
                "gate.skipped",
                stage="REVIEWING",
                plain="The review gate could not run",
                reason=self.ctx.review_gate_result["skip_reason"],
            )

        # The gate records a verdict; VERIFYING is what decides whether
        # the run is releasable. Proceeding unconditionally here is fine
        # now -- it was not, when COMPLETED was the next state and
        # nothing anywhere read summary["passed"].
        self.ctx.completed_stages.append("REVIEWING")
        self.ctx.current_state = PipelineState.VERIFYING
        self._log("Orchestrator", "REVIEWING stage complete → VERIFYING")
        self._save_checkpoint()
        self._check_cost()
        self._update_findings_memory()

    # ------------------------------------------------------------------
    # VERIFYING — hold the finished manuscript against the run's own files
    # ------------------------------------------------------------------

    def _run_verifying(self) -> None:
        """Run the deterministic invariant battery and record a verdict.

        This is the first stage in the pipeline that sees the finished
        manuscript and the artifacts at the same time. The Critic runs
        before the Writer and has never seen a paper; the LSAR gate runs
        after but reads a 48,000-character head-slice with no figures in
        it, and until now nothing anywhere read its ``passed`` field.

        The stage always writes ``invariants.json`` and
        ``run_status.json``. Whether a critical finding actually stops
        the run is ``verification.blocking`` in config, default False --
        every one of the fourteen archived papers fails at least one of
        these checks, so switching it on globally on day one would
        terminate every run INCOMPLETE and teach nobody anything.
        Promote checks to blocking individually, on evidence.
        """
        if "VERIFYING" in self.ctx.completed_stages:
            # The stage finishes by moving to COMPLETED or INCOMPLETE, so a
            # run only gets here from a checkpoint that names VERIFYING as
            # the stage to run although it is recorded complete: an
            # interrupt between the two, or a hand edit to re-run it.
            # _prepare_resume has removed the verdict files, so declaring
            # the run COMPLETED here released a paper nothing had checked
            # (the rebuilt status said so). The battery is deterministic
            # and cheap: run it again.
            self._log(
                "Orchestrator",
                "VERIFYING is recorded complete but is the stage to run; "
                "running the invariant battery again",
            )
            self.ctx.completed_stages = [
                s for s in self.ctx.completed_stages if s != "VERIFYING"
            ]
        self._log("Orchestrator", "Starting VERIFYING stage (invariant battery)")

        cfg = self.config.get("verification", {}) or {}
        enabled = cfg.get("enabled", True)
        blocking = bool(cfg.get("blocking", False))
        blocking_codes = set(cfg.get("blocking_codes") or [])

        payload: dict[str, Any] = {"enabled": bool(enabled)}
        findings: list = []
        if enabled:
            try:
                from src.invariants import findings_to_json, run_invariants

                findings = run_invariants(self.ctx.output_dir)
                payload = {**payload, **findings_to_json(findings)}
            except Exception as exc:  # noqa: BLE001
                # A broken detector must not look like a clean run.
                payload["error"] = f"{type(exc).__name__}: {exc}"
                self._log("Orchestrator", f"VERIFYING battery failed: {exc}")

        try:
            with open(
                os.path.join(self.ctx.output_dir, "invariants.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(payload, f, indent=2, default=str)
        except OSError as exc:
            self._log("Orchestrator", f"Could not write invariants.json: {exc}")

        # Instruction compliance. Nothing in this pipeline has ever been
        # able to say whether a Critic instruction was acted on: the one
        # archived instruction whose disposition is documented was
        # applied WITHOUT DISCLOSURE, and two others were ignored, and
        # none of the three left a trace anywhere. An obligation closes
        # when its own test passes against the produced manuscript, never
        # because an agent reported compliance.
        obligations_summary: dict = {}
        try:
            from src.obligations import (
                derive_obligations,
                evaluate_obligations,
                summarize,
            )

            obligations = derive_obligations(self.ctx.review_report)
            if obligations:
                paper_path = os.path.join(self.ctx.output_dir, "paper.tex")
                paper = None
                if os.path.exists(paper_path):
                    with open(paper_path, encoding="utf-8", errors="replace") as f:
                        paper = f.read()
                obligations_summary = summarize(
                    evaluate_obligations(obligations, paper)
                )
                with open(
                    os.path.join(self.ctx.output_dir, "obligations.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(obligations_summary, f, indent=2, default=str)
                by = obligations_summary["by_status"]
                self._log(
                    "Orchestrator",
                    f"Writer obligations: {obligations_summary['n_obligations']} "
                    f"({', '.join(f'{k}={v}' for k, v in sorted(by.items()))})",
                )
        except Exception as exc:  # noqa: BLE001
            self._log("Orchestrator", f"Obligation evaluation failed: {exc}")

        # Optional LLM judge over the finished manuscript. Strictly
        # opt-in (`verification.judge_enabled`), and OFF by default.
        #
        # Off is the honest default for two reasons. It costs a call per
        # run plus one per figure, and -- more importantly -- an A/B that
        # measures whether the pipeline PRODUCES fewer defects must not
        # have a judge in one arm FINDING more of them. Its marginal
        # recovery over the deterministic battery is measured separately,
        # offline, against the answer key.
        verification_report: dict = {}
        if cfg.get("judge_enabled", False):
            try:
                from src.agents.verifier import Verifier

                verifier = Verifier(self.ctx, self.config)
                verifier.skills = self._match_skills_for_stage("Verifier")
                verification_report = verifier.run()
                vf = verification_report.get("findings") or []
                self._log(
                    "Orchestrator",
                    f"Verifier: {len(vf)} finding(s) kept, "
                    f"{verification_report.get('n_dropped_by_validator', 0)} "
                    "dropped by the validator",
                )
            except Exception as exc:  # noqa: BLE001
                self._log("Orchestrator", f"Verifier failed (non-fatal): {exc}")
                verification_report = {"ran": False, "error": str(exc)}

        criticals = [f for f in findings if f.severity == "critical"]
        gate = self._gate_block()
        gate_failed = gate["ran"] and gate["passed"] is False
        gate_not_run = gate["enabled"] and not gate["ran"]
        review = self.ctx.review_report or {}
        unverified = bool(review.get("unverified"))
        literature = self._literature_block()
        lit_degraded = bool(literature and literature.get("degraded"))
        verification_error = payload.get("error")
        verification_ran = bool(enabled) and not verification_error

        if blocking_codes:
            blockers = [f for f in criticals if f.code in blocking_codes]
        elif blocking:
            blockers = criticals
        else:
            blockers = []

        # A battery that crashed evaluated nothing -- including the codes
        # configured to block release. When anything is configured to
        # block, "could not check" cannot mean "passed": the run is held
        # back. In purely advisory mode it is released, but never as
        # clean (the old record said "clean" with the crash sitting in
        # invariants.json).
        battery_blocks = bool(verification_error) and bool(blocking_codes or blocking)
        released = not blockers and not battery_blocks

        advisories = [
            f"{len(criticals)} critical invariant finding(s)" if criticals else "",
            (
                f"verification did not run: {verification_error}"
                if verification_error
                else ("verification is disabled" if not enabled else "")
            ),
            (
                "review gate did not pass"
                + (
                    f" (score {gate['score']:.2f} vs threshold {gate['threshold']})"
                    if isinstance(gate["score"], float)
                    and isinstance(gate["threshold"], (int, float))
                    else ""
                )
                if gate_failed
                else ""
            ),
            (
                f"review gate did not run ({gate['skip_reason']})"
                if gate_not_run
                else ""
            ),
            (
                "the revised paper was not re-reviewed"
                + (
                    f" ({gate['last_cycle_failure']})"
                    if gate.get("last_cycle_failure")
                    else ""
                )
                if gate.get("final_manuscript_reviewed") is False
                else ""
            ),
            "critic verdict was not PASS" if unverified else "",
            (
                f"literature retrieval degraded ({literature.get('n_papers')} papers)"
                if lit_degraded and literature
                else ""
            ),
        ]
        advisories = [a for a in advisories if a]
        reason = "; ".join(advisories) or "clean"

        # One headline code, most serious first; ``reason`` still lists
        # every advisory. A paper the Critic never passed outranks a gate
        # that could not run: the first is about the manuscript, the
        # second about the installation.
        if blockers:
            reason_code = "BLOCKING_FINDINGS"
        elif not verification_ran:
            reason_code = "VERIFICATION_NOT_RUN"
        elif gate_failed:
            reason_code = "GATE_FAILED"
        elif unverified:
            reason_code = "CRITIC_UNVERIFIED"
        elif gate_not_run:
            reason_code = "GATE_NOT_RUN"
        elif criticals or lit_degraded:
            reason_code = "ADVISORY_FINDINGS"
        else:
            reason_code = "CLEAN"

        state = "COMPLETED" if released else "INCOMPLETE"
        status = self._write_run_status(
            {
                "state": state,
                "released": released,
                "reason": reason,
                "reason_code": reason_code,
                # What did NOT stop the release, stated separately so a
                # console can print "Release: YES" without a list of
                # things that read like blockers after it.
                "advisories": advisories if released else [],
                "verification": {
                    "enabled": bool(enabled),
                    "ran": verification_ran,
                    "error": verification_error,
                },
                "blocking_mode": "codes" if blocking_codes else ("all" if blocking else "advisory"),
                "invariant_counts": payload.get("counts", {}),
                "invariant_codes": payload.get("codes", []),
                "blocking_findings": [f.code for f in blockers],
                "writer_obligations": {
                    k: obligations_summary.get(k)
                    for k in ("n_obligations", "by_status", "compliance_rate")
                }
                if obligations_summary
                else None,
                "verifier": {
                    "ran": bool(verification_report.get("ran")),
                    "n_findings": len(verification_report.get("findings") or []),
                    "n_dropped_by_validator": verification_report.get(
                        "n_dropped_by_validator"
                    ),
                }
                if verification_report
                else None,
            }
        )

        counts = payload.get("counts", {})
        self._log(
            "Orchestrator",
            "Invariant battery: "
            f"{counts.get('critical', 0)} critical, {counts.get('major', 0)} major, "
            f"{counts.get('minor', 0)} minor "
            f"({', '.join(payload.get('codes', [])) or 'none'})",
        )
        events.emit(
            self.ctx,
            "verify.end",
            stage="VERIFYING",
            plain=(
                "Paper released" if released else "Paper held back: not fit to release"
            ),
            released=released,
            reason_code=reason_code,
            counts=counts,
        )

        self.ctx.completed_stages.append("VERIFYING")
        if not released:
            self.ctx.current_state = PipelineState.INCOMPLETE
            if blockers:
                self.ctx.errors.append(
                    "Release blocked by "
                    f"{len(blockers)} critical invariant finding(s): "
                    + ", ".join(sorted({f.code for f in blockers}))
                )
            else:
                self.ctx.errors.append(
                    "Release blocked: the invariant battery could not run "
                    f"({verification_error}), so the checks configured to "
                    "block release were never evaluated"
                )
            self._log(
                "Orchestrator",
                f"VERIFYING: release BLOCKED → INCOMPLETE ({status['reason']})",
            )
        else:
            self.ctx.current_state = PipelineState.COMPLETED
            self._log(
                "Orchestrator",
                f"VERIFYING stage complete → COMPLETED ({status['reason']})",
            )
        self._save_checkpoint()

    # ------------------------------------------------------------------
    # Revision cascade (SPEC §5.3)
    # ------------------------------------------------------------------

    def _execute_revisions(self) -> None:
        """Re-run the targeted agent and everything downstream of it.

        ``revision_instructions`` may now carry a ``Writer`` key, which
        this loop deliberately ignores: CRITIQUING precedes WRITING, so
        at revision time the Writer has not run and there is nothing to
        re-run. Those instructions are obligations on the manuscript the
        Writer is about to produce, and they are delivered at WRITING by
        :meth:`_writer_obligations`.
        """
        if not self.ctx.review_report:
            return
        agent_order = self.task_template.get_agent_order()
        instructions = self.ctx.review_report["revision_instructions"]
        targeted = [a for a in agent_order if instructions.get(a)]
        if not targeted:
            return
        start_idx = agent_order.index(targeted[0])
        for agent_name in agent_order[start_idx:]:
            self._run_agent(agent_name, revision_instructions=instructions.get(agent_name))

    def _run_agent(
        self,
        agent_name: str,
        revision_instructions: Optional[str] = None,
    ) -> None:
        agent_map = {
            "ProblemFormulator": self.problem_formulator,
            "DataEngineer": self.data_engineer,
            "Analyst": self.analyst,
        }
        agent = agent_map[agent_name]
        self._inject_skills(agent, agent_name)
        if agent_name == "ProblemFormulator":
            # Phase 3b.6 / 6.6 wire-up: in revision mode, the PF must see
            # the cycle-0 spec to preserve its locked invariants (estimand,
            # method battery, methodological_concerns, etc.). 3b.5's
            # F-LOCKED-SPEC-INVARIANTS evidence: without this wiring, PF
            # cycle-1 re-derived everything from scratch and silently
            # dropped the cycle-0 ESC-07 flag, renamed methods, and
            # restricted the adjustment set without justification.
            #
            # Preference order for the spec to preserve:
            #   1. self.ctx.research_spec (the cycle-0 PF refinement) when
            #      it exists — captures both the original locked
            #      invariants AND PF's prior analysis (e.g., flagged
            #      ESC-07 concerns).
            #   2. self.ctx.locked_research_spec (the CLI-loaded spec) as
            #      a fallback for the unlikely case of revising before
            #      cycle 0 emitted anything.
            prior_spec = (
                self.ctx.research_spec or self.ctx.locked_research_spec
            )
            result = agent.run(
                revision_instructions=revision_instructions,
                locked_research_spec=prior_spec,
            )
        else:
            result = agent.run(revision_instructions=revision_instructions)
        if agent_name == "ProblemFormulator":
            self.ctx.research_spec = self._carry_locked_guidance(
                result.get("research_spec")
            )
            self.ctx.literature_context = result.get("literature_context")
            self.ctx.retrieved_literature = result.get("retrieved_literature")
            self._save_formulating_outputs()
            self._note_literature_status()
        elif agent_name == "DataEngineer":
            self.ctx.data_report = result
            _emit_sample_metric(self.ctx, result, stage="REVISING")
        elif agent_name == "Analyst":
            self.ctx.results_object = result
            _emit_results_metric(self.ctx, result, stage="REVISING")

    # ------------------------------------------------------------------
    # Output file helpers
    # ------------------------------------------------------------------

    def _save_formulating_outputs(self) -> None:
        """Persist research_spec.json and literature_context.json to the run directory."""
        if self.ctx.research_spec is not None:
            path = os.path.join(self.ctx.output_dir, "research_spec.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.ctx.research_spec, f, indent=2)
        if self.ctx.literature_context is not None:
            path = os.path.join(self.ctx.output_dir, "literature_context.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.ctx.literature_context, f, indent=2)
        # Arc P3: the full retrieved pool the Writer draws depth from.
        # Persisted so a resumed run (and post-hoc analysis of how deep
        # the pool actually was) does not lose it.
        if getattr(self.ctx, "retrieved_literature", None) is not None:
            path = os.path.join(self.ctx.output_dir, "retrieved_literature.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.ctx.retrieved_literature, f, indent=2)

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _pre_critic_short_circuit(self, pre_result: PreCriticResult) -> None:
        """Act on critical pre-review findings without calling the Critic.

        Every critical finding used to end the run as PRE_CRITIC_ABORT,
        although each names a target agent and revision instructions were
        built for it; the synthesised verdict was ABORT whenever this code
        ran, so its REVISE branch was unreachable. On a real study the
        Analyst had not run the nested comparison an "above and beyond"
        question promised (pcc_07), which one Analyst revision can add,
        and the paid run stopped as not resumable.

        Now a finding no revision can fix still stops the run
        (PRE_CRITIC_ABORT). When every critical finding is revisable, the
        targeted agents are re-run through the ordinary REVISING cascade
        while cycles remain, and the check runs again at the next
        CRITIQUING. A revisable finding still failing when the cycles run
        out stops the run as PRE_CRITIC_UNRESOLVED. It does NOT fall
        through to WRITING (UNVERIFIED) the way an unresolved Critic
        REVISE does: the finding is that the paper's central result is
        missing, and a paper built around that gap reads fluently and
        must not be written at all.

        It also stops as PRE_CRITIC_UNRESOLVED, with cycles left, when a
        finding the previous cycle sent back comes back with the agent's
        own word that another run will not clear it (``stop_on_repeat``:
        the not-run record with a reason the instruction asks for when
        the test cannot run, or a second timeout). Sending the identical
        instruction again spent every remaining cycle, each up to four
        executions at the full time limit, on an outcome the agent had
        already reported.
        """
        sent_back = self._pre_critic_checks_sent_back()
        futile = [
            f for f in pre_result.revisable_failures
            if f.stop_on_repeat and f.check_id in sent_back
        ]
        report = self._synthesize_pre_critic_report(pre_result)
        self.ctx.review_report = report
        cycles_left = self.ctx.revision_cycle < self.ctx.max_revision_cycles
        revise = (
            report["overall_verdict"] == "REVISE" and cycles_left and not futile
        )
        verdict = "REVISE" if revise else "ABORT"
        self._log(
            "Orchestrator",
            f"Pre-Critic guard found critical failures → short-circuit verdict: {verdict}",
        )
        checks = [f.to_dict() for f in pre_result.failures if f.severity == "critical"]

        if revise:
            self.ctx.revision_cycle += 1
            self.ctx.current_state = PipelineState.REVISING
            report["effective_verdict"] = "REVISE"
            driving = pre_result.revisable_failures
            rerun = [a for a, text in report["revision_instructions"].items() if text]
            targets = ", ".join(rerun)
            summary = "; ".join(f"{f.check_id}: {f.message}" for f in driving)
            self._log(
                "Orchestrator",
                f"Pre-Critic guard: revision cycle {self.ctx.revision_cycle} of "
                f"{self.ctx.max_revision_cycles} re-runs {targets} for "
                f"{', '.join(f.check_id for f in driving)}",
            )
            events.emit(
                self.ctx,
                "warning",
                stage="CRITIQUING",
                code="PRE_CRITIC_REVISE",
                message=_one_line(
                    f"Sent back to {targets} (revision {self.ctx.revision_cycle} "
                    f"of {self.ctx.max_revision_cycles}): {summary}"
                ),
                # The message is cut at 500 characters; an interface words
                # the event from these fields instead of parsing it.
                checks=[f.check_id for f in driving],
                targets=rerun,
                revision=self.ctx.revision_cycle,
                max_revisions=self.ctx.max_revision_cycles,
            )
        else:
            fatal = pre_result.fatal_failures
            if fatal or report["overall_verdict"] == "ABORT":
                code = "PRE_CRITIC_ABORT"
                lead = (fatal or pre_result.revisable_failures)[0]
                message = f"{lead.check_id}: {lead.message}"
                self.ctx.errors.append(
                    f"Pre-Critic guard issued ABORT: {pre_result.failures}"
                )
            else:
                code = "PRE_CRITIC_UNRESOLVED"
                used = f"{self.ctx.revision_cycle} of {self.ctx.max_revision_cycles}"
                if futile:
                    lead = futile[0]
                    message = (
                        f"{lead.check_id} was still failing after revision "
                        f"{used}, and another revision would not change it: "
                        f"{lead.stop_on_repeat}. {lead.message}"
                    )
                else:
                    lead = pre_result.revisable_failures[0]
                    message = (
                        f"{lead.check_id} was still failing when the revision "
                        f"cycles ran out ({used} used): {lead.message}"
                    )
                self.ctx.errors.append(
                    f"Pre-Critic guard: {code} with {used} revision cycles "
                    f"used: {pre_result.failures}"
                )
            report["overall_verdict"] = "ABORT"
            report["effective_verdict"] = "ABORT"
            report["stop_code"] = code
            self.ctx.abort_info = _abort_record(
                "CRITIQUING", code, message, checks=checks
            )
            events.emit(
                self.ctx,
                "error",
                stage="CRITIQUING",
                code=code,
                message=self.ctx.abort_info["message"],
            )
            self.ctx.current_state = PipelineState.ABORTED
            self._log("Orchestrator", f"Pre-Critic guard stopped the run [{code}]: {message}")

        events.emit(
            self.ctx,
            "verdict",
            stage="CRITIQUING",
            cycle=self.ctx.revision_cycle,
            plain=f"Automatic pre-review check: {verdict}",
            critic_score=report.get("overall_quality_score"),
            verdict=verdict,
            unverified=False,
            source="pre_critic",
        )

    def _pre_critic_checks_sent_back(self) -> set[str]:
        """Check ids the previous cycle's pre-review revision was for.

        Read from the review the run holds when CRITIQUING starts again:
        REVISING leaves it in place, and the checkpoint keeps it across a
        resume. Empty unless that review was a pre-review REVISE (a
        pre-review ABORT written by an older version sent nothing back).
        """
        prior = self.ctx.review_report
        if not isinstance(prior, dict):
            return set()
        if prior.get("_source") != "pre_critic_short_circuit":
            return set()
        if prior.get("effective_verdict") != "REVISE":
            return set()
        return {
            str(f.get("check_id"))
            for f in prior.get("pre_critic_findings") or []
            if isinstance(f, dict)
            and f.get("severity") == "critical"
            and f.get("revisable")
        }

    def _synthesize_pre_critic_report(self, pre_result: PreCriticResult) -> dict:
        """Build a minimal review_report from pre-critic failures without an LLM call.

        ``overall_verdict`` is ABORT when any critical finding cannot be
        fixed by a revision (or no revisable one names an agent the
        cascade can re-run), otherwise REVISE; the caller turns a REVISE
        with no cycles left into a stop.
        """
        order = list(self.task_template.get_agent_order())
        driving = [
            f for f in pre_result.revisable_failures if f.target_agent in order
        ]
        fatal = bool(pre_result.fatal_failures) or not driving
        verdict = "ABORT" if fatal else "REVISE"

        def _issues_for(agent: str) -> list[dict]:
            return [
                {
                    "severity": f.severity,
                    "category": f.check_id,
                    "description": f.message,
                    "recommendation": f.instruction,
                    "target_agent": agent,
                    "revisable": bool(f.revisable),
                }
                for f in pre_result.failures
                if f.target_agent == agent
            ]

        ri: dict[str, Optional[str]] = {
            "ProblemFormulator": None,
            "DataEngineer": None,
            "Analyst": None,
        }
        if driving:
            # The cascade re-runs the earliest targeted agent and every
            # agent after it. Major findings ride along for agents that
            # are re-run anyway; a major aimed further upstream must not
            # widen the cascade -- the Critic sees it next cycle.
            start = min(order.index(f.target_agent) for f in driving)
            rerun = set(order[start:])
            for agent in order:
                items = [f for f in driving if f.target_agent == agent]
                items += [
                    f for f in pre_result.failures
                    if f.severity == "major" and f.target_agent == agent
                    and agent in rerun
                ]
                if items:
                    ri[agent] = self._pre_critic_instruction_text(items)

        return {
            "overall_verdict": verdict,
            "overall_quality_score": 1,
            "problem_formulation_review": {"score": 5, "issues": _issues_for("ProblemFormulator")},
            "data_preparation_review": {"score": 1, "issues": _issues_for("DataEngineer")},
            "analysis_review": {"score": 1, "issues": _issues_for("Analyst")},
            "substantive_review": {
                "score": 1,
                "educational_meaningfulness": "Pre-Critic automated check failed before substantive review.",
                "issues": [],
            },
            "revision_instructions": ri,
            "pre_critic_findings": [f.to_dict() for f in pre_result.failures],
            "_source": "pre_critic_short_circuit",
        }

    def _pre_critic_instruction_text(self, items: list) -> str:
        """One agent's revision instructions from the pre-review check."""
        remaining = self.ctx.max_revision_cycles - self.ctx.revision_cycle - 1
        lines = [
            "The automatic pre-review check stopped this study before the "
            "methods review. Items marked REQUIRED must be fixed: the check "
            "runs again when this revision finishes, and "
            + (
                f"{remaining} more revision cycle(s) remain after this one."
                if remaining > 0
                else "this is the last revision cycle, so a REQUIRED item "
                "still failing then stops the study with no paper."
            ),
        ]
        for i, f in enumerate(items, 1):
            tag = "REQUIRED" if f.severity == "critical" else "also fix"
            lines.append(f"\n{i}. [{f.check_id}, {tag}] {f.instruction}")
        return "\n".join(lines)

    def _save_checkpoint(self) -> None:
        """Persist the context atomically (D1).

        Serialised to a string first, so a value json cannot encode raises
        before anything on disk changes; then written to a temp file and
        moved over checkpoint.json in one step. A kill or a full disk
        mid-write can no longer leave the run's only resume point
        half-written.
        """
        path = os.path.join(self.ctx.output_dir, "checkpoint.json")
        text = json.dumps(self.ctx.to_dict(), indent=2)
        _atomic_write_text(path, text)

    def _read_checkpoint(self) -> Optional[dict]:
        """Return the parsed checkpoint, None when there is none.

        Raises :class:`CheckpointCorruptError` naming the file when it
        exists but cannot be parsed, instead of a bare JSONDecodeError out
        of the constructor.
        """
        path = os.path.join(self.ctx.output_dir, "checkpoint.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (ValueError, UnicodeDecodeError) as exc:
            raise CheckpointCorruptError(
                f"{path} exists but is not valid JSON ({exc}). It is this "
                "run's resume point and cannot be read. Move it aside to "
                "start this directory afresh, or restore it from a copy."
            ) from exc
        if not isinstance(data, dict) or "current_state" not in data:
            raise CheckpointCorruptError(
                f"{path} does not look like an EDM-ARS checkpoint (no "
                "current_state). Move it aside to start this directory afresh."
            )
        return data

    def _adopt_checkpoint_identity(self, data: dict) -> None:
        """Make the context describe the checkpointed run (D2).

        ``--resume`` used to take dataset, task type and locked spec from
        the command line while every artifact came from the checkpoint,
        so the README's resume example re-typed a locked causal run as an
        HSLS prediction run with nothing warning anyone. The checkpoint
        wins; a disagreeing flag is reported, not obeyed.
        """
        def _warn(message: str) -> None:
            self._log("Orchestrator", f"WARNING: {message}")
            events.emit(
                self.ctx, "warning", code="RESUME_FLAGS_IGNORED", message=message
            )
            warnings.warn(message, RuntimeWarning, stacklevel=3)

        ck_dataset = data.get("dataset_name")
        if ck_dataset and ck_dataset != self.ctx.dataset_name:
            _warn(
                f"--dataset {self.ctx.dataset_name!r} disagrees with the "
                f"checkpoint in {self.ctx.output_dir}, which is a "
                f"{ck_dataset!r} run. Resuming it as {ck_dataset!r}."
            )
            self.ctx.dataset_name = ck_dataset
            if data.get("raw_data_path"):
                self.ctx.raw_data_path = data["raw_data_path"]
        ck_task = data.get("task_type")
        if ck_task and ck_task != self.ctx.task_type:
            _warn(
                f"task type {self.ctx.task_type!r} (from config or "
                f"--research-spec) disagrees with the checkpoint, which is a "
                f"{ck_task!r} run. Resuming it as {ck_task!r}."
            )
            self.ctx.task_type = ck_task
        if "locked_research_spec" in data:
            ck_locked = data.get("locked_research_spec")
            if (
                self.ctx.locked_research_spec is not None
                and ck_locked != self.ctx.locked_research_spec
            ):
                _warn(
                    "--research-spec differs from the locked spec this run "
                    "started with; the checkpoint's spec is kept."
                )
            self.ctx.locked_research_spec = ck_locked

    #: Fields ``_load_checkpoint`` does not copy: where the run lives and
    #: the live event handle; the identity fields, which
    #: ``_adopt_checkpoint_identity`` already settled (a relocated raw
    #: data path for the SAME dataset is kept); and the revision budget,
    #: which follows the current config.
    _NOT_RESTORED = frozenset(
        {
            "output_dir",
            "event_sink",
            "dataset_name",
            "raw_data_path",
            "task_type",
            "locked_research_spec",
            "max_revision_cycles",
        }
    )

    def _load_checkpoint(self, data: Optional[dict] = None) -> None:
        if data is None:
            data = self._read_checkpoint()
            if data is None:
                return
        loaded = PipelineContext.from_dict(data)
        # Mutate in place so agent references stay valid. Every dataclass
        # field is copied, so a field added later cannot be silently left
        # behind the way paper_outline and run_start_time were.
        for f in dataclasses.fields(PipelineContext):
            if f.name in self._NOT_RESTORED:
                continue
            setattr(self.ctx, f.name, getattr(loaded, f.name))
        # from_dict built a plain list for ctx.log; wrap it again so
        # agent notes keep reaching events.jsonl.
        events.attach(self.ctx, self.ctx.output_dir)
        self._resumed = True
        self._log("Orchestrator", f"Resumed from checkpoint (state={loaded.current_state})")

    # ------------------------------------------------------------------
    # Logging and cost tracking
    # ------------------------------------------------------------------

    def _log(self, agent: str, message: str) -> None:
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "agent": agent,
            "message": message,
        }
        # list.append, not self.ctx.log.append: the wrapped log would
        # mirror this line as an "agent.note" AND the emit below would
        # send it as a "log" event. One line, one event.
        list.append(self.ctx.log, entry)
        events.emit(
            self.ctx,
            "log",
            stage=getattr(self.ctx, "current_state", None),
            agent=agent,
            message=message,
        )
        log_path = os.path.join(self.ctx.output_dir, "pipeline.log")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{entry['timestamp']} [{agent}] {message}\n")

    def _check_cost(self) -> None:
        """Compare measured spend against the run budget (K1).

        This used to multiply the summed token count by a hardcoded
        0.000015 — $15 per million, an Anthropic-era rate left in the
        code after the stack moved to DeepSeek, where it overstates cost
        by roughly 40x. Rates now come from config.yaml and are applied
        to the prompt/completion split the provider actually reported.
        """
        from src.cost import load_pricing, load_usage_best, summarize

        budget = self.config["pipeline"].get("cost_budget_usd", 5.0)
        usages = load_usage_best(self.ctx.output_dir)
        if not usages:
            return
        summary = summarize(usages, load_pricing(self.config))
        if summary.cost_usd is None:
            self._log(
                "Orchestrator",
                f"Cost not priced: {summary.total_tokens:,} tokens across "
                f"{summary.n_calls} calls; no rate configured for "
                f"{', '.join(summary.unpriced_models)}. Add them under "
                "config.yaml pricing.per_million_tokens.",
            )
            return
        note = ""
        if summary.unpriced_models:
            note = (
                f" (LOWER BOUND — no rate for {', '.join(summary.unpriced_models)})"
            )
        if summary.cost_usd > budget:
            self._log(
                "Orchestrator",
                f"WARNING: measured cost ${summary.cost_usd:.4f} exceeds "
                f"budget ${budget:.2f}{note}",
            )
            # The line above reaches only pipeline.log (a "log" event is
            # not shown). Say it once where people look: the console, and
            # `edmars status` / the live view through the warning event.
            if not getattr(self, "_budget_warned", False):
                self._budget_warned = True
                events.emit(
                    self.ctx,
                    "warning",
                    stage=getattr(self.ctx, "current_state", None),
                    code="COST_OVER_BUDGET",
                    message=(
                        f"This study has cost about US${summary.cost_usd:.2f} so "
                        f"far, more than its spending warning of US${budget:.2f}. "
                        "It keeps running: the warning does not stop it."
                    ),
                )

    def _update_findings_memory(self) -> None:
        """Persist this run's findings to the cross-run memory store (non-fatal)."""
        if self.findings_memory is None:
            return
        try:
            run_id = os.path.basename(self.ctx.output_dir)
            start_time = getattr(self.ctx, "run_start_time", "")
            runtime_minutes: float | None = None
            if start_time:
                try:
                    start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
                    if start_dt.tzinfo is None:
                        # Checkpoints written before the stamp became
                        # timezone-aware hold naive UTC.
                        start_dt = start_dt.replace(tzinfo=timezone.utc)
                    now_dt = datetime.now(timezone.utc)
                    runtime_minutes = (now_dt - start_dt).total_seconds() / 60.0
                except Exception:
                    pass
            entry = RunEntry.from_pipeline_context(
                ctx=self.ctx,
                run_id=run_id,
                runtime_minutes=runtime_minutes,
                api_cost_usd=None,
            )
            # Every run reads memory.yaml when it starts and writes its
            # whole copy back when it ends, so two overlapping runs used to
            # lose the first finisher's entry, and two saves racing on the
            # fixed memory.yaml.tmp could interleave into invalid YAML
            # (which the next load silently treats as empty). Serialise
            # the write and re-read the file under the lock (D7).
            mem_path = self._findings_memory_path or getattr(
                self.findings_memory, "path", None
            )
            if not mem_path:
                self.findings_memory.runs = [
                    r for r in self.findings_memory.runs if r.run_id != run_id
                ]
                self.findings_memory.add_run(entry)
                self.findings_memory.save()
            else:
                with _exclusive_lock(
                    mem_path + ".lock", timeout_s=_FINDINGS_LOCK_TIMEOUT_S
                ) as locked:
                    if not locked:
                        self._log(
                            "Orchestrator",
                            "FindingsMemory update skipped: another run held "
                            f"{mem_path}.lock for {_FINDINGS_LOCK_TIMEOUT_S:.0f}s "
                            "(non-fatal)",
                        )
                        return
                    fresh = FindingsMemory.load(mem_path)
                    if not fresh.runs and self.findings_memory.runs:
                        # The file became unreadable since this run
                        # started; do not overwrite the history this run
                        # still holds with a near-empty file.
                        fresh = self.findings_memory
                    # A run resumed after an abort reaches this point a
                    # second time; its later outcome replaces the entry the
                    # abort wrote instead of counting the run twice.
                    fresh.runs = [r for r in fresh.runs if r.run_id != run_id]
                    fresh.add_run(entry)
                    fresh.save()
                    self.findings_memory = fresh
            self._log("Orchestrator", f"FindingsMemory updated: {run_id}")
        except Exception as exc:
            self._log("Orchestrator", f"FindingsMemory update failed (non-fatal): {exc}")

    def _abort(
        self,
        reason: str,
        code: Optional[str] = None,
        exc: Optional[BaseException] = None,
    ) -> None:
        """Stop the run in the current stage and record why.

        ``code`` is one of ``src.errors.ABORT_CODES``; when a caller has
        only the exception, ``exc`` is mapped with ``code_for_exception``.
        The record in ``ctx.abort_info`` is what ``--resume`` retries from
        and what ``run_status.json`` reports.
        """
        stage = _state_name(self.ctx.current_state)
        if stage == "INITIALIZED":
            stage = "FORMULATING"
        if code is None:
            code = code_for_exception(exc) if exc is not None else "UNKNOWN"
        self.ctx.abort_info = _abort_record(stage, code, reason)
        self.ctx.errors.append(reason)
        self.ctx.current_state = PipelineState.ABORTED
        self._log("Orchestrator", f"ABORTED: {reason} [{code}]")
        events.emit(
            self.ctx, "error", stage=stage, code=code, message=_one_line(reason)
        )
        self._save_checkpoint()
        self._update_findings_memory()
