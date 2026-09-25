"""The ``edmars`` command line.

Each command is a thin layer: it sets the output mode, imports the module
that does the work only when the command runs (so ``edmars --help`` and
``edmars version`` stay fast and work even when an optional part is
missing), and calls that module's function from the CLI spec's section 18.

Every command accepts ``--plain`` and ``--yes`` both before and after the
command name, because people type ``edmars run --yes`` far more often
than ``edmars --yes run``.

Errors reach the user as one or two plain sentences, never a traceback:
the traceback goes (with secrets removed) into ``last_error.txt`` in the
cache folder, and ``EDMARS_DEBUG=1`` shows it instead.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import json
import os
import platform
import sys
import traceback
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Annotated, Any, Optional, TypeVar

import click
import typer

from edmars import __version__, ui

ISSUES_URL = "https://github.com/cgpan/edm-ars-public/issues"

_HELP = (
    "EDM-ARS drafts education data-mining research papers from public-use "
    "datasets. New here? Run `edmars setup`, then `edmars new`."
)

app = typer.Typer(
    name="edmars",
    help=_HELP,
    add_completion=False,
    rich_markup_mode=None,
    pretty_exceptions_enable=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)
data_app = typer.Typer(
    help="List, download, import and check datasets.",
    rich_markup_mode=None,
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
app.add_typer(data_app, name="data")


# --- Shared options -------------------------------------------------------------

PlainOpt = Annotated[
    bool,
    typer.Option(
        "--plain",
        help="Plain text: no colour, symbols or animation (for screen readers and logs).",
    ),
]
YesOpt = Annotated[
    bool,
    typer.Option(
        "--yes",
        "-y",
        help="Ask no questions: use defaults, and stop with a message when an answer is needed.",
    ),
]
NoWatchOpt = Annotated[
    bool,
    typer.Option("--no-watch", help="Start it and return; follow it later with `edmars status`."),
]
AcceptOpt = Annotated[
    bool,
    typer.Option(
        "--accept-disclosure",
        help="Accept the disclosure (see `edmars disclaimer` and `edmars privacy`) without being asked.",
    ),
]
RunArg = Annotated[
    Optional[str],
    typer.Argument(
        help="A study folder, or part of its name (default: the running or latest study).",
        show_default=False,
    ),
]

#: Values of the options given before the command name (``edmars --plain x``).
_ROOT_FLAGS: dict[str, bool] = {"plain": False, "yes": False}


class TaskType(str, Enum):
    """Study types ``edmars run --type`` accepts."""

    prediction = "prediction"
    causal_soo = "causal_soo"
    causal_itr = "causal_itr"
    causal_did = "causal_did"
    psychometrics = "psychometrics"


class OpenWhat(str, Enum):
    """What ``edmars results --open`` opens."""

    pdf = "pdf"
    folder = "folder"
    summary = "summary"


def _modes(plain: bool, yes: bool) -> bool:
    """Apply --plain/--yes (from before or after the command); return the --yes state."""
    non_interactive = bool(yes or _ROOT_FLAGS["yes"])
    ui.set_plain(bool(plain or _ROOT_FLAGS["plain"]))
    ui.set_non_interactive(non_interactive)
    ui.set_machine_output(False)
    return non_interactive


# --- Error handling ---------------------------------------------------------------


class FeatureMissing(RuntimeError):
    """A command's module is not part of this build."""

    def __init__(self, module: str) -> None:
        self.module = module
        super().__init__(
            f"This part of EDM-ARS (edmars.{module}) is not included in this copy. "
            "Run `edmars update --check` to look for a complete version."
        )


def _module(name: str) -> ModuleType:
    """Import ``edmars.<name>`` when a command needs it."""
    full = f"edmars.{name}"
    try:
        return importlib.import_module(full)
    except ModuleNotFoundError as exc:
        if exc.name == full:
            raise FeatureMissing(name) from None
        raise


def _report_crash(exc: BaseException) -> None:
    """Tell the user something broke, and keep the details for a bug report."""
    try:
        from edmars import secrets

        redact: Callable[[str], str] = secrets.redact
    except Exception:  # pragma: no cover
        def redact(text: str) -> str:
            return text

    summary = redact(f"{type(exc).__name__}: {exc}".strip().rstrip(":"))
    saved: Path | None = None
    try:
        from edmars import paths

        folder = paths.ensure_dir(paths.cache_dir())
        saved = folder / "last_error.txt"
        details = "".join(traceback.format_exception(exc))
        saved.write_text(
            redact(details) + f"\nedmars {__version__}, Python {platform.python_version()}, "
            f"{platform.platform()}\n",
            encoding="utf-8",
        )
    except Exception:
        saved = None
    ui.fail(f"Something went wrong: {summary}")
    if saved is not None:
        ui.info(f"The technical details were saved to {saved}")
    ui.info(
        "If it happens again, run `edmars doctor --bundle` and attach the file it "
        f"makes to an issue at {ISSUES_URL}"
    )


F = TypeVar("F", bound=Callable[..., Any])


def _friendly(func: F) -> F:
    """Turn exceptions into short messages and exit codes."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except (click.exceptions.Exit, click.ClickException, click.exceptions.Abort, SystemExit):
            raise
        except KeyboardInterrupt:
            ui.say()
            ui.info("Cancelled.")
            raise typer.Exit(130) from None
        except ui.NonInteractiveError as exc:
            ui.fail(str(exc))
            raise typer.Exit(1) from None
        except FeatureMissing as exc:
            ui.fail(str(exc))
            raise typer.Exit(1) from None
        except Exception as exc:
            if _expected_error(exc):
                ui.fail(str(exc))
                raise typer.Exit(1) from None
            if os.environ.get("EDMARS_DEBUG", "").strip() not in ("", "0"):
                raise
            _report_crash(exc)
            raise typer.Exit(1) from None

    return wrapper  # type: ignore[return-value]


def _expected_error(exc: BaseException) -> bool:
    """Errors whose message is already written for the user."""
    try:
        from edmars.secrets import SecretStoreError
        from edmars.settings import SettingsError
    except Exception:  # pragma: no cover
        return False
    return isinstance(exc, (SettingsError, SecretStoreError))


# --- Helpers -------------------------------------------------------------------------


def _settings() -> dict[str, Any]:
    from edmars import settings

    return settings.load()


def _require_ack(settings: dict[str, Any], accept: bool) -> None:
    """Stop unless the current disclosure text has been accepted."""
    from edmars import disclosure

    if disclosure.ensure_acknowledged(settings, accept=accept):
        return
    ui.fail(
        "Before a study can start, you need to read and accept how EDM-ARS works: "
        "what leaves your computer, and what you are responsible for."
    )
    ui.info(
        "Run `edmars setup`, or read `edmars disclaimer` and `edmars privacy` and "
        "add --accept-disclosure."
    )
    raise typer.Exit(1)


def _find_run(settings: dict[str, Any], text: str) -> Path:
    """Resolve a RUN argument: a path, a folder name, or part of one."""
    from edmars import settings as settings_mod

    candidate = Path(text).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    base = settings_mod.studies_dir(settings)
    if (base / text).is_dir():
        return base / text
    matches = (
        sorted(p for p in base.iterdir() if p.is_dir() and text.casefold() in p.name.casefold())
        if base.is_dir()
        else []
    )
    if len(matches) == 1:
        return matches[0]
    if matches:
        shown = ", ".join(m.name for m in matches[:5])
        ui.fail(f'"{text}" matches several studies ({shown}). Type more of the folder name.')
        raise typer.Exit(1)
    ui.fail(f'No study folder matches "{text}". See your studies with `edmars runs`.')
    raise typer.Exit(1)


def _resolve_run(
    settings: dict[str, Any],
    run: str | None,
    *,
    prefer_active: bool,
    active_only: bool = False,
) -> Path:
    """The study a command is about: the one named, else the running or latest one."""
    if run:
        return _find_run(settings, run)
    runner = _module("runner")
    if prefer_active or active_only:
        active = runner.active_run()
        if active:
            return Path(active)
        if active_only:
            ui.fail("No study is running right now.")
            raise typer.Exit(1)
    latest = runner.latest_run(settings)
    if latest:
        return Path(latest)
    ui.fail("You have no studies yet. Start one with `edmars new`.")
    raise typer.Exit(1)


def _watch_then_results(run_dir: Path) -> None:
    """Show the live view; when the study ends, show its results."""
    view = _module("view")
    code = view.watch(run_dir, plain=ui.is_plain())
    if code == 0:
        results = _module("results")
        raise typer.Exit(results.show(run_dir))
    if code == 11:
        ui.info(
            "The study was stopped. Your finished steps are saved; continue it later "
            "with `edmars resume`."
        )
    else:
        ui.info("The study keeps running in the background. Check on it with `edmars status`.")
    raise typer.Exit(0)


def _after_start(run_dir: Path, watch: bool) -> None:
    if not watch:
        ui.info(
            "It runs in the background. Follow it with `edmars status`, and see the "
            "results with `edmars results` when it is done."
        )
        raise typer.Exit(0)
    _watch_then_results(run_dir)


def _preflight_confirm_launch(
    settings: dict[str, Any], plan: Any, *, yes: bool, watch: bool, allow_edit: bool
) -> str:
    """Check the plan, show the confirmation card, start the study.

    Returns "edit" when the user wants to change the plan; otherwise it
    ends the command itself (typer.Exit).
    """
    study = _module("study")
    with ui.status(
        "Checking that this study can run (the first check on a large dataset "
        "can take a few minutes)"
    ):
        checks = study.preflight(plan, settings)
    ui.show_checks(checks)
    if any(check.status == "fail" for check in checks):
        ui.fail("This study cannot start until the problems above are fixed.")
        raise typer.Exit(1)

    ui.panel("Ready to start", study.confirmation_card(plan, settings))
    if not yes:
        # Starting a study spends money on the user's AI account, so it is
        # never started on a default answer: without a terminal, --yes is
        # required.
        if not ui.is_interactive():
            raise ui.NonInteractiveError(
                "Start this study?", hint="Add --yes to start it without being asked."
            )
        choices = [("start", "Start the study")]
        if allow_edit:
            choices.append(("edit", "Change something"))
        choices.append(("cancel", "Cancel"))
        answer = ui.select("Start this study?", choices, default="start")
        if answer == "edit":
            return "edit"
        if answer == "cancel":
            ui.info("No study was started.")
            raise typer.Exit(0)

    runner = _module("runner")
    run_dir = Path(runner.launch(settings, plan))
    ui.ok(f"The study has started. Its folder: {run_dir}")
    _after_start(run_dir, watch)
    return "started"  # pragma: no cover - _after_start always exits


def _confirm_spend(question: str, yes: bool) -> None:
    """Ask before anything that uses the paid AI service again."""
    if yes:
        return
    if not ui.is_interactive():
        raise ui.NonInteractiveError(question, hint="Add --yes to go ahead without being asked.")
    if not ui.confirm(question, default=True):
        ui.info("Nothing was started.")
        raise typer.Exit(0)


def _call_with(func: Callable[..., Any], **available: Any) -> Any:
    """Call ``func`` passing only the arguments its signature names.

    Used for ``edmars review``, whose module function is not fixed by the
    CLI spec: common parameter spellings are mapped onto what we have.
    """
    aliases = {
        "run_dir": "run_dir",
        "run": "run_dir",
        "run_path": "run_dir",
        "path": "run_dir",
        "settings": "settings",
    }
    kwargs: dict[str, Any] = {}
    for name, param in inspect.signature(func).parameters.items():
        source = aliases.get(name)
        if source in available:
            kwargs[name] = available[source]
        elif param.default is inspect.Parameter.empty and param.kind not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise TypeError(f"cannot supply parameter {name!r} of {func.__qualname__}")
    return func(**kwargs)


def _print_document(text: str) -> None:
    if ui.is_plain():
        ui.say(text.rstrip())
        return
    from rich.markdown import Markdown

    ui.console.print(Markdown(text))


def _version_line() -> str:
    return (
        f"edmars {__version__} (Python {platform.python_version()}, "
        f"{platform.system()} {platform.machine()})"
    )


def _parse_options(pairs: list[str] | None) -> dict[str, Any]:
    """``["provider=deepseek", "lsar=no"]`` -> ``{"provider": "deepseek", "lsar": False}``."""
    options: dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise typer.BadParameter(f"expected KEY=VALUE, got {pair!r}", param_hint="--option")
        key, _, value = pair.partition("=")
        key = key.strip()
        if not key:
            raise typer.BadParameter(f"missing KEY in {pair!r}", param_hint="--option")
        lowered = value.strip().lower()
        if lowered in ("true", "yes", "on"):
            options[key] = True
        elif lowered in ("false", "no", "off"):
            options[key] = False
        else:
            options[key] = value.strip()
    return options


def _welcome() -> None:
    try:
        from edmars import disclosure

        first_time = not disclosure.is_acknowledged(_settings())
    except Exception:
        first_time = True
    ui.say(f"EDM-ARS {__version__}: research-paper drafts from public education datasets.")
    ui.say()
    if first_time:
        ui.say("First time here?   edmars setup")
    ui.say("Start a study:     edmars new")
    ui.say("Watch a study:     edmars status")
    ui.say("See the results:   edmars results")
    ui.say("Check this computer: edmars doctor")
    ui.say("All commands:      edmars --help")


# --- Root ----------------------------------------------------------------------------


@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    plain: PlainOpt = False,
    yes: YesOpt = False,
    version: Annotated[bool, typer.Option("--version", help="Show the version and exit.")] = False,
) -> None:
    _ROOT_FLAGS["plain"] = plain
    _ROOT_FLAGS["yes"] = yes
    _modes(plain, yes)
    if version:
        ui.say(_version_line())
        raise typer.Exit(0)
    if ctx.invoked_subcommand is None:
        _welcome()
        raise typer.Exit(0)


# --- Commands --------------------------------------------------------------------------


@app.command("setup")
@_friendly
def setup_cmd(
    section: Annotated[
        Optional[str],
        typer.Argument(help="Only this part of the setup (leave out to see the menu).", show_default=False),
    ] = None,
    option: Annotated[
        Optional[list[str]],
        typer.Option(
            "--option",
            "-o",
            help="Answer a setup question in advance, as KEY=VALUE (repeatable; for --yes).",
        ),
    ] = None,
    accept_disclosure: AcceptOpt = False,
    plain: PlainOpt = False,
    yes: YesOpt = False,
) -> None:
    """Set up EDM-ARS, or change one part of the setup."""
    non_interactive = _modes(plain, yes)
    options = _parse_options(option)
    if accept_disclosure:
        options["accept_disclosure"] = True
    wizard = _module("wizard")
    raise typer.Exit(wizard.run_setup(section, non_interactive=non_interactive, options=options or None))


@app.command("doctor")
@_friendly
def doctor_cmd(
    deep: Annotated[
        bool, typer.Option("--deep", help="Also test your keys online and compile a test PDF (slower).")
    ] = False,
    json_out: Annotated[bool, typer.Option("--json", help="Print the results as JSON.")] = False,
    bundle: Annotated[
        bool, typer.Option("--bundle", help="Make a support file (secrets removed) to attach to an issue.")
    ] = False,
    quick: Annotated[bool, typer.Option("--quick", hidden=True)] = False,
    plain: PlainOpt = False,
    yes: YesOpt = False,
) -> None:
    """Check that this computer is ready to run studies."""
    _modes(plain, yes)
    if json_out:
        ui.set_machine_output(True)
    doctor = _module("doctor")
    raise typer.Exit(doctor.main(deep=deep and not quick, json_out=json_out, bundle=bundle))


@app.command("new")
@_friendly
def new_cmd(
    no_watch: NoWatchOpt = False,
    accept_disclosure: AcceptOpt = False,
    plain: PlainOpt = False,
    yes: YesOpt = False,
) -> None:
    """Start a new study, answering a few questions."""
    non_interactive = _modes(plain, yes)
    if non_interactive or not ui.is_interactive():
        ui.fail(
            "`edmars new` asks questions, so it needs a terminal window. To start a study "
            "without questions, use `edmars run --type ... --yes` (see `edmars run --help`)."
        )
        raise typer.Exit(1)
    settings = _settings()
    _require_ack(settings, accept_disclosure)
    study = _module("study")
    while True:
        plan = study.new_study_interactive(settings)
        if plan is None:
            ui.info("No study was started.")
            raise typer.Exit(0)
        outcome = _preflight_confirm_launch(
            settings, plan, yes=False, watch=not no_watch, allow_edit=True
        )
        if outcome != "edit":
            return


@app.command("run")
@_friendly
def run_cmd(
    type_: Annotated[
        Optional[TaskType],
        typer.Option("--type", "-t", help="The kind of study (required).", case_sensitive=False),
    ] = None,
    prompt: Annotated[
        Optional[str], typer.Option("--prompt", help="Your research question (prediction studies).")
    ] = None,
    example: Annotated[
        Optional[str], typer.Option("--example", help="Start from one of the example studies (its ID).")
    ] = None,
    spec: Annotated[
        Optional[Path], typer.Option("--spec", help="A research spec file (JSON) to run as it is.")
    ] = None,
    dataset: Annotated[Optional[str], typer.Option("--dataset", help="The dataset to use.")] = None,
    venue: Annotated[
        Optional[str], typer.Option("--venue", help="Where the paper is aimed (default: EDM).")
    ] = None,
    no_watch: NoWatchOpt = False,
    accept_disclosure: AcceptOpt = False,
    plain: PlainOpt = False,
    yes: YesOpt = False,
) -> None:
    """Start a study from options instead of questions (for scripts and repeat runs)."""
    non_interactive = _modes(plain, yes)
    if type_ is None:
        # Validated here rather than by click so the message is a sentence,
        # and so --help does not show typer's escaped "\[required]" marker.
        kinds = ", ".join(t.value for t in TaskType)
        ui.fail(f"Say what kind of study to run with --type: one of {kinds}.")
        ui.info("Or answer a few questions instead: `edmars new`.")
        raise typer.Exit(2)
    settings = _settings()
    _require_ack(settings, accept_disclosure)
    study = _module("study")
    try:
        plan = study.plan_from_flags(
            settings,
            task_type=type_.value,
            prompt=prompt,
            example=example,
            spec_path=spec,
            dataset=dataset,
            venue=venue,
        )
    except (ValueError, FileNotFoundError, KeyError) as exc:
        message = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
        ui.fail(str(message))
        raise typer.Exit(1) from None
    _preflight_confirm_launch(settings, plan, yes=non_interactive, watch=not no_watch, allow_edit=False)


@app.command("status")
@_friendly
def status_cmd(run: RunArg = None, plain: PlainOpt = False, yes: YesOpt = False) -> None:
    """Watch a study's progress (the running one, or the latest)."""
    _modes(plain, yes)
    settings = _settings()
    run_dir = _resolve_run(settings, run, prefer_active=True)
    _watch_then_results(run_dir)


@app.command("runs")
@_friendly
def runs_cmd(
    json_out: Annotated[bool, typer.Option("--json", help="Print the list as JSON.")] = False,
    plain: PlainOpt = False,
    yes: YesOpt = False,
) -> None:
    """List your studies."""
    _modes(plain, yes)
    if json_out:
        ui.set_machine_output(True)
    runner = _module("runner")
    items = list(runner.list_runs(_settings()))
    if json_out:
        typer.echo(json.dumps(items, indent=2, default=str))
        return
    if not items:
        ui.info("No studies yet. Start one with `edmars new`.")
        return

    def pick(item: dict[str, Any], *keys: str) -> str:
        for key in keys:
            value = item.get(key)
            if value not in (None, ""):
                return str(value)
        return ""

    rows = []
    for item in items:
        folder = pick(item, "name", "folder") or Path(pick(item, "run_dir", "path")).name
        question = pick(item, "question", "research_question")
        if len(question) > 60:
            question = question[:57] + "..."
        rows.append([folder, pick(item, "started", "started_at"), pick(item, "label", "state", "status"), question])
    ui.table(["Study", "Started", "State", "Question"], rows)


@app.command("results")
@_friendly
def results_cmd(
    run: RunArg = None,
    open_: Annotated[
        Optional[OpenWhat],
        typer.Option("--open", help="Also open the PDF, the folder, or the summary page."),
    ] = None,
    plain: PlainOpt = False,
    yes: YesOpt = False,
) -> None:
    """Show a study's results in plain language."""
    _modes(plain, yes)
    settings = _settings()
    run_dir = _resolve_run(settings, run, prefer_active=False)
    results = _module("results")
    raise typer.Exit(results.show(run_dir, open_.value if open_ is not None else None))


@app.command("stop")
@_friendly
def stop_cmd(run: RunArg = None, plain: PlainOpt = False, yes: YesOpt = False) -> None:
    """Stop the running study (its finished steps are kept)."""
    non_interactive = _modes(plain, yes)
    settings = _settings()
    run_dir = _resolve_run(settings, run, prefer_active=True, active_only=run is None)
    if not non_interactive:
        if not ui.is_interactive():
            raise ui.NonInteractiveError("Stop this study?", hint="Add --yes to stop it without being asked.")
        if not ui.confirm(
            f"Stop the study in {run_dir.name}? Finished steps are saved and you can "
            "continue later with `edmars resume`.",
            default=False,
        ):
            ui.info("The study was not stopped.")
            raise typer.Exit(0)
    runner = _module("runner")
    with ui.status("Stopping the study (this can take up to half a minute)"):
        runner.stop(run_dir)
    ui.ok("Stopped. Continue it later with `edmars resume`.")


@app.command("resume")
@_friendly
def resume_cmd(
    run: RunArg = None,
    no_watch: NoWatchOpt = False,
    accept_disclosure: AcceptOpt = False,
    plain: PlainOpt = False,
    yes: YesOpt = False,
) -> None:
    """Continue a study that stopped before it finished."""
    non_interactive = _modes(plain, yes)
    settings = _settings()
    _require_ack(settings, accept_disclosure)
    run_dir = _resolve_run(settings, run, prefer_active=False)
    _confirm_spend(
        f"Continue the study in {run_dir.name}? It will use your AI service again.",
        non_interactive,
    )
    runner = _module("runner")
    runner.resume(run_dir)
    ui.ok("The study is running again.")
    _after_start(run_dir, watch=not no_watch)


@app.command("review")
@_friendly
def review_cmd(run: RunArg = None, plain: PlainOpt = False, yes: YesOpt = False) -> None:
    """Run the automated reviewer (LSAR) on a finished study's paper."""
    non_interactive = _modes(plain, yes)
    settings = _settings()
    lsar = _module("lsar")
    review = getattr(lsar, "review_run", None) or getattr(lsar, "review", None)
    if review is None:
        raise FeatureMissing("lsar.review")
    run_dir = _resolve_run(settings, run, prefer_active=False)
    _confirm_spend(
        f"Review the paper in {run_dir.name}? This sends it to DeepSeek and usually "
        "takes 20 to 40 minutes.",
        non_interactive,
    )
    outcome = _call_with(review, run_dir=run_dir, settings=settings)
    raise typer.Exit(outcome if isinstance(outcome, int) else 0)


# --- data -------------------------------------------------------------------------------

_DEFAULT_TERMS = (
    "This is public-use data from its provider (for the NCES datasets: the "
    "National Center for Education Statistics).\n"
    "- Use it only for statistical research, and make no attempt to identify "
    "any person or school.\n"
    "- Cite the provider as the source in anything you publish.\n"
    "- EDM-ARS is not affiliated with or endorsed by NCES/IES or any data provider."
)


def _dataset_info(datasets: ModuleType, name: str) -> Any:
    catalog = getattr(datasets, "CATALOG", {})
    if name not in catalog:
        known = ", ".join(sorted(catalog)) or "none"
        ui.fail(f'There is no dataset called "{name}". Known datasets: {known}.')
        raise typer.Exit(1)
    return catalog[name]


def _progress_printer() -> Callable[..., None]:
    """A download progress callback tolerant of (done, total) or (fraction,) calls."""
    state = {"last": -1}

    def report(*args: Any, **_kwargs: Any) -> None:
        numbers = [a for a in args if isinstance(a, (int, float))]
        if not numbers:
            return
        if len(numbers) >= 2 and numbers[1]:
            fraction = float(numbers[0]) / float(numbers[1])
        elif numbers[0] <= 1:
            fraction = float(numbers[0])
        else:
            return
        step = int(max(0.0, min(fraction, 1.0)) * 10)
        if step > state["last"]:
            state["last"] = step
            ui.say(f"  downloaded {step * 10}%")

    return report


def _record_dataset(
    settings: dict[str, Any], name: str, path: Path, check: Any, *, terms_accepted: bool = False
) -> None:
    from edmars import settings as settings_mod

    entry = dict(settings_mod.get(settings, f"datasets.{name}", {}) or {})
    entry["path"] = str(path)
    if terms_accepted:
        entry["terms_accepted_at"] = settings_mod.utc_now()
    if getattr(check, "status", None) == "ok":
        entry["verified_at"] = settings_mod.utc_now()
    settings_mod.set_(settings, f"datasets.{name}", entry)
    settings_mod.save(settings)


@data_app.command("list")
@_friendly
def data_list_cmd(plain: PlainOpt = False, yes: YesOpt = False) -> None:
    """Show the datasets EDM-ARS knows and whether each is ready."""
    _modes(plain, yes)
    datasets = _module("datasets")
    settings = _settings()
    from edmars.model import Check

    for name, info in datasets.CATALOG.items():
        check = datasets.status(name, settings)
        label = getattr(info, "label", None) or getattr(info, "title", None)
        title = f"{name} ({label})" if label and label != name else name
        ui.show_checks([Check(title, check.status, check.detail, check.fix)])


@data_app.command("install")
@_friendly
def data_install_cmd(
    name: Annotated[str, typer.Argument(help="The dataset's name, as `edmars data list` shows it.")],
    accept_terms: Annotated[
        bool, typer.Option("--accept-terms", help="Accept the data provider's terms without being asked.")
    ] = False,
    plain: PlainOpt = False,
    yes: YesOpt = False,
) -> None:
    """Download a public-use dataset, after you accept its terms."""
    _modes(plain, yes)
    datasets = _module("datasets")
    settings = _settings()
    info = _dataset_info(datasets, name)

    accepted = False
    if name == "did_els_hsls_panel":
        with ui.status("Building the combined ELS:2002 + HSLS:09 panel from your two datasets"):
            path = Path(datasets.build_did_panel(settings))
    else:
        terms = getattr(info, "terms", None) or _DEFAULT_TERMS
        if not accept_terms:
            if not ui.is_interactive():
                raise ui.NonInteractiveError(
                    "Do you accept the data provider's terms?",
                    hint="Read them with `edmars data install NAME` in a terminal, or add --accept-terms.",
                )
            ui.panel("Terms of use", str(terms))
            if not ui.confirm("Do you accept these terms?", default=False):
                ui.info("Nothing was downloaded.")
                raise typer.Exit(0)
        accepted = True
        ui.info("Downloading. Large files take a while; if it stops, run the same command to continue.")
        path = Path(
            datasets.download(name, datasets.raw_data_dir(settings), progress=_progress_printer())
        )
    check = datasets.validate_file(name, path)
    ui.show_checks([check])
    _record_dataset(settings, name, path, check, terms_accepted=accepted)
    raise typer.Exit(0 if check.status != "fail" else 1)


@data_app.command("import")
@_friendly
def data_import_cmd(
    name: Annotated[str, typer.Argument(help="The dataset's name, as `edmars data list` shows it.")],
    path: Annotated[
        Path,
        typer.Argument(help="The data file you downloaded yourself.", exists=True, dir_okay=False, readable=True),
    ],
    plain: PlainOpt = False,
    yes: YesOpt = False,
) -> None:
    """Use a dataset file you already downloaded."""
    _modes(plain, yes)
    datasets = _module("datasets")
    settings = _settings()
    _dataset_info(datasets, name)
    stored = Path(datasets.import_file(name, path, settings))
    check = datasets.validate_file(name, stored)
    ui.show_checks([check])
    _record_dataset(settings, name, stored, check)
    raise typer.Exit(0 if check.status != "fail" else 1)


@data_app.command("verify")
@_friendly
def data_verify_cmd(
    name: Annotated[
        Optional[str], typer.Argument(help="Only this dataset (default: all).", show_default=False)
    ] = None,
    plain: PlainOpt = False,
    yes: YesOpt = False,
) -> None:
    """Check that your datasets are present and in the expected format."""
    _modes(plain, yes)
    datasets = _module("datasets")
    settings = _settings()
    from edmars import settings as settings_mod

    names = [name] if name else list(datasets.CATALOG)
    if name:
        _dataset_info(datasets, name)
    checks = []
    for item in names:
        recorded = settings_mod.get(settings, f"datasets.{item}.path")
        if recorded and Path(str(recorded)).exists():
            checks.append(datasets.validate_file(item, Path(str(recorded))))
        else:
            checks.append(datasets.status(item, settings))
    ui.show_checks(checks)
    raise typer.Exit(1 if any(c.status == "fail" for c in checks) else 0)


# --- Information --------------------------------------------------------------------------


@app.command("explain")
@_friendly
def explain_cmd(
    term: Annotated[
        Optional[list[str]],
        typer.Argument(help="The word or abbreviation, e.g. AUC, SHAP, propensity score.", show_default=False),
    ] = None,
    plain: PlainOpt = False,
    yes: YesOpt = False,
) -> None:
    """Explain a term from the results in plain language."""
    _modes(plain, yes)
    from edmars import explain as glossary

    words = " ".join(term or []).strip()
    if not words:
        ui.say("Terms I can explain: " + ", ".join(glossary.term_names()) + ".")
        ui.say("For example: edmars explain AUC")
        return
    ui.say(glossary.explain(words))
    raise typer.Exit(0 if glossary.lookup(words) else 1)


@app.command("privacy")
@_friendly
def privacy_cmd(plain: PlainOpt = False, yes: YesOpt = False) -> None:
    """What EDM-ARS sends off your computer, to whom, and what stays."""
    _modes(plain, yes)
    from edmars import disclosure

    _print_document(disclosure.privacy_text())


@app.command("disclaimer")
@_friendly
def disclaimer_cmd(plain: PlainOpt = False, yes: YesOpt = False) -> None:
    """The disclaimer: what EDM-ARS output is, and what you are responsible for."""
    _modes(plain, yes)
    from edmars import disclosure

    _print_document(disclosure.disclaimer_text())


@app.command("version")
@_friendly
def version_cmd(plain: PlainOpt = False, yes: YesOpt = False) -> None:
    """Show the version of EDM-ARS."""
    _modes(plain, yes)
    ui.say(_version_line())


@app.command("update")
@_friendly
def update_cmd(
    check: Annotated[bool, typer.Option("--check", help="Only check whether a newer version exists.")] = False,
    plain: PlainOpt = False,
    yes: YesOpt = False,
) -> None:
    """Check for a newer version of EDM-ARS and say how to install it."""
    _modes(plain, yes)
    maintenance = _module("maintenance")
    raise typer.Exit(maintenance.update(check_only=check))


@app.command("uninstall")
@_friendly
def uninstall_cmd(
    remove_datasets: Annotated[
        Optional[bool],
        typer.Option("--remove-datasets/--keep-datasets", help="Also delete downloaded datasets (default: ask)."),
    ] = None,
    remove_studies: Annotated[
        Optional[bool],
        typer.Option("--remove-studies/--keep-studies", help="Also delete your study folders (default: ask)."),
    ] = None,
    plain: PlainOpt = False,
    yes: YesOpt = False,
) -> None:
    """Remove EDM-ARS's settings, stored keys and downloads from this computer."""
    non_interactive = _modes(plain, yes)
    maintenance = _module("maintenance")
    raise typer.Exit(
        maintenance.uninstall(
            assume_yes=non_interactive,
            remove_datasets=remove_datasets,
            remove_studies=remove_studies,
        )
    )


def main(argv: list[str] | None = None) -> None:
    """Run the command line (the launcher and ``python -m edmars`` call this)."""
    app(args=argv, prog_name="edmars")


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
