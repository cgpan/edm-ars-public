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
from edmars.model import EXIT_CODES_HELP, EXIT_ERROR, EXIT_READY, EXIT_STOPPED

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


class PaperFormat(str, Enum):
    """What ``edmars run --paper-format`` accepts."""

    conference = "conference"
    journal = "journal"


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
    """Errors whose message is already written for the user.

    Settings and key-store errors, the runner's refusals (another study
    is running, nothing to resume), the study flow's refusals, and every
    error that marks itself ``user_facing`` (a wrong dataset file, a full
    disk, a failed download): the message is the whole story, so it is
    printed as it is, not as a crash report.
    """
    if getattr(exc, "user_facing", False):
        return True
    expected: list[type[BaseException]] = []
    for module, name in (
        ("edmars.settings", "SettingsError"),
        ("edmars.secrets", "SecretStoreError"),
        ("edmars.runner", "RunnerError"),
        ("edmars.study", "StudyError"),
    ):
        try:
            cls = getattr(importlib.import_module(module), name)
        except Exception:  # noqa: BLE001 -- a missing part cannot have raised it
            continue
        if isinstance(cls, type) and issubclass(cls, BaseException):
            expected.append(cls)
    return isinstance(exc, tuple(expected))


# --- Helpers -------------------------------------------------------------------------


def _exit(code: Any) -> typer.Exit:
    """An Exit carrying a module's return value (anything but an int means 0)."""
    return typer.Exit(code if isinstance(code, int) else 0)


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


def _require_ai_key(settings: dict[str, Any]) -> None:
    """Stop `edmars new` before the first question when no AI key is saved.

    The pipeline's own start-up check would find it too, but only after
    every question and the Start button, and the answers would be lost.
    """
    from edmars import providers
    from edmars import secrets as edsecrets
    from edmars import settings as settings_mod

    provider = str(settings_mod.get(settings, "provider", "deepseek") or "deepseek")
    info = providers.PROVIDERS.get(provider)
    if provider == "local" or info is None:
        return  # a local server needs no key; an unknown one is setup's to explain
    if edsecrets.secret_source(info.env_var):
        return
    name = info.label.split(" (")[0].split(" - ")[0]
    ui.fail(f"There is no {name} key saved on this computer, so a study cannot start yet.")
    ui.info("Add it with `edmars setup ai`, then run `edmars new` again. Nothing was spent.")
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


def _outside_studies(settings: dict[str, Any], run_dir: Path) -> bool:
    """True when ``run_dir`` is not inside this user's studies folder."""
    from edmars import settings as settings_mod

    try:
        run_dir.resolve().relative_to(settings_mod.studies_dir(settings).resolve())
    except (ValueError, OSError):
        return True
    return False


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
    """Show the live view; when the study ends, show its results.

    Exits with the result screen's code (edmars.model.EXIT_CODES_HELP);
    the view's own return values (10 left running, 11 stopped from the
    view) never reach the shell.
    """
    view = _module("view")
    code = view.watch(run_dir, plain=ui.is_plain())
    if code == 0:
        results = _module("results")
        raise _exit(results.show(run_dir))
    if code == 11:
        ui.info(
            "The study was stopped. Your finished steps are saved; continue it later "
            "with `edmars resume`."
        )
        raise typer.Exit(EXIT_STOPPED)
    if code == 1:
        raise typer.Exit(EXIT_ERROR)
    ui.info("The study keeps running in the background. Check on it with `edmars status`.")
    raise typer.Exit(EXIT_READY)


def _after_start(run_dir: Path, watch: bool) -> None:
    if not watch:
        ui.info(
            "It runs in the background. Follow it with `edmars status`, and see the "
            "results with `edmars results` when it is done."
        )
        raise typer.Exit(0)
    _watch_then_results(run_dir)


def _pipeline_check(settings: dict[str, Any], plan: Any) -> list[Any]:
    """Run the pipeline's own start-up check; stop the command if it fails.

    The feasibility check asks "can this question be answered with this
    data?"; this one asks "would the pipeline start?" (keys, models, the
    data file, R, the reviewer), using the exact config and environment a
    launch uses, so a study that would stop in its first second never
    gets a folder. Returns its checks when the study may start.
    """
    runner = _module("runner")
    with ui.status("Checking that the pipeline can start this study"):
        checks = runner.pipeline_check(settings, plan)
    failed = [check for check in checks if check.status == "fail"]
    shown = [check for check in checks if check.status in ("fail", "warn")]
    if shown:
        ui.show_checks(shown)
    if failed:
        ui.fail("The study was not started: fix the problems above, then try again.")
        raise typer.Exit(1)
    return list(checks)


#: What the card says when the pipeline's pre-start check found that the
#: review gate will not run; its warning, printed just above, has the details.
_REVIEW_OFF_BY_PREFLIGHT = "The check before the start could not load LSAR (see the warning above)."


def _review_blocked_by(checks: list[Any], plan: Any) -> str | None:
    """Why the review the study asks for will not run, from the pre-start check.

    On the Mac test the check warned "LSAR ... could not be imported ...;
    the review gate will not run", and the card printed right after it
    still said "Review: automated peer review (LSAR) on".
    """
    if not getattr(plan, "review", False):
        return None
    codes = getattr(_module("runner"), "REVIEW_OFF_CODES", frozenset())
    if any(getattr(check, "code", None) in codes for check in checks):
        return _REVIEW_OFF_BY_PREFLIGHT
    return None


def _launch(settings: dict[str, Any], plan: Any, *, watch: bool) -> None:
    """Start the study and follow it (ends the command)."""
    runner = _module("runner")
    run_dir = Path(runner.launch(settings, plan))
    ui.ok(f"The study has started. Its folder: {run_dir}")
    _after_start(run_dir, watch)


def _show_study_checks(checks: list[Any]) -> None:
    """The free check's list: only the name of a check that passed, as
    ``edmars new`` shows it; problems keep their sentence and their fix."""
    for check in checks:
        if check.status == "ok":
            ui.ok(check.name)
        else:
            ui.show_checks([check])


def _preflight_confirm_launch(settings: dict[str, Any], plan: Any, *, yes: bool, watch: bool) -> None:
    """``edmars run``: check the plan, show the confirmation card, start it."""
    study = _module("study")
    with ui.status(
        "Checking that this study can run (the first check on a large dataset "
        "can take a few minutes)"
    ):
        checks = study.preflight(plan, settings)
    _show_study_checks(checks)
    if study.blocking(checks):
        ui.fail("This study cannot start until the problems above are fixed.")
        raise typer.Exit(1)
    blocked = _review_blocked_by(_pipeline_check(settings, plan), plan)

    ui.panel("Ready to start", study.confirmation_card(plan, settings, review_blocked=blocked))
    if not yes:
        # Starting a study spends money on the user's AI account, so it is
        # never started on a default answer: without a terminal, --yes is
        # required.
        if not ui.is_interactive():
            raise ui.NonInteractiveError(
                "Start this study?", hint="Add --yes to start it without being asked."
            )
        answer = ui.select(
            "Start this study?", [("start", "Start the study"), ("cancel", "Cancel")], default="start"
        )
        if answer == "cancel":
            ui.info("No study was started.")
            raise typer.Exit(0)
    _launch(settings, plan, watch=watch)


def _review_available(settings: dict[str, Any]) -> bool:
    """Whether LSAR is set up well enough for a study to ask for a review."""
    from edmars import settings as settings_mod

    home = settings_mod.get(settings, "lsar.home")
    return bool(
        settings_mod.get(settings, "lsar.enabled", False)
        and settings_mod.get(settings, "provider", "deepseek") != "local"
        and home
        and Path(str(home)).is_dir()
    )


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


def _setup_option(flag: str, help_text: str) -> Any:
    return typer.Option(flag, help=help_text, show_default=False)


@app.command("setup")
@_friendly
def setup_cmd(
    section: Annotated[
        Optional[str],
        typer.Argument(help="Only this part of the setup (leave out to see the menu).", show_default=False),
    ] = None,
    provider: Annotated[
        Optional[str], _setup_option("--provider", "With --yes: deepseek, openai, anthropic or local.")
    ] = None,
    key_env: Annotated[
        Optional[str],
        _setup_option("--key-env", "With --yes: NAME of the environment variable that holds the AI key."),
    ] = None,
    deepseek_key_env: Annotated[
        Optional[str],
        _setup_option("--deepseek-key-env", "With --yes: NAME of the variable holding a DeepSeek key for the reviewer."),
    ] = None,
    no_key_check: Annotated[
        bool, typer.Option("--no-key-check", help="With --yes: do not test the keys online.")
    ] = False,
    studies_dir: Annotated[
        Optional[str], _setup_option("--studies-dir", "With --yes: the folder for your studies.")
    ] = None,
    dataset_action: Annotated[
        Optional[str], _setup_option("--dataset-action", "With --yes: download, import or skip.")
    ] = None,
    dataset_path: Annotated[
        Optional[str], _setup_option("--dataset-path", "With --yes and --dataset-action import: the file.")
    ] = None,
    latex_action: Annotated[
        Optional[str], _setup_option("--latex-action", "With --yes: auto, system, tinytex or skip.")
    ] = None,
    r_action: Annotated[Optional[str], _setup_option("--r-action", "With --yes: skip, find or install.")] = None,
    rscript: Annotated[Optional[str], _setup_option("--rscript", "With --yes: the path to Rscript.")] = None,
    lsar_action: Annotated[
        Optional[str], _setup_option("--lsar-action", "With --yes: auto, manual or skip (the automated reviewer).")
    ] = None,
    option: Annotated[
        Optional[list[str]],
        typer.Option(
            "--option",
            "-o",
            help="Answer any other setup question in advance, as KEY=VALUE (repeatable; "
            "see --list-options).",
        ),
    ] = None,
    list_options: Annotated[
        bool, typer.Option("--list-options", help="List every KEY that --option accepts, and stop.")
    ] = False,
    accept_disclosure: AcceptOpt = False,
    plain: PlainOpt = False,
    yes: YesOpt = False,
) -> None:
    """Set up EDM-ARS, or change one part of the setup."""
    non_interactive = _modes(plain, yes)
    wizard = _module("wizard")
    if list_options:
        for key, (env_name, meaning) in wizard.NONINTERACTIVE_OPTIONS.items():
            ui.say(f"{key} (or {env_name}): {meaning}")
        raise typer.Exit(0)
    options = _parse_options(option)
    named = {
        "provider": provider,
        "key_env": key_env,
        "deepseek_key_env": deepseek_key_env,
        "studies_dir": studies_dir,
        "dataset_action": dataset_action,
        "dataset_path": dataset_path,
        "latex_action": latex_action,
        "r_action": r_action,
        "rscript": rscript,
        "lsar_action": lsar_action,
    }
    options.update({k: v for k, v in named.items() if v is not None})
    if no_key_check:
        options["check_keys"] = False
    if accept_disclosure:
        options["accept_disclosure"] = True
    if options and not non_interactive:
        ui.info("Answers given as options are used only with --yes; setup will ask instead.")
    raise _exit(wizard.run_setup(section, non_interactive=non_interactive, options=options or None))


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
    quick: Annotated[
        bool,
        typer.Option("--quick", help="Only check the installation itself (what the installer runs)."),
    ] = False,
    plain: PlainOpt = False,
    yes: YesOpt = False,
) -> None:
    """Check that this computer is ready to run studies."""
    _modes(plain, yes)
    if json_out:
        ui.set_machine_output(True)
    doctor = _module("doctor")
    raise _exit(doctor.main(deep=deep and not quick, json_out=json_out, bundle=bundle, quick=quick))


@app.command("new", epilog=EXIT_CODES_HELP)
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
    _require_ai_key(settings)
    study = _module("study")
    # new_study_interactive runs the feasibility check, the options and the
    # confirmation card itself (R4-R6); it returns a plan only after the
    # user pressed Start.
    plan = study.new_study_interactive(settings)
    if plan is None:
        ui.info("No study was started.")
        raise typer.Exit(0)
    if _review_blocked_by(_pipeline_check(settings, plan), plan):
        # `edmars new` shows its card before this check. The card already
        # says so when LSAR's files or packages are missing; a problem only
        # the import finds is said here, before the study starts.
        ui.warn("The automated peer review will not run for this study (see the warning above); "
                "the study goes ahead without it.")
    _launch(settings, plan, watch=not no_watch)


@app.command("run", epilog=EXIT_CODES_HELP)
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
    paper_format: Annotated[
        Optional[PaperFormat],
        typer.Option(
            "--paper-format",
            help="conference or journal (default: journal for a journal --venue, "
            "otherwise your setup's choice).",
        ),
    ] = None,
    review: Annotated[
        Optional[bool],
        typer.Option(
            "--review/--no-review",
            help="Run the automated peer review (LSAR) after the paper is written "
            "(default: your setup's choice).",
            show_default=False,
        ),
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
    import dataclasses

    if paper_format is not None:
        plan = dataclasses.replace(plan, paper_format=paper_format.value)
    if review is not None:
        if review and not _review_available(settings):
            ui.fail("The automated reviewer is not set up, so --review cannot be used.")
            ui.info("Set it up with `edmars setup reviewer`, or leave out --review.")
            raise typer.Exit(1)
        plan = dataclasses.replace(plan, review=review)
    _preflight_confirm_launch(settings, plan, yes=non_interactive, watch=not no_watch)


@app.command("status", epilog=EXIT_CODES_HELP)
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
        started = _local_time(pick(item, "started", "started_at"))
        rows.append([folder, started, pick(item, "label", "state", "status"), question])
    ui.table(["Study", "Started", "State", "Question"], rows)


def _local_time(value: str) -> str:
    """A UTC ISO time as local "YYYY-MM-DD HH:MM", the clock the live view
    uses. ``--json`` keeps the UTC value."""
    from edmars.runstate import parse_ts

    ts = parse_ts(value) if value else None
    return ts.astimezone().strftime("%Y-%m-%d %H:%M") if ts is not None else value


@app.command("results", epilog=EXIT_CODES_HELP)
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
    raise _exit(results.show(run_dir, open_.value if open_ is not None else None))


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


@app.command("resume", epilog=EXIT_CODES_HELP)
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
    if _outside_studies(settings, run_dir):
        ui.warn(
            f"{run_dir} is not in your studies folder, so it may have come from someone "
            "else. Continuing a study runs AI-written analysis code on this computer and "
            "uses your AI key. Only continue a study you started yourself or got from "
            "someone you trust."
        )
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
def review_cmd(
    run: RunArg = None,
    accept_disclosure: AcceptOpt = False,
    plain: PlainOpt = False,
    yes: YesOpt = False,
) -> None:
    """Run the automated reviewer (LSAR) on a finished study's paper."""
    from edmars import estimates

    non_interactive = _modes(plain, yes)
    settings = _settings()
    lsar = _module("lsar")
    review = getattr(lsar, "review_run", None) or getattr(lsar, "review", None)
    if review is None:
        raise FeatureMissing("lsar.review")
    run_dir = _resolve_run(settings, run, prefer_active=False)
    # A study with no paper is said so first, before the notice, the
    # "usually takes 10-40 minutes" question or anything else.
    no_paper = getattr(lsar, "no_paper_reason", None)
    reason = no_paper(run_dir) if callable(no_paper) else None
    if reason:
        ui.fail(reason)
        raise typer.Exit(1)
    # The review sends the paper to DeepSeek, which the notice describes;
    # like new, run and resume, it needs the current notice accepted.
    _require_ack(settings, accept_disclosure)
    _confirm_spend(
        f"Review the paper in {run_dir.name}? This sends it to DeepSeek, usually "
        f"takes {estimates.MANUAL_REVIEW_TIME} and costs {estimates.MANUAL_REVIEW_COST}.",
        non_interactive,
    )
    outcome = _call_with(review, run_dir=run_dir, settings=settings)
    raise _exit(outcome)


# --- data -------------------------------------------------------------------------------

def _dataset_info(datasets: ModuleType, name: str) -> Any:
    catalog = getattr(datasets, "CATALOG", {})
    if name not in catalog:
        known = ", ".join(sorted(catalog)) or "none"
        ui.fail(f'There is no dataset called "{name}". Known datasets: {known}.')
        raise typer.Exit(1)
    return catalog[name]


def _progress_printer() -> ui.TransferProgress:
    """A progress callback for ``datasets``: a bar in a terminal, lines in
    plain mode, each with MB done, speed and time left (ui.TransferProgress).

    ``datasets`` reports ``(done, total, phase)``: download, then verify
    (the zip) and convert for HSLS:09, or extract and verify for a file
    used as it comes; each phase gets its own bar or lines, so a 2 GB
    conversion does not look like a hang after "downloaded 100%".
    """
    return ui.TransferProgress()


def _installed_and_valid(datasets: Any, name: str, settings: dict[str, Any]) -> bool:
    """Whether ``datasets.install`` will find a usable copy already in place."""
    expected = getattr(datasets, "expected_path", None)
    if expected is None:
        return False
    path = Path(expected(name, settings))
    return path.is_file() and str(getattr(datasets.validate_file(name, path), "status", "")) == "ok"


def _save_dataset_path(settings: dict[str, Any], name: str, path: Path) -> None:
    """Remember where a dataset is (``datasets`` records hashes itself)."""
    from edmars import settings as settings_mod

    if settings_mod.get(settings, f"datasets.{name}.path") != str(path):
        settings_mod.set_(settings, f"datasets.{name}.path", str(path))
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

    if getattr(info, "source", "download") == "manual":
        # Nothing to download (yet): install() explains how to import it,
        # without a terms screen or a "Downloading" line first.
        datasets.install(name, settings)
    if name == "did_els_hsls_panel":
        with ui.status("Building the combined ELS:2002 + HSLS:09 panel from your two datasets"):
            path = Path(datasets.install(name, settings))
    else:
        terms = datasets.terms_text(name)
        if terms and not datasets.terms_accepted(name, settings):
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
            datasets.accept_terms(name, settings)
            from edmars import settings as settings_mod

            settings_mod.save(settings)
        if _installed_and_valid(datasets, name, settings):
            # install() only re-hashes a valid copy (and downloads again
            # only if it changed), so "Downloading" would be wrong here.
            ui.info("Already installed: checking that the file is complete and unchanged. "
                    "A copy that changed is downloaded again.")
        else:
            ui.info("Downloading. Large files take a while; if it stops, run the same command to continue.")
        progress = _progress_printer()
        try:
            path = Path(datasets.install(name, settings, progress=progress))
        finally:
            progress.close()
    check = datasets.validate_file(name, path)
    ui.show_checks([check])
    if check.status != "fail":
        _save_dataset_path(settings, name, path)
    raise typer.Exit(0 if check.status != "fail" else 1)


@data_app.command("import")
@_friendly
def data_import_cmd(
    name: Annotated[str, typer.Argument(help="The dataset's name, as `edmars data list` shows it.")],
    path: Annotated[
        Path,
        typer.Argument(
            help="The data file you downloaded yourself (for HSLS:09 also the NCES .zip, which is converted).",
            exists=True, dir_okay=False, readable=True,
        ),
    ],
    plain: PlainOpt = False,
    yes: YesOpt = False,
) -> None:
    """Use a dataset file you already downloaded."""
    _modes(plain, yes)
    datasets = _module("datasets")
    settings = _settings()
    _dataset_info(datasets, name)
    with ui.status("Checking the file and putting it in place (a 2 GB file takes a minute)"):
        stored = Path(datasets.import_file(name, path, settings))
    check = datasets.validate_file(name, stored)
    ui.show_checks([check])
    if check.status != "fail":
        _save_dataset_path(settings, name, stored)
    raise typer.Exit(0 if check.status != "fail" else 1)


@data_app.command("verify")
@_friendly
def data_verify_cmd(
    name: Annotated[
        Optional[str], typer.Argument(help="Only this dataset (default: all installed ones).", show_default=False)
    ] = None,
    plain: PlainOpt = False,
    yes: YesOpt = False,
) -> None:
    """Re-read your datasets and check they are complete and unchanged."""
    _modes(plain, yes)
    datasets = _module("datasets")
    settings = _settings()
    if name:
        _dataset_info(datasets, name)
    names = [name] if name else list(datasets.CATALOG)
    checks = []
    for item in names:
        current = datasets.status(item, settings)
        if current.status == "fail" or (not name and current.status != "ok"):
            # Not installed: nothing to re-read (status says how to get it).
            checks.append(current)
            continue
        with ui.status(f"Re-reading {item} and checking its fingerprint (SHA-256)"):
            checks.append(datasets.verify(item, settings))
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
    raise _exit(maintenance.update(check_only=check))


@app.command("after-install", hidden=True)
@_friendly
def after_install_cmd(
    state_file: Annotated[
        Optional[Path],
        typer.Option("--state-file", help="Write setup=... and reviewer=... lines here for the installer."),
    ] = None,
    plain: PlainOpt = False,
    yes: YesOpt = False,
) -> None:
    """The installer's last step: keep the reviewer working in the new environment."""
    _modes(plain, yes)
    maintenance = _module("maintenance")
    raise _exit(maintenance.after_install(state_file))


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
    """Remove EDM-ARS's settings, stored keys, automated reviewer and caches from this computer."""
    non_interactive = _modes(plain, yes)
    maintenance = _module("maintenance")
    raise _exit(
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
