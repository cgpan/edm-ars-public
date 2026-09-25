"""`edmars setup`: the guided setup wizard (screens S0 to S11).

Written for education researchers who have never used a terminal: plain
English, one decision per screen, the recommended answer pre-selected,
links printed in full, keys pasted hidden and checked live, and a clear
"Saved in <place>" after every key. Progress is saved after each screen,
so an interrupted setup continues where it stopped.

Screens
    S0 Welcome                      S6 Datasets (HSLS:09 download or import)
    S1 What leaves your computer    S7 PDF typesetting (LaTeX)
       (acknowledgement, required)  S8 R for measurement studies
    S2 Checking your computer       S9 Automated peer review (LSAR)
    S3 Where to keep your studies   S10 Your name and advanced options
    S4 AI service and key           S11 Final check, then "start a study?"
    S5 Literature search key

``run_setup(section)`` runs one section (``edmars setup ai`` and so on; see
:data:`SECTIONS`). With no section it runs the whole flow, resumes an
unfinished one, or, once setup is complete, asks "What would you like to
change?".

Non-interactive mode (``--yes``, CI) never prompts. It takes its answers
from the ``options`` dict and, for keys the dict lacks, from the
environment variables in :data:`NONINTERACTIVE_OPTIONS`. API keys are never
passed as values: an option names the environment variable that holds one.

Return codes: 0 finished, 1 could not finish (a required answer or step
failed), 2 unknown section, 130 the user quit (answers so far are saved).
"""

from __future__ import annotations

import os
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from edmars import doctor as _doctor
from edmars import estimates

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

#: The screens, in order.
SCREENS: tuple[str, ...] = tuple(f"S{i}" for i in range(12))

SCREEN_TITLES: dict[str, str] = {
    "S0": "Welcome",
    "S1": "What leaves your computer",
    "S2": "Checking your computer",
    "S3": "Where to keep your studies",
    "S4": "Choose an AI service",
    "S5": "Literature search",
    "S6": "Datasets",
    "S7": "PDF typesetting",
    "S8": "R for measurement studies",
    "S9": "Automated peer review",
    "S10": "Your name and advanced options",
    "S11": "Final check",
}

#: ``edmars setup <section>`` names and the screen each one runs.
SECTIONS: dict[str, str] = {
    "welcome": "S0",
    "disclosure": "S1",
    "computer": "S2",
    "folders": "S3",
    "ai": "S4",
    "literature": "S5",
    "datasets": "S6",
    "pdf": "S7",
    "r": "S8",
    "reviewer": "S9",
    "advanced": "S10",
    "check": "S11",
}

#: Other words people will type for the same sections.
SECTION_ALIASES: dict[str, str] = {
    "notice": "disclosure", "ack": "disclosure", "acknowledgement": "disclosure",
    "acknowledgment": "disclosure", "privacy": "disclosure", "terms": "disclosure",
    "system": "computer", "studies": "folders", "folder": "folders",
    "provider": "ai", "service": "ai", "key": "ai", "keys": "ai", "llm": "ai",
    "s2": "literature", "semantic-scholar": "literature", "semanticscholar": "literature",
    "citations": "literature", "data": "datasets", "dataset": "datasets",
    "latex": "pdf", "tex": "pdf", "tinytex": "pdf", "pdfs": "pdf",
    "rstats": "r", "psychometrics": "r", "measurement": "r",
    "lsar": "reviewer", "review": "reviewer", "reviews": "reviewer",
    "author": "advanced", "name": "advanced", "budget": "advanced", "models": "advanced",
    "doctor": "check", "final": "check",
}

#: Words that restart the whole flow from the welcome screen.
START_OVER_WORDS: frozenset[str] = frozenset({"all", "full", "start-over", "startover", "restart", "everything"})

#: Non-interactive option -> (environment variable read when the option is
#: absent, meaning). The CLI maps its flags onto these keys.
NONINTERACTIVE_OPTIONS: dict[str, tuple[str, str]] = {
    "accept_disclosure": ("EDMARS_ACCEPT_DISCLOSURE", "true to accept the notice (required on a first setup)"),
    "studies_dir": ("EDMARS_STUDIES_DIR", "folder for studies (default ~/EDM-ARS/studies)"),
    "provider": ("EDMARS_PROVIDER", "deepseek | openai | anthropic | local"),
    "key_env": ("EDMARS_KEY_ENV", "NAME of the environment variable holding the provider key "
                                  "(default: the provider's own, e.g. DEEPSEEK_API_KEY)"),
    "deepseek_key_env": ("EDMARS_DEEPSEEK_KEY_ENV", "NAME of the variable holding a DeepSeek key for the reviewer"),
    "semantic_scholar_key_env": ("EDMARS_S2_KEY_ENV", "NAME of the variable holding a Semantic Scholar key"),
    "check_keys": ("EDMARS_CHECK_KEYS", "false to skip the live key checks (no network)"),
    "allow_key_file": ("EDMARS_ALLOW_KEY_FILE", "true to keep keys in a private file when the credential "
                                                "store does not work"),
    "base_url": ("EDMARS_BASE_URL", "server address for provider=local, e.g. http://localhost:11434/v1"),
    "model": ("EDMARS_MODEL", "model name for provider=local or openai (used for every step)"),
    "dataset": ("EDMARS_DATASET", "dataset for dataset_action (default hsls09_public)"),
    "dataset_action": ("EDMARS_DATASET_ACTION", "download | import | skip (default: keep what is there)"),
    "dataset_path": ("EDMARS_DATASET_PATH", "file to import when dataset_action=import"),
    "latex_action": ("EDMARS_LATEX_ACTION", "auto | system | tinytex | skip (default auto)"),
    "latex_test": ("EDMARS_LATEX_TEST", "true to run the LaTeX test compile"),
    "r_action": ("EDMARS_R_ACTION", "skip | find | install (install = find R, then add packages)"),
    "rscript": ("EDMARS_RSCRIPT", "path to Rscript for r_action find/install"),
    "lsar_action": ("EDMARS_LSAR_ACTION", "auto | manual | skip (default skip)"),
    "author_name": ("EDMARS_AUTHOR_NAME", "your name for the paper's author line"),
    "affiliation": ("EDMARS_AFFILIATION", "your university or organization"),
    "venue": ("EDMARS_VENUE", "EDM | JEDM | JLA | AERA_OPEN"),
    "paper_format": ("EDMARS_PAPER_FORMAT", "conference | journal"),
    "budget_usd": ("EDMARS_BUDGET_USD", "spending warning per study in US$ (empty = none)"),
}

#: Venues offered in S10: id -> label.
VENUES: dict[str, str] = {
    "EDM": "EDM conference (recommended; reviewer scores are benchmarked against accepted EDM papers)",
    "JEDM": "Journal of Educational Data Mining (reviewer gives a score only, no benchmark)",
    "JLA": "Journal of Learning Analytics (reviewer gives a score only, no benchmark)",
    "AERA_OPEN": "AERA Open (reviewer gives a score only, no benchmark)",
}

#: Presets for a model server on the user's own computer.
LOCAL_PRESETS: tuple[tuple[str, str], ...] = (
    ("http://localhost:11434/v1", "Ollama (http://localhost:11434/v1)"),
    ("http://localhost:1234/v1", "LM Studio (http://localhost:1234/v1)"),
    ("http://localhost:8000/v1", "vLLM (http://localhost:8000/v1)"),
)

#: The pipeline steps a local model must cover (the shipped model blocks).
MODEL_STAGES: tuple[str, ...] = (
    "problem_formulator", "data_engineer", "analyst", "critic", "writer",
    "revision_writer", "outline_agent", "verifier",
)

STAGE_LABELS: dict[str, str] = {
    "problem_formulator": "framing the question",
    "data_engineer": "preparing the data",
    "analyst": "running the analysis",
    "critic": "internal methods review",
    "writer": "writing the paper",
    "revision_writer": "revising after review",
    "outline_agent": "outlining the paper",
    "verifier": "final checks",
}

SEMANTIC_SCHOLAR_ENV = "SEMANTIC_SCHOLAR_API_KEY"
SEMANTIC_SCHOLAR_FORM = "https://www.semanticscholar.org/product/api#api-key-form"
DEEPSEEK_ENV = "DEEPSEEK_API_KEY"
TAVILY_ENV = "TAVILY_API_KEY"
LOCAL_PLACEHOLDER_KEY = "local"
HSLS = "hsls09_public"
HSLS_FILENAME = "hsls_17_student_pets_sr_v1_0.csv"
HSLS_DOWNLOAD_BYTES_NEEDED = int(2.5 * 1024 ** 3)
CRAN_PAGES: dict[str, str] = {
    "win32": "https://cloud.r-project.org/bin/windows/base/",
    "darwin": "https://cloud.r-project.org/bin/macosx/",
    "linux": "https://cloud.r-project.org/bin/linux/",
}

NCES_TERMS = (
    "- This is public-use data from the National Center for Education Statistics (NCES).\n"
    "- Do not try to identify any student, family, teacher or school.\n"
    "- Cite NCES as the source of the data in anything you publish.\n"
    "- EDM-ARS is not affiliated with or endorsed by NCES or IES."
)

_BACK = "__back__"
_QUIT = "__quit__"


class _Back(Exception):
    """The user chose "Go back"."""


class _Quit(Exception):
    """The user chose to quit (or pressed Ctrl+C); progress is already saved."""


class _Abort(Exception):
    """Setup cannot continue; ``code`` is the return code."""

    def __init__(self, code: int = 1) -> None:
        super().__init__(code)
        self.code = code


@dataclass
class _KeyResult:
    """What a live key check said (mirrors ``edmars.providers.KeyCheck``)."""

    status: str
    message: str = ""
    balance: str | None = None
    models: list[str] | None = None


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

_ASCII_MAP = {
    "\u2014": "-", "\u2013": "-", "\u2026": "...", "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"', "\u00b7": "-", "\u2022": "-", "\u2713": "[ok]",
    "\u2717": "[x]", "\u2192": "->", "\u25b8": ">", "\u00a0": " ",
}


def _plain() -> bool:
    try:
        from edmars import ui

        return bool(ui.is_plain())
    except Exception:
        return True


def _t(text: str) -> str:
    """Text for the screen: ASCII punctuation in plain mode."""
    if _plain():
        for src, dst in _ASCII_MAP.items():
            text = text.replace(src, dst)
    return text


def _nb(text: str) -> str:
    """No square brackets: dynamic text must never be read as console markup."""
    return str(text).replace("[", "(").replace("]", ")")


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _clean_path(text: str) -> Path:
    """A path as people paste it: quotes, ``& '...'`` from PowerShell drag-and-drop, ``~``."""
    raw = text.strip()
    if raw.startswith("& "):
        raw = raw[2:].strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
        raw = raw[1:-1]
    return Path(os.path.expandvars(raw)).expanduser()


def _clean_key(text: str) -> str:
    key = (text or "").strip()
    if len(key) >= 2 and key[0] == key[-1] and key[0] in "\"'":
        key = key[1:-1].strip()
    if key.lower().startswith("bearer "):
        key = key[7:].strip()
    return key


def _normalize_rscript(path: Path) -> Path | None:
    """Accept Rscript itself, R's ``bin`` folder, the R install folder, or ``R.exe``."""
    exe = "Rscript.exe" if os.name == "nt" else "Rscript"
    if path.is_file():
        if path.name.lower() in ("r.exe", "rterm.exe", "rgui.exe", "r"):
            sibling = path.parent / exe
            return sibling if sibling.is_file() else None
        return path
    if path.is_dir():
        for candidate in (path / exe, path / "bin" / exe, path / "bin" / "x64" / exe,
                          path / "Resources" / "bin" / exe):
            if candidate.is_file():
                return candidate
    return None


def _can_open_browser() -> bool:
    if _plain():
        return False
    if sys.platform in ("win32", "darwin"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _open_url(url: str) -> bool:
    import webbrowser

    try:
        return bool(webbrowser.open(url, new=2))
    except Exception:
        return False


def _read_dotenv(path: Path) -> dict[str, str]:
    """Parse ``NAME=value`` lines of an old repo ``.env`` (values are never shown)."""
    values: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return values
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip()
        if name.lower().startswith("export "):
            name = name[7:].strip()
        value = value.strip().strip("\"'")
        if name and value:
            values[name] = value
    return values


def _looks_like_placeholder(value: str) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker in ("your-key", "your_key", "your key", "changeme", "placeholder", "xxx"))


# ---------------------------------------------------------------------------
# The wizard
# ---------------------------------------------------------------------------

class _Wizard:
    """One setup session: holds the settings dict and the user's answers."""

    def __init__(self, settings: dict[str, Any], *, non_interactive: bool, options: Mapping[str, Any]) -> None:
        self.s = settings
        self.ni = non_interactive
        self.opts = dict(options)
        self.errors: list[str] = []
        self.allow_back = False
        self.in_flow = False
        self.start_study = False
        self.last_models: list[str] | None = None

    # -- modules (imported lazily so a missing optional piece never breaks import)
    @property
    def ui(self) -> Any:
        from edmars import ui

        return ui

    # -- options ----------------------------------------------------------------
    def opt(self, key: str, default: Any = None) -> Any:
        if key in self.opts and self.opts[key] not in (None, ""):
            return self.opts[key]
        env_name = NONINTERACTIVE_OPTIONS.get(key, ("", ""))[0]
        if env_name and os.environ.get(env_name, "") != "":
            return os.environ[env_name]
        return default

    # -- settings ---------------------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        return _doctor.cfg(self.s, dotted, default)

    def set(self, dotted: str, value: Any) -> None:
        from edmars import settings as st

        st.set_(self.s, dotted, value)

    def save(self) -> None:
        from edmars import settings as st

        st.save(self.s)

    # -- output -----------------------------------------------------------------
    def say(self, text: str) -> None:
        try:
            self.ui.console.print(_t(text), markup=False, highlight=False)
        except Exception:
            print(_t(text))

    def ok(self, text: str) -> None:
        self.ui.ok(_t(_nb(text)))

    def info(self, text: str) -> None:
        self.ui.info(_t(_nb(text)))

    def warn(self, text: str) -> None:
        self.ui.warn(_t(_nb(text)))

    def fail(self, text: str) -> None:
        self.ui.fail(_t(_nb(text)))

    def panel(self, title: str, body: str) -> None:
        self.ui.panel(_t(_nb(title)), _t(_nb(body)))

    def header(self, sid: str, body: str, *, title: str | None = None) -> None:
        n = int(sid[1:])
        heading = title or SCREEN_TITLES[sid]
        prefix = f"Step {n} of 11 \u00b7 " if n else ""
        self.panel(prefix + heading, body)

    def error(self, text: str) -> None:
        """A non-interactive step failed: report it and remember for the exit code."""
        self.errors.append(text)
        self.fail(text)

    # -- questions --------------------------------------------------------------
    def choose(self, message: str, choices: Sequence[tuple[str, str]], default: str | None = None,
               *, back: bool | None = None, quit_: bool = True) -> str:
        options = [(value, _t(_nb(label))) for value, label in choices]
        if self.allow_back if back is None else back:
            options.append((_BACK, "Go back"))
        if quit_:
            options.append((_QUIT, _t("Quit setup for now (your answers so far are saved)")))
        values = [value for value, _ in options]
        if default not in values:
            default = values[0] if values else None
        answer = self.ui.select(_t(message), options, default=default)
        if answer is None or answer == _QUIT:
            raise _Quit()
        if answer == _BACK:
            raise _Back()
        return str(answer)

    def ask_text(self, message: str, default: str | None = None) -> str:
        answer = self.ui.text(_t(message), default=default)
        if answer is None:
            raise _Quit()
        return str(answer).strip()

    def ask_secret(self, message: str) -> str:
        answer = self.ui.secret(_t(message))
        if answer is None:
            raise _Quit()
        return _clean_key(str(answer))

    def yes(self, message: str, default: bool = True) -> bool:
        answer = self.ui.confirm(_t(message), default=default)
        if answer is None:
            raise _Quit()
        return bool(answer)

    def show_checks(self, checks: Sequence[Any], title: str | None = None) -> None:
        _doctor.render_checks(checks, title=_t(title) if title else None)

    # -- flow -------------------------------------------------------------------
    def run_screens(self, screens: Sequence[str], *, flow: bool) -> None:
        """Run screens in order. In the flow, Back moves to the previous screen
        and progress is recorded; in a single section, Back leaves the section."""
        i = 0
        self.in_flow = flow
        while i < len(screens):
            sid = screens[i]
            self.allow_back = (i > 0) if flow else True
            try:
                getattr(self, f"screen_{sid.lower()}")()
            except _Back:
                if not flow:
                    return
                i = max(0, i - 1)
                continue
            if flow:
                self.set("setup_progress.last_completed_screen", sid)
                if sid == "S11":
                    self.set("setup_progress.completed_at", _now())
            self.save()
            i += 1

    def run(self, target: str | None) -> int:
        if self.ni:
            return self.run_noninteractive(target)
        if target is not None and target != "__all__":
            self.run_screens([SECTIONS[target]], flow=False)
            return 0
        if target == "__all__":
            self.run_screens(SCREENS, flow=True)
            return 0
        from edmars import disclosure

        last = str(self.get("setup_progress.last_completed_screen", "") or "")
        if last == "S11":
            if not disclosure.is_acknowledged(self.s):
                self.info("The notice about what leaves your computer has changed since you accepted it. "
                          "Please read it again.")
                self.run_screens(["S1"], flow=False)
                if not disclosure.is_acknowledged(self.s):
                    return 130
            return self.change_menu()
        start = 0
        if last in SCREENS:
            done = SCREENS.index(last)
            nxt = SCREENS[done + 1]
            self.allow_back = False
            where = f"step {done}" if done else "the welcome screen"
            answer = self.choose(
                f"You stopped setup after {where} last time. Continue where you left off?",
                [("resume", f"Continue with step {int(nxt[1:])}: {SCREEN_TITLES[nxt]}"),
                 ("restart", "Start from the beginning (your earlier answers are kept as defaults)")],
                default="resume", back=False)
            start = done + 1 if answer == "resume" else 0
        self.run_screens(SCREENS[start:], flow=True)
        return 0

    def change_menu(self) -> int:
        menu: list[tuple[str, str]] = [
            ("ai", "Change the AI service or its key"),
            ("literature", "Add or change the Semantic Scholar key"),
            ("datasets", "Datasets"),
            ("pdf", "PDF typesetting (LaTeX)"),
            ("r", "R for measurement studies"),
            ("reviewer", "Automated peer review (LSAR)"),
            ("folders", "Where studies are kept"),
            ("advanced", "Your name on papers, spending warning, paper format, venue"),
            ("disclosure", "Read the notice about what leaves your computer again"),
            ("check", "Check everything again"),
            ("start-over", "Start over (keeps your studies, datasets and saved keys)"),
            ("done", "Nothing, I'm done"),
        ]
        while True:
            self.allow_back = False
            answer = self.choose("EDM-ARS is already set up. What would you like to change?", menu,
                                 default="done", back=False, quit_=False)
            if answer == "done":
                return 0
            if answer == "start-over":
                self.run_screens(SCREENS, flow=True)
                return 0
            self.run_screens([SECTIONS[answer]], flow=False)

    # -- non-interactive ----------------------------------------------------------
    def run_noninteractive(self, target: str | None) -> int:
        unknown = sorted(k for k in self.opts if k not in NONINTERACTIVE_OPTIONS)
        if unknown:
            self.warn("Ignoring unknown setup option(s): " + ", ".join(unknown))
        if target is None or target == "__all__":
            screens: Sequence[str] = SCREENS
            flow = True
        else:
            screens = [SECTIONS[target]]
            flow = False
        self.in_flow = flow
        for sid in screens:
            getattr(self, f"screen_{sid.lower()}")()
            if flow:
                self.set("setup_progress.last_completed_screen", sid)
                if sid == "S11":
                    self.set("setup_progress.completed_at", _now())
            self.save()
        if self.errors:
            self.fail(f"Setup finished with {len(self.errors)} problem(s); see the messages above.")
            return 1
        return 0

    # =========================================================================
    # S0 Welcome
    # =========================================================================
    def screen_s0(self) -> None:
        body = (
            "EDM-ARS turns a research question into a complete draft research paper, using public "
            "education datasets and an AI service you choose.\n\n"
            "Setup takes about 10-20 minutes, plus download time. You will:\n"
            "  1. Choose where to keep your studies\n"
            "  2. Connect an AI service (you'll paste a key)\n"
            "  3. Optionally connect a literature-search service\n"
            "  4. Download the HSLS:09 dataset (about 300 MB)\n"
            "  5. Choose optional add-ons: PDF typesetting, measurement-study support, automated reviewer\n\n"
            "You can stop at any time and continue later with `edmars setup`."
        )
        self.header("S0", body, title="Welcome to EDM-ARS")
        if self.ni:
            return
        self.choose("Ready?", [("continue", "Continue")], default="continue", back=False)

    # =========================================================================
    # S1 Disclosure acknowledgement (required)
    # =========================================================================
    def screen_s1(self) -> None:
        from edmars import disclosure

        version = str(getattr(disclosure, "ACK_VERSION", "?"))
        if disclosure.is_acknowledged(self.s) and (self.ni or self.in_flow):
            # `edmars setup disclosure` (a single section) always shows it again.
            when = str(self.get("acknowledged.at", "") or "")
            self.ok(f"You already accepted the notice about what leaves your computer (version {version}"
                    + (f", {when[:10]}" if when else "") + ").")
            return
        text = disclosure.ack_text()
        self.panel(f"Step 1 of 11 \u00b7 Before you start: what leaves your computer (version {version})", text)
        if self.ni:
            if _truthy(self.opt("accept_disclosure")):
                disclosure.record_ack(self.s)
                self.save()
                self.ok(f"Notice accepted (version {version}) because --accept-disclosure was given.")
                return
            self.error("Setup needs you to accept the notice above. Read it in full with `edmars disclaimer` "
                       "and `edmars privacy`, then run setup again with --accept-disclosure.")
            raise _Abort(1)
        while True:
            answer = self.choose(
                "Do you understand and accept this?",
                [("accept", "I understand and accept \u2014 continue"),
                 ("disclaimer", "Read the full disclaimer first"),
                 ("privacy", "Read the full privacy notice first")],
                default="accept")
            if answer in ("disclaimer", "privacy"):
                # Long Markdown texts are printed literally, not as a panel.
                self.say(disclosure.disclaimer_text() if answer == "disclaimer" else disclosure.privacy_text())
                continue
            disclosure.record_ack(self.s)
            self.save()
            self.ok(f"Thank you. Your acceptance is saved (version {version}).")
            return

    # =========================================================================
    # S2 Checking your computer
    # =========================================================================
    def screen_s2(self) -> None:
        self.header("S2", "Checking your computer. Nothing is installed or changed in this step.")
        checks = _doctor.system_checks(self.s)
        self.show_checks(checks)
        if self.ni:
            return
        self.choose("Continue?", [("continue", "Continue")], default="continue")

    # =========================================================================
    # S3 Studies folder
    # =========================================================================
    def screen_s3(self) -> None:
        from edmars import paths

        default = Path(paths.default_studies_dir()).expanduser()
        current_raw = self.get("studies_dir", None)
        current = Path(os.path.expandvars(str(current_raw))).expanduser() if current_raw else default
        if self.ni:
            chosen = _clean_path(str(self.opt("studies_dir"))) if self.opt("studies_dir") else current
            if not chosen.is_absolute():
                chosen = chosen.resolve()
            problem = self._prepare_dir(chosen)
            if problem:
                self.error(problem)
                return
            provider = paths.sync_provider(chosen)
            if provider:
                self.warn(f"{chosen} is synced to {provider}. Syncing can slow studies down a lot and uploads "
                          "your working files to the cloud.")
            self.set("studies_dir", str(chosen))
            self.ok(f"Studies will be kept in {chosen}")
            return

        self.header("S3", "Where should EDM-ARS keep your studies? Each study gets its own folder with the "
                          "paper, figures and results.")
        while True:
            choices: list[tuple[str, str]] = [("default", f"{default} (recommended)")]
            if current != default:
                choices.insert(0, ("current", f"{current} (your current choice)"))
            choices.append(("other", "Choose another folder\u2026"))
            answer = self.choose("Where should studies go?", choices,
                                 default="current" if current != default else "default")
            if answer == "default":
                chosen = default
            elif answer == "current":
                chosen = current
            else:
                typed = self.ask_text("Type or paste the full path of the folder (it is created if needed)",
                                      default=str(current))
                if not typed:
                    continue
                chosen = _clean_path(typed)
                if not chosen.is_absolute():
                    self.warn("Please give the full path, for example one that starts with a drive letter or ~.")
                    continue
            problem = self._prepare_dir(chosen)
            if problem:
                self.fail(problem)
                continue
            provider = paths.sync_provider(chosen)
            if provider:
                self.warn(f"This folder is synced to {provider}. Syncing can slow studies down a lot and "
                          "uploads your working files to the cloud.")
                alt: list[tuple[str, str]] = []
                if chosen != default and not paths.sync_provider(default):
                    alt.append(("default", f"Use {default} instead"))
                alt += [("keep", "Keep my choice"), ("other", "Choose another folder")]
                what = self.choose("What would you like to do?", alt, default=alt[0][0], back=False)
                if what == "other":
                    continue
                if what == "default":
                    chosen = default
                    problem = self._prepare_dir(chosen)
                    if problem:
                        self.fail(problem)
                        continue
            self.set("studies_dir", str(chosen))
            self.save()
            self.ok(f"Studies will be kept in {chosen}")
            return

    @staticmethod
    def _prepare_dir(folder: Path) -> str | None:
        """Create the folder and prove it is writable; a message when not."""
        try:
            folder.mkdir(parents=True, exist_ok=True)
            probe = folder / ".edmars-write-test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError as exc:
            return f"EDM-ARS can't use {folder}: {_doctor.redact(str(exc))}"
        return None

    # =========================================================================
    # S4 AI service
    # =========================================================================
    def _provider_ids(self) -> list[str]:
        order = ["deepseek", "openai", "anthropic", "local"]
        try:
            from edmars import providers

            known = list(providers.PROVIDERS)
        except Exception:
            known = order
        return [p for p in order if p in known] + [p for p in known if p not in order]

    def screen_s4(self) -> None:
        current = str(self.get("provider", "deepseek") or "deepseek")
        if self.ni:
            self._s4_noninteractive(current)
            return
        body = (
            "EDM-ARS needs an AI service to write code and text. You pay the service directly.\n\n"
            "Tip: a ChatGPT Plus or Claude Pro subscription is not an API key. You need a separate "
            "developer account with prepaid credit.\n\n"
            "Each service handles your text under its own terms. DeepSeek states that it processes data "
            "in the People's Republic of China; OpenAI and Anthropic are based in the United States. "
            "Some institutions restrict particular services, so check yours."
        )
        self.header("S4", body)
        descriptions = {
            "deepseek": "DeepSeek (recommended): lowest cost; EDM-ARS's quality checks were tuned with it",
            "openai": "OpenAI (ChatGPT API): works; costs more; paper quality is less tested",
            "anthropic": "Anthropic (Claude API): works; costs more; paper quality is less tested",
            "local": "A model on my own computer or server (advanced, experimental)",
        }
        while True:
            choices = [(p, descriptions.get(p) or str(_doctor.provider_meta(p).get("menu_label") or p))
                       for p in self._provider_ids()]
            provider_id = self.choose("Which AI service do you want to use?", choices, default=current)
            try:
                if provider_id == "local":
                    ready = self._local_flow()
                else:
                    ready = self._key_flow(provider_id)
            except _Back:
                continue
            self._set_provider(provider_id)
            label = _doctor.provider_meta(provider_id)["label"]
            if ready and self._needs_model_choice(provider_id):
                ready = self._choose_one_model(provider_id)
            if ready:
                self.ok(f"EDM-ARS will use {label}.")
            else:
                self.warn(f"EDM-ARS will use {label}, but no working key is saved yet. Studies can't start "
                          "until one is. Add it any time with `edmars setup ai`.")
            if provider_id in ("openai", "anthropic"):
                self.info("This service works with EDM-ARS but is less tested. The cost display shows token "
                          "counts only, because EDM-ARS has no price list for its models.")
            return

    def _needs_model_choice(self, provider_id: str) -> bool:
        """True when neither the shipped config nor the settings name a
        model for this service (OpenAI: config.yaml ships no model list,
        and the pipeline refuses to guess one -- defect E1)."""
        if provider_id == "local" or self.get("models", None):
            return False
        try:
            from edmars import providers

            return not providers.default_models(provider_id)
        except Exception:
            return False

    def _choose_one_model(self, provider_id: str) -> bool:
        """Ask for the model every step will use; False when none was chosen."""
        label = _doctor.provider_meta(provider_id)["label"]
        self.info(f"EDM-ARS has no recommended {label} models, so choose the one every step will use. Pick "
                  "the strongest model your account offers: the steps that write analysis code depend on it. "
                  "You can choose a model per step later in `edmars setup advanced`.")
        available = sorted(self.last_models or [])
        if available:
            shown = [(m, m) for m in available[:40]] + [("__type__", "Type a model name")]
            pick = self.choose(f"Which {label} model should EDM-ARS use?", shown, default=shown[0][0], back=False)
            model = pick if pick != "__type__" else self.ask_text("Model name")
        else:
            model = self.ask_text(f"Model name (as {label} lists it)")
        if not model:
            self.warn("No model chosen; studies can't start until one is. Run `edmars setup ai` again.")
            return False
        self.set("models", {stage: model for stage in self._stage_keys()})
        self.save()
        self.ok(f"Every step will use {model}.")
        return True

    def _set_provider(self, provider_id: str) -> None:
        previous = self.get("provider", None)
        # A local server's per-step models were just written by _local_flow.
        if provider_id != "local" and previous and previous != provider_id and self.get("models", None):
            self.set("models", {})
            if not self._needs_model_choice(provider_id):
                self.info("Your model choices were reset to the recommended ones for the new service.")
        self.set("provider", provider_id)
        if provider_id != "local":
            self.set("provider_base_url", None)
        self.save()

    def _s4_noninteractive(self, current: str) -> None:
        provider_id = str(self.opt("provider", current) or "deepseek").lower()
        if provider_id not in self._provider_ids():
            self.error(f"Unknown AI service '{provider_id}'. Choose one of: {', '.join(self._provider_ids())}.")
            return
        if provider_id == "local":
            self._local_noninteractive()
            return
        if self._key_noninteractive(provider_id, self.opt("key_env")):
            self._set_provider(provider_id)
            if self._needs_model_choice(provider_id):
                model = str(self.opt("model", "") or "")
                if not model:
                    self.error(f"provider={provider_id} needs a model name, because EDM-ARS has no recommended "
                               f"models for it: add --option model=<model id> (or set EDMARS_MODEL).")
                    return
                self.set("models", {stage: model for stage in self._stage_keys()})
                self.ok(f"Every step will use {model}.")
            self.ok(f"EDM-ARS will use {_doctor.provider_meta(provider_id)['label']}.")

    # -- key handling -------------------------------------------------------------
    def _instructions(self, provider_id: str) -> str:
        meta = _doctor.provider_meta(provider_id)
        page = meta["key_page"] or ""
        if provider_id == "deepseek":
            return (f"1. Open {page}\n"
                    "2. Sign in or create an account, then add credit (US$2 is plenty to start).\n"
                    "3. Click \"Create new API key\", copy it and paste it below. The key is shown only once.")
        if provider_id == "openai":
            return (f"1. Open {page}\n"
                    "2. Sign in and add prepaid credit under Billing. A ChatGPT Plus subscription is not "
                    "an API key and includes no API credit.\n"
                    "3. Click \"Create new secret key\", copy it and paste it below.")
        if provider_id == "anthropic":
            return (f"1. Open {page}\n"
                    "2. Sign in and add credit under Billing. A Claude Pro subscription is not an API key.\n"
                    "3. Click \"Create Key\", copy it and paste it below.")
        return f"Get a key at {page}" if page else "Paste your key below."

    def _check_key(self, provider_id: str, key: str, base_url: str | None = None) -> _KeyResult:
        from edmars import providers

        if provider_id == "local":
            self.info(f"Checking the server at {base_url}\u2026")
        else:
            self.info(f"Checking the key with {_doctor.provider_meta(provider_id)['label']}\u2026")
        try:
            result = providers.check_key(provider_id, key, base_url=base_url)
        except Exception as exc:  # noqa: BLE001 - a crash is reported, never shown raw
            return _KeyResult("UNKNOWN", _doctor.redact(str(exc), [key]))
        checked = _KeyResult(
            status=str(getattr(result, "status", "UNKNOWN") or "UNKNOWN").upper(),
            message=_doctor.redact(str(getattr(result, "message", "") or ""), [key]),
            balance=getattr(result, "balance", None),
            models=list(getattr(result, "models", None) or []) or None,
        )
        self.last_models = checked.models
        return checked

    def _explain_result(self, provider_id: str, result: _KeyResult) -> None:
        meta = _doctor.provider_meta(provider_id)
        label = meta["label"]
        if result.status == "OK":
            self.ok("Key works." + (f" Balance: {result.balance}" if result.balance else ""))
        elif result.status == "REJECTED":
            self.fail(f"{label} didn't accept this key. Make sure you copied the whole key, with no spaces.")
        elif result.status == "NO_CREDIT":
            where = f" at {meta['topup']}" if meta["topup"] else " in your account's billing page"
            self.warn(f"The key works, but your {label} account has no credit. Add credit{where}, "
                      "then choose \"Check again\".")
        elif result.status == "NETWORK":
            self.warn(f"Couldn't reach {label}. Check your internet connection. University networks "
                      "sometimes block AI services.")
        else:
            detail = f" ({result.message})" if result.message else ""
            self.warn(f"{label} answered in an unexpected way{detail}.")

    def _store_key(self, env_var: str, key: str) -> bool:
        from edmars import secrets

        try:
            store = secrets.set_secret(env_var, key)
        except Exception as exc:  # noqa: BLE001
            store_error = getattr(secrets, "SecretStoreError", None)
            if not (isinstance(store_error, type) and isinstance(exc, store_error)):
                self.fail(f"Couldn't save the key: {_doctor.redact(str(exc), [key])}")
                return False
            # No working credential store (headless Linux, a locked keychain):
            # a private file is the fallback, and only with consent.
            if not self._consent_to_key_file(exc):
                return False
            try:
                store = secrets.set_secret(env_var, key, allow_file=True)
            except Exception as again:  # noqa: BLE001
                self.fail(f"Couldn't save the key: {_doctor.redact(str(again), [key])}")
                return False
        where = _doctor.store_label(store, env_var)
        if store == "keyring":
            self.ok(f"Saved in {where}, not in a file.")
        else:
            self.ok(f"Saved in {where}.")
        env_value = os.environ.get(env_var)
        if env_value and env_value != key:
            self.warn(f"The environment variable {env_var} is also set on this computer and takes priority "
                      "over the saved key. Remove it, or update it, if it holds an old key.")
        return True

    def _missing_models_note(self, provider_id: str, key: str) -> None:
        from edmars import providers

        if provider_id == self.get("provider", None):
            wanted = _doctor.configured_models(self.s, provider_id)
        else:
            # A key for another service (a new choice in S4, or the reviewer's
            # DeepSeek key): the user's per-step overrides belong to the old one.
            try:
                wanted = sorted({str(m) for m in (providers.default_models(provider_id) or {}).values() if m})
            except Exception:
                wanted = []
        if not wanted:
            return
        try:
            missing = providers.missing_models(provider_id, key, wanted, base_url=None)
        except Exception:
            return
        if missing:
            self.warn("Your account can't use these models: " + ", ".join(missing)
                      + ". Studies will stop when they reach them. Choose others in `edmars setup advanced`.")

    def _key_flow(self, provider_id: str, *, purpose: str | None = None) -> bool:
        """Get a working key for ``provider_id`` into secure storage.

        Returns True when a key is available afterwards (checked, or kept
        despite a temporary problem). Raises :class:`_Back` when the user
        goes back from the first question.
        """
        from edmars import paths, secrets

        meta = _doctor.provider_meta(provider_id)
        label = meta["label"]
        env_var = str(meta["env_var"])
        title = f"{label} key" + (f" for {purpose}" if purpose else "")
        self.panel(title, self._instructions(provider_id))
        dotenv_value = _read_dotenv(paths.app_root() / ".env").get(env_var)
        if dotenv_value and _looks_like_placeholder(dotenv_value):
            dotenv_value = None
        candidate = ""
        origin = ""  # stored | env | dotenv | pasted
        recheck = False
        force_paste = False
        while True:
            if not recheck:
                source = secrets.secret_source(env_var)
                choices: list[tuple[str, str]] = []
                if source:
                    choices.append(("keep", f"Use the key already saved in {_doctor.store_label(source, env_var)}"))
                if dotenv_value and source is None:
                    choices.append(("dotenv", "Use the key from the old .env file in the EDM-ARS folder"))
                choices.append(("paste", "Paste a new key" if source else "Paste my key"))
                if _can_open_browser() and meta["key_page"]:
                    choices.append(("open", "Open the key page in my browser"))
                choices.append(("skip", "Skip for now" + (" (the reviewer stays off)" if purpose else
                                                          " (add it later with `edmars setup ai`)")))
                if force_paste:
                    answer, force_paste = "paste", False
                else:
                    answer = self.choose(f"Your {label} key", choices, default=choices[0][0], back=True)
                if answer == "skip":
                    return False
                if answer == "open":
                    if not _open_url(str(meta["key_page"])):
                        self.info(f"Open this address yourself: {meta['key_page']}")
                    continue
                if answer == "keep":
                    candidate = secrets.get_secret(env_var) or ""
                    origin = "env" if source == "env" else "stored"
                elif answer == "dotenv":
                    candidate, origin = dotenv_value or "", "dotenv"
                else:
                    candidate = self.ask_secret(f"Paste your {label} key (it stays hidden; press Enter "
                                                "on an empty line to go back)")
                    origin = "pasted"
                    if not candidate:
                        continue
                    if any(ch.isspace() for ch in candidate):
                        self.fail("That doesn't look like a key: it contains spaces. Copy the key again.")
                        continue
            recheck = False
            result = self._check_key(provider_id, candidate)
            self._explain_result(provider_id, result)
            if result.status == "OK":
                self._missing_models_note(provider_id, candidate)
                return self._keep_key(env_var, candidate, origin)
            follow: list[tuple[str, str]] = []
            if result.status in ("NO_CREDIT", "NETWORK", "UNKNOWN"):
                follow.append(("recheck", "Check again"))
                follow.append(("save", "Keep the key anyway and fix this later"))
            follow.append(("paste", "Use a different key"))
            page = meta["topup"] if result.status == "NO_CREDIT" and meta["topup"] else meta["key_page"]
            if _can_open_browser() and page:
                follow.append(("open", "Open the " + ("top-up" if page == meta["topup"] else "key")
                               + " page in my browser"))
            follow.append(("skip", "Skip for now"))
            nxt = self.choose("What would you like to do?", follow, default=follow[0][0], back=False)
            if nxt == "recheck":
                recheck = True
            elif nxt == "save":
                return self._keep_key(env_var, candidate, origin)
            elif nxt == "paste":
                force_paste = True
            elif nxt == "open":
                if not _open_url(str(page)):
                    self.info(f"Open this address yourself: {page}")
                if result.status != "REJECTED":
                    self.choose("When you're done there:", [("recheck", "Check the key again")],
                                default="recheck", back=False)
                    recheck = True
            elif nxt == "skip":
                return False

    def _consent_to_key_file(self, problem: BaseException) -> bool:
        """Ask before keys go into the fallback file instead of the credential store."""
        from edmars import secrets

        where = getattr(secrets, "secrets_file", None)
        path = str(where()) if callable(where) else "a file in your EDM-ARS settings folder"
        text = (f"This computer's credential store did not keep the key ({_doctor.redact(str(problem))}). "
                f"EDM-ARS can keep it in {path} instead: a plain-text file that only your user account can "
                "read. Anyone who can sign in as you, or a program you run, could read it.")
        if self.ni:
            if _truthy(self.opt("allow_key_file", False)):
                self.warn(text + " Using it because allow_key_file was given.")
                return True
            self.error(text + " To allow that, run setup again with --option allow_key_file=yes, or set the "
                       "key as an environment variable instead.")
            return False
        self.warn(text)
        if self.yes("Keep the key in that file?", default=False):
            return True
        self.info("The key was not saved. You can set it as an environment variable instead "
                  "(ask your IT support how), or try again later.")
        return False

    def _keep_key(self, env_var: str, key: str, origin: str) -> bool:
        """Make sure a key the user wants to keep is where EDM-ARS will find it."""
        if origin == "stored":
            return True
        if origin == "env":
            if self.yes(f"Also save this key in {_doctor.store_label('keyring')} so EDM-ARS finds it in every "
                        "new window?", default=True):
                self._store_key(env_var, key)
            return True
        stored = self._store_key(env_var, key)
        if stored and origin == "dotenv":
            self.info("You can delete the old .env file now; EDM-ARS no longer needs it.")
        return stored

    def _key_noninteractive(self, provider_id: str, key_env: str | None, *, required: bool = True) -> bool:
        """Non-interactive: read the key from ``key_env`` (default: the
        provider's own variable), check it, and store it when it came from a
        differently named variable."""
        from edmars import secrets

        meta = _doctor.provider_meta(provider_id)
        label = meta["label"]
        env_var = str(meta["env_var"])
        source_var = str(key_env or env_var)
        key = os.environ.get(source_var, "").strip() or (secrets.get_secret(env_var) if source_var == env_var else None)
        if not key:
            message = (f"No {label} key found. Put it in the environment variable {env_var}"
                       + (" (or name another variable with --key-env)" if required else "")
                       + ", then run setup again.")
            if required:
                self.error(message)
            else:
                self.warn(message)
            return False
        if _truthy(self.opt("check_keys", True)):
            result = self._check_key(provider_id, key)
            self._explain_result(provider_id, result)
            if result.status == "REJECTED":
                self.error(f"The {label} key in {source_var} was rejected; it was not saved.")
                return False
        else:
            self.info("Skipping the live key check (check_keys is off).")
        if source_var != env_var:
            return self._store_key(env_var, key)
        self.ok(f"Using the {label} key from {_doctor.store_label(secrets.secret_source(env_var) or 'env', env_var)}.")
        return True

    # -- local model server ---------------------------------------------------------
    def _stage_keys(self) -> list[str]:
        stages: list[str] = []
        try:
            from edmars import providers

            for pid in ("local", "openai", "deepseek"):
                for name in providers.default_models(pid) or {}:
                    if name not in stages:
                        stages.append(name)
        except Exception:
            pass
        for name in MODEL_STAGES:
            if name not in stages:
                stages.append(name)
        return stages

    def _local_flow(self) -> bool:
        """S4d: an OpenAI-compatible server on the user's machine or network."""
        self.panel("A model on your own computer or server (experimental)",
                   "EDM-ARS sends prompts of up to about 68,000 tokens. Your model needs a context window of at "
                   "least 128k tokens, or it will silently cut text off. Paper quality with local models has not "
                   "been tested. The automated reviewer still needs a DeepSeek key.")
        current = str(self.get("provider_base_url", "") or "")
        presets = list(LOCAL_PRESETS)
        default = current if any(url == current for url, _ in presets) else ("other" if current else presets[0][0])
        answer = self.choose("Which server do you use?",
                             presets + [("other", "Another OpenAI-compatible server (I'll type its address)")],
                             default=default, back=True)
        if answer == "other":
            while True:
                typed = self.ask_text("Server address (it usually ends in /v1)", default=current or "http://")
                if re.match(r"^https?://\S+$", typed):
                    base_url = typed.rstrip("/")
                    break
                self.warn("The address must start with http:// or https://")
        else:
            base_url = answer
        needs_key = self.choose("Does your server need an API key?",
                                [("no", "No (Ollama and LM Studio don't)"), ("yes", "Yes, I'll paste it")],
                                default="no", back=False)
        key = LOCAL_PLACEHOLDER_KEY
        if needs_key == "yes":
            key = self.ask_secret("Paste the server's key (it stays hidden)") or LOCAL_PLACEHOLDER_KEY
        while True:
            result = self._check_key("local", key, base_url=base_url)
            if result.status == "OK":
                break
            if result.status == "NETWORK":
                self.warn(f"Couldn't reach {base_url}. Is the server running?")
            elif result.status == "REJECTED":
                self.fail("The server didn't accept the key.")
            else:
                self.warn("The server answered in an unexpected way" + (f" ({result.message})." if result.message else "."))
            nxt = self.choose("What would you like to do?",
                              [("retry", "Check again"), ("save", "Save these settings anyway"),
                               ("back", "Choose another service or server")], default="retry", back=False)
            if nxt == "back":
                raise _Back()
            if nxt == "save":
                break
        models = result.models or []
        current_models = self.get("models", {}) or {}
        current_model = next(iter(current_models.values()), None) if isinstance(current_models, dict) else None
        if models:
            shown = [(m, m) for m in models[:40]] + [("__type__", "Type a model name")]
            pick = self.choose("Which model should EDM-ARS use?", shown,
                               default=current_model if current_model in models else shown[0][0], back=False)
            model = pick if pick != "__type__" else self.ask_text("Model name", default=current_model or "")
        else:
            model = self.ask_text("Model name (as your server lists it)", default=current_model or "")
        if not model:
            self.set("models", {})
            self.warn("No model chosen; studies can't start until one is. Run `edmars setup ai` again.")
            return False
        self.set("provider_base_url", base_url)
        self.set("models", {stage: model for stage in self._stage_keys()})
        self._local_key(key, needs_key == "yes")
        self.save()
        self.ok(f"EDM-ARS will send every step to {model} at {base_url}.")
        return True

    def _local_key(self, key: str, real_key: bool) -> None:
        """Store the server key, or a harmless placeholder, under OPENAI_API_KEY.

        The pipeline sends OPENAI_API_KEY to the server with every request, so
        a real OpenAI key saved earlier would reach it: offer to replace it.
        """
        from edmars import secrets

        env_var = str(_doctor.provider_meta("local")["env_var"] or "OPENAI_API_KEY")
        if real_key:
            self._store_key(env_var, key)
            return
        source = secrets.secret_source(env_var)
        if source is None:
            self._store_key(env_var, LOCAL_PLACEHOLDER_KEY)
        elif source == "env":
            if os.environ.get(env_var) != LOCAL_PLACEHOLDER_KEY:
                self.warn(f"The environment variable {env_var} is set on this computer and will be sent to "
                          "your server. If it holds a real OpenAI key, remove it before running studies.")
        elif secrets.get_secret(env_var) != LOCAL_PLACEHOLDER_KEY:
            if self.ni:
                self.warn(f"An OpenAI key is saved as {env_var}, and EDM-ARS sends that variable to your server. "
                          "Run `edmars setup ai` interactively to replace it with a placeholder.")
            elif self.yes(f"An OpenAI key is already saved. EDM-ARS sends {env_var} to your server with every "
                          "request. Replace the saved key with a harmless placeholder?", default=True):
                self._store_key(env_var, LOCAL_PLACEHOLDER_KEY)

    def _local_noninteractive(self) -> None:
        base_url = str(self.opt("base_url", self.get("provider_base_url", "")) or "").rstrip("/")
        if not re.match(r"^https?://\S+$", base_url):
            self.error("provider=local needs a server address: set base_url (EDMARS_BASE_URL), "
                       "for example http://localhost:11434/v1.")
            return
        key_env = self.opt("key_env")
        key = os.environ.get(str(key_env), "") if key_env else ""
        result = _KeyResult("OK")
        if _truthy(self.opt("check_keys", True)):
            result = self._check_key("local", key or LOCAL_PLACEHOLDER_KEY, base_url=base_url)
            if result.status != "OK":
                self.warn(f"Couldn't confirm the server at {base_url} ({result.status}).")
        model = str(self.opt("model", "") or "") or (result.models[0] if result.models else "")
        if not model:
            self.error("provider=local needs a model name: set model (EDMARS_MODEL).")
            return
        self.set("provider", "local")
        self.set("provider_base_url", base_url)
        self.set("models", {stage: model for stage in self._stage_keys()})
        self._local_key(key or LOCAL_PLACEHOLDER_KEY, bool(key))
        self.ok(f"EDM-ARS will send every step to {model} at {base_url} (experimental).")

    # =========================================================================
    # S5 Literature search
    # =========================================================================
    def screen_s5(self) -> None:
        from edmars import secrets

        if self.ni:
            key_env = str(self.opt("semantic_scholar_key_env", SEMANTIC_SCHOLAR_ENV))
            key = os.environ.get(key_env, "").strip()
            if not key:
                have = secrets.secret_source(SEMANTIC_SCHOLAR_ENV) is not None
                self.set("literature.semantic_scholar_key_set", have)
                if not have:
                    self.info("No Semantic Scholar key given; literature search runs without one.")
                return
            if _truthy(self.opt("check_keys", True)) and self._check_s2(key) == "REJECTED":
                self.error(f"The Semantic Scholar key in {key_env} was rejected; it was not saved.")
                return
            if key_env != SEMANTIC_SCHOLAR_ENV and not self._store_key(SEMANTIC_SCHOLAR_ENV, key):
                return
            self.set("literature.semantic_scholar_key_set", True)
            return

        body = (
            "EDM-ARS searches Semantic Scholar and arXiv for related studies and cites them. Semantic Scholar "
            "works without a key but often turns away keyless requests. When that happens your paper may end "
            "up with few real citations.\n\n"
            f"A free key is available from {SEMANTIC_SCHOLAR_FORM} (approval can take a few days).\n\n"
            "arXiv and Crossref need no key."
        )
        self.header("S5", body, title="Literature search (recommended)")
        while True:
            source = secrets.secret_source(SEMANTIC_SCHOLAR_ENV)
            choices: list[tuple[str, str]] = []
            if source:
                choices.append(("keep", f"Keep the key saved in {_doctor.store_label(source, SEMANTIC_SCHOLAR_ENV)}"))
            choices.append(("add", "Replace it with a new key" if source else "Add my Semantic Scholar key now"))
            if _can_open_browser():
                choices.append(("open", "Open the request form in my browser"))
            choices.append(("skip", "Skip for now (add it later with `edmars setup literature`)"))
            answer = self.choose("Semantic Scholar key", choices, default="keep" if source else "skip")
            if answer == "keep":
                self.set("literature.semantic_scholar_key_set", True)
                return
            if answer == "skip":
                self.set("literature.semantic_scholar_key_set", source is not None)
                return
            if answer == "open":
                if not _open_url(SEMANTIC_SCHOLAR_FORM):
                    self.info(f"Open this address yourself: {SEMANTIC_SCHOLAR_FORM}")
                continue
            key = self.ask_secret("Paste your Semantic Scholar key (it stays hidden; empty goes back)")
            if not key:
                continue
            status = self._check_s2(key)
            if status == "REJECTED":
                continue
            if status != "OK" and self.choose(
                    "What would you like to do?", [("save", "Save the key anyway"), ("again", "Try again")],
                    default="save", back=False) != "save":
                continue
            if self._store_key(SEMANTIC_SCHOLAR_ENV, key):
                self.set("literature.semantic_scholar_key_set", True)
                return

    def _check_s2(self, key: str) -> str:
        from edmars import providers

        self.info("Checking the key with Semantic Scholar\u2026")
        try:
            result = providers.check_semantic_scholar(key)
            status = str(getattr(result, "status", "UNKNOWN")).upper()
            message = _doctor.redact(str(getattr(result, "message", "") or ""), [key])
        except Exception as exc:  # noqa: BLE001
            status, message = "UNKNOWN", _doctor.redact(str(exc), [key])
        if status == "OK":
            self.ok("Key works.")
        elif status == "REJECTED":
            self.fail("Semantic Scholar didn't accept this key. Make sure you copied all of it.")
        elif status == "NETWORK":
            self.warn("Couldn't reach Semantic Scholar. Check your internet connection.")
        else:
            self.warn("Semantic Scholar answered in an unexpected way" + (f" ({message})." if message else "."))
        return status

    # =========================================================================
    # S6 Datasets
    # =========================================================================
    def screen_s6(self) -> None:
        from edmars import datasets

        if self.ni:
            self._s6_noninteractive()
            return
        body = (
            "The High School Longitudinal Study of 2009 (HSLS:09) followed 23,503 U.S. 9th graders into "
            "college and work. EDM-ARS uses the public-use file from NCES.\n"
            "Download: about 297 MB \u00b7 On disk: about 2.0 GB \u00b7 Source: nces.ed.gov\n"
            "Supports prediction, cause-and-effect and measurement studies.\n\n"
            "Use it under NCES public-use terms (for example, no attempts to identify individuals) and cite "
            "NCES in your work."
        )
        self.header("S6", body, title="Download the HSLS:09 dataset? (recommended)")
        while True:
            status = datasets.status(HSLS, self.s)
            ready = str(getattr(status, "status", "")) == "ok"
            if ready:
                self.ok(f"HSLS:09 is ready ({_nb(str(getattr(status, 'detail', '')))}).")
                choices = [("continue", "Continue"), ("other", "Other datasets")]
                default = "continue"
            else:
                detail = str(getattr(status, "detail", "") or "")
                if detail:
                    self.info(f"HSLS:09: {detail}")
                choices = [("download", "Download now (recommended)"),
                           ("import", "I already have the file"),
                           ("other", "Other datasets"),
                           ("skip", "Skip for now (add it later with `edmars setup datasets`)")]
                default = "download"
            answer = self.choose("What would you like to do?", choices, default=default)
            if answer in ("continue", "skip"):
                return
            if answer == "download":
                self._download(HSLS, "HSLS:09")
            elif answer == "import":
                self._import(HSLS, "HSLS:09")
            elif answer == "other":
                self._other_datasets()

    def _dataset_label(self, name: str) -> str:
        from edmars import datasets

        info = (getattr(datasets, "CATALOG", {}) or {}).get(name)
        for attr in ("label", "title", "name"):
            value = info.get(attr) if isinstance(info, dict) else getattr(info, attr, None)
            if isinstance(value, str) and value.strip() and value != name:
                return value.strip()
        return {"hsls09_public": "HSLS:09", "els_2002": "ELS:2002",
                "did_els_hsls_panel": "ELS/HSLS two-cohort panel"}.get(name, name)

    def _accept_terms(self, name: str, label: str) -> bool:
        self.panel(f"Terms for {label}", NCES_TERMS)
        if self.ni:
            self.info(f"Terms accepted for {label} because dataset_action=download was given.")
        else:
            answer = self.choose("Do you agree to use the data under these terms?",
                                 [("agree", "I agree \u2014 download it"), ("no", "Don't download")],
                                 default="agree", back=False)
            if answer != "agree":
                return False
        self.set(f"datasets.{name}.terms_accepted_at", _now())
        self.save()
        return True

    def _download(self, name: str, label: str) -> bool:
        from edmars import datasets

        if not self._accept_terms(name, label):
            return False
        dest = Path(datasets.raw_data_dir(self.s))
        try:
            dest.mkdir(parents=True, exist_ok=True)
            free = shutil.disk_usage(dest).free
        except OSError as exc:
            self._report(f"EDM-ARS can't write to {dest}: {_doctor.redact(str(exc))}")
            return False
        if name == HSLS and free < HSLS_DOWNLOAD_BYTES_NEEDED:
            self._report(f"Not enough free disk space in {dest}: {free / 1024 ** 3:.1f} GB free, about 2.5 GB "
                         "needed. Free up space and try again.")
            return False
        self.info(f"Downloading {label} into {dest}. If it stops, run `edmars setup datasets` again and it "
                  "continues where it left off.")
        progress = _DownloadProgress(self)
        try:
            # install() passes settings through, so the file's SHA-256 is
            # recorded on first download (trust on first use).
            path = Path(datasets.install(name, self.s, progress=progress))
        except KeyboardInterrupt:
            progress.close()
            self.warn("Download paused. Run `edmars setup datasets` to continue it.")
            raise _Quit()
        except Exception as exc:  # noqa: BLE001
            progress.close()
            self._report(f"The download stopped: {_doctor.redact(str(exc))}. Run `edmars setup datasets` "
                         "to try again; it continues where it stopped.")
            return False
        progress.close()
        return self._validate_and_record(name, label, path)

    def _validate_and_record(self, name: str, label: str, path: Path) -> bool:
        from edmars import datasets

        chk = datasets.validate_file(name, path)
        if str(getattr(chk, "status", "")) == "fail":
            self._report(f"{label}: {getattr(chk, 'detail', '')}")
            if getattr(chk, "fix", None):
                self.info(str(chk.fix))
            return False
        if str(getattr(chk, "status", "")) == "warn":
            self.warn(f"{label}: {getattr(chk, 'detail', '')}")
        self.set(f"datasets.{name}.path", str(path))
        self.set(f"datasets.{name}.verified_at", _now())
        self.save()
        self.ok(f"{label} is ready: {path}")
        return True

    def _import(self, name: str, label: str) -> bool:
        from edmars import datasets

        while True:
            typed = self.ask_text(f"Type or paste the full path of the {label} file (the labeled CSV, e.g. "
                                  f"{HSLS_FILENAME}, or the NCES .zip). Leave empty to go back.")
            if not typed:
                return False
            path = _clean_path(typed)
            if not path.is_file():
                self.fail(f"There's no file at {path}.")
                continue
            if path.suffix.lower() != ".zip":
                chk = datasets.validate_file(name, path)
                if str(getattr(chk, "status", "")) == "fail":
                    self.fail(f"{label}: {getattr(chk, 'detail', '')}")
                    if getattr(chk, "fix", None):
                        self.info(str(chk.fix))
                    continue
            try:
                stored = Path(datasets.import_file(name, path, self.s))
            except Exception as exc:  # noqa: BLE001
                self.fail(f"Couldn't import the file: {_doctor.redact(str(exc))}")
                continue
            self.set(f"datasets.{name}.path", str(stored))
            self.set(f"datasets.{name}.verified_at", _now())
            self.save()
            self.ok(f"{label} is ready: {stored}")
            return True

    def _other_datasets(self) -> None:
        from edmars import datasets

        while True:
            els_ready = str(getattr(datasets.status("els_2002", self.s), "status", "")) == "ok" \
                if "els_2002" in (datasets.CATALOG or {}) else False
            hsls_ready = str(getattr(datasets.status(HSLS, self.s), "status", "")) == "ok"
            choices: list[tuple[str, str]] = []
            if "els_2002" in (datasets.CATALOG or {}):
                choices.append(("els", "ELS:2002 (experimental): " + ("ready" if els_ready else "download about 18 MB")))
            if "did_els_hsls_panel" in (datasets.CATALOG or {}):
                if els_ready and hsls_ready:
                    choices.append(("panel", "Build the ELS/HSLS two-cohort panel (needed for cohort-gap studies)"))
                else:
                    self.info("The ELS/HSLS two-cohort panel is built automatically once both HSLS:09 and "
                              "ELS:2002 are installed.")
            self.info("ASSISTments: manual install, coming later. Your own data isn't supported yet.")
            choices.append(("done", "Done with other datasets"))
            answer = self.choose("Other datasets", choices, default="done", back=False)
            if answer == "done":
                return
            if answer == "els":
                if els_ready:
                    self.ok("ELS:2002 is already installed.")
                else:
                    self._download("els_2002", "ELS:2002")
            elif answer == "panel":
                self.info("Building the two-cohort panel (this takes a minute or two)\u2026")
                try:
                    built = datasets.build_did_panel(self.s)
                    self.ok(f"Two-cohort panel ready: {built}")
                except Exception as exc:  # noqa: BLE001
                    self.fail(f"Couldn't build the panel: {_doctor.redact(str(exc))}")

    def _s6_noninteractive(self) -> None:
        from edmars import datasets

        name = str(self.opt("dataset", HSLS) or HSLS)
        label = self._dataset_label(name)
        action = str(self.opt("dataset_action", "") or "").lower()
        if name not in (getattr(datasets, "CATALOG", {}) or {}):
            self.error(f"Unknown dataset '{name}'. Known: {', '.join(datasets.CATALOG)}.")
            return
        ready = str(getattr(datasets.status(name, self.s), "status", "")) == "ok"
        if action in ("", "keep"):
            if ready:
                self.ok(f"{label} is ready.")
            else:
                self.info(f"{label} is not installed (use --dataset-action download or import to add it).")
            return
        if action == "skip":
            self.info(f"Skipping {label}.")
            return
        if action == "download":
            if ready:
                self.ok(f"{label} is already installed.")
                return
            self._download(name, label)
            return
        if action == "import":
            raw = self.opt("dataset_path")
            if not raw:
                self.error("dataset_action=import needs dataset_path (EDMARS_DATASET_PATH).")
                return
            path = _clean_path(str(raw))
            if not path.is_file():
                self.error(f"There's no file at {path}.")
                return
            if path.suffix.lower() != ".zip":
                chk = datasets.validate_file(name, path)
                if str(getattr(chk, "status", "")) == "fail":
                    self.error(f"{label}: {getattr(chk, 'detail', '')}")
                    if getattr(chk, "fix", None):
                        self.info(str(chk.fix))
                    return
            try:
                stored = Path(datasets.import_file(name, path, self.s))
            except Exception as exc:  # noqa: BLE001
                self.error(f"Couldn't import {path}: {_doctor.redact(str(exc))}")
                return
            self.set(f"datasets.{name}.path", str(stored))
            self.set(f"datasets.{name}.verified_at", _now())
            self.ok(f"{label} is ready: {stored}")
            return
        self.error(f"Unknown dataset_action '{action}'. Use download, import or skip.")

    def _report(self, text: str) -> None:
        """A failure: an error in non-interactive mode, a message otherwise."""
        if self.ni:
            self.error(text)
        else:
            self.fail(text)

    # =========================================================================
    # S7 PDF typesetting
    # =========================================================================
    def screen_s7(self) -> None:
        from edmars import toolchain

        # toolchain also looks next to a saved pdflatex and in TinyTeX's own
        # folder, which is not on PATH until a new terminal is opened.
        pdflatex = toolchain.find_tex_tool("pdflatex", self.s)
        miktex = bool(toolchain.find_tex_tool("initexmf", self.s))
        mode_now = str(self.get("latex.mode", "") or "")
        if self.ni:
            self._s7_noninteractive(pdflatex)
            return
        body = (
            "EDM-ARS writes papers in LaTeX, the format most journals and conferences use. To make PDFs it "
            "needs a typesetting program. TinyTeX is free and about 300 MB. Without one you get the LaTeX "
            "source and figures (you can make the PDF in Overleaf), but the automated reviewer can't run."
        )
        self.header("S7", body, title="Make PDFs of your papers? (recommended)")
        choices: list[tuple[str, str]] = []
        if pdflatex:
            what = "TinyTeX" if mode_now == "tinytex" else ("MiKTeX" if miktex else "LaTeX")
            choices.append(("system", f"Use the {what} already on this computer"))
            choices.append(("tinytex", "Install TinyTeX instead (about 300 MB)"))
        else:
            choices.append(("tinytex", "Install TinyTeX (recommended, about 300 MB)"))
        choices.append(("skip", "Skip: I'll make PDFs myself, for example in Overleaf"))
        while True:
            answer = self.choose("What would you like to do?", choices,
                                 default="skip" if mode_now == "none" else choices[0][0])
            if answer == "skip":
                self._set_latex("none")
                self.warn("PDFs are off. Studies will produce LaTeX source and figures, and the automated "
                          "reviewer can't run. Turn PDFs on any time with `edmars setup pdf`.")
                return
            if answer == "tinytex":
                if not self.yes("This downloads the official TinyTeX installer from yihui.org and installs it "
                                "in your user folder (no administrator rights needed). It can take 5-15 minutes. "
                                "Go ahead?", default=True):
                    continue
                if not self._install_tinytex():
                    continue
                mode = "tinytex"
            else:
                mode = "tinytex" if mode_now == "tinytex" else "system"
            self._set_latex(mode)
            if self._latex_verify(compile_test=True):
                return
            fix = self.choose("Some parts are missing, so papers may not compile. What would you like to do?",
                              [("keep", "Continue anyway (fix it later; `edmars doctor --deep` rechecks)"),
                               ("again", "Choose again")], default="keep", back=False)
            if fix == "keep":
                return

    def _set_latex(self, mode: str) -> None:
        self.set("latex.mode", mode)
        self.set("latex.pdflatex", self._find_pdflatex(mode) if mode != "none" else None)
        self.save()

    def _find_pdflatex(self, mode: str) -> str | None:
        """The pdflatex that ``mode`` means: TinyTeX's own for "tinytex"
        (right after an install it is not on PATH yet), else the one PATH
        or the saved setting finds."""
        from edmars import toolchain

        if mode == "tinytex":
            for folder in toolchain.tinytex_bin_dirs():
                for name in ("pdflatex.exe", "pdflatex"):
                    candidate = Path(folder) / name
                    if candidate.is_file():
                        return str(candidate)
        return toolchain.find_tex_tool("pdflatex", self.s)

    def _install_tinytex(self) -> bool:
        from edmars import toolchain

        self.info("Installing TinyTeX. Messages from the installer may appear below.")
        try:
            chk = toolchain.install_tinytex(settings=self.s, on_step=self.info)
        except Exception as exc:  # noqa: BLE001
            self._report(f"TinyTeX could not be installed: {_doctor.redact(str(exc))}")
            return False
        self.show_checks([chk])
        if str(getattr(chk, "status", "")) == "fail":
            if self.ni:
                self.errors.append(str(getattr(chk, "detail", "TinyTeX install failed")))
            return False
        return True

    def _latex_verify(self, *, compile_test: bool) -> bool:
        from edmars import toolchain

        checks = list(toolchain.latex_checks(self.s))
        if compile_test:
            self.info("Making two small test PDFs to confirm everything works (up to 4 minutes the first time)\u2026")
            checks += list(toolchain.test_compile(timeout_s=120, settings=self.s))
        self.show_checks(checks)
        return not any(str(getattr(c, "status", "")) == "fail" for c in checks)

    def _s7_noninteractive(self, pdflatex: str | None) -> None:
        action = str(self.opt("latex_action", "auto") or "auto").lower()
        if action == "auto":
            action = "system" if pdflatex else "skip"
            if not pdflatex:
                self.info("No LaTeX found; PDFs stay off (use --latex-action tinytex to install TinyTeX).")
        if action in ("skip", "none"):
            self._set_latex("none")
            return
        if action == "tinytex":
            if not self._install_tinytex():
                self._set_latex("none")
                return
            self._set_latex("tinytex")
        elif action == "system":
            if not pdflatex:
                self.error("latex_action=system, but no pdflatex was found on PATH.")
                return
            self._set_latex("system")
        else:
            self.error(f"Unknown latex_action '{action}'. Use auto, system, tinytex or skip.")
            return
        if not self._latex_verify(compile_test=_truthy(self.opt("latex_test", False))):
            self.warn("LaTeX has problems (see above); papers may not compile.")

    # =========================================================================
    # S8 R for measurement studies
    # =========================================================================
    def screen_s8(self) -> None:
        from edmars import toolchain

        if self.ni:
            self._s8_noninteractive()
            return
        body = (
            "R is a free statistics program. EDM-ARS uses it for factor analysis and item-response models in "
            "measurement (psychometrics) studies. You only need it for those studies."
        )
        self.header("S8", body, title="Measurement studies need R (optional)")
        found = self.get("r.rscript", None) or toolchain.find_rscript(self.s)
        while True:
            choices: list[tuple[str, str]] = [("later", "Not now: ask me when I start a measurement study")]
            if found:
                choices.append(("find", f"Use the R found at {found}"))
            else:
                choices.append(("find", "I already have R: find it"))
                choices.append(("install", "Help me install R and the packages EDM-ARS needs (about 500 MB)"))
            # "Not now" stays the default even when R is found: only measurement
            # studies need it, and checking it may offer package installs.
            answer = self.choose("What would you like to do?", choices, default="later")
            if answer == "later":
                return
            if answer == "install":
                page = CRAN_PAGES.get(sys.platform, CRAN_PAGES["linux"])
                self.panel("Install R",
                           f"1. Download R from the official site: {page}\n"
                           "2. Run the installer with its default options. On Windows it may ask for "
                           "administrator permission.\n"
                           "3. Come back here and choose \"R is installed\".")
                if _can_open_browser():
                    _open_url(page)
                nxt = self.choose("When R is installed:", [("find", "R is installed: find it now"),
                                                           ("later", "I'll do it later")],
                                  default="find", back=False)
                if nxt == "later":
                    return
            if self._find_r(found):
                return
            found = None

    def _find_r(self, found: str | None) -> bool:
        from edmars import toolchain

        rscript = found or toolchain.find_rscript(self.s)
        while not rscript:
            typed = self.ask_text("R wasn't found automatically. Type or paste the path to Rscript (or to R's "
                                  "folder). Leave empty to skip.")
            if not typed:
                return False
            normalized = _normalize_rscript(_clean_path(typed))
            if normalized is None:
                self.fail("There's no Rscript there. On Windows it is usually in "
                          "C:\\Program Files\\R\\R-<version>\\bin.")
                continue
            rscript = str(normalized)
        return self._configure_r(str(rscript), install_packages=None)

    def _configure_r(self, rscript: str, *, install_packages: bool | None) -> bool:
        """Save the Rscript path, check the packages, offer to install missing ones."""
        from edmars import toolchain

        self.set("r.rscript", rscript)
        self.save()
        self.info(f"Checking R at {rscript}\u2026")
        checks = list(toolchain.r_checks(self.s))
        self.show_checks(checks)
        missing = any(str(getattr(c, "status", "")) == "fail" for c in checks)
        if missing:
            if install_packages is None:
                install_packages = self.yes(
                    "Some R packages EDM-ARS needs are missing. Install them now? They come from a dated "
                    "Posit Package Manager snapshot and go into your personal R library.", default=True)
            if install_packages:
                self.info("Installing R packages (this can take several minutes)\u2026")
                try:
                    result = toolchain.install_r_packages(rscript)
                except Exception as exc:  # noqa: BLE001
                    self._report(f"Installing R packages failed: {_doctor.redact(str(exc))}")
                    result = None
                if result is not None:
                    self.show_checks([result])
                checks = list(toolchain.r_checks(self.s))
                missing = any(str(getattr(c, "status", "")) == "fail" for c in checks)
                if missing:
                    self.show_checks(checks)
        self.set("r.packages_ok", not missing)
        self.save()
        if missing:
            self.warn("R is set up, but some packages are still missing. Measurement studies won't run until "
                      "they are installed (`edmars setup r`).")
        else:
            self.ok("R is ready for measurement studies.")
        return True

    def _s8_noninteractive(self) -> None:
        from edmars import toolchain

        action = str(self.opt("r_action", "skip") or "skip").lower()
        if action in ("skip", "later", "none"):
            return
        if action not in ("find", "install"):
            self.error(f"Unknown r_action '{action}'. Use skip, find or install.")
            return
        raw = self.opt("rscript")
        rscript: str | None
        if raw:
            normalized = _normalize_rscript(_clean_path(str(raw)))
            rscript = str(normalized) if normalized else None
            if rscript is None:
                self.error(f"There's no Rscript at {raw}.")
                return
        else:
            rscript = toolchain.find_rscript(self.s)
        if not rscript:
            self.error("R was not found. Install R, or give its path with --rscript.")
            return
        self._configure_r(rscript, install_packages=(action == "install"))
        if action == "install" and not self.get("r.packages_ok", False):
            self.errors.append("R packages are still missing")

    # =========================================================================
    # S9 Automated peer review (LSAR)
    # =========================================================================
    def screen_s9(self) -> None:
        from edmars import secrets

        provider_id = str(self.get("provider", "deepseek") or "deepseek")
        enabled = bool(self.get("lsar.enabled", False))
        if self.ni:
            self._s9_noninteractive()
            return
        body = (
            "LSAR is EDM-ARS's companion reviewer. It reads your finished PDF and scores it from 1 to 10 the "
            "way a conference reviewer would, using accepted papers as the benchmark. If the score is low, "
            "EDM-ARS revises the paper and asks again, up to 2 rounds.\n\n"
            f"It adds about 20-40 minutes per study and costs {estimates.REVIEW_COST_DEEPSEEK}. "
            "It needs a DeepSeek key, because LSAR's scoring was "
            "calibrated with DeepSeek, and it needs PDFs (step 7).\n\n"
            "Good to know: two readings of the same paper can differ by about 2 points. Treat the score as a "
            "rough signal, not a verdict."
        )
        self.header("S9", body, title="Automated peer review (recommended)")
        if enabled:
            default = "auto" if self.get("lsar.auto_review", False) else "manual"
        else:
            default = "skip" if provider_id == "local" else "auto"
        choices = [("auto", "Install it and review every study automatically"),
                   ("manual", "Install it, but only review when I ask (`edmars review`)"),
                   ("skip", "Turn it off" if enabled else "Skip")]
        while True:
            answer = self.choose("What would you like to do?", choices, default=default)
            if answer == "skip":
                self.set("lsar.enabled", False)
                self.set("lsar.auto_review", False)
                self.save()
                self.info("The automated reviewer is off. Turn it on any time with `edmars setup reviewer`.")
                return
            if secrets.secret_source(DEEPSEEK_ENV) is None:
                label = _doctor.provider_meta(provider_id)["label"]
                self.info("The reviewer needs a DeepSeek key, because LSAR's scoring was calibrated with DeepSeek"
                          + (f". Your studies still use {label}." if provider_id != "deepseek" else "."))
                try:
                    have_key = self._key_flow("deepseek", purpose="the automated reviewer")
                except _Back:
                    continue
                if not have_key:
                    self.set("lsar.enabled", False)
                    self.save()
                    self.warn("Without a DeepSeek key the reviewer can't run, so it stays off. Turn it on later "
                              "with `edmars setup reviewer`.")
                    return
            break
        if str(self.get("latex.mode", "") or "") == "none":
            self.warn("The reviewer reads the PDF, and PDF typesetting is off. Turn it on with `edmars setup pdf`, "
                      "or every review will be skipped.")
        if not self._ensure_lsar(consent=None):
            self.set("lsar.enabled", False)
            self.save()
            return
        self.set("lsar.enabled", True)
        self.set("lsar.auto_review", answer == "auto")
        self.save()
        if answer == "auto":
            self.ok("The automated reviewer is on: every study will be reviewed.")
        else:
            self.ok("The automated reviewer is installed. Review a finished study with `edmars review`.")

    def _lsar_ready(self) -> bool:
        from edmars import lsar

        try:
            checks = list(lsar.checks(self.s))
        except Exception:
            return False
        return bool(checks) and not any(str(getattr(c, "status", "")) == "fail" for c in checks)

    def _ensure_lsar(self, *, consent: bool | None) -> bool:
        from edmars import lsar

        if self._lsar_ready():
            self.ok("LSAR is already installed.")
            return True
        repo = str(getattr(lsar, "LSAR_REPO", "cgpan/LSAR-public"))
        if not repo.startswith("http"):
            repo = "https://github.com/" + repo.strip("/")
        ref = str(getattr(lsar, "LSAR_REF", "") or "")
        if consent is None:
            consent = self.yes(f"This downloads LSAR from {repo}" + (f" (version {ref})" if ref else "")
                               + " and installs the Python packages it needs into EDM-ARS's own Python. It takes "
                                 "a few minutes. Go ahead?", default=True)
        if not consent:
            self.info("LSAR was not installed; the reviewer stays off.")
            return False
        self.info("Installing LSAR (this takes a few minutes)\u2026")
        try:
            home = Path(lsar.install(self.s))
            problems = list(lsar.verify(home))
        except Exception as exc:  # noqa: BLE001
            self._report(f"LSAR could not be installed: {_doctor.redact(str(exc))}")
            return False
        if problems:
            self._report("LSAR was downloaded but is not usable: " + "; ".join(_nb(p) for p in problems))
            return False
        self.set("lsar.home", str(home))
        if ref and not self.get("lsar.ref", None):
            # install() saves the exact commit it unpacked; the constant is
            # only a branch name ("master"), so it must not replace that.
            self.set("lsar.ref", ref)
        self.save()
        self.ok(f"LSAR is installed in {home}.")
        return True

    def _s9_noninteractive(self) -> None:
        from edmars import secrets

        action = str(self.opt("lsar_action", "skip") or "skip").lower()
        if action in ("skip", "off", "none"):
            if self.get("lsar.enabled", False) and "lsar_action" in self.opts:
                self.set("lsar.enabled", False)
                self.set("lsar.auto_review", False)
            return
        if action not in ("auto", "manual"):
            self.error(f"Unknown lsar_action '{action}'. Use auto, manual or skip.")
            return
        key_env = self.opt("deepseek_key_env")
        if key_env or secrets.secret_source(DEEPSEEK_ENV) is None:
            if not self._key_noninteractive("deepseek", key_env, required=False):
                self.error("The automated reviewer needs a DeepSeek key (its scoring was calibrated with "
                           "DeepSeek). Put one in DEEPSEEK_API_KEY or name its variable with --deepseek-key-env.")
                self.set("lsar.enabled", False)
                return
        if not self._ensure_lsar(consent=True):
            self.set("lsar.enabled", False)
            return
        self.set("lsar.enabled", True)
        self.set("lsar.auto_review", action == "auto")
        self.ok("The automated reviewer is on" + (" for every study." if action == "auto" else " (on request)."))

    # =========================================================================
    # S10 Your name and advanced options
    # =========================================================================
    def screen_s10(self) -> None:
        if self.ni:
            self._s10_noninteractive()
            return
        self.header("S10", "Your name can appear in the author line of your papers. Both questions are optional.")
        name = self.ask_text("Your name for the paper's author line (press Enter to skip)",
                             default=str(self.get("author.name", "") or ""))
        affiliation = self.ask_text("Your university or organization (press Enter to skip)",
                                    default=str(self.get("author.affiliation", "") or ""))
        self.set("author.name", name or None)
        self.set("author.affiliation", affiliation or None)
        self.save()
        answer = self.choose("Change advanced options? Most people skip this.",
                             [("skip", "Skip: keep the recommended defaults"),
                              ("show", "Show advanced options")], default="skip")
        if answer == "show":
            self._advanced_menu()

    def _advanced_menu(self) -> None:
        from edmars import secrets

        while True:
            budget = self.get("defaults.budget_usd", None)
            fmt = str(self.get("defaults.paper_format", "conference") or "conference")
            venue = str(self.get("defaults.venue", "EDM") or "EDM")
            awake = bool(self.get("defaults.keep_awake", True))
            notify = bool(self.get("defaults.notify", True))
            models = self.get("models", {}) or {}
            mailto = self.get("literature.crossref_mailto", None)
            tavily = secrets.secret_source(TAVILY_ENV)
            menu = [
                ("budget", f"Spending warning per study: {('US$' + format(float(budget), 'g')) if budget else 'none'}"),
                ("format", f"Default paper format: {fmt}"),
                ("venue", f"Default venue: {venue}"),
                ("awake", f"Keep the computer awake during studies: {'on' if awake else 'off'}"),
                ("notify", f"Notify me when a study finishes: {'on' if notify else 'off'}"),
                ("models", f"AI models: {'custom' if models else 'recommended'}"),
                ("mailto", f"Contact email for Crossref (optional): {'set' if mailto else 'not set'}"),
                ("tavily", f"Tavily key for the reviewer's web search (optional): {'saved' if tavily else 'not set'}"),
                ("done", "Done"),
            ]
            answer = self.choose("Advanced options", menu, default="done", back=False)
            if answer == "done":
                return
            if answer == "budget":
                self._ask_budget()
            elif answer == "format":
                picked = self.choose("Default paper format",
                                     [("conference", "Conference paper (recommended; ACM two-column)"),
                                      ("journal", "Journal article (APA style, longer)")],
                                     default=fmt, back=False)
                self.set("defaults.paper_format", picked)
            elif answer == "venue":
                picked = self.choose("Default venue", list(VENUES.items()), default=venue, back=False)
                self.set("defaults.venue", picked)
            elif answer == "awake":
                self.set("defaults.keep_awake", self.yes("Keep the computer awake while a study runs?", default=awake))
            elif answer == "notify":
                self.set("defaults.notify", self.yes("Show a notification when a study finishes?", default=notify))
            elif answer == "models":
                self._ask_models()
            elif answer == "mailto":
                typed = self.ask_text("Email address Crossref may use to contact you about heavy use "
                                      "(press Enter to clear)", default=str(mailto or ""))
                if typed and not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", typed):
                    self.warn("That doesn't look like an email address; nothing was changed.")
                else:
                    self.set("literature.crossref_mailto", typed or None)
            elif answer == "tavily":
                key = self.ask_secret("Paste your Tavily key (it stays hidden; empty goes back)")
                if key:
                    self._store_key(TAVILY_ENV, key)
            self.save()

    def _ask_budget(self) -> None:
        current = self.get("defaults.budget_usd", None)
        while True:
            typed = self.ask_text("Warn me when a study costs more than (US$, empty for no warning)",
                                  default=format(float(current), "g") if current else "")
            typed = typed.replace("$", "").replace("US", "").strip()
            if not typed:
                self.set("defaults.budget_usd", None)
                return
            try:
                value = float(typed)
            except ValueError:
                self.warn("Please type a number, for example 2 or 0.50.")
                continue
            if value <= 0:
                self.warn("Please type an amount above zero.")
                continue
            self.set("defaults.budget_usd", value)
            self.info("EDM-ARS will warn you when a study passes this amount. It does not stop the study.")
            return

    def _ask_models(self) -> None:
        from edmars import providers

        provider_id = str(self.get("provider", "deepseek") or "deepseek")
        answer = self.choose("AI models",
                             [("recommended", "Use the recommended models (default)"),
                              ("custom", "Choose a model for each step")], default="recommended", back=False)
        if answer == "recommended":
            if provider_id == "local":
                self.info("A local server always uses the model you chose in `edmars setup ai`.")
                return
            self.set("models", {})
            return
        self.warn("Don't pick weaker models for preparing the data and running the analysis: the code those "
                  "steps write depends on the strongest model.")
        try:
            defaults = dict(providers.default_models(provider_id) or {})
        except Exception:
            defaults = {}
        current = dict(self.get("models", {}) or {})
        chosen: dict[str, str] = {}
        for stage in list(defaults) or list(MODEL_STAGES):
            now = str(current.get(stage) or defaults.get(stage) or "")
            label = STAGE_LABELS.get(stage, stage.replace("_", " "))
            typed = self.ask_text(f"Model for {label} ({stage})", default=now)
            if typed:
                chosen[stage] = typed
        self.set("models", {k: v for k, v in chosen.items() if v != defaults.get(k)} if provider_id != "local" else chosen)

    def _s10_noninteractive(self) -> None:
        for option, dotted in (("author_name", "author.name"), ("affiliation", "author.affiliation")):
            value = self.opt(option)
            if value is not None:
                self.set(dotted, str(value) or None)
        venue = self.opt("venue")
        if venue is not None:
            venue_id = str(venue).upper().replace(" ", "_")
            if venue_id not in VENUES:
                self.error(f"Unknown venue '{venue}'. Use one of: {', '.join(VENUES)}.")
            else:
                self.set("defaults.venue", venue_id)
        fmt = self.opt("paper_format")
        if fmt is not None:
            if str(fmt).lower() not in ("conference", "journal"):
                self.error(f"Unknown paper_format '{fmt}'. Use conference or journal.")
            else:
                self.set("defaults.paper_format", str(fmt).lower())
        budget = self.opt("budget_usd")
        if budget is not None:
            text = str(budget).replace("$", "").strip()
            if not text:
                self.set("defaults.budget_usd", None)
            else:
                try:
                    value = float(text)
                    if value <= 0:
                        raise ValueError
                    self.set("defaults.budget_usd", value)
                except ValueError:
                    self.error(f"budget_usd must be a number above zero, not '{budget}'.")

    # =========================================================================
    # S11 Final check
    # =========================================================================
    def screen_s11(self) -> None:
        self.header("S11", "Checking everything once more.")
        if self.in_flow:
            # Mark setup finished first, so the check does not report the
            # very setup it is concluding as unfinished.
            self.set("setup_progress.last_completed_screen", "S11")
            self.set("setup_progress.completed_at", _now())
            self.save()
        checks = _doctor.run_checks(self.s)
        self.show_checks(checks)
        counts = _doctor.summarize(checks)
        if counts.get("fail", 0):
            self.warn(f"{counts['fail']} thing(s) still need attention; the fixes are listed above. "
                      "Run `edmars doctor` any time to check again.")
        self.panel("Setup complete",
                   "Your settings are saved. Change any of them later with `edmars setup`.\n"
                   "Start a study with `edmars new`. Check your setup with `edmars doctor`.")
        if self.ni or not self.in_flow:
            return
        answer = self.choose("Start your first study now?",
                             [("new", "Yes, start a study"),
                              ("later", "Later (type `edmars new` any time)")],
                             default="later" if counts.get("fail", 0) else "new", back=False, quit_=False)
        self.start_study = answer == "new"


class _DownloadProgress:
    """Progress callback for ``datasets.download``.

    Accepts ``(done, total)`` in bytes, or keyword ``done=``/``total=``.
    Draws a rich bar, or prints a line every 10% in plain mode.
    """

    def __init__(self, wizard: _Wizard) -> None:
        self.wizard = wizard
        self.plain = _plain()
        self._bar: Any = None
        self._task: Any = None
        self._last_decile = -1
        self._last_mb = 0
        self._phase = "download"

    _PHASES = {"download": "Downloading", "extract": "Unpacking", "verify": "Checking"}

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        try:
            done = args[0] if args else kwargs.get("done", kwargs.get("downloaded", 0))
            total = args[1] if len(args) > 1 else kwargs.get("total")
            phase = str(args[2] if len(args) > 2 else kwargs.get("phase", "download"))
            done_i = int(done or 0)
            total_i = int(total) if total else None
        except (TypeError, ValueError):
            return
        if phase != self._phase:
            # datasets reports download, then extract (the 2 GB unzip), then
            # verify: each gets its own bar or its own 10% lines.
            self.close()
            self._last_decile = -1
            self._last_mb = 0
            self._phase = phase
            if self.plain:
                self.wizard.say(f"  {self._PHASES.get(phase, phase.capitalize())}:")
        if self.plain:
            if total_i:
                decile = min(10, done_i * 10 // max(total_i, 1))
                if decile != self._last_decile:
                    self._last_decile = decile
                    self.wizard.say(f"  {decile * 10}% ({done_i / 1024 ** 2:.0f} of {total_i / 1024 ** 2:.0f} MB)")
            elif done_i // (50 * 1024 ** 2) != self._last_mb:
                self._last_mb = done_i // (50 * 1024 ** 2)
                self.wizard.say(f"  {done_i / 1024 ** 2:.0f} MB")
            return
        try:
            if self._bar is None:
                from rich.progress import BarColumn, DownloadColumn, Progress, TimeRemainingColumn, \
                    TransferSpeedColumn

                from edmars import ui

                label = self._PHASES.get(self._phase, "Working")
                # A rich Progress needs the real Console, not ui.console's proxy.
                self._bar = Progress(label, BarColumn(), DownloadColumn(), TransferSpeedColumn(),
                                     TimeRemainingColumn(), console=ui.get_console(), transient=False)
                self._bar.start()
                self._task = self._bar.add_task("download", total=total_i)
            self._bar.update(self._task, completed=done_i, total=total_i)
        except Exception:
            self.plain = True

    def close(self) -> None:
        if self._bar is not None:
            try:
                self._bar.stop()
            except Exception:
                pass
            self._bar = None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def resolve_section(section: str | None) -> str | None:
    """Map a typed section to a :data:`SECTIONS` key, ``"__all__"`` or None.

    Raises KeyError for anything unknown.
    """
    if section is None or not str(section).strip():
        return None
    word = str(section).strip().lower().replace("_", "-")
    if word in START_OVER_WORDS:
        return "__all__"
    if word in SECTIONS:
        return word
    if word in SECTION_ALIASES:
        return SECTION_ALIASES[word]
    raise KeyError(section)


def _load_settings(non_interactive: bool) -> dict[str, Any]:
    from edmars import paths, ui
    from edmars import settings as st

    try:
        return st.load()
    except Exception as exc:  # noqa: BLE001
        where = paths.settings_path()
        ui.fail(_nb(f"Your settings file can't be read: {where} ({_doctor.redact(str(exc))})"))
        if non_interactive:
            ui.info("Move that file aside and run setup again.")
            raise _Abort(1)
        if ui.confirm("Start with fresh settings? The damaged file is kept next to it as settings.yaml.bak.",
                      default=True):
            backup = Path(where).with_name(Path(where).name + ".bak")
            os.replace(where, backup)
            return st.load()
        raise _Abort(1)


def _interrupt_types() -> tuple[type[BaseException], ...]:
    try:
        from edmars import ui

        err = getattr(ui, "NonInteractiveError", None)
        if isinstance(err, type) and issubclass(err, BaseException):
            return (err,)
    except Exception:
        pass
    return ()


def _launch_first_study() -> int:
    """Hand over to `edmars new` exactly as if the user had typed it."""
    from edmars import ui

    try:
        from edmars.cli import app
    except Exception:
        ui.info("Type `edmars new` to start your first study.")
        return 0
    try:
        result = app(["new"], standalone_mode=False, prog_name="edmars")
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001 - includes click.Abort (Ctrl+C inside `new`)
        if type(exc).__name__ == "Abort":
            return 130
        ui.fail(_nb(f"Couldn't start `edmars new` from here ({_doctor.redact(str(exc)) or type(exc).__name__})."))
        ui.info("Type `edmars new` to start your first study.")
        return 1
    return int(result) if isinstance(result, int) else 0


def run_setup(section: str | None = None, *, non_interactive: bool = False,
              options: dict[str, Any] | None = None) -> int:
    """Run the setup wizard, or one section of it. See the module docstring."""
    from edmars import ui

    try:
        target = resolve_section(section)
    except KeyError:
        ui.fail(_nb(f"There is no setup section called '{section}'."))
        ui.info("Sections: " + ", ".join(SECTIONS) + ". Or run `edmars setup` with no section.")
        return 2
    interrupts = _interrupt_types()
    wizard: _Wizard | None = None
    try:
        settings = _load_settings(non_interactive)
        wizard = _Wizard(settings, non_interactive=non_interactive, options=options or {})
        code = wizard.run(target)
    except (_Quit, KeyboardInterrupt):
        if wizard is not None:
            try:
                wizard.save()
            except Exception:
                pass
        ui.info("Setup paused. Your answers so far are saved. Run `edmars setup` to continue.")
        return 130
    except _Abort as exc:
        return exc.code
    except interrupts as exc:  # type: ignore[misc]
        ui.fail(_nb(f"Setup needs an answer, but it is running without a terminal ({exc})."))
        ui.info("Run `edmars setup` in a terminal, or pass --yes with the options for every step.")
        return 1
    if wizard is not None and wizard.start_study:
        return _launch_first_study()
    return code


__all__ = ["run_setup", "resolve_section", "SECTIONS", "SCREENS", "NONINTERACTIVE_OPTIONS"]
