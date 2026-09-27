"""Offline stand-ins for the ``edmars`` modules the wizard and doctor use.

The wizard and doctor import their collaborators lazily
(``from edmars import ui`` inside functions), so a test can swap every one
of them for a fake: :func:`install_fakes` registers each fake both in
``sys.modules`` and as an attribute of the ``edmars`` package, which
covers both import forms. The fakes follow the CLI_SPEC section-18 API
exactly; they never touch the real keyring, the network, or anything
outside ``tmp_path``.

The fake UI is scripted: each prompt pops the next answer from
``fakes.ui.script``. :data:`DEFAULT` accepts the prompt's default, an
exception instance is raised, and a prompt with an empty script fails the
test with the prompt text, so a script that drifts from the screens says
exactly where.
"""

from __future__ import annotations

import copy
import os
import re
import shutil
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

import pytest
import yaml

from edmars.ui import TransferProgress as _RealTransferProgress

#: Accept whatever default the prompt offers.
DEFAULT = object()

SECRET_NAMES = ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "SEMANTIC_SCHOLAR_API_KEY",
                "OPENALEX_API_KEY", "TAVILY_API_KEY", "MINIMAX_API_KEY")


# ---------------------------------------------------------------------------
# edmars.model
# ---------------------------------------------------------------------------

@dataclass
class Check:
    name: str
    status: Literal["ok", "warn", "fail", "info"]
    detail: str
    fix: str | None = None


# ---------------------------------------------------------------------------
# edmars.ui
# ---------------------------------------------------------------------------

class NonInteractiveError(RuntimeError):
    pass


class FakeConsole:
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines

    def print(self, *objects: Any, **_: Any) -> None:
        self.lines.append(" ".join(str(o) for o in objects))


class FakeUI:
    def __init__(self) -> None:
        self.script: list[Any] = []
        self.prompts: list[tuple[str, str, list[tuple[str, str]] | None]] = []
        #: The default each select prompt offered (None: Enter alone does not answer).
        self.defaults: dict[str, str | None] = {}
        self.lines: list[str] = []
        self.plain = True
        self.console: Any = FakeConsole(self.lines)
        self.opened: list[str] = []

    # prompts ------------------------------------------------------------------
    def _next(self, kind: str, message: str, choices: list[tuple[str, str]] | None = None) -> Any:
        self.prompts.append((kind, message, choices))
        if not self.script:
            shown = "" if choices is None else f" choices={[v for v, _ in choices]}"
            raise AssertionError(f"unexpected {kind} prompt: {message!r}{shown}")
        answer = self.script.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    def select(self, message: str, choices: list[tuple[str, str]], default: str | None = None) -> str:
        self.defaults[message] = default
        answer = self._next("select", message, list(choices))
        if answer is DEFAULT:
            # The real prompt asks again when Enter is pressed with no default.
            assert default is not None, f"{message!r} offers no default; the script must pick an option"
            return str(default)
        values = [v for v, _ in choices]
        assert answer in values, f"{answer!r} is not a choice for {message!r}: {values}"
        return str(answer)

    def text(self, message: str, default: str | None = None, validate: Callable[..., Any] | None = None) -> str:
        answer = self._next("text", message)
        return str(default or "") if answer is DEFAULT else str(answer)

    def secret(self, message: str) -> str:
        return str(self._next("secret", message))

    def confirm(self, message: str, default: bool = True) -> bool:
        answer = self._next("confirm", message)
        return default if answer is DEFAULT else bool(answer)

    # output -------------------------------------------------------------------
    def is_plain(self) -> bool:
        return self.plain

    def set_plain(self, flag: bool) -> None:
        self.plain = flag

    def ok(self, msg: str) -> None:
        self.lines.append(f"[ok] {msg}")

    def info(self, msg: str) -> None:
        self.lines.append(f"[i] {msg}")

    def warn(self, msg: str) -> None:
        self.lines.append(f"[!] {msg}")

    def fail(self, msg: str) -> None:
        self.lines.append(f"[x] {msg}")

    def panel(self, title: str, body: str) -> None:
        self.lines.append(f"== {title}\n{body}")

    def open_path(self, path: Any) -> None:
        self.opened.append(str(path))

    @property
    def output(self) -> str:
        return "\n".join(self.lines)

    def module(self) -> types.ModuleType:
        mod = types.ModuleType("edmars.ui")
        for name in ("select", "text", "secret", "confirm", "is_plain", "set_plain", "ok", "info", "warn",
                     "fail", "panel", "open_path"):
            setattr(mod, name, getattr(self, name))
        mod.console = self.console  # type: ignore[attr-defined]
        mod.NonInteractiveError = NonInteractiveError  # type: ignore[attr-defined]
        # The real class: it prints only through the say_fn it is given.
        mod.TransferProgress = _RealTransferProgress  # type: ignore[attr-defined]
        return mod


# ---------------------------------------------------------------------------
# edmars.paths / edmars.settings / edmars.disclosure
# ---------------------------------------------------------------------------

class FakePaths:
    def __init__(self, home: Path, app: Path) -> None:
        self.home = home
        self.app = app

    def app_root(self) -> Path:
        return self.app

    def home_override(self) -> Path | None:
        return self.home

    def config_dir(self) -> Path:
        return self.home

    def data_dir(self) -> Path:
        return self.home / "data"

    def cache_dir(self) -> Path:
        return self.home / "cache"

    def settings_path(self) -> Path:
        return self.home / "settings.yaml"

    def default_studies_dir(self) -> Path:
        return self.home / "EDM-ARS" / "studies"

    def sync_provider(self, path: Path) -> str | None:
        parts = {p.lower() for p in Path(path).parts}
        if "onedrive" in parts:
            return "OneDrive"
        if "dropbox" in parts:
            return "Dropbox"
        return None


SETTINGS_DEFAULTS: dict[str, Any] = {
    "schema": 1,
    "acknowledged": None,
    "studies_dir": None,
    "provider": "deepseek",
    "provider_base_url": None,
    "models": {},
    "literature": {"semantic_scholar_key_set": False, "crossref_mailto": None},
    "datasets": {},
    "latex": {"mode": None, "pdflatex": None},
    "r": {"rscript": None, "packages_ok": False},
    "lsar": {"enabled": False, "auto_review": False, "home": None, "ref": None},
    "defaults": {"venue": "EDM", "paper_format": "conference", "budget_usd": None, "keep_awake": True},
    "author": {"name": None},
    "setup_progress": {"last_completed_screen": None},
}


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (over or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


class FakeSettings:
    def __init__(self, paths: FakePaths) -> None:
        self.paths = paths
        self.DEFAULTS = copy.deepcopy(SETTINGS_DEFAULTS)
        self.saves = 0

    def load(self) -> dict[str, Any]:
        path = self.paths.settings_path()
        data: dict[str, Any] = {}
        if path.is_file():
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError("settings.yaml is not a mapping")
            data = loaded
        return _merge(self.DEFAULTS, data)

    def save(self, settings: dict[str, Any]) -> None:
        path = self.paths.settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(settings, sort_keys=True), encoding="utf-8")
        self.saves += 1

    def get(self, settings: dict[str, Any], dotted: str, default: Any = None) -> Any:
        node: Any = settings
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set_(self, settings: dict[str, Any], dotted: str, value: Any) -> None:
        node = settings
        parts = dotted.split(".")
        for part in parts[:-1]:
            if not isinstance(node.get(part), dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = value


class FakeDisclosure:
    ACK_VERSION = "2026-09-25"

    def ack_text(self) -> str:
        return "WHAT LEAVES YOUR COMPUTER\n- Your question goes to the AI service you choose."

    def disclaimer_text(self) -> str:
        return "# Disclaimer\nAI-generated drafts can be wrong. See [PRIVACY.md](PRIVACY.md)."

    def privacy_text(self) -> str:
        return "# Privacy\nNo telemetry."

    def is_acknowledged(self, settings: dict[str, Any]) -> bool:
        ack = settings.get("acknowledged") or {}
        return isinstance(ack, dict) and ack.get("version") == self.ACK_VERSION

    def record_ack(self, settings: dict[str, Any]) -> None:
        settings["acknowledged"] = {"version": self.ACK_VERSION, "at": "2026-09-25T10:00:00Z"}


# ---------------------------------------------------------------------------
# edmars.secrets / edmars.proc
# ---------------------------------------------------------------------------

class SecretStoreError(RuntimeError):
    """Stands in for edmars.secrets.SecretStoreError (no credential store)."""


class FakeSecrets:
    SecretStoreError = SecretStoreError

    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.backend = "keyring"
        self.set_calls: list[str] = []
        self.fail_on_set: BaseException | None = None
        self.file_fallback_works = True

    def get_secret(self, name: str) -> str | None:
        return os.environ.get(name) or self.store.get(name)

    def set_secret(self, name: str, value: str, *, allow_file: bool = False) -> str:
        if self.fail_on_set is not None and not (allow_file and self.file_fallback_works):
            raise self.fail_on_set
        self.store[name] = value
        self.set_calls.append(name)
        return "file" if self.fail_on_set is not None else self.backend

    def secrets_file(self) -> Path:
        return Path("~") / ".config" / "edm-ars" / "secrets.env"

    def delete_secret(self, name: str) -> None:
        self.store.pop(name, None)

    def secret_source(self, name: str) -> str | None:
        if os.environ.get(name):
            return "env"
        if name in self.store:
            return self.backend
        return None

    def stored_location(self, name: str) -> str | None:
        return self.backend if name in self.store else None

    def child_secrets(self, names: Iterable[str]) -> dict[str, str]:
        return {n: v for n in names if (v := self.get_secret(n))}

    def redact(self, text: str) -> str:
        values = [v for v in self.store.values() if len(v) >= 8]
        values += [os.environ[n] for n in SECRET_NAMES if len(os.environ.get(n, "")) >= 8]
        for value in sorted(set(values), key=len, reverse=True):
            text = text.replace(value, "<redacted>")
        return re.sub(r"sk-[A-Za-z0-9_-]{8,}", "sk-<redacted>", text)


class FakeProc:
    def __init__(self) -> None:
        self.tools: dict[str, str] = {}
        self.alive: set[int] = set()

    def which(self, name: str) -> str | None:
        return self.tools.get(name)

    def pid_alive(self, pid: int | None, *, started_at: float | None = None) -> bool:
        return pid in self.alive

    def run(self, *a: Any, **k: Any) -> Any:  # pragma: no cover - must never be called
        raise AssertionError("the wizard/doctor must not run processes in these tests")

    def spawn_detached(self, *a: Any, **k: Any) -> int:  # pragma: no cover
        raise AssertionError("no spawning in these tests")


# ---------------------------------------------------------------------------
# edmars.providers
# ---------------------------------------------------------------------------

@dataclass
class ProviderInfo:
    id: str
    label: str
    env_var: str
    key_page: str | None
    topup_url: str | None = None


@dataclass
class KeyCheck:
    status: str
    message: str
    balance: str | None = None
    models: list[str] | None = None


class FakeProviders:
    def __init__(self) -> None:
        self.PROVIDERS = {
            "deepseek": ProviderInfo("deepseek", "DeepSeek (recommended)", "DEEPSEEK_API_KEY",
                                     "https://platform.deepseek.com/api_keys", "https://platform.deepseek.com/top_up"),
            "openai": ProviderInfo("openai", "OpenAI (ChatGPT API) — works, less tested", "OPENAI_API_KEY",
                                   "https://platform.openai.com/api-keys"),
            "anthropic": ProviderInfo("anthropic", "Anthropic (Claude API) — works, less tested",
                                      "ANTHROPIC_API_KEY", "https://platform.claude.com/settings/keys"),
            "local": ProviderInfo("local", "A model on my own computer/server — experimental",
                                  "OPENAI_API_KEY", None),
        }
        self.KeyCheck = KeyCheck
        #: key -> result, or a list of results returned in turn (the last one repeats)
        self.results: dict[str, KeyCheck | list[KeyCheck]] = {}
        self.s2_results: dict[str, KeyCheck] = {}
        self.local_models = ["llama3.1:70b", "qwen2.5:72b"]
        self.retired: set[str] = set()
        self.calls: list[tuple[str, str | None]] = []

    def check_key(self, provider_id: str, key: str, base_url: str | None = None, timeout: int = 15) -> KeyCheck:
        self.calls.append((provider_id, base_url))
        if key in self.results:
            found = self.results[key]
            if isinstance(found, list):
                return found.pop(0) if len(found) > 1 else found[0]
            return found
        if provider_id == "local":
            return KeyCheck("OK", "server answered", models=list(self.local_models))
        return KeyCheck("OK", "key accepted", balance="US$4.87")

    def check_semantic_scholar(self, key: str) -> KeyCheck:
        return self.s2_results.get(key, KeyCheck("OK", "accepted"))

    def default_models(self, provider_id: str) -> dict[str, str]:
        if provider_id == "deepseek":
            return {"problem_formulator": "deepseek-v4-pro", "data_engineer": "deepseek-v4-pro",
                    "analyst": "deepseek-v4-pro", "critic": "deepseek-v4-pro", "writer": "deepseek-v4-pro",
                    "revision_writer": "deepseek-v4-pro", "outline_agent": "deepseek-flash",
                    "verifier": "deepseek-flash"}
        if provider_id == "anthropic":
            # The real module reads the shipped top-level ``models`` block.
            return {"problem_formulator": "anthropic-model", "writer": "anthropic-model"}
        # openai: the shipped config.yaml has no openai block, so the real
        # default_models("openai") is empty and setup must ask for a model.
        return {}

    def missing_models(self, provider_id: str, key: str, models: Iterable[str],
                       base_url: str | None = None) -> list[str]:
        return [m for m in models if m in self.retired]


# ---------------------------------------------------------------------------
# edmars.datasets / edmars.toolchain / edmars.lsar / edmars.runner / edmars.cli
# ---------------------------------------------------------------------------

@dataclass
class DatasetInfo:
    name: str
    label: str
    filename: str
    experimental: bool = False


LABELED_HEADER = "STU_ID,X1SEX,X1RACE,X3TGPAACAD,X4EVRATNDCLG\n"


class FakeDatasets:
    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root
        self.CATALOG = {
            "hsls09_public": DatasetInfo("hsls09_public", "HSLS:09", "hsls_17_student_pets_sr_v1_0.csv"),
            "els_2002": DatasetInfo("els_2002", "ELS:2002", "els_02_12_byf3pststu_v1_0.csv", True),
            "did_els_hsls_panel": DatasetInfo("did_els_hsls_panel", "ELS/HSLS panel", "panel.csv", True),
        }
        self.ready: set[str] = set()
        self.download_error: BaseException | None = None
        self.progress_calls: list[tuple[Any, ...]] = []
        self.imported: list[Path] = []

    def raw_data_dir(self, settings: dict[str, Any] | None = None) -> Path:
        return self.data_root

    def status(self, name: str, settings: dict[str, Any]) -> Check:
        label = self.CATALOG[name].label
        if name in self.ready:
            return Check(f"Dataset: {label}", "ok", "installed and verified")
        return Check(f"Dataset: {label}", "fail", "not installed", f"Run `edmars data install {name}`.")

    def validate_file(self, name: str, path: Path) -> Check:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        if not lines or "X1SEX" not in lines[0]:
            return Check("Dataset file", "fail", "This file is not the HSLS:09 student file.")
        if len(lines) > 1 and "Male" not in lines[1] and "Female" not in lines[1]:
            return Check("Dataset file", "fail",
                         "This is the numeric-coded version. EDM-ARS needs the labeled CSV (with text such as 'Male').",
                         "Download the labeled CSV from NCES, or choose Download now.")
        return Check("Dataset file", "ok", "labeled HSLS:09 student file")

    def download(self, name: str, dest_dir: Path, progress: Callable[..., None] | None = None, *,
                 settings: dict[str, Any] | None = None, session: Any | None = None,
                 force: bool = False) -> Path:
        if self.download_error is not None:
            raise self.download_error
        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        target = Path(dest_dir) / self.CATALOG[name].filename
        target.write_text(LABELED_HEADER + "1,Male,White,3.1,Yes\n", encoding="utf-8")
        if progress is not None:
            # The real module reports (done, total, phase). For HSLS:09: download
            # the zip, check its SHA-256, then convert the CSV inside it.
            for phase in ("download", "verify", "convert"):
                for done in (0, 50, 100):
                    progress(done, 100, phase)
                    self.progress_calls.append((done, 100, phase))
        self.ready.add(name)
        if settings is not None:
            settings.setdefault("datasets", {}).setdefault(name, {})["sha256"] = "0" * 64
        return target

    def install(self, name: str, settings: dict[str, Any], progress: Callable[..., None] | None = None, *,
                session: Any | None = None, force: bool = False) -> Path:
        if name == "did_els_hsls_panel":
            return self.build_did_panel(settings)
        return self.download(name, self.raw_data_dir(settings), progress, settings=settings,
                             session=session, force=force)

    def import_file(self, name: str, path: Path, settings: dict[str, Any]) -> Path:
        chk = self.validate_file(name, path)
        if chk.status == "fail":
            raise ValueError(chk.detail)
        self.data_root.mkdir(parents=True, exist_ok=True)
        target = self.data_root / self.CATALOG[name].filename
        shutil.copyfile(path, target)
        self.imported.append(Path(path))
        self.ready.add(name)
        return target

    def build_did_panel(self, settings: dict[str, Any], *, timeout_s: int = 3600) -> Path:
        self.ready.add("did_els_hsls_panel")
        return self.data_root / "panel.csv"


class FakeToolchain:
    def __init__(self, proc: "FakeProc", tinytex_dir: Path) -> None:
        self.proc = proc
        self.tinytex_dir = tinytex_dir
        self.latex: list[Check] = [Check("LaTeX", "ok", "pdflatex found")]
        self.compile: list[Check] = [Check("Test PDF", "ok", "both test documents compiled")]
        self.rscript: str | None = None
        self.packages_missing = False
        self.tinytex_installed = False
        self.latex_error: BaseException | None = None
        self.compile_calls = 0
        self.xgboost: list[Check] = [Check("XGBoost", "ok", "XGBoost 2.1.4 loads together with scikit-learn")]
        self.miktex_auto: str | None = "1"
        self.events: list[str] = []

    def latex_checks(self, settings: dict[str, Any] | None = None) -> list[Check]:
        if self.latex_error is not None:
            raise self.latex_error
        return list(self.latex)

    def test_compile(self, timeout_s: float = 120, settings: dict[str, Any] | None = None) -> list[Check]:
        self.compile_calls += 1
        self.events.append("test_compile")
        return list(self.compile)

    def miktex_autoinstall(self, settings: dict[str, Any] | None = None) -> str | None:
        return self.miktex_auto

    def set_miktex_autoinstall(self, settings: dict[str, Any] | None = None) -> Check:
        self.events.append("set_miktex_autoinstall")
        self.miktex_auto = "1"
        return Check("MiKTeX automatic package install", "ok",
                     "MiKTeX will now install missing LaTeX packages automatically.")

    def tinytex_bin_dirs(self) -> list[Path]:
        return [self.tinytex_dir]

    def find_tex_tool(self, name: str, settings: dict[str, Any] | None = None) -> str | None:
        saved = ((settings or {}).get("latex") or {}).get("pdflatex")
        if name == "pdflatex" and saved:
            return str(saved)
        found = self.proc.which(name)
        if found:
            return found
        candidate = self.tinytex_dir / name
        return str(candidate) if candidate.exists() else None

    def install_tinytex(self, *, settings: dict[str, Any] | None = None, session: Any | None = None,
                        timeout_s: float = 1800, max_rounds: int = 30,
                        on_step: Callable[[str], None] | None = None) -> Check:
        self.tinytex_installed = True
        # Like the real installer: pdflatex lands in TinyTeX's own folder,
        # which is NOT on PATH in the running process.
        self.tinytex_dir.mkdir(parents=True, exist_ok=True)
        (self.tinytex_dir / "pdflatex").write_text("", encoding="utf-8")
        if on_step is not None:
            on_step("Installing the LaTeX packages the templates need")
        return Check("TinyTeX", "ok", "installed")

    def find_rscript(self, settings: dict[str, Any] | None = None, **kwargs: Any) -> str | None:
        configured = (settings.get("r") or {}).get("rscript")
        return configured or self.rscript

    def r_checks(self, settings: dict[str, Any] | None = None) -> list[Check]:
        if self.packages_missing:
            return [Check("R", "ok", "R 4.5.1"), Check("R packages", "fail", "missing: lavaan, mirt",
                                                       "Run `edmars setup r`.")]
        return [Check("R", "ok", "R 4.5.1"), Check("R packages", "ok", "all installed")]

    def install_r_packages(self, rscript: str, packages: Any = None, *, repo: str | None = None,
                           timeout_s: float = 1800, settings: dict[str, Any] | None = None) -> Check:
        self.packages_missing = False
        self.install_settings = settings
        return Check("R packages", "ok", "installed")

    def docker_info(self) -> Check:
        return Check("Docker", "info", "Docker is not running")

    def xgboost_checks(self, timeout_s: float = 180) -> list[Check]:
        return list(self.xgboost)


class FakeLsar:
    LSAR_REPO = "https://github.com/cgpan/LSAR-public"
    LSAR_REF = "0123abc"

    def __init__(self, home: Path) -> None:
        self.home = home
        self.installed = False
        self.install_calls = 0
        self.commit = "0123abc4567def"
        self.deep_calls: list[bool] = []

    def checks(self, settings: dict[str, Any], *, deep: bool = False) -> list[Check]:
        self.deep_calls.append(deep)
        if self.installed:
            return [Check("Automated reviewer", "ok", f"LSAR installed at {self.home}")]
        return [Check("Automated reviewer", "fail", "LSAR is not installed", "Run `edmars setup reviewer`.")]

    def install(self, settings: dict[str, Any], *, ref: str | None = None, allow_changes: bool = False,
                session: Any | None = None, on_step: Callable[[str], None] | None = None) -> Path:
        self.install_calls += 1
        self.installed = True
        self.home.mkdir(parents=True, exist_ok=True)
        # Like the real install(): it records the exact commit it unpacked.
        settings.setdefault("lsar", {}).update({"home": str(self.home), "ref": self.commit})
        return self.home

    def verify(self, home: Path) -> list[str]:
        return []

    def benchmark_for(self, home: Path, venue: str | None) -> float | None:
        """Like the real one: the venue's p25 in the reviewer's calibration file."""
        try:
            data = yaml.safe_load((Path(home) / "calibration" / "anchors_edm.yaml").read_text(encoding="utf-8"))
        except OSError:
            return None
        if not isinstance(data, dict):
            return None
        if (venue or "EDM") == "EDM":
            value = data.get("overall_p25_full")
        else:
            entry = (data.get("venues") or {}).get(venue)
            value = entry.get("p25") if isinstance(entry, dict) else None
        return float(value) if value is not None else None


class FakeRunner:
    def __init__(self) -> None:
        self.latest: Path | None = None

    def latest_run(self, settings: dict[str, Any]) -> Path | None:
        return self.latest


class FakeCli:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def app(self, args: list[str], standalone_mode: bool = True, prog_name: str | None = None) -> int:
        self.calls.append(list(args))
        return 0


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

@dataclass
class Fakes:
    home: Path
    app: Path
    ui: FakeUI
    paths: FakePaths
    settings: FakeSettings
    disclosure: FakeDisclosure
    secrets: FakeSecrets
    proc: FakeProc
    providers: FakeProviders
    datasets: FakeDatasets
    toolchain: FakeToolchain
    lsar: FakeLsar
    runner: FakeRunner
    cli: FakeCli
    modules: dict[str, types.ModuleType] = field(default_factory=dict)

    def saved(self) -> dict[str, Any]:
        path = self.paths.settings_path()
        return yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else {}

    def write_settings(self, **over: Any) -> None:
        self.settings.save(_merge(self.settings.DEFAULTS, over))


def _module(name: str, obj: Any, attrs: Iterable[str]) -> types.ModuleType:
    mod = types.ModuleType(f"edmars.{name}")
    for attr in attrs:
        setattr(mod, attr, getattr(obj, attr))
    return mod


def install_fakes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Fakes:
    """Replace every collaborator module with a fake rooted in ``tmp_path``."""
    home = tmp_path / "home"
    app = tmp_path / "app"
    (app / "src").mkdir(parents=True)
    (app / "src" / "main.py").write_text("# stub\n", encoding="utf-8")
    (app / "config.yaml").write_text("llm_provider: deepseek\n", encoding="utf-8")
    home.mkdir()

    for name in SECRET_NAMES:
        monkeypatch.delenv(name, raising=False)
    for name in list(os.environ):
        if name.startswith("EDMARS_") or name == "MSYSTEM":
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("EDMARS_HOME", str(home))

    ui = FakeUI()
    paths = FakePaths(home, app)
    fakes = Fakes(
        home=home, app=app, ui=ui, paths=paths, settings=FakeSettings(paths), disclosure=FakeDisclosure(),
        secrets=FakeSecrets(), proc=FakeProc(), providers=FakeProviders(),
        datasets=FakeDatasets(home / "data" / "raw"), toolchain=None, lsar=FakeLsar(home / "lsar"),
        runner=FakeRunner(), cli=FakeCli(),
    )
    fakes.toolchain = FakeToolchain(fakes.proc, home / "TinyTeX" / "bin")
    model = types.ModuleType("edmars.model")
    model.Check = Check  # type: ignore[attr-defined]
    fakes.modules = {
        "model": model,
        "ui": ui.module(),
        "paths": _module("paths", paths, ("app_root", "home_override", "config_dir", "data_dir", "cache_dir",
                                           "settings_path", "default_studies_dir", "sync_provider")),
        "settings": _module("settings", fakes.settings, ("DEFAULTS", "load", "save", "get", "set_")),
        "disclosure": _module("disclosure", fakes.disclosure, ("ACK_VERSION", "ack_text", "disclaimer_text",
                                                               "privacy_text", "is_acknowledged", "record_ack")),
        "secrets": _module("secrets", fakes.secrets, ("get_secret", "set_secret", "delete_secret", "secret_source",
                                                       "child_secrets", "redact", "secrets_file", "stored_location",
                                                       "SecretStoreError")),
        "proc": _module("proc", fakes.proc, ("which", "pid_alive", "run", "spawn_detached")),
        "providers": _module("providers", fakes.providers, ("PROVIDERS", "KeyCheck", "check_key",
                                                             "check_semantic_scholar", "default_models",
                                                             "missing_models")),
        "datasets": _module("datasets", fakes.datasets, ("CATALOG", "status", "validate_file", "download",
                                                          "install", "import_file", "build_did_panel",
                                                          "raw_data_dir")),
        "toolchain": _module("toolchain", fakes.toolchain, ("latex_checks", "test_compile", "install_tinytex",
                                                             "tinytex_bin_dirs", "find_tex_tool",
                                                             "find_rscript", "r_checks", "install_r_packages",
                                                             "docker_info", "xgboost_checks",
                                                             "miktex_autoinstall", "set_miktex_autoinstall")),
        "lsar": _module("lsar", fakes.lsar, ("LSAR_REPO", "LSAR_REF", "checks", "install", "verify",
                                                    "benchmark_for")),
        "runner": _module("runner", fakes.runner, ("latest_run",)),
        "cli": _module("cli", fakes.cli, ("app",)),
    }
    import edmars

    for name, mod in fakes.modules.items():
        monkeypatch.setitem(sys.modules, f"edmars.{name}", mod)
        monkeypatch.setattr(edmars, name, mod, raising=False)
    return fakes
