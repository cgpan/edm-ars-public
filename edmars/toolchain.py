r"""External tools a study needs: LaTeX (for the PDF) and R (for measurement).

LaTeX
    ``latex_checks()`` looks for pdflatex/bibtex/biber/kpsewhich and the
    three classes/styles the paper templates use (acmart, apa7,
    biblatex-apa). ``test_compile()`` builds two tiny documents -- the
    conference template with BibTeX and the APA journal template with
    biblatex/Biber -- in a temporary folder, 120 s each, which is the only
    reliable way to know a real paper will compile. On MiKTeX the
    ``[MPM]AutoInstall`` setting matters: anything but "always" makes a
    background compile wait for a confirmation window nobody can see, so
    ``test_compile`` passes ``--disable-installer`` in that case (a fast,
    explained failure instead of a two-minute hang) and ``latex_checks``
    says how to fix it. ``install_tinytex()`` runs the official TinyTeX
    installer; the caller asks for consent first.

R
    ``find_rscript()`` looks in a saved path, ``EDM_ARS_RSCRIPT``, PATH,
    then every ``R-*`` install under Program Files and the per-user
    ``%LOCALAPPDATA%\Programs\R`` folder, NEWEST VERSION FIRST (the
    pipeline's own lookup tries three hard-coded versions, oldest first,
    before PATH -- see defect F1 -- so the CLI resolves R itself and hands
    the result to the run as ``EDM_ARS_RSCRIPT``). Packages are probed and
    installed with ``Rscript --vanilla``, exactly how the pipeline runs its
    R helpers.

Every process is started through ``edmars.proc``.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from edmars import fetch
from edmars.model import Check

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LATEX_TOOLS: tuple[str, ...] = ("pdflatex", "bibtex", "biber", "kpsewhich")

#: (file kpsewhich looks for, plain name, package that provides it)
LATEX_STYLE_FILES: tuple[tuple[str, str, str], ...] = (
    ("acmart.cls", "Conference paper template (acmart)", "acmart"),
    ("apa7.cls", "APA journal template (apa7)", "apa7"),
    ("apa.bbx", "APA reference style (biblatex-apa)", "biblatex-apa"),
)

TINYTEX_INSTALLER_URLS: dict[str, str] = {
    "windows": "https://yihui.org/tinytex/install-bin-windows.bat",
    "unix": "https://yihui.org/tinytex/install-bin-unix.sh",
}

#: Always installed into TinyTeX: the two templates' classes and their
#: bibliography machinery. Template packages are added by
#: :func:`tinytex_packages`; anything still missing is found by compiling.
TINYTEX_BASE_PACKAGES: tuple[str, ...] = (
    "acmart", "biblatex", "biblatex-apa", "apa7", "biber", "babel-english",
    # acmart's usual dependencies; saves a compile round each.
    "libertine", "inconsolata", "newtx", "totpages", "environ", "trimspaces",
    "ncctools", "comment", "xstring", "hyperxmp", "ifmtarg", "kastrup",
    "microtype", "cmap", "draftwatermark", "preprint", "textcase", "fontaxes",
    "xkeyval", "zref", "refcount", "etoolbox", "natbib", "caption", "float",
    "booktabs", "setspace", "fancyhdr", "geometry",
    # apa7's usual dependencies.
    "scalerel", "threeparttable", "endfloat", "csquotes", "lastpage",
)

#: LaTeX package name -> TeX Live package name (None: part of the base
#: install). Names not listed are assumed to be their own TL package.
_TL_PACKAGE_FOR: dict[str, str | None] = {
    "inputenc": None, "fontenc": None, "ifthen": None, "graphicx": "graphics",
    "amssymb": "amsfonts", "amsthm": "amscls", "subcaption": "caption",
    "nameref": "hyperref", "tabularx": "tools", "algpseudocode": "algorithmicx",
    "algorithm": "algorithms", "bbm": "bbm-macros",
}

R_PACKAGES: tuple[str, ...] = ("jsonlite", "lavaan", "mirt", "CDM", "MASS")
#: Posit Package Manager snapshot: a fixed date, so every user gets the
#: same package versions, served as binaries on Windows and macOS.
R_REPO_SNAPSHOT = "https://packagemanager.posit.co/cran/2026-09-01"
R_MIN_VERSION: tuple[int, int] = (4, 4)
R_DOWNLOAD_PAGE = "https://cran.r-project.org/"

_R_VERSION_DIR = re.compile(r"R-(\d+)\.(\d+)(?:\.(\d+))?", re.IGNORECASE)

TEST_BIB = r"""@article{edmarstest2026,
  author  = {Doe, Jane and Roe, Richard},
  title   = {A Test Reference for the {EDM-ARS} {PDF} Check},
  journal = {Journal of Tests},
  year    = {2026},
  volume  = {1},
  number  = {1},
  pages   = {1--2},
}
"""

ACM_TEST_TEX = r"""\documentclass[sigconf]{acmart}
\setcopyright{none}
\settopmatter{printacmref=false}
\renewcommand\footnotetextcopyrightpermission[1]{}
\begin{document}
\title{EDM-ARS PDF check}
\author{EDM-ARS}
\affiliation{\institution{Test}\country{Test}}
\begin{abstract}
A two-line document that proves the conference template compiles.
\end{abstract}
\maketitle
\section{Test}
One citation \cite{edmarstest2026}.
\bibliographystyle{ACM-Reference-Format}
\bibliography{refs}
\end{document}
"""

APA_TEST_TEX = r"""\documentclass[man,floatsintext,longtable]{apa7}
\usepackage[english]{babel}
\usepackage{csquotes}
\usepackage[style=apa, backend=biber]{biblatex}
\addbibresource{refs.bib}
\title{EDM-ARS PDF check}
\shorttitle{PDF check}
\authorsnames{EDM-ARS}
\authorsaffiliations{{Test}}
\abstract{A short document that proves the journal template compiles.}
\begin{document}
\maketitle
One citation \parencite{edmarstest2026}.
\printbibliography
\end{document}
"""

_MISSING_FILE = re.compile(r"File [`']([^`']+)' not found")
_MISSING_TFM = re.compile(r"=\s*([A-Za-z0-9._+-]+) at [^\n]*not loadable: Metric \(TFM\) file not found")
_CANT_FIND = re.compile(r"I can't find file [`']([^`']+)'")
#: "./test.tex:12: LaTeX Error: ..." (what -file-line-error prints instead of "! ").
_FILE_LINE_ERROR = re.compile(r"^\S+\.tex:\d+: (.+)$")
_ARCH_SUFFIX = re.compile(
    r"\.(?:windows|win32|win64|x86_64-linux|aarch64-linux|x86_64-linuxmusl"
    r"|universal-darwin|x86_64-darwinlegacy|amd64-freebsd)$"
)


# ---------------------------------------------------------------------------
# Running tools
# ---------------------------------------------------------------------------


@dataclass
class ToolResult:
    """What a tool run produced; ``error`` is set when it could not finish."""

    returncode: int | None
    stdout: str = ""
    stderr: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.returncode == 0

    @property
    def output(self) -> str:
        return (self.stdout or "") + (self.stderr or "")


def _tool_args(path: str, *args: str) -> list[str]:
    """Batch files (TinyTeX's tlmgr.bat) run through cmd.exe, never a shell string."""
    if os.name == "nt" and path.lower().endswith((".bat", ".cmd")):
        return ["cmd.exe", "/d", "/c", path, *args]
    return [path, *args]


def _tool_env() -> dict[str, str]:
    """This process's environment without API keys and other secrets.

    TeX, R and the TinyTeX installer never need a key, so none is handed
    to them.
    """
    return {
        k: v for k, v in os.environ.items()
        if not any(h in k.upper() for h in ("API_KEY", "TOKEN", "SECRET", "PASSWORD"))
    }


def _run(
    args: Sequence[str],
    *,
    timeout: float,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    new_session: bool = False,
) -> ToolResult:
    from edmars import proc

    try:
        extra: dict[str, Any] = {"new_session": True} if new_session else {}
        done = proc.run(list(args), timeout=timeout,
                        cwd=str(cwd) if cwd is not None else None,
                        env=dict(env) if env is not None else _tool_env(), **extra)
    except Exception as exc:  # noqa: BLE001 - classify below, re-raise the rest
        if type(exc).__name__ == "TimeoutExpired":
            return ToolResult(None, _as_text(getattr(exc, "stdout", "")),
                              _as_text(getattr(exc, "stderr", "")),
                              f"timed out after {timeout:.0f} s")
        if isinstance(exc, OSError):
            return ToolResult(None, "", "", f"could not start {Path(args[0]).name}: {exc}")
        raise
    return ToolResult(done.returncode, done.stdout or "", done.stderr or "")


def _as_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value or ""


def _is_file(path: str | Path | None) -> bool:
    try:
        return bool(path) and Path(str(path)).is_file()
    except OSError:
        return False


def _settings_get(settings: Mapping[str, Any] | None, dotted: str) -> Any:
    node: Any = settings or {}
    for part in dotted.split("."):
        if not isinstance(node, Mapping):
            return None
        node = node.get(part)
    return node


def _settings_set_and_save(settings: dict[str, Any], values: Mapping[str, Any]) -> None:
    from edmars import settings as settings_mod

    for dotted, value in values.items():
        settings_mod.set_(settings, dotted, value)
    settings_mod.save(settings)


# ---------------------------------------------------------------------------
# LaTeX discovery
# ---------------------------------------------------------------------------


def tinytex_root() -> Path:
    """Where the official TinyTeX installer puts TinyTeX on this OS."""
    if os.name == "nt":
        appdata = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(appdata) / "TinyTeX"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "TinyTeX"
    return Path.home() / ".TinyTeX"


def tinytex_bin_dirs() -> list[Path]:
    """TinyTeX's platform bin folders (``bin/windows``, ``bin/x86_64-linux``...)."""
    bin_root = tinytex_root() / "bin"
    try:
        return sorted(p for p in bin_root.iterdir() if p.is_dir())
    except OSError:
        return []


def _exe_names(name: str) -> list[str]:
    if os.name == "nt":
        return [name + ".exe", name + ".bat", name]
    return [name]


def find_tex_tool(name: str, settings: Mapping[str, Any] | None = None) -> str | None:
    """Path of a TeX program: next to the saved pdflatex, on PATH, or in TinyTeX."""
    from edmars import proc

    def in_dirs(folders: Iterable[Path]) -> str | None:
        for folder in folders:
            for exe in _exe_names(name):
                candidate = folder / exe
                if _is_file(candidate):
                    return str(candidate)
        return None

    saved = _settings_get(settings, "latex.pdflatex")
    if saved and _is_file(saved):
        if name == "pdflatex":
            return str(saved)
        # Tools from the same distribution as the saved pdflatex come first,
        # so a TinyTeX pdflatex is never paired with another TeX's biber.
        sibling = in_dirs([Path(str(saved)).parent])
        if sibling:
            return sibling
    return proc.which(name) or in_dirs(tinytex_bin_dirs())


def latex_bin_dir(settings: Mapping[str, Any] | None = None) -> str | None:
    """Folder to put on the run's PATH so the pipeline finds pdflatex."""
    pdflatex = find_tex_tool("pdflatex", settings)
    return str(Path(pdflatex).parent) if pdflatex else None


def _is_miktex(pdflatex: str) -> bool:
    result = _run([pdflatex, "--version"], timeout=30)
    return "miktex" in result.output.lower()


def miktex_autoinstall(settings: Mapping[str, Any] | None = None) -> str | None:
    """MiKTeX's ``[MPM]AutoInstall`` value ("1" = always), or None if unknown."""
    initexmf = find_tex_tool("initexmf", settings)
    if not initexmf:
        return None
    result = _run([initexmf, "--show-config-value=[MPM]AutoInstall"], timeout=30)
    if not result.ok:
        return None
    value = result.stdout.strip().splitlines()
    return value[-1].strip() if value else ""


def set_miktex_autoinstall(settings: Mapping[str, Any] | None = None) -> Check:
    """Tell MiKTeX to install missing packages without asking (user setting)."""
    title = "MiKTeX automatic package install"
    initexmf = find_tex_tool("initexmf", settings)
    if not initexmf:
        return Check(title, "fail", "MiKTeX's initexmf program was not found.",
                     fix="Open MiKTeX Console > Settings and choose 'Always install missing packages'.")
    result = _run([initexmf, "--set-config-value=[MPM]AutoInstall=1"], timeout=60)
    if result.error or result.returncode != 0:
        return Check(title, "fail",
                     f"MiKTeX refused the change: {_tail(result.output or result.error or '')}",
                     fix="Open MiKTeX Console > Settings and choose 'Always install missing packages'.")
    if miktex_autoinstall(settings) == "1":
        return Check(title, "ok", "MiKTeX will now install missing LaTeX packages automatically.")
    return Check(title, "warn", "The setting was sent, but MiKTeX still reports a different value.",
                 fix="Open MiKTeX Console > Settings and choose 'Always install missing packages'.")


def _tail(text: str, lines: int = 3) -> str:
    kept = [ln.strip() for ln in (text or "").strip().splitlines() if ln.strip()]
    return " / ".join(kept[-lines:])[:400]


def latex_checks(settings: Mapping[str, Any] | None = None) -> list[Check]:
    """What LaTeX is installed and whether the templates' pieces are there."""
    pdflatex = find_tex_tool("pdflatex", settings)
    if not pdflatex:
        return [Check(
            "PDF maker (LaTeX)", "fail",
            "No LaTeX installation was found. Studies still finish, but without a PDF.",
            fix="edmars setup pdf",
        )]
    version = _run([pdflatex, "--version"], timeout=30)
    first = (version.output.strip().splitlines() or ["pdflatex"])[0].strip()
    miktex = "miktex" in version.output.lower()
    checks = [Check("PDF maker (LaTeX)", "ok", f"{first} ({pdflatex})")]

    for tool, title, why in (
        ("bibtex", "Reference tool for conference papers (BibTeX)",
         "Conference-format papers cannot build their reference list."),
        ("biber", "Reference tool for journal papers (Biber)",
         "Journal-format (APA) papers cannot build their reference list."),
    ):
        path = find_tex_tool(tool, settings)
        if path:
            checks.append(Check(title, "ok", path))
        else:
            # Conference format is the default for every study; journal
            # format is opt-in, so a missing Biber is only a warning.
            checks.append(Check(title, "fail" if tool == "bibtex" else "warn",
                                f"Not found. {why}", fix="edmars setup pdf"))

    auto = miktex_autoinstall(settings) if miktex else None
    kpsewhich = find_tex_tool("kpsewhich", settings)
    for filename, title, package in LATEX_STYLE_FILES:
        found = ""
        if kpsewhich:
            result = _run([kpsewhich, filename], timeout=30)
            found = result.stdout.strip() if result.ok else ""
        if found:
            checks.append(Check(title, "ok", found.splitlines()[0]))
        elif miktex and auto == "1":
            checks.append(Check(
                title, "info",
                f"Not installed yet; MiKTeX installs '{package}' automatically the "
                "first time a paper needs it (needs internet at that moment).",
            ))
        else:
            checks.append(Check(
                title, "warn", f"The LaTeX package '{package}' is not installed.",
                fix=("Install it in MiKTeX Console > Packages, or run edmars setup pdf"
                     if miktex else f"tlmgr install {package}  (or: edmars setup pdf)"),
            ))

    if miktex:
        if auto == "1":
            checks.append(Check("MiKTeX automatic package install", "ok",
                                "MiKTeX installs missing LaTeX packages automatically."))
        else:
            checks.append(Check(
                "MiKTeX automatic package install", "warn",
                "MiKTeX is set to ask (or never) before installing a missing LaTeX "
                "package. A study makes its PDF in the background, where that "
                "question cannot be answered, so the PDF step would wait until it "
                "times out.",
                fix="edmars setup pdf  (or run: initexmf --set-config-value=[MPM]AutoInstall=1)",
            ))
    return checks


# ---------------------------------------------------------------------------
# Test compile
# ---------------------------------------------------------------------------


def _first_error(log_text: str) -> str:
    """The first TeX error line ("! ..."), else a BibTeX/Biber error line."""
    lines = log_text.splitlines()
    for line in lines:
        if line.startswith("! "):
            return line[2:].strip()
        match = _FILE_LINE_ERROR.match(line)
        if match:
            return match.group(1).strip()
    for line in lines:
        if line.startswith(("ERROR - ", "I couldn't open", "I found no")):
            return line.strip()
    return ""


def _missing_files(text: str) -> list[str]:
    names: list[str] = []
    for pattern, suffix in ((_MISSING_FILE, ""), (_CANT_FIND, ""), (_MISSING_TFM, ".tfm")):
        for match in pattern.finditer(text):
            name = match.group(1).strip() + suffix
            if name and name not in names:
                names.append(name)
    return names


@dataclass
class _CompileOutcome:
    check: Check
    missing_files: list[str]


def _compile_doc(
    folder: Path,
    *,
    title: str,
    tex: str,
    bib_tool: str,
    bib_path: str | None,
    pdflatex: str,
    extra_flags: Sequence[str],
    timeout_s: float,
    miktex_blocked: bool,
) -> _CompileOutcome:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "test.tex").write_text(tex, encoding="utf-8")
    (folder / "refs.bib").write_text(TEST_BIB, encoding="utf-8")
    if not bib_path:
        return _CompileOutcome(Check(
            title, "fail", f"{bib_tool} was not found, so the reference list cannot be built.",
            fix="edmars setup pdf"), [])

    latex_cmd = [pdflatex, *extra_flags, "-interaction=nonstopmode", "-halt-on-error",
                 "test.tex"]
    steps: list[tuple[str, list[str]]] = [
        ("pdflatex", latex_cmd),
        (bib_tool, _tool_args(bib_path, "test")),
        ("pdflatex", latex_cmd),
        ("pdflatex", latex_cmd),
    ]
    started = time.monotonic()
    deadline = started + timeout_s
    bib_problem = ""
    transcript = ""
    for step_name, args in steps:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return _CompileOutcome(_timeout_check(title, timeout_s, miktex_blocked), [])
        result = _run(args, timeout=remaining, cwd=folder)
        transcript += result.output
        if result.error:
            if result.error.startswith("timed out"):
                return _CompileOutcome(_timeout_check(title, timeout_s, miktex_blocked), [])
            return _CompileOutcome(Check(title, "fail", f"{step_name}: {result.error}",
                                         fix="edmars setup pdf"), [])
        if step_name == "pdflatex" and result.returncode != 0:
            break
        # BibTeX exits 1 for warnings only (like the pipeline, accept it).
        if step_name != "pdflatex" and result.returncode not in ((0, 1) if step_name == "bibtex" else (0,)):
            bib_problem = _first_error(result.output) or f"{step_name} exited with code {result.returncode}"
    log_text = _read(folder / "test.log")
    pdf = folder / "test.pdf"
    elapsed = time.monotonic() - started
    missing = _missing_files(log_text + "\n" + transcript)
    if pdf.is_file() and pdf.stat().st_size > 0:
        if bib_problem:
            return _CompileOutcome(Check(
                title, "warn",
                f"A PDF was made, but the reference list failed: {bib_problem}",
                fix="edmars setup pdf"), missing)
        return _CompileOutcome(Check(title, "ok", f"A test PDF was made in {elapsed:.0f} s."), missing)
    error = _first_error(log_text) or _first_error(transcript) or "pdflatex made no PDF"
    hint = ""
    if missing and miktex_blocked:
        hint = (" MiKTeX would install it, but it is set to ask first; turn on "
                "automatic package install (edmars setup pdf).")
    return _CompileOutcome(Check(title, "fail", f"No PDF: {error}.{hint}",
                                 fix="edmars setup pdf"), missing)


def _timeout_check(title: str, timeout_s: float, miktex_blocked: bool) -> Check:
    why = (" MiKTeX may be waiting for someone to confirm a package install in a "
           "window." if miktex_blocked else " A first-time package install can be slow; "
           "try again.")
    return Check(title, "fail", f"Stopped after {timeout_s:.0f} seconds without a PDF.{why}",
                 fix="edmars setup pdf")


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _test_compile_detailed(
    timeout_s: float, settings: Mapping[str, Any] | None
) -> list[_CompileOutcome]:
    pdflatex = find_tex_tool("pdflatex", settings)
    titles = ("PDF test: conference paper (acmart + BibTeX)",
              "PDF test: journal paper (apa7 + Biber)")
    if not pdflatex:
        return [_CompileOutcome(Check(t, "fail", "No LaTeX installation was found.",
                                      fix="edmars setup pdf"), []) for t in titles]
    miktex = _is_miktex(pdflatex)
    blocked = miktex and miktex_autoinstall(settings) != "1"
    extra = ["--disable-installer"] if blocked else []
    outcomes: list[_CompileOutcome] = []
    with tempfile.TemporaryDirectory(prefix="edmars-pdfcheck-") as tmp:
        for sub, title, tex, tool in (
            ("acm", titles[0], ACM_TEST_TEX, "bibtex"),
            ("apa", titles[1], APA_TEST_TEX, "biber"),
        ):
            outcomes.append(_compile_doc(
                Path(tmp) / sub, title=title, tex=tex, bib_tool=tool,
                bib_path=find_tex_tool(tool, settings), pdflatex=pdflatex,
                extra_flags=extra, timeout_s=timeout_s, miktex_blocked=blocked,
            ))
    return outcomes


def test_compile(
    timeout_s: float = 120, settings: Mapping[str, Any] | None = None
) -> list[Check]:
    """Compile the two test documents; one Check per template."""
    return [o.check for o in _test_compile_detailed(timeout_s, settings)]


# Not a pytest test, despite the name.
test_compile.__test__ = False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# TinyTeX
# ---------------------------------------------------------------------------


def tinytex_packages(app_root: Path | None = None) -> list[str]:
    """TeX Live packages to install: the base list plus every template package."""
    names: list[str] = list(TINYTEX_BASE_PACKAGES)
    if app_root is None:
        try:
            from edmars import paths

            app_root = Path(paths.app_root())
        except Exception:  # noqa: BLE001 - base list is still useful
            app_root = None
    pattern = re.compile(r"\\(?:usepackage|RequirePackage)(?:\[[^\]]*\])?\{([^}]*)\}")
    if app_root is not None:
        for tex in sorted((app_root / "templates").glob("*.tex")):
            for line in _read(tex).splitlines():
                code = re.split(r"(?<!\\)%", line, maxsplit=1)[0]
                for match in pattern.finditer(code):
                    for raw in match.group(1).split(","):
                        latex_name = raw.strip()
                        if not latex_name:
                            continue
                        tl_name = _TL_PACKAGE_FOR.get(latex_name, latex_name)
                        if tl_name and tl_name not in names:
                            names.append(tl_name)
    return names


def find_tlmgr() -> str | None:
    """TinyTeX's tlmgr (``tlmgr.bat`` on Windows)."""
    for folder in tinytex_bin_dirs():
        for exe in _exe_names("tlmgr"):
            candidate = folder / exe
            if _is_file(candidate):
                return str(candidate)
    return None


def _tinytex_pdflatex() -> str | None:
    for folder in tinytex_bin_dirs():
        for exe in _exe_names("pdflatex"):
            if _is_file(folder / exe):
                return str(folder / exe)
    return None


def _tlmgr_install(tlmgr: str, packages: Iterable[str], timeout: float) -> list[str]:
    """Install packages; returns the ones that failed (tries one by one on error)."""
    wanted = [p for p in dict.fromkeys(packages) if p]
    if not wanted:
        return []
    result = _run(_tool_args(tlmgr, "install", *wanted), timeout=timeout)
    if result.ok:
        return []
    failed: list[str] = []
    for package in wanted:
        single = _run(_tool_args(tlmgr, "install", package), timeout=timeout)
        if not single.ok and "already present" not in single.output:
            failed.append(package)
    return failed


def _package_for_file(tlmgr: str, filename: str) -> str | None:
    result = _run(_tool_args(tlmgr, "search", "--global", "--file", "/" + filename),
                  timeout=120)
    current: str | None = None
    for line in result.output.splitlines():
        if line.startswith("tlmgr"):
            continue
        header = re.match(r"^([A-Za-z0-9._+\-]+):\s*$", line)
        if header:
            current = header.group(1)
            continue
        if current and line[:1].isspace() and line.strip().endswith("/" + filename):
            # Binary packages are listed per platform ("biber.windows").
            return _ARCH_SUFFIX.sub("", current)
    return None


def tinytex_installer_command(installer: Path, keep_dir: Path, *,
                              platform: str | None = None) -> list[str]:
    """How to run the official TinyTeX installer on this OS.

    On macOS, install-bin-unix.sh ends by putting TinyTeX on PATH: links
    in /usr/local/bin when that folder is writable, otherwise a
    /etc/paths.d/TinyTeX file written through ``sudo``, which asks for an
    administrator password. edmars captures the installer's output, so
    the user saw a bare "Password:" and nothing else. The installer's
    documented ``--no-path`` skips that step; edmars does not need it,
    because it finds TinyTeX in ~/Library/TinyTeX itself and puts its bin
    folder on every study's PATH (``latex_bin_dir``), as on Windows.

    The script also takes its FIRST argument, if any, as a folder to move
    the downloaded bundle into (``mv "$INSTALLER_FILE" "$1/"``), so
    ``--no-path`` on its own would become that folder, and the move, and
    with it the installer, would fail. A scratch folder goes first. On
    Linux the PATH step only adds links in ~/.local/bin (or ~/bin), with
    no administrator rights, so it runs as before.
    """
    platform = platform or sys.platform
    if platform.startswith("win"):
        return ["cmd.exe", "/d", "/c", str(installer)]
    if platform == "darwin":
        return ["sh", str(installer), str(keep_dir), "--no-path"]
    return ["sh", str(installer)]


def install_tinytex(
    *,
    settings: dict[str, Any] | None = None,
    session: Any | None = None,
    timeout_s: float = 1800,
    max_rounds: int = 30,
    on_step: Callable[[str], None] | None = None,
) -> Check:
    """Install TinyTeX and every LaTeX package the paper templates need.

    Call only after the user agreed: this downloads the official installer
    from yihui.org, which downloads TinyTeX (about 100 MB) and packages
    from CTAN mirrors. When ``settings`` is given, ``latex.mode`` and
    ``latex.pdflatex`` are saved.
    """
    title = "TinyTeX"
    path_note = ""

    def step(text: str) -> None:
        if on_step is not None:
            try:
                on_step(text)
            except Exception:  # noqa: BLE001 - a status line is not fatal
                pass

    tlmgr = find_tlmgr()
    if not tlmgr:
        key = "windows" if os.name == "nt" else "unix"
        url = TINYTEX_INSTALLER_URLS[key]
        with tempfile.TemporaryDirectory(prefix="edmars-tinytex-") as tmp:
            installer = Path(tmp) / Path(url).name
            step("Downloading the TinyTeX installer")
            try:
                fetch.download_file(url, installer, session=session, max_bytes=5_000_000)
            except fetch.DownloadError as exc:
                return Check(title, "fail", str(exc), fix="edmars setup pdf")
            step("Installing TinyTeX (this can take several minutes)")
            args = tinytex_installer_command(installer, Path(tmp))
            # No terminal for it either: should any installer version still
            # ask for a password, it fails at once instead of waiting unseen.
            result = _run(args, timeout=timeout_s, cwd=tmp, new_session=True)
            if result.error or result.returncode != 0:
                return Check(title, "fail",
                             f"The TinyTeX installer failed: {result.error or _tail(result.output)}",
                             fix="edmars setup pdf")
        tlmgr = find_tlmgr()
        if not tlmgr:
            return Check(title, "fail",
                         f"The installer finished, but TinyTeX was not found in {tinytex_root()}.",
                         fix="edmars setup pdf")
        if "--no-path" in args:
            path_note = (f" It was not added to your PATH (that needs an administrator password on "
                         f"a Mac); EDM-ARS finds it in {Path(tlmgr).parent}.")

    step("Installing the LaTeX packages the paper templates use")
    failed = _tlmgr_install(tlmgr, tinytex_packages(), timeout=timeout_s)

    pdflatex = _tinytex_pdflatex()
    local = {"latex": {"mode": "tinytex", "pdflatex": pdflatex}}
    outcomes: list[_CompileOutcome] = []
    tried: set[str] = set()
    for _ in range(max_rounds):
        step("Checking that a test paper compiles")
        outcomes = _test_compile_detailed(120, local)
        if all(o.check.status == "ok" for o in outcomes):
            break
        wanted: list[str] = []
        for outcome in outcomes:
            for filename in outcome.missing_files:
                if filename in tried:
                    continue
                tried.add(filename)
                package = _package_for_file(tlmgr, filename)
                if package and package not in wanted:
                    wanted.append(package)
        if not wanted:
            break
        step("Installing " + ", ".join(wanted))
        failed.extend(_tlmgr_install(tlmgr, wanted, timeout=timeout_s))

    if settings is not None and pdflatex:
        _settings_set_and_save(settings, {"latex.mode": "tinytex", "latex.pdflatex": pdflatex})
    bad = [o.check for o in outcomes if o.check.status != "ok"]
    if not bad:
        note = f" ({len(failed)} optional packages could not be installed)" if failed else ""
        return Check(title, "ok", f"TinyTeX is installed and both test papers compile{note}.{path_note}")
    return Check(title, "warn",
                 "TinyTeX is installed, but a test paper still fails: "
                 + "; ".join(f"{c.name}: {c.detail}" for c in bad) + path_note,
                 fix="edmars doctor --deep")


# ---------------------------------------------------------------------------
# R
# ---------------------------------------------------------------------------


def _version_key(path: str) -> tuple[int, int, int]:
    for part in reversed(Path(path).parts):
        match = _R_VERSION_DIR.fullmatch(part)
        if match:
            return (int(match.group(1)), int(match.group(2)), int(match.group(3) or 0))
    return (0, 0, 0)


def _registry_rscripts() -> list[str]:  # pragma: no cover - Windows registry
    try:
        import winreg  # type: ignore[import-not-found]
    except ImportError:
        return []
    found: list[str] = []
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        try:
            with winreg.OpenKey(hive, r"SOFTWARE\R-core\R") as key:
                install_path, _ = winreg.QueryValueEx(key, "InstallPath")
        except OSError:
            continue
        found.append(str(Path(install_path) / "bin" / "Rscript.exe"))
    return found


def rscript_candidates(
    settings: Mapping[str, Any] | None = None,
    *,
    platform: str | None = None,
    environ: Mapping[str, str] | None = None,
    which: Callable[[str], str | None] | None = None,
    use_registry: bool = True,
) -> list[str]:
    """Every place Rscript may be, in the order they are tried (unchecked)."""
    from edmars import proc

    platform = platform or sys.platform
    env = os.environ if environ is None else environ
    which = which or proc.which
    out: list[str] = []

    def add(value: str | None) -> None:
        if value and value not in out:
            out.append(value)

    add(_settings_get(settings, "r.rscript"))
    add(env.get("EDM_ARS_RSCRIPT"))
    add(which("Rscript"))
    if platform.startswith("win"):
        roots: list[Path] = []
        for var in ("ProgramW6432", "ProgramFiles", "ProgramFiles(x86)"):
            if env.get(var):
                roots.append(Path(env[var]) / "R")
        if env.get("LOCALAPPDATA"):
            roots.append(Path(env["LOCALAPPDATA"]) / "Programs" / "R")
        installs: list[str] = []
        for root in roots:
            try:
                installs.extend(str(p / "bin" / "Rscript.exe") for p in root.glob("R-*"))
            except OSError:
                continue
        for candidate in sorted(dict.fromkeys(installs), key=_version_key, reverse=True):
            add(candidate)
        if use_registry:
            for candidate in _registry_rscripts():
                add(candidate)
    elif platform == "darwin":
        for candidate in ("/Library/Frameworks/R.framework/Resources/bin/Rscript",
                          "/opt/homebrew/bin/Rscript", "/usr/local/bin/Rscript"):
            add(candidate)
    else:
        for candidate in ("/usr/bin/Rscript", "/usr/local/bin/Rscript"):
            add(candidate)
    return out


def find_rscript(settings: Mapping[str, Any] | None = None, **kwargs: Any) -> str | None:
    """The Rscript to use, or None. A saved path wins, then ``EDM_ARS_RSCRIPT``.

    Only real files count: a path to R's ``bin`` FOLDER (a common mistake)
    is skipped instead of being handed to the pipeline, where it fails
    with an opaque PermissionError.
    """
    for candidate in rscript_candidates(settings, **kwargs):
        if _is_file(candidate):
            return candidate
    return None


def _r_script_file(folder: Path, name: str, code: str) -> Path:
    path = folder / name
    path.write_text(code, encoding="utf-8")
    return path


def _probe_code(packages: Sequence[str]) -> str:
    pkgs = ", ".join(f"'{p}'" for p in packages)
    return (
        f"pkgs <- c({pkgs})\n"
        "for (p in pkgs) if (!requireNamespace(p, quietly = TRUE)) cat('MISSING', p, '\\n')\n"
        "cat('R_VERSION', as.character(getRversion()), '\\n')\n"
    )


def _install_code(packages: Sequence[str], repo: str) -> str:
    pkgs = ", ".join(f"'{p}'" for p in packages)
    return (
        f"pkgs <- c({pkgs})\n"
        "lib <- strsplit(Sys.getenv('R_LIBS_USER'), .Platform$path.sep, fixed = TRUE)[[1]][1]\n"
        "if (is.na(lib) || !nzchar(lib)) lib <- .libPaths()[1]\n"
        "lib <- path.expand(lib)\n"
        "dir.create(lib, recursive = TRUE, showWarnings = FALSE)\n"
        ".libPaths(c(lib, .libPaths()))\n"
        "options(timeout = max(600, getOption('timeout')))\n"
        f"install.packages(pkgs, lib = lib, repos = c(CRAN = '{repo}'))\n"
        "for (p in pkgs) if (!requireNamespace(p, quietly = TRUE)) cat('MISSING', p, '\\n')\n"
    )


@dataclass
class RProbe:
    """Result of asking an Rscript which packages it has."""

    version: str | None
    missing: list[str]
    error: str | None = None


def probe_r(rscript: str, packages: Sequence[str] = R_PACKAGES, timeout_s: float = 120) -> RProbe:
    """Run the package probe with ``Rscript --vanilla`` (as the pipeline does)."""
    with tempfile.TemporaryDirectory(prefix="edmars-r-") as tmp:
        script = _r_script_file(Path(tmp), "probe.R", _probe_code(packages))
        result = _run([rscript, "--vanilla", str(script)], timeout=timeout_s, cwd=tmp)
    if result.error:
        return RProbe(None, [], result.error)
    missing = re.findall(r"^MISSING\s+(\S+)", result.stdout, flags=re.MULTILINE)
    version_match = re.search(r"^R_VERSION\s+(\S+)", result.stdout, flags=re.MULTILINE)
    if result.returncode != 0 or not version_match:
        return RProbe(None, missing, f"R exited with code {result.returncode}: {_tail(result.output)}")
    return RProbe(version_match.group(1), missing)


def _version_tuple(text: str | None) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", text or "")[:3])


def r_checks(settings: Mapping[str, Any] | None = None) -> list[Check]:
    """Is R installed, recent enough, and does it have the packages?"""
    rscript = find_rscript(settings)
    if not rscript:
        return [Check(
            "R (for measurement studies)", "warn",
            "R was not found. Only psychometrics (measurement) studies need it.",
            fix=f"Install R from {R_DOWNLOAD_PAGE}, then run: edmars setup r",
        )]
    probe = probe_r(rscript)
    if probe.error:
        return [Check("R (for measurement studies)", "fail",
                      f"Found {rscript}, but it did not run: {probe.error}",
                      fix="edmars setup r")]
    checks: list[Check] = []
    version = _version_tuple(probe.version)
    if version and version[:2] < R_MIN_VERSION:
        checks.append(Check(
            "R (for measurement studies)", "warn",
            f"R {probe.version} at {rscript}; EDM-ARS is tested with R "
            f"{R_MIN_VERSION[0]}.{R_MIN_VERSION[1]} or newer.",
            fix=f"Install a newer R from {R_DOWNLOAD_PAGE}, then run: edmars setup r",
        ))
    else:
        checks.append(Check("R (for measurement studies)", "ok", f"R {probe.version} ({rscript})"))
    if probe.missing:
        checks.append(Check(
            "R packages", "fail",
            "Missing: " + ", ".join(probe.missing) + ". Measurement studies would stop "
            "at the analysis step.",
            fix="edmars setup r",
        ))
    else:
        checks.append(Check("R packages", "ok", "All present: " + ", ".join(R_PACKAGES)))
    return checks


def install_r_packages(
    rscript: str,
    packages: Sequence[str] | None = None,
    *,
    repo: str = R_REPO_SNAPSHOT,
    timeout_s: float = 1800,
) -> Check:
    """Install the missing R packages into the user's R library (consent first)."""
    title = "R packages"
    wanted = list(packages) if packages is not None else list(R_PACKAGES)
    before = probe_r(rscript, wanted)
    if before.error:
        return Check(title, "fail", f"R did not run: {before.error}", fix="edmars setup r")
    if not before.missing:
        return Check(title, "ok", "All present: " + ", ".join(wanted))
    with tempfile.TemporaryDirectory(prefix="edmars-r-") as tmp:
        script = _r_script_file(Path(tmp), "install.R", _install_code(before.missing, repo))
        result = _run([rscript, "--vanilla", str(script)], timeout=timeout_s, cwd=tmp)
    after = probe_r(rscript, wanted)
    if not after.error and not after.missing:
        return Check(title, "ok", "Installed: " + ", ".join(before.missing))
    why = result.error or _tail(result.output, 4)
    still = ", ".join(after.missing) if after.missing else "unknown"
    return Check(
        title, "fail",
        f"Still missing after the install attempt: {still}. R said: {why}",
        fix=(f"In R, run install.packages(c({', '.join(repr(p) for p in after.missing)}), "
             f"repos = '{repo}')"),
    )


def remember_rscript(settings: dict[str, Any], rscript: str | None, packages_ok: bool) -> None:
    """Save the chosen Rscript (the runner passes it on as EDM_ARS_RSCRIPT)."""
    _settings_set_and_save(settings, {"r.rscript": rscript, "r.packages_ok": bool(packages_ok)})


# ---------------------------------------------------------------------------
# XGBoost and its OpenMP library
# ---------------------------------------------------------------------------

#: Run in a fresh Python, as a study's analysis step is: import XGBoost and
#: scikit-learn together, start scikit-learn's OpenMP runtime and, on
#: macOS, list the libomp images dyld has loaded. Prints one JSON line.
_XGBOOST_PROBE = """
import json, sys
out = {"ok": False, "error": None, "version": None, "openmp": None}
try:
    import xgboost, sklearn
    out["version"] = xgboost.__version__
    try:
        from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
        _openmp_effective_n_threads()
    except ImportError:
        pass
    out["ok"] = True
except Exception as exc:
    out["error"] = type(exc).__name__ + ": " + str(exc)
if sys.platform == "darwin":
    try:
        import ctypes
        dyld = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        dyld._dyld_image_count.restype = ctypes.c_uint32
        dyld._dyld_get_image_name.restype = ctypes.c_char_p
        dyld._dyld_get_image_name.argtypes = [ctypes.c_uint32]
        names = [dyld._dyld_get_image_name(i) or b"" for i in range(dyld._dyld_image_count())]
        out["openmp"] = [n.decode("utf-8", "replace") for n in names if n.endswith(b"/libomp.dylib")]
    except Exception:
        pass
print(json.dumps(out))
"""

_OPENMP_README = "install/README.md, Troubleshooting: macOS, XGBoost and the OpenMP library"


def _load_error(text: str) -> str:
    """The telling part of an import error ("Library not loaded: ...").

    XGBoost quotes dyld's messages as a Python list, so a line break in
    them arrives as the two characters backslash and n.
    """
    match = re.search(r"Library not loaded: [^\s\\\"']+", text)
    if match:
        return match.group(0)
    first = (text or "").strip().splitlines()
    return first[0][:300] if first else "unknown error"


def xgboost_checks(timeout_s: float = 180) -> list[Check]:
    """Does XGBoost load next to scikit-learn in a fresh Python, as in a study?

    ``find_spec`` (the package check) only sees that XGBoost is installed.
    On macOS its compiled library also needs an OpenMP library at load
    time: Homebrew's, or scikit-learn's copy that the installer links for
    it. And two different OpenMP copies in one process can stop a study
    with "OMP: Error #15", so on macOS the loaded copies are counted too.
    """
    title = "XGBoost"
    result = _run([sys.executable, "-c", _XGBOOST_PROBE], timeout=timeout_s)
    if result.error:
        return [Check(title, "warn", f"Could not check whether XGBoost loads: {result.error}.",
                      fix="edmars doctor")]
    report: dict[str, Any] | None = None
    for line in reversed(result.stdout.strip().splitlines()):
        try:
            loaded = json.loads(line)
        except ValueError:
            continue
        if isinstance(loaded, dict):
            report = loaded
            break
    if report is None or result.returncode != 0:
        return [Check(title, "fail",
                      f"Python stopped while loading XGBoost (exit code {result.returncode}): "
                      f"{_tail(result.output) or 'no output'}",
                      fix="Reinstall EDM-ARS with the installer.")]
    if not report.get("ok"):
        error = str(report.get("error") or "")
        if "libomp" in error:  # macOS: XGBoost found no OpenMP library
            return [Check(
                title, "fail",
                f"XGBoost does not load: it found no OpenMP library ({_load_error(error)}). "
                "Every study that trains XGBoost would stop.",
                fix=("Run the EDM-ARS installer again: on a Mac without Homebrew's libomp it links "
                     f"scikit-learn's OpenMP library for XGBoost ({_OPENMP_README}). "
                     "Or install Homebrew and run: brew install libomp"),
            )]
        return [Check(title, "fail", f"XGBoost does not load: {_load_error(error)}",
                      fix="Reinstall EDM-ARS with the installer.")]
    version = report.get("version") or "?"
    openmp = [str(p) for p in (report.get("openmp") or [])]
    if len(openmp) > 1:
        return [Check(
            title, "warn",
            f"XGBoost {version} loads, but XGBoost and scikit-learn use two different OpenMP "
            f"libraries ({', '.join(openmp)}). A study that runs both at once can stop with "
            "'OMP: Error #15'.",
            fix=f"See {_OPENMP_README}.",
        )]
    detail = f"XGBoost {version} loads together with scikit-learn"
    if openmp:
        detail += f"; one OpenMP library ({openmp[0]})"
    return [Check(title, "ok", detail)]


# ---------------------------------------------------------------------------
# Docker (information only)
# ---------------------------------------------------------------------------


def docker_info() -> Check:
    """Docker is never used; say so, whether or not it is installed."""
    from edmars import proc

    if proc.which("docker"):
        return Check(
            "Docker", "info",
            "Docker is installed, but EDM-ARS does not use it: studies run the "
            "AI-written analysis code directly on this computer (see `edmars disclaimer`).",
        )
    return Check("Docker", "info", "Not installed; EDM-ARS does not need it.")
