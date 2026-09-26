"""LaTeX and R discovery/probing with fake executables (no real TeX or R runs)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Callable

import pytest

from tests.cli.infra_helpers import (  # noqa: F401 - fixtures
    REPO_ROOT,
    completed,
    serve_bytes,
    edmars_home,
    no_network,
)

from edmars import proc, toolchain  # noqa: E402

pytestmark = pytest.mark.usefixtures("edmars_home", "no_network")


def _load_settings() -> dict[str, Any]:
    from edmars import settings

    return settings.load()


def _touch(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# R discovery (defect F1: newest first, PATH before guesses, per-user installs)
# ---------------------------------------------------------------------------


@pytest.fixture
def r_installs(tmp_path: Path) -> dict[str, Any]:
    pf = tmp_path / "Program Files"
    local = tmp_path / "LocalAppData"
    made = {
        "4.4.1": _touch(pf / "R" / "R-4.4.1" / "bin" / "Rscript.exe"),
        "4.5.0": _touch(pf / "R" / "R-4.5.0" / "bin" / "Rscript.exe"),
        "4.10.0": _touch(pf / "R" / "R-4.10.0" / "bin" / "Rscript.exe"),
        "4.6.1": _touch(local / "Programs" / "R" / "R-4.6.1" / "bin" / "Rscript.exe"),
    }
    env = {"ProgramFiles": str(pf), "ProgramW6432": str(pf), "LOCALAPPDATA": str(local)}
    return {"env": env, "paths": made}


def _candidates(settings: Any, env: dict[str, str],
                which: Callable[[str], str | None] = lambda name: None) -> list[str]:
    return toolchain.rscript_candidates(settings, platform="win32", environ=env,
                                        which=which, use_registry=False)


def test_windows_installs_are_tried_newest_first(r_installs: dict[str, Any]) -> None:
    found = _candidates(None, r_installs["env"])
    p = r_installs["paths"]
    assert found == [p["4.10.0"], p["4.6.1"], p["4.5.0"], p["4.4.1"]]
    assert toolchain.find_rscript(None, platform="win32", environ=r_installs["env"],
                                  which=lambda n: None, use_registry=False) == p["4.10.0"]


def test_path_beats_program_files_and_env_beats_path(
    r_installs: dict[str, Any], tmp_path: Path
) -> None:
    on_path = _touch(tmp_path / "bin" / "Rscript.exe")
    env = dict(r_installs["env"])
    assert _candidates(None, env, lambda n: on_path)[0] == on_path
    override = _touch(tmp_path / "custom" / "Rscript.exe")
    env["EDM_ARS_RSCRIPT"] = override
    assert _candidates(None, env, lambda n: on_path)[:2] == [override, on_path]
    saved = _touch(tmp_path / "saved" / "Rscript.exe")
    first = _candidates({"r": {"rscript": saved}}, env, lambda n: on_path)[0]
    assert first == saved


def test_a_folder_or_missing_path_is_skipped_not_returned(
    r_installs: dict[str, Any], tmp_path: Path
) -> None:
    env = dict(r_installs["env"])
    env["EDM_ARS_RSCRIPT"] = str(tmp_path / "Program Files" / "R" / "R-4.4.1" / "bin")
    got = toolchain.find_rscript({"r": {"rscript": str(tmp_path / "gone.exe")}},
                                 platform="win32", environ=env, which=lambda n: None,
                                 use_registry=False)
    assert got == r_installs["paths"]["4.10.0"]


def test_macos_and_linux_candidates() -> None:
    mac = toolchain.rscript_candidates(None, platform="darwin", environ={},
                                       which=lambda n: None)
    assert "/Library/Frameworks/R.framework/Resources/bin/Rscript" in mac
    assert "/opt/homebrew/bin/Rscript" in mac
    linux = toolchain.rscript_candidates(None, platform="linux", environ={},
                                         which=lambda n: "/usr/bin/Rscript")
    assert linux[0] == "/usr/bin/Rscript"


# ---------------------------------------------------------------------------
# R probe / install
# ---------------------------------------------------------------------------


class FakeR:
    """Answers Rscript --vanilla <file>.R the way the probe/install scripts expect."""

    def __init__(self, missing: list[str], version: str = "4.4.1",
                 installable: bool = True, chosen: str | None = None) -> None:
        self.missing = list(missing)
        self.version = version
        self.installable = installable
        self.chosen = chosen
        self.scripts: list[str] = []

    def __call__(self, args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert args[1] == "--vanilla"
        script = Path(args[2]).read_text(encoding="utf-8")
        self.scripts.append(script)
        if "install.packages" in script:
            said = f"EDMARS_REPO {self.chosen} \n" if self.chosen else ""
            if self.installable:
                self.missing = []
                return completed(args, 0, said, "installing *binary* package 'mirt'")
            return completed(args, 1, said, "Warning: unable to access index for repository")
        out = "".join(f"MISSING {p} \n" for p in self.missing)
        return completed(args, 0, out + f"R_VERSION {self.version} \n", "")


def test_probe_uses_vanilla_and_reports_missing_packages(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeR(["lavaan", "CDM"])
    monkeypatch.setattr(proc, "run", fake)
    probe = toolchain.probe_r("Rscript")
    assert probe.version == "4.4.1"
    assert probe.missing == ["lavaan", "CDM"]
    for package in toolchain.R_PACKAGES:
        assert f"'{package}'" in fake.scripts[0]
    assert "jsonlite" in toolchain.R_PACKAGES  # every certified helper loads it


def test_r_checks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(toolchain, "find_rscript", lambda settings=None, **kw: None)
    only = toolchain.r_checks({})
    assert [c.status for c in only] == ["warn"]
    assert "cran.r-project.org" in (only[0].fix or "")

    rscript = _touch(tmp_path / "Rscript.exe")
    monkeypatch.setattr(toolchain, "find_rscript", lambda settings=None, **kw: rscript)
    monkeypatch.setattr(proc, "run", FakeR(["mirt"]))
    checks = toolchain.r_checks({})
    assert [c.status for c in checks] == ["ok", "fail"]
    assert "mirt" in checks[1].detail and checks[1].fix == "edmars setup r"

    monkeypatch.setattr(proc, "run", FakeR([], version="4.3.2"))
    checks = toolchain.r_checks({})
    assert [c.status for c in checks] == ["warn", "ok"]


def test_install_r_packages_installs_only_what_is_missing_from_the_snapshot(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeR(["mirt"])
    monkeypatch.setattr(proc, "run", fake)
    check = toolchain.install_r_packages("Rscript")
    assert check.status == "ok"
    install_script = next(s for s in fake.scripts if "install.packages" in s)
    assert toolchain.R_REPO_SNAPSHOT in install_script
    assert "c('mirt')" in install_script
    assert "dir.create(lib" in install_script  # user library created first


def test_install_r_packages_failure_says_what_is_still_missing(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(proc, "run", FakeR(["lavaan"], installable=False))
    check = toolchain.install_r_packages("Rscript")
    assert check.status == "fail"
    assert "lavaan" in check.detail and "unable to access index" in check.detail


def test_install_tries_older_snapshots_when_the_newest_has_no_binary_for_this_r() -> None:
    # Found on a real R 4.4.1: the 2026-09-01 snapshot has no R 4.4 binary
    # of mirt 1.47 (Deriv 4.3.0 needs R 4.5), so R compiled the source and
    # the install failed. The script must read each snapshot's binary index
    # for this R, newest first, and use the first that has the package AND
    # its dependencies; source-only platforms keep the newest.
    snapshots = toolchain.R_REPO_SNAPSHOTS
    assert len(snapshots) >= 2 and snapshots[0] == toolchain.R_REPO_SNAPSHOT
    dates = [s.rsplit("/", 1)[-1] for s in snapshots]
    assert dates == sorted(dates, reverse=True)
    code = toolchain._install_code(["mirt"], snapshots)
    listed = "repos <- c(" + ", ".join(f"'{s}'" for s in snapshots) + ")"
    assert listed in code
    assert "type = .Platform$pkgType" in code
    assert ".Platform$pkgType != 'source'" in code
    assert "tools::package_dependencies(pkgs, db = db, recursive = TRUE" in code
    assert "'Depends', 'Imports', 'LinkingTo'" in code
    assert "cat('EDMARS_REPO', repo" in code
    assert "install.packages(pkgs, lib = lib, repos = c(CRAN = repo))" in code


def test_the_snapshot_r_chose_is_named_in_the_result_and_the_fix(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    older = toolchain.R_REPO_SNAPSHOTS[1]
    monkeypatch.setattr(proc, "run", FakeR(["mirt"], chosen=older))
    ok = toolchain.install_r_packages("Rscript")
    assert ok.status == "ok" and older in ok.detail

    monkeypatch.setattr(proc, "run", FakeR(["mirt"], installable=False, chosen=older))
    failed = toolchain.install_r_packages("Rscript")
    assert failed.status == "fail"
    assert f"repos = '{older}'" in (failed.fix or "")


def test_a_repo_passed_in_is_the_only_one_tried(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeR(["CDM"])
    monkeypatch.setattr(proc, "run", fake)
    toolchain.install_r_packages("Rscript", repo="https://cran.example.invalid")
    install_script = next(s for s in fake.scripts if "install.packages" in s)
    assert "repos <- c('https://cran.example.invalid')" in install_script
    assert not any(s in install_script for s in toolchain.R_REPO_SNAPSHOTS)


def test_r_that_times_out_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    def slow(args: list[str], **kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired(args, kwargs.get("timeout"))

    monkeypatch.setattr(proc, "run", slow)
    assert "timed out" in (toolchain.probe_r("Rscript").error or "")


def test_remember_rscript_saves() -> None:
    settings_dict = _load_settings()
    toolchain.remember_rscript(settings_dict, "/opt/R/bin/Rscript", True)
    from edmars import settings

    saved = settings.load()
    assert saved["r"]["rscript"] == "/opt/R/bin/Rscript"
    assert saved["r"]["packages_ok"] is True


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------


def _fake_tex(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, miktex: bool,
              autoinstall: str = "1", present: tuple[str, ...] = ("acmart.cls",),
              compile_ok: bool = True, missing_file: str = "acmart.cls",
              timeout: bool = False) -> list[list[str]]:
    tools = {name: _touch(tmp_path / "texbin" / f"{name}.exe")
             for name in ("pdflatex", "bibtex", "biber", "kpsewhich", "initexmf")}
    if not miktex:
        tools.pop("initexmf")
    monkeypatch.setattr(proc, "which", lambda name: tools.get(name))
    monkeypatch.setattr(toolchain, "tinytex_bin_dirs", lambda: [])
    calls: list[list[str]] = []

    def run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(list(args))
        exe = Path(args[0]).stem
        if "--version" in args:
            banner = ("MiKTeX-pdfTeX 4.19 (MiKTeX 24.4)" if miktex
                      else "pdfTeX 3.141592653-2.6-1.40.26 (TeX Live 2024)")
            return completed(args, 0, banner + "\n")
        if exe == "initexmf":
            return completed(args, 0, autoinstall + "\n")
        if exe == "kpsewhich":
            name = args[1]
            return completed(args, 0, f"/tex/{name}\n") if name in present else completed(args, 1)
        cwd = Path(kwargs["cwd"])
        if timeout:
            raise subprocess.TimeoutExpired(args, kwargs.get("timeout"))
        if exe == "pdflatex":
            if compile_ok:
                (cwd / "test.pdf").write_bytes(b"%PDF-1.5 fake")
                (cwd / "test.log").write_text("Output written on test.pdf\n", encoding="utf-8")
                return completed(args, 0)
            (cwd / "test.log").write_text(
                f"! LaTeX Error: File `{missing_file}' not found.\n\nType X to quit\n",
                encoding="utf-8")
            return completed(args, 1)
        return completed(args, 0)

    monkeypatch.setattr(proc, "run", run)
    return calls


def test_latex_checks_on_miktex_that_asks_before_installing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_tex(monkeypatch, tmp_path, miktex=True, autoinstall="2")
    checks = {c.name: c for c in toolchain.latex_checks()}
    assert checks["PDF maker (LaTeX)"].status == "ok"
    assert "MiKTeX" in checks["PDF maker (LaTeX)"].detail
    assert checks["Conference paper template (acmart)"].status == "ok"
    assert checks["APA journal template (apa7)"].status == "warn"
    auto = checks["MiKTeX automatic package install"]
    assert auto.status == "warn"
    assert "AutoInstall=1" in (auto.fix or "")


def test_latex_checks_on_miktex_that_installs_automatically(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_tex(monkeypatch, tmp_path, miktex=True, autoinstall="1")
    checks = {c.name: c for c in toolchain.latex_checks()}
    assert checks["APA journal template (apa7)"].status == "info"
    assert checks["MiKTeX automatic package install"].status == "ok"


def test_latex_checks_without_latex(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(proc, "which", lambda name: None)
    monkeypatch.setattr(toolchain, "tinytex_bin_dirs", lambda: [])
    checks = toolchain.latex_checks()
    assert len(checks) == 1 and checks[0].status == "fail"
    assert checks[0].fix == "edmars setup pdf"
    compiled = toolchain.test_compile()
    assert [c.status for c in compiled] == ["fail", "fail"]


def test_test_compile_runs_both_templates_with_their_bibliography_tools(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _fake_tex(monkeypatch, tmp_path, miktex=False)
    checks = toolchain.test_compile(timeout_s=120)
    assert [c.status for c in checks] == ["ok", "ok"], [c.detail for c in checks]
    tools = [Path(c[0]).stem for c in calls if "--version" not in c]
    assert tools == ["pdflatex", "bibtex", "pdflatex", "pdflatex",
                     "pdflatex", "biber", "pdflatex", "pdflatex"]
    latex_call = next(c for c in calls if Path(c[0]).stem == "pdflatex" and "--version" not in c)
    assert "-interaction=nonstopmode" in latex_call and "-halt-on-error" in latex_call
    assert "--disable-installer" not in latex_call


def test_blocked_miktex_fails_fast_and_says_why(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _fake_tex(monkeypatch, tmp_path, miktex=True, autoinstall="2", compile_ok=False)
    checks = toolchain.test_compile()
    assert [c.status for c in checks] == ["fail", "fail"]
    assert "acmart.cls" in checks[0].detail
    assert "automatic package install" in checks[0].detail
    latex_calls = [c for c in calls if Path(c[0]).stem == "pdflatex" and "--version" not in c]
    assert all("--disable-installer" in c for c in latex_calls)
    # it stopped at the first failing pdflatex of each document
    assert len(latex_calls) == 2


def test_a_hanging_compile_is_stopped_and_explained(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_tex(monkeypatch, tmp_path, miktex=True, autoinstall="1", timeout=True)
    checks = toolchain.test_compile(timeout_s=5)
    assert all(c.status == "fail" and "Stopped after 5 seconds" in c.detail for c in checks)


def test_missing_file_names_are_parsed_from_logs() -> None:
    log = ("! LaTeX Error: File `acmart.cls' not found.\n"
           "! Font \\T1/LinuxLibertineT-TLF/m/n/10=LinLibertineT-tlf-t1 at 10pt not "
           "loadable: Metric (TFM) file not found.\n")
    assert toolchain._missing_files(log) == ["acmart.cls", "LinLibertineT-tlf-t1.tfm"]


def test_tinytex_package_list_comes_from_the_templates() -> None:
    packages = toolchain.tinytex_packages(REPO_ROOT)
    for name in ("acmart", "apa7", "biblatex-apa", "biber", "caption", "algorithmicx",
                 "booktabs", "csquotes", "multirow"):
        assert name in packages
    assert "inputenc" not in packages and "subcaption" not in packages
    assert len(packages) == len(set(packages))


def test_tlmgr_search_output_maps_files_to_packages(monkeypatch: pytest.MonkeyPatch) -> None:
    output = ("tlmgr: package repository https://mirror.example/tlnet (verified)\n"
              "biber.windows:\n\tbin/windows/biber.exe\n"
              "acmart:\n\ttexmf-dist/tex/latex/acmart/acmart.cls\n")
    monkeypatch.setattr(proc, "run", lambda args, **kw: completed(args, 0, output))
    assert toolchain._package_for_file("tlmgr", "acmart.cls") == "acmart"
    assert toolchain._package_for_file("tlmgr", "biber.exe") == "biber"
    assert toolchain._package_for_file("tlmgr", "nothing.sty") is None


def test_set_miktex_autoinstall(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    initexmf = _touch(tmp_path / "initexmf.exe")
    monkeypatch.setattr(proc, "which", lambda name: initexmf if name == "initexmf" else None)
    monkeypatch.setattr(toolchain, "tinytex_bin_dirs", lambda: [])
    state = {"value": "2"}

    def run(args: list[str], **kwargs: Any) -> Any:
        if any(a.startswith("--set-config-value") for a in args):
            state["value"] = "1"
        return completed(args, 0, state["value"] + "\n")

    monkeypatch.setattr(proc, "run", run)
    assert toolchain.miktex_autoinstall() == "2"
    assert toolchain.set_miktex_autoinstall().status == "ok"
    assert toolchain.miktex_autoinstall() == "1"


def test_saved_pdflatex_brings_its_own_siblings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tiny = tmp_path / "TinyTeX" / "bin" / "windows"
    pdflatex = _touch(tiny / "pdflatex.exe")
    biber = _touch(tiny / "biber.exe")
    other = _touch(tmp_path / "miktex" / "biber.exe")
    monkeypatch.setattr(proc, "which", lambda name: other if name == "biber" else None)
    settings = {"latex": {"mode": "tinytex", "pdflatex": pdflatex}}
    assert toolchain.find_tex_tool("pdflatex", settings) == pdflatex
    assert toolchain.find_tex_tool("biber", settings) == biber
    assert toolchain.find_tex_tool("biber", None) == other
    assert toolchain.latex_bin_dir(settings) == str(tiny)


def test_docker_is_information_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(proc, "which", lambda name: "/usr/bin/docker")
    assert toolchain.docker_info().status == "info"
    monkeypatch.setattr(proc, "which", lambda name: None)
    assert toolchain.docker_info().status == "info"


def test_first_error_reads_both_log_styles() -> None:
    classic = "(./test.tex\n! LaTeX Error: File `acmart.cls' not found.\n\nType X to quit\n"
    assert toolchain._first_error(classic) == "LaTeX Error: File `acmart.cls' not found."
    fle = "./test.tex:3: Undefined control sequence.\n" + r"l.3 \foo" + "\n"
    assert toolchain._first_error(fle) == "Undefined control sequence."
    assert toolchain._first_error("ERROR - Cannot find 'refs.bib'!\n") == "ERROR - Cannot find 'refs.bib'!"


def test_install_tinytex_installs_then_fills_in_whatever_a_test_compile_misses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "TinyTeX"
    monkeypatch.setattr(toolchain, "tinytex_root", lambda: root)
    bin_dir = root / "bin" / "windows"
    tlmgr_name = "tlmgr.bat" if toolchain.os.name == "nt" else "tlmgr"
    pdflatex_name = "pdflatex.exe" if toolchain.os.name == "nt" else "pdflatex"
    installs: list[list[str]] = []
    searches: list[str] = []
    ran_installer: list[list[str]] = []

    def run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if "install-bin" in " ".join(args):
            ran_installer.append(list(args))
            _touch(bin_dir / tlmgr_name)
            _touch(bin_dir / pdflatex_name)
            return completed(args, 0, "TinyTeX installed")
        if "install" in args:
            installs.append([a for a in args[args.index("install") + 1:]])
            return completed(args, 0)
        if "search" in args:
            searches.append(args[-1])
            return completed(args, 0, "totpages:\n\ttexmf-dist/tex/latex/totpages/totpages.sty\n")
        raise AssertionError(args)

    monkeypatch.setattr(proc, "run", run)
    rounds = iter([
        [toolchain._CompileOutcome(toolchain.Check("acm", "fail", "No PDF"), ["totpages.sty"]),
         toolchain._CompileOutcome(toolchain.Check("apa", "ok", "fine"), [])],
        [toolchain._CompileOutcome(toolchain.Check("acm", "ok", "fine"), []),
         toolchain._CompileOutcome(toolchain.Check("apa", "ok", "fine"), [])],
    ])
    seen_settings: list[Any] = []

    def fake_compile(timeout_s: float, settings: Any) -> Any:
        seen_settings.append(settings)
        return next(rounds)

    monkeypatch.setattr(toolchain, "_test_compile_detailed", fake_compile)
    settings = _load_settings()
    url = toolchain.TINYTEX_INSTALLER_URLS["windows" if toolchain.os.name == "nt" else "unix"]
    session = serve_bytes(b"echo installer")

    check = toolchain.install_tinytex(settings=settings, session=session)

    assert check.status == "ok", check.detail
    assert session.calls[0]["url"] == url
    assert len(ran_installer) == 1
    assert "acmart" in installs[0] and "apa7" in installs[0] and "biber" in installs[0]
    assert searches == ["/totpages.sty"] and installs[-1] == ["totpages"]
    assert seen_settings[0]["latex"]["pdflatex"] == str(bin_dir / pdflatex_name)
    from edmars import settings as settings_mod

    saved = settings_mod.load()
    assert saved["latex"]["mode"] == "tinytex"
    assert saved["latex"]["pdflatex"] == str(bin_dir / pdflatex_name)
    assert toolchain._first_error("all fine\n") == ""


def test_tools_never_receive_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "not-for-r")
    seen: list[dict[str, str]] = []

    def run(args: list[str], **kwargs: Any) -> Any:
        seen.append(kwargs["env"])
        return completed(args, 0, "R_VERSION 4.5.1 \n")

    monkeypatch.setattr(proc, "run", run)
    toolchain.probe_r("Rscript")
    assert seen and "DEEPSEEK_API_KEY" not in seen[0] and "PATH" in {k.upper() for k in seen[0]}
