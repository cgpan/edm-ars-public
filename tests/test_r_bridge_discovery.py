"""Finding R, and saying so clearly when it cannot be used.

Released behaviour this replaces:

* F1 -- Windows discovery was three hard-coded paths (R-4.4.1, 4.4.2,
  4.5.0, oldest first) checked BEFORE PATH. Any other release, or a
  per-user install, was only found via PATH -- which the CRAN installer
  does not touch -- and an old listed R beat a newer one on PATH.
* F2 -- the error text pointed at config ``r_bridge.rscript_path``, which
  nothing read.
* An override pointing at R's ``bin`` folder raised a bare
  PermissionError from subprocess; a mistyped override fell through
  silently to some other R.
* The R-gated tests skipped only when Rscript was absent, so a machine
  with R but without lavaan/mirt/CDM failed instead of skipping.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import src.r_bridge as rb
from src.r_bridge import RBridgeError
from src.sandbox import SubprocessExecutor, child_env, create_executor

ROOT = Path(__file__).resolve().parent.parent
EXE = "Rscript.exe" if os.name == "nt" else "Rscript"


def _install(root: Path, version_dir: str) -> Path:
    exe = root / version_dir / "bin" / EXE
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_text("", encoding="utf-8")
    return exe


@pytest.fixture
def no_overrides(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    monkeypatch.delenv("EDM_ARS_RSCRIPT", raising=False)
    return monkeypatch


# --- discovery order ---------------------------------------------------------

def test_newest_version_wins_numerically(tmp_path: Path) -> None:
    for v in ("R-4.4.1", "R-4.5.0", "R-4.10.0", "R-4.6.1", "R-devel"):
        _install(tmp_path, v)
    (tmp_path / "R-4.9.9").mkdir()  # no bin/Rscript: skipped
    found = rb._newest_first([tmp_path], EXE)
    names = [Path(p).parent.parent.name for p in found]
    assert names == ["R-4.10.0", "R-4.6.1", "R-4.5.0", "R-4.4.1", "R-devel"]


def test_a_per_user_install_is_found_and_ties_prefer_the_first_root(tmp_path: Path) -> None:
    system, user = tmp_path / "pf", tmp_path / "local"
    sys_exe = _install(system, "R-4.5.1")
    _install(user, "R-4.5.1")
    user_newer = _install(user, "R-4.6.0")
    found = rb._newest_first([system, user], EXE)
    assert found[0] == str(user_newer)
    assert found[1] == str(sys_exe)


def test_windows_roots_come_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ProgramFiles", str(tmp_path / "pf"))
    monkeypatch.setenv("ProgramW6432", str(tmp_path / "pf"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "la"))
    roots = rb._windows_r_roots()
    assert roots == [tmp_path / "pf" / "R", tmp_path / "la" / "Programs" / "R"]


@pytest.mark.skipif(os.name != "nt", reason="Windows install layout")
def test_windows_discovery_finds_an_unlisted_release(
    tmp_path: Path, no_overrides: pytest.MonkeyPatch
) -> None:
    no_overrides.setenv("ProgramFiles", str(tmp_path / "pf"))
    no_overrides.setenv("ProgramW6432", str(tmp_path / "pf"))
    no_overrides.setenv("LOCALAPPDATA", str(tmp_path / "la"))
    _install(tmp_path / "pf" / "R", "R-4.4.1")
    newest = _install(tmp_path / "la" / "Programs" / "R", "R-4.6.1")
    no_overrides.setattr(rb, "which", lambda name: None)
    assert rb.find_rscript() == str(newest)


@pytest.mark.skipif(os.name == "nt", reason="macOS/Linux install layout")
def test_posix_discovery_checks_framework_and_homebrew(
    tmp_path: Path, no_overrides: pytest.MonkeyPatch
) -> None:
    framework = tmp_path / "framework" / "Rscript"
    brew = tmp_path / "brew" / "Rscript"
    brew.parent.mkdir()
    brew.write_text("", encoding="utf-8")
    no_overrides.setattr(rb, "_POSIX_RSCRIPT_PATHS", (str(framework), str(brew)))
    no_overrides.setattr(rb, "which", lambda name: None)
    assert rb.find_rscript() == str(brew)


def test_path_beats_the_install_folders(no_overrides: pytest.MonkeyPatch) -> None:
    no_overrides.setattr(rb, "which", lambda name: "/on/path/Rscript")
    no_overrides.setattr(rb, "_discovered_installs", lambda: ["/old/R-4.4.1/Rscript"])
    assert rb.find_rscript() == "/on/path/Rscript"


def test_install_folders_are_used_when_path_has_none(no_overrides: pytest.MonkeyPatch) -> None:
    no_overrides.setattr(rb, "which", lambda name: None)
    no_overrides.setattr(rb, "_discovered_installs", lambda: ["/r/R-4.6.1/Rscript", "/r/R-4.4.1/Rscript"])
    assert rb.find_rscript() == "/r/R-4.6.1/Rscript"


def test_env_var_beats_path_and_explicit_beats_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_exe = _install(tmp_path / "env", "R-4.4.1")
    arg_exe = _install(tmp_path / "arg", "R-4.4.1")
    monkeypatch.setenv("EDM_ARS_RSCRIPT", str(env_exe))
    monkeypatch.setattr(rb, "which", lambda name: "/on/path/Rscript")
    assert rb.find_rscript() == str(env_exe)
    assert rb.find_rscript(str(arg_exe)) == str(arg_exe)


def test_nothing_found_names_both_ways_to_fix_it(no_overrides: pytest.MonkeyPatch) -> None:
    no_overrides.setattr(rb, "which", lambda name: None)
    no_overrides.setattr(rb, "_discovered_installs", lambda: [])
    with pytest.raises(RBridgeError) as err:
        rb.find_rscript()
    assert "EDM_ARS_RSCRIPT" in str(err.value)
    assert "r_bridge.rscript_path" in str(err.value)


# --- overrides ---------------------------------------------------------------

def test_an_override_pointing_at_bin_is_resolved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    exe = _install(tmp_path, "R-4.6.1")
    monkeypatch.setenv("EDM_ARS_RSCRIPT", str(exe.parent))
    assert rb.find_rscript() == str(exe)


def test_an_override_pointing_at_the_r_home_is_resolved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    exe = _install(tmp_path, "R-4.6.1")
    monkeypatch.setenv("EDM_ARS_RSCRIPT", str(exe.parent.parent))
    assert rb.find_rscript() == str(exe)


def test_a_quoted_override_is_accepted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    exe = _install(tmp_path, "R-4.6.1")
    monkeypatch.setenv("EDM_ARS_RSCRIPT", f'  "{exe}"  ')
    assert rb.find_rscript() == str(exe)


@pytest.mark.skipif(os.name != "nt", reason="Windows executable suffix")
def test_a_windows_override_without_exe_is_accepted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    exe = _install(tmp_path, "R-4.6.1")
    monkeypatch.setenv("EDM_ARS_RSCRIPT", str(exe.with_suffix("")))
    assert rb.find_rscript() == str(exe)


def test_a_mistyped_override_raises_instead_of_falling_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad = str(tmp_path / "R-4.6.1" / "bni" / EXE)
    monkeypatch.setenv("EDM_ARS_RSCRIPT", bad)
    # A working R exists elsewhere; it must NOT be used silently.
    monkeypatch.setattr(rb, "which", lambda name: "/on/path/Rscript")
    monkeypatch.setattr(rb, "_discovered_installs", lambda: ["/r/R-4.4.1/Rscript"])
    with pytest.raises(RBridgeError) as err:
        rb.find_rscript()
    assert "EDM_ARS_RSCRIPT" in str(err.value)
    assert "bni" in str(err.value)


def test_a_folder_without_rscript_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EDM_ARS_RSCRIPT", str(tmp_path))
    with pytest.raises(RBridgeError, match="folder"):
        rb.find_rscript()


def test_a_bad_explicit_path_raises_too(tmp_path: Path) -> None:
    with pytest.raises(RBridgeError, match="rscript_path"):
        rb.find_rscript(str(tmp_path / "missing" / EXE))


def test_an_empty_env_var_is_treated_as_unset(no_overrides: pytest.MonkeyPatch) -> None:
    no_overrides.setenv("EDM_ARS_RSCRIPT", "   ")
    no_overrides.setattr(rb, "which", lambda name: "/on/path/Rscript")
    assert rb.find_rscript() == "/on/path/Rscript"


# --- run_r_script failure messages --------------------------------------------

def test_an_unstartable_rscript_is_an_rbridgeerror(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rb, "find_rscript", lambda explicit=None: "some/Rscript")

    def _denied(argv: list[str], timeout_s: int) -> None:
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(rb, "_run_rscript", _denied)
    with pytest.raises(RBridgeError, match="Could not start Rscript"):
        rb.run_r_script("cfa_fit.R", {})


def _completed(rc: int, stdout: str = "", stderr: str = "") -> Any:
    def _run(argv: list[str], timeout_s: int) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, rc, stdout=stdout, stderr=stderr)
    return _run


def test_a_missing_package_comes_with_the_install_line(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rb, "find_rscript", lambda explicit=None: "some/Rscript")
    monkeypatch.setattr(
        rb, "_run_rscript",
        _completed(1, stderr="Error in loadNamespace(x) : there is no package called 'jsonlite'\nExecution halted\n"),
    )
    with pytest.raises(RBridgeError) as err:
        rb.run_r_script("cfa_fit.R", {})
    msg = str(err.value)
    assert "exited 1" in msg
    assert 'install.packages("jsonlite")' in msg
    assert "some/Rscript" in msg


def test_the_hint_reads_typographic_quotes_too() -> None:
    hint = rb._missing_package_hint("there is no package called \u2018lavaan\u2019", "R")
    assert 'install.packages("lavaan")' in hint
    assert rb._missing_package_hint("some other failure", "R") == ""


def test_run_r_script_decodes_utf8(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def _fake(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen.update(kwargs)
        return subprocess.CompletedProcess(argv, 1, stdout=None, stderr=None)

    monkeypatch.setattr(rb, "find_rscript", lambda explicit=None: "some/Rscript")
    monkeypatch.setattr("src.r_bridge.subprocess.run", _fake)
    with pytest.raises(RBridgeError, match="exited 1"):
        rb.run_r_script("cfa_fit.R", {})
    assert seen["encoding"] == "utf-8"
    assert seen["errors"] == "replace"


# --- package check -------------------------------------------------------------

def test_missing_packages_are_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rb, "find_rscript", lambda explicit=None: "some/Rscript")
    monkeypatch.setattr(
        rb, "_run_rscript",
        _completed(0, stdout="MISSING mirt\nMISSING CDM\nEDM_ARS_R_CHECK_DONE\n"),
    )
    assert rb.missing_r_packages() == ["mirt", "CDM"]


def test_an_incomplete_check_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rb, "find_rscript", lambda explicit=None: "some/Rscript")
    monkeypatch.setattr(rb, "_run_rscript", _completed(0, stdout="MISSING mirt\n"))
    with pytest.raises(RBridgeError, match="did not complete"):
        rb.missing_r_packages()


def test_package_names_are_validated_before_anything_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*a: Any, **k: Any) -> None:
        raise AssertionError("must not run")

    monkeypatch.setattr(rb, "_run_rscript", _boom)
    with pytest.raises(ValueError):
        rb.missing_r_packages(["lavaan'); system('x"])


def test_the_required_set_covers_what_the_scripts_load() -> None:
    used: set[str] = set()
    for script in (ROOT / "r_helpers").glob("*.R"):
        # Drop comments: one cites "psychometric_gates.py::dif_gate".
        text = "\n".join(
            line.split("#", 1)[0]
            for line in script.read_text(encoding="utf-8").splitlines()
        )
        used |= set(re.findall(r"\blibrary\(([A-Za-z][A-Za-z0-9.]*)\)", text))
        used |= set(re.findall(r"\b([A-Za-z][A-Za-z0-9.]*)::", text))
    assert used <= set(rb.REQUIRED_R_PACKAGES)


def _real_r() -> str | None:
    try:
        return rb.find_rscript()
    except RBridgeError:
        return None


@pytest.mark.skipif(_real_r() is None, reason="Rscript not available")
def test_the_real_check_reports_only_what_is_missing() -> None:
    assert rb.missing_r_packages(["MASS", "definitelyNotAPackage123"]) == [
        "definitelyNotAPackage123"
    ]


# --- F2: config r_bridge.rscript_path reaches generated code --------------------

def test_create_executor_reads_the_config_key() -> None:
    ex = create_executor({"r_bridge": {"rscript_path": "X:/R/bin/Rscript.exe"}})
    assert isinstance(ex, SubprocessExecutor)
    assert ex.rscript_path == "X:/R/bin/Rscript.exe"


@pytest.mark.parametrize("block", [None, {}, {"rscript_path": None}])
def test_an_unset_config_key_exports_nothing(
    block: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("EDM_ARS_RSCRIPT", raising=False)
    ex = create_executor({"r_bridge": block})
    assert "EDM_ARS_RSCRIPT" not in child_env(rscript_path=ex.rscript_path)


def test_the_operators_env_var_wins_over_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EDM_ARS_RSCRIPT", "from-env")
    assert child_env(rscript_path="from-config")["EDM_ARS_RSCRIPT"] == "from-env"
    monkeypatch.setenv("EDM_ARS_RSCRIPT", "")
    assert child_env(rscript_path="from-config")["EDM_ARS_RSCRIPT"] == "from-config"


def test_generated_code_resolves_the_configured_rscript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end: config -> executor -> child env -> find_rscript."""
    monkeypatch.delenv("EDM_ARS_RSCRIPT", raising=False)
    fake = _install(tmp_path / "custom", "R-9.9.9")
    ex = create_executor({"r_bridge": {"rscript_path": str(fake)}})
    out = tmp_path / "out"
    out.mkdir()
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from src.r_bridge import find_rscript\n"
        "print(find_rscript())\n"
    )
    result = ex.run(code, output_dir=str(out), timeout_s=60)
    assert result["returncode"] == 0, result["stderr"]
    assert Path(result["stdout"].strip()) == fake


# --- F3: the docstring names only things that exist ------------------------------

def test_the_module_docstring_names_real_functions() -> None:
    doc = rb.__doc__ or ""
    names = set(re.findall(r"``([A-Za-z_][A-Za-z0-9_]*)\(\)``", doc))
    names |= set(re.findall(r":(?:func|class):`([A-Za-z_][A-Za-z0-9_]*)`", doc))
    assert names, "expected the docstring to reference the bridge API"
    for name in names:
        assert hasattr(rb, name), f"docstring mentions {name}(), which does not exist"


def test_the_skill_states_the_real_resolution_order() -> None:
    skill = (ROOT / "skills/methodology/r-bridge-execution/SKILL.md").read_text(encoding="utf-8")
    env_at = skill.index("EDM_ARS_RSCRIPT")
    assert "r_bridge.rscript_path" in skill
    assert env_at < skill.index("PATH", env_at)


def test_the_module_stays_standard_library_only() -> None:
    """r_bridge.py is copied flat into run output dirs."""
    import ast

    tree = ast.parse((ROOT / "src" / "r_bridge.py").read_text(encoding="utf-8"))
    mods: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods += [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative import"
            mods.append((node.module or "").split(".")[0])
    assert mods
    for mod in mods:
        assert mod in sys.stdlib_module_names or mod == "__future__", mod
