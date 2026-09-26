"""install/install.sh and install/install.ps1, offline.

A real install downloads uv, Python and 0.7-2 GB of packages, so it runs in
CI (.github/workflows/ci.yml, job `installer`), not here. These tests pin
what can be checked without the network: the scripts parse, agree with
each other and with install/README.md, never `exit` a PowerShell session,
refuse what they promise to refuse, and change nothing in a dry run. On
Windows the PowerShell helpers that touch the user's PATH and write the
launcher are run as pure functions, so the logic is tested without
touching the real registry.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALL_SH = REPO_ROOT / "install" / "install.sh"
INSTALL_PS1 = REPO_ROOT / "install" / "install.ps1"
README = REPO_ROOT / "install" / "README.md"

POWERSHELL = shutil.which("powershell") or shutil.which("pwsh")
SH = shutil.which("sh")
ON_WINDOWS = sys.platform == "win32"

GNU_FLAGS = ["--yes", "--no-onboard", "--dir", "--version", "--from-local",
             "--bin-dir", "--no-modify-path", "--dry-run"]
PS_FLAGS = ["-Yes", "-NoOnboard", "-Dir", "-Version", "-FromLocal",
            "-BinDir", "-NoModifyPath", "-DryRun"]


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EDMARS_HOME", str(tmp_path / "edmars-home"))


def _text(path: Path) -> str:
    return path.read_bytes().decode("ascii")  # raises if not pure ASCII


def _clean_env(tmp_path: Path, **extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("EDMARS_")
           and k not in ("PYTHONPATH", "PYTHONHOME", "XDG_DATA_HOME")}
    env["EDMARS_HOME"] = str(tmp_path / "edmars-home")
    env.update(extra)
    return env


# --- static agreement ---------------------------------------------------------------


def test_scripts_are_pure_ascii() -> None:
    # Windows PowerShell 5.1 reads a BOM-less .ps1 in the ANSI code page, and
    # `irm | iex` decodes it however the server says; ASCII survives both.
    _text(INSTALL_SH)
    _text(INSTALL_PS1)


def test_scripts_pin_the_same_uv_and_repository() -> None:
    sh, ps1 = _text(INSTALL_SH), _text(INSTALL_PS1)
    sh_uv = re.search(r'^UV_VERSION="([0-9.]+)"', sh, re.MULTILINE)
    ps_uv = re.search(r"^\s*\$UvVersion = '([0-9.]+)'", ps1, re.MULTILINE)
    assert sh_uv and ps_uv and sh_uv.group(1) == ps_uv.group(1)
    assert re.search(r'^UV_INSTALLER_SHA256="[0-9a-f]{64}"', sh, re.MULTILINE)
    assert re.search(r"^\s*\$UvInstallerSha256 = '[0-9a-f]{64}'", ps1, re.MULTILINE)
    assert 'EDMARS_REPO="cgpan/edm-ars-public"' in sh
    assert "$EdmarsRepo = 'cgpan/edm-ars-public'" in ps1
    assert 'PYTHON_SERIES="3.11"' in sh
    assert "$PythonSeries = '3.11'" in ps1


def test_documented_options_exist_in_both_scripts() -> None:
    sh, ps1, readme = _text(INSTALL_SH), _text(INSTALL_PS1), README.read_text(encoding="utf-8")
    for flag in GNU_FLAGS:
        assert flag in sh, flag
        assert f"'{flag}'" in ps1, flag  # the Windows script accepts GNU spellings too
        assert f"`{flag}" in readme, flag
    for flag in PS_FLAGS:
        assert f"'{flag}'" in ps1, flag
        assert f"`{flag}" in readme, flag
    assert "EDMARS_INSTALL_SOURCE" in sh and "EDMARS_INSTALL_SOURCE" in ps1


def test_launchers_follow_the_cli_contract() -> None:
    # CLI spec section 15: the launcher sets PYTHONUTF8=1 and EDMARS_APP_ROOT
    # and runs the venv's python with `-m edmars`; the app folder reaches the
    # import path through a .pth file, and -P keeps the user's current folder
    # off it.
    sh, ps1 = _text(INSTALL_SH), _text(INSTALL_PS1)
    for text in (sh, ps1):
        assert "EDMARS_APP_ROOT" in text
        assert "PYTHONUTF8=1" in text
        assert "-P -m edmars" in text
        assert "edm_ars_app.pth" in text
        assert "requirements.lock" in text and "requirements-cli.txt" in text
        assert "--no-bin" in text  # never drops python3.11 into ~/.local/bin
        assert "UV_UNMANAGED_INSTALL" in text  # a downloaded uv never edits PATH
        assert "SHA256SUMS" in text


def test_launchers_pass_on_the_uv_that_built_the_environment() -> None:
    # uv creates environments without pip, so edmars.lsar installs LSAR's
    # packages with the uv named in EDMARS_UV. The launcher sets it for
    # every install (a private uv in <dir>/uv is not on PATH), and only when
    # that uv still exists, so a user who later removes their own uv gets
    # edmars' PATH/ensurepip fallback instead of a dead path.
    sh, ps1 = _text(INSTALL_SH), _text(INSTALL_PS1)
    assert re.search(r"printf 'if \[ -x %s \]; then\\n' \"\$\(shell_quote \"\$UV\"\)\"", sh)
    assert "printf '    EDMARS_UV=%s\\n' \"$(shell_quote \"$UV\")\"" in sh
    assert "printf '    export EDMARS_UV\\n'" in sh
    assert ("('if exist \"' + (ConvertTo-CmdPath $uv) + '\" set \"EDMARS_UV=' "
            "+ (ConvertTo-CmdPath $uv) + '\"')") in ps1
    # Not only for a private uv: a uv found in ~/.local/bin or ~/.cargo/bin
    # is not necessarily on PATH either. So no condition on it between the
    # start of the launcher text and the line that runs Python.
    sh_block = sh[sh.index("printf '#!/bin/sh\\n'"): sh.index("printf 'exec %s -P -m edmars")]
    ps_block = ps1[ps1.index("$cmdLines = @("): ps1.index("'exit /b %ERRORLEVEL%'")]
    assert "EDMARS_UV" in sh_block and "UV_PRIVATE" not in sh_block
    assert "EDMARS_UV" in ps_block and "uvPrivate" not in ps_block


def test_scripts_do_not_promise_that_uninstall_removes_the_program() -> None:
    # `edmars uninstall` removes settings and keys; it cannot delete the
    # program it runs from, the launcher or the PATH change. Text the user
    # reads (plan, launcher comments, PATH block) must not claim otherwise.
    for path in (INSTALL_SH, INSTALL_PS1, README):
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"uninstall\W{0,3}\s+removes\s+(it|all|every)", text, re.IGNORECASE), path.name


def test_install_sh_shape() -> None:
    sh = _text(INSTALL_SH)
    assert sh.startswith("#!/bin/sh\n")
    assert "\nset -eu\n" in sh
    assert "\r" not in sh
    # The script only runs once main() is fully downloaded.
    assert sh.rstrip().splitlines()[-1] == 'main "$@"'
    # Prompts read the terminal, so `curl | sh` can still ask.
    assert "</dev/tty" in sh


def test_gitattributes_keeps_install_sh_lf() -> None:
    attrs = (REPO_ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert re.search(r"^\*\.sh\s+text\s+eol=lf\s*$", attrs, re.MULTILINE)


def test_install_ps1_has_no_exit_statement_by_text() -> None:
    # `exit` inside `irm | iex` closes the user's PowerShell window. (The AST
    # test below is exact; this one also runs where PowerShell is absent.)
    for number, line in enumerate(_text(INSTALL_PS1).splitlines(), 1):
        code = line.split("#", 1)[0]
        code = re.sub(r"'[^']*'", "''", code)
        code = re.sub(r'"[^"]*"', '""', code)
        assert not re.search(r"(^|[;{}\s])exit(\s|$|;)", code, re.IGNORECASE), (
            f"install.ps1:{number} uses exit: {line.strip()}"
        )


# --- PowerShell ------------------------------------------------------------------------


def _powershell(script: str, tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    assert POWERSHELL is not None
    path = tmp_path / "probe.ps1"
    path.write_text(script, encoding="utf-8-sig")
    return subprocess.run(
        [POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
         "-File", str(path), *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=120, env=_clean_env(tmp_path),
    )


@pytest.mark.skipif(POWERSHELL is None, reason="PowerShell is not installed")
def test_install_ps1_parses_and_never_exits(tmp_path: Path) -> None:
    script = f"""
$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile('{INSTALL_PS1}', [ref]$tokens, [ref]$errors)
$exits = $ast.FindAll({{ param($n) $n -is [System.Management.Automation.Language.ExitStatementAst] }}, $true)
$params = $ast.FindAll({{ param($n) $n -is [System.Management.Automation.Language.ParamBlockAst] }}, $false)
[Console]::Out.WriteLine((@{{ errors = @($errors | ForEach-Object {{ $_.Message }}); exits = @($exits).Count; params = @($params).Count }} | ConvertTo-Json -Compress))
"""
    result = _powershell(script, tmp_path)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["errors"] == []
    assert report["exits"] == 0
    # No top-level param block: under `irm | iex` its variables would be
    # created in the user's session.
    assert report["params"] == 0


def _function_probe(calls: str) -> str:
    """A script that defines the installer's helpers and runs ``calls``."""
    wanted = ", ".join(
        f"'{name}'"
        for name in ("Test-UnderPath", "Merge-UserPath", "ConvertTo-CmdPath",
                     "Get-SyncProvider", "Test-Excluded", "Get-Download",
                     "Get-ShLauncherText", "Get-PathPlanLine")
    )
    return f"""
$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile('{INSTALL_PS1}', [ref]$tokens, [ref]$errors)
$ExcludeAnywhere = @('.git', '__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache', 'node_modules', '.env')
$ExcludeTop = @('output', 'data', 'dist', 'cache', '.venv', 'venv', 'ideas', 'files')
foreach ($name in @({wanted})) {{
    $fn = $ast.Find({{ param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $name }}, $true)
    if (-not $fn) {{ throw "function $name not found" }}
    . ([scriptblock]::Create($fn.Extent.Text))
}}
$out = [ordered]@{{}}
{calls}
[Console]::Out.WriteLine(($out | ConvertTo-Json -Compress))
"""


@pytest.mark.skipif(not (ON_WINDOWS and POWERSHELL), reason="Windows PowerShell only")
def test_ps1_exclusion_lists_match_the_probe() -> None:
    # The probe below re-declares the two lists; keep them in step.
    ps1 = _text(INSTALL_PS1)
    assert ("$ExcludeAnywhere = @('.git', '__pycache__', '.pytest_cache', '.mypy_cache', "
            "'.ruff_cache', 'node_modules', '.env')") in ps1
    assert "$ExcludeTop = @('output', 'data', 'dist', 'cache', '.venv', 'venv', 'ideas', 'files')" in ps1


@pytest.mark.skipif(not (ON_WINDOWS and POWERSHELL), reason="Windows PowerShell only")
def test_ps1_user_path_merge(tmp_path: Path) -> None:
    calls = r"""
$bin = Join-Path $env:USERPROFILE '.local\bin'
$out.append_expand = Merge-UserPath '%USERPROFILE%\tools;C:\bar' $bin $true
$out.already = Merge-UserPath 'C:\bar;%USERPROFILE%\.local\bin\' $bin $true
$out.already_literal = Merge-UserPath ($bin.ToUpper() + ';C:\x') $bin $true
$out.empty = Merge-UserPath '' 'D:\tools\bin' $true
$out.trailing_semicolon = Merge-UserPath 'C:\a;' 'D:\b' $true
$out.plain_string_kind = Merge-UserPath 'C:\a' $bin $false
$out.bin = $bin
"""
    result = _powershell(_function_probe(calls), tmp_path)
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout.strip().splitlines()[-1])
    assert out["append_expand"] == r"%USERPROFILE%\tools;C:\bar;%USERPROFILE%\.local\bin"
    assert out["already"] is None
    assert out["already_literal"] is None
    assert out["empty"] == r"D:\tools\bin"
    assert out["trailing_semicolon"] == r"C:\a;D:\b"
    assert out["plain_string_kind"] == "C:\\a;" + out["bin"]


@pytest.mark.skipif(not (ON_WINDOWS and POWERSHELL), reason="Windows PowerShell only")
def test_ps1_plan_does_not_promise_a_path_change_when_the_folder_is_on_path(tmp_path: Path) -> None:
    # The plan said "Add <bin> to your user PATH" even when step 7 then
    # said "already on your user PATH". Both now decide with Merge-UserPath.
    calls = r"""
$bin = Join-Path $env:USERPROFILE '.local\bin'
$out.absent = Get-PathPlanLine 'C:\tools' $bin $false
$out.present = Get-PathPlanLine 'C:\tools;%USERPROFILE%\.local\bin\' $bin $false
$out.no_modify = Get-PathPlanLine 'C:\tools' $bin $true
$out.bin = $bin
"""
    result = _powershell(_function_probe(calls), tmp_path)
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout.strip().splitlines()[-1])
    assert out["absent"] == f"  7. Add {out['bin']} to your user PATH (your account only)."
    assert out["present"] == f"  7. Leave your PATH as it is: {out['bin']} is already on your user PATH."
    assert out["no_modify"] == "  7. Leave your PATH alone (-NoModifyPath)."


@pytest.mark.skipif(not (ON_WINDOWS and POWERSHELL), reason="Windows PowerShell only")
def test_ps1_launcher_paths_and_sync_detection(tmp_path: Path) -> None:
    calls = r"""
$out.local = ConvertTo-CmdPath (Join-Path $env:LOCALAPPDATA 'edm-ars\app\1.0.0')
$out.profile = ConvertTo-CmdPath (Join-Path $env:USERPROFILE 'x\venv')
$out.percent = ConvertTo-CmdPath 'D:\100%\edm-ars'
$out.onedrive = Get-SyncProvider 'D:\OneDrive - Contoso\edm-ars'
$out.dropbox = Get-SyncProvider 'D:\Dropbox (Personal)\edm-ars'
$out.gdrive = Get-SyncProvider 'G:\My Drive\edm-ars'  # a sync folder on purpose: audit-allow-path
$out.local_dir = Get-SyncProvider 'C:\edm-ars'
$out.ex_git = Test-Excluded '.git' '.git'
$out.ex_nested_cache = Test-Excluded 'src/__pycache__' '__pycache__'
$out.ex_data = Test-Excluded 'data' 'data'
$out.ex_registry = Test-Excluded 'data_registry' 'data_registry'
$out.ex_run_output = Test-Excluded 'runs/demo/output_2' 'output_2'
$out.ex_fixtures = Test-Excluded 'runs/fixtures' 'fixtures'
$out.ex_nested_output = Test-Excluded 'src/output' 'output'
$out.ex_env = Test-Excluded 'config/local.env' 'local.env'
$out.ex_tmp = Test-Excluded 'tmp_orch_test2' 'tmp_orch_test2'
"""
    result = _powershell(_function_probe(calls), tmp_path)
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout.strip().splitlines()[-1])
    assert out["local"] == r"%LOCALAPPDATA%\edm-ars\app\1.0.0"
    assert out["profile"] == r"%USERPROFILE%\x\venv"
    assert out["percent"] == r"D:\100%%\edm-ars"
    assert out["onedrive"] == "OneDrive"
    assert out["dropbox"] == "Dropbox"
    assert out["gdrive"] == "Google Drive"
    assert out["local_dir"] is None
    assert out["ex_git"] and out["ex_nested_cache"] and out["ex_data"]
    assert out["ex_run_output"] and out["ex_env"] and out["ex_tmp"]
    assert not out["ex_registry"] and not out["ex_fixtures"] and not out["ex_nested_output"]


@pytest.mark.skipif(not (ON_WINDOWS and POWERSHELL and SH), reason="Windows PowerShell and sh")
def test_ps1_writes_an_edmars_command_git_bash_can_run(tmp_path: Path) -> None:
    # bash does not use PATHEXT, so in Git Bash `edmars` never found
    # edmars.cmd. The installer now also writes an sh launcher named edmars.
    fake_python = tmp_path / "venv" / "fake python"
    fake_python.parent.mkdir()
    fake_python.write_bytes(
        b'#!/bin/sh\n'
        b'printf "ROOT=%s\\n" "$EDMARS_APP_ROOT"\n'
        b'printf "UTF8=%s PP=%s\\n" "$PYTHONUTF8" "${PYTHONPATH-unset}"\n'
        b'for a in "$@"; do printf "ARG=%s\\n" "$a"; done\n'
    )
    app_dir = r"C:\Program Files\O'Neil\app\0.1.0"
    launcher = tmp_path / "edmars"
    ps_app = app_dir.replace("'", "''")
    ps_python = str(fake_python).replace("'", "''")
    ps_launcher = str(launcher).replace("'", "''")
    calls = (
        f"$text = Get-ShLauncherText 'MARK' '0.1.0' '{ps_app}' 'C:\\no\\uv.exe' '{ps_python}'\n"
        f"[System.IO.File]::WriteAllText('{ps_launcher}', $text, "
        "(New-Object System.Text.UTF8Encoding $false))\n"
        "$out.ok = $true\n"
    )
    result = _powershell(_function_probe(calls), tmp_path)
    assert result.returncode == 0, result.stderr
    assert b"\r" not in launcher.read_bytes()

    study = r"D:\EDM-ARS\studies\2026-09-25_1200_gpa_ab12"
    run = subprocess.run([str(SH), str(launcher), "resume", study, "--yes"],
                         capture_output=True, text=True, timeout=60,
                         env={**_clean_env(tmp_path), "PYTHONPATH": "x"})
    assert run.returncode == 0, run.stderr
    lines = run.stdout.splitlines()
    assert f"ROOT={app_dir}" in lines
    assert "UTF8=1 PP=unset" in lines
    assert [line[4:] for line in lines if line.startswith("ARG=")] == [
        "-P", "-m", "edmars", "resume", study, "--yes"]


def _run_ps1(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    assert POWERSHELL is not None
    return subprocess.run(
        [POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
         "-File", str(INSTALL_PS1), *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=180, env=_clean_env(tmp_path),
    )


@pytest.mark.skipif(not (ON_WINDOWS and POWERSHELL), reason="Windows PowerShell only")
def test_ps1_dry_run_changes_nothing(tmp_path: Path) -> None:
    base, bin_dir = tmp_path / "base", tmp_path / "bin"
    result = _run_ps1(tmp_path, "-DryRun", "-FromLocal", str(REPO_ROOT),
                      "-Dir", str(base), "-BinDir", str(bin_dir))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Dry run: nothing was downloaded or changed." in result.stdout
    assert "This will:" in result.stdout
    assert f"  7. Add {bin_dir} to your user PATH" in result.stdout
    assert "  5. Install EDM-ARS and the packages it needs (about 1 GB" in result.stdout  # 950 MB measured
    assert not base.exists() and not bin_dir.exists()
    # GNU spellings are the same options.
    result = _run_ps1(tmp_path, "--dry-run", f"--dir={base}", "--bin-dir", str(bin_dir),
                      "--no-modify-path", "--no-onboard")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Leave your PATH alone" in result.stdout
    assert "Stop there" in result.stdout
    assert not base.exists()


@pytest.mark.skipif(not (ON_WINDOWS and POWERSHELL), reason="Windows PowerShell only")
def test_ps1_refuses_bad_input_before_changing_anything(tmp_path: Path) -> None:
    result = _run_ps1(tmp_path, "--bogus")
    assert result.returncode != 0
    assert "unknown option '--bogus'" in result.stdout

    missing = tmp_path / "no-such-checkout"
    result = _run_ps1(tmp_path, "-Yes", "-FromLocal", str(missing), "-Dir", str(tmp_path / "b"))
    assert result.returncode != 0
    assert "does not exist" in result.stdout
    assert not (tmp_path / "b").exists()

    # With long paths enabled the same command would install for real, so
    # this part only runs where the limit applies.
    if not _long_paths_enabled():
        too_long = tmp_path / ("x" * 150)
        result = _run_ps1(tmp_path, "-Yes", "-FromLocal", str(REPO_ROOT), "-Dir", str(too_long))
        assert result.returncode != 0
        assert "too long for Windows" in result.stdout
        assert not too_long.exists()


@pytest.mark.skipif(not (ON_WINDOWS and POWERSHELL), reason="Windows PowerShell only")
def test_ps1_downloads_only_over_https(tmp_path: Path) -> None:
    # install.sh refuses plain http (curl --proto '=https'); install.ps1
    # used to fetch SHA256SUMS and the archive over it, so the fingerprint
    # check proved nothing.
    dest = str(tmp_path / "sums").replace("'", "''")
    calls = f"""
try {{ Get-Download 'http://127.0.0.1:9/SHA256SUMS' '{dest}'; $out.http = 'downloaded' }}
catch {{ $out.http = $_.Exception.Message }}
"""
    result = _powershell(_function_probe(calls), tmp_path)
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout.strip().splitlines()[-1])
    assert "only https://" in out["http"]
    assert not (tmp_path / "sums").exists()

    base = tmp_path / "base"
    result = subprocess.run(
        [POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
         "-File", str(INSTALL_PS1), "-DryRun", "-Dir", str(base), "-BinDir", str(tmp_path / "bin")],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180,
        env=_clean_env(tmp_path, EDMARS_RELEASE_BASE_URL="http://127.0.0.1:9/dist"),
    )
    assert result.returncode != 0
    assert "must start with https://" in result.stdout
    assert not base.exists()


def _long_paths_enabled() -> bool:
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"SYSTEM\CurrentControlSet\Control\FileSystem") as key:
            value, _kind = winreg.QueryValueEx(key, "LongPathsEnabled")
    except OSError:
        return False
    return value == 1


# --- POSIX sh ---------------------------------------------------------------------------


def _run_sh(tmp_path: Path, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
    assert SH is not None
    # A new session has no controlling terminal, so /dev/tty cannot be
    # opened and the script cannot stop to ask (as under CI).
    return subprocess.run(
        [SH, str(INSTALL_SH), *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=120, env=_clean_env(tmp_path, **env), stdin=subprocess.DEVNULL,
        start_new_session=True,
    )


@pytest.mark.skipif(SH is None, reason="sh is not installed")
def test_install_sh_parses(tmp_path: Path) -> None:
    result = subprocess.run([SH, "-n", str(INSTALL_SH)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


#: requirements-dev.txt pins shellcheck-py to the shellcheck the CI lint step
#: gets from Ubuntu 24.04 (0.9.0), so this test and that step agree. Its
#: command sits next to the venv's python even when the venv is not active.
SHELLCHECK = shutil.which("shellcheck") or shutil.which("shellcheck", path=str(Path(sys.executable).parent))


@pytest.mark.skipif(SHELLCHECK is None, reason="shellcheck is not installed")
def test_install_sh_passes_shellcheck() -> None:
    assert SHELLCHECK is not None
    result = subprocess.run([SHELLCHECK, "-s", "sh", str(INSTALL_SH)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout


def _sh_path(path: Path) -> str:
    """``path`` as the sh under test spells it (Git Bash wants /c/...)."""
    if not ON_WINDOWS:
        return str(path)
    cygpath = shutil.which("cygpath")
    if cygpath is None:
        pytest.skip("no cygpath to translate Windows paths for sh")
    return subprocess.run([cygpath, "-u", str(path)], capture_output=True, text=True,
                          check=True).stdout.strip()


def _launcher_writer() -> str:
    """install.sh's own shell_quote() and launcher-writing lines, verbatim."""
    sh = _text(INSTALL_SH)
    quote = re.search(r"^shell_quote\(\) \{\n.*?\n\}\n", sh, re.MULTILINE | re.DOTALL)
    assert quote is not None
    start = sh.index('    LAUNCHER_TMP="$BIN_DIR/.edmars.$$"\n')
    end = sh.index('    mv -f "$LAUNCHER_TMP" "$LAUNCHER"\n')
    return quote.group(0) + sh[start:end]


@pytest.mark.skipif(SH is None, reason="sh is not installed")
@pytest.mark.parametrize("uv_exists", [True, False])
def test_install_sh_launcher_sets_the_environment_and_runs(tmp_path: Path, uv_exists: bool) -> None:
    # Runs the launcher text install.sh writes, with a stand-in Python that
    # prints what it was given. Paths carry a space and a quote on purpose.
    root = tmp_path / "it's here"
    app, bin_dir, venv_bin = root / "app" / "9.9.9", root / "bin", root / "venv" / "bin"
    for folder in (app, bin_dir, venv_bin):
        folder.mkdir(parents=True)
    fake_py = venv_bin / "python"
    fake_py.write_bytes(
        b"#!/bin/sh\n"
        b"printf 'uv=%s\\n' \"${EDMARS_UV-<unset>}\"\n"
        b"printf 'root=%s\\n' \"$EDMARS_APP_ROOT\"\n"
        b"printf 'utf8=%s\\n' \"$PYTHONUTF8\"\n"
        b"printf 'pythonpath=%s\\n' \"${PYTHONPATH-<unset>}\"\n"
        b"for a in \"$@\"; do printf 'arg=%s\\n' \"$a\"; done\n"
    )
    fake_py.chmod(0o755)
    uv = root / "uv" / "uv"
    if uv_exists:
        uv.parent.mkdir()
        uv.write_bytes(b"#!/bin/sh\nexit 0\n")
        uv.chmod(0o755)
    def q(path: Path) -> str:
        return "'" + _sh_path(path).replace("'", "'\\''") + "'"

    harness = (
        "set -eu\n"
        "LAUNCHER_MARK='mark'\nVERSION='9.9.9'\n"
        f"APP_DIR={q(app)}\nVENV_PY={q(fake_py)}\nUV={q(uv)}\nBIN_DIR={q(bin_dir)}\n"
        + _launcher_writer()
        + 'mv -f "$LAUNCHER_TMP" "$BIN_DIR/edmars"\n'
    )
    script = tmp_path / "harness.sh"
    script.write_bytes(harness.encode("utf-8"))
    env = _clean_env(tmp_path, PYTHONPATH="should-be-cleared")
    built = subprocess.run([SH, str(script)], capture_output=True, text=True, env=env)
    assert built.returncode == 0, built.stderr
    launcher = bin_dir / "edmars"
    text = launcher.read_text(encoding="utf-8")
    assert text.startswith("#!/bin/sh\n# mark (EDM-ARS 9.9.9).\n")
    ran = subprocess.run([SH, _sh_path(launcher), "version", "two words"],
                         capture_output=True, text=True, env=env)
    assert ran.returncode == 0, ran.stderr
    lines = ran.stdout.splitlines()
    expected_uv = f"uv={_sh_path(uv)}" if uv_exists else "uv=<unset>"
    assert expected_uv in lines
    assert f"root={_sh_path(app)}" in lines
    assert "utf8=1" in lines and "pythonpath=<unset>" in lines
    assert [line for line in lines if line.startswith("arg=")] == [
        "arg=-P", "arg=-m", "arg=edmars", "arg=version", "arg=two words"]


@pytest.mark.skipif(SH is None or ON_WINDOWS, reason="needs a POSIX sh with POSIX paths")
def test_install_sh_dry_run_changes_nothing(tmp_path: Path) -> None:
    base, bin_dir = tmp_path / "base", tmp_path / "bin"
    home = tmp_path / "home"
    home.mkdir()
    result = _run_sh(tmp_path, "--dry-run", "--from-local", str(REPO_ROOT),
                     "--dir", str(base), "--bin-dir", str(bin_dir), HOME=str(home))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Dry run: nothing was downloaded or changed." in result.stdout
    assert not base.exists() and not bin_dir.exists()
    assert list(home.iterdir()) == []  # no shell start-up file touched


@pytest.mark.skipif(SH is None, reason="sh is not installed")
def test_install_sh_makes_a_uv_found_through_a_relative_path_absolute(tmp_path: Path) -> None:
    # The script changes folder before it first runs uv, so a uv found
    # through a relative PATH entry (PATH=bin:...) must be made absolute.
    work = tmp_path / "work"
    fake_uv = work / "relbin" / "uv"
    fake_uv.parent.mkdir(parents=True)
    fake_uv.write_bytes(b"#!/bin/sh\necho 'uv 0.10.6 (fake)'\n")
    fake_uv.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()
    env = _clean_env(tmp_path, HOME=_sh_path(home))
    env["PATH"] = "relbin" + os.pathsep + env.get("PATH", "")
    result = subprocess.run(
        [SH, _sh_path(INSTALL_SH), "--dry-run", "--from-local", _sh_path(REPO_ROOT),
         "--dir", _sh_path(tmp_path / "base"), "--bin-dir", _sh_path(tmp_path / "bin")],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=120, env=env, cwd=work, stdin=subprocess.DEVNULL,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"({_sh_path(fake_uv)}, version 0.10.6)" in result.stdout, result.stdout


@pytest.mark.skipif(SH is None, reason="sh is not installed")
@pytest.mark.parametrize("in_file, on_path", [(True, True), (False, True), (True, False), (False, False)])
def test_install_sh_decides_the_path_change_from_the_shell_start_up_files(
    tmp_path: Path, in_file: bool, on_path: bool
) -> None:
    # On the owner's Mac the installer ran under an app whose PATH already
    # had ~/.local/bin, so it wrote no block and a new Terminal window could
    # not find edmars (on_path without in_file). The folder is left alone
    # only when a start-up file puts it on PATH and this PATH has it too;
    # the plan says which file it found, or which files it will change, and
    # step 7 then does what the plan said.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    zprofile = home / ".zprofile"
    if in_file:
        zprofile.write_text(f'export PATH="{_sh_path(bin_dir)}:$PATH"\n', encoding="utf-8")
    env = _clean_env(tmp_path, HOME=_sh_path(home), SHELL="/bin/zsh")
    if on_path:
        env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
    result = subprocess.run(
        [SH, _sh_path(INSTALL_SH), "--dry-run", "--from-local", _sh_path(REPO_ROOT),
         "--dir", _sh_path(tmp_path / "base"), "--bin-dir", _sh_path(bin_dir)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=120, env=env, stdin=subprocess.DEVNULL,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    already = (f"  7. Leave your PATH as it is: {_sh_path(zprofile)} already puts "
               f"{_sh_path(bin_dir)} on it.")
    add = (f"  7. Add {_sh_path(bin_dir)} to your PATH (your account only), in a marked block at\n"
           "     the end of ~/.profile, ~/.zshrc.")
    if in_file and on_path:
        assert already in result.stdout and "7. Add" not in result.stdout
    else:
        assert add in result.stdout and "7. Leave" not in result.stdout
    assert sorted(p.name for p in home.iterdir()) == ([".zprofile"] if in_file else [])


_START_UP_LINES = [
    ('export PATH="$HOME/.local/bin:$PATH"', True),
    ('export PATH="${HOME}/.local/bin:${PATH}"', True),
    ("PATH=~/.local/bin:$PATH", True),
    ("path=(~/.local/bin $path)", True),
    ("fish_add_path ~/.local/bin", True),
    ('. "$HOME/.local/bin/env"', True),  # uv's installer writes this line
    ('[ -f ~/.local/bin/env ] && source ~/.local/bin/env', True),
    ('case ":${PATH}:" in *":$HOME/.local/bin:"*) ;; *) export PATH="$HOME/.local/bin:$PATH" ;; esac', True),
    ('  # export PATH="$HOME/.local/bin:$PATH"', False),
    ('export PYTHONPATH="$HOME/.local/bin"', False),
    ('export PATH="$HOME/.local/bin2:$PATH"', False),
    ("alias e=~/.local/bin/edmars", False),
    ('if [ -d "$HOME/.local/bin" ] ; then', False),
]


@pytest.mark.skipif(SH is None, reason="sh is not installed")
def test_puts_on_path_reads_start_up_lines_without_running_them(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    body = "set -eu\n" + _sh_function("puts_on_path") + (
        'for f in "$@"; do if puts_on_path "$f" "$HOME/.local/bin"; then echo yes; else echo no; fi; done\n')
    files = []
    for i, (line, _) in enumerate(_START_UP_LINES):
        path = home / f"rc{i}"
        # A line that ran would print; the function only reads it.
        path.write_text("echo RAN\n" + line + "\n", encoding="utf-8")
        files.append(_sh_path(path))
    script = tmp_path / "probe.sh"
    script.write_bytes(body.encode("utf-8"))
    result = subprocess.run([str(SH), _sh_path(script), *files], capture_output=True, text=True,
                            env=_clean_env(tmp_path, HOME=_sh_path(home)), timeout=60)
    assert result.returncode == 0, result.stderr
    assert "RAN" not in result.stdout
    got = result.stdout.split()
    assert got == ["yes" if expected else "no" for _, expected in _START_UP_LINES], list(
        zip([line for line, _ in _START_UP_LINES], got))


@pytest.mark.skipif(SH is None, reason="sh is not installed")
@pytest.mark.parametrize("shell, found", [
    ("/bin/zsh", ".zshrc"), ("/bin/bash", ".bashrc"), ("/usr/bin/fish", "config.fish"), ("/bin/sh", None)])
def test_path_setup_file_reads_the_files_of_the_login_shell(tmp_path: Path, shell: str, found: str | None) -> None:
    home = tmp_path / "home"
    (home / ".config" / "fish").mkdir(parents=True)
    line = 'export PATH="$HOME/.local/bin:$PATH"\n'
    for name in (".zshrc", ".bashrc", ".config/fish/config.fish"):
        (home / name).write_text(line, encoding="utf-8")
    body = ("set -eu\n" + _sh_function("puts_on_path") + _sh_function("path_setup_file")
            + 'path_setup_file "$HOME/.local/bin" || echo none\n')
    script = tmp_path / "probe.sh"
    script.write_bytes(body.encode("utf-8"))
    result = subprocess.run([str(SH), _sh_path(script)], capture_output=True, text=True,
                            env=_clean_env(tmp_path, HOME=_sh_path(home), SHELL=shell), timeout=60)
    assert result.returncode == 0, result.stderr
    out = result.stdout.strip()
    assert out == ("none" if found is None else _sh_path(home) + "/" + (
        ".config/fish/config.fish" if found == "config.fish" else found))


# --- macOS: an OpenMP library for XGBoost without Homebrew ------------------------------
#
# XGBoost's macOS wheel needs @rpath/libomp.dylib and finds it only in
# Homebrew's folder or in <private Python>/lib; install.sh links
# scikit-learn's bundled copy there (link_openmp). The real dyld check
# (check_openmp) runs only on a Mac, in CI's macos-14 installer job; here
# install.sh's own shell code runs against a fake layout, with a stand-in
# python and a check_openmp that passes when the link resolves to
# scikit-learn's file, which is what dyld needs to map it only once.

_FAKE_ENV_PYTHON = b"""#!/bin/sh
# Stand-in for the environment's python. It answers link_openmp's
# question, and fails the package import the way dyld does while nothing
# is at <python>/lib/libomp.dylib.
case "$2" in
    *find_spec*) printf '%s\\n' "$FAKE_SKLEARN_OMP" "$FAKE_PYTHON_HOME" ;;
    'import numpy'*)
        if [ -e "$FAKE_PYTHON_HOME/lib/libomp.dylib" ] || [ -n "${FAKE_HOMEBREW_LIBOMP:-}" ]; then exit 0; fi
        echo "$FAKE_IMPORT_ERROR" >&2
        exit 1
        ;;
    *) echo "unexpected python call: $2" >&2; exit 3 ;;
esac
"""

_FAKE_CHECK_OPENMP = """
check_openmp() {
    echo checked >>"${FAKE_CHECK_LOG:-/dev/null}"
    if [ "${FAKE_CHECK:-}" = fail ]; then
        echo "expected one OpenMP library, found 2: /opt/homebrew/opt/libomp/lib/libomp.dylib, $FAKE_SKLEARN_OMP" >&2
        return 1
    fi
    if [ "${FAKE_CHECK:-}" = pass ]; then return 0; fi
    [ "$FAKE_PYTHON_HOME/lib/libomp.dylib" -ef "$FAKE_SKLEARN_OMP" ]
}
"""

_LIBOMP_ERROR = ("XGBoostError: dlopen(.../libxgboost.dylib, 0x0006): Library not loaded: "
                 "@rpath/libomp.dylib")


def _sh_function(name: str) -> str:
    """One function from install.sh, verbatim."""
    match = re.search(rf"^{name}\(\) \{{\n.*?\n\}}\n", _text(INSTALL_SH), re.MULTILINE | re.DOTALL)
    assert match is not None, name
    return match.group(0)


class _OmpLayout:
    """An install folder as install.sh leaves it on a Mac, minus the binaries."""

    def __init__(self, tmp_path: Path) -> None:
        # The macOS default install folder has a space in it.
        self.tmp = tmp_path
        self.base = tmp_path / "Application Support" / "edm-ars"
        self.pyroot = self.base / "python"
        self.home = self.pyroot / "cpython-3.11.14-macos-aarch64-none"
        (self.home / "lib").mkdir(parents=True)
        self.link = self.home / "lib" / "libomp.dylib"
        self.sk_omp = self._venv("0.1.0")
        self.python = self.base / "venv-0.1.0" / "bin" / "python"
        self.python.parent.mkdir(parents=True)
        self.python.write_bytes(_FAKE_ENV_PYTHON)
        self.python.chmod(0o755)
        self.env = _clean_env(tmp_path)
        if ON_WINDOWS:
            # Git Bash copies files for `ln -s` unless asked for native links.
            self.env["MSYS"] = "winsymlinks:nativestrict"
        self.env.update(FAKE_SKLEARN_OMP=_sh_path(self.sk_omp), FAKE_PYTHON_HOME=_sh_path(self.home),
                        FAKE_IMPORT_ERROR=_LIBOMP_ERROR)
        probe = tmp_path / "symlink-probe"
        probe.write_bytes(b"x")
        made = subprocess.run([str(SH), "-c", 'ln -s "$1" "$1.link"', "sh", _sh_path(probe)],
                              env=self.env, capture_output=True, text=True)
        if made.returncode != 0 or not Path(str(probe) + ".link").is_symlink():
            pytest.skip("this sh cannot make symbolic links here")

    def _venv(self, version: str) -> Path:
        omp = (self.base / f"venv-{version}" / "lib" / "python3.11" / "site-packages"
               / "sklearn" / ".dylibs" / "libomp.dylib")
        omp.parent.mkdir(parents=True)
        omp.write_bytes(f"libomp from scikit-learn in venv-{version}".encode())
        return omp

    def run(self, body: str, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
        script = self.tmp / "harness.sh"
        script.write_bytes(body.encode("utf-8"))
        return subprocess.run([str(SH), _sh_path(script), *args], capture_output=True, text=True,
                              env={**self.env, **env}, timeout=60)

    def provide(self, mode: str, **env: str) -> subprocess.CompletedProcess[str]:
        body = ("set -eu\n" + _sh_function("link_openmp") + _sh_function("provide_openmp")
                + _FAKE_CHECK_OPENMP + 'provide_openmp "$1" "$2" "$3" "$4"\n')
        return self.run(body, _sh_path(self.python), _sh_path(self.pyroot), mode,
                        _sh_path(self.tmp / "openmp.err"), **env)

    def points_at(self, target: Path) -> bool:
        return self.link.is_symlink() and os.path.samefile(self.link, target)


def _step5_check() -> str:
    """install.sh's package check (step 5), verbatim."""
    sh = _text(INSTALL_SH)
    start = sh.index('    say "Checking that the main packages load..."\n')
    end = sh.index("    # EDM-ARS itself is not a pip package")
    return sh[start:end]


@pytest.mark.skipif(SH is None, reason="sh is not installed")
def test_link_openmp_links_scikit_learns_copy_and_only_there(tmp_path: Path) -> None:
    lay = _OmpLayout(tmp_path)
    # "refresh" never makes a link that is not there yet.
    result = lay.provide("refresh")
    assert result.returncode == 1 and result.stdout == ""
    assert not lay.link.is_symlink()

    result = lay.provide("create")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == _sh_path(lay.link)
    assert lay.points_at(lay.sk_omp)
    assert lay.sk_omp.read_bytes() == b"libomp from scikit-learn in venv-0.1.0"


@pytest.mark.skipif(SH is None, reason="sh is not installed")
@pytest.mark.parametrize("old_state", ["previous version", "deleted version"])
def test_link_openmp_repoints_an_earlier_installs_link_to_this_environment(
    tmp_path: Path, old_state: str
) -> None:
    # An update leaves the link pointing into the previous version's venv
    # (or at nothing, once that venv is removed). Left alone, the new
    # version's XGBoost would load the old venv's copy while its
    # scikit-learn loads its own: two OpenMP runtimes in one process.
    lay = _OmpLayout(tmp_path)
    old = lay._venv("0.0.9")
    os.symlink(old, lay.link)
    if old_state == "deleted version":
        old.unlink()
    result = lay.provide("refresh")
    assert result.returncode == 0, result.stderr
    assert lay.points_at(lay.sk_omp)


@pytest.mark.skipif(SH is None, reason="sh is not installed")
def test_link_openmp_never_replaces_a_real_file(tmp_path: Path) -> None:
    lay = _OmpLayout(tmp_path)
    lay.link.write_bytes(b"someone else's libomp")
    for mode in ("create", "refresh"):
        result = lay.provide(mode)
        assert result.returncode == 1
        assert not lay.link.is_symlink()
        assert lay.link.read_bytes() == b"someone else's libomp"


@pytest.mark.skipif(SH is None, reason="sh is not installed")
def test_link_openmp_needs_scikit_learns_copy_and_the_private_python(tmp_path: Path) -> None:
    lay = _OmpLayout(tmp_path)
    lay.sk_omp.unlink()  # a scikit-learn that ships no libomp.dylib
    assert lay.provide("create").returncode == 1
    assert not lay.link.is_symlink() and not lay.link.exists()

    # A python outside the folder this installer owns is never written to.
    lay = _OmpLayout(tmp_path / "second")
    foreign = tmp_path / "someone-elses-python"
    (foreign / "lib").mkdir(parents=True)
    result = lay.provide("create", FAKE_PYTHON_HOME=_sh_path(foreign))
    assert result.returncode == 1
    assert list((foreign / "lib").iterdir()) == []


@pytest.mark.skipif(SH is None, reason="sh is not installed")
def test_provide_openmp_removes_the_link_when_the_check_fails(tmp_path: Path) -> None:
    lay = _OmpLayout(tmp_path)
    result = lay.provide("create", FAKE_CHECK="fail")
    assert result.returncode == 1 and result.stdout == ""
    assert not lay.link.is_symlink() and not lay.link.exists()
    assert "found 2" in (tmp_path / "openmp.err").read_text(encoding="utf-8")


def _run_step5(lay: _OmpLayout, os_name: str = "Darwin", **env: str) -> subprocess.CompletedProcess[str]:
    tmp_dir = lay.tmp / "installer-tmp"
    tmp_dir.mkdir(exist_ok=True)
    body = (
        "set -eu\n"
        "say() { printf '%s\\n' \"$*\"; }\n"
        "warn() { printf '  ! %s\\n' \"$*\"; }\n"
        "die() { printf 'DIE: %s\\n' \"$*\" >&2; exit 1; }\n"
        + _sh_function("link_openmp") + _sh_function("provide_openmp") + _FAKE_CHECK_OPENMP
        + 'OS=$1\nVENV_PY=$2\nAPP_BASE=$3\nTMP_DIR=$4\nAPP_DIR="$APP_BASE/app/0.1.0"\n'
        + _step5_check()
        + "printf 'OMP_LINK=%s\\n' \"$OMP_LINK\"\n"
    )
    return lay.run(body, os_name, _sh_path(lay.python), _sh_path(lay.base), _sh_path(tmp_dir), **env)


@pytest.mark.skipif(SH is None, reason="sh is not installed")
def test_install_step5_links_openmp_when_xgboost_cannot_find_it(tmp_path: Path) -> None:
    # What CI's macos-14 job hit: no Homebrew, so `import xgboost` failed
    # and the installer stopped, asking for Homebrew.
    lay = _OmpLayout(tmp_path)
    result = _run_step5(lay)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "linking the one that comes with scikit-learn" in result.stdout
    assert f"OMP_LINK={_sh_path(lay.link)}" in result.stdout.splitlines()
    assert lay.points_at(lay.sk_omp)
    assert "DIE" not in result.stderr


@pytest.mark.skipif(SH is None, reason="sh is not installed")
def test_install_step5_repoints_the_link_before_checking(tmp_path: Path) -> None:
    # With the previous version's link still in place the import succeeds
    # at once, so the link must be re-pointed before the check, not after
    # a failure.
    lay = _OmpLayout(tmp_path)
    os.symlink(lay._venv("0.0.9"), lay.link)
    result = _run_step5(lay)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"OMP_LINK={_sh_path(lay.link)}" in result.stdout.splitlines()
    assert "linking the one that comes with scikit-learn" not in result.stdout
    assert lay.points_at(lay.sk_omp)


@pytest.mark.skipif(SH is None, reason="sh is not installed")
def test_install_step5_falls_back_to_homebrew_advice_and_leaves_no_link(tmp_path: Path) -> None:
    lay = _OmpLayout(tmp_path)
    result = _run_step5(lay, FAKE_CHECK="fail")
    assert result.returncode == 1
    assert "brew install libomp" in result.stderr
    assert "found 2" in result.stderr  # the reason is shown, not only the advice
    assert not lay.link.is_symlink() and not lay.link.exists()


@pytest.mark.skipif(SH is None, reason="sh is not installed")
@pytest.mark.parametrize("runtimes", [1, 2])
def test_install_step5_counts_the_openmp_libraries_when_homebrews_libomp_is_there(
    tmp_path: Path, runtimes: int
) -> None:
    # The owner's Mac had Homebrew's libomp: `import xgboost` worked, no
    # link was made, and nothing counted the OpenMP copies, although XGBoost
    # loaded Homebrew's and scikit-learn its own. The same one-library check
    # now runs; two copies are a note (the study there ran fine), not a stop.
    lay = _OmpLayout(tmp_path)
    log = tmp_path / "checks.log"
    result = _run_step5(lay, FAKE_HOMEBREW_LIBOMP="1", FAKE_CHECK="pass" if runtimes == 1 else "fail",
                        FAKE_CHECK_LOG=_sh_path(log))
    assert result.returncode == 0, result.stdout + result.stderr
    assert log.read_text(encoding="utf-8").split() == ["checked"]
    assert "OMP_LINK=" in result.stdout.splitlines() and not lay.link.exists()
    note = "XGBoost and scikit-learn load two different OpenMP libraries (found 2: /opt/homebrew/opt/libomp"
    readme = f"{_sh_path(lay.base)}/app/0.1.0/install/README.md,"
    if runtimes == 1:
        assert "OpenMP" not in result.stdout
    else:
        assert "  ! " + note in result.stdout
        assert readme in result.stdout
        assert 'item "macOS: XGBoost and the OpenMP library"' in result.stdout
    assert "DIE" not in result.stderr


@pytest.mark.skipif(SH is None, reason="sh is not installed")
def test_install_step5_links_nothing_for_other_failures_or_systems(tmp_path: Path) -> None:
    lay = _OmpLayout(tmp_path)
    result = _run_step5(lay, FAKE_IMPORT_ERROR="ModuleNotFoundError: No module named 'fitz'")
    assert result.returncode == 1
    assert "do not load" in result.stderr and "No module named 'fitz'" in result.stderr
    assert not lay.link.exists()

    result = _run_step5(lay, os_name="Linux")
    assert result.returncode == 1
    assert "do not load" in result.stderr and "brew" not in result.stderr
    assert not lay.link.exists()

    # Linux has no dyld to ask: the OpenMP check is macOS-only.
    log = tmp_path / "checks.log"
    result = _run_step5(lay, os_name="Linux", FAKE_HOMEBREW_LIBOMP="1", FAKE_CHECK_LOG=_sh_path(log))
    assert result.returncode == 0, result.stdout + result.stderr
    assert not log.exists()


# Not on Windows: the stand-in uname is found first under a local Git Bash
# but not on the GitHub windows-2022 runner, whose sh reports MINGW from the
# real uname. The Ubuntu tests job runs this, and the macos-14 installer job
# exercises the real Mac path end to end.
@pytest.mark.skipif(SH is None or ON_WINDOWS, reason="needs a POSIX sh whose PATH lookup honours the stand-in uname")
def test_install_sh_plans_a_mac_install_into_application_support(tmp_path: Path) -> None:
    # Stand-in uname and sysctl make install.sh see an Apple Silicon Mac.
    # The macOS default folder has a space in it, and the plan must say
    # that the OpenMP library may be linked.
    fakebin = tmp_path / "fakebin"
    fakebin.mkdir()
    (fakebin / "uname").write_bytes(b'#!/bin/sh\nif [ "${1:-}" = -m ]; then echo arm64; else echo Darwin; fi\n')
    (fakebin / "sysctl").write_bytes(
        b'#!/bin/sh\ncase "$2" in\n    hw.optional.arm64) echo 1 ;;\n'
        b'    hw.memsize) echo 17179869184 ;;\n    *) echo 0 ;;\nesac\n')
    for fake in fakebin.iterdir():
        fake.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()
    env = _clean_env(tmp_path, HOME=_sh_path(home))
    env["PATH"] = str(fakebin) + os.pathsep + env.get("PATH", "")
    result = subprocess.run(
        [SH, _sh_path(INSTALL_SH), "--dry-run", "--from-local", _sh_path(REPO_ROOT)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=120, env=env, stdin=subprocess.DEVNULL,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    out = result.stdout
    assert "Computer:      macOS (Apple Silicon)" in out
    assert f"Install into:  {_sh_path(home)}/Library/Application Support/edm-ars\n" in out
    assert "link the\n     one that comes with scikit-learn into the private Python" in out
    assert "packages it needs (about 0.7 GB" in out  # 715 MB measured on an M1 Pro
    assert "Dry run: nothing was downloaded or changed." in out
    assert list(home.iterdir()) == []


@pytest.mark.skipif(SH is None or ON_WINDOWS, reason="needs a POSIX sh whose PATH lookup honours the stand-in uname")
@pytest.mark.parametrize("machine, size", [("x86_64", "about 2 GB"), ("aarch64", "about 1-2 GB")])
def test_install_sh_plan_estimates_the_size_for_this_platform(tmp_path: Path, machine: str, size: str) -> None:
    # The plan said "about 1.5 GB" everywhere; an install measured 715 MB on
    # an Apple Silicon Mac and about 2.0 GB on Linux x86_64, where XGBoost
    # brings NVIDIA's NCCL library. Linux on arm64 is not measured yet.
    fakebin = tmp_path / "fakebin"
    fakebin.mkdir()
    (fakebin / "uname").write_bytes(
        f'#!/bin/sh\nif [ "${{1:-}}" = -m ]; then echo {machine}; else echo Linux; fi\n'.encode())
    (fakebin / "uname").chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()
    env = _clean_env(tmp_path, HOME=_sh_path(home))
    env["PATH"] = str(fakebin) + os.pathsep + env.get("PATH", "")
    result = subprocess.run(
        [SH, _sh_path(INSTALL_SH), "--dry-run", "--from-local", _sh_path(REPO_ROOT),
         "--dir", _sh_path(tmp_path / "base"), "--bin-dir", _sh_path(tmp_path / "bin")],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=120, env=env, stdin=subprocess.DEVNULL,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"  5. Install EDM-ARS and the packages it needs ({size}" in result.stdout


@pytest.mark.skipif(SH is None or ON_WINDOWS, reason="needs a POSIX sh with POSIX paths")
def test_install_sh_refuses_bad_input_before_changing_anything(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    result = _run_sh(tmp_path, "--bogus", HOME=str(home))
    assert result.returncode == 1
    assert "unknown option '--bogus'" in result.stderr

    result = _run_sh(tmp_path, "--help", HOME=str(home))
    assert result.returncode == 0 and "--from-local" in result.stdout

    # No terminal and no --yes: stop and say how, instead of hanging.
    result = _run_sh(tmp_path, "--from-local", str(REPO_ROOT), "--dir", str(tmp_path / "b"),
                     HOME=str(home))
    if "not supported" not in result.stderr:
        assert result.returncode == 1
        assert "--yes" in result.stderr
        assert not (tmp_path / "b").exists()

    # The default folder inside a sync service is refused unless --dir is given.
    synced_home = tmp_path / "Dropbox" / "home"
    synced_home.mkdir(parents=True)
    result = _run_sh(tmp_path, "--yes", "--from-local", str(REPO_ROOT), HOME=str(synced_home))
    if "not supported" not in result.stderr:
        assert result.returncode == 1
        assert "Dropbox" in result.stderr
        assert list(synced_home.iterdir()) == []


_FAKE_UV = r"""#!/bin/sh
# A stand-in uv: logs each call with the cache it was given, and makes the
# files the installer looks for next.
printf '%s | cache=%s\n' "$*" "${UV_CACHE_DIR-<unset>}" >>"$FAKE_UV_LOG"
case "$1" in
    --version) echo "uv 0.10.6 (fake)" ;;
    python) if [ "$2" = find ]; then echo "$FAKE_BASE_PY"; fi ;;
    venv)
        for last in "$@"; do :; done
        mkdir -p "$last/bin"
        printf '%s\n' '#!/bin/sh' \
            'case "$*" in' \
            '    *sysconfig*) d="$(cd "$(dirname "$0")/.." && pwd)/lib/site"; mkdir -p "$d"; echo "$d" ;;' \
            '    *edmars*) echo "edmars (fake)" ;;' \
            'esac' >"$last/bin/python"
        chmod 755 "$last/bin/python"
        ;;
esac
exit 0
"""


@pytest.mark.skipif(SH is None or ON_WINDOWS, reason="needs a POSIX sh with POSIX paths")
@pytest.mark.parametrize("private", [True, False])
def test_install_sh_keeps_its_own_uvs_cache_in_the_install_folder(tmp_path: Path, private: bool) -> None:
    # With the uv it downloaded, the installer left uv's cache (about 1.8 GB
    # on Linux) in ~/.cache/uv, where nothing listed it for removal. A uv
    # the user already had keeps its own cache.
    base, home = tmp_path / "base", tmp_path / "home"
    home.mkdir()
    uv = base / "uv" / "uv" if private else tmp_path / "own" / "uv"
    uv.parent.mkdir(parents=True)
    uv.write_text(_FAKE_UV, encoding="utf-8")
    uv.chmod(0o755)
    log = tmp_path / "uv.log"
    env = {"FAKE_UV_LOG": str(log), "FAKE_BASE_PY": str(tmp_path / "python3"), "HOME": str(home)}
    if not private:
        env["PATH"] = f"{uv.parent}{os.pathsep}{os.environ.get('PATH', '')}"
    result = _run_sh(tmp_path, "--yes", "--no-onboard", "--no-modify-path", "--from-local",
                     str(REPO_ROOT), "--dir", str(base), "--bin-dir", str(tmp_path / "bin"), **env)
    assert result.returncode == 0, result.stdout + result.stderr
    calls = log.read_text(encoding="utf-8").splitlines()
    installs = [line for line in calls if line.startswith(("python install", "venv", "pip install"))]
    assert len(installs) == 3, calls
    expected = f"cache={base / 'uv' / 'cache'}" if private else "cache=<unset>"
    assert all(line.endswith(expected) for line in installs), calls
    record = json.loads((base / "install.json").read_text(encoding="utf-8"))
    assert record["uv_private"] is private


@pytest.mark.skipif(SH is None or ON_WINDOWS, reason="needs a POSIX sh with POSIX paths")
def test_install_sh_writes_the_path_block_when_only_the_calling_app_had_the_folder(tmp_path: Path) -> None:
    # The Mac test: the installer ran under an app whose PATH had
    # ~/.local/bin; the user's zsh start-up files did not. A block now goes
    # into the start-up files, the output names them and says to open a new
    # window, and install.json lists them for `edmars uninstall`. Run again
    # from that new window, it changes nothing and still lists them.
    base, home = tmp_path / "base", tmp_path / "home"
    bin_dir = home / ".local" / "bin"
    bin_dir.mkdir(parents=True)
    uv = tmp_path / "own" / "uv"
    uv.parent.mkdir(parents=True)
    uv.write_text(_FAKE_UV, encoding="utf-8")
    uv.chmod(0o755)
    env = {"FAKE_UV_LOG": str(tmp_path / "uv.log"), "FAKE_BASE_PY": str(tmp_path / "python3"),
           "HOME": str(home), "SHELL": "/bin/zsh",
           "PATH": f"{bin_dir}{os.pathsep}{uv.parent}{os.pathsep}{os.environ.get('PATH', '')}"}
    args = ("--yes", "--no-onboard", "--from-local", str(REPO_ROOT), "--dir", str(base))

    result = _run_sh(tmp_path, *args, **env)
    assert result.returncode == 0, result.stdout + result.stderr
    out = result.stdout
    assert f"  7. Add {bin_dir} to your PATH (your account only)" in out
    assert f"Added {bin_dir} to your PATH in:\n      {home}/.profile\n      {home}/.zshrc\n" in out
    assert "To use EDM-ARS, type:  edmars   (in a NEW terminal window" in out
    block = (home / ".zshrc").read_text(encoding="utf-8")
    assert '# >>> edm-ars >>>' in block and 'export PATH="$HOME/.local/bin:$PATH"' in block
    record = json.loads((base / "install.json").read_text(encoding="utf-8"))
    assert record["path_modified"] is True
    assert record["path_files"] == [str(home / ".profile"), str(home / ".zshrc")]

    result = _run_sh(tmp_path, *args, **env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"  7. Leave your PATH as it is: {home}/.zshrc already puts {bin_dir} on it." in result.stdout
    assert block == (home / ".zshrc").read_text(encoding="utf-8")
    assert "To use EDM-ARS, type:  edmars\n" in result.stdout
    record = json.loads((base / "install.json").read_text(encoding="utf-8"))
    assert record["path_modified"] is True
    assert record["path_files"] == [str(home / ".profile"), str(home / ".zshrc")]


def test_install_ps1_keeps_its_own_uvs_cache_in_the_install_folder() -> None:
    # The same rule in the Windows installer, checked by text (the real
    # run needs the network): only a private uv gets the cache setting.
    ps1 = _text(INSTALL_PS1)
    block = re.search(r"if \(\$uvPrivate\) \{(?P<body>.*?)\n        \}", ps1, re.DOTALL)
    assert block is not None
    assert "Set-TempEnv 'UV_CACHE_DIR' (Join-Path $base 'uv\\cache')" in block.group("body")
    assert ps1.index("Set-TempEnv 'UV_CACHE_DIR'") < ps1.index("Installing Python $PythonSeries")
