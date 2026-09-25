"""install/install.sh and install/install.ps1, offline.

A real install downloads uv, Python and ~1.5 GB of packages, so it runs in
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
                     "Get-ShLauncherText")
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

    study = r"C:\Users\someone\EDM-ARS\studies\2026-09-25_1200_gpa_ab12"
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


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck is not installed")
def test_install_sh_passes_shellcheck() -> None:
    result = subprocess.run(["shellcheck", "-s", "sh", str(INSTALL_SH)],
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
