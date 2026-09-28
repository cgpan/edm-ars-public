"""Process helpers, exercised with small Python child processes (no network).

Also holds the architecture rule for the whole ``edmars`` package: only
``edmars/proc.py`` may start processes, and nothing uses a shell.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from edmars import proc

PACKAGE = Path(__file__).resolve().parents[2] / "edmars"
PY = sys.executable


def _wait_until(predicate, timeout: float = 20.0) -> bool:  # type: ignore[no-untyped-def]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.1)
    return predicate()


def test_run_captures_utf8_text() -> None:
    code = "import sys; sys.stdout.buffer.write('caf\\u00e9 ok\\n'.encode('utf-8'))"
    result = proc.run([PY, "-c", code], timeout=60)
    assert result.returncode == 0
    assert result.stdout.strip() == "café ok"


def test_run_replaces_undecodable_bytes() -> None:
    code = "import sys; sys.stdout.buffer.write(b'bad \\xff byte\\n')"
    result = proc.run([PY, "-c", code], timeout=60)
    assert "�" in result.stdout


def test_run_passes_env_cwd_and_input(tmp_path: Path) -> None:
    env = dict(os.environ, EDMARS_PROC_TEST="marker")
    code = (
        "import os, sys; print(os.environ['EDMARS_PROC_TEST']); "
        "print(os.getcwd()); print(sys.stdin.read().strip())"
    )
    result = proc.run([PY, "-c", code], env=env, cwd=tmp_path, input="typed", timeout=60)
    lines = result.stdout.splitlines()
    assert lines[0] == "marker"
    assert Path(lines[1]).resolve() == tmp_path.resolve()
    assert lines[2] == "typed"


_SESSION_PROBE = (
    "import os\n"
    "sid = os.getsid(0) if hasattr(os, 'getsid') else -1\n"
    "try:\n"
    "    open('/dev/tty').close()\n"
    "    tty = 'yes'\n"
    "except OSError:\n"
    "    tty = 'no'\n"
    "print(sid, tty)\n"
)


def test_run_can_start_a_program_without_the_terminal() -> None:
    # The TinyTeX installer may run `sudo`, whose "Password:" prompt goes to
    # the terminal while its explanation is captured. In a new session the
    # program has no terminal, so such a prompt fails at once instead.
    result = proc.run([PY, "-c", _SESSION_PROBE], timeout=60, new_session=True)
    assert result.returncode == 0, result.stderr
    if sys.platform == "win32":
        return  # accepted and ignored: Windows programs do not share a terminal this way
    sid, tty = result.stdout.split()
    assert int(sid) != os.getsid(0)
    assert tty == "no"
    same = proc.run([PY, "-c", _SESSION_PROBE], timeout=60)
    assert int(same.stdout.split()[0]) == os.getsid(0)


def test_run_reports_a_missing_program() -> None:
    with pytest.raises(FileNotFoundError):
        proc.run(["edmars-no-such-program-xyz"], timeout=10)
    with pytest.raises(ValueError):
        proc.run([])


def test_run_timeout_stops_the_whole_tree(tmp_path: Path) -> None:
    marker = tmp_path / "grandchild.pid"
    code = (
        "import subprocess, sys, time, pathlib\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        f"pathlib.Path({str(marker)!r}).write_text(str(p.pid))\n"
        "time.sleep(120)\n"
    )
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        proc.run([PY, "-c", code], timeout=5)
    assert time.monotonic() - started < 60
    assert _wait_until(marker.exists, 5)
    grandchild = int(marker.read_text())
    assert _wait_until(lambda: not proc.pid_alive(grandchild), 20)


def test_spawn_detached_logs_and_is_reaped(tmp_path: Path) -> None:
    log = tmp_path / "logs" / "console.log"
    pid = proc.spawn_detached([PY, "-c", "print('hello from child')"], cwd=tmp_path, env=None, log_path=log)
    assert pid > 0
    assert _wait_until(lambda: not proc.pid_alive(pid), 30)
    assert "hello from child" in log.read_text(encoding="utf-8", errors="replace")


def test_spawn_detached_appends_to_an_existing_log(tmp_path: Path) -> None:
    log = tmp_path / "console.log"
    log.write_text("earlier line\n", encoding="utf-8")
    pid = proc.spawn_detached([PY, "-c", "print('later line')"], cwd=tmp_path, env=None, log_path=log)
    assert _wait_until(lambda: not proc.pid_alive(pid), 30)
    text = log.read_text(encoding="utf-8")
    assert text.startswith("earlier line") and "later line" in text


def test_terminate_tree_stops_a_detached_process(tmp_path: Path) -> None:
    pid = proc.spawn_detached([PY, "-c", "import time; time.sleep(120)"], cwd=tmp_path, env=None, log_path=None)
    try:
        assert proc.pid_alive(pid)
        proc.terminate_tree(pid, grace_s=1)
        assert _wait_until(lambda: not proc.pid_alive(pid), 20)
    finally:
        proc.terminate_tree(pid, grace_s=0)


def test_pid_alive_edge_cases() -> None:
    assert not proc.pid_alive(None)
    assert not proc.pid_alive(0)
    assert not proc.pid_alive(-5)
    assert proc.pid_alive(os.getpid())
    # A process created after the recorded start time is a reused pid.
    assert not proc.pid_alive(os.getpid(), started_at=0.0)
    assert proc.pid_alive(os.getpid(), started_at=time.time())


def test_terminate_tree_never_targets_this_process() -> None:
    proc.terminate_tree(os.getpid(), grace_s=0)
    proc.terminate_tree(0)
    assert proc.pid_alive(os.getpid())


def test_which_returns_none_for_unknown_programs() -> None:
    assert proc.which("edmars-no-such-program-xyz") is None


# --- architecture rule --------------------------------------------------------------

_OS_SPAWNERS = {"system", "popen", "startfile"}
_SPAWN_MODULES = {"subprocess", "pty", "multiprocessing"}


def _spawn_sites(tree: ast.AST) -> list[int]:
    """Lines that import a process module or call an ``os`` spawner.

    Parsed rather than grepped, so a docstring that merely mentions
    subprocess is not a violation and an aliased import still is.
    """
    lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] in _SPAWN_MODULES for alias in node.names):
                lines.append(node.lineno)
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").split(".")[0]
            if module in _SPAWN_MODULES:
                lines.append(node.lineno)
            elif module == "os" and any(
                a.name in _OS_SPAWNERS or a.name.startswith(("spawn", "exec")) for a in node.names
            ):
                lines.append(node.lineno)
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == "os" and (
                node.attr in _OS_SPAWNERS or node.attr.startswith(("spawn", "exec"))
            ):
                lines.append(node.lineno)
            elif node.attr.startswith("create_subprocess"):
                lines.append(node.lineno)
    return lines


def _modules() -> list[Path]:
    return sorted(PACKAGE.rglob("*.py"))


def test_only_proc_starts_processes() -> None:
    offenders = []
    for path in _modules():
        if path.name == "proc.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        offenders += [f"{path.relative_to(PACKAGE.parent)}:{n}" for n in _spawn_sites(tree)]
    assert not offenders, "process spawning outside edmars/proc.py: " + ", ".join(offenders)


def test_the_spawn_rule_detects_violations() -> None:
    assert _spawn_sites(ast.parse("import subprocess"))
    assert _spawn_sites(ast.parse("import subprocess as sp"))
    assert _spawn_sites(ast.parse("from subprocess import run"))
    assert _spawn_sites(ast.parse("import os\nos.system('x')"))
    assert _spawn_sites(ast.parse("import os\nos.startfile('x')"))
    assert not _spawn_sites(ast.parse('"""A docstring that mentions subprocess."""'))


def test_nothing_uses_a_shell() -> None:
    offenders = []
    for path in _modules():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.keyword)
                and node.arg == "shell"
                and isinstance(node.value, ast.Constant)
                and node.value.value is True
            ):
                offenders.append(f"{path.relative_to(PACKAGE.parent)}:{node.value.lineno}")
    assert not offenders


@pytest.mark.skipif(sys.platform not in ("win32", "darwin"), reason="keep-awake is Windows/macOS only")
def test_keep_awake_helper_ends_with_the_study(tmp_path: Path) -> None:
    target = proc.spawn_detached([PY, "-c", "import time; time.sleep(120)"], cwd=tmp_path, env=None, log_path=None)
    before = len(proc._KEEP_AWAKE_HELPERS)
    try:
        proc.keep_awake(target)
        assert len(proc._KEEP_AWAKE_HELPERS) == before + 1
        helper = proc._KEEP_AWAKE_HELPERS[-1]
        assert _wait_until(lambda: proc.pid_alive(helper), 5)
        time.sleep(1.0)  # the helper is waiting on the target, not exiting early
        assert proc.pid_alive(helper)
        proc.terminate_tree(target, grace_s=0)
        assert _wait_until(lambda: not proc.pid_alive(helper), 20)
    finally:
        proc.terminate_tree(target, grace_s=0)
        for helper in proc._KEEP_AWAKE_HELPERS[before:]:
            proc.terminate_tree(helper, grace_s=0)


def test_keep_awake_never_raises() -> None:
    proc.keep_awake(-1)
