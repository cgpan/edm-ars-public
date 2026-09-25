"""The only place in ``edmars`` that starts or stops other programs.

Keeping every process call in one module mirrors the pipeline's rule
that subprocess calls live only in ``src/sandbox.py``: there is one place
to audit for ``shell=True`` (never used), for how output is decoded
(always UTF-8 with replacement, so a stray byte cannot crash a check),
and for how children are cleaned up.

Two details matter more than they look:

* :func:`run` enforces its timeout on the whole process TREE. The stock
  ``subprocess.run(timeout=...)`` kills only the direct child, and when a
  grandchild (``pdflatex`` under a wrapper, ``R`` under ``Rscript``)
  still holds the output pipes, the call then blocks forever.
* :func:`spawn_detached` starts the pipeline so that closing the terminal
  does not stop the study: a new process group/session, no console, and
  on Windows an attempt to leave any job object the terminal runs in.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import psutil

StrPath = str | os.PathLike[str]

#: Children started by this process, so :func:`pid_alive` can reap them
#: (an unreaped POSIX child is a zombie that still "exists").
_CHILDREN: dict[int, subprocess.Popen[Any]] = {}

_IS_WINDOWS = sys.platform == "win32"

#: pids of keep-awake helpers started by :func:`keep_awake` (for tests).
_KEEP_AWAKE_HELPERS: list[int] = []


def _argv(args: Sequence[StrPath]) -> list[str]:
    argv = [os.fspath(a) for a in args]
    if not argv:
        raise ValueError("no command given")
    return argv


def run(
    args: Sequence[StrPath],
    *,
    timeout: float | None = None,
    env: Mapping[str, str] | None = None,
    cwd: StrPath | None = None,
    input: str | None = None,  # noqa: A002 - mirrors subprocess.run
) -> subprocess.CompletedProcess[str]:
    """Run a program to completion and capture its output as text.

    Output is decoded as UTF-8 with replacement characters. On timeout
    the whole process tree is stopped and ``subprocess.TimeoutExpired``
    is raised (with whatever output was captured). A missing program
    raises ``FileNotFoundError``; use :func:`which` first when a missing
    tool is an expected situation.
    """
    argv = _argv(args)
    with subprocess.Popen(
        argv,
        stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=dict(env) if env is not None else None,
        cwd=os.fspath(cwd) if cwd is not None else None,
        shell=False,
    ) as child:
        try:
            out, err = child.communicate(input=input, timeout=timeout)
        except subprocess.TimeoutExpired:
            terminate_tree(child.pid, grace_s=0)
            try:
                out, err = child.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                out, err = "", ""  # something outside the tree kept the pipes open
            raise subprocess.TimeoutExpired(argv, timeout or 0, output=out, stderr=err) from None
        except BaseException:
            terminate_tree(child.pid, grace_s=0)
            raise
    return subprocess.CompletedProcess(argv, child.returncode, out, err)


# Windows process-creation flags (defined here so the module imports on POSIX).
_CREATE_NEW_PROCESS_GROUP = 0x00000200
_DETACHED_PROCESS = 0x00000008
_CREATE_NO_WINDOW = 0x08000000
_CREATE_BREAKAWAY_FROM_JOB = 0x01000000


def spawn_detached(
    args: Sequence[StrPath],
    *,
    cwd: StrPath | None = None,
    env: Mapping[str, str] | None = None,
    log_path: Path | None = None,
) -> int:
    """Start a program that keeps running after this one exits; return its pid.

    stdout and stderr are appended to ``log_path`` (or discarded when it is
    None). On Windows the child gets its own process group, no console and
    no window, and leaves the terminal's job object when the job allows it
    (so closing an IDE terminal does not kill a study). On macOS and Linux
    it starts a new session.
    """
    argv = _argv(args)
    log: Any = subprocess.DEVNULL
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = open(log_path, "ab")  # noqa: SIM115 - handed to the child, closed below
    try:
        common: dict[str, Any] = {
            "cwd": os.fspath(cwd) if cwd is not None else None,
            "env": dict(env) if env is not None else None,
            "stdin": subprocess.DEVNULL,
            "stdout": log,
            "stderr": subprocess.STDOUT,
            "close_fds": True,
        }
        if _IS_WINDOWS:
            flags = _CREATE_NEW_PROCESS_GROUP | _DETACHED_PROCESS | _CREATE_NO_WINDOW
            try:
                child = subprocess.Popen(argv, creationflags=flags | _CREATE_BREAKAWAY_FROM_JOB, **common)
            except OSError as exc:
                # ERROR_ACCESS_DENIED: the job forbids breakaway. Start inside it.
                if getattr(exc, "winerror", None) != 5:
                    raise
                child = subprocess.Popen(argv, creationflags=flags, **common)
        else:
            child = subprocess.Popen(argv, start_new_session=True, **common)
    finally:
        if log is not subprocess.DEVNULL:
            log.close()
    _CHILDREN[child.pid] = child
    return child.pid


def which(name: str) -> str | None:
    """Full path of a program on PATH (``.exe`` etc. handled on Windows), or None."""
    return shutil.which(name)


def pid_alive(pid: int | None, *, started_at: float | None = None) -> bool:
    """True when process ``pid`` is running (zombies count as dead).

    ``started_at`` (seconds since the epoch, e.g. when the run was
    launched) guards against pid reuse: a process created well after that
    moment is some other program that inherited the number.
    """
    if not pid or pid <= 0:
        return False
    child = _CHILDREN.get(pid)
    if child is not None:
        return child.poll() is None
    try:
        proc = psutil.Process(pid)
        if proc.status() == psutil.STATUS_ZOMBIE:
            return False
        if started_at is not None and proc.create_time() > started_at + 5:
            return False
        return proc.is_running()
    except psutil.NoSuchProcess:  # includes ZombieProcess
        return False
    except psutil.AccessDenied:
        return True  # exists but belongs to someone else: alive
    except psutil.Error:
        return False


def _tree(parent: psutil.Process) -> list[psutil.Process]:
    try:
        return parent.children(recursive=True)
    except psutil.Error:
        return []


def terminate_tree(pid: int, grace_s: float = 30) -> None:
    """Stop process ``pid`` and everything it started.

    First the process gets up to ``grace_s`` seconds to finish on its own:
    on macOS/Linux it is sent SIGTERM (the pipeline saves its checkpoint
    and exits), on Windows there is no such signal, so the caller writes
    the run's STOP file before calling this. Then every remaining process
    in the tree is terminated, and killed if still alive 5 s later.
    """
    if not pid or pid <= 0 or pid == os.getpid():
        return
    try:
        parent = psutil.Process(pid)
    except psutil.Error:
        return
    family = _tree(parent)
    if grace_s > 0:
        if not _IS_WINDOWS:
            try:
                parent.send_signal(signal.SIGTERM)
            except psutil.Error:
                pass
        try:
            parent.wait(timeout=grace_s)
        except (psutil.TimeoutExpired, psutil.Error):
            pass
    # Children started during the grace period are caught by a fresh walk.
    seen = {p.pid for p in family}
    family += [p for p in _tree(parent) if p.pid not in seen]
    targets = [parent, *family]
    for proc in targets:
        try:
            proc.terminate()
        except psutil.Error:
            pass
    _gone, alive = psutil.wait_procs(targets, timeout=5)
    for proc in alive:
        try:
            proc.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(alive, timeout=5)
    child = _CHILDREN.get(pid)
    if child is not None:
        child.poll()


#: Tiny helper run as a detached process on Windows: hold a "system
#: required" power request until process argv[1] exits. A separate
#: process (rather than a thread in the watcher) keeps the computer awake
#: even after the user closes the window the study was started from.
_KEEP_AWAKE_WINDOWS = """
import ctypes, sys
from ctypes import wintypes
k = ctypes.windll.kernel32
k.OpenProcess.restype = wintypes.HANDLE
k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
k.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
k.CloseHandle.argtypes = [wintypes.HANDLE]
k.SetThreadExecutionState.restype = ctypes.c_uint
k.SetThreadExecutionState.argtypes = [ctypes.c_uint]
handle = k.OpenProcess(0x00100000, False, int(sys.argv[1]))
if handle:
    k.SetThreadExecutionState(0x80000000 | 0x00000001)
    k.WaitForSingleObject(handle, 0xFFFFFFFF)
    k.CloseHandle(handle)
"""


def keep_awake(pid: int) -> None:
    """Stop the computer from sleeping while process ``pid`` runs.

    Windows: a detached helper holds ``SetThreadExecutionState``. macOS:
    ``caffeinate -i -w <pid>``. Linux: not done (desktops differ too
    much). Best effort: never raises.
    """
    try:
        if _IS_WINDOWS:
            helper = spawn_detached([sys.executable, "-c", _KEEP_AWAKE_WINDOWS, str(int(pid))])
            _KEEP_AWAKE_HELPERS.append(helper)
        elif sys.platform == "darwin":
            caffeinate = which("caffeinate")
            if caffeinate:
                helper = spawn_detached([caffeinate, "-i", "-w", str(int(pid))])
                _KEEP_AWAKE_HELPERS.append(helper)
    except Exception:
        pass


def open_with_default_app(path: Path) -> None:
    """Open ``path`` in the default application (PDF viewer, file browser...)."""
    target = os.fspath(path)
    if _IS_WINDOWS:
        os.startfile(target)  # type: ignore[attr-defined,unused-ignore]  # noqa: S606
        return
    if sys.platform == "darwin":
        opener = which("open")
    else:
        opener = which("xdg-open")
    if not opener:
        raise FileNotFoundError("no program to open files with (xdg-open) was found")
    spawn_detached([opener, target])
