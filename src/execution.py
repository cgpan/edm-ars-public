"""Thin compatibility shim — delegates to src.sandbox.

This module is kept for backward compatibility only.  New code should import
directly from src.sandbox.
"""
from __future__ import annotations

from src.sandbox import SubprocessExecutor as _SubprocessExecutor


class CodeExecutor:
    """Backward-compatible wrapper around SubprocessExecutor.

    .. deprecated::
        Import and use src.sandbox.SubprocessExecutor (or DockerSandbox) directly.
    """

    def __init__(self, output_dir: str, timeout_s: int = 300) -> None:
        self.output_dir = output_dir
        self.default_timeout = timeout_s
        self._executor = _SubprocessExecutor()

    def run(self, code: str, timeout_s: int | None = None) -> dict:
        timeout = timeout_s if timeout_s is not None else self.default_timeout
        return self._executor.run(code=code, output_dir=self.output_dir, timeout_s=timeout)
