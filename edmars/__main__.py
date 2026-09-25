"""Entry point for ``python -m edmars`` and the installer's launcher.

The standard streams are switched to UTF-8 before anything is printed.
Without this, a Windows console that still uses a legacy code page (and
any redirected output there) raises ``UnicodeEncodeError`` on the first
check mark or quotation mark, and a novice sees a traceback instead of
the setup wizard. ``errors="replace"`` means an unprintable character
becomes ``?`` rather than a crash.
"""

from __future__ import annotations

import sys


def _force_utf8_streams() -> None:
    """Reconfigure stdout and stderr to UTF-8 where the stream allows it."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue  # replaced by a test harness or an embedding host
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass  # a detached or closed stream: nothing useful to do


def main() -> None:
    """Run the ``edmars`` command line and exit with its status."""
    _force_utf8_streams()
    from edmars.cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
