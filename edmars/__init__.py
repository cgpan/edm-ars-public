"""edmars: the command-line application for EDM-ARS.

This package is the friendly front door to the research pipeline in
``src/``. It installs nothing into the pipeline and imports none of its
side-effecting modules: it keeps settings and keys, checks the computer,
starts the pipeline as a child process and reads the run folder it writes.

Run it with ``edmars`` (the installer's launcher) or ``python -m edmars``.
"""

__version__ = "0.1.0"
