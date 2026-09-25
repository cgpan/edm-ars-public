"""Fixtures for the CLI tests.

The run/live-view/results fixtures and stand-ins live in
``_run_support.py``; importing it here makes ``run_home`` available to
every test module.
"""
from tests.cli._run_support import run_home  # noqa: F401
