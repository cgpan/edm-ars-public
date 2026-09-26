"""No test module may start an external program while it is imported.

pytest imports every test module before it runs a single test. The first
CI run had no TeX: tests/test_arc_p3_p4.py called
``subprocess.run(["pdflatex", "--version"])`` inside a class-level
``skipif``, the call raised FileNotFoundError during collection, and
pytest stopped with "Interrupted: 1 error during collection" on both
Linux and Windows -- no test ran at all.

A test that needs a command-line tool declares it with
``@pytest.mark.requires_tools("pdflatex", ...)``; tests/conftest.py
looks the tools up with shutil.which, which never starts them, and
skips the test naming the ones that are missing. A test that needs R
asks for the ``r_ready`` fixture, which starts R when the test runs.

This file scans every module under tests/ for a process launch in code
that runs at import time: module and class bodies, decorators, default
argument values, and module-level helpers those call.
"""
from __future__ import annotations

import ast
import warnings
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent

#: Functions of these modules start another program.
_LAUNCHERS_BY_MODULE: dict[str, frozenset[str]] = {
    "subprocess": frozenset({
        "run", "call", "check_call", "check_output", "Popen",
        "getoutput", "getstatusoutput",
    }),
    "os": frozenset({
        "system", "popen", "startfile",
        "execl", "execlp", "execv", "execvp", "spawnl", "spawnv",
    }),
}

#: This project's functions that start pdflatex/bibtex/biber or Rscript,
#: matched by name however they were imported.
_PROJECT_LAUNCHERS = frozenset({"compile_latex", "missing_r_packages", "run_r_script"})


#: Import-time launches that cannot stop collection, each with the reason.
#: Keyed by file and by what the scan reports, not by line number.
_ALLOWED: dict[tuple[str, str], str] = {
    ("tests/test_public_paths.py", "_origin_url() -> subprocess.run"): (
        "asks git for the origin URL to decide whether this checkout is the "
        "public mirror; git is needed to have a checkout at all, and the "
        "helper returns '' when git is missing or fails"
    ),
}


def _is_main_guard(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
    )


class _Scanner:
    """Finds process launches in the code a module runs when imported."""

    def __init__(self, tree: ast.Module) -> None:
        self.tree = tree
        # Bare names bound to a launcher: ``from subprocess import run``.
        self.bare: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in _LAUNCHERS_BY_MODULE:
                for alias in node.names:
                    if alias.name in _LAUNCHERS_BY_MODULE[node.module]:
                        self.bare.add(alias.asname or alias.name)
        self.helpers: dict[str, ast.AST] = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

    def _launch(self, call: ast.Call) -> str | None:
        func = call.func
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.attr in _LAUNCHERS_BY_MODULE.get(func.value.id, frozenset())
        ):
            return f"{func.value.id}.{func.attr}"
        if isinstance(func, ast.Name) and func.id in self.bare:
            return func.id
        if isinstance(func, ast.Name) and func.id in _PROJECT_LAUNCHERS:
            return func.id
        if isinstance(func, ast.Attribute) and func.attr in _PROJECT_LAUNCHERS:
            return func.attr
        return None

    def _calls_in(self, node: ast.AST) -> list[tuple[int, str]]:
        """Launches in *node*, following calls to this module's helpers
        one level down (a helper's body runs when an import-time call
        reaches it). Lambda bodies are skipped: defining one runs nothing."""
        found: list[tuple[int, str]] = []
        stack = [node]
        while stack:
            cur = stack.pop()
            if isinstance(cur, ast.Lambda):
                continue
            if isinstance(cur, ast.Call):
                name = self._launch(cur)
                if name:
                    found.append((cur.lineno, name))
                elif isinstance(cur.func, ast.Name) and cur.func.id in self.helpers:
                    for sub in ast.walk(self.helpers[cur.func.id]):
                        if isinstance(sub, ast.Call) and self._launch(sub):
                            found.append(
                                (cur.lineno, f"{cur.func.id}() -> {self._launch(sub)}")
                            )
            stack.extend(ast.iter_child_nodes(cur))
        return found

    def _import_time(self, body: list[ast.stmt]) -> list[tuple[int, str]]:
        found: list[tuple[int, str]] = []
        for stmt in body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for expr in stmt.decorator_list:
                    found += self._calls_in(expr)
                for expr in stmt.args.defaults + stmt.args.kw_defaults:
                    if expr is not None:
                        found += self._calls_in(expr)
            elif isinstance(stmt, ast.ClassDef):
                for expr in stmt.decorator_list + stmt.bases:
                    found += self._calls_in(expr)
                found += self._import_time(stmt.body)
            elif not _is_main_guard(stmt):
                found += self._calls_in(stmt)
        return found

    def offenders(self) -> list[tuple[int, str]]:
        return sorted(self._import_time(self.tree.body))


def _scan(source: str) -> list[tuple[int, str]]:
    with warnings.catch_warnings():
        # Some test sources hold invalid escapes (\e in LaTeX strings);
        # parsing them warns, which is not what this test is about.
        warnings.simplefilter("ignore")
        tree = ast.parse(source)
    return _Scanner(tree).offenders()


def test_no_test_module_starts_a_program_while_it_is_imported() -> None:
    offenders: list[str] = []
    for path in sorted(TESTS.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(TESTS.parent).as_posix()
        for line, name in _scan(path.read_text(encoding="utf-8")):
            if (rel, name) not in _ALLOWED:
                offenders.append(f"{rel}:{line}: {name}")
    assert not offenders, (
        "These run a program while pytest collects the tests, so a machine "
        "without that program cannot collect the suite. Use "
        "@pytest.mark.requires_tools(...) and call the tool inside the test:\n  "
        + "\n  ".join(offenders)
    )


def test_every_allowed_launch_is_still_there() -> None:
    """An allowance whose launch has gone must go too, or it would
    silently cover a new launch that happens to get the same name."""
    for (rel, name), _reason in _ALLOWED.items():
        found = _scan((TESTS.parent / rel).read_text(encoding="utf-8"))
        assert name in {n for _, n in found}, (rel, name)


def test_the_scan_catches_the_skipif_that_stopped_ci() -> None:
    source = (
        "import subprocess\nimport pytest\n\n"
        "class TestStrippedCitationStillCompiles:\n"
        "    @pytest.mark.skipif(\n"
        "        subprocess.run(['pdflatex', '--version'], capture_output=True)"
        ".returncode != 0,\n"
        "        reason='pdflatex not available',\n"
        "    )\n"
        "    def test_compiles(self):\n"
        "        pass\n"
    )
    assert _scan(source) == [(6, "subprocess.run")]


def test_the_scan_follows_a_module_helper_and_a_bare_import() -> None:
    source = (
        "from subprocess import check_output as co\n"
        "def _has_tex():\n"
        "    return co(['kpsewhich', '--version'])\n"
        "HAS_TEX = _has_tex()\n"
    )
    assert _scan(source) == [(4, "_has_tex() -> co")]


def test_the_scan_catches_an_r_probe_at_import() -> None:
    """The shape tests/test_v4_psychometrics.py had: every collection,
    even of one unrelated test, started R to decide a skipif."""
    source = (
        "import pytest\n"
        "try:\n"
        "    from src.r_bridge import missing_r_packages\n"
        "    _MISSING_R = missing_r_packages()\n"
        "except Exception:\n"
        "    _MISSING_R = ['R']\n"
        "needs_r = pytest.mark.skipif(bool(_MISSING_R), reason='no R')\n"
    )
    assert _scan(source) == [(4, "missing_r_packages")]


def test_the_scan_leaves_calls_inside_tests_alone() -> None:
    source = (
        "import subprocess\nimport shutil\nimport pytest\n\n"
        "TEX = shutil.which('pdflatex')\n"
        "run = lambda: subprocess.run(['pdflatex'])\n"
        "def test_compiles(tmp_path):\n"
        "    subprocess.run(['pdflatex', 'p.tex'], cwd=tmp_path)\n"
        "class TestIt:\n"
        "    def test_it(self):\n"
        "        subprocess.check_output(['bibtex', 'p'])\n"
        "if __name__ == '__main__':\n"
        "    subprocess.run(['pytest'])\n"
    )
    assert _scan(source) == []


@pytest.mark.requires_tools("edm-ars-no-such-tool-0123")
def test_requires_tools_skips_a_test_whose_tool_is_missing() -> None:
    """Reported as skipped when the marker works; fails when it does not."""
    pytest.fail("requires_tools ran a test whose tool is not on PATH")
