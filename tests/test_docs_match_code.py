"""The install and privacy documents state facts the code decides.

A document that drifts from the code misleads exactly the reader who
trusts it. Each check below ties one published statement to its source:

* the README's R package line to the packages r_helpers/*.R actually load
  (it listed three the helpers never use and omitted jsonlite, which all
  five call);
* the README's Python range to the dowhy pin that caps it;
* the README's layout block and data filenames to the repository and the
  dataset adapters;
* every third-party import in src/ to a requirements file;
* PRIVACY.md's "characters per failed attempt" to what the DataEngineer
  and Analyst repair prompts really carry.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
PRIVACY = ROOT / "PRIVACY.md"
DISCLAIMER = ROOT / "DISCLAIMER.md"

#: Packages that ship with every R installation (base + recommended).
R_BUILTIN = frozenset({
    "base", "compiler", "datasets", "graphics", "grDevices", "grid",
    "methods", "parallel", "splines", "stats", "stats4", "tcltk", "tools",
    "utils", "boot", "class", "cluster", "codetools", "foreign",
    "KernSmooth", "lattice", "MASS", "Matrix", "mgcv", "nlme", "nnet",
    "rpart", "spatial", "survival",
})

#: Import name -> distribution name, where they differ.
IMPORT_TO_DIST = {
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
    "dotenv": "python-dotenv",
    "fitz": "pymupdf",
    "imblearn": "imbalanced-learn",
}


def _readme() -> str:
    return README.read_text(encoding="utf-8")


def _requirement_names(path: Path) -> set[str]:
    names: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        m = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", line)
        if m:
            names.add(m.group(0).lower().replace("_", "-"))
    return names


# ---------------------------------------------------------------------------
# Prerequisites
# ---------------------------------------------------------------------------


def _packages_used_by_r_helpers() -> set[str]:
    used: set[str] = set()
    for script in (ROOT / "r_helpers").glob("*.R"):
        # Drop comments: they mention things like `gates.py::` that are not
        # R packages.
        text = "\n".join(
            line.split("#", 1)[0]
            for line in script.read_text(encoding="utf-8").splitlines()
        )
        used |= set(re.findall(r"\b(?:library|require)\(\s*[\"']?([A-Za-z][\w.]*)", text))
        used |= set(re.findall(r"requireNamespace\(\s*[\"']([A-Za-z][\w.]*)", text))
        used |= set(re.findall(r"\b([A-Za-z][\w.]*):::?[A-Za-z_.]", text))
    return used


def test_readme_r_packages_are_exactly_what_the_helpers_load() -> None:
    m = re.search(r"install\.packages\(c\(([^)]*)\)\)", _readme())
    assert m, "README lost its install.packages(...) line"
    documented = set(re.findall(r"\"([^\"]+)\"", m.group(1)))
    needed = _packages_used_by_r_helpers() - R_BUILTIN
    assert needed, "found no R package use in r_helpers/*.R; the scan is broken"
    assert needed <= documented, f"README omits {sorted(needed - documented)}"
    assert documented <= needed, (
        f"README asks users to install packages nothing loads: "
        f"{sorted(documented - needed)}"
    )


def test_readme_python_range_matches_the_dowhy_cap() -> None:
    # dowhy<0.13 declares Requires-Python <3.13, so the documented range must
    # stop at 3.12 for as long as that pin stands.
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert re.search(r"^dowhy>=[\d.]+,<0\.13\b", req, re.M)
    readme = _readme()
    assert "Python 3.11 or 3.12" in readme
    assert "3.11 or newer" not in readme
    assert "3.11 or 3.12" in req


def test_readme_layout_block_names_only_paths_that_exist() -> None:
    text = _readme()
    section = text.split("## Repository layout", 1)[1]
    block = section.split("```", 2)[1]
    top_level = [
        line.split()[0]
        for line in block.splitlines()
        if line and not line.startswith(" ")
    ]
    assert top_level, "could not parse the layout block"
    missing = [p for p in top_level if not (ROOT / p).exists()]
    assert not missing, f"README layout lists paths that do not exist: {missing}"


def test_readme_names_every_dataset_file_the_adapters_expect() -> None:
    from src.dataset_adapter import _DATASET_REGISTRY, create_dataset_adapter

    text = _readme()
    for name in _DATASET_REGISTRY:
        expected = "data/raw/" + create_dataset_adapter(name).get_raw_data_filename()
        assert expected in text, f"README does not give the path {expected}"


def test_readme_gives_the_lsar_repository_and_install_step() -> None:
    text = _readme()
    assert "https://github.com/cgpan/LSAR-public" in text
    assert "pip install -r requirements-lsar.txt" in text
    assert "LSAR_HOME" in text


def test_readme_documents_every_contract_exit_code() -> None:
    section = _readme().split("## When a run ends", 1)[1].split("\n## ", 1)[0]
    for code in range(6):
        assert re.search(rf"^\| `{code}` \|", section, re.M), f"exit code {code}"


# ---------------------------------------------------------------------------
# Requirements
# ---------------------------------------------------------------------------


def test_every_third_party_import_in_src_is_declared() -> None:
    stdlib = set(sys.stdlib_module_names)
    local = {p.stem for p in (ROOT / "src").rglob("*.py")}
    # lsar is the companion repository, found through LSAR_HOME, with its
    # own dependencies in requirements-lsar.txt.
    allowed = {"src", "lsar"} | local
    declared = _requirement_names(ROOT / "requirements.txt")

    undeclared: dict[str, str] = {}
    for path in sorted((ROOT / "src").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                top = name.split(".")[0]
                if top in stdlib or top in allowed:
                    continue
                dist = IMPORT_TO_DIST.get(top, top).lower().replace("_", "-")
                if dist not in declared:
                    undeclared.setdefault(top, str(path.relative_to(ROOT)))
    assert not undeclared, (
        "imported in src/ but missing from requirements.txt: "
        + ", ".join(f"{m} ({p})" for m, p in sorted(undeclared.items()))
    )


def test_dev_and_lsar_requirements_exist_and_cover_their_purpose() -> None:
    dev = _requirement_names(ROOT / "requirements-dev.txt")
    assert {"pytest", "ruff", "mypy"} <= dev
    lsar = _requirement_names(ROOT / "requirements-lsar.txt")
    # Not in EDM-ARS's own dependency closure; without tenacity the in-process
    # `from lsar.pipeline import LSARPipeline` fails and the gate cannot run.
    assert {"tenacity", "pymupdf4llm", "arxiv", "jinja2", "httpx"} <= lsar


# ---------------------------------------------------------------------------
# Disclaimer and privacy
# ---------------------------------------------------------------------------


def test_disclaimer_and_privacy_are_linked_near_the_top_of_the_readme() -> None:
    assert DISCLAIMER.is_file() and PRIVACY.is_file()
    head = _readme().split("## Contents", 1)[0]
    assert "(DISCLAIMER.md)" in head and "(PRIVACY.md)" in head
    assert "AI-generated draft" in head


def _max_fix_message_payload() -> int:
    """Largest stderr + stdout excerpt either repair prompt carries."""
    from src.agents.analyst import Analyst
    from src.agents.data_engineer import DataEngineer

    err_mark, out_mark = "§", "¤"
    exec_result = {"stderr": err_mark * 20_000, "stdout": out_mark * 20_000,
                   "returncode": 1}
    sizes = []
    for cls in (DataEngineer, Analyst):
        agent = object.__new__(cls)
        try:
            message = agent._build_fix_message("pass", exec_result, 1)
        except TypeError:  # pragma: no cover - signature drift
            pytest.fail(f"{cls.__name__}._build_fix_message signature changed")
        sizes.append(message.count(err_mark) + message.count(out_mark))
    return max(sizes)


def test_privacy_states_the_real_size_of_error_excerpts() -> None:
    m = re.search(r"up to about ([\d,]+) characters per failed attempt",
                  PRIVACY.read_text(encoding="utf-8"))
    assert m, "PRIVACY.md lost its statement about error excerpts"
    stated = int(m.group(1).replace(",", ""))
    actual = _max_fix_message_payload()
    assert actual <= stated, (
        f"repair prompts carry up to {actual} characters of program output, "
        f"PRIVACY.md says {stated}"
    )
    assert stated <= actual * 1.2, (
        f"PRIVACY.md says {stated}, far above the real {actual}"
    )


def test_privacy_does_not_sell_the_key_scrub_as_isolation() -> None:
    """The scrub keeps keys out of the generated code's own environment
    (and so out of its printed output). Code running as the user can still
    read them from the parent process, the shell profile, the registry or
    .env, so the docs must not promise more, and must not advise moving
    keys out of .env as if that protected them."""
    from src.sandbox import child_env

    assert "DEEPSEEK_API_KEY" not in child_env({"DEEPSEEK_API_KEY": "x", "PATH": "p"})
    privacy = " ".join(PRIVACY.read_text(encoding="utf-8").split())
    assert "not a security barrier" in privacy
    assert "cannot read your keys" not in privacy
    assert "instead of keeping them in" not in privacy
    readme = " ".join(_readme().split())
    assert "could still read a `.env` file on disk" not in readme
    assert "That is not a barrier" in readme


def test_author_instructions_match_the_templates_and_writer() -> None:
    """The README told users to replace two placeholder names in both
    templates. The conference template's extra authors are commented out,
    so renaming them changes nothing; the journal template holds only the
    %%PLACEHOLDER:AUTHORS%% marker the Writer fills from ``paper.authors``,
    and overwriting it fails test_writer_scaffolding."""
    import yaml

    from src.agents.writer import Writer

    readme = " ".join(_readme().split())
    assert "AI_Name" not in readme and "Human_Author_Name" not in readme
    assert "`paper: authors: [...]`" in readme
    journal = (ROOT / "templates" / "paper_template_journal.tex").read_text(encoding="utf-8")
    assert r"\authorsnames{%%PLACEHOLDER:AUTHORS%%}" in journal

    lines = (ROOT / "config.yaml").read_text(encoding="utf-8").splitlines()
    start = lines.index("# paper:")
    example = "\n".join(line[2:] for line in lines[start:start + 2])
    config = yaml.safe_load(example)
    writer = object.__new__(Writer)
    writer.config = config
    assert writer._author_line() == "EDM-ARS, Your Name"


def test_readme_cost_headline_matches_how_run_cost_labels_it() -> None:
    """The headline called the $0.15 figure "measured, not estimated",
    while src/cost.py labels a run priced with an unverified rate
    ``estimated`` and the shipped config routes the outline stage to one."""
    from src.config import load_config
    from src.cost import TokenUsage, load_pricing, summarize

    readme = " ".join(_readme().split())
    assert "measured, not estimated" not in readme
    config = load_config(str(ROOT / "config.yaml"))
    models = config["deepseek"]["models"]
    calls = [
        TokenUsage(agent=agent, model=models[agent], provider="deepseek",
                   prompt_tokens=1000, completion_tokens=100)
        for agent in ("writer", "outline_agent")
    ]
    status = summarize(calls, load_pricing(config)).cost_status
    if status != "measured":
        assert status == "estimated"
        assert "not yet verified" in readme
