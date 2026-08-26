"""Writer scaffolding must never reach the manuscript.

A7, observed in all five journal-track papers. Each compiled PDF carried:

  * 6 ``SUMMARY:`` paragraphs — the sectionwise writer's own handoff notes
    ("This section motivated the study by ..."). The prompt asks for them,
    to give the next section context; nothing stripped them afterwards.
  * 6-11 markdown fences (```` ```latex ````) pasted into the .tex.
  * 6 duplicated ``\\section{X}`` headings — emitted once by the
    assembler, again inside the fenced block.

LSAR called them "repeated auto-generated summaries" and
"placeholder-like text", which is what a human reviewer would conclude
too.

Also pinned here: the byline. The public template shipped
``\\authorsnames{EDM-ARS, AI\\_Name, Human\\_Author\\_Name}`` and printed
those placeholders unfilled.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import patch

import pytest

from src.agents.writer import Writer, strip_scaffolding
from src.context import PipelineContext

ROOT = Path(__file__).resolve().parents[1]


def _writer(tmp_path: Path, config_extra: dict | None = None) -> Writer:
    config = {
        "models": {"writer": "deepseek-v4-pro"},
        "llm_provider": "deepseek",
        "deepseek": {"models": {"writer": "deepseek-v4-pro"}},
        "pipeline": {"random_state": 42},
        "paths": {"agent_prompts": "agent_prompts/", "data_registry": "data_registry/"},
        "sandbox": {"enabled": False},
        "writer": {"outline_first": True},
    }
    config.update(config_extra or {})
    ctx = PipelineContext(
        dataset_name="els_2002",
        raw_data_path=str(tmp_path / "raw.csv"),
        output_dir=str(tmp_path),
    )
    with patch("anthropic.Anthropic"):
        return Writer(ctx, "writer", config)


#: A section as the writer actually emitted it.
DIRTY = """\\section{Introduction}

```latex
\\section{Introduction}
Postsecondary enrolment has risen steadily.
```

SUMMARY: This section motivated the study by framing the enrolment gap.

\\section{Methods}
We used ELS:2002.
"""


def test_all_three_artefacts_are_removed() -> None:
    clean, stats = strip_scaffolding(DIRTY)
    assert stats == {"fences": 2, "summaries": 1, "duplicate_headings": 1}
    assert "```" not in clean
    assert "SUMMARY:" not in clean
    assert clean.count("\\section{Introduction}") == 1


def test_real_prose_survives() -> None:
    """A stripper that eats content is worse than the scaffolding."""
    clean, _ = strip_scaffolding(DIRTY)
    assert "Postsecondary enrolment has risen steadily." in clean
    assert "We used ELS:2002." in clean
    assert "\\section{Methods}" in clean


@pytest.mark.parametrize(
    "fence", ["```latex", "```tex", "```python", "```", "   ```latex   "]
)
def test_every_fence_flavour_is_stripped(fence: str) -> None:
    clean, stats = strip_scaffolding(f"\\section{{X}}\n{fence}\nprose\n")
    assert stats["fences"] >= 1
    assert "```" not in clean


def test_distinct_consecutive_headings_are_kept() -> None:
    """Only an IMMEDIATELY REPEATED heading is a duplicate."""
    text = "\\section{Methods}\n\\section{Results}\nprose\n"
    clean, stats = strip_scaffolding(text)
    assert stats["duplicate_headings"] == 0
    assert "\\section{Methods}" in clean and "\\section{Results}" in clean


def test_subsection_duplicates_are_also_collapsed() -> None:
    text = "\\subsection{Sample}\n\\subsection{Sample}\nprose\n"
    clean, stats = strip_scaffolding(text)
    assert stats["duplicate_headings"] == 1
    assert clean.count("\\subsection{Sample}") == 1


def test_the_word_summary_in_prose_is_not_stripped() -> None:
    """Only a line STARTING with SUMMARY: is a handoff note."""
    text = "\\section{Results}\nIn summary, the model performed well.\n"
    clean, stats = strip_scaffolding(text)
    assert stats["summaries"] == 0
    assert "In summary, the model performed well." in clean


def test_empty_input_is_safe() -> None:
    clean, stats = strip_scaffolding("")
    assert clean == ""
    assert not any(stats.values())


def test_clean_latex_is_left_alone() -> None:
    text = "\\section{Introduction}\nProse without scaffolding.\n"
    clean, stats = strip_scaffolding(text)
    assert not any(stats.values())
    assert clean.strip() == text.strip()


# --- byline -----------------------------------------------------------

def test_journal_template_has_no_hardcoded_byline() -> None:
    tex = (ROOT / "templates" / "paper_template_journal.tex").read_text(
        encoding="utf-8"
    )
    match = re.search(r"\\authorsnames\{([^}]*)\}", tex)
    assert match, "journal template has no authorsnames line"
    # Equality already excludes every possible name, so the old
    # belt-and-braces list of specific leaked strings added nothing -- and
    # one of them was the owner's real name, which made this file itself a
    # place the name appeared. The public mirror's own hygiene test caught
    # it once the file became tracked.
    assert match.group(1) == "%%PLACEHOLDER:AUTHORS%%"


def test_byline_defaults_to_the_system_alone(tmp_path: Path) -> None:
    """A human who wants credit should say so, not inherit a template name."""
    assert _writer(tmp_path)._author_line() == "EDM-ARS"


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("A. Researcher", "A. Researcher"),
        (["A. Researcher", "B. Colleague"], "A. Researcher, B. Colleague"),
        ("", "EDM-ARS"),
        ([], "EDM-ARS"),
        (["  "], "EDM-ARS"),
        (None, "EDM-ARS"),
    ],
    ids=["string", "list", "empty-string", "empty-list", "blank-entry", "unset"],
)
def test_configured_authors_are_used(
    tmp_path: Path, configured: object, expected: str
) -> None:
    writer = _writer(tmp_path, {"paper": {"authors": configured}})
    assert writer._author_line() == expected
