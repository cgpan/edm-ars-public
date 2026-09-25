"""Every screen, and the README, quote the same study times (edmars/estimates.py)."""

from __future__ import annotations

import re
from pathlib import Path

from edmars import estimates, study
from edmars.endstates import messages

REPO_ROOT = Path(__file__).resolve().parents[2]
EDMARS = REPO_ROOT / "edmars"


def _ranges(text: str) -> list[str]:
    """ "usually 20-60 minutes" -> ["20-60"]; en dashes and "to" count as "-"."""
    text = text.replace("–", "-")
    text = re.sub(r"(\d+) to (\d+)", r"\1-\2", text)
    return re.findall(r"\d+-\d+(?= minutes)", text)


def test_the_readme_quotes_the_cli_times() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    note = " ".join(readme.split("**Version 5.**", 1)[1].split("\n\n", 1)[0].split())
    note = note.replace("> ", "")
    assert _ranges(estimates.TIME_WITH_REVIEW)[0] in _ranges(note)
    assert _ranges(estimates.TIME_WITHOUT_REVIEW)[0] in _ranges(note)
    assert "occasionally about 2 hours" in note and "2 hours" in estimates.TIME_WITH_REVIEW


def test_the_card_and_the_live_view_use_the_shared_times() -> None:
    assert study.TIME_WITH_REVIEW == estimates.TIME_WITH_REVIEW
    assert study.TIME_WITHOUT_REVIEW == estimates.TIME_WITHOUT_REVIEW
    reviewing = str(messages()["stages"]["REVIEWING"]["now"])
    assert _ranges(reviewing) == _ranges(estimates.REVIEW_TIME)


def test_no_screen_keeps_an_old_review_time() -> None:
    stale = re.compile(r"20(-| to )40 minutes|35-60 minutes|18.46 minutes")
    for path in [*EDMARS.glob("*.py"), EDMARS / "messages.yaml", REPO_ROOT / "README.md"]:
        text = path.read_text(encoding="utf-8")
        assert not stale.search(text), path.name
