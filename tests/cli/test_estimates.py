"""Every screen, and the README, quote the same study times and prices (edmars/estimates.py)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

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


def _dollars(text: str) -> list[float]:
    """ "US$0.20-0.40 ... US$0.80" -> [0.2, 0.4, 0.8]: every amount, both ends of a range."""
    return [float(x) for pair in re.findall(r"US\$([\d.]+)(?:-([\d.]+))?", text) for x in pair if x]


def _readme_cost_section() -> str:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    cost = readme[readme.index("\n## Cost\n"):]
    return " ".join(cost[:cost.index("\n## ", 1)].split()).replace("–", "-")


def test_the_prices_add_up() -> None:
    # Without the review, plus every review a low score can trigger, is
    # the price with the review: off-peak and in the peak hours.
    low, top, peak_top = _dollars(estimates.COST_DEEPSEEK_WITHOUT_REVIEW)
    low_r, top_r, peak_top_r = _dollars(estimates.COST_DEEPSEEK_WITH_REVIEW)
    one_low, one_high, all_reviews, all_reviews_peak = _dollars(estimates.REVIEW_COST_DEEPSEEK)
    assert low <= low_r and one_low < one_high
    assert abs(top + all_reviews - top_r) < 0.005
    assert abs(peak_top + all_reviews_peak - peak_top_r) < 0.005
    # DeepSeek's peak rate is twice its off-peak rate.
    assert (peak_top, peak_top_r, all_reviews_peak) == pytest.approx((2 * top, 2 * top_r, 2 * all_reviews))
    assert _dollars(estimates.MANUAL_REVIEW_COST) == [one_low, one_high]


def test_the_prices_are_the_readme_cost_section() -> None:
    cost = _readme_cost_section()
    for amount in ("0.18-0.40", "0.36-0.80", "0.20-0.55", "0.40-1.10"):
        assert amount in cost, amount
    assert "US$0.20-0.40" in estimates.COST_DEEPSEEK_WITHOUT_REVIEW
    assert "US$0.80" in estimates.COST_DEEPSEEK_WITHOUT_REVIEW
    assert "US$0.20-0.55" in estimates.COST_DEEPSEEK_WITH_REVIEW
    assert "US$1.10" in estimates.COST_DEEPSEEK_WITH_REVIEW
    # One review: US$0.018-0.025 off-peak, 0.035-0.050 at peak in the README.
    assert "0.018-0.025" in cost and "0.035-0.050" in cost
    assert "stops early has still spent money" in cost


def test_the_peak_hours_are_the_ones_config_yaml_prices() -> None:
    config = yaml.safe_load((REPO_ROOT / "config.yaml").read_text(encoding="utf-8"))
    rates = config["pricing"]["per_million_tokens"]
    for model in config["deepseek"]["models"].values():
        windows = rates[model]["peak_windows_utc"]
        assert windows["days"] == ["mon", "tue", "wed", "thu", "fri"]
        assert " and ".join(windows["hours"]) + " UTC" in estimates.DEEPSEEK_PEAK_HOURS
        assert "weekday" in estimates.DEEPSEEK_PEAK_HOURS


def test_no_screen_keeps_an_old_price() -> None:
    stale = re.compile(r"0\.05-0\.20|US\$0\.01 per review|about US\$0\.06 in all|a few cents")
    for path in [*EDMARS.glob("*.py"), EDMARS / "messages.yaml", REPO_ROOT / "README.md"]:
        text = path.read_text(encoding="utf-8")
        assert not stale.search(text), path.name
    for text in (estimates.COST_DEEPSEEK_WITHOUT_REVIEW, estimates.COST_DEEPSEEK_WITH_REVIEW,
                 estimates.REVIEW_COST_DEEPSEEK, study.COST_OTHER):
        text.encode("ascii")
    assert estimates.STOPPED_EARLY in study.COST_OTHER
