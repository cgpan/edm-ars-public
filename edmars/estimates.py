"""How long a study takes and what it costs, in one place.

Every screen that quotes a duration or a price (the confirmation card,
setup, the reviewer check, `edmars review`, the live view's review step)
reads it from here, so they cannot contradict each other again. The
README's "Version 5" note quotes the same times, and
tests/cli/test_estimates.py keeps the two in step.

They are rough ranges from a handful of DeepSeek runs, not promises:

* A complete gated paper (with the automated review) has taken from
  about 18 minutes to about an hour; one review takes about 7 minutes,
  and a low score can trigger up to 6 plus a revision, about 45 minutes
  on the README's instrumented run.
* Cost follows the README's "Cost" section: the token counts of 11
  archived papers and 6 automated reviews, priced at DeepSeek's rates
  of 2026-09-26 (the ones config.yaml ships) with the default model
  routing. DeepSeek bills twice as much in its weekday peak hours as at
  any other time, so every price says both. A study alone came to
  US$0.18-0.40 off-peak (median 0.25), one review US$0.018-0.025, and
  the six reviews a low score can trigger up to about US$0.15; at peak,
  twice that. Journal-format and psychometric studies have not been
  measured. A study that stops early has still spent money: the Mac
  test's study, stopped by the pre-review checks after its analysis,
  cost about US$0.16 off-peak.

ASCII only, so --plain output is safe.
"""

from __future__ import annotations

#: A study without the automated review.
TIME_WITHOUT_REVIEW = "usually 10-35 minutes"

#: A study with the automated review (the README's "complete gated paper").
TIME_WITH_REVIEW = (
    "usually 20-60 minutes with the automated review, occasionally about 2 hours"
)

#: Said after either time. The ranges are for studies that reach a
#: paper; both Mac test studies were stopped by the checks, after 14 min
#: (2026-09-26) and 8 min 47 s (2026-09-27), well under them. Two early
#: stops are no reason to change the ranges of a complete study, but the
#: card must not suggest a stopped study has hung.
ENDS_SOONER = "A study that stops early ends sooner."

#: What the automated review adds to a study (one to six reviews).
REVIEW_TIME = "about 10-45 minutes"

#: `edmars review` on a finished paper (one standalone LSAR review),
#: said as "usually takes ...".
MANUAL_REVIEW_TIME = "10-40 minutes"

#: When DeepSeek charges its peak (double) rate: config.yaml's
#: ``peak_windows_utc`` for the DeepSeek models, in words.
DEEPSEEK_PEAK_HOURS = (
    "DeepSeek's weekday peak hours, 01:00-04:00 and 06:00-10:00 UTC"
)

#: Said with every price for a paid AI service.
STOPPED_EARLY = "A study that stops early still costs what it used up to then."

#: One review with DeepSeek, and the most a study's reviews add up to
#: (median sampling over two rounds can use up to 6 reviews).
REVIEW_COST_DEEPSEEK = (
    "about US$0.02-0.05 per review with DeepSeek; a low score can trigger up "
    "to 6 reviews, up to about US$0.15 in all (US$0.30 in "
    + DEEPSEEK_PEAK_HOURS + ")"
)

#: `edmars review` on a finished paper: one review.
MANUAL_REVIEW_COST = "about US$0.02-0.05 of DeepSeek credit"

#: A whole study with DeepSeek, without the automated review.
COST_DEEPSEEK_WITHOUT_REVIEW = (
    "roughly US$0.20-0.40 per study with DeepSeek, up to about US$0.80 in "
    + DEEPSEEK_PEAK_HOURS + ". " + STOPPED_EARLY + " You pay DeepSeek directly."
)

#: A whole study with DeepSeek, review included (up to 6 reviews).
COST_DEEPSEEK_WITH_REVIEW = (
    "roughly US$0.20-0.55 per study with DeepSeek, including the automated "
    "review, up to about US$1.10 in " + DEEPSEEK_PEAK_HOURS + ". "
    + STOPPED_EARLY + " You pay DeepSeek directly."
)


def cost_deepseek(review: bool) -> str:
    """The confirmation card's DeepSeek price for a study with or without
    the automated review."""
    return COST_DEEPSEEK_WITH_REVIEW if review else COST_DEEPSEEK_WITHOUT_REVIEW
