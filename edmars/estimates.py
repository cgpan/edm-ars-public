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
* Cost follows the same instrumented run ("Cost" in the README): the
  six agents cost US$0.091 and six sampled reviews US$0.055, about a
  cent each.

ASCII only, so --plain output is safe.
"""

from __future__ import annotations

#: A study without the automated review.
TIME_WITHOUT_REVIEW = "usually 10-35 minutes"

#: A study with the automated review (the README's "complete gated paper").
TIME_WITH_REVIEW = (
    "usually 20-60 minutes with the automated review, occasionally about 2 hours"
)

#: What the automated review adds to a study (one to six reviews).
REVIEW_TIME = "about 10-45 minutes"

#: `edmars review` on a finished paper (one standalone LSAR review),
#: said as "usually takes ...".
MANUAL_REVIEW_TIME = "10-40 minutes"

#: One review with DeepSeek, and the most a study's reviews add up to
#: (median sampling over two rounds can use up to 6 reviews).
REVIEW_COST_DEEPSEEK = (
    "about US$0.01 per review with DeepSeek (a low score can trigger up to 6 "
    "reviews, about US$0.06 in all)"
)

#: A whole study with DeepSeek, review included.
COST_DEEPSEEK = (
    "roughly US$0.05-0.20 per study with DeepSeek, including the automated "
    "review (measured on only a few runs). You pay DeepSeek directly."
)
