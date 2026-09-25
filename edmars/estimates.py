"""What a study costs, in one place.

Every screen that quotes a price (the confirmation card, setup) reads it
from here, so they cannot contradict each other again. The figures
follow the README's instrumented run ("Cost"): the six agents cost
US$0.091 and six sampled reviews US$0.055, about a cent each. They are
rough, not promises. ASCII only, so --plain output is safe.
"""

from __future__ import annotations

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
