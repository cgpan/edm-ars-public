"""V3.0 Phase 3b.6 / 6.3 — verify DoWhy is installed at runtime.

G5's sensitivity-unmeasured-confounding skill calls for DoWhy refuters
(``random_common_cause``, ``placebo_treatment_refuter``). In the 3b.5
smoke test these were skipped with "DoWhy unavailable at runtime;
refuter not executed", which LSAR flagged as a MAJOR weakness.

The 3b.6 unblock added ``dowhy>=0.11,<0.13`` to requirements.txt. This
test asserts the runtime can import the API surface G5 uses, so 3b.7
will execute the refuters rather than degrade.
"""
from __future__ import annotations

import pytest


def test_dowhy_importable() -> None:
    """DoWhy must be installed and importable for G5 refuters to run."""
    import dowhy  # noqa: F401

    # Smoke-test the API surface G5 references: CausalModel constructor +
    # the two named refuters used in 3b.5's sensitivity_analysis.json.
    from dowhy import CausalModel  # noqa: F401
    from dowhy.causal_refuters import (  # noqa: F401
        add_unobserved_common_cause,
        placebo_treatment_refuter,
    )


def test_dowhy_version_in_pinned_range() -> None:
    """Pin range from G5 / requirements.txt: dowhy>=0.11,<0.13."""
    import dowhy

    version = tuple(int(x) for x in dowhy.__version__.split(".")[:2])
    assert (0, 11) <= version < (0, 13), (
        f"dowhy version {dowhy.__version__} outside the [0.11, 0.13) "
        f"range pinned in requirements.txt and G5 SKILL.md"
    )
