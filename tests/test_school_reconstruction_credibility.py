"""Pseudo-school reconstruction must report what makes it judgeable.

A10. Reviewers called the reconstruction "not credible as described" in
four of five papers, and they were right: the manuscripts asserted a
school-aware evaluation design without giving a reader any means to
assess whether the clusters resembled schools.

The decisive number is the SINGLETON SHARE, and nothing was reporting it.
A cluster count on its own is actively misleading here: a fully
degenerate reconstruction — every student in a cluster of one — produces
about 1,000 clusters against an expected 944, a ratio of 1.06 that looks
healthy while the "school-aware" split is indistinguishable from a random
one, and the generalisation claim it supports is empty.

Because SCH_ID is suppressed there is nothing to validate against, so the
artifact now says so in the artifact itself, where a Writer cannot omit
it by accident.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis_helpers import reconstruct_school_ids

FP = ["X1SCHOOLCLI", "X1COUPERTEA"]


def _schools(n_schools: int, per_school: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {v: np.repeat(rng.normal(size=n_schools), per_school) for v in FP}
    )


def _all_unique(n: int, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({v: rng.normal(size=n) for v in FP})


def test_a_healthy_reconstruction_reports_no_singletons() -> None:
    _, meta = reconstruct_school_ids(
        _schools(40, 25), fingerprint_vars=FP, expected_n_schools=40
    )
    assert meta["n_clusters"] == 40
    assert meta["singleton_cluster_pct"] == 0.0
    assert meta["n_clusters_vs_expected_ratio"] == 1.0


def test_a_degenerate_reconstruction_is_exposed_by_the_singleton_share() -> None:
    """The count looks fine. The singleton share does not.

    This is the whole reason the metric exists.
    """
    _, meta = reconstruct_school_ids(
        _all_unique(1000), fingerprint_vars=FP, expected_n_schools=944
    )
    assert meta["singleton_cluster_pct"] == 100.0
    # And the misleading part: the ratio alone suggests success.
    assert 0.9 <= meta["n_clusters_vs_expected_ratio"] <= 1.2, (
        "the cluster count is close to expected while the clustering is "
        "worthless -- which is exactly why the count cannot be the only "
        "thing reported"
    )


def test_a_partly_degenerate_reconstruction_is_quantified() -> None:
    """The observed run: most clusters held a single student."""
    mixed = pd.concat(
        [_schools(10, 20, seed=2), _all_unique(800, seed=3)], ignore_index=True
    )
    _, meta = reconstruct_school_ids(
        mixed, fingerprint_vars=FP, expected_n_schools=944
    )
    assert meta["singleton_cluster_pct"] > 50.0


def test_the_artifact_states_it_cannot_be_validated() -> None:
    """SCH_ID is suppressed, so there is nothing to check against.

    Recorded in the artifact so a Writer cannot omit it by accident.
    """
    _, meta = reconstruct_school_ids(
        _schools(20, 25), fingerprint_vars=FP, expected_n_schools=20
    )
    assert meta["validated_against_ground_truth"] is False
    note = meta["validation_note"].lower()
    assert "suppressed" in note
    assert "provisional" in note
    assert "generalis" in note or "generaliz" in note


def test_no_fingerprint_variables_is_still_survivable() -> None:
    _, meta = reconstruct_school_ids(
        pd.DataFrame({"other": [1, 2, 3]}),
        fingerprint_vars=FP,
        expected_n_schools=944,
    )
    assert meta["n_clusters"] == 0
    assert meta["validation_passed"] is False


@pytest.mark.parametrize(
    ("n_schools", "per_school", "expected_pct"),
    [(50, 20, 0.0), (100, 10, 0.0), (200, 5, 0.0)],
)
def test_evenly_sized_clusters_never_report_singletons(
    n_schools: int, per_school: int, expected_pct: float
) -> None:
    _, meta = reconstruct_school_ids(
        _schools(n_schools, per_school, seed=n_schools),
        fingerprint_vars=FP,
        expected_n_schools=n_schools,
    )
    assert meta["singleton_cluster_pct"] == expected_pct


def test_the_skill_requires_reporting_these_numbers() -> None:
    """The metric is useless if the manuscript never prints it."""
    from pathlib import Path

    skill = (
        Path(__file__).resolve().parents[1]
        / "skills" / "writing" / "hsls09-multilevel-limitations-paragraph"
        / "SKILL.md"
    )
    text = skill.read_text(encoding="utf-8")
    assert "singleton" in text.lower()
    assert "cannot be validated" in text.lower() or "unvalidated" in text.lower()
    assert "provisional" in text.lower()
