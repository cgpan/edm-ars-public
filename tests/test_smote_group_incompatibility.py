"""SMOTE must decline when a group-aware split or CV is in use.

A6, observed on a real run. At an 11% base rate SMOTE oversampled X and y
to 33,266 rows, then the ORIGINAL 18,717 school IDs were handed to a
grouped splitter::

    ValueError: Found input variables with inconsistent numbers of
    samples: [33266, 33266, 18717]

The tempting fix — pad the group vector — is worse than the crash. A
synthetic row interpolated between two students belongs to no school. Any
label given to it is invented, and it would leak that school across the
train/test boundary the grouped split exists to enforce. So the helper
declines and says why, rather than producing a number nobody can defend.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis_helpers import apply_smote


def _imbalanced(n: int = 800, rate: float = 0.11, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=list("abcde"))
    y = (rng.random(n) < rate).astype(int)
    return X, y, rng


def test_declines_when_group_ids_are_supplied() -> None:
    X, y, rng = _imbalanced()
    ids = rng.integers(0, 90, len(X))
    X_res, y_res, meta = apply_smote(X, y, group_ids=ids)

    assert meta["applied"] is False
    assert "reason" in meta
    assert len(X_res) == len(X), "training set must be returned unchanged"
    assert len(y_res) == len(y)


def test_the_returned_lengths_stay_consistent_with_the_group_vector() -> None:
    """The exact crash: X and y grew, group_ids did not."""
    X, y, rng = _imbalanced()
    ids = rng.integers(0, 90, len(X))
    X_res, y_res, _ = apply_smote(X, y, group_ids=ids)
    assert len(X_res) == len(y_res) == len(ids), (
        "a grouped splitter would raise on inconsistent sample counts"
    )


def test_the_reason_names_the_alternatives() -> None:
    """A refusal without a way forward just moves the problem."""
    X, y, rng = _imbalanced()
    reason = apply_smote(X, y, group_ids=rng.integers(0, 50, len(X)))[2]["reason"]
    lowered = reason.lower()
    assert "class weight" in lowered
    assert "threshold" in lowered
    assert "pr-auc" in lowered
    assert "methods" in lowered, "the choice must be reported to the reader"


def test_still_applies_when_no_grouping_is_active() -> None:
    """The guard must not disable SMOTE for ungrouped designs."""
    X, y, _ = _imbalanced()
    X_res, _, meta = apply_smote(X, y)
    assert meta["applied"] is True
    assert len(X_res) > len(X)


def test_explicit_none_behaves_like_omitted() -> None:
    X, y, _ = _imbalanced()
    assert apply_smote(X, y, group_ids=None)[2]["applied"] is True


def test_balanced_data_is_untouched_either_way() -> None:
    """A balanced outcome should not trip SMOTE regardless of grouping."""
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(400, 3)), columns=list("abc"))
    y = (rng.random(400) < 0.5).astype(int)
    assert apply_smote(X, y)[2]["applied"] is False


@pytest.mark.parametrize("n_groups", [2, 50, 18717])
def test_scales_across_cluster_counts(n_groups: int) -> None:
    X, y, rng = _imbalanced(n=600)
    ids = rng.integers(0, n_groups, len(X))
    meta = apply_smote(X, y, group_ids=ids)[2]
    assert meta["applied"] is False
    assert str(len(np.unique(ids))) in meta["reason"]
