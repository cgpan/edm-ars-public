"""``edmars explain``: every term results can show has a plain definition."""

from __future__ import annotations

import pytest

from edmars import explain

REQUIRED = [
    "AUC",
    "RMSE",
    "R-squared",
    "confidence interval",
    "SHAP",
    "calibration",
    "subgroup fairness",
    "ATE",
    "propensity score",
    "matching",
    "IPW",
    "doubly robust",
    "causal forest",
    "treatment rule",
    "gap-in-gaps",
    "difference-in-differences",
    "reliability",
    "omega",
    "CFA",
    "CFI",
    "RMSEA",
    "IRT",
    "GRM",
    "IRT/GRM",
    "DIF",
    "measurement invariance",
    "CDM",
    "LSAR score",
    "UNVERIFIED",
    "INCOMPLETE",
    "invariant finding",
]


@pytest.mark.parametrize("term", REQUIRED)
def test_required_terms_are_explained(term: str) -> None:
    key = explain.lookup(term)
    assert key is not None, term
    text = explain.explain(term)
    assert explain.TERMS[key] in text


def test_every_alias_points_at_a_term() -> None:
    assert all(target in explain.TERMS for target in explain.ALIASES.values())


def test_definitions_are_substantial_but_short() -> None:
    for key, text in explain.TERMS.items():
        assert 150 <= len(text) <= 900, key


@pytest.mark.parametrize(
    "spelling", ["  auc ", "Auc?", "Propensity_Score", "propensity  score", "r2", "CI", "did", "Omega"]
)
def test_lookup_tolerates_spelling(spelling: str) -> None:
    assert explain.lookup(spelling) is not None


def test_unknown_terms_get_suggestions() -> None:
    text = explain.explain("shapp")
    assert text.startswith('No explanation for "shapp"')
    assert "Did you mean: SHAP?" in text
    assert "Terms I can explain:" in text
    assert explain.lookup("xyzzy") is None
    assert "Did you mean" not in explain.explain("xyzzy")
