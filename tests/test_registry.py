import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.registry import RegistryLoader

REGISTRY_PATH = str(
    Path(__file__).parent.parent / "data_registry" / "datasets" / "hsls09_public.yaml"
)


@pytest.fixture(scope="module")
def registry() -> RegistryLoader:
    return RegistryLoader(REGISTRY_PATH)


def test_load_registry(registry: RegistryLoader) -> None:
    assert registry._data["name"] == "hsls09_public"


def test_get_outcomes(registry: RegistryLoader) -> None:
    outcome_names = [o["name"] for o in registry.get_outcomes()]
    assert "X3TGPAMAT" in outcome_names


def test_get_predictors(registry: RegistryLoader) -> None:
    academic = registry.get_predictors(category="academic")
    names = [v["name"] for v in academic]
    assert "X1TXMTSCOR" in names


def test_get_all_predictors(registry: RegistryLoader) -> None:
    all_preds = registry.get_predictors()
    assert len(all_preds) > 0


def test_temporal_order(registry: RegistryLoader) -> None:
    assert registry.validate_temporal_order("base_year", "second_follow_up") is True
    assert registry.validate_temporal_order("second_follow_up", "base_year") is False
    # Same wave: strict less-than → False (catches same-wave leakage)
    assert registry.validate_temporal_order("base_year", "base_year") is False


def test_is_protected(registry: RegistryLoader) -> None:
    assert registry.is_protected_attribute("X1SEX") is True
    assert registry.is_protected_attribute("X1TXMTSC") is False


def test_tier3_exclusion(registry: RegistryLoader) -> None:
    # Prefix pattern W[0-9] matches W1STUDENT
    assert registry.is_excluded("W1STUDENT") is True
    # Suffix pattern _IM$ matches X1TXMTSC_IM
    assert registry.is_excluded("X1TXMTSC_IM") is True
    # Exact match
    assert registry.is_excluded("STU_ID") is True
    # Normal predictor should not be excluded
    assert registry.is_excluded("X1TXMTSC") is False
