"""DatasetAdapter abstraction: encapsulates dataset-specific knowledge.

Each dataset (HSLS:09, ELS:2002, etc.) implements this ABC so agents
can access dataset-specific constants without hardcoding them.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class DatasetAdapter(ABC):
    """Abstract base for dataset-specific configuration."""

    @abstractmethod
    def get_name(self) -> str:
        """Return the dataset identifier (e.g. 'hsls09_public')."""
        ...

    @abstractmethod
    def get_temporal_order(self) -> list[str]:
        """Return ordered list of data collection waves."""
        ...

    @abstractmethod
    def get_missing_codes(self) -> list:
        """Return sentinel values that should be treated as missing/NA."""
        ...

    @abstractmethod
    def get_sample_size(self) -> int:
        """Return the full sample size for feasibility checks."""
        ...

    @abstractmethod
    def get_raw_data_filename(self) -> str:
        """Return the raw data CSV filename (not a full path)."""
        ...

    @abstractmethod
    def get_multilevel_warning(self) -> str | None:
        """Return a multilevel structure warning, or None if not applicable."""
        ...

    @abstractmethod
    def get_protected_attributes(self) -> list[str]:
        """Return variable names used for fairness/subgroup analysis."""
        ...


# Canonical temporal ordering for HSLS:09 — exported for backward-compatible imports
HSLS09_TEMPORAL_ORDER = [
    "base_year",
    "first_follow_up",
    "second_follow_up",
    "update_panel",
]


class HSLS09Adapter(DatasetAdapter):
    """Dataset adapter for the HSLS:09 public-use file."""

    def get_name(self) -> str:
        return "hsls09_public"

    def get_temporal_order(self) -> list[str]:
        return list(HSLS09_TEMPORAL_ORDER)

    def get_missing_codes(self) -> list:
        return [
            -9, -8, -7, -6, -5,
            "Missing",
            "Unit non-response",
            "Data suppressed",
            "Component not applicable",
            "Item legitimate skip/NA",
        ]

    def get_sample_size(self) -> int:
        return 23503

    def get_raw_data_filename(self) -> str:
        return "hsls_17_student_pets_sr_v1_0.csv"

    def get_multilevel_warning(self) -> str | None:
        return (
            "Multilevel structure (students nested in schools) is not modeled. "
            "This is a limitation."
        )

    def get_protected_attributes(self) -> list[str]:
        return ["X1SEX", "X1RACE", "X1SES", "X1SES_U", "X1SESQ5"]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_DATASET_REGISTRY: dict[str, type[DatasetAdapter]] = {
    "hsls09_public": HSLS09Adapter,
}


def create_dataset_adapter(dataset_name: str) -> DatasetAdapter:
    """Create a DatasetAdapter instance for the given dataset name."""
    cls = _DATASET_REGISTRY.get(dataset_name)
    if cls is None:
        raise ValueError(
            f"Unknown dataset: {dataset_name!r}. "
            f"Available: {sorted(_DATASET_REGISTRY.keys())}"
        )
    return cls()
