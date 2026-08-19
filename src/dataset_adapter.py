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

class ELS2002Adapter(DatasetAdapter):
    """ELS:2002 public-use BY-F3 PETS student file (Arc G / Phase A).

    Key contrast with HSLS: the CSV stores NUMERIC CODES (negative
    sentinels), not labeled values — pd.to_numeric is safe, but
    sentinel codes must map to missing BEFORE imputation, and the
    continuous composites (BYSES1, BYMATHSE) have valid negative
    values where only exact codes <= -3 are sentinels.
    """

    def get_name(self) -> str:
        return "els_2002"

    def get_temporal_order(self) -> list[str]:
        return [
            "base_year",
            "first_follow_up",
            "second_follow_up",
            "third_follow_up",
        ]

    def get_missing_codes(self) -> list:
        return [-1, -2, -3, -4, -8, -9]

    def get_sample_size(self) -> int:
        return 16197

    def get_raw_data_filename(self) -> str:
        return "els_2002/els_02_12_byf3pststu_v1_0.csv"

    def get_multilevel_warning(self) -> str | None:
        return (
            "Multilevel structure (students nested in schools) is not "
            "modeled; ELS BRR replicate weights are excluded by tier-3 "
            "rules. This is a limitation."
        )

    def get_protected_attributes(self) -> list[str]:
        return ["BYSEX", "BYRACE", "BYSES1", "BYSES1QU"]


class DIDPanelAdapter(DatasetAdapter):
    """Pre-harmonized ELS x HSLS cross-cohort panel (Phase B)."""

    def get_name(self) -> str:
        return "did_els_hsls_panel"

    def get_temporal_order(self) -> list[str]:
        return ["base_wave", "follow_wave"]

    def get_missing_codes(self) -> list:
        return []

    def get_sample_size(self) -> int:
        return 16862

    def get_raw_data_filename(self) -> str:
        return "did_els_hsls_panel/panel.csv"

    def get_multilevel_warning(self) -> str | None:
        return (
            "School clustering is not carried into the harmonized panel; "
            "cross-cohort SEs are stratified-bootstrap only. This is a "
            "limitation."
        )

    def get_protected_attributes(self) -> list[str]:
        return ["low_ses", "female"]


class ASSISTments0910Adapter(DatasetAdapter):
    """ASSISTments 2009-10 skill-builder log data (V4 wave-2).

    Interaction LOG, not a survey: one row per problem attempt.
    original==1 filters main problems; correctness analyses use the
    FIRST attempt per (user, template). Responses are never imputed
    (sparsity is structural).
    """

    def get_name(self) -> str:
        return "assistments_0910"

    def get_temporal_order(self) -> list[str]:
        return ["single_year"]

    def get_missing_codes(self) -> list:
        return []

    def get_sample_size(self) -> int:
        return 4217  # students

    def get_raw_data_filename(self) -> str:
        return "assistments_0910/skill_builder_0910.csv"

    def get_multilevel_warning(self) -> str | None:
        return (
            "Attempts nest in students (user_id) and classes; analyses "
            "cluster at the student level. School/class IDs are tier-3."
        )

    def get_protected_attributes(self) -> list[str]:
        return []  # no demographics in the public log


_DATASET_REGISTRY: dict[str, type[DatasetAdapter]] = {
    "hsls09_public": HSLS09Adapter,
    "els_2002": ELS2002Adapter,
    "did_els_hsls_panel": DIDPanelAdapter,
    "assistments_0910": ASSISTments0910Adapter,
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
