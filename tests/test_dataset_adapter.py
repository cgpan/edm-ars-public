"""Tests for DatasetAdapter abstraction and HSLS09Adapter."""
import pytest

from src.dataset_adapter import (
    HSLS09Adapter,
    HSLS09_TEMPORAL_ORDER,
    DatasetAdapter,
    create_dataset_adapter,
)


class TestHSLS09Adapter:
    def setup_method(self) -> None:
        self.adapter = HSLS09Adapter()

    def test_get_name(self) -> None:
        assert self.adapter.get_name() == "hsls09_public"

    def test_get_temporal_order(self) -> None:
        order = self.adapter.get_temporal_order()
        assert order == ["base_year", "first_follow_up", "second_follow_up", "update_panel"]
        # Ensure it returns a copy, not the original list
        order.append("extra")
        assert "extra" not in self.adapter.get_temporal_order()

    def test_get_missing_codes(self) -> None:
        codes = self.adapter.get_missing_codes()
        assert -9 in codes
        assert -8 in codes
        assert "Missing" in codes
        assert "Unit non-response" in codes

    def test_get_sample_size(self) -> None:
        assert self.adapter.get_sample_size() == 23503

    def test_get_raw_data_filename(self) -> None:
        assert self.adapter.get_raw_data_filename() == "hsls_17_student_pets_sr_v1_0.csv"

    def test_get_multilevel_warning(self) -> None:
        warning = self.adapter.get_multilevel_warning()
        assert warning is not None
        assert "Multilevel structure" in warning
        assert "limitation" in warning

    def test_get_protected_attributes(self) -> None:
        attrs = self.adapter.get_protected_attributes()
        assert "X1SEX" in attrs
        assert "X1RACE" in attrs

    def test_is_dataset_adapter_subclass(self) -> None:
        assert isinstance(self.adapter, DatasetAdapter)


class TestHSLS09TemporalOrderExport:
    def test_constant_value(self) -> None:
        assert HSLS09_TEMPORAL_ORDER == [
            "base_year",
            "first_follow_up",
            "second_follow_up",
            "update_panel",
        ]


class TestCreateDatasetAdapter:
    def test_hsls09(self) -> None:
        adapter = create_dataset_adapter("hsls09_public")
        assert isinstance(adapter, HSLS09Adapter)
        assert adapter.get_name() == "hsls09_public"

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown dataset"):
            create_dataset_adapter("nonexistent_dataset")
