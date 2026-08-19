"""V3.3 Arc G (G1) — onboarding-kit verification (synthetic fixtures;
the real-acquisition proofs are recorded in the G0/G2 docs)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.onboard_dataset import (
    convert_nces_fixed_width,
    draft_registry,
    parse_sps_layout,
    profile_csv,
)


@pytest.fixture()
def csv_path(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        {
            "STU_ID": range(100),
            "X1SCORE": [50.0 + i * 0.3 for i in range(100)],
            "X1SEX": [i % 2 for i in range(100)],
            "X1CAT": (["A", "B", "C", None] * 25),
            "W1WEIGHT": [1.0] * 100,
            "X1SENT": [-9 if i < 10 else i for i in range(100)],
        }
    )
    p = tmp_path / "toy.csv"
    df.to_csv(p, index=False)
    return p


class TestProfileAndDraft:
    def test_profile_types_and_missingness(self, csv_path: Path) -> None:
        prof = profile_csv(csv_path)
        assert prof["n_rows"] == 100
        assert prof["columns"]["X1CAT"]["pct_missing"] == 25.0
        assert "range" in prof["columns"]["X1SCORE"]
        assert "-9" in prof["columns"]["X1SENT"]["sentinel_candidates"]

    def test_draft_registry_shape(self, csv_path: Path) -> None:
        reg = draft_registry(profile_csv(csv_path), "toy")
        by_name = {v["name"]: v for v in reg["variables"]["auto_profiled"]}
        assert by_name["X1SCORE"]["type"] == "continuous"
        assert by_name["X1SEX"]["type"] == "binary"
        assert by_name["X1CAT"]["type"] == "categorical"
        # tier-3 candidates: ids + weights
        assert "STU_ID" in reg["tier3_exclusion_candidates"]
        assert "W1WEIGHT" in reg["tier3_exclusion_candidates"]
        # honest-empty design_feasibility skeleton + curation checklist
        assert reg["design_feasibility"]["running_variables"] == []
        assert any("design_feasibility" in c for c in reg["curation_checklist"])


class TestNCESFixedWidth:
    def _sps(self, tmp_path: Path) -> Path:
        p = tmp_path / "toy.sps"
        p.write_text(
            "FILE HANDLE X / NAME='toy.dat'.\n"
            "DATA LIST FILE=X /\n"
            "  CHILDID 1-4  SCORE 5-7  FLAG 8-8\n"
            ".\nVARIABLE LABELS CHILDID 'id'.\n",
            encoding="utf-8",
        )
        return p

    def test_parse_sps_layout(self, tmp_path: Path) -> None:
        layout = parse_sps_layout(self._sps(tmp_path))
        assert layout == [("CHILDID", 1, 4), ("SCORE", 5, 7), ("FLAG", 8, 8)]

    def test_convert_fixed_width_with_subset(self, tmp_path: Path) -> None:
        dat = tmp_path / "toy.dat"
        dat.write_text("0001123 1\n0002456 0\n".replace(" ", "9"), encoding="utf-8")
        # rows: CHILDID(1-4) SCORE(5-7) FLAG(8)
        dat.write_text("00011231\n00024560\n", encoding="utf-8")
        out = tmp_path / "out.csv"
        n = convert_nces_fixed_width(
            self._sps(tmp_path), dat, out, columns=["CHILDID", "SCORE"]
        )
        assert n == 2
        df = pd.read_csv(out)
        assert list(df.columns) == ["CHILDID", "SCORE"]
        assert df["SCORE"].astype(int).tolist() == [123, 456]
