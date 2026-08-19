"""Phase B / Stream-1 v2 — deterministic ELS:2002 x HSLS:09 cross-cohort harmonizer.

Builds data/raw/did_els_hsls_panel/panel.csv for the `causal_did` task
type. Design decisions (recorded here because they ARE the estimand):

- Tests are NOT equatable across cohorts (different instruments), so
  score levels/growth cannot be compared. The harmonized outcome is
  the WITHIN-COHORT-WAVE PERCENTILE RANK of the math score (0-100),
  which is invariant to the instrument. The defensible DiD estimand is
  therefore a change in a GROUP GAP in ranks between cohorts (e.g. the
  low-vs-high-SES rank gap), NOT any absolute achievement change.
- group = bottom vs top SES quartile/quintile band (ELS BYSES1QU 1 vs
  4; HSLS X1SESQ5 quintiles 1 vs 5 - both map to "lowest band" vs
  "highest band"; the coarser common denominator is recorded).
- post = cohort (ELS 2002 = 0, HSLS 2009 = 1).
- Both waves are kept (base grade 10/9 and follow grade 12/11) so the
  follow wave provides a placebo/stability surface; grade offsets
  (ELS g10 vs HSLS g9 base) are a recorded limitation.

v2 (stream-1, 2026-07-04) adds harmonized covariates for the M9
composition-adjusted gap change and M10 ML heterogeneity:

- race5: White / Black / Hispanic / AsianPI / Other (ELS BYRACE codes
  7/3/{4,5}/2/{1,6}; HSLS X1RACE labels).
- pared3: HS_or_less / some_college / BA_plus (ELS BYPARED 1-2 / 3-5 /
  6-8; HSLS X1PAREDU labels - note HSLS has no "some college, no
  degree" category, so Associate's alone maps to the middle band; the
  coarseness is a recorded harmonization limitation).
- expect_ba: student expects to COMPLETE a bachelor's or higher (ELS
  BYSTEXP >= 5; HSLS labels "Complete a Bachelor's degree" and above;
  "don't know" -> NaN).
- ses_std: within-cohort z-score of the SES composite (BYSES1 / X1SES).

All sentinel handling per each dataset's conventions (ELS: exact
codes <= -3 for composites, negatives for integer codes; HSLS:
labeled CSV -> to_numeric coercion for scores, label matching for
categoricals).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw" / "did_els_hsls_panel"

PANEL_COLUMNS = [
    "cohort", "low_ses", "female", "race5", "pared3", "expect_ba",
    "ses_std", "rank_base", "rank_follow",
]


def _rank_pct(series: pd.Series) -> pd.Series:
    return series.rank(pct=True) * 100.0


def build_els(root: Path) -> pd.DataFrame:
    df = pd.read_csv(
        root / "data/raw/els_2002/els_02_12_byf3pststu_v1_0.csv",
        usecols=["BYSES1QU", "BYSES1", "BYSEX", "BYRACE", "BYPARED",
                 "BYSTEXP", "BYTXMSTD", "F1TXMSTD"],
        low_memory=False,
    )
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # integer-coded vars + test scores: all negatives are sentinels
    for c in ("BYSES1QU", "BYSEX", "BYRACE", "BYPARED", "BYSTEXP",
              "BYTXMSTD", "F1TXMSTD"):
        df[c] = df[c].where(df[c] >= 0)
    # continuous composite BYSES1: only exact codes <= -3 are sentinels
    df["BYSES1"] = df["BYSES1"].where(df["BYSES1"] > -3)

    df = df.dropna(subset=["BYSES1QU", "BYTXMSTD"])
    df = df[(df["BYSES1QU"] == 1) | (df["BYSES1QU"] == 4)].copy()

    out = pd.DataFrame(index=df.index)
    out["cohort"] = 0
    out["low_ses"] = (df["BYSES1QU"] == 1).astype(int)
    out["female"] = (df["BYSEX"] == 2).astype(int)
    out["race5"] = df["BYRACE"].map(
        {7: "White", 3: "Black", 4: "Hispanic", 5: "Hispanic",
         2: "AsianPI", 1: "Other", 6: "Other"}
    )
    out["pared3"] = pd.cut(
        df["BYPARED"], bins=[0, 2, 5, 8],
        labels=["HS_or_less", "some_college", "BA_plus"],
    ).astype(object)
    out["expect_ba"] = np.where(
        df["BYSTEXP"].isna(), np.nan, (df["BYSTEXP"] >= 5).astype(float)
    )
    out["ses_std"] = (df["BYSES1"] - df["BYSES1"].mean()) / df["BYSES1"].std()
    out["rank_base"] = _rank_pct(df["BYTXMSTD"])
    out["rank_follow"] = _rank_pct(df["F1TXMSTD"])
    return out[PANEL_COLUMNS]


def build_hsls(root: Path) -> pd.DataFrame:
    df = pd.read_csv(
        root / "data/raw/hsls_17_student_pets_sr_v1_0.csv",
        usecols=["X1SESQ5", "X1SES", "X1SEX", "X1RACE", "X1PAREDU",
                 "X1STUEDEXPCT", "X1TXMTSCOR", "X2TXMTSCOR"],
        low_memory=False,
    )
    q = df["X1SESQ5"].astype(str)
    keep = q.str.contains("First quintile", na=False) | q.str.contains(
        "Fifth quintile", na=False
    )
    df = df[keep].copy()
    q = q.loc[df.index]

    for c in ("X1TXMTSCOR", "X2TXMTSCOR"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[df[c] < 0, c] = np.nan
    df["X1SES"] = pd.to_numeric(df["X1SES"], errors="coerce")
    df.loc[df["X1SES"] <= -3, "X1SES"] = np.nan  # valid negatives exist
    df = df.dropna(subset=["X1TXMTSCOR"])
    q = q.loc[df.index]

    race_map = {
        "White, non-Hispanic": "White",
        "Black/African-American, non-Hispanic": "Black",
        "Hispanic, race specified": "Hispanic",
        "Hispanic, no race specified": "Hispanic",
        "Asian, non-Hispanic": "AsianPI",
        "Native Hawaiian/Pacific Islander, non-Hispanic": "AsianPI",
        "Amer. Indian/Alaska Native, non-Hispanic": "Other",
        "More than one race, non-Hispanic": "Other",
    }
    pared_map = {
        "Less than high school": "HS_or_less",
        "High school diploma or GED": "HS_or_less",
        "Associate's degree": "some_college",
        "Bachelor's degree": "BA_plus",
        "Master's degree": "BA_plus",
        "Ph.D/M.D/Law/other high lvl prof degree": "BA_plus",
    }
    ba_plus_labels = {
        "Complete a Bachelor's degree", "Start a Master's degree",
        "Complete a Master's degree", "Start Ph.D/M.D/Law/other prof degree",
        "Complete Ph.D/M.D/Law/other prof degree",
    }
    below_ba_labels = {
        "Less than high school", "High school diploma or GED",
        "Start an Associate's degree", "Complete an Associate's degree",
        "Start a Bachelor's degree",
    }

    out = pd.DataFrame(index=df.index)
    out["cohort"] = 1
    out["low_ses"] = q.str.contains("First quintile").astype(int)
    out["female"] = (df["X1SEX"].astype(str) == "Female").astype(int)
    out["race5"] = df["X1RACE"].astype(str).map(race_map)
    out["pared3"] = df["X1PAREDU"].astype(str).map(pared_map)
    exp = df["X1STUEDEXPCT"].astype(str)
    out["expect_ba"] = np.where(
        exp.isin(ba_plus_labels), 1.0,
        np.where(exp.isin(below_ba_labels), 0.0, np.nan),
    )
    out["ses_std"] = (df["X1SES"] - df["X1SES"].mean()) / df["X1SES"].std()
    out["rank_base"] = _rank_pct(df["X1TXMTSCOR"])
    out["rank_follow"] = _rank_pct(df["X2TXMTSCOR"])
    return out[PANEL_COLUMNS]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    els = build_els(ROOT)
    hsls = build_hsls(ROOT)
    panel = pd.concat([els, hsls], ignore_index=True)
    panel.to_csv(OUT / "panel.csv", index=False)
    gap = panel.groupby(["cohort", "low_ses"])["rank_base"].mean().unstack()
    print(f"panel: {len(panel)} students (ELS {len(els)}, HSLS {len(hsls)})")
    print("mean base rank by cohort x low_ses:")
    print(gap.round(2))
    did = (gap.loc[1, 1] - gap.loc[1, 0]) - (gap.loc[0, 1] - gap.loc[0, 0])
    print(f"raw 2x2 DiD (SES rank-gap change, base wave): {did:.2f} rank points")
    print("\ncovariate coverage (pct non-missing):")
    for c in ("race5", "pared3", "expect_ba", "ses_std"):
        print(f"  {c}: {panel[c].notna().mean():.3f}")
    print("\nrace5 x cohort:")
    print(pd.crosstab(panel["race5"], panel["cohort"]))
    print("\npared3 x cohort:")
    print(pd.crosstab(panel["pared3"], panel["cohort"]))
    print("expect_ba means by cohort:",
          panel.groupby("cohort")["expect_ba"].mean().round(3).to_dict())


if __name__ == "__main__":
    main()
