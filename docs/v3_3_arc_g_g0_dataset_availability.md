# V3.3 Arc G (G0) — Public-Use Dataset Availability Investigation

> Investigated 2026-07-03 per docs/v4_roadmap.md Arc G and the user's
> free-use-data-only constraint. Verdict: **all four targets are public-use
> and free**; three have direct-download paths, one (ELS:2002) routes through
> an interactive NCES wizard (small user step or browser-automation follow-up).

| Dataset | Public-use? | Direct download | Size | Registration | Go/no-go |
|---|---|---|---|---|---|
| **ASSISTments 2009-10 skill-builder** | ✓ | Google Drive file (+ IEEE DataPort mirror) | ~35 MB CSV | none | **GO — first** (smallest; prediction-side per roadmap) |
| **ECLS-K:2011 K-5 PUF** | ✓ | `https://nces.ed.gov/ecls/data/2019/ChildK5p.zip` + SPSS/SAS/Stata syntax at same path | 460 MB ASCII fixed-width | none stated | **GO** (fixed-width + syntax-file parsing → G1 kit requirement) |
| **PISA 2022** | ✓ | OECD file index `https://webfs.oecd.org/pisa2022/index.html` (SAS/SPSS .sav) | multi-GB (student file largest) | none | **GO (staged)** — student questionnaire file only at first; plausible-values sub-arc per roadmap G4 |
| **ELS:2002 PUF** | ✓ | via NCES **EDAT** interactive wizard (`https://nces.ed.gov/edat/`); no direct full-file zip found on the pubid pages | n/a | none stated, but wizard-driven | **GO with gate** — needs either a short manual EDAT export by the user or a browser-automation follow-up; matters most as the Q3 DiD partner cohort |

## Acquisition status (updated 2026-07-03)

| Dataset | Status |
|---|---|
| ASSISTments 2009-10 | ✅ acquired (83 MB CSV, `data/raw/assistments_0910/`); draft registry committed |
| ECLS-K:2011 K-5 PUF | ✅ acquired (482 MB zip + 5 MB syntax, `data/raw/ecls_k_2011/`); layout parses to 26,060 columns |
| PISA 2022 | ✅ acquired (682 MB zip → `CY08MSP_STU_QQQ.SAV`, 2.1 GB uncompressed, `data/raw/pisa_2022/`); user-approved download; plausible-values sub-arc (G4) next |
| ELS:2002 | ✅ acquired (user EDAT export 2026-07-03: 175 MB BY-F3 PETS student CSV, 4,012 cols + codebook, `data/raw/els_2002/`); draft registry committed; **DiD partnership wired both ways — the selector now marks DiD data-feasible on HSLS** (executable task type = Q3 future work) |

## Notes for G1 (onboarding kit requirements derived here)

1. NCES family (ECLS-K, ELS) ships **ASCII fixed-width + SPSS syntax** — the
   kit must parse `.sps` layout (column positions, variable names, value
   labels) to produce a labeled CSV, then draft the Tier-2 registry.
2. PISA ships `.sav` — `pandas.read_spss`/pyreadstat path + **plausible
   values** methodology (10 PVs per domain; Rubin combining) is its own G4
   sub-arc as the roadmap anticipated.
3. ASSISTments is interaction-level (student × problem × skill × correctness)
   — different unit of analysis; feeds the prediction task type
   (knowledge-tracing features), not the causal designs.
4. Suppression caveats (per the HSLS lesson) can only be cataloged after
   acquisition — codebook sweeps for `-5 suppressed` analogues are a G2+ step
   per dataset.

## Sources

- [ELS:2002 available data](https://nces.ed.gov/surveys/els2002/avail_data.asp) · [ELS:2002 BY–F2 PUF page](https://nces.ed.gov/use-work/resource-library/data/data-file/education-longitudinal-study-2002-base-year-second-follow-public-use-data?pubid=2010338) · [EDAT](https://nces.ed.gov/edat/)
- [ECLS data products](https://nces.ed.gov/ecls/dataproducts.asp) (direct file URLs verified on-page)
- [PISA 2022 database](https://www.oecd.org/en/data/datasets/pisa-2022-database.html) · [PISA file index](https://webfs.oecd.org/pisa2022/index.html) · [NCES PISA data files](https://nces.ed.gov/surveys/pisa/datafiles.asp)
- [ASSISTments 2009-10 skill-builder page](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010) · [IEEE DataPort mirror](https://ieee-dataport.org/documents/assistments-dataset-2009-2010)
