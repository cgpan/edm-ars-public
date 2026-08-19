# V3.8 Arc E — Controlled Evaluation Protocol (SPEC §10 Phase 5, extended)

> Status: harness + pilot implemented 2026-07-03. Full study gated on the two
> human-dependent items in §4 (user gates per roadmap §9).

## 1. Design

**Paper matrix** (pipeline arm): N ≈ 10 EDM-ARS papers across the executable
task types on HSLS:09 —

| Cell | Task type | Question source | n papers |
|---|---|---|---|
| P1–P4 | prediction | canonical questions + PF-generated (gap-matrix-driven) | 4 |
| C1–C3 | causal_soo | locked specs (varying treatment/outcome) | 3 |
| I1–I3 | causal_itr | locked specs (varying rule covariates) | 3 |

**Baseline arm**: ≥ 3 human-authored papers on matched questions (user-supplied
or recruited — gate G-E2).

**Review protocol**: every paper (blinded: authorship stripped, formatting
normalized to the ACM template) reviewed by (a) LSAR with the Arc L calibrated
anchors, and (b) ≥ 2 human EDM/LAK-familiar reviewers on the same 8 dimensions
(1–10) plus an overall accept/reject recommendation.

**Analyses** (implemented in `scripts/evaluation_harness.py`):
1. Pipeline-vs-human paper quality: dimension-wise medians + rank-sum tests.
2. LSAR-vs-human agreement: per-dimension Pearson/Spearman r, overall MAD —
   this retro-validates the Arc L calibration.
3. Error census: per-paper defect list (from human reviewer comments) coded
   against the pipeline's F-item taxonomy.
4. Cost/time per paper (tokens + wall clock from run manifests).

## 2. Pilot (implemented now)

Three papers, one per task type, on the current fully-slim + hardened + styled
stack: the 3b.25.5 causal_soo paper (LSAR 7.0), the Arc R R3 causal_itr paper,
and one NEW prediction-task paper (the first of the slim era — also the first
live exercise of full PF generation with the Arc D gap matrix + design memo and
the Arc S archetype skills). The harness's `collect` command builds the pilot
ledger from run directories.

## 3. Harness usage

```
python scripts/evaluation_harness.py collect --runs <run_dir> [<run_dir> ...] --out evaluation/ledger.json
python scripts/evaluation_harness.py agreement --ledger evaluation/ledger.json --human evaluation/human_ratings.csv
```

`human_ratings.csv` columns: `paper_id, reviewer_id, dimension, score` (+
`overall` rows). The agreement command is inert until that file exists.

## 4. User gates (roadmap §9)

- **G-E1 — full pipeline batch** (~7 more papers ≈ $45, ~10 h): approve before
  the harness's `matrix` command generates the remaining cells.
- **G-E2 — human arm**: recruit ≥ 2 reviewers + supply ≥ 3 matched
  human-authored papers. The blinding/normalization tooling is ready.
