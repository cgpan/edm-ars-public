<!-- V2 backtest verdict, 2026-07-11. NULL once deconfounded. -->

# Verification report — Arc T T1a + V2 backtest

## 1. Ownership

`git diff --stat HEAD` is **empty** — nothing was modified. All eight new files are untracked. No file is claimed by both agents; no new file is unclaimed.

| File | Claimed by | On disk | Overlap |
|---|---|---|---|
| `scripts/backtest_ranker.py` (1013 L) | Agent 1 (backtest) | yes | none |
| `tests/test_t1_backtest.py` | Agent 1 | yes | none |
| `src/ideation/slate.py` | Agent 2 (T1a) | yes | none |
| `src/ideation/generate.py` | Agent 2 | yes | none |
| `src/ideation/cards.py` | Agent 2 | yes | none |
| `scripts/run_idea_tournament.py` | Agent 2 | yes | none |
| `tests/test_t1a_generation.py` | Agent 2 | yes | none |

Pre-existing untracked noise claimed by neither and **not** produced by this work (present in the session-start snapshot): `cache/`, `tmp_orch_test/`, `tmp_orch_test2/`, `runs/phase_a_els_prediction_20260704/pipeline_stdout.txt`. Neither agent touched the T0-committed `src/ideation/{feasibility,venue_fit,probe_cache}.py` or `data_registry/venue_fit_rules.yaml` — confirmed by the empty diff.

---

## 2. The backtest — verdict: **the raw number is positive, the finding is null**

### 2a. Script output, verbatim (exit 0)

```
PRIMARY: ledger + median-of-3 + EDM
    n = 5   rho = +0.7906
    permutation (exact, 120 permutations): p(one-sided, rho>0) = 0.0667, p(two-sided) = 0.1333
    smallest one-sided p this n can produce = 0.0333
    bootstrap 95% CI = [-0.304, 1.000] (195/10000 resamples degenerate)
    distinct venue_fit: [0.5, 1.5, 3.0]   distinct feasibility penalties: [0.0]

VERDICT: POSITIVE_BUT_UNDERPOWERED
  direction: POSITIVE   pair separated: True   power: UNDERPOWERED   n (primary): 5
```

Included (5): `phase_a_els_prediction_20260704_attempt3` (6.2), `stream2_els_rigor_20260704_attempt2` (6.6), `stream1_did_v2_20260708` (7.0), `v4_psy_paper2_20260708` (7.3), `v4_psy_paper1_20260708` (7.5). Excluded: 29 of 34, every one with a named reason. Sensitivity: +JEDM n=7 rho +0.6241; +superseded n=9 rho +0.3567; all-gated n=24 rho +0.3217. Pre-registered pair **separated**, margin +3.50 (`phase_b_did` −2.00/gate 3.7 vs `stream1_did_v2` +1.50/gate 7.0).

### 2b. Outcome variable — verified from disk, not from the script

I read `scores.overall_score` out of every `cycle_*/lsar_report.json` and took my own median:

| run | cycles | scores | my median | `gate_summary.final_score` | venue |
|---|---|---|---|---|---|
| phase_a_els_..._attempt3 | 1,102,103 | [6.8, 6.2, 5.7] | 6.2 | 6.2 | EDM |
| stream2_els_rigor_attempt2 | 1,102,103 | [6.0, 6.6, 6.6] | 6.6 | 6.6 | EDM |
| stream1_did_v2 | 1,102,103 | [6.6, 7.2, 7.0] | 7.0 | 7.0 | EDM |
| v4_psy_paper2 | 1,102,103 | [7.3, 6.1, 7.4] | 7.3 | 7.3 | EDM |
| v4_psy_paper1 | 1,102,103 | [7.7, 7.5, 7.1] | 7.5 | 7.5 | EDM |
| phase_b_did (pair) | 1 | [3.7] | 3.7 | 3.7 | EDM |

**All five are genuinely median-of-3, all EDM, all agree with the summary file.** The pair's `phase_b_did` is genuinely a single review. Agent 1's outcome extraction is correct, including its `cycle*100+extra` vs revision-cycle rule (I confirmed `arc_p_validation_20260711` = cycles 1/2/3 → 6.5/3.1/4.2, final 4.2; naive `cycle_1` would report 6.5).

### 2c. Leakage — none in code, **decisive leakage in the rule table**

`venue_fit.py` and `feasibility.py` contain zero occurrences of `gate_summary`, `lsar`, `ledger`, `overall_score`, `final_score`, or `runs/`. The scoring path cannot see the outcome.

But `data_registry/venue_fit_rules.yaml` can. VF-01's evidence reads *"Our own phase_b_did_20260704 … scored … 3.7 Reject"*; VF-04's reads *"The 3.7 → 7.0 recovery … (phase_b_did → stream1_did_v2)"*. Those two rules were authored from the outcomes being tested. VF-02/03/05/06/07 cite only external anchors.

Partitioning the rule table on that line (`scripts/backtest_ranker.py` loader, my own Spearman):

| population | full venue_fit | **out-of-sample rules only** (VF-02/03/05/06/07) | in-sample rules only (VF-01, VF-04) |
|---|---|---|---|
| n = 24 (all gated) | +0.3766, p=0.036 | **+0.0018, p=0.501** | +0.3486, p=0.049 |
| n = 5 (primary) | +0.7906, p=0.066 | +0.7906, p=0.066 | degenerate (VF-04 constant) |

**On the only population with real spread, 100% of the correlation comes from the two rules written knowing the answer. The five externally-mined rules measure exactly zero.**

### 2d. Chance — reproduced independently

My own enumeration: rho = +0.7906, one-sided exact p = 8/120 = 0.0667, floor for this tie structure 4/120 = 0.0333. Matches the script. Concordance: **7 concordant, 1 discordant, 2 ties-in-x**.

**Agent 1's prose misattributes the discordance.** They wrote *"one discordance, from the duplicated ELS idea."* The duplicated ELS pair is a **tie in x** (both det = 0.15) and contributes no discordance. The single discordance is `v4_psy_paper2` (det +0.90, gate 7.3) vs `v4_psy_paper1` (det +0.45, gate 7.5) — i.e. **VF-05, the only rule in the primary population that adds anything beyond task type, points the wrong way**. Adding VF-05 *lowers* rho from +0.8660 to +0.7906. The shipped script output does not contain this error; only the prose report does.

### 2e. Trivial baselines — **a baseline with zero idea content beats the ranker**

Same y, same exact test, n = 5 primary:

| predictor | rho | exact one-sided p | conc/disc |
|---|---|---|---|
| **run recency** (checkpoint timestamp) | **+0.9000** | **0.0417 (5/120)** | 6/0 |
| task_type ladder (pred<causal<psych) | +0.9487 | 0.0333 | 8/0 |
| VF-02 alone ("is a prediction paper") | +0.8660 | 0.1000 | 6/0 |
| **the ranker** (0.30·vf − 0.20·pen) | **+0.7906** | **0.0667** | 7/1 |
| analytic_n | +0.3591 | 0.3167 | 6/3 |

The recency baseline is fully a priori — "the pipeline got better over five days" — and it is the *only* predictor on the primary population that reaches α = 0.05. (The task_type ladder's ordering was chosen by me after seeing the data; I disclose that. The recency baseline was not.)

The ranker and recency are **0.9487 collinear** on the primary population (and +0.7976 on n=24). Partial rho(deterministic, gate | recency) = **−0.4588** on n=5 and **+0.1691, p=0.214** on n=24. At n=5 the −0.46 does not mean the ranker is harmful — it means the two are so collinear the backtest **cannot attribute the correlation to the ranker at all**.

### 2f. Feasibility term — untested, confirmed

Primary penalties: `[0.0]`, zero variance. The composite is a positive affine transform of venue_fit and carries identical ranks. Agent 1 states this correctly.

### 2g. My assessment of whether the backtest supports its conclusion

**It does not support the conclusion drawn, and the honest reading is stronger than "underpowered."** Agent 1 reaches the right *action* (advisory only, do not clear V2) but frames the problem as sample size. The real problems are identification, and they do not go away when n grows:

1. On n = 24, the out-of-sample half of the rule table has **rho = +0.002**. That is not a power problem — it is a measurement of zero.
2. On n = 5, the entire signal is one bit ("prediction papers scored lower"), perfectly confounded with run date, and a pure recency baseline outranks the ranker.
3. The one rule that differentiates beyond task type (VF-05) supplies the sole discordance.
4. The pre-registered pair is entirely in-sample for both of its rules, so its separation is a consistency check on the implementation — as Agent 1 correctly says.

**Not inverted. Null once confounded with run recency and the two outcome-authored rules.** Agent 1's own recommendation — advisory mode, no live selection — is the correct one; the justification should be "the out-of-sample rules measure zero," not "n is small."

---

## 3. Full suite

```
1776 passed, 13 skipped, 1103 warnings in 852.70s (0:14:12)
```

Exit code 0, zero failures. (Higher than either agent's count because I ran `pytest tests/ -q` with no `-k` filter.) Nothing to attribute.

---

## 4. Seam check — **PASS, 13/13**

I built a card per distinct `(dataset, task_type)` cell in the seed-42 slate, ran `compile_spec`, and called the real `src.main.load_locked_research_spec`:

```
SEAM RESULT: 13 loaded / 0 failed  (of 13)
```

Both keys survive on every pair, e.g. `spec.task_type='causal_did' spec.dataset='did_els_hsls_panel' → loaded keeps tt='causal_did' ds='did_els_hsls_panel'`. `rank1_spec.json` from a real offline tournament run also loads (`prediction` / `els_2002`).

One API sharp edge: `IdeaCard.cell` must be a **dict**. Passing a `SlateCell` to the constructor raises `AttributeError: 'SlateCell' object has no attribute 'get'` inside `cards.py:198`. `generate.py` always calls `cell.to_dict()`, so production is fine, but the dataclass accepts the wrong type silently until compile time.

**Agent 2's matrix-vs-loader defect reproduces exactly.** `DATASET_TASK_MATRIX["assistments_0910"]["prediction"] is True`, yet the real loader raises `ValueError: … TEMPORAL VIOLATION: predictor 'attempt_count' (registry wave=single_year, idx=0) does not precede outcome 'correct' (wave=single_year, idx=0)`. Their slate rule S3 correctly excludes the cell and records the loader error as evidence.

---

## 5. C1 check — **clean**

Grep over `src/ideation/` and `scripts/run_idea_tournament.py` finds novelty only in negative positions: a ban list (`BANNED_CARD_KEYS`, enforced at `cards.py:328`, `:334`, `:617`), a prompt prohibition (`generate.py:200`), an ignore-list so a template warning isn't double-counted (`feasibility.py:1231`), and `feasibility.rank_key`'s docstring stating novelty is not a term. No positive novelty score is computed, stored, or ranked anywhere.

Behavioural confirmation: I injected `novelty_score: 0.91` **and** `novelty_score_self_assessment: 5` into a card payload. Both were dropped; the compiled spec's novelty-key list is `[]`; `rank1_spec.json` likewise.

---

## 6. Slate distribution — **genuinely diverse, does not collapse**

Seed 42, n=24 (stubbed/offline, no LLM):

- **datasets**: `els_2002` 8, `hsls09_public` 7, `did_els_hsls_panel` 5, `assistments_0910` 4
- **task types**: `causal_soo` 8, `psychometrics` 6, `prediction` 5, `causal_itr` 3, `causal_did` 2
- **13 distinct (dataset, task) cells**, max **2** per cell (cap 3)
- **patterns**: 9 distinct, 2–3 each; bridge share 3/24 = 12.5%
- **personas**: all 6 at exactly 4
- **gap cells**: 24 candidates, 24 *unique* gap cells

Seed 7 also yields 13 distinct cells with a different allocation (`els_2002` 8, `hsls09_public` 8, `did_els_hsls_panel` 5, `assistments_0910` 3). `slate.json` is byte-identical across two seed-42 builds. The offline end-to-end run reports `collapsed_to_one_dataset: false`. 20 cells total, 14 matrix-feasible, 13 after rule S3 — all three of Agent 2's counts confirmed. Every excluded cell carries a dispositive reason.

Minor: re-running the same `tournament_id`+seed produces byte-identical `slate.json`, `rank1_spec.json`, `feasibility.json`, but `candidates.jsonl` differs in a wall-clock `generated_at` and `ranking_deterministic.json` in an absolute output path. Substance is identical.

Also reconciled: the "renders at exactly 120 words" claim is true under `render_word_count()` (per-field caps sum to exactly 120); the full rendered string is 126 tokens including the header line and section labels. Not a defect, a definition. The Fisher-z planning numbers (n≈9, n≈25) are correct — the 1.06 Spearman variance inflation explains why 25 rather than my naive 24.

---

## 7. Prioritized blocking list

**Blocking — the headline**

1. **Do not let V2 clear the T1b gate in spec §9.** The gate text ("V2 must return rho > 0 before the tournament may select a spec for a live run") is technically satisfied, but the out-of-sample rules measure rho = +0.002 on n=24, and on n=5 the ranker is beaten by a pure run-recency baseline it is 0.95-collinear with. Ship the judged layer in advisory mode as both Agent 1 and the spec prescribe.
2. **Add the two decisive diagnostics to `backtest_ranker.py`** so this cannot be lost: (a) the in-sample/out-of-sample rule partition, (b) at least one confound baseline (run recency) and one trivial baseline printed beside the ranker. Right now the script reports only the ranker's own number, which is what made "POSITIVE_BUT_UNDERPOWERED" look like the whole story.
3. **Fix Agent 1's discordance attribution** in whatever prose gets committed: the discordance is VF-05 on `v4_psy_paper2` vs `v4_psy_paper1`, not the duplicated ELS idea (which is a tie).

**Blocking — cheap and mechanical**

4. `.gitignore` lacks `ideas/`; `scripts/run_idea_tournament.py` defaults `--ideas-dir ideas`. The first default-path run pollutes `git status`.
5. `agent_prompts/idea_generator.yaml` does not exist, so `generate.py` carries `FALLBACK_SYSTEM_PROMPT` — a direct violation of the CLAUDE.md rule "Agent system prompts live in agent_prompts/*.yaml, NEVER hardcoded in Python." Agent 2 disclosed it and left a drop-in-ready hook; it is a five-line file.
6. `config.yaml` has no `ideation:` block; seven tuned values (24 / 3 / 0.80 / 42 / 0.30 / 0.20 / 0.9) live as code defaults, violating "Config values come from config.yaml — never hardcode."

**Non-blocking, should be recorded**

7. `docs/v5_arc_t_spec.md` §6/§9 says "the 12 ledger papers with median-of-3 gate scores." Measured: 12 papers **total**, 7 median-of-3, **5** EDM-calibrated. Also missing: the range-restriction note (median sampling only triggers within ±1.5 of 6.3, so the population is selected on the criterion) and the in-sample status of VF-01/VF-04.
8. `evaluation/ledger.json` carries three `run_dir` values with a literal vertical tab from an unescaped `\v4` — confirmed: `'runs\x0b4_psy_paper1_20260708'`, `…paper2…`, `…cdm_journal_a3`. Anything joining on `run_dir` silently misses those three.
9. `DATASET_TASK_MATRIX` vs `PredictionTemplate` disagreement on single-wave datasets (item 4 above). Rule S3 is a correct workaround with a self-invalidating test, but the underlying disagreement belongs in `feasibility.py` or `task_template.py`.
10. `IdeaCard.cell` accepts a `SlateCell` and fails later inside `compile_spec`. Either type-guard the constructor or coerce.

**Not done, correctly disclosed by both agents:** no live LLM call was ever made (zero real cards exist, so nothing is known about real diversity, dedupe rate, or cost); no V4 shuffle control (nothing to shuffle without a judged layer); no out-of-sample venue-fit table. Item 3 above makes the third of these the highest-value next step, not the lowest.