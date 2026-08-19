<!-- T1b verification, 2026-07-11: blind table also null; 5 of 8 rules never fire on our specs. -->

Full suite complete. Assembling the verification.

---

# Arc T T1b — Skeptical Verification

## Headline answer

**The blind table is also null.** It is more honest than v1 — its counts reproduce independently and its evidence cites no EDM-ARS output — but it does not measure a usable ranking signal, it **inverts** on the calibrated population, and it **fails the spec's own pre-registered pair test**. Two agent-reported claims are false. Suite is green.

---

## 1. Blindness audit — PASSES, with one disclosed contamination

**Declaration**: present in the report and in the table itself (`venue_fit_rules_v2.yaml` lines 1–40).

**Leakage scan of the rule table.** All 8 `evidence:` strings cite **only anchor counts**. Zero references to our runs, gate scores, `phase_b_did`, or `stream1_did_v2`. The leakage vocabulary that does appear is confined to three non-evidence locations:

| Line | Text | Verdict |
|---|---|---|
| 7–20 | `# The V2 backtest of the v1 table came back NULL... rho = +0.002 ... in-sample-only rules measured +0.349` and the list of files not read (`evaluation/ledger.json, runs/**`) | Protocol disclosure. But these are **outcome statistics**, and they are in the file. Reconcilable only because the orchestrator brief supplied them. |
| 232 | `CONVERGENCE DISCLOSURE. v1 carries an in-sample rule pointed at roughly the same target... the CHOICE to measure this feature may not be independent.` | **The one material exposure**, self-disclosed |
| 573 | `The v1 table carries a hand-written carve-out ... derived in-sample and is deliberately NOT imported here.` | Refusal to import contamination |

**The contamination vector is confirmed and is not the agent's fault.** `docs/v5_arc_t_spec.md` line 272 reprints VF-01's in-sample evidence verbatim — *"our own `phase_b_did` scored Relevance 2 / Significance 2 → 3.7 Reject"* — and line 275 the *"3.7 → 7.0 recovery"*. The brief required reading that doc first. Any future "blind" derivation is contaminated on arrival.

**Independent reproduction (my own loader, not their audit script).** I rebuilt the corpus from `$LSAR_HOME/outputs` with my own sha1 dedupe, venue assignment, and regex matcher. n = 34, EDM 15 / JEDM 10 / JLA 9 — reproduces. **All 8 rules' counts including per-venue splits reproduce exactly.** Their own audit script also passes 55/55, exit 0.

**v2's blindness is genuinely stricter than v1's.** v1 tags VF-03/05/06 `external`, yet their evidence cites LSAR *scores* of anchors — *"scored 5.4 Borderline"*, *"jedm_974 external validation scored 7.3"*, *"jla_9035 (6.8)"*. v2 refused to open `scores.json` at all.

---

## 2. Backtest: v1 vs v2

**Method.** I did not modify `scripts/backtest_ranker.py`. I wrote a wrapper (`scratchpad/backtest_v2_table.py`) that imports it and monkey-patches only `score_venue_fit`, so population selection, gate recovery, permutation, recency baseline and verdict logic are byte-identical between arms. v2 is scored through `derive_venue_rules.score_card`. I ran two v2 variants because archived specs carry only 3 of v2's 6 declared card fields (`why_it_matters`, `what_we_would_do`, `what_counts_as_the_result` are absent): "declared" uses the fields as-written, "augmented" maps `target_population`/`design_memo` in so v2 reads the same text surface v1 does. Both give the same answer.

**Validation of my harness**: it reproduces the documented v1 external-only figure exactly (+0.0018 ≈ the published +0.002). This also proves the shipped diagnostic is **dead code** — see Defect D1.

### Venue-fit term only, with 20k-iteration permutation p

| Population | Predictor | rho | perm p | |
|---|---|---:|---:|---|
| **primary, n=5** (ledger + median-of-3 + EDM-calibrated) | v1 full table | **+0.7906** | 0.066 | |
| | v1 external-only | +0.7906 | 0.066 | VF-04 fires 5/5 → constant, ranks unchanged |
| | **v2 blind table** | **−0.6325** | 0.902 | **INVERTED** |
| | run-recency (no idea content) | **+0.9000** | **0.040** | beats both |
| **all_gated, n=24** | v1 full table | +0.3766 | 0.033 | |
| | v1 external-only | **+0.0018** | 0.494 | the honest v1 number |
| | **v2 blind table** | **+0.1467** | 0.245 | not distinguishable from zero |
| | run-recency | +0.2807 | 0.089 | still beats v2 |

Composite deterministic score via the unmodified script: v1 primary +0.7906 / all_gated +0.3217; v2 primary −0.6325 / all_gated +0.1436 (declared), +0.1477 (augmented). Recency collinearity with v2 at n=24 is **0.78**.

### Pre-registered pair (spec sec. 6 / sec. 9)

| Run | realized gate | v1 venue_fit | v2 venue_fit |
|---|---:|---:|---:|
| `phase_b_did_20260704` | 3.7 Reject | −2.00 `[VF-01]` | **−2.50** `[VF2-01, VF2-02]` |
| `stream1_did_v2_20260708` | 7.0 Accept | +1.50 `[VF-04]` | **−2.50** `[VF2-01, VF2-02]` |

**v2 scores them identically.** It cannot distinguish a 3.7 Reject from a 7.0 Accept on the pair whose only difference is the idea — the exact thing the table exists to encode. Not an inversion; a tie. Verdict from the harness: `FALSIFIED_INVERTED`, `pair separated: False`.

In fairness: v1 separates that pair **only because VF-04 was authored from it**. v1's success is circular by construction; v2's failure is honest.

### Verdict

**v2 is worse than v1's headline, no better than v1's honest number, and it fails the spec's gate clause outright.** Against `external-only +0.002` its `+0.147` is nominally higher but p = 0.245, beaten by a zero-content baseline, and 0.78-collinear with it. **This is a null result.**

**One caveat that matters, in v2's favour**: this backtest may not be a fair test of v2's *purpose*. v2 encodes "what these venues publish"; the outcome is an LSAR score on *our* papers. Three of its five positive rules describe study shapes our pipeline **cannot produce at all** (see §3). Their contribution here is not refuted — it is **untestable on this archive**. The null is partly a statement about our candidate space, not only about the table.

---

## 3. Per-rule discrimination

| Rule | sign | anchors (n=34) | **our specs (n=34)** | verdict |
|---|---|---:|---:|---|
| VF2-01 | neg | **0/34 (0.00)** | 23/34 (0.68) | **ANCHOR-DEGENERATE** (declared) |
| VF2-02 | neg | 2/34 (0.06) | **28/34 (0.82)** | **NEAR-CONSTANT on our slate** |
| VF2-03 | neg | 3/34 (0.09) | 2/34 (0.06) | discriminates |
| VF2-04 | pos | 15/34 (0.44) | **0/34** | **never fires** |
| VF2-05 | pos | 9/34 (0.26) | **0/34** | **never fires** |
| VF2-06 | pos | 7/34 (0.21) | **0/34** | **never fires** |
| VF2-07 | pos | 12/34 (0.35) | **0/34** | **never fires** |
| VF2-08 | pos | 10/34 (0.29) | **0/34** | **never fires** |

`applies_to` lists all five task types on every rule, so nothing is filtered out — the zeros are real.

**Most-generous probe** (measurement patterns against *every* string in the whole spec, ignoring the predicates): VF2-04 **0/34**, VF2-08 **0/34**, VF2-05 1/34 — structurally unproducible by this pipeline. VF2-06 21/34 and VF2-07 12/34 *would* fire, but their `predicate_strictness: stricter_than_measurement` predicates zero them out.

**On our own candidate space, v2 collapses to a 3-rule negative-only table, two of which fire on 68% and 82% of specs.** That is why the backtest is null: nearly every candidate gets the same penalty.

---

## 4–7. Design commitments — all verified behaviourally

**C1 — PASS.** AST scan of `src/ideation/*.py`: **0** reads or writes of any novelty-named field (all 21 textual hits are docstrings, banned-key lists, or sanitizer regexes). Behaviourally, `collision_check` returns exactly one numeric leaf, `$.retrieval.n_considered`; no novelty/score/rank key; output **byte-identical across 7 injected self-ratings** (0, 1, 3, 5, 10, 0.43, None). An LLM that invents `novelty`/`feasibility` dimensions has them dropped and the drop recorded as `c1_guard`.

**C4 — PASS.** A stubbed position-biased judge produced exactly **6 calls (2 orientations × k=3)**, **2 distinct messages** (the swap is real), no candidate id in any prompt, and:
`votes={'C-01': 3, 'C-02': 3}`, `orientation_winners={'AB': 'C-01', 'BA': 'C-02'}`, `position_bias=True`, **`winner=None`**. Disagreement is recorded, not averaged away. A consistent judge yields `position_bias=False`.

**C3 — PASS.** Across three regimes (biased judge / judging off / all-tie judge) the deterministic ordering is **complete (5 of 5) and identical**: `C-05, C-01, C-03, C-04, C-02`. The all-tie judge collapses onto it at Spearman 1.0. Shuffle control ran (40 replicates, tie rate recorded).

**#7 — PASS.** `advisory=true`, `authorized_for_live_selection=false`, `v2_status.cleared=false`. Both orderings publish side by side with Spearman + mean rank shift. `allow_live_selection=True` raises `LiveSelectionNotAuthorized`. The digest leads with *"the ranker is NOT validated"* and prints the +0.0018 out-of-sample number.

**Prior-art veto — PASS.** `is_veto`: COLLISION True, UNVERIFIABLE **False**, CLEAR False. A near-duplicate collides with a paperId and a snippet I confirmed is a verbatim substring of title+abstract. The *same* record with an empty `paperId` returns **CLEAR** — uncitable records cannot veto.

---

## 8. Ownership

`git diff --stat` is **empty** — zero modifications to tracked files. All 11 new files map 1:1 to exactly one agent; **no file claimed by two, none unclaimed**. Pre-existing untracked noise (`cache/`, `tmp_orch_test/`, `tmp_orch_test2/`, `runs/phase_a_.../pipeline_stdout.txt`) predates this slice.

`data_registry/venue_fit_rules_v2.yaml` is **ASCII-clean (0 non-ASCII bytes)**, as are all 7 other new files. Four *pre-existing* registry YAMLs are not ASCII (`hsls09_public.yaml` 3 bytes, `methodological_checklist.yaml` 141, `prediction.yaml` 38, `venue_norms.yaml` 5) — not this slice's doing.

---

## 9. Full suite

```
1945 passed, 13 skipped, 1103 warnings in 777.64s (0:12:57)
PYTEST_EXIT=0
```

Zero failures. Nothing to attribute. (All three agents failed to obtain this; it takes 13 minutes when the machine isn't contended.)

---

## Two agent claims that are FALSE

- **Agent 3**: *".gitignore — still lacks `ideas/`"*. It is present at line 58, under the comment `# Arc T idea tournaments (candidates, kills, rankings) — regenerable`.
- **Agent 3**: prior-art corpus re-read *"will dominate the prior-art stage's wall clock"*. Measured: 0.42 s/call warm, 0.015 s preloaded — ~5 s for a 12-candidate field. Real, but minor.

---

## Prioritized blocking list

**BLOCKERS — must resolve before v2 is wired to anything**

1. **Do not adopt v2 as the ranker.** It inverts on the calibrated population (−0.63), ties the pre-registered pair at −2.50, and measures +0.147 (p=0.245) where a zero-content baseline gets +0.28. It fails the sec. 9 gate on the pair clause alone.
2. **`src/ideation/venue_fit.py:313-315` silently zeroes an unknown table.** `_PREDICATES.get(str(rule.get("predicate")))` → `if predicate is None: continue`. Pointing `venue_rules_path` at v2 returns **`score=0.0, codes=[]` with no error and no warning** — I ran it. Anyone passing `--venue-rules …_v2.yaml` gets a silently inert venue-fit term. Must raise or warn on zero recognized predicates.
3. **`scripts/backtest_ranker.py:779` is dead code.** `getattr(r, "venue_fit_rule_deltas", None)` reads an attribute set nowhere in the repo, so `_external_only_rho` **always returns `None`** and the script prints `external-only rho: n/a`. The documented +0.002 cannot be reproduced by the shipped script — the guardrail the v1 table's own header calls load-bearing is not running. My wrapper shows the fix is ~5 lines in `build_rows`.

**HIGH**

4. **Redact `docs/v5_arc_t_spec.md` lines 272 and 275.** They reprint VF-01/VF-04's in-sample evidence including "3.7 Reject" and "3.7 → 7.0". Any future blind derivation is pre-contaminated.
5. **Correct spec line 563.** The gate `"rho > 0"` is satisfied by a headline number whose substance is null. `tournament.py` refuses in code regardless, but the sentence would license a future reader to flip advisory mode off legitimately.
6. **Record in `docs/backlog.md`** that v2 measured null, with the per-rule firing table above — otherwise the same 8 rules get re-derived.

**MEDIUM**

7. **`config.yaml` has no `ideation:` block at all.** Every tuned value in all three slices (judge_samples 3, temp 0.2, purpose_coverage_min 0.60, BT weights 0.30/0.20, prior_sd 1.0, anchor corpus path) is a code default. `priorart.DEFAULT_ANCHOR_CORPUS` is a hardcoded machine path. This violates the project rule and was already open from T1a.
8. **Nothing consumes `venue_fit_rules_v2.yaml`** — confirmed, only its own audit script and test file reference it. It is inert on disk today, which is the correct state given §2.
9. Zero live LLM calls were made anywhere in this slice. Real position-bias rate, tie rate, and false-veto rate against a live S2/arXiv pool remain unmeasured. Stage 4 (absolute AE screen on the top 2) is not implemented.

**The prior-art veto and the judged layer are in materially better shape than the rule table** — C1/C3/C4 hold under adversarial stubs, the veto is structurally conjunctive, and the tournament refuses live selection in code. The rule table is the part that measured zero, and it measured zero honestly.