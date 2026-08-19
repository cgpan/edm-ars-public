<!-- Arc T spec, generated 2026-07-11 from run history + anchor corpus + SOTA. -->

# Arc T (Research Taste) — Implementable Specification

<!-- BLIND-DERIVATION PROTOCOL (2026-07-11) -->
> **If you are deriving venue-fit rules, read this first.**
>
> Sections 6 (VALIDATION) and 7 (COST) below reference OUR OWN runs and their
> realized gate scores. A rule table derived by an agent that has read them is
> contaminated: it cannot then be validated against those same scores. This
> already happened once -- the T1b "blind" derivation was contaminated on
> arrival because the brief told it to read this spec.
>
> Derivation agents must read sections 0-4 ONLY, and must not open
> `docs/v5_arc_t_v2_backtest_verdict.md`, `docs/v5_arc_t_t1b_verdict.md`,
> `evaluation/ledger.json`, or anything under `runs/`.


**Repo**: `<repo root>` (worktree `.claude/worktrees/distracted-satoshi-9078fa`, branch `phase-3-causal-inference`)
**Depends on**: Arc P (complete). **Blocks**: Arc I (I1 benchmark battery reuses the feasibility screen and the ledger fields added here).

**One-line thesis**: the field's ideation literature fails because feasibility is invisible at idea time and LLM novelty judgements are inverted. We can compute feasibility deterministically from registries and data on disk, and we can refuse to ever emit a positive novelty number. Arc T is therefore **a deterministic screen with a small judged ranking bolted on**, not an LLM idea generator with a scorer.

---

## 0. Design commitments (these constrain everything below)

| # | Commitment | Forced by |
|---|---|---|
| C1 | No positive novelty score is ever computed, stored, or ranked on. Novelty exists only as a **veto** with a citation and a quoted snippet. | HindSight rho=-0.291; our own `novelty_score_self_assessment` r=-0.35 vs LSAR Novelty, 8 of 11 values literally the number hard-coded at `agent_prompts/problem_formulator.yaml:50` |
| C2 | Every score component must name the artifact feature it read, in an `evidence` string. A component that cannot cite what it read is deleted, not shipped. | Arc P lesson: citation count hit its venue target while the bibliography filled with 2024-26 oncology papers |
| C3 | Deterministic first, judged last, and the judged layer must be **removable** — the tournament must still produce a defensible ordering with all LLM verdicts shuffled. | Project rule; Arc P's real wins were deterministic |
| C4 | Every judged verdict is order-swapped and k-sampled (k=3, median). Single-sample judgements are banned. | 6.5-vs-3.1 swing on an identical manuscript; LSAR test-retest MAD 1.9; within-run Novelty spread 1.90 |
| C5 | The winner is consumed by the **existing** `--research-spec` locked-spec path. No new orchestrator state. | Parallel work fails at the seams; 20 of 26 archived runs already used the locked path, it is the best-tested code in the repo |
| C6 | The human never blocks the pipeline and is never asked to rate on a scale. Ranking + categorical tags only, ≤5 minutes, file-based. | Scarcest resource; expert-expert agreement on ideas is 56.1% balanced accuracy, so a rating scale would be recording noise at high precision |
| C7 | LSAR / `src/review_gate.py` is never touched, never trained on, never inside any loop. | Same rail as Arc I; it is our only externally calibrated instrument |

---

## 1. WHERE IT PLUGS IN

### 1.1 Position: a pre-pipeline stage, not a PF rewrite

Arc T runs **before** `src.main`, produces a locked `research_spec.json`, and hands it to the unchanged pipeline:

```
scripts/run_idea_tournament.py  →  ideas/<tid>/winner_spec.json
                                        ↓
python -m src.main --research-spec ideas/<tid>/winner_spec.json --output-dir runs/<run>
                                        ↓
              existing FORMULATING (refine mode) → ENGINEERING → ... → REVIEWING
```

`src/main.py::load_locked_research_spec()` already validates and locks; `ProblemFormulator._build_user_message()` already renders `## Locked Research Spec (refine, do not redesign)`. That path carried 20 of the 26 archived runs. Arc T's job is to **author the fixture that until now a human wrote by hand** (`runs/fixtures/*.json`, 7 files).

### 1.2 New modules

```
src/ideation/
  __init__.py
  cards.py           IdeaCard dataclass; fixed-template render; compile_spec()
  slate.py           deterministic (dataset × task_type × pattern × persona) enumeration
  generate.py        T1a: independent per-cell LLM draws; dedupe
  probe_cache.py     Tier-1 parquet column cache (9.3 s build, 0.06 s reads)
  feasibility.py     STAGE 0/1 deterministic screen — the crown jewel
  venue_fit.py       deterministic anchor-derived rule table
  priorart.py        STAGE 2 retrieval + facet rerank + collision veto
  judge.py           STAGE 3 pairwise judge (order-swapped, k=3, flash)
  bradley_terry.py   MAP fit + Laplace posterior + top-k membership probability
  tournament.py      cascade orchestration; writes all artifacts
  taste.py           T2 preference store, judge-bias offsets, exemplar management
scripts/
  run_idea_tournament.py
  taste_session.py            open | ingest
  audit_feasibility.py        replay the screen over the archive (validation)
data_registry/
  venue_fit_rules.yaml        anchor-derived deterministic rules
taste/                        (gitignored except schema README)
  preferences.jsonl
  judge_bias.json
  exemplars.md
ideas/<tournament_id>/        (gitignored; digests committed selectively)
```

### 1.3 Artifacts written (every one is a file; nothing lives only in a log)

| Path | Content |
|---|---|
| `ideas/<tid>/slate.json` | RNG seed, every enumerated cell, quota decisions, which cells were sampled |
| `ideas/<tid>/candidates.jsonl` | every generated card, raw, one JSON per line |
| `ideas/<tid>/killed.jsonl` | every killed card + the KILL code + evidence (**no rejected idea has ever been written to disk in this repo — this file is the training data for everything downstream**) |
| `ideas/<tid>/feasibility.json` | per-candidate `FeasibilityReport` (all checks, not just failures) |
| `ideas/<tid>/priorart/<cid>.json` | nearest prior work, snippet, delta sentence, verdict |
| `ideas/<tid>/matches.jsonl` | every pairwise match: pair, orientation, sample index, verdict, evidence string |
| `ideas/<tid>/ranking.json` | BT posterior means, SDs, top-2 membership probabilities, deterministic tie-break trace |
| `ideas/<tid>/winner_spec.json` | locked research_spec; **must pass `load_locked_research_spec()`** |
| `ideas/<tid>/tournament.md` | human digest incl. the **diversity ledger** and the artifact-audit lines (C2) |
| `ideas/<tid>/taste_session.md` | the 5-minute human artifact (§5) |

### 1.4 Exact edits to existing files

| File:line | Change | Why |
|---|---|---|
| `src/orchestrator.py:239-243` | `n_branches = fm_cfg.get("n_candidate_specs", 1)` — **decouple from `findings_memory.enabled`** | that gate is why multi-branch executed 0 times in 34 archived runs |
| `src/orchestrator.py::_save_formulating_outputs` | if `ctx.idea_card` is set, write `idea_provenance.json` (tournament_id, candidate_id, rank, feasibility report, prior-art verdict) into the run dir | provenance must travel with the run for the ledger join |
| `src/context.py::PipelineContext` | add `idea_card: dict | None = None`, `tournament_id: str | None = None`; checkpoint `schema_version` → `"1.1"`, all reads via `.get` | resume-safe |
| `src/agents/problem_formulator.py::_build_user_message` | new optional `idea_card` section rendered as `## Idea Card (tournament-selected — refine wording, do not redesign)` immediately after the locked-spec block | PF must not silently redesign the winner |
| `src/agents/problem_formulator.py::_select_best_candidate` | **delete the `novelty_score_self_assessment` term**; delegate the whole function body to `src.ideation.feasibility.rank_key(spec)` | C1; it is the only differentiating term today and it is anti-correlated with the criterion |
| `src/agents/problem_formulator.py::_build_registry_var_map` | move to `src/registry.py::build_var_map(registry)`; keep a re-export alias | feasibility.py must not import an agent module |
| `src/main.py:48` | `template.validate_research_spec(spec, registry, dataset_adapter)` — currently `TypeError`s for `task_type: "prediction"` | the tournament's acceptance test runs through this function |
| `src/main.py` `--dry-run` branch | call `src.ideation.feasibility.screen()` and print the report | free win; makes the screen exercised on every dry run |
| `src/task_template.py:123-144` (`PredictionTemplate`) | resolve predictor wave from `var_map[name]["wave"]`, **not** `pred.get("wave")` | measured hole: declaring `X3TGPAMAT` as `wave: base_year` produces 0 warnings today |
| `src/task_template.py:146-167` | replace the sum-of-`pct_missing` retention rule with `feasibility.estimate_analytic_n()` when the probe cache exists; keep the sum rule as a labelled fallback | current rule returned 0 (false abort warning) where truth was 12,960 |
| `src/design_selector.py::did_feasible` | require the dataset to BE a harmonized panel (`design_feasibility.panel_ready: true`) or to declare `policy_timing_variables`; `multi_cohort_partner` alone yields `feasible=True, executable_task_type=None` | today `select_design("hsls09_public", intent="causal")` routes to a task type that cannot execute on HSLS |
| `src/design_selector.py::itr_feasible` | read `design_feasibility.itr_ready` (currently written by `onboard_dataset.py:120` and **never read**) | ASSISTments reports ITR-feasible with 0 protected attributes and `itr_ready: false` |
| `src/design_selector.py::classify_intent` | word-boundary regex (`\bate\b`, `\batt\b` are substring-matching *attitudes*, *climate*, *attendance*, *attainment*) | 4 of 7 forensic phrasings misrouted |
| `src/registry.py::RegistryLoader.is_excluded` | wire into `feasibility.check_tier3_exclusion` | zero production call sites today |
| `evaluation/ledger.json` schema | add `tournament_id`, `candidate_id`, `candidate_rank`, `arm` ∈ {`rank1`,`rank_median`,`baseline`,`none`}; also fix the three `run_dir` values containing a literal vertical tab from an unescaped `\v4` | the paired A/B in §6 joins on these |
| `tests/conftest.py` | redirect `findings_memory.path` to a tmp path for the whole suite | `findings_memory/memory.yaml` is 63/63 pytest residue and the suite rewrites it on every run |

**Not touched, by rule**: `src/review_gate.py`, `src/manuscript_linter.py`, LSAR repo, `agent_prompts/critic.yaml`.

### 1.5 Config block (append to `config.yaml`)

```yaml
ideation:
  enabled: true
  tournament:
    n_candidates: 24              # see §2.3 for the justification
    bridge_framing_quota: 3       # hard cap, ≤15% (LLMs default to 47-64%)
    dedupe_cosine: 0.80
    judge_samples: 3              # k, median-aggregated
    order_swap: true
    swiss_rounds: 5               # full round-robin instead when survivors ≤ 7
    max_survivors_to_tournament: 12
    random_state: 42
  models:                          # judge MUST differ from generator (self-enhancement +10-25%)
    generator: deepseek-v4-pro
    judge: deepseek-v4-flash
    ae_screen: deepseek-v4-pro     # absolute gate on top-2 only
  probe_cache:
    path: cache/tier1/
    rebuild_if_older_than_days: 30
  taste:
    session_cards: 5              # 4 max-uncertainty + 1 anchor
    human_pair_weight: 5.0        # vs 1.0 per judge match
    bias_offset_min_pairs: 20     # offsets pinned to 0 below this
    exemplar_cap: 6               # per polarity, FIFO
  priorart:
    anchor_corpus: '$LSAR_HOME/outputs'   # 34 local anchors, seed the retrieval
    unverifiable_is_not_clear: true
```

---

## 2. GENERATION

### 2.1 The problem being solved is upstream of the tournament

Measured: 20 of 26 archived runs never asked PF to invent anything; the 6 free-generation runs got a 4-word prompt and returned specs with **mean pairwise predictor Jaccard 0.837** (min 0.700, one pair a literal subset), all 5 on the same outcome. 100% of all 26 runs are mathematics. A temperature ramp over 3 draws from one anchored prior cannot fix that. **Diversity must be enforced structurally, in the slate, before any LLM call.**

### 2.2 Slate enumeration (`src/ideation/slate.py`, deterministic, zero LLM)

Four orthogonal axes, enumerated then quota-sampled with `random_state=42`:

1. **Dataset × task_type cell** — the cross product filtered by `feasibility.DATASET_TASK_MATRIX` (new; does not exist anywhere today, so nothing stops `--dataset hsls09_public --task-type causal_did`). 4 datasets × 5 task types = 20 cells; ~11-13 survive.
2. **Opportunity pattern** (8) — this is the diversity axis that matters, not topic:
   `puzzle_anomaly | explanation_gap | measurement_bottleneck | equity_subgroup_gap | replication_transfer | design_upgrade | scope_extension | robustification`
   Plus `bridge_synthesis`, **hard-capped at 3 of 24**. LLMs default to 47-64% bridge framings vs 12.1% for humans, with normalized opportunity-type entropy 0.550-0.758 vs 0.926.
3. **Persona** — ordinary heterogeneous research roles (`psychometrician | equity researcher | policy analyst | replication methodologist | measurement-to-decision analyst | causal econometrician`), each with CoT. Explicitly **not** celebrity-creative personas, which sample from a densely connected region and reduce diversity.
4. **Gap cell** — one sparse cell from `src/gap_miner.py::build_gap_matrix(s2_context)["sparse_cells"]`, assigned round-robin so no two candidates get the same cell twice before all are used.

Each candidate gets exactly one assignment on each axis, written to `slate.json` before generation. A cell is never assigned to more than 3 candidates.

### 2.3 Count: 24 generated. Justification

- **8 opportunity patterns × 3 independent draws** = full pattern coverage with enough redundancy that one bad draw does not eliminate a pattern.
- Over-generation saturates: 4,000 seeds per topic yielded ~5% non-duplicates, monotonically worsening. Planned per-candidate retrieval yields >80% unique. 24 planned draws is on the right side of that curve; 200 unplanned draws is not.
- Attrition budget: 24 generated → ~18-20 after dedupe → ~10-12 after the deterministic KILL screen → ~8-10 after prior-art veto. A Swiss tournament wants 8-12 entrants. The count is set by the tournament's input requirement, working backwards through measured attrition.
- Cost: 24 pro calls ≈ 120k tokens ≈ well under $1 (§7).

### 2.4 Generation mechanics

- **Independent sampling. No cross-agent discussion, no debate, no shared scratchpad.** Agents that communicate converge; the effect intensifies with communication rounds. This deliberately abandons the current `prior_specs` anti-repetition injection at generation time — the slate does that job better and deterministically.
- **Fixed temperature 0.9 for all draws.** The existing `0.70 / 0.85 / 1.00` ramp confounds diversity with quality (candidate 1 is systematically the most conservative *and* wins every tie because `scores.index(max(scores))` resolves to index 0).
- **One planned-retrieval loop per candidate** (Nova-style, 2 iterations): draft → S2/arXiv query built from the card's facets → refine against what came back. Contrast with today, where `s2_context` is fetched **once before the loop and shared by all branches**, so candidates cannot differentiate on retrieval.
- **Exemplars** from `taste/exemplars.md` (≤6 positive, ≤6 negative) are injected into the generator prompt only — never the judge, never LSAR.

### 2.5 Dedupe — two keys, both required

```python
def is_duplicate(a: IdeaCard, b: IdeaCard) -> bool:
    structural = (a.resolved_target, a.method_family, a.dataset) == (b.resolved_target, b.method_family, b.dataset)
    lexical    = tfidf_char4_cosine(a.render(), b.render()) >= 0.80
    return structural or lexical
```

`resolved_target` is a task-type-agnostic resolver — `outcome_variable` **or** `outcome.variable` **or** `treatment.variable` **or** `scale_name`. `outcome_variable` exists in only 6 of the 26 shipped specs, which is why two of the four terms in today's selector silently no-op for causal and psychometrics runs.

Similarity uses `sklearn` TF-IDF char 4-grams (already a dependency, fully deterministic, no new model). Optional upgrade to SPECTER2 embeddings is noted but **not** required — the structural key catches the case embeddings miss (identical substance, different prose), which is exactly the 0.837-Jaccard failure we measured.

---


> **REDACTED FOR BLIND DERIVATION (2026-07-11).** This section previously
> reprinted the in-sample evidence of venue-fit rules VF-01/VF-04 verbatim,
> including our own realized gate scores for `phase_b_did` and
> `stream1_did_v2`. Any agent instructed to read this spec before deriving
> venue rules was therefore contaminated on arrival -- which is exactly what
> happened in the T1b blind derivation. The numbers live in
> `docs/v5_arc_t_v2_backtest_verdict.md`, which blind derivations must not
> read. Do not restore them here.

## 3. SCORING

A four-stage cascade. Cheapest and most reliable first. **Kill before you score.**

### Stage 0 — DETERMINISTIC hard filter (`src/ideation/feasibility.py::screen`)

Free (<0.2 s/candidate, no data load). Runs on all 24.

```python
@dataclass
class CheckResult:
    code: str              # e.g. "F-VAR-ABSENT"
    status: str            # KILL | WARN | OK
    message: str
    evidence: str          # C2: the artifact fact this was read from

@dataclass
class FeasibilityReport:
    candidate_id: str
    verdict: str           # KILL | WARN | CLEAN
    checks: list[CheckResult]
    analytic_n_estimate: int | None
    penalty: float         # sum of WARN weights; enters ranking, never kills
```

| Check | Function | Status on failure | Grounding |
|---|---|---|---|
| Dataset × task_type executable | `check_dataset_task_compatibility` (new `DATASET_TASK_MATRIX`) | KILL | does not exist anywhere today |
| Every named variable exists in registry | `check_variables_exist_in_registry` (via `registry.build_var_map`) | KILL | predictor existence is **unchecked** today; an invented plausible name passes silently |
| Every named column exists in the actual CSV | `check_columns_exist_in_csv` (`probe_cache.header_columns`) | KILL | 0.11 s for all four datasets; of 151 registry names exactly one (`dropout_derived`, `derived: true`) is absent — free insurance against registry drift |
| Temporal ordering, wave resolved **from registry** | `check_temporal_order` | KILL | the one-line fix at `task_template.py:123-144` |
| Tier-3 exclusion (weights, IDs, imputation flags) | `check_tier3_exclusion` (`RegistryLoader.is_excluded`) | KILL | machinery exists, zero call sites; `['W1STUDENT','STU_ID']` → 0 warnings today |
| Dead variable (`pct_missing >= 99`) | `check_dead_variables` | KILL | 7 dead HSLS variables; none detected today |
| Estimator on the certified list | `check_estimator_certified` (`Template.SUPPORTED_METHODS`) | KILL | RD/IV are certified-but-shelved → any RD/IV identification is infeasible **today**, checkable in <1 ms |
| Design feasible for the identification claim | `check_design_feasible` (`select_design`, post-patch) | KILL | after the `did_feasible`/`itr_feasible` fixes |
| Structural completeness for the task type | `check_structural_completeness` (`validate_research_spec`) | KILL | already implemented per task type |
| Protected attributes exist (equity patterns only) | `check_protected_attributes` | KILL | ASSISTments has 0; any equity RQ there is deterministically infeasible and nothing says so today |
| Item bank ≥3 items/factor; reverse flags known | `check_item_bank_adequacy` | KILL | `math_identity` has 2 items and the registry says so |
| Subgroup vars are real | `check_subgroup_variables` | WARN | `['X1GENDERIDENTITY']` → 0 warnings today |
| Tier-2 / uncurated variable used | `check_metadata_verified` | WARN (+penalty) | ELS Tier-2 reports `pct_missing: 0.00` for all 4,012 vars because the profiler is not sentinel-aware; the HSLS Tier-2 file the registry points at **does not exist** |
| Registry-documented pitfall touched | `check_common_pitfalls` | WARN | 11 of 13 pitfalls are LLM-instruction-only today |

**KILL is reserved for logically dispositive facts.** Everything probabilistic is WARN with a penalty weight and enters the tournament. See risk R3.

### Stage 1 — DETERMINISTIC data probes (`feasibility.probe`), survivors only

Needs `src/ideation/probe_cache.py`: a one-time Tier-1 parquet cache — 128 columns, 9.3 s build, 2.4 MB (vs a 1,998 MB source CSV), then 0.06 s per read, ~140× speedup. This is the single lever that moves the honest checks from MEDIUM cost into the free tier.

| Probe | Function | Note |
|---|---|---|
| True complete-case n, sentinel-aware via registry `range` | `estimate_analytic_n` | reproduced curated `pct_missing` to within 0.73 pp on 8/8 HSLS vars; the naive "any negative is missing" rule understates n by 38% |
| Binary outcome class balance | `check_class_balance` | X4EVRATNDCLG measured 75.3/24.7 |
| **Causal positivity / treatment-arm adequacy** | `check_positivity` | same 0.10 extreme-tail threshold `src/causal_data_contract.py` enforces — but **pre-run**, at 0.08 s of compute, instead of after ~$2-3 of pipeline spend. Highest-value single probe in the design. |
| DiD 2×2 cell counts | `check_did_cells` | 1.3 MB panel, 0.02 s; an emptied cell is detectable in 20 ms |
| CDM/IRT scope adequacy | `check_cdm_scope` | against `cdm_support.recommended_scope`; full 83 MB log reads in 0.95 s |

### Stage 2 — Novelty as VETO ONLY (`src/ideation/priorart.py`) — JUDGED, but only negatively

```python
def collision_check(card) -> dict:
    # returns {"verdict": "CLEAR"|"COLLISION"|"UNVERIFIABLE",
    #          "nearest": [{paperId,title,year,venue,snippet}, ...],
    #          "delta_sentence": str|None}
```

1. Decompose the card into three facets — **purpose** (population + outcome), **mechanism** (design + estimator), **evaluation** (what would count as the result).
2. Retrieve via `ProblemFormulator._search_literature()` (existing S2 + arXiv + Jaccard dedup) **seeded with the 34 local anchors** at `$LSAR_HOME/outputs`.
3. **Rerank the retrieved set by facet match, not topical relevance.** This is the lever: facet-based reranking scored 89.66% vs 13.79% for general-relevance reranking on a 58-idea set. (Caveat: the 13.79% baseline is below chance and probably degenerate; trust the direction, not the 6.5× magnitude.)
4. Emit the **single nearest prior work + a quoted snippet + one explicit delta sentence**. Veto if no defensible delta.

**How this avoids both traps, explicitly:**

- **Self-assessment trap**: `novelty_score_self_assessment` is never read by any Arc T code path. It is left in the spec (so `validate_research_spec` still passes) and is asserted-unused by `tests/test_ideation_ranking.py::test_ranking_invariant_to_self_novelty`, which perturbs the field across its whole range on a fixed candidate set and asserts the ranking is byte-identical. It is a prompt echo (8 of 11 in-scale values = the 4 hard-coded at `problem_formulator.yaml:50`), type-unstable (2 dicts, 1 float `0.43` that the current selector silently scores as zero), and scale-inconsistent (1-5 for prediction, 7-on-1-10 for causal.)
- **Anti-correlation trap**: no positive novelty number exists to rank on. A "no paper does this" claim is an unfalsifiable absence-of-evidence claim and is exactly the claim that anti-correlates with impact. Only the negative claim ("this paper already did it, here is the sentence") is verifiable, and only the negative claim is acted on. Corroborating our own corpus: 2 of 8 bottom-band Novelty reviews were punished specifically for an **unsupported** first-claim ("Claims of being the first CDM analysis of ASSISTments are not convincingly supported") — so an unsubstantiated first-claim scores worse than no claim.
- **UNVERIFIABLE is a third state, not CLEAR.** S2 coverage of EDM/LAK/JEDM/JLA is thinner than of arXiv cs.CL, so a clean retrieval is weak evidence. UNVERIFIABLE downweights and is surfaced to the human session.

### Stage 2b — Venue fit: DETERMINISTIC rule table (`src/ideation/venue_fit.py` + `data_registry/venue_fit_rules.yaml`)

Mined from the 34-anchor corpus. Three measured structural facts drive it:

| Rule | Effect | Evidence |
|---|---|---|
| `VF-01` bare observational causal contribution, no second contribution | −2.0 | **0 of 34** anchors run an observational causal estimator as the contribution. [IN-SAMPLE EVIDENCE REDACTED: this rule was additionally justified by one of our own runs and its gate score, which is why it cannot validate a ranker against those scores.] |
| `VF-02` SHAP/feature-importance as the headline | −1.0 | SHAP in 1 of 34; feature importance in 3 of 34 |
| `VF-03` prediction with AUC as the only contribution | −1.5 | AUC mentioned in only 6 of 34; the closest "prediction paper" anchor scored 5.4 Borderline (Rigor 3 / Empirical 2), surviving on its generalizability framing |
| `VF-04` carries a second contribution ∈ {transfer, fairness, measurement-chained-to-decision, replication} | +1.5 | [IN-SAMPLE EVIDENCE REDACTED: justified by a score recovery between two of our own runs on identical data. See the backtest verdict doc, which blind derivations must not read.] |
| `VF-05` cross-context transfer/generalizability claim | +1.5 | jla_9099's entire thesis; jedm_974 external validation at 7.3; learning-curve replication 6.6 |
| `VF-06` measurement model chained to a downstream decision | +1.0 | jla_9035 (6.8) is the only anchor with a measurement model at its core, and it works because CFA serves an institutional decision |
| `VF-07` synthetic-DGP certification as the *only* empirical support | −1.0 | bkt_parametric_constraints scored Empirical Support 5: "Experiments are restricted to synthetic data" |
| `VF-08` venue Ethics weighting | multiplier | Ethics weight 1.0 at EDM, 0.5 JEDM, 0.6 JLA — a fairness-forward framing buys most at the conference |

Deterministic, auditable, each with an `evidence` string. **Do not ground this in `venue_criteria/jedm_2026.yaml` / `jla_2026.yaml`** — those are three generic lines each, still declare `calibrated: false` despite `calibration/anchors_edm.yaml` recording JEDM P25 5.15 and JLA P25 5.4 from these very anchors, and their stated topics ("knowledge tracing, prediction, causal EDM") describe a conversation that 8 of 10 JEDM anchors are not having.

**Data-quality prerequisite**: the EDM 2024 anchor directory stems are rotated relative to their contents (`theory_building_dbr_*/paper.md` is actually the DRL pedagogical-policy paper). Key on the `paper.md` title, never the stem. `metadata.json` title is empty or single-character for 8 of 15.

### Stage 3 — JUDGED pairwise ranking (`src/ideation/judge.py`)

Dimensions judged (deliberately **excluding novelty and feasibility**):

| Dimension | Why judged | Noise control |
|---|---|---|
| Significance / contribution size | not computable | pairwise, order-swapped, k=3 median |
| Venue-conversation fit (residual beyond the rule table) | rules cover structure, not fit within structure | same |
| Clarity & bottleneck specificity | highest human-agreement dimension (76.3% in Idea Arena) | same |
| Framing quality (is the "so what" named?) | not computable | same |

- **Pairwise, not absolute**, for ranking: pairwise 53.3% vs direct scoring 50-51.7% against experts on ideas; 71.4% on papers.
- **Both orders, k=3 samples each = 6 votes per pair.**
- **Judge writes its evidence string before its verdict** (multiple evidence calibration).
- **Fixed card template, hard 120-word cap, truncated before judging** — length carries zero signal (verbosity attacks succeed ~91% on weaker judges).
- **Judge model ≠ generator model** (`deepseek-v4-flash` judge vs `deepseek-v4-pro` generator); self-enhancement bias is +10% to +25% on own outputs.

### Stage 4 — JUDGED absolute AE screen, gate not rank (top-2 only)

An LSAR-derived Associate-Editor screening persona emits an **absolute rubric score with a floor**, k=3 median, on `deepseek-v4-pro`. Absolute is correct here precisely because our own generator is the adversary: absolute scores flip 9% under injected distractor features versus ~35% for pairwise. If both top-2 fall below the floor, the tournament returns **no winner** and reports why — that is a legitimate, valuable outcome.

**Explicitly not** a second reviewer competing with LSAR. It never sees a manuscript, only an idea card, and it cannot promote a candidate — only block one.

---

## 4. SELECTION

### 4.1 Tournament structure

- Survivors ≤ 7 → **full round-robin**. 8-12 → **Swiss, 5 rounds**. >12 → truncate to the 12 with the lowest feasibility penalty (recorded, not silent).
- Every pair: 2 orientations × 3 samples = 6 binary outcomes.
- **Fit Bradley-Terry over all match outcomes; do not do sequential Elo updates.** LLM judges have non-transitive preferences, so a ranking from sequential or single-baseline comparison depends on the ordering. Round-robin + BT lifted Spearman 95.0 → 96.4 and Kendall 82.1 → 86.3 against a human-preference reference.

```python
# src/ideation/bradley_terry.py
def fit(matches, weights, prior_sd=1.0, seed=42) -> BTPosterior:
    """MAP fit with a Gaussian prior; Laplace approximation for the covariance."""
def top_k_membership(posterior, k=2, n_draws=2000, seed=42) -> dict[str, float]:
    """P(candidate in top-k) by sampling the Laplace posterior. Deterministic given seed."""
```

Deterministic terms enter as **prior offsets on the BT strengths**, not as post-hoc additions:
`prior_mean_i = w_vf * venue_fit_score_i - w_pen * feasibility_penalty_i` (weights in config, defaults 0.30 / 0.20 chosen so the deterministic prior can move a candidate roughly one rank but cannot override a unanimous 6-0 judged sweep).

### 4.2 Tie-breaks (fully deterministic, in order)

1. BT posterior mean
2. Fewer WARN codes in the feasibility report
3. Higher deterministic `venue_fit_score`
4. Prior-art verdict CLEAR > UNVERIFIABLE
5. Opportunity-pattern diversity vs the last tournament's winner (prefer the pattern not used last time)
6. Lexicographic `candidate_id` (seeded, so reproducible)

Rule 5 is load-bearing: **no two consecutive tournaments may put the same (dataset, outcome-family) pair at rank 1.** With 53.8% of archived runs on one outcome and 100% mathematics, an unconstrained optimizer will re-derive that concentration.

### 4.3 Winner → locked spec

```python
# src/ideation/cards.py
def compile_spec(card: IdeaCard, feasibility: FeasibilityReport) -> dict:
    """Emit a research_spec dict in the exact shape of runs/fixtures/*.json
    for the card's task_type. Fills task_id, task_type, dataset, research_question,
    and the task-type-specific blocks (predictor_set | treatment/outcome/
    target_estimand_hint/primary_method | group_variable/post_variable |
    scale_name/item_columns). Carries the measured analytic_n and the
    prior-art delta sentence into `expected_contribution`."""
```

**Acceptance test (CI, blocking)**: `tests/test_ideation_seam.py::test_winner_spec_loads_unchanged` compiles a winner for each of the 5 task types and asserts `src.main.load_locked_research_spec(path)` returns without raising. This is the seam, and it is tested with the exact function the CLI uses.

Then the run is launched by the existing command, unchanged:

```
python -m src.main --research-spec ideas/<tid>/winner_spec.json --output-dir runs/<run_id>
```

---

## 5. THE HUMAN ANCHOR

### 5.1 Principle

Do not ask the human to rate. Ask them to break the ties the machine cannot break, and use the answers to correct the **judge**, not just to pick the idea. One human ranking is a weak label, not ground truth — expert-expert agreement on ideas is 56.1% balanced accuracy.

### 5.2 Interface: a markdown file the user edits in place

```
python scripts/taste_session.py open <tid>       # writes ideas/<tid>/taste_session.md
# ... user edits the YAML block at the top, saves, closes ...
python scripts/taste_session.py ingest <tid>     # parses, appends, refits, prints the delta
```

The file:

```markdown
---
# EDIT THESE TWO BLOCKS ONLY. Everything below the line is read-only context.
ranking: [ , , , , ]        # candidate ids, best first
tags:                        # one or more per card
  C-07: [ ]                  # strong | boring-framing | already-done | not-feasible |
  C-11: [ ]                  # wrong-venue | mis-specified
  C-03: [ ]
  C-19: [ ]
  C-02: [ ]                  # <- anchor card, ranked by you on 2026-07-18
note: ""                     # optional, one line
---

## C-07  (prediction · els_2002 · replication_transfer)
[120 words, fixed template: Question / Why it matters / What we'd do /
 What would count as the result / Nearest prior work + delta / n = 12,960, all checks CLEAN]
...
```

### 5.3 Which 5 cards

- **4 by maximum top-2 membership uncertainty** under the BT posterior — not the 4 highest posterior means. Showing the obvious leaders wastes the session.
- **1 anchor**: a card from a previous session whose human ranking is known, re-shown verbatim. Detects human drift and gives a within-session consistency estimate.
- **Plus, every session, exactly one KILLED card** shown in a separate "sanity" block with its KILL code, asked only "was killing this correct? y/n". This is the only false-negative detector the deterministic screen will ever have (risk R3).

### 5.4 Cadence and non-blocking rule

- Triggered when the top-2 posterior credible intervals overlap by more than a configured threshold, **or** every 3rd tournament regardless (to keep collecting anchor drift).
- Maximum one session per day.
- **The tournament never blocks.** If the file is not ingested within a configured window, the run proceeds on the BT ranking and the session is marked `skipped`. The human is advisory by construction.

### 5.5 Storage

```jsonl
# taste/preferences.jsonl  (append-only)
{"session_id":"S-004","ts":"2026-07-25T21:10:00Z","tournament_id":"T-0007",
 "shown":["C-07","C-11","C-03","C-19","C-02"],
 "ranking":["C-11","C-07","C-19","C-03","C-02"],
 "tags":{"C-03":["already-done"],"C-19":["boring-framing"],"C-11":["strong"]},
 "anchor":{"card":"C-02","prior_rank":2,"this_rank":5},
 "kill_probe":{"card":"C-22","code":"F-ITEM-BANK-TOO-FEW","user_agrees":true},
 "note":"", "elapsed_s":268}
```

### 5.6 Three feedback paths, each with a rail

| Path | Mechanism | Rail |
|---|---|---|
| **A. This tournament's ranking** | the 10 implied pairs enter the same BT fit at `human_pair_weight: 5.0` vs 1.0 per judge match | only if the session lands before the decision; otherwise recorded for B and C only |
| **B. Judge bias correction** | per-dimension additive offset fit by minimizing accumulated human-judge disagreement; written to `taste/judge_bias.json`; applied to future tournaments | **pinned to 0.0 until ≥20 human pairs across ≥4 sessions.** Below that, offsets are computed and logged but not applied. Bias is only identifiable under specific conditions, and n=5 is not one of them. |
| **C. Generator exemplars** | top-ranked card → positive exemplar; bottom-ranked-with-a-tag → negative exemplar, tagged with the reason; `taste/exemplars.md`, capped at 6 per polarity, FIFO | injected into the **generator only**. Never the judge (that would close the loop), never LSAR (Arc I rail). |

**Tags carry more information than the ordering at n=5.** `already-done` becomes a prior-art query seed; `not-feasible` becomes a candidate new deterministic check (and if it recurs 3×, a backlog item to make it deterministic); `wrong-venue` becomes a proposed `venue_fit_rules.yaml` rule with a human-visible diff. The tag → deterministic-rule promotion path is how the human's 5 minutes compound instead of evaporating.

---

## 6. VALIDATION

The exit criterion in `docs/v5_roadmap.md` ("user blind-ranks tournament winners vs baseline") inherits the field's ceiling: pre-execution judgement is near-chance for humans (56.1%) and LLMs (53.3%). We have something the field does not — a $5-7 execution path ending in an externally calibrated LSAR gate. That converts idea evaluation from a preference problem into a measurement problem.

### V1 — Deterministic screen audit (free, no LLM, run first, blocking)

`scripts/audit_feasibility.py` replays `feasibility.screen()` over:
- all 26 canonical archived specs (`runs/*/output/research_spec.json`), and
- a mutant set of ≥40 deliberately broken specs, one per KILL/WARN code, each derived from a real archived spec by a single documented mutation.

Reported: kill rate on mutants (target 100% — these are logically dispositive by construction), **false-kill rate on the 26 real specs (target 0%)**, and how many archived runs carried a defect the screen would now catch pre-spend. This is a measurement, not a statistic, and it needs no n.

**Blocking**: any false kill on a real archived spec is a bug, fixed before T1a ships.

### V2 — Rank-inversion backtest (free-ish, offline, blocking before any live selection)

Compile the 26 archived specs into idea cards deterministically (the spec fields *are* the card fields), restrict retrieval to papers published before each run's date, run the full cascade, and correlate tournament rank against realized gate score.

- Population: the **12 ledger papers** with median-of-3 gate scores (the other 12 have single reviews and are not on a common footing — plus a documented provider switch at 3b.23.7 confounds cross-era comparison).
- **The question is not "is rho high". It is "is rho negative."** That is the actual risk, per the -0.29 result.
- **Falsification: rho ≤ 0 → the ranker is inverted and must not ship.** Ship the deterministic screen alone (which cannot invert) and stop.
- **Pre-registered single-pair prediction**: the deterministic `venue_fit` term must rank `phase_b_did_20260704` (3.7 Reject, bare 2×2 DiD) **below** `stream1_did_v2` (7.0 Accept, same data, same estimand, wrapped in M9/M10). Same data, same estimand, difference is purely the idea. If VF-01/VF-04 cannot separate that pair, the rule table is not encoding what we think it is.

### V3 — Live paired A/B (the honest experiment)

For each of 3 tournaments, execute **two** full runs:
- **Arm 1** = tournament rank 1
- **Arm 2** = tournament rank ≈ median (both feasibility-CLEAN and prior-art-CLEAR, so the comparison isolates *ranking*, not screening)

Both scored with the existing median-of-3 gate (`review_gate.median_samples: 3`). 6 runs, ~$40, ~6 hours.

**Falsification: median(rank-1 gate) ≤ median(rank-median gate) across ≥3 pairs → the tournament does not select better ideas.**

**Honest power statement**: with 3 pairs and single-review MAD 1.9 (median-of-3 reduces this but does not eliminate it), we can only detect differences of roughly ≥1.5 gate points. This is a directional check, not a hypothesis test. The primary shipping evidence is V1 + V2; V3 accumulates in the ledger across arcs and only becomes decisive at n≈8-10 pairs.

### V4 — Three control conditions (all required)

| Control | What it isolates | Ships if |
|---|---|---|
| **Shuffle control** — re-rank with all judge verdicts randomly permuted, deterministic terms intact | separates "the tournament works" from "the judge works" | if shuffled ranking performs as well as real ranking on V2, **delete the judged layer** and ship the deterministic screen alone. That is a legitimate and cheap outcome. |
| **Status-quo baseline** — one single-draw PF run at temperature 0.7 with the same user prompt, i.e. exactly today's behaviour, executed and gated | the real baseline. Not "a random candidate from our own generator", which already benefits from slate diversity. | rank-1 must beat this, or Arc T has added cost for nothing |
| **Metric-vs-artifact audit** — mandatory section in `tournament.md` | the Arc P lesson, institutionalized | every score component prints its `evidence` string; the **diversity ledger** prints distinct outcome families / datasets / task types / opportunity patterns among the top 5. If the top-5 concentrates to a single outcome family for 2 consecutive tournaments, that is a red flag **regardless of score**, and it is written to the report as a failure line, not buried. |

### V5 — What would falsify "Arc T improves research taste", stated plainly

1. V2 backtest rho ≤ 0 on the 12 ledger papers → ranker inverted, do not ship it.
2. V3 shows rank-1 ≤ rank-median across ≥3 pairs → ranking is noise.
3. Shuffle control matches real ranking → the judged layer contributes nothing; delete it.
4. User's accumulated rankings agree with tournament ordering at ≤ chance over ≥20 pairs → either the tournament or the taste model is not capturing what the user means; investigate before applying bias offsets.
5. Diversity ledger collapses to one family for 2 consecutive tournaments → the tournament is re-deriving the 80.8% math-self-efficacy concentration it exists to prevent.
6. False-kill rate > 0% on real archived specs → the screen is destroying good ideas; V1 blocks release.

Any of 1, 2, or 3 → keep the deterministic screen (independently valuable, cannot invert, plugs into `--dry-run` and Arc I) and discard the ranker. **That fallback is a design feature, not a failure mode.**

---

## 7. COST

### Per tournament (24 candidates → ~10 survivors)

| Stage | Calls | Model | ≈ Tokens | Wall clock |
|---|---|---|---|---|
| Generation (24 × 2-iteration retrieval loop) | 48 | pro | ~200k | 6-10 min sequential; ~3 min at 4-way concurrency |
| S2 + arXiv retrieval | — | none | 0 | ~2 min (0.5 s request delay) |
| Deterministic screen (Stage 0) | 0 | none | 0 | <5 s for all 24 |
| Data probes (Stage 1, ~12 survivors) | 0 | none | 0 | ~15 s cached (+9.3 s one-time cache build per dataset) |
| Prior-art veto (facet extract + rerank + verdict) | ~30 | flash | ~150k | ~3 min |
| Venue-fit rules | 0 | none | 0 | <1 s |
| Pairwise tournament (25 pairs × 2 orders × 3 samples) | 150 | flash | ~430k | 6-10 min at 4-way concurrency |
| AE absolute screen (top-2 × 3 samples) | 6 | pro | ~40k | ~1 min |
| BT fit + artifacts | 0 | none | 0 | <2 s |
| **Total** | **~234** | mixed | **~820k** | **~20-30 min unattended** |

**Dollar estimate: $1-2 per tournament** (flash-dominated; exact figure depends on current DeepSeek pricing, which I have not verified this session — treat the token counts as the load-bearing number). **The `~$0.2` in `docs/v5_roadmap.md` §Arc T is wrong and should be corrected to `$1-2`.** Comparable published systems report ~$0.50 per idea for idea+design alone.

Human: **~5 minutes**, at most once per tournament, non-blocking.

### Per full paper run

- Unchanged: $5-7, 30-60 min.
- Arc T adds $1-2 and ~25 min **once per research question**, then zero for every re-run of that locked spec (the fixture is reusable, exactly as `runs/fixtures/*.json` are today).
- Net: +20-30% on a first run; the break-even is preventing **one** $5-7 run on an infeasible or venue-mismatched question per five tournaments. `phase_b_did_20260704` (3.7 Reject, attributed to venue fit, ~$6) is a concrete instance the VF-01 rule would have flagged pre-spend.

### V3 validation cost (one-time)

6 paired runs ≈ $40, ~6 hours wall clock, plus 3 tournaments ≈ $5. Budget **$45 and one overnight**.

---

## 8. RISKS (ranked, each with its guard)

| # | Risk | Why it is real here | Guard |
|---|---|---|---|
| **R1** | **Ranker inverted** (novelty-mirage class) — we confidently select worse ideas | LLM novelty rho = -0.291 with impact; our own self-novelty r = -0.35 with LSAR Novelty; the current selector optimizes precisely the anti-correlated signal | C1 (no positive novelty anywhere) + V2 backtest **blocking** with a pre-registered rho ≤ 0 falsification + the shuffle control, which tells us whether the judged layer contributes at all + the deterministic terms alone must produce a defensible ordering (C3) |
| **R2** | **Diversity collapse** — the tournament re-derives the concentration it exists to break | measured: 53.8% one outcome, 80.8% one construct, 100% mathematics, 0.837 mean pairwise predictor Jaccard in the only free-generation regime | structural slate enumeration before any LLM call + per-cell quota (≤3 candidates/cell) + hard `bridge_synthesis` cap at 3 of 24 + independent sampling with no cross-branch communication + tie-break rule 5 (consecutive tournaments cannot repeat a rank-1 (dataset, outcome-family) pair) + the diversity ledger printed as a pass/fail line in `tournament.md` |
| **R3** | **Feasibility false-negative** — a deterministic KILL destroys a good idea, invisibly and forever | this is the failure mode a deterministic screen uniquely enables, and it leaves no trace by construction | KILL reserved for logically dispositive facts only (column absent from the CSV; estimator uncertified; task type unexecutable on that dataset; <3 items for a CFA factor) — everything probabilistic is WARN + penalty and enters the tournament; `killed.jsonl` persists every kill with its code and evidence; `--no-kill` audit mode reruns the cascade with kills demoted; every KILL code requires a unit test with a real archived example; **one killed card per taste session is shown to the user for a y/n false-negative probe**; V1 false-kill rate on the 26 real specs must be 0% |
| **R4** | **Human anchor overfits to one person on a handful of judgements** | n=5 per session; expert-expert agreement on ideas is 56.1%; a single ranking is a noisy draw | judge-bias offsets **pinned to 0.0 until ≥20 pairs across ≥4 sessions**; anchor card every session detects human drift; tags stored separately from ordering so a categorical signal survives when the ordering is noise; exemplars capped at 6/polarity and FIFO'd; preferences enter as weighted BT observations, never as hard constraints; **LSAR is never trained on user preference** (C7); user tags that recur 3× are promoted to a proposed *deterministic* rule with a visible diff, not silently absorbed |
| **R5** | **Judge gaming** — our generator drifts toward judge-pleasing style and the metric goes green while the ideas get worse | exactly the Arc P failure shape; verbosity attacks succeed ~91% on weaker judges; self-enhancement +10-25% | fixed card template with a hard 120-word cap, truncated **before** judging so length carries zero signal; judge model ≠ generator model; absolute rubric only at the Stage-4 gate (9% vs ~35% distractor flip rate); C2 evidence strings make every verdict auditable; the V2 backtest is **re-run after any generator prompt change** |
| **R6** | **Prior-art false-negative** — we clear a collision because S2 has thin EDM coverage | S2 coverage of JEDM/JLA is thinner than of arXiv cs.CL; the anchors reveal that EDM's actual conversation (29% LLM-as-instrument) is not what the venue criteria YAMLs describe | `UNVERIFIABLE` is a distinct third state that is **not** treated as `CLEAR`; the 34-anchor local corpus seeds retrieval; the `already-done` human tag feeds back as a retrieval query seed; the delta sentence must quote a snippet, so a hollow clear is visible |
| **R7** | **Seam failure** — the winner spec is not consumable, or PF silently redesigns it | parallel work fails at the seams; PF's own `_run_multi_branch` fallback at line 316-324 already **drops `locked_research_spec`** | blocking CI test `test_winner_spec_loads_unchanged` for all 5 task types through the exact `load_locked_research_spec()` the CLI uses; `main.py:48` TypeError fixed; idea card rendered under an explicit "refine wording, do not redesign" header; `idea_provenance.json` written into the run dir so the executed spec can be diffed against the winner post-hoc; the `_run_single` fallback bug fixed to preserve the locked spec |
| **R8** | **`findings_memory` pollution poisons anything that reads it** | 63 of 63 entries are pytest fixtures; the outcome (`X3TGPAMAT`) has never been used in production; the suite rewrites the file on every run | Arc T **does not read `memory.yaml`**; `studied_outcomes` for the tournament is derived from `evaluation/ledger.json` + `runs/*/output/research_spec.json`; `tests/conftest.py` redirects the memory path to tmp; purge-and-rebuild is a separate backlog item, not an Arc T dependency |
| **R9** | **Cost creep** — 234 calls per tournament is 10× the roadmap's assumption | roadmap says $0.2 | flash for all 150 tournament matches and all prior-art work; pro only for 48 generation + 6 AE calls; concurrency capped at 4; per-tournament token budget logged in `tournament.md` and compared against the config ceiling, with a warning (not an abort) on exceed — same discipline as `pipeline.cost_budget_usd` |

---

## 9. STAGING

### T0 — Deterministic screen + upstream defect fixes (~0.5 day, zero LLM, independently valuable)

Ships: `src/ideation/feasibility.py`, `src/ideation/probe_cache.py`, `data_registry/venue_fit_rules.yaml`, `src/ideation/venue_fit.py`, plus the six upstream fixes (`did_feasible`, `itr_feasible`, `classify_intent` word boundaries, `main.py:48`, registry-resolved temporal wave, wire `RegistryLoader.is_excluded`), plus `DATASET_TASK_MATRIX`, plus `scripts/audit_feasibility.py`.

CLI: `python -m src.ideation.feasibility --spec runs/fixtures/spec_x1mtheff_x4college.json`
Also wired into `src/main.py --dry-run`, so it is exercised before every live run from day one.

**Gate: V1 must pass (100% mutant kill, 0% false kill on the 26 archived specs) before T1a starts.**

This slice is useful even if the rest of Arc T is abandoned: it turns 11 registry-documented pitfalls from LLM prose into code, moves the causal positivity probe from post-DataEngineer (~$2-3 spent) to pre-run (0.08 s), and gives Arc I's I1 benchmark battery its validity check.

### T1a — Generator + slate + cards (~1 day)

Ships: `slate.py`, `generate.py`, `cards.py`, `scripts/run_idea_tournament.py --stage generate`. Output is `candidates.jsonl` + `killed.jsonl` + a ranking by deterministic terms only (feasibility penalty + venue fit). No judging.

Already strictly better than today: 24 structurally diverse candidates with persisted rejects, versus 1 draw with zero record of any alternative.

### T1b — Prior-art veto + pairwise tournament + BT (~1 day)

Ships: `priorart.py`, `judge.py`, `bradley_terry.py`, `tournament.py`, full cascade.

**Gate: V2 backtest must return rho > 0 on the 12 ledger papers, and must correctly order the `phase_b_did` v1/v2 pair, before the tournament is permitted to select a spec for a live run.** Until then it runs in advisory mode and prints both its ranking and the deterministic-only ranking side by side.

### T1c — PF integration + selector retirement (~0.5 day)

Ships: orchestrator/context/PF edits from §1.4; `_select_best_candidate` delegates to `src.ideation`; `n_candidate_specs` decoupled from `findings_memory.enabled`. `_run_multi_branch` is **kept** — but repurposed to what it is actually good at: generating wording variants of a *locked* spec in refine mode, where a shared prior is correct.

### T2 — Taste memory (~0.5 day)

Ships: `taste.py`, `scripts/taste_session.py open|ingest`, `taste/preferences.jsonl`, exemplar management, the killed-card probe. Bias offsets computed and logged but pinned to 0.0 (n < 20).

### T3 — Live validation (overnight + ~$45)

3 tournaments × (rank-1, rank-median) paired runs + 1 status-quo baseline run. Ledger entries with `tournament_id` / `candidate_rank` / `arm`. Arc report with the V1/V2/V3/V4 results and an explicit statement of which falsification conditions were tested and which were not.

**Arc T exit**: V1 clean, V2 rho > 0, V3 directionally positive across 3 pairs, shuffle control shows the judged layer adds something, diversity ledger not collapsed, and `winner_spec.json` consumed by the unchanged pipeline in ≥3 live runs. If V2 or the shuffle control fails, exit with T0 shipped and the ranker explicitly shelved in `docs/backlog.md` alongside RD and IV — a documented shelf, not a quiet deletion.

---

## Appendix A — `IdeaCard` schema

```jsonc
{
  "candidate_id": "C-07",
  "tournament_id": "T-0007",
  "cell": {"dataset": "els_2002", "task_type": "prediction",
           "opportunity_pattern": "replication_transfer",
           "persona": "replication_methodologist", "gap_cell": ["college_enrollment", "fairness"]},
  "research_question": "...",                    // ≤ 30 words
  "why_it_matters": "...",                       // ≤ 35 words
  "what_we_would_do": "...",                     // ≤ 35 words
  "what_counts_as_the_result": "...",            // ≤ 20 words  (the evaluation facet)
  "resolved_target": "F2EVRATT",                 // task-type-agnostic
  "method_family": "prediction_ml",
  "second_contribution": "transfer",             // transfer|fairness|measurement|replication|null
  "spec_draft": { /* task-type-shaped block, compiled by compile_spec() */ },
  "novelty_score_self_assessment": 4,            // RECORDED, NEVER READ (see test_ranking_invariant_to_self_novelty)
  "generated_at": "2026-07-25T20:14:03Z",
  "generator_model": "deepseek-v4-pro",
  "render_word_count": 118                       // hard cap 120, truncated before judging
}
```

## Appendix B — Match record schema (`matches.jsonl`)

```jsonc
{"pair": ["C-07", "C-11"], "orientation": "AB", "sample": 2,
 "dimension": "significance", "winner": "C-11",
 "evidence": "C-11 names a specific decision that changes (course placement); C-07 states 'improves understanding'.",
 "judge_model": "deepseek-v4-flash", "ts": "..."}
```

## Appendix C — Test names (all blocking in CI)

```
tests/test_ideation_feasibility.py::test_kill_code_<code>_fires_on_real_mutant   # one per KILL code
tests/test_ideation_feasibility.py::test_no_false_kills_on_archived_specs        # V1
tests/test_ideation_ranking.py::test_ranking_invariant_to_self_novelty           # C1
tests/test_ideation_ranking.py::test_ranking_deterministic_given_seed
tests/test_ideation_ranking.py::test_bt_fit_matches_known_round_robin
tests/test_ideation_venue_fit.py::test_did_v1_ranks_below_did_v2                 # pre-registered V2 pair
tests/test_ideation_seam.py::test_winner_spec_loads_unchanged                    # all 5 task types
tests/test_ideation_seam.py::test_idea_provenance_written_to_run_dir
tests/test_ideation_diversity.py::test_bridge_framing_quota_enforced
tests/test_ideation_diversity.py::test_consecutive_winners_differ_in_outcome_family
tests/test_taste.py::test_bias_offset_pinned_below_min_pairs
tests/test_taste.py::test_exemplars_never_reach_judge_or_review_gate             # C7 rail
```