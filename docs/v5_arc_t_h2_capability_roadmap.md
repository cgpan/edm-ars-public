# Arc T / H2 — Capability Roadmap

**Status:** synthesis of three independent read-only investigations of the venue-fit rule table (v2), the 34-anchor LSAR corpus, and the EDM-ARS repository at `<repo root>`. Every number below is either reproduced measurement or explicitly marked NOT MEASURED. Four repository facts were re-verified while writing this document; they are flagged **(verified here)**.

**The finding this document responds to:** a blind-derived venue-fit rule table measured null as a ranker. Five positive rules fire on 0 of 34 of our research specs while firing on 21–44% of published anchors; three negative rules fire on 68–82% of our specs. That is not a ranking bug. It is a statement about what the pipeline can and cannot do, and about which room it is standing in.

**The one-line answer:** part of the gap is a capability deficit that costs real money and real hours (VF2-04/05/08 — learner text we do not have). Part is a framing defect that costs about thirty lines of skill text (VF2-03, VF2-07 tier 1). Part is *already done and simply never claimed* (VF2-06 — two replications sit finished on disk). And part is not a capability question at all but a venue question the owner has to decide (VF2-01, VF2-02).

---

## 1. THE GAP, IN ONE TABLE

Cost classes: **FREE** = edit rules/skills already in the repo, no new data, code, or human hours. **WRITING** = skill-text change that alters generated prose. **CODE** = new or changed `src/`. **DATA** = new acquisition. **HUMAN** = owner hours beyond a sign-off.

| # | Shape | Anchor frequency | Our frequency | What it would take | Cost class |
|---|---|---|---|---|---|
| **VF2-01** (−) | Observational causal estimate as the headline | **0/34** abstract, **0/34** full text. Survives a deliberately broad 37-pattern probe: 19/34 raw hits, **all spurious** (PDF de-hyphenation, non-causal "matching", or explicit disavowals). Causation appears as *design* (RCT/quasi-exp, 14/34), never as *adjustment*. | ~68–82% of specs (taken from the brief, **not re-verified by anyone**) | Reclassify from flat −1.5 penalty to a **venue-routing signal**. Then, if you want the alternative venue calibrated, acquire an AERA Open anchor corpus. | FREE (reclassify) → DATA (calibrate) |
| **VF2-02** (−) | National-survey secondary extract | **2/34** lexical; **1/34** genuine data source (the other is a NAEP motivational citation in `drl_pedagogical_policy`, which uses no NCES data). 0 of the 15 VF2-04 anchors use one. | ~100% — every registered dataset we hold is a national survey | Same routing change. *Plus* the reframing lesson from the one genuine anchor: `fairness_dropout_prediction` (EDM 2024) publishes **HSLS:09 + ML battery + subgroup fairness** at the target venue, because its contribution is a method (differential algorithmic functioning) and a documentation practice — never a predictor ranking. | FREE + WRITING → DATA |
| **VF2-03** (−) | Feature-importance ranking as the takeaway | **3/34** full text, **0/34** abstract; only **2/34** actually compute a ranking (`jla_8905`'s mentions are future work). Counter-corpus: **0/1101** policy/psych abstracts, **0/30** AERA Open full texts after correcting two `shap- ing` artifacts. | High | Move the rule from the idea tournament to the **Writer / OutlineAgent skills + manuscript linter**. The defect is introduced downstream, not by the ProblemFormulator — computing SHAP is harmless; promoting it to the abstract is the whole problem. | **FREE / WRITING** |
| **VF2-04** (+) | LLM/NLP measures a construct from learner text | **15/34** abstract (JEDM 8/10). Learner-authored text or speech present in **23/34** full texts. Provenance: 14/15 authors' own or institutional text; 1 fully open; **0** from a national survey. | **0** — we hold zero characters of learner text | New corpus. HSLS max cell = **142 chars** across 2,000 rows × 9,614 cols, **0** columns over 150 chars; ELS stores numeric codes; ASSISTments `answer_text` is a tier-3-excluded short-answer field. No method work fixes this. | **DATA** + CODE |
| **VF2-05** (+) | Agreement with human coders | **9/34** abstract | 0 | Acquire a corpus that **ships human labels** (ASAP-AES: 2 raters/essay; PERSUADE: discourse annotations) → costs zero new human hours and includes a real human-human benchmark. | **DATA** (+ optional HUMAN 3–4h) |
| **VF2-06** (+) | Cross-context test / replication | **7/34** lexical, but only **3/34** do a genuine second-context test (2 partial, 2 pure noise — one is an OSF link, one an abstract-level limitation). Below the table's own `count_min: 6` floor. | **0/14** abstracts. 11/14 full-text hits are the bare token "generaliz" inside limitations prose — the identical false-positive mode. | Amend the novelty auto-reject (see §2), then claim two replications **already sitting finished on disk**. | **FREE** → CODE (small) |
| **VF2-07** (+) | Names the practice or decision that changes | **12/34** lexical; **7/34** ship a real artifact (12 heuristics, 3 design principles, a metric framework, a screening tool, a released model); 4 are Discussion gestures; 1 is noise. | Full text **13/14** vs anchors 33/34 — statistically indistinguishable. Abstract-level "names a specific decision or artifact": **ours 1/13 vs anchors 7/34**. | Tier 1: add an abstract-content rule (the file currently has **none** — only `\| Abstract \| 200–300 \|`, **verified here** at line 51). Tier 2: actually ship an artifact. | **WRITING** (tier 1) → CODE + HUMAN (tier 2) |
| **VF2-08** (+) | Human-in-the-loop workflow | **10/34** abstract. In all 10, the human is a person doing interpretive work. | 0, and structurally impossible: the string "human" appears **zero** times in `src/orchestrator.py`, `src/review_gate.py`, and `config.yaml`. | ~2–5 owner hours per paper, logged, plus a checkpoint hook and a MANDATORY honesty skill. | **HUMAN** + CODE |

**Overlap that changes the arithmetic:** VF2-04/05/08 are not three capabilities. They share 7, 9, and 5 anchors pairwise/triple-wise; their union is **18 of 34 anchors (53%)**. One capability — *an LLM measures a construct from learner text, checked against human codes, inside a workflow with a real human step* — fires all three at once, worth up to +3.0 at JEDM. Costing them as three arcs triple-counts the work.

---

## 2. WHAT IS ACHIEVABLE NOW

No new data, no new estimator, no new dataset onboarding.

### 2.1 Two replications have ALREADY BEEN RUN and were never framed as one

This is the loudest finding in the whole exercise and the cheapest available win.

**(A) Every prediction paper already reports an out-of-school generalization estimate.**

`runs/v3_8_prediction_pilot_20260703_e1/output/data_report.json` records `school_reconstruction {status: success, n_clusters: 931, expected_n_schools: 944, mean 18.6 students/cluster}` and the warning *"train/test split is school-aware (no school in both sets)"*. The headline **AUC = 0.749 [0.730, 0.768]** is therefore already computed over **3,562 students in schools never seen during training** — the exact estimand `jla_9099` built an entire JLA paper around ("we trained on Course 1 and evaluated on an entirely new cohort in Course 2").

What is missing is not the test. It is the **contrast**. `jla_9099` reports within-institution AUC *beside* cross-institution AUC; we report one number with nothing to compare it to.

The machinery to fix this is already written (**verified here**):
- `src/analysis_helpers.py:806` — `grouped_train_test_split(...)`, `StratifiedGroupKFold`/`GroupShuffleSplit`, no school in both sets.
- `src/analysis_helpers.py:1424` — `bootstrap_auc_difference(y_true, prob_a, prob_b, school_ids=None, ...)`, a **cluster-aware paired bootstrap on the AUC difference**, returning point estimate, CI, and a significance flag.

So the study is: fit the same battery under a random split and under the grouped split, report both AUCs and the difference CI. **One assessment costed this as "add one helper"; it is cheaper than that — the difference-CI helper already exists.** What changes: a thin wrapper in `src/analysis_helpers.py`, one field in `results.json`, one row in `model_comparison.csv`, one sentence in `skills/writing/paper-section-content-prediction/SKILL.md`. HSLS only until §2.4 lands.

**(B) A cross-cohort conceptual replication is complete across two Accepted runs.**

- `runs/v4_psy_paper1_20260708` — HSLS:09, sex-DIF on math self-efficacy items. Gate **7.5 Accept**. Headline: *"No DIF flagged (p<.01 & McFadden ΔR²≥.02)"*, α=.837, ω=.922, CFI=.991, RMSEA=.048.
- `runs/v4_psy_paper2_20260708` — ELS:2002, same question of the BYS89A-U battery. Gate **7.3 Accept**. Headline: *"No DIF flagged (all items: p_overall≥.01 or McFadden ΔR²<.02)"*.

Same construct, same decision rule, two national cohorts seven years apart, same null. That is a measurement-fairness replication, and both halves are already gate-passed. Writing it as one paper costs **human authorship hours only** (or one pipeline run once §2.4 exists).

Mandatory caveats for any such paper: the item batteries differ (5-item BYS89 vs the HSLS items); the ELS run **did not test invariance** (`PSY-03: Measurement invariance (P6) not tested`); ELS CFA fit was mixed (RMSEA above .06). **The DIF null replicates; the invariance level does not transfer.** Say both.

### 2.2 The prompt rule that blocks all of this — ~15 lines, no code

The system is explicitly configured to refuse the VF2-06 shape (**verified here**):

- `skills/task-type/prediction-research-question-design/SKILL.md:96` — `| 1–2 | Replicates a known finding | (Auto-rejected — regenerate.) |`
- Same file, line 107 — *"Research questions MUST build on prior findings rather than replicate"*
- `skills/methodology/findings-memory-novelty-cross-run/SKILL.md:33` — *"Do not replicate prior research questions verbatim"*
- `SPEC.md` — `novelty_score_self_assessment` must be ≥ 3, otherwise regenerate

Every research_spec inspected self-scored novelty 4. The rule is right in spirit — it exists to stop the system re-running its own study — but it does not distinguish two very different things:

| Shape | Verdict |
|---|---|
| Replicating **our own** prior run | Still rejected |
| Testing whether a **published** finding holds in a second context | **Now permitted**, and scored as novel on the transport claim |

**Nothing else in VF2-06 works until this lands.** It is the single highest-leverage edit in this document and it touches two files, no Python.

### 2.3 VF2-03 and VF2-07 tier 1 — one file each

**VF2-03 is priced at the wrong pipeline stage.** It penalizes an *idea* for a defect the Writer introduces. Move it to the Writer/OutlineAgent skills and the manuscript linter as a hard rule: *never let an importance ranking be the headline claim; convert it into a substantive claim or a named decision.* The reframing template comes from the two anchors that actually compute a ranking:

- `chatgpt_cs1_codegen` computes a dual (impurity + 1000-iteration permutation) importance table over 14 metrics, then never names the technique in the abstract: *"models effectively distinguish ChatGPT from human code with up to 88% accuracy, indicating detectable coding style disparities."* → **substantive claim**.
- `gre_test_optional` — the anchor closest to our own shape (supervised prediction on application records) — converts its ranking into *"a focus of attention (FOA) tool for admissions committees"* and *"the factors that one might focus on when GRE scores are unavailable."* → **named decision**. Note it earns its EDM slot on the decision framing, not on the method.

**VF2-07 tier 1** is ~10–15 lines in `skills/writing/paper-section-content-prediction/SKILL.md` (rule_severity: mandatory), which today contains **no abstract-content rule at all** — only a word budget (**verified here**). Require the final abstract sentence to name *an actor, a decision, and what changes*. The evidence this works: our full-text base rate (13/14) already matches the anchors (33/34), and the material is already sitting in our Discussions. `v3_8` already contains *"A school district deploying this model to identify 9th-graders at risk of not enrolling in college would find that the model is substantially less accurate for the very students who might benefit most from early intervention"* and *"fairness audits should be standard practice in educational prediction pipelines."* Promotion from Discussion to abstract is free.

Our own reviewer has already flagged both gaps, unprompted. From `runs/stream2_els_rigor_20260704_attempt2/output/lsar_review/cycle_1/lsar_report.json`: *"The reliance on a single, older dataset (2002 cohort)... may reduce immediate practical impact"* (VF2-06, scored 6 on Significance & Impact). From `v3_8`: *"the absence of feature importance means practitioners cannot identify which modifiable factors to prioritize"* (VF2-07 — the complaint is that we do not say which lever to pull).

### 2.4 Free hygiene that blocks or corrupts everything downstream

1. **Nothing is currently scored against VF2-06 or VF2-07.** `src/ideation/venue_fit.py:25` sets `DEFAULT_RULES_PATH = data_registry/venue_fit_rules.yaml` — the **v1** table (**verified here**). The v2 table is loaded by nothing, and its own integration notes say it is not drop-in loadable because v1 keys rules to named Python predicates while v2 carries declarative clauses. Any plan that measures itself against VF2-06/07 pays a ~40-line clause-evaluator integration first (reference implementation exists at `scripts/derive_venue_rules.py::evaluate_predicate`).
2. **Add a de-hyphenation pass before any future rule derivation.** Verified artifacts in this corpus: `\bate\b` matches "gradu- ate"; `\bcate\b` matches "repli- cate"; `\bels\b` matches "mod- els"; `\bshap\b` matches "shap- ing". Three broad probes and two AERA Open hits were pure artifacts. Short-token regex over `paper.md` is currently unsafe.
3. **The ELS clustering defect is a validity bug, not a venue item.** `runs/stream2_els_rigor_20260704_attempt2` reports **3,715 clusters against 752 expected, cluster_size_median 1** — the fingerprint reconstruction failed, so its "school-aware" split is near-random and its multilevel claim is close to vacuous. Meanwhile `data_registry/datasets/els_2002.yaml:25` declares `cluster_id_variable: F1SCH_ID` with *"752 real school IDs covering ~76% of students"* (**verified here**), which `reconstruct_school_ids()` (`src/analysis_helpers.py:671`) never consults. Fix this before any further ELS multilevel claim ships.

### 2.5 Adjudication: the datasets are not lost

One assessment reported that `data/raw/` holds a single file and that ASSISTments, ELS, ECLS-K and PISA "are not present on this mount... whether the files were moved, lost, or live on another machine was NOT determined." A second assessment located them. **Verified here:**

```
<repo root>/data/raw/                     → hsls_17_student_pets_sr_v1_0.csv only
.../.claude/worktrees/distracted-satoshi-9078fa/data/raw/
    assistments_0910/  did_els_hsls_panel/  ecls_k_2011/
    els_2002/  pisa_2022/  hsls_17_student_pets_sr_v1_0.csv
```

Nothing is lost. But this **is** a real operational hazard, not a non-issue: a run launched from the main checkout can only use HSLS, which silently blocks the DiD, psychometrics-on-ELS and CDM task types. Worth a one-line pre-flight check that resolves the raw-data path and fails loudly.

*(Related doc drift, verified here: `CLAUDE.md` says 53 skills; the tree holds **70** `SKILL.md` files — dataset 10, methodology 31, task-type 15, writing 14. `README.md` still says v1.2.0.)*

---

## 3. WHAT NEEDS NEW DATA

**This project uses public-use files only.** Everything below respects that constraint; where access is gated, it is named as a gate.

### 3.1 Learner text is genuinely absent from everything we hold. Say it plainly.

| Asset | Learner text? | Evidence |
|---|---|---|
| HSLS:09 | **No** | 2,000 rows × 9,614 cols scanned. Longest cell in the entire scan: **142 characters**. Columns with any value over 150 chars: **0**. Every long value is a closed-response category label. The only open-ended-looking column names are `S1MTEXTBOOK`/`S2STEXTBOOK` etc. — textbook-*use* items. NCES public-use files strip verbatim fields. |
| ELS:2002 | **No** | Registry states the CSV stores numeric codes only. |
| ASSISTments 2009-10 | **Almost certainly not, in the sense VF2-04 requires** | `answer_text` exists (11.74% missing, profiled from a 100k-row sample) but is listed in the curated registry's `tier3_exclusion_rules.exact_matches` — the onboarding session saw it and excluded it as administrative. Structurally it is a submitted answer to a math item (the file also carries `answer_type`), i.e. numbers and expressions, not prose. **Content not inspected — treat as UNVERIFIED.** It also shares no constructs with the NCES panels, so it is not a viable cross-context partner for any of them. |
| ECLS-K 2011, PISA 2022 | No, and not onboarded | Acquired (483 MB, 682 MB) but no registry entry, no adapter. Neither carries learner prose either. |
| The 34 anchor papers | Researcher text, not learner text | A meta-science study over them would technically fire the predicate, but the one meta-science anchor (`open_science_edm`) is a position paper with no empirical text measurement. Do not build on this. |

**No amount of code fixes this.** VF2-04/05/08 are hard-blocked on an acquisition.

### 3.2 Candidate acquisitions for the text capability

| Dataset | Content | Human labels | Access | Licence |
|---|---|---|---|---|
| **PERSUADE 2.0** — *recommended first* | 25,000+ argumentative essays, US grades 6–12, 15 prompts, **writer demographics** | Argumentative/discourse element annotations | GitHub `scrosseye/persuade_corpus_2.0` + Kaggle; instant download | **CC BY-NC-SA 4.0 — verified.** NonCommercial + ShareAlike bind any derivative data release. |
| **ASAP-AES** (Hewlett, Kaggle 2012) | ~12,978 essays, 8 prompts | **Two human rater scores per essay** — the canonical agreement substrate | Kaggle, free signup | **NOT verified.** Check the competition rules page before acquisition. |
| CommonLit Evaluate Student Summaries (2023) | Grade 3–12 student summaries | Human content/wording scores | Kaggle | **NOT verified** |
| NCTE Transcripts (Demszky & Hill, 2023) | 1,660 grade 4/5 math classroom transcripts, 317 teachers | Turn-level discourse annotations + observation scores + test scores + demographics | **Request form** — a user gate, like the ELS EDAT export | **NOT verified** |

Why PERSUADE first: it is the only candidate that is simultaneously public, instantly downloadable, licence-verified, human-labelled, **and demographically tagged**. The demographics matter more than they look — they let the project's existing subgroup / fairness / calibration / psychometrics machinery attach to a *text-derived* measure, so the new capability multiplies the old ones instead of replacing them.

### 3.3 The AERA Open anchor corpus (for §5)

Fully open access, AERA's own journal, same scope family and methods as EEPA/AERJ, **371 articles indexed 2022–2025**, full-text PDFs free and unauthenticated at `files.eric.ed.gov/fulltext/{id}.pdf`. Thirty were retrieved in a single pass while producing these assessments — that retrieval *is* the cost proof. EEPA (SAGE), AERJ (SAGE), JREE (T&F), JEP (APA) are paywalled for full text; their abstracts are free via the ERIC API and already measured. Economics of Education Review is not indexed in ERIC and is entirely unmeasured.

### 3.4 Not new data — new code over data we already hold

The **stacked two-cohort file**. `scripts/harmonize_els_hsls.py` already maps six of the seven constructs shared by the HSLS and ELS prediction runs (race5, pared3, expect_ba, ses_std, female, math rank). Two blockers: it filters to SES-band extremes only (`BYSES1QU ∈ {1,4}`, `X1SESQ5` quintiles 1 and 5), and it emits math percentile rank rather than an attendance outcome. Extend it to full sample, add self-efficacy (`BYMATHSE ↔ X1MTHEFF`) and a harmonized outcome (`F2EVRATT ↔ X4EVRATNDCLG`, with the differing follow-up horizons recorded as a limitation); add one registry entry and one `src/dataset_adapter.py` branch.

**Do this the way `did_els_hsls_panel` was done** — one stacked file with a cohort column, which the pipeline already runs end-to-end and which produced two gate-passing DiD papers. **Do not plumb a second dataset through `main.py`.** `src/main.py` takes one `--dataset`, resolves one raw path, and builds a `PipelineContext` with a single `dataset_name`; there is no second-dataset slot anywhere in the chain, and adding one is a substantially larger change that buys nothing the stacked-file precedent does not already buy.

---

## 4. WHAT NEEDS A HUMAN, AND HOW MUCH

The standing budget is occasional ~5-minute sessions. Here is what fits and what does not.

### 4.1 Fits in ~5 minutes

| Item | Why it fits |
|---|---|
| Approve the novelty-rule amendment wording (§2.2) | A yes/no on ~15 lines of skill text |
| Approve the abstract-decision sentence pattern per task type | One-time per task type |
| Gate-threshold sign-off (memory records gate swaps require user approval) | A decision, not labour |
| Licence acceptance for a downloaded corpus | Reading four lines of terms |
| Submitting the NCTE access request form | ~5 min to submit; wait time unknown |
| The venue call in §5 | 5 minutes to *decide*; it may deserve more to *think about* |

### 4.2 Does not fit — VF2-08 has a hard floor of ~2–5 hours per paper

The corpus was read for human effort, and one anchor publishes a full ledger. `jedm_981`: nine stages, ~744 hours total — codebook 2h, educator sense-check 2h, complete coding 150h, build algorithm 300h, test algorithm 250h, output quality check 3h, finalise 25h. **4,631 items in 150 hours = 30.9 items/hour = 1.94 minutes per item.** Only ~7 of those 744 hours are human *judgment*; the bulk is bulk coding.

At that rate:

| Target | Time | In 5-minute sessions |
|---|---|---|
| One item | 1.94 min | 0.4 |
| Smallest IRR round in the corpus (`jedm_981`, 49 items) | ~95 min | **~19** |
| `jedm_974`'s double-coded agreement sample (134 items) | ~4.3 h | **~52** |
| `jedm_981`'s full coding pass | 150 h | absurd |

And those sessions must land on a **fixed item set with a codebook written in advance** — the codebook alone cost 2h + 2h of sense-check — so they cannot be improvised five minutes at a time. **No anchor in the corpus rests a human-in-the-loop claim on less than ~2 hours of human work.** The cheapest human arm in the whole corpus is `jla_9141` at ~6 person-hours (4 experts × 45 min × 2 rounds).

### 4.3 The recommended human spend: ~3–4 hours, and 0 hours if you buy labels

**Buy the labels, don't make them.** A corpus that ships human codes gives you "agreement with human coders" for **zero new human hours**, *with a genuine human-human benchmark included*. `jedm_974` did exactly this for its external validation (κ = 0.86 against another team's coders, alongside its own human-human 0.83 / machine-human 0.85).

If the owner wants to be genuinely in the loop, spend the hours only where they change the paper:
1. **Codebook approval before coding starts** — ~2h.
2. **Adjudication of a pre-specified 30–50 item disagreement sample** — ~1–1.7h.

That is ~3–4 hours, matching the corpus's own smallest judgment stages (`jedm_981` Stage 3 at 2h, Stage 7 at 3h).

### 4.4 Two hard constraints the owner must accept

**With exactly one human you get machine-vs-human κ but never the human-human benchmark** that `jedm_974`'s headline claim rests on. Substitutes: intra-rater test-retest (code ~50 items twice, ≥2 weeks apart, 2 × 95 min) gives a within-human ceiling but not a between-human one.

**LSAR is not a substitute for the human.** It is a language model; using it as the second coder makes the statistic model-vs-model. And claiming VF2-08 with an LLM standing in for the human is **a false methods statement, not a framing choice** — in all 10 VF2-08 anchors the human is a person, and `jedm_981` makes the model's *lack* of theoretical commitment and contextual judgment the substance of its contribution. Since two of the three rules' predicates match on phrases like "human-in-the-loop" and "human-AI collaboration", the ranker would happily reward exactly that lie. **Ship a MANDATORY honesty skill blocking any human-in-the-loop or human-agreement claim without logged human hours in the same change as the capability, not after.** There is nowhere to put such a step today: "human" appears zero times in the orchestrator, review gate, or config.

---

## 5. THE VENUE QUESTION

*Is the pipeline aimed at the wrong venues rather than doing the wrong work?*

**Partly — for two of the eight rules, decisively so. For the other six, no.** This is the section where the owner has to make a call the data does not make for him.

### 5.1 The evidence that VF2-01 and VF2-02 are venue facts, not field facts

A counter-corpus of **1,101 abstracts (2022–2025, ERIC API, exact source match)** was scored with the *shipped* detectors:

| Venue | VF2-01 abstract rate | Wider causal detector |
|---|---|---|
| EEPA | 31/157 = **19.7%** | 28.0% |
| JREE | 17/122 = **13.9%** | 30.3% |
| AERJ | 6/138 = 4.3% | 10.1% |
| AERA Open | 15/371 = 4.0% | 6.7% |
| JEP | 4/313 = 1.3% | 6.1% |
| **Our anchors** | **0/34 = 0.0%** | **0.0%** |

And on the one counter-venue where full text is freely available, a random n=30 (seed 42) sample scored with the same patterns:

| Probe | AERA Open full text | Anchors |
|---|---|---|
| VF2-01 | **9/30 = 30.0%** | 0/34 |
| VF2-02 | **15/30 = 50.0%** | 1/34 genuine |
| Both together | 4/30 | 0/34 |
| **HSLS:09 specifically** | **3/30 = 10%** | 1/34 |

**Three of thirty randomly drawn AERA Open papers use our exact dataset.** This is the single most important number in the exercise: the pipeline is not doing work nobody publishes. It is doing work published constantly, at venues absent from the anchor corpus.

Note also that the abstract-level counter-corpus **badly understates** full-text prevalence — on AERA Open the same detector gives 4.0% on abstracts and 30.0% on full texts, a factor of 7.5. The true EEPA and JREE full-text rates are therefore probably far above 19.7% and 13.9%, but that is NOT MEASURED.

### 5.2 The evidence that retargeting is not sufficient

- **VF2-03 is venue-independent.** 0/34 anchor abstracts, **0/1101** counter-corpus abstracts, **0/30** AERA Open full texts after artifact correction. There is no measured venue anywhere — computational EDM, education policy, or educational psychology — where a feature-importance ranking is a headline contribution. Retargeting cannot fix it; only reframing can.
- **VF2-06 and VF2-07 are craft, not venue.** A cross-context contrast and a named decision are rewarded everywhere.
- **VF2-04/05/08 do not become reachable by moving venues** — but the *pressure* to have them largely disappears, because policy venues run on survey data. That is a strategically significant side effect: retargeting turns a 53%-of-corpus capability gap into a much smaller one.
- **The negative rules are not unconditional even at EDM.** `fairness_dropout_prediction` (EDM 2024) is HSLS:09 + ML battery + protected attributes + subgroup fairness — our exact shape, dataset, outcome and framing — published at the target venue, because the contribution is a *method* plus a *practice recommendation*, never a predictor ranking. An existence proof inside VF2-02's own evidence base.

### 5.3 The three options and their consequences

| Option | What it means | Consequence |
|---|---|---|
| **A — Stay EDM/JEDM/JLA** | Accept the negative rules as binding | Must acquire learner text and build the measurement task type (§3.2, §6 step 8); must stop or heavily downweight causal work; VF2-01's 0/34 is real and the venues *affirmatively disclaim* causal interpretation in their own limitations sections |
| **B — Retarget causal + survey work to the policy/causal family** | AERA Open first (open access), then JREE/EEPA | VF2-01/02 flip from penalties to fit; the four existing causal/prediction task types become on-target overnight; requires a new anchor corpus, a calibration batch, and gate sign-off. Scaffolding partly exists: `$LSAR_HOME/venue_criteria/jree_2026.yaml` already declares family `causal-applied` with topics "causal effects of educational interventions/policies" and "design rigor: RCT, RD, DiD, matching" — but carries `calibrated: false`. **The gap is anchors, not criteria.** |
| **C — Dual-target with venue routing** *(recommended)* | Keep EDM/JEDM/JLA for measurement, psychometric and prediction-*method* work; route causal + survey work to the policy family | Same acquisition cost as B, plus a routing rule instead of a flat penalty. VF2-01 stops being a −1.5 on the *idea* and becomes a signal about *where the idea goes*. |

### 5.4 What the data does not settle

- **Nothing here measures acceptance.** It measures what is published. The 30% VF2-01 rate says the genre is welcome at AERA Open; it does not say our instance of it would pass review.
- **EDM 2026's own CFP lists "causal inference" as in scope**, so declared scope and published record disagree. Thirty-four anchors from one conference year cannot resolve that.
- **No magnitudes.** This work re-derives signs and directions, not calibrated penalties. One HSLS existence proof cannot calibrate a −1.0.

**The recommendation is Option C**, and the honest framing of why: the measurements support the reading that EDM-ARS is a causal-education-research system that has been scoring itself against a computational-EDM rubric. But that is a **goals** question, not a data question, and the owner owns it.

---

## 6. SEQUENCED PLAN

Ordering principle: **unblock before build; prefer items that are cheap and move multiple rules.**

**Step 0 — FREE, blocking everything measured.** Wire the v2 rule table (~40-line clause evaluator; reference impl at `scripts/derive_venue_rules.py::evaluate_predicate`) and add a de-hyphenation pass before any future derivation. *Right now nothing is scored against VF2-06 or VF2-07 at all* because `venue_fit.py:25` loads v1. Every downstream measurement is unverifiable until this lands.

**Step 1 — FREE, ~15 lines, two files.** Amend the novelty auto-reject to distinguish self-replication (still rejected) from second-context testing (permitted). Unblocks VF2-06 entirely. *Nothing in VF2-06 works until this lands.*

**Step 2 — WRITING, ~30 lines, two files.** Add the abstract-content rule to `skills/writing/paper-section-content-prediction/SKILL.md`, and relocate VF2-03 from the idea tournament to the Writer/OutlineAgent skills and the manuscript linter. **Two rules moved, no code, no data, no human.** Best ratio in the document.

**Step 3 — CODE (small).** Within-school vs across-school AUC contrast, reusing `grouped_train_test_split` (`:806`) and `bootstrap_auc_difference(..., school_ids=...)` (`:1424`). Fires VF2-06 for **every future prediction paper**, permanently. HSLS only until step 4.

**Step 4 — CODE (correctness, not venue).** Make `reconstruct_school_ids()` consult `F1SCH_ID` on ELS. 3,715 clusters against 752 expected with median size 1 means the ELS multilevel claim is close to vacuous. Do this before any further ELS multilevel claim ships, independent of everything else here.

**Step 5 — HUMAN authorship, 0 engineering hours.** Write the cross-cohort DIF replication from `v4_psy_paper1` (7.5) and `v4_psy_paper2` (7.3). The highest-value single paper currently available: genuine VF2-06, data we hold, both halves already gate-passed, honest caveats already identified.

**Step 6 — DATA (free) + CODE (~40–80 lines).** AERA Open anchor corpus via a new ERIC fetcher (`scripts/calibration_fetch_ojs.py` is OJS-only and will not work), then `venue_criteria/aera_open_2026.yaml`, regenerate `data_registry/venue_norms.yaml`, and run an LSAR calibration batch over ~16 anchors to produce a P25 gate — exactly the protocol that produced the JEDM 5.15 and JLA 5.4 gates. **This decides §5 with evidence instead of argument.** Ends at a user gate (threshold sign-off).

**Step 7 — CODE.** Extend `scripts/harmonize_els_hsls.py` into a full-sample stacked two-cohort file. Unlocks the DIF replication as a pipeline run *and* a transport test (fit on ELS rows, evaluate on HSLS rows — one extra split rule in the Analyst).

**Step 8 — DATA + CODE, the one large arc.** PERSUADE 2.0 → dataset adapter/registry/skill (`scripts/onboard_dataset.py` already profiles) → `src/agents/measurement.py` + `agent_prompts/measurement.yaml` as a host-side stage before DataEngineer → 6th entry in `_TASK_REGISTRY` → methodology skills for agreement metrics (report κ **and** `jedm_1034`'s correlation/bias/absolute-error decomposition, strictly more informative) and prompt-configuration sweeps (`jedm_1007`'s model × temperature grid is the published design) → Critic checklist rows → prompt-response cache → **the MANDATORY honesty skill, shipped in the same change**. Fires VF2-04/05/08 together — 53% of the anchor corpus. **Do not split this into three arcs.**

Two architectural constraints that force the host-side design: the sandbox has `network_disabled: true`, so LLM-generated analysis code cannot call any model API from inside the executor; and there is no NLP stack anywhere (`requirements.txt` 13 entries, `requirements-sandbox.txt` 12 pinned — zero text/NLP libraries, and offline the sandbox could not download a model at runtime). The host-side stage sidesteps both: it emits a labeled CSV column, and everything downstream works unchanged on an ordinary numeric matrix.

**Step 9 — CODE + HUMAN.** The VF2-07 artifact tier. The 7 anchors that genuinely score on this rule *ship something* — 12 heuristics, 3 design principles, a metric framework, a screening tool, a released model. We ship a paper. A calibrated-by-subgroup screening threshold table, or a released harmonized feature dictionary, is new pipeline output; **deciding which artifact is worth shipping is human judgment.**

**Explicitly deprioritized:** second-dataset plumbing through `main.py`/`PipelineContext`/DataEngineer/Analyst. The `did_els_hsls_panel` precedent gets the same result for far less.

**Expectation-setting, stated plainly:** steps 1–3 should close the gesture-to-abstract gap on VF2-06/07 and the framing defect on VF2-03. They will **not** move us from 1/13 to the anchors' 7/34 on the artifact probe. That gap closes at step 9 or not at all.

---

## 7. WHAT THIS DOES NOT TELL US

**The corpus.** Thirty-four anchors from one EDM year plus recent JEDM/JLA issues, three venues. Base rates from it are not venue-eternal, and a 0-count on n=34 has wide uncertainty — the true rate could be several percent and still produce zero hits. (Bookkeeping detail the shipped table does not record: of 49 ingest directories, 41 hold both files and 34 are unique by sha1; 7 of the 8 empties are failed twins, but `grading_probabilistic_programs_*` has no successful twin, so the EDM slice is 15 of 16 attempted papers.)

**The matching.** Rules fire on abstract/full-text strings. PDF de-hyphenation materially corrupts short-token regex over this corpus — verified: `gradu- ate`→ate, `indi- cate`→cate, `mod- els`→els, `shap- ing`→shap. Three broad probes and two AERA Open hits were pure artifacts. The shipped patterns are long enough to be safe; nothing derived in future is, without a de-hyphenation pass.

**The semantic classifications are readings, not counts.** "3/34 genuine VF2-06", "7/34 ship a real artifact", "14/15 authors'-own provenance" — these are one analyst's reading of abstracts and (for a few) full texts. The lexical counts (7, 12, 15, 9, 10, 0, 2, 3) are mechanical and were independently reproduced against the shipped table.

**VF2-06 may be a bad rule on its own terms.** Semantically measured at 3/34 (or 5/34 counting partials), it falls below the table's own `count_min: 6` positive-band floor and would not have shipped. Take the cheap VF2-06 wins because they are cheap and because a cross-context contrast is good science; do not over-invest in chasing the rule.

**Nobody verified the headline claim about our own specs.** The "3 negative rules fire on 68–82% of our specs" figure was taken as given by all three investigations; one explicitly declined to recompute it. It rests on idea cards none of them read.

**Our-side abstract statistics are biased downward.** The comparison corpus is 14 `paper.tex` files findable under `runs/` (13 after dropping a 10-word stub), dominated by the causal_soo smoketest family plus the ITR and prediction pilots. The newer gate-passing runs (`stream2_els_rigor`, `v4_psy_paper1/2`, `stream1_did_v2`, `e2_validation_cdm_journal`) store only spec/results/review artifacts in the searched tree. If the newer papers are better written, our VF2-07 abstract count is understated. Anchor-side numbers are unaffected.

**NOT MEASURED, and this is the biggest hole:** whether *any* of these changes raises LSAR gate scores. No experiment was run. The only supporting evidence is two reviewer comments, n=1 each. The entire causal chain from "fires the rule" to "scores better" is inferred, not tested.

**Also NOT MEASURED:** engineering hours for any proposed change (no basis; the precedent is that Arc R and the psychometrics arc each shipped a task type of comparable scope). Dollar or wall-clock cost of any proposed run. Compute for the text arc is measured only via `jedm_1034`'s published bill — ~$11 for a single cheap model at ~15k prompts, ~$415 for the four-model comparison that makes such papers publishable rather than a demo, against ~$5–8 for a current run, which is why a prompt-response cache is not optional. ASSISTments `answer_text` content. Whether ELS `F1SCH_ID` actually yields a usable grouped split (read from the registry, not from the CSV). ASAP-AES / CommonLit / NCTE licences. AERA Open's specific CC variant. Economics of Education Review (not in ERIC at all). The HSLS free-text conclusion rests on 2,000 of 23,503 rows.

**What would change the conclusions:**
1. A second EDM year of anchors — would test whether the 0/34s are structural or a one-year artifact.
2. Institutional full-text access to EEPA/AERJ/JREE — would replace abstract proxies that understate by ~7.5×.
3. **An actual A/B: run the same study with and without the abstract-decision sentence and the within/across contrast, and compare gate scores.** This is cheap after steps 1–3 and would convert the central inference of this document into a measurement.
4. Opening ASSISTments `answer_text` — would settle whether we hold any usable learner text at all.
5. A full-file HSLS scan — would close the last 91% of the free-text question.

*Read-only was observed throughout the three investigations; the four repository facts marked "verified here" were confirmed by read-only inspection while writing this document. Nothing in either repository was created, edited, or deleted.*