---
name: natural-academic-prose
layer: writing
description: Humanizer for paper prose — removes statistical AI-writing tells (significance inflation, -ing appendages, copula avoidance, rule-of-three, AI vocabulary, negative parallelism, em-dash overuse) while keeping a precise scholarly voice.
trigger_keywords:
  - paper
  - writing
  - manuscript
  - prose
  - journal
applicable_task_types: []
applicable_datasets: []
applicable_stages:
  - Writer
priority: 1
references_skills:
  - formulaic-construction-ban
  - paper-writing-style-rules
resources: []
version: "1.0"
rule_severity: mandatory
---

# Natural Academic Prose (AI-tell removal)

Reviewers (human and automated) now recognize the statistical
fingerprints of LLM prose. Every rule below names a tell and its fix.
This complements `formulaic-construction-ban` (which bans structural
templates); this skill targets sentence-level tells. The register stays
scholarly — "add personality" advice for blogs does NOT apply — but
scholarly prose is still direct, varied, and concrete.

## The banned tells

1. **Significance inflation.** Never: "marks a pivotal moment",
   "underscores the importance", "a testament to", "reflects broader
   trends", "evolving landscape", "setting the stage for". State what
   the result IS; the reader judges importance.
2. **Superficial -ing appendages.** Never bolt "-ing" phrases onto a
   sentence to fake depth ("...highlighting the interplay between...",
   "...underscoring the need for..."). If the point matters, give it
   its own sentence with a subject and evidence.
3. **Copula avoidance.** Prefer "is/are/has" over "serves as",
   "stands as", "represents", "boasts", "features". "The scale is
   reliable (ω = .92)" beats "the scale demonstrates robust reliability".
4. **AI vocabulary.** Avoid: delve, crucial, pivotal, showcase,
   underscore (verb), tapestry, landscape (abstract), vibrant,
   intricate, fostering, leveraging, robust (as praise), notably,
   moreover-chains. Plain alternatives exist for all of them.
5. **Rule-of-three reflex.** Do not force triplet lists ("innovation,
   insight, and impact"). Use the number of items the content has.
6. **Negative parallelism.** Never "not merely X but Y", "it's not
   just about X; it's about Y". State Y.
7. **False ranges.** No "from X to Y" unless X and Y sit on a real
   scale.
8. **Elegant variation.** Call the same construct by the same name
   every time (the estimator, the scale, the cohort). Synonym cycling
   ("the instrument... the measure... the tool") confuses; repetition
   is correct in science.
9. **Em-dash economy.** At most a few per page; prefer commas,
   parentheses, or a new sentence.
10. **Vague attribution.** Never "researchers have noted",
    "it is widely recognized" — cite the specific work or drop the
    claim. (In Related Work every claim has a \parencite/\cite.)
11. **Filler and hedging stacks.** "In order to"→"to"; "it is important
    to note that"→delete; never stack hedges ("could potentially
    suggest that ... might"). One calibrated hedge per claim, tied to
    the actual uncertainty (a CI, a limitation).
12. **Generic conclusions.** No "promising avenue", "exciting
    direction", "paves the way". Future work names a concrete next
    study and what it would settle.
13. **Announcement sentences.** Don't narrate the paper ("In this
    section we will explore..."). After a heading, start with content.
    Exception: the one-paragraph roadmap at the end of the
    Introduction, which is conventional.
14. **Uniform rhythm.** Vary sentence length deliberately. A short
    sentence lands a result. Longer sentences carry qualification and
    mechanism. If every sentence in a paragraph has 20-28 words,
    rewrite two of them.

## What scholarly voice keeps

- Active voice with "we" for the authors' actions; passive only when
  the actor is genuinely irrelevant.
- Concrete numbers in prose, not adjectives: "AUC 0.81" not "strong
  performance"; "1,586 students" not "a large sample".
- Honest texture: a limitation stated plainly reads as human; a
  limitation wrapped in "despite these challenges, the approach remains
  promising" reads as machine.

## Self-audit pass (Writer must do this)

Before emitting the final LaTeX, re-read the draft asking: "which
sentences would a reader flag as obviously machine-written?" Rewrite
those. Check specifically for tells 1-4 (the highest-frequency ones)
in the Introduction and Discussion, where they cluster.
