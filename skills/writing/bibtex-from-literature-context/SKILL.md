---
name: bibtex-from-literature-context
layer: writing
description: Generate references.bib from literature_context.papers; sanitize keys; every cited key must have an entry; never invent a venue; cite to the venue depth norm; [Author, Year] fallback only when papers is empty.
trigger_keywords:
  - bibtex
  - references
  - citations
  - bib
  - cite
applicable_task_types: []
applicable_datasets: []
applicable_stages:
  - Writer
priority: 1
references_skills: []
resources: []
version: "1.2"
rule_severity: mandatory
---

# BibTeX Generation from `literature_context.papers`

The Writer produces `references.bib` alongside `paper.tex`. The bib
content is built from `literature_context.papers` — never fabricated.

## BibTeX entry shape

For each paper:

```bibtex
@article{paperId,
  author    = {Last, First and Last2, First2},
  title     = {Paper Title},
  year      = {2023},
  journal   = {Journal Name or Venue},
}
```

Pick the entry type by paper kind:

- `@article` for journal papers.
- `@inproceedings` for conference papers.
- `@misc` for arXiv preprints (paperId starts with `arxiv:`); add
  `note = {arXiv preprint}`.

## Key rules

1. **Use the `paperId` as the BibTeX key and the `\cite{}` argument.**
2. **Sanitize colons in arXiv IDs.** `arxiv:2401.12345` becomes
   `arxiv_2401.12345` in the bib key AND in every `\cite{}` reference
   that points to it. The two must match exactly.
3. **Authors** are joined with literal " and " (BibTeX convention) —
   never with commas in author lists.
4. **Venue/journal**: use the `venue` field from the retrieved metadata.
   When it is absent, emit `@misc` with
   `note = {Venue metadata unavailable}`. **NEVER invent a venue.** A
   guessed proceedings name is a fabricated citation that survives into
   the PDF; 29 such entries ("Proceedings of the Educational Data Mining
   Conference", stamped on papers that were never published there) were
   found across shipped manuscripts before this rule existed.
5. **Every key you `\cite{}` MUST have a matching `@entry`.** The two
   sets are checked deterministically after you finish. A cited key with
   no entry renders as `[?]` in the compiled PDF.
6. **Cite ONLY keys listed in the `## Available Citation Keys` block** of
   your user message. Any key outside that list is deleted from the
   manuscript automatically, and the sentence loses its support. If you
   need a citation you do not have, write the claim without one or
   attribute it in prose.
7. **No orphan entries.** Do not pad `references.bib` with entries the
   text never cites; a reference list is a record of engagement, not a
   length target.

## Citation depth by venue

Published papers at our target venues cite far more prior work than a
first draft naturally produces. Anchor-corpus norms (real papers,
`data_registry/venue_norms.yaml`):

| Venue | 25th pct | Median |
|-------|----------|--------|
| EDM (conference) | 15 | 34 |
| JLA (journal) | 47 | 65 |
| JEDM (journal) | 54 | 62 |

Engage the available references across Introduction, Related Work, and
Discussion until the manuscript is in that range. Depth means using the
literature you were given — not citing the same three papers repeatedly.

## Citation rules in the LaTeX body

**Normal path (`literature_context.papers` non-empty):**

- Use `\cite{paperId}` for the papers in `literature_context.papers`,
  working toward the venue depth in the table above.
- Cite substantively in Related Work: position this study against the
  prior work, do not list titles.
- NEVER use `[Author, Year]` placeholder format when real papers exist.

**Fallback path (`literature_context` is null OR `papers` is `[]`):**

- Use `[Author, Year]` placeholder citation format in the LaTeX text (no
  `\cite{}` commands).
- Generate an empty `references.bib` with this comment at the top:

  ```bibtex
  % Semantic Scholar API was unavailable; citations are placeholders only.
  ```

- Do NOT fabricate paper titles, authors, or venues.

The fallback applies ONLY when `papers` is literally empty or
`literature_context` is null/missing. If `papers` has even one entry,
use the normal path.

## APA 7 in-text style

When using `\cite{}`, the resulting in-text format follows APA 7 via the
`ACM-Reference-Format` bibliography style (already set by the template).
Do not manually format author-year citations — let BibTeX handle it.

## Source provenance

Canonical source: `agent_prompts/writer.yaml` §"BibTeX Generation" +
§"Citation Rules".

Merged content from: none — single-sourced.
