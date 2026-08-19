---
name: journal-apa7-style
layer: writing
description: Journal-manuscript writing (APA 7, apa7 LaTeX class, ~8000 words) — expanded IMRaD structure, biblatex \parencite/\textcite citations, no ACM-isms; applies when the run's venue_format is journal.
trigger_keywords:
  - journal
  - manuscript
  - apa
applicable_task_types: []
applicable_datasets: []
applicable_stages:
  - OutlineAgent
  - Writer
priority: 1
references_skills: []
resources: []
version: "1.0"
rule_severity: mandatory
---

# Journal Manuscript Style (APA 7 / apa7 class)

Applies when the run targets a JOURNAL venue (config
`writer.venue_format: journal`; the template is
`templates/paper_template_journal.tex`, apa7 `[man]` + biblatex/biber).

## Length and depth (~8000 words)

Target **7,000–9,000 words** of body text — roughly 3× a conference
paper. The extra length is DEPTH, not padding:

| Section | Words | Journal-depth expectations |
|---|---|---|
| Introduction | 1200–1600 | full problem motivation + theoretical framing + explicit contributions list |
| Literature Review | 1500–2000 | a REAL synthesis organized by themes with a gap analysis — not an annotated list; ≥ 12 citations |
| Methods | 1500–2000 | reproducible detail: data provenance, every decision rule verbatim, estimation specifics, missing-data handling, software cited |
| Results | 1200–1800 | full tables (not excerpts), every claim with its statistic, robustness/sensitivity subsection |
| Discussion | 1000–1400 | interpretation against the literature, implications for research AND practice, honest boundary conditions |
| Limitations & Future Directions | 400–600 | specific, actionable |

## LaTeX rules (apa7 + biblatex — NOT acmart)

- Citations: `\parencite{key}` (parenthetical) and `\textcite{key}`
  (narrative). NEVER `\cite{}` alone, never `[Author, Year]` text.
- NO ACM-isms: no `\begin{acks}`, no CCS concepts, no
  `\bibliographystyle`; the template's `\printbibliography` handles
  references (biber).
- Tables: `table` + `tabularx`/`tabular` with booktabs rules; every
  table and figure is REFERRED TO in text by `\ref`.
- Headings: `\section`/`\subsection` in APA title case.
- The abstract is 150–250 words; keywords 4–6.

## Voice and claims

Journal reviewers read for OVERCLAIM first: every quantitative claim
carries its statistic and decision rule; measurement/causal claim
licenses (invariance level, identification assumptions) are stated
where the claim is made, not only in limitations. Use "we" active
voice; students not subjects.
