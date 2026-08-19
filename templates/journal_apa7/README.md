# APA 7th Edition LaTeX Manuscript Template

A reusable manuscript template built on the [`apa7`](https://ctan.org/pkg/apa7)
document class with `biblatex` (APA style). Replace every placeholder marked
`<< ... >>` with your own content.

## Files

| File | Purpose |
|------|---------|
| `main.tex` | Main manuscript: preamble, title page, abstract, body (Introduction / Methods / Results / Discussion), references, appendix include. |
| `suppl.tex` | Appendix / supplementary material, with `S1`, `S2`, … numbering. Loaded by `main.tex` via `\input{suppl}`. |
| `ref.bib` | Bibliography database with example entries (article, book, chapter, proceedings, report, software). |
| `figures/` | Drop figure image files here and reference them as `figures/<name>`. |

## Compiling

Requires `apa7`, `biblatex`, `biber`, and the packages loaded in the preamble
(all included on Overleaf by default).

1. Set the project compiler to **biber** (Overleaf: Menu → Compiler → biber).
2. Build order: `pdflatex` → `biber` → `pdflatex` → `pdflatex`.

> The template ships without images, so the example `\includegraphics` line in
> the Results section will error until you add a figure to `figures/` (or comment
> the figure environment out).

## Common edits

- **Title page:** `\title`, `\shorttitle`, `\authorsnames`, `\authorsaffiliations`, `\authornote`.
- **Abstract / keywords:** `\abstract{...}`, `\keywords{...}`.
- **Citations:** `\parencite{key}` (parenthetical) and `\textcite{key}` (in-text).
- **Document format:** switch `man` ↔ `jou`, or add `floatsintext`, in the
  `\documentclass[...]{apa7}` options.
