---
name: latex-table-discipline
layer: writing
description: Column-count discipline, resizebox usage rules, and threeparttable for table footnotes inside the float.
trigger_keywords:
  - latex
  - table
  - tabular
  - resizebox
  - threeparttable
  - booktabs
applicable_task_types: []
applicable_datasets: []
applicable_stages:
  - Writer
priority: 1
references_skills: []
resources: []
version: "1.1"
rule_severity: mandatory
---

# LaTeX Table Discipline

Tables in ACM acmart sigconf are narrow (~88mm per column) and float
freely. Three failure modes recur and are addressed below.

## Basic table form

```latex
\begin{table}[h]
\caption{Your caption here.}
\label{tab:label}
\begin{tabular}{lrrr}
\toprule
Column1 & Column2 & Column3 & Column4 \\
\midrule
data & data & data & data \\
\bottomrule
\end{tabular}
\end{table}
```

## CRITICAL: column-count must match

The number of specifier letters in `{lrrr}` MUST exactly match the number
of `&`-separated columns in EVERY row, including the header. A mismatch
causes a compile failure with an unhelpful error message.

**Mandatory counting procedure — follow this every time:**

1. Write out a sample data row first: `Model & AUC & CI & Accuracy & F1 \\`
2. Count the `&` characters: 4 ampersands → **5 columns**
3. Write the tabular spec with exactly that many letters: `{lrrrr}` (1 + 4 = 5)
4. Verify every other row in the table also has exactly 4 `&` characters

Never write the spec before counting. One `r` too many or too few causes
a compile error.

## CRITICAL: wide tables (5+ columns) MUST use `\resizebox`

ACM sigconf is two-column. Any table with 5 or more columns WILL overflow
the column and extend into the margin unless resized. Wrap the `tabular`
in `\resizebox{\columnwidth}{!}{...}`:

```latex
\begin{table}[h]
\caption{Model comparison across all six models.}
\label{tab:model_comparison}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lrrrrr}
\toprule
Model & AUC & Acc. & Prec. & Recall & F1 \\
\midrule
XGBoost & 0.82 & 0.76 & 0.74 & 0.71 & 0.72 \\
\bottomrule
\end{tabular}%
}
\end{table}
```

- Use abbreviated headers to save space: "Acc." not "Accuracy", "Prec."
  not "Precision", "CI" not "95% CI", "LR" not "Logistic Regression".
- The `%` after `{` and after `\end{tabular}` suppresses spurious
  whitespace.
- `\resizebox` comes from `graphicx`, already loaded by the template.
- If using `threeparttable`, wrap the entire
  `\begin{threeparttable}...\end{threeparttable}` block (not just the
  tabular) inside `\resizebox`.

## CRITICAL: narrow tables (fewer than 5 columns) MUST NOT use `\resizebox`

`\resizebox{\columnwidth}{!}` scales the table to fill the full column
width. On a 2–4 column table this makes the font absurdly large. Use a
plain `tabular` without any wrapper.

## Table notes — use `threeparttable`, never loose text after `\end{table}`

Tables float. Any `\noindent{\small ...}` placed after `\end{table}` is
loose body text that will be separated from the table when it floats.

Always use `threeparttable` for tables with footnotes. The note lives
inside the float and stays attached:

```latex
\begin{table}[h]
\caption{Your caption.}
\label{tab:label}
\begin{threeparttable}
\begin{tabular}{lrr}
\toprule
Col1 & Col2 & Col3\tnote{*} \\
\midrule
data & data & data \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\footnotesize
\item[*] Your note text here.
\end{tablenotes}
\end{threeparttable}
\end{table}
```

- Use `\tnote{*}` (or `\tnote{\dag}`) in table cells to place the marker.
- Use `\item[*]` inside `tablenotes` for the note text.
- `threeparttable` is already loaded in the template — do NOT add
  `\usepackage` yourself.

NEVER place a footnote as `\noindent{\small ...}` after `\end{table}`.

## Font/size declarations (CRITICAL — declaration must be inside braces)

`\small`, `\footnotesize`, `\large`, `\itshape`, etc. are *declarations*,
not commands that take arguments. The declaration must be **inside** the
group it scopes:

- CORRECT: `{\small text}` or `
oindent{\small text}`
- WRONG:   `\small{text}` or `
oindent\small{text}` — leaks into the surrounding text

The same rule applies to all size and shape declarations.

## Source provenance

Canonical source: `agent_prompts/writer.yaml` §"Tables" + §"Table Notes".

Merged content from: none — single-sourced.
