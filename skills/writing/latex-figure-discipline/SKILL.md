---
name: latex-figure-discipline
layer: writing
description: Every generated figure gets a full figure environment with \Description{} and a label that matches every \ref to it.
trigger_keywords:
  - latex
  - figure
  - includegraphics
  - description
  - acmart
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

# LaTeX Figure Discipline

Every figure listed in `results.json.figures_generated` MUST be embedded
as a full figure environment in the paper body. Three failure modes
recur and are addressed below.

## Basic figure form

```latex
\begin{figure}[h]
\centering
\includegraphics[width=\columnwidth]{filename.png}
\caption{Your caption here.}
\Description{A text description of the figure for accessibility.}
\label{fig:label}
\end{figure}
```

## Rules (CRITICAL)

1. **Embed every listed figure.** For each filename in
   `figures_generated`, write a complete `\begin{figure}[h]...\end{figure}`
   block at the first relevant location in the text (e.g.,
   `shap_summary.png` in Results §Feature Importance). Never write
   `Figure \ref{fig:xxx} (not shown)` — this produces `??` in the PDF
   because there is no matching `\label{fig:xxx}`.
2. **Label must match `\ref`.** If you write `\ref{fig:shap_summary}` in
   prose, the figure environment MUST contain
   `\label{fig:shap_summary}`. Use consistent, predictable label names:
   `fig:shap_summary`, `fig:shap_importance`, `fig:pdp_<feature>`,
   `fig:roc_curves`, `fig:residual_plot`, etc.
3. **`\Description{}` is required** for every `\begin{figure}` —
   required by acmart for accessibility compliance. Omitting it causes
   a compilation warning and a real accessibility regression.
4. **Do NOT write "(not shown)" or "(see supplementary material)"** —
   all generated figures are available in the output directory and must
   be included.

## Suggested figure placement

| Figure | Section |
|---|---|
| `shap_summary.png`, `shap_importance.png` | §Results / Feature Importance subsection |
| `pdp_*.png` | §Results / Feature Importance subsection (after SHAP plots) |
| `roc_curves.png` | §Results / Model Comparison subsection |
| `calibration_curve.png`, `confusion_matrix.png` | §Results / Model Comparison subsection |
| `residual_plot.png` (regression only) | §Results / Model Comparison subsection |

## Source provenance

Canonical source: `agent_prompts/writer.yaml` §"Figures".

Merged content from: none — single-sourced.
