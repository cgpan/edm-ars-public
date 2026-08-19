---
name: unverified-flag-and-appendix
layer: writing
description: When the Critic verdict is not PASS, prepend a WARNING quote to the Introduction and append the full Critic report as an appendix.
trigger_keywords:
  - unverified
  - critic
  - revision
  - appendix
  - verdict
applicable_task_types: []
applicable_datasets: []
applicable_stages:
  - Writer
priority: 1
references_skills: []
resources: []
version: "1.0"
rule_severity: mandatory
---

# UNVERIFIED Flag and Critic Appendix

If `review_report.overall_verdict != "PASS"` (i.e., the maximum revision
cycles were exhausted without achieving a PASS), the Writer MUST do
both of the following.

## 1. Prepend the WARNING quote block

Place this block at the top of the Introduction body so it appears
immediately after `\maketitle` in the rendered PDF:

```latex
\begin{quote}
\textbf{WARNING: This paper has unresolved methodological issues identified by automated
review. See the Appendix for the full review report.}
\end{quote>
```

The quote environment is intentionally heavy-handed — readers should
not be able to miss the warning.

## 2. Append the full Critic report

Use the `%%PLACEHOLDER:APPENDIX%%` slot to include the Critic report as
a description list:

```latex
\appendix
\section*{Appendix: Automated Critic Review Report}
\begin{description}
\item[Overall Verdict:] REVISE
\item[Quality Score:] X/10
% List each unresolved issue as a \item[Category:] description
\end{description}
```

Pull the issue list from `review_report.json` — every issue with
severity `critical` or `major` should appear with its category,
description, and recommendation.

## When verdict IS PASS

Remove the `%%PLACEHOLDER:APPENDIX%%` line entirely from the template
output. Do not leave a stub appendix.

## Both required — neither alone is sufficient

The WARNING block tells the reader the paper is unverified at a glance;
the appendix tells them exactly what was wrong. Neither substitutes for
the other.

## Source provenance

Canonical source: `agent_prompts/writer.yaml` §"UNVERIFIED Flag".

Merged content from: none — single-sourced. Verdict semantics
(PASS/REVISE/ABORT thresholds, max-revision-cycle handling) live in
the Critic role and are not duplicated here.
