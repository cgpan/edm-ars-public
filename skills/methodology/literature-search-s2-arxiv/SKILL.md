---
name: literature-search-s2-arxiv
layer: methodology
description: Combined Semantic Scholar + arXiv literature search; select 8–12 papers; never fabricate metadata; graceful API-failure fallback.
trigger_keywords:
  - literature
  - citations
  - papers
  - semantic-scholar
  - arxiv
applicable_task_types: []
applicable_datasets: []
applicable_stages:
  - ProblemFormulator
priority: 2
references_skills: []
resources: []
version: "1.0"
---

# Literature Search: Semantic Scholar + arXiv

The ProblemFormulator receives retrieved literature in the user message
(populated by orchestrator code that calls Semantic Scholar and arXiv,
merges by title Jaccard, and dedupes). The agent's job is to **select**
the most relevant papers from that list and assemble
`literature_context.papers` — never to invent metadata.

## Selection rules (CRITICAL — select 8–12 papers)

You MUST include 8–12 papers in `literature_context.papers`. Select papers
that:

1. Directly study the same or similar outcome variable.
2. Use machine learning / EDM methods on similar populations.
3. Study the same predictor constructs (e.g., self-efficacy, SES, prior
   achievement).
4. Provide methodological precedent (e.g., SHAP, model comparison,
   dataset usage).

If the retrieved literature has fewer than 8 papers, include all of
them — do not pad with fabricated entries.

## Verbatim metadata only

Copy the `paperId`, `title`, `authors`, `year`, and `abstract` EXACTLY as
they appear in the retrieved literature. Do NOT modify or fabricate any
fields. Papers not in the retrieved list will be filtered out by the
verification layer.

```json
{
  "literature_context": {
    "search_query": "math GPA prediction educational data mining",
    "papers": [
      {
        "paperId": "abc123",
        "title": "...",
        "authors": ["Last, First", "Last2, First2"],
        "year": 2022,
        "abstract": "..."
      }
    ],
    "novelty_evidence": "1–2 sentences explaining how this study differs from retrieved papers"
  }
}
```

`paperId` for arXiv preprints uses the prefix `arxiv:`, e.g.
`arxiv:2401.12345`. The Writer sanitizes this to `arxiv_2401.12345` for
BibTeX keys; the ProblemFormulator should leave the colon form as-is.

## API-failure fallback

If the API returned a non-200 response or timed out (the orchestrator
indicates this in the user message), set `papers` to an empty list and
note the failure in `novelty_evidence`. The pipeline will continue and
the Writer will use placeholder citations.

```json
{
  "literature_context": {
    "search_query": "...",
    "papers": [],
    "novelty_evidence": "Semantic Scholar and arXiv APIs were unavailable; novelty cannot be assessed against retrieved literature."
  }
}
```

## Novelty evidence

The `novelty_evidence` field must specifically contrast this study against
at least one retrieved paper when papers are available. Generic phrasing
("this study is novel because it uses ML on HSLS") is insufficient and
will fail Critic review.

## Source provenance

Canonical source: `agent_prompts/problem_formulator.yaml` §"Literature
Selection".

The orchestrator-side API code (HTTP calls, title-Jaccard dedup) lives in
`src/agents/problem_formulator.py` and is not part of this prompt-side
skill — that file is intentionally not extracted into a SKILL.md because
it is implementation, not LLM-facing knowledge.
