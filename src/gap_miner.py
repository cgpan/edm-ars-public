"""V3.2 Arc D (D4) — deterministic gap-matrix miner.

Turns the retrieved literature (``literature_context``-shaped
``s2_context``) into an unstudied-cell matrix the ProblemFormulator
must engage with: cells are (outcome-family × method-family), detected
via keyword sweeps over titles + abstracts. Zero LLM calls — the LLM's
job (via the ``gap-driven-question-mining`` instruction) is to pick a
sparse cell and NAME it in ``expected_contribution``, quoting which
retrieved papers sit nearest to it.

Deliberately coarse: the matrix marks where the retrieved corpus is
silent, not where the whole literature is — the PF prompt is told to
phrase claims as "within the retrieved corpus".
"""
from __future__ import annotations

OUTCOME_FAMILIES: dict[str, tuple[str, ...]] = {
    "achievement_scores": ("achievement", "test score", "math score", "theta"),
    "gpa": ("gpa", "grade point"),
    "dropout_completion": ("dropout", "completion", "graduat", "credential"),
    "college_enrollment": ("college", "enrollment", "postsecondary", "attendance"),
    "stem_pathways": ("stem", "major", "science pathway"),
    "noncognitive": ("self-efficacy", "identity", "belonging", "engagement", "motivation"),
}

METHOD_FAMILIES: dict[str, tuple[str, ...]] = {
    "prediction_ml": ("predict", "machine learning", "classifier", "random forest", "xgboost", "neural"),
    "causal_average": ("causal", "treatment effect", "propensity", "matching", "instrumental", "difference-in-differences", "regression discontinuity"),
    "targeting_itr": ("treatment rule", "policy learning", "individualized", "personalized", "targeting", "heterogeneous effect", "cate"),
    "fairness": ("fairness", "bias", "equity", "disparit", "subgroup"),
}


def build_gap_matrix(s2_context: dict | None) -> dict:
    """Count retrieved papers per (outcome-family, method-family) cell."""
    papers = (s2_context or {}).get("papers") or []
    texts = [
        ((p.get("title") or "") + " " + (p.get("abstract") or "")).lower()
        for p in papers
        if isinstance(p, dict)
    ]
    matrix: dict[str, dict[str, int]] = {
        o: {m: 0 for m in METHOD_FAMILIES} for o in OUTCOME_FAMILIES
    }
    for text in texts:
        hit_outcomes = [
            o for o, kws in OUTCOME_FAMILIES.items()
            if any(k in text for k in kws)
        ]
        hit_methods = [
            m for m, kws in METHOD_FAMILIES.items()
            if any(k in text for k in kws)
        ]
        for o in hit_outcomes:
            for m in hit_methods:
                matrix[o][m] += 1
    sparse = sorted(
        (o, m)
        for o, row in matrix.items()
        for m, count in row.items()
        if count == 0
    )
    return {"n_papers": len(texts), "matrix": matrix, "sparse_cells": sparse}


def format_gap_matrix(gap: dict) -> str:
    """Render for injection into the PF user message."""
    if gap["n_papers"] == 0:
        return (
            "## Gap Matrix (deterministic)\n"
            "No retrieved papers — gap claims must be framed as "
            "untested against the literature."
        )
    lines = [
        "## Gap Matrix (deterministic — coverage of the RETRIEVED corpus "
        f"only, n={gap['n_papers']} papers)",
        "| outcome family \\ method | " + " | ".join(METHOD_FAMILIES) + " |",
        "|---|" + "---|" * len(METHOD_FAMILIES),
    ]
    for o, row in gap["matrix"].items():
        lines.append(
            f"| {o} | " + " | ".join(str(row[m]) for m in METHOD_FAMILIES) + " |"
        )
    if gap["sparse_cells"]:
        cells = ", ".join(f"({o} × {m})" for o, m in gap["sparse_cells"][:8])
        lines.append(
            f"Sparse cells (0 retrieved papers): {cells}"
            + (" …" if len(gap["sparse_cells"]) > 8 else "")
        )
    lines.append(
        "Your research question should fill a sparse or thin cell; name "
        "the cell explicitly in `expected_contribution` and phrase the "
        "gap claim as 'within the retrieved corpus'."
    )
    return "\n".join(lines)
