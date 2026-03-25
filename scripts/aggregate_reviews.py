#!/usr/bin/env python
"""Aggregate LSAR reviews across EDM-ARS runs to diagnose systematic weaknesses.

Scans output/run_*/lsar_review/ directories, computes dimensional score
statistics, clusters recurring weaknesses (via LLM), maps issues to EDM-ARS
agents, and generates recommended prompt changes.

Usage:
    python scripts/aggregate_reviews.py
    python scripts/aggregate_reviews.py --output-dir output/ --save-report
    python scripts/aggregate_reviews.py --min-runs 3
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIMENSION_AGENT_MAP: dict[str, str] = {
    "Relevance": "ProblemFormulator",
    "Novelty": "ProblemFormulator",
    "Theoretical/Conceptual Grounding": "Writer",
    "Methodological Rigor": "Analyst",
    "Empirical Support/Results": "Analyst",
    "Empirical Support / Results": "Analyst",
    "Significance & Impact": "Writer",
    "Ethics, Fairness & Equity": "Writer",
    "Clarity of Communication": "Writer",
}

# Canonical dimension names (normalized)
CANONICAL_DIMENSIONS: list[str] = [
    "Relevance",
    "Novelty",
    "Theoretical/Conceptual Grounding",
    "Methodological Rigor",
    "Empirical Support/Results",
    "Significance & Impact",
    "Ethics, Fairness & Equity",
    "Clarity of Communication",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ReviewRecord:
    """A single LSAR review loaded from disk."""

    run_id: str
    timestamp: str
    cycle: int
    scores: dict[str, Any]
    review: Optional[dict[str, Any]]
    research_question: Optional[str]


@dataclass
class DimensionStats:
    """Aggregate statistics for a single review dimension."""

    name: str
    values: list[int] = field(default_factory=list)
    mean: float = 0.0
    std: float = 0.0
    min_val: int = 0
    max_val: int = 0
    trend: str = "→"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Optional[dict]:
    """Load a JSON file, returning None on any error."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def _normalize_dimension(name: str) -> str:
    """Normalize dimension name variants (e.g. space around '/')."""
    return re.sub(r"\s*/\s*", "/", name.strip())


def _extract_timestamp(run_id: str) -> str:
    """Extract timestamp portion from run_YYYYMMDD_HHMMSS."""
    match = re.search(r"(\d{8}_\d{6})", run_id)
    return match.group(1) if match else run_id


def _load_llm_config() -> tuple[Any, str]:
    """Load config.yaml, build Anthropic client and resolve model name.

    Returns (client, model_id) or raises if no API key is available.
    """
    import anthropic

    config: dict[str, Any] = {}
    config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    if config_path.is_file():
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    provider = config.get("llm_provider", "anthropic")

    if provider == "minimax":
        api_key = os.environ.get("MINIMAX_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "llm_provider is 'minimax' but MINIMAX_API_KEY is not set."
            )
        base_url = config.get("minimax", {}).get(
            "base_url", "https://api.minimax.io/anthropic"
        )
        model = config.get("minimax", {}).get("models", {}).get(
            "revision_writer", "MiniMax-M2.7"
        )
        client = anthropic.Anthropic(api_key=api_key, base_url=base_url)
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY is not set.")
        model = "claude-sonnet-4-6"
        client = anthropic.Anthropic(api_key=api_key)

    return client, model


# ---------------------------------------------------------------------------
# 1. Discovery
# ---------------------------------------------------------------------------

def discover_reviews(output_dir: Path) -> list[ReviewRecord]:
    """Find and load all LSAR reviews under output_dir/run_*/lsar_review/."""
    records: list[ReviewRecord] = []

    run_dirs = sorted(
        [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("run_")],
        key=lambda d: d.name,
    )

    for run_dir in run_dirs:
        lsar_dir = run_dir / "lsar_review"
        if not lsar_dir.is_dir():
            continue

        # Determine layout: cycle subdirs or flat
        cycle_dirs = sorted(
            [d for d in lsar_dir.iterdir() if d.is_dir() and d.name.startswith("cycle_")],
            key=lambda d: d.name,
        )

        if cycle_dirs:
            # Use the last (highest) cycle
            review_dir = cycle_dirs[-1]
            cycle_num = int(review_dir.name.split("_")[1]) if "_" in review_dir.name else 1
        else:
            # Flat layout (older runs)
            review_dir = lsar_dir
            cycle_num = 0

        # Load scores
        scores = _load_json(review_dir / "scores.json")
        if scores is None:
            continue  # No scores = skip this review

        # Load review (optional)
        review = _load_json(review_dir / "review.json")

        # Load research question from research_spec.json
        rq: Optional[str] = None
        spec = _load_json(run_dir / "research_spec.json")
        if spec:
            rq = spec.get("research_question")

        run_id = run_dir.name
        records.append(ReviewRecord(
            run_id=run_id,
            timestamp=_extract_timestamp(run_id),
            cycle=cycle_num,
            scores=scores,
            review=review,
            research_question=rq,
        ))

    return records


# ---------------------------------------------------------------------------
# 2. Dimensional statistics
# ---------------------------------------------------------------------------

def compute_dimension_stats(records: list[ReviewRecord]) -> list[DimensionStats]:
    """Compute per-dimension statistics across all reviews."""
    # Collect scores by normalized dimension name
    dim_scores: dict[str, list[int]] = {d: [] for d in CANONICAL_DIMENSIONS}

    for rec in records:
        dimensions = rec.scores.get("dimensions", [])
        for dim in dimensions:
            name = _normalize_dimension(dim.get("name", ""))
            score = dim.get("score", 0)
            if isinstance(score, str):
                try:
                    score = int(score)
                except ValueError:
                    continue
            if name in dim_scores:
                dim_scores[name].append(score)

    # Build stats
    stats_list: list[DimensionStats] = []
    for dim_name in CANONICAL_DIMENSIONS:
        vals = dim_scores[dim_name]
        ds = DimensionStats(name=dim_name, values=vals)

        if vals:
            ds.mean = round(statistics.mean(vals), 1)
            ds.std = round(statistics.stdev(vals), 1) if len(vals) > 1 else 0.0
            ds.min_val = min(vals)
            ds.max_val = max(vals)

            # Trend: first half vs second half
            if len(vals) >= 2:
                mid = len(vals) // 2
                first_half = statistics.mean(vals[:mid])
                second_half = statistics.mean(vals[mid:])
                diff = second_half - first_half
                if diff > 0.5:
                    ds.trend = "↑"
                elif diff < -0.5:
                    ds.trend = "↓"

        stats_list.append(ds)

    return stats_list


# ---------------------------------------------------------------------------
# 3. Weakness extraction
# ---------------------------------------------------------------------------

def extract_weaknesses(records: list[ReviewRecord]) -> list[dict[str, str]]:
    """Extract all weaknesses from reviews, tagged with run_id."""
    weaknesses: list[dict[str, str]] = []
    for rec in records:
        if rec.review is None:
            continue
        for w in rec.review.get("weaknesses", []):
            if isinstance(w, str) and w.strip():
                weaknesses.append({"text": w.strip(), "run_id": rec.run_id})
    return weaknesses


# ---------------------------------------------------------------------------
# 4. Weakness clustering (LLM)
# ---------------------------------------------------------------------------

def cluster_weaknesses(
    weaknesses: list[dict[str, str]],
    client: Any = None,
    model: str = "claude-sonnet-4-6",
) -> Optional[list[dict[str, Any]]]:
    """Cluster weaknesses into themes using an LLM call.

    Returns None if LLM is unavailable or there are too few weaknesses.
    """
    if client is None or len(weaknesses) <= 3:
        return None

    dimension_list = ", ".join(CANONICAL_DIMENSIONS)
    weakness_texts = "\n".join(
        f"- [{w['run_id']}] {w['text']}" for w in weaknesses
    )

    prompt = f"""You are analyzing recurring weaknesses across multiple academic paper reviews.

Group the following weaknesses into up to 5 themes. For each theme provide:
- "theme": short label (3-6 words)
- "count": how many weaknesses fit this theme
- "examples": 1-2 direct quotes (shortened to ~50 words each)
- "dimension": the most relevant review dimension from: [{dimension_list}]

Weaknesses:
{weakness_texts}

Output a JSON array only, no other text. Example:
```json
[
  {{"theme": "Missing survey weights", "count": 3, "examples": ["Survey weights not applied..."], "dimension": "Methodological Rigor"}}
]
```"""

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text
    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*\n?", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\n?```\s*$", "", text.strip(), flags=re.MULTILINE)

    try:
        themes = json.loads(text)
        if isinstance(themes, list):
            return themes
    except json.JSONDecodeError:
        pass
    return None


# ---------------------------------------------------------------------------
# 5. Agent diagnosis
# ---------------------------------------------------------------------------

def build_agent_diagnosis(
    stats: list[DimensionStats],
) -> dict[str, dict[str, Any]]:
    """Compute per-agent average scores from mapped dimensions."""
    # Build normalized map
    norm_map: dict[str, str] = {}
    for dim, agent in DIMENSION_AGENT_MAP.items():
        norm_map[_normalize_dimension(dim)] = agent

    # Collect per-agent dimension scores
    agent_dims: dict[str, list[tuple[str, float]]] = {}
    for ds in stats:
        norm = _normalize_dimension(ds.name)
        agent = norm_map.get(norm)
        if agent and ds.values:
            agent_dims.setdefault(agent, []).append((ds.name, ds.mean))

    # Build diagnosis
    diagnosis: dict[str, dict[str, Any]] = {}
    for agent, dims in agent_dims.items():
        avg = round(statistics.mean([score for _, score in dims]), 1)
        dim_details = ", ".join(f"{name}: {score}" for name, score in dims)
        low = [name for name, score in dims if score < 6.0]
        diagnosis[agent] = {
            "avg": avg,
            "dimensions": dim_details,
            "priority_fixes": low,
        }

    return diagnosis


# ---------------------------------------------------------------------------
# 6. Prompt recommendations (LLM)
# ---------------------------------------------------------------------------

def generate_prompt_recommendations(
    diagnosis: dict[str, dict[str, Any]],
    themes: Optional[list[dict[str, Any]]],
    client: Any = None,
    model: str = "claude-sonnet-4-6",
) -> dict[str, str]:
    """Generate prompt change recommendations for weak agents."""
    if client is None:
        return {}

    # Find agents below threshold
    weak_agents: dict[str, dict[str, Any]] = {
        agent: info for agent, info in diagnosis.items()
        if info["avg"] < 6.0
    }

    if not weak_agents:
        return {}

    # Map themes to agents
    agent_themes: dict[str, list[str]] = {}
    if themes:
        for t in themes:
            dim = _normalize_dimension(t.get("dimension", ""))
            agent = DIMENSION_AGENT_MAP.get(dim)
            if agent and agent in weak_agents:
                agent_themes.setdefault(agent, []).append(t.get("theme", ""))

    recommendations: dict[str, str] = {}

    for agent, info in weak_agents.items():
        theme_list = agent_themes.get(agent, [])
        theme_str = ", ".join(theme_list) if theme_list else "No specific themes identified"

        prompt = f"""You are helping improve an AI agent's system prompt for an automated research paper pipeline.

Agent: {agent}
Role: {"Formulates research questions" if agent == "ProblemFormulator" else "Writes paper prose" if agent == "Writer" else "Runs ML analysis"}
Current weak dimensions: {info['dimensions']}
Average score: {info['avg']}/10
Recurring weakness themes: {theme_str}
Priority fix areas: {', '.join(info['priority_fixes']) if info['priority_fixes'] else 'General improvement needed'}

The agent's system prompt is in agent_prompts/{agent.lower().replace(' ', '_')}.yaml.

Suggest 2-3 specific, actionable additions or changes to the agent's system prompt that would address these weaknesses. Be concrete — reference what the prompt should say, not vague advice.

Output plain text, no JSON."""

        response = client.messages.create(
            model=model,
            max_tokens=1024,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )
        recommendations[agent] = response.content[0].text.strip()

    return recommendations


# ---------------------------------------------------------------------------
# 7. Report formatting
# ---------------------------------------------------------------------------

def format_report(
    records: list[ReviewRecord],
    stats: list[DimensionStats],
    weaknesses: list[dict[str, str]],
    themes: Optional[list[dict[str, Any]]],
    diagnosis: dict[str, dict[str, Any]],
    recommendations: dict[str, str],
) -> str:
    """Format the full aggregate report as markdown."""
    lines: list[str] = []
    lines.append("# LSAR Review Aggregate Report")
    lines.append("")
    lines.append(f"**Runs analyzed:** {len(records)}")
    lines.append(f"**Date range:** {records[0].run_id} → {records[-1].run_id}")
    lines.append("")

    # Run summary
    lines.append("## Runs Included")
    lines.append("")
    for rec in records:
        rq = rec.research_question or "(no research question)"
        rq_short = rq[:100] + "..." if len(rq) > 100 else rq
        overall = rec.scores.get("overall_score", "?")
        rec_str = rec.scores.get("recommendation", "?")
        lines.append(f"- **{rec.run_id}** (cycle {rec.cycle}): "
                      f"score={overall}, rec={rec_str} — {rq_short}")
    lines.append("")

    # (a) Dimensional score summary
    lines.append("## (a) Dimensional Score Summary")
    lines.append("")
    lines.append("| Dimension | Mean | Std | Min | Max | Trend |")
    lines.append("|---|---|---|---|---|---|")
    for ds in stats:
        if ds.values:
            lines.append(
                f"| {ds.name} | {ds.mean} | {ds.std} | {ds.min_val} | "
                f"{ds.max_val} | {ds.trend} |"
            )
        else:
            lines.append(f"| {ds.name} | — | — | — | — | — |")
    lines.append("")

    # Overall
    overall_scores = [rec.scores.get("overall_score", 0) for rec in records]
    if overall_scores:
        lines.append(
            f"**Overall score:** mean={round(statistics.mean(overall_scores), 1)}, "
            f"range=[{min(overall_scores)}, {max(overall_scores)}]"
        )
        lines.append("")

    # (b) Recurring weaknesses
    lines.append("## (b) Recurring Weaknesses")
    lines.append("")

    if themes:
        for i, t in enumerate(themes, 1):
            lines.append(f"### Theme {i}: {t.get('theme', '?')} "
                          f"(count: {t.get('count', '?')}, "
                          f"dimension: {t.get('dimension', '?')})")
            agent = DIMENSION_AGENT_MAP.get(
                _normalize_dimension(t.get("dimension", "")), "?"
            )
            lines.append(f"**Responsible agent:** {agent}")
            lines.append("")
            for ex in t.get("examples", []):
                lines.append(f"> {ex}")
                lines.append("")
    elif weaknesses:
        lines.append("*(Too few reviews for clustering — showing raw weaknesses)*")
        lines.append("")
        for w in weaknesses:
            lines.append(f"- [{w['run_id']}] {w['text'][:200]}")
        lines.append("")
    else:
        lines.append("No weaknesses found in reviews.")
        lines.append("")

    # (c) Agent-level diagnosis
    lines.append("## (c) Agent-Level Diagnosis")
    lines.append("")
    lines.append("| Agent | Avg Score | Dimensions | Priority Fix Areas |")
    lines.append("|---|---|---|---|")

    agent_order = ["ProblemFormulator", "Writer", "Analyst"]
    for agent in agent_order:
        info = diagnosis.get(agent)
        if info:
            fixes = ", ".join(info["priority_fixes"]) if info["priority_fixes"] else "(OK)"
            lines.append(
                f"| {agent} | {info['avg']} | {info['dimensions']} | {fixes} |"
            )
    lines.append("")

    # (d) Recommended prompt changes
    lines.append("## (d) Recommended Prompt Changes")
    lines.append("")

    if recommendations:
        for agent, rec_text in recommendations.items():
            lines.append(f"### {agent}")
            lines.append("")
            lines.append(rec_text)
            lines.append("")
    else:
        all_ok = all(info["avg"] >= 6.0 for info in diagnosis.values())
        if all_ok:
            lines.append("All agents score ≥ 6.0 — no urgent prompt changes needed.")
        else:
            lines.append(
                "*(LLM unavailable — set ANTHROPIC_API_KEY for prompt recommendations)*"
            )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Ensure stdout can handle Unicode on Windows (cp1252 can't print arrows etc.)
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except (AttributeError, OSError):
            pass  # Python < 3.7 or non-reconfigurable stream

    parser = argparse.ArgumentParser(
        description="Aggregate LSAR reviews across EDM-ARS runs."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/",
        help="Directory to scan for run_* dirs (default: output/)",
    )
    parser.add_argument(
        "--min-runs",
        type=int,
        default=1,
        help="Minimum number of reviews required (default: 1)",
    )
    parser.add_argument(
        "--save-report",
        action="store_true",
        help="Save report to output/review_aggregate_report.md",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        print(f"Error: output directory '{output_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # 1. Discover reviews
    print(f"Scanning {output_dir} for LSAR reviews...")
    records = discover_reviews(output_dir)
    print(f"Found {len(records)} review(s).")

    if len(records) < args.min_runs:
        needed = args.min_runs - len(records)
        print(
            f"\nNeed {needed} more run(s) to meet --min-runs={args.min_runs}. "
            f"Run more pipeline executions first."
        )
        sys.exit(0)

    if not records:
        print("No LSAR reviews found. Nothing to aggregate.")
        sys.exit(0)

    # 2. Compute dimensional statistics
    stats = compute_dimension_stats(records)

    # 3. Extract weaknesses
    weaknesses = extract_weaknesses(records)
    print(f"Extracted {len(weaknesses)} weakness(es) across reviews.")

    # 4. Build LLM client (reads config.yaml for provider)
    llm_client: Any = None
    llm_model: str = "claude-sonnet-4-6"
    try:
        llm_client, llm_model = _load_llm_config()
        print(f"LLM provider ready (model: {llm_model})")
    except Exception as e:
        print(f"LLM unavailable ({e}) — skipping clustering & recommendations.")

    # 5. Cluster weaknesses (LLM, optional)
    themes: Optional[list[dict[str, Any]]] = None
    if len(records) >= 2 and len(weaknesses) > 3 and llm_client:
        print("Clustering weaknesses via LLM...")
        themes = cluster_weaknesses(weaknesses, llm_client, llm_model)
        if themes:
            print(f"Identified {len(themes)} theme(s).")
        else:
            print("Clustering returned no results; showing raw weaknesses.")
    elif not llm_client:
        print("LLM not available — skipping clustering.")

    # 6. Agent diagnosis
    diagnosis = build_agent_diagnosis(stats)

    # 7. Prompt recommendations (LLM, optional)
    recommendations: dict[str, str] = {}
    weak_agents = [a for a, info in diagnosis.items() if info["avg"] < 6.0]
    if weak_agents and llm_client:
        print(f"Generating prompt recommendations for: {', '.join(weak_agents)}...")
        recommendations = generate_prompt_recommendations(
            diagnosis, themes, llm_client, llm_model
        )
    elif weak_agents:
        print(f"Agents below 6.0: {', '.join(weak_agents)} (no LLM for recommendations)")

    # 8. Format and output report
    report = format_report(records, stats, weaknesses, themes, diagnosis, recommendations)

    print("\n" + "=" * 72)
    print(report)

    if args.save_report:
        report_path = output_dir / "review_aggregate_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
