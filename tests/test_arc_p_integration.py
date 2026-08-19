"""Arc P residuals — cross-file integration pins.

Six agents implemented the residuals in parallel, each owning disjoint
files. That partition left `src/orchestrator.py` unowned, so the recency
COMPOSER shipped dead while the linter that DEMANDS old references
shipped live — every gated run would have failed on a defect the pipeline
had no mechanism to fix. These tests pin the seams between the pieces,
which is exactly where file-partitioned work breaks.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _mk(pid: str, year: int, rank: int, cites: int, title: str) -> dict:
    return {
        "paperId": pid, "title": title, "year": year, "authors": ["Author, A."],
        "retrieval_rank": rank, "citationCount": cites,
    }


_WORDS = (
    "knowledge tracing cognitive diagnosis item response theory student modeling "
    "skill mastery learning analytics dashboard affect detection regulated "
    "curriculum sequencing peer assessment bayesian network deep transformer "
    "attention memory forgetting spacing retrieval practice feedback timing hints "
    "scaffolding motivation engagement dropout persistence achievement equity "
    "fairness calibration reliability validity invariance"
).split()


def _distinct_titles(n: int, seed: int = 7) -> list[str]:
    """Titles with disjoint-enough token sets to survive Jaccard dedup.

    Learned the hard way: titles like "Study 1"/"Study 2" tokenize to the
    SAME set (short tokens are dropped), so a naive fixture silently
    deduplicates to one paper and every assertion below passes vacuously.
    """
    import random

    rng = random.Random(seed)
    return [" ".join(rng.sample(_WORDS, 8)) for _ in range(n)]


class TestRecencyComposerIsWired:
    """The composer must be reachable from production, not just importable."""

    def test_orchestrator_passes_a_profile(self, tmp_path: Path) -> None:
        import src.orchestrator as orch

        captured: dict = {}
        real = orch_expand = None
        from src import citations

        real = citations.expand_literature_pool

        def spy(selected, pool, target, **kwargs):
            captured.update(kwargs)
            captured["target"] = target
            return real(selected, pool, target, **kwargs)

        o = object.__new__(orch.Orchestrator)
        o.config = {"review_gate": {"venue": "JEDM"}}
        o._log = lambda *a, **k: None

        class Ctx:
            output_dir = str(tmp_path)
            literature_context = {"papers": [
                _mk(f"s{i}", 2025, i, 10, t)
                for i, t in enumerate(_distinct_titles(12))
            ]}
            retrieved_literature = {"papers": [
                _mk(f"p{i}", 2025 if i < 60 else 2006, i, 50, t)
                for i, t in enumerate(_distinct_titles(80, seed=11))
            ]}
        o.ctx = Ctx()

        citations.expand_literature_pool = spy
        try:
            o._expand_literature_for_depth()
        finally:
            citations.expand_literature_pool = real

        assert "profile" in captured and captured["profile"], (
            "orchestrator must pass profile= or the composer is dead code and "
            "the linter's recency ERRORs become unfixable"
        )
        assert "now_year" in captured
        assert captured.get("stats") is not None

    def test_profile_changes_the_composition(self) -> None:
        """Guards against a profile that is passed but inert."""
        from src.citations import composition_age_profile, expand_literature_pool

        titles = _distinct_titles(112, seed=3)
        sel = [_mk(f"s{i}", 2025, i, 10, titles[i]) for i in range(12)]
        pool = (
            [_mk(f"n{i}", 2025, i, 5, titles[12 + i]) for i in range(80)]
            + [_mk(f"o{i}", 2004 + (i % 12), 100 + i, 900, titles[92 + i])
               for i in range(20)]
        )

        def older_than(papers: list[dict], years: int) -> int:
            return sum(
                1 for p in papers
                if p.get("year") and 2026 - int(p["year"]) > years
            )

        flat = expand_literature_pool(sel, pool, 62, now_year=2026)
        composed = expand_literature_pool(
            sel, pool, 62, profile=composition_age_profile("JEDM"), now_year=2026
        )
        assert len(flat) == len(composed) == 62
        assert older_than(flat, 10) == 0, "fixture must reproduce the defect"
        assert older_than(composed, 10) >= 10, (
            "composition must actually place historical work in the bibliography"
        )
        assert older_than(composed, 15) >= 1, "the >15y floor must be reachable"

    def test_degrades_when_pool_has_no_old_work(self) -> None:
        """The real pools we already have are entirely 2024-2026."""
        from src.citations import composition_age_profile, expand_literature_pool

        titles = _distinct_titles(70, seed=5)
        sel = [_mk(f"s{i}", 2025, i, 10, titles[i]) for i in range(10)]
        pool = [_mk(f"n{i}", 2025, i, 5, titles[10 + i]) for i in range(60)]
        got = expand_literature_pool(
            sel, pool, 40, profile=composition_age_profile("JEDM"), now_year=2026
        )
        assert len(got) == 40, "must still fill the target, not crash or stall"


class TestRankPoolScoring:
    def test_on_topic_bonus_is_a_tiebreaker_not_a_dominator(self) -> None:
        """`+ topic` instead of `* topic` made a 0.25 bonus outrank relevance.

        _topicality returns 0.25 and its docstring calls it "a small
        on-topic bonus"; unweighted it consumed ~72% of the relevance
        range, letting a mediocre paper with an on-topic venue leapfrog a
        highly relevant one.
        """
        from src.citations import rank_pool

        top_relevance_offtopic = _mk("a", 2024, 0, 0, "alpha beta gamma delta")
        weak_relevance_ontopic = dict(
            _mk("b", 2024, 400, 0, "epsilon zeta eta theta"),
            venue="Journal of Educational Data Mining",
        )
        scored, _ = rank_pool([top_relevance_offtopic, weak_relevance_ontopic], 2026)
        by_id = {s["paperId"]: s["_score"] for s in scored}
        assert by_id["a"] > by_id["b"], (
            "a rank-0 record must outscore a rank-400 record; the on-topic "
            f"bonus is dominating (scores: {by_id})"
        )
        # and the bonus is bounded by its weight
        assert max(by_id.values()) <= 0.55 + 0.30 + 0.15 + 1e-9


class TestResultsJsonIsStrictJson:
    def test_nan_cannot_reach_results_json(self, tmp_path: Path) -> None:
        """NaN is not valid JSON; strict parsers reject what Python emits."""
        payload = {"measurement_results": {"P1_ctt": {"alpha": math.nan}}}
        with pytest.raises(ValueError):
            json.dumps(payload, allow_nan=False)

    def test_analyst_writes_with_allow_nan_false(self) -> None:
        src = (ROOT / "src" / "agents" / "analyst.py").read_text(encoding="utf-8")
        assert "allow_nan=False" in src, (
            "analyst must not serialize NaN into results.json — a sparse-matrix "
            "CTT run produced 47 NaN item stats that flowed to the Critic"
        )


class TestRefAgeContract:
    """`ref_age.buckets` is a three-way contract.

    The miner (scripts/mine_venue_norms.py) WRITES it; the composer
    (src/citations.py composition_age_profile) and the checker
    (src/manuscript_linter.py venue_age_profile) both READ it. The miner
    first emitted `ref_age.profile`, so both consumers silently fell back
    to the pooled default and every venue profile went unused — a failure
    with no error message anywhere.
    """

    def test_all_three_venues_use_mined_profiles(self) -> None:
        from src.citations import composition_age_profile
        from src.manuscript_linter import load_venue_norms, venue_age_profile

        norms = load_venue_norms()
        for venue in ("EDM", "JEDM", "JLA"):
            profile, tolerance, source = venue_age_profile(venue, norms)
            assert source == venue, (
                f"{venue} fell back to '{source}' — venue_norms.yaml is "
                "missing ref_age.buckets or the key was renamed"
            )
            assert composition_age_profile(venue, norms) == profile, (
                "producer and checker must aim at the SAME distribution"
            )
            assert 0 < tolerance <= 20

    def test_profiles_sum_to_one_and_are_plausible(self) -> None:
        from src.manuscript_linter import load_venue_norms

        norms = load_venue_norms()
        for venue, block in norms.items():
            buckets = (block.get("ref_age") or {}).get("buckets")
            if not buckets:
                continue
            assert abs(sum(buckets.values()) - 1.0) < 0.02, venue
            older = buckets["11_20"] + buckets["gt20"]
            assert older > 0.15, (
                f"{venue}: real papers cite old work; a profile with "
                f"only {older:.0%} older than 10 years is not credible"
            )

    def test_yaml_bin_keys_are_strings_not_ints(self) -> None:
        """YAML 1.1 parses bare 3_5 as the integer 35 (digit separator)."""
        import yaml

        raw = (ROOT / "data_registry" / "venue_norms.yaml").read_text(encoding="utf-8")
        data = yaml.safe_load(raw)
        for block in data["venues"].values():
            buckets = (block.get("ref_age") or {}).get("buckets") or {}
            for key in buckets:
                assert isinstance(key, str), (
                    f"bin key {key!r} parsed as {type(key).__name__}; quote it"
                )
