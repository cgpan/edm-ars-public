"""Citation-quality defects found by adversarial verification of a shipped paper.

Three independent defects, all measured on the manuscript and pool in
``runs/arc_p_validation_20260711``:

1. **F-P5-BIB-NO-DOI** — ``build_bib_entry`` emitted author/title/year plus
   one venue field and dropped ``doi`` entirely: 0 of 62 bib entries carried
   a DOI although 61 of 62 source records had one.
2. **F-P5-VENUE-TYPE-MISLABEL** — the entry-type heuristic matched only five
   journal words, so 23 of 34 ``@inproceedings`` entries were really
   journals (PLoS ONE, Scientific Reports, BMC Public Health, npj Digital
   Medicine, Research in Higher Education, Clinical Cancer Research),
   rendering as "In: Proceedings of PLoS ONE."
3. **F-P5-DEPTH-RECENCY-SKEW** — depth was raised without constraining
   composition, so every one of the 62 references came from 2024-2026.

Everything here is offline and deterministic: no network, no LLM.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from src.citations import (
    AGE_BUCKETS,
    DEFAULT_REF_AGE_PROFILE,
    allocate_age_slots,
    build_bib_entry,
    build_bibtex,
    bucket_of_age,
    classify_entry,
    expand_literature_pool,
    normalize_doi,
    rank_pool,
    reconcile_citations,
    composition_age_profile,
    venue_citation_target,
)

ROOT = Path(__file__).resolve().parent.parent
SHIPPED_POOL = ROOT / "runs" / "arc_p_validation_20260711" / "output" / (
    "retrieved_literature.json"
)


def _paper(pid: str, year: int = 2025, title: str | None = None, **kw) -> dict:
    return {
        "paperId": pid,
        "title": title or f"Study {pid}",
        "authors": ["Author, A."],
        "year": year,
        "abstract": "",
        **kw,
    }


# ---------------------------------------------------------------------------
# (1) DOIs must reach the bibliography
# ---------------------------------------------------------------------------


class TestDoiReachesTheBibliography:
    """F-P5-BIB-NO-DOI: 0/62 shipped entries had a DOI; 61/62 records did."""

    def test_doi_is_emitted_when_the_record_has_one(self) -> None:
        entry = build_bib_entry(
            _paper("p1", venue="Journal of Learning Analytics", doi="10.18608/jla.2025.1")
        )
        assert "doi       = {10.18608/jla.2025.1}," in entry

    def test_doi_is_absent_when_the_record_has_none(self) -> None:
        assert "doi" not in build_bib_entry(_paper("p1", venue="Some Venue")).lower()

    @pytest.mark.parametrize(
        "raw",
        [
            "https://doi.org/10.1371/journal.pone.0289",
            "http://dx.doi.org/10.1371/journal.pone.0289",
            "doi:10.1371/journal.pone.0289",
            "  10.1371/journal.pone.0289  ",
        ],
    )
    def test_doi_is_normalised_to_the_bare_form(self, raw: str) -> None:
        entry = build_bib_entry(_paper("p1", venue="PLoS ONE", doi=raw))
        assert "doi       = {10.1371/journal.pone.0289}," in entry

    @pytest.mark.parametrize("junk", ["", None, "n/a", "unknown", "10", 0])
    def test_garbage_doi_is_dropped_not_emitted(self, junk: object) -> None:
        entry = build_bib_entry(_paper("p1", venue="V", doi=junk))
        assert "doi" not in entry.lower()

    def test_url_is_used_only_as_a_doi_substitute(self) -> None:
        with_url = build_bib_entry(
            _paper("p1", venue="V", url="https://example.org/paper")
        )
        assert "url       = {https://example.org/paper}," in with_url

        with_both = build_bib_entry(
            _paper("p2", venue="V", doi="10.1234/x", url="https://example.org/paper")
        )
        assert "doi       = {10.1234/x}," in with_both
        assert "url" not in with_both.lower(), "DOI wins; do not emit both"

    def test_url_is_never_invented(self) -> None:
        assert "url" not in build_bib_entry(_paper("p1", venue="V")).lower()

    def test_backfilled_entries_also_carry_the_doi(self) -> None:
        """Reconciliation appends entries through the same builder."""
        papers = [_paper("real1", venue="PLoS ONE", doi="10.1371/journal.pone.1")]
        tex = r"\begin{document}Prior work \cite{real1}.\end{document}"
        _, out_bib, stats = reconcile_citations(tex, "", papers)
        assert stats["backfilled"] == 1
        assert "10.1371/journal.pone.1" in out_bib

    def test_every_record_with_a_doi_yields_an_entry_with_a_doi(self) -> None:
        """The shipped-run counterfactual: 61/62 records had one, 0 shipped."""
        papers = [
            _paper(f"p{i}", venue="Journal of X", doi=f"10.1234/x.{i}")
            for i in range(20)
        ] + [_paper("nodoi", venue="Journal of X")]
        bib = build_bibtex(papers)
        assert bib.count("doi       = {") == 20


# ---------------------------------------------------------------------------
# (2) Journals must not be labelled conference proceedings
# ---------------------------------------------------------------------------


class TestEntryTypeClassification:
    """F-P5-VENUE-TYPE-MISLABEL: 23 of 34 @inproceedings were journals."""

    SHIPPED_MISLABELS = [
        ("PLoS ONE", "10.1371/journal.pone.0289"),
        ("Scientific Reports", "10.1038/s41598-023-00001"),
        ("BMC Public Health", "10.1186/s12889-023-00001"),
        ("npj Digital Medicine", "10.1038/s41746-023-00001"),
        ("Research in Higher Education", "10.1007/s11162-023-00001"),
        ("Clinical Cancer Research", "10.1158/1078-0432.CCR-23-0001"),
    ]

    @pytest.mark.parametrize("venue,doi", SHIPPED_MISLABELS)
    def test_shipped_mislabelled_journals_become_articles(
        self, venue: str, doi: str
    ) -> None:
        entry = build_bib_entry(_paper("p1", venue=venue, doi=doi))
        assert "@article{p1" in entry, f"{venue} is a journal, not proceedings"
        assert f"journal = {{{venue}}}" in entry
        assert "booktitle" not in entry

    @pytest.mark.parametrize("venue,_doi", SHIPPED_MISLABELS)
    def test_no_journal_is_ever_rendered_as_proceedings(
        self, venue: str, _doi: str
    ) -> None:
        """Without a DOI the honest answer is @misc — never @inproceedings."""
        entry = build_bib_entry(_paper("p1", venue=venue))
        assert "@inproceedings" not in entry
        assert "booktitle" not in entry
        assert "@misc{p1" in entry
        assert venue in entry, "the venue is still reported, just not as a proceedings"

    def test_explicit_publication_type_beats_venue_keywords(self) -> None:
        entry = build_bib_entry(
            _paper("p1", venue="Frontiers in Education", publicationTypes=["JournalArticle"])
        )
        assert "@article{p1" in entry

    def test_conference_publication_type_is_honoured(self) -> None:
        entry = build_bib_entry(
            _paper("p1", venue="AIED", publicationTypes=["Conference"])
        )
        assert "@inproceedings{p1" in entry
        assert "booktitle = {AIED}" in entry

    def test_both_types_are_broken_by_the_venue_string(self) -> None:
        types = ["JournalArticle", "Conference"]
        journalish = build_bib_entry(
            _paper("p1", venue="Journal of Educational Data Mining", publicationTypes=types)
        )
        conferencish = build_bib_entry(
            _paper("p2", venue="Proceedings of the 17th EDM Conference", publicationTypes=types)
        )
        assert "@article{p1" in journalish
        assert "@inproceedings{p2" in conferencish

    def test_book_types_do_not_become_proceedings(self) -> None:
        entry = build_bib_entry(
            _paper("p1", venue="Handbook of Learning Analytics",
                   publicationTypes=["BookSection"])
        )
        assert "@inproceedings" not in entry
        assert "@misc{p1" in entry

    @pytest.mark.parametrize(
        "venue",
        [
            "Proceedings of the 17th International Conference on EDM",
            "Proc. of the ACM Conference on Learning at Scale",
            "International Workshop on Learning Analytics",
            "IEEE Symposium on Visual Analytics",
            "Annual Meeting of the American Educational Research Association",
        ],
    )
    def test_conference_venue_wording_is_recognised(self, venue: str) -> None:
        assert "@inproceedings{p1" in build_bib_entry(_paper("p1", venue=venue))

    @pytest.mark.parametrize(
        "venue",
        [
            "Journal of Learning Analytics",
            "IEEE Transactions on Learning Technologies",
            "Review of Educational Research",
            "Educational Evaluation and Policy Analysis Quarterly",
            "Annals of Applied Statistics",
        ],
    )
    def test_journal_venue_wording_is_recognised(self, venue: str) -> None:
        assert "@article{p1" in build_bib_entry(_paper("p1", venue=venue))

    def test_ambiguous_doi_registrant_does_not_force_a_guess(self) -> None:
        """ACM and IEEE register BOTH journals and proceedings."""
        for doi in ("10.1145/3576050.3576100", "10.1109/TLT.2023.1234567"):
            entry = build_bib_entry(_paper("p1", venue="Some Outlet", doi=doi))
            assert "@misc{p1" in entry, doi
            assert "@inproceedings" not in entry

    def test_springer_journal_and_proceedings_dois_are_separated(self) -> None:
        journal = build_bib_entry(
            _paper("p1", venue="Instructional Science", doi="10.1007/s11251-023-09999")
        )
        lncs = build_bib_entry(
            _paper("p2", venue="Artificial Intelligence in Education",
                   doi="10.1007/978-3-031-11644-5_1")
        )
        assert "@article{p1" in journal
        assert "@misc{p2" in lncs, "ISBN-derived Springer DOIs are books/proceedings"

    def test_arxiv_records_stay_preprints(self) -> None:
        entry = build_bib_entry(_paper("arxiv:2401.12345", doi="10.48550/arXiv.2401.12345"))
        assert "@misc{arxiv_2401.12345" in entry
        assert "arXiv preprint" in entry
        assert "doi       = {10.48550/arXiv.2401.12345}," in entry

    def test_absent_venue_is_still_declared_not_invented(self) -> None:
        entry = build_bib_entry(_paper("p1"))
        assert "@misc{p1" in entry
        assert "Venue metadata unavailable" in entry

    def test_classify_entry_never_raises_on_junk(self) -> None:
        for junk in (
            {},
            {"paperId": None, "venue": None},
            {"venue": 12345},
            {"venue": "V", "publicationTypes": None},
            {"venue": "V", "publicationTypes": [None, ""]},
            {"venue": "V", "doi": 3.14},
        ):
            entry_type, key, _value = classify_entry(junk)  # type: ignore[arg-type]
            assert entry_type in {"misc", "article", "inproceedings"}
            assert key in {"note", "journal", "booktitle"}

    def test_normalize_doi_rejects_non_dois(self) -> None:
        assert normalize_doi("not a doi") == ""
        assert normalize_doi(None) == ""
        assert normalize_doi("10.1234/abc") == "10.1234/abc"


# ---------------------------------------------------------------------------
# (3) Recency-aware composition
# ---------------------------------------------------------------------------


class TestSlotAllocation:
    @pytest.mark.parametrize("venue", ["EDM", "JEDM", "JLA", None, "NOPE"])
    @pytest.mark.parametrize("n", [0, 1, 13, 34, 55, 62, 65, 100])
    def test_slots_sum_to_target(self, venue: str | None, n: int) -> None:
        slots = allocate_age_slots(n, composition_age_profile(venue))
        assert sum(slots.values()) == n
        assert set(slots) == set(AGE_BUCKETS)
        assert all(v >= 0 for v in slots.values())

    @pytest.mark.parametrize(
        "venue,n,expected",
        [
            ("EDM", 34, [8, 8, 8, 6, 4]),
            ("JEDM", 62, [19, 15, 12, 9, 7]),
            ("JLA", 65, [11, 16, 20, 12, 6]),
            (None, 55, [14, 13, 13, 9, 6]),
        ],
    )
    def test_slots_match_the_published_profile(
        self, venue: str | None, n: int, expected: list[int]
    ) -> None:
        """Pins the citation-recency specification (internal) §1.5 exactly."""
        slots = allocate_age_slots(n, composition_age_profile(venue))
        assert [slots[b] for b in AGE_BUCKETS] == expected

    def test_remainder_ties_favour_the_older_bucket(self) -> None:
        uniform = {b: 0.2 for b in AGE_BUCKETS}
        slots = allocate_age_slots(12, uniform)
        assert [slots[b] for b in AGE_BUCKETS] == [2, 2, 2, 3, 3]

    def test_unknown_venue_uses_the_pooled_default_profile(self) -> None:
        assert composition_age_profile("NOPE") == DEFAULT_REF_AGE_PROFILE
        assert composition_age_profile(None) == DEFAULT_REF_AGE_PROFILE

    def test_mined_norms_override_the_pinned_constants(self) -> None:
        norms = {"EDM": {"ref_age": {"buckets": {
            "le2": 0.1, "3_5": 0.1, "6_10": 0.1, "11_20": 0.1, "gt20": 0.6,
        }}}}
        assert composition_age_profile("EDM", norms)["gt20"] == 0.6
        # A malformed/partial block must not crash or half-apply.
        assert composition_age_profile("EDM", {"EDM": {"ref_age": {"buckets": {"le2": 1.0}}}}) \
            == composition_age_profile("EDM")

    def test_negative_target_is_not_a_crash(self) -> None:
        assert allocate_age_slots(-5, DEFAULT_REF_AGE_PROFILE) == {
            b: 0 for b in AGE_BUCKETS
        }

    @pytest.mark.parametrize(
        "age,bucket",
        [(0, "le2"), (2, "le2"), (3, "3_5"), (5, "3_5"), (6, "6_10"),
         (10, "6_10"), (11, "11_20"), (20, "11_20"), (21, "gt20"), (99, "gt20")],
    )
    def test_bucket_boundaries(self, age: int, bucket: str) -> None:
        assert bucket_of_age(age) == bucket


class TestProducerAndCheckerShareTheBins:
    """The composer and the linter must not disagree about what "old" means."""

    def test_bins_come_from_the_linter(self) -> None:
        import src.citations as citations
        import src.manuscript_linter as linter

        for name in ("AGE_BUCKETS", "FILL_ORDER", "DEFAULT_REF_AGE_PROFILE"):
            if hasattr(linter, name):
                assert getattr(citations, name) == getattr(linter, name), name
        if hasattr(linter, "bucket_of_age"):
            assert all(
                citations.bucket_of_age(a) == linter.bucket_of_age(a)
                for a in range(0, 60)
            )

    def test_per_venue_targets_stay_inside_the_linter_tolerance(self) -> None:
        """Until venue_norms.yaml carries ``ref_age`` the two differ slightly.

        The producer aims at the measured per-venue profile (docs §1.1)
        while the linter checks against the pooled default; every gap must
        stay inside the linter's tolerance or the composer would steer the
        manuscript straight into a warning.
        """
        linter = pytest.importorskip("src.manuscript_linter")
        tolerance = getattr(linter, "DEFAULT_REF_AGE_TOLERANCE_PP", 12.0) / 100.0
        for venue in ("EDM", "JEDM", "JLA"):
            profile = composition_age_profile(venue)
            for bucket in AGE_BUCKETS:
                delta = abs(profile[bucket] - DEFAULT_REF_AGE_PROFILE[bucket])
                assert delta <= tolerance, f"{venue}/{bucket} off by {delta:.3f}"


class TestRecencyAwareComposition:
    NOW = 2026

    def _pool(self, years: dict[int, int], **kw) -> list[dict]:
        """``{year: count}`` -> pool records, retrieval rank in insertion order.

        Titles must differ by a token of 3+ characters: ``_title_tokens``
        drops shorter ones, so ``"... number 7"`` and ``"... number 8"``
        are indistinguishable to the Jaccard dedup.
        """
        pool: list[dict] = []
        for year, count in years.items():
            for i in range(count):
                pool.append(_paper(
                    f"y{year}_{i}",
                    year=year,
                    title=f"Distinct inquiry {year} number{i:03d}",
                    venue="Journal of Learning Analytics",
                    retrieval_rank=len(pool),
                    **kw,
                ))
        return pool

    def _ages(self, out: list[dict]) -> list[int]:
        return [self.NOW - int(p["year"]) for p in out]

    def test_selected_papers_are_never_dropped_or_reordered(self) -> None:
        selected = [_paper(f"s{i}", year=2025) for i in range(10)]
        pool = self._pool({2026: 20, 2010: 20, 1995: 20})
        out = expand_literature_pool(
            selected, pool, target=34,
            profile=composition_age_profile("EDM"), now_year=self.NOW,
        )
        assert [p["paperId"] for p in out[:10]] == [p["paperId"] for p in selected]
        assert len(out) == 34

    def test_never_shrinks_the_selection(self) -> None:
        selected = [_paper(f"s{i}") for i in range(20)]
        out = expand_literature_pool(
            selected, [], target=5,
            profile=composition_age_profile("EDM"), now_year=self.NOW,
        )
        assert len(out) == 20

    def test_old_papers_are_preferred_when_buckets_demand_them(self) -> None:
        """Top-relevance new work must not crowd out the historical tail."""
        pool = self._pool({2026: 50})          # ranks 0-49, best relevance
        pool += self._pool({2008: 10})         # ranks 0-9 but appended last
        for i, paper in enumerate(pool):
            paper["retrieval_rank"] = i        # oldest get the WORST ranks
        out = expand_literature_pool(
            [], pool, target=20,
            profile=composition_age_profile("EDM"), now_year=self.NOW,
        )
        n_old = sum(1 for age in self._ages(out) if age >= 11)
        assert n_old >= 3, f"only {n_old} old records survived ranking"

    def test_not_all_references_come_from_the_last_two_years(self) -> None:
        """Pinned regression for F-P5-DEPTH-RECENCY-SKEW.

        The shipped bibliography was 62/62 from the last two years while the
        count metric read green.
        """
        pool = self._pool({y: 4 for y in range(2001, 2027)})
        out = expand_literature_pool(
            [], pool, target=34,
            profile=composition_age_profile("EDM"), now_year=self.NOW,
        )
        ages = self._ages(out)
        assert len(out) == 34
        assert sum(1 for a in ages if a <= 2) / len(ages) <= 0.55
        assert sum(1 for a in ages if a > 10) >= 2
        assert sum(1 for a in ages if a > 15) >= 1

    def test_bucket_deficit_spills_to_the_next_oldest_not_the_newest(self) -> None:
        pool = self._pool({2026: 20, 2011: 20})    # nothing older than 20 y
        out = expand_literature_pool(
            [], pool, target=34,
            profile=composition_age_profile("EDM"), now_year=self.NOW,
        )
        ages = self._ages(out)
        # EDM/34 gives 11_20 six slots and gt20 four; with gt20 empty the
        # four spilled slots must land on 11_20, not on le2.
        assert sum(1 for a in ages if a == 15) >= 10
        assert len(out) == 34

    def test_offtopic_records_are_not_promoted(self) -> None:
        clinical = [
            _paper("onc", year=2010, venue="Journal of Clinical Oncology",
                   title="Adjuvant therapy trial outcomes"),
            _paper("lancet", year=2005, venue="The Lancet",
                   title="Cardiovascular mortality cohort"),
            _paper("med", year=2001, venue="Some Outlet",
                   fieldsOfStudy=["Medicine"], title="Transplant survival study"),
        ]
        pool = clinical + self._pool({2026: 10})
        out = expand_literature_pool(
            [], pool, target=8,
            profile=composition_age_profile("EDM"), now_year=self.NOW,
        )
        ids = {p["paperId"] for p in out}
        assert ids.isdisjoint({"onc", "lancet", "med"}), (
            "clinical-medicine records must not be promoted even though they "
            "are the only old work in the pool"
        )

    def test_a_selected_offtopic_paper_is_still_kept(self) -> None:
        """Rejection applies to the append path only; selection is the LLM's."""
        selected = [_paper("onc", venue="Journal of Clinical Oncology")]
        out = expand_literature_pool(
            selected, self._pool({2026: 5}), target=4,
            profile=composition_age_profile("EDM"), now_year=self.NOW,
        )
        assert out[0]["paperId"] == "onc"

    def test_arxiv_records_get_neutral_influence_not_zero(self) -> None:
        pool = [
            _paper(f"s2_{i}", year=2025, venue="Journal of X", retrieval_rank=i + 1,
                   citationCount=10 * (i + 1), title=f"Cited work number{i:03d}")
            for i in range(10)
        ]
        pool.append(_paper("arxiv:1", year=2025, retrieval_rank=0,
                           title="Fresh preprint on learning analytics"))
        scored, degraded = rank_pool(pool, self.NOW)
        arxiv = next(p for p in scored if p["paperId"] == "arxiv:1")
        assert arxiv["_infl"] > 0.0, "a 0 prior ranks every preprint last"
        assert "influentialCitationCount" not in degraded
        out = expand_literature_pool(
            [], pool, target=3,
            profile=composition_age_profile("EDM"), now_year=self.NOW,
        )
        assert "arxiv:1" in {p["paperId"] for p in out}

    def test_records_without_year_or_title_are_never_appended(self) -> None:
        pool = [
            {"paperId": "a", "title": "", "year": 2020},
            {"paperId": "b", "title": "Fine", "year": None},
            {"paperId": "c", "title": "Good one", "year": 2021},
        ]
        out = expand_literature_pool(
            [], pool, target=5,
            profile=composition_age_profile("EDM"), now_year=self.NOW,
        )
        assert [p["paperId"] for p in out] == ["c"]

    def test_dedups_by_id_then_by_title(self) -> None:
        selected = [_paper("s1", title="Predicting dropout with machine learning")]
        pool = [
            _paper("s1", year=2024),
            _paper("x9", year=2020, title="Predicting dropout with machine learning"),
            _paper("x10", year=2005, title="A completely different investigation"),
        ]
        out = expand_literature_pool(
            selected, pool, target=10,
            profile=composition_age_profile("EDM"), now_year=self.NOW,
        )
        assert [p["paperId"] for p in out] == ["s1", "x10"]

    def test_pool_metadata_still_wins_over_the_llm_echo(self) -> None:
        echoed = {"paperId": "s1", "title": "Predicting dropout",
                  "authors": ["Author, A."], "year": 2023}
        authoritative = _paper("s1", year=2023, title="Predicting dropout",
                               venue="Journal of Learning Analytics",
                               doi="10.1234/jla.2023.1")
        out = expand_literature_pool(
            [echoed], [authoritative], target=3,
            profile=composition_age_profile("EDM"), now_year=self.NOW,
        )
        assert out[0] is authoritative
        assert "@article{s1" in build_bib_entry(out[0])

    def test_appended_records_are_the_untouched_pool_objects(self) -> None:
        """Scoring must not leak ``_score``/``_bucket`` into the artifacts."""
        pool = self._pool({2026: 5})
        out = expand_literature_pool(
            [], pool, target=3,
            profile=composition_age_profile("EDM"), now_year=self.NOW,
        )
        for paper in out:
            assert paper in pool
            assert not any(k.startswith("_") for k in paper)
        json.dumps(out)  # serialisable, as the orchestrator persists it

    def test_deterministic_under_input_permutation(self) -> None:
        pool = self._pool({y: 4 for y in range(2001, 2027)})
        first = expand_literature_pool(
            [], pool, target=34,
            profile=composition_age_profile("EDM"), now_year=self.NOW,
        )
        shuffled = list(pool)
        random.Random(42).shuffle(shuffled)
        second = expand_literature_pool(
            [], shuffled, target=34,
            profile=composition_age_profile("EDM"), now_year=self.NOW,
        )
        assert [p["paperId"] for p in first] == [p["paperId"] for p in second]

    def test_empty_and_degenerate_inputs_do_not_crash(self) -> None:
        profile = composition_age_profile("EDM")
        assert expand_literature_pool(None, None, 10, profile=profile) == []
        assert expand_literature_pool([], [], 0, profile=profile) == []
        assert expand_literature_pool([], [], -3, profile=profile) == []
        assert len(expand_literature_pool(
            [], self._pool({2026: 3}), 100, profile=profile, now_year=self.NOW
        )) == 3

    def test_string_years_are_tolerated(self) -> None:
        pool = [_paper("a", year="2005", title="Old string year",
                       venue="Journal of X")]
        out = expand_literature_pool(
            [], pool, target=2,
            profile=composition_age_profile("EDM"), now_year=self.NOW,
        )
        assert [p["paperId"] for p in out] == ["a"]


class TestLegacyBehaviourIsPreserved:
    def test_profile_none_reproduces_the_legacy_order(self) -> None:
        """``profile=None`` must be byte-identical legacy behaviour.

        This documents the defect as much as it guards the opt-out: the
        legacy path appends in pool order, which for a year-desc pool is
        100% ``le2``.
        """
        pool = [
            _paper(f"p{i}", year=2026 - (i // 10), title=f"Work number{i:03d} on outcomes")
            for i in range(30)
        ]
        out = expand_literature_pool([], pool, target=12)
        assert [p["paperId"] for p in out] == [f"p{i}" for i in range(12)]
        assert all(p["year"] >= 2025 for p in out)

    def test_legacy_path_reports_no_degraded_signals(self) -> None:
        stats: dict = {}
        expand_literature_pool([], [_paper("a")], target=1, stats=stats)
        assert stats["degraded_signals"] == []
        assert stats["n_appended"] == 1


class TestDegradesOnLegacyPools:
    """The persisted pools carry 7 keys: no counts, no ranks, no types."""

    NOW = 2026

    def _legacy_pool(self, n: int = 100) -> list[dict]:
        return [
            {
                "paperId": f"legacy{i}",
                "title": f"Legacy retrieved study number{i:03d}",
                "authors": ["Author, A."],
                "year": 2024 + (i % 3),
                "abstract": "",
                "venue": "Journal of Postsecondary Student Success",
                "doi": f"10.1234/x.{i}",
            }
            for i in range(n)
        ]

    def test_no_exception_and_target_is_still_reached(self) -> None:
        stats: dict = {}
        out = expand_literature_pool(
            [], self._legacy_pool(), target=62,
            profile=composition_age_profile("JEDM"), now_year=self.NOW, stats=stats,
        )
        assert len(out) == 62
        assert stats["n_appended"] == 62

    def test_degraded_signals_are_named(self) -> None:
        stats: dict = {}
        expand_literature_pool(
            [], self._legacy_pool(), target=62,
            profile=composition_age_profile("JEDM"), now_year=self.NOW, stats=stats,
        )
        assert set(stats["degraded_signals"]) == {
            "retrieval_rank", "influentialCitationCount",
        }

    def test_repeated_calls_agree(self) -> None:
        pool = self._legacy_pool()
        kwargs = dict(profile=composition_age_profile("JEDM"), now_year=self.NOW)
        a = expand_literature_pool([], pool, 62, **kwargs)  # type: ignore[arg-type]
        b = expand_literature_pool([], pool, 62, **kwargs)  # type: ignore[arg-type]
        assert [p["paperId"] for p in a] == [p["paperId"] for p in b]

    def test_stats_expose_the_composition_not_just_the_count(self) -> None:
        stats: dict = {}
        expand_literature_pool(
            [], self._legacy_pool(), target=62,
            profile=composition_age_profile("JEDM"), now_year=self.NOW, stats=stats,
        )
        assert stats["achieved_buckets"]["le2"] == 62
        assert stats["target_slots"]["gt20"] == 7
        assert stats["pool_year_min"] == 2024 and stats["pool_year_max"] == 2026

    @pytest.mark.skipif(not SHIPPED_POOL.exists(), reason="shipped run not present")
    def test_the_real_shipped_pool_still_composes(self) -> None:
        pool = json.loads(SHIPPED_POOL.read_text(encoding="utf-8"))["papers"]
        target = venue_citation_target("JEDM") or 62
        stats: dict = {}
        out = expand_literature_pool(
            pool[:10], pool, target,
            profile=composition_age_profile("JEDM"), now_year=self.NOW, stats=stats,
        )
        assert len(out) == target
        assert stats["n_offtopic_rejected"] > 0, (
            "the six clinical-oncology records that shipped must be refused"
        )
        # Nothing old exists in this pool: the honest outcome is a full list
        # that is still all-recent, plus a loud degradation record.
        assert stats["pool_year_min"] == 2024
        assert set(stats["degraded_signals"]) == {
            "retrieval_rank", "influentialCitationCount",
        }
