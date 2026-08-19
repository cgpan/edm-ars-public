"""Arc P residuals G1 (checker half) + G5 — reference-age distribution.

G1: F-P5-DEPTH-RECENCY-SKEW. The Arc P validation run shipped a paper
whose entire 62-entry bibliography was dated 2024-2026, while the prose
attributed DINA to Junker & Sijtsma and de la Torre with no citation at
all. Every count-shaped metric read green ("62 of 62 available
references"). A count is not a distribution; these tests pin the
distribution check that would have caught it.

G5: ``many-uncited-bib-entries`` fired on every run once references.bib
became a deliberate superset for the reviser to draw from. BibTeX only
typesets entries the document cites, so those entries never reach the
PDF. The check is now scoped to ``\\nocite{*}``, the one construct that
does render them.

Offline: pure string/file inspection, no network, no LLM.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import pytest

from src.manuscript_linter import (
    AGE_BUCKETS,
    DEFAULT_REF_AGE_PROFILE,
    FILL_ORDER,
    LintReport,
    bib_entry_years,
    bucket_of_age,
    lint_manuscript,
    venue_age_profile,
)

ROOT = Path(__file__).resolve().parent.parent
NOW = 2026

#: Codes owned by this feature; used to assert "nothing spurious fired".
RECENCY_CODES = {
    "reference-recency-collapse",
    "no-foundational-references",
    "reference-recency-skew",
    "thin-historical-tail",
    "bib-recency-collapse",
}


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

ABSTRACT = r"\abstract{" + "word " * 40 + "}"


def _bib(years: Sequence[Optional[int]], prefix: str = "ref") -> str:
    entries = []
    for i, year in enumerate(years):
        year_field = "" if year is None else f",\n  year = {{{year}}}"
        entries.append(
            f"@article{{{prefix}{i},\n  author = {{Author, A.}},\n"
            f"  title = {{Study number {i}}}{year_field}\n}}"
        )
    return "\n\n".join(entries) + "\n"


def _tex(keys: Sequence[str], extra: str = "") -> str:
    cites = " ".join(rf"\parencite{{{k}}}" for k in keys)
    return (
        "\\documentclass{article}\n"
        r"\title{A Study of Something}" + "\n" + ABSTRACT + "\n"
        "\\begin{document}\n\\section{Related Work}\n"
        f"{cites}\n{extra}\n"
        "\\bibliography{references}\n\\end{document}\n"
    )


def _run(
    tmp_path: Path,
    years: Sequence[Optional[int]],
    n_cited: Optional[int] = None,
    venue: Optional[str] = None,
    norms_path: Optional[Path] = None,
    extra_tex: str = "",
    now_year: int = NOW,
) -> LintReport:
    """Lint a synthetic run whose bib holds ``years`` (cited first)."""
    keys = [f"ref{i}" for i in range(len(years))]
    cited = keys if n_cited is None else keys[:n_cited]
    (tmp_path / "paper.tex").write_text(_tex(cited, extra_tex), encoding="utf-8")
    (tmp_path / "references.bib").write_text(_bib(years), encoding="utf-8")
    return lint_manuscript(
        tmp_path, venue=venue, norms_path=norms_path,
        write_json=False, now_year=now_year,
    )


def _codes(report: LintReport) -> list[str]:
    return [d.code for d in report.defects]


def _profile_years(slots: dict[str, int], now_year: int = NOW) -> list[int]:
    """Concrete years hitting a per-bucket slot allocation exactly."""
    exemplar = {"le2": 1, "3_5": 4, "6_10": 8, "11_20": 16, "gt20": 26}
    out: list[int] = []
    for bucket in AGE_BUCKETS:
        out += [now_year - exemplar[bucket]] * slots[bucket]
    return out


# EDM at N=34 from the spec's largest-remainder table (§1.5).
EDM_34_SLOTS = {"le2": 8, "3_5": 8, "6_10": 8, "11_20": 6, "gt20": 4}


# ---------------------------------------------------------------------------
# Bucket + parsing primitives
# ---------------------------------------------------------------------------


class TestBuckets:
    @pytest.mark.parametrize(
        "age,bucket",
        [(0, "le2"), (2, "le2"), (3, "3_5"), (5, "3_5"), (6, "6_10"),
         (10, "6_10"), (11, "11_20"), (20, "11_20"), (21, "gt20"), (99, "gt20")],
    )
    def test_boundaries(self, age: int, bucket: str) -> None:
        assert bucket_of_age(age) == bucket

    def test_profile_is_a_distribution(self) -> None:
        assert set(DEFAULT_REF_AGE_PROFILE) == set(AGE_BUCKETS)
        assert sum(DEFAULT_REF_AGE_PROFILE.values()) == pytest.approx(1.0, abs=0.01)

    def test_fill_order_is_oldest_first(self) -> None:
        # The composition side spills scarce old slots first; a reversal
        # would silently re-skew a tight budget back to `le2`.
        assert FILL_ORDER == tuple(reversed(AGE_BUCKETS))


class TestBibYearParsing:
    def test_braced_quoted_and_bare_years(self) -> None:
        bib = (
            '@article{a, year = {2011}}\n'
            '@inproceedings{b, year = "1983"}\n'
            "@misc{c, year = 2001}\n"
        )
        assert bib_entry_years(bib) == {"a": 2011, "b": 1983, "c": 2001}

    def test_missing_and_unparseable_years_are_none_not_raises(self) -> None:
        bib = (
            "@article{a, title = {No year at all}}\n"
            "@article{b, year = {n.d.}}\n"
            "@article{c, year = {forthcoming}}\n"
        )
        assert bib_entry_years(bib) == {"a": None, "b": None, "c": None}

    def test_indented_entry_does_not_swallow_its_successors(self) -> None:
        # A body-slicing parse survives layouts a `(?=\n@)` lookahead
        # would merge into one entry.
        bib = "@article{a,\n  year = {2001}\n}\n   @article{b,\n  year = {2019}\n}\n"
        assert bib_entry_years(bib) == {"a": 2001, "b": 2019}

    def test_year_inside_another_field_is_not_mistaken_for_the_year(self) -> None:
        bib = "@article{a,\n  title = {Trends since 1999},\n  year = {2015}\n}\n"
        assert bib_entry_years(bib) == {"a": 2015}

    def test_duplicate_key_keeps_the_dated_occurrence(self) -> None:
        bib = "@article{a, title = {No year}}\n@article{a, year = {1983}}\n"
        assert bib_entry_years(bib) == {"a": 1983}

    def test_empty_bib_is_empty_dict(self) -> None:
        assert bib_entry_years("") == {}
        assert bib_entry_years("% S2 unavailable\n") == {}


# ---------------------------------------------------------------------------
# G1 — the failure that shipped
# ---------------------------------------------------------------------------


class TestRecencyCollapse:
    def test_all_recent_bibliography_is_an_error(self, tmp_path: Path) -> None:
        """The F-P5-DEPTH-RECENCY-SKEW signature: 100% in `le2`."""
        report = _run(tmp_path, [NOW - (i % 3) for i in range(30)])
        collapse = [d for d in report.errors
                    if d.code == "reference-recency-collapse"]
        assert collapse, _codes(report)
        assert not report.format_clean
        assert "100%" in collapse[0].message

    def test_collapse_gate_is_ten_dated_references(self, tmp_path: Path) -> None:
        assert "reference-recency-collapse" not in _codes(
            _run(tmp_path, [NOW] * 9)
        )
        assert "reference-recency-collapse" in _codes(
            _run(tmp_path, [NOW] * 10)
        )

    def test_no_reference_older_than_15_years_is_an_error(
        self, tmp_path: Path
    ) -> None:
        # 20 refs spread over 8 years: recent-heavy but not collapsed.
        report = _run(tmp_path, [NOW - (i % 9) for i in range(20)])
        assert any(d.code == "no-foundational-references" for d in report.errors)
        assert "reference-recency-collapse" not in _codes(report)

    def test_foundational_floor_gate_is_fifteen(self, tmp_path: Path) -> None:
        years = [NOW - (i % 9) for i in range(14)]
        assert "no-foundational-references" not in _codes(_run(tmp_path, years))
        assert "no-foundational-references" in _codes(
            _run(tmp_path, years + [NOW - 3])
        )

    def test_one_old_reference_clears_the_foundational_floor(
        self, tmp_path: Path
    ) -> None:
        years = [NOW - (i % 9) for i in range(19)] + [NOW - 16]
        report = _run(tmp_path, years)
        assert "no-foundational-references" not in _codes(report)
        assert report.metrics["n_refs_older_than_15"] == 1

    def test_thin_historical_tail_warns(self, tmp_path: Path) -> None:
        # 19 recent + 1 at 26y => 5% older than 10y, below the 15% floor.
        report = _run(tmp_path, [NOW - (i % 9) for i in range(19)] + [NOW - 26])
        thin = [d for d in report.defects if d.code == "thin-historical-tail"]
        assert thin and thin[0].severity == "warn"
        assert report.metrics["frac_refs_older_than_10"] == pytest.approx(0.05)

    def test_healthy_tail_does_not_warn(self, tmp_path: Path) -> None:
        report = _run(tmp_path, _profile_years(EDM_34_SLOTS))
        assert "thin-historical-tail" not in _codes(report)


class TestVenueProfileSkew:
    def test_bibliography_matching_the_anchor_profile_is_clean(
        self, tmp_path: Path
    ) -> None:
        report = _run(tmp_path, _profile_years(EDM_34_SLOTS), venue="EDM")
        assert not (set(_codes(report)) & RECENCY_CODES), _codes(report)

    def test_bucket_deviation_over_tolerance_warns_and_names_the_bin(
        self, tmp_path: Path
    ) -> None:
        # 40% of references in `le2` against whatever the EDM anchor
        # profile says. The deviation is COMPUTED from the live profile
        # rather than hardcoded: this test previously pinned "+15.4pp",
        # which was the deviation from the POOLED default, and broke the
        # moment venue_norms.yaml gained a mined EDM ref_age block.
        from src.manuscript_linter import load_venue_norms, venue_age_profile

        skewed = {"le2": 8, "3_5": 4, "6_10": 4, "11_20": 2, "gt20": 2}
        profile, tolerance, _ = venue_age_profile("EDM", load_venue_norms())
        expected_pp = (8 / 20 - profile["le2"]) * 100
        assert expected_pp > tolerance, (
            "fixture must exceed the tolerance or there is nothing to warn about"
        )

        report = _run(tmp_path, _profile_years(skewed), venue="EDM")
        skew = [d for d in report.defects if d.code == "reference-recency-skew"]
        assert skew, _codes(report)
        assert skew[0].severity == "warn"
        assert "le2" in skew[0].message
        assert f"+{expected_pp:.1f}pp" in skew[0].message, skew[0].message

    def test_skew_is_a_warn_not_an_error(self, tmp_path: Path) -> None:
        """Below-profile recency is worse scholarship, not invalid output."""
        skewed = {"le2": 8, "3_5": 4, "6_10": 4, "11_20": 2, "gt20": 2}
        report = _run(tmp_path, _profile_years(skewed), venue="EDM")
        assert "reference-recency-skew" not in [d.code for d in report.errors]

    def test_venue_absent_falls_back_to_pooled_profile(
        self, tmp_path: Path
    ) -> None:
        report = _run(tmp_path, _profile_years(EDM_34_SLOTS), venue=None)
        assert report.metrics["ref_age_profile_source"] == "default"
        assert report.metrics["ref_age_target_fractions"]["le2"] == pytest.approx(
            DEFAULT_REF_AGE_PROFILE["le2"]
        )

    def test_venue_without_a_ref_age_block_falls_back(self) -> None:
        # Tests the FALLBACK ITSELF with a synthetic norms dict, rather
        # than relying on the live venue_norms.yaml lacking a block. The
        # original pinned venue="EDM" against the file's transient
        # pre-regeneration state and broke when the mined ref_age blocks
        # landed — a test of "what the file happens to contain today"
        # rather than of the behaviour it names.
        from src.manuscript_linter import (
            DEFAULT_REF_AGE_PROFILE,
            DEFAULT_REF_AGE_TOLERANCE_PP,
            venue_age_profile,
        )

        norms_without_block = {"SOMEVENUE": {"refs": {"p25": 40, "median": 50}}}
        profile, tolerance, source = venue_age_profile(
            "SOMEVENUE", norms_without_block
        )
        assert source == "default"
        assert profile == DEFAULT_REF_AGE_PROFILE
        assert tolerance == pytest.approx(DEFAULT_REF_AGE_TOLERANCE_PP)

        # An incomplete block must also fall back, not half-apply.
        partial = {"SOMEVENUE": {"ref_age": {"buckets": {"le2": 0.3}}}}
        _, _, partial_source = venue_age_profile("SOMEVENUE", partial)
        assert partial_source == "default"

    def test_live_venues_now_carry_mined_profiles(self) -> None:
        """The regeneration landed; the venue path must actually be live."""
        from src.manuscript_linter import load_venue_norms, venue_age_profile

        norms = load_venue_norms()
        for venue in ("EDM", "JEDM", "JLA"):
            _, _, source = venue_age_profile(venue, norms)
            assert source == venue, (
                f"{venue} fell back to {source!r}; venue_norms.yaml lost its "
                "ref_age.buckets block or the key was renamed"
            )


class TestVenueProfileWiring:
    """Forward-compat with the venue_norms.yaml ref_age regeneration."""

    def _norms(self, tmp_path: Path, body: str) -> Path:
        path = tmp_path / "norms.yaml"
        path.write_text(body, encoding="utf-8")
        return path

    GOOD = (
        "venues:\n"
        "  JLA:\n"
        "    refs: {p25: 47.0, median: 65}\n"
        "    ref_age:\n"
        "      buckets: {le2: 0.169, \"3_5\": 0.238, \"6_10\": 0.314,"
        " \"11_20\": 0.180, gt20: 0.099}\n"
        "      tolerance_pp: 5\n"
    )

    def test_venue_block_is_used_when_present(self, tmp_path: Path) -> None:
        report = _run(
            tmp_path, _profile_years(EDM_34_SLOTS), venue="JLA",
            norms_path=self._norms(tmp_path, self.GOOD),
        )
        assert report.metrics["ref_age_profile_source"] == "JLA"
        assert report.metrics["ref_age_tolerance_pp"] == pytest.approx(5.0)
        assert report.metrics["ref_age_target_fractions"]["6_10"] == pytest.approx(
            0.314
        )

    def test_tighter_tolerance_actually_bites(self, tmp_path: Path) -> None:
        # 23.5% in `6_10` vs JLA's 31.4% is -7.9pp: inside the pooled 12pp
        # tolerance, outside JLA's 5pp one.
        years = _profile_years(EDM_34_SLOTS)
        assert "reference-recency-skew" not in _codes(_run(tmp_path, years))
        report = _run(
            tmp_path, years, venue="JLA",
            norms_path=self._norms(tmp_path, self.GOOD),
        )
        skew = [d for d in report.defects if d.code == "reference-recency-skew"]
        assert skew and "6_10" in skew[0].message

    @pytest.mark.parametrize(
        "body",
        [
            "venues:\n  JLA:\n    ref_age: {buckets: {le2: 0.5, \"3_5\": 0.5}}\n",
            "venues:\n  JLA:\n    ref_age: {buckets: null}\n",
            "venues:\n  JLA:\n    ref_age: {buckets: {le2: oops, \"3_5\": 0.2,"
            " \"6_10\": 0.2, \"11_20\": 0.2, gt20: 0.2}}\n",
            "venues:\n  JLA:\n    ref_age: {}\n",
        ],
    )
    def test_malformed_ref_age_block_degrades_to_default(
        self, tmp_path: Path, body: str
    ) -> None:
        report = _run(
            tmp_path, _profile_years(EDM_34_SLOTS), venue="JLA",
            norms_path=self._norms(tmp_path, body),
        )
        assert report.metrics["ref_age_profile_source"] == "default"

    def test_bad_tolerance_degrades_but_keeps_the_profile(
        self, tmp_path: Path
    ) -> None:
        body = self.GOOD.replace("tolerance_pp: 5", "tolerance_pp: wide")
        report = _run(
            tmp_path, _profile_years(EDM_34_SLOTS), venue="JLA",
            norms_path=self._norms(tmp_path, body),
        )
        assert report.metrics["ref_age_profile_source"] == "JLA"
        assert report.metrics["ref_age_tolerance_pp"] == pytest.approx(12.0)

    def test_helper_is_callable_standalone(self) -> None:
        profile, tol, source = venue_age_profile("NOPE", {})
        assert profile == DEFAULT_REF_AGE_PROFILE and source == "default"
        assert tol == pytest.approx(12.0)


# ---------------------------------------------------------------------------
# Metrics — the Arc I payload
# ---------------------------------------------------------------------------


class TestRecencyMetrics:
    def test_metrics_expose_the_distribution_not_just_the_count(
        self, tmp_path: Path
    ) -> None:
        report = _run(tmp_path, _profile_years(EDM_34_SLOTS), venue="EDM")
        m = report.metrics
        for key in (
            "ref_year_parsed", "ref_year_missing", "ref_age_median",
            "ref_age_mean", "ref_age_max", "ref_age_buckets",
            "ref_age_fractions", "ref_age_target_fractions",
            "ref_age_profile_source", "n_refs_older_than_10",
            "n_refs_older_than_15", "frac_refs_older_than_10",
            "n_refs_pre_2000", "bib_age_buckets", "bib_age_fractions",
        ):
            assert key in m, f"{key} missing from lint metrics"
        assert m["ref_age_buckets"] == EDM_34_SLOTS
        assert m["ref_year_parsed"] == 34
        assert m["ref_age_max"] == 26
        assert m["n_refs_older_than_15"] == 10  # 6 at 16y + 4 at 26y
        assert m["n_refs_older_than_10"] == 10
        assert m["n_refs_pre_2000"] == 0  # the gt20 exemplar is exactly 2000
        assert sum(m["ref_age_fractions"].values()) == pytest.approx(1.0, abs=1e-3)

    def test_pre_2000_is_reported_but_is_not_a_floor(self, tmp_path: Path) -> None:
        """§1.3: reported as a metric only — no code fires on its absence."""
        years = _profile_years(EDM_34_SLOTS)[:-2] + [1983, 1999]
        report = _run(tmp_path, years, venue="EDM")
        assert report.metrics["n_refs_pre_2000"] == 2
        assert not (set(_codes(report)) & RECENCY_CODES), _codes(report)

    def test_cited_and_bib_distributions_are_separable(
        self, tmp_path: Path
    ) -> None:
        """"What was cited" must be distinguishable from "what was available"."""
        # Cited: 12 refs spanning the profile. Uncited pool tail: 20 brand new.
        cited_years = _profile_years(
            {"le2": 3, "3_5": 3, "6_10": 3, "11_20": 2, "gt20": 1}
        )
        report = _run(tmp_path, cited_years + [NOW] * 20, n_cited=12)
        assert report.metrics["ref_year_parsed"] == 12
        assert sum(report.metrics["bib_age_buckets"].values()) == 32
        assert report.metrics["ref_age_buckets"]["gt20"] == 1
        assert report.metrics["bib_age_buckets"]["le2"] == 23

    def test_unparseable_years_are_counted_not_fatal(
        self, tmp_path: Path
    ) -> None:
        years: list[Optional[int]] = _profile_years(EDM_34_SLOTS)[:]
        years += [None, None, None]
        report = _run(tmp_path, years)
        assert report.metrics["ref_year_missing"] == 3
        assert report.metrics["ref_year_parsed"] == 34
        assert isinstance(report, LintReport)

    def test_bibless_and_yearless_runs_do_not_raise(self, tmp_path: Path) -> None:
        (tmp_path / "paper.tex").write_text(_tex(["a"]), encoding="utf-8")
        report = lint_manuscript(tmp_path, write_json=False, now_year=NOW)
        assert "ref_year_parsed" not in report.metrics  # no bib at all
        (tmp_path / "references.bib").write_text(
            "not bibtex at all {{{", encoding="utf-8"
        )
        assert isinstance(
            lint_manuscript(tmp_path, write_json=False, now_year=NOW), LintReport
        )

    def test_metrics_survive_json_serialisation(self, tmp_path: Path) -> None:
        import json

        keys = [f"ref{i}" for i in range(34)]
        (tmp_path / "paper.tex").write_text(_tex(keys), encoding="utf-8")
        (tmp_path / "references.bib").write_text(
            _bib(_profile_years(EDM_34_SLOTS)), encoding="utf-8"
        )
        lint_manuscript(tmp_path, venue="EDM", now_year=NOW)
        payload = json.loads(
            (tmp_path / "manuscript_lint.json").read_text(encoding="utf-8")
        )
        assert payload["metrics"]["ref_age_buckets"]["gt20"] == 4

    def test_now_year_defaults_without_raising(self, tmp_path: Path) -> None:
        report = _run(tmp_path, [2001, 2015, 2020], now_year=0)  # 0 -> "unset"
        assert report.metrics["ref_age_now_year"] >= 2026


# ---------------------------------------------------------------------------
# The pool, not just the paper
# ---------------------------------------------------------------------------


class TestBibRecencyCollapse:
    def test_skewed_pool_warns_even_when_the_cited_subset_is_not_collapsed(
        self, tmp_path: Path
    ) -> None:
        # 2 old + 10 new cited (83% le2, under the collapse bar), but the
        # 40-entry pool behind them is 95% brand new.
        cited = [NOW - 26, NOW - 16] + [NOW] * 10
        report = _run(tmp_path, cited + [NOW] * 30, n_cited=12)
        assert "reference-recency-collapse" not in _codes(report)
        bib = [d for d in report.defects if d.code == "bib-recency-collapse"]
        assert bib and bib[0].severity == "warn"

    def test_no_pool_tail_means_no_pool_warning(self, tmp_path: Path) -> None:
        # Every entry is cited => there is no separate "pool" to indict.
        report = _run(tmp_path, _profile_years(EDM_34_SLOTS))
        assert "bib-recency-collapse" not in _codes(report)

    def test_healthy_pool_does_not_warn(self, tmp_path: Path) -> None:
        report = _run(
            tmp_path, _profile_years(EDM_34_SLOTS) + [NOW - 30] * 10, n_cited=34
        )
        assert "bib-recency-collapse" not in _codes(report)


# ---------------------------------------------------------------------------
# Short / synthetic manuscripts must never be flagged
# ---------------------------------------------------------------------------


class TestShortBibliographiesAreExempt:
    def test_eight_recent_references_fire_nothing(self, tmp_path: Path) -> None:
        report = _run(tmp_path, [NOW] * 8)
        assert not (set(_codes(report)) & RECENCY_CODES), _codes(report)

    def test_three_reference_fixture_stays_format_clean(
        self, tmp_path: Path
    ) -> None:
        """Pins tests/test_manuscript_linter.py's CLEAN_BIB shape."""
        report = _run(tmp_path, [2020, 2021, 2022])
        assert report.format_clean, _codes(report)
        assert not (set(_codes(report)) & RECENCY_CODES)

    @pytest.mark.parametrize("bib", ["", "% S2 API unavailable\n"])
    def test_tiny_synthetic_manuscripts_fire_nothing(
        self, tmp_path: Path, bib: str
    ) -> None:
        """Pins test_arc_p3_p4.py's TestFrontMatterCompleteness fixtures."""
        (tmp_path / "paper.tex").write_text(
            r"\title{T}\abstract{" + "w " * 40
            + r"}\begin{document}x\end{document}",
            encoding="utf-8",
        )
        (tmp_path / "references.bib").write_text(bib, encoding="utf-8")
        report = lint_manuscript(tmp_path, write_json=False, now_year=NOW)
        assert not (set(_codes(report)) & RECENCY_CODES), _codes(report)


# ---------------------------------------------------------------------------
# G5 — uncited bib entries
# ---------------------------------------------------------------------------


class TestUncitedBibEntriesAreNotNoise:
    def test_uncited_superset_is_counted_but_not_flagged(
        self, tmp_path: Path
    ) -> None:
        """62 entries / 19 cited is the designed Arc P3 reviser pool.

        BibTeX typesets only cited entries, so the other 43 never reach
        the PDF; warning about them on every run is pure noise.
        """
        report = _run(tmp_path, _profile_years(EDM_34_SLOTS) + [NOW - 4] * 28,
                      n_cited=19)
        assert report.metrics["n_bib_uncited"] == 43
        assert "many-uncited-bib-entries" not in _codes(report)
        assert report.metrics["nocite_all"] is False

    def test_nocite_star_makes_them_reader_visible_and_warns(
        self, tmp_path: Path
    ) -> None:
        report = _run(
            tmp_path, _profile_years(EDM_34_SLOTS) + [NOW - 4] * 28,
            n_cited=19, extra_tex=r"\nocite{*}",
        )
        warn = [d for d in report.defects if d.code == "many-uncited-bib-entries"]
        assert warn and warn[0].severity == "warn"
        assert r"\nocite{*}" in warn[0].message
        assert report.metrics["nocite_all"] is True

    def test_nocite_star_is_not_a_citation_key(self, tmp_path: Path) -> None:
        """``*`` must not count as a citation nor as a missing bib entry."""
        report = _run(tmp_path, [2020, 2021, 2022], extra_tex=r"\nocite{*}")
        assert report.metrics["n_citations_distinct"] == 3
        assert "cited-key-missing-from-bib" not in _codes(report), _codes(report)

    def test_explicit_nocite_keys_still_count_as_cited(
        self, tmp_path: Path
    ) -> None:
        keys = [f"ref{i}" for i in range(3)]
        (tmp_path / "paper.tex").write_text(
            _tex(keys[:1], extra=r"\nocite{ref1,ref2}"), encoding="utf-8"
        )
        (tmp_path / "references.bib").write_text(
            _bib([2020, 2021, 2022]), encoding="utf-8"
        )
        report = lint_manuscript(tmp_path, write_json=False, now_year=NOW)
        assert report.metrics["n_citations_distinct"] == 3
        assert report.metrics["n_bib_uncited"] == 0


# ---------------------------------------------------------------------------
# The real artifact
# ---------------------------------------------------------------------------


class TestShippedRunWouldHaveBeenCaught:
    SHIPPED = ROOT / "runs" / "arc_p_validation_20260711" / "output"

    def test_shipped_arc_p_validation_bib_is_flagged(
        self, tmp_path: Path
    ) -> None:
        """The "would have caught it" proof, on the artifact that shipped."""
        if not (self.SHIPPED / "references.bib").exists():
            pytest.skip("shipped Arc P validation run not on this machine")
        for name in ("paper.tex", "references.bib"):
            (tmp_path / name).write_text(
                (self.SHIPPED / name).read_text(encoding="utf-8", errors="replace"),
                encoding="utf-8",
            )
        report = lint_manuscript(
            tmp_path, venue="JEDM", write_json=False, now_year=2026
        )
        codes = _codes(report)
        assert "reference-recency-collapse" in codes, codes
        assert "no-foundational-references" in codes, codes
        assert report.metrics["n_refs_older_than_15"] == 0
        assert report.metrics["ref_age_max"] <= 2
        # …and the count metric it hid behind was, and stays, green.
        assert report.metrics["n_bib_entries"] >= 60
        assert "many-uncited-bib-entries" not in codes
