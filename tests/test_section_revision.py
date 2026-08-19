"""Section-scoped revision for the LSAR review gate (Arc P4 follow-up).

P4's improvement layer had never once landed a revision in a live run:
the whole-document path embeds the entire manuscript and demands the
complete document back, which on a measured 67,167-char paper.tex
(~14.2k tokens) leaves ~11% headroom in a 16,000-token reply budget
*while the model is asked to add prose*. Two live attempts, two
failures: one no-op, one rejected by the float guard.

These tests pin the replacement: only the sections carrying the weakest
dimension are sent and required back, and they are spliced in
deterministically so every other byte is untouched by construction.
Everything here is offline — the provider client is a local stub.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional

import pytest

from src.review_gate import ReviewGate

# ---------------------------------------------------------------------------
# Fixture manuscript — shaped like a real shipped paper.tex
# ---------------------------------------------------------------------------

_FILLER = (
    "Students in the analytic sample varied widely in their measured "
    "outcomes, and the pattern held across every subgroup we examined. "
)

INTRO_PROSE = "Prior work has largely ignored this question. " + _FILLER * 8
RW_PROSE = "Three research traditions meet here. " + _FILLER * 8
METHODS_PROSE = "We fit six model families with an inner cross-validation. " + _FILLER * 8
RESULTS_PROSE = "The gradient boosted model performed best. " + _FILLER * 6
DISC_PROSE = "The findings suggest a modest practical signal. " + _FILLER * 6
LIMIT_PROSE = "School clustering is not modeled. " + _FILLER * 5

TEX = f"""\\documentclass[sigconf]{{acmart}}
\\begin{{document}}
\\title{{Predicting Mathematics Achievement from Ninth-Grade Attitudes}}
\\maketitle
%\\section{{Commented Ghost}}
\\begin{{abstract}}
This study predicts mathematics achievement from ninth-grade attitudes.
\\end{{abstract}}

\\section{{Introduction}}
{INTRO_PROSE}

\\section{{Related Work}}
{RW_PROSE}

\\section{{Methods}}
{METHODS_PROSE}

\\section{{Results}}
{RESULTS_PROSE}
\\begin{{table}}
\\caption{{Model comparison}}
\\begin{{tabular}}{{ll}}
Model & AUC \\\\
XGBoost & 0.82 \\\\
\\end{{tabular}}
\\end{{table}}
\\includegraphics{{shap_summary.png}}

\\section{{Discussion}}
{DISC_PROSE}
\\subsection{{Limitations and Future Directions}}
{LIMIT_PROSE}

\\bibliographystyle{{ACM-Reference-Format}}
\\bibliography{{references}}
\\end{{document}}
"""


class _FakeLLM:
    """Stand-in for the OpenAI-compatible client (deepseek branch)."""

    def __init__(self, responder: Any) -> None:
        self._responder = responder
        self.prompts: list[str] = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs: Any) -> Any:
        self.prompts.append(kwargs["messages"][-1]["content"])
        resp = self._responder
        if isinstance(resp, Exception):
            raise resp
        text = resp(kwargs) if isinstance(resp, Callable) else resp  # type: ignore[arg-type]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
        )


def _gate(
    tmp_path: Path,
    logs: Optional[list[str]] = None,
    max_tokens: int = 2000,
) -> ReviewGate:
    cfg = {
        "llm_provider": "deepseek",
        "deepseek": {"models": {"revision_writer": "deepseek-v4-pro"}},
        "review_gate": {
            "venue": "EDM",
            "revision_model": "deepseek-v4-pro",
            "revision_max_tokens": max_tokens,
        },
    }
    sink = logs if logs is not None else []
    return ReviewGate(cfg, str(tmp_path), log_fn=lambda _a, m: sink.append(m))


_DIAGNOSIS = {
    "overall_score": 4.8,
    "dimension_scores": {"Novelty": 3, "Clarity of Communication": 7},
    "suggested_focus_areas": [
        {"dimension": "Novelty", "score": "3", "target_agent": "Writer"}
    ],
}


def _revise(gate: ReviewGate, tex: str = TEX, diagnosis: dict = _DIAGNOSIS) -> str:
    return gate.revise_from_review(
        paper_tex=tex, report_json={"review": {}}, diagnosis=diagnosis
    )


def _reply(blocks: list, bodies: list[str]) -> str:
    return "\n\n".join(
        f"### SECTION {i + 1}: {b.title}\n```latex\n{bodies[i]}\n```"
        for i, b in enumerate(blocks)
    )


def _targets(gate: ReviewGate, tex: str = TEX, names: Optional[list] = None) -> list:
    blocks = gate._select_target_blocks(tex, names or ["Introduction", "Related Work"])
    return gate._fit_blocks_to_budget(blocks)


# ---------------------------------------------------------------------------
# Splitting / selection
# ---------------------------------------------------------------------------


class TestSectionSplitting:
    def test_finds_abstract_and_all_sections(self, tmp_path: Path) -> None:
        blocks = _gate(tmp_path)._split_sections(TEX)
        titles = [b.title for b in blocks]
        assert titles[0] == "Abstract"
        assert "Introduction" in titles and "Related Work" in titles
        assert "Limitations and Future Directions" in titles

    def test_commented_out_heading_is_ignored(self, tmp_path: Path) -> None:
        assert all(
            b.title != "Commented Ghost"
            for b in _gate(tmp_path)._split_sections(TEX)
        )

    def test_blocks_are_verbatim_slices(self, tmp_path: Path) -> None:
        for b in _gate(tmp_path)._split_sections(TEX):
            assert TEX[b.start:b.end] == b.text

    def test_last_section_does_not_swallow_the_bibliography(
        self, tmp_path: Path
    ) -> None:
        """Splicing a reply that omits \\bibliography would delete it."""
        blocks = _gate(tmp_path)._split_sections(TEX)
        last = [b for b in blocks if b.level == "section"][-1]
        assert "\\bibliography" not in last.text
        assert "\\end{document}" not in last.text

    def test_post_bibliography_appendix_stops_before_end_document(
        self, tmp_path: Path
    ) -> None:
        """The UNVERIFIED path appends an appendix AFTER \\bibliography."""
        tex = TEX.replace(
            "\\end{document}",
            "\\section*{Appendix: Automated Critic Review Report}\nIssues.\n"
            "\\end{document}",
        )
        blocks = _gate(tmp_path)._split_sections(tex)
        appendix = [b for b in blocks if b.title.startswith("Appendix")]
        assert appendix and "\\end{document}" not in appendix[0].text

    def test_sections_do_not_overlap(self, tmp_path: Path) -> None:
        secs = [
            b for b in _gate(tmp_path)._split_sections(TEX) if b.level == "section"
        ]
        for a, b in zip(secs, secs[1:]):
            assert a.end <= b.start

    def test_duplicate_heading_picks_the_block_with_content(
        self, tmp_path: Path
    ) -> None:
        """The sectionwise writer emits \\section{Introduction} twice."""
        dup = TEX.replace(
            "\\section{Introduction}\n",
            "\\section{Introduction}\n\\section{Introduction}\n",
            1,
        )
        picked = _gate(tmp_path)._select_target_blocks(dup, ["Introduction"])
        assert len(picked) == 1
        assert INTRO_PROSE[:40] in picked[0].text

    def test_apa_macro_abstract_is_found(self, tmp_path: Path) -> None:
        tex = (
            "\\title{T}\n\\abstract{Nested \\emph{x} and $y_{2}$ here.}\n"
            "\\begin{document}\n\\section{Introduction}\nprose\n\\end{document}\n"
        )
        blocks = _gate(tmp_path)._split_sections(tex)
        abstract = [b for b in blocks if b.level == "abstract"]
        assert abstract and abstract[0].text.endswith("here.}")


class TestTargetSelection:
    def test_weakest_dimension_selects_its_sections(self, tmp_path: Path) -> None:
        gate = _gate(tmp_path)
        names = gate._target_section_names(["Novelty"])
        assert names == ["Introduction", "Related Work"]
        assert [b.title for b in gate._select_target_blocks(TEX, names)] == [
            "Introduction",
            "Related Work",
        ]

    def test_fuzzy_title_match_reaches_real_headings(self, tmp_path: Path) -> None:
        """Real papers ship "Limitations and Future Directions"."""
        picked = _gate(tmp_path)._select_target_blocks(TEX, ["Limitations"])
        assert [b.title for b in picked] == ["Limitations and Future Directions"]
        assert picked[0].level == "subsection"

    def test_parent_and_child_are_never_both_sent(self, tmp_path: Path) -> None:
        picked = _gate(tmp_path)._select_target_blocks(
            TEX, ["Discussion", "Limitations"]
        )
        assert [b.title for b in picked] == ["Discussion"]

    def test_unknown_section_name_is_skipped(self, tmp_path: Path) -> None:
        assert _gate(tmp_path)._select_target_blocks(TEX, ["Acknowledgements"]) == []

    def test_budget_drops_lowest_priority_sections(self, tmp_path: Path) -> None:
        gate = _gate(tmp_path, max_tokens=400)  # ~1.6k chars of sections
        kept = gate._fit_blocks_to_budget(
            gate._select_target_blocks(TEX, ["Introduction", "Related Work"])
        )
        assert [b.title for b in kept] == ["Introduction"]

    def test_no_focus_dimensions_falls_back_to_defaults(self, tmp_path: Path) -> None:
        assert _gate(tmp_path)._target_section_names([]) == [
            "Introduction",
            "Discussion",
        ]


# ---------------------------------------------------------------------------
# Path choice
# ---------------------------------------------------------------------------


class TestPathChoice:
    def test_measured_shipped_manuscript_does_not_fit(self, tmp_path: Path) -> None:
        """67,167 chars against a 16,000-token budget — the live case."""
        gate = _gate(tmp_path, max_tokens=16000)
        assert not gate._fits_whole_document("x" * 67167)

    def test_short_manuscript_fits(self, tmp_path: Path) -> None:
        gate = _gate(tmp_path, max_tokens=16000)
        assert gate._fits_whole_document("x" * 20000)

    def test_short_paper_still_uses_the_whole_document_path(
        self, tmp_path: Path
    ) -> None:
        gate = _gate(tmp_path, max_tokens=16000)
        short = (
            "\\documentclass{article}\\begin{document}\n\\section{Introduction}\n"
            "short prose\n\\end{document}\n"
        )
        revised_doc = short.replace("short prose", "longer, better prose")
        fake = _FakeLLM(f"```latex\n{revised_doc}\n```")
        gate._llm_client = fake
        out = _revise(gate, tex=short)
        assert "Return the COMPLETE revised paper.tex" in fake.prompts[0]
        # _extract_latex strips the fenced block (pre-existing behaviour)
        assert out == revised_doc.strip()

    def test_long_paper_uses_the_section_path(self, tmp_path: Path) -> None:
        gate = _gate(tmp_path)
        fake = _FakeLLM("no code block here")
        gate._llm_client = fake
        _revise(gate)
        prompt = fake.prompts[0]
        assert "Revise **only**" in prompt
        assert "Return the COMPLETE revised paper.tex" not in prompt

    def test_section_prompt_sends_only_the_target_sections(
        self, tmp_path: Path
    ) -> None:
        gate = _gate(tmp_path)
        fake = _FakeLLM("nothing")
        gate._llm_client = fake
        _revise(gate)
        prompt = fake.prompts[0]
        assert INTRO_PROSE[:60] in prompt
        assert RW_PROSE[:60] in prompt
        assert METHODS_PROSE[:60] not in prompt
        assert RESULTS_PROSE[:60] not in prompt
        assert "0.82" not in prompt  # the results table never leaves the file

    def test_section_prompt_carries_orienting_context(self, tmp_path: Path) -> None:
        gate = _gate(tmp_path)
        fake = _FakeLLM("nothing")
        gate._llm_client = fake
        _revise(gate)
        prompt = fake.prompts[0]
        assert "Predicting Mathematics Achievement" in prompt  # title
        assert "This study predicts mathematics achievement" in prompt  # abstract
        assert "- Methods" in prompt  # outline of untouched sections
        assert "REVISE THIS ONE" in prompt


# ---------------------------------------------------------------------------
# Splicing
# ---------------------------------------------------------------------------


class TestSplicing:
    def test_good_revision_splices_and_leaves_everything_else_identical(
        self, tmp_path: Path
    ) -> None:
        logs: list[str] = []
        gate = _gate(tmp_path, logs)
        blocks = _targets(gate)
        bodies = [
            b.text.strip() + f"\nAdded sentence {i}." for i, b in enumerate(blocks)
        ]
        gate._llm_client = _FakeLLM(_reply(blocks, bodies))

        out = _revise(gate)

        assert "Added sentence 0." in out and "Added sentence 1." in out
        # Every other byte is untouched: strip the additions and the
        # manuscript must be byte-identical to what we started with.
        restored = out.replace("\nAdded sentence 0.", "").replace(
            "\nAdded sentence 1.", ""
        )
        assert restored == TEX
        assert any("spliced 2 of 2" in m for m in logs)

    def test_whole_document_guard_passes_on_a_spliced_result(
        self, tmp_path: Path
    ) -> None:
        """Deterministic splicing makes the float guard nearly free."""
        gate = _gate(tmp_path)
        blocks = _targets(gate)
        bodies = [b.text.strip() + "\nMore framing." for b in blocks]
        gate._llm_client = _FakeLLM(_reply(blocks, bodies))
        out = _revise(gate)
        ok, reason = gate._revision_is_safe(TEX, out)
        assert ok, reason

    def test_single_section_revision(self, tmp_path: Path) -> None:
        gate = _gate(tmp_path)
        diagnosis = dict(
            _DIAGNOSIS,
            suggested_focus_areas=[
                {"dimension": "Significance & Impact", "score": "3",
                 "target_agent": "Writer"}
            ],
        )
        blocks = _targets(gate, names=["Discussion"])
        gate._llm_client = _FakeLLM(
            _reply(blocks, [blocks[0].text.strip() + "\nImplications matter."])
        )
        out = _revise(gate, diagnosis=diagnosis)
        assert out.replace("\nImplications matter.", "") == TEX

    def test_unrequested_section_in_the_reply_is_ignored(
        self, tmp_path: Path
    ) -> None:
        gate = _gate(tmp_path)
        blocks = _targets(gate)
        rogue = "\\section{Results}\nWe now report an AUC of 0.99.\n"
        reply = _reply(blocks, [b.text.strip() + "\nOK." for b in blocks])
        reply += f"\n\n### SECTION 3: Results\n```latex\n{rogue}\n```"
        gate._llm_client = _FakeLLM(reply)
        out = _revise(gate)
        assert "0.99" not in out
        assert out.replace("\nOK.", "") == TEX


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


class TestSectionGuards:
    def _results(self, tmp_path: Path, logs: list[str], body: str) -> str:
        """Drive the Results section (which owns the table) through the
        section-revision path directly: no reviewer dimension routes a
        prose reviser at Results, by design."""
        gate = _gate(tmp_path, logs)
        blocks = _targets(gate, names=["Results"])
        gate._llm_client = _FakeLLM(_reply(blocks, [body(blocks[0].text)]))
        return gate._revise_sections(
            paper_tex=TEX, blocks=blocks, strengths=[], weaknesses=[],
            suggestions=[], questions=[], focus_dims=["Empirical Support / Results"],
            diagnosis={},
        )

    def test_section_that_drops_a_table_is_rejected(self, tmp_path: Path) -> None:
        logs: list[str] = []
        out = self._results(
            tmp_path,
            logs,
            lambda t: t[: t.index("\\begin{table}")]
            + "Prose only, evidence deleted. " * 12,
        )
        assert out == TEX, "a section that destroys evidence must not be spliced"
        assert any("REJECTED" in m and "tables/figures" in m for m in logs)

    def test_edited_number_inside_a_table_is_rejected(self, tmp_path: Path) -> None:
        logs: list[str] = []
        out = self._results(tmp_path, logs, lambda t: t.replace("0.82", "0.91"))
        assert out == TEX
        assert any("REJECTED" in m and "tables/figures" in m for m in logs)

    def test_dropped_figure_is_rejected(self, tmp_path: Path) -> None:
        logs: list[str] = []
        out = self._results(
            tmp_path, logs,
            lambda t: t.replace("\\includegraphics{shap_summary.png}", ""),
        )
        assert out == TEX
        assert any("REJECTED" in m for m in logs)

    def test_untouched_floats_do_not_block_a_prose_edit(
        self, tmp_path: Path
    ) -> None:
        """The guard must not reject a legitimate revision either."""
        logs: list[str] = []
        out = self._results(
            tmp_path, logs,
            lambda t: t.replace(
                "The gradient boosted model performed best.",
                "The gradient boosted model performed best by a clear margin.",
            ),
        )
        assert "by a clear margin" in out
        assert out.replace(" by a clear margin", "") == TEX

    def test_truncated_section_is_rejected(self, tmp_path: Path) -> None:
        logs: list[str] = []
        gate = _gate(tmp_path, logs)
        blocks = _targets(gate)
        cut = blocks[0].text[: len(blocks[0].text) // 3]
        gate._llm_client = _FakeLLM(
            _reply(blocks, [cut, blocks[1].text.strip() + "\nFine."])
        )
        out = _revise(gate)
        assert out.replace("\nFine.", "") == TEX  # only the good one spliced
        assert any("REJECTED" in m and "truncat" in m.lower() for m in logs)

    def test_unbalanced_environment_is_rejected(self, tmp_path: Path) -> None:
        """A reply cut off inside a list keeps its length but breaks LaTeX."""
        logs: list[str] = []
        gate = _gate(tmp_path, logs)
        blocks = _targets(gate, names=["Introduction"])
        broken = blocks[0].text.strip() + "\n\\begin{itemize}\n\\item first"
        gate._llm_client = _FakeLLM(_reply(blocks, [broken]))
        assert _revise(gate) == TEX
        assert any("unbalanced environments" in m for m in logs)

    def test_unbalanced_braces_are_rejected(self, tmp_path: Path) -> None:
        logs: list[str] = []
        gate = _gate(tmp_path, logs)
        blocks = _targets(gate, names=["Introduction"])
        broken = blocks[0].text.strip() + "\nSee \\textbf{the point"
        gate._llm_client = _FakeLLM(_reply(blocks, [broken]))
        assert _revise(gate) == TEX
        assert any("unbalanced braces" in m for m in logs)

    def test_wrong_block_kind_is_rejected(self, tmp_path: Path) -> None:
        logs: list[str] = []
        gate = _gate(tmp_path, logs)
        blocks = _targets(gate, names=["Introduction"])
        gate._llm_client = _FakeLLM(
            "```latex\n\\documentclass{article}\n"
            + blocks[0].text.strip()
            + "\n\\end{document}\n```"
        )
        assert _revise(gate) == TEX

    def test_extra_sectioning_command_is_rejected(self, tmp_path: Path) -> None:
        logs: list[str] = []
        gate = _gate(tmp_path, logs)
        blocks = _targets(gate, names=["Introduction"])
        split = blocks[0].text.strip() + "\n\\section{Bonus Section}\nSurprise.\n"
        gate._llm_client = _FakeLLM(_reply(blocks, [split]))
        assert _revise(gate) == TEX
        assert any("must not be split or merged" in m for m in logs)

    def test_empty_section_is_rejected(self, tmp_path: Path) -> None:
        gate = _gate(tmp_path)
        blocks = _targets(gate, names=["Introduction"])
        gate._llm_client = _FakeLLM(_reply(blocks, ["   "]))
        assert _revise(gate) == TEX


# ---------------------------------------------------------------------------
# Degradation
# ---------------------------------------------------------------------------


class TestDegradation:
    def test_unchanged_sections_are_a_detectable_no_op(self, tmp_path: Path) -> None:
        gate = _gate(tmp_path)
        blocks = _targets(gate)
        gate._llm_client = _FakeLLM(_reply(blocks, [b.text.strip() for b in blocks]))
        assert _revise(gate) == TEX, (
            "an echoed manuscript must come back identical so run_gate's "
            "no-op branch fires instead of a pointless rewrite"
        )

    def test_llm_failure_returns_the_original(self, tmp_path: Path) -> None:
        logs: list[str] = []
        gate = _gate(tmp_path, logs)
        gate._llm_client = _FakeLLM(RuntimeError("502 upstream"))
        assert _revise(gate) == TEX
        assert any("LLM revision call failed" in m for m in logs)

    def test_unparseable_reply_returns_the_original(self, tmp_path: Path) -> None:
        logs: list[str] = []
        gate = _gate(tmp_path, logs)
        gate._llm_client = _FakeLLM("Sure! I would be happy to help with that.")
        assert _revise(gate) == TEX
        assert any("Could not extract any revised section" in m for m in logs)

    def test_missing_section_in_reply_is_reported_and_the_rest_lands(
        self, tmp_path: Path
    ) -> None:
        logs: list[str] = []
        gate = _gate(tmp_path, logs)
        blocks = _targets(gate)
        gate._llm_client = _FakeLLM(
            f"### SECTION 1\n```latex\n{blocks[0].text.strip()}\nOnly one.\n```"
        )
        out = _revise(gate)
        assert out.replace("\nOnly one.", "") == TEX
        assert any("not returned by the model" in m for m in logs)

    def test_manuscript_without_sections_falls_back_to_whole_document(
        self, tmp_path: Path
    ) -> None:
        logs: list[str] = []
        gate = _gate(tmp_path, logs, max_tokens=200)  # nothing fits comfortably
        tex = "\\documentclass{article}\\begin{document}\n" + _FILLER * 10 + (
            "\n\\end{document}\n"
        )
        revised = tex.replace("varied widely", "varied enormously")
        gate._llm_client = _FakeLLM(f"```latex\n{revised}\n```")
        assert _revise(gate, tex=tex) == revised.strip()
        assert any("No target sections could be located" in m for m in logs)


# ---------------------------------------------------------------------------
# Reply parsing
# ---------------------------------------------------------------------------


class TestReplyParsing:
    def test_headings_alone_are_enough(self, tmp_path: Path) -> None:
        """No SECTION markers, blocks returned out of order."""
        gate = _gate(tmp_path)
        blocks = _targets(gate)
        reply = "\n\n".join(
            f"```latex\n{b.text.strip()}\nAdded {i}.\n```"
            for i, b in reversed(list(enumerate(blocks)))
        )
        gate._llm_client = _FakeLLM(reply)
        out = _revise(gate)
        assert out.replace("\nAdded 0.", "").replace("\nAdded 1.", "") == TEX

    def test_unfenced_reply_is_still_parsed(self, tmp_path: Path) -> None:
        gate = _gate(tmp_path)
        blocks = _targets(gate)
        gate._llm_client = _FakeLLM(
            "\n\n".join(b.text.strip() + "\nRaw." for b in blocks)
        )
        out = _revise(gate)
        assert out.replace("\nRaw.", "") == TEX

    def test_a_marker_never_overrides_a_mismatched_heading(
        self, tmp_path: Path
    ) -> None:
        """The dangerous case: "SECTION 1" attached to another section's
        prose would splice Results over the Introduction."""
        logs: list[str] = []
        gate = _gate(tmp_path, logs)
        rogue = "\\section{Results}\n" + "We now report an AUC of 0.99. " * 20
        gate._llm_client = _FakeLLM(f"### SECTION 1\n```latex\n{rogue}\n```")
        assert _revise(gate) == TEX
        assert any("does not match any requested section" in m for m in logs)

    def test_reworded_heading_still_matches_its_section(
        self, tmp_path: Path
    ) -> None:
        gate = _gate(tmp_path)
        blocks = _targets(gate, names=["Introduction"])
        body = blocks[0].text.strip().replace(
            "\\section{Introduction}", "\\section{Introduction and Motivation}"
        )
        gate._llm_client = _FakeLLM(_reply(blocks, [body]))
        out = _revise(gate)
        assert "\\section{Introduction and Motivation}" in out
        assert out.replace(" and Motivation", "") == TEX


# ---------------------------------------------------------------------------
# The no-op log used to claim a recompile that still happened
# ---------------------------------------------------------------------------


class TestNoOpLogIsHonest:
    def _run_gate_with_noop_revision(self, tmp_path: Path) -> tuple[list[str], dict]:
        logs: list[str] = []
        gate = _gate(tmp_path, logs)
        gate.max_cycles = 2
        gate.pass_threshold = 9.0
        (tmp_path / "paper.tex").write_text(TEX, encoding="utf-8")
        pdf = tmp_path / "paper_for_review.pdf"
        pdf.write_bytes(b"%PDF-1.4")

        calls = {"prepare": 0, "compile": 0}

        def _prepare(run_dir: Path, cycle: Optional[int] = None) -> Path:
            calls["prepare"] += 1
            return pdf

        gate.prepare_pdf = _prepare  # type: ignore[method-assign]
        gate.run_lsar = lambda p, c: {  # type: ignore[method-assign]
            "scores": {"overall_score": 4.0, "recommendation": "Reject",
                       "dimensions": [{"name": "Novelty", "score": 5}]},
            "review": {},
        }
        gate._maybe_median_sample = lambda r, p, c: r  # type: ignore[method-assign]
        gate.revise_from_review = (  # type: ignore[method-assign]
            lambda paper_tex, report_json, diagnosis, lint_report=None: paper_tex
        )
        gate._compile_full_latex = lambda d: calls.__setitem__(  # type: ignore[method-assign]
            "compile", calls["compile"] + 1
        )
        gate.run_gate()
        return logs, calls

    def test_message_does_not_claim_a_skipped_recompile(
        self, tmp_path: Path
    ) -> None:
        logs, calls = self._run_gate_with_noop_revision(tmp_path)
        noop = [m for m in logs if "no-op" in m]
        assert noop, logs
        assert "skipping rewrite and recompile" not in noop[0]
        assert "left unchanged" in noop[0]
        assert "next cycle still recompiles" in noop[0].replace("\n", " ")
        # The claim is now true to what the loop does: paper.tex is never
        # rewritten, but cycle 2 does prepare (and therefore compile) the
        # unchanged manuscript again.
        assert calls["compile"] == 0
        assert calls["prepare"] == 2

    def test_paper_tex_is_untouched_by_a_no_op(self, tmp_path: Path) -> None:
        self._run_gate_with_noop_revision(tmp_path)
        assert (tmp_path / "paper.tex").read_text(encoding="utf-8") == TEX


# ---------------------------------------------------------------------------
# Backwards compatibility with the pinned surfaces
# ---------------------------------------------------------------------------


class TestPinnedSurfaces:
    def test_revise_from_review_signature(self) -> None:
        import inspect

        params = inspect.signature(ReviewGate.revise_from_review).parameters
        assert list(params) == [
            "self", "paper_tex", "report_json", "diagnosis", "lint_report",
        ]
        assert params["lint_report"].default is None

    def test_whole_document_prompt_unchanged_for_short_papers(
        self, tmp_path: Path
    ) -> None:
        prompt = _gate(tmp_path)._build_revision_prompt(
            paper_tex="x", strengths=[], weaknesses=[], suggestions=[],
            questions=[], focus_dims=["Novelty"], diagnosis=_DIAGNOSIS,
        )
        assert "Concentrate your edits" in prompt
        assert "Related Work" in prompt


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-q"])
