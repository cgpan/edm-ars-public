"""A revision that did not happen does not buy another review.

The Mac study (round 3): cycle 1's three reviews gave a median of 5.7,
the revision failed ("Could not extract LaTeX from LLM response; keeping
original"), and cycle 2 still paid for three more reviews of the SAME
paper. Their lower median, 5.1, became the final score.

Now a revision that produces no changed manuscript (the call failed, no
LaTeX came back, the guards rejected it, or it changed only whitespace
and comments) ends the gate with the last reviewed result and says why
in gate_summary.json. Reviews of a manuscript that was already reviewed
are pooled into one median, never allowed to replace its score. The
summary also says whether the final score was given to the Writer's
manuscript or to a revised one.

LSAR and the reviser are faked; no paid call is made.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

from src.review_gate import ReviewGate

TEX = (
    "\\documentclass{article}\n\\begin{document}\n"
    "\\section{Introduction}\n" + "Students differ in how they see mathematics. " * 12
    + "\n\\section{Discussion}\n" + "The association is modest. " * 12
    + "\n\\end{document}\n"
)


def _report(score: float) -> dict:
    return {
        "scores": {"overall_score": score, "recommendation": "Weak Reject",
                   "dimensions": [{"name": "Novelty", "score": 5}]},
        "review": {"strengths": ["clear"], "weaknesses": ["thin"]},
    }


class _Chat:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **_kw: Any) -> Any:
        self.calls += 1
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=self.text), finish_reason="stop")])


def _gate(
    tmp_path: Path,
    scores: list[float],
    *,
    max_cycles: int = 2,
    median_samples: int = 3,
    paper_md: Optional[str] = None,
) -> tuple[ReviewGate, list[tuple[str, dict]], list[int]]:
    """A gate whose LSAR returns *scores* in order (a 9.0 threshold, so
    every cycle fails and asks for a revision)."""
    cfg = {
        "llm_provider": "deepseek",
        "deepseek": {"models": {"revision_writer": "deepseek-v4-pro"}},
        "review_gate": {"max_cycles": max_cycles, "median_samples": median_samples,
                        "median_trigger_band": 10.0, "revision_max_tokens": 16000},
    }
    gate = ReviewGate(cfg, str(tmp_path), log_fn=lambda *_: None)
    gate.pass_threshold = 9.0
    (tmp_path / "paper.tex").write_text(TEX, encoding="utf-8")
    pdf = tmp_path / "paper_for_review.pdf"
    pdf.write_bytes(b"%PDF-1.5 stub")
    gate.prepare_pdf = lambda *_a, **_k: pdf  # type: ignore[method-assign]
    queue = list(scores)
    reviewed: list[int] = []

    def run_lsar(_pdf: Any, cycle: int) -> dict:
        reviewed.append(cycle)
        folder = tmp_path / "lsar_review" / f"cycle_{cycle}"
        folder.mkdir(parents=True, exist_ok=True)
        if paper_md is not None:
            (folder / "paper.md").write_text(paper_md, encoding="utf-8")
        return _report(queue.pop(0))

    gate.run_lsar = run_lsar  # type: ignore[method-assign]
    gate._honesty_blockers = lambda: []  # type: ignore[method-assign]
    gate._compile_full_latex = lambda *_a: None  # type: ignore[method-assign]
    seen: list[tuple[str, dict]] = []
    gate.event_fn = lambda t, **kw: seen.append((t, kw))
    return gate, seen, reviewed


class TestAFailedRevisionEndsTheGate:
    def test_the_mac_case_keeps_cycle_1_and_pays_for_no_more_reviews(
        self, tmp_path: Path
    ) -> None:
        gate, seen, reviewed = _gate(tmp_path, [5.7, 5.2, 6.0, 5.1, 4.8, 5.3])
        gate._llm_client = _Chat("I have revised the paper as requested.")

        summary = gate.run_gate()

        assert reviewed == [1, 102, 103], "cycle 2 must not be reviewed"
        assert summary["final_score"] == 5.7
        assert summary["cycles_used"] == 1
        assert summary["revision_failed"] is True
        assert "no LaTeX could be taken from the reply" in summary["revision_failure_reason"]
        assert summary["final_score_from"] == "original"
        assert summary["final_score_cycle"] == 1
        assert summary["revisions_applied"] == 0
        assert summary["final_manuscript_reviewed"] is True
        [warning] = [kw for t, kw in seen if t == "warning"]
        assert warning["code"] == "GATE_REVISION_FAILED"
        on_disk = json.loads((tmp_path / "lsar_review" / "gate_summary.json")
                             .read_text(encoding="utf-8"))
        assert on_disk["revision_failed"] is True
        assert on_disk["final_score_from"] == "original"
        assert (tmp_path / "paper.tex").read_text(encoding="utf-8") == TEX

    def test_no_reviser_ends_the_gate_with_the_reason(self, tmp_path: Path) -> None:
        gate, _, reviewed = _gate(tmp_path, [5.0, 5.0], median_samples=1)
        gate._llm_client = None
        gate.revision_unavailable_reason = "no revision model configured"

        summary = gate.run_gate()

        assert reviewed == [1]
        assert summary["revision_failure_reason"] == (
            "no reviser: no revision model configured")

    def test_a_failed_call_ends_the_gate_with_its_code(self, tmp_path: Path) -> None:
        gate, _, reviewed = _gate(tmp_path, [5.0, 5.0], median_samples=1)
        chat = _Chat("")

        def boom(**_kw: Any) -> Any:
            raise RuntimeError("upstream 502")

        chat.chat.completions.create = boom
        gate._llm_client = chat

        summary = gate.run_gate()

        assert reviewed == [1]
        assert summary["revision_failure_reason"] == "the revision call failed (UNKNOWN)"

    def test_a_rejected_revision_ends_the_gate(self, tmp_path: Path) -> None:
        gate, _, reviewed = _gate(tmp_path, [5.0, 5.0], median_samples=1)
        truncated = TEX.replace("\\end{document}\n", "")
        gate.revise_from_review = lambda **_k: truncated  # type: ignore[method-assign]

        summary = gate.run_gate()

        assert reviewed == [1]
        assert summary["revision_failure_reason"].startswith(
            "the revision was rejected: revision is truncated")

    def test_a_whitespace_and_comment_revision_is_no_revision(
        self, tmp_path: Path
    ) -> None:
        gate, _, reviewed = _gate(tmp_path, [5.0, 5.0], median_samples=1)
        cosmetic = TEX.replace("\\section{Discussion}",
                               "% tightened wording\n\n\\section{Discussion}")
        gate.revise_from_review = lambda **_k: cosmetic  # type: ignore[method-assign]

        summary = gate.run_gate()

        assert reviewed == [1]
        assert "only whitespace or LaTeX comments" in summary["revision_failure_reason"]
        assert (tmp_path / "paper.tex").read_text(encoding="utf-8") == TEX


class TestARealRevisionIsReviewed:
    def test_the_revised_paper_is_reviewed_and_the_summary_says_so(
        self, tmp_path: Path
    ) -> None:
        gate, _, reviewed = _gate(tmp_path, [5.0, 6.0], median_samples=1)
        revised = TEX.replace("The association is modest. ",
                              "The association is modest but consistent. ", 1)
        gate._llm_client = _Chat(f"```latex\n{revised}```")

        summary = gate.run_gate()

        assert reviewed == [1, 2]
        assert summary["final_score"] == 6.0
        assert summary["revision_failed"] is False
        assert summary["revision_failure_reason"] is None
        assert summary["final_score_from"] == "revised"
        assert summary["final_score_cycle"] == 2
        assert summary["revisions_applied"] == 1
        assert [c["manuscript"] for c in summary["per_cycle_scores"]] == [
            "original", "revised"]


class TestReviewsOfTheSameManuscriptArePooled:
    def test_a_re_review_of_the_same_text_is_pooled_not_substituted(
        self, tmp_path: Path
    ) -> None:
        """The LaTeX changed (an unused macro) but the text the reviewer
        read did not: cycle 2's reviews join cycle 1's in one median."""
        gate, _, reviewed = _gate(
            tmp_path, [5.2, 5.7, 6.0, 4.8, 5.1, 5.3], paper_md="# Same paper\n")
        changed = TEX.replace("\\begin{document}",
                              "\\newcommand{\\unused}{x}\n\\begin{document}")
        gate.revise_from_review = lambda **_k: changed  # type: ignore[method-assign]

        summary = gate.run_gate()

        assert reviewed == [1, 102, 103, 2, 202, 203]
        # Sorted: 4.8 5.1 5.2 5.3 5.7 6.0 -> median (5.2 + 5.3) / 2.
        assert summary["final_score"] == 5.25
        second = summary["per_cycle_scores"][1]
        assert second["pooled_with_cycle"] == 1
        assert second["median_sampling"]["pooled_cycles"] == [1, 2]
        assert second["median_sampling"]["n_samples"] == 6

    def test_a_different_text_is_not_pooled(self, tmp_path: Path) -> None:
        gate, _, _ = _gate(tmp_path, [5.0, 6.0], median_samples=1)
        texts = iter(["# First version\n", "# Second version\n"])
        original_run = gate.run_lsar

        def run_lsar(pdf: Any, cycle: int) -> dict:
            report = original_run(pdf, cycle)
            (tmp_path / "lsar_review" / f"cycle_{cycle}" / "paper.md").write_text(
                next(texts), encoding="utf-8")
            return report

        gate.run_lsar = run_lsar  # type: ignore[method-assign]
        revised = TEX.replace("modest. ", "modest, and stable. ", 1)
        gate.revise_from_review = lambda **_k: revised  # type: ignore[method-assign]

        summary = gate.run_gate()

        assert summary["final_score"] == 6.0
        assert summary["per_cycle_scores"][1]["pooled_with_cycle"] is None
