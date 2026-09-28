"""What the review gate can take from a reviser's reply.

The whole-document path accepted exactly two reply shapes: a closed
```latex fence, or an unfenced \\documentclass .. \\end{document} span. A
reply fenced as ```tex, one that returned only the sections it changed,
and one cut off at the token limit were all "Could not extract LaTeX from
LLM response; keeping original" (Mac study, round 3), and the gate then
paid for three more reviews of the unchanged paper.

A complete document is still taken whole. Anything less is spliced into
the existing manuscript section by section, behind the per-section guards
the section path already uses; the preamble and the trailer never come
from the reply.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

from src.review_gate import ReviewGate

INTRO = "We study how ninth-grade attitudes relate to later achievement. " * 3
RELATED = "Prior studies used smaller samples and fewer model families. " * 3
RESULTS = "The gradient boosted model reached the highest AUC. " * 3
DISCUSSION = "The signal is modest and should be read with care. " * 3

TEX = f"""\\documentclass[sigconf]{{acmart}}
\\usepackage{{booktabs}}
\\begin{{document}}
\\title{{Attitudes and Achievement}}
\\begin{{abstract}}
We predict achievement from attitudes in a national sample of students.
\\end{{abstract}}
\\maketitle

\\section{{Introduction}}
{INTRO}

\\section{{Related Work}}
{RELATED}

\\section{{Results}}
{RESULTS}
\\begin{{table}}
\\begin{{tabular}}{{ll}}
Model & AUC \\\\
XGBoost & 0.82 \\\\
\\end{{tabular}}
\\end{{table}}

\\section{{Discussion}}
{DISCUSSION}

\\bibliographystyle{{ACM-Reference-Format}}
\\bibliography{{references}}
\\end{{document}}
"""

NEW_INTRO = "\\section{Introduction}\n" + INTRO + "We now state the gap plainly.\n"
NEW_RELATED = "\\section{Related Work}\n" + RELATED + "We position the study against them.\n"
NEW_DISCUSSION = "\\section{Discussion}\n" + DISCUSSION + "We add an honest limitation.\n"

DIAGNOSIS = {
    "overall_score": 4.8,
    "dimension_scores": {"Novelty": 3},
    "suggested_focus_areas": [{"dimension": "Novelty", "score": "3",
                               "target_agent": "Writer"}],
}


class _Chat:
    def __init__(self, replies: list[tuple[str, Optional[str]]]) -> None:
        self.replies = list(replies)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **_kwargs: Any) -> Any:
        text, finish = self.replies.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=text), finish_reason=finish)])


def _gate(tmp_path: Path, logs: list[str]) -> ReviewGate:
    cfg = {"llm_provider": "deepseek",
           "deepseek": {"models": {"revision_writer": "deepseek-v4-pro"}},
           "review_gate": {"revision_max_tokens": 16000}}
    return ReviewGate(cfg, str(tmp_path), log_fn=lambda _a, m: logs.append(m))


def _revise(gate: ReviewGate, replies: list[tuple[str, Optional[str]]]) -> str:
    gate._llm_client = _Chat(replies)
    assert gate._fits_whole_document(TEX), "fixture must take the whole-document path"
    return gate.revise_from_review(
        paper_tex=TEX, report_json={"review": {}}, diagnosis=DIAGNOSIS)


class TestCompleteDocuments:
    def test_any_latex_fence_label_is_accepted(self, tmp_path: Path) -> None:
        gate = _gate(tmp_path, [])
        revised = TEX.replace(INTRO, INTRO + "Added. ")
        for label in ("latex", "LaTeX", "tex", ""):
            reply = f"Here it is.\n```{label}\n{revised}```\n"
            assert gate._extract_latex(reply) == revised.strip(), label

    def test_a_fence_that_is_not_a_document_is_not_one(self, tmp_path: Path) -> None:
        """The old extraction returned any ```latex block as the paper."""
        gate = _gate(tmp_path, [])
        assert gate._extract_latex(f"```latex\n{NEW_INTRO}```") is None

    def test_a_forgotten_closing_fence_still_yields_the_document(
        self, tmp_path: Path
    ) -> None:
        gate = _gate(tmp_path, [])
        revised = TEX.replace(INTRO, INTRO + "Added. ")
        assert gate._extract_latex(f"```latex\n{revised}") == revised.strip()


class TestPartialReplies:
    def test_returned_sections_are_spliced_into_the_manuscript(
        self, tmp_path: Path
    ) -> None:
        logs: list[str] = []
        gate = _gate(tmp_path, logs)
        reply = f"```latex\n{NEW_INTRO}\n{NEW_RELATED}```"
        out = _revise(gate, [(reply, "stop")])
        assert "We now state the gap plainly." in out
        assert "We position the study against them." in out
        # Everything outside the two sections is byte-identical.
        assert out.startswith(TEX[: TEX.index("\\section{Introduction}")])
        assert out.endswith(TEX[TEX.index("\\section{Results}"):])
        assert gate._revision_is_safe(TEX, out) == (True, "")
        assert not any("Could not extract" in m for m in logs)

    def test_unfenced_sections_are_spliced_too(self, tmp_path: Path) -> None:
        gate = _gate(tmp_path, [])
        out = _revise(gate, [(f"Revised sections:\n\n{NEW_DISCUSSION}", "stop")])
        assert "We add an honest limitation." in out
        assert out.startswith(TEX[: TEX.index("\\section{Introduction}")])

    def test_a_cut_off_document_keeps_its_complete_sections(
        self, tmp_path: Path
    ) -> None:
        """Cut off twice (the retry too): the sections before the cut land,
        the one that was cut does not."""
        logs: list[str] = []
        gate = _gate(tmp_path, logs)
        revised = TEX.replace(INTRO, INTRO + "Sharper framing. ").replace(
            RELATED, RELATED + "Better positioning. ")
        cut = revised[: revised.index("Better positioning.") + len("Better pos")]
        reply = f"```latex\n{cut}"
        out = _revise(gate, [(reply, "length"), (reply, "length")])
        assert "Sharper framing." in out
        assert "Better pos" not in out
        assert any("cut off at the token limit" in m for m in logs)

    def test_a_section_that_touches_a_table_is_refused(self, tmp_path: Path) -> None:
        logs: list[str] = []
        gate = _gate(tmp_path, logs)
        bad = TEX[TEX.index("\\section{Results}"):TEX.index("\\section{Discussion}")]
        bad = bad.replace("0.82", "0.91")
        out = _revise(gate, [(f"```latex\n{bad}```", "stop")])
        assert out == TEX
        assert any("REJECTED" in m for m in logs)

    def test_nothing_usable_keeps_the_original_and_says_why(
        self, tmp_path: Path
    ) -> None:
        logs: list[str] = []
        gate = _gate(tmp_path, logs)
        assert _revise(gate, [("I cannot help with that.", "stop")]) == TEX
        [line] = [m for m in logs if "Could not extract LaTeX" in m]
        assert "no complete document and no known section" in line

    def test_a_reply_cut_off_before_any_section_says_so(self, tmp_path: Path) -> None:
        logs: list[str] = []
        gate = _gate(tmp_path, logs)
        reply = "```latex\n\\documentclass[sigconf]{acmart}\n\\usepackage"
        assert _revise(gate, [(reply, "length"), (reply, "length")]) == TEX
        [line] = [m for m in logs if "Could not extract LaTeX" in m]
        assert "cut off at the token limit" in line


class TestSectionPathReplies:
    def _section_gate(self, tmp_path: Path) -> ReviewGate:
        cfg = {"llm_provider": "deepseek",
               "deepseek": {"models": {"revision_writer": "deepseek-v4-pro"}},
               "review_gate": {"revision_max_tokens": 300}}
        gate = ReviewGate(cfg, str(tmp_path), log_fn=lambda *_: None)
        assert not gate._fits_whole_document(TEX)
        return gate

    def test_a_whole_document_answer_is_matched_by_its_headings(
        self, tmp_path: Path
    ) -> None:
        gate = self._section_gate(tmp_path)
        revised = TEX.replace(INTRO, INTRO + "Sharper framing. ")
        gate._llm_client = _Chat([(f"```latex\n{revised}```", "stop")])
        out = gate.revise_from_review(
            paper_tex=TEX, report_json={"review": {}}, diagnosis=DIAGNOSIS)
        assert "Sharper framing." in out

    def test_a_cut_off_unfenced_answer_loses_its_last_block(
        self, tmp_path: Path
    ) -> None:
        gate = self._section_gate(tmp_path)
        cut_related = NEW_RELATED[: NEW_RELATED.index("We position") + 6]
        reply = NEW_INTRO + "\n" + cut_related
        gate._llm_client = _Chat([(reply, "length"), (reply, "length")])
        gate.revision_retry_max_tokens = 600
        out = gate.revise_from_review(
            paper_tex=TEX, report_json={"review": {}}, diagnosis=DIAGNOSIS)
        assert "We now state the gap plainly." in out
        assert "We pos" not in out
