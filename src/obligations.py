"""Obligations the manuscript owes the review, and whether it paid them.

The Critic runs before the Writer, so a finding it addresses to the
Writer cannot be a request to re-run anything -- the Writer has not run.
It is a condition on the paper that does not exist yet.

Until now there was no channel for that at all. Across 19 archived runs
the Critic addressed 41 findings to the Writer, a third of everything it
found, and its own validator deleted every one before the report was
saved. Six of those runs carry the receipt: ``revision_instructions
targeted unknown agent 'Writer'; ignored``. The discarded instructions
were not cosmetic -- "the RQ promises subgroup fairness analysis on
X1SEX, X1RACE and X1SES, but subgroup_performance is empty", "state the
analyzed n for the DIF model, not the full-sample n", "the SES bins are
labelled
with numeric ranges that are not interpretable to readers".

An obligation here carries its own test. It closes when the test passes
against the produced manuscript, never because an agent reported
compliance -- the one instruction whose disposition is documented in
both directions was *applied without disclosure*, which is its own
defect.

The tests are deliberately weak and deliberately deterministic. A weak
test that runs is worth more than a strong one that needs a judge: the
flagship case is satisfied by asking whether the instructed number
appears anywhere in ``paper.tex``, and the paper that ignored it
contains neither that number nor its unpunctuated form.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = [
    "Obligation",
    "derive_obligations",
    "evaluate_obligations",
    "render_for_writer",
]


@dataclass
class Obligation:
    """One thing the manuscript must do, and how to tell whether it did.

    ``assert_kind``:
        ``tex_contains``  -- every string in ``assert_values`` appears.
        ``tex_absent``    -- no string in ``assert_values`` appears.
        ``manual``        -- no deterministic test; reported as open.
    """

    id: str
    source: str
    severity: str
    instruction: str
    assert_kind: str = "manual"
    assert_values: list[str] = field(default_factory=list)
    status: str = "open"
    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


#: Numerals worth requiring. A bare "5" or a year would bind to almost
#: any manuscript and make the test meaningless.
_NUMERAL = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d{3,}(?:\.\d+)?|\d+\.\d{2,})(?![\w])")
_YEAR = re.compile(r"^(?:19|20)\d{2}$")
#: An instruction that asks for something to be REMOVED, not added.
_NEGATIVE = re.compile(
    r"\b(?:must not|should not|do not|don't|never|remove|delete|drop|avoid|"
    r"stop (?:calling|describing|reporting))\b",
    re.IGNORECASE,
)


def _significant_numerals(text: str) -> list[str]:
    out: list[str] = []
    for m in _NUMERAL.finditer(text):
        tok = m.group(1)
        bare = tok.replace(",", "")
        if _YEAR.match(bare):
            continue
        if tok not in out:
            out.append(tok)
    return out


def derive_obligations(review_report: dict | None) -> list[Obligation]:
    """Build obligations from a review report.

    Two sources, in order of preference:

    1. ``review_report["writer_obligations"]`` -- if the Critic supplied
       explicit records, use them verbatim. This is the good path.
    2. ``revision_instructions["Writer"]`` and any ``issues[*]`` whose
       ``target_agent`` is ``Writer`` -- free text, from which a test is
       derived by taking the significant numerals the instruction names.

    Deriving a test from prose is crude and that is the point: it costs
    nothing, it needs no judge, and on the one archived instruction whose
    outcome is documented it is exactly sufficient.
    """
    if not isinstance(review_report, dict):
        return []
    out: list[Obligation] = []
    seen: set[str] = set()

    explicit = review_report.get("writer_obligations")
    if isinstance(explicit, list):
        for i, rec in enumerate(explicit):
            if not isinstance(rec, dict) or not rec.get("instruction"):
                continue
            vals = rec.get("assert_values")
            if isinstance(vals, str):
                vals = [vals]
            out.append(
                Obligation(
                    id=str(rec.get("id") or f"WO{i + 1:02d}"),
                    source="critic.writer_obligations",
                    severity=str(rec.get("severity") or "major"),
                    instruction=str(rec["instruction"]),
                    assert_kind=str(rec.get("assert_kind") or "manual"),
                    assert_values=[str(v) for v in (vals or [])],
                )
            )
            seen.add(str(rec["instruction"]).strip())

    def _add(text: str, severity: str, source: str) -> None:
        text = (text or "").strip()
        if not text or text in seen:
            return
        seen.add(text)
        nums = _significant_numerals(text)
        if nums and not _NEGATIVE.search(text):
            kind, values = "tex_contains", nums[:4]
        else:
            kind, values = "manual", []
        out.append(
            Obligation(
                id=f"WO{len(out) + 1:02d}",
                source=source,
                severity=severity,
                instruction=text,
                assert_kind=kind,
                assert_values=values,
            )
        )

    ri = review_report.get("revision_instructions")
    if isinstance(ri, dict) and ri.get("Writer"):
        _add(str(ri["Writer"]), "major", "revision_instructions.Writer")

    for section in review_report.values():
        if not isinstance(section, dict):
            continue
        for issue in section.get("issues") or []:
            if not isinstance(issue, dict):
                continue
            if str(issue.get("target_agent") or "") != "Writer":
                continue
            text = issue.get("recommendation") or issue.get("description") or ""
            _add(str(text), str(issue.get("severity") or "minor"), "issues[target_agent=Writer]")
    return out


def _normalize(tex: str) -> str:
    """Make numeral matching survive LaTeX's spacing macros.

    ``4{,}312`` and ``4,312`` and ``4312`` are the same number to a
    reader and three different strings to ``in``.
    """
    t = tex.replace("{,}", ",").replace("\\,", ",").replace(" ", "")
    return t


def evaluate_obligations(
    obligations: list[Obligation], paper_tex: str | None
) -> list[Obligation]:
    """Close each obligation whose test passes against *paper_tex*.

    An obligation with no deterministic test stays ``open`` and says so.
    It is not a failure; it is a thing nobody checked, which is the
    honest report.
    """
    if paper_tex is None:
        for ob in obligations:
            ob.status = "unknown"
            ob.detail = "no manuscript to check against"
        return obligations

    tex = _normalize(paper_tex)
    for ob in obligations:
        if ob.assert_kind == "tex_contains" and ob.assert_values:
            missing = [
                v
                for v in ob.assert_values
                if v not in tex and v.replace(",", "") not in tex.replace(",", "")
            ]
            ob.status = "closed" if not missing else "open"
            ob.detail = (
                "all required values present"
                if not missing
                else f"absent from the manuscript: {', '.join(missing)}"
            )
        elif ob.assert_kind == "tex_absent" and ob.assert_values:
            present = [v for v in ob.assert_values if v in tex]
            ob.status = "closed" if not present else "open"
            ob.detail = (
                "none of the forbidden values appear"
                if not present
                else f"still present: {', '.join(present)}"
            )
        else:
            ob.status = "unchecked"
            ob.detail = "no deterministic test; requires a reader"
    return obligations


def render_for_writer(obligations: list[Obligation]) -> str:
    """The block the Writer sees. Empty string when there is nothing.

    The Writer already receives ``review_report.json`` as raw JSON, and
    receiving it changed nothing -- its prompt has no rule telling it to
    act on anything in there, and an undifferentiated blob is not an
    instruction. This block is.
    """
    if not obligations:
        return ""
    lines = [
        "## Obligations from the Critic — address every one",
        "",
        "These are findings the Critic addressed to you specifically. They",
        "are not background: the manuscript is checked against them after",
        "you write it, and each one names what has to appear. Where an",
        "obligation gives a number, print that number rather than a",
        "different one or none.",
        "",
    ]
    for ob in obligations:
        lines.append(f"- [{ob.severity}] ({ob.id}) {ob.instruction}")
        if ob.assert_kind == "tex_contains" and ob.assert_values:
            lines.append(
                f"    The paper must contain: {', '.join(ob.assert_values)}"
            )
        elif ob.assert_kind == "tex_absent" and ob.assert_values:
            lines.append(
                f"    The paper must NOT contain: {', '.join(ob.assert_values)}"
            )
    lines.append("")
    return "\n".join(lines)


def summarize(obligations: list[Obligation]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for ob in obligations:
        counts[ob.status] = counts.get(ob.status, 0) + 1
    return {
        "n_obligations": len(obligations),
        "by_status": counts,
        "compliance_rate": (
            counts.get("closed", 0) / len(obligations) if obligations else None
        ),
        "obligations": [ob.to_dict() for ob in obligations],
    }
