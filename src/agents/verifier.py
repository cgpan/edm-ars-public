"""Verifier: hold the finished manuscript against the run's own artifacts.

This is the stage the pipeline did not have. The Critic runs BEFORE the
Writer and has never seen a paper — its input is the spec, the data
report and the results object, and the claims it would need to check do
not exist yet. The LSAR gate runs after, but reads a 48,000-character
head-slice with no figures in it, and until this arc nothing anywhere
read its verdict.

So half the catalogued defects were addressed to nobody. Of the 72
defects found in EDM-ARS manuscripts, 7 are reachable from artifacts
alone, 29 more from a manuscript parser, and **35 need one narrow
judgement against evidence that fits in a single prompt** — a subgroup
table, a null ``ablation`` field, a figure. Zero need a panel. That is
what this agent is: one judge, with the evidence in front of it, and a
parse-time validator that throws away anything it cannot substantiate.

Three constraints, all of them load-bearing:

**The rater contract is used verbatim.** ``agent_prompts/verifier_rubric.md``
is a copy of the instrument measured at 98% flag precision against the
built-in reviewers' 67%. Rewording it invalidates that measurement.

**Every finding must carry a verbatim quote from the manuscript.** The
validator drops any finding whose quote is not a literal substring of
``paper.tex``, whose category is outside the whitelist, or that arrives
past the record cap. A finding nobody can open and check is not a
finding.

**Precision here will be lower than 98% and the gap is the number to
report.** That figure was measured cross-family — a Kimi rater against a
DeepSeek generator. This runs same-family by instruction. Nothing here
corrects for that; it is recorded so a reader can discount accordingly.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from src.agents.base import BaseAgent, parse_llm_json

#: Dimensions the rubric defines. Anything else is out of contract.
VALID_DIMENSIONS = frozenset({"D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "G"})

#: Categories this agent must not raise. Novelty, style, venue fit and
#: statistical significance are opinions; the instrument measures whether
#: the paper's claims match its artifacts.
BANNED_CATEGORY_WORDS = (
    "novelty",
    "novel contribution",
    "writing style",
    "prose quality",
    "venue fit",
    "not significant enough",
    "should have used",
    "would have been better",
)

#: A cap, because an unbounded finding list is a way of saying nothing.
MAX_FINDINGS = 12

VALID_SEVERITIES = frozenset({"critical", "major", "minor"})
VALID_DIRECTIONS = frozenset({"flatters", "harms", "none"})

#: Phrases in which a finding concedes there is no discrepancy.
_SELF_NEGATING = re.compile(
    r"(?:[-—,:;]\s*(?:this is\s+)?consistent\b"
    r"|\bwhich is consistent\b"
    r"|\bthe two (?:\w+\s+){0,2}(?:agree|match)\b"
    r"|\bno (?:actual |real )?(?:discrepancy|inconsistency|mismatch)\b"
    r"|\bis correct\b|\bis accurate\b|\bthis is fine\b)",
    re.IGNORECASE,
)


def _normalize_for_quote_match(text: str) -> str:
    """Collapse whitespace and LaTeX spacing so quoting survives wrapping.

    A quote lifted from a wrapped ``.tex`` will not match character for
    character; a quote that is merely *invented* will not match under any
    normalization.
    """
    t = text.replace("{,}", ",").replace("\\,", " ").replace("~", " ")
    t = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?", " ", t)
    t = re.sub(r"[{}$\\]", " ", t)
    return re.sub(r"\s+", " ", t).strip().lower()


class Verifier(BaseAgent):
    """One narrow judge over the produced manuscript."""

    #: Figures are checked one call each, with the caption. A batch of
    #: ten images in one prompt gets one distracted answer; ten narrow
    #: calls get ten answers, and the measured cost of that split was
    #: about $0.10 for 107 figures.
    MAX_FIGURE_CALLS = 8

    def __init__(
        self,
        context,
        config: dict,
        source_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        """``source_dir`` is where the run's artifacts are READ from.

        It defaults to ``ctx.output_dir``, which is the normal in-pipeline
        case. It exists so the offline harness can point at an archived
        run while writing everything -- report, raw response, prompt
        capture -- somewhere else. The evaluation archive is the only
        surviving record of roughly seventy hours of adversarial
        verification and a scoring pass must not leave marks on it.
        """
        super().__init__(context, "Verifier", config, **kwargs)
        self.source_dir = source_dir or self.ctx.output_dir

    # ------------------------------------------------------------------

    def run(self, **kwargs: Any) -> dict:
        paper = self._read("paper.tex")
        if not paper:
            return {
                "ran": False,
                "reason": "no paper.tex to verify",
                "findings": [],
            }

        digest = self._artifact_digest()
        machine = self._machine_checks()
        message = self._build_user_message(paper, digest, machine)

        raw = self.call_llm(message, max_tokens=self._max_tokens())
        findings, dropped, parse_error = self._parse_and_validate(raw, paper)

        figure_findings, figure_notes = self._check_figures(paper)

        result = {
            "ran": True,
            "model": self.model,
            "same_family_as_generator": True,
            "precision_caveat": (
                "The 98% flag precision this contract was measured at came "
                "from a cross-family rater. This ran same-family, which is "
                "expected to be lower by an unmeasured amount."
            ),
            "findings": findings + figure_findings,
            "n_dropped_by_validator": len(dropped),
            "dropped": dropped[:20],
            "figure_notes": figure_notes,
            "parse_error": parse_error,
        }
        self._persist(result, raw)
        return result

    # ------------------------------------------------------------------
    # inputs
    # ------------------------------------------------------------------

    def _read(self, name: str) -> str:
        path = os.path.join(self.source_dir, name)
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                return f.read()
        except OSError:
            return ""

    def _json(self, name: str) -> Any:
        try:
            return json.loads(self._read(name) or "null")
        except ValueError:
            return None

    #: Booleans the analysis wrote about itself. They are the run's own
    #: opinion of the run, and a judge that reads them inherits the
    #: opinion instead of checking it. One archived paper asserted
    #: "the estimand-match check passed for all five methods" from a
    #: field whose value was computed by comparing a string to itself.
    SELF_CERTIFIED_KEYS = (
        "validation_passed",
        "match",
        "significant",
        "converged",
        "passed",
        "model_quality_gate",
    )

    def _strip_self_certified(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {
                k: self._strip_self_certified(v)
                for k, v in obj.items()
                if k not in self.SELF_CERTIFIED_KEYS
            }
        if isinstance(obj, list):
            return [self._strip_self_certified(v) for v in obj]
        return obj

    def _artifact_digest(self) -> str:
        parts = []
        for name in ("research_spec.json", "data_report.json", "results.json"):
            data = self._json(name)
            if data is None:
                continue
            data = self._strip_self_certified(data)
            blob = json.dumps(data, indent=1, default=str)
            if len(blob) > 60_000:
                blob = blob[:60_000] + "\n... [digest truncated]"
            parts += [f"### {name}", "```json", blob, "```", ""]
        for name in ("model_comparison.csv", "feature_importance.csv",
                     "subgroup_performance.csv"):
            text = self._read(name)
            if text:
                parts += [f"### {name}", "```csv", text[:8_000], "```", ""]
        return "\n".join(parts) or "(no artifacts on disk)"

    def _machine_checks(self) -> str:
        """The deterministic battery's own output, as LEADS not verdicts."""
        payload = self._json("invariants.json") or {}
        obligations = self._json("obligations.json") or {}
        out = []
        for f in payload.get("findings", [])[:40]:
            out.append(
                f"- [{f.get('severity')}] {f.get('code')}: {f.get('message', '')[:300]}"
            )
        block = "\n".join(out) or "(none)"
        ob_lines = [
            f"- [{o.get('status')}] {o.get('instruction', '')[:200]}"
            for o in obligations.get("obligations", [])[:12]
        ]
        return (
            "### Deterministic checks already run (LEADS, not verdicts)\n"
            + block
            + "\n\n### Obligations the Critic placed on this manuscript\n"
            + ("\n".join(ob_lines) or "(none)")
        )

    def _build_user_message(self, paper: str, digest: str, machine: str) -> str:
        body = paper
        if len(body) > 90_000:
            # Cut the bibliography first, never the Results. The reviewer
            # in this project head-slices at 48,000 characters and the
            # venue format puts floats after the references, so floats
            # were the first casualty on 104 of 111 real papers.
            body = re.sub(
                r"(?s)\\begin\{thebibliography\}.*?\\end\{thebibliography\}",
                "\n% [bibliography omitted from this digest]\n",
                body,
            )[:90_000]
        return "\n".join(
            [
                f"## paper_id\n{os.path.basename(self.ctx.output_dir)}",
                "",
                "## Manuscript, as produced",
                "```latex",
                body,
                "```",
                "",
                "## ARTEFACT DIGEST",
                digest,
                "",
                "## MACHINE CHECK RESULTS",
                machine,
                "",
                f"Return at most {MAX_FINDINGS} findings. Every one needs a "
                "verbatim quote from the manuscript above and evidence from "
                "the digest. Findings without both are discarded before "
                "anyone reads them.",
            ]
        )

    def _max_tokens(self) -> int:
        per_stage = (self.config.get("per_stage_max_tokens") or {})
        return int(per_stage.get("verifier", per_stage.get("critic", 16000)))

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------

    def _parse_and_validate(
        self, raw: str, paper: str
    ) -> tuple[list[dict], list[dict], str | None]:
        try:
            payload = parse_llm_json(self._last_json_block(raw))
        except Exception as exc:  # noqa: BLE001
            return [], [], f"{type(exc).__name__}: {exc}"

        raw_findings = payload.get("findings")
        if not isinstance(raw_findings, list):
            return [], [], "response had no findings list"

        haystack = _normalize_for_quote_match(paper)
        kept: list[dict] = []
        dropped: list[dict] = []

        for i, f in enumerate(raw_findings):
            if not isinstance(f, dict):
                dropped.append({"index": i, "reason": "not an object"})
                continue
            reason = self._rejection_reason(f, haystack, len(kept))
            if reason:
                dropped.append(
                    {
                        "index": i,
                        "reason": reason,
                        "quote": str(f.get("quote", ""))[:160],
                        "dimension": f.get("dimension"),
                    }
                )
                continue
            kept.append(
                {
                    "id": str(f.get("id") or f"F{len(kept) + 1}"),
                    "dimension": str(f.get("dimension")),
                    "severity": str(f.get("severity")).lower(),
                    "location": str(f.get("location", ""))[:200],
                    "quote": str(f.get("quote", ""))[:400],
                    "problem": str(f.get("problem", ""))[:400],
                    "evidence": str(f.get("evidence", ""))[:600],
                    "direction": str(f.get("direction", "none")).lower(),
                    "source": "verifier",
                }
            )

        for key in ("gate_observations", "unverifiable", "positives", "summary"):
            if key in payload:
                # Carried for the record, not as findings.
                pass
        return kept, dropped, None

    def _rejection_reason(
        self, f: dict, haystack: str, n_kept: int
    ) -> str | None:
        if n_kept >= MAX_FINDINGS:
            return f"over the {MAX_FINDINGS}-finding cap"
        if str(f.get("dimension")) not in VALID_DIMENSIONS:
            return f"dimension {f.get('dimension')!r} is not in the rubric"
        if str(f.get("severity", "")).lower() not in VALID_SEVERITIES:
            return f"severity {f.get('severity')!r} is not critical/major/minor"
        if not str(f.get("evidence", "")).strip():
            return "no evidence; the rubric says no evidence, no finding"

        text = " ".join(
            str(f.get(k, "")) for k in ("problem", "evidence", "location")
        ).lower()
        for banned in BANNED_CATEGORY_WORDS:
            if banned in text:
                return f"out of contract: reads as {banned!r}"

        # A finding that talks itself out of being one. Observed live:
        # "The gap is stated as 0.094 ... and as '0.746 vs. 0.652', which
        # is a difference of 0.094 -- consistent." The model did the
        # arithmetic, found agreement, and filed it anyway.
        if _SELF_NEGATING.search(text):
            return "the finding's own text concludes the values agree"

        quote = str(f.get("quote", "")).strip()
        if not quote:
            return "no quote"
        if len(quote.split()) > 45:
            return f"quote is {len(quote.split())} words, over the 40-word limit"
        needle = _normalize_for_quote_match(quote)
        if len(needle) < 12:
            return "quote too short to locate"
        if needle not in haystack:
            # The one check that cannot be argued with. A quote that is
            # not in the paper is not evidence about the paper.
            return "quote is not a literal substring of the manuscript"
        return None

    @staticmethod
    def _last_json_block(text: str) -> str:
        fence = text.rfind("```json")
        if fence >= 0:
            end = text.find("```", fence + 7)
            return text[fence + 7 : end if end > 0 else len(text)]
        start, stop = text.find("{"), text.rfind("}")
        return text[start : stop + 1] if start >= 0 and stop > start else text

    # ------------------------------------------------------------------
    # figures
    # ------------------------------------------------------------------

    _INCLUDEGRAPHICS = re.compile(
        r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}"
    )

    def _check_figures(self, paper: str) -> tuple[list[dict], list[str]]:
        """One narrow vision call per embedded figure.

        Nothing in this project has ever looked at a figure. The reviewer
        converts the PDF with no ``write_images`` and emits no
        placeholder, so a caption saying "Love Plot of Covariate Balance"
        over a chart listing Age, BMI and Smoking Status -- none of which
        exist in the study -- reads as fine to every text-only check.
        """
        if not self._vision_capable():
            return [], ["verifier model has no image path; figures not inspected"]

        findings: list[dict] = []
        notes: list[str] = []
        seen: set[str] = set()
        for m in self._INCLUDEGRAPHICS.finditer(paper):
            if len(seen) >= self.MAX_FIGURE_CALLS:
                notes.append(
                    f"stopped after {self.MAX_FIGURE_CALLS} figures; "
                    "later figures were not inspected"
                )
                break
            ref = os.path.basename(m.group(1))
            path = self._resolve_figure(ref)
            if path is None:
                notes.append(f"{ref}: referenced but not on disk")
                continue
            if path in seen:
                continue
            seen.add(path)
            caption = self._caption_near(paper, m.start())
            verdict = self._ask_about_figure(path, caption)
            if verdict is None:
                notes.append(f"{os.path.basename(path)}: inspection failed")
                continue
            if verdict.get("mismatch"):
                findings.append(
                    {
                        "id": f"FIG{len(findings) + 1}",
                        "dimension": "D2",
                        "severity": str(verdict.get("severity", "major")).lower(),
                        "location": f"figure {os.path.basename(path)}",
                        "quote": caption[:300],
                        "problem": str(verdict.get("problem", ""))[:400],
                        "evidence": f"{os.path.basename(path)}: "
                        + str(verdict.get("shows", ""))[:300],
                        "direction": "none",
                        "source": "verifier.figure",
                    }
                )
            else:
                notes.append(
                    f"{os.path.basename(path)}: caption matches what the "
                    "figure shows"
                )
        return findings, notes

    def _vision_capable(self) -> bool:
        return self._provider in ("deepseek", "openai")

    def _resolve_figure(self, ref: str) -> str | None:
        stem = ref.split(".")[0]
        for ext in (".png", ".jpg", ".jpeg", ""):
            cand = os.path.join(self.source_dir, stem + ext)
            if os.path.isfile(cand):
                return cand
        return None

    @staticmethod
    def _caption_near(paper: str, pos: int) -> str:
        window = paper[pos : pos + 1200]
        m = re.search(r"\\caption\{(.{0,600}?)\}\s*(?:\\label|\n)", window, re.DOTALL)
        if m:
            return re.sub(r"\s+", " ", m.group(1)).strip()
        back = paper[max(0, pos - 1200) : pos]
        m = re.search(r"\\caption\{(.{0,600}?)\}", back, re.DOTALL)
        return re.sub(r"\s+", " ", m.group(1)).strip() if m else "(no caption found)"

    _FIGURE_QUESTION = (
        "You are checking ONE figure against ONE caption.\n\n"
        "Caption as printed in the paper:\n{caption}\n\n"
        "Look at the image. Answer only about what is visibly in it.\n\n"
        "Return ONLY this JSON:\n"
        '{{"shows": "<what the figure actually plots, one sentence, naming '
        'the axis labels and any variable names you can read>", '
        '"mismatch": true|false, '
        '"severity": "critical|major|minor", '
        '"problem": "<one sentence, only if mismatch is true>"}}\n\n'
        "mismatch is true when the caption describes a different plot, "
        "different variables, or a different direction than the image "
        "shows; when the panel is empty or the axes are unlabelled; or "
        "when the image plots variables that the caption does not "
        "mention. It is false when the caption fairly describes the "
        "image, even loosely. Do not comment on aesthetics."
    )

    def _ask_about_figure(self, path: str, caption: str) -> dict | None:
        try:
            raw = self.call_llm(
                self._FIGURE_QUESTION.format(caption=caption or "(none)"),
                max_tokens=700,
                image_paths=[path],
            )
            return json.loads(self._last_json_block(raw))
        except Exception:  # noqa: BLE001 - one bad figure must not end the stage
            return None

    # ------------------------------------------------------------------

    def _persist(self, result: dict, raw: str) -> None:
        try:
            with open(
                os.path.join(self.ctx.output_dir, "verification_report.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(result, f, indent=2, default=str)
            with open(
                os.path.join(self.ctx.output_dir, "verification_raw.txt"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(raw)
        except OSError:
            pass
