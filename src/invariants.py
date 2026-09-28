"""Deterministic invariants over a finished run directory.

Every check here is arithmetic over artifacts the run already wrote. No
LLM, no network, no judgement about what a sentence means. A run either
satisfies an invariant or it does not, and the answer is the same every
time it is asked -- which is the property the reviewers in this pipeline
do not have (test-retest MAD 1.9 on an 8-point scale).

Why this module exists, in one line each:

- The Critic runs BEFORE the Writer and has never seen a manuscript.
- The LSAR gate runs after, is non-blocking, and reads a head-slice of
  the paper with no figures in it.
- 41 Writer-addressed findings across 19 archived runs were written into
  a channel that deletes them, all of them at severity ``minor``.

So nothing in the pipeline ever held the prose against the numbers. That
is what these checks do.

Two rules the checks obey:

1.  **Check in the direction the errors travel.** Walking outward from
    ``results.json`` to disk cannot see a claim that was never in
    ``results.json``. The checks that matter walk from the manuscript
    back to the artifacts.
2.  **Measure what the system produced.** Every path here resolves to
    the as-produced ``paper.tex``, never a repaired or anonymized copy.
    A QA suite pointed at the rebuilt artifact returned all-clear on a
    corpus with 15 confirmed defects.

Severities: ``critical`` means the paper asserts something its own
artifacts refute; ``major`` means a reader is misled about what was
done; ``minor`` is presentation.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

__all__ = [
    "Finding",
    "RunArtifacts",
    "CHECKS",
    "run_invariants",
    "findings_to_json",
]


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class Finding:
    """One invariant violation.

    ``evidence`` carries the numbers that decided it, so a reader can
    re-derive the verdict without re-running anything. ``defect_ids`` are
    the catalogued defects this check was written against -- provenance,
    not proof.
    """

    code: str
    severity: str
    message: str
    artifact: str = ""
    evidence: dict = field(default_factory=dict)
    defect_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["defect_ids"] = list(self.defect_ids)
        return d


# ---------------------------------------------------------------------------
# Artifact bundle
# ---------------------------------------------------------------------------


def _read_json(path: str) -> Any:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _read_text(path: str) -> str | None:
    for enc in ("utf-8", "latin-1"):
        try:
            with open(path, encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
        except OSError:
            return None
    return None


def _read_csv_rows(path: str) -> list[dict] | None:
    try:
        with open(path, encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    except (OSError, ValueError):
        return None


class RunArtifacts:
    """Lazy accessor over one run's output directory.

    Accepts either the run root (which may hold an ``output/`` subdir) or
    the output directory itself.
    """

    #: Preferred first: the file as the system produced it. A ``.tex``
    #: that some later repair step rewrote is a different object.
    PAPER_CANDIDATES = ("paper.tex.orig-latex", "paper.tex")

    def __init__(self, run_dir: str):
        run_dir = os.path.abspath(run_dir)
        nested = os.path.join(run_dir, "output")
        self.run_dir = run_dir
        self.output_dir = nested if os.path.isdir(nested) else run_dir
        self._cache: dict = {}

    # -- generic ---------------------------------------------------------

    def path(self, name: str) -> str:
        return os.path.join(self.output_dir, name)

    def exists(self, name: str) -> bool:
        return os.path.exists(self.path(name))

    def json(self, name: str) -> Any:
        key = ("json", name)
        if key not in self._cache:
            self._cache[key] = _read_json(self.path(name))
        return self._cache[key]

    def text(self, name: str) -> str | None:
        key = ("text", name)
        if key not in self._cache:
            self._cache[key] = _read_text(self.path(name))
        return self._cache[key]

    def csv_rows(self, name: str) -> list[dict] | None:
        key = ("csv", name)
        if key not in self._cache:
            self._cache[key] = _read_csv_rows(self.path(name))
        return self._cache[key]

    def csv_header(self, name: str) -> list[str] | None:
        try:
            with open(self.path(name), encoding="utf-8", newline="") as f:
                return next(csv.reader(f))
        except (OSError, StopIteration, ValueError):
            return None

    # -- named artifacts -------------------------------------------------

    @property
    def results(self) -> dict:
        return self.json("results.json") or {}

    @property
    def data_report(self) -> dict:
        return self.json("data_report.json") or {}

    @property
    def research_spec(self) -> dict:
        return self.json("research_spec.json") or {}

    @property
    def review_report(self) -> dict:
        return self.json("review_report.json") or {}

    @property
    def paper_name(self) -> str | None:
        for cand in self.PAPER_CANDIDATES:
            if self.exists(cand):
                return cand
        return None

    @property
    def paper(self) -> str | None:
        name = self.paper_name
        return self.text(name) if name else None

    @property
    def images_on_disk(self) -> list[str]:
        key = ("images",)
        if key not in self._cache:
            try:
                names = sorted(
                    f
                    for f in os.listdir(self.output_dir)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))
                    and not f.startswith(("lsar_", "_tmp", "thumb_", "paper"))
                )
            except OSError:
                names = []
            self._cache[key] = names
        return self._cache[key]


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------


def _f(x: Any) -> float | None:
    """Coerce to float, or None. ``bool`` is not a number here."""
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) or math.isinf(v) else v


def _bit_equal(a: Any, b: Any) -> bool:
    """Exact float identity -- no tolerance, deliberately.

    These checks fire on *the same computation under two names*, so the
    values agree to the last bit or the check does not apply. A tolerance
    would turn an exact structural signature into a threshold nobody can
    defend.
    """
    fa, fb = _f(a), _f(b)
    return fa is not None and fb is not None and fa == fb


def _model_metrics(results: dict) -> dict[str, dict]:
    models = results.get("all_models")
    if not isinstance(models, dict):
        return {}
    return {k: v for k, v in models.items() if isinstance(v, dict)}


# ---------------------------------------------------------------------------
# Checks -- results.json internals
# ---------------------------------------------------------------------------


#: Prose that reads a SPECIFIC NUMBER as a positive-class quantity.
#:
#: The looser form of this -- any metric word near any at-risk word --
#: fired on a Methods sentence that merely lists which metrics were
#: computed ("F2 weights recall more heavily, which is appropriate for
#: early-warning use"), which is not a misreading of anything. What
#: distinguishes the real defect is a *value* attached to the positive
#: class: "precision (0.628) ... of true dropout episodes", "flags about
#: half of true episodes", "correctly identifies 341 of the 506".
_METRIC_WORD = re.compile(r"\b(?:precision|recall|F1|F2|sensitivity)\b", re.IGNORECASE)
_POSITIVE_FRAMING = re.compile(
    r"(?:true\s+(?:dropout\s+)?(?:case|episode|positive|non-?completer)s?"
    r"|of (?:the )?(?:true|actual)\s+\w+s?\b"
    r"|positive class"
    r"|at-risk students?\b(?=[^.]{0,60}\d)"
    r"|flags?\s+(?:about\s+)?(?:half|a third|two-thirds|most|\d))",
    re.IGNORECASE,
)
_NUMERAL = re.compile(r"\d+(?:[.,]\d+)?")
#: A direct count claim about the positive class needs no metric word.
_COUNT_CLAIM = re.compile(
    r"(?:correctly\s+)?identifies?\s+[\d,]+\s+of\s+(?:the\s+)?[\d,]+",
    re.IGNORECASE,
)


def _sentences(text: str) -> list[str]:
    """Rough sentence split. Good enough to scope a co-occurrence test."""
    return re.split(r"(?<=[.!?])\s+", text)


#: Blocks that are never prose: floats, table bodies and environments
#: LaTeX does not typeset as text.
_NON_PROSE_ENV = re.compile(
    r"(?s)\\begin\s*\{((?:figure|table|tabular|tabularx|longtable|threeparttable"
    r"|wrapfigure|wraptable|sidewaystable|sidewaysfigure|CCSXML|comment"
    r"|verbatim|Verbatim|lstlisting|minted|thebibliography)\*?)\}"
    r".*?\\end\s*\{\1\}"
)
_UNESCAPED_COMMENT = re.compile(r"(?<!\\)%.*")


def _drop_command_with_arg(tex: str, command: str) -> str:
    r"""Remove every ``\command[...]{...}``, matching braces to any depth.

    A caption routinely nests braces (``$n = 3{,}562$``), so a
    ``[^}]*`` pattern stops at the first ``}`` and leaves the rest of the
    caption behind as a fragment of prose.
    """
    out: list[str] = []
    i = 0
    pat = re.compile(r"\\" + command + r"\*?\s*(?:\[[^\]]*\]\s*)*\{")
    while True:
        m = pat.search(tex, i)
        if m is None:
            out.append(tex[i:])
            return "".join(out)
        out.append(tex[i:m.start()])
        depth, j = 1, m.end()
        while j < len(tex) and depth:
            if tex[j] == "\\":
                j += 2
                continue
            depth += {"{": 1, "}": -1}.get(tex[j], 0)
            j += 1
        out.append("\n\n")
        i = j


def _prose_sentences(tex: str) -> list[str]:
    """Sentences of body prose, with nothing but prose in any of them.

    ``_sentences`` splits on terminal punctuation only, so a table ends
    up inside whichever sentence happens to surround it. On one real
    paper the header row ended at "Bal. Acc.", the split fell there, and
    the next "sentence" was every model-name row of the table followed by
    the paper's actual comparison sentence -- so a check read five model
    names into a sentence that names two. Floats, table bodies, captions
    and non-typeset environments are removed first and replaced by a
    paragraph break, and a paragraph break always ends a sentence.
    """
    body = tex.split(r"\begin{document}", 1)[-1]
    body = _UNESCAPED_COMMENT.sub("", body)
    body = _NON_PROSE_ENV.sub("\n\n", body)
    for cmd in ("caption", "Description"):
        body = _drop_command_with_arg(body, cmd)
    out: list[str] = []
    for para in re.split(r"\n\s*\n", body):
        out.extend(s for s in _sentences(para.strip()) if s.strip())
    return out


def _reads_metric_as_positive_class(paper: str) -> str | None:
    """Return the offending sentence, or None.

    A sentence qualifies when it carries a metric word, a numeral and
    positive-class framing together -- or makes a bare count claim.
    """
    for s in _sentences(paper):
        if _COUNT_CLAIM.search(s):
            return s.strip()
        if (
            _METRIC_WORD.search(s)
            and _NUMERAL.search(s)
            and _POSITIVE_FRAMING.search(s)
        ):
            return s.strip()
    return None


def check_macro_metric_mislabel(a: RunArtifacts) -> list[Finding]:
    """A bare ``recall`` equal to ``balanced_accuracy`` bit-for-bit is macro.

    Balanced accuracy *is* macro-averaged recall. For a binary problem
    with any class imbalance, positive-class recall equalling it to 16
    significant digits is not a coincidence; ``average='macro'`` was
    passed and the value stored under a name every reader takes for the
    positive class.

    Two severities, and the distinction is the whole false-positive
    story. The artifact condition alone is a *latent* one: measured over
    the audited archive it holds on 3 runs, and on 1 of them the paper
    lists the metrics once in Methods and never interprets them, so
    nothing is actually wrong with that manuscript. It becomes critical
    only when the prose reads one of those numbers as a property of the
    positive class -- "flags about half of true episodes", "correctly
    identifies 341 of the 506". Without a manuscript this check reports
    the latent condition and says so.
    """
    out: list[Finding] = []
    paper = a.paper
    offending = _reads_metric_as_positive_class(paper) if paper else None
    reads_as_positive = offending is not None
    for name, m in _model_metrics(a.results).items():
        if "recall" not in m or "balanced_accuracy" not in m:
            continue
        if not _bit_equal(m["recall"], m["balanced_accuracy"]):
            continue
        if reads_as_positive:
            sev = "critical"
            tail = (
                " The manuscript reads one of these as a positive-class "
                "quantity, so it is wrong by roughly the class imbalance."
            )
        else:
            sev = "minor"
            tail = (
                " The manuscript does not appear to interpret them as "
                "positive-class quantities, so this is a latent condition: "
                "fix the naming before anyone does."
            )
        out.append(
            Finding(
                code="INV_MACRO_METRIC_MISLABEL",
                severity=sev,
                message=(
                    f"{name}: results.json stores recall == balanced_accuracy "
                    f"({m['recall']!r}) to full float precision. Balanced "
                    "accuracy is macro-averaged recall, so this 'recall' -- "
                    "and by construction the sibling 'precision' and 'f1' -- "
                    "are macro averages under positive-class names."
                    + tail
                    + " Use classification_metrics(), which reports *_macro "
                    "and *_positive separately."
                ),
                artifact="results.json" + (" + " + (a.paper_name or "") if paper else ""),
                evidence={
                    "model": name,
                    "recall": m["recall"],
                    "balanced_accuracy": m["balanced_accuracy"],
                    "precision": m.get("precision"),
                    "f1": m.get("f1"),
                    "manuscript_reads_as_positive_class": reads_as_positive,
                    "offending_sentence": (offending or "")[:300] or None,
                },
                defect_ids=("J02", "J03", "J04", "J05", "J13") if reads_as_positive else (),
            )
        )
    return out


#: A sentence that reports the paired model comparison by name.
_TEST_REPORT = re.compile(r"\bAUC difference\b|\bnext-best\b|\bnext best\b", re.IGNORECASE)
#: What turns a sentence holding the rounded difference into a report of
#: the test rather than a coincidence of digits.
_COMPARISON_CUE = re.compile(
    r"differen|compar|bootstrap|\\Delta|\bdelta\b|\bversus\b|\bvs\b|outperform"
    r"|runner-up|\bpaired\b|significan|exceed|advantage|margin|improv"
    r"|\b(?:higher|lower|better|worse|greater|smaller) than\b",
    re.IGNORECASE,
)


def _model_name_pattern(name: str) -> re.Pattern:
    """Match a results.json model key however prose spells it.

    Keys are CamelCase identifiers; prose writes "Logistic Regression",
    "random forest", "Stacking Ensemble". Matching the key literally saw
    only XGBoost in "The paired cluster-bootstrap test of the AUC
    difference between Logistic Regression and the runner-up (Random
    Forest)", the one sentence in which a paper named the wrong
    comparator. Word boundaries keep a short key from matching inside a
    longer word.
    """
    parts: list[str] = []
    for chunk in re.split(r"[^A-Za-z0-9]+", name):
        parts += re.findall(r"[A-Z]+(?=[A-Z][a-z]|\d|$)|[A-Z]?[a-z]+|[A-Z]+|\d+", chunk)
    body = r"[\s\-]*".join(re.escape(p) for p in parts) if parts else re.escape(name)
    return re.compile(r"(?<![A-Za-z0-9])" + body + r"s?(?![A-Za-z0-9])", re.IGNORECASE)


def check_comparator_unnamed(a: RunArtifacts) -> list[Finding]:
    """The comparison the paper names is not the comparison that was run.

    Two findings, and keeping them apart matters. Asking merely "does
    results.json name the comparands?" fires on 17 of 18 archived runs,
    because the helper had no such parameter until 2026-09-20 -- a
    17/18 base rate carries almost no information, so that half is
    ``minor`` and exists to stop the metadata gap recurring.

    The discriminating form binds: recover the pair whose AUC difference
    reproduces ``auc_diff`` bit-for-bit, then check it against the model
    the manuscript names. Two delivered papers named RandomForest as the
    tested runner-up while the stored difference was XGBoost minus
    LogisticRegression in one and the reverse in the other, and in both
    it was the paper's only inferential test.
    """
    mct = a.results.get("model_comparison_test")
    if not isinstance(mct, dict):
        return []

    diff = _f(mct.get("auc_diff"))
    metrics = _model_metrics(a.results)
    matches: list[tuple[str, str]] = []
    if diff is not None:
        for na, ma in metrics.items():
            for nb, mb in metrics.items():
                if na == nb:
                    continue
                va, vb = _f(ma.get("auc")), _f(mb.get("auc"))
                if va is None or vb is None:
                    continue
                if va - vb == diff:
                    matches.append((na, nb))

    named = bool(mct.get("model_a")) and bool(mct.get("model_b"))
    truth: tuple[str, str] | None = None
    if named:
        truth = (str(mct["model_a"]), str(mct["model_b"]))
    elif len(matches) == 1:
        truth = matches[0]

    out: list[Finding] = []

    # -- the binding half -------------------------------------------------
    paper = a.paper
    if paper and truth:
        # Scan EVERY sentence that reports this test, not the first
        # regex hit. On one real paper the first match was an unrelated
        # block-incremental sentence and the sentence that actually names
        # the comparator ("...between XGBoost and the next-best
        # individual model (RandomForest)...") came later.
        #
        # Sentences come from prose only. A results table sitting just
        # above the comparison sentence used to be split into it, and the
        # table's model-name rows were read as models the sentence names.
        rounded = {f"{abs(diff):.{k}f}" for k in (3, 4)} if diff is not None else set()
        candidates = [
            s
            for s in _prose_sentences(paper)
            if _TEST_REPORT.search(s)
            # The rounded value alone is not enough: "All five models
            # performed within 0.010 AUC of one another: ..." lists every
            # model beside a number that happens to equal the difference,
            # and reports no test.
            or (any(r in s for r in rounded) and _COMPARISON_CUE.search(s))
        ]
        patterns = {name: _model_name_pattern(name) for name in metrics}
        for window in candidates:
            named_models = [
                name for name, pat in patterns.items() if pat.search(window)
            ]
            wrong = [n for n in named_models if n not in truth]
            if wrong:
                out.append(
                    Finding(
                        code="INV_COMPARATOR_MISNAMED",
                        severity="critical",
                        message=(
                            f"The manuscript's model-comparison sentence names "
                            f"{', '.join(wrong)}, but auc_diff="
                            f"{mct.get('auc_diff')!r} is {truth[0]} minus "
                            f"{truth[1]} to the last floating-point digit. The "
                            "test the paper reports is not the test that was "
                            "run."
                        ),
                        artifact=f"results.json + {a.paper_name}",
                        evidence={
                            "auc_diff": mct.get("auc_diff"),
                            "actual_contrast": list(truth),
                            "named_in_paper": named_models,
                            "sentence": window.strip()[:400],
                        },
                        defect_ids=("J06", "J53", "J66"),
                    )
                )
                break  # one finding per run; the rest say the same thing

    # -- the metadata half ------------------------------------------------
    if not named or mct.get("comparands_unnamed"):
        if diff is None:
            msg = (
                "results.json has a model_comparison_test block with no "
                "auc_diff value and no model_a/model_b. Nothing about this "
                "comparison can be reported: neither the models nor the "
                "difference exists in the artifact."
            )
            sev = "major"
        elif len(matches) == 1:
            msg = (
                f"results.json records auc_diff={mct.get('auc_diff')!r} with no "
                f"model_a/model_b. It resolves uniquely, bit-for-bit, to "
                f"{matches[0][0]} minus {matches[0][1]} -- but nothing in the "
                "run says so, so any prose naming a comparator is a guess. "
                "Pass model_a/model_b to bootstrap_auc_difference."
            )
            sev = "minor"
        else:
            msg = (
                f"results.json records auc_diff={mct.get('auc_diff')!r} with no "
                f"model_a/model_b and it does not resolve to a unique model "
                f"pair ({len(matches)} candidates). The comparison cannot be "
                "written up truthfully from this artifact."
            )
            sev = "major"
        out.append(
            Finding(
                code="INV_COMPARATOR_UNNAMED",
                severity=sev,
                message=msg,
                artifact="results.json",
                evidence={
                    "auc_diff": mct.get("auc_diff"),
                    "recovered_pairs": matches,
                },
            )
        )
    return out


def check_subgroup_false_unavailable(a: RunArtifacts) -> list[Finding]:
    """A warning says a protected attribute is absent; the file has it.

    This is the shape that reaches papers as a *data* limitation. It is
    not one. ``test_protected.csv`` was written with ``index=False`` and
    read with ``index_col=0``, which ate the first column.
    """
    header = a.csv_header("test_protected.csv")
    if not header:
        return []
    present = {h.strip() for h in header}
    warnings = a.results.get("warnings") or []
    if not isinstance(warnings, list):
        warnings = [str(warnings)]
    paper = a.paper
    out: list[Finding] = []
    for w in warnings:
        text = str(w)
        low = text.lower()
        if "not found" not in low and "not available" not in low:
            continue
        # The fixed helper labels its own message PIPELINE: ... NOT of the
        # dataset; that form is the disclosure, not the defect.
        if text.startswith("PIPELINE:"):
            continue
        for col in present:
            if not col or col not in text:
                continue
            # Did the false claim reach the manuscript? That is the
            # difference between a stale warning and a paper asserting
            # something untrue about NCES data.
            in_paper = bool(
                paper
                and re.search(
                    re.escape(col)
                    + r"[^.]{0,200}?(?:not (?:available|carried|present|included)"
                    r"|unavailable|could not be computed)",
                    paper,
                    re.IGNORECASE,
                )
            )
            out.append(
                Finding(
                    code="INV_SUBGROUP_FALSE_UNAVAILABLE",
                    severity="critical",
                    message=(
                        f"results.json warns that {col!r} was unavailable, but "
                        f"{col!r} is a column of test_protected.csv. The "
                        "variable was present; a reader bug lost it."
                        + (
                            " The manuscript repeats this as a limitation, so "
                            "the paper asserts something false about the "
                            "dataset."
                            if in_paper
                            else " The manuscript does not appear to repeat it; "
                            "fix the warning before it is transcribed."
                        )
                    ),
                    artifact="results.json + test_protected.csv"
                    + (f" + {a.paper_name}" if in_paper else ""),
                    evidence={
                        "warning": text[:300],
                        "columns_present": sorted(present),
                        "repeated_in_manuscript": in_paper,
                    },
                    defect_ids=("J35", "J55", "J63") if in_paper else (),
                )
            )
            break
    return out


#: Above this many levels a variable is continuous, not categorical.
#: HSLS's largest genuine categorical (X1FAMINCOME) has 13.
_MAX_CATEGORICAL_LEVELS = 50


def check_dummy_cardinality(a: RunArtifacts) -> list[Finding]:
    """A one-hot group with the wrong number of columns for its variable.

    ``get_dummies(drop_first=True)`` must leave exactly ``k - 1`` columns
    for a ``k``-category variable. Fewer means a category vanished
    between the split that fitted the encoding and the split that
    received it -- which is also the mechanism that slides every
    surviving column onto the wrong label. In one delivered paper the
    column named ``BYRACE_Multiracial`` held 1,479 White students and
    zero multiracial ones, and the paper built a substantive racial
    finding and a partial-dependence figure on it.

    The truth is ``data_report.encoding_report`` when the run wrote one
    (``encode_categoricals`` does); otherwise fall back to the distinct
    values in ``test_protected.csv``.
    """
    header = a.csv_header("test_X.csv")
    if not header:
        return []
    out: list[Finding] = []

    enc = (a.data_report or {}).get("encoding_report")
    if isinstance(enc, dict) and enc:
        for src, entry in enc.items():
            if not isinstance(entry, dict):
                continue
            cats = entry.get("categories_fitted_on_train") or []
            expected = max(0, len(cats) - (1 if entry.get("drop_first", True) else 0))
            actual = sum(1 for h in header if h.startswith(f"{src}_"))
            if expected and actual != expected:
                out.append(
                    Finding(
                        code="INV_DUMMY_CARDINALITY",
                        severity="critical",
                        message=(
                            f"{src!r} has {len(cats)} categories and "
                            f"drop_first={entry.get('drop_first')}, so test_X "
                            f"should carry {expected} dummy column(s); it "
                            f"carries {actual}. A missing column means the "
                            "remaining ones are shifted onto the wrong "
                            "category labels."
                        ),
                        artifact="data_report.json + test_X.csv",
                        evidence={"variable": src, "categories": cats,
                                  "expected": expected, "actual": actual},
                        defect_ids=("J50", "J51", "J56", "J64"),
                    )
                )
        return out

    rows = a.csv_rows("test_protected.csv")
    if not rows:
        return []
    for src in {h.split("_")[0] for h in header if "_" in h}:
        if src not in (rows[0] or {}):
            continue
        cats = {str(r.get(src)).strip() for r in rows if str(r.get(src)).strip()}
        cats.discard("nan")
        actual = sum(1 for h in header if h.startswith(f"{src}_"))
        if len(cats) < 2 or actual == 0:
            continue
        if len(cats) > _MAX_CATEGORICAL_LEVELS:
            # Not a cardinality mismatch -- a CONTINUOUS variable that
            # was one-hot encoded. Three archived runs fire here on
            # X1SES, a socio-economic composite with 3,109 distinct
            # values and 8,435 dummy columns. Reporting that as "should
            # carry 3,109 dummy columns" is advice nobody should take.
            out.append(
                Finding(
                    code="INV_CONTINUOUS_ONE_HOT_ENCODED",
                    severity="critical",
                    message=(
                        f"{src!r} takes {len(cats)} distinct values and has "
                        f"{actual} dummy column(s) in test_X. A variable with "
                        "that many levels is continuous; one-hot encoding it "
                        "produces a matrix no model can train on. Decide "
                        "categorical vs continuous from the DECLARED TYPE in "
                        "the dataset registry, not from the pandas dtype "
                        "after sentinel replacement."
                    ),
                    artifact="test_protected.csv + test_X.csv",
                    evidence={"variable": src, "n_distinct_values": len(cats),
                              "dummy_columns": actual},
                )
            )
            continue
        if actual not in (len(cats), len(cats) - 1):
            out.append(
                Finding(
                    code="INV_DUMMY_CARDINALITY",
                    severity="critical",
                    message=(
                        f"{src!r} takes {len(cats)} distinct values in "
                        f"test_protected.csv, so test_X should carry "
                        f"{len(cats)} or {len(cats) - 1} dummy column(s); it "
                        f"carries {actual}. A missing column means the "
                        "remaining ones are shifted onto the wrong category "
                        "labels."
                    ),
                    artifact="test_protected.csv + test_X.csv",
                    evidence={"variable": src, "distinct_values": sorted(cats),
                              "expected": [len(cats), len(cats) - 1],
                              "actual": actual},
                    defect_ids=("J50", "J51", "J56", "J64"),
                )
            )
    return out


def check_group_label_missing_as_level(a: RunArtifacts) -> list[Finding]:
    """A missing value carried into a results dict as a group label.

    ``str(nan)`` is ``'nan'``, and a groupby then treats it as a band.
    One delivered invariance ladder held most of its cases in such a
    phantom group.
    """
    bad = {"nan", "none", "<na>", "na", "missing", "unit non-response"}
    out: list[Finding] = []

    def walk(node: Any, trail: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(k, str) and k.strip().lower() in bad:
                    n = v.get("n") if isinstance(v, dict) else None
                    out.append(
                        Finding(
                            code="INV_GROUP_LABEL_IS_MISSING",
                            severity="major",
                            message=(
                                f"results.json{trail} contains a group keyed "
                                f"{k!r}"
                                + (f" holding n={n}" if n is not None else "")
                                + ". A missing value was stringified into a "
                                "level and is being reported as though it "
                                "were a real group."
                            ),
                            artifact="results.json",
                            evidence={"path": trail, "key": k, "value": v
                                      if not isinstance(v, (dict, list)) else str(v)[:200]},
                            defect_ids=(),
                        )
                    )
                walk(v, f"{trail}[{k!r}]")
        elif isinstance(node, list):
            for i, v in enumerate(node[:200]):
                walk(v, f"{trail}[{i}]")

    walk(a.results, "")
    return out


#: CSV cells this check has to read. pandas writes boolean dummy columns
#: as ``True``/``False``, not 0/1, and a parser that only accepts floats
#: silently classifies every one of them as unreadable.
_BOOL_CELLS = {"true": 1.0, "false": 0.0, "t": 1.0, "f": 0.0}


def _cell_value(cell: str) -> float | None:
    """Numeric value of a CSV cell, or None when it is not numeric."""
    v = _f(cell)
    if v is not None:
        return v
    return _BOOL_CELLS.get(str(cell).strip().lower())


def check_degenerate_feature_weight(a: RunArtifacts) -> list[Finding]:
    """A feature that is constant in the test set carrying SHAP weight.

    SHAP values are computed on the test matrix. A column with no
    variance there cannot move a prediction, so a non-zero mean |SHAP|
    attached to it means the column is not what its name says.
    """
    rows = a.csv_rows("feature_importance.csv")
    header = a.csv_header("test_X.csv")
    if not rows or not header:
        return []

    weights: dict[str, float] = {}
    for r in rows:
        name = r.get("feature") or r.get("Feature") or r.get("")
        val = None
        for key in ("shap_mean_abs", "mean_abs_shap", "importance", "shap"):
            if key in r:
                val = _f(r[key])
                break
        if name and val is not None:
            weights[str(name).strip()] = val
    if not weights:
        return []

    # Column sums over test_X, streamed (these files reach ~10 MB).
    try:
        with open(a.path("test_X.csv"), encoding="utf-8", newline="") as f:
            rd = csv.reader(f)
            cols = next(rd)
            idx = {c: i for i, c in enumerate(cols) if c in weights}
            if not idx:
                return []
            sums = {c: 0.0 for c in idx}
            uniq: dict[str, set] = {c: set() for c in idx}
            unparsed: set[str] = set()
            n = 0
            for row in rd:
                n += 1
                for c, i in idx.items():
                    v = _cell_value(row[i]) if i < len(row) else None
                    if v is None:
                        unparsed.add(c)
                        continue
                    sums[c] += v
                    if len(uniq[c]) < 3:
                        uniq[c].add(v)
    except (OSError, StopIteration, ValueError):
        return []
    if not n:
        return []

    out: list[Finding] = []
    for c in idx:
        if len(uniq[c]) > 1:
            continue
        if c in unparsed:
            # A column whose values this check could not read is not a
            # constant column. pandas writes one-hot dummies as the
            # strings True/False, which the float parser rejected --
            # leaving `uniq` empty and `sums` at 0.0, so a perfectly
            # varying column was reported as "constant across all 4,697
            # test rows (sum 0.0)". One archived run produced 37 such
            # findings, one per dummy, with exactly one of them real.
            continue
        if weights.get(c, 0.0) == 0.0:
            continue
        out.append(
            Finding(
                code="INV_DEGENERATE_FEATURE_WEIGHT",
                severity="major",
                message=(
                    f"Column {c!r} is constant across all {n} test rows "
                    f"(sum {sums[c]!r}) yet carries mean |SHAP| "
                    f"{weights[c]!r}. A constant column cannot move a test "
                    "prediction; the encoding produced a column that does not "
                    "mean what its name says, and any feature-importance "
                    "sentence about it is about an artifact of the encoding."
                ),
                artifact="feature_importance.csv + test_X.csv",
                evidence={
                    "column": c,
                    "n_test_rows": n,
                    "column_sum": sums[c],
                    "shap_mean_abs": weights[c],
                },
                defect_ids=("J65",),
            )
        )
    return out


def check_shap_rank_disagreement(a: RunArtifacts) -> list[Finding]:
    """Grouped and per-column SHAP rankings disagree about the same feature.

    ``top_feature_groups`` ranks variables; ``feature_importance.csv``
    ranks encoded columns. A superlative ("the weakest predictor", "the
    second-lowest") is true in at most one of them, and the paper never
    says which it means.
    """
    groups = a.results.get("top_feature_groups")
    rows = a.csv_rows("feature_importance.csv")
    if not isinstance(groups, dict) or not groups or not rows:
        return []

    # feature_importance.csv is frequently a top-N truncation, not the
    # full column list. Every firing of this check over the audited
    # archive traced to that truncation and none to a real disagreement,
    # so compare rankings only when the CSV covers every encoded column.
    header = a.csv_header("test_X.csv")
    if not header or len(rows) < len(header):
        return []

    g_sorted = sorted(
        ((k, _f(v)) for k, v in groups.items() if _f(v) is not None),
        key=lambda kv: kv[1],
    )
    if len(g_sorted) < 2:
        return []
    col_vals: list[tuple[str, float]] = []
    for r in rows:
        name = r.get("feature") or r.get("Feature")
        val = None
        for key in ("shap_mean_abs", "mean_abs_shap", "importance", "shap"):
            if key in r:
                val = _f(r[key])
                break
        if name and val is not None:
            col_vals.append((str(name).strip(), val))
    if len(col_vals) < 2:
        return []
    col_sorted = sorted(col_vals, key=lambda kv: kv[1])

    lowest_group = g_sorted[0][0]
    # Where does that group's *best* column rank among all columns?
    member_ranks = [
        i for i, (c, _) in enumerate(col_sorted)
        if c == lowest_group or c.startswith(lowest_group + "_")
    ]
    if not member_ranks:
        return []
    best_rank = min(member_ranks) + 1  # 1 = lowest importance
    if best_rank <= 2:
        return []
    return [
        Finding(
            code="INV_SHAP_RANK_DISAGREEMENT",
            severity="minor",
            message=(
                f"{lowest_group!r} is the lowest-importance entry in "
                f"results.top_feature_groups (n={len(g_sorted)} groups) but "
                f"ranks {best_rank} of {len(col_sorted)} from the bottom in "
                "feature_importance.csv. A superlative about this variable is "
                "true of one ranking and false of the other; the paper must "
                "name which ranking it is quoting."
            ),
            artifact="results.json + feature_importance.csv",
            evidence={
                "group": lowest_group,
                "group_rank_from_bottom": 1,
                "n_groups": len(g_sorted),
                "column_rank_from_bottom": best_rank,
                "n_columns": len(col_sorted),
            },
            defect_ids=("J21", "J38", "J60", "J72"),
        )
    ]


def check_omega_unidimensional_pooling(a: RunArtifacts) -> list[Finding]:
    """An omega-total over a multi-factor model with no factor covariances.

    Pooling every loading into one omega is the unidimensional formula,
    i.e. an unstated assumption that the factors correlate 1.0.
    """
    out: list[Finding] = []

    def walk(node: Any, trail: str) -> None:
        if isinstance(node, dict):
            if "omega_total" in node and _f(node.get("omega_total")) is not None:
                loads = node.get("from_loadings")
                nfac = node.get("n_factors")
                by_factor = node.get("omega_by_factor")
                multi = (isinstance(nfac, int) and nfac > 1) or (
                    isinstance(by_factor, dict) and len(by_factor) > 1
                )
                if multi and not node.get("factor_cor_used"):
                    out.append(
                        Finding(
                            code="INV_OMEGA_UNIDIMENSIONAL_POOLING",
                            severity="critical",
                            message=(
                                f"results.json{trail} reports omega_total="
                                f"{node['omega_total']!r} over "
                                f"{nfac or len(by_factor or {})} factors with "
                                "no factor covariance matrix recorded. That is "
                                "the unidimensional formula, which assumes the "
                                "factors correlate 1.0. Report per-factor omega "
                                "or supply Phi."
                            ),
                            artifact="results.json",
                            evidence={
                                "path": trail,
                                "omega_total": node["omega_total"],
                                "n_factors": nfac,
                                "n_loadings": len(loads) if isinstance(loads, list) else None,
                            },
                            defect_ids=(),
                        )
                    )
            for k, v in node.items():
                walk(v, f"{trail}[{k!r}]")
        elif isinstance(node, list):
            for i, v in enumerate(node[:200]):
                walk(v, f"{trail}[{i}]")

    walk(a.results, "")
    return out


def check_bootstrap_ci_resamples(a: RunArtifacts) -> list[Finding]:
    """A percentile interval reported from too few effective resamples.

    A 95% percentile interval needs the 2.5% tail to exist. Below ~40
    resamples it does not, and the Monte-Carlo error on the width is
    larger than the width's own reported precision.
    """
    out: list[Finding] = []
    MIN = 40

    def walk(node: Any, trail: str) -> None:
        if isinstance(node, dict):
            n = None
            for key in ("n_boot_effective", "n_successful_boot", "n_resamples_effective"):
                if key in node:
                    n = _f(node[key])
                    break
            has_ci = any(
                k in node for k in ("ci_lower", "ci_upper", "rmse_ci_lower", "auc_ci_lower")
            )
            if n is not None and has_ci and n < MIN:
                out.append(
                    Finding(
                        code="INV_BOOTSTRAP_TOO_FEW_RESAMPLES",
                        severity="major",
                        message=(
                            f"results.json{trail} reports a bootstrap interval "
                            f"from {int(n)} effective resamples. A 95% "
                            f"percentile interval has no defined 2.5% tail "
                            f"below ~{MIN}, and the Monte-Carlo error on the "
                            "interval exceeds the precision it is quoted to. "
                            "Report the resample count beside the interval, or "
                            "do not report the interval."
                        ),
                        artifact="results.json",
                        evidence={"path": trail, "n_boot_effective": n,
                                  "ci_lower": node.get("ci_lower"),
                                  "ci_upper": node.get("ci_upper")},
                        defect_ids=(),
                    )
                )
            for k, v in node.items():
                walk(v, f"{trail}[{k!r}]")
        elif isinstance(node, list):
            for i, v in enumerate(node[:200]):
                walk(v, f"{trail}[{i}]")

    walk(a.results, "")
    return out


def check_estimand_population_mismatch(a: RunArtifacts) -> list[Finding]:
    """``estimand_check.match`` true while the estimators ran on different n.

    Comparing the declared estimand *string* across estimators cannot see
    that one of them discarded 31% of the treated units. Five estimates
    defined on three populations are not five estimates of one quantity.
    """
    chk = a.results.get("causal_estimand_check")
    if not isinstance(chk, dict) or not chk.get("match"):
        return []

    ests = a.results.get("estimates") or a.results.get("all_estimators") or {}
    if not isinstance(ests, dict):
        return []
    ns: dict[str, float] = {}
    for name, e in ests.items():
        if not isinstance(e, dict):
            continue
        for key in ("n", "n_analytic", "n_used", "n_effective", "n_matched"):
            v = _f(e.get(key))
            if v is not None:
                ns[name] = v
                break
    distinct = sorted(set(ns.values()))
    if len(distinct) < 2:
        return []
    return [
        Finding(
            code="INV_ESTIMAND_POPULATION_MISMATCH",
            severity="critical",
            message=(
                f"causal_estimand_check.match is true, but the estimators ran "
                f"on {len(distinct)} different sample sizes "
                f"({', '.join(str(int(d)) for d in distinct)}). The check "
                "compares a declared estimand label, not the population each "
                "estimator actually targets. Estimates on different "
                "populations are not evidence of divergence between methods."
            ),
            artifact="results.json",
            evidence={"per_estimator_n": ns, "declared": chk.get("declared")},
            defect_ids=(),
        )
    ]


def check_post_match_balance_worse(a: RunArtifacts) -> list[Finding]:
    """Matching that made balance worse is a bug, not a caliper finding.

    A caliper limits which pairs form; it cannot degrade the balance of
    the pairs it does form.
    """
    out: list[Finding] = []

    def walk(node: Any, trail: str) -> None:
        if isinstance(node, dict):
            pre = None
            post = None
            for k, v in node.items():
                lk = k.lower()
                if "smd" in lk and "pre" in lk:
                    pre = _f(v)
                if "smd" in lk and "post" in lk:
                    post = _f(v)
            if pre is not None and post is not None and post > max(pre, 0.25):
                out.append(
                    Finding(
                        code="INV_POST_MATCH_BALANCE_WORSE",
                        severity="critical",
                        message=(
                            f"results.json{trail}: post-match max SMD {post} "
                            f"exceeds both the pre-match {pre} and the 0.25 "
                            "convention. Matching cannot degrade the balance "
                            "of the pairs it forms, so this is a computation "
                            "fault -- a near-zero-variance one-hot column in "
                            "the SMD denominator is the usual cause. It must "
                            "be re-run, not glossed as 'the caliper was too "
                            "tight'."
                        ),
                        artifact="results.json",
                        evidence={"path": trail, "smd_pre": pre, "smd_post": post},
                        defect_ids=(),
                    )
                )
            for k, v in node.items():
                walk(v, f"{trail}[{k!r}]")
        elif isinstance(node, list):
            for i, v in enumerate(node[:200]):
                walk(v, f"{trail}[{i}]")

    walk(a.results, "")
    return out


# ---------------------------------------------------------------------------
# Checks -- manuscript against artifacts
# ---------------------------------------------------------------------------

_INCLUDEGRAPHICS = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")

#: Any citation command, not only ``\cite``. The first version of this
#: matched commands *beginning* with "cite", and these manuscripts load
#: biblatex and write ``\parencite`` -- so it found zero citations and
#: reported every bibliography as 100% uncited. A citation detector blind
#: to the citation macro its own template loads is the same class of
#: error it exists to catch. Covers \cite \citep \citet \parencite
#: \textcite \autocite \footcite \nocite and their starred forms.
_CITE_CMD = re.compile(r"\\[a-zA-Z]*cite[a-zA-Z]*\*?\s*(?:\[[^\]]*\]\s*)*\{([^}]*)\}")


def check_figures_orphaned(a: RunArtifacts) -> list[Finding]:
    """Figures produced by the analysis that the paper never shows.

    Only the ZERO-figures form is a defect. Measured over the audited
    archive, "some PNG is unembedded" fires on 4 runs and is right about
    1: leaving two of five partial-dependence plots out is ordinary
    editorial selection, not a failure, and treating it as one buries the
    case that matters. A paper that embeds *nothing* while its analysis
    produced figures is different in kind -- the causal run wrote
    ``love_plot.png``, ``propensity_overlap.png`` and
    ``cate_distribution.png``, its revision cycle orphaned all three, and
    the delivered paper had no balance plot despite its own checklist
    requiring one and its headline number being a balance statistic.
    """
    paper = a.paper
    if paper is None:
        return []
    on_disk = a.images_on_disk
    if not on_disk:
        return []
    embedded = {
        os.path.basename(m).split(".")[0] for m in _INCLUDEGRAPHICS.findall(paper)
    }
    orphans = [f for f in on_disk if f.split(".")[0] not in embedded]
    if not orphans:
        return []
    if embedded:
        return [
            Finding(
                code="INV_FIGURES_PARTIALLY_ORPHANED",
                severity="minor",
                message=(
                    f"{len(orphans)} of {len(on_disk)} figure(s) in the output "
                    f"directory are not embedded in {a.paper_name} "
                    f"(the paper embeds {len(embedded)}): "
                    f"{', '.join(orphans[:6])}"
                    + (" ..." if len(orphans) > 6 else "")
                    + ". Usually editorial selection; check that nothing "
                    "load-bearing was dropped."
                ),
                artifact=f"{a.paper_name} + output/*.png",
                evidence={"on_disk": on_disk, "embedded": sorted(embedded),
                          "orphaned": orphans},
            )
        ]
    return [
        Finding(
            code="INV_FIGURES_ORPHANED",
            severity="critical",
            message=(
                f"{a.paper_name} embeds NO figures at all, while "
                f"{len(on_disk)} figure(s) sit in the output directory: "
                f"{', '.join(on_disk[:8])}"
                + (" ..." if len(on_disk) > 8 else "")
                + ". The analysis computed these and the reader never sees "
                "one of them."
            ),
            artifact=f"{a.paper_name} + output/*.png",
            evidence={"on_disk": on_disk, "embedded": [], "orphaned": orphans},
            defect_ids=("T15",),
        )
    ]


def check_figures_claimed_absent(a: RunArtifacts) -> list[Finding]:
    """The paper embeds an image file that is not on disk."""
    paper = a.paper
    if paper is None:
        return []
    refs = _INCLUDEGRAPHICS.findall(paper)
    if not refs:
        return []
    stems = {f.split(".")[0] for f in a.images_on_disk}
    missing = sorted(
        {r for r in refs if os.path.basename(r).split(".")[0] not in stems}
    )
    if not missing:
        return []
    return [
        Finding(
            code="INV_FIGURE_FILE_MISSING",
            severity="critical",
            message=(
                f"{a.paper_name} includes {len(missing)} image(s) with no file "
                f"in the output directory: {', '.join(missing[:8])}"
                + (" ..." if len(missing) > 8 else "")
                + ". Either the figure was never produced or it was produced "
                "under another name."
            ),
            artifact=a.paper_name or "paper.tex",
            evidence={"referenced_missing": missing, "on_disk": a.images_on_disk},
        )
    ]


def check_duplicate_figure_images(a: RunArtifacts) -> list[Finding]:
    """The same image file used under more than one figure number."""
    paper = a.paper
    if paper is None:
        return []
    refs = [os.path.basename(r).split(".")[0] for r in _INCLUDEGRAPHICS.findall(paper)]
    dupes = sorted({r for r in refs if refs.count(r) > 1})
    if not dupes:
        return []
    return [
        Finding(
            code="INV_DUPLICATE_FIGURE_IMAGE",
            severity="major",
            message=(
                f"{a.paper_name} uses the same image under more than one "
                f"figure: {', '.join(dupes)}. At least one caption describes "
                "something the reader is not being shown."
            ),
            artifact=a.paper_name or "paper.tex",
            evidence={"duplicated": dupes, "all_references": refs},
        )
    ]


_PROSE_AUTHORITY = re.compile(
    r"([A-Z][A-Za-z\u00C0-\u024F'`-]+(?:\s+(?:and|&|et\s+al\.?)\s*"
    r"[A-Z][A-Za-z\u00C0-\u024F'`-]*)*)"
    r"(?:'s)?\s*\(\s*((?:19|20)\d{2})[a-z]?\s*\)"
)


def check_prose_authority_uncited(a: RunArtifacts) -> list[Finding]:
    """An author-year authority typed as prose, with no bibliography entry.

    A ``\\cite`` key parser has zero sensitivity to this. One paper
    invoked "Chen's (2007)" seven times as the warrant for every
    invariance verdict in it, with no Chen 2007 anywhere in a 34-entry
    bibliography.
    """
    paper = a.paper
    if paper is None:
        return []
    bib = a.text("references.bib") or ""
    body = re.sub(r"(?s)\\begin\{thebibliography\}.*?\\end\{thebibliography\}", "", paper)

    counts: dict[tuple[str, str], int] = {}
    for m in _PROSE_AUTHORITY.finditer(body):
        start = max(0, m.start() - 12)
        if "cite" in body[start:m.start()]:
            continue
        # Strip a possessive: "Chen's (2007)" and "Chen (2007)" are the
        # same authority, and the surname character class eats the
        # apostrophe, so without this they become two findings and
        # neither matches a bibliography entry for Chen.
        surname = re.sub(r"[’']s?$", "", m.group(1).split()[0]).strip("'’`-")
        if len(surname) < 3 or surname.lower() in _NON_SURNAMES:
            continue
        counts[(surname, m.group(2))] = counts.get((surname, m.group(2)), 0) + 1

    out: list[Finding] = []
    bib_lower = bib.lower()
    for (surname, year), n in sorted(counts.items(), key=lambda kv: -kv[1]):
        if surname.lower() in bib_lower and year in bib:
            continue
        out.append(
            Finding(
                code="INV_PROSE_AUTHORITY_UNCITED",
                severity="major" if n >= 3 else "minor",
                message=(
                    f"{a.paper_name} invokes '{surname} ({year})' {n} time(s) "
                    "as bare prose with no \\cite and no matching "
                    "references.bib entry. A key-parsing citation check cannot "
                    "see this; a reviewer can."
                ),
                artifact=f"{a.paper_name} + references.bib",
                evidence={"surname": surname, "year": year, "occurrences": n},
                defect_ids=("T12",),
            )
        )
    return out


#: Sentence-initial words and LaTeX/section nouns that the author-year
#: regex would otherwise take for surnames.
_NON_SURNAMES = frozenset(
    {
        "the", "this", "these", "those", "and", "but", "for", "our", "their",
        "table", "figure", "section", "appendix", "model", "models", "study",
        "studies", "data", "dataset", "results", "analysis", "chapter",
        "equation", "however", "although", "because", "since", "while",
        "using", "based", "given", "such", "both", "each", "some", "all",
        "in", "of", "to", "by", "on", "at", "we", "it", "as", "from", "with",
        "wave", "cohort", "year", "grade", "public", "national", "first",
        "second", "third", "prior", "recent", "here", "there", "one", "two",
    }
)


def check_unverified_block_present(a: RunArtifacts) -> list[Finding]:
    """A non-PASS run whose manuscript carries no warning block.

    SPEC 4.5 requires the block whenever the verdict is not PASS. The
    flag lives only in ``checkpoint.json``; ``review_report.json`` never
    carried it, so this is checked against both.
    """
    paper = a.paper
    if paper is None:
        return []
    rr = a.review_report
    ckpt = a.json("checkpoint.json") or {}
    verdict = rr.get("overall_verdict")
    unverified = rr.get("unverified")
    if unverified is None:
        unverified = (ckpt.get("review_report") or {}).get("unverified")
    if unverified is None:
        unverified = ckpt.get("unverified")
    not_pass = (verdict is not None and verdict != "PASS") or bool(unverified)
    if not not_pass:
        return []
    marker = re.search(r"UNVERIFIED|unresolved methodological issues", paper, re.I)
    if marker:
        return []
    return [
        Finding(
            code="INV_UNVERIFIED_BLOCK_MISSING",
            severity="critical",
            message=(
                f"The run did not pass review (verdict={verdict!r}, "
                f"unverified={unverified!r}) and {a.paper_name} carries no "
                "warning block. SPEC 4.5 makes that block mandatory; without "
                "it the manuscript presents itself as cleanly reviewed."
            ),
            artifact=a.paper_name or "paper.tex",
            evidence={"overall_verdict": verdict, "unverified": unverified},
        )
    ]


def check_uncited_bib_entries(a: RunArtifacts) -> list[Finding]:
    """Bibliography entries the manuscript never cites."""
    paper = a.paper
    bib = a.text("references.bib")
    if paper is None or not bib:
        return []
    entries = re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", bib)
    if not entries:
        return []
    cited: set[str] = set()
    for m in _CITE_CMD.finditer(paper):
        cited.update(k.strip() for k in m.group(1).split(","))
    uncited = sorted(set(entries) - cited)
    if len(uncited) <= max(2, len(entries) // 10):
        return []
    return [
        Finding(
            code="INV_UNCITED_BIB_ENTRIES",
            severity="minor",
            message=(
                f"{len(uncited)} of {len(entries)} references.bib entries are "
                "never cited in the manuscript. A reference list padded to a "
                "venue's typical length is not a literature review; reference "
                "count is invariant to reference quality."
            ),
            artifact=f"{a.paper_name} + references.bib",
            evidence={
                "n_entries": len(entries),
                "n_cited": len(cited & set(entries)),
                "uncited_sample": uncited[:15],
            },
        )
    ]


def check_dangling_citation_keys(a: RunArtifacts) -> list[Finding]:
    """A ``\\cite`` key with no entry in the bibliography."""
    paper = a.paper
    bib = a.text("references.bib")
    if paper is None or bib is None:
        return []
    entries = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", bib))
    cited: set[str] = set()
    for m in _CITE_CMD.finditer(paper):
        cited.update(k.strip() for k in m.group(1).split(",") if k.strip())
    dangling = sorted(cited - entries)
    if not dangling:
        return []
    return [
        Finding(
            code="INV_DANGLING_CITATION_KEY",
            severity="critical",
            message=(
                f"{len(dangling)} \\cite key(s) have no references.bib entry: "
                f"{', '.join(dangling[:8])}"
                + (" ..." if len(dangling) > 8 else "")
                + ". These render as ? in the compiled PDF."
            ),
            artifact=f"{a.paper_name} + references.bib",
            evidence={"dangling": dangling},
        )
    ]


def check_citation_dump(a: RunArtifacts) -> list[Finding]:
    """A single ``\\cite`` carrying an implausible number of keys.

    A 21-key citation holding 21 of a paper's 26 cited references is not
    support for the sentence it follows; it is the retrieval pool.
    """
    paper = a.paper
    if paper is None:
        return []
    out: list[Finding] = []
    for m in _CITE_CMD.finditer(paper):
        keys = [k.strip() for k in m.group(1).split(",") if k.strip()]
        if len(keys) < 8:
            continue
        ctx = paper[max(0, m.start() - 160):m.start()].replace("\n", " ")
        out.append(
            Finding(
                code="INV_CITATION_DUMP",
                severity="major",
                message=(
                    f"A single \\cite carries {len(keys)} keys. One citation "
                    "cannot be specific support for one sentence at that "
                    "width; each key needs to be attached to the claim it "
                    "actually supports, or dropped."
                ),
                artifact=a.paper_name or "paper.tex",
                evidence={"n_keys": len(keys), "keys": keys[:25],
                          "preceding_text": ctx[-160:]},
                defect_ids=(),
            )
        )
    return out


def check_bibliography_ampersands(a: RunArtifacts) -> list[Finding]:
    """A bare ``&`` or an HTML entity in references.bib.

    An unescaped ampersand in a BibTeX field is a LaTeX alignment tab and
    takes the entry with it; ``&amp;`` renders literally. Measured
    prevalence across fourteen generated papers: 43%. Venue names carry
    ampersands constantly ("Science & Education", "Teaching and Teacher
    Education"), so this is the field where it happens.
    """
    bib = a.text("references.bib")
    if not bib:
        return []
    entities = len(re.findall(r"&(?:amp|lt|gt|quot|#\d+);", bib))
    bare = len(re.findall(r"(?<!\\)&(?!(?:amp|lt|gt|quot|#\d+);)", bib))
    if not entities and not bare:
        return []
    bits = []
    if bare:
        bits.append(f"{bare} unescaped &")
    if entities:
        bits.append(f"{entities} HTML entit{'y' if entities == 1 else 'ies'}")
    return [
        Finding(
            code="INV_BIB_AMPERSAND",
            severity="minor",
            message=(
                f"references.bib contains {' and '.join(bits)}. An unescaped "
                "ampersand is an alignment tab to LaTeX and breaks the entry; "
                "an HTML entity renders literally in the reference list."
            ),
            artifact="references.bib",
            evidence={
                "unescaped_ampersands": bare,
                "html_entities": entities,
                "samples": [
                    m.group(0)
                    for m in list(
                        re.finditer(r".{0,40}(?<!\\)&.{0,40}", bib)
                    )[:4]
                ],
            },
        )
    ]


def check_scaffolding_leaked(a: RunArtifacts) -> list[Finding]:
    """Pipeline machinery typeset as reader-facing prose."""
    paper = a.paper
    if paper is None:
        return []
    patterns = [
        (r"\\Description\{", "acmart \\Description in a non-acmart class"),
        (r"(?<![A-Za-z])\(P[1-9]\)", "bare pipeline step codes (P1)…(P6)"),
        (r"```", "markdown code fence"),
        (r"%%PLACEHOLDER:", "unfilled template placeholder"),
        (r"\bper the study's design\b", "spec checklist text transcribed as prose"),
        (r"\bis stated before it is applied\b", "spec checklist text transcribed as prose"),
        (r"\[---|\+---|—% confidence", "placeholder dash where a number belongs"),
    ]
    out: list[Finding] = []
    for pat, label in patterns:
        hits = re.findall(pat, paper)
        if hits:
            out.append(
                Finding(
                    code="INV_SCAFFOLDING_LEAKED",
                    severity="minor",
                    message=(
                        f"{a.paper_name} contains {len(hits)} instance(s) of "
                        f"{label}. Harmless to validity, and immediately "
                        "legible to a reviewer as machine generation."
                    ),
                    artifact=a.paper_name or "paper.tex",
                    evidence={"pattern": pat, "count": len(hits)},
                )
            )
    return out


#: Body-prose claims worth binding to an artifact. Deliberately narrow.
#:
#: Binding EVERY numeral in the body raises the false-positive floor
#: without raising recall much -- whole-body binding on the two flagship
#: papers produced 36 checked / 0 unmatched and 38 / 0, because every
#: number in them really does ground. What does NOT ground is a specific
#: shape: a count claim about cases ("correctly identifies 341 of the 506
#: true dropout episodes ... flags 1,550 non-dropouts", where the figure
#: being described prints 370 / 545 / 1,375 and none of 341, 506, 1550
#: exists in any artifact), and a named metric set equal to a value.
_COUNT_OF_TOTAL = re.compile(
    r"\b([\d][\d,]*)\s+of\s+(?:the\s+)?([\d][\d,]*)\b(?!\s*%)",
)
_FLAGS_COUNT = re.compile(
    r"\bflags?\s+(?:about\s+|roughly\s+)?([\d][\d,]{2,})\b",
    re.IGNORECASE,
)
_METRIC_EQUALS = re.compile(
    r"\b(AUC|RMSE|R\^?2|accuracy|precision|recall|F1|F2|Brier)\b"
    r"[^.\d]{0,30}?(?:=|of|was|is|at)\s*"
    r"(-?\d+\.\d+)",
    re.IGNORECASE,
)


def check_prose_numeral_unbound(a: RunArtifacts) -> list[Finding]:
    """A counted claim or a named metric value with no source on disk.

    Reuses the manuscript linter's grounding machinery -- the same
    candidate pool, the same print-precision tolerance -- but applies it
    to body prose, which the linter scopes past: it reconciles tables and
    confidence intervals only.

    Fails CLOSED in one direction and open in the other: if no candidate
    pool can be built, it reports that it could not check rather than
    reporting nothing. A reconciliation that silently skips looks
    identical to a clean paper, and that is how a QA suite once passed a
    corpus with 15 known defects.
    """
    paper = a.paper
    if paper is None:
        return []
    try:
        from src.manuscript_linter import _ground_candidates, _matches, _TABULAR_ENV
    except Exception:  # noqa: BLE001
        return []

    cand, info = _ground_candidates(Path(a.output_dir))
    if not cand:
        return [
            Finding(
                code="INV_NUMERAL_BINDING_SKIPPED",
                severity="minor",
                message=(
                    "No analysis artifacts could be loaded to bind the "
                    "manuscript's numbers against, so nothing was checked. "
                    "This is not a clean result."
                ),
                artifact=a.output_dir,
                evidence={"sources": info.get("sources", [])},
            )
        ]

    body = _TABULAR_ENV.sub(" ", paper)
    body = re.sub(r"(?s)\\begin\{thebibliography\}.*?\\end\{thebibliography\}", " ", body)
    # Everything from the appendix on is transcribed machinery -- this
    # pipeline pastes the raw Critic JSON there -- not the paper's own
    # claims.
    body = re.split(r"\\appendix\b|\\section\*?\{\s*Appendix", body)[0]
    # Related Work reports OTHER papers' numbers, which by construction
    # do not ground to this run.
    body = re.sub(
        r"(?is)\\section\*?\{[^}]*related work[^}]*\}.*?(?=\\section)", " ", body
    )
    body = body.replace("{,}", "").replace("\\%", "%").replace("$", "")

    claims: list[tuple[str, str, str]] = []  # (kind, value, sentence)
    for s in _sentences(body):
        # A value in a sentence that cites somebody is their value.
        if _CITE_CMD.search(s):
            continue
        for m in _COUNT_OF_TOTAL.finditer(s):
            for g in (m.group(1), m.group(2)):
                # Year-like and tiny values are too collision-prone and
                # too often structural ("3 of 5 models", "2017").
                v = g.replace(",", "")
                if v.isdigit() and 20 <= int(v) and not (1900 <= int(v) <= 2100):
                    claims.append(("count", g, s))
        for m in _FLAGS_COUNT.finditer(s):
            claims.append(("count", m.group(1), s))
        for m in _METRIC_EQUALS.finditer(s):
            claims.append((m.group(1).lower(), m.group(2), s))

    seen: set = set()
    unmatched = []
    for k, v, s in claims:
        if _matches(cand, v) or (k, v) in seen:
            continue
        seen.add((k, v))
        unmatched.append((k, v, s))
    if not unmatched:
        return []
    return [
        Finding(
            code="INV_PROSE_NUMERAL_UNBOUND",
            severity="critical" if any(k == "count" for k, _, _ in unmatched) else "major",
            message=(
                f"{len(unmatched)} of {len(claims)} counted/metric claim(s) in "
                f"the body print a value that no analysis artifact produces: "
                + "; ".join(f"{v} ({k})" for k, v, _ in unmatched[:6])
                + (" ..." if len(unmatched) > 6 else "")
                + ". A reader takes each of these for a computed result."
            ),
            artifact=f"{a.paper_name} + results.json",
            evidence={
                "n_claims_checked": len(claims),
                "unmatched": [
                    {"kind": k, "value": v, "sentence": s.strip()[:220]}
                    for k, v, s in unmatched[:10]
                ],
                "ground_sources": info.get("sources", []),
            },
            defect_ids=("J00", "J01"),
        )
    ]


#: "best X (0.788) ... worst Y (0.444) ... a gap of 0.528" -- the third
#: number is claimed to be the difference of the first two. Pure
#: arithmetic; no judgement about what anything means.
#:
#: Deliberately narrow. An earlier version also matched "difference",
#: "range" and "spread", and fired 20 times across four papers where the
#: catalogue records two real defects -- because "an AUC difference of
#: 0.0145" is an artifact value, not a subtraction between numbers in its
#: sentence, and "RMSE range 0.667" is not a subtraction at all. Only
#: "gap"/"disparity" survive, and only in a sentence that also frames two
#: extremes, which is the shape the real defects have.
_GAP_WORD = re.compile(
    r"\b(?:gap|disparity)\b(?:\s+of)?\s*[^.\d]{0,20}(\d+\.\d+)",
    re.IGNORECASE,
)
_EXTREMES_FRAMING = re.compile(
    r"\b(?:highest|lowest|best|worst|largest|smallest|top|bottom"
    r"|rang(?:e[ds]?|ing)\s+(?:from|across)|var(?:ies|ied)\s+from"
    r"|between\s+[\d.]+\s+and\s+[\d.]+)\b",
    re.IGNORECASE,
)
_DECIMAL = re.compile(r"(?<![\w.])(\d+\.\d+)(?![\w.])")


def check_stated_gap_arithmetic(a: RunArtifacts) -> list[Finding]:
    """A stated gap that is not the difference of the numbers beside it.

    One paper wrote "best White 0.788, worst NHPI 0.444, a gap of 0.528";
    0.788 - 0.444 is 0.344. Another stated a gap of 0.112 in a sentence
    whose own cells give 0.1088. Both numbers are real and both
    subtractions are wrong, so every check that binds numerals
    individually passes them.
    """
    paper = a.paper
    if paper is None:
        return []
    out: list[Finding] = []
    seen: set = set()
    for s in _sentences(paper):
        gaps = _GAP_WORD.findall(s)
        if not gaps or not _EXTREMES_FRAMING.search(s):
            continue
        values = [float(v) for v in _DECIMAL.findall(s)]
        for g in gaps:
            gv = float(g)
            others = [v for v in values if v != gv]
            if len(others) < 2:
                continue
            # Tolerance is the printed precision of the gap itself.
            dec = len(g.split(".")[1])
            tol = 0.5 * 10 ** (-dec) * 1.02 + 1e-12
            diffs = {
                round(abs(x - y), 10)
                for i, x in enumerate(others)
                for y in others[i + 1:]
            }
            if any(abs(d - gv) <= tol for d in diffs):
                continue
            key = (g, tuple(sorted(others)))
            if key in seen:  # abstract and body print the same sentence
                continue
            seen.add(key)
            plausible = sorted(diffs)[:4]
            out.append(
                Finding(
                    code="INV_STATED_GAP_ARITHMETIC",
                    severity="major",
                    message=(
                        f"A stated gap of {g} is not the difference of any "
                        f"pair of numbers in its own sentence "
                        f"(differences available: "
                        f"{', '.join(f'{d:g}' for d in plausible)}). Every "
                        "number here binds to an artifact individually; the "
                        "subtraction between them does not."
                    ),
                    artifact=a.paper_name or "paper.tex",
                    evidence={
                        "stated_gap": g,
                        "values_in_sentence": values,
                        "available_differences": plausible,
                        "sentence": s.strip()[:300],
                    },
                    defect_ids=("J18", "J39"),
                )
            )
    return out


_MEAN_CLUSTER = re.compile(
    r"\bmean\s+(?:cluster|school|group)\s+size\b[^.\d]{0,40}(\d+\.?\d*)",
    re.IGNORECASE,
)


def check_harmonic_mean_reported_as_mean(a: RunArtifacts) -> list[Finding]:
    """A "mean cluster size" that is the harmonic mean, not the mean.

    Two papers reported 10.4 and 14.5 where the arithmetic means were
    14.06 and 22.46. The harmonic mean of cluster sizes is what the
    design-effect formula uses, so it is a real quantity computed by real
    code -- it is just not what "mean cluster size" says, and it is
    always the smaller of the two.
    """
    paper = a.paper
    if paper is None:
        return []
    stated = _MEAN_CLUSTER.findall(paper)
    if not stated:
        return []

    sizes: list[int] = []
    for name in ("train_school_ids.csv", "test_school_ids.csv"):
        rows = a.csv_rows(name)
        if not rows:
            continue
        counts: dict = {}
        for r in rows:
            key = next(iter(r.values()), None)
            if key is not None:
                counts[key] = counts.get(key, 0) + 1
        sizes.extend(counts.values())
    if len(sizes) < 2:
        return []

    arithmetic = sum(sizes) / len(sizes)
    harmonic = len(sizes) / sum(1.0 / s for s in sizes if s)
    out: list[Finding] = []
    for v in stated:
        x = float(v)
        dec = len(v.split(".")[1]) if "." in v else 0
        tol = 0.5 * 10 ** (-dec) * 1.02 + 0.05
        if abs(x - arithmetic) <= tol:
            continue
        if abs(x - harmonic) <= tol:
            out.append(
                Finding(
                    code="INV_HARMONIC_MEAN_AS_MEAN",
                    severity="minor",
                    message=(
                        f"The paper reports a mean cluster size of {v}. That "
                        f"is the HARMONIC mean ({harmonic:.2f}); the "
                        f"arithmetic mean over {len(sizes)} clusters is "
                        f"{arithmetic:.2f}. The harmonic mean is the right "
                        "quantity for a design effect and the wrong one for "
                        "the phrase 'mean cluster size'."
                    ),
                    artifact=f"{a.paper_name} + *_school_ids.csv",
                    evidence={"stated": v, "arithmetic_mean": round(arithmetic, 4),
                              "harmonic_mean": round(harmonic, 4),
                              "n_clusters": len(sizes)},
                    defect_ids=("J29", "J71"),
                )
            )
        else:
            out.append(
                Finding(
                    code="INV_CLUSTER_SIZE_UNRECONCILED",
                    severity="minor",
                    message=(
                        f"The paper reports a mean cluster size of {v}, which "
                        f"is neither the arithmetic mean ({arithmetic:.2f}) "
                        f"nor the harmonic mean ({harmonic:.2f}) of the "
                        f"{len(sizes)} clusters on disk."
                    ),
                    artifact=f"{a.paper_name} + *_school_ids.csv",
                    evidence={"stated": v, "arithmetic_mean": round(arithmetic, 4),
                              "harmonic_mean": round(harmonic, 4),
                              "n_clusters": len(sizes)},
                )
            )
    return out


_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}
_FLAGGED_COUNT = re.compile(
    r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\s+"
    r"(?:predictors?|variables?|features?)\b[^.]{0,80}?"
    r"(?:exceed(?:ed|s|ing)?|above|over|greater than|more than)\b[^.]{0,30}?"
    # LaTeX writes a percent sign as `\%`, so the backslash has to be
    # optional here or the pattern never matches a real manuscript.
    r"(\d+(?:\.\d+)?)\s*\\?\s*(?:%|percent)",
    re.IGNORECASE,
)


def check_flagged_variable_count(a: RunArtifacts) -> list[Finding]:
    """"Three predictors exceeded 20% missing" when the report lists four.

    ``data_report.variables_flagged`` and ``missingness_summary`` are both
    on disk; counting them is arithmetic.
    """
    paper = a.paper
    dr = a.data_report
    if paper is None or not dr:
        return []
    miss = dr.get("missingness_summary")
    if not isinstance(miss, dict) or not miss:
        return []
    out: list[Finding] = []
    for m in _FLAGGED_COUNT.finditer(paper):
        word, thresh_s = m.group(1).lower(), m.group(2)
        stated = _NUMBER_WORDS.get(word)
        if stated is None:
            try:
                stated = int(word)
            except ValueError:
                continue
        try:
            thresh = float(thresh_s)
        except ValueError:
            continue
        over = [
            v
            for v, info in miss.items()
            if isinstance(info, dict) and (_f(info.get("pct_missing")) or 0.0) > thresh
        ]
        if len(over) == stated:
            continue
        out.append(
            Finding(
                code="INV_FLAGGED_COUNT_MISMATCH",
                severity="minor",
                message=(
                    f"The paper says {stated} predictor(s) exceeded "
                    f"{thresh_s}% missing; data_report.missingness_summary "
                    f"lists {len(over)} ({', '.join(sorted(over)[:8])})."
                ),
                artifact=f"{a.paper_name} + data_report.json",
                evidence={"stated": stated, "actual": len(over),
                          "threshold_pct": thresh, "over_threshold": sorted(over)},
                defect_ids=("J11",),
            )
        )
    return out


_IMPUTATION_WORDS = {
    "mode": ("mode",),
    "median": ("median",),
    "mean": ("mean",),
    "iterativeimputer": ("iterativeimputer", "multiple imputation", "mice", "chained"),
}


#: Both orders the papers actually use: "mode-imputed" / "mode
#: imputation", and "imputed with the mode".
_IMPUTATION_CLAIM = re.compile(
    r"\b(?:(mode|median|mean|IterativeImputer|multiple imputation)"
    r"[- ]?(?:imputed|imputation|imputing)"
    # "imputed WITH the mode", and also "imputed missing values USING
    # IterativeImputer" -- the short gap matters, because without it the
    # only claim the regex saw in a sentence naming both methods was the
    # second one, and four variables were attributed to it.
    r"|imput(?:ed|ing)\s+(?:\w+\s+){0,3}?(?:with|using|by|via)\s+"
    r"(?:the\s+|their\s+)?"
    r"(mode|median|mean|IterativeImputer|multiple imputation))",
    re.IGNORECASE,
)


def _enclosing_sentence(text: str, pos: int) -> tuple[str, int]:
    """The sentence containing *pos*, and its start offset in *text*."""
    start = text.rfind(".", 0, pos) + 1
    end = text.find(".", pos)
    end = end + 1 if end != -1 else len(text)
    return text[start:end], start


#: Where one clause of a methods sentence ends and the next begins: a
#: semicolon, or a comma before a coordinating conjunction.
_CLAUSE_BREAK = re.compile(r";|,\s+(?=(?:and|but|while|whereas)\b)")


def _clause_span(sentence: str, pos: int) -> tuple[int, int]:
    """Start and end offsets of the clause of *sentence* holding *pos*.

    Breaks inside parentheses or brackets are not clause breaks: "(X1RACE,
    X1SEX, and X1LOCALE)" is one list.
    """
    lo, hi = 0, len(sentence)
    for m in _CLAUSE_BREAK.finditer(sentence):
        prefix = sentence[: m.start()]
        if prefix.count("(") > prefix.count(")") or prefix.count("[") > prefix.count("]"):
            continue
        if m.start() < pos:
            lo = m.end()
        else:
            hi = m.start()
            break
    return lo, hi


def check_imputation_method_mismatch(a: RunArtifacts) -> list[Finding]:
    """The paper names an imputation method the data report contradicts.

    One paper stated a variable was mode-imputed where
    ``data_report.missingness_summary`` records ``IterativeImputer`` --
    a difference that matters, because single imputation propagates no
    uncertainty into any standard error and mode imputation of an ordinal
    expectation variable is a different modelling claim entirely.
    """
    paper = a.paper
    dr = a.data_report
    if paper is None or not dr:
        return []
    miss = dr.get("missingness_summary")
    if not isinstance(miss, dict):
        return []
    out: list[Finding] = []
    for var, info in miss.items():
        if not isinstance(info, dict):
            continue
        actual = str(info.get("imputation_method") or "").strip().lower()
        if not actual:
            continue
        for vm in re.finditer(r"(?<![\w])" + re.escape(var) + r"(?![\w])", paper):
            sent, sent_start = _enclosing_sentence(paper, vm.start())
            claims = list(_IMPUTATION_CLAIM.finditer(sent))
            if not claims:
                continue
            # Attribute to the clause the variable sits IN, which is the
            # nearest claim that has already opened -- not the nearest by
            # raw distance. A methods sentence often names several:
            # "imputed missing values using IterativeImputer for the
            # continuous and ordinal variables (BYSES1, BYPARED) and mode
            # imputation for BYSEX". BYSES1 is 88 characters past the
            # start of the first claim and 38 before the start of the
            # second, so "nearest start" hands it to `mode` and produced
            # four spurious findings on one paper. When no claim precedes
            # the variable -- "Categorical variables (X2STUEDEXPCT, ...)
            # were imputed with the mode" -- take the first that follows.
            #
            # And look inside the variable's own clause first. "Continuous
            # predictors were imputed using IterativeImputer; categorical
            # predictors (X1RACE, X1SEX) were imputed using the mode" has
            # a claim BEFORE X1RACE, but in the other clause; the one that
            # names its method comes after the variables. Only a clause
            # with no claim of its own falls back to the whole sentence.
            pos = vm.start() - sent_start
            lo, hi = _clause_span(sent, pos)
            pool = [m for m in claims if lo <= m.start() < hi] or claims
            preceding = [m for m in pool if m.end() <= pos]
            nearest = preceding[-1] if preceding else pool[0]
            claimed = (nearest.group(1) or nearest.group(2) or "").strip().lower()
            if not claimed:
                continue
            keys = _IMPUTATION_WORDS.get(claimed.replace(" ", ""), (claimed,))
            if any(k in actual for k in keys) or claimed in actual:
                break
            out.append(
                Finding(
                    code="INV_IMPUTATION_METHOD_MISMATCH",
                    severity="major",
                    message=(
                        f"The paper describes {var} as {claimed}-imputed; "
                        f"data_report records {info.get('imputation_method')!r}. "
                        "Single imputation propagates no uncertainty into any "
                        "standard error, so which one ran is a claim about "
                        "the intervals too."
                    ),
                    artifact=f"{a.paper_name} + data_report.json",
                    evidence={"variable": var, "claimed": claimed,
                              "actual": info.get("imputation_method"),
                              "sentence": sent.strip()[:300]},
                    defect_ids=("J19",),
                )
            )
            break
    return out


_PCT_IN_PROSE = re.compile(r"(?<![\w.])(\d{1,2}(?:\.\d+)?)\s*\\?\s*(?:%|percent)")


def _all_numbers(obj, out: set) -> None:
    if isinstance(obj, dict):
        for v in obj.values():
            _all_numbers(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _all_numbers(v, out)
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (int, float)):
        out.add(round(float(obj), 4))
    elif isinstance(obj, str):
        for t in re.findall(r"\d+(?:\.\d+)?", obj):
            out.add(round(float(t), 4))


def _outcome_rates(a: RunArtifacts) -> set:
    """Class balance derived from the y CSVs, as percentages.

    The rubric licenses "numbers derivable by simple arithmetic from
    artefacts". Without this, three correct sentences in three different
    papers read as spec-only.
    """
    out: set = set()
    counts: Counter = Counter()
    for name in ("train_y.csv", "test_y.csv"):
        rows = a.csv_rows(name)
        if not rows:
            continue
        split: Counter = Counter()
        for r in rows:
            v = next(iter(r.values()), None)
            if v is not None:
                split[str(v).strip()] += 1
        counts += split
        n = sum(split.values())
        if n:
            for k, c in split.items():
                out.add(round(100.0 * c / n, 4))
    n = sum(counts.values())
    if n:
        for k, c in counts.items():
            out.add(round(100.0 * c / n, 4))
    return out


def check_percentage_from_spec_not_run(a: RunArtifacts) -> list[Finding]:
    """A percentage that matches the plan and nothing the run computed.

    ``research_spec.json`` carries pre-analysis estimates --
    ``potential_limitations`` text, expected missingness, an anticipated
    class split. Those are guesses made before the data was touched, and
    a paper that prints one is reporting the plan as a result.

    Measured over the five delivered JEDM papers: two firings, both
    catalogued defects (J15 "approximately 15% non-persisters" against a
    measured 20.0%; J61 "14.8% missingness" against a recorded 13.95),
    zero false positives once the y-CSV-derived rates are counted as
    computed.
    """
    paper = a.paper
    spec = a.research_spec
    if paper is None or not spec:
        return []

    planned: set = set()
    computed: set = set()
    _all_numbers(spec, planned)
    _all_numbers(a.data_report, computed)
    _all_numbers(a.results, computed)
    computed |= {round(x * 100, 4) for x in list(computed) if 0 < x < 1}
    computed |= _outcome_rates(a)
    if not planned or not computed:
        return []

    body = re.sub(
        r"(?s)\\begin\{thebibliography\}.*?\\end\{thebibliography\}", " ", paper
    )
    out: list[Finding] = []
    seen: set = set()
    for m in _PCT_IN_PROSE.finditer(body):
        v = round(float(m.group(1)), 4)
        if v in seen:
            continue
        if not any(abs(v - s) < 0.05 for s in planned):
            continue
        if any(abs(v - c) < 0.05 for c in computed):
            continue
        seen.add(v)
        ctx = re.sub(r"\s+", " ", body[max(0, m.start() - 140) : m.end() + 60])
        out.append(
            Finding(
                code="INV_PERCENTAGE_FROM_SPEC_NOT_RUN",
                severity="major",
                message=(
                    f"The manuscript prints {m.group(1)}%, which appears in "
                    "research_spec.json but matches nothing in data_report, "
                    "results.json, or the outcome CSVs. The spec carries "
                    "pre-analysis estimates; a number the paper reports has "
                    "to come from what ran."
                ),
                artifact=f"{a.paper_name} + research_spec.json",
                evidence={"value": m.group(1), "sentence": ctx.strip()[:260]},
                defect_ids=("J15", "J61"),
            )
        )
    return out


def check_class_balance_sample(a: RunArtifacts) -> list[Finding]:
    """``class_balance`` counts that sum to n_train, not analytic_n.

    Needs no manuscript. Measured across the five delivered JEDM papers:
    bachelors {6469, 4021} sums to 10,490 = n_train against an
    analytic_n of 13,250; twoyear {6268, 4096} sums to 10,364 = n_train
    against 12,942. The field sits in a report about the analytic
    sample, so a paper reading it as the analytic sample is reading what
    the artifact says.
    """
    dr = a.data_report
    cb = dr.get("class_balance")
    if not isinstance(cb, dict) or not cb:
        return []
    vals = [v for v in (_f(x) for x in cb.values()) if v is not None]
    if not vals or any(0 < v < 1 for v in vals):
        return []  # proportions, not counts
    total = sum(vals)
    n_train = _f(dr.get("n_train"))
    analytic = _f(dr.get("analytic_n"))
    if n_train is None or analytic is None or n_train == analytic:
        return []
    if abs(total - n_train) > 1 or abs(total - analytic) <= 1:
        return []
    return [
        Finding(
            code="INV_CLASS_BALANCE_WRONG_SAMPLE",
            severity="major",
            message=(
                f"data_report.class_balance counts sum to {int(total)}, which "
                f"is n_train ({int(n_train)}), not the analytic sample "
                f"({int(analytic)}). The field sits in a report about the "
                "analytic sample; anything reading it as the analytic "
                "sample's class split is off by the whole test set."
            ),
            artifact="data_report.json",
            evidence={"class_balance": cb, "sum": total,
                      "n_train": n_train, "analytic_n": analytic},
            defect_ids=("J16",),
        )
    ]


#: LITERAL direction. "lowest RMSE" means the minimum RMSE, full stop.
_SUP_MAX = re.compile(r"\b(highest|largest|greatest)\b", re.IGNORECASE)
_SUP_MIN = re.compile(r"\b(lowest|smallest)\b", re.IGNORECASE)
#: QUALITY. These mean "the good end", so the metric's polarity decides.
#: The lookbehind excludes ORDINALS. "the next-best individual model
#: (RandomForest)" is a claim about second place, and RandomForest IS
#: second by AUC -- that sentence's real defect is which models the test
#: compared, which is INV_COMPARATOR_MISNAMED's job. Without this,
#: every "next-best" and "second-highest" is read as a claim about the
#: maximum and refuted by the maximum.
_SUP_BEST = re.compile(
    r"(?<!next-)(?<!next )(?<!second-)(?<!second )(?<!third-)(?<!third )"
    r"\b(best|top|strongest|most accurate|best[- ]performing)\b",
    re.IGNORECASE,
)
_SUP_WORST = re.compile(
    r"\b(worst|weakest|least accurate|worst[- ]performing)\b", re.IGNORECASE
)

_LOWER_IS_BETTER = ("rmse", "mae", "brier", "error")

#: A superlative that refers to no named model.
_GENERIC = re.compile(
    r"\b(?:the\s+)?(?:best|top|worst)[- ](?:model|performer|performing)\b",
    re.IGNORECASE,
)

_CITE_CMD = re.compile(r"\\[a-zA-Z]*cite[a-zA-Z]*\*?\s*(?:\[[^\]]*\]\s*)*\{([^}]*)\}")

_FLOAT_BLOCK = re.compile(
    r"(?s)\\begin\{(figure|table)\*?\}.*?\\end\{\1\*?\}|\\Description\{[^}]*\}"
)

_METRIC_WORDS = (
    ("auc", r"\bAUC\b|area under the receiver"),
    ("rmse", r"\bRMSE\b|root mean squared error"),
    ("r2", r"\bR\$?\^?2|coefficient of determination"),
    ("accuracy", r"\baccuracy\b"),
    ("f1", r"\bF1\b"),
    ("brier", r"\bBrier\b"),
)


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", t.lower())


def _names_in(model: str, sentence: str) -> bool:
    """Does the sentence name the model, however a writer spells it?

    ``all_models`` keys are CamelCase -- LogisticRegression,
    RandomForest -- and prose writes "logistic regression", "random
    forest". Matching the identifier literally found only XGBoost in
    "Logistic regression achieved the highest AUC ... followed closely
    by XGBoost", a CORRECT sentence, and attributed the superlative to
    XGBoost.
    """
    return _norm(model) in _norm(sentence)


def _distance_to(model: str, sentence: str, sup: re.Match) -> int | None:
    """Characters between the superlative and the nearest mention of *model*.

    ``None`` when the model is not mentioned, is further than 90
    characters away, or has a "than" / "between" / "compared" in the
    span -- those make the sentence a comparison rather than a claim
    about a maximum.
    """
    target = _norm(model)
    best = None
    for m in re.finditer(r"[A-Za-z][A-Za-z .\-]{2,34}", sentence):
        if target not in _norm(m.group(0)):
            continue
        lo, hi = sorted((m.start(), sup.start()))
        d = hi - lo
        if d > 90:
            continue
        if re.search(
            r"\b(than|between|versus|vs\.?|compared|relative to)\b",
            sentence[lo:hi],
            re.IGNORECASE,
        ):
            continue
        if best is None or d < best:
            best = d
    return best


#: "the highest AUC among the individual models" legitimately excludes
#: the stacking ensemble -- that is this pipeline's own selection rule.
_INDIVIDUAL_ONLY = re.compile(r"\bindividual\b|\bexcluding the (?:stacking|ensemble)\b",
                              re.IGNORECASE)
_IS_ENSEMBLE = re.compile(r"stacking|ensemble", re.IGNORECASE)


def check_superlative_contradicted(a: RunArtifacts) -> list[Finding]:
    """A superlative about a model that the results table contradicts.

    One paper called XGBoost's 0.7041860483168286 the "highest point
    estimate" with StackingEnsemble at 0.7045036583370043 in the same
    table.

    Fires only when a sentence names exactly one model the run knows
    about and the superlative attaches to it: "highest among the
    individual models" legitimately excludes the ensemble, and a
    sentence naming two is making a comparison rather than a claim about
    the maximum.
    """
    paper = a.paper
    scores = []
    for name, m in _model_metrics(a.results).items():
        vals = {k: v for k, v in ((k, _f(v)) for k, v in m.items()) if v is not None}
        if vals:
            scores.append((name, vals))
    if paper is None or len(scores) < 2:
        return []

    names = [n for n, _ in scores]
    out: list[Finding] = []
    seen_roots: set = set()

    for s in _sentences(_FLOAT_BLOCK.sub(" ", paper)):
        if _CITE_CMD.search(s) or _GENERIC.search(s):
            continue
        sup = kind = None
        for pat, k in (
            (_SUP_MAX, "max"),
            (_SUP_MIN, "min"),
            (_SUP_BEST, "best"),
            (_SUP_WORST, "worst"),
        ):
            sup = pat.search(s)
            if sup:
                kind = k
                break
        if sup is None:
            continue

        # The claim is about the model NEAREST the superlative, however
        # many the sentence goes on to name. The defect this exists for
        # is exactly a ranking sentence: "XGBoost achieved the highest
        # point estimate (AUC = 0.704), followed closely by the stacking
        # ensemble (AUC = 0.705), random forest ..., logistic regression
        # ..., and elastic net ...". Five models, and the one the
        # superlative attaches to is not the maximum.
        distances = {
            n: d
            for n in names
            if (d := _distance_to(n, s, sup)) is not None
        }
        if not distances:
            continue
        claimed = min(distances, key=lambda n: distances[n])

        metric = None
        for key, pat in _METRIC_WORDS:
            if re.search(pat, s, re.IGNORECASE):
                metric = key
                break
        if metric is None:
            continue

        pool = scores
        if _INDIVIDUAL_ONLY.search(s):
            pool = [(n, v) for n, v in scores if not _IS_ENSEMBLE.search(n)]
        have = [(n, v[metric]) for n, v in pool if metric in v]
        if len(have) < 2 or claimed not in dict(have):
            continue
        named = [claimed]

        if kind == "max":
            wants_max = True
        elif kind == "min":
            wants_max = False
        else:
            wants_max = kind == "best"
            if any(w in metric for w in _LOWER_IS_BETTER):
                wants_max = not wants_max

        winner = (max if wants_max else min)(have, key=lambda kv: kv[1])
        if winner[0] == named[0]:
            continue
        root = (named[0], winner[0], metric)
        if root in seen_roots:
            continue
        seen_roots.add(root)
        out.append(
            Finding(
                code="INV_SUPERLATIVE_CONTRADICTED",
                severity="major",
                message=(
                    f"The manuscript calls {named[0]} the "
                    f"{sup.group(1).lower()} on {metric}, but results.json "
                    f"gives {winner[0]} {winner[1]!r} against "
                    f"{named[0]}'s {dict(have)[named[0]]!r}."
                ),
                artifact=f"{a.paper_name} + results.json",
                evidence={
                    "claimed": named[0],
                    "actual": winner[0],
                    "metric": metric,
                    "values": dict(have),
                    "sentence": re.sub(r"\s+", " ", s).strip()[:300],
                },
                defect_ids=("J38",),
            )
        )
    return out


#: pdflatex's own words when it gives up and writes nothing. Any one of
#: them is sufficient; a fatal run normally prints all three.
_FATAL_LATEX = (
    "no output PDF file produced",
    "Emergency stop",
    "Fatal error occurred",
)

#: An undefined citation as the kernel, natbib and biblatex actually print
#: it. All three put "on page N" between the key and "undefined"::
#:
#:     LaTeX Warning: Citation `foo2020' on page 1 undefined on input line 3.
#:     Package natbib Warning: Citation `foo2020' on page 1 undefined on ...
#:     LaTeX Warning: Citation 'foo2020' on page 1 undefined on input line 5.
#:
#: (the last one is biblatex, which opens with a straight quote). The
#: pattern this replaced required "' undefined" straight after the key,
#: matched none of them, and so never fired on a real log: a PDF full of
#: [?] was released as clean. "on page N" stays optional for the
#: pre-2.09-style message some classes still emit.
_UNDEFINED_CITATION = re.compile(
    r"Citation [`']([^'\s]+)' (?:on page \S+ )?undefined"
)
#: Older biblatex reports a key missing from the .bib this way, over
#: several ``(biblatex)``-prefixed continuation lines.
_BIBLATEX_MISSING_ENTRY = re.compile(
    r"The following entry could not be found\s*\n\(biblatex\)\s+in the "
    r"database:\s*\n\(biblatex\)\s+(\S+)"
)
#: TeX hard-wraps its log at ``max_print_line`` (79 in TeX Live and
#: MiKTeX), so a long citation key arrives split across two lines.
_TEX_LOG_LINE_WIDTH = 79


def _unwrap_tex_log(log: str) -> str:
    """Rejoin lines TeX split at the log width, so a pattern can see a
    warning whole. A line exactly as wide as the limit is a wrapped one;
    joining the rare genuine 79-character line to its successor only
    concatenates text and cannot manufacture a match."""
    out: list[str] = []
    carry = ""
    for line in log.splitlines():
        if len(line) >= _TEX_LOG_LINE_WIDTH:
            carry += line
            continue
        out.append(carry + line)
        carry = ""
    if carry:
        out.append(carry)
    return "\n".join(out)


def _undefined_citations(log: str) -> list[str]:
    """Keys the final LaTeX pass reported as undefined, in log order."""
    text = _unwrap_tex_log(log)
    keys = _UNDEFINED_CITATION.findall(text)
    keys += _BIBLATEX_MISSING_ENTRY.findall(text)
    return keys


def _no_log_compile_record(a: "RunArtifacts") -> dict:
    """What ``latex_compile.json`` says about a compile that left no log.

    The orchestrator writes that file after every compile; it is the only
    place the reason survives when pdflatex never started (not installed,
    not on PATH) and so never wrote ``paper.log``.
    """
    record = a.json("latex_compile.json")
    if not isinstance(record, dict):
        return {}
    raw_steps = record.get("steps")
    steps: list = raw_steps if isinstance(raw_steps, list) else []
    first_bad = next(
        (
            s
            for s in steps
            if isinstance(s, dict) and s.get("returncode") not in (0, 1)
        ),
        None,
    )
    return {
        "missing_tool": record.get("missing_tool"),
        "failed_step": (first_bad or {}).get("cmd") or record.get("failed_step"),
        "stderr": str((first_bad or {}).get("stderr") or "")[:300],
        "returncode": (first_bad or {}).get("returncode"),
    }


def check_latex_compile_errors(a: RunArtifacts) -> list[Finding]:
    """Errors in the run's own LaTeX log.

    Read from the log the run wrote, not from a rebuild. A log check
    pointed at a repaired artifact reported zero errors for a paper whose
    original had roughly 1,000 characters destroyed by math-mode
    collapse.

    A compile that produced NO PDF is reported separately and as
    critical. This check used to grade every ``!`` line the same way, so
    a run whose pdflatex hit an emergency stop and wrote nothing scored
    four majors, and ``run_status.json`` recorded ``released: true,
    reason: clean`` for a paper that does not exist. Three errors with a
    PDF beside them and a fatal abort with no PDF at all are not the
    same finding.
    """
    log = a.text("paper.log")
    if not log:
        # No log is not the same as no compile. The orchestrator compiles
        # every manuscript it writes; when pdflatex is not installed or
        # not on PATH it never starts, writes neither paper.log nor
        # paper.pdf, and this check used to return nothing -- so the one
        # blocking code could not fire and a run with no PDF at all was
        # released as clean. A manuscript with neither a log nor a PDF
        # beside it is a deliverable that was not produced. A directory
        # with no manuscript (an aborted run) or with a PDF (a log
        # cleaned up afterwards) still claims nothing.
        if a.paper_name is None or a.exists("paper.pdf"):
            return []
        record = _no_log_compile_record(a)
        tool = record.get("missing_tool")
        if tool:
            why = (
                f"{tool} was not found, so the compile never ran. Install a "
                "TeX distribution (TeX Live, MiKTeX or MacTeX) and make sure "
                f"{tool} is on the PATH this pipeline runs with."
            )
        elif record.get("failed_step"):
            why = (
                f"the compile step `{record['failed_step']}` failed "
                f"(rc={record.get('returncode')}) before writing a log"
                + (f": {record['stderr']}" if record.get("stderr") else ".")
            )
        else:
            why = (
                "pdflatex never ran or died before writing its log (is a "
                "TeX distribution installed and pdflatex on PATH?)."
            )
        return [
            Finding(
                code="INV_LATEX_NO_PDF",
                severity="critical",
                message=(
                    f"LaTeX did not produce a PDF: {a.paper_name} has no "
                    f"paper.log and no paper.pdf beside it; {why}"
                ),
                artifact=a.paper_name,
                evidence={
                    "fatal_markers": [],
                    "paper_pdf_present": False,
                    "paper_log_present": False,
                    "compile_ran": False,
                    "missing_tool": tool,
                    "failed_step": record.get("failed_step"),
                    "errors": [],
                },
            )
        ]
    errors = [
        ln.strip()
        for ln in log.splitlines()
        if ln.startswith("! ") or ln.startswith("!pdfTeX error")
    ]
    undefined = _undefined_citations(log)
    unused_opts = re.findall(r"Unused global option\(s\):\s*\n?\s*\[([^\]]*)\]", log)
    out: list[Finding] = []
    markers = [m for m in _FATAL_LATEX if m in log]
    has_pdf = a.exists("paper.pdf")
    if markers or not has_pdf:
        # The log exists, so a compile was attempted. Either pdflatex said
        # it gave up, or there is no PDF where one should be. Both mean
        # the deliverable was not produced.
        out.append(
            Finding(
                code="INV_LATEX_NO_PDF",
                severity="critical",
                message=(
                    "LaTeX did not produce a PDF. "
                    + (
                        f"paper.log says: {markers[0]}. "
                        if markers
                        else "paper.log records no fatal marker, but no "
                        "paper.pdf sits beside it. "
                    )
                    + (
                        f"First error: {errors[0][:120]}"
                        if errors
                        else "The log records no `!` error line, so the "
                        "failure is somewhere the log does not name."
                    )
                ),
                artifact="paper.log",
                evidence={
                    "fatal_markers": markers,
                    "paper_pdf_present": has_pdf,
                    "errors": errors[:10],
                },
            )
        )
    elif errors:
        out.append(
            Finding(
                code="INV_LATEX_COMPILE_ERROR",
                severity="major",
                message=(
                    f"paper.log records {len(errors)} LaTeX error(s): "
                    f"{errors[0][:120]}"
                    + (f" (+{len(errors) - 1} more)" if len(errors) > 1 else "")
                ),
                artifact="paper.log",
                evidence={"errors": errors[:10]},
            )
        )
    if undefined:
        out.append(
            Finding(
                code="INV_UNDEFINED_CITATION",
                severity="critical",
                message=(
                    f"paper.log reports {len(set(undefined))} undefined "
                    f"citation(s): {', '.join(sorted(set(undefined))[:6])}. "
                    "These render as ? in the PDF a reviewer receives."
                ),
                artifact="paper.log",
                evidence={"undefined": sorted(set(undefined))[:20]},
            )
        )
    if unused_opts:
        out.append(
            Finding(
                code="INV_UNUSED_CLASS_OPTION",
                severity="minor",
                message=(
                    "paper.log reports unused document-class option(s): "
                    f"{', '.join(unused_opts)}. LaTeX silently ignored an "
                    "option the template meant to set -- usually a typo, and "
                    "the document therefore rendered under class defaults."
                ),
                artifact="paper.log",
                evidence={"unused_options": unused_opts},
            )
        )
    return out


_VERBATIM = re.compile(r"(?s)\\begin\{(verbatim|lstlisting|minted)\}.*?\\end\{\1\}")
_MATH = re.compile(r"(?s)\$\$.*?\$\$|\$[^$]*\$|\\\[.*?\\\]|\\\(.*?\\\)")
_COMMENT = re.compile(r"(?<!\\)%.*")

_BEGIN_ENV = re.compile(r"\\begin\s*\{([^}]*)\}")
_END_ENV = re.compile(r"\\end\s*\{([^}]*)\}")
#: ``\newenvironment{x}{...\begin{y}...}{...\end{y}...}`` balances across
#: two arguments that this counter never sees as a pair. Drop the
#: declaration rather than teach the counter to brace-match.
_ENV_DEFINITION = re.compile(
    r"\\(?:re)?newenvironment\s*\*?\s*\{[^}]*\}(?:\s*\[[^\]]*\])*"
)


def check_unbalanced_environments(a: RunArtifacts) -> list[Finding]:
    r"""A ``\begin{env}`` the manuscript never closes.

    Written after a delivered paper opened ``\begin{CCSXML}`` in the ACM
    preamble, wrote the XML closing tag ``</CCSXML>``, and never wrote
    ``\end{CCSXML}``. ``CCSXML`` is a ``comment`` environment, so LaTeX
    swallowed the entire document looking for its end and aborted with
    "File ended while scanning use of \next". No PDF was produced, and
    the run was released.

    The compile log reports the *symptom*, and only if a log survived.
    This reads the manuscript, so it fires whether or not anything was
    compiled, and it names the environment rather than the line where
    TeX finally gave up -- which in that paper was 850 lines away.

    Comments and verbatim blocks are stripped first: a ``%``-commented
    ``\begin`` opens nothing, and a listing may legitimately print one.
    """
    tex = a.paper
    if not tex:
        return []
    body = _VERBATIM.sub(" ", tex)
    body = _COMMENT.sub(" ", body)
    body = _ENV_DEFINITION.sub(" ", body)

    counts: dict[str, int] = {}
    for m in _BEGIN_ENV.finditer(body):
        name = m.group(1).strip()
        counts[name] = counts.get(name, 0) + 1
    for m in _END_ENV.finditer(body):
        name = m.group(1).strip()
        counts[name] = counts.get(name, 0) - 1
    unbalanced = {k: v for k, v in counts.items() if v != 0}
    if not unbalanced:
        return []

    def _phrase(name: str, n: int) -> str:
        if n > 0:
            return f"{name}: opened {n} time(s) and never closed"
        return f"{name}: closed {-n} time(s) more than it was opened"

    detail = "; ".join(_phrase(k, v) for k, v in sorted(unbalanced.items()))
    return [
        Finding(
            code="INV_LATEX_ENVIRONMENT_UNBALANCED",
            severity="critical",
            message=(
                f"{a.paper_name} does not balance "
                f"{len(unbalanced)} environment(s) -- {detail}. LaTeX reads "
                "past the intended end of the group, and the error it "
                "finally reports names neither the environment nor the "
                "line that opened it."
            ),
            artifact=a.paper_name or "paper.tex",
            evidence={"unbalanced": unbalanced},
        )
    ]


def _prose_only(tex: str) -> str:
    """Body text with math, verbatim, comments and macro args removed."""
    body = tex.split(r"\begin{document}", 1)[-1]
    body = _VERBATIM.sub(" ", body)
    body = _COMMENT.sub(" ", body)
    body = _MATH.sub(" ", body)
    # Drop the arguments of commands where a bare special is legitimate.
    body = re.sub(r"\\(?:url|href|path|verb|label|ref|[a-zA-Z]*cite[a-zA-Z]*)"
                  r"\*?(?:\[[^\]]*\])*\{[^}]*\}", " ", body)
    return body


def check_unescaped_latex_specials(a: RunArtifacts) -> list[Finding]:
    """A bare ``&``, ``_``, ``#`` or ``%`` in prose.

    ``_`` is the expensive one: a bare underscore in ``(F1SCH_ID)``
    opened math mode in one delivered paper and ran roughly 1,000
    characters together in italic, overprinting Table 1.
    """
    paper = a.paper
    if paper is None:
        return []
    body = _prose_only(paper)
    # Strip tabular bodies: '&' is the column separator there.
    body = re.sub(r"(?s)\\begin\{(tabular|tabularx|longtable)\}.*?\\end\{\1\}", " ", body)
    hits: dict[str, list[str]] = {}
    for ch in ("_", "&", "#"):
        for m in re.finditer(r"(?<!\\)" + re.escape(ch), body):
            ctx = body[max(0, m.start() - 40):m.start() + 40].replace("\n", " ")
            hits.setdefault(ch, []).append(ctx.strip())
    if not hits:
        return []
    total = sum(len(v) for v in hits.values())
    return [
        Finding(
            code="INV_UNESCAPED_LATEX_SPECIAL",
            severity="major" if "_" in hits else "minor",
            message=(
                f"{a.paper_name} contains {total} unescaped LaTeX special(s) "
                f"in prose ({', '.join(f'{k} x{len(v)}' for k, v in hits.items())}). "
                "A bare underscore opens math mode and can run a page of text "
                "together in italic."
            ),
            artifact=a.paper_name or "paper.tex",
            evidence={k: v[:5] for k, v in hits.items()},
        )
    ]


def check_alt_text_as_body(a: RunArtifacts) -> list[Finding]:
    """``\\Description{}`` used outside acmart, so its argument is typeset.

    ``\\Description`` is an acmart command. Under apa7 LaTeX discards the
    undefined command and sets its argument as prose, so "Bar chart of
    mean absolute SHAP values by feature..." appears in the body.
    """
    paper = a.paper
    if paper is None:
        return []
    if "acmart" in paper.split(r"\begin{document}", 1)[0]:
        return []
    hits = re.findall(r"\\Description\s*\{([^}]{0,120})", paper)
    if not hits:
        return []
    return [
        Finding(
            code="INV_ALT_TEXT_AS_BODY",
            severity="major",
            message=(
                f"{a.paper_name} uses \\Description {len(hits)} time(s) in a "
                "non-acmart document. LaTeX drops the undefined command and "
                "typesets its argument, so the alt text appears as body "
                f"prose -- e.g. {hits[0][:80]!r}."
            ),
            artifact=a.paper_name or "paper.tex",
            evidence={"count": len(hits), "samples": hits[:4]},
        )
    ]


def check_handtyped_crossrefs(a: RunArtifacts) -> list[Finding]:
    """"Table 3" typed as a literal where a ``\\ref`` belongs.

    A hand-typed number stops tracking the float it names the moment
    anything is inserted above it.
    """
    paper = a.paper
    if paper is None:
        return []
    body = _prose_only(paper)
    hits = [
        m.group(0)
        # Floats only. Section numbers are conventionally typed out and
        # do not move; a float's number changes the moment another float
        # is inserted above it.
        for m in re.finditer(r"(?<![\\\w])(?:Table|Figure)~?\s+\d+\b", body)
    ]
    if len(hits) < 2:
        return []
    return [
        Finding(
            code="INV_HANDTYPED_CROSSREF",
            severity="minor",
            message=(
                f"{a.paper_name} refers to floats by literal number "
                f"{len(hits)} time(s) ({', '.join(sorted(set(hits))[:6])}) "
                "instead of \\ref. These stop matching the float as soon as "
                "anything is inserted above it."
            ),
            artifact=a.paper_name or "paper.tex",
            evidence={"count": len(hits), "samples": sorted(set(hits))[:10]},
        )
    ]


def check_promised_analyses_missing(a: RunArtifacts) -> list[Finding]:
    """An analysis the spec commissioned that the paper never reports.

    The spec is the run's own contract. A commissioned heterogeneity
    analysis that appears nowhere in the manuscript, with no disclosure
    that it was dropped, reads as a design choice rather than a failure.
    """
    spec = a.research_spec
    paper = a.paper
    if not spec or paper is None:
        return []
    body = paper.lower()
    out: list[Finding] = []

    subgroups = spec.get("subgroup_analyses") or []
    if isinstance(subgroups, list):
        reported = a.results.get("subgroup_performance") or {}
        for attr in subgroups:
            name = str(attr)
            if not name:
                continue
            in_results = isinstance(reported, dict) and name in reported
            in_paper = name.lower() in body
            if not in_results and not in_paper:
                out.append(
                    Finding(
                        code="INV_PROMISED_ANALYSIS_MISSING",
                        severity="major",
                        message=(
                            f"research_spec commissions a subgroup analysis on "
                            f"{name!r}. It appears in neither "
                            "results.subgroup_performance nor the manuscript, "
                            "and nothing discloses that it was dropped."
                        ),
                        artifact="research_spec.json + " + (a.paper_name or "paper.tex"),
                        evidence={"attribute": name,
                                  "subgroups_reported": sorted(reported)
                                  if isinstance(reported, dict) else None},
                        defect_ids=(),
                    )
                )

    for key, label in (
        ("subgroup_of_interest_for_m5", "heterogeneity subgroup"),
        ("primary_method", "primary method"),
    ):
        val = spec.get(key) or (spec.get("causal_estimand") or {}).get(key)
        if isinstance(val, str) and val and val.lower() not in body:
            out.append(
                Finding(
                    code="INV_PROMISED_ANALYSIS_MISSING",
                    severity="major",
                    message=(
                        f"research_spec sets {key}={val!r} ({label}) and the "
                        "string appears nowhere in the manuscript. Either it "
                        "was run and not reported, or dropped and not "
                        "disclosed."
                    ),
                    artifact="research_spec.json + " + (a.paper_name or "paper.tex"),
                    evidence={"spec_key": key, "value": val},
                )
            )
    return out


def check_sample_size_consistency(a: RunArtifacts) -> list[Finding]:
    """A headline n that does not match the n the analysis ran on.

    When a results block records both a total and its per-group counts,
    the counts must sum to the total it is reported beside.
    """
    out: list[Finding] = []

    def walk(node: Any, trail: str) -> None:
        if isinstance(node, dict):
            counts = node.get("group_counts")
            n_an = _f(node.get("n_analyzed"))
            n_tot = _f(node.get("n"))
            if isinstance(counts, dict) and counts:
                s = sum(v for v in (_f(x) for x in counts.values()) if v is not None)
                for label, val in (("n_analyzed", n_an), ("n", n_tot)):
                    if val is not None and val != s:
                        out.append(
                            Finding(
                                code="INV_N_GROUP_SUM_MISMATCH",
                                severity="major",
                                message=(
                                    f"results.json{trail}: {label}={int(val)} "
                                    f"but the group counts sum to {int(s)} "
                                    f"({', '.join(f'{k}={v}' for k, v in list(counts.items())[:6])}). "
                                    "The n reported beside these estimates is "
                                    "not the n they were computed on."
                                ),
                                artifact="results.json",
                                evidence={"path": trail, label: val,
                                          "group_counts": counts, "sum": s},
                                defect_ids=(),
                            )
                        )
            for k, v in node.items():
                walk(v, f"{trail}[{k!r}]")
        elif isinstance(node, list):
            for i, v in enumerate(node[:200]):
                walk(v, f"{trail}[{i}]")

    walk(a.results, "")
    return out


#: Below this many body words a .tex is not a short paper, it is a
#: failure wearing a paper's file name. The observed case was 20 words:
#: a title, \bibliographystyle, \bibliography and \end{document}.
_MIN_BODY_WORDS = 400


def check_manuscript_is_empty(a: RunArtifacts) -> list[Finding]:
    """A paper.tex with essentially no body.

    One run wrote a complete 69 KB manuscript, had the response cut off
    at the token ceiling before it could close the document, and shipped
    287 bytes: the reassembler's body regex needs a closing boundary, the
    truncated response had none, and an empty body substituted cleanly
    into the template. The pipeline recorded COMPLETED.

    Every other check in this module looks at what the paper SAYS. This
    one exists because a paper that says nothing passes all of them.
    """
    paper = a.paper
    if paper is None:
        return []
    body = paper.split(r"\maketitle", 1)[-1]
    body = re.sub(r"(?s)\\begin\{thebibliography\}.*?\\end\{thebibliography\}", " ", body)
    words = re.findall(r"[A-Za-z][A-Za-z'-]+", body)
    if len(words) >= _MIN_BODY_WORDS:
        return []
    return [
        Finding(
            code="INV_MANUSCRIPT_EMPTY",
            severity="critical",
            message=(
                f"{a.paper_name} has {len(words)} words of body text after "
                f"\\maketitle (floor {_MIN_BODY_WORDS}). This is not a short "
                "paper; it is a failed write that produced a file. Check "
                "whether the Writer response was truncated at the token "
                "limit -- the reassembler needs a closing structural "
                "boundary and a cut-off response has none."
            ),
            artifact=a.paper_name or "paper.tex",
            evidence={
                "body_words": len(words),
                "paper_bytes": len(paper),
                "floor": _MIN_BODY_WORDS,
            },
        )
    ]


def check_zero_models_trained(a: RunArtifacts) -> list[Finding]:
    """A prediction run that trained nothing still produced a paper."""
    res = a.results
    if not res:
        return []
    if res.get("primary_metric") is None and not res.get("all_models"):
        return []
    models = _model_metrics(res)
    if len(models) >= 1:
        return []
    return [
        Finding(
            code="INV_ZERO_MODELS",
            severity="critical",
            message=(
                "results.json declares a prediction task but all_models is "
                "empty. A run that trained no models has no result to write "
                "up."
            ),
            artifact="results.json",
            evidence={"all_models": res.get("all_models")},
        )
    ]


# ---------------------------------------------------------------------------
# Registry and driver
# ---------------------------------------------------------------------------

#: Every check, in report order. Each is ``(RunArtifacts) -> [Finding]``
#: and must be side-effect free.
CHECKS: tuple[Callable[[RunArtifacts], list[Finding]], ...] = (
    check_macro_metric_mislabel,
    check_comparator_unnamed,
    check_subgroup_false_unavailable,
    check_dummy_cardinality,
    check_group_label_missing_as_level,
    check_degenerate_feature_weight,
    check_shap_rank_disagreement,
    check_omega_unidimensional_pooling,
    check_bootstrap_ci_resamples,
    check_estimand_population_mismatch,
    check_post_match_balance_worse,
    check_sample_size_consistency,
    check_manuscript_is_empty,
    check_zero_models_trained,
    check_promised_analyses_missing,
    check_prose_numeral_unbound,
    check_percentage_from_spec_not_run,
    check_class_balance_sample,
    check_superlative_contradicted,
    check_stated_gap_arithmetic,
    check_harmonic_mean_reported_as_mean,
    check_flagged_variable_count,
    check_imputation_method_mismatch,
    check_latex_compile_errors,
    check_unbalanced_environments,
    check_unescaped_latex_specials,
    check_alt_text_as_body,
    check_handtyped_crossrefs,
    check_figures_orphaned,
    check_figures_claimed_absent,
    check_duplicate_figure_images,
    check_unverified_block_present,
    check_prose_authority_uncited,
    check_dangling_citation_keys,
    check_uncited_bib_entries,
    check_citation_dump,
    check_bibliography_ampersands,
    check_scaffolding_leaked,
)

SEVERITY_ORDER = {"critical": 0, "major": 1, "minor": 2}


def run_invariants(
    run_dir: str,
    checks: Iterable[Callable[[RunArtifacts], list[Finding]]] | None = None,
) -> list[Finding]:
    """Run every invariant over one run directory.

    A check that raises is reported as a finding rather than aborting the
    battery -- a broken detector must not look like a clean run.
    """
    a = RunArtifacts(run_dir)
    findings: list[Finding] = []
    for check in checks if checks is not None else CHECKS:
        try:
            findings.extend(check(a) or [])
        except Exception as exc:  # noqa: BLE001
            findings.append(
                Finding(
                    code="INV_CHECK_ERROR",
                    severity="minor",
                    message=f"{check.__name__} raised {type(exc).__name__}: {exc}",
                    artifact=run_dir,
                    evidence={"check": check.__name__},
                )
            )
    findings.sort(key=lambda f: (SEVERITY_ORDER.get(f.severity, 9), f.code))
    return findings


def findings_to_json(findings: list[Finding]) -> dict:
    """Serialize findings with the counts a gate reads."""
    counts = {"critical": 0, "major": 0, "minor": 0}
    for f in findings:
        counts[f.severity] = counts.get(f.severity, 0) + 1
    return {
        "n_findings": len(findings),
        "counts": counts,
        "codes": sorted({f.code for f in findings}),
        "findings": [f.to_dict() for f in findings],
    }
