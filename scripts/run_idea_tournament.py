"""Arc T - the idea tournament driver. T1a ships ``--stage generate``.

What this stage does, in order:

1. Enumerate a deterministic slate (``slate.json``) - every cell, every
   quota decision, the RNG seed.
2. Draw ONE idea per cell, independently (``candidates.jsonl``).
3. Compile each card into a locked research_spec and run the shipped
   deterministic screen over it (``feasibility.json``).
4. Persist every rejected card with its KILL code and the evidence
   behind it (``killed.jsonl``). No rejected idea has ever been written
   to disk in this repo before; this file is the training data for
   every downstream slice, and it is the only false-negative detector
   the screen will ever have.
5. Rank the survivors on the DETERMINISTIC terms only - venue fit minus
   feasibility penalty (``ranking_deterministic.json``).

There is NO judging in T1a. Nothing here calls a model except step 2,
and step 2 goes through ``BaseAgent.call_llm``. ``--offline`` replaces
even that with a template stub so the plumbing can be exercised without
spending anything.

Usage:
    python scripts/run_idea_tournament.py --stage generate --offline
    python scripts/run_idea_tournament.py --stage generate \
        --tournament-id T-0007 --n-candidates 24
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ideation import cards as C  # noqa: E402
from src.ideation import feasibility as F  # noqa: E402
from src.ideation import generate as G  # noqa: E402
from src.ideation import slate as S  # noqa: E402
from src.ideation import venue_fit as V  # noqa: E402

DEFAULT_IDEAS_DIR = "ideas"

#: Spec sec. 4.1. These weights turn the deterministic terms into a BT
#: prior offset in T1b; in T1a they ARE the ranking. Chosen so the
#: deterministic prior can move a candidate about one rank but cannot
#: override a unanimous judged sweep later.
DEFAULT_WEIGHT_VENUE_FIT = 0.30
DEFAULT_WEIGHT_PENALTY = 0.20

_TID = re.compile(r"^T-(\d{4})$")


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def next_tournament_id(base_dir: str | os.PathLike[str]) -> str:
    """``T-0001``, or one past the highest existing id in ``base_dir``."""
    base = Path(base_dir)
    highest = 0
    if base.is_dir():
        for child in base.iterdir():
            match = _TID.match(child.name)
            if match and child.is_dir():
                highest = max(highest, int(match.group(1)))
    return f"T-{highest + 1:04d}"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, ensure_ascii=False)
        handle.write("\n")


def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
            count += 1
    return count


def _cfg(config: dict, *keys: str, default: Any = None) -> Any:
    node: Any = config or {}
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node if node is not None else default


# --------------------------------------------------------------------------
# The generate stage
# --------------------------------------------------------------------------


def run_generate_stage(
    *,
    tournament_id: str,
    out_dir: str | os.PathLike[str],
    config: dict | None = None,
    n_candidates: int = S.DEFAULT_N_CANDIDATES,
    seed: int = S.DEFAULT_SEED,
    datasets: Sequence[str] | None = None,
    task_types: Sequence[str] | None = None,
    call_llm: G.LLMCaller | None = None,
    generator_model: str | None = None,
    registry_dir: str | os.PathLike[str] | None = None,
    raw_data_dir: str | os.PathLike[str] | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    use_column_cache: bool = True,
    run_probes: bool = False,
    dedupe_cosine: float | None = None,
    max_attempts: int = 2,
    venue: str | None = None,
    venue_rules_path: str | os.PathLike[str] | None = None,
    offline: bool = False,
    log: Callable[[str], None] | None = None,
) -> dict:
    """Run the generate stage and write every artifact. Returns a summary.

    ``call_llm`` is injectable so the whole stage runs offline in tests
    with a stub. When it is None and ``offline`` is False, the caller is
    built from config and routed through ``BaseAgent.call_llm``.
    """
    config = config or {}
    emit = log or (lambda message: None)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- 1. slate ------------------------------------------------------
    slate = S.build_slate(
        tournament_id,
        n_candidates=n_candidates,
        seed=seed,
        datasets=datasets,
        task_types=task_types,
        bridge_quota=int(
            _cfg(
                config,
                "ideation",
                "tournament",
                "bridge_framing_quota",
                default=S.DEFAULT_BRIDGE_QUOTA,
            )
        ),
        registry_dir=registry_dir,
    )
    slate_payload = slate.to_dict()
    slate_payload["stage"] = "generate"
    _write_json(out / "slate.json", slate_payload)
    emit(S.format_slate(slate))

    if not slate.cells:
        summary = {
            "tournament_id": tournament_id,
            "stage": "generate",
            "error": "the slate is empty: no (dataset x task_type) cell was "
            "enumerable under rules S1/S3",
        }
        _write_json(out / "generate_summary.json", summary)
        return summary

    # --- 2. generation -------------------------------------------------
    registries = G.registries_for(
        [cell.dataset for cell in slate.cells], registry_dir
    )
    threshold = (
        dedupe_cosine
        if dedupe_cosine is not None
        else float(
            _cfg(
                config,
                "ideation",
                "tournament",
                "dedupe_cosine",
                default=G.DEFAULT_DEDUPE_COSINE,
            )
        )
    )
    notes: list[str] = []
    if call_llm is None:
        if offline:
            call_llm = G.offline_caller(registry_dir=registry_dir)
            generator_model = G.OFFLINE_MODEL_ID
            notes.append(
                "offline mode: cards come from a deterministic template, not "
                "a model. The ranking below is evidence that the plumbing "
                "works and nothing else. Two known stub artifacts: the "
                "template echoes the opportunity-pattern name inside the "
                "research question, which fires the venue-fit transfer "
                "keywords (VF-04/VF-05) on most cards; and no stub card "
                "resolves a target, so the structural dedupe key is always "
                "None and only the lexical key can fire. Any D-DUPLICATE "
                "kill in this run is therefore a fact about the template, "
                "not about the ideas."
            )
        else:
            call_llm, resolved_model = G.make_llm_caller(
                config,
                dataset=slate.cells[0].dataset,
                task_type=slate.cells[0].task_type,
                output_dir=str(out),
            )
            generator_model = generator_model or resolved_model

    result = G.generate_cards(
        slate,
        call_llm=call_llm,
        generator_model=generator_model or "",
        registry_dir=registry_dir,
        registries=registries,
        dedupe_cosine=threshold,
        max_attempts=max_attempts,
        on_event=emit,
    )
    emit(f"generation: {result.summary()}")

    # --- 3. compile + screen -------------------------------------------
    survivors: list[dict] = []
    killed: list[dict] = list(result.killed)
    reports: list[dict] = []

    for card in result.cards:
        spec = C.compile_spec(
            card, registry=registries.get(card.dataset or ""),
            registry_dir=registry_dir,
        )
        context = F.make_context(
            spec,
            dataset=card.dataset,
            task_type=card.task_type,
            registry=registries.get(card.dataset or ""),
            registry_dir=registry_dir,
            raw_data_dir=raw_data_dir,
            cache_dir=cache_dir,
            card=card.to_dict(),
            use_column_cache=use_column_cache,
        )
        report = F.screen(
            spec,
            candidate_id=card.candidate_id,
            context=context,
            run_probes=run_probes,
        )
        reports.append(report.to_dict())

        if report.analytic_n_estimate is not None:
            # Recompile so the measured n travels with the spec. Only
            # expected_contribution changes; the screen already ran on
            # text identical in every field a check reads.
            spec = C.compile_spec(
                card,
                report,
                registry=registries.get(card.dataset or ""),
                registry_dir=registry_dir,
            )

        if report.verdict == F.KILL:
            killed.append(
                G.kill_record(
                    card,
                    stage="feasibility_screen",
                    kill_code=report.kill_codes[0],
                    evidence="; ".join(
                        f"{check.code}: {check.message} [read: {check.evidence}]"
                        for check in report.kills
                    ),
                    detail={
                        "kill_codes": report.kill_codes,
                        "warn_codes": report.warn_codes,
                        "penalty": round(report.penalty, 4),
                        # The spec that was actually screened. killed.jsonl
                        # is the training data for everything downstream,
                        # and a kill without the artifact it was passed on
                        # cannot be re-examined later.
                        "spec": spec,
                    },
                )
            )
            killed[-1]["kill_codes"] = report.kill_codes
            continue

        fit = V.score_venue_fit(
            spec, venue=venue, card=card.to_dict(), rules_path=venue_rules_path
        )
        # R7 (seam failure) as a deterministic ranking term, not a kill:
        # a spec the pipeline cannot consume is not deleted, it is sorted
        # below every spec that loads, with the loader's own error string
        # recorded. That guarantees rank 1 is consumable whenever any
        # survivor is, without a KILL rule whose false-negative rate this
        # slice has no way to measure.
        spec_path = out / "specs" / f"{card.candidate_id}.json"
        _write_json(spec_path, spec)
        survivors.append(
            {
                "candidate_id": card.candidate_id,
                "card": card.to_dict(),
                "spec": spec,
                "feasibility": report.to_dict(),
                "venue_fit": fit.to_dict(),
                "seam_check": _seam_check(spec_path),
            }
        )

    # --- 4. deterministic ranking --------------------------------------
    weight_vf = float(
        _cfg(
            config, "ideation", "tournament", "weight_venue_fit",
            default=DEFAULT_WEIGHT_VENUE_FIT,
        )
    )
    weight_pen = float(
        _cfg(
            config, "ideation", "tournament", "weight_feasibility_penalty",
            default=DEFAULT_WEIGHT_PENALTY,
        )
    )
    ranked = rank_survivors(survivors, weight_vf=weight_vf, weight_pen=weight_pen)
    for entry in survivors:
        entry["rank"] = next(
            r["rank"] for r in ranked if r["candidate_id"] == entry["candidate_id"]
        )
        entry["deterministic_score"] = next(
            r["score"] for r in ranked if r["candidate_id"] == entry["candidate_id"]
        )

    # --- 5. artifacts ---------------------------------------------------
    _write_jsonl(out / "candidates.jsonl", survivors)
    _write_jsonl(out / "killed.jsonl", killed)
    _write_json(out / "feasibility.json", reports)

    seam = {"checked": False}
    if ranked:
        top_id = ranked[0]["candidate_id"]
        top = next(s for s in survivors if s["candidate_id"] == top_id)
        spec_path = out / "rank1_spec.json"
        _write_json(spec_path, top["spec"])
        seam = _seam_check(spec_path)

    ranking = {
        "tournament_id": tournament_id,
        "stage": "generate",
        "method": "deterministic_only",
        "method_note": (
            "venue_fit and the feasibility penalty only. No judged term, no "
            "prior-art veto, no Bradley-Terry fit - those arrive in T1b. "
            "C1: no novelty term exists at any weight."
        ),
        "weights": {
            "venue_fit": weight_vf,
            "feasibility_penalty": weight_pen,
        },
        "tie_breaks": [
            "0. the compiled spec loads through "
            "src.main.load_locked_research_spec (a spec the pipeline "
            "cannot consume sorts last; it is NOT killed)",
            "1. deterministic score (venue_fit * w_vf - penalty * w_pen)",
            "2. fewer WARN codes in the feasibility report",
            "3. higher raw venue_fit score",
            "4. lexicographic candidate_id (seeded, so reproducible)",
            "NOT AVAILABLE in T1a: prior-art verdict (rule 4 of spec sec. "
            "4.2) and the consecutive-winner diversity rule (rule 5), both "
            "of which need artifacts T1b produces.",
        ],
        "diversity_ledger": _diversity_ledger(survivors, ranked),
        "seam_check_rank1": seam,
        "notes": notes,
        "ranking": ranked,
    }
    _write_json(out / "ranking_deterministic.json", ranking)

    summary = {
        "tournament_id": tournament_id,
        "stage": "generate",
        "out_dir": str(out),
        "seed": seed,
        "generator_model": generator_model or "",
        "slate_cells": len(slate.cells),
        "generation": result.summary(),
        "survivors": len(survivors),
        "survivor_specs_the_pipeline_cannot_load": sum(
            1 for row in ranked if row["spec_loads"] is False
        ),
        "killed": len(killed),
        "kill_codes": _count(row.get("kill_code") for row in killed),
        "rank1": ranked[0]["candidate_id"] if ranked else None,
        "seam_check_rank1": seam,
        "diversity_ledger": ranking["diversity_ledger"],
        "notes": notes,
    }
    _write_json(out / "generate_summary.json", summary)
    return summary


def rank_survivors(
    survivors: Sequence[dict],
    *,
    weight_vf: float = DEFAULT_WEIGHT_VENUE_FIT,
    weight_pen: float = DEFAULT_WEIGHT_PENALTY,
) -> list[dict]:
    """Deterministic ordering. Higher score first; ties broken in order.

    Every row carries the evidence strings behind both terms (C2), so a
    ranking can be audited without re-running anything.
    """
    rows: list[dict] = []
    for entry in survivors:
        report = entry["feasibility"]
        fit = entry["venue_fit"]
        seam = entry.get("seam_check") or {}
        penalty = float(report.get("penalty") or 0.0)
        fit_score = float(fit.get("score") or 0.0)
        warn_codes = [
            check["code"]
            for check in report.get("checks") or []
            if check.get("status") == "WARN"
        ]
        rows.append(
            {
                "candidate_id": entry["candidate_id"],
                "spec_loads": bool(seam.get("passed")) if seam.get("checked") else None,
                "score": round(weight_vf * fit_score - weight_pen * penalty, 4),
                "venue_fit_score": fit_score,
                "feasibility_penalty": round(penalty, 4),
                "feasibility_verdict": report.get("verdict"),
                "analytic_n_estimate": report.get("analytic_n_estimate"),
                "warn_codes": warn_codes,
                "cell": (entry.get("card") or {}).get("cell", {}),
                "evidence": {
                    "venue_fit": [
                        f"{hit['code']} {hit['delta']:+.2f}: {hit['why']} "
                        f"[anchor evidence: {hit['evidence']}]"
                        for hit in fit.get("hits") or []
                    ],
                    "feasibility_penalty": [
                        f"{check['code']} +{check['penalty']:.2f}: "
                        f"{check['message']} [read: {check['evidence']}]"
                        for check in report.get("checks") or []
                        if check.get("status") == "WARN"
                    ],
                    "seam": [
                        (
                            "compiled spec loads through "
                            f"{seam.get('loader')}"
                            if seam.get("passed")
                            else f"compiled spec REJECTED by "
                            f"{seam.get('loader')}: {seam.get('error')}"
                        )
                    ]
                    if seam.get("checked")
                    else [],
                },
            }
        )
    rows.sort(
        key=lambda row: (
            0 if row["spec_loads"] is not False else 1,
            -row["score"],
            len(row["warn_codes"]),
            -row["venue_fit_score"],
            row["candidate_id"],
        )
    )
    for index, row in enumerate(rows, start=1):
        row["rank"] = index
    return rows


def _seam_check(spec_path: Path) -> dict:
    """Does the top-ranked compiled spec load through the real loader?

    R7 is the seam risk: a winner the pipeline cannot consume. Checking
    it here, on every generate run, costs nothing and turns a T1b
    surprise into a T1a line item.
    """
    try:
        from src.main import load_locked_research_spec

        load_locked_research_spec(str(spec_path))
        return {
            "checked": True,
            "passed": True,
            "loader": "src.main.load_locked_research_spec",
            "path": str(spec_path),
        }
    except Exception as exc:
        return {
            "checked": True,
            "passed": False,
            "loader": "src.main.load_locked_research_spec",
            "path": str(spec_path),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _count(values: Iterable[Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        key = str(value)
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def _diversity_ledger(survivors: Sequence[dict], ranked: Sequence[dict]) -> dict:
    """The V4 metric-vs-artifact audit line, computed over the survivors.

    Printed, never buried: if the top 5 collapse onto one outcome family
    or one dataset, that is a red flag regardless of score.
    """
    top_ids = {row["candidate_id"] for row in ranked[:5]}
    top = [s for s in survivors if s["candidate_id"] in top_ids]

    def _facet(rows: Sequence[dict], key: str) -> dict[str, int]:
        return _count((row.get("card") or {}).get("cell", {}).get(key) for row in rows)

    targets = _count(
        (row.get("card") or {}).get("resolved_target") for row in top
    )
    known_targets = {k: v for k, v in targets.items() if k != "None"}
    return {
        "n_survivors": len(survivors),
        "top_k": len(top),
        "top_datasets": _facet(top, "dataset"),
        "top_task_types": _facet(top, "task_type"),
        "top_opportunity_patterns": _facet(top, "opportunity_pattern"),
        "top_resolved_targets": targets,
        "all_datasets": _facet(survivors, "dataset"),
        "all_task_types": _facet(survivors, "task_type"),
        "all_opportunity_patterns": _facet(survivors, "opportunity_pattern"),
        "collapsed_to_one_dataset": len(_facet(top, "dataset")) <= 1 and len(top) > 1,
        # Only a collapse onto a KNOWN target counts. Cards whose target
        # could not be resolved are unknown, not identical, and calling
        # that a collapse would raise a false alarm on exactly the cards
        # we know least about.
        "collapsed_to_one_target": len(known_targets) == 1
        and len(top) > 1
        and sum(known_targets.values()) == len(top),
        "unresolved_targets_in_top_k": targets.get("None", 0),
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--stage", default="generate", choices=["generate"],
        help="T1a ships the generate stage only",
    )
    parser.add_argument("--tournament-id", default=None, dest="tournament_id")
    parser.add_argument("--ideas-dir", default=DEFAULT_IDEAS_DIR, dest="ideas_dir")
    parser.add_argument("--out-dir", default=None, dest="out_dir")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--n-candidates", type=int, default=None, dest="n_candidates"
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--task-types", nargs="*", default=None, dest="task_types")
    parser.add_argument("--registry-dir", default=None, dest="registry_dir")
    parser.add_argument("--raw-data-dir", default=None, dest="raw_data_dir")
    parser.add_argument("--cache-dir", default=None, dest="cache_dir")
    parser.add_argument("--venue", default=None)
    parser.add_argument(
        "--dedupe-cosine", type=float, default=None, dest="dedupe_cosine"
    )
    parser.add_argument(
        "--max-attempts", type=int, default=2, dest="max_attempts",
        help="LLM attempts per cell before the card is killed as unparseable",
    )
    parser.add_argument(
        "--probes", action="store_true",
        help="also run the Stage-1 data probes (needs the raw data files)",
    )
    parser.add_argument(
        "--offline", action="store_true",
        help=(
            "no LLM: draw every card from a deterministic template stub. "
            "Exercises the slate, compile, screen, rank and artifact "
            "plumbing without spending anything."
        ),
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    config: dict = {}
    try:
        from src.config import load_config

        config = load_config(args.config)
    except Exception as exc:
        print(f"WARNING: could not load {args.config!r} ({exc}); using defaults",
              file=sys.stderr)

    ideas_dir = Path(args.ideas_dir)
    tournament_id = args.tournament_id or next_tournament_id(ideas_dir)
    out_dir = Path(args.out_dir) if args.out_dir else ideas_dir / tournament_id

    n_candidates = args.n_candidates
    if n_candidates is None:
        n_candidates = int(
            _cfg(
                config, "ideation", "tournament", "n_candidates",
                default=S.DEFAULT_N_CANDIDATES,
            )
        )
    seed = args.seed
    if seed is None:
        seed = int(
            _cfg(
                config, "ideation", "tournament", "random_state",
                default=_cfg(
                    config, "pipeline", "random_state", default=S.DEFAULT_SEED
                ),
            )
        )

    def _log(message: str) -> None:
        if not args.quiet:
            print(message)

    summary = run_generate_stage(
        tournament_id=tournament_id,
        out_dir=out_dir,
        config=config,
        n_candidates=n_candidates,
        seed=seed,
        datasets=args.datasets,
        task_types=args.task_types,
        registry_dir=args.registry_dir,
        raw_data_dir=args.raw_data_dir,
        cache_dir=args.cache_dir,
        run_probes=args.probes,
        dedupe_cosine=args.dedupe_cosine,
        max_attempts=args.max_attempts,
        venue=args.venue,
        offline=args.offline,
        log=_log,
    )
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    if summary.get("error"):
        return 2
    return 0 if summary.get("survivors") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
