"""EDM-ARS: Educational Data Mining Automated Research System — CLI entry point."""

import argparse
import json
import os
import sys
from datetime import datetime

import yaml
from dotenv import load_dotenv

load_dotenv()

from src.config import load_config
from src.context import PipelineContext
from src.dataset_adapter import create_dataset_adapter
from src.orchestrator import Orchestrator
from src.task_template import create_task_template


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_DATASET = "hsls09_public"

# Prose markers of validation warnings that are ADVISORY, not blocking.
# Measured 2026-07-25 over the archive: PredictionTemplate's
# sum-of-pct_missing retention rule fires on 6 of 6 archived prediction
# specs (estimated n = 0 for five of them, 1,663 for the sixth) where the
# executed runs carried analytic_n = 14,039 (ELS) and 17,335 (HSLS).
# Escalating it to a hard load failure would reject every real prediction
# spec, so it is printed and stepped over. docs/v5_arc_t_spec.md §1.4
# (task_template.py:146-167) replaces the rule with
# ``feasibility.estimate_analytic_n()``; this list shrinks to () then.
_ADVISORY_WARNING_MARKERS: tuple[str, ...] = ("Estimated analytic_n",)


def _is_advisory_warning(warning: str) -> bool:
    return any(marker in warning for marker in _ADVISORY_WARNING_MARKERS)


def _load_registry_for_dataset(dataset: str, registry_dir: str | None = None) -> dict:
    """Load a dataset registry YAML, trying cwd then the project root."""
    rel = os.path.join(registry_dir or "data_registry", "datasets", f"{dataset}.yaml")
    candidates = [rel, os.path.join(_PROJECT_ROOT, rel)]
    for candidate in candidates:
        if os.path.exists(candidate):
            with open(candidate, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    raise ValueError(
        f"No registry YAML for dataset {dataset!r}; looked in "
        f"{candidates}. A locked research_spec cannot be validated "
        f"without its dataset registry."
    )


def load_locked_research_spec(
    path: str,
    dataset: str | None = None,
    registry_dir: str | None = None,
) -> dict:
    """Load and structurally validate a locked research_spec JSON file.

    Used by the ``--research-spec`` CLI flag (Phase 3b.4 / B6) and
    callable directly by tests.

    The dataset whose registry the spec is validated against is resolved
    as ``spec["dataset"]`` -> the ``dataset`` argument -> ``hsls09_public``.
    ``prediction`` specs are validated against that registry (temporal
    ordering, Tier-3 exclusion); the causal/psychometrics templates
    ignore the registry and validate structurally.

    Raises:
        FileNotFoundError: if the path does not exist.
        json.JSONDecodeError: if the file is not valid JSON.
        ValueError: if the spec lacks ``task_type``, names an unknown
            dataset, has no loadable registry, or fails structural
            validation under the corresponding TaskTemplate.
    """
    with open(path, encoding="utf-8") as f:
        spec = json.load(f)

    if not isinstance(spec, dict):
        raise ValueError(
            f"Locked research_spec must be a JSON object (got {type(spec).__name__})"
        )

    task_type = spec.get("task_type")
    if not task_type:
        raise ValueError(
            "Locked research_spec must declare 'task_type' "
            "(e.g., 'causal_soo')"
        )

    resolved_dataset = spec.get("dataset") or dataset or _DEFAULT_DATASET
    # create_dataset_adapter raises ValueError on an unknown dataset —
    # that is itself a legitimate structural failure of the locked spec.
    adapter = create_dataset_adapter(resolved_dataset)
    registry = _load_registry_for_dataset(resolved_dataset, registry_dir)

    template = create_task_template(task_type)
    warnings = template.validate_research_spec(spec, registry, adapter)

    blocking = [w for w in warnings if not _is_advisory_warning(w)]
    for advisory in (w for w in warnings if _is_advisory_warning(w)):
        print(
            f"ADVISORY (non-blocking) for {path!r}: {advisory}",
            file=sys.stderr,
        )
    if blocking:
        joined = "\n  - ".join(blocking)
        raise ValueError(
            f"Locked research_spec at {path!r} failed structural "
            f"validation:\n  - {joined}"
        )
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EDM-ARS: Educational Data Mining Automated Research System"
    )
    parser.add_argument(
        "--dataset",
        default="hsls09_public",
        help="Dataset name (default: hsls09_public)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        dest="output_dir",
        help="Output directory (auto-generated as output/run_YYYYMMDD_HHMMSS if not specified)",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available in output directory",
    )

    parser.add_argument("--prompt", default=None, help="Optional research direction or question")

    parser.add_argument(
        "--research-spec",
        default=None,
        dest="research_spec",
        help=(
            "Path to a JSON file containing a locked research_spec. "
            "If provided, ProblemFormulator runs in 'refine' mode "
            "against this spec rather than generating a new one from "
            "scratch. The spec's 'task_type' field overrides the "
            "config.yaml pipeline.task_type."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help=(
            "Validate config + locked research_spec, instantiate the "
            "orchestrator, and exit before any LLM call. No real API "
            "spend. Useful as a pre-flight check before live runs."
        ),
    )

    args = parser.parse_args()

    config = load_config(args.config)
    task_type = config["pipeline"].get("task_type", "prediction")

    locked_spec: dict | None = None
    if args.research_spec:
        locked_spec = load_locked_research_spec(
            args.research_spec, dataset=args.dataset
        )
        # The locked spec's task_type is authoritative — it determines
        # which TaskTemplate the orchestrator instantiates.
        task_type = locked_spec["task_type"]

    # Build output directory path (absolute so subprocess cwd doesn't matter)
    if args.output_dir is not None:
        output_dir = os.path.abspath(args.output_dir)
    else:
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        output_dir = os.path.abspath(os.path.join(config["paths"]["output_base"], run_name))

    # Build raw data path using dataset adapter (no hardcoded filename)
    adapter = create_dataset_adapter(args.dataset)
    raw_data_path = os.path.abspath(os.path.join(
        config["paths"]["raw_data"],
        adapter.get_raw_data_filename(),
    ))

    ctx = PipelineContext(
        dataset_name=args.dataset,
        raw_data_path=raw_data_path,
        output_dir=output_dir,
        task_type=task_type,
        max_revision_cycles=config["pipeline"]["max_revision_cycles"],
        locked_research_spec=locked_spec,
    )

    # Without --resume, remove any stale checkpoint so we get a clean run
    if not args.resume:
        checkpoint_path = os.path.join(output_dir, "checkpoint.json")
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

    orchestrator = Orchestrator(ctx, config, config_path=args.config)

    if args.dry_run:
        # Pre-flight: validate that orchestrator init succeeded, the
        # task template registered, the skills load, the locked spec (if
        # any) is structurally valid. No LLM calls, no spend.
        print("DRY RUN — pre-flight summary:")
        print(f"  config:               {args.config}")
        print(f"  llm_provider:         {config.get('llm_provider')}")
        print(f"  task_type:            {ctx.task_type}")
        print(f"  task_template:        {type(orchestrator.task_template).__name__}")
        print(f"  dataset:              {ctx.dataset_name}")
        print(f"  raw_data_path:        {ctx.raw_data_path}")
        print(f"  raw_data exists:      {os.path.exists(ctx.raw_data_path)}")
        print(f"  output_dir:           {ctx.output_dir}")
        print(f"  locked_research_spec: {'set' if ctx.locked_research_spec else 'none'}")
        print(f"  skill_registry count: {orchestrator.skill_registry.count()}")
        print(
            f"  skills by layer:      "
            f"{orchestrator.skill_registry.count_by_layer()}"
        )
        # Render skill matches per stage so we can inspect what each
        # agent would receive without actually calling the LLM.
        for stage in (
            "ProblemFormulator",
            "DataEngineer",
            "Analyst",
            "Critic",
            "Writer",
        ):
            matched = orchestrator._match_skills_for_stage(stage)
            names = [s.name for s in matched]
            print(f"  skills @ {stage}: {len(matched)} -> {names}")
        return

    result_ctx = orchestrator.run(user_prompt=args.prompt)

    print(f"Pipeline complete. Final state: {result_ctx.current_state}")
    print(f"Output directory: {result_ctx.output_dir}")
    if result_ctx.errors:
        print(f"Errors: {result_ctx.errors}", file=sys.stderr)


if __name__ == "__main__":
    main()
