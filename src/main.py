"""EDM-ARS: Educational Data Mining Automated Research System — CLI entry point."""

import argparse
import os
import sys
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

from src.config import load_config
from src.context import PipelineContext
from src.dataset_adapter import create_dataset_adapter
from src.orchestrator import Orchestrator


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

    args = parser.parse_args()

    config = load_config(args.config)
    task_type = config["pipeline"].get("task_type", "prediction")

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
    )

    # Without --resume, remove any stale checkpoint so we get a clean run
    if not args.resume:
        checkpoint_path = os.path.join(output_dir, "checkpoint.json")
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

    orchestrator = Orchestrator(ctx, config, config_path=args.config)
    result_ctx = orchestrator.run(user_prompt=args.prompt)

    print(f"Pipeline complete. Final state: {result_ctx.current_state}")
    print(f"Output directory: {result_ctx.output_dir}")
    if result_ctx.errors:
        print(f"Errors: {result_ctx.errors}", file=sys.stderr)


if __name__ == "__main__":
    main()
