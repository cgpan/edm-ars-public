"""Test that DataEngineer copies analysis_helpers.py into the execution dir.

Mirrors the Analyst's use of `shutil.copy2(_HELPERS_SRC, ...)` so generated
DE code can `import analysis_helpers`. This is the regression test for the
Phase 2c Checkpoint 4b failure (Apr 2026), where the missing copy caused
intermittent DE crashes whenever the LLM happened to generate code that
imported the helper.
"""
from __future__ import annotations

import filecmp
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.data_engineer import _HELPERS_SRC, DataEngineer
from src.config import load_config
from src.context import PipelineContext

CONFIG_PATH = str(Path(__file__).parent.parent / "config.yaml")


def _minimal_research_spec() -> dict:
    return {
        "research_question": "Test question.",
        "outcome_variable": "X3TGPAMAT",
        "outcome_type": "continuous",
        "predictor_set": [
            {"variable": "X1TXMTSC", "rationale": "Test.", "wave": "base_year"},
        ],
        "target_population": "Test sample.",
        "subgroup_analyses": [],
        "expected_contribution": "Test.",
        "potential_limitations": [],
        "novelty_score_self_assessment": 3,
    }


def _minimal_data_report() -> dict:
    return {"validation_passed": True, "warnings": []}


class TestHelpersCopy:
    """The DataEngineer must copy analysis_helpers.py into output_dir before
    invoking the executor on the LLM-generated code."""

    def test_helpers_file_is_present_when_executor_is_called(
        self, tmp_path: Path
    ) -> None:
        config = load_config(CONFIG_PATH)
        ctx = PipelineContext(
            dataset_name="hsls09_public",
            raw_data_path="data/raw/nonexistent.csv",
            output_dir=str(tmp_path),
            max_revision_cycles=2,
        )
        ctx.research_spec = _minimal_research_spec()

        observed: dict[str, bool] = {"helper_present_at_exec_time": False}

        def _fake_execute_code(code: str, timeout_s: int = 300) -> dict:
            # Capture whether the helper file is present at the moment
            # the executor runs the generated code.
            observed["helper_present_at_exec_time"] = (tmp_path / "analysis_helpers.py").exists()
            # Short-circuit: report a "successful" run that produced nothing,
            # forcing _validate_outputs to add missing-file warnings (we don't
            # care; we just need to verify the copy happened).
            return {"stdout": json.dumps(_minimal_data_report()), "stderr": "", "returncode": 0}

        with patch("anthropic.Anthropic"):
            agent = DataEngineer(ctx, "data_engineer", config)
            agent.call_llm = lambda *a, **k: (  # type: ignore[assignment]
                "```python\nimport analysis_helpers\nimport json\n"
                f"print(json.dumps({json.dumps(_minimal_data_report())}))\n```\n"
                "```json\n" + json.dumps(_minimal_data_report()) + "\n```\n"
            )
            agent.execute_code = _fake_execute_code  # type: ignore[assignment]
            agent.run()

        assert observed["helper_present_at_exec_time"], (
            "DataEngineer must copy analysis_helpers.py into output_dir BEFORE "
            "invoking the executor; the helper was not present at exec time."
        )

    def test_copied_helpers_file_is_byte_identical_to_source(
        self, tmp_path: Path
    ) -> None:
        config = load_config(CONFIG_PATH)
        ctx = PipelineContext(
            dataset_name="hsls09_public",
            raw_data_path="data/raw/nonexistent.csv",
            output_dir=str(tmp_path),
            max_revision_cycles=2,
        )
        ctx.research_spec = _minimal_research_spec()

        with patch("anthropic.Anthropic"):
            agent = DataEngineer(ctx, "data_engineer", config)
            agent.call_llm = lambda *a, **k: (  # type: ignore[assignment]
                "```python\nimport json\nprint(json.dumps({}))\n```\n"
                "```json\n" + json.dumps(_minimal_data_report()) + "\n```\n"
            )
            agent.execute_code = (  # type: ignore[assignment]
                lambda code, timeout_s=300: {
                    "stdout": json.dumps(_minimal_data_report()),
                    "stderr": "",
                    "returncode": 0,
                }
            )
            agent.run()

        copied = tmp_path / "analysis_helpers.py"
        assert copied.exists(), "helpers file was not copied into output_dir"
        assert filecmp.cmp(_HELPERS_SRC, str(copied), shallow=False), (
            "Copied helpers file must be byte-identical to the source."
        )
