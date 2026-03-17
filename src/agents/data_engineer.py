from __future__ import annotations

import json
import os
import re
from typing import Any

import pandas as pd
import yaml

from src.agents.base import BaseAgent, parse_llm_json


class DataEngineer(BaseAgent):
    """Prepares analysis-ready train/test splits from raw HSLS:09 data."""

    MAX_RETRIES = 3

    def run(
        self,
        research_spec: dict | None = None,
        revision_instructions: str | None = None,
        **kwargs: Any,
    ) -> dict:
        spec = research_spec if research_spec is not None else self.ctx.research_spec
        if spec is None:
            raise ValueError(
                "research_spec is required but not found in kwargs or context"
            )

        registry = self.load_registry()
        registry_yaml = yaml.dump(registry, default_flow_style=False, allow_unicode=True)

        user_message = self._build_user_message(spec, registry_yaml, revision_instructions)
        llm_response = self.call_llm(user_message, max_tokens=8192)

        try:
            code = self._extract_code_block(llm_response)
        except ValueError:
            # LLM omitted code fences — re-prompt explicitly
            llm_response = self.call_llm(
                user_message
                + "\n\nIMPORTANT: Your previous response contained no ```python code block. "
                "You MUST output your complete Python solution inside a ```python ... ``` fence. "
                "Start your response with ```python on the first line.",
                max_tokens=8192,
            )
            code = self._extract_code_block(llm_response)
        last_response = llm_response

        # Save generated code for debugging
        code_path = os.path.join(self.ctx.output_dir, "data_engineer_generated.py")
        with open(code_path, "w", encoding="utf-8") as f:
            f.write(code)

        # Execute with up to MAX_RETRIES retry attempts on failure
        for attempt in range(self.MAX_RETRIES + 1):
            exec_result = self.execute_code(code)
            if exec_result["returncode"] == 0:
                break
            if attempt == self.MAX_RETRIES:
                self.ctx.errors.append(
                    f"DataEngineer: code execution failed after {self.MAX_RETRIES + 1} "
                    f"attempts. Last stderr: {exec_result['stderr'][:500]}"
                )
                break
            fix_message = self._build_fix_message(code, exec_result, attempt + 1)
            fix_response = self.call_llm(fix_message, max_tokens=8192)
            last_response = fix_response
            try:
                code = self._extract_code_block(fix_response)
            except ValueError:
                # No new code block returned — stop retrying
                break

        data_report = self._read_data_report(last_response)
        data_report = self._validate_outputs(data_report)

        report_path = os.path.join(self.ctx.output_dir, "data_report.json")
        with open(report_path, "w") as f:
            json.dump(data_report, f, indent=2)

        return data_report

    # ------------------------------------------------------------------
    # Message builders
    # ------------------------------------------------------------------

    def _build_user_message(
        self,
        spec: dict,
        registry_yaml: str,
        revision_instructions: str | None,
    ) -> str:
        parts = [
            "## Research Specification",
            "```json",
            json.dumps(spec, indent=2),
            "```",
            "",
            "## Dataset Registry (YAML)",
            "```yaml",
            registry_yaml,
            "```",
            "",
            "## Raw Data File Path",
            f"`{self.ctx.raw_data_path}`",
            "",
            "## Output Directory",
            f"`{self.ctx.output_dir}`",
            "",
            "## Task",
            (
                "Generate Python data preparation code and the expected data_report.json. "
                "Save all output CSV files and data_report.json to the Output Directory above. "
                "Use absolute paths when writing files — do NOT rely on the working directory."
            ),
        ]
        if revision_instructions:
            parts += [
                "",
                "## Revision Instructions from Critic",
                revision_instructions,
            ]
        return "\n".join(parts)

    def _build_fix_message(
        self, code: str, exec_result: dict, attempt: int
    ) -> str:
        return (
            f"The Python code you generated failed to execute "
            f"(attempt {attempt}/{self.MAX_RETRIES}).\n\n"
            "## Failed Code\n"
            f"```python\n{code}\n```\n\n"
            "## stderr\n"
            f"```\n{exec_result['stderr'][:2000]}\n```\n\n"
            "## stdout\n"
            f"```\n{exec_result['stdout'][:500]}\n```\n\n"
            "Please output a corrected ```python code block followed by the "
            "expected ```json data_report block. Apply all the same data "
            "preparation requirements as before."
        )

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_code_block(text: str) -> str:
        """Extract the first ```python ... ``` block from LLM output."""
        match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Fall back to any fenced block
        match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        raise ValueError("No Python code block found in LLM response")

    @staticmethod
    def _extract_json_block(text: str) -> str:
        """Extract the first ```json ... ``` block from LLM output."""
        match = re.search(r"```json\s*\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        raise ValueError("No JSON block found in LLM response")

    def _read_data_report(self, fallback_llm_response: str) -> dict:
        """Read data_report.json written by the generated code, or fall back to LLM JSON."""
        report_path = os.path.join(self.ctx.output_dir, "data_report.json")
        if os.path.exists(report_path):
            with open(report_path) as f:
                return json.load(f)
        # Try to parse the JSON block from the LLM response
        try:
            raw_json = self._extract_json_block(fallback_llm_response)
            return json.loads(raw_json)
        except (ValueError, json.JSONDecodeError):
            pass
        # Last resort: return a minimal report flagged as failed
        return {
            "dataset": self.ctx.dataset_name,
            "original_n": 0,
            "analytic_n": 0,
            "n_train": 0,
            "n_test": 0,
            "outcome_variable": "",
            "outcome_type": "",
            "class_balance": None,
            "n_predictors_raw": 0,
            "n_predictors_encoded": 0,
            "missingness_summary": {},
            "variables_flagged": [],
            "validation_passed": False,
            "warnings": ["data_report.json was not written and could not be parsed from LLM output"],
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_outputs(self, data_report: dict) -> dict:
        """
        Read the generated CSVs and enforce SPEC §4.2 validation checks.
        Mutates and returns data_report with updated fields and any new warnings.
        """
        output_dir = self.ctx.output_dir
        issues: list[str] = []

        # Check all four required CSV files exist
        required_files = ["train_X.csv", "train_y.csv", "test_X.csv", "test_y.csv"]
        for fname in required_files:
            if not os.path.exists(os.path.join(output_dir, fname)):
                issues.append(f"Missing output file: {fname}")

        # Check test_protected.csv (non-fatal warning if missing — subgroup analysis
        # will be skipped by Analyst but the pipeline can continue)
        if not os.path.exists(os.path.join(output_dir, "test_protected.csv")):
            data_report.setdefault("warnings", []).append(
                "test_protected.csv was not written; subgroup analysis will be skipped."
            )

        if issues:
            data_report["validation_passed"] = False
            data_report.setdefault("warnings", []).extend(issues)
            self._ensure_multilevel_warning(data_report)
            return data_report

        # Load CSVs
        try:
            train_X = pd.read_csv(os.path.join(output_dir, "train_X.csv"))
            train_y = pd.read_csv(os.path.join(output_dir, "train_y.csv"))
            test_X = pd.read_csv(os.path.join(output_dir, "test_X.csv"))
            test_y = pd.read_csv(os.path.join(output_dir, "test_y.csv"))
        except Exception as exc:
            data_report["validation_passed"] = False
            data_report.setdefault("warnings", []).append(
                f"Failed to read output CSV files: {exc}"
            )
            self._ensure_multilevel_warning(data_report)
            return data_report

        # Check: no NaN remaining
        for name, df in [
            ("train_X", train_X),
            ("train_y", train_y),
            ("test_X", test_X),
            ("test_y", test_y),
        ]:
            nan_count = int(df.isna().sum().sum())
            if nan_count > 0:
                issues.append(f"NaN values remain in {name}: {nan_count} cells")

        # Check: no zero-variance (constant) predictors
        zero_var_cols = [
            col for col in train_X.columns if train_X[col].nunique() <= 1
        ]
        if zero_var_cols:
            issues.append(f"Zero-variance predictors found: {zero_var_cols}")

        # Update report with ground-truth counts from files
        data_report["n_train"] = len(train_X)
        data_report["n_test"] = len(test_X)
        data_report["n_predictors_encoded"] = len(train_X.columns)

        # Propagate issues
        if issues:
            data_report["validation_passed"] = False
            data_report.setdefault("warnings", []).extend(issues)

        self._ensure_multilevel_warning(data_report)
        return data_report

    def _ensure_multilevel_warning(self, data_report: dict) -> None:
        multilevel_msg = self.dataset_adapter.get_multilevel_warning()
        if multilevel_msg is None:
            return
        warnings = data_report.setdefault("warnings", [])
        if not any(multilevel_msg in w for w in warnings):
            warnings.append(multilevel_msg)
