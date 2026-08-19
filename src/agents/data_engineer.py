from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime
from typing import Any

import pandas as pd
import yaml

from src.agents.base import BaseAgent, parse_llm_json

# Path to the deterministic analysis helpers module. Mirrors the constant
# in src/agents/analyst.py so both agents copy from the same source. The
# helpers live at src/analysis_helpers.py -- one level up from this file.
_HELPERS_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "analysis_helpers.py",
)


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

        # Copy deterministic helpers into output_dir so generated code can
        # `import analysis_helpers`. Mirrors the Analyst's pattern
        # (src/agents/analyst.py); pre-Phase-2c the DataEngineer was
        # missing this copy and the helper-using code path failed
        # intermittently. Failure to copy is logged but non-fatal — some
        # generated code may inline reconstruction logic and not need the
        # helper.
        helpers_dst = os.path.join(self.ctx.output_dir, "analysis_helpers.py")
        try:
            shutil.copy2(_HELPERS_SRC, helpers_dst)
        except OSError as exc:
            self.ctx.log.append({
                "timestamp": datetime.utcnow().isoformat(),
                "agent": self.agent_name,
                "message": f"WARNING: Could not copy analysis_helpers.py: {exc}",
            })

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
        with open(report_path, "w", encoding="utf-8") as f:
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

    @staticmethod
    def _stderr_hint(stderr: str) -> str:
        """Targeted diagnosis hints for cryptic pandas/sklearn errors the
        LLM repeatedly fails to root-cause from the traceback alone
        (F-A2-DE-DUPLICATE-KEEP-COLS: deepseek saw the to_numeric
        TypeError three times without connecting it to duplicate column
        labels)."""
        hints = []
        if "arg must be a list, tuple, 1-d array, or Series" in stderr:
            hints.append(
                "DIAGNOSIS: this TypeError almost always means DUPLICATE "
                "COLUMN LABELS — `df[col]` returned a DataFrame, not a "
                "Series, because `col` appears twice (e.g. a variable "
                "listed in both predictors and subgroup columns). Build "
                "the column selection with list(dict.fromkeys([...])) "
                "so every label is unique."
            )
        if "cannot reindex on an axis with duplicate labels" in stderr:
            hints.append(
                "DIAGNOSIS: duplicate column labels in the frame — "
                "de-duplicate the selection list before slicing."
            )
        return ("\n\n" + "\n".join(hints)) if hints else ""

    def _build_fix_message(
        self, code: str, exec_result: dict, attempt: int
    ) -> str:
        return (
            f"The Python code you generated failed to execute "
            f"(attempt {attempt}/{self.MAX_RETRIES}).\n\n"
            "## Failed Code\n"
            f"```python\n{code}\n```\n\n"
            "## stderr\n"
            f"```\n{exec_result['stderr'][:2000]}\n```"
            f"{self._stderr_hint(exec_result.get('stderr', ''))}\n\n"
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
            with open(report_path, encoding="utf-8") as f:
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

        causal_did runs use a split-less panel contract instead — see
        :meth:`_validate_panel_outputs`.
        """
        spec = getattr(self.ctx, "research_spec", None) or {}
        if spec.get("task_type") == "causal_did":
            return self._validate_panel_outputs(data_report)
        if spec.get("task_type") == "psychometrics":
            return self._validate_items_outputs(data_report)

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

        # Check: no zero-variance (constant) predictors. Phase 3b.5 narrow
        # exception #1: drop+warn instead of fail. Zero-variance one-hot
        # columns are an artifact of school-aware splits where rare
        # categories appear in test but not train (or vice versa). They
        # carry no predictive/causal signal and the prediction model
        # battery + causal estimators are robust to dropping them. The
        # original behavior (validation fail) blocks downstream Analyst /
        # Critic / Writer / LSAR stages from running on otherwise-valid
        # data — that is in scope for unblocking under the in-phase fix
        # policy. The DataEngineer agent's generated code should also
        # drop these proactively (post-3b.5 skill-content work).
        zero_var_cols = [
            col for col in train_X.columns if train_X[col].nunique() <= 1
        ]
        if zero_var_cols:
            train_X = train_X.drop(columns=zero_var_cols)
            test_X = test_X.drop(columns=zero_var_cols, errors="ignore")
            train_X.to_csv(os.path.join(output_dir, "train_X.csv"), index=False)
            test_X.to_csv(os.path.join(output_dir, "test_X.csv"), index=False)
            data_report.setdefault("warnings", []).append(
                f"Dropped {len(zero_var_cols)} zero-variance predictor(s) "
                f"introduced during one-hot encoding (rare categories in "
                f"test but not train, or vice versa): {zero_var_cols}"
            )

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

    def _validate_panel_outputs(self, data_report: dict) -> dict:
        """causal_did panel contract (V4 Phase B): split-less validation.

        Checks panel_analytic.csv against the research_spec's design
        columns instead of the prediction-mode train/test artifacts.
        """
        output_dir = self.ctx.output_dir
        spec = getattr(self.ctx, "research_spec", None) or {}
        issues: list[str] = []

        path = os.path.join(output_dir, "panel_analytic.csv")
        if not os.path.exists(path):
            issues.append("Missing output file: panel_analytic.csv")
        else:
            try:
                panel = pd.read_csv(path)
            except Exception as exc:
                panel = None
                issues.append(f"Failed to read panel_analytic.csv: {exc}")
            if panel is not None:
                group = spec.get("group_variable", "")
                post = spec.get("post_variable", "")
                outcome = (spec.get("outcome") or {}).get("variable", "")
                required = [c for c in (group, post, outcome) if c]
                missing = [c for c in required if c not in panel.columns]
                if missing:
                    issues.append(
                        f"panel_analytic.csv missing required columns: {missing}"
                    )
                else:
                    if len(panel) < 1000:
                        issues.append(
                            f"Panel too small: {len(panel)} rows < 1000"
                        )
                    if panel[outcome].isna().any():
                        issues.append(
                            f"Primary outcome '{outcome}' has NaN rows — "
                            "drop them, never impute an outcome."
                        )
                    for col in (group, post):
                        vals = set(pd.unique(panel[col].dropna()))
                        if not vals <= {0, 1}:
                            issues.append(
                                f"Design column '{col}' is not binary 0/1: "
                                f"{sorted(vals)[:6]}"
                            )
                    cells = panel.groupby([group, post]).size()
                    if len(cells) < 4 or int(cells.min()) < 200:
                        issues.append(
                            f"Degenerate 2x2 design: cell counts "
                            f"{cells.to_dict()} (need all four cells >= 200)"
                        )
                    data_report["analytic_n"] = len(panel)
                    data_report["n_train"] = len(panel)
                    data_report["n_test"] = 0  # no split by design

        if issues:
            data_report["validation_passed"] = False
            data_report.setdefault("warnings", []).extend(issues)
        self._ensure_multilevel_warning(data_report)
        return data_report


    def _validate_items_outputs(self, data_report: dict) -> dict:
        """psychometrics item-matrix contract (V4): split-less validation.

        Checks items_analytic.csv against the research_spec's item and
        grouping columns; items must be integer categories or NaN, never
        imputed (missingness is expected and reported, not an error).
        """
        output_dir = self.ctx.output_dir
        spec = getattr(self.ctx, "research_spec", None) or {}
        issues: list[str] = []

        path = os.path.join(output_dir, "items_analytic.csv")
        if not os.path.exists(path):
            issues.append("Missing output file: items_analytic.csv")
        else:
            try:
                items = pd.read_csv(path)
            except Exception as exc:
                items = None
                issues.append(f"Failed to read items_analytic.csv: {exc}")
            if items is not None:
                item_cols = spec.get("item_columns") or []
                if spec.get("item_construction"):
                    # log-mode WINS over any item_columns the PF refine
                    # step invented (F-W2-PF-ITEMCOLS: its output schema
                    # demands the field, so it fills it with junk)
                    item_cols = []
                if not item_cols and spec.get("item_construction"):
                    # log-mode: items are DERIVED (t<template_id> columns)
                    item_cols = [c for c in items.columns
                                 if str(c).startswith("t")
                                 and str(c) != "user_id"]
                    if len(item_cols) < 3:
                        issues.append(
                            "Log-mode item matrix has fewer than 3 derived "
                            f"item columns ({len(item_cols)})")
                    if not os.path.exists(os.path.join(output_dir,
                                                       "q_matrix.json")):
                        issues.append(
                            "Missing q_matrix.json (required in "
                            "item_construction mode)")
                group_cols = spec.get("grouping_vars") or []
                missing = [c for c in item_cols + group_cols
                           if c not in items.columns]
                if missing:
                    issues.append(
                        f"items_analytic.csv missing columns: {missing}")
                else:
                    if len(items) < 1000:
                        issues.append(
                            f"Item matrix too small: {len(items)} rows < 1000")
                    for c in item_cols:
                        v = pd.to_numeric(items[c], errors="coerce")
                        vals = set(v.dropna().unique())
                        if not vals:
                            issues.append(f"Item '{c}' has no observed values")
                        elif not all(float(x).is_integer() and 0 <= x <= 10
                                     for x in vals):
                            # 0/1 = binary correctness (CDM/log data);
                            # 1..k = Likert categories - both valid
                            issues.append(
                                f"Item '{c}' has non-categorical values: "
                                f"{sorted(vals)[:6]} (expect integers 0-10; "
                                "sentinels must be NaN)")
                        if v.notna().mean() < 0.5:
                            data_report.setdefault("warnings", []).append(
                                f"Item '{c}' is {100*(1-v.notna().mean()):.0f}% "
                                "missing - flag as limitation.")
                    data_report["analytic_n"] = int(len(items))
                    data_report["n_train"] = int(len(items))
                    data_report["n_test"] = 0  # no split by design

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
