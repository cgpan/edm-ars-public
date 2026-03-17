from __future__ import annotations

import json
import os
import re
import shutil
from typing import Any

from src.agents.base import BaseAgent

# Path to the deterministic analysis helpers module (relative to this file)
_HELPERS_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_helpers.py")

# Required top-level keys in results.json
_REQUIRED_KEYS = {
    "best_model",
    "best_metric_value",
    "primary_metric",
    "all_models",
    "top_features",
    "subgroup_performance",
    "figures_generated",
    "tables_generated",
    "errors",
    "warnings",
}

# Default AUC threshold above which leakage is suspected (overridden by TaskTemplate)
_AUC_SUSPICION_THRESHOLD_DEFAULT = 0.95

# Backward-compatible alias for tests that import this constant
_AUC_SUSPICION_THRESHOLD = _AUC_SUSPICION_THRESHOLD_DEFAULT

# ---------------------------------------------------------------------------
# Error classification for targeted repair prompts (inspired by AutoResearchClaw)
# ---------------------------------------------------------------------------

_REPAIR_HINTS: dict[str, str] = {
    "ImportError": (
        "The error is an ImportError. Only use packages available in the sandbox: "
        "scikit-learn, xgboost, shap, pandas, numpy, matplotlib, seaborn. "
        "Do NOT import lightgbm — use xgboost instead."
    ),
    "MemoryError": (
        "The error is a MemoryError. Reduce memory usage: "
        "(1) set n_estimators ≤ 100 for RF/XGBoost; "
        "(2) set SHAP sample cap to ≤ 500 rows; "
        "(3) avoid storing large intermediate arrays."
    ),
    "ConvergenceWarning": (
        "The error involves convergence. Fix: "
        "increase MLP max_iter to 1000, add early_stopping=True; "
        "increase LogisticRegression max_iter to 2000."
    ),
    "FileNotFoundError": (
        "The error is a FileNotFoundError. "
        "All file paths MUST be ABSOLUTE. "
        "Use the paths from the ## Data File Paths section in the original prompt exactly."
    ),
    "SHAPTimeout": (
        "SHAP computation timed out. Apply the fallback rule: "
        "skip SHAP for MLP and use the next-best non-MLP individual model for all SHAP outputs. "
        "Document the fallback in results.json.warnings."
    ),
    "ValueError": (
        "The error is a ValueError. Check: "
        "(1) that y arrays have the correct shape (1-D); "
        "(2) that label encoders are fit on train, then applied to test; "
        "(3) that no NaN values remain in the feature matrices."
    ),
    "TypeError": (
        "The error is a TypeError. Check: "
        "that all feature columns are numeric after encoding, "
        "and that sparse matrices are converted to dense where required by SHAP."
    ),
    "RuntimeError": (
        "Fix the runtime error shown in stderr. "
        "Wrap each model training block in a try/except that logs errors to "
        "results['errors'] and continues to the next model."
    ),
}


def _classify_error(stderr: str) -> str:
    """Classify an execution error from stderr into a category for targeted repair."""
    s = stderr.lower()
    if "importerror" in s or "modulenotfounderror" in s or "no module named" in s:
        return "ImportError"
    if "memoryerror" in s or "out of memory" in s or "oom" in s:
        # Docker OOM-kill produces exit code 137 and "killed" in stderr/stdout
        return "MemoryError"
    if "convergencewarning" in s or "did not converge" in s or "max_iter" in s:
        return "ConvergenceWarning"
    if "filenotfounderror" in s or "no such file or directory" in s:
        return "FileNotFoundError"
    if "shap" in s and ("timeout" in s or "timeouterror" in s or "timed out" in s):
        return "SHAPTimeout"
    if "valueerror" in s:
        return "ValueError"
    if "typeerror" in s:
        return "TypeError"
    return "RuntimeError"


class Analyst(BaseAgent):
    """Trains, tunes, and evaluates an ML model battery on prepared HSLS:09 splits.

    Generates a Python analysis script via the LLM, executes it in a subprocess,
    reads results.json from the output directory, validates the schema, and flags
    suspicious AUC values.
    """

    MAX_RETRIES = 3
    # Generous timeout for code that includes SHAP computation (SPEC §4.3)
    EXEC_TIMEOUT_S = 600

    def run(
        self,
        data_report: dict | None = None,
        research_spec: dict | None = None,
        revision_instructions: str | None = None,
        **kwargs: Any,
    ) -> dict:
        report = data_report if data_report is not None else self.ctx.data_report
        spec = research_spec if research_spec is not None else self.ctx.research_spec

        if report is None:
            raise ValueError("data_report is required but not found in kwargs or context")
        if spec is None:
            raise ValueError("research_spec is required but not found in kwargs or context")

        # Copy deterministic helpers into output_dir so generated code can import them
        helpers_dst = os.path.join(self.ctx.output_dir, "analysis_helpers.py")
        try:
            shutil.copy2(_HELPERS_SRC, helpers_dst)
        except OSError as exc:
            self.ctx.log.append({
                "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
                "agent": self.agent_name,
                "message": f"WARNING: Could not copy analysis_helpers.py: {exc}",
            })

        user_message = self._build_user_message(report, spec, revision_instructions)
        llm_response = self.call_llm(user_message, max_tokens=16000)

        try:
            code = self._extract_code_block(llm_response)
        except ValueError:
            # LLM omitted code fences — re-prompt explicitly
            llm_response = self.call_llm(
                user_message
                + "\n\nIMPORTANT: Your previous response contained no ```python code block. "
                "You MUST output your complete Python solution inside a ```python ... ``` fence. "
                "Start your response with ```python on the first line.",
                max_tokens=16000,
            )
            code = self._extract_code_block(llm_response)
        last_response = llm_response

        # Execute with up to MAX_RETRIES retry attempts on failure
        for attempt in range(self.MAX_RETRIES + 1):
            exec_result = self.execute_code(code, timeout_s=self.EXEC_TIMEOUT_S)
            if exec_result["returncode"] == 0:
                break
            if attempt == self.MAX_RETRIES:
                self.ctx.errors.append(
                    f"Analyst: code execution failed after {self.MAX_RETRIES + 1} "
                    f"attempts. Last stderr: {exec_result['stderr'][:500]}"
                )
                break
            partial_results = self._read_partial_results()
            fix_message = self._build_fix_message(code, exec_result, attempt + 1, partial_results)
            fix_response = self.call_llm(fix_message, max_tokens=16000)
            last_response = fix_response
            try:
                code = self._extract_code_block(fix_response)
            except ValueError:
                # LLM returned no new code block — stop retrying
                break

        results = self._read_results(last_response)
        results = self._validate_results(results)

        # Persist to disk
        results_path = os.path.join(self.ctx.output_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        return results

    # ------------------------------------------------------------------
    # Message builders
    # ------------------------------------------------------------------

    def _build_user_message(
        self,
        data_report: dict,
        research_spec: dict,
        revision_instructions: str | None,
    ) -> str:
        output_dir = self.ctx.output_dir
        parts = [
            "## Data Report",
            "```json",
            json.dumps(data_report, indent=2),
            "```",
            "",
            "## Research Specification",
            "```json",
            json.dumps(research_spec, indent=2),
            "```",
            "",
            "## Data File Paths",
            f"- train_X: `{os.path.join(output_dir, 'train_X.csv')}`",
            f"- train_y: `{os.path.join(output_dir, 'train_y.csv')}`",
            f"- test_X:  `{os.path.join(output_dir, 'test_X.csv')}`",
            f"- test_y:  `{os.path.join(output_dir, 'test_y.csv')}`",
            f"- test_protected (subgroup labels, pre-encoding): "
            f"`{os.path.join(output_dir, 'test_protected.csv')}`",
            "",
            "## Analysis Helpers (REQUIRED — import and use these, do NOT reimplement)",
            "`analysis_helpers.py` is pre-installed in your working directory.",
            "```python",
            "import analysis_helpers",
            "",
            "# SHAP — always use safe_shap_values; never call explainer.shap_values() directly",
            "shap_vals = analysis_helpers.safe_shap_values(explainer, X_shap)",
            "",
            "# SHAP plots",
            "saved = analysis_helpers.save_shap_plots(shap_vals, X_shap, output_dir)",
            "",
            "# PDP plots (top 3 features by mean |SHAP|)",
            "saved += analysis_helpers.save_pdp_plots(model, train_X, top_feat_names, output_dir)",
            "",
            "# Subgroup analysis — ALWAYS use this; never reconstruct from test_X columns",
            "subgroup_results = analysis_helpers.run_subgroup_analysis(",
            "    model, test_X, test_y_arr, 'test_protected.csv',",
            "    research_spec['subgroup_analyses'], is_classification, warnings_list",
            ")",
            "",
            "# Bootstrap CI",
            "ci_lower, ci_upper = analysis_helpers.bootstrap_ci(y_true, y_pred, metric_fn)",
            "```",
            "",
            "## Output Directory",
            f"`{output_dir}`",
            "",
            "## Task",
            (
                "Generate Python analysis code that trains the six-model battery "
                "(Logistic/Linear Regression, Random Forest, XGBoost, ElasticNet, MLP, "
                "StackingEnsemble), tunes hyperparameters via 5-fold inner CV on the training "
                "set only for RF/XGBoost/ElasticNet/MLP, builds StackingEnsemble from the 5 "
                "tuned base models, evaluates all 6 models on the held-out test set, computes "
                "SHAP interpretability outputs for the best individual model (StackingEnsemble "
                "excluded from SHAP), generates all required figures and CSVs, and writes "
                "results.json to the Output Directory above. "
                "Use absolute paths for all file writes — do NOT rely on the working directory."
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
        self,
        code: str,
        exec_result: dict,
        attempt: int,
        partial_results: dict | None = None,
    ) -> str:
        error_type = _classify_error(exec_result["stderr"])
        hint = _REPAIR_HINTS.get(error_type, _REPAIR_HINTS["RuntimeError"])

        # Use last 3K of stderr (AutoResearchClaw pattern) for maximum context
        stderr_snippet = exec_result["stderr"][-3000:]
        stdout_snippet = exec_result["stdout"][-500:]

        parts = [
            f"The Python analysis code failed (attempt {attempt}/{self.MAX_RETRIES}).",
            "",
            f"**Error type: {error_type}**",
            f"**Targeted fix required:** {hint}",
            "",
            "## Failed Code",
            f"```python\n{code}\n```",
            "",
            "## stderr (last 3 000 chars)",
            f"```\n{stderr_snippet}\n```",
        ]

        if stdout_snippet.strip():
            parts += [
                "",
                "## stdout (last 500 chars)",
                f"```\n{stdout_snippet}\n```",
            ]

        # If partial results were written before the crash, tell the LLM to preserve them
        if partial_results and partial_results.get("all_models"):
            completed_models = list(partial_results["all_models"].keys())
            parts += [
                "",
                "## Partial Results Already Written",
                f"These models already completed and are in results.json: {completed_models}.",
                "Preserve them — do NOT re-train models that already succeeded.",
                "Only fix the failing model(s) and write the complete merged results.json.",
            ]

        parts += [
            "",
            "Output a corrected ```python code block. "
            "Apply all the same analysis requirements: "
            "six models (LR, RF, XGBoost, ElasticNet, MLP, StackingEnsemble); "
            "inner CV tuning; test-set evaluation only; "
            "SHAP for best individual model; all figures and CSVs; results.json.",
        ]
        return "\n".join(parts)

    def _read_partial_results(self) -> dict | None:
        """Read results.json if it was partially written before a crash."""
        results_path = os.path.join(self.ctx.output_dir, "results.json")
        if not os.path.exists(results_path):
            return None
        try:
            with open(results_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

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

    def _read_results(self, fallback_llm_response: str) -> dict:
        """Read results.json written by the generated code, or fall back to LLM JSON."""
        results_path = os.path.join(self.ctx.output_dir, "results.json")
        if os.path.exists(results_path):
            with open(results_path) as f:
                return json.load(f)

        # Try to parse the JSON block from the LLM response
        try:
            raw_json = self._extract_json_block(fallback_llm_response)
            return json.loads(raw_json)
        except (ValueError, json.JSONDecodeError):
            pass

        # Last resort: return a minimal failed results object
        return {
            "best_model": "",
            "best_metric_value": 0.0,
            "primary_metric": "",
            "all_models": {},
            "top_features": [],
            "subgroup_performance": {},
            "figures_generated": [],
            "tables_generated": [],
            "errors": [
                "results.json was not written and could not be parsed from LLM output"
            ],
            "warnings": [],
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_results(self, results: dict) -> dict:
        """Enforce schema requirements and flag suspicious AUC values."""
        results.setdefault("errors", [])
        results.setdefault("warnings", [])

        # Check required top-level keys
        missing_keys = _REQUIRED_KEYS - results.keys()
        if missing_keys:
            results["errors"].append(
                f"results.json is missing required keys: {sorted(missing_keys)}"
            )

        # Ensure all_models is a dict (may be empty on total failure).
        # Only add a type error when the key is present but the wrong type;
        # missing keys are already covered by the block above.
        if "all_models" in results and not isinstance(results["all_models"], dict):
            results["errors"].append("results.json 'all_models' must be a dict")
            results["all_models"] = {}
        elif "all_models" not in results:
            results["all_models"] = {}

        # Flag suspicious metric values (potential data leakage)
        primary = results.get("primary_metric", "")
        outcome_type = "binary" if primary == "AUC" else "continuous"
        eval_metrics = self.task_template.get_evaluation_metrics(outcome_type)
        suspicion_threshold = eval_metrics.get(
            "suspicion_threshold", _AUC_SUSPICION_THRESHOLD_DEFAULT
        )

        if primary == "AUC" and suspicion_threshold is not None:
            best_auc = results.get("best_metric_value", 0.0)
            if isinstance(best_auc, (int, float)) and best_auc > suspicion_threshold:
                results["warnings"].append(
                    f"Suspiciously high AUC detected: {best_auc:.4f} > "
                    f"{suspicion_threshold}. Potential data leakage — "
                    "Critic should investigate."
                )
            # Also check per-model AUC values
            for model_name, metrics in results.get("all_models", {}).items():
                model_auc = metrics.get("auc", 0.0)
                if isinstance(model_auc, (int, float)) and model_auc > suspicion_threshold:
                    results["warnings"].append(
                        f"Suspiciously high AUC for {model_name}: {model_auc:.4f}. "
                        "Potential data leakage."
                    )

        # Ensure top_features is a list
        if not isinstance(results.get("top_features"), list):
            results["warnings"].append("top_features is missing or not a list; defaulting to []")
            results["top_features"] = []

        return results
