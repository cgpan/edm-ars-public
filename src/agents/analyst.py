from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime
from typing import Any

from src.agents.base import BaseAgent

# Path to the deterministic analysis helpers module (relative to this file).
# The helpers live at src/analysis_helpers.py -- one level up from this
# file (which lives in src/agents/). The pre-Phase-2c-recovery version
# pointed at src/agents/analysis_helpers.py and silently failed every
# copy; the LLM retry loop masked the bug because the LLM eventually
# generated inline code that didn't need the helper.
_HELPERS_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "analysis_helpers.py",
)
# V4 psychometrics: the psy_* helpers call the R bridge; copy it flat
# next to analysis_helpers.py so output-dir execution can import it.
_R_BRIDGE_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "r_bridge.py",
)

# Required top-level keys in results.json for the PREDICTION task type
# (SPEC §4.3). This set is prediction-shaped: `all_models`, `top_features`
# and `subgroup_performance` do not exist in a psychometrics or causal
# results object. Do NOT apply it unconditionally — resolve the active
# task type's contract through Analyst._required_results_keys() instead
# (G4 / F-P5-PSY-SCHEMA-KEYS: applying this set to every task type
# injected a phantom "results.json is missing required keys: [...]" error
# into every psychometrics and causal run, which the Critic then read as
# a genuine analysis failure).
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

# Fallback contract for task types whose TaskTemplate declares no
# required results keys: the two lists every downstream consumer reads.
_MINIMUM_REQUIRED_KEYS = {"errors", "warnings"}

# Per-task-type contracts known to the Analyst. This exists only because
# TaskTemplate does not yet expose a required-results-keys hook; once
# `get_required_results_keys()` lands on the templates (see
# src/task_template.py) this table should be deleted and the hook used
# for every task type.
_TASK_REQUIRED_KEYS: dict[str, set[str]] = {
    "prediction": _REQUIRED_KEYS,
}

# Block keys inside results.json["measurement_results"] are named
# "<METHOD_ID>_<label>" (e.g. "P1_ctt", "P6_invariance"). Used by the
# post-Analyst method-battery scope assertion (G3).
_METHOD_BLOCK_RE = re.compile(r"^([A-Za-z]+[0-9]+)")

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
    "KeyError": (
        "The error is a KeyError, likely because the target column name was hardcoded as "
        "'target' or 'y'. The target column in train_y.csv is named after the ACTUAL outcome "
        "variable (e.g. 'X4EVRATNDCLG', 'X3TGPAMAT'). Read it dynamically: "
        "target_col = pd.read_csv('train_y.csv').columns[0]; "
        "train_y = pd.read_csv('train_y.csv')[target_col].values"
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


def _sanitize_nonfinite(
    obj: Any, path: str = "$"
) -> tuple[Any, list[str]]:
    """Replace non-finite reals (NaN/Inf) with None, returning the
    sanitized structure and the JSON-paths of every replacement.

    I3 (AERA_OPEN audit): a NaN follow-wave probe crashed
    ``json.dump(..., allow_nan=False)`` mid-stream, truncating
    results.json and killing the REVISING stage. Review hardening:
    matches ``numbers.Real`` rather than bare ``float`` so numpy
    float32/float64 scalars are caught and normalized to Python floats,
    and non-finite float dict KEYS (which also crash strict dumps) are
    sanitized too.
    """
    import numbers

    if isinstance(obj, bool):
        return obj, []
    if isinstance(obj, numbers.Real) and not isinstance(obj, int):
        v = float(obj)
        if v != v or v in (float("inf"), float("-inf")):
            return None, [path]
        return v, []
    if isinstance(obj, dict):
        out: dict = {}
        paths: list[str] = []
        for k, v in obj.items():
            k2 = k
            if (
                isinstance(k, numbers.Real)
                and not isinstance(k, (bool, int))
                and (float(k) != float(k) or float(k) in (float("inf"), float("-inf")))
            ):
                k2 = "non-finite-key"
                paths.append(f"{path}.<key>")
            out[k2], sub = _sanitize_nonfinite(v, f"{path}.{k2}")
            paths.extend(sub)
        return out, paths
    if isinstance(obj, list):
        out_l: list = []
        paths = []
        for i, v in enumerate(obj):
            clean, sub = _sanitize_nonfinite(v, f"{path}[{i}]")
            out_l.append(clean)
            paths.extend(sub)
        return out_l, paths
    return obj, []


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
    if "keyerror" in s:
        return "KeyError"
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
            shutil.copy2(
                _R_BRIDGE_SRC,
                os.path.join(self.ctx.output_dir, "r_bridge.py"),
            )
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
        # G3: deterministic post-Analyst scope assertion against the
        # locked spec (no-op unless the spec declares a method_battery).
        results = self._check_method_battery_scope(results, spec)

        # Persist to disk. allow_nan=False: NaN/Infinity are NOT valid
        # JSON — Python emits bare `NaN` tokens that json.load accepts
        # but strict parsers reject (a sparse-matrix CTT run produced 47
        # NaN item stats this way). I3 (AERA_OPEN audit): a NaN that
        # slips through must NOT crash the stage — json.dump STREAMS
        # into the open file, so the raise left a truncated, invalid
        # results.json on disk and the whole REVISING stage fell over,
        # discarding an otherwise-correct revision. Sanitize non-finite
        # floats to null (recording each path in warnings), serialize to
        # a string first, and only then touch the file.
        results, nonfinite_paths = _sanitize_nonfinite(results)
        if nonfinite_paths:
            # The sandbox-written file may carry warnings as a non-list
            # (review finding: a string here crashed the append).
            w = results.get("warnings")
            if not isinstance(w, list):
                results["warnings"] = [str(w)] if w else []
            results["warnings"].append(
                "Non-finite values (NaN/Inf) recorded as null at: "
                + ", ".join(nonfinite_paths[:20])
                + (" ..." if len(nonfinite_paths) > 20 else "")
            )
            self.ctx.log.append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "agent": self.agent_name,
                    "message": (
                        f"Sanitized {len(nonfinite_paths)} non-finite "
                        "value(s) to null before serializing results.json "
                        "(I3 guard)"
                    ),
                }
            )
        serialized = json.dumps(results, indent=2, allow_nan=False)
        results_path = os.path.join(self.ctx.output_dir, "results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            f.write(serialized)

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
        # Inject pipeline configuration (MLP toggle, class imbalance settings)
        mlp_enabled = self.config.get("pipeline", {}).get("mlp_enabled", True)
        imbalance_cfg = self.config.get("class_imbalance", {})

        # causal_did: split-less panel contract — the prediction-shaped
        # file list below would be a lie (no train/test CSVs exist), and
        # the first live run showed the LLM GUESSES a filename when the
        # panel is not named explicitly (F-B1-ANALYST-PANEL-FILENAME:
        # it tried analytic_panel.csv; DataEngineer writes
        # panel_analytic.csv).
        if research_spec.get("task_type") == "causal_did":
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
                "## Data File Path (EXACT — do not guess or rename)",
                f"- analytic panel: `{os.path.join(output_dir, 'panel_analytic.csv')}`",
                "",
                "There is NO train/test split in DiD mode — the estimator runs "
                "on the full panel. Load exactly this file.",
                "",
                "## Analysis Helpers (REQUIRED — import and use these, do NOT reimplement)",
                "`analysis_helpers.py` is available in your working directory.",
                "```python",
                "import analysis_helpers",
                "core = analysis_helpers.did_gap_in_gaps(df, outcome_col, group_col, post_col)",
                "probe = analysis_helpers.did_placebo_follow_wave(df, base_col, follow_col, group_col, post_col)",
                "```",
                "",
                f"Write results.json to `{os.path.join(output_dir, 'results.json')}` "
                "conforming to the schema in your system prompt.",
            ]
            if revision_instructions:
                parts += ["", "## Revision Instructions", revision_instructions]
            return "\n".join(parts)

        parts = [
            "## Configuration",
            f"- mlp_enabled: {str(mlp_enabled).lower()}",
            f"- minority_threshold: {imbalance_cfg.get('minority_threshold', 0.20)}",
            f"- smote_random_state: {imbalance_cfg.get('smote_random_state', 42)}",
            f"- smote_k_neighbors: {imbalance_cfg.get('smote_k_neighbors', 5)}",
            f"- fbeta_beta: {imbalance_cfg.get('fbeta_beta', 2)}",
            f"- ablation_enabled: {str(imbalance_cfg.get('ablation_enabled', True)).lower()}",
            "",
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
            "**CRITICAL: The target column in train_y.csv/test_y.csv is named after the "
            f"actual outcome variable (`{research_spec.get('outcome_variable', 'UNKNOWN')}`), "
            "NOT 'target'. Load it as:**",
            "```python",
            "train_y_df = pd.read_csv(train_y_path)",
            "target_col = train_y_df.columns[0]  # actual variable name",
            "train_y_arr = train_y_df[target_col].values",
            "```",
            "",
            "## Analysis Helpers (REQUIRED — import and use these, do NOT reimplement)",
            "`analysis_helpers.py` is available in your working directory "
            "(the orchestrator copies it in before your code runs).",
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
                "Generate Python analysis code that trains the model battery "
                "(see Configuration section above for mlp_enabled and class_imbalance settings), "
                "tunes hyperparameters via 5-fold inner CV on the training set only, "
                "builds StackingEnsemble from the tuned base models, evaluates all models "
                "on the held-out test set, applies SMOTE if class imbalance is detected "
                "(with ablation comparison if enabled), computes SHAP interpretability "
                "outputs for the best individual model (StackingEnsemble excluded from SHAP), "
                "generates all required figures and CSVs, and writes results.json to the "
                "Output Directory above. "
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
            "Apply all the same analysis requirements from the original prompt: "
            "model battery per Configuration (check mlp_enabled); "
            "inner CV tuning; test-set evaluation only; SMOTE if imbalanced; "
            "SHAP for best individual model; all figures and CSVs; results.json.",
        ]
        return "\n".join(parts)

    def _read_partial_results(self) -> dict | None:
        """Read results.json if it was partially written before a crash."""
        results_path = os.path.join(self.ctx.output_dir, "results.json")
        if not os.path.exists(results_path):
            return None
        try:
            with open(results_path, encoding="utf-8") as f:
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
            with open(results_path, encoding="utf-8") as f:
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

    def _required_results_keys(self) -> set[str]:
        """Return the required top-level results.json keys for this task type.

        Source of truth is the TaskTemplate: if it declares the contract
        (``get_required_results_keys()`` or a ``REQUIRED_RESULTS_KEYS``
        attribute) that declaration wins. Templates that declare nothing
        fall back to the minimum contract (errors/warnings) — never to
        the prediction schema, which is what produced the phantom
        "missing required keys" error on psychometrics and causal runs
        (G4 / F-P5-PSY-SCHEMA-KEYS).

        Prediction keeps its full SPEC §4.3 contract via
        :data:`_TASK_REQUIRED_KEYS` until the templates grow the hook.
        """
        template = getattr(self, "task_template", None)

        declared: Any = None
        getter = getattr(template, "get_required_results_keys", None)
        if callable(getter):
            try:
                declared = getter()
            except Exception:  # a broken hook must not break a healthy run
                declared = None
        if declared is None:
            declared = getattr(template, "REQUIRED_RESULTS_KEYS", None)
        if declared is not None and not isinstance(declared, (str, bytes)):
            try:
                return {str(k) for k in declared}
            except TypeError:
                pass

        task_name = ""
        try:
            if template is not None:
                task_name = str(template.get_name())
        except Exception:
            task_name = ""
        return set(_TASK_REQUIRED_KEYS.get(task_name, _MINIMUM_REQUIRED_KEYS))

    def _validate_results(self, results: dict) -> dict:
        """Enforce schema requirements and flag suspicious AUC values."""
        results.setdefault("errors", [])
        results.setdefault("warnings", [])

        # Check required top-level keys (task-type-specific, never the
        # prediction schema by default — see _required_results_keys)
        required_keys = self._required_results_keys()
        missing_keys = required_keys - results.keys()
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

        # Ensure top_features is a list. Only complain about ABSENCE when
        # the task type actually contracts for it — a measurement or DiD
        # results.json legitimately has no SHAP feature ranking, and the
        # warning was reaching the Critic as a real defect (same phantom
        # class as the missing-keys error above). A present-but-wrong-type
        # value is always a defect, whatever the task type.
        if "top_features" in results or "top_features" in required_keys:
            if not isinstance(results.get("top_features"), list):
                results["warnings"].append(
                    "top_features is missing or not a list; defaulting to []"
                )
                results["top_features"] = []

        # Validate ablation presence for imbalanced classification
        data_report = getattr(self.ctx, "data_report", None) or {}
        is_imbalanced = data_report.get("is_imbalanced", False)
        ablation_enabled = self.config.get("class_imbalance", {}).get("ablation_enabled", True)
        if is_imbalanced and ablation_enabled and results.get("ablation") is None:
            results["warnings"].append(
                "data_report.is_imbalanced is true and ablation_enabled is true, "
                "but results.json contains no 'ablation' key. "
                "The Analyst may not have performed the SMOTE ablation comparison."
            )

        return results

    # ------------------------------------------------------------------
    # Post-Analyst method-battery scope assertion (G3)
    # ------------------------------------------------------------------

    @staticmethod
    def _block_method_id(block_key: str) -> str:
        """Map a measurement_results block key to its method ID.

        ``"P1_ctt" -> "P1"``, ``"P6_invariance" -> "P6"``, ``"P1" -> "P1"``.
        Returns ``""`` for keys with no ``<letters><digits>`` prefix:
        ``measurement_results`` may carry ancillary non-method entries and
        flagging those as scope creep would be exactly the kind of phantom
        complaint this pass is removing elsewhere. A method whose block is
        misnamed still surfaces — as the (more serious) missing-block error.
        """
        match = _METHOD_BLOCK_RE.match(block_key.strip())
        if match:
            return match.group(1).upper()
        return ""

    def _log_scope(self, message: str) -> None:
        """Append a scope-assertion line to the pipeline context log."""
        log = getattr(self.ctx, "log", None)
        if isinstance(log, list):
            log.append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "agent": self.agent_name,
                    "message": message,
                }
            )

    def _check_method_battery_scope(self, results: dict, research_spec: dict) -> dict:
        """Assert the produced blocks match the locked ``method_battery``.

        Deterministic post-Analyst check mirroring the orchestrator's
        post-DE pre-flight in spirit: the locked research_spec is the
        contract, and drift from it is reported explicitly rather than
        discovered by reading a 24-minute run's output (G3 /
        F-P5-BATTERY-SCOPE-CREEP — the Analyst ran the full P1-P7 battery
        against a locked ``["P1","P7"]`` in 1 of 2 observed runs).

        Severity, following the pre-flight's "report, do not crash a
        healthy run" stance:
          * MISSING requested blocks -> ``results["errors"]`` (the locked
            battery was not delivered; this IS an analysis failure).
          * EXTRA unrequested blocks -> ``results["warnings"]`` (wasted
            compute and unusable output the Critic would otherwise reason
            about as if it were requested).

        No-op when the spec declares no ``method_battery`` (prediction and
        causal task types), so this never fires prediction-shaped noise.
        """
        results.setdefault("errors", [])
        results.setdefault("warnings", [])

        battery = research_spec.get("method_battery") if research_spec else None
        if not isinstance(battery, list) or not battery:
            return results

        requested: list[str] = []
        for method in battery:
            method_id = str(method).strip().upper()
            if method_id and method_id not in requested:
                requested.append(method_id)
        if not requested:
            return results

        blocks = results.get("measurement_results")
        if not isinstance(blocks, dict):
            message = (
                f"METHOD BATTERY SCOPE VIOLATION: the locked method_battery is "
                f"{requested}, but results.json has no 'measurement_results' "
                f"dict — none of the requested methods produced a block."
            )
            results["errors"].append(message)
            self._log_scope(message)
            return results

        produced: dict[str, list[str]] = {}
        for block_key in blocks:
            method_id = self._block_method_id(str(block_key))
            if not method_id:
                continue
            produced.setdefault(method_id, []).append(str(block_key))

        missing = [m for m in requested if m not in produced]
        extra_ids = [m for m in sorted(produced) if m not in requested]
        extra_keys = sorted(k for m in extra_ids for k in produced[m])

        if missing:
            message = (
                f"METHOD BATTERY SCOPE VIOLATION: the locked method_battery is "
                f"{requested}, but results.json.measurement_results has no block "
                f"for {missing}. Present blocks: {sorted(blocks)}. Every locked "
                f"method must produce its block — a silently dropped method is "
                f"an analysis failure, not a stylistic choice."
            )
            results["errors"].append(message)
            self._log_scope(message)

        if extra_keys:
            message = (
                f"METHOD BATTERY SCOPE CREEP: results.json.measurement_results "
                f"contains unrequested blocks {extra_keys} (method IDs "
                f"{extra_ids}); the locked method_battery is {requested}. These "
                f"methods were never requested — they cost compute and must not "
                f"be reported as findings."
            )
            results["warnings"].append(message)
            self._log_scope(message)

        if not missing and not extra_keys:
            self._log_scope(
                f"Method-battery scope check passed: measurement_results blocks "
                f"match the locked method_battery {requested}"
            )

        return results
