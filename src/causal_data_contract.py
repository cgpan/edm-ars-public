"""V3.0 Phase 3b.12 / §12.2 — fail-fast guardrail for the causal-mode
analytic-CSV contract.

After the DataEngineer stage produces ``train_X.csv`` and before the
Analyst runs, the Orchestrator asserts that the treatment column
(declared in ``research_spec.treatment.variable``) is present. This
catches the F-3b11-DE-MISSING-TREATMENT-COLUMN failure surface
runtime-deterministically rather than letting the Analyst silently
substitute a proxy.

The matching skill (``causal-data-engineer-contract``) carries the
prescriptive form for the LLM; this module enforces the contract at
the orchestrator boundary.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


class CausalDataContractError(RuntimeError):
    """The DataEngineer's analytic CSV violates the causal_soo contract.

    Raised when ``train_X.csv`` is missing the treatment column declared
    in ``research_spec.treatment.variable`` (or its operationalized form,
    where applicable). The pipeline halts at this boundary; the Analyst
    is not invoked.

    See ``skills/methodology/causal-data-engineer-contract/SKILL.md`` for
    the contract specification and the prescriptive carve-out recipe the
    DataEngineer is expected to follow.
    """


def _expected_treatment_names(research_spec: dict) -> list[str]:
    """Compute the acceptable column names for the treatment.

    The treatment may appear under its original variable name (e.g.
    ``X1MTHEFF``) OR under an operationalized form (e.g.
    ``X1MTHEFF_binary`` for ``median_split_binary``). Both are allowed
    by the contract — the DataEngineer chooses which to emit per its
    operationalization step. The Analyst handles either via the
    ``resolve_encoded_columns`` discipline (D1).
    """
    treatment = research_spec.get("treatment") or {}
    var = treatment.get("variable")
    if not var:
        # No treatment declared — caller has a malformed spec; the
        # assertion can't say anything useful, so allow it through and
        # let upstream validation surface the spec error. (We cannot
        # invent a "missing-treatment-name" contract failure when the
        # spec itself never declared a treatment.)
        return []
    operationalization = treatment.get("operationalization")

    expected = [var]
    # Map of recognized operationalizations → standard column-name suffix.
    # When the project supports a new operationalization, extend this
    # mapping. Unknown operationalizations fall through to the
    # original variable name (which still satisfies the contract if
    # the DataEngineer retains the raw column).
    operationalization_suffixes = {
        "median_split_binary": "_binary",
        "quartile_split": "_quartile",
    }
    suffix = operationalization_suffixes.get(operationalization or "")
    if suffix is not None:
        expected.append(var + suffix)
    return expected


def assert_causal_soo_data_contract(
    train_X_path: Path | str,
    research_spec: dict,
) -> None:
    """Verify ``train_X.csv`` complies with the causal_soo data contract.

    No-op for non-causal task types (the prediction-task DataEngineer
    contract is unchanged and unaffected by this guardrail).

    For ``task_type=causal_soo``, the function loads the CSV header and
    checks that at least one of the acceptable treatment-column names
    is present. Acceptable names come from
    ``research_spec.treatment.variable`` (always allowed) plus an
    operationalization-suffixed form (e.g. ``..._binary`` for
    ``median_split_binary``).

    Args:
        train_X_path: path to the analytic feature CSV produced by the
            DataEngineer stage.
        research_spec: the active research spec (typically
            ``ctx.research_spec`` at the post-ENGINEERING orchestrator
            boundary).

    Raises:
        CausalDataContractError: if ``task_type=causal_soo`` and the
            treatment column is absent. The message names the expected
            column(s), the actual columns, and cites the new skill so
            future debug sessions can trace the violation to the
            documented contract source.
    """
    # Hard gate: only causal_soo is subject to this contract.
    task_type = research_spec.get("task_type")
    if task_type not in ("causal_soo", "causal_itr"):
        return

    expected_names = _expected_treatment_names(research_spec)
    if not expected_names:
        # Spec lacks a declared treatment — bail silently; this is a
        # spec-validation issue, not a DE-output issue.
        return

    path = Path(train_X_path)
    # Read just the header (one row) — the analytic CSV may be huge in
    # pathological encoding-explosion cases (cf. F-3b11-DE-CONTINUOUS-AS-
    # CATEGORICAL with 19,716 columns).
    header_df = pd.read_csv(path, nrows=0)
    actual_columns = list(header_df.columns)

    if not any(name in actual_columns for name in expected_names):
        treatment_var = research_spec["treatment"]["variable"]
        raise CausalDataContractError(
            f"DataEngineer produced {path} missing the treatment column "
            f"'{treatment_var}' declared in research_spec.treatment.variable. "
            f"Expected one of: {expected_names}. Got {len(actual_columns)} "
            f"columns; first 20 = {actual_columns[:20]}. Causal mode "
            f"requires the treatment column to be carved out alongside "
            f"the adjustment_set; see causal-data-engineer-contract skill."
        )


# ---------------------------------------------------------------------------
# V4 Arc H / Phase 3b.23.7 — matrix-level pre-flight (post-DE, pre-Analyst)
# ---------------------------------------------------------------------------
#
# The 3b.12 header check above catches a MISSING treatment column but not
# the failure shapes observed in 3b.17 / 3b.23.5, where the CSV existed
# and had the right header but the ENCODING was wrong: object-dtype label
# passthroughs, continuous covariates one-hot-exploded, or an encoding
# whose propensity overlap is degenerate (extreme_tail_fraction >= 0.10,
# the positivity-violation tier in causal-positivity-diagnostics). These
# matrix-level assertions close that gap deterministically. On violation
# the orchestrator performs ONE targeted DataEngineer retry with the
# violation text injected, then aborts if the retry still violates.

def repair_dummied_treatment(
    output_dir: Path | str,
    research_spec: dict,
) -> str | None:
    """Deterministically repair a one-hot-encoded treatment column.

    3b.23.7 attempt-2 failure shape: the DataEngineer emitted
    ``X1MTHEFF_binary_0`` + ``X1MTHEFF_binary_1`` dummy PAIRS instead of
    the single binary treatment column — and repeated the mistake on a
    targeted retry. The pair encodes identical information, so the
    orchestrator repairs it in place rather than burning a retry:
    ``{name} = {name}_1``; both dummies dropped; train_X and test_X
    rewritten. The repair only fires when the pair is complementary
    (every row has exactly one of the two set), so it can never invent
    data.

    Returns a human-readable description of the repair when one was
    applied (the orchestrator logs it and appends it to
    ``data_report.warnings``), else None.
    """
    if research_spec.get("task_type") not in ("causal_soo", "causal_itr"):
        return None
    out = Path(output_dir)
    repaired: list[str] = []
    for csv_name in ("train_X.csv", "test_X.csv"):
        path = out / csv_name
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for name in _expected_treatment_names(research_spec):
            if name in df.columns:
                continue
            lo, hi = f"{name}_0", f"{name}_1"
            if lo not in df.columns or hi not in df.columns:
                continue
            pair_sum = df[lo].fillna(0) + df[hi].fillna(0)
            if not bool((pair_sum == 1).all()):
                continue  # not a complementary dummy pair — leave it
            df[name] = df[hi].astype(int)
            df = df.drop(columns=[lo, hi])
            df.to_csv(path, index=False)
            repaired.append(f"{csv_name}: {lo}+{hi} -> {name}")
            break
    if repaired:
        return (
            "Deterministic treatment-dummy repair applied ("
            + "; ".join(repaired)
            + ") — the DataEngineer one-hot-encoded the binary treatment; "
            "the complementary dummy pair was collapsed back to a single "
            "0/1 column. See causal-data-engineer-contract skill."
        )
    return None


# Mirror of the causal-positivity-diagnostics skill conventions:
# tails are propensity < 0.05 or > 0.95; fraction >= 0.10 is the
# positivity-violation tier.
_PS_TAIL_LOW = 0.05
_PS_TAIL_HIGH = 0.95
_PS_TAIL_FRACTION_VIOLATION = 0.10

# Guard: skip the PS sanity fit when the matrix is degenerate-small
# (the analytic_n >= 1000 orchestrator gate normally guarantees this).
_PS_MIN_ROWS = 200


def _resolve_treatment_column(
    columns: list[str], research_spec: dict
) -> str | None:
    """Return the treatment column actually present, or None."""
    for name in _expected_treatment_names(research_spec):
        if name in columns:
            return name
    return None


def _registry_continuous_vars(registry: dict) -> set[str]:
    """Collect variable names typed 'continuous' anywhere in the registry."""
    continuous: set[str] = set()
    variables = registry.get("variables", {}) or {}

    def _walk(node: object) -> None:
        if isinstance(node, dict):
            name = node.get("name")
            if name and node.get("type") == "continuous":
                continuous.add(str(name))
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(variables)
    return continuous


def assert_causal_soo_matrix_contract(
    output_dir: Path | str,
    research_spec: dict,
    registry: dict | None = None,
) -> None:
    """Matrix-level D1-contract assertions on the DataEngineer's outputs.

    Runs after :func:`assert_causal_soo_data_contract` (header check) at
    the same orchestrator boundary. No-op for non-causal task types.

    Checks (each raises :class:`CausalDataContractError` with an
    actionable message the orchestrator injects into a targeted retry):

    1. **Treatment is binary with both classes present** — exactly two
       distinct non-null values in the resolved treatment column.
    2. **No object-dtype columns** in ``train_X.csv`` — raw label-string
       passthroughs mean the D1 registry-type dispatch was skipped.
    3. **Continuous covariates stay single numeric columns** — a
       spec-listed adjustment covariate typed ``continuous`` in the
       registry must appear as one numeric column, not a one-hot dummy
       explosion (the F-3b15-DE-CONTINUOUS-AS-CATEGORICAL shape).
    4. **Propensity-overlap sanity** — a bounded logistic fit on the
       train matrix must not put >= 10% of rows in the extreme tails
       (ps < 0.05 or > 0.95), mirroring the positivity-violation tier
       in ``causal-positivity-diagnostics`` (F-3b23.5 shape).
    """
    task_type = research_spec.get("task_type")
    if task_type not in ("causal_soo", "causal_itr"):
        return

    out = Path(output_dir)
    train_X_path = out / "train_X.csv"
    train_X = pd.read_csv(train_X_path)
    columns = list(train_X.columns)

    treatment_col = _resolve_treatment_column(columns, research_spec)
    if treatment_col is None:
        # The header check reports this case with the richer message.
        assert_causal_soo_data_contract(train_X_path, research_spec)
        return

    # --- Check 1: binary treatment with both classes present ----------
    treatment_values = train_X[treatment_col].dropna().unique()
    if len(treatment_values) != 2:
        raise CausalDataContractError(
            f"Treatment column '{treatment_col}' in {train_X_path} has "
            f"{len(treatment_values)} distinct non-null value(s) "
            f"({sorted(map(str, treatment_values[:5]))}); the causal_soo "
            f"contract requires a binary treatment with BOTH classes "
            f"present in the training split. Re-derive the treatment per "
            f"research_spec.treatment.operationalization and verify the "
            f"split preserved both classes."
        )

    # --- Check 2: no object-dtype passthrough columns ------------------
    object_cols = [c for c in columns if train_X[c].dtype == object]
    if object_cols:
        raise CausalDataContractError(
            f"train_X.csv contains {len(object_cols)} object-dtype "
            f"column(s) (first 10: {object_cols[:10]}). Raw label strings "
            f"mean the registry-type-aware encoding dispatch (D1) was "
            f"skipped for these columns. Every categorical covariate must "
            f"be one-hot or numerically encoded; every continuous "
            f"covariate must be numeric. See causal-data-engineer-contract."
        )

    # --- Check 3: continuous covariates stay single numeric columns ----
    if registry:
        continuous_vars = _registry_continuous_vars(registry)
        adjustment = [
            str(v.get("variable") if isinstance(v, dict) else v)
            for v in (research_spec.get("adjustment_set") or [])
        ]
        for var in adjustment:
            if var not in continuous_vars:
                continue
            dummies = [c for c in columns if c.startswith(var + "_")]
            if var not in columns and dummies:
                raise CausalDataContractError(
                    f"Continuous covariate '{var}' (registry type "
                    f"'continuous') is absent as a single numeric column "
                    f"but appears one-hot-expanded into {len(dummies)} "
                    f"dummy column(s) (e.g. {dummies[:5]}). Continuous "
                    f"variables must pass through as single numeric "
                    f"columns (F-3b15-DE-CONTINUOUS-AS-CATEGORICAL). "
                    f"Fix the D1 dispatch for this variable."
                )

    # --- Check 4: propensity-overlap sanity bound -----------------------
    if len(train_X) >= _PS_MIN_ROWS:
        tail_fraction = _ps_extreme_tail_fraction(train_X, treatment_col)
        if tail_fraction >= _PS_TAIL_FRACTION_VIOLATION:
            raise CausalDataContractError(
                f"Propensity-overlap sanity check failed: a bounded "
                f"logistic fit on train_X.csv puts "
                f"{tail_fraction:.1%} of rows in the extreme tails "
                f"(ps < {_PS_TAIL_LOW} or > {_PS_TAIL_HIGH}); the "
                f"positivity-violation tier begins at "
                f"{_PS_TAIL_FRACTION_VIOLATION:.0%} "
                f"(causal-positivity-diagnostics). This encoding will "
                f"produce a positivity violation downstream (F-3b23.5 "
                f"shape). Revisit the covariate encoding: verify every "
                f"adjustment covariate uses its registry type, avoid "
                f"near-deterministic treatment proxies, and standardize "
                f"continuous covariates."
            )


def _ps_extreme_tail_fraction(
    train_X: "pd.DataFrame", treatment_col: str
) -> float:
    """Fraction of rows with propensity outside [0.05, 0.95].

    Bounded, deterministic logistic fit — a sanity probe, not the
    Analyst's estimator. Numeric covariates only; NaNs median-filled
    for the probe (the DE contract requires no NaNs, but the probe must
    not crash if that check is ever relaxed).
    """
    from sklearn.linear_model import LogisticRegression

    y = train_X[treatment_col].astype(float)
    X = train_X.drop(columns=[treatment_col]).select_dtypes("number")
    if X.shape[1] == 0:
        return 0.0
    X = X.fillna(X.median(numeric_only=True))
    # Standardize for a stable bounded fit; guard zero-variance columns.
    std = X.std(numeric_only=True).replace(0, 1.0)
    X = (X - X.mean(numeric_only=True)) / std
    model = LogisticRegression(max_iter=200, C=1.0, random_state=42)
    model.fit(X.values, y.values)
    ps = model.predict_proba(X.values)[:, 1]
    in_tail = (ps < _PS_TAIL_LOW) | (ps > _PS_TAIL_HIGH)
    return float(in_tail.mean())
