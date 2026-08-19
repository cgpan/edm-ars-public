"""V3.1 Arc R (R2) — synthetic-DGP validation gate for the ITR battery.

Standing discipline from docs/v4_roadmap.md: every new estimator
battery must recover known ground truth on simulated data BEFORE any
live run. ITR errors are silent without an oracle — this gate is the
only place regret is observable.

The implementation here mirrors the M6/M7 skill recipes exactly
(cross-fitted DR pseudo-outcomes; shallow policy tree on rule
covariates via weighted classification; cross-fitted DR policy value;
best-constant baselines) using sklearn only, so a green gate certifies
the *recipes*, not merely this file.

Two DGPs, both grounded in HSLS-like covariate scales:

- ``heterogeneous``: tau(x) = +0.15 when SES < 0 else -0.05 → the
  oracle rule is "treat iff SES < 0". Gate: learned-rule agreement
  with the oracle ≥ 80% AND regret (oracle value − learned value,
  computed on the DGP's true potential outcomes) ≤ 0.02.
- ``null``: tau(x) = 0 → gate: the cross-fitted value GAIN over the
  best constant policy must be small (|gain| ≤ 0.02) — no false
  targeting story.

Run directly (``python scripts/itr_synthetic_gate.py``) for a human
report; pytest drives ``run_gate`` in tests/test_itr_synthetic_gate.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
RULE_COVS = ["X1SES", "X1TXMTSCOR"]
ADJUSTMENT = ["X1SES", "X1TXMTSCOR", "X1SEX", "X1SCHOOLBEL"]


def make_dgp(kind: str, n: int = 6000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Simulate an HSLS-scaled dataset with known potential outcomes."""
    rng = np.random.default_rng(seed)
    ses = rng.normal(0.0, 0.78, n)                    # X1SES-like
    math = rng.normal(51.0, 10.0, n)                  # X1TXMTSCOR-like
    sex = rng.integers(0, 2, n).astype(float)
    belong = rng.normal(0.0, 1.0, n)                  # X1SCHOOLBEL-like

    # Confounded treatment: higher SES + math → more likely treated.
    logits = 0.6 * ses + 0.03 * (math - 51.0)
    e = 1.0 / (1.0 + np.exp(-logits))
    t = (rng.random(n) < e).astype(float)

    base = 0.35 + 0.10 * ses + 0.004 * (math - 51.0) + 0.02 * belong
    if kind == "heterogeneous":
        tau = np.where(ses < 0.0, 0.15, -0.05)
    elif kind == "null":
        tau = np.zeros(n)
    else:
        raise ValueError(f"unknown DGP kind: {kind!r}")

    y0 = base + rng.normal(0.0, 0.15, n)
    y1 = y0 + tau
    y = np.where(t == 1.0, y1, y0)
    return pd.DataFrame(
        {
            "X1SES": ses,
            "X1TXMTSCOR": math,
            "X1SEX": sex,
            "X1SCHOOLBEL": belong,
            "T": t,
            "Y": y,
            "_y0": y0,
            "_y1": y1,
            "_tau": tau,
        }
    )


def _dr_pseudo_outcomes(df: pd.DataFrame, n_folds: int = 5) -> np.ndarray:
    """Cross-fitted DR pseudo-outcomes per the M6 recipe."""
    X = df[ADJUSTMENT].to_numpy()
    t = df["T"].to_numpy()
    y = df["Y"].to_numpy()
    gamma = np.zeros(len(df))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    for train_idx, test_idx in kf.split(X):
        Xtr, Xte = X[train_idx], X[test_idx]
        ttr, ytr = t[train_idx], y[train_idx]
        ps = LogisticRegression(max_iter=1000).fit(Xtr, ttr)
        e_hat = np.clip(ps.predict_proba(Xte)[:, 1], 0.02, 0.98)
        mu1 = GradientBoostingRegressor(random_state=RANDOM_STATE).fit(
            Xtr[ttr == 1], ytr[ttr == 1]
        )
        mu0 = GradientBoostingRegressor(random_state=RANDOM_STATE).fit(
            Xtr[ttr == 0], ytr[ttr == 0]
        )
        m1, m0 = mu1.predict(Xte), mu0.predict(Xte)
        tt, yy = t[test_idx], y[test_idx]
        gamma[test_idx] = (
            m1
            - m0
            + tt * (yy - m1) / e_hat
            - (1 - tt) * (yy - m0) / (1 - e_hat)
        )
    return gamma


def learn_policy_tree(
    df: pd.DataFrame, gamma: np.ndarray
) -> DecisionTreeClassifier:
    """M6: shallow policy tree on RULE covariates via weighted classification."""
    tree = DecisionTreeClassifier(
        max_depth=2, min_samples_leaf=200, random_state=RANDOM_STATE
    )
    tree.fit(
        df[RULE_COVS].to_numpy(),
        (gamma > 0).astype(int),
        sample_weight=np.abs(gamma),
    )
    return tree


def crossfit_policy_gain(df: pd.DataFrame, n_folds: int = 5) -> dict:
    """M7: cross-fitted policy value + gain over the best constant.

    The rule evaluated on fold k is learned on the complement folds.
    Values are DR estimates of the mean outcome under each policy.
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    gamma = _dr_pseudo_outcomes(df, n_folds=n_folds)
    # DR estimate of the untreated-mean component, reused across policies.
    X = df[ADJUSTMENT].to_numpy()
    t = df["T"].to_numpy()
    y = df["Y"].to_numpy()
    mu0_hat = np.zeros(len(df))
    e_hat = np.zeros(len(df))
    for tr, te in kf.split(X):
        m0 = GradientBoostingRegressor(random_state=RANDOM_STATE).fit(
            X[tr][t[tr] == 0], y[tr][t[tr] == 0]
        )
        mu0_hat[te] = m0.predict(X[te])
        ps = LogisticRegression(max_iter=1000).fit(X[tr], t[tr])
        e_hat[te] = np.clip(ps.predict_proba(X[te])[:, 1], 0.02, 0.98)
    v0_component = mu0_hat + (1 - t) * (y - mu0_hat) / (1 - e_hat)

    pi_hat = np.zeros(len(df))
    for tr, te in kf.split(X):
        tree = learn_policy_tree(df.iloc[tr], gamma[tr])
        pi_hat[te] = tree.predict(df.iloc[te][RULE_COVS].to_numpy())

    v_rule = float(np.mean(v0_component + pi_hat * gamma))
    v_all = float(np.mean(v0_component + gamma))
    v_none = float(np.mean(v0_component))
    best_constant = max(v_all, v_none)
    return {
        "policy_value": v_rule,
        "value_treat_all": v_all,
        "value_treat_none": v_none,
        "gain_vs_best_constant": v_rule - best_constant,
        "pi_hat": pi_hat,
    }


def run_gate(n: int = 6000) -> dict:
    """Run both DGP checks; returns verdicts + diagnostics."""
    out: dict = {}

    het = make_dgp("heterogeneous", n=n)
    gamma = _dr_pseudo_outcomes(het)
    tree = learn_policy_tree(het, gamma)
    pi = tree.predict(het[RULE_COVS].to_numpy())
    oracle = (het["_tau"].to_numpy() > 0).astype(int)
    agreement = float(np.mean(pi == oracle))
    # True values from the DGP's potential outcomes (oracle-observable).
    def true_value(policy: np.ndarray) -> float:
        return float(
            np.mean(np.where(policy == 1, het["_y1"], het["_y0"]))
        )
    regret = true_value(oracle) - true_value(pi)
    out["heterogeneous"] = {
        "oracle_agreement": agreement,
        "regret": regret,
        "passed": bool(agreement >= 0.80 and regret <= 0.02),
    }

    null = make_dgp("null", n=n)
    gain = crossfit_policy_gain(null)["gain_vs_best_constant"]
    out["null"] = {
        "gain_vs_best_constant": gain,
        "passed": bool(abs(gain) <= 0.02),
    }
    out["gate_passed"] = out["heterogeneous"]["passed"] and out["null"]["passed"]
    return out


if __name__ == "__main__":
    result = run_gate()
    for k, v in result.items():
        print(k, "->", v)
    raise SystemExit(0 if result["gate_passed"] else 1)
