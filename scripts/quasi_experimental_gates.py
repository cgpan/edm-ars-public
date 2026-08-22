"""V3.4–V3.6 Arc Q — synthetic-DGP gates for RD, IV, and DiD.

Per the v4 roadmap (internal) Arc Q and the standing synthetic-first rule:
the estimator machinery for each quasi-experimental design must recover
known ground truth (and stay honest on null DGPs) BEFORE any live use.
On HSLS the design selector marks all three infeasible-with-reasons —
these gates certify the machinery so dataset onboarding (Arc G) can
flip each design on by populating ``design_feasibility``.

Implementations are deliberately dependency-light (numpy/pandas/
statsmodels-free — plain OLS via numpy lstsq) so the same recipes can
run inside the sandbox unchanged.

Designs:
- RD: local-linear estimation at the cutoff within a bandwidth,
  triangular kernel; naive-vs-half-bandwidth sensitivity; a density
  (McCrary-style) manipulation check via binned counts.
- IV: 2SLS with first-stage F; weak-instrument honesty (the gate
  includes a weak-IV DGP where the machinery must FLAG, not report a
  confident wrong answer).
- DiD: 2x2 and event-study coefficients on a staggered-adoption DGP
  with unit+time effects; parallel-trends pre-test on a violation DGP
  must FLAG.
"""
from __future__ import annotations

import numpy as np

RANDOM_STATE = 42


def _ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (beta, se) for OLS with homoskedastic SEs."""
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    sigma2 = float(resid @ resid) / dof
    se = np.sqrt(np.diag(sigma2 * XtX_inv))
    return beta, se


# --------------------------------- RD ---------------------------------

def make_rd_dgp(n: int = 8000, effect: float = 0.25, seed: int = RANDOM_STATE):
    rng = np.random.default_rng(seed)
    running = rng.uniform(-1, 1, n)          # centered running variable
    treated = (running >= 0).astype(float)   # sharp cutoff at 0
    y = (
        0.5
        + 0.8 * running
        + 0.3 * running**2
        + effect * treated
        + rng.normal(0, 0.3, n)
    )
    return running, treated, y


def rd_estimate(running: np.ndarray, y: np.ndarray, bandwidth: float = 0.3) -> dict:
    """Local-polynomial (order 2) RD at cutoff 0, triangular kernel."""
    mask = np.abs(running) <= bandwidth
    r, yy = running[mask], y[mask]
    t = (r >= 0).astype(float)
    w = 1 - np.abs(r) / bandwidth
    # Local-QUADRATIC (order-2 polynomial each side): the curvature
    # terms absorb the O(h^2) smoothing bias that plain local-linear
    # leaves in the cutoff contrast (the rdrobust-style correction).
    X = np.column_stack(
        [np.ones_like(r), t, r, t * r, r**2, t * r**2]
    )
    Wsq = np.sqrt(w)
    beta, se = _ols(yy * Wsq, X * Wsq[:, None])
    return {
        "estimate": float(beta[1]),
        "se": float(se[1]),
        "n_in_bandwidth": int(mask.sum()),
        "bandwidth": bandwidth,
    }


def rd_density_check(running: np.ndarray, n_bins: int = 40) -> dict:
    """McCrary-style manipulation check: compare bin counts adjacent to
    the cutoff; a large jump flags sorting."""
    edges = np.linspace(-1, 1, n_bins + 1)
    counts, _ = np.histogram(running, bins=edges)
    left = counts[n_bins // 2 - 1]
    right = counts[n_bins // 2]
    ratio = float(right / max(left, 1))
    return {"left": int(left), "right": int(right), "ratio": ratio,
            "manipulation_flag": bool(ratio > 1.5 or ratio < 1 / 1.5)}


# --------------------------------- IV ---------------------------------

def make_iv_dgp(
    n: int = 8000, effect: float = 0.5, instrument_strength: float = 1.0,
    seed: int = RANDOM_STATE,
):
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1, n)                  # instrument
    u = rng.normal(0, 1, n)                  # unobserved confounder
    t = instrument_strength * z + 0.8 * u + rng.normal(0, 1, n)
    y = effect * t + 1.0 * u + rng.normal(0, 1, n)
    return z, t, y


def iv_2sls(z: np.ndarray, t: np.ndarray, y: np.ndarray) -> dict:
    """2SLS with first-stage F and the weak-IV flag (F < 10)."""
    Z = np.column_stack([np.ones_like(z), z])
    beta_fs, se_fs = _ols(t, Z)
    f_stat = float((beta_fs[1] / se_fs[1]) ** 2)
    t_hat = Z @ beta_fs
    X2 = np.column_stack([np.ones_like(t_hat), t_hat])
    beta, se = _ols(y, X2)
    return {
        "estimate": float(beta[1]),
        "se": float(se[1]),
        "first_stage_F": f_stat,
        "weak_instrument_flag": bool(f_stat < 10.0),
    }


# --------------------------------- DiD --------------------------------

def make_did_dgp(
    n_units: int = 400, n_periods: int = 8, effect: float = 0.6,
    parallel: bool = True, seed: int = RANDOM_STATE,
):
    """Staggered adoption: half the units treat at period 4; unit and
    time fixed effects; optional pre-trend violation."""
    rng = np.random.default_rng(seed)
    unit_fe = rng.normal(0, 1, n_units)
    time_fe = np.linspace(0, 0.5, n_periods)
    treated_unit = (np.arange(n_units) < n_units // 2).astype(float)
    rows = []
    for i in range(n_units):
        for p in range(n_periods):
            post = float(p >= 4)
            trend_violation = (
                0.0 if parallel else 0.15 * p * treated_unit[i]
            )
            y = (
                unit_fe[i]
                + time_fe[p]
                + effect * treated_unit[i] * post
                + trend_violation
                + rng.normal(0, 0.3)
            )
            rows.append((i, p, treated_unit[i], post, y))
    arr = np.array(rows)
    return arr  # columns: unit, period, treated_unit, post, y


def did_estimate(panel: np.ndarray) -> dict:
    """2x2 DiD via the interaction coefficient with unit/time demeaning."""
    unit, period, g, post, y = panel.T
    d = g * post
    # Two-way demeaning (within transformation).
    def demean(v: np.ndarray) -> np.ndarray:
        v = v.copy().astype(float)
        for ids in (unit, period):
            means = {k: v[ids == k].mean() for k in np.unique(ids)}
            v -= np.array([means[k] for k in ids])
        return v
    y_t, d_t = demean(y), demean(d)
    X = np.column_stack([np.ones_like(d_t), d_t])
    beta, se = _ols(y_t, X)
    return {"estimate": float(beta[1]), "se": float(se[1])}


def did_pretrend_check(panel: np.ndarray) -> dict:
    """Pre-period placebo: interaction of treated-group with a linear
    trend using ONLY pre periods; significant slope flags violation."""
    unit, period, g, post, y = panel.T
    pre = post == 0
    p, gg, yy = period[pre], g[pre], y[pre]
    X = np.column_stack([np.ones_like(p), p, gg, p * gg])
    beta, se = _ols(yy, X)
    t_stat = float(beta[3] / se[3])
    return {"pretrend_slope": float(beta[3]), "t": t_stat,
            "violation_flag": bool(abs(t_stat) > 2.5)}


# -------------------------------- gate --------------------------------

def run_gate(n: int = 8000, n_reps: int = 20) -> dict:
    """Replicated gate: single-seed checks conflate estimator bias with
    draw noise (found at authoring: a one-seed RD 'bias' of 0.05 was
    mostly noise). Each design is replicated over ``n_reps`` seeds and
    judged on MEAN estimate (bias) with per-rep flags aggregated."""
    out: dict = {}

    # RD: mean recovery + bandwidth sensitivity + clean density
    rd_estimates, rd_half_shifts, rd_flags = [], [], []
    for rep in range(n_reps):
        running, _, y = make_rd_dgp(n=n, effect=0.25, seed=RANDOM_STATE + rep)
        est = rd_estimate(running, y, bandwidth=0.3)
        est_half = rd_estimate(running, y, bandwidth=0.15)
        rd_estimates.append(est["estimate"])
        rd_half_shifts.append(abs(est_half["estimate"] - est["estimate"]))
        rd_flags.append(rd_density_check(running)["manipulation_flag"])
    rd_mean = float(np.mean(rd_estimates))
    out["rd"] = {
        "mean_estimate": rd_mean,
        "bias": abs(rd_mean - 0.25),
        "mean_half_bw_shift": float(np.mean(rd_half_shifts)),
        "density_false_positive_rate": float(np.mean(rd_flags)),
        "passed": bool(
            abs(rd_mean - 0.25) <= 0.02
            and float(np.mean(rd_flags)) <= 0.10
        ),
    }

    # IV: mean strong-instrument recovery AND weak-instrument honesty
    iv_estimates, weak_flags = [], []
    for rep in range(n_reps):
        z, t, yiv = make_iv_dgp(n=n, effect=0.5, seed=RANDOM_STATE + rep)
        iv_estimates.append(iv_2sls(z, t, yiv)["estimate"])
        zw, tw, yw = make_iv_dgp(
            n=n, effect=0.5, instrument_strength=0.01, seed=RANDOM_STATE + rep
        )
        weak_flags.append(iv_2sls(zw, tw, yw)["weak_instrument_flag"])
    iv_mean = float(np.mean(iv_estimates))
    out["iv"] = {
        "mean_estimate": iv_mean,
        "bias": abs(iv_mean - 0.5),
        "weak_flag_rate": float(np.mean(weak_flags)),
        "passed": bool(abs(iv_mean - 0.5) <= 0.02 and np.mean(weak_flags) >= 0.95),
    }

    # DiD: mean recovery + pre-trend honesty both ways
    did_estimates, clean_flags, bad_flags = [], [], []
    for rep in range(n_reps):
        panel = make_did_dgp(effect=0.6, parallel=True, seed=RANDOM_STATE + rep)
        did_estimates.append(did_estimate(panel)["estimate"])
        clean_flags.append(did_pretrend_check(panel)["violation_flag"])
        panel_bad = make_did_dgp(effect=0.6, parallel=False, seed=RANDOM_STATE + rep)
        bad_flags.append(did_pretrend_check(panel_bad)["violation_flag"])
    did_mean = float(np.mean(did_estimates))
    out["did"] = {
        "mean_estimate": did_mean,
        "bias": abs(did_mean - 0.6),
        "clean_false_positive_rate": float(np.mean(clean_flags)),
        "violation_detection_rate": float(np.mean(bad_flags)),
        "passed": bool(
            abs(did_mean - 0.6) <= 0.02
            and np.mean(clean_flags) <= 0.10
            and np.mean(bad_flags) >= 0.95
        ),
    }

    out["gate_passed"] = all(out[d]["passed"] for d in ("rd", "iv", "did"))
    return out




# ---------------------------------------------------------------------------
# Stream-1 (2026-07-04): synthetic certification for M9 / M10 (did v2)
# ---------------------------------------------------------------------------
# Standing rule (Arc R): every new estimator battery passes a replicated
# synthetic-DGP gate BEFORE live use. Tests run these at the certified
# defaults below - do not downscale in tests.


def make_did_v2_dgp(n=6000, tau=-3.0, beta_h=10.0, comp_shift=0.15,
                    tau_het=0.0, seed=RANDOM_STATE):
    """Two-cohort panel with a compositional shift.

    H is a binary covariate (think parent education) whose prevalence
    shifts BETWEEN COHORTS within the low-SES group by ``comp_shift``.
    Y depends on H (beta_h), so the RAW gap-in-gaps is confounded by
    composition; the composition-fixed gap change is ``tau`` (plus
    ``tau_het * H`` when heterogeneity is on).

    Analytic raw-M8 bias = -comp_shift * beta_h (low-SES H prevalence
    FALLS by comp_shift in cohort 1 here).
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    g = rng.integers(0, 2, n)          # group (low_ses)
    p = rng.integers(0, 2, n)          # cohort (post)
    base_h = np.where(g == 1, 0.45, 0.75)
    h_prev = base_h - comp_shift * ((g == 1) & (p == 1))
    h = (rng.random(n) < h_prev).astype(float)
    noise_cov = rng.normal(0, 1, n)
    y = (
        40.0
        + 5.0 * g
        + 2.0 * p
        + beta_h * h
        + (tau + tau_het * h) * (g * p)
        + rng.normal(0, 8.0, n)
    )
    return pd.DataFrame({
        "low_ses": g, "cohort": p, "H": h.astype(object).astype(str),
        "noise": noise_cov, "rank_base": y,
    })


def did_dr_gate(n=6000, n_reps=12, coverage_reps=6, n_boot=100,
                tau=-3.0, beta_h=10.0, comp_shift=0.15):
    """Certify M9 (within-group cohort AIPW standardization).

    - M9 mean bias small while the RAW 2x2 carries the built-in
      composition bias (sanity that adjustment does real work).
    - CI coverage of tau over ``coverage_reps`` bootstrap runs.
    """
    import numpy as np

    from src.analysis_helpers import did_dr_gap_change, did_gap_in_gaps

    m9_est, m8_est = [], []
    for rep in range(n_reps):
        df = make_did_v2_dgp(n=n, tau=tau, beta_h=beta_h,
                             comp_shift=comp_shift, seed=1000 + rep)
        m9 = did_dr_gap_change(df, "rank_base", "low_ses", "cohort",
                               ["H", "noise"], n_boot=0 or 10,
                               random_state=rep)
        m8 = did_gap_in_gaps(df, "rank_base", "low_ses", "cohort",
                             n_boot=10, random_state=rep)
        m9_est.append(m9["point_estimate"])
        m8_est.append(m8["point_estimate"])

    m9_bias = float(np.mean(m9_est) - tau)
    raw_expected = tau - comp_shift * beta_h
    m8_bias_vs_true = float(np.mean(m8_est) - tau)
    m8_vs_expected = float(np.mean(m8_est) - raw_expected)

    covered = 0
    for rep in range(coverage_reps):
        df = make_did_v2_dgp(n=n, tau=tau, beta_h=beta_h,
                             comp_shift=comp_shift, seed=2000 + rep)
        m9 = did_dr_gap_change(df, "rank_base", "low_ses", "cohort",
                               ["H", "noise"], n_boot=n_boot,
                               random_state=rep)
        if m9["ci_lower"] <= tau <= m9["ci_upper"]:
            covered += 1

    return {
        "m9_bias": m9_bias,
        "m8_bias_vs_true": m8_bias_vs_true,
        "m8_bias_vs_predicted_confounding": m8_vs_expected,
        "coverage": covered / coverage_reps,
        "passed": bool(
            abs(m9_bias) < 0.3
            and abs(m8_bias_vs_true) > 0.8      # raw is visibly confounded
            and abs(m8_vs_expected) < 0.6       # ...by the predicted amount
            and covered / coverage_reps >= 0.8
        ),
    }


def did_het_gate(n=4000, n_reps=5, null_reps=4, n_boot=30):
    """Certify M10 (ML heterogeneity of the gap change).

    Inference is CONTRAST-based (absolute per-level tau carries shared
    boosted-model regularization bias; contrasts cancel it):
    - Recovery: pairwise H1 - H0 difference recovers tau_het = -3.
    - Null honesty: constant tau -> pairwise-difference CI covers 0.
    """
    import numpy as np

    from src.analysis_helpers import did_ml_heterogeneity

    diffs = []
    for rep in range(n_reps):
        df = make_did_v2_dgp(n=n, tau=-1.0, tau_het=-3.0, seed=3000 + rep)
        out = did_ml_heterogeneity(df, "rank_base", "low_ses", "cohort",
                                   ["H", "noise"], subgroup_cols=["H"],
                                   n_boot=25, random_state=rep)
        pw = out["subgroups"]["H"].get("pairwise_difference")
        if pw and pw["levels"] == ["1.0", "0.0"]:
            diffs.append(pw["estimate"])
        elif pw:  # orientation flipped
            diffs.append(-pw["estimate"])

    diff_bias = float(np.mean(diffs) - (-3.0)) if diffs else float("nan")

    null_ok = 0
    for rep in range(null_reps):
        df = make_did_v2_dgp(n=n, tau=-2.0, tau_het=0.0, seed=4000 + rep)
        out = did_ml_heterogeneity(df, "rank_base", "low_ses", "cohort",
                                   ["H", "noise"], subgroup_cols=["H"],
                                   n_boot=n_boot, random_state=rep)
        pw = out["subgroups"]["H"].get("pairwise_difference")
        if pw and pw["ci"][0] <= 0.0 <= pw["ci"][1]:
            null_ok += 1

    return {
        "pairwise_recovery_bias": diff_bias,
        "n_recovery_reps_effective": len(diffs),
        "null_cover_rate": null_ok / null_reps,
        "passed": bool(
            len(diffs) == n_reps
            and abs(diff_bias) < 0.7
            and null_ok / null_reps >= 0.75
        ),
    }


def run_did_v2_gate():
    """Certified entry point for the stream-1 M9/M10 battery."""
    dr = did_dr_gate()
    het = did_het_gate()
    return {"did_dr": dr, "did_het": het,
            "passed": bool(dr["passed"] and het["passed"])}


if __name__ == "__main__":
    import json as _json
    print(_json.dumps(run_did_v2_gate(), indent=1))
    print("--- original Q-era gate ---")
    result = run_gate()
    for k, v in result.items():
        print(k, "->", v)
    raise SystemExit(0 if result["gate_passed"] else 1)
