"""V4 psychometrics — synthetic certification gates (standing Arc-R rule).

Every psychometric estimator passes a replicated known-truth simulation
gate BEFORE live use. Tests run these at the certified defaults — do not
downscale in tests. All estimation goes through the certified r_helpers
via src/r_bridge (the same scripts live runs will call).
"""
from __future__ import annotations

import numpy as np

from src.r_bridge import run_r_script

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Simulators
# ---------------------------------------------------------------------------

def sim_congeneric(n: int, loadings: list[float], seed: int,
                   missing: float = 0.05) -> dict:
    rng = np.random.default_rng(seed)
    f = rng.normal(0, 1, n)
    items = {}
    for i, lam in enumerate(loadings, 1):
        x = lam * f + rng.normal(0, np.sqrt(1 - lam ** 2), n)
        x[rng.random(n) < missing] = np.nan
        items[f"v{i}"] = [None if np.isnan(v) else float(v) for v in x]
    return items


def sim_grm(n: int, a_params: list[float], seed: int,
            theta_shift: np.ndarray | float = 0.0,
            dif_item: int | None = None, dif_shift: float = 0.0) -> dict:
    """Graded responses (4 categories) from known discriminations; fixed
    threshold sets per item for reproducibility."""
    rng = np.random.default_rng(seed)
    n_items = len(a_params)
    theta = rng.normal(0, 1, n) + theta_shift
    thresholds = [np.sort(rng.normal(0, 0.9, 3)) for _ in range(n_items)]
    items = {}
    for i, a in enumerate(a_params, 1):
        shift = dif_shift if (dif_item is not None and i == dif_item) else 0.0
        cum = np.column_stack(
            [1 / (1 + np.exp(-a * (theta - b + shift)))
             for b in thresholds[i - 1]]
        )
        u = rng.random(n)
        items[f"g{i}"] = [int(c) + 1 for c in (u[:, None] < cum).sum(axis=1)]
    return items


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

def cfa_gate(n: int = 1500, n_reps: int = 5) -> dict:
    """P3: standardized-loading recovery + fit sanity under FIML."""
    loadings = [0.8, 0.7, 0.6, 0.75, 0.65]
    model = "F =~ " + " + ".join(f"v{i}" for i in range(1, 6))
    errs, cfis = [], []
    for rep in range(n_reps):
        items = sim_congeneric(n, loadings, seed=100 + rep)
        out = run_r_script("cfa_fit.R", {"items": items, "model": model})
        est = {r["item"]: r["est_std"] for r in out["loadings"]}
        errs += [abs(est[f"v{i}"] - lam)
                 for i, lam in enumerate(loadings, 1)]
        cfis.append(out["fit"]["cfi"])
    return {
        "mean_abs_loading_error": float(np.mean(errs)),
        "min_cfi_true_model": float(np.min(cfis)),
        "passed": bool(np.mean(errs) < 0.05 and np.min(cfis) > 0.95),
    }


def grm_gate(n: int = 2000, n_reps: int = 4) -> dict:
    """P4: discrimination recovery + reliability sanity."""
    true_a = [1.8, 1.4, 1.1, 1.6, 1.3]
    errs, rels, conv = [], [], []
    for rep in range(n_reps):
        items = sim_grm(n, true_a, seed=200 + rep)
        out = run_r_script("irt_grm.R", {"items": items,
                                         "itemtype": "graded"})
        conv.append(bool(out["converged"]))
        est_a = [p["a"] for p in out["params"]]
        errs += [abs(e - t) for e, t in zip(est_a, true_a)]
        rels.append(out["marginal_reliability"])
    return {
        "mean_abs_a_error": float(np.mean(errs)),
        "mean_marginal_reliability": float(np.mean(rels)),
        "all_converged": all(conv),
        "passed": bool(all(conv) and np.mean(errs) < 0.25
                       and 0.5 < np.mean(rels) < 0.95),
    }


def dif_gate(n: int = 2000, n_reps: int = 5) -> dict:
    """P5: known-DIF hit rate + null false-positive honesty."""
    true_a = [1.5, 1.5, 1.5, 1.5, 1.5]
    hits, false_pos, n_null_items = 0, 0, 0
    for rep in range(n_reps):
        rng = np.random.default_rng(300 + rep)
        grp = (rng.random(n) < 0.5).astype(int)
        # group independent of theta; group-dependent shift on item 3 only
        items = _sim_grm_group_dif(n, true_a, seed=300 + rep,
                                   grp=grp, dif_item=3, dif_shift=0.9)
        out = run_r_script("dif_ordinal.R", {
            "items": items,
            "group": ["m" if g else "f" for g in grp],
        })
        for it in out["items"]:
            if it["item"] == "g3":
                hits += int(bool(it["flagged"]))
            else:
                n_null_items += 1
                false_pos += int(bool(it["flagged"]))
    return {
        "hit_rate": hits / n_reps,
        "false_positive_rate": false_pos / max(1, n_null_items),
        "passed": bool(hits / n_reps >= 0.8
                       and false_pos / max(1, n_null_items) <= 0.05),
    }


def _sim_grm_group_dif(n, a_params, seed, grp, dif_item, dif_shift):
    rng = np.random.default_rng(seed)
    theta = rng.normal(0, 1, n)
    thresholds = [np.sort(rng.normal(0, 0.9, 3)) for _ in a_params]
    items = {}
    for i, a in enumerate(a_params, 1):
        shift = dif_shift * grp if i == dif_item else 0.0
        cum = np.column_stack(
            [1 / (1 + np.exp(-a * (theta - b + shift)))
             for b in thresholds[i - 1]]
        )
        u = rng.random(n)
        items[f"g{i}"] = [int(c) + 1 for c in (u[:, None] < cum).sum(axis=1)]
    return items


def invariance_gate(n: int = 2000, n_reps: int = 3,
                    null_reps: int = 3) -> dict:
    """P6: violation detection + full-invariance null honesty."""
    lam = [0.8, 0.7, 0.65, 0.75]
    model = "F =~ " + " + ".join(f"v{i}" for i in range(1, 5))

    def _run(seed, break_level):
        rng = np.random.default_rng(seed)
        f = rng.normal(0, 1, n)
        grp = (rng.random(n) < 0.5).astype(int)
        items = {}
        for i, l in enumerate(lam, 1):
            l_g = l - (0.35 if (break_level == "metric" and i == 1) else 0.0) * grp
            x = l_g * f + rng.normal(0, np.sqrt(1 - l ** 2), n)
            if break_level == "scalar" and i == 2:
                x = x + 0.5 * grp  # intercept shift, loadings equal
            items[f"v{i}"] = [float(v) for v in x]
        return run_r_script("invariance_ladder.R", {
            "items": items, "group": ["A" if g else "B" for g in grp],
            "model": model,
        })

    null_ok = sum(
        _run(400 + r, None)["highest_level_held"] == "scalar"
        for r in range(null_reps)
    )
    metric_detect = sum(
        _run(500 + r, "metric")["highest_level_held"] == "configural"
        for r in range(n_reps)
    )
    scalar_detect = sum(
        _run(600 + r, "scalar")["highest_level_held"] == "metric"
        for r in range(n_reps)
    )
    return {
        "null_reaches_scalar_rate": null_ok / null_reps,
        "metric_break_detected_rate": metric_detect / n_reps,
        "scalar_break_detected_rate": scalar_detect / n_reps,
        "passed": bool(null_ok / null_reps >= 0.66
                       and metric_detect / n_reps >= 0.66
                       and scalar_detect / n_reps >= 0.66),
    }




def sim_dina(n, K, prevalences, guess, slip, items_per_attr, seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    alpha = (rng.random((n, K)) < prevalences).astype(int)
    responses, qm = {}, {}
    j = 0
    for a in range(K):
        for _ in range(items_per_attr):
            j += 1
            eta = alpha[:, a]
            p = np.where(eta == 1, 1 - slip, guess)
            responses[f"it{j}"] = [int(x) for x in
                                   (rng.random(n) < p).astype(int)]
            qm[f"it{j}"] = [a + 1]
    return responses, qm, alpha


def cdm_gate(n=1500, n_reps=3, guess=0.15, slip=0.12):
    """P7: DINA guess/slip recovery + attribute-prevalence accuracy."""
    import numpy as np

    prevs = [0.55, 0.45, 0.6]
    g_err, s_err, p_err = [], [], []
    for rep in range(n_reps):
        responses, qm, _ = sim_dina(n, 3, prevs, guess, slip, 4,
                                    seed=700 + rep)
        out = run_r_script("cdm_fit.R", {
            "responses": responses, "q_matrix": qm,
            "attributes": ["A1", "A2", "A3"], "model": "DINA",
        }, timeout_s=600)
        g_err += [abs(r["guess"] - guess) for r in out["item_params"]]
        s_err += [abs(r["slip"] - slip) for r in out["item_params"]]
        prev = list(out["attribute_prevalence"].values())
        p_err += [abs(e - t) for e, t in zip(prev, prevs)]
    # E2b: comparison honesty on simple-structure (single-attribute)
    # Q-matrices - must detect degeneracy and retain DINA.
    responses, qm, _ = sim_dina(n, 3, prevs, guess, slip, 4, seed=750)
    cmp_out = run_r_script("cdm_fit.R", {
        "responses": responses, "q_matrix": qm,
        "attributes": ["A1", "A2", "A3"], "model": "compare",
    }, timeout_s=900)
    comp = cmp_out.get("comparison") or {}
    compare_ok = bool(comp.get("degenerate_single_attribute")
                      and comp.get("selected") == "DINA"
                      and cmp_out["item_params"][0].get("guess") is not None)
    return {
        "mean_abs_guess_error": float(np.mean(g_err)),
        "mean_abs_slip_error": float(np.mean(s_err)),
        "mean_abs_prevalence_error": float(np.mean(p_err)),
        "compare_degeneracy_honest": compare_ok,
        "passed": bool(np.mean(g_err) < 0.05 and np.mean(s_err) < 0.05
                       and np.mean(p_err) < 0.08 and compare_ok),
    }


def run_psychometric_gate() -> dict:
    """Certified entry point for the V4 P-battery (P3-P6)."""
    cfa = cfa_gate()
    grm = grm_gate()
    dif = dif_gate()
    inv = invariance_gate()
    cdm = cdm_gate()
    return {
        "cfa": cfa, "grm": grm, "dif": dif, "invariance": inv, "cdm": cdm,
        "passed": bool(cfa["passed"] and grm["passed"]
                       and dif["passed"] and inv["passed"]
                       and cdm["passed"]),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run_psychometric_gate(), indent=1))
