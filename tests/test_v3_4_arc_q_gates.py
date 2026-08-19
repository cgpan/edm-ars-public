"""V3.4-V3.6 Arc Q — replicated synthetic gates for RD / IV / DiD.

Machinery-certification per the synthetic-first standing rule. On HSLS
all three designs are selector-infeasible (Arc D verdicts with
reasons); these gates certify the recipes so dataset onboarding can
flip each design on. Smaller n/reps here for test speed; thresholds
match the script.
"""
from __future__ import annotations

from scripts.quasi_experimental_gates import (
    did_estimate,
    did_pretrend_check,
    iv_2sls,
    make_did_dgp,
    make_iv_dgp,
    make_rd_dgp,
    rd_density_check,
    rd_estimate,
    run_gate,
)


class TestReplicatedGate:
    def test_full_gate_passes(self) -> None:
        # Run at the CERTIFIED configuration — downscaling n/reps
        # changes the statistics the thresholds were set against
        # (found at authoring: n=5000/reps=8 flipped a sub-gate).
        result = run_gate()
        assert result["rd"]["passed"], result["rd"]
        assert result["iv"]["passed"], result["iv"]
        assert result["did"]["passed"], result["did"]
        assert result["gate_passed"]


class TestHonestyChecks:
    def test_weak_instrument_flags(self) -> None:
        z, t, y = make_iv_dgp(n=5000, instrument_strength=0.01)
        assert iv_2sls(z, t, y)["weak_instrument_flag"] is True

    def test_strong_instrument_does_not_flag(self) -> None:
        z, t, y = make_iv_dgp(n=5000, instrument_strength=1.0)
        assert iv_2sls(z, t, y)["weak_instrument_flag"] is False

    def test_pretrend_violation_flags(self) -> None:
        panel = make_did_dgp(parallel=False)
        assert did_pretrend_check(panel)["violation_flag"] is True

    def test_manipulated_density_flags(self) -> None:
        import numpy as np
        rng = np.random.default_rng(0)
        running = rng.uniform(-1, 1, 6000)
        # sort 60% of just-below-cutoff mass to just-above (manipulation)
        near = (running > -0.05) & (running < 0)
        flip = rng.random(near.sum()) < 0.6
        running[np.where(near)[0][flip]] *= -1
        assert rd_density_check(running)["manipulation_flag"] is True


class TestPointEstimators:
    def test_rd_single_draw_reasonable(self) -> None:
        running, _, y = make_rd_dgp(n=8000, effect=0.25)
        est = rd_estimate(running, y)
        assert abs(est["estimate"] - 0.25) < 0.12  # single-draw tolerance

    def test_did_single_draw_reasonable(self) -> None:
        est = did_estimate(make_did_dgp(effect=0.6))
        assert abs(est["estimate"] - 0.6) < 0.08
