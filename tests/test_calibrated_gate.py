"""Arc L wiring — calibrated P25 gate in ReviewGate (user decision 2026-07-03)."""
from __future__ import annotations

from pathlib import Path

import yaml

from src.review_gate import ReviewGate


def _gate(tmp_path: Path, calib: dict | None):
    cfg: dict = {"review_gate": {"pass_threshold": 5.5, "dimension_floor": 3}}
    if calib is not None:
        p = tmp_path / "anchors_edm.yaml"
        p.write_text(yaml.dump(calib), encoding="utf-8")
        cfg["review_gate"]["calibration_path"] = str(p)
    return ReviewGate(cfg, str(tmp_path), log_fn=lambda *_: None)


def _report(overall: float, dims: dict[str, int]) -> dict:
    return {"scores": {"overall_score": overall, "recommendation": "x",
                       "dimensions": [{"name": k, "score": v} for k, v in dims.items()]}}


def test_calibrated_threshold_overrides_absolute(tmp_path: Path) -> None:
    gate = _gate(tmp_path, {"overall_p25_full": 6.3, "n_anchors": 12,
                            "per_dimension_p25_full": {"Novelty": 5.0}})
    assert gate.pass_threshold == 6.3
    assert "calibrated" in gate.calibration_source
    passed, diag = gate.evaluate_gate(_report(6.2, {"Novelty": 6}))
    assert passed is False  # 6.2 fails the calibrated 6.3
    assert diag["threshold_used"] == 6.3
    passed2, _ = gate.evaluate_gate(_report(6.4, {"Novelty": 6}))
    assert passed2 is True


def test_missing_calibration_falls_back(tmp_path: Path) -> None:
    gate = _gate(tmp_path, None)
    assert gate.pass_threshold == 5.5
    assert "absolute" in gate.calibration_source


def test_unreadable_calibration_falls_back(tmp_path: Path) -> None:
    cfg = {"review_gate": {"pass_threshold": 5.5,
                           "calibration_path": str(tmp_path / "missing.yaml")}}
    gate = ReviewGate(cfg, str(tmp_path), log_fn=lambda *_: None)
    assert gate.pass_threshold == 5.5


def test_per_dimension_p25_is_advisory_not_blocking(tmp_path: Path) -> None:
    gate = _gate(tmp_path, {"overall_p25_full": 6.0,
                            "per_dimension_p25_full": {"Novelty": 7.0}})
    passed, diag = gate.evaluate_gate(_report(6.5, {"Novelty": 5}))
    assert passed is True  # advisory: Novelty below P25 does not block
    assert any("Novelty" in x for x in diag["below_calibrated_p25_advisory"])


class TestMedianSampling:
    """Arc L follow-up: borderline-triggered k-sample median (MAD 1.9)."""

    def _gate_with_reports(self, tmp_path, scores: list[float], first: float):
        import yaml as _y
        p = tmp_path / "anchors.yaml"
        p.write_text(_y.dump({"overall_p25_full": 6.3}), encoding="utf-8")
        cfg = {"review_gate": {"pass_threshold": 5.5, "dimension_floor": 3,
                               "calibration_path": str(p),
                               "median_samples": 3, "median_trigger_band": 1.5}}
        gate = ReviewGate(cfg, str(tmp_path), log_fn=lambda *_: None)
        seq = iter(scores)
        gate.run_lsar = lambda pdf, cycle: {"scores": {"overall_score": next(seq),
                                                       "recommendation": "x",
                                                       "dimensions": []}}
        first_report = {"scores": {"overall_score": first, "recommendation": "x",
                                   "dimensions": []}}
        return gate, first_report

    def test_borderline_triggers_median(self, tmp_path) -> None:
        gate, first = self._gate_with_reports(tmp_path, [7.9, 4.9], first=6.0)
        report = gate._maybe_median_sample(first, tmp_path / "p.pdf", cycle=1)
        # samples 6.0, 7.9, 4.9 -> median 6.0
        assert report["scores"]["overall_score"] == 6.0
        assert report["scores"]["median_sampling"]["n_samples"] == 3

    def test_clear_score_skips_sampling(self, tmp_path) -> None:
        gate, first = self._gate_with_reports(tmp_path, [], first=8.5)
        report = gate._maybe_median_sample(first, tmp_path / "p.pdf", cycle=1)
        assert "median_sampling" not in report["scores"]

    def test_disabled_when_k_is_one(self, tmp_path) -> None:
        gate, first = self._gate_with_reports(tmp_path, [], first=6.3)
        gate.config["review_gate"]["median_samples"] = 1
        report = gate._maybe_median_sample(first, tmp_path / "p.pdf", cycle=1)
        assert "median_sampling" not in report["scores"]


class TestVenueAnchoredCalibration:
    def test_anchored_venue_is_calibrated(self, tmp_path) -> None:
        import inspect

        import yaml

        from src.review_gate import ReviewGate

        calib = {"overall_p25_full": 6.3,
                 "venues": {"JEDM": {"p25": 5.15, "n_anchors": 10}}}
        p = tmp_path / "anchors.yaml"
        p.write_text(yaml.dump(calib), encoding="utf-8")
        base = {"enabled": True, "pass_threshold": 5.5,
                "calibration_path": str(p)}

        sig = list(inspect.signature(ReviewGate.__init__).parameters)

        def make(venue):
            cfg = {"review_gate": dict(base, venue=venue)}
            args = [cfg]
            for name in sig[2:]:
                args.append(str(tmp_path) if "dir" in name or "path" in name
                            else (lambda *a, **k: None))
            return ReviewGate(*args)

        jedm = make("JEDM")
        assert jedm.advisory_mode is False
        assert abs(jedm.pass_threshold - 5.15) < 1e-9
        assert "venue-anchored" in jedm.calibration_source

        jla = make("JLA")  # no anchors -> advisory stays
        assert jla.advisory_mode is True
