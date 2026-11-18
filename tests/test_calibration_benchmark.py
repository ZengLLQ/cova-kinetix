"""Tests for kinetic parameter calibration, bootstrap CIs, Sobol sensitivity, and benchmark metrics."""

from __future__ import annotations

from cova_kinetix.benchmark import compute_bedroc, compute_roc_auc, run_benchmark_suite
from cova_kinetix.calibration import calibrate_kinetics_and_sensitivity


def test_calibration_reduces_rmse_and_produces_valid_sobol_indices() -> None:
    cal = calibrate_kinetics_and_sensitivity(n_bootstrap=10, n_sobol_samples=24, seed=42)
    assert cal.calibrated_rmse < cal.initial_rmse
    assert cal.r2_score > 0.70
    for k, (lo, hi) in cal.bootstrap_ci_95.items():
        val = cal.fitted_params[k]
        assert lo <= val + 1e-6 and val - 1e-6 <= hi
    assert len(cal.sobol_first_order) == 5
    for k in cal.sobol_first_order:
        assert 0.0 <= cal.sobol_first_order[k] <= cal.sobol_total_order[k] <= 1.0


def test_benchmark_metrics_and_roc_auc_enrichment() -> None:
    assert compute_roc_auc([True, True, False, False], [0.9, 0.8, 0.2, 0.1]) == 1.0
    assert compute_bedroc([True, False, False, False], [0.9, 0.3, 0.2, 0.1], alpha=20.0) > 0.9

    report = run_benchmark_suite(target_name="EGFR_C797S", seed=42)
    assert report.n_compounds == 16
    assert report.roc_auc >= 0.75
    assert report.ef_10_pct >= 1.0


def test_screen_compound_library_top_k_and_btk_target() -> None:
    from cova_kinetix.benchmark import screen_compound_library

    hits = screen_compound_library(target_name="BTK_C481", top_k=5, seed=42)
    assert len(hits) == 5
    scores = [h.composite_score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_bedroc_extremes_bound_zero_to_one() -> None:
    best = compute_bedroc([True, True, False, False], [0.9, 0.8, 0.2, 0.1], alpha=20.0)
    worst = compute_bedroc([False, False, True, True], [0.9, 0.8, 0.2, 0.1], alpha=20.0)
    assert abs(best - 1.0) < 1e-9
    assert abs(worst - 0.0) < 1e-9
