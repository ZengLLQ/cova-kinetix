"""Tests for Well-Tempered Metadynamics FES reconstruction and Infrequent MetaD kinetics."""

from __future__ import annotations

import numpy as np

from cova_kinetix.metadynamics import (
    evaluate_unbinding_potential_and_grad,
    run_well_tempered_metadynamics,
)


def test_analytical_potential_gradients_match_central_finite_difference() -> None:
    s1, s2 = 0.55, 0.25
    eps = 1e-5
    _, du1, du2 = evaluate_unbinding_potential_and_grad(s1, s2, -7.5, 5.0)
    u1_p, _, _ = evaluate_unbinding_potential_and_grad(s1 + eps, s2, -7.5, 5.0)
    u1_m, _, _ = evaluate_unbinding_potential_and_grad(s1 - eps, s2, -7.5, 5.0)
    u2_p, _, _ = evaluate_unbinding_potential_and_grad(s1, s2 + eps, -7.5, 5.0)
    u2_m, _, _ = evaluate_unbinding_potential_and_grad(s1, s2 - eps, -7.5, 5.0)

    fd1 = (float(u1_p) - float(u1_m)) / (2.0 * eps)
    fd2 = (float(u2_p) - float(u2_m)) / (2.0 * eps)
    assert np.isclose(float(du1), fd1, atol=1e-6)
    assert np.isclose(float(du2), fd2, atol=1e-6)


def test_wt_metad_reconstructs_bound_minimum_and_unbinding_kinetics() -> None:
    res = run_well_tempered_metadynamics(
        delta_g_bind_kcal=-7.5,
        barrier_extra_kcal=4.8,
        n_steps=2000,
        n_infrequent_trajectories=8,
        seed=42,
    )
    assert res.delta_g_bind_kcal < -1.0
    assert res.barrier_off_kcal > res.barrier_on_kcal
    assert res.unbiased_k_off_per_s > 0.0
    assert res.mean_residence_time_s > 0.0
    assert np.all(res.acceleration_factors >= 1.0)
    assert 0.0 <= res.ks_pvalue <= 1.0


def test_deeper_binding_well_lengthens_residence_time() -> None:
    weak = run_well_tempered_metadynamics(
        delta_g_bind_kcal=-4.5, barrier_extra_kcal=3.5, n_steps=1200, n_infrequent_trajectories=6, seed=11
    )
    strong = run_well_tempered_metadynamics(
        delta_g_bind_kcal=-9.5, barrier_extra_kcal=6.0, n_steps=1200, n_infrequent_trajectories=6, seed=11
    )
    assert strong.mean_residence_time_s > weak.mean_residence_time_s


def test_wt_metad_bias_gamma_validation() -> None:
    import pytest

    with pytest.raises(ValueError):
        run_well_tempered_metadynamics(bias_factor_gamma=1.0)
