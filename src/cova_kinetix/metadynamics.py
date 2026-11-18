"""
Well-Tempered Metadynamics (WT-MetaD) free-energy surface reconstruction and
Infrequent Metadynamics rare-event ligand unbinding kinetics with Tiwary-Parrinello
time reweighting and Kolmogorov-Smirnov Poisson-process certification.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from scipy.stats import expon, kstest

from cova_kinetix.types import (
    GAS_CONSTANT_KCAL,
    FloatArray,
    MetadynamicsResult,
)


def construct_unbinding_potential(
    delta_g_bind_kcal: float = -7.5,
    barrier_extra_kcal: float = 5.5,
) -> Tuple[FloatArray, FloatArray, FloatArray]:
    """
    Build a smooth 2D benchmark/pocket-calibrated potential energy surface U(s1, s2)
    (kcal/mol) over unbinding distance s1 in [0.0, 2.2] nm and gatekeeper
    alignment s2 in [-1.2, +1.2] rad, plus its analytical gradient interpolator.
    """
    s1_grid = np.linspace(0.0, 2.2, 65, dtype=np.float64)
    s2_grid = np.linspace(-1.2, 1.2, 49, dtype=np.float64)
    g1, g2 = np.meshgrid(s1_grid, s2_grid, indexing="ij")
    u_2d = evaluate_unbinding_potential_and_grad(g1, g2, delta_g_bind_kcal, barrier_extra_kcal)[0]
    return s1_grid, s2_grid, u_2d


def evaluate_unbinding_potential_and_grad(
    s1: FloatArray | float,
    s2: FloatArray | float,
    delta_g_bind_kcal: float = -7.5,
    barrier_extra_kcal: float = 5.5,
) -> Tuple[FloatArray, FloatArray, FloatArray]:
    """
    Evaluate the analytical 2D ligand-unbinding potential U(s1, s2) and its
    partial derivatives (dU/ds1, dU/ds2) in kcal/mol per CV unit.

    Features a deep bound basin near (0.28, 0.0), a secondary intermediate state
    near (0.72, 0.42), a transition-state barrier near s1 = 1.05, and a flat
    unbound solvent plateau for s1 >= 1.50.
    """
    s1_arr = np.asarray(s1, dtype=np.float64)
    s2_arr = np.asarray(s2, dtype=np.float64)

    depth_bound = max(2.5, abs(delta_g_bind_kcal))
    barrier_ts = depth_bound + max(1.5, barrier_extra_kcal)

    # Bound basin Gaussian centered at (0.28, 0.0)
    b1 = (s1_arr - 0.28) / 0.16
    b2 = s2_arr / 0.42
    g_bound = np.exp(-0.5 * (b1 ** 2 + b2 ** 2))
    u_bound = -depth_bound * g_bound
    du_bound_ds1 = depth_bound * g_bound * (b1 / 0.16)
    du_bound_ds2 = depth_bound * g_bound * (b2 / 0.42)

    # Intermediate encounter basin near (0.70, 0.40)
    i1 = (s1_arr - 0.70) / 0.16
    i2 = (s2_arr - 0.40) / 0.38
    g_inter = np.exp(-0.5 * (i1 ** 2 + i2 ** 2))
    inter_depth = 0.45 * depth_bound
    u_inter = -inter_depth * g_inter
    du_inter_ds1 = inter_depth * g_inter * (i1 / 0.16)
    du_inter_ds2 = inter_depth * g_inter * (i2 / 0.38)

    # Transition-state ridge near s1 = 1.05
    t1 = (s1_arr - 1.05) / 0.17
    g_ts = np.exp(-0.5 * (t1 ** 2))
    ts_height = max(1.8, barrier_ts - depth_bound * 0.85)
    u_ts = ts_height * g_ts
    du_ts_ds1 = -ts_height * g_ts * (t1 / 0.17)
    du_ts_ds2 = np.zeros_like(s1_arr)

    # Harmonic s2 confinement (gatekeeper channel) + steep soft walls at boundaries
    u_conf = 2.4 * (s2_arr ** 2)
    du_conf_ds2 = 4.8 * s2_arr

    wall_left = 45.0 * np.maximum(0.0, 0.08 - s1_arr) ** 2
    dwall_left = -90.0 * np.maximum(0.0, 0.08 - s1_arr)
    wall_right = 45.0 * np.maximum(0.0, s1_arr - 2.05) ** 2
    dwall_right = 90.0 * np.maximum(0.0, s1_arr - 2.05)
    wall_s2 = 35.0 * (np.maximum(0.0, np.abs(s2_arr) - 1.02) ** 2)
    dwall_s2 = 70.0 * np.maximum(0.0, np.abs(s2_arr) - 1.02) * np.sign(s2_arr)

    u_tot = u_bound + u_inter + u_ts + u_conf + wall_left + wall_right + wall_s2
    du_ds1 = du_bound_ds1 + du_inter_ds1 + du_ts_ds1 + dwall_left + dwall_right
    du_ds2 = du_bound_ds2 + du_inter_ds2 + du_ts_ds2 + du_conf_ds2 + dwall_s2
    return u_tot, du_ds1, du_ds2


def _bilinear_bias_and_grad(
    bias_grid: FloatArray,
    s1_grid: FloatArray,
    s2_grid: FloatArray,
    s1: float,
    s2: float,
) -> Tuple[float, float, float]:
    """Evaluate bias V(s1, s2) and its finite-element gradient on the 2D CV grid."""
    ds1 = float(s1_grid[1] - s1_grid[0])
    ds2 = float(s2_grid[1] - s2_grid[0])
    n1, n2 = bias_grid.shape

    u = float(np.clip((s1 - s1_grid[0]) / ds1, 0.0, n1 - 1.0001))
    v = float(np.clip((s2 - s2_grid[0]) / ds2, 0.0, n2 - 1.0001))
    i = int(math.floor(u))
    j = int(math.floor(v))
    fx = u - i
    fy = v - j

    v00 = float(bias_grid[i, j])
    v10 = float(bias_grid[i + 1, j])
    v01 = float(bias_grid[i, j + 1])
    v11 = float(bias_grid[i + 1, j + 1])

    val = (1.0 - fx) * (1.0 - fy) * v00 + fx * (1.0 - fy) * v10 + (1.0 - fx) * fy * v01 + fx * fy * v11
    dv_ds1 = ((1.0 - fy) * (v10 - v00) + fy * (v11 - v01)) / ds1
    dv_ds2 = ((1.0 - fx) * (v01 - v00) + fx * (v11 - v10)) / ds2
    return val, dv_ds1, dv_ds2


def run_well_tempered_metadynamics(
    delta_g_bind_kcal: float = -7.5,
    barrier_extra_kcal: float = 5.0,
    n_steps: int = 2400,
    n_infrequent_trajectories: int = 10,
    temperature_k: float = 310.15,
    bias_factor_gamma: float = 12.0,
    initial_height_kcal: float = 0.35,
    sigma_s1: float = 0.065,
    sigma_s2: float = 0.11,
    deposition_stride: int = 8,
    dt_ps: float = 0.02,
    gamma_friction_inv_ps: float = 4.0,
    seed: int = 42,
) -> MetadynamicsResult:
    """
    Execute a full 2D Well-Tempered Metadynamics (WT-MetaD) simulation to
    reconstruct the ligand-unbinding free-energy surface F(s1, s2), followed by
    an Infrequent Metadynamics ensemble to recover unbiased off-rates k_off and
    validate Poisson exit-time statistics via Kolmogorov-Smirnov testing.
    """
    if bias_factor_gamma <= 1.0:
        raise ValueError("bias_factor_gamma must exceed 1.0.")

    rng = np.random.default_rng(seed)
    kt_kcal = GAS_CONSTANT_KCAL * temperature_k
    beta = 1.0 / kt_kcal
    delta_t_kcal = kt_kcal * (bias_factor_gamma - 1.0)

    s1_grid, s2_grid, _ = construct_unbinding_potential(delta_g_bind_kcal, barrier_extra_kcal)
    g1, g2 = np.meshgrid(s1_grid, s2_grid, indexing="ij")
    bias_grid = np.zeros_like(g1)

    # Overdamped Brownian integrator step scale: dx = -(D / kT) * dU * dt + sqrt(2 D dt) * N(0,1)
    # Use effective CV diffusion constant D = 1 / gamma_friction
    diff_cv = 1.0 / max(gamma_friction_inv_ps, 0.1)
    noise_amp = math.sqrt(2.0 * diff_cv * dt_ps)

    s1 = 0.28
    s2 = 0.0
    traj_s1 = np.zeros(n_steps, dtype=np.float64)
    traj_s2 = np.zeros(n_steps, dtype=np.float64)

    for step in range(n_steps):
        _, du1, du2 = evaluate_unbinding_potential_and_grad(s1, s2, delta_g_bind_kcal, barrier_extra_kcal)
        v_bias, dv1, dv2 = _bilinear_bias_and_grad(bias_grid, s1_grid, s2_grid, s1, s2)

        f1 = -(float(du1) + dv1)
        f2 = -(float(du2) + dv2)
        s1 += (diff_cv * beta * f1 * dt_ps) + noise_amp * float(rng.normal())
        s2 += (diff_cv * beta * f2 * dt_ps) + noise_amp * float(rng.normal())
        s1 = float(np.clip(s1, 0.02, 2.18))
        s2 = float(np.clip(s2, -1.18, 1.18))

        traj_s1[step] = s1
        traj_s2[step] = s2

        if step % deposition_stride == 0:
            w_t = initial_height_kcal * math.exp(-v_bias / max(delta_t_kcal, 1e-6))
            gauss = w_t * np.exp(-0.5 * (((g1 - s1) / sigma_s1) ** 2 + ((g2 - s2) / sigma_s2) ** 2))
            bias_grid += gauss

    # Well-Tempered FES estimator: F(s) = -(gamma / (gamma - 1)) * V_bias(s)
    fes_2d = -(bias_factor_gamma / (bias_factor_gamma - 1.0)) * bias_grid
    fes_2d -= float(np.min(fes_2d))

    # Marginal 1D FES F(s1) via Boltzmann integration over s2
    boltz_s2 = np.exp(-beta * (fes_2d - np.min(fes_2d)))
    prob_s1 = np.maximum(np.trapezoid(boltz_s2, s2_grid, axis=1), 1e-30)
    fes_1d = -kt_kcal * np.log(prob_s1)
    fes_1d -= float(np.min(fes_1d))

    # Extract bound minimum (s1 < 0.60), transition barrier (0.75 <= s1 <= 1.30), unbound plateau (s1 >= 1.45)
    bound_mask = s1_grid < 0.60
    ts_mask = (s1_grid >= 0.75) & (s1_grid <= 1.32)
    unbound_mask = s1_grid >= 1.45
    f_bound = float(np.min(fes_1d[bound_mask]))
    f_ts = float(np.max(fes_1d[ts_mask]))
    f_unbound = float(np.mean(fes_1d[unbound_mask]))

    barrier_off = max(0.5, f_ts - f_bound)
    barrier_on = max(0.2, f_ts - f_unbound)
    delta_g_est = f_bound - f_unbound

    # Infrequent Metadynamics unbinding ensemble for k_off & Poisson KS validation
    rescaled_times, accel_factors = _run_infrequent_metad_ensemble(
        n_traj=n_infrequent_trajectories,
        delta_g_bind_kcal=delta_g_bind_kcal,
        barrier_extra_kcal=barrier_extra_kcal,
        temperature_k=temperature_k,
        bias_factor_gamma=bias_factor_gamma,
        seed=seed + 101,
    )

    mean_res_s = float(np.mean(rescaled_times))
    k_off_per_s = 1.0 / max(mean_res_s, 1e-12)
    ks_res = kstest(rescaled_times, expon(scale=mean_res_s).cdf)
    ks_stat = float(ks_res.statistic)
    ks_pval = float(ks_res.pvalue)

    return MetadynamicsResult(
        cv1_grid=s1_grid,
        cv2_grid=s2_grid,
        fes_2d_kcal=fes_2d,
        fes_1d_cv1_kcal=fes_1d,
        bias_potential_2d_kcal=bias_grid,
        sample_trajectory_s1=traj_s1,
        sample_trajectory_s2=traj_s2,
        delta_g_bind_kcal=float(delta_g_est),
        barrier_off_kcal=float(barrier_off),
        barrier_on_kcal=float(barrier_on),
        unbiased_k_off_per_s=float(k_off_per_s),
        mean_residence_time_s=mean_res_s,
        rescaled_unbinding_times_s=rescaled_times,
        acceleration_factors=accel_factors,
        ks_statistic=ks_stat,
        ks_pvalue=ks_pval,
        poisson_certified=bool(ks_pval >= 0.05),
    )


def _run_infrequent_metad_ensemble(
    n_traj: int,
    delta_g_bind_kcal: float,
    barrier_extra_kcal: float,
    temperature_k: float,
    bias_factor_gamma: float,
    seed: int,
) -> Tuple[FloatArray, FloatArray]:
    """
    Simulate an ensemble of infrequent metadynamics trajectories starting from
    the bound basin until crossing the unbound dividing surface s1 >= 1.45 nm,
    accumulating Tiwary-Parrinello accelerated physical times.
    """
    rng = np.random.default_rng(seed)
    kt_kcal = GAS_CONSTANT_KCAL * temperature_k
    beta = 1.0 / kt_kcal
    delta_t_kcal = kt_kcal * (bias_factor_gamma - 1.0)

    s1_grid, s2_grid, _ = construct_unbinding_potential(delta_g_bind_kcal, barrier_extra_kcal)
    g1, g2 = np.meshgrid(s1_grid, s2_grid, indexing="ij")

    # Map microscopic CV step to Kramers-calibrated physical time scale
    dt_ps = 0.025
    diff_cv = 0.25
    noise_amp = math.sqrt(2.0 * diff_cv * dt_ps)
    stride = 25
    w0 = 0.55
    sigma1, sigma2 = 0.08, 0.14
    max_steps = 1600

    # Calibrated physical barrier prefactor so typical drug residence spans 1 - 10^4 s
    kramers_scale_s = 1e-6 * math.exp(beta * max(0.0, abs(delta_g_bind_kcal) + barrier_extra_kcal - 12.5))

    rescaled_list = []
    accel_list = []
    for _ in range(max(3, n_traj)):
        bias = np.zeros_like(g1)
        s1, s2 = 0.28 + float(rng.normal(0.0, 0.02)), float(rng.normal(0.0, 0.05))
        t_rescaled_raw = 0.0
        sum_exp = 0.0
        steps_taken = 0
        for step in range(max_steps):
            _, du1, du2 = evaluate_unbinding_potential_and_grad(s1, s2, delta_g_bind_kcal, barrier_extra_kcal)
            v_b, dv1, dv2 = _bilinear_bias_and_grad(bias, s1_grid, s2_grid, s1, s2)
            accel_step = math.exp(min(30.0, beta * v_b))
            t_rescaled_raw += dt_ps * accel_step
            sum_exp += accel_step
            steps_taken += 1

            f1 = -(float(du1) + dv1)
            f2 = -(float(du2) + dv2)
            s1 += (diff_cv * beta * f1 * dt_ps) + noise_amp * float(rng.normal())
            s2 += (diff_cv * beta * f2 * dt_ps) + noise_amp * float(rng.normal())
            s1 = float(np.clip(s1, 0.02, 2.18))
            s2 = float(np.clip(s2, -1.18, 1.18))

            if s1 >= 1.45:
                break
            if step % stride == 0:
                w_t = w0 * math.exp(-v_b / max(delta_t_kcal, 1e-6))
                bias += w_t * np.exp(-0.5 * (((g1 - s1) / sigma1) ** 2 + ((g2 - s2) / sigma2) ** 2))

        alpha_traj = sum_exp / max(steps_taken, 1)
        accel_list.append(alpha_traj)
        rescaled_list.append(max(1e-4, t_rescaled_raw * kramers_scale_s))

    return np.asarray(rescaled_list, dtype=np.float64), np.asarray(accel_list, dtype=np.float64)
