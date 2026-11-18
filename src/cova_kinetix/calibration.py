"""
Biophysical model calibration, Monte Carlo bootstrap confidence intervals, and
Saltelli Sobol global sensitivity analysis over covalent potency and cellular
target occupancy parameters.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
from scipy.optimize import minimize

from cova_kinetix.covalent_kinetics import barrier_to_eyring_rate, simulate_covalent_kinetics
from cova_kinetix.types import CalibrationReport, FloatArray


def _synthetic_kinetic_assay_panel(seed: int = 42) -> Tuple[FloatArray, FloatArray, FloatArray]:
    """
    Generate a deterministic noisy experimental covalent potency dataset
    ``(x_features, y_log_kinact_ki_obs)`` representing biochemical k_inact/K_i
    assays across 18 covalent ligands spanning varying EHT electrophilicity,
    pocket non-covalent affinity, and TPSA desolvation.
    """
    rng = np.random.default_rng(seed)
    n = 18
    # Columns: [noncov_dg_kcal, local_omega_ev, tpsa, base_barrier_kcal]
    dg_vals = rng.uniform(-10.5, -5.0, size=n)
    omega_vals = rng.uniform(1.1, 3.8, size=n)
    tpsa_vals = rng.uniform(45.0, 125.0, size=n)
    base_barriers = rng.uniform(18.5, 21.5, size=n)
    x = np.column_stack([dg_vals, omega_vals, tpsa_vals, base_barriers])

    # Ground-truth physical coupling parameters
    true_params = np.array([0.21, 0.016, 0.92], dtype=np.float64)
    y_clean = _predict_log_kinact_ki_batch(x, true_params)
    y_obs = y_clean + rng.normal(0.0, 0.18, size=n)
    return x, y_obs, true_params


def _predict_log_kinact_ki_batch(x: FloatArray, params: FloatArray) -> FloatArray:
    """
    Predict log10(k_inact / K_i [M^-1 s^-1]) from feature matrix ``x`` and
    parameter vector ``[eht_coupling, desolv_weight, dg_scale]``.
    """
    eht_c, desolv_w, dg_scale = float(params[0]), float(params[1]), float(params[2])
    kt = 0.001987204 * 310.15
    out = np.zeros(x.shape[0], dtype=np.float64)
    for i in range(x.shape[0]):
        dg_nc, local_omega, tpsa, base_bar = x[i]
        dg_eff = dg_scale * dg_nc + desolv_w * tpsa
        ki_m = math.exp(dg_eff / kt)
        barrier = max(14.5, base_bar - eht_c * (local_omega - 2.2))
        k_inact = barrier_to_eyring_rate(barrier)
        out[i] = math.log10(max(k_inact / max(ki_m, 1e-12), 1e-6))
    return out


def _saltelli_sobol_indices(
    n_samples: int = 128,
    seed: int = 42,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Compute Saltelli first-order (S_i) and total-order (S_{Ti}) Sobol variance
    sensitivity indices on 24-hour covalent target occupancy across 5 key biophysical
    inputs: (Ki_uM, barrier_kcal, gsh_t_half_min, k_puu, target_half_life_h).
    """
    rng = np.random.default_rng(seed)
    param_names = [
        "ki_uM",
        "activation_barrier_kcal",
        "gsh_t_half_min",
        "k_puu",
        "target_half_life_h",
    ]
    bounds = [
        (0.05, 2.50),    # Ki (uM)
        (18.0, 22.0),    # Activation barrier (kcal/mol)
        (25.0, 360.0),   # GSH half-life (min)
        (0.15, 1.80),    # Intracellular K_puu
        (6.0, 36.0),     # Target turnover t1/2 (h)
    ]
    d = len(param_names)

    def _draw_unit(rows: int) -> FloatArray:
        return rng.uniform(0.0, 1.0, size=(rows, d))

    def _scale(u_mat: FloatArray) -> FloatArray:
        out = np.empty_like(u_mat)
        for j, (lo, hi) in enumerate(bounds):
            out[:, j] = lo + u_mat[:, j] * (hi - lo)
        return out

    def _eval_batch(phys_mat: FloatArray) -> FloatArray:
        y = np.zeros(phys_mat.shape[0], dtype=np.float64)
        for r in range(phys_mat.shape[0]):
            ki, bar, gsh, kpuu, thalf = phys_mat[r]
            tr = simulate_covalent_kinetics(
                ki_uM=float(ki),
                activation_barrier_kcal=float(bar),
                warhead_class="acrylamide",
                gsh_t_half_min=float(gsh),
                k_puu=float(kpuu),
                total_hours=24.0,
                target_half_life_h=float(thalf),
                n_output_points=49,
            )
            y[r] = tr.occupancy_auc_h / 24.0
        return y

    u_a = _draw_unit(n_samples)
    u_b = _draw_unit(n_samples)
    y_a = _eval_batch(_scale(u_a))
    y_b = _eval_batch(_scale(u_b))

    y_all = np.concatenate([y_a, y_b])
    var_y = max(float(np.var(y_all)), 1e-12)

    s_first: Dict[str, float] = {}
    s_total: Dict[str, float] = {}

    for j, name in enumerate(param_names):
        u_ab_j = u_a.copy()
        u_ab_j[:, j] = u_b[:, j]
        y_ab_j = _eval_batch(_scale(u_ab_j))

        # Jansen (1999) / Saltelli (2010) variance estimators
        si = float(np.mean(y_b * (y_ab_j - y_a))) / var_y
        sti = 0.5 * float(np.mean((y_a - y_ab_j) ** 2)) / var_y
        s_first[name] = float(np.clip(si, 0.0, 1.0))
        s_total[name] = float(np.clip(max(sti, s_first[name]), 0.0, 1.0))

    return s_first, s_total


def calibrate_kinetics_and_sensitivity(
    n_bootstrap: int = 24,
    n_sobol_samples: int = 64,
    seed: int = 42,
) -> CalibrationReport:
    """
    Calibrate EHT-to-barrier coupling, TPSA desolvation weight, and docking
    affinity scaling against the reference kinetic dataset via bounded L-BFGS-B,
    quantify parameter uncertainty via residual bootstrap, and compute Saltelli
    Sobol sensitivity indices.
    """
    x, y_obs, _ = _synthetic_kinetic_assay_panel(seed=seed)
    p0 = np.array([0.10, 0.030, 0.75], dtype=np.float64)
    bounds = [(0.02, 0.50), (0.002, 0.060), (0.50, 1.30)]

    y_init = _predict_log_kinact_ki_batch(x, p0)
    init_rmse = float(np.sqrt(np.mean((y_init - y_obs) ** 2)))

    def _loss(p: FloatArray) -> float:
        pred = _predict_log_kinact_ki_batch(x, p)
        return float(np.mean((pred - y_obs) ** 2))

    opt = minimize(_loss, p0, method="L-BFGS-B", bounds=bounds)
    p_fit = np.asarray(opt.x, dtype=np.float64)
    y_fit = _predict_log_kinact_ki_batch(x, p_fit)
    cal_rmse = float(np.sqrt(np.mean((y_fit - y_obs) ** 2)))
    ss_res = float(np.sum((y_obs - y_fit) ** 2))
    ss_tot = float(np.sum((y_obs - np.mean(y_obs)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

    # Residual bootstrap 95% confidence intervals
    rng = np.random.default_rng(seed + 7)
    residuals = y_obs - y_fit
    boot_samples = np.zeros((max(8, n_bootstrap), 3), dtype=np.float64)
    for b in range(boot_samples.shape[0]):
        resamp_res = rng.choice(residuals, size=len(residuals), replace=True)
        y_boot = y_fit + resamp_res
        res_b = minimize(
            lambda p: float(np.mean((_predict_log_kinact_ki_batch(x, p) - y_boot) ** 2)),
            p_fit,
            method="L-BFGS-B",
            bounds=bounds,
        )
        boot_samples[b] = res_b.x

    param_keys = ("eht_coupling_coeff", "desolvation_weight", "dg_noncov_scale")
    fitted_dict = {k: float(v) for k, v in zip(param_keys, p_fit)}
    ci_95: Dict[str, Tuple[float, float]] = {}
    for j, k in enumerate(param_keys):
        lo = float(np.percentile(boot_samples[:, j], 2.5))
        hi = float(np.percentile(boot_samples[:, j], 97.5))
        ci_95[k] = (lo, hi)

    s_first, s_total = _saltelli_sobol_indices(n_samples=n_sobol_samples, seed=seed + 19)

    return CalibrationReport(
        fitted_params=fitted_dict,
        initial_rmse=init_rmse,
        calibrated_rmse=cal_rmse,
        r2_score=float(r2),
        bootstrap_ci_95=ci_95,
        sobol_first_order=s_first,
        sobol_total_order=s_total,
    )
