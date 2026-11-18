"""
Two-step covalent inhibitor binding kinetics, cellular target-turnover PK/PD ODEs,
glutathione off-target clearance, and time-dependent IC50(t) estimation.

Models the coupled mechanism:
    E + I <==[k_on, k_off]==> E*I <==[k_inact, k_rev]==> E-I
with endogenous protein synthesis/degradation (k_syn = k_deg * E0), competing
intracellular cofactor (e.g., ATP/substrate competition 1 + [S]/Km), plasma
one-compartment dosing kinetics, and intracellular unbound accumulation K_{p,uu}.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from cova_kinetix.types import (
    GAS_CONSTANT_KCAL,
    WARHEAD_TEMPLATES,
    FloatArray,
    KineticTrace,
    SimulationDivergenceError,
)


def barrier_to_eyring_rate(
    barrier_kcal: float,
    temperature_k: float = 310.15,
    transmission_kappa: float = 1.0,
) -> float:
    """
    Convert an activation free-energy barrier Delta G_double_dagger (kcal/mol) into a
    unimolecular rate constant (s^-1) via Eyring transition state theory:

    .. math::
        k = \\kappa \\frac{k_B T}{h} \\exp\\!\\left(-\\frac{\\Delta G^{\\ddagger}}{R T}\\right)
    """
    kb_over_h_t = 2.0836619e10 * temperature_k  # (kB / h) * T in s^-1
    exponent = -barrier_kcal / (GAS_CONSTANT_KCAL * temperature_k)
    return float(transmission_kappa * kb_over_h_t * math.exp(max(exponent, -100.0)))


def simulate_covalent_kinetics(
    ki_uM: float,
    activation_barrier_kcal: float,
    warhead_class: Optional[str] = "acrylamide",
    gsh_t_half_min: float = 180.0,
    k_puu: float = 1.0,
    dose_cmax_uM: float = 0.35,
    pk_half_life_h: float = 6.0,
    dosing_interval_h: float = 24.0,
    total_hours: float = 48.0,
    target_conc_uM: float = 0.01,
    target_half_life_h: float = 18.0,
    atp_competition_ratio: float = 2.5,
    k_on_per_uM_s: float = 2.5,
    n_output_points: int = 241,
    include_turnover: bool = True,
    temperature_k: float = 310.15,
) -> KineticTrace:
    """
    Integrate the stiff two-step covalent engagement and cellular target turnover
    differential equations over ``[0, total_hours]``.

    Parameters
    ----------
    ki_uM : float
        Reversible encounter dissociation constant K_i (uM).
    activation_barrier_kcal : float
        Covalent bond-formation activation barrier Delta G_double_dagger (kcal/mol).
    warhead_class : Optional[str]
        Warhead identifier used to determine microscopic reversibility k_rev.
    gsh_t_half_min : float
        Glutathione half-life (minutes) driving intracellular electrophile depletion.
    k_puu : float
        Unbound intracellular-to-plasma accumulation ratio from membrane permeation.
    dose_cmax_uM : float
        Peak unbound plasma concentration per dose (uM).
    pk_half_life_h : float
        Systemic plasma elimination half-life (hours).
    dosing_interval_h : float
        Oral repeat dosing interval (hours, e.g., 24.0 for QD or 12.0 for BID).
    total_hours : float
        Simulation horizon (hours).
    target_conc_uM : float
        Baseline total cellular target concentration E0 (uM).
    target_half_life_h : float
        Endogenous protein degradation half-life (hours).
    atp_competition_ratio : float
        Competitive substrate/ATP ratio ``[S]/Km`` reducing apparent k_on.
    k_on_per_uM_s : float
        Encounter association rate constant (uM^-1 s^-1) before competition.
    n_output_points : int
        Number of evaluation time points.
    include_turnover : bool
        If False, disables protein synthesis/degradation and repeat dosing to
        evaluate closed-system mass conservation.
    temperature_k : float
        Simulation temperature (K).

    Returns
    -------
    KineticTrace
        Time-courses of target states, PK exposure, second-order k_inact/K_i,
        occupancy AUC, and 1h/24h IC50 shifts.
    """
    ki_uM_safe = max(float(ki_uM), 1e-5)
    k_inact_s = barrier_to_eyring_rate(activation_barrier_kcal, temperature_k=temperature_k)
    if warhead_class is None:
        k_inact_s = 0.0
        k_rev_s = 0.0
    else:
        tpl = WARHEAD_TEMPLATES.get(warhead_class)
        k_rev_s = tpl.k_rev_base_s if (tpl is not None and tpl.reversible) else 0.0

    # Convert rates to per-hour units for smooth integration across hours
    sec_per_h = 3600.0
    k_on_eff_uM_h = (k_on_per_uM_s / (1.0 + max(0.0, atp_competition_ratio))) * sec_per_h
    k_off_h = (k_on_per_uM_s * ki_uM_safe) * sec_per_h
    k_inact_h = k_inact_s * sec_per_h
    k_rev_h = k_rev_s * sec_per_h

    k_deg_h = (math.log(2.0) / max(target_half_life_h, 0.1)) if include_turnover else 0.0
    k_syn_uM_h = k_deg_h * target_conc_uM
    k_elim_h = (math.log(2.0) / max(pk_half_life_h, 0.1)) if include_turnover else 0.0
    k_gsh_h = (math.log(2.0) / max(gsh_t_half_min / 60.0, 1e-3)) if include_turnover else 0.0

    def _plasma_drive_uM(t_h: float) -> float:
        if not include_turnover:
            return dose_cmax_uM
        # Superposition of oral 1-compartment boluses with rapid absorption (ka = 1.8 h^-1)
        ka = 1.8
        total_cp = 0.0
        n_doses = int(math.floor(t_h / max(dosing_interval_h, 1.0))) + 1
        for d in range(n_doses):
            dt = t_h - d * dosing_interval_h
            if dt >= 0.0:
                norm_peak = (math.exp(-k_elim_h * dt) - math.exp(-ka * dt))
                # Normalize peak height to dose_cmax_uM
                if abs(ka - k_elim_h) > 1e-6:
                    tmax = math.log(ka / k_elim_h) / (ka - k_elim_h)
                    peak_factor = max(1e-6, math.exp(-k_elim_h * tmax) - math.exp(-ka * tmax))
                else:
                    peak_factor = 0.3679
                total_cp += dose_cmax_uM * max(0.0, norm_peak) / peak_factor
        return total_cp

    # State y = [E_free, EI_noncov, E_I_cov, I_intra, GSH_adduct]
    def _ode_rhs(t_h: float, y: FloatArray) -> List[float]:
        e_free, ei_nc, ei_cov, i_intra, _gsh_add = y
        i_eff = max(0.0, i_intra)

        bind_rate = k_on_eff_uM_h * max(0.0, e_free) * i_eff
        unbind_rate = k_off_h * max(0.0, ei_nc)
        inact_rate = k_inact_h * max(0.0, ei_nc)
        rev_rate = k_rev_h * max(0.0, ei_cov)
        gsh_clear = k_gsh_h * i_eff

        if include_turnover:
            cp_uM = _plasma_drive_uM(t_h)
            k_perm_h = 3.5  # rapid cellular equilibration scaled by K_puu
            flux_in = k_perm_h * (k_puu * cp_uM - i_eff)
        else:
            flux_in = 0.0

        d_efree = k_syn_uM_h - k_deg_h * e_free - bind_rate + unbind_rate
        d_einc = bind_rate - unbind_rate - inact_rate + rev_rate - k_deg_h * ei_nc
        d_eicov = inact_rate - rev_rate - k_deg_h * ei_cov
        d_i = flux_in - bind_rate + unbind_rate - gsh_clear
        d_gsh = gsh_clear
        return [d_efree, d_einc, d_eicov, d_i, d_gsh]

    t_eval = np.linspace(0.0, total_hours, n_output_points, dtype=np.float64)
    y0 = np.array([target_conc_uM, 0.0, 0.0, 0.0 if include_turnover else dose_cmax_uM, 0.0], dtype=np.float64)

    sol = solve_ivp(
        _ode_rhs,
        (0.0, total_hours),
        y0,
        t_eval=t_eval,
        method="Radau",
        rtol=1e-6,
        atol=1e-9,
    )
    if not sol.success:
        raise SimulationDivergenceError(f"Covalent ODE integration failed: {sol.message}")

    e_free_arr = np.maximum(0.0, sol.y[0])
    ei_nc_arr = np.maximum(0.0, sol.y[1])
    ei_cov_arr = np.maximum(0.0, sol.y[2])
    i_intra_arr = np.maximum(0.0, sol.y[3])
    gsh_arr = np.maximum(0.0, sol.y[4])

    total_target = np.maximum(e_free_arr + ei_nc_arr + ei_cov_arr, 1e-12)
    free_frac = e_free_arr / total_target
    nc_frac = ei_nc_arr / total_target
    cov_frac = ei_cov_arr / total_target
    occ_frac = (ei_nc_arr + ei_cov_arr) / total_target

    mass_err = float(np.max(np.abs(total_target - target_conc_uM)) / target_conc_uM)

    plasma_arr = np.array([_plasma_drive_uM(float(t)) for t in t_eval], dtype=np.float64)
    ki_m = ki_uM_safe * 1e-6
    k_inact_over_ki = k_inact_s / ki_m

    occ_auc = float(np.trapezoid(occ_frac, t_eval))
    peak_occ = float(np.max(occ_frac))
    # Trough occupancy over final dosing interval (excluding pre-dose t=0 baseline)
    tail_start = max(0.5, total_hours - min(dosing_interval_h, 0.5 * total_hours))
    tail_mask = t_eval >= tail_start
    trough_occ = float(np.min(occ_frac[tail_mask])) if np.any(tail_mask) else float(occ_frac[-1])

    ic50_1h = estimate_time_dependent_ic50(
        ki_uM_safe, k_inact_s, k_rev_s, incubation_h=1.0, atp_competition_ratio=atp_competition_ratio
    )
    ic50_24h = estimate_time_dependent_ic50(
        ki_uM_safe, k_inact_s, k_rev_s, incubation_h=24.0, atp_competition_ratio=atp_competition_ratio
    )

    return KineticTrace(
        times_h=t_eval,
        free_target_frac=free_frac,
        noncovalent_ei_frac=nc_frac,
        covalent_adduct_frac=cov_frac,
        total_occupancy_frac=occ_frac,
        plasma_conc_uM=plasma_arr,
        intracellular_conc_uM=i_intra_arr,
        gsh_adduct_uM=gsh_arr,
        ki_uM=ki_uM_safe,
        k_inact_per_s=k_inact_s,
        k_rev_per_s=k_rev_s,
        k_inact_over_ki=k_inact_over_ki,
        occupancy_auc_h=occ_auc,
        peak_occupancy=peak_occ,
        trough_occupancy=trough_occ,
        ic50_1h_uM=ic50_1h,
        ic50_24h_uM=ic50_24h,
        mass_balance_error=mass_err,
    )


def estimate_time_dependent_ic50(
    ki_uM: float,
    k_inact_per_s: float,
    k_rev_per_s: float = 0.0,
    incubation_h: float = 1.0,
    atp_competition_ratio: float = 0.0,
) -> float:
    """
    Compute the concentration (uM) achieving 50% fractional target occupancy after
    fixed pre-incubation time ``incubation_h`` under constant inhibitor excess
    (Krippendorff-style rapid-equilibrium two-step model).

    For irreversible inhibitors (k_rev = 0), occupancy satisfies:
    ``1 - exp(-k_inact * t * I / (Ki_app + I)) = 0.5``.
    """
    ki_app = max(ki_uM, 1e-6) * (1.0 + max(0.0, atp_competition_ratio))
    if k_inact_per_s <= 1e-12:
        return ki_app

    t_sec = max(incubation_h, 1e-4) * 3600.0
    if k_rev_per_s <= 1e-12:
        # Exact closed-form irreversible Krippendorff equation
        # exp(-k_inact * t * I / (Ki_app + I)) = 0.5 -> k_obs * t = ln(2)
        target_rate = math.log(2.0) / t_sec
        if target_rate >= k_inact_per_s:
            return ki_app * 100.0
        return float(ki_app * target_rate / (k_inact_per_s - target_rate))

    # Reversible-covalent occupancy via exact two-state exponential relaxation
    def _occ_minus_half(conc_uM: float) -> float:
        k_fwd = k_inact_per_s * conc_uM / (ki_app + conc_uM)
        k_sum = k_fwd + k_rev_per_s
        occ_ss = k_fwd / max(k_sum, 1e-15)
        occ_t = occ_ss * (1.0 - math.exp(-k_sum * t_sec))
        return occ_t - 0.5

    if _occ_minus_half(1e4) <= 0.0:
        return 1e4
    return float(brentq(_occ_minus_half, 1e-7, 1e4, maxiter=80))
