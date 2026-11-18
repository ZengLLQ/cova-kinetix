"""Tests for two-step covalent kinetics, mass conservation, reversibility, and IC50(t)."""

from __future__ import annotations

import math

import numpy as np

from cova_kinetix.covalent_kinetics import (
    barrier_to_eyring_rate,
    estimate_time_dependent_ic50,
    simulate_covalent_kinetics,
)


def test_closed_system_target_mass_conservation() -> None:
    trace = simulate_covalent_kinetics(
        ki_uM=0.25,
        activation_barrier_kcal=20.0,
        warhead_class="acrylamide",
        dose_cmax_uM=1.0,
        total_hours=12.0,
        include_turnover=False,
    )
    assert trace.mass_balance_error < 1e-6
    totals = trace.free_target_frac + trace.noncovalent_ei_frac + trace.covalent_adduct_frac
    assert np.allclose(totals, 1.0, atol=1e-6)
    assert trace.covalent_adduct_frac[-1] > trace.covalent_adduct_frac[0]


def test_irreversible_time_dependent_ic50_shifts_down_with_incubation() -> None:
    k_inact = barrier_to_eyring_rate(20.2)
    ic50_1h = estimate_time_dependent_ic50(ki_uM=0.50, k_inact_per_s=k_inact, k_rev_per_s=0.0, incubation_h=1.0)
    ic50_24h = estimate_time_dependent_ic50(ki_uM=0.50, k_inact_per_s=k_inact, k_rev_per_s=0.0, incubation_h=24.0)
    assert ic50_24h < ic50_1h


def test_reversible_vs_irreversible_covalent_washout_difference() -> None:
    irrev = simulate_covalent_kinetics(
        ki_uM=0.10,
        activation_barrier_kcal=20.0,
        warhead_class="acrylamide",
        dose_cmax_uM=0.40,
        pk_half_life_h=2.0,
        dosing_interval_h=48.0,
        total_hours=24.0,
        include_turnover=True,
    )
    rev = simulate_covalent_kinetics(
        ki_uM=0.10,
        activation_barrier_kcal=20.0,
        warhead_class="nitrile",
        dose_cmax_uM=0.40,
        pk_half_life_h=2.0,
        dosing_interval_h=48.0,
        total_hours=24.0,
        include_turnover=True,
    )
    # Reversible covalent nitrile dissociates faster after PK clearance than irreversible acrylamide
    assert rev.trough_occupancy < irrev.trough_occupancy
    assert math.isclose(irrev.k_rev_per_s, 0.0, abs_tol=1e-12)
    assert rev.k_rev_per_s > 0.0


def test_eyring_barrier_monotonicity_and_atp_competition() -> None:
    assert barrier_to_eyring_rate(19.0) > barrier_to_eyring_rate(21.0)

    low_atp = simulate_covalent_kinetics(
        ki_uM=0.20,
        activation_barrier_kcal=20.0,
        atp_competition_ratio=0.0,
        total_hours=6.0,
        include_turnover=False,
    )
    high_atp = simulate_covalent_kinetics(
        ki_uM=0.20,
        activation_barrier_kcal=20.0,
        atp_competition_ratio=15.0,
        total_hours=6.0,
        include_turnover=False,
    )
    assert low_atp.ic50_1h_uM < high_atp.ic50_1h_uM
