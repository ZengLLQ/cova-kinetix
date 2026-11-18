"""Tests for Inhomogeneous Solubility-Diffusion (ISD) bilayer permeation & pH partitioning."""

from __future__ import annotations

import math

import numpy as np

from cova_kinetix.fragment_graph import parse_smiles
from cova_kinetix.membrane_permeation import compute_neutral_fraction, simulate_membrane_permeation


def test_henderson_hasselbalch_neutral_fraction_limits() -> None:
    f_acid_low = compute_neutral_fraction(pka_acid=4.2, pka_base=None, ph=2.0)
    f_acid_high = compute_neutral_fraction(pka_acid=4.2, pka_base=None, ph=8.0)
    assert f_acid_low > 0.99
    assert f_acid_high < 0.001

    f_base_eq = compute_neutral_fraction(pka_acid=None, pka_base=7.4, ph=7.4)
    assert math.isclose(f_base_eq, 0.5, abs_tol=1e-9)


def test_lipophilic_ligand_more_permeable_than_polar_zwitterion() -> None:
    lipophilic = parse_smiles("C=CC(=O)Nc1ccc(Oc2ccccc2)cc1", name="lipophilic", embed_3d=False)
    polar_acid = parse_smiles("C=CC(=O)Nc1ccc(C(=O)O)cc1O", name="polar_acid", embed_3d=False)

    p_lipo = simulate_membrane_permeation(lipophilic, ph_apical=7.4)
    p_polar = simulate_membrane_permeation(polar_acid, ph_apical=7.4)

    assert p_lipo.p_eff_cm_s > p_polar.p_eff_cm_s
    assert p_lipo.barrier_peak_kcal < p_polar.barrier_peak_kcal
    assert np.all(np.isfinite(p_lipo.delta_g_z_kcal))
    assert np.all(p_lipo.diffusivity_z_cm2_s > 0.0)
    # Bulk water baseline must be symmetric and near zero
    assert abs(float(p_lipo.delta_g_z_kcal[0])) < 0.25
    assert abs(float(p_lipo.delta_g_z_kcal[-1])) < 0.25


def test_efflux_ratio_increases_with_tpsa_and_hbd() -> None:
    low_tpsa = parse_smiles("C=CC(=O)Nc1ccccc1", embed_3d=False)
    high_tpsa = parse_smiles("C=CC(=O)Nc1ccc(C(=O)NC(=O)NCCO)cc1O", embed_3d=False)
    p_low = simulate_membrane_permeation(low_tpsa)
    p_high = simulate_membrane_permeation(high_tpsa)
    assert p_high.efflux_ratio > p_low.efflux_ratio
    assert p_low.p_app_apical_to_basolateral > p_high.p_app_apical_to_basolateral
