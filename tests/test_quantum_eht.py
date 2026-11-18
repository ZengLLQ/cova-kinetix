"""Tests for 3D STO Extended Huckel quantum chemistry, invariances, and Fukui reactivity."""

from __future__ import annotations

import math

import numpy as np

from cova_kinetix.fragment_graph import parse_smiles
from cova_kinetix.quantum_eht import (
    _diatomic_sto_overlaps,
    build_eht_matrices,
    compute_eht_activation_barrier,
    expand_explicit_hydrogens_3d,
    run_extended_huckel,
)


def test_sto_overlap_self_normalization_and_orthogonality() -> None:
    s_ss, s_sp, s_ps, s_sig, s_pi = _diatomic_sto_overlaps(2, 1.625, 1.625, 2, 1.625, 1.625, 1e-10)
    assert math.isclose(s_ss, 1.0, abs_tol=1e-9)
    assert math.isclose(s_sp, 0.0, abs_tol=1e-9)
    assert math.isclose(s_ps, 0.0, abs_tol=1e-9)
    assert math.isclose(s_sig, 1.0, abs_tol=1e-9)
    assert math.isclose(s_pi, 1.0, abs_tol=1e-9)


def test_eht_electron_and_charge_conservation() -> None:
    mol = parse_smiles("C=CC(=O)Nc1ccccc1", name="acrylanilide", embed_3d=True, seed=42)
    eht = run_extended_huckel(mol)
    assert math.isclose(eht.trace_ps_electrons, float(eht.n_valence_electrons), abs_tol=1e-8)
    assert math.isclose(float(np.sum(eht.mulliken_charges)), float(mol.formal_charge), abs_tol=1e-8)
    assert math.isclose(float(np.sum(eht.lowdin_charges)), float(mol.formal_charge), abs_tol=1e-8)
    assert math.isclose(float(np.sum(eht.fukui_plus)), 1.0, abs_tol=1e-8)
    assert math.isclose(float(np.sum(eht.fukui_minus)), 1.0, abs_tol=1e-8)


def test_eht_so3_rotational_and_translational_invariance() -> None:
    mol = parse_smiles("C=CC(=O)NC", name="N_methyl_acrylamide", embed_3d=True, seed=5)
    symbols, coords, _ = expand_explicit_hydrogens_3d(mol)
    res1 = run_extended_huckel(mol, explicit_symbols=symbols, explicit_coords=coords)

    # Arbitrary rigid 3D rotation + translation
    theta = 0.73
    phi = -0.41
    rz = np.array([[math.cos(theta), -math.sin(theta), 0.0], [math.sin(theta), math.cos(theta), 0.0], [0.0, 0.0, 1.0]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(phi), -math.sin(phi)], [0.0, math.sin(phi), math.cos(phi)]])
    rot = rz @ rx
    coords_rot = coords @ rot.T + np.array([12.5, -4.3, 8.1])
    res2 = run_extended_huckel(mol, explicit_symbols=symbols, explicit_coords=coords_rot)

    assert np.allclose(res1.orbital_energies_ev, res2.orbital_energies_ev, atol=1e-9)
    assert np.allclose(res1.mulliken_charges, res2.mulliken_charges, atol=1e-9)
    assert np.allclose(res1.fukui_plus, res2.fukui_plus, atol=1e-8)


def test_overlap_matrix_symmetry_and_positive_definite() -> None:
    mol = parse_smiles("C=CC(=O)N", name="acrylamide", embed_3d=True, seed=3)
    symbols, coords, _ = expand_explicit_hydrogens_3d(mol)
    s_mat, h_mat, _, _ = build_eht_matrices(symbols, coords)
    assert np.allclose(s_mat, s_mat.T, atol=1e-12)
    assert np.allclose(h_mat, h_mat.T, atol=1e-12)
    eigvals = np.linalg.eigvalsh(s_mat)
    assert float(np.min(eigvals)) > 0.05


def test_chloroacetamide_more_reactive_than_nitrile_and_pka_sensitivity() -> None:
    cl_mol = parse_smiles("ClCC(=O)Nc1ccccc1", embed_3d=True, seed=9)
    nit_mol = parse_smiles("N#Cc1ccccc1", embed_3d=True, seed=9)
    eht_cl = run_extended_huckel(cl_mol)
    eht_nit = run_extended_huckel(nit_mol)

    assert eht_cl.predicted_gsh_t_half_min < eht_nit.predicted_gsh_t_half_min
    b_low_pka = compute_eht_activation_barrier(eht_cl, cl_mol.warhead_class, nucleophile_pka=6.5)
    b_high_pka = compute_eht_activation_barrier(eht_cl, cl_mol.warhead_class, nucleophile_pka=9.5)
    assert b_low_pka < b_high_pka


def test_mayer_bond_order_distinguishes_bonded_from_nonbonded() -> None:
    mol = parse_smiles("C=CC=O", name="acrolein", embed_3d=True, seed=42)
    eht = run_extended_huckel(mol)
    # C=C double bond (0-1) and C=O double bond (2-3) exceed single bond (1-2)
    assert eht.mayer_bond_orders[0, 1] > 1.2
    assert eht.mayer_bond_orders[2, 3] > 1.1
    assert eht.mayer_bond_orders[1, 2] > 0.7
    assert eht.mayer_bond_orders[0, 3] < 0.25


def test_butynamide_selective_vs_chloroacetamide_hyperreactive() -> None:
    butyn = parse_smiles("CC#CC(=O)Nc1ccccc1", embed_3d=True, seed=42)
    cl_ac = parse_smiles("ClCC(=O)Nc1ccccc1", embed_3d=True, seed=42)
    e_but = run_extended_huckel(butyn)
    e_cl = run_extended_huckel(cl_ac)
    assert e_cl.hyper_reactive_flag is True
    assert e_but.hyper_reactive_flag is False
    assert e_but.predicted_gsh_t_half_min > e_cl.predicted_gsh_t_half_min
