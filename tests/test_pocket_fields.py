"""Tests for 3D Poisson-Boltzmann grid solver, induced-fit rotamers, and Burgi-Dunitz docking."""

from __future__ import annotations

import numpy as np

from cova_kinetix.fragment_graph import parse_smiles
from cova_kinetix.pocket_fields import (
    align_covalent_ligand,
    build_target_pocket,
    solve_poisson_boltzmann_grid,
)
from cova_kinetix.quantum_eht import run_extended_huckel


def test_poisson_boltzmann_convergence_and_dielectric_bounds() -> None:
    pocket = build_target_pocket("EGFR_C797S")
    grid = solve_poisson_boltzmann_grid(pocket, grid_n=19, max_iter=200, tol=1e-5)
    assert grid.pb_residual < 5e-4
    assert np.all(np.isfinite(grid.electrostatic_potential_kt))
    assert float(np.min(grid.dielectric_grid)) >= 4.0 - 1e-6
    assert float(np.max(grid.dielectric_grid)) <= 78.4 + 1e-6


def test_pb_ionic_strength_screening_reduces_far_field_potential() -> None:
    pocket = build_target_pocket("MPRO_C145")
    g_low = solve_poisson_boltzmann_grid(pocket, grid_n=17, ionic_strength_m=0.01)
    g_high = solve_poisson_boltzmann_grid(pocket, grid_n=17, ionic_strength_m=0.50)
    # Higher ionic strength screens electrostatic potential magnitude in solvent corners
    corner_low = abs(float(g_low.electrostatic_potential_kt[1, 1, 1]))
    corner_high = abs(float(g_high.electrostatic_potential_kt[1, 1, 1]))
    assert corner_high < corner_low


def test_induced_fit_alignment_positions_warhead_near_nucleophile() -> None:
    pocket = build_target_pocket("BTK_C481")
    grid = solve_poisson_boltzmann_grid(pocket, grid_n=19)
    mol = parse_smiles(
        "C=CC(=O)N1CCCC(n2nc(-c3ccc(Oc4ccccc4)cc3)c3c(N)ncnc32)C1",
        name="Ibrutinib_analog",
        embed_3d=True,
        seed=42,
    )
    eht = run_extended_huckel(mol)
    dock = align_covalent_ligand(mol, pocket, pb_grid=grid, mulliken_charges=eht.mulliken_charges)
    assert 1.8 <= dock.nucleophile_distance_a <= 5.5
    assert 30.0 <= dock.burgi_dunitz_angle_deg <= 175.0
    assert 1e-4 <= dock.effective_ki_uM <= 1e4
    assert dock.transformed_coords_3d.shape == (len(mol.atoms), 3)


def test_all_four_target_pockets_and_invalid_target_error() -> None:
    import pytest

    for target in ("EGFR_C797S", "BTK_C481", "KRAS_G12C", "MPRO_C145"):
        pocket = build_target_pocket(target)
        assert pocket.target_name == target
        assert any(r.is_nucleophile for r in pocket.residues)

    with pytest.raises(ValueError):
        build_target_pocket("UNKNOWN_KINASE")


def test_noncovalent_ligand_docks_without_warhead() -> None:
    pocket = build_target_pocket("EGFR_C797S")
    grid = solve_poisson_boltzmann_grid(pocket, grid_n=17)
    decoy = parse_smiles("COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1", embed_3d=True, seed=42)
    dock = align_covalent_ligand(decoy, pocket, pb_grid=grid)
    assert decoy.warhead_class is None
    assert dock.effective_ki_uM > 0.0
