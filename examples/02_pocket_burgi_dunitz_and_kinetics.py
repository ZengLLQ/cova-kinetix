"""Example 02: 3D Poisson-Boltzmann pocket grid, induced-fit alignment & two-step kinetics."""

from __future__ import annotations

from cova_kinetix.covalent_kinetics import simulate_covalent_kinetics
from cova_kinetix.fragment_graph import parse_smiles
from cova_kinetix.pocket_fields import (
    align_covalent_ligand,
    build_target_pocket,
    solve_poisson_boltzmann_grid,
)
from cova_kinetix.quantum_eht import compute_eht_activation_barrier, run_extended_huckel
from cova_kinetix.visualizer import render_covalent_kinetics, render_pocket_slice_contour


def main() -> None:
    """Align Osimertinib analog in EGFR C797S pocket and simulate 48h target occupancy."""
    smiles = "C=CC(=O)Nc1cc(Nc2nccc(-c3cn(C)c4ccccc34)n2)c(OC)cc1N(C)CCN(C)C"
    mol = parse_smiles(smiles, name="Osimertinib_analog", embed_3d=True, seed=42)
    pocket = build_target_pocket("EGFR_C797S")
    pb_grid = solve_poisson_boltzmann_grid(pocket, grid_n=21)
    eht = run_extended_huckel(mol)
    docking = align_covalent_ligand(mol, pocket, pb_grid=pb_grid, mulliken_charges=eht.mulliken_charges)
    barrier = compute_eht_activation_barrier(
        eht,
        mol.warhead_class,
        nucleophile_pka=pocket.nucleophile_pka,
        physiological_ph=pocket.physiological_ph,
    )
    kinetics = simulate_covalent_kinetics(
        ki_uM=docking.effective_ki_uM,
        activation_barrier_kcal=barrier,
        warhead_class=mol.warhead_class,
        gsh_t_half_min=eht.predicted_gsh_t_half_min,
        total_hours=48.0,
    )
    print(render_pocket_slice_contour(pb_grid, docking))
    print()
    print(render_covalent_kinetics(kinetics))


if __name__ == "__main__":
    main()
