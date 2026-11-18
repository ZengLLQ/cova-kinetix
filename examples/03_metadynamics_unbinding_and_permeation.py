"""Example 03: Well-Tempered Metadynamics unbinding & ISD lipid bilayer permeation."""

from __future__ import annotations

from cova_kinetix.fragment_graph import parse_smiles
from cova_kinetix.membrane_permeation import simulate_membrane_permeation
from cova_kinetix.metadynamics import run_well_tempered_metadynamics
from cova_kinetix.visualizer import render_metadynamics_fes, render_permeation_profile


def main() -> None:
    """Simulate rare-event unbinding kinetics and POPC transmembrane permeability."""
    mol = parse_smiles(
        "C=CC(=O)N1CCCC(n2nc(-c3ccc(Oc4ccccc4)cc3)c3c(N)ncnc32)C1",
        name="Ibrutinib_analog",
        embed_3d=False,
    )
    perm = simulate_membrane_permeation(mol, ph_apical=7.4, ph_cytosol=7.2)
    metad = run_well_tempered_metadynamics(
        delta_g_bind_kcal=-8.2,
        barrier_extra_kcal=5.2,
        n_steps=1600,
        n_infrequent_trajectories=8,
        seed=42,
    )
    print(render_permeation_profile(perm))
    print()
    print(render_metadynamics_fes(metad))


if __name__ == "__main__":
    main()
