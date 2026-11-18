"""Example 01: Extended Huckel quantum chemistry & condensed Fukui warhead screening."""

from __future__ import annotations

from cova_kinetix.fragment_graph import parse_smiles
from cova_kinetix.quantum_eht import run_extended_huckel
from cova_kinetix.visualizer import render_eht_summary


def main() -> None:
    """Compare warhead electrophilicity, Fukui f+, and GSH half-life across motifs."""
    panel = [
        ("Acrylamide_anilide", "C=CC(=O)Nc1ccccc1"),
        ("Cyanoacrylamide_anilide", "CC=C(C#N)C(=O)Nc1ccccc1"),
        ("Chloroacetamide_anilide", "ClCC(=O)Nc1ccccc1"),
        ("Vinyl_sulfone_tolyl", "C=CS(=O)(=O)c1ccc(C)cc1"),
        ("Nitrile_peptidomimetic", "CC(=O)NC(C)C#N"),
    ]
    for name, smi in panel:
        mol = parse_smiles(smi, name=name, embed_3d=True, seed=42)
        eht = run_extended_huckel(mol)
        print(render_eht_summary(mol, eht, top_atoms=5))
        print()


if __name__ == "__main__":
    main()
