"""Tests for SMILES parsing, GF(2) ring perception, 3D embedding, and BRICS cleavage."""

from __future__ import annotations

import numpy as np
import pytest

from cova_kinetix.fragment_graph import (
    REFERENCE_COVALENT_LIBRARY,
    cleave_brics,
    graft_warhead,
    parse_smiles,
    to_canonical_smiles,
)
from cova_kinetix.types import SmilesParseError


def test_parse_all_reference_covalent_library_smiles() -> None:
    for name, smi, _, _, _ in REFERENCE_COVALENT_LIBRARY:
        mol = parse_smiles(smi, name=name, embed_3d=True, seed=42)
        assert len(mol.atoms) >= 8
        assert mol.molecular_weight > 100.0
        assert mol.coords_3d is not None
        assert mol.coords_3d.shape == (len(mol.atoms), 3)
        assert np.all(np.isfinite(mol.coords_3d))


def test_gf2_ring_perception_and_aromaticity() -> None:
    naphthalene = parse_smiles("c1ccc2ccccc2c1", name="naphthalene", embed_3d=False)
    assert len(naphthalene.rings) == 2
    assert all(len(r) == 6 for r in naphthalene.rings)
    assert all(a.aromatic for a in naphthalene.atoms)

    spiro = parse_smiles("C1CC2(C1)CCC2", name="spiro", embed_3d=False)
    assert len(spiro.rings) == 2


def test_warhead_detection_variants() -> None:
    acry = parse_smiles("C=CC(=O)Nc1ccccc1", embed_3d=False)
    assert acry.warhead_class == "acrylamide"
    assert acry.electrophile_atom_idx == 0

    cl_acet = parse_smiles("ClCC(=O)Nc1ccccc1", embed_3d=False)
    assert cl_acet.warhead_class == "chloroacetamide"
    assert cl_acet.electrophile_atom_idx == 1

    vs = parse_smiles("C=CS(=O)(=O)c1ccccc1", embed_3d=False)
    assert vs.warhead_class == "vinyl_sulfone"

    nitrile = parse_smiles("CC(=O)NC(C)C#N", embed_3d=False)
    assert nitrile.warhead_class == "nitrile"


def test_distance_geometry_bond_lengths_realistic() -> None:
    mol = parse_smiles("C=CC(=O)Nc1ccccc1", name="phenylacrylamide", embed_3d=True, seed=7)
    assert mol.coords_3d is not None
    for bond in mol.bonds:
        d = float(np.linalg.norm(mol.coords_3d[bond.atom_a] - mol.coords_3d[bond.atom_b]))
        assert 1.10 <= d <= 1.85


def test_brics_cleavage_and_warhead_grafting_roundtrip() -> None:
    mol = parse_smiles("C=CC(=O)Nc1ccc(-c2ncnc3ccccc23)cc1", embed_3d=False)
    frags = cleave_brics(mol)
    assert len(frags) >= 2
    assert any(f.attachment_labels for f in frags)

    grafted = graft_warhead("[1*]c1ccc(Nc2ncnc3ccccc23)cc1", "acrylamide", seed=11)
    assert grafted.warhead_class == "acrylamide"
    assert grafted.electrophile_atom_idx is not None
    can = to_canonical_smiles(grafted)
    reparsed = parse_smiles(can, embed_3d=False)
    assert len(reparsed.atoms) == len(grafted.atoms)


def test_invalid_smiles_raises_error() -> None:
    with pytest.raises(SmilesParseError):
        parse_smiles("")
    with pytest.raises(SmilesParseError):
        parse_smiles("C1CC")  # unclosed ring
    with pytest.raises(SmilesParseError):
        parse_smiles("C(CC")  # unclosed branch


def test_physicochemical_descriptors_and_additional_warheads() -> None:
    dmac = parse_smiles("CN(C)CC=CC(=O)Nc1ccccc1", embed_3d=False)
    assert dmac.warhead_class == "dimethylaminomethyl_acrylamide"
    assert dmac.pka_base is not None and dmac.pka_base > 7.5

    butyn = parse_smiles("CC#CC(=O)Nc1ccccc1", embed_3d=False)
    assert butyn.warhead_class == "butynamide"

    keto = parse_smiles("CC(=O)C(=O)NCc1ccccc1", embed_3d=False)
    assert keto.warhead_class == "alpha_ketoamide"

    assert 0.0 < dmac.qed <= 1.0
    assert 1.0 <= dmac.sa_score <= 10.0
    assert dmac.tpsa > 20.0


def test_attach_fragment_missing_dummy_raises() -> None:
    from cova_kinetix.fragment_graph import attach_fragment

    with pytest.raises(SmilesParseError):
        attach_fragment("c1ccccc1", "[1*]C")
