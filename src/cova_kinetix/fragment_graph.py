"""
SMILES/fragment molecular graph engine, GF(2) ring perception, physicochemical
descriptors, 3D distance-geometry coordinate embedding, and BRICS retrosynthetic
bond cleavage and warhead grafting.

All graph algorithms are implemented in pure Python + NumPy/SciPy without
requiring external cheminformatics C-extensions.
"""

from __future__ import annotations

import math
import re
from collections import deque
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.optimize import minimize

from cova_kinetix.types import (
    ELEMENT_PARAMS,
    WARHEAD_TEMPLATES,
    AtomRecord,
    BondRecord,
    FloatArray,
    FragmentPiece,
    MolecularGraph,
    SmilesParseError,
)

# Wildman-Crippen simplified atom-level logP contributions by (symbol, aromatic, n_hetero_neighbors)
_BASE_LOGP: Dict[str, float] = {
    "C": 0.144,
    "N": -0.320,
    "O": -0.290,
    "S": 0.648,
    "F": 0.420,
    "Cl": 0.689,
    "Br": 0.846,
    "P": 0.450,
    "B": -0.100,
    "H": 0.123,
    "*": 0.0,
}


def parse_smiles(smiles: str, name: str = "ligand", embed_3d: bool = True, seed: int = 42) -> MolecularGraph:
    """
    Parse a SMILES string into a fully annotated MolecularGraph.

    Supports organic-subset atoms, lowercase aromatic atoms (``c``, ``n``, ``o``,
    ``s``, ``p``, ``b``), bracketed atoms with formal charges/hydrogens/dummy
    labels (``[1*]``, ``[NH3+]``, ``[nH]``), ring closures ``1..9`` and
    ``%10..%99``, nested branches ``(...)``, and explicit bond orders.

    Parameters
    ----------
    smiles : str
        Input SMILES string.
    name : str
        Human-readable identifier for the molecule.
    embed_3d : bool
        Whether to generate 3D coordinates via distance geometry + UFF relaxation.
    seed : int
        Random seed used by 3D distance-geometry tie-breaking.

    Returns
    -------
    MolecularGraph
        Parsed topology, rings, physicochemical properties, warhead annotation,
        and optional 3D coordinates.
    """
    raw = smiles.strip()
    if not raw:
        raise SmilesParseError("Empty SMILES string provided.")

    atoms: List[AtomRecord] = []
    bonds: List[BondRecord] = []
    ring_opens: Dict[int, Tuple[int, Optional[float]]] = {}
    branch_stack: List[int] = []
    prev_atom: Optional[int] = None
    pending_bond: Optional[float] = None

    i = 0
    n = len(raw)
    while i < n:
        ch = raw[i]

        if ch in ("-", "=", "#", ":", "/", "\\"):
            bond_map = {"-": 1.0, "=": 2.0, "#": 3.0, ":": 1.5, "/": 1.0, "\\": 1.0}
            pending_bond = bond_map[ch]
            i += 1
            continue

        if ch == ".":
            prev_atom = None
            pending_bond = None
            i += 1
            continue

        if ch == "(":
            if prev_atom is None:
                raise SmilesParseError(f"Branch opened before any atom in SMILES: {raw}")
            branch_stack.append(prev_atom)
            i += 1
            continue

        if ch == ")":
            if not branch_stack:
                raise SmilesParseError(f"Unmatched ')' in SMILES: {raw}")
            prev_atom = branch_stack.pop()
            pending_bond = None
            i += 1
            continue

        if ch == "%" or ch.isdigit():
            if ch == "%":
                if i + 2 >= n or not raw[i + 1 : i + 3].isdigit():
                    raise SmilesParseError(f"Malformed two-digit ring closure in SMILES: {raw}")
                ring_id = int(raw[i + 1 : i + 3])
                i += 3
            else:
                ring_id = int(ch)
                i += 1
            if prev_atom is None:
                raise SmilesParseError(f"Ring digit '{ring_id}' appeared before an atom in SMILES: {raw}")
            if ring_id in ring_opens:
                other_atom, open_bond = ring_opens.pop(ring_id)
                if other_atom == prev_atom:
                    raise SmilesParseError(f"Self-loop ring closure {ring_id} in SMILES: {raw}")
                order = pending_bond if pending_bond is not None else open_bond
                if order is None:
                    order = 1.5 if (atoms[other_atom].aromatic and atoms[prev_atom].aromatic) else 1.0
                _add_unique_bond(bonds, other_atom, prev_atom, order, atoms)
            else:
                ring_opens[ring_id] = (prev_atom, pending_bond)
            pending_bond = None
            continue

        if ch == "[":
            close_idx = raw.find("]", i + 1)
            if close_idx == -1:
                raise SmilesParseError(f"Unclosed '[' bracket atom in SMILES: {raw}")
            bracket_token = raw[i + 1 : close_idx]
            atom_rec = _parse_bracket_atom(bracket_token, len(atoms))
            i = close_idx + 1
        else:
            atom_rec, step = _parse_organic_atom(raw, i, len(atoms))
            i += step

        atoms.append(atom_rec)
        if prev_atom is not None:
            if pending_bond is not None:
                order = pending_bond
            elif atoms[prev_atom].aromatic and atom_rec.aromatic:
                order = 1.5
            else:
                order = 1.0
            _add_unique_bond(bonds, prev_atom, atom_rec.index, order, atoms)
        prev_atom = atom_rec.index
        pending_bond = None

    if branch_stack:
        raise SmilesParseError(f"Unclosed '(' branch in SMILES: {raw}")
    if ring_opens:
        raise SmilesParseError(f"Unclosed ring closures {sorted(ring_opens.keys())} in SMILES: {raw}")
    if not atoms:
        raise SmilesParseError(f"No atoms parsed from SMILES: {raw}")

    mol = MolecularGraph(name=name, smiles=raw, atoms=atoms, bonds=bonds)
    _annotate_topology_and_properties(mol)
    if embed_3d:
        mol.coords_3d = embed_3d_coordinates(mol, seed=seed)
        for idx, atom in enumerate(mol.atoms):
            atom.coords = (
                float(mol.coords_3d[idx, 0]),
                float(mol.coords_3d[idx, 1]),
                float(mol.coords_3d[idx, 2]),
            )
    return mol


def _add_unique_bond(
    bonds: List[BondRecord],
    a: int,
    b: int,
    order: float,
    atoms: List[AtomRecord],
) -> None:
    lo, hi = (a, b) if a < b else (b, a)
    for existing in bonds:
        elo = min(existing.atom_a, existing.atom_b)
        ehi = max(existing.atom_a, existing.atom_b)
        if elo == lo and ehi == hi:
            return
    is_arom = abs(order - 1.5) < 1e-6 or (atoms[a].aromatic and atoms[b].aromatic and order == 1.5)
    bonds.append(BondRecord(atom_a=a, atom_b=b, order=order, aromatic=is_arom))


def _parse_organic_atom(raw: str, pos: int, atom_idx: int) -> Tuple[AtomRecord, int]:
    ch = raw[pos]
    if pos + 1 < len(raw):
        two = raw[pos : pos + 2]
        if two in ("Cl", "Br"):
            return AtomRecord(index=atom_idx, symbol=two), 2
    if ch in ("B", "C", "N", "O", "P", "S", "F", "I", "*"):
        sym = "Cl" if ch == "I" else ch  # map rare I gracefully to halogen table if encountered
        if ch == "*":
            return AtomRecord(index=atom_idx, symbol="*", dummy_label=0), 1
        return AtomRecord(index=atom_idx, symbol=sym), 1
    if ch in ("b", "c", "n", "o", "p", "s"):
        return AtomRecord(index=atom_idx, symbol=ch.upper(), aromatic=True), 1
    raise SmilesParseError(f"Unsupported SMILES character '{ch}' at position {pos} in '{raw}'")


def _parse_bracket_atom(token: str, atom_idx: int) -> AtomRecord:
    m = re.match(
        r"^(?P<iso>\d+)?(?P<sym>\*|[A-Za-z][a-z]?)(?P<h>H\d*)?(?P<chg>[+-]+\d*)?$",
        token,
    )
    if m is None:
        raise SmilesParseError(f"Malformed bracket atom '[{token}]'")
    iso_str = m.group("iso")
    raw_sym = m.group("sym")
    h_str = m.group("h")
    chg_str = m.group("chg")

    dummy_label: Optional[int] = None
    if raw_sym == "*":
        sym = "*"
        dummy_label = int(iso_str) if iso_str is not None else 0
        aromatic = False
    else:
        aromatic = raw_sym.islower()
        sym = raw_sym.capitalize()
        if sym not in ELEMENT_PARAMS:
            raise SmilesParseError(f"Unsupported element '{sym}' in bracket atom '[{token}]'")

    explicit_h = 0
    if h_str is not None:
        explicit_h = 1 if len(h_str) == 1 else int(h_str[1:])

    formal_charge = 0
    if chg_str is not None:
        sign = 1 if chg_str[0] == "+" else -1
        if len(chg_str) == 1:
            formal_charge = sign
        elif chg_str[1:].isdigit():
            formal_charge = sign * int(chg_str[1:])
        else:
            formal_charge = sign * len(chg_str)

    return AtomRecord(
        index=atom_idx,
        symbol=sym,
        formal_charge=formal_charge,
        aromatic=aromatic,
        implicit_h=explicit_h,
        isotope=int(iso_str) if (iso_str is not None and raw_sym != "*") else None,
        dummy_label=dummy_label,
    )


def build_adjacency(mol: MolecularGraph) -> List[List[Tuple[int, BondRecord]]]:
    """Build adjacency lists ``[(neighbor_idx, BondRecord), ...]`` for all atoms."""
    adj: List[List[Tuple[int, BondRecord]]] = [[] for _ in mol.atoms]
    for bond in mol.bonds:
        adj[bond.atom_a].append((bond.atom_b, bond))
        adj[bond.atom_b].append((bond.atom_a, bond))
    return adj


def perceive_rings(mol: MolecularGraph) -> List[Tuple[int, ...]]:
    """
    Compute a cycle basis (Smallest Set of Smallest Rings approximation) using
    shortest-path BFS over non-tree chords followed by GF(2) linear independence
    filtering over bond incidence vectors.
    """
    n_atoms = len(mol.atoms)
    n_bonds = len(mol.bonds)
    if n_atoms == 0 or n_bonds == 0:
        return []

    adj = build_adjacency(mol)
    bond_index: Dict[Tuple[int, int], int] = {}
    for b_idx, bond in enumerate(mol.bonds):
        lo, hi = (bond.atom_a, bond.atom_b) if bond.atom_a < bond.atom_b else (bond.atom_b, bond.atom_a)
        bond_index[(lo, hi)] = b_idx

    candidate_cycles: List[Tuple[int, Tuple[int, ...]]] = []
    for b_idx, bond in enumerate(mol.bonds):
        u, v = bond.atom_a, bond.atom_b
        # Shortest path from u to v avoiding direct bond (u, v)
        parent: Dict[int, int] = {u: -1}
        queue: deque[int] = deque([u])
        found = False
        while queue and not found:
            curr = queue.popleft()
            for nbr, _ in adj[curr]:
                if (curr == u and nbr == v) or (curr == v and nbr == u):
                    continue
                if nbr not in parent:
                    parent[nbr] = curr
                    if nbr == v:
                        found = True
                        break
                    queue.append(nbr)
        if found:
            path: List[int] = []
            node = v
            while node != -1:
                path.append(node)
                node = parent[node]
            vec_mask = 0
            for k in range(len(path)):
                a = path[k]
                b = path[(k + 1) % len(path)]
                lo, hi = (a, b) if a < b else (b, a)
                vec_mask ^= 1 << bond_index[(lo, hi)]
            candidate_cycles.append((vec_mask, tuple(path)))

    # Sort shortest cycles first and retain GF(2) independent cycles
    candidate_cycles.sort(key=lambda item: len(item[1]))
    basis_pivots: Dict[int, int] = {}
    selected: List[Tuple[int, ...]] = []
    seen_sets: Set[frozenset[int]] = set()
    for mask, cyc in candidate_cycles:
        cyc_set = frozenset(cyc)
        if cyc_set in seen_sets:
            continue
        reduced = mask
        while reduced:
            msb = reduced.bit_length() - 1
            if msb in basis_pivots:
                reduced ^= basis_pivots[msb]
            else:
                basis_pivots[msb] = reduced
                seen_sets.add(cyc_set)
                selected.append(cyc)
                break
    return selected


def _annotate_topology_and_properties(mol: MolecularGraph) -> None:
    """Populate rings, aromaticity, implicit H, hybridization, descriptors, and warhead tags."""
    rings = perceive_rings(mol)
    mol.rings = rings
    ring_atoms: Set[int] = set()
    ring_bond_pairs: Set[Tuple[int, int]] = set()
    for cyc in rings:
        ring_atoms.update(cyc)
        for k in range(len(cyc)):
            a = cyc[k]
            b = cyc[(k + 1) % len(cyc)]
            ring_bond_pairs.add((min(a, b), max(a, b)))

    for atom in mol.atoms:
        atom.in_ring = atom.index in ring_atoms
    for bond in mol.bonds:
        bond.in_ring = (min(bond.atom_a, bond.atom_b), max(bond.atom_a, bond.atom_b)) in ring_bond_pairs

    # Huckel 4n+2 aromaticity verification for 5/6-membered rings marked lowercase or alternating double bonds
    for cyc in rings:
        if len(cyc) in (5, 6) and all(mol.atoms[idx].aromatic for idx in cyc):
            for k in range(len(cyc)):
                a = cyc[k]
                b = cyc[(k + 1) % len(cyc)]
                for bond in mol.bonds:
                    if {bond.atom_a, bond.atom_b} == {a, b}:
                        bond.order = 1.5
                        bond.aromatic = True

    adj = build_adjacency(mol)
    for atom in mol.atoms:
        if atom.symbol == "*":
            atom.implicit_h = 0
            atom.hybridization = "sp3"
            continue
        bond_sum = sum(b.order for _, b in adj[atom.index])
        max_order = max((b.order for _, b in adj[atom.index]), default=1.0)
        if atom.aromatic or abs(max_order - 1.5) < 1e-6 or abs(max_order - 2.0) < 1e-6:
            atom.hybridization = "sp2"
        elif abs(max_order - 3.0) < 1e-6:
            atom.hybridization = "sp"
        else:
            atom.hybridization = "sp3"

        # Fill implicit hydrogens only if not explicitly specified in bracket atom with explicit H
        params = ELEMENT_PARAMS[atom.symbol]
        effective_valence = params.default_valences[0] + atom.formal_charge
        if atom.symbol == "N" and atom.formal_charge == 0 and bond_sum > 3.1:
            effective_valence = 5
        if atom.symbol == "S" and bond_sum > 2.1:
            effective_valence = 4 if bond_sum <= 4.1 else 6
        if atom.symbol == "P" and bond_sum > 3.1:
            effective_valence = 5
        if atom.aromatic and atom.symbol == "N" and atom.implicit_h > 0:
            pass  # keep explicit [nH]
        elif atom.implicit_h == 0:
            rem = effective_valence - bond_sum
            if atom.aromatic and atom.symbol == "N" and len(adj[atom.index]) == 2:
                rem = 0.0  # pyridine-like nitrogen
            atom.implicit_h = max(0, int(round(rem)))

    # Compute physicochemical descriptors
    mw = 0.0
    logp = 0.0
    tpsa = 0.0
    hbd = 0
    hba = 0
    rot_bonds = 0
    total_charge = 0
    for atom in mol.atoms:
        if atom.symbol == "*":
            continue
        params = ELEMENT_PARAMS[atom.symbol]
        mw += params.mass + atom.implicit_h * ELEMENT_PARAMS["H"].mass
        total_charge += atom.formal_charge
        logp += _BASE_LOGP.get(atom.symbol, 0.0) + atom.implicit_h * 0.12
        if atom.aromatic and atom.symbol == "C":
            logp += 0.05
        if atom.symbol in ("N", "O"):
            if atom.implicit_h > 0:
                hbd += 1
            hba += 1
            tpsa += _ertl_tpsa_atom(atom, adj[atom.index])
        elif atom.symbol == "S":
            tpsa += 19.2 if len(adj[atom.index]) <= 2 else 34.1

    for bond in mol.bonds:
        if (
            not bond.in_ring
            and abs(bond.order - 1.0) < 1e-6
            and mol.atoms[bond.atom_a].symbol not in ("*", "H")
            and mol.atoms[bond.atom_b].symbol not in ("*", "H")
            and len(adj[bond.atom_a]) > 1
            and len(adj[bond.atom_b]) > 1
        ):
            # Exclude amide C-N partial double bonds from rotatable count
            if not _is_amide_bond(mol, bond, adj):
                rot_bonds += 1

    mol.molecular_weight = float(mw)
    mol.logp = float(logp)
    mol.tpsa = float(tpsa)
    mol.hbd = hbd
    mol.hba = hba
    mol.rotatable_bonds = rot_bonds
    mol.formal_charge = total_charge
    mol.sa_score = _compute_sa_score(mol)
    mol.qed = _compute_qed(mol)
    mol.pka_acid, mol.pka_base = _estimate_micro_pka(mol, adj)
    mol.warhead_class, mol.electrophile_atom_idx = detect_covalent_warhead(mol)


def _ertl_tpsa_atom(atom: AtomRecord, neighbors: List[Tuple[int, BondRecord]]) -> float:
    deg = len(neighbors)
    max_order = max((b.order for _, b in neighbors), default=1.0)
    if atom.symbol == "O":
        if atom.implicit_h > 0:
            return 20.23
        if abs(max_order - 2.0) < 1e-6:
            return 17.07
        if atom.aromatic:
            return 13.14
        return 9.23
    if atom.symbol == "N":
        if atom.aromatic:
            return 15.79 if atom.implicit_h > 0 else 12.89
        if abs(max_order - 3.0) < 1e-6:
            return 23.79
        if atom.implicit_h >= 2:
            return 26.02
        if atom.implicit_h == 1:
            return 12.03
        return 3.24 if deg >= 3 else 12.36
    return 0.0


def _is_amide_bond(
    mol: MolecularGraph,
    bond: BondRecord,
    adj: List[List[Tuple[int, BondRecord]]],
) -> bool:
    for c_idx, n_idx in ((bond.atom_a, bond.atom_b), (bond.atom_b, bond.atom_a)):
        if mol.atoms[c_idx].symbol == "C" and mol.atoms[n_idx].symbol == "N":
            for nbr, nbond in adj[c_idx]:
                if mol.atoms[nbr].symbol == "O" and abs(nbond.order - 2.0) < 1e-6:
                    return True
    return False


def _compute_sa_score(mol: MolecularGraph) -> float:
    """Synthetic accessibility heuristic on [1.0 (trivial), 10.0 (highly complex)]."""
    heavy = sum(1 for a in mol.atoms if a.symbol not in ("H", "*"))
    n_rings = len(mol.rings)
    shared_ring_atoms = sum(
        1 for a in mol.atoms if sum(1 for r in mol.rings if a.index in r) >= 2
    )
    hetero = sum(1 for a in mol.atoms if a.symbol in ("N", "O", "S", "P", "B", "F", "Cl", "Br"))
    score = 1.2 + 0.045 * heavy + 0.35 * n_rings + 0.55 * shared_ring_atoms + 0.08 * hetero
    return float(min(10.0, max(1.0, score)))


def _compute_qed(mol: MolecularGraph) -> float:
    """Bickerton-style quantitative estimate of drug-likeness using smooth desirability Gaussians."""
    def _gauss(val: float, center: float, width: float) -> float:
        return math.exp(-0.5 * ((val - center) / width) ** 2)

    d_mw = _gauss(mol.molecular_weight, 340.0, 110.0)
    d_logp = _gauss(mol.logp, 2.5, 1.8)
    d_tpsa = _gauss(mol.tpsa, 72.0, 35.0)
    d_hbd = _gauss(float(mol.hbd), 1.5, 1.8)
    d_hba = _gauss(float(mol.hba), 4.5, 2.8)
    d_rot = _gauss(float(mol.rotatable_bonds), 4.0, 3.2)
    weights = (0.20, 0.22, 0.18, 0.14, 0.14, 0.12)
    vals = (d_mw, d_logp, d_tpsa, d_hbd, d_hba, d_rot)
    log_qed = sum(w * math.log(max(v, 1e-6)) for w, v in zip(weights, vals))
    return float(math.exp(log_qed))


def _estimate_micro_pka(
    mol: MolecularGraph,
    adj: List[List[Tuple[int, BondRecord]]],
) -> Tuple[Optional[float], Optional[float]]:
    """Heuristic acidic/basic micro-pKa assignment for Henderson-Hasselbalch partitioning."""
    pka_acid: Optional[float] = None
    pka_base: Optional[float] = None
    for atom in mol.atoms:
        nbrs = adj[atom.index]
        if atom.symbol == "O" and atom.implicit_h == 1:
            for n_idx, _ in nbrs:
                if mol.atoms[n_idx].symbol == "C":
                    is_carboxyl = any(
                        mol.atoms[m].symbol == "O" and abs(b.order - 2.0) < 1e-6
                        for m, b in adj[n_idx]
                        if m != atom.index
                    )
                    if is_carboxyl:
                        pka_acid = 4.2 if pka_acid is None else min(pka_acid, 4.2)
                    elif mol.atoms[n_idx].aromatic:
                        pka_acid = 9.8 if pka_acid is None else min(pka_acid, 9.8)
        elif atom.symbol == "N" and not atom.aromatic and atom.formal_charge >= 0:
            is_amide_or_aniline = any(
                mol.atoms[n].aromatic
                or any(mol.atoms[m].symbol == "O" and abs(b.order - 2.0) < 1e-6 for m, b in adj[n])
                for n, _ in nbrs
            )
            if not is_amide_or_aniline:
                # Aliphatic amine basic pKa (e.g., dimethylaminomethyl warhead ~ 8.4)
                pka_base = 8.4 if pka_base is None else max(pka_base, 8.4)
            elif any(mol.atoms[n].aromatic for n, _ in nbrs):
                pka_base = 4.5 if pka_base is None else max(pka_base, 4.5)
    return pka_acid, pka_base


def detect_covalent_warhead(mol: MolecularGraph) -> Tuple[Optional[str], Optional[int]]:
    """
    Identify canonical covalent warhead motifs and locate the primary electrophilic
    carbon atom index.

    Returns
    -------
    Tuple[Optional[str], Optional[int]]
        ``(warhead_name, electrophile_atom_idx)``. If no covalent warhead is found,
        returns ``(None, None)``.
    """
    adj = build_adjacency(mol)

    # 1. Chloroacetamide: N-C(=O)-C-Cl
    for atom in mol.atoms:
        if atom.symbol == "C":
            nbrs = adj[atom.index]
            has_cl = any(mol.atoms[n].symbol == "Cl" for n, _ in nbrs)
            carbonyl_c = [
                n
                for n, b in nbrs
                if mol.atoms[n].symbol == "C"
                and abs(b.order - 1.0) < 1e-6
                and any(mol.atoms[m].symbol == "O" and abs(ob.order - 2.0) < 1e-6 for m, ob in adj[n])
            ]
            if has_cl and carbonyl_c:
                return "chloroacetamide", atom.index

    # 2. Vinyl sulfone: S(=O)(=O)-C=C
    for atom in mol.atoms:
        if atom.symbol == "S":
            oxo_count = sum(
                1 for n, b in adj[atom.index] if mol.atoms[n].symbol == "O" and abs(b.order - 2.0) < 1e-6
            )
            if oxo_count >= 2:
                for c_alpha, _ in adj[atom.index]:
                    if mol.atoms[c_alpha].symbol == "C":
                        for c_beta, b2 in adj[c_alpha]:
                            if c_beta != atom.index and mol.atoms[c_beta].symbol == "C" and abs(b2.order - 2.0) < 1e-6:
                                return "vinyl_sulfone", c_beta

    # 3. Cyanoacrylamide vs Acrylamide / Dimethylaminomethyl-acrylamide: C(=O)-C=C
    for atom in mol.atoms:
        if atom.symbol == "C":
            has_carbonyl_o = any(
                mol.atoms[n].symbol == "O" and abs(b.order - 2.0) < 1e-6 for n, b in adj[atom.index]
            )
            if not has_carbonyl_o:
                continue
            for c_alpha, b_ca in adj[atom.index]:
                if mol.atoms[c_alpha].symbol != "C" or abs(b_ca.order - 1.0) > 1e-6:
                    continue
                if c_alpha_is_olefin := [
                    c_beta
                    for c_beta, b_db in adj[c_alpha]
                    if c_beta != atom.index
                    and not mol.atoms[c_beta].in_ring
                    and mol.atoms[c_beta].symbol == "C"
                    and abs(b_db.order - 2.0) < 1e-6
                ]:
                    c_beta = c_alpha_is_olefin[0]
                    has_cyano = any(
                        mol.atoms[n].symbol == "C"
                        and any(mol.atoms[m].symbol == "N" and abs(tb.order - 3.0) < 1e-6 for m, tb in adj[n])
                        for n, _ in adj[c_alpha]
                        if n != atom.index and n != c_beta
                    )
                    if has_cyano:
                        return "cyanoacrylamide", c_beta
                    # Check dimethylaminomethyl tail on beta carbon
                    has_amine_tail = any(
                        mol.atoms[n].symbol == "C"
                        and any(mol.atoms[m].symbol == "N" and not mol.atoms[m].aromatic for m, _ in adj[n])
                        for n, _ in adj[c_beta]
                        if n != c_alpha
                    )
                    if has_amine_tail:
                        return "dimethylaminomethyl_acrylamide", c_beta
                    return "acrylamide", c_beta
                # Check butynamide C(=O)-C#C
                for c_beta, b_tb in adj[c_alpha]:
                    if (
                        c_beta != atom.index
                        and mol.atoms[c_beta].symbol == "C"
                        and abs(b_tb.order - 3.0) < 1e-6
                    ):
                        return "butynamide", c_beta

    # 4. Alpha-ketoamide: C(=O)-C(=O)-N
    for atom in mol.atoms:
        if atom.symbol == "C":
            has_o = any(mol.atoms[n].symbol == "O" and abs(b.order - 2.0) < 1e-6 for n, b in adj[atom.index])
            has_n = any(mol.atoms[n].symbol == "N" for n, _ in adj[atom.index])
            if has_o and has_n:
                for adj_c, _ in adj[atom.index]:
                    if mol.atoms[adj_c].symbol == "C" and any(
                        mol.atoms[m].symbol == "O" and abs(b.order - 2.0) < 1e-6 for m, b in adj[adj_c]
                    ):
                        return "alpha_ketoamide", adj_c

    # 5. Nitrile: C#N
    for atom in mol.atoms:
        if atom.symbol == "C":
            for n_idx, bond in adj[atom.index]:
                if mol.atoms[n_idx].symbol == "N" and abs(bond.order - 3.0) < 1e-6:
                    return "nitrile", atom.index

    return None, None


def embed_3d_coordinates(mol: MolecularGraph, seed: int = 42) -> FloatArray:
    """
    Generate physically reasonable 3D heavy-atom coordinates via metric-matrix
    distance geometry followed by UFF-like harmonic bond/angle/repulsion minimization.

    Parameters
    ----------
    mol : MolecularGraph
        Molecular graph whose heavy atoms are embedded.
    seed : int
        Deterministic seed for metric-matrix jitter and torsion symmetry breaking.

    Returns
    -------
    FloatArray
        Array of shape ``(n_atoms, 3)`` in Angstroms.
    """
    n = len(mol.atoms)
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if n == 1:
        return np.zeros((1, 3), dtype=np.float64)

    rng = np.random.default_rng(seed)
    adj = build_adjacency(mol)

    # Ideal bond lengths from covalent radii minus bond-order contraction
    ideal_bonds: List[Tuple[int, int, float]] = []
    lb = np.full((n, n), 2.35, dtype=np.float64)
    ub = np.full((n, n), 18.0, dtype=np.float64)
    np.fill_diagonal(lb, 0.0)
    np.fill_diagonal(ub, 0.0)

    for bond in mol.bonds:
        a, b = bond.atom_a, bond.atom_b
        ra = ELEMENT_PARAMS[mol.atoms[a].symbol].covalent_radius
        rb = ELEMENT_PARAMS[mol.atoms[b].symbol].covalent_radius
        contraction = (
            0.0 if bond.order <= 1.05 else (0.11 if bond.order <= 1.6 else (0.18 if bond.order <= 2.2 else 0.30))
        )
        d0 = max(1.05, ra + rb - contraction)
        ideal_bonds.append((a, b, d0))
        lb[a, b] = lb[b, a] = d0 - 0.02
        ub[a, b] = ub[b, a] = d0 + 0.02

    # 1-3 angle distance targets
    ideal_13: List[Tuple[int, int, float]] = []
    bond_len_lookup: Dict[Tuple[int, int], float] = {
        (min(a, b), max(a, b)): d0 for a, b, d0 in ideal_bonds
    }
    for center in range(n):
        nbrs = [nbr for nbr, _ in adj[center]]
        if len(nbrs) < 2:
            continue
        hyb = mol.atoms[center].hybridization
        theta = math.radians(180.0 if hyb == "sp" else (120.0 if hyb == "sp2" else 109.5))
        for p in range(len(nbrs)):
            for q in range(p + 1, len(nbrs)):
                u, v = nbrs[p], nbrs[q]
                d1 = bond_len_lookup[(min(center, u), max(center, u))]
                d2 = bond_len_lookup[(min(center, v), max(center, v))]
                d13 = math.sqrt(max(0.5, d1 * d1 + d2 * d2 - 2.0 * d1 * d2 * math.cos(theta)))
                ideal_13.append((u, v, d13))
                lb[u, v] = lb[v, u] = max(lb[u, v], d13 - 0.08)
                ub[u, v] = ub[v, u] = min(ub[u, v], d13 + 0.08)

    # Triangle inequality smoothing (Floyd-Warshall)
    for k in range(n):
        for i in range(n):
            dik = ub[i, k]
            for j in range(i + 1, n):
                cand_ub = dik + ub[k, j]
                if cand_ub < ub[i, j]:
                    ub[i, j] = ub[j, i] = cand_ub
                cand_lb1 = lb[i, k] - ub[k, j]
                cand_lb2 = lb[j, k] - ub[i, k]
                cand_lb = max(cand_lb1, cand_lb2)
                if cand_lb > lb[i, j]:
                    lb[i, j] = lb[j, i] = cand_lb

    # Sample distance matrix inside smoothed bounds & compute metric Gram matrix
    sample_frac: FloatArray = rng.uniform(0.25, 0.75, size=(n, n))
    sample_frac = 0.5 * (sample_frac + sample_frac.T)
    dist_mat = lb + sample_frac * np.maximum(0.0, ub - lb)
    np.fill_diagonal(dist_mat, 0.0)
    dist_sq = dist_mat ** 2
    row_mean = dist_sq.mean(axis=1, keepdims=True)
    col_mean = dist_sq.mean(axis=0, keepdims=True)
    gram = -0.5 * (dist_sq - row_mean - col_mean + dist_sq.mean())
    eigvals, eigvecs = np.linalg.eigh(gram)
    order = np.argsort(eigvals)[::-1]
    top_vals = np.maximum(eigvals[order[:3]], 1e-4)
    k_dim = min(3, n)
    coords0 = np.zeros((n, 3), dtype=np.float64)
    coords0[:, :k_dim] = eigvecs[:, order[:k_dim]] * np.sqrt(top_vals[:k_dim])
    coords0 += rng.normal(0.0, 0.05, size=(n, 3))

    # Build planar ring sets for soft planarity enforcement
    aromatic_rings = [r for r in mol.rings if len(r) in (5, 6) and all(mol.atoms[idx].aromatic for idx in r)]
    bonded_or_13: Set[Tuple[int, int]] = {(min(a, b), max(a, b)) for a, b, _ in ideal_bonds}
    bonded_or_13.update((min(u, v), max(u, v)) for u, v, _ in ideal_13)
    nonbonded_pairs = [
        (i, j) for i in range(n) for j in range(i + 1, n) if (i, j) not in bonded_or_13
    ]

    # Vectorized UFF-like energy + analytical gradient (20-100x faster than numerical BFGS)
    bond_arr_a = np.array([a for a, _, _ in ideal_bonds], dtype=np.int64)
    bond_arr_b = np.array([b for _, b, _ in ideal_bonds], dtype=np.int64)
    bond_d0 = np.array([d0 for _, _, d0 in ideal_bonds], dtype=np.float64)

    ang_arr_u = np.array([u for u, _, _ in ideal_13], dtype=np.int64)
    ang_arr_v = np.array([v for _, v, _ in ideal_13], dtype=np.int64)
    ang_d13 = np.array([d13 for _, _, d13 in ideal_13], dtype=np.float64)

    nb_arr_i = np.array([i for i, _ in nonbonded_pairs], dtype=np.int64)
    nb_arr_j = np.array([j for _, j in nonbonded_pairs], dtype=np.int64)

    def _uff_energy_and_grad(flat: FloatArray) -> Tuple[float, FloatArray]:
        x = flat.reshape((n, 3))
        grad = np.zeros((n, 3), dtype=np.float64)
        e = 0.0

        if len(bond_arr_a) > 0:
            diff_b = x[bond_arr_a] - x[bond_arr_b]
            r_b = np.sqrt(np.sum(diff_b ** 2, axis=1) + 1e-12)
            delta_b = r_b - bond_d0
            e += 450.0 * float(np.sum(delta_b ** 2))
            g_b = (900.0 * delta_b / r_b)[:, None] * diff_b
            np.add.at(grad, bond_arr_a, g_b)
            np.add.at(grad, bond_arr_b, -g_b)

        if len(ang_arr_u) > 0:
            diff_a = x[ang_arr_u] - x[ang_arr_v]
            r_a = np.sqrt(np.sum(diff_a ** 2, axis=1) + 1e-12)
            delta_a = r_a - ang_d13
            e += 120.0 * float(np.sum(delta_a ** 2))
            g_a = (240.0 * delta_a / r_a)[:, None] * diff_a
            np.add.at(grad, ang_arr_u, g_a)
            np.add.at(grad, ang_arr_v, -g_a)

        if len(nb_arr_i) > 0:
            diff_nb = x[nb_arr_i] - x[nb_arr_j]
            r2_nb = np.sum(diff_nb ** 2, axis=1) + 1e-12
            mask = r2_nb < 6.76
            if np.any(mask):
                r_m = np.sqrt(r2_nb[mask])
                delta_m = 2.6 - r_m
                e += 22.0 * float(np.sum(delta_m ** 2))
                g_m = (-44.0 * delta_m / r_m)[:, None] * diff_nb[mask]
                np.add.at(grad, nb_arr_i[mask], g_m)
                np.add.at(grad, nb_arr_j[mask], -g_m)

        for ring in aromatic_rings:
            r_idx = list(ring)
            pts = x[r_idx]
            mean_p = pts.mean(axis=0, keepdims=True)
            centered = pts - mean_p
            cov = centered.T @ centered
            eig_vals, eig_vecs = np.linalg.eigh(cov)
            min_v = max(0.0, float(eig_vals[0]))
            e += 80.0 * min_v
            normal = eig_vecs[:, 0]
            proj = centered @ normal
            grad[r_idx] += 160.0 * proj[:, None] * normal[None, :]

        return e, grad.ravel()

    res = minimize(
        _uff_energy_and_grad,
        coords0.ravel(),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 90},
    )
    coords_opt = np.asarray(res.x, dtype=np.float64).reshape((n, 3))
    coords_opt -= coords_opt.mean(axis=0, keepdims=True)
    return np.asarray(coords_opt, dtype=np.float64)


def to_canonical_smiles(mol: MolecularGraph) -> str:
    """
    Serialize a MolecularGraph into a deterministic canonical SMILES representation
    using Morgan-like degree/atomic-number invariant canonical atom ordering.
    """
    n = len(mol.atoms)
    if n == 0:
        return ""
    adj = build_adjacency(mol)

    # 3 rounds of Morgan-style invariant hashing
    ranks = [
        (
            ELEMENT_PARAMS[a.symbol].atomic_number * 100
            + len(adj[a.index]) * 10
            + (1 if a.aromatic else 0)
            + (a.dummy_label or 0) * 1000
        )
        for a in mol.atoms
    ]
    for _ in range(3):
        new_ranks = []
        for idx in range(n):
            nbr_sig = tuple(sorted(ranks[nbr] for nbr, _ in adj[idx]))
            new_ranks.append(hash((ranks[idx], nbr_sig)) & 0x7FFFFFFF)
        ranks = new_ranks

    visited: Set[int] = set()
    closure_bonds: Dict[Tuple[int, int], int] = {}
    atom_closures: Dict[int, List[Tuple[int, int, float]]] = {i: [] for i in range(n)}
    next_ring_id = 1

    # DFS tree construction to assign ring closures
    components = []
    remaining = set(range(n))
    while remaining:
        start = min(remaining, key=lambda idx: (ranks[idx], idx))
        comp_tokens: List[str] = []
        parent_map: Dict[int, int] = {start: -1}

        def _collect_closures(u: int) -> None:
            nonlocal next_ring_id
            visited.add(u)
            remaining.discard(u)
            ordered_nbrs = sorted(adj[u], key=lambda item: (ranks[item[0]], item[0]))
            for v, bond in ordered_nbrs:
                if v == parent_map[u]:
                    continue
                pair = (min(u, v), max(u, v))
                if v not in visited:
                    parent_map[v] = u
                    _collect_closures(v)
                elif pair not in closure_bonds:
                    rid = next_ring_id
                    next_ring_id += 1
                    closure_bonds[pair] = rid
                    atom_closures[u].append((rid, v, bond.order))
                    atom_closures[v].append((rid, u, bond.order))

        _collect_closures(start)

        def _emit(u: int, from_atom: int) -> str:
            token = _format_atom_smiles(mol.atoms[u])
            for rid, other, b_ord in sorted(atom_closures[u], key=lambda x: x[0]):
                b_sym = _bond_symbol(b_ord, mol.atoms[u], mol.atoms[other]) if u < other else ""
                r_str = f"{rid}" if rid < 10 else f"%{rid}"
                token += f"{b_sym}{r_str}"
            tree_children = [
                (v, bond)
                for v, bond in sorted(adj[u], key=lambda item: (ranks[item[0]], item[0]))
                if parent_map.get(v) == u
            ]
            if not tree_children:
                return token
            for idx_c, (v, bond) in enumerate(tree_children):
                b_sym = _bond_symbol(bond.order, mol.atoms[u], mol.atoms[v])
                sub = b_sym + _emit(v, u)
                if idx_c < len(tree_children) - 1:
                    token += f"({sub})"
                else:
                    token += sub
            return token

        comp_tokens.append(_emit(start, -1))
        components.append("".join(comp_tokens))

    return ".".join(sorted(components))


def _format_atom_smiles(atom: AtomRecord) -> str:
    if atom.symbol == "*":
        lbl = atom.dummy_label if atom.dummy_label is not None else 0
        return f"[{lbl}*]" if lbl > 0 else "[*]"
    if atom.formal_charge != 0 or (atom.aromatic and atom.symbol == "N" and atom.implicit_h > 0):
        sym = atom.symbol.lower() if atom.aromatic else atom.symbol
        h_part = f"H{atom.implicit_h}" if atom.implicit_h > 1 else ("H" if atom.implicit_h == 1 else "")
        chg_part = ""
        if atom.formal_charge > 0:
            chg_part = "+" if atom.formal_charge == 1 else f"+{atom.formal_charge}"
        elif atom.formal_charge < 0:
            chg_part = "-" if atom.formal_charge == -1 else f"{atom.formal_charge}"
        return f"[{sym}{h_part}{chg_part}]"
    if atom.aromatic and atom.symbol in ("C", "N", "O", "S", "P", "B"):
        return atom.symbol.lower()
    return atom.symbol


def _bond_symbol(order: float, a: AtomRecord, b: AtomRecord) -> str:
    if abs(order - 2.0) < 1e-6:
        return "="
    if abs(order - 3.0) < 1e-6:
        return "#"
    if abs(order - 1.5) < 1e-6 or (a.aromatic and b.aromatic):
        return ""
    return ""


def cleave_brics(mol: MolecularGraph) -> List[FragmentPiece]:
    """
    Apply BRICS-like retrosynthetic acyclic bond disconnections (amide C-N,
    biaryl C-C, aryl-heteroatom, and aliphatic ether/amine bonds) to produce
    capped dummy-labeled fragments.
    """
    adj = build_adjacency(mol)
    cut_bonds: List[Tuple[int, int, int, int]] = []  # (a, b, label_a, label_b)

    for bond in mol.bonds:
        if bond.in_ring or abs(bond.order - 1.0) > 1e-6:
            continue
        u, v = bond.atom_a, bond.atom_b
        su, sv = mol.atoms[u].symbol, mol.atoms[v].symbol
        if su == "*" or sv == "*":
            continue
        if _is_amide_bond(mol, bond, adj):
            cut_bonds.append((u, v, 1, 5))
        elif mol.atoms[u].in_ring and mol.atoms[v].in_ring:
            cut_bonds.append((u, v, 14, 14))
        elif (mol.atoms[u].aromatic and sv in ("N", "O", "S")) or (
            mol.atoms[v].aromatic and su in ("N", "O", "S")
        ):
            cut_bonds.append((u, v, 16, 13))

    if not cut_bonds:
        return [
            FragmentPiece(
                smiles=to_canonical_smiles(mol),
                heavy_atom_count=sum(1 for a in mol.atoms if a.symbol not in ("H", "*")),
                attachment_labels=(),
                ring_count=len(mol.rings),
                role="intact_scaffold",
            )
        ]

    # Build fragmented graph with dummy caps inserted on cleaved bonds
    new_atoms = [
        AtomRecord(
            index=a.index,
            symbol=a.symbol,
            formal_charge=a.formal_charge,
            aromatic=a.aromatic,
            implicit_h=a.implicit_h,
            dummy_label=a.dummy_label,
        )
        for a in mol.atoms
    ]
    cut_set = {(min(u, v), max(u, v)): (u, v, lu, lv) for u, v, lu, lv in cut_bonds}
    new_bonds: List[BondRecord] = []
    for bond in mol.bonds:
        pair = (min(bond.atom_a, bond.atom_b), max(bond.atom_a, bond.atom_b))
        if pair in cut_set:
            u, v, lu, lv = cut_set[pair]
            du = len(new_atoms)
            new_atoms.append(AtomRecord(index=du, symbol="*", dummy_label=lu))
            dv = len(new_atoms)
            new_atoms.append(AtomRecord(index=dv, symbol="*", dummy_label=lv))
            new_bonds.append(BondRecord(atom_a=u, atom_b=du, order=1.0))
            new_bonds.append(BondRecord(atom_a=v, atom_b=dv, order=1.0))
        else:
            new_bonds.append(
                BondRecord(atom_a=bond.atom_a, atom_b=bond.atom_b, order=bond.order, aromatic=bond.aromatic)
            )

    # Extract connected components as independent FragmentPieces
    frag_adj: List[List[Tuple[int, BondRecord]]] = [[] for _ in new_atoms]
    for b in new_bonds:
        frag_adj[b.atom_a].append((b.atom_b, b))
        frag_adj[b.atom_b].append((b.atom_a, b))

    seen: Set[int] = set()
    pieces: List[FragmentPiece] = []
    for start in range(len(new_atoms)):
        if start in seen:
            continue
        comp_nodes: List[int] = []
        queue: deque[int] = deque([start])
        seen.add(start)
        while queue:
            curr = queue.popleft()
            comp_nodes.append(curr)
            for nbr, _ in frag_adj[curr]:
                if nbr not in seen:
                    seen.add(nbr)
                    queue.append(nbr)
        idx_map = {old: new_i for new_i, old in enumerate(comp_nodes)}
        sub_atoms = [
            AtomRecord(
                index=idx_map[old],
                symbol=new_atoms[old].symbol,
                formal_charge=new_atoms[old].formal_charge,
                aromatic=new_atoms[old].aromatic,
                implicit_h=new_atoms[old].implicit_h,
                dummy_label=new_atoms[old].dummy_label,
            )
            for old in comp_nodes
        ]
        sub_bonds: List[BondRecord] = []
        for b in new_bonds:
            if b.atom_a in idx_map and b.atom_b in idx_map:
                sub_bonds.append(
                    BondRecord(
                        atom_a=idx_map[b.atom_a],
                        atom_b=idx_map[b.atom_b],
                        order=b.order,
                        aromatic=b.aromatic,
                    )
                )
        sub_mol = MolecularGraph(name="frag", smiles="", atoms=sub_atoms, bonds=sub_bonds)
        sub_mol.rings = perceive_rings(sub_mol)
        frag_smi = to_canonical_smiles(sub_mol)
        heavy = sum(1 for a in sub_atoms if a.symbol not in ("H", "*"))
        if heavy == 0:
            continue
        labels = tuple(sorted(a.dummy_label or 0 for a in sub_atoms if a.symbol == "*"))
        role = (
            "warhead_motif"
            if ("C=C" in frag_smi or "C#N" in frag_smi or "CCl" in frag_smi)
            else "scaffold_fragment"
        )
        pieces.append(
            FragmentPiece(
                smiles=frag_smi,
                heavy_atom_count=heavy,
                attachment_labels=labels,
                ring_count=len(sub_mol.rings),
                role=role,
            )
        )
    return pieces


def attach_fragment(
    scaffold_smiles: str,
    fragment_smiles: str,
    name: str = "assembled",
    embed_3d: bool = True,
    seed: int = 42,
) -> MolecularGraph:
    """
    Fuse two SMILES strings by connecting the neighbor of the first dummy atom ``[*]``
    in ``scaffold_smiles`` directly to the neighbor of the first dummy atom in
    ``fragment_smiles``, deleting both dummy placeholders.
    """
    mol_a = parse_smiles(scaffold_smiles, name="part_a", embed_3d=False)
    mol_b = parse_smiles(fragment_smiles, name="part_b", embed_3d=False)

    dummy_a = next((a.index for a in mol_a.atoms if a.symbol == "*"), None)
    dummy_b = next((a.index for a in mol_b.atoms if a.symbol == "*"), None)
    if dummy_a is None or dummy_b is None:
        raise SmilesParseError("Both scaffold and fragment SMILES must contain a dummy attachment atom '[*]'.")

    adj_a = build_adjacency(mol_a)
    adj_b = build_adjacency(mol_b)
    if not adj_a[dummy_a] or not adj_b[dummy_b]:
        raise SmilesParseError("Dummy atom must have at least one covalent neighbor.")

    anchor_a = adj_a[dummy_a][0][0]
    anchor_b = adj_b[dummy_b][0][0]

    combined_atoms: List[AtomRecord] = []
    map_a: Dict[int, int] = {}
    for a in mol_a.atoms:
        if a.index == dummy_a:
            continue
        new_i = len(combined_atoms)
        map_a[a.index] = new_i
        combined_atoms.append(
            AtomRecord(
                index=new_i,
                symbol=a.symbol,
                formal_charge=a.formal_charge,
                aromatic=a.aromatic,
                dummy_label=a.dummy_label,
            )
        )
    map_b: Dict[int, int] = {}
    for b in mol_b.atoms:
        if b.index == dummy_b:
            continue
        new_i = len(combined_atoms)
        map_b[b.index] = new_i
        combined_atoms.append(
            AtomRecord(
                index=new_i,
                symbol=b.symbol,
                formal_charge=b.formal_charge,
                aromatic=b.aromatic,
                dummy_label=b.dummy_label,
            )
        )

    combined_bonds: List[BondRecord] = []
    for bond in mol_a.bonds:
        if dummy_a not in (bond.atom_a, bond.atom_b):
            combined_bonds.append(
                BondRecord(map_a[bond.atom_a], map_a[bond.atom_b], bond.order, bond.aromatic)
            )
    for bond in mol_b.bonds:
        if dummy_b not in (bond.atom_a, bond.atom_b):
            combined_bonds.append(
                BondRecord(map_b[bond.atom_a], map_b[bond.atom_b], bond.order, bond.aromatic)
            )
    combined_bonds.append(BondRecord(map_a[anchor_a], map_b[anchor_b], 1.0, False))

    tmp = MolecularGraph(name=name, smiles="", atoms=combined_atoms, bonds=combined_bonds)
    tmp.rings = perceive_rings(tmp)
    canonical = to_canonical_smiles(tmp)
    return parse_smiles(canonical, name=name, embed_3d=embed_3d, seed=seed)


def graft_warhead(
    scaffold_smiles: str,
    warhead_name: str,
    name: str = "covalent_candidate",
    seed: int = 42,
) -> MolecularGraph:
    """Graft a named covalent warhead template from ``WARHEAD_TEMPLATES`` onto a scaffold."""
    if warhead_name not in WARHEAD_TEMPLATES:
        raise SmilesParseError(f"Unknown warhead '{warhead_name}'. Available: {sorted(WARHEAD_TEMPLATES.keys())}")
    tpl = WARHEAD_TEMPLATES[warhead_name]
    mol = attach_fragment(scaffold_smiles, tpl.smiles_fragment, name=name, seed=seed)
    return mol


# Curated realistic covalent kinase & viral protease reference compounds (parseable SMILES)
REFERENCE_COVALENT_LIBRARY: Sequence[Tuple[str, str, str, bool, float]] = (
    # (name, smiles, target, is_active, ref_log10_kinact_over_Ki)
    ("Osimertinib_analog", "C=CC(=O)Nc1cc(Nc2nccc(-c3cn(C)c4ccccc34)n2)c(OC)cc1N(C)CCN(C)C", "EGFR_C797S", True, 6.45),
    ("Afatinib_analog", "C=CC(=O)Nc1ccc2ncnc(Nc3ccc(F)c(Cl)c3)c2c1OC", "EGFR_C797S", True, 5.95),
    ("Rociletinib_analog", "C=CC(=O)Nc1cccc(Nc2ncc(C(F)(F)F)c(Nc3ccc(N4CCN(C)CC4)cc3)n2)c1", "EGFR_C797S", True, 5.40),
    ("WZ4002_analog", "C=CC(=O)Nc1cccc(Oc2nc(Nc3ccc(N4CCN(C)CC4)cc3)ncc2Cl)c1", "EGFR_C797S", True, 5.65),
    ("Ibrutinib_analog", "C=CC(=O)N1CCCC(n2nc(-c3ccc(Oc4ccccc4)cc3)c3c(N)ncnc32)C1", "BTK_C481", True, 6.30),
    ("Acalabrutinib_analog", "CC#CC(=O)N1CCCC(c2nc(-c3ccc(C(=O)Nc4ccccn4)cc3)c3c(N)nccn23)C1", "BTK_C481", True, 5.80),
    ("Zanubrutinib_analog", "C=CC(=O)N1CCC(C2CCNc3c(C(=O)N)nn(-c4ccc(Oc5ccccc5)cc4)c32)CC1", "BTK_C481", True, 6.10),
    (
        "Sotorasib_analog",
        "C=CC(=O)N1CCN(c2nc(=O)n(-c3c(C)ccnc3C(C)C)c3nc(-c4c(O)cccc4F)c(F)cc23)CC1",
        "KRAS_G12C",
        True,
        5.25,
    ),
    ("Adagrasib_analog", "C=CC(=O)N1CCN(c2nc(OCC3CCCN3C)nc3c2CCN(c2cccc4cccc(Cl)c24)C3)CC1", "KRAS_G12C", True, 5.55),
    (
        "Nirmatrelvir_analog",
        "CC(C)(C)C(NC(=O)C(F)(F)F)C(=O)N1CC2(C1)C(C(=O)NC(CC1CCNC1=O)C#N)C2",
        "MPRO_C145",
        True,
        5.70,
    ),
    (
        "Boceprevir_analog",
        "CC(C)(C)NC(=O)NC(C(C)(C)C)C(=O)N1CC2(C1)C(C(=O)NC(CC1CCC1)C(=O)C(=O)N)C2",
        "MPRO_C145",
        True,
        5.10,
    ),
    (
        "Cyano_BTK_reversible",
        "CC(C)(C)C=C(C#N)C(=O)N1CCCC(n2nc(-c3ccc(Oc4ccccc4)cc3)c3c(N)ncnc32)C1",
        "BTK_C481",
        True,
        5.20,
    ),
    ("Gefitinib_noncov_decoy", "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1", "EGFR_C797S", False, 1.20),
    ("Erlotinib_noncov_decoy", "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC", "EGFR_C797S", False, 1.10),
    ("Chloroacet_promiscuous", "ClCC(=O)Nc1ccc(Cl)cc1", "EGFR_C797S", False, 2.30),
    ("Vinylsulfone_decoy", "C=CS(=O)(=O)c1ccc(C)cc1", "BTK_C481", False, 2.15),
)
