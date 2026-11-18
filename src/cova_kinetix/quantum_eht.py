"""
Extended Huckel Theory (EHT) semi-empirical quantum chemistry and conceptual DFT
reactivity engine in pure Python + NumPy/SciPy.

Computes exact 3D Slater-type orbital (STO) overlap integrals in prolate
spheroidal coordinates via Gauss-Laguerre / Gauss-Legendre quadrature, builds the
Wolfsberg-Helmholz Hamiltonian, solves the generalized eigenvalue problem via
Lowdin symmetric orthogonalization, and derives Mulliken/Lowdin populations,
Mayer bond orders, Parr electrophilicity, condensed Fukui functions (f+, f-, f0),
dual descriptors, and predicted glutathione (GSH) off-target reactivity half-life.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
from scipy.linalg import eigh
from scipy.special import roots_laguerre, roots_legendre

from cova_kinetix.fragment_graph import build_adjacency, embed_3d_coordinates
from cova_kinetix.types import (
    ANGSTROM_TO_BOHR,
    ELEMENT_PARAMS,
    WOLFSBERG_HELMHOLZ_K,
    EHTResult,
    FloatArray,
    MolecularGraph,
    QuantumConvergenceError,
)

# Precompute quadrature nodes and weights once at module import for fast STO integration
_XI_T, _XI_W = roots_laguerre(24)
_ETA_X, _ETA_W = roots_legendre(16)
_T_GRID, _ETA_GRID = np.meshgrid(_XI_T, _ETA_X, indexing="ij")
_W_GRID = np.outer(_XI_W, _ETA_W)


def _sto_norm(n: int, zeta: float) -> float:
    """Analytical radial normalization constant for Slater-type orbital R_{n}(r) = N r^{n-1} e^{-zeta r}."""
    return float(((2.0 * zeta) ** (n + 0.5)) / math.sqrt(float(math.factorial(2 * n))))


def _diatomic_sto_overlaps(
    na: int,
    zeta_sa: float,
    zeta_pa: Optional[float],
    nb: int,
    zeta_sb: float,
    zeta_pb: Optional[float],
    r_bohr: float,
) -> Tuple[float, float, float, float, float]:
    """
    Compute exact 3D diatomic STO overlap integrals aligned along the A -> B z'-axis:
    ``(S_ss, S_sA_pB, S_pA_sB, S_pp_sigma, S_pp_pi)``.

    Uses prolate spheroidal coordinates (xi, eta, phi) with exponential shift
    t = 0.5 * R * (zeta_a + zeta_b) * (xi - 1) so the quadrature is exact for all
    internuclear separations R > 0.
    """
    if r_bohr < 1e-8:
        return (1.0, 0.0, 0.0, 1.0, 1.0)
    if r_bohr > 14.0:
        return (0.0, 0.0, 0.0, 0.0, 0.0)

    def _compute_shared(za: float, zb: float) -> Tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
        alpha = 0.5 * r_bohr * (za + zb)
        beta = 0.5 * r_bohr * (za - zb)
        xi = 1.0 + _T_GRID / alpha
        eta = _ETA_GRID
        r_a = 0.5 * r_bohr * (xi + eta)
        r_b = 0.5 * r_bohr * (xi - eta)
        rad_a = r_a ** (na - 1) if na > 1 else np.ones_like(r_a)
        rad_b = r_b ** (nb - 1) if nb > 1 else np.ones_like(r_b)
        jac = (0.5 * r_bohr) ** 3 * (xi ** 2 - eta ** 2) / alpha
        base = _W_GRID * _sto_norm(na, za) * _sto_norm(nb, zb) * rad_a * rad_b * np.exp(-alpha - beta * eta) * jac
        return xi, eta, r_a, r_b, base

    # Fast path when s and p share the same Slater exponent (true for H, B, C, N, O, F, P, S, Cl)
    same_za = (zeta_pa is None) or abs(zeta_sa - zeta_pa) < 1e-12
    same_zb = (zeta_pb is None) or abs(zeta_sb - zeta_pb) < 1e-12

    xi, eta, r_a, r_b, base_ss = _compute_shared(zeta_sa, zeta_sb)
    s_ss = float(np.sum(base_ss * 0.5))

    s_s_pz = 0.0
    if zeta_pb is not None:
        if same_zb:
            b_sp = base_ss
            r_b_sp = r_b
            xi_sp, eta_sp = xi, eta
        else:
            xi_sp, eta_sp, _, r_b_sp, b_sp = _compute_shared(zeta_sa, zeta_pb)
        z_b = 0.5 * r_bohr * (xi_sp * eta_sp - 1.0)
        s_s_pz = float(np.sum(b_sp * (0.5 * math.sqrt(3.0)) * (z_b / np.maximum(r_b_sp, 1e-12))))

    s_pz_s = 0.0
    if zeta_pa is not None:
        if same_za:
            b_ps = base_ss
            r_a_ps = r_a
            xi_ps, eta_ps = xi, eta
        else:
            xi_ps, eta_ps, r_a_ps, _, b_ps = _compute_shared(zeta_pa, zeta_sb)
        z_a = 0.5 * r_bohr * (xi_ps * eta_ps + 1.0)
        s_pz_s = float(np.sum(b_ps * (0.5 * math.sqrt(3.0)) * (z_a / np.maximum(r_a_ps, 1e-12))))

    if zeta_pa is not None and zeta_pb is not None:
        if same_za and same_zb:
            b_pp = base_ss
            xi_pp, eta_pp, r_a_pp, r_b_pp = xi, eta, r_a, r_b
        else:
            xi_pp, eta_pp, r_a_pp, r_b_pp, b_pp = _compute_shared(zeta_pa, zeta_pb)
        inv_rab = 1.0 / np.maximum(r_a_pp * r_b_pp, 1e-12)
        z_a_pp = 0.5 * r_bohr * (xi_pp * eta_pp + 1.0)
        z_b_pp = 0.5 * r_bohr * (xi_pp * eta_pp - 1.0)
        s_sig = float(np.sum(b_pp * 1.5 * (z_a_pp * z_b_pp) * inv_rab))
        rho2 = ((0.5 * r_bohr) ** 2) * np.maximum(0.0, (xi_pp ** 2 - 1.0) * (1.0 - eta_pp ** 2))
        s_pi = float(np.sum(b_pp * 0.75 * rho2 * inv_rab))
    else:
        s_sig = 0.0
        s_pi = 0.0
    return (s_ss, s_s_pz, s_pz_s, s_sig, s_pi)


def expand_explicit_hydrogens_3d(mol: MolecularGraph) -> Tuple[List[str], FloatArray, List[int]]:
    """
    Expand implicit hydrogens on a MolecularGraph into explicit 3D atomic positions
    using local VSEPR tetrahedral/trigonal/linear placement.

    Returns
    -------
    Tuple[List[str], FloatArray, List[int]]
        ``(symbols, coords_angstrom, heavy_atom_map)`` where ``heavy_atom_map[i]``
        is the original heavy atom index for heavy atoms or ``-1`` for explicit H.
    """
    if mol.coords_3d is None:
        heavy_coords = embed_3d_coordinates(mol)
    else:
        heavy_coords = np.asarray(mol.coords_3d, dtype=np.float64)

    adj = build_adjacency(mol)
    symbols: List[str] = []
    coords_list: List[FloatArray] = []
    heavy_map: List[int] = []

    for idx, atom in enumerate(mol.atoms):
        if atom.symbol == "*":
            continue
        symbols.append(atom.symbol)
        coords_list.append(heavy_coords[idx].copy())
        heavy_map.append(idx)

    r_h_cov = ELEMENT_PARAMS["H"].covalent_radius
    for idx, atom in enumerate(mol.atoms):
        if atom.symbol == "*" or atom.implicit_h <= 0:
            continue
        origin = heavy_coords[idx]
        r_xh = ELEMENT_PARAMS[atom.symbol].covalent_radius + r_h_cov
        nbr_vecs: List[FloatArray] = []
        for nbr, _ in adj[idx]:
            if mol.atoms[nbr].symbol != "*":
                v = heavy_coords[nbr] - origin
                norm_v = float(np.linalg.norm(v))
                if norm_v > 1e-6:
                    nbr_vecs.append(v / norm_v)

        h_dirs = _vsepr_hydrogen_directions(nbr_vecs, atom.implicit_h, atom.hybridization)
        for d in h_dirs:
            symbols.append("H")
            coords_list.append(origin + r_xh * d)
            heavy_map.append(-1)

    return symbols, np.vstack(coords_list), heavy_map


def _vsepr_hydrogen_directions(
    nbr_dirs: List[FloatArray],
    n_h: int,
    hybridization: str,
) -> List[FloatArray]:
    """Generate unit vectors for ``n_h`` hydrogens around a central atom given heavy neighbor unit vectors."""
    if not nbr_dirs:
        tetra = [
            np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0),
            np.array([1.0, -1.0, -1.0]) / math.sqrt(3.0),
            np.array([-1.0, 1.0, -1.0]) / math.sqrt(3.0),
            np.array([-1.0, -1.0, 1.0]) / math.sqrt(3.0),
        ]
        return tetra[:n_h]

    mean_dir = np.sum(nbr_dirs, axis=0)
    norm_m = float(np.linalg.norm(mean_dir))
    if norm_m < 1e-6:
        # Linear opposite neighbors -> pick orthogonal vector
        ref = np.array([1.0, 0.0, 0.0]) if abs(nbr_dirs[0][0]) < 0.8 else np.array([0.0, 1.0, 0.0])
        u = np.cross(nbr_dirs[0], ref)
        u /= max(float(np.linalg.norm(u)), 1e-12)
        if n_h == 1:
            return [u]
        return [u, -u][:n_h]

    back = -mean_dir / norm_m
    if n_h == 1:
        return [back]

    # Build orthonormal frame around `back`
    ref = np.array([1.0, 0.0, 0.0]) if abs(back[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(back, ref)
    e1 /= max(float(np.linalg.norm(e1)), 1e-12)
    e2 = np.cross(back, e1)

    if len(nbr_dirs) == 1 and hybridization == "sp2" and n_h == 2:
        # Trigonal planar terminal =CH2 or -NH2
        half_ang = math.radians(60.0)
        return [
            math.cos(half_ang) * back + math.sin(half_ang) * e1,
            math.cos(half_ang) * back - math.sin(half_ang) * e1,
        ]

    # Tetrahedral cone around `back`
    cone_ang = math.radians(54.735)
    out: List[FloatArray] = []
    for k in range(n_h):
        phi = 2.0 * math.pi * k / float(n_h)
        vec = (
            math.cos(cone_ang) * back
            + math.sin(cone_ang) * (math.cos(phi) * e1 + math.sin(phi) * e2)
        )
        out.append(vec / max(float(np.linalg.norm(vec)), 1e-12))
    return out


def build_eht_matrices(
    symbols: List[str],
    coords_angstrom: FloatArray,
) -> Tuple[FloatArray, FloatArray, List[Tuple[int, str]], int]:
    """
    Construct the 3D STO overlap matrix ``S`` and Wolfsberg-Helmholz Hamiltonian ``H``.

    Returns
    -------
    Tuple[FloatArray, FloatArray, List[Tuple[int, str]], int]
        ``(S, H, orbital_map, total_valence_electrons)`` where ``orbital_map[mu]``
        gives ``(atom_idx, orbital_type)``.
    """
    n_atoms = len(symbols)
    orbital_map: List[Tuple[int, str]] = []
    atom_orb_slices: List[Tuple[int, int]] = []
    diag_h: List[float] = []
    total_valence = 0

    for a_idx, sym in enumerate(symbols):
        p = ELEMENT_PARAMS[sym]
        total_valence += p.valence_electrons
        start = len(orbital_map)
        orbital_map.append((a_idx, "s"))
        diag_h.append(-p.vsip_s)
        if p.zeta_p is not None and p.vsip_p is not None:
            for orb in ("px", "py", "pz"):
                orbital_map.append((a_idx, orb))
                diag_h.append(-p.vsip_p)
        atom_orb_slices.append((start, len(orbital_map)))

    n_orb = len(orbital_map)
    s_mat = np.eye(n_orb, dtype=np.float64)
    coords_bohr = coords_angstrom * ANGSTROM_TO_BOHR

    for i in range(n_atoms):
        pi = ELEMENT_PARAMS[symbols[i]]
        si, ei = atom_orb_slices[i]
        has_pi = ei - si == 4
        for j in range(i + 1, n_atoms):
            pj = ELEMENT_PARAMS[symbols[j]]
            sj, ej = atom_orb_slices[j]
            has_pj = ej - sj == 4

            diff = coords_bohr[j] - coords_bohr[i]
            r_bohr = float(np.linalg.norm(diff))
            if r_bohr < 1e-9 or r_bohr > 14.0:
                continue
            u = diff / r_bohr

            s_ss, s_sp, s_ps, s_sig, s_pi = _diatomic_sto_overlaps(
                pi.principal_n,
                pi.zeta_s,
                pi.zeta_p,
                pj.principal_n,
                pj.zeta_s,
                pj.zeta_p,
                r_bohr,
            )

            s_mat[si, sj] = s_ss
            s_mat[sj, si] = s_ss

            if has_pj:
                vec_sp = s_sp * u
                s_mat[si, sj + 1 : sj + 4] = vec_sp
                s_mat[sj + 1 : sj + 4, si] = vec_sp

            if has_pi:
                vec_ps = s_ps * u
                s_mat[si + 1 : si + 4, sj] = vec_ps
                s_mat[sj, si + 1 : si + 4] = vec_ps

            if has_pi and has_pj:
                block_pp = s_sig * np.outer(u, u) + s_pi * (np.eye(3) - np.outer(u, u))
                s_mat[si + 1 : si + 4, sj + 1 : sj + 4] = block_pp
                s_mat[sj + 1 : sj + 4, si + 1 : si + 4] = block_pp.T

    # Wolfsberg-Helmholz Hamiltonian H_{uv} = 0.5 * K * (H_uu + H_vv) * S_{uv}
    h_diag = np.asarray(diag_h, dtype=np.float64)
    h_mat = 0.5 * WOLFSBERG_HELMHOLZ_K * (h_diag[:, None] + h_diag[None, :]) * s_mat
    np.fill_diagonal(h_mat, h_diag)
    return s_mat, h_mat, orbital_map, total_valence


def run_extended_huckel(
    mol: MolecularGraph,
    explicit_symbols: Optional[List[str]] = None,
    explicit_coords: Optional[FloatArray] = None,
) -> EHTResult:
    """
    Execute Extended Huckel Theory (EHT) electronic structure calculation and
    conceptual DFT reactivity analysis on a ``MolecularGraph``.

    Parameters
    ----------
    mol : MolecularGraph
        Input molecule.
    explicit_symbols : Optional[List[str]]
        Optional pre-expanded atomic symbols (including hydrogens).
    explicit_coords : Optional[FloatArray]
        Optional pre-expanded 3D coordinates (Angstroms).

    Returns
    -------
    EHTResult
        Orbital eigenvalues, Mulliken/Lowdin charges, Mayer bond orders,
        condensed Fukui indices, electrophilicity index, and predicted GSH
        half-life.
    """
    if explicit_symbols is not None and explicit_coords is not None:
        symbols = explicit_symbols
        coords = np.asarray(explicit_coords, dtype=np.float64)
        heavy_map = list(range(len(mol.atoms)))
    else:
        symbols, coords, heavy_map = expand_explicit_hydrogens_3d(mol)

    s_mat, h_mat, orbital_map, neutral_val_e = build_eht_matrices(symbols, coords)
    n_val_e = max(2, neutral_val_e - mol.formal_charge)

    # Lowdin symmetric orthogonalization S^{-1/2}
    s_vals, s_vecs = np.linalg.eigh(s_mat)
    if float(np.min(s_vals)) <= 1e-8:
        raise QuantumConvergenceError(
            f"Near-linear dependency in STO overlap matrix (min eigenvalue={float(np.min(s_vals)):.3e})."
        )
    inv_sqrt = s_vecs @ np.diag(1.0 / np.sqrt(s_vals)) @ s_vecs.T
    sqrt_s = s_vecs @ np.diag(np.sqrt(s_vals)) @ s_vecs.T

    h_prime = inv_sqrt @ h_mat @ inv_sqrt
    h_prime = 0.5 * (h_prime + h_prime.T)
    eps, c_prime = eigh(h_prime)
    c_mo = inv_sqrt @ c_prime

    # Closed-shell Aufbau electron occupation
    n_orb = len(eps)
    n_occ = min(n_orb, n_val_e // 2)
    rem_e = n_val_e - 2 * n_occ
    occ_vec = np.zeros(n_orb, dtype=np.float64)
    occ_vec[:n_occ] = 2.0
    if rem_e > 0 and n_occ < n_orb:
        occ_vec[n_occ] = float(rem_e)

    homo_idx = max(0, n_occ - 1)
    lumo_idx = min(n_orb - 1, homo_idx + 1)
    homo_ev = float(eps[homo_idx])
    lumo_ev = float(eps[lumo_idx])
    gap_ev = max(0.05, lumo_ev - homo_ev)

    # Conceptual DFT global reactivity indices (Koopmans-like)
    mu_ev = 0.5 * (homo_ev + lumo_ev)
    eta_ev = 0.5 * gap_ev
    omega_ev = (mu_ev ** 2) / (2.0 * eta_ev)

    # Density matrix P = C * diag(occ) * C^T
    p_mat = (c_mo * occ_vec[None, :]) @ c_mo.T
    ps_mat = p_mat @ s_mat
    trace_ps = float(np.trace(ps_mat))

    # Lowdin density matrix P_lowdin = S^{1/2} P S^{1/2}
    p_lowdin = sqrt_s @ p_mat @ sqrt_s

    n_atoms_total = len(symbols)
    atom_orbs: List[List[int]] = [[] for _ in range(n_atoms_total)]
    for mu, (a_idx, _) in enumerate(orbital_map):
        atom_orbs[a_idx].append(mu)

    mulliken_q = np.zeros(n_atoms_total, dtype=np.float64)
    lowdin_q = np.zeros(n_atoms_total, dtype=np.float64)
    for a_idx, sym in enumerate(symbols):
        z_val = float(ELEMENT_PARAMS[sym].valence_electrons)
        orbs = atom_orbs[a_idx]
        mulliken_q[a_idx] = z_val - float(np.sum(np.diag(ps_mat)[orbs]))
        lowdin_q[a_idx] = z_val - float(np.sum(np.diag(p_lowdin)[orbs]))

    # Mayer bond orders B_{AB} = sum_{u in A, v in B} (PS)_{uv} (PS)_{vu}
    mayer = np.zeros((n_atoms_total, n_atoms_total), dtype=np.float64)
    for a_idx in range(n_atoms_total):
        oa = atom_orbs[a_idx]
        for b_idx in range(a_idx + 1, n_atoms_total):
            ob = atom_orbs[b_idx]
            sub_ab = ps_mat[np.ix_(oa, ob)]
            sub_ba = ps_mat[np.ix_(ob, oa)]
            bo = float(np.sum(sub_ab * sub_ba.T))
            mayer[a_idx, b_idx] = bo
            mayer[b_idx, a_idx] = bo

    # Condensed Fukui indices via frontier orbital Mulliken density weights
    # Average over near-degenerate frontier MOs within 0.05 eV for rotational symmetry
    lumo_mask = np.abs(eps - lumo_ev) <= 0.05
    homo_mask = np.abs(eps - homo_ev) <= 0.05
    sc_mat = s_mat @ c_mo

    fukui_plus = np.zeros(n_atoms_total, dtype=np.float64)
    fukui_minus = np.zeros(n_atoms_total, dtype=np.float64)
    lumo_indices = np.where(lumo_mask)[0]
    homo_indices = np.where(homo_mask)[0]

    for a_idx in range(n_atoms_total):
        oa = atom_orbs[a_idx]
        fukui_plus[a_idx] = float(
            np.mean([np.sum(c_mo[oa, k] * sc_mat[oa, k]) for k in lumo_indices])
        )
        fukui_minus[a_idx] = float(
            np.mean([np.sum(c_mo[oa, k] * sc_mat[oa, k]) for k in homo_indices])
        )

    # Normalize positive Fukui weights cleanly
    fukui_plus = np.maximum(0.0, fukui_plus)
    fukui_minus = np.maximum(0.0, fukui_minus)
    if np.sum(fukui_plus) > 1e-12:
        fukui_plus /= np.sum(fukui_plus)
    if np.sum(fukui_minus) > 1e-12:
        fukui_minus /= np.sum(fukui_minus)

    fukui_rad = 0.5 * (fukui_plus + fukui_minus)
    dual_desc = fukui_plus - fukui_minus

    # Identify top heavy-atom electrophile center (prefer detected warhead carbon if present)
    n_heavy = sum(1 for h in heavy_map if h >= 0)
    heavy_fplus = fukui_plus[:n_heavy]
    top_heavy_idx = int(np.argmax(heavy_fplus)) if n_heavy > 0 else 0
    max_fplus_heavy = float(heavy_fplus[top_heavy_idx]) if n_heavy > 0 else 0.0

    # Local electrophilicity omega_k = omega * f_k^+ predicts GSH reactivity.
    # For SN2 alpha-haloamides or nitriles where LUMO localizes on the activating carbonyl/pi*,
    # combine the electrophilic atom and peak warhead LUMO density.
    warhead_fplus = (
        float(fukui_plus[mol.electrophile_atom_idx])
        if mol.electrophile_atom_idx is not None and mol.electrophile_atom_idx < n_heavy
        else max_fplus_heavy
    )
    eff_fplus = max(warhead_fplus, 0.35 * max_fplus_heavy, 0.05)
    local_omega = omega_ev * eff_fplus

    # Empirical calibration connecting local electrophilicity + warhead class to GSH t1/2 (min)
    gsh_t_half = _predict_gsh_half_life(mol.warhead_class, local_omega, omega_ev)
    hyper_reactive = (mol.warhead_class is not None) and (gsh_t_half < 25.0)

    return EHTResult(
        smiles=mol.smiles,
        n_atoms_total=n_atoms_total,
        n_orbitals=n_orb,
        n_valence_electrons=n_val_e,
        orbital_energies_ev=eps,
        homo_index=homo_idx,
        lumo_index=lumo_idx,
        homo_ev=homo_ev,
        lumo_ev=lumo_ev,
        homo_lumo_gap_ev=gap_ev,
        chemical_potential_ev=mu_ev,
        chemical_hardness_ev=eta_ev,
        electrophilicity_omega_ev=omega_ev,
        mulliken_charges=mulliken_q,
        lowdin_charges=lowdin_q,
        mayer_bond_orders=mayer,
        fukui_plus=fukui_plus,
        fukui_minus=fukui_minus,
        fukui_radical=fukui_rad,
        dual_descriptor=dual_desc,
        top_electrophile_heavy_idx=top_heavy_idx,
        max_fukui_plus_heavy=max_fplus_heavy,
        predicted_gsh_t_half_min=gsh_t_half,
        hyper_reactive_flag=hyper_reactive,
        trace_ps_electrons=trace_ps,
    )


def _predict_gsh_half_life(
    warhead_class: Optional[str],
    local_omega_ev: float,
    global_omega_ev: float,
    eht_slope: float = 0.06,
) -> float:
    """
    Estimate glutathione (GSH) pseudo-first-order half-life (minutes) at pH 7.4, 37 C
    from warhead class intrinsic kinetics and EHT local electrophilicity.
    """
    if warhead_class is None:
        return 1e4
    from cova_kinetix.types import WARHEAD_TEMPLATES

    tpl = WARHEAD_TEMPLATES.get(warhead_class)
    base_t_half = tpl.gsh_t_half_min_ref if tpl is not None else 120.0
    # EHT global omega typically spans 30-55 eV; local omega = omega * f+ spans 2-14 eV
    delta_log10 = -eht_slope * (local_omega_ev - 6.5) - 0.01 * (global_omega_ev - 42.0)
    delta_log10 = float(np.clip(delta_log10, -0.8, 0.8))
    return float(max(1.0, base_t_half * (10.0 ** delta_log10)))


def compute_eht_activation_barrier(
    eht: EHTResult,
    warhead_class: Optional[str],
    nucleophile_pka: float,
    physiological_ph: float = 7.4,
    eht_coupling_coeff: float = 0.12,
) -> float:
    """
    Compute the effective chemical inactivation barrier Delta G_double_dagger (kcal/mol)
    combining warhead intrinsic barrier, EHT local electrophilicity modulation, and
    nucleophile thiolate/amine Henderson-Hasselbalch ionization fraction.
    """
    from cova_kinetix.types import GAS_CONSTANT_KCAL, WARHEAD_TEMPLATES

    if warhead_class is None:
        return 35.0  # effectively non-covalent barrier

    tpl = WARHEAD_TEMPLATES.get(warhead_class)
    base_barrier = tpl.intrinsic_barrier_kcal if tpl is not None else 21.0
    local_omega = eht.electrophilicity_omega_ev * max(eht.max_fukui_plus_heavy, 0.05)

    # Higher local electrophilicity lowers the activation barrier around EHT midpoint 6.5 eV
    eht_shift = -eht_coupling_coeff * (local_omega - 6.5)
    eht_shift = float(np.clip(eht_shift, -1.5, 1.5))

    # Thiolate/deprotonated fraction penalty: f_nu = 1 / (1 + 10^(pKa - pH))
    f_nu = 1.0 / (1.0 + 10.0 ** (nucleophile_pka - physiological_ph))
    f_nu = max(f_nu, 1e-6)
    pka_penalty = -GAS_CONSTANT_KCAL * 310.15 * math.log(f_nu)
    # Reference baseline at pKa=7.8, pH=7.4 -> f_nu_ref ~ 0.285 -> penalty ~ 0.77 kcal/mol
    ref_penalty = -GAS_CONSTANT_KCAL * 310.15 * math.log(1.0 / (1.0 + 10.0 ** (7.8 - 7.4)))
    delta_pka = pka_penalty - ref_penalty

    return float(max(14.5, base_barrier + eht_shift + delta_pka))
