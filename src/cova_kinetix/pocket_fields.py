"""
3D protein binding-site grids, linearized Poisson-Boltzmann electrostatic solver,
induced-fit nucleophilic side-chain rotamer sampling, and Burgi-Dunitz covalent
pre-reaction pose alignment.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from cova_kinetix.fragment_graph import build_adjacency
from cova_kinetix.types import (
    BURGI_DUNITZ_IDEAL_DEG,
    ELEMENT_PARAMS,
    GAS_CONSTANT_KCAL,
    NUCLEOPHILE_PROFILES,
    WARHEAD_TEMPLATES,
    BindingPocket,
    DockingAlignment,
    FloatArray,
    MolecularGraph,
    PocketGrid,
    PocketResidue,
    SimulationDivergenceError,
)


def build_target_pocket(target_name: str) -> BindingPocket:
    """
    Construct a representative 3D protein binding pocket around the nucleophilic
    cysteine anchor for canonical covalent drug discovery targets:
    ``EGFR_C797S``, ``BTK_C481``, ``KRAS_G12C``, or ``MPRO_C145``.
    """
    builders = {
        "EGFR_C797S": _build_egfr_c797s_pocket,
        "BTK_C481": _build_btk_c481_pocket,
        "KRAS_G12C": _build_kras_g12c_pocket,
        "MPRO_C145": _build_mpro_c145_pocket,
    }
    key = target_name.strip().upper()
    if key not in builders:
        raise ValueError(f"Unknown target pocket '{target_name}'. Available: {sorted(builders.keys())}")
    return builders[key]()


def _build_egfr_c797s_pocket() -> BindingPocket:
    """EGFR T790M/L858R ATP-hinge pocket with Cys797 solvent-front nucleophile."""
    residues = [
        PocketResidue(
            "C797", "Cys", "SG", "S", (4.2, 0.8, 0.2), -0.35, hbd=True,
            is_nucleophile=True, chi1_pivot=(5.5, 1.4, -0.4), chi1_axis=(0.0, 0.3, 0.95),
        ),
        PocketResidue("M793_N", "Met", "N", "N", (-3.2, 0.6, 0.0), -0.35, hbd=True),
        PocketResidue("M793_O", "Met", "O", "O", (-3.0, -1.6, 0.2), -0.50, hba=True),
        PocketResidue("K745", "Lys", "NZ", "N", (-1.5, 3.8, 1.4), 0.75, hbd=True),
        PocketResidue("E762", "Glu", "OE1", "O", (-2.6, 4.4, -1.2), -0.75, hba=True),
        PocketResidue("D855", "Asp", "OD1", "O", (0.4, 3.2, -2.5), -0.65, hba=True),
        PocketResidue("M790", "Met", "CE", "C", (-4.4, 2.2, -1.4), 0.0, hydrophobic=True),
        PocketResidue("L718", "Leu", "CD1", "C", (1.2, -2.8, 2.4), 0.0, hydrophobic=True),
        PocketResidue("V726", "Val", "CG1", "C", (-0.8, 2.4, 2.8), 0.0, hydrophobic=True),
        PocketResidue("L844", "Leu", "CD2", "C", (0.6, -0.8, -3.2), 0.0, hydrophobic=True),
        PocketResidue("A743", "Ala", "CB", "C", (-2.2, -1.8, 2.2), 0.0, hydrophobic=True),
        PocketResidue("G796", "Gly", "CA", "C", (2.6, -2.6, -0.8), 0.05, hydrophobic=True),
    ]
    return BindingPocket("EGFR_C797S", "Cys", "C797", 8.1, residues, (0.0, 0.0, 0.0), box_half_width=9.5)


def _build_btk_c481_pocket() -> BindingPocket:
    """BTK kinase hinge + back-pocket subsite with Cys481 nucleophile."""
    residues = [
        PocketResidue(
            "C481", "Cys", "SG", "S", (4.0, 1.0, 0.4), -0.35, hbd=True,
            is_nucleophile=True, chi1_pivot=(5.3, 1.5, -0.3), chi1_axis=(0.1, 0.2, 0.97),
        ),
        PocketResidue("M477_N", "Met", "N", "N", (-3.1, 0.4, 0.2), -0.35, hbd=True),
        PocketResidue("E475_O", "Glu", "O", "O", (-3.3, -1.5, -0.1), -0.50, hba=True),
        PocketResidue("K430", "Lys", "NZ", "N", (-1.8, 3.6, 1.2), 0.75, hbd=True),
        PocketResidue("T474", "Thr", "OG1", "O", (-4.5, 2.0, -0.8), -0.45, hbd=True, hba=True),
        PocketResidue("D539", "Asp", "OD2", "O", (0.2, 3.1, -2.6), -0.65, hba=True),
        PocketResidue("L408", "Leu", "CD1", "C", (1.0, -2.6, 2.5), 0.0, hydrophobic=True),
        PocketResidue("Y476", "Tyr", "CZ", "C", (-1.8, -3.2, -1.8), 0.0, hydrophobic=True),
        PocketResidue("F540", "Phe", "CE1", "C", (-3.8, 3.4, -2.2), 0.0, hydrophobic=True),
        PocketResidue("V416", "Val", "CG2", "C", (-0.6, 2.3, 2.9), 0.0, hydrophobic=True),
        PocketResidue("L528", "Leu", "CD2", "C", (0.8, -0.9, -3.1), 0.0, hydrophobic=True),
    ]
    return BindingPocket("BTK_C481", "Cys", "C481", 7.9, residues, (0.0, 0.0, 0.0), box_half_width=9.5)


def _build_kras_g12c_pocket() -> BindingPocket:
    """KRAS G12C Switch-II pocket with GDP-adjacent Cys12 nucleophile."""
    residues = [
        PocketResidue(
            "C12", "Cys", "SG", "S", (3.9, 0.4, 0.3), -0.30, hbd=True,
            is_nucleophile=True, chi1_pivot=(5.2, 0.9, -0.3), chi1_axis=(0.0, -0.2, 0.98),
        ),
        PocketResidue("K16", "Lys", "NZ", "N", (2.6, 3.2, 1.8), 0.75, hbd=True),
        PocketResidue("E62", "Glu", "OE1", "O", (-2.2, 3.4, -1.5), -0.75, hba=True),
        PocketResidue("R68", "Arg", "NH1", "N", (-3.4, 1.8, 1.4), 0.80, hbd=True),
        PocketResidue("D69", "Asp", "OD1", "O", (-3.6, -1.2, -0.8), -0.70, hba=True),
        PocketResidue("H95", "His", "NE2", "N", (-1.6, -2.9, 1.6), 0.25, hbd=True, hba=True),
        PocketResidue("Y96", "Tyr", "OH", "O", (-2.4, -2.4, -2.2), -0.45, hbd=True, hba=True),
        PocketResidue("M72", "Met", "CE", "C", (-4.2, 0.2, -2.1), 0.0, hydrophobic=True),
        PocketResidue("V9", "Val", "CG1", "C", (0.5, -2.5, 2.5), 0.0, hydrophobic=True),
        PocketResidue("I100", "Ile", "CD1", "C", (-0.8, 2.1, -3.1), 0.0, hydrophobic=True),
    ]
    return BindingPocket("KRAS_G12C", "Cys", "C12", 8.2, residues, (0.0, 0.0, 0.0), box_half_width=9.5)


def _build_mpro_c145_pocket() -> BindingPocket:
    """SARS-CoV-2 Mpro catalytic dyad pocket (Cys145 thiolate + His41 general base)."""
    residues = [
        PocketResidue(
            "C145", "Cys", "SG", "S", (3.8, 0.6, 0.0), -0.55, hbd=True,
            is_nucleophile=True, chi1_pivot=(5.1, 1.2, -0.5), chi1_axis=(0.2, 0.1, 0.97),
        ),
        PocketResidue("H41", "His", "NE2", "N", (2.4, 3.3, 1.6), 0.45, hbd=True, hba=True),
        PocketResidue("G143_N", "Gly", "N", "N", (2.5, -1.6, 1.4), -0.35, hbd=True),  # oxyanion hole
        PocketResidue("E166_O", "Glu", "OE1", "O", (-3.1, 0.8, 0.3), -0.75, hba=True),
        PocketResidue("E166_N", "Glu", "N", "N", (-2.8, -1.5, -0.2), -0.35, hbd=True),
        PocketResidue("H163", "His", "NE2", "N", (-4.1, 2.2, 1.5), 0.25, hbd=True, hba=True),
        PocketResidue("Q189", "Gln", "OE1", "O", (-1.4, 3.4, -2.0), -0.45, hba=True),
        PocketResidue("M49", "Met", "CE", "C", (0.6, 2.4, 2.9), 0.0, hydrophobic=True),
        PocketResidue("M165", "Met", "CE", "C", (-1.5, -2.7, 2.2), 0.0, hydrophobic=True),
        PocketResidue("L167", "Leu", "CD1", "C", (-3.8, -1.9, -2.2), 0.0, hydrophobic=True),
    ]
    return BindingPocket("MPRO_C145", "Cys", "C145", 6.9, residues, (0.0, 0.0, 0.0), box_half_width=9.5)


def solve_poisson_boltzmann_grid(
    pocket: BindingPocket,
    grid_n: int = 23,
    ionic_strength_m: float = 0.15,
    eps_protein: float = 4.0,
    eps_solvent: float = 78.4,
    temperature_k: float = 310.15,
    max_iter: int = 220,
    tol: float = 1e-5,
) -> PocketGrid:
    """
    Solve the 3D linearized Poisson-Boltzmann equation on a cubic grid centered
    at ``pocket.box_center``:

    .. math::
        -\\nabla \\cdot (\\varepsilon(\\mathbf{r}) \\nabla \\phi(\\mathbf{r}))
        + \\varepsilon(\\mathbf{r}) \\kappa^2(\\mathbf{r}) \\phi(\\mathbf{r})
        = 4\\pi \\frac{e^2}{k_B T} \\rho(\\mathbf{r})

    Uses Cloud-In-Cell trilinear charge splatting, screened Debye-Huckel far-field
    Dirichlet boundary values, and vectorized Red-Black Successive Over-Relaxation.
    """
    if grid_n < 7 or grid_n % 2 == 0:
        raise ValueError("grid_n must be an odd integer >= 7.")

    half_w = pocket.box_half_width
    origin = np.asarray(pocket.box_center, dtype=np.float64) - half_w
    h = (2.0 * half_w) / float(grid_n - 1)
    coords_1d = [origin[d] + h * np.arange(grid_n, dtype=np.float64) for d in range(3)]
    gx, gy, gz = np.meshgrid(coords_1d[0], coords_1d[1], coords_1d[2], indexing="ij")

    # Bjerrum length factor in vacuum at T (e^2 / (4*pi*eps0*kB*T) in Angstroms)
    bjerrum_vac_a = 167101.0 / temperature_k  # ~ 538.8 A at 310.15 K
    kappa_bulk = math.sqrt(8.0 * math.pi * (bjerrum_vac_a / eps_solvent) * 6.02214e-4 * ionic_strength_m)

    eps_grid = np.full((grid_n, grid_n, grid_n), eps_solvent, dtype=np.float64)
    kappa2_grid = np.full((grid_n, grid_n, grid_n), kappa_bulk ** 2, dtype=np.float64)
    steric_grid = np.zeros((grid_n, grid_n, grid_n), dtype=np.float64)
    charge_density = np.zeros((grid_n, grid_n, grid_n), dtype=np.float64)

    for res in pocket.residues:
        rx, ry, rz = res.position
        vdw = ELEMENT_PARAMS.get(res.element, ELEMENT_PARAMS["C"]).vdw_radius
        r_probe = vdw + 1.1
        dist2 = (gx - rx) ** 2 + (gy - ry) ** 2 + (gz - rz) ** 2
        inside = dist2 <= r_probe ** 2
        eps_grid[inside] = eps_protein
        kappa2_grid[inside] = 0.0
        steric_grid += np.exp(-dist2 / (2.0 * (0.65 * vdw) ** 2))

        # Trilinear CIC splatting of partial charge onto 8 surrounding nodes
        fx = (rx - origin[0]) / h
        fy = (ry - origin[1]) / h
        fz = (rz - origin[2]) / h
        ix, iy, iz = int(math.floor(fx)), int(math.floor(fy)), int(math.floor(fz))
        if 0 <= ix < grid_n - 1 and 0 <= iy < grid_n - 1 and 0 <= iz < grid_n - 1:
            dx, dy, dz = fx - ix, fy - iy, fz - iz
            for ox in (0, 1):
                wx = (1.0 - dx) if ox == 0 else dx
                for oy in (0, 1):
                    wy = (1.0 - dy) if oy == 0 else dy
                    for oz in (0, 1):
                        wz = (1.0 - dz) if oz == 0 else dz
                        charge_density[ix + ox, iy + oy, iz + oz] += (res.partial_charge * wx * wy * wz) / (h ** 3)

    # Initialize potential with Debye-Huckel analytical estimate (also fixes Dirichlet boundaries)
    phi = np.zeros((grid_n, grid_n, grid_n), dtype=np.float64)
    for res in pocket.residues:
        if abs(res.partial_charge) < 1e-9:
            continue
        rx, ry, rz = res.position
        r = np.sqrt((gx - rx) ** 2 + (gy - ry) ** 2 + (gz - rz) ** 2 + (0.6 * h) ** 2)
        phi += (bjerrum_vac_a / eps_solvent) * res.partial_charge * np.exp(-kappa_bulk * r) / r

    rhs_inner = 4.0 * math.pi * bjerrum_vac_a * charge_density * (h ** 2)

    # Harmonic-mean face dielectric coefficients for flux conservation
    c_eps = eps_grid[1:-1, 1:-1, 1:-1]
    eps_xp = 2.0 * c_eps * eps_grid[2:, 1:-1, 1:-1] / (c_eps + eps_grid[2:, 1:-1, 1:-1])
    eps_xm = 2.0 * c_eps * eps_grid[:-2, 1:-1, 1:-1] / (c_eps + eps_grid[:-2, 1:-1, 1:-1])
    eps_yp = 2.0 * c_eps * eps_grid[1:-1, 2:, 1:-1] / (c_eps + eps_grid[1:-1, 2:, 1:-1])
    eps_ym = 2.0 * c_eps * eps_grid[1:-1, :-2, 1:-1] / (c_eps + eps_grid[1:-1, :-2, 1:-1])
    eps_zp = 2.0 * c_eps * eps_grid[1:-1, 1:-1, 2:] / (c_eps + eps_grid[1:-1, 1:-1, 2:])
    eps_zm = 2.0 * c_eps * eps_grid[1:-1, 1:-1, :-2] / (c_eps + eps_grid[1:-1, 1:-1, :-2])

    denom = (
        eps_xp + eps_xm + eps_yp + eps_ym + eps_zp + eps_zm
        + eps_grid[1:-1, 1:-1, 1:-1] * kappa2_grid[1:-1, 1:-1, 1:-1] * (h ** 2)
    )
    rhs_slice = rhs_inner[1:-1, 1:-1, 1:-1]

    # Checkerboard Red-Black masks on interior slice
    idx = np.arange(1, grid_n - 1)
    ii, jj, kk = np.meshgrid(idx, idx, idx, indexing="ij")
    red_mask = ((ii + jj + kk) % 2) == 0
    black_mask = ~red_mask

    omega = 1.68  # optimal SOR factor for 3D elliptic grid
    residual = 1.0
    iters_done = 0

    for it in range(1, max_iter + 1):
        max_diff = 0.0
        for mask in (red_mask, black_mask):
            nbr_sum = (
                eps_xp * phi[2:, 1:-1, 1:-1]
                + eps_xm * phi[:-2, 1:-1, 1:-1]
                + eps_yp * phi[1:-1, 2:, 1:-1]
                + eps_ym * phi[1:-1, :-2, 1:-1]
                + eps_zp * phi[1:-1, 1:-1, 2:]
                + eps_zm * phi[1:-1, 1:-1, :-2]
            )
            gs_target = (nbr_sum + rhs_slice) / denom
            interior = phi[1:-1, 1:-1, 1:-1]
            delta = omega * (gs_target - interior)
            step_diff = float(np.max(np.abs(delta[mask])))
            max_diff = max(max_diff, step_diff)
            interior[mask] += delta[mask]
        iters_done = it
        residual = max_diff
        if not np.isfinite(residual):
            raise SimulationDivergenceError("Poisson-Boltzmann SOR solver diverged to NaN/Inf.")
        if residual < tol:
            break

    return PocketGrid(
        origin=origin,
        spacing=h,
        shape=(grid_n, grid_n, grid_n),
        electrostatic_potential_kt=phi,
        dielectric_grid=eps_grid,
        steric_occupancy=steric_grid,
        pb_iterations=iters_done,
        pb_residual=residual,
    )


def _rodrigues_rotate_point(
    point: FloatArray,
    pivot: FloatArray,
    axis: FloatArray,
    angle_deg: float,
) -> FloatArray:
    """Rotate a 3D point around an axis passing through ``pivot`` by ``angle_deg``."""
    theta = math.radians(angle_deg)
    k = axis / max(float(np.linalg.norm(axis)), 1e-12)
    v = point - pivot
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    v_rot = v * cos_t + np.cross(k, v) * sin_t + k * float(np.dot(k, v)) * (1.0 - cos_t)
    return pivot + v_rot


def _euler_matrix(alpha: float, beta: float, gamma: float) -> FloatArray:
    """Build 3x3 ZYX rotation matrix from Euler angles (radians)."""
    ca, sa = math.cos(alpha), math.sin(alpha)
    cb, sb = math.cos(beta), math.sin(beta)
    cg, sg = math.cos(gamma), math.sin(gamma)
    rz = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cb, 0.0, sb], [0.0, 1.0, 0.0], [-sb, 0.0, cb]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cg, -sg], [0.0, sg, cg]])
    return np.asarray(rz @ ry @ rx, dtype=np.float64)


def _interpolate_pb_potential(grid: PocketGrid, coords: FloatArray) -> FloatArray:
    """Trilinear interpolation of the Poisson-Boltzmann potential at arbitrary 3D points."""
    nx, ny, nz = grid.shape
    h = grid.spacing
    rel = (coords - grid.origin[None, :]) / h
    ix = np.clip(np.floor(rel[:, 0]).astype(int), 0, nx - 2)
    iy = np.clip(np.floor(rel[:, 1]).astype(int), 0, ny - 2)
    iz = np.clip(np.floor(rel[:, 2]).astype(int), 0, nz - 2)
    dx = np.clip(rel[:, 0] - ix, 0.0, 1.0)
    dy = np.clip(rel[:, 1] - iy, 0.0, 1.0)
    dz = np.clip(rel[:, 2] - iz, 0.0, 1.0)

    phi = grid.electrostatic_potential_kt
    out = np.zeros(coords.shape[0], dtype=np.float64)
    for ox in (0, 1):
        wx = (1.0 - dx) if ox == 0 else dx
        for oy in (0, 1):
            wy = (1.0 - dy) if oy == 0 else dy
            for oz in (0, 1):
                wz = (1.0 - dz) if oz == 0 else dz
                out += wx * wy * wz * phi[ix + ox, iy + oy, iz + oz]
    return out


def align_covalent_ligand(
    mol: MolecularGraph,
    pocket: BindingPocket,
    pb_grid: Optional[PocketGrid] = None,
    mulliken_charges: Optional[FloatArray] = None,
    desolvation_weight: float = 0.018,
    temperature_k: float = 310.15,
) -> DockingAlignment:
    """
    Perform induced-fit nucleophilic side-chain rotamer sampling and rigid-body
    pose optimization inside ``pocket``, evaluating Poisson-Boltzmann
    electrostatics, soft-core sterics, H-bond complementarity, hydrophobic burial,
    and Burgi-Dunitz covalent pre-reaction alignment.
    """
    if pb_grid is None:
        pb_grid = solve_poisson_boltzmann_grid(pocket, grid_n=19)

    if mol.coords_3d is None:
        raise ValueError("MolecularGraph must have 3D coordinates before docking alignment.")

    n_heavy = len(mol.atoms)
    base_coords = np.asarray(mol.coords_3d[:n_heavy], dtype=np.float64)
    base_coords = base_coords - base_coords.mean(axis=0, keepdims=True)

    if mulliken_charges is not None and len(mulliken_charges) >= n_heavy:
        atom_q = np.asarray(mulliken_charges[:n_heavy], dtype=np.float64)
    else:
        atom_q = np.array(
            [0.15 if a.symbol == "C" else (-0.25 if a.symbol in ("N", "O") else 0.0) for a in mol.atoms],
            dtype=np.float64,
        )

    adj = build_adjacency(mol)
    lig_hbd = np.array([a.symbol in ("N", "O") and a.implicit_h > 0 for a in mol.atoms], dtype=bool)
    lig_hba = np.array([a.symbol in ("N", "O") for a in mol.atoms], dtype=bool)
    lig_hydro = np.array(
        [a.symbol in ("C", "S", "Cl", "Br", "F") and a.formal_charge == 0 for a in mol.atoms],
        dtype=bool,
    )

    # Find electrophilic atom & adjacent reference bond partner for Burgi-Dunitz angle
    elec_idx = mol.electrophile_atom_idx
    bd_partner_idx: Optional[int] = None
    if elec_idx is not None and adj[elec_idx]:
        # Prefer carbonyl or double-bonded partner to define electrophile axis
        bd_partner_idx = max(adj[elec_idx], key=lambda item: item[1].order)[0]

    warhead_tpl = WARHEAD_TEMPLATES.get(mol.warhead_class or "")
    target_bd_deg = warhead_tpl.burgi_dunitz_target_deg if warhead_tpl is not None else BURGI_DUNITZ_IDEAL_DEG

    nu_res = next((r for r in pocket.residues if r.is_nucleophile), pocket.residues[0])
    nu_profile = NUCLEOPHILE_PROFILES.get(pocket.nucleophile_residue, NUCLEOPHILE_PROFILES["Cys"])
    other_residues = [r for r in pocket.residues if not r.is_nucleophile]
    other_pos = np.array([r.position for r in other_residues], dtype=np.float64)
    other_vdw = np.array(
        [ELEMENT_PARAMS.get(r.element, ELEMENT_PARAMS["C"]).vdw_radius for r in other_residues],
        dtype=np.float64,
    )
    other_hbd = np.array([r.hbd for r in other_residues], dtype=bool)
    other_hba = np.array([r.hba for r in other_residues], dtype=bool)
    other_hydro = np.array([r.hydrophobic for r in other_residues], dtype=bool)
    lig_vdw = np.array(
        [ELEMENT_PARAMS.get(a.symbol, ELEMENT_PARAMS["C"]).vdw_radius for a in mol.atoms],
        dtype=np.float64,
    )

    kt_kcal = GAS_CONSTANT_KCAL * temperature_k
    best_alignment: Optional[DockingAlignment] = None

    # Sample each discrete chi1 rotamer of the target nucleophile
    base_nu_pos = np.asarray(nu_res.position, dtype=np.float64)
    pivot = np.asarray(
        nu_res.chi1_pivot if nu_res.chi1_pivot is not None else (base_nu_pos + np.array([1.2, 0.5, 0.0])),
        dtype=np.float64,
    )
    axis = np.asarray(nu_res.chi1_axis if nu_res.chi1_axis is not None else np.array([0.0, 0.0, 1.0]), dtype=np.float64)

    for rot_idx, (chi1_deg, rot_strain) in enumerate(
        zip(nu_profile.chi1_rotamers_deg, nu_profile.rotamer_strain_kcal)
    ):
        rot_nu_pos = _rodrigues_rotate_point(base_nu_pos, pivot, axis, chi1_deg - nu_profile.chi1_rotamers_deg[0])

        def _score_pose(params: FloatArray) -> Tuple[float, Dict[str, float], FloatArray]:
            tx, ty, tz, a_ang, b_ang, g_ang = params
            rot_mat = _euler_matrix(a_ang, b_ang, g_ang)
            xyz = base_coords @ rot_mat.T + np.array([tx, ty, tz], dtype=np.float64)

            # 1. Soft-core steric repulsion & attractive vdW dispersion
            diff_mat = xyz[:, None, :] - other_pos[None, :, :]
            d_mat = np.sqrt(np.sum(diff_mat ** 2, axis=2) + 1e-8)
            sum_vdw = lig_vdw[:, None] + other_vdw[None, :]
            overlap = np.maximum(0.0, sum_vdw - 0.55 - d_mat)
            steric_rep = 2.8 * float(np.sum(overlap ** 2))
            vdw_attr = -0.14 * float(np.sum(np.exp(-((d_mat - (sum_vdw + 0.35)) / 0.85) ** 2)))

            # 2. Directional H-bond complementarity
            hb_score = 0.0
            if np.any(lig_hbd) and np.any(other_hba):
                d_da = d_mat[np.ix_(lig_hbd, other_hba)]
                hb_score -= 1.35 * float(np.sum(np.exp(-((d_da - 2.85) / 0.55) ** 2)))
            if np.any(lig_hba) and np.any(other_hbd):
                d_ad = d_mat[np.ix_(lig_hba, other_hbd)]
                hb_score -= 1.35 * float(np.sum(np.exp(-((d_ad - 2.85) / 0.55) ** 2)))

            # 3. Hydrophobic contact burial
            hydro_score = 0.0
            if np.any(lig_hydro) and np.any(other_hydro):
                d_hh = d_mat[np.ix_(lig_hydro, other_hydro)]
                hydro_score -= 0.16 * float(np.sum(np.exp(-((d_hh - 3.9) / 0.9) ** 2)))

            # 4. Linearized Poisson-Boltzmann electrostatic score (kcal/mol)
            phi_vals = _interpolate_pb_potential(pb_grid, xyz)
            elec_kcal = 0.35 * kt_kcal * float(np.sum(atom_q * phi_vals))
            elec_kcal = float(np.clip(elec_kcal, -4.5, 4.5))

            # 5. Desolvation burial penalty (proportional to TPSA)
            desolv_kcal = desolvation_weight * mol.tpsa

            # 6. Pre-reaction Burgi-Dunitz covalent trajectory alignment
            if elec_idx is not None:
                vec_nu_c = xyz[elec_idx] - rot_nu_pos
                nu_dist = float(np.linalg.norm(vec_nu_c))
                traj_pen = 1.6 * (nu_dist - 3.35) ** 2
                if bd_partner_idx is not None:
                    vec_c_x = xyz[bd_partner_idx] - xyz[elec_idx]
                    n_c_x = float(np.linalg.norm(vec_c_x))
                    cos_bd = float(np.dot(-vec_nu_c, vec_c_x)) / max(nu_dist * n_c_x, 1e-8)
                    cos_bd = float(np.clip(cos_bd, -1.0, 1.0))
                    bd_deg = math.degrees(math.acos(cos_bd))
                    traj_pen += 0.0045 * (bd_deg - target_bd_deg) ** 2
                else:
                    bd_deg = target_bd_deg
            else:
                nu_dist = 8.5
                bd_deg = 0.0
                traj_pen = 0.0

            # 6. Rigid-body translational/rotational entropy offset (+2.4 kcal/mol)
            dg_noncov = 2.4 + vdw_attr + steric_rep + hb_score + hydro_score + elec_kcal + desolv_kcal
            # Cap extreme favorable binding energy to realistic drug-like range [-13.5, +4.0]
            dg_noncov = float(np.clip(dg_noncov, -13.5, 4.0))
            total_obj = dg_noncov + rot_strain + traj_pen
            breakdown = {
                "dg_noncov": dg_noncov,
                "elec": elec_kcal,
                "steric": steric_rep + vdw_attr,
                "hbond": hb_score,
                "desolv": desolv_kcal,
                "hydro": hydro_score,
                "nu_dist": nu_dist,
                "bd_deg": bd_deg,
                "traj_pen": traj_pen,
                "total": total_obj,
            }
            return total_obj, breakdown, xyz

        # Initial guess shifts electrophile near rot_nu_pos - (2.8, 0, 0)
        init_shift = np.array(pocket.box_center, dtype=np.float64)
        if elec_idx is not None:
            init_shift = rot_nu_pos - np.array([3.2, 0.2, 0.0]) - base_coords[elec_idx]
        x0 = np.array([init_shift[0], init_shift[1], init_shift[2], 0.0, 0.0, 0.0], dtype=np.float64)

        opt = minimize(
            lambda p: _score_pose(p)[0],
            x0,
            method="Powell",
            options={"maxiter": 65, "xtol": 1e-2, "ftol": 1e-2},
        )
        _, stats, final_xyz = _score_pose(opt.x)
        dg_eff = stats["dg_noncov"] + rot_strain + 0.35 * stats["traj_pen"]
        ki_m = math.exp(dg_eff / kt_kcal)
        ki_uM = float(np.clip(ki_m * 1e6, 1e-4, 1e4))

        cand = DockingAlignment(
            ligand_name=mol.name,
            target_name=pocket.target_name,
            rotamer_idx=rot_idx,
            chi1_deg=chi1_deg,
            rotamer_strain_kcal=rot_strain,
            noncovalent_dg_kcal=stats["dg_noncov"],
            electrostatic_kcal=stats["elec"],
            steric_kcal=stats["steric"],
            hbond_kcal=stats["hbond"],
            desolvation_kcal=stats["desolv"],
            hydrophobic_kcal=stats["hydro"],
            nucleophile_distance_a=stats["nu_dist"],
            burgi_dunitz_angle_deg=stats["bd_deg"],
            trajectory_penalty_kcal=stats["traj_pen"],
            effective_ki_uM=ki_uM,
            transformed_coords_3d=final_xyz,
        )
        if best_alignment is None or cand.effective_ki_uM < best_alignment.effective_ki_uM:
            best_alignment = cand

    assert best_alignment is not None
    return best_alignment
