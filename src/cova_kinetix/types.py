"""
Domain constants, biophysical parameter tables, exceptions, and typed dataclasses
for CovaKinetix.

All physical units are documented explicitly:
- Lengths: Angstroms (A) unless stated as Bohr (a0).
- Energies: kcal/mol for molecular mechanics, free energies, and kinetic barriers;
  electron-volts (eV) for Extended Huckel molecular orbital eigenvalues.
- Concentrations: Molar (M) or micromolar (uM) as noted.
- Rates: s^-1 for unimolecular inactivation/dissociation, M^-1 s^-1 for bimolecular
  second-order covalent efficiency (k_inact / K_i).
- Permeability: cm/s for transmembrane effective permeability P_eff.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

# Fundamental physical & conversion constants
BOHR_TO_ANGSTROM: float = 0.529177210903
ANGSTROM_TO_BOHR: float = 1.0 / BOHR_TO_ANGSTROM
GAS_CONSTANT_KCAL: float = 0.00198720425864083  # kcal / (mol * K)
EV_TO_KCAL_MOL: float = 23.06054783061903
KCAL_TO_KJ: float = 4.184
AVOGADRO: float = 6.02214076e23
ELEMENTARY_CHARGE_COULOMB: float = 1.602176634e-19
WOLFSBERG_HELMHOLZ_K: float = 1.75
BURGI_DUNITZ_IDEAL_DEG: float = 107.0


class CovaKinetixError(Exception):
    """Base exception for invalid chemistry, numerical divergence, or input errors."""


class SmilesParseError(CovaKinetixError):
    """Raised when a SMILES string violates lexical, ring-closure, or valence rules."""


class QuantumConvergenceError(CovaKinetixError):
    """Raised when the Extended Huckel overlap matrix becomes singular."""


class SimulationDivergenceError(CovaKinetixError):
    """Raised when a Poisson-Boltzmann, ODE, or metadynamics solver diverges."""


@dataclass(frozen=True)
class ElementParams:
    """
    Atomic, valence, and Slater-type orbital Extended Huckel parameters.

    Attributes
    ----------
    symbol : str
        Element symbol (e.g., 'C', 'N', 'O', 'S').
    atomic_number : int
        Atomic number Z.
    mass : float
        Standard atomic weight (g/mol).
    covalent_radius : float
        Single-bond covalent radius (Angstrom).
    vdw_radius : float
        Bondi van der Waals radius (Angstrom).
    electronegativity : float
        Pauling electronegativity scale.
    valence_electrons : int
        Number of neutral valence electrons.
    principal_n : int
        Valence shell principal quantum number used for STO basis functions.
    zeta_s : float
        Slater orbital exponent for valence s orbital (Bohr^-1).
    zeta_p : Optional[float]
        Slater orbital exponent for valence p orbitals (Bohr^-1), None for H.
    vsip_s : float
        Valence state ionization potential for s orbital (eV, positive magnitude).
    vsip_p : Optional[float]
        Valence state ionization potential for p orbitals (eV, positive magnitude).
    default_valences : Tuple[int, ...]
        Allowed neutral bond-order sums used for implicit hydrogen filling.
    """

    symbol: str
    atomic_number: int
    mass: float
    covalent_radius: float
    vdw_radius: float
    electronegativity: float
    valence_electrons: int
    principal_n: int
    zeta_s: float
    zeta_p: Optional[float]
    vsip_s: float
    vsip_p: Optional[float]
    default_valences: Tuple[int, ...]


# Extended Huckel Hoffmann (1963) & Alvarez standard STO valence parameters
ELEMENT_PARAMS: Dict[str, ElementParams] = {
    "H": ElementParams("H", 1, 1.008, 0.31, 1.20, 2.20, 1, 1, 1.300, None, 13.60, None, (1,)),
    "B": ElementParams("B", 5, 10.81, 0.84, 1.92, 2.04, 3, 2, 1.300, 1.300, 15.20, 8.50, (3,)),
    "C": ElementParams("C", 6, 12.011, 0.76, 1.70, 2.55, 4, 2, 1.625, 1.625, 21.40, 11.40, (4,)),
    "N": ElementParams("N", 7, 14.007, 0.71, 1.55, 3.04, 5, 2, 1.950, 1.950, 26.00, 13.40, (3, 5)),
    "O": ElementParams("O", 8, 15.999, 0.66, 1.52, 3.44, 6, 2, 2.275, 2.275, 32.30, 14.80, (2,)),
    "F": ElementParams("F", 9, 18.998, 0.57, 1.47, 3.98, 7, 2, 2.425, 2.425, 40.00, 18.10, (1,)),
    "P": ElementParams("P", 15, 30.974, 1.07, 1.80, 2.19, 5, 3, 1.600, 1.600, 18.60, 14.00, (3, 5)),
    "S": ElementParams("S", 16, 32.06, 1.05, 1.80, 2.58, 6, 3, 1.817, 1.817, 20.00, 13.30, (2, 4, 6)),
    "Cl": ElementParams("Cl", 17, 35.45, 1.02, 1.75, 3.16, 7, 3, 2.033, 2.033, 30.00, 15.00, (1,)),
    "Br": ElementParams("Br", 35, 79.904, 1.20, 1.85, 2.96, 7, 3, 2.200, 2.150, 27.00, 13.10, (1,)),
    "*": ElementParams("*", 0, 0.0, 0.70, 1.40, 2.50, 1, 1, 1.200, None, 13.60, None, (1,)),
}


@dataclass(frozen=True)
class WarheadTemplate:
    """
    Covalent reactive warhead specification for structure-based design.

    Attributes
    ----------
    name : str
        Identifier of the warhead class.
    smiles_fragment : str
        Attachment-ready fragment SMILES with ``[1*]`` marking the scaffold link.
    electrophile_role : str
        Chemical mechanism class (e.g., 'michael_acceptor', 'sn2_halide', 'reversible_nitrile').
    target_residues : Tuple[str, ...]
        Compatible nucleophilic amino-acid side chains.
    intrinsic_barrier_kcal : float
        Baseline activation free-energy barrier Delta G_double_dagger (kcal/mol).
    reversible : bool
        Whether the covalent adduct exhibits appreciable microscopic reversibility.
    k_rev_base_s : float
        Baseline reverse hydrolysis rate k_rev (s^-1) if reversible else 0.0.
    gsh_t_half_min_ref : float
        Reference glutathione half-life in minutes at pH 7.4, 37 C for unsubstituted motif.
    burgi_dunitz_target_deg : float
        Optimal nucleophilic trajectory angle (degrees).
    """

    name: str
    smiles_fragment: str
    electrophile_role: str
    target_residues: Tuple[str, ...]
    intrinsic_barrier_kcal: float
    reversible: bool
    k_rev_base_s: float
    gsh_t_half_min_ref: float
    burgi_dunitz_target_deg: float = 107.0


WARHEAD_TEMPLATES: Dict[str, WarheadTemplate] = {
    "acrylamide": WarheadTemplate(
        name="acrylamide",
        smiles_fragment="[1*]NC(=O)C=C",
        electrophile_role="michael_acceptor",
        target_residues=("Cys",),
        intrinsic_barrier_kcal=20.8,
        reversible=False,
        k_rev_base_s=0.0,
        gsh_t_half_min_ref=180.0,
        burgi_dunitz_target_deg=107.0,
    ),
    "butynamide": WarheadTemplate(
        name="butynamide",
        smiles_fragment="[1*]NC(=O)C#CC",
        electrophile_role="alkyne_michael_acceptor",
        target_residues=("Cys",),
        intrinsic_barrier_kcal=20.6,
        reversible=False,
        k_rev_base_s=0.0,
        gsh_t_half_min_ref=240.0,
        burgi_dunitz_target_deg=110.0,
    ),
    "dimethylaminomethyl_acrylamide": WarheadTemplate(
        name="dimethylaminomethyl_acrylamide",
        smiles_fragment="[1*]NC(=O)C=CCN(C)C",
        electrophile_role="michael_acceptor_tuned",
        target_residues=("Cys",),
        intrinsic_barrier_kcal=20.2,
        reversible=False,
        k_rev_base_s=0.0,
        gsh_t_half_min_ref=115.0,
        burgi_dunitz_target_deg=107.0,
    ),
    "cyanoacrylamide": WarheadTemplate(
        name="cyanoacrylamide",
        smiles_fragment="[1*]NC(=O)C(C#N)=CC",
        electrophile_role="reversible_michael",
        target_residues=("Cys",),
        intrinsic_barrier_kcal=19.6,
        reversible=True,
        k_rev_base_s=2.5e-4,
        gsh_t_half_min_ref=45.0,
        burgi_dunitz_target_deg=106.0,
    ),
    "chloroacetamide": WarheadTemplate(
        name="chloroacetamide",
        smiles_fragment="[1*]NC(=O)CCl",
        electrophile_role="sn2_halide",
        target_residues=("Cys",),
        intrinsic_barrier_kcal=18.2,
        reversible=False,
        k_rev_base_s=0.0,
        gsh_t_half_min_ref=18.0,
        burgi_dunitz_target_deg=175.0,
    ),
    "vinyl_sulfone": WarheadTemplate(
        name="vinyl_sulfone",
        smiles_fragment="[1*]S(=O)(=O)C=C",
        electrophile_role="michael_acceptor",
        target_residues=("Cys", "Lys"),
        intrinsic_barrier_kcal=19.3,
        reversible=False,
        k_rev_base_s=0.0,
        gsh_t_half_min_ref=35.0,
        burgi_dunitz_target_deg=108.0,
    ),
    "nitrile": WarheadTemplate(
        name="nitrile",
        smiles_fragment="[1*]C#N",
        electrophile_role="reversible_nitrile",
        target_residues=("Cys", "Ser"),
        intrinsic_barrier_kcal=20.5,
        reversible=True,
        k_rev_base_s=6.0e-4,
        gsh_t_half_min_ref=720.0,
        burgi_dunitz_target_deg=105.0,
    ),
    "alpha_ketoamide": WarheadTemplate(
        name="alpha_ketoamide",
        smiles_fragment="[1*]C(=O)C(=O)NC",
        electrophile_role="reversible_hemithioacetal",
        target_residues=("Cys", "Ser"),
        intrinsic_barrier_kcal=19.9,
        reversible=True,
        k_rev_base_s=4.2e-4,
        gsh_t_half_min_ref=480.0,
        burgi_dunitz_target_deg=107.0,
    ),
}


@dataclass(frozen=True)
class NucleophileProfile:
    """
    Target amino-acid nucleophile thermodynamic and rotamer parameters.

    Attributes
    ----------
    residue_name : str
        Three-letter amino acid code ('Cys', 'Lys', 'Ser', 'Thr', 'Tyr').
    nucleophile_atom : str
        Reactive heteroatom symbol ('S', 'N', 'O').
    default_pka : float
        Typical active-site micro-pKa.
    intrinsic_nucleophilicity : float
        Relative Brønsted nucleophilicity multiplier in active ionized state.
    chi1_rotamers_deg : Tuple[float, ...]
        Standard Dunbrack chi1 staggered rotamer angles (degrees).
    rotamer_strain_kcal : Tuple[float, ...]
        Baseline conformational strain penalties (kcal/mol) for each chi1 rotamer.
    """

    residue_name: str
    nucleophile_atom: str
    default_pka: float
    intrinsic_nucleophilicity: float
    chi1_rotamers_deg: Tuple[float, ...]
    rotamer_strain_kcal: Tuple[float, ...]


NUCLEOPHILE_PROFILES: Dict[str, NucleophileProfile] = {
    "Cys": NucleophileProfile("Cys", "S", 7.8, 1.00, (-60.0, 180.0, 60.0), (0.0, 0.55, 1.10)),
    "Lys": NucleophileProfile("Lys", "N", 9.8, 0.42, (-60.0, 180.0, 60.0), (0.0, 0.45, 0.95)),
    "Ser": NucleophileProfile("Ser", "O", 11.8, 0.25, (60.0, -60.0, 180.0), (0.0, 0.35, 0.70)),
    "Thr": NucleophileProfile("Thr", "O", 12.2, 0.20, (-60.0, 60.0, 180.0), (0.0, 0.50, 0.90)),
    "Tyr": NucleophileProfile("Tyr", "O", 9.9, 0.32, (-60.0, 180.0, 60.0), (0.0, 0.60, 1.40)),
}


@dataclass
class AtomRecord:
    """Single atom node inside a MolecularGraph."""

    index: int
    symbol: str
    formal_charge: int = 0
    aromatic: bool = False
    implicit_h: int = 0
    isotope: Optional[int] = None
    dummy_label: Optional[int] = None
    in_ring: bool = False
    hybridization: str = "sp3"
    coords: Optional[Tuple[float, float, float]] = None


@dataclass
class BondRecord:
    """Undirected covalent bond edge inside a MolecularGraph."""

    atom_a: int
    atom_b: int
    order: float = 1.0
    aromatic: bool = False
    in_ring: bool = False


@dataclass
class MolecularGraph:
    """
    Pure-Python molecular graph with 2D topology, physicochemical properties, and 3D coords.
    """

    name: str
    smiles: str
    atoms: List[AtomRecord]
    bonds: List[BondRecord]
    rings: List[Tuple[int, ...]] = field(default_factory=list)
    coords_3d: Optional[FloatArray] = None
    molecular_weight: float = 0.0
    logp: float = 0.0
    tpsa: float = 0.0
    hbd: int = 0
    hba: int = 0
    rotatable_bonds: int = 0
    formal_charge: int = 0
    sa_score: float = 1.0
    qed: float = 0.5
    pka_acid: Optional[float] = None
    pka_base: Optional[float] = None
    warhead_class: Optional[str] = None
    electrophile_atom_idx: Optional[int] = None


@dataclass
class FragmentPiece:
    """BRICS-cleaved molecular fragment bearing exit-vector dummy labels."""

    smiles: str
    heavy_atom_count: int
    attachment_labels: Tuple[int, ...]
    ring_count: int
    role: str = "scaffold_fragment"


@dataclass
class EHTResult:
    """
    Extended Huckel Theory quantum chemistry and condensed Fukui reactivity output.
    """

    smiles: str
    n_atoms_total: int
    n_orbitals: int
    n_valence_electrons: int
    orbital_energies_ev: FloatArray
    homo_index: int
    lumo_index: int
    homo_ev: float
    lumo_ev: float
    homo_lumo_gap_ev: float
    chemical_potential_ev: float
    chemical_hardness_ev: float
    electrophilicity_omega_ev: float
    mulliken_charges: FloatArray
    lowdin_charges: FloatArray
    mayer_bond_orders: FloatArray
    fukui_plus: FloatArray
    fukui_minus: FloatArray
    fukui_radical: FloatArray
    dual_descriptor: FloatArray
    top_electrophile_heavy_idx: int
    max_fukui_plus_heavy: float
    predicted_gsh_t_half_min: float
    hyper_reactive_flag: bool
    trace_ps_electrons: float


@dataclass
class PocketResidue:
    """Single coarse-atom or functional-group residue inside a protein binding pocket."""

    residue_id: str
    residue_name: str
    atom_name: str
    element: str
    position: Tuple[float, float, float]
    partial_charge: float
    hbd: bool = False
    hba: bool = False
    hydrophobic: bool = False
    is_nucleophile: bool = False
    chi1_pivot: Optional[Tuple[float, float, float]] = None
    chi1_axis: Optional[Tuple[float, float, float]] = None


@dataclass
class BindingPocket:
    """3D protein binding site definition around a target nucleophilic residue."""

    target_name: str
    nucleophile_residue: str
    nucleophile_id: str
    nucleophile_pka: float
    residues: List[PocketResidue]
    box_center: Tuple[float, float, float]
    box_half_width: float = 10.0
    physiological_ph: float = 7.4


@dataclass
class PocketGrid:
    """3D discretized Poisson-Boltzmann electrostatic and steric potential grid."""

    origin: FloatArray
    spacing: float
    shape: Tuple[int, int, int]
    electrostatic_potential_kt: FloatArray
    dielectric_grid: FloatArray
    steric_occupancy: FloatArray
    pb_iterations: int
    pb_residual: float


@dataclass
class DockingAlignment:
    """Induced-fit ligand placement and Burgi-Dunitz covalent pre-reaction alignment."""

    ligand_name: str
    target_name: str
    rotamer_idx: int
    chi1_deg: float
    rotamer_strain_kcal: float
    noncovalent_dg_kcal: float
    electrostatic_kcal: float
    steric_kcal: float
    hbond_kcal: float
    desolvation_kcal: float
    hydrophobic_kcal: float
    nucleophile_distance_a: float
    burgi_dunitz_angle_deg: float
    trajectory_penalty_kcal: float
    effective_ki_uM: float
    transformed_coords_3d: FloatArray


@dataclass
class KineticTrace:
    """Two-step covalent binding and cellular target-turnover PK/PD trajectory."""

    times_h: FloatArray
    free_target_frac: FloatArray
    noncovalent_ei_frac: FloatArray
    covalent_adduct_frac: FloatArray
    total_occupancy_frac: FloatArray
    plasma_conc_uM: FloatArray
    intracellular_conc_uM: FloatArray
    gsh_adduct_uM: FloatArray
    ki_uM: float
    k_inact_per_s: float
    k_rev_per_s: float
    k_inact_over_ki: float
    occupancy_auc_h: float
    peak_occupancy: float
    trough_occupancy: float
    ic50_1h_uM: float
    ic50_24h_uM: float
    mass_balance_error: float


@dataclass
class MetadynamicsResult:
    """Well-Tempered and Infrequent Metadynamics free-energy surface and unbinding kinetics."""

    cv1_grid: FloatArray
    cv2_grid: FloatArray
    fes_2d_kcal: FloatArray
    fes_1d_cv1_kcal: FloatArray
    bias_potential_2d_kcal: FloatArray
    sample_trajectory_s1: FloatArray
    sample_trajectory_s2: FloatArray
    delta_g_bind_kcal: float
    barrier_off_kcal: float
    barrier_on_kcal: float
    unbiased_k_off_per_s: float
    mean_residence_time_s: float
    rescaled_unbinding_times_s: FloatArray
    acceleration_factors: FloatArray
    ks_statistic: float
    ks_pvalue: float
    poisson_certified: bool


@dataclass
class PermeationProfile:
    """Inhomogeneous Solubility-Diffusion (ISD) transmembrane permeability analysis."""

    z_grid_a: FloatArray
    delta_g_z_kcal: FloatArray
    diffusivity_z_cm2_s: FloatArray
    resistance_integrand: FloatArray
    neutral_fraction: float
    p_eff_cm_s: float
    log10_p_eff: float
    efflux_ratio: float
    p_app_apical_to_basolateral: float
    k_puu_intracellular: float
    barrier_peak_kcal: float
    permeability_class: str


@dataclass
class EvolutionCandidate:
    """Multi-objective de novo covalent fragment evolution candidate."""

    smiles: str
    warhead_name: str
    scaffold_smiles: str
    noncovalent_dg_kcal: float
    k_inact_over_ki: float
    log10_p_eff: float
    fukui_plus_max: float
    gsh_t_half_min: float
    sa_score: float
    pareto_rank: int
    crowding_distance: float


@dataclass
class CalibrationReport:
    """Model calibration, bootstrap confidence intervals, and Saltelli Sobol sensitivity."""

    fitted_params: Dict[str, float]
    initial_rmse: float
    calibrated_rmse: float
    r2_score: float
    bootstrap_ci_95: Dict[str, Tuple[float, float]]
    sobol_first_order: Dict[str, float]
    sobol_total_order: Dict[str, float]


@dataclass
class BenchmarkReport:
    """Multi-compound covalent screening and kinetic ranking validation report."""

    target_name: str
    n_compounds: int
    n_actives: int
    roc_auc: float
    bedroc_20: float
    ef_10_pct: float
    spearman_rho_kinact_ki: float
    log_kinact_ki_rmse: float
    records: List[ScreeningRecord]


@dataclass
class ScreeningRecord:
    """Unified screening score record for a single compound against a target pocket."""

    name: str
    smiles: str
    warhead_class: str
    is_active_ref: bool
    ref_log_kinact_ki: Optional[float]
    predicted_ki_uM: float
    predicted_k_inact_per_s: float
    predicted_log_kinact_ki: float
    predicted_fukui_plus: float
    predicted_gsh_t_half_min: float
    predicted_log10_p_eff: float
    occupancy_24h: float
    burgi_dunitz_deg: float
    nucleophile_dist_a: float
    composite_score: float
    hyper_reactive_penalty: bool
