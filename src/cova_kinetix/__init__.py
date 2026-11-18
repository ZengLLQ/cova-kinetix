"""
CovaKinetix: Structure-Based Covalent Drug Discovery & Quantum-Kinetic Simulation Platform.

Integrates SMILES/BRICS fragment graph perception, 3D Slater-type orbital Extended
Huckel Theory (EHT) condensed Fukui reactivity, 3D linearized Poisson-Boltzmann
pocket grids with induced-fit rotamer and Burgi-Dunitz trajectory alignment,
two-step covalent occupancy PK/PD ODEs, well-tempered and infrequent metadynamics
unbinding kinetics, inhomogeneous solubility-diffusion bilayer permeability, and
NSGA-II multi-objective fragment evolution.
"""

from cova_kinetix.types import (
    AVOGADRO,
    BOHR_TO_ANGSTROM,
    ELEMENT_PARAMS,
    EV_TO_KCAL_MOL,
    GAS_CONSTANT_KCAL,
    NUCLEOPHILE_PROFILES,
    WARHEAD_TEMPLATES,
    BindingPocket,
    BenchmarkReport,
    CalibrationReport,
    CovaKinetixError,
    DockingAlignment,
    EHTResult,
    ElementParams,
    EvolutionCandidate,
    FragmentPiece,
    KineticTrace,
    MetadynamicsResult,
    MolecularGraph,
    NucleophileProfile,
    PermeationProfile,
    PocketGrid,
    PocketResidue,
    ScreeningRecord,
    WarheadTemplate,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "AVOGADRO",
    "BOHR_TO_ANGSTROM",
    "EV_TO_KCAL_MOL",
    "GAS_CONSTANT_KCAL",
    "ELEMENT_PARAMS",
    "WARHEAD_TEMPLATES",
    "NUCLEOPHILE_PROFILES",
    "CovaKinetixError",
    "ElementParams",
    "WarheadTemplate",
    "NucleophileProfile",
    "MolecularGraph",
    "FragmentPiece",
    "EHTResult",
    "PocketResidue",
    "BindingPocket",
    "PocketGrid",
    "DockingAlignment",
    "KineticTrace",
    "MetadynamicsResult",
    "PermeationProfile",
    "EvolutionCandidate",
    "CalibrationReport",
    "BenchmarkReport",
    "ScreeningRecord",
]
