"""
End-to-end covalent drug discovery pipeline orchestrator and JSON telemetry exporter.

Unifies SMILES parsing -> Extended Huckel quantum reactivity -> 3D Poisson-Boltzmann
induced-fit pocket alignment -> ISD bilayer permeation -> Well-Tempered Metadynamics
unbinding kinetics -> two-step covalent PK/PD occupancy simulation -> JSON export.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from cova_kinetix.covalent_kinetics import simulate_covalent_kinetics
from cova_kinetix.fragment_graph import parse_smiles
from cova_kinetix.membrane_permeation import simulate_membrane_permeation
from cova_kinetix.metadynamics import run_well_tempered_metadynamics
from cova_kinetix.pocket_fields import align_covalent_ligand, build_target_pocket, solve_poisson_boltzmann_grid
from cova_kinetix.quantum_eht import compute_eht_activation_barrier, run_extended_huckel
from cova_kinetix.types import (
    BindingPocket,
    DockingAlignment,
    EHTResult,
    KineticTrace,
    MetadynamicsResult,
    MolecularGraph,
    PermeationProfile,
    PocketGrid,
)


@dataclass
class FullCharacterizationResult:
    """Complete multi-scale characterization bundle for a single covalent ligand-target pair."""

    molecule: MolecularGraph
    pocket: BindingPocket
    pb_grid: PocketGrid
    eht: EHTResult
    docking: DockingAlignment
    permeation: PermeationProfile
    kinetics: KineticTrace
    metadynamics: MetadynamicsResult
    activation_barrier_kcal: float

    def to_summary_dict(self) -> Dict[str, Any]:
        """Serialize scalar telemetry and physics observables to a JSON-safe dictionary."""
        return {
            "ligand_name": self.molecule.name,
            "smiles": self.molecule.smiles,
            "target": self.pocket.target_name,
            "warhead_class": self.molecule.warhead_class,
            "physicochemical": {
                "molecular_weight": round(self.molecule.molecular_weight, 3),
                "logp": round(self.molecule.logp, 3),
                "tpsa": round(self.molecule.tpsa, 3),
                "hbd": self.molecule.hbd,
                "hba": self.molecule.hba,
                "qed": round(self.molecule.qed, 4),
                "sa_score": round(self.molecule.sa_score, 3),
            },
            "quantum_eht": {
                "homo_ev": round(self.eht.homo_ev, 4),
                "lumo_ev": round(self.eht.lumo_ev, 4),
                "gap_ev": round(self.eht.homo_lumo_gap_ev, 4),
                "electrophilicity_omega_ev": round(self.eht.electrophilicity_omega_ev, 4),
                "max_fukui_plus_heavy": round(self.eht.max_fukui_plus_heavy, 4),
                "predicted_gsh_t_half_min": round(self.eht.predicted_gsh_t_half_min, 2),
                "hyper_reactive_flag": self.eht.hyper_reactive_flag,
            },
            "pocket_docking": {
                "rotamer_chi1_deg": round(self.docking.chi1_deg, 1),
                "noncovalent_dg_kcal": round(self.docking.noncovalent_dg_kcal, 3),
                "effective_ki_uM": round(self.docking.effective_ki_uM, 5),
                "nucleophile_distance_a": round(self.docking.nucleophile_distance_a, 3),
                "burgi_dunitz_angle_deg": round(self.docking.burgi_dunitz_angle_deg, 2),
            },
            "membrane_permeation": {
                "p_eff_cm_s": float(f"{self.permeation.p_eff_cm_s:.4e}"),
                "log10_p_eff": round(self.permeation.log10_p_eff, 3),
                "efflux_ratio": round(self.permeation.efflux_ratio, 3),
                "k_puu_intracellular": round(self.permeation.k_puu_intracellular, 3),
                "permeability_class": self.permeation.permeability_class,
            },
            "covalent_kinetics": {
                "activation_barrier_kcal": round(self.activation_barrier_kcal, 3),
                "k_inact_per_s": float(f"{self.kinetics.k_inact_per_s:.4e}"),
                "k_inact_over_ki": float(f"{self.kinetics.k_inact_over_ki:.4e}"),
                "log10_k_inact_over_ki": round(math.log10(max(self.kinetics.k_inact_over_ki, 1e-3)), 3),
                "peak_occupancy": round(self.kinetics.peak_occupancy, 4),
                "trough_occupancy": round(self.kinetics.trough_occupancy, 4),
                "ic50_1h_uM": round(self.kinetics.ic50_1h_uM, 5),
                "ic50_24h_uM": round(self.kinetics.ic50_24h_uM, 5),
            },
            "metadynamics": {
                "delta_g_bind_kcal": round(self.metadynamics.delta_g_bind_kcal, 3),
                "barrier_off_kcal": round(self.metadynamics.barrier_off_kcal, 3),
                "unbiased_k_off_per_s": float(f"{self.metadynamics.unbiased_k_off_per_s:.4e}"),
                "mean_residence_time_s": round(self.metadynamics.mean_residence_time_s, 3),
                "ks_pvalue": round(self.metadynamics.ks_pvalue, 4),
                "poisson_certified": self.metadynamics.poisson_certified,
            },
        }


def run_full_characterization(
    smiles: str,
    target_name: str = "EGFR_C797S",
    ligand_name: str = "CovaLead",
    metad_steps: int = 1600,
    n_metad_trajectories: int = 8,
    seed: int = 42,
    export_json_path: Optional[str | Path] = None,
) -> FullCharacterizationResult:
    """
    Execute the complete multi-scale CovaKinetix physics workflow on a single
    covalent ligand against a chosen protein pocket target.
    """
    mol = parse_smiles(smiles, name=ligand_name, embed_3d=True, seed=seed)
    pocket = build_target_pocket(target_name)
    pb_grid = solve_poisson_boltzmann_grid(pocket, grid_n=21)
    eht = run_extended_huckel(mol)
    docking = align_covalent_ligand(mol, pocket, pb_grid=pb_grid, mulliken_charges=eht.mulliken_charges)
    barrier = compute_eht_activation_barrier(
        eht,
        mol.warhead_class,
        nucleophile_pka=pocket.nucleophile_pka,
        physiological_ph=pocket.physiological_ph,
    )
    permeation = simulate_membrane_permeation(mol)
    kinetics = simulate_covalent_kinetics(
        ki_uM=docking.effective_ki_uM,
        activation_barrier_kcal=barrier,
        warhead_class=mol.warhead_class,
        gsh_t_half_min=eht.predicted_gsh_t_half_min,
        k_puu=permeation.k_puu_intracellular,
        total_hours=48.0,
    )
    metad = run_well_tempered_metadynamics(
        delta_g_bind_kcal=docking.noncovalent_dg_kcal,
        barrier_extra_kcal=4.8,
        n_steps=metad_steps,
        n_infrequent_trajectories=n_metad_trajectories,
        seed=seed,
    )

    result = FullCharacterizationResult(
        molecule=mol,
        pocket=pocket,
        pb_grid=pb_grid,
        eht=eht,
        docking=docking,
        permeation=permeation,
        kinetics=kinetics,
        metadynamics=metad,
        activation_barrier_kcal=barrier,
    )

    if export_json_path is not None:
        out_path = Path(export_json_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result.to_summary_dict(), indent=2) + "\n", encoding="utf-8")

    return result
