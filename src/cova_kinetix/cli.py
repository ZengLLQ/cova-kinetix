"""
Command-line interface for CovaKinetix (``cova-kinetix`` and ``python -m cova_kinetix.cli``).

Provides 8 operational subcommands: ``demo``, ``quantum``, ``screen``, ``permeate``,
``metad``, ``evolve``, ``calibrate``, and ``benchmark``.
"""

from __future__ import annotations

import argparse
import json
from typing import List, Optional

from cova_kinetix.benchmark import run_benchmark_suite, screen_compound_library
from cova_kinetix.calibration import calibrate_kinetics_and_sensitivity
from cova_kinetix.fragment_evolution import evolve_covalent_ligands
from cova_kinetix.fragment_graph import parse_smiles
from cova_kinetix.membrane_permeation import simulate_membrane_permeation
from cova_kinetix.metadynamics import run_well_tempered_metadynamics
from cova_kinetix.pipeline import run_full_characterization
from cova_kinetix.quantum_eht import run_extended_huckel
from cova_kinetix.visualizer import (
    render_benchmark_table,
    render_calibration_sobol,
    render_covalent_kinetics,
    render_eht_summary,
    render_evolution_pareto,
    render_metadynamics_fes,
    render_permeation_profile,
    render_pocket_slice_contour,
    render_screening_records,
)


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argparse CLI parser and all subcommands."""
    parser = argparse.ArgumentParser(
        prog="cova-kinetix",
        description="CovaKinetix: Structure-based covalent drug discovery and quantum-kinetic simulator.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # 1. demo
    p_demo = sub.add_parser("demo", help="Run full end-to-end covalent discovery & kinetic workflow.")
    p_demo.add_argument(
        "--smiles",
        default="C=CC(=O)Nc1cc(Nc2nccc(-c3cn(C)c4ccccc34)n2)c(OC)cc1N(C)CCN(C)C",
        help="Input covalent ligand SMILES.",
    )
    p_demo.add_argument("--target", default="EGFR_C797S", help="Target pocket name.")
    p_demo.add_argument("--metad-steps", type=int, default=1200, help="WT-MetaD simulation steps.")
    p_demo.add_argument("--seed", type=int, default=42, help="Deterministic RNG seed.")
    p_demo.add_argument("--export-json", default=None, help="Optional JSON summary export path.")

    # 2. quantum
    p_q = sub.add_parser("quantum", help="Run Extended Huckel quantum chemistry & Fukui reactivity.")
    p_q.add_argument("--smiles", required=True, help="Ligand SMILES string.")
    p_q.add_argument("--name", default="QueryLigand", help="Ligand label.")
    p_q.add_argument("--seed", type=int, default=42, help="3D embedding seed.")

    # 3. screen
    p_scr = sub.add_parser("screen", help="Screen the covalent & decoy library against a pocket.")
    p_scr.add_argument(
        "--target",
        default="EGFR_C797S",
        help="Target pocket (EGFR_C797S, BTK_C481, KRAS_G12C, MPRO_C145).",
    )
    p_scr.add_argument("--top-k", type=int, default=10, help="Number of top candidates to display.")
    p_scr.add_argument("--seed", type=int, default=42, help="Deterministic RNG seed.")

    # 4. permeate
    p_perm = sub.add_parser("permeate", help="Run ISD lipid bilayer permeation simulation.")
    p_perm.add_argument("--smiles", required=True, help="Ligand SMILES string.")
    p_perm.add_argument("--ph", type=float, default=7.4, help="Apical physiological pH.")
    p_perm.add_argument("--name", default="Permeant", help="Ligand label.")

    # 5. metad
    p_met = sub.add_parser("metad", help="Run Well-Tempered & Infrequent Metadynamics unbinding.")
    p_met.add_argument("--dg-bind", type=float, default=-7.5, help="Bound-state free energy well depth (kcal/mol).")
    p_met.add_argument("--barrier-extra", type=float, default=5.0, help="Extra transition-state barrier (kcal/mol).")
    p_met.add_argument("--steps", type=int, default=2000, help="WT-MetaD trajectory steps.")
    p_met.add_argument("--n-trajectories", type=int, default=8, help="Infrequent MetaD trajectories for KS test.")
    p_met.add_argument("--seed", type=int, default=42, help="Deterministic RNG seed.")

    # 6. evolve
    p_evo = sub.add_parser("evolve", help="Run NSGA-II multi-objective covalent fragment evolution.")
    p_evo.add_argument("--target", default="EGFR_C797S", help="Target pocket name.")
    p_evo.add_argument("--generations", type=int, default=4, help="Number of NSGA-II generations.")
    p_evo.add_argument("--pop-size", type=int, default=14, help="Population size.")
    p_evo.add_argument("--seed", type=int, default=42, help="Deterministic RNG seed.")

    # 7. calibrate
    p_cal = sub.add_parser("calibrate", help="Run parameter calibration, bootstrap CI, and Sobol sensitivity.")
    p_cal.add_argument("--n-bootstrap", type=int, default=20, help="Bootstrap resamples.")
    p_cal.add_argument("--n-sobol", type=int, default=48, help="Saltelli Sobol base sample count.")
    p_cal.add_argument("--seed", type=int, default=42, help="Deterministic RNG seed.")

    # 8. benchmark
    p_bench = sub.add_parser("benchmark", help="Run multi-compound covalent benchmark & enrichment validation.")
    p_bench.add_argument("--target", default="EGFR_C797S", help="Target pocket name.")
    p_bench.add_argument("--seed", type=int, default=42, help="Deterministic RNG seed.")
    p_bench.add_argument("--json-out", action="store_true", help="Emit JSON metrics summary.")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point returning POSIX exit code 0 on success."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "demo":
        res = run_full_characterization(
            smiles=args.smiles,
            target_name=args.target,
            ligand_name="Osimertinib_analog",
            metad_steps=args.metad_steps,
            seed=args.seed,
            export_json_path=args.export_json,
        )
        print(render_eht_summary(res.molecule, res.eht))
        print()
        print(render_pocket_slice_contour(res.pb_grid, res.docking))
        print()
        print(render_permeation_profile(res.permeation))
        print()
        print(render_covalent_kinetics(res.kinetics))
        print()
        print(render_metadynamics_fes(res.metadynamics))
        return 0

    if args.command == "quantum":
        mol = parse_smiles(args.smiles, name=args.name, embed_3d=True, seed=args.seed)
        eht = run_extended_huckel(mol)
        print(render_eht_summary(mol, eht))
        return 0

    if args.command == "screen":
        recs = screen_compound_library(target_name=args.target, top_k=args.top_k, seed=args.seed)
        print(render_screening_records(recs, target_name=args.target))
        return 0

    if args.command == "permeate":
        mol = parse_smiles(args.smiles, name=args.name, embed_3d=False)
        perm = simulate_membrane_permeation(mol, ph_apical=args.ph)
        print(render_permeation_profile(perm))
        return 0

    if args.command == "metad":
        metad = run_well_tempered_metadynamics(
            delta_g_bind_kcal=args.dg_bind,
            barrier_extra_kcal=args.barrier_extra,
            n_steps=args.steps,
            n_infrequent_trajectories=args.n_trajectories,
            seed=args.seed,
        )
        print(render_metadynamics_fes(metad))
        return 0

    if args.command == "evolve":
        cands = evolve_covalent_ligands(
            target_name=args.target,
            n_generations=args.generations,
            population_size=args.pop_size,
            seed=args.seed,
        )
        print(render_evolution_pareto(cands))
        return 0

    if args.command == "calibrate":
        cal = calibrate_kinetics_and_sensitivity(
            n_bootstrap=args.n_bootstrap,
            n_sobol_samples=args.n_sobol,
            seed=args.seed,
        )
        print(render_calibration_sobol(cal))
        return 0

    if args.command == "benchmark":
        rep = run_benchmark_suite(target_name=args.target, seed=args.seed)
        if args.json_out:
            summary = {
                "target_name": rep.target_name,
                "n_compounds": rep.n_compounds,
                "n_actives": rep.n_actives,
                "roc_auc": round(rep.roc_auc, 4),
                "bedroc_20": round(rep.bedroc_20, 4),
                "ef_10_pct": round(rep.ef_10_pct, 4),
                "spearman_rho_kinact_ki": round(rep.spearman_rho_kinact_ki, 4),
                "log_kinact_ki_rmse": round(rep.log_kinact_ki_rmse, 4),
            }
            print(json.dumps(summary, indent=2))
        else:
            print(render_benchmark_table(rep))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
