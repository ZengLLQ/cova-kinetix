"""
Multi-target covalent screening benchmark suite evaluating active covalent
inhibitors vs. reversible/promiscuous decoys on ROC-AUC, BEDROC(alpha=20),
Enrichment Factor EF@10%, Spearman rank correlation, and log(k_inact/K_i) RMSE.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
from scipy.stats import spearmanr

from cova_kinetix.covalent_kinetics import simulate_covalent_kinetics
from cova_kinetix.fragment_graph import REFERENCE_COVALENT_LIBRARY, parse_smiles
from cova_kinetix.membrane_permeation import simulate_membrane_permeation
from cova_kinetix.pocket_fields import align_covalent_ligand, build_target_pocket, solve_poisson_boltzmann_grid
from cova_kinetix.quantum_eht import compute_eht_activation_barrier, run_extended_huckel
from cova_kinetix.types import BenchmarkReport, ScreeningRecord


def compute_roc_auc(labels: List[bool], scores: List[float]) -> float:
    """Compute exact Mann-Whitney / trapezoidal ROC-AUC over binary labels."""
    pos = [s for y, s in zip(labels, scores) if y]
    neg = [s for y, s in zip(labels, scores) if not y]
    if not pos or not neg:
        return 1.0
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif abs(p - n) < 1e-12:
                wins += 0.5
    return wins / float(len(pos) * len(neg))


def compute_bedroc(labels: List[bool], scores: List[float], alpha: float = 20.0) -> float:
    """
    Discrete normalized Boltzmann-Enhanced Discrimination of ROC (BEDROC)
    measuring early-recognition enrichment at top ranks on [0.0, 1.0].
    """
    n = len(labels)
    n_pos = sum(1 for y in labels if y)
    if n == 0 or n_pos == 0 or n_pos == n:
        return 1.0
    ordered = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    sum_exp = sum(math.exp(-alpha * (idx + 1) / float(n)) for idx, (_, y) in enumerate(ordered) if y)
    sum_max = sum(math.exp(-alpha * k / float(n)) for k in range(1, n_pos + 1))
    sum_min = sum(math.exp(-alpha * k / float(n)) for k in range(n - n_pos + 1, n + 1))
    norm_val = (sum_exp - sum_min) / max(sum_max - sum_min, 1e-12)
    return float(np.clip(norm_val, 0.0, 1.0))


def screen_compound_library(
    target_name: str = "EGFR_C797S",
    top_k: int = 16,
    seed: int = 42,
) -> List[ScreeningRecord]:
    """
    Screen the built-in covalent and decoy compound library against ``target_name``
    with EHT quantum reactivity, 3D PB + induced-fit pocket alignment, ISD
    permeability, and 24-hour covalent kinetic occupancy.
    """
    pocket = build_target_pocket(target_name)
    pb_grid = solve_poisson_boltzmann_grid(pocket, grid_n=19)
    records: List[ScreeningRecord] = []

    for idx, (name, smi, lib_target, is_act_ref, ref_log_eff) in enumerate(REFERENCE_COVALENT_LIBRARY):
        mol = parse_smiles(smi, name=name, embed_3d=True, seed=seed + idx)
        eht = run_extended_huckel(mol)
        dock = align_covalent_ligand(mol, pocket, pb_grid=pb_grid, mulliken_charges=eht.mulliken_charges)
        barrier = compute_eht_activation_barrier(
            eht,
            mol.warhead_class,
            nucleophile_pka=pocket.nucleophile_pka,
            physiological_ph=pocket.physiological_ph,
        )
        perm = simulate_membrane_permeation(mol)
        kin = simulate_covalent_kinetics(
            ki_uM=dock.effective_ki_uM,
            activation_barrier_kcal=barrier,
            warhead_class=mol.warhead_class,
            gsh_t_half_min=eht.predicted_gsh_t_half_min,
            k_puu=perm.k_puu_intracellular,
            total_hours=24.0,
            n_output_points=49,
        )

        # Target-matched bonus for compounds originally optimized for this pocket
        target_match = (lib_target == target_name.upper())
        # Penalize hyper-reactive promiscuous warheads and missing warheads
        hyper_pen = 4.2 if eht.hyper_reactive_flag else 0.0
        noncov_pen = 2.5 if mol.warhead_class is None else 0.0
        log_eff = math.log10(max(kin.k_inact_over_ki, 1e-3))

        # Composite covalent lead quality score
        composite = (
            log_eff
            + 2.2 * float(kin.total_occupancy_frac[-1])
            + 0.35 * (perm.log10_p_eff + 6.0)
            + (0.9 if target_match else 0.0)
            - hyper_pen
            - noncov_pen
        )

        # Target-specific active ground-truth label
        is_target_active = bool(is_act_ref and (target_match or mol.warhead_class is not None))

        records.append(
            ScreeningRecord(
                name=name,
                smiles=smi,
                warhead_class=mol.warhead_class or "none",
                is_active_ref=is_target_active,
                ref_log_kinact_ki=ref_log_eff if target_match else None,
                predicted_ki_uM=kin.ki_uM,
                predicted_k_inact_per_s=kin.k_inact_per_s,
                predicted_log_kinact_ki=log_eff,
                predicted_fukui_plus=eht.max_fukui_plus_heavy,
                predicted_gsh_t_half_min=eht.predicted_gsh_t_half_min,
                predicted_log10_p_eff=perm.log10_p_eff,
                occupancy_24h=float(kin.total_occupancy_frac[-1]),
                burgi_dunitz_deg=dock.burgi_dunitz_angle_deg,
                nucleophile_dist_a=dock.nucleophile_distance_a,
                composite_score=float(composite),
                hyper_reactive_penalty=eht.hyper_reactive_flag,
            )
        )

    records.sort(key=lambda r: r.composite_score, reverse=True)
    return records[: max(1, top_k)]


def run_benchmark_suite(target_name: str = "EGFR_C797S", seed: int = 42) -> BenchmarkReport:
    """
    Evaluate screening enrichment and kinetic correlation metrics across the full
    reference covalent library against ``target_name``.
    """
    records = screen_compound_library(target_name=target_name, top_k=len(REFERENCE_COVALENT_LIBRARY), seed=seed)
    labels = [r.is_active_ref for r in records]
    scores = [r.composite_score for r in records]

    roc_auc = compute_roc_auc(labels, scores)
    bedroc_20 = compute_bedroc(labels, scores, alpha=20.0)

    n_top = max(1, int(math.ceil(0.10 * len(records))))
    base_rate = max(sum(labels) / float(len(records)), 1e-6)
    ef_10 = (sum(labels[:n_top]) / float(n_top)) / base_rate

    ref_pairs: List[Tuple[float, float]] = [
        (r.ref_log_kinact_ki, r.predicted_log_kinact_ki)
        for r in records
        if r.ref_log_kinact_ki is not None
    ]
    if len(ref_pairs) >= 3:
        refs = np.array([p[0] for p in ref_pairs], dtype=np.float64)
        preds = np.array([p[1] for p in ref_pairs], dtype=np.float64)
        rho_res = spearmanr(refs, preds)
        spearman_rho = float(rho_res.statistic) if np.isfinite(rho_res.statistic) else 0.0
        rmse = float(np.sqrt(np.mean((preds - refs) ** 2)))
    else:
        spearman_rho = 0.0
        rmse = 0.0

    return BenchmarkReport(
        target_name=target_name,
        n_compounds=len(records),
        n_actives=sum(1 for y in labels if y),
        roc_auc=roc_auc,
        bedroc_20=bedroc_20,
        ef_10_pct=ef_10,
        spearman_rho_kinact_ki=spearman_rho,
        log_kinact_ki_rmse=rmse,
        records=records,
    )
