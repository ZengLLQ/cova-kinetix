"""
Strict 7-bit ASCII terminal visualization engine for CovaKinetix.

Every rendered character is guaranteed to lie within ASCII codes 0-127 so
Windows cp1252, legacy pipes, and headless CI consoles never encounter
UnicodeEncodeError.
"""

from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np

from cova_kinetix.types import (
    BenchmarkReport,
    CalibrationReport,
    DockingAlignment,
    EHTResult,
    EvolutionCandidate,
    FloatArray,
    KineticTrace,
    MetadynamicsResult,
    MolecularGraph,
    PermeationProfile,
    PocketGrid,
    ScreeningRecord,
)


def assert_ascii(text: str) -> str:
    """Verify every character in ``text`` is strictly 7-bit ASCII (ord <= 127)."""
    for idx, ch in enumerate(text):
        if ord(ch) > 127:
            raise ValueError(f"Non-ASCII character {repr(ch)} (ord={ord(ch)}) at index {idx}.")
    return text


def _ascii_curve(
    x_vals: FloatArray,
    y_vals: FloatArray,
    title: str,
    x_label: str,
    y_label: str,
    width: int = 62,
    height: int = 12,
    marker: str = "*",
) -> str:
    """Render a 1D curve onto a crisp 7-bit ASCII grid."""
    y_min = float(np.min(y_vals))
    y_max = float(np.max(y_vals))
    if abs(y_max - y_min) < 1e-9:
        y_max = y_min + 1.0

    canvas = [[" " for _ in range(width)] for _ in range(height)]
    sample_idx = np.linspace(0, len(x_vals) - 1, width).astype(int)
    for col, idx in enumerate(sample_idx):
        norm_y = (float(y_vals[idx]) - y_min) / (y_max - y_min)
        row = height - 1 - int(round(norm_y * (height - 1)))
        row = max(0, min(height - 1, row))
        canvas[row][col] = marker

    lines = [f"+-- {title} " + "-" * max(2, width - len(title) + 4) + "+"]
    for r in range(height):
        val_tick = y_max - (r / max(1, height - 1)) * (y_max - y_min)
        row_str = "".join(canvas[r])
        lines.append(f"{val_tick:7.2f} |{row_str}|")
    lines.append("        +" + "-" * width + "+")
    lines.append(f"         {x_vals[0]:<10.2f} {x_label:^38} {x_vals[-1]:>10.2f}  ({y_label})")
    return assert_ascii("\n".join(lines))


def render_eht_summary(mol: MolecularGraph, eht: EHTResult, top_atoms: int = 8) -> str:
    """Render Extended Huckel MO ladder and condensed Fukui f+ / dual descriptor table."""
    lines: List[str] = [
        f"=== Extended Huckel Quantum Reactivity: {mol.name} ===",
        f"Warhead       : {mol.warhead_class or 'none'} (Electrophile atom #{mol.electrophile_atom_idx})",
        f"Basis / e-    : {eht.n_orbitals} STOs | {eht.n_valence_electrons} valence e- "
        f"(Tr(PS)={eht.trace_ps_electrons:.4f})",
        f"HOMO / LUMO   : {eht.homo_ev:.3f} eV / {eht.lumo_ev:.3f} eV  (Gap = {eht.homo_lumo_gap_ev:.3f} eV)",
        f"Global CDFT   : mu={eht.chemical_potential_ev:.3f} eV | eta={eht.chemical_hardness_ev:.3f} eV | "
        f"omega={eht.electrophilicity_omega_ev:.3f} eV",
        f"GSH Stability : t1/2 = {eht.predicted_gsh_t_half_min:.1f} min  | Hyper-reactive = {eht.hyper_reactive_flag}",
        "",
        "Top Electrophilic Heavy Atoms (Condensed Fukui f+):",
        "Idx  Sym   q_Mulliken   f+ (Nu-atk)   f- (El-atk)   Dual df    Bar",
        "-" * 74,
    ]
    n_heavy = len(mol.atoms)
    ranked = sorted(range(n_heavy), key=lambda i: float(eht.fukui_plus[i]), reverse=True)[:top_atoms]
    max_fp = max(float(eht.fukui_plus[i]) for i in ranked) if ranked else 1.0
    for idx in ranked:
        sym = mol.atoms[idx].symbol
        q_m = float(eht.mulliken_charges[idx])
        fp = float(eht.fukui_plus[idx])
        fm = float(eht.fukui_minus[idx])
        df = float(eht.dual_descriptor[idx])
        bar_len = int(round(18.0 * fp / max(max_fp, 1e-9)))
        bar = "#" * bar_len + "." * (18 - bar_len)
        tag = "*" if idx == mol.electrophile_atom_idx else " "
        lines.append(f"{idx:3d}{tag} {sym:<3s}  {q_m:+9.4f}     {fp:8.4f}      {fm:8.4f}    {df:+8.4f}   [{bar}]")
    return assert_ascii("\n".join(lines))


def render_pocket_slice_contour(pb_grid: PocketGrid, docking: DockingAlignment) -> str:
    """Render middle-Z slice of the 3D Poisson-Boltzmann potential and docking readout."""
    z_mid = pb_grid.shape[2] // 2
    phi_slice = pb_grid.electrostatic_potential_kt[:, :, z_mid]
    ramp = " .:-=+*#%@"
    vmax = max(float(np.max(np.abs(phi_slice))), 1.0)
    lines: List[str] = [
        f"=== 3D Poisson-Boltzmann Electrostatic Slice & Induced-Fit Alignment ({docking.target_name}) ===",
        f"PB Solver     : {pb_grid.pb_iterations} SOR iters | residual={pb_grid.pb_residual:.2e} kT",
        f"Rotamer chi1  : {docking.chi1_deg:+.1f} deg (strain={docking.rotamer_strain_kcal:.2f} kcal/mol)",
        f"Noncov dG     : {docking.noncovalent_dg_kcal:.2f} kcal/mol -> Ki = {docking.effective_ki_uM:.4f} uM",
        f"Breakdown     : steric={docking.steric_kcal:.2f} | hbond={docking.hbond_kcal:.2f} | "
        f"elec={docking.electrostatic_kcal:.2f} | desolv={docking.desolvation_kcal:.2f}",
        f"Pre-reaction  : d(Nu-C_elec) = {docking.nucleophile_distance_a:.2f} A | "
        f"Burgi-Dunitz = {docking.burgi_dunitz_angle_deg:.1f} deg",
        "",
        "PB Potential |phi(x,y,z_mid)| Contour ('.' low -> '@' high magnitude):",
        "+" + "-" * (phi_slice.shape[1] * 2) + "+",
    ]
    for r in range(phi_slice.shape[0]):
        row_chars = []
        for c in range(phi_slice.shape[1]):
            norm = min(1.0, abs(float(phi_slice[r, c])) / vmax)
            ch = ramp[int(round(norm * (len(ramp) - 1)))]
            row_chars.append(ch + ch)
        lines.append("|" + "".join(row_chars) + "|")
    lines.append("+" + "-" * (phi_slice.shape[1] * 2) + "+")
    return assert_ascii("\n".join(lines))


def render_permeation_profile(perm: PermeationProfile) -> str:
    """Render ISD transmembrane Delta G(z) profile and permeability summary."""
    header = (
        f"=== Inhomogeneous Solubility-Diffusion (ISD) Membrane Permeation ===\n"
        f"P_eff = {perm.p_eff_cm_s:.3e} cm/s (log10={perm.log10_p_eff:.2f}, class={perm.permeability_class})\n"
        f"f_neutral = {perm.neutral_fraction:.3f} | Efflux Ratio = {perm.efflux_ratio:.2f} | "
        f"P_app(A->B) = {perm.p_app_apical_to_basolateral:.3e} | K_p,uu = {perm.k_puu_intracellular:.3f}"
    )
    curve = _ascii_curve(
        perm.z_grid_a,
        perm.delta_g_z_kcal,
        title="Bilayer Potential of Mean Force Delta G(z)",
        x_label="Bilayer Depth z (A)",
        y_label="kcal/mol",
        marker="*",
    )
    return assert_ascii(header + "\n" + curve)


def render_metadynamics_fes(metad: MetadynamicsResult) -> str:
    """Render Well-Tempered Metadynamics 1D FES F(s1) and Infrequent MetaD kinetics."""
    header = (
        f"=== Well-Tempered & Infrequent Metadynamics Unbinding Kinetics ===\n"
        f"dG_bind = {metad.delta_g_bind_kcal:.2f} kcal/mol | Barrier_off = {metad.barrier_off_kcal:.2f} kcal/mol | "
        f"Barrier_on = {metad.barrier_on_kcal:.2f} kcal/mol\n"
        f"k_off   = {metad.unbiased_k_off_per_s:.3e} s^-1 | tau_res = {metad.mean_residence_time_s:.2f} s\n"
        f"KS Test : D = {metad.ks_statistic:.3f}, p = {metad.ks_pvalue:.4f} -> "
        f"Poisson Certified = {metad.poisson_certified}"
    )
    curve = _ascii_curve(
        metad.cv1_grid,
        metad.fes_1d_cv1_kcal,
        title="Reconstructed 1D Unbinding FES F(s1)",
        x_label="Unbinding CV s1 (nm)",
        y_label="kcal/mol",
        marker="o",
    )
    return assert_ascii(header + "\n" + curve)


def render_covalent_kinetics(kin: KineticTrace) -> str:
    """Render two-step covalent occupancy time course and IC50(t) shift readout."""
    header = (
        f"=== Two-Step Covalent Engagement & Cellular Turnover PK/PD ===\n"
        f"Ki = {kin.ki_uM:.4f} uM | k_inact = {kin.k_inact_per_s:.3e} s^-1 | k_rev = {kin.k_rev_per_s:.3e} s^-1\n"
        f"k_inact/Ki = {kin.k_inact_over_ki:.3e} M^-1 s^-1 (log10 = {math.log10(max(kin.k_inact_over_ki, 1e-3)):.2f})\n"
        f"Occupancy  : Peak = {kin.peak_occupancy:.3f} | Trough = {kin.trough_occupancy:.3f} | "
        f"AUC = {kin.occupancy_auc_h:.2f} h\n"
        f"Time-Dep   : IC50(1h) = {kin.ic50_1h_uM:.4f} uM -> IC50(24h) = {kin.ic50_24h_uM:.4f} uM"
    )
    curve = _ascii_curve(
        kin.times_h,
        kin.total_occupancy_frac * 100.0,
        title="Target Occupancy (%) Over Dosing Horizon",
        x_label="Time (hours)",
        y_label="% Occ",
        marker="#",
    )
    return assert_ascii(header + "\n" + curve)


def render_evolution_pareto(candidates: Sequence[EvolutionCandidate], max_rows: int = 10) -> str:
    """Render NSGA-II multi-objective covalent fragment evolution Pareto leaderboard."""
    lines: List[str] = [
        "=== NSGA-II Multi-Objective Covalent Fragment Evolution Pareto Front ===",
        "Rank  Warhead                         log(kinact/Ki)  log(Peff)  GSH_t1/2  SA    SMILES",
        "-" * 92,
    ]
    for c in list(candidates)[:max_rows]:
        log_eff = math.log10(max(c.k_inact_over_ki, 1e-3))
        smi_disp = c.smiles if len(c.smiles) <= 34 else c.smiles[:31] + "..."
        lines.append(
            f"{c.pareto_rank:4d}  {c.warhead_name:<30s}  {log_eff:12.2f}  {c.log10_p_eff:9.2f}  "
            f"{c.gsh_t_half_min:7.1f}m  {c.sa_score:4.2f}  {smi_disp}"
        )
    return assert_ascii("\n".join(lines))


def render_calibration_sobol(cal: CalibrationReport) -> str:
    """Render calibration improvement, 95% bootstrap CIs, and Sobol sensitivity bars."""
    lines: List[str] = [
        "=== Model Calibration & Saltelli Sobol Global Sensitivity ===",
        f"RMSE : {cal.initial_rmse:.4f} -> {cal.calibrated_rmse:.4f} (R^2 = {cal.r2_score:.4f})",
        "",
        "Fitted Parameters & 95% Bootstrap CIs:",
    ]
    for k, val in cal.fitted_params.items():
        lo, hi = cal.bootstrap_ci_95[k]
        lines.append(f"  {k:<22s} = {val:8.5f}  (95% CI: [{lo:.5f}, {hi:.5f}])")
    lines.append("")
    lines.append("Saltelli Sobol Sensitivity on 24h Occupancy:")
    lines.append("Parameter                S_1 (Main)   S_T (Total)   Sensitivity Bar")
    lines.append("-" * 74)
    for k in cal.sobol_first_order:
        s1 = cal.sobol_first_order[k]
        st = cal.sobol_total_order[k]
        bars = "#" * int(round(20.0 * min(1.0, st)))
        lines.append(f"{k:<23s}  {s1:10.3f}   {st:11.3f}   [{bars:<20s}]")
    return assert_ascii("\n".join(lines))


def render_benchmark_table(report: BenchmarkReport) -> str:
    """Render multi-compound covalent benchmark metrics and ranked screening table."""
    lines: List[str] = [
        f"=== Covalent Screening Benchmark: {report.target_name} ===",
        f"Compounds={report.n_compounds} (Actives={report.n_actives}) | ROC-AUC={report.roc_auc:.3f} | "
        f"BEDROC20={report.bedroc_20:.3f} | EF10%={report.ef_10_pct:.2f} | "
        f"Spearman_rho={report.spearman_rho_kinact_ki:.3f}",
        "",
        "Compound                 Warhead            Act  log(eff)  Occ24h  logPeff  GSH_t1/2  Score",
        "-" * 92,
    ]
    for r in report.records:
        act_str = "YES" if r.is_active_ref else "no "
        flag = "!" if r.hyper_reactive_penalty else " "
        lines.append(
            f"{r.name:<23s}  {r.warhead_class:<17s}  {act_str}  {r.predicted_log_kinact_ki:8.2f}  "
            f"{r.occupancy_24h:6.3f}  {r.predicted_log10_p_eff:7.2f}  "
            f"{r.predicted_gsh_t_half_min:6.1f}m{flag} {r.composite_score:6.2f}"
        )
    return assert_ascii("\n".join(lines))


def render_screening_records(records: Sequence[ScreeningRecord], target_name: str) -> str:
    """Render top screening hits for a target pocket."""
    n_act = sum(1 for r in records if r.is_active_ref)
    dummy = BenchmarkReport(target_name, len(records), n_act, 1.0, 1.0, 1.0, 1.0, 0.0, list(records))
    return render_benchmark_table(dummy)
