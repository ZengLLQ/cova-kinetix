"""
Inhomogeneous Solubility-Diffusion (ISD) transmembrane lipid-bilayer permeability
and Henderson-Hasselbalch pH-partitioning model for intracellular unbound
accumulation K_{p,uu}.

Evaluates the spatial potential of mean force Delta G(z) and position-dependent
diffusivity D(z) across z in [-30, +30] Angstroms (bulk aqueous phase -> zwitterionic
polar headgroup interface -> dense acyl region -> disordered hydrocarbon interior)
and integrates the resistance R = integral(exp(beta * Delta G(z)) / D(z) dz).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from cova_kinetix.types import (
    GAS_CONSTANT_KCAL,
    MolecularGraph,
    PermeationProfile,
)


def compute_neutral_fraction(
    pka_acid: Optional[float],
    pka_base: Optional[float],
    ph: float = 7.4,
) -> float:
    """
    Compute the neutral microspecies fraction via Henderson-Hasselbalch equilibria
    for monoprotic acids, bases, or ampholytes at a given pH.
    """
    acid_term = 10.0 ** (ph - pka_acid) if pka_acid is not None else 0.0
    base_term = 10.0 ** (pka_base - ph) if pka_base is not None else 0.0
    return float(1.0 / (1.0 + acid_term + base_term))


def simulate_membrane_permeation(
    mol: MolecularGraph,
    ph_apical: float = 7.4,
    ph_cytosol: float = 7.2,
    temperature_k: float = 310.15,
    bilayer_half_thickness_a: float = 30.0,
    n_grid: int = 241,
    dielectric_barrier_scale: float = 1.0,
) -> PermeationProfile:
    """
    Simulate passive transmembrane permeation through a POPC/DPPC lipid bilayer
    using the Inhomogeneous Solubility-Diffusion (ISD) integral:

    .. math::
        \\frac{1}{P_{\\text{eff}}} = \\int_{-L/2}^{+L/2}
        \\frac{\\exp(\\beta \\Delta G(z))}{D(z)}\\,dz

    Parameters
    ----------
    mol : MolecularGraph
        Permeating ligand with MW, logP, TPSA, HBD, HBA, formal_charge, and pKa.
    ph_apical : float
        Extracellular / apical compartment pH.
    ph_cytosol : float
        Intracellular cytosolic compartment pH.
    temperature_k : float
        Absolute temperature (K).
    bilayer_half_thickness_a : float
        Half-width of simulation slab in Angstroms (default 30.0 A -> [-30, +30]).
    n_grid : int
        Number of spatial quadrature nodes across the slab.
    dielectric_barrier_scale : float
        Calibrated scalar multiplier on the electrostatic Born/TPSA desolvation barrier.

    Returns
    -------
    PermeationProfile
        Spatial free-energy barrier, diffusivity profile, P_eff (cm/s), efflux ratio,
        apical-to-basolateral P_app, and intracellular K_{p,uu}.
    """
    if n_grid < 51 or n_grid % 2 == 0:
        raise ValueError("n_grid must be an odd integer >= 51.")

    z_a = np.linspace(-bilayer_half_thickness_a, bilayer_half_thickness_a, n_grid, dtype=np.float64)
    abs_z = np.abs(z_a)
    kt_kcal = GAS_CONSTANT_KCAL * temperature_k
    beta = 1.0 / kt_kcal

    # Smooth membrane structural profiles (0 in bulk water, 1 inside hydrocarbon interior)
    # Hydrocarbon interior: |z| < 14 A; Headgroup zone: 14 <= |z| <= 22 A
    hydrocarbon_frac = 0.5 * (1.0 - np.tanh((abs_z - 14.5) / 2.6))
    headgroup_bell = np.exp(-0.5 * ((abs_z - 17.5) / 2.4) ** 2)
    ordered_acyl_bell = np.exp(-0.5 * ((abs_z - 9.5) / 3.2) ** 2)

    # Spatially varying dielectric constant eps(z) from 78.4 (water) to 2.2 (bilayer center)
    eps_z = 2.2 + (78.4 - 2.2) * (1.0 - hydrocarbon_frac)

    # 1. Born electrostatic + polar TPSA/HBD desolvation penalty (kcal/mol)
    # Desolvation scales with (1/eps(z) - 1/78.4)
    born_factor = (1.0 / eps_z - 1.0 / 78.4)
    polar_cost = dielectric_barrier_scale * (
        0.055 * mol.tpsa + 1.15 * mol.hbd + 0.45 * mol.hba + 14.0 * (mol.formal_charge ** 2)
    )
    dg_polar = polar_cost * (born_factor / (1.0 / 2.2 - 1.0 / 78.4))

    # 2. Hydrophobic partitioning stabilization inside lipid acyl phase (kcal/mol)
    # logP > 0 lowers free energy inside bilayer (-RT ln(10) * logP * 0.72)
    dg_hydrophobic = -0.72 * kt_kcal * math.log(10.0) * mol.logp * hydrocarbon_frac

    # 3. Headgroup dipole & polar stabilization (amphiphilic dip near 17.5 A)
    dg_headgroup = -(0.35 + 0.015 * mol.tpsa) * headgroup_bell

    # 4. Acyl-chain lateral packing / molecular volume cavity work (kcal/mol)
    mw_excess = max(0.0, (mol.molecular_weight - 220.0) / 100.0)
    dg_cavity = 0.95 * mw_excess * ordered_acyl_bell

    delta_g_z = dg_polar + dg_hydrophobic + dg_headgroup + dg_cavity
    # Shift bulk water reference strictly to zero
    delta_g_z -= 0.5 * (delta_g_z[0] + delta_g_z[-1])

    # Position-dependent diffusivity D(z) in cm^2/s via Stokes-Einstein + microviscosity
    # Bulk water diffusivity ~ 5.5e-6 * (300 / MW)^{1/3} cm^2/s
    d_water = 5.5e-6 * ((300.0 / max(mol.molecular_weight, 80.0)) ** (1.0 / 3.0))
    # Viscous resistance increases in ordered acyl region (~12x) and drops slightly at center (~5x)
    viscosity_factor = 1.0 + 11.5 * ordered_acyl_bell + 4.0 * hydrocarbon_frac
    diffusivity_z = d_water / viscosity_factor

    # ISD integral: convert dz from Angstroms to cm (1 A = 1e-8 cm)
    dz_cm = (z_a[1] - z_a[0]) * 1e-8
    # Clamp exponent smoothly to avoid float overflow on extreme charged peptides
    exponent = np.clip(beta * delta_g_z, -40.0, 45.0)
    integrand_cm_inv = np.exp(exponent) / diffusivity_z
    resistance = float(np.trapezoid(integrand_cm_inv, dx=dz_cm))

    p_intrinsic = 1.0 / max(resistance, 1e-18)
    f_neutral_apical = compute_neutral_fraction(mol.pka_acid, mol.pka_base, ph_apical)
    f_neutral_cytosol = compute_neutral_fraction(mol.pka_acid, mol.pka_base, ph_cytosol)

    # Effective neutral-species permeation (plus minor paracellular / ion-pair floor)
    p_eff = max(1e-10, p_intrinsic * max(f_neutral_apical, 0.015))
    log10_peff = math.log10(p_eff)

    # Empirical P-gp efflux risk from TPSA, HBD, MW (bivalent efflux > 1 when TPSA > 90 & HBD >= 2)
    efflux_logit = 0.032 * (mol.tpsa - 88.0) + 0.55 * (mol.hbd - 1.5) + 0.004 * (mol.molecular_weight - 420.0)
    efflux_ratio = float(1.0 + math.exp(np.clip(efflux_logit, -6.0, 4.5)))
    p_app_ab = p_eff / math.sqrt(efflux_ratio)

    # Unbound intracellular accumulation ratio K_{p,uu}: pH-trapping * passive/efflux balance
    ph_trap = max(0.25, min(3.5, f_neutral_apical / max(f_neutral_cytosol, 1e-4)))
    k_puu = float(np.clip(ph_trap / efflux_ratio, 0.08, 2.5))

    if p_eff >= 1e-5:
        perm_class = "High"
    elif p_eff >= 1e-6:
        perm_class = "Moderate"
    else:
        perm_class = "Low"

    return PermeationProfile(
        z_grid_a=z_a,
        delta_g_z_kcal=delta_g_z,
        diffusivity_z_cm2_s=diffusivity_z,
        resistance_integrand=integrand_cm_inv,
        neutral_fraction=f_neutral_apical,
        p_eff_cm_s=float(p_eff),
        log10_p_eff=float(log10_peff),
        efflux_ratio=efflux_ratio,
        p_app_apical_to_basolateral=float(p_app_ab),
        k_puu_intracellular=k_puu,
        barrier_peak_kcal=float(np.max(delta_g_z)),
        permeability_class=perm_class,
    )
