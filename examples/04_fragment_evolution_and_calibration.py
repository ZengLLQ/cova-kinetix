"""Example 04: Multi-objective covalent fragment evolution & Sobol sensitivity."""

from __future__ import annotations

from cova_kinetix.calibration import calibrate_kinetics_and_sensitivity
from cova_kinetix.fragment_evolution import evolve_covalent_ligands
from cova_kinetix.visualizer import render_calibration_sobol, render_evolution_pareto


def main() -> None:
    """Run NSGA-II covalent scaffold-warhead evolution and Saltelli Sobol sensitivity."""
    candidates = evolve_covalent_ligands(
        target_name="BTK_C481",
        warhead_pool=("acrylamide", "cyanoacrylamide", "nitrile", "vinyl_sulfone", "alpha_ketoamide"),
        n_generations=3,
        population_size=12,
        seed=42,
    )
    cal = calibrate_kinetics_and_sensitivity(n_bootstrap=16, n_sobol_samples=32, seed=42)
    print(render_evolution_pareto(candidates, max_rows=8))
    print()
    print(render_calibration_sobol(cal))


if __name__ == "__main__":
    main()
