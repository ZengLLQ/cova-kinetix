"""Example 05: Multi-target covalent inhibitor vs. decoy screening benchmark."""

from __future__ import annotations

from cova_kinetix.benchmark import run_benchmark_suite
from cova_kinetix.visualizer import render_benchmark_table


def main() -> None:
    """Run covalent virtual screening benchmark on EGFR_C797S and MPRO_C145."""
    for target in ("EGFR_C797S", "MPRO_C145"):
        report = run_benchmark_suite(target_name=target, seed=42)
        print(render_benchmark_table(report))
        print()


if __name__ == "__main__":
    main()
