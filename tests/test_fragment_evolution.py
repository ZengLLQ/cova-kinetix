"""Tests for NSGA-II non-dominated sorting, crowding distance, and covalent fragment evolution."""

from __future__ import annotations

import math

from cova_kinetix.fragment_evolution import (
    assign_crowding_distance,
    evolve_covalent_ligands,
    fast_non_dominated_sort,
)


def test_nsga2_non_dominated_sorting_and_crowding() -> None:
    objs = [
        (1.0, 5.0),
        (2.0, 2.0),
        (5.0, 1.0),
        (3.0, 3.0),  # dominated by (2.0, 2.0)
    ]
    fronts = fast_non_dominated_sort(objs)
    assert len(fronts) == 2
    assert set(fronts[0]) == {0, 1, 2}
    assert fronts[1] == [3]

    dist = assign_crowding_distance(fronts, objs)
    assert math.isinf(dist[0]) and math.isinf(dist[2])
    assert dist[1] > 0.0


def test_evolve_covalent_ligands_returns_ranked_pareto_candidates() -> None:
    cands = evolve_covalent_ligands(
        target_name="EGFR_C797S",
        warhead_pool=("acrylamide", "cyanoacrylamide", "nitrile", "vinyl_sulfone"),
        n_generations=2,
        population_size=10,
        seed=42,
    )
    assert len(cands) >= 6
    assert cands[0].pareto_rank == 0
    ranks = [c.pareto_rank for c in cands]
    assert ranks == sorted(ranks)
    assert all(c.k_inact_over_ki > 0.0 for c in cands)


def test_nsga2_handles_empty_objectives_gracefully() -> None:
    assert fast_non_dominated_sort([]) == []
    assert assign_crowding_distance([], []) == {}
