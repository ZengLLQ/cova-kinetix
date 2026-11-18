"""
Multi-objective de novo covalent fragment evolution and NSGA-II Pareto optimization.

Combines BRICS-compatible hinge/back-pocket scaffold libraries, aryl/heteroaryl
exit-vector substituents, and canonical covalent warheads (`WARHEAD_TEMPLATES`),
filtering candidates by Extended Huckel Fukui reactivity bounds (GSH stability)
and ISD membrane permeability before ranking via NSGA-II non-dominated sorting
and crowding distance.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

from cova_kinetix.covalent_kinetics import barrier_to_eyring_rate
from cova_kinetix.fragment_graph import attach_fragment, graft_warhead
from cova_kinetix.membrane_permeation import simulate_membrane_permeation
from cova_kinetix.pocket_fields import align_covalent_ligand, build_target_pocket, solve_poisson_boltzmann_grid
from cova_kinetix.quantum_eht import compute_eht_activation_barrier, run_extended_huckel
from cova_kinetix.types import (
    WARHEAD_TEMPLATES,
    BindingPocket,
    EvolutionCandidate,
    PocketGrid,
)

# Hinge-binding & allosteric covalent scaffold templates with [1*] warhead exit vector
DEFAULT_HINGE_SCAFFOLDS: Tuple[str, ...] = (
    "[1*]c1cccc(Nc2ncnc3ccc(OC)cc23)c1",          # 4-anilinoquinazoline (EGFR-like)
    "[1*]c1cc(Nc2nccc(-c3cn(C)c4ccccc34)n2)ccc1OC",  # pyrimidine-indole (Osimertinib-like)
    "[1*]C1CCCN(c2nc(-c3ccc(Oc4ccccc4)cc3)c3c(N)ncnc3n2)C1",  # pyrazolopyrimidine-like
    "[1*]N1CCN(c2nc(OC)nc3c2cc(F)c(-c2ccccc2O)c3)CC1",        # switch-II piperazine
    "[1*]c1ccc(NC(=O)c2ccc(F)cc2)cc1",              # benzamide-phenyl scaffold
    "[1*]C(CC1CCNC1=O)NC(=O)c1cc(F)ccc1",           # peptidomimetic lactam scaffold
)

# Exit-vector decorator motifs that swap into a [1*] scaffold when building hybrid cores
DEFAULT_WARHEAD_CHOICES: Tuple[str, ...] = (
    "acrylamide",
    "dimethylaminomethyl_acrylamide",
    "cyanoacrylamide",
    "vinyl_sulfone",
    "nitrile",
    "alpha_ketoamide",
    "chloroacetamide",
)

DEFAULT_SUBSTITUENTS: Tuple[str, ...] = (
    "[1*]C",
    "[1*]OC",
    "[1*]F",
    "[1*]c1ccccc1",
    "[1*]CCN(C)C",
    "[1*]C(F)(F)F",
)


def _evaluate_candidate(
    scaffold_smiles: str,
    warhead_name: str,
    pocket: BindingPocket,
    pb_grid: PocketGrid,
    seed: int = 42,
) -> EvolutionCandidate:
    """Evaluate a single (scaffold, warhead) pair across EHT, pocket, kinetics, and ISD."""
    mol = graft_warhead(scaffold_smiles, warhead_name, name=f"evo_{warhead_name}", seed=seed)
    eht = run_extended_huckel(mol)
    docking = align_covalent_ligand(mol, pocket, pb_grid=pb_grid, mulliken_charges=eht.mulliken_charges)
    barrier = compute_eht_activation_barrier(
        eht,
        mol.warhead_class,
        nucleophile_pka=pocket.nucleophile_pka,
        physiological_ph=pocket.physiological_ph,
    )
    k_inact = barrier_to_eyring_rate(barrier)
    ki_m = max(docking.effective_ki_uM, 1e-5) * 1e-6
    k_inact_over_ki = k_inact / ki_m
    perm = simulate_membrane_permeation(mol)

    return EvolutionCandidate(
        smiles=mol.smiles,
        warhead_name=warhead_name,
        scaffold_smiles=scaffold_smiles,
        noncovalent_dg_kcal=docking.noncovalent_dg_kcal,
        k_inact_over_ki=k_inact_over_ki,
        log10_p_eff=perm.log10_p_eff,
        fukui_plus_max=eht.max_fukui_plus_heavy,
        gsh_t_half_min=eht.predicted_gsh_t_half_min,
        sa_score=mol.sa_score,
        pareto_rank=0,
        crowding_distance=0.0,
    )


def fast_non_dominated_sort(objectives: List[Tuple[float, ...]]) -> List[List[int]]:
    """
    Deb et al. (2002) NSGA-II fast non-dominated sorting over minimization objectives.
    """
    n = len(objectives)
    if n == 0:
        return []
    domination_sets: List[List[int]] = [[] for _ in range(n)]
    dom_counts = [0] * n
    fronts: List[List[int]] = [[]]

    for p in range(n):
        op = objectives[p]
        for q in range(p + 1, n):
            oq = objectives[q]
            p_le_q = all(a <= b for a, b in zip(op, oq))
            p_lt_q = any(a < b for a, b in zip(op, oq))
            q_le_p = all(b <= a for a, b in zip(op, oq))
            q_lt_p = any(b < a for a, b in zip(op, oq))
            if p_le_q and p_lt_q:
                domination_sets[p].append(q)
                dom_counts[q] += 1
            elif q_le_p and q_lt_p:
                domination_sets[q].append(p)
                dom_counts[p] += 1

    for p in range(n):
        if dom_counts[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front: List[int] = []
        for p in fronts[i]:
            for q in domination_sets[p]:
                dom_counts[q] -= 1
                if dom_counts[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    if not fronts[-1]:
        fronts.pop()
    return fronts


def assign_crowding_distance(
    fronts: List[List[int]],
    objectives: List[Tuple[float, ...]],
) -> Dict[int, float]:
    """Compute NSGA-II crowding distances within each Pareto front."""
    distances: Dict[int, float] = {}
    if not objectives:
        return distances
    n_obj = len(objectives[0])

    for front in fronts:
        for idx in front:
            distances[idx] = 0.0
        if len(front) <= 2:
            for idx in front:
                distances[idx] = float("inf")
            continue
        for m in range(n_obj):
            ordered = sorted(front, key=lambda idx: objectives[idx][m])
            f_min = objectives[ordered[0]][m]
            f_max = objectives[ordered[-1]][m]
            distances[ordered[0]] = float("inf")
            distances[ordered[-1]] = float("inf")
            span = f_max - f_min
            if span <= 1e-12:
                continue
            for k in range(1, len(ordered) - 1):
                if math.isinf(distances[ordered[k]]):
                    continue
                distances[ordered[k]] += (objectives[ordered[k + 1]][m] - objectives[ordered[k - 1]][m]) / span
    return distances


def evolve_covalent_ligands(
    target_name: str = "EGFR_C797S",
    seed_scaffolds: Sequence[str] = DEFAULT_HINGE_SCAFFOLDS,
    warhead_pool: Sequence[str] = DEFAULT_WARHEAD_CHOICES,
    n_generations: int = 4,
    population_size: int = 16,
    gsh_min_t_half_min: float = 25.0,
    seed: int = 42,
) -> List[EvolutionCandidate]:
    """
    Run multi-objective NSGA-II evolution over (scaffold, warhead) covalent designs.

    Minimization objectives per candidate:
    1. ``-log10(k_inact / K_i)`` (maximize second-order covalent efficiency)
    2. ``-log10(P_eff)`` (maximize transmembrane bilayer permeability)
    3. ``SA_score`` (minimize synthetic complexity)
    4. ``GSH hyper-reactivity penalty`` (penalize t1/2 < ``gsh_min_t_half_min``)

    Returns
    -------
    List[EvolutionCandidate]
        Final population sorted by Pareto rank (ascending) then crowding distance
        (descending), with rank 0 representing the non-dominated Pareto front.
    """
    rng = np.random.default_rng(seed)
    pocket = build_target_pocket(target_name)
    pb_grid = solve_poisson_boltzmann_grid(pocket, grid_n=19)

    valid_warheads = [w for w in warhead_pool if w in WARHEAD_TEMPLATES]
    if not valid_warheads:
        valid_warheads = ["acrylamide", "nitrile"]

    # Initial population sampling all scaffolds and warheads
    pairs: List[Tuple[str, str]] = []
    for s_smi in seed_scaffolds:
        for w_name in valid_warheads:
            pairs.append((s_smi, w_name))
    rng.shuffle(pairs)  # type: ignore[arg-type]
    initial_pairs = pairs[: max(population_size, 6)]

    cache: Dict[Tuple[str, str], EvolutionCandidate] = {}

    def _get_or_eval(s_smi: str, w_name: str, eval_seed: int) -> EvolutionCandidate:
        key = (s_smi, w_name)
        if key not in cache:
            cache[key] = _evaluate_candidate(s_smi, w_name, pocket, pb_grid, seed=eval_seed)
        return cache[key]

    population: List[EvolutionCandidate] = []
    for idx, (s_smi, w_name) in enumerate(initial_pairs):
        population.append(_get_or_eval(s_smi, w_name, seed + idx))

    def _objectives_for(cands: List[EvolutionCandidate]) -> List[Tuple[float, ...]]:
        objs: List[Tuple[float, ...]] = []
        for c in cands:
            log_eff = -math.log10(max(c.k_inact_over_ki, 1e-3))
            neg_perm = -c.log10_p_eff
            gsh_pen = max(0.0, (gsh_min_t_half_min - c.gsh_t_half_min) / 10.0)
            objs.append((log_eff, neg_perm, c.sa_score, gsh_pen))
        return objs

    for gen in range(max(1, n_generations)):
        # Generate offspring via warhead swap, scaffold crossover, and substituent grafting
        offspring: List[EvolutionCandidate] = []
        for k in range(len(population)):
            parent = population[int(rng.integers(0, len(population)))]
            op = int(rng.integers(0, 3))
            if op == 0:
                # Warhead swap mutation
                new_wh = valid_warheads[int(rng.integers(0, len(valid_warheads)))]
                offspring.append(_get_or_eval(parent.scaffold_smiles, new_wh, seed + gen * 100 + k))
            elif op == 1:
                # Scaffold swap + parent warhead crossover
                other_scaff = seed_scaffolds[int(rng.integers(0, len(seed_scaffolds)))]
                offspring.append(_get_or_eval(other_scaff, parent.warhead_name, seed + gen * 100 + k))
            else:
                # Grow scaffold with small substituent at exit vector if feasible, keeping warhead slot
                try:
                    sub = DEFAULT_SUBSTITUENTS[int(rng.integers(0, len(DEFAULT_SUBSTITUENTS)))]
                    # Append simple substituent decoration onto scaffold SMILES if exit vector exists
                    grown = (
                        attach_fragment(parent.scaffold_smiles, sub, embed_3d=False)
                        if "[2*]" in parent.scaffold_smiles
                        else None
                    )
                    scaff_use = grown.smiles if grown is not None else parent.scaffold_smiles
                except Exception:
                    scaff_use = parent.scaffold_smiles
                new_wh = valid_warheads[int(rng.integers(0, len(valid_warheads)))]
                offspring.append(_get_or_eval(scaff_use, new_wh, seed + gen * 100 + k))

        # Deduplicate combined pool by canonical SMILES
        merged_map: Dict[str, EvolutionCandidate] = {}
        for c in population + offspring:
            if c.smiles not in merged_map:
                merged_map[c.smiles] = c
        combined = list(merged_map.values())

        objs = _objectives_for(combined)
        fronts = fast_non_dominated_sort(objs)
        crowding = assign_crowding_distance(fronts, objs)

        next_pop: List[EvolutionCandidate] = []
        for rank, front in enumerate(fronts):
            for idx in front:
                combined[idx].pareto_rank = rank
                combined[idx].crowding_distance = crowding[idx]
            if len(next_pop) + len(front) <= population_size:
                next_pop.extend(combined[idx] for idx in front)
            else:
                needed = population_size - len(next_pop)
                front_sorted = sorted(front, key=lambda idx: crowding[idx], reverse=True)
                next_pop.extend(combined[idx] for idx in front_sorted[:needed])
                break
        population = next_pop

    final_objs = _objectives_for(population)
    final_fronts = fast_non_dominated_sort(final_objs)
    final_crowd = assign_crowding_distance(final_fronts, final_objs)
    for rank, front in enumerate(final_fronts):
        for idx in front:
            population[idx].pareto_rank = rank
            population[idx].crowding_distance = final_crowd[idx]

    population.sort(key=lambda c: (c.pareto_rank, -c.crowding_distance, -c.k_inact_over_ki))
    return population
