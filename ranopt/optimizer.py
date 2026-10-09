"""Genetic algorithm that minimises network energy under QoS constraints.

Decision variables per cell: transmit power, electrical downtilt, sleep on/off.
Objective: minimise total input power relative to the baseline, with penalties
if coverage or throughput fall below the allowed degradation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .model import (Evaluation, KPIs, ModelParams, NetworkState, baseline_state,
                    evaluate)
from .scenario import Scenario


@dataclass
class GAConfig:
    pop_size: int = 80
    generations: int = 120
    elite: int = 2
    tournament: int = 3
    crossover_rate: float = 0.9
    mutation_rate: float = 0.12
    power_sigma_db: float = 1.5
    tilt_sigma_deg: float = 1.0
    sleep_flip_prob: float = 0.02
    power_min_dbm: float = 30.0
    power_max_dbm: float = 46.0
    tilt_min_deg: float = 0.0
    tilt_max_deg: float = 14.0
    max_coverage_drop: float = 0.01      # absolute (0.01 = 1 percentage point)
    max_edge_drop: float = 0.15          # relative (0.15 = 15 %)
    max_mean_drop: float = 0.20          # relative drop of mean user throughput
    seed: int = 0


@dataclass
class OptResult:
    best: NetworkState
    baseline: NetworkState
    best_eval: Evaluation
    baseline_eval: Evaluation
    history: dict = field(default_factory=dict)


def fitness(k: KPIs, base: KPIs, ga: GAConfig) -> float:
    cov_short = max(0.0, base.coverage - ga.max_coverage_drop - k.coverage)
    edge_short = max(0.0, base.edge_tput_mbps * (1.0 - ga.max_edge_drop) - k.edge_tput_mbps)
    edge_short /= max(base.edge_tput_mbps, 1e-9)
    mean_short = max(0.0, base.mean_tput_mbps * (1.0 - ga.max_mean_drop) - k.mean_tput_mbps)
    mean_short /= max(base.mean_tput_mbps, 1e-9)
    return (k.power_w / base.power_w + 100.0 * cov_short
            + 5.0 * edge_short + 5.0 * mean_short)


def _polish(scn, params, ga, state, base_k, best_fit) -> tuple[NetworkState, float]:
    """Greedy refinement: lower each cell's power in 1 dB steps while feasible."""
    rng = np.random.default_rng(ga.seed + 1)
    state = state.copy()
    for c in rng.permutation(scn.n_cells):
        if not state.active[c]:
            continue
        while state.power_dbm[c] - 1.0 >= ga.power_min_dbm:
            trial = state.copy()
            trial.power_dbm[c] -= 1.0
            f = fitness(evaluate(scn, params, trial).kpis, base_k, ga)
            if f < best_fit - 1e-12:
                state, best_fit = trial, f
            else:
                break
    return state, best_fit


def optimize(scn: Scenario, params: ModelParams | None = None,
             ga: GAConfig | None = None,
             progress: Callable[[int, int], None] | None = None) -> OptResult:
    params = params or ModelParams()
    ga = ga or GAConfig()
    rng = np.random.default_rng(ga.seed)
    c = scn.n_cells

    base = baseline_state(scn, params)
    base_eval = evaluate(scn, params, base)
    base_k = base_eval.kpis

    pop_p = np.clip(base.power_dbm + rng.normal(0, 3.0, (ga.pop_size, c)),
                    ga.power_min_dbm, ga.power_max_dbm)
    pop_t = np.clip(base.tilt_deg + rng.normal(0, 2.0, (ga.pop_size, c)),
                    ga.tilt_min_deg, ga.tilt_max_deg)
    pop_a = rng.random((ga.pop_size, c)) > 0.05
    pop_p[0], pop_t[0], pop_a[0] = base.power_dbm, base.tilt_deg, base.active

    def score(i):
        st = NetworkState(pop_p[i], pop_t[i], pop_a[i])
        ev = evaluate(scn, params, st)
        return fitness(ev.kpis, base_k, ga), ev.kpis

    fits = np.empty(ga.pop_size)
    kp = [None] * ga.pop_size
    for i in range(ga.pop_size):
        fits[i], kp[i] = score(i)

    hist = {"best_fitness": [], "mean_fitness": [], "best_power_ratio": [],
            "best_coverage": []}

    for gen in range(ga.generations):
        order = np.argsort(fits)
        hist["best_fitness"].append(float(fits[order[0]]))
        hist["mean_fitness"].append(float(fits.mean()))
        hist["best_power_ratio"].append(kp[order[0]].power_w / base_k.power_w)
        hist["best_coverage"].append(kp[order[0]].coverage)
        if progress:
            progress(gen + 1, ga.generations)

        new_p, new_t, new_a = [pop_p[order[:ga.elite]]], [pop_t[order[:ga.elite]]], [pop_a[order[:ga.elite]]]
        n_child = ga.pop_size - ga.elite
        cp, ct, ca = np.empty((n_child, c)), np.empty((n_child, c)), np.empty((n_child, c), bool)
        for j in range(n_child):
            a = min(rng.integers(0, ga.pop_size, ga.tournament), key=lambda i: fits[i])
            b = min(rng.integers(0, ga.pop_size, ga.tournament), key=lambda i: fits[i])
            if rng.random() < ga.crossover_rate:
                mask = rng.random(c) < 0.5
            else:
                mask = np.ones(c, bool)
            cp[j] = np.where(mask, pop_p[a], pop_p[b])
            ct[j] = np.where(mask, pop_t[a], pop_t[b])
            ca[j] = np.where(mask, pop_a[a], pop_a[b])
            m = rng.random(c) < ga.mutation_rate
            cp[j] += m * rng.normal(0, ga.power_sigma_db, c)
            ct[j] += m * rng.normal(0, ga.tilt_sigma_deg, c)
            ca[j] ^= rng.random(c) < ga.sleep_flip_prob
            if not ca[j].any():
                ca[j][rng.integers(c)] = True
        pop_p = np.vstack(new_p + [np.clip(cp, ga.power_min_dbm, ga.power_max_dbm)])
        pop_t = np.vstack(new_t + [np.clip(ct, ga.tilt_min_deg, ga.tilt_max_deg)])
        pop_a = np.vstack(new_a + [ca])
        for i in range(ga.pop_size):
            fits[i], kp[i] = score(i)

    i_best = int(np.argmin(fits))
    best = NetworkState(pop_p[i_best].copy(), pop_t[i_best].copy(), pop_a[i_best].copy())
    best, _ = _polish(scn, params, ga, best, base_k, float(fits[i_best]))
    best_eval = evaluate(scn, params, best, maps=True)
    base_eval = evaluate(scn, params, base, maps=True)
    return OptResult(best=best, baseline=base, best_eval=best_eval,
                     baseline_eval=base_eval, history=hist)
