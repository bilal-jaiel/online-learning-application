"""
Experiment 3: Non-Stationary Environment
-------------------------------------------
3 phases of competition: low → medium → high.
Compares:
  - Combinatorial-UCB (baseline, no adaptation)
  - Primal-Dual Pacing
  - Sliding-Window UCB
  - Change-Detection UCB
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from tqdm import tqdm

from environments.non_stationary_env  import NonStationaryMultiCampaignEnv
from algorithms.combinatorial_ucb     import CombinatorialUCB
from algorithms.primal_dual           import PrimalDualPacing
from algorithms.sliding_window_ucb    import SlidingWindowUCB
from algorithms.change_detection_ucb  import ChangeDetectionUCB
from utils.metrics                    import summarize_run
from utils.plots                      import (plot_cumulative_rewards,
                                               plot_cumulative_regret,
                                               plot_budget_consumption,
                                               plot_comparison_table)

# ─── Configuration ────────────────────────────────────────────────────────────
T      = 5000
BUDGET = 1000.0
BIDS   = np.linspace(0, 1, 11)
SEED   = 42

VALUES = np.array([0.9, 0.75, 0.6, 0.8, 0.7])
N      = len(VALUES)

CONFLICT_EDGES = [(0, 1), (2, 3)]
CONFLICTS = {i: set() for i in range(N)}
for i, j in CONFLICT_EDGES:
    CONFLICTS[i].add(j)
    CONFLICTS[j].add(i)

# Phase boundaries
PHASE_LENGTHS = [T // 3, T // 3, T - 2 * (T // 3)]
PHASE_BOUNDARIES = [PHASE_LENGTHS[0], PHASE_LENGTHS[0] + PHASE_LENGTHS[1]]

# Phase 1: low competition (easy to win with low bids)
PHASE1 = {
    'length': PHASE_LENGTHS[0],
    'dists':  ['beta'] * N,
    'dist_params': [(2, 5)] * N,   # competing bids tend to be low
}
# Phase 2: medium competition
PHASE2 = {
    'length': PHASE_LENGTHS[1],
    'dists':  ['beta'] * N,
    'dist_params': [(3, 3)] * N,   # symmetric → medium
}
# Phase 3: high competition (must bid high to win)
PHASE3 = {
    'length': PHASE_LENGTHS[2],
    'dists':  ['beta'] * N,
    'dist_params': [(5, 2)] * N,   # competing bids tend to be high
}

PHASES = [PHASE1, PHASE2, PHASE3]

# Oracle per phase (approximate)
from scipy.stats import beta as beta_dist

def phase_oracle(values, bids, dist_params, conflicts, conflict_edges):
    from utils.oracle import get_independent_sets
    ind_sets = get_independent_sets(len(values), conflicts)
    best_total = 0.0
    for subset in ind_sets:
        total = 0.0
        for i in subset:
            a, b = dist_params[i]
            p_win = beta_dist.cdf(bids, a, b)
            r = (values[i] - bids) * p_win
            total += r.max()
        best_total = max(best_total, total)
    return best_total

phase_oracles = [
    phase_oracle(VALUES, BIDS, p['dist_params'], CONFLICTS, CONFLICT_EDGES)
    for p in PHASES
]
ORACLE_PER_ROUND = np.array([
    phase_oracles[0]] * PHASE_LENGTHS[0] +
    [phase_oracles[1]] * PHASE_LENGTHS[1] +
    [phase_oracles[2]] * PHASE_LENGTHS[2]
)
print(f"Oracle/round per phase: {[f'{x:.4f}' for x in phase_oracles]}")

# ─── Run helper ───────────────────────────────────────────────────────────────
def run_ns(algo, env_kwargs):
    env = NonStationaryMultiCampaignEnv(**env_kwargs)
    env.reset()
    log = []

    for t in tqdm(range(T), desc=type(algo).__name__, leave=False):
        if env.is_done:
            for _ in range(T - t):
                log.append({'total_reward': 0.0, 'total_cost': 0.0,
                             'won': np.zeros(N, dtype=bool),
                             'rewards': np.zeros(N), 'costs': np.zeros(N)})
            break

        bids_vec = algo.select_bids(remaining_budget=env.remaining_budget,
                                     remaining_rounds=T - t)
        result   = env.step(bids_vec)
        algo.update(bids_vector=bids_vec,
                    rewards=result['rewards'],
                    costs_obs=result['costs'],
                    won=result['won'])
        log.append(result)

    return log

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    np.random.seed(SEED)  # UCB tie-breaking uses the global generator
    env_kwargs = dict(values=VALUES, budget=BUDGET, T=T,
                      phases=PHASES, conflict_edges=CONFLICT_EDGES, seed=SEED)

    algos = {
        'Comb-UCB':     CombinatorialUCB(N, BIDS, CONFLICTS, BUDGET, T),
        'Primal-Dual':  PrimalDualPacing(N, BIDS, CONFLICTS, BUDGET, T),
        'SW-UCB (W=300)': SlidingWindowUCB(N, BIDS, CONFLICTS, BUDGET, T, W=300),
        'CD-UCB':       ChangeDetectionUCB(N, BIDS, CONFLICTS, BUDGET, T,
                                            epsilon=0.05, h=0.5, min_samples=30),
    }

    rewards_dict = {}
    costs_dict   = {}
    summaries    = {}

    for name, algo in algos.items():
        log = run_ns(algo, env_kwargs)
        summary = summarize_run(log, BUDGET)
        rewards_dict[name] = summary['rewards']
        costs_dict[name]   = summary['costs']
        summaries[name]    = summary
        print(f"{name:20s} | reward={summary['total_reward']:.2f} "
              f"| budget_used={summary['budget_used_frac']:.2%} "
              f"| win_rate={summary['win_rate']:.2%}")

    plot_cumulative_rewards(rewards_dict,
                            title='Cumulative Reward (Non-Stationary)',
                            filename='exp3_cumulative_reward.png',
                            phase_boundaries=PHASE_BOUNDARIES)
    plot_cumulative_regret(rewards_dict, ORACLE_PER_ROUND,
                           title='Cumulative Regret (Non-Stationary)',
                           filename='exp3_cumulative_regret.png',
                           phase_boundaries=PHASE_BOUNDARIES)
    plot_budget_consumption(costs_dict, BUDGET,
                            title='Budget Consumption (Non-Stationary)',
                            filename='exp3_budget_consumption.png',
                            phase_boundaries=PHASE_BOUNDARIES)
    plot_comparison_table(summaries, filename='exp3_comparison_table.png')
    print("\n[Exp 3] Done. Plots saved to results/")


if __name__ == '__main__':
    main()
