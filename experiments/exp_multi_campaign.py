"""
Experiment 2: Multi-Campaign, Stochastic Environment
-------------------------------------------------------
Compares:
  - Random multi-policy
  - Greedy multi-policy
  - Combinatorial-UCB
  - Primal-Dual Pacing
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from tqdm import tqdm

from environments.multi_campaign_env  import MultiCampaignEnv
from algorithms.random_policy         import RandomMultiPolicy
from algorithms.greedy                import GreedyMultiPolicy
from algorithms.combinatorial_ucb     import CombinatorialUCB
from algorithms.primal_dual           import PrimalDualPacing
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

DISTS       = ['beta', 'beta', 'uniform', 'beta', 'beta']
DIST_PARAMS = [(2, 5), (5, 2), (0, 1), (3, 3), (2, 2)]

CONFLICT_EDGES = [(0, 1), (2, 3)]   # Camp 0 vs 1, Camp 2 vs 3; Camp 4 free

CONFLICTS = {i: set() for i in range(N)}
for i, j in CONFLICT_EDGES:
    CONFLICTS[i].add(j)
    CONFLICTS[j].add(i)

# ─── Oracle approximation ─────────────────────────────────────────────────────
from scipy.stats import beta as beta_dist

def best_expected_reward_per_campaign(values, bids, dists, dist_params):
    """Best expected reward per campaign under its distribution."""
    best = []
    for i in range(len(values)):
        if dists[i] == 'beta':
            a, b = dist_params[i]
            p_win = beta_dist.cdf(bids, a, b)
        else:
            low, high = dist_params[i]
            p_win = np.clip((bids - low) / (high - low), 0, 1)
        r = (values[i] - bids) * p_win
        best.append(r.max())
    return best

best_per_campaign = best_expected_reward_per_campaign(VALUES, BIDS, DISTS, DIST_PARAMS)
# Oracle selects best independent set and their optimal bids
# Camp 4 always included (no conflict)
# Between 0 and 1: pick best
# Between 2 and 3: pick best
oracle_camps = [
    np.argmax([best_per_campaign[0], best_per_campaign[1]]),     # 0 or 1
    2 + np.argmax([best_per_campaign[2], best_per_campaign[3]]), # 2 or 3
    4,
]
ORACLE_PER_ROUND = sum(best_per_campaign[c] for c in oracle_camps)
print(f"Oracle campaigns: {oracle_camps}")
print(f"Oracle expected reward/round: {ORACLE_PER_ROUND:.4f}")

# ─── Run helper ───────────────────────────────────────────────────────────────
def run_multi(algo, env):
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
                    won=result['won'],
                    competing_bids=result['competing_bids'])
        log.append(result)

    return log

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    np.random.seed(SEED)  # UCB tie-breaking uses the global generator
    env_kwargs = dict(values=VALUES, budget=BUDGET, T=T,
                      dists=DISTS, dist_params=DIST_PARAMS,
                      conflict_edges=CONFLICT_EDGES, seed=SEED)

    algos = {
        'Random':       RandomMultiPolicy(N, BIDS, CONFLICTS, seed=SEED),
        'Greedy':       GreedyMultiPolicy(N, BIDS, CONFLICTS),
        'Comb-UCB':     CombinatorialUCB(N, BIDS, CONFLICTS, BUDGET, T),
        'Primal-Dual':  PrimalDualPacing(N, BIDS, CONFLICTS, BUDGET, T),
    }

    rewards_dict = {}
    costs_dict   = {}
    summaries    = {}

    for name, algo in algos.items():
        env = MultiCampaignEnv(**env_kwargs)
        log = run_multi(algo, env)
        summary = summarize_run(log, BUDGET)
        rewards_dict[name] = summary['rewards']
        costs_dict[name]   = summary['costs']
        summaries[name]    = summary
        print(f"{name:15s} | reward={summary['total_reward']:.2f} "
              f"| budget_used={summary['budget_used_frac']:.2%} "
              f"| win_rate={summary['win_rate']:.2%}")

    plot_cumulative_rewards(rewards_dict,
                            title='Cumulative Reward (Multi-Campaign)',
                            filename='exp2_cumulative_reward.png')
    plot_cumulative_regret(rewards_dict, ORACLE_PER_ROUND,
                           title='Cumulative Regret (Multi-Campaign)',
                           filename='exp2_cumulative_regret.png')
    plot_budget_consumption(costs_dict, BUDGET,
                            title='Budget Consumption (Multi-Campaign)',
                            filename='exp2_budget_consumption.png')
    plot_comparison_table(summaries, filename='exp2_comparison_table.png')
    print("\n[Exp 2] Done. Plots saved to results/")


if __name__ == '__main__':
    main()
