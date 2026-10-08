"""
Experiment 1: Single Campaign, Stochastic Environment
--------------------------------------------------------
Compares:
  - Random baseline
  - Greedy
  - UCB1 (no budget)
  - BudgetedUCB1
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from tqdm import tqdm

from environments.single_campaign_env import SingleCampaignEnv
from algorithms.random_policy         import RandomPolicy
from algorithms.greedy                import GreedyPolicy
from algorithms.ucb1                  import UCB1
from algorithms.budgeted_ucb          import BudgetedUCB1
from utils.metrics                    import summarize_run
from utils.plots                      import (plot_cumulative_rewards,
                                               plot_cumulative_regret,
                                               plot_budget_consumption,
                                               plot_comparison_table)

# ─── Configuration ────────────────────────────────────────────────────────────
T      = 2000
BUDGET = 400.0
VALUE  = 0.8
BIDS   = np.linspace(0, 1, 11)
SEED   = 42

# Expected oracle reward: best fixed bid in hindsight
# For Beta(2,5): P(win|b) = CDF_beta(b) → reward = (v-b)*P(win|b)
# We approximate the oracle as the best expected-reward bid.
from scipy.stats import beta as beta_dist

def oracle_expected_reward(v, bids, a, b):
    """Expected reward per round for each bid under Beta(a,b) competing bids."""
    p_win = beta_dist.cdf(bids, a, b)
    return (v - bids) * p_win

oracle_r = oracle_expected_reward(VALUE, BIDS, 2, 5)
ORACLE_PER_ROUND = oracle_r.max()
print(f"Oracle best bid: {BIDS[np.argmax(oracle_r)]:.2f} "
      f"| Expected reward/round: {ORACLE_PER_ROUND:.4f}")

# ─── Run helper ───────────────────────────────────────────────────────────────
def run_single(algo, env):
    env.reset()
    log = []
    remaining = env.budget

    for t in tqdm(range(T), desc=type(algo).__name__, leave=False):
        if env.is_done:
            # Pad remaining rounds with 0 reward
            for _ in range(T - t):
                log.append({'reward': 0.0, 'cost': 0.0, 'won': False,
                             'total_reward': 0.0, 'total_cost': 0.0})
            break

        kwargs = dict(remaining_budget=env.remaining_budget,
                      remaining_rounds=T - t)

        if hasattr(algo, 'select_bid'):
            bid = algo.select_bid(**kwargs)
        else:
            bid = 0.0

        # Clip bid to feasible range
        bid = min(bid, env.remaining_budget)
        bid = max(bid, 0.0)

        if bid > 1e-9:
            result = env.step(bid)
            if hasattr(algo, 'update'):
                algo.update(bid=bid, reward=result['reward'],
                            cost=result['cost'], won=result['won'])
        else:
            # Skip this round: still counts as a round
            env.t += 1
            result = {
                'won': False, 'reward': 0.0, 'cost': 0.0,
                'remaining_budget': env.remaining_budget,
                'competing_bid': 0.0,
            }

        result['total_reward'] = result['reward']
        result['total_cost']   = result['cost']
        log.append(result)

    return log

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    np.random.seed(SEED)  # UCB tie-breaking uses the global generator
    env_kwargs = dict(value=VALUE, budget=BUDGET, T=T,
                      dist='beta', dist_params=(2, 5), seed=SEED)

    algos = {
        'Random':       RandomPolicy(BIDS, seed=SEED),
        'Greedy':       GreedyPolicy(BIDS),
        'UCB1':         UCB1(BIDS),
        'BudgetedUCB1': BudgetedUCB1(BIDS, BUDGET, T, alpha=1.0),
    }

    rewards_dict = {}
    costs_dict   = {}
    summaries    = {}

    for name, algo in algos.items():
        env = SingleCampaignEnv(**env_kwargs)
        log = run_single(algo, env)
        summary = summarize_run(log, BUDGET)
        rewards_dict[name] = summary['rewards']
        costs_dict[name]   = summary['costs']
        summaries[name]    = summary
        print(f"{name:15s} | reward={summary['total_reward']:.2f} "
              f"| budget_used={summary['budget_used_frac']:.2%} "
              f"| win_rate={summary['win_rate']:.2%}")

    plot_cumulative_rewards(rewards_dict,
                            title='Cumulative Reward (Single Campaign)',
                            filename='exp1_cumulative_reward.png')
    plot_cumulative_regret(rewards_dict, ORACLE_PER_ROUND,
                           title='Cumulative Regret (Single Campaign)',
                           filename='exp1_cumulative_regret.png')
    plot_budget_consumption(costs_dict, BUDGET,
                            title='Budget Consumption (Single Campaign)',
                            filename='exp1_budget_consumption.png')
    plot_comparison_table(summaries, filename='exp1_comparison_table.png')
    print("\n[Exp 1] Done. Plots saved to results/")


if __name__ == '__main__':
    main()
