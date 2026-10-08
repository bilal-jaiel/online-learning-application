"""
Metrics
--------
Helper functions to compute cumulative regret, cumulative reward,
and budget consumption from episode logs.
"""

import numpy as np


def compute_cumulative_reward(rewards):
    return np.cumsum(rewards)


def compute_cumulative_regret(rewards, oracle_rewards):
    """
    Pseudo-regret: cumulative gap between oracle and agent rewards.
    oracle_rewards : float or array of per-round oracle rewards.
    """
    oracle = np.full(len(rewards), oracle_rewards) \
        if np.isscalar(oracle_rewards) else np.array(oracle_rewards)
    return np.cumsum(oracle - np.array(rewards))


def compute_budget_consumption(costs, total_budget):
    cumcost = np.cumsum(costs)
    return cumcost / total_budget  # fraction of budget used


def compute_win_rate(won_flags):
    """won_flags: list of bool or 0/1"""
    arr = np.array(won_flags, dtype=float)
    return np.cumsum(arr) / (np.arange(len(arr)) + 1)


def summarize_run(log, total_budget):
    """
    log : list of step dicts returned by env.step()
    Returns a summary dict.
    """
    rewards = [s.get('total_reward', s.get('reward', 0)) for s in log]
    costs   = [s.get('total_cost',  s.get('cost', 0))   for s in log]
    wons    = [int(np.any(s.get('won', False))) for s in log]

    return {
        'total_reward':     sum(rewards),
        'total_cost':       sum(costs),
        'n_rounds':         len(log),
        'win_rate':         np.mean(wons),
        'budget_used_frac': sum(costs) / total_budget,
        'avg_reward':       np.mean(rewards),
        'rewards':          np.array(rewards),
        'costs':            np.array(costs),
    }
