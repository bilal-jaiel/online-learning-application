"""
Oracle / Combinatorial Solver
-------------------------------
Given UCB scores (or any scores) per (campaign, bid) pair,
find the best feasible superarm:
  - At most one bid per campaign
  - No conflicting campaigns selected together
  - Total expected cost <= remaining_budget

For N=5 campaigns this brute-force is fast enough.
For larger N, replace with a greedy or LP-based solver.
"""

import numpy as np
from itertools import product


def get_independent_sets(N, conflicts):
    """
    Enumerate all independent sets (subsets with no conflict edges).
    conflicts : dict {i: set of j incompatible with i}
    Returns list of frozensets.
    """
    all_campaigns = list(range(N))
    independent_sets = []

    def is_independent(subset):
        s = set(subset)
        for i in s:
            if conflicts[i] & s:
                return False
        return True

    # Power set (2^N subsets), feasible for N <= 15
    for mask in range(1, 1 << N):
        subset = [i for i in range(N) if mask & (1 << i)]
        if is_independent(subset):
            independent_sets.append(frozenset(subset))

    return independent_sets


def oracle_select(scores, costs, remaining_budget, conflicts, bids):
    """
    Select the superarm (campaign subset + bid per campaign) maximising
    sum of scores subject to budget and conflict constraints.

    Parameters
    ----------
    scores : np.ndarray of shape (N, K)
        scores[i, k] = score of playing bid bids[k] on campaign i.
    costs : np.ndarray of shape (N, K)
        costs[i, k] = expected cost of playing bid bids[k] on campaign i.
    remaining_budget : float
    conflicts : dict {i: set of j}
    bids : np.ndarray of shape (K,)

    Returns
    -------
    bids_vector : np.ndarray of shape (N,)
        bids_vector[i] = chosen bid for campaign i (0 = skip).
    best_score : float
    """
    N, K = scores.shape
    best_score = -np.inf
    best_bids = np.zeros(N)

    # Enumerate independent sets
    ind_sets = get_independent_sets(N, conflicts)

    for subset in ind_sets:
        subset = list(subset)
        if not subset:
            continue

        # Skip bid=0 (index 0): it can never win a first-price auction
        valid_bid_indices = list(range(1, K)) if K > 1 else list(range(K))

        if len(subset) <= 6:
            # Exact enumeration over bid choices
            for bid_combo in product(valid_bid_indices, repeat=len(subset)):
                total_cost = sum(costs[subset[j], bid_combo[j]]
                                 for j in range(len(subset)))
                if total_cost > remaining_budget + 1e-9:
                    continue
                total_score = sum(scores[subset[j], bid_combo[j]]
                                  for j in range(len(subset)))
                if total_score > best_score:
                    best_score = total_score
                    bv = np.zeros(N)
                    for j, camp in enumerate(subset):
                        bv[camp] = bids[bid_combo[j]]
                    best_bids = bv
        else:
            # Greedy fallback for large subsets
            bv = np.zeros(N)
            budget_left = remaining_budget
            score_sum = 0.0
            for camp in subset:
                best_k = -1
                best_s = -np.inf
                for k in range(K):
                    if costs[camp, k] <= budget_left and scores[camp, k] > best_s:
                        best_s = scores[camp, k]
                        best_k = k
                if best_k >= 0:
                    bv[camp] = bids[best_k]
                    budget_left -= costs[camp, best_k]
                    score_sum += best_s
            if score_sum > best_score:
                best_score = score_sum
                best_bids = bv

    return best_bids, best_score
