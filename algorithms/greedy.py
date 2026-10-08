"""
Greedy Policy
--------------
Single-campaign: always play the bid with the highest empirical mean reward.
Multi-campaign: pick best bid per campaign (by empirical mean), then use oracle.
"""

import numpy as np
from utils.oracle import oracle_select


class GreedyPolicy:
    """Single-campaign greedy bidder."""

    def __init__(self, bids):
        self.bids = np.array(bids)
        K = len(bids)
        self.counts = np.zeros(K)
        self.means  = np.zeros(K)

    def select_bid(self, remaining_budget, **kwargs):
        feasible_mask = (self.bids <= remaining_budget) & (self.bids > 1e-9)
        if not feasible_mask.any():
            return 0.0
        scores = np.where(feasible_mask, self.means, -np.inf)
        return self.bids[np.argmax(scores)]

    def update(self, bid, reward, cost, won, **kwargs):
        k = np.argmin(np.abs(self.bids - bid))
        self.counts[k] += 1
        self.means[k] += (reward - self.means[k]) / self.counts[k]


class GreedyMultiPolicy:
    """Multi-campaign greedy bidder."""

    def __init__(self, N, bids, conflicts):
        self.N = N
        self.bids = np.array(bids)
        K = len(bids)
        self.conflicts = conflicts
        # mean_reward[i, k] and mean_cost[i, k]
        self.counts      = np.zeros((N, K))
        self.mean_reward = np.zeros((N, K))
        self.mean_cost   = np.zeros((N, K))

    def select_bids(self, remaining_budget, **kwargs):
        scores = self.mean_reward.copy()
        costs  = self.mean_cost.copy()
        # Initialise costs to bid values before any observation
        for k, b in enumerate(self.bids):
            for i in range(self.N):
                if self.counts[i, k] == 0:
                    costs[i, k] = b  # assume we win = pay b

        bids_vec, _ = oracle_select(scores, costs, remaining_budget,
                                    self.conflicts, self.bids)
        return bids_vec

    def update(self, bids_vector, rewards, costs_obs, won, **kwargs):
        for i in range(self.N):
            if bids_vector[i] > 0:
                k = np.argmin(np.abs(self.bids - bids_vector[i]))
                self.counts[i, k] += 1
                n = self.counts[i, k]
                self.mean_reward[i, k] += (rewards[i] - self.mean_reward[i, k]) / n
                self.mean_cost[i, k]   += (costs_obs[i] - self.mean_cost[i, k]) / n
