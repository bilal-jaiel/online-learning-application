"""
Random Policy
--------------
Baseline: pick a random bid uniformly from the bid grid at each round.
For multi-campaign: pick random feasible subset and random bids.
"""

import numpy as np
from utils.oracle import get_independent_sets


class RandomPolicy:
    """Single-campaign random bidder."""

    def __init__(self, bids, seed=None):
        self.bids = np.array(bids)
        self.rng = np.random.default_rng(seed)

    def select_bid(self, remaining_budget, **kwargs):
        feasible = self.bids[self.bids <= remaining_budget]
        if len(feasible) == 0:
            return 0.0
        return self.rng.choice(feasible)

    def update(self, bid, reward, cost, won, **kwargs):
        pass  # stateless


class RandomMultiPolicy:
    """Multi-campaign random bidder respecting conflict and budget constraints."""

    def __init__(self, N, bids, conflicts, seed=None):
        self.N = N
        self.bids = np.array(bids)
        self.conflicts = conflicts
        self.ind_sets = get_independent_sets(N, conflicts)
        self.rng = np.random.default_rng(seed)

    def select_bids(self, remaining_budget, **kwargs):
        bids_vector = np.zeros(self.N)

        # Pick a random independent set
        feasible_sets = [s for s in self.ind_sets if len(s) > 0]
        if not feasible_sets:
            return bids_vector

        subset = list(self.rng.choice(feasible_sets))  # random ind. set

        for camp in subset:
            feasible_bids = self.bids[self.bids <= remaining_budget]
            if len(feasible_bids) == 0:
                continue
            bids_vector[camp] = self.rng.choice(feasible_bids)
            remaining_budget -= bids_vector[camp]  # pessimistic budget check

        return bids_vector

    def update(self, bids_vector, rewards, costs_obs=None, won=None, **kwargs):
        pass
