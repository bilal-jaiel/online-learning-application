"""
Multi-Campaign Stochastic Environment
---------------------------------------
N campaigns, each with its own valuation and competing bid distribution.
Conflict graph: incompatible campaigns cannot both be played in the same round.
Budget constraint: total budget shared across all campaigns and all rounds.
"""

import numpy as np


class MultiCampaignEnv:
    """
    Parameters
    ----------
    values : array-like of shape (N,)
        Valuation v_i for each campaign.
    budget : float
        Total budget for all T rounds.
    T : int
        Total number of rounds.
    dists : list of str
        Distribution name for each campaign ('beta' or 'uniform').
    dist_params : list of tuple
        Distribution parameters for each campaign.
    conflict_edges : list of tuple (i, j)
        Pairs of campaigns that are mutually exclusive per round.
    seed : int, optional
    """

    def __init__(self, values, budget, T, dists, dist_params,
                 conflict_edges=None, seed=None):
        self.values = np.array(values, dtype=float)
        self.N = len(self.values)
        self.budget = float(budget)
        self.T = T
        self.dists = dists
        self.dist_params = dist_params
        self.conflict_edges = conflict_edges or []
        self.rng = np.random.default_rng(seed)

        # Build adjacency set for fast conflict lookup
        self.conflicts = {i: set() for i in range(self.N)}
        for i, j in self.conflict_edges:
            self.conflicts[i].add(j)
            self.conflicts[j].add(i)

        self.reset()

    def reset(self):
        self.remaining_budget = self.budget
        self.t = 0
        return self

    def _sample_competing_bids(self):
        m = np.zeros(self.N)
        for i in range(self.N):
            if self.dists[i] == 'beta':
                a, b = self.dist_params[i]
                m[i] = self.rng.beta(a, b)
            elif self.dists[i] == 'uniform':
                low, high = self.dist_params[i]
                m[i] = self.rng.uniform(low, high)
        return m

    def is_feasible(self, selected_campaigns):
        """
        Check if the set of selected campaigns violates no conflict constraint.
        selected_campaigns : list of campaign indices
        """
        selected = set(selected_campaigns)
        for i in selected:
            if self.conflicts[i] & selected:
                return False
        return True

    def step(self, bids_vector):
        """
        Parameters
        ----------
        bids_vector : array of shape (N,)
            bid[i] > 0 means we participate in campaign i with that bid.
            bid[i] = 0 means we skip campaign i.

        Returns
        -------
        dict with per-campaign and aggregate info.
        """
        assert self.t < self.T, "Episode done. Call reset()."
        bids_vector = np.array(bids_vector, dtype=float)
        assert len(bids_vector) == self.N

        # Validate conflict constraints
        active = np.where(bids_vector > 0)[0].tolist()
        assert self.is_feasible(active), "Conflict constraint violated!"

        # Validate budget
        total_possible_cost = bids_vector[active].sum()
        # (We check actual cost after outcomes)

        m = self._sample_competing_bids()

        won = np.zeros(self.N, dtype=bool)
        rewards = np.zeros(self.N)
        costs = np.zeros(self.N)

        for i in active:
            if bids_vector[i] >= m[i]:
                won[i] = True
                costs[i] = bids_vector[i]
                rewards[i] = self.values[i] - bids_vector[i]

        total_cost = costs.sum()
        total_reward = rewards.sum()

        self.remaining_budget -= total_cost
        self.t += 1

        return {
            'won': won,                          # bool array (N,)
            'rewards': rewards,                  # float array (N,)
            'costs': costs,                      # float array (N,)
            'total_reward': total_reward,
            'total_cost': total_cost,
            'remaining_budget': self.remaining_budget,
            'competing_bids': m,                 # float array (N,)
        }

    @property
    def is_done(self):
        return self.t >= self.T or self.remaining_budget < 1e-9

    def __repr__(self):
        return (f"MultiCampaignEnv(N={self.N}, B={self.budget}, T={self.T}, "
                f"conflicts={self.conflict_edges})")
