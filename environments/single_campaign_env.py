"""
Single Campaign Stochastic Environment
---------------------------------------
First-price auction: if bid >= m_t, we win, pay our bid, get utility v - b.
Competing bid m_t is drawn i.i.d. from a fixed distribution each round.
"""

import numpy as np


class SingleCampaignEnv:
    """
    Parameters
    ----------
    value : float
        The valuation v of winning the campaign (utility = v - bid if win).
    budget : float
        Total budget for all T rounds.
    T : int
        Total number of rounds.
    dist : str
        Distribution for the competing bid. One of 'beta', 'uniform'.
    dist_params : tuple
        Parameters for the distribution (a, b) for Beta, or (low, high) for Uniform.
    seed : int, optional
    """

    def __init__(self, value=0.8, budget=200.0, T=2000,
                 dist='beta', dist_params=(2, 5), seed=None):
        self.value = value
        self.budget = budget
        self.T = T
        self.dist = dist
        self.dist_params = dist_params
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.remaining_budget = self.budget
        self.t = 0
        return self

    def _sample_competing_bid(self):
        if self.dist == 'beta':
            a, b = self.dist_params
            return self.rng.beta(a, b)
        elif self.dist == 'uniform':
            low, high = self.dist_params
            return self.rng.uniform(low, high)
        else:
            raise ValueError(f"Unknown distribution: {self.dist}")

    def step(self, bid: float):
        """
        Parameters
        ----------
        bid : float in [0, 1]

        Returns
        -------
        dict with keys: won, reward, cost, remaining_budget, competing_bid
        """
        assert self.t < self.T, "Episode is done. Call reset()."
        assert bid <= self.remaining_budget + 1e-9, "Bid exceeds remaining budget."

        m = self._sample_competing_bid()
        won = bid >= m
        cost = bid if won else 0.0
        reward = (self.value - bid) if won else 0.0

        self.remaining_budget -= cost
        self.t += 1

        return {
            'won': won,
            'reward': reward,
            'cost': cost,
            'remaining_budget': self.remaining_budget,
            'competing_bid': m,
        }

    @property
    def is_done(self):
        return self.t >= self.T or self.remaining_budget < 1e-9

    def __repr__(self):
        return (f"SingleCampaignEnv(v={self.value}, B={self.budget}, T={self.T}, "
                f"dist={self.dist}{self.dist_params})")
