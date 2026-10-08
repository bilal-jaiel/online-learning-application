"""
Budgeted UCB1: Single Campaign
---------------------------------
Same as UCB1 but:
  1. Only considers bids that the remaining budget can afford.
  2. Applies a pacing factor: when remaining budget per remaining round
     is low, it penalises expensive bids.
"""

import numpy as np


class BudgetedUCB1:
    """
    Parameters
    ----------
    bids : array-like
        Discrete bid grid.
    budget : float
        Total initial budget.
    T : int
        Total number of rounds.
    alpha : float
        Pacing aggressiveness (0 = no pacing, 1 = hard pacing).
    """

    def __init__(self, bids, budget, T, alpha=1.0):
        self.bids   = np.array(bids)
        self.budget = budget
        self.T      = T
        self.alpha  = alpha
        K = len(bids)
        self.K      = K
        self._valid = self.bids > 1e-9   # bid=0 can never win
        self.counts = np.zeros(K)
        self.means  = np.zeros(K)
        self.t      = 0

    def _ucb_index(self, remaining_budget, remaining_rounds):
        rho = remaining_budget / max(remaining_rounds, 1)  # avg budget per round

        with np.errstate(divide='ignore', invalid='ignore'):
            exploration = np.where(
                self.counts > 0,
                np.sqrt(2 * np.log(max(self.t, 1)) / self.counts),
                np.inf
            )

        ucb = self.means + exploration

        # Pacing penalty: penalise bids that exceed the remaining pace
        pacing_penalty = self.alpha * np.maximum(self.bids - rho, 0)
        ucb -= pacing_penalty

        return ucb

    def select_bid(self, remaining_budget, remaining_rounds=None, **kwargs):
        self.t += 1
        remaining_rounds = remaining_rounds or max(self.T - self.t + 1, 1)

        # Mask bids that exceed budget or are zero
        feasible = (self.bids <= remaining_budget + 1e-9) & self._valid
        if not feasible.any():
            return 0.0

        ucb = self._ucb_index(remaining_budget, remaining_rounds)
        ucb[~feasible] = -np.inf

        # Break ties among unpulled arms
        inf_arms = np.where(np.isposinf(ucb))[0]
        if len(inf_arms) > 0:
            return self.bids[np.random.choice(inf_arms)]
        return self.bids[np.argmax(ucb)]

    def update(self, bid, reward, **kwargs):
        k = np.argmin(np.abs(self.bids - bid))
        self.counts[k] += 1
        self.means[k]  += (reward - self.means[k]) / self.counts[k]
