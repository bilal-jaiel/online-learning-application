"""
UCB1: Single Campaign, No Budget Constraint
----------------------------------------------
Each discrete bid is an arm.
UCB1 index: empirical_mean + sqrt(2 * log(t) / n_pulls)
"""

import numpy as np


class UCB1:
    """
    Parameters
    ----------
    bids : array-like
        Discrete bid grid, e.g. np.linspace(0, 1, 11).
    """

    def __init__(self, bids):
        self.bids = np.array(bids)
        # Exclude bid=0: it can never win a first-price auction
        self._valid = self.bids > 1e-9
        K = len(bids)
        self.K = K
        self.counts = np.zeros(K)        # n_k : number of pulls of arm k
        self.means  = np.zeros(K)        # empirical mean reward of arm k
        self.t      = 0                  # global round counter

    def _ucb_index(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            exploration = np.where(
                self.counts > 0,
                np.sqrt(2 * np.log(max(self.t, 1)) / self.counts),
                np.inf  # unpulled arms -> always explore first
            )
        ucb = self.means + exploration
        ucb[~self._valid] = -np.inf      # never pick bid=0
        return ucb

    def select_bid(self, **kwargs):
        self.t += 1
        ucb = self._ucb_index()
        # Break ties randomly among unpulled arms
        inf_arms = np.where(np.isposinf(ucb))[0]
        if len(inf_arms) > 0:
            return self.bids[np.random.choice(inf_arms)]
        return self.bids[np.argmax(ucb)]

    def update(self, bid, reward, **kwargs):
        k = np.argmin(np.abs(self.bids - bid))
        self.counts[k] += 1
        self.means[k]  += (reward - self.means[k]) / self.counts[k]

    @property
    def ucb_values(self):
        return self._ucb_index()
