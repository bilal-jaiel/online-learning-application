"""
Combinatorial UCB: Multi-Campaign, Budget + Conflict Constraints
------------------------------------------------------------------
Arms    : (campaign i, bid b_k) pairs
Superarm: feasible set of (i, b_k) pairs, one per campaign, no conflicts,
          total cost <= remaining budget.

UCB index per arm:
  ucb_reward[i,k] = mean_reward[i,k] + sqrt(2 log t / n[i,k])
  lcb_cost[i,k]   = max(0, mean_cost[i,k] - sqrt(2 log t / n[i,k]))

Oracle: picks superarm maximising sum of ucb_reward scores.
"""

import numpy as np
from utils.oracle import oracle_select


class CombinatorialUCB:
    """
    Parameters
    ----------
    N : int          Number of campaigns.
    bids : array     Discrete bid grid of shape (K,).
    conflicts : dict {i: set of j} conflict adjacency.
    budget : float   Total budget.
    T : int          Total rounds.
    """

    def __init__(self, N, bids, conflicts, budget, T):
        self.N       = N
        self.bids    = np.array(bids)
        self.K       = len(bids)
        self.conflicts = conflicts
        self.budget  = budget
        self.T       = T

        self.counts      = np.zeros((N, self.K))
        self.mean_reward = np.zeros((N, self.K))
        self.mean_cost   = np.zeros((N, self.K))
        self.t           = 0

    def _compute_ucb_scores(self):
        log_t = np.log(max(self.t, 1))
        with np.errstate(divide='ignore', invalid='ignore'):
            bonus = np.where(
                self.counts > 0,
                np.sqrt(2 * log_t / self.counts),
                np.inf
            )
        ucb_reward = self.mean_reward + bonus

        # LCB cost: lower bound to encourage cost-efficiency
        lcb_cost = np.maximum(0, self.mean_cost - bonus)
        # Fallback: if never pulled, cost = bid value
        for k, b in enumerate(self.bids):
            lcb_cost[:, k] = np.where(
                self.counts[:, k] > 0,
                lcb_cost[:, k],
                b  # assume win = pay bid
            )
        return ucb_reward, lcb_cost

    def select_bids(self, remaining_budget, **kwargs):
        self.t += 1
        ucb_reward, lcb_cost = self._compute_ucb_scores()
        bids_vec, _ = oracle_select(ucb_reward, lcb_cost,
                                    remaining_budget, self.conflicts, self.bids)
        return bids_vec

    def update(self, bids_vector, rewards, costs_obs, won, competing_bids=None, **kwargs):
        for i in range(self.N):
            if bids_vector[i] > 0:
                k = np.argmin(np.abs(self.bids - bids_vector[i]))
                self.counts[i, k] += 1
                n = self.counts[i, k]
                self.mean_reward[i, k] += (rewards[i] - self.mean_reward[i, k]) / n
                self.mean_cost[i, k]   += (costs_obs[i] - self.mean_cost[i, k]) / n
