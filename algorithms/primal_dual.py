"""
Primal-Dual Pacing Algorithm: Multi-Campaign
----------------------------------------------
Uses a Lagrange multiplier λ to pace budget consumption.

Score per arm: ucb_reward[i,k] - λ * estimated_cost[i,k]
λ update (projected gradient ascent on dual):
    λ ← max(0, λ - η * (ρ - cost_t))
where ρ = B/T is the target spend per round.

When we overspend (cost_t > ρ): λ increases → expensive bids penalised.
When we underspend (cost_t < ρ): λ decreases → algorithm bids more aggressively.
"""

import numpy as np
from utils.oracle import oracle_select


class PrimalDualPacing:
    """
    Parameters
    ----------
    N, bids, conflicts, budget, T : see CombinatorialUCB.
    eta : float
        Dual step size. Default: 1/sqrt(T).
    lambda_init : float
        Initial Lagrange multiplier.
    """

    def __init__(self, N, bids, conflicts, budget, T,
                 eta=None, lambda_init=0.0):
        self.N         = N
        self.bids      = np.array(bids)
        self.K         = len(bids)
        self.conflicts = conflicts
        self.budget    = budget
        self.T         = T
        self.rho       = budget / T            # target spend per round
        self.eta       = eta or 1.0 / np.sqrt(T)
        self.lambda_   = lambda_init

        self.counts      = np.zeros((N, self.K))
        self.mean_reward = np.zeros((N, self.K))
        self.mean_cost   = np.zeros((N, self.K))
        self.t           = 0

    def _compute_ucb(self):
        log_t = np.log(max(self.t, 1))
        with np.errstate(divide='ignore', invalid='ignore'):
            bonus = np.where(
                self.counts > 0,
                np.sqrt(2 * log_t / self.counts),
                1.0   # optimistic initialisation
            )
        ucb_reward = self.mean_reward + bonus

        # Estimated cost: mean cost or bid value if never pulled
        est_cost = self.mean_cost.copy()
        for k, b in enumerate(self.bids):
            est_cost[:, k] = np.where(
                self.counts[:, k] > 0,
                est_cost[:, k],
                b
            )
        return ucb_reward, est_cost

    def select_bids(self, remaining_budget, **kwargs):
        self.t += 1
        ucb_reward, est_cost = self._compute_ucb()

        # Lagrangian score: reward - λ * cost
        scores = ucb_reward - self.lambda_ * est_cost

        bids_vec, _ = oracle_select(scores, est_cost,
                                    remaining_budget, self.conflicts, self.bids)
        return bids_vec

    def update(self, bids_vector, rewards, costs_obs, won, **kwargs):
        total_cost = costs_obs.sum() if hasattr(costs_obs, '__len__') else costs_obs

        # Dual update: projected gradient ascent
        self.lambda_ = max(0.0, self.lambda_ - self.eta * (self.rho - total_cost))

        for i in range(self.N):
            if bids_vector[i] > 0:
                k = np.argmin(np.abs(self.bids - bids_vector[i]))
                self.counts[i, k] += 1
                n = self.counts[i, k]
                self.mean_reward[i, k] += (rewards[i] - self.mean_reward[i, k]) / n
                self.mean_cost[i, k]   += (costs_obs[i] - self.mean_cost[i, k]) / n

    @property
    def current_lambda(self):
        return self.lambda_
