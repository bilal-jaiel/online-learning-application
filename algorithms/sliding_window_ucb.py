"""
Sliding-Window Combinatorial UCB
---------------------------------
Variant of Combinatorial-UCB for non-stationary environments.
Only the W most recent observations per arm are used to estimate means.
This allows the algorithm to "forget" old distributions and adapt.
"""

import numpy as np
from collections import deque
from utils.oracle import oracle_select


class SlidingWindowUCB:
    """
    Parameters
    ----------
    N, bids, conflicts, budget, T : standard params.
    W : int
        Window size (number of recent rounds to keep per arm).
    """

    def __init__(self, N, bids, conflicts, budget, T, W=300):
        self.N         = N
        self.bids      = np.array(bids)
        self.K         = len(bids)
        self.conflicts = conflicts
        self.budget    = budget
        self.T         = T
        self.W         = W
        self.t         = 0

        # Sliding window buffers: reward_history[i][k] = deque of recent rewards
        self.reward_history = [[deque(maxlen=W) for _ in range(self.K)]
                               for _ in range(N)]
        self.cost_history   = [[deque(maxlen=W) for _ in range(self.K)]
                               for _ in range(N)]

    def _window_stats(self):
        """Compute window mean and count for each (i, k)."""
        mean_r = np.zeros((self.N, self.K))
        mean_c = np.zeros((self.N, self.K))
        counts = np.zeros((self.N, self.K))
        for i in range(self.N):
            for k in range(self.K):
                rh = self.reward_history[i][k]
                ch = self.cost_history[i][k]
                if rh:
                    mean_r[i, k] = np.mean(rh)
                    mean_c[i, k] = np.mean(ch)
                    counts[i, k] = len(rh)
                else:
                    mean_c[i, k] = self.bids[k]  # prior: pay bid if win
        return mean_r, mean_c, counts

    def _compute_ucb(self):
        mean_r, mean_c, counts = self._window_stats()
        n_w = np.minimum(counts, self.W)
        with np.errstate(divide='ignore', invalid='ignore'):
            bonus = np.where(
                counts > 0,
                np.sqrt(2 * np.log(min(self.t, self.W)) / n_w),
                np.inf
            )
        ucb_reward = mean_r + bonus
        return ucb_reward, mean_c

    def select_bids(self, remaining_budget, **kwargs):
        self.t += 1
        ucb_reward, est_cost = self._compute_ucb()
        bids_vec, _ = oracle_select(ucb_reward, est_cost,
                                    remaining_budget, self.conflicts, self.bids)
        return bids_vec

    def update(self, bids_vector, rewards, costs_obs, won, **kwargs):
        for i in range(self.N):
            if bids_vector[i] > 0:
                k = np.argmin(np.abs(self.bids - bids_vector[i]))
                self.reward_history[i][k].append(rewards[i])
                self.cost_history[i][k].append(costs_obs[i])
