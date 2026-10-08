"""
Change-Detection Combinatorial UCB
------------------------------------
Monitors per-arm reward sequences with a CUSUM-like test.
When a change is detected for any arm, all statistics are reset.

Detection: for each arm (i,k) with enough samples,
  g_t = max(0, g_{t-1} + (x_t - mu_hat - epsilon))
  if g_t > h: change detected → reset all.

Parameters
  epsilon : minimum shift magnitude to detect
  h       : detection threshold
"""

import numpy as np
from utils.oracle import oracle_select


class ChangeDetectionUCB:
    """
    Parameters
    ----------
    N, bids, conflicts, budget, T : standard params.
    epsilon : float   CUSUM sensitivity (minimum mean shift to detect).
    h : float         CUSUM threshold.
    min_samples : int Minimum pulls before testing.
    """

    def __init__(self, N, bids, conflicts, budget, T,
                 epsilon=0.05, h=0.5, min_samples=30):
        self.N         = N
        self.bids      = np.array(bids)
        self.K         = len(bids)
        self.conflicts = conflicts
        self.budget    = budget
        self.T         = T
        self.epsilon   = epsilon
        self.h         = h
        self.min_samples = min_samples
        self.t         = 0
        self._reset_stats()

    def _reset_stats(self):
        self.counts      = np.zeros((self.N, self.K))
        self.mean_reward = np.zeros((self.N, self.K))
        self.mean_cost   = np.zeros((self.N, self.K))
        # CUSUM statistic per arm
        self.cusum_pos   = np.zeros((self.N, self.K))  # detect upward shift
        self.cusum_neg   = np.zeros((self.N, self.K))  # detect downward shift
        self.n_resets    = getattr(self, 'n_resets', 0) + 1

    def _compute_ucb(self):
        log_t = np.log(max(self.t, 1))
        with np.errstate(divide='ignore', invalid='ignore'):
            bonus = np.where(
                self.counts > 0,
                np.sqrt(2 * log_t / self.counts),
                np.inf
            )
        ucb_reward = self.mean_reward + bonus
        est_cost   = self.mean_cost.copy()
        for k, b in enumerate(self.bids):
            est_cost[:, k] = np.where(
                self.counts[:, k] > 0, est_cost[:, k], b)
        return ucb_reward, est_cost

    def select_bids(self, remaining_budget, **kwargs):
        self.t += 1
        ucb_reward, est_cost = self._compute_ucb()
        bids_vec, _ = oracle_select(ucb_reward, est_cost,
                                    remaining_budget, self.conflicts, self.bids)
        return bids_vec

    def update(self, bids_vector, rewards, costs_obs, won, **kwargs):
        change_detected = False

        for i in range(self.N):
            if bids_vector[i] > 0:
                k = np.argmin(np.abs(self.bids - bids_vector[i]))
                self.counts[i, k] += 1
                n = self.counts[i, k]
                old_mean = self.mean_reward[i, k]
                self.mean_reward[i, k] += (rewards[i] - old_mean) / n
                self.mean_cost[i, k]   += (costs_obs[i] - self.mean_cost[i, k]) / n

                # CUSUM update
                if n >= self.min_samples:
                    deviation = rewards[i] - old_mean
                    self.cusum_pos[i, k] = max(
                        0, self.cusum_pos[i, k] + deviation - self.epsilon)
                    self.cusum_neg[i, k] = max(
                        0, self.cusum_neg[i, k] - deviation - self.epsilon)

                    if (self.cusum_pos[i, k] > self.h or
                            self.cusum_neg[i, k] > self.h):
                        change_detected = True

        if change_detected:
            self._reset_stats()
