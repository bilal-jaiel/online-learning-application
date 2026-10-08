"""
Non-Stationary Multi-Campaign Environment
------------------------------------------
The competing bid distributions change across predefined time intervals.
Within each interval the environment is stationary (piecewise-stationary).
"""

import numpy as np
from environments.multi_campaign_env import MultiCampaignEnv


class NonStationaryMultiCampaignEnv:
    """
    Parameters
    ----------
    values : array-like of shape (N,)
    budget : float
    T : int
    phases : list of dict
        Each dict has:
          'length'      : int, number of rounds in this phase
          'dists'       : list of str per campaign
          'dist_params' : list of tuple per campaign
    conflict_edges : list of tuple (i, j)
    seed : int, optional
    """

    def __init__(self, values, budget, T, phases, conflict_edges=None, seed=None):
        self.values = np.array(values, dtype=float)
        self.N = len(self.values)
        self.budget = float(budget)
        self.T = T
        self.phases = phases
        self.conflict_edges = conflict_edges or []
        self.seed = seed

        # Validate phases cover T rounds
        total = sum(p['length'] for p in phases)
        assert total == T, f"Phases sum to {total} but T={T}"

        # Build per-phase environments (reuse MultiCampaignEnv logic)
        self._phase_envs = []
        for p in phases:
            env = MultiCampaignEnv(
                values=values,
                budget=budget,  # budget managed externally
                T=p['length'],
                dists=p['dists'],
                dist_params=p['dist_params'],
                conflict_edges=conflict_edges,
                seed=seed,
            )
            self._phase_envs.append(env)

        # Build a flat list mapping round -> phase index
        self._round_to_phase = []
        for idx, p in enumerate(phases):
            self._round_to_phase.extend([idx] * p['length'])

        self.reset()

    def reset(self):
        self.remaining_budget = self.budget
        self.t = 0
        self._current_phase_idx = 0
        self._phase_t = 0  # round index within current phase
        return self

    def _sample_competing_bids(self):
        phase = self.phases[self._current_phase_idx]
        rng = np.random.default_rng(self.seed + self.t if self.seed else None)
        m = np.zeros(self.N)
        for i in range(self.N):
            dist = phase['dists'][i]
            params = phase['dist_params'][i]
            if dist == 'beta':
                m[i] = rng.beta(*params)
            elif dist == 'uniform':
                m[i] = rng.uniform(*params)
        return m

    def is_feasible(self, selected_campaigns):
        return self._phase_envs[0].is_feasible(selected_campaigns)

    def step(self, bids_vector):
        assert self.t < self.T, "Episode done. Call reset()."
        bids_vector = np.array(bids_vector, dtype=float)

        # Advance phase if needed
        phase_idx = self._round_to_phase[self.t]
        if phase_idx != self._current_phase_idx:
            self._current_phase_idx = phase_idx
            self._phase_t = 0

        active = np.where(bids_vector > 0)[0].tolist()
        assert self.is_feasible(active), "Conflict constraint violated!"

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
        self._phase_t += 1

        return {
            'won': won,
            'rewards': rewards,
            'costs': costs,
            'total_reward': total_reward,
            'total_cost': total_cost,
            'remaining_budget': self.remaining_budget,
            'competing_bids': m,
            'phase': self._current_phase_idx,
        }

    @property
    def is_done(self):
        return self.t >= self.T or self.remaining_budget < 1e-9

    def current_phase(self):
        return self._round_to_phase[min(self.t, self.T - 1)]
