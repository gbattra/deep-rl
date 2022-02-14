# Greg Attra
# 01/28/2022

import numpy as np


class Domain:
    """
    Representation of the distribution of rewards over actions
    """

    def __init__(self, mean: float, std: float, n_actions: int) -> None:
        self.mean = mean
        self.std = std
        self.n_actions = n_actions
        self.q_star = self._sample(mean, std, n_actions)

    def _sample(self, mean: float, std: float, n_actions: int) -> np.array:
        """
        Sample from a normal distribution
        """
        return np.random.normal(mean, std, n_actions)

    def reward(self, action: int) -> float:
        """
        Sample the reward from a normal dist using q* as mean
        """
        mean = self.q_star[action]
        reward = self._sample(mean, self.std, 1)[0]
        return reward
