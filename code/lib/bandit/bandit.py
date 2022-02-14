# Greg Attra
# 01/28/2022

import numpy as np


class Bandit:
    """
    Representation of a k-armed bandit
    """

    def __init__(self, k: int) -> None:
        self.k = k
        self.q = np.zeros(k)
        self.n = np.zeros(k)
        self.r_sum = 0

    def act(self) -> int:
        """
        Sample an action randomly
        """
        return np.random.randint(0, self.k)

    def update(self, a: int, r: float) -> None:
        """
        Update q given the action chosen and the reward received
        """
        self.r_sum += r
        self.n[a] += 1
        self.q[a] = self.q[a] + (1.0 / float(self.n[a]) * (r - self.q[a]))


class EpsilonGreedyBandit(Bandit):
    """
    Bandit implementing epsilon greedy action selection algo
    """
    def __init__(self, k: int, eps: float) -> None:
        super().__init__(k)
        self.eps = eps

    def act(self) -> int:
        """
        Make action selection using epsilon greedy algorithm
        """
        if np.random.random() <= self.eps:
            # make random choice
            return super().act()
        return np.argmax(self.q)


class OptimisticGreedyBandit(EpsilonGreedyBandit):
    """
    Epsilon greedy bandit with optimistic initial q values
    """
    def __init__(
            self,
            k: int,
            q_init: float,
            eps: float,
            alpha: float) -> None:
        super().__init__(k, eps)
        self.q = np.ones(k) * q_init
        self.alpha = alpha

    def update(self, a: int, r: float) -> None:
        """
        Update q given the action chosen and the reward received
        """
        self.r_sum += r
        self.n[a] += 1
        self.q[a] = self.q[a] + self.alpha * (r - self.q[a])

    
class UcbBandit(Bandit):
    def __init__(self, c: float, k: int) -> None:
        super().__init__(k)
        self.c = c

    def act(self) -> int:
        """
        Make action decision using UCB algorithm
        """
        if 0 in self.n:
            a = np.where(self.n == 0)[0][0]
            return a

        t = np.sum(self.n)
        a = np.argmax(self.q + (self.c * np.sqrt(np.log(t)/self.n)))
        return a
