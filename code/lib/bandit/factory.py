# Greg Attra
# 01/28/2022

from tkinter import E
from typing import Generic, TypeVar
from lib.bandit.bandit import OptimisticGreedyBandit
from lib.bandit.bandit import UcbBandit
from lib.bandit.bandit import Bandit, EpsilonGreedyBandit

from lib.bandit.domain import Domain


T = TypeVar('T')

class Factory(Generic[T]):
    def instance(self) -> T:
        pass


class DomainFactory(Factory[Domain]):
    def __init__(self, mean: float, std: float, n_actions: int) -> None:
        self.mean = mean
        self.std = std
        self.n_actions = n_actions

    def instance(self) -> Domain:
        return Domain(self.mean, self.std, self.n_actions)


class BanditFactory(Factory[Bandit]):
    def __init__(self, k: int) -> None:
        self.k = k

    def instance(self) -> Bandit:
        return Bandit(self.k)


class EpsilonGreedyBanditFactory(BanditFactory):
    def __init__(self, k: int, eps: float) -> None:
        super().__init__(k)
        self.eps = eps
    
    def instance(self) -> EpsilonGreedyBandit:
        return EpsilonGreedyBandit(self.k, self.eps)


class OptimisticGreedyBanditFactory(EpsilonGreedyBanditFactory):
    def __init__(self, q_init: float, eps: float, alpha: float, k: int) -> None:
        super().__init__(k, eps)
        self.q_init = q_init
        self.alpha = alpha

    def instance(self) -> OptimisticGreedyBandit:
        return OptimisticGreedyBandit(self.k, self.q_init, self.eps, self.alpha)


class UcbBanditFactory(BanditFactory):
    def __init__(self, c: float, k: int) -> None:
        super().__init__(k)
        self.c = c

    def instance(self) -> UcbBandit:
        return UcbBandit(self.c, self.k)
