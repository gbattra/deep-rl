# Greg Attra
# 01/28/2022

from lib.bandit.factory import Factory
from lib.bandit.analytics import BanditAnalytics
from lib.bandit.domain import Domain
from lib.bandit.bandit import Bandit


class TestSuite:
    def __init__(
            self,
            n_problems: int,
            n_steps: int,
            bandit_factory: Factory[Bandit],
            domain_factory: Factory[Domain],
            analytics: BanditAnalytics) -> None:
        self.n_problems = n_problems
        self.n_steps = n_steps
        self.bandit_factory = bandit_factory
        self.domain_factory = domain_factory
        self.analytics = analytics

    def run(self) -> None:
        for p in range(self.n_problems):
            domain = self.domain_factory.instance()
            self.analytics.update_qstar(p, domain.q_star)

            bandit = self.bandit_factory.instance()
            for s in range(self.n_steps):
                a = bandit.act()
                r = domain.reward(a)
                bandit.update(a, r)
                self.analytics.update_ar_hist(p, s, a, r)
