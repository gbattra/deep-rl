#!/usr/bin/env python

# Greg Attra
# 01/28/2022

import argparse
import matplotlib.pyplot as plt
import numpy as np
from lib.bandit.factory import OptimisticGreedyBanditFactory
from lib.bandit.analytics import OptimisticGreedyBanditAnalytics
from lib.bandit.analytics import UcbBanditAnalytics
from lib.bandit.factory import UcbBanditFactory
from lib.bandit.factory import EpsilonGreedyBanditFactory
from lib.bandit.factory import DomainFactory
from lib.bandit.testsuite import TestSuite


def main():
    parser = argparse.ArgumentParser('Run an epsilon greedy bandit testsuite')
    parser.add_argument('-p', help='The number of problems to run',
                        type=int, default=10)
    parser.add_argument('-s', help='The number of steps to run per problem',
                        type=int, default=10)
    parser.add_argument('-k', help='The number of arms on the bandit',
                        type=int, default=10)
    parser.add_argument('-mu', help='The mean of q*',
                        type=float, default=.0)
    parser.add_argument('-std', help='The std of q*',
                        type=float, default=1.)
    parser.add_argument('-eps', help='Epsilon values for each bandit',
                        nargs='+', type=float, default=[.0, .0, .1, .1])
    parser.add_argument('-c', help='The c param for the UCB action selectin algo',
                        type=float, default=2.)
    parser.add_argument('-q', help='The initial q values for optimistic greedy',
                        nargs='+', type=float, default=[.0, 5., .0, 5.])
    args = parser.parse_args()

    analytics = []
    fig, ax = plt.subplots(2)
    for q_init, eps in zip(args.q, args.eps):
        np.random.seed(0)
        analytics = OptimisticGreedyBanditAnalytics(q_init, eps, args.p, args.s, args.k)
        testsuite = TestSuite(
            args.p,
            args.s,
            OptimisticGreedyBanditFactory(q_init, eps, args.k),
            DomainFactory(args.mu, args.std, args.k),
            analytics)
        testsuite.run()
        analytics.splot(ax)

    np.random.seed(0)
    analytics = UcbBanditAnalytics(args.c, args.p, args.s, args.k)
    testsuite = TestSuite(
        args.p,
        args.s,
        UcbBanditFactory(args.c, args.k),
        DomainFactory(args.mu, args.std, args.k),
        analytics)
    testsuite.run()
    analytics.splot(ax)

    plt.show()


if __name__ == '__main__':
    main()
