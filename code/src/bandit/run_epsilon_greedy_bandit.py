#!/usr/bin/env python

# Greg Attra
# 01/28/2022

import argparse
import matplotlib.pyplot as plt

from lib.bandit.analytics import EpsilonGreedyBanditAnalytics
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
                        nargs='+', type=float, default=[.1])
    args = parser.parse_args()

    analytics = []
    fig, ax = plt.subplots(2)
    for eps in args.eps:
        analytics = EpsilonGreedyBanditAnalytics(eps, args.p, args.s, args.k)
        testsuite = TestSuite(
            args.p,
            args.s,
            EpsilonGreedyBanditFactory(args.k, eps),
            DomainFactory(args.mu, args.std, args.k),
            analytics)
        testsuite.run()
        analytics.splot(ax)
    ax[0].legend()

    plt.show()


if __name__ == '__main__':
    main()
