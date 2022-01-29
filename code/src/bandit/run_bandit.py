#!/usr/bin/env python

# Greg Attra
# 01/28/2022

import argparse
from lib.bandit.factory import BanditFactory, DomainFactory
from lib.bandit.analytics import BanditAnalytics
from lib.bandit.testsuite import TestSuite


def main():
    parser = argparse.ArgumentParser('Run a bandit testsuite')
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
    args = parser.parse_args()

    analytics = BanditAnalytics(args.p, args.s, args.k)
    testsuite = TestSuite(
        args.p,
        args.s,
        BanditFactory(args.k),
        DomainFactory(args.mu, args.std, args.k),
        analytics)
    testsuite.run()
    analytics.plot()


if __name__ == '__main__':
    main()
