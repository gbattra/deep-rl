#!/usr/bin/env python

# Greg Attra
# 01/28/2022

import argparse
from lib.bandit.analytics import Analytics
from lib.bandit.bandit import make_bandit
from lib.bandit.domain import make_domain

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

    analytics = Analytics(args.p, args.s, args.k)
    testsuite = TestSuite(args.p, args.s, make_bandit, make_domain, analytics)
    testsuite.run(args.mu, args.std, args.k)
    analytics.plot()


if __name__ == '__main__':
    main()
