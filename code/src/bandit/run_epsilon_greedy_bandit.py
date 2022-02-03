#!/usr/bin/env python

# Greg Attra
# 01/28/2022

import argparse
import matplotlib.pyplot as plt
import numpy as np

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
    args = parser.parse_args()

    analytics = []
    fig, ax = plt.subplots(2)
    colors = [
        (.8, .4, .4),
        (.4, .8, .4),
        (.4, .4, .8)
    ]
    eps = [0.0, 0.01, 0.1]
    for color, eps in zip(colors, eps):
        np.random.seed(0)
        analytics = EpsilonGreedyBanditAnalytics(eps, args.p, args.s, args.k)
        testsuite = TestSuite(
            args.p,
            args.s,
            EpsilonGreedyBanditFactory(args.k, eps),
            DomainFactory(args.mu, args.std, args.k),
            analytics)
        testsuite.run()
        analytics.splot(ax, color, 'solid')
    
    opt_rwds = np.amax(analytics.q_stars, axis=1)
    avg_opt_rwd = np.sum(opt_rwds) / float(analytics.n_problems)
    ax[0].hlines(
        avg_opt_rwd,
        xmin=0, xmax=analytics.n_steps,
        color='k',
        linestyle='solid',
        label='Upper Bound')

    stde_avg_rwd = 1.96 * (np.std(opt_rwds) / np.sqrt(analytics.n_problems))
    y_neg = avg_opt_rwd - stde_avg_rwd
    y_pos = avg_opt_rwd + stde_avg_rwd
    ax[0].fill_between(
        range(analytics.n_steps),
        y_neg,
        y_pos,
        alpha=0.2,
        color=(0.25, 0.25, 0.25))

    ax[0].legend()
    ax[1].legend()

    plt.show()


if __name__ == '__main__':
    main()
