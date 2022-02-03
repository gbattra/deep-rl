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
    parser.add_argument('-a', help='The learning rate for the incremental update',
                        type=float, default=.1)
    args = parser.parse_args()

    analytics = []
    fig, ax = plt.subplots(2)
    eps = [.0, .0, .1, .1]
    q_inits = [.0, 5., .0, 5.]
    styles = [
        ((1., .5, .5), 'dashed'),
        ((1., .0, .0), 'solid'),
        ((.5, .5, 1.), 'dashed'),
        ((.0, .0, 1.), 'solid')
    ]
    for q_init, eps, style in zip(q_inits, eps, styles):
        np.random.seed(0)
        analytics = OptimisticGreedyBanditAnalytics(q_init, eps, args.p, args.s, args.k)
        testsuite = TestSuite(
            args.p,
            args.s,
            OptimisticGreedyBanditFactory(q_init, eps, args.a, args.k),
            DomainFactory(args.mu, args.std, args.k),
            analytics)
        testsuite.run()

        color, linestyle = style
        analytics.splot(ax, color, linestyle)
    
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

    np.random.seed(0)
    c = 2.
    analytics = UcbBanditAnalytics(c, args.p, args.s, args.k)
    testsuite = TestSuite(
        args.p,
        args.s,
        UcbBanditFactory(c, args.k),
        DomainFactory(args.mu, args.std, args.k),
        analytics)
    testsuite.run()
    analytics.splot(ax, (.2, .7, .4), 'solid')

    ax[0].legend()
    ax[1].legend()

    plt.show()


if __name__ == '__main__':
    main()
