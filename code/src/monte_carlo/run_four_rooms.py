# Greg Attra
# 02/25/2022

'''
Exectuable for running four rooms MC control
'''

from cProfile import label
import matplotlib.pyplot as plt
import numpy as np

from lib.monte_carlo.env import FourRoomsEnv
from lib.monte_carlo.algorithms import on_policy_mc_control_epsilon_soft

SHORTEST_PATH = 20
N_EPISODES = int(10**4)
GOAL_RWD = 1.0
N_TRIALS = 2
GAMMA = .99

def main():
    env = FourRoomsEnv()
    epsilons = [.0, .01, 0.1]
    colors = [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1)
    ]
    for eps, clr in zip(epsilons, colors):
        trial_returns = []
        for t in range(N_TRIALS):
            returns = on_policy_mc_control_epsilon_soft(
                env,
                N_EPISODES,
                GAMMA,
                eps)
            trial_returns.append(returns)

        avg_ret = np.average(trial_returns, axis=0)
        plt.plot(avg_ret, color=clr, label=f'Epsilon: {str(eps)}')
        
        stde_avg_ret = 1.96 * (np.std(avg_ret) / np.sqrt(N_TRIALS))
        y_neg = avg_ret - stde_avg_ret
        y_pos = avg_ret + stde_avg_ret
        plt.fill_between(
            range(N_EPISODES),
            y_neg,
            y_pos,
            alpha=0.2,
            color=clr)

    opt_rwd = (GAMMA**SHORTEST_PATH) * GOAL_RWD
    plt.plot(
        [opt_rwd] * N_EPISODES,
        color='k',
        linestyle='solid',
        label='Upper Bound')

    plt.legend(loc='upper left')
    plt.show()

if __name__ == '__main__':
    main()
