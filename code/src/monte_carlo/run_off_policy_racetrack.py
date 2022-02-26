# Greg Attra
# 02/24/2022

'''
Executable for running MC eval on racetrack problem
'''

from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np

from lib.monte_carlo.algorithms import off_policy_mc_control_epsilon_soft
from lib.monte_carlo.racetracks import Racetrack, track0, track1

N_EPISODES = int(10**4)
STEP_RWD = -1.0
N_TRIALS = 10
GAMMA = .99
EPS = 0.1

def plot_returns(
        idx: int,
        trial_returns: np.ndarray,
        color: Tuple[float, float, float]) -> None:
    avg_ret = np.average(trial_returns, axis=0)
    plt.plot(avg_ret, color=color, label=f'Track: {str(idx)}')
    
    stde_avg_ret = 1.96 * (np.std(avg_ret) / np.sqrt(N_TRIALS))
    y_neg = avg_ret - stde_avg_ret
    y_pos = avg_ret + stde_avg_ret
    plt.fill_between(
        range(N_EPISODES),
        y_neg,
        y_pos,
        alpha=0.2,
        color=color)

def main():
    tracks = [track0(), track1()]
    b_colors = [
        (1, 0, 0),
        (0, 1, 0)
    ]
    t_colors = [
        (0.5, 0, 0),
        (0, 0.5, 0)
    ]
    styles = [
        'dashed',
        'dotted'
    ]
    for i, track in enumerate(tracks):
        print(f'------------------------------')
        print(f'Track: {i}')
        print(f'------------------------------')

        b_trial_returns = np.zeros((N_TRIALS, N_EPISODES))
        t_trial_returns = np.zeros((N_TRIALS, N_EPISODES))
        env = Racetrack(track)
        for t in range(N_TRIALS): 
            print(f'Trial: {t + 1} / {N_TRIALS}')
            b_returns, t_returns = off_policy_mc_control_epsilon_soft(
                env,
                N_EPISODES,
                GAMMA,
                EPS)
            b_trial_returns[t, :] = b_returns
            t_trial_returns[t, :] = t_returns

        plot_returns(i, b_trial_returns, b_colors[i])
        plot_returns(i, t_trial_returns, t_colors[i])

    for t, track in enumerate(tracks):
        shortest_path = np.linalg.norm(track)
        opt_rwd = (GAMMA**shortest_path) * (STEP_RWD * shortest_path)
        plt.plot(
            [opt_rwd] * N_EPISODES,
            color='k',
            linestyle=styles[t],
            label=f'Upper Bound - Track {str(t)}')

    plt.legend(loc='upper left')
    plt.show()

if __name__ == '__main__':
    main()
