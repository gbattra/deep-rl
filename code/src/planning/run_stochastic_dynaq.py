# Greg Attra
# 03.12.2022

'''
Executable for running Dyna-Q implementation for stochastic environments
'''

import numpy as np
import matplotlib.pyplot as plt

from tqdm import trange
from lib.planning.dyna_q import dyna_q_plus
from lib.planning.mazes import Maze, shortcut_maze_end, shortcut_maze_start


TOGGLE_STEP: int = 1000
N_TIMESTEPS: int = 5000
N_TRIALS: int = 10
GAMMA: float = .95
ALPHA: float = .1
EPSILON: float = .1


def run_stochastic_dynaq_plus():
    print('Dyna-Q+')
    env = Maze(
        shortcut_maze_start(),
        shortcut_maze_end(),
        TOGGLE_STEP,
        .2)
    n_plan_steps = [10, 100, 250]
    k_vals = [0, 1e-3]
    styles = ['solid', 'dotted']
    colors = [
        (1., .0, .0),
        (.0, 1., .0),
        (.0, .0, 1.)
    ]
    for k, style in zip(k_vals, styles):
        config_results = []
        for i in range(len(n_plan_steps)):
            print(f'Trials for {n_plan_steps[i]} plan steps')
            trial_results = np.zeros((N_TRIALS, N_TIMESTEPS))
            for t in trange(N_TRIALS, desc='Trial', leave=False):
                results = dyna_q_plus(
                    env,
                    k,
                    ALPHA,
                    EPSILON,
                    GAMMA,
                    n_plan_steps[i],
                    N_TIMESTEPS)
                trial_results[t, :] = results['stats']
            config_results.append(trial_results)
    
        for n_plan, stats, color in zip(n_plan_steps, config_results, colors):
            avg_eps = np.average(stats, axis=0)
            plt.plot(
                avg_eps,
                label=f'{n_plan} plan steps | k = {k} {"(Dyna-Q)" if k == 0 else ""}',
                color=color,
                linestyle=style)

    plt.title('Stochastic Shortcut Maze | Dyna-Q+')
    plt.xlabel('Timesteps')
    plt.ylabel('Avg. Episodes')
    plt.legend()
    plt.show()


def main():
    run_stochastic_dynaq_plus()


if __name__ == '__main__':
    main()
