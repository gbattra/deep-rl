# Greg Attra
# 03/26/22

'''
Executable for running linear function approximation on the four rooms problem
'''

import math
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

from lib.function_approximation.algorithms import semigrad_onestep_sarsa
from lib.function_approximation.features import xy_features, xy_poly_features, all_features
from lib.function_approximation.four_rooms import FourRooms, doors, four_rooms_arena, goal
from lib.function_approximation.policy import q_function

ALPHA: float = 0.001
GAMMA: float = 0.9
EPSILON: float = 0.1
N_EPISODES: int = 100
N_TRIALS: int = 3


def main():
    arena = four_rooms_arena()
    env = FourRooms(arena)

    feature_sets = [
        # (3, lambda s: xy_features(s, arena.shape[1])),
        # (6, lambda s: xy_walls_goal__features(s, arena.shape[1]))
        (13, lambda s: all_features(s, arena.shape[0], doors(), goal()))
    ]
    for n_feats, X in feature_sets:
        Q = q_function
        total_results = np.zeros((N_TRIALS, N_EPISODES))
        for t in trange(N_TRIALS, desc='Trial', leave=False):
            results = semigrad_onestep_sarsa(
                env,
                Q,
                X,
                n_feats,
                ALPHA,
                EPSILON,
                GAMMA,
                N_EPISODES)
            total_results[t, :] = results['episode_lengths']
        
        color = (1., .0, .0)
        avg_ret = np.average(total_results, axis=0)
        plt.plot(avg_ret, color=color, label=f'{n_feats} Features')
        
        stde_avg_ret = 1.96 * (np.std(total_results) / np.sqrt(N_EPISODES))
        y_neg = avg_ret - stde_avg_ret
        y_pos = avg_ret + stde_avg_ret
        plt.fill_between(
            range(N_EPISODES),
            y_neg,
            y_pos,
            alpha=0.2,
            color=color)
    
    plt.title('Naiive Features Approximation')
    plt.legend()
    plt.show()

    

if __name__ == '__main__':
    main()
