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
from lib.function_approximation.features import xy_features, all_features
from lib.function_approximation.four_rooms import FourRooms, doors, four_rooms_arena, goal
from lib.function_approximation.policy import q_function

ALPHA: float = 0.01
GAMMA: float = 0.9
EPSILON: float = 0.1
N_EPISODES: int = 100
N_TRIALS: int = 10


def main():
    arena = four_rooms_arena()
    print(arena)    
    env = FourRooms(arena)

    feature_sets = [
        (0, 35, lambda s: all_features(s, arena, doors(), goal()), (1., .0, .0)),
        (0.1, 35, lambda s: all_features(s, arena, doors(), goal()), (.0, .0, 1.))
    ]
    for alpha, n_feats, X, color in feature_sets:
        Q = q_function
        total_results = np.zeros((N_TRIALS, N_EPISODES))
        for t in trange(N_TRIALS, desc='Trial', leave=False):
            results = semigrad_onestep_sarsa(
                env,
                Q,
                X,
                n_feats,
                alpha,
                EPSILON,
                GAMMA,
                N_EPISODES)
            total_results[t, :] = results['episode_lengths']
        
        avg_ret = np.average(total_results, axis=0)
        plt.plot(avg_ret, color=color, label=f'{n_feats} Features | Alpha: {alpha} {"(inference only)" if alpha == 0 else ""}')
        
        stde_avg_ret = 1.96 * (np.std(total_results) / np.sqrt(N_EPISODES))
        y_neg = avg_ret - stde_avg_ret
        y_pos = avg_ret + stde_avg_ret
        plt.fill_between(
            range(N_EPISODES),
            y_neg,
            y_pos,
            alpha=0.2,
            color=color)
    
    plt.xlabel('Episode')
    plt.ylabel('Avg Episode Length')
    plt.title('Naiive Features Approximation')
    plt.legend()
    plt.show()

    

if __name__ == '__main__':
    main()
