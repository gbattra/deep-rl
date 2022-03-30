# Greg Attra
# 03/26/22

'''
Executable for running linear function approximation on the four rooms problem
'''

import math
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

from lib.function_approximation.aggregators import segment
from lib.function_approximation.algorithms import semigrad_onestep_sarsa
from lib.function_approximation.features import get_feature_extractor, one_hot_encode
from lib.function_approximation.four_rooms import FourRooms, four_rooms_arena
from lib.function_approximation.policy import q_function

ALPHA: float = 0.01
GAMMA: float = 0.99
EPSILON: float = 0.1
N_EPISODES: int = 100
N_TRIALS: int = 10


def main():
    arena = four_rooms_arena()
    env = FourRooms(arena)

    seg_sizes = [1, 2, 3]
    colors = [
        (1., .0, .0),
        (.0, 1., .0),
        (.0, .0, 1.)
    ]

    print('Running segmentation experiments...')
    for seg_size, color in zip(seg_sizes, colors):
        print(f'Segment Size: {seg_size}')
        n_feats = int(math.ceil(arena.shape[0] / seg_size) * arena.shape[1])
        segmentor = lambda s: segment(s, arena.shape[1], seg_size)
        X = get_feature_extractor(
            n_feats,
            one_hot_encode,
            segmentor)
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
        
        avg_ret = np.average(total_results, axis=0)
        plt.plot(avg_ret, color=color, label=f'Segment Size: {seg_size}')
        
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
    plt.title('State Aggregation')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
