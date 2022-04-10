# Greg Attra
# 04.09.22

'''
Executable for running REINFORCE algorithm
'''

import math
import matplotlib.pyplot as plt
import numpy as np

from lib.function_approximation.features import get_feature_extractor, one_hot_encode
from lib.policy_gradient.algorithms import reinforce
from lib.domains.four_rooms import four_rooms_arena, FourRooms
from lib.policy_gradient.policy import softmax_policy


GAMMA: float = 0.99
ALPHA: float = 0.1

N_EPISODES: int = 1000
N_STEPS: int = 10000

def main():
    arena = four_rooms_arena()
    env = FourRooms(arena, 0.2, 450)

    X = lambda s: one_hot_encode(s, arena.size)
    results = reinforce(
        env,
        softmax_policy,
        X,
        GAMMA,
        ALPHA,
        arena.size,
        N_EPISODES,
        N_STEPS
    )

    plt.plot(results['durations'])
    plt.show()


if __name__ == '__main__':
    main()
