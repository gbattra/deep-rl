# Greg Attra
# 04.09.22

'''
Executable for running actor critic algorithm
'''

import matplotlib.pyplot as plt

from lib.domains.four_rooms import four_rooms_arena, FourRooms
from lib.function_approximation.features import one_hot_encode
from lib.policy_gradient.algorithms import actor_critic
from lib.policy_gradient.policy import softmax_policy


GAMMA: float = 0.99
ALPHA_P: float = 0.1
ALPHA_Q: float = 0.1
N_EPISODES: int = 1000


def main():
    arena = four_rooms_arena()
    env = FourRooms(arena, 0.2, 450)
    X = lambda s: one_hot_encode(s, arena.size)
    Q = lambda x, W: x.dot(W)
    results = actor_critic(
        env,
        softmax_policy,
        Q,
        X,
        GAMMA,
        ALPHA_P,
        ALPHA_Q,
        arena.size,
        N_EPISODES
    )

    plt.plot(results['durations'])
    plt.show()


if __name__ == '__main__':
    main()