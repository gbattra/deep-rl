# Greg Attra
# 02/24/2022

'''
Implementation of MC Evaluation for Blackjack
'''

import gym
import numpy as np

from typing import Callable, Dict, Tuple
from lib.monte_carlo.algorithms import generate_episode

def space_dims(space: gym.Space) -> Tuple:
    '''
    Convert a gym Space object into an n-dim array
    '''
    shape = ()
    for dim in space:
        shape = shape + (dim.n, )
    return shape


def first_visit_mc_eval(
        policy: Callable,
        env: gym.Env,
        discount_factor: float,
        n_eps: int) -> np.ndarray:
    obs_space_dims = space_dims(env.observation_space)

    V = np.zeros(obs_space_dims)
    R = np.zeros(obs_space_dims)
    N = np.zeros(obs_space_dims)

    i = 0
    while i < n_eps:
        steps = generate_episode(env, policy)
        G = 0
        episode_R = np.zeros(obs_space_dims)
        episode_N = np.zeros(obs_space_dims)
        for s, a, r in reversed(steps):
            G = (discount_factor * G) + r

            # convert state space to dims
            idx = ()
            for dim in range(len(obs_space_dims)):
                idx = idx + (int(s[dim]),)

            # we can avoid searching through the earlier steps
            # to see if the state is reached earlier for first-vist
            # by sipmly overriding our existing reward for s for this
            # episode only. Then after the episode ends, we update R & V
            episode_R[idx] = G
            episode_N[idx] = 1.

        # print(episode_R[:, :, 1])
        R = R + episode_R
        # print(R[:, :, 1])
        N = N + episode_N
        # print(N[:, :, 1])

        i += 1

    # print(R[:, :, 1])
    N[N < 1] = 1.
    V = R / N
    return V
