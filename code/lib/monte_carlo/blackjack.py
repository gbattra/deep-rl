# Greg Attra
# 02/24/2022

'''
Implementation of MC Evaluation for Blackjack
'''

from collections import defaultdict
from copy import deepcopy
from email import policy
import gym
import numpy as np

from typing import Callable, Dict, Tuple

from tqdm import trange
from lib.monte_carlo.policy import create_blackjack_policy
from lib.monte_carlo.algorithms import generate_episode

def space_dims(space: gym.Space) -> Tuple:
    '''
    Convert a gym Space object into an n-dim array
    '''
    shape = ()
    for dim in space:
        shape = shape + (dim.n, )
    return shape


def first_visit_mc(
        env: gym.Env,
        discount_factor: float,
        n_eps: int) -> np.ndarray:

    policy = create_blackjack_policy(dict())
    obs_space_dims = space_dims(env.observation_space)

    V = np.zeros(obs_space_dims)
    R = np.zeros(obs_space_dims)
    N = np.zeros(obs_space_dims)

    for _ in trange(n_eps, desc='Episode:'):
        episode = generate_episode(env, policy)
        G = 0
        episode_R = np.zeros(obs_space_dims)
        episode_N = np.zeros(obs_space_dims)
        for t in range(len(episode) - 1, -1, -1):
            s, a, r = episode[t]
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

        R = R + episode_R
        N = N + episode_N

    N[N < 1] = 1.
    V = R / N
    return V, policy


def first_visit_mc_es(
        env: gym.Env,
        gamma: float,
        n_eps: int) -> Tuple[defaultdict, Callable]:

    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))

    policy = create_blackjack_policy(Q)

    for _ in trange(n_eps, desc='Episode:'):
        episode = generate_episode(env, policy, True)
        G = 0

        q = Q.copy()
        n = N.copy()

        for t in range(len(episode) - 1, -1, -1):
            s, a, r = episode[t]
            G = (gamma * G) + r

            # instead of checking if s encountered earlier, just override current value
            n[s][a] = N[s][a] + 1
            q[s][a] = Q[s][a] + ((1. / float(n[s][a])) * (G - Q[s][a]))
        
        for s in q:
            N[s] = n[s]
            Q[s] = q[s]

    return Q, policy
