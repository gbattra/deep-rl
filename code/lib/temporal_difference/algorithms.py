# Greg Attra
# 03/05/2022

'''
Algorithms for TD learning
'''

from collections import defaultdict
import numpy as np

from gym import Env
from tqdm import trange
from lib.utils.policy import (
    create_epsilon_policy,
    policy_probability_e_greedy)


def sarsa(
        env: Env,
        alpha: float,
        epsilon: float,
        gamma: float,
        n_episodes: int) -> np.ndarray:
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    policy = create_epsilon_policy(Q, epsilon)
    returns = np.zeros(n_episodes)
    for e in trange(n_episodes, desc='Epsiode', leave=False):
        s = env.reset()
        a = policy(s)
        G = 0
        done = False
        while not done:
            s_prime, r, done, _ = env.step(a)
            G = (gamma * G) + r
            a_prime = policy(s_prime)
            Q[s][a] = Q[s][a] + (alpha * (r + (gamma * Q[s_prime][a_prime]) - Q[s][a]))
            s = s_prime
            a = a_prime
        returns[e] = G
    return returns


def expected_sarsa(env: Env,
        alpha: float,
        epsilon: float,
        gamma: float,
        n_episodes: int) -> np.ndarray:
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    policy = create_epsilon_policy(Q, epsilon)
    returns = np.zeros(n_episodes)
    for e in trange(n_episodes, desc='Epsiode', leave=False):
        s = env.reset()
        a = policy(s)
        G = 0
        done = False
        while not done:
            s_prime, r, done, _ = env.step(a)
            G = (gamma * G) + r
            a_prime = policy(s_prime)
            action_weights = policy_probability_e_greedy(Q[s], epsilon)
            expected_rwd = np.dot(Q[s], action_weights)
            Q[s][a] = Q[s][a] + (alpha * (r + (gamma * expected_rwd) - Q[s][a]))
            s = s_prime
            a = a_prime
        returns[e] = G
    return returns
