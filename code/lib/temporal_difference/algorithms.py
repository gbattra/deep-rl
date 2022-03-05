# Greg Attra
# 03/05/2022

'''
Algorithms for TD learning
'''

from collections import defaultdict
from typing import Dict, List, Tuple
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


def compute_gain(
        Q: Dict[str, np.ndarray],
        n_steps: int,
        T: int,
        gamma: float,
        tau: int,
        obs: List[Tuple[Tuple[int, int], int, float, bool]]) \
        -> float:
    I = min(tau + n_steps, T)
    G = 0
    for i in range(tau + 1, I):
        _, _, r_t = obs[i]
        G += (gamma ** (i - tau - 1)) * r_t
    if tau + n_steps < T:
        s_t, a_t, _ = obs[tau + n_steps - 1]
        G += (gamma ** n_steps) * Q[s_t][a_t]
    return G


def n_step_sarsa(
        env: Env,
        n_steps: int,
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
        T = np.inf
        tau = -np.inf
        t = 0
        obs = []
        ep_G = 0
        while tau < T-1:
            if t < T:
                s_prime, r, done, _ = env.step(a)
                ep_G += r
                obs.append((s, a, r))
                if done:
                    T = t + 1
                else:
                    a = policy(s_prime)
                    s = s_prime
            tau = t - n_steps + 1
            if tau >= 0:
                G = compute_gain(Q, n_steps, T, gamma, tau, obs)
                s_tau, a_tau, _ = obs[tau]
                Q[s_tau][a_tau] += alpha * (G - Q[s_tau][a_tau])
            t += 1
        returns[e] = ep_G
    return returns
