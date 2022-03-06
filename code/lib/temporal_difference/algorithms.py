# Greg Attra
# 03/05/2022

'''
Algorithms for TD learning
'''

from collections import defaultdict
from typing import Dict, List, Optional, Tuple
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
        n_episodes: int,
        learning: bool = True,
        Q: Optional[Dict[str, np.ndarray]] = None,
        V: Optional[Dict[str, float]] = None) -> Dict:
    if Q is None:
        Q = defaultdict(lambda: np.zeros(env.action_space.n))
    if V is None:
        V = defaultdict(lambda: .0)
    V_targets = defaultdict(lambda: [])

    policy = create_epsilon_policy(Q, epsilon)
    returns = np.zeros(n_episodes)
    episode_lengths = np.zeros(n_episodes)
    for e in trange(n_episodes, desc='Epsiode', leave=False):
        s = env.reset()
        a = policy(s)
        G = 0
        i = 0
        done = False
        while not done:
            s_prime, r, done, _ = env.step(a)
            G = (gamma * G) + r
            a_prime = policy(s_prime)
            V_target = r + (gamma * V[s_prime])
            if learning:
                Q[s][a] = Q[s][a] + (alpha * (r + (gamma * Q[s_prime][a_prime]) - Q[s][a]))
                V[s] = V[s] + (alpha * (V_target - V[s]))
            V_targets[s].append(V_target)

            s = s_prime
            a = a_prime
            i += 1
        returns[e] = G
        episode_lengths[e] = i
    return {
        'Q': Q,
        'V': V,
        'returns': returns,
        'episode_lengths': episode_lengths,
        'V_targets': V_targets
    }


def expected_sarsa(env: Env,
        alpha: float,
        epsilon: float,
        gamma: float,
        n_episodes: int) -> Dict:
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = create_epsilon_policy(Q, epsilon)
    returns = np.zeros(n_episodes)
    episode_lengths = np.zeros(n_episodes)
    for e in trange(n_episodes, desc='Epsiode', leave=False):
        s = env.reset()
        a = policy(s)
        G = 0
        done = False
        i = 0
        while not done:
            s_prime, r, done, _ = env.step(a)
            G = (gamma * G) + r
            a_prime = policy(s_prime)
            action_weights = policy_probability_e_greedy(Q[s], epsilon)
            expected_rwd = np.dot(Q[s], action_weights)
            Q[s][a] = Q[s][a] + (alpha * (r + (gamma * expected_rwd) - Q[s][a]))
            s = s_prime
            a = a_prime
            i += 1
        returns[e] = G
        episode_lengths[e] = i
    return {
        'Q': Q,
        'returns': returns,
        'episode_lengths': episode_lengths
    }


def n_step_sarsa(
        env: Env,
        n_steps: int,
        alpha: float,
        epsilon: float,
        gamma: float,
        n_episodes: int,
        learning: bool = True,
        Q: Optional[Dict[str, np.ndarray]] = None,
        V: Optional[Dict[str, float]] = None) -> Dict:
    if Q is None:
        Q = defaultdict(lambda: np.zeros(env.action_space.n))
    if V is None:
        V = defaultdict(lambda: .0)
    V_targets = defaultdict(lambda: [])
    policy = create_epsilon_policy(Q, epsilon)
    returns = np.zeros(n_episodes)
    episode_lengths = np.zeros(n_episodes)

    def compute_gain(
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
                G = compute_gain(tau, obs)
                s_tau, a_tau, _ = obs[tau]
                if learning:
                    Q[s_tau][a_tau] = Q[s_tau][a_tau] + (alpha * (G - Q[s_tau][a_tau]))
                    V[s_tau] = V[s_tau] + (alpha * (G - V[s_tau]))
                V_targets[s_tau].append(G)
            t += 1
        returns[e] = ep_G
        episode_lengths[e] = t
    return {
        'Q': Q,
        'V': V,
        'V_targets': V_targets,
        'returns': returns,
        'episode_lengths': episode_lengths
    }


def q_learning(
        env: Env,
        alpha: float,
        epsilon: float,
        gamma: float,
        n_episodes: int) -> Dict:
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    b_policy = create_epsilon_policy(Q, epsilon)
    t_policy = create_epsilon_policy(Q, .0)
    returns = np.zeros(n_episodes)
    episode_lengths = np.zeros(n_episodes)

    for e in trange(n_episodes, desc='Epsiode', leave=False):
        s = env.reset()
        G = 0
        i = 0
        done = False
        while not done:
            a = b_policy(s)
            s_prime, r, done, _ = env.step(a)
            G = (gamma * G) + r
            a_prime = t_policy(s_prime)
            Q[s][a] = Q[s][a] + (alpha * (r + (gamma * Q[s_prime][a_prime]) - Q[s][a]))
            s = s_prime
            i += 1
        returns[e] = G
        episode_lengths[e] = i
    return {'Q': Q, 'returns': returns, 'episode_lengths': episode_lengths}
