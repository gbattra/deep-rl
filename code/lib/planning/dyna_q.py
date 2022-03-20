# Greg Attra
# 03.12.2022

'''
Implementation of Dyna-Q algorithms
'''

import random
from typing import Dict, Tuple
import numpy as np

from gym import Env
from collections import defaultdict

from tqdm import trange
from lib.utils.policy import create_epsilon_policy


def dyna_q_plus(
        env: Env,
        k: float,
        alpha: float,
        epsilon: float,
        gamma: float,
        n_plan_steps: int,
        n_timesteps: int) -> Dict:
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    M = defaultdict(lambda: [(0, None, 0) for _ in range(env.action_space.n)])

    policy = create_epsilon_policy(Q, epsilon)
    s = env.reset()
    n_eps = 0
    timestep_stats = np.zeros(n_timesteps)
    for t in trange(n_timesteps, desc='Timestep', leave=False):
        a = policy(s)
        s_next, r, done, _ = env.step(a)
        Q[s][a] = Q[s][a] + (alpha * (r + (gamma * Q[s_next][np.argmax(Q[s_next])]) - Q[s][a]))
        M[s][a] = (r, s_next, t)

        for _ in range(n_plan_steps):
            s_sample = env.observation_space.sample()
            a_sample = env.action_space.sample()
            r_prime, s_prime, t_prime = M[s_sample][a_sample]
            if s_prime is None:
                s_prime = s_sample
            tau = t - t_prime
            r_bonus = r_prime + (k * np.sqrt(tau) if tau > 1 else 0)
            Q[s_sample][a_sample] = Q[s_sample][a_sample] + \
                (alpha * (r_bonus + (gamma * Q[s_prime][np.argmax(Q[s_prime])]) - Q[s_sample][a_sample]))
        if done:
            n_eps += 1
        timestep_stats[t] = n_eps
        s = s_next

    return {'Q': Q, 'M': M, 'stats': timestep_stats}


def stochastic_dyna_q(
        env: Env,
        k: float,
        alpha: float,
        epsilon: float,
        gamma: float,
        n_plan_steps: int,
        n_timesteps: int) -> Dict:
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    M = defaultdict(lambda: [defaultdict(lambda: [0, 0]) for _ in range(env.action_space.n)])
    T = defaultdict(lambda: [defaultdict(lambda: 0) for _ in range(env.action_space.n)])
    N = defaultdict(lambda: [0 for _ in range(env.action_space.n)])

    policy = create_epsilon_policy(Q, epsilon)
    s = env.reset()
    n_eps = 0
    timestep_stats = np.zeros(n_timesteps)
    for t in trange(n_timesteps, desc='Timestep', leave=False):
        a = policy(s)
        s_next, r, done, _ = env.step(a)

        Q[s][a] = Q[s][a] + (alpha * (r + (gamma * Q[s_next][np.argmax(Q[s_next])]) - Q[s][a]))
        T[s][a][s_next] += 1
        M[s][a][s_next] = r, t
        N[s][a] += 1

        for _ in range(n_plan_steps):
            s_sample = env.observation_space.sample()
            a_sample = env.action_space.sample()
            r_total = 0
            for s_prime in M[s_sample][a_sample].keys():
                r_prime, t_prime = M[s_sample][a_sample][s_prime]
                tau = t - t_prime
                bonus = r_prime + (k * np.sqrt(tau) if tau > 1 else 0)
                r_total += (T[s_sample][a_sample][s_prime] / N[s_sample][a_sample]) * bonus
            Q[s_sample][a_sample] = Q[s_sample][a_sample] + \
                (alpha * (r_total + (gamma * Q[s_prime][np.argmax(Q[s_prime])]) - Q[s_sample][a_sample]))
        if done:
            n_eps += 1
        timestep_stats[t] = n_eps
        s = s_next

    return {'Q': Q, 'M': M, 'stats': timestep_stats}
