# Greg Attra
# 03.12.2022

'''
Implementation of Dyna-Q algorithms
'''

import random
from typing import Dict
import numpy as np

from gym import Env
from collections import defaultdict

from tqdm import trange
from lib.utils.policy import create_epsilon_policy


def dyna_q(
        env: Env,
        alpha: float,
        epsilon: float,
        gamma: float,
        n_plan_steps: int,
        n_timesteps: int) -> Dict:
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    M = defaultdict(lambda: {'r': 0, 's': (0, 0)})

    sa_samples = set()
    policy = create_epsilon_policy(Q, epsilon)
    s = env.reset()
    n_eps = 0
    timestep_stats = np.zeros(n_timesteps)
    for t in trange(n_timesteps, desc='Timestep', leave=False):
        a = policy(s)
        sa_samples.add((s, a))
        s_next, r, done, _ = env.step(a)
        Q[s][a] = Q[s][a] + (alpha * (r + (gamma * Q[s_next][np.argmax(Q[s_next])]) - Q[s][a]))
        M[s + (a,)] = {'r': r, 's': s_next}
        for _ in range(n_plan_steps):
            s_sample, a_sample = random.sample(sa_samples, 1)[0]
            r_prime, s_prime = M[s_sample + (a_sample,)]['r'], M[s_sample + (a_sample,)]['s']
            Q[s_sample][a_sample] = Q[s_sample][a_sample] + \
                (alpha * (r_prime + (gamma * Q[s_prime][np.argmax(Q[s_prime])]) - Q[s_sample][a_sample]))
        if done:
            n_eps += 1
        timestep_stats[t] = n_eps
        s = s_next

    return {'Q': Q, 'M': M, 'stats': timestep_stats}

def dyna_q_plus(
        env: Env,
        k: float,
        alpha: float,
        epsilon: float,
        gamma: float,
        n_plan_steps: int,
        n_timesteps: int) -> Dict:
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))
    M = defaultdict(lambda: {'r': 0, 's': None})

    sa_samples = set()
    policy = create_epsilon_policy(Q, epsilon)
    s = env.reset()
    n_eps = 0
    timestep_stats = np.zeros(n_timesteps)
    for t in trange(n_timesteps, desc='Timestep', leave=False):
        a = policy(s)
        sa_samples.add(s)
        s_next, r, done, _ = env.step(a)

        curr_t = N[s][a]
        N[s][a] = t

        tau = t - curr_t
        r_bonus = r + (k * np.sqrt(tau + .1e-100))

        Q[s][a] = Q[s][a] + (alpha * (r_bonus + (gamma * Q[s_next][np.argmax(Q[s_next])]) - Q[s][a]))
        M[s + (a,)] = {'r': r_bonus, 's': s_next}

        for _ in range(n_plan_steps):
            s_sample = random.sample(sa_samples, 1)[0]
            a_sample = random.sample(range(env.action_space.n), 1)[0]
            r_prime, s_prime = M[s_sample + (a_sample,)]['r'], M[s_sample + (a_sample,)]['s']
            if s_prime is None:
                s_prime = s_sample
            Q[s_sample][a_sample] = Q[s_sample][a_sample] + \
                (alpha * (r_prime + (gamma * Q[s_prime][np.argmax(Q[s_prime])]) - Q[s_sample][a_sample]))
        if done:
            n_eps += 1
        timestep_stats[t] = n_eps
        s = s_next

    return {'Q': Q, 'M': M, 'stats': timestep_stats}
