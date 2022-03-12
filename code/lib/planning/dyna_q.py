# Greg Attra
# 03.12.2022

'''
Implementation of Dyna-Q algorithms
'''

import random
import numpy as np

from gym import Env
from collections import defaultdict
from lib.utils.policy import create_epsilon_policy


def dyna_q(
        env: Env,
        alpha: float,
        epsilon: float,
        gamma: float,
        n_planning: int):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    M = defaultdict(lambda: (0, None))

    sa_samples = set()
    policy = create_epsilon_policy(Q, epsilon)
    s = env.reset()
    t = 0
    while True:
        a = policy(s)
        sa_samples.add((s, a))
        s_next, r, done, _ = env.step(a)
        Q[s][a] = Q[s][a] + (alpha * (r + (gamma * np.argmax(Q[s_next]) - Q[s][a])))
        M[s][a] = {'r': r, 's_next': s_next}
        for _ in range(n_planning):
            s_sample, a_sample = random.sample(sa_samples)
            r_prime, s_prime = M[s_sample][a_sample]
            Q[s_sample][a_sample] = Q[s_sample][a_sample] + \
                (alpha * (r_prime + (gamma * np.argmax(Q[s_prime]) - Q[s_sample][a_sample])))
        s = s_next
        t += 1

def dyna_q_plus():
    pass
