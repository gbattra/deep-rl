# Greg Attra
# 03/25/22

'''
Function approximation algos
'''


from gym import Env
import numpy as np

from typing import Callable, Dict

from tqdm import trange

from lib.function_approximation.policy import generate_policy


def semigrad_onestep_sarsa(
        env: Env,
        Q: Callable,
        X: Callable,
        n_feats: int,
        alpha: float,
        epsilon: float,
        gamma: float,
        n_episodes: int) -> Dict:
    W = np.zeros((n_feats, env.action_space.n))
    policy = generate_policy(Q, epsilon)

    episode_lengths = np.zeros(n_episodes)
    for e in trange(n_episodes, desc='Episode', leave=False):
        s = env.reset()
        x = X(s)
        a = policy(x, W)
        done = False
        t = 0
        while not done:
            s_prime, r, done, _ = env.step(a)
            x_prime = X(s_prime)
            a_next = policy(x_prime, W)
            target = r + (gamma * Q(x_prime, W)[a_next])
            prediction = Q(x, W)[a]
            L = target - prediction
            W[:, a] += alpha * L * x
            x = x_prime
            a = a_next
            t += 1
        episode_lengths[e] = t
        

    return {'W': W, 'episode_lengths': episode_lengths}
