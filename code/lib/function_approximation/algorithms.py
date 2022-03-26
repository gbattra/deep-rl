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

    for _ in trange(n_episodes, desc='Episode'):
        s = env.reset()
        x = X(s)
        a = policy(x, W)
        done = False
        i = 0
        # term = 1000000
        while not done:
            s_prime, r, done, _ = env.step(a)
            x_prime = X(s_prime)
            if done:
                W[:, a] = W[:, a] + ((alpha * (r - Q(x, W)[a])) * x)
            a_next = policy(x_prime, W)
            target = r + (gamma * Q(x_prime, W)[a_next])
            prediction = Q(x, W)[a]
            L = target - prediction
            W[:, a] = W[:, a] + ((alpha * L) * x)

            x = x_prime
            a = a_next
            i += 1
            # if i % term == 0:
            #     done = True

    return {'W': W}
