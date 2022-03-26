# Greg Attra
# 03/25/22

'''
Function approximation algos
'''


from gym import Env
import numpy as np

from typing import Callable, Dict

from lib.function_approximation.policy import generate_policy, gradients


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

    for _ in range(n_episodes):
        s = env.reset()
        x = X(s)
        a = policy(x, W)
        done = False
        while not done:
            s_prime, r, done, _ = env.step(a)
            x_prime = X(s_prime)
            if done:
                W[:, a] = W[:, a] - ((alpha * (r - Q(x, W)[a])) * x)
            a_next = policy(x_prime, W)
            target = r + (gamma * Q(x_prime, W)[a_next])
            prediction = Q(x, W)[a]
            W[: a] = W[:, a] - ((alpha * (target - prediction)) * x)

            x = x_prime
            a = a_next

    return {'W': W}
