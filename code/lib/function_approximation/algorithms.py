# Greg Attra
# 03/25/22

'''
Function approximation algos
'''


from gym import Env
import numpy as np

from typing import Callable

from lib.function_approximation.policy import generate_policy, gradients


def semigrad_onestep_sarsa(
        env: Env,
        Q: Callable,
        feature_extractor: Callable,
        n_feats: int,
        alpha: float,
        epsilon: float,
        gamma: float,
        n_episodes: int):
    W = np.zeros(n_feats)
    
    policy = generate_policy(Q, epsilon)

    for _ in range(n_episodes):
        s = env.reset()
        s_feat = feature_extractor(s)
        a = policy(s_feat, W)
        done = False
        while not done:
            s_prime, r, done, _ = env.step(a)
            s_prime_feat = feature_extractor(s_prime)
            if done:
                W[:, a] = W[:, a] - (alpha * (r - Q(s_feat, W)[a])) * gradients(s_feat, W, r, Q(s_feat, W)[a], a)
            a_next = policy(s_prime_feat, W)
            target = r + (gamma * Q(s_prime_feat, W)[a_next])
            prediction = Q(s_feat, W)[a]
            W[: a] = W[:, a] - (alpha * (target - prediction)) * gradients(s_feat, W, target, prediction, a)

            s_feat = s_prime_feat
            a = a_next
