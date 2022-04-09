# Greg Attra
# 04.07.22

'''
Implementation of policy gradient algorithms
'''


from typing import Callable
import gym
import numpy as np

from tqdm import trange

from lib.monte_carlo.algorithms import generate_episode


def reinforce(
        env: gym.Env,
        policy: Callable,
        features: Callable,
        gamma: float,
        alpha: float,
        n_features: int,
        n_episodes: int,
        n_steps: int):
    # initialize policy parameters
    O = np.zeros((n_features, env.action_space.n))

    for e in trange(n_episodes, 'Episode'):
        # generate an episode
        episode = generate_episode(
            env,
            lambda s: np.random.choice(
                enumerate(range(env.action_space.n)),
                p=policy(s))[0],
            n_steps)

        T = len(episode)
        G = 0
        for t in trange(T - 1, -1, -1, desc='Step'):
            s, a, r = episode[t]
            # compute return
            G = r + (gamma * G)

            # update policy parameters
            grad = features(s, a) + np.sum([(
                policy(s)[b] * features(s, b) for b in env.action_space.n)])

            # update policy weights
            O[:, a] = O[:, a] + (alpha * (gamma ** (T - t)) * G * grad)
