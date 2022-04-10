# Greg Attra
# 04.07.22

'''
Implementation of policy gradient algorithms
'''


from itertools import count
from typing import Callable, Dict
import gym
import numpy as np

from tqdm import trange

from lib.monte_carlo.algorithms import generate_episode


def reinforce(
        env: gym.Env,
        policy: Callable,
        X: Callable,
        gamma: float,
        alpha: float,
        n_features: int,
        n_episodes: int,
        n_steps: int) -> Dict:
    # initialize policy parameters
    O = np.zeros((n_features, env.action_space.n))

    durations = np.zeros(n_episodes)
    for e in trange(n_episodes, desc='Episode', leave=False):
        # generate an episode
        episode = generate_episode(
            env,
            lambda s: np.random.choice(
                range(env.action_space.n),
                p=policy(X(s), O)))

        T = len(episode)
        G = 0
        for t in trange(T, desc='Step', leave=False):
            s, a, r = episode[t]
            # print(f's: {s}')
            # compute return
            G = 0
            for k in range(t, T):
                _, _, r = episode[k]
                G += (gamma ** (k - t - 1)) * r

            pred = policy(X(s), O)

            # update policy parameters
            grad = X(s) - np.sum(
                np.array([
                    (policy(X(s), O)[b] * X(s)) for b in range(env.action_space.n)
                ]),
                axis=0)
            # print(grad)
            # update policy weights
            O[:, a] += alpha * (G - pred[a]) * X(s)
        
        durations[e] = T

    return {
        'O': O,
        'durations': durations
    }


def actor_critic(
        env: gym.Env,
        policy: Callable,
        Q: Callable,
        X: Callable,
        gamma: float,
        alpha_p: float,
        alpha_v: float,
        n_features: int,
        n_episodes: int) -> Dict:
    O = np.zeros((n_features, env.action_space.n))
    W = np.zeros((n_features, env.action_space.n))

    def choose_action(state) -> int:
        return np.random.choice(
                range(env.action_space.n),
                p=policy(X(state), O))

    durations = np.zeros(n_episodes)
    for e in trange(n_episodes, desc='Episode', leave=False):
        s = env.reset()
        I = 1.
        for t in count():
            a = choose_action(s)
            s_next, r, done, _ = env.step(a)
            a_next = choose_action(s_next)

            q_next = Q(X(s_next), W)[a_next] if not done else .0
            L = (r + (gamma * q_next)) - Q(X(s), W)[a]
            # print(L)
            W[:, a] += alpha_v * L * X(s)

            p_grad = X(s) - np.sum(
                np.array([
                    (policy(X(s), O)[b] * X(s)) for b in range(env.action_space.n)
                ]),
                axis=0)
            # print(p_grad)
            O[:, a] += alpha_p * I * L * X(s)

            I *= gamma
            s = s_next
            a = a_next

            if done:
                durations[e] = t
                break
    
    return {
        'O': O,
        'W': W,
        'durations': durations
    }