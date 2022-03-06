import gym
from typing import Callable, Dict, Optional, Tuple
from collections import defaultdict
from tqdm import trange
import numpy as np
from lib.monte_carlo.policy import create_blackjack_policy, create_epsilon_policy


def generate_episode(env: gym.Env, policy: Callable, es: bool = False):
    """A function to generate one episode and collect the sequence of (s, a, r) tuples

    This function will be useful for implementing the MC methods

    Args:
        env (gym.Env): a Gym API compatible environment
        policy (Callable): A function that represents the policy.
        es (bool): Whether to use exploring starts or not
    """
    episode = []
    state = env.reset()
    while True:
        if es and len(episode) == 0:
            action = env.action_space.sample()
        else:
            action = policy(state)

        next_state, reward, done, _ = env.step(action)
        episode.append((state, action, reward))
        if done:
            break
        state = next_state

    return episode


def on_policy_mc_control_epsilon_soft(
        env: gym.Env,
        num_episodes: int,
        gamma: float,
        epsilon: float,
        learning: bool = True,
        Q: Optional[Dict[str, np.ndarray]] = None,
        V: Optional[Dict[str, float]] = None) -> np.ndarray:
    if Q is None:
        Q = defaultdict(lambda: np.zeros(env.action_space.n))
    if V is None:
        V = defaultdict(lambda: .0)
    N = defaultdict(lambda: np.zeros(env.action_space.n))
    V = defaultdict(lambda: 0)
    V_targets = defaultdict(lambda: [])

    policy = create_epsilon_policy(Q, epsilon)
    returns = np.zeros(num_episodes)
    episode_lengths = np.zeros(num_episodes)
    
    for i in trange(num_episodes, desc="Episode", leave=False):
        episode = generate_episode(env, policy)
        G = 0

        for t in range(len(episode) - 1, -1, -1):
            s, a, r = episode[t]
            G = (gamma * G) + r
            V_targets.append(G)
            N[s][a] = N[s][a] + 1
            if learning:
                Q[s][a] = Q[s][a] + ((1. / N[s][a]) * (G - Q[s][a]))
                V[s] = V[s] + ((1./np.sum(N[s])) * (G - V[s]))
            V_targets[s].append(G)

        returns[i] = G
        episode_lengths[i] = len(episode)
    return {
        'Q': Q,
        'V': V,
        'returns': returns,
        'episode_lengths': episode_lengths,
        'V_targets': V_targets
    }



def off_policy_mc_control(
    env: gym.Env, num_episodes: int, gamma: float, epsilon: float) -> np.ndarray:
    """On-policy Monte Carlo policy control for epsilon soft policies.

    Args:
        env (gym.Env): a Gym API compatible environment
        num_episodes (int): Number of episodes
        gamma (float): Discount factor of MDP
        epsilon (float): Parameter for epsilon soft policy (0 <= epsilon <= 1)
    Returns:

    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    C = defaultdict(lambda: np.zeros(env.action_space.n))

    t_policy = create_epsilon_policy(Q, .0)
    b_policy = create_epsilon_policy(Q, epsilon)

    b_returns = np.zeros(num_episodes)
    t_returns = np.zeros(num_episodes)
    
    for i in trange(num_episodes, desc="Episode", leave=False):
        episode = generate_episode(env, b_policy)

        G = 0.
        W = 1.
        for t in range(len(episode) - 1, -1, -1):
            s, a, r = episode[t]
            G = (gamma * G) + r
            C[s][a] += W
            Q[s][a] = Q[s][a] + ((W / C[s][a]) * (G - Q[s][a]))
            
            p_pi = epsilon / float(env.action_space.n) if a != t_policy(s) else 1. - epsilon + (epsilon / float(env.action_space.n))
            p_b = 1. - epsilon if a == t_policy(s) else epsilon
            W = W * (p_pi / p_b)
            if W == .0:
                break

        b_returns[i] = G

        # do a rollout
        if i % 100 == 0:
            t_episode = generate_episode(env, t_policy)
            t_G = 0

            for e in range(len(t_episode) - 1, -1, -1):
                _, _, t_r = t_episode[e]
                t_G = (gamma * t_G) + t_r
            t_returns[i:i*100] = t_G

    return b_returns, t_returns