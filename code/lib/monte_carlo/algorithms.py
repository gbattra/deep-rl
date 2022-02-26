import gym
from typing import Callable, Tuple
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
    N = defaultdict(lambda: np.zeros(env.action_space.n))

    policy = create_epsilon_policy(Q, epsilon)
    returns = np.zeros(num_episodes)
    
    for i in trange(num_episodes, desc="Episode", leave=False):
        # TODO Q4
        # For each episode calculate the return
        # Update Q
        # Note there is no need to update the policy here directly.
        # By updating Q, the policy will automatically be updated.
        episode = generate_episode(env, policy)
        G = 0

        q = Q.copy()
        n = N.copy()

        for t in range(len(episode) - 1, -1, -1):
            s, a, r = episode[t]
            G = (gamma * G) + r
            
            n[s][a] = N[s][a] + 1
            q[s][a] = Q[s][a] + ((1. / float(n[s][a])) * (G - Q[s][a]))

        returns[i] = G
        for s in q:
            N[s] = n[s]
            Q[s] = q[s]
    return returns



def off_policy_mc_control_epsilon_soft(
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
    N = defaultdict(lambda: np.zeros(env.action_space.n))

    b_policy = create_epsilon_policy(Q, epsilon)
    t_policy = create_epsilon_policy(Q, .0)
    b_returns = np.zeros(num_episodes)
    t_returns = np.zeros(num_episodes)
    
    for i in trange(num_episodes, desc="Episode", leave=False):
        # TODO Q4
        # For each episode calculate the return
        # Update Q
        # Note there is no need to update the policy here directly.
        # By updating Q, the policy will automatically be updated.
        episode = generate_episode(env, b_policy)
        G = 0

        q = Q.copy()
        n = N.copy()

        for t in range(len(episode) - 1, -1, -1):
            s, a, r = episode[t]
            G = (gamma * G) + r
            
            n[s][a] = N[s][a] + 1
            q[s][a] = Q[s][a] + ((1. / float(n[s][a])) * (G - Q[s][a]))

        b_returns[i] = G
        for s in q:
            N[s] = n[s]
            Q[s] = q[s]

    for i in trange(num_episodes, desc="Episode", leave=False):
        # TODO Q4
        # For each episode calculate the return
        # Update Q
        # Note there is no need to update the policy here directly.
        # By updating Q, the policy will automatically be updated.
        episode = generate_episode(env, t_policy)
        G = 0

        q = Q.copy()
        n = N.copy()

        for t in range(len(episode) - 1, -1, -1):
            s, a, r = episode[t]
            G = (gamma * G) + r
        t_returns[t] = G
        
    return b_returns, t_returns