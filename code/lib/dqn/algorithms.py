# Greg Attra
# 03.30.22

'''
Algorithms for DQN
'''

from itertools import count
from typing import Callable, Dict, List
from gym import Env
import torch
import numpy as np
from torch import nn
from tqdm import trange
from lib.dqn.buffer import ReplayBuffer, Transition
from lib.dqn.nn import Dqn, NeuralNetwork
from lib.dqn.policy import generate_dqn_policy

MAX_ITER = 5000

def dqn(
        env: Env,
        policy_net: Dqn,
        target_net: Dqn,
        buffer: ReplayBuffer,
        optimize: Callable[[Dqn, Dqn, ReplayBuffer], None],
        plotter: Callable[[List[int], str], None],
        target_update_freq: int,
        n_episodes: int,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay: float) -> Dict:
    policy = generate_dqn_policy(
        policy_net,
        env.action_space.n,
        epsilon_start,
        epsilon_end,
        epsilon_decay)

    # sync target net weights with policy net
    target_net.load_state_dict(policy_net.state_dict())

    durations = []
    rewards = []
    for e in trange(n_episodes, desc='Episode', leave=False):
        s = env.reset()
        total_rwd = 0
        for t in count():
            # choose action and step env
            a = policy(s, sum(durations) + t)
            s_prime, r, done, _ = env.step(a)
            env.render()
            total_rwd += r

            # store transition in replay buffer
            buffer.add(Transition(s, a, s_prime, r, done))

            # set current state as next state
            s = s_prime

            # one step optimization of policy net
            optimize(policy_net, target_net, buffer)
        
            # update target network periodically
            if t % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done or t > MAX_ITER:
                # optimize(policy_net, target_net, buffer)
                durations.append(t+1)
                rewards.append(total_rwd)
                plotter([durations], ['durations'])
                break

    return {'durations': durations, 'rewards': rewards}
