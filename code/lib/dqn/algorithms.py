# Greg Attra
# 03.30.22

'''
Algorithms for DQN
'''

from itertools import count
from typing import Callable, Dict
from gym import Env
import torch

from torch import nn
from tqdm import trange
from lib.dqn.buffer import ReplayBuffer, Transition
from lib.dqn.nn import Dqn, NeuralNetwork
from lib.dqn.policy import generate_dqn_policy


def dqn(
        env: Env,
        policy_net: Dqn,
        target_net: Dqn,
        buffer: ReplayBuffer,
        optimize: Callable[[Dqn, ReplayBuffer], None],
        target_update_freq: int,
        n_episodes: int,
        n_actions: int,
        epsilon: float) -> Dict:
    policy = generate_dqn_policy(dqn, n_actions, epsilon)

    # sync target net weights with policy net
    target_net.load_state_dict(policy_net.state_dict())

    for e in trange(n_episodes, desc='Episode', leave=False):
        s = env.reset()
        s_tensor = torch.from_numpy(s)
        for t in count():
            # choose action and step env
            a_tensor = policy(s_tensor)
            s_prime, r, done, _ = env.step(a_tensor.item())
            r_tensor = torch.tensor([r])

            # store transition in replay buffer
            buffer.add(Transition(s_tensor, a_tensor, r_tensor))

            # set current state as next state
            s_tensor = torch.from_numpy(s_prime)

            # one step optimization of policy net
            optimize(policy_net, buffer)

            if done:
                break
        
        # update target network periodically
        if e % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())
