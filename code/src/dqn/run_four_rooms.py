# Greg Attra
# 03.30.22

'''
Run DQN on the four rooms domain
'''

from typing import Callable
import torch
from lib.domains.four_rooms import FourRooms, FourRoomsOneHot, four_rooms_arena
from lib.dqn.algorithms import dqn
from lib.dqn.buffer import ReplayBuffer
from lib.dqn.nn import Dqn, optimize_dqn, simple_dqn_network


TARGET_UPDATE_FREQ: int = 1000
REPLAY_BUFFER_SIZE: int = 100000
BATCH_SIZE: int = 64
GAMMA: float = 0.99
N_EPISODES: int = 100
N_ACTIONS: int = 4
EPSILON: float = 0.1
LEARNING_RATE: float = 0.01


def main():
    arena = four_rooms_arena()
    env = FourRoomsOneHot(arena)
    input_size = arena.shape[0]*arena.shape[1]
    # input_size = 1
    output_size = env.action_space.n
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
    policy_net = Dqn(simple_dqn_network(input_size, output_size)).to(device)
    target_net = Dqn(simple_dqn_network(input_size, output_size)).to(device)

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.MSELoss()

    optimize: Callable[[Dqn, ReplayBuffer], None] = \
        lambda p_net, t_net, buff: optimize_dqn(
            p_net,
            t_net,
            buff,
            loss_fn,
            optimizer,
            GAMMA,
            BATCH_SIZE)

    results = dqn(
        env=env,
        policy_net=policy_net,
        target_net=target_net,
        buffer=buffer,
        optimize=optimize,
        target_update_freq=TARGET_UPDATE_FREQ,
        n_episodes=N_EPISODES,
        n_actions=N_ACTIONS,
        epsilon=EPSILON)


if __name__ == '__main__':
    main()
