# Greg Attra
# 03.30.22

'''
Run DQN on the four rooms domain
'''

from typing import Callable
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
from lib.dqn.utils import plot_durations
from lib.domains.four_rooms import FourRooms, FourRoomsArena, FourRoomsCoords, FourRoomsOneHot, four_rooms_arena
from lib.dqn.algorithms import dqn
from lib.dqn.buffer import ReplayBuffer
from lib.dqn.nn import Dqn, optimize_dqn


TARGET_UPDATE_FREQ: int = 10000
REPLAY_BUFFER_SIZE: int = 100000
BATCH_SIZE: int = 64
GAMMA: float = 0.99
N_EPISODES: int = 1000
N_TRIALS: int = 2
EPSILON_START: float = .1
EPSILON_END: float = .1
EPSILON_DECAY: float = 1.
LEARNING_RATE: float = .002
N_STEPS: int = 2000


def simple_dqn_network(input_size: int, dim_size: int, output_size: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_size, 64),
        nn.ReLU(),
        nn.Linear(64, output_size)
    )


def main():
    arena = four_rooms_arena()
    env = FourRoomsOneHot(arena, noise=0.2)
    # env = FourRoomsCoords(arena)
    # env = FourRooms(arena)
    # env = FourRoomsArena(arena)
    input_size = env.observation_size
    output_size = env.action_space.n
    dim_size = 64
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Device: {device}')

    total_results = np.zeros((N_TRIALS, N_EPISODES))
    for t in trange(N_TRIALS, desc='Trial', leave=False):
        buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
        policy_net = Dqn(simple_dqn_network(input_size, dim_size, output_size)).to(device)
        target_net = Dqn(simple_dqn_network(input_size, dim_size, output_size)).to(device)

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
            plotter=plot_durations,
            target_update_freq=TARGET_UPDATE_FREQ,
            n_episodes=N_EPISODES,
            n_steps=N_STEPS,
            epsilon_start=EPSILON_START,
            epsilon_end=EPSILON_END,
            epsilon_decay=EPSILON_DECAY,
            render=False)
        total_results[t, :] = results['durations']

    avg_ret = np.average(total_results, axis=0)
    plt.plot(avg_ret, color=(1., .0, .0))
    
    stde_avg_ret = 1.96 * (np.std(total_results, axis=0) / np.sqrt(N_EPISODES))
    y_neg = avg_ret - stde_avg_ret
    y_pos = avg_ret + stde_avg_ret
    plt.fill_between(
        range(N_EPISODES),
        y_neg,
        y_pos,
        alpha=0.2,
        color=(1., .0, .0))

    plt.xlabel('Episode')
    plt.ylabel('Avg Episode Length')
    plt.title('Four Rooms DQN')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
