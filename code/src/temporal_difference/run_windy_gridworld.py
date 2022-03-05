# Greg Attra
# 02/24/2022

'''
Executable for running MC and TD algorithms on Windy Gridworld
'''

from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from lib.temporal_difference.algorithms import q_learning

from lib.temporal_difference.windy_gridworld import (
    n_step_sarsa_windy_gridworld,
    windy_gridworld_1,
    WindyGridworld,
    monte_carlo_windy_gridworld,
    sarsa_windy_gridworld,
    q_learning_windy_gridworld)

ALPHA = .5
GAMMA = 1.
EPS = .1
N_EPISODES = int(10**3)
N_TRIALS = 10
N_STEPS = 4


def plot_returns(all_returns: np.ndarray) -> None:
    for label, color, returns in all_returns:
        avg_ret = np.average(returns, axis=0)
        plt.plot(avg_ret, color=color, label=label)
        
        stde_avg_ret = 1.96 * (np.std(avg_ret) / np.sqrt(N_TRIALS))
        y_neg = avg_ret - stde_avg_ret
        y_pos = avg_ret + stde_avg_ret
        plt.fill_between(
            range(N_EPISODES),
            y_neg,
            y_pos,
            alpha=0.2,
            color=color)
    
    plt.ylabel('Avg. Return')
    plt.xlabel('Episode')
    plt.legend()
    plt.show()

def plot_episode_lengths(all_episode_lengths: np.ndarray) -> None:
    for label, color, episode_lengths in all_episode_lengths:
        avg_ret = np.average(episode_lengths, axis=0)
        plt.plot(avg_ret, color=color, label=label)
    plt.ylabel('Avg. Episode Length')
    plt.xlabel('Episode')
    plt.legend()
    plt.show()


def run_monte_carlo_control():
    wind_grid = windy_gridworld_1()
    env = WindyGridworld(wind_grid)
    returns = monte_carlo_windy_gridworld(
        env,
        N_TRIALS,
        N_EPISODES,
        GAMMA,
        EPS)
    return returns


def run_sarsa() -> np.ndarray:
    wind_grid = windy_gridworld_1()
    env = WindyGridworld(wind_grid)
    returns = sarsa_windy_gridworld(
        env,
        N_TRIALS,
        N_EPISODES,
        GAMMA,
        EPS,
        ALPHA)
    return returns


def run_expected_sarsa() -> np.ndarray:
    wind_grid = windy_gridworld_1()
    env = WindyGridworld(wind_grid)
    returns = sarsa_windy_gridworld(
        env,
        N_TRIALS,
        N_EPISODES,
        GAMMA,
        EPS,
        ALPHA)
    return returns


def run_n_step_sarsa() -> np.ndarray:
    wind_grid = windy_gridworld_1()
    env = WindyGridworld(wind_grid)
    returns = n_step_sarsa_windy_gridworld(
        env,
        N_STEPS,
        N_TRIALS,
        N_EPISODES,
        GAMMA,
        EPS,
        ALPHA)
    return returns


def run_q_learning() -> np.ndarray:
    wind_grid = windy_gridworld_1()
    env = WindyGridworld(wind_grid)
    returns = q_learning_windy_gridworld(
        env,
        N_TRIALS,
        N_EPISODES,
        GAMMA,
        EPS,
        ALPHA)
    return returns


def main():
    algorithms = [
        ('Monte Carlo', run_monte_carlo_control, (1., .0, .0)),
        ('SARSA', run_sarsa, (.0, 1., .0)),
        ('Expected SARSA', run_expected_sarsa, (.0, .0, 1.)),
        ('n-Step SARSA', run_n_step_sarsa, (1., .0, 1.)),
        ('Q-Learning', run_q_learning, (.0, .0, .0))
    ]
    algo_returns = []
    algo_episode_lengths = []
    tranges = trange(len(algorithms), desc='Algorithm', leave=False)
    for a in tranges:
        label, algorithm, color = algorithms[a]
        tranges.set_description(f'Algorithm: {label}')
        returns, episode_lengths = algorithm()
        algo_returns.append((label, color, returns))
        algo_episode_lengths.append((label, color, episode_lengths))
    plot_returns(algo_returns)
    plot_episode_lengths(algo_episode_lengths)

if __name__ == '__main__':
    main()