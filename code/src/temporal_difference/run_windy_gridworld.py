# Greg Attra
# 02/24/2022

'''
Executable for running MC and TD algorithms on Windy Gridworld
'''

from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
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


def plot_returns(
    returns: np.ndarray,
    label: str,
    color: Tuple[float, float, float]) -> None:
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
    # mc_returns = run_monte_carlo_control()
    # plot_returns(mc_returns, 'Monte Carlo', (1., .0, .0))
    # sarsa_returns = run_sarsa()
    # expected_sarsa_returns = run_expected_sarsa()
    # returns = run_n_step_sarsa()
    returns = run_q_learning()
    plot_returns(returns, 'Q-Learning', (1., .0, .0))
    
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()