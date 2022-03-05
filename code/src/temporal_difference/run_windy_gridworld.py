# Greg Attra
# 02/24/2022

'''
Executable for running MC and TD algorithms on Windy Gridworld
'''

from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

from lib.temporal_difference.windy_gridworld import (
    windy_gridworld_1,
    WindyGridworld,
    monte_carlo_windy_gridworld,
    sarsa_windy_gridworld)

ALPHA = 0.5
GAMMA = 1.
EPS = .1
N_EPISODES = int(10**3)
N_TRIALS = 10


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
    mc_returns = monte_carlo_windy_gridworld(
        env,
        N_TRIALS,
        N_EPISODES,
        GAMMA,
        EPS)
    return mc_returns


def run_sarsa() -> np.ndarray:
    wind_grid = windy_gridworld_1()
    env = WindyGridworld(wind_grid)
    sarsa_returns = sarsa_windy_gridworld(
        env,
        N_TRIALS,
        N_EPISODES,
        GAMMA,
        EPS,
        ALPHA)
    return sarsa_returns


def run_expected_sarsa() -> np.ndarray:
    wind_grid = windy_gridworld_1()
    env = WindyGridworld(wind_grid)
    expected_sarsa_returns = sarsa_windy_gridworld(
        env,
        N_TRIALS,
        N_EPISODES,
        GAMMA,
        EPS,
        ALPHA)
    return expected_sarsa_returns

def main():
    # mc_returns = run_monte_carlo_control()
    # plot_returns(mc_returns, 'Monte Carlo', (1., .0, .0))
    # sarsa_returns = run_sarsa()
    expected_sarsa_returns = run_expected_sarsa()
    plot_returns(expected_sarsa_returns, 'Expected SARSA', (1., .0, .0))
    
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()