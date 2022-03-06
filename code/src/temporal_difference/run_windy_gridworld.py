# Greg Attra
# 02/24/2022

'''
Executable for running MC and TD algorithms on Windy Gridworld
'''

from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import trange


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


def plot_returns(all_returns: np.ndarray, title: str) -> None:
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
    plt.title(title)
    plt.legend()
    plt.show()

def plot_episode_lengths(all_episode_lengths: np.ndarray, title: str) -> None:
    for label, color, episode_lengths in all_episode_lengths:
        avg_ret = np.average(episode_lengths, axis=0)
        plt.plot(avg_ret, color=color, label=label)
    plt.ylabel('Avg. Episode Length')
    plt.xlabel('Episode')
    plt.title(title)
    plt.legend()
    plt.show()


def run_monte_carlo_control(env: WindyGridworld):
    returns, episode_lengths = monte_carlo_windy_gridworld(
        env,
        N_TRIALS,
        N_EPISODES,
        GAMMA,
        .25)
    return returns, episode_lengths


def run_sarsa(env: WindyGridworld) -> Tuple[np.ndarray, np.ndarray]:
    returns, episode_lengths = sarsa_windy_gridworld(
        env,
        N_TRIALS,
        N_EPISODES,
        GAMMA,
        EPS,
        ALPHA)
    return returns, episode_lengths


def run_expected_sarsa(env: WindyGridworld) -> Tuple[np.ndarray, np.ndarray]:
    returns, episode_lengths = sarsa_windy_gridworld(
        env,
        N_TRIALS,
        N_EPISODES,
        GAMMA,
        EPS,
        ALPHA)
    return returns, episode_lengths


def run_n_step_sarsa(env: WindyGridworld) -> Tuple[np.ndarray, np.ndarray]:
    returns, episode_lengths = n_step_sarsa_windy_gridworld(
        env,
        N_STEPS,
        N_TRIALS,
        N_EPISODES,
        GAMMA,
        EPS,
        ALPHA)
    return returns, episode_lengths


def run_q_learning(env: WindyGridworld) -> Tuple[np.ndarray, np.ndarray]:
    returns, episode_lengths = q_learning_windy_gridworld(
        env,
        N_TRIALS,
        N_EPISODES,
        GAMMA,
        EPS,
        ALPHA)
    return returns, episode_lengths


def run_experiment(env: WindyGridworld) -> Tuple[np.ndarray, np.ndarray]:
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
        returns, episode_lengths = algorithm(env)
        algo_returns.append((label, color, returns))
        algo_episode_lengths.append((label, color, episode_lengths))
    return algo_returns, algo_episode_lengths


def main():
    parser = argparse.ArgumentParser('Run a bandit testsuite')
    parser.add_argument(
        '--king-moves', help='Enable king moves', action='store_true')
    parser.add_argument(
        '--noop-action', help='Enable noop action', action='store_true')
    parser.add_argument(
        '--stochastic-wind', help='Enable stochastic wind', action='store_true')
    parser.add_argument(
        '--run-all', help='Run all experiments', action='store_true')
    parser.add_argument(
        '--max-t', help='Max episode timesteps', type=int, default=5000)
    args = parser.parse_args()

    if args.run_all:
        experiments = [
            (False, False, False, 5000),
            (True, False, False, 5000),
            (True, True, False, 5000),
            (True, True, True, 5000),
        ]

        all_returns = []
        all_episode_lengths = []
        experiment_titles = []
        for king_moves, noop_action, stochastic_wind, max_t in experiments:
            wind_grid = windy_gridworld_1()
            env = WindyGridworld(wind_grid, king_moves, noop_action, stochastic_wind, max_t)
            experiment_titles.append(env.to_string())
            algo_returns, algo_episode_lengths = run_experiment(env)
            all_returns.append(algo_returns)
            all_episode_lengths.append(algo_episode_lengths)
        
        for returns, episode_lengths, title in zip(all_returns, all_episode_lengths, experiment_titles):
            plot_returns(returns, title)
            plot_episode_lengths(episode_lengths, title)
        
        exit()
    else:
        wind_grid = windy_gridworld_1()
        env = WindyGridworld(wind_grid, args.king_moves, args.noop_action, args.stochastic_wind, args.max_t)
        returns, episode_lengths = run_experiment(env)
        plot_returns(returns, env.to_string())
        plot_episode_lengths(episode_lengths, env.to_string())
        exit()


if __name__ == '__main__':
    main()