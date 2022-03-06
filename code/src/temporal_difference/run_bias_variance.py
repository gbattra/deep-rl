# Greg Attra
# 03/06/2022

'''
Executable for evaluating bias/variance tradeoff
'''

from collections import defaultdict
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

from typing import Callable, Dict, List
from lib.monte_carlo.algorithms import on_policy_mc_control_epsilon_soft
from lib.temporal_difference.algorithms import n_step_sarsa, sarsa
from lib.temporal_difference.windy_gridworld import WindyGridworld, windy_gridworld_1


ALPHA = .5
GAMMA = 1.
EPS = .1
N_EPISODES = int(10**3)
N_TRIALS = 10
N_STEPS = 4
N_EVAL_EPS = 100


@dataclass
class Experiment:
    algo_name: str
    algo: Callable[[WindyGridworld, Dict[str, np.ndarray], Dict[str, float], int, bool], Dict]
    n_episodes: int
    Q: Dict[str, np.ndarray]
    V: Dict[str, float]
    V_targets: Dict[str, np.ndarray]


def td_0(
        env: WindyGridworld,
        Q: Dict[str, np.ndarray],
        V: Dict[str, float],
        n_eps: int,
        learning: bool = True) -> Dict:
    return sarsa(
        env,
        ALPHA,
        EPS,
        GAMMA,
        n_eps,
        learning,
        Q,
        V)


def n_step_td(
        env: WindyGridworld,
        Q: Dict[str, np.ndarray],
        V: Dict[str, float],
        n_eps: int,
        learning: bool = True) -> Dict:
    return n_step_sarsa(
        env,
        N_STEPS,
        ALPHA,
        EPS,
        GAMMA,
        n_eps,
        learning,
        Q,
        V)


def mc_control(
        env: WindyGridworld,
        Q: Dict[str, np.ndarray],
        V: Dict[str, float],
        n_eps: int,
        learning: bool = True) -> Dict:
    return on_policy_mc_control_epsilon_soft(
        env,
        n_eps,
        GAMMA,
        EPS,
        learning,
        Q,
        V)


def plot_evaluation(title: str, V_targets: np.ndarray) -> None:
    plt.title(title)
    plt.hist(V_targets)
    plt.xlabel('V(S) Targets')
    plt.ylabel('Target Count')
    plt.show()


def main():
    # generate fixed policy
    wind_grid = windy_gridworld_1()
    env = WindyGridworld(wind_grid)
    print('Generating fixed policy')
    results = sarsa(env, ALPHA, EPS, GAMMA, N_EPISODES)
    Q = results['Q']

    # generate training experiments / learn V(S)
    algorithms = [
        ('TD(0)', td_0),
        ('n-Step TD', n_step_td),
        ('MC Control', mc_control)
    ]
    n_episodes = [1, 10, 50]
    training_experiments: List[Experiment] = []
    print('Running Training Experiments')
    for algo_name, algo in algorithms:
        for n_eps in n_episodes:
            print(f'{algo_name} - N episodes: {n_eps}')
            V = defaultdict(lambda: .0)
            results = algo(env, Q, V, n_eps)
            train_experiment = Experiment(
                algo_name=algo_name,
                algo=algo,
                n_episodes=n_eps,
                Q=Q,
                V=results['V'],
                V_targets=results['V_targets'])
            training_experiments.append(train_experiment)
    
    # run evaluation experiments using learned V(S)
    eval_experiments: List[Experiment] = []
    print('Running evaluation experiments')
    for train_experiment in training_experiments:
        print(f'Evaluating: {train_experiment.algo_name}')
        results = train_experiment.algo(
            env,
            Q,
            train_experiment.V,
            N_EVAL_EPS,
            False)
        eval_experiment = Experiment(
            algo_name=train_experiment.algo_name,
            algo=train_experiment.algo,
            n_episodes=train_experiment.n_episodes,
            Q=Q,
            V=train_experiment.V,
            V_targets=results['V_targets'])
        eval_experiments.append(eval_experiment)

    for evaluation in eval_experiments:
        plot_evaluation(
            evaluation.algo_name + ' N Episodes: ' + str(evaluation.n_episodes),
            evaluation.V_targets[env.reset()])


if __name__ == '__main__':
    main()
