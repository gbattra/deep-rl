# Greg Attra
# 03/06/2022

'''
Executable for evaluating bias/variance tradeoff
'''

from collections import defaultdict
from dataclasses import dataclass
import numpy as np

from typing import Callable, Dict, List
from lib.temporal_difference.algorithms import sarsa
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
    algo: Callable[[Dict[str, np.ndarray], Dict[str, float], int, bool], Dict]
    n_episodes: int
    Q: Dict[str, np.ndarray]
    V: Dict[str, float]
    V_targets: Dict[str, np.ndarray]


def td_0(
        Q: Dict[str, np.ndarray],
        V: Dict[str, float],
        n_eps: int,
        learning: bool = True) -> Dict:
    pass


def n_step_td(
        Q: Dict[str, np.ndarray],
        V: Dict[str, float],
        n_eps: int,
        learning: bool = True) -> Dict:
    pass


def mc_control(
        Q: Dict[str, np.ndarray],
        V: Dict[str, float],
        n_eps: int,
        learning: bool = True) -> Dict:
    pass


def main():
    # generate fixed policy
    wind_grid = windy_gridworld_1()
    env = WindyGridworld(wind_grid)
    results = sarsa(env, ALPHA, EPS, GAMMA, N_EPISODES)
    Q = results['Q']

    # generate training experiments / learn V(S)
    algorithms = [
        ('TD(0)', td_0),
        ('n-Step Control', n_step_td)
        ('MC Control', mc_control)
    ]
    n_episodes = [1, 10, 50]
    training_experiments: List[Experiment] = []
    for algo_name, algo in algorithms:
        for n_eps in n_episodes:
            V = defaultdict(lambda: .0)
            results = algo(Q, V, n_eps)
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
    for train_experiment in training_experiments:
        results = train_experiment.algo(
            Q,
            train_experiment.V,
            N_EVAL_EPS,
            False)
        eval_experiment = Experiment(
            algo_name=train_experiment.algo_name,
            algo=train_experiment.algo,
            n_episodes=100,
            Q=Q,
            V=train_experiment.V,
            V_targets=results['V_targets'])
        eval_experiments.append(eval_experiment)


if __name__ == '__main__':
    main()
