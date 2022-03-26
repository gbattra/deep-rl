# Greg Attra
# 03/26/22

'''
Executable for running linear function approximation on the four rooms problem
'''

from lib.function_approximation.aggregators import get_aggregator, segment
from lib.function_approximation.algorithms import semigrad_onestep_sarsa
from lib.function_approximation.features import get_feature_extractor, one_hot_encode
from lib.function_approximation.four_rooms import FourRooms, four_rooms_arena
from lib.function_approximation.policy import q_function


ALPHA: float = 0.01
GAMMA: float = 0.9
EPSILON: float = 0.1
N_EPISODES: int = 100


def main():
    arena = four_rooms_arena()
    env = FourRooms(arena)
    seg_size = 1
    n_feats = int(arena.size / seg_size)
    segmentor = get_aggregator(seg_size, segment)
    X = get_feature_extractor(
        n_feats,
        one_hot_encode,
        segmentor)
    Q = q_function
    W = semigrad_onestep_sarsa(
        env,
        Q,
        X,
        n_feats,
        ALPHA,
        EPSILON,
        GAMMA,
        N_EPISODES)
    

if __name__ == '__main__':
    main()
