# Greg Attra
# 03/25/22

'''
Feature extractors
'''

import numpy as np

from typing import Callable


def get_feature_extractor(
        N: int,
        extract: Callable,
        aggregator: Callable) -> Callable:
    return lambda s: extract(aggregator(s), N)


def one_hot_encode(
        s: int,
        feat_size: int) -> np.ndarray:
    enc = np.zeros(feat_size)
    enc[s] = 1
    return enc


def xy_features(
        s: int,
        x_dim: int) -> np.ndarray:
    y = s // x_dim
    x = s % x_dim
    return np.array([y, x, 1.])
