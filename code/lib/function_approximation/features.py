# Greg Attra
# 03/25/22

'''
Feature extractors
'''

import numpy as np

from typing import Callable, List, Tuple


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


def xy_poly_features(
        s: int,
        x_dim: int) -> np.ndarray:
    y = s // x_dim
    x = s % x_dim

    return np.array([
        y,
        x,
        x*y,
        y*y,
        x*x,
        1.
    ])


def all_features(
        s: int,
        x_dim: int,
        doors: List[Tuple[int, int]],
        goal: Tuple[int, int]):
    y = (s // x_dim)
    x = (s % x_dim)
    feats = [1., y, x, goal[0] - y, goal[1] - x_dim]
    for door in doors:
        feats.append(door[0] - y)
        feats.append(door[1] - x)
    return np.array(feats)
    