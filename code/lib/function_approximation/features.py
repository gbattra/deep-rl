# Greg Attra
# 03/25/22

'''
Feature extractors
'''

import numpy as np

from typing import Callable
from lib.function_approximation.aggregators import get_aggregator


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
