# Greg Attra
# 03/25/22

'''
Feature extractors
'''

import numpy as np

from typing import Callable
from aggregators import get_aggregator


def get_feature_extractor(
        N: int,
        extract: Callable,
        aggregate: Callable) -> Callable:
    aggregator = get_aggregator(N, aggregate)
    return lambda s: extract(aggregator(s), N)


def one_hot_encoder(
        s: int,
        N: int) -> np.ndarray:
    enc = np.zeros(N)
    enc[s] = 1
    return enc
