# Greg Attra
# 03/25/22

'''
Aggregator methods
'''

from typing import Callable


def get_aggregator(N: int, aggregate: Callable) -> Callable:
    return lambda s: aggregate(s, N)

def segment(s: int, N_segments: int) -> int:
    return s // N_segments

def zone(s: int, N_zones: int) -> int:
    pass
