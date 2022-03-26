# Greg Attra
# 03/25/22

'''
Aggregator methods
'''

from typing import Callable


def get_aggregator(N: int, aggregate: Callable) -> Callable:
    return lambda s: aggregate(s, N)

def segment(s: int, seg_size: int) -> int:
    return s // seg_size

def zone(s: int, zone_size: int) -> int:
    pass
